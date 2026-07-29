# OmniTransfer Training Appendix: 2026-07-29

## Scope

This appendix freezes the development experiments used to select the
OmniTransfer training defaults. These are controlled development results, not
formal test claims. MobileViews labels in this slice are self-supervised, and
the ASE rows retain their original gold status.

The method and data path remain fixed across runs:

- one `omnitransfer.ui_correspondence_pair.v1` schema;
- one 64-dimensional, two-layer, four-head OmniTransfer matcher;
- one shuffled stream of cross-page and same-page augmented pairs;
- one symmetric actionable-node correspondence objective;
- no resource-id input, absolute-position input, selector fusion, or rule
  fallback.

## Immutable Inputs

```text
code commit used for the ablation: 7637591
server release:
  /home/wuzewen/omnitransfer_eval_20260620/releases/
  omnitransfer_contextual_mutual_20260729_v1

train_mixed_256.jsonl:
  SHA-256 3e6d1a92ec6738ff51c34abdecd651793f571f971dbe4941203632e82a82ca1a
  requested rows 256 = 128 gold + 128 self_supervised
  accepted pairs 249 = 121 ASE gold + 128 MobileViews self_supervised

dev_mixed_128.jsonl:
  SHA-256 0ca757bd118e050c0b486ad227989ed0ef3d3fa18d8c6efa01fed81f97e4d327
  requested rows 128 = 64 gold + 64 self_supervised
  accepted pairs 123 = 59 ASE gold + 64 MobileViews self_supervised
  evaluated bidirectional positive rows 1,098
```

Each three-epoch run performs 661 optimizer updates per epoch: 249 cross-page
pairs and 412 lazily generated augmented-view pairs. Seed is 17, learning rate
is `3e-4`, and the optimizer is AdamW.

## Environment

```text
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.7.1+cu126
CUDA runtime reported by PyTorch: 12.6
default pair-head parameters: 579,926
mutual-projection ablation parameters: 564,892
```

The server also hosted unrelated VLLM processes during latency measurement.
The reported latency is therefore a conservative development measurement, not
an isolated hardware benchmark.

## Three-Epoch Ablation

`mask` is the probability that a supervised actionable node loses its own
text, description, and crop while retaining class, action state, and all
relations. `visual` is per-node crop dropout in same-page augmented views.

| Assignment | mask | visual | Top-1 | R@3 | R@5 | model p95 | e2e p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair MLP | 0.00 | 0.00 | 64.39 | 82.51 | 90.53 | 5.61 ms | 31.23 ms |
| pair MLP | 0.25 | 0.00 | 68.12 | 83.42 | 90.44 | 5.64 ms | 29.24 ms |
| pair MLP | 0.00 | 0.50 | 69.67 | 85.25 | 90.98 | 8.36 ms | 41.06 ms |
| pair MLP | 0.25 | 0.50 | 67.85 | 84.43 | 91.71 | 4.59 ms | 22.61 ms |
| mutual projection | 0.25 | 0.50 | 64.48 | 83.42 | 91.89 | 3.72 ms | 21.57 ms |
| pair MLP | 0.10 | 0.25 | 68.12 | 85.97 | 92.44 | 5.11 ms | 24.95 ms |
| pair MLP | 0.15 | 0.25 | 67.76 | 83.42 | 91.07 | 6.15 ms | 31.03 ms |
| pair MLP | 0.25 | 0.25 | 66.58 | 84.61 | 91.53 | 4.69 ms | 23.60 ms |
| pair MLP | **0.05** | **0.50** | **69.67** | **85.79** | **91.17** | **4.01 ms** | **22.28 ms** |
| pair MLP | 0.10 | 0.50 | 68.49 | 84.43 | 90.53 | 4.00 ms | 22.07 ms |

The selected default is pair MLP with `mask=0.05` and `visual=0.50`. Relative
to the unaugmented pair-head control, it improves Top-1 by 5.28 percentage
points while preserving the explicit anonymous-anchor training signal. The
mutual-projection head is smaller and faster but is retained only as an
ablation because its Top-1 is lower.

## One-Epoch Diagnostics

One-epoch mixed runs show that the stronger augmentations need more than one
pass: pair-head `mask=0.25, visual=0.50` moves from 64.03% at one epoch to
67.85% at three epochs. The plain pair head moves only from 64.21% to 64.39%.

An earlier 256-row pilot accidentally selected only the leading ASE block. Its
76.28% pair-head and 69.78% mutual-head scores are retained in the artifact
directory but are excluded from default selection because the training input
was not mixed.

## Failure Ledger

The following runs are intentionally preserved as invalid:

- `baseline` and `contextual_mutual`: CPU-only PyTorch was invoked with a CUDA
  device and produced no checkpoint.
- `mixed_mutual_plain`: CUDA out of memory during dev evaluation on a GPU
  already occupied by an unrelated VLLM worker.
- `mixed_pair_m10_v50_e3`: the first attempt hit the same external GPU-memory
  condition; `mixed_pair_m10_v50_e3_retry` is the completed rerun.

Invalid runs must never enter an accuracy table.

## Artifact Contract

The complete server directory is:

```text
/home/wuzewen/omnitransfer_eval_20260620/runs/
omnitransfer_ablation_20260729_v1
```

Every completed run contains:

```text
train.log       exact stdout/stderr and progress events
metrics.jsonl   run start, interval metrics, epoch metrics, evaluation, checkpoint
report.json     configuration, data adapters, full history, metrics, latency
checkpoint.pt   model state and training metadata
pid             original server process id
```

The final trainer additionally records the complete command, code revision,
input and validation SHA-256 values, runtime versions, all model and
augmentation parameters, checkpoint SHA-256, and parameter count. A root
`experiment_manifest.json` indexes both completed and invalid attempts without
deleting their evidence.
