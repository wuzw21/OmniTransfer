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

## Full-Scale Training Record

The selected configuration completed on the complete unified training split at
`2026-07-29T14:35:04Z`. The reported score is a development result: most
MobileViews correspondences are self-supervised, so it is not a held-out
human-gold test claim.

```text
code commit: e1bb5ec
server release:
  /home/wuzewen/omnitransfer_eval_20260620/releases/
  omnitransfer_contextual_pair_20260729_v2
run directory:
  /home/wuzewen/omnitransfer_eval_20260620/runs/
  omnitransfer_full_contextual_pair_cuda3_seed17_20260729_v1
launch PID: 606546
device: cuda:3 (NVIDIA GeForce RTX 4090)
seed: 17
epochs: 1
learning rate: 3e-4
progress interval: 250 optimizer updates
assignment head: pair_mlp
context mask probability: 0.05
visual crop dropout probability: 0.50
```

The immutable full inputs are:

```text
train.jsonl:
  /home/wuzewen/Projects/Omni/OmniTransfer/runs/
  ui-correspondence-mixed-within-app-v3-20260729/train.jsonl
  SHA-256 81db4605997ab243984362022c68e64e31c5bda70bd737371efbf06aebd00448

dev.jsonl:
  /home/wuzewen/Projects/Omni/OmniTransfer/runs/
  ui-correspondence-mixed-within-app-v3-20260729/dev.jsonl
  SHA-256 9e00134a1cffb827f52fdf82bc3fc3f8a4c3fc4713abec02f68050429e39162a
```

The training adapter accepted 26,494 cross-page pairs (25,729 MobileViews
self-supervised and 765 ASE gold), 13,713 unique graphs for same-page
augmentation, and 928,468 actionable correspondences. It filtered 23,801
non-actionable correspondences and skipped 81 rows with too few actionable
correspondences. The dev adapter accepted 2,501 pairs and 37,807 actionable
correspondences. Stable ids are label-only; raw resource ids and absolute
positions are absent from the model input.

The exact launch command is preserved in the first `run_start` event of
`metrics.jsonl`. The first three 250-update windows were:

| Updates | Running loss | Cross-page | Same-page augmented | Positive labels |
|---:|---:|---:|---:|---:|
| 250 | 1.202553 | 174 | 76 | 39,588 |
| 500 | 0.817922 | 347 | 153 | 50,960 |
| 750 | 0.853295 | 513 | 237 | 57,709 |

The completed epoch contained 40,134 optimizer updates: 26,494 cross-page
pairs and 13,640 same-page augmented pairs, with 2,248,472 positive labels.
The epoch-average loss was `0.473681`.

Evaluation covers both directions of 2,501 dev pairs: 5,002 directional
examples and 40,635 positive rows.

| Metric | Result |
|---|---:|
| Top-1 / Recall@1 | 94.91% |
| Recall@3 | 98.19% |
| Recall@5 | 99.07% |
| warm model p50 / p95 | 4.08 / 5.99 ms |
| warm end-to-end p50 / p95 | 17.42 / 27.94 ms |
| model calls under 50 ms | 100.00% |
| end-to-end calls under 50 ms | 99.44% |

The checkpoint has 579,926 parameters:

```text
checkpoint.pt:
  SHA-256 11e3aa9a952bb084270e945c70305279722ae81a4191d6f0fddff073d0e2149a
report.json:
  SHA-256 6a3d07688d5f67f36f4935736d56bd8827904bcc3eb086fb2b892cd6ad9e8867
metrics.jsonl:
  SHA-256 7b4845f0786e6161c6f00f2af446673610771073d26b6e6c03ff375d5f06a667
train.log:
  SHA-256 415d5fb643209e6b7a5644aa1c9c656eacb94b366799dccc038102a6ae2e50c5
```

The run directory is the source of truth for the complete append-only
trajectory and final artifacts.

## ASE Frozen-Test Contamination Diagnostic

The full-scale checkpoint was also evaluated against the preserved ASE 2023
test declaration of 240 screen pairs and 1,592 authored mappings. This result
is **not formal**. The mixed within-App V3 split reassigned the original ASE
records before training: 191 of the 240 frozen-test screen pairs entered the
new training split, 27 entered dev, and only 22 remained in test. Consequently,
this checkpoint must not be promoted using the historical ASE test.

The diagnostic uses the canonical actionable-correspondence adapter and scores
both source-to-target and target-to-source directions. It is therefore also not
numerically identical to the historical one-direction 1,592-query protocol.

```text
run directory:
  /home/wuzewen/omnitransfer_eval_20260620/runs/
  omnitransfer_ase_frozen_contaminated_diagnostic_20260729_v1
input test.jsonl:
  SHA-256 7a629d112ae83723259fe7c2881bb63ae4a432197e9687adbdde45862e9b6113
checkpoint.pt:
  SHA-256 11e3aa9a952bb084270e945c70305279722ae81a4191d6f0fddff073d0e2149a
report.json:
  SHA-256 1b3871347aa221c6346973a0574d3585e6e5c87480204670684636b86cca2ebe
```

The adapter accepted 226 of 240 screen pairs, 763 actionable
correspondences, and 1,524 bidirectional positive rows. Fourteen screen pairs
contained no accepted actionable correspondence; 1,100 non-actionable
candidate correspondences were filtered.

| Slice by V3 reassignment | Frozen-test screen pairs | Evaluated positive rows | Top-1 |
|---|---:|---:|---:|
| reassigned to V3 train | 191 | 1,200 | 61.75% |
| reassigned to V3 dev | 27 | 172 | 67.44% |
| reassigned to V3 test | 22 | 152 | 65.13% |
| complete contaminated diagnostic | 240 | 1,524 | 62.73% |

The complete diagnostic has Recall@3 `83.27%`, Recall@5 `91.93%`, warm model
p95 `5.87 ms`, and warm end-to-end p95 `33.43 ms`. Even the train-overlap
slice is low, so the result is not explained by unseen-pair generalization
alone. The full training stream was dominated by MobileViews (25,729 accepted
cross-page pairs versus 765 ASE pairs), while the ASE task is strict
iOS-to-Android matching. This is evidence of objective/domain imbalance, not a
valid replacement for the old formal score.

A valid comparison requires retraining from a pool that excludes every
historical ASE dev/test record before optimization, followed by evaluation on
the untouched declared split with one frozen metric contract.

## Development-Domain Split Diagnostic

The aggregate 94.91% dev result was decomposed by dataset without changing the
checkpoint, adapter, or metric. This confirms that the aggregate is dominated
by MobileViews rather than representing uniform cross-platform performance.

| Dev source | Accepted pairs | Positive rows | Top-1 | R@3 | R@5 |
|---|---:|---:|---:|---:|---:|
| MobileViews self-supervised | 2,397 | 39,933 | 95.58% | 98.52% | 99.26% |
| ASE gold | 104 | 702 | 56.84% | 79.34% | 88.32% |
| Aggregate | 2,501 | 40,635 | 94.91% | 98.19% | 99.07% |

MobileViews contributes 98.27% of the evaluated positive rows. The training
cross-page stream has the same imbalance: 25,729 MobileViews pairs versus 765
ASE pairs, or 97.11% versus 2.89%. Effective ASE supervision is smaller still:
the actionable adapter filtered 484 of 838 explicit ASE dev correspondences,
compared with 1,093 of 38,546 MobileViews correspondences.

The correspondence topology also differs. MobileViews dev retains 21,816
shared-target rows among 37,453 accepted correspondences, whereas ASE dev
retains only 2 among 354. Thus the aggregate combines a dominant
same-corpus, Android-to-Android, self-supervised many-to-many task with a small
strict iOS-to-Android gold task. Page and component overlap are zero, but the
MobileViews train/dev rows share the collection process, platform, Apps, XML
conventions, label generator, and correspondence topology; this is the
same-distribution sense used in the analysis.

```text
run directory:
  /home/wuzewen/omnitransfer_eval_20260620/runs/
  omnitransfer_domain_split_diagnostic_20260729_v1
ase_dev.report.json:
  SHA-256 2dea654785a73b4c223d5ae251f6884aa9b0410025f638a18af9704bd3c1ca7a
mobileviews_dev.report.json:
  SHA-256 024dbc873906ff153a6a644b27db1145db006fe4e589b6734f70530f3eb9ec75
```

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
