# MobileViews Shortcut-Free 64D Matcher Run

Date: 2026-07-28

Commit: `352c6c0`

Status: training complete; held-out reviewed-gold evaluation pending

## Method Boundary

The matcher uses one 64D node state:

```text
node state = text/content-desc + class/action state + visual crop
```

Raw resource id, absolute coordinates, absolute box size, page order, and
offline stable ids are not model inputs. The bbox is used only to crop the
node image and to form within-page relative relations. Cross-page geometric
features are a zero tensor.

Every retained node participates in relation-aware self-attention and
bidirectional cross-attention. Training labels and ranking candidates are
strictly actionable-node to actionable-node. Non-actionable text and icon
nodes are context only; descendant labels are not lifted to actionable
ancestors.

The model has 579,926 trainable parameters, two layers, four heads, and one
symmetric correspondence objective with bidirectional consistency. There is no
selector, rule-score fusion, grounding branch, RL branch, or coordinate replay.

## Data Audit

Input:

```text
/home/wuzewen/omnitransfer_eval_20260620/runs/
mobileviews_only_dataset_20260724_v2/train_dataset/train.jsonl
SHA-256 e19e3e5cce0e84e2eefa7837b273be5f4a9e07ad568cf96340137ac37e79a78b
```

| Item | Count |
|---|---:|
| Input page pairs | 15,244 |
| Accepted page pairs | 15,118 |
| Unique page graphs | 3,302 |
| Accepted actionable correspondences | 164,651 |
| Explicit one-to-one correspondences before action filtering | 168,816 |
| Non-actionable correspondences filtered | 4,093 |
| Ambiguous one-to-many rows filtered | 22,301 |
| Target-collision rows filtered | 282 |
| Pairs skipped after filtering | 126 |

All accepted records are MobileViews training records. GUIOdyssey is absent.

## Frozen Run

```text
release:
/home/wuzewen/omnitransfer_eval_20260620/releases/
omnitransfer_mobileviews_shortcut_free_20260728_v1

run:
/home/wuzewen/omnitransfer_eval_20260620/runs/
mobileviews_shortcut_free_64d_cuda3_seed17_20260728_v3
```

Configuration: physical RTX 4090 GPU 3, seed 17, one epoch, learning rate
`3e-4`, training from random initialization. The optimizer completed 15,118
cross-page updates and 3,173 valid same-page augmented updates. Another 129
page graphs produced fewer than two retained actionable identities after
augmentation and were skipped. The epoch mean loss was 0.352669 over 18,291
updates and 395,802 observed bidirectional positive labels.

The attempted GPU-0 run `...cuda0_seed17_20260728_v2` failed because an
unrelated process occupied 20.45 GiB of GPU memory. It produced no checkpoint
and is retained as an infrastructure-failure record, not a model result.

## Training-Set Fit and Latency

These metrics are computed on the same 15,118 pairs used for training. They
measure optimization and implementation behavior, not generalization.

| Metric | Result |
|---|---:|
| Top-1 | 80.81% |
| Recall@3 | 84.32% |
| Recall@5 | 86.74% |
| Model p50 | 3.96 ms |
| Model p95 | 5.61 ms |
| Model max | 10.67 ms |
| End-to-end p50 | 17.24 ms |
| End-to-end p95 | 26.08 ms |
| End-to-end max | 339.29 ms |

All 30,236 warmed model measurements are below 50 ms. With screenshot/input
construction included, 30,202 of 30,236 measurements are below 50 ms and 34
are above it. Therefore model compute passes the 50 ms target, while the strict
every-image end-to-end target does not yet pass.

Artifacts:

```text
checkpoint.pt
SHA-256 379ad4178b8016633e3087d316f25cdf74d3b441ba26dbe854b9b13cea4bcefd

report.json
SHA-256 0a812213a341d557d8ca39106d3a1797fe90e12497aaa25b2f46deb221771623
```

## Evidence Boundary

The run proves that the shortcut-free representation is trainable at the
required model-compute latency. It does not prove that removing resource id and
absolute position improves cross-app or cross-device generalization. That claim
requires the frozen MobileViews App-disjoint reviewed-gold dev/test split and
the reviewed hard actionable-container slice. The current unreviewed queue is
not formal gold and must not be used for a paper headline metric.
