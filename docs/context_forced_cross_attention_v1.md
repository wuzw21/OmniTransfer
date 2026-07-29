# Context-Forced Cross-Attention V1

Status: implementation and unified-data pipeline complete; formal training not started

Opened: 2026-07-29

## Purpose

This version starts a new experiment without changing the current runtime
checkpoint. Its goal is to retain one simple relation cross-attention matcher
while making the training data force the model to use neighboring text, icons,
and structure.

The frozen reference remains:

```text
seeded-pair-evidence-mutual-v2
Top-1: 78.83%
Recall@5: 94.54%
new-image p95: 48.82 ms
```

The clean relation cross-attention starting point is:

```text
Top-1: 70.73%
Recall@5: 91.39%
```

## Architecture Boundary

The model contains one path:

```text
node encoder
-> within-screen relation self-attention
-> bidirectional source-target cross-attention
-> dense affinity matrix
-> symmetric assignment loss
```

Node inputs are limited to text/content description, class/action state, and
the visual crop. Absolute coordinates, absolute box centers, page order, and
raw resource identifiers are excluded. Relative geometry is allowed only
between nodes on the same screen.

There is no selector fusion, handcrafted score, grounding branch, RL branch,
learned NULL row, or coordinate replay.

V1 keeps two attention layers. Layer count is not increased in the main run:
two layers already allow one round of within-screen relation aggregation and
one round of cross-screen refinement while preserving the 50 ms target. A
1/2/3-layer comparison is an ablation, not a silent architecture change.

## One Dataset Contract

Every dataset and every split uses:

```text
omnitransfer.mapping_page_pair.v1
pair_id, split, label_status
source {page_id, platform, screenshot_path, graph}
target {page_id, platform, screenshot_path, graph}
matches [{source_node_id, target_node_ids[], label}]
partition_keys, provenance, slices
```

One row contains both full UI graphs and any number of set-valued node
correspondences. Train/dev/test differ only in `split` and label provenance.
ASE queries are grouped by screen pair; MobileViews already emits this schema.
GUIOdyssey is not included.

The dataset writer validates and writes records in one pass. Memory therefore
scales with split identity sets rather than the serialized UI graphs; it does
not materialize the multi-gigabyte MobileViews corpus twice.

The only learned matcher entrypoint is:

```text
scripts/train_mapping_page_pairs.py
```

The removed query-only, graph-only pretraining, and GUIOdyssey mixed trainers
cannot create a second optimization path. Same-page augmentation and
cross-page correspondence learning are shuffled inside one optimizer loop.

## Training Change

The architecture stays fixed. V1 changes only supervision:

1. Every actionable target node on the page is a ranking candidate.
2. Repeated rows with similar class and structure remain explicit hard
   negatives.
3. The source action node is sometimes stripped of its own easy semantic or
   visual cue while its child and neighboring context remains visible.
4. Same-page augmentation may initialize the encoder, but final training and
   checkpoint selection use the clean iOS-to-Android split.
5. One shared projection produces a dense affinity matrix after each attention
   layer. Row- and column-normalized log assignments receive equal layer-wise,
   bidirectional set-valued supervision.

## Frozen Data

```text
record schema: omnitransfer.mapping_page_pair.v1
ASE records:   1,052 page pairs / 6,698 match rows
train:         639 page pairs / 4,092 match rows / 32 Apps
dev:           173 page pairs / 1,014 match rows / 8 Apps
test:          240 page pairs / 1,592 match rows / 10 Apps
overlap:       zero pair, page, and partition-key overlap
GUIOdyssey:    false
```

The actionable training adapter currently accepts 590 ASE train page pairs and
2,184 actionable edges. It retains 522 set-valued rows. Non-actionable target
labels remain context and are never promoted into synthetic action nodes.

The test set is evaluated once after model selection. MobileViews automatic
labels cannot select this checkpoint and cannot be reported as the formal test
result.

## Acceptance Gates

The version is successful only if all of the following hold:

```text
clean test Top-1 > 78.83%
clean test Recall@5 >= 94.54%
fixed RTX 4090 new-image p95 < 50 ms
fixed RTX 4090 model compute max < 50 ms
no selector, rule, or coordinate fallback enters the matcher
```

Phone latency is reported separately and is not inferred from RTX 4090
measurements. Low confidence or low margin returns transfer failure to the VLM.

## Appendix Log Contract

Every run writes `*.metrics.jsonl` with schema
`omnitransfer.matcher_training_metrics.v1`:

```text
run_start:
  input paths and SHA-256
  matcher and augmentation configuration
  seed, epochs, learning rate
  Python, PyTorch, CUDA, and GPU

training_progress / epoch_end:
  total loss
  final and intermediate assignment loss
  cycle loss
  supervised layer count
  context-masked node count/rate
  self-supervised and cross-page pair counts
  positive edge count
  Dev Top-1 and Recall@K

evaluation / checkpoint:
  split and record counts
  Top-1 and Recall@K
  warm model/input/end-to-end latency
  parameter count and checkpoint SHA-256
```

## Required Ablations

Only three comparisons are required:

```text
base cross-attention
+ all actionable hard negatives
+ context-forced masking
```

This keeps the causal question identifiable: whether forcing local contextual
evidence improves cross-platform node matching without adding another model
branch.
