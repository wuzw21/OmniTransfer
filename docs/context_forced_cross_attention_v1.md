# Context-Forced Cross-Attention V1

Status: open, design-only, not trained

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

## Training Change

The architecture stays fixed. V1 changes only supervision:

1. Every actionable target node on the page is a ranking candidate.
2. Repeated rows with similar class and structure remain explicit hard
   negatives.
3. The source action node is sometimes stripped of its own easy semantic or
   visual cue while its child and neighboring context remains visible.
4. Same-page augmentation may initialize the encoder, but final training and
   checkpoint selection use the clean iOS-to-Android split.
5. The loss remains the sum of source-to-target and target-to-source
   cross-entropy.

## Frozen Data

```text
dataset: ASE 2023 clean relative-XML iOS -> Android
train:   4,124 queries / 32 Apps
dev:     1,014 queries / 8 Apps
test:    1,592 queries / 10 Apps
overlap: zero Apps
```

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
