# Local-Affinity Cross-Attention Matcher: Evidence Report

Date: 2026-07-28
Project: OmniTransfer
Status: training and automatic diagnostics complete; formal MobileViews human
gold pending

## 1. Bottom Line

The architectural idea is useful, lightweight, and scalable:

> Learn a dense source-target node affinity matrix, then propagate matched
> neighbor evidence through learned within-page relations.

The implementation is one matcher with one forward path. It does not fuse a
selector, rule score, grounding model, RL policy, or coordinate fallback.

The new checkpoint fits its enlarged MobileViews training set and transfers to
unseen MobileViews Apps under automatic stable-ID labels. It is not yet a
formally validated production matcher:

- held-out MobileViews labels still require human review;
- absolute match probability is poorly calibrated even when Top-1 ranking is
  correct;
- the unchanged model performs poorly on the public ASE iOS-to-Android widget
  benchmark, whose labels are not action-target nodes;
- all latency measurements are from an RTX 4090, not a phone.

## 2. Method

### 2.1 Inputs

For every selected node, the model uses:

```text
node token =
    text/content-description tokens
  + class and action-state features
  + visual crop features
```

The matcher does not receive:

- raw resource ID;
- absolute x/y or bbox center;
- page index or fixed row number;
- `origin_id`, `view_str`, `state_str`, or `structure_str`.

Stable IDs are permitted only in the offline data builder to construct
self-supervised labels.

### 2.2 Within-Page Relations

All selected nodes on each page participate in relation-biased self-attention.
The relation tensor describes within-page hierarchy and relative layout, such
as parent/child/sibling/ancestor, row/column relation, overlap, relative
displacement, relative scale, IoU, and tree distance.

For page relation vector \(r_{ik}\):

```text
P_ik = softmax(q_i k_k / sqrt(d) + MLP_relation(r_ik))
```

The diagonal is removed from the final relation-support matrix and each
remaining row is renormalized. A node therefore receives support from other
nodes, not from a duplicate of its own score.

### 2.3 Cross-Page Matching and Explicit Local Propagation

Bidirectional cross-attention exchanges source and target context. A shared
pair head first predicts a dense base affinity matrix:

```text
A0_ij = MLP([h_i, h_j, |h_i-h_j|, h_i*h_j])
```

The final score is:

```text
S = A0 + P_source A0 P_target^T
```

This is the essential difference from the previous checkpoint. If a source
child labeled `Sound` strongly matches a target child labeled `Sound`, that
specific child-to-child affinity directly supports the two actionable
containers that attend to those children. Evidence remains a correspondence,
instead of being pooled into an indistinct parent embedding.

Only actionable source and target nodes receive correspondence supervision and
can be returned for execution. Non-actionable text/icon nodes remain context.

### 2.4 Size and Complexity

| Item | Value |
|---|---:|
| Hidden dimension | 64 |
| Token dimension | 48 |
| Relation hidden dimension | 24 |
| Attention heads | 4 |
| Layers | 2 |
| Trainable parameters | 579,926 |

For \(N_s\) source and \(N_t\) target nodes, attention and affinity complexity
is \(O(N_s^2 + N_t^2 + N_sN_t)\). The configured context targets are 48 source
and 64 target nodes, while retaining every actionable candidate. The explicit
propagation adds no trainable parameters.

## 3. Training Objective

Both data sources are the same `TrainingPair` type:

1. two augmented views of one page;
2. two real MobileViews pages with multiple offline correspondences.

They enter one shuffled stream, one optimizer, and one symmetric partial
assignment loss:

```text
L = 0.5 * (CE(S_source_to_target, Y_source_to_target)
         + CE(S_target_to_source, Y_target_to_source))
    + 0.05 * L_bidirectional_consistency
```

Every same-screen actionable target participates as a negative. One-to-many
rows, target collisions, non-actionable correspondences, and unmatched rows
without an explicit NULL label are not treated as positives.

## 4. MobileViews Data

### 4.1 Source and Split

The versioned source is `mllmTeam/MobileViews`,
`Apps_CompleteTraces` batches 001-010, revision
`c734715eb90a8f3ac0c80973f93ee63950e5a8c8`.

The complete capacity scan found 1,063 non-test Apps that can form strict page
pairs. The old 320-App budget had no preserved reproducible selection rule, so
the new split uses all usable Apps and caps each train/dev App at 50 page pairs
to prevent a few repetitive traces from dominating.

| Split | Apps | Raw page pairs | Raw correspondences | Status |
|---|---:|---:|---:|---|
| Train | 957 | 25,585 | 257,158 | self-supervised |
| Dev candidate | 106 | 2,640 | 35,949 | unreviewed |
| Test candidate | 14 | 2,266 | 19,813 | gold review required |

Train/dev/test App overlap is zero. GUIOdyssey is not included.

### 4.2 Training Adapter

| Item | Count |
|---|---:|
| Input page pairs | 25,585 |
| Accepted cross-page pairs | 24,226 |
| Unique page graphs | 14,342 |
| Accepted actionable correspondences | 196,337 |
| One-to-many rows filtered | 50,825 |
| Target collisions filtered | 2,587 |
| Non-actionable correspondences filtered | 6,466 |
| Pairs with too few remaining labels | 1,359 |

Of 14,342 page graphs, 13,921 produced a valid same-page augmented pair. The
single training epoch therefore performed:

```text
24,226 cross-page + 13,921 augmented-page = 38,147 updates
```

## 5. RTX 4090 Training Run

| Item | Value |
|---|---|
| Host | `9207` / `baldursgate` |
| Device | physical GPU 3, RTX 4090 |
| PyTorch | 2.9.0+cu128 |
| Seed | 17 |
| Epochs | 1 |
| Learning rate | 3e-4 |
| Release | `local-affinity-propagation-20260728-v4` |
| Run | `cuda3_seed17_cap50_affinity_v2` |

Training history:

| Metric | Value |
|---|---:|
| Mean loss | 0.244425 |
| Updates | 38,147 |
| Cross-page updates | 24,226 |
| Augmented-page updates | 13,921 |
| Positive bidirectional labels seen | 691,350 |

The last complete 1,000-update windows were in the 0.14-0.17 loss range.

Checkpoint:

```text
schema: omnitransfer_relation_matcher_v3
SHA-256: 6e55edbdf87972372488ef4aa4088a7cb8c140bddae79809fa727b49625e9ef3
```

## 6. Results

### 6.1 Training-Set Fit Sanity Check

This table verifies optimization and the data path; it is not generalization.

| Metric | Result |
|---|---:|
| Page pairs | 24,226 |
| Positive labels | 392,674 |
| Top-1 | 91.88% |
| Recall@3 | 94.92% |
| Recall@5 | 95.89% |
| Model p95 | 5.91 ms |
| End-to-end p95 | 23.74 ms |
| Model samples under 50 ms | 100% |
| End-to-end samples under 50 ms | 99.44% |

The maximum end-to-end outlier is 763.74 ms, while maximum model-only latency
is 18.36 ms. XML/graph/image preparation, not matcher inference, produces the
long tail.

### 6.2 Held-Out 106-App MobileViews Diagnostic

The adapter accepted 2,532 page pairs and 22,735 forward correspondences,
evaluated bidirectionally as 45,470 labels.

| Metric | Result |
|---|---:|
| Top-1 | 91.53% |
| Recall@3 | 96.26% |
| Recall@5 | 97.39% |
| Model p95, 100-pair sample | 5.70 ms |
| End-to-end p95, 100-pair sample | 22.80 ms |

These Apps do not overlap training Apps. However, their labels are unreviewed
stable-ID proposals, so the numbers are automatic-label agreement, not formal
accuracy.

### 6.3 Frozen 14-App Test Diagnostic

The strict test pool contains 10,749 actionable-to-actionable tasks from 2,266
page pairs.

| Automatic diagnostic | Result |
|---|---:|
| Matcher Top-1 proposal agreement | 97.82% |
| Matcher `p >= 0.5` coverage | 21.72% |
| Matcher selective agreement | 100.00% |
| Selector coverage | 79.24% |
| Selector selective agreement | 99.15% |
| Selector overall proposal agreement | 78.57% |
| Matcher/selector disagreement | 23.13% |

The ranking and absolute pair probability are not the same quantity. The
matcher usually ranks the stable-ID proposal first, but its `p >= 0.5` gate
rejects most tasks. Thresholds must be calibrated on human-reviewed dev gold;
these automatic proposals cannot choose a production threshold.

### 6.4 Human-Review Queue

The corrected review queue contains exactly 1,000 tasks. Every displayed source
and target is actionable and has a click bbox. It spans 13 of the 14 frozen
test Apps; the remaining App did not contribute a task after difficulty and
diversity constraints.

| Difficulty tag | Tasks |
|---|---:|
| Low match probability | 979 |
| Matcher/selector disagreement | 860 |
| Matcher Top-1 proposal error | 194 |
| Selector abstains | 167 |
| Low matcher margin | 101 |
| Textless control | 593 |
| Small target | 514 |
| Trace state transition | 583 |
| Large position shift | 38 |

These tags are prioritization signals. The proposed correspondence is still
`gold_review_required`.

### 6.5 ASE 2023 External Gold Stress Test

The unchanged MobileViews-only checkpoint was evaluated on all 1,592 public ASE
test queries:

| Metric | Result |
|---|---:|
| Top-1 | 7.60% |
| Recall@3 | 15.95% |
| Recall@5 | 20.98% |
| MRR | 16.69% |
| Inference p95 | 108.54 ms |

This is worse than the historical identity selector Top-1 of 37.19%. It proves
that the current checkpoint is not a universal iOS-to-Android matcher.

The benchmark contract is also different from the current action-target
contract: all 6,730 ASE source nodes have action flags false, and 880 of 1,592
test gold targets are non-actionable under the same predicate. ASE labels
visual widgets/bboxes rather than guaranteed event-receiving nodes. Directly
adding ASE train labels would violate actionable-to-actionable supervision, so
they were not used to retrain this checkpoint.

## 7. What the Evidence Supports

Supported:

1. explicit local affinity propagation learns and improves the intended
   child/icon/text-to-container evidence path;
2. the architecture remains simple and has only 579,926 parameters;
3. training scales to 957 Apps and 38,147 mixed updates;
4. unseen-App MobileViews automatic-label agreement is high;
5. model-only RTX 4090 p95 is below 6 ms on MobileViews.

Not yet supported:

1. formal MobileViews Top-1 before human review;
2. calibrated NULL/abstention behavior;
3. reliable iOS-to-Android generalization;
4. a strict every-input 50 ms end-to-end guarantee;
5. execution on an Android phone.

## 8. Next Required Work

The next work should not add architecture branches:

1. review the corrected 1,000-task MobileViews queue;
2. use reviewed dev—not test—to calibrate probability and margin gates;
3. report formal matcher and selector results only on accepted human gold;
4. construct a separate cross-platform action-target dataset by binding each
   visual widget to the actual event-receiving node before using ASE-like data;
5. export the same matcher to a phone runtime, cache page encodings, and measure
   graph preparation, visual encoding, matcher, and postprocess separately.

## 9. Artifacts

Local:

- `docs/learned_cross_attention_matcher.md`
- `src/omnitransfer/learned_matcher.py`
- `scripts/train_mapping_page_pairs.py`
- `runtime/datasets/mobileviews_hard_acceptance_1000_20260728_affinity_v1`

Server:

- release:
  `/home/wuzewen/Projects/Omni/OmniTransfer/releases/local-affinity-propagation-20260728-v6`
- run:
  `/home/wuzewen/Projects/Omni/OmniTransfer/runs/local-affinity-propagation-20260728-v2`
- checkpoint report SHA-256:
  `ff5d9027e01e06ad6edbcd3c7ea6db704d673497578cac7a9b810a7b9fa86832`
- held-out dev report SHA-256:
  `1efbb752ed3997e55cbfd0b7ee74db01228b25f85e6e352f48799b4c1de77c9b`
- test diagnostic report SHA-256:
  `b205183d3dfe21d5c9b6b0e879a522829a29560905b3e2a46115cca1e5fdf470`
- corrected review manifest SHA-256:
  `470e4a1d5e1097cea0201ac28404dbaf5cf23d222cbf50360e39e2c60026cbda`
- ASE report SHA-256:
  `664f09e6ec83449f7e812603bc68fcd7602c8b1cae4ab267875330907a1ff34c`
