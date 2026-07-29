# OmniTransfer: Learning Cross-Platform UI Correspondence for GUI Replay

Internal short-paper snapshot, 29 July 2026.

This document freezes the current empirical reference and the next method
hypothesis. It is not yet a submission draft: related-work citations must be
searched and verified before publication.

```text
frozen reference: seeded-pair-evidence-mutual-v2
next version:      context-forced-cross-attention-v1
next status:       design only; no result claimed
```

## Abstract

GUI replay systems often retain the correct operation but lose the element on
which that operation should execute after a platform or layout change. We
formulate this failure as source-conditioned UI correspondence: given an
interacted node on an iOS source screen and the UI graph of an Android target
screen, rank the target node that represents the same control. We construct a
leakage-audited benchmark from 6,730 original iOS-to-Android mappings and split
it by App into 4,124 training, 1,014 development, and 1,592 test queries. Our
frozen accuracy-latency reference is a 670K-parameter pair-evidence mutual
matcher that reaches 78.83% Top-1 and 94.54% Recall@5, with 48.82 ms p95
new-image latency on an RTX 4090. We use this result as an empirical baseline,
not as the final architectural claim. The next OmniTransfer model retains one
relation-aware cross-attention matcher and strengthens its supervision with
context-forced augmentation and same-screen semantic hard negatives, avoiding
selector fusion, rule scores, and auxiliary execution paths.

## 1. Problem

A recorded GUI trace already specifies the operation:

```text
click / type / scroll
```

Replay therefore does not need to infer what to do again. It needs to relocate
the operated control:

```text
source screen G_s
+ interacted source node q_s
+ target screen G_t
-> corresponding target node q_t
```

This problem differs from instruction grounding. Instruction grounding maps a
sentence to a point on one screen. OmniTransfer maps a concrete source control,
including its neighboring UI evidence, to the corresponding control in another
layout or platform.

The runtime must fail closed. If the matcher cannot distinguish the best
candidate from alternatives, it returns transfer failure and control goes back
to the VLM. Source coordinates are never replayed directly on the target.

## 2. Benchmark

We use the public ASE 2023 vision-based widget-mapping data as the primary
cross-platform benchmark. Each original annotation provides an iOS source
bounding box and the corresponding Android target bounding box. The boxes bind
to real nodes in the source and target XML hierarchies; the model ranks the
visible target XML candidates.

| Split | Queries | Apps |
|---|---:|---:|
| Train | 4,124 | 32 |
| Dev | 1,014 | 8 |
| Test | 1,592 | 10 |
| Total | 6,730 | 50 |

The split unit is App. Train, development, and test Apps do not overlap. The
clean bundle contains 2,104 screens and 1,052 iOS-Android screen pairs. A target
screen contains 48.81 candidate nodes on average.

This benchmark currently contains positive correspondence annotations. It does
not contain a reviewed set of truly absent targets, so it cannot establish
learned NULL recall.

## 3. Frozen Empirical Reference

### 3.1 Pair-evidence mutual matcher

The strongest accuracy-latency reference computes one dense source-target
matrix from semantic, visual, attribute, context, geometry, and local-anchor
evidence. Row and column normalization provide the two matching directions:

```text
P(t_j | s_i)
P(s_i | t_j)
```

The final ranking score is their log-space mean:

```text
M_ij = 0.5 * (log P(t_j | s_i) + log P(s_i | t_j))
```

The model has 670,459 parameters and is trained for three epochs with seed 17.
It is intentionally frozen as the reference that a simpler relation
cross-attention model must exceed.

### 3.2 What “Pair-Confidence No-NULL” means

The No-NULL variant does not add a synthetic NULL row, NULL column, or absence
class to the assignment matrix. It ranks only real target nodes. A separate
absolute pair-confidence head estimates whether the highest-ranked real pair is
safe enough to execute.

At runtime:

```text
if confidence < threshold or top1_margin < threshold:
    return transfer_failure
else:
    execute top1 target
```

No-NULL therefore describes an assignment design, not an assertion that every
source control has a target. Rejection happens after ranking and returns to the
VLM.

On the frozen test set, this variant has 77.45% rank Recall@1. With the default
confidence threshold, it executes 82.91% of queries and is correct on 85.53% of
the executed subset. Counting abstentions as unresolved queries gives 70.92%
correct execution over the full test set. This variant is safer than always
executing, but it is not the most accurate ranker.

## 4. Results

All results below use the same App-disjoint iOS-to-Android split.

| Method | Top-1 | Recall@3 | Recall@5 | Parameters | New-image p95 |
|---|---:|---:|---:|---:|---:|
| Relation cross-attention | 70.73% | 87.12% | 91.39% | 821,015 | 50.21 ms |
| Dot-product mutual | 66.46% | 86.75% | 92.21% | 611,902 | 46.92 ms |
| Pair-evidence mutual, raw best | **79.02%** | 90.45% | 93.53% | 670,459 | 51.83 ms |
| Pair-evidence mutual, frozen reference | 78.83% | **91.08%** | **94.54%** | 670,459 | **48.82 ms** |

The 79.02% run has the highest raw Top-1, but its p95 exceeds 50 ms. We
therefore freeze the 78.83% checkpoint as the accuracy-latency reference. Its
model-only p95 is 8.35 ms; input construction accounts for 34.54 ms p95.

The 48.82 ms number is a percentile, not an every-image guarantee. Five of 100
new-image measurements exceeded 50 ms, and the maximum was 70.22 ms.

## 5. A Simpler Cross-Attention Method

We name the next experiment `context-forced-cross-attention-v1`. It starts from
the frozen 70.73% relation cross-attention baseline and must be evaluated
against the 78.83% accuracy-latency reference. Creating this version does not
replace the runtime checkpoint and does not modify the frozen baseline.

The empirical reference is accurate but exposes several manually separated
evidence matrices. Our intended paper method is smaller conceptually:

```text
node encoder
-> within-screen relation self-attention
-> bidirectional source-target cross-attention
-> one dense affinity head
-> symmetric correspondence loss
```

Each node token contains:

```text
text / content description
class and action state
visual crop
```

Raw resource identifiers, absolute coordinates, absolute box centers, and page
order are excluded. Geometry appears only as a relative relation between nodes
on the same screen. Every selected node participates in attention, while only
actionable nodes are execution candidates.

### 5.1 Why the existing cross-attention baseline is weaker

The 70.73% baseline can observe local context, but the supervision does not
force it to use that context. Easy pairs allow the model to rely on the
interacted node itself or on coarse layout correlations. Repeated list rows
then become difficult: several actionable containers share the same class and
structure, while the distinguishing title is stored in a non-actionable child.

The failure is primarily a training problem. Adding more feature branches would
make the method less identifiable without ensuring that cross-attention learns
the desired evidence.

### 5.2 Context-forced training

We propose to strengthen the same model with three data and loss changes:

1. **All actionable hard negatives.** For every source node, the denominator of
   the assignment loss contains every actionable target node on the page.
   Isomorphic sibling rows are therefore explicit negatives rather than
   accidentally omitted candidates.
2. **Source-anchor masking.** During training, randomly mask the interacted
   node’s own text, identifier-like tokens, visual crop, or coarse location
   cues while preserving surrounding child text and icons. The correct target
   remains unchanged, forcing attention to use local relations.
3. **Cross-platform final supervision.** Same-page augmented views may pretrain
   the encoder, but final optimization and model selection use only the clean
   iOS-to-Android train and development splits. MobileViews scores cannot select
   the final checkpoint.

The objective remains one symmetric assignment loss:

```text
L = CE(source -> target) + CE(target -> source)
```

No selector score, rule vote, grounding model, RL policy, or auxiliary runtime
branch enters the matcher. Confidence and margin calibration use the
development split after training; they decide rejection but do not change the
ranking architecture.

This section is a testable hypothesis. No result in Table 2 is attributed to
context-forced cross-attention until that model is trained and evaluated on the
frozen split.

## 6. Limitations

The current benchmark covers iOS-to-Android mappings but only 50 Apps. It does
not yet measure foldables, WebViews, tablets, or reviewed absent-target cases.
The benchmark also reflects the UI trees and screenshots available in the
original collection.

The frozen best model uses explicit feature families and is therefore an
empirical reference rather than the simplest final method. Conversely, the
proposed cross-attention enhancement is architecturally cleaner but has not yet
demonstrated an improvement over 70.73%.

The latency evaluation uses an RTX 4090. No result in this paper establishes
phone latency, energy use, or memory consumption. The full preprocessing path
also has a long tail above 50 ms.

## 7. Reproducibility Marker

The frozen checkpoint identity is:

```text
release:
/home/wuzewen/Projects/Omni/OmniTransfer/releases/
pair_evidence_mutual_v2_e3e9e2f0_20260722

checkpoint:
results/seeded_visual_seed17.pt

checkpoint SHA-256:
e2c879b5046c86e7493f3eb18c46a24bbe180ae5c483e779a53097384fdc4ad5

test report SHA-256:
8e318eac1f06e28cc390baf1cfcf1673d839d2d757bda33dc27ab6e7836a37a2
```

The machine-readable record is
`artifacts/manifests/best_ios_android_matcher_20260729.json`.
