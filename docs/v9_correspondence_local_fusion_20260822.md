# V9 correspondence-conditioned local fusion — 2026-08-22

## Decision

The production candidate is one end-to-end matcher:

1. reuse the complete trained v9 multimodal backbone;
2. learn up to five useful local neighbours from the union of XML, spatial,
   and local-order relations;
3. combine local relation compatibility and the current soft correspondence as
   independent evidence;
4. update one real-node correspondence matrix three times;
5. read one normalized 1024D page/state vector from the same encoded nodes.

There is no NULL class, inference bonus, rule reranker, modality gate, hard
card classifier, or second page encoder.  Resource IDs and node IDs are label
and graph-join data only.

## First-principles review

Three independent reviews were compared: Android/iOS implementation, ordinary
user intent, and model/training/runtime design.  They agreed on the following
core concept:

> A node is identified by the user object/action it represents, using its own
> multimodal evidence, its position inside a local object, and the current
> correspondence of nearby evidence.

A card or form group must therefore be soft evidence, not a hard first stage.
Android and iOS expose different wrapper depths; a mistaken hard group would
permanently delete the correct endpoint.  Missing text, bbox, clickability, or
one relation means “unknown”, not “incompatible”.  A candidate relation exists
when structural **or** spatial **or** local-order evidence is available.

## Corrected algorithm

The v9 node encoder produces a 64D state from text/content description,
deterministic visual evidence, and XML/state/geometry using its trained
missing-aware softmax modality fusion.

Training gives every relation candidate a soft selection gradient.  Inference
uses the exact highest five neighbours, so runtime remains bounded.  For a
source-target candidate `(i, j)` and neighbour pair `(u, v)`, the model forms
an additive learned evidence value from relation compatibility and the current
soft correspondence `P[u, v]`.  It normalizes valid neighbour-pair evidence
with softmax and also records soft-OR support and coverage.  This replaces the
old multiplication:

```text
sigmoid(relation compatibility) * P[u, v]
```

which suppressed useful structure whenever the initial correspondence was
uncertain.  The learned correction is added to the v9 backbone score without
a fixed `tanh` cap; the assignment loss learns its scale.  Three successive
matrices are supervised by the same symmetric,
set-valued correspondence cross-entropy.  The objective weights are 0.1 for
the initial node matrix and 0.2/0.3/0.4 for the three updates.  Every non-gold
node in the target page is already a negative; high-scoring same-card,
same-semantic, sibling, parent/child, and wrapper/leaf errors dominate the
cross-entropy gradient without a second loss.

## Input and runtime bugs fixed

- Flat observations with repeated resource IDs now receive distinct internal
  node IDs.  Repeated RecyclerView/card nodes no longer overwrite topology.
- Target context reserves additional local evidence beyond bounded actionable
  candidates, so iOS semantic nodes without bbox can still support a bounded
  endpoint.
- Exact Top-1/Top-2 ties now abstain instead of being resolved by XML/node-ID
  order.
- `encode_page`, `match_page`, and `map_node` are separate reusable operations.
  Page screenshot/XML/local encoding and the all-node score matrix can be
  cached; repeated node maps only read a row.
- Torch page embedding calls `encode_page` once instead of matching a page to
  itself and encoding it twice.

## State embedding

Node width remains 64D.  Sixteen learned attention slots read the same locally
encoded nodes, each slot is normalized, and their concatenation is normalized
again:

```text
16 slots × 64 dimensions = 1024D
```

The state readout is compressed back into the pair scorer, so the same
correspondence loss trains it.  Until explicit state-transition invariance
supervision exists, the API exposes one honest page/state embedding; legacy
stable/active names are compatibility aliases, not unsupported claims.

## Experiment evidence

All experiments use ASE reviewed node correspondences and run in an isolated
RTX 4090 workspace.

- New local-only path, 16-page capacity check: fixed Train Top-1 100%; unary
  98.30%; local correspondence corrected four additional rows and hurt none.
  The path is active and has sufficient fitting capacity.
- New local-only path, full training: the first six Dev checkpoints rose only
  from 65.13% to 66.54%; refinement help and hurt were nearly balanced.  This
  proves that migrating only 28 v9 node parameters and relearning the old
  cross-page backbone from 839 page pairs is the cause of the persistent
  60–70% ceiling.
- Complete v9-backbone migration transfers 99 compatible parameters.  Before
  new training, it reaches Dev Top-1 73.75%, Recall@3 87.16%, Recall@5 90.29%.
  Warm RTX 4090 matcher latency is p50 10.16 ms / p95 13.13 ms; end-to-end is
  p50 27.98 ms / p95 89.03 ms / max 178.84 ms.
- Joint six-epoch training with a capped local correction selects epoch 5 at
  Dev Top-1 76.76%.  Its fixed-checkpoint Train Top-1 is 95.99%, while strict
  Test Top-1 is 80.79% (Recall@3 92.30%, Recall@5 94.79%).  Model latency on
  Test is p50 11.10 ms / p95 14.28 ms; end-to-end p50 30.42 ms / p95 80.87 ms.
- Causal Dev ablation found that disconnecting state context, neighbour
  fusion, or the entire local correction changed no Top-1 decisions.
  Disconnecting current correspondence feedback changed only three of 1,566
  decisions (76.76% to 76.56%).  Layer 2 and layer 3 had the same Top-1.
- The cause is a score-scale bug: the old `tanh` limited local correction to
  ±1 (observed RMS 0.93) while the trained v9 backbone score had RMS about 15.
  The local evidence could not overtake close wrong candidates.  The cap was
  removed; this is a correction to the same score path, not a gate or bonus.
- With the cap removed, six-epoch joint training selects epoch 3 at Dev Top-1
  77.01% and fixed-checkpoint Train Top-1 95.00%.  At that checkpoint, Dev
  rises from 73.12% after update 1 to 75.54% after update 2 and 77.01% after
  update 3.  The local path is now causally capable of changing rankings, but
  the 17.99-point Train/Dev gap shows that scale was only one bug; cross-app
  generalization is now the limiting factor.
- Robust correspondence-preserving augmentation selects epoch 5 at Dev Top-1
  77.91% with fixed-checkpoint Train Top-1 93.16%.  Its Dev update accuracies
  are 74.39%, 76.88%, and 77.91%, so every correspondence round has positive
  net value.  Freezing all migrated v9 parameters peaks at only 75.93%; joint
  low-rate adaptation is required, while measured v9 association-layer drift
  remains small (about 1.22% relative L2 change).
- A second causal audit found that the complete local correction contributes
  +0.19 points, but the old pre-aggregation of neighbours into each node
  contributes 0.00 and raw soft-correspondence feedback contributes -0.32.
  The pre-aggregation path was deleted.  Raw `P[u,v]` was replaced by its
  bidirectionally normalized relative support: uniform correspondence is zero
  evidence, and a neighbour match is positive only when it stands out in both
  its source row and target column.  This directly represents “is the current
  neighbour correspondence trustworthy?” without a gate or auxiliary loss.
- The final clean checkpoint removes page-state input from node mapping because
  its causal Top-1 effect was exactly zero.  The same encoder still emits the
  requested normalized 1024D state vector as an independent page artifact.
  Removing the inactive mapping path reduces the model from 790,563 to 761,883
  parameters and slightly improves strict Dev Top-1 to 78.22%.

Final clean Dev ablation (1,566 directed correspondence rows):

| Variant | Top-1 | Change |
|---|---:|---:|
| Complete clean model | 78.22% | — |
| Remove learned neighbour relation content | 77.46% | -0.77 |
| Remove relative correspondence support | 77.59% | -0.64 |
| Remove the whole learned local correction | 77.39% | -0.83 |
| Reconnect/remove page state in mapping | 78.22% | 0.00 |

The selected hard-row training run is 0.45 points above the otherwise
identical uniform run after both are migrated to the clean model.  This is a
training-only weighting of the same cross-entropy; it adds no inference logic.
The model-derived difficulty score is useful for collecting data: the hardest
10% of Dev rows are all errors and contain 45.91% of all errors; the hardest
20% contain 86.55% of all errors.

Clean isolated RTX 4090 latency is 9.70 ms p50 / 10.81 ms p95 for the model,
and 27.18 ms p50 / 85.46 ms p95 / 176.42 ms max end-to-end.  It therefore
satisfies the 100–200 ms runtime requirement with substantial margin.

This checkpoint is **not accepted for production**.  It remains below the old
released Dev result (78.67%) and far below the requested 90% acceptance target.
Its contribution is a smaller, causally verified research backbone: every
mapping component has measurable value and the previously inactive paths have
been removed.  The Test split is not reopened for this Dev-rejected candidate.

The complete-backbone result is the only valid starting point for subsequent
training and ablation.  The earlier local-only run is retained as a rejected
ablation and must not be promoted.

## Required ablations

The acceptance sequence changes exactly one component at a time:

1. complete frozen v9 backbone with fresh near-zero local correction;
2. learned neighbour selection versus fixed neighbours;
3. additive local evidence versus the rejected multiplicative AND;
4. one versus three correspondence updates;
5. 1024D state readout disconnected versus connected to matching;
6. uniform full-data training, followed only if useful by row-level hard-example
   replay.

Checkpoint selection uses fixed-model Dev Top-1, then Dev loss.  Test is run
once after selection.  The target is strict ASE Test Top-1 above 90%; it is an
experimental acceptance threshold, not a theoretical guarantee.
