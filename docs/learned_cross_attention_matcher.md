# Local Relation Cross-Attention Matcher

## Core Idea

OmniTransfer learns which target node corresponds to a source node after a GUI
changes device, density, viewport, or layout. It does not learn where a natural
language instruction should click, and it does not combine a learned score with
a selector or hand-written matching formula.

The paper-facing statement is:

> Given two UI graphs, OmniTransfer jointly encodes node semantics, appearance,
> and within-page relative structure, predicts a dense source-target
> correspondence matrix, and explicitly propagates matched-neighbor evidence
> back to each actionable candidate pair.

There is one matcher, one forward path, and one training objective.

## Training Unit

Every training example has the same representation:

```text
TrainingPair(source_graph, target_graph, correspondences)
```

The pair can come from either of two sources:

1. **Augmented-view pair.** Two views are generated from one MobileViews page.
   Known transformation identity constructs the correspondence labels.
2. **Cross-page pair.** Two MobileViews pages provide multiple aligned nodes
   under a real layout or device change.

The origin or stable node identity is used only to construct offline labels. It
is never passed to the matcher. Both pair sources use the same tensors, matcher,
loss, optimizer, and training history.

## Model

### Node Encoding

For every node `i`, the model builds one 64-dimensional state by summing three
learned embeddings:

```text
x_i = text/content-desc embedding + class/action-state embedding + visual crop embedding
```

The action state contains only clickable, editable, scrollable, and enabled.
A small shared CNN encodes the screenshot crop when available; a learned
missing-visual vector keeps XML-only nodes on the same path. Raw resource id,
absolute coordinates, absolute box size, page order, and offline stable ids are
not model inputs. The bbox is used only to extract the visual crop and construct
within-page relative relations.

### Local Relation Encoding

For nodes `i` and `k` on the same page, relation features describe hierarchy and
relative layout:

```text
parent / child / sibling / ancestor
same row / same column / overlap / neighbor
relative position / relative size / IoU / tree distance
```

A relation MLP produces an attention bias rather than a fixed similarity score:

```text
P_ik = softmax(q_i k_k / sqrt(d) + MLP_relation(r_ik))
```

This is how the model learns local relationships without observing a node's
absolute screen position.

### Cross-Page Matching

Each matcher layer performs relation-aware self-attention on both pages and
then bidirectional cross-attention:

```text
source <- attend(source, target)
target <- attend(target, source)
```

The model first predicts a dense base affinity matrix from contextual node
descriptors. There is no direct cross-page geometry input:

```text
A0_ij = MLP([h_i, h_j, |h_i-h_j|, h_i*h_j])
```

The final score performs one explicit local correspondence propagation:

```text
S = A0 + P_source A0 P_target^T
```

`P_source` and `P_target` are the learned within-page relation-attention
matrices with the diagonal removed and each remaining row renormalized. The
support term therefore comes from other nodes rather than duplicating a node's
own base score. Consequently, a strong text-node or icon-node correspondence
directly supports the pair of actionable containers that attend to those
nodes. Unlike independent descriptor pooling, the local evidence is propagated
in affinity space and therefore retains which source evidence matched which
target evidence. All selected nodes participate in this computation; only
actionable rows and columns are supervised and returned as interaction targets.

The default model uses hidden size 64, two layers, four heads, at least 48 source
context slots and 64 target context slots, and 579,926 trainable parameters.
If a page has more actionable nodes than the nominal context limit, every
actionable node is retained so it remains an explicit candidate hard negative.

## Unified Training

Each epoch builds one shuffled stream containing both augmented-view pairs and
cross-page pairs. Augmented views are generated lazily when sampled so the
complete MobileViews training set does not need to be materialized in memory.

For known correspondences, training minimizes one symmetric objective:

```text
L = 0.5 * (CE(S_source_to_target, Y_source_to_target)
         + CE(S_target_to_source, Y_target_to_source))
    + lambda * L_bidirectional_consistency
```

The consistency term encourages the probability of a labeled pair to agree in
both directions. Multiple correspondences from the same page pair are learned
together. Supervised rows and ranking candidates are actionable nodes only.
Non-actionable text and icon nodes remain attention context, while unannotated
rows are ignored rather than treated as negative or NULL. Descendant labels are
never lifted into synthetic actionable-container labels.

The current core does not claim learned NULL rejection. A no-match objective is
added only when formal human-labeled NULL examples exist. Until then, formal
matching results report positive Top-1 and Recall@K.

## Data Boundary

MobileViews is the unified training, validation, and primary test source.
Canonical App identity is split before page-pair construction, so train, dev,
and test share no App, page, pair, or partition identity.

Self-supervised proposals may enter only the training split. Formal dev and
test records require reviewed or original gold correspondences. Ambiguous
one-to-many rows and conflicting target assignments are filtered from the
current loss. ASE 2023 is retained only as an external cross-platform gold test.

## Evaluation

The primary table reports:

```text
Top-1
Recall@3
Recall@5
model p50 / p95 latency
end-to-end p50 / p95 latency
parameter count
```

The main comparison is against the rule matcher and identity selector as
baselines. Their scores never enter the learned matcher. A hard slice contains
reviewed cases where these baselines disagree with the learned matcher or fail
under large layout changes.

The runtime target is less than 50 ms per page pair on one RTX 4090 after model
warmup, with graph/input preparation and model time reported separately.

## Current Smoke Result

The deterministic two-pair implementation smoke uses four augmented-view pairs
and two cross-page pairs in one optimizer loop. Two identical runs produce:

```text
training updates: 6
known bidirectional labels: 212
Top-1: 75.0%
Recall@3: 87.5%
Recall@5: 100.0%
checkpoint SHA-256: identical across runs
```

This result verifies the implementation and reproducibility only. It is not a
generalization score and must not appear as the paper's formal accuracy.

## Non-Goals

The core method does not use:

- a grounding VLM at runtime;
- selector or resource-id score fusion;
- hand-written geometric weighting;
- coordinate passthrough;
- reinforcement learning;
- a separate page-embedding branch;
- GUIOdyssey training examples.

If matching is absent, invalid, or below the calibrated runtime confidence, the
transfer fails and control returns to the VLM. Source-device coordinates are
never replayed directly on the target device.

## Related Work

- SuperGlue: contextual descriptors and learned matching.
  <https://arxiv.org/abs/1911.11763>
- LightGlue: lightweight alternating self/cross attention.
  <https://arxiv.org/abs/2306.13643>
- LoFTR: descriptors conditioned jointly on two inputs.
  <https://arxiv.org/abs/2104.00680>
- GlueStick: learned matching with explicit structure.
  <https://arxiv.org/abs/2304.02008>
- TEMdroid: contextual widget matching for GUI testing.
  <https://doi.org/10.1145/3597503.3623322>
- Vision-Based Widget Mapping, ASE 2023: cross-platform widget mapping.
  <https://doi.org/10.1109/ASE56229.2023.00068>
- MobileViews: large-scale mobile GUI data.
  <https://arxiv.org/abs/2409.14337>
