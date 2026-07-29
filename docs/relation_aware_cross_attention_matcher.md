# OmniTransfer

## Core Idea

OmniTransfer learns which target node corresponds to a source node after a GUI
changes device, density, viewport, or layout. It does not learn where a natural
language instruction should click, and it does not combine a learned score with
a selector or hand-written matching formula.

The paper-facing statement is:

> Given two UI graphs, OmniTransfer jointly encodes node semantics, appearance,
> and within-page relative structure, exchanges information with bidirectional
> cross-attention, and predicts a dense source-target correspondence matrix.

There is one matcher, one forward path, and one training objective.

## Training Unit

Every training example has the same representation:

```text
CorrespondencePair(source_graph, target_graph, correspondences)
```

The pair can come from either of two transformations of the unified training
split:

1. **Augmented-view pair.** Two views are generated from one training page.
   Known transformation identity constructs auxiliary correspondence labels.
2. **Cross-page pair.** Human annotations or offline self-supervised alignment
   supply multiple aligned actionable nodes under a real platform or layout
   change. Label provenance remains explicit and determines which evaluation
   claims are valid.

The origin or stable node identity is used only to construct offline labels. It
is never passed to the matcher. Both pair sources use the same tensors, matcher,
loss, optimizer, and training history.

## Model

### Node Encoding

For every node `i`, OmniTransfer builds one 64-dimensional state by summing three
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

### Learned Graph Attention

For nodes `i` and `k` on the same page, relation features describe hierarchy and
relative layout:

```text
parent / child / sibling / ancestor
same row / same column / overlap / neighbor
relative position / relative size / IoU / tree distance
```

A learned compatibility function determines how much node `k` changes node
`i`:

```text
e_ik = MLP_score([q_i, k_k, |q_i-k_k|, q_i*k_k, MLP_relation(r_ik)])
A_ik = softmax_k(e_ik)
h_i' = h_i + sum_k A_ik W_v h_k
```

Every selected node can participate. "Local" is therefore not a coordinate
window and there is no fixed distance weight: the model learns which semantic,
tree, and relative-layout relations are useful. The residual preserves the
node's own evidence as the primary signal while context changes its descriptor.
Absolute screen position is never observed.

### Cross-Page Matching

Each matcher layer first applies learned graph attention independently to both
pages. One shared projected affinity then performs bidirectional cross-attention:

```text
C_ij = (W_c h_i^source) dot (W_c h_j^target) / sqrt(d)
source_i <- source_i + MLP([source_i, softmax_j(C_ij) W_v target_j])
target_j <- target_j + MLP([target_j, softmax_i(C_ij) W_v source_i])
```

The score for source node `i` and target node `j` is then learned only from
their target-conditioned contextual descriptors:

```text
S_ij = MLP([h_i, h_j, |h_i - h_j|, h_i * h_j])
P_ij = 0.5 * (log softmax_j(S_ij) + log softmax_i(S_ij))
```

Only actionable rows and columns enter the partial assignment. Non-actionable
text and icon nodes remain graph context. Every layer emits a partial
assignment and receives the same symmetric supervision, so cross-attention
progressively refines correspondence rather than merely decorating a final
pointwise classifier.

The default model uses hidden size 64, two layers, four heads, 48 source context
nodes, and 64 target context nodes. It has 557,400 trainable parameters; the
exact count is stored with every checkpoint.
If a page has more actionable nodes than the nominal context limit, every
actionable node is retained so it remains an explicit candidate hard negative.

## Unified Training

Each epoch builds one shuffled stream containing both augmented-view pairs and
cross-page pairs from the declared unified training split. Augmented views are
generated lazily when sampled so they do not create another serialized dataset
or trainer.
Every actionable node is retained in both augmented views. Node dropout applies
only to non-actionable context, so sibling buttons, rows, and menu items remain
real in-screen hard negatives.

The same-page augmentation masks text and descriptions, perturbs relations,
drops non-actionable context, injects distractors, and hides visual crops for
half of the nodes by default. In addition, 5% of supervised action nodes lose
their own text, description, and crop inside the loss computation. The node
must then be identified from class/action state and attention over the
surrounding nodes. This is the direct training signal for anonymous clickable
containers; it does not synthesize a parent label.

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

The current method does not claim learned NULL rejection. A no-match objective is
added only when formal human-labeled NULL examples exist. Until then, formal
matching results report positive Top-1 and Recall@K.

## Data Boundary

Every record uses `omnitransfer.ui_correspondence_pair.v1`. MobileViews
Android-to-Android layouts and ASE iOS-to-Android mappings enter one pool before
the split is assigned; dataset origin never creates a second model or
optimization path. Splits are formed inside each App by page-connected
components. Exact pair, page, component, and partition identities cannot cross
splits.

Automatic MobileViews correspondences retain
`label_status=self_supervised`. They may train the matcher and support
page-disjoint diagnostics, but they are never reported as reviewed-gold
accuracy. Formal modern Android-to-Android results require the reviewed
Benchmark B export; formal ASE results use only its original human-gold test
annotations. Ambiguous equivalent targets are represented as set-valued labels
rather than forced into a false one-to-one mapping.

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

The default-strength selection, complete development ablation, and artifact
contract are frozen in `docs/omnitransfer_experiment_appendix_20260729.md`.

## Historical Implementation Smoke

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
