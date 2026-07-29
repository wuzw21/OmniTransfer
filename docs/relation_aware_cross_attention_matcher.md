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

A learned relation gate determines how much node `k` changes node `i`:

```text
[g_ik, b_ik] = MLP_relation(r_ik)
e_ik = 2 * sigmoid(g_ik) * (q_i dot k_k / sqrt(d)) + b_ik
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
S_ij = (W_m h_i) dot (W_m h_j) / sqrt(d)
     + matchability(h_i) + matchability(h_j)
P_ij = 0.5 * (log softmax_j(S_ij) + log softmax_i(S_ij))
```

Only actionable rows and columns enter the partial assignment. Non-actionable
text and icon nodes remain graph context. Every layer emits a partial
assignment and receives the same symmetric supervision, so cross-attention
progressively refines correspondence rather than merely decorating a final
pointwise classifier.

The default model uses hidden size 64, two layers, four heads, 48 source context
nodes, and 64 target context nodes. It has 540,197 trainable parameters; the
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

## BMOCA Three-Environment Robustness Study

The 2026-07-30 BMOCA study adapts the successful traces in
`bmoca-three-env-success-aligned-dataset-v5` to the same
`omnitransfer.ui_correspondence_pair.v1` representation used by every other
training source. The raw corpus contains 226 traces and 726 action pairs. After
rebinding executed actions to the full XML graphs, collapsing repeated evidence,
and retaining set-valued labels, it yields 521 directed page pairs and 625
correspondence edges.

The frozen seed-17 split is made within each App by page-connected components:

| Partition | Page pairs | Correspondence edges |
| --- | ---: | ---: |
| Train | 421 | 520 |
| Reserved dev | 50 | 50 |
| Reserved test | 50 | 55 |

No page, page-pair, or connected component occurs in more than one partition.
The held-out evaluation is therefore an unseen-page-component test inside known
Apps, not an unseen-App or unseen-task test. The trace alignments are automatic,
currently have `label_status=unreviewed`, and are a robustness diagnostic rather
than a formal human-gold result.

Frozen data SHA-256 values are:

```text
train.jsonl       446a715ca7e5d3c6fc2b369e95e80a789a557aa7102bba183c01fdf485ce6e5d
diagnostic.jsonl  0b10e25b86197090acd358198c9a00656f8969384d6c0a33049f9bd1f11bb513
pool.jsonl        fe2abdd60241138abf43bc6257e6e4c60cb079d77977b65e26ccd8559cc1e0bc
manifest.json     9e787639d39becd2974cd0cadee3c1d412e3c079a25f1022002e96eb9770d0e2
```

All runs use the same 540,197-parameter, two-layer matcher, seed 17, and
symmetric partial-assignment loss. The action context-mask rate is listed per
run and is 5% unless noted otherwise. The complete per-update metrics, epoch
losses, commands, environment, input hashes, and checkpoint hashes remain in
each server run directory.

| Run | Initialization and treatment | Epochs | Train fit | Dev Top-1 | Test Top-1 | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | scratch, pre-fix graph materialization, mask 0.10 | 8 | 95.56% | 94.00% | 98.18% | 96.19% |
| v2 | scratch, pre-fix graph materialization, mask 0.05 | 4 | 91.80% | 90.00% | 90.91% | 90.48% |
| v3 | scratch, frozen v2 data, mask 0.05 | 6 | 95.37% | 92.00% | 96.36% | 94.29% |
| **v4** | v1 warm start, frozen v2 data, mask 0.05 | 2 | **97.39%** | **94.00%** | **100.00%** | **97.14%** |
| v5 | v4 warm start, learned visual crops | 2 | 97.30% | 94.00% | 100.00% | 97.14% |

The retained v4 checkpoint is:

```text
/home/wuzewen/omnitransfer_eval_20260620/runs/
  bmoca_graph_attention_v2data_finetune_cuda1_seed17_20260730_v4/checkpoint.pt
SHA-256 ec536c95cf855c042b458b55320e9239962d74bdf5d5856fa29c44c6907ec1bf
```

On its 210 bidirectional held-out queries, v4 matches 204. Dev Recall@3 and
Recall@5 are both 98%; test Recall@3 and Recall@5 are both 100%. Model p95 is
4.24 ms on dev and 6.01 ms on test; end-to-end p95 is 18.94 ms and 21.15 ms,
respectively. Every measured query is below 50 ms.

The exact-identity selector baseline reaches 92.00% dev and 94.55% test Top-1.
Its selective accuracy is 100%, but it covers only those same fractions. It is
reported only as a baseline and is not fused into OmniTransfer.

The learned visual-crop experiment is deliberately rejected as the default:
all 1,425 screenshots resolve correctly and 23,890 of 24,198 selected nodes
have valid crop boxes; the remaining boundless root/container nodes use the
learned missing-visual vector. Nevertheless, v5 does not improve the six
remaining dev errors and adds a runtime screenshot dependency. These errors are
a single repeated anonymous Settings-row cluster. Keeping v4 avoids a needless
modality dependency and preserves the simpler method. The experiment supports
the 95% robustness target on the combined held-out set, but the
unreviewed-label boundary and the 94% dev slice must remain visible in any paper
claim.

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

The learned-gate implementation was also exercised on a real MobileViews
608-by-608 actionable-node pair. One forward and backward pass used 683 MiB
peak allocated CUDA memory. This stress test verifies that attention and
assignment do not materialize a high-dimensional pair-feature volume. Its
1.47-second forward time is an extreme-node-count scalability diagnostic, not
evidence for the 50 ms runtime target; formal latency reports must include node
count and the long tail.

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
