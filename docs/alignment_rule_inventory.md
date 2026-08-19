# OmniTransfer alignment rule inventory

This inventory distinguishes three different things that were previously
mixed together:

1. **learnable evidence**: features or relation channels whose contribution is
   fitted from correspondence labels;
2. **runtime contracts**: safety gates such as candidate validity, NULL, and
   fallback behavior; and
3. **data/review policy**: preprocessing, hard-example selection, and manual
   review classification.

## Current rule surfaces

The repository currently has 18 mapping-affecting rule surfaces:

| Surface | Current responsibility | Correct long-term owner |
| --- | --- | --- |
| `ui_graph.py` | XML/JSON normalization, visibility, coordinate-space interpretation, local graph construction | graph parser + coordinate contract |
| `ios_adapter.py` | iOS XML normalization and display/screenshot coordinate conversion | platform adapter |
| `build_vision_widget_pair_pool.py` | pair construction and normalized target geometry | dataset builder |
| `learned_matcher.py` | node encoding, direct pair features, typed local relation bases | model + `unified_alignment` |
| `geometric_matcher.py` | trainable node/pair fusion, one multi-hop relation graph, one association score | trainable matcher |
| `numpy_v9_matcher.py` | frozen runtime reproduction of the model | exported checkpoint runtime |
| `self_supervised.py` | relation-preserving augmentations, candidate/NULL training policy, losses | training objective |
| `mapping_training.py` | correspondence extraction and assignment label policy | dataset/training contract |
| `mapping_dataset.py` | schema, split, leakage, and label validation | dataset contract |
| `runtime.py` | bounded all-node candidate set, confidence/margin, fallback contract | runtime safety policy |
| `mapping_baselines.py` | exact identity baselines | diagnostic baseline only |
| `utg_mapping_candidates.py` | legacy weighted candidate proposal | legacy diagnostic path; not runtime mapping |
| `icon_hard_set.py` | hard-case selection and difficulty reasons | sampling/diagnostics only |
| `mapping_pair_review.py` | review queue selection and live ranking API | review adapter |
| `build_unified_error_review.py` | unified-error joins and review labels | permanent regression/evidence store |
| `build_utg_point_mapping_review.py` | UTG review queue and manual-label export | review adapter |
| `train_geometric_v9_matcher.py` | training configuration and experiment gates | experiment runner |
| `review_annotation_template.html` | manual point capture, verdict, and browser-side validation | review UI contract |

## Quantitative audit

An AST audit over `src/omnitransfer/*.py` and `scripts/*.py` found:

- 32 Python modules with branching/selection logic;
- 1,511 branch-like AST nodes;
- 823 comparison nodes; and
- 289 calls to selector-style builtins (`min`, `max`, `sorted`, `any`, `all`,
  or `sum`).

These numbers are an audit signal, not a claim that every branch is a model
rule. They show why adding another special-case scorer is unsafe: the same
concept is currently implemented in multiple owners, notably the PyTorch and
NumPy matcher paths and the review/UTG candidate paths.

## Required architecture

All learnable alignment evidence now has one registry in
`src/omnitransfer/unified_alignment.py`. It exposes only two model inputs: the
multimodal node/pair encoding and the multi-hop local relation graph. The
association network learns their interaction from labelled correspondences;
there are no parallel direct/anchor/context/residual score heads. It
deliberately has no app names, translated label lists, fixed score weights,
resource-id lookup, or source coordinate replay.

The training objective also contains a generic relation-preservation term
(`relation_consistency_loss`). It learns the compatibility between source and
target relation types from labelled node pairs; it does not directly add a
positive or negative logit to any candidate. The existing assignment loss
trains the single association score, while the generic relation-consistency
term trains relation compatibility. Human knowledge is therefore injected at
representation/compatibility level rather than implemented as a review rescue
branch.

The main path deliberately removes two non-learning branches: equivalent-graph
identity replay and the deterministic Spatial-XML post-reranker. Both could
override a model decision without being represented in the training loss.

The remaining code is grouped by seam rather than by individual exception:

| Unified seam | Kept responsibility | Removed from the main path |
| --- | --- | --- |
| `ui_graph.py` + `ios_adapter.py` | normalize platform observations into one graph | platform-specific mapper branches |
| `unified_alignment.py` | name learnable evidence and transfer metadata | duplicate strategy/contract modules |
| `learned_matcher.py` + `geometric_matcher.py` | encode nodes, propagate one local relation graph, learn one association score and bidirectional decoding | fixed human score overrides and parallel voting heads |
| `runtime.py` | validate candidates and project relative action offsets | equivalent XML identity shortcut |
| offline review/baseline tools | diagnose, label, and measure errors | no second production mapper |

Human knowledge enters through generic supervision and invariants: semantic
compatibility, affordance compatibility, relation preservation, local
neighborhood consistency, and relative geometry. These are feature families,
not deterministic verdicts. The unified-error Pinterest example remains a
held-out regression and a human-labelled training example; it is not a
Pinterest-specific rule.
