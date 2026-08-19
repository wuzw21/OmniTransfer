# OmniTransfer unified transfer pipeline

This document describes the current release pipeline and its stable public
seam. The normative design target for the replacement matcher is
[`interaction_graph_matching_spec.md`](interaction_graph_matching_spec.md).
The replacement keeps the public seam, adapters, explicit failure behavior,
and relative action projection while moving internal correspondence from raw
pair states to context-enriched Canonical Nodes. Parent and child endpoints are
never silently merged. The two implementations must not coexist as production
fallbacks.

OmniTransfer solves one problem: given a source page, a target page, and a
source action point, identify the corresponding target UI control and return
the complete ranked target candidate set. It never copies the source screen
coordinate to the target device.

The public seam is `rank_action_candidates`; its request/result contract is
`omnitransfer.transfer-contract.v2`. Every adapter, training record, NumPy
runtime result, and review row must describe the same four-stage dataflow:

```text
source XML + screenshot + action point
                 │
                 ▼
      1. Multimodal UI encoder
         XML/screenshot → Unified UIGraph
                 │
                 ▼
      2. Relation-aware association
         candidate pairs + XML and local relative relations
                 │
                 ▼
      3. Bidirectional candidate decoder
         source→target and target→source consistency
         → complete ranking, confidence, margin
                 │
                 ▼
      4. Relative action projector
         source point → source-node offset
         offset → every target candidate
```

## One owner per stage

| Stage | Canonical owner | Contract crossing the seam |
| --- | --- | --- |
| UI encoding | `ui_graph.py`, `ios_adapter.py`, `visual_descriptor.py` | `UIGraph` / `UINode` |
| Association | `learned_matcher.py`, `geometric_matcher.py`, `unified_alignment.py` | one fused pair state plus one multi-hop local relation graph |
| Decoding | `geometric_matcher.py` and frozen `numpy_v9_matcher.py` | bidirectional logits and ranked candidates |
| Projection | `runtime.py` public transfer seam | normalized in-node offset projected to candidate bounds |

`mapping_pair_review.py` and the HTML review are adapters of the same public
seam. They do not get a review-only mapper. Training uses the same graph,
relation, strategy, and assignment contracts; it only swaps the learned
parameters and labels.

## Invariants

- XML, text, content description, class, affordance state, screenshot crop,
  hierarchy, and local relative relations enter through `UIGraph`.
- Whole-page absolute position is weak evidence. Local context is represented
  by one relation graph whose channels include parent/child, sibling,
  ancestor/descendant, local direction/proximity, and normalized kinship
  distance. The same graph layer is applied repeatedly, so multiple hops are
  learned by depth rather than by separate voting algorithms.
- The decoder returns all valid target candidates, not only Top-1.
- The projector uses only the source point's relative position inside the
  selected source node. It never replays a source-device coordinate.
- A missing or low-confidence mapping is an explicit transfer failure so the
  caller can invoke its VLM fallback.
- `resource_id`, app identity, and review-specific corrections are not mapping
  evidence.

The machine-readable version of this contract is
`src/omnitransfer/unified_alignment.py`.

The runtime has one scoring path:

```text
multimodal node encoder
    -> fused candidate-pair state
    -> K repeated layers of one local relation function
    -> one association score
    -> bidirectional assignment normalization
```

It does not short-circuit equivalent XML pages and it does not swap the
learned winner with a deterministic reranker. Local relations and geometry
are model features, so their contribution is learned from correspondence
labels.
