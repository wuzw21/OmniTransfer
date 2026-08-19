# OmniTransfer Interaction-Graph Matching Specification

Status: normative design target; not yet a trained runtime release.

This document defines the one allowed design for the next OmniTransfer matcher.
It keeps the public `rank_action_candidates` interface and replaces the current
pair-state association implementation only after the implementation, export,
and acceptance gates in this specification pass. Domain terms have their
canonical meanings in [`CONTEXT.md`](../CONTEXT.md).

## 1. Objective and scope

OmniTransfer receives source and target UI observations plus a source action.
It must identify the source Canonical Node, return the complete ranked set of
target Canonical Nodes, estimate whether a valid correspondence exists, and
project the source action inside every target candidate.

The matcher does not choose a task action, execute on a device, own fallback
policy, or treat an XML path, resource ID, app identity, or source coordinate as
a stable cross-platform identity.

The governing principle is:

> Layout and vision are equally necessary observations of a user-facing
> Canonical Node. XML structure routes evidence but is never ground truth.

## 2. Stable external seam

The only runtime seam remains `rank_action_candidates`. Callers provide the
source observation, target observation, and source action. They receive one
result containing:

- the resolved source Canonical Node;
- every valid target candidate in rank order;
- each candidate's Execution Target, visual bounds, projected action, score,
  calibrated probability, and uncertainty evidence;
- source and target Stable Configuration embeddings;
- source and target Active State embeddings;
- explicit mapping success or Transfer Failure metadata;
- model, schema, checkpoint, and release provenance.

No caller selects an encoder, relation scorer, grouping rule, or assignment
algorithm. Those choices are hidden inside one deep matcher module and tested
through the public seam.

## 3. Required data model

### 3.1 UI Observation

An observation must synchronize XML/accessibility content with its screenshot.
It must preserve platform, screen dimensions, screenshot dimensions, capture
time when available, and every visible window root. An adapter must reject or
mark unsynchronized observations; it must not silently combine unrelated XML
and screenshots.

Android and iOS adapters may normalize source formats, but matching code must
consume one platform-neutral observation contract. Coordinate conversion is an
adapter responsibility and occurs exactly once.

### 3.2 Raw Node

Every Raw Node must retain, when present:

- text, content description, class/role, resource ID, and package;
- clickable, editable, scrollable, enabled, checkable, checked, selected,
  focused, long-clickable, expanded/collapsed, and password state;
- screen, window, and visual bounds with their coordinate spaces;
- raw parent, children, sibling order, depth, window identity, and Z-order;
- source node path and source-specific identifiers as provenance only;
- explicit presence masks for every missing attribute family.

The node path and resource ID must never be used as direct correspondence
answers. Repeated IDs and paths changed by wrapper insertion are expected.

### 3.3 Observation Graph

The graph must contain typed edges from independent evidence sources:

| Edge family | Required examples | Interpretation |
| --- | --- | --- |
| Structural | parent, child, sibling, ancestor distance | Noisy XML evidence |
| Spatial | contains, overlaps, near, row, column, direction | Relative local layout |
| Local role | semantic adjacency, shared visual group, shared hit region | Context without endpoint equivalence |
| Collection | same list, same repeated item, item order | Dynamic list context |
| Window | same layer, occludes, above, active window | Popup and overlay context |
| Temporal | persisted, appeared, disappeared, state-changed | Optional transition evidence |

Every edge carries a source and reliability. Raw XML edges are not privileged
over visual containment or interaction evidence. Whole-page absolute position
may be retained as weak context but cannot be a decisive edge family.

### 3.4 Contextual Node Representation

Matching preserves the labelled canonical node as the correspondence endpoint.
Parents, children, siblings, ancestors and nearby visual nodes contribute typed
context but are never silently expanded into equivalent labels. This prevents a
semantic child or clickable parent from changing the training target.

Clickability is execution metadata, not identity evidence. Clickable and
non-clickable nodes are retained under the same sampling policy and can both be
important local neighbors. Execution resolves the selected target node under
the runtime contract or fails closed.

## 4. Layout design requirements

Layout describes local organization, not screen coordinates.

### Required

- Encode relative direction, distance, overlap, containment, size ratio, row,
  column, sibling order, tree distance, list membership, and window layer.
- Use both XML topology and screenshot-derived spatial topology.
- Represent parent/child distance continuously so a wrapper insertion weakens
  evidence instead of changing a relation from true to false.
- Detect repeated list-item structure and give every item a local item group.
- Isolate active dialogs, popups, and overlays as Window Layers while retaining
  their relationship to the obscured base page.
- Allow local groups to overlap and allow attention to escape a wrong group.
- Grow the receptive field by repeating one local aggregation layer.

### Forbidden

- Absolute-coordinate nearest-neighbor matching.
- XML path, child index, or resource ID lookup as a production answer.
- A single scalar `relation_score` added after node similarity.
- Hard rejection solely because parent depth or wrapper count differs.
- Flattening every window and list item into one undifferentiated page graph.

## 5. Vision design requirements

Vision describes what the user can distinguish and what XML fails to expose.

### Required

- Extract a spatially preserving multi-scale screenshot feature map once per
  observation.
- Obtain each unit's internal appearance, an expanded context ring, its local
  group region, and a low-resolution page context from that shared feature map.
- Preserve icon geometry, edges, selection marks, text appearance, occlusion,
  and contrast; do not reduce a crop to global channel averages.
- Keep separate presence and quality signals for empty, clipped, tiny, covered,
  or invalid crops.
- Use visual containment and segmentation cues to repair XML wrappers and to
  distinguish adjacent textless controls.
- Share the same visual encoder between Android and iOS.

### Forbidden

- Treating vision as a late tie-breaker.
- Encoding only the exact XML bounding box without surrounding context.
- Averaging text, vision, and layout branch scores.
- Substituting OCR text for icon or geometry evidence.
- Re-encoding the full screenshot independently for every candidate pair.

## 6. Unified evidence representation

Each Canonical Node is represented as typed evidence rather than three
pre-averaged branches. Required token families are:

- semantic: text and content description;
- visual: internal, context-ring, and group-region features;
- role: normalized control class and affordance;
- state: actionable and mutable control state;
- geometry: local size and aspect evidence;
- reliability: modality presence, crop quality, XML confidence, and layer
  visibility;
- group: collection, region, and window membership.

Missing evidence uses explicit missing tokens. Zero vectors must not ambiguously
mean both "missing" and "observed empty". The representation dimension is an
implementation choice validated by ablation; it is not a hand-authored semantic
partition such as fixed text and vision halves.

Equal importance does not mean a fixed 50/50 mixture. The model must learn a
sharp reliability-dependent choice: a text-bearing control may be dominated by
semantic evidence, a textless icon by visual and local-layout evidence, and a
noisy WebView node by group and window context. No modality is globally weak,
and no modality receives a fixed per-candidate bonus.

## 7. Contextual correspondence core

The core is one repeated Contextual Correspondence Layer. Every layer performs
the same conceptual work:

1. **Localize**: select reliability-weighted structural, spatial, interaction,
   collection, and window neighbors.
2. **Contextualize**: update each unit from those neighbors; relations route
   messages and do not emit independent verdicts.
3. **Align**: exchange information bidirectionally between source and target
   soft groups, from coarse regions to individual units.
4. **Refine**: update unit identity using the cross-observation evidence.
5. **Estimate**: emit the current candidate distribution, NULL matchability,
   and representation stability.

The same layer is stacked. Early layers resolve strong visual or semantic
matches; deeper layers increase the local receptive field for repeated icons,
lists, wrappers, and cross-platform rearrangements. Every layer receives
mapping and matchability supervision.

The source action node is never pruned. Other nodes may be removed from later
layers only when they are both stable and confidently unmatchable. Inference may
exit early when the requested mapping and its local support are stable.

This design adopts iterative refinement and matchability from LightGlue and
soft hierarchical node-to-group interaction from AutoGameUI. It does not copy
their keypoint, absolute-layout, or integer-programming assumptions.

## 8. Decoder, confidence, and action projection

The decoder must preserve all valid target candidates. It must support
set-valued labels, shared targets, and no-correspondence cases. A one-to-one
Hungarian solution may exist as a diagnostic baseline but is not the production
selector.

Mapping success is calibrated from jointly learned evidence:

- NULL matchability;
- candidate probability distribution;
- top-one/top-two separation;
- source-to-target and target-to-source agreement;
- local-context preservation;
- prediction stability across layers;
- input and crop quality.

No fixed confidence threshold is a model claim until calibrated on a disjoint
development split. Calibration and ranking scores must be returned separately.

Action projection uses only the source action's normalized position inside the
source Canonical Node and the candidate node's visual/action bounds. It
must never replay a source-device coordinate.

## 9. Page-level representation

The shared encoder must produce two readouts without a second page encoder:

- **Stable Configuration embedding**: persistent page organization with
  transient layers and capture-specific subtree expansion discounted;
- **Active State embedding**: the currently visible and actionable composition,
  including dialogs, ads, selection state, and active window.

Both readouts use learned page/group tokens and contextual aggregation. Mean
pooling all nodes is forbidden as the sole page representation.

## 10. Training specification

### 10.1 Training records

Gold labels identify exact source and target canonical nodes and retain every
explicitly labelled valid target. Cleaning may annotate slices or quarantine a
conflict, but it must not expand parent/child endpoints. Page-level splits must
prevent app, trace, screenshot, or near-duplicate leakage.

Self-supervised and synthetic records are auxiliary evidence. They must be
marked separately and cannot become headline gold evaluation rows.

### 10.2 Online XML and visual corruption

Training must randomly compose mapping-preserving corruptions instead of
requiring a brittle manual stratified sampler:

- insert, remove, or reorder wrapper nodes;
- move clickability between a parent and visual child;
- merge or expand semantics subtrees;
- insert, remove, duplicate, reorder, or scroll list items;
- expand a WebView into many nodes or collapse it into one region;
- add or remove transient dialog, permission, advertisement, and popup layers;
- drop text, content description, resource ID, class detail, or bounds;
- change checked, selected, focused, expanded, and enabled state;
- crop, occlude, rescale, recolor, or slightly move visual controls;
- perturb Android/iOS role and wrapper conventions without changing function.

Corruptions run only on training data and preserve a replayable provenance log.

### 10.3 Learned objectives

Training must supervise four observable capabilities:

- full candidate ranking and set-valued correspondence;
- valid counterpart versus NULL matchability;
- preservation of reliable local context under noisy edges;
- Stable Configuration and Active State representation.

Human knowledge enters as graph construction, corruption policy, and labelled
outcomes. It must not enter as app-specific score bonuses or post-model rescue
rules.

## 11. Runtime and performance requirements

- Parse and encode each observation once; candidate count must not multiply
  screenshot encoding cost.
- Bound local neighborhoods and use adaptive depth/pruning for runtime cost.
- Preserve the requested source unit and top active-window candidates under all
  pruning policies.
- Export one frozen mobile checkpoint whose NumPy/runtime implementation is
  numerically validated against the training implementation.
- Missing, corrupt, or incompatible checkpoints fail closed.
- Review, evaluation, page embedding, and production runtime use the same
  model, adapter, feature schema, and candidate ranking.

## 12. Acceptance matrix

No matcher release is valid without the following disjoint slices:

| Slice | Required behavior |
| --- | --- |
| Textless adjacent icons | Distinguish controls by icon and local sibling role |
| Non-clickable local context | Retain and use semantic/structural neighbors independent of actionability |
| Recycler/list mutation | Survive insertion, deletion, recycling, and reorder |
| Popup/ad/permission layer | Respect active layer and avoid obscured background |
| WebView expansion | Remain stable across coarse and highly expanded XML |
| Control state change | Preserve identity across checked/selected/focused changes |
| Android/iOS | Use the same representation and coordinate contract |
| Unified-error | Pass every permanent human-reviewed regression |
| No correspondence | Abstain without source-coordinate replay |

Report at minimum Top-1, MRR, Top-k recall, NULL AUROC, expected calibration
error, false-success rate, per-slice metrics, and mobile latency. Aggregate
improvement cannot hide a regression on unified-error, active-window safety, or
no-correspondence cases.

## 13. Ownership and migration

The target ownership is:

| Concern | Sole owner |
| --- | --- |
| Android/iOS normalization and coordinate spaces | platform adapters + `ui_graph.py` |
| Observation Graph and contextual node construction | one interaction-graph module |
| Typed evidence and repeated correspondence layers | `learned_matcher.py` + `geometric_matcher.py` |
| Frozen mobile reproduction | `numpy_v9_matcher.py` or its single successor |
| Ranking, failure, and relative projection | `runtime.py` |
| Human evidence and permanent regressions | canonical review + unified-error store |

Migration replaces the current unary-affinity/pair-state/relation-score core; it
must not add a second production mapper. Old tests that assert internal pair
states are replaced by tests through the matcher interface. Existing public
result fields remain stable unless a separately reviewed schema migration is
performed.

The new release may be declared only after a new checkpoint is trained,
exported, hash-pinned, calibrated, compared against the current release on all
acceptance slices, and verified in the live review workbench.

## 14. Design review checklist

Every implementation proposal must answer all of the following before code is
accepted:

1. Which domain object owns this information: Raw Node, Canonical Node,
   Window Layer, Observation Graph, or mapping result?
2. Is the evidence structural, spatial, interaction, collection, window,
   temporal, semantic, visual, state, or reliability evidence?
3. Is missingness explicit?
4. Can wrapper insertion, list recycling, popup insertion, or state change alter
   the answer incorrectly?
5. Does the implementation update node/unit representation, or merely add a
   late score? Late evidence scores are rejected.
6. Is the behavior learned and visible to the training objective?
7. Is there exactly one production owner and one public runtime path?
8. Does the frozen runtime reproduce training inference?
9. Is failure explicit and source-coordinate replay impossible?
10. Which acceptance slice proves the design works?
