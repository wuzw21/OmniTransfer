# OmniTransfer Correspondence Context

OmniTransfer describes a visible mobile interface as an imperfect observation
of user-facing controls. Its language distinguishes capture artifacts from the
stable interaction concepts used for cross-device correspondence.

## Observation

**UI Observation**:
A synchronized XML/accessibility snapshot and screenshot from one visible
mobile state.
_Avoid_: Page XML, layout truth

**Raw Node**:
One node emitted by an Android or iOS observation tree. A Raw Node is capture
evidence and is not assumed to be a stable control identity.
_Avoid_: Element, widget identity

**Window Layer**:
One independently ordered visible surface, such as the application window, a
dialog, a popup, or a system overlay.
_Avoid_: XML root, container

**Visual Region**:
A bounded screenshot area that supplies appearance evidence for one or more
Raw Nodes or Canonical Nodes.
_Avoid_: XML bounds

**Observation Graph**:
The typed, reliability-bearing graph formed from Raw Nodes, Visual Regions,
Window Layers, and their observed relationships.
_Avoid_: XML tree, ground-truth hierarchy

## Canonical matching node

**Canonical Node**:
The normalized Android/iOS observation node used as an exact correspondence
endpoint. Its representation includes local nodes and Visual Regions, but those
neighbors do not become equivalent labels.
_Avoid_: XML path identity, merged parent/child unit

**Execution Target**:
The selected target node and projected in-node action returned after matching.
Executability is resolved by the runtime contract and is not node identity.
_Avoid_: Clickability match, source-coordinate replay

**Local Context**:
The bounded set of structural, spatial, visual, and window-layer relationships
that gives a Canonical Node its role.
_Avoid_: Absolute page position, surrounding nodes

**Transient Layer**:
A Window Layer whose presence is temporary relative to the underlying page,
such as an advertisement, permission dialog, or popup menu.
_Avoid_: Noise node, bad XML

## Correspondence

**Correspondence**:
A learned functional relationship between a source Canonical Node and a target
Canonical Node.
_Avoid_: Node-id match, coordinate match

**Candidate Ranking**:
The ordered set of every valid target Canonical Node for one source action,
including scores and calibrated uncertainty.
_Avoid_: Top-1 prediction, assignment result

**Matchability**:
The calibrated probability that a source Canonical Node has a valid target
counterpart in the current target observation.
_Avoid_: Similarity score, confidence threshold

**Transfer Failure**:
The explicit result produced when no target correspondence is sufficiently
supported. It permits the caller to use its normal fallback path.
_Avoid_: Source-coordinate replay, empty success

## Page Representation

**Stable Configuration**:
The persistent organization of a page after discounting transient layers and
capture-specific tree expansion.
_Avoid_: Screenshot hash, XML identity

**Active State**:
The currently actionable composition of the Stable Configuration and all
visible Window Layers.
_Avoid_: Page name, app state string
