# OmniTransfer Long-Term Matcher Rules

## Normative matcher design

- Every matcher architecture, feature, training, export, runtime, and review
  change must follow `docs/interaction_graph_matching_spec.md`.
- Canonical domain terms are defined in `CONTEXT.md`. Matching preserves the
  labelled canonical node and enriches it with noisy XML and visual context;
  it never treats a parent and semantic child as interchangeable endpoints.
- Layout and vision are both required evidence. Neither may be implemented as a
  late fixed score or a review-only correction.
- The current released pipeline remains authoritative until a replacement
  checkpoint passes every acceptance slice in the normative specification.

## Mandatory unified-error contract

- `unified-error` is a permanent regression category, not a one-off review
  artifact.
- Every matcher or coordinate-adapter change must pass
  `tests/test_human_alignment_rules.py` before it is considered valid.
- The canonical Pinterest regression is `Display Options` on iOS mapping to
  Android `Sort boards by`, never to the unrelated `Find ideas` action.
- Whole-page absolute position is weak evidence. Mapping must prioritize the
  local semantic role, nearest branching parent, sibling-role sequence,
  direct-child order, and local geometry.
- Clickability is not a correspondence identity signal. Clickable and
  non-clickable nodes both participate in local context and training sampling.
- Learnable alignment evidence is registered in
  `omnitransfer.unified_alignment`; the PyTorch strategy head and frozen
  NumPy runtime must consume the same exported feature contract. Do not add a
  review-only correction, app-specific label list, fixed prior, or second
  mapper.
- Resource IDs, app identity, and source-device coordinate passthrough are not
  valid evidence for cross-platform mapping.

The detailed contract and verification command are in
`docs/unified_error_contract.md`.
