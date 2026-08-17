# Unified Review Workbench Long-Term Rule

All human-facing inspection, annotation, pair review, transfer-error analysis,
and benchmark-audit workbenches must use the single canonical HTML shell:

`tests/vector/review_annotation_template.html`

Do not create an ad-hoc reviewer, duplicate the template, embed a new bespoke
HTML page in a generator, or use a one-off visualization as a replacement for
the canonical workbench. Different datasets and review queues must be expressed
through `summary.review_ui.protocol`, `summary.review_ui.template_ids`,
`summary.review_ui.template_overrides`, and
`summary.review_ui.diagnostic_overlay`, with the complete payload preserved in
the canonical sidecar when needed.

Every visual mapping view must show the real source and target screenshots and
must label source, gold, and method prediction explicitly. Overlay coordinates
must declare their coordinate space and be transformed through the actual
rendered image rectangle; normalized coordinates must never be interpreted as
image pixels. If the canonical shell is missing or broken, restore it before
producing review output. Do not work around its absence by creating another
viewer.

# Unified Matching Dataset Long-Term Rule

All reviewed correspondences must already use
`omnitransfer.ui_correspondence_pair.v1` before entering this repository.
Do not add source-specific import adapters, trainers, objectives, or model paths.

Split inside each canonical App by page-connected components. The same App
should appear in train, dev, and test when it has enough independent
components; exact pair, page, and component identities must never cross
splits. App-disjoint evaluation is an additional generalization slice, not the
primary split. `view_str`, `origin_id`, `state_str`, and `structure_str` may be
used only by offline self-supervised label construction and must never enter
matcher model inputs. Self-supervised proposals may enter only the training
split or a page-disjoint held-out self-supervised split. Their
`label_status=self_supervised` must be preserved so automatic validation is
never reported as reviewed-gold evaluation. Other non-gold dev/test assignments
remain review candidates. Unmatched nodes remain ignored unless a human
explicitly labels NULL.

# Unified iOS/Android XML Matching Long-Term Rule

The core model input is one iOS XML tree, one Android XML tree, and a source XML
node. Both trees are converted by the same platform-neutral converter into the
same graph schema. Existing dataset source-node/target-node pairs are the only
supervision; the converter must not rewrite labels, mark labeled nodes as
clickable, lift labels to ancestors, or create canonical actionable groups.

Every XML node is preserved as model evidence and as a possible output node.
TextView, ImageView, containers, and other non-actionable nodes must not be
filtered. A training page may not be skipped because it has too few actionable
correspondences. Training, validation, and inference use the same all-node
candidate policy inside the matcher, not an evaluation harness.

The learned node representation uses text/content description, class and action
state, visual crop, XML hierarchy, and normalized within-page spatial relations.
Raw absolute device coordinates and page order are not model inputs. The only
trainable architecture is geometric-v9 with the all-node candidate policy.

# Candidate-Only Runtime Boundary

OmniTransfer is a policy-free candidate generator. Its runtime API returns the
complete ranked target candidates, their target bounds and projected points,
scores, margin, and matcher provenance. It must not accept or reject a mapping,
select an action for execution, enforce page identity, decide fallback, or own a
runtime/evaluation harness. Those responsibilities belong to OmniFlow.

The only public runtime operation is `rank_action_candidates`. Do not add an
alias, compatibility wrapper, selected target, or selection policy. Invalid
inputs and unavailable matchers may be represented as an empty candidate
response with diagnostic status, but OmniTransfer must never manufacture or
pass through source-device coordinates.
