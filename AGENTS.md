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

Normalize MobileViews and every admissible action-target correspondence source
into one `omnitransfer.ui_correspondence_pair.v1` pool before assigning splits.
Dataset origin never creates another trainer, objective, or model path.
GUIOdyssey remains outside these matcher splits.

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

# Shortcut-Free Actionable Matching Long-Term Rule

The learned node representation uses only the node's text/content description,
class and action state, and visual crop. Raw resource ids, absolute coordinates,
absolute box size, and page order are not model inputs. Geometry may appear
only as within-page relative relations between selected node tokens.

Training supervision is actionable-node to actionable-node only. Non-actionable
text and icon nodes remain attention context but are never relabeled or lifted
to an actionable ancestor. Every retained same-screen actionable node is a
candidate hard negative. Ambiguous evidence or a low calibrated ranking margin
is transfer failure and returns control to the VLM.
