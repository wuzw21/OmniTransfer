# Text-less Icon Mapping: Research, Method, and Experiments

Date: 2026-08-19

## Problem statement

The target is cross-platform node mapping, not icon classification. Given one
source XML node, the mapper must rank every target XML node while preserving the
existing geometric-v9 architecture, screenshot evidence, XML hierarchy, and
within-page relations. The difficult slice is a small visual control whose text
and content description are empty. Resource IDs are identity shortcuts and are
not model inputs.

The observed failure was not evidence that CNNs cannot represent icons. It was
the combination of an invalid visual crop contract and a legacy visual encoder
whose global average pooling was poorly matched to tiny glyph discrimination.
The old model could therefore assign nearly identical visual vectors to visibly
different icons, while its final rank probability still appeared close to one.
Rank probability is a relative softmax over candidates; it is not calibrated
visual similarity.

## Related work and design implications

- [WebArena resources](https://github.com/web-arena-x/webarena/blob/main/resources/README.md)
  release 179 human trajectories as Playwright traces. A trace exposes concrete
  HTML and network traffic; the experiment traces additionally render the
  accessibility-tree observation, raw prediction, parsed action, and screenshot.
  This makes traversal-local IDs and transient DOM state observable, but does
  not make them stable semantic identities. The useful invariants are role,
  local hierarchy, sibling relationships, normalized layout, and rendered
  appearance.
- [WebArena](https://arxiv.org/abs/2307.13854) evaluates functional completion
  rather than exact action-sequence imitation. For mapping, the corresponding
  lesson is to rank the current target tree from current evidence instead of
  replaying a historical DOM/node ID.
- [MobileViews](https://arxiv.org/abs/2409.14337) contains more than 1.2M
  screenshot-view-hierarchy pairs from more than 30K Android apps, six screen
  resolutions, and complete GUI trajectories. Its paper explicitly treats
  screenshots and view hierarchies as complementary: pixels retain exact visual
  state and image components; VHs retain hierarchy, boxes, classes, and action
  properties. This supports the existing OmniTransfer multimodal design rather
  than a screenshot-only icon side path.
- [SeeClick and ScreenSpot](https://arxiv.org/abs/2401.10935) separate
  icon/widget grounding from text grounding and report that all evaluated
  models struggle more on icons/widgets. This is the right hard slice for the
  benchmark, but SeeClick solves language-to-coordinate grounding rather than
  cross-tree correspondence.
- [ScreenQA](https://github.com/google-research-datasets/screen_qa) describes
  icons with state-aware labels such as `on`/`off` and selected/unselected.
  Filled and outline variants should therefore remain related while retaining
  enough fill/edge evidence to distinguish state.
- [Rico](https://interactionmining.org/rico) remains useful for broad static UI
  coverage, but MobileViews reports that Rico has about 63K screens from 9K apps
  at one resolution. It is not used as a claimed formal result here because the
  raw Rico corpus is not present in the current workspace.

## What changes and what remains invariant

Observed dynamic evidence includes text values, badges, selected state,
viewport size, absolute pixel coordinates, scroll position, dialogs, ads,
network-loaded WebView content, and traversal-local node IDs. Stable evidence is
relative rather than absolute:

1. local icon shape and fill/outline state;
2. normalized position inside the page and parent;
3. parent/child and ancestor role path;
4. child ordinal, sibling count, and repeated row/column slot;
5. neighboring labels and controls;
6. action/class family after platform-neutral normalization.

Geometric-v9 already models all-node candidates and typed within-page relations.
This work changes the first visual descriptor and its crop contract; it does not
add an icon matcher, acceptance gate, coordinate passthrough, or alternate
runtime API.

## Root-cause evidence

1. The old diagnostic pool mixed XML coordinates with Retina screenshot pixel
   coordinates. In the audited 129-row pool, 111 source icon boxes were outside
   the interpreted graph bounds. Torch clipped those boxes to a border line and
   still marked them visually valid; NumPy ignored the explicit visual box.
2. After repairing the shared coordinate contract, the released v9 model rose
   from 18.60% to 51.84% Top-1 on the 272-direction text-less icon diagnostic
   slice. This isolates crop validity as the first major failure.
3. The legacy encoder is three convolutions followed by adaptive global average
   pooling. On audited small icons, distinct glyphs collapsed to cosine values
   around 0.998. Global averaging removes the spatial sign pattern needed to
   distinguish plus, share, save, status, and toolbar glyphs.
4. Replacing the encoder without retraining raised the icon diagnostic result to
   68.01% Top-1 / 84.56% Top-3, but reduced full ASE dev to 64.37% Top-1. The
   existing fusion and relation weights were trained against the old CNN output
   distribution. A raw swap is therefore not a valid release.

## Deterministic 48D visual descriptor

Every node keeps one 32x32 RGB patch and one 48D visual slot:

- 16D low-frequency background-relative luminance DCT, including foreground
  fill in the DC slot;
- 12D centered 2x2x3 gradient-orientation histogram;
- 8D horizontal and vertical foreground occupancy bands;
- 8D background-relative color mean/std, fill ratio, and edge density;
- 4D contrast, saturation, centeredness, and border cleanliness.

The descriptor is deterministic and inversion-robust. It preserves related
filled/outline states without making them identical, and separates different
shapes more strongly. The same implementation exists in Torch and NumPy. Both
now use the original screenshot dimensions to interpret explicit visual boxes,
the same canvas resize, and align-corners continuous bilinear bbox sampling.

The visual encoder has no trainable parameters. Training migrates every
non-visual geometric-v9 tensor and learns the existing descriptor fusion,
relations, and assignment head through the main mapping loss. A raw visual
contrastive loss is disabled for the deterministic encoder because it has no
trainable visual parameters; reporting it as optimization would be misleading.

## Hard-set construction

`scripts/build_icon_hard_set.py` deterministically selects canonical
`omnitransfer.ui_correspondence_pair.v1` rows. It never uses model scores or
resource IDs for selection. Every selected source must be text-less and
icon-sized. Per-icon reasons include tiny area, crowded icon page, WebView
ancestry, repeated sibling slot, same-row/column ambiguity, missing resource ID,
cross-platform semantic/class asymmetry, state change, target alternatives, and
large normalized displacement.

Current artifacts:

| Set | Page pairs | Icon matches | Important slices |
| --- | ---: | ---: | --- |
| Cross-platform diagnostic | 36 | 135 | 14 apps; 117 crowded; 86 tiny; 102 row/column |
| ASE dev reviewed-gold | 12 | 25 | 6 WebView; 21 crowded; 18 row/column |

The cross-platform queue is available at
`output/vision_widget_icon_hard_review/review.html`; the ASE gold queue is at
`output/ase_dev_icon_hard_review/review.html`. Both are generated from the one
canonical `tests/vector/review_annotation_template.html` shell. The first queue
contains exactly 135 tasks and the second exactly 25; no reviewer-side icon
filter silently drops hard-set IDs.

## Baselines and current evidence

Exact identity baselines use every XML node as a candidate and abstain when no
exact identity exists. They are offline evaluation only and are not runtime
fallbacks.

| Evaluation | Text exact Top-1 | Resource-ID exact Top-1 | Coverage |
| --- | ---: | ---: | ---: |
| ASE dev, 1,566 directional rows | 35.95% | 0.00% | text 48.02% |
| ASE dev hard icons, 52 directional rows | 0.00% | 0.00% | 0.00% |
| Cross-platform hard icons, 270 directional rows | 0.00% | 0.00% | text 0.37% |

The historical broken icon-pool result was 18.60% Top-1 / 29.84% Top-3 with
6.59% descriptor Top-1. The repaired deterministic descriptor alone reached
53.68% Top-1 / 70.22% Top-3 on the 272-direction diagnostic slice. These two
figures are diagnostic, not reviewed-gold claims; the current ASE dev hard set
is the held-out gold icon gate.

The migrated candidate is now available at
`output/icon-deterministic-full-a/candidate.pt` with its NumPy runtime export at
`output/icon-deterministic-full-a/candidate.npz`. It was trained for one epoch
on 839 reviewed ASE pairs after migrating 62 non-visual parameter tensors from
the released direct-text v9 checkpoint. On the 115-pair held-out ASE dev split
it reaches 72.80% Top-1 / 89.40% Top-3 / 93.04% Top-5. On the held-out ASE
hard-icon slice (52 directional gold rows), it reaches 76.92% / 96.15% /
98.08%, compared with the released CNN's 50.00% / 51.92% / 61.54% and the
untrained deterministic replacement's 67.31% / 80.77% / 84.62%.

Warm model latency for the migrated candidate is 11.73 ms P50 on full dev and
9.80 ms P50 on the hard-icon slice; end-to-end warm input latency is dominated
by screenshot/graph preparation at 587.16 ms and 910.30 ms P50 respectively.
The Torch/NumPy audit covered 27 valid A-to-B hard-icon source rows: Top-1 was
identical on 27/27, while the complete Top-5 list was identical on 25/27 and
the maximum per-candidate probability difference was 0.0415. The candidate is
therefore suitable for Top-1 mapping inspection, but the NumPy export is not
claimed as exact full-ranking parity until the low-energy visual descriptor
differences are resolved.

## Reproduction

```bash
PYTHONPATH=src python scripts/build_icon_hard_set.py \
  --input runtime/datasets/ase_human_gold_all_nodes_within_app_v1/dev.jsonl \
  --output output/ase_dev_icon_hard_set.jsonl \
  --minimum-score 2.0

PYTHONPATH=src python scripts/evaluate_mapping_baselines.py \
  --input output/ase_dev_icon_hard_set.jsonl \
  --split dev \
  --output output/ase_dev_icon_hard_identity_baselines.json

PYTHONPATH=src python scripts/build_ui_correspondence_review.py \
  --input output/ase_dev_icon_hard_set.jsonl \
  --output-dir output/ase_dev_icon_hard_review \
  --icon-only \
  --checkpoint /absolute/geometric-v9.npz \
  --serve
```

## Endpoint audit and relation-depth follow-up

The unified ASE test error audit now separates strict benchmark errors from
pending label conflicts. Of 1,286 directional test correspondences, the
confidence-adaptive decoder has 248 strict errors. Forty of those predict a
target with exactly the source semantics while the declared target is blank
(32) or has conflicting semantics (8). They remain errors until manual review;
the report exposes the resulting 80.72%-to-83.83% strict-to-best-case interval
instead of silently relabeling the benchmark.

This exposed a mismatch in cleaning schema v1: the manifest prohibited
parent-child equivalence, but exact semantics within a two-hop target family
were exempted from quarantine. Schema v2 compares only exact declared target
endpoints. On the 839-pair ASE training set it quarantines 239 of 5,380 mapping
rows, compared with 38 in v1. The 201 newly exposed rows are review candidates,
not automatically corrected labels.

A screenshot-complete v2 fine-tune was evaluated against the resumed
checkpoint on all 230 dev screenshots. The trained epoch reached 75.93% versus
the 76.31% resumed baseline, so checkpoint selection retained the original
weights. Quarantining conflicts is necessary for future supervision but simply
removing 4.44% of labels does not improve an already-trained model; reviewed
replacement endpoints or additional cross-layout mappings are required.

The relation architecture itself is useful. On held-out ASE dev, descriptor,
unary assignment, and the three shared association layers reach 63.92%,
65.96%, 72.86%, 75.67%, and 76.25% Top-1 respectively. The failure is fixed
depth on some rows, not relation reasoning as a whole. Selecting the sharper
of the final two layer distributions raises dev Top-1 to 76.76%; the same
frozen policy reaches 80.72% on test versus 80.02% at the final layer.

## MobileViews structural corpus

MobileViews is Android-only and therefore is not cross-platform gold. A
leakage-safe builder selects pages using text-less nodes with semantic local
relatives, comparable siblings, list structures, transient layers, and deep
hierarchy. Clickability, resource IDs, package names, graph IDs, and origin IDs
are not node-identity inputs.

The first materialized corpus contains 2,000 pages from 1,368 packages and
130,989 exact same-observation node identities. All selected pages contain
text-less local-context and sibling-hard-negative structure; 1,905 contain
lists and 311 contain popup/dialog evidence. An initial remote evaluation was
invalid because its ASE JSON retained workstation-absolute screenshot paths;
that run is excluded. After rematerializing all 230 dev screenshots on the GPU
host, the original checkpoint reached 76.31% and the 200-update
MobileViews-only pilot reached 68.58% Top-1. The pilot is therefore rejected:
same-screen structural identity cannot replace cross-layout supervision.
MobileViews remains auxiliary data, and any future mixed model must improve ASE
dev before test evaluation or runtime release.

The replacement state-transition corpus pairs different observations from the
same package/activity with a source-row gap of at most four. Resource IDs and
unique semantic keys create pseudo labels only and remain unavailable to the
model. The current v2 materialization contains 1,000 pairs from 438 packages
and 16,684 pseudo matches: 777 pairs change node count, 766 contain list
structure, 174 contain popup/dialog evidence, and none has identical XML. Its
labels are intentionally `unreviewed`; it must pass sampled review and be mixed
at controlled weight rather than treated as formal gold.
