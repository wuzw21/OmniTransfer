# Relation-Aware Cross-Attention Matcher

## Decision

The next OmniTransfer core is a small learned matcher over two structured UI
graphs. It is not a weighted rule formula, a resource-id lookup, an occurrence
rule, or coordinate passthrough.

The paper-facing statement is:

> OmniTransfer learns contextual widget descriptors from local UI topology and
> relative layout, conditions source and target descriptors on each other with
> bidirectional cross-attention, and predicts a partial target assignment that
> includes a learnable NULL outcome.

The runtime safety contract is unchanged: an absent, ambiguous, low-confidence,
or failed mapping is a transfer failure. The caller can then invoke its VLM
fallback. Source-device coordinates must never be replayed as a silent mapping.

## Why The Previous Neural Diagnostic Is Insufficient

The local July 2026 artifacts provide a useful negative result:

| Matcher | Test queries | Top1 | Mean latency | p95 latency |
|---|---:|---:|---:|---:|
| previous structured attention, CE | 1,661 | 0.6689 | 8.50 ms | 15.63 ms |
| previous structured attention, pairwise auxiliary | 1,661 | 0.6749 | 8.42 ms | 15.40 ms |
| spatial structured ranker | 1,661 | 0.7489 | 0.14 ms | 0.23 ms |
| strongest recorded structured pipeline | 1,661 | 0.7652 | 15.16 ms | 21.37 ms |

The previous neural model attends only among target candidate feature rows and
optionally to history rows. Its source information has already been collapsed
into deterministic pair features. Naming that path "cross attention" does not
make it a source-graph to target-graph matcher.

The same-screen pretraining prototype had four additional problems:

1. text, content description, resource id, and class were each reduced to one
   fixed hash scalar;
2. positive identity was preserved across weak views, making the task too easy;
3. dropped nodes were omitted instead of supervising a NULL assignment;
4. the resulting checkpoint was not used by the deployed matcher.

The new implementation addresses these points in
`src/omnitransfer/learned_matcher.py` and
`src/omnitransfer/self_supervised.py`.

## Architecture

### 1. Shared Multimodal UI Tokens

For node `i`, the input contains:

```text
text/content-desc/resource-id/class/context subword buckets
role and action flags
normalized bbox, center, size, and area
tree depth and local degree
one shared 32x32 node-crop CNN feature when screenshots exist
```

Hashing is used only to assign sparse subword buckets. The bucket embedding and
all evidence-composition weights are learned. This differs from turning each
string into a fixed scalar similarity. A learned missing-visual token is used
when a screenshot or bbox is absent, so structured-only inputs remain on the
same encoder path rather than switching to a fallback matcher.

### 2. Learned Local And Position Relations

For every pair of nodes on the same screen, the relation input contains:

```text
parent / child / sibling / ancestor / descendant
same-row / same-column / overlap / local-neighbor
relative dx / dy
relative width / height
IoU and tree distance
```

These values are not summed with hand-written weights. A relation MLP produces
per-head attention biases:

```text
A_ij = softmax(q_i k_j / sqrt(d) + MLP_rel(r_ij))
```

This is the precise mechanism behind the claim that the model learns local
relationships and positional relationships.

### 3. Bidirectional Source-Target Cross-Attention

Each layer alternates:

```text
relation-biased self-attention on source nodes
relation-biased self-attention on target nodes
source queries attending to target keys/values
target queries attending to source keys/values
```

Two layers, hidden size 96, and four heads are the default. The implementation
stays below 1.5M trainable parameters, including the small jointly trained crop
CNN. It does not require OCR, BERT, CLIP, a VLM, or a platform-specific visual
branch.

### 4. Partial Assignment Without A Bijection Assumption

GUI mapping is not necessarily one-to-one. A clickable wrapper and its visible
label can be equivalent replay targets, a source widget can disappear, and
platforms can split or merge controls. Therefore the core predicts independent
source-to-target categorical distributions with a learnable NULL column:

```text
p(t_j | s_i, G_s, G_t), j in target candidates union {NULL}
```

The supervised loss accepts a set of equivalent gold candidates:

```text
L_set = -log sum_{j in Gold(s_i)} p(t_j | s_i)
```

For self-supervised graph pairs, both directions are trained and a soft cycle
consistency term encourages reciprocal evidence. Sinkhorn is not the default
because its near-bijection prior is a poor fit for wrapper/label equivalence and
one-to-NULL transfer.

Explicit cross-screen labels use true partial assignment. A listed source-target
node pair is positive; nodes omitted by a weak annotation are ignored rather
than treated as NULL. NULL is supervised only by an explicit human
`no_correspondence` label or by a transformation whose dropped/distractor
identity is known. This distinction prevents sparse GUIOdyssey action labels
from teaching universal abstention.

## Unified Mapping Dataset

Training and formal evaluation use the identical page-level record contract
`omnitransfer.mapping_page_pair.v1`. A record contains one source UI graph, one
target UI graph, and multiple source-node assignments. Each source node maps to
a set of equivalent target nodes or to an explicit NULL (`no_correspondence`).
The schema therefore does not assume that a page pair contains only one click
or that correspondence is one-to-one.

`scripts/build_unified_mapping_dataset.py` writes:

```text
train.jsonl
dev.jsonl
test.jsonl
diagnostic.jsonl
review_dev_candidates.jsonl
review_test_candidates.jsonl
manifest.json
```

The distinction between files is supervision quality, not input shape. ASE
2023 public mappings are gold and retain their authored app-disjoint split.
GUIOdyssey trajectory alignment is weak supervision: its training partition may
enter `train.jsonl`, while its held-out partitions enter `diagnostic.jsonl` and
are frozen as `unreviewed` human-review candidates. Only exported rows with
`annotation.status=reviewed` and a valid `correspondence` or
`no_correspondence` label may replace a weak held-out row in formal `dev.jsonl`
or `test.jsonl`.

Every GUIOdyssey diagnostic row records `provenance.reserved_split=dev|test`.
The leakage audit uses that frozen assignment when checking task, episode,
trajectory, page, and pair isolation; changing a file name cannot move a sample
between partitions. The manifest records row counts, match counts, byte sizes,
and SHA-256 for every emitted file. Writes are atomic.

Until reviewed GUIOdyssey exports exist, the formal dev/test files contain ASE
gold only. GUIOdyssey held-out results remain diagnostic and cannot be reported
as benchmark accuracy.

## Mixed Gold And Weak Training

`scripts/train_mixed_generalization_matcher.py` uses one model, one optimizer,
and the same bidirectional assignment head for both sources. Every ASE gold
optimizer step includes the mean loss of a deterministic shard of GUIOdyssey
weak pairs, scaled by `correspondence_weight`, before one backward pass. Weak
pairs therefore cannot dominate merely because they outnumber gold queries,
and there is no rule/selector side path.

ASE uses its authored app-disjoint `train/dev/test` field without rehashing.
GUIOdyssey is split by connected meta-task, episode, and trajectory groups;
all three identifiers have zero train/test overlap. Device-pair distributions
are stratified and reported as slices because the current six-device graph is
fully connected and cannot support a non-empty device-disjoint split.

Held-out trajectory alignment remains a pseudo-label diagnostic. Only exported
rows with `annotation.status=reviewed` and labels `correspondence` or
`no_correspondence` are formal GUIOdyssey gold; `uncertain`, unusable, draft,
and unreviewed rows are excluded.

### Mixed-Training Baseline (2026-07-22)

The frozen three-epoch baseline continues the first full epoch for two more
epochs at learning rate `1e-4`, with `correspondence_weight=0.35`. It trains
821,015 parameters on 4,124 ASE gold training queries and 24,239 accepted
GUIOdyssey weak trajectory pairs. The authored ASE test split contains 1,592
queries and shares no app split with training. The GUIOdyssey held-out split
contains 5,377 accepted weak pairs and has zero meta-task, episode, and
trajectory overlap with its training split.

| Frozen checkpoint | ASE Top1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| mixed epoch 1 | 69.85% | 86.37% | 91.14% | 79.24% |
| mixed epoch 3 | 76.13% | 89.57% | 93.84% | 83.87% |
| rule/anchor baseline | 36.06% | - | 59.48% | 47.49% |
| selector baseline | 37.19% | - | 40.20% | 39.11% |

The independent frozen evaluator exactly reproduces the epoch-3 accuracy. The
GUIOdyssey weak held-out diagnostic is 100% Top1, but it is deliberately not a
headline result: these pseudo-graphs contain an easily identified action target
and do not replace human correspondence gold. There is currently no exported
reviewed GUIOdyssey JSONL, so formal human-gold accuracy and NULL rejection are
unavailable.

On a shared RTX 4090, the independent epoch-3 frozen run measures 7.66 ms p95
model time but 54.10 ms p95 end-to-end new-image time; 93 of 100 images finish
under 50 ms and the maximum is 69.64 ms. A second in-training measurement gives
42.42 ms p95 and 97 of 100 under 50 ms. The strict 100-of-100 latency gate is
therefore not met and must be rerun on an isolated GPU before architectural
changes are justified.

## Self-Supervised Training

### Pretraining Data

Recommended order:

1. MobileViews view hierarchies for scale and diverse Android layout structure;
2. Uni-GUI-OpenMobile for newer AndroidWorld screens, open-source apps, and
   trajectory-level UI element metadata;
3. OS-Atlas AndroidWorld grounding for an independently collected modern
   Android source;
4. WebUI 70K for domain-disjoint web and viewport diversity;
5. AndroidControl accessibility forests for broad modern app and task coverage;
6. RICO for an independent mobile UI domain;
7. validated OmniTransfer/AndroidWorld observations for in-domain structure,
   never using formal test episodes as training data;
8. public widget-mapping train splits for supervised fine-tuning.

MobileViews is the primary first source. The official MobileViews-600K release
provides more than 600,000 globally image-deduplicated screenshot/view-
hierarchy pairs from more than 20,000 Google Play apps. The first official
parquet shard contains 150,001 rows, is 25,922,626,062 bytes, and has SHA-256
`440ceb895a4819de6e929fb39f7b8b6f869068f24c2862a92f876ddb687cca42`.
The exact downloaded release and shard digest must be pinned in each
experiment. RICO, UIBert, ActionBERT, and Screen2Vec motivate UI representation
pretraining but are not same-task baselines.

`scripts/import_mobileviews.py` streams the official `image_content` and
`json_content` columns, writes pre-resized lossless 384px PNGs, and assigns
whole packages—not individual screens—to immutable train/dev/test files. Rows
with unknown package or screenshot/tree geometry mismatch are excluded and
counted in the manifest. DroidBot `visible=false` off-screen scroll content is
removed before augmentation, with visible descendants reconnected to their
nearest visible ancestor. Package, stable ids, and source coordinates are never
matcher inputs.

The pinned Uni-GUI-OpenMobile snapshot at revision
`774eb20f97724bf64faa9e66b1572f05b1688e38` contains 2,640 trajectories and
about 25.9K steps from 19 open-source Android apps. Its audited repository has
34,344 PNGs and 7,193 JSON files (9,384,688,170 bytes). Each metadata step
provides screen geometry plus bounded UI element attributes. The adapter forms
containment edges before augmentation, removes repeated screenshots by SHA-256,
and freezes whole app packages into one split. Reviewed action boxes are kept
only as optional labels; the inference path remains the same matcher.

The pinned `biglab/webui-70k-elements` release adds 173,546 train screenshot
rows and semantic element boxes in 11,396,715,289 bytes. Its converted parquet
commit is `8b74f86a8418b4578ad85e73eaba3d6e7280ed9d`. The original WebUI collection
groups pages by domain before its 70/10/20 split; the separate official
validation and test releases remain frozen. The adapter clips elements to the
visible viewport and creates flat graph nodes, allowing pairwise learned
position relations without inventing a DOM hierarchy or a rule matcher.

The pinned OS-Atlas release at revision
`e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f` contributes an independently
collected AndroidWorld subset with 15,905 unique screenshot names and 89,860
bounded elements. The selected annotation and image files are 31,860,297 and
2,325,202,926 bytes with SHA-256 digests
`8c53594c8e21975b7ba498824b687297f2f7bc43460a07b9c959f63b674a38b0` and
`d88fe06b2cb6a8db149cd0c9a2c1c03b9c61599bf467bfc002fea25fa2841d93`.
The full upstream repository is about 816 GB and is never cloned with LFS
smudging. Because the released annotations omit app identity, its deterministic
image-level dev/test files are diagnostics, not evidence of cross-app
generalization. Element identities supervise only independent train-time views.

AndroidControl contributes 15,283 demonstrations over 833 apps and 40 app
categories. Unlike coordinate-only trajectory corpora, every observation has a
screenshot and serialized Android accessibility forest with true child edges,
bounded nodes, text, content descriptions, class names, resource names, and
interaction state. `scripts/import_android_control.py` parses the GZIP TFRecord
and protobuf wire formats without TensorFlow, resizes each screen once, and
writes atomic per-shard graph bundles while streaming directly from the public
objects. Thus the 49,930,448,595-byte raw release does not occupy persistent
9207 storage. The official 13,603/137/1,543 train/validation/test episode split
and published IDD, task-unseen, app-unseen, and category-unseen test subsets are
preserved. Goals, step instructions, and actions remain label-only metadata;
they cannot become a coordinate or policy shortcut in the matcher.

No audited public mobile GUI source currently provides the foldable mapping
unit needed here: the same logical screen paired across folded, half-opened,
unfolded, and rear-display postures with screenshots and UI trees. MobileViews,
OpenMobile, and the sampled AndroidControl records use fixed phone canvases;
GUI-Odyssey does not publish posture-paired UI trees. Aspect-ratio or layout
jitter must not be described as foldable data. The planned `FoldTransfer`
benchmark will collect versioned Pixel Fold emulator observations for CLOSED,
HALF_OPENED, OPENED, and REAR_DISPLAY_MODE, including hinge geometry and
WebView cases. Stable node identity is label-only, held-out apps/tasks are
frozen, and every posture uses the same learned matcher path.

Layout augmentation keeps two distinct signals aligned: transformed boxes feed
the numeric and relation encoder, while each local visual crop remains attached
to its latent element. This simulates layout reflow without cropping an unrelated
region and ensures the compact CNN is jointly trained on valid local appearance.
Independent brightness, contrast, and channel perturbations are applied to the
two train-time crop sets, preventing exact pixels from becoming a shortcut;
these transforms are absent from inference and add no runtime branch.

### Constructed Views

Each unlabeled graph produces two views with known latent identity. The view
generator performs:

```text
attribute masking
node and edge dropout
global scale and translation
per-node bbox perturbation
independent local brightness, contrast, and channel perturbation
node-order permutation
duplicate-looking distractor injection
```

Dropped nodes and injected distractors explicitly supervise NULL. Duplicate
text/class controls on the same screen are hard negatives. Later extensions
should add wrapper split/merge and list reordering to generate many-to-one gold
sets, rather than forcing an artificial bijection.

### Curriculum

1. self-supervised structure pretraining;
2. model-mined same-screen hard negatives;
3. supervised fine-tuning on app-disjoint public widget pairs;
4. optional fine-tuning on validated successful transfer pairs;
5. confidence calibration on dev only.

The first sequential WebUI pilot is retained as a negative control. Continuing
the MobileViews checkpoint for one epoch on 18,602 WebUI screens reduced frozen
MobileViews dev positive Top1 from 95.40% to 89.12% and NULL recall from 92.49%
to 82.47%; test changed from 95.11%/91.01% to 88.89%/80.35%. Consequently,
single-domain continuation is not the training recipe. Every subsequent run
mixes rehearsal graphs from all earlier domains and reports each frozen domain
separately.

Test apps, formal AndroidWorld episodes, and future target versions must not be
used for mining or calibration.

### Verified Outcome Preference

Full PPO is unnecessary for a one-step target decision. Train-app-only attempts
with an official validator become pairwise preferences on the same logits:

```text
verified success > verified failure
verified success > NULL
NULL > verified failure
```

The added loss is `-logsigmoid(logit_win - logit_lose)`. Environment failures
and contradictory candidate outcomes are excluded. There is no second policy,
reward-model branch, or inference-time rescue path.

## Benchmark V2

### Tracks

| Track | Data | Purpose |
|---|---|---|
| Cross-platform positive mapping | Vision-Based Widget Mapping | iOS-to-Android candidate ranking; 6,730 mappings across 50 apps in the local canonical import |
| Cross-app/version event matching | TEMdroid/SemFinder | semantic and contextual hard negatives; closest classical event-matching task |
| Structural robustness | MobileViews/RICO held-out apps | controlled attribute, topology, layout, duplicate, split/merge, and NULL perturbations |
| Modern web and WebView | Mind2Web, WebLINX, ReproBreak, generated WebSight renders | DOM/AX, viewport/theme/locale/CSS reflow, locator changes, and native-hosted WebView |
| Selective transfer safety | constructed plus reviewed no-match cases | false positives, abstention, and risk-coverage |
| End-to-end replay | AndroidWorld | official validator success after deterministic preflight |

The existing Vision-Based Widget Mapping import has no NULL examples and 6,146
of 6,730 rows contain duplicate candidate signatures. Benchmark V2 therefore
must report wrapper/label equivalence sets and add a separately versioned NULL
track. It must not relabel the original test set in place.

### Splits

Use immutable app-disjoint train/dev/test manifests. The current benchmark's
app-disjoint split is the right principle, but the manifest should be written
once and versioned so later hyperparameter searches cannot change test app
composition. Model initialization seed and split seed are separate CLI values;
multi-seed runs keep `split_seed=17` frozen.

For pretraining corpora, split by app/package or trace family. Never split
screens from the same app independently across train and test.

### Metrics

Primary:

```text
set-valued Top1
Recall@1/3/5
MRR
```

Safety:

```text
NULL accuracy
false-positive rate on NULL
wrong-target rate
coverage and selective accuracy
risk-coverage curve / AURC
```

Efficiency:

```text
parameter count and checkpoint size
CPU and device p50/p95/max latency
decoded-image under-50ms rate, asset preparation, and stage breakdown
peak memory
early-exit and pruning rates, when enabled
```

End-to-end AndroidWorld success is reported only with the official validator.
Environment failures are repaired and rerun with the same frozen seed and task
parameters; they are not method failures.

### Rule-Resistant Disagreement Slice

The benchmark includes a deliberately hard slice where the frozen learned
matcher and a frozen weighted selector disagree. The selector is used only to
mine and report cases; its output, component scores, and decision never become
matcher inputs or labels.

Raw scores from two model families are not subtracted because their scales are
not calibrated. `scripts/mine_matcher_disagreements.py` instead prioritizes:

1. learned Top1 is in the set-valued gold while selector Top1 is not;
2. the learned matcher ranks every acceptable target substantially above the
   selector's gold rank;
3. the selector is confidently wrong by its own within-model margin;
4. the learned NULL outcome prevents a selector false positive;
5. duplicate labels, large candidate sets, cross-form-factor screens, and
   WebView contexts increase diagnostic value when those metadata exist.

This slice measures the intended contribution: local topology and relative
position evidence resolving cases that fixed attribute/position weighting
cannot resolve. It is constructed from frozen predictions on a held-out split,
followed by human verification for weakly labeled sources such as GUIOdyssey.
Easy agreement cases remain in the full benchmark so the hard slice cannot
replace overall accuracy, NULL safety, and latency reporting.

When several hard anchors share the same source and target screenshots, the
miner also emits one page-pair record containing all source points, acceptable
target points, and matching edges. Training can score the page pair once and
supervise several independent rows of the assignment matrix. This is batched
many-anchor supervision, not a Sinkhorn one-to-one constraint; equivalent
targets and NULL remain valid for each source anchor.

## Experiment Matrix

Run the matrix in gates so the full campaign is spent only on viable variants.
All headline configurations use at least three seeds; finalists use five seeds
and paired bootstrap confidence intervals.

### Gate A: Data And Loss Sanity (12 runs)

```text
same-screen identity only
+ attribute masking
+ layout transform
+ node/edge dropout
+ duplicate distractors
+ NULL supervision
CE vs symmetric CE
+ cycle consistency
random negatives vs model-mined hard negatives
single gold vs set-valued gold
with and without app-disjoint split
leakage audit
```

### Gate B: Architecture (20 runs)

```text
pair MLP without attention
target self-attention only (previous diagnostic family)
true cross-attention without relation bias
relation self-attention without cross-attention
full relation-aware cross-attention
absolute bbox only vs relative position only vs both
no tree relations / no layout relations / no local context
hidden size 64 / 96 / 128
one / two / three layers
two / four / eight heads
shared vs unshared source-target weights
dot-product head vs learned pair head
NULL head ablation
```

### Gate C: Pretraining And Transfer (18 runs)

```text
random initialization
MobileViews 10k / 100k / full
RICO only
MobileViews plus RICO
freeze encoder vs full fine-tune
zero / one / three / eight fine-tune epochs
TEMdroid hard-negative curriculum
model-mined hard negatives round 1 / round 2
with and without validated successful-pair memory
```

### Gate D: Robustness And Safety (16 runs)

```text
text removed / changed / translated
resource id removed / renamed
class changed
layout translation / scale / reflow
duplicate labels
wrapper-label equivalence
list reordering
widget deletion (NULL)
extra distractor controls
cross-platform and cross-version slices
probability calibration
margin calibration
risk-coverage sweep
```

### Gate E: Deployment (10 runs)

```text
PyTorch CPU
TorchScript or torch.compile diagnostic
ONNX Runtime
candidate caps 32 / 64 / 128
one-layer early exit
learned confidence pruning
Android device latency
checkpoint quantization diagnostic
```

This is 76 named configurations before seed repetition. Only the best dev
configuration from each family is evaluated on the immutable test set.

## Promotion Gates

Do not replace the runtime matcher until all gates hold:

1. the exact 1,661-query app-held-out test is at least better than the recorded
   0.7652 structured pipeline, without test-set model selection;
2. NULL false-positive rate and coverage are reported from a versioned no-match
   track;
3. after model-only warmup and source/image decode preparation, every measured
   target image on one RTX 4090 is below 50 ms compute, with separate asset
   preparation and graph/input/model/postprocess breakdowns;
4. checkpoint loading, missing checkpoint, low confidence, and NULL all produce
   explicit transfer failure;
5. no branch replays the source coordinate when mapping fails;
6. end-to-end AndroidWorld evaluation passes deterministic preflight and uses the
   official validator.

Until then, existing structured and rule systems remain baselines, not the new
paper method.

## Related Work And Primary Sources

- SuperGlue, CVPR 2020: learned contextual descriptors, dustbin, and partial
  assignment. <https://arxiv.org/abs/1911.11763>
- LightGlue, ICCV 2023: alternating self/cross attention, matchability,
  confidence, pruning, and early exit. <https://arxiv.org/abs/2306.13643>
- LoFTR, CVPR 2021: descriptors conditioned on both inputs through self/cross
  attention. <https://arxiv.org/abs/2104.00680>
- MatchFormer, ACCV 2022: interleaved extraction and matching.
  <https://arxiv.org/abs/2203.09645>
- GlueStick, ICCV 2023: explicit structural connectivity in learned matching.
  <https://arxiv.org/abs/2304.02008>
- TEMdroid, ICSE 2024: contextual widget matching and two-stage hard-negative
  mining. <https://doi.org/10.1145/3597503.3623322>
- Vision-Based Widget Mapping, ASE 2023: public cross-platform widget-mapping
  evaluation. <https://doi.org/10.1109/ASE56229.2023.00068>
- UIBert, IJCAI 2021: generic multimodal UI representation pretraining.
  <https://doi.org/10.24963/ijcai.2021/235>
- ActionBERT, AAAI 2021: using interaction traces as UI representation signal.
  <https://doi.org/10.1609/aaai.v35i7.16741>
- Screen2Vec, CHI 2021: self-supervised screen and component embeddings.
  <https://doi.org/10.1145/3411764.3445049>
- MobileViews: million-scale diverse mobile GUI data.
  <https://arxiv.org/abs/2409.14337>

The name "nicro" did not resolve to a GUI matching paper in Crossref, OpenAlex,
GitHub, or web search. It should not be cited until the exact title or URL is
identified.
