# Evaluation

## Primary OmniTransfer Dataset

OmniTransfer is built from one mixed `omnitransfer.ui_correspondence_pair.v1`
pool.
Dataset adapters run before splitting and never create dataset-specific model
paths. The canonical splitter assigns page-connected components inside each
App, so the same App may appear across train/dev/test while pair, page, and
component overlap remains zero. Page-disjoint MobileViews proposals may occupy
dev/test with `label_status=self_supervised`; those splits support fast
development comparisons but are not formal metrics. Formal paper metrics use
only reviewed or original `label_status=gold` records.

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py \
  --correspondence-pairs runtime/datasets/mobileviews/pairs.jsonl \
  --output-dir runtime/datasets/unified_mapping
```

The primary result measures transfer to unseen pages, states, layouts, and
devices inside known Apps. App-disjoint evaluation is reported separately as a
harder generalization slice.

## Legacy ASE Cross-Platform Baseline

Use the public `RuihuaJi/vision-based-widget-mapping` dataset only after binding
the public boxes to real XML nodes and converting the screen hierarchies to the
relative-XML contract.

Canonical clean path:

```text
runtime/evals/vision_widget_mapping/clean_relative_xml_v1/
```

Build it with:

```bash
python scripts/clean_widget_mapping_dataset.py \
  --input runtime/evals/vision_widget_mapping/testset.txt \
  --output runtime/evals/vision_widget_mapping/clean_relative_xml_v1
```

The output contains:

```text
queries.jsonl   source-node -> target-node correspondence rows
mappings.jsonl  canonical lightweight mapping edges; no repeated candidate list
screens/        one normalized XML hierarchy per unique screen
screens.jsonl   screen paths, dimensions, and candidate counts
audit.json      binding, contamination, coordinate, and split gates
manifest.json   source and output hashes
```

`mappings.jsonl` plus `screens/` is the canonical cleaned dataset.
`queries.jsonl` is a derived compatibility view for the current learned-matcher
training loader and repeats the fixed target candidate list for each query.

Use `scripts/evaluate_clean_widget_mapping.py` for deterministic baseline
evaluation on this dataset. It reads the frozen split from each query, ranks
only the declared target candidates, and scores against the set-valued
equivalent gold node ids. It does not call the production transfer API or fall
back to source coordinates.

Every structural `bounds` value is relative to the source XML viewport and lies
in `[0,1]`. Every `visual-bbox` is relative to the screenshot. Public annotation
boxes are used only to bind source and gold XML nodes; they never become target
candidates or replacement semantics. A binding failure aborts the build instead
of dropping a row.

Required audit gates:

```text
queries = 6,730
source binding = 100%
target binding = 100%
app-name semantic fallbacks = 0
public pseudo-candidates = 0
duplicate candidate ids = 0
gold missing from candidates = 0
relative bbox out of range = 0
app overlap across train/dev/test = 0
```

The old file below is retained only as a historical contaminated artifact and
must not be used for new training or headline evaluation:

```text
runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.jsonl
```

Expected clean summary:

```text
mappings / queries = 6,730
unique screens = 2,104
unique screen pairs = 1,052
fixed visible target XML candidates = 51,348
mean target candidates per screen = 48.81
train / dev / test = 4,124 / 1,014 / 1,592
```

## Diagnostic Dataset

TEMdroid/SemFinder is retained as metadata-only diagnostic data. It does not
include the same screenshot/XML grounding richness and should not be used as the
main visual/layout robustness dataset.

## Canonical UI Correspondence Evaluation

Every learned matcher result uses the same
`omnitransfer.ui_correspondence_pair.v1` adapter as training. Raw graph files are
not a second evaluation format.

```bash
PYTHONPATH=src:. python scripts/evaluate_relation_aware_matcher.py \
  --input runtime/datasets/unified_mapping/test.jsonl \
  --split test \
  --checkpoint runtime/models/relation_matcher.pt \
  --device cuda \
  --output runtime/reports/page_pair_test.json
```

Local path:

```text
runtime/evals/temdroid_semfinder_action_transfer_compare_20260615/
```

## Metrics

The main metric is Top1 replay target grounding accuracy:

```text
Top1 = count(argmax_i score_i == gold) / number_of_queries
```

Safety metrics are equally important for record-and-replay:

```text
false_positive_rate = concrete target predicted when gold is NULL
abstain_accuracy = NULL predicted when target is absent or ambiguous
wrong_target_rate = concrete target predicted but candidate is not gold
```

Latency is measured after model-only warmup with one query per decoded target
image and a prepared recorded-source image. Reports include graph construction,
tensor transfer plus batched visual sampling, synchronized model execution,
postprocessing, p50, p95, max, and the under-50ms rate. PNG decode/resize is
reported separately as asset preparation because production consumes captured
pixels. Runtime promotion requires every compute sample in the fixed RTX 4090
suite to remain below 50 ms.

## Baseline Families

- fixed coordinate replay
- text/resource/class feature scorer
- raw UI-tree exact replay
- UI-tree plus anchor replay
- OmniFlow anchor transfer (`get_new_coordinates`)
- paper IH-layout approximation
- legacy action-transfer structured ranker
- relation-aware cross-attention matcher
- attention-based grounding baseline
- LoRA bbox-generation baseline
- two-layer source-conditioned grounder

## Legacy Structured Baseline

The paper-facing structured matcher is exposed as:

```text
action-transfer
```

It is the recorded structured baseline over pair, graph, listwise, and coordinate
prior evidence. Historical artifacts may still show the internal backend name
`spatial_fusion_gbdt`; that name is a legacy implementation id. It must not be
presented as the new learned cross-attention method.

## Learned Core

The proposed core is the relation-aware cross-attention matcher described in
`docs/relation_aware_cross_attention_matcher.md`. Benchmark V2 accepts equivalent gold
candidate sets and NULL targets, and reports ranking, abstention, safety, and
latency metrics. The learned core is promoted to runtime only after the recorded
accuracy, safety, and latency gates pass.

The dataset roles and modern Android, web, WebView, DOM/AX, and frozen-eval
sources are versioned in `datasets/unified_ui_registry.v1.json`. Outcome logs
from train apps may add verified winner-loser preferences; environment failures
and formal evaluation episodes are excluded.

## OmniFlow Anchor Baseline

The benchmark exposes the production OmniFlow anchor-transfer adapter as:

```text
omniflow_anchor_transfer
```

Legacy aliases remain accepted for old artifacts:

```text
omniflow
ours
omniflow_action_transfer
```

Recommended local comparison:

```bash
PYTHONPATH=.:src python tests/vector/benchmarks/action_transfer_rich_eval.py \
  --dataset-id vision_widget_mapping \
  --matcher-suite paper_with_omniflow_anchor \
  --eval-split test \
  --output runtime/evals/vision_widget_mapping/rich_eval.omniflow_anchor_fulltest.json
```

This baseline evaluates `get_new_coordinates` as a matcher-only source-to-target
grounding component. It is not a full OmniFlow replay-pipeline result.

## Reporting Rule

This outline repository does not promote any historical diagnostic number as a
new headline result. Existing result files are preserved as local artifacts and
listed in `docs/archive_manifest.md`; future reports should cite the exact
artifact, dataset split, matcher name, and latency setting.
