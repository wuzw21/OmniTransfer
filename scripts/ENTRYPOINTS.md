# Supported Scripts

Only the following repository scripts are supported. Do not add a second entry
for an existing function or a source-format compatibility importer.

## Dataset

Raw iOS screen captures are normalized once through the shared adapter before
they enter any pair pool or review artifact:

```bash
PYTHONPATH=src python scripts/normalize_ios_screens.py \
  --input /absolute/raw-ios-screens.jsonl \
  --repo-root /absolute/capture-root \
  --output /absolute/canonical-ios-screens.jsonl
```

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py \
  --correspondence-pairs /absolute/pairs.jsonl \
  --output-dir /absolute/dataset
```

Canonical labels can be audited without changing any correspondence endpoint:

```bash
PYTHONPATH=src python scripts/clean_mapping_dataset.py \
  --input /absolute/dataset/train.jsonl \
  --output-dir /absolute/cleaned-train
```

This writes `cleaned.jsonl`, `quarantine.jsonl`, and a hash manifest. The
policy excludes clickability and parent-child equivalence.

## Review

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_review.py \
  --input /absolute/dataset/pool.jsonl \
  --output-dir /absolute/review
```

Existing UTG evidence can be converted into the same review contract with
`build_utg_point_mapping_review.py`; `utg_mapping_candidates.py` is its private
helper. Both use `tests/vector/review_annotation_template.html`.

Every trained checkpoint must also produce a categorized error audit and merge
representative failures into the same live review component:

```bash
PYTHONPATH=src python scripts/build_model_error_review.py \
  --dataset /absolute/dataset/test.jsonl \
  --predictions /absolute/run/test.predictions.jsonl \
  --train /absolute/dataset/train.jsonl \
  --existing-review-dir /absolute/previous-unified-review \
  --output /absolute/current-unified-review
```

## Train And Evaluate

```bash
PYTHONPATH=src python scripts/train_geometric_v9_matcher.py \
  --input /absolute/dataset/train.jsonl \
  --validation-input /absolute/dataset/dev.jsonl \
  --output /absolute/model.pt

PYTHONPATH=src:. python scripts/evaluate_geometric_v9_matcher.py \
  --input /absolute/dataset/test.jsonl \
  --split test \
  --checkpoint /absolute/model.pt \
  --output /absolute/report.json
```

Training supports only geometric-v9 with all XML nodes as candidates. A
pretrained checkpoint can only resume the same model exactly.

The hard-coded local-context branch is offline-only and exists to test whether
the four difficult data slices contain useful signal before changing training:

```bash
PYTHONPATH=src python scripts/evaluate_local_context_algorithm.py \
  --input /absolute/dataset/test.jsonl \
  --learned-predictions /absolute/run/test.predictions.jsonl \
  --output /absolute/local-context-report.json
```

## Page Configuration Embedding

The page readout is an option of the same frozen matcher, not a second encoder.
It runs the learned page-attention head after local-relation and contextual
refinement. The optional comparison emits cosine similarity without collapsing
multiple historical pages into an average vector.

```bash
PYTHONPATH=src python scripts/embed_page.py \
  --input /absolute/source.xml \
  --screenshot /absolute/source.png \
  --compare-input /absolute/current.xml \
  --compare-screenshot /absolute/current.png \
  --output /absolute/page-embedding.json
```

## Runtime Release

```bash
PYTHONPATH=src python scripts/export_runtime_checkpoint.py \
  --input /absolute/model.pt \
  --output /absolute/model.npz \
  --torch-output /absolute/direct-text-model.pt
```

This single command performs the optional direct-text migration and deterministic
NumPy export. Runtime candidate ranking is exposed only by
`omnitransfer.rank_action_candidates`.

## Text-less Icon Audit

```bash
PYTHONPATH=src python scripts/build_icon_hard_set.py \
  --input /absolute/canonical-pairs.jsonl \
  --output /absolute/icon-hard-set.jsonl \
  --minimum-score 2.0

PYTHONPATH=src python scripts/evaluate_mapping_baselines.py \
  --input /absolute/icon-hard-set.jsonl \
  --split diagnostic \
  --output /absolute/icon-identity-baselines.json
```

The hard-set builder preserves `omnitransfer.ui_correspondence_pair.v1` and
uses no model score or resource ID for selection. Review the resulting JSONL
through `build_ui_correspondence_review.py`; do not create another HTML shell.
