# Supported Scripts

Only the following repository scripts are supported. Do not add a second entry
for an existing function or a source-format compatibility importer.

## Dataset

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py \
  --correspondence-pairs /absolute/pairs.jsonl \
  --output-dir /absolute/dataset
```

## Review

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_review.py \
  --input /absolute/dataset/pool.jsonl \
  --output-dir /absolute/review
```

Existing UTG evidence can be converted into the same review contract with
`build_utg_point_mapping_review.py`; `utg_mapping_candidates.py` is its private
helper. Both use `tests/vector/review_annotation_template.html`.

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
