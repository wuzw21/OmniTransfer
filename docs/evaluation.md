# Evaluation

Training and evaluation consume only
`omnitransfer.ui_correspondence_pair.v1`. Exact pair, page, and page-component
identities must not overlap across train, dev, and test. Human-reviewed rows are
the only source for checkpoint selection and headline metrics.

## Build

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py \
  --correspondence-pairs /absolute/reviewed_pairs.jsonl \
  --split-seed 17 \
  --train-percent 80 \
  --dev-percent 10 \
  --output-dir /absolute/dataset
```

The builder writes `pool.jsonl`, split JSONL files, and one manifest containing
input hashes, label counts, split counts, and leakage checks.

## Evaluate

```bash
PYTHONPATH=src:. python scripts/evaluate_geometric_v9_matcher.py \
  --input /absolute/dataset/test.jsonl \
  --split test \
  --checkpoint /absolute/model.pt \
  --device cuda \
  --output /absolute/report.json
```

The evaluator uses the same canonical loader and all-node candidate contract as
training. Report Top-1, Recall@3, Recall@5, wrong-target rate, warmed model
latency, end-to-end latency, checkpoint SHA-256, dataset hashes, code revision,
hardware, and timing method. Removed baselines and dataset-specific evaluators
must not be reintroduced as top-level scripts.
