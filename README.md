# OmniTransfer

OmniTransfer maps one recorded source UI element to a complete ranked set of
target UI-element candidates on another device. It does not choose an action,
accept or reject a candidate, execute on a device, or own fallback policy.

```text
source XML + source node/point + target XML
    -> ranked target nodes/points + evidence
```

## Public Runtime

The only public runtime operation is `rank_action_candidates`.

```python
from omnitransfer import rank_action_candidates

result = rank_action_candidates(
    source_xml=source_xml,
    target_xml=target_xml,
    source_point=(x, y),
    action_type="click",
    top_k=5,
)
```

Callers may provide `source_element_id` instead of `source_point`. Every result
contains the full target ranking, projected target coordinates, candidate
bounds, pair confidence, rank probability, margin, release id, feature-schema
hash, and checkpoint hash. An empty ranking is a transfer failure; OmniTransfer
never replays the source coordinate on the target device.

## Fixed Runtime Release

```text
release id                 omnitransfer-direct-text-alignment-v9.3.1-mobile
mapping mode               omnitransfer_direct_text_alignment_v9
architecture               direct text + v9 geometric + Spatial-XML alignment
training data              765 ASE 2023 reviewed-gold page pairs
development data           104 disjoint ASE 2023 reviewed-gold page pairs
parameters                 177,372 total; 7,978 alignment; 0 token-lookup
feature schema             omnitransfer-direct-text-spatial-xml-v9
feature schema SHA256      fc1706baeff6d8b0e43f5da9b03a62a51086472764d440ed7b4c3a41f71c52f8
NumPy checkpoint SHA256    b8a6735bd97a7163ad186ec4c869eebbb05634522472035bd9c3b9bc323c5e9e
```

The mobile runtime bundles and verifies the NumPy checkpoint. Runtime selection
policy stays outside OmniTransfer. Training checkpoints and experimental model
families are not runtime fallbacks.

## Repository Layout

```text
src/omnitransfer/
  runtime.py                 public candidate-ranking boundary
  numpy_v9_matcher.py        frozen NumPy inference
  learned_matcher.py         geometric-v9 features and checkpoint I/O
  geometric_matcher.py       single trainable model
  page_embedding.py          frozen page embedding
  ui_graph.py                UI graph contract
  mapping_dataset.py         correspondence dataset contract
  mapping_pair_review.py     canonical review renderer

scripts/
  build_ui_correspondence_dataset.py
  build_ui_correspondence_review.py
  build_utg_point_mapping_review.py
  train_geometric_v9_matcher.py
  evaluate_geometric_v9_matcher.py
  export_runtime_checkpoint.py
```

The supported script inventory is maintained in
[`scripts/ENTRYPOINTS.md`](scripts/ENTRYPOINTS.md). A script absent from that
file is internal, pending migration, or unsupported; do not build a second
entrypoint for the same function.

## Canonical Offline Workflow

Build one `omnitransfer.ui_correspondence_pair.v1` pool:

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py \
  --correspondence-pairs /absolute/reviewed_pairs.jsonl \
  --output-dir /absolute/dataset
```

Review source-to-target point mappings through the one review workbench:

```bash
PYTHONPATH=src python scripts/build_ui_correspondence_review.py \
  --input /absolute/dataset/pool.jsonl \
  --output-dir /absolute/review
```

Convert current UTG collections into the same review and export contract:

```bash
PYTHONPATH=src python scripts/build_utg_point_mapping_review.py \
  --batch legacy=/absolute/review-current \
  --output-root /absolute/point-mapping-review
```

Train and evaluate through one model path:

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

## Data Boundary

Raw datasets, screenshots, XML dumps, APKs, model runs, review output, and
human annotations are external assets and are never source-controlled. Their
exact hashes and roles belong in manifests. Repository cleanup must not delete
or rewrite those assets.

Android device exploration, DroidBot lifecycle, and source-format conversion are
external data responsibilities and are not implemented in this repository.

## Validation

```bash
python -m py_compile \
  src/omnitransfer/runtime.py \
  src/omnitransfer/numpy_v9_matcher.py \
  src/omnitransfer/learned_matcher.py \
  src/omnitransfer/page_embedding.py \
  scripts/build_ui_correspondence_dataset.py \
  scripts/build_ui_correspondence_review.py \
  scripts/train_geometric_v9_matcher.py \
  scripts/evaluate_geometric_v9_matcher.py

PYTHONPATH=src pytest -q \
  tests/test_public_api.py \
  tests/test_runtime.py \
  tests/test_mapping_training.py \
  tests/test_ui_correspondence_review.py
```

See [`CODEBASE_CLEANUP_README.md`](CODEBASE_CLEANUP_README.md) for the final
cleanup inventory and retained seams.
