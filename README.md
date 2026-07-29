# OmniTransfer

OmniTransfer is a lightweight research repository outline for replay-time UI
grounding in GUI record-and-replay systems.

The task is:

```text
input:  recorded source UI graph G_s, source element e_s, target replay graph G_t
output: corresponding target replay element e_t
```

It is not action prediction. The recorded trace or cached function already
decides the operation type; OmniTransfer only relocates the recorded target on
the current replay screen.

## Repository Layout

```text
OmniTransfer/
  docs/
    problem.md
    method.md
    evaluation.md
    related_work.md
    archive_manifest.md

  src/omnitransfer/
    schema.py
    importers.py
    features.py
    matchers.py
    eval.py
    reports.py

  scripts/
    import_dataset.py
    run_eval.py
    run_external_baseline.py
    summarize_results.py

  tests/
    test_schema.py
    test_outline_imports.py

  artifacts/
    manifests/
    summaries/
```

## Data Policy

Raw research files are kept on disk under:

```text
/Users/wuzewen/Projects/Omni/OmniTransfer/runtime/evals/
```

They are not tracked in this outline repo. See
`docs/archive_manifest.md` and `artifacts/manifests/runtime_paths.md` for the
current path inventory.

## One-Line Method

Record-and-replay fails when recorded coordinates or selectors no longer
identify the intended UI target. OmniTransfer treats replay as
source-conditioned GUI grounding: first find plausible target regions on the
current screen, then select the candidate corresponding to the recorded source
element and local context.

## Minimal Function Interface

The paper-facing component is `action-transfer`: it maps a recorded source UI
target to the corresponding target UI element on the current screen. It is not
an action predictor; the replay trace still provides the action type.

The minimal API should accept two UI XML trees plus the recorded source target:

```python
result = action_transfer(
    source_xml=source_xml,
    target_xml=target_xml,
    source_point=(x, y),
    action_type="click",
    top_k=1,
)
```

`source_point` is the recorded coordinate in the source screen. It is used to
locate the source element inside `source_xml`. If the caller already knows the
source element id, it can pass that instead:

`new_x` and `new_y` are always expressed in the target XML coordinate system.
OmniTransfer does not convert them to Android display pixels or screenshot
pixels; callers must normalize or dispatch coordinates at their own boundary.

```python
result = action_transfer(
    source_xml=source_xml,
    target_xml=target_xml,
    source_element_id="source-node-id",
    action_type="click",
    top_k=1,
)
```

Required inputs:

```text
source_xml                 recorded source screen XML
target_xml                 current target screen XML
source_point or source_element_id
```

Optional inputs:

```text
action_type                click / input / swipe; used as a compatibility hint
top_k                      number of ranked target candidates to return
history                    prior successful source-target mappings, if available
```

Return value:

```text
target_candidate_id        selected target UI element
target_bbox                target element bounds
target_center              click/input center point
score                      model score for the selected candidate
margin                     top1-top2 score gap
top_candidates             ranked candidates when top_k > 1
```

Only two XML trees are not enough by themselves, because the matcher also needs
to know which source element from the recorded screen should be relocated.

## Smoke Check

```bash
python -m py_compile \
  src/omnitransfer/*.py \
  scripts/import_mobileviews.py \
  scripts/build_unified_mapping_dataset.py \
  scripts/train_mapping_page_pairs.py \
  scripts/benchmark_matcher_latency.py \
  scripts/import_dataset.py \
  scripts/run_eval.py \
  scripts/run_external_baseline.py \
  scripts/summarize_results.py \
  tests/test_schema.py \
  tests/test_outline_imports.py

PYTHONPATH=src python -m pytest tests/test_schema.py tests/test_outline_imports.py
```

## Self-Supervised UI Graph Pretraining

For MobileViews/RICO-style unlabeled UI hierarchies, OmniTransfer can construct
training pairs without manual source-target widget labels:

```text
raw UI graph G
  -> topology/layout/attribute view G_a
  -> independently perturbed view G_b with hard distractors
  -> bidirectional actionable-node ranking
```

The transformation is known because the augmentation function creates both
views and preserves latent `origin_id` only for label construction. The matcher
does not receive `origin_id`. It learns from structured UI fields and relations:

```text
mask text/content-desc/class
drop nodes and graph edges while ignoring unmatched rows
global and local layout perturbation
shuffle node order
inject duplicate-looking same-screen hard negatives
jointly train the lightweight screenshot crop encoder
learn tree/local/relative-position attention bias
apply bidirectional source-target cross-attention
```

The 64D node state uses only text/content description, class/action state, and
the visual crop. Raw resource id and absolute position/size are not model
inputs. All retained nodes participate in attention, but labels and ranking
candidates are strictly actionable-node to actionable-node.

Materialize an official MobileViews parquet shard once. The importer reads only
`image_content/json_content`, pre-resizes screenshots losslessly to the same
384px visual canvas used at runtime, recovers the package from DroidBot
metadata, and writes immutable package-disjoint train/dev/test graph files:

```bash
PYTHONPATH=src python scripts/import_mobileviews.py \
  --input runtime/datasets/mobileviews_600k/shards/MobileViews_0-150000.parquet \
  --output runtime/datasets/mobileviews_600k/graphs_50k_seed17 \
  --max-screens 50000 \
  --image-long-side 384 \
  --seed 17
```

Raw graph files are data-source artifacts, not a second training format. Build
MobileViews page pairs first; same-page augmentation is then generated inside
the one page-pair trainer:

```bash
PYTHONPATH=src python scripts/build_mobileviews_allowlist_pair_pool.py \
  --traces-root runtime/datasets/mobileviews/traces \
  --app-allowlist runtime/datasets/mobileviews/train_apps.json \
  --output-dir runtime/datasets/mobileviews/page_pairs_train
```

Aligned MobileViews/Mind2Web page pairs can update the same model without a
second scoring path. Automatic diagnostic labels require explicit opt-in and
only globally one-to-one rows enter the cross-page loss:

```bash
PYTHONPATH=src python scripts/train_mapping_page_pairs.py \
  --input runtime/datasets/mobileviews/diagnostic.jsonl \
  --allow-unreviewed-pseudo \
  --device cuda \
  --output runtime/models/mapping_self_supervised.pt
```

Each epoch mixes dual augmented views and aligned cross-page pairs in one
optimizer loop. Both pair sources use the same encoder, cross-attention matcher,
and symmetric correspondence objective. Stable dataset IDs are used only to
construct offline labels.

The official first shard contains 150,001 globally deduplicated screens. Its
expected size is 25,922,626,062 bytes and SHA-256 is
`440ceb895a4819de6e929fb39f7b8b6f869068f24c2862a92f876ddb687cca42`.
Unknown packages and screenshot/hierarchy geometry mismatches are excluded;
the manifest records every exclusion and proves package-set disjointness.

For newer AndroidWorld-derived screens, Uni-GUI-OpenMobile contributes about
25.9K trajectory steps from 2,640 tasks over 19 open-source apps. Every step
includes a screenshot and visible UI element attributes/bounds. The importer
deduplicates repeated screens, constructs containment relations for the same
Unified UIGraph representation, and keeps action boxes as label-only metadata:

```bash
PYTHONPATH=src python scripts/import_openmobile.py \
  --input runtime/datasets/unigui_openmobile_20260720/source \
  --output runtime/datasets/unigui_openmobile_20260720/graphs_seed17 \
  --image-long-side 384 \
  --seed 17
```

OS-Atlas adds a second, independently collected AndroidWorld grounding source.
The pinned `android_world` subset contains 15,905 screenshots and 89,860
bounded elements in a 2.36 GB release. Exact LFS paths must be selected so the
816 GB full repository is never downloaded accidentally. Its source does not
expose app identities, so image-level dev/test files are diagnostic only; all
formal generalization claims continue to use frozen app/domain-disjoint data:

```bash
PYTHONPATH=src python scripts/download_hf_lfs_files.py \
  --repository runtime/datasets/os_atlas_aw_20260720/source.part \
  --file-manifest artifacts/manifests/os_atlas_aw_20260720.json \
  --repo-id OS-Copilot/OS-Atlas-data \
  --revision e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f \
  --path mobile_domain/aw_mobile.json \
  --path mobile_domain/mobile_images.zip \
  --expected-files 2

PYTHONPATH=src python scripts/import_os_atlas_mobile.py \
  --annotations runtime/datasets/os_atlas_aw_20260720/source/mobile_domain/aw_mobile.json \
  --images runtime/datasets/os_atlas_aw_20260720/extracted \
  --output runtime/datasets/os_atlas_aw_20260720/graphs_seed17 \
  --image-long-side 384 \
  --seed 17
```

AndroidControl adds substantially broader app coverage: 15,283 demonstrations
across 833 Android apps with screenshots and true accessibility forests. Its
20 official shards total 49.93 GB, so the importer accepts HTTPS inputs and
converts one shard at a time without retaining the raw TFRecord. The official
episode train/validation/test split and published test subsplits remain frozen:

```bash
PYTHONPATH=src python scripts/import_android_control.py \
  --input \
    https://storage.googleapis.com/gresearch/android_control/android_control-00000-of-00020 \
  --output runtime/datasets/android_control_20260720/shard_00000 \
  --image-long-side 384 \
  --min-nodes 4
```

Each completed shard is atomic and contains `graphs.train.jsonl`,
`graphs.dev.jsonl`, and `graphs.test.jsonl`. Actions and instructions are kept
only as label metadata; the matcher still receives the same learned UI graph
features and never consumes source coordinates as target predictions.

For Web/WebView diversity, WebUI 70K contributes 173,546 screenshot rows with
visible semantic element boxes. Download the fixed converted-parquet commit,
accepting the dataset's research copyright terms, then import train only; the
official domain-disjoint validation and test releases remain frozen:

```bash
PYTHONPATH=src python scripts/download_hf_parquet_files.py \
  --repo-id biglab/webui-70k-elements \
  --revision 8b74f86a8418b4578ad85e73eaba3d6e7280ed9d \
  --output runtime/datasets/webui_70k_20260720/parquet

PYTHONPATH=src python scripts/import_webui.py \
  --input runtime/datasets/webui_70k_20260720/parquet/*.parquet \
  --output runtime/datasets/webui_70k_20260720/graphs_train \
  --dataset-id biglab/webui-70k-elements \
  --source-revision 8b74f86a8418b4578ad85e73eaba3d6e7280ed9d \
  --split train
```

Every additional source is adapted to `omnitransfer.mapping_page_pair.v1`
before it can enter optimization. There is no graph-only pretraining command
and no sequential domain-specific trainer.

Evaluate the frozen page-pair split without changing adapters:

```bash
PYTHONPATH=src:. python scripts/evaluate_ui_graph_matcher.py \
  --input runtime/datasets/unified_mapping/test.jsonl \
  --split test \
  --checkpoint runtime/models/context_forced_cross_attention_v1.pt \
  --device cuda \
  --output runtime/reports/context_forced_cross_attention_v1_test.json
```

Convert every source-target dataset to the same page-pair rows, then use the
one canonical training entrypoint:

```bash
PYTHONPATH=src python scripts/build_unified_mapping_dataset.py \
  --ase-queries runtime/evals/vision_widget_mapping/clean_relative_xml_v1/queries.jsonl \
  --page-pairs runtime/datasets/mobileviews/train-*.jsonl \
  --output-dir runtime/datasets/unified_mapping

PYTHONPATH=src python scripts/train_mapping_page_pairs.py \
  --input runtime/datasets/unified_mapping/train.jsonl \
  --validation-input runtime/datasets/unified_mapping/dev.jsonl \
  --assignment-head mutual_projection \
  --context-mask-probability 0.35 \
  --epochs 3 \
  --device cuda \
  --output runtime/models/context_forced_cross_attention_v1.pt
```

Enforce the decoded-image RTX 4090 compute budget after model-only warmup; PNG
decode/resize remains separately reported as asset preparation:

```bash
PYTHONPATH=src python scripts/benchmark_matcher_latency.py \
  --input runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.jsonl \
  --checkpoint runtime/models/relation_matcher_finetuned.pt \
  --device cuda \
  --budget-ms 50 \
  --fail-on-budget \
  --output runtime/reports/relation_matcher_latency.json
```

The machine-readable data roles for Android, iOS, DOM/AX, WebView, screenshots,
and frozen evaluation are in `datasets/unified_ui_registry.v1.json`.

Run the initial multi-seed architecture grid on one frozen split:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/run_learned_matcher_grid.py \
  --input runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.jsonl \
  --output-dir runtime/experiments/relation_matcher_grid \
  --seeds 17 29 41 \
  --split-seed 17 \
  --source-context-nodes 32 48 \
  --num-layers 1 2 \
  --epochs 3 \
  --device cuda
```

The method, benchmark expansion, 76-configuration experiment matrix, and
runtime promotion gates are specified in
`docs/learned_cross_attention_matcher.md`.
