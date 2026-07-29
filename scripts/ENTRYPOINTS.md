# Script Entrypoints

Status: RCAM benchmark implemented

The standalone repo keeps canonical data, training, and evaluation entrypoints.

## Entrypoints

- `PYTHONPATH=src python scripts/import_dataset.py --input ... --output ...`
  - Normalize action-transfer JSONL and infer set-valued wrapper/label equivalence.
- `python scripts/clean_widget_mapping_dataset.py --input ... --output ...`
  - Bind every public widget pair to real source/target XML nodes, write one
    relative-coordinate XML per screen, remove public/XML duplicate candidates,
    and freeze app-disjoint train/dev/test rows with a strict contamination audit.
- `PYTHONPATH=src python scripts/import_mobileviews.py --input ... --output ...`
  - Stream official `image_content/json_content` parquet rows into lossless
    384px images and immutable UIGraph ingestion shards. Importer partitions
    are not canonical matcher splits.
- `PYTHONPATH=src python scripts/import_openmobile.py --input ... --output ...`
  - Import Uni-GUI-OpenMobile trajectories with per-screen UI elements into deduplicated, package-disjoint UIGraph JSONL files.
- `PYTHONPATH=src python scripts/download_hf_lfs_files.py --repository ... --path ...`
  - Download only explicitly selected LFS paths from a fixed Hugging Face revision and verify their SHA-256 values.
- `PYTHONPATH=src python scripts/import_os_atlas_mobile.py --annotations ... --images ... --output ...`
  - Import OS-Atlas AndroidWorld screenshots and grouped grounding boxes into the same Unified UIGraph matcher path.
- `PYTHONPATH=src python scripts/import_android_control.py --input ... --output ...`
  - Stream official AndroidControl GZIP TFRecord shards into frozen accessibility-tree UIGraph bundles without TensorFlow.
- `PYTHONPATH=src python scripts/import_gui_odyssey.py --max-episodes ... --output ...`
  - Import GUIOdyssey CLICK steps with SAM2 boxes as weak pseudo UIGraph records for action-grounding distillation.
- `PYTHONPATH=src python scripts/build_gui_odyssey_review_queue.py --annotations ... --output ...`
  - Rank cross-device, foldable, browser/WebView, layout-shift, and hard-NULL candidates for human-only correspondence labeling.
- `PYTHONPATH=src python scripts/download_gui_odyssey_review_images.py --queue ... --output ...`
  - Download only queue-referenced PNGs from pinned individual-file mirrors and verify their dimensions and SHA-256 values.
- `PYTHONPATH=src python scripts/embed_gui_odyssey_review_images.py --queue ... --images ... --output ...`
  - Build a single-file review page with compressed embedded images for browsers that block local subresources and loopback URLs.
- `PYTHONPATH=src python scripts/download_hf_parquet_files.py --repo-id ... --revision ... --output ...`
  - Download a fixed Hugging Face converted-parquet release and verify every shard by the Hub-provided SHA-256.
- `PYTHONPATH=src python scripts/import_webui.py --input ... --output ... --source-revision ... --split train`
  - Import WebUI screenshots and visible semantic element boxes into the same Unified UIGraph matcher path.
- `PYTHONPATH=src python scripts/build_ui_correspondence_dataset.py --ase-queries ... --correspondence-pairs ... --output-dir ...`
  - Normalize every admitted source into one
    `omnitransfer.ui_correspondence_pair.v1` pool, then split inside each App by
    page-connected components. Dataset origin does not create separate training
    logic; pair/page/component overlap is forbidden, while App overlap is
    intentional.
- `PYTHONPATH=src python scripts/build_ui_correspondence_review.py --input ... --output-dir ...`
  - Render UI correspondence records through the canonical review workbench.
- `PYTHONPATH=src python scripts/train_relation_aware_matcher.py --input train.jsonl --validation-input dev.jsonl --output ...`
  - The only RCAM training entrypoint. It mixes same-page augmentation and cross-page set-valued correspondences in one optimizer loop and records appendix-ready JSONL metrics.
- `PYTHONPATH=src:. python scripts/evaluate_relation_aware_matcher.py --input test.jsonl --split test --checkpoint ... --output ...`
  - Evaluate Top-1, Recall@K, and warm latency through the same UI-correspondence adapter used by training.
- `PYTHONPATH=src python scripts/evaluate_learned_matcher.py --input ... --checkpoint ... --eval-split test --report ... --predictions ...`
  - Legacy frozen-query checkpoint evaluation only; it is not a training path.
- `PYTHONPATH=src:scripts python scripts/build_frozen_gui_odyssey_review.py --candidates ... --raw-pairs ... --reserved-split test --output ...`
  - Render a diverse multi-node review batch from exact pair IDs in a frozen GUIOdyssey dev/test partition; ranking never creates labels.
- `PYTHONPATH=src python scripts/train_mutual_matcher.py --input ... --output ... --report ...`
- `PYTHONPATH=src python scripts/evaluate_mutual_matcher.py --input ... --checkpoint ... --eval-split test --report ... --predictions ...`
  - Archived best-checkpoint reproduction and evaluation only. This query-format
    baseline is retained for comparison, but it is not part of the canonical
    data or training pipeline.
- `PYTHONPATH=src python scripts/evaluate_clean_widget_mapping.py --input runtime/evals/vision_widget_mapping/clean_relative_xml_v1/queries.jsonl --mode anchor --output ...`
  - Evaluate deterministic similarity, local-anchor, or identity-selector baselines on the frozen clean candidate and gold sets.
- `PYTHONPATH=src python scripts/mine_matcher_disagreements.py --queries ... --learned-predictions ... --selector-predictions ... --output ...`
  - Build the held-out rule-resistant slice from set-valued gold-rank, Top1, margin, and NULL disagreements without feeding selector scores to the matcher.
- `PYTHONPATH=src python scripts/benchmark_matcher_latency.py --input ... --checkpoint ... --output ...`
  - Enforce the strict decoded-image RTX 4090 compute budget with synchronized stage timing.
- `PYTHONPATH=src python scripts/run_relation_aware_matcher_grid.py --input train.jsonl --validation-input dev.jsonl --output-dir ...`
  - Run layer/context ablations by repeatedly invoking the one canonical RCAM trainer.
- `PYTHONPATH=src python scripts/run_eval.py --queries ... --predictions ... --output ...`
  - Report ranking, abstention, risk-coverage, slice, and latency metrics.
- `python scripts/run_external_baseline.py`
  - Generate predictions for external baselines such as LightGlue or a VLM API.
- `python scripts/summarize_results.py`
  - Summarize recorded evaluation artifacts.

## Not Included

This outline does not include OmniFlow provider, AndroidWorld launcher, OOB
native integration, Function replay, device execution scripts, or raw runtime
data. Use the OmniFlow monorepo for those runtime paths.
