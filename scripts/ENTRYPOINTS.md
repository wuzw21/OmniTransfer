# Script Entrypoints

Status: Learned matcher benchmark implemented

The standalone repo keeps canonical data, training, and evaluation entrypoints.

## Entrypoints

- `PYTHONPATH=src python scripts/import_dataset.py --input ... --output ...`
  - Normalize action-transfer JSONL and infer set-valued wrapper/label equivalence.
- `python scripts/clean_widget_mapping_dataset.py --input ... --output ...`
  - Bind every public widget pair to real source/target XML nodes, write one
    relative-coordinate XML per screen, remove public/XML duplicate candidates,
    and freeze app-disjoint train/dev/test rows with a strict contamination audit.
- `PYTHONPATH=src python scripts/import_mobileviews.py --input ... --output ...`
  - Stream official `image_content/json_content` parquet rows into lossless 384px images and package-disjoint UIGraph JSONL files.
- `PYTHONPATH=src python scripts/freeze_mobileviews_app_split.py --traces-root ... --frozen-test-apps ... --output-dir ...`
  - Index all extracted complete-trace batches, reject duplicate App identities,
    scan strict pair capacity, and freeze App-disjoint train/dev-candidate/test
    allowlists without GUIOdyssey data.
- `PYTHONPATH=src python scripts/build_mobileviews_allowlist_pair_pool.py --traces-root ... --app-allowlist ... --output-dir ...`
  - Build self-supervised MobileViews training pairs or an unreviewed
    diagnostic pool from a frozen App allowlist; offline identity fields remain
    label-only.
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
- `PYTHONPATH=src python scripts/pretrain_ui_graph_matcher.py --input ... --output ...`
  - Self-supervise the relation-aware matcher on one or more UI graph streams; `--pretrained` continues the same encoder and matcher checkpoint.
- `PYTHONPATH=src python scripts/train_mapping_page_pairs.py --input ... --pretrained ... --output ... --allow-unreviewed-pseudo`
  - Mix dual augmented views and strict cross-page correspondences in one optimizer loop with one matcher and one symmetric objective. Unreviewed automatic labels require explicit opt-in; ambiguous and target-collision rows are filtered.
- `PYTHONPATH=src:. python scripts/evaluate_ui_graph_matcher.py --input ... --checkpoint ... --output ...`
  - Evaluate positive Top1 and Recall@K on immutable UI graph files without updating the checkpoint.
- `PYTHONPATH=src python scripts/train_learned_matcher.py --input ... --pretrained ... --output ...`
- `PYTHONPATH=src python scripts/evaluate_learned_matcher.py --input ... --checkpoint ... --eval-split test --report ... --predictions ...`
  - Jointly fine-tune the encoder and matcher with set-valued/NULL labels and optional verified outcome preferences.
- `PYTHONPATH=src python scripts/train_mixed_generalization_matcher.py --ase-queries ... --gui-pairs ... --gui-annotations ... --checkpoint ... --report ...`
  - Mix the frozen ASE gold train split with leakage-audited GUIOdyssey trajectory correspondences in each optimizer step; evaluate ASE gold, held-out GUI weak pairs, optional exported human gold, device-pair slices, NULL behavior, and warm image latency.
- `PYTHONPATH=src python scripts/build_unified_mapping_dataset.py --ase-queries ... --gui-pairs ... --gui-annotations ... --output-dir ...`
  - Build one page-pair schema for train/dev/test, keep held-out GUIOdyssey weak labels diagnostic-only, and freeze dev/test candidates for human gold review.
- `PYTHONPATH=src:scripts python scripts/build_frozen_gui_odyssey_review.py --candidates ... --raw-pairs ... --reserved-split test --output ...`
  - Render a diverse multi-node review batch from exact pair IDs in a frozen GUIOdyssey dev/test partition; ranking never creates labels.
- `PYTHONPATH=src python scripts/train_mutual_matcher.py --input ... --output ... --report ...`
- `PYTHONPATH=src python scripts/evaluate_mutual_matcher.py --input ... --checkpoint ... --eval-split test --report ... --predictions ...`
  - Train or evaluate the explicit semantic/visual/attribute/context/geometry/anchor feature-matrix matcher with one bidirectionally normalized final matrix.
- `PYTHONPATH=src python scripts/evaluate_clean_widget_mapping.py --input runtime/evals/vision_widget_mapping/clean_relative_xml_v1/queries.jsonl --mode anchor --output ...`
  - Evaluate deterministic similarity, local-anchor, or identity-selector baselines on the frozen clean candidate and gold sets.
- `PYTHONPATH=src python scripts/mine_matcher_disagreements.py --queries ... --learned-predictions ... --selector-predictions ... --output ...`
  - Build the held-out rule-resistant slice from set-valued gold-rank, Top1, margin, and NULL disagreements without feeding selector scores to the matcher.
- `PYTHONPATH=src python scripts/benchmark_matcher_latency.py --input ... --checkpoint ... --output ...`
  - Enforce the strict decoded-image RTX 4090 compute budget with synchronized stage timing.
- `PYTHONPATH=src python scripts/run_learned_matcher_grid.py --input ... --output-dir ...`
  - Run the frozen multi-seed context-size/layer grid and collect one summary.
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
