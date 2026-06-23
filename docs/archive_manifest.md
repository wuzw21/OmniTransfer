# Archive Manifest

This repository outline does not track raw runtime data in git. The files remain
on disk and should be preserved.

## Runtime Root

```text
/Users/wuzewen/Projects/Omni/OmniTransfer/runtime/
```

Current size:

```text
runtime = 7.9G
runtime/evals = 7.9G
runtime/long_term_memory = 128K
```

## Primary Data

```text
runtime/evals/vision_widget_mapping/
```

Contains:

- public dataset screenshots
- XML UI trees
- original zip/z01 archives
- cloned source snapshot
- copied paper PDF/text
- imported ActionTransfer JSONL
- train/dev/test splits
- local evaluation artifacts
- remote 9207 evaluation artifacts

Important files:

```text
runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.jsonl
runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.summary.json
runtime/evals/vision_widget_mapping/rich_eval.omnitransfer_local_rescue_floor_fulltest_20260620.json
runtime/evals/vision_widget_mapping/remote_9207_20260620/rich_eval.lightglue_fulltest_20260620.json
runtime/evals/vision_widget_mapping/remote_9207_20260620_trainable/rich_eval.trainable_omnitransfer_model_train1000_eval500_20260620.json
```

## Diagnostic Data

```text
runtime/evals/temdroid_semfinder_action_transfer_compare_20260615/
```

Contains SemFinder/TEMdroid metadata-only import, Top1 view, and comparison
summaries.

## Long-Term Notes

```text
runtime/long_term_memory/action_transfer_v2_design_from_gbdt_20260617.md
runtime/long_term_memory/action_transfer_v2_lightglue_on_device_20260616.md
runtime/long_term_memory/action_transfer_history_conditioned_related_work_20260617.md
runtime/long_term_memory/action_transfer_gold_dataset_rules_20260611.md
```

## Preservation Rule

Do not delete these runtime files during repo cleanup. They are the source
materials behind the outline.
