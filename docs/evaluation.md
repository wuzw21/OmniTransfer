# Evaluation

## Primary Dataset

Use the public `RuihuaJi/vision-based-widget-mapping` dataset after importing it
into canonical ActionTransfer JSONL.

Local path:

```text
runtime/evals/vision_widget_mapping/action_transfer_queries.with_images.jsonl
```

Recorded summary:

```text
queries = 6,730
target candidates = 265,590
positives = 6,730
```

## Diagnostic Dataset

TEMdroid/SemFinder is retained as metadata-only diagnostic data. It does not
include the same screenshot/XML grounding richness and should not be used as the
main visual/layout robustness dataset.

Local path:

```text
runtime/evals/temdroid_semfinder_action_transfer_compare_20260615/
```

## Metric

The main metric is Top1 target grounding accuracy:

```text
Top1 = count(argmax_i score_i == gold) / number_of_queries
```

Latency should be reported with average and p95.

## Baseline Families

- fixed coordinate replay
- text/resource/class feature scorer
- paper IH-layout approximation
- GBDT structured ranker
- LightGlue external visual matcher
- optional VLM API baseline
- OmniTransfer structured ranker

## Reporting Rule

This outline repository does not promote any historical diagnostic number as a
new headline result. Existing result files are preserved as local artifacts and
listed in `docs/archive_manifest.md`; future reports should cite the exact
artifact, dataset split, matcher name, and latency setting.
