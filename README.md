# OmniTransfer

OmniTransfer is a lightweight research repository outline for source-context
UI grounding relocation on mobile GUI agents.

The task is:

```text
input:  source UI graph G_s, source point/node p_s, target UI graph G_t
output: target point/node p_t
```

It is not action prediction. The cached function or planner decides the action
type; OmniTransfer only relocates the target grounding.

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

OmniTransfer ranks target UI candidates using source grounding features,
target-candidate features, local UI structure, and optional historical context.
The first deployable version is a structured ranker rather than a VLM or
end-to-end action model.

## Smoke Check

```bash
python -m py_compile \
  src/omnitransfer/*.py \
  scripts/import_dataset.py \
  scripts/run_eval.py \
  scripts/run_external_baseline.py \
  scripts/summarize_results.py \
  tests/test_schema.py \
  tests/test_outline_imports.py

PYTHONPATH=src python -m pytest tests/test_schema.py tests/test_outline_imports.py
```
