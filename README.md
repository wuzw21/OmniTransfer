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
