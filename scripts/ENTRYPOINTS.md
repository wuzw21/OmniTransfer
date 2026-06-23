# Script Entrypoints

Status: Outline

The standalone repo currently keeps only stable placeholder entrypoints. The
real implementation will be migrated after the schema and data import contract
are frozen.

## Planned Entrypoints

- `python scripts/import_dataset.py`
  - Convert public datasets into canonical `Query` rows.
- `python scripts/run_eval.py`
  - Run Top1 grounding evaluation for one dataset and one matcher.
- `python scripts/run_external_baseline.py`
  - Generate predictions for external baselines such as LightGlue or a VLM API.
- `python scripts/summarize_results.py`
  - Summarize recorded evaluation artifacts.

## Not Included

This outline does not include OmniFlow provider, AndroidWorld launcher, OOB
native integration, Function replay, device execution scripts, or raw runtime
data. Use the OmniFlow monorepo for those runtime paths.
