#!/usr/bin/env python3
"""Evaluate candidate predictions with set-valued and NULL-aware metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from omnitransfer.benchmark import evaluate_slices
from omnitransfer.importers import load_queries
from omnitransfer.schema import Prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    predictions = _load_predictions(args.predictions)
    metrics = {
        name: asdict(summary)
        for name, summary in evaluate_slices(queries, predictions).items()
    }
    payload = {
        "schema_version": "omnitransfer_benchmark_v2",
        "queries": str(args.queries.resolve()),
        "predictions": str(args.predictions.resolve()),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["all"], ensure_ascii=False))


def _load_predictions(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "")
            if not query_id:
                raise ValueError(f"prediction row {line_number} missing query_id")
            selected = row.get("selected_candidate_id")
            predictions.append(
                Prediction(
                    query_id=query_id,
                    selected_candidate_id=str(selected) if selected not in (None, "") else None,
                    scores={
                        str(candidate_id): float(score)
                        for candidate_id, score in dict(row.get("scores") or {}).items()
                    },
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return predictions


if __name__ == "__main__":
    main()
