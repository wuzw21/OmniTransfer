#!/usr/bin/env python3
"""Score gold correspondences by probability the current model missed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnitransfer.hard_example_scoring import (
    score_prediction_hardness,
    summarize_hard_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    scored_rows: list[dict[str, Any]] = []
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hardness = score_prediction_hardness(row)
            scored_rows.append(
                {
                    "schema_version": "omnitransfer.hard_correspondence_score.v1",
                    "graph_pair": row.get("graph_pair"),
                    "direction": row.get("direction"),
                    "source": _node_ref(row.get("source")),
                    "gold_targets": [
                        _node_ref(node) for node in row.get("gold_targets") or []
                    ],
                    "prediction": _node_ref(row.get("prediction")),
                    "correct": bool(row.get("correct")),
                    "gold_rank": int(row.get("gold_rank") or 0),
                    **hardness,
                }
            )

    scored_rows.sort(
        key=lambda row: -float(row["difficulty_score"])
    )
    summary_rows = [
        {
            "difficulty_score": row["difficulty_score"],
            "correct": row.get("correct"),
        }
        for row in scored_rows
    ]
    report = {
        "schema_version": "omnitransfer.hard_correspondence_scores.v1",
        "definition": "difficulty_score = 100 * (1 - model probability on accepted gold endpoints)",
        "runtime_effect": "none; training sampler metadata only",
        "probability_exact_rows": sum(
            bool(row["probability_is_exact"])
            for row in scored_rows
        ),
        **summarize_hard_examples(summary_rows),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def _node_ref(value: Any) -> dict[str, Any]:
    node = value if isinstance(value, dict) else {}
    return {
        "index": node.get("index"),
        "node_id": node.get("node_id"),
    }


if __name__ == "__main__":
    main()
