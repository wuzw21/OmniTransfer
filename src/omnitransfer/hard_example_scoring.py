"""Model-derived difficulty scores for gold UI correspondences.

The score is training-data metadata only.  It never changes matcher scores or
runtime ranking.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def score_prediction_hardness(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return how much probability the model failed to assign to the gold set.

    A score of zero means all probability belongs to accepted gold endpoints;
    100 means the model assigned no represented probability to them.  Set-valued
    gold is handled by summing the probability of every accepted endpoint.
    """

    all_candidates = row.get("all_candidates")
    probability_is_exact = isinstance(all_candidates, list)
    candidates = (
        all_candidates
        if probability_is_exact
        else list(row.get("top_candidates") or [])
    )
    gold_targets = list(row.get("gold_targets") or [])
    gold_probability = sum(
        float(candidate.get("rank_probability") or 0.0)
        for candidate in candidates
        if any(_same_node(candidate, gold) for gold in gold_targets)
    )
    gold_probability = min(1.0, max(0.0, gold_probability))
    return {
        "difficulty_score": round(100.0 * (1.0 - gold_probability), 6),
        "gold_probability": round(gold_probability, 9),
        "probability_is_exact": probability_is_exact,
    }


def summarize_hard_examples(
    rows: list[Mapping[str, Any]],
    *,
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> dict[str, Any]:
    """Measure whether the score concentrates actual errors near the top."""

    ranked = sorted(
        rows,
        key=lambda row: -float(row.get("difficulty_score") or 0.0),
    )
    total_errors = sum(not bool(row.get("correct")) for row in ranked)
    hardest: dict[str, dict[str, float | int]] = {}
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fractions must be in (0, 1]")
        selected = ranked[: math.ceil(len(ranked) * fraction)]
        selected_errors = sum(not bool(row.get("correct")) for row in selected)
        hardest[f"{fraction:.2f}"] = {
            "rows": len(selected),
            "error_rate": selected_errors / len(selected) if selected else 0.0,
            "error_capture": selected_errors / total_errors if total_errors else 0.0,
        }
    return {
        "rows": len(ranked),
        "errors": total_errors,
        "top1_accuracy": (
            (len(ranked) - total_errors) / len(ranked) if ranked else 0.0
        ),
        "hardest_fractions": hardest,
    }


def _same_node(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_id = left.get("node_id")
    right_id = right.get("node_id")
    if left_id is not None and right_id is not None:
        return str(left_id) == str(right_id)
    return left.get("index") is not None and left.get("index") == right.get("index")
