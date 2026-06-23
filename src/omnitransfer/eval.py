"""Evaluation helpers for Top1 grounding accuracy."""

from __future__ import annotations

from collections.abc import Iterable

from omnitransfer.schema import Prediction, Query


def top1_accuracy(queries: Iterable[Query], predictions: Iterable[Prediction]) -> float:
    """Compute Top1 target grounding accuracy."""

    gold = {query.query_id: query.gold_candidate_id for query in queries}
    total = 0
    correct = 0
    for prediction in predictions:
        expected = gold.get(prediction.query_id)
        if expected is None:
            continue
        total += 1
        if prediction.selected_candidate_id == expected:
            correct += 1
    return correct / total if total else 0.0
