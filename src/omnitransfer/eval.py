"""Evaluation helpers for Top1 grounding accuracy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from omnitransfer.schema import Prediction, Query


@dataclass(frozen=True)
class RankingMetrics:
    """Set-valued ranking, abstention, and safety metrics."""

    total: int
    positive_total: int
    null_total: int
    top1_accuracy: float
    recall_at_k: dict[int, float]
    mean_reciprocal_rank: float
    null_accuracy: float
    false_positive_rate: float
    wrong_target_rate: float
    coverage: float
    selective_accuracy: float
    area_under_risk_coverage: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    under_50ms_rate: float


def top1_accuracy(queries: Iterable[Query], predictions: Iterable[Prediction]) -> float:
    """Compute Top1 target grounding accuracy."""

    gold = {
        query.query_id: frozenset(query.acceptable_gold_candidate_ids()) for query in queries
    }
    total = 0
    correct = 0
    for prediction in predictions:
        expected = gold.get(prediction.query_id)
        if expected is None:
            continue
        total += 1
        if (
            prediction.selected_candidate_id in expected
            if expected
            else prediction.selected_candidate_id is None
        ):
            correct += 1
    return correct / total if total else 0.0


def ranking_metrics(
    queries: Iterable[Query],
    predictions: Iterable[Prediction],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> RankingMetrics:
    """Evaluate equivalent targets and NULL without assuming a bijection."""

    query_by_id = {query.query_id: query for query in queries}
    recall_hits = {k: 0 for k in ks if k > 0}
    total = positive_total = null_total = correct = null_correct = 0
    wrong_targets = false_positives = covered = covered_correct = 0
    reciprocal_rank_sum = 0.0
    confidence_outcomes: list[tuple[float, bool]] = []
    latencies: list[float] = []
    for prediction in predictions:
        query = query_by_id.get(prediction.query_id)
        if query is None:
            continue
        total += 1
        gold_ids = frozenset(query.acceptable_gold_candidate_ids())
        selected = prediction.selected_candidate_id
        ranked = prediction.ranked_candidate_ids()
        if not gold_ids:
            null_total += 1
            if selected is None:
                correct += 1
                null_correct += 1
            else:
                false_positives += 1
                wrong_targets += 1
        else:
            positive_total += 1
            if selected in gold_ids:
                correct += 1
            elif selected is not None:
                wrong_targets += 1
            for rank, candidate_id in enumerate(ranked, start=1):
                if candidate_id in gold_ids:
                    reciprocal_rank_sum += 1.0 / rank
                    break
            for k in recall_hits:
                if any(candidate_id in gold_ids for candidate_id in ranked[:k]):
                    recall_hits[k] += 1
        if selected is not None:
            covered += 1
            if selected in gold_ids:
                covered_correct += 1
        confidence = _prediction_confidence(prediction)
        confidence_outcomes.append(
            (
                confidence,
                selected in gold_ids if gold_ids else selected is None,
            )
        )
        if prediction.metadata.get("latency_ms") is not None:
            latencies.append(float(prediction.metadata["latency_ms"]))
    return RankingMetrics(
        total=total,
        positive_total=positive_total,
        null_total=null_total,
        top1_accuracy=correct / total if total else 0.0,
        recall_at_k={
            k: hits / positive_total if positive_total else 0.0
            for k, hits in recall_hits.items()
        },
        mean_reciprocal_rank=(
            reciprocal_rank_sum / positive_total if positive_total else 0.0
        ),
        null_accuracy=null_correct / null_total if null_total else 0.0,
        false_positive_rate=false_positives / null_total if null_total else 0.0,
        wrong_target_rate=wrong_targets / total if total else 0.0,
        coverage=covered / total if total else 0.0,
        selective_accuracy=covered_correct / covered if covered else 0.0,
        area_under_risk_coverage=_area_under_risk_coverage(confidence_outcomes),
        average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 50.0),
        p95_latency_ms=_percentile(latencies, 95.0),
        max_latency_ms=max(latencies, default=0.0),
        under_50ms_rate=(
            sum(latency < 50.0 for latency in latencies) / len(latencies)
            if latencies
            else 0.0
        ),
    )


def _prediction_confidence(prediction: Prediction) -> float:
    if prediction.selected_candidate_id is None:
        if prediction.metadata.get("null_probability") is not None:
            return float(prediction.metadata["null_probability"])
        return max(0.0, 1.0 - max(prediction.scores.values(), default=0.0))
    return float(
        prediction.metadata.get(
            "top1_probability",
            prediction.scores.get(prediction.selected_candidate_id, 0.0),
        )
    )


def _area_under_risk_coverage(outcomes: list[tuple[float, bool]]) -> float:
    if not outcomes:
        return 0.0
    ranked = sorted(outcomes, key=lambda item: -item[0])
    correct = 0
    risks: list[float] = []
    for covered, (_, is_correct) in enumerate(ranked, start=1):
        correct += int(is_correct)
        risks.append(1.0 - correct / covered)
    return sum(risks) / len(risks)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile / 100.0))))
    return float(ordered[index])
