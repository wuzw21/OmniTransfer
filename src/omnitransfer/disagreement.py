"""Mine benchmark cases where a learned matcher separates from a rule selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

from omnitransfer.schema import Prediction, Query


@dataclass(frozen=True)
class DisagreementCase:
    """One query prioritized by calibrated rank disagreement, not raw score scale."""

    query_id: str
    category: str
    disagreement_score: float
    gold_candidate_ids: tuple[str, ...]
    learned_selected_candidate_id: str | None
    selector_selected_candidate_id: str | None
    learned_gold_rank: int | None
    selector_gold_rank: int | None
    rank_advantage: int
    learned_margin: float
    selector_margin: float
    candidate_count: int
    source_point: tuple[float, float] | None
    gold_target_points: tuple[tuple[float, float], ...]
    learned_target_point: tuple[float, float] | None
    selector_target_point: tuple[float, float] | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return asdict(self)


def mine_disagreements(
    queries: Iterable[Query],
    learned_predictions: Iterable[Prediction],
    selector_predictions: Iterable[Prediction],
    *,
    min_rank_advantage: int = 1,
    include_regressions: bool = False,
) -> list[DisagreementCase]:
    """Find rule-resistant cases using correctness and within-model ranks.

    Raw logits from two different models are not directly comparable. This
    miner therefore uses set-valued gold rank, top-1 disagreement, each
    model's own margin, and NULL behavior. The selector is a diagnostic
    baseline only and never becomes a matcher input or a training target.
    """

    query_by_id = {query.query_id: query for query in queries}
    learned_by_id = _index_predictions(learned_predictions, name="learned")
    selector_by_id = _index_predictions(selector_predictions, name="selector")
    shared_ids = sorted(set(query_by_id) & set(learned_by_id) & set(selector_by_id))
    cases: list[DisagreementCase] = []
    for query_id in shared_ids:
        query = query_by_id[query_id]
        learned = learned_by_id[query_id]
        selector = selector_by_id[query_id]
        gold_ids = query.acceptable_gold_candidate_ids()
        learned_rank = _best_gold_rank(learned, gold_ids)
        selector_rank = _best_gold_rank(selector, gold_ids)
        learned_correct = _is_correct(learned, gold_ids)
        selector_correct = _is_correct(selector, gold_ids)
        learned_margin = _prediction_margin(learned)
        selector_margin = _prediction_margin(selector)
        rank_advantage = _rank_advantage(
            learned_rank,
            selector_rank,
            candidate_count=len(query.target_candidates),
        )
        category = _category(
            gold_ids=gold_ids,
            learned=learned,
            selector=selector,
            learned_correct=learned_correct,
            selector_correct=selector_correct,
            rank_advantage=rank_advantage,
            min_rank_advantage=min_rank_advantage,
        )
        if category is None:
            if not include_regressions:
                continue
            category = _regression_category(
                gold_ids=gold_ids,
                learned=learned,
                selector=selector,
                learned_correct=learned_correct,
                selector_correct=selector_correct,
                rank_advantage=rank_advantage,
                min_rank_advantage=min_rank_advantage,
            )
            if category is None:
                continue
        score = _disagreement_score(
            category=category,
            learned_rank=learned_rank,
            selector_rank=selector_rank,
            learned_margin=learned_margin,
            selector_margin=selector_margin,
            candidate_count=len(query.target_candidates),
            metadata=query.metadata,
        )
        cases.append(
            DisagreementCase(
                query_id=query_id,
                category=category,
                disagreement_score=round(score, 6),
                gold_candidate_ids=gold_ids,
                learned_selected_candidate_id=learned.selected_candidate_id,
                selector_selected_candidate_id=selector.selected_candidate_id,
                learned_gold_rank=learned_rank,
                selector_gold_rank=selector_rank,
                rank_advantage=rank_advantage,
                learned_margin=round(learned_margin, 6),
                selector_margin=round(selector_margin, 6),
                candidate_count=len(query.target_candidates),
                source_point=_source_point(query),
                gold_target_points=_candidate_points(query, gold_ids),
                learned_target_point=_candidate_point(
                    query, learned.selected_candidate_id
                ),
                selector_target_point=_candidate_point(
                    query, selector.selected_candidate_id
                ),
                metadata=_case_metadata(query),
            )
        )
    return sorted(cases, key=lambda case: (-case.disagreement_score, case.query_id))


def group_disagreements_by_page_pair(
    cases: Iterable[DisagreementCase],
    *,
    max_anchors_per_pair: int = 0,
) -> list[dict[str, Any]]:
    """Group multiple hard anchors on the same screenshot pair."""

    grouped: dict[tuple[str, str], list[DisagreementCase]] = {}
    for case in cases:
        source_path = str(case.metadata.get("source_screenshot_path") or "")
        target_path = str(case.metadata.get("target_screenshot_path") or "")
        if not source_path or not target_path:
            continue
        grouped.setdefault((source_path, target_path), []).append(case)
    page_pairs = []
    for (source_path, target_path), anchors in grouped.items():
        anchors = sorted(
            anchors,
            key=lambda case: (-case.disagreement_score, case.query_id),
        )[: max_anchors_per_pair or None]
        pair_key = f"{source_path}\n{target_path}"
        page_pairs.append(
            {
                "schema_version": "omnitransfer_rule_resistant_page_pair_v1",
                "page_pair_id": "rule-resistant-"
                + hashlib.blake2b(pair_key.encode("utf-8"), digest_size=10).hexdigest(),
                "source_screenshot_path": source_path,
                "target_screenshot_path": target_path,
                "anchor_count": len(anchors),
                "max_disagreement_score": max(
                    case.disagreement_score for case in anchors
                ),
                "metadata": {
                    key: anchors[0].metadata[key]
                    for key in ("app", "category")
                    if key in anchors[0].metadata
                },
                "anchors": [case.to_dict() for case in anchors],
            }
        )
    return sorted(
        page_pairs,
        key=lambda row: (-row["max_disagreement_score"], row["page_pair_id"]),
    )


def prediction_from_mapping(row: Mapping[str, Any]) -> Prediction:
    """Normalize a JSONL or rich-eval prediction into the stable schema."""

    selected = row.get("selected_candidate_id")
    if selected in ("", "NULL", "__NULL__"):
        selected = None
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        scores = {
            str(candidate["candidate_id"]): float(candidate.get("score") or 0.0)
            for candidate in row.get("top_candidates") or ()
            if candidate.get("candidate_id") not in (None, "NULL", "__NULL__")
        }
    metadata = dict(row.get("metadata") or {})
    for key in ("margin", "score", "correct", "gold_candidate_id"):
        if key in row and key not in metadata:
            metadata[key] = row[key]
    return Prediction(
        query_id=str(row["query_id"]),
        selected_candidate_id=str(selected) if selected is not None else None,
        scores={str(key): float(value) for key, value in scores.items()},
        metadata=metadata,
    )


def _index_predictions(
    predictions: Iterable[Prediction],
    *,
    name: str,
) -> dict[str, Prediction]:
    indexed: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.query_id in indexed:
            raise ValueError(f"duplicate {name} prediction for {prediction.query_id}")
        indexed[prediction.query_id] = prediction
    return indexed


def _best_gold_rank(prediction: Prediction, gold_ids: tuple[str, ...]) -> int | None:
    if not gold_ids:
        return 1 if prediction.selected_candidate_id is None else None
    gold = set(gold_ids)
    return next(
        (
            rank
            for rank, candidate_id in enumerate(prediction.ranked_candidate_ids(), start=1)
            if candidate_id in gold
        ),
        None,
    )


def _is_correct(prediction: Prediction, gold_ids: tuple[str, ...]) -> bool:
    if not gold_ids:
        return prediction.selected_candidate_id is None
    return prediction.selected_candidate_id in set(gold_ids)


def _rank_advantage(
    learned_rank: int | None,
    selector_rank: int | None,
    *,
    candidate_count: int,
) -> int:
    missing_rank = max(candidate_count, 1) + 1
    return (selector_rank or missing_rank) - (learned_rank or missing_rank)


def _category(
    *,
    gold_ids: tuple[str, ...],
    learned: Prediction,
    selector: Prediction,
    learned_correct: bool,
    selector_correct: bool,
    rank_advantage: int,
    min_rank_advantage: int,
) -> str | None:
    if not gold_ids and learned_correct and not selector_correct:
        return "learned_null_selector_false_positive"
    if learned_correct and not selector_correct:
        return "learned_correct_selector_wrong"
    if (
        gold_ids
        and learned.selected_candidate_id != selector.selected_candidate_id
        and rank_advantage >= min_rank_advantage
    ):
        return "learned_rank_advantage"
    return None


def _regression_category(
    *,
    gold_ids: tuple[str, ...],
    learned: Prediction,
    selector: Prediction,
    learned_correct: bool,
    selector_correct: bool,
    rank_advantage: int,
    min_rank_advantage: int,
) -> str | None:
    if not gold_ids and selector_correct and not learned_correct:
        return "learned_false_positive_selector_null"
    if selector_correct and not learned_correct:
        return "selector_correct_learned_wrong"
    if (
        gold_ids
        and learned.selected_candidate_id != selector.selected_candidate_id
        and rank_advantage <= -min_rank_advantage
    ):
        return "selector_rank_advantage"
    return None


def _prediction_margin(prediction: Prediction) -> float:
    value = prediction.metadata.get("margin")
    if value is not None:
        try:
            margin = float(value)
        except (TypeError, ValueError):
            margin = 0.0
        return margin if math.isfinite(margin) else 0.0
    ranked = sorted(prediction.scores.values(), reverse=True)
    if not ranked:
        return 0.0
    return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)


def _disagreement_score(
    *,
    category: str,
    learned_rank: int | None,
    selector_rank: int | None,
    learned_margin: float,
    selector_margin: float,
    candidate_count: int,
    metadata: Mapping[str, Any],
) -> float:
    learned_rr = 1.0 / learned_rank if learned_rank else 0.0
    selector_rr = 1.0 / selector_rank if selector_rank else 0.0
    direction = -1.0 if category.startswith("selector_") or category.startswith("learned_false") else 1.0
    correctness_bonus = 2.0 if "correct" in category or "null" in category else 0.8
    margin_evidence = max(selector_margin, 0.0) + max(learned_margin, 0.0)
    candidate_pressure = min(math.log2(max(candidate_count, 1)) / 6.0, 1.0)
    duplicate_pressure = 0.25 if metadata.get("duplicate_candidate_signatures") else 0.0
    score = correctness_bonus + abs(learned_rr - selector_rr) + 0.35 * min(margin_evidence, 1.0)
    score += 0.25 * candidate_pressure + duplicate_pressure
    return direction * score


def _case_metadata(query: Query) -> dict[str, Any]:
    allowed = (
        "app",
        "category",
        "widget_type",
        "source_screenshot_path",
        "target_screenshot_path",
        "mapping_cardinality",
        "duplicate_candidate_signatures",
        "cross_form_factor",
        "web_or_webview_context",
    )
    return {key: query.metadata[key] for key in allowed if key in query.metadata}


def _source_point(query: Query) -> tuple[float, float] | None:
    try:
        return float(query.source["x"]), float(query.source["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _candidate_points(
    query: Query,
    candidate_ids: tuple[str, ...],
) -> tuple[tuple[float, float], ...]:
    points = [
        point
        for candidate_id in candidate_ids
        if (point := _candidate_point(query, candidate_id)) is not None
    ]
    return tuple(dict.fromkeys(points))


def _candidate_point(
    query: Query,
    candidate_id: str | None,
) -> tuple[float, float] | None:
    if candidate_id is None:
        return None
    candidate = next(
        (
            candidate
            for candidate in query.target_candidates
            if candidate.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None or candidate.bbox is None:
        return None
    x1, y1, x2, y2 = candidate.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
