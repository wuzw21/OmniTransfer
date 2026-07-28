"""Hard-case scoring and balanced selection for mapping acceptance review."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable


@dataclass(frozen=True)
class HardMappingCandidate:
    """One source-node mapping query with method diagnostics."""

    task_id: str
    pair_id: str
    app: str
    semantic_key: str
    matcher_correct: bool
    selector_correct: bool
    selector_abstained: bool
    methods_disagree: bool
    matcher_margin: float
    matcher_probability: float
    position_shift: float
    target_area_fraction: float
    textless: bool
    sequence_transition: bool
    payload: dict[str, Any]


def score_hard_mapping_candidate(
    candidate: HardMappingCandidate,
) -> tuple[float, list[str]]:
    """Score rule-resistant mapping cases without using selector as model input."""

    score = 0.0
    reasons: list[str] = []
    if not candidate.matcher_correct:
        score += 4.0
        reasons.append("matcher_top1_error")
    if candidate.selector_abstained:
        score += 3.5
        reasons.append("selector_abstains")
    elif not candidate.selector_correct:
        score += 3.0
        reasons.append("selector_top1_error")
    if candidate.methods_disagree:
        score += 2.0
        reasons.append("matcher_selector_disagreement")
    if candidate.matcher_margin < 0.15:
        score += 1.5 * (1.0 - candidate.matcher_margin / 0.15)
        reasons.append("low_matcher_margin")
    if candidate.matcher_probability < 0.60:
        score += 1.0 * (1.0 - candidate.matcher_probability / 0.60)
        reasons.append("low_match_probability")
    if candidate.position_shift > 0.20:
        score += min(1.0, candidate.position_shift)
        reasons.append("large_position_shift")
    if candidate.target_area_fraction < 0.01:
        score += 0.75
        reasons.append("small_target")
    if candidate.textless:
        score += 0.75
        reasons.append("textless_control")
    if candidate.sequence_transition:
        score += 0.5
        reasons.append("trace_state_transition")
    return round(score, 6), reasons


def select_balanced_hard_candidates(
    candidates: Iterable[HardMappingCandidate],
    *,
    task_limit: int,
    max_tasks_per_app: int,
    max_tasks_per_pair: int,
    max_tasks_per_semantic: int = 20,
    minimum_difficulty: float = 0.0,
) -> list[HardMappingCandidate]:
    """Select the highest-scoring tasks under diversity caps."""

    if (
        min(task_limit, max_tasks_per_app, max_tasks_per_pair, max_tasks_per_semantic)
        <= 0
    ):
        raise ValueError("selection limits must be positive")
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.payload["difficulty_score"]),
            _stable_key(item.task_id),
        ),
    )
    app_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    selected: list[HardMappingCandidate] = []
    for candidate in ranked:
        if float(candidate.payload["difficulty_score"]) < minimum_difficulty:
            continue
        if app_counts[candidate.app] >= max_tasks_per_app:
            continue
        if pair_counts[candidate.pair_id] >= max_tasks_per_pair:
            continue
        if semantic_counts[candidate.semantic_key] >= max_tasks_per_semantic:
            continue
        selected.append(candidate)
        app_counts[candidate.app] += 1
        pair_counts[candidate.pair_id] += 1
        semantic_counts[candidate.semantic_key] += 1
        if len(selected) >= task_limit:
            break
    return selected


def _stable_key(value: str) -> str:
    return hashlib.blake2b(value.encode(), digest_size=12).hexdigest()
