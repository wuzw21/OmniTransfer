"""Verified outcome preferences for the single learned UI matcher."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.schema import Query


_NULL_IDS = {"", "null", "none", "nil", "no_match", "abstain", "__null__"}


@dataclass(frozen=True)
class OutcomePreference:
    """One verified winner-loser target preference for a query."""

    query_id: str
    winner_candidate_id: str | None
    loser_candidate_id: str | None
    weight: float = 1.0


def load_outcome_preferences(
    path: str | Path,
    queries: Iterable[Query],
) -> dict[str, tuple[OutcomePreference, ...]]:
    """Convert validator outcomes or explicit pairs into ranking preferences."""

    query_by_id = {query.query_id: query for query in queries}
    explicit: list[OutcomePreference] = []
    attempts: dict[str, dict[str | None, set[bool]]] = defaultdict(
        lambda: defaultdict(set)
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "")
            query = query_by_id.get(query_id)
            if query is None:
                raise ValueError(
                    f"outcome row {line_number} references unknown query {query_id!r}"
                )
            if row.get("environment_failure") is True:
                continue
            if "winner_candidate_id" in row or "loser_candidate_id" in row:
                preference = OutcomePreference(
                    query_id=query_id,
                    winner_candidate_id=_candidate_id(row.get("winner_candidate_id")),
                    loser_candidate_id=_candidate_id(row.get("loser_candidate_id")),
                    weight=float(row.get("weight") or 1.0),
                )
                _validate_preference(preference, query, line_number=line_number)
                explicit.append(preference)
                continue
            if not isinstance(row.get("verified_success"), bool):
                raise ValueError(
                    f"outcome row {line_number} requires boolean verified_success"
                )
            candidate_id = _candidate_id(row.get("candidate_id"))
            _validate_candidate(candidate_id, query, line_number=line_number)
            attempts[query_id][candidate_id].add(bool(row["verified_success"]))

    preferences = list(explicit)
    for query_id, candidate_outcomes in attempts.items():
        stable = {
            candidate_id: next(iter(outcomes))
            for candidate_id, outcomes in candidate_outcomes.items()
            if len(outcomes) == 1
        }
        winners = sorted(
            (candidate_id for candidate_id, success in stable.items() if success),
            key=_candidate_sort_key,
        )
        losers = sorted(
            (candidate_id for candidate_id, success in stable.items() if not success),
            key=_candidate_sort_key,
        )
        if winners:
            comparison_losers = losers or [None]
            preferences.extend(
                OutcomePreference(query_id, winner, loser)
                for winner in winners
                for loser in comparison_losers
                if winner != loser
            )
        else:
            preferences.extend(
                OutcomePreference(query_id, None, loser)
                for loser in losers
                if loser is not None
            )

    grouped: dict[str, list[OutcomePreference]] = defaultdict(list)
    seen: set[tuple[str, str | None, str | None]] = set()
    for preference in preferences:
        key = (
            preference.query_id,
            preference.winner_candidate_id,
            preference.loser_candidate_id,
        )
        if key in seen:
            continue
        seen.add(key)
        grouped[preference.query_id].append(preference)
    return {query_id: tuple(rows) for query_id, rows in grouped.items()}


def preference_ranking_loss(
    logits: Any,
    candidate_ids: tuple[str, ...],
    preferences: Iterable[OutcomePreference],
) -> Any | None:
    """Apply a weighted log-sigmoid winner-loser loss to matcher logits."""

    torch = _require_torch()
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    index_by_id[None] = len(candidate_ids)
    losses: list[Any] = []
    weights: list[float] = []
    for preference in preferences:
        winner_index = index_by_id.get(preference.winner_candidate_id)
        loser_index = index_by_id.get(preference.loser_candidate_id)
        if winner_index is None or loser_index is None:
            raise ValueError("outcome preference references a missing candidate")
        losses.append(
            -torch.nn.functional.logsigmoid(
                logits[winner_index] - logits[loser_index]
            )
        )
        weights.append(max(float(preference.weight), 0.0))
    if not losses:
        return None
    weight_tensor = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)


def _validate_preference(
    preference: OutcomePreference,
    query: Query,
    *,
    line_number: int,
) -> None:
    if preference.winner_candidate_id == preference.loser_candidate_id:
        raise ValueError(f"outcome row {line_number} has identical winner and loser")
    if preference.weight <= 0.0:
        raise ValueError(f"outcome row {line_number} has non-positive weight")
    _validate_candidate(preference.winner_candidate_id, query, line_number=line_number)
    _validate_candidate(preference.loser_candidate_id, query, line_number=line_number)


def _validate_candidate(
    candidate_id: str | None,
    query: Query,
    *,
    line_number: int,
) -> None:
    candidate_ids = {candidate.candidate_id for candidate in query.target_candidates}
    if candidate_id is not None and candidate_id not in candidate_ids:
        raise ValueError(
            f"outcome row {line_number} references missing candidate {candidate_id!r}"
        )


def _candidate_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return None if normalized.lower() in _NULL_IDS else normalized


def _candidate_sort_key(candidate_id: str | None) -> tuple[int, str]:
    return (candidate_id is None, candidate_id or "")


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for outcome preference learning. "
            "Install omnitransfer[train]."
        ) from exc
    return torch
