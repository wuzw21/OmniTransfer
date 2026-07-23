"""Core data structures for UI grounding relocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One target UI grounding candidate."""

    candidate_id: str
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    """One source-grounding to target-screen relocation query."""

    query_id: str
    source: dict[str, Any]
    target_candidates: tuple[Candidate, ...]
    gold_candidate_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def acceptable_gold_candidate_ids(self) -> tuple[str, ...]:
        """Return every semantically equivalent gold target for set-valued eval."""

        values = self.metadata.get("gold_equivalent_candidate_ids") or ()
        if isinstance(values, str):
            values = (values,)
        normalized = [str(value) for value in values if str(value)]
        if self.gold_candidate_id:
            normalized.append(self.gold_candidate_id)
        return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True)
class Prediction:
    """Ranker output for one relocation query."""

    query_id: str
    selected_candidate_id: str | None
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def ranked_candidate_ids(self) -> tuple[str, ...]:
        """Return concrete candidates ordered by score and stable id."""

        ranked = sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))
        candidate_ids = [candidate_id for candidate_id, _ in ranked]
        if self.selected_candidate_id and self.selected_candidate_id not in candidate_ids:
            candidate_ids.insert(0, self.selected_candidate_id)
        return tuple(candidate_ids)
