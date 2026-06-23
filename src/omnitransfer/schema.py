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


@dataclass(frozen=True)
class Prediction:
    """Ranker output for one relocation query."""

    query_id: str
    selected_candidate_id: str | None
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
