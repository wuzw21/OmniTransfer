"""Canonical JSONL importers for source-to-target UI grounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.schema import Candidate, Query


_NULL_IDS = {"", "null", "none", "nil", "no_match", "abstain"}


def load_queries(path: str | Path) -> list[Query]:
    """Load canonical or legacy-compatible OmniTransfer JSONL queries."""

    input_path = Path(path)
    queries: list[Query] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            queries.append(query_from_dict(row, index=index, source_path=input_path))
    return queries


def query_from_dict(
    row: dict[str, Any],
    *,
    index: int = 0,
    source_path: str | Path | None = None,
) -> Query:
    """Normalize one action-transfer record without requiring one-to-one labels."""

    raw_candidates = row.get("target_candidates") or row.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raise ValueError(f"query row {index} has non-list target_candidates")
    candidates = tuple(
        _candidate_from_dict(candidate, index=candidate_index)
        for candidate_index, candidate in enumerate(raw_candidates)
    )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"query row {index} has duplicate candidate ids")
    gold_candidate_id = _normalize_gold_id(
        row.get("gold_candidate_id", row.get("gold_id", row.get("target_id")))
    )
    metadata = dict(row.get("metadata") or row.get("meta") or {})
    for key in ("dataset", "split", "label_status"):
        if row.get(key) not in (None, ""):
            metadata.setdefault(key, row[key])
    if source_path is not None:
        metadata.setdefault("canonical_jsonl_path", str(Path(source_path).resolve()))
    equivalents = _equivalent_gold_ids(
        candidates,
        gold_candidate_id=gold_candidate_id,
        explicit=metadata.get("gold_equivalent_candidate_ids"),
    )
    if equivalents:
        metadata["gold_equivalent_candidate_ids"] = list(equivalents)
    return Query(
        query_id=str(row.get("query_id") or row.get("case_id") or row.get("id") or f"q_{index}"),
        source=dict(row.get("source") or row.get("source_action") or row.get("query") or {}),
        target_candidates=candidates,
        gold_candidate_id=gold_candidate_id,
        metadata=metadata,
    )


def write_queries(queries: Iterable[Query], path: str | Path) -> None:
    """Write the stable lightweight query schema as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "source": query.source,
                        "target_candidates": [
                            {
                                "candidate_id": candidate.candidate_id,
                                "bbox": list(candidate.bbox) if candidate.bbox else None,
                                **candidate.metadata,
                            }
                            for candidate in query.target_candidates
                        ],
                        "gold_candidate_id": query.gold_candidate_id,
                        "metadata": query.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _candidate_from_dict(row: Any, *, index: int) -> Candidate:
    if not isinstance(row, dict):
        raise ValueError(f"candidate {index} is not an object")
    candidate_id = str(
        row.get("candidate_id") or row.get("node_id") or row.get("id") or f"candidate_{index}"
    )
    bbox = _bbox(row.get("bbox", row.get("bounds", row.get("normalized_bbox"))))
    metadata = dict(row)
    metadata.pop("candidate_id", None)
    metadata.pop("node_id", None)
    metadata.pop("id", None)
    metadata.pop("bbox", None)
    metadata.pop("bounds", None)
    nested = metadata.pop("metadata", None)
    if isinstance(nested, dict):
        metadata = {**nested, **metadata}
    return Candidate(candidate_id=candidate_id, bbox=bbox, metadata=metadata)


def _equivalent_gold_ids(
    candidates: tuple[Candidate, ...],
    *,
    gold_candidate_id: str | None,
    explicit: Any,
) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(explicit, str):
        values.append(explicit)
    elif isinstance(explicit, (list, tuple, set)):
        values.extend(str(value) for value in explicit if str(value))
    if gold_candidate_id:
        values.append(gold_candidate_id)
        gold = next(
            (candidate for candidate in candidates if candidate.candidate_id == gold_candidate_id),
            None,
        )
        if gold is not None and gold.bbox is not None:
            values.extend(
                candidate.candidate_id
                for candidate in candidates
                if candidate.bbox == gold.bbox
            )
    return tuple(dict.fromkeys(values))


def _normalize_gold_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return None if normalized.lower() in _NULL_IDS else normalized


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = map(float, value)
    except (TypeError, ValueError):
        return None
    return (left, top, right, bottom) if right > left and bottom > top else None
