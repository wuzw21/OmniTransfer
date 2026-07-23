"""Page-pair review records for cross-platform widget mapping datasets."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.schema import Candidate, Query


SCHEMA_VERSION = "omnitransfer_widget_mapping_pair_review_v1"


def build_widget_mapping_pair_review(
    queries: Iterable[Query],
    *,
    input_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group per-anchor queries into screen pairs with multiple correspondences."""

    input_file = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    grouped: dict[tuple[str, str, str, str], list[Query]] = defaultdict(list)
    query_count = 0
    for query in queries:
        query_count += 1
        key = (
            str(query.metadata.get("source_screen") or ""),
            str(query.metadata.get("target_screen") or ""),
            str(query.metadata.get("source_screenshot_path") or ""),
            str(query.metadata.get("target_screenshot_path") or ""),
        )
        grouped[key].append(query)

    rows = [
        _pair_record(pair_queries, input_path=input_file, output_dir=output)
        for _, pair_queries in sorted(grouped.items())
    ]
    rows.sort(key=lambda row: (row["app"].lower(), row["source"]["screen"], row["pair_id"]))
    correspondence_count = sum(len(row["correspondences"]) for row in rows)
    alias_groups = sum(row["quality"]["target_alias_groups"] for row in rows)
    missing_images = sum(
        int(not row[side]["image_exists"])
        for row in rows
        for side in ("source", "target")
    )
    manifest = {
        "schema_version": "omnitransfer_widget_mapping_pair_review_manifest_v1",
        "review_schema_version": SCHEMA_VERSION,
        "input": str(input_file),
        "queries": query_count,
        "screen_pairs": len(rows),
        "correspondences": correspondence_count,
        "target_alias_groups": alias_groups,
        "missing_images": missing_images,
        "policy": {
            "review_unit": "source_target_screen_pair",
            "multiple_correspondences_per_pair": True,
            "rule_labels_for_training": False,
            "coordinate_passthrough": False,
            "human_can_replace_gold_candidates": True,
        },
    }
    return rows, manifest


def _pair_record(
    queries: list[Query],
    *,
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    first = queries[0]
    source_path = _resolve_asset_path(
        str(first.metadata.get("source_screenshot_path") or ""),
        input_path=input_path,
    )
    target_path = _resolve_asset_path(
        str(first.metadata.get("target_screenshot_path") or ""),
        input_path=input_path,
    )
    pair_key = "\n".join(
        (
            str(first.metadata.get("source_screen") or source_path),
            str(first.metadata.get("target_screen") or target_path),
        )
    )
    pair_id = f"widget_pair_{hashlib.sha256(pair_key.encode()).hexdigest()[:16]}"
    candidates = _merged_candidates(queries)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    correspondences = [
        _correspondence(query, candidate_by_id=candidate_by_id)
        for query in sorted(queries, key=lambda query: query.query_id)
    ]
    bbox_groups: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if bbox is not None:
            bbox_groups[tuple(bbox)].append(candidate["candidate_id"])
    alias_groups = [
        {"bbox": list(bbox), "candidate_ids": sorted(candidate_ids)}
        for bbox, candidate_ids in bbox_groups.items()
        if len(candidate_ids) > 1
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "app": str(first.metadata.get("app") or "unknown"),
        "category": str(first.metadata.get("category") or ""),
        "source": _screen_record(
            side="source",
            query=first,
            image_path=source_path,
            output_dir=output_dir,
        ),
        "target": _screen_record(
            side="target",
            query=first,
            image_path=target_path,
            output_dir=output_dir,
        ),
        "target_candidates": candidates,
        "correspondences": correspondences,
        "quality": {
            "query_count": len(queries),
            "target_candidate_count": len(candidates),
            "target_alias_groups": len(alias_groups),
            "target_aliases": alias_groups,
            "source_node_match": sorted(
                {
                    str((query.source.get("metadata") or {}).get("node_match") or "unknown")
                    for query in queries
                }
            ),
        },
        "annotation": {
            "pair_status": None,
            "notes": "",
            "mapping_overrides": {},
        },
    }


def _screen_record(
    *,
    side: str,
    query: Query,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    width = query.metadata.get(f"{side}_coordinate_width")
    height = query.metadata.get(f"{side}_coordinate_height")
    return {
        "screen": str(query.metadata.get(f"{side}_screen") or ""),
        "platform": str(
            (query.source.get("metadata") or {}).get("platform")
            if side == "source"
            else "android"
        ),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "image_path": str(image_path),
        "image_url": os.path.relpath(image_path, output_dir).replace(os.sep, "/"),
        "image_exists": image_path.is_file(),
        "xml_path": str(query.metadata.get(f"{side}_xml_path") or ""),
    }


def _merged_candidates(queries: list[Query]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for query in queries:
        for candidate in query.target_candidates:
            row = _candidate_record(candidate)
            existing = candidates.get(candidate.candidate_id)
            if existing is not None and existing.get("bbox") != row.get("bbox"):
                raise ValueError(
                    f"candidate {candidate.candidate_id} has inconsistent boxes within one pair"
                )
            candidates[candidate.candidate_id] = row
    return sorted(
        candidates.values(),
        key=lambda row: (
            tuple(row["bbox"] or (float("inf"),) * 4),
            row["candidate_id"],
        ),
    )


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    metadata = candidate.metadata
    return {
        "candidate_id": candidate.candidate_id,
        "bbox": list(candidate.bbox) if candidate.bbox else None,
        "text": str(metadata.get("text") or ""),
        "content_desc": str(metadata.get("content_desc") or ""),
        "resource_id": str(metadata.get("resource_id") or ""),
        "class_name": str(metadata.get("class_name") or metadata.get("node_tag") or ""),
        "clickable": bool(metadata.get("clickable")),
        "candidate_source": str(metadata.get("candidate_source") or ""),
        "node_index": metadata.get("node_index"),
    }


def _correspondence(
    query: Query,
    *,
    candidate_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_bbox = _bbox(query.source.get("bounds") or query.source.get("bbox"))
    point = _point(query.source, source_bbox)
    gold_ids = list(query.acceptable_gold_candidate_ids())
    gold_candidates = [candidate_by_id[candidate_id] for candidate_id in gold_ids]
    gold_boxes = []
    seen_boxes: set[tuple[float, float, float, float]] = set()
    for candidate in gold_candidates:
        bbox = candidate.get("bbox")
        if bbox is None or tuple(bbox) in seen_boxes:
            continue
        seen_boxes.add(tuple(bbox))
        gold_boxes.append(bbox)
    return {
        "mapping_id": query.query_id,
        "source": {
            "bbox": list(source_bbox) if source_bbox else None,
            "point": list(point) if point else None,
            "text": str(query.source.get("text") or ""),
            "content_desc": str(query.source.get("content_desc") or ""),
            "resource_id": str(query.source.get("resource_id") or ""),
            "class_name": str(query.source.get("class_name") or ""),
            "action_type": str(query.source.get("action_type") or "click"),
            "widget_type": str(query.metadata.get("widget_type") or ""),
        },
        "gold_candidate_ids": gold_ids,
        "gold_bboxes": gold_boxes,
        "annotation": {
            "status": None,
            "corrected_candidate_ids": None,
            "notes": "",
        },
    }


def _resolve_asset_path(value: str, *, input_path: Path) -> Path:
    if not value:
        return Path("")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        input_path.parent / path,
        input_path.parent.parent / path,
        Path.cwd() / path,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = map(float, value)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _point(
    source: dict[str, Any],
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    try:
        if source.get("x") is not None and source.get("y") is not None:
            return float(source["x"]), float(source["y"])
    except (TypeError, ValueError):
        pass
    if bbox is None:
        return None
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
