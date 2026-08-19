"""Deterministic hard-case selection for text-less mobile icons."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair

ICON_HARD_SET_SCHEMA = "omnitransfer.icon_hard_set.v1"


def build_icon_hard_records(
    records: Iterable[dict[str, Any]],
    *,
    limit: int = 0,
    minimum_score: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return canonical records containing only difficult text-less icon labels."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if minimum_score < 0.0:
        raise ValueError("minimum_score must be non-negative")
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    input_pairs = 0
    input_matches = 0
    for raw_record in records:
        record = validate_ui_correspondence_pair(raw_record)
        input_pairs += 1
        input_matches += len(record["matches"])
        hard_record = _hard_record(record)
        if hard_record is None:
            continue
        score = float(hard_record["provenance"]["hard_set"]["score"])
        if score < minimum_score:
            continue
        ranked.append((score, hard_record["pair_id"], hard_record))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if limit:
        ranked = ranked[:limit]
    selected = [item[2] for item in ranked]
    reason_counts: dict[str, int] = {}
    selected_icons = 0
    for record in selected:
        hard_set = record["provenance"]["hard_set"]
        selected_icons += len(hard_set["source_items"])
        for item in hard_set["source_items"]:
            for reason in item["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return selected, {
        "schema_version": ICON_HARD_SET_SCHEMA,
        "input_pairs": input_pairs,
        "input_matches": input_matches,
        "selected_pairs": len(selected),
        "selected_icons": selected_icons,
        "selected_matches": sum(len(record["matches"]) for record in selected),
        "minimum_score": minimum_score,
        "limit": limit,
        "reason_counts": dict(sorted(reason_counts.items())),
        "selection_boundary": {
            "model_scores_used": False,
            "resource_ids_used_for_selection": False,
            "textless_source_required": True,
            "canonical_pair_schema_preserved": True,
        },
    }


def is_small_visual_control(
    node: dict[str, Any],
    graph: dict[str, Any],
    *,
    require_textless: bool,
) -> bool:
    """Identify a bounded icon-sized control without using identity fields."""

    if require_textless and (
        str(node.get("text") or "").strip()
        or str(node.get("content_desc") or "").strip()
    ):
        return False
    geometry = _normalized_bbox(node, graph)
    if geometry is None:
        return False
    left, top, right, bottom = geometry
    width = right - left
    height = bottom - top
    area = width * height
    aspect_ratio = max(width, height) / max(min(width, height), 1e-9)
    if area <= 0.0 or area > 0.05 or max(width, height) > 0.40 or aspect_ratio > 6.0:
        return False
    class_name = str(node.get("class_name") or node.get("class") or "").lower()
    container_tokens = ("viewgroup", "layout", "frame", "application", "window", "root")
    icon_tokens = ("image", "icon", "button", "control", "switch", "checkbox", "radio")
    return any(token in class_name for token in icon_tokens) or (
        bool(node.get("clickable"))
        and not any(token in class_name for token in container_tokens)
    )


def _hard_record(record: dict[str, Any]) -> dict[str, Any] | None:
    source_graph = record["source"]["graph"]
    target_graph = record["target"]["graph"]
    source_nodes = {str(node["node_id"]): node for node in source_graph["nodes"]}
    target_nodes = {str(node["node_id"]): node for node in target_graph["nodes"]}
    source_icons = {
        node_id
        for node_id, node in source_nodes.items()
        if is_small_visual_control(node, source_graph, require_textless=True)
    }
    target_icons = {
        node_id
        for node_id, node in target_nodes.items()
        if is_small_visual_control(node, target_graph, require_textless=False)
    }
    selected_matches = [
        deepcopy(match)
        for match in record["matches"]
        if match["label"] == "correspondence"
        and str(match["source_node_id"]) in source_icons
        and any(str(target_id) in target_icons for target_id in match["target_node_ids"])
    ]
    if not selected_matches:
        return None
    source_parent_counts = _parent_icon_counts(source_nodes, source_icons)
    target_parent_counts = _parent_icon_counts(target_nodes, target_icons)
    source_rows, source_columns = _alignment_counts(source_nodes, source_icons, source_graph)
    target_rows, target_columns = _alignment_counts(target_nodes, target_icons, target_graph)
    source_items = []
    for match in selected_matches:
        source_id = str(match["source_node_id"])
        source_node = source_nodes[source_id]
        targets = [
            target_nodes[str(target_id)]
            for target_id in match["target_node_ids"]
            if str(target_id) in target_nodes
        ]
        score, reasons = _icon_difficulty(
            source_node,
            targets,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            source_graph=source_graph,
            target_graph=target_graph,
            source_icon_count=len(source_icons),
            target_icon_count=len(target_icons),
            source_parent_counts=source_parent_counts,
            target_parent_counts=target_parent_counts,
            source_rows=source_rows,
            source_columns=source_columns,
            target_rows=target_rows,
            target_columns=target_columns,
        )
        source_items.append(
            {
                "source_node_id": source_id,
                "score": score,
                "reasons": reasons,
                "target_node_ids": [str(value) for value in match["target_node_ids"]],
            }
        )
    source_items.sort(key=lambda item: (-item["score"], item["source_node_id"]))
    result = deepcopy(record)
    result["matches"] = selected_matches
    hard_score = max(item["score"] for item in source_items)
    result["provenance"] = {
        **result["provenance"],
        "difficulty_score": hard_score,
        "hard_set": {
            "schema_version": ICON_HARD_SET_SCHEMA,
            "score": hard_score,
            "source_icon_candidates": len(source_icons),
            "target_icon_candidates": len(target_icons),
            "source_items": source_items,
        },
    }
    result["slices"] = {**result["slices"], "hard_set": "textless_icon_v1"}
    return validate_ui_correspondence_pair(result)


def _icon_difficulty(
    source: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    source_nodes: dict[str, dict[str, Any]],
    target_nodes: dict[str, dict[str, Any]],
    source_graph: dict[str, Any],
    target_graph: dict[str, Any],
    source_icon_count: int,
    target_icon_count: int,
    source_parent_counts: dict[str, int],
    target_parent_counts: dict[str, int],
    source_rows: dict[str, int],
    source_columns: dict[str, int],
    target_rows: dict[str, int],
    target_columns: dict[str, int],
) -> tuple[float, list[str]]:
    score = 1.0
    reasons = ["textless_source_icon"]
    source_id = str(source["node_id"])
    source_bbox = _normalized_bbox(source, source_graph)
    if source_bbox is not None and _bbox_area(source_bbox) <= 0.0025:
        score += 1.0
        reasons.append("tiny_icon")
    if max(source_icon_count, target_icon_count) >= 8:
        score += 1.0
        reasons.append("icon_crowded_page")
    if _has_ancestor_class(source, source_nodes, "webview"):
        score += 1.5
        reasons.append("webview_source")
    source_parent = str(source.get("parent_id") or "")
    if source_parent_counts.get(source_parent, 0) >= 3:
        score += 1.0
        reasons.append("repeated_sibling_slot")
    if source_rows.get(source_id, 0) >= 3 or source_columns.get(source_id, 0) >= 3:
        score += 0.75
        reasons.append("same_row_or_column_icons")
    if not str(source.get("resource_id") or "").strip():
        score += 0.5
        reasons.append("no_resource_id")
    if len(targets) > 1:
        score += 0.5
        reasons.append("multiple_valid_targets")
    if any(str(target.get("text") or target.get("content_desc") or "").strip() for target in targets):
        score += 0.5
        reasons.append("cross_platform_semantic_asymmetry")
    if targets and not any(_same_control_family(source, target) for target in targets):
        score += 0.5
        reasons.append("cross_platform_class_mismatch")
    if targets and _position_shift(source, targets[0], source_graph, target_graph) > 0.20:
        score += 0.5
        reasons.append("large_layout_displacement")
    if any(_state_differs(source, target) for target in targets):
        score += 1.0
        reasons.append("state_sensitive")
    for target in targets:
        target_id = str(target["node_id"])
        target_parent = str(target.get("parent_id") or "")
        if target_parent_counts.get(target_parent, 0) >= 3:
            score += 0.25
            reasons.append("target_repeated_sibling_slot")
            break
        if target_rows.get(target_id, 0) >= 3 or target_columns.get(target_id, 0) >= 3:
            score += 0.25
            reasons.append("target_same_row_or_column_icons")
            break
    return score, reasons


def _normalized_bbox(
    node: dict[str, Any],
    graph: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    value = node.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
        width = float(graph.get("width") or 1.0)
        height = float(graph.get("height") or 1.0)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0 or right <= left or bottom <= top:
        return None
    normalized = (left / width, top / height, right / width, bottom / height)
    if any(value < 0.0 or value > 1.0 for value in normalized):
        return None
    return normalized


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _parent_icon_counts(
    nodes: dict[str, dict[str, Any]],
    icon_ids: set[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_id in icon_ids:
        parent_id = str(nodes[node_id].get("parent_id") or "")
        counts[parent_id] = counts.get(parent_id, 0) + 1
    return counts


def _alignment_counts(
    nodes: dict[str, dict[str, Any]],
    icon_ids: set[str],
    graph: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    centers = {}
    for node_id in icon_ids:
        bbox = _normalized_bbox(nodes[node_id], graph)
        if bbox is not None:
            centers[node_id] = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    rows = {
        node_id: sum(abs(center[1] - other[1]) <= 0.03 for other in centers.values())
        for node_id, center in centers.items()
    }
    columns = {
        node_id: sum(abs(center[0] - other[0]) <= 0.03 for other in centers.values())
        for node_id, center in centers.items()
    }
    return rows, columns


def _has_ancestor_class(
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    token: str,
) -> bool:
    parent_id = str(node.get("parent_id") or "")
    visited = set()
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent = nodes.get(parent_id)
        if parent is None:
            return False
        class_name = str(parent.get("class_name") or parent.get("class") or "").lower()
        if token in class_name:
            return True
        parent_id = str(parent.get("parent_id") or "")
    return False


def _same_control_family(first: dict[str, Any], second: dict[str, Any]) -> bool:
    families = ("image", "button", "switch", "checkbox", "radio", "icon")
    first_class = str(first.get("class_name") or "").lower()
    second_class = str(second.get("class_name") or "").lower()
    return any(token in first_class and token in second_class for token in families)


def _position_shift(
    source: dict[str, Any],
    target: dict[str, Any],
    source_graph: dict[str, Any],
    target_graph: dict[str, Any],
) -> float:
    source_bbox = _normalized_bbox(source, source_graph)
    target_bbox = _normalized_bbox(target, target_graph)
    if source_bbox is None or target_bbox is None:
        return 0.0
    source_center = ((source_bbox[0] + source_bbox[2]) / 2.0, (source_bbox[1] + source_bbox[3]) / 2.0)
    target_center = ((target_bbox[0] + target_bbox[2]) / 2.0, (target_bbox[1] + target_bbox[3]) / 2.0)
    return abs(source_center[0] - target_center[0]) + abs(source_center[1] - target_center[1])


def _state_differs(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_metadata = first.get("metadata") if isinstance(first.get("metadata"), dict) else {}
    second_metadata = second.get("metadata") if isinstance(second.get("metadata"), dict) else {}
    for key in ("selected", "checked", "activated", "focused"):
        if key in first_metadata and key in second_metadata and first_metadata[key] != second_metadata[key]:
            return True
    return False
