"""Stable replay-time action transfer interface."""

from __future__ import annotations

from typing import Any

from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record


def action_transfer(
    *,
    source_xml: str,
    target_xml: str,
    source_point: tuple[float, float] | None = None,
    source_element_id: str | None = None,
    action_type: str = "click",
    top_k: int = 1,
    history: Any = None,
) -> dict[str, Any]:
    """Relocate one recorded source target onto the current UI tree."""

    source = graph_from_record({"xml": source_xml}, graph_id="source")
    target = graph_from_record({"xml": target_xml}, graph_id="target")
    source_node = _source_node(source, source_point, source_element_id)
    if source_node is None or source_node.bbox is None:
        return {
            "mapped": False,
            "mapping_mode": "missing_source_target",
            "reason": "source_point_or_element_id_required",
        }
    point = source_point or _center(source_node.bbox)
    ranked = sorted(
        (
            (_score(source_node, candidate, action_type), candidate)
            for candidate in target.nodes
            if candidate.bbox is not None and candidate.enabled
        ),
        key=lambda item: (-item[0], item[1].node_id),
    )
    ranked = [item for item in ranked if item[0] > 0]
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return {
            "mapped": False,
            "mapping_mode": "ambiguous" if ranked else "absent",
            "reason": "target_identity_not_unique",
        }
    score, target_node = ranked[0]
    target_point = _project(point, source_node.bbox, target_node.bbox)
    candidates = [
        {
            "candidate_id": candidate.node_id,
            "bbox": list(candidate.bbox or ()),
            "score": candidate_score,
        }
        for candidate_score, candidate in ranked[: max(1, int(top_k))]
    ]
    return {
        "mapped": True,
        "mapping_mode": "element_identity",
        "new_x": target_point[0],
        "new_y": target_point[1],
        "src_element": _node_dict(source_node),
        "target_candidate_id": target_node.node_id,
        "target_bbox": list(target_node.bbox),
        "target_center": list(_center(target_node.bbox)),
        "score": score,
        "margin": score - (ranked[1][0] if len(ranked) > 1 else 0.0),
        "top_candidates": candidates,
        "action_type": action_type,
    }


def _source_node(
    graph: UIGraph,
    point: tuple[float, float] | None,
    element_id: str | None,
) -> UINode | None:
    normalized = str(element_id or "").strip()
    if normalized:
        return next(
            (
                node
                for node in graph.nodes
                if normalized in {node.node_id, node.origin_id, node.resource_id}
            ),
            None,
        )
    if point is None:
        return None
    x, y = point
    containing = [
        node
        for node in graph.nodes
        if node.bbox is not None
        and node.bbox[0] <= x <= node.bbox[2]
        and node.bbox[1] <= y <= node.bbox[3]
    ]
    return min(containing, key=lambda node: _area(node.bbox), default=None)


def _score(source: UINode, target: UINode, action_type: str) -> float:
    score = 0.0
    stable_identity = False
    if source.resource_id and source.resource_id == target.resource_id:
        score += 8.0
        stable_identity = True
    elif _tail(source.resource_id) and _tail(source.resource_id) == _tail(target.resource_id):
        score += 6.0
        stable_identity = True
    for source_value, target_value in (
        (source.text, target.text),
        (source.content_desc, target.content_desc),
    ):
        if source_value and _text(source_value) == _text(target_value):
            score += 4.0
            stable_identity = True
    if not stable_identity:
        root_scroll = action_type in {"swipe", "scroll"} and source.parent_id is None
        if not root_scroll:
            return 0.0
    if source.class_name and _tail(source.class_name) == _tail(target.class_name):
        score += 1.0
    score += 0.5 * sum(
        source_value == target_value
        for source_value, target_value in (
            (source.clickable, target.clickable),
            (source.editable, target.editable),
            (source.scrollable, target.scrollable),
        )
    )
    return score


def _project(point: tuple[float, float], source: tuple[float, float, float, float], target: tuple[float, float, float, float]) -> tuple[float, float]:
    source_width = max(1.0, source[2] - source[0])
    source_height = max(1.0, source[3] - source[1])
    relative_x = max(0.0, min(1.0, (point[0] - source[0]) / source_width))
    relative_y = max(0.0, min(1.0, (point[1] - source[1]) / source_height))
    return (
        target[0] + relative_x * (target[2] - target[0]),
        target[1] + relative_y * (target[3] - target[1]),
    )


def _node_dict(node: UINode) -> dict[str, Any]:
    return {
        "id": node.node_id,
        "resource_id": node.resource_id,
        "class": node.class_name,
        "bounds": list(node.bbox or ()),
        "clickable": node.clickable,
        "editable": node.editable,
        "scrollable": node.scrollable,
    }


def _center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0


def _area(bounds: tuple[float, float, float, float] | None) -> float:
    return float("inf") if bounds is None else (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _tail(value: str) -> str:
    return _text(value).rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
