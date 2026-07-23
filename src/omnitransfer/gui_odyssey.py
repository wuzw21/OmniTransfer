"""Weak GUIOdyssey adapters for action-grounding distillation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omnitransfer.ui_graph import BBox, UIGraph, UINode


REGION_FRACTIONS: tuple[tuple[str, BBox], ...] = (
    ("screen", (0.0, 0.0, 1.0, 1.0)),
    ("status_bar", (0.0, 0.0, 1.0, 0.08)),
    ("content", (0.0, 0.08, 1.0, 0.92)),
    ("nav_bar", (0.0, 0.92, 1.0, 1.0)),
    ("top_left", (0.0, 0.08, 0.5, 0.5)),
    ("top_right", (0.5, 0.08, 1.0, 0.5)),
    ("bottom_left", (0.0, 0.5, 0.5, 0.92)),
    ("bottom_right", (0.5, 0.5, 1.0, 0.92)),
)


def graph_from_gui_odyssey_step(
    episode: dict[str, Any],
    step: dict[str, Any],
    *,
    graph_id: str | None = None,
) -> UIGraph:
    """Convert one GUIOdyssey action step into a weak pseudo UI graph.

    GUIOdyssey does not publish UI trees. For matcher pretraining we therefore
    keep only generic layout regions plus one SAM/action grounded pseudo-node.
    The click point and instruction stay in graph metadata as labels; they are
    not a runtime matching branch.
    """

    device = episode.get("device_info") if isinstance(episode.get("device_info"), dict) else {}
    task = episode.get("task_info") if isinstance(episode.get("task_info"), dict) else {}
    width = _positive_float(device.get("w") or episode.get("w"))
    height = _positive_float(device.get("h") or episode.get("h"))
    if width is None or height is None:
        raise ValueError("GUIOdyssey step requires positive device width and height")

    action = str(step.get("action") or "").upper()
    bbox = _action_bbox(step, width=width, height=height)
    point = _action_point(step, width=width, height=height)
    if action != "CLICK" or bbox is None or point is None:
        raise ValueError("GUIOdyssey importer keeps only CLICK steps with a target box")

    episode_id = str(episode.get("episode_id") or task.get("episode_id") or "")
    step_index = int(step.get("step") or 0)
    resolved_id = graph_id or f"guiodyssey:{episode_id}:{step_index}"
    nodes = [*_region_nodes(width=width, height=height)]
    parent_id = _smallest_region_parent(nodes, bbox)
    target = UINode(
        node_id="action_target",
        origin_id="action_target",
        parent_id=parent_id,
        text="",
        content_desc="",
        resource_id="",
        class_name="guiodyssey.sam2_action_target",
        bbox=bbox,
        clickable=True,
        enabled=True,
        depth=2 if parent_id else 1,
        metadata={
            "visible": True,
            "source": "sam2_bbox",
            "weak_grounding_label": True,
        },
    )
    nodes.append(target)
    nodes = _attach_child(nodes, parent_id=parent_id, child_id=target.node_id)
    return UIGraph(
        graph_id=resolved_id,
        nodes=tuple(nodes),
        width=width,
        height=height,
        metadata={
            "source_format": "guiodyssey_weak_action",
            "dataset": "guiodyssey",
            "license": "cc-by-4.0",
            "episode_id": episode_id,
            "trajectory_step": step_index,
            "screenshot": str(step.get("screenshot") or ""),
            "screenshot_path": str(step.get("screenshot_path") or ""),
            "device_name": str(device.get("device_name") or episode.get("device_name") or ""),
            "device_product": str(device.get("product") or ""),
            "category": str(task.get("category") or episode.get("category") or ""),
            "apps": tuple(str(value) for value in task.get("app") or episode.get("app") or ()),
            "task_instruction_label": str(task.get("instruction") or episode.get("instruction") or ""),
            "low_level_instruction_label": str(step.get("low_level_instruction") or ""),
            "action_type_label": action,
            "action_point_label": point,
            "action_bbox_label": bbox,
            "action_target_node_id": target.node_id,
            "supervision": "sam2_bbox_and_action_point_label_only",
        },
    )


def usable_gui_odyssey_steps(episode: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield steps that can produce a weak action graph."""

    for step in episode.get("steps") or ():
        if not isinstance(step, dict):
            continue
        try:
            graph_from_gui_odyssey_step(episode, step)
        except ValueError:
            continue
        yield step


def _region_nodes(*, width: float, height: float) -> list[UINode]:
    nodes: list[UINode] = []
    for name, fraction in REGION_FRACTIONS:
        bbox = (
            fraction[0] * width,
            fraction[1] * height,
            fraction[2] * width,
            fraction[3] * height,
        )
        is_root = name == "screen"
        nodes.append(
            UINode(
                node_id=name,
                origin_id=name,
                parent_id=None if is_root else "screen",
                class_name=f"guiodyssey.region.{name}",
                bbox=bbox,
                enabled=True,
                depth=0 if is_root else 1,
                child_ids=() if is_root else (),
                metadata={"visible": True, "synthetic_region": True},
            )
        )
    return _attach_screen_children(nodes)


def _attach_screen_children(nodes: list[UINode]) -> list[UINode]:
    children = tuple(node.node_id for node in nodes if node.parent_id == "screen")
    return [
        UINode(**{**node.__dict__, "child_ids": children})
        if node.node_id == "screen"
        else node
        for node in nodes
    ]


def _attach_child(nodes: list[UINode], *, parent_id: str | None, child_id: str) -> list[UINode]:
    if not parent_id:
        return nodes
    return [
        UINode(
            **{
                **node.__dict__,
                "child_ids": tuple(dict.fromkeys((*node.child_ids, child_id))),
            }
        )
        if node.node_id == parent_id
        else node
        for node in nodes
    ]


def _smallest_region_parent(nodes: list[UINode], bbox: BBox) -> str | None:
    candidates = [
        node
        for node in nodes
        if node.node_id != "screen" and node.bbox and _contains(node.bbox, bbox)
    ]
    if not candidates:
        return "screen"
    return min(candidates, key=lambda node: _area(node.bbox or bbox)).node_id


def _action_bbox(step: dict[str, Any], *, width: float, height: float) -> BBox | None:
    bbox = _parse_bbox(step.get("sam2_bbox"), width=width, height=height)
    if bbox is not None:
        return bbox
    point = _action_point(step, width=width, height=height)
    if point is None:
        return None
    radius = max(8.0, min(width, height) * 0.02)
    x, y = point
    return (
        max(0.0, x - radius),
        max(0.0, y - radius),
        min(width, x + radius),
        min(height, y + radius),
    )


def _action_point(step: dict[str, Any], *, width: float, height: float) -> tuple[float, float] | None:
    info = step.get("info")
    if not isinstance(info, list) or not info:
        return None
    point = info[0]
    if not isinstance(point, list) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None
    return x * width / 1000.0, y * height / 1000.0


def _parse_bbox(value: Any, *, width: float, height: float) -> BBox | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    )


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
