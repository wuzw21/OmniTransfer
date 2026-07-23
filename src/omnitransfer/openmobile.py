"""Adapters for the Uni-GUI-OpenMobile screenshot/UI-element trajectories."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.mobileviews import materialize_mobileviews_image
from omnitransfer.ui_graph import BBox, UIGraph, UINode


def graph_from_openmobile_step(
    step: dict[str, Any],
    *,
    graph_id: str,
    package: str,
    app: str = "",
    episode_id: str = "",
    screenshot_path: str = "",
    action: dict[str, Any] | None = None,
) -> UIGraph:
    """Convert one OpenMobile metadata step to the canonical UI graph."""

    screen_size = step.get("logical_screen_size") or step.get("screen_resolution")
    if not isinstance(screen_size, (list, tuple)) or len(screen_size) < 2:
        raise ValueError("OpenMobile step requires logical_screen_size")
    width = _positive_float(screen_size[0])
    height = _positive_float(screen_size[1])
    if width is None or height is None:
        raise ValueError("OpenMobile screen dimensions must be positive")
    raw_elements = step.get("ui_elements")
    if not isinstance(raw_elements, list):
        raise ValueError("OpenMobile step requires a ui_elements list")

    elements: list[dict[str, Any]] = []
    boxes: list[BBox] = []
    for value in raw_elements:
        if not isinstance(value, dict) or value.get("is_visible") is False:
            continue
        bbox = _openmobile_bbox(value.get("bbox_pixels"), width=width, height=height)
        if bbox is None:
            continue
        elements.append(value)
        boxes.append(bbox)
    if not elements:
        raise ValueError("OpenMobile step contains no visible bounded elements")

    parent_indices = _containment_parents(elements, boxes)
    children: dict[int, list[int]] = {}
    for index, parent_index in enumerate(parent_indices):
        if parent_index is not None:
            children.setdefault(parent_index, []).append(index)
    depths = _depths(parent_indices)
    nodes = tuple(
        UINode(
            node_id=f"n{index}",
            origin_id=f"n{index}",
            parent_id=(
                f"n{parent_indices[index]}"
                if parent_indices[index] is not None
                else None
            ),
            text=str(element.get("text") or element.get("hint_text") or ""),
            content_desc=str(element.get("content_description") or ""),
            resource_id=str(
                element.get("resource_name") or element.get("resource_id") or ""
            ),
            class_name=str(element.get("class_name") or ""),
            bbox=boxes[index],
            clickable=bool(element.get("is_clickable")),
            editable=bool(element.get("is_editable")),
            scrollable=bool(element.get("is_scrollable")),
            enabled=element.get("is_enabled") is not False,
            depth=depths[index],
            child_ids=tuple(f"n{child_index}" for child_index in children.get(index, ())),
            metadata={
                "package_name": str(element.get("package_name") or ""),
                "checkable": bool(element.get("is_checkable")),
                "focusable": bool(element.get("is_focusable")),
                "long_clickable": bool(element.get("is_long_clickable")),
                "selected": bool(element.get("is_selected")),
                "source_index": index,
                "visible": True,
            },
        )
        for index, element in enumerate(elements)
    )
    metadata: dict[str, Any] = {
        "source_format": "unigui_openmobile_ui_elements",
        "dataset": "unigui_openmobile",
        "package": package,
        "app": app or package,
        "episode_id": episode_id,
        "trajectory_step": int(step.get("step") or 0),
        "screenshot_path": screenshot_path,
        "relation_source": "strict_bbox_containment",
    }
    if action:
        metadata.update(_action_metadata(action))
    return UIGraph(
        graph_id=graph_id,
        nodes=nodes,
        width=width,
        height=height,
        metadata=metadata,
    )


def attach_openmobile_image(
    graph: UIGraph,
    *,
    screenshot_path: str,
    image_size: tuple[int, int],
) -> UIGraph:
    """Attach a resized screenshot and audit source-image geometry."""

    image_width, image_height = image_size
    mismatch = not (
        graph.width is not None
        and graph.height is not None
        and abs(graph.width - image_width) <= 1.0
        and abs(graph.height - image_height) <= 1.0
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "screenshot_path": screenshot_path,
            "source_image_width": image_width,
            "source_image_height": image_height,
            "geometry_mismatch": mismatch,
        },
    )


def materialize_openmobile_image(
    source: Path,
    destination: Path,
    *,
    long_side: int = 384,
) -> tuple[int, int]:
    """Write one OpenMobile screenshot as a lossless pre-resized PNG."""

    return materialize_mobileviews_image(
        source.read_bytes(),
        destination,
        long_side=long_side,
    )


def infer_openmobile_package(
    task: dict[str, Any],
    steps: Iterable[dict[str, Any]],
) -> str:
    """Return the trajectory-level package used for leakage-free splitting."""

    package = str(task.get("app_package") or "").strip()
    if package:
        return package
    packages = task.get("app_packages")
    if isinstance(packages, list):
        for value in packages:
            package = str(value or "").strip()
            if package:
                return package
    counts: Counter[str] = Counter()
    for step in steps:
        for element in step.get("ui_elements") or ():
            if not isinstance(element, dict):
                continue
            package = str(element.get("package_name") or "").strip()
            if package and package not in _SYSTEM_PACKAGES:
                counts[package] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _containment_parents(
    elements: list[dict[str, Any]],
    boxes: list[BBox],
) -> list[int | None]:
    areas = [_bbox_area(bbox) for bbox in boxes]
    packages = [str(element.get("package_name") or "") for element in elements]
    parents: list[int | None] = []
    for child_index, child_box in enumerate(boxes):
        candidates = [
            parent_index
            for parent_index, parent_box in enumerate(boxes)
            if parent_index != child_index
            and areas[parent_index] > areas[child_index]
            and _contains(parent_box, child_box)
        ]
        if not candidates:
            parents.append(None)
            continue
        parents.append(
            min(
                candidates,
                key=lambda parent_index: (
                    packages[parent_index] != packages[child_index],
                    areas[parent_index],
                    parent_index,
                ),
            )
        )
    return parents


def _depths(parents: list[int | None]) -> list[int]:
    depths: dict[int, int] = {}

    def resolve(index: int, trail: set[int]) -> int:
        if index in depths:
            return depths[index]
        parent = parents[index]
        if parent is None or parent in trail:
            depth = 0
        else:
            depth = resolve(parent, {*trail, index}) + 1
        depths[index] = depth
        return depth

    return [resolve(index, set()) for index in range(len(parents))]


def _openmobile_bbox(value: Any, *, width: float, height: float) -> BBox | None:
    if not isinstance(value, dict):
        return None
    try:
        x1 = float(value["x_min"])
        y1 = float(value["y_min"])
        x2 = float(value["x_max"])
        y2 = float(value["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _action_metadata(action: dict[str, Any]) -> dict[str, Any]:
    plan = action.get("plan")
    arguments = plan.get("arguments") if isinstance(plan, dict) else None
    action_type = arguments.get("action") if isinstance(arguments, dict) else ""
    bbox = action.get("bbox")
    return {
        "action_type_label": str(action_type or ""),
        "action_bbox_label": bbox if isinstance(bbox, list) else None,
        "action_is_use": action.get("is_use") is not False,
        "action_is_reviewed": bool(action.get("is_reviewed")),
    }


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _bbox_area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


_SYSTEM_PACKAGES = {
    "com.android.systemui",
    "com.google.android.apps.nexuslauncher",
}
