"""Adapters for the OS-Atlas mobile grounding release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnitransfer.mobileviews import materialize_mobileviews_image
from omnitransfer.ui_graph import BBox, UIGraph, UINode


def graph_from_os_atlas_record(
    record: dict[str, Any],
    *,
    graph_id: str,
    image_size: tuple[int, int],
    screenshot_path: str = "",
    dataset: str = "OS-Copilot/OS-Atlas-data",
    subset: str = "android_world",
) -> UIGraph:
    """Convert one grouped OS-Atlas grounding record to a canonical UI graph."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("OS-Atlas image dimensions must be positive")
    raw_elements = record.get("elements")
    if not isinstance(raw_elements, list):
        raise ValueError("OS-Atlas records require an elements list")

    elements: list[dict[str, Any]] = []
    boxes: list[BBox] = []
    source_indices: list[int] = []
    for source_index, value in enumerate(raw_elements):
        if not isinstance(value, dict):
            continue
        bbox = _normalized_bbox(value.get("bbox"), width=width, height=height)
        if bbox is None:
            continue
        elements.append(value)
        boxes.append(bbox)
        source_indices.append(source_index)
    if not elements:
        raise ValueError("OS-Atlas record contains no bounded elements")

    parent_indices = _containment_parents(boxes)
    children: dict[int, list[int]] = {}
    for index, parent_index in enumerate(parent_indices):
        if parent_index is not None:
            children.setdefault(parent_index, []).append(index)
    depths = _depths(parent_indices)
    nodes = tuple(
        UINode(
            node_id=f"e{source_indices[index]}",
            origin_id=f"e{source_indices[index]}",
            parent_id=(
                f"e{source_indices[parent_indices[index]]}"
                if parent_indices[index] is not None
                else None
            ),
            text=str(element.get("instruction") or ""),
            class_name=str(element.get("data_type") or ""),
            bbox=boxes[index],
            clickable=_is_clickable(str(element.get("data_type") or "")),
            editable=_is_editable(str(element.get("data_type") or "")),
            scrollable=_is_scrollable(str(element.get("data_type") or "")),
            depth=depths[index],
            child_ids=tuple(
                f"e{source_indices[child_index]}"
                for child_index in children.get(index, ())
            ),
            metadata={
                "source_index": source_indices[index],
                "grounding_instruction": str(element.get("instruction") or ""),
            },
        )
        for index, element in enumerate(elements)
    )
    return UIGraph(
        graph_id=graph_id,
        nodes=nodes,
        width=float(width),
        height=float(height),
        metadata={
            "source_format": "os_atlas_mobile_grounding",
            "dataset": dataset,
            "subset": subset,
            "platform": "android",
            "image_filename": str(record.get("img_filename") or ""),
            "screenshot_path": screenshot_path,
            "relation_source": "strict_bbox_containment",
            "split_unit": "image_uuid",
            "app_identity_available": False,
        },
    )


def materialize_os_atlas_image(
    source: Path,
    destination: Path,
    *,
    long_side: int = 384,
) -> tuple[int, int]:
    """Write one OS-Atlas screenshot as a lossless pre-resized PNG."""

    return materialize_mobileviews_image(
        source.read_bytes(),
        destination,
        long_side=long_side,
    )


def _normalized_bbox(value: Any, *, width: float, height: float) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 * width, y1 * height, x2 * width, y2 * height


def _containment_parents(boxes: list[BBox]) -> list[int | None]:
    areas = [_bbox_area(bbox) for bbox in boxes]
    parents: list[int | None] = []
    for child_index, child_box in enumerate(boxes):
        candidates = [
            parent_index
            for parent_index, parent_box in enumerate(boxes)
            if parent_index != child_index
            and areas[parent_index] > areas[child_index]
            and _contains(parent_box, child_box)
        ]
        parents.append(
            min(candidates, key=lambda index: (areas[index], index))
            if candidates
            else None
        )
    return parents


def _depths(parents: list[int | None]) -> list[int]:
    depths: dict[int, int] = {}

    def resolve(index: int, trail: set[int]) -> int:
        if index in depths:
            return depths[index]
        parent = parents[index]
        depth = (
            0
            if parent is None or parent in trail
            else resolve(parent, {*trail, index}) + 1
        )
        depths[index] = depth
        return depth

    return [resolve(index, set()) for index in range(len(parents))]


def _bbox_area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _class_suffix(class_name: str) -> str:
    return class_name.rsplit(".", 1)[-1].lower()


def _is_clickable(class_name: str) -> bool:
    return _class_suffix(class_name) in {
        "autocompleteedittext",
        "button",
        "checkbox",
        "checkedtextview",
        "edittext",
        "imagebutton",
        "radiobutton",
        "ratingbar",
        "seekbar",
        "spinner",
        "switch",
        "togglebutton",
    }


def _is_editable(class_name: str) -> bool:
    return _class_suffix(class_name) in {"autocompleteedittext", "edittext"}


def _is_scrollable(class_name: str) -> bool:
    return _class_suffix(class_name) in {
        "gridview",
        "listview",
        "recyclerview",
        "scrollview",
        "viewpager",
    }
