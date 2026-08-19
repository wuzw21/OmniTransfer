"""Canonical iOS screen normalization for the OmniTransfer data boundary.

The adapter is the one place where iOS capture conventions are converted into
the shared screen record consumed by graph construction, review, training, and
runtime diagnostics.  It deliberately does not invent UI semantics: the
platform-neutral XML parser remains the owner of node identity, labels,
clickability, and hierarchy.

iOS captures commonly mix two coordinate spaces:

* XML bounds in logical points (for example 414x736), and
* screenshot/visual bounds in Retina pixels (for example 1242x2208).

The canonical contract keeps ``graph.nodes[*].bbox`` in XML coordinates and
``graph.nodes[*].metadata.visual_bbox`` in screenshot pixels.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from omnitransfer.ui_graph import graph_from_record, graph_to_record


IOS_ADAPTER_SCHEMA_VERSION = "omnitransfer.ios_screen_adapter.v1"
IOS_PLATFORM_NAMES = frozenset({"ios", "iphoneos", "ipad os", "ipados"})


class IOSAdapterError(ValueError):
    """Raised when a raw iOS capture cannot become a canonical screen."""


def _resolve_path(value: str | Path, *, root: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def _node_payload(node: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a parser node without changing its semantic attributes."""

    metadata = dict(node.get("metadata") or {})
    if metadata.get("visual_bbox") is not None:
        metadata["visual_bbox_coordinate_space"] = "page_pixels"
    return {
        "node_id": node["node_id"],
        "origin_id": node.get("origin_id") or node["node_id"],
        "parent_id": node.get("parent_id"),
        "text": node.get("text", ""),
        "content_desc": node.get("content_desc", ""),
        "resource_id": node.get("resource_id", ""),
        "class_name": node.get("class_name", ""),
        "bbox": node.get("bbox"),
        "visual_bbox": metadata.get("visual_bbox"),
        "clickable": bool(node.get("clickable")),
        "editable": bool(node.get("editable")),
        "scrollable": bool(node.get("scrollable")),
        "enabled": bool(node.get("enabled", True)),
        "depth": node.get("depth", 0),
        "child_ids": node.get("child_ids", []),
        "metadata": metadata,
    }


def correct_ios_coordinate_spaces(
    graph: dict[str, Any], *, visual_width: float = 0, visual_height: float = 0
) -> None:
    """Normalize iOS XML and visual bounds in-place.

    This only changes numeric coordinate representation. It never changes the
    node list, hierarchy, labels, resource IDs, or actionability flags.
    """

    width = float(graph.get("width") or 0)
    height = float(graph.get("height") or 0)
    nodes = graph.get("nodes") or []
    boxes = [node.get("bbox") for node in nodes if node.get("bbox")]
    if not boxes or width <= 1 or height <= 1:
        return

    scale_x = width / visual_width if visual_width > 1 else 1.0
    scale_y = height / visual_height if visual_height > 1 else 1.0
    relative_difference = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-9)
    max_right = max(float(box[2]) for box in boxes)
    max_bottom = max(float(box[3]) for box in boxes)
    # The parsed graph is already in the declared XML coordinate space when
    # normalized bounds have been expanded by graph_from_record.  Retina
    # dimensions alone are therefore not evidence that XML bboxes need to be
    # divided.  Only rescale when the actual bboxes exceed the declared graph
    # extent, which is the legacy screenshot-pixel XML case.
    screenshot_scaled_xml = relative_difference <= 0.03 and (
        (scale_x <= 0.8 and (max_right > width * 1.25 or max_bottom > height * 1.25))
        or (scale_x >= 1.25 and (max_right < width / 1.25 or max_bottom < height / 1.25))
    )

    def restore_xml_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
        values = [float(value) for value in bbox]
        if max(values) <= 1.000001:
            return [
                values[0] * width,
                values[1] * height,
                values[2] * width,
                values[3] * height,
            ]
        if screenshot_scaled_xml:
            return [
                values[0] * scale_x,
                values[1] * scale_y,
                values[2] * scale_x,
                values[3] * scale_y,
            ]
        return values

    def restore_visual_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
        values = [float(value) for value in bbox]
        if max(values) <= 1.000001 and visual_width > 1 and visual_height > 1:
            return [
                values[0] * visual_width,
                values[1] * visual_height,
                values[2] * visual_width,
                values[3] * visual_height,
            ]
        return values

    for node in nodes:
        bbox = node.get("bbox")
        if bbox:
            node["bbox"] = restore_xml_bbox(bbox)
        visual_bbox = node.get("visual_bbox")
        if visual_bbox:
            node["visual_bbox"] = restore_visual_bbox(visual_bbox)
        metadata = node.get("metadata")
        if isinstance(metadata, dict) and metadata.get("visual_bbox"):
            metadata["visual_bbox"] = restore_visual_bbox(metadata["visual_bbox"])
            metadata["visual_bbox_coordinate_space"] = "page_pixels"


def load_ios_screen(raw: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Convert one raw iOS screen row into the canonical screen record.

    ``root`` is used only to resolve relative asset paths. Missing XML is a
    hard preprocessing error; a failed normalization must not silently pass
    source-device coordinates downstream.
    """

    platform = str(raw.get("platform") or "").strip().lower()
    if platform not in IOS_PLATFORM_NAMES:
        raise IOSAdapterError(
            f"load_ios_screen expects an iOS row, got platform={raw.get('platform')!r}"
        )
    if not raw.get("screen_id"):
        raise IOSAdapterError("iOS screen row is missing screen_id")
    if not raw.get("xml_path"):
        raise IOSAdapterError(f"iOS screen {raw['screen_id']} is missing xml_path")

    screen_id = str(raw["screen_id"])
    xml_path = _resolve_path(str(raw["xml_path"]), root=root)
    screenshot_path = (
        _resolve_path(str(raw["screenshot_path"]), root=root)
        if raw.get("screenshot_path")
        else None
    )
    xml = xml_path.read_text(encoding="utf-8")
    display_width = float(raw.get("screenshot_width") or 0)
    display_height = float(raw.get("screenshot_height") or 0)
    graph = graph_from_record(
        {
            "xml": xml,
            "width": raw.get("original_xml_width"),
            "height": raw.get("original_xml_height"),
            "display_width": display_width or None,
            "display_height": display_height or None,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        },
        graph_id=screen_id,
    )
    graph_record = graph_to_record(graph)
    graph_record["nodes"] = [_node_payload(node) for node in graph_record["nodes"]]
    display_width = display_width or float(graph_record["width"] or 0)
    display_height = display_height or float(graph_record["height"] or 0)
    correct_ios_coordinate_spaces(
        graph_record,
        visual_width=display_width,
        visual_height=display_height,
    )
    graph_record.setdefault("metadata", {})["visual_display_size"] = [
        display_width,
        display_height,
    ]
    graph_record["metadata"]["ios_adapter_schema_version"] = IOS_ADAPTER_SCHEMA_VERSION

    return {
        "screen_id": screen_id,
        "page_id": screen_id,
        "platform": "ios",
        "screenshot_path": str(screenshot_path) if screenshot_path else "",
        "display_width": display_width,
        "display_height": display_height,
        "xml": xml,
        "graph": graph_record,
        "adapter": {
            "name": IOS_ADAPTER_SCHEMA_VERSION,
            "xml_bbox_coordinate_space": "logical_xml_pixels",
            "visual_bbox_coordinate_space": "screenshot_pixels",
            "source_xml_path": str(xml_path),
        },
    }


def attach_xml_payload(
    screen: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Attach sibling XML for a canonical page used by review/diagnostics.

    This helper does not reparse or modify the canonical graph. It only makes
    the already-normalized XML evidence available to later consumers.
    """

    result = copy.deepcopy(dict(screen))
    if result.get("xml") or result.get("xml_text"):
        return result
    screenshot_path = result.get("screenshot_path")
    if not screenshot_path:
        return result
    xml_path = _resolve_path(Path(str(screenshot_path)).with_suffix(".xml"), root=root)
    if xml_path.is_file():
        result["xml"] = xml_path.read_text(encoding="utf-8")
    return result
