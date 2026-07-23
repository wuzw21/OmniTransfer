"""Adapters for the WebUI screenshot and element-box parquet releases."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.ui_graph import UIGraph, UINode


DEVICE_SCALES = {
    "default": 1.0,
    "iPad-Mini": 2.0,
    "iPad-Pro": 2.0,
    "iPhone-13 Pro": 3.0,
    "iPhone-SE": 3.0,
}

_CLICKABLE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "menuitem",
    "option",
    "radio",
    "tab",
}
_EDITABLE_ROLES = {"combobox", "input", "searchbox", "textarea", "textbox"}
_SCROLLABLE_ROLES = {"scrollbar"}


def graph_from_webui_record(
    record: dict[str, Any],
    *,
    graph_id: str,
    image_size: tuple[int, int],
    screenshot_path: str = "",
    dataset: str = "biglab/webui-70k-elements",
    split: str = "train",
    min_area: float = 4.0,
) -> UIGraph:
    """Convert one WebUI screenshot and its visible element boxes to a graph."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("WebUI image dimensions must be positive")
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    labels = record.get("labels")
    boxes = record.get("contentBoxes")
    if not isinstance(labels, list) or not isinstance(boxes, list):
        raise ValueError("WebUI records require labels and contentBoxes lists")
    if len(labels) != len(boxes):
        raise ValueError("WebUI labels and contentBoxes must have equal length")
    key_name = str(record.get("key_name") or "")
    scale = webui_device_scale(key_name)
    nodes: list[UINode] = []
    for source_index, (label_values, box_values) in enumerate(zip(labels, boxes)):
        bbox = _visible_bbox(
            box_values,
            scale=scale,
            width=float(width),
            height=float(height),
            min_area=min_area,
        )
        if bbox is None:
            continue
        normalized_labels = _labels(label_values)
        roles = {label.lower() for label in normalized_labels}
        node_id = f"e{source_index}"
        nodes.append(
            UINode(
                node_id=node_id,
                origin_id=node_id,
                class_name=" ".join(normalized_labels),
                bbox=bbox,
                clickable=bool(roles.intersection(_CLICKABLE_ROLES)),
                editable=bool(roles.intersection(_EDITABLE_ROLES)),
                scrollable=bool(roles.intersection(_SCROLLABLE_ROLES)),
                metadata={
                    "source_format": "webui_element_boxes",
                    "source_index": source_index,
                    "semantic_labels": normalized_labels,
                },
            )
        )
    if not nodes:
        raise ValueError("WebUI record contains no visible bounded elements")
    return UIGraph(
        graph_id=graph_id,
        nodes=tuple(nodes),
        width=float(width),
        height=float(height),
        metadata={
            "source_format": "webui_element_boxes",
            "dataset": dataset,
            "platform": "web",
            "split": split,
            "viewport": key_name,
            "device_scale": scale,
            "screenshot_path": screenshot_path,
            "source_elements": len(labels),
            "visible_bounded_elements": len(nodes),
            "official_domain_disjoint_split": True,
        },
    )


def materialize_webui_image(
    image: Any,
    destination: Path,
    *,
    long_side: int = 384,
) -> tuple[int, int]:
    """Write a lossless resized WebUI image and return its source dimensions."""

    if long_side <= 0:
        raise ValueError("long_side must be positive")
    encoded = _image_bytes(image)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "Materializing WebUI images requires Pillow. Install omnitransfer[train]."
        ) from exc
    with Image.open(BytesIO(encoded)) as screenshot:
        screenshot = screenshot.convert("RGB")
        source_size = (screenshot.width, screenshot.height)
        if max(screenshot.size) > long_side:
            scale = long_side / max(screenshot.size)
            screenshot = screenshot.resize(
                (
                    max(1, round(screenshot.width * scale)),
                    max(1, round(screenshot.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        screenshot.save(temporary, format="PNG", compress_level=3)
        temporary.replace(destination)
    return source_size


def webui_device_scale(key_name: str) -> float:
    """Return the scale used by the official WebUI preprocessing code."""

    device_name = key_name.split("_", 1)[0]
    try:
        return DEVICE_SCALES[device_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported WebUI viewport: {key_name!r}") from exc


def _image_bytes(image: Any) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, dict):
        value = image.get("bytes")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
    raise ValueError("WebUI image must contain encoded bytes")


def _visible_bbox(
    values: Any,
    *,
    scale: float,
    width: float,
    height: float,
    min_area: float,
) -> tuple[float, float, float, float] | None:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return None
    coordinates = list(values)
    if len(coordinates) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) * scale for value in coordinates)
    except (TypeError, ValueError):
        return None
    x1 = min(max(0.0, x1), width)
    y1 = min(max(0.0, y1), height)
    x2 = min(max(0.0, x2), width)
    y2 = min(max(0.0, y2), height)
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) < min_area:
        return None
    return x1, y1, x2, y2


def _labels(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Iterable):
        return ()
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )
