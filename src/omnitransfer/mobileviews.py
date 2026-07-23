"""Streaming adapters for the official MobileViews screenshot/VH parquet files."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record


def graph_from_mobileviews_record(
    record: dict[str, Any],
    *,
    graph_id: str,
    screenshot_path: str = "",
    image_size: tuple[int, int] | None = None,
) -> UIGraph:
    """Convert one official ``image_content/json_content`` row to a UI graph."""

    payload = decode_mobileviews_json(record.get("json_content", record))
    graph = _visible_subgraph(graph_from_record(payload, graph_id=graph_id))
    package = infer_mobileviews_package(record, payload)
    extent_width, extent_height = _graph_extent(graph)
    normalized = UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=extent_width or graph.width,
        height=extent_height or graph.height,
        metadata={
            **graph.metadata,
            "source_format": "mobileviews_droidbot_json",
            "dataset": "mobileviews_600k",
            "package": package,
            "app": package,
            "foreground_activity": str(payload.get("foreground_activity") or ""),
            "screenshot_path": "",
            "geometry_mismatch": False,
        },
    )
    if image_size is not None or screenshot_path:
        return attach_mobileviews_image(
            normalized,
            screenshot_path=screenshot_path,
            image_size=image_size,
        )
    return normalized


def attach_mobileviews_image(
    graph: UIGraph,
    *,
    screenshot_path: str,
    image_size: tuple[int, int] | None,
) -> UIGraph:
    """Attach a materialized screenshot while preserving original coordinates."""

    extent_width, extent_height = _graph_extent(graph)
    image_width = float(image_size[0]) if image_size else None
    image_height = float(image_size[1]) if image_size else None
    geometry_mismatch, geometry_in_bounds_rate = _graph_geometry_mismatch(
        graph,
        image_width=image_width,
        image_height=image_height,
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=image_width or extent_width or graph.width,
        height=image_height or extent_height or graph.height,
        metadata={
            **graph.metadata,
            "screenshot_path": screenshot_path,
            "geometry_mismatch": geometry_mismatch,
            "geometry_in_bounds_rate": geometry_in_bounds_rate,
        },
    )


def decode_mobileviews_json(value: Any) -> dict[str, Any]:
    """Decode the JSON column emitted by PyArrow or Hugging Face datasets."""

    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    raise ValueError("MobileViews json_content must decode to an object")


def infer_mobileviews_package(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    """Recover an app split key from explicit metadata or hierarchy nodes."""

    for mapping in (record, payload):
        for key in ("package", "app_package", "package_name", "app_id", "apk"):
            value = str(mapping.get(key) or "").strip()
            if value:
                return value
    activity = str(payload.get("foreground_activity") or "").strip()
    if "/" in activity:
        package = activity.split("/", 1)[0].strip()
        if package:
            return package
    packages = Counter(
        str(node.get("package") or "").strip()
        for node in _iter_node_payloads(payload)
        if str(node.get("package") or "").strip()
    )
    if packages:
        return packages.most_common(1)[0][0]
    return ""


def materialize_mobileviews_image(
    image_content: Any,
    destination: Path,
    *,
    long_side: int = 384,
) -> tuple[int, int]:
    """Write a lossless, pre-resized PNG and return the source dimensions."""

    if long_side <= 0:
        raise ValueError("long_side must be positive")
    if not isinstance(image_content, (bytes, bytearray, memoryview)):
        raise ValueError("MobileViews image_content must contain encoded image bytes")
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "Materializing MobileViews images requires Pillow. "
            "Install omnitransfer[train]."
        ) from exc
    with Image.open(BytesIO(bytes(image_content))) as image:
        image = image.convert("RGB")
        source_size = (image.width, image.height)
        if max(image.size) > long_side:
            scale = long_side / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        image.save(temporary, format="PNG", compress_level=3)
        temporary.replace(destination)
    return source_size


def split_name_for_group(
    group: str,
    *,
    seed: int,
    dev_percent: int = 10,
    test_percent: int = 10,
) -> str:
    """Assign one package atomically to a deterministic train/dev/test split."""

    if not group:
        raise ValueError("group must be non-empty")
    if dev_percent < 0 or test_percent < 0 or dev_percent + test_percent >= 100:
        raise ValueError("dev_percent and test_percent must sum below 100")
    digest = hashlib.blake2b(
        f"{seed}:{group}".encode("utf-8"),
        digest_size=8,
    ).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + dev_percent:
        return "dev"
    return "train"


def _iter_node_payloads(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    stack: list[Any] = [payload]
    visited: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
            continue
        if not isinstance(item, dict) or id(item) in visited:
            continue
        visited.add(id(item))
        if any(
            key in item
            for key in ("package", "class", "viewClass", "resource_id", "resourceId")
        ):
            yield item
        for key in ("viewHierarchy", "root", "document", "views", "nodes", "children", "child"):
            value = item.get(key)
            if isinstance(value, (dict, list)):
                stack.append(value)


def _visible_subgraph(graph: UIGraph) -> UIGraph:
    kept_ids = {
        node.node_id for node in graph.nodes if node.metadata.get("visible", True)
    }
    if len(kept_ids) == len(graph.nodes):
        return graph
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    parent_by_id: dict[str, str | None] = {}
    for node in graph.nodes:
        if node.node_id not in kept_ids:
            continue
        parent_id = node.parent_id
        while parent_id is not None and parent_id not in kept_ids:
            parent = nodes_by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        parent_by_id[node.node_id] = parent_id
    children_by_parent: dict[str, list[str]] = {}
    for node_id, parent_id in parent_by_id.items():
        if parent_id is not None:
            children_by_parent.setdefault(parent_id, []).append(node_id)
    depths = _visible_depths(parent_by_id)
    nodes = tuple(
        UINode(
            **{
                **node.__dict__,
                "parent_id": parent_by_id[node.node_id],
                "child_ids": tuple(children_by_parent.get(node.node_id, ())),
                "depth": depths[node.node_id],
            }
        )
        for node in graph.nodes
        if node.node_id in kept_ids
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "source_nodes": len(graph.nodes),
            "invisible_nodes_removed": len(graph.nodes) - len(nodes),
        },
    )


def _visible_depths(parent_by_id: dict[str, str | None]) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node_id in parent_by_id:
        depth = 0
        parent_id = parent_by_id[node_id]
        visited = {node_id}
        while parent_id is not None and parent_id not in visited:
            visited.add(parent_id)
            depth += 1
            parent_id = parent_by_id.get(parent_id)
        depths[node_id] = depth
    return depths


def _graph_extent(graph: UIGraph) -> tuple[float | None, float | None]:
    width = max((node.bbox[2] for node in graph.nodes if node.bbox), default=0.0)
    height = max((node.bbox[3] for node in graph.nodes if node.bbox), default=0.0)
    return width or None, height or None


def _graph_geometry_mismatch(
    graph: UIGraph,
    *,
    image_width: float | None,
    image_height: float | None,
) -> tuple[bool, float | None]:
    if not image_width or not image_height:
        return False, None
    boxes = [
        node.bbox
        for node in graph.nodes
        if node.bbox is not None
        and node.bbox[2] > node.bbox[0]
        and node.bbox[3] > node.bbox[1]
    ]
    if not boxes:
        return True, 0.0
    tolerance_x = image_width * 0.02
    tolerance_y = image_height * 0.02
    in_bounds = sum(
        bbox[0] >= -tolerance_x
        and bbox[1] >= -tolerance_y
        and bbox[2] <= image_width + tolerance_x
        and bbox[3] <= image_height + tolerance_y
        for bbox in boxes
    )
    rate = in_bounds / len(boxes)
    return rate < 0.90, rate
