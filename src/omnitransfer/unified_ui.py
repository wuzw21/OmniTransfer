"""Adapters from native, DOM, accessibility, and WebView records to one UI graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from omnitransfer.mobileviews import graph_from_mobileviews_record
from omnitransfer.openmobile import graph_from_openmobile_step
from omnitransfer.gui_odyssey import graph_from_gui_odyssey_step
from omnitransfer.os_atlas import graph_from_os_atlas_record
from omnitransfer.ui_graph import BBox, UIGraph, UINode, graph_from_record
from omnitransfer.webui import graph_from_webui_record


@dataclass(frozen=True)
class WebViewTransform:
    """Map DOM viewport coordinates into the native WebView host rectangle."""

    viewport_width: float
    viewport_height: float
    scroll_x: float = 0.0
    scroll_y: float = 0.0


def graph_from_unified_record(
    record: dict[str, Any],
    *,
    graph_id: str | None = None,
) -> UIGraph:
    """Normalize supported UI formats into the single OmniTransfer graph schema."""

    source_format = str(record.get("source_format") or record.get("format") or "").lower()
    resolved_id = graph_id or str(
        record.get("graph_id")
        or record.get("screen_id")
        or record.get("id")
        or "screen"
    )
    if "json_content" in record:
        image_width = _optional_float(record.get("image_width"))
        image_height = _optional_float(record.get("image_height"))
        image_size = (
            (round(image_width), round(image_height))
            if image_width and image_height
            else None
        )
        return graph_from_mobileviews_record(
            record,
            graph_id=resolved_id,
            screenshot_path=str(record.get("screenshot_path") or ""),
            image_size=image_size,
        )
    if source_format in {"unigui_openmobile", "openmobile_ui_elements"} or (
        "ui_elements" in record and "logical_screen_size" in record
    ):
        return graph_from_openmobile_step(
            record,
            graph_id=resolved_id,
            package=str(record.get("package") or record.get("app_package") or ""),
            app=str(record.get("app") or ""),
            episode_id=str(record.get("episode_id") or ""),
            screenshot_path=str(record.get("screenshot_path") or ""),
            action=record.get("action") if isinstance(record.get("action"), dict) else None,
        )
    if source_format == "guiodyssey_weak_action" or (
        "device_info" in record and "task_info" in record and "step" in record
    ):
        step = record.get("step")
        if not isinstance(step, dict):
            raise ValueError("GUIOdyssey unified records require a step object")
        return graph_from_gui_odyssey_step(record, step, graph_id=resolved_id)
    if source_format == "webui_element_boxes" or (
        "labels" in record and "contentBoxes" in record and "key_name" in record
    ):
        image_width = _optional_float(record.get("image_width") or record.get("width"))
        image_height = _optional_float(record.get("image_height") or record.get("height"))
        if not image_width or not image_height:
            raise ValueError("WebUI unified records require image dimensions")
        return graph_from_webui_record(
            record,
            graph_id=resolved_id,
            image_size=(round(image_width), round(image_height)),
            screenshot_path=str(record.get("screenshot_path") or ""),
            dataset=str(record.get("dataset") or "biglab/webui-70k-elements"),
            split=str(record.get("split") or "train"),
        )
    if source_format == "os_atlas_mobile_grounding" or (
        "img_filename" in record and "elements" in record
    ):
        image_width = _optional_float(record.get("image_width") or record.get("width"))
        image_height = _optional_float(record.get("image_height") or record.get("height"))
        if not image_width or not image_height:
            raise ValueError("OS-Atlas unified records require image dimensions")
        return graph_from_os_atlas_record(
            record,
            graph_id=resolved_id,
            image_size=(round(image_width), round(image_height)),
            screenshot_path=str(record.get("screenshot_path") or ""),
            dataset=str(record.get("dataset") or "OS-Copilot/OS-Atlas-data"),
            subset=str(record.get("subset") or "android_world"),
        )
    if source_format in {"dom", "html", "browsergym_dom"} or "dom_snapshot" in record:
        snapshot = record.get("dom_snapshot") or record.get("dom") or record
        return graph_from_dom_snapshot(
            snapshot,
            graph_id=resolved_id,
            width=_optional_float(record.get("width") or record.get("viewport_width")),
            height=_optional_float(record.get("height") or record.get("viewport_height")),
            screenshot_path=str(record.get("screenshot_path") or ""),
            source_format=source_format or "dom",
        )
    if source_format in {"ax", "axtree", "accessibility", "browsergym_ax"} or (
        "accessibility_tree" in record
    ):
        tree = record.get("accessibility_tree") or record.get("ax_tree") or record
        return graph_from_dom_snapshot(
            tree,
            graph_id=resolved_id,
            width=_optional_float(record.get("width") or record.get("viewport_width")),
            height=_optional_float(record.get("height") or record.get("viewport_height")),
            screenshot_path=str(record.get("screenshot_path") or ""),
            source_format=source_format or "accessibility",
        )
    if source_format in {"webview", "hybrid_webview"} or "webview" in record:
        payload = record.get("webview") or record
        native_record = payload.get("native") or payload.get("native_record")
        dom_record = payload.get("dom") or payload.get("dom_snapshot")
        if not isinstance(native_record, dict) or not isinstance(dom_record, dict):
            raise ValueError("WebView records require native and dom objects")
        native = graph_from_unified_record(native_record, graph_id=f"{resolved_id}:native")
        dom = graph_from_dom_snapshot(
            dom_record,
            graph_id=f"{resolved_id}:dom",
            width=_optional_float(payload.get("viewport_width")),
            height=_optional_float(payload.get("viewport_height")),
            source_format="webview_dom",
        )
        return merge_webview_graph(
            native,
            dom,
            host_node_id=str(payload.get("host_node_id") or ""),
            transform=WebViewTransform(
                viewport_width=float(payload.get("viewport_width") or 0.0),
                viewport_height=float(payload.get("viewport_height") or 0.0),
                scroll_x=float(payload.get("scroll_x") or 0.0),
                scroll_y=float(payload.get("scroll_y") or 0.0),
            ),
            graph_id=resolved_id,
            screenshot_path=str(record.get("screenshot_path") or ""),
        )
    graph = graph_from_record(record, graph_id=resolved_id)
    metadata = {
        **graph.metadata,
        **dict(record.get("metadata") or {}),
        "source_format": source_format or graph.metadata.get("source_format", "native"),
    }
    for key in ("screenshot_path", "app", "package", "platform", "dataset", "split"):
        if record.get(key) not in (None, ""):
            metadata[key] = record[key]
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata=metadata,
    )


def graph_from_dom_snapshot(
    snapshot: dict[str, Any],
    *,
    graph_id: str,
    width: float | None = None,
    height: float | None = None,
    screenshot_path: str = "",
    source_format: str = "dom",
) -> UIGraph:
    """Parse a normalized DOM or accessibility snapshot into a UI graph."""

    payload = snapshot.get("root") or snapshot.get("document") or snapshot
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        nodes = _nodes_from_flat_dom(payload["nodes"])
    elif isinstance(payload, list):
        nodes = _nodes_from_flat_dom(payload)
    elif isinstance(payload, dict):
        nodes = _nodes_from_nested_dom(payload)
    else:
        raise ValueError("DOM/accessibility snapshot must be an object or node list")
    if not nodes:
        raise ValueError("DOM/accessibility snapshot contains no nodes")
    inferred_width, inferred_height = _infer_size(nodes)
    return UIGraph(
        graph_id=graph_id,
        nodes=tuple(nodes),
        width=width or inferred_width,
        height=height or inferred_height,
        metadata={
            "source_format": source_format,
            "screenshot_path": screenshot_path,
        },
    )


def merge_webview_graph(
    native: UIGraph,
    dom: UIGraph,
    *,
    host_node_id: str,
    transform: WebViewTransform,
    graph_id: str,
    screenshot_path: str = "",
) -> UIGraph:
    """Attach a transformed DOM subtree under one native WebView host node."""

    host = next((node for node in native.nodes if node.node_id == host_node_id), None)
    if host is None or host.bbox is None:
        raise ValueError("WebView host node with bounds is required")
    if transform.viewport_width <= 0.0 or transform.viewport_height <= 0.0:
        raise ValueError("WebView viewport dimensions must be positive")
    root_ids = tuple(node.node_id for node in dom.nodes if node.parent_id is None)
    prefixed_roots = tuple(f"webview:{node_id}" for node_id in root_ids)
    native_nodes = tuple(
        UINode(
            **{
                **node.__dict__,
                "child_ids": (
                    tuple(dict.fromkeys((*node.child_ids, *prefixed_roots)))
                    if node.node_id == host_node_id
                    else node.child_ids
                ),
            }
        )
        for node in native.nodes
    )
    dom_nodes = tuple(
        _transform_webview_node(
            node,
            host=host,
            transform=transform,
            host_node_id=host_node_id,
            root_ids=set(root_ids),
        )
        for node in dom.nodes
    )
    return UIGraph(
        graph_id=graph_id,
        nodes=(*native_nodes, *dom_nodes),
        width=native.width,
        height=native.height,
        metadata={
            **native.metadata,
            "source_format": "webview",
            "screenshot_path": screenshot_path or native.metadata.get("screenshot_path", ""),
            "native_graph_id": native.graph_id,
            "dom_graph_id": dom.graph_id,
            "webview_host_node_id": host_node_id,
        },
    )


def _nodes_from_flat_dom(items: Iterable[Any]) -> list[UINode]:
    rows = [item for item in items if isinstance(item, dict)]
    ids = [_dom_node_id(item, index=index) for index, item in enumerate(rows)]
    nodes: list[UINode] = []
    for index, item in enumerate(rows):
        parent_id = _optional_text(
            item.get("parent_id")
            or item.get("parentId")
            or item.get("parent")
        )
        children = item.get("children") or item.get("child_ids") or item.get("childIds") or ()
        child_ids = tuple(
            _optional_text(child.get("id") if isinstance(child, dict) else child)
            for child in children
        )
        child_ids = tuple(child_id for child_id in child_ids if child_id)
        nodes.append(
            _dom_node(
                item,
                node_id=ids[index],
                parent_id=parent_id,
                child_ids=child_ids,
                depth=int(item.get("depth") or 0),
            )
        )
    return nodes


def _nodes_from_nested_dom(root: dict[str, Any]) -> list[UINode]:
    nodes: list[UINode] = []

    def walk(item: dict[str, Any], *, path: str, parent_id: str | None, depth: int) -> None:
        node_id = _dom_node_id(item, fallback=path)
        children = [child for child in item.get("children") or () if isinstance(child, dict)]
        child_ids = tuple(
            _dom_node_id(child, fallback=f"{path}.{index}")
            for index, child in enumerate(children)
        )
        nodes.append(
            _dom_node(
                item,
                node_id=node_id,
                parent_id=parent_id,
                child_ids=child_ids,
                depth=depth,
            )
        )
        for index, child in enumerate(children):
            walk(child, path=f"{path}.{index}", parent_id=node_id, depth=depth + 1)

    walk(root, path="0", parent_id=None, depth=0)
    return nodes


def _dom_node(
    item: dict[str, Any],
    *,
    node_id: str,
    parent_id: str | None,
    child_ids: tuple[str, ...],
    depth: int,
) -> UINode:
    attributes = _attributes(item.get("attributes"))
    tag = str(
        item.get("tag")
        or item.get("tagName")
        or item.get("nodeName")
        or item.get("role")
        or "node"
    ).lower()
    role = str(item.get("role") or attributes.get("role") or tag).lower()
    text = str(
        item.get("text")
        or item.get("innerText")
        or item.get("value")
        or item.get("name")
        or ""
    )
    content_desc = str(
        item.get("accessible_name")
        or item.get("ax_name")
        or item.get("description")
        or attributes.get("aria-label")
        or attributes.get("title")
        or attributes.get("alt")
        or ""
    )
    resource_id = str(
        attributes.get("data-testid")
        or attributes.get("data-test")
        or attributes.get("id")
        or attributes.get("name")
        or ""
    )
    clickable = bool(
        item.get("clickable")
        or item.get("onclick")
        or attributes.get("onclick")
        or tag in {"a", "button", "summary", "option"}
        or role in {"button", "link", "menuitem", "option", "tab", "checkbox", "radio"}
    )
    editable = bool(
        item.get("editable")
        or tag in {"input", "textarea", "select"}
        or role in {"textbox", "combobox", "searchbox", "spinbutton"}
        or str(attributes.get("contenteditable") or "").lower() == "true"
    )
    scrollable = bool(
        item.get("scrollable")
        or str(item.get("overflow") or attributes.get("overflow") or "").lower()
        in {"auto", "scroll"}
    )
    disabled = item.get("disabled") or attributes.get("disabled")
    return UINode(
        node_id=node_id,
        origin_id=str(item.get("origin_id") or item.get("backendNodeId") or node_id),
        parent_id=parent_id,
        text=text,
        content_desc=content_desc,
        resource_id=resource_id,
        class_name=role,
        bbox=_bbox(item.get("bbox") or item.get("bounds") or item.get("rect") or item.get("boundingBox")),
        clickable=clickable,
        editable=editable,
        scrollable=scrollable,
        enabled=not _truthy(disabled),
        depth=depth,
        child_ids=child_ids,
        metadata={
            "source_format": "dom",
            "tag": tag,
            "role": role,
            "attributes": attributes,
        },
    )


def _transform_webview_node(
    node: UINode,
    *,
    host: UINode,
    transform: WebViewTransform,
    host_node_id: str,
    root_ids: set[str],
) -> UINode:
    host_left, host_top, host_right, host_bottom = host.bbox or (0.0, 0.0, 0.0, 0.0)
    scale_x = (host_right - host_left) / transform.viewport_width
    scale_y = (host_bottom - host_top) / transform.viewport_height
    bbox = None
    if node.bbox is not None:
        bbox = (
            host_left + (node.bbox[0] - transform.scroll_x) * scale_x,
            host_top + (node.bbox[1] - transform.scroll_y) * scale_y,
            host_left + (node.bbox[2] - transform.scroll_x) * scale_x,
            host_top + (node.bbox[3] - transform.scroll_y) * scale_y,
        )
    parent_id = host_node_id if node.node_id in root_ids else (
        f"webview:{node.parent_id}" if node.parent_id else host_node_id
    )
    return UINode(
        node_id=f"webview:{node.node_id}",
        origin_id=f"webview:{node.origin_id}",
        parent_id=parent_id,
        text=node.text,
        content_desc=node.content_desc,
        resource_id=node.resource_id,
        class_name=node.class_name,
        bbox=bbox,
        clickable=node.clickable,
        editable=node.editable,
        scrollable=node.scrollable,
        enabled=node.enabled,
        depth=host.depth + 1 + node.depth,
        child_ids=tuple(f"webview:{child_id}" for child_id in node.child_ids),
        metadata={**node.metadata, "source_format": "webview_dom"},
    )


def _dom_node_id(item: dict[str, Any], *, index: int = 0, fallback: str = "") -> str:
    return str(
        item.get("node_id")
        or item.get("backendNodeId")
        or item.get("backend_node_id")
        or item.get("id")
        or fallback
        or index
    )


def _attributes(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, list):
        return {
            str(value[index]): str(value[index + 1])
            for index in range(0, len(value) - 1, 2)
        }
    return {}


def _bbox(value: Any) -> BBox | None:
    if isinstance(value, dict):
        try:
            left = float(value.get("x", value.get("left", 0.0)))
            top = float(value.get("y", value.get("top", 0.0)))
            if value.get("right") is not None and value.get("bottom") is not None:
                right = float(value["right"])
                bottom = float(value["bottom"])
            else:
                right = left + float(value.get("width", 0.0))
                bottom = top + float(value.get("height", 0.0))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            left, top, right, bottom = map(float, value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    return (left, top, right, bottom) if right > left and bottom > top else None


def _infer_size(nodes: list[UINode]) -> tuple[float, float]:
    width = max((node.bbox or (0.0, 0.0, 1.0, 1.0))[2] for node in nodes)
    height = max((node.bbox or (0.0, 0.0, 1.0, 1.0))[3] for node in nodes)
    return max(1.0, width), max(1.0, height)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
