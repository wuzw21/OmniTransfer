"""UI hierarchy parsing utilities for graph-based relocation training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable


BBox = tuple[float, float, float, float]

_XML_OBSERVATION_ALIAS_PATHS = "_omnitransfer_observation_alias_paths"


@dataclass(frozen=True)
class UINode:
    """One UI element in a parsed hierarchy graph."""

    node_id: str
    origin_id: str
    parent_id: str | None = None
    text: str = ""
    content_desc: str = ""
    resource_id: str = ""
    class_name: str = ""
    bbox: BBox | None = None
    clickable: bool = False
    editable: bool = False
    scrollable: bool = False
    enabled: bool = True
    depth: int = 0
    child_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UIGraph:
    """A screen-level UI hierarchy graph."""

    graph_id: str
    nodes: tuple[UINode, ...]
    width: float | None = None
    height: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_by_origin(self) -> dict[str, UINode]:
        return {node.origin_id: node for node in self.nodes}


def visual_bbox_fraction(
    graph: UIGraph,
    node: UINode,
    *,
    image_width: float,
    image_height: float,
) -> BBox | None:
    """Resolve one node crop into normalized screenshot coordinates."""

    declared_visual_bbox = node.metadata.get("visual_bbox")
    if declared_visual_bbox is not None:
        bbox = _finite_bbox(declared_visual_bbox)
        if bbox is None:
            return None
        coordinate_space = str(
            node.metadata.get("visual_bbox_coordinate_space") or ""
        )
        if coordinate_space in {"page_pixels", "screenshot_pixels", "visual_pixels"}:
            display_size = graph.metadata.get("visual_display_size")
            if isinstance(display_size, (list, tuple)) and len(display_size) == 2:
                try:
                    bbox_width = float(display_size[0])
                    bbox_height = float(display_size[1])
                except (TypeError, ValueError):
                    return None
            else:
                bbox_width = float(image_width)
                bbox_height = float(image_height)
        else:
            bbox_width = float(graph.width or image_width)
            bbox_height = float(graph.height or image_height)
    else:
        bbox = _finite_bbox(node.bbox)
        if bbox is None:
            return None
        bbox_width = float(graph.width or image_width)
        bbox_height = float(graph.height or image_height)

    if bbox_width <= 0.0 or bbox_height <= 0.0:
        return None
    left, top, right, bottom = bbox
    tolerance = 1e-6
    if (
        left < -tolerance
        or top < -tolerance
        or right > bbox_width + tolerance
        or bottom > bbox_height + tolerance
    ):
        return None
    return (
        min(max(left / bbox_width, 0.0), 1.0),
        min(max(top / bbox_height, 0.0), 1.0),
        min(max(right / bbox_width, 0.0), 1.0),
        min(max(bottom / bbox_height, 0.0), 1.0),
    )


def _finite_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = bbox
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def graph_from_record(record: dict[str, Any], *, graph_id: str | None = None) -> UIGraph:
    """Parse one MobileViews/RICO-like record into a canonical UI graph.

    The parser is intentionally permissive: MobileViews follows DroidBot-style
    view hierarchy fields, while RICO-style dumps use nested JSON trees. Callers
    can also pass a record whose hierarchy payload is an XML string.
    """

    payload = _extract_hierarchy_payload(record)
    resolved_id = graph_id or str(
        record.get("id")
        or record.get("screen_id")
        or record.get("image_id")
        or record.get("app_package")
        or "screen"
    )
    width, height = _extract_screen_size(record, payload)

    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("<"):
            return _with_record_metadata(
                _graph_from_xml(
                    stripped,
                    graph_id=resolved_id,
                    width=width,
                    height=height,
                ),
                record,
            )
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("Hierarchy payload is neither XML nor JSON") from exc

    if isinstance(payload, list):
        nodes = _nodes_from_flat_list(payload)
    elif isinstance(payload, dict) and _looks_like_flat_node_container(payload):
        nodes = _nodes_from_flat_list(list(payload.get("nodes") or payload.get("views") or []))
    elif isinstance(payload, dict):
        nodes = _nodes_from_nested_tree(payload)
    else:
        raise ValueError("Unsupported hierarchy payload type")

    if width is None or height is None:
        width, height = _infer_screen_size(nodes, width=width, height=height)

    return _with_record_metadata(UIGraph(
        graph_id=resolved_id,
        nodes=tuple(nodes),
        width=width,
        height=height,
        metadata={"source_format": "json"},
    ), record)


def _with_record_metadata(graph: UIGraph, record: dict[str, Any]) -> UIGraph:
    metadata = dict(graph.metadata)
    screenshot_path = record.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path.strip():
        metadata["screenshot_path"] = screenshot_path.strip()
    display_width = _optional_float(
        record.get("display_width") or record.get("screenshot_width")
    )
    display_height = _optional_float(
        record.get("display_height") or record.get("screenshot_height")
    )
    if display_width is not None and display_height is not None:
        metadata["visual_display_size"] = (display_width, display_height)
    visual_rgb = record.get("visual_rgb")
    if isinstance(visual_rgb, dict):
        metadata["visual_rgb"] = dict(visual_rgb)
    return replace(graph, metadata=metadata)


def graph_to_record(graph: UIGraph) -> dict[str, Any]:
    """Serialize a UI graph into a JSON-friendly dictionary."""

    return {
        "graph_id": graph.graph_id,
        "width": graph.width,
        "height": graph.height,
        "nodes": [
            {
                "node_id": node.node_id,
                "origin_id": node.origin_id,
                "parent_id": node.parent_id,
                "text": node.text,
                "content_desc": node.content_desc,
                "resource_id": node.resource_id,
                "class_name": node.class_name,
                "bbox": list(node.bbox) if node.bbox else None,
                "clickable": node.clickable,
                "editable": node.editable,
                "scrollable": node.scrollable,
                "enabled": node.enabled,
                "depth": node.depth,
                "child_ids": list(node.child_ids),
                "metadata": dict(node.metadata),
            }
            for node in graph.nodes
        ],
        "metadata": dict(graph.metadata),
    }


def local_context_graph(
    graph: UIGraph,
    *,
    anchor_node_id: str,
    max_nodes: int = 48,
) -> UIGraph:
    """Return a bounded topology-and-layout neighborhood around one source node."""

    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if len(graph.nodes) <= max_nodes:
        return graph
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    anchor = nodes_by_id.get(anchor_node_id)
    if anchor is None:
        raise ValueError(f"anchor node {anchor_node_id!r} is absent from graph")
    neighbors: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for node in graph.nodes:
        if node.parent_id in nodes_by_id:
            neighbors[node.node_id].add(node.parent_id)
            neighbors[node.parent_id].add(node.node_id)
        for child_id in node.child_ids:
            if child_id in nodes_by_id:
                neighbors[node.node_id].add(child_id)
                neighbors[child_id].add(node.node_id)
    hops = {anchor_node_id: 0}
    frontier = [anchor_node_id]
    while frontier:
        node_id = frontier.pop(0)
        for neighbor_id in sorted(neighbors[node_id]):
            if neighbor_id in hops:
                continue
            hops[neighbor_id] = hops[node_id] + 1
            frontier.append(neighbor_id)
    anchor_center = _normalized_center(anchor.bbox, graph)
    ranked = sorted(
        graph.nodes,
        key=lambda node: (
            hops.get(node.node_id, math.inf),
            _center_distance(anchor_center, _normalized_center(node.bbox, graph)),
            node.node_id,
        ),
    )
    selected_ids = {node.node_id for node in ranked[:max_nodes]}
    selected_ids.add(anchor_node_id)
    selected_nodes = tuple(
        UINode(
            **{
                **node.__dict__,
                "parent_id": node.parent_id if node.parent_id in selected_ids else None,
                "child_ids": tuple(
                    child_id for child_id in node.child_ids if child_id in selected_ids
                ),
            }
        )
        for node in graph.nodes
        if node.node_id in selected_ids
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=selected_nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "context_anchor_node_id": anchor_node_id,
            "context_original_nodes": len(graph.nodes),
            "context_max_nodes": max_nodes,
        },
    )


def multi_anchor_context_graph(
    graph: UIGraph,
    *,
    anchor_node_ids: Iterable[str],
    max_nodes: int,
) -> UIGraph:
    """Keep every anchor plus a bounded topology neighborhood around them."""

    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    anchor_ids = tuple(dict.fromkeys(str(node_id) for node_id in anchor_node_ids))
    if not anchor_ids:
        raise ValueError("at least one anchor node is required")
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    missing = [node_id for node_id in anchor_ids if node_id not in nodes_by_id]
    if missing:
        raise ValueError(f"anchor nodes are absent from graph: {missing!r}")
    if len(graph.nodes) <= max_nodes or len(anchor_ids) >= max_nodes:
        selected_ids = set(anchor_ids) if len(anchor_ids) >= max_nodes else set(nodes_by_id)
    else:
        neighbors: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
        for node in graph.nodes:
            if node.parent_id in nodes_by_id:
                neighbors[node.node_id].add(node.parent_id)
                neighbors[node.parent_id].add(node.node_id)
            for child_id in node.child_ids:
                if child_id in nodes_by_id:
                    neighbors[node.node_id].add(child_id)
                    neighbors[child_id].add(node.node_id)
        selected_ids = set(anchor_ids)
        frontier = list(anchor_ids)
        while frontier and len(selected_ids) < max_nodes:
            node_id = frontier.pop(0)
            for neighbor_id in sorted(neighbors[node_id]):
                if neighbor_id in selected_ids:
                    continue
                selected_ids.add(neighbor_id)
                frontier.append(neighbor_id)
                if len(selected_ids) >= max_nodes:
                    break
        if len(selected_ids) < max_nodes:
            anchor_centers = [
                _normalized_center(nodes_by_id[node_id].bbox, graph)
                for node_id in anchor_ids
            ]
            remaining = sorted(
                (
                    node
                    for node in graph.nodes
                    if node.node_id not in selected_ids
                ),
                key=lambda node: (
                    min(
                        _center_distance(
                            anchor_center,
                            _normalized_center(node.bbox, graph),
                        )
                        for anchor_center in anchor_centers
                    ),
                    node.node_id,
                ),
            )
            selected_ids.update(
                node.node_id for node in remaining[: max_nodes - len(selected_ids)]
            )
    selected_nodes = tuple(
        UINode(
            **{
                **node.__dict__,
                "parent_id": node.parent_id if node.parent_id in selected_ids else None,
                "child_ids": tuple(
                    child_id for child_id in node.child_ids if child_id in selected_ids
                ),
            }
        )
        for node in graph.nodes
        if node.node_id in selected_ids
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=selected_nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "context_anchor_node_ids": anchor_ids,
            "context_original_nodes": len(graph.nodes),
            "context_max_nodes": max_nodes,
        },
    )


def _extract_hierarchy_payload(record: dict[str, Any]) -> Any:
    for key in (
        "view_hierarchy",
        "viewHierarchy",
        "vh",
        "hierarchy",
        "ui_hierarchy",
        "xml",
        "nodes",
        "views",
    ):
        if key in record and record[key] not in (None, ""):
            value = record[key]
            if key in {"nodes", "views"}:
                return {key: value}
            return value
    raise ValueError("Record does not contain a UI hierarchy payload")


def _extract_screen_size(record: dict[str, Any], payload: Any) -> tuple[float | None, float | None]:
    width = _optional_float(
        record.get("width")
        or record.get("screen_width")
        or record.get("screenshot_width")
        or record.get("device_width")
    )
    height = _optional_float(
        record.get("height")
        or record.get("screen_height")
        or record.get("screenshot_height")
        or record.get("device_height")
    )
    if isinstance(payload, dict):
        width = width or _optional_float(payload.get("width") or payload.get("screen_width"))
        height = height or _optional_float(payload.get("height") or payload.get("screen_height"))
    return width, height


def _looks_like_flat_node_container(payload: dict[str, Any]) -> bool:
    values = payload.get("nodes") or payload.get("views")
    return isinstance(values, list)


def _nodes_from_flat_list(items: list[Any]) -> list[UINode]:
    raw_nodes = [item for item in items if isinstance(item, dict)]
    declared_ids = [
        _node_id_from_payload(item, fallback=str(index))
        for index, item in enumerate(raw_nodes)
    ]
    node_ids: list[str] = []
    used_ids: set[str] = set()
    for index, declared_id in enumerate(declared_ids):
        node_id = declared_id
        suffix = index
        while node_id in used_ids:
            node_id = f"{declared_id}#{suffix}"
            suffix += 1
        node_ids.append(node_id)
        used_ids.add(node_id)
    internal_ids_by_declared: dict[str, list[tuple[int, str]]] = {}
    for index, (declared_id, node_id) in enumerate(
        zip(declared_ids, node_ids, strict=True)
    ):
        internal_ids_by_declared.setdefault(declared_id, []).append((index, node_id))

    def resolve_parent(index: int, item: dict[str, Any]) -> str | None:
        declared_parent = _parent_id_from_payload(item)
        if declared_parent is None:
            return None
        candidates = internal_ids_by_declared.get(declared_parent, ())
        if not candidates:
            return declared_parent if declared_parent in used_ids else None
        preceding = [candidate for candidate in candidates if candidate[0] < index]
        return (preceding[-1] if preceding else candidates[0])[1]

    children_by_parent: dict[str | None, list[str]] = {}
    parent_by_id = {
        node_id: resolve_parent(index, item)
        for index, (item, node_id) in enumerate(
            zip(raw_nodes, node_ids, strict=True)
        )
    }
    depths = _depths_from_parents(parent_by_id)
    for node_id, parent_id in parent_by_id.items():
        children_by_parent.setdefault(parent_id, []).append(node_id)

    nodes: list[UINode] = []
    for index, item in enumerate(raw_nodes):
        node_id = node_ids[index]
        parent_id = parent_by_id[node_id]
        explicit_depth = _optional_float(item.get("depth"))
        nodes.append(
            _node_from_payload(
                item,
                node_id=node_id,
                parent_id=parent_id,
                child_ids=tuple(children_by_parent.get(node_id, ())),
                depth=int(explicit_depth) if explicit_depth is not None else depths[node_id],
            )
        )
    return nodes


def _nodes_from_nested_tree(root: dict[str, Any]) -> list[UINode]:
    nodes: list[UINode] = []

    def walk(item: dict[str, Any], path: str, parent_id: str | None, depth: int) -> str:
        node_id = _node_id_from_payload(item, fallback=path)
        children = [child for child in _iter_children(item) if isinstance(child, dict)]
        child_ids = tuple(
            _node_id_from_payload(child, fallback=f"{path}.{index}")
            for index, child in enumerate(children)
        )
        nodes.append(
            _node_from_payload(
                item,
                node_id=node_id,
                parent_id=parent_id,
                child_ids=child_ids,
                depth=depth,
            )
        )
        for index, child in enumerate(children):
            walk(child, f"{path}.{index}", node_id, depth + 1)
        return node_id

    walk(root, "0", None, 0)
    return nodes


def _graph_from_xml(xml_text: str, *, graph_id: str, width: float | None, height: float | None) -> UIGraph:
    raw_root = ET.fromstring(xml_text)
    root, aliases_collapsed = _canonicalize_xml_capture_aliases(raw_root)
    nodes: list[UINode] = []

    declared_width, declared_height = _xml_screen_size(root)
    width = width or declared_width
    height = height or declared_height

    def walk(element: ET.Element, path: str, parent_id: str | None, depth: int) -> str:
        attrs = dict(element.attrib)
        attrs.setdefault("class", element.tag.rsplit("}", 1)[-1])
        attrs.setdefault(
            "origin_id",
            attrs.get("id")
            or attrs.get("resource-id")
            or attrs.get("name")
            or attrs.get("index")
            or path,
        )
        node_id = path
        child_ids = tuple(f"{path}.{index}" for index, _ in enumerate(list(element)))
        nodes.append(
            _node_from_payload(
                attrs,
                node_id=node_id,
                parent_id=parent_id,
                child_ids=child_ids,
                depth=depth,
            )
        )
        for index, child in enumerate(list(element)):
            walk(child, f"{path}.{index}", node_id, depth + 1)
        return node_id

    walk(root, "0", None, 0)
    if width is None or height is None:
        width, height = _infer_screen_size(nodes, width=width, height=height)
    return UIGraph(
        graph_id=graph_id,
        nodes=tuple(nodes),
        width=width,
        height=height,
        metadata={
            "source_format": "xml",
            "observation_aliases_collapsed": aliases_collapsed,
        },
    )


def _canonicalize_xml_capture_aliases(
    root: ET.Element,
) -> tuple[ET.Element, int]:
    """Collapse repeated observations of the same visible XML hit region.

    Some Android dumps concatenate several captures of one scroll container
    under the same window root.  Those copies are Raw Node aliases, not distinct
    visible controls.  A stable resource ID is only used to identify aliases
    inside one observation and one exact hit region; it is never used as a
    cross-observation correspondence answer.
    """

    aliases_collapsed = 0

    def canonical(
        aliases: list[tuple[ET.Element, str]],
    ) -> ET.Element:
        nonlocal aliases_collapsed
        first, _ = aliases[0]
        element = copy.copy(first)
        element[:] = []
        original_paths = tuple(path for _, path in aliases)
        if len(original_paths) > 1:
            aliases_collapsed += len(original_paths) - 1
            element.attrib[_XML_OBSERVATION_ALIAS_PATHS] = json.dumps(
                original_paths,
                ensure_ascii=False,
            )

        child_groups: list[list[tuple[ET.Element, str]]] = []
        child_group_indices: dict[tuple[Any, ...], int] = {}
        for alias, alias_path in aliases:
            for child_index, child in enumerate(list(alias)):
                child_path = f"{alias_path}.{child_index}"
                key = _xml_capture_alias_key(child)
                group_index = child_group_indices.get(key)
                if group_index is None:
                    child_group_indices[key] = len(child_groups)
                    child_groups.append([(child, child_path)])
                else:
                    child_groups[group_index].append((child, child_path))
        for group in child_groups:
            element.append(canonical(group))
        return element

    return canonical([(root, "0")]), aliases_collapsed


def _xml_capture_alias_key(element: ET.Element) -> tuple[Any, ...]:
    attributes = element.attrib
    class_name = str(attributes.get("class") or element.tag.rsplit("}", 1)[-1])
    resource_id = str(
        attributes.get("resource-id")
        or attributes.get("resource_id")
        or ""
    )
    bbox = _bounds_from_payload(attributes)
    if resource_id and bbox is not None:
        return (
            "stable-hit-region",
            class_name,
            resource_id,
            bbox,
            str(attributes.get("text") or ""),
            str(attributes.get("content-desc") or attributes.get("content_desc") or ""),
            str(attributes.get("clickable") or ""),
            str(attributes.get("enabled") or ""),
            str(attributes.get("selected") or ""),
            str(attributes.get("checked") or ""),
        )
    return ("exact-subtree", _xml_subtree_fingerprint(element))


def _xml_subtree_fingerprint(element: ET.Element) -> tuple[Any, ...]:
    attributes = tuple(
        sorted(
            (str(key), str(value))
            for key, value in element.attrib.items()
            if str(key) not in {"index", _XML_OBSERVATION_ALIAS_PATHS}
        )
    )
    return (
        element.tag,
        attributes,
        tuple(_xml_subtree_fingerprint(child) for child in list(element)),
    )


def _node_from_payload(
    payload: dict[str, Any],
    *,
    node_id: str,
    parent_id: str | None,
    child_ids: tuple[str, ...],
    depth: int,
) -> UINode:
    class_name = _text_from_payload(
        payload,
        ("class", "class_name", "className", "viewClass", "type"),
    )
    return UINode(
        node_id=node_id,
        origin_id=str(
            payload.get("origin_id")
            or payload.get("origin-id")
            or payload.get("originId")
            or node_id
        ),
        parent_id=parent_id,
        text=_text_from_payload(payload, ("text", "viewText", "label", "name")),
        content_desc=_text_from_payload(
            payload,
            (
                "content-desc",
                "content_desc",
                "contentDescription",
                "content_description",
                "description",
            ),
        ),
        resource_id=_text_from_payload(
            payload,
            ("resource-id", "resource_id", "resourceId", "view_id", "viewId"),
        ),
        class_name=class_name,
        bbox=_bounds_from_payload(payload),
        clickable=_clickable_from_payload(payload, class_name),
        editable=_bool_from_payload(
            payload,
            ("editable", "is_editable", "isEditable", "input"),
        ),
        scrollable=_bool_from_payload(
            payload,
            ("scrollable", "is_scrollable", "isScrollable"),
        ),
        enabled=not _explicit_false(
            _first_present(payload, ("enabled", "is_enabled", "isEnabled"))
        ),
        depth=depth,
        child_ids=child_ids,
        metadata=_node_metadata(payload),
    )


def _node_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    visible_raw = _first_present(payload, ("visible", "is_visible", "isVisible"))
    metadata: dict[str, Any] = {
        "raw_keys": tuple(sorted(str(key) for key in payload.keys())),
        "visible": not _explicit_false(visible_raw),
        "visible_present": visible_raw is not None,
    }
    declared_metadata = payload.get("metadata")
    if isinstance(declared_metadata, dict):
        metadata.update(declared_metadata)
    state_aliases = {
        "checked": ("checked", "is_checked", "isChecked"),
        "selected": ("selected", "is_selected", "isSelected"),
        "focused": ("focused", "is_focused", "isFocused"),
        "expanded": ("expanded", "is_expanded", "isExpanded"),
        "password": ("password", "is_password", "isPassword", "secure"),
    }
    for name, aliases in state_aliases.items():
        raw_value = _first_present(payload, aliases)
        metadata[f"{name}_present"] = raw_value is not None
        metadata[name] = _truthy(raw_value) if raw_value is not None else False
    long_clickable = _first_present(
        payload,
        ("long-clickable", "long_clickable", "isLongClickable"),
    )
    metadata["long_clickable"] = (
        _truthy(long_clickable) if long_clickable is not None else False
    )
    visual_bbox = _parse_bounds(
        _first_present(payload, ("visual-bbox", "visual_bbox"))
    )
    if visual_bbox is not None:
        metadata["visual_bbox"] = visual_bbox
    bbox_coordinate_space = _first_present(
        payload,
        (
            "bbox-coordinate-space",
            "bbox_coordinate_space",
            "xml-bbox-coordinate-space",
            "xml_bbox_coordinate_space",
            "coordinate-space",
            "coordinate_space",
        ),
    )
    if bbox_coordinate_space is not None:
        metadata["bbox_coordinate_space"] = str(bbox_coordinate_space)
    visual_bbox_coordinate_space = _first_present(
        payload,
        (
            "visual-bbox-coordinate-space",
            "visual_bbox_coordinate_space",
            "visual-coordinate-space",
            "visual_coordinate_space",
        ),
    )
    if visual_bbox_coordinate_space is not None:
        metadata["visual_bbox_coordinate_space"] = str(
            visual_bbox_coordinate_space
        )
    if "candidate" in payload:
        metadata["candidate"] = _truthy(payload.get("candidate"))
    if "node-id" in payload:
        metadata["declared_node_id"] = str(payload["node-id"])
    raw_alias_paths = payload.get(_XML_OBSERVATION_ALIAS_PATHS)
    if raw_alias_paths:
        try:
            alias_paths = tuple(str(value) for value in json.loads(raw_alias_paths))
        except (TypeError, ValueError, json.JSONDecodeError):
            alias_paths = ()
        if alias_paths:
            metadata["observation_alias_paths"] = alias_paths
            metadata["observation_alias_count"] = len(alias_paths)
    return metadata


def _node_id_from_payload(payload: dict[str, Any], *, fallback: str) -> str:
    return str(
        payload.get("node_id")
        or payload.get("nodeId")
        or payload.get("hash")
        or payload.get("temp_id")
        or payload.get("resource-id")
        or payload.get("resource_id")
        or fallback
    )


def _parent_id_from_payload(payload: dict[str, Any]) -> str | None:
    value = _first_present(
        payload,
        ("parent_id", "parent", "parentId", "parent_hash"),
    )
    if value in (None, "", -1, "-1"):
        return None
    return str(value)


def _depths_from_parents(parent_by_id: dict[str, str | None]) -> dict[str, int]:
    depths: dict[str, int] = {}

    def resolve(node_id: str, trail: set[str]) -> int:
        if node_id in depths:
            return depths[node_id]
        parent_id = parent_by_id.get(node_id)
        if parent_id is None or parent_id not in parent_by_id or parent_id in trail:
            depth = 0
        else:
            depth = resolve(parent_id, {*trail, node_id}) + 1
        depths[node_id] = depth
        return depth

    for node_id in parent_by_id:
        resolve(node_id, set())
    return depths


def _iter_children(payload: dict[str, Any]) -> Iterable[Any]:
    for key in ("children", "child", "nodes"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return ()


def _text_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_present(payload, keys)
    if value is None:
        return ""
    return str(value)


def _bool_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return _truthy(_first_present(payload, keys))


def _clickable_from_payload(payload: dict[str, Any], class_name: str) -> bool:
    """Read clickability, inferring it for UI formats with typed controls.

    XCTest XML commonly identifies buttons with ``type``/``class`` but does
    not emit Android's explicit ``clickable`` attribute. Treating those
    controls as non-clickable creates a false cross-platform state mismatch.
    Explicit values still win, including an explicit ``false``.
    """

    keys = ("clickable", "is_clickable", "isClickable")
    if any(key in payload for key in keys):
        return _bool_from_payload(payload, keys)
    normalized = re.sub(r"[^a-z0-9]", "", str(class_name).lower())
    return normalized.endswith(
        (
            "button",
            "checkbox",
            "link",
            "menuitem",
            "radiobutton",
            "searchfield",
            "securetextfield",
            "slider",
            "spinner",
            "stepper",
            "switch",
            "textfield",
        )
    )


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _explicit_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return str(value).strip().lower() in {"0", "false", "no", "n"}


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bounds(value: Any) -> BBox | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
        if len(numbers) >= 4:
            return _ordered_bbox(numbers[0], numbers[1], numbers[2], numbers[3])
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value)
    ):
        try:
            return _ordered_bbox(
                float(value[0][0]),
                float(value[0][1]),
                float(value[1][0]),
                float(value[1][1]),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return _ordered_bbox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        left = _optional_float(value.get("left") or value.get("x1") or value.get("x"))
        top = _optional_float(value.get("top") or value.get("y1") or value.get("y"))
        right = _optional_float(value.get("right") or value.get("x2"))
        bottom = _optional_float(value.get("bottom") or value.get("y2"))
        width = _optional_float(value.get("width") or value.get("w"))
        height = _optional_float(value.get("height") or value.get("h"))
        if left is not None and top is not None and right is not None and bottom is not None:
            return _ordered_bbox(left, top, right, bottom)
        if left is not None and top is not None and width is not None and height is not None:
            return _ordered_bbox(left, top, left + width, top + height)
    return None


def _bounds_from_payload(payload: dict[str, Any]) -> BBox | None:
    bounds = _parse_bounds(
        _first_present(
            payload,
            (
                "bounds",
                "bbox",
                "box",
                "frame",
                "visible_bounds",
                "visibleBounds",
                "bounds_in_screen",
            ),
        )
    )
    if bounds is not None:
        return bounds
    left = _optional_float(payload.get("x"))
    top = _optional_float(payload.get("y"))
    width = _optional_float(payload.get("width"))
    height = _optional_float(payload.get("height"))
    if left is None or top is None or width is None or height is None:
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return _ordered_bbox(left, top, left + width, top + height)


def _xml_screen_size(root: ET.Element) -> tuple[float | None, float | None]:
    width = _optional_float(root.attrib.get("width"))
    height = _optional_float(root.attrib.get("height"))
    if width and height and width > 0.0 and height > 0.0:
        return width, height
    for element in root.iter():
        class_name = str(
            element.attrib.get("type")
            or element.attrib.get("class")
            or element.tag.rsplit("}", 1)[-1]
        ).lower()
        if not (class_name.endswith("application") or class_name.endswith("window")):
            continue
        bounds = _bounds_from_payload(dict(element.attrib))
        if bounds is not None and bounds[2] > 0.0 and bounds[3] > 0.0:
            return bounds[2], bounds[3]
    return None, None


def _ordered_bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _normalized_center(
    bbox: BBox | None,
    graph: UIGraph,
) -> tuple[float, float] | None:
    if bbox is None:
        return None
    width = float(
        graph.width
        or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[2] for node in graph.nodes)
    )
    height = float(
        graph.height
        or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[3] for node in graph.nodes)
    )
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        (bbox[0] + bbox[2]) / (2.0 * width),
        (bbox[1] + bbox[3]) / (2.0 * height),
    )


def _center_distance(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> float:
    if first is None or second is None:
        return math.inf
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _infer_screen_size(
    nodes: list[UINode],
    *,
    width: float | None,
    height: float | None,
) -> tuple[float | None, float | None]:
    max_x = max((node.bbox[2] for node in nodes if node.bbox), default=0.0)
    max_y = max((node.bbox[3] for node in nodes if node.bbox), default=0.0)
    return width or max_x or None, height or max_y or None
