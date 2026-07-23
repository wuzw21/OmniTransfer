"""Streaming adapters for the official AndroidControl TFRecord release."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gzip
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterable, Iterator
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from omnitransfer.mobileviews import materialize_mobileviews_image
from omnitransfer.ui_graph import BBox, UIGraph, UINode


FeatureValue = list[bytes] | list[int]


def iter_android_control_examples(source: str | Path) -> Iterable[dict[str, FeatureValue]]:
    """Yield decoded TensorFlow Example features from one GZIP TFRecord shard."""

    with _open_gzip_source(source) as handle:
        for payload in _iter_tfrecord_payloads(handle):
            yield parse_tf_example(payload)


def parse_tf_example(payload: bytes) -> dict[str, FeatureValue]:
    """Decode the bytes-list and int64-list fields used by AndroidControl."""

    features_payload = _first_bytes_field(payload, 1)
    if features_payload is None:
        raise ValueError("TensorFlow Example does not contain Features")
    decoded: dict[str, FeatureValue] = {}
    for field_number, wire_type, entry_payload in _wire_fields(features_payload):
        if field_number != 1 or wire_type != 2:
            continue
        key_payload = _first_bytes_field(entry_payload, 1)
        feature_payload = _first_bytes_field(entry_payload, 2)
        if key_payload is None or feature_payload is None:
            continue
        key = key_payload.decode("utf-8", errors="replace")
        bytes_list = _first_bytes_field(feature_payload, 1)
        if bytes_list is not None:
            decoded[key] = [
                value
                for number, value_type, value in _wire_fields(bytes_list)
                if number == 1 and value_type == 2
            ]
            continue
        int_list = _first_bytes_field(feature_payload, 3)
        if int_list is not None:
            decoded[key] = _repeated_varints(int_list, 1)
    return decoded


def graph_from_android_control_forest(
    forest_payload: bytes,
    *,
    graph_id: str,
    width: int,
    height: int,
    episode_id: int,
    step_index: int,
    screenshot_path: str = "",
    official_split: str = "",
    test_subsplits: Iterable[str] = (),
) -> UIGraph:
    """Convert one serialized Android accessibility forest to a canonical graph."""

    if width <= 0 or height <= 0:
        raise ValueError("AndroidControl screen dimensions must be positive")
    raw_nodes: list[dict[str, Any]] = []
    for window_index, window_payload in enumerate(_repeated_bytes(forest_payload, 1)):
        tree_payload = _first_bytes_field(window_payload, 11)
        if tree_payload is None:
            continue
        window_type = _first_varint_field(window_payload, 6, default=0)
        window_layer = _first_varint_field(window_payload, 4, default=0)
        window_title = _decode_string(_first_bytes_field(window_payload, 5))
        for node_payload in _repeated_bytes(tree_payload, 1):
            node = _parse_accessibility_node(
                node_payload,
                window_index=window_index,
                width=width,
                height=height,
            )
            if node is None:
                continue
            node["window_type"] = window_type
            node["window_layer"] = window_layer
            node["window_title"] = window_title
            raw_nodes.append(node)
    if not raw_nodes:
        raise ValueError("AndroidControl forest contains no bounded nodes")

    valid_ids = {str(node["node_id"]) for node in raw_nodes}
    parent_by_id: dict[str, str] = {}
    for node in raw_nodes:
        parent_id = str(node["node_id"])
        for child_id in node["child_ids"]:
            if child_id in valid_ids and child_id not in parent_by_id:
                parent_by_id[child_id] = parent_id
    computed_depths = _depths_from_parents(valid_ids, parent_by_id)
    nodes = tuple(
        UINode(
            node_id=str(node["node_id"]),
            origin_id=str(node["node_id"]),
            parent_id=parent_by_id.get(str(node["node_id"])),
            text=str(node["text"]),
            content_desc=str(node["content_desc"]),
            resource_id=str(node["resource_id"]),
            class_name=str(node["class_name"]),
            bbox=node["bbox"],
            clickable=bool(node["clickable"]),
            editable=bool(node["editable"]),
            scrollable=bool(node["scrollable"]),
            enabled=bool(node["enabled"]),
            depth=computed_depths[str(node["node_id"])],
            child_ids=tuple(
                child_id for child_id in node["child_ids"] if child_id in valid_ids
            ),
            metadata={
                "package_name": node["package_name"],
                "hint_text": node["hint_text"],
                "checkable": node["checkable"],
                "focusable": node["focusable"],
                "long_clickable": node["long_clickable"],
                "selected": node["selected"],
                "visible": node["visible"],
                "window_index": node["window_index"],
                "window_type": node["window_type"],
                "window_layer": node["window_layer"],
                "window_title": node["window_title"],
                "drawing_order": node["drawing_order"],
                "source_depth": node["source_depth"],
            },
        )
        for node in raw_nodes
    )
    package_counts = Counter(
        str(node.metadata.get("package_name") or "")
        for node in nodes
        if str(node.metadata.get("package_name") or "")
    )
    package = package_counts.most_common(1)[0][0] if package_counts else ""
    return UIGraph(
        graph_id=graph_id,
        nodes=nodes,
        width=float(width),
        height=float(height),
        metadata={
            "source_format": "android_control_accessibility_forest",
            "dataset": "google-research/AndroidControl",
            "platform": "android",
            "package": package,
            "episode_id": episode_id,
            "trajectory_step": step_index,
            "split": official_split,
            "split_group": f"android_control_episode:{episode_id}",
            "test_subsplits": sorted(set(test_subsplits)),
            "screenshot_path": screenshot_path,
            "relation_source": "accessibility_child_ids",
        },
    )


def materialize_android_control_image(
    image_content: bytes,
    destination: Path,
    *,
    long_side: int = 384,
) -> tuple[int, int]:
    """Write one AndroidControl screenshot as a lossless pre-resized PNG."""

    return materialize_mobileviews_image(
        image_content,
        destination,
        long_side=long_side,
    )


def feature_bytes(features: dict[str, FeatureValue], name: str) -> list[bytes]:
    """Return one bytes-list feature with strict type checking."""

    values = features.get(name, [])
    if values and not isinstance(values[0], bytes):
        raise ValueError(f"AndroidControl feature {name!r} is not a bytes list")
    return list(values)


def feature_ints(features: dict[str, FeatureValue], name: str) -> list[int]:
    """Return one int64-list feature with strict type checking."""

    values = features.get(name, [])
    if values and isinstance(values[0], bytes):
        raise ValueError(f"AndroidControl feature {name!r} is not an int list")
    return [int(value) for value in values]


def source_name(source: str | Path) -> str:
    """Return a stable shard name for a local path or URL."""

    text = str(source)
    if text.startswith(("http://", "https://")):
        return Path(urlparse(text).path).name
    return Path(text).name


def _parse_accessibility_node(
    payload: bytes,
    *,
    window_index: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    unique_id = _first_varint_field(payload, 1, default=-1)
    bbox_payload = _first_bytes_field(payload, 2)
    bbox = _parse_bbox(bbox_payload, width=width, height=height)
    if unique_id < 0 or bbox is None:
        return None
    child_ids = tuple(
        f"w{window_index}:n{child_id}" for child_id in _repeated_varints(payload, 25)
    )
    return {
        "node_id": f"w{window_index}:n{unique_id}",
        "bbox": bbox,
        "class_name": _decode_string(_first_bytes_field(payload, 3)),
        "content_desc": _decode_string(_first_bytes_field(payload, 4)),
        "hint_text": _decode_string(_first_bytes_field(payload, 5)),
        "package_name": _decode_string(_first_bytes_field(payload, 6)),
        "text": _decode_string(_first_bytes_field(payload, 7)),
        "resource_id": _decode_string(_first_bytes_field(payload, 10)),
        "checkable": bool(_first_varint_field(payload, 12, default=0)),
        "clickable": bool(_first_varint_field(payload, 14, default=0)),
        "editable": bool(_first_varint_field(payload, 15, default=0)),
        "enabled": bool(_first_varint_field(payload, 16, default=1)),
        "focusable": bool(_first_varint_field(payload, 17, default=0)),
        "long_clickable": bool(_first_varint_field(payload, 19, default=0)),
        "scrollable": bool(_first_varint_field(payload, 21, default=0)),
        "selected": bool(_first_varint_field(payload, 22, default=0)),
        "visible": bool(_first_varint_field(payload, 23, default=1)),
        "child_ids": child_ids,
        "window_index": window_index,
        "source_depth": _first_varint_field(payload, 27, default=0),
        "drawing_order": _first_varint_field(payload, 30, default=0),
    }


def _parse_bbox(payload: bytes | None, *, width: int, height: int) -> BBox | None:
    if payload is None:
        return None
    left = _signed_int32(_first_varint_field(payload, 1, default=0))
    top = _signed_int32(_first_varint_field(payload, 2, default=0))
    right = _signed_int32(_first_varint_field(payload, 3, default=0))
    bottom = _signed_int32(_first_varint_field(payload, 4, default=0))
    x1 = float(max(0, min(width, left)))
    y1 = float(max(0, min(height, top)))
    x2 = float(max(0, min(width, right)))
    y2 = float(max(0, min(height, bottom)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _depths_from_parents(
    node_ids: set[str],
    parent_by_id: dict[str, str],
) -> dict[str, int]:
    depths: dict[str, int] = {}

    def resolve(node_id: str, trail: set[str]) -> int:
        if node_id in depths:
            return depths[node_id]
        parent_id = parent_by_id.get(node_id)
        if parent_id is None or parent_id not in node_ids or parent_id in trail:
            depth = 0
        else:
            depth = resolve(parent_id, {*trail, node_id}) + 1
        depths[node_id] = depth
        return depth

    for node_id in node_ids:
        resolve(node_id, set())
    return depths


@contextmanager
def _open_gzip_source(source: str | Path) -> Iterator[BinaryIO]:
    text = str(source)
    if text.startswith(("http://", "https://")):
        request = Request(text, headers={"User-Agent": "OmniTransfer/0.1"})
        with urlopen(request, timeout=120) as response:
            with gzip.GzipFile(fileobj=response, mode="rb") as handle:
                yield handle
        return
    with gzip.open(Path(source).expanduser(), "rb") as handle:
        yield handle


def _iter_tfrecord_payloads(handle: BinaryIO) -> Iterable[bytes]:
    while True:
        header = handle.read(12)
        if not header:
            return
        if len(header) != 12:
            raise ValueError("Truncated TFRecord header")
        length = struct.unpack("<Q", header[:8])[0]
        payload = _read_exact(handle, length)
        _read_exact(handle, 4)
        yield payload


def _read_exact(handle: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("Truncated TFRecord payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wire_fields(payload: bytes) -> Iterable[tuple[int, int, Any]]:
    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number <= 0:
            raise ValueError("Invalid protobuf field number")
        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(payload):
                raise ValueError("Truncated protobuf fixed64 field")
            value = payload[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_varint(payload, offset)
            end = offset + length
            if end > len(payload):
                raise ValueError("Truncated protobuf length-delimited field")
            value = payload[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(payload):
                raise ValueError("Truncated protobuf fixed32 field")
            value = payload[offset:end]
            offset = end
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload) and shift < 70:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("Invalid protobuf varint")


def _first_bytes_field(payload: bytes, field_number: int) -> bytes | None:
    return next(
        (
            value
            for number, wire_type, value in _wire_fields(payload)
            if number == field_number and wire_type == 2
        ),
        None,
    )


def _first_varint_field(payload: bytes, field_number: int, *, default: int) -> int:
    return next(
        (
            int(value)
            for number, wire_type, value in _wire_fields(payload)
            if number == field_number and wire_type == 0
        ),
        default,
    )


def _repeated_bytes(payload: bytes, field_number: int) -> list[bytes]:
    return [
        value
        for number, wire_type, value in _wire_fields(payload)
        if number == field_number and wire_type == 2
    ]


def _repeated_varints(payload: bytes, field_number: int) -> list[int]:
    values: list[int] = []
    for number, wire_type, value in _wire_fields(payload):
        if number != field_number:
            continue
        if wire_type == 0:
            values.append(int(value))
        elif wire_type == 2:
            offset = 0
            while offset < len(value):
                item, offset = _read_varint(value, offset)
                values.append(item)
    return values


def _decode_string(payload: bytes | None) -> str:
    return payload.decode("utf-8", errors="replace") if payload is not None else ""


def _signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value
