import gzip
from io import BytesIO
import struct

from PIL import Image

from omnitransfer.android_control import (
    graph_from_android_control_forest,
    iter_android_control_examples,
    materialize_android_control_image,
    parse_tf_example,
)


def test_tensorflow_example_and_gzip_tfrecord_are_parsed_without_tensorflow(
    tmp_path,
) -> None:
    payload = _example(
        {
            "episode_id": _int_feature([42]),
            "screenshots": _bytes_feature([b"png-a", b"png-b"]),
        }
    )
    path = tmp_path / "android_control-00000-of-00020"
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(b"\0" * 4)
        handle.write(payload)
        handle.write(b"\0" * 4)

    direct = parse_tf_example(payload)
    streamed = list(iter_android_control_examples(path))

    assert direct["episode_id"] == [42]
    assert direct["screenshots"] == [b"png-a", b"png-b"]
    assert streamed == [direct]


def test_android_control_forest_preserves_attributes_and_true_tree_relations() -> None:
    root = _node(
        unique_id=0,
        bbox=(0, 0, 100, 200),
        class_name="android.widget.FrameLayout",
        package="com.example",
        child_ids=[1],
        enabled=True,
    )
    child = _node(
        unique_id=1,
        bbox=(10, 30, 90, 80),
        class_name="android.widget.Button",
        package="com.example",
        text="Continue",
        content_desc="Continue setup",
        resource_id="com.example:id/continue",
        clickable=True,
        enabled=True,
    )
    tree = _message(1, root) + _message(1, child)
    window = _varint_field(4, 3) + _varint_field(6, 1) + _message(11, tree)
    forest = _message(1, window)

    graph = graph_from_android_control_forest(
        forest,
        graph_id="android_control:42:0",
        width=100,
        height=200,
        episode_id=42,
        step_index=0,
        screenshot_path="images/42.png",
        official_split="test",
        test_subsplits=("app_unseen", "task_unseen"),
    )

    assert graph.metadata["package"] == "com.example"
    assert graph.metadata["split_group"] == "android_control_episode:42"
    assert graph.metadata["test_subsplits"] == ["app_unseen", "task_unseen"]
    assert len(graph.nodes) == 2
    assert graph.nodes[0].child_ids == ("w0:n1",)
    assert graph.nodes[1].parent_id == "w0:n0"
    assert graph.nodes[1].depth == 1
    assert graph.nodes[1].bbox == (10.0, 30.0, 90.0, 80.0)
    assert graph.nodes[1].resource_id == "com.example:id/continue"
    assert graph.nodes[1].content_desc == "Continue setup"
    assert graph.nodes[1].clickable is True


def test_android_control_image_materialization_is_lossless_and_resized(tmp_path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (200, 100), color=(20, 30, 40)).save(buffer, format="PNG")
    destination = tmp_path / "screen.png"

    source_size = materialize_android_control_image(
        buffer.getvalue(), destination, long_side=100
    )

    assert source_size == (200, 100)
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.size == (100, 50)


def _node(
    *,
    unique_id: int,
    bbox: tuple[int, int, int, int],
    class_name: str,
    package: str,
    child_ids: list[int] | None = None,
    text: str = "",
    content_desc: str = "",
    resource_id: str = "",
    clickable: bool = False,
    enabled: bool = False,
) -> bytes:
    rect = b"".join(
        _varint_field(index, value) for index, value in enumerate(bbox, start=1)
    )
    payload = b"".join(
        (
            _varint_field(1, unique_id),
            _message(2, rect),
            _string_field(3, class_name),
            _string_field(4, content_desc),
            _string_field(6, package),
            _string_field(7, text),
            _string_field(10, resource_id),
            _varint_field(14, int(clickable)),
            _varint_field(16, int(enabled)),
            _varint_field(23, 1),
        )
    )
    if child_ids:
        packed = b"".join(_varint(value) for value in child_ids)
        payload += _message(25, packed)
    return payload


def _example(features: dict[str, bytes]) -> bytes:
    entries = b"".join(
        _message(1, _string_field(1, key) + _message(2, value))
        for key, value in features.items()
    )
    return _message(1, entries)


def _bytes_feature(values: list[bytes]) -> bytes:
    return _message(1, b"".join(_message(1, value) for value in values))


def _int_feature(values: list[int]) -> bytes:
    return _message(3, _message(1, b"".join(_varint(value) for value in values)))


def _string_field(field_number: int, value: str) -> bytes:
    return _message(field_number, value.encode()) if value else b""


def _message(field_number: int, payload: bytes) -> bytes:
    return _varint((field_number << 3) | 2) + _varint(len(payload)) + payload


def _varint_field(field_number: int, value: int) -> bytes:
    return _varint(field_number << 3) + _varint(value)


def _varint(value: int) -> bytes:
    payload = bytearray()
    while value > 0x7F:
        payload.append((value & 0x7F) | 0x80)
        value >>= 7
    payload.append(value)
    return bytes(payload)
