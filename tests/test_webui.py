from io import BytesIO

from PIL import Image
import pytest

from omnitransfer.unified_ui import graph_from_unified_record
from omnitransfer.webui import (
    graph_from_webui_record,
    materialize_webui_image,
    webui_device_scale,
)


def _record() -> dict:
    return {
        "labels": [["generic"], ["StaticText", "link"], ["textbox"]],
        "contentBoxes": [
            [0, 0, 100, 200],
            [10, 20, 80, 60],
            [20, 220, 90, 260],
        ],
        "key_name": "default_1280-720",
    }


def test_webui_adapter_keeps_only_visible_boxes_and_semantic_roles() -> None:
    graph = graph_from_webui_record(
        _record(),
        graph_id="webui:0",
        image_size=(100, 200),
        screenshot_path="images/0.png",
    )

    assert graph.width == 100
    assert graph.height == 200
    assert graph.metadata["source_format"] == "webui_element_boxes"
    assert graph.metadata["official_domain_disjoint_split"] is True
    assert len(graph.nodes) == 2
    assert graph.nodes[1].node_id == "e1"
    assert graph.nodes[1].class_name == "StaticText link"
    assert graph.nodes[1].clickable is True
    assert graph.nodes[1].bbox == (10.0, 20.0, 80.0, 60.0)


def test_webui_unified_route_uses_same_graph_schema() -> None:
    graph = graph_from_unified_record(
        {
            **_record(),
            "source_format": "webui_element_boxes",
            "graph_id": "webui:unified",
            "image_width": 100,
            "image_height": 200,
            "screenshot_path": "images/unified.png",
        }
    )

    assert graph.graph_id == "webui:unified"
    assert graph.metadata["screenshot_path"] == "images/unified.png"
    assert len(graph.nodes) == 2


def test_webui_official_device_scale_and_unknown_viewport_are_explicit() -> None:
    assert webui_device_scale("default_1536-864") == 1.0
    assert webui_device_scale("iPad-Pro") == 2.0
    assert webui_device_scale("iPhone-SE") == 3.0
    with pytest.raises(ValueError, match="Unsupported WebUI viewport"):
        webui_device_scale("unknown-device")


def test_webui_image_materialization_is_lossless_and_resized(tmp_path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(buffer, format="JPEG")
    destination = tmp_path / "screen.png"

    source_size = materialize_webui_image(
        {"bytes": buffer.getvalue(), "path": None},
        destination,
        long_side=100,
    )

    assert source_size == (200, 100)
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.size == (100, 50)
