from io import BytesIO

from PIL import Image

from omnitransfer.os_atlas import (
    graph_from_os_atlas_record,
    materialize_os_atlas_image,
)
from omnitransfer.unified_ui import graph_from_unified_record


def _record() -> dict:
    return {
        "img_filename": "screen.png",
        "elements": [
            {
                "instruction": "Settings panel",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "data_type": "android.widget.FrameLayout",
            },
            {
                "instruction": "Search",
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "data_type": "android.widget.EditText",
            },
            {"instruction": "invalid", "bbox": [0.5, 0.5, 0.5, 0.8]},
        ],
    }


def test_os_atlas_adapter_scales_boxes_and_builds_relations() -> None:
    graph = graph_from_os_atlas_record(
        _record(),
        graph_id="os_atlas:aw:0",
        image_size=(100, 200),
        screenshot_path="images/screen.png",
    )

    assert graph.width == 100
    assert graph.height == 200
    assert graph.metadata["source_format"] == "os_atlas_mobile_grounding"
    assert graph.metadata["app_identity_available"] is False
    assert len(graph.nodes) == 2
    assert graph.nodes[0].child_ids == ("e1",)
    assert graph.nodes[1].parent_id == "e0"
    assert graph.nodes[1].bbox == (10.0, 40.0, 90.0, 80.0)
    assert graph.nodes[1].text == "Search"
    assert graph.nodes[1].clickable is True
    assert graph.nodes[1].editable is True


def test_os_atlas_unified_route_uses_same_graph_schema() -> None:
    graph = graph_from_unified_record(
        {
            **_record(),
            "source_format": "os_atlas_mobile_grounding",
            "graph_id": "os_atlas:unified",
            "image_width": 100,
            "image_height": 200,
            "screenshot_path": "images/screen.png",
        }
    )

    assert graph.graph_id == "os_atlas:unified"
    assert graph.metadata["screenshot_path"] == "images/screen.png"
    assert len(graph.nodes) == 2


def test_os_atlas_image_materialization_is_lossless_and_resized(tmp_path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(buffer, format="JPEG")
    source = tmp_path / "source.jpg"
    source.write_bytes(buffer.getvalue())
    destination = tmp_path / "screen.png"

    source_size = materialize_os_atlas_image(source, destination, long_side=100)

    assert source_size == (200, 100)
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.size == (100, 50)
