from __future__ import annotations

import json
from pathlib import Path

from omnitransfer.ios_adapter import correct_ios_coordinate_spaces
from scripts.build_vision_widget_pair_pool import _load_screens


def test_scale_normalized_bboxes_restores_retina_pixel_boxes_to_xml_space() -> None:
    graph = {
        "width": 414.0,
        "height": 736.0,
        "nodes": [
            {
                "bbox": [0.0, 330.0, 621.0, 468.0],
                "visual_bbox": [0.0, 330.0, 621.0, 468.0],
            }
        ],
    }

    correct_ios_coordinate_spaces(
        graph,
        visual_width=1242.0,
        visual_height=2208.0,
    )

    assert graph["nodes"][0]["bbox"] == [0.0, 110.0, 207.0, 156.0]
    assert graph["nodes"][0]["visual_bbox"] == [0.0, 330.0, 621.0, 468.0]


def test_scale_normalized_bboxes_expands_relative_boxes_in_xml_space() -> None:
    graph = {
        "width": 414.0,
        "height": 736.0,
        "nodes": [
            {
                "bbox": [0.0, 0.25, 0.5, 0.5],
                "visual_bbox": [0.0, 0.25, 0.5, 0.5],
            }
        ],
    }

    correct_ios_coordinate_spaces(
        graph,
        visual_width=1242.0,
        visual_height=2208.0,
    )

    assert graph["nodes"][0]["bbox"] == [0.0, 184.0, 207.0, 368.0]
    assert graph["nodes"][0]["visual_bbox"] == [0.0, 552.0, 621.0, 1104.0]


def test_vision_widget_pool_builder_routes_ios_rows_through_adapter(tmp_path: Path) -> None:
    xml_path = tmp_path / "screen.xml"
    xml_path.write_text(
        '<AppiumAUT><XCUIElementTypeButton label="Go" x="0" y="330" width="621" height="138"/></AppiumAUT>',
        encoding="utf-8",
    )
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(b"placeholder")
    screens_path = tmp_path / "screens.jsonl"
    screens_path.write_text(
        json.dumps(
            {
                "screen_id": "ios-screen",
                "platform": "ios",
                "xml_path": "screen.xml",
                "screenshot_path": "screen.png",
                "original_xml_width": 414,
                "original_xml_height": 736,
                "screenshot_width": 1242,
                "screenshot_height": 2208,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    screen = _load_screens(screens_path, tmp_path)["ios-screen"]

    assert screen["adapter"]["name"] == "omnitransfer.ios_screen_adapter.v1"
    node = screen["graph"]["nodes"][1]
    assert node["bbox"] == [0.0, 110.0, 207.0, 156.0]
