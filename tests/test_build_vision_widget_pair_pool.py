from __future__ import annotations

from scripts.build_vision_widget_pair_pool import _scale_normalized_bboxes


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

    _scale_normalized_bboxes(
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

    _scale_normalized_bboxes(
        graph,
        visual_width=1242.0,
        visual_height=2208.0,
    )

    assert graph["nodes"][0]["bbox"] == [0.0, 184.0, 207.0, 368.0]
    assert graph["nodes"][0]["visual_bbox"] == [0.0, 552.0, 621.0, 1104.0]
