import json

from omnitransfer.mobileviews import (
    attach_mobileviews_image,
    graph_from_mobileviews_record,
    split_name_for_group,
)


def test_mobileviews_flat_droidbot_record_preserves_relations_and_package() -> None:
    record = {
        "json_content": json.dumps(
            {
                "foreground_activity": "com.example.app/.MainActivity",
                "width": 2340,
                "height": 1080,
                "views": [
                    {
                        "temp_id": 0,
                        "parent": -1,
                        "children": [1],
                        "package": "com.example.app",
                        "class": "android.widget.FrameLayout",
                        "bounds": [[0, 0], [1080, 1920]],
                    },
                    {
                        "temp_id": 1,
                        "parent": 0,
                        "children": [],
                        "package": "com.example.app",
                        "class": "android.widget.Button",
                        "resource_id": "com.example.app:id/continue_button",
                        "content_description": "Continue",
                        "clickable": True,
                        "bounds": [[40, 100], [360, 220]],
                    },
                    {
                        "temp_id": 2,
                        "parent": 0,
                        "children": [],
                        "package": "com.example.app",
                        "class": "android.widget.TextView",
                        "visible": False,
                        "bounds": [[0, 2000], [1080, 2100]],
                    },
                ],
            }
        ),
        "image_content": b"unused in graph parsing",
    }

    graph = graph_from_mobileviews_record(record, graph_id="mobileviews:0")

    assert graph.width == 1080
    assert graph.height == 1920
    assert graph.metadata["package"] == "com.example.app"
    assert graph.nodes[0].parent_id is None
    assert graph.nodes[0].child_ids == ("1",)
    assert graph.nodes[1].parent_id == "0"
    assert graph.nodes[1].depth == 1
    assert graph.nodes[1].bbox == (40.0, 100.0, 360.0, 220.0)
    assert graph.nodes[1].content_desc == "Continue"
    assert graph.nodes[1].clickable is True
    assert len(graph.nodes) == 2
    assert graph.metadata["invisible_nodes_removed"] == 1


def test_mobileviews_nested_core_record_supports_camel_case_fields() -> None:
    graph = graph_from_mobileviews_record(
        {
            "json_content": {
                "viewHierarchy": {
                    "package": "com.example.webview",
                    "viewClass": "android.webkit.WebView",
                    "bounds": [0, 0, 400, 800],
                    "children": [
                        {
                            "viewClass": "android.widget.TextView",
                            "viewText": "Account",
                            "resourceId": "com.example.webview:id/account",
                            "contentDescription": "Open account",
                            "isClickable": True,
                            "bounds": [20, 40, 220, 120],
                        }
                    ],
                }
            }
        },
        graph_id="mobileviews:nested",
    )

    assert graph.metadata["package"] == "com.example.webview"
    assert graph.nodes[1].class_name == "android.widget.TextView"
    assert graph.nodes[1].text == "Account"
    assert graph.nodes[1].resource_id == "com.example.webview:id/account"
    assert graph.nodes[1].content_desc == "Open account"
    assert graph.nodes[1].clickable is True


def test_mobileviews_image_geometry_and_package_split_are_explicit() -> None:
    graph = graph_from_mobileviews_record(
        {
            "json_content": {
                "package": "com.example",
                "viewHierarchy": {
                    "viewClass": "root",
                    "bounds": [0, 0, 100, 200],
                },
            }
        },
        graph_id="mobileviews:geometry",
    )

    attached = attach_mobileviews_image(
        graph,
        screenshot_path="images/0.png",
        image_size=(100, 200),
    )

    assert attached.metadata["screenshot_path"] == "images/0.png"
    assert attached.metadata["geometry_mismatch"] is False
    assert attached.metadata["geometry_in_bounds_rate"] == 1.0
    assert split_name_for_group("com.example", seed=17) == split_name_for_group(
        "com.example",
        seed=17,
    )
