import pytest

from omnitransfer.unified_ui import (
    WebViewTransform,
    graph_from_dom_snapshot,
    merge_webview_graph,
)
from omnitransfer.ui_graph import graph_from_record


def test_dom_adapter_maps_roles_attributes_and_hierarchy() -> None:
    graph = graph_from_dom_snapshot(
        {
            "nodeName": "body",
            "backendNodeId": "root",
            "bbox": [0, 0, 400, 800],
            "children": [
                {
                    "nodeName": "button",
                    "backendNodeId": "submit",
                    "innerText": "Submit",
                    "attributes": {"data-testid": "submit-order", "aria-label": "Submit order"},
                    "bbox": {"x": 20, "y": 40, "width": 100, "height": 60},
                }
            ],
        },
        graph_id="dom",
    )

    assert graph.width == 400
    assert graph.height == 800
    assert graph.nodes[0].child_ids == ("submit",)
    assert graph.nodes[1].parent_id == "root"
    assert graph.nodes[1].clickable is True
    assert graph.nodes[1].resource_id == "submit-order"
    assert graph.nodes[1].content_desc == "Submit order"


def test_webview_adapter_attaches_dom_under_native_host() -> None:
    native = graph_from_record(
        {
            "screen_id": "native",
            "width": 600,
            "height": 1200,
            "nodes": [
                {
                    "node_id": "web-host",
                    "class": "android.webkit.WebView",
                    "bounds": [100, 200, 500, 1000],
                }
            ],
        }
    )
    dom = graph_from_dom_snapshot(
        {
            "nodeName": "body",
            "backendNodeId": "root",
            "bbox": [0, 0, 400, 800],
            "children": [
                {
                    "nodeName": "button",
                    "backendNodeId": "buy",
                    "innerText": "Buy",
                    "bbox": [20, 40, 120, 100],
                }
            ],
        },
        graph_id="dom",
    )

    graph = merge_webview_graph(
        native,
        dom,
        host_node_id="web-host",
        transform=WebViewTransform(viewport_width=400, viewport_height=800),
        graph_id="hybrid",
    )

    assert graph.metadata["source_format"] == "webview"
    assert graph.nodes[0].child_ids == ("webview:root",)
    assert graph.nodes[1].parent_id == "web-host"
    assert graph.nodes[2].parent_id == "webview:root"
    assert graph.nodes[2].bbox == (120.0, 240.0, 220.0, 300.0)


def test_webview_adapter_rejects_missing_transform() -> None:
    native = graph_from_record(
        {
            "nodes": [
                {
                    "node_id": "host",
                    "class": "android.webkit.WebView",
                    "bounds": [0, 0, 100, 100],
                }
            ]
        }
    )
    dom = graph_from_dom_snapshot(
        {"nodeName": "body", "backendNodeId": "root", "bbox": [0, 0, 100, 100]},
        graph_id="dom",
    )

    with pytest.raises(ValueError, match="viewport dimensions"):
        merge_webview_graph(
            native,
            dom,
            host_node_id="host",
            transform=WebViewTransform(viewport_width=0, viewport_height=0),
            graph_id="invalid",
        )
