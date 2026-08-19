from __future__ import annotations

from copy import deepcopy

from omnitransfer.icon_hard_set import build_icon_hard_records


def _node(
    node_id: str,
    *,
    parent_id: str | None,
    class_name: str,
    bbox: list[float],
    content_desc: str = "",
) -> dict:
    return {
        "node_id": node_id,
        "origin_id": node_id,
        "parent_id": parent_id,
        "text": "",
        "content_desc": content_desc,
        "resource_id": "",
        "class_name": class_name,
        "bbox": bbox,
        "clickable": "Image" in class_name,
        "editable": False,
        "scrollable": False,
        "enabled": True,
        "depth": node_id.count("."),
        "child_ids": [],
        "metadata": {},
    }


def _record(pair_id: str = "pair") -> dict:
    source_nodes = [
        _node("root", parent_id=None, class_name="Root", bbox=[0, 0, 100, 100]),
        _node("web", parent_id="root", class_name="WebView", bbox=[0, 0, 100, 100]),
        _node("row", parent_id="web", class_name="LinearLayout", bbox=[0, 0, 100, 30]),
    ]
    target_nodes = [
        _node("root", parent_id=None, class_name="Root", bbox=[0, 0, 100, 100]),
        _node("row", parent_id="root", class_name="LinearLayout", bbox=[0, 60, 100, 90]),
    ]
    matches = []
    for index, left in enumerate((10, 30, 50)):
        source_id = f"source-{index}"
        target_id = f"target-{index}"
        source_nodes.append(
            _node(
                source_id,
                parent_id="row",
                class_name="ImageView",
                bbox=[left, 10, left + 4, 14],
            )
        )
        target_nodes.append(
            _node(
                target_id,
                parent_id="row",
                class_name="ImageButton",
                bbox=[left + 5, 70, left + 9, 74],
                content_desc=f"Action {index}",
            )
        )
        matches.append(
            {
                "source_node_id": source_id,
                "target_node_ids": [target_id],
                "label": "correspondence",
            }
        )
    return {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": pair_id,
        "split": "diagnostic",
        "label_status": "self_supervised",
        "source": {
            "page_id": f"{pair_id}-source",
            "platform": "ios",
            "screenshot_path": "/tmp/source.png",
            "graph": {"graph_id": "source", "width": 100, "height": 100, "nodes": source_nodes},
        },
        "target": {
            "page_id": f"{pair_id}-target",
            "platform": "android",
            "screenshot_path": "/tmp/target.png",
            "graph": {"graph_id": "target", "width": 100, "height": 100, "nodes": target_nodes},
        },
        "matches": matches,
        "partition_keys": [f"component:{pair_id}"],
        "provenance": {"dataset": "synthetic"},
        "slices": {"app": "Synthetic"},
    }


def test_hard_set_records_observable_icon_reasons() -> None:
    selected, manifest = build_icon_hard_records([_record()])

    assert manifest["selected_pairs"] == 1
    assert manifest["selected_icons"] == 3
    assert manifest["selection_boundary"]["model_scores_used"] is False
    hard_set = selected[0]["provenance"]["hard_set"]
    assert len(hard_set["source_items"]) == 3
    reasons = set(hard_set["source_items"][0]["reasons"])
    assert {
        "textless_source_icon",
        "tiny_icon",
        "webview_source",
        "repeated_sibling_slot",
        "same_row_or_column_icons",
        "no_resource_id",
        "cross_platform_semantic_asymmetry",
        "large_layout_displacement",
    } <= reasons


def test_hard_set_limit_is_deterministic() -> None:
    easier = _record("easier")
    source_nodes = easier["source"]["graph"]["nodes"]
    source_nodes[:] = [node for node in source_nodes if node["node_id"] != "web"]
    next(node for node in source_nodes if node["node_id"] == "row")["parent_id"] = "root"
    harder = deepcopy(_record("harder"))

    selected, manifest = build_icon_hard_records([easier, harder], limit=1)

    assert manifest["selected_pairs"] == 1
    assert selected[0]["pair_id"] == "harder"
