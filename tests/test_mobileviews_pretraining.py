from __future__ import annotations

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair
from omnitransfer.mobileviews_pretraining import (
    mobileviews_page_slices,
    mobileviews_self_supervised_pair,
    mobileviews_state_transition_pairs,
    select_mobileviews_pages,
)


def _graph(graph_id: str, package: str, *, clickable: bool = False) -> dict:
    return {
        "graph_id": graph_id,
        "width": 100,
        "height": 200,
        "metadata": {"package": package, "screenshot_path": f"/{graph_id}.png"},
        "nodes": [
            {
                "node_id": "root", "origin_id": "root", "parent_id": None,
                "text": "", "content_desc": "", "resource_id": "dialog_list",
                "class_name": "RecyclerView", "bbox": [0, 0, 100, 200],
                "clickable": not clickable, "editable": False, "scrollable": True,
                "enabled": True, "depth": 8, "child_ids": ["icon", "label"], "metadata": {},
            },
            {
                "node_id": "icon", "origin_id": "icon", "parent_id": "root",
                "text": "", "content_desc": "", "resource_id": "icon",
                "class_name": "ImageView", "bbox": [0, 0, 20, 20],
                "clickable": clickable, "editable": False, "scrollable": False,
                "enabled": True, "depth": 9, "child_ids": [], "metadata": {},
            },
            {
                "node_id": "label", "origin_id": "label", "parent_id": "root",
                "text": "Settings", "content_desc": "", "resource_id": "label",
                "class_name": "TextView", "bbox": [20, 0, 90, 20],
                "clickable": clickable, "editable": False, "scrollable": False,
                "enabled": True, "depth": 9, "child_ids": [], "metadata": {},
            },
        ],
    }


def test_mobileviews_selection_ignores_clickability_and_keeps_local_slices() -> None:
    left = _graph("left", "pkg", clickable=False)
    right = _graph("right", "pkg", clickable=True)

    assert mobileviews_page_slices(left) == mobileviews_page_slices(right)
    selected_left, manifest_left = select_mobileviews_pages(
        [left], maximum_pages=1, per_package_cap=1
    )
    selected_right, manifest_right = select_mobileviews_pages(
        [right], maximum_pages=1, per_package_cap=1
    )

    assert [row["graph_id"] for row in selected_left] == ["left"]
    assert [row["graph_id"] for row in selected_right] == ["right"]
    assert manifest_left["slice_page_counts"] == manifest_right["slice_page_counts"]
    assert manifest_left["selection_uses_clickability"] is False


def test_mobileviews_pair_maps_every_node_without_cross_platform_pseudo_gold() -> None:
    graph = _graph("page", "pkg")

    pair = validate_ui_correspondence_pair(mobileviews_self_supervised_pair(graph))

    assert pair["label_status"] == "self_supervised"
    assert len(pair["matches"]) == len(graph["nodes"])
    assert pair["provenance"]["cross_platform_gold"] is False
    assert pair["slices"]["textless_with_local_semantics"] >= 1
    assert pair["slices"]["same_parent_sibling_hard_negative_groups"] >= 1


def test_state_transition_uses_identity_only_as_pseudo_label_evidence() -> None:
    source = _graph("state-1", "pkg", clickable=False)
    target = _graph("state-2", "pkg", clickable=True)
    for row, graph in enumerate((source, target), start=1):
        graph["metadata"].update(
            {
                "foreground_activity": "pkg/MainActivity",
                "source_row": row,
            }
        )
    target["nodes"][2]["text"] = "Settings updated"

    pairs, manifest = mobileviews_state_transition_pairs(
        [source, target], maximum_pairs=1, minimum_matches=3
    )
    pair = validate_ui_correspondence_pair(pairs[0])

    assert pair["source"]["page_id"] != pair["target"]["page_id"]
    assert pair["label_status"] == "unreviewed"
    assert len(pair["matches"]) == 3
    assert pair["provenance"]["identity_fields_are_label_only"] is True
    assert "resource_id" in pair["provenance"]["forbidden_model_inputs"]
    assert manifest["selection_uses_clickability"] is False
    assert manifest["model_receives_label_identity_fields"] is False
