from __future__ import annotations

from copy import deepcopy

from omnitransfer.local_context_algorithm import LocalContextAlgorithm


def _node(node_id, *, text="", parent=None, children=None, bbox=None, class_name="View", clickable=False):
    return {
        "node_id": node_id,
        "text": text,
        "content_desc": "",
        "parent_id": parent,
        "child_ids": children or [],
        "bbox": bbox,
        "class_name": class_name,
        "clickable": clickable,
    }


def test_one_sided_semantics_use_local_neighbor() -> None:
    source = [
        _node("sr", children=["si", "sl"]),
        _node("si", parent="sr", bbox=[0.0, 0.0, 0.2, 0.2], class_name="ImageView"),
        _node("sl", text="Search", parent="sr", bbox=[0.2, 0.0, 0.7, 0.2], class_name="TextView"),
    ]
    target = [
        _node("tr", children=["wrong", "gold"]),
        _node("wrong", parent="tr", bbox=[0.0, 0.0, 0.4, 0.2], class_name="View"),
        _node("gold", text="Search", parent="tr", bbox=[0.7, 0.0, 0.9, 0.2], class_name="ImageView"),
    ]

    ranked = LocalContextAlgorithm().rank(source, target, "si")

    assert ranked[0]["node_id"] == "gold"
    assert ranked[0]["evidence"]["one_sided_bridge"] > 0


def test_textless_icon_uses_neighbor_semantics_and_sibling_slot() -> None:
    source = [
        _node("sr", children=["search", "filter", "plus"]),
        _node("search", text="Search", parent="sr", bbox=[0.0, 0.0, 0.5, 0.2]),
        _node("filter", parent="sr", bbox=[0.55, 0.0, 0.7, 0.2], class_name="ImageView"),
        _node("plus", parent="sr", bbox=[0.8, 0.0, 0.95, 0.2], class_name="ImageView"),
    ]
    target = [
        _node("tr", children=["label", "gold", "wrong"]),
        _node("label", text="Search", parent="tr", bbox=[0.0, 0.0, 0.5, 0.2]),
        _node("gold", parent="tr", bbox=[0.55, 0.0, 0.7, 0.2], class_name="ImageView"),
        _node("wrong", parent="tr", bbox=[0.8, 0.0, 0.95, 0.2], class_name="ImageView"),
    ]

    ranked = LocalContextAlgorithm().rank(source, target, "filter")

    assert ranked[0]["node_id"] == "gold"


def test_clickability_does_not_change_ranking() -> None:
    source = [_node("s", text="Search", clickable=False)]
    target = [_node("a", text="Search", clickable=False), _node("b", text="Other", clickable=True)]
    algorithm = LocalContextAlgorithm()
    before = [row["node_id"] for row in algorithm.rank(source, target, "s")]
    changed = deepcopy(target)
    changed[0]["clickable"] = True
    changed[1]["clickable"] = False
    after = [row["node_id"] for row in algorithm.rank(source, changed, "s")]

    assert before == after
