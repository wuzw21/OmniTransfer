from __future__ import annotations

from omnitransfer.mapping_data_cleaning import clean_ui_correspondence_records


def _node(
    node_id: str,
    *,
    text: str = "",
    parent_id: str | None = None,
    child_ids: list[str] | None = None,
    bbox: list[float] | None = None,
    class_name: str = "android.view.View",
) -> dict:
    return {
        "node_id": node_id,
        "origin_id": node_id,
        "parent_id": parent_id,
        "child_ids": child_ids or [],
        "text": text,
        "content_desc": "",
        "class_name": class_name,
        "bbox": bbox or [0.0, 0.0, 1.0, 1.0],
        "clickable": False,
        "enabled": True,
    }


def _record(source_nodes: list[dict], target_nodes: list[dict], match: dict) -> dict:
    return {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair",
        "split": "train",
        "label_status": "gold",
        "source": {
            "page_id": "app/source",
            "platform": "ios",
            "screenshot_path": "",
            "graph": {"width": 1.0, "height": 1.0, "nodes": source_nodes},
        },
        "target": {
            "page_id": "app/target",
            "platform": "android",
            "screenshot_path": "",
            "graph": {"width": 1.0, "height": 1.0, "nodes": target_nodes},
        },
        "matches": [match],
        "partition_keys": ["app:test"],
        "provenance": {"dataset": "test"},
        "slices": {"app": "test"},
    }


def test_cleaner_preserves_endpoints_and_ignores_clickability() -> None:
    source = [
        _node("sp", child_ids=["sc"]),
        _node("sc", text="Search", parent_id="sp", bbox=[0.1, 0.1, 0.9, 0.9]),
    ]
    target = [
        _node("tp", child_ids=["tc"]),
        _node("tc", text="Search", parent_id="tp", bbox=[0.1, 0.1, 0.9, 0.9]),
    ]
    record = _record(
        source,
        target,
        {"source_node_id": "sp", "target_node_ids": ["tp"], "label": "correspondence"},
    )

    result = clean_ui_correspondence_records([record])
    cleaned_match = result.cleaned[0]["matches"][0]

    assert cleaned_match["source_node_id"] == "sp"
    assert cleaned_match["target_node_ids"] == ["tp"]
    assert "source_equivalent_node_ids" not in cleaned_match
    assert "parent_child_semantic_unit" not in cleaned_match["quality_slices"]
    assert result.manifest["clickability_used_for_cleaning"] is False
    assert result.manifest["correspondence_endpoints_changed"] is False
    assert result.manifest["parent_child_equivalence"] == "disabled"


def test_cleaner_marks_one_sided_and_textless_local_context() -> None:
    source = [
        _node("root", child_ids=["icon", "label"]),
        _node("icon", parent_id="root", bbox=[0.1, 0.1, 0.2, 0.2]),
        _node("label", text="Search", parent_id="root", bbox=[0.2, 0.1, 0.5, 0.2]),
    ]
    target = [
        _node("troot", child_ids=["ticon", "tlabel"]),
        _node("ticon", parent_id="troot", bbox=[0.1, 0.1, 0.2, 0.2]),
        _node("tlabel", text="Search", parent_id="troot", bbox=[0.2, 0.1, 0.5, 0.2]),
    ]
    textless = _record(
        source,
        target,
        {"source_node_id": "icon", "target_node_ids": ["ticon"], "label": "correspondence"},
    )
    one_sided_target = [
        _node("troot", child_ids=["ticon"]),
        _node("ticon", text="Search", parent_id="troot", bbox=[0.1, 0.1, 0.2, 0.2]),
    ]
    one_sided = _record(
        source,
        one_sided_target,
        {"source_node_id": "icon", "target_node_ids": ["ticon"], "label": "correspondence"},
    )
    one_sided["pair_id"] = "one-sided"
    one_sided["source"]["page_id"] = "app/source-2"
    one_sided["target"]["page_id"] = "app/target-2"

    result = clean_ui_correspondence_records([textless, one_sided])
    slices = [set(row["matches"][0]["quality_slices"]) for row in result.cleaned]

    assert "textless_with_local_semantics" in slices[0]
    assert "one_sided_semantics" in slices[1]


def test_cleaner_quarantines_exact_semantic_conflict_outside_gold_family() -> None:
    source = [_node("s", text="Create Account")]
    target = [
        _node("gold", text="Have an account? Sign in"),
        _node("other", text="Create Account", bbox=[0.0, 0.5, 1.0, 1.0]),
    ]
    record = _record(
        source,
        target,
        {"source_node_id": "s", "target_node_ids": ["gold"], "label": "correspondence"},
    )

    result = clean_ui_correspondence_records([record])

    assert result.cleaned == ()
    assert len(result.quarantine) == 1
    assert result.quarantine[0]["provenance"]["cleaning_reason"] == (
        "exact_semantic_conflict_outside_gold_family"
    )
    assert result.quarantine[0]["provenance"]["cleaning_conflicting_target_node_ids"] == ["other"]


def test_cleaner_marks_only_comparable_siblings_as_hard_negatives() -> None:
    source = [_node("s", text="First")]
    target = [
        _node("root", child_ids=["gold", "hard", "decorative"]),
        _node("gold", text="First", parent_id="root", bbox=[0.0, 0.0, 0.4, 0.1], class_name="TextView"),
        _node("hard", text="Second", parent_id="root", bbox=[0.0, 0.1, 0.4, 0.2], class_name="TextView"),
        _node("decorative", parent_id="root", bbox=[0.9, 0.0, 0.91, 0.01], class_name="ImageView"),
    ]
    record = _record(
        source,
        target,
        {"source_node_id": "s", "target_node_ids": ["gold"], "label": "correspondence"},
    )

    result = clean_ui_correspondence_records([record])

    assert "same_parent_sibling_hard_negative" in result.cleaned[0]["matches"][0]["quality_slices"]
