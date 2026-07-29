import json
from pathlib import Path

from omnitransfer.mapping_training import load_ui_correspondence_pairs
from omnitransfer.learned_matcher import MatcherConfig


def _node(node_id: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "origin_id": node_id,
        "text": node_id,
        "class_name": "android.widget.Button",
        "bbox": [0, 0, 20, 20],
        "clickable": True,
    }


def _record() -> dict[str, object]:
    return {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair-1",
        "split": "diagnostic",
        "label_status": "unreviewed",
        "source": {
            "page_id": "source-page",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "width": 100,
                "height": 200,
                "nodes": [_node(f"s{i}") for i in range(4)],
            },
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "width": 100,
                "height": 200,
                "nodes": [_node(f"t{i}") for i in range(5)],
            },
        },
        "matches": [
            {
                "source_node_id": "s0",
                "target_node_ids": ["t0"],
                "label": "correspondence",
            },
            {
                "source_node_id": "s1",
                "target_node_ids": ["t1"],
                "label": "correspondence",
            },
            {
                "source_node_id": "s2",
                "target_node_ids": ["t2", "t3"],
                "label": "correspondence",
            },
            {
                "source_node_id": "s3",
                "target_node_ids": ["t1"],
                "label": "correspondence",
            },
        ],
        "partition_keys": ["mobileviews:app:test"],
        "provenance": {"dataset": "MobileViews_Apps_CompleteTraces"},
        "slices": {"platform": "android"},
    }


def test_unreviewed_pairs_require_explicit_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    pairs, graphs, manifest = load_ui_correspondence_pairs([path])

    assert pairs == []
    assert graphs == []
    assert manifest["skipped"] == {"unreviewed_requires_opt_in": 1}


def test_adapter_filters_diagnostic_rows_by_reserved_split(tmp_path: Path) -> None:
    dev = _record()
    dev["pair_id"] = "pair-dev"
    dev["provenance"]["reserved_split"] = "dev"
    test = _record()
    test["pair_id"] = "pair-test"
    test["source"]["page_id"] = "source-test"
    test["target"]["page_id"] = "target-test"
    test["provenance"]["reserved_split"] = "test"
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(
        "\n".join((json.dumps(dev), json.dumps(test))) + "\n",
        encoding="utf-8",
    )

    pairs, _, manifest = load_ui_correspondence_pairs(
        [path],
        allow_unreviewed_pseudo=True,
        allowed_reserved_splits=frozenset({"test"}),
    )

    assert len(pairs) == 1
    assert pairs[0].graph_a.graph_id == "source-test"
    assert manifest["allowed_reserved_splits"] == ["test"]
    assert manifest["skipped"] == {"reserved_split:dev": 1}


def test_adapter_retains_set_valued_and_shared_target_labels(tmp_path: Path) -> None:
    record = _record()
    record["matches"].append(
        {"source_node_id": "s4", "target_node_ids": ["t4"], "label": "correspondence"}
    )
    record["source"]["graph"]["nodes"].append(_node("s4"))
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, graphs, manifest = load_ui_correspondence_pairs(
        [path],
        allow_unreviewed_pseudo=True,
        minimum_correspondences=2,
    )

    assert len(pairs) == 1
    assert len(graphs) == 2
    assert pairs[0].origin_ids == (
        "s0->t0",
        "s1->t1",
        "s2->t2",
        "s2->t3",
        "s3->t1",
        "s4->t4",
    )
    assert pairs[0].positive_targets_a_to_b[2] == (2, 3)
    assert pairs[0].positive_targets_b_to_a[1] == (1, 3)
    assert manifest["accepted_correspondences"] == 6
    assert manifest["set_valued_match_rows_retained"] == 1
    assert manifest["shared_target_rows_retained"] == 2


def test_adapter_supervises_only_actionable_to_actionable_pairs(tmp_path: Path) -> None:
    record = _record()
    record["matches"] = record["matches"][:2]
    record["source"]["graph"]["nodes"][1]["clickable"] = False
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, _, manifest = load_ui_correspondence_pairs(
        [path],
        allow_unreviewed_pseudo=True,
        minimum_correspondences=1,
    )

    assert len(pairs) == 1
    assert pairs[0].origin_ids == ("s0->t0",)
    assert manifest["non_actionable_correspondences_filtered"] == 1
    assert {node.node_id for node in pairs[0].graph_a.nodes} == {
        "s0",
        "s1",
        "s2",
        "s3",
    }


def test_adapter_resolves_materialized_review_screenshot(tmp_path: Path) -> None:
    record = _record()
    record["matches"] = record["matches"][:2]
    record["source"]["screenshot_path"] = "/missing/screen.png"
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    materialized = screenshots / "digest_screen.png"
    materialized.write_bytes(b"not-decoded-during-adaptation")

    pairs, _, _ = load_ui_correspondence_pairs(
        [path],
        allow_unreviewed_pseudo=True,
        screenshot_root=screenshots,
    )

    assert pairs[0].graph_a.metadata["screenshot_path"] == str(materialized)


def test_adapter_bounds_local_context_around_all_matches(tmp_path: Path) -> None:
    record = _record()
    record["matches"] = record["matches"][:2]
    source_context_node = _node("s4")
    source_context_node["clickable"] = False
    record["source"]["graph"]["nodes"].append(source_context_node)
    for index in range(5, 20):
        source_node = _node(f"s{index}")
        target_node = _node(f"t{index}")
        source_node["clickable"] = False
        target_node["clickable"] = False
        record["source"]["graph"]["nodes"].append(source_node)
        record["target"]["graph"]["nodes"].append(target_node)
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, _, manifest = load_ui_correspondence_pairs(
        [path],
        matcher_config=MatcherConfig(source_context_nodes=4, target_context_nodes=5),
        allow_unreviewed_pseudo=True,
    )

    assert len(pairs[0].graph_a.nodes) == 4
    assert len(pairs[0].graph_b.nodes) == 5
    assert {"s0", "s1"} <= {node.node_id for node in pairs[0].graph_a.nodes}
    assert {"t0", "t1"} <= {node.node_id for node in pairs[0].graph_b.nodes}
    assert manifest["training_boundary"]["local_context_limits"] == {
        "source": 4,
        "target": 5,
    }


def test_adapter_keeps_every_actionable_hard_negative_beyond_context_limit(
    tmp_path: Path,
) -> None:
    record = _record()
    record["matches"] = record["matches"][:2]
    path = tmp_path / "diagnostic.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, _, _ = load_ui_correspondence_pairs(
        [path],
        matcher_config=MatcherConfig(source_context_nodes=2, target_context_nodes=2),
        allow_unreviewed_pseudo=True,
    )

    assert len(pairs[0].graph_a.nodes) == 4
    assert len(pairs[0].graph_b.nodes) == 5
    assert all(node.clickable for node in pairs[0].graph_a.nodes)
    assert all(node.clickable for node in pairs[0].graph_b.nodes)


def test_adapter_does_not_lift_descendant_labels_to_actionable_ancestors(
    tmp_path: Path,
) -> None:
    record = _record()
    record["source"]["graph"]["nodes"] = [
        {
            "node_id": "source-row",
            "origin_id": "source-row",
            "class_name": "android.widget.LinearLayout",
            "bbox": [0, 40, 100, 100],
            "clickable": True,
            "child_ids": ["source-title", "source-summary"],
        },
        {
            "node_id": "source-title",
            "origin_id": "source-title",
            "parent_id": "source-row",
            "text": "Sound",
            "resource_id": "android:id/title",
            "class_name": "android.widget.TextView",
            "bbox": [20, 50, 60, 70],
        },
        {
            "node_id": "source-summary",
            "origin_id": "source-summary",
            "parent_id": "source-row",
            "text": "Volume, vibration, Do Not Disturb",
            "resource_id": "android:id/summary",
            "class_name": "android.widget.TextView",
            "bbox": [20, 70, 90, 90],
        },
    ]
    record["target"]["graph"]["nodes"] = [
        {
            "node_id": "target-sound-row",
            "origin_id": "target-sound-row",
            "class_name": "android.widget.LinearLayout",
            "bbox": [0, 20, 100, 70],
            "clickable": True,
            "child_ids": ["target-sound-title", "target-sound-summary"],
        },
        {
            "node_id": "target-sound-title",
            "origin_id": "target-sound-title",
            "parent_id": "target-sound-row",
            "text": "Sound",
            "resource_id": "android:id/title",
            "class_name": "android.widget.TextView",
            "bbox": [20, 30, 60, 45],
        },
        {
            "node_id": "target-sound-summary",
            "origin_id": "target-sound-summary",
            "parent_id": "target-sound-row",
            "text": "Volume, vibration, Do Not Disturb",
            "resource_id": "android:id/summary",
            "class_name": "android.widget.TextView",
            "bbox": [20, 45, 90, 60],
        },
        {
            "node_id": "target-privacy-row",
            "origin_id": "target-privacy-row",
            "class_name": "android.widget.LinearLayout",
            "bbox": [0, 80, 100, 130],
            "clickable": True,
            "child_ids": ["target-privacy-title"],
        },
        {
            "node_id": "target-privacy-title",
            "origin_id": "target-privacy-title",
            "parent_id": "target-privacy-row",
            "text": "Privacy",
            "resource_id": "android:id/title",
            "class_name": "android.widget.TextView",
            "bbox": [20, 90, 65, 105],
        },
    ]
    record["matches"] = [
        {
            "source_node_id": "source-title",
            "target_node_ids": ["target-sound-title"],
            "label": "correspondence",
        },
        {
            "source_node_id": "source-summary",
            "target_node_ids": ["target-sound-summary"],
            "label": "correspondence",
        },
    ]
    path = tmp_path / "anonymous-actionable-row.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs, _, manifest = load_ui_correspondence_pairs(
        [path],
        allow_unreviewed_pseudo=True,
        minimum_correspondences=1,
    )

    assert pairs == []
    assert manifest["explicit_correspondences"] == 2
    assert manifest["non_actionable_correspondences_filtered"] == 2
    assert manifest["accepted_correspondences"] == 0
