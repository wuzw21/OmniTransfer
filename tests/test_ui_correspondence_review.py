import json
from pathlib import Path

from PIL import Image

from omnitransfer.mapping_pair_review import build_mapping_pair_review


def test_ui_correspondence_review_copies_images_and_embeds_matches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (100, 200), "white").save(source)
    Image.new("RGB", (200, 100), "black").save(target)
    record = {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair-1",
        "split": "diagnostic",
        "label_status": "unreviewed",
        "source": {
            "page_id": "source-page",
            "platform": "android",
            "screenshot_path": str(source),
            "graph": {
                "graph_id": "source-page",
                "width": 100,
                "height": 200,
                "nodes": [{"node_id": "s1", "origin_id": "origin-1", "bbox": [1, 2, 30, 40]}],
            },
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": str(target),
            "graph": {
                "graph_id": "target-page",
                "width": 200,
                "height": 100,
                "nodes": [{"node_id": "t1", "origin_id": "origin-1", "bbox": [4, 5, 60, 70]}],
            },
        },
        "matches": [{"source_node_id": "s1", "target_node_ids": ["t1"], "label": "correspondence"}],
        "partition_keys": ["test:pair"],
        "provenance": {"dataset": "test-dataset"},
        "method_tags": [
            {"id": "ours_vs_selector_disagree", "label": "我们的方案 ≠ selector", "count": 1}
        ],
        "slices": {"track": "test"},
    }
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    manifest = build_mapping_pair_review([input_path], tmp_path / "review")

    assert manifest["pairs"] == 1
    assert manifest["matches"] == 1
    assert manifest["screenshots"] == 2
    html = (tmp_path / "review" / "review.html").read_text(encoding="utf-8")
    assert "pair-1" in html
    assert "source-page" in html
    assert "target-page" in html
    assert "target_points" in html
    assert "page_pixels" in html
    assert "可标多个" in html
    assert "source_query_point" in html
    assert "source_queries" in html
    assert "matcherMappedPoint" in html
    assert "重置 Source" in html
    assert "review.source_query_point && !review.source_node_id" in html
    assert "未命中可匹配 Source 节点" in html
    assert "Target 吸附" in html
    assert "node.node_id === task.source?.node?.node_id" in html
    assert "我们的方案 ≠ selector" in html
    assert "只看 ours ≠ selector" in html
    filtered_manifest = build_mapping_pair_review(
        [input_path], tmp_path / "method-review", method_tag="ours_vs_selector_disagree"
    )
    assert filtered_manifest["pairs"] == 1


def test_review_groups_multiple_mappings_by_page_pair(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (100, 100), "white").save(source)
    Image.new("RGB", (100, 100), "black").save(target)
    record = {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair-many",
        "split": "diagnostic",
        "label_status": "unreviewed",
        "source": {
            "page_id": "source-page",
            "platform": "android",
            "screenshot_path": str(source),
            "graph": {
                "graph_id": "source-page",
                "width": 100,
                "height": 100,
                "nodes": [
                    {"node_id": "s1", "origin_id": "os1", "bbox": [1, 2, 20, 20]},
                    {"node_id": "s2", "origin_id": "os2", "bbox": [30, 2, 50, 20]},
                ],
            },
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": str(target),
            "graph": {
                "graph_id": "target-page",
                "width": 100,
                "height": 100,
                "nodes": [
                    {"node_id": "t1", "origin_id": "ot1", "bbox": [1, 30, 20, 50]},
                    {"node_id": "t2", "origin_id": "ot2", "bbox": [30, 30, 50, 50]},
                ],
            },
        },
        "matches": [
            {"source_node_id": "s1", "target_node_ids": ["t1"], "label": "correspondence"},
            {"source_node_id": "s2", "target_node_ids": ["t2"], "label": "correspondence"},
        ],
        "partition_keys": ["test:pair-many"],
        "provenance": {"dataset": "test-dataset"},
        "slices": {"track": "test"},
    }
    input_path = tmp_path / "pairs.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    build_mapping_pair_review([input_path], tmp_path / "review")

    payload = json.loads(
        (tmp_path / "review" / "review.html.payload.json").read_text(encoding="utf-8")
    )
    assert payload["summary"]["review_ui"]["multi_mapping"] is True
    assert payload["summary"]["task_count"] == 1
    assert [row["mapping_id"] for row in payload["pairs"][0]["mappings"]] == ["S1-M1", "S2-M2"]
    assert payload["pairs"][0]["source"]["candidates"]
    assert payload["pairs"][0]["target"]["candidates"]
