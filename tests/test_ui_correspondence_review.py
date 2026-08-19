import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest
from PIL import Image

import omnitransfer.mapping_pair_review as mapping_pair_review
from omnitransfer.mapping_pair_review import build_mapping_pair_review
from omnitransfer.learned_matcher import MatcherConfig, build_geometric_v9_matcher
from omnitransfer.numpy_v9_matcher import save_numpy_unified_association_checkpoint


def _icon_review_record(source: Path, target: Path) -> dict:
    def node(node_id: str, x: int, y: int, class_name: str) -> dict:
        return {
            "node_id": node_id,
            "origin_id": f"origin-{node_id}",
            "class_name": class_name,
            "bbox": [x, y, x + 20, y + 20],
            "clickable": True,
        }

    return {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "icon-pair-1",
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
                    node("s1", 10, 10, "android.widget.ImageView"),
                    node("s2", 50, 10, "android.widget.ImageView"),
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
                    node("t1", 10, 50, "android.widget.ImageView"),
                    node("t2", 50, 50, "android.widget.ImageView"),
                ],
            },
        },
        "matches": [
            {"source_node_id": "s1", "target_node_ids": [], "label": "no_correspondence"}
        ],
        "partition_keys": ["test:icon-pair"],
        "provenance": {"dataset": "test-icons"},
        "slices": {"track": "test"},
    }


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
            "display_width": 300,
            "display_height": 600,
            "graph": {
                "graph_id": "source-page",
                "width": 100,
                "height": 200,
                "nodes": [{"node_id": "s1", "origin_id": "origin-1", "text": "Line\u2028Break", "bbox": [1, 2, 30, 40], "metadata": {"visual_bbox": [3, 4, 90, 100]}}],
            },
            "xml": "<hierarchy bounds=\"[0,0][100,200]\" />",
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": str(target),
            "display_width": 600,
            "display_height": 300,
            "graph": {
                "graph_id": "target-page",
                "width": 200,
                "height": 100,
                "nodes": [{"node_id": "t1", "origin_id": "origin-1", "bbox": [4, 5, 60, 70]}],
            },
            "xml": "<hierarchy bounds=\"[0,0][200,100]\" />",
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
    input_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "review.html.unified_manual.index.json").write_text(
        json.dumps({"schema_version": "test.index.v1", "rows": []}), encoding="utf-8"
    )
    manifest = build_mapping_pair_review([input_path], review_dir)

    assert manifest["pairs"] == 1
    assert manifest["matches"] == 1
    assert manifest["screenshots"] == 2
    html = (tmp_path / "review" / "review.html").read_text(encoding="utf-8")
    assert "pair-1" in html
    assert "source-page" in html
    assert "Line\u2028Break" in html
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
    assert "remainingManualMode" in html
    assert "rank_action_candidates" in html
    assert "source_offset" in html
    assert "sourceOffset(sourcePoint, snappedNode)" in html
    assert "live_ours_status" in html
    assert "bindStageFallback" in html
    assert "stagePoint" in html
    assert "migrateLegacyLiveState" in html
    assert "toVisualPoint" in html
    assert "point?.coordinate_space === 'screenshot_pixels'" in html
    assert "canonicalOursNode" in html
    assert "screenshot_pixels" in html
    assert "实时计算的 Target 点" in html
    assert "静态文件模式不能实时映射" in html
    assert "test.index.v1" in html
    payload = json.loads((tmp_path / "review" / "review.html.payload.json").read_text(encoding="utf-8"))
    assert payload["pairs"][0]["source"]["xml"].startswith("<hierarchy")
    assert payload["pairs"][0]["target"]["xml"].startswith("<hierarchy")
    assert payload["pairs"][0]["source"]["width"] == 100
    assert payload["pairs"][0]["source"]["height"] == 200
    assert payload["pairs"][0]["source"]["display_width"] == 300
    assert payload["pairs"][0]["source"]["display_height"] == 600
    assert payload["pairs"][0]["source"]["candidates"][0]["visual_bbox"] == [
        3.0,
        4.0,
        90.0,
        100.0,
    ]
    assert "实时计算中" in html
    assert "node.node_id === task.source?.node?.node_id" in html
    assert "我们的方案 ≠ selector" in html
    assert "只看 ours ≠ selector" in html
    filtered_manifest = build_mapping_pair_review(
        [input_path], tmp_path / "method-review", method_tag="ours_vs_selector_disagree"
    )
    assert filtered_manifest["pairs"] == 1


def test_icon_review_requires_a_matcher_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (100, 100), "white").save(source)
    Image.new("RGB", (100, 100), "black").save(target)
    input_path = tmp_path / "icons.jsonl"
    input_path.write_text(
        json.dumps(_icon_review_record(source, target)) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="NumPy matcher checkpoint"):
        build_mapping_pair_review([input_path], tmp_path / "review", icon_only=True)


def test_review_server_handles_live_mapping_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "review.html").write_text("review", encoding="utf-8")
    captured: dict = {}

    def fake_rank_action_candidates(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {
            "schema_version": "omnitransfer.candidate-ranking.v1",
            "status": "scored",
            "candidates": [{"candidate_id": "target-node", "score": 0.99}],
        }

    monkeypatch.setattr(
        mapping_pair_review, "rank_action_candidates", fake_rank_action_candidates
    )
    server = mapping_pair_review.create_review_server(
        review_dir, host="127.0.0.1", port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            "source_xml": "<hierarchy />",
            "target_xml": "<hierarchy />",
            "source_element_id": "source-node",
            "source_point": {"x": 12.5, "y": 20.0},
            "source_offset": {"x": 0.25, "y": 0.75},
            "source_screenshot_path": "screenshots/source.png",
            "target_screenshot_path": "screenshots/target.png",
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/rank_action_candidates",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
        relative_payload = {
            **payload,
            "source_xml": '<node coordinate-space="xml-relative-0-1" />',
            "source_size": {"width": 100, "height": 200},
            "source_point": {"x": 50, "y": 25},
        }
        relative_request = Request(
            f"http://127.0.0.1:{server.server_port}/api/rank_action_candidates",
            data=json.dumps(relative_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(relative_request, timeout=2) as response:
            relative_result = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["status"] == "scored"
    assert relative_result["status"] == "scored"
    assert captured["source_element_id"] == "source-node"
    assert captured["source_point"] == (0.5, 0.125)
    assert captured["source_offset"] == (0.25, 0.75)
    assert captured["source_screenshot_path"] == str(
        review_dir / "screenshots/source.png"
    )
    assert captured["target_screenshot_path"] == str(
        review_dir / "screenshots/target.png"
    )


def test_icon_review_creates_one_task_per_icon_with_predictions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (100, 100), "white").save(source)
    Image.new("RGB", (100, 100), "black").save(target)
    input_path = tmp_path / "icons.jsonl"
    input_path.write_text(
        json.dumps(_icon_review_record(source, target)) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "unified-association.npz"
    model = build_geometric_v9_matcher(
        MatcherConfig(hidden_dim=32, association_dim=24, dropout=0.0)
    ).eval()
    save_numpy_unified_association_checkpoint(
        checkpoint,
        model.state_dict(),
        config=MatcherConfig(hidden_dim=32, association_dim=24, dropout=0.0),
    )

    manifest = build_mapping_pair_review(
        [input_path], tmp_path / "review", icon_only=True, checkpoint=checkpoint
    )

    assert manifest["icon_only"] is True
    assert manifest["pairs"] == 2
    payload = json.loads(
        (tmp_path / "review" / "review.html.payload.json").read_text(encoding="utf-8")
    )
    assert payload["summary"]["review_ui"]["protocol"] == "icon_visual_review"
    assert payload["summary"]["review_ui"]["icon_candidate_policy"] == {
        "text_and_content_desc_empty": True,
        "preferred_bbox": "visual_bbox_then_bbox",
        "max_area_ratio": 0.05,
        "max_dimension_ratio": 0.40,
        "max_aspect_ratio": 6.0,
    }
    assert payload["summary"]["review_ui"]["keyboard_shortcuts"] == {
        "s": "save_draft",
        "1": "correct_correspondence",
        "2": "wrong_correspondence",
    }
    assert len(payload["pairs"]) == 2
    for task in payload["pairs"]:
        assert len(task["source_queries"]) == 1
        query = next(iter(task["source_queries"].values()))
        assert query["source_node"]["node_id"].startswith("s")
        assert "matcher_prediction" in query

    html = (tmp_path / "review" / "review.html").read_text(encoding="utf-8")
    assert "omnitransfer_icon_annotations.jsonl" in html
    assert "omnitransfer_icon_gold_pairs.jsonl" in html
    assert "先点击 Source，再查看自动 P" in html


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
