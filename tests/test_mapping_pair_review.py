import json
from pathlib import Path

from PIL import Image

from omnitransfer.mapping_pair_review import build_mapping_pair_review


def test_mapping_pair_review_copies_images_and_embeds_matches(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (100, 200), "white").save(source)
    Image.new("RGB", (200, 100), "black").save(target)
    record = {
        "schema_version": "omnitransfer.mapping_page_pair.v1",
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
