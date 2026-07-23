import json
from pathlib import Path
import subprocess
import sys

from omnitransfer.importers import query_from_dict, write_queries
from omnitransfer.widget_mapping_review import build_widget_mapping_pair_review


def _query(query_id: str, source_bbox: list[int], gold_id: str):
    return query_from_dict(
        {
            "query_id": query_id,
            "source": {
                "text": query_id,
                "class_name": "XCUIElementTypeButton",
                "bounds": source_bbox,
                "x": (source_bbox[0] + source_bbox[2]) / 2,
                "y": (source_bbox[1] + source_bbox[3]) / 2,
                "metadata": {"platform": "ios", "node_match": "node_not_found"},
            },
            "target_candidates": [
                {
                    "candidate_id": "target-a",
                    "text": "First",
                    "class_name": "android.widget.Button",
                    "bbox": [10, 100, 50, 140],
                    "node_index": 2,
                },
                {
                    "candidate_id": "target-a-alias",
                    "text": "First",
                    "class_name": "android.widget.TextView",
                    "bbox": [10, 100, 50, 140],
                },
                {
                    "candidate_id": "target-b",
                    "text": "Second",
                    "class_name": "android.widget.Button",
                    "bbox": [80, 100, 130, 140],
                    "node_index": 3,
                },
            ],
            "gold_candidate_id": gold_id,
            "metadata": {
                "app": "Example",
                "category": "Tools",
                "source_screen": "Tools/Example/iOS/0",
                "target_screen": "Tools/Example/Android/0",
                "source_screenshot_path": "assets/source.png",
                "target_screenshot_path": "assets/target.png",
                "source_xml_path": "assets/source.xml",
                "target_xml_path": "assets/target.xml",
                "source_coordinate_width": 200,
                "source_coordinate_height": 400,
                "target_coordinate_width": 180,
                "target_coordinate_height": 400,
            },
        }
    )


def test_pair_review_groups_multiple_queries_on_the_same_screen_pair(tmp_path: Path) -> None:
    input_path = tmp_path / "queries.jsonl"
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "source.png").write_bytes(b"source")
    (assets / "target.png").write_bytes(b"target")
    queries = [
        _query("source-a", [10, 20, 40, 50], "target-a"),
        _query("source-b", [80, 20, 120, 50], "target-b"),
    ]

    rows, manifest = build_widget_mapping_pair_review(
        queries,
        input_path=input_path,
        output_dir=tmp_path / "review",
    )

    assert manifest["queries"] == 2
    assert manifest["screen_pairs"] == 1
    assert manifest["correspondences"] == 2
    assert rows[0]["quality"]["target_alias_groups"] == 1
    assert len(rows[0]["correspondences"]) == 2
    assert rows[0]["correspondences"][0]["gold_bboxes"] == [[10.0, 100.0, 50.0, 140.0]]
    assert rows[0]["source"]["image_exists"] is True
    assert rows[0]["target"]["image_exists"] is True


def test_pair_review_script_writes_multi_mapping_html(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "source.png").write_bytes(b"source")
    (assets / "target.png").write_bytes(b"target")
    input_path = tmp_path / "queries.jsonl"
    write_queries(
        [
            _query("source-a", [10, 20, 40, 50], "target-a"),
            _query("source-b", [80, 20, 120, 50], "target-b"),
        ],
        input_path,
    )
    output = tmp_path / "review"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_widget_mapping_pair_review.py",
            "--input",
            str(input_path),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    reviewer = (output / "review.html").read_text(encoding="utf-8")
    assert manifest["screen_pairs"] == 1
    assert manifest["correspondences"] == 2
    assert "同一 source_screen + target_screen" in reviewer
    assert "toggleCandidates" in reviewer
    assert "导出 JSONL" in reviewer
    assert r"join('\n')+'\n'" in reviewer
    assert (output / "screen_pairs.jsonl").is_file()
