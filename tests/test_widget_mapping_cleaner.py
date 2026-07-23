from __future__ import annotations

import json
from pathlib import Path
import struct

from omnitransfer.ui_graph import graph_from_record
from omnitransfer.widget_mapping_cleaner import clean_widget_mapping_dataset


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


def test_cleaner_binds_real_nodes_and_writes_relative_xml(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    source_dir = dataset / "Tools" / "Example" / "iOS"
    target_dir = dataset / "Tools" / "Example" / "Android"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (source_dir / "0.xml").write_text(
        """<AppiumAUT><XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="100" height="200"><XCUIElementTypeButton type="XCUIElementTypeButton" label="Delete" x="10" y="20" width="20" height="10" clickable="true"/></XCUIElementTypeApplication></AppiumAUT>""",
        encoding="utf-8",
    )
    (target_dir / "0.xml").write_text(
        """<hierarchy width="300" height="600" bounds="[0,0][300,600]"><node class="android.widget.Button" text="Delete" bounds="[30,300][90,330]" clickable="true"/><node class="android.widget.Button" text="Archive" bounds="[120,300][210,330]" clickable="true"/></hierarchy>""",
        encoding="utf-8",
    )
    (source_dir / "0.png").write_bytes(_png_header(300, 600))
    (target_dir / "0.png").write_bytes(_png_header(300, 640))
    testset = dataset / "testset.txt"
    testset.write_text(
        "XCUIElementTypeButton[30,60][90,90] android.widget.Button[30,300][90,330] Example Tools/Example/iOS/0 Tools/Example/Android/0 T0\n",
        encoding="utf-8",
    )

    output = tmp_path / "clean"
    result = clean_widget_mapping_dataset(testset, output)

    query = json.loads((output / "queries.jsonl").read_text(encoding="utf-8"))
    mapping = json.loads((output / "mappings.jsonl").read_text(encoding="utf-8"))
    assert result["query_count"] == 1
    assert query["source"]["text"] == "Delete"
    assert query["source"]["text"] != "Example"
    assert query["source"]["metadata"]["semantics_source"] == "bound_xml_node"
    assert query["source"]["bounds"] == [0.1, 0.1, 0.3, 0.15]
    assert all(
        not candidate["candidate_id"].startswith("public_")
        for candidate in query["target_candidates"]
    )
    assert len({row["candidate_id"] for row in query["target_candidates"]}) == 2
    assert query["gold_candidate_id"] in {
        row["candidate_id"] for row in query["target_candidates"]
    }
    source_xml = Path(query["metadata"]["source_xml_path"])
    if not source_xml.is_absolute():
        source_xml = Path(__file__).resolve().parents[1] / source_xml
    graph = graph_from_record(
        {"xml": source_xml.read_text(encoding="utf-8")}, graph_id="source"
    )
    assert graph.width == 1.0
    assert graph.height == 1.0
    bound = next(node for node in graph.nodes if node.node_id == query["source"]["node_id"])
    assert bound.bbox == (0.1, 0.1, 0.3, 0.15)
    assert bound.metadata["visual_bbox"] == (0.1, 0.1, 0.3, 0.15)
    assert mapping["source"]["node_id"] == query["source"]["node_id"]
    assert mapping["target"]["node_id"] == query["gold_candidate_id"]
    assert "target_candidates" not in mapping


def test_cleaner_uses_real_same_bbox_nodes_as_set_valued_gold(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    source_dir = dataset / "Tools" / "Example" / "iOS"
    target_dir = dataset / "Tools" / "Example" / "Android"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (source_dir / "0.xml").write_text(
        """<AppiumAUT><XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="100" height="200"><XCUIElementTypeButton type="XCUIElementTypeButton" label="Delete" x="10" y="20" width="20" height="10"/></XCUIElementTypeApplication></AppiumAUT>""",
        encoding="utf-8",
    )
    (target_dir / "0.xml").write_text(
        """<hierarchy width="300" height="600" bounds="[0,0][300,600]"><node class="android.widget.Button" bounds="[30,300][90,330]" clickable="true"><node class="android.widget.TextView" text="Delete" bounds="[30,300][90,330]"/></node></hierarchy>""",
        encoding="utf-8",
    )
    (source_dir / "0.png").write_bytes(_png_header(300, 600))
    (target_dir / "0.png").write_bytes(_png_header(300, 600))
    testset = dataset / "testset.txt"
    testset.write_text(
        "XCUIElementTypeButton[30,60][90,90] android.widget.Button[30,300][90,330] Example Tools/Example/iOS/0 Tools/Example/Android/0 T0\n",
        encoding="utf-8",
    )

    output = tmp_path / "clean"
    clean_widget_mapping_dataset(testset, output)
    query = json.loads((output / "queries.jsonl").read_text(encoding="utf-8"))

    assert len(query["metadata"]["gold_equivalent_candidate_ids"]) == 2
    assert set(query["metadata"]["gold_equivalent_candidate_ids"]) == {
        row["candidate_id"] for row in query["target_candidates"]
    }


def test_cleaner_excludes_hidden_xml_nodes_from_candidates(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    source_dir = dataset / "Tools" / "Example" / "iOS"
    target_dir = dataset / "Tools" / "Example" / "Android"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (source_dir / "0.xml").write_text(
        """<AppiumAUT><XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="100" height="200"><XCUIElementTypeButton type="XCUIElementTypeButton" label="Delete" x="10" y="20" width="20" height="10"/></XCUIElementTypeApplication></AppiumAUT>""",
        encoding="utf-8",
    )
    (target_dir / "0.xml").write_text(
        """<hierarchy width="300" height="600" bounds="[0,0][300,600]"><node class="android.widget.Button" text="Delete" bounds="[30,300][90,330]" clickable="true" visible="true"/><node class="android.widget.Button" text="Hidden" bounds="[120,300][210,330]" clickable="true" visible="false"/></hierarchy>""",
        encoding="utf-8",
    )
    (source_dir / "0.png").write_bytes(_png_header(300, 600))
    (target_dir / "0.png").write_bytes(_png_header(300, 600))
    testset = dataset / "testset.txt"
    testset.write_text(
        "XCUIElementTypeButton[30,60][90,90] android.widget.Button[30,300][90,330] Example Tools/Example/iOS/0 Tools/Example/Android/0 T0\n",
        encoding="utf-8",
    )

    output = tmp_path / "clean"
    clean_widget_mapping_dataset(testset, output)
    query = json.loads((output / "queries.jsonl").read_text(encoding="utf-8"))

    assert [candidate["text"] for candidate in query["target_candidates"]] == [
        "Delete"
    ]
