import csv
import json
from pathlib import Path

from PIL import Image

from omnitransfer.mobileviews_trace_pairs import build_mobileviews_trace_pair_pilot


def _state(path: Path, state_id: str, text: str) -> None:
    views = [
        {
            "temp_id": 0,
            "parent": -1,
            "children": [1, 2, 3],
            "visible": True,
            "bounds": [[0, 0], [100, 200]],
            "class": "android.widget.FrameLayout",
            "view_str": "root",
        },
        {
            "temp_id": 1,
            "parent": 0,
            "visible": True,
            "bounds": [[0, 0], [50, 50]],
            "class": "android.widget.Button",
            "text": "Save",
            "clickable": True,
            "view_str": "save",
        },
        {
            "temp_id": 2,
            "parent": 0,
            "visible": True,
            "bounds": [[0, 60], [50, 100]],
            "class": "android.widget.TextView",
            "text": text,
            "view_str": "row",
        },
        {
            "temp_id": 3,
            "parent": 0,
            "visible": True,
            "bounds": [[0, 110], [50, 150]],
            "class": "android.widget.TextView",
            "text": text,
            "view_str": "row",
        },
    ]
    (path / f"state_{state_id}.json").write_text(
        json.dumps({"width": 100, "height": 200, "views": views}), encoding="utf-8"
    )
    Image.new("RGB", (100, 200), "white").save(path / f"screen_{state_id}.jpg")


def test_mobileviews_trace_pair_is_unreviewed_and_set_valued(tmp_path: Path) -> None:
    states = tmp_path / "states"
    states.mkdir()
    _state(states, "1", "Alpha")
    _state(states, "2", "Beta")
    with (tmp_path / "screenshot_state_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["screen_id", "state_str", "structure_str", "vh_json_id", "vh_xml_id"],
        )
        writer.writeheader()
        for state_id in ("1", "2"):
            writer.writerow(
                {
                    "screen_id": f"states/screen_{state_id}.jpg",
                    "state_str": f"content-{state_id}",
                    "structure_str": "same-layout",
                    "vh_json_id": f"states/state_{state_id}.json",
                    "vh_xml_id": "",
                }
            )

    records, manifest = build_mobileviews_trace_pair_pilot(tmp_path, pair_limit=10)

    assert manifest["selected_pairs"] == 1
    assert manifest["set_valued_rows"] == 2
    assert len(records) == 1
    record = records[0]
    assert record["split"] == "diagnostic"
    assert record["label_status"] == "unreviewed"
    assert record["provenance"]["annotation"] == "automatic_same_view_str_proposal"
    row_matches = [match for match in record["matches"] if len(match["target_node_ids"]) == 2]
    assert len(row_matches) == 2
