import json
from pathlib import Path
import subprocess
import sys

from omnitransfer.gui_odyssey import graph_from_gui_odyssey_step


def _episode() -> dict:
    return {
        "episode_id": "ep1",
        "device_info": {"w": 1000, "h": 2000, "device_name": "Pixel Fold"},
        "task_info": {
            "category": "Multi_Apps",
            "app": ["Opera", "Tasks"],
            "instruction": "Find a recipe and save ingredients.",
        },
        "steps": [
            {
                "step": 0,
                "screenshot": "ep1_0.png",
                "action": "CLICK",
                "info": [[500, 900], [500, 900]],
                "sam2_bbox": [470, 860, 540, 940],
                "low_level_instruction": "Open Opera.",
            },
            {"step": 1, "action": "CLICK", "info": "KEY_HOME", "sam2_bbox": []},
        ],
    }


def test_gui_odyssey_step_becomes_weak_action_graph() -> None:
    graph = graph_from_gui_odyssey_step(_episode(), _episode()["steps"][0])

    assert graph.width == 1000
    assert graph.height == 2000
    assert graph.metadata["source_format"] == "guiodyssey_weak_action"
    assert graph.metadata["device_name"] == "Pixel Fold"
    assert graph.metadata["action_point_label"] == (500.0, 1800.0)
    assert graph.metadata["action_bbox_label"] == (470.0, 1720.0, 540.0, 1880.0)
    target = next(node for node in graph.nodes if node.node_id == "action_target")
    assert target.clickable is True
    assert target.bbox == (470.0, 1720.0, 540.0, 1880.0)
    assert target.text == ""
    assert target.content_desc == ""


def test_gui_odyssey_importer_writes_split_graphs(tmp_path: Path) -> None:
    annotation = tmp_path / "episode.json"
    annotation.write_text(json.dumps(_episode()), encoding="utf-8")
    output = tmp_path / "bundle"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/import_gui_odyssey.py",
            "--annotations",
            str(annotation),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    assert manifest["accepted_screens"] == 1
    assert (output / "manifest.json").is_file()
    graph_rows = [
        line
        for path in output.glob("graphs.*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(graph_rows) == 1
