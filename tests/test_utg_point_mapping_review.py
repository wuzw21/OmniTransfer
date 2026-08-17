import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_utg_point_mapping_review.py"
    spec = importlib.util.spec_from_file_location("build_utg_point_mapping_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _state(root: Path, role: str, state_id: str, width: int, height: int) -> dict:
    screenshot = root / f"{role}-{state_id}.png"
    state_path = root / f"{role}-{state_id}.json"
    Image.new("RGB", (width, height), "white").save(screenshot)
    state = {
        "schema_version": "omnitransfer.utg_state.v1",
        "state_id": state_id,
        "nodes": [
            {
                "node_key": f"{role}-root",
                "class": "android.widget.FrameLayout",
                "resource_id": "",
                "text": "",
                "content_desc": "",
                "clickable": False,
                "enabled": True,
                "scrollable": False,
                "bounds": [0, 0, width, height],
            },
            {
                "node_key": f"{role}-button",
                "class": "android.widget.Button",
                "resource_id": "app:id/next",
                "text": "Next",
                "content_desc": "",
                "clickable": True,
                "enabled": True,
                "scrollable": False,
                "bounds": [10, 20, 50, 70],
            },
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {"json": str(state_path), "screenshot": str(screenshot), "complete": True}


def _candidates(tmp_path: Path) -> list[dict]:
    source_evidence = _state(tmp_path, "source", "source-state", 100, 200)
    target_evidence = _state(tmp_path, "small", "target-state", 200, 100)
    source_transition = {
        "role": "source",
        "event_index": 7,
        "source_page": {"page_id": "source-state", "width": 100, "height": 200},
        "source_state_evidence": source_evidence,
        "action": {
            "type": "tap",
            "node_key": "source-button",
            "x": 30,
            "y": 45,
            "bounds": [10, 20, 50, 70],
        },
    }
    target_transition = {
        "role": "small",
        "event_index": 3,
        "source_page": {"page_id": "target-state", "width": 200, "height": 100},
        "source_state_evidence": target_evidence,
        "action": {
            "type": "tap",
            "node_key": "small-button",
            "x": 30,
            "y": 45,
            "bounds": [10, 20, 50, 70],
        },
    }
    return [
        {
            "candidate_id": "candidate-1",
            "target_role": "small",
            "reasons": ["small_margin"],
            "ambiguity_priority": 0.9,
            "source_transition": source_transition,
            "target_candidates": [
                {"rank": 1, "score": 0.8, "transition": target_transition},
                {"rank": 2, "score": 0.7, "transition": target_transition},
            ],
        }
    ]


def test_build_tasks_emits_one_fixed_source_point_mapping_per_target_page(
    tmp_path: Path,
) -> None:
    module = _load_module()

    tasks, audit = module.build_tasks("legacy", "app.example", _candidates(tmp_path))

    assert audit == {
        "candidate_groups": 1,
        "deduplicated_target_pages": 1,
        "review_tasks": 1,
        "skipped": {},
        "target_roles": {"small": 1},
    }
    task = tasks[0]
    assert task["source"]["node"]["node_id"] == "source-button"
    assert task["source"]["point"] == {
        "x": 30.0,
        "y": 45.0,
        "coordinate_space": "page_pixels",
        "node_id": "source-button",
    }
    assert task["target"]["node"] is None
    assert {node["node_id"] for node in task["target"]["candidates"]} == {
        "small-root",
        "small-button",
    }
    assert task["gold_proposal"] is None
    assert task["matcher_prediction"]["node"]["node_id"] == "small-button"
    assert task["matcher_prediction"]["point"] == {
        "x": 30.0,
        "y": 45.0,
        "coordinate_space": "page_pixels",
        "node_id": "small-button",
    }
    assert task["matcher_prediction"]["reason"] == "utg_action_candidate_non_gold"
    assert task["matcher_prediction"]["probability"] == 0.8


def test_write_app_review_uses_canonical_minimal_mapping_workbench(
    tmp_path: Path,
) -> None:
    module = _load_module()
    tasks, audit = module.build_tasks("legacy", "app.example", _candidates(tmp_path))

    manifest = module.write_app_review(
        "legacy",
        "app.example",
        tasks,
        audit,
        tmp_path / "review",
    )

    assert manifest["tasks"] == 1
    html = (tmp_path / "review" / "review.html").read_text(encoding="utf-8")
    assert "review_annotation_template.html" not in html
    assert '"protocol": "human_action_mapping"' in html
    assert '"fixed_source": true' in html
    assert '"minimal": true' in html
    assert "UTG 自动候选 P（非金标）" in html
    assert "task.matcher_prediction?.point" in html
    assert "omnitransfer.ui_correspondence_pair.v1" in html
    assert "omnitransfer_ui_correspondence_pairs.jsonl" in html
    assert "确认 P / H 映射" in html
    assert "目标不存在（NULL）" in html
    assert "Cluster 应拆分" not in html
    assert "有效行为分歧" not in html
    assert (tmp_path / "review" / "tasks.jsonl").is_file()


def test_root_index_lists_apps_directly_for_one_batch(tmp_path: Path) -> None:
    module = _load_module()
    reports = [
        {
            "batch": "legacy",
            "apps": 2,
            "tasks": 3,
            "app_reports": [
                {"app": "app.one", "tasks": 3},
                {"app": "app.two", "tasks": 0},
            ],
        }
    ]

    module._write_root_index(tmp_path, reports)

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "app.one" in html
    assert "app.two" in html
    assert 'href="legacy/app.one/review.html"' in html
    assert "3 个点映射" in html
    assert "0 个点映射" in html
