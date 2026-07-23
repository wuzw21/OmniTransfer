import json
from pathlib import Path
import subprocess
import sys

from omnitransfer.gui_odyssey_review import (
    build_gui_odyssey_review_queue,
    build_gui_odyssey_sequence_alignment_queue,
)


def _episode(episode_id: str, device: str, instruction: str, point: tuple[int, int]) -> dict:
    dimensions = (2208, 1840) if "Fold" in device else (1080, 2400)
    x, y = point
    return {
        "episode_id": episode_id,
        "device_info": {"w": dimensions[0], "h": dimensions[1], "device_name": device},
        "task_info": {
            "category": "Information_Management",
            "app": ["Opera", "Tasks"],
            "meta_task": "Search for information and save it.",
            "instruction": "Use Opera and Tasks.",
        },
        "steps": [
            {
                "step": 0,
                "screenshot": f"{episode_id}_0.png",
                "action": "CLICK",
                "info": [[x, y], [x, y]],
                "sam2_bbox": [x - 40, y - 30, x + 40, y + 30],
                "low_level_instruction": instruction,
                "description": "Opera search results with a save control.",
            }
        ],
    }


def test_review_queue_prioritizes_cross_form_factor_layout_shift() -> None:
    episodes = [
        _episode("phone", "Pixel 8 Pro", "Tap the save button.", (150, 200)),
        _episode("fold", "Pixel Fold", "Tap the save button.", (850, 760)),
        _episode("tablet", "Pixel Tablet", "Select the save button.", (500, 500)),
    ]

    rows, manifest = build_gui_odyssey_review_queue(episodes, limit=2)

    assert rows
    assert manifest["policy"]["gold_label"] == "human_only"
    assert rows[0]["annotation"]["label"] is None
    assert rows[0]["selection"]["is_gold_label"] is False
    assert "bbox_normalized" not in rows[0]["source"]
    assert rows[0]["annotation"]["source_nodes"][0]["point_normalized"] == rows[0][
        "source"
    ]["point_normalized"]
    assert rows[0]["annotation"]["target_nodes"][0]["point_normalized"] == rows[0][
        "target"
    ]["point_normalized"]
    assert rows[0]["annotation"]["matches"] == []
    assert rows[0]["target"]["form_factor"] in {"fold", "tablet"}
    assert "cross_form_factor" in rows[0]["selection"]["reasons"]


def test_review_queue_can_require_available_screenshots() -> None:
    episodes = [
        _episode("phone", "Pixel 8 Pro", "Tap the save button.", (150, 200)),
        _episode("fold", "Pixel Fold", "Tap the save button.", (850, 760)),
        _episode("missing", "Pixel Tablet", "Tap the save button.", (500, 500)),
    ]

    rows, manifest = build_gui_odyssey_review_queue(
        episodes,
        limit=1,
        available_screenshots={"phone_0.png", "fold_0.png"},
    )

    assert rows[0]["source"]["episode_id"] != "missing"
    assert rows[0]["target"]["episode_id"] != "missing"
    assert manifest["image_availability_filter"] == "pinned_individual_mirror"


def test_review_queue_rejects_mislabeled_open_transition() -> None:
    good_phone = _episode("phone", "Pixel 8 Pro", "Open the Google Play Store.", (150, 200))
    good_fold = _episode("fold", "Pixel Fold", "Open the Google Play Store.", (850, 760))
    bad_tablet = _episode("bad", "Pixel Tablet", "Open the Google Play Store.", (500, 500))
    for episode in (good_phone, good_fold):
        episode["steps"].append(
            {
                "step": 1,
                "action": "COMPLETE",
                "description": "The Google Play Store app page is open.",
                "low_level_instruction": "Task complete.",
            }
        )
    bad_tablet["steps"].append(
        {
            "step": 1,
            "action": "CLICK",
            "description": "The Instagram notification settings page is open.",
            "low_level_instruction": "Toggle Instagram notifications off.",
        }
    )

    rows, manifest = build_gui_odyssey_review_queue(
        [good_phone, good_fold, bad_tablet],
        limit=1,
    )

    assert manifest["eligible_steps"] == 2
    assert rows[0]["source"]["episode_id"] != "bad"
    assert rows[0]["target"]["episode_id"] != "bad"
    assert "action_outcome_consistent" in rows[0]["selection"]["reasons"]


def test_review_queue_excludes_launcher_and_home_screens() -> None:
    desktop = _episode("desktop", "Pixel 8 Pro", "Open the Opera browser.", (150, 200))
    desktop["steps"][0]["description"] = (
        "This is an Android home screen displaying app icons over a wallpaper."
    )
    in_app_phone = _episode("phone", "Pixel 8 Pro", "Tap the save button.", (150, 200))
    in_app_fold = _episode("fold", "Pixel Fold", "Tap the save button.", (850, 760))

    rows, manifest = build_gui_odyssey_review_queue(
        [desktop, in_app_phone, in_app_fold],
        limit=1,
    )

    assert manifest["eligible_steps"] == 2
    assert rows[0]["source"]["screen_context"] == "in_app"
    assert rows[0]["target"]["screen_context"] == "in_app"
    assert "in_app_only" in rows[0]["selection"]["reasons"]


def test_review_queue_uses_action_points_instead_of_sam_bbox_geometry() -> None:
    phone = _episode("phone", "Pixel 8 Pro", "Download the image.", (150, 200))
    fold = _episode("fold", "Pixel Fold", "Download the image.", (850, 760))
    bad_tablet = _episode("bad", "Pixel Tablet", "Download the image.", (500, 175))
    bad_tablet["steps"][0]["sam2_bbox"] = [100, 100, 900, 250]

    rows, _ = build_gui_odyssey_review_queue([phone, fold, bad_tablet], limit=1)

    row = rows[0]
    assert "bbox_scale_ratio" not in row["selection"]
    assert "bbox" not in row["source"]
    assert "bbox" not in row["target"]
    assert row["annotation"]["source_nodes"][0]["point_normalized"] == row["source"][
        "point_normalized"
    ]


def _trajectory_episode(
    episode_id: str,
    device: str,
    steps: list[tuple[str, str, str, tuple[int, int] | None]],
) -> dict:
    episode = _episode(episode_id, device, "placeholder", (100, 100))
    episode["task_info"] = {
        "category": "Information_Management",
        "app": ["Opera", "Tasks"],
        "meta_task": "Search for information and save it.",
        "instruction": "Use Opera and Tasks.",
    }
    episode["steps"] = []
    for index, (action, instruction, description, point) in enumerate(steps):
        step = {
            "step": index,
            "screenshot": f"{episode_id}_{index}.png",
            "action": action,
            "low_level_instruction": instruction,
            "description": description,
        }
        if point is not None:
            x, y = point
            step["info"] = [[x, y], [x, y]]
            step["sam2_bbox"] = [x - 30, y - 20, x + 30, y + 20]
        episode["steps"].append(step)
    return episode


def test_sequence_alignment_is_monotonic_and_one_to_one() -> None:
    phone = _trajectory_episode(
        "phone-sequence",
        "Pixel 8 Pro",
        [
            ("CLICK", "Use Opera search.", "Opera search page.", (200, 250)),
            ("TEXT", "Type weather.", "Opera search input.", None),
            ("CLICK", "Execute the search query.", "Opera suggestions page.", (800, 900)),
            ("CLICK", "Save the result in Tasks.", "Tasks result page.", (700, 500)),
        ],
    )
    fold = _trajectory_episode(
        "fold-sequence",
        "Pixel Fold",
        [
            ("CLICK", "Open the search control in Opera.", "Opera search page.", (200, 250)),
            ("TEXT", "Enter weather.", "Opera search input.", None),
            ("SCROLL", "Scroll suggestions.", "Opera suggestions page.", None),
            ("CLICK", "Run the search query.", "Opera suggestions page.", (800, 900)),
            ("CLICK", "Save this result to Tasks.", "Tasks result page.", (700, 500)),
        ],
    )

    rows, manifest = build_gui_odyssey_sequence_alignment_queue(
        [phone, fold],
        limit=10,
        min_step_score=0.45,
    )

    assert len(rows) >= 2
    source_positions = [row["selection"]["source_sequence_position"] for row in rows]
    target_positions = [row["selection"]["target_sequence_position"] for row in rows]
    assert source_positions == sorted(set(source_positions))
    assert target_positions == sorted(set(target_positions))
    assert all(row["selection"]["one_to_one"] for row in rows)
    assert all(row["selection"]["monotonic"] for row in rows)
    assert manifest["policy"]["gold_label"] == "human_only"


def test_sequence_alignment_rejects_generic_instruction_across_active_apps() -> None:
    phone = _trajectory_episode(
        "phone-app-conflict",
        "Pixel 8 Pro",
        [("CLICK", "Execute the search query.", "Opera search suggestions.", (500, 500))],
    )
    fold = _trajectory_episode(
        "fold-app-conflict",
        "Pixel Fold",
        [("CLICK", "Execute the search query.", "Tasks search suggestions.", (500, 500))],
    )

    rows, _ = build_gui_odyssey_sequence_alignment_queue(
        [phone, fold],
        limit=10,
        min_step_score=0.1,
        min_aligned_steps=1,
    )

    assert rows == []


def test_review_queue_script_writes_standalone_reviewer(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for episode in (
        _episode("phone", "Pixel 8 Pro", "Tap the save button.", (150, 200)),
        _episode("fold", "Pixel Fold", "Tap the save button.", (850, 760)),
    ):
        (annotations / f'{episode["episode_id"]}.json').write_text(
            json.dumps(episode), encoding="utf-8"
        )
    output = tmp_path / "review"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_gui_odyssey_review_queue.py",
            "--annotations",
            str(annotations),
            "--output",
            str(output),
            "--limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    assert manifest["selected_pairs"] == 1
    assert (output / "review_queue.jsonl").is_file()
    reviewer = (output / "review.html").read_text(encoding="utf-8")
    assert "完成节点对应" in reviewer
    assert "建立所选节点对应" in reviewer
    assert "source_nodes" in reviewer
    assert "no_correspondence" in reviewer

    rerendered = tmp_path / "rerendered"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_gui_odyssey_review_queue.py",
            "--review-queue",
            str(output / "review_queue.jsonl"),
            "--output",
            str(rerendered),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    migrated = json.loads((rerendered / "review_queue.jsonl").read_text().splitlines()[0])
    assert migrated["annotation"]["source_nodes"]
    assert migrated["annotation"]["matches"] == []
    assert "bbox" not in migrated["source"]


def test_review_queue_script_writes_sequence_alignment_smoke(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    episodes = [
        _trajectory_episode(
            "phone",
            "Pixel 8 Pro",
            [
                ("CLICK", "Use Opera search.", "Opera search page.", (150, 200)),
                ("CLICK", "Save the result in Tasks.", "Tasks result page.", (700, 500)),
            ],
        ),
        _trajectory_episode(
            "fold",
            "Pixel Fold",
            [
                ("CLICK", "Use Opera search.", "Opera search page.", (800, 700)),
                ("CLICK", "Save this result to Tasks.", "Tasks result page.", (700, 500)),
            ],
        ),
    ]
    for episode in episodes:
        (annotations / f'{episode["episode_id"]}.json').write_text(
            json.dumps(episode),
            encoding="utf-8",
        )
    output = tmp_path / "sequence-review"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_gui_odyssey_review_queue.py",
            "--annotations",
            str(annotations),
            "--output",
            str(output),
            "--limit",
            "4",
            "--sequence-alignment",
            "--min-step-score",
            "0.4",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(result.stdout)
    assert manifest["selected_click_pairs"] == 2
    rows = [json.loads(line) for line in (output / "review_queue.jsonl").read_text().splitlines()]
    assert all(row["selection"]["candidate_kind"] == "sequence_aligned_correspondence" for row in rows)
    assert rows[0]["source"]["next_instruction"]
    assert "trajectory" in (output / "review.html").read_text(encoding="utf-8")
