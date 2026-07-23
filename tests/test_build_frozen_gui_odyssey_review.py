import json
import os
from pathlib import Path
import subprocess
import sys


def test_frozen_review_preserves_pair_ids_and_split(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    output = tmp_path / "review"
    candidates = [{"pair_id": "pair-a"}, {"pair_id": "pair-b"}]
    raw_rows = [
        _raw_pair("pair-a", "Pixel Fold", "Medium Phone", 0.95),
        _raw_pair("pair-b", "Pixel Tablet", "Pixel 8 Pro", 0.90),
    ]
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_frozen_gui_odyssey_review.py",
            "--candidates",
            str(candidate_path),
            "--raw-pairs",
            str(raw_path),
            "--output",
            str(output),
            "--reserved-split",
            "test",
            "--limit",
            "2",
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "src:scripts"},
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [
        json.loads(line)
        for line in (output / "review_queue.jsonl").read_text().splitlines()
    ]
    assert {row["pair_id"] for row in rows} == {"pair-a", "pair-b"}
    assert all(row["selection"]["reserved_split"] == "test" for row in rows)
    assert all("semantic_score" in row["selection"] for row in rows)
    assert all("source_sequence_position" in row["selection"] for row in rows)
    assert all(row["annotation"]["status"] == "unreviewed" for row in rows)
    assert all(len(row["annotation"]["source_nodes"]) == 1 for row in rows)
    assert all(len(row["annotation"]["target_nodes"]) == 1 for row in rows)
    assert rows[0]["annotation"]["source_nodes"][0]["point_normalized"] == [500, 800]
    assert rows[0]["annotation"]["target_nodes"][0]["point_normalized"] == [600, 400]
    assert rows[0]["source"]["point"] == [500, 1600]
    assert rows[0]["target"]["point"] == [1200, 400]
    assert all(row["annotation"]["matches"] == [] for row in rows)
    assert (output / "review.html").is_file()


def _raw_pair(
    pair_id: str,
    source_device: str,
    target_device: str,
    score: float,
) -> dict:
    return {
        "pair_id": pair_id,
        "episode_pair_id": f"trajectory-{pair_id}",
        "apps": ["Firefox"],
        "meta_task": f"Task {pair_id}",
        "source": {
            "episode_id": f"source-{pair_id}",
            "device_name": source_device,
            "width": 1000,
            "height": 2000,
            "step_index": 1,
            "screenshot": f"source-{pair_id}.png",
            "point": [500, 800],
            "instruction": f"Open {pair_id}.",
            "description": "A browser downloads page.",
        },
        "target": {
            "episode_id": f"target-{pair_id}",
            "device_name": target_device,
            "width": 2000,
            "height": 1000,
            "step_index": 2,
            "screenshot": f"target-{pair_id}.png",
            "point": [600, 400],
            "instruction": "Open settings.",
            "description": "A browser downloads page.",
        },
        "alignment": {"step_score": score, "episode_score": score},
    }
