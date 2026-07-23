import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def test_embed_reviewer_is_single_file(tmp_path: Path) -> None:
    images = tmp_path / "screenshots"
    images.mkdir()
    Image.new("RGB", (40, 80), "navy").save(images / "a.png")
    row = {
        "schema_version": "omnitransfer_guiodyssey_pair_review_v2",
        "pair_id": "pair",
        "source": _side("a.png"),
        "target": _side("a.png"),
        "selection": {
            "candidate_kind": "likely_correspondence",
            "priority_score": 1.0,
            "semantic_score": 1.0,
            "reasons": [],
        },
        "annotation": {"status": "unreviewed", "label": None},
    }
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "review_standalone.html"

    subprocess.run(
        [
            sys.executable,
            "scripts/embed_gui_odyssey_review_images.py",
            "--queue",
            str(queue),
            "--images",
            str(images),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    html = output.read_text(encoding="utf-8")
    assert "data:image/webp;base64," in html
    assert "location.replace" not in html
    assert r"join('\n') + '\n'" in html


def _side(filename: str) -> dict:
    return {
        "screenshot": filename,
        "image_url": f"screenshots/{filename}",
        "bbox_normalized": [100, 100, 200, 200],
        "width": 40,
        "height": 80,
        "instruction": "Tap the control.",
        "device_name": "Phone",
        "apps": ["App"],
        "description": "A screen.",
    }
