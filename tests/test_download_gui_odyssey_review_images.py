import importlib.util
import json
from pathlib import Path
import struct


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_gui_odyssey_review_images.py"
SPEC = importlib.util.spec_from_file_location("download_gui_odyssey_review_images", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_png_dimensions_reads_ihdr(tmp_path: Path) -> None:
    path = tmp_path / "screen.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 2208, 1840))

    assert MODULE._png_dimensions(path) == (2208, 1840)
    assert MODULE._compatible_dimensions((1840, 2208), (2208, 1840)) is True


def test_queue_images_can_limit_pair_count(tmp_path: Path) -> None:
    queue = tmp_path / "pairs.jsonl"
    queue.write_text(
        "\n".join(
            json.dumps(
                {
                    "source": {
                        "screenshot": f"source-{index}.png",
                        "width": 720,
                        "height": 1280,
                    },
                    "target": {
                        "screenshot": f"target-{index}.png",
                        "width": 2208,
                        "height": 1840,
                    },
                }
            )
            for index in range(3)
        ),
        encoding="utf-8",
    )

    requested = MODULE._queue_images(queue, max_pairs=2)

    assert set(requested) == {
        "source-0.png",
        "target-0.png",
        "source-1.png",
        "target-1.png",
    }
