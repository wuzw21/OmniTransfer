#!/usr/bin/env python3
"""Build a single-file GUIOdyssey reviewer with embedded WebP images."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageOps

from build_gui_odyssey_review_queue import _review_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed review screenshots into one browser-independent HTML file."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edge", type=int, default=1400)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()
    if args.max_edge <= 0 or not 1 <= args.quality <= 100:
        raise SystemExit("--max-edge must be positive and --quality must be between 1 and 100")

    queue_path = args.queue.expanduser().resolve()
    images = args.images.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encoded: dict[str, str] = {}
    for row in rows:
        for side in ("source", "target"):
            filename = str(row[side]["screenshot"])
            if filename not in encoded:
                encoded[filename] = _image_data_uri(
                    images / filename,
                    max_edge=args.max_edge,
                    quality=args.quality,
                )
            row[side]["image_url"] = encoded[filename]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_review_html(rows, redirect_file=False), encoding="utf-8")
    manifest = {
        "schema_version": "omnitransfer_guiodyssey_standalone_review_v1",
        "queue": str(queue_path),
        "image_root": str(images),
        "pair_count": len(rows),
        "unique_images": len(encoded),
        "max_edge": args.max_edge,
        "webp_quality": args.quality,
        "output": str(output),
        "output_bytes": output.stat().st_size,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _image_data_uri(path: Path, *, max_edge: int, quality: int) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="WEBP", quality=quality, method=6)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{payload}"


if __name__ == "__main__":
    main()
