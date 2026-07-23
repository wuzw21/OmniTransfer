#!/usr/bin/env python3
"""Download only GUIOdyssey screenshots referenced by a review queue."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from omnitransfer.gui_odyssey_images import MIRRORS, gui_odyssey_image_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the small screenshot subset referenced by a GUIOdyssey review queue."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_pairs < 0:
        raise SystemExit("--max-pairs must be non-negative")

    requested = _queue_images(
        args.queue.expanduser().resolve(),
        max_pairs=args.max_pairs,
    )
    index = gui_odyssey_image_index(set(requested))
    missing = sorted(set(requested) - set(index))
    if missing:
        raise RuntimeError(f"{len(missing)} queue screenshots are absent from pinned mirrors: {missing[:5]}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                filename,
                requested[filename],
                index[filename],
                output,
            ): filename
            for filename in sorted(requested)
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: row["filename"])
    manifest = {
        "schema_version": "omnitransfer_guiodyssey_review_images_v1",
        "queue": str(args.queue.expanduser().resolve()),
        "image_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "mirrors": [
            {"repo_id": repo_id, "revision": revision, "role": "individual_file_transport"}
            for repo_id, revision in MIRRORS
        ],
        "images": records,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "images"}, ensure_ascii=False))


def _queue_images(
    path: Path,
    *,
    max_pairs: int = 0,
) -> dict[str, tuple[int, int]]:
    requested: dict[str, tuple[int, int]] = {}
    rows_read = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if max_pairs and rows_read >= max_pairs:
            break
        rows_read += 1
        row = json.loads(line)
        for side in ("source", "target"):
            item = row[side]
            filename = str(item["screenshot"])
            dimensions = (round(float(item["width"])), round(float(item["height"])))
            previous = requested.setdefault(filename, dimensions)
            if previous != dimensions:
                raise ValueError(f"conflicting dimensions for {filename}: {previous} != {dimensions}")
    return requested


def _download_one(
    filename: str,
    expected_dimensions: tuple[int, int],
    source: tuple[str, str, str],
    output: Path,
) -> dict[str, Any]:
    repo_id, revision, remote_path = source
    destination = output / filename
    if destination.is_file():
        dimensions = _png_dimensions(destination)
        if _compatible_dimensions(dimensions, expected_dimensions):
            return _image_record(
                destination,
                repo_id,
                revision,
                remote_path,
                expected_dimensions=expected_dimensions,
                reused=True,
            )
    url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
        f"{quote(remote_path, safe='/')}?download=true"
    )
    part = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "OmniTransfer/0.1 GUIOdyssey review"})
    with urlopen(request, timeout=180) as response, part.open("wb") as writer:
        while chunk := response.read(1024 * 1024):
            writer.write(chunk)
    dimensions = _png_dimensions(part)
    if not _compatible_dimensions(dimensions, expected_dimensions):
        part.unlink(missing_ok=True)
        raise ValueError(f"unexpected dimensions for {filename}: {dimensions} != {expected_dimensions}")
    os.replace(part, destination)
    return _image_record(
        destination,
        repo_id,
        revision,
        remote_path,
        expected_dimensions=expected_dimensions,
        reused=False,
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as reader:
        header = reader.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def _compatible_dimensions(
    actual: tuple[int, int],
    annotation: tuple[int, int],
) -> bool:
    return actual == annotation or actual == tuple(reversed(annotation))


def _image_record(
    path: Path,
    repo_id: str,
    revision: str,
    remote_path: str,
    *,
    expected_dimensions: tuple[int, int],
    reused: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
    width, height = _png_dimensions(path)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "width": width,
        "height": height,
        "annotation_width": expected_dimensions[0],
        "annotation_height": expected_dimensions[1],
        "dimension_relation": "exact" if (width, height) == expected_dimensions else "orientation_swapped",
        "repo_id": repo_id,
        "revision": revision,
        "remote_path": remote_path,
        "reused": reused,
    }


if __name__ == "__main__":
    main()
