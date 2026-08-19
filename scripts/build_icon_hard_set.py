#!/usr/bin/env python3
"""Build a deterministic text-less icon hard set from canonical pair JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.icon_hard_set import build_icon_hard_records
from omnitransfer.mapping_dataset import (
    validate_ui_correspondence_pair,
    write_ui_correspondence_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    args = parser.parse_args()
    records = []
    for path in args.input:
        with path.expanduser().resolve().open(encoding="utf-8") as handle:
            records.extend(
                validate_ui_correspondence_pair(json.loads(line))
                for line in handle
                if line.strip()
            )
    selected, hard_manifest = build_icon_hard_records(
        records,
        limit=args.limit,
        minimum_score=args.minimum_score,
    )
    if not selected:
        raise SystemExit("No difficult text-less icon correspondences were selected")
    pool_manifest = write_ui_correspondence_pool(selected, args.output)
    manifest = {
        **hard_manifest,
        "inputs": [str(path.expanduser().resolve()) for path in args.input],
        "output": str(args.output.expanduser().resolve()),
        "pool": pool_manifest,
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
