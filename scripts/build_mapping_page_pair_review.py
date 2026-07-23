#!/usr/bin/env python3
"""Build a static HTML reviewer for unified mapping page-pair JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_pair_review import build_mapping_pair_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=0)
    args = parser.parse_args()
    manifest = build_mapping_pair_review(
        args.input,
        args.output_dir,
        pair_limit=args.pair_limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
