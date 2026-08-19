#!/usr/bin/env python3
"""Normalize raw iOS screen rows into canonical screen records.

This is the explicit preprocessing seam for iOS. Downstream dataset builders,
reviewers, and matchers consume its canonical graph contract rather than
reimplementing iOS XML or screenshot coordinate handling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.ios_adapter import load_ios_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw screens JSONL")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="canonical screens JSONL")
    args = parser.parse_args()

    rows = []
    with args.input.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(load_ios_screen(raw, root=args.repo_root.expanduser().resolve()))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"line {line_number}: iOS normalization failed: {exc}") from exc

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(json.dumps({"screens": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
