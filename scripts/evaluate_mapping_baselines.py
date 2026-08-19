#!/usr/bin/env python3
"""Evaluate exact text/resource-id baselines on canonical UI pair JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.mapping_baselines import evaluate_exact_identity_baselines
from omnitransfer.mapping_dataset import validate_ui_correspondence_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.input:
        with path.expanduser().resolve().open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = validate_ui_correspondence_pair(json.loads(line))
                if record["split"] == args.split:
                    records.append(record)
    if not records:
        raise SystemExit(f"No {args.split} records were accepted")
    report = {
        "inputs": [
            {
                "path": str(path.expanduser().resolve()),
                "sha256": file_sha256(path.expanduser().resolve()),
            }
            for path in args.input
        ],
        "split": args.split,
        **evaluate_exact_identity_baselines(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
