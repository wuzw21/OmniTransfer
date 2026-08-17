#!/usr/bin/env python3
"""Split canonical UI correspondence pairs with leakage auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.mapping_dataset import (
    iter_split_ui_correspondence_pool,
    plan_ui_correspondence_pool,
    validate_ui_correspondence_pair,
    write_ui_correspondence_dataset,
    write_ui_correspondence_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correspondence-pairs",
        nargs="+",
        type=Path,
        required=True,
        help="omnitransfer.ui_correspondence_pair.v1 JSONL files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--dev-percent", type=int, default=10)
    args = parser.parse_args()

    inputs = [
        _input_manifest(path.expanduser().resolve())
        for path in args.correspondence_pairs
    ]
    output = args.output_dir.expanduser().resolve()
    pool = write_ui_correspondence_pool(
        _iter_inputs(args.correspondence_pairs),
        output / "pool.jsonl",
    )
    split_plan = plan_ui_correspondence_pool(
        output / "pool.jsonl",
        seed=args.split_seed,
        train_percent=args.train_percent,
        dev_percent=args.dev_percent,
    )
    manifest = write_ui_correspondence_dataset(
        iter_split_ui_correspondence_pool(output / "pool.jsonl", split_plan),
        output,
        metadata={
            "inputs": inputs,
            "pool": pool,
            "split": split_plan.audit,
            "protocol": {
                "record_schema": "omnitransfer.ui_correspondence_pair.v1",
                "split_protocol": "within_app_page_component_v1",
                "pair_page_component_overlap": "forbidden",
                "formal_metrics_require_label_status": "gold",
            },
        },
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _iter_inputs(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for value in paths:
        path = value.expanduser().resolve()
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield validate_ui_correspondence_pair(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid UI correspondence pair {path}:{line_number}: {exc}"
                    ) from exc
                rows += 1
        if rows == 0:
            raise ValueError(f"UI-correspondence JSONL is empty: {path}")


def _input_manifest(path: Path) -> dict[str, Any]:
    return {
        "kind": "ui_correspondence_pairs",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    main()
