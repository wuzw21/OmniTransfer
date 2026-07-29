#!/usr/bin/env python3
"""Build one leakage-audited UI correspondence dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.importers import load_queries
from omnitransfer.mapping_dataset import (
    adapt_ase_queries,
    validate_ui_correspondence_pair,
    write_ui_correspondence_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-queries", type=Path)
    parser.add_argument("--ase-assets-root", type=Path)
    parser.add_argument(
        "--correspondence-pairs",
        "--page-pairs",
        dest="correspondence_pairs",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "Existing omnitransfer.ui_correspondence_pair.v1 JSONL files. "
            "--page-pairs is a compatibility alias."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs: list[dict[str, Any]] = []
    adapters: dict[str, Any] = {}
    ase_records: list[dict[str, Any]] = []
    if args.ase_queries is not None:
        query_path = args.ase_queries.expanduser().resolve()
        asset_root = (
            args.ase_assets_root.expanduser().resolve()
            if args.ase_assets_root is not None
            else query_path.parent.parent
        )
        ase_records = adapt_ase_queries(
            load_queries(query_path),
            asset_root=asset_root,
        )
        inputs.append(_input_manifest(query_path, kind="ase_queries"))
        adapters["ase2023"] = {
            "records": len(ase_records),
            "asset_root": str(asset_root),
            "output_schema": "omnitransfer.ui_correspondence_pair.v1",
        }

    correspondence_adapter = {
        "records": 0,
        "files": len(args.correspondence_pairs),
        "validated_without_schema_conversion": True,
    }
    for pair_path in args.correspondence_pairs:
        path = pair_path.expanduser().resolve()
        inputs.append(_input_manifest(path, kind="ui_correspondence_pairs"))
    if args.correspondence_pairs:
        adapters["existing_ui_correspondence_pairs"] = correspondence_adapter

    if not ase_records and not args.correspondence_pairs:
        raise SystemExit(
            "Provide --ase-queries and/or at least one --correspondence-pairs file"
        )

    def records() -> Iterable[dict[str, Any]]:
        yield from ase_records
        for pair_path in args.correspondence_pairs:
            path = pair_path.expanduser().resolve()
            for row in _iter_correspondence_pairs(path):
                correspondence_adapter["records"] += 1
                yield row

    manifest = write_ui_correspondence_dataset(
        records(),
        args.output_dir,
        metadata={
            "inputs": inputs,
            "protocol": {
                "record_schema_identical_across_datasets_and_splits": True,
                "one_training_entrypoint": "scripts/train_relation_aware_matcher.py",
                "train_dev_test_differ_only_by_split_and_label_status": True,
                "guiodyssey_included": False,
                "stable_ids_are_offline_labels_only": True,
            },
            "adapters": adapters,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _iter_correspondence_pairs(path: Path) -> Iterable[dict[str, Any]]:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = validate_ui_correspondence_pair(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid UI correspondence pair {path}:{line_number}: {exc}"
                ) from exc
            rows += 1
            yield row
    if rows == 0:
        raise ValueError(f"UI-correspondence JSONL is empty: {path}")


def _input_manifest(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    main()
