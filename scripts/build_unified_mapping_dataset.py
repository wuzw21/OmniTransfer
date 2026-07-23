#!/usr/bin/env python3
"""Build one leakage-audited page-pair dataset for training and evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from omnitransfer.generalization import split_gui_odyssey_pair_rows
from omnitransfer.importers import load_queries
from omnitransfer.mapping_dataset import (
    adapt_ase_queries,
    adapt_gui_odyssey_human_reviews,
    adapt_gui_odyssey_rows,
    validate_mapping_page_pair,
    write_mapping_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-queries", type=Path, required=True)
    parser.add_argument("--gui-pairs", type=Path, required=True)
    parser.add_argument("--gui-annotations", type=Path, required=True)
    parser.add_argument("--gui-screenshots", type=Path)
    parser.add_argument("--gui-human-review", type=Path)
    parser.add_argument("--human-screenshots", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()

    gui_rows = _load_jsonl(args.gui_pairs)
    gui_split = split_gui_odyssey_pair_rows(gui_rows, seed=args.split_seed)
    ase_records = adapt_ase_queries(load_queries(args.ase_queries))

    gui_train, train_adapter = adapt_gui_odyssey_rows(
        list(gui_split.splits["train"]),
        args.gui_annotations,
        split="train",
        screenshot_dir=args.gui_screenshots,
    )
    diagnostics: list[dict[str, Any]] = []
    review_candidates: dict[str, list[dict[str, Any]]] = {}
    heldout_adapters: dict[str, dict[str, Any]] = {}
    heldout_pair_ids: dict[str, set[str]] = {}
    for reserved_split in ("dev", "test"):
        weak_records, adapter = adapt_gui_odyssey_rows(
            list(gui_split.splits[reserved_split]),
            args.gui_annotations,
            split="diagnostic",
            screenshot_dir=args.gui_screenshots,
        )
        for record in weak_records:
            record["provenance"]["reserved_split"] = reserved_split
        diagnostics.extend(weak_records)
        review_candidates[reserved_split] = [
            _review_candidate(record, reserved_split=reserved_split)
            for record in weak_records
        ]
        heldout_adapters[reserved_split] = adapter
        heldout_pair_ids[reserved_split] = {
            str(row.get("pair_id") or "") for row in gui_split.splits[reserved_split]
        }

    human_gold: list[dict[str, Any]] = []
    human_manifest: dict[str, Any] = {
        "status": "unavailable",
        "reason": "no exported reviewed GUIOdyssey JSONL was provided",
        "accepted_pairs": 0,
    }
    if args.gui_human_review is not None:
        review_rows = _load_jsonl(args.gui_human_review, allow_empty=True)
        screenshot_dir = args.human_screenshots or args.gui_human_review.parent
        human_manifest = {
            "status": "loaded",
            "input_rows_read": len(review_rows),
            "unknown_pair_ids": [],
            "splits": {},
        }
        known_pair_ids = set().union(*heldout_pair_ids.values())
        human_manifest["unknown_pair_ids"] = sorted(
            {
                str(row.get("pair_id") or "")
                for row in review_rows
                if str(row.get("pair_id") or "") not in known_pair_ids
            }
        )
        for reserved_split in ("dev", "test"):
            split_rows = [
                row
                for row in review_rows
                if str(row.get("pair_id") or "") in heldout_pair_ids[reserved_split]
            ]
            gold_records, adapter = adapt_gui_odyssey_human_reviews(
                split_rows,
                split=reserved_split,
                screenshot_dir=screenshot_dir,
            )
            human_gold.extend(gold_records)
            human_manifest["splits"][reserved_split] = adapter
        human_manifest["accepted_pairs"] = len(human_gold)

    reviewed_ids = {record["pair_id"] for record in human_gold}
    diagnostics = [
        record for record in diagnostics if record["pair_id"] not in reviewed_ids
    ]
    review_candidates = {
        split: [
            record for record in records if record["pair_id"] not in reviewed_ids
        ]
        for split, records in review_candidates.items()
    }
    records = [*ase_records, *gui_train, *diagnostics, *human_gold]
    manifest = write_mapping_dataset(
        records,
        args.output_dir,
        review_candidates=review_candidates,
        metadata={
            "split_seed": args.split_seed,
            "inputs": {
                "ase_queries": str(args.ase_queries.expanduser().resolve()),
                "gui_pairs": str(args.gui_pairs.expanduser().resolve()),
                "gui_annotations": str(args.gui_annotations.expanduser().resolve()),
                "gui_screenshots": _resolved_or_none(args.gui_screenshots),
                "gui_human_review": _resolved_or_none(args.gui_human_review),
                "human_screenshots": _resolved_or_none(args.human_screenshots),
            },
            "protocol": {
                "record_schema_identical_across_splits": True,
                "ase": "public authored app-disjoint split",
                "guiodyssey": gui_split.audit,
                "weak_labels": "train or diagnostic only",
                "formal_dev_test": "gold labels only",
            },
            "adapters": {
                "guiodyssey_train": train_adapter,
                "guiodyssey_heldout": heldout_adapters,
                "guiodyssey_human_gold": human_manifest,
            },
        },
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _review_candidate(
    record: dict[str, Any],
    *,
    reserved_split: str,
) -> dict[str, Any]:
    candidate = deepcopy(record)
    candidate["label_status"] = "unreviewed"
    candidate["provenance"] = {
        **candidate["provenance"],
        "annotation": "trajectory_alignment_review_candidate",
        "reserved_split": reserved_split,
    }
    return validate_mapping_page_pair(candidate)


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {index} is not an object: {resolved}")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"JSONL is empty: {resolved}")
    return rows


def _resolved_or_none(path: Path | None) -> str | None:
    return str(path.expanduser().resolve()) if path is not None else None


if __name__ == "__main__":
    main()
