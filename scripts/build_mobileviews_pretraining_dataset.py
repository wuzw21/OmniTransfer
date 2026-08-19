#!/usr/bin/env python3
"""Build a balanced MobileViews corpus for canonical two-view pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_dataset import write_ui_correspondence_dataset
from omnitransfer.mobileviews_pretraining import (
    mobileviews_self_supervised_pair,
    mobileviews_state_transition_pairs,
    select_mobileviews_pages,
)


def _records(path: Path, *, path_prefix_from: str, path_prefix_to: str):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                screenshot = str(metadata.get("screenshot_path") or "")
                if path_prefix_from and screenshot.startswith(path_prefix_from):
                    metadata["screenshot_path"] = path_prefix_to + screenshot[len(path_prefix_from) :]
                    record["metadata"] = metadata
                yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-pages", type=int, default=5000)
    parser.add_argument(
        "--mode",
        choices=("state-transition", "same-observation"),
        default="state-transition",
    )
    parser.add_argument("--per-package-cap", type=int, default=4)
    parser.add_argument("--maximum-row-gap", type=int, default=4)
    parser.add_argument("--minimum-matches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--screenshot-path-prefix-from", default="")
    parser.add_argument("--screenshot-path-prefix-to", default="")
    args = parser.parse_args()
    records = _records(
        args.input,
        path_prefix_from=args.screenshot_path_prefix_from,
        path_prefix_to=args.screenshot_path_prefix_to,
    )
    if args.mode == "state-transition":
        pairs, selection = mobileviews_state_transition_pairs(
            records,
            maximum_pairs=args.maximum_pages,
            per_package_cap=args.per_package_cap,
            maximum_row_gap=args.maximum_row_gap,
            minimum_matches=args.minimum_matches,
        )
    else:
        selected, selection = select_mobileviews_pages(
            records,
            maximum_pages=args.maximum_pages,
            per_package_cap=args.per_package_cap,
            seed=args.seed,
        )
        pairs = (mobileviews_self_supervised_pair(graph) for graph in selected)
    manifest = write_ui_correspondence_dataset(
        pairs,
        args.output_dir,
        metadata={"mode": args.mode, "selection": selection},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
