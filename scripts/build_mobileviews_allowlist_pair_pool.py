#!/usr/bin/env python3
"""Build an exhaustive MobileViews pair pool for a frozen App allowlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_dataset import write_mapping_dataset
from omnitransfer.mobileviews_multi_app_pairs import (
    build_mobileviews_allowlist_pair_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--app-allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-limit-per-app", type=int, default=1000)
    parser.add_argument("--minimum-matches", type=int, default=2)
    parser.add_argument("--candidate-pairs-per-structure", type=int, default=2000)
    parser.add_argument(
        "--split",
        choices=("train", "diagnostic"),
        default="diagnostic",
    )
    parser.add_argument(
        "--label-status",
        choices=("self_supervised", "unreviewed"),
        default="unreviewed",
    )
    parser.add_argument("--reserved-split", choices=("dev", "test"))
    args = parser.parse_args()
    app_names = [
        line.strip()
        for line in args.app_allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records, pool = build_mobileviews_allowlist_pair_pool(
        args.traces_root,
        app_names,
        pair_limit_per_app=args.pair_limit_per_app,
        minimum_matches=args.minimum_matches,
        candidate_pairs_per_structure=args.candidate_pairs_per_structure,
        split=args.split,
        label_status=args.label_status,
        reserved_split=args.reserved_split,
    )
    if not records:
        raise SystemExit("No allowlisted MobileViews pairs were found")
    manifest = write_mapping_dataset(records, args.output_dir, metadata={"pool": pool})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
