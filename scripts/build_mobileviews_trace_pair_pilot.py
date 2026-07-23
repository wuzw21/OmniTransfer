#!/usr/bin/env python3
"""Build an unreviewed MobileViews complete-trace page-pair pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_dataset import write_mapping_dataset
from omnitransfer.mobileviews_trace_pairs import build_mobileviews_trace_pair_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=20)
    parser.add_argument("--minimum-matches", type=int, default=2)
    args = parser.parse_args()

    records, pilot = build_mobileviews_trace_pair_pilot(
        args.trace_dir,
        pair_limit=args.pair_limit,
        minimum_matches=args.minimum_matches,
    )
    manifest = write_mapping_dataset(
        records,
        args.output_dir,
        metadata={"pilot": pilot},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
