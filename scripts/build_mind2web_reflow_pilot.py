#!/usr/bin/env python3
"""Capture one Mind2Web HTML/MHTML asset as same-DOM reflow pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_dataset import write_mapping_dataset
from omnitransfer.mind2web_reflow import ReflowViewport, capture_mind2web_reflow_pair


def _viewport(value: str) -> ReflowViewport:
    try:
        name, dimensions = value.split(":", 1)
        width, height = dimensions.lower().split("x", 1)
        return ReflowViewport(name, int(width), int(height))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("viewport must be NAME:WIDTHxHEIGHT") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=_viewport, default=_viewport("desktop:1280x900"))
    parser.add_argument(
        "--target",
        type=_viewport,
        action="append",
        default=None,
        help="repeat for multiple targets, e.g. phone:390x844",
    )
    parser.add_argument("--task-id", default="local-smoke")
    parser.add_argument("--action-id", default="page")
    args = parser.parse_args()

    targets = args.target or [_viewport("phone:390x844")]
    records = []
    stats = []
    for target in targets:
        record, pair_stats = capture_mind2web_reflow_pair(
            args.page,
            args.output_dir / "screenshots",
            source_viewport=args.source,
            target_viewport=target,
            task_id=args.task_id,
            action_id=args.action_id,
        )
        records.append(record)
        stats.append({"target": target.name, **pair_stats})
    manifest = write_mapping_dataset(
        records,
        args.output_dir,
        metadata={
            "pilot": {
                "schema_version": "omnitransfer.mind2web_reflow_pilot.v1",
                "pairs": stats,
                "promotion_requirement": "raw MHTML/DOMSnapshot plus source-render validation and human review",
            }
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
