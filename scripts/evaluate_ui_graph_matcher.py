#!/usr/bin/env python3
"""Evaluate a matcher on the canonical mapping page-pair schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.learned_matcher import LearnedGraphMatcher, MatcherConfig, parameter_count
from omnitransfer.mapping_training import load_mapping_training_pairs
from omnitransfer.self_supervised import TrainingPair, evaluate_correspondence_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test", "diagnostic"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument("--allow-unreviewed-pseudo", action="store_true")
    parser.add_argument("--screenshot-root", type=Path)
    args = parser.parse_args()

    matcher = LearnedGraphMatcher.from_checkpoint(args.checkpoint, device=args.device)
    pairs, adapter = load_evaluation_pairs(
        args.input,
        split=args.split,
        config=matcher.config,
        max_pairs=args.max_pairs,
        allow_unreviewed_pseudo=args.allow_unreviewed_pseudo,
        screenshot_root=args.screenshot_root,
    )
    metrics = evaluate_correspondence_pairs(
        matcher.model,
        pairs,
        device=args.device,
        matcher_config=matcher.config,
        latency_repeats=args.latency_repeats,
    )
    report = {
        "schema_version": "omnitransfer.page_pair_evaluation.v1",
        "record_schema": "omnitransfer.mapping_page_pair.v1",
        "split": args.split,
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in args.input
        ],
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": file_sha256(args.checkpoint),
            "parameter_count": parameter_count(matcher.model),
        },
        "device": args.device,
        "adapter": adapter,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False))


def load_evaluation_pairs(
    inputs: list[Path],
    *,
    split: str,
    config: MatcherConfig,
    max_pairs: int = 0,
    allow_unreviewed_pseudo: bool = False,
    screenshot_root: Path | None = None,
) -> tuple[list[TrainingPair], dict[str, Any]]:
    """Use the training adapter unchanged, constrained to one evaluation split."""

    pairs, _, adapter = load_mapping_training_pairs(
        inputs,
        matcher_config=config,
        allow_unreviewed_pseudo=allow_unreviewed_pseudo,
        minimum_correspondences=1,
        max_pairs=max_pairs,
        screenshot_root=screenshot_root,
        allowed_splits=frozenset({split}),
    )
    if not pairs:
        raise SystemExit(f"No actionable {split} page pairs were accepted")
    return pairs, adapter


if __name__ == "__main__":
    main()
