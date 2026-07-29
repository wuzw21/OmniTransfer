#!/usr/bin/env python3
"""Benchmark decoded-image compute for a learned OmniTransfer checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from omnitransfer.benchmark import (
    benchmark_image_latency,
    partition_queries_by_declared_split,
)
from omnitransfer.importers import load_queries
from omnitransfer.learned_matcher import RelationAwareMatcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("all", "train", "dev", "test"), default="test")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--budget-ms", type=float, default=50.0)
    parser.add_argument("--fail-on-budget", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.input)
    if args.split != "all":
        queries = partition_queries_by_declared_split(queries)[args.split]
    matcher = RelationAwareMatcher.from_checkpoint(args.checkpoint, device=args.device)
    summary = benchmark_image_latency(
        matcher.model,
        queries,
        config=matcher.config,
        device=args.device,
        budget_ms=args.budget_ms,
        max_images=args.max_images,
    )
    payload = {
        "schema_version": "omnitransfer_decoded_image_latency_v1",
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "split": args.split,
        "latency": asdict(summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["latency"], ensure_ascii=False))
    if args.fail_on_budget and not summary.budget_met:
        raise SystemExit(
            f"latency budget failed: max={summary.max_latency_ms:.2f}ms "
            f">= {summary.budget_ms:.2f}ms"
        )


if __name__ == "__main__":
    main()
