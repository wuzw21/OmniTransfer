#!/usr/bin/env python3
"""Evaluate a frozen explicit-evidence mutual matcher without updating it."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from omnitransfer.benchmark import (
    benchmark_image_latency,
    evaluate_slices,
    predict_queries,
    split_queries_by_group,
)
from omnitransfer.importers import load_queries
from omnitransfer.learned_matcher import (
    PEMM_V3_FEATURE_SCHEMA_ID,
    PEMM_V3_FEATURE_SCHEMA_SHA256,
    parameter_count,
)
from omnitransfer.mutual_matcher import MutualGraphMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--eval-split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--min-probability", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    parser.add_argument("--latency-images", type=int, default=100)
    parser.add_argument("--fail-on-latency", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.input)
    splits = split_queries_by_group(queries, seed=args.split_seed)
    eval_queries = splits[args.eval_split][: args.limit_eval or None]
    if not eval_queries:
        raise SystemExit(f"App/group split produced an empty {args.eval_split} partition.")
    matcher = MutualGraphMatcher.from_checkpoint(args.checkpoint, device=args.device)
    image_latency = benchmark_image_latency(
        matcher.model,
        eval_queries,
        config=matcher.config,
        device=args.device,
        budget_ms=args.latency_budget_ms,
        max_images=args.latency_images,
        feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
    )
    predictions = predict_queries(
        matcher.model,
        eval_queries,
        config=matcher.config,
        device=args.device,
        min_probability=args.min_probability,
        min_margin=args.min_margin,
        feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
    )
    metrics = {
        name: asdict(summary)
        for name, summary in evaluate_slices(eval_queries, predictions).items()
    }
    report = {
        "schema_version": "omnitransfer_frozen_mutual_matcher_evaluation_v3",
        "architecture": "explicit_pair_evidence_mutual_assignment_no_null",
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_seed": args.split_seed,
        "eval_split": args.eval_split,
        "eval_queries": len(eval_queries),
        "matcher_config": asdict(matcher.config),
        "parameter_count": parameter_count(matcher.model),
        "feature_schema_id": PEMM_V3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": PEMM_V3_FEATURE_SCHEMA_SHA256,
        "thresholds": {
            "min_probability": args.min_probability,
            "min_margin": args.min_margin,
        },
        "metrics": metrics,
        "new_image_latency": asdict(image_latency),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(
            "".join(
                json.dumps(
                    {
                        "query_id": prediction.query_id,
                        "selected_candidate_id": prediction.selected_candidate_id,
                        "scores": prediction.scores,
                        "metadata": prediction.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for prediction in predictions
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {"metrics": metrics["all"], "new_image_latency": asdict(image_latency)},
            ensure_ascii=False,
        )
    )
    if args.fail_on_latency and not image_latency.budget_met:
        raise SystemExit(
            f"new-image latency gate failed: max={image_latency.max_latency_ms:.2f}ms "
            f">= {image_latency.budget_ms:.2f}ms"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
