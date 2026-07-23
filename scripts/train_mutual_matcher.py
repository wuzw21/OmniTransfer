#!/usr/bin/env python3
"""Train the explicit-evidence mutual UI matcher on a frozen query split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from omnitransfer.benchmark import (
    benchmark_image_latency,
    evaluate_slices,
    fine_tune_matcher,
    predict_queries,
    split_queries_by_group,
)
from omnitransfer.importers import load_queries
from omnitransfer.learned_matcher import MatcherConfig, parameter_count
from omnitransfer.mutual_matcher import (
    build_mutual_assignment_matcher,
    save_mutual_matcher_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--target-context-nodes", type=int, default=64)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    parser.add_argument("--latency-images", type=int, default=100)
    parser.add_argument("--fail-on-latency", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.input)
    splits = split_queries_by_group(queries, seed=args.split_seed)
    train_queries = splits["train"][: args.limit_train or None]
    eval_queries = splits[args.eval_split][: args.limit_eval or None]
    if not train_queries or not eval_queries:
        raise SystemExit(
            f"App/group split produced an empty train or {args.eval_split} partition."
        )
    config = MatcherConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        source_context_nodes=args.source_context_nodes,
        target_context_nodes=args.target_context_nodes,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_mutual_assignment_matcher(config)
    result = fine_tune_matcher(
        train_queries,
        model=model,
        config=config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    image_latency = benchmark_image_latency(
        result.model,
        eval_queries,
        config=config,
        device=args.device,
        budget_ms=args.latency_budget_ms,
        max_images=args.latency_images,
    )
    predictions = predict_queries(
        result.model,
        eval_queries,
        config=config,
        device=args.device,
        min_probability=args.min_probability,
        min_margin=args.min_margin,
    )
    metrics = {
        name: asdict(summary)
        for name, summary in evaluate_slices(eval_queries, predictions).items()
    }
    report = {
        "schema_version": "omnitransfer_mutual_matcher_experiment_v2",
        "architecture": "explicit_pair_evidence_mutual_assignment",
        "input": str(args.input.resolve()),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "eval_split": args.eval_split,
        "train_queries": len(train_queries),
        "eval_queries": len(eval_queries),
        "matcher_config": asdict(config),
        "parameter_count": parameter_count(result.model),
        "history": list(result.history),
        "thresholds": {
            "min_probability": args.min_probability,
            "min_margin": args.min_margin,
        },
        "metrics": metrics,
        "new_image_latency": asdict(image_latency),
    }
    save_mutual_matcher_checkpoint(
        args.output,
        result.model,
        config=config,
        metadata=report,
    )
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
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


if __name__ == "__main__":
    main()
