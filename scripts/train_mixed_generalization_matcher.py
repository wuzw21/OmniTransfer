#!/usr/bin/env python3
"""Train one matcher on ASE gold and GUIOdyssey weak correspondences."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from omnitransfer.benchmark import (
    benchmark_image_latency,
    evaluate_slices,
    fine_tune_matcher,
    partition_queries_by_declared_split,
    predict_queries,
)
from omnitransfer.generalization import split_gui_odyssey_pair_rows
from omnitransfer.gui_odyssey_pairs import (
    load_gui_odyssey_human_review_pairs,
    make_gui_odyssey_training_pairs,
)
from omnitransfer.importers import load_queries
from omnitransfer.learned_matcher import (
    LearnedGraphMatcher,
    MatcherConfig,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.self_supervised import TrainingPair, evaluate_correspondence_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-queries", type=Path, required=True)
    parser.add_argument("--gui-pairs", type=Path, required=True)
    parser.add_argument("--gui-annotations", type=Path, required=True)
    parser.add_argument("--gui-screenshots", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--human-screenshots", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--anchor-baseline-report", type=Path)
    parser.add_argument("--selector-baseline-report", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--correspondence-weight", type=float, default=0.35)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--target-context-nodes", type=int, default=64)
    parser.add_argument("--limit-ase-train", type=int, default=0)
    parser.add_argument("--limit-ase-eval", type=int, default=0)
    parser.add_argument("--limit-gui-train", type=int, default=0)
    parser.add_argument("--limit-gui-eval", type=int, default=0)
    parser.add_argument("--latency-images", type=int, default=100)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.correspondence_weight < 0.0:
        raise SystemExit("--correspondence-weight must be non-negative")

    ase_splits = partition_queries_by_declared_split(load_queries(args.ase_queries))
    ase_train = _limited(ase_splits["train"], args.limit_ase_train)
    ase_test = _limited(ase_splits["test"], args.limit_ase_eval)
    gui_rows = _load_jsonl(args.gui_pairs)
    gui_split = split_gui_odyssey_pair_rows(gui_rows, seed=args.split_seed)

    if args.pretrained:
        loaded = LearnedGraphMatcher.from_checkpoint(args.pretrained, device=args.device)
        model = loaded.model
        config = loaded.config
    else:
        model = None
        config = MatcherConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
        )

    gui_train, gui_train_manifest = make_gui_odyssey_training_pairs(
        _limited(gui_split.splits["train"], args.limit_gui_train),
        args.gui_annotations,
        screenshot_dir=args.gui_screenshots,
        matcher_config=config,
    )
    gui_test, gui_test_manifest = make_gui_odyssey_training_pairs(
        _limited(gui_split.splits["test"], args.limit_gui_eval),
        args.gui_annotations,
        screenshot_dir=args.gui_screenshots,
        matcher_config=config,
    )
    if not ase_train or not ase_test or not gui_train or not gui_test:
        raise SystemExit("mixed-data train/test inputs must all be non-empty")

    result = fine_tune_matcher(
        ase_train,
        model=model,
        config=config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        correspondence_pairs=gui_train,
        correspondence_weight=args.correspondence_weight,
    )
    ase_predictions = predict_queries(
        result.model,
        ase_test,
        config=config,
        device=args.device,
    )
    ase_metrics = {
        name: asdict(metrics)
        for name, metrics in evaluate_slices(ase_test, ase_predictions).items()
    }
    image_latency = benchmark_image_latency(
        result.model,
        ase_test,
        config=config,
        device=args.device,
        budget_ms=args.latency_budget_ms,
        max_images=args.latency_images,
    )
    weak_metrics = evaluate_correspondence_pairs(
        result.model,
        gui_test,
        device=args.device,
        matcher_config=config,
    )
    weak_device_slices = {
        name: evaluate_correspondence_pairs(
            result.model,
            pairs,
            device=args.device,
            matcher_config=config,
            latency_repeats=0,
        )
        for name, pairs in _device_pair_slices(gui_test).items()
    }
    human_gold = _evaluate_human_gold(
        result.model,
        args.human_review,
        screenshot_dir=args.human_screenshots,
        config=config,
        device=args.device,
    )
    report = {
        "schema_version": "omnitransfer_mixed_generalization_experiment_v1",
        "method": {
            "name": "relation_aware_bidirectional_cross_attention_matcher",
            "training_seam": "single_model_single_optimizer_assignment_losses",
            "coordinate_passthrough": False,
            "rule_or_selector_bypass": False,
            "matcher_config": asdict(config),
            "parameter_count": parameter_count(result.model),
        },
        "inputs": {
            "ase_queries": str(args.ase_queries.resolve()),
            "gui_pairs": str(args.gui_pairs.resolve()),
            "gui_annotations": str(args.gui_annotations.resolve()),
            "gui_screenshots": (
                str(args.gui_screenshots.resolve()) if args.gui_screenshots else None
            ),
            "human_review": (
                str(args.human_review.resolve()) if args.human_review else None
            ),
        },
        "protocol": {
            "ase": "dataset_authored_app_disjoint_train_dev_test",
            "guiodyssey": gui_split.audit,
            "formal_guiodyssey_gold_source": "exported_human_review_only",
        },
        "counts": {
            "ase_declared_splits": {name: len(rows) for name, rows in ase_splits.items()},
            "ase_train_used": len(ase_train),
            "ase_test_used": len(ase_test),
            "gui_train_used": len(gui_train),
            "gui_test_used": len(gui_test),
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "correspondence_weight": args.correspondence_weight,
            "seed": args.seed,
            "history": list(result.history),
            "gui_train_adapter": gui_train_manifest,
        },
        "evaluation": {
            "ase_2023_public_gold": {
                "label_status": "formal_gold",
                "metrics": ase_metrics,
            },
            "guiodyssey_heldout_trajectory": {
                "label_status": "weak_pseudo_label_not_formal_gold",
                "adapter": gui_test_manifest,
                "metrics": weak_metrics,
                "device_pair_slices": weak_device_slices,
            },
            "guiodyssey_human_review": human_gold,
            "warm_image_latency": asdict(image_latency),
            "baselines": {
                "rule_matcher": _load_baseline(args.anchor_baseline_report),
                "selector": _load_baseline(args.selector_baseline_report),
            },
        },
    }
    save_matcher_checkpoint(
        args.checkpoint,
        result.model,
        config=config,
        metadata=report,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "report": str(args.report),
                "ase_test_top1": ase_metrics["all"]["top1_accuracy"],
                "gui_weak_top1": weak_metrics["top1_accuracy"],
                "gui_human_status": human_gold["status"],
                "warm_latency_p95_ms": image_latency.p95_latency_ms,
            },
            ensure_ascii=False,
        )
    )


def _evaluate_human_gold(
    model: Any,
    review_path: Path | None,
    *,
    screenshot_dir: Path | None,
    config: MatcherConfig,
    device: str,
) -> dict[str, Any]:
    if review_path is None or not review_path.is_file():
        return {
            "status": "unavailable",
            "label_status": "formal_human_gold",
            "accepted_pairs": 0,
            "reason": "no exported reviewed JSONL was provided",
        }
    pairs, manifest = load_gui_odyssey_human_review_pairs(
        review_path,
        screenshot_dir=screenshot_dir,
        matcher_config=config,
    )
    if not pairs:
        return {
            "status": "unavailable",
            "label_status": "formal_human_gold",
            "accepted_pairs": 0,
            "reason": "export contains no reviewed correspondence/no_correspondence rows",
            "manifest": manifest,
        }
    return {
        "status": "evaluated",
        "label_status": "formal_human_gold",
        "accepted_pairs": len(pairs),
        "manifest": manifest,
        "metrics": evaluate_correspondence_pairs(
            model,
            pairs,
            device=device,
            matcher_config=config,
        ),
    }


def _device_pair_slices(
    pairs: list[TrainingPair],
) -> dict[str, list[TrainingPair]]:
    grouped: dict[str, list[TrainingPair]] = defaultdict(list)
    for pair in pairs:
        devices = sorted(
            (
                str(pair.graph_a.metadata.get("device_name") or "unknown"),
                str(pair.graph_b.metadata.get("device_name") or "unknown"),
            )
        )
        grouped[" | ".join(devices)].append(pair)
    return dict(sorted(grouped.items()))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _limited(values: Any, limit: int) -> list[Any]:
    rows = list(values)
    return rows[:limit] if limit > 0 else rows


def _load_baseline(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "unavailable"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "loaded",
        "path": str(path.resolve()),
        "summary": payload.get("summary", payload),
    }


if __name__ == "__main__":
    main()
