#!/usr/bin/env python3
"""Train the OmniTransfer node-correspondence matcher."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from omnitransfer.experiment_logging import (
    TrainingMetricLog,
    file_sha256,
    runtime_environment,
    source_revision,
)
from omnitransfer.learned_matcher import (
    MatcherConfig,
    OmniTransferMatcher,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.mapping_training import load_ui_correspondence_pairs
from omnitransfer.self_supervised import (
    AugmentConfig,
    DEFAULT_CONTEXT_MASK_PROBABILITY,
    evaluate_correspondence_pairs,
    train_omnitransfer_matcher,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--validation-input",
        nargs="+",
        type=Path,
        default=(),
        help=(
            "Held-out dev UI-correspondence JSONL files with the same record "
            "schema; label_status distinguishes development pseudo-labels "
            "from formal gold."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--assignment-head",
        choices=("pair_mlp", "mutual_projection"),
    )
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--target-context-nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--context-mask-probability",
        type=float,
        default=DEFAULT_CONTEXT_MASK_PROBABILITY,
    )
    parser.add_argument(
        "--visual-dropout-probability",
        type=float,
        default=AugmentConfig.visual_dropout_prob,
        help=(
            "Probability of hiding each node crop in an augmented view; "
            "the XML/context path remains unchanged."
        ),
    )
    parser.add_argument("--metrics-log", type=Path)
    parser.add_argument("--minimum-correspondences", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--allow-unreviewed-pseudo", action="store_true")
    parser.add_argument("--screenshot-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pretrained_model = None
    if args.pretrained is not None:
        pretrained = OmniTransferMatcher.from_checkpoint(
            args.pretrained,
            device=args.device,
        )
        config = replace(
            pretrained.config,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
        )
        if (
            args.assignment_head is not None
            and args.assignment_head != config.assignment_head
        ):
            raise SystemExit(
                "--assignment-head cannot change a pretrained checkpoint architecture"
            )
        pretrained_model = pretrained.model
    else:
        config = MatcherConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
            assignment_head=args.assignment_head or "pair_mlp",
        )
    pairs, graphs, adapter = load_ui_correspondence_pairs(
        args.input,
        matcher_config=config,
        allow_unreviewed_pseudo=args.allow_unreviewed_pseudo,
        minimum_correspondences=args.minimum_correspondences,
        max_pairs=args.max_pairs,
        screenshot_root=args.screenshot_root,
    )
    if not pairs:
        raise SystemExit("No actionable correspondence pairs were accepted")
    validation_pairs = []
    validation_adapter = None
    if args.validation_input:
        validation_pairs, _, validation_adapter = load_ui_correspondence_pairs(
            args.validation_input,
            matcher_config=config,
            minimum_correspondences=1,
            screenshot_root=args.screenshot_root,
            allowed_splits=frozenset({"dev"}),
        )
        if not validation_pairs:
            raise SystemExit("No held-out dev correspondence pairs were accepted")
    preview = {
        "adapter": adapter,
        "matcher_config": asdict(config),
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "training": "one_shuffled_correspondence_optimizer",
        "context_mask_probability": args.context_mask_probability,
        "validation_adapter": validation_adapter,
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if not 0.0 <= args.visual_dropout_probability <= 1.0:
        raise SystemExit("--visual-dropout-probability must be in [0, 1]")
    augment = AugmentConfig(
        visual_dropout_prob=args.visual_dropout_probability,
    )
    input_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for path in args.input
    ]
    validation_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for path in args.validation_input
    ]
    revision = source_revision()
    command = [str(value) for value in sys.argv]
    metrics_path = args.metrics_log or args.output.with_suffix(".metrics.jsonl")
    metric_log = TrainingMetricLog(metrics_path)
    metric_log.start(
        {
            "architecture": "omnitransfer",
            "objective": "symmetric_actionable_correspondence",
            "code_revision": revision,
            "command": command,
            "inputs": input_artifacts,
            "validation_inputs": validation_artifacts,
            "matcher_config": asdict(config),
            "augmentation": asdict(augment),
            "context_mask_probability": args.context_mask_probability,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "runtime": runtime_environment(args.device),
        }
    )

    def report_progress(metrics: dict[str, float]) -> None:
        metric_log.append("training_progress", metrics)
        print(
            json.dumps({"event": "training_progress", **metrics}),
            flush=True,
        )

    model, history = train_omnitransfer_matcher(
        graphs,
        pairs,
        model=pretrained_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        augment_config=augment,
        matcher_config=config,
        context_mask_probability=args.context_mask_probability,
        progress_callback=report_progress,
        progress_interval=args.progress_interval,
        validation_pairs=validation_pairs,
    )
    evaluation_pairs = validation_pairs or pairs
    evaluation_split = "dev" if validation_pairs else "training_fit_smoke"
    metrics = evaluate_correspondence_pairs(
        model,
        evaluation_pairs,
        device=args.device,
        matcher_config=config,
    )
    report = {
        "schema_version": "omnitransfer.matcher_experiment.v1",
        "architecture": "omnitransfer",
        "objective": "symmetric_actionable_correspondence",
        "code_revision": revision,
        "command": command,
        "inputs": input_artifacts,
        "validation_inputs": validation_artifacts,
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "adapter": adapter,
        "validation_adapter": validation_adapter,
        "matcher_config": asdict(config),
        "parameter_count": parameter_count(model),
        "augmentation": asdict(augment),
        "context_mask_probability": args.context_mask_probability,
        "metrics_log": str(metrics_path.resolve()),
        "training": {"history": history},
        "evaluation": {
            "split": evaluation_split,
            "metrics": metrics,
        },
        "evaluation_boundary": "formal metrics require held-out human gold",
    }
    checkpoint_metadata = {
        key: value for key, value in report.items() if key != "evaluation"
    }
    save_matcher_checkpoint(
        args.output,
        model,
        config=config,
        metadata=checkpoint_metadata,
    )
    checkpoint_artifact = {
        "path": str(args.output.resolve()),
        "sha256": file_sha256(args.output),
        "parameter_count": parameter_count(model),
    }
    report["checkpoint"] = checkpoint_artifact
    for row in history:
        metric_log.append("epoch_end", row)
    metric_log.append(
        "evaluation",
        {
            "split": evaluation_split,
            "top1_accuracy": metrics["top1_accuracy"],
            "recall_at_k": metrics["recall_at_k"],
            "warm_model_latency_ms": metrics["warm_model_latency_ms"],
            "warm_end_to_end_latency_ms": metrics["warm_end_to_end_latency_ms"],
        },
    )
    metric_log.append(
        "checkpoint",
        checkpoint_artifact,
    )
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "accepted_pairs": len(pairs),
                "evaluation_split": evaluation_split,
                "top1": metrics["top1_accuracy"],
                "warm_p95_ms": metrics["warm_model_latency_ms"]["p95"],
                "warm_end_to_end_p95_ms": metrics["warm_end_to_end_latency_ms"]["p95"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
