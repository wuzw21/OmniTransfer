#!/usr/bin/env python3
"""Train the canonical OmniTransfer geometric-v9 matcher."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from omnitransfer.experiment_logging import (
    TrainingMetricLog,
    file_sha256,
    runtime_environment,
    source_revision,
)
from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    NODE_DESCRIPTOR_DIM,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    MatcherConfig,
    GeometricMatcher,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.mapping_training import load_ui_correspondence_pairs
from omnitransfer.self_supervised import (
    AugmentConfig,
    evaluate_correspondence_pairs,
    train_geometric_v9_matcher,
)


def scaled_augment_config(
    strength: float,
    *,
    visual_dropout_probability: float,
) -> AugmentConfig:
    """Scale the canonical two-view perturbation family with one knob."""

    base = AugmentConfig()

    def probability(value: float) -> float:
        return min(1.0, value * strength)

    return replace(
        base,
        mask_text_prob=probability(base.mask_text_prob),
        mask_content_desc_prob=probability(base.mask_content_desc_prob),
        mask_class_prob=probability(base.mask_class_prob),
        drop_node_prob=probability(base.drop_node_prob),
        edge_dropout_prob=probability(base.edge_dropout_prob),
        bbox_jitter=base.bbox_jitter * strength,
        global_translation=base.global_translation * strength,
        global_scale=base.global_scale * strength,
        visual_brightness=base.visual_brightness * strength,
        visual_contrast=base.visual_contrast * strength,
        visual_channel_scale=base.visual_channel_scale * strength,
        visual_dropout_prob=probability(visual_dropout_probability),
        distractor_prob=probability(base.distractor_prob),
        max_distractors=round(base.max_distractors * strength),
        shuffle_nodes=strength > 0.0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--validation-input", nargs="+", type=Path, default=())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Resume an exact geometric-v9 checkpoint without architecture conversion.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=NODE_DESCRIPTOR_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--target-context-nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--descriptor-loss-weight", type=float, default=0.20)
    parser.add_argument("--descriptor-learning-rate-scale", type=float, default=0.25)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--visual-dropout-probability",
        type=float,
        default=AugmentConfig.visual_dropout_prob,
    )
    parser.add_argument("--augmentation-strength", type=float, default=1.0)
    parser.add_argument("--metrics-log", type=Path)
    parser.add_argument("--minimum-correspondences", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--synthetic-pairs-per-page", type=int, default=8)
    parser.add_argument("--cross-page-pair-repeats", type=int, default=1)
    parser.add_argument("--allow-unreviewed-pseudo", action="store_true")
    parser.add_argument("--screenshot-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    non_negative = {
        "--synthetic-pairs-per-page": args.synthetic_pairs_per_page,
        "--cross-page-pair-repeats": args.cross_page_pair_repeats,
        "--descriptor-loss-weight": args.descriptor_loss_weight,
    }
    for name, value in non_negative.items():
        if value < 0:
            raise SystemExit(f"{name} must be non-negative")
    if args.synthetic_pairs_per_page == 0 and args.cross_page_pair_repeats == 0:
        raise SystemExit("training requires synthetic or cross-page pairs")
    if args.num_layers <= 0:
        raise SystemExit("--num-layers must be positive")
    if not 0.0 < args.descriptor_learning_rate_scale <= 1.0:
        raise SystemExit("--descriptor-learning-rate-scale must be in (0, 1]")
    for name, value in {
        "--visual-dropout-probability": args.visual_dropout_probability,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    if not 0.0 <= args.augmentation_strength <= 2.0:
        raise SystemExit("--augmentation-strength must be in [0, 2]")


def _load_geometric_checkpoint(path: Path, *, device: str) -> GeometricMatcher:
    matcher = GeometricMatcher.from_checkpoint(path, device=device)
    if matcher.config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise SystemExit("only geometric-v9 checkpoints are supported")
    if matcher.config.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise SystemExit("training resumes only all-node geometric-v9 checkpoints")
    return matcher


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("training requires PyTorch") from exc
    torch.manual_seed(args.seed)

    if args.pretrained is None:
        config = MatcherConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
            architecture=OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
            candidate_policy=ALL_NODE_CANDIDATE_POLICY,
        )
        pretrained_model = None
        pretrained_mode = None
    else:
        pretrained = _load_geometric_checkpoint(args.pretrained, device=args.device)
        config = pretrained.config
        pretrained_model = pretrained.model
        pretrained_mode = "exact_geometric_v9_resume"

    augment = scaled_augment_config(
        args.augmentation_strength,
        visual_dropout_probability=args.visual_dropout_probability,
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
        raise SystemExit("No correspondence pairs were accepted")
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
        "validation_adapter": validation_adapter,
        "matcher_config": asdict(config),
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "pretrained_mode": pretrained_mode,
        "augmentation": asdict(augment),
        "synthetic_pairs_per_page": args.synthetic_pairs_per_page,
        "cross_page_pair_repeats": args.cross_page_pair_repeats,
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    input_artifacts = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in args.input
    ]
    validation_artifacts = [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in args.validation_input
    ]
    revision = source_revision()
    command = [str(value) for value in sys.argv]
    metrics_path = args.metrics_log or args.output.with_suffix(".metrics.jsonl")
    metric_log = TrainingMetricLog(metrics_path)
    metric_log.start(
        {
            "architecture": config.architecture,
            "objective": "symmetric_all_node_correspondence",
            "code_revision": revision,
            "command": command,
            "inputs": input_artifacts,
            "validation_inputs": validation_artifacts,
            "matcher_config": asdict(config),
            "pretrained_mode": pretrained_mode,
            "augmentation": asdict(augment),
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "runtime": runtime_environment(args.device),
        }
    )

    def report_progress(metrics: dict[str, float]) -> None:
        metric_log.append("training_progress", metrics)
        print(json.dumps({"event": "training_progress", **metrics}), flush=True)

    def report_epoch(metrics: dict[str, float]) -> None:
        metric_log.append("epoch_end", metrics)
        print(json.dumps({"event": "epoch_end", **metrics}), flush=True)

    latest_checkpoint = args.output.with_name(
        f"{args.output.stem}.latest{args.output.suffix}"
    )

    def save_latest_checkpoint(current_model: object, metrics: dict[str, float]) -> None:
        temporary = latest_checkpoint.with_suffix(f"{latest_checkpoint.suffix}.tmp")
        save_matcher_checkpoint(
            temporary,
            current_model,
            config=config,
            metadata={
                "schema_version": "omnitransfer.training_latest.v1",
                "objective": "symmetric_all_node_correspondence",
                "code_revision": revision,
                "matcher_config": asdict(config),
                "training_progress": metrics,
                "inputs": input_artifacts,
                "validation_inputs": validation_artifacts,
            },
        )
        temporary.replace(latest_checkpoint)

    model, history = train_geometric_v9_matcher(
        graphs,
        pairs,
        model=pretrained_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        augment_config=augment,
        matcher_config=config,
        progress_callback=report_progress,
        epoch_callback=report_epoch,
        checkpoint_callback=save_latest_checkpoint,
        progress_interval=args.progress_interval,
        validation_pairs=validation_pairs,
        synthetic_pairs_per_graph=args.synthetic_pairs_per_page,
        cross_page_pair_repeats=args.cross_page_pair_repeats,
        descriptor_weight=args.descriptor_loss_weight,
        descriptor_learning_rate_scale=args.descriptor_learning_rate_scale,
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
        "architecture": config.architecture,
        "objective": "symmetric_all_node_correspondence",
        "code_revision": revision,
        "command": command,
        "inputs": input_artifacts,
        "validation_inputs": validation_artifacts,
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "pretrained_mode": pretrained_mode,
        "adapter": adapter,
        "validation_adapter": validation_adapter,
        "matcher_config": asdict(config),
        "parameter_count": parameter_count(model),
        "augmentation": asdict(augment),
        "training": {"history": history},
        "evaluation": {"split": evaluation_split, "metrics": metrics},
        "evaluation_boundary": "formal metrics require held-out human gold",
    }
    save_matcher_checkpoint(
        args.output,
        model,
        config=config,
        metadata={key: value for key, value in report.items() if key != "evaluation"},
    )
    checkpoint_artifact = {
        "path": str(args.output.resolve()),
        "sha256": file_sha256(args.output),
        "parameter_count": parameter_count(model),
    }
    report["checkpoint"] = checkpoint_artifact
    metric_log.append("evaluation", {"split": evaluation_split, **metrics})
    metric_log.append("checkpoint", checkpoint_artifact)
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
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
