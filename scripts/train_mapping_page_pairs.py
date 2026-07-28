#!/usr/bin/env python3
"""Train one matcher with page augmentation and strict cross-page pseudo labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from omnitransfer.learned_matcher import (
    LearnedGraphMatcher,
    MatcherConfig,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.mapping_training import load_mapping_training_pairs
from omnitransfer.self_supervised import (
    AugmentConfig,
    evaluate_correspondence_pairs,
    train_mapping_matcher,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--target-context-nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--minimum-correspondences", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--allow-unreviewed-pseudo", action="store_true")
    parser.add_argument("--screenshot-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pretrained_model = None
    if args.pretrained is not None:
        pretrained = LearnedGraphMatcher.from_checkpoint(
            args.pretrained,
            device=args.device,
        )
        config = replace(
            pretrained.config,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
        )
        pretrained_model = pretrained.model
    else:
        config = MatcherConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            source_context_nodes=args.source_context_nodes,
            target_context_nodes=args.target_context_nodes,
        )
    pairs, graphs, adapter = load_mapping_training_pairs(
        args.input,
        matcher_config=config,
        allow_unreviewed_pseudo=args.allow_unreviewed_pseudo,
        minimum_correspondences=args.minimum_correspondences,
        max_pairs=args.max_pairs,
        screenshot_root=args.screenshot_root,
    )
    if not pairs:
        raise SystemExit("No strict correspondence pairs were accepted")
    preview = {
        "adapter": adapter,
        "matcher_config": asdict(config),
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "training": "mixed_self_supervised_and_cross_page_pairs",
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    augment = AugmentConfig()

    def report_progress(metrics: dict[str, float]) -> None:
        print(
            json.dumps({"event": "training_progress", **metrics}),
            flush=True,
        )

    model, history = train_mapping_matcher(
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
        progress_interval=args.progress_interval,
    )
    metrics = evaluate_correspondence_pairs(
        model,
        pairs,
        device=args.device,
        matcher_config=config,
    )
    report = {
        "schema_version": "omnitransfer.mixed_mapping_experiment.v1",
        "architecture": (
            "local_relation_cross_attention_with_affinity_propagation"
            if config.local_affinity_propagation
            else "legacy_local_relation_cross_attention_matcher"
        ),
        "objective": "symmetric_partial_assignment_with_bidirectional_consistency",
        "inputs": [str(path.resolve()) for path in args.input],
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "adapter": adapter,
        "matcher_config": asdict(config),
        "parameter_count": parameter_count(model),
        "augmentation": asdict(augment),
        "training": {"history": history},
        "training_set_fit_smoke": metrics,
        "evaluation_boundary": "formal metrics require held-out human gold",
    }
    checkpoint_metadata = {
        key: value for key, value in report.items() if key != "training_set_fit_smoke"
    }
    save_matcher_checkpoint(
        args.output,
        model,
        config=config,
        metadata=checkpoint_metadata,
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
                "top1_fit_smoke": metrics["top1_accuracy"],
                "warm_p95_ms": metrics["warm_model_latency_ms"]["p95"],
                "warm_end_to_end_p95_ms": metrics["warm_end_to_end_latency_ms"]["p95"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
