#!/usr/bin/env python3
"""Train the one OmniTransfer page-local matcher from paired observations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omnitransfer.learned_matcher import (
    GeometricMatcher,
    MatcherConfig,
    V9_BACKBONE_PARAMETER_PREFIXES,
    build_geometric_v9_matcher,
    initialize_node_encoder_from_checkpoint,
    initialize_v9_backbone_from_checkpoint,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.mapping_training import load_ui_correspondence_pairs
from omnitransfer.self_supervised import (
    AugmentConfig,
    CorrespondencePair,
    batch_supervised_node_matching_loss,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train local UI correspondence directly from reviewed cross-platform "
            "node pairs with the v9 multimodal encoder, learned local fusion, "
            "and repeated correspondence updates. No page loss, NULL, gates, "
            "teacher, rerank, or auxiliary losses are used."
        )
    )
    parser.add_argument(
        "--input",
        "--unlabeled-input",
        action="append",
        dest="inputs",
        required=True,
        type=Path,
        help="Reviewed UI correspondence-pair JSONL used as node supervision.",
    )
    parser.add_argument(
        "--validation-input",
        action="append",
        default=[],
        type=Path,
        help="Held-out page-pair JSONL with split=dev.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument(
        "--screenshot-root",
        type=Path,
        help="Portable root used to relocate screenshot paths stored on another host.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--association-layers",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Ablate the number of correspondence updates; production stays fixed after Dev selection.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--node-learning-rate-scale",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for the initialized v9 node encoder.",
    )
    parser.add_argument(
        "--freeze-v9-backbone",
        action="store_true",
        help="Ablation only: train the new local path without changing migrated v9 parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-validation-pairs", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("node", "joint"),
        default="joint",
        help="Train unary node correspondence alone or the complete matcher.",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initialize-from",
        type=Path,
        help="Continue from an exact checkpoint of this matcher.",
    )
    initialization.add_argument(
        "--initialize-node-from",
        type=Path,
        help="Load only compatible node/unary parameters into a fresh matcher.",
    )
    initialization.add_argument(
        "--initialize-v9-from",
        type=Path,
        help="Restore the compatible trained v9 backbone; initialize new local fusion fresh.",
    )
    parser.add_argument(
        "--hard-scores",
        type=Path,
        help="Compact correspondence difficulty scores used only for sampling.",
    )
    parser.add_argument(
        "--hard-replay-ratio",
        type=float,
        default=0.0,
        help="Additional hard-page samples per ordinary training pair.",
    )
    parser.add_argument(
        "--hard-row-weight",
        type=float,
        default=0.0,
        help="Upweight difficult correspondence rows inside the same cross-entropy.",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least two")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if not 0.0 < args.node_learning_rate_scale <= 1.0:
        raise SystemExit("--node-learning-rate-scale must be in (0, 1]")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be non-negative")
    if args.max_pairs < 0 or args.max_validation_pairs < 0:
        raise SystemExit("pair limits must be non-negative")
    if not 0.0 <= args.hard_replay_ratio <= 1.0:
        raise SystemExit("--hard-replay-ratio must be in [0, 1]")
    if args.hard_replay_ratio and args.hard_scores is None:
        raise SystemExit("--hard-replay-ratio requires --hard-scores")
    if args.hard_row_weight < 0.0:
        raise SystemExit("--hard-row-weight must be non-negative")


def _page_batches(
    pairs: Iterable[CorrespondencePair],
    *,
    batch_size: int,
    rng: random.Random,
    pair_difficulty: dict[tuple[str, str], float] | None = None,
    hard_replay_ratio: float = 0.0,
) -> list[tuple[CorrespondencePair, ...]]:
    """Visit every pair once and optionally replay model-scored hard pages."""

    values = list(pairs)
    if pair_difficulty and hard_replay_ratio > 0.0:
        weights = [
            max(0.0, float(pair_difficulty.get(_pair_key(pair), 0.0)))
            for pair in values
        ]
        if sum(weights) > 0.0:
            values.extend(
                rng.choices(
                    values,
                    weights=weights,
                    k=math.ceil(len(values) * hard_replay_ratio),
                )
            )
    rng.shuffle(values)
    return [
        tuple(values[start : start + batch_size])
        for start in range(0, len(values), batch_size)
    ]


def _pair_key(pair: Any) -> tuple[str, str]:
    return tuple(sorted((str(pair.graph_a.graph_id), str(pair.graph_b.graph_id))))


def _load_pair_difficulty(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            graph_pair = row.get("graph_pair") or {}
            source = str(graph_pair.get("source") or "")
            target = str(graph_pair.get("target") or "")
            if not source or not target:
                continue
            totals[tuple(sorted((source, target)))].append(
                float(row.get("difficulty_score") or 0.0)
            )
    return {
        key: statistics.fmean(values)
        for key, values in totals.items()
        if values
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return statistics.fmean(values) if values else 0.0


def _run_epoch(
    model: Any,
    pairs: list[CorrespondencePair],
    *,
    optimizer: Any | None,
    config: MatcherConfig,
    augment: AugmentConfig | None,
    device: str,
    batch_size: int,
    rng: random.Random,
    progress_every: int,
    torch: Any,
    score_stage: str = "final",
    pair_difficulty: dict[tuple[str, str], float] | None = None,
    hard_replay_ratio: float = 0.0,
    hard_row_weight: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    batches = _page_batches(
        pairs,
        batch_size=batch_size,
        rng=rng,
        pair_difficulty=pair_difficulty,
        hard_replay_ratio=hard_replay_ratio,
    )
    if not batches:
        raise ValueError("supervised node training needs at least one page pair")
    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    for step, batch in enumerate(batches, start=1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast("cpu", enabled=False)
        )
        with torch.set_grad_enabled(training), context:
            loss, details = batch_supervised_node_matching_loss(
                model,
                batch,
                device=device,
                matcher_config=config,
                rng=rng,
                augment_config=augment,
                score_stage=score_stage,
                hard_row_weight=hard_row_weight,
                return_details=True,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite page loss at batch {step}")
        if optimizer is not None:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient at batch {step}")
            optimizer.step()
            details["gradient_norm"] = float(gradient_norm.detach().cpu())
        rows.append(details)
        if progress_every and step % progress_every == 0:
            print(
                json.dumps(
                    {
                        "batch": step,
                        "batches": len(batches),
                        "metric_scope": "online_pre_update_batches",
                        "loss": _mean(rows[-progress_every:], "total_loss"),
                        "node_top1": _mean(
                            rows[-progress_every:], "node_top1_accuracy"
                        ),
                        "unary_top1": _mean(
                            rows[-progress_every:], "unary_node_top1_accuracy"
                        ),
                        "refinement_help": sum(
                            row.get("refinement_help", 0.0)
                            for row in rows[-progress_every:]
                        ),
                        "refinement_hurt": sum(
                            row.get("refinement_hurt", 0.0)
                            for row in rows[-progress_every:]
                        ),
                        "score_margin": _mean(
                            rows[-progress_every:], "node_score_margin"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    elapsed = time.perf_counter() - start
    supervised_query_rows = sum(
        float(row.get("supervised_query_rows", 0.0)) for row in rows
    )
    node_top1_correct = sum(
        float(row.get("node_top1_correct", 0.0)) for row in rows
    )
    unary_node_top1_correct = sum(
        float(row.get("unary_node_top1_correct", 0.0)) for row in rows
    )
    layer_accuracies = {}
    layer_count = int(max(
        (row.get("supervised_layers", 0.0) for row in rows),
        default=0.0,
    ))
    for layer_index in range(1, layer_count + 1):
        layer_correct = sum(
            float(row.get(f"layer_{layer_index}_top1_correct", 0.0))
            for row in rows
        )
        layer_accuracies[f"layer_{layer_index}_top1_accuracy"] = (
            layer_correct / supervised_query_rows
            if supervised_query_rows
            else 0.0
        )
    return {
        "metric_scope": (
            "online_pre_update_batches"
            if training
            else "fixed_model_full_pass"
        ),
        "score_stage": score_stage,
        "loss": _mean(rows, "total_loss"),
        "positive_score": _mean(rows, "node_positive_score"),
        "negative_score": _mean(rows, "node_hard_negative_score"),
        "score_margin": _mean(rows, "node_score_margin"),
        "node_top1_accuracy": (
            node_top1_correct / supervised_query_rows
            if supervised_query_rows
            else 0.0
        ),
        "unary_node_top1_accuracy": (
            unary_node_top1_correct / supervised_query_rows
            if supervised_query_rows
            else 0.0
        ),
        **layer_accuracies,
        "refinement_help": sum(
            float(row.get("refinement_help", 0.0)) for row in rows
        ),
        "refinement_hurt": sum(
            float(row.get("refinement_hurt", 0.0)) for row in rows
        ),
        "human_node_labels": sum(
            float(row.get("human_node_labels", 0.0)) for row in rows
        ),
        "supervised_query_rows": supervised_query_rows,
        "gradient_norm": _mean(rows, "gradient_norm"),
        "batches": float(len(batches)),
        "pairs": float(sum(len(batch) for batch in batches)),
        "seconds": elapsed,
        "pairs_per_second": sum(len(batch) for batch in batches)
        / max(elapsed, 1e-9),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parameter_distribution(model: Any) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        groups[name.split(".", 1)[0]] += int(parameter.numel())
    return dict(sorted(groups.items(), key=lambda item: (-item[1], item[0])))


def _optimizer_parameter_groups(
    model: Any,
    *,
    weight_decay: float,
    learning_rate: float = 3e-4,
    node_learning_rate_scale: float = 0.1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[bool, bool], list[Any]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_node_encoder = name.startswith(V9_BACKBONE_PARAMETER_PREFIXES)
        uses_decay = parameter.ndim >= 2 and "embedding" not in name
        grouped[(is_node_encoder, uses_decay)].append(parameter)
    return [
        {
            "params": parameters,
            "weight_decay": weight_decay if uses_decay else 0.0,
            "lr": (
                learning_rate * node_learning_rate_scale
                if is_node_encoder
                else learning_rate
            ),
        }
        for (is_node_encoder, uses_decay), parameters in grouped.items()
        if parameters
    ]


def _freeze_v9_backbone(model: Any) -> tuple[str, ...]:
    """Freeze only parameters restored from v9 for a causal training ablation."""

    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith(V9_BACKBONE_PARAMETER_PREFIXES):
            parameter.requires_grad_(False)
            frozen.append(name)
    return tuple(frozen)


def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("training requires PyTorch") from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    config = MatcherConfig(association_layers=args.association_layers)
    train_pairs, _, train_manifest = load_ui_correspondence_pairs(
        args.inputs,
        matcher_config=config,
        max_pairs=args.max_pairs,
        screenshot_root=args.screenshot_root,
        allowed_splits=frozenset({"train", "diagnostic"}),
    )
    if not train_pairs:
        raise SystemExit("no reviewed training page pairs were accepted")
    validation_pairs: list[CorrespondencePair] = []
    validation_manifest = None
    if args.validation_input:
        validation_pairs, _, validation_manifest = load_ui_correspondence_pairs(
            args.validation_input,
            matcher_config=config,
            max_pairs=args.max_validation_pairs,
            screenshot_root=args.screenshot_root,
            allowed_splits=frozenset({"dev"}),
        )
        if not validation_pairs:
            raise SystemExit("no reviewed validation page pairs were accepted")

    transferred_node_parameters: tuple[str, ...] = ()
    if args.initialize_from is not None:
        restored = GeometricMatcher.from_checkpoint(
            args.initialize_from,
            device=args.device,
        )
        if restored.config != config:
            raise SystemExit(
                "--initialize-from requires the current matcher configuration; "
                "use --initialize-node-from for an older checkpoint"
            )
        model = restored.model
    else:
        model = build_geometric_v9_matcher(config).to(args.device)
        if args.initialize_node_from is not None:
            transferred_node_parameters = initialize_node_encoder_from_checkpoint(
                model,
                args.initialize_node_from,
            )
        elif args.initialize_v9_from is not None:
            transferred_node_parameters = initialize_v9_backbone_from_checkpoint(
                model,
                args.initialize_v9_from,
            )
    frozen_v9_parameters = (
        _freeze_v9_backbone(model) if args.freeze_v9_backbone else ()
    )
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(
            model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            node_learning_rate_scale=args.node_learning_rate_scale,
        ),
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    train_augment = None if args.no_augmentation else AugmentConfig()
    validation_augment = None
    pair_difficulty = _load_pair_difficulty(args.hard_scores)
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_validation_accuracy = -math.inf
    best_epoch = 0
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_pairs,
            optimizer=optimizer,
            config=config,
            augment=train_augment,
            device=args.device,
            batch_size=args.batch_size,
            rng=random.Random(args.seed * 1000 + epoch),
            progress_every=args.progress_every,
            torch=torch,
            score_stage=("unary" if args.stage == "node" else "final"),
            pair_difficulty=pair_difficulty,
            hard_replay_ratio=args.hard_replay_ratio,
            hard_row_weight=args.hard_row_weight,
        )
        validation_metrics = None
        if validation_pairs:
            with torch.no_grad():
                validation_metrics = _run_epoch(
                    model,
                    validation_pairs,
                    optimizer=None,
                    config=config,
                    augment=validation_augment,
                    device=args.device,
                    batch_size=args.batch_size,
                    rng=random.Random(args.seed),
                    progress_every=0,
                    torch=torch,
                    score_stage=("unary" if args.stage == "node" else "final"),
                )
        selection_loss = (
            validation_metrics["loss"] if validation_metrics else train_metrics["loss"]
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selected": False,
        }
        history.append(row)
        selection_accuracy = (
            validation_metrics["node_top1_accuracy"]
            if validation_metrics
            else train_metrics["node_top1_accuracy"]
        )
        if (
            selection_accuracy > best_validation_accuracy
            or (
                selection_accuracy == best_validation_accuracy
                and selection_loss < best_validation
            )
        ):
            best_validation = selection_loss
            best_validation_accuracy = selection_accuracy
            best_epoch = epoch
            save_matcher_checkpoint(
                args.output,
                model,
                config=config,
                metadata={
                    "objective": "symmetric_cross_platform_node_cross_entropy",
                    "node_correspondence_labels_consumed": True,
                    "epoch": epoch,
                    "validation_loss": selection_loss,
                    "validation_top1_accuracy": selection_accuracy,
                    "training_stage": args.stage,
                },
            )
            row["selected"] = True
        print(json.dumps(row, sort_keys=True), flush=True)

    for row in history:
        row["selected"] = row["epoch"] == best_epoch
    best_matcher = GeometricMatcher.from_checkpoint(
        args.output,
        device=args.device,
    )
    with torch.no_grad():
        best_checkpoint_train_metrics = _run_epoch(
            best_matcher.model,
            train_pairs,
            optimizer=None,
            config=best_matcher.config,
            augment=None,
            device=args.device,
            batch_size=args.batch_size,
            rng=random.Random(args.seed),
            progress_every=0,
            torch=torch,
            score_stage=("unary" if args.stage == "node" else "final"),
        )
    report_path = args.report or args.output.with_suffix(".json")
    history_path = args.history or args.output.with_suffix(".history.jsonl")
    report = {
        "schema_version": "omnitransfer.supervised_page_local_training.v1",
        "method": (
            "complete trained v9 multimodal and contextual backbone -> learned "
            "five-neighbour local fusion -> three correspondence-conditioned "
            "updates -> one "
            "symmetric cross-platform node cross-entropy"
        ),
        "training_stage": args.stage,
        "initialization": {
            "exact_checkpoint": (
                str(args.initialize_from.resolve())
                if args.initialize_from is not None
                else None
            ),
            "node_checkpoint": (
                str(args.initialize_node_from.resolve())
                if args.initialize_node_from is not None
                else None
            ),
            "v9_backbone_checkpoint": (
                str(args.initialize_v9_from.resolve())
                if args.initialize_v9_from is not None
                else None
            ),
            "transferred_node_parameters": len(transferred_node_parameters),
        },
        "hard_replay": {
            "score_file": (
                str(args.hard_scores.resolve())
                if args.hard_scores is not None
                else None
            ),
            "scored_page_pairs": len(pair_difficulty),
            "ratio": args.hard_replay_ratio,
        },
        "hard_rows": {
            "weight": args.hard_row_weight,
            "definition": "1 + weight * (1 - detached gold probability)",
            "runtime_effect": "none",
        },
        "metric_contract": {
            "history.train": (
                "online pre-update batch aggregate; diagnostic only"
            ),
            "history.validation": "fixed epoch-end model full pass",
            "best_checkpoint_train": (
                "selected checkpoint fixed-model full Train pass"
            ),
            "checkpoint_selection": "fixed validation Top-1 then validation loss",
        },
        "matcher_config": asdict(config),
        "parameters": parameter_count(model),
        "parameter_distribution": _parameter_distribution(model),
        "active_model_config": {
            "node_width": config.hidden_dim,
            "local_relation_width": config.relation_hidden_dim,
            "local_neighbours_per_node": config.local_neighbor_limit,
            "correspondence_stages": config.association_layers,
            "trained_v9_contextual_layers": config.association_layers,
            "null_class": False,
            "modality_router": True,
            "gates": False,
            "reranking": False,
            "auxiliary_losses": False,
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "node_learning_rate_scale": args.node_learning_rate_scale,
            "node_learning_rate": (
                args.learning_rate * args.node_learning_rate_scale
            ),
            "freeze_v9_backbone": args.freeze_v9_backbone,
            "frozen_v9_parameters": len(frozen_v9_parameters),
        },
        "device": args.device,
        "gpu": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if args.device.startswith("cuda")
            else None
        ),
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "best_validation_top1_accuracy": best_validation_accuracy,
        "best_checkpoint_train": best_checkpoint_train_metrics,
        "elapsed_seconds": time.perf_counter() - run_start,
        "checkpoint": str(args.output.resolve()),
        "history": history,
    }
    _atomic_json(report_path, report)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    print(json.dumps({"result": report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
