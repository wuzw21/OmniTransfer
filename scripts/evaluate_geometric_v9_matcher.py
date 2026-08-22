#!/usr/bin/env python3
"""Evaluate the sparse anchor-identity transport matcher."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.learned_matcher import (
    GeometricMatcher,
    MatcherConfig,
    parameter_count,
)
from omnitransfer.mapping_training import load_ui_correspondence_pairs
from omnitransfer.self_supervised import (
    CorrespondencePair,
    evaluate_correspondence_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional JSONL path for per-gold-row rankings and score components",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "diagnostic"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=2)
    parser.add_argument(
        "--score-stage",
        choices=("unary", "final"),
        default="final",
        help="Rank with the separately trainable node stage or final matcher.",
    )
    parser.add_argument(
        "--mask-visual",
        action="store_true",
        help="Evaluate the checkpoint with visual evidence marked unavailable.",
    )
    parser.add_argument(
        "--mask-text",
        action="store_true",
        help="Evaluate the checkpoint with semantic token evidence removed.",
    )
    parser.add_argument(
        "--mask-xml",
        action="store_true",
        help="Evaluate the checkpoint with XML, state, layout, and relations zeroed.",
    )
    parser.add_argument(
        "--disable-text-anchor-residual",
        action="store_true",
        help="Zero only the sparse text anchor residual output projection.",
    )
    parser.add_argument(
        "--disable-visual-anchor-residual",
        action="store_true",
        help="Zero only the sparse visual anchor residual output projection.",
    )
    parser.add_argument(
        "--disable-xml-anchor-residual",
        action="store_true",
        help="Zero only the sparse XML anchor residual output projection.",
    )
    parser.add_argument(
        "--disable-anchor-identity-transport",
        action="store_true",
        help="Zero the anchor-identity transport output for causal ablation.",
    )
    parser.add_argument(
        "--shuffle-anchor-identities",
        action="store_true",
        help="Shuffle transported counterpart identities without changing anchors.",
    )
    parser.add_argument(
        "--disable-relation-types",
        action="store_true",
        help="Zero relation-type embeddings while preserving relation edges.",
    )
    parser.add_argument(
        "--disable-state-context",
        action="store_true",
        help="Disconnect the 1024D page state from node-pair scoring.",
    )
    parser.add_argument(
        "--disable-neighbour-fusion",
        action="store_true",
        help="Remove selected within-page relation content from pair scoring.",
    )
    parser.add_argument(
        "--disable-correspondence-feedback",
        action="store_true",
        help="Remove the current soft correspondence from local relation matching.",
    )
    parser.add_argument(
        "--disable-local-correction",
        action="store_true",
        help="Use the trained v9 contextual score without the learned local correction.",
    )
    parser.add_argument("--semantic-exact-decoder-bonus", type=float)
    parser.add_argument("--semantic-exact-decoder-top-k", type=int)
    parser.add_argument("--semantic-exact-decoder-max-margin", type=float)
    parser.add_argument("--include-all-candidates", action="store_true")
    parser.add_argument("--allow-unreviewed-pseudo", action="store_true")
    parser.add_argument(
        "--reserved-split",
        choices=("dev", "test"),
        help=(
            "Evaluate only diagnostic review candidates assigned to this "
            "frozen held-out split."
        ),
    )
    parser.add_argument("--screenshot-root", type=Path)
    args = parser.parse_args()

    matcher = GeometricMatcher.from_checkpoint(args.checkpoint, device=args.device)
    matcher.config = replace(
        matcher.config,
        semantic_exact_decoder_bonus=(
            args.semantic_exact_decoder_bonus
            if args.semantic_exact_decoder_bonus is not None
            else matcher.config.semantic_exact_decoder_bonus
        ),
        semantic_exact_decoder_top_k=(
            args.semantic_exact_decoder_top_k
            if args.semantic_exact_decoder_top_k is not None
            else matcher.config.semantic_exact_decoder_top_k
        ),
        semantic_exact_decoder_max_margin=(
            args.semantic_exact_decoder_max_margin
            if args.semantic_exact_decoder_max_margin is not None
            else matcher.config.semantic_exact_decoder_max_margin
        ),
    )
    residual_ablation = disable_node_anchor_residuals(
        matcher.model,
        text=args.disable_text_anchor_residual,
        visual=args.disable_visual_anchor_residual,
        xml=args.disable_xml_anchor_residual,
    )
    transport_ablation = configure_anchor_transport_ablation(
        matcher.model,
        disable=args.disable_anchor_identity_transport,
        shuffle_identities=args.shuffle_anchor_identities,
        disable_relation_types=args.disable_relation_types,
    )
    local_fusion_ablation = configure_local_fusion_ablation(
        matcher.model,
        disable_state_context=args.disable_state_context,
        disable_neighbour_fusion=args.disable_neighbour_fusion,
        disable_correspondence_feedback=args.disable_correspondence_feedback,
        disable_local_correction=args.disable_local_correction,
    )
    pairs, adapter = load_evaluation_pairs(
        args.input,
        split=args.split,
        config=matcher.config,
        max_pairs=args.max_pairs,
        allow_unreviewed_pseudo=args.allow_unreviewed_pseudo,
        screenshot_root=args.screenshot_root,
        reserved_split=args.reserved_split,
    )
    predictions: list[dict[str, Any]] = []
    evaluation_model = (
        _ablation_model(
            matcher.model,
            mask_text=args.mask_text,
            mask_visual=args.mask_visual,
            mask_xml=args.mask_xml,
        )
        if args.mask_text or args.mask_visual or args.mask_xml
        else matcher.model
    )
    metrics = evaluate_correspondence_pairs(
        evaluation_model,
        pairs,
        device=args.device,
        matcher_config=matcher.config,
        latency_repeats=args.latency_repeats,
        prediction_callback=(predictions.append if args.predictions else None),
        include_all_candidates=args.include_all_candidates,
        score_stage=args.score_stage,
    )
    prediction_artifact = None
    if args.predictions is not None:
        prediction_artifact = write_predictions(args.predictions, predictions)
    report = {
        "schema_version": "omnitransfer.geometric_v9_evaluation.v1",
        "record_schema": "omnitransfer.ui_correspondence_pair.v1",
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
        "architecture": matcher.config.architecture,
        "matcher_config": asdict(matcher.config),
        "device": args.device,
        "score_stage": args.score_stage,
        "ablation": {
            "text_masked": args.mask_text,
            "visual_masked": args.mask_visual,
            "xml_evidence_zeroed": args.mask_xml,
            "text_anchor_residual_disabled": args.disable_text_anchor_residual,
            "visual_anchor_residual_disabled": (
                args.disable_visual_anchor_residual
            ),
            "xml_anchor_residual_disabled": args.disable_xml_anchor_residual,
            "anchor_identity_transport_disabled": (
                args.disable_anchor_identity_transport
            ),
            "anchor_identities_shuffled": args.shuffle_anchor_identities,
            "relation_types_disabled": args.disable_relation_types,
            "disabled_residual_weight_stats": residual_ablation,
            "anchor_transport_ablation_stats": transport_ablation,
            "state_context_disabled": args.disable_state_context,
            "neighbour_fusion_disabled": args.disable_neighbour_fusion,
            "correspondence_feedback_disabled": (
                args.disable_correspondence_feedback
            ),
            "local_correction_disabled": args.disable_local_correction,
            "local_fusion_ablation_stats": local_fusion_ablation,
        },
        "adapter": adapter,
        "metrics": metrics,
        "predictions": prediction_artifact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False))


def write_predictions(
    path: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write one JSONL row per evaluated correspondence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": len(predictions),
    }


def _ablation_model(
    model: Any,
    *,
    mask_text: bool,
    mask_visual: bool,
    mask_xml: bool,
) -> Any:
    import torch

    class EvidenceAblation(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *inputs: Any, **kwargs: Any) -> dict[str, Any]:
            masked_inputs = list(inputs)
            if mask_text:
                masked_inputs[0] = torch.zeros_like(masked_inputs[0])
                masked_inputs[3] = torch.zeros_like(masked_inputs[3])
            if mask_xml:
                for index in (1, 2, 4, 5):
                    masked_inputs[index] = torch.zeros_like(masked_inputs[index])
            if mask_visual:
                masked_inputs[7] = torch.zeros_like(masked_inputs[7])
                masked_inputs[9] = torch.zeros_like(masked_inputs[9])
            return self.wrapped(*masked_inputs, **kwargs)

    return EvidenceAblation(model)


def disable_node_anchor_residuals(
    model: Any,
    *,
    text: bool = False,
    visual: bool = False,
    xml: bool = False,
) -> dict[str, dict[str, float]]:
    """Disable selected function-preserving residuals for eval-only ablation."""

    import torch

    requested = {
        "text": (text, ("text_token_output",)),
        "visual": (visual, ("visual_encoder", "sparse_output")),
        "xml": (xml, ("xml_anchor_output",)),
    }
    statistics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, (enabled, path) in requested.items():
            if not enabled:
                continue
            module = model
            for attribute in path:
                module = getattr(module, attribute, None)
                if module is None:
                    raise ValueError(
                        f"checkpoint has no {name} node-anchor residual"
                    )
            weight = getattr(module, "weight", None)
            if weight is None:
                raise ValueError(
                    f"checkpoint {name} node-anchor residual has no weight"
                )
            statistics[name] = {
                "l2_norm_before_zero": float(weight.norm().detach().cpu()),
                "max_abs_before_zero": float(weight.abs().max().detach().cpu()),
            }
            weight.zero_()
    return statistics


def configure_anchor_transport_ablation(
    model: Any,
    *,
    disable: bool = False,
    shuffle_identities: bool = False,
    disable_relation_types: bool = False,
) -> dict[str, Any]:
    """Apply causal ablations to the single anchor-transport path."""

    if not (disable or shuffle_identities or disable_relation_types):
        return {}
    import torch

    layer = getattr(model, "association_layer", None)
    projection = getattr(layer, "transport_output", None)
    relation_types = getattr(layer, "relation_type_embedding", None)
    counterpart_projection = getattr(layer, "counterpart_projection", None)
    weight = getattr(projection, "weight", None)
    if weight is None or relation_types is None or counterpart_projection is None:
        raise ValueError("checkpoint has no sparse anchor-identity transport")
    statistics: dict[str, Any] = {}
    with torch.no_grad():
        statistics["transport_output"] = {
            "l2_norm_before_zero": float(weight.norm().detach().cpu()),
            "max_abs_before_zero": float(weight.abs().max().detach().cpu()),
            "nonzero_parameters_before_zero": float(
                weight.ne(0.0).sum().detach().cpu()
            ),
        }
        statistics["relation_type_embedding"] = {
            "l2_norm_before_zero": float(relation_types.norm().detach().cpu()),
            "max_abs_before_zero": float(
                relation_types.abs().max().detach().cpu()
            ),
        }
        if disable:
            weight.zero_()
        if disable_relation_types:
            relation_types.zero_()
    if shuffle_identities:
        counterpart_projection.register_forward_pre_hook(
            lambda _module, inputs: (inputs[0].roll(shifts=1, dims=0),)
        )
    statistics["disabled"] = disable
    statistics["identities_shuffled"] = shuffle_identities
    statistics["relation_types_disabled"] = disable_relation_types
    return statistics


def configure_local_fusion_ablation(
    model: Any,
    *,
    disable_state_context: bool = False,
    disable_neighbour_fusion: bool = False,
    disable_correspondence_feedback: bool = False,
    disable_local_correction: bool = False,
) -> dict[str, dict[str, float]]:
    """Disconnect one learned evidence path for causal evaluation only."""

    requested = {
        "state_context": (
            disable_state_context,
            getattr(model, "state_to_pair", None),
        ),
        "neighbour_fusion": (
            disable_neighbour_fusion,
            (
                model.relation_encoder[3]
                if getattr(model, "relation_encoder", None) is not None
                else None
            ),
        ),
        "correspondence_feedback": (
            disable_correspondence_feedback,
            getattr(model, "correspondence_evidence", None),
        ),
        "local_correction": (
            disable_local_correction,
            (
                model.pair_scorer[-1]
                if getattr(model, "pair_scorer", None) is not None
                else None
            ),
        ),
    }
    if not any(enabled for enabled, _module in requested.values()):
        return {}
    import torch

    statistics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, (enabled, module) in requested.items():
            if not enabled:
                continue
            if module is None and name == "state_context":
                statistics[name] = {"already_independent": 1.0}
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                raise ValueError(f"checkpoint has no {name} weight")
            stats = {
                "l2_norm_before_zero": float(weight.norm().detach().cpu()),
                "max_abs_before_zero": float(weight.abs().max().detach().cpu()),
                "nonzero_parameters_before_zero": float(
                    weight.ne(0.0).sum().detach().cpu()
                ),
            }
            weight.zero_()
            bias = getattr(module, "bias", None)
            if bias is not None:
                stats["bias_l2_norm_before_zero"] = float(
                    bias.norm().detach().cpu()
                )
                bias.zero_()
            statistics[name] = stats
    return statistics


def load_evaluation_pairs(
    inputs: list[Path],
    *,
    split: str,
    config: MatcherConfig,
    max_pairs: int = 0,
    allow_unreviewed_pseudo: bool = False,
    screenshot_root: Path | None = None,
    reserved_split: str | None = None,
) -> tuple[list[CorrespondencePair], dict[str, Any]]:
    """Use the training adapter unchanged, constrained to one evaluation split."""

    pairs, _, adapter = load_ui_correspondence_pairs(
        inputs,
        matcher_config=config,
        allow_unreviewed_pseudo=allow_unreviewed_pseudo,
        minimum_correspondences=1,
        max_pairs=max_pairs,
        screenshot_root=screenshot_root,
        allowed_splits=frozenset({split}),
        allowed_reserved_splits=(
            frozenset({reserved_split}) if reserved_split is not None else None
        ),
    )
    if not pairs:
        raise SystemExit(f"No actionable {split} correspondence pairs were accepted")
    return pairs, adapter


if __name__ == "__main__":
    main()
