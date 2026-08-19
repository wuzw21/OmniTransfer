"""Self-supervised training utilities for relation-aware UI matching."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    DETERMINISTIC_ICON_VISUAL_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    XML_NODE_FEATURE_DIM,
    EncodedGraph,
    MatcherConfig,
    build_geometric_v9_matcher,
    confidence_adaptive_assignment_row,
    encode_graph,
    matcher_inputs,
)
from omnitransfer.unified_alignment import relation_consistency_loss
from omnitransfer.ui_graph import BBox, UIGraph, UINode

FEATURE_DIM = XML_NODE_FEATURE_DIM


@dataclass(frozen=True)
class AugmentConfig:
    """Controls topology, attribute, and layout perturbations for two UI views."""

    mask_text_prob: float = 0.10
    mask_content_desc_prob: float = 0.10
    mask_class_prob: float = 0.10
    drop_node_prob: float = 0.15
    edge_dropout_prob: float = 0.10
    bbox_jitter: float = 0.02
    global_translation: float = 0.05
    global_scale: float = 0.08
    visual_brightness: float = 0.10
    visual_contrast: float = 0.12
    visual_channel_scale: float = 0.08
    visual_dropout_prob: float = 0.10
    distractor_prob: float = 0.20
    max_distractors: int = 4
    shuffle_nodes: bool = True
    min_nodes: int = 4


@dataclass(frozen=True)
class CorrespondencePair:
    """Two related graph views with known correspondences and ignored rows."""

    graph_a: UIGraph
    graph_b: UIGraph
    encoded_a: EncodedGraph
    encoded_b: EncodedGraph
    targets_a_to_b: tuple[int, ...]
    targets_b_to_a: tuple[int, ...]
    positive_targets_a_to_b: tuple[tuple[int, ...], ...]
    positive_targets_b_to_a: tuple[tuple[int, ...], ...]
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    origin_ids: tuple[str, ...]

def make_correspondence_training_pair(
    graph_a: UIGraph,
    graph_b: UIGraph,
    correspondences: Iterable[tuple[str, str]],
    *,
    matcher_config: MatcherConfig | None = None,
    allow_empty: bool = False,
    unmatched_as_null: bool = False,
) -> CorrespondencePair:
    """Construct partial-assignment labels from explicit cross-graph node pairs."""

    config = matcher_config or MatcherConfig()
    index_a = {node.node_id: index for index, node in enumerate(graph_a.nodes)}
    index_b = {node.node_id: index for index, node in enumerate(graph_b.nodes)}
    if allow_empty and not unmatched_as_null:
        raise ValueError("empty correspondence requires explicit NULL supervision")
    unmatched_target = -1 if unmatched_as_null else -2
    positive_a_to_b = [set() for _ in graph_a.nodes]
    positive_b_to_a = [set() for _ in graph_b.nodes]
    source_indices: list[int] = []
    target_indices: list[int] = []
    labels: list[str] = []
    for source_id, target_id in dict.fromkeys(correspondences):
        if source_id not in index_a:
            raise ValueError(f"source correspondence node is absent: {source_id}")
        if target_id not in index_b:
            raise ValueError(f"target correspondence node is absent: {target_id}")
        source_index = index_a[source_id]
        target_index = index_b[target_id]
        positive_a_to_b[source_index].add(target_index)
        positive_b_to_a[target_index].add(source_index)
        source_indices.append(source_index)
        target_indices.append(target_index)
        labels.append(f"{source_id}->{target_id}")
    if not source_indices and not allow_empty:
        raise ValueError("at least one correspondence is required")
    targets_a_to_b = [
        min(targets) if targets else unmatched_target
        for targets in positive_a_to_b
    ]
    targets_b_to_a = [
        min(targets) if targets else unmatched_target
        for targets in positive_b_to_a
    ]
    return CorrespondencePair(
        graph_a=graph_a,
        graph_b=graph_b,
        encoded_a=encode_graph(graph_a, config=config),
        encoded_b=encode_graph(graph_b, config=config),
        targets_a_to_b=tuple(targets_a_to_b),
        targets_b_to_a=tuple(targets_b_to_a),
        positive_targets_a_to_b=tuple(
            tuple(sorted(targets)) for targets in positive_a_to_b
        ),
        positive_targets_b_to_a=tuple(
            tuple(sorted(targets)) for targets in positive_b_to_a
        ),
        source_indices=tuple(source_indices),
        target_indices=tuple(target_indices),
        origin_ids=tuple(labels),
    )


def augment_graph(
    graph: UIGraph,
    *,
    rng: random.Random,
    config: AugmentConfig | None = None,
    view_id: str = "view",
) -> UIGraph:
    """Create a structurally perturbed UI view while retaining latent identity."""

    cfg = config or AugmentConfig()
    kept_nodes = _select_kept_nodes(graph.nodes, rng=rng, config=cfg)
    kept_ids = {node.node_id for node in kept_nodes}
    transform = _sample_global_transform(rng, config=cfg)
    augmented: list[UINode] = []
    for node in kept_nodes:
        child_ids = tuple(
            child_id
            for child_id in node.child_ids
            if child_id in kept_ids and rng.random() >= cfg.edge_dropout_prob
        )
        parent_id = node.parent_id if node.parent_id in kept_ids else None
        augmented.append(
            _augment_node(
                node,
                graph=graph,
                rng=rng,
                config=cfg,
                transform=transform,
                view_id=view_id,
                parent_id=parent_id,
                child_ids=child_ids,
            )
        )
    augmented.extend(
        _distractor_nodes(
            augmented,
            graph=graph,
            rng=rng,
            config=cfg,
            view_id=view_id,
        )
    )
    if cfg.shuffle_nodes:
        rng.shuffle(augmented)
    visual_transform = {
        "brightness": rng.uniform(-cfg.visual_brightness, cfg.visual_brightness),
        "contrast": 1.0 + rng.uniform(-cfg.visual_contrast, cfg.visual_contrast),
        "channel_scale": [
            1.0 + rng.uniform(-cfg.visual_channel_scale, cfg.visual_channel_scale)
            for _ in range(3)
        ],
    }
    return UIGraph(
        graph_id=f"{graph.graph_id}:{view_id}",
        nodes=tuple(augmented),
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "augmentation": view_id,
            "relation_preserving": True,
            "visual_transform": visual_transform,
            "distractor_count": sum(
                node.origin_id.startswith("__distractor__:") for node in augmented
            ),
        },
    )


def make_training_pair(
    graph: UIGraph,
    *,
    rng: random.Random,
    config: AugmentConfig | None = None,
    matcher_config: MatcherConfig | None = None,
) -> CorrespondencePair | None:
    """Construct identity correspondences between two augmented graph views."""

    cfg = config or AugmentConfig()
    model_cfg = matcher_config or MatcherConfig()
    graph_a = augment_graph(graph, rng=rng, config=cfg, view_id="a")
    graph_b = augment_graph(graph, rng=rng, config=cfg, view_id="b")
    index_a = {
        node.origin_id: index
        for index, node in enumerate(graph_a.nodes)
        if not node.origin_id.startswith("__distractor__:")
    }
    index_b = {
        node.origin_id: index
        for index, node in enumerate(graph_b.nodes)
        if not node.origin_id.startswith("__distractor__:")
    }
    common = tuple(sorted(set(index_a).intersection(index_b)))
    if len(common) < 2:
        return None
    targets_a_to_b = tuple(index_b.get(node.origin_id, -1) for node in graph_a.nodes)
    targets_b_to_a = tuple(index_a.get(node.origin_id, -1) for node in graph_b.nodes)
    return CorrespondencePair(
        graph_a=graph_a,
        graph_b=graph_b,
        encoded_a=encode_graph(graph_a, config=model_cfg),
        encoded_b=encode_graph(graph_b, config=model_cfg),
        targets_a_to_b=targets_a_to_b,
        targets_b_to_a=targets_b_to_a,
        positive_targets_a_to_b=tuple(
            (target,) if target >= 0 else ()
            for target in targets_a_to_b
        ),
        positive_targets_b_to_a=tuple(
            (target,) if target >= 0 else ()
            for target in targets_b_to_a
        ),
        source_indices=tuple(index_a[origin_id] for origin_id in common),
        target_indices=tuple(index_b[origin_id] for origin_id in common),
        origin_ids=common,
    )


def _is_actionable(node: UINode) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def matching_loss(
    model: Any,
    pair: CorrespondencePair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    descriptor_weight: float = 0.20,
    descriptor_temperature: float = 0.10,
    visual_descriptor_weight: float = 1.0,
    visual_descriptor_temperature: float = 0.07,
    strategy_weight: float = 0.10,
    matchability_weight: float = 0.20,
    return_details: bool = False,
) -> Any:
    """Compute the canonical assignment and node-descriptor objective."""

    if descriptor_weight < 0.0:
        raise ValueError("descriptor_weight must be non-negative")
    if descriptor_temperature <= 0.0:
        raise ValueError("descriptor_temperature must be positive")
    if visual_descriptor_weight < 0.0:
        raise ValueError("visual_descriptor_weight must be non-negative")
    if visual_descriptor_temperature <= 0.0:
        raise ValueError("visual_descriptor_temperature must be positive")
    if strategy_weight < 0.0:
        raise ValueError("strategy_weight must be non-negative")
    if matchability_weight < 0.0:
        raise ValueError("matchability_weight must be non-negative")
    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    inputs = matcher_inputs(
        pair.graph_a,
        pair.graph_b,
        config=config,
        device=device,
    )
    output = model(*inputs)
    logits_ab = output["logits_ab"]
    candidates_a = _candidate_indices(pair.graph_a.nodes, config)
    candidates_b = _candidate_indices(pair.graph_b.nodes, config)
    layer_scores = tuple(output.get("assignment_scores_by_layer") or ())
    scores = layer_scores or (logits_ab,)
    layer_losses = tuple(
        0.5
        * (
            _partial_assignment_loss(
                layer,
                pair.positive_targets_a_to_b,
                candidate_indices=candidates_b,
                torch=torch,
                device=device,
            )
            + _partial_assignment_loss(
                layer.T,
                pair.positive_targets_b_to_a,
                candidate_indices=candidates_a,
                torch=torch,
                device=device,
            )
        )
        for layer in scores
    )
    layer_weights = tuple(
        float(value)
        for value in (
            output.get("assignment_loss_weights")
            or (1.0 / len(layer_losses),) * len(layer_losses)
        )
    )
    if len(layer_weights) != len(layer_losses):
        raise ValueError("assignment loss weights must align with model layers")
    if any(value < 0.0 for value in layer_weights) or not any(
        value > 0.0 for value in layer_weights
    ):
        raise ValueError("assignment loss weights must contain a positive value")
    assignment_loss = sum(
        weight * layer_loss
        for weight, layer_loss in zip(layer_weights, layer_losses, strict=True)
    )
    source_matchability = tuple(
        output.get("source_matchability_by_layer") or ()
    )
    target_matchability = tuple(
        output.get("target_matchability_by_layer") or ()
    )
    matchability_loss = assignment_loss.new_zeros(())
    if source_matchability or target_matchability:
        if not (
            len(source_matchability)
            == len(target_matchability)
            == len(layer_weights)
        ):
            raise ValueError("matchability logits must align with model layers")
        per_layer_matchability = tuple(
            0.5
            * (
                _node_matchability_loss(
                    source_logits,
                    pair.targets_a_to_b,
                    torch=torch,
                    device=device,
                )
                + _node_matchability_loss(
                    target_logits,
                    pair.targets_b_to_a,
                    torch=torch,
                    device=device,
                )
            )
            for source_logits, target_logits in zip(
                source_matchability, target_matchability, strict=True
            )
        )
        matchability_loss = sum(
            weight * layer_loss
            for weight, layer_loss in zip(
                layer_weights, per_layer_matchability, strict=True
            )
        )
    descriptor_loss = assignment_loss.new_zeros(())
    visual_descriptor_loss = assignment_loss.new_zeros(())
    source_descriptors = output.get("source_descriptors")
    target_descriptors = output.get("target_descriptors")
    if source_descriptors is not None and target_descriptors is not None:
        normalized_source = torch.nn.functional.normalize(
            source_descriptors,
            dim=-1,
        )
        normalized_target = torch.nn.functional.normalize(
            target_descriptors,
            dim=-1,
        )
        descriptor_affinity = (
            normalized_source @ normalized_target.T
        ) / float(descriptor_temperature)
        descriptor_loss = 0.5 * (
            _partial_assignment_loss(
                descriptor_affinity,
                pair.positive_targets_a_to_b,
                candidate_indices=candidates_b,
                torch=torch,
                device=device,
            )
            + _partial_assignment_loss(
                descriptor_affinity.T,
                pair.positive_targets_b_to_a,
                candidate_indices=candidates_a,
                torch=torch,
                device=device,
            )
        )
    source_visual_descriptors = output.get("source_visual_descriptors")
    target_visual_descriptors = output.get("target_visual_descriptors")
    source_visual_mask = output.get("source_visual_mask")
    target_visual_mask = output.get("target_visual_mask")
    visual_descriptor_trainable = bool(
        config.visual_encoder != DETERMINISTIC_ICON_VISUAL_ENCODER
        and (
            getattr(source_visual_descriptors, "requires_grad", False)
            or getattr(target_visual_descriptors, "requires_grad", False)
        )
    )
    if (
        source_visual_descriptors is not None
        and target_visual_descriptors is not None
        and source_visual_mask is not None
        and target_visual_mask is not None
        and visual_descriptor_trainable
    ):
        normalized_source_visual = torch.nn.functional.normalize(
            source_visual_descriptors,
            dim=-1,
        )
        normalized_target_visual = torch.nn.functional.normalize(
            target_visual_descriptors,
            dim=-1,
        )
        visual_affinity = (
            normalized_source_visual @ normalized_target_visual.T
        ) / float(visual_descriptor_temperature)
        visual_descriptor_loss = 0.5 * (
            _masked_partial_assignment_loss(
                visual_affinity,
                pair.positive_targets_a_to_b,
                candidate_indices=candidates_b,
                source_mask=source_visual_mask,
                target_mask=target_visual_mask,
                torch=torch,
                device=device,
            )
            + _masked_partial_assignment_loss(
                visual_affinity.T,
                pair.positive_targets_b_to_a,
                candidate_indices=candidates_a,
                source_mask=target_visual_mask,
                target_mask=source_visual_mask,
                torch=torch,
                device=device,
            )
        )
    loss = (
        assignment_loss
        + float(matchability_weight) * matchability_loss
        + float(descriptor_weight)
        * (
            descriptor_loss
            + float(visual_descriptor_weight) * visual_descriptor_loss
        )
    )
    strategy_loss = relation_consistency_loss(
        output["source_relation_bases"],
        output["target_relation_bases"],
        model.relation_compatibility,
        pair.positive_targets_a_to_b,
        pair.positive_targets_b_to_a,
        torch=torch,
    )
    loss = loss + float(strategy_weight) * strategy_loss
    if not return_details:
        return loss
    final_loss = layer_losses[-1]
    intermediate_loss = (
        sum(layer_losses[:-1]) / len(layer_losses[:-1])
        if len(layer_losses) > 1
        else final_loss.new_zeros(())
    )
    return loss, {
        "assignment_loss": float(assignment_loss.detach().cpu()),
        "matchability_loss": float(matchability_loss.detach().cpu()),
        "matchability_weight": float(matchability_weight),
        "descriptor_loss": float(descriptor_loss.detach().cpu()),
        "visual_descriptor_loss": float(visual_descriptor_loss.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
        "final_assignment_loss": float(final_loss.detach().cpu()),
        "intermediate_assignment_loss": float(intermediate_loss.detach().cpu()),
        "supervised_layers": float(len(layer_losses)),
        "descriptor_weight": float(descriptor_weight),
        "visual_descriptor_weight": float(visual_descriptor_weight),
        "visual_descriptor_trainable": float(visual_descriptor_trainable),
        "strategy_loss": float(strategy_loss.detach().cpu()),
        "strategy_weight": float(strategy_weight),
    }


def evaluate_correspondence_pairs(
    model: Any,
    pairs: Iterable[CorrespondencePair],
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
    latency_repeats: int = 2,
    prediction_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate bidirectional correspondences and warmed model performance."""

    if latency_repeats < 0:
        raise ValueError("latency_repeats must be non-negative")
    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    pair_list = list(pairs)
    model.to(device)
    model.eval()
    positive_total = 0
    positive_correct = 0
    descriptor_positive_total = 0
    descriptor_positive_correct = 0
    null_total = 0
    null_correct = 0
    recall_hits = {value: 0 for value in ks if value > 0}
    descriptor_recall_hits = {value: 0 for value in ks if value > 0}
    model_latencies: list[float] = []
    input_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    with torch.no_grad():
        for pair in pair_list:
            inputs = matcher_inputs(
                pair.graph_a,
                pair.graph_b,
                config=config,
                device=device,
            )
            output = model(*inputs)
            _synchronize(device, torch)
            for _ in range(latency_repeats):
                _synchronize(device, torch)
                total_started = time.perf_counter()
                input_started = time.perf_counter()
                inputs = matcher_inputs(
                    pair.graph_a,
                    pair.graph_b,
                    config=config,
                    device=device,
                )
                _synchronize(device, torch)
                input_latencies.append((time.perf_counter() - input_started) * 1000.0)
                model_started = time.perf_counter()
                output = model(*inputs)
                _synchronize(device, torch)
                model_latencies.append((time.perf_counter() - model_started) * 1000.0)
                end_to_end_latencies.append(
                    (time.perf_counter() - total_started) * 1000.0
                )
            descriptor_affinity = None
            if (
                output.get("source_descriptors") is not None
                and output.get("target_descriptors") is not None
            ):
                normalized_source = torch.nn.functional.normalize(
                    output["source_descriptors"],
                    dim=-1,
                )
                normalized_target = torch.nn.functional.normalize(
                    output["target_descriptors"],
                    dim=-1,
                )
                descriptor_affinity = normalized_source @ normalized_target.T
            for (
                direction,
                descriptor_logits,
                positive_targets,
                source_nodes,
                candidate_nodes,
                source_graph,
                target_graph,
                component_transpose,
            ) in (
                (
                    "a_to_b",
                    descriptor_affinity,
                    pair.positive_targets_a_to_b,
                    pair.graph_a.nodes,
                    pair.graph_b.nodes,
                    pair.graph_a,
                    pair.graph_b,
                    False,
                ),
                (
                    "b_to_a",
                    (
                        descriptor_affinity.T
                        if descriptor_affinity is not None
                        else None
                    ),
                    pair.positive_targets_b_to_a,
                    pair.graph_b.nodes,
                    pair.graph_a.nodes,
                    pair.graph_b,
                    pair.graph_a,
                    True,
                ),
            ):
                candidate_indices = _candidate_indices(candidate_nodes, config)
                for row_index, target_indices in enumerate(positive_targets):
                    if not target_indices:
                        continue
                    selected_logits, selected_layer = confidence_adaptive_assignment_row(
                        output,
                        source_index=row_index,
                        candidate_indices=candidate_indices,
                        transpose=component_transpose,
                    )
                    ranked_positions = torch.argsort(
                        selected_logits,
                        descending=True,
                    ).tolist()
                    ranked = [
                        candidate_indices[position] for position in ranked_positions
                    ]
                    positive_total += 1
                    positive_correct += int(ranked[0] in target_indices)
                    for value in recall_hits:
                        recall_hits[value] += int(
                            any(target in ranked[:value] for target in target_indices)
                        )
                    if prediction_callback is not None:
                        probabilities = torch.softmax(selected_logits, dim=0)
                        candidate_positions = {
                            candidate_index: position
                            for position, candidate_index in enumerate(
                                candidate_indices
                            )
                        }
                        gold_rank = min(
                            (
                                rank
                                for rank, target_index in enumerate(ranked, 1)
                                if target_index in target_indices
                            ),
                            default=None,
                        )
                        prediction_callback(
                            {
                                "schema_version": (
                                    "omnitransfer.correspondence_prediction.v1"
                                ),
                                "graph_pair": {
                                    "source": (
                                        pair.graph_b.graph_id
                                        if component_transpose
                                        else pair.graph_a.graph_id
                                    ),
                                    "target": (
                                        pair.graph_a.graph_id
                                        if component_transpose
                                        else pair.graph_b.graph_id
                                    ),
                                },
                                "direction": direction,
                                "source": _evaluation_node(
                                    source_nodes[row_index],
                                    index=row_index,
                                ),
                                "gold_targets": [
                                    _evaluation_node(
                                        candidate_nodes[target_index],
                                        index=target_index,
                                    )
                                    for target_index in target_indices
                                ],
                                "prediction": _evaluation_node(
                                    candidate_nodes[ranked[0]],
                                    index=ranked[0],
                                ),
                                "correct": ranked[0] in target_indices,
                                "gold_rank": gold_rank,
                                "top_candidates": [
                                    {
                                        **_evaluation_node(
                                            candidate_nodes[target_index],
                                            index=target_index,
                                        ),
                                        "rank_probability": float(
                                            probabilities[
                                                candidate_positions[target_index]
                                            ]
                                            .detach()
                                            .cpu()
                                        ),
                                        "log_assignment": float(
                                            selected_logits[
                                                candidate_positions[target_index]
                                            ]
                                            .detach()
                                            .cpu()
                                        ),
                                    }
                                    for target_index in ranked[:5]
                                ],
                                "score_components": _evaluation_score_components(
                                    output,
                                    assignment_row=selected_logits,
                                    descriptor_affinity=descriptor_logits,
                                    row_index=row_index,
                                    predicted_index=ranked[0],
                                    gold_indices=target_indices,
                                    transpose=component_transpose,
                                    selected_layer=selected_layer,
                                ),
                            }
                        )
                    if descriptor_logits is None:
                        continue
                    descriptor_ranked_positions = torch.argsort(
                        descriptor_logits[row_index, candidate_indices],
                        descending=True,
                    ).tolist()
                    descriptor_ranked = [
                        candidate_indices[position]
                        for position in descriptor_ranked_positions
                    ]
                    descriptor_positive_total += 1
                    descriptor_positive_correct += int(
                        descriptor_ranked[0] in target_indices
                    )
                    for value in descriptor_recall_hits:
                        descriptor_recall_hits[value] += int(
                            any(
                                target in descriptor_ranked[:value]
                                for target in target_indices
                            )
                        )
    return {
        "schema_version": "omnitransfer_correspondence_metrics_v1",
        "decoder": "confidence_adaptive_last_two",
        "pair_count": len(pair_list),
        "direction_count": len(pair_list) * 2,
        "positive_total": positive_total,
        "null_total": null_total,
        "top1_accuracy": positive_correct / positive_total if positive_total else 0.0,
        "descriptor_positive_total": descriptor_positive_total,
        "descriptor_top1_accuracy": (
            descriptor_positive_correct / descriptor_positive_total
            if descriptor_positive_total
            else 0.0
        ),
        "recall_at_k": {
            value: recall_hits[value] / positive_total if positive_total else 0.0
            for value in recall_hits
        },
        "descriptor_recall_at_k": {
            value: (
                descriptor_recall_hits[value] / descriptor_positive_total
                if descriptor_positive_total
                else 0.0
            )
            for value in descriptor_recall_hits
        },
        "null_accuracy": null_correct / null_total if null_total else 0.0,
        "false_positive_rate": (1.0 - null_correct / null_total if null_total else 0.0),
        "warm_model_latency_ms": {
            "samples": len(model_latencies),
            "p50": _percentile(model_latencies, 50.0),
            "p95": _percentile(model_latencies, 95.0),
            "max": max(model_latencies, default=0.0),
            "under_50ms_rate": (
                sum(value < 50.0 for value in model_latencies) / len(model_latencies)
                if model_latencies
                else 0.0
            ),
        },
        "warm_input_latency_ms": {
            "samples": len(input_latencies),
            "p50": _percentile(input_latencies, 50.0),
            "p95": _percentile(input_latencies, 95.0),
            "max": max(input_latencies, default=0.0),
        },
        "warm_end_to_end_latency_ms": {
            "samples": len(end_to_end_latencies),
            "p50": _percentile(end_to_end_latencies, 50.0),
            "p95": _percentile(end_to_end_latencies, 95.0),
            "max": max(end_to_end_latencies, default=0.0),
            "under_50ms_rate": (
                sum(value < 50.0 for value in end_to_end_latencies)
                / len(end_to_end_latencies)
                if end_to_end_latencies
                else 0.0
            ),
        },
    }


def _evaluation_node(node: UINode, *, index: int) -> dict[str, Any]:
    """Return shortcut-audit-safe node evidence for an evaluation row."""

    return {
        "index": int(index),
        "node_id": node.node_id,
        "text": node.text,
        "content_desc": node.content_desc,
        "class_name": node.class_name,
        "bbox": list(node.bbox) if node.bbox is not None else None,
        "clickable": bool(node.clickable),
        "editable": bool(node.editable),
        "scrollable": bool(node.scrollable),
        "enabled": bool(node.enabled),
    }


def _evaluation_score_components(
    output: dict[str, Any],
    *,
    assignment_row: Any,
    descriptor_affinity: Any | None,
    row_index: int,
    predicted_index: int,
    gold_indices: tuple[int, ...],
    transpose: bool,
    selected_layer: int,
) -> dict[str, dict[str, float]]:
    """Expose how each learned or explicit score treats prediction and gold."""

    raw_components = {"descriptor_affinity": descriptor_affinity}
    model_components = {
        name: output.get(name)
        for name in (
            "affinity",
            "association_score",
        )
    }
    transformer_residuals = tuple(
        output.get("matching_residuals_by_layer") or ()
    )
    if transformer_residuals:
        model_components["transformer_residual"] = transformer_residuals[-1]
    components: dict[str, dict[str, float]] = {}
    assignment_prediction = float(assignment_row[predicted_index].detach().cpu())
    assignment_gold = max(
        float(assignment_row[target_index].detach().cpu())
        for target_index in gold_indices
    )
    components["log_assignment"] = {
        "prediction": assignment_prediction,
        "best_gold": assignment_gold,
        "gold_minus_prediction": assignment_gold - assignment_prediction,
        "selected_association_layer": float(selected_layer + 1),
    }
    for name, values in raw_components.items():
        if values is None or getattr(values, "ndim", 0) != 2:
            continue
        predicted = float(values[row_index, predicted_index].detach().cpu())
        gold = max(
            float(values[row_index, target_index].detach().cpu())
            for target_index in gold_indices
        )
        components[name] = {
            "prediction": predicted,
            "best_gold": gold,
            "gold_minus_prediction": gold - predicted,
        }
    for name, values in model_components.items():
        if values is None or getattr(values, "ndim", 0) != 2:
            continue
        matrix = values.T if transpose else values
        predicted = float(matrix[row_index, predicted_index].detach().cpu())
        gold = max(
            float(matrix[row_index, target_index].detach().cpu())
            for target_index in gold_indices
        )
        components[name] = {
            "prediction": predicted,
            "best_gold": gold,
            "gold_minus_prediction": gold - predicted,
        }
    return components




def train_geometric_v9_matcher(
    graphs: Iterable[UIGraph],
    correspondence_pairs: Iterable[CorrespondencePair],
    *,
    model: Any | None = None,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 17,
    device: str = "cpu",
    augment_config: AugmentConfig | None = None,
    matcher_config: MatcherConfig | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
    checkpoint_callback: (
        Callable[[Any, dict[str, float]], None] | None
    ) = None,
    progress_interval: int = 1000,
    validation_pairs: Iterable[CorrespondencePair] = (),
    synthetic_pairs_per_graph: int = 1,
    cross_page_pair_repeats: int = 1,
    descriptor_weight: float = 0.20,
    descriptor_learning_rate_scale: float = 0.25,
    visual_descriptor_weight: float = 1.0,
    visual_descriptor_temperature: float = 0.07,
    strategy_weight: float = 0.10,
    matchability_weight: float = 0.20,
) -> tuple[Any, list[dict[str, float]]]:
    """Train geometric-v9 on canonical cross-page and synthetic pairs."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if synthetic_pairs_per_graph < 0:
        raise ValueError("synthetic_pairs_per_graph must be non-negative")
    if cross_page_pair_repeats < 0:
        raise ValueError("cross_page_pair_repeats must be non-negative")
    if synthetic_pairs_per_graph == 0 and cross_page_pair_repeats == 0:
        raise ValueError("training requires synthetic or cross-page pairs")
    if descriptor_weight < 0.0:
        raise ValueError("descriptor_weight must be non-negative")
    if visual_descriptor_weight < 0.0:
        raise ValueError("visual_descriptor_weight must be non-negative")
    if visual_descriptor_temperature <= 0.0:
        raise ValueError("visual_descriptor_temperature must be positive")
    if strategy_weight < 0.0:
        raise ValueError("strategy_weight must be non-negative")
    if matchability_weight < 0.0:
        raise ValueError("matchability_weight must be non-negative")
    if not 0.0 < descriptor_learning_rate_scale <= 1.0:
        raise ValueError(
            "descriptor_learning_rate_scale must be in (0, 1]"
        )
    graph_list = list(graphs)
    cross_page_pairs = list(correspondence_pairs)
    dev_pairs = list(validation_pairs)
    if not cross_page_pairs:
        raise ValueError("at least one cross-page correspondence pair is required")
    config = matcher_config or MatcherConfig()
    if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("training supports only the geometric-v9 matcher")
    torch = _require_torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    augmentation = augment_config or AugmentConfig()
    resumed_from_model = model is not None
    matcher = (model or build_geometric_v9_matcher(config)).to(device)
    descriptor_prefixes = (
        "token_embedding",
        "token_projection",
        "text_projection",
        "numeric_projection",
        "xml_projection",
        "missing_",
        "descriptor_",
    )
    descriptor_parameters = []
    context_parameters = []
    for name, parameter in matcher.named_parameters():
        destination = (
            descriptor_parameters
            if name.startswith(descriptor_prefixes)
            else context_parameters
        )
        destination.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": descriptor_parameters,
                "lr": learning_rate,
                "group_name": "descriptor",
            },
            {
                "params": context_parameters,
                "lr": learning_rate,
                "group_name": "context",
            },
        ],
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    best_dev_key: tuple[float, float, float] | None = None
    best_dev_state: dict[str, Any] | None = None
    best_dev_epoch = 0
    if resumed_from_model and dev_pairs:
        baseline_metrics = evaluate_correspondence_pairs(
            matcher,
            dev_pairs,
            device=device,
            matcher_config=config,
            latency_repeats=0,
        )
        best_dev_key = (
            float(baseline_metrics["top1_accuracy"]),
            float(baseline_metrics["recall_at_k"].get(5, 0.0)),
            float("inf"),
        )
        best_dev_state = {
            name: value.detach().cpu().clone()
            for name, value in matcher.state_dict().items()
        }
    for epoch in range(max(1, int(epochs))):
        for parameter_group in optimizer.param_groups:
            if parameter_group.get("group_name") == "descriptor":
                parameter_group["lr"] = (
                    learning_rate * descriptor_learning_rate_scale
                )
            else:
                parameter_group["lr"] = learning_rate
        training_items: list[tuple[str, CorrespondencePair | UIGraph]] = [
            ("cross_page", pair)
            for pair in cross_page_pairs
            for _ in range(cross_page_pair_repeats)
        ]
        training_items.extend(
            ("self_supervised", graph)
            for graph in graph_list
            for _ in range(synthetic_pairs_per_graph)
        )
        rng.shuffle(training_items)
        matcher.train()
        losses: list[float] = []
        loss_details: list[dict[str, float]] = []
        self_supervised_pairs = 0
        aligned_cross_page_pairs = 0
        positive_labels = 0
        for pair_kind, item in training_items:
            if pair_kind == "self_supervised":
                pair = make_training_pair(
                    item,
                    rng=rng,
                    config=augmentation,
                    matcher_config=config,
                )
                if pair is None:
                    continue
            else:
                pair = item
            optimizer.zero_grad(set_to_none=True)
            loss, details = matching_loss(
                matcher,
                pair,
                device=device,
                matcher_config=config,
                descriptor_weight=descriptor_weight,
                visual_descriptor_weight=visual_descriptor_weight,
                visual_descriptor_temperature=visual_descriptor_temperature,
                strategy_weight=strategy_weight,
                matchability_weight=matchability_weight,
                return_details=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            loss_details.append(details)
            self_supervised_pairs += int(pair_kind == "self_supervised")
            aligned_cross_page_pairs += int(pair_kind == "cross_page")
            positive_labels += sum(
                len(targets)
                for targets in (
                    *pair.positive_targets_a_to_b,
                    *pair.positive_targets_b_to_a,
                )
            )
            if progress_callback is not None and len(losses) % progress_interval == 0:
                window = losses[-progress_interval:]
                progress_metrics = {
                    "epoch": float(epoch + 1),
                    "pairs": float(len(losses)),
                    "running_loss": sum(window) / len(window),
                    "self_supervised_pairs": float(self_supervised_pairs),
                    "cross_page_pairs": float(aligned_cross_page_pairs),
                    "positive_labels": float(positive_labels),
                    **_mean_loss_details(loss_details[-progress_interval:]),
                }
                if checkpoint_callback is not None:
                    checkpoint_callback(matcher, dict(progress_metrics))
                progress_callback(progress_metrics)
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "loss": sum(losses) / len(losses),
            "pairs": float(len(losses)),
            "self_supervised_pairs": float(self_supervised_pairs),
            "cross_page_pairs": float(aligned_cross_page_pairs),
            "positive_labels": float(positive_labels),
            "descriptor_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "context_learning_rate": float(
                optimizer.param_groups[1]["lr"]
            ),
            **_mean_loss_details(loss_details),
        }
        if dev_pairs:
            dev_metrics = evaluate_correspondence_pairs(
                matcher,
                dev_pairs,
                device=device,
                matcher_config=config,
                latency_repeats=0,
            )
            epoch_metrics.update(
                {
                    "dev_pairs": float(dev_metrics["pair_count"]),
                    "dev_positive_rows": float(dev_metrics["positive_total"]),
                    "dev_top1_accuracy": float(dev_metrics["top1_accuracy"]),
                    "dev_descriptor_top1_accuracy": float(
                        dev_metrics.get("descriptor_top1_accuracy", 0.0)
                    ),
                    **{
                        f"dev_recall_at_{value}": float(score)
                        for value, score in dev_metrics["recall_at_k"].items()
                    },
                }
            )
            dev_key = (
                float(dev_metrics["top1_accuracy"]),
                float(dev_metrics["recall_at_k"].get(5, 0.0)),
                -float(epoch_metrics["loss"]),
            )
            if best_dev_key is None or dev_key > best_dev_key:
                best_dev_key = dev_key
                best_dev_epoch = epoch + 1
                best_dev_state = {
                    name: value.detach().cpu().clone()
                    for name, value in matcher.state_dict().items()
                }
        history.append(epoch_metrics)
        if checkpoint_callback is not None:
            checkpoint_callback(matcher, dict(epoch_metrics))
        if epoch_callback is not None:
            epoch_callback(dict(epoch_metrics))
    if best_dev_state is not None:
        matcher.load_state_dict(best_dev_state)
        for row in history:
            row["selected_checkpoint"] = float(
                int(row["epoch"]) == best_dev_epoch
            )
            row["selected_pretrained_checkpoint"] = float(
                best_dev_epoch == 0
            )
    matcher.eval()
    return matcher, history


def _select_kept_nodes(
    nodes: tuple[UINode, ...],
    *,
    rng: random.Random,
    config: AugmentConfig,
) -> list[UINode]:
    if not nodes:
        return []
    # Node retention is structural, never actionability-based. Non-clickable
    # labels, icons and containers are first-class context for correspondence.
    kept = [node for node in nodes if rng.random() >= config.drop_node_prob]
    minimum = min(max(1, config.min_nodes), len(nodes))
    if len(kept) < minimum:
        kept_ids = {node.node_id for node in kept}
        missing = [node for node in nodes if node.node_id not in kept_ids]
        rng.shuffle(missing)
        kept.extend(missing[: minimum - len(kept)])
    node_map = {node.node_id: node for node in nodes}
    kept_ids = {node.node_id for node in kept}
    # Retain the immediate parent and adjacent siblings of each sampled node so
    # augmentation does not erase the local relation that the matcher learns.
    for node in tuple(kept):
        parent_id = node.parent_id
        parent = node_map.get(parent_id) if parent_id else None
        if parent is None:
            continue
        kept_ids.add(parent.node_id)
        siblings = [value for value in parent.child_ids if value in node_map]
        if node.node_id in siblings:
            index = siblings.index(node.node_id)
            kept_ids.update(siblings[max(0, index - 1) : index + 2])
    return [node for node in nodes if node.node_id in kept_ids]


def _sample_global_transform(
    rng: random.Random,
    *,
    config: AugmentConfig,
) -> tuple[float, float, float, float]:
    scale_x = 1.0 + rng.uniform(-config.global_scale, config.global_scale)
    scale_y = 1.0 + rng.uniform(-config.global_scale, config.global_scale)
    translate_x = rng.uniform(-config.global_translation, config.global_translation)
    translate_y = rng.uniform(-config.global_translation, config.global_translation)
    return scale_x, scale_y, translate_x, translate_y


def _augment_node(
    node: UINode,
    *,
    graph: UIGraph,
    rng: random.Random,
    config: AugmentConfig,
    transform: tuple[float, float, float, float],
    view_id: str,
    parent_id: str | None,
    child_ids: tuple[str, ...],
) -> UINode:
    mask_text = rng.random() < config.mask_text_prob
    mask_content_desc = rng.random() < config.mask_content_desc_prob
    visual_disabled = bool(node.metadata.get("visual_disabled")) or (
        rng.random() < config.visual_dropout_prob
    )
    semantic_retained = bool(
        (node.text and not mask_text)
        or (node.content_desc and not mask_content_desc)
    )
    if not semantic_retained and visual_disabled:
        if node.text:
            mask_text = False
        elif node.content_desc:
            mask_content_desc = False
    return UINode(
        node_id=f"{view_id}:{node.node_id}",
        origin_id=node.origin_id,
        parent_id=f"{view_id}:{parent_id}" if parent_id else None,
        text="" if mask_text else node.text,
        content_desc=(
            "" if mask_content_desc else node.content_desc
        ),
        resource_id=node.resource_id,
        class_name="" if rng.random() < config.mask_class_prob else node.class_name,
        bbox=_transform_bbox(
            node.bbox,
            graph=graph,
            rng=rng,
            jitter=config.bbox_jitter,
            transform=transform,
        ),
        clickable=node.clickable,
        editable=node.editable,
        scrollable=node.scrollable,
        enabled=node.enabled,
        depth=node.depth,
        child_ids=tuple(f"{view_id}:{child_id}" for child_id in child_ids),
        metadata={
            **node.metadata,
            "augmented_view": view_id,
            "visual_bbox": node.metadata.get("visual_bbox") or node.bbox,
            "visual_disabled": visual_disabled,
        },
    )


def _distractor_nodes(
    nodes: list[UINode],
    *,
    graph: UIGraph,
    rng: random.Random,
    config: AugmentConfig,
    view_id: str,
) -> list[UINode]:
    if config.max_distractors <= 0 or config.distractor_prob <= 0.0:
        return []
    candidates = [node for node in nodes if node.bbox is not None]
    rng.shuffle(candidates)
    distractors: list[UINode] = []
    for source in candidates:
        if len(distractors) >= config.max_distractors:
            break
        if rng.random() >= config.distractor_prob:
            continue
        index = len(distractors)
        distractors.append(
            UINode(
                node_id=f"{view_id}:distractor:{index}",
                origin_id=f"__distractor__:{view_id}:{index}",
                parent_id=None,
                text=source.text,
                content_desc=source.content_desc,
                resource_id="",
                class_name=source.class_name,
                bbox=_shift_bbox(source.bbox, graph=graph, rng=rng),
                clickable=source.clickable,
                editable=source.editable,
                scrollable=source.scrollable,
                enabled=source.enabled,
                depth=source.depth,
                child_ids=(),
                metadata={
                    "augmented_view": view_id,
                    "hard_negative": True,
                    "visual_bbox": source.metadata.get("visual_bbox") or source.bbox,
                },
            )
        )
    return distractors


def _transform_bbox(
    bbox: BBox | None,
    *,
    graph: UIGraph,
    rng: random.Random,
    jitter: float,
    transform: tuple[float, float, float, float],
) -> BBox | None:
    if bbox is None:
        return None
    graph_width = float(graph.width or max(bbox[2], 1.0))
    graph_height = float(graph.height or max(bbox[3], 1.0))
    scale_x, scale_y, translate_x, translate_y = transform
    x1 = (bbox[0] / graph_width - 0.5) * scale_x + 0.5 + translate_x
    y1 = (bbox[1] / graph_height - 0.5) * scale_y + 0.5 + translate_y
    x2 = (bbox[2] / graph_width - 0.5) * scale_x + 0.5 + translate_x
    y2 = (bbox[3] / graph_height - 0.5) * scale_y + 0.5 + translate_y
    if jitter > 0.0:
        x1 += rng.uniform(-jitter, jitter)
        y1 += rng.uniform(-jitter, jitter)
        x2 += rng.uniform(-jitter, jitter)
        y2 += rng.uniform(-jitter, jitter)
    x1, x2 = sorted((_clip01(x1), _clip01(x2)))
    y1, y2 = sorted((_clip01(y1), _clip01(y2)))
    if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
        return bbox
    return x1 * graph_width, y1 * graph_height, x2 * graph_width, y2 * graph_height


def _shift_bbox(
    bbox: BBox | None,
    *,
    graph: UIGraph,
    rng: random.Random,
) -> BBox | None:
    if bbox is None:
        return None
    width = float(graph.width or max(bbox[2], 1.0))
    height = float(graph.height or max(bbox[3], 1.0))
    node_width = bbox[2] - bbox[0]
    node_height = bbox[3] - bbox[1]
    shift_x = rng.uniform(-0.25, 0.25) * width
    shift_y = rng.uniform(-0.25, 0.25) * height
    left = max(0.0, min(width - node_width, bbox[0] + shift_x))
    top = max(0.0, min(height - node_height, bbox[1] + shift_y))
    return left, top, left + node_width, top + node_height


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _partial_assignment_loss(
    logits: Any,
    positive_targets: tuple[tuple[int, ...], ...],
    *,
    candidate_indices: tuple[int, ...],
    torch: Any,
    device: str,
) -> Any:
    supervised_rows = [
        index for index, targets in enumerate(positive_targets) if targets
    ]
    if not supervised_rows:
        raise ValueError("partial assignment has no supervised rows")
    if not candidate_indices:
        raise ValueError("partial assignment has no actionable target candidates")
    candidate_positions = {
        index: position for position, index in enumerate(candidate_indices)
    }
    missing = [
        target
        for index in supervised_rows
        for target in positive_targets[index]
        if target not in candidate_positions
    ]
    if missing:
        raise ValueError("supervised targets must be actionable")
    row_indices = torch.tensor(supervised_rows, dtype=torch.long, device=device)
    column_indices = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    selected_logits = logits[row_indices][:, column_indices]
    log_probabilities = torch.log_softmax(selected_logits, dim=1)
    losses = []
    for row_position, row_index in enumerate(supervised_rows):
        positive_positions = torch.tensor(
            [
                candidate_positions[target]
                for target in positive_targets[row_index]
            ],
            dtype=torch.long,
            device=device,
        )
        losses.append(
            -torch.logsumexp(
                log_probabilities[row_position, positive_positions],
                dim=0,
            )
        )
    return torch.stack(losses).mean()










def _node_matchability_loss(
    logits: Any,
    targets: tuple[int, ...],
    *,
    torch: Any,
    device: str,
) -> Any:
    """Train explicit match/NULL confidence while ignoring unknown rows."""

    if logits.ndim != 1 or logits.shape[0] != len(targets):
        raise ValueError("matchability requires one logit per source node")
    supervised = [index for index, target in enumerate(targets) if target != -2]
    if not supervised:
        return logits.new_zeros(())
    indices = torch.tensor(supervised, dtype=torch.long, device=device)
    labels = torch.tensor(
        [float(targets[index] >= 0) for index in supervised],
        dtype=logits.dtype,
        device=device,
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[indices], labels
    )


def _masked_partial_assignment_loss(
    logits: Any,
    positive_targets: tuple[tuple[int, ...], ...],
    *,
    candidate_indices: tuple[int, ...],
    source_mask: Any,
    target_mask: Any,
    torch: Any,
    device: str,
) -> Any:
    """Apply the visual descriptor objective only to nodes with patches."""

    candidate_positions = {
        index: position for position, index in enumerate(candidate_indices)
    }
    source_available = source_mask.reshape(-1).gt(0.0)
    target_available = target_mask.reshape(-1).gt(0.0)
    supervised_rows: list[int] = []
    valid_targets: dict[int, tuple[int, ...]] = {}
    for source_index, targets in enumerate(positive_targets):
        if not bool(source_available[source_index].detach().cpu()):
            continue
        filtered = tuple(
            target_index
            for target_index in targets
            if target_index in candidate_positions
            and bool(target_available[target_index].detach().cpu())
        )
        if filtered:
            supervised_rows.append(source_index)
            valid_targets[source_index] = filtered
    if not supervised_rows:
        return logits.new_zeros(())
    row_indices = torch.tensor(supervised_rows, dtype=torch.long, device=device)
    column_indices = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    selected_logits = logits[row_indices][:, column_indices]
    log_probabilities = torch.log_softmax(selected_logits, dim=1)
    losses = []
    for row_position, row_index in enumerate(supervised_rows):
        positive_positions = torch.tensor(
            [candidate_positions[target] for target in valid_targets[row_index]],
            dtype=torch.long,
            device=device,
        )
        losses.append(
            -torch.logsumexp(
                log_probabilities[row_position, positive_positions],
                dim=0,
            )
        )
    return torch.stack(losses).mean()


def _mean_loss_details(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "assignment_loss",
        "descriptor_loss",
        "visual_descriptor_loss",
        "total_loss",
        "final_assignment_loss",
        "intermediate_assignment_loss",
        "supervised_layers",
        "descriptor_weight",
        "visual_descriptor_weight",
        "visual_descriptor_trainable",
    )
    return {
        key: sum(row[key] for row in rows) / max(len(rows), 1)
        for key in keys
    }


def _candidate_indices(
    nodes: Iterable[UINode],
    config: MatcherConfig,
) -> tuple[int, ...]:
    node_tuple = tuple(nodes)
    if config.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise ValueError("geometric-v9 requires the all-node candidate policy")
    return tuple(range(len(node_tuple)))


def _synchronize(device: str, torch: Any) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100.0)
    index = max(0, min(len(ordered) - 1, index))
    return float(ordered[index])


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for matcher training. Install omnitransfer[train]."
        ) from exc
    return torch
