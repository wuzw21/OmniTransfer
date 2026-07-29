"""Self-supervised training utilities for relation-aware UI matching."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Callable, Iterable

from omnitransfer.learned_matcher import (
    EncodedGraph,
    MatcherConfig,
    NUMERIC_FEATURE_DIM,
    build_relation_aware_matcher,
    encode_graph,
    matcher_inputs,
)
from omnitransfer.ui_graph import BBox, UIGraph, UINode, multi_anchor_context_graph


FEATURE_DIM = NUMERIC_FEATURE_DIM
DEFAULT_CONTEXT_MASK_PROBABILITY = 0.25


@dataclass(frozen=True)
class AugmentConfig:
    """Controls topology, attribute, and layout perturbations for two UI views."""

    mask_text_prob: float = 0.35
    mask_content_desc_prob: float = 0.35
    mask_class_prob: float = 0.10
    drop_node_prob: float = 0.15
    edge_dropout_prob: float = 0.10
    bbox_jitter: float = 0.02
    global_translation: float = 0.05
    global_scale: float = 0.08
    visual_brightness: float = 0.10
    visual_contrast: float = 0.12
    visual_channel_scale: float = 0.08
    visual_dropout_prob: float = 0.50
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

    @property
    def features_a(self) -> list[list[float]]:
        """Compatibility view of learned numeric node inputs."""

        return [list(values) for values in self.encoded_a.numeric_features]

    @property
    def features_b(self) -> list[list[float]]:
        """Compatibility view of learned numeric node inputs."""

        return [list(values) for values in self.encoded_b.numeric_features]


# Compatibility name retained for existing checkpoints and downstream code.
TrainingPair = CorrespondencePair


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
        if not (
            _is_actionable(graph_a.nodes[source_index])
            and _is_actionable(graph_b.nodes[target_index])
        ):
            raise ValueError("correspondences must be actionable-to-actionable")
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
    if len(graph.nodes) > model_cfg.source_context_nodes:
        actionable_ids = tuple(
            node.node_id for node in graph.nodes if _is_actionable(node)
        )
        if len(actionable_ids) < 2:
            return None
        graph = multi_anchor_context_graph(
            graph,
            anchor_node_ids=actionable_ids,
            max_nodes=max(model_cfg.source_context_nodes, len(actionable_ids)),
        )
    graph_a = augment_graph(graph, rng=rng, config=cfg, view_id="a")
    graph_b = augment_graph(graph, rng=rng, config=cfg, view_id="b")
    index_a = {
        node.origin_id: index
        for index, node in enumerate(graph_a.nodes)
        if not node.origin_id.startswith("__distractor__:") and _is_actionable(node)
    }
    index_b = {
        node.origin_id: index
        for index, node in enumerate(graph_b.nodes)
        if not node.origin_id.startswith("__distractor__:") and _is_actionable(node)
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


def build_lightglue_style_matcher(
    *,
    input_dim: int = FEATURE_DIM,
    hidden_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
) -> Any:
    """Compatibility constructor for the relation-aware cross-attention core."""

    if input_dim != FEATURE_DIM:
        raise ValueError(f"input_dim must be {FEATURE_DIM}, got {input_dim}")
    return build_relation_aware_matcher(
        MatcherConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    )


def matching_loss(
    model: Any,
    pair: CorrespondencePair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    cycle_weight: float = 0.05,
    context_mask_probability: float = 0.0,
    rng: random.Random | None = None,
    return_details: bool = False,
) -> Any:
    """Compute symmetric layer-wise assignment loss on known correspondences."""

    torch = _require_torch()
    if not 0.0 <= context_mask_probability <= 1.0:
        raise ValueError("context_mask_probability must be between zero and one")
    masking_rng = rng or random.Random(0)
    masked_source_indices = tuple(
        index
        for index in pair.source_indices
        if masking_rng.random() < context_mask_probability
    )
    masked_target_indices = tuple(
        index
        for index in pair.target_indices
        if masking_rng.random() < context_mask_probability
    )
    inputs = matcher_inputs(
        pair.graph_a,
        pair.graph_b,
        config=matcher_config or MatcherConfig(),
        device=device,
        source_context_mask_indices=masked_source_indices,
        target_context_mask_indices=masked_target_indices,
    )
    output = model(*inputs)
    logits_ab = output["logits_ab"]
    logits_ba = output["logits_ba"]
    actionable_a = _actionable_indices(pair.graph_a.nodes)
    actionable_b = _actionable_indices(pair.graph_b.nodes)
    layer_scores = tuple(output.get("assignment_scores_by_layer") or ())
    if layer_scores:
        layer_losses = tuple(
            0.5
            * (
                _partial_assignment_loss(
                    scores,
                    pair.positive_targets_a_to_b,
                    candidate_indices=actionable_b,
                    torch=torch,
                    device=device,
                )
                + _partial_assignment_loss(
                    scores.T,
                    pair.positive_targets_b_to_a,
                    candidate_indices=actionable_a,
                    torch=torch,
                    device=device,
                )
            )
            for scores in layer_scores
        )
    else:
        layer_losses = (
            0.5
            * (
                _partial_assignment_loss(
                    logits_ab,
                    pair.positive_targets_a_to_b,
                    candidate_indices=actionable_b,
                    torch=torch,
                    device=device,
                )
                + _partial_assignment_loss(
                    logits_ba,
                    pair.positive_targets_b_to_a,
                    candidate_indices=actionable_a,
                    torch=torch,
                    device=device,
                )
            ),
        )
    assignment_loss = torch.stack(layer_losses).mean()
    loss = assignment_loss
    cycle_loss = assignment_loss.new_zeros(())
    if cycle_weight > 0.0 and pair.source_indices:
        source_indices = torch.tensor(
            pair.source_indices, dtype=torch.long, device=device
        )
        target_indices = torch.tensor(
            pair.target_indices, dtype=torch.long, device=device
        )
        positions_a = {index: position for position, index in enumerate(actionable_a)}
        positions_b = {index: position for position, index in enumerate(actionable_b)}
        probabilities_ab = torch.softmax(logits_ab[:, actionable_b], dim=1)
        probabilities_ba = torch.softmax(logits_ba[:, actionable_a], dim=1)
        forward_positions = torch.tensor(
            [positions_b[index] for index in pair.target_indices],
            dtype=torch.long,
            device=device,
        )
        reverse_positions = torch.tensor(
            [positions_a[index] for index in pair.source_indices],
            dtype=torch.long,
            device=device,
        )
        forward = probabilities_ab[source_indices, forward_positions]
        reverse = probabilities_ba[target_indices, reverse_positions]
        cycle_loss = torch.mean(torch.abs(forward - reverse))
        loss = loss + float(cycle_weight) * cycle_loss
    if return_details:
        intermediate = (
            torch.stack(layer_losses[:-1]).mean()
            if len(layer_losses) > 1
            else assignment_loss.new_zeros(())
        )
        return loss, {
            "assignment_loss": float(assignment_loss.detach().cpu()),
            "final_assignment_loss": float(layer_losses[-1].detach().cpu()),
            "intermediate_assignment_loss": float(intermediate.detach().cpu()),
            "cycle_loss": float(cycle_loss.detach().cpu()),
            "supervised_layers": float(len(layer_losses)),
            "context_masked_nodes": float(
                len(masked_source_indices) + len(masked_target_indices)
            ),
        }
    return loss


def evaluate_correspondence_pairs(
    model: Any,
    pairs: Iterable[CorrespondencePair],
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
    latency_repeats: int = 2,
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
    null_total = 0
    null_correct = 0
    recall_hits = {value: 0 for value in ks if value > 0}
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
            for logits, positive_targets, candidate_nodes in (
                (
                    output["logits_ab"],
                    pair.positive_targets_a_to_b,
                    pair.graph_b.nodes,
                ),
                (
                    output["logits_ba"],
                    pair.positive_targets_b_to_a,
                    pair.graph_a.nodes,
                ),
            ):
                candidate_indices = _actionable_indices(candidate_nodes)
                for row_index, target_indices in enumerate(positive_targets):
                    if not target_indices:
                        continue
                    ranked_positions = torch.argsort(
                        logits[row_index, candidate_indices], descending=True
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
    return {
        "schema_version": "omnitransfer_correspondence_metrics_v1",
        "pair_count": len(pair_list),
        "direction_count": len(pair_list) * 2,
        "positive_total": positive_total,
        "null_total": null_total,
        "top1_accuracy": positive_correct / positive_total if positive_total else 0.0,
        "recall_at_k": {
            value: recall_hits[value] / positive_total if positive_total else 0.0
            for value in recall_hits
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


def evaluate_self_supervised_matcher(
    model: Any,
    graphs: Iterable[UIGraph],
    *,
    seed: int = 29,
    device: str = "cpu",
    augment_config: AugmentConfig | None = None,
    matcher_config: MatcherConfig | None = None,
) -> dict[str, float]:
    """Evaluate positive Top1 and NULL recall on deterministic held-out views."""

    torch = _require_torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = augment_config or AugmentConfig()
    model_cfg = matcher_config or MatcherConfig()
    positive_total = positive_correct = null_total = null_correct = 0
    pair_total = 0
    model.eval()
    with torch.no_grad():
        for graph in graphs:
            pair = make_training_pair(
                graph,
                rng=rng,
                config=cfg,
                matcher_config=model_cfg,
            )
            if pair is None:
                continue
            pair_total += 1
            output = model(
                *matcher_inputs(
                    pair.graph_a,
                    pair.graph_b,
                    config=model_cfg,
                    device=device,
                )
            )
            predictions_ab = output["logits_ab"].argmax(dim=1).tolist()
            predictions_ba = output["logits_ba"].argmax(dim=1).tolist()
            for prediction, target in zip(predictions_ab, pair.targets_a_to_b):
                if target < 0:
                    null_total += 1
                    null_correct += int(prediction == len(pair.graph_b.nodes))
                else:
                    positive_total += 1
                    positive_correct += int(prediction == target)
            for prediction, target in zip(predictions_ba, pair.targets_b_to_a):
                if target < 0:
                    null_total += 1
                    null_correct += int(prediction == len(pair.graph_a.nodes))
                else:
                    positive_total += 1
                    positive_correct += int(prediction == target)
    total = positive_total + null_total
    return {
        "pairs": float(pair_total),
        "top1_accuracy": (positive_correct + null_correct) / total if total else 0.0,
        "positive_top1": positive_correct / positive_total if positive_total else 0.0,
        "null_recall": null_correct / null_total if null_total else 0.0,
        "positive_total": float(positive_total),
        "null_total": float(null_total),
    }


def train_relation_aware_matcher(
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
    cycle_weight: float = 0.05,
    context_mask_probability: float = DEFAULT_CONTEXT_MASK_PROBABILITY,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    progress_interval: int = 1000,
    validation_pairs: Iterable[CorrespondencePair] = (),
) -> tuple[Any, list[dict[str, float]]]:
    """Train one matcher on mixed augmented-view and cross-page pairs."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    graph_list = list(graphs)
    cross_page_pairs = list(correspondence_pairs)
    dev_pairs = list(validation_pairs)
    if not cross_page_pairs:
        raise ValueError("at least one cross-page correspondence pair is required")
    torch = _require_torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    config = matcher_config or MatcherConfig()
    augmentation = augment_config or AugmentConfig()
    matcher = (model or build_relation_aware_matcher(config)).to(device)
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float]] = []
    for epoch in range(max(1, int(epochs))):
        training_items: list[tuple[str, CorrespondencePair | UIGraph]] = [
            ("cross_page", pair) for pair in cross_page_pairs
        ]
        training_items.extend(("self_supervised", graph) for graph in graph_list)
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
                cycle_weight=cycle_weight,
                context_mask_probability=context_mask_probability,
                rng=rng,
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
                progress_callback(
                    {
                        "epoch": float(epoch + 1),
                        "pairs": float(len(losses)),
                        "running_loss": sum(window) / len(window),
                        "self_supervised_pairs": float(self_supervised_pairs),
                        "cross_page_pairs": float(aligned_cross_page_pairs),
                        "positive_labels": float(positive_labels),
                        **_mean_loss_details(loss_details[-progress_interval:]),
                    }
                )
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "loss": sum(losses) / len(losses),
            "pairs": float(len(losses)),
            "self_supervised_pairs": float(self_supervised_pairs),
            "cross_page_pairs": float(aligned_cross_page_pairs),
            "positive_labels": float(positive_labels),
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
                    **{
                        f"dev_recall_at_{value}": float(score)
                        for value, score in dev_metrics["recall_at_k"].items()
                    },
                }
            )
        history.append(epoch_metrics)
    matcher.eval()
    return matcher, history


# OmniTransfer has one optimizer implementation. The descriptive and older
# public names remain aliases so existing experiment commands keep working.
train_omnitransfer_matcher = train_relation_aware_matcher
train_mapping_matcher = train_relation_aware_matcher


def _select_kept_nodes(
    nodes: tuple[UINode, ...],
    *,
    rng: random.Random,
    config: AugmentConfig,
) -> list[UINode]:
    if not nodes:
        return []
    actionable = [node for node in nodes if _is_actionable(node)]
    context = [
        node
        for node in nodes
        if not _is_actionable(node) and rng.random() >= config.drop_node_prob
    ]
    kept = [*actionable, *context]
    minimum = min(max(1, config.min_nodes, len(actionable)), len(nodes))
    if len(kept) < minimum:
        kept_ids = {node.node_id for node in kept}
        missing = [node for node in nodes if node.node_id not in kept_ids]
        rng.shuffle(missing)
        kept.extend(missing[: minimum - len(kept)])
    return kept


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
    return UINode(
        node_id=f"{view_id}:{node.node_id}",
        origin_id=node.origin_id,
        parent_id=f"{view_id}:{parent_id}" if parent_id else None,
        text="" if rng.random() < config.mask_text_prob else node.text,
        content_desc=(
            "" if rng.random() < config.mask_content_desc_prob else node.content_desc
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
            "visual_disabled": bool(node.metadata.get("visual_disabled"))
            or rng.random() < config.visual_dropout_prob,
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


def _mean_loss_details(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "assignment_loss",
        "final_assignment_loss",
        "intermediate_assignment_loss",
        "cycle_loss",
        "supervised_layers",
        "context_masked_nodes",
    )
    return {
        key: sum(row[key] for row in rows) / max(len(rows), 1)
        for key in keys
    }


def _actionable_indices(nodes: Iterable[UINode]) -> tuple[int, ...]:
    return tuple(index for index, node in enumerate(nodes) if _is_actionable(node))


def _synchronize(device: str, torch: Any) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile / 100.0))
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
