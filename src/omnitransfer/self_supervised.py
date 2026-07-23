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
from omnitransfer.ui_graph import BBox, UIGraph, UINode, local_context_graph


FEATURE_DIM = NUMERIC_FEATURE_DIM


@dataclass(frozen=True)
class AugmentConfig:
    """Controls topology, attribute, and layout perturbations for two UI views."""

    mask_text_prob: float = 0.35
    mask_content_desc_prob: float = 0.35
    mask_resource_id_prob: float = 0.50
    mask_class_prob: float = 0.10
    drop_node_prob: float = 0.15
    edge_dropout_prob: float = 0.10
    bbox_jitter: float = 0.02
    global_translation: float = 0.05
    global_scale: float = 0.08
    visual_brightness: float = 0.10
    visual_contrast: float = 0.12
    visual_channel_scale: float = 0.08
    distractor_prob: float = 0.20
    max_distractors: int = 4
    shuffle_nodes: bool = True
    min_nodes: int = 4


@dataclass(frozen=True)
class TrainingPair:
    """Two related graph views with positive, hard-negative, and NULL labels."""

    graph_a: UIGraph
    graph_b: UIGraph
    encoded_a: EncodedGraph
    encoded_b: EncodedGraph
    targets_a_to_b: tuple[int, ...]
    targets_b_to_a: tuple[int, ...]
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


def make_correspondence_training_pair(
    graph_a: UIGraph,
    graph_b: UIGraph,
    correspondences: Iterable[tuple[str, str]],
    *,
    matcher_config: MatcherConfig | None = None,
    allow_empty: bool = False,
    unmatched_as_null: bool = False,
) -> TrainingPair:
    """Construct partial-assignment labels from explicit cross-graph node pairs."""

    config = matcher_config or MatcherConfig()
    index_a = {node.node_id: index for index, node in enumerate(graph_a.nodes)}
    index_b = {node.node_id: index for index, node in enumerate(graph_b.nodes)}
    if allow_empty and not unmatched_as_null:
        raise ValueError("empty correspondence requires explicit NULL supervision")
    unmatched_target = -1 if unmatched_as_null else -2
    targets_a_to_b = [unmatched_target] * len(graph_a.nodes)
    targets_b_to_a = [unmatched_target] * len(graph_b.nodes)
    source_indices: list[int] = []
    target_indices: list[int] = []
    labels: list[str] = []
    for source_id, target_id in correspondences:
        if source_id not in index_a:
            raise ValueError(f"source correspondence node is absent: {source_id}")
        if target_id not in index_b:
            raise ValueError(f"target correspondence node is absent: {target_id}")
        source_index = index_a[source_id]
        target_index = index_b[target_id]
        if targets_a_to_b[source_index] >= 0 or targets_b_to_a[target_index] >= 0:
            raise ValueError("correspondences must be one-to-one")
        targets_a_to_b[source_index] = target_index
        targets_b_to_a[target_index] = source_index
        source_indices.append(source_index)
        target_indices.append(target_index)
        labels.append(f"{source_id}->{target_id}")
    if not source_indices and not allow_empty:
        raise ValueError("at least one correspondence is required")
    return TrainingPair(
        graph_a=graph_a,
        graph_b=graph_b,
        encoded_a=encode_graph(graph_a, config=config),
        encoded_b=encode_graph(graph_b, config=config),
        targets_a_to_b=tuple(targets_a_to_b),
        targets_b_to_a=tuple(targets_b_to_a),
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
) -> TrainingPair | None:
    """Construct labels from known view transformations, including NULL targets."""

    cfg = config or AugmentConfig()
    model_cfg = matcher_config or MatcherConfig()
    if len(graph.nodes) > model_cfg.source_context_nodes:
        graph = local_context_graph(
            graph,
            anchor_node_id=rng.choice(graph.nodes).node_id,
            max_nodes=model_cfg.source_context_nodes,
        )
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
    return TrainingPair(
        graph_a=graph_a,
        graph_b=graph_b,
        encoded_a=encode_graph(graph_a, config=model_cfg),
        encoded_b=encode_graph(graph_b, config=model_cfg),
        targets_a_to_b=targets_a_to_b,
        targets_b_to_a=targets_b_to_a,
        source_indices=tuple(index_a[origin_id] for origin_id in common),
        target_indices=tuple(index_b[origin_id] for origin_id in common),
        origin_ids=common,
    )


def build_lightglue_style_matcher(
    *,
    input_dim: int = FEATURE_DIM,
    hidden_dim: int = 96,
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
    pair: TrainingPair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    cycle_weight: float = 0.05,
) -> Any:
    """Compute symmetric partial-assignment loss with a learned NULL target."""

    torch = _require_torch()
    inputs = matcher_inputs(
        pair.graph_a,
        pair.graph_b,
        config=matcher_config or MatcherConfig(),
        device=device,
    )
    output = model(*inputs)
    logits_ab = output["logits_ab"]
    logits_ba = output["logits_ba"]
    loss_ab = _partial_assignment_loss(
        logits_ab,
        pair.targets_a_to_b,
        null_index=len(pair.graph_b.nodes),
        torch=torch,
        device=device,
    )
    loss_ba = _partial_assignment_loss(
        logits_ba,
        pair.targets_b_to_a,
        null_index=len(pair.graph_a.nodes),
        torch=torch,
        device=device,
    )
    loss = 0.5 * (loss_ab + loss_ba)
    if cycle_weight > 0.0 and pair.source_indices:
        probabilities_ab = torch.softmax(logits_ab, dim=1)
        probabilities_ba = torch.softmax(logits_ba, dim=1)
        source_indices = torch.tensor(
            pair.source_indices, dtype=torch.long, device=device
        )
        target_indices = torch.tensor(
            pair.target_indices, dtype=torch.long, device=device
        )
        forward = probabilities_ab[source_indices, target_indices]
        reverse = probabilities_ba[target_indices, source_indices]
        cycle_loss = torch.mean(torch.abs(forward - reverse))
        loss = loss + float(cycle_weight) * cycle_loss
    return loss


def evaluate_correspondence_pairs(
    model: Any,
    pairs: Iterable[TrainingPair],
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
    latency_repeats: int = 2,
) -> dict[str, Any]:
    """Evaluate bidirectional positive, NULL, and warmed model performance."""

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
                started = time.perf_counter()
                output = model(*inputs)
                _synchronize(device, torch)
                model_latencies.append((time.perf_counter() - started) * 1000.0)
            for logits, targets in (
                (output["logits_ab"], pair.targets_a_to_b),
                (output["logits_ba"], pair.targets_b_to_a),
            ):
                null_index = int(logits.shape[1]) - 1
                for row_index, target_index in enumerate(targets):
                    if target_index == -2:
                        continue
                    ranked = torch.argsort(logits[row_index], descending=True).tolist()
                    if target_index >= 0:
                        positive_total += 1
                        positive_correct += int(ranked[0] == target_index)
                        for value in recall_hits:
                            recall_hits[value] += int(target_index in ranked[:value])
                    else:
                        null_total += 1
                        null_correct += int(ranked[0] == null_index)
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
        "false_positive_rate": (
            1.0 - null_correct / null_total if null_total else 0.0
        ),
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


def train_self_supervised_matcher(
    graphs: list[UIGraph],
    *,
    model: Any | None = None,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    hidden_dim: int = 96,
    num_heads: int = 4,
    num_layers: int = 2,
    seed: int = 13,
    device: str = "cpu",
    augment_config: AugmentConfig | None = None,
    matcher_config: MatcherConfig | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    progress_interval: int = 1000,
) -> tuple[Any, list[dict[str, float]]]:
    """Train the relation-aware matcher on unlabeled UI graphs."""

    torch = _require_torch()
    rng = random.Random(seed)
    cfg = augment_config or AugmentConfig()
    model_cfg = matcher_config or MatcherConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    if model is None:
        model = build_relation_aware_matcher(model_cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    history: list[dict[str, float]] = []
    training_graphs = list(graphs)
    for epoch in range(max(1, int(epochs))):
        model.train()
        rng.shuffle(training_graphs)
        losses: list[float] = []
        positive_labels = null_labels = 0
        for graph in training_graphs:
            pair = make_training_pair(
                graph,
                rng=rng,
                config=cfg,
                matcher_config=model_cfg,
            )
            if pair is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = matching_loss(
                model,
                pair,
                device=device,
                matcher_config=model_cfg,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels = (*pair.targets_a_to_b, *pair.targets_b_to_a)
            positive_labels += sum(target >= 0 for target in labels)
            null_labels += sum(target < 0 for target in labels)
            if progress_callback is not None and len(losses) % progress_interval == 0:
                window = losses[-progress_interval:]
                progress_callback(
                    {
                        "epoch": float(epoch + 1),
                        "pairs": float(len(losses)),
                        "running_loss": sum(window) / len(window),
                        "positive_labels": float(positive_labels),
                        "null_labels": float(null_labels),
                    }
                )
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": sum(losses) / max(len(losses), 1),
                "pairs": float(len(losses)),
                "positive_labels": float(positive_labels),
                "null_labels": float(null_labels),
            }
        )
    model.eval()
    return model, history


def _select_kept_nodes(
    nodes: tuple[UINode, ...],
    *,
    rng: random.Random,
    config: AugmentConfig,
) -> list[UINode]:
    if not nodes:
        return []
    kept = [node for node in nodes if rng.random() >= config.drop_node_prob]
    minimum = min(max(1, config.min_nodes), len(nodes))
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
        resource_id=(
            "" if rng.random() < config.mask_resource_id_prob else node.resource_id
        ),
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
    targets: tuple[int, ...],
    *,
    null_index: int,
    torch: Any,
    device: str,
) -> Any:
    supervised_rows = [index for index, target in enumerate(targets) if target != -2]
    if not supervised_rows:
        raise ValueError("partial assignment has no supervised rows")
    row_indices = torch.tensor(supervised_rows, dtype=torch.long, device=device)
    labels = torch.tensor(
        [targets[index] if targets[index] >= 0 else null_index for index in supervised_rows],
        dtype=torch.long,
        device=device,
    )
    return torch.nn.functional.cross_entropy(logits[row_indices], labels)


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
