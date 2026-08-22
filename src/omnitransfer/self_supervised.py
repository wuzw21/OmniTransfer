"""Self-supervised training utilities for relation-aware UI matching."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    XML_NODE_FEATURE_DIM,
    EncodedGraph,
    MatcherConfig,
    build_geometric_v9_matcher,
    encode_graph,
    matcher_inputs,
    page_inputs,
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
    wrapper_collapse_prob: float = 0.0
    semantic_relocation_prob: float = 0.0
    bbox_jitter: float = 0.02
    global_translation: float = 0.05
    global_scale: float = 0.08
    visual_brightness: float = 0.10
    visual_contrast: float = 0.12
    visual_channel_scale: float = 0.08
    visual_dropout_prob: float = 0.10
    distractor_prob: float = 0.20
    max_distractors: int = 4
    # XML/observation order is evidence.  Order shuffling is an explicit
    # ablation family, never part of the default correspondence augmentation.
    shuffle_nodes: bool = False
    min_nodes: int = 4
    family_curriculum: bool = False


AUGMENTATION_FAMILIES = (
    "semantic",
    "visual",
    "xml_wrapper",
    "layout",
    "order",
)


def _sample_family_augment_config(
    config: AugmentConfig,
    *,
    rng: random.Random,
) -> tuple[AugmentConfig, tuple[str, ...]]:
    draw = rng.random()
    family_count = 1 if draw < 0.4375 else 2 if draw < 0.8125 else 3
    selected = tuple(rng.sample(AUGMENTATION_FAMILIES, family_count))
    selected_set = frozenset(selected)
    sampled = replace(
        config,
        mask_text_prob=(config.mask_text_prob if "semantic" in selected_set else 0.0),
        mask_content_desc_prob=(
            config.mask_content_desc_prob if "semantic" in selected_set else 0.0
        ),
        mask_class_prob=(config.mask_class_prob if "semantic" in selected_set else 0.0),
        semantic_relocation_prob=0.0,
        visual_brightness=(config.visual_brightness if "visual" in selected_set else 0.0),
        visual_contrast=(config.visual_contrast if "visual" in selected_set else 0.0),
        visual_channel_scale=(
            config.visual_channel_scale if "visual" in selected_set else 0.0
        ),
        visual_dropout_prob=(
            config.visual_dropout_prob if "visual" in selected_set else 0.0
        ),
        drop_node_prob=(
            config.drop_node_prob if "xml_wrapper" in selected_set else 0.0
        ),
        edge_dropout_prob=(
            config.edge_dropout_prob if "xml_wrapper" in selected_set else 0.0
        ),
        wrapper_collapse_prob=(
            config.wrapper_collapse_prob
            if "xml_wrapper" in selected_set
            else 0.0
        ),
        bbox_jitter=(config.bbox_jitter if "layout" in selected_set else 0.0),
        global_translation=(
            config.global_translation if "layout" in selected_set else 0.0
        ),
        global_scale=(config.global_scale if "layout" in selected_set else 0.0),
        shuffle_nodes="order" in selected_set,
        distractor_prob=0.0,
        max_distractors=0,
        family_curriculum=False,
    )
    return sampled, selected


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


@dataclass(frozen=True)
class UnlabeledPagePair:
    """Two independently captured pages believed to represent one UI state."""

    graph_a: UIGraph
    graph_b: UIGraph
    pair_id: str = ""
    dataset: str = "unknown"
    sampling_weight: float = 1.0
    group_id: str = ""


def unlabeled_page_pair(
    pair: CorrespondencePair,
    *,
    sampling_weight: float = 1.0,
) -> UnlabeledPagePair:
    """Drop node labels while retaining the real cross-page observations."""

    if sampling_weight <= 0.0:
        raise ValueError("unlabeled pair sampling weight must be positive")
    return UnlabeledPagePair(
        graph_a=pair.graph_a,
        graph_b=pair.graph_b,
        pair_id=str(pair.graph_a.metadata.get("pair_id") or ""),
        dataset=str(pair.graph_a.metadata.get("dataset") or "unknown"),
        sampling_weight=float(sampling_weight),
        group_id=str(pair.graph_a.metadata.get("app_id") or ""),
    )


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
    protected_node_ids: Iterable[str] = (),
) -> UIGraph:
    """Create a structurally perturbed UI view while retaining latent identity."""

    cfg = config or AugmentConfig()
    selected_families: tuple[str, ...] = ()
    if cfg.family_curriculum:
        cfg, selected_families = _sample_family_augment_config(cfg, rng=rng)
    kept_nodes = _select_kept_nodes(
        graph.nodes,
        rng=rng,
        config=cfg,
        protected_node_ids=frozenset(protected_node_ids),
    )
    kept_ids = {node.node_id for node in kept_nodes}
    parent_ids = {
        node.node_id: node.parent_id if node.parent_id in kept_ids else None
        for node in kept_nodes
    }
    children_by_parent: dict[str, list[str]] = {}
    for node_id, parent_id in parent_ids.items():
        if parent_id is not None:
            children_by_parent.setdefault(parent_id, []).append(node_id)
    for wrapper in kept_nodes:
        wrapper_parent = parent_ids[wrapper.node_id]
        wrapper_children = children_by_parent.get(wrapper.node_id, ())
        if (
            wrapper_parent is None
            or not wrapper_children
            or rng.random() >= cfg.wrapper_collapse_prob
        ):
            continue
        for child_id in wrapper_children:
            parent_ids[child_id] = wrapper_parent
    for node_id, parent_id in tuple(parent_ids.items()):
        if parent_id is not None and rng.random() < cfg.edge_dropout_prob:
            parent_ids[node_id] = None
    children_by_parent = {}
    for node_id, parent_id in parent_ids.items():
        if parent_id is not None:
            children_by_parent.setdefault(parent_id, []).append(node_id)
    semantic_values = {
        node.node_id: (node.text, node.content_desc)
        for node in kept_nodes
    }
    nodes_by_id = {node.node_id: node for node in kept_nodes}
    for node in kept_nodes:
        parent_id = node.parent_id
        if (
            parent_id not in nodes_by_id
            or not (node.text or node.content_desc)
            or rng.random() >= cfg.semantic_relocation_prob
        ):
            continue
        parent_text, parent_desc = semantic_values[parent_id]
        if parent_text or parent_desc:
            continue
        semantic_values[parent_id] = (node.text, node.content_desc)
        semantic_values[node.node_id] = ("", "")
    transform = _sample_global_transform(rng, config=cfg)
    augmented: list[UINode] = []
    for node in kept_nodes:
        child_ids = tuple(children_by_parent.get(node.node_id, ()))
        parent_id = parent_ids[node.node_id]
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
                semantic_values=semantic_values[node.node_id],
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
            "augmentation_families": selected_families,
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


def augment_correspondence_pair(
    pair: CorrespondencePair,
    *,
    rng: random.Random,
    config: AugmentConfig,
    matcher_config: MatcherConfig,
    view_id: str,
) -> CorrespondencePair:
    """Corrupt both observed pages while retaining every labelled endpoint."""

    protected_a = {
        pair.graph_a.nodes[index].node_id
        for index, targets in enumerate(pair.positive_targets_a_to_b)
        if targets
    }
    protected_b = {
        pair.graph_b.nodes[index].node_id
        for index, targets in enumerate(pair.positive_targets_b_to_a)
        if targets
    }
    view_config = config
    selected_families: tuple[str, ...] = ()
    if config.family_curriculum:
        view_config, selected_families = _sample_family_augment_config(
            config,
            rng=rng,
        )
    graph_a = augment_graph(
        pair.graph_a,
        rng=rng,
        config=view_config,
        view_id=f"{view_id}_a",
        protected_node_ids=protected_a,
    )
    graph_b = augment_graph(
        pair.graph_b,
        rng=rng,
        config=view_config,
        view_id=f"{view_id}_b",
        protected_node_ids=protected_b,
    )
    if selected_families:
        graph_a = replace(
            graph_a,
            metadata={
                **graph_a.metadata,
                "augmentation_families": selected_families,
            },
        )
        graph_b = replace(
            graph_b,
            metadata={
                **graph_b.metadata,
                "augmentation_families": selected_families,
            },
        )
    current_a = {
        str(node.metadata.get("original_node_id")): node.node_id
        for node in graph_a.nodes
        if node.metadata.get("original_node_id")
    }
    current_b = {
        str(node.metadata.get("original_node_id")): node.node_id
        for node in graph_b.nodes
        if node.metadata.get("original_node_id")
    }
    correspondences = []
    expected_edges = set()
    for source_index, target_indices in enumerate(pair.positive_targets_a_to_b):
        source_id = pair.graph_a.nodes[source_index].node_id
        for target_index in target_indices:
            target_id = pair.graph_b.nodes[target_index].node_id
            expected_edges.add((source_id, target_id))
            correspondences.append((current_a[source_id], current_b[target_id]))
    if len(correspondences) != len(expected_edges):
        raise ValueError("cross-page augmentation changed the labelled edge set")
    return make_correspondence_training_pair(
        graph_a,
        graph_b,
        correspondences,
        matcher_config=matcher_config,
    )


def augment_unlabeled_page_pair(
    pair: UnlabeledPagePair,
    *,
    rng: random.Random,
    config: AugmentConfig | None = None,
) -> UnlabeledPagePair:
    """Independently perturb two real pages without inventing correspondences."""

    augmentation = config or AugmentConfig()
    return UnlabeledPagePair(
        graph_a=augment_graph(
            pair.graph_a,
            rng=rng,
            config=augmentation,
            view_id="cross_a",
        ),
        graph_b=augment_graph(
            pair.graph_b,
            rng=rng,
            config=augmentation,
            view_id="cross_b",
        ),
        pair_id=pair.pair_id,
        dataset=pair.dataset,
        sampling_weight=pair.sampling_weight,
        group_id=pair.group_id,
    )


def _is_actionable(node: UINode) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def _active_state_reconstruction_loss(
    output: dict[str, Any],
    *,
    torch: Any,
) -> tuple[Any, int]:
    """Decode only explicitly observed page-state fields from the active subspace."""

    losses = []
    observed_labels = 0
    reference = output["logits_ab"]
    for prefix in ("source", "target"):
        logits = output.get(f"{prefix}_active_state_logits")
        targets = output.get(f"{prefix}_active_state_targets")
        mask = output.get(f"{prefix}_active_state_mask")
        if logits is None or targets is None or mask is None:
            continue
        observed = mask.to(dtype=torch.bool)
        observed_labels += int(observed.sum().detach().cpu())
        if observed.any():
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[observed],
                    targets[observed].to(dtype=logits.dtype),
                )
            )
    if not losses:
        return reference.sum() * 0.0, observed_labels
    return torch.stack(losses).mean(), observed_labels


def matching_loss(
    model: Any,
    pair: CorrespondencePair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    descriptor_weight: float = 0.0,
    descriptor_temperature: float = 0.10,
    visual_descriptor_weight: float = 0.0,
    visual_descriptor_temperature: float = 0.07,
    strategy_weight: float = 0.0,
    matchability_weight: float = 0.20,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.5,
    refinement_stability_weight: float = 0.0,
    active_state_weight: float = 0.0,
    page_embedding_weight: float = 0.0,
    page_embedding_negative_pair: CorrespondencePair | None = None,
    return_details: bool = False,
) -> Any:
    """Compute the canonical assignment and node-descriptor objective."""

    if any(
        value != 0.0
        for value in (
            descriptor_weight,
            visual_descriptor_weight,
            strategy_weight,
            hard_negative_weight,
            refinement_stability_weight,
            active_state_weight,
            page_embedding_weight,
        )
    ):
        raise ValueError(
            "router-anchor training supports only assignment and NULL losses"
        )
    if matchability_weight != 0.20:
        raise ValueError("router-anchor NULL loss weight is fixed at 0.20")
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
    if hard_negative_weight < 0.0:
        raise ValueError("hard_negative_weight must be non-negative")
    if hard_negative_margin < 0.0:
        raise ValueError("hard_negative_margin must be non-negative")
    if refinement_stability_weight < 0.0:
        raise ValueError("refinement_stability_weight must be non-negative")
    if active_state_weight < 0.0:
        raise ValueError("active_state_weight must be non-negative")
    if page_embedding_weight < 0.0:
        raise ValueError("page_embedding_weight must be non-negative")
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
    hard_ab, stability_ab = _refinement_objective(
        scores,
        pair.positive_targets_a_to_b,
        candidate_indices=candidates_b,
        hard_negative_margin=hard_negative_margin,
        torch=torch,
        device=device,
    )
    hard_ba, stability_ba = _refinement_objective(
        tuple(layer.T for layer in scores),
        pair.positive_targets_b_to_a,
        candidate_indices=candidates_a,
        hard_negative_margin=hard_negative_margin,
        torch=torch,
        device=device,
    )
    hard_negative_loss = 0.5 * (hard_ab + hard_ba)
    refinement_stability_loss = 0.5 * (stability_ab + stability_ba)
    active_state_loss, active_state_observed_labels = (
        _active_state_reconstruction_loss(output, torch=torch)
    )
    source_matchability = tuple(
        output.get("source_matchability_by_layer") or ()
    )
    target_matchability = tuple(
        output.get("target_matchability_by_layer") or ()
    )
    matchability_loss = assignment_loss.new_zeros(())
    if matchability_weight > 0.0 and (
        source_matchability or target_matchability
    ):
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
    if (
        descriptor_weight > 0.0
        and source_descriptors is not None
        and target_descriptors is not None
    ):
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
        (
            getattr(source_visual_descriptors, "requires_grad", False)
            or getattr(target_visual_descriptors, "requires_grad", False)
        )
    )
    if (
        descriptor_weight > 0.0
        and visual_descriptor_weight > 0.0
        and source_visual_descriptors is not None
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
        + float(hard_negative_weight) * hard_negative_loss
        + float(refinement_stability_weight) * refinement_stability_loss
        + float(active_state_weight) * active_state_loss
        + float(matchability_weight) * matchability_loss
        + float(descriptor_weight)
        * (
            descriptor_loss
            + float(visual_descriptor_weight) * visual_descriptor_loss
        )
    )
    strategy_loss = assignment_loss.new_zeros(())
    if strategy_weight > 0.0:
        strategy_loss = relation_consistency_loss(
            output["source_relation_bases"],
            output["target_relation_bases"],
            model.relation_compatibility,
            pair.positive_targets_a_to_b,
            pair.positive_targets_b_to_a,
            torch=torch,
        )
    loss = loss + float(strategy_weight) * strategy_loss
    page_loss = assignment_loss.new_zeros(())
    page_positive_cosine = assignment_loss.new_zeros(())
    page_negative_cosine = assignment_loss.new_zeros(())
    if page_embedding_weight > 0.0:
        if page_embedding_negative_pair is None:
            raise ValueError(
                "page embedding loss requires a negative correspondence pair"
            )
        negative_output = model(
            *matcher_inputs(
                page_embedding_negative_pair.graph_a,
                page_embedding_negative_pair.graph_b,
                config=config,
                device=device,
            )
        )
        (
            page_loss,
            page_positive_cosine,
            page_negative_cosine,
        ) = _page_embedding_contrastive_loss_from_outputs(
            output,
            negative_output,
            torch=torch,
        )
        loss = loss + float(page_embedding_weight) * page_loss
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
        "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
        "hard_negative_weight": float(hard_negative_weight),
        "hard_negative_margin": float(hard_negative_margin),
        "refinement_stability_loss": float(
            refinement_stability_loss.detach().cpu()
        ),
        "refinement_stability_weight": float(refinement_stability_weight),
        "active_state_reconstruction_loss": float(
            active_state_loss.detach().cpu()
        ),
        "active_state_weight": float(active_state_weight),
        "active_state_observed_labels": float(active_state_observed_labels),
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
        "page_embedding_loss": float(page_loss.detach().cpu()),
        "page_embedding_weight": float(page_embedding_weight),
        "page_embedding_positive_cosine": float(
            page_positive_cosine.detach().cpu()
        ),
        "page_embedding_negative_cosine": float(
            page_negative_cosine.detach().cpu()
        ),
    }


def cross_page_soft_consistency_loss(
    student: Any,
    teacher: Any,
    pair: UnlabeledPagePair,
    *,
    rng: random.Random,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    augment_config: AugmentConfig | None = None,
    confidence_threshold: float = 0.20,
    margin_threshold: float = 0.02,
    temperature: float = 1.0,
    min_stable_layers: int = 2,
    return_details: bool = False,
) -> Any:
    """Distill bidirectional soft matches between two real page observations."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("teacher confidence threshold must be in [0, 1]")
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("teacher margin threshold must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("teacher temperature must be positive")
    if min_stable_layers <= 0:
        raise ValueError("teacher stable layers must be positive")
    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    augmented = augment_unlabeled_page_pair(
        pair,
        rng=rng,
        config=augment_config,
    )
    with torch.no_grad():
        teacher_output = teacher(
            *matcher_inputs(
                pair.graph_a,
                pair.graph_b,
                config=config,
                device=device,
            )
        )
    student_output = student(
        *matcher_inputs(
            augmented.graph_a,
            augmented.graph_b,
            config=config,
            device=device,
        )
    )
    loss_ab, details_ab = _soft_direction_consistency_loss(
        teacher_output,
        student_output,
        clean_source=pair.graph_a,
        clean_target=pair.graph_b,
        augmented_source=augmented.graph_a,
        augmented_target=augmented.graph_b,
        transpose=False,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        temperature=temperature,
        min_stable_layers=min_stable_layers,
        matcher_config=config,
        torch=torch,
        device=device,
    )
    loss_ba, details_ba = _soft_direction_consistency_loss(
        teacher_output,
        student_output,
        clean_source=pair.graph_b,
        clean_target=pair.graph_a,
        augmented_source=augmented.graph_b,
        augmented_target=augmented.graph_a,
        transpose=True,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        temperature=temperature,
        min_stable_layers=min_stable_layers,
        matcher_config=config,
        torch=torch,
        device=device,
    )
    available = [
        loss
        for loss, details in ((loss_ab, details_ab), (loss_ba, details_ba))
        if details["soft_teacher_rows"] > 0.0
    ]
    if not available:
        loss = student_output["logits_ab"].sum() * 0.0
    else:
        loss = sum(available) / len(available)
    row_count = details_ab["soft_teacher_rows"] + details_ba["soft_teacher_rows"]
    details = {
        "assignment_loss": 0.0,
        "descriptor_loss": 0.0,
        "visual_descriptor_loss": 0.0,
        "total_loss": float(loss.detach().cpu()),
        "final_assignment_loss": 0.0,
        "intermediate_assignment_loss": 0.0,
        "supervised_layers": 0.0,
        "soft_teacher_rows": row_count,
        "soft_teacher_candidate_rows": (
            details_ab["soft_teacher_candidate_rows"]
            + details_ba["soft_teacher_candidate_rows"]
        ),
        "soft_teacher_mean_confidence": (
            (
                details_ab["soft_teacher_mean_confidence"]
                * details_ab["soft_teacher_rows"]
                + details_ba["soft_teacher_mean_confidence"]
                * details_ba["soft_teacher_rows"]
            )
            / max(row_count, 1.0)
        ),
        "soft_teacher_mean_margin": (
            (
                details_ab["soft_teacher_mean_margin"]
                * details_ab["soft_teacher_rows"]
                + details_ba["soft_teacher_mean_margin"]
                * details_ba["soft_teacher_rows"]
            )
            / max(row_count, 1.0)
        ),
        "soft_teacher_bidirectional_rows": (
            details_ab["soft_teacher_bidirectional_rows"]
            + details_ba["soft_teacher_bidirectional_rows"]
        ),
        "soft_teacher_stable_rows": (
            details_ab["soft_teacher_stable_rows"]
            + details_ba["soft_teacher_stable_rows"]
        ),
    }
    if return_details:
        return loss, details
    return loss


def _soft_direction_consistency_loss(
    teacher_output: dict[str, Any],
    student_output: dict[str, Any],
    *,
    clean_source: UIGraph,
    clean_target: UIGraph,
    augmented_source: UIGraph,
    augmented_target: UIGraph,
    transpose: bool,
    confidence_threshold: float,
    margin_threshold: float,
    temperature: float,
    min_stable_layers: int,
    matcher_config: MatcherConfig,
    torch: Any,
    device: str,
) -> tuple[Any, dict[str, float]]:
    clean_source_candidates = _candidate_indices(clean_source.nodes, matcher_config)
    clean_target_candidates = _candidate_indices(clean_target.nodes, matcher_config)
    augmented_source_candidates = set(
        _candidate_indices(augmented_source.nodes, matcher_config)
    )
    augmented_target_candidates = set(
        _candidate_indices(augmented_target.nodes, matcher_config)
    )
    source_retained = _retained_augmented_indices(augmented_source)
    target_retained = _retained_augmented_indices(augmented_target)
    teacher_layers = tuple(teacher_output.get("assignment_scores_by_layer") or ())
    student_layers = tuple(student_output.get("assignment_scores_by_layer") or ())
    if not teacher_layers or not student_layers:
        raise ValueError("soft correspondence training requires assignment layers")
    if transpose:
        teacher_layers = tuple(layer.T for layer in teacher_layers)
        student_layers = tuple(layer.T for layer in student_layers)
    teacher_logits = teacher_layers[-1]
    student_logits = student_layers[-1]
    source_candidate_positions = {
        index: position for position, index in enumerate(clean_source_candidates)
    }
    source_tensor = torch.tensor(
        clean_source_candidates,
        dtype=torch.long,
        device=device,
    )
    target_tensor = torch.tensor(
        clean_target_candidates,
        dtype=torch.long,
        device=device,
    )
    candidate_logits = teacher_logits[source_tensor][:, target_tensor]
    row_probabilities = torch.softmax(candidate_logits / temperature, dim=1)
    column_probabilities = torch.softmax(candidate_logits / temperature, dim=0)
    mutual = torch.sqrt(
        (row_probabilities * column_probabilities).clamp_min(1e-12)
    )
    mutual = mutual / mutual.sum(dim=1, keepdim=True).clamp_min(1e-12)
    top_values, top_positions = mutual.max(dim=1)
    if mutual.shape[1] > 1:
        top_two = mutual.topk(k=2, dim=1).values
        margins = top_two[:, 0] - top_two[:, 1]
    else:
        margins = top_values
    reverse_top_positions = column_probabilities.argmax(dim=0)
    layer_window = teacher_layers[-min(min_stable_layers, len(teacher_layers)) :]
    stable_rows = torch.ones(
        len(clean_source_candidates),
        dtype=torch.bool,
        device=device,
    )
    for layer in layer_window:
        layer_logits = layer[source_tensor][:, target_tensor]
        stable_rows &= layer_logits.argmax(dim=1).eq(top_positions)
        stable_rows &= layer_logits.argmax(dim=0)[top_positions].eq(
            torch.arange(len(clean_source_candidates), device=device)
        )
    bidirectional_rows = reverse_top_positions[top_positions].eq(
        torch.arange(len(clean_source_candidates), device=device)
    )
    eligible = (
        top_values.ge(confidence_threshold)
        & margins.ge(margin_threshold)
        & bidirectional_rows
        & stable_rows
    )
    selected_clean_rows: list[int] = []
    selected_augmented_rows: list[int] = []
    selected_clean_columns: list[int] = []
    selected_augmented_columns: list[int] = []
    for clean_target_index in clean_target_candidates:
        original_id = clean_target.nodes[clean_target_index].node_id
        augmented_index = target_retained.get(original_id)
        if augmented_index is None or augmented_index not in augmented_target_candidates:
            continue
        selected_clean_columns.append(clean_target_index)
        selected_augmented_columns.append(augmented_index)
    clean_column_positions = {
        index: position for position, index in enumerate(clean_target_candidates)
    }
    retained_top_rows = []
    for clean_source_index in clean_source_candidates:
        source_position = source_candidate_positions[clean_source_index]
        if not bool(eligible[source_position].item()):
            continue
        augmented_index = source_retained.get(clean_source.nodes[clean_source_index].node_id)
        if augmented_index is None or augmented_index not in augmented_source_candidates:
            continue
        top_clean_target = clean_target_candidates[
            int(top_positions[source_position].item())
        ]
        if top_clean_target not in selected_clean_columns:
            continue
        selected_clean_rows.append(clean_source_index)
        selected_augmented_rows.append(augmented_index)
        retained_top_rows.append(source_position)
    zero = student_logits.sum() * 0.0
    if not selected_clean_rows or not selected_clean_columns:
        return zero, {
            "soft_teacher_rows": 0.0,
            "soft_teacher_candidate_rows": float(len(clean_source_candidates)),
            "soft_teacher_mean_confidence": 0.0,
            "soft_teacher_mean_margin": 0.0,
            "soft_teacher_bidirectional_rows": float(bidirectional_rows.sum().item()),
            "soft_teacher_stable_rows": float(stable_rows.sum().item()),
        }
    clean_row_positions = torch.tensor(
        retained_top_rows,
        dtype=torch.long,
        device=device,
    )
    clean_column_position_tensor = torch.tensor(
        [clean_column_positions[index] for index in selected_clean_columns],
        dtype=torch.long,
        device=device,
    )
    soft_targets = mutual[clean_row_positions][
        :, clean_column_position_tensor
    ]
    soft_targets = soft_targets / soft_targets.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    student_rows = torch.tensor(
        selected_augmented_rows,
        dtype=torch.long,
        device=device,
    )
    student_columns = torch.tensor(
        selected_augmented_columns,
        dtype=torch.long,
        device=device,
    )
    student_log_probabilities = torch.log_softmax(
        student_logits[student_rows][:, student_columns] / temperature,
        dim=1,
    )
    confidences = top_values[clean_row_positions]
    selected_margins = margins[clean_row_positions]
    row_weights = confidences
    pairwise_layers = tuple(teacher_output.get("pairwise_support_by_layer") or ())
    if pairwise_layers:
        support = pairwise_layers[-1].T if transpose else pairwise_layers[-1]
        support_rows = torch.tensor(
            selected_clean_rows,
            dtype=torch.long,
            device=device,
        )
        support_columns = torch.tensor(
            selected_clean_columns,
            dtype=torch.long,
            device=device,
        )
        selected_support = support[support_rows][:, support_columns]
        top_in_retained = soft_targets.argmax(dim=1, keepdim=True)
        top_support = selected_support.gather(1, top_in_retained).squeeze(1)
        expected_support = (soft_targets * selected_support).sum(dim=1)
        relation_reliability = torch.sigmoid(top_support - expected_support)
        row_weights = row_weights * (0.5 + 0.5 * relation_reliability)
    row_losses = -(soft_targets.detach() * student_log_probabilities).sum(dim=1)
    loss = (row_weights.detach() * row_losses).sum() / row_weights.sum().clamp_min(
        1e-12
    )
    return loss, {
        "soft_teacher_rows": float(len(selected_clean_rows)),
        "soft_teacher_candidate_rows": float(len(clean_source_candidates)),
        "soft_teacher_mean_confidence": float(confidences.mean().detach().cpu()),
        "soft_teacher_mean_margin": float(selected_margins.mean().detach().cpu()),
        "soft_teacher_bidirectional_rows": float(bidirectional_rows.sum().item()),
        "soft_teacher_stable_rows": float(stable_rows.sum().item()),
    }


def _retained_augmented_indices(graph: UIGraph) -> dict[str, int]:
    return {
        str(node.metadata["original_node_id"]): index
        for index, node in enumerate(graph.nodes)
        if "original_node_id" in node.metadata
    }


def page_embedding_contrastive_loss(
    model: Any,
    pair: CorrespondencePair,
    negative_pair: CorrespondencePair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    return_details: bool = False,
) -> Any:
    """Align positive page pairs while separating a deterministic negative."""

    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    output = model(
        *matcher_inputs(
            pair.graph_a,
            pair.graph_b,
            config=config,
            device=device,
        )
    )
    negative_output = model(
        *matcher_inputs(
            negative_pair.graph_a,
            negative_pair.graph_b,
            config=config,
            device=device,
        )
    )
    loss, positive_cosine, negative_cosine = (
        _page_embedding_contrastive_loss_from_outputs(
            output,
            negative_output,
            torch=torch,
        )
    )
    if not return_details:
        return loss
    return loss, {
        "page_embedding_loss": float(loss.detach().cpu()),
        "page_embedding_positive_cosine": float(positive_cosine.detach().cpu()),
        "page_embedding_negative_cosine": float(negative_cosine.detach().cpu()),
    }


def unlabeled_page_embedding_contrastive_loss(
    model: Any,
    pair: UnlabeledPagePair,
    negative_pair: UnlabeledPagePair,
    *,
    rng: random.Random,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    augment_config: AugmentConfig | None = None,
    active_state_weight: float = 0.0,
    return_details: bool = False,
) -> Any:
    """Train page identity from paired observations without node targets."""

    torch = _require_torch()
    if active_state_weight < 0.0:
        raise ValueError("active_state_weight must be non-negative")
    config = matcher_config or MatcherConfig()
    positive = augment_unlabeled_page_pair(pair, rng=rng, config=augment_config)
    negative = augment_unlabeled_page_pair(
        negative_pair,
        rng=rng,
        config=augment_config,
    )
    positive_output = model(
        *matcher_inputs(
            positive.graph_a,
            positive.graph_b,
            config=config,
            device=device,
        )
    )
    negative_output = model(
        *matcher_inputs(
            negative.graph_a,
            negative.graph_b,
            config=config,
            device=device,
        )
    )
    loss, positive_cosine, negative_cosine = (
        _page_embedding_contrastive_loss_from_outputs(
            positive_output,
            negative_output,
            torch=torch,
        )
    )
    active_state_loss, active_state_observed_labels = (
        _active_state_reconstruction_loss(positive_output, torch=torch)
    )
    loss = loss + float(active_state_weight) * active_state_loss
    if not return_details:
        return loss
    return loss, {
        "assignment_loss": 0.0,
        "descriptor_loss": 0.0,
        "visual_descriptor_loss": 0.0,
        "total_loss": float(loss.detach().cpu()),
        "page_embedding_loss": float(loss.detach().cpu()),
        "page_embedding_positive_cosine": float(positive_cosine.detach().cpu()),
        "page_embedding_negative_cosine": float(negative_cosine.detach().cpu()),
        "unlabeled_page_embedding_loss": float(loss.detach().cpu()),
        "active_state_reconstruction_loss": float(
            active_state_loss.detach().cpu()
        ),
        "active_state_weight": float(active_state_weight),
        "active_state_observed_labels": float(active_state_observed_labels),
    }


def batch_unlabeled_page_embedding_contrastive_loss(
    model: Any,
    pairs: Iterable[UnlabeledPagePair],
    *,
    rng: random.Random,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    augment_config: AugmentConfig | None = None,
    temperature: float = 0.1,
    bridge_weight: float = 0.0,
    active_state_weight: float = 0.0,
    return_details: bool = False,
) -> Any:
    """Train local correspondence from page-pair identity alone.

    Every source page is compared with every target page in the batch.  The
    diagonal is the only supervision.  Each page score is aggregated from the
    model's single local node-score matrix, so this one cross-entropy trains
    node appearance, local relations, page representation, and localisation
    without constructing node pseudo-labels.
    """

    if temperature <= 0.0:
        raise ValueError("page InfoNCE temperature must be positive")
    if bridge_weight != 0.0 or active_state_weight != 0.0:
        raise ValueError("page-local training has no auxiliary losses")
    pair_list = list(pairs)
    if len(pair_list) < 2:
        raise ValueError("batch page InfoNCE requires at least two page pairs")
    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    augmented_pairs = []
    for pair in pair_list:
        augmented_pairs.append(
            augment_unlabeled_page_pair(pair, rng=rng, config=augment_config)
        )
    source_pages = [
        model.encode_page(
            *page_inputs(
                pair.graph_a,
                config=config,
                device=device,
            )
        )
        for pair in augmented_pairs
    ]
    target_pages = [
        model.encode_page(
            *page_inputs(
                pair.graph_b,
                config=config,
                device=device,
            )
        )
        for pair in augmented_pairs
    ]
    score_rows = []
    for source_page in source_pages:
        scores = []
        for target_page in target_pages:
            output = model.match_pages(source_page, target_page)
            scores.append(output["page_pair_score"])
        score_rows.append(torch.stack(scores))
    page_scores = torch.stack(score_rows)
    logits = page_scores / float(temperature)
    labels = torch.arange(len(pair_list), dtype=torch.long, device=logits.device)
    loss = 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )
    if not return_details:
        return loss
    diagonal = torch.diagonal(page_scores)
    off_diagonal = page_scores[
        ~torch.eye(len(pair_list), dtype=torch.bool, device=page_scores.device)
    ]
    negative_score = (
        off_diagonal.mean() if off_diagonal.numel() else page_scores.new_zeros(())
    )
    page_pair_accuracy = 0.5 * (
        logits.argmax(dim=1).eq(labels).to(logits.dtype).mean()
        + logits.argmax(dim=0).eq(labels).to(logits.dtype).mean()
    )
    return loss, {
        "objective": "page_pair_cross_entropy",
        "assignment_loss": 0.0,
        "descriptor_loss": 0.0,
        "visual_descriptor_loss": 0.0,
        "total_loss": float(loss.detach().cpu()),
        "page_embedding_loss": float(loss.detach().cpu()),
        "page_pair_positive_score": float(diagonal.mean().detach().cpu()),
        "page_pair_negative_score": float(negative_score.detach().cpu()),
        "page_pair_accuracy": float(page_pair_accuracy.detach().cpu()),
        "unlabeled_page_embedding_loss": float(loss.detach().cpu()),
        "page_embedding_batch_size": float(len(pair_list)),
        "page_embedding_infonce_loss": float(loss.detach().cpu()),
        "positive_labels": float(len(pair_list)),
        "node_correspondence_labels": 0.0,
    }


def batch_self_view_node_matching_loss(
    model: Any,
    pairs: Iterable[UnlabeledPagePair],
    *,
    rng: random.Random,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    augment_config: AugmentConfig | None = None,
    return_details: bool = False,
) -> Any:
    """Train node matching from two transformed views of each real page.

    The transforms preserve latent node identity, so every target is generated
    automatically from the observation itself.  This is one bidirectional
    real-node cross-entropy: there is no page classification term, NULL class,
    pseudo-label model, gate, bonus, or reranking loss.
    """

    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    page_graphs = [
        graph
        for pair in pairs
        for graph in (pair.graph_a, pair.graph_b)
    ]
    total = None
    label_count = 0
    correct = 0
    positive_scores = []
    hard_negative_scores = []
    view_count = 0
    for graph in page_graphs:
        training_pair = make_training_pair(
            graph,
            rng=rng,
            config=augment_config,
            matcher_config=config,
        )
        if training_pair is None:
            continue
        view_count += 1
        output = model(
            *matcher_inputs(
                training_pair.graph_a,
                training_pair.graph_b,
                config=config,
                device=device,
            )
        )
        for logits, targets in (
            (output["logits_ab"], training_pair.targets_a_to_b),
            (output["logits_ba"], training_pair.targets_b_to_a),
        ):
            target_tensor = torch.as_tensor(targets, device=logits.device)
            rows = target_tensor.ge(0).nonzero(as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            selected_targets = target_tensor[rows].long()
            selected_logits = logits[rows]
            direction_loss = torch.nn.functional.cross_entropy(
                selected_logits,
                selected_targets,
                reduction="sum",
            )
            total = direction_loss if total is None else total + direction_loss
            label_count += int(rows.numel())
            correct += int(
                selected_logits.argmax(dim=1).eq(selected_targets).sum().detach().cpu()
            )
            positive_scores.append(
                selected_logits.gather(1, selected_targets[:, None]).mean()
            )
            negative_mask = torch.ones_like(selected_logits, dtype=torch.bool)
            negative_mask.scatter_(1, selected_targets[:, None], False)
            hard_negative_scores.append(
                selected_logits.masked_fill(~negative_mask, -1e4).max(dim=1).values.mean()
            )
    if total is None or label_count == 0:
        raise ValueError("self-view node matching produced no retained nodes")
    loss = total / label_count
    if not return_details:
        return loss
    positive_score = torch.stack(positive_scores).mean()
    hard_negative_score = torch.stack(hard_negative_scores).mean()
    return loss, {
        "objective": "self_view_node_cross_entropy",
        "total_loss": float(loss.detach().cpu()),
        "node_top1_accuracy": correct / label_count,
        "node_positive_score": float(positive_score.detach().cpu()),
        "node_hard_negative_score": float(hard_negative_score.detach().cpu()),
        "node_score_margin": float(
            (positive_score - hard_negative_score).detach().cpu()
        ),
        "generated_node_labels": float(label_count),
        "human_node_labels": 0.0,
        "page_views": float(view_count),
    }


def batch_supervised_node_matching_loss(
    model: Any,
    pairs: Iterable[CorrespondencePair],
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    rng: random.Random | None = None,
    augment_config: AugmentConfig | None = None,
    score_stage: str = "final",
    hard_row_weight: float = 0.0,
    return_details: bool = False,
) -> Any:
    """Train the one score matrix from real cross-platform node gold.

    Set-valued labels use a marginal cross-entropy: probability assigned to
    any reviewed target is correct. All other nodes on the same target page
    are hard negatives. The same loss is evaluated in both directions.
    """

    torch = _require_torch()
    config = matcher_config or MatcherConfig()
    if score_stage not in {"unary", "final"}:
        raise ValueError("score_stage must be 'unary' or 'final'")
    if hard_row_weight < 0.0:
        raise ValueError("hard_row_weight must be non-negative")
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("supervised node matching requires at least one page pair")
    total = None
    total_row_weight = None
    query_rows = 0
    correct = 0
    unary_correct = 0
    refinement_help = 0
    refinement_hurt = 0
    human_labels = 0
    positive_scores = []
    hard_negative_scores = []
    layer_correct: list[int] = []
    supervised_layer_count = 0
    supervised_score_matrix_count = 0
    for pair_index, original_pair in enumerate(pair_list):
        pair = original_pair
        if augment_config is not None:
            if rng is None:
                raise ValueError("augmentation requires an explicit random generator")
            pair = augment_correspondence_pair(
                original_pair,
                rng=rng,
                config=augment_config,
                matcher_config=config,
                view_id=f"supervised_{pair_index}",
            )
        human_labels += len(original_pair.origin_ids)
        output = model(
            *matcher_inputs(
                pair.graph_a,
                pair.graph_b,
                config=config,
                device=device,
            ),
            node_only=score_stage == "unary",
        )
        unary_logits_ab = output.get("unary_logits", output["logits_ab"])
        final_logits_ab = (
            unary_logits_ab if score_stage == "unary" else output["logits_ab"]
        )
        refinement_layers_ab = (
            ()
            if score_stage == "unary"
            else tuple(output.get("assignment_scores_by_layer") or (final_logits_ab,))
        )
        score_matrices_ab = (unary_logits_ab, *refinement_layers_ab)
        supervised_layer_count = len(refinement_layers_ab)
        supervised_score_matrix_count = len(score_matrices_ab)
        if not layer_correct:
            layer_correct = [0] * supervised_layer_count
        elif len(layer_correct) != supervised_layer_count:
            raise ValueError("all page pairs must use the same correspondence depth")
        if supervised_score_matrix_count == 4:
            layer_weights = (0.1, 0.2, 0.3, 0.4)
        else:
            raw_weights = tuple(range(1, supervised_score_matrix_count + 1))
            normalizer = float(sum(raw_weights))
            layer_weights = tuple(value / normalizer for value in raw_weights)
        for logits, score_matrices, refinement_layers, unary_logits, positive_targets in (
            (
                final_logits_ab,
                score_matrices_ab,
                refinement_layers_ab,
                unary_logits_ab,
                pair.positive_targets_a_to_b,
            ),
            (
                final_logits_ab.T,
                tuple(layer.T for layer in score_matrices_ab),
                tuple(layer.T for layer in refinement_layers_ab),
                unary_logits_ab.T,
                pair.positive_targets_b_to_a,
            ),
        ):
            for row_index, target_indices in enumerate(positive_targets):
                if not target_indices:
                    continue
                row = logits[row_index]
                target_tensor = torch.as_tensor(
                    target_indices,
                    dtype=torch.long,
                    device=row.device,
                )
                row_loss = sum(
                    weight
                    * (
                        torch.logsumexp(layer[row_index], dim=0)
                        - torch.logsumexp(
                            layer[row_index][target_tensor],
                            dim=0,
                        )
                    )
                    for weight, layer in zip(
                        layer_weights,
                        score_matrices,
                        strict=True,
                    )
                )
                gold_probability = torch.exp(-row_loss.detach()).clamp(0.0, 1.0)
                row_weight = 1.0 + float(hard_row_weight) * (
                    1.0 - gold_probability
                )
                weighted_row_loss = row_weight * row_loss
                total = (
                    weighted_row_loss
                    if total is None
                    else total + weighted_row_loss
                )
                total_row_weight = (
                    row_weight
                    if total_row_weight is None
                    else total_row_weight + row_weight
                )
                query_rows += 1
                predicted = int(row.argmax().detach().cpu())
                unary_predicted = int(
                    unary_logits[row_index].argmax().detach().cpu()
                )
                final_is_correct = predicted in target_indices
                unary_is_correct = unary_predicted in target_indices
                for layer_index, layer in enumerate(refinement_layers):
                    layer_predicted = int(
                        layer[row_index].argmax().detach().cpu()
                    )
                    layer_correct[layer_index] += int(
                        layer_predicted in target_indices
                    )
                correct += int(final_is_correct)
                unary_correct += int(unary_is_correct)
                refinement_help += int(not unary_is_correct and final_is_correct)
                refinement_hurt += int(unary_is_correct and not final_is_correct)
                positive_scores.append(row[target_tensor].max())
                negative_mask = torch.ones_like(row, dtype=torch.bool)
                negative_mask[target_tensor] = False
                if negative_mask.any():
                    hard_negative_scores.append(row[negative_mask].max())
    if total is None or query_rows == 0:
        raise ValueError("supervised node matching produced no labelled query rows")
    if total_row_weight is None:
        raise ValueError("supervised node matching produced no row weights")
    loss = total / total_row_weight
    if not return_details:
        return loss
    positive_score = torch.stack(positive_scores).mean()
    hard_negative_score = (
        torch.stack(hard_negative_scores).mean()
        if hard_negative_scores
        else positive_score.new_zeros(())
    )
    return loss, {
        "objective": "symmetric_cross_platform_node_cross_entropy",
        "score_stage": score_stage,
        "total_loss": float(loss.detach().cpu()),
        "node_top1_accuracy": correct / query_rows,
        "unary_node_top1_accuracy": unary_correct / query_rows,
        "unary_node_top1_correct": float(unary_correct),
        "refinement_help": float(refinement_help),
        "refinement_hurt": float(refinement_hurt),
        "node_top1_correct": float(correct),
        "node_positive_score": float(positive_score.detach().cpu()),
        "node_hard_negative_score": float(hard_negative_score.detach().cpu()),
        "node_score_margin": float(
            (positive_score - hard_negative_score).detach().cpu()
        ),
        "human_node_labels": float(human_labels),
        "supervised_query_rows": float(query_rows),
        "page_pair_labels": 0.0,
        "supervised_layers": float(supervised_layer_count),
        "supervised_score_matrices": float(supervised_score_matrix_count),
        "hard_row_weight": float(hard_row_weight),
        "mean_training_row_weight": float(
            (total_row_weight / query_rows).detach().cpu()
        ),
        **{
            f"layer_{index + 1}_top1_accuracy": value / query_rows
            for index, value in enumerate(layer_correct)
        },
        **{
            f"layer_{index + 1}_top1_correct": float(value)
            for index, value in enumerate(layer_correct)
        },
    }


def _page_embedding_contrastive_loss_from_outputs(
    output: Any,
    negative_output: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, Any]:
    positive_source = torch.nn.functional.normalize(
        output["source_config_embedding"], dim=0
    )
    positive_target = torch.nn.functional.normalize(
        output["target_config_embedding"], dim=0
    )
    negative_source = torch.nn.functional.normalize(
        negative_output["source_config_embedding"], dim=0
    )
    negative_target = torch.nn.functional.normalize(
        negative_output["target_config_embedding"], dim=0
    )
    positive_cosine = 0.5 * (
        (positive_source * positive_target).sum()
        + (positive_target * positive_source).sum()
    )
    negative_cosine = 0.5 * (
        (positive_source * negative_target).sum()
        + (positive_target * negative_source).sum()
    )
    logits = torch.stack(
        (
            torch.stack(
                (
                    (positive_source * positive_target).sum(),
                    (positive_source * negative_target).sum(),
                )
            ),
            torch.stack(
                (
                    (positive_target * positive_source).sum(),
                    (positive_target * negative_source).sum(),
                )
            ),
        )
    ) / 0.1
    labels = torch.zeros(2, dtype=torch.long, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    bridge_losses = []
    for prefix in ("source", "target"):
        bridge = output.get(f"{prefix}_pair_state_bridge")
        state = output.get(f"{prefix}_state_embedding")
        if bridge is not None and state is not None:
            bridge_losses.append(
                1.0
                - torch.nn.functional.cosine_similarity(
                    bridge, state.detach(), dim=0
                )
            )
    if bridge_losses:
        loss = loss + 0.1 * torch.stack(bridge_losses).mean()
    return loss, positive_cosine, negative_cosine


def relation_invariance_loss(
    model: Any,
    pair: CorrespondencePair,
    *,
    device: str = "cpu",
    matcher_config: MatcherConfig | None = None,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.10,
    return_details: bool = False,
) -> Any:
    """Preserve local relation states across two structural graph views.

    This objective consumes only the local graph update before cross-page
    attention. It therefore uses same-observation identity as augmentation
    alignment evidence, not as a cross-page correspondence training target.
    """

    torch = _require_torch()
    if contrastive_weight < 0.0:
        raise ValueError("contrastive_weight must be non-negative")
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    config = matcher_config or MatcherConfig()
    inputs = matcher_inputs(
        pair.graph_a,
        pair.graph_b,
        config=config,
        device=device,
    )
    output = model(*inputs)
    source_layers = tuple(output.get("source_local_states_by_layer") or ())
    target_layers = tuple(output.get("target_local_states_by_layer") or ())
    if not source_layers or len(source_layers) != len(target_layers):
        raise ValueError("model does not expose aligned local relation states")

    source_bases = output["source_relation_bases"]
    target_bases = output["target_relation_bases"]
    losses: list[Any] = []
    contrastive_losses: list[Any] = []
    active_rows = 0
    for source_index, target_indices in enumerate(pair.positive_targets_a_to_b):
        if not target_indices:
            continue
        source_has_relations = source_bases[1:, source_index, :].sum() > 0.0
        for target_index in target_indices:
            target_has_relations = target_bases[1:, target_index, :].sum() > 0.0
            if not (source_has_relations or target_has_relations):
                continue
            active_rows += 1
            for source_states, target_states in zip(
                source_layers, target_layers, strict=True
            ):
                source_vector = torch.nn.functional.normalize(
                    source_states[source_index], dim=0
                )
                target_vector = torch.nn.functional.normalize(
                    target_states[target_index], dim=0
                )
                losses.append(1.0 - (source_vector * target_vector).sum())
    for target_index, source_indices in enumerate(pair.positive_targets_b_to_a):
        if not source_indices:
            continue
        target_has_relations = target_bases[1:, target_index, :].sum() > 0.0
        for source_index in source_indices:
            source_has_relations = source_bases[1:, source_index, :].sum() > 0.0
            if not (source_has_relations or target_has_relations):
                continue
            for source_states, target_states in zip(
                source_layers, target_layers, strict=True
            ):
                source_vector = torch.nn.functional.normalize(
                    source_states[source_index], dim=0
                )
                target_vector = torch.nn.functional.normalize(
                    target_states[target_index], dim=0
                )
                losses.append(1.0 - (target_vector * source_vector).sum())
    for source_states, target_states in zip(
        source_layers, target_layers, strict=True
    ):
        source_vectors = torch.nn.functional.normalize(source_states, dim=-1)
        target_vectors = torch.nn.functional.normalize(target_states, dim=-1)
        relation_logits = (source_vectors @ target_vectors.T) / float(
            contrastive_temperature
        )
        source_contrastive = _partial_assignment_loss(
            relation_logits,
            pair.positive_targets_a_to_b,
            candidate_indices=tuple(range(target_states.shape[0])),
            torch=torch,
            device=device,
        )
        target_contrastive = _partial_assignment_loss(
            relation_logits.T,
            pair.positive_targets_b_to_a,
            candidate_indices=tuple(range(source_states.shape[0])),
            torch=torch,
            device=device,
        )
        contrastive_losses.append(0.5 * (source_contrastive + target_contrastive))
    invariance_loss = (
        source_layers[-1].new_zeros(())
        if not losses
        else torch.stack(losses).mean()
    )
    contrastive_loss = (
        source_layers[-1].new_zeros(())
        if not contrastive_losses
        else torch.stack(contrastive_losses).mean()
    )
    loss = invariance_loss + float(contrastive_weight) * contrastive_loss
    if not return_details:
        return loss
    return loss, {
        "assignment_loss": 0.0,
        "descriptor_loss": 0.0,
        "visual_descriptor_loss": 0.0,
        "total_loss": float(loss.detach().cpu()),
        "final_assignment_loss": 0.0,
        "intermediate_assignment_loss": 0.0,
        "supervised_layers": 0.0,
        "descriptor_weight": 0.0,
        "visual_descriptor_weight": 0.0,
        "visual_descriptor_trainable": 0.0,
        "relation_invariance_loss": float(loss.detach().cpu()),
        "relation_invariance_raw_loss": float(invariance_loss.detach().cpu()),
        "relation_invariance_rows": float(active_rows),
        "relation_invariance_layers": float(len(source_layers)),
        "relation_contrastive_loss": float(contrastive_loss.detach().cpu()),
        "relation_contrastive_weight": float(contrastive_weight),
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
    include_all_candidates: bool = False,
    score_stage: str = "final",
) -> dict[str, Any]:
    """Evaluate bidirectional correspondences and warmed model performance."""

    if latency_repeats < 0:
        raise ValueError("latency_repeats must be non-negative")
    if score_stage not in {"unary", "final"}:
        raise ValueError("score_stage must be 'unary' or 'final'")
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
    route_weight_sums = torch.zeros(3, dtype=torch.float32, device=device)
    route_available_counts = torch.zeros(3, dtype=torch.float32, device=device)
    route_core_support_counts = torch.zeros(3, dtype=torch.float32, device=device)
    route_argmax_counts = torch.zeros(3, dtype=torch.float32, device=device)
    route_core_support_total = torch.zeros((), dtype=torch.float32, device=device)
    route_effective_support_total = torch.zeros(
        (), dtype=torch.float32, device=device
    )
    route_entropy_total = torch.zeros((), dtype=torch.float32, device=device)
    route_top1_mass_total = torch.zeros((), dtype=torch.float32, device=device)
    route_node_count = 0
    route_outputs_seen = False
    with torch.no_grad():
        for pair in pair_list:
            inputs = matcher_inputs(
                pair.graph_a,
                pair.graph_b,
                config=config,
                device=device,
            )
            output = model(*inputs, node_only=score_stage == "unary")
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
                output = model(*inputs, node_only=score_stage == "unary")
                _synchronize(device, torch)
                model_latencies.append((time.perf_counter() - model_started) * 1000.0)
                end_to_end_latencies.append(
                    (time.perf_counter() - total_started) * 1000.0
                )
            for prefix, token_ids in (
                ("source", inputs[0]),
                ("target", inputs[3]),
            ):
                if f"{prefix}_route_weights" not in output:
                    continue
                route_outputs_seen = True
                weights = output[f"{prefix}_route_weights"].to(torch.float32)
                core_weights = output.get(
                    f"{prefix}_route_core_weights",
                    output[f"{prefix}_route_weights"],
                ).to(torch.float32)
                visual_available = output[f"{prefix}_visual_mask"].reshape(-1).gt(0)
                available = torch.stack(
                    (
                        output.get(
                            f"{prefix}_text_mask",
                            token_ids.ne(0).any(dim=1, keepdim=True),
                        ).reshape(-1).gt(0),
                        visual_available,
                        torch.ones_like(visual_available),
                    ),
                    dim=1,
                )
                available_float = available.to(torch.float32)
                route_weight_sums += (weights * available_float).sum(dim=0)
                route_available_counts += available_float.sum(dim=0)
                route_core_support_counts += (
                    core_weights.gt(1e-8) & available
                ).to(torch.float32).sum(dim=0)
                route_argmax_counts += torch.bincount(
                    weights.argmax(dim=1), minlength=3
                ).to(torch.float32)
                route_core_support_total += core_weights.gt(1e-8).sum(dim=1).to(
                    torch.float32
                ).sum()
                route_effective_support_total += weights.ge(0.01).sum(dim=1).to(
                    torch.float32
                ).sum()
                route_entropy_total += -(
                    weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()
                ).sum()
                route_top1_mass_total += weights.max(dim=1).values.sum()
                route_node_count += int(weights.shape[0])
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
                    final_scores = (
                        output["logits_ba"]
                        if component_transpose
                        else output["logits_ab"]
                    )
                    if score_stage == "unary":
                        unary_scores = output.get("unary_logits", output["logits_ab"])
                        final_scores = (
                            unary_scores.T if component_transpose else unary_scores
                        )
                    selected_logits = final_scores[row_index][list(candidate_indices)]
                    selected_layer = len(
                        tuple(output.get("assignment_scores_by_layer") or ())
                    ) - 1
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
                                "all_candidates": (
                                    [
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
                                        for target_index in ranked
                                    ]
                                    if include_all_candidates
                                    else None
                                ),
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
                        descriptor_logits[row_index, list(candidate_indices)],
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
    modality_names = ("text", "visual", "xml")
    route_weight_values = route_weight_sums.detach().cpu().tolist()
    route_available_values = route_available_counts.detach().cpu().tolist()
    route_core_support_values = (
        route_core_support_counts.detach().cpu().tolist()
    )
    route_argmax_values = route_argmax_counts.detach().cpu().tolist()
    route_denominator = float(route_node_count) if route_node_count else 1.0
    return {
        "schema_version": "omnitransfer_correspondence_metrics_v1",
        "score_stage": score_stage,
        "decoder": (
            "v9_local_fusion_correspondence_iteration_v1"
            if route_outputs_seen
            else "page_local_real_node_ranking_v1"
        ),
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
        "route_diagnostics": ({
            "modality_order": modality_names,
            "availability_rate": {
                name: route_available_values[index] / route_denominator
                for index, name in enumerate(modality_names)
            },
            "mean_weight_when_available": {
                name: (
                    route_weight_values[index] / route_available_values[index]
                    if route_available_values[index]
                    else 0.0
                )
                for index, name in enumerate(modality_names)
            },
            "argmax_rate": {
                name: route_argmax_values[index] / route_denominator
                for index, name in enumerate(modality_names)
            },
            "sparse_core_support_rate_when_available": {
                name: (
                    route_core_support_values[index]
                    / route_available_values[index]
                    if route_available_values[index]
                    else 0.0
                )
                for index, name in enumerate(modality_names)
            },
            "mean_sparse_core_support": (
                float(route_core_support_total.detach().cpu()) / route_denominator
            ),
            "mean_effective_support_at_0_01": (
                float(route_effective_support_total.detach().cpu())
                / route_denominator
            ),
            "mean_entropy": (
                float(route_entropy_total.detach().cpu()) / route_denominator
            ),
            "mean_top1_mass": (
                float(route_top1_mass_total.detach().cpu()) / route_denominator
            ),
        } if route_outputs_seen else None),
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
    unlabeled_page_pairs: Iterable[UnlabeledPagePair] = (),
    model: Any | None = None,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 17,
    device: str = "cpu",
    validation_device: str | None = None,
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
    cross_page_augmented_views: int = 0,
    unlabeled_pair_repeats: int = 0,
    use_cross_page_soft_teacher: bool = False,
    unlabeled_page_embedding_weight: float = 0.0,
    unlabeled_page_batch_size: int = 0,
    papt_update_ratio: float = 0.0,
    cross_page_soft_teacher_weight: float = 1.0,
    teacher_confidence_threshold: float = 0.20,
    teacher_margin_threshold: float = 0.02,
    teacher_temperature: float = 1.0,
    teacher_min_stable_layers: int = 2,
    descriptor_weight: float = 0.0,
    descriptor_learning_rate_scale: float = 0.25,
    visual_descriptor_weight: float = 0.0,
    visual_descriptor_temperature: float = 0.07,
    strategy_weight: float = 0.0,
    matchability_weight: float = 0.20,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.5,
    refinement_stability_weight: float = 0.0,
    active_state_weight: float = 0.0,
    train_only_visual_context: bool = False,
    train_only_relations: bool = False,
    train_only_direct_evidence: bool = False,
    train_only_pairwise_correspondence: bool = False,
    train_only_layout: bool = False,
    train_only_pair_state: bool = False,
    train_only_neighbor_context: bool = False,
    train_only_correspondence_core: bool = False,
    relation_invariance_weight: float = 0.0,
    relation_contrastive_weight: float = 0.0,
    page_embedding_weight: float = 0.0,
    disable_anchor_identity_transport: bool = False,
) -> tuple[Any, list[dict[str, float]]]:
    """Train geometric-v9 on canonical cross-page and synthetic pairs."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if synthetic_pairs_per_graph < 0:
        raise ValueError("synthetic_pairs_per_graph must be non-negative")
    if cross_page_pair_repeats < 0:
        raise ValueError("cross_page_pair_repeats must be non-negative")
    if cross_page_augmented_views < 0:
        raise ValueError("cross_page_augmented_views must be non-negative")
    if unlabeled_pair_repeats < 0:
        raise ValueError("unlabeled_pair_repeats must be non-negative")
    if (
        synthetic_pairs_per_graph == 0
        and cross_page_pair_repeats == 0
        and cross_page_augmented_views == 0
        and unlabeled_pair_repeats == 0
    ):
        raise ValueError("training requires synthetic, labelled, or unlabeled pairs")
    if cross_page_soft_teacher_weight < 0.0:
        raise ValueError("cross-page soft teacher weight must be non-negative")
    if unlabeled_page_embedding_weight < 0.0:
        raise ValueError("unlabeled page embedding weight must be non-negative")
    if unlabeled_page_batch_size < 0:
        raise ValueError("unlabeled page batch size must be non-negative")
    if unlabeled_page_batch_size == 1:
        raise ValueError("unlabeled page batch size must be zero or at least two")
    if papt_update_ratio < 0.0:
        raise ValueError("PAPT update ratio must be non-negative")
    if not 0.0 <= teacher_confidence_threshold <= 1.0:
        raise ValueError("teacher confidence threshold must be in [0, 1]")
    if not 0.0 <= teacher_margin_threshold <= 1.0:
        raise ValueError("teacher margin threshold must be in [0, 1]")
    if teacher_temperature <= 0.0:
        raise ValueError("teacher temperature must be positive")
    if teacher_min_stable_layers <= 0:
        raise ValueError("teacher stable layers must be positive")
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
    if hard_negative_weight < 0.0:
        raise ValueError("hard_negative_weight must be non-negative")
    if hard_negative_margin < 0.0:
        raise ValueError("hard_negative_margin must be non-negative")
    if refinement_stability_weight < 0.0:
        raise ValueError("refinement_stability_weight must be non-negative")
    if active_state_weight < 0.0:
        raise ValueError("active_state_weight must be non-negative")
    if relation_invariance_weight < 0.0:
        raise ValueError("relation_invariance_weight must be non-negative")
    if relation_contrastive_weight < 0.0:
        raise ValueError("relation_contrastive_weight must be non-negative")
    if page_embedding_weight < 0.0:
        raise ValueError("page_embedding_weight must be non-negative")
    if any(
        value != 0.0
        for value in (
            descriptor_weight,
            visual_descriptor_weight,
            strategy_weight,
            hard_negative_weight,
            refinement_stability_weight,
            active_state_weight,
            relation_invariance_weight,
            relation_contrastive_weight,
            page_embedding_weight,
            unlabeled_page_embedding_weight,
        )
    ):
        raise ValueError(
            "router-anchor training supports only assignment and NULL losses"
        )
    if matchability_weight != 0.20:
        raise ValueError("router-anchor NULL loss weight is fixed at 0.20")
    if sum(
        (
            train_only_visual_context,
            train_only_relations,
            train_only_direct_evidence,
            train_only_pairwise_correspondence,
            train_only_layout,
            train_only_pair_state,
            train_only_neighbor_context,
            train_only_correspondence_core,
        )
    ) > 1:
        raise ValueError(
            "visual-context-only, relation-only, direct-evidence-only, and "
            "pairwise-only/layout-only/pair-state-only/neighbor-only/core-only modes are exclusive"
        )
    if any(
        (
            train_only_visual_context,
            train_only_relations,
            train_only_direct_evidence,
            train_only_pairwise_correspondence,
            train_only_layout,
            train_only_pair_state,
            train_only_neighbor_context,
            train_only_correspondence_core,
        )
    ):
        raise ValueError("router-anchor training does not expose train-only bypasses")
    if not 0.0 < descriptor_learning_rate_scale <= 1.0:
        raise ValueError(
            "descriptor_learning_rate_scale must be in (0, 1]"
        )
    graph_list = list(graphs)
    cross_page_pairs = list(correspondence_pairs)
    unlabeled_pairs = list(unlabeled_page_pairs)
    dev_pairs = list(validation_pairs)
    if (
        unlabeled_pairs
        or unlabeled_pair_repeats
        or use_cross_page_soft_teacher
        or unlabeled_page_batch_size
        or papt_update_ratio != 0.0
    ):
        raise ValueError(
            "router-anchor training does not expose unlabeled auxiliary objectives"
        )
    if (
        unlabeled_pairs
        and unlabeled_pair_repeats > 0
        and not use_cross_page_soft_teacher
        and unlabeled_page_embedding_weight == 0.0
    ):
        raise ValueError("unlabeled page pairs require an enabled objective")
    if not cross_page_pairs and not graph_list and not unlabeled_pairs:
        raise ValueError("at least one correspondence pair, page pair, or graph is required")
    config = matcher_config or MatcherConfig()
    if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("training supports only the geometric-v9 matcher")
    torch = _require_torch()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    augmentation = augment_config or AugmentConfig()
    if cross_page_augmented_views and not augmentation.family_curriculum:
        raise ValueError(
            "cross_page_augmented_views requires family curriculum"
        )
    resumed_from_model = model is not None
    matcher = (model or build_geometric_v9_matcher(config)).to(device)
    if disable_anchor_identity_transport:
        transport_output = getattr(
            getattr(matcher, "association_layer", None),
            "transport_output",
            None,
        )
        if transport_output is None:
            raise ValueError("matcher has no anchor-identity transport output")
        with torch.no_grad():
            transport_output.weight.zero_()
        transport_output.weight.requires_grad_(False)
    teacher = None
    if (
        unlabeled_pairs
        and unlabeled_pair_repeats > 0
        and use_cross_page_soft_teacher
    ):
        if not resumed_from_model:
            raise ValueError("unlabeled soft-teacher training requires a pretrained model")
        if cross_page_soft_teacher_weight <= 0.0:
            raise ValueError("unlabeled soft-teacher training requires a positive weight")
        teacher = copy.deepcopy(matcher).to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    if (
        unlabeled_pairs
        and unlabeled_pair_repeats > 0
        and unlabeled_page_embedding_weight > 0.0
        and len(unlabeled_pairs) < 2
    ):
        raise ValueError("unlabeled page embedding training requires two page pairs")
    if train_only_visual_context:
        if not resumed_from_model:
            raise ValueError(
                "visual-context-only training requires a pretrained model"
            )
        residual_names = {
            name
            for name, _ in matcher.named_parameters()
            if name.startswith("visual_context_to_hidden.")
        }
        if not residual_names:
            raise ValueError(
                "visual-context-only training requires a visual context residual"
            )
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in residual_names)
    if train_only_relations:
        if not resumed_from_model:
            raise ValueError("relation-only training requires a pretrained model")
        relation_names = {
            name
            for name, _ in matcher.named_parameters()
            if name == "relation_compatibility" or ".local." in name
        }
        if not relation_names:
            raise ValueError("relation-only training requires local relation parameters")
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in relation_names)
    if train_only_direct_evidence:
        if not resumed_from_model:
            raise ValueError(
                "direct-evidence-only training requires a pretrained model"
            )
        if not config.direct_pair_evidence:
            raise ValueError(
                "direct-evidence-only training requires direct pair evidence"
            )
        evidence_names = {
            name
            for name, _ in matcher.named_parameters()
            if name.startswith("direct_pair_projection.")
        }
        if not evidence_names:
            raise ValueError(
                "direct-evidence-only training requires a direct evidence projection"
            )
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in evidence_names)
    if train_only_pairwise_correspondence:
        if not resumed_from_model:
            raise ValueError("pairwise-only training requires a pretrained model")
        if not config.pairwise_local_correspondence:
            raise ValueError(
                "pairwise-only training requires pairwise local correspondence"
            )
        pairwise_names = {
            name
            for name, _ in matcher.named_parameters()
            if ".pairwise_relation_projection." in name
            or ".pairwise_gate." in name
        }
        if not pairwise_names:
            raise ValueError(
                "pairwise-only training requires pairwise relation projections"
            )
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in pairwise_names)
    if train_only_layout:
        if not resumed_from_model:
            raise ValueError("layout-only training requires a pretrained model")
        layout_names = {
            name
            for name, _ in matcher.named_parameters()
            if name.startswith("layout_projection.")
        }
        if not layout_names:
            raise ValueError(
                "layout-only training requires parent-relative layout features"
            )
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in layout_names)
    if train_only_pair_state:
        if not resumed_from_model:
            raise ValueError("pair-state-only training requires a pretrained model")
        if not config.correspondence_pair_state:
            raise ValueError(
                "pair-state-only training requires correspondence pair state"
            )
        pair_state_names = {
            name
            for name, _ in matcher.named_parameters()
            if ".pair_state_" in name
            or ".source_pair_feedback." in name
            or ".target_pair_feedback." in name
        }
        if not pair_state_names:
            raise ValueError(
                "pair-state-only training requires pair-state parameters"
            )
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in pair_state_names)
    if train_only_neighbor_context:
        if not resumed_from_model:
            raise ValueError("neighbor-only training requires a pretrained model")
        if not config.learned_multi_neighbor_context:
            raise ValueError(
                "neighbor-only training requires learned multi-neighbor context"
            )
        neighbor_names = {
            name
            for name, _ in matcher.named_parameters()
            if ".neighbor_" in name
        }
        if not neighbor_names:
            raise ValueError("neighbor-only training requires neighbor parameters")
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in neighbor_names)
    if train_only_correspondence_core:
        if not resumed_from_model:
            raise ValueError("core-only training requires a pretrained model")
        if not (
            config.correspondence_pair_state
            and config.learned_multi_neighbor_context
        ):
            raise ValueError(
                "core-only training requires pair state and multi-neighbor context"
            )
        core_names = {
            name
            for name, _ in matcher.named_parameters()
            if ".neighbor_" in name
            or ".pair_state_" in name
            or ".source_pair_feedback." in name
            or ".target_pair_feedback." in name
            or name.startswith("stable_state_")
            or name.startswith("active_state_")
            or name.startswith("stable_pair_state_bridge.")
            or name.startswith("active_pair_state_bridge.")
            or name.startswith("state_projection.")
        }
        if not core_names:
            raise ValueError("core-only training requires correspondence parameters")
        for name, parameter in matcher.named_parameters():
            parameter.requires_grad_(name in core_names)
    if train_only_visual_context:
        trainable_group_name = "visual_context"
    elif train_only_direct_evidence:
        trainable_group_name = "direct_evidence"
    elif train_only_relations:
        trainable_group_name = "relations"
    elif train_only_pairwise_correspondence:
        trainable_group_name = "pairwise"
    elif train_only_layout:
        trainable_group_name = "layout"
    elif train_only_pair_state:
        trainable_group_name = "pair_state"
    elif train_only_neighbor_context:
        trainable_group_name = "neighbor_context"
    elif train_only_correspondence_core:
        trainable_group_name = "correspondence_core"
    else:
        trainable_group_name = "context"
    descriptor_prefixes = (
        "token_embedding",
        "token_projection",
        "text_projection",
        "text_to_hidden",
        "numeric_projection",
        "xml_projection",
        "layout_projection",
        "xml_to_hidden",
        "visual_to_hidden",
        "missing_",
        "descriptor_",
    )
    descriptor_parameters = []
    context_parameters = []
    for name, parameter in matcher.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            descriptor_parameters
            if name.startswith(descriptor_prefixes)
            else context_parameters
        )
        destination.append(parameter)
    parameter_groups = []
    if descriptor_parameters:
        parameter_groups.append(
            {
                "params": descriptor_parameters,
                "lr": learning_rate,
                "group_name": "layout" if train_only_layout else "descriptor",
            }
        )
    if context_parameters:
        parameter_groups.append(
            {
                "params": context_parameters,
                "lr": learning_rate,
                "group_name": trainable_group_name,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_dev_key: tuple[float, float, float] | None = None
    best_dev_state: dict[str, Any] | None = None
    best_dev_epoch = 0
    dev_device = validation_device or device
    if resumed_from_model and dev_pairs:
        matcher.to(dev_device)
        baseline_metrics = evaluate_correspondence_pairs(
            matcher,
            dev_pairs,
            device=dev_device,
            matcher_config=config,
            latency_repeats=0,
        )
        matcher.to(device)
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
        training_items: list[
            tuple[
                str,
                CorrespondencePair
                | UnlabeledPagePair
                | tuple[UnlabeledPagePair, ...]
                | UIGraph,
            ]
        ] = []
        for pair in cross_page_pairs:
            is_relation_auxiliary = (
                relation_invariance_weight > 0.0
                and pair.graph_a.metadata.get("label_status") == "self_supervised"
            )
            training_items.extend(
                (
                    "relation_auxiliary" if is_relation_auxiliary else "cross_page",
                    pair,
                )
                for _ in range(cross_page_pair_repeats)
            )
            if not is_relation_auxiliary:
                training_items.extend(
                    ("cross_page_augmented", pair)
                    for _ in range(cross_page_augmented_views)
                )
        training_items.extend(
            ("self_supervised", graph)
            for graph in graph_list
            for _ in range(synthetic_pairs_per_graph)
        )
        sampled_unlabeled_pages: list[UnlabeledPagePair] = []
        for pair in unlabeled_pairs:
            expected_repeats = unlabeled_pair_repeats * pair.sampling_weight
            sampled_repeats = int(expected_repeats)
            if rng.random() < expected_repeats - sampled_repeats:
                sampled_repeats += 1
            if use_cross_page_soft_teacher:
                training_items.extend(
                    ("cross_page_soft", pair) for _ in range(sampled_repeats)
                )
            if unlabeled_page_embedding_weight > 0.0:
                sampled_unlabeled_pages.extend(
                    pair for _ in range(sampled_repeats)
                )
        if unlabeled_page_embedding_weight > 0.0:
            rng.shuffle(sampled_unlabeled_pages)
            if unlabeled_page_batch_size >= 2:
                page_batches = _page_infonce_batches(
                    sampled_unlabeled_pages,
                    batch_size=unlabeled_page_batch_size,
                    rng=rng,
                )
                if papt_update_ratio > 0.0 and cross_page_pairs:
                    maximum_updates = max(
                        1,
                        round(len(cross_page_pairs) * papt_update_ratio),
                    )
                    page_batches = page_batches[:maximum_updates]
                training_items.extend(
                    ("unlabeled_page_batch", batch) for batch in page_batches
                )
            else:
                training_items.extend(
                    ("unlabeled_page", pair)
                    for pair in sampled_unlabeled_pages
                )
        if not training_items and unlabeled_pairs and unlabeled_pair_repeats > 0:
            fallback = max(unlabeled_pairs, key=lambda pair: pair.sampling_weight)
            if use_cross_page_soft_teacher:
                training_items.append(("cross_page_soft", fallback))
            if unlabeled_page_embedding_weight > 0.0:
                if unlabeled_page_batch_size >= 2 and len(unlabeled_pairs) >= 2:
                    training_items.append(
                        (
                            "unlabeled_page_batch",
                            tuple(unlabeled_pairs[:unlabeled_page_batch_size]),
                        )
                    )
                else:
                    training_items.append(("unlabeled_page", fallback))
        rng.shuffle(training_items)
        page_negative_items = [
            item for kind, item in training_items if kind == "cross_page"
        ]
        page_negative_index = 0
        unlabeled_negative_index = 0
        matcher.train()
        losses: list[float] = []
        loss_details: list[dict[str, float]] = []
        self_supervised_pairs = 0
        aligned_cross_page_pairs = 0
        augmented_cross_page_pairs = 0
        augmentation_family_counts = {
            family: 0 for family in AUGMENTATION_FAMILIES
        }
        augmentation_cardinality_counts = {1: 0, 2: 0, 3: 0}
        soft_cross_page_pairs = 0
        unlabeled_page_embedding_pairs = 0
        soft_teacher_rows = 0
        positive_labels = 0
        for pair_index, (pair_kind, item) in enumerate(training_items, start=1):
            if pair_kind == "unlabeled_page_batch":
                pair = None
            elif pair_kind == "self_supervised":
                pair = make_training_pair(
                    item,
                    rng=rng,
                    config=augmentation,
                    matcher_config=config,
                )
                if pair is None:
                    continue
            elif pair_kind == "cross_page_augmented":
                pair = augment_correspondence_pair(
                    item,
                    rng=rng,
                    config=augmentation,
                    matcher_config=config,
                    view_id=f"cross_{epoch + 1}_{pair_index}",
                )
                selected_families = tuple(
                    pair.graph_a.metadata.get("augmentation_families") or ()
                )
                if len(selected_families) not in augmentation_cardinality_counts:
                    raise ValueError("cross-page curriculum produced an empty view")
                augmentation_cardinality_counts[len(selected_families)] += 1
                for family in selected_families:
                    augmentation_family_counts[str(family)] += 1
            else:
                pair = item
            optimizer.zero_grad(set_to_none=True)
            negative_pair = None
            if pair_kind == "cross_page" and page_embedding_weight > 0.0:
                for offset in range(1, len(page_negative_items) + 1):
                    candidate = page_negative_items[
                        (page_negative_index + offset) % len(page_negative_items)
                    ]
                    if (
                        candidate.graph_a.graph_id != pair.graph_a.graph_id
                        or candidate.graph_b.graph_id != pair.graph_b.graph_id
                    ):
                        negative_pair = candidate
                        break
                page_negative_index += 1
            if pair_kind == "unlabeled_page_batch":
                loss, details = batch_unlabeled_page_embedding_contrastive_loss(
                    matcher,
                    item,
                    rng=rng,
                    device=device,
                    matcher_config=config,
                    augment_config=augmentation,
                    active_state_weight=active_state_weight,
                    return_details=True,
                )
                loss = unlabeled_page_embedding_weight * loss
                details["total_loss"] = float(loss.detach().cpu())
                details["page_embedding_weight"] = float(
                    unlabeled_page_embedding_weight
                )
            elif pair_kind == "cross_page_soft":
                if teacher is None:
                    raise ValueError("soft cross-page item requires a frozen teacher")
                loss, details = cross_page_soft_consistency_loss(
                    matcher,
                    teacher,
                    item,
                    rng=rng,
                    device=device,
                    matcher_config=config,
                    augment_config=augmentation,
                    confidence_threshold=teacher_confidence_threshold,
                    margin_threshold=teacher_margin_threshold,
                    temperature=teacher_temperature,
                    min_stable_layers=teacher_min_stable_layers,
                    return_details=True,
                )
                loss = cross_page_soft_teacher_weight * loss
                details["total_loss"] = float(loss.detach().cpu())
                pair = None
            elif pair_kind == "unlabeled_page":
                negative_unlabeled_pair = None
                for offset in range(1, len(unlabeled_pairs) + 1):
                    candidate = unlabeled_pairs[
                        (unlabeled_negative_index + offset) % len(unlabeled_pairs)
                    ]
                    if (
                        candidate.graph_a.graph_id != item.graph_a.graph_id
                        or candidate.graph_b.graph_id != item.graph_b.graph_id
                    ):
                        negative_unlabeled_pair = candidate
                        break
                unlabeled_negative_index += 1
                if negative_unlabeled_pair is None:
                    raise ValueError(
                        "unlabeled page embedding requires a distinct negative pair"
                    )
                loss, details = unlabeled_page_embedding_contrastive_loss(
                    matcher,
                    item,
                    negative_unlabeled_pair,
                    rng=rng,
                    device=device,
                    matcher_config=config,
                    augment_config=augmentation,
                    active_state_weight=active_state_weight,
                    return_details=True,
                )
                loss = unlabeled_page_embedding_weight * loss
                details["total_loss"] = float(loss.detach().cpu())
                details["page_embedding_weight"] = float(
                    unlabeled_page_embedding_weight
                )
                pair = None
            elif pair_kind == "relation_auxiliary":
                relation_pair = make_training_pair(
                    item.graph_a,
                    rng=rng,
                    config=augmentation,
                    matcher_config=config,
                )
                if relation_pair is None:
                    continue
                relation_loss, details = relation_invariance_loss(
                    matcher,
                    relation_pair,
                    device=device,
                    matcher_config=config,
                    contrastive_weight=relation_contrastive_weight,
                    return_details=True,
                )
                loss = relation_invariance_weight * relation_loss
                pair = relation_pair
            else:
                effective_page_embedding_weight = (
                    page_embedding_weight
                    if pair_kind == "cross_page" and negative_pair is not None
                    else 0.0
                )
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
                    hard_negative_weight=hard_negative_weight,
                    hard_negative_margin=hard_negative_margin,
                    refinement_stability_weight=refinement_stability_weight,
                    active_state_weight=active_state_weight,
                    page_embedding_weight=effective_page_embedding_weight,
                    page_embedding_negative_pair=negative_pair,
                    return_details=True,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch + 1} "
                    f"pair={pair_index} kind={pair_kind}"
                )
            loss.backward()
            transport_output = getattr(
                getattr(matcher, "association_layer", None),
                "transport_output",
                None,
            )
            transport_output_gradient = (
                transport_output.weight.grad
                if transport_output is not None
                else None
            )
            upstream_transport_parameters = tuple(
                parameter
                for name, parameter in matcher.named_parameters()
                if name.startswith(
                    (
                        "association_layer.center_projection.",
                        "association_layer.neighbor_projection.",
                        "association_layer.counterpart_projection.",
                        "association_layer.relation_type_embedding",
                    )
                )
            )
            upstream_transport_gradient_norm = sum(
                float(parameter.grad.detach().norm().cpu())
                for parameter in upstream_transport_parameters
                if parameter.grad is not None
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                matcher.parameters(),
                max_norm=1.0,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm at epoch={epoch + 1} "
                    f"pair={pair_index} kind={pair_kind}"
                )
            details["unclipped_gradient_norm"] = float(
                gradient_norm.detach().cpu()
            )
            details["gradient_clip_triggered"] = float(gradient_norm > 1.0)
            details["transport_output_gradient_norm"] = (
                float(transport_output_gradient.detach().norm().cpu())
                if transport_output_gradient is not None
                else 0.0
            )
            details["transport_upstream_gradient_norm"] = (
                upstream_transport_gradient_norm
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            loss_details.append(details)
            self_supervised_pairs += int(pair_kind == "self_supervised")
            aligned_cross_page_pairs += int(
                pair_kind in {"cross_page", "cross_page_augmented"}
            )
            augmented_cross_page_pairs += int(
                pair_kind == "cross_page_augmented"
            )
            soft_cross_page_pairs += int(pair_kind == "cross_page_soft")
            unlabeled_page_embedding_pairs += (
                len(item)
                if pair_kind == "unlabeled_page_batch"
                else int(pair_kind == "unlabeled_page")
            )
            soft_teacher_rows += int(details.get("soft_teacher_rows", 0.0))
            if pair is not None:
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
                    "augmented_cross_page_pairs": float(
                        augmented_cross_page_pairs
                    ),
                    **{
                        f"augmentation_family_{family}": float(count)
                        for family, count in augmentation_family_counts.items()
                    },
                    **{
                        f"augmentation_cardinality_{cardinality}": float(count)
                        for cardinality, count in (
                            augmentation_cardinality_counts.items()
                        )
                    },
                    "soft_cross_page_pairs": float(soft_cross_page_pairs),
                    "unlabeled_page_embedding_pairs": float(
                        unlabeled_page_embedding_pairs
                    ),
                    "soft_teacher_rows": float(soft_teacher_rows),
                    "positive_labels": float(positive_labels),
                    **_mean_loss_details(loss_details[-progress_interval:]),
                }
                if checkpoint_callback is not None:
                    checkpoint_callback(matcher, dict(progress_metrics))
                progress_callback(progress_metrics)
        learning_rates = {
            str(group.get("group_name")): float(group["lr"])
            for group in optimizer.param_groups
        }
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "loss": sum(losses) / len(losses),
            "pairs": float(len(losses)),
            "self_supervised_pairs": float(self_supervised_pairs),
            "cross_page_pairs": float(aligned_cross_page_pairs),
            "augmented_cross_page_pairs": float(augmented_cross_page_pairs),
            **{
                f"augmentation_family_{family}": float(count)
                for family, count in augmentation_family_counts.items()
            },
            **{
                f"augmentation_cardinality_{cardinality}": float(count)
                for cardinality, count in augmentation_cardinality_counts.items()
            },
            "soft_cross_page_pairs": float(soft_cross_page_pairs),
            "unlabeled_page_embedding_pairs": float(
                unlabeled_page_embedding_pairs
            ),
            "soft_teacher_rows": float(soft_teacher_rows),
            "relation_auxiliary_pairs": float(
                sum(kind == "relation_auxiliary" for kind, _ in training_items)
            ),
            "positive_labels": float(positive_labels),
            "descriptor_learning_rate": learning_rates.get("descriptor", 0.0),
            "context_learning_rate": learning_rates.get("context", 0.0),
            "visual_context_learning_rate": learning_rates.get(
                "visual_context", 0.0
            ),
            "direct_evidence_learning_rate": learning_rates.get(
                "direct_evidence", 0.0
            ),
            "relation_learning_rate": learning_rates.get("relations", 0.0),
            "pairwise_learning_rate": learning_rates.get("pairwise", 0.0),
            "layout_learning_rate": learning_rates.get("layout", 0.0),
            "pair_state_learning_rate": learning_rates.get("pair_state", 0.0),
            "neighbor_context_learning_rate": learning_rates.get(
                "neighbor_context", 0.0
            ),
            "correspondence_core_learning_rate": learning_rates.get(
                "correspondence_core", 0.0
            ),
            "relation_invariance_weight": float(relation_invariance_weight),
            "hard_negative_weight": float(hard_negative_weight),
            "hard_negative_margin": float(hard_negative_margin),
            "refinement_stability_weight": float(
                refinement_stability_weight
            ),
            "active_state_weight": float(active_state_weight),
            "relation_contrastive_weight": float(relation_contrastive_weight),
            "page_embedding_weight": float(page_embedding_weight),
            "unlabeled_page_embedding_weight": float(
                unlabeled_page_embedding_weight
            ),
            "unlabeled_page_batch_size": float(unlabeled_page_batch_size),
            "papt_update_ratio": float(papt_update_ratio),
            "papt_updates": float(
                sum(kind == "unlabeled_page_batch" for kind, _ in training_items)
            ),
            "cross_page_soft_teacher_weight": float(
                cross_page_soft_teacher_weight
            ),
            "teacher_confidence_threshold": float(teacher_confidence_threshold),
            "teacher_margin_threshold": float(teacher_margin_threshold),
            "teacher_temperature": float(teacher_temperature),
            "teacher_min_stable_layers": float(teacher_min_stable_layers),
            **_mean_loss_details(loss_details),
        }
        if dev_pairs:
            matcher.to(dev_device)
            dev_metrics = evaluate_correspondence_pairs(
                matcher,
                dev_pairs,
                device=dev_device,
                matcher_config=config,
                latency_repeats=0,
            )
            matcher.to(device)
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
    if (
        train_only_visual_context
        or train_only_relations
        or train_only_direct_evidence
        or train_only_pairwise_correspondence
        or train_only_layout
        or train_only_pair_state
        or train_only_neighbor_context
        or train_only_correspondence_core
    ):
        for parameter in matcher.parameters():
            parameter.requires_grad_(True)
    matcher.eval()
    return matcher, history


def _page_infonce_batches(
    pairs: Iterable[UnlabeledPagePair],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[UnlabeledPagePair, ...]]:
    """Build batches with same-app hard negatives when metadata permits."""

    if batch_size < 2:
        raise ValueError("page InfoNCE batch size must be at least two")
    pool = list(pairs)
    grouped: dict[str, list[UnlabeledPagePair]] = {}
    for pair in pool:
        if pair.group_id:
            grouped.setdefault(pair.group_id, []).append(pair)
    anchors: list[tuple[UnlabeledPagePair, UnlabeledPagePair]] = []
    anchored_ids: set[int] = set()
    for group in grouped.values():
        rng.shuffle(group)
        for index in range(0, len(group) - 1, 2):
            first, second = group[index : index + 2]
            anchors.append((first, second))
            anchored_ids.update((id(first), id(second)))
    rng.shuffle(anchors)
    remaining = [pair for pair in pool if id(pair) not in anchored_ids]
    rng.shuffle(remaining)
    batches: list[tuple[UnlabeledPagePair, ...]] = []
    while anchors:
        batch: list[UnlabeledPagePair] = []
        while anchors and len(batch) + 2 <= batch_size:
            batch.extend(anchors.pop())
        while remaining and len(batch) < batch_size:
            batch.append(remaining.pop())
        batches.append(tuple(batch))
    while len(remaining) >= 2:
        batch = tuple(remaining[-batch_size:])
        del remaining[-len(batch) :]
        if len(batch) >= 2:
            batches.append(batch)
    return batches


def _select_kept_nodes(
    nodes: tuple[UINode, ...],
    *,
    rng: random.Random,
    config: AugmentConfig,
    protected_node_ids: frozenset[str] = frozenset(),
) -> list[UINode]:
    if not nodes:
        return []
    # Node retention is structural, never actionability-based. Non-clickable
    # labels, icons and containers are first-class context for correspondence.
    kept = [
        node
        for node in nodes
        if node.node_id in protected_node_ids
        or rng.random() >= config.drop_node_prob
    ]
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
    semantic_values: tuple[str, str],
) -> UINode:
    text, content_desc = semantic_values
    mask_text = rng.random() < config.mask_text_prob
    mask_content_desc = rng.random() < config.mask_content_desc_prob
    visual_disabled = bool(node.metadata.get("visual_disabled")) or (
        rng.random() < config.visual_dropout_prob
    )
    return UINode(
        node_id=f"{view_id}:{node.node_id}",
        origin_id=node.origin_id,
        parent_id=f"{view_id}:{parent_id}" if parent_id else None,
        text="" if mask_text else text,
        content_desc=(
            "" if mask_content_desc else content_desc
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
            "original_node_id": node.node_id,
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
        raise ValueError("partial assignment has no ranked target candidates")
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
        raise ValueError("supervised targets must be ranked candidates")
    row_indices = torch.tensor(supervised_rows, dtype=torch.long, device=device)
    column_indices = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    selected_logits = logits[row_indices][:, column_indices]
    log_probabilities = torch.log_softmax(selected_logits, dim=1)
    positive_coordinates = tuple(
        (row_position, candidate_positions[target])
        for row_position, row_index in enumerate(supervised_rows)
        for target in positive_targets[row_index]
    )
    positive_mask = torch.zeros(
        log_probabilities.shape,
        dtype=torch.bool,
        device=device,
    )
    coordinates = torch.tensor(
        positive_coordinates,
        dtype=torch.long,
        device=device,
    )
    positive_mask[coordinates[:, 0], coordinates[:, 1]] = True
    positive_log_probability = torch.logsumexp(
        log_probabilities.masked_fill(~positive_mask, float("-inf")),
        dim=1,
    )
    return -positive_log_probability.mean()


def _refinement_objective(
    layer_scores: tuple[Any, ...],
    positive_targets: tuple[tuple[int, ...], ...],
    *,
    candidate_indices: tuple[int, ...],
    hard_negative_margin: float,
    torch: Any,
    device: str,
) -> tuple[Any, Any]:
    """Penalize the strongest non-gold and any loss of gold margin by refinement."""

    if not layer_scores:
        raise ValueError("refinement objective requires at least one layer")
    zero = layer_scores[-1].sum() * 0.0
    candidate_positions = {
        index: position for position, index in enumerate(candidate_indices)
    }
    positive_mask = torch.zeros(
        (len(positive_targets), len(candidate_indices)),
        dtype=torch.bool,
        device=device,
    )
    positive_rows = []
    positive_columns = []
    for row_index, targets in enumerate(positive_targets):
        for target in targets:
            position = candidate_positions.get(target)
            if position is not None:
                positive_rows.append(row_index)
                positive_columns.append(position)
    if positive_rows:
        positive_mask[
            torch.tensor(positive_rows, dtype=torch.long, device=device),
            torch.tensor(positive_columns, dtype=torch.long, device=device),
        ] = True
    valid = positive_mask.any(dim=1) & (~positive_mask).any(dim=1)
    if not bool(valid.any()):
        return zero, zero
    candidate_index = torch.tensor(
        candidate_indices, dtype=torch.long, device=device
    )
    margins = []
    for layer in layer_scores:
        selected = layer[:, candidate_index]
        gold_score = torch.logsumexp(
            selected.masked_fill(~positive_mask, -torch.inf), dim=1
        )
        negative_score = selected.masked_fill(positive_mask, -torch.inf).max(dim=1).values
        margins.append(gold_score - negative_score)
    margin_stack = torch.stack(margins)
    hard_loss = torch.nn.functional.softplus(
        float(hard_negative_margin) - margin_stack[-1, valid]
    ).mean()
    stability_loss = (
        torch.relu(
            margin_stack[:-1, valid].detach() - margin_stack[1:, valid]
        ).mean()
        if len(layer_scores) > 1
        else zero
    )
    return hard_loss, stability_loss










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
        "hard_negative_loss",
        "hard_negative_weight",
        "hard_negative_margin",
        "refinement_stability_loss",
        "refinement_stability_weight",
        "active_state_reconstruction_loss",
        "active_state_weight",
        "active_state_observed_labels",
        "page_embedding_loss",
        "page_embedding_weight",
        "page_embedding_positive_cosine",
        "page_embedding_negative_cosine",
        "unlabeled_page_embedding_loss",
        "relation_invariance_loss",
        "relation_invariance_raw_loss",
        "relation_invariance_rows",
        "relation_invariance_layers",
        "relation_contrastive_loss",
        "relation_contrastive_weight",
        "soft_teacher_rows",
        "soft_teacher_candidate_rows",
        "soft_teacher_mean_confidence",
        "soft_teacher_mean_margin",
        "soft_teacher_bidirectional_rows",
        "soft_teacher_stable_rows",
        "unclipped_gradient_norm",
        "gradient_clip_triggered",
        "transport_output_gradient_norm",
        "transport_upstream_gradient_norm",
    )
    return {
        key: sum(row.get(key, 0.0) for row in rows) / max(len(rows), 1)
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
