"""Geometric-v9 feature encoding, checkpoint I/O, and inference adapter."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from omnitransfer.ui_graph import (
    BBox,
    UIGraph,
    UINode,
    local_context_graph,
    multi_anchor_context_graph,
    visual_bbox_fraction,
)

RELATION_FEATURE_DIM = 18
XML_NODE_FEATURE_DIM = 32
TEXT_DESCRIPTOR_DIM = 48
VISUAL_DESCRIPTOR_DIM = 48
MULTISCALE_VISUAL_DESCRIPTOR_DIM = VISUAL_DESCRIPTOR_DIM * 2
XML_DESCRIPTOR_DIM = 32
NODE_DESCRIPTOR_DIM = (
    TEXT_DESCRIPTOR_DIM + VISUAL_DESCRIPTOR_DIM + XML_DESCRIPTOR_DIM
)
GEOMETRIC_FEATURE_SCHEMA_ID = "omnitransfer-direct-pair-evidence-v7"
OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE = (
    "omnitransfer_geometric_alignment_v9"
)
DIRECT_PAIR_EVIDENCE_NAMES = (
    "semantic_exact",
    "semantic_token_dice",
    "semantic_trigram_dice",
    "semantic_containment",
    "both_semantic_present",
    "exactly_one_semantic_present",
    "editable_both",
    "editable_mismatch",
    "scrollable_both",
    "scrollable_mismatch",
    "toggle_both",
    "toggle_mismatch",
    "clickable_both",
    "clickable_mismatch",
    "class_hash_cosine",
    "class_token_dice",
    "enabled_both",
    "enabled_mismatch",
)
TYPED_RELATION_NAMES = (
    "identity",
    "parent",
    "child",
    "sibling",
    "ancestor",
    "descendant",
    "same_row",
    "same_column",
    "overlap",
    "neighbor_left",
    "neighbor_right",
    "neighbor_above",
    "neighbor_below",
    "near",
    "kinship_proximity",
    "control_to_context",
    "context_to_control",
)
ALIGNMENT_RELATION_FEATURE_INDICES = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    17,
)
LEARNED_TOKEN_LOOKUP_ENCODER = "learned_token_lookup"
DIRECT_TEXT_EVIDENCE_ENCODER = "direct_text_evidence"
ALL_NODE_CANDIDATE_POLICY = "all_nodes"
LEGACY_GLOBAL_POOL_VISUAL_ENCODER = "legacy_global_pool_v1"
SPATIAL_CNN_VISUAL_ENCODER = "spatial_cnn_v2"
DETERMINISTIC_ICON_VISUAL_ENCODER = "deterministic_icon_v1"
MULTISCALE_HASH_VISUAL_ENCODER = "multiscale_hash_v2"


@dataclass(frozen=True)
class MatcherConfig:
    """Configuration for the compact learned matcher."""

    vocab_size: int = 8192
    max_tokens: int = 48
    token_dim: int = 48
    hidden_dim: int = NODE_DESCRIPTOR_DIM
    relation_hidden_dim: int = 24
    association_dim: int = 64
    association_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    visual_patch_size: int = 32
    visual_canvas_size: int = 384
    visual_encoder: str = DETERMINISTIC_ICON_VISUAL_ENCODER
    visual_context_scale: float = 3.0
    source_context_nodes: int = 48
    target_context_nodes: int = 64
    architecture: str = OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
    assignment_head: str = "partial_assignment"
    text_encoder: str = LEARNED_TOKEN_LOOKUP_ENCODER
    candidate_policy: str = ALL_NODE_CANDIDATE_POLICY


def visual_descriptor_dim(visual_encoder: str) -> int:
    return (
        MULTISCALE_VISUAL_DESCRIPTOR_DIM
        if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER
        else VISUAL_DESCRIPTOR_DIM
    )


@dataclass(frozen=True)
class EncodedGraph:
    """Torch-independent learned-matcher inputs for one UI graph."""

    graph_id: str
    node_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    numeric_features: tuple[tuple[float, ...], ...]
    relation_features: Any


@dataclass(frozen=True)
class _RelationContext:
    graph: UIGraph
    bboxes: tuple[BBox | None, ...]
    centers: tuple[tuple[float, float], ...]
    sizes: tuple[tuple[float, float], ...]
    node_indices: dict[str, int]
    ancestor_sets: dict[str, frozenset[str]]
    ancestor_paths: dict[str, tuple[str, ...]]
    path_positions: dict[str, dict[str, int]]


@dataclass(frozen=True)
class LearnedMatch:
    """One learned source-to-target match or a safe abstention."""

    target_node: UINode | None
    probability: float
    margin: float
    reason: str
    scores: tuple[tuple[str, float], ...]


def encode_graph(
    graph: UIGraph,
    *,
    config: MatcherConfig | None = None,
    feature_schema_id: str | None = None,
) -> EncodedGraph:
    """Encode UI attributes and pairwise relations without fixed match weights."""

    cfg = config or MatcherConfig()
    feature_schema_id = feature_schema_id or GEOMETRIC_FEATURE_SCHEMA_ID
    if feature_schema_id != GEOMETRIC_FEATURE_SCHEMA_ID:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    return EncodedGraph(
        graph_id=graph.graph_id,
        node_ids=tuple(node.node_id for node in graph.nodes),
        origin_ids=tuple(node.origin_id for node in graph.nodes),
        token_ids=tuple(
            _multimodal_text_token_ids(node, config=cfg)
            for node in graph.nodes
        ),
        numeric_features=tuple(
            _multimodal_xml_features(node, graph) for node in graph.nodes
        ),
        relation_features=_relation_features(graph),
    )


def cross_relation_features(
    source: UIGraph,
    target: UIGraph,
    *,
    feature_schema_id: str = GEOMETRIC_FEATURE_SCHEMA_ID,
) -> Any:
    """Return direct pair evidence under the geometric-v9 contract."""

    if feature_schema_id != GEOMETRIC_FEATURE_SCHEMA_ID:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    return tuple(
        tuple(
            direct_pair_evidence_features(source_node, target_node)
            for target_node in target.nodes
        )
        for source_node in source.nodes
    )


def direct_pair_evidence_features(
    source: UINode,
    target: UINode,
) -> tuple[float, ...]:
    """Return symmetric shortcut-free evidence for one candidate node pair."""

    source_fields = _semantic_fields(source)
    target_fields = _semantic_fields(target)
    source_tokens = _semantic_tokens(source_fields)
    target_tokens = _semantic_tokens(target_fields)
    source_trigrams = _semantic_trigrams(source_fields)
    target_trigrams = _semantic_trigrams(target_fields)
    semantic_exact = bool(
        source_fields
        and target_fields
        and any(left == right for left in source_fields for right in target_fields)
    )
    semantic_containment = bool(
        source_fields
        and target_fields
        and any(
            (left in right or right in left) and min(len(left), len(right)) >= 3
            for left in source_fields
            for right in target_fields
        )
    )
    source_class = _normalize_text(source.class_name)
    target_class = _normalize_text(target.class_name)
    source_class_tokens = set(source_class.split())
    target_class_tokens = set(target_class.split())
    source_class_hash = _hashed_attribute_features(
        source.class_name,
        XML_NODE_FEATURE_DIM - 4,
    )
    target_class_hash = _hashed_attribute_features(
        target.class_name,
        XML_NODE_FEATURE_DIM - 4,
    )
    source_toggle = _is_toggle_node(source)
    target_toggle = _is_toggle_node(target)
    values = (
        float(semantic_exact),
        _set_dice(source_tokens, target_tokens),
        _set_dice(source_trigrams, target_trigrams),
        float(semantic_containment),
        float(bool(source_fields) and bool(target_fields)),
        float(bool(source_fields) != bool(target_fields)),
        float(source.editable and target.editable),
        float(source.editable != target.editable),
        float(source.scrollable and target.scrollable),
        float(source.scrollable != target.scrollable),
        float(source_toggle and target_toggle),
        float(source_toggle != target_toggle),
        float(source.clickable and target.clickable),
        float(source.clickable != target.clickable),
        sum(
            left * right
            for left, right in zip(
                source_class_hash,
                target_class_hash,
                strict=True,
            )
        ),
        _set_dice(source_class_tokens, target_class_tokens),
        float(source.enabled and target.enabled),
        float(source.enabled != target.enabled),
    )
    if len(values) != len(DIRECT_PAIR_EVIDENCE_NAMES):
        raise AssertionError("unexpected direct pair evidence dimension")
    if len(values) != RELATION_FEATURE_DIM:
        raise AssertionError("direct pair evidence must fit the cross-pair tensor")
    return values


def mutual_log_assignment(affinity: Any) -> Any:
    """Turn one dense affinity matrix into a bidirectionally normalized score."""

    torch = _require_torch()
    if affinity.ndim != 2:
        raise ValueError("affinity must be a two-dimensional matrix")
    if affinity.shape[0] == 0 or affinity.shape[1] == 0:
        raise ValueError("affinity must contain at least one source and target node")
    return 0.5 * (
        torch.log_softmax(affinity, dim=1)
        + torch.log_softmax(affinity, dim=0)
    )


def confidence_adaptive_assignment_row(
    output: dict[str, Any],
    *,
    source_index: int,
    candidate_indices: Iterable[int],
    transpose: bool = False,
) -> tuple[Any, int]:
    """Select the sharper of the last two shared association layers."""

    candidates = list(candidate_indices)
    assignment_layers = tuple(output.get("assignment_scores_by_layer") or ())
    selected_layer = len(assignment_layers) - 1
    final_matrix = output["logits_ba"] if transpose else output["logits_ab"]
    selected_logits = final_matrix[source_index][candidates]
    if len(assignment_layers) < 2 or len(candidates) < 2:
        return selected_logits, selected_layer
    previous_matrix = assignment_layers[-2].T if transpose else assignment_layers[-2]
    previous_logits = previous_matrix[source_index][candidates]
    torch = _require_torch()
    previous_probability = torch.softmax(previous_logits, dim=0)
    final_probability = torch.softmax(selected_logits, dim=0)
    previous_margin = torch.topk(previous_probability, 2).values.diff().abs()[0]
    final_margin = torch.topk(final_probability, 2).values.diff().abs()[0]
    if previous_margin > final_margin:
        return previous_logits, len(assignment_layers) - 2
    return selected_logits, selected_layer


def typed_relation_bases(relations: Any, numeric_features: Any) -> Any:
    """Build row-normalized UI relation bases for structured matching."""

    torch = _require_torch()
    if relations.ndim != 3 or relations.shape[-1] != RELATION_FEATURE_DIM:
        raise ValueError(
            f"relations must have shape [nodes, nodes, {RELATION_FEATURE_DIM}]"
        )
    node_count = int(relations.shape[0])
    if relations.shape[1] != node_count:
        raise ValueError("typed relations require one square within-page graph")
    if numeric_features.ndim != 2 or numeric_features.shape[0] != node_count:
        raise ValueError("numeric features must contain one row per node")
    if numeric_features.shape[1] < 4:
        raise ValueError("numeric features must expose action and enabled state")

    identity = relations[..., 0].clamp(0.0, 1.0)
    non_identity = 1.0 - identity
    local = relations[..., 17].clamp(0.0, 1.0) * non_identity
    same_row = relations[..., 6].clamp(0.0, 1.0) * local
    same_column = relations[..., 7].clamp(0.0, 1.0) * local
    kinship_proximity = (
        (1.0 - relations[..., 16].clamp(0.0, 1.0)) * local
    )
    delta_x = relations[..., 9]
    delta_y = relations[..., 10]
    actionable = (
        numeric_features[:, :3].sum(dim=1).gt(0.0)
        & numeric_features[:, 3].gt(0.0)
    )
    context = ~actionable
    control_to_context = (
        actionable[:, None].to(relations.dtype)
        * context[None, :].to(relations.dtype)
        * local
    )
    context_to_control = (
        context[:, None].to(relations.dtype)
        * actionable[None, :].to(relations.dtype)
        * local
    )
    bases = torch.stack(
        (
            identity,
            relations[..., 1].clamp(0.0, 1.0),
            relations[..., 2].clamp(0.0, 1.0),
            relations[..., 3].clamp(0.0, 1.0),
            relations[..., 4].clamp(0.0, 1.0),
            relations[..., 5].clamp(0.0, 1.0),
            same_row,
            same_column,
            relations[..., 8].clamp(0.0, 1.0) * non_identity,
            delta_x.lt(0.0).to(relations.dtype) * same_row,
            delta_x.gt(0.0).to(relations.dtype) * same_row,
            delta_y.lt(0.0).to(relations.dtype) * same_column,
            delta_y.gt(0.0).to(relations.dtype) * same_column,
            local,
            kinship_proximity,
            control_to_context,
            context_to_control,
        ),
        dim=0,
    )
    normalizer = bases.sum(dim=-1, keepdim=True)
    return torch.where(
        normalizer > 0.0,
        bases / normalizer.clamp_min(torch.finfo(bases.dtype).eps),
        bases,
    )






def build_geometric_v9_matcher(
    config: MatcherConfig | None = None,
) -> Any:
    """Build the canonical geometric-v9 matcher."""

    cfg = config or MatcherConfig()
    if cfg.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("only the geometric-v9 matcher is supported")
    from omnitransfer.geometric_matcher import build_geometric_matcher

    return build_geometric_matcher(cfg)


def matcher_inputs(
    source: UIGraph,
    target: UIGraph,
    *,
    config: MatcherConfig | None = None,
    device: str | Any = "cpu",
    feature_schema_id: str | None = None,
) -> tuple[Any, ...]:
    """Convert two complete UI graphs to geometric-v9 tensors."""

    torch = _require_torch()
    cfg = config or MatcherConfig()
    feature_schema_id = feature_schema_id or GEOMETRIC_FEATURE_SCHEMA_ID
    encoded_source = encode_graph(
        source,
        config=cfg,
        feature_schema_id=feature_schema_id,
    )
    encoded_target = encode_graph(
        target,
        config=cfg,
        feature_schema_id=feature_schema_id,
    )
    source_token_ids = torch.as_tensor(
        encoded_source.token_ids,
        dtype=torch.long,
        device=device,
    )
    target_token_ids = torch.as_tensor(
        encoded_target.token_ids,
        dtype=torch.long,
        device=device,
    )
    source_numeric = torch.as_tensor(
        encoded_source.numeric_features,
        dtype=torch.float32,
        device=device,
    )
    target_numeric = torch.as_tensor(
        encoded_target.numeric_features,
        dtype=torch.float32,
        device=device,
    )
    source_relations = torch.as_tensor(
        encoded_source.relation_features,
        dtype=torch.float32,
        device=device,
    )
    target_relations = torch.as_tensor(
        encoded_target.relation_features,
        dtype=torch.float32,
        device=device,
    )
    # The contextual matcher compares final node states as one dense matrix.
    # Keep this compatibility input shape for exporters, but do not rebuild the
    # retired Python-level candidate-pair evidence table on every mapping.
    pair_relations = torch.zeros(
        (
            len(source.nodes),
            len(target.nodes),
            len(DIRECT_PAIR_EVIDENCE_NAMES),
        ),
        dtype=torch.float32,
        device=device,
    )
    source_visual, source_visual_mask = _visual_inputs(
        source,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        visual_encoder=cfg.visual_encoder,
        context_scale=cfg.visual_context_scale,
        torch=torch,
        device=device,
    )
    target_visual, target_visual_mask = _visual_inputs(
        target,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        visual_encoder=cfg.visual_encoder,
        context_scale=cfg.visual_context_scale,
        torch=torch,
        device=device,
    )
    return (
        source_token_ids,
        source_numeric,
        source_relations,
        target_token_ids,
        target_numeric,
        target_relations,
        pair_relations,
        source_visual,
        source_visual_mask,
        target_visual,
        target_visual_mask,
    )


class GeometricMatcher:
    """Inference adapter that abstains instead of replaying source coordinates."""

    def __init__(
        self,
        model: Any,
        *,
        config: MatcherConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config or MatcherConfig()
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> GeometricMatcher:
        torch = _require_torch()
        payload = torch.load(Path(path), map_location=device)
        config_payload = dict(payload["matcher_config"])
        if "visual_encoder" not in config_payload:
            config_payload["visual_encoder"] = LEGACY_GLOBAL_POOL_VISUAL_ENCODER
        config = MatcherConfig(**config_payload)
        if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
            raise ValueError("only geometric-v9 checkpoints are supported")
        model = build_geometric_v9_matcher(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, config=config, device=device)

    def predict(
        self,
        source: UIGraph,
        target: UIGraph,
        *,
        source_node_id: str,
        candidate_node_ids: Iterable[str] | None = None,
        min_probability: float = 0.0,
        min_margin: float = 0.0,
    ) -> LearnedMatch:
        torch = _require_torch()
        source_node = next(
            (node for node in source.nodes if node.node_id == source_node_id),
            None,
        )
        if source_node is None:
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        source_context = local_context_graph(
            source,
            anchor_node_id=source_node_id,
            max_nodes=self.config.source_context_nodes,
        )
        source_index = next(
            (
                index
                for index, node in enumerate(source_context.nodes)
                if node.node_id == source_node_id
            ),
            None,
        )
        if source_index is None:
            raise AssertionError("source context dropped its anchor node")
        target_node_ids = {node.node_id for node in target.nodes}
        allowed = set(
            candidate_node_ids or (node.node_id for node in target.nodes)
        ) & target_node_ids
        if not allowed:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        target_context = multi_anchor_context_graph(
            target,
            anchor_node_ids=allowed,
            max_nodes=max(self.config.target_context_nodes, len(allowed)),
        )
        candidate_indices = [
            index
            for index, node in enumerate(target_context.nodes)
            if node.node_id in allowed
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        inputs = matcher_inputs(
            source_context,
            target_context,
            config=self.config,
            device=self.device,
            feature_schema_id=GEOMETRIC_FEATURE_SCHEMA_ID,
        )
        with torch.no_grad():
            output = self.model(*inputs)
            affinity_layers = tuple(output.get("affinities_by_layer") or ())
            selected_logits, selected_layer = confidence_adaptive_assignment_row(
                output,
                source_index=source_index,
                candidate_indices=candidate_indices,
            )
            selected_affinity_matrix = (
                affinity_layers[selected_layer]
                if affinity_layers
                else output["affinity"]
            )
            selected_affinity = selected_affinity_matrix[source_index][candidate_indices]
            rank_probabilities = torch.softmax(selected_logits, dim=0)
            match_probabilities = torch.sigmoid(selected_affinity)
        ranked = sorted(
            (
                (
                    target_context.nodes[index].node_id,
                    float(rank_probabilities[position]),
                )
                for position, index in enumerate(candidate_indices)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        best_position = next(
            position
            for position, index in enumerate(candidate_indices)
            if target_context.nodes[index].node_id == best_id
        )
        match_probability = float(match_probabilities[best_position])
        second_probability = max(
            (score for _, score in ranked[1:]),
            default=0.0,
        )
        margin = best_probability - second_probability
        scores = tuple(ranked)
        if match_probability < min_probability or margin < min_margin:
            return LearnedMatch(
                None, match_probability, margin, "learned_low_confidence", scores
            )
        target_node = next(node for node in target.nodes if node.node_id == best_id)
        return LearnedMatch(
            target_node, match_probability, margin, "learned_match", scores
        )






def save_matcher_checkpoint(
    path: str | Path,
    model: Any,
    *,
    config: MatcherConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a state-dict checkpoint with an explicit architecture contract."""

    torch = _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("only geometric-v9 checkpoints are supported")
    torch.save(
        {
            "schema_version": "omnitransfer.learned_geometric_alignment.v9",
            "matcher_config": asdict(config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )


def parameter_count(model: Any) -> int:
    """Return trainable parameter count for experiment reporting."""

    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def initialize_direct_text_from_lookup(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Remove the learned lookup while preserving its zero-output behavior.

    Direct lexical pair evidence remains unchanged.  For node encoding, a
    text-bearing node receives the old projection bias, which is exactly what
    the lookup model emits when its embedding table is zeroed.
    """

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    if "token_embedding.weight" not in source_state:
        raise ValueError("source model has no learned token lookup")
    if "present_text" not in target_state:
        raise ValueError("target model is not a direct-text model")
    transferred: list[str] = []
    for target_name, target_value in target_state.items():
        source_name = (
            "text_projection.bias"
            if target_name == "present_text"
            else target_name
        )
        source_value = source_state.get(source_name)
        if source_value is None or source_value.shape != target_value.shape:
            raise ValueError(
                f"direct-text migration cannot initialize {target_name}"
            )
        target_state[target_name] = source_value.detach().clone()
        transferred.append(target_name)
    target_model.load_state_dict(target_state)
    return tuple(transferred)


def initialize_nonvisual_from_model(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Migrate geometric-v9 while replacing only its visual encoder."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred: list[str] = []
    for target_name, target_value in target_state.items():
        if target_name.startswith("visual_encoder."):
            continue
        source_value = source_state.get(target_name)
        if (
            source_value is not None
            and target_name == "missing_visual"
            and target_value.ndim == source_value.ndim == 1
            and target_value.shape[0] == source_value.shape[0] * 2
        ):
            widened = target_value.detach().clone().zero_()
            widened[: source_value.shape[0]] = source_value.detach()
            target_state[target_name] = widened
            transferred.append(target_name)
            continue
        if (
            source_value is not None
            and target_name == "visual_to_hidden.weight"
            and target_value.ndim == source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1] == source_value.shape[1] * 2
        ):
            widened = target_value.detach().clone().zero_()
            widened[:, : source_value.shape[1]] = source_value.detach()
            target_state[target_name] = widened
            transferred.append(target_name)
            continue
        if source_value is None or source_value.shape != target_value.shape:
            if target_name == "missing_visual" or target_name.startswith(
                "visual_to_hidden."
            ):
                continue
            raise ValueError(
                f"visual encoder migration cannot initialize {target_name}"
            )
        target_state[target_name] = source_value.detach().clone()
        transferred.append(target_name)
    target_model.load_state_dict(target_state)
    return tuple(transferred)


def prepare_visual_asset(
    screenshot_path: str | Path,
    *,
    canvas_size: int = 384,
) -> tuple[int, int, int]:
    """Decode and resize one screenshot before latency-critical matching."""

    path = Path(screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"screenshot is missing: {path}")
    array, _ = _load_rgb_array(str(path.resolve()), canvas_size)
    return tuple(int(value) for value in array.shape)


def _visual_inputs(
    graph: UIGraph,
    *,
    patch_size: int,
    canvas_size: int,
    visual_encoder: str = DETERMINISTIC_ICON_VISUAL_ENCODER,
    context_scale: float = 3.0,
    torch: Any,
    device: str | Any,
) -> tuple[Any, Any]:
    screenshot_path = str(graph.metadata.get("screenshot_path") or "")
    channels = 6 if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER else 3
    empty = torch.zeros(
        (len(graph.nodes), channels, patch_size, patch_size),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(graph.nodes), 1), dtype=torch.float32, device=device)
    if not screenshot_path:
        return empty, mask
    path = Path(screenshot_path)
    if not path.is_file():
        return empty, mask
    image_array, original_size = _load_rgb_array(str(path.resolve()), canvas_size)
    image = torch.as_tensor(image_array, dtype=torch.uint8, device=device)
    image = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
    normalized_boxes: list[tuple[float, float, float, float]] = []
    normalized_context_boxes: list[tuple[float, float, float, float]] = []
    available: list[float] = []
    for node in graph.nodes:
        if bool(node.metadata.get("visual_disabled")):
            normalized_boxes.append((-1.0, -1.0, -1.0, -1.0))
            normalized_context_boxes.append((-1.0, -1.0, -1.0, -1.0))
            available.append(0.0)
            continue
        bbox = visual_bbox_fraction(
            graph,
            node,
            image_width=original_size[0],
            image_height=original_size[1],
        )
        if bbox is None:
            normalized_boxes.append((-1.0, -1.0, -1.0, -1.0))
            normalized_context_boxes.append((-1.0, -1.0, -1.0, -1.0))
            available.append(0.0)
            continue
        normalized_boxes.append(
            (
                2.0 * bbox[0] - 1.0,
                2.0 * bbox[1] - 1.0,
                2.0 * bbox[2] - 1.0,
                2.0 * bbox[3] - 1.0,
            )
        )
        context_bbox = _expanded_bbox_fraction(bbox, context_scale)
        normalized_context_boxes.append(
            (
                2.0 * context_bbox[0] - 1.0,
                2.0 * context_bbox[1] - 1.0,
                2.0 * context_bbox[2] - 1.0,
                2.0 * context_bbox[3] - 1.0,
            )
        )
        available.append(1.0)
    boxes = torch.as_tensor(
        normalized_boxes,
        dtype=torch.float32,
        device=device,
    )
    patches = _sample_visual_boxes(
        image,
        boxes,
        patch_size=patch_size,
        torch=torch,
        device=device,
    )
    if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER:
        context_boxes = torch.as_tensor(
            normalized_context_boxes,
            dtype=torch.float32,
            device=device,
        )
        context_patches = _sample_visual_boxes(
            image,
            context_boxes,
            patch_size=patch_size,
            torch=torch,
            device=device,
        )
        patches = torch.cat((patches, context_patches), dim=1)
    patches = _apply_visual_transform(patches, graph=graph, torch=torch, device=device)
    return (
        patches,
        torch.tensor(available, dtype=torch.float32, device=device).unsqueeze(1),
    )


def _sample_visual_boxes(
    image: Any,
    boxes: Any,
    *,
    patch_size: int,
    torch: Any,
    device: str | Any,
) -> Any:
    offsets = torch.linspace(0.0, 1.0, patch_size, device=device)
    grid_x = (
        boxes[:, 0, None, None]
        + (boxes[:, 2] - boxes[:, 0])[:, None, None] * offsets[None, None, :]
    )
    grid_y = (
        boxes[:, 1, None, None]
        + (boxes[:, 3] - boxes[:, 1])[:, None, None] * offsets[None, :, None]
    )
    grid_x = grid_x.expand(-1, patch_size, -1)
    grid_y = grid_y.expand(-1, -1, patch_size)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return torch.nn.functional.grid_sample(
        image.expand(len(boxes), -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _expanded_bbox_fraction(
    bbox: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    scale = max(1.0, float(scale))
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    half_width = (right - left) * scale * 0.5
    half_height = (bottom - top) * scale * 0.5
    return (
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(1.0, center_x + half_width),
        min(1.0, center_y + half_height),
    )


def _visual_bbox(node: UINode) -> BBox | None:
    value = node.metadata.get("visual_bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
        except (TypeError, ValueError):
            return node.bbox
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return node.bbox


def _apply_visual_transform(
    patches: Any,
    *,
    graph: UIGraph,
    torch: Any,
    device: str | Any,
) -> Any:
    value = graph.metadata.get("visual_transform")
    if not isinstance(value, dict):
        return patches
    try:
        brightness = float(value.get("brightness") or 0.0)
        contrast = float(value.get("contrast") or 1.0)
        channel_scale = tuple(float(item) for item in value.get("channel_scale") or ())
    except (TypeError, ValueError):
        return patches
    if len(channel_scale) != 3 or contrast <= 0.0:
        return patches
    gains = torch.tensor(
        channel_scale,
        dtype=patches.dtype,
        device=device,
    ).view(1, 3, 1, 1)
    if patches.shape[1] != 3:
        gains = gains.repeat(1, patches.shape[1] // 3, 1, 1)
    return (((patches - 0.5) * contrast + 0.5 + brightness) * gains).clamp_(0.0, 1.0)


@lru_cache(maxsize=32)
def _load_rgb_array(
    screenshot_path: str,
    canvas_size: int,
) -> tuple[Any, tuple[int, int]]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Visual UI encoding requires NumPy and Pillow. Install omnitransfer[train]."
        ) from exc

    with Image.open(screenshot_path) as image:
        image = image.convert("RGB")
        original_size = (image.width, image.height)
        if canvas_size > 0 and max(image.size) > canvas_size:
            scale = canvas_size / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        return np.array(image, dtype=np.uint8, copy=True), original_size




def _multimodal_text_token_ids(
    node: UINode,
    *,
    config: MatcherConfig,
) -> tuple[int, ...]:
    pieces: list[str] = []
    for field_name, value in (
        ("text", node.text),
        ("desc", node.content_desc),
    ):
        normalized = _normalize_text(value)
        if not normalized:
            continue
        pieces.append(f"field:{field_name}")
        words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        for word in words:
            pieces.append(f"{field_name}:word:{word}")
            if len(word) >= 3:
                padded = f"^{word}$"
                pieces.extend(
                    f"{field_name}:ngram:{padded[index : index + 3]}"
                    for index in range(len(padded) - 2)
                )
    token_ids: list[int] = []
    seen: set[int] = set()
    for piece in pieces:
        token_id = _token_bucket(piece, config.vocab_size)
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
        if len(token_ids) >= config.max_tokens:
            break
    return tuple(token_ids + [0] * (config.max_tokens - len(token_ids)))










def _multimodal_xml_features(
    node: UINode,
    graph: UIGraph,
) -> tuple[float, ...]:
    bbox = _normalized_bbox(node.bbox, graph)
    if bbox is None:
        has_bbox = width = height = area = aspect = 0.0
    else:
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        aspect = _clip(
            math.log(max(width, 1e-6) / max(height, 1e-6)) / 4.0,
            -1.0,
            1.0,
        )
        has_bbox = 1.0
    class_features = _hashed_attribute_features(
        node.class_name,
        XML_NODE_FEATURE_DIM - 10,
    )
    values = (
        float(node.clickable),
        float(node.editable),
        float(node.scrollable),
        float(node.enabled),
        has_bbox,
        width,
        height,
        area,
        aspect,
        min(float(node.depth) / 32.0, 1.0),
        *class_features,
    )
    if len(values) != XML_NODE_FEATURE_DIM:
        raise AssertionError("unexpected XML node feature dimension")
    return values


def _hashed_attribute_features(value: str, dimension: int) -> tuple[float, ...]:
    if dimension <= 0:
        raise ValueError("hashed attribute dimension must be positive")
    normalized = _normalize_text(value)
    if not normalized:
        return (0.0,) * dimension
    pieces = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    compact = "".join(pieces)
    if len(compact) >= 3:
        padded = f"^{compact}$"
        pieces.extend(
            padded[index : index + 3]
            for index in range(len(padded) - 2)
        )
    values = [0.0] * dimension
    for piece in pieces:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        values[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm > 0.0:
        values = [value / norm for value in values]
    return tuple(values)




def _relation_features(
    graph: UIGraph,
) -> Any:
    context = _relation_context(graph)
    return _relation_matrix(context, context, same_graph=True)


def _node_relation(
    source: UINode,
    target: UINode,
    source_graph: UIGraph,
    target_graph: UIGraph,
) -> tuple[float, ...]:
    same_graph = (
        source_graph is target_graph or source_graph.graph_id == target_graph.graph_id
    )
    source_context = _relation_context(source_graph)
    target_context = _relation_context(target_graph)
    source_index = next(
        index
        for index, node in enumerate(source_graph.nodes)
        if node.node_id == source.node_id
    )
    target_index = next(
        index
        for index, node in enumerate(target_graph.nodes)
        if node.node_id == target.node_id
    )
    return _contextual_node_relation(
        source_index,
        target_index,
        source_context,
        target_context,
        same_graph=same_graph,
    )


def _relation_context(graph: UIGraph) -> _RelationContext:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    bboxes = tuple(_normalized_bbox(node.bbox, graph) for node in graph.nodes)
    ancestor_paths: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes:
        path = [node.node_id]
        current = node
        visited = {node.node_id}
        while current.parent_id and current.parent_id not in visited:
            path.append(current.parent_id)
            visited.add(current.parent_id)
            parent = nodes_by_id.get(current.parent_id)
            if parent is None:
                break
            current = parent
        ancestor_paths[node.node_id] = tuple(path)
    return _RelationContext(
        graph=graph,
        bboxes=bboxes,
        centers=tuple(_center(bbox) for bbox in bboxes),
        sizes=tuple(_size(bbox) for bbox in bboxes),
        node_indices={node.node_id: index for index, node in enumerate(graph.nodes)},
        ancestor_sets={
            node_id: frozenset(path[1:]) for node_id, path in ancestor_paths.items()
        },
        ancestor_paths=ancestor_paths,
        path_positions={
            node_id: {ancestor_id: index for index, ancestor_id in enumerate(path)}
            for node_id, path in ancestor_paths.items()
        },
    )


def _relation_matrix(
    source_context: _RelationContext,
    target_context: _RelationContext,
    *,
    same_graph: bool,
    include_cross_graph_geometry: bool = False,
) -> Any:
    np = _require_numpy()
    source_count = len(source_context.graph.nodes)
    target_count = len(target_context.graph.nodes)
    values = np.zeros(
        (source_count, target_count, RELATION_FEATURE_DIM),
        dtype=np.float32,
    )
    if not same_graph and not include_cross_graph_geometry:
        return values
    source_centers = np.asarray(source_context.centers, dtype=np.float32)
    target_centers = np.asarray(target_context.centers, dtype=np.float32)
    source_sizes = np.asarray(source_context.sizes, dtype=np.float32)
    target_sizes = np.asarray(target_context.sizes, dtype=np.float32)
    delta_x = target_centers[None, :, 0] - source_centers[:, None, 0]
    delta_y = target_centers[None, :, 1] - source_centers[:, None, 1]
    source_width = source_sizes[:, None, 0]
    source_height = source_sizes[:, None, 1]
    target_width = target_sizes[None, :, 0]
    target_height = target_sizes[None, :, 1]
    same_row = (
        np.abs(delta_y)
        <= np.maximum(
            np.maximum(source_height, target_height),
            0.02,
        )
        * 0.5
    )
    same_column = (
        np.abs(delta_x)
        <= np.maximum(
            np.maximum(source_width, target_width),
            0.02,
        )
        * 0.5
    )
    overlap = _pairwise_iou(source_context.bboxes, target_context.bboxes, np=np)
    parent = np.zeros((source_count, target_count), dtype=bool)
    child = np.zeros_like(parent)
    sibling = np.zeros_like(parent)
    ancestor = np.zeros_like(parent)
    descendant = np.zeros_like(parent)
    identity = np.zeros_like(parent)
    tree_distance = np.full((source_count, target_count), 16.0, dtype=np.float32)
    if same_graph:
        source_ids = np.asarray(
            [node.node_id for node in source_context.graph.nodes],
            dtype=object,
        )
        target_ids = np.asarray(
            [node.node_id for node in target_context.graph.nodes],
            dtype=object,
        )
        source_parents = np.asarray(
            [node.parent_id or "" for node in source_context.graph.nodes],
            dtype=object,
        )
        target_parents = np.asarray(
            [node.parent_id or "" for node in target_context.graph.nodes],
            dtype=object,
        )
        identity = source_ids[:, None] == target_ids[None, :]
        parent = source_ids[:, None] == target_parents[None, :]
        child = source_parents[:, None] == target_ids[None, :]
        sibling = (
            (source_parents[:, None] == target_parents[None, :])
            & (source_parents[:, None] != "")
            & ~identity
        )
        for target_index, target_node in enumerate(target_context.graph.nodes):
            for ancestor_id in target_context.ancestor_sets.get(
                target_node.node_id, ()
            ):
                source_index = source_context.node_indices.get(ancestor_id)
                if source_index is not None:
                    ancestor[source_index, target_index] = True
        for source_index, source_node in enumerate(source_context.graph.nodes):
            for ancestor_id in source_context.ancestor_sets.get(
                source_node.node_id, ()
            ):
                target_index = target_context.node_indices.get(ancestor_id)
                if target_index is not None:
                    descendant[source_index, target_index] = True
            for target_index, target_node in enumerate(target_context.graph.nodes):
                tree_distance[source_index, target_index] = _context_tree_distance(
                    source_node.node_id,
                    target_node.node_id,
                    source_context,
                )
    values[..., 0] = identity
    values[..., 1] = parent
    values[..., 2] = child
    values[..., 3] = sibling
    values[..., 4] = ancestor
    values[..., 5] = descendant
    values[..., 6] = same_row
    values[..., 7] = same_column
    values[..., 8] = overlap > 0.0
    values[..., 9] = np.clip(delta_x, -1.0, 1.0)
    values[..., 10] = np.clip(delta_y, -1.0, 1.0)
    values[..., 11] = np.minimum(np.abs(delta_x), 1.0)
    values[..., 12] = np.minimum(np.abs(delta_y), 1.0)
    values[..., 13] = np.clip(
        np.log(np.maximum(target_width, 1e-6) / np.maximum(source_width, 1e-6)) / 4.0,
        -1.0,
        1.0,
    )
    values[..., 14] = np.clip(
        np.log(np.maximum(target_height, 1e-6) / np.maximum(source_height, 1e-6)) / 4.0,
        -1.0,
        1.0,
    )
    values[..., 15] = overlap
    values[..., 16] = np.minimum(tree_distance / 16.0, 1.0)
    values[..., 17] = sibling | parent | child | (np.hypot(delta_x, delta_y) <= 0.25)
    return values


def _pairwise_iou(
    source_bboxes: tuple[BBox | None, ...],
    target_bboxes: tuple[BBox | None, ...],
    *,
    np: Any,
) -> Any:
    source = np.asarray(
        [bbox or (0.0, 0.0, 0.0, 0.0) for bbox in source_bboxes],
        dtype=np.float32,
    )
    target = np.asarray(
        [bbox or (0.0, 0.0, 0.0, 0.0) for bbox in target_bboxes],
        dtype=np.float32,
    )
    left = np.maximum(source[:, None, 0], target[None, :, 0])
    top = np.maximum(source[:, None, 1], target[None, :, 1])
    right = np.minimum(source[:, None, 2], target[None, :, 2])
    bottom = np.minimum(source[:, None, 3], target[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    source_area = np.maximum(0.0, source[:, 2] - source[:, 0]) * np.maximum(
        0.0,
        source[:, 3] - source[:, 1],
    )
    target_area = np.maximum(0.0, target[:, 2] - target[:, 0]) * np.maximum(
        0.0,
        target[:, 3] - target[:, 1],
    )
    union = source_area[:, None] + target_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _contextual_node_relation(
    source_index: int,
    target_index: int,
    source_context: _RelationContext,
    target_context: _RelationContext,
    *,
    same_graph: bool,
) -> tuple[float, ...]:
    source = source_context.graph.nodes[source_index]
    target = target_context.graph.nodes[target_index]
    source_bbox = source_context.bboxes[source_index]
    target_bbox = target_context.bboxes[target_index]
    source_center = source_context.centers[source_index]
    target_center = target_context.centers[target_index]
    delta_x = target_center[0] - source_center[0]
    delta_y = target_center[1] - source_center[1]
    source_width, source_height = source_context.sizes[source_index]
    target_width, target_height = target_context.sizes[target_index]
    parent = same_graph and target.parent_id == source.node_id
    child = same_graph and source.parent_id == target.node_id
    sibling = bool(
        same_graph
        and source.parent_id
        and source.parent_id == target.parent_id
        and source.node_id != target.node_id
    )
    ancestor = same_graph and source.node_id in target_context.ancestor_sets.get(
        target.node_id,
        (),
    )
    descendant = same_graph and target.node_id in source_context.ancestor_sets.get(
        source.node_id,
        (),
    )
    same_row = abs(delta_y) <= max(source_height, target_height, 0.02) * 0.5
    same_column = abs(delta_x) <= max(source_width, target_width, 0.02) * 0.5
    overlap = _iou(source_bbox, target_bbox)
    tree_distance = (
        _context_tree_distance(
            source.node_id,
            target.node_id,
            source_context,
        )
        if same_graph
        else 16
    )
    values = (
        float(same_graph and source.node_id == target.node_id),
        float(parent),
        float(child),
        float(sibling),
        float(ancestor),
        float(descendant),
        float(same_row),
        float(same_column),
        float(overlap > 0.0),
        _clip(delta_x, -1.0, 1.0),
        _clip(delta_y, -1.0, 1.0),
        min(abs(delta_x), 1.0),
        min(abs(delta_y), 1.0),
        _clip(
            math.log(max(target_width, 1e-6) / max(source_width, 1e-6)) / 4.0, -1.0, 1.0
        ),
        _clip(
            math.log(max(target_height, 1e-6) / max(source_height, 1e-6)) / 4.0,
            -1.0,
            1.0,
        ),
        overlap,
        min(float(tree_distance) / 16.0, 1.0),
        float(sibling or parent or child or math.hypot(delta_x, delta_y) <= 0.25),
    )
    if len(values) != RELATION_FEATURE_DIM:
        raise AssertionError("unexpected relation feature dimension")
    return values


def _context_tree_distance(
    source_id: str,
    target_id: str,
    context: _RelationContext,
) -> int:
    if source_id == target_id:
        return 0
    source_positions = context.path_positions.get(source_id, {})
    target_path = context.ancestor_paths.get(target_id, (target_id,))
    distances = (
        source_positions[node_id] + target_index
        for target_index, node_id in enumerate(target_path)
        if node_id in source_positions
    )
    return min(distances, default=16)


def _normalized_bbox(bbox: BBox | None, graph: UIGraph) -> BBox | None:
    if bbox is None:
        return None
    width = float(
        graph.width
        or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[2] for node in graph.nodes)
    )
    height = float(
        graph.height
        or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[3] for node in graph.nodes)
    )
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        _clip(bbox[0] / width, 0.0, 1.0),
        _clip(bbox[1] / height, 0.0, 1.0),
        _clip(bbox[2] / width, 0.0, 1.0),
        _clip(bbox[3] / height, 0.0, 1.0),
    )


def _center(bbox: BBox | None) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _size(bbox: BBox | None) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0
    return max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])


def _iou(first: BBox | None, second: BBox | None) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_ancestor(ancestor_id: str, node_id: str, graph: UIGraph) -> bool:
    nodes = {node.node_id: node for node in graph.nodes}
    current = nodes.get(node_id)
    visited: set[str] = set()
    while (
        current is not None and current.parent_id and current.parent_id not in visited
    ):
        if current.parent_id == ancestor_id:
            return True
        visited.add(current.parent_id)
        current = nodes.get(current.parent_id)
    return False


def _tree_distance(source_id: str, target_id: str, graph: UIGraph) -> int:
    if source_id == target_id:
        return 0
    source_path = _ancestor_path(source_id, graph)
    target_path = _ancestor_path(target_id, graph)
    source_positions = {node_id: index for index, node_id in enumerate(source_path)}
    distances = [
        source_positions[node_id] + target_index
        for target_index, node_id in enumerate(target_path)
        if node_id in source_positions
    ]
    return min(distances) if distances else 16


def _ancestor_path(node_id: str, graph: UIGraph) -> list[str]:
    nodes = {node.node_id: node for node in graph.nodes}
    path = [node_id]
    current = nodes.get(node_id)
    visited = {node_id}
    while (
        current is not None and current.parent_id and current.parent_id not in visited
    ):
        path.append(current.parent_id)
        visited.add(current.parent_id)
        current = nodes.get(current.parent_id)
    return path


def _normalize_text(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    # Android accessibility labels often append the control role (for
    # example ``Notifications, Tab``).  The role is not part of the
    # cross-platform semantic label and must not turn an exact label into a
    # weak containment-only match.
    value = re.sub(r"\s*,\s*tab\s*$", "", value, flags=re.IGNORECASE)
    return " ".join(
        re.findall(r"[^\W_]+", value.lower(), flags=re.UNICODE)
    )


def _semantic_fields(node: UINode) -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            _normalize_text(node.text),
            _normalize_text(node.content_desc),
        )
        if value
    )


def _semantic_tokens(fields: tuple[str, ...]) -> set[str]:
    return {
        token
        for field in fields
        for token in re.findall(r"[^\W_]+", field, flags=re.UNICODE)
    }


def _semantic_trigrams(fields: tuple[str, ...]) -> set[str]:
    trigrams: set[str] = set()
    for token in _semantic_tokens(fields):
        if len(token) < 3:
            trigrams.add(token)
            continue
        padded = f"^{token}$"
        trigrams.update(
            padded[index : index + 3]
            for index in range(len(padded) - 2)
        )
    return trigrams


def _set_dice(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return 2.0 * len(first & second) / (len(first) + len(second))


def _is_toggle_node(node: UINode) -> bool:
    normalized_class = _normalize_text(node.class_name)
    return any(
        marker in normalized_class
        for marker in ("switch", "checkbox", "check box", "radio")
    )


def _token_bucket(piece: str, vocab_size: int) -> int:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (vocab_size - 1) + 1


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the learned matcher. Install omnitransfer[train]."
        ) from exc
    return torch


def _require_numpy() -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "NumPy is required for the learned matcher. Install omnitransfer[train]."
        ) from exc
    return np
