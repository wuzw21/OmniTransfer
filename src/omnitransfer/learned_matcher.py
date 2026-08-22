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
LEGACY_XML_NODE_FEATURE_DIM = 32
PARENT_RELATIVE_LAYOUT_FEATURE_DIM = 6
NODE_STATE_FEATURE_DIM = 12
XML_NODE_FEATURE_DIM = (
    LEGACY_XML_NODE_FEATURE_DIM + PARENT_RELATIVE_LAYOUT_FEATURE_DIM
)
STATE_AWARE_XML_NODE_FEATURE_DIM = XML_NODE_FEATURE_DIM + NODE_STATE_FEATURE_DIM
TEXT_DESCRIPTOR_DIM = 48
VISUAL_DESCRIPTOR_DIM = 48
MULTISCALE_VISUAL_DESCRIPTOR_DIM = VISUAL_DESCRIPTOR_DIM * 2
XML_DESCRIPTOR_DIM = 32
NODE_DESCRIPTOR_DIM = (
    TEXT_DESCRIPTOR_DIM + VISUAL_DESCRIPTOR_DIM + XML_DESCRIPTOR_DIM
)
LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID = "omnitransfer-direct-pair-evidence-v7"
GEOMETRIC_FEATURE_SCHEMA_ID = "omnitransfer-parent-relative-layout-v8"
STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID = "omnitransfer-state-aware-v9"
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
HASHED_NGRAM_TEXT_ENCODER = "hashed_ngram_v1"
DIRECT_TEXT_EVIDENCE_ENCODER = "direct_text_evidence"
LEGACY_NODE_ANCHOR_ENCODER = "router_anchor_v1"
SPARSE_NODE_ANCHOR_ENCODER = "sparse_node_anchor_v2"
DIRECT_CONCAT_NODE_ENCODER = "direct_concat_v1"
SPARSEMAX_MODALITY_ROUTER = "sparsemax_v1"
ST_HARD_TOP1_MODALITY_ROUTER = "st_hard_top1_v1"
NO_MODALITY_ROUTER = "none"
SOFTMAX_MODALITY_ROUTER = "softmax_v9"
LEGACY_UNIFORM_TYPED_RELATION_SELECTOR = "uniform_typed_v1"
LEGACY_SPARSE_USEFUL_NEIGHBOR_SELECTOR = "sparse_useful_v1"
ALL_NODE_CANDIDATE_POLICY = "all_nodes"
LEGACY_GLOBAL_POOL_VISUAL_ENCODER = "legacy_global_pool_v1"
SPATIAL_CNN_VISUAL_ENCODER = "spatial_cnn_v2"
DETERMINISTIC_ICON_VISUAL_ENCODER = "deterministic_icon_v1"
MULTISCALE_HASH_VISUAL_ENCODER = "multiscale_hash_v2"
MULTISCALE_RESIDUAL_VISUAL_ENCODER = "multiscale_residual_v3"
SEMANTIC_EXACT_DECODER_TOP_K = 0
SEMANTIC_EXACT_DECODER_MAX_MARGIN = 1.0
REPLACEMENT_SCORE_UPDATE = "replacement_v1"
UNARY_RESIDUAL_SCORE_UPDATE = "unary_plus_local_v2"
FIXED_NEIGHBOR_SELECTION = "fixed_priority_v1"
LEARNED_NEIGHBOR_SELECTION = "learned_relation_v2"


def is_multiscale_visual_encoder(visual_encoder: str) -> bool:
    return visual_encoder in {
        MULTISCALE_HASH_VISUAL_ENCODER,
        MULTISCALE_RESIDUAL_VISUAL_ENCODER,
    }


@dataclass(frozen=True)
class MatcherConfig:
    """Configuration for the compact learned matcher."""

    vocab_size: int = 8192
    max_tokens: int = 48
    token_dim: int = 48
    hidden_dim: int = 64
    relation_hidden_dim: int = 24
    association_dim: int = 64
    association_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.05
    router_policy: str = SOFTMAX_MODALITY_ROUTER
    router_temperature: float = 0.25
    router_softmax_leak: float = 0.0
    visual_patch_size: int = 32
    visual_canvas_size: int = 384
    visual_encoder: str = DETERMINISTIC_ICON_VISUAL_ENCODER
    visual_context_scale: float = 3.0
    source_context_nodes: int = 48
    target_context_nodes: int = 64
    architecture: str = OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
    assignment_head: str = "partial_assignment"
    text_encoder: str = LEARNED_TOKEN_LOOKUP_ENCODER
    node_anchor_encoder: str = LEGACY_NODE_ANCHOR_ENCODER
    candidate_policy: str = ALL_NODE_CANDIDATE_POLICY
    direct_pair_evidence: bool = False
    local_semantic_context: bool = False
    pairwise_local_correspondence: bool = True
    correspondence_pair_state: bool = True
    learned_multi_neighbor_context: bool = True
    state_embedding_dim: int = 1024
    semantic_exact_decoder_bonus: float = 0.0
    semantic_exact_decoder_top_k: int = SEMANTIC_EXACT_DECODER_TOP_K
    semantic_exact_decoder_max_margin: float = SEMANTIC_EXACT_DECODER_MAX_MARGIN
    feature_schema_id: str = STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID
    score_update: str = UNARY_RESIDUAL_SCORE_UPDATE
    neighbor_selection: str = LEARNED_NEIGHBOR_SELECTION
    local_neighbor_limit: int = 5


def visual_descriptor_dim(visual_encoder: str) -> int:
    return (
        MULTISCALE_VISUAL_DESCRIPTOR_DIM
        if is_multiscale_visual_encoder(visual_encoder)
        else VISUAL_DESCRIPTOR_DIM
    )


def xml_node_feature_dim(feature_schema_id: str) -> int:
    if feature_schema_id == LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID:
        return LEGACY_XML_NODE_FEATURE_DIM
    if feature_schema_id == GEOMETRIC_FEATURE_SCHEMA_ID:
        return XML_NODE_FEATURE_DIM
    if feature_schema_id == STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID:
        return STATE_AWARE_XML_NODE_FEATURE_DIM
    raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")


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


@dataclass(frozen=True)
class EncodedPage:
    """One observation encoded once for page matching and repeated node maps."""

    graph: UIGraph
    output: dict[str, Any]


@dataclass(frozen=True)
class PageMatch:
    """One cached all-node score matrix between two encoded observations."""

    source: EncodedPage
    target: EncodedPage
    output: dict[str, Any]


def encode_graph(
    graph: UIGraph,
    *,
    config: MatcherConfig | None = None,
    feature_schema_id: str | None = None,
) -> EncodedGraph:
    """Encode UI attributes and pairwise relations without fixed match weights."""

    cfg = config or MatcherConfig()
    feature_schema_id = feature_schema_id or cfg.feature_schema_id
    if feature_schema_id not in {
        LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID,
        GEOMETRIC_FEATURE_SCHEMA_ID,
        STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID,
    }:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    relation_context = (
        _relation_context(graph) if cfg.local_semantic_context else None
    )
    return EncodedGraph(
        graph_id=graph.graph_id,
        node_ids=tuple(node.node_id for node in graph.nodes),
        origin_ids=tuple(node.origin_id for node in graph.nodes),
        token_ids=tuple(
            _multimodal_text_token_ids(
                node,
                graph=graph,
                relation_context=relation_context,
                config=cfg,
            )
            for node in graph.nodes
        ),
        numeric_features=tuple(
            _multimodal_xml_features(
                node,
                graph,
                feature_schema_id=feature_schema_id,
            )
            for node in graph.nodes
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

    if feature_schema_id not in {
        LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID,
        GEOMETRIC_FEATURE_SCHEMA_ID,
    }:
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
        LEGACY_XML_NODE_FEATURE_DIM - 4,
    )
    target_class_hash = _hashed_attribute_features(
        target.class_name,
        LEGACY_XML_NODE_FEATURE_DIM - 4,
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


def typed_relation_bases(
    relations: Any,
    numeric_features: Any,
    *,
    feature_schema_id: str = GEOMETRIC_FEATURE_SCHEMA_ID,
) -> Any:
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

    xml_node_feature_dim(feature_schema_id)
    identity = relations[..., 0].clamp(0.0, 1.0)
    non_identity = 1.0 - identity
    local = relations[..., 17].clamp(0.0, 1.0) * non_identity
    sibling = relations[..., 3].clamp(0.0, 1.0)
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
            sibling,
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


def _discard_legacy_relation_neighbor_selector(
    config_payload: dict[str, Any],
) -> None:
    selector = config_payload.pop(
        "relation_neighbor_selector",
        LEGACY_UNIFORM_TYPED_RELATION_SELECTOR,
    )
    if selector == LEGACY_SPARSE_USEFUL_NEIGHBOR_SELECTOR:
        raise ValueError(
            "sparse_useful_v1 checkpoints are rejected because the selector "
            "failed its causal ablation"
        )
    if selector != LEGACY_UNIFORM_TYPED_RELATION_SELECTOR:
        raise ValueError(f"unsupported legacy relation neighbor selector: {selector}")


def page_inputs(
    graph: UIGraph,
    *,
    config: MatcherConfig | None = None,
    device: str | Any = "cpu",
    feature_schema_id: str | None = None,
) -> tuple[Any, ...]:
    """Encode one observation into reusable matcher input tensors."""

    torch = _require_torch()
    cfg = config or MatcherConfig()
    feature_schema_id = feature_schema_id or cfg.feature_schema_id
    encoded = encode_graph(
        graph,
        config=cfg,
        feature_schema_id=feature_schema_id,
    )
    token_ids = torch.as_tensor(
        encoded.token_ids,
        dtype=torch.long,
        device=device,
    )
    numeric = torch.as_tensor(
        encoded.numeric_features,
        dtype=torch.float32,
        device=device,
    )
    relations = torch.as_tensor(
        encoded.relation_features,
        dtype=torch.float32,
        device=device,
    )
    visual, visual_mask = _visual_inputs(
        graph,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        visual_encoder=cfg.visual_encoder,
        context_scale=cfg.visual_context_scale,
        torch=torch,
        device=device,
    )
    return token_ids, numeric, relations, visual, visual_mask


def matcher_inputs(
    source: UIGraph,
    target: UIGraph,
    *,
    config: MatcherConfig | None = None,
    device: str | Any = "cpu",
    feature_schema_id: str | None = None,
) -> tuple[Any, ...]:
    """Convert two observations to the model's stable public input order."""

    source_inputs = page_inputs(
        source,
        config=config,
        device=device,
        feature_schema_id=feature_schema_id,
    )
    target_inputs = page_inputs(
        target,
        config=config,
        device=device,
        feature_schema_id=feature_schema_id,
    )
    return (
        *source_inputs[:3],
        *target_inputs[:3],
        *source_inputs[3:],
        *target_inputs[3:],
    )


class GeometricMatcher:
    """Inference adapter that abstains instead of replaying source coordinates."""

    def __init__(
        self,
        model: Any,
        *,
        config: MatcherConfig | None = None,
        device: str = "cpu",
        checkpoint_load_mode: str = "fresh_model",
    ) -> None:
        self.model = model
        self.config = config or MatcherConfig()
        self.device = device
        self.checkpoint_load_mode = checkpoint_load_mode
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
        config_payload.setdefault(
            "feature_schema_id",
            LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID,
        )
        config_payload.setdefault("correspondence_pair_state", False)
        config_payload.setdefault("learned_multi_neighbor_context", False)
        config_payload.setdefault("score_update", REPLACEMENT_SCORE_UPDATE)
        config_payload.setdefault("neighbor_selection", FIXED_NEIGHBOR_SELECTION)
        config_payload.setdefault("local_neighbor_limit", 8)
        _discard_legacy_relation_neighbor_selector(config_payload)
        if int(config_payload.get("state_embedding_dim") or 0) <= 0:
            config_payload["state_embedding_dim"] = MatcherConfig().state_embedding_dim
        config = MatcherConfig(**config_payload)
        if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
            raise ValueError("only geometric-v9 checkpoints are supported")
        model = build_geometric_v9_matcher(config)
        state_dict = dict(payload["state_dict"])
        if payload.get("schema_version") == "omnitransfer.page_local_matcher.v1":
            target_state = model.state_dict()
            deprecated = {
                "local_context_projection.weight",
                "local_fusion_norm.weight",
                "local_fusion_norm.bias",
                "state_to_pair.weight",
                "page_pair_context.0.weight",
                "page_pair_context.0.bias",
                "page_pair_context.1.weight",
                "page_pair_context.1.bias",
            }
            unexpected = set(state_dict).difference(target_state)
            if unexpected.difference(deprecated):
                raise ValueError(
                    "checkpoint contains unsupported parameters: "
                    + ", ".join(sorted(unexpected.difference(deprecated)))
                )
            compatible = {}
            for name, target in target_state.items():
                value = state_dict.get(name)
                if value is None:
                    continue
                if value.shape == target.shape:
                    compatible[name] = value
                    continue
                if name in {
                    "pair_scorer.0.weight",
                    "pair_scorer.0.bias",
                } and value.ndim == target.ndim == 1:
                    compatible[name] = value[: target.shape[0]]
                    continue
                if (
                    name == "pair_scorer.1.weight"
                    and value.ndim == target.ndim == 2
                    and value.shape[0] == target.shape[0]
                    and value.shape[1] > target.shape[1]
                ):
                    compatible[name] = value[:, : target.shape[1]]
            missing = set(target_state).difference(compatible)
            if missing:
                raise ValueError(
                    "checkpoint is missing required parameters: "
                    + ", ".join(sorted(missing))
                )
            model.load_state_dict(compatible)
            checkpoint_load_mode = (
                "page_local_matcher_without_preaggregation"
                if unexpected
                else "exact_page_local_matcher"
            )
        elif "association_layer.transport_output.weight" in state_dict:
            raise ValueError(
                "iterative router-anchor checkpoints cannot initialize the "
                "page-local matcher; retraining is required"
            )
        else:
            migrated_state = model.state_dict()
            for name, value in state_dict.items():
                target = migrated_state.get(name)
                if (
                    target is not None
                    and target.shape == value.shape
                    and not name.startswith("association_layer.")
                ):
                    migrated_state[name] = value
            model.load_state_dict(migrated_state)
            checkpoint_load_mode = "compatible_encoder_initialization"
        return cls(
            model,
            config=config,
            device=device,
            checkpoint_load_mode=checkpoint_load_mode,
        )

    def encode_page(self, observation: UIGraph) -> EncodedPage:
        """Encode screenshot, XML, local relations, and state exactly once."""

        torch = _require_torch()
        inputs = page_inputs(
            observation,
            config=self.config,
            device=self.device,
            feature_schema_id=self.config.feature_schema_id,
        )
        with torch.inference_mode():
            output = self.model.encode_page(*inputs)
        return EncodedPage(graph=observation, output=output)

    def match_page(self, source: EncodedPage, target: EncodedPage) -> PageMatch:
        """Compute the complete correspondence matrix once for two pages."""

        torch = _require_torch()
        with torch.inference_mode():
            output = self.model.match_pages(source.output, target.output)
        return PageMatch(source=source, target=target, output=output)

    def map_node(
        self,
        page_match: PageMatch,
        *,
        source_node_id: str,
        candidate_node_ids: Iterable[str] | None = None,
        min_probability: float = 0.0,
        min_margin: float = 0.0,
    ) -> LearnedMatch:
        """Read one source row from a cached page match without recomputation."""

        torch = _require_torch()
        source_index = next(
            (
                index
                for index, node in enumerate(page_match.source.graph.nodes)
                if node.node_id == source_node_id
            ),
            None,
        )
        if source_index is None:
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        allowed = set(
            candidate_node_ids
            or (node.node_id for node in page_match.target.graph.nodes)
        )
        candidate_indices = [
            index
            for index, node in enumerate(page_match.target.graph.nodes)
            if node.node_id in allowed
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        with torch.inference_mode():
            selected_logits = page_match.output["logits_ab"][source_index][
                candidate_indices
            ]
            rank_probabilities = torch.softmax(selected_logits, dim=0)
            match_probability = float(rank_probabilities.max())
        ranked = sorted(
            (
                (
                    page_match.target.graph.nodes[index].node_id,
                    float(rank_probabilities[position]),
                )
                for position, index in enumerate(candidate_indices)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        second_probability = max(
            (score for _, score in ranked[1:]),
            default=0.0,
        )
        margin = best_probability - second_probability
        scores = tuple(ranked)
        if match_probability < min_probability or margin <= min_margin:
            return LearnedMatch(
                None,
                match_probability,
                margin,
                "learned_low_confidence",
                scores,
            )
        target_node = next(
            node for node in page_match.target.graph.nodes if node.node_id == best_id
        )
        return LearnedMatch(
            target_node,
            match_probability,
            margin,
            "learned_match",
            scores,
        )

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
            max_nodes=min(
                len(target.nodes),
                max(
                    self.config.target_context_nodes,
                    len(allowed) + 3 * self.config.local_neighbor_limit,
                ),
            ),
        )
        candidate_indices = [
            index
            for index, node in enumerate(target_context.nodes)
            if node.node_id in allowed
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        source_page = self.encode_page(source_context)
        target_page = self.encode_page(target_context)
        page_match = self.match_page(source_page, target_page)
        return self.map_node(
            page_match,
            source_node_id=source_node_id,
            candidate_node_ids=(
                target_context.nodes[index].node_id for index in candidate_indices
            ),
            min_probability=min_probability,
            min_margin=min_margin,
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
            "schema_version": "omnitransfer.page_local_matcher.v1",
            "matcher_config": asdict(config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )


NODE_ENCODER_PARAMETER_PREFIXES = (
    "token_embedding.",
    "text_projection.",
    "xml_projection.",
    "visual_encoder.",
    "present_text",
    "missing_text",
    "missing_visual",
    "text_to_hidden.",
    "visual_to_hidden.",
    "xml_to_hidden.",
    "modality_type",
    "modality_score.",
    "input_norm.",
    "input_feed_forward.",
    "input_output_norm.",
    "logit_scale",
)
V9_BACKBONE_PARAMETER_PREFIXES = (
    *NODE_ENCODER_PARAMETER_PREFIXES,
    "relation_compatibility",
    "association_layers.",
    "matchability_head.",
    "page_attention.",
)


def initialize_node_encoder_from_checkpoint(
    model: Any,
    checkpoint: str | Path,
) -> tuple[str, ...]:
    """Load only compatible node/unary parameters into a fresh matcher."""

    return _initialize_parameters_from_checkpoint(
        model,
        checkpoint,
        prefixes=NODE_ENCODER_PARAMETER_PREFIXES,
        description="node encoder",
    )


def initialize_v9_backbone_from_checkpoint(
    model: Any,
    checkpoint: str | Path,
) -> tuple[str, ...]:
    """Restore the trained v9 backbone while leaving new local fusion fresh."""

    return _initialize_parameters_from_checkpoint(
        model,
        checkpoint,
        prefixes=V9_BACKBONE_PARAMETER_PREFIXES,
        description="v9 backbone",
    )


def _initialize_parameters_from_checkpoint(
    model: Any,
    checkpoint: str | Path,
    *,
    prefixes: tuple[str, ...],
    description: str,
) -> tuple[str, ...]:
    """Load a named compatible subset, widening the current XML input safely."""

    torch = _require_torch()
    payload = torch.load(Path(checkpoint), map_location="cpu")
    source_state = dict(payload["state_dict"])
    target_state = model.state_dict()
    transferred: list[str] = []
    for name, source_value in source_state.items():
        if not name.startswith(prefixes):
            continue
        target_value = target_state.get(name)
        if target_value is None:
            continue
        if (
            name == "xml_projection.1.weight"
            and target_value.ndim == source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1] > source_value.shape[1]
        ):
            widened = target_value.detach().clone().zero_()
            widened[:, : source_value.shape[1]] = source_value.to(
                target_value.device
            )
            target_state[name] = widened
            transferred.append(name)
            continue
        if target_value.shape != source_value.shape:
            continue
        target_state[name] = source_value.to(target_value.device)
        transferred.append(name)
    if not transferred:
        raise ValueError(f"checkpoint has no compatible {description} parameters")
    model.load_state_dict(target_state)
    return tuple(sorted(transferred))


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
    """Migrate compatible geometric-v9 weights into a rebuilt encoder."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred: list[str] = []
    for target_name, target_value in target_state.items():
        if target_name.startswith("visual_encoder.features."):
            source_name = target_name.replace(
                "visual_encoder.features.",
                "visual_encoder.",
                1,
            )
            source_value = source_state.get(source_name)
            if source_value is not None and source_value.shape == target_value.shape:
                target_state[target_name] = source_value.detach().clone()
                transferred.append(target_name)
            continue
        if target_name.startswith("visual_encoder.dense_readout."):
            source_name = target_name.replace(
                "visual_encoder.dense_readout.",
                "visual_encoder.9.",
                1,
            )
            source_value = source_state.get(source_name)
            if source_value is not None and source_value.shape == target_value.shape:
                target_state[target_name] = source_value.detach().clone()
                transferred.append(target_name)
            continue
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
        if (
            source_value is not None
            and target_name == "visual_to_hidden.weight"
            and target_value.ndim == source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and source_value.shape[1] == target_value.shape[1] * 2
        ):
            target_state[target_name] = source_value[
                :, : target_value.shape[1]
            ].detach().clone()
            transferred.append(target_name)
            continue
        if target_name == "visual_context_to_hidden.weight":
            widened_source = source_state.get("visual_to_hidden.weight")
            residual = target_value.detach().clone().zero_()
            if (
                widened_source is not None
                and widened_source.ndim == residual.ndim == 2
                and widened_source.shape[0] == residual.shape[0]
                and widened_source.shape[1] == residual.shape[1] * 2
            ):
                residual.copy_(widened_source[:, residual.shape[1] :])
            target_state[target_name] = residual
            transferred.append(target_name)
            continue
        if source_value is None or source_value.shape != target_value.shape:
            if target_name.startswith("association_layer.") or (
                target_name.startswith("direct_pair_projection.")
            ) or (
                ".pairwise_relation_projection." in target_name
            ) or (
                ".pairwise_gate." in target_name
            ) or (
                ".pair_state_" in target_name
            ) or (
                ".source_pair_feedback." in target_name
            ) or (
                ".target_pair_feedback." in target_name
            ) or (
                ".neighbor_" in target_name
            ) or (
                target_name.startswith("state_embedding_projection.")
            ) or (
                target_name.startswith("stable_state_")
            ) or (
                target_name.startswith("active_state_")
            ) or (
                target_name.startswith("active_state_decoder.")
            ) or (
                target_name.startswith("stable_pair_state_bridge.")
            ) or (
                target_name.startswith("active_pair_state_bridge.")
            ) or (
                target_name == "layout_projection.weight"
            ) or (
                target_name == "state_projection.weight"
            ) or (
                target_name.startswith("text_token_")
            ) or (
                target_name.startswith("xml_anchor_")
            ):
                continue
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
    channels = 6 if is_multiscale_visual_encoder(visual_encoder) else 3
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
    if is_multiscale_visual_encoder(visual_encoder):
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


def configure_visual_image_cache(maxsize: int) -> int:
    """Resize the process-local decoded screenshot cache without changing runtime."""

    if maxsize <= 0:
        raise ValueError("visual image cache size must be positive")
    global _load_rgb_array
    uncached_loader = getattr(_load_rgb_array, "__wrapped__", _load_rgb_array)
    _load_rgb_array.cache_clear()
    _load_rgb_array = lru_cache(maxsize=int(maxsize))(uncached_loader)
    return int(maxsize)




def _multimodal_text_token_ids(
    node: UINode,
    *,
    graph: UIGraph,
    relation_context: _RelationContext | None,
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
    if (
        config.local_semantic_context
        and not _semantic_fields(node)
    ):
        if relation_context is None:
            raise ValueError("local semantic context requires relation context")
        for relation, anchor in _local_semantic_anchors(
            node,
            graph,
            relation_context=relation_context,
        ):
            pieces.append(f"context:relation:{relation}")
            for value in (anchor.text, anchor.content_desc):
                normalized = _normalize_text(value)
                if not normalized:
                    continue
                for word in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE):
                    for field_name in ("text", "desc"):
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


def _local_semantic_anchors(
    node: UINode,
    graph: UIGraph,
    *,
    relation_context: _RelationContext | None = None,
) -> tuple[tuple[str, UINode], ...]:
    """Return the nearest labelled relatives without changing node identity."""

    context = relation_context or _relation_context(graph)
    node_index = context.node_indices[node.node_id]
    node_center = context.centers[node_index]
    candidates: list[tuple[int, float, int, str, UINode]] = []
    for index, candidate in enumerate(graph.nodes):
        if candidate.node_id == node.node_id or not _semantic_fields(candidate):
            continue
        tree_distance = _context_tree_distance(
            node.node_id,
            candidate.node_id,
            context,
        )
        candidate_center = context.centers[index]
        spatial_distance = math.hypot(
            candidate_center[0] - node_center[0],
            candidate_center[1] - node_center[1],
        )
        if candidate.parent_id == node.node_id:
            relation = "child"
            priority = 0
        elif node.parent_id == candidate.node_id:
            relation = "parent"
            priority = 0
        elif node.parent_id and node.parent_id == candidate.parent_id:
            relation = "sibling"
            priority = 1
        elif tree_distance <= 3:
            relation = "local_branch"
            priority = 2
        elif spatial_distance <= 0.15:
            relation = "spatial_neighbor"
            priority = 3
        else:
            continue
        candidates.append(
            (priority, spatial_distance, index, relation, candidate)
        )
    if not candidates:
        return ()
    candidates.sort(key=lambda value: value[:3])
    _, _, _, relation, candidate = candidates[0]
    return ((relation, candidate),)










def _multimodal_xml_features(
    node: UINode,
    graph: UIGraph,
    *,
    feature_schema_id: str,
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
    common = (
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
    )
    class_features = _hashed_attribute_features(
        node.class_name,
        LEGACY_XML_NODE_FEATURE_DIM - len(common),
    )
    legacy_values = (*common, *class_features)
    if feature_schema_id == LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID:
        values = legacy_values
    elif feature_schema_id == GEOMETRIC_FEATURE_SCHEMA_ID:
        values = (
            *legacy_values,
            *_parent_relative_layout_features(node, graph),
        )
    elif feature_schema_id == STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID:
        values = (
            *legacy_values,
            *_parent_relative_layout_features(node, graph),
            *_node_state_features(node),
        )
    else:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    if len(values) != xml_node_feature_dim(feature_schema_id):
        raise AssertionError("unexpected XML node feature dimension")
    return values


def _node_state_features(node: UINode) -> tuple[float, ...]:
    values = []
    for name in (
        "visible",
        "checked",
        "selected",
        "focused",
        "expanded",
        "password",
    ):
        values.extend(
            (
                float(bool(node.metadata.get(name, False))),
                float(bool(node.metadata.get(f"{name}_present", False))),
            )
        )
    return tuple(values)


def _parent_relative_layout_features(
    node: UINode,
    graph: UIGraph,
) -> tuple[float, ...]:
    nodes_by_id = {candidate.node_id: candidate for candidate in graph.nodes}
    parent = nodes_by_id.get(node.parent_id or "")
    siblings = parent.child_ids if parent is not None else (node.node_id,)
    try:
        sibling_index = siblings.index(node.node_id)
    except ValueError:
        sibling_index = 0
    sibling_count = max(len(siblings), 1)
    sibling_position = (
        float(sibling_index) / float(sibling_count - 1)
        if sibling_count > 1
        else 0.5
    )
    sibling_count_feature = min(math.log2(float(sibling_count) + 1.0) / 6.0, 1.0)

    node_bbox = _normalized_bbox(node.bbox, graph)
    parent_bbox = _normalized_bbox(parent.bbox, graph) if parent is not None else None
    if node_bbox is None or parent_bbox is None:
        parent_center_x = parent_center_y = 0.0
        parent_width = parent_height = 0.0
    else:
        parent_box_width = max(parent_bbox[2] - parent_bbox[0], 1e-6)
        parent_box_height = max(parent_bbox[3] - parent_bbox[1], 1e-6)
        node_center_x = (node_bbox[0] + node_bbox[2]) / 2.0
        node_center_y = (node_bbox[1] + node_bbox[3]) / 2.0
        parent_center_x = _clip(
            2.0 * (node_center_x - parent_bbox[0]) / parent_box_width - 1.0,
            -1.0,
            1.0,
        )
        parent_center_y = _clip(
            2.0 * (node_center_y - parent_bbox[1]) / parent_box_height - 1.0,
            -1.0,
            1.0,
        )
        parent_width = _clip(
            (node_bbox[2] - node_bbox[0]) / parent_box_width,
            0.0,
            1.0,
        )
        parent_height = _clip(
            (node_bbox[3] - node_bbox[1]) / parent_box_height,
            0.0,
            1.0,
        )
    return (
        2.0 * sibling_position - 1.0,
        sibling_count_feature,
        parent_center_x,
        parent_center_y,
        parent_width,
        parent_height,
    )


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
    source_bbox_present = np.asarray(
        [bbox is not None for bbox in source_context.bboxes],
        dtype=bool,
    )
    target_bbox_present = np.asarray(
        [bbox is not None for bbox in target_context.bboxes],
        dtype=bool,
    )
    bbox_pair_present = source_bbox_present[:, None] & target_bbox_present[None, :]
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
    ) & bbox_pair_present
    same_column = (
        np.abs(delta_x)
        <= np.maximum(
            np.maximum(source_width, target_width),
            0.02,
        )
        * 0.5
    ) & bbox_pair_present
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
    values[..., 9] = np.where(
        bbox_pair_present, np.clip(delta_x, -1.0, 1.0), 0.0
    )
    values[..., 10] = np.where(
        bbox_pair_present, np.clip(delta_y, -1.0, 1.0), 0.0
    )
    values[..., 11] = np.where(
        bbox_pair_present, np.minimum(np.abs(delta_x), 1.0), 0.0
    )
    values[..., 12] = np.where(
        bbox_pair_present, np.minimum(np.abs(delta_y), 1.0), 0.0
    )
    values[..., 13] = np.where(
        bbox_pair_present,
        np.clip(
            np.log(
                np.maximum(target_width, 1e-6)
                / np.maximum(source_width, 1e-6)
            )
            / 4.0,
            -1.0,
            1.0,
        ),
        0.0,
    )
    values[..., 14] = np.where(
        bbox_pair_present,
        np.clip(
            np.log(
                np.maximum(target_height, 1e-6)
                / np.maximum(source_height, 1e-6)
            )
            / 4.0,
            -1.0,
            1.0,
        ),
        0.0,
    )
    values[..., 15] = np.where(bbox_pair_present, overlap, 0.0)
    values[..., 16] = np.minimum(tree_distance / 16.0, 1.0)
    values[..., 17] = (
        sibling
        | parent
        | child
        | ancestor
        | descendant
        | (bbox_pair_present & (np.hypot(delta_x, delta_y) <= 0.25))
    )
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


def hashed_ngram_descriptor_torch(token_ids: Any, *, torch: Any) -> Any:
    """Create a fixed CountSketch-style descriptor from hashed text pieces."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must be a two-dimensional tensor")
    token_ids = token_ids.to(dtype=torch.long)
    valid = token_ids.gt(0)
    descriptor = torch.zeros(
        (token_ids.shape[0], TEXT_DESCRIPTOR_DIM),
        dtype=torch.float32,
        device=token_ids.device,
    )
    bucket_multipliers = (1315423911, 2654435761, 374761393, 668265263)
    sign_multipliers = (31, 131, 911, 3571)
    offsets = (17, 31, 47, 73)
    for bucket_multiplier, sign_multiplier, offset in zip(
        bucket_multipliers,
        sign_multipliers,
        offsets,
        strict=True,
    ):
        buckets = torch.remainder(
            token_ids * bucket_multiplier + offset,
            TEXT_DESCRIPTOR_DIM,
        )
        signs = torch.where(
            torch.remainder(token_ids * sign_multiplier + offset, 2).eq(0),
            torch.ones_like(token_ids, dtype=torch.float32),
            -torch.ones_like(token_ids, dtype=torch.float32),
        )
        descriptor.scatter_add_(1, buckets, signs * valid.to(torch.float32))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float32)
    return descriptor / count


def hashed_ngram_descriptor_numpy(token_ids: Any) -> Any:
    """NumPy counterpart of the fixed signed text descriptor."""

    np = _require_numpy()
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be a two-dimensional array")
    valid = token_ids > 0
    descriptor = np.zeros(
        (token_ids.shape[0], TEXT_DESCRIPTOR_DIM),
        dtype=np.float32,
    )
    bucket_multipliers = (1315423911, 2654435761, 374761393, 668265263)
    sign_multipliers = (31, 131, 911, 3571)
    offsets = (17, 31, 47, 73)
    rows = np.broadcast_to(
        np.arange(token_ids.shape[0])[:, None],
        token_ids.shape,
    )
    for bucket_multiplier, sign_multiplier, offset in zip(
        bucket_multipliers,
        sign_multipliers,
        offsets,
        strict=True,
    ):
        buckets = (
            token_ids * bucket_multiplier + offset
        ) % TEXT_DESCRIPTOR_DIM
        signs = np.where(
            (token_ids * sign_multiplier + offset) % 2 == 0,
            1.0,
            -1.0,
        )
        np.add.at(
            descriptor,
            (rows, buckets),
            (signs * valid).astype(np.float32),
        )
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1).astype(np.float32)
    return descriptor / count


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
