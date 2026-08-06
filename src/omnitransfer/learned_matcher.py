"""Lightweight relation-aware cross-attention for UI graph matching."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from omnitransfer.ui_graph import BBox, UIGraph, UINode, local_context_graph

# Keep the tensor width checkpoint-compatible with the deployed matcher.  Only
# the first four action-state channels carry information; the remaining
# channels are reserved zeros and contain no geometry.
NUMERIC_FEATURE_DIM = 18
RELATION_FEATURE_DIM = 18
XML_NODE_FEATURE_DIM = 32
TEXT_DESCRIPTOR_DIM = 48
VISUAL_DESCRIPTOR_DIM = 48
XML_DESCRIPTOR_DIM = 32
NODE_DESCRIPTOR_DIM = (
    TEXT_DESCRIPTOR_DIM + VISUAL_DESCRIPTOR_DIM + XML_DESCRIPTOR_DIM
)
RCAM_FEATURE_SCHEMA_ID = "rcam-node-context-v1"
PEMM_V3_FEATURE_SCHEMA_ID = "pemm-v3-node-context-v1"
OMNITRANSFER_V4_FEATURE_SCHEMA_ID = "omnitransfer-node-descriptor-v4"
OMNITRANSFER_V7_FEATURE_SCHEMA_ID = "omnitransfer-direct-pair-evidence-v7"
PEMM_V3_FEATURE_SCHEMA_SPEC = (
    "tokens=text,content_desc,resource_id,class,action_type,parent_text,"
    "screen_region,sibling_texts,nearby_texts;"
    "numeric=action_state,normalized_bbox,depth,child_count,parent_presence;"
    "relations=within_graph_tree_layout,cross_graph_normalized_geometry"
)
PEMM_V3_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    PEMM_V3_FEATURE_SCHEMA_SPEC.encode("utf-8")
).hexdigest()
LEGACY_ATTENTION_ARCHITECTURE = "relation_bias_v1"
LEARNED_GRAPH_ATTENTION_ARCHITECTURE = "learned_graph_cross_attention_v2"
LIGHTGLUE_GRAPH_MATCHING_ARCHITECTURE = "lightglue_graph_matching_v3"
OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE = "omnitransfer_graph_matching_v4"
OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE = (
    "omnitransfer_structured_matching_v5"
)
OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE = (
    "omnitransfer_association_matching_v6"
)
OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE = (
    "omnitransfer_evidence_matching_v7"
)
OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE = (
    "omnitransfer_anchor_matching_v8"
)
OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE = (
    "omnitransfer_transformer_adapter_v10"
)
OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE = (
    "omnitransfer_v8_graph_transformer_v11"
)
OMNITRANSFER_UNIFIED_GRAPH_TRANSFORMER_ARCHITECTURE = (
    "omnitransfer_unified_graph_transformer_v12"
)
OMNITRANSFER_EVIDENCE_FUSION_ARCHITECTURE = (
    "omnitransfer_evidence_fusion_v13"
)
OMNITRANSFER_CALIBRATED_FUSION_ARCHITECTURE = (
    "omnitransfer_calibrated_fusion_v14"
)
OMNITRANSFER_RELATIONAL_CONSENSUS_ARCHITECTURE = (
    "omnitransfer_relational_consensus_v15"
)
OMNITRANSFER_NORMALIZED_CONSENSUS_ARCHITECTURE = (
    "omnitransfer_normalized_consensus_v16"
)
OMNITRANSFER_TRANSFORMER_CONSENSUS_ARCHITECTURE = (
    "omnitransfer_transformer_consensus_v17"
)
OMNITRANSFER_NONLINEAR_CONSENSUS_ARCHITECTURE = (
    "omnitransfer_nonlinear_consensus_v18"
)
OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE = (
    "omnitransfer_context_matching_v9"
)
OMNITRANSFER_LOCAL_ALIGNMENT_ARCHITECTURE = (
    "omnitransfer_local_alignment_v9"
)
OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE = (
    "omnitransfer_geometric_alignment_v9"
)
OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE = (
    "omnitransfer_geometry_refinement_v9"
)
OMNITRANSFER_MATCHING_ARCHITECTURE = "omnitransfer_matching"
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
DIRECT_SEMANTIC_EVIDENCE_NAMES = DIRECT_PAIR_EVIDENCE_NAMES[:4]
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
    "control_to_context",
    "context_to_control",
)
LEARNED_EDGE_FEATURE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
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
SPATIAL_XML_ALIGNMENT_TOP_K = 5
SPATIAL_XML_ALIGNMENT_MAX_LOG_GAP = 5.0
SPATIAL_XML_ALIGNMENT_DUPLICATE_MAX_LOG_GAP = 2.0
LEARNED_TOKEN_LOOKUP_ENCODER = "learned_token_lookup"
DIRECT_TEXT_EVIDENCE_ENCODER = "direct_text_evidence"
ACTIONABLE_CANDIDATE_POLICY = "actionable"
ALL_NODE_CANDIDATE_POLICY = "all_nodes"


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
    num_layers: int = 4
    dropout: float = 0.05
    visual_patch_size: int = 32
    visual_canvas_size: int = 384
    source_context_nodes: int = 48
    target_context_nodes: int = 64
    architecture: str = OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE
    assignment_head: str = "partial_assignment"
    text_encoder: str = LEARNED_TOKEN_LOOKUP_ENCODER
    candidate_policy: str = ACTIONABLE_CANDIDATE_POLICY


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
    feature_schema_id = feature_schema_id or _feature_schema_for_config(cfg)
    if feature_schema_id == RCAM_FEATURE_SCHEMA_ID:
        token_encoder = _node_token_ids
        numeric_encoder = _node_numeric_features
    elif feature_schema_id in {
        OMNITRANSFER_V4_FEATURE_SCHEMA_ID,
        OMNITRANSFER_V7_FEATURE_SCHEMA_ID,
    }:
        token_encoder = _multimodal_text_token_ids
        numeric_encoder = _multimodal_xml_features
    elif feature_schema_id == PEMM_V3_FEATURE_SCHEMA_ID:
        token_encoder = _pemm_v3_node_token_ids
        numeric_encoder = _pemm_v3_node_numeric_features
    else:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    return EncodedGraph(
        graph_id=graph.graph_id,
        node_ids=tuple(node.node_id for node in graph.nodes),
        origin_ids=tuple(node.origin_id for node in graph.nodes),
        token_ids=tuple(token_encoder(node, config=cfg) for node in graph.nodes),
        numeric_features=tuple(
            numeric_encoder(node, graph) for node in graph.nodes
        ),
        relation_features=_relation_features(graph),
    )


def cross_relation_features(
    source: UIGraph,
    target: UIGraph,
    *,
    feature_schema_id: str = RCAM_FEATURE_SCHEMA_ID,
) -> Any:
    """Return cross-page relations under an explicit feature contract."""

    if feature_schema_id not in {
        RCAM_FEATURE_SCHEMA_ID,
        PEMM_V3_FEATURE_SCHEMA_ID,
        OMNITRANSFER_V4_FEATURE_SCHEMA_ID,
        OMNITRANSFER_V7_FEATURE_SCHEMA_ID,
    }:
        raise ValueError(f"unsupported matcher feature schema: {feature_schema_id}")
    if feature_schema_id == OMNITRANSFER_V7_FEATURE_SCHEMA_ID:
        return tuple(
            tuple(
                direct_pair_evidence_features(source_node, target_node)
                for target_node in target.nodes
            )
            for source_node in source.nodes
        )
    source_context = _relation_context(source)
    target_context = _relation_context(target)
    same_graph = source is target or source.graph_id == target.graph_id
    return _relation_matrix(
        source_context,
        target_context,
        same_graph=same_graph,
        include_cross_graph_geometry=(
            feature_schema_id == PEMM_V3_FEATURE_SCHEMA_ID
        ),
    )


def _feature_schema_for_config(config: MatcherConfig) -> str:
    if config.architecture in {
        OMNITRANSFER_MATCHING_ARCHITECTURE,
        OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
        OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
        OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
    }:
        return OMNITRANSFER_V4_FEATURE_SCHEMA_ID
    if config.architecture in {
        OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
        OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
        OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
        OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
        OMNITRANSFER_UNIFIED_GRAPH_TRANSFORMER_ARCHITECTURE,
        OMNITRANSFER_EVIDENCE_FUSION_ARCHITECTURE,
        OMNITRANSFER_CALIBRATED_FUSION_ARCHITECTURE,
        OMNITRANSFER_RELATIONAL_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_NORMALIZED_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_TRANSFORMER_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_NONLINEAR_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
        OMNITRANSFER_LOCAL_ALIGNMENT_ARCHITECTURE,
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
        OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE,
    }:
        return OMNITRANSFER_V7_FEATURE_SCHEMA_ID
    return RCAM_FEATURE_SCHEMA_ID


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


def semantic_anchor_relation_vote(
    direct_pair_evidence: Any,
    source_bases: Any,
    target_bases: Any,
    relation_compatibility: Any,
    *,
    minimum_semantic_score: float = 1.0,
    minimum_margin: float = 0.5,
) -> Any:
    """Vote for node pairs using bidirectionally unique semantic anchors."""

    torch = _require_torch()
    if direct_pair_evidence.ndim != 3:
        raise ValueError("direct pair evidence must be three-dimensional")
    source_count, target_count, feature_count = direct_pair_evidence.shape
    relation_count = len(TYPED_RELATION_NAMES)
    if feature_count < len(DIRECT_SEMANTIC_EVIDENCE_NAMES):
        raise ValueError("direct pair evidence is missing semantic features")
    if source_bases.shape != (relation_count, source_count, source_count):
        raise ValueError("source relation bases do not align with pair evidence")
    if target_bases.shape != (relation_count, target_count, target_count):
        raise ValueError("target relation bases do not align with pair evidence")
    if relation_compatibility.shape != (relation_count, relation_count):
        raise ValueError("relation compatibility has an unexpected shape")
    if source_count == 0 or target_count == 0:
        return direct_pair_evidence.new_zeros((source_count, target_count))

    semantic_scores = direct_pair_evidence[
        ..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)
    ].sum(dim=-1)
    source_best_scores, source_best_targets = semantic_scores.max(dim=1)
    target_best_scores, target_best_sources = semantic_scores.max(dim=0)
    source_margins = source_best_scores
    target_margins = target_best_scores
    if target_count > 1:
        source_top_two = torch.topk(semantic_scores, 2, dim=1).values
        source_margins = source_top_two[:, 0] - source_top_two[:, 1]
    if source_count > 1:
        target_top_two = torch.topk(semantic_scores, 2, dim=0).values
        target_margins = target_top_two[0] - target_top_two[1]
    source_indices = torch.arange(source_count, device=semantic_scores.device)
    mutual = target_best_sources[source_best_targets].eq(source_indices)
    retained = (
        mutual
        & source_best_scores.ge(float(minimum_semantic_score))
        & source_margins.ge(float(minimum_margin))
        & target_margins[source_best_targets].ge(float(minimum_margin))
    )
    source_anchors = source_indices[retained]
    if source_anchors.numel() == 0:
        return semantic_scores.new_zeros((source_count, target_count))
    target_anchors = source_best_targets[source_anchors]

    source_binary = source_bases.gt(0).to(source_bases.dtype)
    target_binary = target_bases.gt(0).to(target_bases.dtype)
    source_outgoing = source_binary[:, :, source_anchors]
    target_outgoing = target_binary[:, :, target_anchors]
    source_incoming = source_binary[:, source_anchors, :].permute(0, 2, 1)
    target_incoming = target_binary[:, target_anchors, :].permute(0, 2, 1)
    vote = torch.einsum(
        "ria,rq,qja->ij",
        source_outgoing,
        relation_compatibility,
        target_outgoing,
    )
    vote = vote + torch.einsum(
        "ria,rq,qja->ij",
        source_incoming,
        relation_compatibility,
        target_incoming,
    )
    support = torch.einsum(
        "ria,qja->ija",
        source_outgoing,
        target_outgoing,
    ).gt(0).sum(dim=-1)
    support = support.to(vote.dtype) + torch.einsum(
        "ria,qja->ija",
        source_incoming,
        target_incoming,
    ).gt(0).sum(dim=-1).to(vote.dtype)
    return vote / support.clamp_min(1.0)


def relational_consensus_features(
    log_assignment: Any,
    source_bases: Any,
    target_bases: Any,
) -> Any:
    """Return typed relation agreement around the frozen v8 assignment.

    The feature is permutation equivariant and uses only within-page typed
    relations.  Each candidate pair receives one channel for every source and
    target relation-type combination.  The frozen v8 assignment supplies soft
    anchors, while identity edges are removed so a candidate cannot vote for
    itself.
    """

    torch = _require_torch()
    relation_count = len(TYPED_RELATION_NAMES)
    if log_assignment.ndim != 2:
        raise ValueError("log assignment must be two-dimensional")
    source_count, target_count = log_assignment.shape
    if source_bases.shape != (relation_count, source_count, source_count):
        raise ValueError("source relation bases do not align with assignment")
    if target_bases.shape != (relation_count, target_count, target_count):
        raise ValueError("target relation bases do not align with assignment")
    if source_count == 0 or target_count == 0:
        return log_assignment.new_zeros(
            (source_count, target_count, relation_count * relation_count)
        )

    anchor_weights = log_assignment.exp().detach()
    source_relations = source_bases.clone()
    target_relations = target_bases.clone()
    source_relations[0] = 0.0
    target_relations[0] = 0.0
    outgoing = torch.einsum(
        "ria,ab,qjb->ijrq",
        source_relations,
        anchor_weights,
        target_relations,
    )
    incoming = torch.einsum(
        "rai,ab,qbj->ijrq",
        source_relations,
        anchor_weights,
        target_relations,
    )
    return (outgoing + incoming).flatten(start_dim=2)


def relative_geometry_consensus_features(
    log_assignment: Any,
    source_relations: Any,
    target_relations: Any,
) -> Any:
    """Return continuous within-page relation agreement around soft anchors.

    The feature compares only relative edges inside each page.  It excludes
    identity and relative node-size channels, so global translation, display
    size, and page order are not model inputs.  Source/target exchange simply
    transposes the candidate matrix, preserving matcher symmetry.
    """

    torch = _require_torch()
    if log_assignment.ndim != 2:
        raise ValueError("log assignment must be two-dimensional")
    source_count, target_count = log_assignment.shape
    expected_source = (source_count, source_count, RELATION_FEATURE_DIM)
    expected_target = (target_count, target_count, RELATION_FEATURE_DIM)
    if source_relations.shape != expected_source:
        raise ValueError("source relations do not align with assignment")
    if target_relations.shape != expected_target:
        raise ValueError("target relations do not align with assignment")
    feature_count = len(ALIGNMENT_RELATION_FEATURE_INDICES)
    if source_count == 0 or target_count == 0:
        return log_assignment.new_zeros(
            (source_count, target_count, feature_count)
        )

    feature_indices = torch.tensor(
        ALIGNMENT_RELATION_FEATURE_INDICES,
        dtype=torch.long,
        device=log_assignment.device,
    )
    source = source_relations.index_select(-1, feature_indices).clone()
    target = target_relations.index_select(-1, feature_indices).clone()
    source_indices = torch.arange(source_count, device=log_assignment.device)
    target_indices = torch.arange(target_count, device=log_assignment.device)
    source[source_indices, source_indices] = 0.0
    target[target_indices, target_indices] = 0.0
    anchor_weights = log_assignment.exp().detach()
    outgoing = torch.einsum(
        "iak,ab,jbk->ijk",
        source,
        anchor_weights,
        target,
    )
    incoming = torch.einsum(
        "aik,ab,bjk->ijk",
        source,
        anchor_weights,
        target,
    )
    return outgoing + incoming


def spatial_xml_alignment_choice(
    scores: Iterable[float],
    *,
    source_node: UINode,
    target_nodes: tuple[UINode, ...],
    source_graph: UIGraph,
    target_graph: UIGraph,
    top_k: int = SPATIAL_XML_ALIGNMENT_TOP_K,
    max_log_gap: float = SPATIAL_XML_ALIGNMENT_MAX_LOG_GAP,
    duplicate_max_log_gap: float = SPATIAL_XML_ALIGNMENT_DUPLICATE_MAX_LOG_GAP,
) -> int | None:
    """Return a conservative spatial/XML override for one model ranking.

    The rule compares candidates in two scale-free coordinate systems: the
    whole page and the largest branching XML ancestor.  It overrides the
    model only when both systems select the same candidate within the model's
    low-confidence top-k set.  Text and application identity are deliberately
    absent, so repeated labels and blank controls remain distinguishable.
    """

    score_values = tuple(float(value) for value in scores)
    if len(score_values) != len(target_nodes):
        raise ValueError("scores and target nodes must have equal length")
    if (
        len(score_values) < 2
        or top_k < 2
        or max_log_gap < 0.0
        or duplicate_max_log_gap < 0.0
    ):
        return None
    ranked_positions = sorted(
        range(len(score_values)),
        key=lambda position: (
            -score_values[position],
            target_nodes[position].node_id,
        ),
    )[:top_k]
    model_position = ranked_positions[0]
    source_page = _normalized_node_center(source_node, source_graph)
    source_xml = _dominant_xml_container_coordinates(source_node, source_graph)
    if source_page is None or source_xml is None:
        return None
    source_local, source_ordinal = source_xml

    page_positions: dict[int, tuple[float, float]] = {}
    local_positions: dict[int, tuple[float, float]] = {}
    ordinal_positions: dict[int, float] = {}
    for position in ranked_positions:
        target_node = target_nodes[position]
        page = _normalized_node_center(target_node, target_graph)
        target_xml = _dominant_xml_container_coordinates(
            target_node,
            target_graph,
        )
        if page is None or target_xml is None:
            return None
        local, ordinal = target_xml
        page_positions[position] = page
        local_positions[position] = local
        if ordinal is not None:
            ordinal_positions[position] = ordinal

    def nearest(
        source_position: tuple[float, float],
        candidate_positions: dict[int, tuple[float, float]],
    ) -> int:
        return min(
            ranked_positions,
            key=lambda position: (
                math.dist(source_position, candidate_positions[position]),
                -score_values[position],
                target_nodes[position].node_id,
            ),
        )

    page_choice = nearest(source_page, page_positions)
    local_choice = nearest(source_local, local_positions)
    ordinal_choice = None
    if source_ordinal is not None and len(ordinal_positions) == len(ranked_positions):
        ordinal_choice = min(
            ranked_positions,
            key=lambda position: (
                abs(source_ordinal - ordinal_positions[position]),
                -score_values[position],
                target_nodes[position].node_id,
            ),
        )
    xml_agrees = page_choice == local_choice or page_choice == ordinal_choice
    if page_choice == model_position or not xml_agrees:
        return None
    allowed_gap = max_log_gap
    if _observable_role_signature(
        target_nodes[model_position]
    ) == _observable_role_signature(target_nodes[page_choice]):
        allowed_gap = min(allowed_gap, duplicate_max_log_gap)
    if score_values[model_position] - score_values[page_choice] > allowed_gap:
        return None
    return page_choice


def spatial_xml_alignment_rerank_logits(
    logits: Any,
    *,
    source_node: UINode,
    target_nodes: tuple[UINode, ...],
    source_graph: UIGraph,
    target_graph: UIGraph,
) -> tuple[Any, bool]:
    """Swap the model winner with a dual-coordinate alignment consensus."""

    if getattr(logits, "ndim", None) != 1:
        raise ValueError("spatial/XML alignment expects one-dimensional logits")
    detached_scores = tuple(float(value.detach().cpu()) for value in logits)
    choice = spatial_xml_alignment_choice(
        detached_scores,
        source_node=source_node,
        target_nodes=target_nodes,
        source_graph=source_graph,
        target_graph=target_graph,
    )
    if choice is None:
        return logits, False
    model_position = min(
        range(len(detached_scores)),
        key=lambda position: (
            -detached_scores[position],
            target_nodes[position].node_id,
        ),
    )
    reranked = logits.clone()
    model_score = reranked[model_position].clone()
    reranked[model_position] = reranked[choice]
    reranked[choice] = model_score
    return reranked, True


def local_semantic_context_score(
    direct_pair_evidence: Any,
    source_bases: Any,
    target_bases: Any,
    source_numeric: Any,
    target_numeric: Any,
) -> Any:
    """Score controls through semantic evidence on their XML neighborhoods."""

    torch = _require_torch()
    if direct_pair_evidence.ndim != 3:
        raise ValueError("direct pair evidence must be three-dimensional")
    source_count, target_count, feature_count = direct_pair_evidence.shape
    relation_count = len(TYPED_RELATION_NAMES)
    if feature_count < len(DIRECT_SEMANTIC_EVIDENCE_NAMES):
        raise ValueError("direct pair evidence is missing semantic features")
    if source_bases.shape != (relation_count, source_count, source_count):
        raise ValueError("source relation bases do not align with pair evidence")
    if target_bases.shape != (relation_count, target_count, target_count):
        raise ValueError("target relation bases do not align with pair evidence")
    if source_numeric.shape[0] != source_count:
        raise ValueError("source numeric features do not align with pair evidence")
    if target_numeric.shape[0] != target_count:
        raise ValueError("target numeric features do not align with pair evidence")
    if source_numeric.shape[1] < 4 or target_numeric.shape[1] < 4:
        raise ValueError("numeric features must expose action and enabled state")
    if source_count == 0 or target_count == 0:
        return direct_pair_evidence.new_zeros((source_count, target_count))

    semantic_scores = direct_pair_evidence[
        ..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)
    ].sum(dim=-1)
    hierarchy_indices = tuple(
        TYPED_RELATION_NAMES.index(name)
        for name in ("parent", "child", "sibling", "ancestor", "descendant")
    )
    source_actionable = (
        source_numeric[:, :3].sum(dim=1).gt(0.0)
        & source_numeric[:, 3].gt(0.0)
    )
    target_actionable = (
        target_numeric[:, :3].sum(dim=1).gt(0.0)
        & target_numeric[:, 3].gt(0.0)
    )
    source_links = source_bases[list(hierarchy_indices)].gt(0).any(dim=0)
    target_links = target_bases[list(hierarchy_indices)].gt(0).any(dim=0)
    source_links = (source_links | source_links.T) & (~source_actionable)[None, :]
    target_links = (target_links | target_links.T) & (~target_actionable)[None, :]

    source_to_target_context = torch.where(
        target_links[None, :, :],
        semantic_scores[:, None, :],
        torch.zeros_like(semantic_scores)[:, None, :],
    ).amax(dim=2)
    source_context_to_target = torch.where(
        source_links[:, :, None],
        semantic_scores[None, :, :],
        torch.zeros_like(semantic_scores)[None, :, :],
    ).amax(dim=1)
    source_context_best = source_context_to_target
    context_to_context = torch.where(
        target_links[None, :, :],
        source_context_best[:, None, :],
        torch.zeros_like(source_context_best)[:, None, :],
    ).amax(dim=2)
    return torch.maximum(
        torch.maximum(source_to_target_context, source_context_to_target),
        context_to_context,
    )


def _build_legacy_relation_aware_matcher(
    config: MatcherConfig | None = None,
) -> Any:
    """Build the frozen v1 architecture for checkpoint reproduction only."""

    torch = _require_torch()
    nn = torch.nn
    cfg = config or MatcherConfig()
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if cfg.source_context_nodes <= 0:
        raise ValueError("source_context_nodes must be positive")
    if cfg.target_context_nodes <= 0:
        raise ValueError("target_context_nodes must be positive")
    if cfg.visual_canvas_size < cfg.visual_patch_size:
        raise ValueError("visual_canvas_size must cover one visual patch")
    if cfg.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if cfg.association_dim <= 0:
        raise ValueError("association_dim must be positive")
    if cfg.association_layers <= 0:
        raise ValueError("association_layers must be positive")
    if cfg.assignment_head not in {"pair_mlp", "mutual_projection"}:
        raise ValueError(
            "assignment_head must be 'pair_mlp' or 'mutual_projection'"
        )

    class RelationSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3, bias=False)
            self.relation_bias = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.num_heads),
            )
            self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.norm_attention = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.norm_output = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head_dim = cfg.hidden_dim // cfg.num_heads
            self.scale = self.head_dim**-0.5

        def forward(self, states: Any, relations: Any) -> Any:
            node_count = int(states.shape[0])
            qkv = self.qkv(states).reshape(
                node_count,
                3,
                cfg.num_heads,
                self.head_dim,
            )
            query, key, value = qkv.unbind(dim=1)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            scores = (query @ key.transpose(-1, -2)) * self.scale
            scores = scores + self.relation_bias(relations).permute(2, 0, 1)
            attention = torch.softmax(scores, dim=-1)
            context = attention @ value
            context = context.transpose(0, 1).reshape(node_count, cfg.hidden_dim)
            states = self.norm_attention(states + self.dropout(self.output(context)))
            return self.norm_output(states + self.dropout(self.feed_forward(states)))

    class MatcherLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attention = RelationSelfAttention()
            self.cross_attention = nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
            self.cross_feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            source_states: Any,
            target_states: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> tuple[Any, Any]:
            source_states = self.self_attention(source_states, source_relations)
            target_states = self.self_attention(target_states, target_relations)
            source_context = self.cross_attention(
                source_states.unsqueeze(0),
                target_states.unsqueeze(0),
                target_states.unsqueeze(0),
                need_weights=False,
            )[0].squeeze(0)
            target_context = self.cross_attention(
                target_states.unsqueeze(0),
                source_states.unsqueeze(0),
                source_states.unsqueeze(0),
                need_weights=False,
            )[0].squeeze(0)
            source_states = self.cross_norm(
                source_states + self.dropout(source_context)
            )
            target_states = self.cross_norm(
                target_states + self.dropout(target_context)
            )
            source_states = self.output_norm(
                source_states + self.dropout(self.cross_feed_forward(source_states))
            )
            target_states = self.output_norm(
                target_states + self.dropout(self.cross_feed_forward(target_states))
            )
            return source_states, target_states

    class RelationAwareCrossAttentionMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(
                cfg.vocab_size,
                cfg.token_dim,
                padding_idx=0,
            )
            self.token_projection = nn.Linear(cfg.token_dim, cfg.hidden_dim)
            self.numeric_projection = nn.Sequential(
                nn.LayerNorm(NUMERIC_FEATURE_DIM),
                nn.Linear(NUMERIC_FEATURE_DIM, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, cfg.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.missing_visual = nn.Parameter(torch.zeros(cfg.hidden_dim))
            self.input_norm = nn.LayerNorm(cfg.hidden_dim)
            self.layers = nn.ModuleList(MatcherLayer() for _ in range(cfg.num_layers))
            if cfg.assignment_head == "pair_mlp":
                self.cross_relation_projection = nn.Sequential(
                    nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                    nn.GELU(),
                )
                pair_dim = cfg.hidden_dim * 4 + cfg.relation_hidden_dim
                self.pair_head = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim, 1),
                )
                self.matchability = nn.Linear(cfg.hidden_dim, 1)
            else:
                self.affinity_projection = nn.Linear(
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    bias=False,
                )

        def _encode_nodes(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> Any:
            mask = token_ids.ne(0).unsqueeze(-1)
            embedded = self.token_embedding(token_ids)
            token_sum = (embedded * mask).sum(dim=1)
            token_count = mask.sum(dim=1).clamp_min(1)
            token_states = self.token_projection(token_sum / token_count)
            numeric_states = self.numeric_projection(numeric_features)
            visual_states = self.visual_encoder(visual_patches)
            missing = self.missing_visual.unsqueeze(0).expand(
                visual_states.shape[0], -1
            )
            visual_states = visual_mask * visual_states + (1.0 - visual_mask) * missing
            return self.input_norm(token_states + numeric_states + visual_states)

        def _score_pairs(
            self,
            source_states: Any,
            target_states: Any,
            pair_relations: Any,
        ) -> Any:
            source = source_states[:, None, :].expand(-1, target_states.shape[0], -1)
            target = target_states[None, :, :].expand(source_states.shape[0], -1, -1)
            relation = self.cross_relation_projection(pair_relations)
            pair = torch.cat(
                [source, target, torch.abs(source - target), source * target, relation],
                dim=-1,
            )
            logits = self.pair_head(pair).squeeze(-1)
            return (
                logits
                + self.matchability(source_states)
                + self.matchability(target_states).T
            )

        def _mutual_assignment(
            self,
            source_states: Any,
            target_states: Any,
        ) -> tuple[Any, Any]:
            source = self.affinity_projection(source_states)
            target = self.affinity_projection(target_states)
            affinity = (source @ target.T) / math.sqrt(cfg.hidden_dim)
            return mutual_log_assignment(affinity), affinity

        def forward(
            self,
            source_token_ids: Any,
            source_numeric: Any,
            source_relations: Any,
            target_token_ids: Any,
            target_numeric: Any,
            target_relations: Any,
            source_target_relations: Any,
            source_visual: Any,
            source_visual_mask: Any,
            target_visual: Any,
            target_visual_mask: Any,
        ) -> dict[str, Any]:
            source_states = self._encode_nodes(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            target_states = self._encode_nodes(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            assignment_scores_by_layer = []
            affinities_by_layer = []
            for layer in self.layers:
                source_states, target_states = layer(
                    source_states,
                    target_states,
                    source_relations,
                    target_relations,
                )
                if cfg.assignment_head == "mutual_projection":
                    assignment, affinity = self._mutual_assignment(
                        source_states,
                        target_states,
                    )
                    assignment_scores_by_layer.append(assignment)
                    affinities_by_layer.append(affinity)
            if cfg.assignment_head == "mutual_projection":
                pair_logits = assignment_scores_by_layer[-1]
                return {
                    "logits_ab": pair_logits,
                    "logits_ba": pair_logits.T,
                    "affinity": affinities_by_layer[-1],
                    "assignment_scores_by_layer": tuple(
                        assignment_scores_by_layer
                    ),
                    "affinities_by_layer": tuple(affinities_by_layer),
                    "source_states": source_states,
                    "target_states": target_states,
                }
            pair_logits = self._score_pairs(
                source_states,
                target_states,
                source_target_relations,
            )
            reverse_relations = source_target_relations.transpose(0, 1).clone()
            reverse_relations[..., 9] = -reverse_relations[..., 9]
            reverse_relations[..., 10] = -reverse_relations[..., 10]
            reverse_relations[..., 13] = -reverse_relations[..., 13]
            reverse_relations[..., 14] = -reverse_relations[..., 14]
            reverse_logits = self._score_pairs(
                target_states,
                source_states,
                reverse_relations,
            )
            return {
                "logits_ab": pair_logits,
                "logits_ba": reverse_logits,
                "affinity": pair_logits,
                "source_states": source_states,
                "target_states": target_states,
            }

    return RelationAwareCrossAttentionMatcher()


def _build_learned_graph_cross_attention_matcher(
    config: MatcherConfig,
) -> Any:
    """Build graph-context attention followed by bidirectional soft matching."""

    torch = _require_torch()
    nn = torch.nn
    cfg = config
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if cfg.source_context_nodes <= 0:
        raise ValueError("source_context_nodes must be positive")
    if cfg.target_context_nodes <= 0:
        raise ValueError("target_context_nodes must be positive")
    if cfg.visual_canvas_size < cfg.visual_patch_size:
        raise ValueError("visual_canvas_size must cover one visual patch")
    if cfg.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if cfg.assignment_head != "partial_assignment":
        raise ValueError(
            "learned graph cross-attention requires 'partial_assignment'"
        )

    class LearnedGraphAttention(nn.Module):
        """Learn which same-screen node relations should update each node."""

        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.key = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.value = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.relation_weight = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.num_heads * 2),
            )
            self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.norm_attention = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.norm_output = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head_dim = cfg.hidden_dim // cfg.num_heads
            self.scale = self.head_dim**-0.5

        def forward(self, states: Any, relations: Any) -> tuple[Any, Any]:
            node_count = int(states.shape[0])
            query = self.query(states).reshape(
                node_count, cfg.num_heads, self.head_dim
            ).transpose(0, 1)
            key = self.key(states).reshape(
                node_count, cfg.num_heads, self.head_dim
            ).transpose(0, 1)
            value = self.value(states).reshape(
                node_count, cfg.num_heads, self.head_dim
            ).transpose(0, 1)
            content_scores = (
                query @ key.transpose(-1, -2)
            ) * self.scale
            relation_parameters = self.relation_weight(relations)
            relation_gate, relation_bias = relation_parameters.chunk(2, dim=-1)
            relation_gate = (
                2.0 * torch.sigmoid(relation_gate)
            ).permute(2, 0, 1)
            relation_bias = relation_bias.permute(2, 0, 1)
            scores = content_scores * relation_gate + relation_bias
            attention = torch.softmax(scores, dim=-1)
            context = attention @ value
            context = context.transpose(0, 1).reshape(node_count, cfg.hidden_dim)
            states = self.norm_attention(
                states + self.dropout(self.output(context))
            )
            states = self.norm_output(
                states + self.dropout(self.feed_forward(states))
            )
            return states, attention

    class BidirectionalCrossAttention(nn.Module):
        """Exchange target-conditioned evidence through one soft affinity."""

        def __init__(self) -> None:
            super().__init__()
            self.affinity_projection = nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim,
                bias=False,
            )
            self.value = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.update = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim * 2),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            source_states: Any,
            target_states: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            projected_source = self.affinity_projection(source_states)
            projected_target = self.affinity_projection(target_states)
            affinity = (
                projected_source @ projected_target.T
            ) / math.sqrt(cfg.hidden_dim)
            attention_ab = torch.softmax(affinity, dim=1)
            attention_ba = torch.softmax(affinity, dim=0).T
            source_message = attention_ab @ self.value(target_states)
            target_message = attention_ba @ self.value(source_states)
            source_update = self.update(
                torch.cat([source_states, source_message], dim=-1)
            )
            target_update = self.update(
                torch.cat([target_states, target_message], dim=-1)
            )
            source_states = self.output_norm(
                source_states + self.dropout(source_update)
            )
            target_states = self.output_norm(
                target_states + self.dropout(target_update)
            )
            return (
                source_states,
                target_states,
                affinity,
                attention_ab,
                attention_ba,
            )

    class GraphMatchingLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph_attention = LearnedGraphAttention()
            self.cross_attention = BidirectionalCrossAttention()

        def forward(
            self,
            source_states: Any,
            target_states: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> tuple[Any, Any, dict[str, Any]]:
            source_states, source_attention = self.graph_attention(
                source_states,
                source_relations,
            )
            target_states, target_attention = self.graph_attention(
                target_states,
                target_relations,
            )
            (
                source_states,
                target_states,
                cross_affinity,
                attention_ab,
                attention_ba,
            ) = self.cross_attention(source_states, target_states)
            return source_states, target_states, {
                "source_graph": source_attention,
                "target_graph": target_attention,
                "cross_affinity": cross_affinity,
                "source_to_target": attention_ab,
                "target_to_source": attention_ba,
            }

    class PartialAssignmentHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compatibility_projection = nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim,
                bias=False,
            )
            self.matchability = nn.Linear(cfg.hidden_dim, 1)

        def forward(self, source_states: Any, target_states: Any) -> Any:
            source = self.compatibility_projection(source_states)
            target = self.compatibility_projection(target_states)
            return (
                (source @ target.T) / math.sqrt(cfg.hidden_dim)
                + self.matchability(source_states)
                + self.matchability(target_states).T
            )

    class SymmetricUnaryHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim * 2),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

        def forward(self, source_states: Any, target_states: Any) -> Any:
            normalized_source = torch.nn.functional.normalize(
                source_states,
                dim=-1,
            )
            normalized_target = torch.nn.functional.normalize(
                target_states,
                dim=-1,
            )
            cosine = normalized_source @ normalized_target.T
            source = source_states[:, None, :].expand(
                -1,
                target_states.shape[0],
                -1,
            )
            target = target_states[None, :, :].expand(
                source_states.shape[0],
                -1,
                -1,
            )
            symmetric_pair = torch.cat(
                [torch.abs(source - target), source * target],
                dim=-1,
            )
            return (
                self.logit_scale.exp().clamp(max=100.0) * cosine
                + self.residual(symmetric_pair).squeeze(-1)
            )

    class AssociationGraphLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message_projection = nn.Linear(
                cfg.association_dim,
                cfg.association_dim,
                bias=False,
            )
            self.message_norm = nn.LayerNorm(cfg.association_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.association_dim, cfg.association_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.association_dim * 2, cfg.association_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.association_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            pair_states: Any,
            anchor_weights: Any,
            source_bases: Any,
            target_bases: Any,
            relation_compatibility: Any,
        ) -> Any:
            anchored_states = pair_states * anchor_weights.unsqueeze(-1)
            source_messages = torch.einsum(
                "rik,kjd->rijd",
                source_bases,
                anchored_states,
            )
            typed_messages = torch.einsum(
                "rq,rijd->qijd",
                relation_compatibility,
                source_messages,
            )
            messages = torch.einsum(
                "qjl,qild->ijd",
                target_bases,
                typed_messages,
            )
            pair_states = self.message_norm(
                pair_states
                + self.dropout(self.message_projection(messages))
            )
            return self.output_norm(
                pair_states + self.dropout(self.feed_forward(pair_states))
            )

    class LearnableEdgeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            edge_feature_dim = len(LEARNED_EDGE_FEATURE_INDICES)
            self.register_buffer(
                "feature_indices",
                torch.tensor(LEARNED_EDGE_FEATURE_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.projection = nn.Sequential(
                nn.LayerNorm(edge_feature_dim),
                nn.Linear(edge_feature_dim, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.relation_hidden_dim),
                nn.GELU(),
            )
            self.basis = nn.Linear(
                cfg.relation_hidden_dim,
                cfg.num_heads,
                bias=False,
            )
            self.relevance = nn.Linear(
                cfg.relation_hidden_dim,
                cfg.num_heads,
            )

        def forward(self, relations: Any) -> tuple[Any, Any, Any, Any]:
            edge_states = self.projection(
                torch.index_select(relations, -1, self.feature_indices)
            )
            basis_distribution = torch.softmax(
                self.basis(edge_states),
                dim=-1,
            )
            edge_relevance = torch.softmax(
                self.relevance(edge_states).permute(2, 0, 1),
                dim=-1,
            )
            learned_bases = torch.einsum(
                "hik,ikq->hqik",
                edge_relevance,
                basis_distribution,
            )
            return (
                learned_bases,
                edge_states,
                edge_relevance,
                basis_distribution,
            )

    class RelationConditionedPairAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if cfg.association_dim % cfg.num_heads != 0:
                raise ValueError(
                    "association_dim must be divisible by num_heads"
                )
            self.head_dim = cfg.association_dim // cfg.num_heads
            self.query = nn.Linear(cfg.association_dim, cfg.association_dim)
            self.value = nn.Linear(cfg.association_dim, cfg.association_dim)
            self.output = nn.Linear(cfg.association_dim, cfg.association_dim)
            self.edge_encoder = LearnableEdgeEncoder()
            self.message_norm = nn.LayerNorm(cfg.association_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.association_dim, cfg.association_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.association_dim * 2, cfg.association_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.association_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            pair_states: Any,
            soft_assignment: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> tuple[Any, dict[str, Any]]:
            source_count, target_count, _ = pair_states.shape
            (
                source_bases,
                source_edge_states,
                source_edge_relevance,
                source_basis_distribution,
            ) = self.edge_encoder(source_relations)
            (
                target_bases,
                target_edge_states,
                target_edge_relevance,
                target_basis_distribution,
            ) = self.edge_encoder(target_relations)
            values = self.value(pair_states).reshape(
                source_count,
                target_count,
                cfg.num_heads,
                self.head_dim,
            ).permute(2, 0, 1, 3)
            anchored_values = values * soft_assignment[None, ..., None]
            source_messages = torch.einsum(
                "hqik,hkmd->hqimd",
                source_bases,
                anchored_values,
            )
            messages = torch.einsum(
                "hqjl,hqild->hijd",
                target_bases,
                source_messages,
            )
            queries = self.query(pair_states).reshape(
                source_count,
                target_count,
                cfg.num_heads,
                self.head_dim,
            ).permute(2, 0, 1, 3)
            attention_gate = torch.sigmoid(
                torch.sum(queries * messages, dim=-1)
                / math.sqrt(self.head_dim)
            )
            attended = (
                attention_gate[..., None] * messages
            ).permute(1, 2, 0, 3).reshape(
                source_count,
                target_count,
                cfg.association_dim,
            )
            states = self.message_norm(
                pair_states + self.dropout(self.output(attended))
            )
            states = self.output_norm(
                states + self.dropout(self.feed_forward(states))
            )
            return states, {
                "attention_gate": attention_gate,
                "source_edge_states": source_edge_states,
                "target_edge_states": target_edge_states,
                "source_edge_relevance": source_edge_relevance,
                "target_edge_relevance": target_edge_relevance,
                "source_basis_distribution": source_basis_distribution,
                "target_basis_distribution": target_basis_distribution,
                "source_relation_bases": source_bases,
                "target_relation_bases": target_bases,
            }

    class GraphTransformerLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            edge_feature_dim = len(LEARNED_EDGE_FEATURE_INDICES)
            self.register_buffer(
                "edge_feature_indices",
                torch.tensor(LEARNED_EDGE_FEATURE_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.self_attention_norm = nn.LayerNorm(cfg.hidden_dim)
            self.cross_attention_norm = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward_norm = nn.LayerNorm(cfg.hidden_dim)
            self.self_attention = nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cross_attention = nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.structural_attention_bias = nn.Sequential(
                nn.LayerNorm(edge_feature_dim),
                nn.Linear(edge_feature_dim, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.num_heads),
            )
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.dropout = nn.Dropout(cfg.dropout)

        def _self_attention(
            self,
            states: Any,
            relations: Any,
        ) -> tuple[Any, Any]:
            normalized = self.self_attention_norm(states)
            edge_features = torch.index_select(
                relations,
                -1,
                self.edge_feature_indices,
            )
            attention_bias = self.structural_attention_bias(
                edge_features
            ).permute(2, 0, 1)
            message, _ = self.self_attention(
                normalized.unsqueeze(0),
                normalized.unsqueeze(0),
                normalized.unsqueeze(0),
                attn_mask=attention_bias,
                need_weights=False,
            )
            return states + self.dropout(message.squeeze(0)), attention_bias

        def forward(
            self,
            source_states: Any,
            target_states: Any,
            source_relations: Any,
            target_relations: Any,
            cross_attention_bias: Any | None = None,
        ) -> tuple[Any, Any, dict[str, Any]]:
            source_states, source_attention_bias = self._self_attention(
                source_states,
                source_relations,
            )
            target_states, target_attention_bias = self._self_attention(
                target_states,
                target_relations,
            )
            normalized_source = self.cross_attention_norm(source_states)
            normalized_target = self.cross_attention_norm(target_states)
            source_cross_attention_bias = None
            target_cross_attention_bias = None
            if cross_attention_bias is not None:
                source_cross_attention_bias = cross_attention_bias.unsqueeze(
                    0
                ).expand(cfg.num_heads, -1, -1)
                target_cross_attention_bias = cross_attention_bias.T.unsqueeze(
                    0
                ).expand(cfg.num_heads, -1, -1)
            source_message, _ = self.cross_attention(
                normalized_source.unsqueeze(0),
                normalized_target.unsqueeze(0),
                normalized_target.unsqueeze(0),
                attn_mask=source_cross_attention_bias,
                need_weights=False,
            )
            target_message, _ = self.cross_attention(
                normalized_target.unsqueeze(0),
                normalized_source.unsqueeze(0),
                normalized_source.unsqueeze(0),
                attn_mask=target_cross_attention_bias,
                need_weights=False,
            )
            source_states = source_states + self.dropout(
                source_message.squeeze(0)
            )
            target_states = target_states + self.dropout(
                target_message.squeeze(0)
            )
            source_states = source_states + self.dropout(
                self.feed_forward(self.feed_forward_norm(source_states))
            )
            target_states = target_states + self.dropout(
                self.feed_forward(self.feed_forward_norm(target_states))
            )
            return source_states, target_states, {
                "source_attention_bias": source_attention_bias,
                "target_attention_bias": target_attention_bias,
                "cross_attention_bias": cross_attention_bias,
            }

    class PairwiseMatchingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim * 2),
                nn.Linear(cfg.hidden_dim * 2, cfg.association_dim),
                nn.GELU(),
                nn.Linear(cfg.association_dim, 1, bias=False),
            )
            nn.init.zeros_(self.projection[-1].weight)

        def forward(self, source_states: Any, target_states: Any) -> Any:
            source = source_states[:, None, :]
            target = target_states[None, :, :]
            pair_features = torch.cat(
                (
                    torch.abs(source - target),
                    source * target,
                ),
                dim=-1,
            )
            return self.projection(pair_features).squeeze(-1)

    class LearnedGraphCrossAttentionMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                self.token_embedding = nn.Embedding(
                    cfg.vocab_size,
                    cfg.token_dim,
                    padding_idx=0,
                )
            elif cfg.text_encoder == DIRECT_TEXT_EVIDENCE_ENCODER:
                self.present_text = nn.Parameter(
                    torch.zeros(TEXT_DESCRIPTOR_DIM)
                )
            else:
                raise ValueError(f"unsupported text encoder: {cfg.text_encoder}")
            if cfg.architecture in {
                OMNITRANSFER_MATCHING_ARCHITECTURE,
                OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
                OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                    self.text_projection = nn.Linear(
                        cfg.token_dim,
                        TEXT_DESCRIPTOR_DIM,
                    )
                self.xml_projection = nn.Sequential(
                    nn.LayerNorm(XML_NODE_FEATURE_DIM),
                    nn.Linear(XML_NODE_FEATURE_DIM, XML_DESCRIPTOR_DIM),
                    nn.GELU(),
                    nn.Linear(XML_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                )
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(
                        32,
                        VISUAL_DESCRIPTOR_DIM,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.missing_text = nn.Parameter(
                    torch.zeros(TEXT_DESCRIPTOR_DIM)
                )
                self.missing_visual = nn.Parameter(
                    torch.zeros(VISUAL_DESCRIPTOR_DIM)
                )
                self.descriptor_fusion = nn.Sequential(
                    nn.LayerNorm(NODE_DESCRIPTOR_DIM),
                    nn.Linear(NODE_DESCRIPTOR_DIM, NODE_DESCRIPTOR_DIM),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(NODE_DESCRIPTOR_DIM, NODE_DESCRIPTOR_DIM),
                )
                self.descriptor_norm = nn.LayerNorm(NODE_DESCRIPTOR_DIM)
                self.descriptor_projection = nn.Linear(
                    NODE_DESCRIPTOR_DIM,
                    cfg.hidden_dim,
                    bias=False,
                )
            else:
                self.token_projection = nn.Linear(cfg.token_dim, cfg.hidden_dim)
                self.numeric_projection = nn.Sequential(
                    nn.LayerNorm(NUMERIC_FEATURE_DIM),
                    nn.Linear(NUMERIC_FEATURE_DIM, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(
                        32,
                        cfg.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.missing_visual = nn.Parameter(torch.zeros(cfg.hidden_dim))
            self.input_norm = nn.LayerNorm(cfg.hidden_dim)
            if cfg.architecture in {
                OMNITRANSFER_MATCHING_ARCHITECTURE,
                OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                self.unary_head = SymmetricUnaryHead()
                self.relation_compatibility = nn.Parameter(
                    torch.eye(len(TYPED_RELATION_NAMES))
                )
                self.voting_strengths = nn.Parameter(
                    torch.zeros(cfg.num_layers)
                )
                if cfg.architecture == OMNITRANSFER_MATCHING_ARCHITECTURE:
                    pair_feature_dim = NODE_DESCRIPTOR_DIM * 2 + 5
                    self.canonical_pair_encoder = nn.Sequential(
                        nn.LayerNorm(pair_feature_dim),
                        nn.Linear(pair_feature_dim, cfg.association_dim),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                        nn.Linear(cfg.association_dim, cfg.association_dim),
                    )
                    self.canonical_pair_score = nn.Linear(
                        cfg.association_dim,
                        1,
                        bias=False,
                    )
                    self.relation_attention = RelationConditionedPairAttention()
                    self.relation_score = nn.Linear(
                        cfg.association_dim,
                        1,
                        bias=False,
                    )
                    nn.init.zeros_(self.relation_score.weight)
                if (
                    cfg.architecture
                    in {
                        OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                        OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                        OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
                    }
                ):
                    pair_feature_dim = NODE_DESCRIPTOR_DIM * 2 + 5
                    self.association_pair_encoder = nn.Sequential(
                        nn.LayerNorm(pair_feature_dim),
                        nn.Linear(pair_feature_dim, cfg.association_dim),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                        nn.Linear(cfg.association_dim, cfg.association_dim),
                    )
                    self.association_layers = nn.ModuleList(
                        AssociationGraphLayer()
                        for _ in range(cfg.association_layers)
                    )
                    self.association_score = nn.Linear(
                        cfg.association_dim,
                        1,
                        bias=False,
                    )
                    nn.init.zeros_(self.association_score.weight)
                    if (
                        cfg.architecture
                        in {
                            OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                            OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                            OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                            OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                            OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
                        }
                    ):
                        self.direct_evidence_head = nn.Linear(
                            1,
                            1,
                            bias=False,
                        )
                        nn.init.zeros_(self.direct_evidence_head.weight)
                        if cfg.architecture in {
                            OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                            OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                            OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                            OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
                        }:
                            self.anchor_voting_strength = nn.Parameter(
                                torch.tensor(math.log(math.expm1(2.0)))
                            )
                        if cfg.architecture in {
                            OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                            OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                        }:
                            self.transformer_layers = nn.ModuleList(
                                GraphTransformerLayer()
                                for _ in range(cfg.association_layers)
                            )
                            self.matching_heads = nn.ModuleList(
                                PairwiseMatchingHead()
                                for _ in range(cfg.association_layers)
                            )
                        if (
                            cfg.architecture
                            == OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE
                        ):
                            self.context_voting_strength = nn.Parameter(
                                torch.tensor(math.log(math.expm1(0.5)))
                            )
            else:
                self.layers = nn.ModuleList(
                    GraphMatchingLayer() for _ in range(cfg.num_layers)
                )
                if cfg.architecture in {
                    LIGHTGLUE_GRAPH_MATCHING_ARCHITECTURE,
                    OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
                }:
                    self.assignment_heads = nn.ModuleList(
                        PartialAssignmentHead() for _ in range(cfg.num_layers)
                    )
                else:
                    self.assignment_head = PartialAssignmentHead()

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            if mode and getattr(self, "_geometry_refinement_reranker", False):
                for module in self.children():
                    module.eval()
                self.geometry_alignment_score.train()
            elif mode and getattr(self, "_local_alignment_reranker", False):
                for module in self.children():
                    module.eval()
                self.local_alignment_score.train()
            elif mode and getattr(
                self,
                "_transformer_consensus_fusion",
                False,
            ):
                for module in self.children():
                    module.eval()
                self.fusion_score.train()
            elif mode and cfg.architecture in {
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            }:
                for name, module in self.named_children():
                    if name not in {"transformer_layers", "matching_heads"}:
                        module.eval()
                self.transformer_layers.train()
                self.matching_heads.train()
            return self

        def _encode_nodes(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> tuple[Any, Any]:
            mask = token_ids.ne(0).unsqueeze(-1)
            pooled_tokens = None
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                embedded = self.token_embedding(token_ids)
                token_sum = (embedded * mask).sum(dim=1)
                token_count = mask.sum(dim=1).clamp_min(1)
                pooled_tokens = token_sum / token_count
            visual_states = self.visual_encoder(visual_patches)
            missing_visual = self.missing_visual.unsqueeze(0).expand(
                visual_states.shape[0], -1
            )
            visual_states = visual_mask * visual_states + (
                1.0 - visual_mask
            ) * missing_visual
            if cfg.architecture in {
                OMNITRANSFER_MATCHING_ARCHITECTURE,
                OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
                OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                text_mask = token_ids.ne(0).any(dim=1, keepdim=True).to(
                    dtype=visual_states.dtype
                )
                if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                    text_states = self.text_projection(pooled_tokens)
                else:
                    text_states = self.present_text.unsqueeze(0).expand(
                        token_ids.shape[0], -1
                    )
                missing_text = self.missing_text.unsqueeze(0).expand(
                    text_states.shape[0], -1
                )
                text_states = text_mask * text_states + (
                    1.0 - text_mask
                ) * missing_text
                xml_states = self.xml_projection(numeric_features)
                raw_descriptor = torch.cat(
                    [text_states, visual_states, xml_states],
                    dim=-1,
                )
                descriptor = self.descriptor_norm(
                    raw_descriptor + self.descriptor_fusion(raw_descriptor)
                )
                states = self.input_norm(
                    self.descriptor_projection(descriptor)
                )
                return states, descriptor, raw_descriptor
            if pooled_tokens is None:
                raise ValueError(
                    "direct text evidence requires the multimodal matcher"
                )
            token_states = self.token_projection(pooled_tokens)
            numeric_states = self.numeric_projection(numeric_features)
            states = self.input_norm(
                token_states + numeric_states + visual_states
            )
            return states, states, states

        @staticmethod
        def _symmetric_pair_features(
            source_values: Any,
            target_values: Any,
            source_available: Any,
            target_available: Any,
        ) -> tuple[Any, Any, Any]:
            source = source_values[:, None, :]
            target = target_values[None, :, :]
            both_available = source_available[:, None] * target_available[None, :]
            exactly_one_available = torch.abs(
                source_available[:, None] - target_available[None, :]
            )
            features = torch.cat(
                [torch.abs(source - target), source * target],
                dim=-1,
            )
            return (
                features * both_available.unsqueeze(-1),
                both_available,
                exactly_one_available,
            )

        def _association_pair_states(
            self,
            source_modalities: Any,
            target_modalities: Any,
            source_text_available: Any,
            target_text_available: Any,
            source_visual_available: Any,
            target_visual_available: Any,
            anchor_weights: Any,
        ) -> Any:
            source_text, source_visual, source_xml = torch.split(
                source_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            target_text, target_visual, target_xml = torch.split(
                target_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            text, text_both, text_one = self._symmetric_pair_features(
                source_text,
                target_text,
                source_text_available,
                target_text_available,
            )
            visual, visual_both, visual_one = self._symmetric_pair_features(
                source_visual,
                target_visual,
                source_visual_available,
                target_visual_available,
            )
            xml, _, _ = self._symmetric_pair_features(
                source_xml,
                target_xml,
                torch.ones_like(source_text_available),
                torch.ones_like(target_text_available),
            )
            availability = torch.stack(
                [text_both, text_one, visual_both, visual_one, anchor_weights],
                dim=-1,
            )
            return self.association_pair_encoder(
                torch.cat([text, visual, xml, availability], dim=-1)
            )

        def _canonical_pair_states(
            self,
            source_modalities: Any,
            target_modalities: Any,
            source_token_ids: Any,
            target_token_ids: Any,
            source_visual_mask: Any,
            target_visual_mask: Any,
            anchor_weights: Any,
        ) -> Any:
            source_text, source_visual, source_xml = torch.split(
                source_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            target_text, target_visual, target_xml = torch.split(
                target_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            source_text_available = source_token_ids.ne(0).any(dim=1).to(
                source_modalities.dtype
            )
            target_text_available = target_token_ids.ne(0).any(dim=1).to(
                target_modalities.dtype
            )
            source_visual_available = source_visual_mask.squeeze(-1).to(
                source_modalities.dtype
            )
            target_visual_available = target_visual_mask.squeeze(-1).to(
                target_modalities.dtype
            )
            text, text_both, text_one = self._symmetric_pair_features(
                source_text,
                target_text,
                source_text_available,
                target_text_available,
            )
            visual, visual_both, visual_one = self._symmetric_pair_features(
                source_visual,
                target_visual,
                source_visual_available,
                target_visual_available,
            )
            xml, _, _ = self._symmetric_pair_features(
                source_xml,
                target_xml,
                torch.ones_like(source_text_available),
                torch.ones_like(target_text_available),
            )
            availability = torch.stack(
                [
                    text_both,
                    text_one,
                    visual_both,
                    visual_one,
                    anchor_weights,
                ],
                dim=-1,
            )
            return self.canonical_pair_encoder(
                torch.cat(
                    [
                        text,
                        visual,
                        xml,
                        availability,
                    ],
                    dim=-1,
                )
            )

        @staticmethod
        def _actionable_indices(numeric_features: Any) -> Any:
            mask = (
                numeric_features[:, :3].sum(dim=1).gt(0.0)
                & numeric_features[:, 3].gt(0.0)
            )
            return torch.nonzero(mask, as_tuple=False).flatten()

        def _candidate_indices(self, numeric_features: Any) -> Any:
            if cfg.candidate_policy == ALL_NODE_CANDIDATE_POLICY:
                return torch.arange(
                    numeric_features.shape[0],
                    device=numeric_features.device,
                )
            return self._actionable_indices(numeric_features)

        def _partial_assignment(
            self,
            affinity: Any,
            source_numeric: Any,
            target_numeric: Any,
        ) -> Any:
            source_indices = self._candidate_indices(source_numeric)
            target_indices = self._candidate_indices(target_numeric)
            assignment = torch.full_like(affinity, -1.0e4)
            if source_indices.numel() == 0 or target_indices.numel() == 0:
                return assignment
            candidate_affinity = affinity[
                source_indices[:, None],
                target_indices[None, :],
            ]
            candidate_assignment = mutual_log_assignment(candidate_affinity)
            assignment[
                source_indices[:, None],
                target_indices[None, :],
            ] = candidate_assignment
            return assignment

        def _relation_vote(
            self,
            soft_assignment: Any,
            source_bases: Any,
            target_bases: Any,
        ) -> Any:
            compatibility = 0.5 * (
                self.relation_compatibility
                + self.relation_compatibility.T
            )
            source_messages = torch.einsum(
                "rik,kl->ril",
                source_bases,
                soft_assignment,
            )
            typed_messages = torch.einsum(
                "rq,ril->qil",
                compatibility,
                source_messages,
            )
            vote = torch.einsum(
                "qil,qjl->ij",
                typed_messages,
                target_bases,
            )
            centered = vote - vote.mean()
            root_mean_square = centered.square().mean().clamp_min(
                torch.finfo(vote.dtype).eps
            ).sqrt()
            return centered / root_mean_square

        def _canonical_forward(
            self,
            source_states: Any,
            target_states: Any,
            source_modalities: Any,
            target_modalities: Any,
            source_token_ids: Any,
            target_token_ids: Any,
            source_visual_mask: Any,
            target_visual_mask: Any,
            source_numeric: Any,
            target_numeric: Any,
            source_relations: Any,
            target_relations: Any,
            detach_unary_for_relation: bool = False,
        ) -> dict[str, Any]:
            descriptor_affinity = self.unary_head(source_states, target_states)
            initial_soft_assignment = torch.exp(
                mutual_log_assignment(descriptor_affinity)
            )
            pair_states = self._canonical_pair_states(
                source_modalities,
                target_modalities,
                source_token_ids,
                target_token_ids,
                source_visual_mask,
                target_visual_mask,
                initial_soft_assignment,
            )
            unary_affinity = descriptor_affinity + self.canonical_pair_score(
                pair_states
            ).squeeze(-1)
            if detach_unary_for_relation:
                unary_affinity = unary_affinity.detach()
                pair_states = pair_states.detach()
            unary_assignment = self._partial_assignment(
                unary_affinity,
                source_numeric,
                target_numeric,
            )
            soft_assignment = torch.exp(mutual_log_assignment(unary_affinity))
            relation_states = pair_states
            affinity = unary_affinity
            assignments = [unary_assignment]
            affinities = [unary_affinity]
            attention_diagnostics = []
            relation_residual = torch.zeros_like(unary_affinity)
            for _ in range(cfg.num_layers):
                relation_states, diagnostics = self.relation_attention(
                    relation_states,
                    soft_assignment,
                    source_relations,
                    target_relations,
                )
                relation_residual = self.relation_score(
                    relation_states
                ).squeeze(-1)
                affinity = unary_affinity + relation_residual
                assignments.append(
                    self._partial_assignment(
                        affinity,
                        source_numeric,
                        target_numeric,
                    )
                )
                affinities.append(affinity)
                attention_diagnostics.append(diagnostics)
                soft_assignment = torch.exp(mutual_log_assignment(affinity))
            assignment_loss_weights = tuple(
                0.5 ** (cfg.num_layers - layer_index)
                for layer_index in range(cfg.num_layers + 1)
            )
            final_diagnostics = attention_diagnostics[-1]
            return {
                "logits_ab": assignments[-1],
                "logits_ba": assignments[-1].T,
                "affinity": affinity,
                "unary_affinity": unary_affinity,
                "descriptor_affinity": descriptor_affinity,
                "relation_residual": relation_residual,
                "soft_assignment": soft_assignment,
                "source_relation_bases": final_diagnostics[
                    "source_relation_bases"
                ],
                "target_relation_bases": final_diagnostics[
                    "target_relation_bases"
                ],
                "assignment_scores_by_layer": tuple(assignments),
                "assignment_loss_weights": assignment_loss_weights,
                "affinities_by_layer": tuple(affinities),
                "attention_by_layer": tuple(attention_diagnostics),
            }

        def _structured_forward(
            self,
            source_states: Any,
            target_states: Any,
            source_numeric: Any,
            target_numeric: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> dict[str, Any]:
            unary_affinity = self.unary_head(source_states, target_states)
            source_bases = typed_relation_bases(
                source_relations,
                source_numeric,
            )
            target_bases = typed_relation_bases(
                target_relations,
                target_numeric,
            )
            soft_assignment = torch.exp(
                mutual_log_assignment(unary_affinity)
            )
            assignments = []
            affinities = []
            relation_votes = []
            soft_assignments = []
            attention_diagnostics = []
            for round_index in range(cfg.num_layers):
                relation_vote = self._relation_vote(
                    soft_assignment,
                    source_bases,
                    target_bases,
                )
                strength = torch.nn.functional.softplus(
                    self.voting_strengths[round_index]
                )
                affinity = unary_affinity + strength * relation_vote
                assignment = self._partial_assignment(
                    affinity,
                    source_numeric,
                    target_numeric,
                )
                soft_assignment = torch.exp(
                    mutual_log_assignment(affinity)
                )
                assignments.append(assignment)
                affinities.append(affinity)
                relation_votes.append(relation_vote)
                soft_assignments.append(soft_assignment)
                attention_diagnostics.append(
                    {
                        "soft_assignment": soft_assignment,
                        "relation_vote": relation_vote,
                    }
                )
            return {
                "logits_ab": assignments[-1],
                "logits_ba": assignments[-1].T,
                "affinity": affinities[-1],
                "unary_affinity": unary_affinity,
                "assignment_scores_by_layer": tuple(assignments),
                "affinities_by_layer": tuple(affinities),
                "relation_votes_by_layer": tuple(relation_votes),
                "soft_assignments_by_layer": tuple(soft_assignments),
                "attention_by_layer": tuple(attention_diagnostics),
                "source_relation_bases": source_bases,
                "target_relation_bases": target_bases,
            }

        def _association_forward(
            self,
            structured_output: dict[str, Any],
            source_modalities: Any,
            target_modalities: Any,
            source_token_ids: Any,
            target_token_ids: Any,
            source_visual_mask: Any,
            target_visual_mask: Any,
            source_numeric: Any,
            target_numeric: Any,
            direct_pair_evidence: Any,
        ) -> dict[str, Any]:
            base_affinity = structured_output["affinity"]
            direct_evidence_residual = torch.zeros_like(base_affinity)
            if cfg.architecture in {
                OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                if direct_pair_evidence.shape != (
                    base_affinity.shape[0],
                    base_affinity.shape[1],
                    RELATION_FEATURE_DIM,
                ):
                    raise ValueError(
                        "direct pair evidence must align with candidate pairs"
                    )
                direct_evidence_residual = self.direct_evidence_head(
                    direct_pair_evidence[
                        ..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)
                    ].sum(dim=-1, keepdim=True)
                ).squeeze(-1)
            anchor_affinity = base_affinity + direct_evidence_residual
            anchor_weights = torch.exp(mutual_log_assignment(anchor_affinity))
            pair_states = self._association_pair_states(
                source_modalities,
                target_modalities,
                source_token_ids.ne(0).any(dim=1).to(base_affinity.dtype),
                target_token_ids.ne(0).any(dim=1).to(base_affinity.dtype),
                source_visual_mask.squeeze(-1).to(base_affinity.dtype),
                target_visual_mask.squeeze(-1).to(base_affinity.dtype),
                anchor_weights,
            )
            compatibility = 0.5 * (
                self.relation_compatibility
                + self.relation_compatibility.T
            )
            states_by_layer = []
            for layer in self.association_layers:
                pair_states = layer(
                    pair_states,
                    anchor_weights,
                    structured_output["source_relation_bases"],
                    structured_output["target_relation_bases"],
                    compatibility,
                )
                states_by_layer.append(pair_states)
            residual = self.association_score(pair_states).squeeze(-1)
            anchor_relation_vote = torch.zeros_like(base_affinity)
            anchor_vote_residual = torch.zeros_like(base_affinity)
            if cfg.architecture in {
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                anchor_relation_vote = semantic_anchor_relation_vote(
                    direct_pair_evidence,
                    structured_output["source_relation_bases"],
                    structured_output["target_relation_bases"],
                    compatibility,
                )
                anchor_vote_residual = torch.nn.functional.softplus(
                    self.anchor_voting_strength
                ) * anchor_relation_vote
            context_relation_score = torch.zeros_like(base_affinity)
            context_vote_residual = torch.zeros_like(base_affinity)
            if cfg.architecture == OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE:
                context_relation_score = local_semantic_context_score(
                    direct_pair_evidence,
                    structured_output["source_relation_bases"],
                    structured_output["target_relation_bases"],
                    source_numeric,
                    target_numeric,
                )
                context_vote_residual = torch.nn.functional.softplus(
                    self.context_voting_strength
                ) * context_relation_score
            affinity = (
                base_affinity
                + residual
                + direct_evidence_residual
                + anchor_vote_residual
                + context_vote_residual
            )
            assignment = self._partial_assignment(
                affinity,
                source_numeric,
                target_numeric,
            )
            return {
                **structured_output,
                "logits_ab": assignment,
                "logits_ba": assignment.T,
                "affinity": affinity,
                "base_affinity": base_affinity,
                "association_residual": residual,
                "direct_evidence_residual": direct_evidence_residual,
                "anchor_relation_vote": anchor_relation_vote,
                "anchor_vote_residual": anchor_vote_residual,
                "context_relation_score": context_relation_score,
                "context_vote_residual": context_vote_residual,
                "direct_pair_evidence": direct_pair_evidence,
                "association_states_by_layer": tuple(states_by_layer),
                "base_assignment_scores_by_layer": structured_output[
                    "assignment_scores_by_layer"
                ],
                "base_affinities_by_layer": structured_output[
                    "affinities_by_layer"
                ],
                "assignment_scores_by_layer": (assignment,),
                "affinities_by_layer": (affinity,),
            }

        def _fusion_evidence_features(self, v8_output: dict[str, Any]) -> Any:
            """Return page-calibrated direct and v8 score evidence."""

            score_components = torch.stack(
                tuple(
                    v8_output[name]
                    for name in (
                        "logits_ab",
                        "affinity",
                        "unary_affinity",
                        "base_affinity",
                        "association_residual",
                        "direct_evidence_residual",
                        "anchor_vote_residual",
                    )
                ),
                dim=-1,
            )
            centered = score_components - score_components.mean(
                dim=(0, 1),
                keepdim=True,
            )
            standardized = centered / centered.square().mean(
                dim=(0, 1),
                keepdim=True,
            ).clamp_min(torch.finfo(centered.dtype).eps).sqrt()
            return torch.cat(
                (
                    v8_output["direct_pair_evidence"],
                    torch.tanh(score_components / 5.0),
                    standardized,
                ),
                dim=-1,
            )

        def _normalized_consensus_features(
            self,
            v8_output: dict[str, Any],
        ) -> Any:
            """Return page-standardized typed relation consensus."""

            consensus_features = relational_consensus_features(
                v8_output["logits_ab"],
                v8_output["source_relation_bases"],
                v8_output["target_relation_bases"],
            )
            consensus_centered = consensus_features - (
                consensus_features.mean(dim=(0, 1), keepdim=True)
            )
            return consensus_centered / (
                consensus_centered.square()
                .mean(dim=(0, 1), keepdim=True)
                .clamp_min(torch.finfo(consensus_centered.dtype).eps)
                .sqrt()
            )

        def _local_alignment_forward(
            self,
            v9_output: dict[str, Any],
            source_numeric: Any,
            target_numeric: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> dict[str, Any]:
            """Learn one zero-initialized residual from local v9 evidence."""

            score_components = torch.stack(
                tuple(
                    v9_output[name]
                    for name in (
                        "unary_affinity",
                        "base_affinity",
                        "association_residual",
                        "direct_evidence_residual",
                        "anchor_vote_residual",
                        "context_relation_score",
                        "context_vote_residual",
                        "affinity",
                    )
                ),
                dim=-1,
            )
            centered = score_components - score_components.mean(
                dim=(0, 1),
                keepdim=True,
            )
            standardized = centered / centered.square().mean(
                dim=(0, 1),
                keepdim=True,
            ).clamp_min(torch.finfo(centered.dtype).eps).sqrt()
            local_consensus = self._normalized_consensus_features(v9_output)
            geometry_consensus = None
            if (
                getattr(self, "_geometric_alignment_reranker", False)
                or getattr(self, "_geometry_refinement_reranker", False)
            ):
                geometry_consensus = relative_geometry_consensus_features(
                    v9_output["logits_ab"],
                    source_relations,
                    target_relations,
                )
                geometry_centered = geometry_consensus - (
                    geometry_consensus.mean(dim=(0, 1), keepdim=True)
                )
                geometry_consensus = geometry_centered / (
                    geometry_centered.square()
                    .mean(dim=(0, 1), keepdim=True)
                    .clamp_min(torch.finfo(geometry_centered.dtype).eps)
                    .sqrt()
                )
            local_feature_parts = [
                v9_output["direct_pair_evidence"],
                torch.tanh(score_components / 5.0),
                standardized,
                local_consensus,
            ]
            if getattr(self, "_geometric_alignment_reranker", False):
                local_feature_parts.append(geometry_consensus)
            local_features = torch.cat(
                tuple(local_feature_parts),
                dim=-1,
            )
            local_residual = self.local_alignment_score(local_features).squeeze(-1)
            geometry_residual = torch.zeros_like(local_residual)
            if getattr(self, "_geometry_refinement_reranker", False):
                geometry_residual = self.geometry_alignment_score(
                    torch.cat((local_features, geometry_consensus), dim=-1)
                ).squeeze(-1)
            residual = local_residual + geometry_residual
            v9_affinity = v9_output["affinity"]
            affinity = v9_affinity + residual
            assignment = self._partial_assignment(
                affinity,
                source_numeric,
                target_numeric,
            )
            return {
                **v9_output,
                "logits_ab": assignment,
                "logits_ba": assignment.T,
                "affinity": affinity,
                "v9_affinity": v9_affinity,
                "local_alignment_residual": residual,
                "local_alignment_base_residual": local_residual,
                "geometry_alignment_residual": geometry_residual,
                "local_alignment_consensus": local_consensus,
                "local_alignment_geometry": geometry_consensus,
                "assignment_scores_by_layer": (assignment,),
                "assignment_loss_weights": (1.0,),
                "affinities_by_layer": (affinity,),
            }

        def _evidence_fusion_forward(
            self,
            v8_output: dict[str, Any],
            source_states: Any,
            target_states: Any,
            source_numeric: Any,
            target_numeric: Any,
        ) -> dict[str, Any]:
            """Learn one residual from the complete v8 evidence bundle."""

            base_affinity = v8_output["affinity"]
            evidence_features = self._fusion_evidence_features(v8_output)
            if getattr(self, "_relational_consensus_fusion", False):
                if getattr(self, "_normalized_relational_consensus", False):
                    consensus_features = self._normalized_consensus_features(
                        v8_output
                    )
                else:
                    consensus_features = relational_consensus_features(
                        v8_output["logits_ab"],
                        v8_output["source_relation_bases"],
                        v8_output["target_relation_bases"],
                    )
                residual = self.fusion_score(
                    torch.cat((evidence_features, consensus_features), dim=-1)
                ).squeeze(-1)
            elif getattr(self, "_calibrated_evidence_fusion", False):
                residual = self.fusion_score(evidence_features).squeeze(-1)
            else:
                source = source_states[:, None, :]
                target = target_states[None, :, :]
                node_features = torch.cat(
                    (
                        torch.abs(source - target),
                        source * target,
                    ),
                    dim=-1,
                )
                node_state = self.fusion_node_encoder(node_features)
                evidence_state = self.fusion_evidence_encoder(
                    evidence_features
                )
                residual = self.fusion_score(
                    torch.cat((node_state, evidence_state), dim=-1)
                ).squeeze(-1)
            affinity = base_affinity + residual
            assignment = self._partial_assignment(
                affinity,
                source_numeric,
                target_numeric,
            )
            return {
                **v8_output,
                "logits_ab": assignment,
                "logits_ba": assignment.T,
                "affinity": affinity,
                "base_affinity": base_affinity,
                "matching_residuals_by_layer": (residual,),
                "assignment_scores_by_layer": (assignment,),
                "assignment_loss_weights": (1.0,),
                "affinities_by_layer": (affinity,),
            }

        def _residual_adapter_forward(
            self,
            v8_output: dict[str, Any],
            source_states: Any,
            target_states: Any,
            source_numeric: Any,
            target_numeric: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> dict[str, Any]:
            if getattr(self, "_evidence_fusion_reranker", False):
                return self._evidence_fusion_forward(
                    v8_output,
                    source_states,
                    target_states,
                    source_numeric,
                    target_numeric,
                )
            base_affinity = v8_output["affinity"]
            assignments = []
            affinities = []
            residuals = []
            states_by_layer = []
            attention_by_layer = []
            cross_attention_priors = []
            cross_attention_prior = None
            if (
                cfg.architecture
                == OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE
            ):
                cross_attention_prior = mutual_log_assignment(base_affinity)
            current_affinity = base_affinity
            for layer, score_head in zip(
                self.transformer_layers,
                self.matching_heads,
                strict=True,
            ):
                (
                    source_states,
                    target_states,
                    diagnostics,
                ) = layer(
                    source_states,
                    target_states,
                    source_relations,
                    target_relations,
                    cross_attention_bias=cross_attention_prior,
                )
                residual = score_head(source_states, target_states)
                residual_base = (
                    current_affinity
                    if getattr(
                        self,
                        "_cumulative_transformer_residuals",
                        False,
                    )
                    else base_affinity
                )
                affinity = residual_base + residual
                assignment = self._partial_assignment(
                    affinity,
                    source_numeric,
                    target_numeric,
                )
                assignments.append(assignment)
                affinities.append(affinity)
                residuals.append(residual)
                states_by_layer.append((source_states, target_states))
                attention_by_layer.append(diagnostics)
                if cross_attention_prior is not None:
                    cross_attention_priors.append(cross_attention_prior)
                    cross_attention_prior = mutual_log_assignment(affinity)
                current_affinity = affinity
            assignment_loss_weights = tuple(
                0.5 ** (len(assignments) - layer_index - 1)
                for layer_index in range(len(assignments))
            )
            if getattr(self, "_transformer_consensus_fusion", False):
                consensus_features = self._normalized_consensus_features(
                    v8_output
                )
                fusion_residual = self.fusion_score(
                    torch.cat(
                        (
                            self._fusion_evidence_features(v8_output),
                            consensus_features,
                        ),
                        dim=-1,
                    )
                ).squeeze(-1)
                affinity = affinities[-1] + fusion_residual
                assignment = self._partial_assignment(
                    affinity,
                    source_numeric,
                    target_numeric,
                )
                residuals.append(fusion_residual)
                affinities.append(affinity)
                assignments.append(assignment)
                assignment_loss_weights = (0.0,) * (
                    len(assignments) - 1
                ) + (1.0,)
            return {
                **v8_output,
                "logits_ab": assignments[-1],
                "logits_ba": assignments[-1].T,
                "affinity": affinities[-1],
                "base_affinity": base_affinity,
                "v8_assignment": v8_output["logits_ab"],
                "matching_residuals_by_layer": tuple(residuals),
                "hidden_states_by_layer": tuple(states_by_layer),
                "attention_biases_by_layer": tuple(attention_by_layer),
                "cross_attention_priors_by_layer": tuple(
                    cross_attention_priors
                ),
                "assignment_scores_by_layer": tuple(assignments),
                "assignment_loss_weights": assignment_loss_weights,
                "affinities_by_layer": tuple(affinities),
            }

        def forward(
            self,
            source_token_ids: Any,
            source_numeric: Any,
            source_relations: Any,
            target_token_ids: Any,
            target_numeric: Any,
            target_relations: Any,
            source_target_relations: Any,
            source_visual: Any,
            source_visual_mask: Any,
            target_visual: Any,
            target_visual_mask: Any,
            detach_unary_for_relation: bool = False,
        ) -> dict[str, Any]:
            source_states, source_descriptors, source_modalities = self._encode_nodes(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            target_states, target_descriptors, target_modalities = self._encode_nodes(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            if cfg.architecture == OMNITRANSFER_MATCHING_ARCHITECTURE:
                output = self._canonical_forward(
                    source_states,
                    target_states,
                    source_modalities,
                    target_modalities,
                    source_token_ids,
                    target_token_ids,
                    source_visual_mask,
                    target_visual_mask,
                    source_numeric,
                    target_numeric,
                    source_relations,
                    target_relations,
                    detach_unary_for_relation,
                )
                return {
                    **output,
                    "source_descriptors": source_descriptors,
                    "target_descriptors": target_descriptors,
                    "source_states": source_states,
                    "target_states": target_states,
                }
            if cfg.architecture in {
                OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            }:
                output = self._structured_forward(
                    source_states,
                    target_states,
                    source_numeric,
                    target_numeric,
                    source_relations,
                    target_relations,
                )
                if (
                    cfg.architecture
                    in {
                        OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                        OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                        OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
                    }
                ):
                    output = self._association_forward(
                        output,
                        source_modalities,
                        target_modalities,
                        source_token_ids,
                        target_token_ids,
                        source_visual_mask,
                        target_visual_mask,
                        source_numeric,
                        target_numeric,
                        source_target_relations,
                    )
                if getattr(self, "_local_alignment_reranker", False):
                    output = self._local_alignment_forward(
                        output,
                        source_numeric,
                        target_numeric,
                        source_relations,
                        target_relations,
                    )
                if cfg.architecture in {
                    OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
                    OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
                }:
                    output = self._residual_adapter_forward(
                        output,
                        source_states,
                        target_states,
                        source_numeric,
                        target_numeric,
                        source_relations,
                        target_relations,
                    )
                return {
                    **output,
                    "source_descriptors": source_descriptors,
                    "target_descriptors": target_descriptors,
                    "source_states": source_states,
                    "target_states": target_states,
                }
            assignments = []
            affinities = []
            attention_diagnostics = []
            for layer_index, layer in enumerate(self.layers):
                source_states, target_states, diagnostics = layer(
                    source_states,
                    target_states,
                    source_relations,
                    target_relations,
                )
                assignment_head = (
                    self.assignment_heads[layer_index]
                    if cfg.architecture in {
                        LIGHTGLUE_GRAPH_MATCHING_ARCHITECTURE,
                        OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
                    }
                    else self.assignment_head
                )
                affinity = assignment_head(source_states, target_states)
                assignment = self._partial_assignment(
                    affinity,
                    source_numeric,
                    target_numeric,
                )
                affinities.append(affinity)
                assignments.append(assignment)
                attention_diagnostics.append(diagnostics)
            return {
                "logits_ab": assignments[-1],
                "logits_ba": assignments[-1].T,
                "affinity": affinities[-1],
                "assignment_scores_by_layer": tuple(assignments),
                "affinities_by_layer": tuple(affinities),
                "attention_by_layer": tuple(attention_diagnostics),
                "source_descriptors": source_descriptors,
                "target_descriptors": target_descriptors,
                "source_states": source_states,
                "target_states": target_states,
            }

    return LearnedGraphCrossAttentionMatcher()


def build_relation_aware_matcher(
    config: MatcherConfig | None = None,
) -> Any:
    """Build the versioned OmniTransfer matcher behind one public interface."""

    cfg = config or MatcherConfig()
    if cfg.candidate_policy not in {
        ACTIONABLE_CANDIDATE_POLICY,
        ALL_NODE_CANDIDATE_POLICY,
    }:
        raise ValueError(
            "candidate_policy must be 'actionable' or 'all_nodes'"
        )
    if cfg.architecture in {
        OMNITRANSFER_LOCAL_ALIGNMENT_ARCHITECTURE,
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
        OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE,
    }:
        torch = _require_torch()
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
            )
        )
        score_component_count = 8
        feature_dim = (
            len(DIRECT_PAIR_EVIDENCE_NAMES)
            + score_component_count * 2
            + len(TYPED_RELATION_NAMES) ** 2
        )
        if cfg.architecture == OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
            feature_dim += len(ALIGNMENT_RELATION_FEATURE_INDICES)
        model.local_alignment_score = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim),
            torch.nn.Linear(feature_dim, cfg.relation_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.relation_hidden_dim, 1, bias=False),
        )
        torch.nn.init.zeros_(model.local_alignment_score[-1].weight)
        if cfg.architecture == OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE:
            geometry_feature_dim = (
                feature_dim + len(ALIGNMENT_RELATION_FEATURE_INDICES)
            )
            model.geometry_alignment_score = torch.nn.Sequential(
                torch.nn.LayerNorm(geometry_feature_dim),
                torch.nn.Linear(
                    geometry_feature_dim,
                    cfg.relation_hidden_dim,
                ),
                torch.nn.GELU(),
                torch.nn.Linear(cfg.relation_hidden_dim, 1, bias=False),
            )
            torch.nn.init.zeros_(model.geometry_alignment_score[-1].weight)
        model._local_alignment_reranker = True
        model._geometric_alignment_reranker = (
            cfg.architecture == OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
        )
        model._geometry_refinement_reranker = (
            cfg.architecture == OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE
        )
        return model
    if cfg.architecture in {
        OMNITRANSFER_TRANSFORMER_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_NONLINEAR_CONSENSUS_ARCHITECTURE,
    }:
        torch = _require_torch()
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            )
        )
        evidence_feature_dim = RELATION_FEATURE_DIM + 7 * 2
        consensus_feature_dim = len(TYPED_RELATION_NAMES) ** 2
        fusion_feature_dim = evidence_feature_dim + consensus_feature_dim
        if cfg.architecture == OMNITRANSFER_NONLINEAR_CONSENSUS_ARCHITECTURE:
            model.fusion_score = torch.nn.Sequential(
                torch.nn.LayerNorm(fusion_feature_dim),
                torch.nn.Linear(fusion_feature_dim, 32),
                torch.nn.GELU(),
                torch.nn.Linear(32, 1, bias=False),
            )
            torch.nn.init.zeros_(model.fusion_score[-1].weight)
        else:
            model.fusion_score = torch.nn.Linear(
                fusion_feature_dim,
                1,
                bias=False,
            )
            torch.nn.init.zeros_(model.fusion_score.weight)
        model._transformer_consensus_fusion = True
        return model
    if cfg.architecture in {
        OMNITRANSFER_RELATIONAL_CONSENSUS_ARCHITECTURE,
        OMNITRANSFER_NORMALIZED_CONSENSUS_ARCHITECTURE,
    }:
        torch = _require_torch()
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            )
        )
        evidence_feature_dim = RELATION_FEATURE_DIM + 7 * 2
        consensus_feature_dim = len(TYPED_RELATION_NAMES) ** 2
        model.fusion_score = torch.nn.Linear(
            evidence_feature_dim + consensus_feature_dim,
            1,
            bias=False,
        )
        torch.nn.init.zeros_(model.fusion_score.weight)
        model._evidence_fusion_reranker = True
        model._relational_consensus_fusion = True
        model._normalized_relational_consensus = (
            cfg.architecture == OMNITRANSFER_NORMALIZED_CONSENSUS_ARCHITECTURE
        )
        return model
    if cfg.architecture == OMNITRANSFER_CALIBRATED_FUSION_ARCHITECTURE:
        torch = _require_torch()
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            )
        )
        evidence_feature_dim = RELATION_FEATURE_DIM + 7 * 2
        model.fusion_score = torch.nn.Linear(
            evidence_feature_dim,
            1,
            bias=False,
        )
        torch.nn.init.zeros_(model.fusion_score.weight)
        model._evidence_fusion_reranker = True
        model._calibrated_evidence_fusion = True
        return model
    if cfg.architecture == OMNITRANSFER_EVIDENCE_FUSION_ARCHITECTURE:
        torch = _require_torch()
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            )
        )
        score_component_count = 7
        evidence_feature_dim = (
            RELATION_FEATURE_DIM + score_component_count * 2
        )
        model.fusion_node_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(cfg.hidden_dim * 2),
            torch.nn.Linear(cfg.hidden_dim * 2, cfg.association_dim),
            torch.nn.GELU(),
        )
        model.fusion_evidence_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(evidence_feature_dim),
            torch.nn.Linear(evidence_feature_dim, cfg.association_dim),
            torch.nn.GELU(),
        )
        model.fusion_score = torch.nn.Sequential(
            torch.nn.LayerNorm(cfg.association_dim * 2),
            torch.nn.Linear(cfg.association_dim * 2, cfg.association_dim),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.association_dim, 1, bias=False),
        )
        torch.nn.init.zeros_(model.fusion_score[-1].weight)
        model._evidence_fusion_reranker = True
        return model
    if (
        cfg.architecture
        == OMNITRANSFER_UNIFIED_GRAPH_TRANSFORMER_ARCHITECTURE
    ):
        model = _build_learned_graph_cross_attention_matcher(
            replace(
                cfg,
                architecture=OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
            )
        )
        model._cumulative_transformer_residuals = True
        return model
    if cfg.architecture in {
        OMNITRANSFER_MATCHING_ARCHITECTURE,
        LEARNED_GRAPH_ATTENTION_ARCHITECTURE,
        LIGHTGLUE_GRAPH_MATCHING_ARCHITECTURE,
        OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE,
        OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE,
        OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE,
        OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE,
        OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE,
        OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE,
        OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE,
        OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE,
    }:
        return _build_learned_graph_cross_attention_matcher(cfg)
    if cfg.architecture == LEGACY_ATTENTION_ARCHITECTURE:
        return _build_legacy_relation_aware_matcher(cfg)
    raise ValueError(f"unsupported matcher architecture: {cfg.architecture}")


# Canonical paper-facing builder. Keep the descriptive implementation name
# checkpoint-compatible without exposing it as a second model.
build_omnitransfer_matcher = build_relation_aware_matcher


def matcher_inputs(
    source: UIGraph,
    target: UIGraph,
    *,
    config: MatcherConfig | None = None,
    device: str | Any = "cpu",
    source_context_mask_indices: Iterable[int] = (),
    target_context_mask_indices: Iterable[int] = (),
    source_text_mask_indices: Iterable[int] = (),
    target_text_mask_indices: Iterable[int] = (),
    feature_schema_id: str | None = None,
) -> tuple[Any, ...]:
    """Convert two UI graphs to tensors accepted by the learned matcher.

    Context-forced indices lose their own text/description and visual crop.
    Text-forced indices lose only text/description, retaining visual evidence.
    Class/action state and every within-screen relation always remain available.
    """

    torch = _require_torch()
    cfg = config or MatcherConfig()
    feature_schema_id = feature_schema_id or _feature_schema_for_config(cfg)
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
    source_masked = _validated_mask_indices(
        source_context_mask_indices,
        node_count=len(source.nodes),
        side="source",
    )
    target_masked = _validated_mask_indices(
        target_context_mask_indices,
        node_count=len(target.nodes),
        side="target",
    )
    source_text_masked = tuple(
        sorted(
            set(source_masked).union(
                _validated_mask_indices(
                    source_text_mask_indices,
                    node_count=len(source.nodes),
                    side="source_text",
                )
            )
        )
    )
    target_text_masked = tuple(
        sorted(
            set(target_masked).union(
                _validated_mask_indices(
                    target_text_mask_indices,
                    node_count=len(target.nodes),
                    side="target_text",
                )
            )
        )
    )
    source_token_rows = list(encoded_source.token_ids)
    target_token_rows = list(encoded_target.token_ids)
    for index in source_text_masked:
        source_token_rows[index] = _masked_context_token_ids(
            source.nodes[index],
            config=cfg,
        )
    for index in target_text_masked:
        target_token_rows[index] = _masked_context_token_ids(
            target.nodes[index],
            config=cfg,
        )
    source_token_ids = torch.as_tensor(
        source_token_rows,
        dtype=torch.long,
        device=device,
    )
    target_token_ids = torch.as_tensor(
        target_token_rows,
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
    pair_relations = torch.as_tensor(
        cross_relation_features(
            source,
            target,
            feature_schema_id=feature_schema_id,
        ),
        dtype=torch.float32,
        device=device,
    )
    source_visual, source_visual_mask = _visual_inputs(
        source,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        torch=torch,
        device=device,
    )
    target_visual, target_visual_mask = _visual_inputs(
        target,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        torch=torch,
        device=device,
    )
    if source_masked:
        source_visual_mask[list(source_masked)] = 0.0
    if target_masked:
        target_visual_mask[list(target_masked)] = 0.0
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


class RelationAwareMatcher:
    """Inference adapter that abstains instead of replaying source coordinates."""

    feature_schema_id: str | None = None

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
    ) -> RelationAwareMatcher:
        torch = _require_torch()
        payload = torch.load(Path(path), map_location=device)
        config_values = dict(payload["matcher_config"])
        if "architecture" not in config_values:
            config_values["architecture"] = LEGACY_ATTENTION_ARCHITECTURE
        config = MatcherConfig(**config_values)
        model = build_relation_aware_matcher(config)
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
        if (
            self.config.candidate_policy == ACTIONABLE_CANDIDATE_POLICY
            and not _is_actionable(source_node)
        ):
            return LearnedMatch(None, 0.0, 0.0, "source_node_not_actionable", ())
        if (
            self.config.candidate_policy == ACTIONABLE_CANDIDATE_POLICY
            and len(source.nodes) > self.config.source_context_nodes
        ):
            source = local_context_graph(
                source,
                anchor_node_id=source_node_id,
                max_nodes=self.config.source_context_nodes,
            )
        source_index = next(
            (
                index
                for index, node in enumerate(source.nodes)
                if node.node_id == source_node_id
            ),
            None,
        )
        if source_index is None:
            raise AssertionError("source context dropped its anchor node")
        allowed = set(candidate_node_ids or (node.node_id for node in target.nodes))
        candidate_indices = [
            index
            for index, node in enumerate(target.nodes)
            if node.node_id in allowed
            and (
                self.config.candidate_policy == ALL_NODE_CANDIDATE_POLICY
                or _is_actionable(node)
            )
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        inputs = matcher_inputs(
            source,
            target,
            config=self.config,
            device=self.device,
            feature_schema_id=self.feature_schema_id,
        )
        with torch.no_grad():
            output = self.model(*inputs)
            selected_logits = output["logits_ab"][source_index][candidate_indices]
            selected_affinity = output["affinity"][source_index][candidate_indices]
            if (
                self.config.architecture
                == OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
            ):
                selected_logits, _ = spatial_xml_alignment_rerank_logits(
                    selected_logits,
                    source_node=source.nodes[source_index],
                    target_nodes=tuple(
                        target.nodes[index] for index in candidate_indices
                    ),
                    source_graph=source,
                    target_graph=target,
                )
            rank_probabilities = torch.softmax(selected_logits, dim=0)
            match_probabilities = torch.sigmoid(selected_affinity)
        ranked = sorted(
            (
                (target.nodes[index].node_id, float(rank_probabilities[position]))
                for position, index in enumerate(candidate_indices)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        best_position = next(
            position
            for position, index in enumerate(candidate_indices)
            if target.nodes[index].node_id == best_id
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


# Canonical paper-facing adapter plus the older public compatibility name.
OmniTransferMatcher = RelationAwareMatcher
LearnedGraphMatcher = RelationAwareMatcher


def _is_actionable(node: UINode) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def is_actionable(node: UINode) -> bool:
    """Public compatibility wrapper for the canonical actionable predicate."""

    return _is_actionable(node)


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
    schema_versions = {
        OMNITRANSFER_MATCHING_ARCHITECTURE: (
            "omnitransfer.node_relation_assignment"
        ),
        OMNITRANSFER_GRAPH_MATCHING_ARCHITECTURE: (
            "omnitransfer.multimodal_graph_matching.v4"
        ),
        OMNITRANSFER_STRUCTURED_MATCHING_ARCHITECTURE: (
            "omnitransfer.structured_graph_matching.v5"
        ),
        OMNITRANSFER_ASSOCIATION_MATCHING_ARCHITECTURE: (
            "omnitransfer.association_graph_matching.v6"
        ),
        OMNITRANSFER_EVIDENCE_MATCHING_ARCHITECTURE: (
            "omnitransfer.direct_pair_evidence_matching.v7"
        ),
        OMNITRANSFER_ANCHOR_MATCHING_ARCHITECTURE: (
            "omnitransfer.semantic_anchor_matching.v8"
        ),
        OMNITRANSFER_TRANSFORMER_ADAPTER_ARCHITECTURE: (
            "omnitransfer.v8_transformer_adapter.v10"
        ),
        OMNITRANSFER_V8_GRAPH_TRANSFORMER_ARCHITECTURE: (
            "omnitransfer.v8_graph_transformer.v11"
        ),
        OMNITRANSFER_UNIFIED_GRAPH_TRANSFORMER_ARCHITECTURE: (
            "omnitransfer.unified_graph_transformer.v12"
        ),
        OMNITRANSFER_EVIDENCE_FUSION_ARCHITECTURE: (
            "omnitransfer.evidence_fusion.v13"
        ),
        OMNITRANSFER_CALIBRATED_FUSION_ARCHITECTURE: (
            "omnitransfer.calibrated_fusion.v14"
        ),
        OMNITRANSFER_RELATIONAL_CONSENSUS_ARCHITECTURE: (
            "omnitransfer.relational_consensus.v15"
        ),
        OMNITRANSFER_NORMALIZED_CONSENSUS_ARCHITECTURE: (
            "omnitransfer.normalized_consensus.v16"
        ),
        OMNITRANSFER_TRANSFORMER_CONSENSUS_ARCHITECTURE: (
            "omnitransfer.transformer_consensus.v17"
        ),
        OMNITRANSFER_NONLINEAR_CONSENSUS_ARCHITECTURE: (
            "omnitransfer.nonlinear_consensus.v18"
        ),
        OMNITRANSFER_CONTEXT_MATCHING_ARCHITECTURE: (
            "omnitransfer.local_semantic_context_matching.v9"
        ),
        OMNITRANSFER_LOCAL_ALIGNMENT_ARCHITECTURE: (
            "omnitransfer.learned_local_alignment.v9"
        ),
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE: (
            "omnitransfer.learned_geometric_alignment.v9"
        ),
        OMNITRANSFER_GEOMETRY_REFINEMENT_ARCHITECTURE: (
            "omnitransfer.learned_geometry_refinement.v9"
        ),
        LIGHTGLUE_GRAPH_MATCHING_ARCHITECTURE: (
            "omnitransfer.lightglue_graph_matching.v3"
        ),
        LEARNED_GRAPH_ATTENTION_ARCHITECTURE: (
            "omnitransfer.graph_cross_attention_matcher.v2"
        ),
        LEGACY_ATTENTION_ARCHITECTURE: (
            "omnitransfer.relation_aware_cross_attention_matcher.v1"
        ),
    }
    try:
        schema_version = schema_versions[config.architecture]
    except KeyError as exc:
        raise ValueError(
            f"unsupported matcher architecture: {config.architecture}"
        ) from exc
    torch.save(
        {
            "schema_version": schema_version,
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


def initialize_node_descriptor_from_matcher(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy only shape-compatible 128D node-descriptor state into a matcher."""

    descriptor_prefixes = (
        "token_embedding.",
        "text_projection.",
        "xml_projection.",
        "visual_encoder.",
        "missing_text",
        "missing_visual",
        "descriptor_fusion.",
        "descriptor_norm.",
        "descriptor_projection.",
        "input_norm.",
    )
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name.startswith(descriptor_prefixes)
        and name in target_state
        and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible node descriptor state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_association_matcher_from_structured(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy every shape-compatible v5 parameter into a v6 matcher."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible structured state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_evidence_matcher_from_association(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy every shape-compatible v6 parameter into a v7 matcher."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible association state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_anchor_matcher_from_evidence(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy every shape-compatible v7 parameter into a v8 matcher."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible evidence state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_context_matcher_from_anchor(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy every shape-compatible v8 parameter into a v9 matcher."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible anchor state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_local_alignment_from_context(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy the complete v9 base into a zero-residual local alignment model."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible v9 state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


def initialize_transformer_adapter_from_v8(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Copy the complete v8 matcher into a zero-residual Transformer adapter."""

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError("source matcher has no compatible v8 state")
    target_model.load_state_dict(transferred, strict=False)
    return tuple(sorted(transferred))


initialize_transformer_adapter_from_anchor = (
    initialize_transformer_adapter_from_v8
)


def initialize_omnitransfer_matcher_from_anchor(
    target_model: Any,
    source_model: Any,
) -> tuple[str, ...]:
    """Warm-start the canonical matcher without copying hard anchor voting."""

    descriptor_prefixes = (
        "token_embedding.",
        "text_projection.",
        "xml_projection.",
        "visual_encoder.",
        "missing_text",
        "missing_visual",
        "descriptor_fusion.",
        "descriptor_norm.",
        "descriptor_projection.",
        "input_norm.",
    )
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    transferred = {
        name: value
        for name, value in source_state.items()
        if name.startswith(descriptor_prefixes)
        and name in target_state
        and target_state[name].shape == value.shape
    }
    target_model.load_state_dict(transferred, strict=False)
    if not transferred:
        raise ValueError("v8 matcher has no canonical warm-start parameters")
    return tuple(sorted(transferred))


def prepare_visual_asset(
    screenshot_path: str | Path,
    *,
    canvas_size: int = 384,
) -> tuple[int, int, int]:
    """Decode and resize one screenshot before latency-critical matching."""

    path = Path(screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"screenshot is missing: {path}")
    array = _load_rgb_array(str(path.resolve()), canvas_size)
    return tuple(int(value) for value in array.shape)


def _visual_inputs(
    graph: UIGraph,
    *,
    patch_size: int,
    canvas_size: int,
    torch: Any,
    device: str | Any,
) -> tuple[Any, Any]:
    screenshot_path = str(graph.metadata.get("screenshot_path") or "")
    empty = torch.zeros(
        (len(graph.nodes), 3, patch_size, patch_size),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(graph.nodes), 1), dtype=torch.float32, device=device)
    if not screenshot_path:
        return empty, mask
    path = Path(screenshot_path)
    if not path.is_file():
        return empty, mask
    image_array = _load_rgb_array(str(path.resolve()), canvas_size)
    image = torch.as_tensor(image_array, dtype=torch.uint8, device=device)
    image = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
    graph_width = float(graph.width or image_array.shape[1])
    graph_height = float(graph.height or image_array.shape[0])
    normalized_boxes: list[tuple[float, float, float, float]] = []
    available: list[float] = []
    for node in graph.nodes:
        if bool(node.metadata.get("visual_disabled")):
            normalized_boxes.append((-1.0, -1.0, -1.0, -1.0))
            available.append(0.0)
            continue
        visual_bbox = _visual_bbox(node)
        if visual_bbox is None or graph_width <= 0.0 or graph_height <= 0.0:
            normalized_boxes.append((-1.0, -1.0, -1.0, -1.0))
            available.append(0.0)
            continue
        normalized_boxes.append(
            (
                _clip(2.0 * visual_bbox[0] / graph_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[1] / graph_height - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[2] / graph_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[3] / graph_height - 1.0, -1.0, 1.0),
            )
        )
        available.append(1.0)
    boxes = torch.as_tensor(
        normalized_boxes,
        dtype=torch.float32,
        device=device,
    )
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
    patches = torch.nn.functional.grid_sample(
        image.expand(len(graph.nodes), -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patches = _apply_visual_transform(patches, graph=graph, torch=torch, device=device)
    return (
        patches,
        torch.tensor(available, dtype=torch.float32, device=device).unsqueeze(1),
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
    return (((patches - 0.5) * contrast + 0.5 + brightness) * gains).clamp_(0.0, 1.0)


@lru_cache(maxsize=512)
def _load_rgb_array(screenshot_path: str, canvas_size: int) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Visual UI encoding requires NumPy and Pillow. Install omnitransfer[train]."
        ) from exc

    with Image.open(screenshot_path) as image:
        image = image.convert("RGB")
        if canvas_size > 0 and max(image.size) > canvas_size:
            scale = canvas_size / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        return np.array(image, dtype=np.uint8, copy=True)


def _node_token_ids(node: UINode, *, config: MatcherConfig) -> tuple[int, ...]:
    pieces: list[str] = ["node"]
    fields: list[tuple[str, str]] = [
        ("text", node.text),
        ("desc", node.content_desc),
        ("class", node.class_name),
    ]
    for field_name, value in fields:
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


def _pemm_v3_node_token_ids(
    node: UINode,
    *,
    config: MatcherConfig,
) -> tuple[int, ...]:
    """Reproduce the exact token contract used to train PEMM v3."""

    pieces: list[str] = ["node"]
    fields: list[tuple[str, str]] = [
        ("text", node.text),
        ("desc", node.content_desc),
        ("resource", node.resource_id),
        ("class", node.class_name),
    ]
    metadata = node.metadata or {}
    for field_name in ("action_type", "parent_text", "screen_region"):
        fields.append((field_name, str(metadata.get(field_name) or "")))
    for field_name in ("sibling_texts", "nearby_texts"):
        values = metadata.get(field_name) or ()
        if isinstance(values, str):
            values = (values,)
        fields.append((field_name, " ".join(str(value) for value in values)))
    for field_name, value in fields:
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


def _masked_context_token_ids(
    node: UINode,
    *,
    config: MatcherConfig,
) -> tuple[int, ...]:
    masked = replace(node, text="", content_desc="")
    feature_schema_id = _feature_schema_for_config(config)
    if feature_schema_id in {
        OMNITRANSFER_V4_FEATURE_SCHEMA_ID,
        OMNITRANSFER_V7_FEATURE_SCHEMA_ID,
    }:
        return _multimodal_text_token_ids(masked, config=config)
    if feature_schema_id == PEMM_V3_FEATURE_SCHEMA_ID:
        return _pemm_v3_node_token_ids(masked, config=config)
    return _node_token_ids(masked, config=config)


def _validated_mask_indices(
    indices: Iterable[int],
    *,
    node_count: int,
    side: str,
) -> tuple[int, ...]:
    normalized = tuple(sorted({int(index) for index in indices}))
    if any(index < 0 or index >= node_count for index in normalized):
        raise ValueError(f"{side} context mask index is outside the graph")
    return normalized


def _node_numeric_features(node: UINode, graph: UIGraph) -> tuple[float, ...]:
    del graph
    values = (
        float(node.clickable),
        float(node.editable),
        float(node.scrollable),
        float(node.enabled),
        *(0.0 for _ in range(NUMERIC_FEATURE_DIM - 4)),
    )
    if len(values) != NUMERIC_FEATURE_DIM:
        raise AssertionError("unexpected numeric feature dimension")
    return values


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


def _pemm_v3_node_numeric_features(
    node: UINode,
    graph: UIGraph,
) -> tuple[float, ...]:
    """Reproduce the exact numeric contract used to train PEMM v3."""

    bbox = _normalized_bbox(node.bbox, graph)
    if bbox is None:
        x1 = y1 = x2 = y2 = center_x = center_y = 0.0
        width = height = area = aspect = 0.0
        has_bbox = 0.0
    else:
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        area = width * height
        aspect = _clip(
            math.log(max(width, 1e-6) / max(height, 1e-6)) / 4.0,
            -1.0,
            1.0,
        )
        has_bbox = 1.0
    values = (
        float(node.clickable),
        float(node.editable),
        float(node.scrollable),
        float(node.enabled),
        has_bbox,
        x1,
        y1,
        x2,
        y2,
        center_x,
        center_y,
        width,
        height,
        area,
        aspect,
        min(float(node.depth) / 32.0, 1.0),
        min(float(len(node.child_ids)) / 16.0, 1.0),
        float(node.parent_id is not None),
    )
    if len(values) != NUMERIC_FEATURE_DIM:
        raise AssertionError("unexpected numeric feature dimension")
    return values


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


def _normalized_node_center(
    node: UINode,
    graph: UIGraph,
) -> tuple[float, float] | None:
    bbox = _normalized_bbox(node.bbox, graph)
    if bbox is None:
        return None
    return _center(bbox)


def _dominant_xml_container_coordinates(
    node: UINode,
    graph: UIGraph,
) -> tuple[tuple[float, float], float | None] | None:
    """Return container-local geometry and direct-branch order."""

    node_center = _normalized_node_center(node, graph)
    if node_center is None:
        return None
    nodes_by_id = {candidate.node_id: candidate for candidate in graph.nodes}
    current = node
    visited = {node.node_id}
    best_position: tuple[float, float] | None = None
    best_ordinal: float | None = None
    best_child_count = 1
    while current.parent_id and current.parent_id not in visited:
        branch_id = current.node_id
        visited.add(current.parent_id)
        parent = nodes_by_id.get(current.parent_id)
        if parent is None:
            break
        parent_bbox = _normalized_bbox(parent.bbox, graph)
        child_count = len(parent.child_ids)
        if parent_bbox is not None and child_count >= 2:
            left, top, right, bottom = parent_bbox
            width = right - left
            height = bottom - top
            if width > 0.0 and height > 0.0 and child_count > best_child_count:
                best_position = (
                    (node_center[0] - left) / width,
                    (node_center[1] - top) / height,
                )
                best_ordinal = (
                    parent.child_ids.index(branch_id) / (child_count - 1)
                    if branch_id in parent.child_ids
                    else None
                )
                best_child_count = child_count
        current = parent
    if best_position is None:
        return None
    return best_position, best_ordinal


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


def _observable_role_signature(node: UINode) -> tuple[Any, ...]:
    """Describe when two target instances expose the same observable role."""

    return (
        _semantic_fields(node),
        _normalize_text(node.class_name),
        bool(node.clickable),
        bool(node.editable),
        bool(node.scrollable),
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
