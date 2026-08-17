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

from omnitransfer.ui_graph import BBox, UIGraph, UINode

RELATION_FEATURE_DIM = 18
XML_NODE_FEATURE_DIM = 32
TEXT_DESCRIPTOR_DIM = 48
VISUAL_DESCRIPTOR_DIM = 48
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
    architecture: str = OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
    assignment_head: str = "partial_assignment"
    text_encoder: str = LEARNED_TOKEN_LOOKUP_ENCODER
    candidate_policy: str = ALL_NODE_CANDIDATE_POLICY


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
        config = MatcherConfig(**dict(payload["matcher_config"]))
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
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        inputs = matcher_inputs(
            source,
            target,
            config=self.config,
            device=self.device,
            feature_schema_id=GEOMETRIC_FEATURE_SCHEMA_ID,
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
    screenshot_width = float(image_array.shape[1])
    screenshot_height = float(image_array.shape[0])
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
        coordinate_space = str(
            node.metadata.get("visual_bbox_coordinate_space") or ""
        )
        bbox_width = (
            screenshot_width if coordinate_space == "page_pixels" else graph_width
        )
        bbox_height = (
            screenshot_height if coordinate_space == "page_pixels" else graph_height
        )
        normalized_boxes.append(
            (
                _clip(2.0 * visual_bbox[0] / bbox_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[1] / bbox_height - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[2] / bbox_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[3] / bbox_height - 1.0, -1.0, 1.0),
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
