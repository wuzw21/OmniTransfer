"""Stable replay-time action transfer interface."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from omnitransfer.learned_matcher import (
    GeometricMatcher,
)
from omnitransfer.numpy_v9_matcher import NumpyGeometricAlignmentMatcher
from omnitransfer.page_embedding import (
    DEFAULT_PAGE_EMBEDDING_CHECKPOINT,
    DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256,
    configured_matcher_checkpoint,
)
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record

_MIN_ANCHOR_OFFSET = -1.0
_MAX_ANCHOR_OFFSET = 2.0
_MATCHER_RELEASE = "omnitransfer-point-conditioned-sparse-graph-v10"
_MATCHER_MODE = "omnitransfer_point_conditioned_sparse_graph_v10"
_DEFAULT_MATCHER_CHECKPOINT = DEFAULT_PAGE_EMBEDDING_CHECKPOINT
_DEFAULT_MATCHER_SHA256 = DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256
_MATCHER_FEATURE_SCHEMA_ID = "omnitransfer-point-conditioned-sparse-graph-v10"
_MATCHER_FEATURE_SCHEMA_SPEC = (
    "node_encoder=learned_token_lookup,content_desc,class,action_state,deterministic_icon_v1;"
    "modality_fusion=sharp_learned_attention_missing_aware;"
    "within_page_relations=point_conditioned_sparse_graph,typed_hierarchy,sibling,"
    "ancestor,descendant,row,column,overlap,neighbors,near,control_context;"
    "refinement=two_layer_sparse_local_graph_plus_bidirectional_cross_page_attention;"
    "assignment=partial_bidirectional_matchability;"
    "page_embedding=attention_pool_normalized;"
    "forbidden=resource_id_lookup,node_id_lookup,raw_absolute_position,"
    "source_coordinate_passthrough,app_identity"
)
_MATCHER_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    _MATCHER_FEATURE_SCHEMA_SPEC.encode("utf-8")
).hexdigest()
_CANDIDATE_RANKING_SCHEMA_VERSION = "omnitransfer.candidate-ranking.v1"


def _configured_matcher_checkpoint() -> Path:
    return configured_matcher_checkpoint()


def _configured_matcher_mode() -> str:
    return _MATCHER_MODE


def rank_action_candidates(
    *,
    target_xml: str,
    source_xml: str | None = None,
    source_point: tuple[float, float] | None = None,
    source_element_id: str | None = None,
    source_element: dict[str, Any] | None = None,
    source_offset: tuple[float, float] | None = None,
    source_coordinate_space: str | None = None,
    target_display_size: tuple[float, float] | None = None,
    source_screenshot_path: str | None = None,
    target_screenshot_path: str | None = None,
    source_visual_rgb: dict[str, Any] | None = None,
    target_visual_rgb: dict[str, Any] | None = None,
    source_package_name: str | None = None,
    target_package_name: str | None = None,
    source_activity_name: str | None = None,
    target_activity_name: str | None = None,
    action_type: str = "click",
    top_k: int = 1,
    history: Any = None,
) -> dict[str, Any]:
    """Return the complete candidate ranking without accepting or rejecting it.

    Confidence and page-identity policy belong to the caller. This API only
    validates the inputs required to score, runs the matcher, and projects the
    recorded within-node offset onto every ranked target candidate.
    """

    del source_element, source_coordinate_space, target_display_size, history
    identity = _page_identity(
        source_package_name=source_package_name,
        target_package_name=target_package_name,
        source_activity_name=source_activity_name,
        target_activity_name=target_activity_name,
    )
    if not source_xml:
        return _candidate_ranking_result(
            status="invalid_input",
            reason="source_graph_required",
            action_type=action_type,
            top_k=top_k,
            identity=identity,
        )
    try:
        source = graph_from_record(
            {
                "xml": source_xml,
                "screenshot_path": source_screenshot_path,
                "visual_rgb": source_visual_rgb,
            },
            graph_id="source",
        )
        target = graph_from_record(
            {
                "xml": target_xml,
                "screenshot_path": target_screenshot_path,
                "visual_rgb": target_visual_rgb,
            },
            graph_id="target",
        )
    except Exception as error:
        return _candidate_ranking_result(
            status="invalid_input",
            reason="graph_parse_failed",
            action_type=action_type,
            top_k=top_k,
            identity=identity,
            error=str(error) or type(error).__name__,
        )
    source_node = _source_node(source, source_point, source_element_id)
    if source_node is None:
        return _candidate_ranking_result(
            status="invalid_input",
            reason="source_target_missing",
            source=source,
            target=target,
            action_type=action_type,
            top_k=top_k,
            identity=identity,
        )
    source_execution_node = _execution_target(
        source,
        source_node,
        action_type=action_type,
    )
    offset = _source_offset(
        source_execution_node or source_node,
        source_point,
        source_offset,
    )
    if offset is None:
        return _candidate_ranking_result(
            status="invalid_input",
            reason="source_point_or_offset_required",
            source=source,
            target=target,
            source_node=source_node,
            action_type=action_type,
            top_k=top_k,
            identity=identity,
        )
    equivalent_target = _equivalent_graph_target(source, target, source_node)
    if equivalent_target is not None:
        return _candidate_ranking_result(
            status="scored",
            reason="equivalent_ui_graph",
            mapping_mode="equivalent_ui_graph",
            source=source,
            target=target,
            source_node=source_node,
            offset=offset,
            ranked=[(1.0, equivalent_target)],
            pair_confidence=1.0,
            margin=1.0,
            action_type=action_type,
            top_k=top_k,
            identity=identity,
        )
    candidate_node_ids = tuple(
        node.node_id
        for node in target.nodes
        if node.bbox is not None and node.enabled
    )
    try:
        matcher = _get_matcher()
        matcher_metadata = _matcher_metadata(matcher)
        match = matcher.predict(
            source,
            target,
            source_node_id=source_node.node_id,
            candidate_node_ids=candidate_node_ids,
            min_probability=0.0,
            min_margin=0.0,
        )
    except Exception as error:
        return _candidate_ranking_result(
            status="matcher_unavailable",
            reason="matcher_unavailable",
            source=source,
            target=target,
            source_node=source_node,
            offset=offset,
            action_type=action_type,
            top_k=top_k,
            identity=identity,
            error=str(error) or type(error).__name__,
        )
    ranked = _learned_candidates(target, match.scores)
    return _candidate_ranking_result(
        status="scored" if ranked else "no_candidates",
        reason=str(match.reason or ("ranked" if ranked else "target_candidates_missing")),
        source=source,
        target=target,
        source_node=source_node,
        offset=offset,
        ranked=ranked,
        pair_confidence=float(match.probability),
        margin=float(match.margin),
        action_type=action_type,
        top_k=top_k,
        identity=identity,
        matcher_metadata=matcher_metadata,
    )


def action_transfer(**kwargs: Any) -> dict[str, Any]:
    """Deprecated name for the policy-free candidate-ranking API.

    The compatibility name intentionally does not select, reject, or add a
    ``mapped`` decision. Callers own all selection and fallback policy.
    """

    return rank_action_candidates(**kwargs)


def _candidate_ranking_result(
    *,
    status: str,
    reason: str,
    action_type: str,
    top_k: int,
    identity: dict[str, Any],
    mapping_mode: str | None = None,
    source: UIGraph | None = None,
    target: UIGraph | None = None,
    source_node: UINode | None = None,
    offset: tuple[float, float] | None = None,
    ranked: list[tuple[float, UINode]] | None = None,
    pair_confidence: float | None = None,
    margin: float | None = None,
    error: str | None = None,
    matcher_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    if mapping_mode is None:
        mapping_mode = _configured_matcher_mode()
    ranked_values = list(ranked or ())
    candidates = _projected_candidate_dicts(
        ranked_values,
        offset,
        target,
        action_type=action_type,
    )
    result: dict[str, Any] = {
        "schema_version": _CANDIDATE_RANKING_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "mapping_mode": mapping_mode,
        "src_element": _node_dict(source_node) if source_node is not None else {},
        "source_size": _graph_size(source),
        "target_size": _graph_size(target),
        "score": pair_confidence,
        "pair_confidence": pair_confidence,
        "rank_probability": candidates[0]["score"] if candidates else None,
        "margin": margin,
        "candidates": candidates,
        "top_candidates": candidates[: max(1, int(top_k))],
        "action_type": action_type,
        **identity,
    }
    if error:
        result["error"] = error
    if matcher_metadata:
        result.update(matcher_metadata)
    else:
        result["matcher_release"] = _MATCHER_RELEASE
    return result


def _projected_candidate_dicts(
    ranked: list[tuple[float, UINode]],
    offset: tuple[float, float] | None,
    graph: UIGraph | None,
    *,
    action_type: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate_score, candidate in ranked:
        bounds = candidate.bbox
        if bounds is None:
            continue
        execution = (
            _execution_target(graph, candidate, action_type=action_type)
            if graph is not None
            else None
        )
        execution_bounds = execution.bbox if execution is not None else bounds
        projected = _project_offset(offset or (0.5, 0.5), execution_bounds)
        candidates.append(
            {
                "candidate_id": _public_node_id(candidate),
                "resource_id": candidate.resource_id,
                "text": candidate.text,
                "content_desc": candidate.content_desc,
                "class": candidate.class_name,
                "bbox": list(bounds),
                "execution_candidate_id": (
                    _public_node_id(execution) if execution is not None else ""
                ),
                "execution_bbox": list(execution_bounds),
                "executable": execution is not None,
                "score": float(candidate_score),
                "new_x": projected[0],
                "new_y": projected[1],
            }
        )
    return candidates


def _execution_target(
    graph: UIGraph,
    node: UINode,
    *,
    action_type: str,
) -> UINode | None:
    """Resolve physical action ownership without changing correspondence identity."""

    nodes = {candidate.node_id: candidate for candidate in graph.nodes}
    current: UINode | None = node
    while current is not None:
        if current.bbox is not None and current.enabled:
            if action_type == "click" and current.clickable:
                return current
            if action_type == "input_text" and current.editable:
                return current
            if action_type == "swipe" and current.scrollable:
                return current
            if action_type == "long_press" and bool(
                current.metadata.get("long_clickable")
            ):
                return current
        current = nodes.get(current.parent_id or "")
    return None


def _page_identity(
    *,
    source_package_name: str | None,
    target_package_name: str | None,
    source_activity_name: str | None,
    target_activity_name: str | None,
) -> dict[str, Any]:
    source_package = str(source_package_name or "").strip()
    target_package = str(target_package_name or "").strip()
    return {
        "source_package_name": source_package,
        "target_package_name": target_package,
        "source_activity_name": str(source_activity_name or ""),
        "target_activity_name": str(target_activity_name or ""),
        "page_identity_match": (
            None
            if not source_package or not target_package
            else source_package == target_package
        ),
    }


def _get_matcher() -> Any:
    return _load_matcher(str(_configured_matcher_checkpoint()))


@lru_cache(maxsize=4)
def _load_matcher(checkpoint: str) -> Any:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"OmniTransfer checkpoint missing: {path}")
    if path.suffix == ".npz":
        return NumpyGeometricAlignmentMatcher.from_checkpoint(path)
    device = str(os.environ.get("OMNITRANSFER_MATCHER_DEVICE") or "cpu").strip()
    return GeometricMatcher.from_checkpoint(path, device=device)


def _matcher_metadata(
    matcher: Any,
) -> dict[str, str]:
    backend = str(getattr(matcher, "backend", "pytorch"))
    checkpoint = _configured_matcher_checkpoint()
    return {
        "matcher_release": _MATCHER_RELEASE,
        "matcher_backend": backend,
        "matcher_checkpoint": str(checkpoint),
        "matcher_checkpoint_sha256": (
            _DEFAULT_MATCHER_SHA256
            if checkpoint == _DEFAULT_MATCHER_CHECKPOINT.resolve()
            else ""
        ),
        "matcher_feature_schema": _MATCHER_FEATURE_SCHEMA_ID,
        "matcher_feature_schema_sha256": _MATCHER_FEATURE_SCHEMA_SHA256,
    }


def _top_rank_probability(ranked: list[tuple[float, UINode]]) -> float:
    return float(ranked[0][0]) if ranked else 0.0


def _learned_candidates(
    graph: UIGraph,
    scores: tuple[tuple[str, float], ...],
) -> list[tuple[float, UINode]]:
    nodes = {node.node_id: node for node in graph.nodes}
    return sorted(
        (
            (float(score), nodes[node_id])
            for node_id, score in scores
            if node_id in nodes
        ),
        key=lambda item: (-item[0], item[1].node_id),
    )


def _equivalent_graph_target(
    source: UIGraph,
    target: UIGraph,
    source_node: UINode,
) -> UINode | None:
    if source.width != target.width or source.height != target.height:
        return None
    if len(source.nodes) != len(target.nodes):
        return None
    if any(
        _structural_identity(left) != _structural_identity(right)
        for left, right in zip(source.nodes, target.nodes, strict=True)
    ):
        return None
    target_by_id = {node.node_id: node for node in target.nodes}
    target_node = target_by_id.get(source_node.node_id)
    if target_node is None or target_node.bbox is None:
        return None
    source_subtree = _subtree(source, source_node.node_id)
    target_subtree = _subtree(target, target_node.node_id)
    if len(source_subtree) != len(target_subtree):
        return None
    if any(
        _identity_key(left) != _identity_key(right)
        for left, right in zip(source_subtree, target_subtree, strict=True)
    ):
        return None
    has_semantic_anchor = any(
        node.text or node.content_desc for node in source_subtree
    )
    has_unique_resource = bool(source_node.resource_id) and sum(
        node.resource_id == source_node.resource_id for node in source.nodes
    ) == 1
    return target_node if has_semantic_anchor or has_unique_resource else None


def _structural_identity(node: UINode) -> tuple[Any, ...]:
    return (
        node.node_id,
        node.parent_id,
        node.child_ids,
        node.origin_id,
        node.resource_id,
        node.class_name,
        node.bbox,
        node.clickable,
        node.editable,
        node.scrollable,
        node.enabled,
    )


def _subtree(graph: UIGraph, root_id: str) -> tuple[UINode, ...]:
    by_id = {node.node_id: node for node in graph.nodes}
    pending = [root_id]
    nodes: list[UINode] = []
    while pending:
        node = by_id.get(pending.pop(0))
        if node is None:
            return ()
        nodes.append(node)
        pending.extend(node.child_ids)
    return tuple(nodes)


def _source_node(
    graph: UIGraph,
    point: tuple[float, float] | None,
    element_id: str | None,
) -> UINode | None:
    normalized = str(element_id or "").strip()
    if normalized:
        return next(
            (
                node
                for node in graph.nodes
                if normalized in {node.node_id, node.origin_id, node.resource_id}
            ),
            None,
        )
    if point is None:
        return None
    x, y = point
    containing = [
        node
        for node in graph.nodes
        if node.bbox is not None
        and node.bbox[0] <= x <= node.bbox[2]
        and node.bbox[1] <= y <= node.bbox[3]
    ]
    actionable = [
        node
        for node in containing
        if node.enabled and (node.clickable or node.editable or node.scrollable)
    ]
    candidates = actionable or containing
    return min(
        candidates,
        key=lambda node: (
            _area(node.bbox),
            not _has_stable_identity(node),
            -node.depth,
            node.node_id,
        ),
        default=None,
    )


def _has_stable_identity(node: UINode) -> bool:
    return bool(node.resource_id or node.text or node.content_desc)


def _identity_key(node: UINode) -> tuple[Any, ...]:
    return (
        _tail(node.resource_id),
        _text(node.text),
        _text(node.content_desc),
        _tail(node.class_name),
        node.clickable,
        node.editable,
        node.scrollable,
    )


def _source_offset(
    source: UINode,
    point: tuple[float, float] | None,
    explicit: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point is not None and source.bbox is not None:
        return _offset(point, source.bbox)
    if explicit is not None:
        try:
            offset = float(explicit[0]), float(explicit[1])
        except (IndexError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            and _MIN_ANCHOR_OFFSET <= value <= _MAX_ANCHOR_OFFSET
            for value in offset
        ):
            return None
        return offset
    if source.bbox is not None:
        return 0.5, 0.5
    return None


def _offset(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    width = max(1.0, bounds[2] - bounds[0])
    height = max(1.0, bounds[3] - bounds[1])
    return (
        _clamp((float(point[0]) - bounds[0]) / width),
        _clamp((float(point[1]) - bounds[1]) / height),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _project_offset(
    offset: tuple[float, float],
    target: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        target[0] + offset[0] * (target[2] - target[0]),
        target[1] + offset[1] * (target[3] - target[1]),
    )


def _node_dict(node: UINode) -> dict[str, Any]:
    return {
        "id": _public_node_id(node),
        "resource_id": node.resource_id,
        "text": node.text,
        "content_desc": node.content_desc,
        "class": node.class_name,
        "bounds": list(node.bbox or ()),
        "clickable": node.clickable,
        "editable": node.editable,
        "scrollable": node.scrollable,
    }


def _public_node_id(node: UINode) -> str:
    return str(node.resource_id or node.origin_id or node.node_id)


def _graph_size(graph: UIGraph | None) -> list[float] | None:
    if graph is None or graph.width is None or graph.height is None:
        return None
    if graph.width <= 0 or graph.height <= 0:
        return None
    return [graph.width, graph.height]


def _center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0


def _area(bounds: tuple[float, float, float, float] | None) -> float:
    return float("inf") if bounds is None else (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _tail(value: str) -> str:
    return _text(value).rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
