"""Historical 64D element cosine plus local-anchor transfer baseline.

This module is deliberately disconnected from the production transfer API. It
restores the deterministic dual-perspective element representation and local
anchor diffusion used by the pre-simplification OmniFlow implementation for
offline comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from omnitransfer.ui_graph import BBox, UIGraph, UINode


_LINE_PATTERN = re.compile(
    r"^(?P<source_class>\S+)(?P<source_bbox>\[[^\]]+\]\[[^\]]+\])\s+"
    r"(?P<target_class>\S+)(?P<target_bbox>\[[^\]]+\]\[[^\]]+\])\s+"
    r"(?P<app>\S+)\s+(?P<source_screen>\S+)\s+(?P<target_screen>\S+)\s+"
    r"(?P<widget_type>\S+)$"
)
_PROMINENCE_BINS = (0.005, 0.02, 0.08)
_ACTION_ID_HINTS = ("btn_", "button_", "ib_", "fab_", "action_")
_INPUT_ID_HINTS = ("et_", "edit_", "input_", "search_", "txt_")
_ICON_CLASSES = {
    "imageview",
    "imagebutton",
    "appcompatimageview",
    "circleimageview",
    "roundedimageview",
    "shapeableimageview",
}
_ACTION_CLASS_TOKENS = (
    "button",
    "image",
    "textview",
    "statictext",
    "textfield",
    "searchfield",
    "edittext",
    "switch",
    "checkbox",
    "radiobutton",
    "toggle",
    "cell",
    "link",
)


@dataclass(frozen=True)
class PublicWidgetPair:
    """One public iOS-to-Android widget annotation row."""

    row_index: int
    source_class: str
    source_bbox: BBox
    target_class: str
    target_bbox: BBox
    app: str
    source_screen: str
    target_screen: str
    widget_type: str


@dataclass(frozen=True)
class BoundNode:
    """An annotation binding and all equivalent XML-node ids."""

    node: UINode
    equivalent_node_ids: tuple[str, ...]
    normalized_bbox: BBox
    score: float
    iou: float


@dataclass(frozen=True)
class AnchorVoteCandidate:
    """One ranked target node and its decomposed evidence."""

    node_id: str
    score: float
    support: float
    semantic_similarity: float
    geometric_log_score: float
    anchor_count: int
    projected_point: tuple[float, float]


@dataclass(frozen=True)
class AnchorVoteResult:
    """Ranking and historical execution-gate decision for one source node."""

    candidates: tuple[AnchorVoteCandidate, ...]
    selected_node_id: str | None
    execute: bool
    reason: str
    latency_ms: float

    @property
    def ranked_node_ids(self) -> tuple[str, ...]:
        return tuple(candidate.node_id for candidate in self.candidates)


def load_public_widget_pairs(path: str | Path) -> list[PublicWidgetPair]:
    """Parse the 6,730-row public widget-pair text file."""

    pairs: list[PublicWidgetPair] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            match = _LINE_PATTERN.match(text)
            if match is None:
                raise ValueError(f"malformed_public_widget_pair:{row_index + 1}")
            values = match.groupdict()
            pairs.append(
                PublicWidgetPair(
                    row_index=row_index,
                    source_class=values["source_class"],
                    source_bbox=_parse_bbox(values["source_bbox"]),
                    target_class=values["target_class"],
                    target_bbox=_parse_bbox(values["target_bbox"]),
                    app=values["app"],
                    source_screen=values["source_screen"],
                    target_screen=values["target_screen"],
                    widget_type=values["widget_type"],
                )
            )
    return pairs


def normalize_public_bbox(
    bbox: BBox,
    graph: UIGraph,
    screenshot_size: tuple[int, int] | None,
) -> BBox:
    """Map screenshot-pixel annotations into an XML's native coordinates.

    Retina-style scale is applied only when width and height imply the same
    isotropic transform. Android screenshot/XML height differences caused by
    system chrome therefore remain in the XML coordinate system.
    """

    if screenshot_size is None or not graph.width or not graph.height:
        return bbox
    screenshot_width, screenshot_height = screenshot_size
    if screenshot_width <= 0 or screenshot_height <= 0:
        return bbox
    scale_x = float(graph.width) / float(screenshot_width)
    scale_y = float(graph.height) / float(screenshot_height)
    relative_difference = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-9)
    scaled_coordinate_space = relative_difference <= 0.03 and (
        scale_x <= 0.8 or scale_x >= 1.25
    )
    if not scaled_coordinate_space:
        return bbox
    return (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )


def bind_public_bbox(
    graph: UIGraph,
    bbox: BBox,
    class_name: str,
    *,
    screenshot_size: tuple[int, int] | None = None,
) -> BoundNode | None:
    """Bind one public bbox to real XML nodes without fabricating semantics."""

    normalized = normalize_public_bbox(bbox, graph, screenshot_size)
    ranked: list[tuple[float, float, float, UINode]] = []
    for node in graph.nodes:
        if node.bbox is None:
            continue
        overlap = _iou(normalized, node.bbox)
        center_inside = _contains(node.bbox, _center(normalized))
        class_match = _class_tail(class_name) == _class_tail(node.class_name)
        area_similarity = _area_similarity(normalized, node.bbox)
        score = 3.0 * overlap + 1.5 * float(class_match) + 0.25 * float(center_inside)
        score += 0.25 * area_similarity
        ranked.append((score, overlap, _area(node.bbox), node))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].node_id))
    score, overlap, _, node = ranked[0]
    class_match = _class_tail(class_name) == _class_tail(node.class_name)
    if overlap < 0.50 and not (
        class_match
        and _contains(node.bbox or normalized, _center(normalized))
        and _area_similarity(normalized, node.bbox or normalized) >= 0.50
    ):
        return None
    equivalent_ids = tuple(
        candidate.node_id
        for _, candidate_overlap, _, candidate in ranked
        if candidate_overlap >= 0.95
        and _class_tail(candidate.class_name) == _class_tail(node.class_name)
    )
    return BoundNode(
        node=node,
        equivalent_node_ids=equivalent_ids or (node.node_id,),
        normalized_bbox=normalized,
        score=float(score),
        iou=float(overlap),
    )


def encode_graph_64d(graph: UIGraph) -> dict[str, np.ndarray]:
    """Encode every graph node with the historical deterministic 64D layout."""

    by_id = {node.node_id: node for node in graph.nodes}
    sibling_counts: dict[str, int] = {}
    for node in graph.nodes:
        if node.parent_id:
            sibling_counts[node.parent_id] = sibling_counts.get(node.parent_id, 0) + 1
    root_area = max(1.0, float(graph.width or 1.0) * float(graph.height or 1.0))
    signature_cache: dict[tuple[str, int], str] = {}

    def subtree_signature(node_id: str, depth: int) -> str:
        cache_key = (node_id, depth)
        if cache_key in signature_cache:
            return signature_cache[cache_key]
        node = by_id[node_id]
        token = (
            f"{_class_tail(node.class_name)}|t{int(bool(_node_text(node)))}|"
            f"c{min(len(node.child_ids), 5)}"
        )
        if depth <= 0 or not node.child_ids:
            signature_cache[cache_key] = token
            return token
        child_signatures = sorted(
            {
                subtree_signature(child_id, depth - 1)
                for child_id in node.child_ids
                if child_id in by_id
            }
        )[:3]
        signature = token + "->[" + ",".join(child_signatures) + "]"
        signature_cache[cache_key] = signature
        return signature

    vectors: dict[str, np.ndarray] = {}
    for node in graph.nodes:
        class_tail = _class_tail(node.class_name)
        class_lower = class_tail.lower()
        text = _node_text(node)
        has_text = len(text) >= 2
        is_icon = class_lower in _ICON_CLASSES or "image" in class_lower or "icon" in class_lower
        content_type = np.zeros(4, dtype=np.float32)
        content_type[3 if has_text and is_icon else 0 if has_text else 1 if is_icon else 2] = 1.0
        clickable = node.clickable or "button" in class_lower or "cell" in class_lower
        editable = node.editable or any(
            token in class_lower for token in ("edittext", "textfield", "searchfield")
        )
        scrollable = node.scrollable or any(
            token in class_lower
            for token in ("recyclerview", "listview", "scrollview", "gridview", "viewpager")
        )
        checkable = any(
            token in class_lower for token in ("switch", "checkbox", "radio", "toggle")
        )
        affordance = np.asarray(
            [clickable, editable, scrollable, checkable], dtype=np.float32
        )
        area_ratio = _area(node.bbox) / root_area if node.bbox else 0.0
        prominence = np.zeros(4, dtype=np.float32)
        prominence[_bucket(area_ratio, _PROMINENCE_BINS)] = 1.0
        visual_state = np.asarray(
            [False, not node.enabled, False, area_ratio > 0.02 and clickable],
            dtype=np.float32,
        )
        attributes = np.asarray(
            [clickable, False, editable, editable, scrollable, checkable, node.enabled, False],
            dtype=np.float32,
        )
        has_siblings = bool(
            node.parent_id and sibling_counts.get(node.parent_id, 0) > 1
        )
        hierarchy = np.asarray([not node.child_ids, has_siblings], dtype=np.float32)
        resource_id = str(node.resource_id or "").lower()
        id_hint = np.asarray(
            [
                any(prefix in resource_id for prefix in _ACTION_ID_HINTS),
                any(prefix in resource_id for prefix in _INPUT_ID_HINTS),
            ],
            dtype=np.float32,
        )
        human = np.concatenate(
            [
                _signed_hash(text, 16),
                content_type,
                affordance,
                prominence,
                visual_state,
            ]
        ).astype(np.float32)
        programmer = np.concatenate(
            [
                _signed_hash(class_tail, 8),
                attributes,
                _signed_hash(subtree_signature(node.node_id, 2), 12),
                hierarchy,
                id_hint,
            ]
        ).astype(np.float32)
        human_norm = float(np.linalg.norm(human))
        programmer_norm = float(np.linalg.norm(programmer))
        if human_norm > 1e-6:
            human = human / human_norm * 0.55
        if programmer_norm > 1e-6:
            programmer = programmer / programmer_norm * 0.45
        vectors[node.node_id] = np.concatenate([human, programmer]).astype(np.float32)
    return vectors


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return finite cosine similarity, or zero for either zero vector."""

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    denominator = float(np.linalg.norm(first_array) * np.linalg.norm(second_array))
    if denominator <= 1e-9:
        return 0.0
    return float(np.dot(first_array, second_array) / denominator)


class AnchorVote64DMatcher:
    """Offline matcher using 64D cosine and candidate-local anchor diffusion."""

    semantic_temperature = 0.05
    geometric_sigma = 0.15
    projection_distance_deadzone = 0.02
    projection_tree_distance_weight = 0.15
    min_vote_support = 0.45
    min_vote_margin = 0.05
    min_action_similarity = 0.30

    def __init__(self, source_graph: UIGraph, target_graph: UIGraph):
        self.source_graph = source_graph
        self.target_graph = target_graph
        self.source_by_id = {node.node_id: node for node in source_graph.nodes}
        self.target_by_id = {node.node_id: node for node in target_graph.nodes}
        self.source_vectors = encode_graph_64d(source_graph)
        self.target_vectors = encode_graph_64d(target_graph)
        self.source_node_ids = tuple(self.source_vectors)
        self.target_node_ids = tuple(self.target_vectors)
        self.source_indices = {
            node_id: index for index, node_id in enumerate(self.source_node_ids)
        }
        self.target_indices = {
            node_id: index for index, node_id in enumerate(self.target_node_ids)
        }
        source_matrix = np.vstack(
            [self.source_vectors[node_id] for node_id in self.source_node_ids]
        ).astype(np.float32, copy=False)
        target_matrix = np.vstack(
            [self.target_vectors[node_id] for node_id in self.target_node_ids]
        ).astype(np.float32, copy=False)
        source_norms = np.linalg.norm(source_matrix, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)
        source_norms[source_norms <= 1e-9] = 1.0
        target_norms[target_norms <= 1e-9] = 1.0
        self.similarity_matrix = (source_matrix / source_norms) @ (
            target_matrix / target_norms
        ).T
        self.target_candidates = tuple(_action_candidates(target_graph))
        self.selector_candidates = tuple(
            node
            for node in target_graph.nodes
            if node.bbox is not None and node.enabled
        )
        self.target_diagonal = max(
            1.0,
            math.hypot(float(target_graph.width or 1.0), float(target_graph.height or 1.0)),
        )
        self.scale_x = float(target_graph.width or 1.0) / max(
            1.0, float(source_graph.width or 1.0)
        )
        self.scale_y = float(target_graph.height or 1.0) / max(
            1.0, float(source_graph.height or 1.0)
        )
        self._source_context_cache: dict[str, tuple[UINode, ...]] = {}
        self._target_context_cache: dict[str, tuple[UINode, ...]] = {}
        self._source_ancestors = _ancestor_chains(self.source_by_id)
        self._target_ancestors = _ancestor_chains(self.target_by_id)
        self._source_relation_cache: dict[
            tuple[str, str], dict[str, bool | int | None]
        ] = {}
        self._target_relation_cache: dict[
            tuple[str, str], dict[str, bool | int | None]
        ] = {}
        self._tree_distance_cache: dict[tuple[bool, str, str], int | None] = {}
        self._reliability_cache: dict[int, float] = {}

    def rank(
        self,
        source_node_id: str,
        *,
        source_point: tuple[float, float] | None = None,
        candidate_node_ids: Iterable[str] | None = None,
        use_anchors: bool = True,
    ) -> AnchorVoteResult:
        """Rank target nodes and apply the historical support/margin gate."""

        import time

        started = time.perf_counter()
        source = self.source_by_id.get(source_node_id)
        if source is None or source.bbox is None:
            return AnchorVoteResult((), None, False, "source_node_unavailable", 0.0)
        allowed_ids = set(candidate_node_ids or ())
        candidates = (
            tuple(
                node
                for node in self.target_graph.nodes
                if node.node_id in allowed_ids
                and node.bbox is not None
                and node.enabled
            )
            if allowed_ids
            else self.target_candidates
        )
        if not candidates:
            return AnchorVoteResult((), None, False, "no_target_candidates", 0.0)
        source_center = _center(source.bbox)
        resolved_point = source_point or source_center
        raw: list[tuple[UINode, float, float, int, tuple[float, float]]] = []
        for target in candidates:
            semantic = self._similarity(source.node_id, target.node_id)
            geometric, anchor_count = (
                self._geometric_log_score(source, target) if use_anchors else (0.0, 0)
            )
            score = semantic / self.semantic_temperature + geometric
            projected = (
                self._predict_point(resolved_point, source, target)
                if use_anchors
                else _proportional_projection(resolved_point, source.bbox, target.bbox)
            )
            raw.append((target, float(score), float(semantic), anchor_count, projected))
        logits = np.asarray([item[1] for item in raw], dtype=np.float64)
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= max(float(np.sum(probabilities)), 1e-12)
        ranked = sorted(
            (
                AnchorVoteCandidate(
                    node_id=item[0].node_id,
                    score=item[1],
                    support=float(probability),
                    semantic_similarity=item[2],
                    geometric_log_score=float(
                        item[1] - item[2] / self.semantic_temperature
                    ),
                    anchor_count=item[3],
                    projected_point=item[4],
                )
                for item, probability in zip(raw, probabilities, strict=True)
            ),
            key=lambda item: (-item.support, item.node_id),
        )
        best = ranked[0]
        second_support = ranked[1].support if len(ranked) > 1 else 0.0
        margin = best.support - second_support
        checks = {
            "support": best.support >= self.min_vote_support,
            "margin": margin >= self.min_vote_margin,
            "action_similarity": best.semantic_similarity >= self.min_action_similarity,
            "anchors": not use_anchors
            or best.anchor_count >= 1
            or best.semantic_similarity >= 0.95,
        }
        failed = [name for name, passed in checks.items() if not passed]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AnchorVoteResult(
            candidates=tuple(ranked),
            selected_node_id=best.node_id,
            execute=not failed,
            reason="ok" if not failed else "low_" + "_".join(failed),
            latency_ms=elapsed_ms,
        )

    def rank_selector(
        self,
        source_node_id: str,
        *,
        source_point: tuple[float, float] | None = None,
        candidate_node_ids: Iterable[str] | None = None,
    ) -> AnchorVoteResult:
        """Run the current exact-identity selector without embeddings or anchors."""

        import time

        started = time.perf_counter()
        source = self.source_by_id.get(source_node_id)
        if source is None or source.bbox is None:
            return AnchorVoteResult((), None, False, "source_node_unavailable", 0.0)
        allowed_ids = set(candidate_node_ids or ())
        candidates = (
            tuple(
                node
                for node in self.target_graph.nodes
                if node.node_id in allowed_ids
                and node.bbox is not None
                and node.enabled
            )
            if allowed_ids
            else self.selector_candidates
        )
        scored = [
            (_identity_selector_score(source, target), target)
            for target in candidates
        ]
        scored = [item for item in scored if item[0] > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1].node_id))
        if not scored:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return AnchorVoteResult(
                (), None, False, "no_identity_candidate", elapsed_ms
            )
        logits = np.asarray([item[0] for item in scored], dtype=np.float64)
        logits -= float(np.max(logits))
        supports = np.exp(logits)
        supports /= max(float(np.sum(supports)), 1e-12)
        point = source_point or _center(source.bbox)
        candidates = tuple(
            AnchorVoteCandidate(
                node_id=target.node_id,
                score=float(score),
                support=float(support),
                semantic_similarity=float(score),
                geometric_log_score=0.0,
                anchor_count=0,
                projected_point=_proportional_projection(
                    point, source.bbox, target.bbox
                ),
            )
            for (score, target), support in zip(scored, supports, strict=True)
        )
        top_score = scored[0][0]
        unique = sum(score == top_score for score, _ in scored) == 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AnchorVoteResult(
            candidates=candidates,
            selected_node_id=candidates[0].node_id if unique else None,
            execute=unique,
            reason="ok" if unique else "target_identity_not_unique",
            latency_ms=elapsed_ms,
        )

    def _geometric_log_score(self, source: UINode, target: UINode) -> tuple[float, int]:
        source_anchors = [
            node for node in self._context(source, source_side=True) if node.node_id != source.node_id
        ]
        target_anchors = [
            node for node in self._context(target, source_side=False) if node.node_id != target.node_id
        ]
        if not source_anchors or not target_anchors:
            return -math.log(max(1, len(self.target_candidates))), 0
        source_anchors.sort(key=lambda node: self._normalized_distance(source, node))
        log_terms: list[float] = []
        weights: list[float] = []
        used_targets: set[str] = set()
        for source_anchor in source_anchors[:40]:
            source_distance = self._normalized_distance(source, source_anchor)
            best: tuple[float, UINode, float, float] | None = None
            for target_anchor in target_anchors:
                if target_anchor.node_id in used_targets:
                    continue
                feature_similarity = self._similarity(
                    source_anchor.node_id, target_anchor.node_id
                )
                if feature_similarity <= 0.0:
                    continue
                relation_distance = self._relation_distance(
                    source_anchor, target_anchor, source, target
                )
                pair_score = (
                    feature_similarity
                    * (1.0 / (1.0 + relation_distance))
                    * max(1e-4, self._reliability(source_anchor))
                    * max(1e-4, self._reliability(target_anchor))
                    * math.exp(-source_distance / 0.35)
                )
                if best is None or pair_score > best[0]:
                    best = (pair_score, target_anchor, feature_similarity, relation_distance)
            if best is None or best[0] < 1e-4:
                continue
            pair_score, target_anchor, feature_similarity, relation_distance = best
            used_targets.add(target_anchor.node_id)
            projected = self._project(
                _center(source.bbox or (0.0, 0.0, 0.0, 0.0)),
                _center(source_anchor.bbox or (0.0, 0.0, 0.0, 0.0)),
                _center(target_anchor.bbox or (0.0, 0.0, 0.0, 0.0)),
            )
            projected_distance = (
                0.0
                if target.bbox and _contains(target.bbox, projected)
                else _point_to_bbox_distance(projected, target.bbox) / self.target_diagonal
            )
            distance_loss = max(
                0.0, projected_distance - self.projection_distance_deadzone
            )
            combined_distance = distance_loss + (
                self.projection_tree_distance_weight * relation_distance
            )
            projection_support = math.exp(
                -(2.0 * combined_distance) / self.geometric_sigma
            )
            vote = max(feature_similarity, 1e-10) * max(projection_support, 1e-10)
            log_terms.append(math.log(max(vote, 1e-10)))
            weights.append(max(1e-4, pair_score))
            if len(log_terms) >= 12:
                break
        if not log_terms:
            return -math.log(max(1, len(self.target_candidates))), 0
        weight_array = np.asarray(weights, dtype=np.float64)
        weight_array /= max(float(np.sum(weight_array)), 1e-12)
        terms = np.asarray(log_terms, dtype=np.float64) + np.log(weight_array + 1e-10)
        maximum = float(np.max(terms))
        score = maximum + math.log(float(np.sum(np.exp(terms - maximum))))
        return float(score), len(log_terms)

    def _predict_point(
        self,
        source_point: tuple[float, float],
        source: UINode,
        target: UINode,
    ) -> tuple[float, float]:
        source_anchors = [
            node
            for node in self._context(source, source_side=True)
            if node.node_id != source.node_id
            and not _bbox_inside(source.bbox, node.bbox)
            and not _bbox_inside(node.bbox, source.bbox)
        ]
        target_anchors = [
            node
            for node in self._context(target, source_side=False)
            if node.node_id != target.node_id
            and not _bbox_inside(target.bbox, node.bbox)
            and not _bbox_inside(node.bbox, target.bbox)
        ]
        source_anchors.sort(key=lambda node: self._normalized_distance(source, node))
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        used_targets: set[str] = set()
        for source_anchor in source_anchors[:40]:
            best: tuple[float, UINode] | None = None
            for target_anchor in target_anchors:
                if target_anchor.node_id in used_targets:
                    continue
                similarity = self._similarity(
                    source_anchor.node_id, target_anchor.node_id
                )
                if similarity <= 0.0:
                    continue
                relation_distance = self._relation_distance(
                    source_anchor, target_anchor, source, target
                )
                pair_score = (
                    similarity
                    * (1.0 / (1.0 + relation_distance))
                    * max(1e-4, self._reliability(source_anchor))
                    * max(1e-4, self._reliability(target_anchor))
                    * math.exp(-self._normalized_distance(source, source_anchor) / 0.35)
                )
                if best is None or pair_score > best[0]:
                    best = (pair_score, target_anchor)
            if best is None or best[0] < 1e-4:
                continue
            weight, target_anchor = best
            used_targets.add(target_anchor.node_id)
            projected_x, projected_y = self._project(
                source_point,
                _center(source_anchor.bbox or source.bbox or (0.0, 0.0, 0.0, 0.0)),
                _center(target_anchor.bbox or target.bbox or (0.0, 0.0, 0.0, 0.0)),
            )
            if target.bbox:
                projected_x = min(max(projected_x, target.bbox[0]), target.bbox[2])
                projected_y = min(max(projected_y, target.bbox[1]), target.bbox[3])
            weighted_x += weight * projected_x
            weighted_y += weight * projected_y
            total_weight += weight
            if len(used_targets) >= 12:
                break
        if total_weight > 1e-10:
            return weighted_x / total_weight, weighted_y / total_weight
        return _proportional_projection(source_point, source.bbox, target.bbox)

    def _context(self, node: UINode, *, source_side: bool) -> tuple[UINode, ...]:
        cache = self._source_context_cache if source_side else self._target_context_cache
        if node.node_id in cache:
            return cache[node.node_id]
        graph = self.source_graph if source_side else self.target_graph
        by_id = self.source_by_id if source_side else self.target_by_id
        selected: list[tuple[float, UINode]] = []
        for anchor in graph.nodes:
            if anchor.bbox is None or self._reliability(anchor) < 0.08:
                continue
            relation = self._relation_signature(anchor, node, by_id)
            tree_distance = relation["tree_distance"]
            spatial = self._normalized_distance(anchor, node)
            if (
                relation["same_element"]
                or relation["anchor_inside_action"]
                or relation["action_inside_anchor"]
                or relation["same_parent"]
                or (tree_distance is not None and tree_distance <= 4)
                or spatial <= 0.35
            ):
                tree_rank = float(tree_distance) if tree_distance is not None else 8.0
                selected.append((tree_rank + spatial * 4.0 - self._reliability(anchor), anchor))
        selected.sort(key=lambda item: (item[0], item[1].node_id))
        cache[node.node_id] = tuple(item[1] for item in selected[:40])
        return cache[node.node_id]

    def _relation_distance(
        self,
        source_anchor: UINode,
        target_anchor: UINode,
        source: UINode,
        target: UINode,
    ) -> float:
        source_relation = self._relation_signature(source_anchor, source, self.source_by_id)
        target_relation = self._relation_signature(target_anchor, target, self.target_by_id)
        if not source_anchor.bbox or not target_anchor.bbox or not source.bbox or not target.bbox:
            geometry_error = 1.0
        else:
            source_anchor_center = _center(source_anchor.bbox)
            target_anchor_center = _center(target_anchor.bbox)
            source_center = _center(source.bbox)
            target_center = _center(target.bbox)
            geometry_error = math.hypot(
                (source_center[0] - source_anchor_center[0]) * self.scale_x
                - (target_center[0] - target_anchor_center[0]),
                (source_center[1] - source_anchor_center[1]) * self.scale_y
                - (target_center[1] - target_anchor_center[1]),
            ) / self.target_diagonal
        source_tree = source_relation["tree_distance"]
        target_tree = target_relation["tree_distance"]
        if source_tree is None and target_tree is None:
            tree_error = 0.0
        elif source_tree is None or target_tree is None:
            tree_error = 1.0
        else:
            tree_error = abs(float(source_tree) - float(target_tree)) / max(
                1.0, float(source_tree), float(target_tree)
            )
        role_keys = (
            "same_element",
            "anchor_inside_action",
            "action_inside_anchor",
            "same_parent",
        )
        role_error = sum(
            bool(source_relation[key]) != bool(target_relation[key]) for key in role_keys
        ) / len(role_keys)
        return float(max(0.0, geometry_error + tree_error + role_error))

    def _relation_signature(
        self,
        anchor: UINode,
        node: UINode,
        by_id: dict[str, UINode],
    ) -> dict[str, bool | int | None]:
        source_side = by_id is self.source_by_id
        cache = self._source_relation_cache if source_side else self._target_relation_cache
        cache_key = (anchor.node_id, node.node_id)
        if cache_key in cache:
            return cache[cache_key]
        relation: dict[str, bool | int | None] = {
            "same_element": anchor.node_id == node.node_id,
            "anchor_inside_action": _bbox_inside(anchor.bbox, node.bbox),
            "action_inside_anchor": _bbox_inside(node.bbox, anchor.bbox),
            "same_parent": bool(anchor.parent_id and anchor.parent_id == node.parent_id),
            "tree_distance": self._tree_distance(
                anchor.node_id,
                node.node_id,
                source_side=source_side,
            ),
        }
        cache[cache_key] = relation
        return relation

    def _similarity(self, source_node_id: str, target_node_id: str) -> float:
        return float(
            self.similarity_matrix[
                self.source_indices[source_node_id], self.target_indices[target_node_id]
            ]
        )

    def _tree_distance(
        self,
        first_node_id: str,
        second_node_id: str,
        *,
        source_side: bool,
    ) -> int | None:
        cache_key = (source_side, first_node_id, second_node_id)
        if cache_key in self._tree_distance_cache:
            return self._tree_distance_cache[cache_key]
        chains = self._source_ancestors if source_side else self._target_ancestors
        first_chain = chains.get(first_node_id, {})
        second_chain = chains.get(second_node_id, {})
        common = set(first_chain).intersection(second_chain)
        if not common:
            self._tree_distance_cache[cache_key] = None
            return None
        distance = min(
            first_chain[node_id] + second_chain[node_id] for node_id in common
        )
        self._tree_distance_cache[cache_key] = distance
        self._tree_distance_cache[(source_side, second_node_id, first_node_id)] = distance
        return distance

    def _normalized_distance(self, first: UINode, second: UINode) -> float:
        if first.bbox is None or second.bbox is None:
            return 1.0
        first_center = _center(first.bbox)
        second_center = _center(second.bbox)
        return math.hypot(
            first_center[0] - second_center[0], first_center[1] - second_center[1]
        ) / self.target_diagonal

    def _reliability(self, node: UINode) -> float:
        cache_key = id(node)
        if cache_key in self._reliability_cache:
            return self._reliability_cache[cache_key]
        ratio = _area(node.bbox) / max(1.0, self.target_diagonal**2 / 2.0)
        if ratio >= 0.75:
            value = 0.05
        elif ratio >= 0.25:
            value = 0.20
        elif ratio >= 0.10:
            value = 0.55
        else:
            value = 1.0
        self._reliability_cache[cache_key] = value
        return value

    def _project(
        self,
        source_point: tuple[float, float],
        source_anchor: tuple[float, float],
        target_anchor: tuple[float, float],
    ) -> tuple[float, float]:
        return (
            target_anchor[0] + (source_point[0] - source_anchor[0]) * self.scale_x,
            target_anchor[1] + (source_point[1] - source_anchor[1]) * self.scale_y,
        )


def _action_candidates(graph: UIGraph) -> list[UINode]:
    page_area = max(1.0, float(graph.width or 1.0) * float(graph.height or 1.0))
    candidates: list[UINode] = []
    for node in graph.nodes:
        if node.bbox is None or _area(node.bbox) <= 0.0:
            continue
        if _area(node.bbox) / page_area >= 0.75:
            continue
        class_lower = _class_tail(node.class_name).lower()
        if (
            node.clickable
            or node.editable
            or node.text
            or node.content_desc
            or node.resource_id
            or any(token in class_lower for token in _ACTION_CLASS_TOKENS)
        ):
            candidates.append(node)
    return candidates


def _identity_selector_score(source: UINode, target: UINode) -> float:
    score = 0.0
    stable_identity = False
    if source.resource_id and source.resource_id == target.resource_id:
        score += 8.0
        stable_identity = True
    elif _selector_tail(source.resource_id) and _selector_tail(
        source.resource_id
    ) == _selector_tail(target.resource_id):
        score += 6.0
        stable_identity = True
    for source_value, target_value in (
        (source.text, target.text),
        (source.content_desc, target.content_desc),
    ):
        if source_value and _normalized_text(source_value) == _normalized_text(
            target_value
        ):
            score += 4.0
            stable_identity = True
    if not stable_identity:
        return 0.0
    if source.class_name and _selector_tail(source.class_name) == _selector_tail(
        target.class_name
    ):
        score += 1.0
    score += 0.5 * sum(
        source_value == target_value
        for source_value, target_value in (
            (source.clickable, target.clickable),
            (source.editable, target.editable),
            (source.scrollable, target.scrollable),
        )
    )
    return score


def _signed_hash(value: str, dimension: int) -> np.ndarray:
    clean_text = re.sub(r"[^\w\u4e00-\u9fff]", "", str(value or "").lower())[:10]
    vector = np.zeros(dimension, dtype=np.float32)
    if not clean_text:
        return vector
    padded = "^" + clean_text + "$"
    for index in range(max(1, len(padded) - 1)):
        digest = hashlib.md5(padded[index : index + 2].encode("utf-8")).digest()
        vector[digest[0] % dimension] += 1.0 if digest[1] % 2 == 0 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else vector


def _node_text(node: UINode) -> str:
    chunks: list[str] = []
    seen: set[str] = set()
    for value in (node.text, node.content_desc, _resource_tail(node.resource_id)):
        normalized = " ".join(str(value or "").split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            chunks.append(normalized)
    return " ".join(chunks)


def _resource_tail(value: str) -> str:
    text = str(value or "").split("/")[-1].split(":")[-1]
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def _selector_tail(value: str) -> str:
    return _normalized_text(value).rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _class_tail(value: str) -> str:
    text = str(value or "").rsplit("}", 1)[-1].rsplit(".", 1)[-1]
    for prefix in ("XCUIElementType", "UIA"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.lower()


def _bucket(value: float, bins: tuple[float, ...]) -> int:
    for index, threshold in enumerate(bins):
        if value <= threshold:
            return index
    return len(bins)


def _parse_bbox(value: str) -> BBox:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
    if len(numbers) != 4 or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        raise ValueError(f"invalid_bbox:{value}")
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _center(bbox: BBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _area(bbox: BBox | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _area_similarity(first: BBox, second: BBox) -> float:
    first_area = _area(first)
    second_area = _area(second)
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0
    return min(first_area, second_area) / max(first_area, second_area)


def _iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _contains(bbox: BBox, point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _bbox_inside(inner: BBox | None, outer: BBox | None) -> bool:
    if inner is None or outer is None:
        return False
    tolerance = 2.0
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _point_to_bbox_distance(point: tuple[float, float], bbox: BBox | None) -> float:
    if bbox is None:
        return math.inf
    dx = max(bbox[0] - point[0], 0.0, point[0] - bbox[2])
    dy = max(bbox[1] - point[1], 0.0, point[1] - bbox[3])
    return math.hypot(dx, dy)


def _ancestor_chains(by_id: dict[str, UINode]) -> dict[str, dict[str, int]]:
    def ancestors(node_id: str) -> dict[str, int]:
        values: dict[str, int] = {}
        cursor = node_id
        depth = 0
        while cursor and cursor not in values:
            values[cursor] = depth
            node = by_id.get(cursor)
            cursor = node.parent_id if node and node.parent_id else ""
            depth += 1
        return values

    return {node_id: ancestors(node_id) for node_id in by_id}


def _proportional_projection(
    point: tuple[float, float],
    source_bbox: BBox | None,
    target_bbox: BBox | None,
) -> tuple[float, float]:
    if source_bbox is None or target_bbox is None:
        return point
    source_width = max(1e-9, source_bbox[2] - source_bbox[0])
    source_height = max(1e-9, source_bbox[3] - source_bbox[1])
    ratio_x = (point[0] - source_bbox[0]) / source_width
    ratio_y = (point[1] - source_bbox[1]) / source_height
    return (
        target_bbox[0] + ratio_x * (target_bbox[2] - target_bbox[0]),
        target_bbox[1] + ratio_y * (target_bbox[3] - target_bbox[1]),
    )


__all__ = [
    "AnchorVote64DMatcher",
    "AnchorVoteCandidate",
    "AnchorVoteResult",
    "BoundNode",
    "PublicWidgetPair",
    "bind_public_bbox",
    "cosine_similarity",
    "encode_graph_64d",
    "load_public_widget_pairs",
    "normalize_public_bbox",
]
