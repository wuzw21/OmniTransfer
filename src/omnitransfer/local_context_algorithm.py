"""Optional local-context algorithm for diagnosing UI correspondence failures.

This is an offline comparison branch, not a runtime fallback. It deliberately
ignores clickability and never rewrites annotation endpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class LocalContextConfig:
    direct_semantic_weight: float = 5.0
    one_sided_bridge_weight: float = 4.0
    local_context_weight: float = 3.2
    sibling_slot_weight: float = 1.4
    local_geometry_weight: float = 1.1
    class_family_weight: float = 0.7
    conflicting_semantic_penalty: float = 2.2
    spatial_neighbors: int = 4
    temperature: float = 0.35


class LocalContextAlgorithm:
    """Ranks all target nodes from own-node and typed local-context evidence."""

    def __init__(self, config: LocalContextConfig | None = None) -> None:
        self.config = config or LocalContextConfig()

    def rank(
        self,
        source_nodes: list[dict[str, Any]],
        target_nodes: list[dict[str, Any]],
        source_node_id: str,
    ) -> list[dict[str, Any]]:
        source = _GraphContext(source_nodes, self.config.spatial_neighbors)
        target = _GraphContext(target_nodes, self.config.spatial_neighbors)
        pair = _PairEvidence(source, target)
        return self._rank_prepared(source, target, pair, str(source_node_id))

    def rank_many(
        self,
        source_nodes: list[dict[str, Any]],
        target_nodes: list[dict[str, Any]],
        source_node_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Rank several source nodes while sharing one page-context build."""

        source = _GraphContext(source_nodes, self.config.spatial_neighbors)
        target = _GraphContext(target_nodes, self.config.spatial_neighbors)
        pair = _PairEvidence(source, target)
        return {
            str(source_id): self._rank_prepared(source, target, pair, str(source_id))
            for source_id in source_node_ids
        }

    def _rank_prepared(
        self,
        source: "_GraphContext",
        target: "_GraphContext",
        pair: "_PairEvidence",
        source_id: str,
    ) -> list[dict[str, Any]]:
        scored = []
        for candidate_id in target.ids:
            evidence = self.score(source, target, source_id, candidate_id, pair=pair)
            scored.append(
                {
                    "node_id": candidate_id,
                    "score": evidence["score"],
                    "evidence": evidence,
                }
            )
        scored.sort(key=lambda row: (-row["score"], row["node_id"]))
        if scored:
            probabilities = _softmax(
                [row["score"] for row in scored], self.config.temperature
            )
            for row, probability in zip(scored, probabilities, strict=True):
                row["confidence"] = probability
        return scored

    def score(
        self,
        source: "_GraphContext",
        target: "_GraphContext",
        source_id: str,
        target_id: str,
        *,
        pair: "_PairEvidence | None" = None,
    ) -> dict[str, float]:
        pair = pair or _PairEvidence(source, target)
        left = source.nodes[source_id]
        right = target.nodes[target_id]
        left_semantics = _semantics(left)
        right_semantics = _semantics(right)
        direct = pair.semantic(source_id, target_id)
        bridge = _one_sided_bridge(source, target, pair, source_id, target_id)
        role = pair.role(source_id, target_id)
        bridge *= role
        context = _context_similarity(source, target, pair, source_id, target_id)
        sibling = _sibling_slot_similarity(source, target, source_id, target_id)
        geometry = _local_geometry_similarity(source, target, pair, source_id, target_id)
        class_family = role
        conflict = float(bool(left_semantics and right_semantics) and direct < 0.2)
        score = (
            self.config.direct_semantic_weight * direct
            + self.config.one_sided_bridge_weight * bridge
            + self.config.local_context_weight * context
            + self.config.sibling_slot_weight * sibling
            + self.config.local_geometry_weight * geometry
            + self.config.class_family_weight * class_family
            - self.config.conflicting_semantic_penalty * conflict
        )
        return {
            "score": score,
            "direct_semantic": direct,
            "one_sided_bridge": bridge,
            "local_context": context,
            "sibling_slot": sibling,
            "local_geometry": geometry,
            "class_family": class_family,
            "semantic_conflict": conflict,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "omnitransfer.local_context_algorithm.v1",
            "purpose": "offline_diagnostic_branch",
            "candidate_policy": "all_nodes",
            "clickability_used": False,
            "absolute_screen_position_used": False,
            "parent_child_equivalence": False,
            "pooling": "sharp_top_evidence",
            "parameters": asdict(self.config),
        }


class _GraphContext:
    def __init__(self, nodes: list[dict[str, Any]], spatial_neighbors: int) -> None:
        self.nodes = {str(node["node_id"]): node for node in nodes}
        self.ids = tuple(self.nodes)
        self._spatial_neighbors = spatial_neighbors
        self._relations = {
            node_id: self._build_relations(node_id) for node_id in self.ids
        }

    def relations(self, node_id: str) -> tuple[tuple[str, str, float], ...]:
        return self._relations[str(node_id)]

    def _build_relations(self, node_id: str) -> tuple[tuple[str, str, float], ...]:
        node = self.nodes[node_id]
        relations: dict[str, tuple[str, float]] = {}

        parent = str(node.get("parent_id") or "")
        if parent in self.nodes:
            relations[parent] = ("parent", 1.0)
            grandparent = str(self.nodes[parent].get("parent_id") or "")
            if grandparent in self.nodes:
                relations.setdefault(grandparent, ("ancestor", 0.65))
            for sibling_id in self.nodes[parent].get("child_ids") or []:
                sibling_id = str(sibling_id)
                if sibling_id != node_id and sibling_id in self.nodes:
                    relations.setdefault(sibling_id, ("sibling", 1.0))

        for child_id in node.get("child_ids") or []:
            child_id = str(child_id)
            if child_id not in self.nodes:
                continue
            relations[child_id] = ("child", 1.0)
            for grandchild_id in self.nodes[child_id].get("child_ids") or []:
                grandchild_id = str(grandchild_id)
                if grandchild_id in self.nodes:
                    relations.setdefault(grandchild_id, ("descendant", 0.65))

        center = _center(node)
        if center is not None:
            nearest = []
            for other_id, other in self.nodes.items():
                if other_id == node_id or other_id in relations:
                    continue
                other_center = _center(other)
                if other_center is None:
                    continue
                distance = math.dist(center, other_center)
                nearest.append((distance, other_id, _direction(center, other_center)))
            for distance, other_id, direction in sorted(nearest)[: self._spatial_neighbors]:
                relations.setdefault(other_id, (f"near_{direction}", 1.0 / (1.0 + 4.0 * distance)))
        return tuple(
            (other_id, relation, strength)
            for other_id, (relation, strength) in relations.items()
        )


class _PairEvidence:
    """Caches expensive node-pair evidence once for a page pair."""

    def __init__(self, source: _GraphContext, target: _GraphContext) -> None:
        self._source = source
        self._target = target
        self._semantic: dict[tuple[str, str], float] = {}
        self._role: dict[tuple[str, str], float] = {}
        self._shape: dict[tuple[str, str], float] = {}

    def semantic(self, source_id: str, target_id: str) -> float:
        key = (source_id, target_id)
        if key not in self._semantic:
            self._semantic[key] = _set_similarity(
                _semantics(self._source.nodes[source_id]),
                _semantics(self._target.nodes[target_id]),
            )
        return self._semantic[key]

    def role(self, source_id: str, target_id: str) -> float:
        key = (source_id, target_id)
        if key not in self._role:
            self._role[key] = _role_compatibility(
                self._source.nodes[source_id], self._target.nodes[target_id]
            )
        return self._role[key]

    def shape(self, source_id: str, target_id: str) -> float:
        key = (source_id, target_id)
        if key not in self._shape:
            self._shape[key] = _shape_similarity(
                self._source.nodes[source_id], self._target.nodes[target_id]
            )
        return self._shape[key]


def _one_sided_bridge(
    source: _GraphContext,
    target: _GraphContext,
    pair: _PairEvidence,
    source_id: str,
    target_id: str,
) -> float:
    left = _semantics(source.nodes[source_id])
    right = _semantics(target.nodes[target_id])
    if bool(left) == bool(right):
        return 0.0
    if left:
        return max(
            strength * pair.semantic(source_id, neighbor_id)
            for neighbor_id, _, strength in target.relations(target_id)
        ) if target.relations(target_id) else 0.0
    return max(
        strength * pair.semantic(neighbor_id, target_id)
        for neighbor_id, _, strength in source.relations(source_id)
    ) if source.relations(source_id) else 0.0


def _context_similarity(
    source: _GraphContext,
    target: _GraphContext,
    pair: _PairEvidence,
    source_id: str,
    target_id: str,
) -> float:
    left = source.relations(source_id)
    right = target.relations(target_id)
    if not left or not right:
        return 0.0
    evidence = []
    for left_id, left_relation, left_strength in left:
        for right_id, right_relation, right_strength in right:
            relation = _relation_compatibility(left_relation, right_relation)
            if relation <= 0.0:
                continue
            semantic = pair.semantic(left_id, right_id)
            node = semantic if semantic > 0.0 else (
                0.35 * pair.role(left_id, right_id)
                + 0.15 * pair.shape(left_id, right_id)
            )
            evidence.append(relation * min(left_strength, right_strength) * node)
    return _sharp_pool(evidence)


def _relation_compatibility(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_group = left.split("_", 1)[0]
    right_group = right.split("_", 1)[0]
    if left_group == right_group == "near":
        return 0.65
    upward = {"parent", "ancestor"}
    downward = {"child", "descendant"}
    if left in upward and right in upward:
        return 0.7
    if left in downward and right in downward:
        return 0.7
    if {left, right} <= {"sibling", "near_left", "near_right", "near_up", "near_down"}:
        return 0.35
    return 0.08


def _sharp_pool(values: list[float]) -> float:
    if not values:
        return 0.0
    ranked = sorted(values, reverse=True)
    return min(1.0, ranked[0] + (0.3 * ranked[1] if len(ranked) > 1 else 0.0))


def _sibling_slot_similarity(
    source: _GraphContext,
    target: _GraphContext,
    source_id: str,
    target_id: str,
) -> float:
    left = _sibling_slot(source, source_id)
    right = _sibling_slot(target, target_id)
    if left is None or right is None:
        return 0.0
    return max(0.0, 1.0 - abs(left - right))


def _sibling_slot(graph: _GraphContext, node_id: str) -> float | None:
    node = graph.nodes[node_id]
    parent_id = str(node.get("parent_id") or "")
    if parent_id not in graph.nodes:
        return None
    siblings = [
        str(value)
        for value in graph.nodes[parent_id].get("child_ids") or []
        if str(value) in graph.nodes
    ]
    if len(siblings) <= 1 or node_id not in siblings:
        return None
    siblings.sort(key=lambda value: (_reading_order(graph.nodes[value]), value))
    return siblings.index(node_id) / (len(siblings) - 1)


def _local_geometry_similarity(
    source: _GraphContext,
    target: _GraphContext,
    pair: _PairEvidence,
    source_id: str,
    target_id: str,
) -> float:
    left = _local_box(source, source_id)
    right = _local_box(target, target_id)
    if left is None or right is None:
        return pair.shape(source_id, target_id)
    deltas = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return math.exp(-2.5 * sum(deltas) / len(deltas))


def _local_box(graph: _GraphContext, node_id: str) -> tuple[float, float, float, float] | None:
    node = graph.nodes[node_id]
    box = _box(node)
    parent_id = str(node.get("parent_id") or "")
    parent = _box(graph.nodes.get(parent_id)) if parent_id in graph.nodes else None
    if box is None or parent is None:
        return None
    px1, py1, px2, py2 = parent
    pw, ph = px2 - px1, py2 - py1
    if pw <= 1e-6 or ph <= 1e-6:
        return None
    x1, y1, x2, y2 = box
    return ((x1 - px1) / pw, (y1 - py1) / ph, (x2 - x1) / pw, (y2 - y1) / ph)


def _shape_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_box, right_box = _box(left), _box(right)
    if left_box is None or right_box is None:
        return 0.0
    lw, lh = left_box[2] - left_box[0], left_box[3] - left_box[1]
    rw, rh = right_box[2] - right_box[0], right_box[3] - right_box[1]
    if min(lw, lh, rw, rh) <= 1e-6:
        return 0.0
    return math.exp(-abs(math.log(lw / rw)) - abs(math.log(lh / rh)))


def _semantics(node: Mapping[str, Any]) -> set[str]:
    return {
        value
        for field in ("text", "content_desc")
        if (value := _normalize(str(node.get(field) or "")))
    }


def _set_similarity(left: set[str], right: set[str]) -> float:
    return max((_text_similarity(a, b) for a in left for b in right), default=0.0)


def _text_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence_score = SequenceMatcher(None, left, right).ratio()
    return max(token_score, sequence_score * 0.85)


def _normalize(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _class_family(node: Mapping[str, Any]) -> str:
    value = str(node.get("class_name") or "").casefold()
    if any(token in value for token in ("image", "icon")):
        return "image"
    if any(token in value for token in ("text", "label", "statictext")):
        return "text"
    if any(token in value for token in ("edit", "textfield", "searchfield")):
        return "input"
    if "button" in value:
        return "button"
    if any(token in value for token in ("list", "recycler", "collection", "table")):
        return "collection"
    if any(token in value for token in ("layout", "group", "other", "view")):
        return "container"
    return value.rsplit(".", 1)[-1] or "unknown"


def _role_compatibility(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_family, right_family = _class_family(left), _class_family(right)
    if left_family == right_family:
        return 1.0
    pair = {left_family, right_family}
    if pair <= {"image", "button"}:
        return 0.85
    if pair <= {"input", "container"} or pair <= {"button", "container"}:
        return 0.6
    if pair == {"image", "container"}:
        return 0.55
    if "unknown" in pair:
        return 0.5
    if pair == {"image", "text"}:
        return 0.15
    return 0.3


def _box(node: Mapping[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not node:
        return None
    value = node.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return box  # type: ignore[return-value]


def _center(node: Mapping[str, Any]) -> tuple[float, float] | None:
    box = _box(node)
    if box is None:
        return None
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _reading_order(node: Mapping[str, Any]) -> tuple[float, float]:
    center = _center(node)
    return (center[1], center[0]) if center is not None else (math.inf, math.inf)


def _direction(left: tuple[float, float], right: tuple[float, float]) -> str:
    dx, dy = right[0] - left[0], right[1] - left[1]
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def _softmax(values: list[float], temperature: float) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp((value - maximum) / max(temperature, 1e-6)) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


__all__ = ["LocalContextAlgorithm", "LocalContextConfig"]
