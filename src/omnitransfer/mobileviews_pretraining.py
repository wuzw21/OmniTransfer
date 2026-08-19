"""Build auditable MobileViews pages for unified self-supervised matching.

MobileViews does not provide Android-to-iOS correspondence gold.  This module
therefore never invents cross-page labels.  It selects structurally useful
Android pages, emits an identity correspondence between two declared views of
the same observation, and leaves all corruption to the canonical stochastic
two-view trainer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from typing import Any, Iterable

from omnitransfer.mapping_dataset import UI_CORRESPONDENCE_PAIR_SCHEMA


_LIST_MARKERS = ("list", "recycler", "scroll", "collection", "table")
_TRANSIENT_MARKERS = (
    "dialog",
    "popup",
    "modal",
    "overlay",
    "bottomsheet",
    "bottom_sheet",
    "permission",
)


def mobileviews_page_slices(graph: dict[str, Any]) -> dict[str, Any]:
    """Describe useful local-relation supervision without using clickability."""

    nodes = list(graph.get("nodes") or [])
    by_id = {str(node.get("node_id")): node for node in nodes}
    children: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is not None:
            children.setdefault(str(parent), []).append(node)

    def semantic(node: dict[str, Any]) -> bool:
        return bool(str(node.get("text") or "").strip() or str(node.get("content_desc") or "").strip())

    textless_local = 0
    for node in nodes:
        if semantic(node):
            continue
        related: list[dict[str, Any]] = []
        parent_id = node.get("parent_id")
        if parent_id is not None:
            parent = by_id.get(str(parent_id))
            if parent:
                related.append(parent)
            related.extend(children.get(str(parent_id), ()))
        related.extend(children.get(str(node.get("node_id")), ()))
        if any(semantic(value) for value in related if value is not node):
            textless_local += 1

    sibling_groups = sum(len(group) >= 2 for group in children.values())
    searchable = " ".join(
        str(value or "").casefold()
        for node in nodes
        for value in (
            node.get("class_name"),
            node.get("resource_id"),
            node.get("text"),
            node.get("content_desc"),
        )
    )
    max_depth = max((int(node.get("depth") or 0) for node in nodes), default=0)
    return {
        "semantic_nodes": sum(semantic(node) for node in nodes),
        "textless_with_local_semantics": textless_local,
        "same_parent_sibling_hard_negative_groups": sibling_groups,
        "list_structure": any(marker in searchable for marker in _LIST_MARKERS),
        "popup_or_dialog": any(marker in searchable for marker in _TRANSIENT_MARKERS),
        "deep_hierarchy": max_depth >= 8,
        "max_depth": max_depth,
        "node_count": len(nodes),
        "clickability_used_for_selection": False,
    }


def select_mobileviews_pages(
    records: Iterable[dict[str, Any]],
    *,
    maximum_pages: int,
    per_package_cap: int = 4,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, package-diverse structural pretraining corpus."""

    if maximum_pages <= 0:
        raise ValueError("maximum_pages must be positive")
    if per_package_cap <= 0:
        raise ValueError("per_package_cap must be positive")
    candidates: list[tuple[tuple[int, int, int, int], str, dict[str, Any], dict[str, Any]]] = []
    scanned = 0
    for record in records:
        scanned += 1
        metadata = record.get("metadata") or {}
        package = str(metadata.get("package") or metadata.get("app") or "").strip()
        if not package or not list(record.get("nodes") or []):
            continue
        slices = mobileviews_page_slices(record)
        if slices["semantic_nodes"] == 0 and slices["textless_with_local_semantics"] == 0:
            continue
        graph_id = str(record.get("graph_id") or "")
        digest = hashlib.blake2b(
            f"{seed}:{package}:{graph_id}".encode(), digest_size=8
        ).hexdigest()
        priority = (
            int(bool(slices["textless_with_local_semantics"])),
            int(slices["popup_or_dialog"] or slices["list_structure"]),
            int(bool(slices["same_parent_sibling_hard_negative_groups"])),
            int(slices["deep_hierarchy"]),
        )
        candidates.append((priority, digest, record, slices))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    package_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    slice_counts: Counter[str] = Counter()
    for _, _, record, slices in candidates:
        package = str((record.get("metadata") or {}).get("package") or (record.get("metadata") or {}).get("app"))
        if package_counts[package] >= per_package_cap:
            continue
        selected.append(record)
        package_counts[package] += 1
        for name in (
            "textless_with_local_semantics",
            "same_parent_sibling_hard_negative_groups",
            "list_structure",
            "popup_or_dialog",
            "deep_hierarchy",
        ):
            if slices[name]:
                slice_counts[name] += 1
        if len(selected) >= maximum_pages:
            break
    manifest = {
        "schema_version": "omnitransfer.mobileviews_pretraining_selection.v1",
        "scanned_pages": scanned,
        "eligible_pages": len(candidates),
        "selected_pages": len(selected),
        "packages": len(package_counts),
        "per_package_cap": per_package_cap,
        "seed": seed,
        "slice_page_counts": dict(sorted(slice_counts.items())),
        "selection_uses_clickability": False,
        "label_policy": "same_observation_identity_before_canonical_two_view_augmentation",
    }
    return selected, manifest


def mobileviews_self_supervised_pair(graph: dict[str, Any]) -> dict[str, Any]:
    """Emit one leakage-safe identity pair for canonical two-view augmentation."""

    graph_id = str(graph.get("graph_id") or "").strip()
    if not graph_id:
        raise ValueError("MobileViews graph_id is absent")
    nodes = list(graph.get("nodes") or [])
    if not nodes:
        raise ValueError("MobileViews graph has no nodes")
    metadata = dict(graph.get("metadata") or {})
    package = str(metadata.get("package") or metadata.get("app") or "").strip()
    if not package:
        raise ValueError("MobileViews package is absent")
    screenshot = str(metadata.get("screenshot_path") or "")
    slices = mobileviews_page_slices(graph)
    source_graph = deepcopy(graph)
    target_graph = deepcopy(graph)
    source_graph["graph_id"] = f"{graph_id}:view:a"
    target_graph["graph_id"] = f"{graph_id}:view:b"
    return {
        "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "pair_id": f"mobileviews-self:{hashlib.blake2b(graph_id.encode(), digest_size=10).hexdigest()}",
        "split": "train",
        "label_status": "self_supervised",
        "source": {
            "page_id": source_graph["graph_id"],
            "platform": "android",
            "screenshot_path": screenshot,
            "graph": source_graph,
        },
        "target": {
            "page_id": target_graph["graph_id"],
            "platform": "android",
            "screenshot_path": screenshot,
            "graph": target_graph,
        },
        "matches": [
            {
                "source_node_id": str(node["node_id"]),
                "target_node_ids": [str(node["node_id"])],
                "label": "correspondence",
            }
            for node in nodes
        ],
        "partition_keys": [f"mobileviews:package:{package}"],
        "provenance": {
            "dataset": "MobileViews_Screenshots_ViewHierarchies",
            "package": package,
            "source_graph_id": graph_id,
            "annotation": "same_observation_identity_before_augmentation",
            "forbidden_model_inputs": [
                "origin_id",
                "resource_id",
                "package",
                "graph_id",
                "clickability_as_identity",
            ],
            "cross_platform_gold": False,
        },
        "slices": {"app": package, "platform": "android", **slices},
    }


def mobileviews_state_transition_pairs(
    records: Iterable[dict[str, Any]],
    *,
    maximum_pairs: int,
    per_package_cap: int = 4,
    maximum_row_gap: int = 4,
    minimum_matches: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build adjacent-state pseudo pairs without exposing identity to the model."""

    if maximum_pairs <= 0 or per_package_cap <= 0:
        raise ValueError("pair and package limits must be positive")
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    package_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    scanned = 0
    candidates = 0
    for graph in records:
        scanned += 1
        metadata = graph.get("metadata") or {}
        package = str(metadata.get("package") or metadata.get("app") or "").strip()
        activity = str(metadata.get("foreground_activity") or "").strip()
        if not package or not activity or not list(graph.get("nodes") or []):
            continue
        key = (package, activity)
        left = previous.get(key)
        previous[key] = graph
        if left is None or package_counts[package] >= per_package_cap:
            continue
        left_row = int((left.get("metadata") or {}).get("source_row") or -1)
        right_row = int(metadata.get("source_row") or -1)
        if left_row < 0 or right_row <= left_row or right_row - left_row > maximum_row_gap:
            continue
        pair = _mobileviews_state_pair(left, graph, minimum_matches=minimum_matches)
        if pair is None:
            continue
        candidates += 1
        selected.append(pair)
        package_counts[package] += 1
        reason_counts.update(pair["provenance"]["label_evidence_counts"])
        if len(selected) >= maximum_pairs:
            break
    manifest = {
        "schema_version": "omnitransfer.mobileviews_state_selection.v1",
        "scanned_pages": scanned,
        "candidate_pairs": candidates,
        "selected_pairs": len(selected),
        "packages": len(package_counts),
        "per_package_cap": per_package_cap,
        "maximum_row_gap": maximum_row_gap,
        "minimum_matches": minimum_matches,
        "label_evidence_counts": dict(sorted(reason_counts.items())),
        "selection_uses_clickability": False,
        "model_receives_label_identity_fields": False,
        "label_policy": "unique_cross_state_identity_keys_for_pseudo_labels_only",
    }
    return selected, manifest


def _mobileviews_state_pair(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    minimum_matches: int,
) -> dict[str, Any] | None:
    if _state_signature(source) == _state_signature(target):
        return None
    source_by_key = _unique_identity_keys(source)
    target_by_key = _unique_identity_keys(target)
    shared = sorted(source_by_key.keys() & target_by_key.keys())
    matches = []
    evidence: Counter[str] = Counter()
    used_source: set[str] = set()
    used_target: set[str] = set()
    for key in shared:
        source_node = source_by_key[key]
        target_node = target_by_key[key]
        source_id = str(source_node["node_id"])
        target_id = str(target_node["node_id"])
        if source_id in used_source or target_id in used_target:
            continue
        used_source.add(source_id)
        used_target.add(target_id)
        evidence[key[0]] += 1
        matches.append(
            {
                "source_node_id": source_id,
                "target_node_ids": [target_id],
                "label": "correspondence",
                "quality_slices": ["mobileviews_adjacent_state_pseudo"],
            }
        )
    if len(matches) < minimum_matches:
        return None
    source_meta = dict(source.get("metadata") or {})
    target_meta = dict(target.get("metadata") or {})
    package = str(source_meta.get("package") or source_meta.get("app"))
    source_id = str(source["graph_id"])
    target_id = str(target["graph_id"])
    pair_digest = hashlib.blake2b(f"{source_id}->{target_id}".encode(), digest_size=10).hexdigest()
    source_slices = mobileviews_page_slices(source)
    target_slices = mobileviews_page_slices(target)
    return {
        "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "pair_id": f"mobileviews-state:{pair_digest}",
        "split": "train",
        "label_status": "unreviewed",
        "source": {
            "page_id": source_id,
            "platform": "android",
            "screenshot_path": str(source_meta.get("screenshot_path") or ""),
            "graph": deepcopy(source),
        },
        "target": {
            "page_id": target_id,
            "platform": "android",
            "screenshot_path": str(target_meta.get("screenshot_path") or ""),
            "graph": deepcopy(target),
        },
        "matches": matches,
        "partition_keys": [f"mobileviews:package:{package}"],
        "provenance": {
            "dataset": "MobileViews_Screenshots_ViewHierarchies",
            "package": package,
            "annotation": "adjacent_state_unique_identity_pseudo",
            "label_evidence_counts": dict(sorted(evidence.items())),
            "identity_fields_are_label_only": True,
            "forbidden_model_inputs": [
                "origin_id", "resource_id", "package", "graph_id", "source_row",
                "clickability_as_identity",
            ],
            "cross_platform_gold": False,
        },
        "slices": {
            "app": package,
            "platform": "android",
            "state_transition": True,
            "node_count_delta": abs(len(source.get("nodes") or []) - len(target.get("nodes") or [])),
            "list_structure": bool(source_slices["list_structure"] or target_slices["list_structure"]),
            "popup_or_dialog": bool(source_slices["popup_or_dialog"] or target_slices["popup_or_dialog"]),
        },
    }


def _unique_identity_keys(graph: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in graph.get("nodes") or []:
        class_name = str(node.get("class_name") or "").casefold().rsplit(".", 1)[-1]
        resource_id = str(node.get("resource_id") or "").strip().casefold()
        semantics = " ".join(
            value for value in (
                str(node.get("text") or "").strip().casefold(),
                str(node.get("content_desc") or "").strip().casefold(),
            ) if value
        )
        if resource_id:
            grouped[("resource_id_class", resource_id, class_name)].append(node)
        elif semantics:
            grouped[("semantic_class", semantics, class_name)].append(node)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def _state_signature(graph: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Detect repeated XML states while deliberately ignoring clickability."""

    return tuple(
        sorted(
            (
                str(node.get("node_id") or ""),
                str(node.get("parent_id") or ""),
                str(node.get("class_name") or ""),
                str(node.get("resource_id") or ""),
                str(node.get("text") or ""),
                str(node.get("content_desc") or ""),
                tuple(node.get("bbox") or ()),
            )
            for node in graph.get("nodes") or []
        )
    )
