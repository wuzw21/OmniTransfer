"""Conservative cleaning and slice annotation for correspondence labels."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from copy import deepcopy
import re
from typing import Any, Iterable, Mapping

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair


CLEANING_SCHEMA_VERSION = "omnitransfer.mapping_data_cleaning.v1"


@dataclass(frozen=True)
class CleaningResult:
    cleaned: tuple[dict[str, Any], ...]
    quarantine: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def clean_ui_correspondence_records(
    records: Iterable[dict[str, Any]],
    *,
    quarantine_conflicts: bool = True,
) -> CleaningResult:
    """Annotate difficult slices and optionally quarantine label conflicts.

    Correspondence endpoints are never rewritten. Clickability is deliberately
    absent from every decision: any node may provide local structural evidence.
    """

    cleaned_records: list[dict[str, Any]] = []
    quarantine_records: list[dict[str, Any]] = []
    slice_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    input_pairs = 0
    input_matches = 0
    for raw in records:
        record = validate_ui_correspondence_pair(raw)
        input_pairs += 1
        input_matches += len(record["matches"])
        source_nodes = _node_map(record["source"])
        target_nodes = _node_map(record["target"])
        pair_slices = _pair_slices(record, source_nodes, target_nodes)
        retained: list[dict[str, Any]] = []
        for match in record["matches"]:
            if match["label"] != "correspondence":
                retained.append(match)
                continue
            cleaned, reason = _clean_match(
                match,
                source_nodes=source_nodes,
                target_nodes=target_nodes,
                pair_slices=pair_slices,
                quarantine_conflicts=quarantine_conflicts,
            )
            if reason:
                reason_counts[reason] += 1
                quarantine_records.append(_quarantine_record(record, match, reason))
                continue
            slice_counts.update(cleaned["quality_slices"])
            retained.append(cleaned)
        if retained:
            cleaned_record = deepcopy(record)
            cleaned_record["matches"] = retained
            cleaned_record.setdefault("provenance", {})["cleaning"] = {
                "schema_version": CLEANING_SCHEMA_VERSION,
                "clickability_used": False,
                "original_match_count": len(record["matches"]),
                "retained_match_count": len(retained),
            }
            cleaned_records.append(validate_ui_correspondence_pair(cleaned_record))
    manifest = {
        "schema_version": CLEANING_SCHEMA_VERSION,
        "input_pairs": input_pairs,
        "input_matches": input_matches,
        "cleaned_pairs": len(cleaned_records),
        "cleaned_matches": sum(len(record["matches"]) for record in cleaned_records),
        "quarantined_matches": len(quarantine_records),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "slice_counts": dict(sorted(slice_counts.items())),
        "clickability_used_for_cleaning": False,
        "correspondence_endpoints_changed": False,
        "parent_child_equivalence": "disabled",
        "source_data_mutated": False,
        "policy": {
            "semantic_unit_expansion": "disabled",
            "semantic_conflict": "quarantine_exact_match_outside_gold_family",
            "formal_eval": "never_relabel_in_place",
        },
    }
    return CleaningResult(
        cleaned=tuple(cleaned_records),
        quarantine=tuple(quarantine_records),
        manifest=manifest,
    )


def _clean_match(
    match: Mapping[str, Any],
    *,
    source_nodes: Mapping[str, Mapping[str, Any]],
    target_nodes: Mapping[str, Mapping[str, Any]],
    pair_slices: set[str],
    quarantine_conflicts: bool,
) -> tuple[dict[str, Any], str | None]:
    source_id = str(match["source_node_id"])
    original_targets = [str(value) for value in match["target_node_ids"]]
    source_family = _family(source_nodes, source_id)
    target_family = set().union(
        *(_family(target_nodes, target_id) for target_id in original_targets)
    )
    direct_source_semantics = _semantic_values(source_nodes.get(source_id))
    gold_semantics = set().union(
        *(_semantic_values(target_nodes.get(target_id)) for target_id in original_targets)
    )
    exact_anywhere = {
        node_id
        for node_id, node in target_nodes.items()
        if direct_source_semantics & _semantic_values(node)
    }
    exact_in_gold_family = exact_anywhere & target_family
    if (
        quarantine_conflicts
        and direct_source_semantics
        and exact_anywhere
        and not exact_in_gold_family
    ):
        return dict(match), "exact_semantic_conflict_outside_gold_family"

    slices = set(pair_slices)
    source_has_semantics = bool(direct_source_semantics)
    target_has_semantics = bool(gold_semantics)
    if source_has_semantics != target_has_semantics:
        slices.add("one_sided_semantics")
    if not source_has_semantics and not target_has_semantics:
        source_context = _family_semantics(source_nodes, source_family - {source_id})
        target_context = _family_semantics(target_nodes, target_family - set(original_targets))
        if source_context and target_context:
            slices.add("textless_with_local_semantics")
    if _has_sibling_hard_negative(target_nodes, original_targets):
        slices.add("same_parent_sibling_hard_negative")
    cleaned = deepcopy(dict(match))
    cleaned["quality_slices"] = sorted(slices)
    cleaned["cleaning_schema_version"] = CLEANING_SCHEMA_VERSION
    return cleaned, None


def _pair_slices(
    record: Mapping[str, Any],
    source_nodes: Mapping[str, Mapping[str, Any]],
    target_nodes: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    slices: set[str] = set()
    source_count = len(source_nodes)
    target_count = len(target_nodes)
    ratio = max(source_count, target_count) / max(1, min(source_count, target_count))
    source_depth = max((_depth(node) for node in source_nodes.values()), default=0)
    target_depth = max((_depth(node) for node in target_nodes.values()), default=0)
    source_transient = _has_transient_layer(source_nodes.values())
    target_transient = _has_transient_layer(target_nodes.values())
    if ratio >= 1.8 or abs(source_depth - target_depth) >= 4:
        slices.add("xml_hierarchy_drift")
    if source_transient != target_transient:
        slices.add("popup_or_dialog_change")
    if _has_list_structure(source_nodes.values()) or _has_list_structure(target_nodes.values()):
        slices.add("list_structure")
    return slices


def _node_map(page: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(node["node_id"]): node
        for node in (page.get("graph") or {}).get("nodes", [])
    }


def _family(
    nodes: Mapping[str, Mapping[str, Any]], node_id: str, *, max_hops: int = 2
) -> set[str]:
    family: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(str(node_id), 0)])
    while queue:
        current, distance = queue.popleft()
        if current in family or current not in nodes or distance > max_hops:
            continue
        family.add(current)
        if distance == max_hops:
            continue
        node = nodes[current]
        parent = str(node.get("parent_id") or "")
        if parent:
            queue.append((parent, distance + 1))
        for child_id in node.get("child_ids") or []:
            queue.append((str(child_id), distance + 1))
        if parent:
            for sibling_id, sibling in nodes.items():
                if str(sibling.get("parent_id") or "") == parent:
                    queue.append((sibling_id, distance + 1))
    return family


def _semantic_values(node: Mapping[str, Any] | None) -> set[str]:
    if not node:
        return set()
    return {
        normalized
        for field in ("text", "content_desc")
        if (normalized := _normalize(str(node.get(field) or "")))
    }


def _family_semantics(
    nodes: Mapping[str, Mapping[str, Any]], node_ids: set[str]
) -> set[str]:
    return set().union(*(_semantic_values(nodes.get(node_id)) for node_id in node_ids)) if node_ids else set()


def _has_sibling_hard_negative(
    nodes: Mapping[str, Mapping[str, Any]], target_ids: list[str]
) -> bool:
    selected = set(target_ids)
    for target_id in target_ids:
        target = nodes.get(target_id) or {}
        parent = str(target.get("parent_id") or "")
        if not parent:
            continue
        target_semantics = _semantic_values(target)
        target_class = _class_family(target)
        for node_id, sibling in nodes.items():
            if node_id in selected or str(sibling.get("parent_id") or "") != parent:
                continue
            sibling_semantics = _semantic_values(sibling)
            comparable_semantics = bool(target_semantics) == bool(sibling_semantics)
            if (
                comparable_semantics
                and target_class == _class_family(sibling)
                and _bbox_size_similarity(target, sibling) >= 0.45
            ):
                return True
    return False


def _class_family(node: Mapping[str, Any]) -> str:
    value = str(node.get("class_name") or "").casefold()
    for family, tokens in (
        ("image", ("image", "icon")),
        ("text", ("text", "label", "statictext")),
        ("input", ("edit", "textfield", "searchfield")),
        ("button", ("button",)),
        ("container", ("layout", "group", "other", "view")),
    ):
        if any(token in value for token in tokens):
            return family
    return value.rsplit(".", 1)[-1]


def _bbox_size_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    boxes = []
    for node in (left, right):
        value = node.get("bbox")
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return 0.0
        width = max(0.0, float(value[2]) - float(value[0]))
        height = max(0.0, float(value[3]) - float(value[1]))
        if width <= 0.0 or height <= 0.0:
            return 0.0
        boxes.append((width, height))
    return min(boxes[0][0], boxes[1][0]) / max(boxes[0][0], boxes[1][0]) * (
        min(boxes[0][1], boxes[1][1]) / max(boxes[0][1], boxes[1][1])
    )


def _has_transient_layer(nodes: Iterable[Mapping[str, Any]]) -> bool:
    tokens = ("dialog", "popup", "modal", "permission", "alert", "advert", "overlay")
    return any(
        any(token in " ".join(str(node.get(field) or "").casefold() for field in ("class_name", "resource_id", "text", "content_desc")) for token in tokens)
        for node in nodes
    )


def _has_list_structure(nodes: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        bool(node.get("scrollable"))
        or any(token in str(node.get("class_name") or "").casefold() for token in ("list", "recycler", "collection", "table"))
        for node in nodes
    )


def _depth(node: Mapping[str, Any]) -> int:
    try:
        return int(node.get("depth") or str(node.get("node_id") or "").count("."))
    except (TypeError, ValueError):
        return 0


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _quarantine_record(
    record: Mapping[str, Any], match: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    quarantined = deepcopy(dict(record))
    source_nodes = _node_map(record["source"])
    target_nodes = _node_map(record["target"])
    source_semantics = _semantic_values(source_nodes.get(str(match["source_node_id"])))
    conflicting_target_ids = sorted(
        node_id
        for node_id, node in target_nodes.items()
        if source_semantics & _semantic_values(node)
        and node_id not in {str(value) for value in match["target_node_ids"]}
    )
    quarantined["pair_id"] = f"quarantine:{record['pair_id']}:{match['source_node_id']}"
    quarantined["split"] = "diagnostic"
    quarantined["label_status"] = "unreviewed"
    quarantined["matches"] = [deepcopy(dict(match))]
    quarantined.setdefault("provenance", {}).update(
        {
            "cleaning_schema_version": CLEANING_SCHEMA_VERSION,
            "cleaning_reason": reason,
            "original_pair_id": record["pair_id"],
            "cleaning_conflicting_target_node_ids": conflicting_target_ids,
        }
    )
    quarantined.setdefault("slices", {})["cleaning_quarantine"] = reason
    return validate_ui_correspondence_pair(quarantined)


__all__ = [
    "CLEANING_SCHEMA_VERSION",
    "CleaningResult",
    "clean_ui_correspondence_records",
]
