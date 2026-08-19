"""Offline exact-identity baselines for canonical correspondence pairs."""

from __future__ import annotations

import re
from typing import Any, Iterable

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair


def evaluate_exact_identity_baselines(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate text and resource-id equality without geometric model evidence."""

    totals = {
        name: {"rows": 0, "covered": 0, "top1": 0, "top3": 0, "top5": 0}
        for name in ("text_exact", "resource_id_exact", "text_or_resource_exact")
    }
    pair_count = 0
    for raw_record in records:
        record = validate_ui_correspondence_pair(raw_record)
        pair_count += 1
        source_nodes = {
            str(node["node_id"]): node for node in record["source"]["graph"]["nodes"]
        }
        target_nodes = {
            str(node["node_id"]): node for node in record["target"]["graph"]["nodes"]
        }
        forward = {
            str(match["source_node_id"]): {str(value) for value in match["target_node_ids"]}
            for match in record["matches"]
            if match["label"] == "correspondence"
        }
        reverse: dict[str, set[str]] = {}
        for source_id, target_ids in forward.items():
            for target_id in target_ids:
                reverse.setdefault(target_id, set()).add(source_id)
        _evaluate_direction(source_nodes, target_nodes, forward, totals)
        _evaluate_direction(target_nodes, source_nodes, reverse, totals)
    return {
        "schema_version": "omnitransfer.exact_identity_baselines.v1",
        "pair_count": pair_count,
        "direction_count": pair_count * 2,
        "candidate_policy": "all_nodes",
        "identity_only": True,
        "baselines": {
            name: {
                **values,
                "coverage": values["covered"] / values["rows"] if values["rows"] else 0.0,
                "top1_accuracy": values["top1"] / values["rows"] if values["rows"] else 0.0,
                "recall_at_3": values["top3"] / values["rows"] if values["rows"] else 0.0,
                "recall_at_5": values["top5"] / values["rows"] if values["rows"] else 0.0,
            }
            for name, values in totals.items()
        },
    }


def _evaluate_direction(
    source_nodes: dict[str, dict[str, Any]],
    target_nodes: dict[str, dict[str, Any]],
    gold: dict[str, set[str]],
    totals: dict[str, dict[str, int]],
) -> None:
    for source_id, gold_ids in gold.items():
        source = source_nodes[source_id]
        for name in totals:
            ranked = _rank_exact(source, target_nodes, baseline=name)
            totals[name]["rows"] += 1
            totals[name]["covered"] += int(bool(ranked))
            totals[name]["top1"] += int(bool(ranked) and ranked[0] in gold_ids)
            totals[name]["top3"] += int(any(node_id in gold_ids for node_id in ranked[:3]))
            totals[name]["top5"] += int(any(node_id in gold_ids for node_id in ranked[:5]))


def _rank_exact(
    source: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    *,
    baseline: str,
) -> list[str]:
    source_text = _text_values(source)
    source_resource = _resource_value(source)
    ranked = []
    for node_id, candidate in candidates.items():
        text_match = bool(source_text & _text_values(candidate))
        resource_match = bool(
            source_resource and source_resource == _resource_value(candidate)
        )
        matched = {
            "text_exact": text_match,
            "resource_id_exact": resource_match,
            "text_or_resource_exact": text_match or resource_match,
        }[baseline]
        if matched:
            ranked.append(node_id)
    return sorted(ranked)


def _text_values(node: dict[str, Any]) -> set[str]:
    return {
        normalized
        for field in ("text", "content_desc")
        if (normalized := _normalize_text(str(node.get(field) or "")))
    }


def _resource_value(node: dict[str, Any]) -> str:
    return str(node.get("resource_id") or "").strip().casefold()


def _normalize_text(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))
