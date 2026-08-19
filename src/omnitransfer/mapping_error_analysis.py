"""Evidence-first diagnostics for UI correspondence prediction errors."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def node_semantic(node: Mapping[str, Any] | None) -> str:
    if not node:
        return ""
    return " ".join(
        str(node.get(field) or "").strip()
        for field in ("text", "content_desc")
        if str(node.get(field) or "").strip()
    ).strip().casefold()


def availability_slice(source: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    source_has_text = bool(node_semantic(source))
    target_has_text = bool(node_semantic(target))
    if source_has_text and target_has_text:
        return "both_text"
    if source_has_text or target_has_text:
        return "mixed"
    return "both_textless"


def _ancestors(nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> set[str]:
    ancestors: set[str] = set()
    current = str(node_id)
    while current and current not in ancestors:
        ancestors.add(current)
        node = nodes.get(current)
        parent = node.get("parent_id") if node else None
        current = str(parent) if parent is not None else ""
    return ancestors


def structural_relation(
    predicted_id: str,
    gold_ids: Iterable[str],
    nodes: Mapping[str, Mapping[str, Any]],
) -> str:
    gold_ids = tuple(str(value) for value in gold_ids)
    if predicted_id in gold_ids:
        return "same_node"
    predicted = nodes.get(str(predicted_id), {})
    for gold_id in gold_ids:
        gold = nodes.get(gold_id, {})
        if gold_id in _ancestors(nodes, predicted_id):
            return "prediction_descendant_of_gold"
        if predicted_id in _ancestors(nodes, gold_id):
            return "prediction_ancestor_of_gold"
        if predicted and gold and predicted.get("parent_id") == gold.get("parent_id"):
            return "same_parent_sibling"
    return "different_branch_or_unresolved"


def classify_prediction_error(
    row: Mapping[str, Any],
    *,
    target_nodes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify one wrong ranking without claiming a noisy label is ground truth."""

    source = row.get("source") or {}
    prediction = row.get("prediction") or {}
    gold_targets = list(row.get("gold_targets") or [])
    best_gold = gold_targets[0] if gold_targets else {}
    descriptor = (row.get("score_components") or {}).get("descriptor_affinity") or {}
    descriptor_delta = float(descriptor.get("gold_minus_prediction") or 0.0)
    relation_stage = "relation_hurt" if descriptor_delta > 1e-7 else "encoder_wrong_or_tied"
    nodes = dict(target_nodes or {})
    for node in [prediction, *gold_targets]:
        if node and node.get("node_id") is not None:
            nodes.setdefault(str(node["node_id"]), node)
    relation = structural_relation(
        str(prediction.get("node_id") or ""),
        [str(node.get("node_id") or "") for node in gold_targets],
        nodes,
    )
    source_text = node_semantic(source)
    predicted_text = node_semantic(prediction)
    gold_texts = {node_semantic(node) for node in gold_targets} - {""}
    exact_source_prediction = bool(source_text) and predicted_text == source_text
    exact_source_gold = source_text in gold_texts
    needs_label_audit = exact_source_prediction and not exact_source_gold
    label_audit_reason = None
    if needs_label_audit:
        label_audit_reason = (
            "prediction_exact_source_semantics_gold_blank"
            if not gold_texts
            else "prediction_exact_source_semantics_gold_disagrees"
        )
    rank = int(row.get("gold_rank") or 0)
    return {
        "availability": availability_slice(source, best_gold),
        "model_stage": relation_stage,
        "descriptor_gold_minus_prediction": descriptor_delta,
        "structural_relation": relation,
        "granularity_mismatch": relation
        in {"prediction_descendant_of_gold", "prediction_ancestor_of_gold"},
        "needs_label_audit": needs_label_audit,
        "label_audit_reason": label_audit_reason,
        "gold_rank_bucket": "rank_2_3" if 1 < rank <= 3 else "rank_4_5" if rank <= 5 else "rank_gt_5",
    }


def analyze_prediction_errors(
    predictions: Iterable[Mapping[str, Any]],
    *,
    train_availability_counts: Mapping[str, int] | None = None,
    target_nodes_by_page: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = list(predictions)
    errors: list[dict[str, Any]] = []
    slice_totals: Counter[str] = Counter()
    slice_errors: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    label_audit_counts: Counter[str] = Counter()
    app_totals: Counter[str] = Counter()
    app_errors: Counter[str] = Counter()
    for row in rows:
        gold = (row.get("gold_targets") or [{}])[0]
        availability = availability_slice(row.get("source") or {}, gold)
        slice_totals[availability] += 1
        app = _app_name(row)
        app_totals[app] += 1
        if bool(row.get("correct")):
            continue
        target_page = str((row.get("graph_pair") or {}).get("target") or "")
        diagnosis = classify_prediction_error(
            row,
            target_nodes=(target_nodes_by_page or {}).get(target_page),
        )
        enriched = dict(row)
        enriched["diagnosis"] = diagnosis
        errors.append(enriched)
        slice_errors[availability] += 1
        stage_counts[diagnosis["model_stage"]] += 1
        structure_counts[diagnosis["structural_relation"]] += 1
        rank_counts[diagnosis["gold_rank_bucket"]] += 1
        if diagnosis["needs_label_audit"]:
            label_audit_counts[str(diagnosis["label_audit_reason"])] += 1
        app_errors[app] += 1
    slices = {
        name: {
            "total": slice_totals[name],
            "errors": slice_errors[name],
            "error_rate": slice_errors[name] / slice_totals[name]
            if slice_totals[name]
            else 0.0,
        }
        for name in ("both_text", "mixed", "both_textless")
    }
    apps = [
        {
            "app": app,
            "total": total,
            "errors": app_errors[app],
            "error_rate": app_errors[app] / total if total else 0.0,
        }
        for app, total in app_totals.items()
    ]
    apps.sort(key=lambda item: (-item["error_rate"], -item["errors"], item["app"]))
    training_coverage = {
        name: int((train_availability_counts or {}).get(name, 0))
        for name in ("both_text", "mixed", "both_textless")
    }
    hard_coverage = training_coverage["mixed"] + training_coverage["both_textless"]
    total_coverage = sum(training_coverage.values())
    hard_fraction = hard_coverage / total_coverage if total_coverage else 0.0
    strict_correct = len(rows) - len(errors)
    audit_pending = sum(label_audit_counts.values())
    return {
        "schema_version": "omnitransfer.mapping_error_analysis.v1",
        "summary": {
            "total_predictions": len(rows),
            "correct": strict_correct,
            "errors": len(errors),
            "error_rate": len(errors) / len(rows) if rows else 0.0,
        },
        "slices": slices,
        "model_stage": dict(sorted(stage_counts.items())),
        "structural_relation": dict(sorted(structure_counts.items())),
        "gold_rank": dict(sorted(rank_counts.items())),
        "label_audit": {
            "pending_errors": audit_pending,
            "reasons": dict(sorted(label_audit_counts.items())),
            "strict_top1_accuracy": strict_correct / len(rows) if rows else 0.0,
            "top1_accuracy_if_all_pending_gold_are_wrong": (
                (strict_correct + audit_pending) / len(rows) if rows else 0.0
            ),
            "policy": (
                "pending rows remain strict errors until human review; "
                "never train against or relabel them automatically"
            ),
        },
        "apps": apps,
        "training_coverage": training_coverage,
        "diagnosis": {
            "dataset_too_small": (
                "hard_slices_underrepresented"
                if total_coverage and hard_fraction < 0.25
                else "not_established_from_count_alone"
            ),
            "hard_slice_fraction": hard_fraction,
            "interpretation": (
                "More random pairs are unlikely to fix the dominant failures; "
                "collect mixed-availability, textless, and cross-platform granularity cases."
            ),
        },
        "errors": errors,
    }


def _app_name(row: Mapping[str, Any]) -> str:
    source = str((row.get("graph_pair") or {}).get("source") or "")
    parts = source.split("/")
    return parts[1] if len(parts) > 1 else source or "unknown"


def count_training_availability(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        source_nodes = {
            str(node.get("node_id")): node
            for node in (record.get("source") or {}).get("graph", {}).get("nodes", [])
        }
        target_nodes = {
            str(node.get("node_id")): node
            for node in (record.get("target") or {}).get("graph", {}).get("nodes", [])
        }
        for match in record.get("matches") or []:
            source = source_nodes.get(str(match.get("source_node_id")), {})
            targets = [
                target_nodes.get(str(node_id), {})
                for node_id in match.get("target_node_ids") or []
            ]
            if targets:
                counts[availability_slice(source, targets[0])] += 1
    return dict(counts)


def index_nodes_by_page(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        for side in ("source", "target"):
            page = record.get(side) or {}
            page_id = str(page.get("page_id") or "")
            for node in (page.get("graph") or {}).get("nodes", []):
                result[page_id][str(node.get("node_id"))] = node
    return dict(result)
