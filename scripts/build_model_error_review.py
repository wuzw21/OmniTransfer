#!/usr/bin/env python3
"""Analyze model errors and merge representative cases into one live review."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.ios_adapter import attach_xml_payload
from omnitransfer.mapping_error_analysis import (
    analyze_prediction_errors,
    count_training_availability,
    index_nodes_by_page,
)
from omnitransfer.mapping_pair_review import build_mapping_pair_review


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate on physical ``\n`` records. ``str.splitlines`` also splits on
    # Unicode separators (for example U+2028) that legitimately occur inside
    # captured UI text and would corrupt an otherwise valid JSON record.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selection_groups(error: dict[str, Any]) -> list[str]:
    diagnosis = error["diagnosis"]
    groups = [diagnosis["availability"], diagnosis["model_stage"]]
    if diagnosis["needs_label_audit"]:
        groups.append("needs_label_audit")
    if diagnosis["granularity_mismatch"]:
        groups.append("granularity_mismatch")
    if diagnosis["structural_relation"] == "same_parent_sibling":
        groups.append("same_parent_sibling")
    if diagnosis["gold_rank_bucket"] == "rank_gt_5":
        groups.append("rank_gt_5")
    return groups


def select_balanced_errors(errors: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(errors):
        return list(errors)
    priority = [
        "needs_label_audit",
        "relation_hurt",
        "granularity_mismatch",
        "both_textless",
        "mixed",
        "same_parent_sibling",
        "rank_gt_5",
        "encoder_wrong_or_tied",
        "both_text",
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in errors:
        for group in _selection_groups(error):
            buckets[group].append(error)
    # Label conflicts are not ordinary model errors. Keep every one in the
    # review queue before balancing the remaining true-error slices, otherwise
    # a small display limit can hide bad supervision behind model statistics.
    selected = [
        row for row in errors if row["diagnosis"].get("needs_label_audit")
    ][:limit]
    seen = {_error_key(row) for row in selected}
    cursor = Counter()
    while len(selected) < limit:
        progressed = False
        for group in priority:
            rows = buckets[group]
            while cursor[group] < len(rows):
                row = rows[cursor[group]]
                cursor[group] += 1
                key = _error_key(row)
                if key in seen:
                    continue
                selected.append(row)
                seen.add(key)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if len(selected) < limit:
        for row in errors:
            key = _error_key(row)
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= limit:
                break
    return selected


def _error_key(row: dict[str, Any]) -> str:
    pair = row.get("graph_pair") or {}
    source = row.get("source") or {}
    return "|".join(
        [
            str(pair.get("source") or ""),
            str(pair.get("target") or ""),
            str(row.get("direction") or ""),
            str(source.get("node_id") or source.get("index") or ""),
        ]
    )


def _oriented_record(
    error: dict[str, Any],
    records_by_pages: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pages = error["graph_pair"]
    source_page = str(pages["source"])
    target_page = str(pages["target"])
    base = records_by_pages.get((source_page, target_page))
    reverse = False
    if base is None:
        base = records_by_pages.get((target_page, source_page))
        reverse = base is not None
    if base is None:
        raise KeyError(f"prediction pages not found in dataset: {source_page} -> {target_page}")
    record = copy.deepcopy(base)
    if reverse:
        record["source"], record["target"] = record["target"], record["source"]
        source_platform = record.get("slices", {}).get("source_platform")
        target_platform = record.get("slices", {}).get("target_platform")
        record.setdefault("slices", {})["source_platform"] = target_platform
        record["slices"]["target_platform"] = source_platform
    source_id = str(error["source"]["node_id"])
    gold_ids = [str(node["node_id"]) for node in error["gold_targets"]]
    digest = hashlib.blake2b(_error_key(error).encode(), digest_size=8).hexdigest()
    record["pair_id"] = f"current-model-error:{digest}"
    record["matches"] = [
        {
            "source_node_id": source_id,
            "target_node_ids": gold_ids,
            "label": "correspondence",
        }
    ]
    record["source"] = attach_xml_payload(record["source"])
    record["target"] = attach_xml_payload(record["target"])
    record["method_tags"] = [
        {
            "id": error["diagnosis"]["model_stage"],
            "label": error["diagnosis"]["model_stage"],
            "count": 1,
        }
    ]
    record.setdefault("provenance", {}).update(
        {
            "unified_review": True,
            "review_source": "current_model_test_error",
            "original_pair_id": base["pair_id"],
        }
    )
    prediction = error.get("prediction") or {}
    diagnosis = error["diagnosis"]
    index_row = {
        "pair_id": record["pair_id"],
        "source_label": "S1",
        "target_label": "M1",
        "ours_node_id": prediction.get("node_id"),
        "ours_node": prediction,
        "reference_node_id": gold_ids[0] if gold_ids else None,
        "gold_node_id": gold_ids[0] if gold_ids else None,
        "gold_node_ids": gold_ids,
        "review_group": f"current_model::{diagnosis['model_stage']}::{diagnosis['availability']}",
        "review_status": "needs_review",
        "reference_status": "public_gold_needs_error_audit",
        "reason_code": diagnosis["structural_relation"],
        "reason_label": _reason_label(diagnosis),
        "node_flags": {
            "ours_class": prediction.get("class_name"),
            "reference_class": (error.get("gold_targets") or [{}])[0].get("class_name"),
            "ours_clickable": prediction.get("clickable"),
            "reference_clickable": (error.get("gold_targets") or [{}])[0].get("clickable"),
            "same_parent": diagnosis["structural_relation"] == "same_parent_sibling",
        },
        "gold_rank": error.get("gold_rank"),
        "descriptor_gold_minus_prediction": diagnosis["descriptor_gold_minus_prediction"],
        "needs_label_audit": diagnosis["needs_label_audit"],
        "source_mapping": copy.deepcopy(record["matches"][0]),
        "note": _review_note(error),
        "original_index_source": "relation_slots_l3_h64_seed17_final.test.predictions.jsonl",
    }
    return record, index_row


def _reason_label(diagnosis: dict[str, Any]) -> str:
    labels = {
        "prediction_descendant_of_gold": "预测落在 gold 的语义子节点：跨平台控件粒度可能不一致",
        "prediction_ancestor_of_gold": "预测落在 gold 的可交互父节点：跨平台控件粒度可能不一致",
        "same_parent_sibling": "预测与 gold 是兄弟节点：局部角色区分失败",
        "different_branch_or_unresolved": "预测进入不同结构分支",
    }
    label = labels.get(diagnosis["structural_relation"], diagnosis["structural_relation"])
    if diagnosis["needs_label_audit"]:
        label += "；模型与 source 文本完全一致，需复核 gold"
    return label


def _review_note(error: dict[str, Any]) -> str:
    diagnosis = error["diagnosis"]
    return (
        f"availability={diagnosis['availability']}；stage={diagnosis['model_stage']}；"
        f"gold_rank={error.get('gold_rank')}；"
        f"descriptor(gold-pred)={diagnosis['descriptor_gold_minus_prediction']:.4f}。"
        "G 是数据集 gold，M 是模型预测；两者都必须结合截图/XML 人工复核。"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.expanduser().resolve()
    prediction_path = args.predictions.expanduser().resolve()
    train_path = args.train.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = _read_jsonl(dataset_path)
    predictions = _read_jsonl(prediction_path)
    train = _read_jsonl(train_path)
    report = analyze_prediction_errors(
        predictions,
        train_availability_counts=count_training_availability(train),
        target_nodes_by_page=index_nodes_by_page(dataset),
    )
    errors = report.pop("errors")
    selected = select_balanced_errors(errors, args.max_current_errors)
    records_by_pages = {
        (str(row["source"]["page_id"]), str(row["target"]["page_id"])): row
        for row in dataset
    }
    current_pairs: list[dict[str, Any]] = []
    current_index: list[dict[str, Any]] = []
    for error in selected:
        pair, index_row = _oriented_record(error, records_by_pages)
        current_pairs.append(pair)
        current_index.append(index_row)
    current_pair_path = output / "current_model_error_pairs.jsonl"
    all_error_path = output / "all_model_errors.jsonl"
    report_path = output / "error_analysis.json"
    _write_jsonl(current_pair_path, current_pairs)
    _write_jsonl(all_error_path, errors)
    report["review_selection"] = {
        "selected": len(selected),
        "available_errors": len(errors),
        "groups": dict(
            sorted(Counter(group for row in selected for group in _selection_groups(row)).items())
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_inputs = [current_pair_path]
    index_rows = list(current_index)
    existing = args.existing_review_dir.expanduser().resolve() if args.existing_review_dir else None
    if existing:
        old_pairs = existing / "unified_error_pairs.jsonl"
        old_index = existing / "review.html.unified_manual.index.json"
        if old_pairs.is_file() and old_index.is_file():
            review_inputs.insert(0, old_pairs)
            # Put the checkpoint currently under audit first. Historical and
            # manual rows remain in the same queue immediately afterwards.
            index_rows = index_rows + json.loads(old_index.read_text(encoding="utf-8")).get("rows", [])
    cleaning_quarantine = (
        args.cleaning_quarantine.expanduser().resolve()
        if args.cleaning_quarantine
        else None
    )
    cleaning_rows = 0
    if cleaning_quarantine:
        review_inputs.append(cleaning_quarantine)
        for record in _read_jsonl(cleaning_quarantine):
            conflicts = list(
                record.get("provenance", {}).get(
                    "cleaning_conflicting_target_node_ids", []
                )
            )
            for match_index, match in enumerate(record["matches"], start=1):
                gold_ids = [str(value) for value in match.get("target_node_ids") or []]
                index_rows.append(
                    {
                        "pair_id": record["pair_id"],
                        "source_label": f"S{match_index}",
                        "target_label": f"M{match_index}",
                        "ours_node_id": conflicts[0] if conflicts else None,
                        "ours_node_ids": conflicts,
                        "reference_node_id": gold_ids[0] if gold_ids else None,
                        "gold_node_id": gold_ids[0] if gold_ids else None,
                        "gold_node_ids": gold_ids,
                        "review_group": "data_cleaning::exact_semantic_conflict",
                        "review_status": "needs_review",
                        "reference_status": "public_gold_needs_label_audit",
                        "reason_code": record.get("provenance", {}).get("cleaning_reason"),
                        "reason_label": "source 语义在 target 另一分支精确出现，请复核原始 gold",
                        "node_flags": {},
                        "source_mapping": copy.deepcopy(match),
                        "note": "清洗器没有改标签；G 是原始 gold，M 是语义冲突候选。",
                        "original_index_source": str(cleaning_quarantine),
                    }
                )
                cleaning_rows += 1
    index_path = output / "review.html.unified_manual.index.json"
    index_path.write_text(
        json.dumps(
            {"schema_version": "omnitransfer.unified_error_review_index.v2", "rows": index_rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    review_manifest = build_mapping_pair_review(review_inputs, output, external_payload=False)
    manifest = {
        "schema_version": "omnitransfer.model_error_review_manifest.v1",
        "current_model_errors": len(errors),
        "current_model_selected": len(selected),
        "cleaning_quarantine_rows": cleaning_rows,
        "unified_review_rows": len(index_rows),
        "unified_review_pairs": review_manifest["pairs"],
        "inputs": {
            "dataset": {"path": str(dataset_path), "sha256": file_sha256(dataset_path)},
            "predictions": {"path": str(prediction_path), "sha256": file_sha256(prediction_path)},
            "train": {"path": str(train_path), "sha256": file_sha256(train_path)},
            "existing_review_dir": str(existing) if existing else None,
            "cleaning_quarantine": str(cleaning_quarantine) if cleaning_quarantine else None,
        },
        "artifacts": {
            "review": "review.html",
            "analysis": report_path.name,
            "all_errors": all_error_path.name,
            "selected_pairs": current_pair_path.name,
            "index": index_path.name,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-review-dir", type=Path)
    parser.add_argument("--cleaning-quarantine", type=Path)
    parser.add_argument("--max-current-errors", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
