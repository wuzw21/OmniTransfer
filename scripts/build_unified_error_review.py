#!/usr/bin/env python3
"""Materialize one review queue for mapping errors and recent manual labels.

The queue deliberately keeps public/self-supervised/manual evidence separate in
the review metadata.  It is a review artifact, not a new gold dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.ios_adapter import attach_xml_payload


_ROW_RE = re.compile(r":row_(\d+):")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _node_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["node_id"]): node
        for node in record["target"]["graph"].get("nodes", [])
    }


def _ancestors(nodes: dict[str, dict[str, Any]], node_id: str) -> set[str]:
    result: set[str] = set()
    current = str(node_id)
    while current and current not in result:
        result.add(current)
        node = nodes.get(current)
        parent = node.get("parent_id") if node else None
        current = str(parent) if parent is not None else ""
    return result


def _contains(outer: list[float] | None, inner: list[float] | None) -> bool:
    if not outer or not inner or len(outer) != 4 or len(inner) != 4:
        return False
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _reason(
    record: dict[str, Any], ours_node_id: str | None, reference_node_id: str | None
) -> dict[str, Any]:
    if not ours_node_id or not reference_node_id:
        return {"code": "unresolved_reference", "label": "参考节点尚未对齐"}
    if ours_node_id == reference_node_id:
        return {"code": "same_node", "label": "预测与参考节点相同"}
    nodes = _node_map(record)
    ours = nodes.get(str(ours_node_id), {})
    reference = nodes.get(str(reference_node_id), {})
    if str(ours_node_id) in _ancestors(nodes, str(reference_node_id)):
        return {
            "code": "ours_ancestor_of_reference",
            "label": "预测为参考节点的父级节点",
        }
    if str(reference_node_id) in _ancestors(nodes, str(ours_node_id)):
        return {
            "code": "reference_ancestor_of_ours",
            "label": "预测为参考节点的子级节点",
        }
    if _contains(ours.get("bbox"), reference.get("bbox")):
        return {
            "code": "ours_bbox_contains_reference",
            "label": "预测框包含参考框，但节点不是同一节点",
        }
    if _contains(reference.get("bbox"), ours.get("bbox")):
        return {
            "code": "reference_bbox_contains_ours",
            "label": "参考框包含预测框，但节点不是同一节点",
        }
    if ours.get("parent_id") == reference.get("parent_id"):
        return {
            "code": "same_parent_sibling_confusion",
            "label": "预测与参考节点是同一父级下的兄弟节点",
        }
    return {
        "code": "different_branch_or_unrelated",
        "label": "预测与参考节点位于不同结构分支",
    }


def _node_flags(record: dict[str, Any], ours_id: str | None, reference_id: str | None) -> dict[str, Any]:
    nodes = _node_map(record)
    ours = nodes.get(str(ours_id), {}) if ours_id else {}
    reference = nodes.get(str(reference_id), {}) if reference_id else {}
    return {
        "ours_class": ours.get("class_name"),
        "reference_class": reference.get("class_name"),
        "ours_clickable": ours.get("clickable"),
        "reference_clickable": reference.get("clickable"),
        "class_mismatch": bool(ours and reference) and ours.get("class_name") != reference.get("class_name"),
        "clickable_mismatch": bool(ours and reference) and bool(ours.get("clickable")) != bool(reference.get("clickable")),
        "same_parent": bool(ours and reference) and ours.get("parent_id") == reference.get("parent_id"),
    }


def _pair_row(record: dict[str, Any], source_label: str) -> dict[str, Any]:
    index = int(source_label[1:]) - 1
    if index < 0 or index >= len(record["matches"]):
        raise ValueError(f"{record['pair_id']} has no {source_label}")
    return record["matches"][index]


def _base_index_row(
    record: dict[str, Any],
    source_label: str,
    *,
    group: str,
    status: str,
    ours_node_id: str | None,
    reference_node_id: str | None,
    reference_status: str,
    note: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    match = _pair_row(record, source_label)
    if reference_node_id is None and match.get("target_node_ids"):
        reference_node_id = str(match["target_node_ids"][0])
    reason = _reason(record, ours_node_id, reference_node_id)
    return {
        "pair_id": record["pair_id"],
        "source_label": source_label,
        "target_label": f"M{source_label[1:]}",
        "ours_node_id": ours_node_id,
        "reference_node_id": reference_node_id,
        "gold_node_id": reference_node_id,
        "review_group": group,
        "review_status": status,
        "reference_status": reference_status,
        "reason_code": reason["code"],
        "reason_label": reason["label"],
        "node_flags": _node_flags(record, ours_node_id, reference_node_id),
        "source_mapping": copy.deepcopy(match),
        "note": note,
        **extra,
    }


def _find_page_pair(records: dict[str, dict[str, Any]], source: str, target: str) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records.values()
        if record["source"].get("page_id") == source
        and record["target"].get("page_id") == target
    ]
    return candidates[0] if len(candidates) == 1 else None


def build(args: argparse.Namespace) -> dict[str, Any]:
    pool_path = args.pool.expanduser().resolve()
    records = {record["pair_id"]: record for record in _read_jsonl(pool_path)}
    remaining_rows = json.loads(args.remaining.expanduser().resolve().read_text(encoding="utf-8"))["rows"]
    parent_rows = json.loads(args.parent.read_text(encoding="utf-8"))["rows"]
    annotations = _read_jsonl(args.annotations.expanduser().resolve())

    selected_ids: list[str] = []
    index_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    def add_record_id(pair_id: str) -> None:
        if pair_id in records and pair_id not in selected_ids:
            selected_ids.append(pair_id)

    for row in remaining_rows:
        pair_id = str(row["pair_id"])
        record = records.get(pair_id)
        if not record:
            continue
        add_record_id(pair_id)
        key = (pair_id, str(row["source_label"]), "model_error")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        index_rows.append(
            _base_index_row(
                record,
                str(row["source_label"]),
                group="model_error_reference_pending",
                status="needs_review",
                ours_node_id=str(row.get("ours_node_id") or "") or None,
                reference_node_id=str(row.get("gold_node_id") or "") or None,
                reference_status="self_supervised_reference",
                note="来自 remaining_manual.v2；当前 reference 仍是 self-supervised/pseudo label，需人工确认。",
                original_index_source="review.html.remaining_manual.v2.index.json",
            )
        )

    for row in parent_rows:
        pair_id = str(row["pair_id"])
        record = records.get(pair_id)
        if not record:
            continue
        add_record_id(pair_id)
        key = (pair_id, str(row["source_label"]), "parent_exception")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        ours = row.get("ours_node") or {}
        index_rows.append(
            _base_index_row(
                record,
                str(row["source_label"]),
                group="manual_parent_exception",
                status="previously_accepted_reaudit",
                ours_node_id=str(ours.get("node_id") or "") or None,
                reference_node_id=None,
                reference_status="self_supervised_reference_plus_manual_accept",
                note="之前人工标注：父级节点实际可点击，视为正确；本次仍放入统一队列复核。",
                original_index_source="parent_node_manual_annotations.accepted.json",
                previous_decision=row.get("decision"),
            )
        )

    annotation_join: list[dict[str, Any]] = []
    for annotation in annotations:
        source = str(annotation.get("source_screen") or "")
        target = str(annotation.get("target_screen") or "")
        record = _find_page_pair(records, source, target)
        mapping_ids = annotation.get("annotation", {}).get("mappings", {})
        for mapping_id, mapping_annotation in mapping_ids.items():
            if not record:
                annotation_join.append({
                    "mapping_id": mapping_id,
                    "join_status": "page_pair_not_found",
                    "annotation": mapping_annotation,
                })
                continue
            add_record_id(record["pair_id"])
            current_ids = list(record.get("provenance", {}).get("mapping_ids", []))
            current_index = current_ids.index(mapping_id) if mapping_id in current_ids else None
            if current_index is None:
                match = _ROW_RE.search(mapping_id)
                row_number = int(match.group(1)) if match else None
                current_index = next(
                    (index for index, value in enumerate(current_ids) if f":row_{row_number}:" in value),
                    None,
                ) if row_number is not None else None
            if current_index is None or current_index >= len(record["matches"]):
                annotation_join.append({
                    "pair_id": record["pair_id"],
                    "mapping_id": mapping_id,
                    "join_status": "mapping_id_not_found",
                    "current_mapping_ids": current_ids,
                    "annotation": mapping_annotation,
                })
                continue
            source_label = f"S{current_index + 1}"
            current_mapping_id = current_ids[current_index]
            reference_id = str(record["matches"][current_index]["target_node_ids"][0]) if record["matches"][current_index].get("target_node_ids") else None
            key = (record["pair_id"], source_label, "browser_annotation")
            if key not in seen_keys:
                seen_keys.add(key)
                index_rows.append(
                    _base_index_row(
                        record,
                        source_label,
                        group="manual_browser_annotation",
                        status="wrong_annotation_join_needs_review",
                        ours_node_id=reference_id,
                        reference_node_id=reference_id,
                        reference_status="self_supervised_reference_not_manual_gold",
                        note="浏览器人工标注为 wrong；当前 mapping hash 不一致，仅按 row 序号临时对齐，必须人工确认。",
                        original_index_source=".playwright-cli/widget-mapping-pair-annotations.jsonl",
                        annotation_mapping_id=mapping_id,
                        current_mapping_id=current_mapping_id,
                        join_status="row_ordinal_fallback_hash_mismatch" if current_mapping_id != mapping_id else "exact_mapping_id",
                        annotation_status=mapping_annotation.get("status"),
                        reason_code="manual_annotation_current_mapping_unresolved",
                        reason_label="人工标注为 wrong，但当前 mapping hash 未精确对齐",
                    )
                )
            annotation_join.append({
                "pair_id": record["pair_id"],
                "mapping_id": mapping_id,
                "current_mapping_id": current_mapping_id,
                "join_status": "row_ordinal_fallback_hash_mismatch" if current_mapping_id != mapping_id else "exact_mapping_id",
                "annotation": mapping_annotation,
            })

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_path = output / "unified_error_pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as handle:
        for pair_id in selected_ids:
            record = copy.deepcopy(records[pair_id])
            for side in ("source", "target"):
                record[side] = attach_xml_payload(record[side])
            record.setdefault("provenance", {})["unified_review"] = True
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    index_path = output / "review.html.unified_manual.index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "omnitransfer.unified_error_review_index.v1",
                "rows": index_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    payload_path = output / "review.html.payload.json"
    if payload_path.is_file():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["unified_manual_index"] = {
            "schema_version": "omnitransfer.unified_error_review_index.v1",
            "rows": index_rows,
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    annotation_path = output / "manual_annotation_join.json"
    annotation_path.write_text(
        json.dumps(annotation_join, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reason_counts = Counter(row["reason_code"] for row in index_rows)
    group_counts = Counter(row["review_group"] for row in index_rows)
    manifest = {
        "schema_version": "omnitransfer.unified_error_review_manifest.v1",
        "review_not_gold": True,
        "pairs": len(selected_ids),
        "review_rows": len(index_rows),
        "groups": dict(sorted(group_counts.items())),
        "reasons": dict(sorted(reason_counts.items())),
        "inputs": {
            "pool": {"path": str(pool_path), "sha256": file_sha256(pool_path)},
            "remaining": {"path": str(args.remaining.resolve()), "sha256": file_sha256(args.remaining.resolve())},
            "parent": {"path": str(args.parent.resolve()), "sha256": file_sha256(args.parent.resolve())},
            "annotations": {"path": str(args.annotations.resolve()), "sha256": file_sha256(args.annotations.resolve())},
        },
        "artifacts": {
            "pairs": pair_path.name,
            "index": index_path.name,
            "annotation_join": annotation_path.name,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--remaining", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
