#!/usr/bin/env python3
"""Evaluate the optional local-context algorithm on canonical mapping pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.local_context_algorithm import LocalContextAlgorithm
from omnitransfer.mapping_data_cleaning import clean_ui_correspondence_records
from omnitransfer.mapping_dataset import validate_ui_correspondence_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--learned-predictions", type=Path)
    parser.add_argument(
        "--gated-threshold",
        type=float,
        help="Optional dev-selected learned-confidence threshold for diagnostic fusion",
    )
    args = parser.parse_args()

    records = _read_jsonl(args.input.expanduser().resolve(), validate_records=True)
    annotated = clean_ui_correspondence_records(
        records, quarantine_conflicts=False
    ).cleaned
    algorithm = LocalContextAlgorithm()
    prediction_rows = []
    metrics = _empty_metrics()
    slice_index: dict[tuple[str, str, str], set[str]] = {}
    for record in annotated:
        rows, record_slice_index = _evaluate_record(algorithm, record, metrics)
        prediction_rows.extend(rows)
        slice_index.update(record_slice_index)

    report: dict[str, Any] = {
        **algorithm.manifest(),
        "dataset": str(args.input.expanduser().resolve()),
        "pairs": len(annotated),
        "algorithm": _finalize_metrics(metrics),
    }
    if args.learned_predictions:
        learned = _read_jsonl(args.learned_predictions.expanduser().resolve())
        report["learned_model"] = _score_existing_predictions(learned, slice_index)
        if args.gated_threshold is not None:
            report["confidence_gated_branch"] = _score_confidence_gated_branch(
                learned,
                prediction_rows,
                slice_index,
                threshold=args.gated_threshold,
            )
        report["comparison_note"] = (
            "The algorithm is an offline hard-coded branch; learned-model numbers "
            "come from the supplied frozen prediction file on the same oriented rows."
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    prediction_path = (
        args.predictions.expanduser().resolve()
        if args.predictions
        else output.with_suffix(".predictions.jsonl")
    )
    _write_jsonl(prediction_path, prediction_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _evaluate_record(algorithm, record, metrics):
    source_nodes = record["source"]["graph"]["nodes"]
    target_nodes = record["target"]["graph"]["nodes"]
    forward, reverse = _oriented_gold(record)
    forward_rankings = algorithm.rank_many(source_nodes, target_nodes, list(forward))
    reverse_rankings = algorithm.rank_many(target_nodes, source_nodes, list(reverse))
    rows = []
    index = {}
    directions = (
        (
            "a_to_b",
            record["source"],
            record["target"],
            forward,
            forward_rankings,
        ),
        (
            "b_to_a",
            record["target"],
            record["source"],
            reverse,
            reverse_rankings,
        ),
    )
    for direction, source_page, target_page, gold, rankings in directions:
        source_map = _node_map(source_page)
        target_map = _node_map(target_page)
        for source_id, payload in gold.items():
            gold_ids, slices = payload
            ranking = rankings[source_id]
            ranked_ids = [row["node_id"] for row in ranking]
            gold_rank = min(
                (ranked_ids.index(gold_id) + 1 for gold_id in gold_ids if gold_id in ranked_ids),
                default=0,
            )
            prediction = ranking[0]
            correct = prediction["node_id"] in gold_ids
            _update_metrics(metrics, slices, correct, gold_rank)
            key = (str(source_page["page_id"]), str(target_page["page_id"]), source_id)
            index[key] = set(slices)
            best_gold = max(
                (row for row in ranking if row["node_id"] in gold_ids),
                key=lambda row: row["score"],
            )
            rows.append(
                {
                    "schema_version": "omnitransfer.correspondence_prediction.v1",
                    "graph_pair": {"source": key[0], "target": key[1]},
                    "direction": direction,
                    "source": _node_summary(source_map[source_id]),
                    "gold_targets": [_node_summary(target_map[value]) for value in gold_ids],
                    "prediction": _node_summary(target_map[prediction["node_id"]]),
                    "correct": correct,
                    "gold_rank": gold_rank,
                    "quality_slices": sorted(slices),
                    "top_candidates": [
                        {
                            **_node_summary(target_map[row["node_id"]]),
                            "rank_probability": row["confidence"],
                            "algorithm_score": row["score"],
                            "evidence": row["evidence"],
                        }
                        for row in ranking[:5]
                    ],
                    "score_components": {
                        "descriptor_affinity": {
                            "prediction": prediction["score"],
                            "best_gold": best_gold["score"],
                            "gold_minus_prediction": best_gold["score"] - prediction["score"],
                        }
                    },
                }
            )
    return rows, index


def _oriented_gold(record):
    forward: dict[str, tuple[set[str], set[str]]] = {}
    reverse_targets: dict[str, set[str]] = defaultdict(set)
    reverse_slices: dict[str, set[str]] = defaultdict(set)
    for match in record["matches"]:
        if match["label"] != "correspondence":
            continue
        source_id = str(match["source_node_id"])
        target_ids = {str(value) for value in match["target_node_ids"]}
        slices = set(match.get("quality_slices") or [])
        if source_id in forward:
            forward[source_id][0].update(target_ids)
            forward[source_id][1].update(slices)
        else:
            forward[source_id] = (set(target_ids), set(slices))
        for target_id in target_ids:
            reverse_targets[target_id].add(source_id)
            reverse_slices[target_id].update(slices)
    reverse = {
        target_id: (source_ids, reverse_slices[target_id])
        for target_id, source_ids in reverse_targets.items()
    }
    return forward, reverse


def _empty_metrics():
    return defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})


def _update_metrics(metrics, slices, correct, rank):
    for name in {"all", *slices}:
        values = metrics[name]
        values["total"] += 1
        values["top1"] += int(correct)
        values["top3"] += int(0 < rank <= 3)
        values["top5"] += int(0 < rank <= 5)


def _finalize_metrics(metrics):
    return {
        name: {
            **values,
            "top1_accuracy": values["top1"] / values["total"] if values["total"] else 0.0,
            "recall_at_3": values["top3"] / values["total"] if values["total"] else 0.0,
            "recall_at_5": values["top5"] / values["total"] if values["total"] else 0.0,
        }
        for name, values in sorted(metrics.items())
    }


def _score_existing_predictions(rows, slice_index):
    metrics = _empty_metrics()
    missing = 0
    for row in rows:
        pair = row.get("graph_pair") or {}
        source = row.get("source") or {}
        key = (str(pair.get("source") or ""), str(pair.get("target") or ""), str(source.get("node_id") or ""))
        slices = slice_index.get(key)
        if slices is None:
            missing += 1
            continue
        _update_metrics(metrics, slices, bool(row.get("correct")), int(row.get("gold_rank") or 0))
    return {"metrics": _finalize_metrics(metrics), "unmatched_prediction_rows": missing}


def _score_confidence_gated_branch(learned_rows, algorithm_rows, slice_index, *, threshold):
    algorithm = {
        (
            str(row["graph_pair"]["source"]),
            str(row["graph_pair"]["target"]),
            str(row["source"]["node_id"]),
        ): row
        for row in algorithm_rows
    }
    totals = defaultdict(lambda: {"total": 0, "top1": 0})
    overrides = 0
    for learned in learned_rows:
        key = (
            str(learned["graph_pair"]["source"]),
            str(learned["graph_pair"]["target"]),
            str(learned["source"]["node_id"]),
        )
        slices = slice_index.get(key)
        if slices is None or key not in algorithm:
            continue
        learned_candidates = learned.get("top_candidates") or []
        learned_prediction = str(learned["prediction"]["node_id"])
        algorithm_prediction = str(algorithm[key]["prediction"]["node_id"])
        confidence = float((learned_candidates or [{}])[0].get("rank_probability") or 0.0)
        candidate_ids = {str(row["node_id"]) for row in learned_candidates}
        prediction = learned_prediction
        if confidence < threshold and algorithm_prediction in candidate_ids:
            prediction = algorithm_prediction
            overrides += int(prediction != learned_prediction)
        gold = {str(row["node_id"]) for row in learned.get("gold_targets") or []}
        for name in {"all", *slices}:
            totals[name]["total"] += 1
            totals[name]["top1"] += int(prediction in gold)
    return {
        "purpose": "offline_diagnostic_only",
        "threshold": threshold,
        "policy": "override_low_confidence_only_when_algorithm_top1_is_in_learned_top5",
        "overrides": overrides,
        "metrics": {
            name: {
                **values,
                "top1_accuracy": values["top1"] / values["total"] if values["total"] else 0.0,
            }
            for name, values in sorted(totals.items())
        },
    }


def _node_map(page):
    return {str(node["node_id"]): node for node in page["graph"]["nodes"]}


def _node_summary(node):
    return {
        field: node.get(field)
        for field in (
            "node_id",
            "text",
            "content_desc",
            "class_name",
            "bbox",
            "clickable",
            "editable",
            "scrollable",
            "enabled",
        )
    }


def _read_jsonl(path: Path, *, validate_records: bool = False):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if validate_records:
        return [validate_ui_correspondence_pair(row) for row in rows]
    return rows


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
