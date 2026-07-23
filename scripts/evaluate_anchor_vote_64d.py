#!/usr/bin/env python3
"""Evaluate the isolated 64D plus local-anchor baseline on public pairs."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import multiprocessing
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_baseline = import_module("omnitransfer.baselines.anchor_vote_64d")
_ui_graph = import_module("omnitransfer.ui_graph")
AnchorVote64DMatcher = _baseline.AnchorVote64DMatcher
BoundNode = _baseline.BoundNode
PublicWidgetPair = _baseline.PublicWidgetPair
bind_public_bbox = _baseline.bind_public_bbox
load_public_widget_pairs = _baseline.load_public_widget_pairs
UIGraph = _ui_graph.UIGraph
graph_from_record = _ui_graph.graph_from_record


DEFAULT_DATASET = REPO_ROOT / "runtime/evals/vision_widget_mapping/testset.txt"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime/evals/vision_widget_mapping/anchor_vote_64d.baseline.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--inspect", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--heldout-percent", type=int, default=20)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--mode",
        choices=("anchor", "similarity", "selector"),
        default="anchor",
    )
    args = parser.parse_args()
    report = evaluate(
        args.dataset,
        limit=max(0, args.limit),
        inspect=max(0, args.inspect),
        progress_every=max(0, args.progress_every),
        heldout_percent=args.heldout_percent,
        workers=max(1, args.workers),
        mode=args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report["summary"], "output": str(args.output)}, indent=2))


def evaluate(
    dataset: Path,
    *,
    limit: int = 0,
    inspect: int = 20,
    progress_every: int = 100,
    heldout_percent: int = 20,
    workers: int = 1,
    mode: str = "anchor",
) -> dict[str, Any]:
    """Run binding, ranking, gating, and app-held-out reporting."""

    if heldout_percent <= 0 or heldout_percent >= 100:
        raise ValueError("heldout_percent must be between 1 and 99")
    if mode not in {"anchor", "similarity", "selector"}:
        raise ValueError("mode must be anchor, similarity, or selector")
    pairs = load_public_widget_pairs(dataset)
    if limit:
        pairs = pairs[:limit]
    grouped: dict[tuple[str, str], list[PublicWidgetPair]] = {}
    for pair in pairs:
        grouped.setdefault((pair.source_screen, pair.target_screen), []).append(pair)
    tasks = [
        (str(dataset.parent), group, heldout_percent, mode)
        for _, group in sorted(grouped.items())
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    processed = 0
    next_progress = progress_every
    if workers == 1:
        results = map(_evaluate_screen_pair, tasks)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=min(workers, len(tasks)))
        results = pool.imap_unordered(_evaluate_screen_pair, tasks, chunksize=1)
    try:
        for group_rows in results:
            rows.extend(group_rows)
            processed += len(group_rows)
            if progress_every and processed >= next_progress:
                elapsed = time.perf_counter() - started
                print(
                    f"processed={processed}/{len(pairs)} elapsed_sec={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += progress_every
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    rows.sort(key=lambda row: int(row["row_index"]))
    source_bound = sum(row.get("source_binding") is not None for row in rows)
    target_bound = sum(row.get("target_binding") is not None for row in rows)
    inspections = rows[:inspect]
    divergences = [
        row
        for row in rows
        if row.get("status") == "evaluated" and not row.get("top1_correct")
    ][:20]
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    heldout = [row for row in evaluated if row.get("heldout")]
    elapsed = time.perf_counter() - started
    summary = {
        "baseline": (
            "anchor_vote_64d_local"
            if mode == "anchor"
            else "cosine_64d_only"
            if mode == "similarity"
            else "identity_selector_only"
        ),
        "historical_source_commit": "25909482",
        "raw_total": len(pairs),
        "source_binding_count": source_bound,
        "source_binding_coverage": _ratio(source_bound, len(pairs)),
        "target_gold_binding_count": target_bound,
        "target_gold_binding_coverage": _ratio(target_bound, len(pairs)),
        "evaluable_query_count": len(evaluated),
        "all_evaluable": _metrics(evaluated),
        "app_heldout": {
            "split_rule": (
                f"sha1('anchor_vote_64d.v1:'+app) modulo 100 < {heldout_percent}"
            ),
            "query_count": len(heldout),
            **_metrics(heldout),
        },
        "wall_time_sec": elapsed,
        "unique_screen_pairs": len(grouped),
        "workers": min(workers, len(tasks)),
        "first_divergence_count": len(divergences),
    }
    return {
        "schema": "omnitransfer.anchor_vote_64d.eval.v1",
        "summary": summary,
        "manual_inspection": inspections,
        "first_divergences": divergences,
        "rows": rows,
    }


def _evaluate_screen_pair(
    task: tuple[str, list[PublicWidgetPair], int, str],
) -> list[dict[str, Any]]:
    dataset_root_text, pairs, heldout_percent, mode = task
    dataset_root = Path(dataset_root_text)
    source_screen = pairs[0].source_screen
    target_screen = pairs[0].target_screen
    source_graph = _screen_graph(dataset_root, source_screen)
    target_graph = _screen_graph(dataset_root, target_screen)
    source_image_size = _png_size(dataset_root / f"{source_screen}.png")
    target_image_size = _png_size(dataset_root / f"{target_screen}.png")
    matcher = AnchorVote64DMatcher(source_graph, target_graph)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        source_binding = bind_public_bbox(
            source_graph,
            pair.source_bbox,
            pair.source_class,
            screenshot_size=source_image_size,
        )
        target_binding = bind_public_bbox(
            target_graph,
            pair.target_bbox,
            pair.target_class,
            screenshot_size=target_image_size,
        )
        row = _base_row(pair, source_binding, target_binding)
        if source_binding is None or target_binding is None:
            row["status"] = (
                "source_and_target_binding_failed"
                if source_binding is None and target_binding is None
                else "source_binding_failed"
                if source_binding is None
                else "target_binding_failed"
            )
            rows.append(row)
            continue
        if mode == "selector":
            result = matcher.rank_selector(
                source_binding.node.node_id,
                source_point=_center(source_binding.normalized_bbox),
            )
        else:
            result = matcher.rank(
                source_binding.node.node_id,
                source_point=_center(source_binding.normalized_bbox),
                use_anchors=mode == "anchor",
            )
        gold_ids = set(target_binding.equivalent_node_ids)
        ranked_ids = result.ranked_node_ids
        top1_correct = bool(result.selected_node_id in gold_ids)
        recall5 = any(node_id in gold_ids for node_id in ranked_ids[:5])
        row.update(
            {
                "status": "evaluated",
                "selected_node_id": result.selected_node_id,
                "top1_correct": top1_correct,
                "recall_at_5": recall5,
                "execute": result.execute,
                "execute_correct": result.execute and top1_correct,
                "gate_reason": result.reason,
                "latency_ms": result.latency_ms,
                "heldout": _is_heldout(pair.app, heldout_percent),
                "top_candidates": [
                    {
                        "node_id": candidate.node_id,
                        "support": candidate.support,
                        "semantic_similarity": candidate.semantic_similarity,
                        "geometric_log_score": candidate.geometric_log_score,
                        "anchor_count": candidate.anchor_count,
                        "projected_point": list(candidate.projected_point),
                        **_node_description(target_graph, candidate.node_id),
                    }
                    for candidate in result.candidates[:5]
                ],
            }
        )
        rows.append(row)
    return rows


def _screen_graph(dataset_root: Path, screen: str) -> UIGraph:
    xml_path = dataset_root / f"{screen}.xml"
    return graph_from_record(
        {"xml": xml_path.read_text(encoding="utf-8")}, graph_id=screen
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [row for row in rows if row.get("selected_node_id")]
    executed = [row for row in rows if row.get("execute")]
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
    return {
        "top1": _ratio(sum(bool(row.get("top1_correct")) for row in rows), len(rows)),
        "recall_at_5": _ratio(sum(bool(row.get("recall_at_5")) for row in rows), len(rows)),
        "forced_wrong_target_rate": _ratio(
            sum(not bool(row.get("top1_correct")) for row in predicted), len(predicted)
        ),
        "prediction_coverage": _ratio(len(predicted), len(rows)),
        "abstention_rate": _ratio(len(rows) - len(executed), len(rows)),
        "execution_coverage": _ratio(len(executed), len(rows)),
        "executed_wrong_target_rate": _ratio(
            sum(not bool(row.get("top1_correct")) for row in executed), len(executed)
        ),
        "latency_ms_mean": _ratio(sum(latencies), len(latencies)),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }


def _base_row(
    pair: PublicWidgetPair,
    source_binding: BoundNode | None,
    target_binding: BoundNode | None,
) -> dict[str, Any]:
    return {
        "row_index": pair.row_index,
        "app": pair.app,
        "source_screen": pair.source_screen,
        "target_screen": pair.target_screen,
        "widget_type": pair.widget_type,
        "source_class": pair.source_class,
        "target_class": pair.target_class,
        "public_source_bbox": list(pair.source_bbox),
        "public_target_bbox": list(pair.target_bbox),
        "source_binding": _binding_description(source_binding),
        "target_binding": _binding_description(target_binding),
    }


def _binding_description(binding: BoundNode | None) -> dict[str, Any] | None:
    if binding is None:
        return None
    node = binding.node
    return {
        "node_id": node.node_id,
        "equivalent_node_ids": list(binding.equivalent_node_ids),
        "normalized_bbox": list(binding.normalized_bbox),
        "xml_bbox": list(node.bbox) if node.bbox else None,
        "iou": binding.iou,
        "class_name": node.class_name,
        "text": node.text,
        "content_desc": node.content_desc,
        "resource_id": node.resource_id,
    }


def _node_description(graph: UIGraph, node_id: str) -> dict[str, Any]:
    node = next((item for item in graph.nodes if item.node_id == node_id), None)
    if node is None:
        return {}
    return {
        "bbox": list(node.bbox) if node.bbox else None,
        "class_name": node.class_name,
        "text": node.text,
        "content_desc": node.content_desc,
        "resource_id": node.resource_id,
    }


def _png_size(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", header[16:24])


def _is_heldout(app: str, percent: int) -> bool:
    digest = hashlib.sha1(f"anchor_vote_64d.v1:{app}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return values[index]


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


if __name__ == "__main__":
    main()
