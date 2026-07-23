#!/usr/bin/env python3
"""Evaluate deterministic offline matchers on canonical clean XML queries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import multiprocessing
import os
from pathlib import Path
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
UIGraph = _ui_graph.UIGraph
graph_from_record = _ui_graph.graph_from_record


DEFAULT_INPUT = (
    REPO_ROOT
    / "runtime/evals/vision_widget_mapping/clean_relative_xml_v1/queries.jsonl"
)


@dataclass(frozen=True)
class CleanQuery:
    query_id: str
    row_index: int
    split: str
    app: str
    widget_type: str
    source_xml_path: str
    target_xml_path: str
    source_node_id: str
    source_point: tuple[float, float] | None
    candidate_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("anchor", "similarity", "selector"),
        default="anchor",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--inspect", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    report = evaluate(
        args.input,
        mode=args.mode,
        limit=max(0, args.limit),
        inspect=max(0, args.inspect),
        progress_every=max(0, args.progress_every),
        workers=max(1, args.workers),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({**report["summary"], "output": str(args.output)}, indent=2))


def evaluate(
    input_path: Path,
    *,
    mode: str,
    limit: int = 0,
    inspect: int = 20,
    progress_every: int = 250,
    workers: int = 1,
) -> dict[str, Any]:
    if mode not in {"anchor", "similarity", "selector"}:
        raise ValueError("mode must be anchor, similarity, or selector")
    input_path = input_path.resolve()
    queries = _load_queries(input_path, limit=limit)
    grouped: dict[tuple[str, str], list[CleanQuery]] = defaultdict(list)
    for query in queries:
        grouped[(query.source_xml_path, query.target_xml_path)].append(query)
    tasks = [(rows, mode) for _, rows in sorted(grouped.items())]
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
                    f"processed={processed}/{len(queries)} elapsed_sec={elapsed:.1f}",
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
    failures = [row for row in rows if row["status"] != "evaluated"]
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    split_metrics = {
        split: _metrics([row for row in evaluated if row["split"] == split])
        for split in ("train", "dev", "test")
    }
    summary = {
        "baseline": {
            "anchor": "anchor_vote_64d_local",
            "similarity": "cosine_64d_only",
            "selector": "identity_selector_only",
        }[mode],
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "query_count": len(queries),
        "evaluated_query_count": len(evaluated),
        "failure_count": len(failures),
        "unique_screen_pairs": len(grouped),
        "candidate_scope": "clean_query_target_candidates_only",
        "gold_scope": "gold_equivalent_candidate_ids",
        "all": _metrics(evaluated),
        "splits": split_metrics,
        "wall_time_sec": time.perf_counter() - started,
        "workers": min(workers, len(tasks)),
    }
    return {
        "schema_version": "omnitransfer_clean_widget_mapping_eval_v1",
        "summary": summary,
        "failures": failures[: max(inspect, 20)],
        "first_divergences": [
            row for row in evaluated if not row["top1_correct"]
        ][:inspect],
        "rows": rows,
    }


def _load_queries(input_path: Path, *, limit: int) -> list[CleanQuery]:
    queries: list[CleanQuery] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            source = row.get("source") or {}
            source_xml_path = _resolve_path(metadata.get("source_xml_path"), input_path)
            target_xml_path = _resolve_path(metadata.get("target_xml_path"), input_path)
            candidate_ids = tuple(
                str(candidate.get("candidate_id") or "")
                for candidate in row.get("target_candidates") or ()
                if str(candidate.get("candidate_id") or "")
            )
            gold_values = metadata.get("gold_equivalent_candidate_ids") or ()
            if isinstance(gold_values, str):
                gold_values = (gold_values,)
            gold_ids = tuple(
                dict.fromkeys(
                    [
                        *(str(value) for value in gold_values if str(value)),
                        str(row.get("gold_candidate_id") or ""),
                    ]
                )
            )
            gold_ids = tuple(value for value in gold_ids if value)
            source_point = _source_point(source)
            query = CleanQuery(
                query_id=str(row.get("query_id") or f"query_{line_number}"),
                row_index=int(metadata.get("row_index", line_number - 1)),
                split=str(row.get("split") or metadata.get("split") or "unknown"),
                app=str(metadata.get("app") or "unknown"),
                widget_type=str(metadata.get("widget_type") or "unknown"),
                source_xml_path=str(source_xml_path),
                target_xml_path=str(target_xml_path),
                source_node_id=str(
                    metadata.get("source_node_id")
                    or source.get("node_id")
                    or source.get("id")
                    or ""
                ),
                source_point=source_point,
                candidate_ids=candidate_ids,
                gold_ids=gold_ids,
            )
            if not query.source_node_id:
                raise ValueError(f"query {query.query_id} has no source node id")
            if not query.candidate_ids:
                raise ValueError(f"query {query.query_id} has no target candidates")
            if not set(query.gold_ids).intersection(query.candidate_ids):
                raise ValueError(f"query {query.query_id} has no gold target candidate")
            queries.append(query)
            if limit and len(queries) >= limit:
                break
    if not queries:
        raise ValueError(f"no queries loaded from {input_path}")
    return queries


def _evaluate_screen_pair(
    task: tuple[list[CleanQuery], str],
) -> list[dict[str, Any]]:
    queries, mode = task
    source_graph = _screen_graph(Path(queries[0].source_xml_path))
    target_graph = _screen_graph(Path(queries[0].target_xml_path))
    matcher = AnchorVote64DMatcher(source_graph, target_graph)
    source_ids = {node.node_id for node in source_graph.nodes}
    target_ids = {node.node_id for node in target_graph.nodes}
    rows: list[dict[str, Any]] = []
    for query in queries:
        missing_candidates = set(query.candidate_ids).difference(target_ids)
        base = {
            "query_id": query.query_id,
            "row_index": query.row_index,
            "split": query.split,
            "app": query.app,
            "widget_type": query.widget_type,
            "source_node_id": query.source_node_id,
            "gold_ids": list(query.gold_ids),
            "candidate_count": len(query.candidate_ids),
        }
        if query.source_node_id not in source_ids:
            rows.append({**base, "status": "source_node_missing"})
            continue
        if missing_candidates:
            rows.append(
                {
                    **base,
                    "status": "target_candidates_missing",
                    "missing_candidate_count": len(missing_candidates),
                }
            )
            continue
        if mode == "selector":
            result = matcher.rank_selector(
                query.source_node_id,
                source_point=query.source_point,
                candidate_node_ids=query.candidate_ids,
            )
        else:
            result = matcher.rank(
                query.source_node_id,
                source_point=query.source_point,
                candidate_node_ids=query.candidate_ids,
                use_anchors=mode == "anchor",
            )
        gold_ids = set(query.gold_ids)
        ranked_ids = result.ranked_node_ids
        top1_correct = result.selected_node_id in gold_ids
        gold_rank = next(
            (index for index, node_id in enumerate(ranked_ids, start=1) if node_id in gold_ids),
            None,
        )
        rows.append(
            {
                **base,
                "status": "evaluated",
                "selected_node_id": result.selected_node_id,
                "top1_correct": top1_correct,
                "recall_at_5": gold_rank is not None and gold_rank <= 5,
                "gold_rank": gold_rank,
                "execute": result.execute,
                "execute_correct": result.execute and top1_correct,
                "gate_reason": result.reason,
                "latency_ms": result.latency_ms,
                "top_candidates": [
                    {
                        "node_id": candidate.node_id,
                        "support": candidate.support,
                        "semantic_similarity": candidate.semantic_similarity,
                        "geometric_log_score": candidate.geometric_log_score,
                        "anchor_count": candidate.anchor_count,
                    }
                    for candidate in result.candidates[:5]
                ],
            }
        )
    return rows


def _screen_graph(path: Path) -> UIGraph:
    return graph_from_record(
        {"xml": path.read_text(encoding="utf-8")},
        graph_id=str(path),
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [row for row in rows if row.get("selected_node_id")]
    executed = [row for row in rows if row.get("execute")]
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
    reciprocal_ranks = [
        1.0 / int(row["gold_rank"])
        for row in rows
        if row.get("gold_rank") is not None
    ]
    return {
        "query_count": len(rows),
        "top1_accuracy": _ratio(
            sum(bool(row.get("top1_correct")) for row in rows), len(rows)
        ),
        "recall_at_5": _ratio(
            sum(bool(row.get("recall_at_5")) for row in rows), len(rows)
        ),
        "mean_reciprocal_rank": _ratio(sum(reciprocal_ranks), len(rows)),
        "prediction_coverage": _ratio(len(predicted), len(rows)),
        "wrong_target_rate": _ratio(
            sum(not bool(row.get("top1_correct")) for row in predicted), len(rows)
        ),
        "execution_coverage": _ratio(len(executed), len(rows)),
        "selective_accuracy": _ratio(
            sum(bool(row.get("execute_correct")) for row in executed), len(executed)
        ),
        "executed_wrong_target_rate": _ratio(
            sum(not bool(row.get("top1_correct")) for row in executed), len(executed)
        ),
        "latency_ms_mean": _ratio(sum(latencies), len(latencies)),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }


def _source_point(source: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(source["x"]), float(source["y"])
    except (KeyError, TypeError, ValueError):
        bounds = source.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return None
        return (
            (float(bounds[0]) + float(bounds[2])) / 2.0,
            (float(bounds[1]) + float(bounds[3])) / 2.0,
        )


def _resolve_path(value: Any, input_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"query in {input_path} has a missing XML path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * quantile))))
    return values[index]


if __name__ == "__main__":
    main()
