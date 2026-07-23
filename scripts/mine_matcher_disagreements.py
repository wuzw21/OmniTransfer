#!/usr/bin/env python3
"""Build a rule-resistant benchmark slice from frozen matcher predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from omnitransfer.disagreement import (
    group_disagreements_by_page_pair,
    mine_disagreements,
    prediction_from_mapping,
)
from omnitransfer.importers import load_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rich-eval", type=Path)
    source.add_argument("--learned-predictions", type=Path)
    parser.add_argument("--selector-predictions", type=Path)
    parser.add_argument("--learned-method", default="learned_cross_attention")
    parser.add_argument("--selector-method", default="paper_ih_layout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-pair-output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-rank-advantage", type=int, default=1)
    parser.add_argument("--include-regressions", action="store_true")
    parser.add_argument("--max-anchors-per-page-pair", type=int, default=8)
    args = parser.parse_args()

    method_metadata: dict[str, Any] = {}
    if args.rich_eval:
        learned_rows, selector_rows, method_metadata = _predictions_from_rich_eval(
            args.rich_eval,
            learned_method=args.learned_method,
            selector_method=args.selector_method,
        )
    else:
        if args.selector_predictions is None:
            parser.error("--selector-predictions is required with --learned-predictions")
        learned_rows = _load_jsonl(args.learned_predictions)
        selector_rows = _load_jsonl(args.selector_predictions)
    cases = mine_disagreements(
        load_queries(args.queries),
        (prediction_from_mapping(row) for row in learned_rows),
        (prediction_from_mapping(row) for row in selector_rows),
        min_rank_advantage=max(1, args.min_rank_advantage),
        include_regressions=args.include_regressions,
    )
    selected = cases[: args.limit or None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in selected),
        encoding="utf-8",
    )
    page_pairs = group_disagreements_by_page_pair(
        selected,
        max_anchors_per_pair=max(0, args.max_anchors_per_page_pair),
    )
    page_pair_path = args.page_pair_output or args.output.with_name(
        args.output.stem + ".page_pairs.jsonl"
    )
    page_pair_path.parent.mkdir(parents=True, exist_ok=True)
    page_pair_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in page_pairs),
        encoding="utf-8",
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = {
        "schema_version": "omnitransfer_matcher_disagreement_slice_v1",
        "queries": str(args.queries.resolve()),
        "learned_method": args.learned_method,
        "selector_method": args.selector_method,
        "learned_method_metadata": method_metadata,
        "selection_basis": (
            "set_valued_gold_rank_top1_disagreement_within_model_margin_and_null_safety"
        ),
        "raw_cross_model_score_subtraction": False,
        "selector_used_as_model_input": False,
        "total_cases": len(cases),
        "written_cases": len(selected),
        "page_pairs": len(page_pairs),
        "page_pair_output": str(page_pair_path.resolve()),
        "multi_anchor_page_pairs": sum(row["anchor_count"] > 1 for row in page_pairs),
        "category_counts": dict(Counter(case.category for case in selected)),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def _predictions_from_rich_eval(
    path: Path,
    *,
    learned_method: str,
    selector_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matchers = payload.get("matchers") or {}
    missing = [name for name in (learned_method, selector_method) if name not in matchers]
    if missing:
        raise ValueError(f"missing rich-eval matcher(s): {', '.join(missing)}")
    metadata = matchers[learned_method].get("metadata") or {}
    public_metadata = {
        key: metadata[key]
        for key in (
            "model_type",
            "hot_path_model_family",
            "base_backend",
            "selection_policy",
        )
        if key in metadata
    }
    return (
        list(matchers[learned_method]["predictions"]),
        list(matchers[selector_method]["predictions"]),
        public_metadata,
    )


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
