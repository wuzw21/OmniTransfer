#!/usr/bin/env python3
"""Evaluate one learned matcher on frozen self-supervised UI graph views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.learned_matcher import LearnedGraphMatcher
from omnitransfer.self_supervised import AugmentConfig, evaluate_self_supervised_matcher
from omnitransfer.ui_graph import UIGraph
from scripts.pretrain_ui_graph_matcher import iter_graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="ui_graph_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--min-nodes", type=int, default=2)
    args = parser.parse_args()
    if args.max_screens < 0:
        raise SystemExit("--max-screens must be non-negative")
    if args.min_nodes < 2:
        raise SystemExit("--min-nodes must be at least two")

    graphs = load_eval_graphs(
        args.input,
        limit=args.max_screens,
        min_nodes=args.min_nodes,
    )
    if not graphs:
        raise SystemExit("No usable evaluation graphs found")
    matcher = LearnedGraphMatcher.from_checkpoint(args.checkpoint, device=args.device)
    augmentation = AugmentConfig()
    metrics = evaluate_self_supervised_matcher(
        matcher.model,
        graphs,
        seed=args.seed,
        device=args.device,
        augment_config=augmentation,
        matcher_config=matcher.config,
    )
    report = {
        "schema_version": "omnitransfer_ui_graph_eval_v1",
        "label": args.label,
        "inputs": [str(Path(value).resolve()) for value in args.input],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "seed": args.seed,
        "min_nodes": args.min_nodes,
        "loaded_graphs": len(graphs),
        "augmentation": augmentation.__dict__,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False))


def load_eval_graphs(
    inputs: list[str],
    *,
    limit: int,
    min_nodes: int,
) -> list[UIGraph]:
    """Load an immutable ordered evaluation slice from one or more files."""

    graphs: list[UIGraph] = []
    for value in inputs:
        for graph in iter_graphs(value):
            if len(graph.nodes) < min_nodes:
                continue
            graphs.append(graph)
            if limit and len(graphs) >= limit:
                return graphs
    return graphs


if __name__ == "__main__":
    main()
