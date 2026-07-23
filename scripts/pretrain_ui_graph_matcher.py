#!/usr/bin/env python3
"""Pretrain a UI graph matcher from unlabeled MobileViews/RICO-like hierarchies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from omnitransfer.learned_matcher import (
    LearnedGraphMatcher,
    MatcherConfig,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.self_supervised import (
    AugmentConfig,
    evaluate_self_supervised_matcher,
    make_training_pair,
    train_self_supervised_matcher,
)
from omnitransfer.ui_graph import UIGraph
from omnitransfer.unified_ui import graph_from_unified_record


def iter_graphs(input_path: str, *, split: str = "train") -> Iterable[UIGraph]:
    """Yield parsed UI graphs from local files or optional HF datasets."""

    if input_path.startswith("hf://"):
        dataset_id = input_path.removeprefix("hf://")
        yield from _iter_hf_graphs(dataset_id, split=split)
        return

    path = Path(input_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield _graph_from_record_safe(
                    json.loads(line), f"{path.stem}:{line_number}"
                )
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for index, record in enumerate(records):
            yield _graph_from_record_safe(record, f"{path.stem}:{index}")
        return
    if path.suffix == ".parquet":
        yield from _iter_parquet_graphs(path)
        return
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised UI graph matcher pretraining. The input can be "
            "MobileViews/RICO-like JSONL/JSON/Parquet records with a UI hierarchy."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Local JSONL/JSON/Parquet files or hf://dataset_id values",
    )
    parser.add_argument(
        "--dev-input",
        nargs="+",
        default=None,
        help="Optional frozen dev graph files",
    )
    parser.add_argument(
        "--test-input",
        nargs="+",
        default=None,
        help="Optional frozen test graph files",
    )
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument(
        "--split", default="train", help="HF split name when --input uses hf://"
    )
    parser.add_argument("--max-screens", type=int, default=1000)
    parser.add_argument("--max-dev-screens", type=int, default=0)
    parser.add_argument("--max-test-screens", type=int, default=0)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--source-context-nodes", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--drop-node-prob", type=float, default=0.15)
    parser.add_argument("--distractor-prob", type=float, default=0.20)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse data and construct the first augmented pair; do not import torch.",
    )
    args = parser.parse_args()

    graphs = _load_usable_graphs(
        args.input,
        split=args.split,
        limit=args.max_screens,
        min_nodes=args.min_nodes,
    )
    if not graphs:
        raise SystemExit("No usable UI graphs found.")

    augment_config = AugmentConfig(
        drop_node_prob=args.drop_node_prob,
        distractor_prob=args.distractor_prob,
        max_distractors=args.max_distractors,
    )
    pretrained_model = None
    if args.pretrained:
        pretrained = LearnedGraphMatcher.from_checkpoint(
            args.pretrained,
            device=args.device,
        )
        matcher_config = pretrained.config
        pretrained_model = pretrained.model
    else:
        matcher_config = MatcherConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            source_context_nodes=args.source_context_nodes,
        )
    if bool(args.dev_input) != bool(args.test_input):
        raise SystemExit("--dev-input and --test-input must be provided together")
    if args.dev_input and args.test_input:
        graph_splits = {
            "train": graphs,
            "dev": _load_usable_graphs(
                args.dev_input,
                split="dev",
                limit=args.max_dev_screens,
                min_nodes=args.min_nodes,
            ),
            "test": _load_usable_graphs(
                args.test_input,
                split="test",
                limit=args.max_test_screens,
                min_nodes=args.min_nodes,
            ),
        }
        assert_graph_splits_disjoint(graph_splits)
        split_mode = "frozen_input_files"
    else:
        graph_splits = split_graphs_by_group(
            graphs,
            seed=args.seed,
            dev_percent=args.dev_percent,
            test_percent=args.test_percent,
        )
        split_mode = "deterministic_group_hash"
    rng = random.Random(args.seed)
    preview_pair = make_training_pair(
        graphs[0],
        rng=rng,
        config=augment_config,
        matcher_config=matcher_config,
    )
    if preview_pair is None:
        raise SystemExit(
            "Could not construct an augmented training pair from the first graph."
        )

    print(
        json.dumps(
            {
                "loaded_graphs": len(graphs),
                "numeric_feature_dim": len(preview_pair.features_a[0]),
                "relation_feature_dim": len(
                    preview_pair.encoded_a.relation_features[0][0]
                ),
                "first_graph_nodes": len(graphs[0].nodes),
                "first_pair_common_nodes": len(preview_pair.origin_ids),
                "first_pair_null_labels": sum(
                    target < 0
                    for target in (
                        *preview_pair.targets_a_to_b,
                        *preview_pair.targets_b_to_a,
                    )
                ),
                "split_counts": {
                    name: len(items) for name, items in graph_splits.items()
                },
                "split_mode": split_mode,
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    model, history = train_self_supervised_matcher(
        graph_splits["train"],
        model=pretrained_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seed=args.seed,
        device=args.device,
        augment_config=augment_config,
        matcher_config=matcher_config,
        progress_callback=lambda event: print(
            json.dumps({"training_progress": event}),
            flush=True,
        ),
        progress_interval=args.log_interval,
    )
    evaluation = {
        name: evaluate_self_supervised_matcher(
            model,
            items,
            seed=args.seed + offset,
            device=args.device,
            augment_config=augment_config,
            matcher_config=matcher_config,
        )
        for offset, (name, items) in enumerate(
            (("dev", graph_splits["dev"]), ("test", graph_splits["test"])),
            start=1,
        )
    }
    summary = {
        "inputs": args.input,
        "pretrained": str(args.pretrained.resolve()) if args.pretrained else None,
        "history": history,
        "evaluation": evaluation,
        "parameter_count": parameter_count(model),
        "matcher_config": matcher_config.__dict__,
        "augmentation": augment_config.__dict__,
        "split_counts": {name: len(items) for name, items in graph_splits.items()},
        "split_mode": split_mode,
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.output is not None:
        save_matcher_checkpoint(
            args.output,
            model,
            config=matcher_config,
            metadata=summary,
        )


def _graph_from_record_safe(record: Any, graph_id: str) -> UIGraph:
    if not isinstance(record, dict):
        raise ValueError(f"Expected dict record for {graph_id}")
    return graph_from_unified_record(
        record,
        graph_id=str(record.get("graph_id") or graph_id),
    )


def _load_usable_graphs(
    input_paths: list[str],
    *,
    split: str,
    limit: int,
    min_nodes: int = 4,
) -> list[UIGraph]:
    if limit < 0:
        raise ValueError("graph limits must be non-negative")
    if min_nodes < 2:
        raise ValueError("min_nodes must be at least two")
    graphs: list[UIGraph] = []
    for input_path in input_paths:
        for graph in iter_graphs(input_path, split=split):
            if len(graph.nodes) < min_nodes:
                continue
            graphs.append(graph)
            if limit and len(graphs) >= limit:
                return graphs
    return graphs


def _iter_hf_graphs(dataset_id: str, *, split: str) -> Iterable[UIGraph]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install `datasets` to stream HuggingFace datasets."
        ) from exc
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    for index, record in enumerate(dataset):
        try:
            yield graph_from_unified_record(
                record, graph_id=f"{dataset_id}:{split}:{index}"
            )
        except ValueError:
            continue


def _iter_parquet_graphs(path: Path) -> Iterable[UIGraph]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install `pyarrow` to read parquet MobileViews shards."
        ) from exc
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=256):
        for record in batch.to_pylist():
            try:
                yield graph_from_unified_record(
                    record, graph_id=f"{path.stem}:{row_index}"
                )
            except ValueError:
                pass
            row_index += 1


def split_graphs_by_group(
    graphs: list[UIGraph],
    *,
    seed: int,
    dev_percent: int = 10,
    test_percent: int = 10,
) -> dict[str, list[UIGraph]]:
    """Create deterministic app/group-disjoint self-supervised splits."""

    if dev_percent < 0 or test_percent < 0 or dev_percent + test_percent >= 100:
        raise ValueError(
            "dev_percent and test_percent must be non-negative and sum below 100"
        )
    splits = {"train": [], "dev": [], "test": []}
    for graph in graphs:
        group = _graph_group(graph)
        digest = hashlib.blake2b(
            f"{seed}:{group}".encode("utf-8"),
            digest_size=8,
        ).digest()
        bucket = int.from_bytes(digest, "big") % 100
        if bucket < test_percent:
            split = "test"
        elif bucket < test_percent + dev_percent:
            split = "dev"
        else:
            split = "train"
        splits[split].append(graph)
    if graphs and not splits["train"]:
        splits["train"].append(graphs[0])
        for name in ("dev", "test"):
            if graphs[0] in splits[name]:
                splits[name].remove(graphs[0])
    return splits


def assert_graph_splits_disjoint(splits: dict[str, list[UIGraph]]) -> None:
    """Reject package/app leakage across externally materialized graph files."""

    groups = {
        name: {_graph_group(graph) for graph in graphs}
        for name, graphs in splits.items()
    }
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = groups.get(first, set()).intersection(groups.get(second, set()))
        if overlap:
            raise ValueError(
                f"graph group leakage between {first} and {second}: {sorted(overlap)}"
            )


def _graph_group(graph: UIGraph) -> str:
    metadata = graph.metadata or {}
    for key in ("split_group", "app", "package", "package_name", "app_id", "apk"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return graph.graph_id


if __name__ == "__main__":
    main()
