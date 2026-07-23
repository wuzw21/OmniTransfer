#!/usr/bin/env python3
"""Stream official MobileViews parquet shards into app-disjoint UIGraph files."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.mobileviews import (
    attach_mobileviews_image,
    graph_from_mobileviews_record,
    materialize_mobileviews_image,
    split_name_for_group,
)
from omnitransfer.ui_graph import UIGraph, graph_to_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert image_content/json_content MobileViews parquet rows to "
            "lossless 384px assets and app-disjoint Unified UIGraph JSONL files."
        )
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-screens", type=int, default=50_000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-long-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--allow-geometry-mismatch", action="store_true")
    args = parser.parse_args()
    if args.max_screens <= 0:
        raise SystemExit("--max-screens must be positive")
    if args.start_row < 0:
        raise SystemExit("--start-row must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.image_long_side <= 0:
        raise SystemExit("--image-long-side must be positive")
    if args.min_nodes < 1:
        raise SystemExit("--min-nodes must be positive")

    inputs = [path.expanduser().resolve() for path in args.input]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"MobileViews parquet inputs are missing: {missing}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    split_packages: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }
    node_total = 0
    node_max = 0
    rows_scanned = 0

    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(
                (output / f"graphs.{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in ("train", "dev", "test")
        }
        for input_path in inputs:
            shard = input_path.stem
            for row_index, record in iter_mobileviews_rows(
                input_path,
                batch_size=args.batch_size,
                start_row=args.start_row,
            ):
                rows_scanned += 1
                graph_id = f"mobileviews:{shard}:{row_index}"
                try:
                    graph = graph_from_mobileviews_record(record, graph_id=graph_id)
                except (TypeError, ValueError, json.JSONDecodeError):
                    skipped["invalid_hierarchy"] += 1
                    continue
                package = str(graph.metadata.get("package") or "")
                if not package:
                    skipped["missing_package"] += 1
                    continue
                if len(graph.nodes) < args.min_nodes:
                    skipped["too_few_nodes"] += 1
                    continue
                image_relative = Path("images") / shard / f"{row_index:08d}.png"
                image_path = output / image_relative
                try:
                    image_size = materialize_mobileviews_image(
                        record.get("image_content"),
                        image_path,
                        long_side=args.image_long_side,
                    )
                except (OSError, TypeError, ValueError):
                    skipped["invalid_image"] += 1
                    continue
                graph = attach_mobileviews_image(
                    graph,
                    screenshot_path=str(image_path),
                    image_size=image_size,
                )
                if (
                    graph.metadata.get("geometry_mismatch")
                    and not args.allow_geometry_mismatch
                ):
                    image_path.unlink(missing_ok=True)
                    skipped["geometry_mismatch"] += 1
                    continue
                split = split_name_for_group(
                    package,
                    seed=args.seed,
                    dev_percent=args.dev_percent,
                    test_percent=args.test_percent,
                )
                graph = _with_split(graph, split=split, source_row=row_index)
                outputs[split].write(json.dumps(graph_to_record(graph), ensure_ascii=False) + "\n")
                split_counts[split] += 1
                split_packages[split].add(package)
                node_total += len(graph.nodes)
                node_max = max(node_max, len(graph.nodes))
                accepted = sum(split_counts.values())
                if accepted % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "accepted": accepted,
                                "rows_scanned": rows_scanned,
                                "split_counts": dict(split_counts),
                            }
                        ),
                        flush=True,
                    )
                if accepted >= args.max_screens:
                    break
            if sum(split_counts.values()) >= args.max_screens:
                break

    _assert_disjoint(split_packages)
    accepted = sum(split_counts.values())
    manifest = {
        "schema_version": "omnitransfer_mobileviews_bundle_v1",
        "dataset": "mllmTeam/MobileViews",
        "source_revision": "MobileViews-600K",
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sidecar_sha256(path),
            }
            for path in inputs
        ],
        "adapter": "image_content/json_content -> Unified UIGraph",
        "split_unit": "package",
        "split_seed": args.seed,
        "split_percent": {
            "train": 100 - args.dev_percent - args.test_percent,
            "dev": args.dev_percent,
            "test": args.test_percent,
        },
        "image_long_side": args.image_long_side,
        "lossless_image_format": "png",
        "rows_scanned": rows_scanned,
        "accepted_screens": accepted,
        "skipped": dict(sorted(skipped.items())),
        "split_counts": {
            split: split_counts[split] for split in ("train", "dev", "test")
        },
        "split_package_counts": {
            split: len(split_packages[split]) for split in ("train", "dev", "test")
        },
        "node_count": node_total,
        "average_nodes": node_total / accepted if accepted else 0.0,
        "max_nodes": node_max,
        "policies": {
            "unknown_package": "exclude",
            "geometry_mismatch": (
                "include" if args.allow_geometry_mismatch else "exclude"
            ),
            "origin_id": "label_only",
            "formal_evaluation": "never_train",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def iter_mobileviews_rows(
    path: Path,
    *,
    batch_size: int,
    start_row: int = 0,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield only the two official columns without loading a shard into memory."""

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "Reading MobileViews parquet requires PyArrow. Install omnitransfer[train]."
        ) from exc
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["image_content", "json_content"],
    ):
        for record in batch.to_pylist():
            if row_index >= start_row:
                yield row_index, record
            row_index += 1


def _with_split(graph: UIGraph, *, split: str, source_row: int) -> UIGraph:
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "split": split,
            "source_row": source_row,
        },
    )


def _assert_disjoint(split_packages: dict[str, set[str]]) -> None:
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_packages[first].intersection(split_packages[second])
        if overlap:
            raise RuntimeError(f"package leakage between {first} and {second}: {sorted(overlap)}")


def _sidecar_sha256(path: Path) -> str | None:
    for candidate in (
        path.with_name(path.name + ".sha256"),
        path.with_suffix(".sha256"),
    ):
        if not candidate.is_file():
            continue
        value = candidate.read_text(encoding="utf-8").strip().split()
        if value and len(value[0]) == 64:
            return value[0]
    return None


if __name__ == "__main__":
    main()
