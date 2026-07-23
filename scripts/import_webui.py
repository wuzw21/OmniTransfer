#!/usr/bin/env python3
"""Stream WebUI element parquet shards into Unified UIGraph JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.ui_graph import graph_to_record
from omnitransfer.webui import graph_from_webui_record, materialize_webui_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WebUI screenshot and content-box parquet rows into one "
            "lossless-image Unified UIGraph stream."
        )
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default="biglab/webui-70k-elements")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-long-side", type=int, default=384)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--min-area", type=float, default=4.0)
    args = parser.parse_args()
    if args.max_screens < 0 or args.start_row < 0:
        raise SystemExit("screen limits and --start-row must be non-negative")
    if args.batch_size <= 0 or args.image_long_side <= 0 or args.min_nodes <= 0:
        raise SystemExit("batch size, image size, and minimum nodes must be positive")
    if args.min_area < 0:
        raise SystemExit("--min-area must be non-negative")

    inputs = [path.expanduser().resolve() for path in args.input]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"WebUI parquet inputs are missing: {missing}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    graphs_path = output / f"graphs.{args.split}.jsonl"
    skipped: Counter[str] = Counter()
    accepted = rows_scanned = node_total = node_max = 0
    viewport_counts: Counter[str] = Counter()

    with graphs_path.open("w", encoding="utf-8") as graphs_file:
        stop = False
        for input_path in inputs:
            shard = input_path.stem
            for row_index, record in iter_webui_rows(
                input_path,
                batch_size=args.batch_size,
                start_row=args.start_row,
            ):
                rows_scanned += 1
                graph_id = f"webui:{args.split}:{shard}:{row_index}"
                image_relative = Path("images") / shard / f"{row_index:08d}.png"
                image_path = output / image_relative
                try:
                    image_size = materialize_webui_image(
                        record.get("image"),
                        image_path,
                        long_side=args.image_long_side,
                    )
                except (OSError, TypeError, ValueError):
                    skipped["invalid_image"] += 1
                    continue
                try:
                    graph = graph_from_webui_record(
                        record,
                        graph_id=graph_id,
                        image_size=image_size,
                        screenshot_path=str(image_path),
                        dataset=args.dataset_id,
                        split=args.split,
                        min_area=args.min_area,
                    )
                except (TypeError, ValueError):
                    image_path.unlink(missing_ok=True)
                    skipped["invalid_elements"] += 1
                    continue
                if len(graph.nodes) < args.min_nodes:
                    image_path.unlink(missing_ok=True)
                    skipped["too_few_nodes"] += 1
                    continue
                graphs_file.write(
                    json.dumps(graph_to_record(graph), ensure_ascii=False) + "\n"
                )
                accepted += 1
                node_total += len(graph.nodes)
                node_max = max(node_max, len(graph.nodes))
                viewport_counts[str(graph.metadata.get("viewport") or "unknown")] += 1
                if accepted % 1000 == 0:
                    print(
                        json.dumps(
                            {"accepted": accepted, "rows_scanned": rows_scanned},
                        ),
                        flush=True,
                    )
                if args.max_screens and accepted >= args.max_screens:
                    stop = True
                    break
            if stop:
                break

    manifest = {
        "schema_version": "omnitransfer_webui_bundle_v1",
        "dataset": args.dataset_id,
        "source_revision": args.source_revision,
        "split": args.split,
        "official_split_unit": "domain",
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in inputs
        ],
        "adapter": "image + labels + contentBoxes -> Unified UIGraph",
        "image_long_side": args.image_long_side,
        "lossless_image_format": "png",
        "rows_scanned": rows_scanned,
        "accepted_screens": accepted,
        "skipped": dict(sorted(skipped.items())),
        "viewport_counts": dict(sorted(viewport_counts.items())),
        "node_count": node_total,
        "average_nodes": node_total / accepted if accepted else 0.0,
        "max_nodes": node_max,
        "policies": {
            "copyright_terms": "explicit_researcher_acceptance_required",
            "offscreen_or_unbounded_elements": "exclude",
            "origin_id": "augmentation_label_only",
            "element_boxes": "model_input_geometry_not_match_rule",
            "formal_evaluation": "official_val_and_test_never_train",
        },
    }
    temporary = output / "manifest.json.part"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False))


def iter_webui_rows(
    path: Path,
    *,
    batch_size: int,
    start_row: int = 0,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield official WebUI columns without loading a shard into memory."""

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "Reading WebUI parquet requires PyArrow. Install omnitransfer[train]."
        ) from exc
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["image", "labels", "contentBoxes", "key_name"],
    ):
        for record in batch.to_pylist():
            if row_index >= start_row:
                yield row_index, record
            row_index += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
