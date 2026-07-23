#!/usr/bin/env python3
"""Import OS-Atlas mobile grounding data into Unified UIGraph files."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

from omnitransfer.mobileviews import split_name_for_group
from omnitransfer.os_atlas import (
    graph_from_os_atlas_record,
    materialize_os_atlas_image,
)
from omnitransfer.ui_graph import graph_to_record


DEFAULT_REVISION = "e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert grouped OS-Atlas mobile grounding screenshots and boxes "
            "to lossless 384px assets and Unified UIGraph JSONL files."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset-id", default="OS-Copilot/OS-Atlas-data")
    parser.add_argument("--subset", default="android_world")
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--image-long-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--min-nodes", type=int, default=2)
    args = parser.parse_args()
    if args.max_screens < 0:
        raise SystemExit("--max-screens must be non-negative; zero imports all screens")
    if args.image_long_side <= 0 or args.min_nodes <= 0:
        raise SystemExit("image-long-side and min-nodes must be positive")

    annotations = args.annotations.expanduser().resolve()
    image_root = args.images.expanduser().resolve()
    if not annotations.is_file():
        raise SystemExit(f"OS-Atlas annotations are missing: {annotations}")
    if not image_root.is_dir():
        raise SystemExit(f"OS-Atlas image root is missing: {image_root}")
    records = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("OS-Atlas annotations must contain a top-level list")
    image_paths, duplicate_basenames = index_images(image_root)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    seen_images: set[str] = set()
    node_total = 0
    node_max = 0
    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(
                (output / f"graphs.{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in ("train", "dev", "test")
        }
        for row_index, record in enumerate(records):
            if args.max_screens and sum(split_counts.values()) >= args.max_screens:
                break
            if not isinstance(record, dict):
                skipped["invalid_record"] += 1
                continue
            filename = str(record.get("img_filename") or "").strip()
            source_image = image_paths.get(filename)
            if not filename or source_image is None:
                skipped["missing_image"] += 1
                continue
            image_digest = _sha256(source_image)
            if image_digest in seen_images:
                skipped["duplicate_image"] += 1
                continue
            split = split_name_for_group(
                filename,
                seed=args.seed,
                dev_percent=args.dev_percent,
                test_percent=args.test_percent,
            )
            relative_image = Path("images") / split / filename
            destination = output / relative_image
            try:
                image_size = materialize_os_atlas_image(
                    source_image,
                    destination,
                    long_side=args.image_long_side,
                )
                graph = graph_from_os_atlas_record(
                    record,
                    graph_id=f"os_atlas:{args.subset}:{row_index}",
                    image_size=image_size,
                    screenshot_path=str(destination),
                    dataset=args.dataset_id,
                    subset=args.subset,
                )
            except (OSError, TypeError, ValueError):
                destination.unlink(missing_ok=True)
                skipped["invalid_screen"] += 1
                continue
            if len(graph.nodes) < args.min_nodes:
                destination.unlink(missing_ok=True)
                skipped["too_few_nodes"] += 1
                continue
            outputs[split].write(json.dumps(graph_to_record(graph), ensure_ascii=False) + "\n")
            seen_images.add(image_digest)
            split_counts[split] += 1
            node_total += len(graph.nodes)
            node_max = max(node_max, len(graph.nodes))
            if sum(split_counts.values()) % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "accepted": sum(split_counts.values()),
                            "rows_scanned": row_index + 1,
                        }
                    ),
                    flush=True,
                )

    accepted = sum(split_counts.values())
    manifest = {
        "schema_version": "omnitransfer_os_atlas_mobile_bundle_v1",
        "dataset": args.dataset_id,
        "subset": args.subset,
        "source_revision": args.source_revision,
        "license": "Apache-2.0",
        "annotation_path": str(annotations),
        "annotation_bytes": annotations.stat().st_size,
        "annotation_sha256": _sha256(annotations),
        "source_records": len(records),
        "indexed_images": len(image_paths),
        "duplicate_image_basenames": duplicate_basenames,
        "accepted_screens": accepted,
        "split_counts": dict(sorted(split_counts.items())),
        "split_unit": "image_uuid",
        "split_warning": "source release does not expose app ids; held-out splits are diagnostics, not app-disjoint evaluation",
        "skipped": dict(sorted(skipped.items())),
        "node_count": node_total,
        "average_nodes": node_total / accepted if accepted else 0.0,
        "max_nodes": node_max,
        "image_long_side": args.image_long_side,
        "lossless_image_format": "png",
        "policies": {
            "origin_id": "augmentation_label_only",
            "grounding_boxes": "model_input_geometry_not_match_rule",
            "correspondence": "same_element_under_independent_train_only_augmentations",
            "formal_evaluation": "frozen_external_benchmarks_only",
        },
    }
    temporary = output / "manifest.json.part"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False))


def index_images(root: Path) -> tuple[dict[str, Path], int]:
    """Index unique image basenames without guessing ambiguous paths."""

    paths: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if path.name in paths:
            ambiguous.add(path.name)
        else:
            paths[path.name] = path
    for name in ambiguous:
        paths.pop(name, None)
    return paths, len(ambiguous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
