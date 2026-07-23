#!/usr/bin/env python3
"""Build a compact, checksummed training bundle from canonical UI queries."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from omnitransfer.benchmark import split_queries_by_group
from omnitransfer.importers import load_queries, write_queries
from omnitransfer.schema import Candidate, Query


_PATH_KEYS = {
    "source_xml_path",
    "source_screenshot_path",
    "target_xml_path",
    "target_screenshot_path",
    "xml_path",
    "screenshot_path",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-queries", type=int, default=100)
    parser.add_argument("--dev-queries", type=int, default=0)
    parser.add_argument("--test-queries", type=int, default=20)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--runtime-prefix")
    parser.add_argument("--image-long-side", type=int, default=0)
    args = parser.parse_args()
    if args.image_long_side < 0:
        raise SystemExit("--image-long-side must be non-negative")

    queries = load_queries(args.input)
    splits = split_queries_by_group(queries, seed=args.seed)
    selected_train = splits["train"] if args.all_queries else _take(
        splits["train"],
        args.train_queries,
    )
    selected_dev = splits["dev"] if args.all_queries else _take(
        splits["dev"],
        args.dev_queries,
    )
    selected_test = splits["test"] if args.all_queries else _take(
        splits["test"],
        args.test_queries,
    )
    selected = [
        *selected_train,
        *selected_dev,
        *selected_test,
    ]
    if not selected:
        raise SystemExit("No queries selected for compact bundle")
    output = args.output.resolve()
    runtime_prefix = args.runtime_prefix or f"{output.name}/assets"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    rewritten = [
        _rewrite_query(
            query,
            assets=assets,
            runtime_prefix=runtime_prefix,
            copied=copied,
            image_long_side=args.image_long_side,
        )
        for query in selected
    ]
    train_end = len(selected_train)
    dev_end = train_end + len(selected_dev)
    rewritten_splits = {
        "train": rewritten[:train_end],
        "dev": rewritten[train_end:dev_end],
        "test": rewritten[dev_end:],
    }
    query_path = output / "queries.jsonl"
    write_queries(rewritten, query_path)
    split_manifest_path = output / "splits.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "omnitransfer_app_disjoint_splits_v1",
                "split_seed": args.seed,
                "splits": {
                    name: [query.query_id for query in rows]
                    for name, rows in rewritten_splits.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pretrain_graphs = {
        name: _write_pretrain_graphs(
            originals,
            rewritten_splits[name],
            output=output,
            split=name,
        )
        for name, originals in (
            ("train", selected_train),
            ("dev", selected_dev),
            ("test", selected_test),
        )
    }
    manifest = {
        "schema_version": "omnitransfer_compact_training_bundle_v1",
        "source": str(args.input.resolve()),
        "source_sha256": _sha256(args.input.resolve()),
        "query_sha256": _sha256(query_path),
        "split_manifest_sha256": _sha256(split_manifest_path),
        "seed": args.seed,
        "runtime_prefix": runtime_prefix,
        "image_long_side": args.image_long_side,
        "selected_counts": {
            "train": len(selected_train),
            "dev": len(selected_dev),
            "test": len(selected_test),
            "total": len(rewritten),
        },
        "pretrain_graphs": pretrain_graphs,
        "assets": [
            {
                "source": source,
                "bundled": bundled,
                "sha256": _sha256(output.parent / bundled),
            }
            for source, bundled in sorted(copied.items())
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "queries": len(rewritten),
                "assets": len(copied),
                "pretrain_graphs": {
                    name: rows["count"] for name, rows in pretrain_graphs.items()
                },
            }
        )
    )


def _rewrite_query(
    query: Query,
    *,
    assets: Path,
    runtime_prefix: str,
    copied: dict[str, str],
    image_long_side: int,
) -> Query:
    metadata = _with_coordinate_sizes(query)
    return replace(
        query,
        source=_rewrite_mapping(
            query.source,
            assets=assets,
            runtime_prefix=runtime_prefix,
            copied=copied,
            image_long_side=image_long_side,
        ),
        target_candidates=tuple(
            Candidate(
                candidate_id=candidate.candidate_id,
                bbox=candidate.bbox,
                metadata=_rewrite_mapping(
                    candidate.metadata,
                    assets=assets,
                    runtime_prefix=runtime_prefix,
                    copied=copied,
                    image_long_side=image_long_side,
                ),
            )
            for candidate in query.target_candidates
        ),
        metadata=_rewrite_mapping(
            metadata,
            assets=assets,
            runtime_prefix=runtime_prefix,
            copied=copied,
            image_long_side=image_long_side,
        ),
    )


def _with_coordinate_sizes(query: Query) -> dict[str, Any]:
    metadata = dict(query.metadata)
    if metadata.get("coordinate_space") == "xml_relative_0_1":
        metadata.pop("source_coordinate_width", None)
        metadata.pop("source_coordinate_height", None)
        metadata.pop("target_coordinate_width", None)
        metadata.pop("target_coordinate_height", None)
        return metadata
    source_metadata = query.source.get("metadata") or {}
    paths = {
        "source": metadata.get("source_screenshot_path")
        or source_metadata.get("source_screenshot_path"),
        "target": metadata.get("target_screenshot_path"),
    }
    for prefix, path_value in paths.items():
        if not isinstance(path_value, str) or not path_value:
            continue
        size = _image_size(path_value)
        if size is None:
            continue
        metadata[f"{prefix}_coordinate_width"] = size[0]
        metadata[f"{prefix}_coordinate_height"] = size[1]
    return metadata


def _rewrite_mapping(
    mapping: dict[str, Any],
    *,
    assets: Path,
    runtime_prefix: str,
    copied: dict[str, str],
    image_long_side: int,
) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _PATH_KEYS and isinstance(value, str) and value:
            rewritten[key] = _copy_asset(
                value,
                assets=assets,
                runtime_prefix=runtime_prefix,
                copied=copied,
                image_long_side=image_long_side,
                resize_image=key.endswith("screenshot_path"),
            )
        elif isinstance(value, dict):
            rewritten[key] = _rewrite_mapping(
                value,
                assets=assets,
                runtime_prefix=runtime_prefix,
                copied=copied,
                image_long_side=image_long_side,
            )
        else:
            rewritten[key] = value
    return rewritten


def _copy_asset(
    source_value: str,
    *,
    assets: Path,
    runtime_prefix: str,
    copied: dict[str, str],
    image_long_side: int,
    resize_image: bool,
) -> str:
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"referenced UI asset is missing: {source_value}")
    source_key = str(source)
    if source_key in copied:
        return copied[source_key]
    resize_long_side = image_long_side if resize_image else 0
    digest = hashlib.sha256(
        f"{source_key}:{resize_long_side}".encode("utf-8")
    ).hexdigest()[:16]
    suffix = ".png" if resize_long_side else source.suffix.lower()
    destination = assets / f"{digest}{suffix}"
    if resize_long_side:
        _resize_image(source, destination, long_side=resize_long_side)
    else:
        shutil.copy2(source, destination)
    bundled = f"{runtime_prefix.rstrip('/')}/{destination.name}"
    copied[source_key] = bundled
    return bundled


def _resize_image(source: Path, destination: Path, *, long_side: int) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Resizing bundle screenshots requires Pillow. Install omnitransfer[train]."
        ) from exc
    with Image.open(source) as image:
        image = image.convert("RGB")
        if max(image.size) > long_side:
            scale = long_side / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        image.save(destination, format="PNG", optimize=True)


@lru_cache(maxsize=4096)
def _image_size(path_value: str) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Reading bundle image sizes requires Pillow. Install omnitransfer[train]."
        ) from exc
    try:
        with Image.open(Path(path_value).expanduser().resolve()) as image:
            return int(image.width), int(image.height)
    except (FileNotFoundError, OSError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_pretrain_graphs(
    originals: list[Query],
    rewritten: list[Query],
    *,
    output: Path,
    split: str,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for original, bundled in zip(originals, rewritten, strict=True):
        app = str(original.metadata.get("app") or "unknown")
        source_platform = str(
            (original.source.get("metadata") or {}).get("platform") or "ios"
        )
        pairs = (
            (
                "source",
                source_platform,
                bundled.metadata.get("source_xml_path"),
                bundled.metadata.get("source_screenshot_path"),
            ),
            (
                "target",
                "android",
                bundled.metadata.get("target_xml_path"),
                bundled.metadata.get("target_screenshot_path"),
            ),
        )
        for side, platform, xml_path, screenshot_path in pairs:
            if not isinstance(xml_path, str) or not xml_path:
                continue
            if xml_path in records:
                continue
            xml_asset = output.parent / xml_path
            if not xml_asset.is_file():
                raise FileNotFoundError(f"bundled XML is missing: {xml_asset}")
            graph_digest = hashlib.sha256(xml_path.encode("utf-8")).hexdigest()[:16]
            records[xml_path] = {
                "source_format": f"{platform}_xml",
                "screen_id": f"{app}:{side}:{graph_digest}",
                "app": app,
                "platform": platform,
                "dataset": "vision_widget_mapping",
                "split": split,
                "xml": xml_asset.read_text(encoding="utf-8"),
                "screenshot_path": screenshot_path or "",
            }
    path = output / f"pretrain_graphs.{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "count": len(records),
        "path": path.name,
        "sha256": _sha256(path),
    }


def _take(queries: list[Query], limit: int) -> list[Query]:
    if limit < 0:
        raise ValueError("query limits must be non-negative")
    return queries[:limit] if limit else []


if __name__ == "__main__":
    main()
