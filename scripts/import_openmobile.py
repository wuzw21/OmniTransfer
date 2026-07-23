#!/usr/bin/env python3
"""Import Uni-GUI-OpenMobile into package-disjoint Unified UIGraph files."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from typing import Any

from omnitransfer.openmobile import (
    attach_openmobile_image,
    graph_from_openmobile_step,
    infer_openmobile_package,
    materialize_openmobile_image,
)
from omnitransfer.ui_graph import UIGraph, graph_to_record


DEFAULT_REVISION = "774eb20f97724bf64faa9e66b1572f05b1688e38"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Uni-GUI-OpenMobile screenshots and per-step UI elements "
            "to lossless 384px assets and package-disjoint UIGraph JSONL files."
        )
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--image-long-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--allow-geometry-mismatch", action="store_true")
    args = parser.parse_args()
    if args.max_screens < 0:
        raise SystemExit("--max-screens must be non-negative; zero imports all screens")
    if args.image_long_side <= 0:
        raise SystemExit("--image-long-side must be positive")
    if args.min_nodes <= 0:
        raise SystemExit("--min-nodes must be positive")

    inputs = [path.expanduser().resolve() for path in args.input]
    missing = [str(path) for path in inputs if not path.is_dir()]
    if missing:
        raise SystemExit(f"OpenMobile roots are missing: {missing}")
    metadata_files = sorted(
        path for root in inputs for path in root.rglob("metadata.json")
    )
    if not metadata_files:
        raise SystemExit("No OpenMobile metadata.json files found")

    skipped: Counter[str] = Counter()
    trajectories: list[tuple[Path, dict[str, Any], dict[str, Any], str]] = []
    for metadata_path in metadata_files:
        metadata = _read_json(metadata_path)
        if metadata is None:
            skipped["invalid_metadata"] += 1
            continue
        steps = metadata.get("steps")
        if not isinstance(steps, list):
            skipped["missing_steps"] += 1
            continue
        task = _read_json(metadata_path.with_name("task.json")) or {}
        package = infer_openmobile_package(task, steps)
        if not package:
            skipped["missing_package"] += len(steps)
            continue
        trajectories.append((metadata_path, metadata, task, package))
    package_splits = assign_package_splits(
        {package for _, _, _, package in trajectories},
        seed=args.seed,
        dev_percent=args.dev_percent,
        test_percent=args.test_percent,
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    split_packages = {name: set() for name in ("train", "dev", "test")}
    seen_images: set[str] = set()
    node_total = 0
    node_max = 0
    trajectories_with_screens: set[str] = set()

    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(
                (output / f"graphs.{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in ("train", "dev", "test")
        }
        stop = False
        for metadata_path, metadata, task, package in trajectories:
            steps = metadata.get("steps")
            assert isinstance(steps, list)
            episode_id = str(task.get("episode_id") or metadata_path.parent.name)
            app = str(task.get("app") or package)
            split = package_splits[package]
            actions = {
                int(action.get("step") or 0): action
                for action in task.get("data") or ()
                if isinstance(action, dict)
            }
            for step in steps:
                if not isinstance(step, dict):
                    skipped["invalid_step"] += 1
                    continue
                step_index = int(step.get("step") or 0)
                source_image = metadata_path.parent / f"screenshot_step{step_index}.png"
                if not source_image.is_file():
                    skipped["missing_image"] += 1
                    continue
                image_digest = _sha256(source_image)
                if image_digest in seen_images:
                    skipped["duplicate_image"] += 1
                    continue
                graph_id = f"openmobile:{episode_id}:{step_index}"
                try:
                    graph = graph_from_openmobile_step(
                        step,
                        graph_id=graph_id,
                        package=package,
                        app=app,
                        episode_id=episode_id,
                        action=actions.get(step_index),
                    )
                except (TypeError, ValueError):
                    skipped["invalid_ui_elements"] += 1
                    continue
                if len(graph.nodes) < args.min_nodes:
                    skipped["too_few_nodes"] += 1
                    continue
                image_relative = Path("images") / episode_id / f"{step_index:04d}.png"
                image_path = output / image_relative
                try:
                    image_size = materialize_openmobile_image(
                        source_image,
                        image_path,
                        long_side=args.image_long_side,
                    )
                except (OSError, TypeError, ValueError):
                    skipped["invalid_image"] += 1
                    continue
                graph = attach_openmobile_image(
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
                graph = _with_split(
                    graph,
                    split=split,
                    source_episode=metadata_path.parent.name,
                    image_sha256=image_digest,
                )
                outputs[split].write(
                    json.dumps(graph_to_record(graph), ensure_ascii=False) + "\n"
                )
                seen_images.add(image_digest)
                split_counts[split] += 1
                split_packages[split].add(package)
                trajectories_with_screens.add(episode_id)
                node_total += len(graph.nodes)
                node_max = max(node_max, len(graph.nodes))
                accepted = sum(split_counts.values())
                if accepted % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "accepted": accepted,
                                "trajectories": len(trajectories_with_screens),
                                "split_counts": dict(split_counts),
                            }
                        ),
                        flush=True,
                    )
                if args.max_screens and accepted >= args.max_screens:
                    stop = True
                    break
            if stop:
                break

    _assert_disjoint(split_packages)
    accepted = sum(split_counts.values())
    manifest = {
        "schema_version": "omnitransfer_unigui_openmobile_bundle_v1",
        "dataset": "UI-MOPD/Uni-GUI-OpenMobile",
        "license": "Apache-2.0",
        "source_revision": args.source_revision,
        "inputs": [str(path) for path in inputs],
        "adapter": "metadata.steps.ui_elements + screenshot -> Unified UIGraph",
        "split_unit": "app_package",
        "split_seed": args.seed,
        "split_percent": {
            "train": 100 - args.dev_percent - args.test_percent,
            "dev": args.dev_percent,
            "test": args.test_percent,
        },
        "split_assignment": "deterministic_package_rank_with_exact_group_counts",
        "image_long_side": args.image_long_side,
        "lossless_image_format": "png",
        "metadata_files_scanned": len(metadata_files),
        "accepted_trajectories": len(trajectories_with_screens),
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
            "duplicate_screens": "exclude_by_source_sha256",
            "unknown_package": "exclude",
            "geometry_mismatch": (
                "include" if args.allow_geometry_mismatch else "exclude"
            ),
            "origin_id": "label_only",
            "action_bbox": "label_metadata_only",
            "formal_evaluation": "never_train",
        },
    }
    temporary = output / "manifest.json.part"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_split(
    graph: UIGraph,
    *,
    split: str,
    source_episode: str,
    image_sha256: str,
) -> UIGraph:
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "split": split,
            "source_episode": source_episode,
            "source_image_sha256": image_sha256,
        },
    )


def _assert_disjoint(split_packages: dict[str, set[str]]) -> None:
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_packages[first].intersection(split_packages[second])
        if overlap:
            raise RuntimeError(
                f"package leakage between {first} and {second}: {sorted(overlap)}"
            )


def assign_package_splits(
    packages: set[str],
    *,
    seed: int,
    dev_percent: int,
    test_percent: int,
) -> dict[str, str]:
    """Assign exact package counts to deterministic disjoint splits."""

    if dev_percent < 0 or test_percent < 0 or dev_percent + test_percent >= 100:
        raise ValueError("dev/test percentages must be non-negative and sum below 100")
    if not packages:
        return {}
    ranked = sorted(
        packages,
        key=lambda package: (
            hashlib.blake2b(
                f"{seed}:{package}".encode("utf-8"),
                digest_size=8,
            ).digest(),
            package,
        ),
    )
    available_holdouts = max(0, len(ranked) - 1)
    test_count = min(
        available_holdouts,
        max(1, round(len(ranked) * test_percent / 100)) if test_percent else 0,
    )
    available_holdouts -= test_count
    dev_count = min(
        available_holdouts,
        max(1, round(len(ranked) * dev_percent / 100)) if dev_percent else 0,
    )
    assignments = {package: "train" for package in ranked}
    for package in ranked[:test_count]:
        assignments[package] = "test"
    for package in ranked[test_count : test_count + dev_count]:
        assignments[package] = "dev"
    return assignments


if __name__ == "__main__":
    main()
