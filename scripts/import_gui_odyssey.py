#!/usr/bin/env python3
"""Import GUIOdyssey annotations into weak action-grounding UIGraph files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from omnitransfer.gui_odyssey import graph_from_gui_odyssey_step
from omnitransfer.ui_graph import graph_to_record


DEFAULT_REPO_ID = "hflqf88888/GUIOdyssey"
DEFAULT_REVISION = "61632d0f3f4d51d7e9561ce4f84347dd06b2019d"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GUIOdyssey CLICK steps with SAM2 boxes into weak pseudo "
            "UIGraph records for matcher representation pretraining."
        )
    )
    parser.add_argument("--annotations", type=Path, nargs="*", default=())
    parser.add_argument("--hf-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_episodes < 0 or args.max_screens < 0:
        raise SystemExit("limits must be non-negative")
    if args.dev_percent < 0 or args.test_percent < 0 or args.dev_percent + args.test_percent >= 100:
        raise SystemExit("dev/test percentages must be non-negative and sum below 100")

    episodes = _load_episodes(
        annotation_paths=tuple(args.annotations),
        repo_id=args.hf_repo_id,
        revision=args.revision,
        max_episodes=args.max_episodes,
    )
    output = args.output.expanduser().resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    split_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    node_total = 0
    accepted = 0
    writers = {}
    try:
        if not args.dry_run:
            writers = {
                split: (output / f"graphs.{split}.jsonl").open("w", encoding="utf-8")
                for split in ("train", "dev", "test")
            }
        for episode in episodes:
            split = _split_for_episode(
                episode,
                seed=args.seed,
                dev_percent=args.dev_percent,
                test_percent=args.test_percent,
            )
            for step in episode.get("steps") or ():
                if not isinstance(step, dict):
                    skipped["invalid_step"] += 1
                    continue
                try:
                    graph = graph_from_gui_odyssey_step(episode, step)
                except ValueError as exc:
                    skipped[str(exc)] += 1
                    continue
                graph = _with_split(graph, split=split)
                accepted += 1
                split_counts[split] += 1
                device_counts[str(graph.metadata.get("device_name") or "unknown")] += 1
                node_total += len(graph.nodes)
                if not args.dry_run:
                    writers[split].write(json.dumps(graph_to_record(graph), ensure_ascii=False) + "\n")
                if args.max_screens and accepted >= args.max_screens:
                    break
            if args.max_screens and accepted >= args.max_screens:
                break
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        "schema_version": "omnitransfer_guiodyssey_weak_bundle_v1",
        "dataset": "hflqf88888/GUIOdyssey",
        "license": "cc-by-4.0",
        "source_revision": args.revision,
        "adapter": "annotation.steps.sam2_bbox + action point -> weak pseudo UIGraph",
        "split_unit": "episode_id",
        "split_seed": args.seed,
        "split_percent": {
            "train": 100 - args.dev_percent - args.test_percent,
            "dev": args.dev_percent,
            "test": args.test_percent,
        },
        "episodes_loaded": len(episodes),
        "accepted_screens": accepted,
        "split_counts": {split: split_counts[split] for split in ("train", "dev", "test")},
        "device_counts": dict(sorted(device_counts.items())),
        "average_nodes": node_total / accepted if accepted else 0.0,
        "skipped": dict(sorted(skipped.items())),
        "policies": {
            "ui_tree_absent": "pseudo_regions_plus_sam2_action_target_only",
            "action_coordinates": "label_metadata_only_never_runtime_fallback",
            "sam2_bbox": "teacher_grounding_label_for_representation_pretraining",
            "formal_correspondence_eval": "not_gold_widget_mapping",
        },
    }
    if not args.dry_run:
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False))


def _load_episodes(
    *,
    annotation_paths: tuple[Path, ...],
    repo_id: str,
    revision: str,
    max_episodes: int,
) -> list[dict[str, Any]]:
    if annotation_paths:
        episodes = []
        for path in annotation_paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                episodes.append(value)
            elif isinstance(value, list):
                episodes.extend(item for item in value if isinstance(item, dict) and "steps" in item)
        return episodes[:max_episodes] if max_episodes else episodes

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install huggingface_hub to import GUIOdyssey from HF") from exc
    index_path = Path(
        hf_hub_download(
            repo_id,
            "all_annot.json",
            repo_type="dataset",
            revision=revision,
        )
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, list):
        raise ValueError("GUIOdyssey all_annot.json must be a list")
    selected = index[:max_episodes] if max_episodes else index
    episodes = []
    for row in selected:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            continue
        path = Path(
            hf_hub_download(
                repo_id,
                f"annotations/{episode_id}.json",
                repo_type="dataset",
                revision=revision,
            )
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            episodes.append(value)
    return episodes


def _split_for_episode(
    episode: dict[str, Any],
    *,
    seed: int,
    dev_percent: int,
    test_percent: int,
) -> str:
    episode_id = str(episode.get("episode_id") or "")
    digest = hashlib.blake2b(f"{seed}:{episode_id}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + dev_percent:
        return "dev"
    return "train"


def _with_split(graph, *, split: str):
    return graph.__class__(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={**graph.metadata, "split": split},
    )


if __name__ == "__main__":
    main()
