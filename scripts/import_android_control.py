#!/usr/bin/env python3
"""Stream official AndroidControl shards into frozen Unified UIGraph files."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from omnitransfer.android_control import (
    feature_bytes,
    feature_ints,
    graph_from_android_control_forest,
    iter_android_control_examples,
    materialize_android_control_image,
    source_name,
)
from omnitransfer.ui_graph import UIGraph, graph_to_record


DEFAULT_SPLITS = "https://storage.googleapis.com/gresearch/android_control/splits.json"
DEFAULT_TEST_SUBSPLITS = (
    "https://storage.googleapis.com/gresearch/android_control/test_subsplits.json"
)
TEST_SUBSPLITS = ("IDD", "task_unseen", "app_unseen", "category_unseen")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream GZIP TFRecord shards from local files or HTTPS, parse the "
            "official accessibility forests without TensorFlow, and write 384px "
            "Unified UIGraph bundles using the official episode splits."
        )
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--test-subsplits", default=DEFAULT_TEST_SUBSPLITS)
    parser.add_argument("--image-long-side", type=int, default=384)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--max-screens", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()
    if args.image_long_side <= 0 or args.min_nodes <= 0:
        raise SystemExit("image-long-side and min-nodes must be positive")
    if args.max_screens < 0 or args.log_interval <= 0:
        raise SystemExit("max-screens must be non-negative and log-interval positive")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"AndroidControl output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part-{os.getpid()}")
    temporary.mkdir(parents=True)

    split_payload, split_sha256 = _load_json(args.splits)
    test_payload, test_sha256 = _load_json(args.test_subsplits)
    split_by_episode = _invert_split_manifest(split_payload)
    test_subsplits_by_episode = _invert_test_subsplits(test_payload)
    split_counts: Counter[str] = Counter()
    test_subsplit_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    official_episode_counts: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    node_total = 0
    node_max = 0
    records_scanned = 0
    observations_scanned = 0
    seen_observations: set[tuple[bytes, bytes]] = set()
    source_summaries: list[dict[str, Any]] = []

    with ExitStack() as stack:
        outputs = {
            split: stack.enter_context(
                (temporary / f"graphs.{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in ("train", "dev", "test")
        }
        test_outputs = {
            name: stack.enter_context(
                (temporary / f"graphs.test.{name}.jsonl").open(
                    "w", encoding="utf-8"
                )
            )
            for name in TEST_SUBSPLITS
        }
        stop = False
        for source in args.input:
            shard_name = source_name(source)
            shard_records = 0
            shard_observations = 0
            shard_accepted = 0
            for record_index, features in enumerate(iter_android_control_examples(source)):
                records_scanned += 1
                shard_records += 1
                episode_values = feature_ints(features, "episode_id")
                if not episode_values:
                    skipped["missing_episode_id"] += 1
                    continue
                episode_id = episode_values[0]
                official_name = split_by_episode.get(episode_id)
                if official_name is None:
                    skipped["episode_absent_from_official_splits"] += 1
                    continue
                output_split = {
                    "train": "train",
                    "validation": "dev",
                    "test": "test",
                }[official_name]
                screenshots = feature_bytes(features, "screenshots")
                forests = feature_bytes(features, "accessibility_trees")
                widths = feature_ints(features, "screenshot_widths")
                heights = feature_ints(features, "screenshot_heights")
                actions = feature_bytes(features, "actions")
                instructions = feature_bytes(features, "step_instructions")
                goals = feature_bytes(features, "goal")
                observation_count = min(
                    len(screenshots), len(forests), len(widths), len(heights)
                )
                if observation_count == 0:
                    skipped["missing_observations"] += 1
                    continue
                if len({len(screenshots), len(forests), len(widths), len(heights)}) != 1:
                    skipped["observation_length_mismatch"] += 1
                for step_index in range(observation_count):
                    observations_scanned += 1
                    shard_observations += 1
                    dedupe_key = (
                        hashlib.sha256(screenshots[step_index]).digest(),
                        hashlib.sha256(forests[step_index]).digest(),
                    )
                    if dedupe_key in seen_observations:
                        skipped["duplicate_observation"] += 1
                        continue
                    relative_image = (
                        Path("images")
                        / shard_name
                        / f"episode_{episode_id}_step_{step_index}.png"
                    )
                    materialized_image = temporary / relative_image
                    final_image = output / relative_image
                    try:
                        source_size = materialize_android_control_image(
                            screenshots[step_index],
                            materialized_image,
                            long_side=args.image_long_side,
                        )
                        if source_size != (widths[step_index], heights[step_index]):
                            materialized_image.unlink(missing_ok=True)
                            skipped["geometry_mismatch"] += 1
                            continue
                        graph = graph_from_android_control_forest(
                            forests[step_index],
                            graph_id=(
                                f"android_control:{episode_id}:{step_index}:"
                                f"{shard_name}"
                            ),
                            width=widths[step_index],
                            height=heights[step_index],
                            episode_id=episode_id,
                            step_index=step_index,
                            screenshot_path=str(final_image),
                            official_split=official_name,
                            test_subsplits=test_subsplits_by_episode.get(
                                episode_id, ()
                            ),
                        )
                    except (OSError, TypeError, ValueError):
                        materialized_image.unlink(missing_ok=True)
                        skipped["invalid_observation"] += 1
                        continue
                    if len(graph.nodes) < args.min_nodes:
                        materialized_image.unlink(missing_ok=True)
                        skipped["too_few_nodes"] += 1
                        continue
                    graph = _attach_label_metadata(
                        graph,
                        source_shard=shard_name,
                        source_record=record_index,
                        goal=_decode_optional(goals, 0),
                        next_action=_decode_optional(actions, step_index),
                        step_instruction=_decode_optional(instructions, step_index),
                    )
                    serialized = json.dumps(
                        graph_to_record(graph), ensure_ascii=False
                    ) + "\n"
                    outputs[output_split].write(serialized)
                    if output_split == "test":
                        for name in graph.metadata.get("test_subsplits", ()):
                            if name not in test_outputs:
                                continue
                            test_outputs[name].write(serialized)
                            test_subsplit_counts[name] += 1
                    seen_observations.add(dedupe_key)
                    split_counts[output_split] += 1
                    official_episode_counts[official_name].add(episode_id)
                    node_total += len(graph.nodes)
                    node_max = max(node_max, len(graph.nodes))
                    shard_accepted += 1
                    accepted = sum(split_counts.values())
                    if accepted % args.log_interval == 0:
                        print(
                            json.dumps(
                                {
                                    "accepted": accepted,
                                    "records_scanned": records_scanned,
                                    "observations_scanned": observations_scanned,
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
            source_summaries.append(
                {
                    "source": source,
                    "shard": shard_name,
                    "records_scanned": shard_records,
                    "observations_scanned": shard_observations,
                    "accepted_screens": shard_accepted,
                }
            )
            if stop:
                break

    accepted = sum(split_counts.values())
    manifest = {
        "schema_version": "omnitransfer_android_control_bundle_v1",
        "dataset": "google-research/AndroidControl",
        "license": "Apache-2.0",
        "sources": source_summaries,
        "splits_source": args.splits,
        "splits_sha256": split_sha256,
        "test_subsplits_source": args.test_subsplits,
        "test_subsplits_sha256": test_sha256,
        "adapter": "GZIP TFRecord + AndroidAccessibilityForest -> Unified UIGraph",
        "split_unit": "official_episode_id",
        "official_split_counts": {
            name: len(values) for name, values in official_episode_counts.items()
        },
        "image_long_side": args.image_long_side,
        "lossless_image_format": "png",
        "records_scanned": records_scanned,
        "observations_scanned": observations_scanned,
        "accepted_screens": accepted,
        "split_counts": {
            split: split_counts[split] for split in ("train", "dev", "test")
        },
        "test_subsplit_counts": {
            name: test_subsplit_counts[name] for name in TEST_SUBSPLITS
        },
        "skipped": dict(sorted(skipped.items())),
        "node_count": node_total,
        "average_nodes": node_total / accepted if accepted else 0.0,
        "max_nodes": node_max,
        "policies": {
            "origin_id": "augmentation_label_only",
            "actions_and_instructions": "label_metadata_only",
            "correspondence": "same_node_under_independent_train_only_augmentations",
            "test": "official_frozen_episode_split_with_published_subsplits",
            "inference": "relation_aware_cross_attention_matcher_no_shortcuts",
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(manifest, ensure_ascii=False))


def _load_json(source: str) -> tuple[dict[str, Any], str]:
    if source.startswith(("http://", "https://")):
        request = Request(source, headers={"User-Agent": "OmniTransfer/0.1"})
        with urlopen(request, timeout=120) as response:
            content = response.read()
    else:
        content = Path(source).expanduser().read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must contain an object: {source}")
    return payload, hashlib.sha256(content).hexdigest()


def _invert_split_manifest(payload: dict[str, Any]) -> dict[int, str]:
    inverted: dict[int, str] = {}
    for split in ("train", "validation", "test"):
        values = payload.get(split)
        if not isinstance(values, list):
            raise ValueError(f"AndroidControl split manifest is missing {split!r}")
        for value in values:
            episode_id = int(value)
            if episode_id in inverted:
                raise ValueError(f"AndroidControl episode appears in multiple splits: {episode_id}")
            inverted[episode_id] = split
    return inverted


def _invert_test_subsplits(payload: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    values_by_episode: dict[int, list[str]] = {}
    for name, values in payload.items():
        if not isinstance(values, list):
            continue
        for value in values:
            values_by_episode.setdefault(int(value), []).append(str(name))
    return {
        episode_id: tuple(sorted(names))
        for episode_id, names in values_by_episode.items()
    }


def _attach_label_metadata(
    graph: UIGraph,
    *,
    source_shard: str,
    source_record: int,
    goal: str,
    next_action: str,
    step_instruction: str,
) -> UIGraph:
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "source_shard": source_shard,
            "source_record": source_record,
            "goal_label": goal,
            "next_action_label": next_action,
            "step_instruction_label": step_instruction,
        },
    )


def _decode_optional(values: list[bytes], index: int) -> str:
    if index < 0 or index >= len(values):
        return ""
    return values[index].decode("utf-8", errors="replace")


if __name__ == "__main__":
    main()
