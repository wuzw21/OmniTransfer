"""Weak GUIOdyssey correspondence pairs for cross-attention training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from omnitransfer.gui_odyssey import graph_from_gui_odyssey_step
from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.self_supervised import (
    CorrespondencePair,
    make_correspondence_training_pair,
)
from omnitransfer.ui_graph import UIGraph, UINode


PAIR_SCHEMA = "omniflow.guiodyssey_sequence_filtered_click_pair.v1"
HUMAN_REVIEW_SCHEMAS = frozenset(
    {
        "omnitransfer_guiodyssey_node_review_v1",
        "omnitransfer_guiodyssey_sequence_node_alignment_v1",
    }
)


def load_gui_odyssey_human_review_pairs(
    review_path: str | Path,
    *,
    screenshot_dir: str | Path | None = None,
    matcher_config: MatcherConfig | None = None,
) -> tuple[list[CorrespondencePair], dict[str, Any]]:
    """Load exported human node correspondences as formal gold pairs."""

    path = Path(review_path).expanduser().resolve()
    rows = (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    screenshots = (
        Path(screenshot_dir).expanduser().resolve()
        if screenshot_dir is not None
        else path.parent
    )
    training_pairs: list[CorrespondencePair] = []
    labels: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    input_rows = 0
    for row in rows:
        input_rows += 1
        if row.get("schema_version") not in HUMAN_REVIEW_SCHEMAS:
            skipped["unsupported_schema"] += 1
            continue
        annotation = row.get("annotation")
        if not isinstance(annotation, dict) or annotation.get("status") != "reviewed":
            skipped["not_reviewed"] += 1
            continue
        label = str(annotation.get("label") or "")
        if label not in {"correspondence", "no_correspondence"}:
            skipped[f"label:{label or 'missing'}"] += 1
            continue
        try:
            source_graph = _human_review_graph(
                row,
                annotation,
                side="source",
                screenshots=screenshots,
            )
            target_graph = _human_review_graph(
                row,
                annotation,
                side="target",
                screenshots=screenshots,
            )
            matches = tuple(
                (str(match["source_node_id"]), str(match["target_node_id"]))
                for match in annotation.get("matches") or ()
                if isinstance(match, dict)
            )
            if label == "correspondence" and not matches:
                raise ValueError("correspondence review has no node matches")
            if label == "no_correspondence" and matches:
                raise ValueError("no_correspondence review contains node matches")
            pair = make_correspondence_training_pair(
                source_graph,
                target_graph,
                matches,
                matcher_config=matcher_config,
                allow_empty=label == "no_correspondence",
                unmatched_as_null=label == "no_correspondence",
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped[f"invalid_review:{exc}"] += 1
            continue
        training_pairs.append(pair)
        labels[label] += 1
    return training_pairs, {
        "schema_version": "omnitransfer_guiodyssey_human_gold_manifest_v1",
        "input_rows_read": input_rows,
        "accepted_pairs": len(training_pairs),
        "labels": dict(sorted(labels.items())),
        "skipped": dict(sorted(skipped.items())),
        "supervision": "human_reviewed_benchmark_gold",
        "candidate_space": "human_annotated_nodes",
    }


def load_gui_odyssey_training_pairs(
    pair_path: str | Path,
    annotation_dir: str | Path,
    *,
    screenshot_dir: str | Path | None = None,
    max_pairs: int = 0,
    require_bbox_contains_point: bool = True,
    matcher_config: MatcherConfig | None = None,
) -> tuple[list[CorrespondencePair], dict[str, Any]]:
    """Load weak point correspondences as partial-assignment graph pairs."""

    pairs = Path(pair_path).expanduser().resolve()
    with pairs.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return make_gui_odyssey_training_pairs(
            rows,
            annotation_dir,
            screenshot_dir=screenshot_dir,
            max_pairs=max_pairs,
            require_bbox_contains_point=require_bbox_contains_point,
            matcher_config=matcher_config,
        )


def make_gui_odyssey_training_pairs(
    rows: Iterable[dict[str, Any]],
    annotation_dir: str | Path,
    *,
    screenshot_dir: str | Path | None = None,
    max_pairs: int = 0,
    require_bbox_contains_point: bool = True,
    matcher_config: MatcherConfig | None = None,
) -> tuple[list[CorrespondencePair], dict[str, Any]]:
    """Convert selected weak pair rows through the canonical graph adapter."""

    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    annotations = Path(annotation_dir).expanduser().resolve()
    screenshots = (
        Path(screenshot_dir).expanduser().resolve() if screenshot_dir is not None else None
    )
    episode_cache: dict[str, dict[str, Any]] = {}
    training_pairs: list[CorrespondencePair] = []
    skipped: Counter[str] = Counter()
    input_rows = 0
    pairs_with_both_screenshots = 0
    for row in rows:
        input_rows += 1
        if row.get("schema_version") != PAIR_SCHEMA:
            skipped["unsupported_schema"] += 1
            continue
        if bool((row.get("selection") or {}).get("gold_label")):
            skipped["unexpected_gold_label"] += 1
            continue
        try:
            pair = _make_weak_gui_odyssey_pair(
                row,
                annotations=annotations,
                screenshots=screenshots,
                episode_cache=episode_cache,
                require_bbox_contains_point=require_bbox_contains_point,
                matcher_config=matcher_config,
            )
        except _ActionPointOutsideBBox:
            skipped["action_bbox_excludes_point"] += 1
            continue
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            skipped[f"invalid_endpoint:{exc}"] += 1
            continue
        training_pairs.append(pair)
        if pair.graph_a.metadata.get("screenshot_path") and pair.graph_b.metadata.get(
            "screenshot_path"
        ):
            pairs_with_both_screenshots += 1
        if max_pairs and len(training_pairs) >= max_pairs:
            break
    manifest = {
        "schema_version": "omnitransfer_guiodyssey_pair_adapter_manifest_v1",
        "source_pair_schema": PAIR_SCHEMA,
        "input_rows_read": input_rows,
        "accepted_pairs": len(training_pairs),
        "pairs_with_both_screenshots": pairs_with_both_screenshots,
        "skipped": dict(sorted(skipped.items())),
        "coordinate_space": "guiodyssey_normalized_1000_scaled_to_device_pixels",
        "supervision": "weak_correspondence_not_benchmark_gold",
        "training_interface": "bidirectional_partial_assignment_ignore_unlabeled_explicit_null_only",
    }
    return training_pairs, manifest


class _ActionPointOutsideBBox(ValueError):
    pass


def _make_weak_gui_odyssey_pair(
    row: dict[str, Any],
    *,
    annotations: Path,
    screenshots: Path | None,
    episode_cache: dict[str, dict[str, Any]],
    require_bbox_contains_point: bool,
    matcher_config: MatcherConfig | None,
) -> CorrespondencePair:
    source_episode, source_step = _resolve_endpoint(
        row.get("source"), annotations=annotations, cache=episode_cache
    )
    target_episode, target_step = _resolve_endpoint(
        row.get("target"), annotations=annotations, cache=episode_cache
    )
    pair_id = str(row.get("pair_id") or "guiodyssey-pair")
    source_graph = _endpoint_graph(
        source_episode,
        source_step,
        pair_id=pair_id,
        side="source",
        screenshots=screenshots,
    )
    target_graph = _endpoint_graph(
        target_episode,
        target_step,
        pair_id=pair_id,
        side="target",
        screenshots=screenshots,
    )
    if require_bbox_contains_point and (
        not _action_bbox_contains_point(source_graph)
        or not _action_bbox_contains_point(target_graph)
    ):
        raise _ActionPointOutsideBBox(pair_id)
    return make_correspondence_training_pair(
        source_graph,
        target_graph,
        (("action_target", "action_target"),),
        matcher_config=matcher_config,
    )


def _resolve_endpoint(
    endpoint: Any,
    *,
    annotations: Path,
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(endpoint, dict):
        raise TypeError("endpoint is not an object")
    episode_id = str(endpoint.get("episode_id") or "")
    if not episode_id:
        raise ValueError("endpoint has no episode_id")
    if episode_id not in cache:
        path = annotations / f"{episode_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"annotation is missing: {path}")
        cache[episode_id] = json.loads(path.read_text(encoding="utf-8"))
    episode = cache[episode_id]
    step_index = int(endpoint.get("step_index"))
    step = next(
        (
            candidate
            for candidate in episode.get("steps") or ()
            if isinstance(candidate, dict) and int(candidate.get("step") or 0) == step_index
        ),
        None,
    )
    if step is None:
        raise KeyError(f"step {step_index} is absent from episode {episode_id}")
    return episode, step


def _endpoint_graph(
    episode: dict[str, Any],
    step: dict[str, Any],
    *,
    pair_id: str,
    side: str,
    screenshots: Path | None,
) -> UIGraph:
    screenshot_name = str(step.get("screenshot") or "")
    screenshot_path = screenshots / screenshot_name if screenshots and screenshot_name else None
    resolved_step = {
        **step,
        "screenshot_path": str(screenshot_path.resolve())
        if screenshot_path is not None and screenshot_path.is_file()
        else "",
    }
    graph = graph_from_gui_odyssey_step(
        episode,
        resolved_step,
        graph_id=f"{pair_id}:{side}",
    )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=graph.nodes,
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "pair_id": pair_id,
            "pair_side": side,
        },
    )


def _action_bbox_contains_point(graph: UIGraph) -> bool:
    point = graph.metadata.get("action_point_label")
    target = next((node for node in graph.nodes if node.node_id == "action_target"), None)
    if target is None or target.bbox is None or not isinstance(point, tuple) or len(point) != 2:
        return False
    return (
        target.bbox[0] <= point[0] <= target.bbox[2]
        and target.bbox[1] <= point[1] <= target.bbox[3]
    )


def _human_review_graph(
    row: dict[str, Any],
    annotation: dict[str, Any],
    *,
    side: str,
    screenshots: Path,
) -> UIGraph:
    endpoint = row.get(side)
    if not isinstance(endpoint, dict):
        raise TypeError(f"{side} endpoint is absent")
    width = float(endpoint.get("width") or 0.0)
    height = float(endpoint.get("height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{side} dimensions are invalid")
    raw_nodes = annotation.get(f"{side}_nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError(f"{side} review nodes are absent")
    nodes: list[Any] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise TypeError(f"{side} review node is not an object")
        node_id = str(raw_node.get("node_id") or "")
        point = raw_node.get("point_normalized")
        if not node_id or not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{side} review node is incomplete")
        center_x = float(point[0]) * width / 1000.0
        center_y = float(point[1]) * height / 1000.0
        half_width = max(2.0, width * 0.012)
        half_height = max(2.0, height * 0.012)
        nodes.append(
            UINode(
                node_id=node_id,
                origin_id=node_id,
                text=str(raw_node.get("label") or ""),
                class_name="guiodyssey.human_point_node",
                bbox=(
                    max(0.0, center_x - half_width),
                    max(0.0, center_y - half_height),
                    min(width, center_x + half_width),
                    min(height, center_y + half_height),
                ),
                clickable=True,
                enabled=True,
                metadata={
                    "human_annotated": True,
                    "provenance": str(raw_node.get("provenance") or "human_review"),
                },
            )
        )
    image_url = str(endpoint.get("image_url") or endpoint.get("screenshot") or "")
    screenshot_path = screenshots / image_url if image_url else None
    return UIGraph(
        graph_id=f"{row.get('pair_id') or 'human-review'}:{side}",
        nodes=tuple(nodes),
        width=width,
        height=height,
        metadata={
            "dataset": "guiodyssey",
            "supervision": "human_reviewed_node_correspondence",
            "pair_id": str(row.get("pair_id") or ""),
            "pair_side": side,
            "episode_id": str(endpoint.get("episode_id") or ""),
            "device_name": str(endpoint.get("device_name") or ""),
            "screenshot_path": (
                str(screenshot_path.resolve())
                if screenshot_path is not None and screenshot_path.is_file()
                else ""
            ),
        },
    )
