"""Training adapters for the unified mapping page-pair dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.mapping_dataset import validate_mapping_page_pair
from omnitransfer.self_supervised import TrainingPair, make_correspondence_training_pair
from omnitransfer.ui_graph import (
    UIGraph,
    UINode,
    graph_from_record,
    multi_anchor_context_graph,
)


def load_mapping_training_pairs(
    paths: Iterable[str | Path],
    *,
    matcher_config: MatcherConfig | None = None,
    allow_unreviewed_pseudo: bool = False,
    minimum_correspondences: int = 2,
    max_pairs: int = 0,
    screenshot_root: str | Path | None = None,
) -> tuple[list[TrainingPair], list[UIGraph], dict[str, Any]]:
    """Load strict one-to-one cross-page labels plus their unlabeled pages."""

    if minimum_correspondences <= 0:
        raise ValueError("minimum_correspondences must be positive")
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    pairs: list[TrainingPair] = []
    config = matcher_config or MatcherConfig()
    graphs: dict[str, UIGraph] = {}
    skipped: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    input_rows = 0
    accepted_correspondences = 0
    explicit_correspondences = 0
    ambiguous_rows = 0
    conflicting_rows = 0
    non_actionable_correspondences = 0
    assets = (
        Path(screenshot_root).expanduser().resolve()
        if screenshot_root is not None
        else None
    )
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                input_rows += 1
                try:
                    record = validate_mapping_page_pair(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    skipped[f"invalid:{exc}"] += 1
                    continue
                label_status = record["label_status"]
                if record["split"] not in {"train", "diagnostic"}:
                    skipped[f"split:{record['split']}"] += 1
                    continue
                if label_status == "unreviewed" and not allow_unreviewed_pseudo:
                    skipped["unreviewed_requires_opt_in"] += 1
                    continue
                if label_status not in {
                    "gold",
                    "weak",
                    "self_supervised",
                    "unreviewed",
                }:
                    skipped[f"label_status:{label_status}"] += 1
                    continue
                source_graph = _page_graph(
                    record, side="source", screenshot_root=assets
                )
                target_graph = _page_graph(
                    record, side="target", screenshot_root=assets
                )
                correspondences, row_stats = _strict_correspondences(record)
                explicit_correspondences += len(correspondences)
                ambiguous_rows += row_stats["ambiguous_rows"]
                conflicting_rows += row_stats["conflicting_rows"]
                source_nodes = {node.node_id: node for node in source_graph.nodes}
                target_nodes = {node.node_id: node for node in target_graph.nodes}
                actionable_correspondences = [
                    (source_id, target_id)
                    for source_id, target_id in correspondences
                    if _is_actionable(source_nodes[source_id])
                    and _is_actionable(target_nodes[target_id])
                ]
                non_actionable_correspondences += len(correspondences) - len(
                    actionable_correspondences
                )
                correspondences = actionable_correspondences
                if len(correspondences) < minimum_correspondences:
                    skipped["too_few_strict_correspondences"] += 1
                    continue
                source_actionable_ids = tuple(
                    node.node_id for node in source_graph.nodes if _is_actionable(node)
                )
                target_actionable_ids = tuple(
                    node.node_id for node in target_graph.nodes if _is_actionable(node)
                )
                source_graph = multi_anchor_context_graph(
                    source_graph,
                    anchor_node_ids=source_actionable_ids,
                    max_nodes=max(
                        config.source_context_nodes,
                        len(source_actionable_ids),
                    ),
                )
                target_graph = multi_anchor_context_graph(
                    target_graph,
                    anchor_node_ids=target_actionable_ids,
                    max_nodes=max(
                        config.target_context_nodes,
                        len(target_actionable_ids),
                    ),
                )
                try:
                    pair = make_correspondence_training_pair(
                        source_graph,
                        target_graph,
                        correspondences,
                        matcher_config=config,
                    )
                except ValueError as exc:
                    skipped[f"invalid_pair:{exc}"] += 1
                    continue
                pairs.append(pair)
                graphs[source_graph.graph_id] = source_graph
                graphs[target_graph.graph_id] = target_graph
                accepted_correspondences += len(correspondences)
                datasets[str(record["provenance"].get("dataset") or "unknown")] += 1
                labels[label_status] += 1
                if max_pairs and len(pairs) >= max_pairs:
                    break
        if max_pairs and len(pairs) >= max_pairs:
            break
    manifest = {
        "schema_version": "omnitransfer.mapping_training_adapter.v1",
        "input_rows": input_rows,
        "accepted_pairs": len(pairs),
        "accepted_graphs": len(graphs),
        "accepted_correspondences": accepted_correspondences,
        "explicit_correspondences": explicit_correspondences,
        "ambiguous_match_rows_filtered": ambiguous_rows,
        "conflicting_target_rows_filtered": conflicting_rows,
        "non_actionable_correspondences_filtered": non_actionable_correspondences,
        "allow_unreviewed_pseudo": allow_unreviewed_pseudo,
        "minimum_correspondences": minimum_correspondences,
        "screenshot_root": str(assets) if assets is not None else None,
        "datasets": dict(sorted(datasets.items())),
        "label_statuses": dict(sorted(labels.items())),
        "skipped": dict(sorted(skipped.items())),
        "training_boundary": {
            "stable_ids_are_label_only": True,
            "set_valued_rows_are_filtered": True,
            "target_collisions_are_filtered": True,
            "actionable_correspondence_only": True,
            "non_actionable_nodes_are_context_only": True,
            "same_screen_actionable_nodes_are_candidates": True,
            "raw_resource_id_model_input": False,
            "absolute_position_model_input": False,
            "local_context_limits": {
                "source": config.source_context_nodes,
                "target": config.target_context_nodes,
            },
            "model_input": (
                "text_content_desc_class_action_visual_plus_within_page_relations"
            ),
        },
    }
    return pairs, list(graphs.values()), manifest


def _is_actionable(node: UINode) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def _page_graph(
    record: dict[str, Any],
    *,
    side: str,
    screenshot_root: Path | None,
) -> UIGraph:
    page = record[side]
    graph = graph_from_record(page["graph"], graph_id=page["page_id"])
    screenshot_path = _resolve_screenshot_path(
        str(page["screenshot_path"]),
        screenshot_root=screenshot_root,
    )
    return replace(
        graph,
        metadata={
            **graph.metadata,
            "screenshot_path": screenshot_path,
            "pair_id": record["pair_id"],
            "dataset": record["provenance"].get("dataset"),
            "platform": page["platform"],
        },
    )


def _resolve_screenshot_path(value: str, *, screenshot_root: Path | None) -> str:
    original = Path(value).expanduser()
    if original.is_file() or screenshot_root is None or not value:
        return str(original)
    exact = screenshot_root / original.name
    if exact.is_file():
        return str(exact)
    matches = sorted(screenshot_root.rglob(f"*_{original.name}"))
    if len(matches) == 1:
        return str(matches[0])
    return str(original)


def _strict_correspondences(
    record: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    single_target_rows = [
        match
        for match in record["matches"]
        if match["label"] == "correspondence" and len(match["target_node_ids"]) == 1
    ]
    target_counts = Counter(match["target_node_ids"][0] for match in single_target_rows)
    correspondences = [
        (match["source_node_id"], match["target_node_ids"][0])
        for match in single_target_rows
        if target_counts[match["target_node_ids"][0]] == 1
    ]
    return correspondences, {
        "ambiguous_rows": sum(
            match["label"] == "correspondence" and len(match["target_node_ids"]) != 1
            for match in record["matches"]
        ),
        "conflicting_rows": sum(
            target_counts[match["target_node_ids"][0]] > 1
            for match in single_target_rows
        ),
    }
