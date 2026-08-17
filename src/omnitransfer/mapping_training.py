"""Training adapters for the unified UI correspondence-pair dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.learned_matcher import ALL_NODE_CANDIDATE_POLICY, MatcherConfig
from omnitransfer.mapping_dataset import validate_ui_correspondence_pair
from omnitransfer.self_supervised import (
    CorrespondencePair,
    make_correspondence_training_pair,
)
from omnitransfer.ui_graph import (
    UIGraph,
    graph_from_record,
)


def load_ui_correspondence_pairs(
    paths: Iterable[str | Path],
    *,
    matcher_config: MatcherConfig | None = None,
    allow_unreviewed_pseudo: bool = False,
    minimum_correspondences: int = 1,
    max_pairs: int = 0,
    screenshot_root: str | Path | None = None,
    allowed_splits: frozenset[str] = frozenset({"train", "diagnostic"}),
    allowed_reserved_splits: frozenset[str] | None = None,
) -> tuple[list[CorrespondencePair], list[UIGraph], dict[str, Any]]:
    """Load set-valued UI correspondence labels through the canonical adapter."""

    if minimum_correspondences <= 0:
        raise ValueError("minimum_correspondences must be positive")
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if allowed_reserved_splits is not None and not allowed_reserved_splits <= {
        "dev",
        "test",
    }:
        raise ValueError("reserved splits may contain only dev and test")
    pairs: list[CorrespondencePair] = []
    config = matcher_config or MatcherConfig()
    if config.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise ValueError("training accepts only the all-node candidate policy")
    graphs: dict[str, UIGraph] = {}
    skipped: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    input_rows = 0
    accepted_correspondences = 0
    explicit_correspondences = 0
    set_valued_rows = 0
    shared_target_rows = 0
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
                    record = validate_ui_correspondence_pair(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    skipped[f"invalid:{exc}"] += 1
                    continue
                label_status = record["label_status"]
                if record["split"] not in allowed_splits:
                    skipped[f"split:{record['split']}"] += 1
                    continue
                if allowed_reserved_splits is not None:
                    reserved_split = str(
                        record["provenance"].get("reserved_split") or "missing"
                    )
                    if reserved_split not in allowed_reserved_splits:
                        skipped[f"reserved_split:{reserved_split}"] += 1
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
                correspondences, row_stats = _correspondence_edges(record)
                explicit_correspondences += len(correspondences)
                set_valued_rows += row_stats["set_valued_rows"]
                shared_target_rows += row_stats["shared_target_rows"]
                if len(correspondences) < minimum_correspondences:
                    skipped["too_few_correspondences"] += 1
                    continue
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
        "schema_version": "omnitransfer.ui_correspondence_adapter.v1",
        "input_rows": input_rows,
        "accepted_pairs": len(pairs),
        "accepted_graphs": len(graphs),
        "accepted_correspondences": accepted_correspondences,
        "explicit_correspondences": explicit_correspondences,
        "set_valued_match_rows_retained": set_valued_rows,
        "shared_target_rows_retained": shared_target_rows,
        "candidate_policy": config.candidate_policy,
        "allow_unreviewed_pseudo": allow_unreviewed_pseudo,
        "minimum_correspondences": minimum_correspondences,
        "allowed_splits": sorted(allowed_splits),
        "allowed_reserved_splits": (
            sorted(allowed_reserved_splits)
            if allowed_reserved_splits is not None
            else None
        ),
        "screenshot_root": str(assets) if assets is not None else None,
        "datasets": dict(sorted(datasets.items())),
        "label_statuses": dict(sorted(labels.items())),
        "skipped": dict(sorted(skipped.items())),
        "training_boundary": {
            "stable_ids_are_label_only": True,
            "set_valued_rows_are_retained": True,
            "shared_targets_are_retained": True,
            "all_xml_nodes_are_candidates": True,
            "all_xml_nodes_are_preserved": True,
            "raw_resource_id_model_input": False,
            "absolute_position_model_input": False,
            "model_input": (
                "text_content_desc_class_action_visual_plus_within_page_relations"
            ),
        },
    }
    return pairs, list(graphs.values()), manifest


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
    if not original.is_absolute():
        relative = screenshot_root / original
        if relative.is_file():
            return str(relative)
    for marker in ("runtime", "traces", "screenshots"):
        if marker not in original.parts:
            continue
        suffix = Path(*original.parts[original.parts.index(marker) :])
        materialized = screenshot_root / suffix
        if materialized.is_file():
            return str(materialized)
    exact = screenshot_root / original.name
    if exact.is_file():
        return str(exact)
    matches = sorted(screenshot_root.rglob(f"*_{original.name}"))
    if len(matches) == 1:
        return str(matches[0])
    return str(original)


def _correspondence_edges(
    record: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    correspondence_rows = [
        match for match in record["matches"] if match["label"] == "correspondence"
    ]
    correspondences = [
        (match["source_node_id"], target_id)
        for match in correspondence_rows
        for target_id in match["target_node_ids"]
    ]
    target_counts = Counter(target_id for _, target_id in correspondences)
    return correspondences, {
        "set_valued_rows": sum(
            len(match["target_node_ids"]) > 1 for match in correspondence_rows
        ),
        "shared_target_rows": sum(
            target_counts[target_id] > 1
            for match in correspondence_rows
            for target_id in match["target_node_ids"]
        ),
    }
