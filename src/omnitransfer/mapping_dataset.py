"""Unified page-pair dataset contract for learned UI mapping."""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.schema import Query
from omnitransfer.gui_odyssey_pairs import (
    HUMAN_REVIEW_SCHEMAS,
    PAIR_SCHEMA,
    _ActionPointOutsideBBox,
    _human_review_graph,
    _make_weak_gui_odyssey_pair,
)
from omnitransfer.ui_graph import UIGraph, graph_from_record, graph_to_record


MAPPING_PAGE_PAIR_SCHEMA = "omnitransfer.mapping_page_pair.v1"
_SPLITS = frozenset({"train", "dev", "test", "diagnostic"})
_LABEL_STATUSES = frozenset({"gold", "weak", "self_supervised", "unreviewed"})


def adapt_ase_queries(
    queries: list[Query],
    *,
    asset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Group authored ASE queries into gold multi-match page-pair records."""

    resolved_assets = (
        Path(asset_root).expanduser().resolve()
        if asset_root is not None
        else None
    )
    grouped: dict[tuple[str, str], list[Query]] = defaultdict(list)
    for query in queries:
        source_page_id = _ase_page_id(query, "source")
        target_page_id = _ase_page_id(query, "target")
        grouped[(source_page_id, target_page_id)].append(query)

    records: list[dict[str, Any]] = []
    for (source_page_id, target_page_id), page_queries in sorted(grouped.items()):
        splits = {_required_query_metadata(query, "split") for query in page_queries}
        apps = {_required_query_metadata(query, "app") for query in page_queries}
        if len(splits) != 1:
            raise ValueError(f"ASE page pair spans splits: {source_page_id} -> {target_page_id}")
        if len(apps) != 1:
            raise ValueError(f"ASE page pair spans apps: {source_page_id} -> {target_page_id}")
        split = next(iter(splits))
        app = next(iter(apps))
        source_nodes = {
            _source_node_id(query): _node_record_from_payload(
                query.source,
                node_id=_source_node_id(query),
            )
            for query in page_queries
        }
        target_nodes = {
            candidate.candidate_id: _node_record_from_payload(
                {**candidate.metadata, "bbox": candidate.bbox},
                node_id=candidate.candidate_id,
            )
            for query in page_queries
            for candidate in query.target_candidates
        }
        source_graph = _page_graph(
            source_page_id,
            _resolve_asset_path(
                str(page_queries[0].metadata.get("source_xml_path") or ""),
                asset_root=resolved_assets,
            ),
            source_nodes,
        )
        target_graph = _page_graph(
            target_page_id,
            _resolve_asset_path(
                str(page_queries[0].metadata.get("target_xml_path") or ""),
                asset_root=resolved_assets,
            ),
            target_nodes,
        )
        matches_by_source: dict[str, list[str]] = defaultdict(list)
        mapping_ids: list[str] = []
        for query in page_queries:
            mapping_ids.append(query.query_id)
            source_id = _source_node_id(query)
            for target_id in query.acceptable_gold_candidate_ids():
                if target_id not in matches_by_source[source_id]:
                    matches_by_source[source_id].append(target_id)
        _mark_action_anchors(
            source_graph,
            matches_by_source,
            evidence="ase_public_mapping_source",
        )
        record = {
            "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
            "pair_id": _stable_pair_id("ase", source_page_id, target_page_id),
            "split": split,
            "label_status": "gold",
            "source": {
                "page_id": source_page_id,
                "platform": _ase_platform(page_queries[0], "source"),
                "screenshot_path": _resolve_asset_path(
                    str(
                        page_queries[0].metadata.get("source_screenshot_path")
                        or ""
                    ),
                    asset_root=resolved_assets,
                ),
                "graph": source_graph,
            },
            "target": {
                "page_id": target_page_id,
                "platform": _ase_platform(page_queries[0], "target"),
                "screenshot_path": _resolve_asset_path(
                    str(
                        page_queries[0].metadata.get("target_screenshot_path")
                        or ""
                    ),
                    asset_root=resolved_assets,
                ),
                "graph": target_graph,
            },
            "matches": [
                {
                    "source_node_id": source_id,
                    "target_node_ids": target_ids,
                    "label": "correspondence",
                }
                for source_id, target_ids in sorted(matches_by_source.items())
            ],
            "partition_keys": [f"ase:app:{app}"],
            "provenance": {
                "dataset": "ase2023_vision_based_widget_mapping",
                "annotation": "public_gold",
                "mapping_ids": sorted(mapping_ids),
            },
            "slices": {
                "app": app,
                "source_platform": _ase_platform(page_queries[0], "source"),
                "target_platform": _ase_platform(page_queries[0], "target"),
                "form_factor": "phone",
            },
        }
        records.append(validate_mapping_page_pair(record))
    return records


def adapt_gui_odyssey_rows(
    rows: list[dict[str, Any]],
    annotation_dir: str | Path,
    *,
    split: str,
    screenshot_dir: str | Path | None = None,
    require_bbox_contains_point: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert GUIOdyssey weak rows to the unified train/diagnostic schema."""

    if split not in {"train", "diagnostic"}:
        raise ValueError("GUIOdyssey weak labels may only enter train or diagnostic")
    annotations = Path(annotation_dir).expanduser().resolve()
    screenshots = (
        Path(screenshot_dir).expanduser().resolve() if screenshot_dir is not None else None
    )
    cache: dict[str, dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    pairs_with_both_screenshots = 0
    for row in rows:
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
                episode_cache=cache,
                require_bbox_contains_point=require_bbox_contains_point,
                matcher_config=None,
            )
            record = _gui_odyssey_weak_record(row, pair.graph_a, pair.graph_b, split=split)
        except _ActionPointOutsideBBox:
            skipped["action_bbox_excludes_point"] += 1
            continue
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            skipped[f"invalid_pair:{exc}"] += 1
            continue
        records.append(validate_mapping_page_pair(record))
        if record["source"]["screenshot_path"] and record["target"]["screenshot_path"]:
            pairs_with_both_screenshots += 1
    return records, {
        "schema_version": "omnitransfer.mapping_guiodyssey_adapter_manifest.v1",
        "input_rows_read": len(rows),
        "accepted_pairs": len(records),
        "pairs_with_both_screenshots": pairs_with_both_screenshots,
        "skipped": dict(sorted(skipped.items())),
        "label_status": "weak",
        "split": split,
    }


def adapt_gui_odyssey_human_reviews(
    rows: list[dict[str, Any]],
    *,
    split: str,
    screenshot_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert reviewed GUIOdyssey multi-point labels to formal gold records."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("human GUIOdyssey gold requires train, dev, or test split")
    screenshots = Path(screenshot_dir).expanduser().resolve()
    skipped: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for row in rows:
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
                row, annotation, side="source", screenshots=screenshots
            )
            target_graph = _human_review_graph(
                row, annotation, side="target", screenshots=screenshots
            )
            record = _gui_odyssey_human_record(
                row,
                annotation,
                source_graph,
                target_graph,
                split=split,
                label=label,
            )
            records.append(validate_mapping_page_pair(record))
        except (KeyError, TypeError, ValueError) as exc:
            skipped[f"invalid_review:{exc}"] += 1
            continue
        labels[label] += 1
    return records, {
        "schema_version": "omnitransfer.mapping_guiodyssey_human_adapter_manifest.v1",
        "input_rows_read": len(rows),
        "accepted_pairs": len(records),
        "labels": dict(sorted(labels.items())),
        "skipped": dict(sorted(skipped.items())),
        "label_status": "gold",
        "split": split,
    }


def audit_mapping_page_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate rows and reject pair, page, or partition leakage across splits."""

    if not records:
        raise ValueError("mapping dataset is empty")
    normalized = [validate_mapping_page_pair(record) for record in records]
    pair_splits: dict[str, set[str]] = defaultdict(set)
    page_splits: dict[str, set[str]] = defaultdict(set)
    partition_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    for record in normalized:
        split = record["split"]
        assignment_split = _assignment_split(record)
        pair_splits[record["pair_id"]].add(assignment_split)
        for side in ("source", "target"):
            page_splits[record[side]["page_id"]].add(assignment_split)
        for key in record["partition_keys"]:
            partition_splits[key].add(assignment_split)
        split_counts[split] += 1
        assignment_counts[assignment_split] += 1
        label_counts[record["label_status"]] += 1
        dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
        for match in record["matches"]:
            match_counts[match["label"]] += 1

    duplicate_pairs = _cross_split_values(pair_splits)
    if duplicate_pairs:
        raise ValueError(f"pair id leakage: {_first_overlap(duplicate_pairs)}")
    page_overlap = _cross_split_values(page_splits)
    if page_overlap:
        raise ValueError(f"page leakage: {_first_overlap(page_overlap)}")
    partition_overlap = _cross_split_values(partition_splits)
    if partition_overlap:
        raise ValueError(f"partition leakage: {_first_overlap(partition_overlap)}")
    return {
        "schema_version": "omnitransfer.mapping_dataset_audit.v1",
        "valid": True,
        "records": len(normalized),
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "match_label_counts": dict(sorted(match_counts.items())),
        "overlap": {"pair_id": {}, "page_id": {}, "partition_key": {}},
    }


def write_mapping_dataset(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    review_candidates: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stream validated rows into atomic split files with leakage auditing."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = ("train", "dev", "test", "diagnostic")
    final_paths = {split: output / f"{split}.jsonl" for split in splits}
    temporary_paths = {
        split: path.with_suffix(path.suffix + ".part")
        for split, path in final_paths.items()
    }
    handles: dict[str, Any] = {}
    pair_splits: dict[str, set[str]] = defaultdict(set)
    page_splits: dict[str, set[str]] = defaultdict(set)
    partition_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    file_match_counts: Counter[str] = Counter()
    record_count = 0
    try:
        handles = {
            split: temporary_paths[split].open("w", encoding="utf-8")
            for split in splits
        }
        for raw_record in records:
            record = validate_mapping_page_pair(raw_record)
            split = record["split"]
            assignment_split = _assignment_split(record)
            _record_assignment(pair_splits, record["pair_id"], assignment_split, "pair id")
            for side in ("source", "target"):
                _record_assignment(
                    page_splits,
                    record[side]["page_id"],
                    assignment_split,
                    "page",
                )
            for key in record["partition_keys"]:
                _record_assignment(
                    partition_splits,
                    key,
                    assignment_split,
                    "partition",
                )
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handles[split].write(encoded + "\n")
            record_count += 1
            split_counts[split] += 1
            assignment_counts[assignment_split] += 1
            label_counts[record["label_status"]] += 1
            dataset_counts[
                str(record["provenance"].get("dataset") or "unknown")
            ] += 1
            file_match_counts[split] += len(record["matches"])
            for match in record["matches"]:
                match_counts[match["label"]] += 1
        if record_count == 0:
            raise ValueError("mapping dataset is empty")
        for handle in handles.values():
            handle.close()
        handles.clear()
        for split in splits:
            temporary_paths[split].replace(final_paths[split])
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    audit = {
        "schema_version": "omnitransfer.mapping_dataset_audit.v1",
        "valid": True,
        "records": record_count,
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "match_label_counts": dict(sorted(match_counts.items())),
        "overlap": {"pair_id": {}, "page_id": {}, "partition_key": {}},
    }
    files: dict[str, Any] = {}
    for split in splits:
        path = final_paths[split]
        files[path.name] = _file_manifest(
            path,
            records=split_counts[split],
            matches=file_match_counts[split],
        )

    for reserved_split, candidates in sorted((review_candidates or {}).items()):
        if reserved_split not in {"dev", "test"}:
            raise ValueError(f"unsupported review candidate split: {reserved_split}")
        rows = [validate_mapping_page_pair(record) for record in candidates]
        if any(record["split"] != "diagnostic" for record in rows):
            raise ValueError("review candidates must use the diagnostic split")
        if any(record["label_status"] != "unreviewed" for record in rows):
            raise ValueError("review candidates must be unreviewed")
        path = output / f"review_{reserved_split}_candidates.jsonl"
        ordered = sorted(rows, key=lambda row: row["pair_id"])
        _write_jsonl_atomic(path, ordered)
        files[path.name] = _file_manifest(
            path,
            records=len(ordered),
            matches=sum(len(row["matches"]) for row in ordered),
        )

    manifest = {
        "schema_version": "omnitransfer.mapping_dataset_manifest.v1",
        "record_schema": MAPPING_PAGE_PAIR_SCHEMA,
        "audit": audit,
        "files": files,
        "metadata": deepcopy(metadata or {}),
    }
    _write_text_atomic(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def validate_mapping_page_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized page-pair record or reject an invalid dataset row."""

    if not isinstance(record, dict):
        raise TypeError("mapping page pair must be an object")
    value = deepcopy(record)
    if value.get("schema_version") != MAPPING_PAGE_PAIR_SCHEMA:
        raise ValueError("unsupported mapping page-pair schema")
    _required_text(value, "pair_id")
    split = _required_text(value, "split")
    if split not in _SPLITS:
        raise ValueError(f"unsupported split: {split}")
    label_status = _required_text(value, "label_status")
    if label_status not in _LABEL_STATUSES:
        raise ValueError(f"unsupported label status: {label_status}")
    if split in {"dev", "test"} and label_status != "gold":
        raise ValueError(f"{split} contains non-gold labels")

    source_ids = _validate_page(value.get("source"), side="source")
    target_ids = _validate_page(value.get("target"), side="target")
    matches = value.get("matches")
    if not isinstance(matches, list) or not matches:
        raise ValueError("matches must be a non-empty list")
    seen_sources: set[str] = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise TypeError(f"matches[{index}] must be an object")
        source_id = _required_text(match, "source_node_id")
        if source_id not in source_ids:
            raise ValueError(f"match source node is absent: {source_id}")
        if source_id in seen_sources:
            raise ValueError(f"source node has duplicate match rows: {source_id}")
        seen_sources.add(source_id)
        label = _required_text(match, "label")
        target_node_ids = match.get("target_node_ids")
        if not isinstance(target_node_ids, list):
            raise TypeError(f"matches[{index}].target_node_ids must be a list")
        normalized_targets = list(dict.fromkeys(str(node_id) for node_id in target_node_ids))
        if label == "correspondence" and not normalized_targets:
            raise ValueError("correspondence match has no target nodes")
        if label == "no_correspondence" and normalized_targets:
            raise ValueError("no_correspondence match contains target nodes")
        if label not in {"correspondence", "no_correspondence"}:
            raise ValueError(f"unsupported match label: {label}")
        missing = [node_id for node_id in normalized_targets if node_id not in target_ids]
        if missing:
            raise ValueError(f"match target nodes are absent: {missing}")
        match["target_node_ids"] = normalized_targets

    partition_keys = value.get("partition_keys")
    if not isinstance(partition_keys, list) or not partition_keys:
        raise ValueError("partition_keys must be a non-empty list")
    value["partition_keys"] = list(
        dict.fromkeys(_nonempty_text(item, "partition key") for item in partition_keys)
    )
    if not isinstance(value.get("provenance"), dict):
        raise TypeError("provenance must be an object")
    if not isinstance(value.get("slices"), dict):
        raise TypeError("slices must be an object")
    return value


def _gui_odyssey_weak_record(
    row: dict[str, Any],
    source_graph: UIGraph,
    target_graph: UIGraph,
    *,
    split: str,
) -> dict[str, Any]:
    source_endpoint = row.get("source") if isinstance(row.get("source"), dict) else {}
    target_endpoint = row.get("target") if isinstance(row.get("target"), dict) else {}
    source_episode = _nonempty_text(
        source_endpoint.get("episode_id"), "source.episode_id"
    )
    target_episode = _nonempty_text(
        target_endpoint.get("episode_id"), "target.episode_id"
    )
    source_step = int(source_endpoint.get("step_index"))
    target_step = int(target_endpoint.get("step_index"))
    task = _nonempty_text(
        row.get("meta_task") or source_graph.metadata.get("meta_task"), "meta_task"
    )
    selection = row.get("selection") if isinstance(row.get("selection"), dict) else {}
    trajectory = str(
        selection.get("trajectory_pair_id")
        or row.get("episode_pair_id")
        or " | ".join(sorted((source_episode, target_episode)))
    )
    source_page_id = f"guiodyssey:{source_episode}:step:{source_step}"
    target_page_id = f"guiodyssey:{target_episode}:step:{target_step}"
    source_record = graph_to_record(source_graph)
    target_record = graph_to_record(target_graph)
    source_record["graph_id"] = source_page_id
    target_record["graph_id"] = target_page_id
    source_device = str(
        source_endpoint.get("device_name")
        or source_graph.metadata.get("device_name")
        or "unknown"
    )
    target_device = str(
        target_endpoint.get("device_name")
        or target_graph.metadata.get("device_name")
        or "unknown"
    )
    return {
        "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
        "pair_id": str(row.get("pair_id") or _stable_pair_id("guiodyssey", source_page_id, target_page_id)),
        "split": split,
        "label_status": "weak",
        "source": {
            "page_id": source_page_id,
            "platform": "android",
            "screenshot_path": str(source_graph.metadata.get("screenshot_path") or ""),
            "graph": source_record,
        },
        "target": {
            "page_id": target_page_id,
            "platform": "android",
            "screenshot_path": str(target_graph.metadata.get("screenshot_path") or ""),
            "graph": target_record,
        },
        "matches": [
            {
                "source_node_id": "action_target",
                "target_node_ids": ["action_target"],
                "label": "correspondence",
            }
        ],
        "partition_keys": [
            f"guiodyssey:task:{task}",
            *(f"guiodyssey:episode:{episode}" for episode in sorted((source_episode, target_episode))),
            f"guiodyssey:trajectory:{trajectory}",
        ],
        "provenance": {
            "dataset": "guiodyssey",
            "annotation": "trajectory_alignment_weak",
            "source_pair_id": str(row.get("pair_id") or ""),
            "selection": deepcopy(selection),
        },
        "slices": {
            "task": task,
            "source_device": source_device,
            "target_device": target_device,
            "source_form_factor": _form_factor(source_device),
            "target_form_factor": _form_factor(target_device),
            "cross_form_factor": _form_factor(source_device) != _form_factor(target_device),
            "browser_or_webview": bool(
                source_endpoint.get("web_or_webview_context")
                or target_endpoint.get("web_or_webview_context")
            ),
        },
    }


def _gui_odyssey_human_record(
    row: dict[str, Any],
    annotation: dict[str, Any],
    source_graph: UIGraph,
    target_graph: UIGraph,
    *,
    split: str,
    label: str,
) -> dict[str, Any]:
    source_endpoint = row.get("source") if isinstance(row.get("source"), dict) else {}
    target_endpoint = row.get("target") if isinstance(row.get("target"), dict) else {}
    source_episode = _nonempty_text(
        source_endpoint.get("episode_id"), "source.episode_id"
    )
    target_episode = _nonempty_text(
        target_endpoint.get("episode_id"), "target.episode_id"
    )
    task = _nonempty_text(
        row.get("meta_task")
        or source_endpoint.get("meta_task")
        or target_endpoint.get("meta_task"),
        "meta_task",
    )
    selection = row.get("selection") if isinstance(row.get("selection"), dict) else {}
    trajectory = str(
        selection.get("trajectory_pair_id")
        or row.get("episode_pair_id")
        or " | ".join(sorted((source_episode, target_episode)))
    )
    source_page_id = _review_page_id(source_endpoint, source_episode, side="source")
    target_page_id = _review_page_id(target_endpoint, target_episode, side="target")
    source_record = graph_to_record(source_graph)
    target_record = graph_to_record(target_graph)
    source_record["graph_id"] = source_page_id
    target_record["graph_id"] = target_page_id
    matches_by_source: dict[str, list[str]] = defaultdict(list)
    if label == "correspondence":
        for match in annotation.get("matches") or ():
            if not isinstance(match, dict):
                raise TypeError("review match is not an object")
            source_id = _nonempty_text(match.get("source_node_id"), "source_node_id")
            target_id = _nonempty_text(match.get("target_node_id"), "target_node_id")
            if target_id not in matches_by_source[source_id]:
                matches_by_source[source_id].append(target_id)
        if not matches_by_source:
            raise ValueError("correspondence review has no matches")
        matches = [
            {
                "source_node_id": source_id,
                "target_node_ids": target_ids,
                "label": "correspondence",
            }
            for source_id, target_ids in matches_by_source.items()
        ]
    else:
        matches = [
            {
                "source_node_id": node.node_id,
                "target_node_ids": [],
                "label": "no_correspondence",
            }
            for node in source_graph.nodes
        ]
    source_device = str(source_endpoint.get("device_name") or "unknown")
    target_device = str(target_endpoint.get("device_name") or "unknown")
    return {
        "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
        "pair_id": str(row.get("pair_id") or _stable_pair_id("guiodyssey-review", source_page_id, target_page_id)),
        "split": split,
        "label_status": "gold",
        "source": {
            "page_id": source_page_id,
            "platform": "android",
            "screenshot_path": str(source_graph.metadata.get("screenshot_path") or ""),
            "graph": source_record,
        },
        "target": {
            "page_id": target_page_id,
            "platform": "android",
            "screenshot_path": str(target_graph.metadata.get("screenshot_path") or ""),
            "graph": target_record,
        },
        "matches": matches,
        "partition_keys": [
            f"guiodyssey:task:{task}",
            *(f"guiodyssey:episode:{episode}" for episode in sorted((source_episode, target_episode))),
            f"guiodyssey:trajectory:{trajectory}",
        ],
        "provenance": {
            "dataset": "guiodyssey",
            "annotation": "human_reviewed_gold",
            "review_label": label,
        },
        "slices": {
            "task": task,
            "source_device": source_device,
            "target_device": target_device,
            "source_form_factor": _form_factor(source_device),
            "target_form_factor": _form_factor(target_device),
            "cross_form_factor": _form_factor(source_device) != _form_factor(target_device),
            "browser_or_webview": bool(
                source_endpoint.get("web_or_webview_context")
                or target_endpoint.get("web_or_webview_context")
            ),
        },
    }


def _validate_page(page: Any, *, side: str) -> set[str]:
    if not isinstance(page, dict):
        raise TypeError(f"{side} page must be an object")
    _required_text(page, "page_id")
    _required_text(page, "platform")
    if not isinstance(page.get("screenshot_path"), str):
        raise TypeError(f"{side}.screenshot_path must be a string")
    graph = page.get("graph")
    if not isinstance(graph, dict):
        raise TypeError(f"{side}.graph must be an object")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{side}.graph.nodes must be a non-empty list")
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"{side}.graph.nodes[{index}] must be an object")
        node_id = _required_text(node, "node_id")
        _required_text(node, "origin_id")
        if node_id in node_ids:
            raise ValueError(f"duplicate {side} node id: {node_id}")
        node_ids.add(node_id)
    return node_ids


def _required_text(value: dict[str, Any], key: str) -> str:
    return _nonempty_text(value.get(key), key)


def _nonempty_text(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is absent")
    return normalized


def _ase_page_id(query: Query, side: str) -> str:
    for key in (f"{side}_screen", f"{side}_xml_path", f"{side}_screenshot_path"):
        value = str(query.metadata.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"ASE query {query.query_id} has no {side} page identity")


def _required_query_metadata(query: Query, key: str) -> str:
    return _nonempty_text(query.metadata.get(key), f"query {query.query_id} metadata.{key}")


def _source_node_id(query: Query) -> str:
    return _nonempty_text(
        query.metadata.get("source_node_id")
        or query.source.get("node_id")
        or query.source.get("id"),
        f"query {query.query_id} source node id",
    )


def _node_record_from_payload(payload: dict[str, Any], *, node_id: str) -> dict[str, Any]:
    bbox = payload.get("bbox", payload.get("bounds"))
    excluded = {
        "id",
        "node_id",
        "origin_id",
        "parent_id",
        "text",
        "content_desc",
        "resource_id",
        "class_name",
        "bbox",
        "bounds",
        "clickable",
        "editable",
        "scrollable",
        "enabled",
        "depth",
        "child_ids",
        "metadata",
    }
    metadata = {
        **(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
        **{key: value for key, value in payload.items() if key not in excluded},
    }
    return {
        "node_id": node_id,
        "origin_id": str(payload.get("origin_id") or node_id),
        "parent_id": payload.get("parent_id"),
        "text": str(payload.get("text") or ""),
        "content_desc": str(payload.get("content_desc") or ""),
        "resource_id": str(payload.get("resource_id") or ""),
        "class_name": str(payload.get("class_name") or ""),
        "bbox": list(bbox) if bbox is not None else None,
        "clickable": bool(payload.get("clickable", False)),
        "editable": bool(payload.get("editable", False)),
        "scrollable": bool(payload.get("scrollable", False)),
        "enabled": bool(payload.get("enabled", True)),
        "depth": int(payload.get("depth") or 0),
        "child_ids": [str(value) for value in payload.get("child_ids") or ()],
        "metadata": metadata,
    }


def _page_graph(
    page_id: str,
    xml_path: str,
    required_nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = Path(xml_path).expanduser() if xml_path else None
    if path is not None and path.is_file():
        graph = graph_to_record(
            graph_from_record(
                {"id": page_id, "hierarchy": path.read_text(encoding="utf-8")},
                graph_id=page_id,
            )
        )
    else:
        graph = {
            "graph_id": page_id,
            "width": None,
            "height": None,
            "nodes": [],
            "metadata": {"source_format": "query_fallback"},
        }
    existing = {str(node.get("node_id")) for node in graph["nodes"]}
    graph["nodes"].extend(
        node for node_id, node in sorted(required_nodes.items()) if node_id not in existing
    )
    return graph


def _resolve_asset_path(value: str, *, asset_root: Path | None) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_absolute() or asset_root is None:
        return str(path)
    return str((asset_root / path).resolve())


def _mark_action_anchors(
    graph: dict[str, Any],
    source_matches: dict[str, list[str]],
    *,
    evidence: str,
) -> None:
    """Record that an authored source mapping row is an executed action anchor."""

    source_ids = set(source_matches)
    for node in graph["nodes"]:
        if str(node.get("node_id")) not in source_ids:
            continue
        node["clickable"] = True
        metadata = node.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            node["metadata"] = metadata
        metadata["actionability_evidence"] = evidence


def _ase_platform(query: Query, side: str) -> str:
    if side == "source":
        source_metadata = query.source.get("metadata")
        if isinstance(source_metadata, dict) and source_metadata.get("platform"):
            return str(source_metadata["platform"]).lower()
    page_id = _ase_page_id(query, side).lower()
    if "ios" in page_id:
        return "ios"
    if "android" in page_id:
        return "android"
    return "unknown"


def _stable_pair_id(namespace: str, source_page_id: str, target_page_id: str) -> str:
    digest = hashlib.blake2b(
        f"{namespace}\0{source_page_id}\0{target_page_id}".encode(),
        digest_size=10,
    ).hexdigest()
    return f"{namespace}-page-pair-{digest}"


def _form_factor(device_name: str) -> str:
    normalized = device_name.lower()
    if "fold" in normalized:
        return "foldable"
    if "tablet" in normalized or "pixel c" in normalized:
        return "tablet"
    if "phone" in normalized or "pixel" in normalized:
        return "phone"
    return "unknown"


def _review_page_id(endpoint: dict[str, Any], episode_id: str, *, side: str) -> str:
    if endpoint.get("step_index") is not None:
        suffix = f"step:{int(endpoint['step_index'])}"
    else:
        asset = str(endpoint.get("image_url") or endpoint.get("screenshot") or side)
        suffix = f"asset:{asset}"
    return f"guiodyssey:{episode_id}:{suffix}"


def _cross_split_values(values: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        key: sorted(splits)
        for key, splits in sorted(values.items())
        if len(splits) > 1
    }


def _assignment_split(record: dict[str, Any]) -> str:
    split = record["split"]
    if split != "diagnostic":
        return split
    reserved_split = str(record["provenance"].get("reserved_split") or "")
    if not reserved_split:
        return split
    if reserved_split not in {"dev", "test"}:
        raise ValueError(f"unsupported diagnostic reserved split: {reserved_split}")
    return reserved_split


def _first_overlap(overlap: dict[str, list[str]]) -> str:
    key = next(iter(overlap))
    return f"{key} -> {overlap[key]}"


def _record_assignment(
    assignments: dict[str, set[str]],
    key: str,
    split: str,
    label: str,
) -> None:
    assigned = assignments[key]
    assigned.add(split)
    if len(assigned) > 1:
        raise ValueError(f"{label} leakage: {key} -> {sorted(assigned)}")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _file_manifest(path: Path, *, records: int, matches: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.name,
        "records": records,
        "matches": matches,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
