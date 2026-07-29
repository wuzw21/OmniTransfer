"""Unified UI correspondence dataset contract for learned UI mapping."""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass, replace
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
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record, graph_to_record


UI_CORRESPONDENCE_PAIR_SCHEMA = "omnitransfer.ui_correspondence_pair.v1"
_LEGACY_UI_CORRESPONDENCE_PAIR_SCHEMAS = frozenset(
    {"omnitransfer.mapping_page_pair.v1"}
)
_SPLITS = frozenset({"train", "dev", "test", "diagnostic"})
_LABEL_STATUSES = frozenset({"gold", "weak", "self_supervised", "unreviewed"})


@dataclass(frozen=True)
class UICorrespondenceSplitPlan:
    """Frozen within-App assignment for one canonical correspondence pool."""

    assignments: dict[str, str]
    component_ids: dict[str, str]
    app_ids: dict[str, str]
    audit: dict[str, Any]
    pool_sha256: str


def adapt_ase_queries(
    queries: list[Query],
    *,
    asset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Group authored ASE queries into gold multi-match UI correspondence records."""

    resolved_assets = (
        Path(asset_root).expanduser().resolve() if asset_root is not None else None
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
            raise ValueError(
                "ASE correspondence pair spans splits: "
                f"{source_page_id} -> {target_page_id}"
            )
        if len(apps) != 1:
            raise ValueError(
                "ASE correspondence pair spans apps: "
                f"{source_page_id} -> {target_page_id}"
            )
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
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
            "pair_id": _stable_pair_id("ase", source_page_id, target_page_id),
            "split": split,
            "label_status": "gold",
            "source": {
                "page_id": source_page_id,
                "platform": _ase_platform(page_queries[0], "source"),
                "screenshot_path": _resolve_asset_path(
                    str(page_queries[0].metadata.get("source_screenshot_path") or ""),
                    asset_root=resolved_assets,
                ),
                "graph": source_graph,
            },
            "target": {
                "page_id": target_page_id,
                "platform": _ase_platform(page_queries[0], "target"),
                "screenshot_path": _resolve_asset_path(
                    str(page_queries[0].metadata.get("target_screenshot_path") or ""),
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
        records.append(validate_ui_correspondence_pair(record))
    return records


def adapt_bmoca_trace_corpus(
    corpus_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt BMOCA offline action alignments into diagnostic correspondence rows.

    BMOCA action alignment is automatic evidence rather than reviewed gold.  The
    adapter therefore fixes every row to ``diagnostic``/``unreviewed`` and keeps
    the original pair ids and alignment scores in provenance.  Endpoint ids such
    as ``e57`` belong to the trace collector, so they are rebound to canonical
    XML-tree node ids using the recorded bbox, class, actionability, and semantic
    attributes.  These fields are used only to construct offline labels.
    """

    root = Path(corpus_root).expanduser().resolve()
    manifest = _read_json_object(root / "manifest.json")
    if manifest.get("schema_version") != "omniflow.offline-trace-corpus.v1":
        raise ValueError("unsupported BMOCA offline trace manifest")
    memory = _read_json_object(root / "pair_memory.json")
    if memory.get("schema_version") != "omniflow.transfer-pair-memory.v1":
        raise ValueError("unsupported BMOCA transfer pair memory")
    raw_pairs = memory.get("pairs")
    if not isinstance(raw_pairs, dict):
        raise TypeError("BMOCA pair_memory.pairs must be an object")

    trace_by_run: dict[str, dict[str, Any]] = {}
    for raw_trace in manifest.get("traces") or ():
        if not isinstance(raw_trace, dict):
            raise TypeError("BMOCA manifest trace must be an object")
        run_id = _nonempty_text(raw_trace.get("run_id"), "trace.run_id")
        catalog = raw_trace.get("state_catalog")
        if not isinstance(catalog, dict):
            raise TypeError(f"BMOCA trace {run_id} has no state catalog")
        relative_path = _nonempty_text(catalog.get("path"), "state_catalog.path")
        trace_by_run[run_id] = {
            **raw_trace,
            "state_catalog_path": (root / relative_path).resolve(),
        }

    state_catalogs: dict[str, dict[str, dict[str, Any]]] = {}
    graph_cache: dict[tuple[str, str], UIGraph] = {}
    executed_anchor_ids_by_graph: dict[str, set[str]] = defaultdict(set)

    def trace_state(run_id: str, state_id: str) -> dict[str, Any]:
        trace = trace_by_run.get(run_id)
        if trace is None:
            raise ValueError(f"BMOCA endpoint references unknown run: {run_id}")
        if run_id not in state_catalogs:
            catalog = _read_json_object(Path(trace["state_catalog_path"]))
            states = catalog.get("states")
            if not isinstance(states, dict):
                raise TypeError(f"BMOCA state catalog {run_id} has invalid states")
            state_catalogs[run_id] = states
        state = state_catalogs[run_id].get(state_id)
        if not isinstance(state, dict):
            raise ValueError(f"BMOCA endpoint references unknown state: {state_id}")
        return state

    def endpoint_graph(
        endpoint: dict[str, Any],
    ) -> tuple[UIGraph, dict[str, Any], tuple[str, str]]:
        run_id = _nonempty_text(endpoint.get("run_id"), "endpoint.run_id")
        state_id = _nonempty_text(
            endpoint.get("state_id") or endpoint.get("page_id"),
            "endpoint.state_id",
        )
        key = (run_id, state_id)
        state = trace_state(run_id, state_id)
        if key not in graph_cache:
            display = state.get("display")
            if not isinstance(display, dict):
                display = {}
            xml = _nonempty_text(state.get("xml"), "state.xml")
            graph_cache[key] = graph_from_record(
                {
                    "id": state_id,
                    "xml": xml,
                    "width": display.get("width") or endpoint.get("width"),
                    "height": display.get("height") or endpoint.get("height"),
                },
                graph_id=_bmoca_page_id(trace_by_run[run_id], state_id),
            )
        return graph_cache[key], trace_by_run[run_id], key

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    unresolved: Counter[str] = Counter()
    input_pairs = 0
    duplicate_edges = 0
    executed_nodes_marked_actionable = 0
    for pair_id, raw_pair in sorted(raw_pairs.items()):
        input_pairs += 1
        if not isinstance(raw_pair, dict):
            unresolved["pair_not_object"] += 1
            continue
        source_endpoint = raw_pair.get("source")
        target_endpoint = raw_pair.get("target")
        if not isinstance(source_endpoint, dict) or not isinstance(
            target_endpoint, dict
        ):
            unresolved["missing_endpoint"] += 1
            continue
        if (
            source_endpoint.get("action_tool") != "click"
            or target_endpoint.get("action_tool") != "click"
        ):
            unresolved["non_click_action"] += 1
            continue
        try:
            source_graph, source_trace, source_graph_key = endpoint_graph(
                source_endpoint
            )
            target_graph, target_trace, target_graph_key = endpoint_graph(
                target_endpoint
            )
            source_node = _bind_bmoca_endpoint(source_endpoint, source_graph)
            target_node = _bind_bmoca_endpoint(target_endpoint, target_graph)
        except (KeyError, TypeError, ValueError) as exc:
            unresolved[f"invalid:{exc}"] += 1
            continue
        if source_node is None:
            unresolved["source_node_unresolved"] += 1
            continue
        if target_node is None:
            unresolved["target_node_unresolved"] += 1
            continue
        if not _is_actionable_ui_node(source_node):
            source_graph = _mark_bmoca_executed_node_actionable(
                source_graph, source_node.node_id
            )
            graph_cache[source_graph_key] = source_graph
            source_node = next(
                node
                for node in source_graph.nodes
                if node.node_id == source_node.node_id
            )
            executed_nodes_marked_actionable += 1
        executed_anchor_ids_by_graph[source_graph.graph_id].add(source_node.node_id)
        if not _is_actionable_ui_node(target_node):
            target_graph = _mark_bmoca_executed_node_actionable(
                target_graph, target_node.node_id
            )
            graph_cache[target_graph_key] = target_graph
            target_node = next(
                node
                for node in target_graph.nodes
                if node.node_id == target_node.node_id
            )
            executed_nodes_marked_actionable += 1
        executed_anchor_ids_by_graph[target_graph.graph_id].add(target_node.node_id)

        group_key = (source_graph.graph_id, target_graph.graph_id)
        group = grouped.setdefault(
            group_key,
            {
                "source_graph": source_graph,
                "target_graph": target_graph,
                "source_graph_key": source_graph_key,
                "target_graph_key": target_graph_key,
                "source_endpoint": source_endpoint,
                "target_endpoint": target_endpoint,
                "source_trace": source_trace,
                "target_trace": target_trace,
                "matches": defaultdict(set),
                "pair_ids": [],
                "tasks": set(),
                "scores": [],
            },
        )
        group["source_graph"] = graph_cache[source_graph_key]
        group["target_graph"] = graph_cache[target_graph_key]
        if target_node.node_id in group["matches"][source_node.node_id]:
            duplicate_edges += 1
        group["matches"][source_node.node_id].add(target_node.node_id)
        group["pair_ids"].append(str(raw_pair.get("pair_id") or pair_id))
        group["tasks"].update(
            str(trace.get("task_id") or "")
            for trace in (source_trace, target_trace)
            if trace.get("task_id")
        )
        evidence = raw_pair.get("evidence")
        if isinstance(evidence, dict) and evidence.get("alignment_score") is not None:
            group["scores"].append(float(evidence["alignment_score"]))

    records: list[dict[str, Any]] = []
    output_correspondences = 0
    for (source_page_id, target_page_id), group in sorted(grouped.items()):
        source_graph = graph_cache[group["source_graph_key"]]
        target_graph = graph_cache[group["target_graph_key"]]
        for node_id in sorted(executed_anchor_ids_by_graph[source_graph.graph_id]):
            source_graph = _mark_bmoca_executed_node_actionable(source_graph, node_id)
        for node_id in sorted(executed_anchor_ids_by_graph[target_graph.graph_id]):
            target_graph = _mark_bmoca_executed_node_actionable(target_graph, node_id)
        source_record = graph_to_record(source_graph)
        target_record = graph_to_record(target_graph)
        source_endpoint = group["source_endpoint"]
        target_endpoint = group["target_endpoint"]
        source_trace = group["source_trace"]
        target_trace = group["target_trace"]
        matches = [
            {
                "source_node_id": source_id,
                "target_node_ids": sorted(target_ids),
                "label": "correspondence",
            }
            for source_id, target_ids in sorted(group["matches"].items())
        ]
        output_correspondences += sum(
            len(match["target_node_ids"]) for match in matches
        )
        tasks = sorted(group["tasks"])
        app = str(
            source_endpoint.get("package_name")
            or target_endpoint.get("package_name")
            or "unknown"
        )
        records.append(
            validate_ui_correspondence_pair(
                {
                    "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
                    "pair_id": _stable_pair_id("bmoca", source_page_id, target_page_id),
                    "split": "diagnostic",
                    "label_status": "unreviewed",
                    "source": {
                        "page_id": source_page_id,
                        "platform": "android",
                        "screenshot_path": _bmoca_screenshot_path(
                            root, source_endpoint
                        ),
                        "graph": source_record,
                    },
                    "target": {
                        "page_id": target_page_id,
                        "platform": "android",
                        "screenshot_path": _bmoca_screenshot_path(
                            root, target_endpoint
                        ),
                        "graph": target_record,
                    },
                    "matches": matches,
                    "partition_keys": [
                        *(f"bmoca:task:{task}" for task in tasks),
                        (
                            "bmoca:environment-pair:"
                            f"{source_trace.get('environment_id', 'unknown')}"
                            f"->{target_trace.get('environment_id', 'unknown')}"
                        ),
                    ],
                    "provenance": {
                        "dataset": "bmoca_three_environment_aligned_success",
                        "annotation": "offline_trace_dp_alignment_unreviewed",
                        "source_pair_ids": sorted(set(group["pair_ids"])),
                        "alignment_score": {
                            "minimum": min(group["scores"])
                            if group["scores"]
                            else None,
                            "maximum": max(group["scores"])
                            if group["scores"]
                            else None,
                        },
                        "offline_label_binding": (
                            "bbox_class_actionability_semantics_to_xml_node"
                        ),
                    },
                    "slices": {
                        "app": app,
                        "tasks": tasks,
                        "source_environment": str(
                            source_trace.get("environment_id") or "unknown"
                        ),
                        "target_environment": str(
                            target_trace.get("environment_id") or "unknown"
                        ),
                        "action": "click",
                    },
                }
            )
        )

    report = {
        "schema_version": "omnitransfer.bmoca_trace_adapter.v1",
        "input_pairs": input_pairs,
        "output_records": len(records),
        "output_match_rows": sum(len(record["matches"]) for record in records),
        "output_correspondences": output_correspondences,
        "duplicate_edges_collapsed": duplicate_edges,
        "executed_nodes_marked_actionable": executed_nodes_marked_actionable,
        "unresolved_endpoints": sum(unresolved.values()),
        "skipped": dict(sorted(unresolved.items())),
        "split": "diagnostic",
        "label_status": "unreviewed",
        "formal_gold": False,
        "full_xml_graphs": len(graph_cache),
    }
    return records, report


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
        Path(screenshot_dir).expanduser().resolve()
        if screenshot_dir is not None
        else None
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
            record = _gui_odyssey_weak_record(
                row, pair.graph_a, pair.graph_b, split=split
            )
        except _ActionPointOutsideBBox:
            skipped["action_bbox_excludes_point"] += 1
            continue
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            skipped[f"invalid_pair:{exc}"] += 1
            continue
        records.append(validate_ui_correspondence_pair(record))
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
            records.append(validate_ui_correspondence_pair(record))
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


def write_ui_correspondence_pool(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze heterogeneous inputs as one canonical, unsplit JSONL pool."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pair_ids: set[str] = set()
    record_count = 0
    match_count = 0
    dataset_counts: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw_record in records:
                record = validate_ui_correspondence_pair(raw_record)
                pair_id = record["pair_id"]
                if pair_id in pair_ids:
                    raise ValueError(f"duplicate pair id in unified pool: {pair_id}")
                pair_ids.add(pair_id)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                record_count += 1
                match_count += len(record["matches"])
                dataset_counts[
                    str(record["provenance"].get("dataset") or "unknown")
                ] += 1
        if record_count == 0:
            raise ValueError("UI correspondence pool is empty")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "omnitransfer.ui_correspondence_pool_manifest.v1",
        "record_schema": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "path": path.name,
        "records": record_count,
        "matches": match_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_semantics": "source_declaration_only_until_pool_split",
    }


def plan_ui_correspondence_pool(
    pool_path: str | Path,
    *,
    seed: int = 17,
    train_percent: int = 80,
    dev_percent: int = 10,
) -> UICorrespondenceSplitPlan:
    """Plan a deterministic page-disjoint split inside each App."""

    if train_percent <= 0 or dev_percent < 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave non-empty train and test ranges")
    path = Path(pool_path).expanduser().resolve()
    union = _StringUnionFind()
    pair_ids: set[str] = set()
    page_keys_by_pair: dict[str, tuple[str, str]] = {}
    app_ids: dict[str, str] = {}
    dataset_counts: Counter[str] = Counter()
    for record in _iter_ui_correspondence_pool(path):
        pair_id = record["pair_id"]
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair id in unified pool: {pair_id}")
        pair_ids.add(pair_id)
        app_id = _correspondence_app_id(record)
        source_page = f"{app_id}\0{record['source']['page_id']}"
        target_page = f"{app_id}\0{record['target']['page_id']}"
        union.join(source_page, target_page)
        page_keys_by_pair[pair_id] = (source_page, target_page)
        app_ids[pair_id] = app_id
        dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
    if not pair_ids:
        raise ValueError("UI correspondence pool is empty")

    component_pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    component_pages: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair_id, page_keys in page_keys_by_pair.items():
        component_key = (app_ids[pair_id], union.root(page_keys[0]))
        component_pairs[component_key].append(pair_id)
        component_pages[component_key].update(page_keys)

    components_by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_ids: dict[str, str] = {}
    for component_key, pair_ids in component_pairs.items():
        app_id = component_key[0]
        pages = sorted(component_pages[component_key])
        component_id = hashlib.blake2b(
            f"{app_id}\0{chr(0).join(pages)}".encode(),
            digest_size=10,
        ).hexdigest()
        ordered_pair_ids = tuple(sorted(pair_ids))
        components_by_app[app_id].append(
            {
                "component_id": component_id,
                "pair_ids": ordered_pair_ids,
                "pages": tuple(pages),
                "size": len(ordered_pair_ids),
            }
        )
        for pair_id in ordered_pair_ids:
            component_ids[pair_id] = component_id

    percentages = {
        "train": train_percent,
        "dev": dev_percent,
        "test": 100 - train_percent - dev_percent,
    }
    assignments: dict[str, str] = {}
    component_assignments: dict[str, str] = {}
    app_split_counts: dict[str, dict[str, int]] = {}
    for app_id, components in sorted(components_by_app.items()):
        assigned = _assign_app_components(
            components,
            seed=seed,
            percentages=percentages,
        )
        counts: Counter[str] = Counter()
        for component in components:
            component_id = component["component_id"]
            split = assigned[component_id]
            component_assignments[component_id] = split
            for pair_id in component["pair_ids"]:
                assignments[pair_id] = split
                counts[split] += 1
        app_split_counts[app_id] = {
            split: counts[split] for split in ("train", "dev", "test")
        }

    page_splits: dict[str, set[str]] = defaultdict(set)
    app_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for pair_id, split in assignments.items():
        split_counts[split] += 1
        app_splits[app_ids[pair_id]].add(split)
        for page_key in page_keys_by_pair[pair_id]:
            page_splits[page_key].add(split)
    page_overlap = _cross_split_values(page_splits)
    if page_overlap:
        raise AssertionError(
            f"page leakage in pool split: {_first_overlap(page_overlap)}"
        )
    apps_in_multiple_splits = sorted(
        app_id for app_id, splits in app_splits.items() if len(splits) > 1
    )
    audit = {
        "schema_version": "omnitransfer.ui_correspondence_pool_split.v1",
        "seed": seed,
        "requested_percentages": percentages,
        "app_protocol": "within_app_page_component",
        "dataset_protocol": "one_mixed_pool",
        "records": len(assignments),
        "components": len(component_assignments),
        "apps": len(app_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "apps_in_multiple_splits": apps_in_multiple_splits,
        "apps_in_multiple_splits_count": len(apps_in_multiple_splits),
        "app_split_counts": app_split_counts,
        "overlap": {
            "pair_id": {},
            "page_id": {},
            "component_id": {},
        },
    }
    return UICorrespondenceSplitPlan(
        assignments=assignments,
        component_ids=component_ids,
        app_ids=app_ids,
        audit=audit,
        pool_sha256=_sha256_file(path),
    )


def iter_split_ui_correspondence_pool(
    pool_path: str | Path,
    plan: UICorrespondenceSplitPlan,
) -> Iterable[dict[str, Any]]:
    """Apply a frozen pool plan and emit canonical train/review/test records."""

    path = Path(pool_path).expanduser().resolve()
    if _sha256_file(path) != plan.pool_sha256:
        raise ValueError("unified correspondence pool changed after split planning")
    seen: set[str] = set()
    for source_record in _iter_ui_correspondence_pool(path):
        record = deepcopy(source_record)
        pair_id = record["pair_id"]
        if pair_id not in plan.assignments:
            raise ValueError(f"pair is absent from split plan: {pair_id}")
        assigned_split = plan.assignments[pair_id]
        provenance = record["provenance"]
        provenance["source_split"] = record["split"]
        provenance["source_label_status"] = record["label_status"]
        provenance.pop("reserved_split", None)
        provenance["split_protocol"] = "within_app_page_component_v1"
        record["partition_keys"] = [f"pool:component:{plan.component_ids[pair_id]}"]
        record["slices"]["app"] = plan.app_ids[pair_id]
        if _is_mobileviews_self_supervised_proposal(record):
            record["split"] = assigned_split
            record["label_status"] = "self_supervised"
        elif assigned_split == "train":
            record["split"] = "train"
        elif record["label_status"] == "gold":
            record["split"] = assigned_split
        else:
            record["split"] = "diagnostic"
            record["label_status"] = "unreviewed"
            provenance["reserved_split"] = assigned_split
        seen.add(pair_id)
        yield validate_ui_correspondence_pair(record)
    missing = sorted(set(plan.assignments) - seen)
    if missing:
        raise ValueError(f"split plan contains pairs absent from pool: {missing[:3]}")


def audit_ui_correspondence_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate rows and reject pair, page, or partition leakage across splits."""

    if not records:
        raise ValueError("UI correspondence dataset is empty")
    normalized = [validate_ui_correspondence_pair(record) for record in records]
    pair_splits: dict[str, set[str]] = defaultdict(set)
    page_splits: dict[str, set[str]] = defaultdict(set)
    partition_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
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
        split_label_counts[split][record["label_status"]] += 1
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
        "schema_version": "omnitransfer.ui_correspondence_audit.v1",
        "valid": True,
        "records": len(normalized),
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "split_label_status_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "match_label_counts": dict(sorted(match_counts.items())),
        "overlap": {"pair_id": {}, "page_id": {}, "partition_key": {}},
    }


def write_ui_correspondence_dataset(
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
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
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
            record = validate_ui_correspondence_pair(raw_record)
            split = record["split"]
            assignment_split = _assignment_split(record)
            _record_assignment(
                pair_splits, record["pair_id"], assignment_split, "pair id"
            )
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
            split_label_counts[split][record["label_status"]] += 1
            dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
            file_match_counts[split] += len(record["matches"])
            for match in record["matches"]:
                match_counts[match["label"]] += 1
        if record_count == 0:
            raise ValueError("UI correspondence dataset is empty")
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
        "schema_version": "omnitransfer.ui_correspondence_audit.v1",
        "valid": True,
        "records": record_count,
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "split_label_status_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
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
        rows = [validate_ui_correspondence_pair(record) for record in candidates]
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
        "schema_version": "omnitransfer.ui_correspondence_dataset_manifest.v1",
        "record_schema": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "audit": audit,
        "files": files,
        "metadata": deepcopy(metadata or {}),
    }
    _write_text_atomic(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def validate_ui_correspondence_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized UI correspondence record or reject an invalid dataset row."""

    if not isinstance(record, dict):
        raise TypeError("UI correspondence pair must be an object")
    value = deepcopy(record)
    if value.get("schema_version") not in {
        UI_CORRESPONDENCE_PAIR_SCHEMA,
        *_LEGACY_UI_CORRESPONDENCE_PAIR_SCHEMAS,
    }:
        raise ValueError("unsupported UI correspondence-pair schema")
    value["schema_version"] = UI_CORRESPONDENCE_PAIR_SCHEMA
    _required_text(value, "pair_id")
    split = _required_text(value, "split")
    if split not in _SPLITS:
        raise ValueError(f"unsupported split: {split}")
    label_status = _required_text(value, "label_status")
    if label_status not in _LABEL_STATUSES:
        raise ValueError(f"unsupported label status: {label_status}")
    if split in {"dev", "test"} and label_status not in {
        "gold",
        "self_supervised",
    }:
        raise ValueError(f"{split} contains unsupported labels: {label_status}")

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
        normalized_targets = list(
            dict.fromkeys(str(node_id) for node_id in target_node_ids)
        )
        if label == "correspondence" and not normalized_targets:
            raise ValueError("correspondence match has no target nodes")
        if label == "no_correspondence" and normalized_targets:
            raise ValueError("no_correspondence match contains target nodes")
        if label not in {"correspondence", "no_correspondence"}:
            raise ValueError(f"unsupported match label: {label}")
        missing = [
            node_id for node_id in normalized_targets if node_id not in target_ids
        ]
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
        "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "pair_id": str(
            row.get("pair_id")
            or _stable_pair_id("guiodyssey", source_page_id, target_page_id)
        ),
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
            *(
                f"guiodyssey:episode:{episode}"
                for episode in sorted((source_episode, target_episode))
            ),
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
            "cross_form_factor": _form_factor(source_device)
            != _form_factor(target_device),
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
        "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "pair_id": str(
            row.get("pair_id")
            or _stable_pair_id("guiodyssey-review", source_page_id, target_page_id)
        ),
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
            *(
                f"guiodyssey:episode:{episode}"
                for episode in sorted((source_episode, target_episode))
            ),
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
            "cross_form_factor": _form_factor(source_device)
            != _form_factor(target_device),
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
    return _nonempty_text(
        query.metadata.get(key), f"query {query.query_id} metadata.{key}"
    )


def _source_node_id(query: Query) -> str:
    return _nonempty_text(
        query.metadata.get("source_node_id")
        or query.source.get("node_id")
        or query.source.get("id"),
        f"query {query.query_id} source node id",
    )


def _node_record_from_payload(
    payload: dict[str, Any], *, node_id: str
) -> dict[str, Any]:
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
        **(
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        ),
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


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _bmoca_page_id(trace: dict[str, Any], state_id: str) -> str:
    environment = _nonempty_text(
        trace.get("environment_id") or "unknown",
        "trace.environment_id",
    )
    return f"bmoca:environment:{environment}:state:{state_id}"


def _bmoca_screenshot_path(root: Path, endpoint: dict[str, Any]) -> str:
    value = str(endpoint.get("screenshot_path") or "")
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _bind_bmoca_endpoint(
    endpoint: dict[str, Any],
    graph: UIGraph,
) -> UINode | None:
    """Resolve one collector-local endpoint id to one canonical XML node."""

    raw_node = endpoint.get("node")
    if not isinstance(raw_node, dict):
        raise TypeError("BMOCA endpoint.node must be an object")
    bounds = raw_node.get("bounds", raw_node.get("bbox"))
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("BMOCA endpoint node has invalid bounds")
    wanted_bbox = tuple(float(value) for value in bounds)
    attributes = raw_node.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    wanted_class = _bmoca_class_name(
        attributes.get("class") or raw_node.get("class_name")
    )
    candidates = [
        node
        for node in graph.nodes
        if node.bbox is not None
        and all(
            abs(actual - wanted) <= 0.5
            for actual, wanted in zip(node.bbox, wanted_bbox, strict=True)
        )
        and (not wanted_class or _bmoca_class_name(node.class_name) == wanted_class)
    ]
    if not candidates:
        return None

    expects_actionable = bool(
        attributes.get("clickable")
        or attributes.get("editable")
        or attributes.get("scrollable")
    )
    if expects_actionable:
        actionable = [
            node
            for node in candidates
            if node.enabled and (node.clickable or node.editable or node.scrollable)
        ]
        if actionable:
            candidates = actionable

    wanted_resource = _bmoca_resource_leaf(
        attributes.get("resource_id") or raw_node.get("resource_id")
    )
    if wanted_resource:
        resource_matches = [
            node
            for node in candidates
            if _bmoca_resource_leaf(node.resource_id) == wanted_resource
        ]
        if resource_matches:
            candidates = resource_matches

    for raw_value, field in (
        (
            attributes.get("text", raw_node.get("text")),
            lambda node: node.text,
        ),
        (
            attributes.get(
                "content_description",
                raw_node.get("content_desc"),
            ),
            lambda node: node.content_desc,
        ),
    ):
        wanted = _bmoca_semantic_text(raw_value)
        if not wanted:
            continue
        semantic_matches = [
            node for node in candidates if _bmoca_semantic_text(field(node)) == wanted
        ]
        if semantic_matches:
            candidates = semantic_matches

    return candidates[0] if len(candidates) == 1 else None


def _is_actionable_ui_node(node: UINode) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def _mark_bmoca_executed_node_actionable(
    graph: UIGraph,
    node_id: str,
) -> UIGraph:
    """Use the observed click only as offline evidence of actionability."""

    found = False
    nodes: list[UINode] = []
    for node in graph.nodes:
        if node.node_id != node_id:
            nodes.append(node)
            continue
        found = True
        nodes.append(
            replace(
                node,
                clickable=True,
                metadata={
                    **node.metadata,
                    "actionability_evidence": "bmoca_executed_click",
                },
            )
        )
    if not found:
        raise ValueError(f"BMOCA action node is absent from graph: {node_id}")
    return replace(graph, nodes=tuple(nodes))


def _bmoca_class_name(value: Any) -> str:
    return str(value or "").strip().casefold().rsplit(".", 1)[-1]


def _bmoca_resource_leaf(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    return normalized.rsplit("/", 1)[-1]


def _bmoca_semantic_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


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
        node
        for node_id, node in sorted(required_nodes.items())
        if node_id not in existing
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
    return f"{namespace}-correspondence-{digest}"


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


def _iter_ui_correspondence_pool(path: Path) -> Iterable[dict[str, Any]]:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = validate_ui_correspondence_pair(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid unified correspondence pool {path}:{line_number}: {exc}"
                ) from exc
            rows += 1
            yield record
    if rows == 0:
        raise ValueError(f"UI correspondence pool is empty: {path}")


def _correspondence_app_id(record: dict[str, Any]) -> str:
    slices = record["slices"]
    provenance = record["provenance"]
    for values in (slices, provenance):
        for key in (
            "app",
            "app_id",
            "package",
            "package_name",
            "app_package",
            "domain",
            "website",
        ):
            normalized = str(values.get(key) or "").strip()
            if normalized:
                return _normalize_app_id(normalized)
    trace = str(provenance.get("trace") or "").strip()
    if trace:
        return _normalize_app_id(trace)
    dataset = str(provenance.get("dataset") or "unknown").strip()
    return f"{dataset}:unscoped"


def _normalize_app_id(value: str) -> str:
    normalized = value.strip()
    if normalized.lower().endswith(".apk"):
        normalized = normalized[:-4]
    return normalized


def _assign_app_components(
    components: list[dict[str, Any]],
    *,
    seed: int,
    percentages: dict[str, int],
) -> dict[str, str]:
    if not components:
        return {}
    stable = sorted(
        components,
        key=lambda component: (
            int(component["size"]),
            _stable_bucket(seed, str(component["component_id"])),
            str(component["component_id"]),
        ),
    )
    assignments: dict[str, str] = {}
    counts: Counter[str] = Counter()
    remaining = list(stable)

    reserve = ["test"]
    if percentages["dev"] > 0:
        reserve.append("dev")
    for split in reserve:
        if len(remaining) <= 1:
            break
        component = remaining.pop(0)
        component_id = str(component["component_id"])
        assignments[component_id] = split
        counts[split] += int(component["size"])

    total = sum(int(component["size"]) for component in components)
    targets = {
        split: total * percentage / 100.0 for split, percentage in percentages.items()
    }
    enabled_splits = tuple(
        split for split in ("train", "dev", "test") if percentages[split] > 0
    )
    for component in sorted(
        remaining,
        key=lambda item: (
            -int(item["size"]),
            _stable_bucket(seed, str(item["component_id"])),
            str(item["component_id"]),
        ),
    ):
        split = max(
            enabled_splits,
            key=lambda name: (
                targets[name] - counts[name],
                percentages[name],
                name == "train",
            ),
        )
        component_id = str(component["component_id"])
        assignments[component_id] = split
        counts[split] += int(component["size"])
    return assignments


def _stable_bucket(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _is_mobileviews_self_supervised_proposal(record: dict[str, Any]) -> bool:
    if record["label_status"] not in {"unreviewed", "self_supervised"}:
        return False
    provenance = record["provenance"]
    dataset = str(provenance.get("dataset") or "").lower()
    annotation = str(provenance.get("annotation") or "").lower()
    return "mobileviews" in dataset and annotation.startswith("automatic_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StringUnionFind:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def root(self, value: str) -> str:
        self.parents.setdefault(value, value)
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.root(parent)
        return self.parents[value]

    def join(self, left: str, right: str) -> None:
        left_root = self.root(left)
        right_root = self.root(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def _cross_split_values(values: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        key: sorted(splits) for key, splits in sorted(values.items()) if len(splits) > 1
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


# Read-only compatibility names for artifacts produced before the RCAM naming
# migration. Validation always normalizes legacy rows to the canonical schema.
MAPPING_PAGE_PAIR_SCHEMA = UI_CORRESPONDENCE_PAIR_SCHEMA
validate_mapping_page_pair = validate_ui_correspondence_pair
audit_mapping_page_pairs = audit_ui_correspondence_pairs
write_mapping_dataset = write_ui_correspondence_dataset
