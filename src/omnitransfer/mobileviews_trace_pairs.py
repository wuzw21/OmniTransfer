"""Conservative page-pair proposals from MobileViews complete traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from omnitransfer.mapping_dataset import (
    MAPPING_PAGE_PAIR_SCHEMA,
    validate_mapping_page_pair,
)
from omnitransfer.mobileviews import attach_mobileviews_image, graph_from_mobileviews_record
from omnitransfer.ui_graph import UIGraph, UINode, graph_to_record


@dataclass(frozen=True)
class MobileViewsTraceState:
    """One state referenced by ``screenshot_state_mapping.csv``."""

    state_id: str
    state_str: str
    structure_str: str
    screenshot_path: Path
    json_path: Path


def build_mobileviews_trace_pair_pilot(
    trace_dir: str | Path,
    *,
    pair_limit: int = 20,
    minimum_matches: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic, unreviewed page-pair proposals from one trace.

    ``structure_str`` is used only to retrieve candidate page pairs. Node labels
    are proposed from DroidBot ``view_str`` values and remain explicitly
    unreviewed. Repeated target identities produce a set-valued correspondence.
    """

    root = Path(trace_dir).expanduser().resolve()
    if pair_limit <= 0:
        raise ValueError("pair_limit must be positive")
    if minimum_matches <= 0:
        raise ValueError("minimum_matches must be positive")
    states = load_mobileviews_trace_states(root)
    groups: dict[str, list[MobileViewsTraceState]] = defaultdict(list)
    for state in states:
        groups[state.structure_str].append(state)

    candidates: list[tuple[tuple[float, int, str, str], dict[str, Any], dict[str, Any]]] = []
    rejected = Counter()
    for structure_str, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        for source, target in itertools.combinations(sorted(group, key=lambda item: item.state_id), 2):
            try:
                record, stats = _build_pair(root, source, target)
            except ValueError as exc:
                reason = (
                    "no_correspondence_proposals"
                    if "no conservative correspondence proposals" in str(exc)
                    else "invalid_state"
                )
                rejected[reason] += 1
                continue
            except (OSError, json.JSONDecodeError):
                rejected["invalid_state"] += 1
                continue
            if stats["match_rows"] < minimum_matches:
                rejected["too_few_matches"] += 1
                continue
            score = (
                float(stats["changed_semantic_fraction"]),
                int(stats["match_rows"]),
                source.state_id,
                target.state_id,
            )
            candidates.append((score, record, stats))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:pair_limit]
    records = [item[1] for item in selected]
    stats = [item[2] for item in selected]
    manifest = {
        "schema_version": "omnitransfer.mobileviews_trace_pair_pilot.v1",
        "trace_dir": str(root),
        "states": len(states),
        "structure_groups": len(groups),
        "repeated_structure_groups": sum(len(group) > 1 for group in groups.values()),
        "candidate_pairs": len(candidates),
        "selected_pairs": len(records),
        "match_rows": sum(item["match_rows"] for item in stats),
        "single_target_rows": sum(item["single_target_rows"] for item in stats),
        "set_valued_rows": sum(item["set_valued_rows"] for item in stats),
        "target_links": sum(item["target_links"] for item in stats),
        "rejected": dict(sorted(rejected.items())),
        "label_boundary": {
            "status": "unreviewed",
            "split": "diagnostic",
            "page_retrieval_signal": "structure_str",
            "node_proposal_signal": "view_str",
            "matcher_inputs_must_exclude": ["structure_str", "state_str", "view_str", "origin_id"],
            "null_labels": "not_proposed_without_human_review",
        },
    }
    return records, manifest


def load_mobileviews_trace_states(trace_dir: str | Path) -> list[MobileViewsTraceState]:
    """Load and resolve the state mapping of one extracted complete trace."""

    root = Path(trace_dir).expanduser().resolve()
    mapping_path = root / "screenshot_state_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    states: list[MobileViewsTraceState] = []
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            screenshot = _resolve_trace_member(root, str(row.get("screen_id") or ""))
            state_json = _resolve_trace_member(root, str(row.get("vh_json_id") or ""))
            state_id = state_json.stem.removeprefix("state_")
            structure_str = str(row.get("structure_str") or "").strip()
            if not state_id or not structure_str:
                continue
            states.append(
                MobileViewsTraceState(
                    state_id=state_id,
                    state_str=str(row.get("state_str") or "").strip(),
                    structure_str=structure_str,
                    screenshot_path=screenshot,
                    json_path=state_json,
                )
            )
    return states


def _build_pair(
    trace_dir: Path,
    source: MobileViewsTraceState,
    target: MobileViewsTraceState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_payload = json.loads(source.json_path.read_text(encoding="utf-8"))
    target_payload = json.loads(target.json_path.read_text(encoding="utf-8"))
    source_graph, source_labels = _labelled_graph(source, source_payload)
    target_graph, target_labels = _labelled_graph(target, target_payload)

    target_by_view: dict[str, list[str]] = defaultdict(list)
    for node_id, label in target_labels.items():
        if label["eligible"] and label["view_str"]:
            target_by_view[label["view_str"]].append(node_id)
    matches: list[dict[str, Any]] = []
    for node in source_graph.nodes:
        label = source_labels[node.node_id]
        target_ids = sorted(target_by_view.get(label["view_str"], ()))
        if label["eligible"] and target_ids:
            matches.append(
                {
                    "source_node_id": node.node_id,
                    "target_node_ids": target_ids,
                    "label": "correspondence",
                }
            )
    if not matches:
        raise ValueError("trace pair has no conservative correspondence proposals")

    trace_name = trace_dir.name
    source_page_id = f"mobileviews:{trace_name}:state:{source.state_id}"
    target_page_id = f"mobileviews:{trace_name}:state:{target.state_id}"
    source_record = graph_to_record(source_graph)
    target_record = graph_to_record(target_graph)
    source_record["graph_id"] = source_page_id
    target_record["graph_id"] = target_page_id
    pair_id = _pair_id(trace_name, source.state_id, target.state_id)
    record = validate_mapping_page_pair(
        {
            "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": "diagnostic",
            "label_status": "unreviewed",
            "source": {
                "page_id": source_page_id,
                "platform": "android",
                "screenshot_path": str(source.screenshot_path),
                "graph": source_record,
            },
            "target": {
                "page_id": target_page_id,
                "platform": "android",
                "screenshot_path": str(target.screenshot_path),
                "graph": target_record,
            },
            "matches": matches,
            "partition_keys": [f"mobileviews:trace:{trace_name}"],
            "provenance": {
                "dataset": "MobileViews_Apps_CompleteTraces",
                "annotation": "automatic_same_view_str_proposal",
                "trace": trace_name,
                "source_state_str": source.state_str,
                "target_state_str": target.state_str,
                "structure_str": source.structure_str,
                "label_only_fields": ["structure_str", "state_str", "view_str", "origin_id"],
            },
            "slices": {
                "platform": "android",
                "same_app": True,
                "same_structure": True,
                "dynamic_state_change": source.state_str != target.state_str,
                "cross_form_factor": False,
                "track": "trace_state_variation",
            },
        }
    )
    target_counts = [len(match["target_node_ids"]) for match in matches]
    semantic_source = {
        (label["text"], label["content_desc"], label["resource_id"])
        for label in source_labels.values()
        if label["eligible"]
    }
    semantic_target = {
        (label["text"], label["content_desc"], label["resource_id"])
        for label in target_labels.values()
        if label["eligible"]
    }
    union = semantic_source | semantic_target
    stats = {
        "match_rows": len(matches),
        "single_target_rows": sum(count == 1 for count in target_counts),
        "set_valued_rows": sum(count > 1 for count in target_counts),
        "target_links": sum(target_counts),
        "changed_semantic_fraction": 1.0 - (len(semantic_source & semantic_target) / max(1, len(union))),
    }
    return record, stats


def _labelled_graph(
    state: MobileViewsTraceState,
    payload: dict[str, Any],
) -> tuple[UIGraph, dict[str, dict[str, Any]]]:
    raw_views = [view for view in payload.get("views", ()) if isinstance(view, dict)]
    enriched = dict(payload)
    enriched_views: list[dict[str, Any]] = []
    label_by_raw_id: dict[str, dict[str, Any]] = {}
    occurrence = Counter()
    for index, raw in enumerate(raw_views):
        view = dict(raw)
        raw_id = str(view.get("temp_id", index))
        view_str = str(view.get("view_str") or "").strip()
        occurrence[view_str] += 1
        view["node_id"] = f"state-{state.state_id}-node-{raw_id}"
        view["parent_id"] = (
            None
            if view.get("parent") in (None, -1, "-1")
            else f"state-{state.state_id}-node-{view['parent']}"
        )
        view["origin_id"] = f"mobileviews:{view_str or 'missing'}:{occurrence[view_str]}"
        enriched_views.append(view)
        label_by_raw_id[view["node_id"]] = {
            "view_str": view_str,
            "text": str(view.get("text") or "").strip(),
            "content_desc": str(view.get("content_description") or "").strip(),
            "resource_id": str(view.get("resource_id") or "").strip(),
            "eligible": _eligible_anchor(view),
        }
    enriched["views"] = enriched_views
    graph = graph_from_mobileviews_record(
        enriched,
        graph_id=f"mobileviews:{state.state_id}",
    )
    if state.screenshot_path.is_file():
        try:
            from PIL import Image

            with Image.open(state.screenshot_path) as image:
                image_size = image.size
        except Exception:
            image_size = None
        graph = attach_mobileviews_image(
            graph,
            screenshot_path=str(state.screenshot_path),
            image_size=image_size,
        )
    nodes = tuple(
        UINode(
            **{
                **node.__dict__,
                "metadata": {
                    **node.metadata,
                    "label_view_str": label_by_raw_id[node.node_id]["view_str"],
                    "label_only": True,
                },
            }
        )
        for node in graph.nodes
    )
    return UIGraph(**{**graph.__dict__, "nodes": nodes}), label_by_raw_id


def _eligible_anchor(view: dict[str, Any]) -> bool:
    if view.get("visible") is False or not str(view.get("view_str") or "").strip():
        return False
    semantic = any(
        str(view.get(key) or "").strip()
        for key in ("text", "content_description", "resource_id")
    )
    return bool(
        semantic
        or view.get("clickable")
        or view.get("editable")
        or view.get("scrollable")
    )


def _resolve_trace_member(root: Path, value: str) -> Path:
    relative = Path(value.strip())
    direct = root / relative
    if direct.exists():
        return direct
    if relative.parts and relative.parts[0] == root.name:
        stripped = root.joinpath(*relative.parts[1:])
        if stripped.exists():
            return stripped
    return direct


def _pair_id(trace_name: str, source_state: str, target_state: str) -> str:
    digest = hashlib.blake2b(
        f"{trace_name}\0{source_state}\0{target_state}".encode(), digest_size=10
    ).hexdigest()
    return f"mobileviews-trace-pair-{digest}"
