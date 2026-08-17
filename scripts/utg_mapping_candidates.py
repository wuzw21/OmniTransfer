#!/usr/bin/env python3
"""Internal UTG candidate mining for the canonical point-mapping reviewer."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SUPPORTED_ACTION_TYPES = frozenset({"tap", "long_press", "input_text"})
TARGET_ROLES = ("small", "fold")
EVIDENCE_KEYS = ("before_xml", "before_screenshot", "after_xml", "after_screenshot")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def action_target(action: dict[str, Any]) -> dict[str, Any]:
    target = action.get("target")
    return target if isinstance(target, dict) else action


def action_type(action: dict[str, Any]) -> str:
    return normalize_text(action.get("type"))


def class_name(action: dict[str, Any]) -> str:
    return normalize_text(action_target(action).get("class_name")).rsplit(".", 1)[-1]


def resource_id(action: dict[str, Any]) -> str:
    return normalize_text(action_target(action).get("resource_id"))


def visible_text(action: dict[str, Any]) -> str:
    return normalize_text(action_target(action).get("text"))


def valid_bounds(action: dict[str, Any]) -> bool:
    bounds = action_target(action).get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return False
    try:
        left, top, right, bottom = (float(value) for value in bounds)
    except (TypeError, ValueError):
        return False
    return right > left and bottom > top and left >= 0 and top >= 0


def _tokens(value: Any) -> set[str]:
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value or ""))
    return {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", raw.lower())
        if len(token) > 1 and token not in {"com", "android", "id"}
    }


def _jaccard(left: Any, right: Any) -> float | None:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return None
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _string_similarity(left: Any, right: Any) -> float | None:
    left_value, right_value = normalize_text(left), normalize_text(right)
    if not left_value or not right_value:
        return None
    return SequenceMatcher(None, left_value, right_value).ratio()


def _class_similarity(source_action: dict[str, Any], target_action: dict[str, Any]) -> float:
    source_class, target_class = class_name(source_action), class_name(target_action)
    if not source_class or not target_class:
        return 0.0
    if source_class == target_class:
        return 1.0
    families = (
        {"imageview", "imagebutton"},
        {"button", "textview"},
        {"edittext", "autocompletetextview"},
    )
    return 0.65 if any(source_class in family and target_class in family for family in families) else 0.0


def _geometry_similarity(
    source_action: dict[str, Any],
    source_page: dict[str, Any],
    target_action: dict[str, Any],
    target_page: dict[str, Any],
) -> float | None:
    source_bounds = action_target(source_action).get("bounds")
    target_bounds = action_target(target_action).get("bounds")
    source_width, source_height = source_page.get("width"), source_page.get("height")
    target_width, target_height = target_page.get("width"), target_page.get("height")
    if not source_bounds or not target_bounds or not source_width or not source_height or not target_width or not target_height:
        return None
    source_x = (source_bounds[0] + source_bounds[2]) / (2 * source_width)
    source_y = (source_bounds[1] + source_bounds[3]) / (2 * source_height)
    target_x = (target_bounds[0] + target_bounds[2]) / (2 * target_width)
    target_y = (target_bounds[1] + target_bounds[3]) / (2 * target_height)
    return max(0.0, 1.0 - math.hypot(source_x - target_x, source_y - target_y) / 0.75)


def _effect_similarity(source_edge: dict[str, Any], target_edge: dict[str, Any]) -> float | None:
    source_effect, target_effect = source_edge.get("effect") or {}, target_edge.get("effect") or {}
    keys = ("state_changed", "xml_changed", "activity_changed")
    available = [key for key in keys if key in source_effect and key in target_effect]
    if not available:
        return None
    return sum(bool(source_effect[key]) == bool(target_effect[key]) for key in available) / len(available)


def action_signature(action: dict[str, Any], page: dict[str, Any]) -> tuple[Any, ...]:
    action_kind, resource, text, class_value = action_type(action), resource_id(action), visible_text(action), class_name(action)
    if resource:
        return action_kind, "resource", resource, class_value
    if text:
        return action_kind, "text", text, class_value
    bounds = action_target(action).get("bounds") or [0, 0, 0, 0]
    width, height = max(1, float(page.get("width") or 1)), max(1, float(page.get("height") or 1))
    center_x = (float(bounds[0]) + float(bounds[2])) / (2 * width)
    center_y = (float(bounds[1]) + float(bounds[3])) / (2 * height)
    return action_kind, "geometry", class_value, round(center_x * 4), round(center_y * 6)


def score_action(
    source_edge: dict[str, Any],
    source_page: dict[str, Any],
    target_edge: dict[str, Any],
    target_page: dict[str, Any],
) -> dict[str, Any] | None:
    source_action, target_action = source_edge.get("action") or {}, target_edge.get("action") or {}
    if action_type(source_action) not in SUPPORTED_ACTION_TYPES or action_type(source_action) != action_type(target_action):
        return None
    if not valid_bounds(source_action) or not valid_bounds(target_action):
        return None
    class_score = _class_similarity(source_action, target_action)
    if class_score == 0.0:
        return None
    source_resource, target_resource = resource_id(source_action), resource_id(target_action)
    source_text, target_text = visible_text(source_action), visible_text(target_action)
    resource_score = None
    if source_resource and target_resource:
        resource_score = 1.0 if source_resource == target_resource else _jaccard(source_resource, target_resource)
    text_score = _string_similarity(source_text, target_text)
    geometry_score = _geometry_similarity(source_action, source_page, target_action, target_page)
    effect_score = _effect_similarity(source_edge, target_edge)
    weighted = (
        (0.32, resource_score),
        (0.25, text_score),
        (0.20, class_score),
        (0.13, geometry_score),
        (0.10, effect_score),
    )
    available = [(weight, value) for weight, value in weighted if value is not None]
    if not available:
        return None
    score = sum(weight * value for weight, value in available) / sum(weight for weight, _ in available)
    return {
        "score": round(score, 6),
        "features": {
            "resource": None if resource_score is None else round(resource_score, 6),
            "text": None if text_score is None else round(text_score, 6),
            "class": round(class_score, 6),
            "geometry": None if geometry_score is None else round(geometry_score, 6),
            "effect": None if effect_score is None else round(effect_score, 6),
        },
    }


def _resolve_state_evidence(source: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(source.get("path") or "")).expanduser().resolve().parent
    state_id = str(state.get("state_id") or "")
    xml = root / str(state.get("xml") or "")
    screenshot = root / str(state.get("screenshot") or "")
    state_json = root / "states" / f"{state_id}.json"
    return {
        "xml": str(xml),
        "screenshot": str(screenshot),
        "json": str(state_json),
        "complete": xml.is_file() and screenshot.is_file() and state_json.is_file(),
    }


def _resolve_transition_evidence(source: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(source.get("path") or "")).expanduser().resolve().parent
    values = edge.get("evidence") or {}
    resolved = {
        key: str(root / str(values.get(key) or "")) if values.get(key) else None
        for key in EVIDENCE_KEYS
    }
    resolved["complete"] = all(value is not None and Path(value).is_file() for value in resolved.values())
    return resolved


def _page_reference(page: dict[str, Any]) -> dict[str, Any]:
    return {
        key: page.get(key)
        for key in (
            "page_key", "page_id", "screenshot_path", "width", "height", "node_count",
            "package", "activity", "device_serial", "device_role",
        )
    }


def transition_record(
    role: str,
    edge: dict[str, Any],
    source_cluster_id: str,
    target_cluster_id: str,
    source: dict[str, Any],
    states: dict[str, dict[str, Any]],
    pages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_state, target_state = str(edge.get("source_state") or ""), str(edge.get("target_state") or "")
    source_key, target_key = f"{role}:{source_state}", f"{role}:{target_state}"
    return {
        "role": role,
        "event_index": edge.get("event_index"),
        "source_page_key": source_key,
        "target_page_key": target_key,
        "source_cluster_id": source_cluster_id,
        "target_cluster_id": target_cluster_id,
        "source_page": _page_reference(pages[source_key]),
        "target_page": _page_reference(pages[target_key]),
        "source_state_evidence": _resolve_state_evidence(source, states[source_state]),
        "target_state_evidence": _resolve_state_evidence(source, states[target_state]),
        "action": edge.get("action") or {},
        "effect": edge.get("effect") or {},
        "transition_evidence": _resolve_transition_evidence(source, edge),
    }


def _transition_ref(transition: dict[str, Any]) -> dict[str, Any]:
    target = action_target(transition["action"])
    return {
        "role": transition["role"],
        "event_index": transition["event_index"],
        "source_page_key": transition["source_page_key"],
        "target_page_key": transition["target_page_key"],
        "source_cluster_id": transition["source_cluster_id"],
        "target_cluster_id": transition["target_cluster_id"],
        "action": {
            "type": transition["action"].get("type"),
            "resource_id": target.get("resource_id"),
            "text": target.get("text"),
            "class_name": target.get("class_name"),
            "bounds": target.get("bounds"),
        },
        "effect": {
            key: transition["effect"].get(key)
            for key in ("state_changed", "xml_changed", "activity_changed")
        },
    }


def _candidate_id(key: tuple[Any, ...]) -> str:
    value = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _prepare_edges(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Counter[str]]:
    summary = payload["summary"]
    clusters = summary.get("cluster_catalog") or []
    cluster_by_page = {
        str(member.get("page_key")): str(cluster.get("cluster_id"))
        for cluster in clusters for member in cluster.get("members") or []
    }
    pages = {
        str(member.get("page_key")): member
        for cluster in clusters for member in cluster.get("members") or []
    }
    sources = summary.get("utg_sources") or {}
    edges_by_cluster: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    rejection_reasons: Counter[str] = Counter()
    for role, source in sources.items():
        states = {str(state.get("state_id")): state for state in source.get("states") or []}
        package = next((str(state.get("package")) for state in states.values() if state.get("package")), "")
        for edge in source.get("utg_edges") or []:
            source_state, target_state = str(edge.get("source_state") or ""), str(edge.get("target_state") or "")
            source_key, target_key = f"{role}:{source_state}", f"{role}:{target_state}"
            if source_key not in cluster_by_page or target_key not in cluster_by_page:
                rejection_reasons["state_not_clustered"] += 1
                continue
            if not edge.get("in_app") or str(edge.get("target_package") or "") != package or pages[source_key].get("package") != package:
                rejection_reasons["not_app_internal"] += 1
                continue
            action = edge.get("action") or {}
            if action_type(action) not in SUPPORTED_ACTION_TYPES or not valid_bounds(action):
                rejection_reasons["not_node_mapping_action"] += 1
                continue
            if not _resolve_state_evidence(source, states[source_state])["complete"] or not _resolve_state_evidence(source, states[target_state])["complete"]:
                rejection_reasons["incomplete_state_evidence"] += 1
                continue
            edges_by_cluster[cluster_by_page[source_key]][role].append({
                "role": role,
                "edge": edge,
                "source": source,
                "states": states,
                "source_cluster_id": cluster_by_page[source_key],
                "target_cluster_id": cluster_by_page[target_key],
                "source_page": pages[source_key],
                "target_page": pages[target_key],
            })
    return edges_by_cluster, pages, sources, rejection_reasons


def build_candidates(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = payload.get("summary") or {}
    if summary.get("schema_version") != "omnitransfer.utg_cluster_review.v1":
        raise ValueError(f"unsupported_review_schema:{summary.get('schema_version')}")
    edges_by_cluster, pages, sources, rejection_reasons = _prepare_edges(payload)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_query_count = 0
    for source_cluster_id, by_role in edges_by_cluster.items():
        source_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for item in by_role.get("source", []):
            source_groups[action_signature(item["edge"]["action"], item["source_page"])].append(item)
        for signature, source_items in source_groups.items():
            source_items.sort(key=lambda item: (item["edge"].get("event_index") is None, item["edge"].get("event_index") or 0))
            representative_source = source_items[0]
            source_after_cluster = representative_source["target_cluster_id"]
            for target_role in TARGET_ROLES:
                scored: dict[tuple[Any, ...], dict[str, Any]] = {}
                for target_item in by_role.get(target_role, []):
                    result = score_action(
                        representative_source["edge"], representative_source["source_page"],
                        target_item["edge"], target_item["source_page"],
                    )
                    if result is None:
                        rejection_reasons["weak_action_compatibility"] += 1
                        continue
                    target_key = (
                        action_signature(target_item["edge"]["action"], target_item["source_page"]),
                        target_item["target_cluster_id"],
                    )
                    scored[target_key] = max(
                        scored.get(target_key, {"score": -1.0, "item": target_item, "result": result}),
                        {"score": result["score"], "item": target_item, "result": result},
                        key=lambda row: row["score"],
                    )
                ranked = sorted(scored.values(), key=lambda row: (-row["score"], row["item"]["edge"].get("event_index") is None, row["item"]["edge"].get("event_index") or 0))
                if not ranked:
                    continue
                raw_query_count += 1
                top_score = ranked[0]["score"]
                second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
                margin = top_score - second_score
                target_divergence = ranked[0]["item"]["target_cluster_id"] != source_after_cluster
                same_action_conflicts = [
                    row for row in ranked
                    if action_signature(row["item"]["edge"]["action"], row["item"]["source_page"]) == signature
                    and row["item"]["target_cluster_id"] != source_after_cluster
                ]
                same_action_different_target_cluster = bool(same_action_conflicts)
                small_margin = len(ranked) > 1 and second_score >= 0.35 and margin <= 0.12
                boundary_band = 0.42 <= top_score <= 0.82
                easy = top_score >= 0.90 and margin > 0.18 and not target_divergence
                reasons = []
                if same_action_different_target_cluster:
                    reasons.append("same_action_different_target_cluster")
                if small_margin:
                    reasons.append("small_margin")
                if target_divergence:
                    reasons.append("target_cluster_divergence")
                if boundary_band:
                    reasons.append("boundary_score")
                if easy or (top_score < 0.42 and not same_action_different_target_cluster) or not reasons:
                    if easy:
                        rejection_reasons["obvious_large_margin"] += 1
                    else:
                        rejection_reasons["not_confusing_enough"] += 1
                    continue
                candidate_key = (target_role, source_cluster_id, signature)
                candidate = grouped.setdefault(
                    candidate_key,
                    {
                        "candidate_id": _candidate_id(candidate_key),
                        "target_role": target_role,
                        "source_cluster_id": source_cluster_id,
                        "source_action_signature": signature,
                        "source_transition": transition_record(
                            "source", representative_source["edge"], source_cluster_id,
                            source_after_cluster, sources["source"], representative_source["states"], pages,
                        ),
                        "source_transition_refs": [],
                        "target_candidates": [],
                        "reasons": reasons,
                        "action_group_status": "endpoint_cluster_conflict" if same_action_different_target_cluster else "ambiguous",
                        "top_score": round(top_score, 6),
                        "second_score": round(second_score, 6),
                        "margin": round(margin, 6),
                    },
                )
                candidate["source_transition_refs"] = [
                    _transition_ref(transition_record("source", item["edge"], source_cluster_id, item["target_cluster_id"], sources["source"], item["states"], pages))
                    for item in source_items
                ]
                selected_rows = [row for row in ranked if row["score"] >= max(0.35, top_score - 0.18)][:5]
                if same_action_conflicts and same_action_conflicts[0] not in selected_rows:
                    selected_rows = [*selected_rows[:4], same_action_conflicts[0]]
                candidate["target_candidates"] = [
                    {
                        "rank": index,
                        "score": round(row["score"], 6),
                        "features": row["result"]["features"],
                        "same_action_as_source": action_signature(row["item"]["edge"]["action"], row["item"]["source_page"]) == signature,
                        "endpoint_cluster_conflict": action_signature(row["item"]["edge"]["action"], row["item"]["source_page"]) == signature and row["item"]["target_cluster_id"] != source_after_cluster,
                        "target_cluster_relation": "same_target_cluster" if row["item"]["target_cluster_id"] == source_after_cluster else "target_cluster_divergence",
                        "transition": transition_record(target_role, row["item"]["edge"], source_cluster_id, row["item"]["target_cluster_id"], sources[target_role], row["item"]["states"], pages),
                    }
                    for index, row in enumerate(selected_rows, start=1)
                ]
    candidates = list(grouped.values())
    for candidate in candidates:
        base_priority = (
            (0.45 * max(0.0, 1.0 - candidate["margin"] / 0.12) if "small_margin" in candidate["reasons"] else 0.0)
            + (0.35 if "target_cluster_divergence" in candidate["reasons"] else 0.0)
            + (0.20 * max(0.0, 1.0 - abs(candidate["top_score"] - 0.62) / 0.20) if "boundary_score" in candidate["reasons"] else 0.0)
        )
        candidate["ambiguity_priority"] = round(
            min(1.0, 0.65 + 0.35 * base_priority)
            if "same_action_different_target_cluster" in candidate["reasons"]
            else base_priority,
            6,
        )
    candidates.sort(key=lambda candidate: (-candidate["ambiguity_priority"], candidate["target_role"], candidate["source_cluster_id"], candidate["candidate_id"]))
    audit = {
        "candidate_groups": len(candidates),
        "raw_source_queries": raw_query_count,
        "target_candidates": sum(len(candidate["target_candidates"]) for candidate in candidates),
        "reasons": dict(Counter(reason for candidate in candidates for reason in candidate["reasons"])),
        "target_roles": dict(Counter(candidate["target_role"] for candidate in candidates)),
        "rejection_reasons": dict(rejection_reasons),
    }
    return candidates, audit
