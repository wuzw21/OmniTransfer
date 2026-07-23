"""Human-review queue construction for GUIOdyssey correspondence pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import Any, Iterable

from omnitransfer.gui_odyssey import graph_from_gui_odyssey_step


SCHEMA_VERSION = "omnitransfer_guiodyssey_node_review_v1"
SEQUENCE_ALIGNMENT_SCHEMA_VERSION = "omnitransfer_guiodyssey_sequence_node_alignment_v1"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CORE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "app",
        "application",
        "button",
        "choose",
        "click",
        "go",
        "icon",
        "in",
        "into",
        "navigate",
        "of",
        "on",
        "open",
        "page",
        "press",
        "screen",
        "select",
        "tap",
        "the",
        "to",
    }
)
BROWSER_HINTS = frozenset(
    {"browser", "chrome", "chromium", "duckduckgo", "firefox", "opera", "web", "website"}
)
OPEN_TARGET_STOPWORDS = CORE_STOPWORDS | frozenset(
    {"again", "browser", "from", "launch", "reopen", "through", "using", "via"}
)
DESKTOP_PHRASES = (
    "android home screen",
    "app drawer",
    "app grid",
    "device home screen",
    "device's home screen",
    "home screen",
    "launcher screen",
)
def build_gui_odyssey_sequence_alignment_queue(
    episodes: Iterable[dict[str, Any]],
    *,
    limit: int = 50,
    min_step_score: float = 0.58,
    min_aligned_steps: int = 2,
    seed: int = 31,
    images_url_prefix: str = "screenshots",
    available_screenshots: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one-to-one click pairs from monotonic cross-device trajectory alignment."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= min_step_score <= 1.0:
        raise ValueError("min_step_score must be between zero and one")
    if min_aligned_steps <= 0:
        raise ValueError("min_aligned_steps must be positive")
    episode_list = [episode for episode in episodes if isinstance(episode, dict)]
    click_entries = _review_entries(
        episode_list,
        images_url_prefix=images_url_prefix,
        available_screenshots=available_screenshots,
    )
    click_by_step = {
        (entry["episode_id"], entry["step_index"]): entry
        for entry in click_entries
    }
    profiles = [
        _trajectory_profile(episode, click_by_step=click_by_step)
        for episode in episode_list
    ]
    profiles = [profile for profile in profiles if profile is not None]
    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile["meta_key"], profile["apps_key"])].append(profile)

    episode_candidates: list[dict[str, Any]] = []
    for group in grouped.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left["device_name"] == right["device_name"]:
                    continue
                alignment = _align_trajectory_steps(left["steps"], right["steps"])
                strong = [match for match in alignment if match[2] >= 0.35]
                if len(strong) < min_aligned_steps:
                    continue
                coverage = len(strong) / max(1, min(len(left["steps"]), len(right["steps"])))
                mean_score = sum(match[2] for match in strong) / len(strong)
                episode_score = 0.75 * mean_score + 0.25 * coverage
                episode_candidates.append(
                    {
                        "left": left,
                        "right": right,
                        "alignment": alignment,
                        "episode_score": episode_score,
                        "coverage": coverage,
                        "mean_step_score": mean_score,
                    }
                )
    episode_candidates.sort(
        key=lambda row: (
            -row["episode_score"],
            _seeded_key(
                f'{row["left"]["episode_id"]}:{row["right"]["episode_id"]}',
                seed,
            ),
        )
    )

    selected_episode_pairs: list[dict[str, Any]] = []
    used_episodes: set[str] = set()
    for candidate in episode_candidates:
        left_id = candidate["left"]["episode_id"]
        right_id = candidate["right"]["episode_id"]
        if left_id in used_episodes or right_id in used_episodes:
            continue
        selected_episode_pairs.append(candidate)
        used_episodes.update((left_id, right_id))

    rows: list[dict[str, Any]] = []
    rejected = Counter()
    for episode_pair_rank, candidate in enumerate(selected_episode_pairs, start=1):
        left = candidate["left"]
        right = candidate["right"]
        episode_pair_id = "guiodyssey-trajectory-" + hashlib.blake2b(
            f'{left["episode_id"]}|{right["episode_id"]}'.encode("utf-8"),
            digest_size=10,
        ).hexdigest()
        for alignment_rank, (left_position, right_position, step_score) in enumerate(
            candidate["alignment"],
            start=1,
        ):
            left_step = left["steps"][left_position]
            right_step = right["steps"][right_position]
            left_entry = left_step.get("click_entry")
            right_entry = right_step.get("click_entry")
            if left_entry is None or right_entry is None:
                rejected["not_click_pair"] += 1
                continue
            if step_score < min_step_score:
                rejected["step_score_below_threshold"] += 1
                continue
            if _active_app_conflict(left_step, right_step):
                rejected["active_app_conflict"] += 1
                continue
            exact_instruction = _normalized_text(left_entry["instruction"]) == _normalized_text(
                right_entry["instruction"]
            )
            row = _pair_record(
                left_entry,
                right_entry,
                candidate_kind="likely_correspondence",
                semantic_score=step_score,
                app_overlap=_jaccard(left_entry["app_tokens"], right_entry["app_tokens"]),
                description_overlap=_jaccard(
                    left_entry["description_tokens"],
                    right_entry["description_tokens"],
                ),
                exact_instruction=exact_instruction,
                same_meta_task=True,
                seed=seed,
            )
            source_is_left = row["source"]["episode_id"] == left["episode_id"]
            source_position, target_position = (
                (left_position, right_position)
                if source_is_left
                else (right_position, left_position)
            )
            row["schema_version"] = SEQUENCE_ALIGNMENT_SCHEMA_VERSION
            row["selection"].update(
                {
                    "candidate_kind": "sequence_aligned_correspondence",
                    "trajectory_pair_id": episode_pair_id,
                    "episode_pair_rank": episode_pair_rank,
                    "alignment_rank": alignment_rank,
                    "episode_alignment_score": round(candidate["episode_score"], 6),
                    "episode_alignment_coverage": round(candidate["coverage"], 6),
                    "mean_aligned_step_score": round(candidate["mean_step_score"], 6),
                    "step_alignment_score": round(step_score, 6),
                    "source_sequence_position": source_position,
                    "target_sequence_position": target_position,
                    "source_trajectory_length": len(left["steps"] if source_is_left else right["steps"]),
                    "target_trajectory_length": len(right["steps"] if source_is_left else left["steps"]),
                    "one_to_one": True,
                    "monotonic": True,
                }
            )
            row["selection"]["reasons"] = list(
                dict.fromkeys(
                    [
                        *row["selection"]["reasons"],
                        "same_meta_task_and_app_set",
                        "trajectory_monotonic_alignment",
                        "episode_one_to_one",
                        "step_one_to_one",
                    ]
                )
            )
            rows.append(row)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break

    rows.sort(
        key=lambda row: (
            row["selection"]["episode_pair_rank"],
            row["selection"]["alignment_rank"],
            row["pair_id"],
        )
    )
    manifest = {
        "schema_version": "omnitransfer_guiodyssey_sequence_alignment_manifest_v1",
        "review_schema_version": SEQUENCE_ALIGNMENT_SCHEMA_VERSION,
        "episodes_loaded": len(episode_list),
        "eligible_click_steps": len(click_entries),
        "trajectory_profiles": len(profiles),
        "episode_pair_candidates": len(episode_candidates),
        "selected_episode_pairs": len(selected_episode_pairs),
        "selected_click_pairs": len(rows),
        "min_step_score": min_step_score,
        "min_aligned_steps": min_aligned_steps,
        "rejected": dict(sorted(rejected.items())),
        "policy": {
            "episode_pairing": "exact_meta_task_and_app_set_cross_device",
            "sequence_alignment": "monotonic_one_to_one_dynamic_programming",
            "step_pairing": "click_node_to_click_node_with_current_and_successor_context",
            "human_annotation": "multi_node_graph_edges",
            "gold_label": "human_only",
            "rule_labels_for_training": False,
            "coordinate_fallback": False,
        },
    }
    return rows, manifest


def build_gui_odyssey_review_queue(
    episodes: Iterable[dict[str, Any]],
    *,
    limit: int = 200,
    likely_fraction: float = 0.8,
    seed: int = 31,
    images_url_prefix: str = "screenshots",
    available_screenshots: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank diverse pairs for human review without assigning rule labels."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= likely_fraction <= 1.0:
        raise ValueError("likely_fraction must be between zero and one")
    entries = _review_entries(
        episodes,
        images_url_prefix=images_url_prefix,
        available_screenshots=available_screenshots,
    )
    likely, hard_null = _rank_pair_candidates(entries, seed=seed)
    likely_limit = round(limit * likely_fraction)
    selected_likely = _diverse_take(likely, limit=likely_limit)
    selected_hard = _diverse_take(hard_null, limit=limit - len(selected_likely))
    if len(selected_likely) + len(selected_hard) < limit:
        used = {row["pair_id"] for row in (*selected_likely, *selected_hard)}
        remainder = [row for row in (*likely, *hard_null) if row["pair_id"] not in used]
        selected_hard.extend(_diverse_take(remainder, limit=limit - len(used)))
    rows = sorted(
        (*selected_likely, *selected_hard),
        key=lambda row: (-row["selection"]["priority_score"], row["pair_id"]),
    )[:limit]
    for rank, row in enumerate(rows, start=1):
        row["selection"]["rank"] = rank
    manifest = {
        "schema_version": "omnitransfer_guiodyssey_review_manifest_v1",
        "review_schema_version": SCHEMA_VERSION,
        "eligible_steps": len(entries),
        "likely_candidates": len(likely),
        "hard_null_candidates": len(hard_null),
        "selected_pairs": len(rows),
        "selected_kinds": dict(sorted(Counter(row["selection"]["candidate_kind"] for row in rows).items())),
        "selected_devices": dict(
            sorted(
                Counter(
                    f'{row["source"]["device_name"]}->{row["target"]["device_name"]}'
                    for row in rows
                ).items()
            )
        ),
        "seed": seed,
        "likely_fraction": likely_fraction,
        "image_availability_filter": "pinned_individual_mirror" if available_screenshots is not None else "none",
        "policy": {
            "rules": "candidate_selection_only",
            "gold_label": "human_only",
            "rule_labels_for_training": False,
            "coordinate_fallback": False,
        },
    }
    return rows, manifest


def _review_entries(
    episodes: Iterable[dict[str, Any]],
    *,
    images_url_prefix: str,
    available_screenshots: set[str] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    prefix = images_url_prefix.rstrip("/")
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        device = episode.get("device_info") if isinstance(episode.get("device_info"), dict) else {}
        task = episode.get("task_info") if isinstance(episode.get("task_info"), dict) else {}
        episode_id = str(episode.get("episode_id") or "")
        apps = tuple(str(value) for value in task.get("app") or ())
        steps = [step for step in episode.get("steps") or () if isinstance(step, dict)]
        for step_position, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            try:
                graph = graph_from_gui_odyssey_step(episode, step)
            except ValueError:
                continue
            target = next(node for node in graph.nodes if node.node_id == "action_target")
            if target.bbox is None:
                continue
            instruction = str(step.get("low_level_instruction") or "").strip()
            description = str(step.get("description") or "")
            if _is_desktop_screen(description):
                continue
            tokens = _tokens(instruction)
            core_tokens = tokens - CORE_STOPWORDS
            screenshot = str(step.get("screenshot") or "")
            if (
                not screenshot
                or not core_tokens
                or (available_screenshots is not None and screenshot not in available_screenshots)
            ):
                continue
            successor = steps[step_position + 1] if step_position + 1 < len(steps) else None
            transition_consistency = _open_transition_consistency(
                instruction,
                successor=successor,
            )
            if transition_consistency is False:
                continue
            searchable = " ".join(
                (
                    instruction,
                    str(task.get("meta_task") or ""),
                    *apps,
                )
            ).lower()
            entries.append(
                {
                    "entry_id": f"{episode_id}:{int(step.get('step') or 0)}",
                    "episode_id": episode_id,
                    "step_index": int(step.get("step") or 0),
                    "screenshot": screenshot,
                    "image_url": f"{prefix}/{screenshot}" if prefix else screenshot,
                    "width": graph.width,
                    "height": graph.height,
                    "bbox": list(target.bbox),
                    "point": list(graph.metadata["action_point_label"]),
                    "instruction": instruction,
                    "description": description,
                    "task_instruction": str(task.get("instruction") or ""),
                    "meta_task": str(task.get("meta_task") or ""),
                    "category": str(task.get("category") or ""),
                    "apps": list(apps),
                    "device_name": str(device.get("device_name") or "unknown"),
                    "device_product": str(device.get("product") or ""),
                    "form_factor": _form_factor(str(device.get("device_name") or "")),
                    "tokens": tokens,
                    "core_tokens": core_tokens,
                    "description_tokens": _tokens(str(step.get("description") or "")),
                    "app_tokens": _tokens(" ".join(apps)),
                    "meta_tokens": _tokens(str(task.get("meta_task") or "")),
                    "semantic_key": " ".join(sorted(core_tokens)),
                    "screen_context": "in_app",
                    "open_transition_verified": transition_consistency,
                    "web_or_webview_context": bool(_tokens(searchable) & BROWSER_HINTS),
                }
            )
    return entries


def _trajectory_profile(
    episode: dict[str, Any],
    *,
    click_by_step: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any] | None:
    device = episode.get("device_info") if isinstance(episode.get("device_info"), dict) else {}
    task = episode.get("task_info") if isinstance(episode.get("task_info"), dict) else {}
    episode_id = str(episode.get("episode_id") or "")
    meta_key = _normalized_text(str(task.get("meta_task") or ""))
    apps = tuple(str(value) for value in task.get("app") or () if str(value).strip())
    apps_key = tuple(sorted(_normalized_text(app) for app in apps))
    device_name = str(device.get("device_name") or "unknown")
    raw_steps = [step for step in episode.get("steps") or () if isinstance(step, dict)]
    if not episode_id or not meta_key or not apps_key or not raw_steps:
        return None
    steps = []
    for position, step in enumerate(raw_steps):
        instruction = str(step.get("low_level_instruction") or "")
        description = str(step.get("description") or "")
        successor = raw_steps[position + 1] if position + 1 < len(raw_steps) else None
        successor_text = "" if successor is None else " ".join(
            (
                str(successor.get("description") or ""),
                str(successor.get("low_level_instruction") or ""),
            )
        )
        step_index = int(step.get("step") or position)
        click_entry = click_by_step.get((episode_id, step_index))
        if click_entry is not None:
            click_entry = {
                **click_entry,
                "previous_instruction": str(
                    raw_steps[position - 1].get("low_level_instruction") or ""
                )
                if position > 0
                else "",
                "next_instruction": str(successor.get("low_level_instruction") or "")
                if successor is not None
                else "",
            }
        steps.append(
            {
                "position": position,
                "step_index": step_index,
                "action": str(step.get("action") or "").upper(),
                "instruction": instruction,
                "tokens": _tokens(instruction),
                "core_tokens": _tokens(instruction) - CORE_STOPWORDS,
                "description_tokens": _tokens(description),
                "successor_tokens": _tokens(successor_text),
                "active_apps": _mentioned_apps(apps, f"{instruction} {description}"),
                "click_entry": click_entry,
            }
        )
    return {
        "episode_id": episode_id,
        "device_name": device_name,
        "form_factor": _form_factor(device_name),
        "meta_key": meta_key,
        "apps_key": apps_key,
        "steps": steps,
    }


def _mentioned_apps(apps: tuple[str, ...], text: str) -> frozenset[str]:
    text_tokens = _tokens(text)
    mentioned = set()
    for app in apps:
        normalized = _normalized_text(app)
        app_tokens = _tokens(normalized) - CORE_STOPWORDS
        if normalized and normalized in _normalized_text(text):
            mentioned.add(normalized)
        elif app_tokens and app_tokens <= text_tokens:
            mentioned.add(normalized)
    return frozenset(mentioned)


def _align_trajectory_steps(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    gap_penalty = -0.16
    rows = len(left) + 1
    columns = len(right) + 1
    scores = [[0.0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for left_index in range(1, rows):
        scores[left_index][0] = scores[left_index - 1][0] + gap_penalty
        moves[left_index][0] = "up"
    for right_index in range(1, columns):
        scores[0][right_index] = scores[0][right_index - 1] + gap_penalty
        moves[0][right_index] = "left"
    for left_index in range(1, rows):
        for right_index in range(1, columns):
            pair_score = _trajectory_step_score(
                left[left_index - 1],
                right[right_index - 1],
            )
            choices = (
                (scores[left_index - 1][right_index - 1] + pair_score, "diag"),
                (scores[left_index - 1][right_index] + gap_penalty, "up"),
                (scores[left_index][right_index - 1] + gap_penalty, "left"),
            )
            scores[left_index][right_index], moves[left_index][right_index] = max(
                choices,
                key=lambda item: (item[0], item[1] == "diag"),
            )
    alignment = []
    left_index = len(left)
    right_index = len(right)
    while left_index > 0 or right_index > 0:
        move = moves[left_index][right_index]
        if move == "diag":
            score = _trajectory_step_score(left[left_index - 1], right[right_index - 1])
            alignment.append((left_index - 1, right_index - 1, score))
            left_index -= 1
            right_index -= 1
        elif move == "up":
            left_index -= 1
        else:
            right_index -= 1
    alignment.reverse()
    return alignment


def _trajectory_step_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    if not left["action"] or left["action"] != right["action"]:
        return -0.65
    core = _jaccard(left["core_tokens"], right["core_tokens"])
    full = _jaccard(left["tokens"], right["tokens"])
    description = _jaccard(left["description_tokens"], right["description_tokens"])
    successor = _jaccard(left["successor_tokens"], right["successor_tokens"])
    exact = float(_normalized_text(left["instruction"]) == _normalized_text(right["instruction"]))
    active_app_score = _active_app_score(left, right)
    score = (
        0.38 * core
        + 0.12 * full
        + 0.20 * description
        + 0.14 * successor
        + 0.08 * exact
        + 0.08 * active_app_score
        + 0.08
    )
    if _active_app_conflict(left, right):
        score -= 0.42
    return max(-1.0, min(1.0, score))


def _active_app_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_apps = left["active_apps"]
    right_apps = right["active_apps"]
    if not left_apps and not right_apps:
        return 0.0
    return _jaccard(left_apps, right_apps)


def _active_app_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(left["active_apps"] and right["active_apps"] and not left["active_apps"] & right["active_apps"])


def _rank_pair_candidates(
    entries: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_index: dict[str, list[int]] = defaultdict(list)
    app_index: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        for token in entry["core_tokens"]:
            token_index[token].append(index)
        for token in entry["app_tokens"]:
            app_index[token].append(index)
    pair_indices: set[tuple[int, int]] = set()
    for buckets in (token_index, app_index):
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            stable = sorted(bucket, key=lambda index: _seeded_key(entries[index]["entry_id"], seed))[:256]
            for offset, left in enumerate(stable):
                for right in stable[offset + 1 : offset + 13]:
                    if entries[left]["episode_id"] != entries[right]["episode_id"]:
                        pair_indices.add((min(left, right), max(left, right)))
    likely: list[dict[str, Any]] = []
    hard_null: list[dict[str, Any]] = []
    for left_index, right_index in pair_indices:
        left = entries[left_index]
        right = entries[right_index]
        semantic = _semantic_score(left, right)
        app_overlap = _jaccard(left["app_tokens"], right["app_tokens"])
        description_overlap = _jaccard(left["description_tokens"], right["description_tokens"])
        same_meta_task = bool(left["meta_tokens"] and left["meta_tokens"] == right["meta_tokens"])
        exact_instruction = _normalized_text(left["instruction"]) == _normalized_text(right["instruction"])
        if semantic >= 0.55 or (same_meta_task and semantic >= 0.43):
            likely.append(
                _pair_record(
                    left,
                    right,
                    candidate_kind="likely_correspondence",
                    semantic_score=semantic,
                    app_overlap=app_overlap,
                    description_overlap=description_overlap,
                    exact_instruction=exact_instruction,
                    same_meta_task=same_meta_task,
                    seed=seed,
                )
            )
        elif app_overlap > 0.0 and semantic <= 0.42 and description_overlap >= 0.08:
            hard_null.append(
                _pair_record(
                    left,
                    right,
                    candidate_kind="hard_null_candidate",
                    semantic_score=semantic,
                    app_overlap=app_overlap,
                    description_overlap=description_overlap,
                    exact_instruction=exact_instruction,
                    same_meta_task=same_meta_task,
                    seed=seed,
                )
            )
    def priority_key(row: dict[str, Any]) -> tuple[float, str]:
        return -row["selection"]["priority_score"], row["pair_id"]

    return sorted(likely, key=priority_key), sorted(hard_null, key=priority_key)


def _pair_record(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    candidate_kind: str,
    semantic_score: float,
    app_overlap: float,
    description_overlap: float,
    exact_instruction: bool,
    same_meta_task: bool,
    seed: int,
) -> dict[str, Any]:
    source, target = _orient_pair(left, right, seed=seed)
    device_diversity = _device_diversity(source, target)
    layout_shift = _layout_shift(source, target)
    if candidate_kind == "likely_correspondence":
        priority = (
            0.60 * semantic_score
            + 0.22 * device_diversity
            + 0.18 * layout_shift
        )
    else:
        priority = (
            0.34 * app_overlap
            + 0.26 * description_overlap
            + 0.22 * device_diversity
            + 0.18 * layout_shift
        )
    if "fold" in {source["form_factor"], target["form_factor"]}:
        priority += 0.08
    if source["web_or_webview_context"] or target["web_or_webview_context"]:
        priority += 0.04
    reasons = []
    if exact_instruction:
        reasons.append("exact_atomic_instruction")
    elif semantic_score >= 0.55:
        reasons.append("atomic_instruction_paraphrase")
    if same_meta_task:
        reasons.append("same_meta_task_family")
    if source["form_factor"] != target["form_factor"]:
        reasons.append("cross_form_factor")
    if "fold" in {source["form_factor"], target["form_factor"]}:
        reasons.append("includes_foldable")
    if source["web_or_webview_context"] or target["web_or_webview_context"]:
        reasons.append("browser_or_webview_context")
    if layout_shift >= 0.45:
        reasons.append("large_layout_shift")
    if candidate_kind == "hard_null_candidate":
        reasons.append("same_app_semantic_conflict")
    reasons.append("in_app_only")
    if source["open_transition_verified"] is True and target["open_transition_verified"] is True:
        reasons.append("action_outcome_consistent")
    pair_key = "|".join(sorted((source["entry_id"], target["entry_id"])))
    pair_id = "guiodyssey-pair-" + hashlib.blake2b(pair_key.encode("utf-8"), digest_size=10).hexdigest()
    source_public = _public_entry(source)
    target_public = _public_entry(target)
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "source": source_public,
        "target": target_public,
        "selection": {
            "candidate_kind": candidate_kind,
            "priority_score": round(priority, 6),
            "semantic_score": round(semantic_score, 6),
            "device_diversity": round(device_diversity, 6),
            "layout_shift": round(layout_shift, 6),
            "reasons": reasons,
            "is_gold_label": False,
        },
        "annotation": {
            "status": "unreviewed",
            "label": None,
            "source_nodes": [
                _seed_review_node("source-1", source_public, source["instruction"])
            ],
            "target_nodes": [
                _seed_review_node("target-1", target_public, target["instruction"])
            ],
            "matches": [],
            "notes": "",
            "annotator": "",
        },
    }


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "entry_id",
        "episode_id",
        "step_index",
        "screenshot",
        "image_url",
        "width",
        "height",
        "point",
        "instruction",
        "description",
        "task_instruction",
        "meta_task",
        "category",
        "apps",
        "device_name",
        "device_product",
        "form_factor",
        "web_or_webview_context",
        "open_transition_verified",
        "screen_context",
    )
    public = {key: entry[key] for key in keys}
    public["previous_instruction"] = str(entry.get("previous_instruction") or "")
    public["next_instruction"] = str(entry.get("next_instruction") or "")
    public["point_normalized"] = [
        entry["point"][0] / entry["width"] * 1000.0,
        entry["point"][1] / entry["height"] * 1000.0,
    ]
    return public


def _seed_review_node(
    node_id: str,
    entry: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "point_normalized": list(entry["point_normalized"]),
        "label": label,
        "provenance": "guiodyssey_action_point",
    }


def _diverse_take(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected = []
    episode_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    target_form_counts: Counter[str] = Counter()
    transition_cap = max(2, math.ceil(limit * 0.30))
    target_form_cap = max(2, math.ceil(limit * 0.55))
    for row in rows:
        source = row["source"]
        target = row["target"]
        semantic_key = _normalized_text(source["instruction"])
        transition = f'{source["device_name"]}->{target["device_name"]}'
        if episode_counts[source["episode_id"]] >= 2 or episode_counts[target["episode_id"]] >= 2:
            continue
        if semantic_counts[semantic_key] >= 4:
            continue
        if transition_counts[transition] >= transition_cap:
            continue
        if target_form_counts[target["form_factor"]] >= target_form_cap:
            continue
        selected.append(row)
        episode_counts[source["episode_id"]] += 1
        episode_counts[target["episode_id"]] += 1
        semantic_counts[semantic_key] += 1
        transition_counts[transition] += 1
        target_form_counts[target["form_factor"]] += 1
        if len(selected) >= limit:
            break
    return selected


def _semantic_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    full = _jaccard(left["tokens"], right["tokens"])
    core = _jaccard(left["core_tokens"], right["core_tokens"])
    apps = _jaccard(left["app_tokens"], right["app_tokens"])
    meta = 1.0 if left["meta_tokens"] and left["meta_tokens"] == right["meta_tokens"] else 0.0
    exact = 1.0 if _normalized_text(left["instruction"]) == _normalized_text(right["instruction"]) else 0.0
    return min(1.0, 0.42 * core + 0.23 * full + 0.15 * apps + 0.12 * meta + 0.08 * exact)


def _device_diversity(left: dict[str, Any], right: dict[str, Any]) -> float:
    score = 0.0
    if left["device_name"] != right["device_name"]:
        score += 0.35
    if left["form_factor"] != right["form_factor"]:
        score += 0.35
    left_ratio = left["width"] / left["height"]
    right_ratio = right["width"] / right["height"]
    score += min(abs(math.log(left_ratio / right_ratio)) / 1.2, 1.0) * 0.2
    left_area = left["width"] * left["height"]
    right_area = right["width"] * right["height"]
    score += min(abs(math.log(left_area / right_area)) / 2.0, 1.0) * 0.1
    return min(score, 1.0)


def _layout_shift(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_point = (
        left["point"][0] / left["width"],
        left["point"][1] / left["height"],
    )
    right_point = (
        right["point"][0] / right["width"],
        right["point"][1] / right["height"],
    )
    return min(1.0, math.dist(left_point, right_point) / math.sqrt(2.0))


def _orient_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_rank = {"phone": 0, "tablet": 1, "fold": 2}
    left_rank = target_rank[left["form_factor"]]
    right_rank = target_rank[right["form_factor"]]
    if left_rank != right_rank:
        return (left, right) if left_rank < right_rank else (right, left)
    return (left, right) if _seeded_key(left["entry_id"], seed) < _seeded_key(right["entry_id"], seed) else (right, left)


def _form_factor(device_name: str) -> str:
    lowered = device_name.lower()
    if "fold" in lowered:
        return "fold"
    if "tablet" in lowered:
        return "tablet"
    return "phone"


def _open_transition_consistency(
    instruction: str,
    *,
    successor: dict[str, Any] | None,
) -> bool | None:
    normalized = _normalized_text(instruction)
    if not re.match(r"^(open|launch|reopen)\b", normalized):
        return None
    target_tokens = _tokens(normalized) - OPEN_TARGET_STOPWORDS
    if not target_tokens or successor is None:
        return None
    observed_tokens = _tokens(
        " ".join(
            (
                str(successor.get("description") or ""),
                str(successor.get("low_level_instruction") or ""),
            )
        )
    )
    required_overlap = 1 if len(target_tokens) <= 2 else 2
    return len(target_tokens & observed_tokens) >= required_overlap


def _is_desktop_screen(description: str) -> bool:
    normalized = " ".join(description.lower().split())
    if any(phrase in normalized for phrase in DESKTOP_PHRASES):
        return True
    return "displaying" in normalized and "app icons" in normalized and "wallpaper" in normalized


def _tokens(value: str) -> frozenset[str]:
    return frozenset(TOKEN_PATTERN.findall(value.lower()))


def _normalized_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _seeded_key(value: str, seed: int) -> str:
    return hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).hexdigest()
