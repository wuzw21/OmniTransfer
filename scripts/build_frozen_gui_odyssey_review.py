#!/usr/bin/env python3
"""Render human review rows from an already frozen GUIOdyssey partition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from build_gui_odyssey_review_queue import _review_html


BROWSER_HINTS = frozenset(
    {"browser", "chrome", "chromium", "duckduckgo", "firefox", "opera", "web", "website"}
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved-split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--images-url-prefix", default="screenshots")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    candidate_rows = _load_jsonl(args.candidates)
    candidate_ids = {str(row.get("pair_id") or "") for row in candidate_rows}
    raw_by_id = {
        str(row.get("pair_id") or ""): row
        for row in _load_jsonl(args.raw_pairs)
        if str(row.get("pair_id") or "") in candidate_ids
    }
    missing = sorted(candidate_ids - set(raw_by_id))
    if missing:
        raise ValueError(f"{len(missing)} frozen candidates have no raw pair: {missing[:5]}")

    ranked = sorted(
        (
            _review_row(
                raw_by_id[pair_id],
                reserved_split=args.reserved_split,
                images_url_prefix=args.images_url_prefix,
                seed=args.seed,
            )
            for pair_id in candidate_ids
        ),
        key=lambda row: (-row["selection"]["priority_score"], row["pair_id"]),
    )
    selected = _diverse_take(ranked, limit=args.limit)
    for rank, row in enumerate(selected, start=1):
        row["selection"]["rank"] = rank

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / "review_queue.jsonl"
    _write_text_atomic(
        queue_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    )
    _write_text_atomic(output / "review.html", _review_html(selected))
    manifest = {
        "schema_version": "omnitransfer.frozen_guiodyssey_review_manifest.v1",
        "reserved_split": args.reserved_split,
        "candidate_source": str(args.candidates.expanduser().resolve()),
        "raw_pair_source": str(args.raw_pairs.expanduser().resolve()),
        "candidate_count": len(candidate_ids),
        "selected_count": len(selected),
        "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        "selection_only_not_labels": True,
        "selected_slices": {
            "cross_form_factor": sum(
                bool(row["selection"]["cross_form_factor"]) for row in selected
            ),
            "includes_foldable": sum(
                bool(row["selection"]["includes_foldable"]) for row in selected
            ),
            "browser_or_webview": sum(
                bool(row["selection"]["browser_or_webview_context"])
                for row in selected
            ),
            "device_pairs": dict(
                sorted(
                    Counter(
                        f'{row["source"]["device_name"]} -> {row["target"]["device_name"]}'
                        for row in selected
                    ).items()
                )
            ),
        },
        "policy": {
            "partition": "preserve frozen split and pair ids",
            "ranking": "alignment confidence plus form-factor, layout, and atomic-instruction diversity",
            "semantic_deduplication": "at most one selected pair per normalized source atomic instruction",
            "human_annotation": "multi-node correspondence or explicit NULL",
            "rule_labels": False,
        },
    }
    _write_text_atomic(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _review_row(
    row: dict[str, Any],
    *,
    reserved_split: str,
    images_url_prefix: str,
    seed: int,
) -> dict[str, Any]:
    source = _endpoint(row, "source", images_url_prefix=images_url_prefix)
    target = _endpoint(row, "target", images_url_prefix=images_url_prefix)
    source_form = source["form_factor"]
    target_form = target["form_factor"]
    cross_form = source_form != target_form
    includes_foldable = "foldable" in {source_form, target_form}
    searchable = " ".join(
        (
            source["instruction"],
            target["instruction"],
            source["description"],
            target["description"],
            *source["apps"],
        )
    ).lower()
    browser = any(hint in searchable for hint in BROWSER_HINTS)
    alignment = row.get("alignment") if isinstance(row.get("alignment"), dict) else {}
    step_score = float(alignment.get("step_score") or 0.0)
    episode_score = float(alignment.get("episode_score") or 0.0)
    layout_shift = _layout_shift(source, target)
    priority = (
        0.48 * step_score
        + 0.22 * episode_score
        + 0.16 * layout_shift
        + 0.08 * float(cross_form)
        + 0.04 * float(includes_foldable)
        + 0.02 * float(browser)
    )
    priority += _stable_jitter(str(row.get("pair_id") or ""), seed)
    reasons = ["frozen_test_partition", "adapter_validated", "in_app_only"]
    if cross_form:
        reasons.append("cross_form_factor")
    if includes_foldable:
        reasons.append("includes_foldable")
    if browser:
        reasons.append("browser_or_webview_context")
    if layout_shift >= 0.35:
        reasons.append("large_layout_shift")
    if source["instruction"].strip().lower() == target["instruction"].strip().lower():
        reasons.append("exact_atomic_instruction")
    return {
        "schema_version": "omnitransfer_guiodyssey_sequence_node_alignment_v1",
        "pair_id": str(row.get("pair_id") or ""),
        "source": source,
        "target": target,
        "selection": {
            "candidate_kind": "frozen_sequence_correspondence_candidate",
            "review_namespace": "formal-test-deduplicated-action-points-v5",
            "reserved_split": reserved_split,
            "priority_score": round(priority, 6),
            "semantic_score": round(step_score, 6),
            "step_alignment_score": round(step_score, 6),
            "episode_alignment_score": round(episode_score, 6),
            "source_sequence_position": int(
                (row.get("source") or {}).get("sequence_position") or 0
            ),
            "target_sequence_position": int(
                (row.get("target") or {}).get("sequence_position") or 0
            ),
            "layout_shift": round(layout_shift, 6),
            "cross_form_factor": cross_form,
            "includes_foldable": includes_foldable,
            "browser_or_webview_context": browser,
            "trajectory_pair_id": str(row.get("episode_pair_id") or ""),
            "reasons": reasons,
            "is_gold_label": False,
        },
        "annotation": {
            "status": "unreviewed",
            "label": None,
            "source_nodes": [_review_node("source-1", source)],
            "target_nodes": [_review_node("target-1", target)],
            "matches": [],
            "notes": "",
            "annotator": "",
        },
    }


def _review_node(node_id: str, endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "point_normalized": list(endpoint["point_normalized"]),
        "label": str(endpoint.get("instruction") or node_id),
        "provenance": "guiodyssey_action_point",
    }


def _endpoint(
    row: dict[str, Any],
    side: str,
    *,
    images_url_prefix: str,
) -> dict[str, Any]:
    raw = row.get(side)
    if not isinstance(raw, dict):
        raise ValueError(f"{side} endpoint is absent")
    width = float(raw.get("width") or 0.0)
    height = float(raw.get("height") or 0.0)
    point = raw.get("point")
    if width <= 0.0 or height <= 0.0 or not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"{side} endpoint geometry is invalid")
    point_normalized = [float(point[0]), float(point[1])]
    if not all(0.0 <= coordinate <= 1000.0 for coordinate in point_normalized):
        raise ValueError(f"{side} endpoint point is outside normalized 0-1000 space")
    screenshot = str(raw.get("screenshot") or "")
    prefix = images_url_prefix.rstrip("/")
    device_name = str(raw.get("device_name") or "unknown")
    apps = [str(value) for value in row.get("apps") or ()]
    return {
        "entry_id": f'{raw.get("episode_id")}:{int(raw.get("step_index") or 0)}',
        "episode_id": str(raw.get("episode_id") or ""),
        "step_index": int(raw.get("step_index") or 0),
        "screenshot": screenshot,
        "image_url": f"{prefix}/{screenshot}" if prefix else screenshot,
        "width": width,
        "height": height,
        "point": [
            point_normalized[0] * width / 1000.0,
            point_normalized[1] * height / 1000.0,
        ],
        "point_coordinate_space": "screenshot_pixels",
        "instruction": str(raw.get("instruction") or ""),
        "description": str(raw.get("description") or ""),
        "task_instruction": "",
        "meta_task": str(row.get("meta_task") or ""),
        "category": "",
        "apps": apps,
        "device_name": device_name,
        "device_product": "",
        "form_factor": _form_factor(device_name),
        "web_or_webview_context": False,
        "open_transition_verified": True,
        "screen_context": "in_app",
        "previous_instruction": "",
        "next_instruction": "",
        "point_normalized": point_normalized,
        "normalized_coordinate_space": "relative_0_1000",
    }


def _layout_shift(source: dict[str, Any], target: dict[str, Any]) -> float:
    source_ratio = source["width"] / source["height"]
    target_ratio = target["width"] / target["height"]
    aspect = min(abs(math.log(source_ratio / target_ratio)) / 1.5, 1.0)
    source_point = source["point_normalized"]
    target_point = target["point_normalized"]
    point_shift = min(
        math.hypot(source_point[0] - target_point[0], source_point[1] - target_point[1])
        / 800.0,
        1.0,
    )
    return 0.55 * aspect + 0.45 * point_shift


def _diverse_take(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    episode_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    for row in rows:
        source = row["source"]
        target = row["target"]
        task = source["meta_task"]
        device_pair = f'{source["device_name"]}->{target["device_name"]}'
        instruction = " ".join(str(source["instruction"]).casefold().split())
        instruction_key = instruction or row["pair_id"]
        if episode_counts[source["episode_id"]] >= 2:
            continue
        if episode_counts[target["episode_id"]] >= 2:
            continue
        if task_counts[task] >= max(4, math.ceil(limit * 0.08)):
            continue
        if device_counts[device_pair] >= max(5, math.ceil(limit * 0.14)):
            continue
        if instruction_counts[instruction_key] >= 1:
            continue
        selected.append(row)
        episode_counts[source["episode_id"]] += 1
        episode_counts[target["episode_id"]] += 1
        task_counts[task] += 1
        device_counts[device_pair] += 1
        instruction_counts[instruction_key] += 1
        if len(selected) >= limit:
            break
    return selected


def _form_factor(device_name: str) -> str:
    normalized = device_name.lower()
    if "fold" in normalized:
        return "foldable"
    if "tablet" in normalized or "pixel c" in normalized:
        return "tablet"
    return "phone"


def _stable_jitter(value: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") / (2**32 - 1) * 1e-6


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL is empty or invalid: {resolved}")
    return rows


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
