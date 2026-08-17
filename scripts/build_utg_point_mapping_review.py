#!/usr/bin/env python3
"""Build per-App Source-point to Target-point review queues from UTG evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utg_mapping_candidates import build_candidates  # noqa: E402


REVIEW_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "vector"
    / "review_annotation_template.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        help="Named UTG review batch in LABEL=/absolute/review-current form.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def parse_batch(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"invalid_batch:{value}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return label, path


def _state(evidence: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(evidence.get("json") or "")).expanduser().resolve()
    if not evidence.get("complete") or not path.is_file():
        raise ValueError("state_evidence_incomplete")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("nodes"), list):
        raise ValueError("state_nodes_missing")
    return value


def _bbox(node: dict[str, Any]) -> list[float] | None:
    bounds = node.get("bounds") or node.get("bbox")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return None
    try:
        values = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _node_payload(node: dict[str, Any]) -> dict[str, Any] | None:
    bounds = _bbox(node)
    if bounds is None:
        return None
    node_id = str(node.get("node_key") or node.get("node_id") or "")
    if not node_id:
        return None
    return {
        "node_id": node_id,
        "origin_id": str(node.get("path") or node.get("origin_id") or ""),
        "bbox": bounds,
        "text": str(node.get("text") or ""),
        "content_desc": str(node.get("content_desc") or ""),
        "resource_id": str(node.get("resource_id") or ""),
        "class_name": str(node.get("class") or node.get("class_name") or ""),
        "clickable": bool(node.get("clickable")),
        "editable": bool(node.get("editable")),
        "scrollable": bool(node.get("scrollable")),
        "enabled": bool(node.get("enabled", True)),
    }


def _dimensions(
    transition: dict[str, Any],
    screenshot: Path,
) -> tuple[float, float]:
    page = transition.get("source_page") or {}
    width = float(page.get("width") or 0)
    height = float(page.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    with Image.open(screenshot) as image:
        return float(image.width), float(image.height)


def _page_payload(
    transition: dict[str, Any],
    state: dict[str, Any],
    *,
    selected_node: dict[str, Any] | None,
    fixed_point: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence = transition.get("source_state_evidence") or {}
    screenshot = Path(str(evidence.get("screenshot") or "")).expanduser().resolve()
    if not screenshot.is_file():
        raise ValueError("state_screenshot_missing")
    width, height = _dimensions(transition, screenshot)
    candidates = [
        payload
        for node in state.get("nodes") or []
        if (payload := _node_payload(node)) is not None
    ]
    return {
        "page_id": str(
            (transition.get("source_page") or {}).get("page_id")
            or state.get("state_id")
            or ""
        ),
        "screenshot_path": str(screenshot),
        "width": width,
        "height": height,
        "node": selected_node,
        "point": fixed_point,
        "candidates": candidates,
    }


def _source_node(
    transition: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = transition.get("action") or {}
    node_key = str(action.get("node_key") or "")
    node = next(
        (
            value
            for value in state.get("nodes") or []
            if str(value.get("node_key") or value.get("node_id") or "") == node_key
        ),
        None,
    )
    payload = _node_payload(node or {})
    if payload is None:
        raise ValueError("source_action_node_missing")
    x = action.get("x")
    y = action.get("y")
    try:
        point_x = float(x)
        point_y = float(y)
    except (TypeError, ValueError):
        bounds = payload["bbox"]
        point_x = (bounds[0] + bounds[2]) / 2.0
        point_y = (bounds[1] + bounds[3]) / 2.0
    return payload, {
        "x": point_x,
        "y": point_y,
        "coordinate_space": "page_pixels",
        "node_id": payload["node_id"],
    }


def _action_node(
    transition: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = transition.get("action") or {}
    target = action.get("target") if isinstance(action.get("target"), dict) else action
    node_key = str(target.get("node_key") or action.get("node_key") or "")
    node = next(
        (
            value
            for value in state.get("nodes") or []
            if str(value.get("node_key") or value.get("node_id") or "") == node_key
        ),
        None,
    )
    payload = _node_payload(node or {})
    if payload is None:
        raise ValueError("target_action_node_missing")
    try:
        point_x = float(target.get("x", action.get("x")))
        point_y = float(target.get("y", action.get("y")))
    except (TypeError, ValueError):
        bounds = payload["bbox"]
        point_x = (bounds[0] + bounds[2]) / 2.0
        point_y = (bounds[1] + bounds[3]) / 2.0
    return payload, {
        "x": point_x,
        "y": point_y,
        "coordinate_space": "page_pixels",
        "node_id": payload["node_id"],
    }


def _task_id(values: tuple[Any, ...]) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return "utg-point-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def build_tasks(
    batch: str,
    app: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
    raw_target_pages = 0
    for group in candidates:
        source = group.get("source_transition") or {}
        source_action = source.get("action") or {}
        for target in group.get("target_candidates") or []:
            transition = target.get("transition") or {}
            target_page_id = str((transition.get("source_page") or {}).get("page_id") or "")
            key = (
                source.get("event_index"),
                source_action.get("node_key"),
                transition.get("role"),
                target_page_id,
            )
            raw_target_pages += 1
            current = selected.get(key)
            if current is None or float(target.get("score") or 0) > float(
                current[1].get("score") or 0
            ):
                selected[key] = (group, target)

    skipped: Counter[str] = Counter()
    tasks: list[dict[str, Any]] = []
    target_roles: Counter[str] = Counter()
    for key, (group, target) in sorted(
        selected.items(),
        key=lambda item: (
            -float(item[1][0].get("ambiguity_priority") or 0),
            str(item[0]),
        ),
    ):
        source_transition = group.get("source_transition") or {}
        target_transition = target.get("transition") or {}
        try:
            source_state = _state(source_transition.get("source_state_evidence") or {})
            target_state = _state(target_transition.get("source_state_evidence") or {})
            source_node, source_point = _source_node(source_transition, source_state)
            target_node, target_point = _action_node(target_transition, target_state)
            source_page = _page_payload(
                source_transition,
                source_state,
                selected_node=source_node,
                fixed_point=source_point,
            )
            target_page = _page_payload(
                target_transition,
                target_state,
                selected_node=None,
                fixed_point=None,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped[str(exc)] += 1
            continue
        if not target_page["candidates"]:
            skipped["target_nodes_missing"] += 1
            continue
        role = str(target_transition.get("role") or group.get("target_role") or "")
        pair_id = _task_id((batch, app, *key))
        tasks.append(
            {
                "task_id": pair_id,
                "pair_id": pair_id,
                "app": app,
                "label_status": "gold_review_required",
                "difficulty_score": float(group.get("ambiguity_priority") or 0),
                "difficulty_reasons": [
                    *[str(value) for value in group.get("reasons") or []],
                    f"target_role:{role}",
                ],
                "source": source_page,
                "target": target_page,
                "gold_proposal": None,
                "gold_proposals": [],
                "matcher_prediction": {
                    "node": target_node,
                    "top1_node": target_node,
                    "point": target_point,
                    "accepted": True,
                    "reason": "utg_action_candidate_non_gold",
                    "probability": float(target.get("score") or 0.0),
                    "margin": float(group.get("margin") or 0.0),
                },
                "selector_prediction": {
                    "node": None,
                    "reason": "not_shown_in_human_review",
                    "candidate_count": len(target_page["candidates"]),
                },
                "provenance": {
                    "dataset": "AndroidWorld Legacy UTG",
                    "batch": batch,
                    "app": app,
                    "source_event_index": source_transition.get("event_index"),
                    "target_role": role,
                    "target_page_selection": "utg_high_confusion_candidate",
                    "automatic_proposal": "target_utg_action_non_gold",
                    "human_target_required": True,
                },
            }
        )
        target_roles[role] += 1
    audit = {
        "candidate_groups": len(candidates),
        "deduplicated_target_pages": raw_target_pages - len(selected),
        "review_tasks": len(tasks),
        "skipped": dict(sorted(skipped.items())),
        "target_roles": dict(sorted(target_roles.items())),
    }
    return tasks, audit


def _copy_screenshots(tasks: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    result = json.loads(json.dumps(tasks))
    for task in result:
        for side in ("source", "target"):
            source = Path(task[side]["screenshot_path"]).expanduser().resolve()
            if source not in copied:
                digest = hashlib.blake2b(str(source).encode(), digest_size=8).hexdigest()
                destination = screenshots / f"{digest}_{source.name}"
                shutil.copy2(source, destination)
                copied[source] = f"screenshots/{destination.name}"
            task[side]["screenshot_path"] = copied[source]
    return result


def _review_payload(
    batch: str,
    app: str,
    tasks: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary": {
            "schema_version": "omnitransfer.utg_point_mapping_review.v1",
            "batch": batch,
            "app": app,
            "task_count": len(tasks),
            "audit": audit,
            "review_ui": {
                "protocol": "human_action_mapping",
                "template_ids": [
                    "confirm_mapping",
                    "no_correspondence",
                    "discard_bad_evidence",
                ],
                "template_overrides": {
                    "confirm_mapping": "确认 P / H 映射",
                    "no_correspondence": "目标不存在（NULL）",
                    "discard_bad_evidence": "截图或结构不可用",
                },
                "diagnostic_overlay": {
                    "enabled": False,
                    "coordinate_space": "page_pixels",
                },
                "supports_multiple_target_points": True,
                "coordinate_space": "page_pixels",
                "fixed_source": True,
                "minimal": True,
                "show_candidate_toggle": False,
                "prediction_label": "UTG 自动候选 P（非金标）",
            },
        },
        "pairs": tasks,
    }


def write_app_review(
    batch: str,
    app: str,
    tasks: list[dict[str, Any]],
    audit: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    materialized = _copy_screenshots(tasks, destination)
    payload = _review_payload(batch, app, materialized, audit)
    (destination / "review.html.payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in materialized),
        encoding="utf-8",
    )
    template = REVIEW_TEMPLATE.read_text(encoding="utf-8")
    marker = "__OMNITRANSFER_REVIEW_PAYLOAD__"
    if template.count(marker) != 1:
        raise ValueError("canonical_review_template_payload_marker_invalid")
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (destination / "review.html").write_text(
        template.replace(marker, embedded),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omnitransfer.utg_point_mapping_review_manifest.v1",
        "batch": batch,
        "app": app,
        "tasks": len(materialized),
        "source_pages": len({task["source"]["page_id"] for task in materialized}),
        "target_pages": len({task["target"]["page_id"] for task in materialized}),
        "target_roles": audit["target_roles"],
        "review_file": "review.html",
        "tasks_file": "tasks.jsonl",
        "label_output": "omnitransfer_ui_correspondence_pairs.jsonl",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_batch_index(label: str, output: Path, reports: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f'<a class="row" href="{html.escape(report["app"])}/review.html">'
        f'<span>{html.escape(report["app"])}</span>'
        f'<strong>{report["tasks"]} 个点映射</strong></a>'
        for report in reports
    )
    (output / label / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(label)} · Source → Target 点映射</title><style>"
        "body{margin:0;background:#f5f4ef;color:#171714;font:15px system-ui,sans-serif}main{max-width:940px;margin:auto;padding:56px 24px}"
        "a{color:inherit;text-decoration:none}.row{display:flex;justify-content:space-between;padding:17px 0;border-bottom:1px solid #d8d6cc}.row:hover span{color:#166534}"
        "strong{font-size:13px;color:#67675f}h1{font-size:36px;margin:0 0 8px}p{color:#67675f;margin:0 0 30px}</style><main>"
        f"<h1>{html.escape(label)} · 点映射审核</h1><p>{len(reports)} 个 App；点击 Target 对应点，或选择 NULL。</p>{rows}</main>",
        encoding="utf-8",
    )


def build_batch(label: str, review_root: Path, output_root: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for payload_path in sorted(review_root.glob("*/review.html.payload.json")):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        candidates, candidate_audit = build_candidates(payload)
        app = payload_path.parent.name
        tasks, audit = build_tasks(label, app, candidates)
        audit["candidate_audit"] = candidate_audit
        manifest = write_app_review(
            label,
            app,
            tasks,
            audit,
            output_root / label / app,
        )
        reports.append(manifest)
    _write_batch_index(label, output_root, reports)
    return {
        "batch": label,
        "review_root": str(review_root),
        "apps": len(reports),
        "apps_with_tasks": sum(report["tasks"] > 0 for report in reports),
        "tasks": sum(report["tasks"] for report in reports),
        "source_pages": sum(report["source_pages"] for report in reports),
        "target_pages": sum(report["target_pages"] for report in reports),
        "app_reports": reports,
    }


def _write_root_index(output: Path, reports: list[dict[str, Any]]) -> None:
    if len(reports) == 1:
        report = reports[0]
        rows = "\n".join(
            f'<a href="{html.escape(report["batch"])}/{html.escape(app["app"])}/review.html">'
            f'<span>{html.escape(app["app"])}</span><strong>{app["tasks"]} 个点映射</strong></a>'
            for app in report["app_reports"]
        )
        subtitle = f'{report["apps"]} 个 App · {report["tasks"]} 个 Source → Target 点映射'
    else:
        rows = "\n".join(
            f'<a href="{html.escape(report["batch"])}/index.html">'
            f'<span>{html.escape(report["batch"])}</span>'
            f'<strong>{report["apps"]} Apps · {report["tasks"]} 点映射</strong></a>'
            for report in reports
        )
        subtitle = f"{len(reports)} 个批次"
    (output / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>UTG 点映射审核</title>"
        "<style>body{font:16px system-ui,sans-serif;max-width:900px;margin:70px auto;padding:0 24px;background:#f5f4ef;color:#171714}"
        "a{display:flex;justify-content:space-between;gap:20px;color:inherit;text-decoration:none;padding:17px 0;border-bottom:1px solid #d8d6cc}"
        "a:hover span{color:#166534}strong{font-size:13px;color:#67675f}p{color:#67675f;margin-bottom:30px}</style>"
        f"<h1>Source → Target 点映射审核</h1><p>{html.escape(subtitle)}</p>{rows}",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    if not output.is_absolute():
        raise SystemExit("--output-root must be absolute")
    output.mkdir(parents=True, exist_ok=True)
    reports = [
        build_batch(label, review_root, output)
        for label, review_root in (parse_batch(value) for value in args.batch)
    ]
    _write_root_index(output, reports)
    summary = {
        "schema_version": "omnitransfer.utg_point_mapping_review_summary.v1",
        "review_task": "source_point_to_target_point_or_null",
        "batches": reports,
        "totals": {
            "apps": sum(report["apps"] for report in reports),
            "apps_with_tasks": sum(report["apps_with_tasks"] for report in reports),
            "tasks": sum(report["tasks"] for report in reports),
            "source_pages": sum(report["source_pages"] for report in reports),
            "target_pages": sum(report["target_pages"] for report in reports),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), **summary["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
