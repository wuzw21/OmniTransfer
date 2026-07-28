#!/usr/bin/env python3
"""Build an App-balanced hard mapping acceptance review queue."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from omnitransfer.baselines.anchor_vote_64d import AnchorVote64DMatcher
from omnitransfer.hard_acceptance import (
    HardMappingCandidate,
    score_hard_mapping_candidate,
    select_balanced_hard_candidates,
)
from omnitransfer.learned_matcher import LearnedGraphMatcher, is_actionable
from omnitransfer.mapping_dataset import validate_mapping_page_pair
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-limit", type=int, default=1000)
    parser.add_argument("--max-tasks-per-app", type=int, default=80)
    parser.add_argument("--max-tasks-per-pair", type=int, default=8)
    parser.add_argument("--max-tasks-per-semantic", type=int, default=20)
    parser.add_argument("--minimum-difficulty", type=float, default=0.5)
    parser.add_argument("--matcher-min-probability", type=float, default=0.5)
    args = parser.parse_args()

    records = _load_records(args.input)
    matcher = LearnedGraphMatcher.from_checkpoint(args.checkpoint, device=args.device)
    candidates = _build_candidates(
        records,
        matcher=matcher,
        matcher_min_probability=args.matcher_min_probability,
    )
    selected = select_balanced_hard_candidates(
        candidates,
        task_limit=args.task_limit,
        max_tasks_per_app=args.max_tasks_per_app,
        max_tasks_per_pair=args.max_tasks_per_pair,
        max_tasks_per_semantic=args.max_tasks_per_semantic,
        minimum_difficulty=args.minimum_difficulty,
    )
    if len(selected) < args.task_limit:
        raise SystemExit(
            f"Only {len(selected)} tasks survived diversity caps; requested {args.task_limit}"
        )
    manifest = _write_review(
        selected,
        output_dir=args.output_dir,
        input_path=args.input,
        checkpoint=args.checkpoint,
        pool_size=len(candidates),
        limits={
            "task_limit": args.task_limit,
            "max_tasks_per_app": args.max_tasks_per_app,
            "max_tasks_per_pair": args.max_tasks_per_pair,
            "max_tasks_per_semantic": args.max_tasks_per_semantic,
            "minimum_difficulty": args.minimum_difficulty,
            "matcher_min_probability": args.matcher_min_probability,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(validate_mapping_page_pair(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid mapping pair at line {line_number}") from exc
    return records


def _build_candidates(
    records: list[dict[str, Any]],
    *,
    matcher: LearnedGraphMatcher,
    matcher_min_probability: float,
) -> list[HardMappingCandidate]:
    candidates: list[HardMappingCandidate] = []
    for record in records:
        source_graph = _page_graph(record, "source")
        target_graph = _page_graph(record, "target")
        target_by_id = {node.node_id: node for node in target_graph.nodes}
        source_by_id = {node.node_id: node for node in source_graph.nodes}
        target_ids = [
            node.node_id
            for node in target_graph.nodes
            if node.bbox is not None and is_actionable(node)
        ]
        strict = _strict_actionable_matches(
            record,
            source_by_id=source_by_id,
            target_by_id=target_by_id,
        )
        if not strict:
            continue
        selector = AnchorVote64DMatcher(source_graph, target_graph)
        for source_id, gold_id in strict:
            source = source_by_id[source_id]
            gold = target_by_id[gold_id]
            learned = matcher.predict(
                source_graph,
                target_graph,
                source_node_id=source_id,
                candidate_node_ids=target_ids,
                min_probability=matcher_min_probability,
            )
            selector_result = selector.rank_selector(
                source_id,
                candidate_node_ids=target_ids,
            )
            matcher_top1_id = learned.scores[0][0] if learned.scores else None
            matcher_id = learned.target_node.node_id if learned.target_node else None
            selector_id = (
                selector_result.candidates[0].node_id
                if selector_result.candidates
                else None
            )
            task_id = _task_id(record["pair_id"], source_id, gold_id)
            payload = _task_payload(
                record,
                task_id=task_id,
                source=source,
                gold=gold,
                matcher_id=matcher_id,
                matcher_top1_id=matcher_top1_id,
                matcher=learned,
                matcher_min_probability=matcher_min_probability,
                selector_id=selector_id,
                selector_result=selector_result,
                target_by_id=target_by_id,
            )
            candidate = HardMappingCandidate(
                task_id=task_id,
                pair_id=record["pair_id"],
                app=str(record["provenance"].get("trace") or "unknown"),
                semantic_key=_semantic_key(source),
                matcher_correct=matcher_top1_id == gold_id,
                selector_correct=selector_id == gold_id,
                selector_abstained=selector_id is None,
                methods_disagree=matcher_id != selector_id,
                matcher_margin=float(learned.margin),
                matcher_probability=float(learned.probability),
                position_shift=_position_shift(
                    source, gold, source_graph, target_graph
                ),
                target_area_fraction=_area_fraction(gold, target_graph),
                textless=not bool(source.text or source.content_desc),
                sequence_transition=record["slices"].get("track")
                == "trace_state_transition",
                payload=payload,
            )
            score, reasons = score_hard_mapping_candidate(candidate)
            payload["difficulty_score"] = score
            payload["difficulty_reasons"] = reasons
            candidates.append(candidate)
    return candidates


def _page_graph(record: dict[str, Any], side: str) -> UIGraph:
    page = record[side]
    graph = graph_from_record(page["graph"], graph_id=page["page_id"])
    return replace(
        graph,
        metadata={**graph.metadata, "screenshot_path": page["screenshot_path"]},
    )


def _strict_matches(record: dict[str, Any]) -> list[tuple[str, str]]:
    singles = [
        match
        for match in record["matches"]
        if match["label"] == "correspondence" and len(match["target_node_ids"]) == 1
    ]
    target_counts = Counter(match["target_node_ids"][0] for match in singles)
    return [
        (match["source_node_id"], match["target_node_ids"][0])
        for match in singles
        if target_counts[match["target_node_ids"][0]] == 1
    ]


def _strict_actionable_matches(
    record: dict[str, Any],
    *,
    source_by_id: dict[str, UINode],
    target_by_id: dict[str, UINode],
) -> list[tuple[str, str]]:
    """Keep one-to-one click mappings that both sides can actually execute."""

    selected: list[tuple[str, str]] = []
    for source_id, target_id in _strict_matches(record):
        source = source_by_id.get(source_id)
        target = target_by_id.get(target_id)
        if (
            source is None
            or target is None
            or source.bbox is None
            or target.bbox is None
            or not is_actionable(source)
            or not is_actionable(target)
        ):
            continue
        selected.append((source_id, target_id))
    return selected


def _task_payload(
    record: dict[str, Any],
    *,
    task_id: str,
    source: UINode,
    gold: UINode,
    matcher_id: str | None,
    matcher_top1_id: str | None,
    matcher: Any,
    matcher_min_probability: float,
    selector_id: str | None,
    selector_result: Any,
    target_by_id: dict[str, UINode],
) -> dict[str, Any]:
    matcher_node = target_by_id.get(str(matcher_id))
    matcher_top1_node = target_by_id.get(str(matcher_top1_id))
    selector_node = target_by_id.get(str(selector_id))
    return {
        "task_id": task_id,
        "pair_id": record["pair_id"],
        "app": str(record["provenance"].get("trace") or "unknown"),
        "label_status": "gold_review_required",
        "coordinate_space": "page_pixels",
        "source": _page_payload(record["source"], source),
        "target": _page_payload(record["target"], gold),
        "gold_proposal": _node_payload(gold),
        "matcher_prediction": {
            "node": _node_payload(matcher_node),
            "top1_node": _node_payload(matcher_top1_node),
            "accepted": matcher_node is not None,
            "minimum_probability": matcher_min_probability,
            "probability": float(matcher.probability),
            "margin": float(matcher.margin),
            "reason": matcher.reason,
            "top5": [
                {"node_id": node_id, "score": float(score)}
                for node_id, score in matcher.scores[:5]
            ],
        },
        "selector_prediction": {
            "node": _node_payload(selector_node),
            "reason": selector_result.reason,
            "candidate_count": len(selector_result.candidates),
            "latency_ms": float(selector_result.latency_ms),
        },
        "recall": {
            "strategy": "mobileviews_matcher_selector_gap_v1",
            "source": record["provenance"].get("annotation"),
            "label_policy": "automatic_view_str_proposal_requires_human_confirmation",
        },
        "slices": record["slices"],
        "provenance": record["provenance"],
    }


def _page_payload(page: dict[str, Any], node: UINode) -> dict[str, Any]:
    graph = page["graph"]
    return {
        "page_id": page["page_id"],
        "screenshot_path": str(Path(page["screenshot_path"]).expanduser().resolve()),
        "width": float(graph.get("width") or 0),
        "height": float(graph.get("height") or 0),
        "node": _node_payload(node),
    }


def _node_payload(node: UINode | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": node.node_id,
        "origin_id": node.origin_id,
        "view_str": str(node.metadata.get("label_view_str") or ""),
        "bbox": list(node.bbox) if node.bbox else None,
        "text": node.text,
        "content_desc": node.content_desc,
        "resource_id": node.resource_id,
        "class_name": node.class_name,
        "clickable": node.clickable,
        "editable": node.editable,
    }


def _semantic_key(node: UINode) -> str:
    return (
        "|".join(
            value.strip().lower()
            for value in (
                node.resource_id,
                node.text,
                node.content_desc,
                node.class_name,
            )
            if value.strip()
        )
        or "textless"
    )


def _position_shift(
    source: UINode,
    target: UINode,
    source_graph: UIGraph,
    target_graph: UIGraph,
) -> float:
    if not source.bbox or not target.bbox:
        return 1.0
    source_center = (
        (source.bbox[0] + source.bbox[2]) / 2,
        (source.bbox[1] + source.bbox[3]) / 2,
    )
    target_center = (
        (target.bbox[0] + target.bbox[2]) / 2,
        (target.bbox[1] + target.bbox[3]) / 2,
    )
    source_xy = (
        source_center[0] / max(1.0, float(source_graph.width or 1.0)),
        source_center[1] / max(1.0, float(source_graph.height or 1.0)),
    )
    target_xy = (
        target_center[0] / max(1.0, float(target_graph.width or 1.0)),
        target_center[1] / max(1.0, float(target_graph.height or 1.0)),
    )
    return math.dist(source_xy, target_xy)


def _area_fraction(node: UINode, graph: UIGraph) -> float:
    if not node.bbox:
        return 0.0
    area = max(0.0, node.bbox[2] - node.bbox[0]) * max(0.0, node.bbox[3] - node.bbox[1])
    return area / max(1.0, float(graph.width or 1.0) * float(graph.height or 1.0))


def _task_id(pair_id: str, source_id: str, target_id: str) -> str:
    digest = hashlib.blake2b(
        f"{pair_id}\0{source_id}\0{target_id}".encode(), digest_size=12
    ).hexdigest()
    return f"hard-mapping-{digest}"


def _write_review(
    selected: list[HardMappingCandidate],
    *,
    output_dir: Path,
    input_path: Path,
    checkpoint: Path,
    pool_size: int,
    limits: dict[str, int],
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    for candidate in selected:
        task = json.loads(json.dumps(candidate.payload))
        for side in ("source", "target"):
            original = task[side]["screenshot_path"]
            if original not in copied:
                path = Path(original)
                digest = hashlib.blake2b(str(path).encode(), digest_size=8).hexdigest()
                destination = screenshots / f"{digest}_{path.name}"
                shutil.copy2(path, destination)
                copied[original] = f"screenshots/{destination.name}"
            task[side]["screenshot_path"] = copied[original]
        tasks.append(task)
    app_counts = Counter(task["app"] for task in tasks)
    reason_counts = Counter(
        reason for task in tasks for reason in task["difficulty_reasons"]
    )
    payload = {
        "summary": {
            "schema_version": "omnitransfer.hard_mapping_acceptance.v1",
            "label_status": "gold_review_required",
            "task_count": len(tasks),
            "candidate_pool": pool_size,
            "apps": len(app_counts),
            "app_counts": dict(sorted(app_counts.items())),
            "difficulty_reason_counts": dict(sorted(reason_counts.items())),
            "limits": limits,
            "input": str(input_path.expanduser().resolve()),
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "required_retrain_protocol": "exclude every selected app before formal scoring",
            "candidate_boundary": (
                "strict_one_to_one_actionable_source_to_actionable_target_with_bbox"
            ),
            "review_ui": {
                "protocol": "selector_signal_gap",
                "template_ids": [
                    "correct_correspondence",
                    "wrong_correspondence",
                    "ambiguous_or_absent",
                    "discard_bad_evidence",
                ],
                "template_overrides": {},
                "diagnostic_overlay": {
                    "enabled": True,
                    "methods": [
                        "gold_proposal",
                        "matcher_prediction",
                        "selector_prediction",
                    ],
                    "coordinate_space": "page_pixels",
                },
            },
        },
        "pairs": tasks,
    }
    sidecar = output / "review.html.payload.json"
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks)
    )
    template_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "vector"
        / "review_annotation_template.html"
    )
    template = template_path.read_text(encoding="utf-8")
    light_payload = {"summary": payload["summary"], "pairs": tasks}
    review_html = template.replace(
        "__OMNITRANSFER_REVIEW_PAYLOAD__",
        json.dumps(light_payload, ensure_ascii=False).replace("</", "<\\/"),
    )
    (output / "review.html").write_text(review_html, encoding="utf-8")
    manifest = {
        "schema_version": "omnitransfer.hard_mapping_acceptance_manifest.v1",
        "tasks": len(tasks),
        "candidate_pool": pool_size,
        "apps": len(app_counts),
        "screenshots": len(copied),
        "app_counts": dict(sorted(app_counts.items())),
        "difficulty_reason_counts": dict(sorted(reason_counts.items())),
        "review_file": "review.html",
        "sidecar": sidecar.name,
        "label_status": "gold_review_required",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    main()
