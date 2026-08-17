"""Static reviewer for unified mapping page-pair records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from PIL import Image

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair


def build_mapping_pair_review(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    pair_limit: int = 0,
    complex_only: bool = False,
    complex_limit: int = 0,
    external_payload: bool = False,
    method_tag: str | None = None,
) -> dict[str, Any]:
    """Materialize screenshots and a standalone review page for pair JSONL files."""

    if pair_limit < 0:
        raise ValueError("pair_limit must be non-negative")
    input_paths = [Path(path).expanduser().resolve() for path in inputs]
    records: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(validate_ui_correspondence_pair(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid pair at {path}:{line_number}") from exc
    if pair_limit:
        records = records[:pair_limit]
    if not records:
        raise ValueError("review input contains no pairs")

    output = Path(output_dir).expanduser().resolve()
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    queue = [
        _review_item(record, screenshots=screenshots, copied=copied)
        for record in records
    ]
    queue.sort(
        key=lambda item: (
            not bool(item.get("method_tags")),
            -float(item["difficulty_score"]),
            item["pair_id"],
        )
    )
    if method_tag:
        queue = [
            item
            for item in queue
            if method_tag in {
                tag if isinstance(tag, str) else tag.get("id")
                for tag in item.get("method_tags", [])
            }
        ]
    if complex_only:
        queue = [item for item in queue if item["difficulty_score"] > 0]
    if complex_limit:
        if complex_limit < 0:
            raise ValueError("complex_limit must be non-negative")
        queue = queue[:complex_limit]
    if not queue:
        raise ValueError(
            "review input contains no complex pairs" if complex_only else "review input contains no pairs"
        )
    payload = _review_payload(queue)
    review_path = output / "review.html"
    review_path.write_text(
        _review_html(payload, external_payload=external_payload), encoding="utf-8"
    )
    sidecar = output / "review.html.payload.json"
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omnitransfer.mapping_pair_review_manifest.v1",
        "pairs": len(queue),
        "matches": sum(len(item["matches"]) for item in queue),
        "datasets": dict(sorted(_counts(item["dataset"] for item in queue).items())),
        "labels": dict(
            sorted(
                _counts(
                    match["label"]
                    for item in queue
                    for match in item["matches"]
                ).items()
            )
        ),
        "screenshots": len(copied),
        "inputs": [str(path) for path in input_paths],
        "review_file": "review.html",
        "sidecar": sidecar.name,
        "complex_only": complex_only,
        "complex_pairs": sum(item["difficulty_score"] > 0 for item in queue),
        "external_payload": external_payload,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _complexity(record: dict[str, Any]) -> tuple[float, list[str]]:
    """Rank review effort from observable ambiguity, without changing matching."""

    matches = record["matches"]
    source_nodes = {
        str(node["node_id"]): node for node in record["source"]["graph"].get("nodes", [])
    }
    target_nodes = {
        str(node["node_id"]): node for node in record["target"]["graph"].get("nodes", [])
    }
    score = 0.0
    reasons: list[str] = []
    if len(matches) > 1:
        score += min(0.3, 0.12 * (len(matches) - 1))
        reasons.append(f"multi_mapping:{len(matches)}")
    alternatives = sum(max(0, len(match["target_node_ids"]) - 1) for match in matches)
    if alternatives:
        score += min(0.3, 0.1 * alternatives)
        reasons.append(f"target_alternatives:{alternatives}")
    source_semantics = []
    icon_only = 0
    for match in matches:
        node = source_nodes.get(str(match["source_node_id"]), {})
        semantic = " ".join(
            str(node.get(field) or "").strip()
            for field in ("text", "content_desc", "resource_id")
        ).strip().lower()
        source_semantics.append(semantic)
        if not semantic:
            icon_only += 1
    if icon_only:
        score += min(0.25, 0.12 * icon_only)
        reasons.append(f"icon_or_no_text:{icon_only}")
    repeated = len(source_semantics) - len(set(source_semantics))
    if repeated > 0:
        score += min(0.2, 0.1 * repeated)
        reasons.append(f"repeated_source_semantics:{repeated}")
    displacements = []
    for match in matches:
        source = source_nodes.get(str(match["source_node_id"]), {})
        target = target_nodes.get(str(match["target_node_ids"][0]), {}) if match["target_node_ids"] else {}
        source_box, target_box = source.get("bbox"), target.get("bbox")
        if source_box and target_box:
            source_center = ((source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2)
            target_center = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
            displacements.append(abs(source_center[0] - target_center[0]) + abs(source_center[1] - target_center[1]))
    if displacements and max(displacements) > 0:
        score += 0.05
        reasons.append("layout_position_changes")
    external = record.get("provenance", {}).get("difficulty_score")
    if external is not None:
        try:
            score += min(0.2, max(0.0, float(external)))
            reasons.append("upstream_uncertainty")
        except (TypeError, ValueError):
            pass
    return min(1.0, score), reasons or ["low_ambiguity"]


def _review_item(
    record: dict[str, Any],
    *,
    screenshots: Path,
    copied: dict[Path, str],
) -> dict[str, Any]:
    source_nodes = {
        str(node["node_id"]): node for node in record["source"]["graph"]["nodes"]
    }
    target_nodes = {
        str(node["node_id"]): node for node in record["target"]["graph"]["nodes"]
    }
    source_ids = {str(match["source_node_id"]) for match in record["matches"]}
    target_ids = {
        str(node_id)
        for match in record["matches"]
        for node_id in match["target_node_ids"]
    }
    dataset = str(record["provenance"].get("dataset") or "unknown")
    difficulty_score, difficulty_reasons = _complexity(record)
    return {
        "pair_id": record["pair_id"],
        "dataset": dataset,
        "label_status": record["label_status"],
        "slices": record["slices"],
        "provenance": record["provenance"],
        "difficulty_score": difficulty_score,
        "difficulty_reasons": difficulty_reasons,
        "method_tags": record.get("method_tags", []),
        "method_comparison": record.get("method_comparison"),
        "source": _review_side(
            record["source"],
            nodes=[source_nodes[node_id] for node_id in sorted(source_ids)],
            screenshots=screenshots,
            copied=copied,
        ),
        "target": _review_side(
            record["target"],
            nodes=[target_nodes[node_id] for node_id in sorted(target_ids)],
            screenshots=screenshots,
            copied=copied,
        ),
        "matches": record["matches"],
    }


def _review_side(
    page: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    screenshots: Path,
    copied: dict[Path, str],
) -> dict[str, Any]:
    source = Path(page["screenshot_path"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source not in copied:
        digest = hashlib.blake2b(str(source).encode(), digest_size=8).hexdigest()
        destination = screenshots / f"{digest}_{source.name}"
        shutil.copy2(source, destination)
        copied[source] = f"screenshots/{destination.name}"
    graph = page["graph"]
    width = float(page.get("display_width") or graph.get("width") or 0)
    height = float(page.get("display_height") or graph.get("height") or 0)
    if width <= 0 or height <= 0:
        with Image.open(source) as image:
            width, height = image.size
    review_nodes = [_review_node(node) for node in nodes]
    all_nodes = [_review_node(node) for node in graph.get("nodes", [])]
    return {
        "page_id": page["page_id"],
        "platform": page["platform"],
        "image_url": copied[source],
        "width": width,
        "height": height,
        "nodes": review_nodes,
        "candidates": all_nodes,
    }


def _review_node(node: dict[str, Any]) -> dict[str, Any]:
    bbox = node.get("bbox")
    return {
        "node_id": str(node["node_id"]),
        "origin_id": str(node.get("origin_id") or ""),
        "text": str(node.get("text") or ""),
        "content_desc": str(node.get("content_desc") or ""),
        "resource_id": str(node.get("resource_id") or ""),
        "class_name": str(node.get("class_name") or ""),
        "bbox": [float(value) for value in bbox] if bbox else None,
        "visual_bbox": [float(value) for value in node.get("visual_bbox", [])] if node.get("visual_bbox") else None,
        "clickable": bool(node.get("clickable")),
        "editable": bool(node.get("editable")),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _review_payload(queue: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = []
    for item in queue:
        source_nodes = {node["node_id"]: node for node in item["source"]["nodes"]}
        target_nodes = {node["node_id"]: node for node in item["target"]["nodes"]}
        mappings = []
        for match_index, match in enumerate(item["matches"]):
            source_node = source_nodes[match["source_node_id"]]
            target_nodes_for_match = [
                target_nodes[node_id] for node_id in match["target_node_ids"]
            ]
            source_label = f"S{match_index + 1}"
            target_label = f"M{match_index + 1}"
            mappings.append(
                {
                    "mapping_id": f"{source_label}-{target_label}",
                    "source_label": source_label,
                    "target_label": target_label,
                    "label": f"{source_label} → {target_label}",
                    "source_node": source_node,
                    "target_nodes": target_nodes_for_match,
                    "target_node_ids": list(match["target_node_ids"]),
                    "label_hint": match["label"],
                    "matcher_prediction": {
                        "node": None,
                        "top1_node": None,
                        "accepted": False,
                        "reason": "not_evaluated",
                        "probability": 0.0,
                        "margin": 0.0,
                    },
                }
            )
        tasks.append(
            {
                "task_id": item["pair_id"],
                "pair_id": item["pair_id"],
                "app": item["dataset"],
                "label_status": item["label_status"],
                "difficulty_score": item["difficulty_score"],
                "difficulty_reasons": item["difficulty_reasons"],
                "method_tags": item.get("method_tags", []),
                "method_comparison": item.get("method_comparison"),
                "source": _workbench_page(item["source"], None),
                "target": _workbench_page(item["target"], None),
                "mappings": mappings,
                "provenance": item["provenance"],
            }
        )
    return {
        "summary": {
            "schema_version": "omnitransfer.mapping_pair_review.v2",
            "task_count": len(tasks),
            "review_ui": {
                "protocol": "mapping_pair_correspondence",
                "multi_mapping": True,
                "template_ids": [
                    "correct_correspondence",
                    "wrong_correspondence",
                    "ambiguous_or_absent",
                    "discard_bad_evidence",
                ],
                "template_overrides": {},
                "diagnostic_overlay": {
                    "enabled": True,
                    "methods": ["gold_proposal"],
                    "coordinate_space": "page_pixels",
                },
            },
        },
        "pairs": tasks,
    }


def _workbench_page(
    page: dict[str, Any],
    node: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "page_id": page["page_id"],
        "width": page["width"],
        "height": page["height"],
        "screenshot_path": page["image_url"],
        "node": node,
        "nodes": page.get("nodes", []),
        "candidates": page.get("candidates", page.get("nodes", [])),
    }


def _review_html(payload: dict[str, Any], *, external_payload: bool = False) -> str:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "vector"
        / "review_annotation_template.html"
    )
    if not template_path.is_file():
        raise FileNotFoundError(f"canonical review template missing: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    marker = "__OMNITRANSFER_REVIEW_PAYLOAD__"
    if template.count(marker) != 1:
        raise ValueError("canonical review template payload marker invalid")
    embedded = (
        "await fetch('./review.html.payload.json').then(response => response.json())"
        if external_payload
        else json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    return template.replace(marker, embedded)
