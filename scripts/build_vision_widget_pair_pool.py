#!/usr/bin/env python3
"""Convert the clean vision-widget mappings into canonical page-pair records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from omnitransfer.ui_graph import graph_from_record, graph_to_record


def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    node_metadata: dict[str, Any] = {}
    if metadata.get("visual_bbox") is not None:
        node_metadata["visual_bbox_coordinate_space"] = "page_pixels"
    return {
        "node_id": node["node_id"],
        "origin_id": node.get("origin_id") or node["node_id"],
        "parent_id": node.get("parent_id"),
        "text": node.get("text", ""),
        "content_desc": node.get("content_desc", ""),
        "resource_id": node.get("resource_id", ""),
        "class_name": node.get("class_name", ""),
        "bbox": node.get("bbox"),
        "visual_bbox": metadata.get("visual_bbox"),
        "clickable": bool(node.get("clickable")),
        "editable": bool(node.get("editable")),
        "scrollable": bool(node.get("scrollable")),
        "enabled": bool(node.get("enabled", True)),
        "depth": node.get("depth", 0),
        "child_ids": node.get("child_ids", []),
        "metadata": node_metadata,
    }


def _scale_normalized_bboxes(
    graph: dict[str, Any], *, visual_width: float = 0, visual_height: float = 0
) -> None:
    """Restore model ``bbox`` values to the graph/XML coordinate space.

    Some iOS dumps expose logical XML dimensions (for example 414x736) while
    their bounds are expressed in Retina screenshot pixels (for example
    1242x2208).  Passing those mixed coordinates to the matcher clips the
    lower part of the source page to one normalized position and destroys the
    relative-layout evidence.  The graph is the canonical coordinate space;
    screenshot dimensions are used only to detect and undo an isotropic scale;
    ``visual_bbox`` remains in screenshot pixels for the review workbench.
    """

    width = float(graph.get("width") or 0)
    height = float(graph.get("height") or 0)
    nodes = graph.get("nodes") or []
    boxes = [node.get("bbox") for node in nodes if node.get("bbox")]
    if not boxes or width <= 1 or height <= 1:
        return

    scale_x = width / visual_width if visual_width > 1 else 1.0
    scale_y = height / visual_height if visual_height > 1 else 1.0
    relative_difference = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-9)
    screenshot_scaled_xml = relative_difference <= 0.03 and (
        scale_x <= 0.8 or scale_x >= 1.25
    )

    def restore_xml_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
        values = [float(value) for value in bbox]
        maximum = max(values)
        if maximum <= 1.000001:
            # Relative XML bounds must be expanded in XML dimensions, never
            # in screenshot dimensions.
            return [
                values[0] * width,
                values[1] * height,
                values[2] * width,
                values[3] * height,
            ]
        if screenshot_scaled_xml:
            return [
                values[0] * scale_x,
                values[1] * scale_y,
                values[2] * scale_x,
                values[3] * scale_y,
            ]
        return values

    def restore_visual_bbox(bbox: list[float] | tuple[float, ...]) -> list[float]:
        values = [float(value) for value in bbox]
        if max(values) <= 1.000001 and visual_width > 1 and visual_height > 1:
            return [
                values[0] * visual_width,
                values[1] * visual_height,
                values[2] * visual_width,
                values[3] * visual_height,
            ]
        return values

    for node in nodes:
        bbox = node.get("bbox")
        if bbox:
            node["bbox"] = restore_xml_bbox(bbox)
        visual_bbox = node.get("visual_bbox")
        if visual_bbox:
            node["visual_bbox"] = restore_visual_bbox(visual_bbox)


def _load_screens(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    screens: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            screen_id = str(raw["screen_id"])
            xml_path = (root / raw["xml_path"]).resolve()
            screenshot_path = (root / raw["screenshot_path"]).resolve()
            graph = graph_from_record(
                {
                    "xml": xml_path.read_text(encoding="utf-8"),
                    "width": raw.get("original_xml_width"),
                    "height": raw.get("original_xml_height"),
                },
                graph_id=screen_id,
            )
            graph_record = graph_to_record(graph)
            graph_record["nodes"] = [_node_payload(node) for node in graph_record["nodes"]]
            _scale_normalized_bboxes(
                graph_record,
                visual_width=float(raw.get("screenshot_width") or 0),
                visual_height=float(raw.get("screenshot_height") or 0),
            )
            screens[screen_id] = {
                "screen_id": screen_id,
                "page_id": screen_id,
                "platform": str(raw.get("platform") or "android"),
                "screenshot_path": str(screenshot_path),
                "display_width": float(raw.get("screenshot_width") or graph_record["width"]),
                "display_height": float(raw.get("screenshot_height") or graph_record["height"]),
                "graph": graph_record,
            }
    return screens


def _load_method_disagreements(path: Path | None, root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            source_path = metadata.get("source_screenshot_path") or row.get("source_screenshot_path")
            target_path = metadata.get("target_screenshot_path") or row.get("target_screenshot_path")
            if not source_path or not target_path:
                continue
            key = (str((root / source_path).resolve()), str((root / target_path).resolve()))
            entry = index.setdefault(key, {"anchor_count": 0, "category_counts": Counter(), "query_ids": [], "anchors": []})
            entry["anchor_count"] += 1
            category = str(row.get("category") or "method_disagreement")
            entry["category_counts"][category] += 1
            if row.get("query_id"):
                entry["query_ids"].append(str(row["query_id"]))
            entry["anchors"].append(
                {
                    "query_id": str(row.get("query_id") or ""),
                    "category": category,
                    "source_point": row.get("source_point"),
                    "ours_point": row.get("learned_target_point"),
                    "selector_point": row.get("selector_target_point"),
                    "gold_points": row.get("gold_target_points") or [],
                    "ours_rank": row.get("learned_gold_rank"),
                    "selector_rank": row.get("selector_gold_rank"),
                }
            )
    for entry in index.values():
        entry["category_counts"] = dict(sorted(entry["category_counts"].items()))
    return index


def _method_tags(disagreement: dict[str, Any]) -> list[dict[str, Any]]:
    category_counts = disagreement.get("category_counts") or {}
    tags = [{"id": "ours_vs_selector_disagree", "label": "我们的方案 ≠ selector", "count": disagreement["anchor_count"]}]
    labels = {
        "learned_correct_selector_wrong": ("ours_correct_selector_wrong", "我们的对 / selector 错"),
        "selector_correct_learned_wrong": ("selector_correct_ours_wrong", "selector 对 / 我们的错"),
        "learned_rank_advantage": ("ours_rank_advantage", "我们的排序更靠前"),
        "selector_rank_advantage": ("selector_rank_advantage", "selector 排序更靠前"),
    }
    for category, count in category_counts.items():
        if category in labels:
            tag_id, label = labels[category]
            tags.append({"id": tag_id, "label": label, "count": count})
    if disagreement["anchor_count"] > 1:
        tags.append({"id": "multi_anchor_disagreement", "label": "页面组内多处分歧", "count": disagreement["anchor_count"]})
    return tags


def build_pool(
    mappings_path: Path,
    screens_path: Path,
    root: Path,
    disagreements_path: Path | None = None,
) -> list[dict[str, Any]]:
    screens = _load_screens(screens_path, root)
    disagreements = _load_method_disagreements(disagreements_path, root)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    with mappings_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            app = str(raw["app"])
            source_id = str(raw["source"]["screen_id"])
            target_id = str(raw["target"]["screen_id"])
            source = screens[source_id]
            target = screens[target_id]
            key = (app, source_id, target_id)
            record = grouped.setdefault(
                key,
                {
                    "schema_version": "omnitransfer.ui_correspondence_pair.v1",
                    "pair_id": "",
                    "split": "diagnostic",
                    "label_status": "self_supervised",
                    "source": source,
                    "target": target,
                    "matches": [],
                    "partition_keys": [f"vision_widget:app:{app}"],
                    "provenance": {
                        "dataset": "vision_widget_mapping",
                        "app": app,
                        "source_split": raw.get("split"),
                        "mapping_ids": [],
                        "prelabel": "dataset_equivalent_widget",
                    },
                    "slices": {"app": app, "source_platform": source["platform"], "target_platform": target["platform"]},
                },
            )
            source_node_id = str(raw["source"]["node_id"])
            target_node_id = str(raw["target"]["node_id"])
            if any(match["source_node_id"] == source_node_id for match in record["matches"]):
                continue
            record["matches"].append(
                {
                    "source_node_id": source_node_id,
                    "target_node_ids": [target_node_id],
                    "label": "correspondence",
                    "source_point": _normalized_graph_point(
                        source["graph"], source_node_id
                    ),
                    "target_points": [],
                }
            )
            record["provenance"]["mapping_ids"].append(str(raw.get("mapping_id") or ""))
    rows = []
    for (app, source_id, target_id), record in sorted(grouped.items()):
        disagreement = disagreements.get(
            (str(Path(record["source"]["screenshot_path"]).resolve()), str(Path(record["target"]["screenshot_path"]).resolve()))
        )
        if disagreement:
            record["method_tags"] = _method_tags(disagreement)
            record["method_comparison"] = {
                "ours_label": "ours",
                "selector_label": "selector",
                "anchor_count": disagreement["anchor_count"],
                "category_counts": disagreement["category_counts"],
                "query_ids": disagreement["query_ids"],
                "anchors": disagreement["anchors"],
            }
        digest = hashlib.sha256(f"{app}\0{source_id}\0{target_id}".encode()).hexdigest()[:20]
        record["pair_id"] = f"vision-widget:{app}:{digest}"
        rows.append(record)
    return rows


def _normalized_graph_point(
    graph: dict[str, Any], node_id: str
) -> dict[str, Any] | None:
    """Return a node center in the graph's normalized XML coordinate space."""

    width = float(graph.get("width") or 0.0)
    height = float(graph.get("height") or 0.0)
    node = next(
        (
            item
            for item in graph.get("nodes", ())
            if item.get("node_id") == node_id
        ),
        None,
    )
    bbox = node.get("bbox") if node else None
    if not bbox or width <= 0.0 or height <= 0.0:
        return None
    return {
        "coordinate_space": "normalized_0_1",
        "x": ((float(bbox[0]) + float(bbox[2])) / 2.0) / width,
        "y": ((float(bbox[1]) + float(bbox[3])) / 2.0) / height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--screens", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--disagreements", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_pool(
        args.mappings.resolve(),
        args.screens.resolve(),
        args.repo_root.resolve(),
        args.disagreements.resolve() if args.disagreements else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Escape Unicode line/paragraph separators so JSONL remains one record per line.
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(json.dumps({"records": len(rows), "matches": sum(len(row["matches"]) for row in rows), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
