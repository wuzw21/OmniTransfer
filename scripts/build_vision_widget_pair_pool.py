#!/usr/bin/env python3
"""Convert the clean vision-widget mappings into canonical page-pair records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from omnitransfer.ios_adapter import load_ios_screen
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


def _expand_relative_coordinate_spaces(
    graph_record: dict[str, Any], *, display_width: float, display_height: float
) -> None:
    """Materialize normalized XML/screenshot bounds before pool construction."""

    width = float(graph_record.get("width") or 0.0)
    height = float(graph_record.get("height") or 0.0)
    for node in graph_record.get("nodes") or []:
        bbox = node.get("bbox")
        if bbox and max(float(value) for value in bbox) <= 1.000001:
            node["bbox"] = [
                float(bbox[0]) * width,
                float(bbox[1]) * height,
                float(bbox[2]) * width,
                float(bbox[3]) * height,
            ]
        metadata = node.get("metadata") or {}
        visual_bbox = metadata.get("visual_bbox")
        if visual_bbox and max(float(value) for value in visual_bbox) <= 1.000001:
            metadata["visual_bbox"] = [
                float(visual_bbox[0]) * display_width,
                float(visual_bbox[1]) * display_height,
                float(visual_bbox[2]) * display_width,
                float(visual_bbox[3]) * display_height,
            ]
        node["metadata"] = metadata


def _load_screens(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    screens: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            screen_id = str(raw["screen_id"])
            if str(raw.get("platform") or "").strip().lower() in {"ios", "iphoneos", "ipad os", "ipados"}:
                screens[screen_id] = load_ios_screen(raw, root=root)
                continue
            xml_path = (root / raw["xml_path"]).resolve()
            screenshot_path = (root / raw["screenshot_path"]).resolve()
            xml = xml_path.read_text(encoding="utf-8")
            graph = graph_from_record(
                {
                    "xml": xml,
                    "width": raw.get("original_xml_width"),
                    "height": raw.get("original_xml_height"),
                },
                graph_id=screen_id,
            )
            graph_record = graph_to_record(graph)
            if "relative_0_1" in str(raw.get("coordinate_space") or ""):
                _expand_relative_coordinate_spaces(
                    graph_record,
                    display_width=float(raw.get("screenshot_width") or graph_record["width"]),
                    display_height=float(raw.get("screenshot_height") or graph_record["height"]),
                )
            graph_record["nodes"] = [_node_payload(node) for node in graph_record["nodes"]]
            screens[screen_id] = {
                "screen_id": screen_id,
                "page_id": screen_id,
                "platform": str(raw.get("platform") or "android"),
                "screenshot_path": str(screenshot_path),
                "display_width": float(raw.get("screenshot_width") or graph_record["width"]),
                "display_height": float(raw.get("screenshot_height") or graph_record["height"]),
                "xml": xml,
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
