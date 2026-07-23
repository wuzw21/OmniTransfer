"""Build contamination-free relative-XML widget-mapping data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from omnitransfer.baselines.anchor_vote_64d import (
    BoundNode,
    PublicWidgetPair,
    bind_public_bbox,
    load_public_widget_pairs,
)
from omnitransfer.ui_graph import BBox, UIGraph, UINode, graph_from_record


SCHEMA_VERSION = "omnitransfer_widget_mapping_relative_xml_v1"
SCREEN_SCHEMA_VERSION = "omnitransfer_relative_ui_xml_v1"
SPLIT_SEED = 17


@dataclass(frozen=True)
class ScreenData:
    """One original screen and its normalized XML representation."""

    screen_id: str
    platform: str
    graph: UIGraph
    screenshot_size: tuple[int, int]
    output_path: Path
    query_path: str
    screenshot_path: str
    relative_bboxes: dict[str, BBox | None]
    visual_bboxes: dict[str, BBox | None]
    candidate_node_ids: tuple[str, ...]
    node_indices: dict[str, int]


def clean_widget_mapping_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Convert public pairs into relative XML and canonical clean queries.

    The output keeps one normalized XML file per screen. Query rows reference
    real source and target XML nodes, use only real target XML nodes as
    candidates, and never copy app names or public annotations into semantics.
    """

    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"clean output already exists: {output}")
    pairs = load_public_widget_pairs(dataset)
    grouped: dict[tuple[str, str], list[PublicWidgetPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.source_screen, pair.target_screen)].append(pair)

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    queries_path = temporary / "queries.jsonl"
    mappings_path = temporary / "mappings.jsonl"
    screens_path = temporary / "screens.jsonl"
    split_counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    split_apps: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    screen_count = 0
    candidate_total = 0
    target_candidate_total = 0
    equivalent_gold_total = 0
    clipped_node_bboxes = 0
    omitted_node_bboxes = 0
    source_iou_total = 0.0
    target_iou_total = 0.0
    source_binding_count = 0
    target_binding_count = 0

    try:
        with (
            queries_path.open("w", encoding="utf-8") as query_handle,
            mappings_path.open("w", encoding="utf-8") as mapping_handle,
            screens_path.open("w", encoding="utf-8") as screen_handle,
        ):
            ordered_groups = sorted(
                grouped.items(),
                key=lambda item: min(pair.row_index for pair in item[1]),
            )
            for (source_screen, target_screen), screen_pairs in ordered_groups:
                source = _prepare_screen(
                    dataset.parent,
                    temporary,
                    source_screen,
                    platform="ios",
                    final_output=output,
                )
                target = _prepare_screen(
                    dataset.parent,
                    temporary,
                    target_screen,
                    platform="android",
                    final_output=output,
                )
                for screen in (source, target):
                    screen_count += 1
                    candidate_total += len(screen.candidate_node_ids)
                    clipped_node_bboxes += sum(
                        _bbox_was_clipped(node.bbox, screen.graph)
                        for node in screen.graph.nodes
                        if node.bbox is not None
                    )
                    omitted_node_bboxes += sum(
                        node.bbox is not None
                        and screen.relative_bboxes.get(node.node_id) is None
                        for node in screen.graph.nodes
                    )
                    screen_handle.write(
                        json.dumps(_screen_manifest_row(screen), ensure_ascii=False) + "\n"
                    )

                target_candidates = _candidate_rows(target)
                target_candidate_total += len(target_candidates)
                target_candidate_ids = {
                    candidate["candidate_id"] for candidate in target_candidates
                }
                for pair in screen_pairs:
                    source_binding = bind_public_bbox(
                        source.graph,
                        pair.source_bbox,
                        pair.source_class,
                        screenshot_size=source.screenshot_size,
                    )
                    target_binding = bind_public_bbox(
                        target.graph,
                        pair.target_bbox,
                        pair.target_class,
                        screenshot_size=target.screenshot_size,
                    )
                    if source_binding is None or target_binding is None:
                        raise ValueError(
                            f"row {pair.row_index} failed XML binding: "
                            f"source={source_binding is not None} "
                            f"target={target_binding is not None}"
                        )
                    source_binding_count += 1
                    target_binding_count += 1
                    source_iou_total += source_binding.iou
                    target_iou_total += target_binding.iou
                    if target_binding.node.node_id not in target_candidate_ids:
                        raise ValueError(
                            f"row {pair.row_index} gold node is not a fixed XML candidate: "
                            f"{target_binding.node.node_id}"
                        )
                    split = _split_for_app(pair.app, split_seed)
                    split_counts[split] += 1
                    split_apps[split].add(pair.app)
                    equivalent_ids = _gold_equivalent_ids(
                        target,
                        target_binding,
                        allowed_ids=target_candidate_ids,
                    )
                    equivalent_gold_total += len(equivalent_ids)
                    query_handle.write(
                        json.dumps(
                            _query_row(
                                pair,
                                source,
                                target,
                                source_binding,
                                target_binding,
                                target_candidates=target_candidates,
                                equivalent_ids=equivalent_ids,
                                split=split,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mapping_handle.write(
                        json.dumps(
                            _mapping_row(
                                pair,
                                source,
                                target,
                                source_binding,
                                target_binding,
                                equivalent_ids=equivalent_ids,
                                split=split,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        overlaps = {
            left: sorted(split_apps[left] & split_apps[right])
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
            if split_apps[left] & split_apps[right]
        }
        if overlaps:
            raise ValueError(f"app-disjoint split violation: {overlaps}")
        query_count = sum(split_counts.values())
        audit = {
            "schema_version": SCHEMA_VERSION,
            "raw_pair_count": len(pairs),
            "query_count": query_count,
            "screen_count": screen_count,
            "screen_pair_count": len(grouped),
            "source_binding_count": source_binding_count,
            "target_binding_count": target_binding_count,
            "source_binding_coverage": _ratio(source_binding_count, len(pairs)),
            "target_binding_coverage": _ratio(target_binding_count, len(pairs)),
            "mean_source_binding_iou": _ratio(source_iou_total, source_binding_count),
            "mean_target_binding_iou": _ratio(target_iou_total, target_binding_count),
            "source_semantics_fallback_count": 0,
            "public_candidate_count": 0,
            "duplicate_candidate_id_count": 0,
            "gold_missing_from_candidates": 0,
            "relative_bbox_out_of_range_count": 0,
            "candidate_total_across_screens": candidate_total,
            "target_candidate_total_across_screens": target_candidate_total,
            "mean_candidates_per_target_screen": _ratio(
                target_candidate_total, len(grouped)
            ),
            "equivalent_gold_id_total": equivalent_gold_total,
            "clipped_node_bbox_count": clipped_node_bboxes,
            "omitted_offscreen_or_invalid_bbox_count": omitted_node_bboxes,
            "split_seed": split_seed,
            "split_counts": split_counts,
            "split_app_counts": {
                split: len(apps) for split, apps in split_apps.items()
            },
            "split_app_overlap_count": 0,
        }
        _validate_clean_output(queries_path, screens_path, expected_queries=len(pairs))
        audit_path = temporary / "audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_dataset": str(dataset),
            "source_sha256": _sha256(dataset),
            "coordinate_space": "xml_relative_0_1",
            "visual_coordinate_space": "screenshot_relative_0_1",
            "screen_format": "nested_xml",
            "queries": "queries.jsonl",
            "mappings": "mappings.jsonl",
            "screens": "screens.jsonl",
            "audit": "audit.json",
            "query_count": len(pairs),
            "screen_count": screen_count,
            "split_seed": split_seed,
            "split_counts": split_counts,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["queries_sha256"] = _sha256(queries_path)
        manifest["mappings_sha256"] = _sha256(mappings_path)
        manifest["screens_sha256"] = _sha256(screens_path)
        manifest["audit_sha256"] = _sha256(audit_path)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.rename(output)
        return {**manifest, "output": str(output), "audit_summary": audit}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _prepare_screen(
    dataset_root: Path,
    temporary: Path,
    screen_id: str,
    *,
    platform: str,
    final_output: Path,
) -> ScreenData:
    xml_path = dataset_root / f"{screen_id}.xml"
    screenshot_path = dataset_root / f"{screen_id}.png"
    graph = graph_from_record(
        {"xml": xml_path.read_text(encoding="utf-8")}, graph_id=screen_id
    )
    if not graph.width or not graph.height:
        raise ValueError(f"screen has no XML viewport: {screen_id}")
    screenshot_size = _png_size(screenshot_path)
    relative_bboxes = {
        node.node_id: _relative_xml_bbox(node.bbox, graph)
        for node in graph.nodes
    }
    visual_bboxes = {
        node.node_id: _relative_visual_bbox(
            node.bbox,
            graph,
            screenshot_size=screenshot_size,
        )
        for node in graph.nodes
    }
    candidate_node_ids = tuple(
        node.node_id
        for node in graph.nodes
        if _is_fixed_candidate(node, relative_bboxes[node.node_id])
    )
    output_path = temporary / "screens" / f"{screen_id}.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _relative_xml_text(
            graph,
            screen_id=screen_id,
            platform=platform,
            screenshot_size=screenshot_size,
            relative_bboxes=relative_bboxes,
            visual_bboxes=visual_bboxes,
            candidate_node_ids=set(candidate_node_ids),
        ),
        encoding="utf-8",
    )
    final_xml_path = final_output / output_path.relative_to(temporary)
    return ScreenData(
        screen_id=screen_id,
        platform=platform,
        graph=graph,
        screenshot_size=screenshot_size,
        output_path=output_path,
        query_path=_portable_path(final_xml_path),
        screenshot_path=_portable_path(screenshot_path),
        relative_bboxes=relative_bboxes,
        visual_bboxes=visual_bboxes,
        candidate_node_ids=candidate_node_ids,
        node_indices={node.node_id: index for index, node in enumerate(graph.nodes)},
    )


def _relative_xml_text(
    graph: UIGraph,
    *,
    screen_id: str,
    platform: str,
    screenshot_size: tuple[int, int],
    relative_bboxes: dict[str, BBox | None],
    visual_bboxes: dict[str, BBox | None],
    candidate_node_ids: set[str],
) -> str:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    roots = [node for node in graph.nodes if node.parent_id is None]
    if len(roots) != 1:
        raise ValueError(f"screen {screen_id} must contain exactly one XML root")

    def build(node: UINode) -> ET.Element:
        attrs = {
            "node-id": node.node_id,
            "origin-id": node.origin_id,
            "class": node.class_name,
            "text": node.text,
            "content-desc": node.content_desc,
            "resource-id": node.resource_id,
            "clickable": _bool_text(node.clickable),
            "editable": _bool_text(node.editable),
            "scrollable": _bool_text(node.scrollable),
            "enabled": _bool_text(node.enabled),
            "visible": _bool_text(bool(node.metadata.get("visible", True))),
            "candidate": _bool_text(node.node_id in candidate_node_ids),
        }
        bbox = relative_bboxes[node.node_id]
        visual_bbox = visual_bboxes[node.node_id]
        if bbox is not None:
            attrs["bounds"] = _bbox_text(bbox)
        if visual_bbox is not None:
            attrs["visual-bbox"] = _bbox_text(visual_bbox)
        if node.bbox is not None and bbox is not None and _bbox_was_clipped(
            node.bbox, graph
        ):
            attrs["bounds-clipped"] = "true"
        element = ET.Element("node", attrs)
        for child_id in node.child_ids:
            child = nodes_by_id.get(child_id)
            if child is not None:
                element.append(build(child))
        return element

    root = build(roots[0])
    root.attrib.update(
        {
            "schema-version": SCREEN_SCHEMA_VERSION,
            "screen-id": screen_id,
            "platform": platform,
            "coordinate-space": "xml-relative-0-1",
            "visual-coordinate-space": "screenshot-relative-0-1",
            "width": "1",
            "height": "1",
            "original-xml-width": _number_text(float(graph.width)),
            "original-xml-height": _number_text(float(graph.height)),
            "screenshot-width": str(screenshot_size[0]),
            "screenshot-height": str(screenshot_size[1]),
        }
    )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _candidate_rows(screen: ScreenData) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    by_id = {node.node_id: node for node in screen.graph.nodes}
    for node_id in screen.candidate_node_ids:
        node = by_id[node_id]
        candidates.append(
            {
                "candidate_id": node.node_id,
                "bbox": list(_required_bbox(screen.relative_bboxes[node.node_id])),
                "text": node.text,
                "content_desc": node.content_desc,
                "resource_id": node.resource_id,
                "class_name": node.class_name,
                "clickable": node.clickable,
                "editable": node.editable,
                "scrollable": node.scrollable,
                "enabled": node.enabled,
                "node_index": screen.node_indices[node.node_id],
                "visual_bbox": (
                    list(screen.visual_bboxes[node.node_id])
                    if screen.visual_bboxes[node.node_id] is not None
                    else None
                ),
            }
        )
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError(f"duplicate candidate ids on screen {screen.screen_id}")
    return candidates


def _query_row(
    pair: PublicWidgetPair,
    source: ScreenData,
    target: ScreenData,
    source_binding: BoundNode,
    target_binding: BoundNode,
    *,
    target_candidates: list[dict[str, Any]],
    equivalent_ids: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    source_node = source_binding.node
    source_bbox = _required_bbox(source.relative_bboxes[source_node.node_id])
    return {
        "query_id": _query_id(pair),
        "source": {
            "id": source_node.node_id,
            "node_id": source_node.node_id,
            "text": source_node.text,
            "content_desc": source_node.content_desc,
            "resource_id": source_node.resource_id,
            "class_name": source_node.class_name,
            "bounds": list(source_bbox),
            "x": (source_bbox[0] + source_bbox[2]) / 2.0,
            "y": (source_bbox[1] + source_bbox[3]) / 2.0,
            "clickable": source_node.clickable,
            "editable": source_node.editable,
            "scrollable": source_node.scrollable,
            "enabled": source_node.enabled,
            "metadata": {
                "node_index": source.node_indices[source_node.node_id],
                "screen": source.screen_id,
                "platform": source.platform,
                "binding_iou": source_binding.iou,
                "semantics_source": "bound_xml_node",
            },
        },
        "target_candidates": target_candidates,
        "gold_candidate_id": target_binding.node.node_id,
        "dataset": "vision_widget_mapping",
        "split": split,
        "label_status": "public_pair_bound_to_xml",
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "coordinate_space": "xml_relative_0_1",
            "visual_coordinate_space": "screenshot_relative_0_1",
            "app": pair.app,
            "category": pair.source_screen.split("/", 1)[0],
            "widget_type": pair.widget_type,
            "row_index": pair.row_index,
            "source_screen": pair.source_screen,
            "target_screen": pair.target_screen,
            "source_xml_path": source.query_path,
            "target_xml_path": target.query_path,
            "source_screenshot_path": source.screenshot_path,
            "target_screenshot_path": target.screenshot_path,
            "source_node_id": source_node.node_id,
            "target_node_id": target_binding.node.node_id,
            "gold_equivalent_candidate_ids": list(equivalent_ids),
            "target_binding_iou": target_binding.iou,
        },
    }


def _mapping_row(
    pair: PublicWidgetPair,
    source: ScreenData,
    target: ScreenData,
    source_binding: BoundNode,
    target_binding: BoundNode,
    *,
    equivalent_ids: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mapping_id": _query_id(pair),
        "split": split,
        "app": pair.app,
        "category": pair.source_screen.split("/", 1)[0],
        "widget_type": pair.widget_type,
        "row_index": pair.row_index,
        "source": {
            "screen_id": source.screen_id,
            "xml_path": source.query_path,
            "node_id": source_binding.node.node_id,
            "bbox": list(_required_bbox(source.relative_bboxes[source_binding.node.node_id])),
            "binding_iou": source_binding.iou,
        },
        "target": {
            "screen_id": target.screen_id,
            "xml_path": target.query_path,
            "node_id": target_binding.node.node_id,
            "bbox": list(_required_bbox(target.relative_bboxes[target_binding.node.node_id])),
            "equivalent_node_ids": list(equivalent_ids),
            "binding_iou": target_binding.iou,
        },
    }


def _gold_equivalent_ids(
    screen: ScreenData,
    binding: BoundNode,
    *,
    allowed_ids: set[str],
) -> tuple[str, ...]:
    gold_bbox = screen.relative_bboxes[binding.node.node_id]
    values = [binding.node.node_id, *binding.equivalent_node_ids]
    if gold_bbox is not None:
        values.extend(
            node.node_id
            for node in screen.graph.nodes
            if node.node_id in allowed_ids
            and screen.relative_bboxes[node.node_id] == gold_bbox
        )
    return tuple(dict.fromkeys(node_id for node_id in values if node_id in allowed_ids))


def _relative_xml_bbox(bbox: BBox | None, graph: UIGraph) -> BBox | None:
    if bbox is None or not graph.width or not graph.height:
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None
    width = float(graph.width)
    height = float(graph.height)
    clipped = (
        max(0.0, min(width, bbox[0])),
        max(0.0, min(height, bbox[1])),
        max(0.0, min(width, bbox[2])),
        max(0.0, min(height, bbox[3])),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return _round_bbox(
        (
            clipped[0] / width,
            clipped[1] / height,
            clipped[2] / width,
            clipped[3] / height,
        )
    )


def _relative_visual_bbox(
    bbox: BBox | None,
    graph: UIGraph,
    *,
    screenshot_size: tuple[int, int],
) -> BBox | None:
    if bbox is None or not graph.width or not graph.height:
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None
    screenshot_width, screenshot_height = screenshot_size
    scale_x = float(graph.width) / float(screenshot_width)
    scale_y = float(graph.height) / float(screenshot_height)
    relative_difference = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-9)
    screenshot_scaled_xml = relative_difference <= 0.03 and (
        scale_x <= 0.8 or scale_x >= 1.25
    )
    if screenshot_scaled_xml:
        values = (
            bbox[0] / float(graph.width),
            bbox[1] / float(graph.height),
            bbox[2] / float(graph.width),
            bbox[3] / float(graph.height),
        )
    else:
        values = (
            bbox[0] / float(screenshot_width),
            bbox[1] / float(screenshot_height),
            bbox[2] / float(screenshot_width),
            bbox[3] / float(screenshot_height),
        )
    clipped = tuple(max(0.0, min(1.0, value)) for value in values)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return _round_bbox(clipped)


def _is_fixed_candidate(node: UINode, bbox: BBox | None) -> bool:
    if bbox is None:
        return False
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return bool(node.metadata.get("visible", True)) and area > 0.0 and area < 0.98


def _bbox_was_clipped(bbox: BBox, graph: UIGraph) -> bool:
    if not graph.width or not graph.height:
        return False
    if not all(math.isfinite(value) for value in bbox):
        return False
    return bool(
        bbox[0] < 0.0
        or bbox[1] < 0.0
        or bbox[2] > float(graph.width)
        or bbox[3] > float(graph.height)
    )


def _screen_manifest_row(screen: ScreenData) -> dict[str, Any]:
    return {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "screen_id": screen.screen_id,
        "platform": screen.platform,
        "xml_path": screen.query_path,
        "screenshot_path": screen.screenshot_path,
        "original_xml_width": screen.graph.width,
        "original_xml_height": screen.graph.height,
        "screenshot_width": screen.screenshot_size[0],
        "screenshot_height": screen.screenshot_size[1],
        "node_count": len(screen.graph.nodes),
        "candidate_count": len(screen.candidate_node_ids),
        "coordinate_space": "xml_relative_0_1",
        "visual_coordinate_space": "screenshot_relative_0_1",
    }


def _validate_clean_output(
    queries_path: Path,
    screens_path: Path,
    *,
    expected_queries: int,
) -> None:
    query_count = 0
    for line_number, line in enumerate(queries_path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        query_count += 1
        row = json.loads(line)
        source = row["source"]
        _validate_bbox(source["bounds"], label=f"query {line_number} source")
        candidate_ids: set[str] = set()
        for candidate in row["target_candidates"]:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in candidate_ids:
                raise ValueError(f"query {line_number} has duplicate candidate ids")
            candidate_ids.add(candidate_id)
            _validate_bbox(candidate["bbox"], label=f"query {line_number} candidate")
        if row["gold_candidate_id"] not in candidate_ids:
            raise ValueError(f"query {line_number} gold is absent from candidates")
        if source.get("metadata", {}).get("semantics_source") != "bound_xml_node":
            raise ValueError(f"query {line_number} contains non-XML source semantics")
        if any(candidate_id.startswith("public_") for candidate_id in candidate_ids):
            raise ValueError(f"query {line_number} contains public pseudo-candidates")
    if query_count != expected_queries:
        raise ValueError(
            f"clean query count mismatch: expected={expected_queries} actual={query_count}"
        )
    screen_count = sum(1 for line in screens_path.open(encoding="utf-8") if line.strip())
    if screen_count <= 0:
        raise ValueError("clean screen manifest is empty")


def _validate_bbox(value: Any, *, label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} has invalid bbox")
    if not all(isinstance(item, (int, float)) and 0.0 <= float(item) <= 1.0 for item in value):
        raise ValueError(f"{label} bbox is outside [0,1]")
    if float(value[2]) <= float(value[0]) or float(value[3]) <= float(value[1]):
        raise ValueError(f"{label} bbox is empty")


def _split_for_app(app: str, seed: int) -> str:
    digest = hashlib.blake2b(f"{seed}:{app}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    return "train" if bucket < 70 else "dev" if bucket < 85 else "test"


def _query_id(pair: PublicWidgetPair) -> str:
    value = "|".join(
        (
            pair.source_screen,
            pair.target_screen,
            pair.source_class,
            str(pair.source_bbox),
            pair.target_class,
            str(pair.target_bbox),
            str(pair.row_index),
        )
    )
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"vision_widget_mapping:{pair.app}:row_{pair.row_index}:{digest}"


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _portable_path(path: Path) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path.resolve())


def _required_bbox(value: BBox | None) -> BBox:
    if value is None:
        raise ValueError("required XML node has no valid relative bbox")
    return value


def _bbox_text(bbox: BBox) -> str:
    return f"[{_number_text(bbox[0])},{_number_text(bbox[1])}]" f"[{_number_text(bbox[2])},{_number_text(bbox[3])}]"


def _round_bbox(bbox: Iterable[float]) -> BBox:
    values = tuple(round(float(value), 9) for value in bbox)
    return values[0], values[1], values[2], values[3]


def _number_text(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "SCREEN_SCHEMA_VERSION",
    "clean_widget_mapping_dataset",
]
