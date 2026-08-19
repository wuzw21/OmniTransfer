"""Static reviewer for unified mapping page-pair records."""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

from PIL import Image

from omnitransfer.mapping_dataset import validate_ui_correspondence_pair
from omnitransfer.numpy_v9_matcher import NumpyGeometricAlignmentMatcher
from omnitransfer.runtime import rank_action_candidates
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record

_MAX_REVIEW_REQUEST_BYTES = 32 * 1024 * 1024


def build_mapping_pair_review(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    pair_limit: int = 0,
    complex_only: bool = False,
    complex_limit: int = 0,
    external_payload: bool = False,
    method_tag: str | None = None,
    icon_only: bool = False,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize screenshots and a standalone review page for pair JSONL files."""

    if pair_limit < 0:
        raise ValueError("pair_limit must be non-negative")
    input_paths = [Path(path).expanduser().resolve() for path in inputs]
    records: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
    matcher = None
    if icon_only:
        if checkpoint is None:
            raise ValueError("icon-only review requires a NumPy matcher checkpoint")
        matcher = NumpyGeometricAlignmentMatcher.from_checkpoint(checkpoint)

    output = Path(output_dir).expanduser().resolve()
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    queue = []
    for record in records:
        item = _review_item(
            record,
            screenshots=screenshots,
            copied=copied,
            icon_only=icon_only,
            matcher=matcher,
        )
        if isinstance(item, list):
            queue.extend(item)
        elif item is not None:
            queue.append(item)
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
    payload = _review_payload(queue, icon_review=icon_only)
    # Unified error review adds its row index after the initial pair-review
    # build.  Preserve that index inside embedded HTML as well, so the
    # generated file remains usable from file:// without a sidecar fetch.
    unified_index_path = output / "review.html.unified_manual.index.json"
    if unified_index_path.is_file():
        payload["unified_manual_index"] = json.loads(
            unified_index_path.read_text(encoding="utf-8")
        )
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
        "icon_only": icon_only,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def create_review_server(
    review_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    root = Path(review_dir).expanduser().resolve()
    if not (root / "review.html").is_file():
        raise FileNotFoundError(f"review.html missing from review directory: {root}")

    class ReviewRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/rank_action_candidates":
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"status": "invalid_request", "reason": "endpoint_not_found"},
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
                if content_length <= 0 or content_length > _MAX_REVIEW_REQUEST_BYTES:
                    raise ValueError("invalid_content_length")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request_body_must_be_object")
                result = _rank_review_request(payload, root)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "status": "invalid_request",
                        "reason": str(error) or type(error).__name__,
                    },
                )
                return
            except Exception as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "status": "matcher_unavailable",
                        "reason": "live_mapping_failed",
                        "error": str(error) or type(error).__name__,
                    },
                )
                return
            self._write_json(HTTPStatus.OK, result)

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), ReviewRequestHandler)


def serve_mapping_pair_review(
    review_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = create_review_server(review_dir, host=host, port=port)
    print(f"Review workbench: http://{host}:{server.server_port}/review.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _rank_review_request(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    source_point = payload.get("source_point")
    point = None
    if isinstance(source_point, dict):
        x = float(source_point["x"])
        y = float(source_point["y"])
        # Relative XML captures are parsed by the runtime in [0, 1] space,
        # while older review pages sent canonical page pixels.  Accept both
        # during the migration; new pages already send values <= 1.
        source_xml = str(payload.get("source_xml") or "")
        source_size = payload.get("source_size") or {}
        if "xml-relative-0-1" in source_xml and max(abs(x), abs(y)) > 1.000001:
            width = float(source_size.get("width") or 0)
            height = float(source_size.get("height") or 0)
            if width > 0 and height > 0:
                x, y = x / width, y / height
        point = (x, y)
    source_offset = payload.get("source_offset")
    offset = None
    if isinstance(source_offset, dict):
        offset = (float(source_offset["x"]), float(source_offset["y"]))
    elif isinstance(source_offset, (list, tuple)) and len(source_offset) == 2:
        offset = (float(source_offset[0]), float(source_offset[1]))
    return rank_action_candidates(
        target_xml=str(payload.get("target_xml") or ""),
        source_xml=str(payload.get("source_xml") or ""),
        source_point=point,
        source_offset=offset,
        source_element_id=str(
            payload.get("source_element_id") or payload.get("source_node_id") or ""
        )
        or None,
        source_screenshot_path=_review_screenshot_path(
            payload.get("source_screenshot_path"), root
        ),
        target_screenshot_path=_review_screenshot_path(
            payload.get("target_screenshot_path"), root
        ),
        action_type=str(payload.get("action_type") or "click"),
        top_k=max(1, min(int(payload.get("top_k") or 5), 100)),
    )


def _review_screenshot_path(value: Any, root: Path) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https", "data"}:
        return None
    if parsed.scheme == "file":
        return str(Path(unquote(parsed.path)).expanduser().resolve())
    path = Path(unquote(parsed.path)).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


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
    icon_only: bool = False,
    matcher: NumpyGeometricAlignmentMatcher | None = None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if icon_only:
        return _icon_review_item(
            record,
            screenshots=screenshots,
            copied=copied,
            matcher=matcher,
        )
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


def _icon_review_item(
    record: dict[str, Any],
    *,
    screenshots: Path,
    copied: dict[Path, str],
    matcher: NumpyGeometricAlignmentMatcher | None,
) -> list[dict[str, Any]] | None:
    if matcher is None:
        raise ValueError("icon review matcher is required")
    source_graph = _matcher_graph(record["source"])
    target_graph = _matcher_graph(record["target"])
    hard_set = record.get("provenance", {}).get("hard_set")
    hard_items = (
        {
            str(item["source_node_id"]): item
            for item in hard_set.get("source_items", [])
            if isinstance(item, dict) and item.get("source_node_id")
        }
        if isinstance(hard_set, dict)
        else {}
    )
    source_icon_nodes = tuple(
        node
        for node in source_graph.nodes
        if (
            node.node_id in hard_items
            if hard_items
            else _is_icon_candidate(node, source_graph)
        )
    )
    if not source_icon_nodes:
        return None
    target_by_id = {node.node_id: node for node in target_graph.nodes}
    source_page = _review_side(
        record["source"],
        nodes=[],
        screenshots=screenshots,
        copied=copied,
    )
    target_page = _review_side(
        record["target"],
        nodes=[],
        screenshots=screenshots,
        copied=copied,
    )
    predictions = matcher.predict_many(
        source_graph,
        target_graph,
        source_node_ids=[node.node_id for node in source_icon_nodes],
        candidate_node_ids=target_by_id,
    )
    items = []
    for source_node in source_icon_nodes:
        prediction = predictions[source_node.node_id]
        hard_item = hard_items.get(source_node.node_id, {})
        target_node = target_by_id.get(
            prediction.target_node.node_id if prediction.target_node else ""
        )
        items.append(
            {
                "pair_id": f"{record['pair_id']}::icon::{source_node.node_id}",
                "dataset": str(record["provenance"].get("dataset") or "unknown"),
                "label_status": record["label_status"],
                "slices": record["slices"],
                "provenance": record["provenance"],
                "difficulty_score": float(hard_item.get("score") or 1.0),
                "difficulty_reasons": list(
                    hard_item.get("reasons") or ["icon_visual_review"]
                ),
                "method_tags": record.get("method_tags", []),
                "method_comparison": record.get("method_comparison"),
                "source": source_page,
                "target": target_page,
                "source_queries": {
                    source_node.node_id: {
                        "source_node": _review_node_from_ui(source_node),
                        "matcher_prediction": _matcher_prediction(prediction, target_node),
                    }
                },
                "icon_source_node_ids": [source_node.node_id],
                "matches": [
                    {
                        "source_node_id": source_node.node_id,
                        "target_node_ids": [],
                        "label": "no_correspondence",
                    }
                ],
            }
        )
    return items


def _matcher_graph(page: dict[str, Any]) -> UIGraph:
    graph_record = dict(page["graph"])
    graph_record["screenshot_path"] = page["screenshot_path"]
    if page.get("display_width") is not None:
        graph_record["display_width"] = page["display_width"]
    if page.get("display_height") is not None:
        graph_record["display_height"] = page["display_height"]
    return graph_from_record(graph_record, graph_id=page["page_id"])


def _is_icon_candidate(node: UINode, graph: UIGraph) -> bool:
    if (
        (node.bbox is None and not node.metadata.get("visual_bbox"))
        or node.text.strip()
        or node.content_desc.strip()
    ):
        return False
    bbox = node.metadata.get("visual_bbox") or node.bbox
    if not bbox or len(bbox) != 4:
        return False
    width = float(graph.width or 1.0)
    height = float(graph.height or 1.0)
    box_width = max(0.0, bbox[2] - bbox[0])
    box_height = max(0.0, bbox[3] - bbox[1])
    area_ratio = box_width * box_height / max(width * height, 1.0)
    max_side_ratio = max(box_width / width, box_height / height)
    aspect_ratio = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
    if (
        area_ratio <= 0.0
        or area_ratio > 0.05
        or max_side_ratio > 0.40
        or aspect_ratio > 6.0
    ):
        return False
    class_name = node.class_name.lower()
    container_tokens = ("viewgroup", "layout", "frame", "application", "window", "root")
    icon_tokens = ("image", "icon", "button", "control")
    return any(token in class_name for token in icon_tokens) or (
        node.clickable and not any(token in class_name for token in container_tokens)
    )


def _review_node_from_ui(node: UINode) -> dict[str, Any]:
    bbox = list(node.bbox) if node.bbox else None
    return {
        "node_id": node.node_id,
        "origin_id": node.origin_id,
        "text": node.text,
        "content_desc": node.content_desc,
        "resource_id": node.resource_id,
        "class_name": node.class_name,
        "bbox": bbox,
        "visual_bbox": list(node.metadata.get("visual_bbox")) if node.metadata.get("visual_bbox") else None,
        "clickable": node.clickable,
        "editable": node.editable,
    }


def _matcher_prediction(
    prediction: Any,
    target_node: UINode | None,
) -> dict[str, Any]:
    point = None
    if target_node and target_node.bbox:
        point = {
            "x": (target_node.bbox[0] + target_node.bbox[2]) / 2.0,
            "y": (target_node.bbox[1] + target_node.bbox[3]) / 2.0,
            "coordinate_space": "page_pixels",
            "node_id": target_node.node_id,
        }
    return {
        "node": _review_node_from_ui(target_node) if target_node else None,
        "top1_node": _review_node_from_ui(target_node) if target_node else None,
        "accepted": bool(target_node),
        "reason": prediction.reason,
        "probability": prediction.probability,
        "margin": prediction.margin,
        "point": point,
        "scores": [
            {"node_id": node_id, "score": score}
            for node_id, score in prediction.scores
        ],
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
    # Review interaction coordinates must stay in the graph/XML space. The
    # screenshot may be Retina-scaled (for example 1242x2208 for a 414x736
    # iOS XML page); display dimensions are retained as display metadata only.
    width = float(graph.get("width") or page.get("display_width") or 0)
    height = float(graph.get("height") or page.get("display_height") or 0)
    if width <= 0 or height <= 0:
        with Image.open(source) as image:
            width, height = image.size
    review_nodes = [_review_node(node) for node in nodes]
    all_nodes = [_review_node(node) for node in graph.get("nodes", [])]
    review_page = {
        "page_id": page["page_id"],
        "platform": page["platform"],
        "image_url": copied[source],
        "width": width,
        "height": height,
        "display_width": float(page.get("display_width") or width),
        "display_height": float(page.get("display_height") or height),
        "nodes": review_nodes,
        "candidates": all_nodes,
    }
    xml = page.get("xml") or page.get("xml_text")
    if xml:
        review_page["xml"] = str(xml)
    return review_page


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


def _review_payload(
    queue: list[dict[str, Any]],
    *,
    icon_review: bool = False,
) -> dict[str, Any]:
    tasks = []
    for item in queue:
        if icon_review:
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
                    "source": item["source"],
                    "target": item["target"],
                    "source_queries": item["source_queries"],
                    "provenance": item["provenance"],
                    "verdict_ids": [
                        "correct_correspondence",
                        "wrong_correspondence",
                        "discard_bad_evidence",
                    ],
                }
            )
            continue
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
                "protocol": "icon_visual_review" if icon_review else "mapping_pair_correspondence",
                "multi_mapping": not icon_review,
                "template_ids": (
                    [
                        "correct_correspondence",
                        "wrong_correspondence",
                        "ambiguous_or_absent",
                        "discard_bad_evidence",
                    ]
                    if not icon_review
                    else [
                        "correct_correspondence",
                        "wrong_correspondence",
                        "discard_bad_evidence",
                    ]
                ),
                "template_overrides": {},
                "keyboard_shortcuts": (
                    {"s": "save_draft", "1": "correct_correspondence", "2": "wrong_correspondence"}
                    if icon_review
                    else {}
                ),
                "show_candidate_toggle": True,
                "diagnostic_overlay": {
                    "enabled": True,
                    "methods": ["gold_proposal"],
                    "coordinate_space": "page_pixels",
                },
                "icon_candidate_policy": (
                    {
                        "text_and_content_desc_empty": True,
                        "preferred_bbox": "visual_bbox_then_bbox",
                        "max_area_ratio": 0.05,
                        "max_dimension_ratio": 0.40,
                        "max_aspect_ratio": 6.0,
                    }
                    if icon_review
                    else None
                ),
            },
        },
        "pairs": tasks,
    }


def _workbench_page(
    page: dict[str, Any],
    node: dict[str, Any] | None,
) -> dict[str, Any]:
    workbench_page = {
        "page_id": page["page_id"],
        "width": page["width"],
        "height": page["height"],
        "display_width": page.get("display_width", page["width"]),
        "display_height": page.get("display_height", page["height"]),
        "screenshot_path": page["image_url"],
        "node": node,
        "nodes": page.get("nodes", []),
        "candidates": page.get("candidates", page.get("nodes", [])),
    }
    # The live reviewer calls the canonical runtime matcher, whose input
    # contract is XML. Keep the source/target hierarchy in the workbench
    # payload; dropping it makes every live request fail as source_graph_required.
    xml = page.get("xml") or page.get("xml_text")
    if xml:
        workbench_page["xml"] = str(xml)
    return workbench_page


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
