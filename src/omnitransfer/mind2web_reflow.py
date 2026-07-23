"""Same-DOM viewport reflow capture for Mind2Web-style web pages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
from typing import Any

from omnitransfer.mapping_dataset import MAPPING_PAGE_PAIR_SCHEMA, validate_mapping_page_pair


@dataclass(frozen=True)
class ReflowViewport:
    """Named browser viewport used by a reflow track."""

    name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("viewport name must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("viewport dimensions must be positive")


DEFAULT_REFLOW_VIEWPORTS = (
    ReflowViewport("desktop", 1280, 900),
    ReflowViewport("phone", 390, 844),
    ReflowViewport("tablet", 820, 1180),
    ReflowViewport("foldable", 673, 841),
    ReflowViewport("webview", 412, 732),
)


def capture_mind2web_reflow_pair(
    page_path: str | Path,
    output_dir: str | Path,
    *,
    source_viewport: ReflowViewport,
    target_viewport: ReflowViewport,
    task_id: str = "local-smoke",
    action_id: str = "page",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render one HTML/MHTML page twice in the same Chromium page.

    Stable origins are injected before the first capture and survive viewport
    resize. They are label-generation metadata and must not be model features.
    """

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional browser dependency
        raise RuntimeError(
            "Mind2Web reflow capture requires Playwright. Install playwright and its Chromium browser."
        ) from exc

    source_path = Path(page_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_screenshot = destination / f"{task_id}_{action_id}_{source_viewport.name}.png"
    target_screenshot = destination / f"{task_id}_{action_id}_{target_viewport.name}.png"

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as exc:
            executable = _system_chromium_executable()
            if executable is None:
                raise RuntimeError(
                    "Playwright Chromium is absent and no system Chrome/Chromium was found. "
                    "Run `playwright install chromium`."
                ) from exc
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(executable),
            )
        page = browser.new_page(
            viewport={"width": source_viewport.width, "height": source_viewport.height},
            device_scale_factor=1,
        )
        page.goto(source_path.as_uri(), wait_until="load")
        page.evaluate(_INJECT_ORIGINS_JS)
        page.wait_for_timeout(100)
        source_capture = page.evaluate(_CAPTURE_GRAPH_JS, "source")
        page.screenshot(path=str(source_screenshot), full_page=False)

        page.set_viewport_size(
            {"width": target_viewport.width, "height": target_viewport.height}
        )
        page.wait_for_timeout(150)
        target_capture = page.evaluate(_CAPTURE_GRAPH_JS, "target")
        page.screenshot(path=str(target_screenshot), full_page=False)
        browser.close()

    record = build_mind2web_reflow_pair(
        source_capture,
        target_capture,
        source_screenshot=source_screenshot,
        target_screenshot=target_screenshot,
        source_viewport=source_viewport,
        target_viewport=target_viewport,
        task_id=task_id,
        action_id=action_id,
        source_asset=source_path,
    )
    stats = _pair_stats(record)
    return record, stats


def build_mind2web_reflow_pair(
    source_capture: dict[str, Any],
    target_capture: dict[str, Any],
    *,
    source_screenshot: str | Path,
    target_screenshot: str | Path,
    source_viewport: ReflowViewport,
    target_viewport: ReflowViewport,
    task_id: str,
    action_id: str,
    source_asset: str | Path,
) -> dict[str, Any]:
    """Convert two same-DOM captures into the common page-pair schema."""

    source_nodes = _normalize_capture_nodes(source_capture, side="source")
    target_nodes = _normalize_capture_nodes(target_capture, side="target")
    target_by_origin = {node["origin_id"]: node["node_id"] for node in target_nodes}
    matches: list[dict[str, Any]] = []
    for node in source_nodes:
        if not bool(node["metadata"].get("label_eligible")):
            continue
        target_id = target_by_origin.get(node["origin_id"])
        matches.append(
            {
                "source_node_id": node["node_id"],
                "target_node_ids": [target_id] if target_id else [],
                "label": "correspondence" if target_id else "no_correspondence",
            }
        )
    if not matches:
        raise ValueError("source capture has no eligible visible anchors")

    source_page_id = f"mind2web:{task_id}:{action_id}:{source_viewport.name}"
    target_page_id = f"mind2web:{task_id}:{action_id}:{target_viewport.name}"
    pair_id = _pair_id(task_id, action_id, source_viewport.name, target_viewport.name)
    return validate_mapping_page_pair(
        {
            "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": "diagnostic",
            "label_status": "unreviewed",
            "source": {
                "page_id": source_page_id,
                "platform": "web",
                "screenshot_path": str(Path(source_screenshot).expanduser().resolve()),
                "graph": {
                    "graph_id": source_page_id,
                    "width": source_viewport.width,
                    "height": source_viewport.height,
                    "nodes": source_nodes,
                    "metadata": {"viewport": source_viewport.name},
                },
            },
            "target": {
                "page_id": target_page_id,
                "platform": "web",
                "screenshot_path": str(Path(target_screenshot).expanduser().resolve()),
                "graph": {
                    "graph_id": target_page_id,
                    "width": target_viewport.width,
                    "height": target_viewport.height,
                    "nodes": target_nodes,
                    "metadata": {"viewport": target_viewport.name},
                },
            },
            "matches": matches,
            "partition_keys": [f"mind2web:task:{task_id}"],
            "provenance": {
                "dataset": "Mind2Web-Reflow",
                "annotation": "same_dom_origin_after_viewport_resize",
                "task_id": task_id,
                "action_id": action_id,
                "source_asset": str(Path(source_asset).expanduser().resolve()),
                "label_only_fields": ["backend_node_id", "data-omnitransfer-origin", "origin_id"],
                "render_validation": "required_before_promotion_to_gold",
            },
            "slices": {
                "platform": "web",
                "source_viewport": source_viewport.name,
                "target_viewport": target_viewport.name,
                "cross_form_factor": source_viewport.name != target_viewport.name,
                "browser_or_webview": True,
                "track": "same_dom_reflow",
            },
        }
    )


def _normalize_capture_nodes(capture: dict[str, Any], *, side: str) -> list[dict[str, Any]]:
    rows = capture.get("nodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{side} capture has no nodes")
    visible_origins = {
        str(row.get("origin") or "") for row in rows if str(row.get("origin") or "")
    }
    node_id_by_origin = {origin: f"{side}-node-{index}" for index, origin in enumerate(sorted(visible_origins))}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        origin = str(row.get("origin") or "").strip()
        if not origin or origin not in node_id_by_origin:
            continue
        parent_origin = str(row.get("parentOrigin") or "").strip()
        normalized.append(
            {
                "node_id": node_id_by_origin[origin],
                "origin_id": origin,
                "parent_id": node_id_by_origin.get(parent_origin),
                "text": str(row.get("text") or ""),
                "content_desc": str(row.get("contentDesc") or ""),
                "resource_id": str(row.get("resourceId") or ""),
                "class_name": str(row.get("className") or ""),
                "bbox": [float(value) for value in row.get("bbox") or ()],
                "clickable": bool(row.get("clickable")),
                "editable": bool(row.get("editable")),
                "scrollable": bool(row.get("scrollable")),
                "enabled": not bool(row.get("disabled")),
                "depth": int(row.get("depth") or 0),
                "child_ids": [],
                "metadata": {
                    "role": str(row.get("role") or ""),
                    "label_eligible": bool(row.get("labelEligible")),
                    "label_only": True,
                },
            }
        )
    children: dict[str, list[str]] = {}
    for node in normalized:
        if node["parent_id"]:
            children.setdefault(node["parent_id"], []).append(node["node_id"])
    for node in normalized:
        node["child_ids"] = sorted(children.get(node["node_id"], ()))
    return normalized


def _pair_stats(record: dict[str, Any]) -> dict[str, Any]:
    labels = [match["label"] for match in record["matches"]]
    source_nodes = {
        node["node_id"]: node for node in record["source"]["graph"]["nodes"]
    }
    target_nodes = {
        node["node_id"]: node for node in record["target"]["graph"]["nodes"]
    }
    moved = 0
    compared = 0
    source_width = float(record["source"]["graph"]["width"])
    source_height = float(record["source"]["graph"]["height"])
    target_width = float(record["target"]["graph"]["width"])
    target_height = float(record["target"]["graph"]["height"])
    for match in record["matches"]:
        if not match["target_node_ids"]:
            continue
        source_bbox = source_nodes[match["source_node_id"]]["bbox"]
        target_bbox = target_nodes[match["target_node_ids"][0]]["bbox"]
        source_geometry = _normalized_geometry(source_bbox, source_width, source_height)
        target_geometry = _normalized_geometry(target_bbox, target_width, target_height)
        compared += 1
        if max(abs(left - right) for left, right in zip(source_geometry, target_geometry)) > 0.03:
            moved += 1
    return {
        "source_nodes": len(source_nodes),
        "target_nodes": len(target_nodes),
        "anchors": len(labels),
        "correspondences": labels.count("correspondence"),
        "nulls": labels.count("no_correspondence"),
        "layout_changed_correspondences": moved,
        "layout_changed_rate": moved / max(1, compared),
    }


def _normalized_geometry(bbox: list[float], width: float, height: float) -> tuple[float, ...]:
    left, top, right, bottom = bbox
    return (
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    )


def _pair_id(task_id: str, action_id: str, source: str, target: str) -> str:
    digest = hashlib.blake2b(
        f"{task_id}\0{action_id}\0{source}\0{target}".encode(), digest_size=10
    ).hexdigest()
    return f"mind2web-reflow-pair-{digest}"


def _system_chromium_executable() -> Path | None:
    candidates: list[Path]
    if platform.system() == "Darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    elif platform.system() == "Windows":
        candidates = [
            Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        ]
    else:
        candidates = [
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
        ]
    return next((path for path in candidates if path.is_file()), None)


_INJECT_ORIGINS_JS = """
() => {
  const seen = new Set();
  let next = 0;
  for (const element of document.querySelectorAll('*')) {
    const backend = element.getAttribute('backend_node_id') ||
      element.getAttribute('data-backend-node-id') ||
      element.getAttribute('data-backend_node_id');
    let origin = backend ? `backend:${backend}` : `injected:${next++}`;
    while (seen.has(origin)) origin = `${origin}:duplicate:${next++}`;
    seen.add(origin);
    element.setAttribute('data-omnitransfer-origin', origin);
  }
  return seen.size;
}
"""


_CAPTURE_GRAPH_JS = """
(side) => {
  const nodes = [];
  const all = Array.from(document.querySelectorAll('[data-omnitransfer-origin]'));
  const visible = new Set();
  for (const element of all) {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    if (rect.width <= 0 || rect.height <= 0 || style.display === 'none' ||
        style.visibility === 'hidden' || Number(style.opacity) === 0 ||
        rect.right <= 0 || rect.bottom <= 0 || rect.left >= innerWidth || rect.top >= innerHeight) continue;
    visible.add(element);
  }
  const ownText = (element) => Array.from(element.childNodes)
    .filter(node => node.nodeType === Node.TEXT_NODE)
    .map(node => node.textContent || '').join(' ').replace(/\\s+/g, ' ').trim().slice(0, 512);
  const interactiveTags = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'OPTION', 'SUMMARY']);
  for (const element of all) {
    if (!visible.has(element)) continue;
    const rect = element.getBoundingClientRect();
    let parent = element.parentElement;
    while (parent && !visible.has(parent)) parent = parent.parentElement;
    const role = element.getAttribute('role') || '';
    const text = ownText(element);
    const contentDesc = element.getAttribute('aria-label') || element.getAttribute('alt') || '';
    const clickable = interactiveTags.has(element.tagName) || role === 'button' || role === 'link' ||
      element.hasAttribute('onclick') || getComputedStyle(element).cursor === 'pointer';
    const editable = element.matches('input,textarea,[contenteditable="true"]');
    const scrollable = element.scrollHeight > element.clientHeight || element.scrollWidth > element.clientWidth;
    nodes.push({
      side,
      origin: element.getAttribute('data-omnitransfer-origin'),
      parentOrigin: parent ? parent.getAttribute('data-omnitransfer-origin') : '',
      text,
      contentDesc,
      resourceId: element.id || element.getAttribute('name') || '',
      className: element.tagName.toLowerCase(),
      role,
      bbox: [
        Math.max(0, rect.left),
        Math.max(0, rect.top),
        Math.min(innerWidth, rect.right),
        Math.min(innerHeight, rect.bottom)
      ],
      clickable,
      editable,
      scrollable,
      disabled: element.matches(':disabled,[aria-disabled="true"]'),
      depth: (() => { let depth = 0; let node = element; while (node.parentElement) { depth++; node = node.parentElement; } return depth; })(),
      labelEligible: clickable || editable || Boolean(text) || Boolean(contentDesc)
    });
  }
  return {url: location.href, title: document.title, nodes};
}
"""
