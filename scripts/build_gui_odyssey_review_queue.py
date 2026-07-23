#!/usr/bin/env python3
"""Build GUIOdyssey correspondence candidates and a node review workspace."""

from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
from typing import Any

from omnitransfer.gui_odyssey_images import gui_odyssey_image_index
from omnitransfer.gui_odyssey_review import (
    build_gui_odyssey_review_queue,
    build_gui_odyssey_sequence_alignment_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or re-render a GUIOdyssey node correspondence review queue."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", type=Path, nargs="+")
    source.add_argument(
        "--review-queue",
        type=Path,
        help="Re-render an existing JSONL queue without rebuilding candidates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-url-prefix", default="screenshots")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--likely-percent", type=int, default=80)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--require-individual-image-mirror", action="store_true")
    parser.add_argument("--sequence-alignment", action="store_true")
    parser.add_argument("--min-step-score", type=float, default=0.58)
    parser.add_argument("--min-aligned-steps", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.likely_percent <= 100:
        raise SystemExit("--likely-percent must be between 0 and 100")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.review_queue is not None:
        rows = _upgrade_review_rows(_load_review_queue(args.review_queue))
        (output / "review_queue.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (output / "review.html").write_text(_review_html(rows), encoding="utf-8")
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["review_schema_version"] = rows[0]["schema_version"] if rows else None
            manifest["human_annotation"] = "multi_node_graph_edges"
            manifest["candidate_queue_origin"] = "existing_queue_migrated_without_reranking"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "pairs_loaded": len(rows),
                    "queue_file": "review_queue.jsonl",
                    "review_file": "review.html",
                }
            )
        )
        return

    episodes = _load_episodes(tuple(args.annotations or ()))
    available_screenshots = None
    if args.require_individual_image_mirror:
        requested = {
            str(step.get("screenshot") or "")
            for episode in episodes
            for step in episode.get("steps") or ()
            if isinstance(step, dict) and step.get("screenshot")
        }
        available_screenshots = set(gui_odyssey_image_index(requested))
    if args.sequence_alignment:
        rows, manifest = build_gui_odyssey_sequence_alignment_queue(
            episodes,
            limit=args.limit,
            min_step_score=args.min_step_score,
            min_aligned_steps=args.min_aligned_steps,
            seed=args.seed,
            images_url_prefix=args.images_url_prefix,
            available_screenshots=available_screenshots,
        )
    else:
        rows, manifest = build_gui_odyssey_review_queue(
            episodes,
            limit=args.limit,
            likely_fraction=args.likely_percent / 100.0,
            seed=args.seed,
            images_url_prefix=args.images_url_prefix,
            available_screenshots=available_screenshots,
        )
    queue_path = output / "review_queue.jsonl"
    queue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "review.html").write_text(_review_html(rows), encoding="utf-8")
    manifest = {
        **manifest,
        "episodes_loaded": len(episodes),
        "queue_file": queue_path.name,
        "review_file": "review.html",
        "images_url_prefix": args.images_url_prefix,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _load_episodes(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    episodes: list[dict[str, Any]] = []
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("steps"), list):
            episodes.append(value)
        elif isinstance(value, list):
            episodes.extend(
                item for item in value if isinstance(item, dict) and isinstance(item.get("steps"), list)
            )
    return episodes


def _load_review_queue(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return [
        value
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
        if isinstance((value := json.loads(line)), dict)
    ]


def _upgrade_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    upgraded = []
    for raw in rows:
        row = deepcopy(raw)
        sequence = bool(row.get("selection", {}).get("trajectory_pair_id"))
        row["schema_version"] = (
            "omnitransfer_guiodyssey_sequence_node_alignment_v1"
            if sequence
            else "omnitransfer_guiodyssey_node_review_v1"
        )
        for side in ("source", "target"):
            entry = row[side]
            entry.pop("bbox", None)
            entry.pop("bbox_normalized", None)
        selection = row["selection"]
        selection.pop("bbox_quality", None)
        selection.pop("bbox_scale_ratio", None)
        selection["reasons"] = [
            reason for reason in selection.get("reasons", ()) if "bbox" not in reason
        ]
        previous = row.get("annotation") or {}
        row["annotation"] = {
            "status": previous.get("status") or "unreviewed",
            "label": previous.get("label"),
            "source_nodes": previous.get("source_nodes")
            or [_review_node("source-1", row["source"])],
            "target_nodes": previous.get("target_nodes")
            or [_review_node("target-1", row["target"])],
            "matches": previous.get("matches") or [],
            "notes": previous.get("notes") or "",
            "annotator": previous.get("annotator") or "",
        }
        upgraded.append(row)
    return upgraded


def _review_node(node_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "point_normalized": list(entry["point_normalized"]),
        "label": str(entry.get("instruction") or node_id),
        "provenance": "guiodyssey_action_point",
    }


def _review_html(rows: list[dict[str, Any]], *, redirect_file: bool = True) -> str:
    del redirect_file
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape("OmniTransfer · GUIOdyssey 节点对应标注")
    template = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #080b12; color: #eef2f8; }
header { position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 11px 16px; background: #0d111bcc; border-bottom: 1px solid #283041; backdrop-filter: blur(16px); }
header strong { margin-right: 8px; letter-spacing: .01em; }
button, input { font: inherit; }
button { padding: 8px 12px; border: 1px solid #394359; border-radius: 7px; background: #151b28; color: #eef2f8; cursor: pointer; }
button:hover { border-color: #6f7f9f; background: #1d2535; }
button:focus-visible, input:focus-visible { outline: 2px solid #63e6be; outline-offset: 2px; }
.primary { border-color: #2f9e76; background: #147d5c; }
.danger { color: #ffb4b4; }
main { max-width: 1680px; margin: auto; padding: 16px; }
.meta { display: grid; grid-template-columns: 1fr auto; gap: 12px; margin-bottom: 12px; color: #aeb9ce; font-size: 13px; }
.reason { color: #7f8da8; }
.score { font-variant-numeric: tabular-nums; }
.screens { position: relative; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
.screen { position: relative; z-index: 2; min-width: 0; padding: 12px; background: #0e131e; border: 1px solid #252d3d; border-radius: 10px; }
.screen h3 { margin: 0 0 10px; font-size: 14px; color: #b7c2d7; }
.image-wrap { position: relative; min-height: 220px; background: #03050a; overflow: hidden; cursor: crosshair; user-select: none; }
.image-wrap img { display: block; width: 100%; height: auto; pointer-events: none; }
.node { position: absolute; z-index: 4; width: 30px; height: 30px; padding: 0; transform: translate(-50%, -50%); border: 2px solid #07110e; border-radius: 50%; background: #63e6be; color: #07110e; font-size: 11px; font-weight: 800; box-shadow: 0 0 0 3px #63e6be55, 0 5px 14px #000a; touch-action: none; }
.node:hover { background: #8cf0d1; }
.node.selected { background: #ffd166; box-shadow: 0 0 0 4px #ffd16655, 0 5px 14px #000a; }
.node-editor { display: grid; grid-template-columns: auto minmax(120px, 1fr) auto; gap: 8px; align-items: center; min-height: 43px; padding-top: 9px; }
.node-editor input, header input { min-width: 0; padding: 8px 9px; border: 1px solid #333d51; border-radius: 6px; background: #090d15; color: #eef2f8; }
.instruction { margin: 8px 0 5px; font-size: 16px; font-weight: 680; }
.small { color: #97a5bf; font-size: 12px; line-height: 1.45; }
.hint { color: #6f7f9f; font-size: 12px; }
.edge-layer { position: absolute; inset: 0; z-index: 6; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
.edge-layer line { stroke: #ffd166; stroke-width: 3; stroke-linecap: round; filter: drop-shadow(0 2px 3px #000); }
.relation-bar { display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: start; margin-top: 14px; padding: 12px 0; border-top: 1px solid #252d3d; border-bottom: 1px solid #252d3d; }
.relations { display: flex; flex-wrap: wrap; gap: 7px; min-height: 34px; align-items: center; }
.relation { display: inline-flex; align-items: center; gap: 7px; padding: 6px 8px; border: 1px solid #584d2e; border-radius: 999px; background: #211d12; color: #ffe29a; font-size: 12px; }
.relation button { padding: 1px 5px; border: 0; background: transparent; color: #ffe29a; }
.notes { width: 100%; margin-top: 12px; padding: 10px; background: #090d15; color: #eef2f8; border: 1px solid #333d51; border-radius: 7px; }
.actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 12px; }
.done { color: #63e6be; }
@media (max-width: 900px) { .screens { grid-template-columns: 1fr; } .edge-layer { display: none; } .meta { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header><strong>__TITLE__</strong><span id="progress"></span><button id="prev">上一条</button><button id="next">下一条</button><button id="download">导出 JSONL</button><label>标注人 <input id="annotator" size="10"></label></header>
<main>
  <div class="meta"><div><span id="kind"></span> · <span class="reason" id="reasons"></span></div><div class="score" id="score"></div></div>
  <div class="screens" id="screens"><section class="screen" id="source"></section><section class="screen" id="target"></section><svg class="edge-layer" id="edgeLayer"></svg></div>
  <div class="relation-bar"><button class="primary" id="connect">建立所选节点对应</button><div class="relations" id="relations"></div></div>
  <input class="notes" id="notes" placeholder="可选备注：节点为什么对应、为什么无对应，或截图有什么问题">
  <div class="actions">
    <button class="primary" data-label="correspondence">1 · 完成节点对应</button>
    <button data-label="no_correspondence">2 · 目标屏无对应</button>
    <button data-label="uncertain">3 · 不确定</button>
    <button data-label="unusable_screen">4 · 截图不可用</button>
    <button id="clear">清除本条</button>
  </div>
</main>
<script>
function upgradeItem(raw) {
  const item = JSON.parse(JSON.stringify(raw));
  const sequence = Boolean(item.selection?.trajectory_pair_id);
  item.schema_version = sequence ? 'omnitransfer_guiodyssey_sequence_node_alignment_v1' : 'omnitransfer_guiodyssey_node_review_v1';
  for (const side of ['source', 'target']) {
    delete item[side].bbox;
    delete item[side].bbox_normalized;
  }
  delete item.selection.bbox_quality;
  delete item.selection.bbox_scale_ratio;
  item.selection.reasons = (item.selection.reasons || []).filter(reason => !reason.includes('bbox'));
  const old = item.annotation || {};
  item.annotation = {
    status: old.status || 'unreviewed',
    label: old.label || null,
    source_nodes: old.source_nodes || [{node_id: 'source-1', point_normalized: item.source.point_normalized, label: item.source.instruction, provenance: 'guiodyssey_action_point'}],
    target_nodes: old.target_nodes || [{node_id: 'target-1', point_normalized: item.target.point_normalized, label: item.target.instruction, provenance: 'guiodyssey_action_point'}],
    matches: old.matches || [],
    notes: old.notes || '',
    annotator: old.annotator || ''
  };
  return item;
}
const queue = __PAYLOAD__.map(upgradeItem);
const reviewNamespace = queue[0]?.selection?.review_namespace || '';
const storageKey = 'omnitransfer-guiodyssey-node-review-v1' + (reviewNamespace ? ':' + reviewNamespace : '');
const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
const selected = {};
let index = 0;
let dragState = null;
const byId = id => document.getElementById(id);
function persist() { localStorage.setItem(storageKey, JSON.stringify(saved)); }
function annotationFor(item) { return JSON.parse(JSON.stringify({...item.annotation, ...(saved[item.pair_id] || {})})); }
function selectionFor(item, annotation) {
  const current = selected[item.pair_id] || {};
  const sourceExists = annotation.source_nodes.some(node => node.node_id === current.source);
  const targetExists = annotation.target_nodes.some(node => node.node_id === current.target);
  selected[item.pair_id] = {
    source: sourceExists ? current.source : annotation.source_nodes[0]?.node_id || null,
    target: targetExists ? current.target : annotation.target_nodes[0]?.node_id || null
  };
  return selected[item.pair_id];
}
function nodeName(nodeId) { return nodeId.replace('source-', 'S').replace('target-', 'T'); }
function panel(side, row, annotation, current) {
  const nodes = annotation[side + '_nodes'];
  const nodeMarkup = nodes.map(node => {
    const point = node.point_normalized;
    const active = current[side] === node.node_id ? ' selected' : '';
    return `<button class="node${active}" data-node-id="${escapeHtml(node.node_id)}" aria-label="${side} node ${escapeHtml(nodeName(node.node_id))}" title="${escapeHtml(node.label || node.node_id)}" style="left:${point[0] / 10}%;top:${point[1] / 10}%">${escapeHtml(nodeName(node.node_id))}</button>`;
  }).join('');
  const selectedNode = nodes.find(node => node.node_id === current[side]);
  const editor = selectedNode
    ? `<div class="node-editor"><strong>${escapeHtml(nodeName(selectedNode.node_id))}</strong><input data-node-label="${side}" value="${escapeAttribute(selectedNode.label || '')}" aria-label="${side} node label"><button class="danger" data-delete-node="${side}">删除节点</button></div>`
    : '<div class="node-editor"><span class="hint">点击截图新增节点；拖动节点可调整位置</span></div>';
  const context = row.previous_instruction || row.next_instruction ? `<div class="small"><strong>前：</strong>${escapeHtml(row.previous_instruction || '—')}<br><strong>当前：</strong>${escapeHtml(row.instruction)}<br><strong>后：</strong>${escapeHtml(row.next_instruction || '—')}</div>` : '';
  return `<h3>${side === 'source' ? '源端节点' : '目标节点'} · 点击新增 / 拖动调整 / 点击选中</h3><div class="image-wrap" data-side="${side}"><img src="${row.image_url}" alt="${row.screenshot}">${nodeMarkup}</div>${editor}<div class="instruction">${escapeHtml(row.instruction)}</div>${context}<div class="small">${escapeHtml(row.device_name)} · ${row.width}×${row.height} · ${escapeHtml(row.apps.join(', '))}<br>${escapeHtml(row.description)}</div>`;
}
function render() {
  if (!queue.length) { byId('progress').textContent = '没有候选'; return; }
  const item = queue[index], annotation = annotationFor(item), current = selectionFor(item, annotation);
  const reviewedCount = queue.filter(row => saved[row.pair_id]?.status === 'reviewed').length;
  byId('progress').innerHTML = `${index + 1} / ${queue.length} · <span class="done">已标 ${reviewedCount}</span>`;
  byId('kind').textContent = item.selection.candidate_kind;
  byId('reasons').textContent = item.selection.reasons.join(' · ');
  const alignment = item.selection.trajectory_pair_id ? ` · trajectory ${item.selection.episode_alignment_score.toFixed(3)} · step ${item.selection.source_sequence_position}→${item.selection.target_sequence_position}` : '';
  byId('score').textContent = `priority ${item.selection.priority_score.toFixed(3)} · semantic ${item.selection.semantic_score.toFixed(3)}${alignment}`;
  byId('source').innerHTML = panel('source', item.source, annotation, current);
  byId('target').innerHTML = panel('target', item.target, annotation, current);
  byId('relations').innerHTML = annotation.matches.length
    ? annotation.matches.map((match, matchIndex) => `<span class="relation">${escapeHtml(nodeName(match.source_node_id))} ↔ ${escapeHtml(nodeName(match.target_node_id))}<button data-remove-match="${matchIndex}" aria-label="删除对应">×</button></span>`).join('')
    : '<span class="hint">尚未建立对应；允许一对多、多对一和多对多</span>';
  byId('notes').value = annotation.notes || '';
  document.querySelectorAll('[data-label]').forEach(button => button.style.outline = button.dataset.label === annotation.label ? '3px solid #63e6be' : 'none');
  bindWorkspace();
  requestAnimationFrame(drawEdges);
}
function bindWorkspace() {
  document.querySelectorAll('.image-wrap').forEach(wrap => wrap.onclick = event => {
    if (event.target.closest('.node')) return;
    addNode(wrap.dataset.side, pointFromEvent(wrap, event));
  });
  document.querySelectorAll('.node').forEach(node => {
    node.onclick = event => { event.stopPropagation(); selectNode(node.closest('.image-wrap').dataset.side, node.dataset.nodeId); };
    node.onpointerdown = event => beginDrag(node, event);
  });
  document.querySelectorAll('[data-node-label]').forEach(input => input.onchange = () => updateNodeLabel(input.dataset.nodeLabel, input.value));
  document.querySelectorAll('[data-delete-node]').forEach(button => button.onclick = () => deleteSelectedNode(button.dataset.deleteNode));
  document.querySelectorAll('[data-remove-match]').forEach(button => button.onclick = () => removeMatch(Number(button.dataset.removeMatch)));
}
function pointFromEvent(wrap, event) {
  const rect = wrap.getBoundingClientRect();
  return [Math.max(0, Math.min(1000, (event.clientX - rect.left) / rect.width * 1000)), Math.max(0, Math.min(1000, (event.clientY - rect.top) / rect.height * 1000))];
}
function saveAnnotation(item, annotation) { saved[item.pair_id] = annotation; persist(); }
function nextNodeId(side, nodes) {
  let number = 1;
  const used = new Set(nodes.map(node => node.node_id));
  while (used.has(`${side}-${number}`)) number++;
  return `${side}-${number}`;
}
function addNode(side, point) {
  const item = queue[index], annotation = annotationFor(item), key = side + '_nodes';
  const nodeId = nextNodeId(side, annotation[key]);
  annotation[key].push({node_id: nodeId, point_normalized: point, label: '手工节点', provenance: 'human_click'});
  annotation.status = annotation.status === 'reviewed' ? 'reviewed' : 'draft';
  selected[item.pair_id] = {...selectionFor(item, annotation), [side]: nodeId};
  saveAnnotation(item, annotation);
  render();
}
function selectNode(side, nodeId) {
  const item = queue[index], annotation = annotationFor(item);
  selected[item.pair_id] = {...selectionFor(item, annotation), [side]: nodeId};
  render();
}
function beginDrag(node, event) {
  event.preventDefault();
  event.stopPropagation();
  const item = queue[index], annotation = annotationFor(item), wrap = node.closest('.image-wrap');
  dragState = {side: wrap.dataset.side, nodeId: node.dataset.nodeId, wrap};
  selected[item.pair_id] = {...selectionFor(item, annotation), [dragState.side]: dragState.nodeId};
  node.setPointerCapture?.(event.pointerId);
}
document.onpointermove = event => {
  if (!dragState || !queue[index]) return;
  const item = queue[index], annotation = annotationFor(item);
  const node = annotation[dragState.side + '_nodes'].find(candidate => candidate.node_id === dragState.nodeId);
  if (!node) return;
  node.point_normalized = pointFromEvent(dragState.wrap, event);
  annotation.status = annotation.status === 'reviewed' ? 'reviewed' : 'draft';
  saveAnnotation(item, annotation);
  const element = document.querySelector(`.image-wrap[data-side="${dragState.side}"] .node[data-node-id="${dragState.nodeId}"]`);
  if (element) { element.style.left = `${node.point_normalized[0] / 10}%`; element.style.top = `${node.point_normalized[1] / 10}%`; }
  drawEdges();
};
document.onpointerup = () => { dragState = null; };
function updateNodeLabel(side, value) {
  const item = queue[index], annotation = annotationFor(item), current = selectionFor(item, annotation);
  const node = annotation[side + '_nodes'].find(candidate => candidate.node_id === current[side]);
  if (!node) return;
  node.label = value.trim() || node.node_id;
  annotation.status = annotation.status === 'reviewed' ? 'reviewed' : 'draft';
  saveAnnotation(item, annotation);
  render();
}
function deleteSelectedNode(side) {
  const item = queue[index], annotation = annotationFor(item), current = selectionFor(item, annotation), nodeId = current[side];
  if (!nodeId) return;
  annotation[side + '_nodes'] = annotation[side + '_nodes'].filter(node => node.node_id !== nodeId);
  annotation.matches = annotation.matches.filter(match => match[side + '_node_id'] !== nodeId);
  current[side] = annotation[side + '_nodes'][0]?.node_id || null;
  annotation.status = 'draft';
  annotation.label = null;
  saveAnnotation(item, annotation);
  render();
}
function connectSelected() {
  const item = queue[index], annotation = annotationFor(item), current = selectionFor(item, annotation);
  if (!current.source || !current.target) return;
  const exists = annotation.matches.some(match => match.source_node_id === current.source && match.target_node_id === current.target);
  if (!exists) annotation.matches.push({source_node_id: current.source, target_node_id: current.target});
  annotation.status = 'draft';
  annotation.label = null;
  saveAnnotation(item, annotation);
  render();
}
function removeMatch(matchIndex) {
  const item = queue[index], annotation = annotationFor(item);
  annotation.matches.splice(matchIndex, 1);
  annotation.status = 'draft';
  annotation.label = null;
  saveAnnotation(item, annotation);
  render();
}
function drawEdges() {
  const item = queue[index];
  if (!item) return;
  const annotation = annotationFor(item), screens = byId('screens'), layer = byId('edgeLayer'), base = screens.getBoundingClientRect();
  layer.setAttribute('viewBox', `0 0 ${base.width} ${base.height}`);
  layer.innerHTML = annotation.matches.map(match => {
    const sourceNode = document.querySelector(`.image-wrap[data-side="source"] .node[data-node-id="${match.source_node_id}"]`);
    const targetNode = document.querySelector(`.image-wrap[data-side="target"] .node[data-node-id="${match.target_node_id}"]`);
    if (!sourceNode || !targetNode) return '';
    const sourceBox = sourceNode.getBoundingClientRect(), targetBox = targetNode.getBoundingClientRect();
    return `<line x1="${sourceBox.left + sourceBox.width / 2 - base.left}" y1="${sourceBox.top + sourceBox.height / 2 - base.top}" x2="${targetBox.left + targetBox.width / 2 - base.left}" y2="${targetBox.top + targetBox.height / 2 - base.top}"></line>`;
  }).join('');
}
function label(value) {
  const item = queue[index], annotation = annotationFor(item);
  if (value === 'correspondence' && annotation.matches.length === 0) { alert('请先选择左右节点并建立至少一条对应。'); return; }
  if (value === 'no_correspondence' && (!annotation.source_nodes.length || !annotation.target_nodes.length)) { alert('请先在左右页面各标至少一个节点；目标节点作为 NULL 拒识的干扰候选。'); return; }
  if (value === 'no_correspondence') annotation.matches = [];
  annotation.status = 'reviewed';
  annotation.label = value;
  annotation.notes = byId('notes').value;
  annotation.annotator = byId('annotator').value;
  annotation.reviewed_at = new Date().toISOString();
  saveAnnotation(item, annotation);
  if (index < queue.length - 1) index++;
  render();
}
function exportRows() {
  const rows = queue.filter(item => saved[item.pair_id]?.status === 'reviewed').map(item => ({...item, annotation: annotationFor(item)}));
  const blob = new Blob([rows.map(row => JSON.stringify(row)).join('\\n') + '\\n'], {type: 'application/jsonl'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'guiodyssey_node_reviewed.jsonl';
  link.click();
  URL.revokeObjectURL(link.href);
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])); }
function escapeAttribute(value) { return escapeHtml(value).replace(/`/g, '&#96;'); }
document.querySelectorAll('[data-label]').forEach(button => button.onclick = () => label(button.dataset.label));
byId('connect').onclick = connectSelected;
byId('prev').onclick = () => { index = Math.max(0, index - 1); render(); };
byId('next').onclick = () => { index = Math.min(queue.length - 1, index + 1); render(); };
byId('download').onclick = exportRows;
byId('clear').onclick = () => {
  if (!queue[index]) return;
  delete saved[queue[index].pair_id];
  delete selected[queue[index].pair_id];
  persist();
  render();
};
byId('notes').onchange = () => { if (!queue[index]) return; const annotation = annotationFor(queue[index]); annotation.notes = byId('notes').value; annotation.status = annotation.status === 'reviewed' ? 'reviewed' : 'draft'; saveAnnotation(queue[index], annotation); };
window.onresize = drawEdges;
document.onkeydown = event => {
  if (event.target.tagName === 'INPUT') return;
  const labels = {1: 'correspondence', 2: 'no_correspondence', 3: 'uncertain', 4: 'unusable_screen'};
  if (labels[event.key]) document.querySelector(`[data-label="${labels[event.key]}"]`)?.click();
};
try { render(); } catch (error) {
  document.querySelector('main').innerHTML = `<section class="screen"><h2>页面初始化失败</h2><pre>${escapeHtml(error.stack || error)}</pre></section>`;
  console.error(error);
}
</script>
</body>
</html>"""
    return template.replace("__TITLE__", title).replace("__PAYLOAD__", payload)


if __name__ == "__main__":
    main()
