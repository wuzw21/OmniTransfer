"""Static reviewer for unified mapping page-pair records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from PIL import Image

from omnitransfer.mapping_dataset import validate_mapping_page_pair


def build_mapping_pair_review(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    pair_limit: int = 0,
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
                records.append(validate_mapping_page_pair(json.loads(line)))
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
    review_path = output / "review.html"
    review_path.write_text(_review_html(queue), encoding="utf-8")
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
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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
    return {
        "pair_id": record["pair_id"],
        "dataset": dataset,
        "label_status": record["label_status"],
        "slices": record["slices"],
        "provenance": record["provenance"],
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
    width = float(graph.get("width") or 0)
    height = float(graph.get("height") or 0)
    if width <= 0 or height <= 0:
        with Image.open(source) as image:
            width, height = image.size
    return {
        "page_id": page["page_id"],
        "platform": page["platform"],
        "image_url": copied[source],
        "width": width,
        "height": height,
        "nodes": [_review_node(node) for node in nodes],
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
        "clickable": bool(node.get("clickable")),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _review_html(queue: list[dict[str, Any]]) -> str:
    payload = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>OmniTransfer · Mapping Pair Pilot</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #090c13; color: #edf2f8; }}
header {{ position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap; align-items: center; gap: 9px; padding: 10px 15px; border-bottom: 1px solid #293145; background: #0d111bdd; backdrop-filter: blur(14px); }}
button, select {{ padding: 7px 10px; border: 1px solid #39445d; border-radius: 7px; background: #161c29; color: inherit; font: inherit; cursor: pointer; }}
button:hover {{ border-color: #6f81a5; }}
main {{ max-width: 1780px; margin: auto; padding: 14px; }}
.meta {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; margin-bottom: 10px; color: #a8b5cc; font-size: 13px; }}
.screens {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
.screen {{ min-width: 0; padding: 11px; border: 1px solid #273044; border-radius: 10px; background: #0f141f; }}
.screen h3 {{ margin: 0 0 9px; font-size: 14px; color: #bac6da; }}
.image-wrap {{ position: relative; overflow: hidden; min-height: 220px; background: #03050a; }}
.image-wrap img {{ display: block; width: 100%; height: auto; }}
.box {{ position: absolute; z-index: 3; min-width: 5px; min-height: 5px; border: 3px solid #63e6be; background: #63e6be22; box-shadow: 0 0 0 1px #06110e, 0 2px 7px #000b; pointer-events: none; }}
.box.target {{ border-color: #ffd166; background: #ffd16622; }}
.box.context {{ opacity: .22; border-width: 2px; }}
.box span {{ position: absolute; top: -22px; left: -2px; padding: 2px 5px; border-radius: 4px; background: #0a1113e8; color: #a9ffe4; font-size: 11px; white-space: nowrap; }}
.box.target span {{ color: #ffe6a8; }}
.details {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, .7fr); gap: 13px; margin-top: 13px; }}
.node-card, .match-list {{ padding: 12px; border: 1px solid #273044; border-radius: 9px; background: #0f141f; }}
.node-card h4 {{ margin: 0 0 8px; }}
.node-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.node-info {{ min-width: 0; color: #bac6da; font-size: 13px; line-height: 1.55; overflow-wrap: anywhere; }}
.match-list {{ max-height: 330px; overflow: auto; }}
.match-row {{ width: 100%; margin-bottom: 6px; text-align: left; color: #b8c4d8; }}
.match-row.active {{ border-color: #63e6be; background: #193126; color: #eafff8; }}
.null {{ color: #ff9f9f; }}
.tag {{ display: inline-block; margin-right: 5px; padding: 2px 6px; border-radius: 999px; background: #202a3d; color: #aebfe0; font-size: 11px; }}
@media (max-width: 920px) {{ .screens, .details, .node-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header><strong>OmniTransfer · Mapping Pair Pilot</strong><span id="progress"></span><button id="prevPair">上一组</button><button id="nextPair">下一组</button><button id="prevMatch">上一个节点</button><button id="nextMatch">下一个节点</button><label><input type="checkbox" id="showAll"> 显示全部框</label><select id="dataset"></select></header>
<main><div class="meta"><div id="identity"></div><div id="stats"></div></div><div class="screens"><section class="screen" id="source"></section><section class="screen" id="target"></section></div><div class="details"><section class="node-card" id="nodeCard"></section><section class="match-list" id="matchList"></section></div></main>
<script>
const queue = {payload};
let filtered = queue.slice(), pairIndex = 0, matchIndex = 0;
const byId = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
function label(node) {{ return node?.text || node?.content_desc || node?.resource_id || node?.class_name || node?.node_id || '—'; }}
function nodeMap(side) {{ return Object.fromEntries(side.nodes.map(node => [node.node_id, node])); }}
function box(node, side, page, context=false) {{
  if (!node?.bbox) return '';
  const [x1,y1,x2,y2] = node.bbox;
  const left = Math.max(0, Math.min(100, x1/page.width*100)), top = Math.max(0, Math.min(100, y1/page.height*100));
  const width = Math.max(.3, Math.min(100-left, (x2-x1)/page.width*100)), height = Math.max(.3, Math.min(100-top, (y2-y1)/page.height*100));
  return `<div class="box ${{side}}${{context?' context':''}}" style="left:${{left}}%;top:${{top}}%;width:${{width}}%;height:${{height}}%"><span>${{side==='source'?'S':'T'}} · ${{esc(label(node).slice(0,42))}}</span></div>`;
}}
function panel(sideName, item, match) {{
  const page = item[sideName], nodes = nodeMap(page), activeIds = sideName === 'source' ? [match.source_node_id] : match.target_node_ids;
  let overlays = '';
  if (byId('showAll').checked) overlays += page.nodes.filter(node => !activeIds.includes(node.node_id)).map(node => box(node, sideName, page, true)).join('');
  overlays += activeIds.map(id => box(nodes[id], sideName, page)).join('');
  return `<h3>${{sideName === 'source' ? 'Source' : 'Target'}} · ${{esc(page.page_id)}} · ${{page.width}}×${{page.height}}</h3><div class="image-wrap"><img src="${{esc(page.image_url)}}" alt="${{esc(page.page_id)}}">${{overlays}}</div>`;
}}
function info(node) {{
  if (!node) return '<span class="null">目标视口无对应节点（NULL）</span>';
  return `<strong>${{esc(label(node))}}</strong><br><span class="tag">${{esc(node.class_name)}}</span>${{node.clickable?'<span class="tag">clickable</span>':''}}<br>node: ${{esc(node.node_id)}}<br>origin(label-only): ${{esc(node.origin_id)}}<br>bbox: ${{esc(JSON.stringify(node.bbox))}}`;
}}
function render() {{
  if (!filtered.length) return;
  const item = filtered[pairIndex], match = item.matches[matchIndex], sourceNodes = nodeMap(item.source), targetNodes = nodeMap(item.target);
  byId('progress').textContent = `${{pairIndex+1}} / ${{filtered.length}} · 节点 ${{matchIndex+1}} / ${{item.matches.length}}`;
  byId('identity').innerHTML = `<span class="tag">${{esc(item.dataset)}}</span><span class="tag">${{esc(item.label_status)}}</span>${{esc(item.pair_id)}}`;
  const nulls = item.matches.filter(value => value.label === 'no_correspondence').length;
  byId('stats').textContent = `${{item.matches.length}} anchors · ${{nulls}} NULL`;
  byId('source').innerHTML = panel('source', item, match);
  byId('target').innerHTML = panel('target', item, match);
  const sourceNode = sourceNodes[match.source_node_id], targets = match.target_node_ids.map(id => targetNodes[id]);
  byId('nodeCard').innerHTML = `<h4>${{esc(match.label)}}</h4><div class="node-grid"><div class="node-info"><b>Source</b><br>${{info(sourceNode)}}</div><div class="node-info"><b>Target set (${{targets.length}})</b><br>${{targets.length ? targets.map(info).join('<hr>') : info(null)}}</div></div>`;
  byId('matchList').innerHTML = item.matches.map((value, index) => `<button class="match-row${{index===matchIndex?' active':''}}" data-index="${{index}}">${{index+1}} · ${{esc(label(sourceNodes[value.source_node_id]).slice(0,52))}} → ${{value.target_node_ids.length || 'NULL'}}</button>`).join('');
  document.querySelectorAll('.match-row').forEach(button => button.onclick = () => {{ matchIndex = Number(button.dataset.index); render(); }});
}}
function movePair(delta) {{ pairIndex = (pairIndex + delta + filtered.length) % filtered.length; matchIndex = 0; render(); }}
function moveMatch(delta) {{ const count = filtered[pairIndex].matches.length; matchIndex = (matchIndex + delta + count) % count; render(); }}
const datasets = ['全部', ...new Set(queue.map(item => item.dataset))];
byId('dataset').innerHTML = datasets.map(value => `<option>${{esc(value)}}</option>`).join('');
byId('dataset').onchange = () => {{ filtered = byId('dataset').value === '全部' ? queue.slice() : queue.filter(item => item.dataset === byId('dataset').value); pairIndex = 0; matchIndex = 0; render(); }};
byId('prevPair').onclick = () => movePair(-1); byId('nextPair').onclick = () => movePair(1);
byId('prevMatch').onclick = () => moveMatch(-1); byId('nextMatch').onclick = () => moveMatch(1);
byId('showAll').onchange = render;
document.onkeydown = event => {{ if (event.key === 'ArrowLeft') movePair(-1); if (event.key === 'ArrowRight') movePair(1); if (event.key === 'ArrowUp') moveMatch(-1); if (event.key === 'ArrowDown') moveMatch(1); }};
render();
</script>
</body>
</html>"""
