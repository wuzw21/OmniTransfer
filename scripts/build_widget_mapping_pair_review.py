#!/usr/bin/env python3
"""Build an image-led reviewer for multi-anchor source-target screen pairs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from omnitransfer.importers import load_queries
from omnitransfer.widget_mapping_review import build_widget_mapping_pair_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group widget queries by screen pair and build a human review UI."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_widget_mapping_pair_review(
        load_queries(input_path),
        input_path=input_path,
        output_dir=output,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
        manifest = {
            **manifest,
            "screen_pairs": len(rows),
            "correspondences": sum(len(row["correspondences"]) for row in rows),
            "limit": args.limit,
        }
    queue_path = output / "screen_pairs.jsonl"
    queue_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "review.html").write_text(_review_html(rows), encoding="utf-8")
    manifest = {
        **manifest,
        "queue_file": queue_path.name,
        "review_file": "review.html",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def _review_html(rows: list[dict], *, title: str = "OmniTransfer Pair Audit") -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0a0b0d;
  --surface: #111317;
  --surface-2: #171a20;
  --line: #2a2e36;
  --muted: #8d949f;
  --text: #f0f2f5;
  --accent: #ffb454;
  --good: #6fe0b2;
  --bad: #ff6f78;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); min-width: 980px; }}
button, input, select, textarea {{ font: inherit; }}
button {{ color: var(--text); background: transparent; border: 1px solid var(--line); border-radius: 7px; padding: 8px 11px; cursor: pointer; }}
button:hover {{ border-color: #555c68; background: #1c2027; }}
button.primary {{ border-color: var(--accent); color: #16110a; background: var(--accent); font-weight: 700; }}
header {{
  height: 62px; position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 10px;
  padding: 0 18px; background: rgba(10,11,13,.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(16px);
}}
.brand {{ font-size: 17px; font-weight: 780; letter-spacing: -.02em; margin-right: 8px; white-space: nowrap; }}
.brand span {{ color: var(--accent); }}
.progress {{ color: var(--muted); font-variant-numeric: tabular-nums; margin-right: auto; white-space: nowrap; }}
.control {{ color: var(--muted); background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 8px 10px; }}
#jump {{ width: 72px; }}
.protocol-warning {{ display: none; padding: 9px 18px; color: #17100a; background: var(--accent); font-weight: 650; }}
.workspace {{ padding: 14px 18px 32px; animation: enter .28s ease both; }}
.pair-head {{ display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: end; padding: 8px 0 13px; border-bottom: 1px solid var(--line); }}
.eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 760; letter-spacing: .12em; text-transform: uppercase; }}
h1 {{ margin: 4px 0 0; font-size: 23px; line-height: 1.15; letter-spacing: -.03em; }}
.pair-meta {{ color: var(--muted); font-size: 13px; line-height: 1.5; text-align: right; }}
.stage {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin-top: 14px; background: var(--line); border: 1px solid var(--line); }}
.screen {{ min-width: 0; background: var(--surface); }}
.screen-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
.screen-title {{ font-weight: 720; }}
.screen-detail {{ color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.image-shell {{ position: relative; min-height: 520px; height: calc(100vh - 260px); display: grid; place-items: center; overflow: hidden; background: #050607; }}
.image-shell img {{ display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; }}
.overlay {{ position: absolute; pointer-events: none; }}
.overlay rect {{ vector-effect: non-scaling-stroke; transition: opacity .12s ease, stroke-width .12s ease, filter .12s ease; }}
.overlay .candidate {{ fill: transparent; stroke: #d9dee733; stroke-width: 1; pointer-events: auto; cursor: crosshair; }}
.overlay .mapping {{ fill-opacity: .08; stroke-width: 2.5; pointer-events: auto; cursor: pointer; }}
.overlay .mapping-label {{ font: 700 18px ui-sans-serif, sans-serif; paint-order: stroke; stroke: #050607; stroke-width: 5px; }}
.overlay.dim .mapping:not(.active) {{ opacity: .15; }}
.overlay .mapping.active {{ stroke-width: 4; filter: drop-shadow(0 0 5px currentColor); }}
.image-error {{ display: none; position: absolute; inset: 24px; align-content: center; text-align: center; color: var(--bad); border: 1px dashed var(--bad); padding: 20px; background: #12090b; }}
.image-error code {{ display: block; color: #f0a3a8; overflow-wrap: anywhere; margin: 10px 0; }}
.review {{ display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 18px; margin-top: 18px; }}
.mappings {{ border-top: 1px solid var(--line); }}
.mapping-row {{ display: grid; grid-template-columns: 40px minmax(180px, 1fr) 40px minmax(180px, 1fr) auto; gap: 10px; align-items: center; padding: 10px 8px; border-bottom: 1px solid var(--line); cursor: pointer; transition: background .13s ease, transform .13s ease; }}
.mapping-row:hover, .mapping-row.active {{ background: var(--surface-2); transform: translateX(2px); }}
.mapping-number {{ width: 27px; height: 27px; display: grid; place-items: center; border-radius: 50%; color: #0b0c0e; font-size: 12px; font-weight: 800; }}
.mapping-arrow {{ color: var(--muted); text-align: center; }}
.mapping-copy {{ min-width: 0; }}
.mapping-copy strong {{ display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mapping-copy span {{ display: block; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mapping-status {{ display: flex; gap: 5px; }}
.mapping-status button {{ padding: 6px 8px; font-size: 12px; }}
.mapping-status button.selected {{ border-color: var(--good); color: var(--good); }}
.inspector {{ border-left: 1px solid var(--line); padding-left: 18px; }}
.inspector h2 {{ margin: 0 0 8px; font-size: 14px; }}
.stats {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 14px; padding: 10px 0 14px; color: var(--muted); font-size: 13px; border-bottom: 1px solid var(--line); }}
.stats b {{ color: var(--text); font-variant-numeric: tabular-nums; }}
.pair-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 14px 0; }}
.pair-actions button.selected {{ border-color: var(--accent); color: var(--accent); }}
textarea {{ width: 100%; min-height: 88px; resize: vertical; color: var(--text); background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 9px; }}
details {{ margin-top: 14px; color: var(--muted); font-size: 12px; }}
details summary {{ color: var(--text); cursor: pointer; font-weight: 680; }}
details pre {{ white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.5; }}
.hint {{ margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.55; }}
.empty {{ min-height: 70vh; display: grid; place-items: center; color: var(--muted); }}
@keyframes enter {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
</style>
</head>
<body>
<div class="protocol-warning" id="protocolWarning">当前使用 file:// 打开；页面可工作。若浏览器阻止图片，请在本目录运行 <code>python -m http.server 8765</code> 后访问 <code>http://127.0.0.1:8765/review.html</code>。</div>
<header>
  <div class="brand">OmniTransfer <span>Pair Audit</span></div>
  <div class="progress" id="progress">载入中</div>
  <button id="prev">← 上一个</button><button id="next">下一个 →</button>
  <input class="control" id="jump" type="number" min="1" aria-label="跳转序号">
  <select class="control" id="appFilter"><option value="">全部 App</option></select>
  <select class="control" id="statusFilter"><option value="">全部状态</option><option value="unreviewed">只看未审核</option><option value="approved">已通过</option><option value="needs_fix">需修正</option><option value="rejected">已拒绝</option></select>
  <input class="control" id="annotator" placeholder="标注人" size="8">
  <button class="primary" id="export">导出 JSONL</button>
</header>
<main class="workspace" id="workspace"></main>
<script>
const pairs={payload};
const palette=['#ffb454','#69d9ff','#f785ff','#78e08f','#ff7c8a','#a99cff','#e8dc74','#6fe0b2','#ff9f68','#78a8ff'];
const storageKey='omnitransfer-widget-pair-audit-v1';
const annotations=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
let viewIndices=[];
let viewPosition=0;
let activeMappingId=null;
const byId=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[char]);
const mappingAnnotation=(pair,id)=>annotations[pair.pair_id]?.mappings?.[id]||{{}};
function save(){{localStorage.setItem(storageKey,JSON.stringify(annotations));}}
function pairAnnotation(pair){{return annotations[pair.pair_id]||{{mappings:{{}},pair_status:null,notes:''}};}}
function ensurePair(pair){{if(!annotations[pair.pair_id])annotations[pair.pair_id]={{mappings:{{}},pair_status:null,notes:'',annotator:'',updated_at:null}};return annotations[pair.pair_id];}}
function currentPair(){{return pairs[viewIndices[viewPosition]];}}
function applyFilters(keepPair=true){{
  const previous=currentPair()?.pair_id;
  const app=byId('appFilter').value,status=byId('statusFilter').value;
  viewIndices=pairs.map((_,index)=>index).filter(index=>{{const pair=pairs[index],saved=annotations[pair.pair_id];if(app&&pair.app!==app)return false;if(status==='unreviewed'&&saved?.pair_status)return false;if(status&&status!=='unreviewed'&&saved?.pair_status!==status)return false;return true;}});
  viewPosition=keepPair?Math.max(0,viewIndices.findIndex(index=>pairs[index].pair_id===previous)):0;
  if(viewPosition<0)viewPosition=0;render();
}}
function screenPanel(pair,side){{
  const screen=pair[side],target=side==='target';
  return `<section class="screen"><div class="screen-head"><span class="screen-title">${{target?'目标布局':'源端页面'}}</span><span class="screen-detail">${{esc(screen.screen)}} · ${{screen.width||'?'}}×${{screen.height||'?'}}</span></div><div class="image-shell" id="${{side}}Shell"><img id="${{side}}Image" src="${{esc(screen.image_url)}}" alt="${{esc(screen.screen)}}"><svg class="overlay" id="${{side}}Overlay"></svg><div class="image-error" id="${{side}}Error"><strong>图片加载失败</strong><code>${{esc(screen.image_url)}}</code><button onclick="retryImage('${{side}}')">重试</button></div></div></section>`;
}}
function render(){{
  if(!viewIndices.length){{byId('workspace').innerHTML='<div class="empty">当前筛选没有 screen pair</div>';byId('progress').textContent=`0 / 0 · 总计 ${{pairs.length}}`;return;}}
  const pair=currentPair(),saved=pairAnnotation(pair);if(!activeMappingId||!pair.correspondences.some(row=>row.mapping_id===activeMappingId))activeMappingId=pair.correspondences[0]?.mapping_id||null;
  const reviewed=Object.values(annotations).filter(row=>row.pair_status).length;
  byId('progress').textContent=`${{viewPosition+1}} / ${{viewIndices.length}} · 已审 ${{reviewed}} / ${{pairs.length}} · 本页 ${{pair.correspondences.length}} 点`;
  byId('jump').value=viewPosition+1;
  byId('workspace').innerHTML=`<div class="pair-head"><div><div class="eyebrow">${{esc(pair.app)}} · ${{esc(pair.category||'Widget Mapping')}}</div><h1>${{esc(pair.source.screen)}} → ${{esc(pair.target.screen)}}</h1></div><div class="pair-meta">${{pair.pair_id}}<br>${{pair.quality.target_candidate_count}} candidates · ${{pair.quality.target_alias_groups}} alias groups</div></div><div class="stage">${{screenPanel(pair,'source')}}${{screenPanel(pair,'target')}}</div><div class="review"><section><div class="mappings" id="mappingList"></div></section><aside class="inspector"><h2>Pair 审核</h2><div class="stats"><span>原始 queries</span><b>${{pair.quality.query_count}}</b><span>对应点</span><b>${{pair.correspondences.length}}</b><span>候选控件</span><b>${{pair.quality.target_candidate_count}}</b><span>重复框组</span><b>${{pair.quality.target_alias_groups}}</b><span>source node_match</span><b>${{esc(pair.quality.source_node_match.join(', '))}}</b></div><div class="pair-actions"><button data-pair-status="approved">A · 整页正确</button><button data-pair-status="needs_fix">F · 需要修正</button><button data-pair-status="rejected">X · 整页拒绝</button><button id="clearPair">清除</button></div><textarea id="pairNotes" placeholder="记录页面错位、错误 gold、候选缺失等">${{esc(saved.notes||'')}}</textarea><details><summary>这组 pair 如何构建</summary><pre>同一 source_screen + target_screen 的 per-anchor query 被合并。\n每个彩色编号是一条 source widget → target widget 对应。\n灰框是目标 XML 候选；选中一条 mapping 后点击灰框可替换其 target。\n多个 candidate ID 若 bbox 相同，会作为同一可视目标一起选择。</pre></details><div class="hint">快捷键：J/K 下一页/上一页；A/F/X 设置整页状态；1/2/3/4 设置当前 mapping 正确/错误/不确定/删除。</div></aside></div>`;
  wireImages(pair);renderMappingList(pair);wireReview(pair);
}}
function wireImages(pair){{
  ['source','target'].forEach(side=>{{const image=byId(side+'Image');image.onload=()=>drawOverlay(pair,side);image.onerror=()=>{{byId(side+'Error').style.display='grid';}};if(image.complete&&image.naturalWidth)drawOverlay(pair,side);}});
}}
function overlayGeometry(side){{const image=byId(side+'Image'),shell=byId(side+'Shell'),rect=image.getBoundingClientRect(),shellRect=shell.getBoundingClientRect(),svg=byId(side+'Overlay');svg.style.left=(rect.left-shellRect.left)+'px';svg.style.top=(rect.top-shellRect.top)+'px';svg.style.width=rect.width+'px';svg.style.height=rect.height+'px';return svg;}}
function candidateGroups(pair){{const groups=new Map();pair.target_candidates.forEach(candidate=>{{if(!candidate.bbox)return;const key=candidate.bbox.join(',');if(!groups.has(key))groups.set(key,{{bbox:candidate.bbox,ids:[],labels:[]}});const group=groups.get(key);group.ids.push(candidate.candidate_id);group.labels.push(candidate.text||candidate.content_desc||candidate.resource_id||candidate.class_name||candidate.candidate_id);}});return [...groups.values()];}}
function drawOverlay(pair,side){{
  const screen=pair[side],svg=overlayGeometry(side);svg.setAttribute('viewBox',`0 0 ${{screen.width}} ${{screen.height}}`);const saved=pairAnnotation(pair);
  let markup='';
  if(side==='target')candidateGroups(pair).forEach(group=>{{const [x1,y1,x2,y2]=group.bbox;markup+=`<rect class="candidate" x="${{x1}}" y="${{y1}}" width="${{x2-x1}}" height="${{y2-y1}}" data-candidates="${{esc(group.ids.join('|'))}}"><title>${{esc(group.labels.join(' / '))}}</title></rect>`;}});
  pair.correspondences.forEach((mapping,index)=>{{const annotation=mappingAnnotation(pair,mapping.mapping_id),color=palette[index%palette.length],active=mapping.mapping_id===activeMappingId?' active':'',boxes=side==='source'?[mapping.source.bbox]:selectedBoxes(pair,mapping,annotation);boxes.filter(Boolean).forEach((box,boxIndex)=>{{const [x1,y1,x2,y2]=box;markup+=`<rect class="mapping${{active}}" data-mapping="${{esc(mapping.mapping_id)}}" x="${{x1}}" y="${{y1}}" width="${{x2-x1}}" height="${{y2-y1}}" stroke="${{color}}" fill="${{color}}"><title>#${{index+1}} ${{esc(mapping.source.text||mapping.source.content_desc||mapping.mapping_id)}}</title></rect>`;if(side==='source'&&boxIndex===0)markup+=`<text class="mapping-label" x="${{x1+5}}" y="${{Math.max(18,y1+18)}}" fill="${{color}}">${{index+1}}</text>`;}});}});
  svg.innerHTML=markup;
  svg.querySelectorAll('.mapping').forEach(node=>{{node.onmouseenter=()=>focusMapping(pair,node.dataset.mapping);node.onclick=()=>focusMapping(pair,node.dataset.mapping);}});
  if(side==='target')svg.querySelectorAll('.candidate').forEach(node=>node.onclick=event=>{{event.stopPropagation();toggleCandidates(pair,node.dataset.candidates.split('|'));}});
}}
function selectedIds(mapping,annotation){{return annotation.corrected_candidate_ids??mapping.gold_candidate_ids;}}
function selectedBoxes(pair,mapping,annotation){{const ids=new Set(selectedIds(mapping,annotation)),seen=new Set(),boxes=[];pair.target_candidates.forEach(candidate=>{{if(!ids.has(candidate.candidate_id)||!candidate.bbox)return;const key=candidate.bbox.join(',');if(!seen.has(key)){{seen.add(key);boxes.push(candidate.bbox);}}}});return boxes;}}
function focusMapping(pair,id){{activeMappingId=id;renderMappingList(pair);drawOverlay(pair,'source');drawOverlay(pair,'target');}}
function toggleCandidates(pair,ids){{if(!activeMappingId)return;const pairSaved=ensurePair(pair),mapping=pair.correspondences.find(row=>row.mapping_id===activeMappingId),current=new Set(selectedIds(mapping,pairSaved.mappings[activeMappingId]||{{}}));const removing=ids.every(id=>current.has(id));ids.forEach(id=>removing?current.delete(id):current.add(id));pairSaved.mappings[activeMappingId]={{...(pairSaved.mappings[activeMappingId]||{{}}),status:'corrected',corrected_candidate_ids:[...current],updated_at:new Date().toISOString()}};pairSaved.pair_status='needs_fix';pairSaved.updated_at=new Date().toISOString();save();render();}}
function mappingTargetLabel(pair,mapping,annotation){{const ids=selectedIds(mapping,annotation),labels=ids.map(id=>pair.target_candidates.find(row=>row.candidate_id===id)).filter(Boolean).map(row=>row.text||row.content_desc||row.resource_id||row.class_name||row.candidate_id);return labels.join(' / ')||'没有目标';}}
  function renderMappingList(pair){{const list=byId('mappingList'),saved=pairAnnotation(pair);list.innerHTML=pair.correspondences.map((mapping,index)=>{{const annotation=saved.mappings?.[mapping.mapping_id]||{{}},color=palette[index%palette.length],active=mapping.mapping_id===activeMappingId?' active':'',sourceLabel=mapping.source.text||mapping.source.content_desc||mapping.source.resource_id||mapping.source.class_name||'未命名控件';return `<div class="mapping-row${{active}}" data-mapping-row="${{esc(mapping.mapping_id)}}"><span class="mapping-number" style="background:${{color}}">${{index+1}}</span><span class="mapping-copy"><strong>${{esc(sourceLabel)}}</strong><span>${{esc(mapping.source.class_name)}} · ${{esc(mapping.source.widget_type)}}</span></span><span class="mapping-arrow">→</span><span class="mapping-copy"><strong>${{esc(mappingTargetLabel(pair,mapping,annotation))}}</strong><span>${{selectedIds(mapping,annotation).length}} candidate id(s)</span></span><span class="mapping-status">${{statusButton(mapping.mapping_id,'correct','1',annotation.status)}}${{statusButton(mapping.mapping_id,'wrong','2',annotation.status)}}${{statusButton(mapping.mapping_id,'uncertain','3',annotation.status)}}${{statusButton(mapping.mapping_id,'drop','4',annotation.status)}}</span></div>`;}}).join('');list.querySelectorAll('[data-mapping-row]').forEach(row=>row.onclick=()=>focusMapping(pair,row.dataset.mappingRow));list.querySelectorAll('[data-map-status]').forEach(button=>button.onclick=event=>{{event.stopPropagation();setMappingStatus(pair,button.dataset.mapping,button.dataset.mapStatus);}});}}
  function statusButton(mappingId,status,key,current){{const labels={{correct:'正确',wrong:'错误',uncertain:'不确定',drop:'删除'}};return `<button class="${{status===current?'selected':''}}" data-map-status="${{status}}" data-mapping="${{esc(mappingId)}}">${{key}}·${{labels[status]}}</button>`;}}
function setMappingStatus(pair,id,status){{const pairSaved=ensurePair(pair);pairSaved.mappings[id]={{...(pairSaved.mappings[id]||{{}}),status,updated_at:new Date().toISOString()}};if(status!=='correct')pairSaved.pair_status='needs_fix';pairSaved.annotator=byId('annotator').value;pairSaved.updated_at=new Date().toISOString();save();render();}}
function setPairStatus(pair,status){{const saved=ensurePair(pair);saved.pair_status=status;saved.annotator=byId('annotator').value;saved.notes=byId('pairNotes').value;saved.updated_at=new Date().toISOString();save();if(viewPosition<viewIndices.length-1)viewPosition++;render();}}
function wireReview(pair){{const saved=pairAnnotation(pair);document.querySelectorAll('[data-pair-status]').forEach(button=>{{button.classList.toggle('selected',button.dataset.pairStatus===saved.pair_status);button.onclick=()=>setPairStatus(pair,button.dataset.pairStatus);}});byId('clearPair').onclick=()=>{{delete annotations[pair.pair_id];save();render();}};byId('pairNotes').onchange=()=>{{const row=ensurePair(pair);row.notes=byId('pairNotes').value;row.updated_at=new Date().toISOString();save();}};}}
function retryImage(side){{const image=byId(side+'Image'),error=byId(side+'Error');error.style.display='none';const url=new URL(image.src,location.href);url.searchParams.set('retry',Date.now());image.src=url.href;}}
function move(delta){{if(!viewIndices.length)return;viewPosition=Math.max(0,Math.min(viewIndices.length-1,viewPosition+delta));activeMappingId=null;render();}}
function exportJsonl(){{const rows=pairs.filter(pair=>annotations[pair.pair_id]).map(pair=>({{schema_version:'omnitransfer_widget_mapping_pair_annotation_v1',pair_id:pair.pair_id,app:pair.app,source_screen:pair.source.screen,target_screen:pair.target.screen,annotation:annotations[pair.pair_id]}}));const blob=new Blob([rows.map(row=>JSON.stringify(row)).join('\\n')+'\\n'],{{type:'application/jsonl'}}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='widget_mapping_pair_annotations.jsonl';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}}
function populateApps(){{[...new Set(pairs.map(pair=>pair.app))].sort().forEach(app=>{{const option=document.createElement('option');option.value=app;option.textContent=app;byId('appFilter').append(option);}});}}
byId('prev').onclick=()=>move(-1);byId('next').onclick=()=>move(1);byId('export').onclick=exportJsonl;byId('jump').onchange=()=>{{viewPosition=Math.max(0,Math.min(viewIndices.length-1,Number(byId('jump').value)-1));activeMappingId=null;render();}};byId('appFilter').onchange=()=>applyFilters(false);byId('statusFilter').onchange=()=>applyFilters(false);window.onresize=()=>{{const pair=currentPair();if(pair){{drawOverlay(pair,'source');drawOverlay(pair,'target');}}}};
document.onkeydown=event=>{{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;const pair=currentPair();if(!pair)return;if(event.key.toLowerCase()==='j')move(1);else if(event.key.toLowerCase()==='k')move(-1);else if(event.key.toLowerCase()==='a')setPairStatus(pair,'approved');else if(event.key.toLowerCase()==='f')setPairStatus(pair,'needs_fix');else if(event.key.toLowerCase()==='x')setPairStatus(pair,'rejected');else if('1234'.includes(event.key)&&activeMappingId)setMappingStatus(pair,activeMappingId,{{'1':'correct','2':'wrong','3':'uncertain','4':'drop'}}[event.key]);}};
if(location.protocol==='file:')byId('protocolWarning').style.display='block';populateApps();applyFilters(false);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
