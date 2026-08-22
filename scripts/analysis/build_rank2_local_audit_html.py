#!/usr/bin/env python3
"""Build a compact visual audit for Rank-2 local-vs-node errors."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text(node: dict) -> str:
    return str(node.get("text") or node.get("content_desc") or "").strip()


def center(node: dict) -> tuple[float, float]:
    bbox = node.get("bbox") or [0, 0, 0, 0]
    return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2


def distance(left: dict, right: dict) -> float:
    return math.dist(center(left), center(right))


def parent(node_id: str) -> str:
    return str(node_id).rsplit(".", 1)[0] if "." in str(node_id) else ""


def relation(left: dict, right: dict) -> str:
    left_id, right_id = str(left["node_id"]), str(right["node_id"])
    if left_id.startswith(right_id + ".") or right_id.startswith(left_id + "."):
        return "ancestor"
    if parent(left_id) == parent(right_id):
        return "sibling"
    return "branch"


def same_bbox(left: dict, right: dict) -> bool:
    a, b = left.get("bbox"), right.get("bbox")
    return bool(a and b and max(abs(float(x) - float(y)) for x, y in zip(a, b)) < 0.012)


def category(row: dict) -> str:
    gold = row["gold_targets"][0]
    prediction = row["prediction"]
    if same_bbox(prediction, gold):
        return "same_bbox_endpoint"
    if not text(row["source"]) or not text(gold):
        return "textless_close" if distance(prediction, gold) < 0.1 else "textless_far"
    structure = relation(prediction, gold)
    if structure != "branch":
        return f"text_{structure}"
    return "text_close" if distance(prediction, gold) < 0.1 else "text_far"


def select_rows(rows: list[dict], limit: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[category(row)].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: (row["graph_pair"]["source"], row["source"]["node_id"]))
    selected, seen_apps = [], defaultdict(set)
    order = ["same_bbox_endpoint", "textless_close", "textless_far", "text_sibling", "text_ancestor", "text_close", "text_far"]
    while len(selected) < min(limit, len(rows)):
        progressed = False
        for name in order:
            values = buckets.get(name, [])
            if not values:
                continue
            index = next((i for i, row in enumerate(values) if row["graph_pair"]["source"].split("/")[1] not in seen_apps[name]), 0)
            row = values.pop(index)
            selected.append(row)
            seen_apps[name].add(row["graph_pair"]["source"].split("/")[1])
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def data_uri(path: str) -> str:
    source = Path(path)
    mime = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode()}"


def orient(record: dict, source_page: str, target_page: str) -> tuple[dict, dict, list[tuple[str, str]]]:
    if record["source"]["page_id"] == source_page:
        anchors = [(match["source_node_id"], target) for match in record["matches"] for target in match["target_node_ids"]]
        return record["source"], record["target"], anchors
    anchors = [(target, match["source_node_id"]) for match in record["matches"] for target in match["target_node_ids"]]
    return record["target"], record["source"], anchors


def box(node: dict, css_class: str, label: str) -> str:
    bbox = node.get("bbox")
    if not bbox:
        return ""
    x1, y1, x2, y2 = (100 * float(value) for value in bbox)
    return (
        f'<div class="box {css_class}" style="left:{x1:.3f}%;top:{y1:.3f}%;width:{max(.3,x2-x1):.3f}%;height:{max(.3,y2-y1):.3f}%">'
        f'<span>{html.escape(label)}</span></div>'
    )


def node_line(prefix: str, node: dict) -> str:
    value = text(node) or "∅"
    return f"<div><b>{prefix}</b> {html.escape(node.get('class_name','').split('.')[-1])} · {html.escape(value[:90])} · {html.escape(str(node['node_id']))}</div>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=35)
    parser.add_argument("--verdicts", type=Path)
    args = parser.parse_args()
    verdicts = {}
    if args.verdicts:
        verdicts = {int(row["number"]): row for row in json.loads(args.verdicts.read_text())["rows"]}
    records = read_jsonl(args.dataset)
    by_pages = {}
    for record in records:
        left, right = record["source"]["page_id"], record["target"]["page_id"]
        by_pages[(left, right)] = by_pages[(right, left)] = record
    rank2 = [row for row in read_jsonl(args.predictions) if row.get("gold_rank") == 2]
    selected = select_rows(rank2, args.limit)
    cards = []
    manifest = []
    for number, row in enumerate(selected, 1):
        source_page, target_page = row["graph_pair"]["source"], row["graph_pair"]["target"]
        source_record, target_record, anchors = orient(by_pages[(source_page, target_page)], source_page, target_page)
        source_nodes = {node["node_id"]: node for node in source_record["graph"]["nodes"]}
        target_nodes = {node["node_id"]: node for node in target_record["graph"]["nodes"]}
        gold_ids = {node["node_id"] for node in row["gold_targets"]}
        source_id = row["source"]["node_id"]
        local = []
        for source_anchor_id, target_anchor_id in anchors:
            if source_anchor_id == source_id or target_anchor_id in gold_ids:
                continue
            if source_anchor_id not in source_nodes or target_anchor_id not in target_nodes:
                continue
            local.append((distance(row["source"], source_nodes[source_anchor_id]), source_nodes[source_anchor_id], target_nodes[target_anchor_id]))
        local.sort(key=lambda value: value[0])
        local = local[:3]
        source_boxes = box(row["source"], "source", "S")
        target_boxes = box(row["prediction"], "wrong", "P1")
        for gold in row["gold_targets"]:
            target_boxes += box(gold, "gold", "G")
        for anchor_index, (_, source_anchor, target_anchor) in enumerate(local, 1):
            source_boxes += box(source_anchor, "anchor", f"A{anchor_index}")
            target_boxes += box(target_anchor, "anchor", f"A{anchor_index}")
        cat = category(row)
        gap = float(row["top_candidates"][0]["log_assignment"]) - float(row["top_candidates"][1]["log_assignment"])
        verdict = verdicts.get(number)
        verdict_text = (
            f'<b>{html.escape(verdict["verdict"])}</b>：{html.escape(verdict["reason"])}'
            if verdict
            else "人工结论：□ 单节点/视觉　□ endpoint粒度/标注　□ 局部对应可救　□ 页面或gold待审"
        )
        cards.append(
            f'<article class="card" data-category="{cat}"><header>#{number} · {html.escape(cat)} · {html.escape(source_page)} → {html.escape(target_page)}</header>'
            f'<div class="screens"><div class="screen"><img src="{data_uri(source_record["screenshot_path"])}">{source_boxes}</div>'
            f'<div class="screen"><img src="{data_uri(target_record["screenshot_path"])}">{target_boxes}</div></div>'
            f'<div class="meta">{node_line("S", row["source"])}{node_line("P1", row["prediction"])}{node_line("G", row["gold_targets"][0])}'
            f'<div><b>gap</b> {gap:.3f} · <b>P1↔G distance</b> {distance(row["prediction"], row["gold_targets"][0]):.3f} · <b>nearby labeled anchors</b> {len(local)}</div>'
            f'<div class="verdict">{verdict_text}</div></div></article>'
        )
        manifest.append({"number": number, "category": cat, "verdict": verdict, "graph_pair": row["graph_pair"], "source": row["source"], "prediction": row["prediction"], "gold_targets": row["gold_targets"], "local_anchors": [{"source": s, "target": t} for _, s, t in local]})
    counts = defaultdict(int)
    for row in rank2:
        counts[category(row)] += 1
    document = f'''<!doctype html><meta charset="utf-8"><title>Rank-2 local upper-bound audit</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#0b1020;color:#e8eefc;margin:0;padding:20px}}h1{{margin:0 0 8px}}.summary{{color:#adbbd8;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.card{{background:#141c31;border:1px solid #2a3858;border-radius:12px;overflow:hidden}}header{{padding:10px 12px;font-weight:700;background:#1b2743}}.screens{{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:8px}}.screen{{position:relative;line-height:0;background:#000}}.screen img{{width:100%;height:auto}}.box{{position:absolute;border:3px solid;box-sizing:border-box;min-width:4px;min-height:4px}}.box span{{position:absolute;left:0;top:-18px;font:700 11px/15px monospace;background:#111d;padding:1px 3px}}.source{{border-color:#4da3ff}}.source span{{background:#1670c5}}.wrong{{border-color:#ff4d67}}.wrong span{{background:#b51f37}}.gold{{border-color:#50e391}}.gold span{{background:#16834c}}.anchor{{border-color:#ffb84d;border-style:dashed}}.anchor span{{background:#986113}}.meta{{padding:0 12px 12px;font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}.verdict{{margin-top:8px;padding:7px;background:#0d1426;color:#fff1a8}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
</style><h1>Rank‑2：局部信息上限人工抽样</h1><div class="summary">全量 129 条；抽样 {len(selected)} 条。蓝=S，红=P1错误，绿=gold，橙=A1–A3（同页对中最近的其他 gold correspondence）。类别计数：{html.escape(json.dumps(dict(sorted(counts.items())),ensure_ascii=False))}</div><main class="grid">{''.join(cards)}</main>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps({"counts": counts, "samples": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rank2": len(rank2), "samples": len(selected), "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
