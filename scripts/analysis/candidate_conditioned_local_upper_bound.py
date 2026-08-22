#!/usr/bin/env python3
"""Probe the upper bound of candidate-conditioned local correspondence.

This is an analysis-only oracle.  For each query it may use the gold mappings
of *other* nodes on the same page pair as local anchors, but it never uses the
query node's gold mapping when constructing candidate features.  A fixed probe
is trained on Train predictions and evaluated once on Dev predictions.

The result answers a narrow question: if the iterative matcher supplied good
local correspondences, does the current Top-K candidate set contain enough
relational evidence to choose the correct endpoint?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def gold_indices(row: dict) -> set[int]:
    return {int(node["index"]) for node in row.get("gold_targets") or []}


def pair_key(row: dict) -> tuple[str, str]:
    pair = row["graph_pair"]
    return str(pair["source"]), str(pair["target"])


def path_parts(node_id: str) -> tuple[int, ...]:
    output = []
    for part in str(node_id).split("."):
        try:
            output.append(int(part))
        except ValueError:
            output.append(-1)
    return tuple(output)


def is_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return len(left) < len(right) and right[: len(left)] == left


def sign_bucket(value: float, epsilon: float = 0.025) -> int:
    if value < -epsilon:
        return -1
    if value > epsilon:
        return 1
    return 0


def node_geometry(node: dict) -> tuple[float, float, float, float]:
    bbox = node.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1), max(0.0, y2 - y1)


def relation(origin: dict, neighbor: dict) -> np.ndarray:
    """Observable relation from one node to another, without semantic text."""
    ox, oy, ow, oh = node_geometry(origin)
    nx, ny, nw, nh = node_geometry(neighbor)
    dx, dy = nx - ox, ny - oy
    origin_path, neighbor_path = path_parts(origin["node_id"]), path_parts(neighbor["node_id"])
    common = 0
    for left, right in zip(origin_path, neighbor_path):
        if left != right:
            break
        common += 1
    same_parent = len(origin_path) > 1 and origin_path[:-1] == neighbor_path[:-1]
    return np.asarray(
        [
            dx,
            dy,
            math.hypot(dx, dy),
            math.log((nw + 1e-3) / (ow + 1e-3)),
            math.log((nh + 1e-3) / (oh + 1e-3)),
            float(sign_bucket(dx)),
            float(sign_bucket(dy)),
            float(abs(dy) <= 0.5 * max(oh, nh, 0.02)),
            float(abs(dx) <= 0.5 * max(ow, nw, 0.02)),
            float(same_parent),
            float(is_prefix(origin_path, neighbor_path)),
            float(is_prefix(neighbor_path, origin_path)),
            float(max(-8, min(8, len(neighbor_path) - len(origin_path)))),
            float(min(16, len(origin_path) + len(neighbor_path) - 2 * common)),
            float(sign_bucket(float(neighbor_path[-1] - origin_path[-1]), 0.0) if same_parent else 0),
        ],
        dtype=np.float32,
    )


def relation_errors(source_relation: np.ndarray, target_relation: np.ndarray) -> np.ndarray:
    difference = np.abs(source_relation - target_relation)
    # Geometry is continuous. The remaining entries are exact relation facts.
    return np.asarray(
        [
            difference[0],
            difference[1],
            difference[2],
            difference[3],
            difference[4],
            float(difference[5] > 0),
            float(difference[6] > 0),
            float(difference[7] > 0),
            float(difference[8] > 0),
            float(difference[9] > 0),
            float(difference[10] > 0),
            float(difference[11] > 0),
            min(8.0, difference[12]),
            min(16.0, difference[13]),
            float(difference[14] > 0),
        ],
        dtype=np.float32,
    )


def role(node: dict) -> str:
    class_name = str(node.get("class_name", "")).lower()
    if node.get("editable"):
        return "input"
    if any(token in class_name for token in ("button", "switch", "checkbox", "radio", "toggle", "slider", "picker")):
        return "control"
    if any(token in class_name for token in ("image", "icon")):
        return "image"
    if any(token in class_name for token in ("statictext", "textview", "label")):
        return "label"
    if any(token in class_name for token in ("other", "cell", "viewgroup", "layout", "scroll", "collection", "table")):
        return "wrapper"
    return "other"


def has_text(node: dict) -> bool:
    return bool(str(node.get("text", "")).strip() or str(node.get("content_desc", "")).strip())


def aggregate_errors(errors: list[np.ndarray]) -> np.ndarray:
    if not errors:
        return np.full(15 * 5 + 8, 9.0, dtype=np.float32)
    matrix = np.stack(errors)
    scalar = (
        5.0 * matrix[:, 0]
        + 5.0 * matrix[:, 1]
        + 2.0 * matrix[:, 2]
        + 0.25 * matrix[:, 3]
        + 0.25 * matrix[:, 4]
        + matrix[:, 5:12].sum(axis=1)
        + 0.15 * matrix[:, 12]
        + 0.08 * matrix[:, 13]
        + matrix[:, 14]
    )
    order = np.argsort(scalar)
    best = matrix[order[: min(5, len(order))]]
    statistics = np.concatenate(
        [
            matrix.min(axis=0),
            matrix.mean(axis=0),
            np.quantile(matrix, 0.25, axis=0),
            best[: min(3, len(best))].mean(axis=0),
            best.mean(axis=0),
        ]
    )
    counts = np.asarray(
        [
            len(errors),
            float(np.sum(scalar < 0.5)),
            float(np.sum(scalar < 1.0)),
            float(np.sum(scalar < 2.0)),
            float(np.sum(scalar < 4.0)),
            float(np.min(scalar)),
            float(np.mean(np.sort(scalar)[: min(3, len(scalar))])),
            float(np.mean(np.sort(scalar)[: min(5, len(scalar))])),
        ],
        dtype=np.float32,
    )
    return np.concatenate([statistics, counts]).astype(np.float32)


class OracleFeatureBuilder:
    def __init__(self, rows: list[dict]):
        anchors: dict[tuple[str, str], list[tuple[int, dict, dict]]] = defaultdict(list)
        for row in rows:
            for target in row.get("gold_targets") or []:
                anchors[pair_key(row)].append((int(row["source"]["index"]), row["source"], target))
        self.anchors = anchors

    def features(self, row: dict, candidate: dict) -> dict[str, np.ndarray]:
        source = row["source"]
        source_index = int(source["index"])
        candidate_index = int(candidate["index"])
        other_source_anchors = [anchor for anchor in self.anchors[pair_key(row)] if anchor[0] != source_index]
        # The clean local oracle excludes every gold edge incident to either
        # endpoint under test.  Otherwise merely observing that another source
        # claims this candidate leaks a strong negative label.
        usable = [anchor for anchor in other_source_anchors if int(anchor[2]["index"]) != candidate_index]
        errors = [relation_errors(relation(source, anchor_source), relation(candidate, anchor_target)) for _, anchor_source, anchor_target in usable]
        occupied = sum(int(anchor_target["index"]) == candidate_index for _, _, anchor_target in other_source_anchors)
        sx, sy, sw, sh = node_geometry(source)
        tx, ty, tw, th = node_geometry(candidate)
        source_path, target_path = path_parts(source["node_id"]), path_parts(candidate["node_id"])
        pair = np.asarray(
            [
                float(candidate.get("log_assignment", -30.0)),
                float(candidate.get("rank_probability", 0.0)),
                abs(sx - tx),
                abs(sy - ty),
                abs(sw - tw),
                abs(sh - th),
                float(role(source) == role(candidate)),
                float(has_text(source) == has_text(candidate)),
                float(bool(source.get("clickable")) == bool(candidate.get("clickable"))),
                float(bool(source.get("editable")) == bool(candidate.get("editable"))),
                float(bool(source.get("scrollable")) == bool(candidate.get("scrollable"))),
                float(abs(len(source_path) - len(target_path))),
                float(abs(source_path[-1] - target_path[-1]) if source_path and target_path else 16),
            ],
            dtype=np.float32,
        )
        return {
            "base": pair[:2],
            "pair": pair,
            "local": aggregate_errors(errors),
            "assignment": np.asarray([float(occupied), float(occupied > 0)], dtype=np.float32),
        }


def candidate_rows(rows: list[dict], top_k: int, builder: OracleFeatureBuilder, variant: str):
    x, y, groups = [], [], []
    for row_index, row in enumerate(rows):
        gold = gold_indices(row)
        candidates = (row.get("top_candidates") or [])[:top_k]
        if not candidates or not any(int(candidate["index"]) in gold for candidate in candidates):
            continue
        for candidate in candidates:
            features = builder.features(row, candidate)
            if variant == "base":
                vector = features["base"]
            elif variant == "pair":
                vector = features["pair"]
            elif variant == "local":
                vector = np.concatenate([features["base"], features["local"]])
            elif variant == "local_assignment":
                vector = np.concatenate([features["base"], features["local"], features["assignment"]])
            else:
                raise ValueError(variant)
            x.append(vector)
            y.append(float(int(candidate["index"]) in gold))
            groups.append(row_index)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8), np.asarray(groups, dtype=np.int32)


def fit_probe(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    positive = max(1, int(y.sum()))
    negative = max(1, len(y) - positive)
    weights = np.where(y == 1, negative / positive, 1.0)
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=1.0,
        random_state=17,
    ).fit(x, y, sample_weight=weights)


def center_distance(left: dict, right: dict) -> float:
    lx, ly, _, _ = node_geometry(left)
    rx, ry, _, _ = node_geometry(right)
    return math.hypot(lx - rx, ly - ry)


def evaluate(rows: list[dict], top_k: int, builder: OracleFeatureBuilder, model, variant: str) -> dict:
    correct = fixes = harms = rank2_fixes = rank2_total = textless_fixes = close_fixes = 0
    reachable = 0
    vectors: list[np.ndarray] = []
    offsets: list[tuple[int, int] | None] = []
    for row in rows:
        candidates = (row.get("top_candidates") or [])[:top_k]
        if not candidates:
            offsets.append(None)
            continue
        gold = gold_indices(row)
        before = int(candidates[0]["index"]) in gold
        is_reachable = any(int(candidate["index"]) in gold for candidate in candidates)
        reachable += int(is_reachable)
        if is_reachable:
            start = len(vectors)
            for candidate in candidates:
                features = builder.features(row, candidate)
                if variant == "base":
                    vector = features["base"]
                elif variant == "pair":
                    vector = features["pair"]
                elif variant == "local":
                    vector = np.concatenate([features["base"], features["local"]])
                else:
                    vector = np.concatenate([features["base"], features["local"], features["assignment"]])
                vectors.append(vector)
            offsets.append((start, len(vectors)))
        else:
            offsets.append(None)
    all_scores = model.predict_proba(np.asarray(vectors, dtype=np.float32))[:, 1] if vectors else np.asarray([])
    for row, offset in zip(rows, offsets):
        candidates = (row.get("top_candidates") or [])[:top_k]
        if not candidates:
            continue
        gold = gold_indices(row)
        before = int(candidates[0]["index"]) in gold
        if offset is not None:
            start, stop = offset
            scores = all_scores[start:stop]
            chosen = candidates[int(np.argmax(scores))]
        else:
            chosen = candidates[0]
        after = int(chosen["index"]) in gold
        correct += int(after)
        fixes += int(after and not before)
        harms += int(before and not after)
        if row.get("gold_rank") == 2:
            rank2_total += 1
            rank2_fixes += int(after)
            if after and not before:
                gold_node = next(candidate for candidate in candidates if int(candidate["index"]) in gold)
                if not has_text(row["source"]) or not has_text(gold_node):
                    textless_fixes += 1
                if center_distance(candidates[0], gold_node) < 0.1:
                    close_fixes += 1
    return {
        "correct": correct,
        "top1": correct / len(rows),
        "reachable": reachable,
        "fixes": fixes,
        "harms": harms,
        "net": fixes - harms,
        "rank2_total": rank2_total,
        "rank2_fixed": rank2_fixes,
        "rank2_fix_rate": rank2_fixes / rank2_total if rank2_total else 0.0,
        "rank2_textless_fixes": textless_fixes,
        "rank2_close_fixes": close_fixes,
    }


def ceilings(rows: list[dict]) -> dict:
    output = {"rows": len(rows)}
    for top_k in (1, 2, 3, 5, 10):
        count = sum((row.get("gold_rank") or 10**9) <= top_k for row in rows)
        output[f"top{top_k}_count"] = count
        output[f"top{top_k}"] = count / len(rows)
    rank2 = [row for row in rows if row.get("gold_rank") == 2]
    textless = 0
    close = 0
    for row in rank2:
        gold = gold_indices(row)
        gold_node = next(candidate for candidate in row["top_candidates"] if int(candidate["index"]) in gold)
        textless += int(not has_text(row["source"]) or not has_text(gold_node))
        close += int(center_distance(row["top_candidates"][0], gold_node) < 0.1)
    output["rank2_count"] = len(rank2)
    output["rank2_textless"] = textless
    output["rank2_close"] = close
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    train, dev = read_jsonl(args.train), read_jsonl(args.dev)
    train_builder, dev_builder = OracleFeatureBuilder(train), OracleFeatureBuilder(dev)
    results = {}
    for variant in ("base", "pair", "local", "local_assignment"):
        train_x, train_y, _ = candidate_rows(train, args.top_k, train_builder, variant)
        model = fit_probe(train_x, train_y)
        results[variant] = {
            "train_examples": len(train_y),
            "train_positive": int(train_y.sum()),
            "features": int(train_x.shape[1]),
            "dev": evaluate(dev, args.top_k, dev_builder, model, variant),
        }
    report = {
        "protocol": (
            "Analysis-only Train-to-Dev probe. Candidate features may use gold mappings of other nodes "
            "from the same page pair. Clean local features exclude gold edges incident to the query source "
            "or candidate target. The local_assignment variant separately exposes target occupancy as an "
            "optimistic one-to-one-assignment oracle. Dev labels are used only for reporting."
        ),
        "top_k": args.top_k,
        "ceilings": ceilings(dev),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
