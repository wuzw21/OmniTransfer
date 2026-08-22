#!/usr/bin/env python3
"""Measure whether exact structural one-hot features can correct Top-2 errors.

Hyperparameters are selected on an app-held-out part of Train. Dev is used only
once for final reporting. The correction may only exchange current Top-1/Top-2.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def path_parts(node_id: str) -> tuple[int, ...]:
    result = []
    for part in str(node_id).split("."):
        try:
            result.append(int(part))
        except ValueError:
            result.append(-1)
    return tuple(result)


def parent_id(node_id: str) -> str:
    return str(node_id).rsplit(".", 1)[0] if "." in str(node_id) else ""


def role(node: dict) -> str:
    class_name = str(node.get("class_name", "")).lower()
    if node.get("editable"):
        return "input"
    if any(token in class_name for token in ("button", "switch", "checkbox", "radio", "toggle", "slider", "stepper", "picker")):
        return "control"
    if any(token in class_name for token in ("image", "icon")):
        return "image"
    if any(token in class_name for token in ("statictext", "textview", "label")):
        return "label"
    if any(token in class_name for token in ("other", "cell", "viewgroup", "framelayout", "linearlayout", "scrollview", "collection", "table", "window", "application", "navigationbar", "toolbar")):
        return "wrapper"
    return "control" if node.get("clickable") else "other"


def clipped(value: int | None, maximum: int) -> str:
    if value is None or value < 0:
        return "missing"
    return str(value) if value <= maximum else f"{maximum}+"


def unit_bin(value: float | None, count: int) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    return str(max(0, min(count - 1, int(value * count))))


def sparse_i32(matrix):
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


class FeatureBuilder:
    def __init__(self, rows: list[dict]):
        pages: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            source_page = row["graph_pair"]["source"]
            target_page = row["graph_pair"]["target"]
            pages[source_page][row["source"]["node_id"]] = row["source"]
            for field in ("all_candidates", "top_candidates", "gold_targets"):
                for node in row.get(field) or []:
                    pages[target_page][node["node_id"]] = node
        self.meta = self._build_meta(pages)

    @staticmethod
    def _build_meta(pages: dict[str, dict[str, dict]]) -> dict[tuple[str, str], dict]:
        result = {}
        for page, nodes in pages.items():
            children: dict[str, list[str]] = defaultdict(list)
            for node_id in nodes:
                children[parent_id(node_id)].append(node_id)
            for node_id, node in nodes.items():
                parts = path_parts(node_id)
                bbox = node.get("bbox") or [0, 0, 0, 0]
                result[(page, node_id)] = {
                    "depth": len(parts) - 1,
                    "child_index": parts[-1] if parts else -1,
                    "parent_slot": parts[-2] if len(parts) >= 2 else -1,
                    "grand_slot": parts[-3] if len(parts) >= 3 else -1,
                    "great_slot": parts[-4] if len(parts) >= 4 else -1,
                    "sibling_count": len(children[parent_id(node_id)]),
                    "child_count": len(children[node_id]),
                    "leaf": len(children[node_id]) == 0,
                    "x": (bbox[0] + bbox[2]) / 2,
                    "y": (bbox[1] + bbox[3]) / 2,
                    "w": max(0, bbox[2] - bbox[0]),
                    "h": max(0, bbox[3] - bbox[1]),
                    "role": role(node),
                    "class": str(node.get("class_name", "")).split(".")[-1],
                    "action": f"c{int(bool(node.get('clickable')))}e{int(bool(node.get('editable')))}s{int(bool(node.get('scrollable')))}",
                    "presence": "text" if str(node.get("text", "")).strip() else ("desc" if str(node.get("content_desc", "")).strip() else "none"),
                }
        return result

    def node_meta(self, page: str, node: dict) -> dict:
        return self.meta[(page, node["node_id"])]

    def candidate_features(self, row: dict, candidate: dict, sets: set[str]) -> dict[str, float]:
        source = self.node_meta(row["graph_pair"]["source"], row["source"])
        target = self.node_meta(row["graph_pair"]["target"], candidate)
        output = {}

        def add(name: str, value: str) -> None:
            output[f"{name}={value}"] = 1.0

        if "ordinal" in sets:
            for name, maximum in (("child_index", 15), ("parent_slot", 15), ("grand_slot", 15), ("great_slot", 15), ("depth", 24), ("sibling_count", 20), ("child_count", 20)):
                add(name, f"{clipped(source[name], maximum)}->{clipped(target[name], maximum)}")
            add("tail2", f"{clipped(source['parent_slot'], 15)}.{clipped(source['child_index'], 15)}->{clipped(target['parent_slot'], 15)}.{clipped(target['child_index'], 15)}")
            add("tail3", f"{clipped(source['grand_slot'], 15)}.{clipped(source['parent_slot'], 15)}.{clipped(source['child_index'], 15)}->{clipped(target['grand_slot'], 15)}.{clipped(target['parent_slot'], 15)}.{clipped(target['child_index'], 15)}")
        if "role" in sets:
            add("role", f"{source['role']}->{target['role']}")
            add("leaf", f"{int(source['leaf'])}->{int(target['leaf'])}")
            add("role_leaf", f"{source['role']}/{int(source['leaf'])}->{target['role']}/{int(target['leaf'])}")
            add("class", f"{source['class']}->{target['class']}")
            add("action", f"{source['action']}->{target['action']}")
            add("presence", f"{source['presence']}->{target['presence']}")
        if "position" in sets:
            for name, count in (("x", 8), ("y", 12), ("w", 8), ("h", 8)):
                add(name, f"{unit_bin(source[name], count)}->{unit_bin(target[name], count)}")
            add("leading_trailing", f"{unit_bin(source['x'], 3)}->{unit_bin(target['x'], 3)}")
            add("row_slot", f"{unit_bin(source['y'], 20)}/{unit_bin(source['x'], 5)}->{unit_bin(target['y'], 20)}/{unit_bin(target['x'], 5)}")
        if "card" in sets:
            add("card_slots", f"{clipped(source['great_slot'], 15)}.{clipped(source['grand_slot'], 15)}.{clipped(source['parent_slot'], 15)}->{clipped(target['great_slot'], 15)}.{clipped(target['grand_slot'], 15)}.{clipped(target['parent_slot'], 15)}")
            add("card_y", f"{unit_bin(source['y'], 8)}->{unit_bin(target['y'], 8)}")
        return output

    def difference(self, row: dict, sets: set[str], use_base: bool) -> dict[str, float]:
        first, second = row["top_candidates"][:2]
        output: dict[str, float] = defaultdict(float)
        for key, value in self.candidate_features(row, first, sets).items():
            output[key] += value
        for key, value in self.candidate_features(row, second, sets).items():
            output[key] -= value
        if use_base:
            output["base_log_gap"] = float(first.get("log_assignment", 0)) - float(second.get("log_assignment", 0))
        return dict(output)


def gold_indices(row: dict) -> set[int]:
    return {int(node["index"]) for node in row.get("gold_targets") or []}


def eligible(row: dict) -> bool:
    if len(row.get("top_candidates") or []) < 2:
        return False
    gold = gold_indices(row)
    first, second = row["top_candidates"][:2]
    return (int(first["index"]) in gold) != (int(second["index"]) in gold)


def label(row: dict) -> int:
    return int(int(row["top_candidates"][0]["index"]) in gold_indices(row))


def app_group(row: dict) -> str:
    return "/".join(row["graph_pair"]["source"].split("/")[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    builder = FeatureBuilder(train + dev)
    train_choices = [row for row in train if eligible(row)]
    dev_rows = [row for row in dev if len(row.get("top_candidates") or []) >= 2]
    labels = np.asarray([label(row) for row in train_choices])
    groups = np.asarray([app_group(row) for row in train_choices])
    fit_indices, validation_indices = next(GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=17).split(np.arange(len(train_choices)), labels, groups))
    variants = {
        "ordinal": ({"ordinal"}, False),
        "role_leaf": ({"role"}, False),
        "position": ({"position"}, False),
        "card": ({"card"}, False),
        "all_exact": ({"ordinal", "role", "position", "card"}, False),
        "all_exact_plus_base": ({"ordinal", "role", "position", "card"}, True),
    }
    results = {}
    for name, (sets, use_base) in variants.items():
        fit_rows = [train_choices[index] for index in fit_indices]
        validation_rows = [train_choices[index] for index in validation_indices]
        vectorizer = DictVectorizer()
        fit_x = sparse_i32(vectorizer.fit_transform([builder.difference(row, sets, use_base) for row in fit_rows]))
        validation_x = sparse_i32(vectorizer.transform([builder.difference(row, sets, use_base) for row in validation_rows]))
        fit_y, validation_y = labels[fit_indices], labels[validation_indices]
        best = None
        for class_weight in (None, "balanced"):
            for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                model = LogisticRegression(C=c_value, max_iter=3000, class_weight=class_weight, solver="liblinear", random_state=17).fit(fit_x, fit_y)
                probabilities = model.predict_proba(validation_x)[:, 1]
                for threshold in np.linspace(0.05, 0.95, 181):
                    accuracy = float(np.mean((probabilities >= threshold) == validation_y))
                    flips = int(np.sum(probabilities < threshold))
                    key = (accuracy, -flips, -abs(math.log10(c_value) + 1), int(class_weight is None))
                    if best is None or key > best[0]:
                        best = (key, c_value, class_weight, float(threshold), accuracy, flips)
        _, c_value, class_weight, threshold, validation_accuracy, validation_flips = best
        final_vectorizer = DictVectorizer()
        all_x = sparse_i32(final_vectorizer.fit_transform([builder.difference(row, sets, use_base) for row in train_choices]))
        final_model = LogisticRegression(C=c_value, max_iter=3000, class_weight=class_weight, solver="liblinear", random_state=17).fit(all_x, labels)
        dev_x = sparse_i32(final_vectorizer.transform([builder.difference(row, sets, use_base) for row in dev_rows]))
        dev_probabilities = final_model.predict_proba(dev_x)[:, 1]
        fixes = harms = flips = textless_fixes = close_fixes = 0
        fixed_examples, harmed_examples = [], []
        for row, keep, probability in zip(dev_rows, dev_probabilities >= threshold, dev_probabilities):
            if keep:
                continue
            flips += 1
            gold = gold_indices(row)
            first, second = row["top_candidates"][:2]
            before, after = int(first["index"]) in gold, int(second["index"]) in gold
            record = {"graph_pair": row["graph_pair"], "source": row["source"]["node_id"], "from": first["node_id"], "to": second["node_id"], "p_keep": float(probability)}
            if after and not before:
                fixes += 1
                fixed_examples.append(record)
                if not str(row["source"].get("text", "")).strip() and not str(row["source"].get("content_desc", "")).strip():
                    textless_fixes += 1
                first_center = ((first["bbox"][0] + first["bbox"][2]) / 2, (first["bbox"][1] + first["bbox"][3]) / 2)
                second_center = ((second["bbox"][0] + second["bbox"][2]) / 2, (second["bbox"][1] + second["bbox"][3]) / 2)
                if math.dist(first_center, second_center) < 0.1:
                    close_fixes += 1
            elif before and not after:
                harms += 1
                harmed_examples.append(record)
        baseline = sum(int(row["top_candidates"][0]["index"]) in gold_indices(row) for row in dev_rows)
        results[name] = {
            "n_features": len(final_vectorizer.feature_names_), "selected_C": c_value, "class_weight": class_weight,
            "threshold": threshold, "train_validation_always_keep_accuracy": float(np.mean(validation_y == 1)),
            "train_validation_accuracy": validation_accuracy, "train_validation_flips": validation_flips,
            "dev_flips": flips, "dev_fixes": fixes, "dev_harms": harms, "dev_net": fixes - harms,
            "dev_new_correct": baseline + fixes - harms, "dev_new_top1": (baseline + fixes - harms) / len(dev_rows),
            "dev_textless_fixes": textless_fixes, "dev_close_pair_fixes": close_fixes,
            "fixed_examples": fixed_examples[:20], "harmed_examples": harmed_examples[:20],
        }
    rank2 = [row for row in dev if row.get("gold_rank") == 2]
    evidence = {}
    for key in ("child_index", "parent_slot", "grand_slot", "great_slot", "depth", "sibling_count", "child_count", "role", "leaf", "class", "action"):
        counts = Counter()
        for row in rank2:
            source = builder.node_meta(row["graph_pair"]["source"], row["source"])
            target_page = row["graph_pair"]["target"]
            equalities = [int(source[key] == builder.node_meta(target_page, candidate)[key]) for candidate in row["top_candidates"][:2]]
            counts["gold" if equalities[1] > equalities[0] else ("wrong" if equalities[0] > equalities[1] else "tie")] += 1
        evidence[key] = dict(counts)
    report = {
        "protocol": "Train-only logistic Top-2 correction; app-held-out Train split selects C/class-weight/threshold; Dev only reports final result.",
        "data": {"train_rows": len(train), "train_choices": len(train_choices), "train_keep": int(labels.sum()), "train_flip": int((1 - labels).sum()), "dev_rows": len(dev), "dev_baseline_correct": sum(bool(row.get("correct")) for row in dev), "dev_rank2_errors": len(rank2), "validation_apps": sorted(set(groups[validation_indices]))},
        "rank2_exact_equality_evidence": evidence,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    compact = {name: {key: value[key] for key in ("n_features", "selected_C", "class_weight", "threshold", "train_validation_always_keep_accuracy", "train_validation_accuracy", "dev_flips", "dev_fixes", "dev_harms", "dev_net", "dev_new_correct", "dev_new_top1", "dev_textless_fixes", "dev_close_pair_fixes")} for name, value in results.items()}
    print(json.dumps({"output": str(args.output), "data": report["data"], "evidence": evidence, "results": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
