#!/usr/bin/env python3
"""Measure how every contextual layer changes correspondence accuracy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from omnitransfer.learned_matcher import (
    GeometricMatcher,
    matcher_inputs,
    mutual_log_assignment,
)
from omnitransfer.mapping_error_analysis import availability_slice
from omnitransfer.mapping_training import load_ui_correspondence_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    matcher = GeometricMatcher.from_checkpoint(args.checkpoint, device=args.device)
    pairs, _, adapter = load_ui_correspondence_pairs(
        [args.dataset],
        matcher_config=matcher.config,
        allowed_splits=frozenset({args.split}),
    )
    stage_hits: Counter[str] = Counter()
    slice_hits: dict[str, Counter[str]] = defaultdict(Counter)
    slice_totals: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    total = 0
    matcher.model.eval()
    with torch.no_grad():
        for pair in pairs:
            output = matcher.model(
                *matcher_inputs(
                    pair.graph_a,
                    pair.graph_b,
                    config=matcher.config,
                    device=args.device,
                )
            )
            descriptor = torch.nn.functional.normalize(
                output["source_descriptors"], dim=-1
            ) @ torch.nn.functional.normalize(output["target_descriptors"], dim=-1).T
            stages = [
                ("descriptor", descriptor),
                ("unary_assignment", mutual_log_assignment(output["unary_affinity"])),
                *[
                    (f"association_layer_{index}", scores)
                    for index, scores in enumerate(
                        output["assignment_scores_by_layer"], start=1
                    )
                ],
            ]
            adaptive_name = "confidence_adaptive_last_two"
            for transpose, positives, source_nodes, target_nodes in (
                (False, pair.positive_targets_a_to_b, pair.graph_a.nodes, pair.graph_b.nodes),
                (True, pair.positive_targets_b_to_a, pair.graph_b.nodes, pair.graph_a.nodes),
            ):
                for source_index, gold_indices in enumerate(positives):
                    if not gold_indices:
                        continue
                    availability = availability_slice(
                        {"text": source_nodes[source_index].text, "content_desc": source_nodes[source_index].content_desc},
                        {"text": target_nodes[gold_indices[0]].text, "content_desc": target_nodes[gold_indices[0]].content_desc},
                    )
                    hits: list[int] = []
                    for name, values in stages:
                        row = values.T[source_index] if transpose else values[source_index]
                        hit = int(int(torch.argmax(row)) in gold_indices)
                        stage_hits[name] += hit
                        slice_hits[availability][name] += hit
                        hits.append(hit)
                    if len(stages) >= 4:
                        previous = stages[-2][1].T[source_index] if transpose else stages[-2][1][source_index]
                        final = stages[-1][1].T[source_index] if transpose else stages[-1][1][source_index]
                        previous_probability = torch.softmax(previous, dim=0)
                        final_probability = torch.softmax(final, dim=0)
                        previous_margin = torch.topk(previous_probability, 2).values.diff().abs()[0]
                        final_margin = torch.topk(final_probability, 2).values.diff().abs()[0]
                        adaptive = previous if previous_margin > final_margin else final
                        adaptive_hit = int(int(torch.argmax(adaptive)) in gold_indices)
                        stage_hits[adaptive_name] += adaptive_hit
                        slice_hits[availability][adaptive_name] += adaptive_hit
                        hits.append(adaptive_hit)
                    total += 1
                    slice_totals[availability] += 1
                    transitions["".join(str(value) for value in hits)] += 1
    names = [name for name, _ in stages] + [adaptive_name]
    report = {
        "schema_version": "omnitransfer.association_depth_audit.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "positive_rows": total,
        "stage_top1": {
            name: stage_hits[name] / total if total else 0.0 for name in names
        },
        "slice_top1": {
            availability: {
                name: hits[name] / slice_totals[availability]
                for name in names
            }
            for availability, hits in sorted(slice_hits.items())
        },
        "transition_patterns": dict(transitions.most_common()),
        "adapter": adapter,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
