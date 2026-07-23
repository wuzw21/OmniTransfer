#!/usr/bin/env python3
"""Smoke-test GUIOdyssey weak pairs with the real cross-attention matcher."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from omnitransfer.gui_odyssey_pairs import load_gui_odyssey_training_pairs
from omnitransfer.learned_matcher import (
    MatcherConfig,
    build_relation_aware_matcher,
    parameter_count,
)
from omnitransfer.self_supervised import matching_loss


def _resolve_device(requested: str, torch: object) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _synchronize(device: str, torch: object) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    args = parser.parse_args()

    import torch

    device = _resolve_device(args.device, torch)
    config = MatcherConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0,
    )
    pairs, adapter_manifest = load_gui_odyssey_training_pairs(
        args.pairs,
        args.annotations,
        screenshot_dir=args.screenshots,
        max_pairs=args.max_pairs,
        matcher_config=config,
    )
    if not pairs:
        raise SystemExit("no usable GUIOdyssey pairs were loaded")

    model = build_relation_aware_matcher(config).to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    elapsed_ms: list[float] = []
    for pair in pairs:
        _synchronize(device, torch)
        started = time.perf_counter()
        loss = matching_loss(
            model,
            pair,
            device=device,
            matcher_config=config,
        )
        (loss / len(pairs)).backward()
        _synchronize(device, torch)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        losses.append(float(loss.detach().cpu()))

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    payload = {
        "schema_version": "omnitransfer_guiodyssey_cross_attention_smoke_v1",
        "status": "passed"
        if losses and all(math.isfinite(value) for value in losses) and finite_gradients
        else "failed",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
        "matcher": {
            "architecture": "relation_aware_bidirectional_cross_attention",
            "config": asdict(config),
            "trainable_parameters": parameter_count(model),
        },
        "adapter": adapter_manifest,
        "execution": {
            "pairs_executed": len(pairs),
            "source_node_counts": [len(pair.graph_a.nodes) for pair in pairs],
            "target_node_counts": [len(pair.graph_b.nodes) for pair in pairs],
            "mean_loss": sum(losses) / len(losses),
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "gradients_present": bool(gradients),
            "all_gradients_finite": finite_gradients,
            "mean_forward_backward_ms": sum(elapsed_ms) / len(elapsed_ms),
            "max_forward_backward_ms": max(elapsed_ms),
        },
        "limitations": {
            "formal_benchmark_gold": False,
            "structural_only": adapter_manifest["pairs_with_both_screenshots"] == 0,
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
