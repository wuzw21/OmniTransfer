#!/usr/bin/env python3
"""Run Relation-Aware Cross-Attention Matcher ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--source-context-nodes", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--context-mask-probability", type=float, default=0.35)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_relation_aware_matcher.py")
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        for context_nodes in args.source_context_nodes:
            for num_layers in args.num_layers:
                run_id = f"seed{seed}_ctx{context_nodes}_layers{num_layers}"
                run_dir = output_dir / run_id
                report_path = run_dir / "report.json"
                model_path = run_dir / "model.pt"
                log_path = run_dir / "train.log"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(trainer),
                    "--input",
                    *(str(path.resolve()) for path in args.input),
                    "--validation-input",
                    *(str(path.resolve()) for path in args.validation_input),
                    "--output",
                    str(model_path),
                    "--report",
                    str(report_path),
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(seed),
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--num-heads",
                    str(args.num_heads),
                    "--num-layers",
                    str(num_layers),
                    "--source-context-nodes",
                    str(context_nodes),
                    "--context-mask-probability",
                    str(args.context_mask_probability),
                    "--device",
                    args.device,
                ]
                if args.max_pairs:
                    command.extend(("--max-pairs", str(args.max_pairs)))
                if args.force or not report_path.is_file():
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(
                            command,
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=os.environ.copy(),
                        )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                metrics = report["evaluation"]["metrics"]
                latency = metrics["warm_end_to_end_latency_ms"]
                runs.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "source_context_nodes": context_nodes,
                        "num_layers": num_layers,
                        "epochs": args.epochs,
                        "top1_accuracy": metrics["top1_accuracy"],
                        "recall_at_5": metrics["recall_at_k"]["5"],
                        "p95_end_to_end_ms": latency["p95"],
                        "max_end_to_end_ms": latency["max"],
                        "under_50ms_rate": latency["under_50ms_rate"],
                        "final_loss": report["training"]["history"][-1]["loss"],
                        "report": str(report_path),
                        "model": str(model_path),
                    }
                )
                summary = {
                    "schema_version": "omnitransfer.relation_aware_matcher_grid.v1",
                    "record_schema": "omnitransfer.ui_correspondence_pair.v1",
                    "inputs": [str(path.resolve()) for path in args.input],
                    "validation_inputs": [
                        str(path.resolve()) for path in args.validation_input
                    ],
                    "runs": runs,
                }
                (output_dir / "grid_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(runs[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
