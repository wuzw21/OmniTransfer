#!/usr/bin/env python3
"""Run a reproducible learned-matcher diagnostic grid and collect reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--source-context-nodes", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--latency-images", type=int, default=20)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.pretrained:
        from omnitransfer.learned_matcher import LearnedGraphMatcher

        pretrained = LearnedGraphMatcher.from_checkpoint(args.pretrained, device="cpu")
        expected = {
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
        }
        for name, value in expected.items():
            actual = getattr(pretrained.config, name)
            if value != actual:
                raise SystemExit(
                    f"--pretrained uses {name}={actual}, but the grid requested {value}."
                )
        if set(args.source_context_nodes) != {pretrained.config.source_context_nodes}:
            raise SystemExit(
                "--pretrained requires --source-context-nodes "
                f"{pretrained.config.source_context_nodes}."
            )
        if set(args.num_layers) != {pretrained.config.num_layers}:
            raise SystemExit(
                f"--pretrained requires --num-layers {pretrained.config.num_layers}."
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_script = Path(__file__).with_name("train_learned_matcher.py")
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
                    str(train_script),
                    "--input",
                    str(args.input.resolve()),
                    "--output",
                    str(model_path),
                    "--report",
                    str(report_path),
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(seed),
                    "--split-seed",
                    str(args.split_seed),
                    "--eval-split",
                    args.eval_split,
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--num-heads",
                    str(args.num_heads),
                    "--num-layers",
                    str(num_layers),
                    "--source-context-nodes",
                    str(context_nodes),
                    "--device",
                    args.device,
                    "--latency-images",
                    str(args.latency_images),
                ]
                if args.pretrained:
                    command.extend(("--pretrained", str(args.pretrained.resolve())))
                if args.limit_train:
                    command.extend(("--limit-train", str(args.limit_train)))
                if args.limit_eval:
                    command.extend(("--limit-eval", str(args.limit_eval)))
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
                metrics = report["metrics"]["all"]
                latency = report["new_image_latency"]
                runs.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "source_context_nodes": context_nodes,
                        "num_layers": num_layers,
                        "epochs": args.epochs,
                        "pretrained": report["pretrained"],
                        "top1_accuracy": metrics["top1_accuracy"],
                        "recall_at_5": metrics["recall_at_k"]["5"],
                        "wrong_target_rate": metrics["wrong_target_rate"],
                        "max_latency_ms": latency["max_latency_ms"],
                        "latency_budget_met": latency["budget_met"],
                        "final_loss": report["history"][-1]["loss"],
                        "report": str(report_path),
                        "model": str(model_path),
                    }
                )
                summary = {
                    "schema_version": "omnitransfer_diagnostic_grid_v1",
                    "input": str(args.input.resolve()),
                    "pretrained": (
                        str(args.pretrained.resolve()) if args.pretrained else None
                    ),
                    "split_seed": args.split_seed,
                    "eval_split": args.eval_split,
                    "runs": runs,
                }
                (output_dir / "grid_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(runs[-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
