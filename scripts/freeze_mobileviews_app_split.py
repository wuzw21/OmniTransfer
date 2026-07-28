#!/usr/bin/env python3
"""Freeze leakage-free MobileViews train and dev-candidate App allowlists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omnitransfer.mobileviews_multi_app_pairs import (
    build_mobileviews_allowlist_pair_pool,
    discover_mobileviews_trace_apps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--frozen-test-apps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--pair-limit-per-app", type=int, default=1000)
    parser.add_argument("--minimum-matches", type=int, default=2)
    parser.add_argument("--candidate-pairs-per-structure", type=int, default=2000)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        raise SystemExit("--dev-fraction must be between zero and one")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    app_index, index_manifest = discover_mobileviews_trace_apps(args.traces_root)
    test_apps = sorted(
        {
            _canonical_app_id(line)
            for line in args.frozen_test_apps.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )
    missing_test_apps = sorted(set(test_apps) - set(app_index))
    ambiguous_test_apps = sorted(
        set(test_apps) & set(index_manifest["ambiguous_app_paths"])
    )
    if missing_test_apps or ambiguous_test_apps:
        raise SystemExit(
            "frozen test Apps are not uniquely available: "
            f"missing={missing_test_apps}, ambiguous={ambiguous_test_apps}"
        )

    candidate_apps = sorted(set(app_index) - set(test_apps))
    _, capacity = build_mobileviews_allowlist_pair_pool(
        args.traces_root,
        candidate_apps,
        pair_limit_per_app=args.pair_limit_per_app,
        minimum_matches=args.minimum_matches,
        candidate_pairs_per_structure=args.candidate_pairs_per_structure,
        split="diagnostic",
        label_status="unreviewed",
    )
    usable_apps = sorted(
        app_name
        for app_name, pair_count in capacity["per_app_pair_counts"].items()
        if pair_count > 0
    )
    if len(usable_apps) < 2:
        raise SystemExit("fewer than two non-test Apps produce strict page pairs")
    ranked_apps = sorted(
        usable_apps,
        key=lambda app_name: (_split_key(app_name, args.seed), app_name),
    )
    dev_count = min(
        len(ranked_apps) - 1,
        max(1, round(len(ranked_apps) * args.dev_fraction)),
    )
    dev_apps = sorted(ranked_apps[:dev_count])
    train_apps = sorted(ranked_apps[dev_count:])
    empty_apps = sorted(set(candidate_apps) - set(usable_apps))

    _write_lines(output / "train_apps.txt", train_apps)
    _write_lines(output / "dev_candidate_apps.txt", dev_apps)
    _write_lines(output / "test_apps.txt", test_apps)
    _write_lines(output / "empty_apps.txt", empty_apps)
    (output / "app_index.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "pair_capacity.json").write_text(
        json.dumps(capacity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omnitransfer.mobileviews_app_split.v2",
        "dataset": "MobileViews_Apps_CompleteTraces",
        "guiodyssey_included": False,
        "split_unit": "canonical_app_id",
        "algorithm": "strict_pair_capacity_then_blake2b_rank",
        "seed": args.seed,
        "minimum_matches": args.minimum_matches,
        "dev_fraction": args.dev_fraction,
        "counts": {
            "trace_dirs": index_manifest["trace_dirs_discovered"],
            "unique_apps": index_manifest["apps_discovered"],
            "ambiguous_apps": index_manifest["ambiguous_apps_discovered"],
            "usable_non_test_apps": len(usable_apps),
            "train_apps": len(train_apps),
            "dev_candidate_apps": len(dev_apps),
            "test_apps": len(test_apps),
            "empty_non_test_apps": len(empty_apps),
            "strict_page_pairs": capacity["pairs_selected"],
        },
        "disjoint": {
            "train_dev": not bool(set(train_apps) & set(dev_apps)),
            "train_test": not bool(set(train_apps) & set(test_apps)),
            "dev_test": not bool(set(dev_apps) & set(test_apps)),
        },
        "label_policy": {
            "train": "self_supervised_strict_partial_assignment",
            "dev_candidate": "unreviewed_until_human_acceptance",
            "test": "frozen_mobileviews_hard_acceptance",
            "unmatched_nodes": "ignored_without_human_null_label",
        },
        "frozen_test_source": str(args.frozen_test_apps.expanduser().resolve()),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _canonical_app_id(app_name: str) -> str:
    normalized = app_name.strip()
    return normalized[:-4] if normalized.endswith(".apk") else normalized


def _split_key(app_name: str, seed: int) -> str:
    return hashlib.blake2b(
        f"{seed}:{app_name}".encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


if __name__ == "__main__":
    main()
