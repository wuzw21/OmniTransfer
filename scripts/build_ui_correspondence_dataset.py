#!/usr/bin/env python3
"""Build one leakage-audited UI correspondence dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.importers import load_queries
from omnitransfer.mapping_dataset import (
    adapt_ase_queries,
    adapt_bmoca_trace_corpus,
    iter_split_ui_correspondence_pool,
    plan_ui_correspondence_pool,
    validate_ui_correspondence_pair,
    write_ui_correspondence_pool,
    write_ui_correspondence_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ase-queries", type=Path)
    parser.add_argument("--ase-assets-root", type=Path)
    parser.add_argument(
        "--bmoca-trace-corpus",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "BMOCA offline-trace corpus directories containing manifest.json, "
            "pair_memory.json, and per-run XML state catalogs. Automatic "
            "alignments remain unreviewed."
        ),
    )
    parser.add_argument(
        "--correspondence-pairs",
        "--page-pairs",
        dest="correspondence_pairs",
        nargs="*",
        type=Path,
        default=(),
        help=(
            "Existing omnitransfer.ui_correspondence_pair.v1 JSONL files. "
            "--page-pairs is a compatibility alias."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument(
        "--preserve-declared-splits",
        action="store_true",
        help="Compatibility mode; skip the canonical within-App pool split.",
    )
    args = parser.parse_args()

    inputs: list[dict[str, Any]] = []
    adapters: dict[str, Any] = {}
    ase_records: list[dict[str, Any]] = []
    if args.ase_queries is not None:
        query_path = args.ase_queries.expanduser().resolve()
        asset_root = (
            args.ase_assets_root.expanduser().resolve()
            if args.ase_assets_root is not None
            else query_path.parent.parent
        )
        ase_records = adapt_ase_queries(
            load_queries(query_path),
            asset_root=asset_root,
        )
        inputs.append(_input_manifest(query_path, kind="ase_queries"))
        adapters["ase2023"] = {
            "records": len(ase_records),
            "asset_root": str(asset_root),
            "output_schema": "omnitransfer.ui_correspondence_pair.v1",
        }

    bmoca_records: list[dict[str, Any]] = []
    bmoca_reports: list[dict[str, Any]] = []
    for corpus_value in args.bmoca_trace_corpus:
        corpus = corpus_value.expanduser().resolve()
        records, report = adapt_bmoca_trace_corpus(corpus)
        bmoca_records.extend(records)
        bmoca_reports.append({"corpus_root": str(corpus), **report})
        inputs.extend(
            (
                _input_manifest(
                    corpus / "manifest.json",
                    kind="bmoca_offline_trace_manifest",
                ),
                _input_manifest(
                    corpus / "pair_memory.json",
                    kind="bmoca_transfer_pair_memory",
                ),
            )
        )
    if bmoca_reports:
        adapters["bmoca_offline_trace_alignment"] = {
            "records": len(bmoca_records),
            "corpora": bmoca_reports,
            "output_schema": "omnitransfer.ui_correspondence_pair.v1",
            "formal_gold": False,
        }

    correspondence_adapter = {
        "records": 0,
        "files": len(args.correspondence_pairs),
        "validated_without_schema_conversion": True,
    }
    for pair_path in args.correspondence_pairs:
        path = pair_path.expanduser().resolve()
        inputs.append(_input_manifest(path, kind="ui_correspondence_pairs"))
    if args.correspondence_pairs:
        adapters["existing_ui_correspondence_pairs"] = correspondence_adapter

    if not ase_records and not bmoca_records and not args.correspondence_pairs:
        raise SystemExit(
            "Provide --ase-queries, --bmoca-trace-corpus, and/or at least one "
            "--correspondence-pairs file"
        )

    def records() -> Iterable[dict[str, Any]]:
        yield from ase_records
        yield from bmoca_records
        for pair_path in args.correspondence_pairs:
            path = pair_path.expanduser().resolve()
            for row in _iter_correspondence_pairs(path):
                correspondence_adapter["records"] += 1
                yield row

    protocol = {
        "record_schema_identical_across_datasets_and_splits": True,
        "one_training_entrypoint": "scripts/train_relation_aware_matcher.py",
        "train_dev_test_differ_only_by_split_and_label_status": True,
        "guiodyssey_included": False,
        "stable_ids_are_offline_labels_only": True,
        "dataset_sources_share_one_pool_and_one_splitter": True,
    }
    output_dir = args.output_dir.expanduser().resolve()
    if args.preserve_declared_splits:
        protocol["split_protocol"] = "preserve_source_declarations"
        manifest = write_ui_correspondence_dataset(
            records(),
            output_dir,
            metadata={
                "inputs": inputs,
                "protocol": protocol,
                "adapters": adapters,
            },
        )
    else:
        pool = write_ui_correspondence_pool(records(), output_dir / "pool.jsonl")
        split_plan = plan_ui_correspondence_pool(
            output_dir / "pool.jsonl",
            seed=args.split_seed,
            train_percent=args.train_percent,
            dev_percent=args.dev_percent,
        )
        protocol.update(
            {
                "split_protocol": "within_app_page_component_v1",
                "same_app_may_span_train_dev_test": True,
                "pair_page_component_overlap": "forbidden",
                "mobileviews_self_supervised_follows_assigned_split": True,
                "formal_metrics_require_label_status": "gold",
                "app_disjoint_primary_test": False,
                "app_disjoint_evaluation": "optional_generalization_slice",
            }
        )
        manifest = write_ui_correspondence_dataset(
            iter_split_ui_correspondence_pool(
                output_dir / "pool.jsonl",
                split_plan,
            ),
            output_dir,
            metadata={
                "inputs": inputs,
                "protocol": protocol,
                "pool": pool,
                "split": split_plan.audit,
                "adapters": adapters,
            },
        )
    print(json.dumps(manifest, ensure_ascii=False))


def _iter_correspondence_pairs(path: Path) -> Iterable[dict[str, Any]]:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = validate_ui_correspondence_pair(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid UI correspondence pair {path}:{line_number}: {exc}"
                ) from exc
            rows += 1
            yield row
    if rows == 0:
        raise ValueError(f"UI-correspondence JSONL is empty: {path}")


def _input_manifest(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    main()
