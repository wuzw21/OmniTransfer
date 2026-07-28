"""Balanced MobileViews complete-trace sampling across applications."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

from omnitransfer.mobileviews_trace_pairs import (
    build_mobileviews_sequence_pair_pilot,
    build_mobileviews_trace_pair_pilot,
)


def discover_mobileviews_trace_apps(
    traces_root: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Index canonical App identities across extracted MobileViews batches."""

    root = Path(traces_root).expanduser().resolve()
    trace_dirs, source_roots = _discover_trace_dirs(root)
    trace_index, ambiguous_apps = _index_trace_dirs(trace_dirs)
    return trace_index, {
        "schema_version": "omnitransfer.mobileviews_trace_app_index.v1",
        "traces_root": str(root),
        "source_roots": [str(path.relative_to(root) or ".") for path in source_roots],
        "trace_dirs_discovered": len(trace_dirs),
        "apps_discovered": len(trace_index),
        "ambiguous_apps_discovered": len(ambiguous_apps),
        "app_paths": {
            app_name: str(path.relative_to(root))
            for app_name, path in sorted(trace_index.items())
        },
        "ambiguous_app_paths": {
            app_name: [str(path.relative_to(root)) for path in paths]
            for app_name, paths in sorted(ambiguous_apps.items())
        },
    }


def build_mobileviews_allowlist_pair_pool(
    traces_root: str | Path,
    app_names: list[str],
    *,
    pair_limit_per_app: int = 1000,
    minimum_matches: int = 2,
    candidate_pairs_per_structure: int = 2000,
    split: str = "diagnostic",
    label_status: str = "unreviewed",
    reserved_split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an exhaustive repeated-structure and transition pool for frozen Apps."""

    root = Path(traces_root).expanduser().resolve()
    if not app_names:
        raise ValueError("app_names must not be empty")
    if (
        pair_limit_per_app <= 0
        or minimum_matches <= 0
        or candidate_pairs_per_structure < 0
    ):
        raise ValueError("pair limits and minimum_matches must be positive")
    if (split, label_status) not in {
        ("diagnostic", "unreviewed"),
        ("train", "self_supervised"),
    }:
        raise ValueError(
            "MobileViews pools require diagnostic/unreviewed or "
            "train/self_supervised supervision"
        )
    if reserved_split is not None and (
        (split, label_status) != ("diagnostic", "unreviewed")
        or reserved_split not in {"dev", "test"}
    ):
        raise ValueError(
            "reserved_split requires diagnostic/unreviewed data and dev or test"
        )
    trace_index, index_manifest = discover_mobileviews_trace_apps(root)
    ambiguous_apps = index_manifest["ambiguous_app_paths"]
    requested_apps = sorted({_app_identity(name) for name in app_names})
    records: list[dict[str, Any]] = []
    per_app_counts: dict[str, int] = {}
    rejected = Counter()
    for app_name in requested_apps:
        if app_name in ambiguous_apps:
            rejected["ambiguous_app"] += 1
            per_app_counts[app_name] = 0
            continue
        trace_dir = trace_index.get(app_name)
        if trace_dir is None:
            rejected["missing_app"] += 1
            per_app_counts[app_name] = 0
            continue
        try:
            structure_records, _ = build_mobileviews_trace_pair_pilot(
                trace_dir,
                pair_limit=pair_limit_per_app,
                minimum_matches=minimum_matches,
                candidate_pairs_per_structure=candidate_pairs_per_structure,
            )
            sequence_records, _ = build_mobileviews_sequence_pair_pilot(
                trace_dir,
                pair_limit=pair_limit_per_app,
                minimum_matches=minimum_matches,
            )
        except (FileNotFoundError, OSError, ValueError):
            rejected["invalid_trace"] += 1
            continue
        merged: dict[str, dict[str, Any]] = {}
        for source, app_records in (
            ("repeated_structure", structure_records),
            ("sequence_transition", sequence_records),
        ):
            for record in app_records:
                pair_id = str(record["pair_id"])
                if pair_id in merged:
                    continue
                record["provenance"]["sampling"] = {
                    "strategy": "frozen_app_exhaustive_pool",
                    "source": source,
                }
                record["provenance"]["app_id"] = app_name
                record["provenance"]["trace_source_relative"] = str(
                    trace_dir.relative_to(root)
                )
                app_partition = f"mobileviews:app:{app_name}"
                if app_partition not in record["partition_keys"]:
                    record["partition_keys"].append(app_partition)
                record["split"] = split
                record["label_status"] = label_status
                if label_status == "self_supervised":
                    record["provenance"]["supervision"] = (
                        "view_str_label_only_strict_partial_assignment"
                    )
                if reserved_split is not None:
                    record["provenance"]["reserved_split"] = reserved_split
                merged[pair_id] = record
                if len(merged) >= pair_limit_per_app:
                    break
            if len(merged) >= pair_limit_per_app:
                break
        app_values = list(merged.values())
        records.extend(app_values)
        per_app_counts[app_name] = len(app_values)
    return records, {
        "schema_version": "omnitransfer.mobileviews_allowlist_pair_pool.v1",
        "traces_root": str(root),
        "trace_dirs_discovered": index_manifest["trace_dirs_discovered"],
        "apps_discovered": index_manifest["apps_discovered"],
        "ambiguous_apps_discovered": index_manifest["ambiguous_apps_discovered"],
        "ambiguous_app_paths": ambiguous_apps,
        "apps_requested": len(requested_apps),
        "apps_selected": sum(value > 0 for value in per_app_counts.values()),
        "pairs_selected": len(records),
        "pair_limit_per_app": pair_limit_per_app,
        "minimum_matches": minimum_matches,
        "candidate_pairs_per_structure": candidate_pairs_per_structure,
        "split": split,
        "label_status": label_status,
        "reserved_split": reserved_split,
        "per_app_pair_counts": per_app_counts,
        "rejected": dict(sorted(rejected.items())),
        "label_boundary": (
            "view_str_is_label_only_and_never_model_input"
            if label_status == "self_supervised"
            else "unreviewed_exhaustive_pool_for_human_acceptance_review"
        ),
    }


def build_mobileviews_multi_app_pair_pilot(
    traces_root: str | Path,
    *,
    app_limit: int = 40,
    pairs_per_app: int = 3,
    total_pair_limit: int = 100,
    minimum_matches: int = 2,
    seed: int = 17,
    include_sequence_pairs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build App-balanced pairs, supplementing sparse apps with trace transitions."""

    root = Path(traces_root).expanduser().resolve()
    if app_limit <= 0 or pairs_per_app <= 0 or total_pair_limit <= 0:
        raise ValueError("sampling limits must be positive")
    trace_dirs = sorted(
        _discover_trace_dirs(root)[0],
        key=lambda path: _sample_key(path, root=root, seed=seed),
    )
    per_app_records: list[tuple[str, list[dict[str, Any]]]] = []
    rejected = Counter()
    sequence_pairs_selected = 0
    scanned = 0
    for trace_dir in trace_dirs:
        if len(per_app_records) >= app_limit:
            break
        scanned += 1
        try:
            candidates, _ = build_mobileviews_trace_pair_pilot(
                trace_dir,
                pair_limit=max(20, pairs_per_app * 8),
                minimum_matches=minimum_matches,
                candidate_pairs_per_structure=max(8, pairs_per_app * 4),
            )
        except (FileNotFoundError, OSError, ValueError):
            rejected["invalid_trace"] += 1
            continue
        selected: list[dict[str, Any]] = []
        seen_structures: set[str] = set()
        seen_pair_ids: set[str] = set()
        for record in candidates:
            structure = str(record["provenance"].get("structure_str") or "")
            if not structure or structure in seen_structures:
                continue
            seen_structures.add(structure)
            seen_pair_ids.add(str(record.get("pair_id") or ""))
            record["provenance"]["sampling"] = {
                "strategy": "round_robin_app_then_unique_structure",
                "pairs_per_app": pairs_per_app,
                "source": "repeated_structure",
            }
            selected.append(record)
            if len(selected) >= pairs_per_app:
                break
        if include_sequence_pairs and len(selected) < pairs_per_app:
            try:
                sequence_candidates, _ = build_mobileviews_sequence_pair_pilot(
                    trace_dir,
                    pair_limit=max(50, pairs_per_app * 8),
                    minimum_matches=minimum_matches,
                )
            except (FileNotFoundError, OSError, ValueError):
                sequence_candidates = []
                rejected["invalid_sequence_trace"] += 1
            for record in sequence_candidates:
                pair_id = str(record.get("pair_id") or "")
                if not pair_id or pair_id in seen_pair_ids:
                    continue
                structure = str(record["provenance"].get("structure_str") or "")
                seen_pair_ids.add(pair_id)
                record["provenance"]["sampling"] = {
                    "strategy": "round_robin_app_then_unique_structure",
                    "pairs_per_app": pairs_per_app,
                    "source": "sequence_transition",
                }
                selected.append(record)
                sequence_pairs_selected += 1
                if len(selected) >= pairs_per_app:
                    break
        if not selected:
            rejected["no_usable_pairs"] += 1
            continue
        per_app_records.append((trace_dir.name, selected))

    records: list[dict[str, Any]] = []
    for pair_rank in range(pairs_per_app):
        for _, app_records in per_app_records:
            if pair_rank < len(app_records):
                records.append(app_records[pair_rank])
            if len(records) >= total_pair_limit:
                break
        if len(records) >= total_pair_limit:
            break
    manifest = {
        "schema_version": "omnitransfer.mobileviews_multi_app_pair_pilot.v1",
        "traces_root": str(root),
        "trace_dirs_discovered": len(trace_dirs),
        "trace_dirs_scanned": scanned,
        "apps_selected": len(per_app_records),
        "pairs_selected": len(records),
        "sequence_pairs_selected": sequence_pairs_selected,
        "include_sequence_pairs": include_sequence_pairs,
        "pairs_per_app_limit": pairs_per_app,
        "total_pair_limit": total_pair_limit,
        "sampling_seed": seed,
        "selected_apps": [app for app, _ in per_app_records],
        "per_app_pair_counts": {
            app: len(app_records) for app, app_records in per_app_records
        },
        "rejected": dict(sorted(rejected.items())),
        "sampling_boundary": {
            "app_balanced": True,
            "unique_structure_within_app": "repeated_structure_candidates_only",
            "sequence_transition_pairs_may_share_structure": True,
            "labels": "unreviewed_diagnostic",
        },
    }
    return records, manifest


def _sample_key(path: Path, *, root: Path, seed: int) -> tuple[str, str]:
    relative = str(path.relative_to(root))
    digest = hashlib.blake2b(
        f"{seed}:{relative}".encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return digest, relative


def _discover_trace_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    batch_roots = sorted(path for path in root.glob("*_v1") if path.is_dir())
    source_roots = batch_roots or [root]
    trace_dirs: set[Path] = set()
    for source_root in source_roots:
        trace_dirs.update(
            path.parent for path in source_root.rglob("screenshot_state_mapping.csv")
        )
        trace_dirs.update(
            path.parent.parent for path in source_root.rglob("states/state_*.json")
        )
    return sorted(trace_dirs), source_roots


def _index_trace_dirs(
    trace_dirs: list[Path],
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    candidates: dict[str, list[Path]] = {}
    for trace_dir in trace_dirs:
        candidates.setdefault(_app_identity(trace_dir.name), []).append(trace_dir)
    ambiguous = {
        app_name: paths for app_name, paths in candidates.items() if len(paths) > 1
    }
    index = {
        app_name: paths[0] for app_name, paths in candidates.items() if len(paths) == 1
    }
    return index, ambiguous


def _app_identity(app_name: str) -> str:
    normalized = app_name.strip()
    return normalized[:-4] if normalized.endswith(".apk") else normalized
