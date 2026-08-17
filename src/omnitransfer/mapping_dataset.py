"""Unified UI correspondence dataset contract for learned UI mapping."""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

UI_CORRESPONDENCE_PAIR_SCHEMA = "omnitransfer.ui_correspondence_pair.v1"
_SPLITS = frozenset({"train", "dev", "test", "diagnostic"})
_LABEL_STATUSES = frozenset({"gold", "weak", "self_supervised", "unreviewed"})


@dataclass(frozen=True)
class UICorrespondenceSplitPlan:
    """Frozen within-App assignment for one canonical correspondence pool."""

    assignments: dict[str, str]
    component_ids: dict[str, str]
    app_ids: dict[str, str]
    audit: dict[str, Any]
    pool_sha256: str


def write_ui_correspondence_pool(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze heterogeneous inputs as one canonical, unsplit JSONL pool."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pair_ids: set[str] = set()
    record_count = 0
    match_count = 0
    dataset_counts: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw_record in records:
                record = validate_ui_correspondence_pair(raw_record)
                pair_id = record["pair_id"]
                if pair_id in pair_ids:
                    raise ValueError(f"duplicate pair id in unified pool: {pair_id}")
                pair_ids.add(pair_id)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                record_count += 1
                match_count += len(record["matches"])
                dataset_counts[
                    str(record["provenance"].get("dataset") or "unknown")
                ] += 1
        if record_count == 0:
            raise ValueError("UI correspondence pool is empty")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": "omnitransfer.ui_correspondence_pool_manifest.v1",
        "record_schema": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "path": path.name,
        "records": record_count,
        "matches": match_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "split_semantics": "source_declaration_only_until_pool_split",
    }


def plan_ui_correspondence_pool(
    pool_path: str | Path,
    *,
    seed: int = 17,
    train_percent: int = 80,
    dev_percent: int = 10,
) -> UICorrespondenceSplitPlan:
    """Plan a deterministic page-disjoint split inside each App."""

    if train_percent <= 0 or dev_percent < 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave non-empty train and test ranges")
    path = Path(pool_path).expanduser().resolve()
    union = _StringUnionFind()
    pair_ids: set[str] = set()
    page_keys_by_pair: dict[str, tuple[str, str]] = {}
    app_ids: dict[str, str] = {}
    dataset_counts: Counter[str] = Counter()
    for record in _iter_ui_correspondence_pool(path):
        pair_id = record["pair_id"]
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair id in unified pool: {pair_id}")
        pair_ids.add(pair_id)
        app_id = _correspondence_app_id(record)
        source_page = f"{app_id}\0{record['source']['page_id']}"
        target_page = f"{app_id}\0{record['target']['page_id']}"
        union.join(source_page, target_page)
        page_keys_by_pair[pair_id] = (source_page, target_page)
        app_ids[pair_id] = app_id
        dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
    if not pair_ids:
        raise ValueError("UI correspondence pool is empty")

    component_pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    component_pages: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair_id, page_keys in page_keys_by_pair.items():
        component_key = (app_ids[pair_id], union.root(page_keys[0]))
        component_pairs[component_key].append(pair_id)
        component_pages[component_key].update(page_keys)

    components_by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_ids: dict[str, str] = {}
    for component_key, pair_ids in component_pairs.items():
        app_id = component_key[0]
        pages = sorted(component_pages[component_key])
        component_id = hashlib.blake2b(
            f"{app_id}\0{chr(0).join(pages)}".encode(),
            digest_size=10,
        ).hexdigest()
        ordered_pair_ids = tuple(sorted(pair_ids))
        components_by_app[app_id].append(
            {
                "component_id": component_id,
                "pair_ids": ordered_pair_ids,
                "pages": tuple(pages),
                "size": len(ordered_pair_ids),
            }
        )
        for pair_id in ordered_pair_ids:
            component_ids[pair_id] = component_id

    percentages = {
        "train": train_percent,
        "dev": dev_percent,
        "test": 100 - train_percent - dev_percent,
    }
    assignments: dict[str, str] = {}
    component_assignments: dict[str, str] = {}
    app_split_counts: dict[str, dict[str, int]] = {}
    for app_id, components in sorted(components_by_app.items()):
        assigned = _assign_app_components(
            components,
            seed=seed,
            percentages=percentages,
        )
        counts: Counter[str] = Counter()
        for component in components:
            component_id = component["component_id"]
            split = assigned[component_id]
            component_assignments[component_id] = split
            for pair_id in component["pair_ids"]:
                assignments[pair_id] = split
                counts[split] += 1
        app_split_counts[app_id] = {
            split: counts[split] for split in ("train", "dev", "test")
        }

    page_splits: dict[str, set[str]] = defaultdict(set)
    app_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for pair_id, split in assignments.items():
        split_counts[split] += 1
        app_splits[app_ids[pair_id]].add(split)
        for page_key in page_keys_by_pair[pair_id]:
            page_splits[page_key].add(split)
    page_overlap = _cross_split_values(page_splits)
    if page_overlap:
        raise AssertionError(
            f"page leakage in pool split: {_first_overlap(page_overlap)}"
        )
    apps_in_multiple_splits = sorted(
        app_id for app_id, splits in app_splits.items() if len(splits) > 1
    )
    audit = {
        "schema_version": "omnitransfer.ui_correspondence_pool_split.v1",
        "seed": seed,
        "requested_percentages": percentages,
        "app_protocol": "within_app_page_component",
        "dataset_protocol": "one_mixed_pool",
        "records": len(assignments),
        "components": len(component_assignments),
        "apps": len(app_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "apps_in_multiple_splits": apps_in_multiple_splits,
        "apps_in_multiple_splits_count": len(apps_in_multiple_splits),
        "app_split_counts": app_split_counts,
        "overlap": {
            "pair_id": {},
            "page_id": {},
            "component_id": {},
        },
    }
    return UICorrespondenceSplitPlan(
        assignments=assignments,
        component_ids=component_ids,
        app_ids=app_ids,
        audit=audit,
        pool_sha256=_sha256_file(path),
    )


def iter_split_ui_correspondence_pool(
    pool_path: str | Path,
    plan: UICorrespondenceSplitPlan,
) -> Iterable[dict[str, Any]]:
    """Apply a frozen pool plan and emit canonical train/review/test records."""

    path = Path(pool_path).expanduser().resolve()
    if _sha256_file(path) != plan.pool_sha256:
        raise ValueError("unified correspondence pool changed after split planning")
    seen: set[str] = set()
    for source_record in _iter_ui_correspondence_pool(path):
        record = deepcopy(source_record)
        pair_id = record["pair_id"]
        if pair_id not in plan.assignments:
            raise ValueError(f"pair is absent from split plan: {pair_id}")
        assigned_split = plan.assignments[pair_id]
        provenance = record["provenance"]
        provenance["source_split"] = record["split"]
        provenance["source_label_status"] = record["label_status"]
        provenance.pop("reserved_split", None)
        provenance["split_protocol"] = "within_app_page_component_v1"
        record["partition_keys"] = [f"pool:component:{plan.component_ids[pair_id]}"]
        record["slices"]["app"] = plan.app_ids[pair_id]
        if _is_mobileviews_self_supervised_proposal(record):
            record["split"] = assigned_split
            record["label_status"] = "self_supervised"
        elif assigned_split == "train":
            record["split"] = "train"
        elif record["label_status"] == "gold":
            record["split"] = assigned_split
        else:
            record["split"] = "diagnostic"
            record["label_status"] = "unreviewed"
            provenance["reserved_split"] = assigned_split
        seen.add(pair_id)
        yield validate_ui_correspondence_pair(record)
    missing = sorted(set(plan.assignments) - seen)
    if missing:
        raise ValueError(f"split plan contains pairs absent from pool: {missing[:3]}")


def audit_ui_correspondence_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate rows and reject pair, page, or partition leakage across splits."""

    if not records:
        raise ValueError("UI correspondence dataset is empty")
    normalized = [validate_ui_correspondence_pair(record) for record in records]
    pair_splits: dict[str, set[str]] = defaultdict(set)
    page_splits: dict[str, set[str]] = defaultdict(set)
    partition_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    for record in normalized:
        split = record["split"]
        assignment_split = _assignment_split(record)
        pair_splits[record["pair_id"]].add(assignment_split)
        for side in ("source", "target"):
            page_splits[record[side]["page_id"]].add(assignment_split)
        for key in record["partition_keys"]:
            partition_splits[key].add(assignment_split)
        split_counts[split] += 1
        assignment_counts[assignment_split] += 1
        label_counts[record["label_status"]] += 1
        split_label_counts[split][record["label_status"]] += 1
        dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
        for match in record["matches"]:
            match_counts[match["label"]] += 1

    duplicate_pairs = _cross_split_values(pair_splits)
    if duplicate_pairs:
        raise ValueError(f"pair id leakage: {_first_overlap(duplicate_pairs)}")
    page_overlap = _cross_split_values(page_splits)
    if page_overlap:
        raise ValueError(f"page leakage: {_first_overlap(page_overlap)}")
    partition_overlap = _cross_split_values(partition_splits)
    if partition_overlap:
        raise ValueError(f"partition leakage: {_first_overlap(partition_overlap)}")
    return {
        "schema_version": "omnitransfer.ui_correspondence_audit.v1",
        "valid": True,
        "records": len(normalized),
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "split_label_status_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "match_label_counts": dict(sorted(match_counts.items())),
        "overlap": {"pair_id": {}, "page_id": {}, "partition_key": {}},
    }


def write_ui_correspondence_dataset(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    review_candidates: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Stream validated rows into atomic split files with leakage auditing."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = ("train", "dev", "test", "diagnostic")
    final_paths = {split: output / f"{split}.jsonl" for split in splits}
    temporary_paths = {
        split: path.with_suffix(path.suffix + ".part")
        for split, path in final_paths.items()
    }
    handles: dict[str, Any] = {}
    pair_splits: dict[str, set[str]] = defaultdict(set)
    page_splits: dict[str, set[str]] = defaultdict(set)
    partition_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    file_match_counts: Counter[str] = Counter()
    record_count = 0
    try:
        handles = {
            split: temporary_paths[split].open("w", encoding="utf-8")
            for split in splits
        }
        for raw_record in records:
            record = validate_ui_correspondence_pair(raw_record)
            split = record["split"]
            assignment_split = _assignment_split(record)
            _record_assignment(
                pair_splits, record["pair_id"], assignment_split, "pair id"
            )
            for side in ("source", "target"):
                _record_assignment(
                    page_splits,
                    record[side]["page_id"],
                    assignment_split,
                    "page",
                )
            for key in record["partition_keys"]:
                _record_assignment(
                    partition_splits,
                    key,
                    assignment_split,
                    "partition",
                )
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handles[split].write(encoded + "\n")
            record_count += 1
            split_counts[split] += 1
            assignment_counts[assignment_split] += 1
            label_counts[record["label_status"]] += 1
            split_label_counts[split][record["label_status"]] += 1
            dataset_counts[str(record["provenance"].get("dataset") or "unknown")] += 1
            file_match_counts[split] += len(record["matches"])
            for match in record["matches"]:
                match_counts[match["label"]] += 1
        if record_count == 0:
            raise ValueError("UI correspondence dataset is empty")
        for handle in handles.values():
            handle.close()
        handles.clear()
        for split in splits:
            temporary_paths[split].replace(final_paths[split])
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    audit = {
        "schema_version": "omnitransfer.ui_correspondence_audit.v1",
        "valid": True,
        "records": record_count,
        "matches": sum(match_counts.values()),
        "split_counts": dict(sorted(split_counts.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "label_status_counts": dict(sorted(label_counts.items())),
        "split_label_status_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "match_label_counts": dict(sorted(match_counts.items())),
        "overlap": {"pair_id": {}, "page_id": {}, "partition_key": {}},
    }
    files: dict[str, Any] = {}
    for split in splits:
        path = final_paths[split]
        files[path.name] = _file_manifest(
            path,
            records=split_counts[split],
            matches=file_match_counts[split],
        )

    for reserved_split, candidates in sorted((review_candidates or {}).items()):
        if reserved_split not in {"dev", "test"}:
            raise ValueError(f"unsupported review candidate split: {reserved_split}")
        rows = [validate_ui_correspondence_pair(record) for record in candidates]
        if any(record["split"] != "diagnostic" for record in rows):
            raise ValueError("review candidates must use the diagnostic split")
        if any(record["label_status"] != "unreviewed" for record in rows):
            raise ValueError("review candidates must be unreviewed")
        path = output / f"review_{reserved_split}_candidates.jsonl"
        ordered = sorted(rows, key=lambda row: row["pair_id"])
        _write_jsonl_atomic(path, ordered)
        files[path.name] = _file_manifest(
            path,
            records=len(ordered),
            matches=sum(len(row["matches"]) for row in ordered),
        )

    manifest = {
        "schema_version": "omnitransfer.ui_correspondence_dataset_manifest.v1",
        "record_schema": UI_CORRESPONDENCE_PAIR_SCHEMA,
        "audit": audit,
        "files": files,
        "metadata": deepcopy(metadata or {}),
    }
    _write_text_atomic(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def validate_ui_correspondence_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized UI correspondence record or reject an invalid dataset row."""

    if not isinstance(record, dict):
        raise TypeError("UI correspondence pair must be an object")
    value = deepcopy(record)
    if value.get("schema_version") != UI_CORRESPONDENCE_PAIR_SCHEMA:
        raise ValueError("unsupported UI correspondence-pair schema")
    _required_text(value, "pair_id")
    split = _required_text(value, "split")
    if split not in _SPLITS:
        raise ValueError(f"unsupported split: {split}")
    label_status = _required_text(value, "label_status")
    if label_status not in _LABEL_STATUSES:
        raise ValueError(f"unsupported label status: {label_status}")
    if split in {"dev", "test"} and label_status not in {
        "gold",
        "self_supervised",
    }:
        raise ValueError(f"{split} contains unsupported labels: {label_status}")

    source_ids = _validate_page(value.get("source"), side="source")
    target_ids = _validate_page(value.get("target"), side="target")
    matches = value.get("matches")
    if not isinstance(matches, list) or not matches:
        raise ValueError("matches must be a non-empty list")
    seen_sources: set[str] = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise TypeError(f"matches[{index}] must be an object")
        source_id = _required_text(match, "source_node_id")
        if source_id not in source_ids:
            raise ValueError(f"match source node is absent: {source_id}")
        if source_id in seen_sources:
            raise ValueError(f"source node has duplicate match rows: {source_id}")
        seen_sources.add(source_id)
        label = _required_text(match, "label")
        target_node_ids = match.get("target_node_ids")
        if not isinstance(target_node_ids, list):
            raise TypeError(f"matches[{index}].target_node_ids must be a list")
        normalized_targets = list(
            dict.fromkeys(str(node_id) for node_id in target_node_ids)
        )
        if label == "correspondence" and not normalized_targets:
            raise ValueError("correspondence match has no target nodes")
        if label == "no_correspondence" and normalized_targets:
            raise ValueError("no_correspondence match contains target nodes")
        if label not in {"correspondence", "no_correspondence"}:
            raise ValueError(f"unsupported match label: {label}")
        missing = [
            node_id for node_id in normalized_targets if node_id not in target_ids
        ]
        if missing:
            raise ValueError(f"match target nodes are absent: {missing}")
        match["target_node_ids"] = normalized_targets

    partition_keys = value.get("partition_keys")
    if not isinstance(partition_keys, list) or not partition_keys:
        raise ValueError("partition_keys must be a non-empty list")
    value["partition_keys"] = list(
        dict.fromkeys(_nonempty_text(item, "partition key") for item in partition_keys)
    )
    if not isinstance(value.get("provenance"), dict):
        raise TypeError("provenance must be an object")
    if not isinstance(value.get("slices"), dict):
        raise TypeError("slices must be an object")
    return value


def _validate_page(page: Any, *, side: str) -> set[str]:
    if not isinstance(page, dict):
        raise TypeError(f"{side} page must be an object")
    _required_text(page, "page_id")
    _required_text(page, "platform")
    if not isinstance(page.get("screenshot_path"), str):
        raise TypeError(f"{side}.screenshot_path must be a string")
    graph = page.get("graph")
    if not isinstance(graph, dict):
        raise TypeError(f"{side}.graph must be an object")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{side}.graph.nodes must be a non-empty list")
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"{side}.graph.nodes[{index}] must be an object")
        node_id = _required_text(node, "node_id")
        _required_text(node, "origin_id")
        if node_id in node_ids:
            raise ValueError(f"duplicate {side} node id: {node_id}")
        node_ids.add(node_id)
    return node_ids


def _required_text(value: dict[str, Any], key: str) -> str:
    return _nonempty_text(value.get(key), key)


def _nonempty_text(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is absent")
    return normalized


def _iter_ui_correspondence_pool(path: Path) -> Iterable[dict[str, Any]]:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = validate_ui_correspondence_pair(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid unified correspondence pool {path}:{line_number}: {exc}"
                ) from exc
            rows += 1
            yield record
    if rows == 0:
        raise ValueError(f"UI correspondence pool is empty: {path}")


def _correspondence_app_id(record: dict[str, Any]) -> str:
    slices = record["slices"]
    provenance = record["provenance"]
    for values in (slices, provenance):
        for key in (
            "app",
            "app_id",
            "package",
            "package_name",
            "app_package",
            "domain",
            "website",
        ):
            normalized = str(values.get(key) or "").strip()
            if normalized:
                return _normalize_app_id(normalized)
    trace = str(provenance.get("trace") or "").strip()
    if trace:
        return _normalize_app_id(trace)
    dataset = str(provenance.get("dataset") or "unknown").strip()
    return f"{dataset}:unscoped"


def _normalize_app_id(value: str) -> str:
    normalized = value.strip()
    if normalized.lower().endswith(".apk"):
        normalized = normalized[:-4]
    return normalized


def _assign_app_components(
    components: list[dict[str, Any]],
    *,
    seed: int,
    percentages: dict[str, int],
) -> dict[str, str]:
    if not components:
        return {}
    stable = sorted(
        components,
        key=lambda component: (
            int(component["size"]),
            _stable_bucket(seed, str(component["component_id"])),
            str(component["component_id"]),
        ),
    )
    assignments: dict[str, str] = {}
    counts: Counter[str] = Counter()
    remaining = list(stable)

    reserve = ["test"]
    if percentages["dev"] > 0:
        reserve.append("dev")
    for split in reserve:
        if len(remaining) <= 1:
            break
        component = remaining.pop(0)
        component_id = str(component["component_id"])
        assignments[component_id] = split
        counts[split] += int(component["size"])

    total = sum(int(component["size"]) for component in components)
    targets = {
        split: total * percentage / 100.0 for split, percentage in percentages.items()
    }
    enabled_splits = tuple(
        split for split in ("train", "dev", "test") if percentages[split] > 0
    )
    for component in sorted(
        remaining,
        key=lambda item: (
            -int(item["size"]),
            _stable_bucket(seed, str(item["component_id"])),
            str(item["component_id"]),
        ),
    ):
        split = max(
            enabled_splits,
            key=lambda name: (
                targets[name] - counts[name],
                percentages[name],
                name == "train",
            ),
        )
        component_id = str(component["component_id"])
        assignments[component_id] = split
        counts[split] += int(component["size"])
    return assignments


def _stable_bucket(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _is_mobileviews_self_supervised_proposal(record: dict[str, Any]) -> bool:
    if record["label_status"] not in {"unreviewed", "self_supervised"}:
        return False
    provenance = record["provenance"]
    dataset = str(provenance.get("dataset") or "").lower()
    annotation = str(provenance.get("annotation") or "").lower()
    return "mobileviews" in dataset and annotation.startswith("automatic_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StringUnionFind:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def root(self, value: str) -> str:
        self.parents.setdefault(value, value)
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.root(parent)
        return self.parents[value]

    def join(self, left: str, right: str) -> None:
        left_root = self.root(left)
        right_root = self.root(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def _cross_split_values(values: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        key: sorted(splits) for key, splits in sorted(values.items()) if len(splits) > 1
    }


def _assignment_split(record: dict[str, Any]) -> str:
    split = record["split"]
    if split != "diagnostic":
        return split
    reserved_split = str(record["provenance"].get("reserved_split") or "")
    if not reserved_split:
        return split
    if reserved_split not in {"dev", "test"}:
        raise ValueError(f"unsupported diagnostic reserved split: {reserved_split}")
    return reserved_split


def _first_overlap(overlap: dict[str, list[str]]) -> str:
    key = next(iter(overlap))
    return f"{key} -> {overlap[key]}"


def _record_assignment(
    assignments: dict[str, set[str]],
    key: str,
    split: str,
    label: str,
) -> None:
    assigned = assignments[key]
    assigned.add(split)
    if len(assigned) > 1:
        raise ValueError(f"{label} leakage: {key} -> {sorted(assigned)}")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _file_manifest(path: Path, *, records: int, matches: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.name,
        "records": records,
        "matches": matches,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
