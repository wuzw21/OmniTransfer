"""Leakage-audited generalization protocols for correspondence pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable


@dataclass(frozen=True)
class GeneralizationSplit:
    """Deterministic pair partitions and their contamination audit."""

    splits: dict[str, tuple[dict[str, Any], ...]]
    audit: dict[str, Any]


def split_gui_odyssey_pair_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int = 17,
    train_percent: int = 70,
    dev_percent: int = 15,
) -> GeneralizationSplit:
    """Split pairs with task, episode, and trajectory leakage prevented."""

    if train_percent <= 0 or dev_percent < 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave non-empty train and test ranges")
    values = tuple(rows)
    if len(values) < 3:
        raise ValueError("at least three GUIOdyssey pairs are required")
    union = _UnionFind()
    for row in values:
        task = _required(row.get("meta_task"), "meta_task")
        source_episode = _endpoint_value(row, "source", "episode_id")
        target_episode = _endpoint_value(row, "target", "episode_id")
        union.join(f"task:{task}", f"episode:{source_episode}")
        union.join(f"task:{task}", f"episode:{target_episode}")

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        source_episode = _endpoint_value(row, "source", "episode_id")
        components[union.root(f"episode:{source_episode}")].append(row)
    if len(components) < 3:
        raise ValueError("task/episode connectivity leaves fewer than three split groups")

    percentages = {
        "train": train_percent,
        "dev": dev_percent,
        "test": 100 - train_percent - dev_percent,
    }
    targets = {
        split: len(values) * percentage / 100.0
        for split, percentage in percentages.items()
    }
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in percentages}
    ordered = sorted(
        components.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_bucket(seed, item[0]),
            item[0],
        ),
    )
    for _, component_rows in ordered:
        split = max(
            assigned,
            key=lambda name: (
                targets[name] - len(assigned[name]),
                percentages[name],
                name,
            ),
        )
        assigned[split].extend(component_rows)
    if any(not rows_for_split for rows_for_split in assigned.values()):
        raise ValueError("task/episode grouping produced an empty split")

    frozen = {
        split: tuple(sorted(rows_for_split, key=_row_sort_key))
        for split, rows_for_split in assigned.items()
    }
    audit = _split_audit(frozen, seed=seed, percentages=percentages)
    for field in ("meta_task", "episode_id", "trajectory_id"):
        if audit["overlap"][field]:
            raise AssertionError(f"{field} leakage detected: {audit['overlap'][field]}")
    return GeneralizationSplit(frozen, audit)


def _split_audit(
    splits: dict[str, tuple[dict[str, Any], ...]],
    *,
    seed: int,
    percentages: dict[str, int],
) -> dict[str, Any]:
    fields: dict[str, dict[str, set[str]]] = {
        "meta_task": defaultdict(set),
        "episode_id": defaultdict(set),
        "trajectory_id": defaultdict(set),
    }
    device_pairs: dict[str, Counter[str]] = {}
    for split, rows in splits.items():
        devices: Counter[str] = Counter()
        for row in rows:
            fields["meta_task"][_required(row.get("meta_task"), "meta_task")].add(split)
            for side in ("source", "target"):
                fields["episode_id"][_endpoint_value(row, side, "episode_id")].add(split)
            fields["trajectory_id"][_trajectory_id(row)].add(split)
            pair = sorted(
                (
                    _endpoint_value(row, "source", "device_name"),
                    _endpoint_value(row, "target", "device_name"),
                )
            )
            devices[" | ".join(pair)] += 1
        device_pairs[split] = devices
    overlap = {
        field: {
            value: sorted(split_names)
            for value, split_names in sorted(values.items())
            if len(split_names) > 1
        }
        for field, values in fields.items()
    }
    return {
        "schema_version": "omnitransfer_guiodyssey_generalization_split_v1",
        "seed": seed,
        "requested_percentages": percentages,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "groups": {
            split: {
                "meta_tasks": len({str(row.get("meta_task")) for row in rows}),
                "episodes": len(
                    {
                        _endpoint_value(row, side, "episode_id")
                        for row in rows
                        for side in ("source", "target")
                    }
                ),
                "trajectories": len({_trajectory_id(row) for row in rows}),
            }
            for split, rows in splits.items()
        },
        "overlap": overlap,
        "device_protocol": "stratified_and_reported_not_disjoint",
        "device_pair_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in device_pairs.items()
        },
    }


class _UnionFind:
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


def _endpoint_value(row: dict[str, Any], side: str, key: str) -> str:
    endpoint = row.get(side)
    if not isinstance(endpoint, dict):
        raise ValueError(f"{side} endpoint is absent")
    return _required(endpoint.get(key), f"{side}.{key}")


def _trajectory_id(row: dict[str, Any]) -> str:
    selection = row.get("selection") if isinstance(row.get("selection"), dict) else {}
    value = selection.get("trajectory_pair_id") or row.get("episode_pair_id")
    if value:
        return str(value)
    return " | ".join(
        sorted(
            (
                _endpoint_value(row, "source", "episode_id"),
                _endpoint_value(row, "target", "episode_id"),
            )
        )
    )


def _required(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is absent")
    return normalized


def _stable_bucket(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("pair_id") or ""), _trajectory_id(row)
