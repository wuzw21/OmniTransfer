import json
from pathlib import Path

import pytest

from omnitransfer.mapping_dataset import (
    MAPPING_PAGE_PAIR_SCHEMA,
    adapt_ase_queries,
    adapt_gui_odyssey_human_reviews,
    adapt_gui_odyssey_rows,
    audit_mapping_page_pairs,
    validate_mapping_page_pair,
    write_mapping_dataset,
)
from omnitransfer.schema import Candidate, Query


def test_page_pair_schema_accepts_multiple_matches_for_train_and_test() -> None:
    base = {
        "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
        "pair_id": "pair-1",
        "label_status": "gold",
        "source": {
            "page_id": "source-page",
            "platform": "ios",
            "screenshot_path": "source.png",
            "graph": {
                "graph_id": "source-page",
                "width": 100,
                "height": 200,
                "nodes": [
                    {"node_id": "source-a", "origin_id": "source-a"},
                    {"node_id": "source-b", "origin_id": "source-b"},
                ],
            },
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": "target.png",
            "graph": {
                "graph_id": "target-page",
                "width": 200,
                "height": 100,
                "nodes": [
                    {"node_id": "target-a", "origin_id": "target-a"},
                    {"node_id": "target-b", "origin_id": "target-b"},
                    {"node_id": "target-c", "origin_id": "target-c"},
                ],
            },
        },
        "matches": [
            {
                "source_node_id": "source-a",
                "target_node_ids": ["target-a", "target-b"],
                "label": "correspondence",
            },
            {
                "source_node_id": "source-b",
                "target_node_ids": ["target-c"],
                "label": "correspondence",
            },
        ],
        "partition_keys": ["ase:app:example"],
        "provenance": {"dataset": "ase2023", "annotation": "public_gold"},
        "slices": {"form_factor": "phone"},
    }

    train = validate_mapping_page_pair({**base, "split": "train"})
    test = validate_mapping_page_pair({**base, "split": "test"})

    assert train.keys() == test.keys()
    assert len(train["matches"]) == 2
    assert train["matches"][0]["target_node_ids"] == ["target-a", "target-b"]


def test_ase_queries_are_grouped_into_one_multi_match_page_pair() -> None:
    common_metadata = {
        "app": "Example",
        "source_screen": "Example/iOS/0",
        "target_screen": "Example/Android/0",
        "source_screenshot_path": "ios.png",
        "target_screenshot_path": "android.png",
        "source_xml_path": "",
        "target_xml_path": "",
    }
    queries = [
        Query(
            query_id="mapping-a",
            source={
                "node_id": "source-a",
                "text": "Search",
                "class_name": "XCUIElementTypeButton",
                "bounds": [0.7, 0.0, 0.9, 0.1],
                "metadata": {"platform": "ios"},
            },
            target_candidates=(
                Candidate(
                    "target-a",
                    bbox=(0.6, 0.0, 0.8, 0.1),
                    metadata={"content_desc": "Search", "class_name": "ImageButton"},
                ),
                Candidate("target-a-label", bbox=(0.6, 0.0, 0.8, 0.1)),
            ),
            gold_candidate_id="target-a",
            metadata={
                **common_metadata,
                "gold_equivalent_candidate_ids": ["target-a-label"],
            },
        ),
        Query(
            query_id="mapping-b",
            source={
                "node_id": "source-b",
                "text": "Library",
                "class_name": "XCUIElementTypeButton",
                "bounds": [0.7, 0.8, 0.9, 0.9],
                "metadata": {"platform": "ios"},
            },
            target_candidates=(
                Candidate("target-b", bbox=(0.6, 0.8, 0.8, 0.9)),
            ),
            gold_candidate_id="target-b",
            metadata=common_metadata,
        ),
    ]
    for query in queries:
        query.metadata["split"] = "train"

    records = adapt_ase_queries(queries)

    assert len(records) == 1
    record = records[0]
    assert record["split"] == "train"
    assert record["label_status"] == "gold"
    assert record["partition_keys"] == ["ase:app:Example"]
    assert [match["source_node_id"] for match in record["matches"]] == [
        "source-a",
        "source-b",
    ]
    assert record["matches"][0]["target_node_ids"] == [
        "target-a-label",
        "target-a",
    ]


def test_gui_odyssey_weak_pair_uses_the_same_page_pair_schema(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for episode_id, device_name, point in (
        ("episode-phone", "Medium Phone", [500, 500]),
        ("episode-fold", "Pixel Fold", [250, 750]),
    ):
        episode = {
            "episode_id": episode_id,
            "device_info": {"w": 1000, "h": 1000, "device_name": device_name},
            "task_info": {
                "category": "General_Tool",
                "app": ["Settings"],
                "meta_task": "Open settings.",
            },
            "steps": [
                {
                    "step": 3,
                    "screenshot": f"{episode_id}_3.png",
                    "action": "CLICK",
                    "info": [point, point],
                    "sam2_bbox": [point[0] - 50, point[1] - 50, point[0] + 50, point[1] + 50],
                    "low_level_instruction": "Open Apps.",
                }
            ],
        }
        (annotations / f"{episode_id}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    rows = [
        {
            "schema_version": "omniflow.guiodyssey_sequence_filtered_click_pair.v1",
            "pair_id": "gui-pair",
            "meta_task": "Open settings.",
            "source": {
                "episode_id": "episode-phone",
                "step_index": 3,
                "device_name": "Medium Phone",
            },
            "target": {
                "episode_id": "episode-fold",
                "step_index": 3,
                "device_name": "Pixel Fold",
            },
            "selection": {
                "gold_label": False,
                "trajectory_pair_id": "trajectory-phone-fold",
            },
        }
    ]

    records, manifest = adapt_gui_odyssey_rows(
        rows,
        annotations,
        split="train",
    )

    assert manifest["accepted_pairs"] == 1
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == MAPPING_PAGE_PAIR_SCHEMA
    assert record["label_status"] == "weak"
    assert record["matches"] == [
        {
            "source_node_id": "action_target",
            "target_node_ids": ["action_target"],
            "label": "correspondence",
        }
    ]
    assert record["partition_keys"] == [
        "guiodyssey:task:Open settings.",
        "guiodyssey:episode:episode-fold",
        "guiodyssey:episode:episode-phone",
        "guiodyssey:trajectory:trajectory-phone-fold",
    ]


def test_dataset_audit_rejects_partition_leakage() -> None:
    def record(pair_id: str, split: str, source_page: str, partition: str) -> dict:
        return {
            "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": split,
            "label_status": "gold",
            "source": {
                "page_id": source_page,
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": source_page,
                    "nodes": [{"node_id": "source", "origin_id": "source"}],
                },
            },
            "target": {
                "page_id": f"target-{pair_id}",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"target-{pair_id}",
                    "nodes": [{"node_id": "target", "origin_id": "target"}],
                },
            },
            "matches": [
                {
                    "source_node_id": "source",
                    "target_node_ids": ["target"],
                    "label": "correspondence",
                }
            ],
            "partition_keys": [partition],
            "provenance": {"dataset": "example", "annotation": "gold"},
            "slices": {},
        }

    rows = [
        record("train-pair", "train", "source-train", "task:shared"),
        record("test-pair", "test", "source-test", "task:shared"),
    ]

    with pytest.raises(ValueError, match="partition leakage"):
        audit_mapping_page_pairs(rows)


def test_diagnostic_uses_its_frozen_reserved_split_for_leakage_audit() -> None:
    def record(pair_id: str, split: str, reserved_split: str | None) -> dict:
        provenance = {"dataset": "guiodyssey", "annotation": "test"}
        if reserved_split is not None:
            provenance["reserved_split"] = reserved_split
        return {
            "schema_version": MAPPING_PAGE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": split,
            "label_status": "weak" if split == "diagnostic" else "gold",
            "source": {
                "page_id": f"source-{pair_id}",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"source-{pair_id}",
                    "nodes": [{"node_id": "source", "origin_id": "source"}],
                },
            },
            "target": {
                "page_id": f"target-{pair_id}",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"target-{pair_id}",
                    "nodes": [{"node_id": "target", "origin_id": "target"}],
                },
            },
            "matches": [
                {
                    "source_node_id": "source",
                    "target_node_ids": ["target"],
                    "label": "correspondence",
                }
            ],
            "partition_keys": ["guiodyssey:task:shared"],
            "provenance": provenance,
            "slices": {},
        }

    audit = audit_mapping_page_pairs(
        [record("candidate", "diagnostic", "dev"), record("gold", "dev", None)]
    )
    assert audit["assignment_counts"] == {"dev": 2}

    with pytest.raises(ValueError, match="partition leakage"):
        audit_mapping_page_pairs(
            [
                record("candidate", "diagnostic", "test"),
                record("gold", "dev", None),
            ]
        )


def test_reviewed_gui_odyssey_multi_point_pair_can_enter_formal_test(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "schema_version": "omnitransfer_guiodyssey_sequence_node_alignment_v1",
            "pair_id": "reviewed-pair",
            "meta_task": "Open Apps settings.",
            "source": {
                "episode_id": "review-source",
                "device_name": "Medium Phone",
                "width": 1000,
                "height": 2000,
                "image_url": "source.png",
            },
            "target": {
                "episode_id": "review-target",
                "device_name": "Pixel Fold",
                "width": 2000,
                "height": 1000,
                "image_url": "target.png",
            },
            "selection": {"trajectory_pair_id": "review-trajectory"},
            "annotation": {
                "status": "reviewed",
                "label": "correspondence",
                "source_nodes": [
                    {"node_id": "source-apps", "point_normalized": [500, 500]},
                    {"node_id": "source-search", "point_normalized": [900, 100]},
                ],
                "target_nodes": [
                    {"node_id": "target-apps", "point_normalized": [400, 500]},
                    {"node_id": "target-search", "point_normalized": [850, 100]},
                    {"node_id": "target-search-label", "point_normalized": [800, 100]},
                ],
                "matches": [
                    {"source_node_id": "source-apps", "target_node_id": "target-apps"},
                    {"source_node_id": "source-search", "target_node_id": "target-search"},
                    {
                        "source_node_id": "source-search",
                        "target_node_id": "target-search-label",
                    },
                ],
            },
        }
    ]

    records, manifest = adapt_gui_odyssey_human_reviews(
        rows,
        split="test",
        screenshot_dir=tmp_path,
    )

    assert manifest["accepted_pairs"] == 1
    assert records[0]["label_status"] == "gold"
    assert records[0]["matches"][1] == {
        "source_node_id": "source-search",
        "target_node_ids": ["target-search", "target-search-label"],
        "label": "correspondence",
    }


def test_writer_emits_same_schema_for_train_and_test(tmp_path: Path) -> None:
    queries = []
    for split, app in (("train", "TrainApp"), ("test", "TestApp")):
        queries.append(
            Query(
                query_id=f"{split}-mapping",
                source={
                    "node_id": f"{split}-source",
                    "metadata": {"platform": "ios"},
                },
                target_candidates=(Candidate(f"{split}-target"),),
                gold_candidate_id=f"{split}-target",
                metadata={
                    "split": split,
                    "app": app,
                    "source_screen": f"{app}/iOS/0",
                    "target_screen": f"{app}/Android/0",
                    "source_screenshot_path": "",
                    "target_screenshot_path": "",
                    "source_xml_path": "",
                    "target_xml_path": "",
                },
            )
        )
    records = adapt_ase_queries(queries)

    manifest = write_mapping_dataset(records, tmp_path / "dataset")

    assert manifest["audit"]["split_counts"] == {"test": 1, "train": 1}
    train = json.loads((tmp_path / "dataset" / "train.jsonl").read_text().strip())
    test = json.loads((tmp_path / "dataset" / "test.jsonl").read_text().strip())
    assert train.keys() == test.keys()
    assert train["schema_version"] == MAPPING_PAGE_PAIR_SCHEMA
