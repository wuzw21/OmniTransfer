import json
from pathlib import Path

import pytest

from omnitransfer.mapping_dataset import (
    UI_CORRESPONDENCE_PAIR_SCHEMA,
    adapt_ase_queries,
    adapt_bmoca_trace_corpus,
    adapt_gui_odyssey_human_reviews,
    adapt_gui_odyssey_rows,
    audit_ui_correspondence_pairs,
    iter_split_ui_correspondence_pool,
    plan_ui_correspondence_pool,
    validate_ui_correspondence_pair,
    write_ui_correspondence_pool,
    write_ui_correspondence_dataset,
)
from omnitransfer.schema import Candidate, Query


def test_ui_correspondence_schema_accepts_multiple_matches_for_train_and_test() -> None:
    base = {
        "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
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

    train = validate_ui_correspondence_pair({**base, "split": "train"})
    test = validate_ui_correspondence_pair({**base, "split": "test"})

    assert train.keys() == test.keys()
    assert len(train["matches"]) == 2
    assert train["matches"][0]["target_node_ids"] == ["target-a", "target-b"]

    self_supervised_test = validate_ui_correspondence_pair(
        {**base, "split": "test", "label_status": "self_supervised"}
    )
    assert self_supervised_test["label_status"] == "self_supervised"
    with pytest.raises(ValueError, match="unsupported labels"):
        validate_ui_correspondence_pair(
            {**base, "split": "test", "label_status": "unreviewed"}
        )


def test_legacy_schema_is_read_but_normalized_to_ui_correspondence() -> None:
    record = {
        "schema_version": "omnitransfer.mapping_page_pair.v1",
        "pair_id": "legacy-pair",
        "split": "train",
        "label_status": "gold",
        "source": {
            "page_id": "source",
            "platform": "ios",
            "screenshot_path": "",
            "graph": {
                "graph_id": "source",
                "nodes": [{"node_id": "s", "origin_id": "s"}],
            },
        },
        "target": {
            "page_id": "target",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "graph_id": "target",
                "nodes": [{"node_id": "t", "origin_id": "t"}],
            },
        },
        "matches": [
            {
                "source_node_id": "s",
                "target_node_ids": ["t"],
                "label": "correspondence",
            }
        ],
        "partition_keys": ["legacy:app"],
        "provenance": {"dataset": "legacy"},
        "slices": {},
    }

    normalized = validate_ui_correspondence_pair(record)

    assert normalized["schema_version"] == UI_CORRESPONDENCE_PAIR_SCHEMA


def test_ase_queries_are_grouped_into_one_multi_match_ui_correspondence() -> None:
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
            target_candidates=(Candidate("target-b", bbox=(0.6, 0.8, 0.8, 0.9)),),
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
    source_nodes = {
        node["node_id"]: node for node in record["source"]["graph"]["nodes"]
    }
    assert source_nodes["source-a"]["clickable"] is True
    assert (
        source_nodes["source-a"]["metadata"]["actionability_evidence"]
        == "ase_public_mapping_source"
    )


def test_bmoca_trace_corpus_groups_actions_and_binds_full_xml_nodes(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "bmoca"
    source_trace = corpus / "traces" / "trace-source"
    target_trace = corpus / "traces" / "trace-target"
    source_trace_2 = corpus / "traces" / "trace-source-2"
    target_trace_2 = corpus / "traces" / "trace-target-2"
    source_trace.mkdir(parents=True)
    target_trace.mkdir(parents=True)
    source_trace_2.mkdir(parents=True)
    target_trace_2.mkdir(parents=True)
    source_xml = """
    <hierarchy width="1080" height="1920">
      <android.widget.FrameLayout bounds="[0,0][1080,1920]">
        <android.widget.Button text="ALPHA" resource-id="pkg:id/alpha"
          clickable="true" bounds="[10,20][210,120]" />
        <android.view.View resource-id="pkg:id/image_preview"
          clickable="false" bounds="[0,200][1080,1600]" />
        <android.view.View resource-id="pkg:id/overlay_view"
          clickable="true" bounds="[0,200][1080,1600]" />
        <android.widget.SeekBar content-desc="Media volume"
          resource-id="android:id/seekbar" clickable="false"
          bounds="[100,1650][980,1750]" />
      </android.widget.FrameLayout>
    </hierarchy>
    """
    target_xml = source_xml.replace("ALPHA", "Alpha").replace(
        "[10,20][210,120]", "[20,30][220,130]"
    )
    (source_trace / "transfer_states.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "run-source",
                "states": {
                    "source-page": {
                        "state_id": "source-page",
                        "xml": source_xml,
                        "package_name": "pkg",
                        "activity_name": "Main",
                        "display": {"width": 1080, "height": 1920},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (target_trace / "transfer_states.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "run-target",
                "states": {
                    "target-page": {
                        "state_id": "target-page",
                        "xml": target_xml,
                        "package_name": "pkg",
                        "activity_name": "Main",
                        "display": {"width": 1080, "height": 1920},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (source_trace_2 / "transfer_states.json").write_text(
        (source_trace / "transfer_states.json")
        .read_text(encoding="utf-8")
        .replace("run-source", "run-source-2"),
        encoding="utf-8",
    )
    (target_trace_2 / "transfer_states.json").write_text(
        (target_trace / "transfer_states.json")
        .read_text(encoding="utf-8")
        .replace("run-target", "run-target-2"),
        encoding="utf-8",
    )
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.offline-trace-corpus.v1",
                "traces": [
                    {
                        "trace_id": "trace-source",
                        "task_id": "task/example",
                        "environment_id": "100",
                        "run_id": "run-source",
                        "state_catalog": {
                            "path": "traces/trace-source/transfer_states.json"
                        },
                    },
                    {
                        "trace_id": "trace-target",
                        "task_id": "task/example",
                        "environment_id": "101",
                        "run_id": "run-target",
                        "state_catalog": {
                            "path": "traces/trace-target/transfer_states.json"
                        },
                    },
                    {
                        "trace_id": "trace-source-2",
                        "task_id": "task/example",
                        "environment_id": "100",
                        "run_id": "run-source-2",
                        "state_catalog": {
                            "path": "traces/trace-source-2/transfer_states.json"
                        },
                    },
                    {
                        "trace_id": "trace-target-2",
                        "task_id": "task/example",
                        "environment_id": "101",
                        "run_id": "run-target-2",
                        "state_catalog": {
                            "path": "traces/trace-target-2/transfer_states.json"
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def endpoint(
        run_id: str,
        page_id: str,
        *,
        node_id: str,
        bounds: list[int],
        class_name: str,
        text: str = "",
        resource_id: str = "",
        clickable: bool = True,
    ) -> dict:
        return {
            "run_id": run_id,
            "state_id": page_id,
            "page_id": page_id,
            "action_tool": "click",
            "package_name": "pkg",
            "screenshot_path": f"{page_id}.png",
            "width": 1080,
            "height": 1920,
            "point": {"x": 100, "y": 70, "coordinate_space": "page_pixels"},
            "node": {
                "node_id": node_id,
                "bounds": bounds,
                "attributes": {
                    "class": class_name,
                    "text": text,
                    "content_description": "",
                    "resource_id": resource_id,
                    "clickable": clickable,
                    "enabled": True,
                },
            },
        }

    pairs = {
        "pair-alpha": {
            "pair_id": "pair-alpha",
            "bidirectional": True,
            "source": endpoint(
                "run-source-2",
                "source-page",
                node_id="e2",
                bounds=[10, 20, 210, 120],
                class_name="Button",
                text="alpha",
                resource_id="alpha",
            ),
            "target": endpoint(
                "run-target-2",
                "target-page",
                node_id="e2",
                bounds=[20, 30, 220, 130],
                class_name="Button",
                text="alpha",
                resource_id="alpha",
            ),
            "evidence": {"alignment_score": 0.95},
        },
        "pair-overlay": {
            "pair_id": "pair-overlay",
            "bidirectional": True,
            "source": endpoint(
                "run-source",
                "source-page",
                node_id="e4",
                bounds=[0, 200, 1080, 1600],
                class_name="View",
                resource_id="overlay_view",
            ),
            "target": endpoint(
                "run-target",
                "target-page",
                node_id="e4",
                bounds=[0, 200, 1080, 1600],
                class_name="View",
                resource_id="overlay_view",
            ),
            "evidence": {"alignment_score": 0.9},
        },
        "pair-slider": {
            "pair_id": "pair-slider",
            "bidirectional": True,
            "source": endpoint(
                "run-source",
                "source-page",
                node_id="e5",
                bounds=[100, 1650, 980, 1750],
                class_name="SeekBar",
                resource_id="seekbar",
                clickable=False,
            ),
            "target": endpoint(
                "run-target",
                "target-page",
                node_id="e5",
                bounds=[100, 1650, 980, 1750],
                class_name="SeekBar",
                resource_id="seekbar",
                clickable=False,
            ),
            "evidence": {"alignment_score": 0.92},
        },
    }
    (corpus / "pair_memory.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-pair-memory.v1",
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )

    records, report = adapt_bmoca_trace_corpus(corpus)

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == UI_CORRESPONDENCE_PAIR_SCHEMA
    assert record["split"] == "diagnostic"
    assert record["label_status"] == "unreviewed"
    assert record["provenance"]["annotation"] == "offline_trace_dp_alignment_unreviewed"
    assert len(record["matches"]) == 3
    source_nodes = {
        node["node_id"]: node for node in record["source"]["graph"]["nodes"]
    }
    target_nodes = {
        node["node_id"]: node for node in record["target"]["graph"]["nodes"]
    }
    for match in record["matches"]:
        assert source_nodes[match["source_node_id"]]["clickable"] is True
        assert all(
            target_nodes[node_id]["clickable"] for node_id in match["target_node_ids"]
        )
    overlay_match = next(
        match
        for match in record["matches"]
        if source_nodes[match["source_node_id"]]["resource_id"].endswith("overlay_view")
    )
    assert all(
        target_nodes[node_id]["resource_id"].endswith("overlay_view")
        for node_id in overlay_match["target_node_ids"]
    )
    assert report["input_pairs"] == 3
    assert report["output_records"] == 1
    assert report["output_correspondences"] == 3
    assert report["unresolved_endpoints"] == 0
    assert report["executed_nodes_marked_actionable"] == 2


def test_ase_adapter_materializes_relative_xml_into_the_page_pair(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "bundle" / "assets"
    assets.mkdir(parents=True)
    source_xml = assets / "source.xml"
    target_xml = assets / "target.xml"
    source_xml.write_text(
        '<hierarchy><node index="0" text="Search" class="Button" '
        'clickable="true" bounds="[0,0][20,20]" /></hierarchy>',
        encoding="utf-8",
    )
    target_xml.write_text(
        '<hierarchy><node index="0" content-desc="Search" class="Button" '
        'clickable="true" bounds="[10,10][30,30]" /></hierarchy>',
        encoding="utf-8",
    )
    query = Query(
        query_id="relative-assets",
        source={
            "node_id": "0",
            "text": "Search",
            "class_name": "Button",
            "clickable": True,
        },
        target_candidates=(
            Candidate(
                "0",
                metadata={
                    "content_desc": "Search",
                    "class_name": "Button",
                    "clickable": True,
                },
            ),
        ),
        gold_candidate_id="0",
        metadata={
            "app": "Example",
            "split": "train",
            "source_screen": "Example/iOS/0",
            "target_screen": "Example/Android/0",
            "source_xml_path": "bundle/assets/source.xml",
            "target_xml_path": "bundle/assets/target.xml",
            "source_screenshot_path": "",
            "target_screenshot_path": "",
        },
    )

    record = adapt_ase_queries([query], asset_root=tmp_path)[0]

    assert record["source"]["graph"]["metadata"]["source_format"] == "xml"
    assert record["target"]["graph"]["metadata"]["source_format"] == "xml"
    assert any(node["text"] == "Search" for node in record["source"]["graph"]["nodes"])
    assert any(
        node["content_desc"] == "Search" for node in record["target"]["graph"]["nodes"]
    )


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
                    "sam2_bbox": [
                        point[0] - 50,
                        point[1] - 50,
                        point[0] + 50,
                        point[1] + 50,
                    ],
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
    assert record["schema_version"] == UI_CORRESPONDENCE_PAIR_SCHEMA
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
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
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
        audit_ui_correspondence_pairs(rows)


def test_diagnostic_uses_its_frozen_reserved_split_for_leakage_audit() -> None:
    def record(pair_id: str, split: str, reserved_split: str | None) -> dict:
        provenance = {"dataset": "guiodyssey", "annotation": "test"}
        if reserved_split is not None:
            provenance["reserved_split"] = reserved_split
        return {
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
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

    audit = audit_ui_correspondence_pairs(
        [record("candidate", "diagnostic", "dev"), record("gold", "dev", None)]
    )
    assert audit["assignment_counts"] == {"dev": 2}

    with pytest.raises(ValueError, match="partition leakage"):
        audit_ui_correspondence_pairs(
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
                    {
                        "source_node_id": "source-search",
                        "target_node_id": "target-search",
                    },
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

    manifest = write_ui_correspondence_dataset(records, tmp_path / "dataset")

    assert manifest["audit"]["split_counts"] == {"test": 1, "train": 1}
    train = json.loads((tmp_path / "dataset" / "train.jsonl").read_text().strip())
    test = json.loads((tmp_path / "dataset" / "test.jsonl").read_text().strip())
    assert train.keys() == test.keys()
    assert train["schema_version"] == UI_CORRESPONDENCE_PAIR_SCHEMA


def test_writer_streams_single_pass_records_and_cleans_leakage_parts(
    tmp_path: Path,
) -> None:
    def record(pair_id: str, split: str, app: str) -> dict:
        return {
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": split,
            "label_status": "gold",
            "source": {
                "page_id": f"{pair_id}:source",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"{pair_id}:source",
                    "nodes": [{"node_id": "source", "origin_id": "source"}],
                },
            },
            "target": {
                "page_id": f"{pair_id}:target",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"{pair_id}:target",
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
            "partition_keys": [f"app:{app}"],
            "provenance": {"dataset": "mobileviews", "annotation": "gold"},
            "slices": {},
        }

    consumed: list[str] = []

    def records():
        for row in (
            record("train-pair", "train", "train"),
            record("test-pair", "test", "test"),
        ):
            consumed.append(row["pair_id"])
            yield row

    manifest = write_ui_correspondence_dataset(records(), tmp_path / "streamed")

    assert consumed == ["train-pair", "test-pair"]
    assert manifest["audit"]["records"] == 2
    assert manifest["files"]["train.jsonl"]["records"] == 1
    assert manifest["files"]["test.jsonl"]["records"] == 1

    leaking = tmp_path / "leaking"
    with pytest.raises(ValueError, match="partition leakage"):
        write_ui_correspondence_dataset(
            (
                row
                for row in (
                    record("train-pair", "train", "shared"),
                    record("test-pair", "test", "shared"),
                )
            ),
            leaking,
        )
    assert not list(leaking.glob("*.part"))


def test_unified_pool_is_split_within_app_without_page_leakage(
    tmp_path: Path,
) -> None:
    def record(index: int, *, label_status: str = "gold") -> dict:
        pair_id = f"same-app-pair-{index}"
        return {
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
            "pair_id": pair_id,
            "split": "diagnostic" if label_status == "unreviewed" else "train",
            "label_status": label_status,
            "source": {
                "page_id": f"same-app-source-{index}",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"same-app-source-{index}",
                    "nodes": [{"node_id": "source", "origin_id": "source"}],
                },
            },
            "target": {
                "page_id": f"same-app-target-{index}",
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": f"same-app-target-{index}",
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
            "partition_keys": ["mobileviews:app:shared"],
            "provenance": {
                "dataset": "MobileViews_Apps_CompleteTraces",
                "annotation": "automatic_same_view_str_proposal",
                "trace": "shared.app.apk",
            },
            "slices": {"app": "shared.app"},
        }

    pool_path = tmp_path / "pool.jsonl"
    pool_manifest = write_ui_correspondence_pool(
        [record(index) for index in range(10)],
        pool_path,
    )
    plan = plan_ui_correspondence_pool(
        pool_path,
        seed=17,
        train_percent=80,
        dev_percent=10,
    )
    split_records = list(iter_split_ui_correspondence_pool(pool_path, plan))

    assert pool_manifest["records"] == 10
    assert {row["slices"]["app"] for row in split_records} == {"shared.app"}
    assert {plan.assignments[row["pair_id"]] for row in split_records} == {
        "train",
        "dev",
        "test",
    }
    assert plan.audit["app_protocol"] == "within_app_page_component"
    assert plan.audit["overlap"]["page_id"] == {}
    assert plan.audit["apps_in_multiple_splits"] == ["shared.app"]
    assert {
        row["split"]
        for row in split_records
        if plan.assignments[row["pair_id"]] in {"dev", "test"}
    } == {"dev", "test"}
    audit_ui_correspondence_pairs(split_records)


@pytest.mark.parametrize(
    ("source_split", "source_label_status"),
    [("diagnostic", "unreviewed"), ("train", "self_supervised")],
)
def test_mobileviews_pool_keeps_connected_pages_together_across_splits(
    tmp_path: Path,
    source_split: str,
    source_label_status: str,
) -> None:
    def record(index: int, source_page: str, target_page: str) -> dict:
        return {
            "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
            "pair_id": f"proposal-{index}",
            "split": source_split,
            "label_status": source_label_status,
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
                "page_id": target_page,
                "platform": "android",
                "screenshot_path": "",
                "graph": {
                    "graph_id": target_page,
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
            "partition_keys": ["mobileviews:trace:shared.app"],
            "provenance": {
                "dataset": "MobileViews_Apps_CompleteTraces",
                "annotation": "automatic_same_view_str_proposal",
                "trace": "shared.app",
            },
            "slices": {},
        }

    pool_path = tmp_path / "pool.jsonl"
    rows = [
        record(0, "page-a", "page-b"),
        record(1, "page-b", "page-c"),
        *(
            record(index, f"page-{index}-a", f"page-{index}-b")
            for index in range(2, 10)
        ),
    ]
    write_ui_correspondence_pool(rows, pool_path)
    plan = plan_ui_correspondence_pool(pool_path)
    split_records = list(iter_split_ui_correspondence_pool(pool_path, plan))

    assert plan.assignments["proposal-0"] == plan.assignments["proposal-1"]
    for row in split_records:
        assigned = plan.assignments[row["pair_id"]]
        assert row["split"] == assigned
        assert row["label_status"] == "self_supervised"
        assert "reserved_split" not in row["provenance"]
    audit = audit_ui_correspondence_pairs(split_records)
    for split in {"train", "dev", "test"}:
        assert audit["split_label_status_counts"][split] == {
            "self_supervised": sum(
                assigned == split for assigned in plan.assignments.values()
            )
        }


def test_within_app_pool_split_can_disable_dev(tmp_path: Path) -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "schema_version": UI_CORRESPONDENCE_PAIR_SCHEMA,
                "pair_id": f"pair-{index}",
                "split": "train",
                "label_status": "gold",
                "source": {
                    "page_id": f"source-{index}",
                    "platform": "android",
                    "screenshot_path": "",
                    "graph": {
                        "graph_id": f"source-{index}",
                        "nodes": [{"node_id": "source", "origin_id": "source"}],
                    },
                },
                "target": {
                    "page_id": f"target-{index}",
                    "platform": "android",
                    "screenshot_path": "",
                    "graph": {
                        "graph_id": f"target-{index}",
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
                "partition_keys": ["app:shared"],
                "provenance": {"dataset": "example"},
                "slices": {"app": "shared"},
            }
        )
    pool_path = tmp_path / "pool.jsonl"
    write_ui_correspondence_pool(rows, pool_path)

    plan = plan_ui_correspondence_pool(
        pool_path,
        train_percent=80,
        dev_percent=0,
    )

    assert set(plan.assignments.values()) == {"train", "test"}
