import json
from pathlib import Path

from omnitransfer.gui_odyssey_pairs import (
    load_gui_odyssey_human_review_pairs,
    load_gui_odyssey_training_pairs,
    make_gui_odyssey_training_pairs,
)
from omnitransfer.learned_matcher import MatcherConfig, build_relation_aware_matcher
from omnitransfer.self_supervised import matching_loss


def _episode(
    episode_id: str,
    *,
    width: int,
    height: int,
    point: list[int],
    bbox: list[int],
) -> dict:
    return {
        "episode_id": episode_id,
        "device_info": {
            "w": width,
            "h": height,
            "device_name": "Pixel Fold" if width > height else "Small Phone",
        },
        "task_info": {
            "category": "General_Tool",
            "app": ["Setting"],
            "meta_task": "Open {} settings.",
        },
        "steps": [
            {
                "step": 3,
                "screenshot": f"{episode_id}_3.png",
                "action": "CLICK",
                "info": [point, point],
                "sam2_bbox": bbox,
                "low_level_instruction": "Open the Apps section.",
            }
        ],
    }


def test_balanced_pair_loads_as_scaled_cross_attention_training_pair(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    source_episode = _episode(
        "source-episode",
        width=720,
        height=1280,
        point=[500, 500],
        bbox=[400, 400, 600, 600],
    )
    target_episode = _episode(
        "target-episode",
        width=2208,
        height=1840,
        point=[250, 750],
        bbox=[200, 700, 300, 800],
    )
    for episode in (source_episode, target_episode):
        (annotations / f"{episode['episode_id']}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    pairs = tmp_path / "balanced_click_pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.guiodyssey_sequence_filtered_click_pair.v1",
                "pair_id": "pair-1",
                "source": {"episode_id": "source-episode", "step_index": 3},
                "target": {"episode_id": "target-episode", "step_index": 3},
                "alignment": {"step_score": 0.9, "episode_score": 0.8},
                "selection": {"candidate_only": True, "gold_label": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    training_pairs, manifest = load_gui_odyssey_training_pairs(
        pairs,
        annotations,
    )

    assert manifest["accepted_pairs"] == 1
    assert manifest["supervision"] == "weak_correspondence_not_benchmark_gold"
    assert manifest["pairs_with_both_screenshots"] == 0
    pair = training_pairs[0]
    source_target = next(node for node in pair.graph_a.nodes if node.node_id == "action_target")
    target_target = next(node for node in pair.graph_b.nodes if node.node_id == "action_target")
    assert source_target.bbox == (288.0, 512.0, 432.0, 768.0)
    assert target_target.bbox == (441.6, 1288.0, 662.4, 1472.0)
    source_index = pair.graph_a.nodes.index(source_target)
    target_index = pair.graph_b.nodes.index(target_target)
    assert pair.targets_a_to_b[source_index] == target_index
    assert pair.targets_b_to_a[target_index] == source_index
    assert sum(index >= 0 for index in pair.targets_a_to_b) == 1
    assert sum(index >= 0 for index in pair.targets_b_to_a) == 1


def test_split_rows_use_the_same_gui_odyssey_training_adapter(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for episode in (
        _episode(
            "source-episode",
            width=720,
            height=1280,
            point=[500, 500],
            bbox=[400, 400, 600, 600],
        ),
        _episode(
            "target-episode",
            width=2208,
            height=1840,
            point=[250, 750],
            bbox=[200, 700, 300, 800],
        ),
    ):
        (annotations / f"{episode['episode_id']}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    rows = [
        {
            "schema_version": "omniflow.guiodyssey_sequence_filtered_click_pair.v1",
            "pair_id": "split-pair",
            "source": {"episode_id": "source-episode", "step_index": 3},
            "target": {"episode_id": "target-episode", "step_index": 3},
            "selection": {"candidate_only": True, "gold_label": False},
        }
    ]

    training_pairs, manifest = make_gui_odyssey_training_pairs(rows, annotations)

    assert manifest["accepted_pairs"] == 1
    assert manifest["input_rows_read"] == 1
    assert training_pairs[0].origin_ids == ("action_target->action_target",)


def test_gui_odyssey_pair_runs_cross_attention_loss(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for episode in (
        _episode(
            "source-episode",
            width=720,
            height=1280,
            point=[500, 500],
            bbox=[400, 400, 600, 600],
        ),
        _episode(
            "target-episode",
            width=2208,
            height=1840,
            point=[250, 750],
            bbox=[200, 700, 300, 800],
        ),
    ):
        (annotations / f"{episode['episode_id']}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.guiodyssey_sequence_filtered_click_pair.v1",
                "pair_id": "pair-forward",
                "source": {"episode_id": "source-episode", "step_index": 3},
                "target": {"episode_id": "target-episode", "step_index": 3},
                "selection": {"candidate_only": True, "gold_label": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    training_pairs, _ = load_gui_odyssey_training_pairs(
        pairs,
        annotations,
        matcher_config=config,
    )
    model = build_relation_aware_matcher(config)

    loss = matching_loss(model, training_pairs[0], matcher_config=config)
    loss.backward()

    assert loss.isfinite().item() is True
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_human_review_loader_keeps_multi_point_gold_and_explicit_null(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "guiodyssey_node_reviewed.jsonl"
    base = {
        "schema_version": "omnitransfer_guiodyssey_sequence_node_alignment_v1",
        "source": {
            "width": 1000,
            "height": 2000,
            "device_name": "Medium Phone",
            "image_url": "screenshots/source.png",
        },
        "target": {
            "width": 2000,
            "height": 1000,
            "device_name": "Pixel Fold",
            "image_url": "screenshots/target.png",
        },
    }
    rows = [
        {
            **base,
            "pair_id": "positive",
            "annotation": {
                "status": "reviewed",
                "label": "correspondence",
                "source_nodes": [
                    {"node_id": "source-a", "point_normalized": [100, 200], "label": "Save"},
                    {"node_id": "source-b", "point_normalized": [300, 400], "label": "Cancel"},
                ],
                "target_nodes": [
                    {"node_id": "target-a", "point_normalized": [800, 200], "label": "Save"},
                    {"node_id": "target-b", "point_normalized": [600, 400], "label": "Cancel"},
                ],
                "matches": [
                    {"source_node_id": "source-a", "target_node_id": "target-a"},
                    {"source_node_id": "source-b", "target_node_id": "target-b"},
                ],
            },
        },
        {
            **base,
            "pair_id": "null",
            "annotation": {
                "status": "reviewed",
                "label": "no_correspondence",
                "source_nodes": [
                    {"node_id": "source-only", "point_normalized": [100, 200], "label": "Delete"}
                ],
                "target_nodes": [
                    {"node_id": "target-only", "point_normalized": [800, 200], "label": "Archive"}
                ],
                "matches": [],
            },
        },
        {
            **base,
            "pair_id": "ignored",
            "annotation": {
                "status": "reviewed",
                "label": "uncertain",
                "source_nodes": [],
                "target_nodes": [],
                "matches": [],
            },
        },
    ]
    review_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    pairs, manifest = load_gui_odyssey_human_review_pairs(review_path)

    assert len(pairs) == 2
    assert pairs[0].origin_ids == ("source-a->target-a", "source-b->target-b")
    assert pairs[1].targets_a_to_b == (-1,)
    assert pairs[1].targets_b_to_a == (-1,)
    assert manifest["labels"] == {"correspondence": 1, "no_correspondence": 1}
    assert manifest["skipped"] == {"label:uncertain": 1}
    assert manifest["supervision"] == "human_reviewed_benchmark_gold"
