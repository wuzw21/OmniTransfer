import math
import random

import pytest

from omnitransfer.learned_matcher import (
    LearnedGraphMatcher,
    MatcherConfig,
    NUMERIC_FEATURE_DIM,
    RELATION_FEATURE_DIM,
    build_relation_aware_matcher,
    encode_graph,
    matcher_inputs,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.self_supervised import AugmentConfig, make_training_pair, matching_loss
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record


def _graph(graph_id: str, *, delete_y: int = 10):
    return graph_from_record(
        {
            "screen_id": graph_id,
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "delete",
                    "parent_id": "root",
                    "text": "Delete item",
                    "resource-id": "app:id/delete_item",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [5, delete_y, 45, delete_y + 20],
                },
                {
                    "node_id": "archive",
                    "parent_id": "root",
                    "text": "Archive item",
                    "resource-id": "app:id/archive_item",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [55, 10, 95, 30],
                },
                {
                    "node_id": "label",
                    "parent_id": "root",
                    "text": "Item 1",
                    "class": "TextView",
                    "bounds": [5, 40, 95, 60],
                },
            ],
        }
    )


def test_graph_encoder_exposes_learned_tokens_and_relations() -> None:
    encoded = encode_graph(_graph("source"))

    assert len(encoded.numeric_features[0]) == NUMERIC_FEATURE_DIM
    assert len(encoded.relation_features[0][0]) == RELATION_FEATURE_DIM
    assert encoded.token_ids[1] != encoded.token_ids[2]
    assert encoded.relation_features[0][1][1] == 1.0
    assert encoded.relation_features[1][2][3] == 1.0
    assert encoded.relation_features[1][2][7] == 0.0


def test_relation_matcher_forward_and_partial_assignment_loss() -> None:
    torch = pytest.importorskip("torch")
    config = MatcherConfig(hidden_dim=64, num_heads=4, num_layers=1, dropout=0.0)
    graph = _graph("train")
    pair = make_training_pair(
        graph,
        rng=random.Random(9),
        config=AugmentConfig(
            drop_node_prob=0.0,
            distractor_prob=1.0,
            max_distractors=1,
            bbox_jitter=0.0,
            global_translation=0.0,
            global_scale=0.0,
        ),
        matcher_config=config,
    )
    assert pair is not None
    model = build_relation_aware_matcher(config)

    output = model(*matcher_inputs(pair.graph_a, pair.graph_b, config=config))
    loss = matching_loss(model, pair, matcher_config=config)

    assert output["logits_ab"].shape == (
        len(pair.graph_a.nodes),
        len(pair.graph_b.nodes) + 1,
    )
    assert output["logits_ba"].shape == (
        len(pair.graph_b.nodes),
        len(pair.graph_a.nodes) + 1,
    )
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert parameter_count(model) < 1_500_000
    assert any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())


def test_inference_adapter_ignores_null_logit() -> None:
    torch = pytest.importorskip("torch")
    source = _graph("source")
    target = _graph("target", delete_y=60)

    class NullModel(torch.nn.Module):
        def forward(self, *inputs):
            source_count = inputs[0].shape[0]
            target_count = inputs[3].shape[0]
            logits_ab = torch.zeros((source_count, target_count + 1))
            logits_ab[:, 1] = 2.0
            logits_ab[:, -1] = 10.0
            logits_ba = torch.zeros((target_count, source_count + 1))
            logits_ba[:, -1] = 10.0
            return {"logits_ab": logits_ab, "logits_ba": logits_ba}

    result = LearnedGraphMatcher(NullModel()).predict(
        source,
        target,
        source_node_id="delete",
    )

    assert result.target_node is not None
    assert result.target_node.node_id == "delete"
    assert result.reason == "learned_match"
    assert all(candidate_id != "__NULL__" for candidate_id, _ in result.scores)


def test_visual_encoder_is_jointly_trained_when_screenshots_exist(tmp_path) -> None:
    pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    screenshot = tmp_path / "screen.png"
    image_module.new("RGB", (100, 100), color=(120, 80, 40)).save(screenshot)
    source_base = _graph("visual-source")
    target_base = _graph("visual-target", delete_y=60)
    source = UIGraph(
        **{
            **source_base.__dict__,
            "metadata": {"screenshot_path": str(screenshot)},
        }
    )
    target = UIGraph(
        **{
            **target_base.__dict__,
            "metadata": {"screenshot_path": str(screenshot)},
        }
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    model = build_relation_aware_matcher(config)
    inputs = matcher_inputs(source, target, config=config)

    output = model(*inputs)
    output["logits_ab"].mean().backward()

    assert len(inputs) == 11
    assert inputs[8].sum().item() == len(source.nodes)
    assert inputs[10].sum().item() == len(target.nodes)
    assert model.visual_encoder[0].weight.grad is not None


def test_visual_crop_follows_element_identity_during_layout_augmentation(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    screenshot = tmp_path / "screen.png"
    image = image_module.new("RGB", (100, 100), color=(0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 50, 100))
    image.save(screenshot)
    node = UINode(
        node_id="moved",
        origin_id="element",
        bbox=(50.0, 0.0, 100.0, 100.0),
        metadata={"visual_bbox": (0.0, 0.0, 50.0, 100.0)},
    )
    graph = UIGraph(
        graph_id="augmented",
        nodes=(node,),
        width=100,
        height=100,
        metadata={"screenshot_path": str(screenshot)},
    )

    source_visual = matcher_inputs(graph, graph)[7]

    assert torch.mean(source_visual[0, 0]) > 0.9
    assert torch.mean(source_visual[0, 2]) < 0.1


def test_train_only_visual_transform_changes_local_appearance(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    screenshot = tmp_path / "screen.png"
    image_module.new("RGB", (20, 20), color=(100, 100, 100)).save(screenshot)
    node = UINode(node_id="node", origin_id="node", bbox=(0, 0, 20, 20))
    plain = UIGraph(
        graph_id="plain",
        nodes=(node,),
        width=20,
        height=20,
        metadata={"screenshot_path": str(screenshot)},
    )
    augmented = UIGraph(
        graph_id="augmented",
        nodes=(node,),
        width=20,
        height=20,
        metadata={
            "screenshot_path": str(screenshot),
            "visual_transform": {
                "brightness": 0.1,
                "contrast": 1.0,
                "channel_scale": [1.0, 1.0, 1.0],
            },
        },
    )

    plain_visual = matcher_inputs(plain, plain)[7]
    augmented_visual = matcher_inputs(augmented, augmented)[7]

    assert torch.mean(augmented_visual) > torch.mean(plain_visual)


def test_visual_checkpoint_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    model = build_relation_aware_matcher(config)
    checkpoint = tmp_path / "matcher.pt"

    save_matcher_checkpoint(checkpoint, model, config=config, metadata={"mock": True})
    restored = LearnedGraphMatcher.from_checkpoint(checkpoint)

    assert restored.config == config
    assert parameter_count(restored.model) == parameter_count(model)
