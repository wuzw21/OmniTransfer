import math
import random

import pytest

from omnitransfer.learned_matcher import (
    RelationAwareMatcher,
    MatcherConfig,
    NUMERIC_FEATURE_DIM,
    RELATION_FEATURE_DIM,
    build_relation_aware_matcher,
    cross_relation_features,
    encode_graph,
    matcher_inputs,
    mutual_log_assignment,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.self_supervised import (
    AugmentConfig,
    make_training_pair,
    matching_loss,
)
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


def test_node_encoding_ignores_resource_id_and_absolute_position() -> None:
    source = graph_from_record(
        {
            "screen_id": "source",
            "width": 100,
            "height": 100,
            "nodes": [
                {
                    "node_id": "action",
                    "text": "Search",
                    "content-desc": "Find items",
                    "resource-id": "source:id/search",
                    "class": "EditText",
                    "clickable": True,
                    "editable": True,
                    "bounds": [0, 0, 40, 20],
                }
            ],
        }
    )
    target = graph_from_record(
        {
            "screen_id": "target",
            "width": 100,
            "height": 100,
            "nodes": [
                {
                    "node_id": "action",
                    "text": "Search",
                    "content-desc": "Find items",
                    "resource-id": "target:id/completely_different",
                    "class": "EditText",
                    "clickable": True,
                    "editable": True,
                    "bounds": [55, 70, 95, 90],
                }
            ],
        }
    )

    encoded_source = encode_graph(source)
    encoded_target = encode_graph(target)

    assert encoded_source.token_ids == encoded_target.token_ids
    assert encoded_source.numeric_features == encoded_target.numeric_features
    assert not cross_relation_features(source, target).any()


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
        len(pair.graph_b.nodes),
    )
    assert output["logits_ba"].shape == (
        len(pair.graph_b.nodes),
        len(pair.graph_a.nodes),
    )
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert parameter_count(model) < 1_500_000
    assert any(
        isinstance(module, torch.nn.MultiheadAttention) for module in model.modules()
    )


def test_mutual_log_assignment_normalizes_both_matching_directions() -> None:
    torch = pytest.importorskip("torch")
    affinity = torch.tensor(
        [[4.0, 1.0, -1.0], [2.0, 3.0, 0.0]],
        requires_grad=True,
    )

    scores = mutual_log_assignment(affinity)
    expected = 0.5 * (
        torch.log_softmax(affinity, dim=1)
        + torch.log_softmax(affinity, dim=0)
    )

    assert torch.allclose(scores, expected)
    scores.sum().backward()
    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()


def test_mutual_projection_reports_one_assignment_per_attention_layer() -> None:
    torch = pytest.importorskip("torch")
    config = MatcherConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        assignment_head="mutual_projection",
    )
    source = _graph("source")
    target = _graph("target")
    model = build_relation_aware_matcher(config)

    output = model(*matcher_inputs(source, target, config=config))

    assert len(output["assignment_scores_by_layer"]) == 2
    assert len(output["affinities_by_layer"]) == 2
    assert torch.equal(
        output["logits_ab"],
        output["assignment_scores_by_layer"][-1],
    )
    assert torch.equal(output["logits_ba"], output["logits_ab"].T)
    assert torch.equal(output["affinity"], output["affinities_by_layer"][-1])


def test_context_masking_input_keeps_class_action_and_relations(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    screenshot = tmp_path / "screen.png"
    image_module.new("RGB", (100, 100), color=(120, 80, 40)).save(screenshot)
    base = _graph("masked-source")
    source = UIGraph(
        **{
            **base.__dict__,
            "metadata": {"screenshot_path": str(screenshot)},
        }
    )
    target = _graph("target")
    plain = matcher_inputs(source, target)

    masked = matcher_inputs(
        source,
        target,
        source_context_mask_indices=(1,),
    )

    assert not torch.equal(masked[0][1], plain[0][1])
    assert torch.equal(masked[0][2], plain[0][2])
    assert torch.count_nonzero(masked[0][1]) > 0
    assert torch.equal(masked[1], plain[1])
    assert torch.equal(masked[2], plain[2])
    assert masked[8][1].item() == 0.0
    assert plain[8][1].item() == 1.0
    assert torch.equal(masked[3], plain[3])


def test_inference_adapter_rejects_low_pair_confidence() -> None:
    torch = pytest.importorskip("torch")
    source = _graph("source")
    target = _graph("target", delete_y=60)

    class LowConfidenceModel(torch.nn.Module):
        def forward(self, *inputs):
            source_count = inputs[0].shape[0]
            target_count = inputs[3].shape[0]
            logits_ab = torch.zeros((source_count, target_count))
            logits_ab[:, 1] = 2.0
            logits_ba = torch.zeros((target_count, source_count))
            affinity = torch.full((source_count, target_count), -10.0)
            return {
                "logits_ab": logits_ab,
                "logits_ba": logits_ba,
                "affinity": affinity,
            }

    result = RelationAwareMatcher(LowConfidenceModel()).predict(
        source,
        target,
        source_node_id="delete",
        min_probability=0.5,
    )

    assert result.target_node is None
    assert result.reason == "learned_low_confidence"
    assert result.probability < 0.5


def test_inference_ranks_only_actionable_target_nodes() -> None:
    torch = pytest.importorskip("torch")
    source = _graph("source")
    target = _graph("target")

    class ContextBiasedModel(torch.nn.Module):
        def forward(self, *inputs):
            source_count = inputs[0].shape[0]
            target_count = inputs[3].shape[0]
            logits_ab = torch.zeros((source_count, target_count))
            logits_ab[:, 3] = 100.0
            logits_ab[:, 1] = 10.0
            logits_ba = torch.zeros((target_count, source_count))
            return {
                "logits_ab": logits_ab,
                "logits_ba": logits_ba,
                "affinity": logits_ab,
            }

    result = RelationAwareMatcher(ContextBiasedModel()).predict(
        source,
        target,
        source_node_id="delete",
    )

    assert result.target_node is not None
    assert result.target_node.node_id == "delete"


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


def test_visual_crop_follows_element_identity_during_layout_augmentation(
    tmp_path,
) -> None:
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
    torch = pytest.importorskip("torch")
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    model = build_relation_aware_matcher(config)
    checkpoint = tmp_path / "matcher.pt"

    save_matcher_checkpoint(checkpoint, model, config=config, metadata={"mock": True})
    restored = RelationAwareMatcher.from_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")

    assert (
        payload["schema_version"]
        == "omnitransfer.relation_aware_cross_attention_matcher.v1"
    )
    assert restored.config == config
    assert parameter_count(restored.model) == parameter_count(model)
