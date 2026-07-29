from __future__ import annotations

import pytest

from omnitransfer.benchmark import matcher_inputs
from omnitransfer.learned_matcher import MatcherConfig, PEMM_V3_FEATURE_SCHEMA_ID
from omnitransfer.mutual_matcher import (
    MutualGraphMatcher,
    build_mutual_assignment_matcher,
    mutual_assignment_logits,
    save_mutual_matcher_checkpoint,
)
from omnitransfer.numpy_matcher import (
    NumpyMutualGraphMatcher,
    save_numpy_mutual_matcher_checkpoint,
)
from omnitransfer.ui_graph import UIGraph, UINode


torch = pytest.importorskip("torch", exc_type=ImportError)


def test_single_matrix_produces_bidirectional_correspondence() -> None:
    affinity = torch.tensor([[6.0, 0.0], [0.0, 6.0]])

    logits_ab, logits_ba = mutual_assignment_logits(affinity)

    assert logits_ab.shape == (2, 2)
    assert logits_ba.shape == (2, 2)
    assert logits_ab.argmax(dim=1).tolist() == [0, 1]
    assert logits_ba.argmax(dim=1).tolist() == [0, 1]
    assert torch.allclose(logits_ab, logits_ba.T)


def test_assignment_matrix_has_no_null_column() -> None:
    logits_ab, logits_ba = mutual_assignment_logits(torch.zeros((1, 1)))

    assert logits_ab.shape == (1, 1)
    assert logits_ba.shape == (1, 1)


def test_model_forward_preserves_one_matrix_bidirectional_invariant() -> None:
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )
    source = _graph("source", ("search", "settings"))
    target = _graph("target", ("settings", "search"))

    model = build_mutual_assignment_matcher(config)
    output = model(*matcher_inputs(source, target, config=config, device="cpu"))

    assert output["logits_ab"].shape == (2, 2)
    assert output["logits_ba"].shape == (2, 2)
    assert torch.isfinite(output["logits_ab"]).all()
    assert torch.isfinite(output["logits_ba"]).all()
    assert torch.allclose(
        output["logits_ab"],
        output["logits_ba"].T,
    )


def test_model_exposes_explicit_pair_feature_matrices() -> None:
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )
    source = _graph("source", ("search", "settings", "help"))
    target = _graph("target", ("settings", "help", "search"))

    model = build_mutual_assignment_matcher(config)
    output = model(*matcher_inputs(source, target, config=config, device="cpu"))

    feature_matrices = output["feature_matrices"]
    assert set(feature_matrices) == {
        "semantic",
        "visual",
        "attributes",
        "context",
        "geometry",
        "anchor",
        "visual_available",
    }
    for matrix in feature_matrices.values():
        assert matrix.shape == (len(source.nodes), len(target.nodes))
        assert torch.isfinite(matrix).all()
    assert torch.count_nonzero(feature_matrices["visual_available"]) == 0


def test_local_anchor_support_participates_in_final_affinity() -> None:
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )
    source = _sibling_graph("source", ("search", "settings", "help"))
    target = _sibling_graph("target", ("settings", "help", "search"))
    model = build_mutual_assignment_matcher(config)

    output = model(*matcher_inputs(source, target, config=config, device="cpu"))
    anchor = output["feature_matrices"]["anchor"]
    output["affinity"].sum().backward()

    assert torch.count_nonzero(anchor) > 0
    assert model.anchor_edge_score[0].weight.grad is not None
    assert torch.count_nonzero(model.anchor_edge_score[0].weight.grad) > 0


def test_visual_crop_evidence_has_an_explicit_trainable_matrix(tmp_path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    source_screenshot = tmp_path / "source.png"
    target_screenshot = tmp_path / "target.png"
    image_module.new("RGB", (100, 100), color=(220, 40, 30)).save(source_screenshot)
    image_module.new("RGB", (100, 100), color=(30, 80, 220)).save(target_screenshot)
    source_base = _graph("source", ("search", "settings"))
    target_base = _graph("target", ("settings", "search"))
    source = UIGraph(
        **{
            **source_base.__dict__,
            "metadata": {"screenshot_path": str(source_screenshot)},
        }
    )
    target = UIGraph(
        **{
            **target_base.__dict__,
            "metadata": {"screenshot_path": str(target_screenshot)},
        }
    )
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
    )
    model = build_mutual_assignment_matcher(config)

    output = model(*matcher_inputs(source, target, config=config, device="cpu"))
    feature_matrices = output["feature_matrices"]
    feature_matrices["visual"].sum().backward()

    assert torch.count_nonzero(feature_matrices["visual_available"]) == 4
    assert torch.count_nonzero(feature_matrices["visual"]) > 0
    assert model.node_encoder.visual_encoder[0].weight.grad is not None


def test_mutual_checkpoint_round_trip_is_independent(tmp_path) -> None:
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )
    source = _graph("source", ("search", "settings"))
    target = _graph("target", ("settings", "search"))
    inputs = matcher_inputs(source, target, config=config, device="cpu")
    model = build_mutual_assignment_matcher(config).eval()
    expected = model(*inputs)["logits_ab"]
    checkpoint = tmp_path / "mutual.pt"

    save_mutual_matcher_checkpoint(
        checkpoint,
        model,
        config=config,
        metadata={"label": "round_trip"},
    )
    loaded = MutualGraphMatcher.from_checkpoint(checkpoint, device="cpu")
    actual = loaded.model(*inputs)["logits_ab"]

    assert loaded.config == config
    assert torch.allclose(actual, expected)


def test_numpy_checkpoint_matches_pytorch_xml_inference(tmp_path) -> None:
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )
    source = _sibling_graph("source", ("search", "settings", "help"))
    target = _sibling_graph("target", ("settings", "help", "search"))
    model = build_mutual_assignment_matcher(config).eval()
    checkpoint = tmp_path / "mutual.npz"
    save_numpy_mutual_matcher_checkpoint(
        checkpoint,
        model.state_dict(),
        config=config,
    )

    expected = model(
        *matcher_inputs(
            source,
            target,
            config=config,
            device="cpu",
            feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
        )
    )["logits_ab"].detach()
    matcher = NumpyMutualGraphMatcher.from_checkpoint(checkpoint)
    actual = torch.from_numpy(matcher._forward(source, target)["logits_ab"])

    assert matcher.config == config
    assert torch.allclose(actual, expected, atol=3e-5, rtol=1e-5)


def _graph(graph_id: str, labels: tuple[str, ...]) -> UIGraph:
    return UIGraph(
        graph_id=graph_id,
        width=100,
        height=100,
        nodes=tuple(
            UINode(
                node_id=f"node_{index}",
                origin_id=f"node_{index}",
                text=label,
                class_name="android.widget.TextView",
                bbox=(10.0, 10.0 + 30.0 * index, 50.0, 30.0 + 30.0 * index),
            )
            for index, label in enumerate(labels)
        ),
    )


def _sibling_graph(graph_id: str, labels: tuple[str, ...]) -> UIGraph:
    root = UINode(
        node_id="root",
        origin_id="root",
        class_name="android.widget.LinearLayout",
        bbox=(0.0, 0.0, 100.0, 100.0),
        child_ids=tuple(f"node_{index}" for index in range(len(labels))),
    )
    children = tuple(
        UINode(
            node_id=f"node_{index}",
            origin_id=f"node_{index}",
            parent_id="root",
            text=label,
            class_name="android.widget.TextView",
            bbox=(10.0, 10.0 + 25.0 * index, 50.0, 30.0 + 25.0 * index),
        )
        for index, label in enumerate(labels)
    )
    return UIGraph(
        graph_id=graph_id,
        width=100,
        height=100,
        nodes=(root, *children),
    )
