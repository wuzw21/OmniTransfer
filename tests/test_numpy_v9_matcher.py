from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnitransfer.learned_matcher import (
    GeometricMatcher,
    MatcherConfig,
    build_geometric_v9_matcher,
    matcher_inputs,
    save_matcher_checkpoint,
)
from omnitransfer.numpy_v9_matcher import (
    NumpyGeometricAlignmentMatcher,
    save_numpy_unified_association_checkpoint,
)
from omnitransfer.ui_graph import graph_from_record

torch = pytest.importorskip("torch", exc_type=ImportError)


SOURCE_XML = """
<hierarchy bounds="[0,0][120,240]">
  <node class="android.view.ViewGroup" bounds="[0,0][120,240]">
    <node content-desc="返回" class="android.widget.Button" clickable="true"
          enabled="true" bounds="[5,10][35,45]" />
    <node content-desc="搜索" class="android.widget.Button" clickable="true"
          enabled="true" bounds="[45,10][75,45]" />
    <node content-desc="收藏" class="android.widget.Button" clickable="true"
          enabled="true" bounds="[85,10][115,45]" />
    <node text="咖啡" class="android.widget.TextView" enabled="true"
          bounds="[10,80][100,115]" />
  </node>
</hierarchy>
"""
TARGET_XML = SOURCE_XML.replace(
    'content-desc="搜索" class="android.widget.Button" clickable="true"\n'
    '          enabled="true" bounds="[45,10][75,45]"',
    'content-desc="收藏" class="android.widget.Button" clickable="true"\n'
    '          enabled="true" bounds="[45,10][75,45]"',
).replace(
    'content-desc="收藏" class="android.widget.Button" clickable="true"\n'
    '          enabled="true" bounds="[85,10][115,45]"',
    'content-desc="搜索" class="android.widget.Button" clickable="true"\n'
    '          enabled="true" bounds="[85,10][115,45]"',
)


def _export(path) -> None:
    torch.manual_seed(17)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        dropout=0.0,
    )
    model = build_geometric_v9_matcher(config).eval()
    torch_checkpoint = path.with_suffix(".pt")
    save_matcher_checkpoint(torch_checkpoint, model, config=config)
    save_numpy_unified_association_checkpoint(
        path,
        model.state_dict(),
        config=config,
    )


def test_v9_numpy_export_is_deterministic_and_pickle_free(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _export(first)
    _export(second)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as checkpoint:
        assert "__schema_version__" in checkpoint.files
        assert "association_layers.0.local.message_projection.weight" in checkpoint.files
        assert "page_attention.1.weight" in checkpoint.files


def test_v9_numpy_matches_pytorch_candidate_ranking(tmp_path: Path) -> None:
    exported = tmp_path / "v9.npz"
    _export(exported)
    source = graph_from_record({"xml": SOURCE_XML}, graph_id="source")
    target = graph_from_record({"xml": TARGET_XML}, graph_id="target")
    source_node = next(node for node in source.nodes if node.content_desc == "搜索")
    candidates = tuple(node.node_id for node in target.nodes if node.clickable)

    pytorch_match = GeometricMatcher.from_checkpoint(exported.with_suffix(".pt")).predict(
        source,
        target,
        source_node_id=source_node.node_id,
        candidate_node_ids=candidates,
    )
    numpy_match = NumpyGeometricAlignmentMatcher.from_checkpoint(exported).predict(
        source,
        target,
        source_node_id=source_node.node_id,
        candidate_node_ids=candidates,
    )

    assert numpy_match.target_node is not None
    assert pytorch_match.target_node is not None
    assert numpy_match.target_node.node_id == pytorch_match.target_node.node_id
    assert [node_id for node_id, _ in numpy_match.scores] == [
        node_id for node_id, _ in pytorch_match.scores
    ]
    np.testing.assert_allclose(
        [score for _, score in numpy_match.scores],
        [score for _, score in pytorch_match.scores],
        rtol=2e-4,
        atol=2e-4,
    )


def test_v9_numpy_predict_many_reuses_page_pair_forward(tmp_path: Path, monkeypatch) -> None:
    exported = tmp_path / "v9.npz"
    _export(exported)
    source = graph_from_record({"xml": SOURCE_XML}, graph_id="source")
    target = graph_from_record({"xml": TARGET_XML}, graph_id="target")
    source_nodes = [node for node in source.nodes if node.clickable]
    candidates = tuple(node.node_id for node in target.nodes if node.clickable)
    matcher = NumpyGeometricAlignmentMatcher.from_checkpoint(exported)
    original_forward = matcher._forward
    calls = 0

    def counted_forward(source_graph, target_graph):
        nonlocal calls
        calls += 1
        return original_forward(source_graph, target_graph)

    monkeypatch.setattr(matcher, "_forward", counted_forward)
    predictions = matcher.predict_many(
        source,
        target,
        source_node_ids=[node.node_id for node in source_nodes],
        candidate_node_ids=candidates,
    )

    assert calls == 1
    assert set(predictions) == {node.node_id for node in source_nodes}
    for node in source_nodes:
        single = matcher.predict(
            source,
            target,
            source_node_id=node.node_id,
            candidate_node_ids=candidates,
        )
        batched = predictions[node.node_id]
        batched_id = batched.target_node.node_id if batched.target_node else None
        single_id = single.target_node.node_id if single.target_node else None
        assert batched_id == single_id
        np.testing.assert_allclose(
            [score for _, score in batched.scores],
            [score for _, score in single.scores],
            rtol=2e-4,
            atol=2e-4,
        )


def test_v9_numpy_page_embedding_matches_contextual_torch_readout(tmp_path: Path) -> None:
    exported = tmp_path / "v9.npz"
    _export(exported)
    graph = graph_from_record({"xml": SOURCE_XML}, graph_id="page")
    torch_matcher = GeometricMatcher.from_checkpoint(exported.with_suffix(".pt"))
    numpy_matcher = NumpyGeometricAlignmentMatcher.from_checkpoint(exported)

    with torch.inference_mode():
        expected = torch_matcher.model(
            *matcher_inputs(graph, graph, config=torch_matcher.config)
        )["source_config_embedding"].numpy()
    actual = numpy_matcher.page_embedding(graph)

    assert actual.shape == (torch_matcher.config.hidden_dim,)
    assert np.linalg.norm(actual) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_runtime_uses_sharper_of_last_two_contextual_layers() -> None:
    source = graph_from_record(
        {"screen_id": "source", "nodes": [{"node_id": "s", "class": "View"}]}
    )
    target = graph_from_record(
        {
            "screen_id": "target",
            "nodes": [
                {"node_id": "correct", "class": "View"},
                {"node_id": "wrong", "class": "View"},
                {"node_id": "other", "class": "View"},
            ],
        }
    )
    matcher = NumpyGeometricAlignmentMatcher({}, config=MatcherConfig())
    previous = np.asarray([[6.0, 1.0, 0.0]], dtype=np.float32)
    final = np.asarray([[1.0, 1.1, 1.0]], dtype=np.float32)
    output = {
        "logits_ab": final,
        "affinity": final,
        "assignment_scores_by_layer": (previous, final),
        "affinities_by_layer": (previous, final),
    }

    prediction = matcher._predict_from_output(
        source,
        target,
        source_index=0,
        candidate_indices=[0, 1, 2],
        output=output,
        min_probability=0.0,
        min_margin=0.0,
    )

    assert prediction.target_node is not None
    assert prediction.target_node.node_id == "correct"
