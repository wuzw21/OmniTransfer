from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omnitransfer.learned_matcher import GeometricMatcher
from omnitransfer.numpy_v9_matcher import (
    NumpyGeometricAlignmentMatcher,
    save_numpy_geometric_v9_checkpoint,
)
from omnitransfer.ui_graph import graph_from_record

torch = pytest.importorskip("torch", exc_type=ImportError)


CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "src/omnitransfer/checkpoints/omnitransfer_direct_text_alignment_v9_20260805/"
    "v9_direct_text_alignment_seed29.pt"
)
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


def _export(path: Path) -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    matcher = GeometricMatcher.from_checkpoint(CHECKPOINT)
    save_numpy_geometric_v9_checkpoint(
        path,
        payload["state_dict"],
        config=matcher.config,
    )


def test_v9_numpy_export_is_deterministic_and_pickle_free(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _export(first)
    _export(second)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as checkpoint:
        assert "__schema_version__" in checkpoint.files
        assert "local_alignment_score.3.weight" in checkpoint.files


def test_v9_numpy_matches_pytorch_candidate_ranking(tmp_path: Path) -> None:
    exported = tmp_path / "v9.npz"
    _export(exported)
    source = graph_from_record({"xml": SOURCE_XML}, graph_id="source")
    target = graph_from_record({"xml": TARGET_XML}, graph_id="target")
    source_node = next(node for node in source.nodes if node.content_desc == "搜索")
    candidates = tuple(node.node_id for node in target.nodes if node.clickable)

    pytorch_match = GeometricMatcher.from_checkpoint(CHECKPOINT).predict(
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
