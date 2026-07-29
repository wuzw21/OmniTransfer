from __future__ import annotations

import numpy as np

from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.numpy_matcher import NumpyMutualGraphMatcher
from omnitransfer.ui_graph import UIGraph, UINode


def test_numpy_inference_adapter_rejects_low_pair_confidence() -> None:
    source = _graph("source", ("search", "settings", "help"))
    target = _graph("target", ("settings", "search", "help"))
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=8,
        num_heads=4,
        num_layers=1,
        source_context_nodes=4,
        target_context_nodes=4,
    )

    class LowConfidenceMatcher(NumpyMutualGraphMatcher):
        def _forward(self, source_graph, target_graph):
            logits_ab = np.zeros((len(source_graph.nodes), len(target_graph.nodes)))
            logits_ab[:, 2] = 2.0
            affinity = np.full_like(logits_ab, -10.0)
            return {"logits_ab": logits_ab, "affinity": affinity}

    result = LowConfidenceMatcher({}, config=config).predict(
        source,
        target,
        source_node_id="node_0",
        min_probability=0.5,
    )

    assert result.target_node is None
    assert result.reason == "learned_low_confidence"
    assert result.probability < 0.5


def _graph(graph_id: str, labels: tuple[str, ...]) -> UIGraph:
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
            clickable=True,
            bbox=(10.0, 10.0 + 25.0 * index, 90.0, 30.0 + 25.0 * index),
        )
        for index, label in enumerate(labels)
    )
    return UIGraph(
        graph_id=graph_id,
        width=100,
        height=100,
        nodes=(root, *children),
    )
