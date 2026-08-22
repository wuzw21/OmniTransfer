from __future__ import annotations

import pytest

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    LEGACY_NODE_ANCHOR_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    RELATION_FEATURE_DIM,
    SOFTMAX_MODALITY_ROUTER,
    STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID,
    MatcherConfig,
    build_geometric_v9_matcher,
    configure_visual_image_cache,
    encode_graph,
    matcher_inputs,
    parameter_count,
)
from omnitransfer.ui_graph import graph_from_record


def _graph(graph_id: str, *, first_x: int = 10, second_x: int = 80):
    return graph_from_record(
        {
            "screen_id": graph_id,
            "width": 100,
            "height": 100,
            "nodes": [
                {
                    "node_id": "root",
                    "class": "Root",
                    "bounds": [0, 0, 100, 100],
                },
                {
                    "node_id": "list",
                    "parent_id": "root",
                    "class": "List",
                    "bounds": [0, 0, 100, 100],
                },
                {
                    "node_id": "first",
                    "parent_id": "list",
                    "text": "First control",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [first_x, 10, first_x + 10, 20],
                },
                {
                    "node_id": "second",
                    "parent_id": "list",
                    "text": "Different control",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [second_x, 10, second_x + 10, 20],
                },
                {
                    "node_id": "context",
                    "parent_id": "list",
                    "text": "Context",
                    "class": "TextView",
                    "bounds": [40, 50, 60, 60],
                },
            ],
        }
    )


def test_default_config_describes_only_the_page_local_model() -> None:
    config = MatcherConfig()

    assert config.architecture == OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
    assert config.candidate_policy == ALL_NODE_CANDIDATE_POLICY
    assert config.node_anchor_encoder == LEGACY_NODE_ANCHOR_ENCODER
    assert config.router_policy == SOFTMAX_MODALITY_ROUTER
    assert config.hidden_dim == 64
    assert config.association_layers == 3
    assert config.assignment_head == "partial_assignment"
    assert config.feature_schema_id == STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID
    assert config.state_embedding_dim == 1024


def test_historical_architectures_are_rejected() -> None:
    with pytest.raises(ValueError, match="only the"):
        build_geometric_v9_matcher(MatcherConfig(architecture="removed_model"))


def test_graph_encoding_retains_structure_order_and_reliability() -> None:
    original = _graph("original")
    reordered = graph_from_record(
        {
            "screen_id": "reordered",
            "width": 100,
            "height": 100,
            "nodes": [
                original.nodes[0].__dict__,
                original.nodes[1].__dict__,
                original.nodes[3].__dict__,
                original.nodes[2].__dict__,
                original.nodes[4].__dict__,
            ],
        }
    )

    encoded = encode_graph(original)
    reordered_encoded = encode_graph(reordered)
    first = encoded.node_ids.index("first")
    reordered_first = reordered_encoded.node_ids.index("first")

    assert len(encoded.relation_features[0][0]) == RELATION_FEATURE_DIM
    assert encoded.numeric_features[first] != reordered_encoded.numeric_features[
        reordered_first
    ]
    assert encoded.relation_features[2][3][3] == 1.0


def test_flat_repeated_resource_ids_remain_distinct_graph_nodes() -> None:
    graph = graph_from_record(
        {
            "screen_id": "repeated-list",
            "nodes": [
                {"node_id": "root", "class": "Root"},
                {
                    "resource-id": "app:id/card",
                    "parent_id": "root",
                    "text": "First",
                },
                {
                    "resource-id": "app:id/card",
                    "parent_id": "root",
                    "text": "Second",
                },
            ],
        }
    )

    assert len(graph.nodes) == 3
    assert len({node.node_id for node in graph.nodes}) == 3
    root = next(node for node in graph.nodes if node.node_id == "root")
    assert len(root.child_ids) == 2
    assert len(set(root.child_ids)) == 2
    assert [node.resource_id for node in graph.nodes[1:]] == [
        "app:id/card",
        "app:id/card",
    ]


def test_structural_or_spatial_evidence_alone_can_create_a_local_relation() -> None:
    graph = graph_from_record(
        {
            "screen_id": "or-local-evidence",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "parent", "class": "Group"},
                {
                    "node_id": "structural-only",
                    "parent_id": "parent",
                    "class": "Button",
                },
                {
                    "node_id": "spatial-a",
                    "class": "Button",
                    "bounds": [10, 10, 20, 20],
                },
                {
                    "node_id": "spatial-b",
                    "class": "TextView",
                    "bounds": [22, 10, 32, 20],
                },
            ],
        }
    )

    encoded = encode_graph(graph)

    assert encoded.relation_features[0][1][17] == 1.0
    assert encoded.relation_features[2][3][17] == 1.0


def test_model_accepts_missing_text_and_visual_as_observations() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = graph_from_record(
        {
            "screen_id": "missing",
            "width": 100,
            "height": 100,
            "nodes": [
                {
                    "node_id": "control",
                    "class": "View",
                    "bounds": [10, 10, 30, 30],
                }
            ],
        }
    )
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(source, source, config=config))

    assert torch.isfinite(output["logits_ab"]).all()
    assert torch.isfinite(output["source_config_embedding"]).all()


def test_page_local_model_is_compact_and_has_no_parallel_control_modules() -> None:
    model = build_geometric_v9_matcher(MatcherConfig())
    names = tuple(name for name, _ in model.named_parameters())

    assert parameter_count(model) < 1_000_000
    assert not any(
        token in name
        for name in names
        for token in ("router", "gate", "null", "rerank")
    )
    assert len(model.association_layers) == 3


def test_visual_image_cache_is_training_configurable() -> None:
    try:
        assert configure_visual_image_cache(7) == 7
    finally:
        configure_visual_image_cache(32)
