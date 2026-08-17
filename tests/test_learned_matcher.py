from dataclasses import replace

import pytest

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    DIRECT_PAIR_EVIDENCE_NAMES,
    DIRECT_TEXT_EVIDENCE_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    RELATION_FEATURE_DIM,
    XML_NODE_FEATURE_DIM,
    MatcherConfig,
    GeometricMatcher,
    build_geometric_v9_matcher,
    direct_pair_evidence_features,
    encode_graph,
    initialize_direct_text_from_lookup,
    matcher_inputs,
    save_matcher_checkpoint,
    spatial_xml_alignment_choice,
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


def test_default_config_is_the_only_trainable_model() -> None:
    config = MatcherConfig()

    assert config.architecture == OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE
    assert config.candidate_policy == ALL_NODE_CANDIDATE_POLICY


def test_historical_architectures_are_rejected() -> None:
    with pytest.raises(ValueError, match="only the geometric-v9"):
        build_geometric_v9_matcher(MatcherConfig(architecture="removed_model"))


def test_graph_encoder_preserves_current_feature_contract() -> None:
    encoded = encode_graph(_graph("source"))

    assert len(encoded.numeric_features[0]) == XML_NODE_FEATURE_DIM
    assert len(encoded.relation_features[0][0]) == RELATION_FEATURE_DIM
    assert encoded.token_ids[2] != encoded.token_ids[3]
    assert encoded.relation_features[0][2][4] == 1.0
    assert encoded.relation_features[2][3][3] == 1.0


def test_direct_pair_evidence_has_one_fixed_schema() -> None:
    source = _graph("source").nodes[2]
    target = replace(_graph("target").nodes[2], text=source.text)

    evidence = direct_pair_evidence_features(source, target)

    assert len(evidence) == len(DIRECT_PAIR_EVIDENCE_NAMES)
    assert evidence[0] == 1.0


def test_spatial_xml_alignment_can_resolve_semantic_ambiguity() -> None:
    source = _graph("source", first_x=10, second_x=80)
    target = _graph("target", first_x=80, second_x=10)
    source_node = source.nodes[2]
    target_nodes = (target.nodes[2], target.nodes[3])

    choice = spatial_xml_alignment_choice(
        (10.0, 8.5),
        source_node=source_node,
        target_nodes=target_nodes,
        source_graph=source,
        target_graph=target,
    )

    assert choice == 1


def test_geometric_v9_checkpoint_round_trips(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = build_geometric_v9_matcher(config).eval()
    source = _graph("source")
    target = _graph("target", first_x=15, second_x=75)

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    assert output["logits_ab"].shape == (len(source.nodes), len(target.nodes))
    assert output["affinity"].shape == output["logits_ab"].shape
    checkpoint = tmp_path / "geometric-v9.pt"
    save_matcher_checkpoint(checkpoint, model, config=config)
    restored = GeometricMatcher.from_checkpoint(checkpoint)
    assert restored.config == config


def test_direct_text_release_conversion_preserves_zero_lookup(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    learned_config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    direct_config = replace(
        learned_config,
        text_encoder=DIRECT_TEXT_EVIDENCE_ENCODER,
    )
    learned = build_geometric_v9_matcher(learned_config).eval()
    direct = build_geometric_v9_matcher(direct_config).eval()
    learned.token_embedding.weight.data.zero_()
    initialize_direct_text_from_lookup(direct, learned)
    source = _graph("source")
    target = _graph("target")

    with torch.no_grad():
        expected = learned(*matcher_inputs(source, target, config=learned_config))
        actual = direct(*matcher_inputs(source, target, config=direct_config))

    assert torch.equal(actual["logits_ab"], expected["logits_ab"])
    checkpoint = tmp_path / "direct-text-v9.pt"
    save_matcher_checkpoint(checkpoint, direct, config=direct_config)
    assert GeometricMatcher.from_checkpoint(checkpoint).config == direct_config
