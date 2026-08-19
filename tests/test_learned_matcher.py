from dataclasses import replace

import pytest

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    DIRECT_PAIR_EVIDENCE_NAMES,
    DIRECT_TEXT_EVIDENCE_ENCODER,
    LEGACY_GLOBAL_POOL_VISUAL_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    RELATION_FEATURE_DIM,
    SPATIAL_CNN_VISUAL_ENCODER,
    TYPED_RELATION_NAMES,
    XML_NODE_FEATURE_DIM,
    MatcherConfig,
    GeometricMatcher,
    build_geometric_v9_matcher,
    confidence_adaptive_assignment_row,
    direct_pair_evidence_features,
    encode_graph,
    initialize_direct_text_from_lookup,
    initialize_nonvisual_from_model,
    matcher_inputs,
    save_matcher_checkpoint,
    typed_relation_bases,
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


def test_local_relation_bases_consume_kinship_distance() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    encoded = encode_graph(_graph("source"))
    relations = torch.as_tensor(encoded.relation_features, dtype=torch.float32)
    numeric = torch.as_tensor(encoded.numeric_features, dtype=torch.float32)

    bases = typed_relation_bases(relations, numeric)
    kinship_index = TYPED_RELATION_NAMES.index("kinship_proximity")

    assert float(bases[kinship_index, 0, 0]) == 0.0
    assert float(bases[kinship_index, 1, 2]) > 0.0
    assert float(bases[kinship_index, 2, 3]) > 0.0


def test_geometric_v9_checkpoint_round_trips(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
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


def test_every_refinement_layer_updates_nodes_and_emits_an_assignment() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=3,
        dropout=0.0,
    )
    model = build_geometric_v9_matcher(config).eval()
    source = _graph("source")
    target = _graph("target", first_x=15, second_x=75)

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    assert len(output["assignment_scores_by_layer"]) == 3
    assert len(output["source_states_by_layer"]) == 3
    assert len(output["target_states_by_layer"]) == 3
    assert not torch.equal(
        output["source_states_by_layer"][0],
        output["source_states_by_layer"][-1],
    )
    assert all(
        scores.shape == (len(source.nodes), len(target.nodes))
        for scores in output["assignment_scores_by_layer"]
    )


def test_shared_backbone_returns_normalized_configuration_embeddings() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        dropout=0.0,
    )
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(
            *matcher_inputs(_graph("source"), _graph("target"), config=config)
        )

    assert output["source_config_embedding"].shape == (config.hidden_dim,)
    assert output["target_config_embedding"].shape == (config.hidden_dim,)
    assert torch.allclose(
        output["source_config_embedding"].norm(),
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_missing_text_and_visual_are_normal_inputs() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        dropout=0.0,
    )
    graph = graph_from_record(
        {
            "screen_id": "missing-modalities",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root"},
                {
                    "node_id": "icon",
                    "parent_id": "root",
                    "class": "ImageView",
                    "clickable": True,
                    "bounds": [20, 20, 40, 40],
                },
            ],
        }
    )
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(graph, graph, config=config))

    assert output["logits_ab"].shape == (2, 2)
    assert torch.isfinite(output["logits_ab"]).all()
    assert torch.isfinite(output["source_config_embedding"]).all()


def test_runtime_context_keeps_every_requested_target_candidate() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        source_context_nodes=2,
        target_context_nodes=2,
        dropout=0.0,
    )
    source = _graph("source")
    target = _graph("target")
    candidate_ids = ("first", "second", "context")
    matcher = GeometricMatcher(build_geometric_v9_matcher(config), config=config)

    prediction = matcher.predict(
        source,
        target,
        source_node_id="first",
        candidate_node_ids=candidate_ids,
    )

    assert {node_id for node_id, _ in prediction.scores} == set(candidate_ids)


def test_direct_text_release_conversion_preserves_zero_lookup(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    learned_config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
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


def test_visual_encoder_migration_preserves_new_visual_parameters() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    base_config = MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        dropout=0.0,
    )
    source = build_geometric_v9_matcher(
        replace(base_config, visual_encoder=LEGACY_GLOBAL_POOL_VISUAL_ENCODER)
    )
    target = build_geometric_v9_matcher(
        replace(base_config, visual_encoder=SPATIAL_CNN_VISUAL_ENCODER)
    )
    for parameter in source.parameters():
        parameter.data.fill_(0.25)
    for parameter in target.parameters():
        parameter.data.fill_(-0.75)

    transferred = initialize_nonvisual_from_model(target, source)
    target_state = target.state_dict()

    assert transferred
    assert all(not name.startswith("visual_encoder.") for name in transferred)
    for name, value in target_state.items():
        expected = -0.75 if name.startswith("visual_encoder.") else 0.25
        assert torch.all(value == expected), name


def test_torch_decoder_uses_sharper_previous_association_layer() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    previous = torch.tensor([[6.0, 1.0, 0.0]])
    final = torch.tensor([[1.0, 1.1, 1.0]])
    output = {
        "logits_ab": final,
        "logits_ba": final.T,
        "assignment_scores_by_layer": (previous, final),
    }

    selected, layer = confidence_adaptive_assignment_row(
        output,
        source_index=0,
        candidate_indices=[0, 1, 2],
    )

    assert layer == 0
    assert int(torch.argmax(selected)) == 0
