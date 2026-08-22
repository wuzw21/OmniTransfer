from __future__ import annotations

import random

import pytest

from omnitransfer.learned_matcher import (
    GeometricMatcher,
    MatcherConfig,
    MULTISCALE_RESIDUAL_VISUAL_ENCODER,
    STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID,
    build_geometric_v9_matcher,
    initialize_node_encoder_from_checkpoint,
    initialize_v9_backbone_from_checkpoint,
    matcher_inputs,
    save_matcher_checkpoint,
)
from omnitransfer.visual_descriptor import multiscale_hash_descriptor_torch
from omnitransfer.self_supervised import (
    _final_margin_row_weight,
    AugmentConfig,
    UnlabeledPagePair,
    batch_self_view_node_matching_loss,
    batch_supervised_node_matching_loss,
    batch_unlabeled_page_embedding_contrastive_loss,
    make_correspondence_training_pair,
)
from omnitransfer.ui_graph import graph_from_record


def _page(graph_id: str, *, reverse_cards: bool = False):
    cards = [
        {
            "node_id": "first-card",
            "parent_id": "root",
            "class": "Card",
            "bounds": [0, 0, 100, 45],
        },
        {
            "node_id": "first-title",
            "parent_id": "first-card",
            "text": "Account",
            "class": "TextView",
            "bounds": [5, 5, 55, 20],
        },
        {
            "node_id": "first-edit",
            "parent_id": "first-card",
            "text": "Edit",
            "class": "Button",
            "clickable": True,
            "bounds": [70, 5, 95, 25],
        },
        {
            "node_id": "second-card",
            "parent_id": "root",
            "class": "Card",
            "bounds": [0, 50, 100, 95],
        },
        {
            "node_id": "second-title",
            "parent_id": "second-card",
            "text": "Privacy",
            "class": "TextView",
            "bounds": [5, 55, 55, 70],
        },
        {
            "node_id": "second-edit",
            "parent_id": "second-card",
            "text": "Edit",
            "class": "Button",
            "clickable": True,
            "bounds": [70, 55, 95, 75],
        },
    ]
    if reverse_cards:
        cards = cards[3:] + cards[:3]
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
                *cards,
            ],
        }
    )


def _page_with_unrelated_missing_boxes(graph_id: str):
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
                    "node_id": "left-group",
                    "parent_id": "root",
                    "class": "Group",
                    "bounds": [0, 0, 45, 100],
                },
                {
                    "node_id": "missing-left",
                    "parent_id": "left-group",
                    "class": "Button",
                },
                {
                    "node_id": "right-group",
                    "parent_id": "root",
                    "class": "Group",
                    "bounds": [55, 0, 100, 100],
                },
                {
                    "node_id": "missing-right",
                    "parent_id": "right-group",
                    "class": "Button",
                },
            ],
        }
    )


def test_missing_bounds_do_not_remove_nodes_from_learned_neighbour_pool() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    page = _page_with_unrelated_missing_boxes("missing-boxes")
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(page, page, config=config))

    node_indices = {node.node_id: index for index, node in enumerate(page.nodes)}
    left = node_indices["missing-left"]
    right = node_indices["missing-right"]
    slots = output["source_relation_neighbor_indices"][left]
    right_slots = slots.eq(right).nonzero(as_tuple=False).flatten()

    assert right_slots.numel() == 1
    assert bool(output["source_relation_mask"][left, right_slots[0]])
    relation = matcher_inputs(page, page, config=config)[2]
    assert torch.count_nonzero(relation[left, right, 9:16]) == 0


def test_neighbor_distance_is_trainable_not_a_fixed_bonus() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    model = build_geometric_v9_matcher(config)

    assert model.neighbor_relation_score.weight.requires_grad
    weights = model.neighbor_relation_score.weight.detach()
    assert torch.count_nonzero(weights[0, 9:17]) > 0


def test_final_scores_learn_through_soft_correspondence_anchors() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    torch.manual_seed(23)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-anchors")
    target = _page("target-anchors", reverse_cards=True)
    model = build_geometric_v9_matcher(config).train()

    output = model(*matcher_inputs(source, target, config=config))
    output["logits_ab"].square().mean().backward()

    assert output["unary_logits"].shape == output["logits_ab"].shape
    assert output["soft_correspondence"].shape == output["logits_ab"].shape
    assert torch.all(output["soft_correspondence"] >= 0.0)
    assert model.unary_logit_scale.grad is not None
    assert float(model.unary_logit_scale.grad.abs()) > 0.0
    assert output["source_neighbor_selection_scores"].requires_grad
    assert model.neighbor_query.weight.grad is not None
    assert float(model.neighbor_query.weight.grad.abs().sum()) > 0.0
    assert model.neighbor_relation_score.weight.grad is not None
    geometry_gradient = model.neighbor_relation_score.weight.grad[..., 9:17]
    assert float(geometry_gradient.abs().sum()) > 0.0


def test_multiscale_visual_residual_starts_stable_then_learns() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    torch.manual_seed(31)
    config = MatcherConfig(
        visual_encoder=MULTISCALE_RESIDUAL_VISUAL_ENCODER,
        dropout=0.0,
    )
    model = build_geometric_v9_matcher(config).train()
    patches = torch.rand(4, 6, 32, 32)

    fixed = multiscale_hash_descriptor_torch(patches, torch=torch)
    initial = model.visual_encoder(patches)

    assert torch.allclose(initial, fixed)
    initial.square().mean().backward()
    assert model.visual_encoder.dense_readout.weight.grad is not None
    assert float(model.visual_encoder.dense_readout.weight.grad.abs().sum()) > 0.0

    optimizer = torch.optim.SGD(model.visual_encoder.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    learned = model.visual_encoder(patches)
    learned.square().mean().backward()

    assert not torch.allclose(learned, fixed)
    assert model.visual_encoder.features[0].weight.grad is not None
    assert float(model.visual_encoder.features[0].weight.grad.abs().sum()) > 0.0


def test_correspondence_is_refined_three_times_through_one_matcher_path() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(association_layers=3, dropout=0.0)
    source = _page("source-three-stage")
    target = _page("target-three-stage", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    scores = output["assignment_scores_by_layer"]
    correspondences = output["soft_correspondence_by_layer"]
    assert len(scores) == len(correspondences) == 3
    assert torch.equal(output["logits_ab"], scores[-1])
    assert all(values.shape == output["unary_logits"].shape for values in scores)
    assert all(
        values.shape == output["unary_logits"].shape
        for values in correspondences
    )
    assert not torch.allclose(correspondences[0], correspondences[-1])


def test_local_correspondence_uses_all_configured_attention_heads() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(num_heads=4, dropout=0.0)
    source = _page("source-multi-anchor")
    target = _page("target-multi-anchor", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    weights = output["local_match_head_weights"]
    valid = (
        output["source_relation_mask"][:, None, None, :, None]
        & output["target_relation_mask"][None, :, None, None, :]
    ).any(dim=(-1, -2)).expand(-1, -1, config.num_heads)
    assert weights.shape[2] == config.num_heads
    assert torch.allclose(
        weights.sum(dim=(-1, -2))[valid],
        torch.ones_like(weights.sum(dim=(-1, -2))[valid]),
    )


def test_v9_node_encoder_fuses_only_available_modalities_into_64d_states() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    assert config.hidden_dim == 64
    assert config.association_layers == 3
    page = _page("v9-modalities")
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(page, page, config=config), node_only=True)

    weights = output["source_route_weights"]
    text_mask = output["source_text_mask"].reshape(-1).bool()
    visual_mask = output["source_visual_mask"].reshape(-1).bool()
    assert output["source_states"].shape == (len(page.nodes), 64)
    assert weights.shape == (len(page.nodes), 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(len(page.nodes)))
    assert torch.count_nonzero(weights[~text_mask, 0]) == 0
    assert torch.count_nonzero(weights[~visual_mask, 1]) == 0


def test_learned_neighbours_stay_explicit_until_cross_page_matching() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    torch.manual_seed(29)
    config = MatcherConfig(
        hidden_dim=64,
        association_layers=3,
        local_neighbor_limit=5,
        dropout=0.0,
    )
    source = _page("source-local-fusion")
    target = _page("target-local-fusion", reverse_cards=True)
    model = build_geometric_v9_matcher(config).train()

    output = model(*matcher_inputs(source, target, config=config))
    weights = output["source_neighbor_selection_weights"]
    mask = output["source_relation_mask"]

    assert torch.allclose(
        output["source_base_states"], output["source_fused_states"]
    )
    assert torch.allclose(
        weights.sum(dim=1),
        mask.any(dim=1).to(weights.dtype),
    )
    assert torch.count_nonzero(weights.masked_select(~mask)) == 0

    output["logits_ab"].square().mean().backward()
    assert model.neighbor_query.weight.grad is not None
    assert float(model.neighbor_query.weight.grad.abs().sum()) > 0.0
    assert model.relation_encoder[3].weight.grad is not None
    assert float(model.relation_encoder[3].weight.grad.abs().sum()) > 0.0


def test_zero_local_correction_preserves_the_unary_scores() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-unary-residual")
    target = _page("target-unary-residual", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()
    for parameter in model.pair_scorer.parameters():
        parameter.data.zero_()

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    assert torch.allclose(
        output["logits_ab"], output["backbone_scores_by_layer"][-1]
    )
    assert torch.count_nonzero(output["local_correction"]) == 0


def test_fresh_local_path_starts_as_a_small_v9_residual() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    torch.manual_seed(31)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-small-residual")
    target = _page("target-small-residual", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    assert float(output["local_correction"].abs().max()) < 0.1
    assert torch.allclose(
        output["source_fused_states"], output["source_base_states"]
    )


def test_correspondence_support_requires_bidirectional_distinctiveness() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    model = build_geometric_v9_matcher(MatcherConfig(dropout=0.0))
    uniform = torch.full((3, 3), 1.0 / 3.0)
    distinctive = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    uniform_support = model._relative_correspondence_support(uniform)
    distinctive_support = model._relative_correspondence_support(distinctive)

    assert torch.allclose(uniform_support, torch.zeros_like(uniform_support))
    assert torch.all(torch.diagonal(distinctive_support) > 0.0)
    assert torch.all(distinctive_support[~torch.eye(3, dtype=torch.bool)] < 0.0)


def test_trained_local_correction_is_not_capped_below_backbone_scale() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-uncapped-residual")
    target = _page("target-uncapped-residual", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()
    with torch.no_grad():
        model.pair_scorer[-1].weight.zero_()
        model.pair_scorer[-1].bias.fill_(3.0)
        output = model(*matcher_inputs(source, target, config=config))

    assert torch.allclose(
        output["local_correction"],
        torch.full_like(output["local_correction"], 3.0),
    )


def test_hard_rows_are_reweighted_inside_the_same_assignment_loss() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    pair = make_correspondence_training_pair(
        _page("source-hard-rows"),
        _page("target-hard-rows", reverse_cards=True),
        (
            ("first-edit", "first-edit"),
            ("second-edit", "second-edit"),
        ),
        matcher_config=config,
    )
    model = build_geometric_v9_matcher(config).eval()

    _loss, details = batch_supervised_node_matching_loss(
        model,
        [pair],
        matcher_config=config,
        hard_row_weight=3.0,
        return_details=True,
    )

    assert details["hard_row_weight"] == 3.0
    assert 1.0 < details["mean_training_row_weight"] <= 4.0


def test_hard_row_weight_uses_only_the_final_rank_two_margin() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    rank_two = torch.tensor([0.0, 0.6, -1.0], requires_grad=True)
    rank_three = torch.tensor([0.0, 0.6, 0.4], requires_grad=True)
    gold = torch.tensor([0], dtype=torch.long)

    rank_two_weight, rank_two_position, rank_two_margin = (
        _final_margin_row_weight(rank_two, gold, hard_row_weight=3.0)
    )
    rank_three_weight, rank_three_position, _ = _final_margin_row_weight(
        rank_three,
        gold,
        hard_row_weight=3.0,
    )

    assert rank_two_position == 2
    assert rank_two_margin == pytest.approx(-0.6)
    assert 2.5 < float(rank_two_weight) < 3.5
    assert rank_three_position == 3
    assert float(rank_three_weight) == 1.0
    assert not rank_two_weight.requires_grad


def test_local_relation_evidence_survives_an_uncertain_correspondence() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-additive-evidence")
    target = _page("target-additive-evidence", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()
    inputs = matcher_inputs(source, target, config=config)

    with torch.no_grad():
        source_page = model.encode_page(
            inputs[0], inputs[1], inputs[2], inputs[6], inputs[7]
        )
        target_page = model.encode_page(
            inputs[3], inputs[4], inputs[5], inputs[8], inputs[9]
        )
        pair_features = model._pair_features(
            source_page["states"], target_page["states"]
        )
        zero_correspondence = torch.zeros(
            len(source.nodes), len(target.nodes), dtype=pair_features.dtype
        )
        context, weights, anchors, _, _ = model._local_match(
            pair_features,
            zero_correspondence,
            source_page["relation_tokens"],
            source_page["relation_mask"],
            source_page["relation_neighbor_indices"],
            target_page["relation_tokens"],
            target_page["relation_mask"],
            target_page["relation_neighbor_indices"],
        )

    valid_pairs = (
        source_page["relation_mask"][:, None, :, None]
        & target_page["relation_mask"][None, :, None, :]
    ).any(dim=(-1, -2))
    assert torch.count_nonzero(anchors) == 0
    assert torch.allclose(
        weights.sum(dim=(-1, -2))[valid_pairs],
        torch.ones_like(weights.sum(dim=(-1, -2))[valid_pairs]),
    )
    assert float(context[valid_pairs].abs().sum()) > 0.0


def test_matcher_exposes_one_real_node_matrix_and_one_page_pair_score() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=128,
        relation_hidden_dim=32,
        association_layers=1,
        dropout=0.0,
        state_embedding_dim=128,
    )
    source = _page("source")
    target = _page("target", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()

    with torch.no_grad():
        output = model(*matcher_inputs(source, target, config=config))

    assert output["logits_ab"].shape == (len(source.nodes), len(target.nodes))
    assert torch.equal(output["logits_ba"], output["logits_ab"].T)
    assert output["page_pair_score"].shape == ()
    assert output["source_states"].shape == (len(source.nodes), 128)
    assert output["target_states"].shape == (len(target.nodes), 128)
    assert output["source_route_weights"].shape == (len(source.nodes), 3)
    assert "source_null_logits_by_layer" not in output
    assert len(output["assignment_scores_by_layer"]) == 1


def test_same_encoder_emits_an_independent_trainable_1024d_state_embedding() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    assert config.state_embedding_dim == 1024
    source = _page("source-state")
    target = _page("target-state", reverse_cards=True)
    model = build_geometric_v9_matcher(config).train()

    output = model(*matcher_inputs(source, target, config=config))

    for key in (
        "source_config_embedding",
        "target_config_embedding",
        "source_stable_state_embedding",
        "target_stable_state_embedding",
        "source_active_state_embedding",
        "target_active_state_embedding",
    ):
        assert output[key].shape == (1024,)
        assert float(torch.linalg.vector_norm(output[key]).detach()) == pytest.approx(
            1.0
        )
    assert torch.allclose(
        output["source_stable_state_embedding"],
        output["source_active_state_embedding"],
    )

    output["logits_ab"].square().mean().backward(retain_graph=True)
    assert model.state_attention.weight.grad is None

    output["source_state_embedding"][0].backward()
    assert model.state_attention.weight.grad is not None
    assert float(model.state_attention.weight.grad.abs().sum()) > 0.0


def test_page_pair_cross_entropy_trains_node_and_local_relation_paths() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(
        hidden_dim=128,
        relation_hidden_dim=32,
        association_layers=1,
        dropout=0.0,
        state_embedding_dim=128,
    )
    model = build_geometric_v9_matcher(config).train()
    original_visual_encoder = model.visual_encoder

    class CountingVisualEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, patches):
            self.calls += 1
            return original_visual_encoder(patches)

    counting_visual_encoder = CountingVisualEncoder()
    model.visual_encoder = counting_visual_encoder
    pairs = (
        UnlabeledPagePair(
            graph_a=_page("account-phone"),
            graph_b=_page("account-tablet", reverse_cards=True),
            pair_id="account",
            group_id="same-app",
        ),
        UnlabeledPagePair(
            graph_a=_page("privacy-phone", reverse_cards=True),
            graph_b=_page("privacy-tablet"),
            pair_id="privacy",
            group_id="same-app",
        ),
    )

    loss, details = batch_unlabeled_page_embedding_contrastive_loss(
        model,
        pairs,
        rng=random.Random(17),
        matcher_config=config,
        augment_config=AugmentConfig(),
        return_details=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert details["objective"] == "page_pair_cross_entropy"
    assert details["positive_labels"] == 2.0
    assert details["node_correspondence_labels"] == 0.0
    assert model.input_feed_forward[0].weight.grad is not None
    assert float(model.input_feed_forward[0].weight.grad.norm()) > 0.0
    assert model.relation_encoder[1].weight.grad is not None
    assert float(model.relation_encoder[1].weight.grad.norm()) > 0.0
    assert counting_visual_encoder.calls == 2 * len(pairs)
    assert not hasattr(model, "page_projection")


def test_runtime_ranks_real_candidates_without_a_null_class() -> None:
    config = MatcherConfig(
        hidden_dim=128,
        relation_hidden_dim=32,
        association_layers=1,
        dropout=0.0,
        state_embedding_dim=128,
    )
    source = _page("source")
    target = _page("target", reverse_cards=True)
    matcher = GeometricMatcher(
        build_geometric_v9_matcher(config),
        config=config,
    )

    result = matcher.predict(
        source,
        target,
        source_node_id="first-edit",
        candidate_node_ids=("first-edit", "second-edit"),
    )

    assert result.reason == "learned_match"
    assert result.target_node is not None
    assert {node_id for node_id, _ in result.scores} == {
        "first-edit",
        "second-edit",
    }
    assert result.probability == pytest.approx(result.scores[0][1])


def test_encoded_pages_and_page_match_are_reused_by_node_mapping() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-cached")
    target = _page("target-cached", reverse_cards=True)
    model = build_geometric_v9_matcher(config)
    matcher = GeometricMatcher(model, config=config)

    source_page = matcher.encode_page(source)
    target_page = matcher.encode_page(target)
    page_match = matcher.match_page(source_page, target_page)
    result = matcher.map_node(
        page_match,
        source_node_id="first-edit",
        candidate_node_ids=("first-edit", "second-edit"),
    )

    assert source_page.graph is source
    assert target_page.graph is target
    assert page_match.output["logits_ab"].shape == (
        len(source.nodes),
        len(target.nodes),
    )
    assert result.reason in {"learned_match", "learned_low_confidence"}
    assert {node_id for node_id, _ in result.scores} == {
        "first-edit",
        "second-edit",
    }


def test_runtime_abstains_when_two_candidates_have_exactly_equal_scores() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source-tie")
    target = _page("target-tie", reverse_cards=True)
    model = build_geometric_v9_matcher(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    matcher = GeometricMatcher(model, config=config)

    result = matcher.predict(
        source,
        target,
        source_node_id="first-edit",
        candidate_node_ids=("first-edit", "second-edit"),
    )

    assert result.target_node is None
    assert result.reason == "learned_low_confidence"
    assert result.margin == 0.0


def test_self_view_loss_trains_local_node_matching_without_human_node_gold() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    model = build_geometric_v9_matcher(config).train()
    pair = UnlabeledPagePair(
        graph_a=_page("phone"),
        graph_b=_page("tablet", reverse_cards=True),
        pair_id="page-only",
    )

    loss, details = batch_self_view_node_matching_loss(
        model,
        (pair,),
        rng=random.Random(19),
        matcher_config=config,
        augment_config=AugmentConfig(drop_node_prob=0.0),
        return_details=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert details["objective"] == "self_view_node_cross_entropy"
    assert details["generated_node_labels"] > 0
    assert details["human_node_labels"] == 0.0
    assert model.relation_encoder[1].weight.grad is not None
    assert float(model.relation_encoder[1].weight.grad.norm()) > 0.0


def test_supervised_loss_uses_cross_platform_set_valued_node_gold() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    model = build_geometric_v9_matcher(config).train()
    pair = make_correspondence_training_pair(
        _page("ios"),
        _page("android", reverse_cards=True),
        (
            ("first-edit", "first-edit"),
            ("first-edit", "second-edit"),
            ("second-edit", "second-edit"),
        ),
        matcher_config=config,
    )

    loss, details = batch_supervised_node_matching_loss(
        model,
        (pair,),
        matcher_config=config,
        return_details=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert details["objective"] == "symmetric_cross_platform_node_cross_entropy"
    assert details["human_node_labels"] == 3.0
    assert details["supervised_query_rows"] == 4.0
    assert details["page_pair_labels"] == 0.0
    assert details["supervised_layers"] == 3.0
    assert 0.0 <= details["layer_1_top1_accuracy"] <= 1.0
    assert 0.0 <= details["layer_2_top1_accuracy"] <= 1.0
    assert details["layer_3_top1_accuracy"] == details["node_top1_accuracy"]
    assert 0.0 <= details["unary_node_top1_accuracy"] <= 1.0
    assert details["refinement_help"] >= 0.0
    assert details["refinement_hurt"] >= 0.0
    assert (
        details["refinement_help"] + details["refinement_hurt"]
        <= details["supervised_query_rows"]
    )
    assert model.relation_encoder[1].weight.grad is not None
    assert float(model.relation_encoder[1].weight.grad.norm()) > 0.0


def test_one_assignment_objective_supervises_all_three_correspondence_updates() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(association_layers=3, dropout=0.0)
    model = build_geometric_v9_matcher(config).eval()
    pair = make_correspondence_training_pair(
        _page("ios-three-losses"),
        _page("android-three-losses", reverse_cards=True),
        (("first-edit", "first-edit"), ("second-edit", "second-edit")),
        matcher_config=config,
    )

    output = model(
        *matcher_inputs(pair.graph_a, pair.graph_b, config=config)
    )
    expected_layers = []
    for scores in (output["unary_logits"], *output["assignment_scores_by_layer"]):
        rows = []
        for matrix, positives in (
            (scores, pair.positive_targets_a_to_b),
            (scores.T, pair.positive_targets_b_to_a),
        ):
            for row_index, target_indices in enumerate(positives):
                if not target_indices:
                    continue
                row = matrix[row_index]
                target = torch.as_tensor(target_indices, dtype=torch.long)
                rows.append(
                    torch.logsumexp(row, dim=0)
                    - torch.logsumexp(row[target], dim=0)
                )
        expected_layers.append(torch.stack(rows).mean())
    expected = sum(
        weight * value
        for weight, value in zip((0.1, 0.2, 0.3, 0.4), expected_layers, strict=True)
    )

    actual, details = batch_supervised_node_matching_loss(
        model,
        (pair,),
        matcher_config=config,
        return_details=True,
    )

    assert details["objective"] == "symmetric_cross_platform_node_cross_entropy"
    assert details["supervised_layers"] == 3.0
    assert details["supervised_score_matrices"] == 4.0
    assert torch.allclose(actual, expected)


def test_node_stage_trains_only_the_unary_correspondence() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    model = build_geometric_v9_matcher(config).train()
    pair = make_correspondence_training_pair(
        _page("ios-node-stage"),
        _page("android-node-stage", reverse_cards=True),
        (("first-edit", "first-edit"), ("second-edit", "second-edit")),
        matcher_config=config,
    )

    loss, details = batch_supervised_node_matching_loss(
        model,
        (pair,),
        matcher_config=config,
        score_stage="unary",
        return_details=True,
    )
    loss.backward()

    assert details["score_stage"] == "unary"
    assert model.input_feed_forward[0].weight.grad is not None
    assert model.pair_scorer[-1].weight.grad is None
    assert model.neighbor_query.weight.grad is None


def test_checkpoint_round_trip_preserves_the_single_path(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    source = _page("source")
    target = _page("target", reverse_cards=True)
    model = build_geometric_v9_matcher(config).eval()
    checkpoint = tmp_path / "page-local.pt"

    with torch.no_grad():
        expected = model(*matcher_inputs(source, target, config=config))["logits_ab"]
    save_matcher_checkpoint(checkpoint, model, config=config)
    restored = GeometricMatcher.from_checkpoint(checkpoint)
    with torch.no_grad():
        actual = restored.model(
            *matcher_inputs(source, target, config=restored.config)
        )["logits_ab"]

    assert restored.checkpoint_load_mode == "exact_page_local_matcher"
    assert torch.allclose(actual, expected)
    forbidden = ("router", "gate", "null", "rerank")
    assert not any(
        token in name
        for name, _ in restored.model.named_parameters()
        for token in forbidden
    )


def test_node_encoder_can_initialize_a_fresh_local_matcher(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    torch.manual_seed(7)
    source = build_geometric_v9_matcher(config)
    checkpoint = tmp_path / "node-stage.pt"
    save_matcher_checkpoint(checkpoint, source, config=config)
    torch.manual_seed(19)
    target = build_geometric_v9_matcher(config)
    original_local = target.pair_scorer[-1].weight.detach().clone()

    transferred = initialize_node_encoder_from_checkpoint(target, checkpoint)

    assert "text_projection.weight" in transferred
    assert "modality_score.1.weight" in transferred
    assert "input_feed_forward.0.weight" in transferred
    assert torch.equal(
        target.text_projection.weight,
        source.text_projection.weight,
    )
    assert torch.equal(
        target.modality_score[1].weight,
        source.modality_score[1].weight,
    )
    assert torch.equal(target.pair_scorer[-1].weight, original_local)


def test_local_order_model_preserves_the_initialized_v9_xml_encoder(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    legacy_config = MatcherConfig(
        feature_schema_id=STATE_AWARE_GEOMETRIC_FEATURE_SCHEMA_ID,
        dropout=0.0,
    )
    source = build_geometric_v9_matcher(legacy_config)
    checkpoint = tmp_path / "state-aware-v9.pt"
    save_matcher_checkpoint(checkpoint, source, config=legacy_config)
    target = build_geometric_v9_matcher(MatcherConfig(dropout=0.0))

    transferred = initialize_v9_backbone_from_checkpoint(target, checkpoint)

    assert "xml_projection.0.weight" in transferred
    assert "xml_projection.1.weight" in transferred
    assert torch.equal(
        target.xml_projection[1].weight,
        source.xml_projection[1].weight,
    )
    assert torch.count_nonzero(target.local_order_projection[-1].weight) == 0


def test_trainable_multiscale_visual_preserves_old_v9_visual_start(tmp_path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    source_config = MatcherConfig(dropout=0.0)
    source = build_geometric_v9_matcher(source_config)
    checkpoint = tmp_path / "fixed-visual-v9.pt"
    save_matcher_checkpoint(checkpoint, source, config=source_config)
    target = build_geometric_v9_matcher(
        MatcherConfig(
            visual_encoder=MULTISCALE_RESIDUAL_VISUAL_ENCODER,
            dropout=0.0,
        )
    )

    transferred = initialize_v9_backbone_from_checkpoint(target, checkpoint)

    assert "missing_visual" in transferred
    assert "visual_to_hidden.weight" in transferred
    assert torch.equal(
        target.missing_visual[: source.missing_visual.shape[0]],
        source.missing_visual,
    )
    assert torch.count_nonzero(
        target.missing_visual[source.missing_visual.shape[0] :]
    ) == 0
    assert torch.equal(
        target.visual_to_hidden.weight[:, : source.visual_to_hidden.in_features],
        source.visual_to_hidden.weight,
    )
    assert torch.count_nonzero(
        target.visual_to_hidden.weight[:, source.visual_to_hidden.in_features :]
    ) == 0


def test_trained_v9_backbone_initializes_without_overwriting_new_local_fusion(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    torch.manual_seed(37)
    source = build_geometric_v9_matcher(config)
    checkpoint = tmp_path / "v9-backbone.pt"
    save_matcher_checkpoint(checkpoint, source, config=config)
    torch.manual_seed(41)
    target = build_geometric_v9_matcher(config)
    original_local = target.pair_scorer[-1].weight.detach().clone()

    transferred = initialize_v9_backbone_from_checkpoint(target, checkpoint)

    assert "association_layers.0.query.weight" in transferred
    assert "relation_compatibility" in transferred
    assert torch.equal(
        target.association_layers[0].query.weight,
        source.association_layers[0].query.weight,
    )
    assert torch.equal(target.pair_scorer[-1].weight, original_local)
