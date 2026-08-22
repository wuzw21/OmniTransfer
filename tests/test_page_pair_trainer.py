import random
from types import SimpleNamespace

import pytest

from omnitransfer.learned_matcher import MatcherConfig, build_geometric_v9_matcher
from omnitransfer.self_supervised import make_correspondence_training_pair
from omnitransfer.ui_graph import graph_from_record
from scripts.train_geometric_v9_matcher import (
    _freeze_v9_backbone,
    _optimizer_parameter_groups,
    _page_batches,
    _run_epoch,
)


def test_page_batches_visit_each_pair_once_per_epoch() -> None:
    pairs = [
        SimpleNamespace(pair_id=f"pair-{index}", group_id="same-app", dataset="d")
        for index in range(8)
    ]

    batches = _page_batches(pairs, batch_size=4, rng=random.Random(3))
    visited = [pair.pair_id for batch in batches for pair in batch]

    assert len(batches) == 2
    assert sorted(visited) == sorted(pair.pair_id for pair in pairs)


def test_page_batches_replay_hard_pairs_without_dropping_easy_pairs() -> None:
    pairs = [
        SimpleNamespace(
            pair_id=f"pair-{index}",
            graph_a=SimpleNamespace(graph_id=f"a-{index}"),
            graph_b=SimpleNamespace(graph_id=f"b-{index}"),
        )
        for index in range(4)
    ]
    difficulty = {
        tuple(sorted((pair.graph_a.graph_id, pair.graph_b.graph_id))): (
            100.0 if pair.pair_id == "pair-2" else 0.0
        )
        for pair in pairs
    }

    batches = _page_batches(
        pairs,
        batch_size=4,
        rng=random.Random(3),
        pair_difficulty=difficulty,
        hard_replay_ratio=0.5,
    )
    visited = [pair.pair_id for batch in batches for pair in batch]

    assert len(visited) == 6
    assert all(pair.pair_id in visited for pair in pairs)
    assert visited.count("pair-2") == 3


def test_epoch_metrics_distinguish_online_training_from_fixed_model() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    config = MatcherConfig(dropout=0.0)
    model = build_geometric_v9_matcher(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pair = make_correspondence_training_pair(
        _page("source"),
        _page("target"),
        (("button", "button"),),
        matcher_config=config,
    )

    online = _run_epoch(
        model,
        [pair],
        optimizer=optimizer,
        config=config,
        augment=None,
        device="cpu",
        batch_size=1,
        rng=random.Random(1),
        progress_every=0,
        torch=torch,
    )
    fixed = _run_epoch(
        model,
        [pair],
        optimizer=None,
        config=config,
        augment=None,
        device="cpu",
        batch_size=1,
        rng=random.Random(1),
        progress_every=0,
        torch=torch,
    )

    assert online["metric_scope"] == "online_pre_update_batches"
    assert fixed["metric_scope"] == "fixed_model_full_pass"


def test_optimizer_does_not_decay_embeddings_norms_or_biases() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    model = build_geometric_v9_matcher(MatcherConfig(dropout=0.0))

    groups = _optimizer_parameter_groups(
        model,
        weight_decay=0.05,
        learning_rate=1e-4,
        node_learning_rate_scale=0.1,
    )
    decay_by_parameter = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }
    learning_rate_by_parameter = {
        id(parameter): float(group["lr"])
        for group in groups
        for parameter in group["params"]
    }

    assert decay_by_parameter[id(model.token_embedding.weight)] == 0.0
    assert decay_by_parameter[id(model.input_norm.weight)] == 0.0
    assert decay_by_parameter[id(model.text_to_hidden.bias)] == 0.0
    assert decay_by_parameter[id(model.text_to_hidden.weight)] == 0.05
    assert learning_rate_by_parameter[id(model.text_to_hidden.weight)] == 1e-5
    assert (
        learning_rate_by_parameter[id(model.association_layers[0].query.weight)]
        == 1e-5
    )
    assert learning_rate_by_parameter[id(model.pair_scorer[1].weight)] == 1e-4


def test_freeze_v9_backbone_leaves_new_local_path_trainable() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    model = build_geometric_v9_matcher(MatcherConfig(dropout=0.0))

    frozen = _freeze_v9_backbone(model)

    assert frozen
    assert not model.association_layers[0].query.weight.requires_grad
    assert not model.text_to_hidden.weight.requires_grad
    assert model.pair_scorer[-1].weight.requires_grad
    assert model.neighbor_query.weight.requires_grad
    assert model.state_attention.weight.requires_grad


def _page(graph_id: str):
    return graph_from_record(
        {
            "screen_id": graph_id,
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "button",
                    "parent_id": "root",
                    "text": "Continue",
                    "class": "Button",
                    "bounds": [20, 20, 80, 50],
                },
            ],
        }
    )
