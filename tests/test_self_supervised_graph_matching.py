import random
from dataclasses import replace

import pytest

from omnitransfer.learned_matcher import MatcherConfig, build_geometric_v9_matcher
from omnitransfer.self_supervised import (
    AugmentConfig,
    augment_graph,
    make_correspondence_training_pair,
    matching_loss,
    train_geometric_v9_matcher,
)
from omnitransfer.ui_graph import UIGraph, UINode


def _graphs() -> tuple[UIGraph, UIGraph]:
    source = UIGraph(
        graph_id="source",
        width=100,
        height=100,
        nodes=(
            UINode(node_id="root-s", origin_id="root-s", class_name="Root"),
            UINode(
                node_id="alpha-s",
                origin_id="alpha-s",
                parent_id="root-s",
                text="Alpha",
                class_name="Button",
                clickable=True,
                bbox=(5, 10, 45, 30),
            ),
            UINode(
                node_id="label-s",
                origin_id="label-s",
                parent_id="root-s",
                text="Description",
                class_name="TextView",
                bbox=(5, 40, 95, 60),
            ),
        ),
    )
    target = UIGraph(
        graph_id="target",
        width=120,
        height=100,
        nodes=(
            UINode(node_id="root-t", origin_id="root-t", class_name="Root"),
            UINode(
                node_id="alpha-t",
                origin_id="alpha-t",
                parent_id="root-t",
                text="Alpha",
                class_name="Button",
                clickable=True,
                bbox=(10, 10, 55, 30),
            ),
            UINode(
                node_id="label-t",
                origin_id="label-t",
                parent_id="root-t",
                text="Description",
                class_name="TextView",
                bbox=(10, 40, 110, 60),
            ),
        ),
    )
    return source, target


def _config() -> MatcherConfig:
    return MatcherConfig(
        hidden_dim=32,
        relation_hidden_dim=16,
        association_dim=24,
        num_heads=4,
        association_layers=2,
        dropout=0.0,
    )


def test_correspondence_pair_preserves_non_actionable_xml_labels() -> None:
    source, target = _graphs()

    pair = make_correspondence_training_pair(
        source,
        target,
        (("alpha-s", "alpha-t"), ("label-s", "label-t")),
        matcher_config=_config(),
    )

    assert pair.positive_targets_a_to_b[1] == (1,)
    assert pair.positive_targets_a_to_b[2] == (2,)
    assert pair.origin_ids == ("alpha-s->alpha-t", "label-s->label-t")


def test_augmentation_preserves_origin_identity() -> None:
    source, _ = _graphs()

    augmented = augment_graph(
        source,
        rng=random.Random(17),
        config=AugmentConfig(drop_node_prob=0.0, distractor_prob=0.0),
    )

    assert {node.origin_id for node in augmented.nodes} == {
        node.origin_id for node in source.nodes
    }


def test_augmentation_node_retention_is_independent_of_clickability() -> None:
    source, _ = _graphs()
    changed = UIGraph(
        graph_id=source.graph_id,
        width=source.width,
        height=source.height,
        nodes=tuple(
            replace(node, clickable=not node.clickable)
            for node in source.nodes
        ),
    )
    config = AugmentConfig(drop_node_prob=0.7, distractor_prob=0.0, min_nodes=1)

    original_augmented = augment_graph(source, rng=random.Random(29), config=config)
    changed_augmented = augment_graph(changed, rng=random.Random(29), config=config)

    assert [node.origin_id for node in original_augmented.nodes] == [
        node.origin_id for node in changed_augmented.nodes
    ]


def test_geometric_matching_loss_and_training_smoke() -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    source, target = _graphs()
    config = _config()
    pair = make_correspondence_training_pair(
        source,
        target,
        (("alpha-s", "alpha-t"), ("label-s", "label-t")),
        matcher_config=config,
    )
    model = build_geometric_v9_matcher(config)

    loss, details = matching_loss(
        model,
        pair,
        matcher_config=config,
        return_details=True,
    )
    trained, history = train_geometric_v9_matcher(
        (),
        (pair,),
        model=model,
        epochs=1,
        matcher_config=config,
        synthetic_pairs_per_graph=0,
    )

    assert loss.item() >= 0.0
    assert details["supervised_layers"] == 2.0
    assert details["visual_descriptor_trainable"] == 0.0
    assert details["visual_descriptor_loss"] == 0.0
    assert details["matchability_loss"] > 0.0
    assert trained.training is False
    assert history[0]["cross_page_pairs"] == 1.0


def test_training_rejects_removed_architectures() -> None:
    source, target = _graphs()
    config = _config()
    pair = make_correspondence_training_pair(
        source,
        target,
        (("alpha-s", "alpha-t"),),
        matcher_config=config,
    )

    with pytest.raises((RuntimeError, ValueError), match="PyTorch|geometric-v9"):
        train_geometric_v9_matcher(
            (),
            (pair,),
            matcher_config=MatcherConfig(architecture="historical"),
            synthetic_pairs_per_graph=0,
        )
