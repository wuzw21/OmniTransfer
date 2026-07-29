import random

import pytest

from omnitransfer.self_supervised import (
    AugmentConfig,
    FEATURE_DIM,
    evaluate_correspondence_pairs,
    make_correspondence_training_pair,
    make_training_pair,
    matching_loss,
    train_mapping_matcher,
)
from omnitransfer.learned_matcher import MatcherConfig, RELATION_FEATURE_DIM
from omnitransfer.ui_graph import (
    UIGraph,
    UINode,
    graph_from_record,
    local_context_graph,
)


def test_graph_from_nested_record_parses_nodes_and_bounds() -> None:
    record = {
        "screen_id": "screen-1",
        "width": 100,
        "height": 200,
        "view_hierarchy": {
            "class": "FrameLayout",
            "bounds": [0, 0, 100, 200],
            "children": [
                {
                    "text": "Search",
                    "resource-id": "com.app:id/search",
                    "class": "android.widget.EditText",
                    "clickable": True,
                    "bounds": "[10,20][90,60]",
                },
                {
                    "content-desc": "Settings",
                    "class": "android.widget.ImageButton",
                    "bounds": [70, 150, 95, 190],
                },
            ],
        },
    }

    graph = graph_from_record(record)

    assert graph.graph_id == "screen-1"
    assert len(graph.nodes) == 3
    assert graph.width == 100
    assert graph.height == 200
    assert graph.nodes[1].text == "Search"
    assert graph.nodes[1].bbox == (10.0, 20.0, 90.0, 60.0)


def test_augmented_views_keep_origin_correspondence() -> None:
    record = {
        "screen_id": "screen-2",
        "width": 100,
        "height": 100,
        "nodes": [
            {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
            {
                "node_id": "search",
                "parent_id": "root",
                "text": "Search",
                "class": "EditText",
                "bounds": [5, 5, 80, 20],
                "clickable": True,
            },
            {
                "node_id": "cancel",
                "parent_id": "root",
                "text": "Cancel",
                "class": "Button",
                "bounds": [82, 5, 98, 20],
                "clickable": True,
            },
            {
                "node_id": "row",
                "parent_id": "root",
                "text": "Result",
                "class": "TextView",
                "bounds": [5, 30, 98, 55],
            },
        ],
    }
    graph = graph_from_record(record)
    pair = make_training_pair(
        graph,
        rng=random.Random(7),
        config=AugmentConfig(
            mask_text_prob=1.0,
            mask_content_desc_prob=1.0,
            mask_class_prob=0.0,
            drop_node_prob=0.0,
            edge_dropout_prob=0.0,
            bbox_jitter=0.0,
        ),
    )

    assert pair is not None
    assert pair.origin_ids == ("cancel", "search")
    assert len(pair.features_a[0]) == FEATURE_DIM
    assert len(pair.encoded_a.relation_features[0][0]) == RELATION_FEATURE_DIM
    assert pair.graph_a.nodes[pair.source_indices[0]].origin_id == pair.origin_ids[0]
    assert pair.graph_b.nodes[pair.target_indices[0]].origin_id == pair.origin_ids[0]
    assert (
        pair.graph_a.metadata["visual_transform"]
        != pair.graph_b.metadata["visual_transform"]
    )
    original_boxes = {node.origin_id: node.bbox for node in graph.nodes}
    assert all(
        node.metadata["visual_bbox"] == original_boxes[node.origin_id]
        for node in pair.graph_a.nodes
        if not node.origin_id.startswith("__distractor__:")
    )


def test_augmented_views_include_hard_distractors_as_ignored_context() -> None:
    graph = graph_from_record(
        {
            "screen_id": "screen-hard-negatives",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "first",
                    "parent_id": "root",
                    "text": "Delete",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [5, 10, 45, 30],
                },
                {
                    "node_id": "second",
                    "parent_id": "root",
                    "text": "Delete",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [55, 10, 95, 30],
                },
                {
                    "node_id": "label",
                    "parent_id": "root",
                    "text": "Archive",
                    "class": "TextView",
                    "bounds": [5, 40, 95, 60],
                },
            ],
        }
    )
    pair = make_training_pair(
        graph,
        rng=random.Random(3),
        config=AugmentConfig(
            drop_node_prob=0.0,
            distractor_prob=1.0,
            max_distractors=2,
            bbox_jitter=0.0,
            global_translation=0.0,
            global_scale=0.0,
        ),
    )

    assert pair is not None
    assert sum(target < 0 for target in pair.targets_a_to_b) == 4
    assert sum(target < 0 for target in pair.targets_b_to_a) == 4
    assert any(node.metadata.get("hard_negative") for node in pair.graph_a.nodes)


def test_same_page_supervision_keeps_context_but_labels_only_actionable_nodes() -> None:
    graph = graph_from_record(
        {
            "screen_id": "actionable-only",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "search",
                    "parent_id": "root",
                    "text": "Search",
                    "class": "EditText",
                    "clickable": True,
                    "editable": True,
                    "bounds": [5, 5, 95, 25],
                },
                {
                    "node_id": "settings",
                    "parent_id": "root",
                    "content-desc": "Settings",
                    "class": "ImageButton",
                    "clickable": True,
                    "bounds": [70, 70, 95, 95],
                },
                {
                    "node_id": "title",
                    "parent_id": "root",
                    "text": "Results",
                    "class": "TextView",
                    "bounds": [5, 35, 95, 55],
                },
            ],
        }
    )

    pair = make_training_pair(
        graph,
        rng=random.Random(11),
        config=AugmentConfig(
            drop_node_prob=0.0,
            distractor_prob=0.0,
            bbox_jitter=0.0,
            global_translation=0.0,
            global_scale=0.0,
        ),
    )

    assert pair is not None
    assert pair.origin_ids == ("search", "settings")
    assert {node.origin_id for node in pair.graph_a.nodes} == {
        "root",
        "search",
        "settings",
        "title",
    }
    assert sum(target >= 0 for target in pair.targets_a_to_b) == 2


def test_matching_loss_ranks_only_actionable_target_candidates() -> None:
    torch = __import__("pytest").importorskip("torch")
    source = UIGraph(
        graph_id="source",
        nodes=(
            UINode(node_id="source-action", origin_id="source-action", clickable=True),
            UINode(node_id="source-context", origin_id="source-context", text="Title"),
        ),
    )
    target = UIGraph(
        graph_id="target",
        nodes=(
            UINode(node_id="target-action", origin_id="target-action", clickable=True),
            UINode(node_id="target-context", origin_id="target-context", text="Title"),
        ),
    )
    pair = make_correspondence_training_pair(
        source,
        target,
        (("source-action", "target-action"),),
    )

    class ContextDistractedMatcher(torch.nn.Module):
        def forward(self, *inputs):
            logits_ab = torch.tensor([[0.0, 100.0], [0.0, 0.0]])
            logits_ba = torch.tensor([[0.0, 100.0], [0.0, 0.0]])
            return {"logits_ab": logits_ab, "logits_ba": logits_ba}

    loss = matching_loss(ContextDistractedMatcher(), pair, cycle_weight=0.0)

    assert float(loss) < 0.01


def test_matching_loss_supervises_every_assignment_layer() -> None:
    torch = __import__("pytest").importorskip("torch")
    source = UIGraph(
        graph_id="source",
        nodes=(
            UINode(node_id="source-action", origin_id="source-action", clickable=True),
            UINode(
                node_id="source-hard-negative",
                origin_id="source-hard-negative",
                clickable=True,
            ),
        ),
    )
    target = UIGraph(
        graph_id="target",
        nodes=(
            UINode(node_id="target-action", origin_id="target-action", clickable=True),
            UINode(
                node_id="target-hard-negative",
                origin_id="target-hard-negative",
                clickable=True,
            ),
        ),
    )
    pair = make_correspondence_training_pair(
        source,
        target,
        (("source-action", "target-action"),),
    )

    class LayerwiseMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros((2, 2)))
            self.final = torch.nn.Parameter(torch.zeros((2, 2)))

        def forward(self, *inputs):
            return {
                "logits_ab": self.final,
                "logits_ba": self.final.T,
                "affinity": self.final,
                "assignment_scores_by_layer": (self.first, self.final),
            }

    model = LayerwiseMatcher()
    loss = matching_loss(model, pair, cycle_weight=0.0)
    loss.backward()

    assert model.first.grad is not None
    assert model.final.grad is not None
    assert torch.count_nonzero(model.first.grad) > 0
    assert torch.count_nonzero(model.final.grad) > 0


def test_correspondence_pair_preserves_set_valued_positive_nodes() -> None:
    source = UIGraph(
        graph_id="source",
        nodes=(
            UINode(node_id="s0", origin_id="s0", clickable=True),
            UINode(node_id="s1", origin_id="s1", clickable=True),
        ),
    )
    target = UIGraph(
        graph_id="target",
        nodes=(
            UINode(node_id="t0", origin_id="t0", clickable=True),
            UINode(node_id="t1", origin_id="t1", clickable=True),
        ),
    )

    pair = make_correspondence_training_pair(
        source,
        target,
        (("s0", "t0"), ("s0", "t1"), ("s1", "t1")),
    )

    assert pair.positive_targets_a_to_b == ((0, 1), (1,))
    assert pair.positive_targets_b_to_a == ((0,), (0, 1))
    assert pair.origin_ids == ("s0->t0", "s0->t1", "s1->t1")


def test_unified_mapping_seed_controls_model_initialization() -> None:
    torch = __import__("pytest").importorskip("torch")
    graph = graph_from_record(
        {
            "screen_id": "seeded-training",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "action",
                    "parent_id": "root",
                    "text": "Continue",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [10, 20, 90, 50],
                },
            ],
        }
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    augment = AugmentConfig(drop_node_prob=0.0, distractor_prob=0.0)
    pair = make_correspondence_training_pair(
        graph,
        graph,
        (("action", "action"),),
        matcher_config=config,
    )

    model_a, history_a = train_mapping_matcher(
        [graph],
        [pair],
        seed=23,
        matcher_config=config,
        augment_config=augment,
    )
    model_b, history_b = train_mapping_matcher(
        [graph],
        [pair],
        seed=23,
        matcher_config=config,
        augment_config=augment,
    )

    assert history_a == history_b
    assert model_a.state_dict().keys() == model_b.state_dict().keys()
    assert all(
        torch.equal(model_a.state_dict()[key], model_b.state_dict()[key])
        for key in model_a.state_dict()
    )


def test_mapping_training_mixes_augmented_and_cross_page_pairs() -> None:
    __import__("pytest").importorskip("torch", exc_type=ImportError)
    source = graph_from_record(
        {
            "screen_id": "mixed-source",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "root", "class": "Root", "bounds": [0, 0, 100, 100]},
                {
                    "node_id": "a",
                    "text": "Alpha",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [5, 10, 45, 30],
                },
                {
                    "node_id": "b",
                    "text": "Beta",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [55, 10, 95, 30],
                },
                {
                    "node_id": "label",
                    "text": "Title",
                    "class": "TextView",
                    "bounds": [5, 40, 95, 60],
                },
            ],
        }
    )
    target = graph_from_record(
        {
            "screen_id": "mixed-target",
            "width": 120,
            "height": 100,
            "nodes": [
                {"node_id": "root-t", "class": "Root", "bounds": [0, 0, 120, 100]},
                {
                    "node_id": "x",
                    "text": "Alpha",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [10, 15, 50, 35],
                },
                {
                    "node_id": "y",
                    "text": "Beta",
                    "class": "Button",
                    "clickable": True,
                    "bounds": [60, 15, 110, 35],
                },
                {
                    "node_id": "label-t",
                    "text": "Title",
                    "class": "TextView",
                    "bounds": [10, 45, 110, 65],
                },
            ],
        }
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    cross_page_pair = make_correspondence_training_pair(
        source,
        target,
        [("a", "x"), ("b", "y")],
        matcher_config=config,
    )

    _, history = train_mapping_matcher(
        [source, target],
        [cross_page_pair],
        epochs=1,
        seed=29,
        matcher_config=config,
        augment_config=AugmentConfig(drop_node_prob=0.0, distractor_prob=0.0),
    )

    assert history == [
        {
            "epoch": 1.0,
            "loss": history[0]["loss"],
            "pairs": 3.0,
            "self_supervised_pairs": 2.0,
            "cross_page_pairs": 1.0,
            "positive_labels": 12.0,
            "assignment_loss": history[0]["assignment_loss"],
            "final_assignment_loss": history[0]["final_assignment_loss"],
            "intermediate_assignment_loss": 0.0,
            "cycle_loss": history[0]["cycle_loss"],
            "supervised_layers": 1.0,
            "context_masked_nodes": 0.0,
        }
    ]


def test_local_context_graph_keeps_anchor_and_bounds_source_cost() -> None:
    graph = graph_from_record(
        {
            "screen_id": "large",
            "width": 1000,
            "height": 100,
            "nodes": [
                {
                    "node_id": f"node-{index}",
                    "class": "TextView",
                    "bounds": [index * 10, 0, index * 10 + 8, 20],
                }
                for index in range(100)
            ],
        }
    )

    context = local_context_graph(graph, anchor_node_id="node-99", max_nodes=16)

    assert len(context.nodes) == 16
    assert any(node.node_id == "node-99" for node in context.nodes)
    assert context.metadata["context_original_nodes"] == 100


def test_correspondence_evaluation_reports_positive_null_and_warm_latency() -> None:
    torch = __import__("pytest").importorskip("torch")
    source = UIGraph(
        graph_id="eval-source",
        width=100,
        height=100,
        nodes=(
            UINode(
                node_id="positive-a",
                origin_id="positive-a",
                bbox=(0, 0, 20, 20),
                clickable=True,
            ),
            UINode(node_id="null-a", origin_id="null-a", bbox=(60, 60, 80, 80)),
        ),
    )
    target = UIGraph(
        graph_id="eval-target",
        width=100,
        height=100,
        nodes=(
            UINode(
                node_id="positive-b",
                origin_id="positive-b",
                bbox=(0, 0, 20, 20),
                clickable=True,
            ),
            UINode(node_id="null-b", origin_id="null-b", bbox=(60, 60, 80, 80)),
        ),
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    pair = make_correspondence_training_pair(
        source,
        target,
        (("positive-a", "positive-b"),),
        matcher_config=config,
        unmatched_as_null=True,
    )

    class PerfectMatcher(torch.nn.Module):
        def forward(self, *inputs):
            source_count = inputs[0].shape[0]
            target_count = inputs[3].shape[0]
            logits_ab = torch.full((source_count, target_count), -10.0)
            logits_ba = torch.full((target_count, source_count), -10.0)
            logits_ab[0, 0] = 10.0
            logits_ba[0, 0] = 10.0
            return {"logits_ab": logits_ab, "logits_ba": logits_ba}

    metrics = evaluate_correspondence_pairs(
        PerfectMatcher(),
        [pair],
        matcher_config=config,
    )

    assert metrics["positive_total"] == 2
    assert metrics["null_total"] == 0
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["recall_at_k"] == {1: 1.0, 3: 1.0, 5: 1.0}
    assert metrics["null_accuracy"] == 0.0
    assert metrics["warm_model_latency_ms"]["samples"] == 2
    assert metrics["warm_input_latency_ms"]["samples"] == 2
    assert metrics["warm_end_to_end_latency_ms"]["samples"] == 2


def test_explicit_no_correspondence_pair_labels_every_node_as_null() -> None:
    source = UIGraph(
        graph_id="null-source",
        nodes=(UINode(node_id="source", origin_id="source"),),
    )
    target = UIGraph(
        graph_id="null-target",
        nodes=(UINode(node_id="target", origin_id="target"),),
    )

    pair = make_correspondence_training_pair(
        source,
        target,
        (),
        allow_empty=True,
        unmatched_as_null=True,
    )

    assert pair.targets_a_to_b == (-1,)
    assert pair.targets_b_to_a == (-1,)
    assert pair.source_indices == ()
    assert pair.target_indices == ()


def test_explicit_correspondence_requires_actionable_nodes_on_both_pages() -> None:
    source = UIGraph(
        graph_id="source",
        nodes=(UINode(node_id="text", origin_id="text", text="Continue"),),
    )
    target = UIGraph(
        graph_id="target",
        nodes=(
            UINode(
                node_id="button",
                origin_id="button",
                text="Continue",
                clickable=True,
            ),
        ),
    )

    with pytest.raises(ValueError, match="actionable-to-actionable"):
        make_correspondence_training_pair(
            source,
            target,
            (("text", "button"),),
        )


def test_partial_correspondence_evaluation_ignores_unannotated_nodes() -> None:
    torch = __import__("pytest").importorskip("torch")
    source = UIGraph(
        graph_id="partial-source",
        nodes=(
            UINode(node_id="matched-a", origin_id="matched-a", clickable=True),
            UINode(node_id="unknown-a", origin_id="unknown-a", clickable=True),
        ),
    )
    target = UIGraph(
        graph_id="partial-target",
        nodes=(
            UINode(node_id="matched-b", origin_id="matched-b", clickable=True),
            UINode(node_id="unknown-b", origin_id="unknown-b", clickable=True),
        ),
    )
    pair = make_correspondence_training_pair(
        source,
        target,
        (("matched-a", "matched-b"),),
    )

    class Matcher(torch.nn.Module):
        def forward(self, *inputs):
            logits_ab = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
            logits_ba = logits_ab.clone()
            return {"logits_ab": logits_ab, "logits_ba": logits_ba}

    metrics = evaluate_correspondence_pairs(Matcher(), [pair], latency_repeats=0)

    assert pair.targets_a_to_b == (0, -2)
    assert pair.targets_b_to_a == (0, -2)
    assert metrics["positive_total"] == 2
    assert metrics["null_total"] == 0
    assert metrics["top1_accuracy"] == 1.0
