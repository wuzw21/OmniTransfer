from __future__ import annotations

import numpy as np

from omnitransfer.baselines.anchor_vote_64d import (
    AnchorVote64DMatcher,
    bind_public_bbox,
    cosine_similarity,
    encode_graph_64d,
    normalize_public_bbox,
)
from omnitransfer.ui_graph import UIGraph, UINode


def _node(
    node_id: str,
    bbox: tuple[float, float, float, float],
    *,
    parent_id: str | None = None,
    children: tuple[str, ...] = (),
    text: str = "",
    class_name: str = "android.widget.TextView",
    clickable: bool = False,
) -> UINode:
    return UINode(
        node_id=node_id,
        origin_id=node_id,
        parent_id=parent_id,
        child_ids=children,
        bbox=bbox,
        text=text,
        class_name=class_name,
        clickable=clickable,
    )


def test_historical_encoder_emits_weighted_64d_halves() -> None:
    graph = UIGraph(
        graph_id="page",
        width=100,
        height=200,
        nodes=(
            _node(
                "button",
                (10, 20, 80, 60),
                text="Search",
                class_name="android.widget.Button",
                clickable=True,
            ),
        ),
    )
    vector = encode_graph_64d(graph)["button"]
    assert vector.shape == (64,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector[:32]), 0.55)
    assert np.isclose(np.linalg.norm(vector[32:]), 0.45)


def test_cosine_prefers_same_semantic_element() -> None:
    graph = UIGraph(
        graph_id="page",
        width=100,
        height=200,
        nodes=(
            _node("search_a", (0, 0, 30, 20), text="Search"),
            _node("search_b", (0, 30, 30, 50), text="Search"),
            _node("settings", (0, 60, 30, 80), text="Settings"),
        ),
    )
    vectors = encode_graph_64d(graph)
    assert cosine_similarity(vectors["search_a"], vectors["search_b"]) > cosine_similarity(
        vectors["search_a"], vectors["settings"]
    )


def test_retina_bbox_normalization_and_android_identity() -> None:
    ios = UIGraph(graph_id="ios", nodes=(), width=414, height=736)
    android = UIGraph(graph_id="android", nodes=(), width=1080, height=2270)
    assert normalize_public_bbox((954, 84, 1074, 204), ios, (1242, 2208)) == (
        318,
        28,
        358,
        68,
    )
    assert normalize_public_bbox((805, 96, 937, 228), android, (1080, 2400)) == (
        805,
        96,
        937,
        228,
    )


def test_public_binding_uses_normalized_bbox_and_real_node_semantics() -> None:
    graph = UIGraph(
        graph_id="ios",
        width=414,
        height=736,
        nodes=(
            _node(
                "search",
                (318, 28, 358, 68),
                text="Search",
                class_name="XCUIElementTypeButton",
            ),
        ),
    )
    binding = bind_public_bbox(
        graph,
        (954, 84, 1074, 204),
        "XCUIElementTypeButton",
        screenshot_size=(1242, 2208),
    )
    assert binding is not None
    assert binding.node.node_id == "search"
    assert binding.node.text == "Search"
    assert binding.iou == 1.0


def test_local_anchor_geometry_breaks_semantic_tie() -> None:
    source = UIGraph(
        graph_id="source",
        width=100,
        height=100,
        nodes=(
            _node("root", (0, 0, 100, 100), children=("anchor", "action")),
            _node("anchor", (10, 10, 30, 30), parent_id="root", text="Account"),
            _node(
                "action",
                (50, 10, 70, 30),
                parent_id="root",
                text="Open",
                class_name="android.widget.Button",
                clickable=True,
            ),
        ),
    )
    target = UIGraph(
        graph_id="target",
        width=200,
        height=200,
        nodes=(
            _node(
                "root",
                (0, 0, 200, 200),
                children=("anchor", "correct", "wrong"),
            ),
            _node("anchor", (20, 20, 60, 60), parent_id="root", text="Account"),
            _node(
                "correct",
                (100, 20, 140, 60),
                parent_id="root",
                text="Open",
                class_name="android.widget.Button",
                clickable=True,
            ),
            _node(
                "wrong",
                (20, 130, 60, 170),
                parent_id="root",
                text="Open",
                class_name="android.widget.Button",
                clickable=True,
            ),
        ),
    )
    result = AnchorVote64DMatcher(source, target).rank("action")
    assert result.ranked_node_ids[0] == "correct"
    assert result.candidates[0].anchor_count >= 1
    assert 100 <= result.candidates[0].projected_point[0] <= 140
    assert 20 <= result.candidates[0].projected_point[1] <= 60


def test_similarity_ablation_reports_no_anchor_evidence() -> None:
    source = UIGraph(
        graph_id="source",
        width=100,
        height=100,
        nodes=(_node("action", (10, 10, 30, 30), text="Search"),),
    )
    target = UIGraph(
        graph_id="target",
        width=100,
        height=100,
        nodes=(
            _node("search", (50, 10, 70, 30), text="Search"),
            _node("settings", (50, 50, 70, 70), text="Settings"),
        ),
    )
    result = AnchorVote64DMatcher(source, target).rank("action", use_anchors=False)
    assert result.ranked_node_ids[0] == "search"
    assert all(candidate.anchor_count == 0 for candidate in result.candidates)
    assert all(candidate.geometric_log_score == 0.0 for candidate in result.candidates)


def test_identity_selector_requires_unique_exact_identity() -> None:
    source = UIGraph(
        graph_id="source",
        width=100,
        height=100,
        nodes=(_node("action", (10, 10, 30, 30), text="Search"),),
    )
    unique_target = UIGraph(
        graph_id="target",
        width=100,
        height=100,
        nodes=(
            _node("search", (50, 10, 70, 30), text="Search"),
            _node("settings", (50, 50, 70, 70), text="Settings"),
        ),
    )
    unique = AnchorVote64DMatcher(source, unique_target).rank_selector("action")
    assert unique.selected_node_id == "search"
    assert unique.execute is True
    duplicate_target = UIGraph(
        graph_id="target",
        width=100,
        height=100,
        nodes=(
            _node("search_a", (50, 10, 70, 30), text="Search"),
            _node("search_b", (50, 50, 70, 70), text="Search"),
        ),
    )
    duplicate = AnchorVote64DMatcher(source, duplicate_target).rank_selector("action")
    assert duplicate.selected_node_id is None
    assert duplicate.reason == "target_identity_not_unique"


def test_explicit_candidate_scope_excludes_other_target_nodes() -> None:
    source = UIGraph(
        graph_id="source",
        width=100,
        height=100,
        nodes=(_node("action", (10, 10, 30, 30), text="Search"),),
    )
    target = UIGraph(
        graph_id="target",
        width=100,
        height=100,
        nodes=(
            _node("excluded", (10, 10, 30, 30), text="Search"),
            _node("allowed", (50, 50, 70, 70), text="Settings"),
        ),
    )
    matcher = AnchorVote64DMatcher(source, target)
    similarity = matcher.rank(
        "action",
        candidate_node_ids=("allowed",),
        use_anchors=False,
    )
    selector = matcher.rank_selector(
        "action",
        candidate_node_ids=("allowed",),
    )
    assert similarity.ranked_node_ids == ("allowed",)
    assert selector.selected_node_id is None
