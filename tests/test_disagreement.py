from __future__ import annotations

from omnitransfer.disagreement import (
    group_disagreements_by_page_pair,
    mine_disagreements,
    prediction_from_mapping,
)
from omnitransfer.schema import Candidate, Prediction, Query


def _query(query_id: str, *, gold: str | None = "gold") -> Query:
    return Query(
        query_id=query_id,
        source={"text": "download"},
        target_candidates=(
            Candidate("wrong", (0.0, 0.0, 10.0, 10.0)),
            Candidate("other", (10.0, 10.0, 20.0, 20.0)),
            Candidate("gold", (20.0, 20.0, 40.0, 40.0)),
        ),
        gold_candidate_id=gold,
        metadata={
            "duplicate_candidate_signatures": 2,
            "app": "Browser",
            "source_screenshot_path": "source.png",
            "target_screenshot_path": "target.png",
        },
    )


def test_mines_learned_correct_selector_confident_wrong() -> None:
    query = _query("q1")
    learned = Prediction(
        "q1",
        "gold",
        {"gold": 0.7, "other": 0.2, "wrong": 0.1},
        {"margin": 0.5},
    )
    selector = Prediction(
        "q1",
        "wrong",
        {"wrong": 0.9, "other": 0.08, "gold": 0.02},
        {"margin": 0.82},
    )

    cases = mine_disagreements([query], [learned], [selector])

    assert len(cases) == 1
    assert cases[0].category == "learned_correct_selector_wrong"
    assert cases[0].learned_gold_rank == 1
    assert cases[0].selector_gold_rank == 3
    assert cases[0].rank_advantage == 2
    assert cases[0].disagreement_score > 2.0
    assert cases[0].gold_target_points == ((30.0, 30.0),)


def test_mines_null_safety_without_treating_selector_as_gold() -> None:
    query = _query("q-null", gold=None)
    learned = Prediction("q-null", None, {"wrong": 0.2}, {"margin": 0.1})
    selector = Prediction("q-null", "wrong", {"wrong": 0.9}, {"margin": 0.8})

    cases = mine_disagreements([query], [learned], [selector])

    assert cases[0].category == "learned_null_selector_false_positive"
    assert cases[0].gold_candidate_ids == ()


def test_set_valued_gold_avoids_false_disagreement() -> None:
    query = Query(
        query_id="q-set",
        source={},
        target_candidates=(Candidate("wrapper"), Candidate("label")),
        gold_candidate_id="wrapper",
        metadata={"gold_equivalent_candidate_ids": ["label"]},
    )
    learned = Prediction("q-set", "label", {"label": 0.8, "wrapper": 0.2})
    selector = Prediction("q-set", "wrapper", {"wrapper": 0.7, "label": 0.3})

    assert mine_disagreements([query], [learned], [selector]) == []


def test_prediction_mapping_normalizes_rich_eval_shape() -> None:
    prediction = prediction_from_mapping(
        {
            "query_id": "q",
            "selected_candidate_id": "NULL",
            "margin": 0.4,
            "top_candidates": [
                {"candidate_id": "a", "score": 0.6},
                {"candidate_id": "NULL", "score": 0.4},
            ],
        }
    )

    assert prediction.selected_candidate_id is None
    assert prediction.scores == {"a": 0.6}
    assert prediction.metadata["margin"] == 0.4


def test_groups_multiple_anchors_on_one_page_pair() -> None:
    query_a = _query("q1")
    query_b = _query("q2")
    learned = [
        Prediction("q1", "gold", {"gold": 0.8, "wrong": 0.2}),
        Prediction("q2", "gold", {"gold": 0.7, "wrong": 0.3}),
    ]
    selector = [
        Prediction("q1", "wrong", {"wrong": 0.8, "gold": 0.2}),
        Prediction("q2", "wrong", {"wrong": 0.7, "gold": 0.3}),
    ]

    page_pairs = group_disagreements_by_page_pair(
        mine_disagreements([query_a, query_b], learned, selector)
    )

    assert len(page_pairs) == 1
    assert page_pairs[0]["anchor_count"] == 2
    assert len(page_pairs[0]["anchors"]) == 2
