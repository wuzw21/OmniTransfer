from __future__ import annotations

from omnitransfer.hard_example_scoring import (
    score_prediction_hardness,
    summarize_hard_examples,
)


def test_hardness_is_model_probability_missing_from_all_gold_endpoints() -> None:
    row = {
        "gold_targets": [
            {"index": 2, "node_id": "gold-a"},
            {"index": 3, "node_id": "gold-b"},
        ],
        "prediction": {"index": 1, "node_id": "wrong"},
        "correct": False,
        "gold_rank": 2,
        "all_candidates": [
            {"index": 1, "node_id": "wrong", "rank_probability": 0.50},
            {"index": 2, "node_id": "gold-a", "rank_probability": 0.30},
            {"index": 3, "node_id": "gold-b", "rank_probability": 0.10},
            {"index": 4, "node_id": "other", "rank_probability": 0.10},
        ],
    }

    scored = score_prediction_hardness(row)

    assert scored["gold_probability"] == 0.4
    assert scored["difficulty_score"] == 60.0
    assert scored["probability_is_exact"] is True


def test_summary_measures_error_concentration_in_hardest_rows() -> None:
    rows = [
        {"difficulty_score": 90.0, "correct": False},
        {"difficulty_score": 70.0, "correct": False},
        {"difficulty_score": 20.0, "correct": True},
        {"difficulty_score": 10.0, "correct": True},
    ]

    summary = summarize_hard_examples(rows, fractions=(0.25, 0.5))

    assert summary["hardest_fractions"]["0.25"] == {
        "rows": 1,
        "error_rate": 1.0,
        "error_capture": 0.5,
    }
    assert summary["hardest_fractions"]["0.50"] == {
        "rows": 2,
        "error_rate": 1.0,
        "error_capture": 1.0,
    }
