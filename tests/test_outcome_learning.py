import json

import pytest

from omnitransfer.importers import query_from_dict
from omnitransfer.outcome_learning import (
    OutcomePreference,
    load_outcome_preferences,
    preference_ranking_loss,
)


def _query():
    return query_from_dict(
        {
            "query_id": "q1",
            "source": {"text": "Delete"},
            "target_candidates": [
                {"candidate_id": "correct", "text": "Delete"},
                {"candidate_id": "wrong", "text": "Delete"},
            ],
            "gold_candidate_id": "correct",
        }
    )


def test_verified_attempts_become_success_over_failure_preferences(tmp_path) -> None:
    path = tmp_path / "outcomes.jsonl"
    rows = [
        {"query_id": "q1", "candidate_id": "correct", "verified_success": True},
        {"query_id": "q1", "candidate_id": "wrong", "verified_success": False},
        {
            "query_id": "q1",
            "candidate_id": "wrong",
            "verified_success": False,
            "environment_failure": True,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    preferences = load_outcome_preferences(path, [_query()])

    assert preferences["q1"] == (
        OutcomePreference("q1", "correct", "wrong"),
    )


def test_verified_failure_without_success_prefers_null(tmp_path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text(
        json.dumps(
            {"query_id": "q1", "candidate_id": "wrong", "verified_success": False}
        )
        + "\n",
        encoding="utf-8",
    )

    preferences = load_outcome_preferences(path, [_query()])

    assert preferences["q1"] == (OutcomePreference("q1", None, "wrong"),)


def test_preference_loss_trains_the_same_candidate_logits() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(3, requires_grad=True)
    loss = preference_ranking_loss(
        logits,
        ("correct", "wrong"),
        (OutcomePreference("q1", "correct", "wrong"),),
    )

    assert loss is not None
    loss.backward()

    assert logits.grad[0] < 0.0
    assert logits.grad[1] > 0.0
    assert logits.grad[2] == 0.0
