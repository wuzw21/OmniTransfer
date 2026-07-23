from omnitransfer.eval import ranking_metrics, top1_accuracy
from omnitransfer.schema import Candidate, Prediction, Query


def test_set_valued_gold_accepts_equivalent_candidate() -> None:
    query = Query(
        query_id="q1",
        source={},
        target_candidates=(Candidate("gold"), Candidate("equivalent")),
        gold_candidate_id="gold",
        metadata={"gold_equivalent_candidate_ids": ["gold", "equivalent"]},
    )
    prediction = Prediction(
        query_id="q1",
        selected_candidate_id="equivalent",
        scores={"equivalent": 0.8, "gold": 0.7},
    )

    assert top1_accuracy([query], [prediction]) == 1.0
    assert ranking_metrics([query], [prediction]).top1_accuracy == 1.0


def test_ranking_metrics_measure_null_safety_and_coverage() -> None:
    queries = [
        Query("positive", {}, (Candidate("gold"), Candidate("wrong")), "gold"),
        Query("null", {}, (Candidate("unsafe"),), None),
    ]
    predictions = [
        Prediction("positive", "wrong", {"wrong": 0.9, "gold": 0.8}),
        Prediction("null", None, {"unsafe": 0.1}),
    ]

    metrics = ranking_metrics(queries, predictions, ks=(1, 2))

    assert metrics.total == 2
    assert metrics.positive_total == 1
    assert metrics.null_total == 1
    assert metrics.top1_accuracy == 0.5
    assert metrics.recall_at_k == {1: 0.0, 2: 1.0}
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.null_accuracy == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.wrong_target_rate == 0.5
    assert metrics.coverage == 0.5
    assert metrics.selective_accuracy == 0.0
    assert metrics.area_under_risk_coverage == 0.75
    assert metrics.average_latency_ms == 0.0
    assert metrics.p95_latency_ms == 0.0
