from omnitransfer import Candidate, Prediction, Query


def test_schema_roundtrip_shape() -> None:
    candidate = Candidate("target-1", bbox=(0.0, 1.0, 2.0, 3.0))
    query = Query("q1", source={"point": [1, 2]}, target_candidates=(candidate,))
    prediction = Prediction("q1", selected_candidate_id="target-1", scores={"target-1": 1.0})

    assert query.target_candidates[0].candidate_id == prediction.selected_candidate_id
