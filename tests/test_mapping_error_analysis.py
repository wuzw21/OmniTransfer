from __future__ import annotations

from omnitransfer.mapping_error_analysis import (
    analyze_prediction_errors,
    classify_prediction_error,
)


def _node(
    node_id: str,
    *,
    text: str = "",
    parent_id: str | None = None,
    clickable: bool = False,
) -> dict:
    return {
        "node_id": node_id,
        "text": text,
        "content_desc": "",
        "class_name": "Button" if clickable else "TextView",
        "parent_id": parent_id,
        "bbox": [0.1, 0.1, 0.2, 0.2],
        "clickable": clickable,
    }


def _prediction(
    *,
    source: dict,
    gold: dict,
    predicted: dict,
    descriptor_delta: float,
    gold_rank: int = 2,
) -> dict:
    return {
        "schema_version": "omnitransfer.correspondence_prediction.v1",
        "graph_pair": {"source": "source", "target": "target"},
        "direction": "a_to_b",
        "source": source,
        "gold_targets": [gold],
        "prediction": predicted,
        "correct": False,
        "gold_rank": gold_rank,
        "top_candidates": [predicted, gold],
        "score_components": {
            "descriptor_affinity": {
                "prediction": 0.5,
                "best_gold": 0.5 + descriptor_delta,
                "gold_minus_prediction": descriptor_delta,
            }
        },
    }


def test_classification_separates_relation_hurt_and_parent_child_granularity() -> None:
    source = _node("s", text="Delete")
    gold = _node("parent", text="Delete", clickable=True)
    predicted = _node("child", text="Delete this notification", parent_id="parent")
    row = _prediction(
        source=source,
        gold=gold,
        predicted=predicted,
        descriptor_delta=0.2,
    )

    diagnosis = classify_prediction_error(
        row,
        target_nodes={"parent": gold, "child": predicted},
    )

    assert diagnosis["availability"] == "both_text"
    assert diagnosis["model_stage"] == "relation_hurt"
    assert diagnosis["structural_relation"] == "prediction_descendant_of_gold"
    assert diagnosis["granularity_mismatch"] is True


def test_analysis_reports_hard_slice_coverage_instead_of_only_total_size() -> None:
    text_error = _prediction(
        source=_node("s1", text="Settings"),
        gold=_node("g1", text="Settings"),
        predicted=_node("p1", text="Search"),
        descriptor_delta=-0.1,
    )
    mixed_error = _prediction(
        source=_node("s2", text=""),
        gold=_node("g2", text="Choose existing photo"),
        predicted=_node("p2", text="Take photo"),
        descriptor_delta=0.1,
        gold_rank=5,
    )
    correct = {
        **_prediction(
            source=_node("s3", text="Home"),
            gold=_node("g3", text="Home"),
            predicted=_node("g3", text="Home"),
            descriptor_delta=0.0,
            gold_rank=1,
        ),
        "correct": True,
    }

    report = analyze_prediction_errors(
        [text_error, mixed_error, correct],
        train_availability_counts={"both_text": 100, "mixed": 4, "both_textless": 2},
    )

    assert report["summary"]["total_predictions"] == 3
    assert report["summary"]["errors"] == 2
    assert report["slices"]["mixed"]["errors"] == 1
    assert report["slices"]["mixed"]["error_rate"] == 1.0
    assert report["training_coverage"]["mixed"] == 4
    assert report["diagnosis"]["dataset_too_small"] == "hard_slices_underrepresented"
