from omnitransfer.hard_acceptance import (
    HardMappingCandidate,
    score_hard_mapping_candidate,
    select_balanced_hard_candidates,
)
from omnitransfer.ui_graph import UINode
from scripts.build_hard_mapping_acceptance_set import (
    _build_candidates,
    _strict_actionable_matches,
)


def _candidate(
    task_id: str, app: str, pair: str, *, hard: bool
) -> HardMappingCandidate:
    candidate = HardMappingCandidate(
        task_id=task_id,
        pair_id=pair,
        app=app,
        semantic_key=task_id,
        matcher_correct=not hard,
        selector_correct=not hard,
        selector_abstained=hard,
        methods_disagree=hard,
        matcher_margin=0.01 if hard else 0.8,
        matcher_probability=0.3 if hard else 0.9,
        position_shift=0.5 if hard else 0.0,
        target_area_fraction=0.005 if hard else 0.1,
        textless=hard,
        sequence_transition=hard,
        payload={},
    )
    score, reasons = score_hard_mapping_candidate(candidate)
    candidate.payload["difficulty_score"] = score
    candidate.payload["difficulty_reasons"] = reasons
    return candidate


def test_hard_score_prioritizes_selector_and_matcher_failures() -> None:
    hard = _candidate("hard", "app-a", "pair-a", hard=True)
    easy = _candidate("easy", "app-a", "pair-b", hard=False)

    assert hard.payload["difficulty_score"] > easy.payload["difficulty_score"]
    assert "selector_abstains" in hard.payload["difficulty_reasons"]
    assert "matcher_top1_error" in hard.payload["difficulty_reasons"]


def test_balanced_selection_caps_apps_and_pairs() -> None:
    candidates = [
        _candidate(f"a-{index}", "app-a", "pair-a", hard=True) for index in range(4)
    ] + [
        _candidate(f"b-{index}", "app-b", f"pair-b-{index}", hard=False)
        for index in range(4)
    ]

    selected = select_balanced_hard_candidates(
        candidates,
        task_limit=4,
        max_tasks_per_app=2,
        max_tasks_per_pair=1,
        minimum_difficulty=0.0,
    )

    assert len(selected) == 3
    assert sum(item.app == "app-a" for item in selected) == 1
    assert sum(item.app == "app-b" for item in selected) == 2


def test_review_candidates_require_actionable_nodes_with_click_boxes() -> None:
    source_nodes = {
        "source-action": _node("source-action", clickable=True),
        "source-label": _node("source-label"),
        "source-no-box": _node("source-no-box", clickable=True, bbox=None),
    }
    target_nodes = {
        "target-action": _node("target-action", clickable=True),
        "target-action-2": _node("target-action-2", clickable=True),
        "target-action-3": _node("target-action-3", clickable=True),
        "target-label": _node("target-label"),
        "target-no-box": _node("target-no-box", clickable=True, bbox=None),
    }
    record = {
        "matches": [
            _match("source-action", "target-action"),
            _match("source-label", "target-action-2"),
            _match("source-action", "target-label"),
            _match("source-no-box", "target-action-3"),
            _match("source-action", "target-no-box"),
        ]
    }

    selected = _strict_actionable_matches(
        record,
        source_by_id=source_nodes,
        target_by_id=target_nodes,
    )

    assert selected == [("source-action", "target-action")]


def test_abstained_matcher_keeps_correct_top1_difficulty_label(monkeypatch) -> None:
    class _Matcher:
        def predict(self, *args, **kwargs):
            del args, kwargs
            return type(
                "Match",
                (),
                {
                    "target_node": None,
                    "scores": (("target-action", 0.8), ("target-other", 0.2)),
                    "probability": 0.2,
                    "margin": 0.6,
                    "reason": "learned_low_confidence",
                },
            )()

    class _Selector:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def rank_selector(self, *args, **kwargs):
            del args, kwargs
            return type(
                "SelectorResult",
                (),
                {
                    "candidates": (),
                    "reason": "no_identity_candidate",
                    "latency_ms": 0.0,
                },
            )()

    monkeypatch.setattr(
        "scripts.build_hard_mapping_acceptance_set.AnchorVote64DMatcher",
        _Selector,
    )
    record = _page_pair_record()

    candidates = _build_candidates(
        [record],
        matcher=_Matcher(),
        matcher_min_probability=0.5,
    )

    assert len(candidates) == 1
    assert candidates[0].matcher_correct is True
    assert "matcher_top1_error" not in candidates[0].payload["difficulty_reasons"]
    assert "low_match_probability" in candidates[0].payload["difficulty_reasons"]


def _node(
    node_id: str,
    *,
    clickable: bool = False,
    bbox: tuple[float, float, float, float] | None = (0.0, 0.0, 10.0, 10.0),
) -> UINode:
    return UINode(
        node_id=node_id,
        origin_id=node_id,
        bbox=bbox,
        clickable=clickable,
    )


def _match(source_id: str, target_id: str) -> dict[str, object]:
    return {
        "source_node_id": source_id,
        "target_node_ids": [target_id],
        "label": "correspondence",
    }


def _page_pair_record() -> dict[str, object]:
    source = _node("source-action", clickable=True)
    target = _node("target-action", clickable=True)
    other = _node("target-other", clickable=True)
    return {
        "pair_id": "pair",
        "source": {
            "page_id": "source-page",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "graph_id": "source-page",
                "width": 100.0,
                "height": 100.0,
                "nodes": [source.__dict__],
                "metadata": {},
            },
        },
        "target": {
            "page_id": "target-page",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "graph_id": "target-page",
                "width": 100.0,
                "height": 100.0,
                "nodes": [target.__dict__, other.__dict__],
                "metadata": {},
            },
        },
        "matches": [_match("source-action", "target-action")],
        "provenance": {"trace": "app"},
        "slices": {"track": "trace_state_transition"},
    }
