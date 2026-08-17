from types import SimpleNamespace

import omnitransfer.runtime as runtime
from omnitransfer import rank_action_candidates


SOURCE_XML = """<hierarchy bounds="[0,0][200,400]"><node text="Connected devices" class="android.widget.TextView" clickable="true" bounds="[20,100][180,160]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][400,800]"><node text="Network &amp; internet" class="android.widget.TextView" clickable="true" bounds="[40,100][360,200]" /><node text="Connected devices" class="android.widget.TextView" clickable="true" bounds="[40,300][360,400]" /></hierarchy>"""


def test_ranking_uses_v9_matcher_for_full_graphs(monkeypatch) -> None:
    calls = []

    class FakeMatcher:
        def predict(self, source, target, **kwargs):
            calls.append((source, target, kwargs))
            return SimpleNamespace(
                target_node=target.nodes[2],
                probability=0.91,
                margin=0.42,
                reason="learned_match",
                scores=((target.nodes[2].node_id, 0.91), ("__NULL__", 0.07)),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: FakeMatcher(), raising=False)

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
        action_type="click",
    )

    assert len(calls) == 1
    assert calls[0][2]["source_node_id"] == "0.0"
    assert result["candidates"]
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert result["candidates"][0]["bbox"] == [40.0, 300.0, 360.0, 400.0]
    assert result["score"] == 0.91
    assert result["margin"] == 0.42


def test_ranking_reports_failure_when_matcher_is_unavailable(monkeypatch) -> None:
    def unavailable():
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(runtime, "_get_matcher", unavailable, raising=False)

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
    )

    assert result["candidates"] == []
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert result["reason"] == "matcher_unavailable"
    assert "new_x" not in result
    assert "new_y" not in result
    assert "checkpoint missing" in result["error"]


def test_ranking_scores_pages_without_policy_rejection(monkeypatch) -> None:
    class FakeMatcher:
        def predict(self, _source, target, **_kwargs):
            selected = next(node for node in target.nodes if node.text == "Connected devices")
            return SimpleNamespace(
                target_node=selected,
                probability=0.91,
                margin=0.42,
                reason="learned_match",
                scores=((selected.node_id, 1.0),),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: FakeMatcher(), raising=False)

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
    )

    assert result["candidates"]
    assert result["candidates"][0]["score"] == 1.0
