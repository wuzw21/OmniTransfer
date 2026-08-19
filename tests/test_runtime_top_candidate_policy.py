from types import SimpleNamespace

import omnitransfer.runtime as runtime


SOURCE_XML = """<hierarchy bounds="[0,0][720,1280]"><node text="Phone" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,881][608,994]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][2208,1840]"><node text="Phone" class="android.widget.EditText" clickable="true" enabled="true" bounds="[1083,1061][2061,1210]" /><node text="Name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[1083,665][2061,814]" /></hierarchy>"""


def test_low_confidence_returns_ranked_targets_without_a_decision(monkeypatch) -> None:
    class Matcher:
        backend = "test"

        def predict(self, _source, target, *, candidate_node_ids, **_kwargs):
            candidates = [
                node for node in target.nodes if node.node_id in candidate_node_ids
            ]
            phone = next(node for node in candidates if node.text == "Phone")
            name = next(node for node in candidates if node.text == "Name")
            return SimpleNamespace(
                target_node=None,
                probability=0.001,
                margin=0.0,
                reason="learned_low_confidence",
                scores=((phone.node_id, 0.51), (name.node_id, 0.49)),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(356.0, 937.5),
        top_k=2,
    )

    assert result["reason"] == "learned_low_confidence"
    assert result["candidates"][0]["bbox"] == [1083.0, 1061.0, 2061.0, 1210.0]
    assert result["candidates"][0]["score"] == 0.51
    assert (
        result["candidates"][0]["new_x"],
        result["candidates"][0]["new_y"],
    ) == (1572.0, 1135.5)
    assert "mapped" not in result
    assert "selection_policy" not in result


def test_no_ranked_target_still_returns_structured_failure(monkeypatch) -> None:
    class Matcher:
        def predict(self, *_args, **_kwargs):
            return SimpleNamespace(
                target_node=None,
                probability=0.0,
                margin=0.0,
                reason="target_candidates_missing",
                scores=(),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml="<hierarchy />",
        source_point=(356.0, 937.5),
    )

    assert result["candidates"] == []
    assert result["reason"] == "target_candidates_missing"
    assert "mapped" not in result
    assert "new_x" not in result
    assert "new_y" not in result


def test_click_candidates_exclude_non_actionable_container(monkeypatch) -> None:
    target_xml = """<hierarchy bounds="[0,0][1000,1000]"><node class="android.widget.LinearLayout" enabled="true" bounds="[0,400][1000,600]"><node content-desc="Create" class="android.widget.ImageButton" clickable="true" enabled="true" bounds="[870,430][980,490]" /></node></hierarchy>"""
    seen: dict[str, tuple[str, ...]] = {}

    class Matcher:
        backend = "test"

        def predict(self, _source, target, *, candidate_node_ids, **_kwargs):
            seen["ids"] = tuple(candidate_node_ids)
            candidates = [
                node for node in target.nodes if node.node_id in set(candidate_node_ids)
            ]
            assert len(candidates) == 1
            assert candidates[0].content_desc == "Create"
            return SimpleNamespace(
                target_node=candidates[0],
                probability=1.0,
                margin=1.0,
                reason="learned_match",
                scores=((candidates[0].node_id, 1.0),),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=target_xml,
        source_point=(30.0, 40.0),
    )

    assert result["status"] == "scored"
    assert len(seen["ids"]) == 1
