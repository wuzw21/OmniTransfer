from types import SimpleNamespace

import omnitransfer.runtime as runtime


SOURCE_XML = """<hierarchy bounds="[0,0][720,1280]"><node text="Phone" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,881][608,994]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][2208,1840]"><node text="Phone" class="android.widget.EditText" clickable="true" enabled="true" bounds="[1083,1061][2061,1210]" /><node text="Name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[1083,665][2061,814]" /></hierarchy>"""


def test_low_confidence_with_ranked_targets_returns_top_coordinate(monkeypatch) -> None:
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

    result = runtime.action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(356.0, 937.5),
        top_k=2,
    )

    assert result["mapped"] is True
    assert result["target_bbox"] == [1083.0, 1061.0, 2061.0, 1210.0]
    assert result["selection_policy"] == "top_candidate_required"
    assert result["matcher_reason"] == "learned_low_confidence"
    assert (result["new_x"], result["new_y"]) == (1572.0, 1135.5)


def test_no_ranked_target_still_returns_structured_failure(monkeypatch) -> None:
    class Matcher:
        backend = "test"

        def predict(self, *_args, **_kwargs):
            return SimpleNamespace(
                target_node=None,
                probability=0.0,
                margin=0.0,
                reason="target_candidates_missing",
                scores=(),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = runtime.action_transfer(
        source_xml=SOURCE_XML,
        target_xml="<hierarchy />",
        source_point=(356.0, 937.5),
    )

    assert result["mapped"] is False
    assert result["reason"] == "target_candidates_missing"
    assert "new_x" not in result
    assert "new_y" not in result
