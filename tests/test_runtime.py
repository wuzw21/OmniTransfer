from types import SimpleNamespace

import pytest

import omnitransfer.runtime as runtime
from omnitransfer import rank_action_candidates


SOURCE_XML = """<hierarchy><node bounds="[0,0][200,100]"><node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[10,10][110,70]" /></node></hierarchy>"""
TARGET_XML = """<hierarchy><node bounds="[0,0][400,400]"><node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[200,220][360,300]" /></node></hierarchy>"""


@pytest.fixture(autouse=True)
def deterministic_matcher(monkeypatch) -> None:
    class Matcher:
        def predict(
            self,
            source,
            target,
            *,
            source_node_id,
            candidate_node_ids,
            **_kwargs,
        ):
            source_node = next(node for node in source.nodes if node.node_id == source_node_id)
            candidates = [
                node for node in target.nodes if node.node_id in set(candidate_node_ids)
            ]
            exact = [node for node in candidates if _same_identity(source_node, node)]
            score = 0.91 if len(exact) == 1 else 0.45
            scores = tuple(
                (node.node_id, score if node in exact else 0.01) for node in candidates
            ) + (("__NULL__", 0.08),)
            return SimpleNamespace(
                target_node=exact[0] if len(exact) == 1 else None,
                probability=score,
                margin=0.42 if len(exact) == 1 else 0.0,
                reason="learned_match" if len(exact) == 1 else "learned_low_confidence",
                scores=scores,
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())


def _same_identity(source, target) -> bool:
    pairs = (
        (source.resource_id, target.resource_id),
        (source.text, target.text),
        (source.content_desc, target.content_desc),
    )
    return any(left and left == right for left, right in pairs)


def test_ranking_uses_matcher_and_projects_source_offset() -> None:
    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(60, 40),
    )

    assert result["candidates"]
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert (result["candidates"][0]["new_x"], result["candidates"][0]["new_y"]) == (280.0, 260.0)
    assert result["candidates"][0]["bbox"] == [200.0, 220.0, 360.0, 300.0]
    assert "mapped" not in result


def test_ranking_prefers_actionable_child_with_same_bounds() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node class="android.widget.FrameLayout" bounds="[104,562][608,675]"><node text="First name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,562][608,675]" /></node></hierarchy>"""
    target_xml = """<hierarchy bounds="[0,0][720,1280]"><node class="android.widget.FrameLayout" bounds="[104,551][608,691]"><node text="First name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,551][608,691]" /></node></hierarchy>"""

    result = rank_action_candidates(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(356, 618.5),
    )

    assert result["candidates"]
    assert result["src_element"]["class"] == "android.widget.EditText"
    assert result["candidates"][0]["bbox"] == [104.0, 551.0, 608.0, 691.0]


def test_ranking_maps_equivalent_ui_graph_without_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_get_matcher",
        lambda: pytest.fail("equivalent graphs must not invoke the learned matcher"),
    )

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=SOURCE_XML,
        source_point=(60, 40),
    )

    assert result["candidates"]
    assert result["mapping_mode"] == "equivalent_ui_graph"
    assert (result["candidates"][0]["new_x"], result["candidates"][0]["new_y"]) == (60.0, 40.0)


def test_ranking_ignores_unrelated_dynamic_text_on_equivalent_graph(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_get_matcher",
        lambda: pytest.fail("equivalent graphs must not invoke the learned matcher"),
    )
    source_xml = SOURCE_XML.replace(
        "</hierarchy>",
        '<node resource-id="com.example:id/storage" text="57% used" '
        'class="android.widget.TextView" bounds="[10,75][110,95]" /></hierarchy>',
    )
    target_xml = source_xml.replace('text="57% used"', 'text="53% used"')

    result = rank_action_candidates(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(60, 40),
    )

    assert result["candidates"]
    assert result["mapping_mode"] == "equivalent_ui_graph"
    assert (result["candidates"][0]["new_x"], result["candidates"][0]["new_y"]) == (60.0, 40.0)


def test_ranking_accepts_source_element_id() -> None:
    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_element_id="com.example:id/submit",
    )

    assert result["candidates"]
    assert (result["candidates"][0]["new_x"], result["candidates"][0]["new_y"]) == (280.0, 260.0)


def test_ranking_allows_matcher_to_disambiguate_duplicate_identity(
    monkeypatch,
) -> None:
    target_xml = TARGET_XML.replace(
        "</hierarchy>",
        '<node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[20,20][180,100]" /></hierarchy>',
    )

    class Matcher:
        def predict(self, source, target, *, candidate_node_ids, **_kwargs):
            candidates = [
                node for node in target.nodes if node.node_id in set(candidate_node_ids)
            ]
            selected = max(candidates, key=lambda node: node.bbox[0])
            return SimpleNamespace(
                target_node=selected,
                probability=0.93,
                margin=0.31,
                reason="mutual_match",
                scores=tuple(
                    (node.node_id, 0.93 if node is selected else 0.12)
                    for node in candidates
                ),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=target_xml,
        source_point=(60, 40),
        top_k=3,
    )

    assert result["candidates"]
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert result["candidates"][0]["bbox"] == [200.0, 220.0, 360.0, 300.0]
    assert sum(
        candidate["resource_id"] == "com.example:id/submit"
        for candidate in result["top_candidates"]
    ) == 2


def test_ranking_returns_top_candidate_when_matcher_abstains() -> None:
    target_xml = TARGET_XML.replace(
        "</hierarchy>",
        '<node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[20,20][180,100]" /></hierarchy>',
    )

    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=target_xml,
        source_point=(60, 40),
        top_k=3,
    )

    assert result["candidates"]
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert "selection_policy" not in result
    assert result["reason"] == "learned_low_confidence"
    assert result["candidates"][0]["bbox"] in (
        [20.0, 20.0, 180.0, 100.0],
        [200.0, 220.0, 360.0, 300.0],
    )
    assert isinstance(result["candidates"][0]["new_x"], float)
    assert isinstance(result["candidates"][0]["new_y"], float)
    assert sum(
        candidate["resource_id"] == "com.example:id/submit"
        for candidate in result["top_candidates"]
    ) == 2


def test_ranking_still_fails_without_target_candidates() -> None:
    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml="<hierarchy />",
        source_point=(60, 40),
    )

    assert result["candidates"] == []
    assert result["reason"] in {
        "learned_low_confidence",
        "target_candidates_missing",
    }
    assert "new_x" not in result
    assert "new_y" not in result


def test_ranking_requires_source_anchor() -> None:
    result = rank_action_candidates(source_xml=SOURCE_XML, target_xml=TARGET_XML)

    assert "mapped" not in result
    assert result["status"] == "invalid_input"
    assert result["reason"] == "source_target_missing"
    assert result["candidates"] == []

def test_ranking_projects_explicit_offset_outside_anchor() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node resource-id="video_name" text="recording.mp4" bounds="[100,700][600,771]" /></hierarchy>"""
    target_xml = """<hierarchy bounds="[0,0][1080,2400]"><node resource-id="video_name" text="recording.mp4" bounds="[189,700][912,771]" /></hierarchy>"""

    result = rank_action_candidates(
        source_xml=source_xml,
        target_xml=target_xml,
        source_element_id="video_name",
        source_offset=(1.138, 0.5),
    )

    assert result["candidates"]
    assert result["candidates"][0]["new_x"] == pytest.approx(1011.774)
    assert result["candidates"][0]["new_y"] == pytest.approx(735.5)


def test_ranking_rejects_unbounded_anchor_offset() -> None:
    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_element_id="com.example:id/submit",
        source_offset=(3.0, 0.5),
    )

    assert "mapped" not in result
    assert result["status"] == "invalid_input"
    assert result["reason"] == "source_point_or_offset_required"
    assert result["candidates"] == []
