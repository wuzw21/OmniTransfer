from types import SimpleNamespace

import pytest

import omnitransfer.runtime as runtime
from omnitransfer import action_transfer
from omnitransfer.runtime import describe_action_target


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


def test_action_transfer_uses_matcher_and_projects_source_offset() -> None:
    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(60, 40),
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert (result["new_x"], result["new_y"]) == (280.0, 260.0)
    assert result["target_bbox"] == [200.0, 220.0, 360.0, 300.0]


def test_action_transfer_prefers_actionable_child_with_same_bounds() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node class="android.widget.FrameLayout" bounds="[104,562][608,675]"><node text="First name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,562][608,675]" /></node></hierarchy>"""
    target_xml = """<hierarchy bounds="[0,0][720,1280]"><node class="android.widget.FrameLayout" bounds="[104,551][608,691]"><node text="First name" class="android.widget.EditText" clickable="true" enabled="true" bounds="[104,551][608,691]" /></node></hierarchy>"""

    result = action_transfer(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(356, 618.5),
    )

    assert result["mapped"] is True
    assert result["src_element"]["class"] == "android.widget.EditText"
    assert result["target_bbox"] == [104.0, 551.0, 608.0, 691.0]


def test_semantic_transfer_ignores_android_display_size() -> None:
    small_display = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(60, 40),
        target_display_size=(1080, 1920),
    )
    large_display = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(60, 40),
        target_display_size=(1440, 3168),
    )

    assert (small_display["new_x"], small_display["new_y"]) == (280.0, 260.0)
    assert (large_display["new_x"], large_display["new_y"]) == (280.0, 260.0)


def test_action_transfer_maps_equivalent_ui_graph_without_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_get_matcher",
        lambda: pytest.fail("equivalent graphs must not invoke the learned matcher"),
    )

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=SOURCE_XML,
        source_point=(60, 40),
        source_package_name="com.example",
        target_package_name="com.example",
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "equivalent_ui_graph"
    assert (result["new_x"], result["new_y"]) == (60.0, 40.0)


def test_action_transfer_ignores_unrelated_dynamic_text_on_equivalent_graph(
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

    result = action_transfer(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(60, 40),
        source_package_name="com.example",
        target_package_name="com.example",
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "equivalent_ui_graph"
    assert (result["new_x"], result["new_y"]) == (60.0, 40.0)


def test_action_transfer_accepts_source_element_id() -> None:
    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_element_id="com.example:id/submit",
    )

    assert result["mapped"] is True
    assert (result["new_x"], result["new_y"]) == (280.0, 260.0)


def test_action_transfer_allows_matcher_to_disambiguate_duplicate_identity(
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

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=target_xml,
        source_point=(60, 40),
        top_k=3,
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert result["target_bbox"] == [200.0, 220.0, 360.0, 300.0]
    assert sum(
        candidate["resource_id"] == "com.example:id/submit"
        for candidate in result["top_candidates"]
    ) == 2


def test_action_transfer_returns_top_candidate_when_matcher_cannot_disambiguate() -> None:
    target_xml = TARGET_XML.replace(
        "</hierarchy>",
        '<node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[20,20][180,100]" /></hierarchy>',
    )

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=target_xml,
        source_point=(60, 40),
        top_k=3,
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert result["selection_policy"] == "top_candidate_required"
    assert result["matcher_reason"] == "learned_low_confidence"
    assert result["target_bbox"] in (
        [20.0, 20.0, 180.0, 100.0],
        [200.0, 220.0, 360.0, 300.0],
    )
    assert isinstance(result["new_x"], float)
    assert isinstance(result["new_y"], float)
    assert sum(
        candidate["resource_id"] == "com.example:id/submit"
        for candidate in result["top_candidates"]
    ) == 2


def test_action_transfer_requires_source_anchor() -> None:
    result = action_transfer(source_xml=SOURCE_XML, target_xml=TARGET_XML)

    assert result == {
        "mapped": False,
        "mapping_mode": "mutual_graph_matcher_no_null_v3",
        "reason": "source_target_missing",
    }


def test_action_transfer_requires_full_source_graph() -> None:
    coordinate_result = action_transfer(
        target_xml=TARGET_XML,
        source_point=(900, 200),
        source_coordinate_space="relative_0_1000",
    )
    descriptor_result = action_transfer(
        target_xml=TARGET_XML,
        source_element={"text": "Submit"},
        source_offset=(0.5, 0.5),
    )

    expected = {
        "mapped": False,
        "mapping_mode": "mutual_graph_matcher_no_null_v3",
        "reason": "source_graph_required",
    }
    assert coordinate_result == expected
    assert descriptor_result == expected


def test_action_transfer_projects_explicit_offset_outside_anchor() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node resource-id="video_name" text="recording.mp4" bounds="[100,700][600,771]" /></hierarchy>"""
    target_xml = """<hierarchy bounds="[0,0][1080,2400]"><node resource-id="video_name" text="recording.mp4" bounds="[189,700][912,771]" /></hierarchy>"""

    result = action_transfer(
        source_xml=source_xml,
        target_xml=target_xml,
        source_element_id="video_name",
        source_offset=(1.138, 0.5),
    )

    assert result["mapped"] is True
    assert result["new_x"] == pytest.approx(1011.774)
    assert result["new_y"] == pytest.approx(735.5)


def test_action_transfer_rejects_unbounded_anchor_offset() -> None:
    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_element_id="com.example:id/submit",
        source_offset=(3.0, 0.5),
    )

    assert result == {
        "mapped": False,
        "mapping_mode": "mutual_graph_matcher_no_null_v3",
        "reason": "source_point_or_offset_required",
    }


def test_describe_action_target_records_related_swipe_endpoint() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node text="Display brightness" bounds="[32,272][688,368]" /></hierarchy>"""

    target = describe_action_target(
        source_xml=source_xml,
        source_point=(640, 320),
        related_point=(719, 320),
    )

    assert target is not None
    assert target["text"] == "Display brightness"
    assert target["offset_x"] == 0.926829268292683
    assert target["offset_y"] == 0.5
    assert target["end_offset_x"] == 1.0
    assert target["end_offset_y"] == 0.5


def test_describe_action_target_rejects_related_point_far_outside_anchor() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node content-desc="Folder: Music, 1 subfolder" clickable="true" bounds="[0,996][720,1108]" /></hierarchy>"""

    target = describe_action_target(
        source_xml=source_xml,
        source_point=(360, 1100),
        related_point=(360, 300),
    )

    assert target is None


def test_describe_action_target_omits_xml_and_dynamic_node_id() -> None:
    target = describe_action_target(source_xml=SOURCE_XML, source_point=(60, 40))

    assert target == {
        "resource_id": "com.example:id/submit",
        "text": "Submit",
        "content_desc": "",
        "class": "android.widget.Button",
        "clickable": True,
        "editable": False,
        "scrollable": False,
        "offset_x": 0.5,
        "offset_y": 0.5,
    }


def test_describe_action_target_records_repeated_occurrence() -> None:
    source_xml = """<hierarchy bounds="[0,0][200,400]"><node text="Date" clickable="true" bounds="[20,100][180,160]" /><node text="Date" clickable="true" bounds="[20,200][180,260]" /></hierarchy>"""

    target = describe_action_target(source_xml=source_xml, source_point=(100, 130))

    assert target is not None
    assert target["occurrence_index"] == 0
    assert target["occurrence_count"] == 2


def test_describe_action_target_records_unlabeled_structural_occurrence() -> None:
    source_xml = """<hierarchy bounds="[0,0][720,1280]"><node content-desc="Settings" clickable="true" bounds="[0,1032][168,1200]" /><node content-desc="Delete" clickable="true" bounds="[164,1074][248,1158]" /><node content-desc="Recording: %s" clickable="true" bounds="[276,1032][444,1200]" /><node clickable="true" bounds="[460,1062][568,1170]" /><node clickable="true" bounds="[552,1032][720,1200]" /></hierarchy>"""

    target = describe_action_target(source_xml=source_xml, source_point=(515, 1116))

    assert target is not None
    assert target["occurrence_index"] == 1
    assert target["occurrence_count"] == 2
    assert target["offset_x"] == pytest.approx(0.5092592593)
