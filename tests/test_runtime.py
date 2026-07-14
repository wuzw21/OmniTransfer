from omnitransfer import action_transfer


SOURCE_XML = """<hierarchy><node bounds="[0,0][200,100]"><node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[10,10][110,70]" /></node></hierarchy>"""
TARGET_XML = """<hierarchy><node bounds="[0,0][400,400]"><node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[200,220][360,300]" /></node></hierarchy>"""


def test_action_transfer_exposes_existing_omniflow_matcher() -> None:
    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(60, 40),
    )

    assert result["mapped"] is True
    assert result["target_center"] == [280.0, 260.0]
    assert result["target_bbox"] == [200.0, 220.0, 360.0, 300.0]


def test_action_transfer_accepts_source_element_id() -> None:
    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_element_id="com.example:id/submit",
    )

    assert result["mapped"] is True
    assert result["target_candidate_id"] == "com.example:id/submit"


def test_action_transfer_rejects_ambiguous_targets() -> None:
    duplicate_target_xml = TARGET_XML.replace(
        "</hierarchy>",
        '<node resource-id="com.example:id/submit" text="Submit" class="android.widget.Button" clickable="true" bounds="[20,20][180,100]" /></hierarchy>',
    )

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=duplicate_target_xml,
        source_point=(60, 40),
    )

    assert result == {
        "mapped": False,
        "mapping_mode": "ambiguous",
        "reason": "target_identity_not_unique",
    }


def test_action_transfer_requires_source_anchor() -> None:
    result = action_transfer(source_xml=SOURCE_XML, target_xml=TARGET_XML)

    assert result == {
        "mapped": False,
        "mapping_mode": "missing_source_target",
        "reason": "source_point_or_element_id_required",
    }
