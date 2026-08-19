from __future__ import annotations

from pathlib import Path

import pytest

from omnitransfer.ios_adapter import IOSAdapterError, load_ios_screen


IOS_XML = """
<AppiumAUT type="Application" width="414" height="736">
  <XCUIElementTypeWindow type="Window" x="0" y="0" width="414" height="736">
    <XCUIElementTypeButton name="Notifications" label="Notifications" enabled="true" x="0" y="330" width="207" height="138" visual_bbox="0,330,621,468"/>
    <XCUIElementTypeButton name="Disabled" label="Disabled" enabled="false" x="0" y="500" width="207" height="50"/>
  </XCUIElementTypeWindow>
</AppiumAUT>
"""


def _raw_screen(tmp_path: Path, *, platform: str = "ios") -> dict[str, object]:
    xml_path = tmp_path / "screen.xml"
    xml_path.write_text(IOS_XML, encoding="utf-8")
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(b"not-an-image")
    return {
        "screen_id": "ios-screen",
        "platform": platform,
        "xml_path": "screen.xml",
        "screenshot_path": "screen.png",
        "original_xml_width": 414,
        "original_xml_height": 736,
        "screenshot_width": 1242,
        "screenshot_height": 2208,
    }


def test_load_ios_screen_is_the_single_coordinate_normalization_boundary(tmp_path: Path) -> None:
    screen = load_ios_screen(_raw_screen(tmp_path), root=tmp_path)

    assert screen["platform"] == "ios"
    assert screen["display_width"] == 1242.0
    assert screen["display_height"] == 2208.0
    assert screen["adapter"]["name"] == "omnitransfer.ios_screen_adapter.v1"
    assert screen["graph"]["width"] == 414.0
    assert screen["graph"]["height"] == 736.0
    assert screen["graph"]["metadata"]["visual_display_size"] == [1242.0, 2208.0]

    button = next(
        node for node in screen["graph"]["nodes"] if node["text"] == "Notifications"
    )
    assert button["bbox"] == [0.0, 330.0, 207.0, 468.0]
    assert button["clickable"] is True
    assert button["visual_bbox"] == [0.0, 330.0, 621.0, 468.0]
    assert button["metadata"]["visual_bbox_coordinate_space"] == "page_pixels"

    disabled = next(node for node in screen["graph"]["nodes"] if node["text"] == "Disabled")
    assert disabled["bbox"] == [0.0, 500.0, 207.0, 550.0]
    assert disabled["clickable"] is True
    assert disabled["enabled"] is False


def test_load_ios_screen_rejects_non_ios_input(tmp_path: Path) -> None:
    with pytest.raises(IOSAdapterError, match="expects an iOS row"):
        load_ios_screen(_raw_screen(tmp_path, platform="android"), root=tmp_path)
