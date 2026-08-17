from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import omnitransfer.runtime as runtime


SOURCE_XML = (
    '<hierarchy bounds="[0,0][100,200]">'
    '<node class="android.widget.Button" clickable="true" enabled="true" '
    'bounds="[10,20][50,80]" />'
    "</hierarchy>"
)
TARGET_XML = SOURCE_XML.replace("[10,20][50,80]", "[40,60][90,140]")


def test_runtime_attaches_screenshots_to_matcher_graphs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_screenshot = (tmp_path / "source.jpg").resolve()
    target_screenshot = (tmp_path / "target.jpg").resolve()
    source_screenshot.write_bytes(b"source")
    target_screenshot.write_bytes(b"target")
    captured: dict[str, object] = {}

    class Matcher:
        def predict(self, source, target, *, candidate_node_ids, **_kwargs):
            captured["source"] = source
            captured["target"] = target
            target_node = next(
                node for node in target.nodes if node.node_id in set(candidate_node_ids)
            )
            return SimpleNamespace(
                target_node=target_node,
                probability=1.0,
                margin=1.0,
                reason="visual_probe",
                scores=((target_node.node_id, 1.0),),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
        source_screenshot_path=str(source_screenshot),
        target_screenshot_path=str(target_screenshot),
    )

    assert result
    assert captured["source"].metadata["screenshot_path"] == str(source_screenshot)
    assert captured["target"].metadata["screenshot_path"] == str(target_screenshot)
