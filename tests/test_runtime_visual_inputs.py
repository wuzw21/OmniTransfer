from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import omnitransfer.runtime as runtime
from omnitransfer.numpy_matcher import NumpyMutualGraphMatcher
from omnitransfer.ui_graph import graph_from_record


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

    result = runtime.action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
        source_screenshot_path=str(source_screenshot),
        target_screenshot_path=str(target_screenshot),
    )

    assert result
    assert captured["source"].metadata["screenshot_path"] == str(source_screenshot)
    assert captured["target"].metadata["screenshot_path"] == str(target_screenshot)


def test_numpy_matcher_uses_raw_rgb_bbox_crops() -> None:
    checkpoint = (
        Path(runtime.__file__).resolve().parent
        / "checkpoints/pair_evidence_mutual_no_null_v3_20260723/no_null_seed17.npz"
    )
    matcher = NumpyMutualGraphMatcher.from_checkpoint(checkpoint)
    pixels = np.zeros((200, 100, 3), dtype=np.uint8)
    pixels[20:80, 10:50, 0] = 255
    visual = {
        "width": 100,
        "height": 200,
        "compression": "zlib",
        "data_base64": base64.b64encode(zlib.compress(pixels.tobytes())).decode(),
    }
    source = graph_from_record(
        {"xml": SOURCE_XML, "visual_rgb": visual},
        graph_id="source",
    )
    target = graph_from_record(
        {"xml": TARGET_XML, "visual_rgb": visual},
        graph_id="target",
    )

    output = matcher._forward(source, target)

    assert np.all(output["visual_available"] == 1.0)
