from omnitransfer.learned_matcher import (
    DIRECT_PAIR_EVIDENCE_NAMES,
    direct_pair_evidence_features,
)
from omnitransfer import rank_action_candidates
import omnitransfer.runtime as runtime
from omnitransfer.ui_graph import graph_from_record


SOURCE_XML = """
<AppiumAUT bounds="[0,0][414,736]">
  <XCUIElementTypeButton type="XCUIElementTypeButton"
      label="Notifications" enabled="true" x="207" y="656"
      width="64" height="60" />
</AppiumAUT>
"""

TARGET_XML = """
<hierarchy bounds="[0,0][1080,2270]">
  <node class="android.widget.LinearLayout" enabled="true"
      bounds="[0,960][1080,1496]" />
  <node class="android.widget.FrameLayout" content-desc="Notifications, Tab"
      clickable="true" enabled="true" bounds="[545,2072][710,2193]" />
</hierarchy>
"""


def test_cross_platform_button_label_is_a_strong_semantic_anchor() -> None:
    source = graph_from_record({"xml": SOURCE_XML}, graph_id="source")
    target = graph_from_record({"xml": TARGET_XML}, graph_id="target")
    source_node = source.nodes[1]
    target_node = target.nodes[2]

    assert source_node.clickable is True
    features = dict(
        zip(
            DIRECT_PAIR_EVIDENCE_NAMES,
            direct_pair_evidence_features(source_node, target_node),
            strict=True,
        )
    )
    assert features["semantic_exact"] == 1.0
    assert features["semantic_containment"] == 1.0


def test_runtime_prefers_the_semantic_button_over_a_large_container(
    monkeypatch,
) -> None:
    class Matcher:
        backend = "test-unified-association"

        def predict(self, _source, target, *, candidate_node_ids, **_kwargs):
            candidates = [
                node for node in target.nodes if node.node_id in candidate_node_ids
            ]
            semantic_button = next(
                node for node in candidates if node.content_desc == "Notifications, Tab"
            )
            return type(
                "Match",
                (),
                {
                    "target_node": semantic_button,
                    "probability": 0.95,
                    "margin": 0.8,
                    "reason": "learned_match",
                    "scores": (
                        (semantic_button.node_id, 0.95),
                    ),
                },
            )()

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())
    result = rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(239.0, 686.0),
        top_k=2,
    )

    assert result["candidates"]
    assert result["candidates"][0]["content_desc"] == "Notifications, Tab"
    assert result["candidates"][0]["class"] == "android.widget.FrameLayout"
