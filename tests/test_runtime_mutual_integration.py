from types import SimpleNamespace

import omnitransfer.runtime as runtime
from omnitransfer import action_transfer
from omnitransfer.runtime import runtime_preflight


SOURCE_XML = """<hierarchy bounds="[0,0][200,400]"><node text="Connected devices" class="android.widget.TextView" clickable="true" bounds="[20,100][180,160]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][400,800]"><node text="Network &amp; internet" class="android.widget.TextView" clickable="true" bounds="[40,100][360,200]" /><node text="Connected devices" class="android.widget.TextView" clickable="true" bounds="[40,300][360,400]" /></hierarchy>"""


def test_action_transfer_uses_mutual_matcher_for_full_graphs(monkeypatch) -> None:
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

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
        action_type="click",
    )

    assert len(calls) == 1
    assert calls[0][2]["source_node_id"] == "0.0"
    assert result["mapped"] is True
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert result["target_bbox"] == [40.0, 300.0, 360.0, 400.0]
    assert result["score"] == 0.91
    assert result["margin"] == 0.42


def test_action_transfer_fails_closed_when_matcher_is_unavailable(monkeypatch) -> None:
    def unavailable():
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(runtime, "_get_matcher", unavailable, raising=False)

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
    )

    assert result["mapped"] is False
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert result["reason"] == "matcher_unavailable"
    assert "checkpoint missing" in result["error"]


def test_action_transfer_rejects_cross_package_pages_before_matching(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_get_matcher",
        lambda: (_ for _ in ()).throw(AssertionError("matcher must not run")),
        raising=False,
    )

    result = action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(100, 130),
        source_package_name="com.android.settings",
        target_package_name="com.google.android.apps.nexuslauncher",
    )

    assert result == {
        "mapped": False,
        "mapping_mode": "page_identity",
        "reason": "target_page_identity_mismatch",
        "source_package_name": "com.android.settings",
        "target_package_name": "com.google.android.apps.nexuslauncher",
    }


def test_action_transfer_never_replays_relative_source_coordinates() -> None:
    result = action_transfer(
        target_xml=TARGET_XML,
        source_point=(900, 200),
        source_coordinate_space="relative_0_1000",
    )

    assert result == {
        "mapped": False,
        "mapping_mode": "mutual_graph_matcher_no_null_v3",
        "reason": "source_graph_required",
    }


def test_runtime_uses_numpy_checkpoint_when_pytorch_is_unavailable(
    monkeypatch,
) -> None:
    runtime._load_matcher.cache_clear()
    monkeypatch.setattr(
        runtime.MutualGraphMatcher,
        "from_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("mutual matching requires PyTorch")
        ),
    )

    matcher = runtime._load_matcher(
        str(runtime._DEFAULT_MATCHER_CHECKPOINT.resolve()),
        str(runtime._DEFAULT_NUMPY_MATCHER_CHECKPOINT.resolve()),
        "cpu",
    )

    assert matcher.backend == "numpy"


def test_runtime_preflight_executes_the_canonical_matcher() -> None:
    result = runtime_preflight()

    assert result["ready"] is True
    assert result["backend"] in {"numpy", "pytorch"}
