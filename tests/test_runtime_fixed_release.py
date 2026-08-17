from types import SimpleNamespace

import pytest

import omnitransfer.runtime as runtime
from omnitransfer.learned_matcher import (
    DIRECT_TEXT_EVIDENCE_ENCODER,
)


SOURCE_XML = """<hierarchy bounds="[0,0][100,100]"><node text="Submit" class="android.widget.Button" clickable="true" enabled="true" bounds="[10,20][50,60]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][200,400]"><node text="Cancel" class="android.widget.Button" clickable="true" enabled="true" bounds="[20,80][80,160]" /><node text="Submit" class="android.widget.Button" clickable="true" enabled="true" bounds="[100,200][180,320]" /></hierarchy>"""


def test_runtime_ignores_checkpoint_override_and_loads_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sentinel = SimpleNamespace(backend="numpy-v9")

    def load(checkpoint: str):
        calls.append(checkpoint)
        return sentinel

    monkeypatch.setattr(runtime, "_load_matcher", load)
    monkeypatch.setenv(
        "OMNITRANSFER_MATCHER_CHECKPOINT",
        "/tmp/unreviewed-experiment.pt",
    )
    monkeypatch.setenv("OMNITRANSFER_MATCHER_DEVICE", "cpu")

    assert runtime._get_matcher() is sentinel
    assert calls == [str(runtime._DEFAULT_MATCHER_CHECKPOINT.resolve())]


def test_frozen_checkpoint_hash_matches_release_manifest() -> None:
    assert (
        runtime.hashlib.sha256(
            runtime._DEFAULT_MATCHER_CHECKPOINT.read_bytes()
        ).hexdigest()
        == runtime._DEFAULT_MATCHER_SHA256
    )


def test_frozen_release_has_no_learned_token_lookup() -> None:
    runtime._load_matcher.cache_clear()
    matcher = runtime._get_matcher()

    assert matcher.backend == "numpy-v9"
    assert matcher.config.text_encoder == DIRECT_TEXT_EVIDENCE_ENCODER
    assert not any("token_embedding" in name for name in matcher.weights)
    assert not any("text_projection" in name for name in matcher.weights)


def test_learned_result_records_frozen_release_and_unambiguous_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Matcher:
        backend = "numpy-v9"

        def predict(
            self,
            _source,
            target,
            *,
            candidate_node_ids,
            min_probability,
            min_margin,
            **_kwargs,
        ):
            assert min_probability == 0.0
            assert min_margin == 0.0
            candidates = [
                node for node in target.nodes if node.node_id in candidate_node_ids
            ]
            selected = next(node for node in candidates if node.text == "Submit")
            other = next(node for node in candidates if node.text == "Cancel")
            return SimpleNamespace(
                target_node=selected,
                probability=0.91,
                margin=0.51,
                reason="learned_match",
                scores=((selected.node_id, 0.73), (other.node_id, 0.22)),
            )

    monkeypatch.setattr(runtime, "_get_matcher", lambda: Matcher())
    monkeypatch.setenv("OMNITRANSFER_MATCHER_MIN_PROBABILITY", "0.99")
    monkeypatch.setenv("OMNITRANSFER_MATCHER_MIN_MARGIN", "0.99")

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
        top_k=2,
    )

    assert result["candidates"]
    assert "mapped" not in result
    assert result["mapping_mode"] == "omnitransfer_direct_text_alignment_v9"
    assert result["matcher_release"] == (
        "omnitransfer-direct-text-alignment-v9.3.1-mobile"
    )
    assert result["matcher_backend"] == "numpy-v9"
    assert (
        result["matcher_checkpoint_sha256"]
        == runtime._DEFAULT_MATCHER_SHA256
    )
    assert result["matcher_feature_schema"] == (
        "omnitransfer-direct-text-spatial-xml-v9"
    )
    assert (
        result["matcher_feature_schema_sha256"]
        == "fc1706baeff6d8b0e43f5da9b03a62a51086472764d440ed7b4c3a41f71c52f8"
    )
    assert result["score"] == pytest.approx(0.91)
    assert result["pair_confidence"] == pytest.approx(0.91)
    assert result["rank_probability"] == pytest.approx(0.73)
    assert result["margin"] == pytest.approx(0.51)
