from types import SimpleNamespace

import pytest

import omnitransfer.runtime as runtime


SOURCE_XML = """<hierarchy bounds="[0,0][100,100]"><node text="Submit" class="android.widget.Button" clickable="true" enabled="true" bounds="[10,20][50,60]" /></hierarchy>"""
TARGET_XML = """<hierarchy bounds="[0,0][200,400]"><node text="Cancel" class="android.widget.Button" clickable="true" enabled="true" bounds="[20,80][80,160]" /><node text="Submit" class="android.widget.Button" clickable="true" enabled="true" bounds="[100,200][180,320]" /></hierarchy>"""


def test_runtime_ignores_checkpoint_override_and_loads_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    sentinel = SimpleNamespace(backend="numpy")

    def load(checkpoint: str, numpy_checkpoint: str, device: str):
        calls.append((checkpoint, numpy_checkpoint, device))
        return sentinel

    monkeypatch.setattr(runtime, "_load_matcher", load)
    monkeypatch.setenv(
        "OMNITRANSFER_MATCHER_CHECKPOINT",
        "/tmp/unreviewed-experiment.pt",
    )
    monkeypatch.setenv("OMNITRANSFER_MATCHER_DEVICE", "cpu")

    assert runtime._get_matcher() is sentinel
    assert calls == [
        (
            str(runtime._DEFAULT_MATCHER_CHECKPOINT.resolve()),
            str(runtime._DEFAULT_NUMPY_MATCHER_CHECKPOINT.resolve()),
            "cpu",
        )
    ]


def test_frozen_checkpoint_hashes_match_release_manifest() -> None:
    assert (
        runtime.hashlib.sha256(
            runtime._DEFAULT_MATCHER_CHECKPOINT.read_bytes()
        ).hexdigest()
        == runtime._DEFAULT_MATCHER_SHA256
    )
    assert (
        runtime.hashlib.sha256(
            runtime._DEFAULT_NUMPY_MATCHER_CHECKPOINT.read_bytes()
        ).hexdigest()
        == runtime._DEFAULT_NUMPY_MATCHER_SHA256
    )


def test_learned_result_records_frozen_release_and_unambiguous_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Matcher:
        backend = "numpy"

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
            assert min_probability == 0.5
            assert min_margin == 0.15
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

    result = runtime.action_transfer(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
        top_k=2,
    )

    assert result["mapped"] is True
    assert result["mapping_mode"] == "mutual_graph_matcher_no_null_v3"
    assert result["matcher_release"] == "pair-evidence-mutual-matcher-v3.0.1"
    assert result["matcher_backend"] == "numpy"
    assert (
        result["matcher_checkpoint_sha256"]
        == runtime._DEFAULT_NUMPY_MATCHER_SHA256
    )
    assert result["matcher_feature_schema"] == "pemm-v3-node-context-v1"
    assert (
        result["matcher_feature_schema_sha256"]
        == "171735252bbdaea3da8c2fd21967963698f89e45c085cc6801172dfff66d2e58"
    )
    assert result["score"] == pytest.approx(0.91)
    assert result["pair_confidence"] == pytest.approx(0.91)
    assert result["rank_probability"] == pytest.approx(0.73)
    assert result["margin"] == pytest.approx(0.51)


def test_runtime_manifest_names_fixed_model_and_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_get_matcher",
        lambda: SimpleNamespace(backend="pytorch"),
    )

    manifest = runtime.runtime_matcher_manifest()

    assert manifest == {
        "matcher_release": "pair-evidence-mutual-matcher-v3.0.1",
        "matcher_backend": "pytorch",
        "matcher_checkpoint_sha256": runtime._DEFAULT_MATCHER_SHA256,
        "matcher_feature_schema": "pemm-v3-node-context-v1",
        "matcher_feature_schema_sha256": (
            "171735252bbdaea3da8c2fd21967963698f89e45c085cc6801172dfff66d2e58"
        ),
        "mapping_mode": "mutual_graph_matcher_no_null_v3",
        "min_pair_confidence": 0.5,
        "min_rank_margin": 0.15,
    }
