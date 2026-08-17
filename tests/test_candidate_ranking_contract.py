from types import SimpleNamespace

import omnitransfer.runtime as runtime


SOURCE_XML = (
    '<hierarchy bounds="[0,0][100,100]">'
    '<node text="Submit" class="android.widget.Button" clickable="true" '
    'enabled="true" bounds="[10,20][50,60]" />'
    "</hierarchy>"
)
TARGET_XML = (
    '<hierarchy bounds="[0,0][200,400]">'
    '<node text="Cancel" class="android.widget.Button" clickable="true" '
    'enabled="true" bounds="[20,80][80,160]" />'
    '<node text="Submit" class="android.widget.Button" clickable="true" '
    'enabled="true" bounds="[100,200][180,320]" />'
    "</hierarchy>"
)


class AbstainingMatcher:
    backend = "test"

    def predict(self, _source, target, **kwargs):
        allowed = set(kwargs["candidate_node_ids"])
        candidates = [
            node
            for node in target.nodes
            if node.node_id in allowed
            and (node.clickable or node.editable or node.scrollable)
        ]
        return SimpleNamespace(
            target_node=None,
            probability=0.0001,
            margin=0.0,
            reason="learned_low_confidence",
            scores=tuple(
                (node.node_id, score)
                for node, score in zip(candidates, (0.6, 0.4), strict=True)
            ),
        )


def test_ranking_returns_every_candidate_when_matcher_abstains(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_get_matcher", lambda: AbstainingMatcher())

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
        top_k=1,
    )

    assert result["schema_version"] == "omnitransfer.candidate-ranking.v1"
    assert result["status"] == "scored"
    assert result["reason"] == "learned_low_confidence"
    assert [candidate["score"] for candidate in result["candidates"]] == [0.6, 0.4]
    assert len(result["candidates"]) == 2
    assert len(result["top_candidates"]) == 1
    assert all(
        set(candidate) >= {"candidate_id", "score", "bbox", "new_x", "new_y"}
        for candidate in result["candidates"]
    )
    assert "mapped" not in result


def test_ranking_returns_complete_contract_when_matcher_is_unavailable(
    monkeypatch,
) -> None:
    def unavailable():
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(runtime, "_get_matcher", unavailable)

    result = runtime.rank_action_candidates(
        source_xml=SOURCE_XML,
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
    )

    assert result["schema_version"] == "omnitransfer.candidate-ranking.v1"
    assert result["status"] == "matcher_unavailable"
    assert result["reason"] == "matcher_unavailable"
    assert result["candidates"] == []
    assert result["top_candidates"] == []
    assert result["score"] is None
    assert result["margin"] is None
    assert result["error"] == "checkpoint missing"
    assert set(result) >= {
        "schema_version",
        "status",
        "reason",
        "mapping_mode",
        "src_element",
        "source_size",
        "target_size",
        "candidates",
        "top_candidates",
        "score",
        "margin",
    }


def test_ranking_returns_complete_contract_when_graph_parse_fails() -> None:
    result = runtime.rank_action_candidates(
        source_xml="<hierarchy>",
        target_xml=TARGET_XML,
        source_point=(30.0, 40.0),
    )

    assert result["schema_version"] == "omnitransfer.candidate-ranking.v1"
    assert result["status"] == "invalid_input"
    assert result["reason"] == "graph_parse_failed"
    assert result["candidates"] == []
    assert result["top_candidates"] == []
    assert "error" in result
    assert "mapped" not in result
