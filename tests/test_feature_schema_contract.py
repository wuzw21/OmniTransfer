import numpy as np
import pytest

from omnitransfer.learned_matcher import (
    MatcherConfig,
    PEMM_V3_FEATURE_SCHEMA_ID,
    PEMM_V3_FEATURE_SCHEMA_SHA256,
    RCAM_FEATURE_SCHEMA_ID,
    cross_relation_features,
    encode_graph,
)
from omnitransfer.ui_graph import UIGraph, UINode


def test_pemm_checkpoint_keeps_its_training_time_feature_contract() -> None:
    source = _graph(
        "source",
        resource_id="app:id/add_alarm",
        bbox=(10.0, 20.0, 40.0, 60.0),
    )
    target = _graph(
        "target",
        resource_id="app:id/expand_alarm",
        bbox=(50.0, 100.0, 150.0, 300.0),
    )
    config = MatcherConfig(max_tokens=24)

    rcam_source = encode_graph(
        source,
        config=config,
        feature_schema_id=RCAM_FEATURE_SCHEMA_ID,
    )
    rcam_target = encode_graph(
        target,
        config=config,
        feature_schema_id=RCAM_FEATURE_SCHEMA_ID,
    )
    pemm_source = encode_graph(
        source,
        config=config,
        feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
    )
    pemm_target = encode_graph(
        target,
        config=config,
        feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
    )

    assert rcam_source.token_ids == rcam_target.token_ids
    assert pemm_source.token_ids != pemm_target.token_ids
    assert rcam_source.numeric_features[0][4:] == (0.0,) * 14
    assert pemm_source.numeric_features[0][4] == 1.0
    assert np.count_nonzero(
        cross_relation_features(
            source,
            target,
            feature_schema_id=RCAM_FEATURE_SCHEMA_ID,
        )
    ) == 0
    assert np.count_nonzero(
        cross_relation_features(
            source,
            target,
            feature_schema_id=PEMM_V3_FEATURE_SCHEMA_ID,
        )
    ) > 0


def test_unknown_feature_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported matcher feature schema"):
        encode_graph(_graph("page"), feature_schema_id="silent-schema-drift")


def test_pemm_feature_contract_hash_is_frozen() -> None:
    assert PEMM_V3_FEATURE_SCHEMA_SHA256 == (
        "171735252bbdaea3da8c2fd21967963698f89e45c085cc6801172dfff66d2e58"
    )


def _graph(
    graph_id: str,
    *,
    resource_id: str = "app:id/action",
    bbox: tuple[float, float, float, float] = (10.0, 20.0, 40.0, 60.0),
) -> UIGraph:
    return UIGraph(
        graph_id=graph_id,
        width=200.0,
        height=400.0,
        nodes=(
            UINode(
                node_id=f"{graph_id}:node",
                origin_id=f"{graph_id}:origin",
                resource_id=resource_id,
                content_desc="Alarm action",
                class_name="android.widget.ImageButton",
                clickable=True,
                enabled=True,
                bbox=bbox,
            ),
        ),
    )
