import pytest

from omnitransfer.benchmark import (
    fine_tune_matcher,
    latency_summary,
    partition_queries_by_declared_split,
    predict_queries,
    query_graphs,
    split_queries_by_group,
)
from omnitransfer.importers import query_from_dict
from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.schema import Prediction
from omnitransfer.self_supervised import make_correspondence_training_pair
from omnitransfer.ui_graph import UIGraph, UINode


def _query(query_id: str, app: str = "app-a"):
    return query_from_dict(
        {
            "query_id": query_id,
            "source": {
                "text": "Delete",
                "class_name": "Button",
                "bounds": [0, 0, 20, 20],
                "clickable": True,
            },
            "target_candidates": [
                {
                    "candidate_id": "delete-wrapper",
                    "text": "Delete",
                    "class_name": "Button",
                    "bounds": [0, 40, 20, 60],
                    "clickable": True,
                },
                {
                    "candidate_id": "delete-label",
                    "text": "Delete",
                    "class_name": "TextView",
                    "bounds": [0, 40, 20, 60],
                },
                {
                    "candidate_id": "archive",
                    "text": "Archive",
                    "class_name": "Button",
                    "bounds": [40, 40, 60, 60],
                    "clickable": True,
                },
            ],
            "gold_candidate_id": "delete-wrapper",
            "metadata": {"app": app, "widget_type": "text"},
        }
    )


def test_importer_infers_set_valued_bbox_equivalence() -> None:
    query = _query("q1")

    assert query.acceptable_gold_candidate_ids() == (
        "delete-wrapper",
        "delete-label",
    )


def test_query_graph_adapter_preserves_candidate_ids() -> None:
    query = _query("q1")
    source, target, source_node_id = query_graphs(query)

    assert source_node_id == "__source__"
    assert [node.node_id for node in target.nodes] == [
        "delete-wrapper",
        "delete-label",
        "archive",
    ]
    assert source.nodes[0].text == "Delete"


def test_query_graph_adapter_binds_ios_scale_and_target_xml_topology(tmp_path) -> None:
    source_xml = tmp_path / "source.xml"
    source_xml.write_text(
        """<AppiumAUT>
        <XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="100" height="200">
          <XCUIElementTypeButton type="XCUIElementTypeButton" label="Delete" x="10" y="20" width="20" height="10" clickable="true"/>
        </XCUIElementTypeApplication>
        </AppiumAUT>""",
        encoding="utf-8",
    )
    target_xml = tmp_path / "target.xml"
    target_xml.write_text(
        """<hierarchy width="300" height="600">
        <android.widget.LinearLayout bounds="[0,0][300,600]">
          <android.widget.Button text="Delete" bounds="[30,300][90,330]" clickable="true"/>
          <android.widget.Button text="Archive" bounds="[120,300][210,330]" clickable="true"/>
        </android.widget.LinearLayout>
        </hierarchy>""",
        encoding="utf-8",
    )
    query = query_from_dict(
        {
            "query_id": "scaled-ios",
            "source": {
                "text": "Delete",
                "class_name": "XCUIElementTypeButton",
                "bounds": [30, 60, 90, 90],
                "x": 60,
                "y": 75,
                "clickable": True,
            },
            "target_candidates": [
                {
                    "candidate_id": "delete",
                    "text": "Delete",
                    "class_name": "android.widget.Button",
                    "bounds": [30, 300, 90, 330],
                    "node_index": 2,
                    "clickable": True,
                },
                {
                    "candidate_id": "archive",
                    "text": "Archive",
                    "class_name": "android.widget.Button",
                    "bounds": [120, 300, 210, 330],
                    "node_index": 3,
                    "clickable": True,
                },
                {
                    "candidate_id": "public-delete",
                    "text": "Delete",
                    "class_name": "android.widget.Button",
                    "bounds": [30, 300, 90, 330],
                    "clickable": True,
                },
            ],
            "gold_candidate_id": "delete",
            "metadata": {
                "source_xml_path": str(source_xml),
                "target_xml_path": str(target_xml),
                "source_coordinate_width": 300,
                "source_coordinate_height": 600,
            },
        }
    )

    source, target, source_node_id = query_graphs(query)

    source_anchor = next(
        node for node in source.nodes if node.node_id == source_node_id
    )
    target_by_id = {node.node_id: node for node in target.nodes}
    assert source_node_id != "__source__"
    assert source_anchor.bbox == (30.0, 60.0, 90.0, 90.0)
    assert source.metadata["source_anchor_bound"] is True
    assert len({node.node_id for node in source.nodes}) == len(source.nodes)
    assert target_by_id["delete"].parent_id == "0.0"
    assert target_by_id["archive"].parent_id == "0.0"
    assert target_by_id["public-delete"].parent_id == "0.0"
    assert target.metadata["target_candidates_bound"] == 3
    assert sum(node.parent_id is not None for node in target.nodes) > 0


def test_group_split_keeps_apps_disjoint() -> None:
    queries = [_query(f"a-{index}", "app-a") for index in range(3)]
    queries += [_query(f"b-{index}", "app-b") for index in range(3)]
    queries += [_query(f"c-{index}", "app-c") for index in range(3)]
    splits = split_queries_by_group(queries)

    app_splits = {}
    for split, rows in splits.items():
        for row in rows:
            app = row.metadata["app"]
            assert app not in app_splits or app_splits[app] == split
            app_splits[app] = split


def test_declared_benchmark_split_is_used_without_rehashing() -> None:
    queries = []
    for split in ("train", "dev", "test"):
        query = _query(f"q-{split}", app="shared-app")
        queries.append(
            type(query)(
                query_id=query.query_id,
                source=query.source,
                target_candidates=query.target_candidates,
                gold_candidate_id=query.gold_candidate_id,
                metadata={**query.metadata, "split": split},
            )
        )

    splits = partition_queries_by_declared_split(queries)

    assert {
        name: [row.query_id for row in values] for name, values in splits.items()
    } == {
        "train": ["q-train"],
        "dev": ["q-dev"],
        "test": ["q-test"],
    }


def test_supervised_fine_tune_runs_on_set_valued_gold() -> None:
    pytest.importorskip("torch")
    query = _query("q1")
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)

    result = fine_tune_matcher(
        [query],
        config=config,
        epochs=1,
        learning_rate=1e-3,
    )
    predictions = predict_queries(result.model, [query], config=config)

    assert result.train_queries == 1
    assert len(result.history) == 1
    assert predictions[0].selected_candidate_id in {
        None,
        "delete-wrapper",
        "delete-label",
        "archive",
    }
    assert predictions[0].metadata["input_ms"] >= 0.0
    assert predictions[0].metadata["model_ms"] >= 0.0


def test_fine_tune_can_mix_self_supervised_null_examples() -> None:
    pytest.importorskip("torch")
    query = _query("q-joint")
    graph = UIGraph(
        graph_id="joint-graph",
        width=100,
        height=100,
        nodes=tuple(
            UINode(
                node_id=f"node-{index}",
                origin_id=f"node-{index}",
                text=f"item {index}",
                bbox=(index * 10, 0, index * 10 + 8, 8),
                clickable=True,
            )
            for index in range(5)
        ),
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)

    result = fine_tune_matcher(
        [query],
        config=config,
        epochs=1,
        learning_rate=1e-3,
        self_supervised_graphs=[graph],
        self_supervised_weight=0.2,
        self_supervised_interval=1,
    )

    assert result.history[0]["self_supervised_pairs"] == 1.0
    assert result.history[0]["self_supervised_positive_labels"] > 0.0


def test_fine_tune_interleaves_gold_queries_and_weak_correspondence_pairs() -> None:
    pytest.importorskip("torch")
    query = _query("q-mixed")
    source = UIGraph(
        graph_id="weak-source",
        width=100,
        height=100,
        nodes=(
            UINode(
                node_id="source-target",
                origin_id="source-target",
                bbox=(5, 5, 25, 25),
                clickable=True,
            ),
            UINode(
                node_id="source-null",
                origin_id="source-null",
                bbox=(60, 60, 80, 80),
                clickable=True,
            ),
        ),
    )
    target = UIGraph(
        graph_id="weak-target",
        width=200,
        height=100,
        nodes=(
            UINode(
                node_id="target-target",
                origin_id="target-target",
                bbox=(10, 5, 50, 25),
                clickable=True,
            ),
            UINode(
                node_id="target-null",
                origin_id="target-null",
                bbox=(120, 60, 160, 80),
                clickable=True,
            ),
        ),
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)
    pair = make_correspondence_training_pair(
        source,
        target,
        (("source-target", "target-target"),),
        matcher_config=config,
    )

    result = fine_tune_matcher(
        [query],
        config=config,
        epochs=1,
        learning_rate=1e-3,
        correspondence_pairs=[pair],
        correspondence_weight=0.3,
    )

    assert result.train_queries == 1
    assert result.train_correspondence_pairs == 1
    assert result.history[0]["correspondence_pairs"] == 1.0
    assert result.history[0]["correspondence_loss"] > 0.0


def test_latency_gate_requires_every_image_under_budget() -> None:
    summary = latency_summary(
        [
            Prediction("fast", None, {}, {"latency_ms": 12.0, "model_ms": 4.0}),
            Prediction("slow", None, {}, {"latency_ms": 51.0, "model_ms": 5.0}),
        ],
        budget_ms=50.0,
    )

    assert summary.samples == 2
    assert summary.max_latency_ms == 51.0
    assert summary.under_budget_rate == 0.5
    assert summary.budget_met is False


def test_supervised_fine_tune_rejects_missing_gold_candidate() -> None:
    pytest.importorskip("torch")
    query = _query("q-missing")
    invalid = type(query)(
        query_id=query.query_id,
        source=query.source,
        target_candidates=query.target_candidates,
        gold_candidate_id="absent",
        metadata={"app": "app-missing"},
    )
    config = MatcherConfig(hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0)

    with pytest.raises(ValueError, match="gold ids absent"):
        fine_tune_matcher([invalid], config=config, epochs=1)
