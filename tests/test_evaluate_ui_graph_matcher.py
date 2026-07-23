import json

from scripts.evaluate_ui_graph_matcher import load_eval_graphs


def test_eval_graph_loader_applies_minimum_and_immutable_limit(tmp_path) -> None:
    input_path = tmp_path / "graphs.test.jsonl"
    records = [
        {
            "graph_id": "small",
            "width": 100,
            "height": 100,
            "nodes": [
                {"node_id": "only", "origin_id": "only", "bbox": [0, 0, 10, 10]}
            ],
        },
        *[
            {
                "graph_id": f"screen-{index}",
                "width": 100,
                "height": 100,
                "nodes": [
                    {
                        "node_id": "a",
                        "origin_id": "a",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "node_id": "b",
                        "origin_id": "b",
                        "bbox": [20, 0, 30, 10],
                    },
                ],
            }
            for index in range(2)
        ],
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    graphs = load_eval_graphs([str(input_path)], limit=1, min_nodes=2)

    assert [graph.graph_id for graph in graphs] == ["screen-0"]
