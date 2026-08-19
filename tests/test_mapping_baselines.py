from omnitransfer.mapping_baselines import evaluate_exact_identity_baselines


def _node(node_id: str, *, text: str = "", resource_id: str = "") -> dict:
    return {
        "node_id": node_id,
        "origin_id": node_id,
        "text": text,
        "content_desc": "",
        "resource_id": resource_id,
        "class_name": "Button",
        "bbox": [0, 0, 1, 1],
    }


def test_exact_identity_baselines_report_uncovered_textless_rows() -> None:
    record = {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair",
        "split": "dev",
        "label_status": "gold",
        "source": {
            "page_id": "source",
            "platform": "ios",
            "screenshot_path": "",
            "graph": {
                "nodes": [
                    _node("text-source", text="Save"),
                    _node("id-source", resource_id="shared-id"),
                    _node("icon-source"),
                ]
            },
        },
        "target": {
            "page_id": "target",
            "platform": "android",
            "screenshot_path": "",
            "graph": {
                "nodes": [
                    _node("text-target", text="Save"),
                    _node("id-target", resource_id="shared-id"),
                    _node("icon-target"),
                ]
            },
        },
        "matches": [
            {"source_node_id": "text-source", "target_node_ids": ["text-target"], "label": "correspondence"},
            {"source_node_id": "id-source", "target_node_ids": ["id-target"], "label": "correspondence"},
            {"source_node_id": "icon-source", "target_node_ids": ["icon-target"], "label": "correspondence"},
        ],
        "partition_keys": ["component"],
        "provenance": {},
        "slices": {},
    }

    report = evaluate_exact_identity_baselines([record])

    assert report["baselines"]["text_exact"]["rows"] == 6
    assert report["baselines"]["text_exact"]["covered"] == 2
    assert report["baselines"]["text_exact"]["top1"] == 2
    assert report["baselines"]["resource_id_exact"]["covered"] == 2
    assert report["baselines"]["resource_id_exact"]["top1"] == 2
    assert report["baselines"]["text_or_resource_exact"]["covered"] == 4
    assert report["baselines"]["text_or_resource_exact"]["top1"] == 4
