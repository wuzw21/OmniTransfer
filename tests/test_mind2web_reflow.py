from omnitransfer.mind2web_reflow import (
    ReflowViewport,
    build_mind2web_reflow_pair,
)


def test_reflow_pair_uses_distinct_node_ids_and_emits_null() -> None:
    source = {
        "nodes": [
            {
                "origin": "backend:10",
                "parentOrigin": "",
                "text": "Menu",
                "className": "button",
                "bbox": [0, 0, 50, 30],
                "clickable": True,
                "labelEligible": True,
            },
            {
                "origin": "backend:11",
                "parentOrigin": "backend:10",
                "text": "Desktop only",
                "className": "a",
                "bbox": [60, 0, 160, 30],
                "clickable": True,
                "labelEligible": True,
            },
        ]
    }
    target = {
        "nodes": [
            {
                "origin": "backend:10",
                "parentOrigin": "",
                "text": "Menu",
                "className": "button",
                "bbox": [0, 0, 40, 40],
                "clickable": True,
                "labelEligible": True,
            }
        ]
    }

    record = build_mind2web_reflow_pair(
        source,
        target,
        source_screenshot="source.png",
        target_screenshot="target.png",
        source_viewport=ReflowViewport("desktop", 1280, 900),
        target_viewport=ReflowViewport("phone", 390, 844),
        task_id="task-1",
        action_id="action-2",
        source_asset="page.mhtml",
    )

    assert record["split"] == "diagnostic"
    assert record["label_status"] == "unreviewed"
    assert record["source"]["graph"]["nodes"][0]["node_id"].startswith("source-")
    assert record["target"]["graph"]["nodes"][0]["node_id"].startswith("target-")
    assert [match["label"] for match in record["matches"]] == [
        "correspondence",
        "no_correspondence",
    ]
