from omnitransfer.openmobile import (
    attach_openmobile_image,
    graph_from_openmobile_step,
    infer_openmobile_package,
)
from omnitransfer.unified_ui import graph_from_unified_record
from scripts.import_openmobile import assign_package_splits


def _step() -> dict:
    return {
        "step": 3,
        "logical_screen_size": [100, 200],
        "ui_elements": [
            {
                "bbox_pixels": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 200},
                "resource_name": "com.example:id/root",
                "class_name": "android.widget.FrameLayout",
                "package_name": "com.example",
                "is_visible": True,
                "is_enabled": True,
            },
            {
                "bbox_pixels": {"x_min": 10, "x_max": 90, "y_min": 30, "y_max": 80},
                "resource_name": "com.example:id/continue_button",
                "text": "Continue",
                "content_description": "Continue setup",
                "class_name": "android.widget.Button",
                "package_name": "com.example",
                "is_visible": True,
                "is_clickable": True,
                "is_enabled": True,
            },
            {
                "bbox_pixels": {"x_min": 0, "x_max": 100, "y_min": 180, "y_max": 220},
                "text": "Hidden",
                "package_name": "com.example",
                "is_visible": False,
            },
        ],
    }


def test_openmobile_adapter_builds_containment_relations_and_attributes() -> None:
    graph = graph_from_openmobile_step(
        _step(),
        graph_id="openmobile:episode:3",
        package="com.example",
        app="example",
        episode_id="episode",
        action={
            "plan": {"arguments": {"action": "click"}},
            "bbox": [[10, 30], [90, 80]],
            "is_reviewed": True,
        },
    )

    assert graph.width == 100
    assert graph.height == 200
    assert graph.metadata["package"] == "com.example"
    assert graph.metadata["action_type_label"] == "click"
    assert len(graph.nodes) == 2
    assert graph.nodes[0].child_ids == ("n1",)
    assert graph.nodes[1].parent_id == "n0"
    assert graph.nodes[1].depth == 1
    assert graph.nodes[1].bbox == (10.0, 30.0, 90.0, 80.0)
    assert graph.nodes[1].resource_id == "com.example:id/continue_button"
    assert graph.nodes[1].content_desc == "Continue setup"
    assert graph.nodes[1].clickable is True


def test_openmobile_unified_route_and_image_geometry_are_explicit() -> None:
    record = {
        **_step(),
        "source_format": "unigui_openmobile",
        "graph_id": "openmobile:unified",
        "app_package": "com.example",
    }
    graph = graph_from_unified_record(record)
    attached = attach_openmobile_image(
        graph,
        screenshot_path="images/example.png",
        image_size=(100, 200),
    )

    assert attached.metadata["screenshot_path"] == "images/example.png"
    assert attached.metadata["geometry_mismatch"] is False
    assert attach_openmobile_image(
        graph,
        screenshot_path="images/bad.png",
        image_size=(200, 100),
    ).metadata["geometry_mismatch"] is True


def test_openmobile_package_prefers_trajectory_metadata() -> None:
    assert infer_openmobile_package(
        {"app_package": "com.task"},
        _step()["ui_elements"],
    ) == "com.task"


def test_openmobile_small_corpus_uses_exact_disjoint_package_counts() -> None:
    assignments = assign_package_splits(
        {f"app.{index}" for index in range(19)},
        seed=17,
        dev_percent=10,
        test_percent=10,
    )

    assert list(assignments.values()).count("train") == 15
    assert list(assignments.values()).count("dev") == 2
    assert list(assignments.values()).count("test") == 2
    assert assignments == assign_package_splits(
        set(reversed(assignments)),
        seed=17,
        dev_percent=10,
        test_percent=10,
    )
