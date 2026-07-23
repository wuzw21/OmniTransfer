import json
from pathlib import Path


def test_unified_dataset_registry_has_unique_ids_and_webview_coverage() -> None:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "unified_ui_registry.v1.json"
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    datasets = payload["datasets"]
    dataset_ids = [dataset["id"] for dataset in datasets]

    assert len(dataset_ids) == len(set(dataset_ids))
    assert any("webview" in dataset["platforms"] for dataset in datasets)
    assert any(
        "self_supervised_pretrain" in dataset["roles"].values()
        for dataset in datasets
    )
    assert any("frozen_eval" in dataset["roles"].values() for dataset in datasets)
    assert "source_coordinates_as_target" in payload["policies"]["forbidden_model_inputs"]
