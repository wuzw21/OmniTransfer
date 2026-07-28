import csv
import json
from pathlib import Path

from PIL import Image

from omnitransfer.mobileviews_multi_app_pairs import (
    build_mobileviews_allowlist_pair_pool,
    build_mobileviews_multi_app_pair_pilot,
    discover_mobileviews_trace_apps,
)


def _trace(root: Path, app: str, *, with_mapping: bool = True) -> None:
    trace = root / app
    states = trace / "states"
    states.mkdir(parents=True)
    for state_id, text in (("1", "Alpha"), ("2", "Beta")):
        views = [
            {
                "temp_id": 0,
                "parent": -1,
                "visible": True,
                "bounds": [[0, 0], [100, 200]],
                "class": "android.widget.FrameLayout",
                "view_str": "root",
            },
            {
                "temp_id": 1,
                "parent": 0,
                "visible": True,
                "bounds": [[10, 10], [60, 50]],
                "class": "android.widget.Button",
                "text": text,
                "clickable": True,
                "view_str": "primary-action",
            },
            {
                "temp_id": 2,
                "parent": 0,
                "visible": True,
                "bounds": [[10, 70], [60, 110]],
                "class": "android.widget.Button",
                "text": "Cancel",
                "clickable": True,
                "view_str": "secondary-action",
            },
        ]
        (states / f"state_{state_id}.json").write_text(
            json.dumps(
                {
                    "state_str": f"content-{state_id}",
                    "state_str_content_free": "same-layout",
                    "width": 100,
                    "height": 200,
                    "views": views,
                }
            ),
            encoding="utf-8",
        )
        Image.new("RGB", (100, 200), "white").save(states / f"screen_{state_id}.jpg")
    if not with_mapping:
        return
    with (trace / "screenshot_state_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "screen_id",
                "state_str",
                "structure_str",
                "vh_json_id",
                "vh_xml_id",
            ],
        )
        writer.writeheader()
        for state_id in ("1", "2"):
            writer.writerow(
                {
                    "screen_id": f"states/screen_{state_id}.jpg",
                    "state_str": f"content-{state_id}",
                    "structure_str": "same-layout",
                    "vh_json_id": f"states/state_{state_id}.json",
                    "vh_xml_id": "",
                }
            )


def test_multi_app_sampler_round_robins_apps_and_caps_structures(
    tmp_path: Path,
) -> None:
    _trace(tmp_path, "app.one")
    _trace(tmp_path, "app.two")

    records, manifest = build_mobileviews_multi_app_pair_pilot(
        tmp_path,
        app_limit=2,
        pairs_per_app=1,
        total_pair_limit=2,
    )

    assert manifest["apps_selected"] == 2
    assert manifest["pairs_selected"] == 2
    assert manifest["sampling_seed"] == 17
    assert manifest["per_app_pair_counts"] == {"app.one": 1, "app.two": 1}
    assert {record["provenance"]["trace"] for record in records} == {
        "app.one",
        "app.two",
    }


def test_multi_app_sampler_reads_native_complete_trace_layout(tmp_path: Path) -> None:
    _trace(tmp_path, "app.native", with_mapping=False)

    records, manifest = build_mobileviews_multi_app_pair_pilot(
        tmp_path,
        app_limit=1,
        pairs_per_app=1,
        total_pair_limit=1,
    )

    assert len(records) == 1
    assert manifest["apps_selected"] == 1
    assert records[0]["provenance"]["trace"] == "app.native"


def test_allowlist_pool_only_reads_frozen_apps(tmp_path: Path) -> None:
    _trace(tmp_path, "app.keep")
    _trace(tmp_path, "app.drop")

    records, manifest = build_mobileviews_allowlist_pair_pool(
        tmp_path,
        ["app.keep"],
    )

    assert manifest["apps_requested"] == 1
    assert manifest["apps_selected"] == 1
    assert {record["provenance"]["trace"] for record in records} == {"app.keep"}


def test_allowlist_pool_indexes_apps_across_nested_batches(tmp_path: Path) -> None:
    _trace(tmp_path / "009_1666_v1", "app.nine")
    _trace(tmp_path / "010_748_v1", "app.ten.apk")

    records, manifest = build_mobileviews_allowlist_pair_pool(
        tmp_path,
        ["app.nine", "app.ten"],
    )

    assert manifest["trace_dirs_discovered"] == 2
    assert manifest["apps_discovered"] == 2
    assert manifest["apps_selected"] == 2
    assert {record["provenance"]["app_id"] for record in records} == {
        "app.nine",
        "app.ten",
    }
    assert {record["provenance"]["trace_source_relative"] for record in records} == {
        "009_1666_v1/app.nine",
        "010_748_v1/app.ten.apk",
    }


def test_discover_trace_apps_reports_paths_and_ambiguity(tmp_path: Path) -> None:
    _trace(tmp_path / "batch.one", "app.unique")
    _trace(tmp_path / "batch.one", "app.same")
    _trace(tmp_path / "batch.two", "app.same.apk")

    app_index, manifest = discover_mobileviews_trace_apps(tmp_path)

    assert set(app_index) == {"app.unique"}
    assert manifest["app_paths"] == {"app.unique": "batch.one/app.unique"}
    assert set(manifest["ambiguous_app_paths"]["app.same"]) == {
        "batch.one/app.same",
        "batch.two/app.same.apk",
    }


def test_discover_trace_apps_prefers_versioned_batches_over_flat_copy(
    tmp_path: Path,
) -> None:
    _trace(tmp_path, "app.duplicate.apk")
    _trace(tmp_path / "010_748_v1", "app.duplicate.apk")

    app_index, manifest = discover_mobileviews_trace_apps(tmp_path)

    assert set(app_index) == {"app.duplicate"}
    assert manifest["source_roots"] == ["010_748_v1"]
    assert manifest["app_paths"] == {"app.duplicate": "010_748_v1/app.duplicate.apk"}


def test_allowlist_pool_rejects_ambiguous_cross_batch_app(tmp_path: Path) -> None:
    _trace(tmp_path / "009_1666_v1", "app.same")
    _trace(tmp_path / "010_748_v1", "app.same.apk")
    _trace(tmp_path / "010_748_v1", "app.unique")

    records, manifest = build_mobileviews_allowlist_pair_pool(
        tmp_path,
        ["app.same", "app.unique"],
    )

    assert {record["provenance"]["app_id"] for record in records} == {"app.unique"}
    assert manifest["rejected"]["ambiguous_app"] == 1
    assert manifest["ambiguous_apps_discovered"] == 1
    assert set(manifest["ambiguous_app_paths"]["app.same"]) == {
        "009_1666_v1/app.same",
        "010_748_v1/app.same.apk",
    }


def test_allowlist_pool_can_emit_self_supervised_training_records(
    tmp_path: Path,
) -> None:
    _trace(tmp_path, "app.train")

    records, manifest = build_mobileviews_allowlist_pair_pool(
        tmp_path,
        ["app.train"],
        split="train",
        label_status="self_supervised",
    )

    assert records
    assert {record["split"] for record in records} == {"train"}
    assert {record["label_status"] for record in records} == {"self_supervised"}
    assert manifest["split"] == "train"
    assert manifest["label_status"] == "self_supervised"


def test_allowlist_pool_rejects_unreviewed_training_records(tmp_path: Path) -> None:
    _trace(tmp_path, "app.train")

    try:
        build_mobileviews_allowlist_pair_pool(
            tmp_path,
            ["app.train"],
            split="train",
            label_status="unreviewed",
        )
    except ValueError as exc:
        assert "diagnostic/unreviewed" in str(exc)
    else:
        raise AssertionError("unreviewed training records must be rejected")


def test_allowlist_pool_rejects_self_supervised_dev_records(tmp_path: Path) -> None:
    _trace(tmp_path, "app.dev")

    try:
        build_mobileviews_allowlist_pair_pool(
            tmp_path,
            ["app.dev"],
            split="dev",
            label_status="self_supervised",
        )
    except ValueError as exc:
        assert "train/self_supervised" in str(exc)
    else:
        raise AssertionError("formal dev records must require reviewed gold labels")


def test_allowlist_pool_marks_diagnostic_dev_reservation(tmp_path: Path) -> None:
    _trace(tmp_path, "app.dev")

    records, manifest = build_mobileviews_allowlist_pair_pool(
        tmp_path,
        ["app.dev"],
        split="diagnostic",
        label_status="unreviewed",
        reserved_split="dev",
    )

    assert {record["provenance"]["reserved_split"] for record in records} == {"dev"}
    assert manifest["reserved_split"] == "dev"


def test_allowlist_pool_rejects_reserved_training_split(tmp_path: Path) -> None:
    _trace(tmp_path, "app.train")

    try:
        build_mobileviews_allowlist_pair_pool(
            tmp_path,
            ["app.train"],
            split="train",
            label_status="self_supervised",
            reserved_split="dev",
        )
    except ValueError as exc:
        assert "reserved_split requires" in str(exc)
    else:
        raise AssertionError("training records cannot reserve a formal split")
