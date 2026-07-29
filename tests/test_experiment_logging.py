import json

from omnitransfer.experiment_logging import (
    METRIC_LOG_SCHEMA,
    TrainingMetricLog,
    file_sha256,
    runtime_environment,
    source_revision,
)


def test_training_metric_log_is_append_only_within_one_run(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    log = TrainingMetricLog(path)

    log.start({"seed": 17, "num_layers": 2})
    log.append("epoch_end", {"epoch": 1, "loss": 0.5})
    log.append("evaluation", {"split": "dev", "top1": 0.8})

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "run_start",
        "epoch_end",
        "evaluation",
    ]
    assert all(row["schema_version"] == METRIC_LOG_SCHEMA for row in rows)
    assert rows[0]["num_layers"] == 2
    assert rows[1]["loss"] == 0.5


def test_start_replaces_stale_events_from_a_previous_run(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    log = TrainingMetricLog(path)
    log.start({"run": "old"})
    log.append("epoch_end", {"epoch": 1})

    log.start({"run": "new"})

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["run"] == "new"


def test_appendix_provenance_includes_input_hash_and_runtime(tmp_path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text('{"row": 1}\n', encoding="utf-8")

    assert len(file_sha256(input_path)) == 64
    environment = runtime_environment("cpu")
    assert environment["device"] == "cpu"
    assert "python" in environment
    assert "platform" in environment


def test_source_revision_prefers_frozen_release_environment(monkeypatch) -> None:
    monkeypatch.setenv("OMNITRANSFER_CODE_REVISION", "immutable-commit")

    assert source_revision() == "immutable-commit"
