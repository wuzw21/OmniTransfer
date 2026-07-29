"""Stable JSONL training metrics for reproducible matcher appendices."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


METRIC_LOG_SCHEMA = "omnitransfer.matcher_metrics.v1"


class TrainingMetricLog:
    """Write one self-describing event stream for a matcher training run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def start(self, payload: Mapping[str, Any]) -> None:
        """Start a new run and replace any stale log at the same output path."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.append("run_start", payload)

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        """Append one timestamped, schema-versioned JSON event."""

        if not event:
            raise ValueError("metric event must be non-empty")
        row = {
            "schema_version": METRIC_LOG_SCHEMA,
            "event": event,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    """Hash an immutable training input or produced checkpoint."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment(device: str) -> dict[str, Any]:
    """Capture the minimum hardware/software provenance needed by an appendix."""

    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch
    except Exception as exc:
        metadata["torch_import_error"] = type(exc).__name__
        return metadata
    metadata["torch"] = str(torch.__version__)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        metadata["cuda_version"] = str(torch.version.cuda)
    return metadata


def source_revision() -> str | None:
    """Return the immutable source revision supplied by a release or git."""

    frozen = os.environ.get("OMNITRANSFER_CODE_REVISION", "").strip()
    if frozen:
        return frozen
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None
