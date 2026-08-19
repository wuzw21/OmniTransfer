#!/usr/bin/env python3
"""Audit canonical mapping labels and write a non-destructive cleaned dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omnitransfer.mapping_data_cleaning import clean_ui_correspondence_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = [path.expanduser().resolve() for path in args.input]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = clean_ui_correspondence_records(_read_records(inputs))
    cleaned_path = output_dir / "cleaned.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    _write_jsonl(cleaned_path, result.cleaned)
    _write_jsonl(quarantine_path, result.quarantine)
    manifest = {
        **result.manifest,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in inputs
        ],
        "outputs": {
            "cleaned": _file_evidence(cleaned_path),
            "quarantine": _file_evidence(quarantine_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _read_records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def _write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_evidence(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
