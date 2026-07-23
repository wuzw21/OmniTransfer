#!/usr/bin/env python3
"""Fetch a small, reproducible Multimodal-Mind2Web rows-server sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_website")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")

    query = urlencode(
        {
            "dataset": "osunlp/Multimodal-Mind2Web",
            "config": "default",
            "split": args.split,
            "offset": args.offset,
            "length": 1,
        }
    )
    endpoint = f"https://datasets-server.huggingface.co/rows?{query}"
    payload = _json(endpoint)
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit("datasets-server returned no rows")
    row = rows[0].get("row") or {}
    raw_html = str(row.get("raw_html") or "")
    if not raw_html:
        raise SystemExit("selected row has no raw_html")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_uid = str(row.get("action_uid") or f"offset-{args.offset}")
    html_path = output / f"{action_uid}.html"
    html_path.write_text(raw_html, encoding="utf-8")
    screenshot = row.get("screenshot") if isinstance(row.get("screenshot"), dict) else {}
    screenshot_path = output / f"{action_uid}_official.jpg"
    if screenshot.get("src"):
        screenshot_path.write_bytes(_bytes(str(screenshot["src"])))
    metadata = {
        "schema_version": "omnitransfer.mind2web_rows_sample.v1",
        "dataset": "osunlp/Multimodal-Mind2Web",
        "split": args.split,
        "offset": args.offset,
        "action_uid": action_uid,
        "annotation_id": row.get("annotation_id"),
        "website": row.get("website"),
        "domain": row.get("domain"),
        "confirmed_task": row.get("confirmed_task"),
        "operation": row.get("operation"),
        "positive_candidates": row.get("pos_candidates"),
        "html_path": str(html_path),
        "official_screenshot_path": str(screenshot_path) if screenshot_path.is_file() else "",
        "official_screenshot_size": {
            "width": screenshot.get("width"),
            "height": screenshot.get("height"),
        },
        "data_boundary": "rows-server raw_html is sufficient for schema smoke, not formal render fidelity",
    }
    (output / "sample.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def _json(url: str) -> dict:
    return json.loads(_bytes(url))


def _bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "omnitransfer-data/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
