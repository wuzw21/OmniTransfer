#!/usr/bin/env python3
"""Compute the unified OmniTransfer configuration embedding for one UI page."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omnitransfer.page_embedding import OmniTransferPageEmbedder


def _graph_id(path: Path, configured: str | None) -> str:
    if configured:
        return configured
    return f"page:{path.stem}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Page XML path")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--graph-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare-input", type=Path, help="Optional second page XML")
    parser.add_argument("--compare-screenshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_path = args.input.expanduser().resolve()
    embedder = OmniTransferPageEmbedder(args.checkpoint, device=args.device)
    source = embedder.embed(
        source_path.read_text(encoding="utf-8"),
        graph_id=_graph_id(source_path, args.graph_id),
        pixels={"path": str(args.screenshot.expanduser().resolve())}
        if args.screenshot
        else {},
    )
    payload = source.as_dict()
    if args.compare_input:
        compare_path = args.compare_input.expanduser().resolve()
        compare = embedder.embed(
            compare_path.read_text(encoding="utf-8"),
            graph_id=_graph_id(compare_path, None),
            pixels={"path": str(args.compare_screenshot.expanduser().resolve())}
            if args.compare_screenshot
            else {},
        )
        payload["comparison"] = {
            "page": compare.as_dict(),
            "cosine_similarity": source.similarity(compare),
        }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
