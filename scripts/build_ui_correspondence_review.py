#!/usr/bin/env python3
"""Build a static HTML reviewer for UI correspondence pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_pair_review import build_mapping_pair_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument(
        "--complex-only",
        action="store_true",
        help="只显示有可观测歧义、无文本元素或布局变化的 page pair",
    )
    parser.add_argument(
        "--complex-limit",
        type=int,
        default=0,
        help="按复杂度降序保留的审核 page pair 数量；0 表示不截断",
    )
    parser.add_argument(
        "--embed-payload",
        action="store_true",
        help="把 payload 内嵌到 HTML；默认由 HTML 从相邻 sidecar JSON 加载",
    )
    parser.add_argument(
        "--method-tag",
        default=None,
        help="只保留带有指定方法标签的 page pair",
    )
    args = parser.parse_args()
    manifest = build_mapping_pair_review(
        args.input,
        args.output_dir,
        pair_limit=args.pair_limit,
        complex_only=args.complex_only,
        complex_limit=args.complex_limit,
        external_payload=not args.embed_payload,
        method_tag=args.method_tag,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
