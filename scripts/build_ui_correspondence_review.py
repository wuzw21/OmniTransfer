#!/usr/bin/env python3
"""Build a static HTML reviewer for UI correspondence pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.mapping_pair_review import (
    build_mapping_pair_review,
    serve_mapping_pair_review,
)


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
    parser.add_argument(
        "--icon-only",
        action="store_true",
        help="进入 text-less icon 流式标注模式：Source 点击后显示 matcher 预测",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="icon-only 模式使用的 NumPy OmniTransfer checkpoint",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="生成后启动统一审阅服务，并提供实时 rank_action_candidates API",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manifest = build_mapping_pair_review(
        args.input,
        args.output_dir,
        pair_limit=args.pair_limit,
        complex_only=args.complex_only,
        complex_limit=args.complex_limit,
        external_payload=not args.embed_payload,
        method_tag=args.method_tag,
        icon_only=args.icon_only,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.serve:
        serve_mapping_pair_review(
            args.output_dir.resolve(), host=args.host, port=args.port
        )


if __name__ == "__main__":
    main()
