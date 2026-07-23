#!/usr/bin/env python3
"""Normalize action-transfer JSONL into the canonical lightweight schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnitransfer.importers import load_queries, write_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = load_queries(args.input)
    write_queries(queries, args.output)
    print(
        json.dumps(
            {
                "queries": len(queries),
                "positive": sum(bool(query.acceptable_gold_candidate_ids()) for query in queries),
                "null": sum(not query.acceptable_gold_candidate_ids() for query in queries),
                "set_valued": sum(
                    len(query.acceptable_gold_candidate_ids()) > 1 for query in queries
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
