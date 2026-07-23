#!/usr/bin/env python3
"""Create contamination-free relative-XML widget-mapping data."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

clean_widget_mapping_dataset = import_module(
    "omnitransfer.widget_mapping_cleaner"
).clean_widget_mapping_dataset


DEFAULT_INPUT = REPO_ROOT / "runtime/evals/vision_widget_mapping/testset.txt"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime/evals/vision_widget_mapping/clean_relative_xml_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()
    result = clean_widget_mapping_dataset(
        args.input,
        args.output,
        split_seed=args.split_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
