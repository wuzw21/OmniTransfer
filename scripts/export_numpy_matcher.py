#!/usr/bin/env python3
"""Export the canonical mutual matcher checkpoint for NumPy inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.numpy_matcher import save_numpy_mutual_matcher_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    payload = torch.load(arguments.source, map_location="cpu")
    if payload.get("schema_version") != "omnitransfer_mutual_matcher_v2":
        raise ValueError("source is not a mutual assignment checkpoint")
    save_numpy_mutual_matcher_checkpoint(
        arguments.output,
        payload["state_dict"],
        config=MatcherConfig(**dict(payload["matcher_config"])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
