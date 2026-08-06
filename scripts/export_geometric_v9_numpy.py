#!/usr/bin/env python3
"""Export the frozen OmniTransfer geometric v9 checkpoint for Android NumPy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from omnitransfer.learned_matcher import MatcherConfig
from omnitransfer.numpy_v9_matcher import save_numpy_geometric_v9_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required only while exporting v9") from error
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = MatcherConfig(**dict(payload["matcher_config"]))
    save_numpy_geometric_v9_checkpoint(
        output,
        payload["state_dict"],
        config=config,
    )
    print("OMNITRANSFER_V9_NUMPY_EXPORT=PASS")
    print(f"input={source}")
    print(f"input_sha256={sha256_file(source)}")
    print(f"output={output}")
    print(f"output_sha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
