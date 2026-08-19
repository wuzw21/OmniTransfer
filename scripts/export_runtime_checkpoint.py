#!/usr/bin/env python3
"""Export one unified-association checkpoint as the NumPy runtime release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.learned_matcher import (
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    GeometricMatcher,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.numpy_v9_matcher import save_numpy_unified_association_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-output",
        type=Path,
        help="Optional exact Torch checkpoint retained for runtime audit.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("runtime export requires PyTorch") from exc

    source_path = args.input.expanduser().resolve()
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source = GeometricMatcher.from_checkpoint(source_path, device="cpu")
    if source.config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise SystemExit("input must be a geometric-v9 checkpoint")

    # Runtime uses the exact learned encoder.  Removing the token table would
    # discard semantic evidence now that candidate-pair text shortcuts are gone.
    target_config = source.config
    target_model = source.model.eval()
    transferred = tuple(target_model.state_dict())

    source_parameters = parameter_count(source.model)
    target_parameters = parameter_count(target_model)
    migration = {
        "schema_version": "omnitransfer.runtime_checkpoint_export.v1",
        "source_checkpoint_sha256": file_sha256(source_path),
        "source_matcher_config": asdict(source.config),
        "target_matcher_config": asdict(target_config),
        "source_parameter_count": source_parameters,
        "target_parameter_count": target_parameters,
        "removed_parameter_count": source_parameters - target_parameters,
        "transferred_state_count": len(transferred),
        "training_performed": False,
    }
    if args.torch_output is not None:
        save_matcher_checkpoint(
            args.torch_output,
            target_model,
            config=target_config,
            metadata={
                **dict(source_payload.get("metadata") or {}),
                "runtime_export": migration,
            },
        )
    save_numpy_unified_association_checkpoint(
        args.output,
        target_model.state_dict(),
        config=target_config,
    )
    print(
        json.dumps(
            {
                **migration,
                "input": str(source_path),
                "output": str(args.output.resolve()),
                "output_sha256": file_sha256(args.output),
                "torch_output": (
                    str(args.torch_output.resolve())
                    if args.torch_output is not None
                    else None
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
