#!/usr/bin/env python3
"""Export one geometric-v9 training checkpoint as the NumPy runtime release."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from omnitransfer.experiment_logging import file_sha256
from omnitransfer.learned_matcher import (
    DIRECT_TEXT_EVIDENCE_ENCODER,
    LEARNED_TOKEN_LOOKUP_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    GeometricMatcher,
    build_geometric_v9_matcher,
    initialize_direct_text_from_lookup,
    parameter_count,
    save_matcher_checkpoint,
)
from omnitransfer.numpy_v9_matcher import save_numpy_geometric_v9_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-output",
        type=Path,
        help="Optional direct-text Torch checkpoint retained for training audit.",
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

    if source.config.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
        target_config = replace(
            source.config,
            text_encoder=DIRECT_TEXT_EVIDENCE_ENCODER,
        )
        target_model = build_geometric_v9_matcher(target_config).eval()
        transferred = initialize_direct_text_from_lookup(target_model, source.model)
    elif source.config.text_encoder == DIRECT_TEXT_EVIDENCE_ENCODER:
        target_config = source.config
        target_model = source.model.eval()
        transferred = tuple(target_model.state_dict())
    else:
        raise SystemExit("unsupported geometric-v9 text encoder")

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
    save_numpy_geometric_v9_checkpoint(
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
