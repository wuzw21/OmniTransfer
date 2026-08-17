"""Canonical geometric-v9 page embedding."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    GeometricMatcher,
    matcher_inputs,
)
from omnitransfer.ui_graph import graph_from_record


class OmniTransferPageEmbedder:
    """Mean-pool frozen geometric-v9 node representations."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        import torch

        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        self.checkpoint_sha256 = _file_sha256(self.checkpoint_path)
        self.matcher = GeometricMatcher.from_checkpoint(
            self.checkpoint_path,
            device=device,
        )
        self.config = replace(
            self.matcher.config,
            candidate_policy=ALL_NODE_CANDIDATE_POLICY,
        )
        self.model = self.matcher.model.to(device).eval()
        self.device = device
        self.embedding_dim = self.config.hidden_dim
        self.architecture = self.matcher.config.architecture
        self.text_encoder = self.matcher.config.text_encoder
        self._torch = torch

    def encode(
        self,
        xml: str,
        *,
        graph_id: str,
        pixels: dict[str, Any],
    ) -> list[float]:
        screenshot = _available_screenshot(pixels)
        record: dict[str, Any] = {
            "xml": xml,
            "width": pixels.get("width"),
            "height": pixels.get("height"),
        }
        if screenshot is not None:
            record["screenshot_path"] = str(screenshot)
        graph = graph_from_record(record, graph_id=graph_id)
        if not graph.nodes:
            raise ValueError(f"page {graph_id} has no XML nodes")
        inputs = matcher_inputs(
            graph,
            graph,
            config=self.config,
            device=self.device,
        )
        with self._torch.inference_mode():
            states, _, _ = self.model.encode_nodes(
                inputs[0],
                inputs[1],
                inputs[7],
                inputs[8],
            )
            vector = self._torch.nn.functional.normalize(
                states.mean(dim=0),
                dim=0,
            )
        return vector.detach().cpu().tolist()


def _available_screenshot(pixels: dict[str, Any]) -> Path | None:
    value = pixels.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["OmniTransferPageEmbedder"]
