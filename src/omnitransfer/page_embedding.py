"""Page-level readout owned by the unified OmniTransfer matcher."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any

from omnitransfer.learned_matcher import (
    ALL_NODE_CANDIDATE_POLICY,
    GeometricMatcher,
    matcher_inputs,
)
from omnitransfer.numpy_v9_matcher import NumpyGeometricAlignmentMatcher
from omnitransfer.ui_graph import UIGraph, graph_from_record


DEFAULT_PAGE_EMBEDDING_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "omnitransfer_unified_association_v1_20260819"
    / "relation_slots_l3_h64_seed17.npz"
)
DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256 = (
    "c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232"
)
PAGE_EMBEDDING_TYPE = "configuration"


@dataclass(frozen=True)
class PageEmbedding:
    """One normalized configuration vector with immutable provenance."""

    vector: tuple[float, ...]
    embedding_type: str
    graph_id: str
    node_count: int
    architecture: str
    text_encoder: str
    backend: str
    checkpoint_path: str
    checkpoint_sha256: str

    def similarity(self, other: "PageEmbedding") -> float:
        if self.embedding_type != other.embedding_type:
            raise ValueError("page embedding types must match")
        if len(self.vector) != len(other.vector):
            raise ValueError("page embedding dimensions must match")
        numerator = sum(left * right for left, right in zip(self.vector, other.vector))
        left_norm = math.sqrt(sum(value * value for value in self.vector))
        right_norm = math.sqrt(sum(value * value for value in other.vector))
        denominator = left_norm * right_norm
        if denominator <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, numerator / denominator))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "omnitransfer.page_embedding.v1",
            "embedding_type": self.embedding_type,
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "dimension": len(self.vector),
            "vector": list(self.vector),
            "architecture": self.architecture,
            "text_encoder": self.text_encoder,
            "backend": self.backend,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


class OmniTransferPageEmbedder:
    """Read the matcher's contextual page-attention token as one page vector.

    A page is paired with itself so the exact same node encoder, typed local
    relations, contextual refinement stack, and learned page-attention head are
    used. There is no second page encoder and no mean-pooling fallback.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        selected = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else DEFAULT_PAGE_EMBEDDING_CHECKPOINT.resolve()
        )
        if not selected.is_file():
            raise FileNotFoundError(f"page embedding checkpoint missing: {selected}")
        self.checkpoint_path = selected
        self.checkpoint_sha256 = _file_sha256(selected)
        if (
            selected == DEFAULT_PAGE_EMBEDDING_CHECKPOINT.resolve()
            and self.checkpoint_sha256 != DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256
        ):
            raise ValueError("default page embedding checkpoint checksum mismatch")
        self.device = device
        self._torch = None
        self.model = None
        self.numpy_matcher = None
        if selected.suffix == ".npz":
            self.numpy_matcher = NumpyGeometricAlignmentMatcher.from_checkpoint(selected)
            self.config = replace(
                self.numpy_matcher.config,
                candidate_policy=ALL_NODE_CANDIDATE_POLICY,
            )
            self.backend = self.numpy_matcher.backend
        elif selected.suffix == ".pt":
            import torch

            matcher = GeometricMatcher.from_checkpoint(selected, device=device)
            self.config = replace(
                matcher.config,
                candidate_policy=ALL_NODE_CANDIDATE_POLICY,
            )
            self.model = matcher.model.to(device).eval()
            self.backend = "torch-unified-association-v1"
            self._torch = torch
        else:
            raise ValueError("page embedding checkpoint must be .npz or .pt")
        self.embedding_dim = self.config.hidden_dim
        self.architecture = self.config.architecture
        self.text_encoder = self.config.text_encoder

    def embed(
        self,
        xml: str,
        *,
        graph_id: str,
        pixels: dict[str, Any] | None = None,
        embedding_type: str = PAGE_EMBEDDING_TYPE,
    ) -> PageEmbedding:
        if embedding_type != PAGE_EMBEDDING_TYPE:
            raise ValueError(
                f"unsupported page embedding type: {embedding_type}; "
                f"available={PAGE_EMBEDDING_TYPE}"
            )
        pixels = dict(pixels or {})
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
        vector = self._embed_graph(graph)
        if len(vector) != self.embedding_dim:
            raise ValueError("page embedding dimension mismatch")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("page embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            raise ValueError("page embedding has zero norm")
        normalized = tuple(float(value / norm) for value in vector)
        return PageEmbedding(
            vector=normalized,
            embedding_type=embedding_type,
            graph_id=graph_id,
            node_count=len(graph.nodes),
            architecture=self.architecture,
            text_encoder=self.text_encoder,
            backend=self.backend,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def encode(
        self,
        xml: str,
        *,
        graph_id: str,
        pixels: dict[str, Any] | None = None,
    ) -> list[float]:
        """Compatibility adapter returning only the configuration vector."""

        return list(self.embed(xml, graph_id=graph_id, pixels=pixels).vector)

    def _embed_graph(self, graph: UIGraph) -> tuple[float, ...]:
        if self.numpy_matcher is not None:
            return tuple(float(value) for value in self.numpy_matcher.page_embedding(graph))
        if self.model is None or self._torch is None:
            raise RuntimeError("page embedding backend is not initialized")
        inputs = matcher_inputs(
            graph,
            graph,
            config=self.config,
            device=self.device,
        )
        with self._torch.inference_mode():
            vector = self.model(*inputs)["source_config_embedding"]
        return tuple(float(value) for value in vector.detach().cpu().tolist())


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


__all__ = [
    "DEFAULT_PAGE_EMBEDDING_CHECKPOINT",
    "DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256",
    "OmniTransferPageEmbedder",
    "PAGE_EMBEDDING_TYPE",
    "PageEmbedding",
]
