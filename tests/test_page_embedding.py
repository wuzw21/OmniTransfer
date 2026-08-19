from types import SimpleNamespace

import pytest
import torch

from omnitransfer import page_embedding


def _embedder(model: object) -> page_embedding.OmniTransferPageEmbedder:
    embedder = object.__new__(page_embedding.OmniTransferPageEmbedder)
    embedder.checkpoint_path = page_embedding.Path("/tmp/model.pt")
    embedder.checkpoint_sha256 = "checkpoint-sha"
    embedder.config = SimpleNamespace(hidden_dim=2)
    embedder.device = "cpu"
    embedder.model = model
    embedder.numpy_matcher = None
    embedder.backend = "torch"
    embedder.embedding_dim = 2
    embedder.architecture = "unified-model"
    embedder.text_encoder = "learned-token"
    embedder._torch = torch
    return embedder


def test_page_embedding_uses_contextual_page_attention_output(monkeypatch) -> None:
    class Model:
        def __call__(self, *_args):
            return {"source_config_embedding": torch.tensor([3.0, 4.0])}

        def encode_nodes(self, *_args):  # pragma: no cover - forbidden fallback
            raise AssertionError("page embedding must not mean-pool unary node states")

    monkeypatch.setattr(
        page_embedding,
        "graph_from_record",
        lambda *_args, **_kwargs: SimpleNamespace(nodes=[object(), object()]),
    )
    monkeypatch.setattr(
        page_embedding,
        "matcher_inputs",
        lambda *_args, **_kwargs: tuple(object() for _ in range(11)),
    )
    embedder = _embedder(Model())

    result = embedder.embed(
        "<hierarchy><node /></hierarchy>",
        graph_id="page",
        pixels={"width": 720, "height": 1280},
    )

    assert result.vector == pytest.approx((0.6, 0.8))
    assert result.embedding_type == "configuration"
    assert result.node_count == 2
    assert result.graph_id == "page"
    assert result.checkpoint_sha256 == "checkpoint-sha"
    assert result.backend == "torch"


def test_encode_remains_a_vector_compatibility_adapter(monkeypatch) -> None:
    expected = page_embedding.PageEmbedding(
        vector=(0.0, 1.0),
        embedding_type="configuration",
        graph_id="page",
        node_count=1,
        architecture="model",
        text_encoder="tokens",
        backend="numpy",
        checkpoint_path="/tmp/model.npz",
        checkpoint_sha256="sha",
    )
    embedder = object.__new__(page_embedding.OmniTransferPageEmbedder)
    monkeypatch.setattr(embedder, "embed", lambda *_args, **_kwargs: expected)

    assert embedder.encode("<node />", graph_id="page", pixels={}) == [0.0, 1.0]


def test_page_embedding_similarity_is_cosine() -> None:
    first = page_embedding.PageEmbedding(
        vector=(1.0, 0.0),
        embedding_type="configuration",
        graph_id="first",
        node_count=1,
        architecture="model",
        text_encoder="tokens",
        backend="numpy",
        checkpoint_path="/tmp/model.npz",
        checkpoint_sha256="sha",
    )
    same = page_embedding.PageEmbedding(**{**first.__dict__, "graph_id": "same"})
    orthogonal = page_embedding.PageEmbedding(
        **{**first.__dict__, "graph_id": "other", "vector": (0.0, 1.0)}
    )

    assert first.similarity(same) == pytest.approx(1.0)
    assert first.similarity(orthogonal) == pytest.approx(0.0)


def test_default_checkpoint_is_the_frozen_runtime_model() -> None:
    assert page_embedding.DEFAULT_PAGE_EMBEDDING_CHECKPOINT.is_file()
    assert page_embedding.DEFAULT_PAGE_EMBEDDING_CHECKPOINT_SHA256 == (
        "0494224f76c410f17d47b4aaaeacf99e2060c1174da628884c287a6922882ada"
    )
