from types import SimpleNamespace

import pytest
import torch

from omnitransfer import page_embedding


def _embedder(model: object) -> page_embedding.OmniTransferPageEmbedder:
    embedder = object.__new__(page_embedding.OmniTransferPageEmbedder)
    embedder.checkpoint_path = page_embedding.Path("/tmp/model.pt")
    embedder.checkpoint_sha256 = "checkpoint-sha"
    embedder.config = SimpleNamespace(hidden_dim=2, state_embedding_dim=0)
    embedder.device = "cpu"
    embedder.model = model
    embedder.numpy_matcher = None
    embedder.backend = "torch"
    embedder.embedding_dims = {
        "configuration": 2,
        "stable_configuration": 2,
        "active_state": 2,
    }
    embedder.embedding_dim = 2
    embedder.architecture = "unified-model"
    embedder.text_encoder = "learned-token"
    embedder._torch = torch
    return embedder


def test_page_embedding_uses_contextual_page_attention_output(monkeypatch) -> None:
    class Model:
        def encode_page(self, *_args):
            return {"page_embedding": torch.tensor([3.0, 4.0])}

        def __call__(self, *_args):  # pragma: no cover - forbidden duplicate work
            raise AssertionError("page embedding must encode one page only once")

        def encode_nodes(self, *_args):  # pragma: no cover - forbidden fallback
            raise AssertionError("page embedding must not mean-pool unary node states")

    monkeypatch.setattr(
        page_embedding,
        "graph_from_record",
        lambda *_args, **_kwargs: SimpleNamespace(nodes=[object(), object()]),
    )
    calls = []

    def single_page_inputs(*_args, **_kwargs):
        calls.append("page")
        return tuple(object() for _ in range(5))

    monkeypatch.setattr(page_embedding, "page_inputs", single_page_inputs)
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
    assert calls == ["page"]


@pytest.mark.parametrize(
    ("embedding_type", "expected"),
    (
        ("stable_configuration", (1.0, 0.0)),
        ("active_state", (0.0, 1.0)),
    ),
)
def test_page_embedding_selects_the_requested_shared_readout(
    monkeypatch,
    embedding_type,
    expected,
) -> None:
    class Model:
        def encode_page(self, *_args):
            return {
                "page_embedding": torch.tensor([1.0, 1.0]),
                "stable_state_embedding": torch.tensor([1.0, 0.0]),
                "active_state_embedding": torch.tensor([0.0, 1.0]),
            }

    monkeypatch.setattr(
        page_embedding,
        "graph_from_record",
        lambda *_args, **_kwargs: SimpleNamespace(nodes=[object()]),
    )
    monkeypatch.setattr(
        page_embedding,
        "page_inputs",
        lambda *_args, **_kwargs: tuple(object() for _ in range(5)),
    )

    result = _embedder(Model()).embed(
        "<hierarchy><node /></hierarchy>",
        graph_id="page",
        embedding_type=embedding_type,
    )

    assert result.vector == pytest.approx(expected)
    assert result.embedding_type == embedding_type


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
        "c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232"
    )
