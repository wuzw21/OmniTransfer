"""Matcher outline."""

from __future__ import annotations

from typing import Protocol

from omnitransfer.schema import Prediction, Query


class Matcher(Protocol):
    """A source-grounding to target-grounding ranker."""

    def predict(self, query: Query) -> Prediction:
        """Rank candidates for one query."""


class StructuredRanker:
    """Placeholder for the deployable structured OmniTransfer ranker."""

    def predict(self, query: Query) -> Prediction:
        """Return the highest-scoring target candidate.

        The real implementation will be migrated from the existing rich-eval
        benchmark after the standalone outline is frozen.
        """

        _ = query
        raise NotImplementedError("StructuredRanker implementation pending.")
