"""Feature extraction outline for UI grounding relocation."""

from __future__ import annotations

from omnitransfer.schema import Candidate, Query


def pair_features(query: Query, candidate: Candidate) -> dict[str, float]:
    """Build structured source-target pair features.

    Planned feature groups:
    - text/resource/class overlap
    - action compatibility flags
    - bbox and screen-region geometry
    - local graph/context relations
    - history-conditioned bias features
    """

    _ = (query, candidate)
    raise NotImplementedError("Feature extraction will be migrated next.")
