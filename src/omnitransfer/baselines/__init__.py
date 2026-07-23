"""Offline-only comparison baselines for OmniTransfer."""

from omnitransfer.baselines.anchor_vote_64d import (
    AnchorVote64DMatcher,
    AnchorVoteResult,
    PublicWidgetPair,
    bind_public_bbox,
    cosine_similarity,
    encode_graph_64d,
    load_public_widget_pairs,
    normalize_public_bbox,
)

__all__ = [
    "AnchorVote64DMatcher",
    "AnchorVoteResult",
    "PublicWidgetPair",
    "bind_public_bbox",
    "cosine_similarity",
    "encode_graph_64d",
    "load_public_widget_pairs",
    "normalize_public_bbox",
]
