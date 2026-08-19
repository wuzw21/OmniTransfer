"""Unified learnable-alignment and transfer contract.

This is the single cross-module seam for OmniTransfer.  It describes the
evidence groups consumed by the trainable matcher and the data contract at
the public transfer boundary.  It does not contain a second mapper, a fixed
score, or a post-hoc rescue rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALIGNMENT_STRATEGY_SCHEMA_ID = "omnitransfer.alignment-strategy.v2"
UNIFIED_ASSOCIATION_SCHEMA_ID = "omnitransfer.unified-association.v1"
TRANSFER_CONTRACT_SCHEMA_ID = "omnitransfer.transfer-contract.v2"
TRANSFER_PIPELINE_MODULES = (
    "multimodal_ui_encoder",
    "relation_aware_association",
    "bidirectional_candidate_decoder",
    "relative_action_projector",
)


@dataclass(frozen=True)
class StrategyFeatureGroup:
    """One learnable evidence group exposed to the strategy aggregator."""

    name: str
    dimension: int
    source: str
    description: str

    def __post_init__(self) -> None:
        if not self.name or not self.source or self.dimension <= 0:
            raise ValueError(
                "strategy feature groups require a name, source, and dimension"
            )


@dataclass(frozen=True)
class AlignmentStrategyRegistry:
    """Immutable registry for the model's learnable evidence groups."""

    schema_id: str
    groups: tuple[StrategyFeatureGroup, ...]

    @property
    def dimension(self) -> int:
        return sum(group.dimension for group in self.groups)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "dimension": self.dimension,
            "groups": [
                {
                    "name": group.name,
                    "dimension": group.dimension,
                    "source": group.source,
                    "description": group.description,
                }
                for group in self.groups
            ],
        }

    def validate_dimension(self, dimension: int) -> None:
        if int(dimension) != self.dimension:
            raise ValueError(
                "alignment strategy feature dimension mismatch: "
                f"expected {self.dimension}, got {dimension}"
            )


def default_alignment_strategy_registry(
    *,
    direct_pair_dimension: int,
    typed_relation_dimension: int,
    geometry_dimension: int,
) -> AlignmentStrategyRegistry:
    """Describe the two learnable inputs to the unified association model.

    The registry assigns no weights and does not create parallel score heads.
    The model fuses node/pair evidence once, then propagates it through one
    multi-hop local relation graph.
    """

    if min(direct_pair_dimension, typed_relation_dimension, geometry_dimension) <= 0:
        raise ValueError("strategy dimensions must be positive")
    groups = (
        StrategyFeatureGroup(
            "node_pair_encoding",
            direct_pair_dimension + 16,
            "multimodal_node_encoder_and_cross_page_pair_features",
            "encoded text, visual, XML, affordance, and pair evidence",
        ),
        StrategyFeatureGroup(
            "multi_hop_local_relation_graph",
            typed_relation_dimension + geometry_dimension,
            "within_page_hierarchy_and_relative_geometry",
            "parent, child, sibling, ancestor, kinship distance, and local geometry",
        ),
    )
    return AlignmentStrategyRegistry(
        schema_id=ALIGNMENT_STRATEGY_SCHEMA_ID,
        groups=groups,
    )


def strategy_metadata(registry: AlignmentStrategyRegistry) -> dict[str, Any]:
    """Return checkpoint/review-safe metadata for one strategy registry."""

    return registry.metadata()


def relation_consistency_loss(
    source_bases: Any,
    target_bases: Any,
    relation_compatibility: Any,
    source_positive_targets: tuple[tuple[int, ...], ...],
    target_positive_sources: tuple[tuple[int, ...], ...],
    *,
    torch: Any,
) -> Any:
    """Learn relation compatibility from correspondence labels."""

    if source_bases.ndim != 3 or target_bases.ndim != 3:
        raise ValueError("relation bases must be three-dimensional")
    relation_count = int(source_bases.shape[0])
    if target_bases.shape[0] != relation_count:
        raise ValueError("source and target relation counts must agree")
    if relation_compatibility.shape != (relation_count, relation_count):
        raise ValueError("relation compatibility has an unexpected shape")

    compatibility = torch.softmax(relation_compatibility, dim=-1)
    losses: list[Any] = []

    def collect(
        source_labels: tuple[tuple[int, ...], ...],
        target_labels: tuple[tuple[int, ...], ...],
    ) -> None:
        for source_index, target_indices in enumerate(source_labels):
            if not target_indices:
                continue
            for other_source, other_targets in enumerate(source_labels):
                if source_index == other_source or not other_targets:
                    continue
                source_relation = source_bases[:, source_index, other_source]
                for target_index in target_indices:
                    for other_target in other_targets:
                        target_relation = target_bases[:, target_index, other_target]
                        predicted = source_relation @ compatibility
                        losses.append(
                            torch.nn.functional.smooth_l1_loss(
                                predicted,
                                target_relation,
                            )
                        )

    collect(source_positive_targets, target_positive_sources)
    if not losses:
        return source_bases.new_zeros(())
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class TransferRequest:
    """One source/target transfer query at the public seam."""

    target_xml: str
    source_xml: str | None = None
    source_point: tuple[float, float] | None = None
    source_element_id: str | None = None
    source_offset: tuple[float, float] | None = None
    source_screenshot_path: str | None = None
    target_screenshot_path: str | None = None
    source_visual_rgb: dict[str, Any] | None = None
    target_visual_rgb: dict[str, Any] | None = None
    action_type: str = "click"
    top_k: int = 1

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema_id": TRANSFER_CONTRACT_SCHEMA_ID,
            "pipeline_modules": list(TRANSFER_PIPELINE_MODULES),
            "action_type": self.action_type,
            "top_k": int(self.top_k),
        }


def transfer_contract_metadata() -> dict[str, Any]:
    """Return stable metadata for runtime, review, and experiment records."""

    return {
        "schema_id": TRANSFER_CONTRACT_SCHEMA_ID,
        "pipeline_modules": list(TRANSFER_PIPELINE_MODULES),
        "dataflow": (
            "source_xml+screenshot -> UIGraph -> candidate_association_graph "
            "-> bidirectional_ranking -> relative_action_projection"
        ),
        "coordinate_policy": "relative_within_source_node_only",
        "failure_policy": "explicit_transfer_failure_then_caller_fallback",
    }
