"""The single trainable OmniTransfer contextual graph matcher."""

from __future__ import annotations

import math
from typing import Any

from omnitransfer.unified_alignment import (
    UNIFIED_ASSOCIATION_SCHEMA_ID,
    default_alignment_strategy_registry,
)


def build_geometric_matcher(config: Any) -> Any:
    """Build the unified node-encoding and iterative graph-matching model."""

    from omnitransfer.learned_matcher import (
        ALIGNMENT_RELATION_FEATURE_INDICES,
        ALL_NODE_CANDIDATE_POLICY,
        DIRECT_PAIR_EVIDENCE_NAMES,
        DIRECT_TEXT_EVIDENCE_ENCODER,
        DETERMINISTIC_ICON_VISUAL_ENCODER,
        LEARNED_TOKEN_LOOKUP_ENCODER,
        LEGACY_GLOBAL_POOL_VISUAL_ENCODER,
        MULTISCALE_HASH_VISUAL_ENCODER,
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
        SPATIAL_CNN_VISUAL_ENCODER,
        TEXT_DESCRIPTOR_DIM,
        TYPED_RELATION_NAMES,
        VISUAL_DESCRIPTOR_DIM,
        XML_DESCRIPTOR_DIM,
        XML_NODE_FEATURE_DIM,
        mutual_log_assignment,
        typed_relation_bases,
        visual_descriptor_dim,
        _require_torch,
    )
    from omnitransfer.visual_descriptor import (
        deterministic_icon_descriptor_torch,
        multiscale_hash_descriptor_torch,
    )

    cfg = config
    if cfg.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("only the geometric-v9 matcher is supported")
    if cfg.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise ValueError("geometric-v9 requires the all-node candidate policy")
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if cfg.source_context_nodes <= 0 or cfg.target_context_nodes <= 0:
        raise ValueError("context node limits must be positive")
    if (
        cfg.visual_canvas_size != 0
        and cfg.visual_canvas_size < cfg.visual_patch_size
    ):
        raise ValueError("visual_canvas_size must cover one visual patch")
    if cfg.association_layers <= 0:
        raise ValueError("association_layers must be positive")
    if cfg.assignment_head != "partial_assignment":
        raise ValueError("geometric-v9 requires partial assignment")

    torch = _require_torch()
    nn = torch.nn
    relation_count = len(TYPED_RELATION_NAMES)
    visual_dim = visual_descriptor_dim(cfg.visual_encoder)

    class LocalGraphLayer(nn.Module):
        """Aggregate one node's typed, nearby UI relatives into its state."""

        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Linear(
                cfg.hidden_dim, cfg.relation_hidden_dim, bias=False
            )
            self.message_projection = nn.Linear(
                relation_count * cfg.relation_hidden_dim,
                cfg.hidden_dim,
                bias=False,
            )
            self.message_norm = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            states: Any,
            relation_bases: Any,
            relation_compatibility: Any,
        ) -> Any:
            # Keep relation types sharp.  A unit diagonal starts close to an
            # identity routing matrix while remaining fully learnable.
            compatibility = torch.softmax(
                5.0 * relation_compatibility, dim=-1
            )
            mixed_bases = torch.einsum(
                "rq,qij->rij", compatibility, relation_bases
            )
            relation_messages = torch.einsum(
                "rij,jd->rid", mixed_bases, self.value(states)
            )
            messages = relation_messages.permute(1, 0, 2).reshape(
                states.shape[0], relation_count * cfg.relation_hidden_dim
            )
            states = self.message_norm(
                states + self.dropout(self.message_projection(messages))
            )
            return self.output_norm(
                states + self.dropout(self.feed_forward(states))
            )

    class ContextualRefinementLayer(nn.Module):
        """One shared local-graph and bidirectional cross-page update."""

        def __init__(self) -> None:
            super().__init__()
            self.local = LocalGraphLayer()
            self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.key = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.value = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.cross_update = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim * 4),
                nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def _cross_context(self, query_states: Any, key_states: Any) -> Any:
            query = self.query(query_states)
            key = self.key(key_states)
            scores = (query @ key.T) / math.sqrt(float(cfg.hidden_dim))
            return torch.softmax(scores, dim=-1) @ self.value(key_states)

        def _update(self, states: Any, context: Any) -> Any:
            evidence = torch.cat(
                (states, context, torch.abs(states - context), states * context),
                dim=-1,
            )
            return self.cross_norm(
                states + self.dropout(self.cross_update(evidence))
            )

        def forward(
            self,
            source_states: Any,
            target_states: Any,
            source_bases: Any,
            target_bases: Any,
            relation_compatibility: Any,
        ) -> tuple[Any, Any]:
            source_local = self.local(
                source_states, source_bases, relation_compatibility
            )
            target_local = self.local(
                target_states, target_bases, relation_compatibility
            )
            source_context = self._cross_context(source_local, target_local)
            target_context = self._cross_context(target_local, source_local)
            return (
                self._update(source_local, source_context),
                self._update(target_local, target_context),
            )

    class GeometricAlignmentMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                self.token_embedding = nn.Embedding(
                    cfg.vocab_size, cfg.token_dim, padding_idx=0
                )
                self.text_projection = nn.Linear(
                    cfg.token_dim, TEXT_DESCRIPTOR_DIM
                )
            elif cfg.text_encoder == DIRECT_TEXT_EVIDENCE_ENCODER:
                self.present_text = nn.Parameter(torch.zeros(TEXT_DESCRIPTOR_DIM))
            else:
                raise ValueError(f"unsupported text encoder: {cfg.text_encoder}")

            self.xml_projection = nn.Sequential(
                nn.LayerNorm(XML_NODE_FEATURE_DIM),
                nn.Linear(XML_NODE_FEATURE_DIM, XML_DESCRIPTOR_DIM),
                nn.GELU(),
                nn.Linear(XML_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
            )
            if cfg.visual_encoder == LEGACY_GLOBAL_POOL_VISUAL_ENCODER:
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(
                        32,
                        VISUAL_DESCRIPTOR_DIM,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
            elif cfg.visual_encoder == SPATIAL_CNN_VISUAL_ENCODER:
                if cfg.visual_patch_size != 32:
                    raise ValueError("spatial_cnn_v2 requires 32x32 visual patches")
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                    nn.GELU(),
                    nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Flatten(),
                    nn.Linear(48 * 8 * 8, VISUAL_DESCRIPTOR_DIM),
                )
            elif cfg.visual_encoder == DETERMINISTIC_ICON_VISUAL_ENCODER:

                class DeterministicIconEncoder(nn.Module):
                    def forward(self, patches: Any) -> Any:
                        return deterministic_icon_descriptor_torch(
                            patches, torch=torch
                        )

                self.visual_encoder = DeterministicIconEncoder()
            elif cfg.visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER:

                class MultiscaleHashEncoder(nn.Module):
                    def forward(self, patches: Any) -> Any:
                        return multiscale_hash_descriptor_torch(
                            patches, torch=torch
                        )

                self.visual_encoder = MultiscaleHashEncoder()
            else:
                raise ValueError(f"unsupported visual encoder: {cfg.visual_encoder}")

            self.missing_text = nn.Parameter(torch.zeros(TEXT_DESCRIPTOR_DIM))
            self.missing_visual = nn.Parameter(torch.zeros(visual_dim))
            self.text_to_hidden = nn.Linear(TEXT_DESCRIPTOR_DIM, cfg.hidden_dim)
            self.visual_to_hidden = nn.Linear(visual_dim, cfg.hidden_dim)
            self.xml_to_hidden = nn.Linear(XML_DESCRIPTOR_DIM, cfg.hidden_dim)
            self.modality_type = nn.Parameter(torch.empty(3, cfg.hidden_dim))
            nn.init.normal_(self.modality_type, std=0.02)
            self.modality_score = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1, bias=False),
            )
            self.input_norm = nn.LayerNorm(cfg.hidden_dim)
            self.input_feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.input_output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.relation_compatibility = nn.Parameter(torch.eye(relation_count))
            self.association_layers = nn.ModuleList(
                ContextualRefinementLayer()
                for _ in range(cfg.association_layers)
            )
            self.matchability_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
            self.page_attention = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1, bias=False),
            )
            feature_dim = (
                len(DIRECT_PAIR_EVIDENCE_NAMES)
                + 16
                + relation_count**2
                + len(ALIGNMENT_RELATION_FEATURE_INDICES)
            )
            self.alignment_strategy = default_alignment_strategy_registry(
                direct_pair_dimension=len(DIRECT_PAIR_EVIDENCE_NAMES),
                typed_relation_dimension=relation_count**2,
                geometry_dimension=len(ALIGNMENT_RELATION_FEATURE_INDICES),
            )
            self.alignment_strategy.validate_dimension(feature_dim)

        def _affinity(
            self, source_states: Any, target_states: Any
        ) -> tuple[Any, Any, Any]:
            normalized_source = torch.nn.functional.normalize(
                source_states, dim=-1
            )
            normalized_target = torch.nn.functional.normalize(
                target_states, dim=-1
            )
            source_matchability = self.matchability_head(source_states).squeeze(-1)
            target_matchability = self.matchability_head(target_states).squeeze(-1)
            affinity = (
                self.logit_scale.exp().clamp(max=100.0)
                * (normalized_source @ normalized_target.T)
                + 0.5
                * (
                    source_matchability[:, None]
                    + target_matchability[None, :]
                )
            )
            return affinity, source_matchability, target_matchability

        def _page_embedding(self, states: Any) -> Any:
            weights = torch.softmax(
                self.page_attention(states).squeeze(-1), dim=0
            )
            return torch.nn.functional.normalize(
                torch.sum(weights[:, None] * states, dim=0), dim=0
            )

        def encode_nodes(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> tuple[Any, Any, Any]:
            pooled_tokens = None
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                mask = token_ids.ne(0).unsqueeze(-1)
                embedded = self.token_embedding(token_ids)
                pooled_tokens = (embedded * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1)

            visual_states = self.visual_encoder(visual_patches)
            visual_states = visual_mask * visual_states + (1.0 - visual_mask) * (
                self.missing_visual.unsqueeze(0).expand(
                    visual_states.shape[0], -1
                )
            )
            text_mask = token_ids.ne(0).any(dim=1, keepdim=True).to(
                dtype=visual_states.dtype
            )
            if pooled_tokens is not None:
                text_states = self.text_projection(pooled_tokens)
            else:
                text_states = self.present_text.unsqueeze(0).expand(
                    token_ids.shape[0], -1
                )
            text_states = text_mask * text_states + (1.0 - text_mask) * (
                self.missing_text.unsqueeze(0).expand(text_states.shape[0], -1)
            )
            xml_states = self.xml_projection(numeric_features)
            visual_hidden = self.visual_to_hidden(visual_states)
            modalities = torch.stack(
                (
                    self.text_to_hidden(text_states),
                    visual_hidden,
                    self.xml_to_hidden(xml_states),
                ),
                dim=1,
            ) + self.modality_type.unsqueeze(0)
            available = torch.cat(
                (
                    text_mask,
                    visual_mask.to(dtype=text_mask.dtype),
                    torch.ones_like(text_mask),
                ),
                dim=1,
            )
            modality_logits = self.modality_score(modalities).squeeze(-1)
            modality_logits = modality_logits.masked_fill(available.eq(0.0), -1e4)
            modality_weights = torch.softmax(modality_logits / 0.25, dim=1)
            fused = torch.sum(modality_weights.unsqueeze(-1) * modalities, dim=1)
            states = self.input_norm(fused)
            states = self.input_output_norm(
                states + self.input_feed_forward(states)
            )
            return states, states, visual_hidden

        def forward(
            self,
            source_token_ids: Any,
            source_numeric: Any,
            source_relations: Any,
            target_token_ids: Any,
            target_numeric: Any,
            target_relations: Any,
            direct_pair_evidence: Any,
            source_visual: Any,
            source_visual_mask: Any,
            target_visual: Any,
            target_visual_mask: Any,
            detach_unary_for_relation: bool = False,
        ) -> dict[str, Any]:
            del detach_unary_for_relation
            (
                source_states,
                source_descriptors,
                source_visual_descriptors,
            ) = self.encode_nodes(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            (
                target_states,
                target_descriptors,
                target_visual_descriptors,
            ) = self.encode_nodes(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            source_bases = typed_relation_bases(source_relations, source_numeric)
            target_bases = typed_relation_bases(target_relations, target_numeric)
            unary_affinity, _, _ = self._affinity(source_states, target_states)

            assignments = []
            affinities = []
            source_states_by_layer = []
            target_states_by_layer = []
            source_matchability_by_layer = []
            target_matchability_by_layer = []
            for layer in self.association_layers:
                source_states, target_states = layer(
                    source_states,
                    target_states,
                    source_bases,
                    target_bases,
                    self.relation_compatibility,
                )
                affinity, source_matchability, target_matchability = self._affinity(
                    source_states, target_states
                )
                affinities.append(affinity)
                assignments.append(mutual_log_assignment(affinity))
                source_states_by_layer.append(source_states)
                target_states_by_layer.append(target_states)
                source_matchability_by_layer.append(source_matchability)
                target_matchability_by_layer.append(target_matchability)

            final_affinity = affinities[-1]
            final_assignment = assignments[-1]
            raw_weights = tuple(range(1, len(assignments) + 1))
            weight_sum = float(sum(raw_weights))
            assignment_weights = tuple(value / weight_sum for value in raw_weights)
            return {
                "logits_ab": final_assignment,
                "logits_ba": final_assignment.T,
                "affinity": final_affinity,
                "association_score": final_affinity,
                "direct_pair_evidence": direct_pair_evidence,
                "unary_affinity": unary_affinity,
                "alignment_strategy_schema": self.alignment_strategy.schema_id,
                "alignment_strategy": self.alignment_strategy.metadata(),
                "association_schema": UNIFIED_ASSOCIATION_SCHEMA_ID,
                "assignment_scores_by_layer": tuple(assignments),
                "assignment_loss_weights": assignment_weights,
                "affinities_by_layer": tuple(affinities),
                "source_states_by_layer": tuple(source_states_by_layer),
                "target_states_by_layer": tuple(target_states_by_layer),
                "source_matchability_by_layer": tuple(
                    source_matchability_by_layer
                ),
                "target_matchability_by_layer": tuple(
                    target_matchability_by_layer
                ),
                "source_relation_bases": source_bases,
                "target_relation_bases": target_bases,
                "source_descriptors": source_descriptors,
                "target_descriptors": target_descriptors,
                "source_visual_descriptors": source_visual_descriptors,
                "target_visual_descriptors": target_visual_descriptors,
                "source_visual_mask": source_visual_mask,
                "target_visual_mask": target_visual_mask,
                "source_states": source_states,
                "target_states": target_states,
                "source_config_embedding": self._page_embedding(source_states),
                "target_config_embedding": self._page_embedding(target_states),
            }

    return GeometricAlignmentMatcher()
