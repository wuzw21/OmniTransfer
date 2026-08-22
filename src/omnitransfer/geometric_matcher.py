"""The single trainable OmniTransfer page-local correspondence model.

The model has one path: reuse the geometric-v9 multimodal node encoder, learn
and fuse a bounded local neighbourhood, then repeatedly refine one real-node
score matrix from the current soft correspondence.  The same encoded nodes
also produce the 1024D state readout.  There is no NULL class, gate, score
bonus, reranker, or auxiliary inference path.
"""

from __future__ import annotations

import math
from typing import Any

from omnitransfer.unified_alignment import UNIFIED_ASSOCIATION_SCHEMA_ID


def build_geometric_matcher(config: Any) -> Any:
    """Build the unified supervised page-local matcher."""

    from omnitransfer.learned_matcher import (
        ALL_NODE_CANDIDATE_POLICY,
        DETERMINISTIC_ICON_VISUAL_ENCODER,
        DIRECT_TEXT_EVIDENCE_ENCODER,
        FIXED_NEIGHBOR_SELECTION,
        HASHED_NGRAM_TEXT_ENCODER,
        LEARNED_NEIGHBOR_SELECTION,
        LEARNED_TOKEN_LOOKUP_ENCODER,
        LEGACY_GLOBAL_POOL_VISUAL_ENCODER,
        MULTISCALE_HASH_VISUAL_ENCODER,
        MULTISCALE_RESIDUAL_VISUAL_ENCODER,
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
        RELATION_FEATURE_DIM,
        SPATIAL_CNN_VISUAL_ENCODER,
        TEXT_DESCRIPTOR_DIM,
        TYPED_RELATION_NAMES,
        UNARY_RESIDUAL_SCORE_UPDATE,
        VISUAL_DESCRIPTOR_DIM,
        XML_DESCRIPTOR_DIM,
        _require_torch,
        hashed_ngram_descriptor_torch,
        mutual_log_assignment,
        typed_relation_bases,
        visual_descriptor_dim,
        xml_node_feature_dim,
    )
    from omnitransfer.visual_descriptor import (
        deterministic_icon_descriptor_torch,
        multiscale_hash_descriptor_torch,
    )

    cfg = config
    if cfg.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("only the canonical OmniTransfer matcher is supported")
    if cfg.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise ValueError("the matcher requires the all-node candidate policy")
    if cfg.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if cfg.relation_hidden_dim <= 0:
        raise ValueError("relation_hidden_dim must be positive")
    if cfg.association_layers <= 0:
        raise ValueError("page-local matching requires correspondence stages")
    if cfg.source_context_nodes <= 0 or cfg.target_context_nodes <= 0:
        raise ValueError("context node limits must be positive")
    if cfg.local_neighbor_limit <= 0:
        raise ValueError("local_neighbor_limit must be positive")
    if cfg.router_temperature <= 0.0:
        raise ValueError("router_temperature must be positive")
    if cfg.state_embedding_dim <= 0:
        raise ValueError("state_embedding_dim must be positive")
    if cfg.state_embedding_dim % cfg.hidden_dim:
        raise ValueError("state_embedding_dim must be divisible by hidden_dim")
    if cfg.neighbor_selection not in {
        FIXED_NEIGHBOR_SELECTION,
        LEARNED_NEIGHBOR_SELECTION,
    }:
        raise ValueError(f"unsupported neighbor selection: {cfg.neighbor_selection}")

    torch = _require_torch()
    nn = torch.nn
    visual_dim = visual_descriptor_dim(cfg.visual_encoder)
    numeric_dim = xml_node_feature_dim(cfg.feature_schema_id)
    max_neighbours = cfg.local_neighbor_limit
    state_slots = cfg.state_embedding_dim // cfg.hidden_dim
    relation_count = len(TYPED_RELATION_NAMES)

    class DeterministicIconEncoder(nn.Module):
        def forward(self, patches: Any) -> Any:
            return deterministic_icon_descriptor_torch(patches, torch=torch)

    class MultiscaleHashEncoder(nn.Module):
        def forward(self, patches: Any) -> Any:
            return multiscale_hash_descriptor_torch(patches, torch=torch)

    def make_visual_encoder() -> Any:
        if cfg.visual_encoder == LEGACY_GLOBAL_POOL_VISUAL_ENCODER:
            return nn.Sequential(
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
        if cfg.visual_encoder == SPATIAL_CNN_VISUAL_ENCODER:
            if cfg.visual_patch_size != 32:
                raise ValueError("spatial visual encoding requires 32x32 crops")
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(48 * 8 * 8, VISUAL_DESCRIPTOR_DIM),
            )
        if cfg.visual_encoder == DETERMINISTIC_ICON_VISUAL_ENCODER:
            return DeterministicIconEncoder()
        if cfg.visual_encoder in {
            MULTISCALE_HASH_VISUAL_ENCODER,
            MULTISCALE_RESIDUAL_VISUAL_ENCODER,
        }:
            return MultiscaleHashEncoder()
        raise ValueError(f"unsupported visual encoder: {cfg.visual_encoder}")

    class LocalGraphLayer(nn.Module):
        """The trained v9 within-page structure update."""

        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Linear(cfg.hidden_dim, cfg.relation_hidden_dim, bias=False)
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
            compatibility = torch.softmax(5.0 * relation_compatibility, dim=-1)
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
        """The trained v9 local and bidirectional cross-page update."""

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
            return (
                self._update(
                    source_local,
                    self._cross_context(source_local, target_local),
                ),
                self._update(
                    target_local,
                    self._cross_context(target_local, source_local),
                ),
            )

    class PageLocalMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                self.token_embedding = nn.Embedding(
                    cfg.vocab_size,
                    cfg.token_dim,
                    padding_idx=0,
                )
                self.text_projection = nn.Linear(
                    cfg.token_dim,
                    TEXT_DESCRIPTOR_DIM,
                )
            elif cfg.text_encoder == HASHED_NGRAM_TEXT_ENCODER:
                pass
            elif cfg.text_encoder == DIRECT_TEXT_EVIDENCE_ENCODER:
                self.present_text = nn.Parameter(torch.zeros(TEXT_DESCRIPTOR_DIM))
            else:
                raise ValueError(f"unsupported text encoder: {cfg.text_encoder}")

            self.visual_encoder = make_visual_encoder()
            self.missing_text = nn.Parameter(torch.zeros(TEXT_DESCRIPTOR_DIM))
            self.missing_visual = nn.Parameter(torch.zeros(visual_dim))
            self.xml_projection = nn.Sequential(
                nn.LayerNorm(numeric_dim),
                nn.Linear(numeric_dim, XML_DESCRIPTOR_DIM),
                nn.GELU(),
                nn.Linear(XML_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
            )
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
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(5.0), dtype=torch.float32)
            )
            self.relation_compatibility = nn.Parameter(torch.eye(relation_count))
            self.association_layers = nn.ModuleList(
                ContextualRefinementLayer()
                for _ in range(cfg.association_layers)
            )
            self.matchability_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1),
            )

            relation_input_dim = cfg.hidden_dim + RELATION_FEATURE_DIM
            self.relation_encoder = nn.Sequential(
                nn.LayerNorm(relation_input_dim),
                nn.Linear(relation_input_dim, cfg.relation_hidden_dim * 2),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim * 2, cfg.relation_hidden_dim),
                nn.LayerNorm(cfg.relation_hidden_dim),
            )
            if cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION:
                self.neighbor_query = nn.Linear(
                    cfg.hidden_dim,
                    cfg.relation_hidden_dim,
                    bias=False,
                )
                self.neighbor_key = nn.Linear(
                    cfg.hidden_dim,
                    cfg.relation_hidden_dim,
                    bias=False,
                )
                self.neighbor_relation_score = nn.Linear(
                    RELATION_FEATURE_DIM,
                    1,
                    bias=False,
                )
            pair_dim = cfg.hidden_dim * 4
            self.relation_query = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, cfg.relation_hidden_dim),
                nn.Tanh(),
            )
            self.relation_key = nn.Linear(
                cfg.relation_hidden_dim,
                cfg.relation_hidden_dim,
                bias=False,
            )
            self.local_pair = nn.Sequential(
                nn.LayerNorm(cfg.relation_hidden_dim * 4 + 2),
                nn.Linear(
                    cfg.relation_hidden_dim * 4 + 2,
                    cfg.relation_hidden_dim * 2,
                ),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim * 2, cfg.relation_hidden_dim),
            )
            self.correspondence_evidence = nn.Linear(1, 1, bias=False)

            self.page_attention = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1, bias=False),
            )
            self.state_attention = nn.Linear(
                cfg.hidden_dim,
                state_slots,
                bias=False,
            )
            self.pair_scorer = nn.Sequential(
                nn.LayerNorm(pair_dim + cfg.relation_hidden_dim),
                nn.Linear(
                    pair_dim + cfg.relation_hidden_dim,
                    cfg.hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )
            nn.init.normal_(self.pair_scorer[-1].weight, std=1e-3)
            nn.init.zeros_(self.pair_scorer[-1].bias)
            nn.init.ones_(self.correspondence_evidence.weight)
            nn.init.orthogonal_(self.state_attention.weight)

        @property
        def unary_logit_scale(self) -> Any:
            """Compatibility name for diagnostics written by the one-stage model."""

            return self.logit_scale

        def _text(self, token_ids: Any) -> tuple[Any, Any]:
            text_mask = token_ids.ne(0).any(dim=1, keepdim=True)
            if cfg.text_encoder == LEARNED_TOKEN_LOOKUP_ENCODER:
                token_mask = token_ids.ne(0).unsqueeze(-1)
                embedded = self.token_embedding(token_ids)
                pooled = (embedded * token_mask).sum(dim=1) / token_mask.sum(
                    dim=1
                ).clamp_min(1)
                text = self.text_projection(pooled)
            elif cfg.text_encoder == HASHED_NGRAM_TEXT_ENCODER:
                text = hashed_ngram_descriptor_torch(token_ids, torch=torch)
            else:
                text = self.present_text.unsqueeze(0).expand(token_ids.shape[0], -1)
            missing = self.missing_text.unsqueeze(0).expand_as(text)
            return torch.where(text_mask, text, missing), text_mask

        def encode_nodes(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> tuple[Any, Any, Any, Any]:
            text, text_mask = self._text(token_ids)
            visual = self.visual_encoder(visual_patches)
            visual = visual_mask * visual + (1.0 - visual_mask) * (
                self.missing_visual.unsqueeze(0).expand_as(visual)
            )
            xml = self.xml_projection(numeric_features)
            modalities = torch.stack(
                (
                    self.text_to_hidden(text),
                    self.visual_to_hidden(visual),
                    self.xml_to_hidden(xml),
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
            route_weights = torch.softmax(
                modality_logits / float(cfg.router_temperature),
                dim=1,
            )
            fused = torch.sum(route_weights.unsqueeze(-1) * modalities, dim=1)
            states = self.input_norm(fused)
            states = self.input_output_norm(
                states + self.input_feed_forward(states)
            )
            return states, visual, route_weights, text_mask

        def _relations(
            self,
            states: Any,
            relations: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            count = states.shape[0]
            available = relations[..., 17].gt(0.0)
            available = available & ~torch.eye(
                count,
                dtype=torch.bool,
                device=relations.device,
            )
            if cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION:
                queries = self.neighbor_query(states)
                keys = self.neighbor_key(states)
                priority = (queries @ keys.T) / math.sqrt(
                    float(cfg.relation_hidden_dim)
                )
                priority = priority + self.neighbor_relation_score(
                    relations
                ).squeeze(-1)
            else:
                priority = relations[..., 1:9].abs().sum(dim=-1)
                priority = priority + 1.0 - 0.5 * (
                    relations[..., 11] + relations[..., 12]
                ).clamp(max=2.0)
            priority = priority.masked_fill(~available, -1e4)
            kept = min(max_neighbours, count)
            indices = priority.topk(kept, dim=1).indices
            mask = available.gather(1, indices)
            selected_scores = priority.gather(1, indices)
            relation_values = relations.gather(
                1,
                indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    relations.shape[-1],
                ),
            )
            selected_tokens = self.relation_encoder(
                torch.cat((states[indices], relation_values), dim=-1)
            )
            masked_scores = selected_scores.masked_fill(~mask, -1e4)
            if cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION:
                selection_weights = torch.softmax(masked_scores, dim=1)
            else:
                selection_weights = mask.to(selected_scores.dtype)
            selection_weights = selection_weights * mask.to(selected_scores.dtype)
            selection_weights = selection_weights / selection_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            if self.training and cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION:
                # Forward remains an exact Top-K model.  The backward pass sees
                # every structurally OR spatially OR order-connected candidate,
                # so an initially excluded useful neighbour can still be learned.
                all_tokens = self.relation_encoder(
                    torch.cat(
                        (
                            states.unsqueeze(0).expand(count, -1, -1),
                            relations,
                        ),
                        dim=-1,
                    )
                )
                full_weights = torch.softmax(priority, dim=1)
                full_weights = full_weights * available.to(full_weights.dtype)
                full_weights = full_weights / full_weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
                hard_weights = torch.zeros_like(full_weights).scatter(
                    1,
                    indices,
                    selection_weights.detach(),
                )
                straight_through_weights = (
                    hard_weights + full_weights - full_weights.detach()
                )
                soft_context = torch.sum(
                    straight_through_weights.unsqueeze(-1) * all_tokens,
                    dim=1,
                )
                # Exact Top-K values in the forward pass; all OR-candidates
                # receive selector gradients in the backward pass.
                selected_tokens = selected_tokens + (
                    soft_context - soft_context.detach()
                ).unsqueeze(1)
            return (
                selected_tokens,
                mask,
                indices,
                selected_scores,
                selection_weights,
            )

        @staticmethod
        def _pair_features(source: Any, target: Any) -> Any:
            source_grid = source[:, None, :].expand(-1, target.shape[0], -1)
            target_grid = target[None, :, :].expand(source.shape[0], -1, -1)
            return torch.cat(
                (
                    source_grid,
                    target_grid,
                    torch.abs(source_grid - target_grid),
                    source_grid * target_grid,
                ),
                dim=-1,
            )

        def _local_match(
            self,
            pair_features: Any,
            soft_correspondence: Any,
            source_relations: Any,
            source_mask: Any,
            source_neighbor_indices: Any,
            target_relations: Any,
            target_mask: Any,
            target_neighbor_indices: Any,
        ) -> tuple[Any, Any, Any, Any]:
            query = self.relation_query(pair_features)
            source_keys = self.relation_key(source_relations)
            target_keys = self.relation_key(target_relations)
            compatibility = torch.einsum(
                "nkr,mlr->nmkl",
                source_keys,
                target_keys,
            ) / math.sqrt(float(cfg.relation_hidden_dim))
            compatibility = compatibility + torch.einsum(
                "nmr,nkr->nmk",
                query,
                source_keys,
            ).unsqueeze(-1)
            compatibility = compatibility + torch.einsum(
                "nmr,mlr->nml",
                query,
                target_keys,
            ).unsqueeze(-2)
            mask = source_mask[:, None, :, None] & target_mask[None, :, None, :]
            anchor_confidence = soft_correspondence[
                source_neighbor_indices[:, None, :, None],
                target_neighbor_indices[None, :, None, :],
            ]
            correspondence_support = self._relative_correspondence_support(
                soft_correspondence
            )[
                source_neighbor_indices[:, None, :, None],
                target_neighbor_indices[None, :, None, :],
            ]
            evidence = compatibility + self.correspondence_evidence(
                correspondence_support.unsqueeze(-1)
            ).squeeze(-1)
            flat_mask = mask.flatten(start_dim=2)
            flat_evidence = evidence.flatten(start_dim=2)
            masked_evidence = flat_evidence.masked_fill(~flat_mask, -1e4)
            flat_weights = torch.softmax(masked_evidence, dim=-1)
            flat_weights = flat_weights * flat_mask.to(flat_weights.dtype)
            flat_weights = flat_weights / flat_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
            weights = flat_weights.reshape_as(evidence)
            valid_count = flat_mask.sum(dim=-1).clamp_min(1)
            valid_any = flat_mask.any(dim=-1)
            soft_or = torch.logsumexp(masked_evidence, dim=-1) - torch.log(
                valid_count.to(masked_evidence.dtype)
            )
            soft_or = torch.where(valid_any, soft_or, torch.zeros_like(soft_or))
            coverage = (
                torch.sigmoid(flat_evidence) * flat_mask.to(flat_evidence.dtype)
            ).sum(dim=-1) / valid_count.to(flat_evidence.dtype)
            source_context = torch.einsum(
                "nmkl,nkr->nmr",
                weights,
                source_relations,
            )
            target_context = torch.einsum(
                "nmkl,mlr->nmr",
                weights,
                target_relations,
            )
            context = self.local_pair(
                torch.cat(
                    (
                        source_context,
                        target_context,
                        torch.abs(source_context - target_context),
                        source_context * target_context,
                        soft_or.unsqueeze(-1),
                        coverage.unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
            support = torch.stack((soft_or, coverage), dim=-1)
            return context, weights, anchor_confidence, support

        @staticmethod
        def _relative_correspondence_support(soft_correspondence: Any) -> Any:
            """Measure whether a match is distinctive in both directions.

            A uniform matrix is zero evidence.  A neighbour pair contributes
            only when it is more plausible than the alternatives in both its
            source row and target column.
            """

            tiny = torch.finfo(soft_correspondence.dtype).tiny
            log_probability = soft_correspondence.clamp_min(tiny).log()
            row_support = (
                log_probability
                - torch.logsumexp(log_probability, dim=1, keepdim=True)
                + math.log(float(soft_correspondence.shape[1]))
            )
            column_support = (
                log_probability
                - torch.logsumexp(log_probability, dim=0, keepdim=True)
                + math.log(float(soft_correspondence.shape[0]))
            )
            return 0.5 * (row_support + column_support)

        def _unary_correspondence(
            self,
            source_states: Any,
            target_states: Any,
        ) -> tuple[Any, Any]:
            affinity = self._affinity(source_states, target_states)
            unary_logits = mutual_log_assignment(affinity)
            return unary_logits, self._soft_correspondence(unary_logits)

        def _affinity(self, source_states: Any, target_states: Any) -> Any:
            source_normalized = torch.nn.functional.normalize(source_states, dim=-1)
            target_normalized = torch.nn.functional.normalize(target_states, dim=-1)
            scale = self.logit_scale.exp().clamp(max=100.0)
            source_matchability = self.matchability_head(source_states).squeeze(-1)
            target_matchability = self.matchability_head(target_states).squeeze(-1)
            return scale * (source_normalized @ target_normalized.T) + 0.5 * (
                source_matchability[:, None] + target_matchability[None, :]
            )

        @staticmethod
        def _soft_correspondence(logits: Any) -> Any:
            source_to_target = torch.softmax(logits, dim=1)
            target_to_source = torch.softmax(logits, dim=0)
            return 0.5 * (source_to_target + target_to_source)

        def _page(self, states: Any) -> tuple[Any, Any]:
            weights = torch.softmax(self.page_attention(states).squeeze(-1), dim=0)
            hidden = torch.sum(weights.unsqueeze(-1) * states, dim=0)
            embedding = torch.nn.functional.normalize(hidden, dim=0)
            return hidden, embedding

        def _state_embedding(self, states: Any) -> Any:
            slot_weights = torch.softmax(
                self.state_attention(states).T,
                dim=1,
            )
            slots = slot_weights @ states
            slots = torch.nn.functional.normalize(slots, dim=-1)
            return torch.nn.functional.normalize(slots.reshape(-1), dim=0)

        @staticmethod
        def _page_score(logits: Any) -> Any:
            row_evidence = torch.logsumexp(logits, dim=1) - math.log(
                float(logits.shape[1])
            )
            column_evidence = torch.logsumexp(logits, dim=0) - math.log(
                float(logits.shape[0])
            )
            row_count = max(1, math.ceil(row_evidence.numel() * 0.5))
            column_count = max(1, math.ceil(column_evidence.numel() * 0.5))
            return 0.5 * (
                row_evidence.topk(row_count).values.mean()
                + column_evidence.topk(column_count).values.mean()
            )

        def encode_page(
            self,
            token_ids: Any,
            numeric: Any,
            relations: Any,
            visual: Any,
            visual_mask: Any,
        ) -> dict[str, Any]:
            """Encode one observation once for reuse across page comparisons."""

            (
                base_states,
                visual_descriptors,
                route_weights,
                text_mask,
            ) = self.encode_nodes(
                token_ids,
                numeric,
                visual,
                visual_mask,
            )
            (
                relation_tokens,
                relation_mask,
                relation_neighbor_indices,
                neighbor_selection_scores,
                neighbor_selection_weights,
            ) = self._relations(base_states, relations)
            # Local relations stay explicit until a source-target pair is
            # compared.  Pre-aggregating them into a node hid region identity
            # and was causally inactive in Dev ablation.
            states = base_states
            page_hidden, _ = self._page(states)
            state_embedding = self._state_embedding(states)
            return {
                "base_states": base_states,
                "states": states,
                "visual_descriptors": visual_descriptors,
                "route_weights": route_weights,
                "text_mask": text_mask,
                "visual_mask": visual_mask,
                "relation_tokens": relation_tokens,
                "relation_mask": relation_mask,
                "relation_neighbor_indices": relation_neighbor_indices,
                "neighbor_selection_scores": neighbor_selection_scores,
                "neighbor_selection_weights": neighbor_selection_weights,
                "relation_bases": typed_relation_bases(
                    relations,
                    numeric,
                    feature_schema_id=cfg.feature_schema_id,
                ),
                "page_hidden": page_hidden,
                "page_embedding": state_embedding,
                # Compatibility aliases: this model publishes one state
                # embedding until stable/active invariance has real supervision.
                "stable_state_embedding": state_embedding,
                "active_state_embedding": state_embedding,
            }

        def match_pages(
            self,
            source_page: dict[str, Any],
            target_page: dict[str, Any],
        ) -> dict[str, Any]:
            """Compare two cached page encodings through the one score path."""

            source_states = source_page["states"]
            target_states = target_page["states"]
            unary_logits, soft_correspondence = self._unary_correspondence(
                source_states,
                target_states,
            )
            assignment_scores = []
            soft_correspondences = []
            local_corrections = []
            local_weights_by_layer = []
            local_anchor_confidence_by_layer = []
            local_support_by_layer = []
            source_states_by_layer = []
            target_states_by_layer = []
            backbone_scores_by_layer = []
            source_fused_states = source_states
            target_fused_states = target_states
            logits = unary_logits
            for association_layer in self.association_layers:
                source_states, target_states = association_layer(
                    source_states,
                    target_states,
                    source_page["relation_bases"],
                    target_page["relation_bases"],
                    self.relation_compatibility,
                )
                pair_features = self._pair_features(source_states, target_states)
                soft_correspondence = self._soft_correspondence(logits)
                soft_correspondences.append(soft_correspondence)
                (
                    local_context,
                    local_weights,
                    local_anchor_confidence,
                    local_support,
                ) = self._local_match(
                    pair_features,
                    soft_correspondence,
                    source_page["relation_tokens"],
                    source_page["relation_mask"],
                    source_page["relation_neighbor_indices"],
                    target_page["relation_tokens"],
                    target_page["relation_mask"],
                    target_page["relation_neighbor_indices"],
                )
                # The v9 assignment logits naturally span well beyond [-1, 1].
                # Capping this residual made the learned local evidence unable
                # to change a ranking even when its neighbourhood was decisive.
                # The shared assignment loss now learns the required scale.
                local_correction = self.pair_scorer(
                    torch.cat(
                        (pair_features, local_context),
                        dim=-1,
                    )
                ).squeeze(-1)
                backbone_logits = mutual_log_assignment(
                    self._affinity(source_states, target_states)
                )
                backbone_scores_by_layer.append(backbone_logits)
                logits = (
                    backbone_logits + local_correction
                    if cfg.score_update == UNARY_RESIDUAL_SCORE_UPDATE
                    else local_correction
                )
                assignment_scores.append(logits)
                local_corrections.append(local_correction)
                local_weights_by_layer.append(local_weights)
                local_anchor_confidence_by_layer.append(local_anchor_confidence)
                local_support_by_layer.append(local_support)
                source_states_by_layer.append(source_states)
                target_states_by_layer.append(target_states)
            logits_ba = logits.T
            return {
                "logits_ab": logits,
                "logits_ba": logits_ba,
                "unary_logits": unary_logits,
                "local_correction": local_correction,
                "soft_correspondence": soft_correspondence,
                "assignment_scores_by_layer": tuple(assignment_scores),
                "soft_correspondence_by_layer": tuple(soft_correspondences),
                "local_corrections_by_layer": tuple(local_corrections),
                "backbone_scores_by_layer": tuple(backbone_scores_by_layer),
                "source_states_by_layer": tuple(source_states_by_layer),
                "target_states_by_layer": tuple(target_states_by_layer),
                "affinity": logits,
                "association_score": logits,
                "association_schema": UNIFIED_ASSOCIATION_SCHEMA_ID,
                "page_pair_score": self._page_score(logits),
                "source_states": source_states,
                "target_states": target_states,
                "source_base_states": source_page["base_states"],
                "target_base_states": target_page["base_states"],
                "source_fused_states": source_fused_states,
                "target_fused_states": target_fused_states,
                "source_descriptors": source_page["base_states"],
                "target_descriptors": target_page["base_states"],
                "source_route_weights": source_page["route_weights"],
                "target_route_weights": target_page["route_weights"],
                "source_text_mask": source_page["text_mask"],
                "target_text_mask": target_page["text_mask"],
                "source_visual_mask": source_page["visual_mask"],
                "target_visual_mask": target_page["visual_mask"],
                "source_visual_descriptors": source_page["visual_descriptors"],
                "target_visual_descriptors": target_page["visual_descriptors"],
                "source_relation_tokens": source_page["relation_tokens"],
                "target_relation_tokens": target_page["relation_tokens"],
                "source_relation_mask": source_page["relation_mask"],
                "target_relation_mask": target_page["relation_mask"],
                "source_relation_neighbor_indices": source_page[
                    "relation_neighbor_indices"
                ],
                "target_relation_neighbor_indices": target_page[
                    "relation_neighbor_indices"
                ],
                "source_neighbor_selection_scores": source_page[
                    "neighbor_selection_scores"
                ],
                "target_neighbor_selection_scores": target_page[
                    "neighbor_selection_scores"
                ],
                "source_neighbor_selection_weights": source_page[
                    "neighbor_selection_weights"
                ],
                "target_neighbor_selection_weights": target_page[
                    "neighbor_selection_weights"
                ],
                "local_match_weights": local_weights,
                "local_anchor_confidence": local_anchor_confidence,
                "local_match_weights_by_layer": tuple(local_weights_by_layer),
                "local_anchor_confidence_by_layer": tuple(
                    local_anchor_confidence_by_layer
                ),
                "local_support_by_layer": tuple(local_support_by_layer),
                "source_config_embedding": source_page["page_embedding"],
                "target_config_embedding": target_page["page_embedding"],
                "source_state_embedding": source_page["page_embedding"],
                "target_state_embedding": target_page["page_embedding"],
                "source_stable_state_embedding": source_page[
                    "stable_state_embedding"
                ],
                "target_stable_state_embedding": target_page[
                    "stable_state_embedding"
                ],
                "source_active_state_embedding": source_page[
                    "active_state_embedding"
                ],
                "target_active_state_embedding": target_page[
                    "active_state_embedding"
                ],
            }

        def forward(
            self,
            source_token_ids: Any,
            source_numeric: Any,
            source_relations: Any,
            target_token_ids: Any,
            target_numeric: Any,
            target_relations: Any,
            source_visual: Any,
            source_visual_mask: Any,
            target_visual: Any,
            target_visual_mask: Any,
            detach_unary_for_relation: bool = False,
            node_only: bool = False,
        ) -> dict[str, Any]:
            del detach_unary_for_relation
            if node_only:
                (
                    source_states,
                    source_visual_descriptors,
                    source_route_weights,
                    source_text_mask,
                ) = self.encode_nodes(
                    source_token_ids,
                    source_numeric,
                    source_visual,
                    source_visual_mask,
                )
                (
                    target_states,
                    target_visual_descriptors,
                    target_route_weights,
                    target_text_mask,
                ) = self.encode_nodes(
                    target_token_ids,
                    target_numeric,
                    target_visual,
                    target_visual_mask,
                )
                unary_logits, soft_correspondence = self._unary_correspondence(
                    source_states,
                    target_states,
                )
                return {
                    "logits_ab": unary_logits,
                    "logits_ba": unary_logits.T,
                    "unary_logits": unary_logits,
                    "local_correction": torch.zeros_like(unary_logits),
                    "soft_correspondence": soft_correspondence,
                    "affinity": unary_logits,
                    "association_score": unary_logits,
                    "association_schema": UNIFIED_ASSOCIATION_SCHEMA_ID,
                    "page_pair_score": self._page_score(unary_logits),
                    "source_states": source_states,
                    "target_states": target_states,
                    "source_descriptors": source_states,
                    "target_descriptors": target_states,
                    "source_route_weights": source_route_weights,
                    "target_route_weights": target_route_weights,
                    "source_text_mask": source_text_mask,
                    "target_text_mask": target_text_mask,
                    "source_visual_mask": source_visual_mask,
                    "target_visual_mask": target_visual_mask,
                    "source_visual_descriptors": source_visual_descriptors,
                    "target_visual_descriptors": target_visual_descriptors,
                }
            source_page = self.encode_page(
                source_token_ids,
                source_numeric,
                source_relations,
                source_visual,
                source_visual_mask,
            )
            target_page = self.encode_page(
                target_token_ids,
                target_numeric,
                target_relations,
                target_visual,
                target_visual_mask,
            )
            return self.match_pages(source_page, target_page)

    return PageLocalMatcher()
