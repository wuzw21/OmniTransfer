"""The single trainable OmniTransfer page-local correspondence model.

The v9 checkpoint keeps its bounded-neighbour path.  The v10 variant exposes a
point-conditioned sparse local graph learned inside a tolerant spatial/tree
envelope, then aligns those graphs through the same real-node correspondence
matrix.  Both variants share one multimodal encoder and one 1024D page readout;
neither adds a match NULL class, reranker, or auxiliary inference path.
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
        POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE,
        POINT_CONDITIONED_SPARSE_GRAPH_SELECTION,
        RELATION_FEATURE_DIM,
        SPATIAL_CNN_VISUAL_ENCODER,
        TEXT_DESCRIPTOR_DIM,
        TYPED_RELATION_NAMES,
        UNARY_RESIDUAL_SCORE_UPDATE,
        VISUAL_DESCRIPTOR_DIM,
        XML_DESCRIPTOR_DIM,
        SUPPORTED_MATCHER_ARCHITECTURES,
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
    if cfg.architecture not in SUPPORTED_MATCHER_ARCHITECTURES:
        raise ValueError("unsupported OmniTransfer matcher architecture")
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
    if cfg.local_neighbor_limit < 0:
        raise ValueError("local_neighbor_limit must be non-negative")
    if cfg.local_graph_layers <= 0:
        raise ValueError("local_graph_layers must be positive")
    if cfg.local_candidate_limit <= 0:
        raise ValueError("local_candidate_limit must be positive")
    if not 0.0 < cfg.local_spatial_radius <= 1.0:
        raise ValueError("local_spatial_radius must be in (0, 1]")
    if cfg.local_tree_hops <= 0:
        raise ValueError("local_tree_hops must be positive")
    if cfg.router_temperature <= 0.0:
        raise ValueError("router_temperature must be positive")
    if cfg.state_embedding_dim <= 0:
        raise ValueError("state_embedding_dim must be positive")
    if cfg.state_embedding_dim % cfg.hidden_dim:
        raise ValueError("state_embedding_dim must be divisible by hidden_dim")
    if cfg.neighbor_selection not in {
        FIXED_NEIGHBOR_SELECTION,
        LEARNED_NEIGHBOR_SELECTION,
        POINT_CONDITIONED_SPARSE_GRAPH_SELECTION,
    }:
        raise ValueError(f"unsupported neighbor selection: {cfg.neighbor_selection}")
    if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
        if cfg.neighbor_selection != POINT_CONDITIONED_SPARSE_GRAPH_SELECTION:
            raise ValueError("v10 requires point-conditioned sparse graph selection")
        if cfg.local_neighbor_limit != 0:
            raise ValueError("v10 learns variable graph support; local limit must be 0")

    torch = _require_torch()
    nn = torch.nn
    visual_dim = visual_descriptor_dim(cfg.visual_encoder)
    numeric_dim = xml_node_feature_dim(cfg.feature_schema_id)
    # A positive value keeps the historical bounded selector.  Zero is the
    # explicit full-local mode used by the correspondence-conditioned learner:
    # every valid within-page relation remains available to each refinement
    # stage, with padding only to the page's maximum valid degree.
    max_neighbours = cfg.local_neighbor_limit
    state_slots = cfg.state_embedding_dim // cfg.hidden_dim
    relation_count = len(TYPED_RELATION_NAMES)

    def entmax15(values: Any, *, dim: int = -1) -> Any:
        """Differentiable 1.5-entmax with exact zeros and variable support."""

        scaled = values / 2.0
        scaled = scaled - scaled.max(dim=dim, keepdim=True).values
        sorted_values = scaled.sort(dim=dim, descending=True).values
        dimension = sorted_values.shape[dim]
        view = [1] * sorted_values.ndim
        view[dim] = dimension
        rho = torch.arange(
            1,
            dimension + 1,
            dtype=values.dtype,
            device=values.device,
        ).view(view)
        mean = sorted_values.cumsum(dim) / rho
        mean_square = sorted_values.square().cumsum(dim) / rho
        variance_sum = rho * (mean_square - mean.square())
        delta = (1.0 - variance_sum) / rho
        taus = mean - delta.clamp_min(torch.finfo(values.dtype).eps).sqrt()
        support = taus.le(sorted_values)
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau_star = taus.gather(dim, support_size - 1)
        probabilities = (scaled - tau_star).clamp_min(0.0).square()
        return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).eps
        )

    class PointConditionedGraphLearner(nn.Module):
        """Learn a sparse graph only inside a tolerant local access envelope."""

        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(
                cfg.hidden_dim,
                cfg.relation_hidden_dim,
                bias=False,
            )
            self.key = nn.Linear(
                cfg.hidden_dim,
                cfg.relation_hidden_dim,
                bias=False,
            )
            self.relation_score = nn.Sequential(
                nn.LayerNorm(RELATION_FEATURE_DIM),
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, 1, bias=False),
            )
            self.bandwidth = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 2),
            )
            self.null_score = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1),
            )
            nn.init.zeros_(self.bandwidth[-1].weight)
            nn.init.zeros_(self.bandwidth[-1].bias)
            nn.init.zeros_(self.null_score[-1].weight)
            nn.init.constant_(self.null_score[-1].bias, -1.0)

        def _locality(
            self,
            relations: Any,
            numeric: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            features = relations.features
            neighbor_indices = relations.neighbor_indices
            bbox_present = numeric[:, 4].gt(0.0)
            bbox_pair = bbox_present[:, None] & bbox_present[neighbor_indices]
            spatial_distance = torch.sqrt(
                features[..., 11].square() + features[..., 12].square()
            )
            spatial_local = bbox_pair & spatial_distance.le(
                float(cfg.local_spatial_radius)
            )
            structural_relation = features[..., 1:6].gt(0.0).any(dim=-1)
            tree_distance = features[..., 16].clamp(0.0, 1.0)
            structural_local = structural_relation & tree_distance.le(
                float(cfg.local_tree_hops) / 16.0
            )
            available = relations.mask & (spatial_local | structural_local)
            return (
                available,
                spatial_local,
                structural_local,
                spatial_distance,
                tree_distance,
            )

        def forward(
            self,
            states: Any,
            relations: Any,
            numeric: Any,
        ) -> dict[str, Any]:
            (
                available,
                spatial_local,
                structural_local,
                spatial_distance,
                tree_distance,
            ) = self._locality(relations, numeric)
            raw_bandwidth = torch.sigmoid(self.bandwidth(states))
            spatial_bandwidth = 0.05 + raw_bandwidth[:, 0] * float(
                cfg.local_spatial_radius
            )
            tree_bandwidth = (1.0 + raw_bandwidth[:, 1] * float(
                cfg.local_tree_hops
            )) / 16.0
            neighbor_keys = self.key(states)[relations.neighbor_indices]
            point_score = torch.sum(
                self.query(states).unsqueeze(1) * neighbor_keys,
                dim=-1,
            ) / math.sqrt(float(cfg.relation_hidden_dim))
            relation_score = self.relation_score(relations.features).squeeze(-1)
            spatial_penalty = torch.where(
                spatial_local,
                spatial_distance / spatial_bandwidth[:, None],
                torch.zeros_like(spatial_distance),
            )
            tree_penalty = torch.where(
                structural_local,
                tree_distance / tree_bandwidth[:, None],
                torch.zeros_like(tree_distance),
            )
            edge_scores = point_score + relation_score - spatial_penalty - tree_penalty
            edge_scores = edge_scores.masked_fill(~available, -1e4)
            logits = torch.cat((edge_scores, self.null_score(states)), dim=1)
            distribution = entmax15(logits, dim=1)
            edge_weights = distribution[:, :-1] * available.to(distribution.dtype)
            null_weights = distribution[:, -1]
            return {
                "locality_mask": available,
                "neighbor_indices": relations.neighbor_indices,
                "spatial_locality_mask": spatial_local & available,
                "structural_locality_mask": structural_local & available,
                "edge_scores": edge_scores,
                "edge_weights": edge_weights,
                "null_weights": null_weights,
                "bandwidths": torch.stack(
                    (spatial_bandwidth, tree_bandwidth * 16.0),
                    dim=1,
                ),
            }

    class SparseGraphUpdate(nn.Module):
        """Update points through learned edge states without destroying identity."""

        def __init__(self) -> None:
            super().__init__()
            self.node_value = nn.Linear(
                cfg.hidden_dim,
                cfg.relation_hidden_dim,
                bias=False,
            )
            self.edge_value = nn.Sequential(
                nn.LayerNorm(RELATION_FEATURE_DIM),
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
            )
            self.message_projection = nn.Linear(
                cfg.relation_hidden_dim,
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

        def forward(self, states: Any, relations: Any, edge_weights: Any) -> Any:
            neighbor_values = self.node_value(states)[relations.neighbor_indices]
            edge_messages = neighbor_values + self.edge_value(relations.features)
            message = torch.sum(edge_weights.unsqueeze(-1) * edge_messages, dim=1)
            updated = self.message_norm(
                states + self.dropout(self.message_projection(message))
            )
            return self.output_norm(
                updated + self.dropout(self.feed_forward(updated))
            )

    class DeterministicIconEncoder(nn.Module):
        def forward(self, patches: Any) -> Any:
            return deterministic_icon_descriptor_torch(patches, torch=torch)

    class MultiscaleHashEncoder(nn.Module):
        def forward(self, patches: Any) -> Any:
            return multiscale_hash_descriptor_torch(patches, torch=torch)

    class TrainableMultiscaleResidualEncoder(nn.Module):
        """Keep the stable descriptor while learning spatial ROI corrections."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Flatten(),
            )
            spatial_size = math.ceil(float(cfg.visual_patch_size) / 4.0)
            self.dense_readout = nn.Linear(
                32 * int(spatial_size) * int(spatial_size),
                visual_dim,
                bias=False,
            )
            # The first forward is exactly the stable descriptor.  The shared
            # assignment loss then learns a spatial correction from the tight
            # ROI and its context ring instead of learning from random vision.
            nn.init.zeros_(self.dense_readout.weight)

        def forward(self, patches: Any) -> Any:
            fixed = multiscale_hash_descriptor_torch(patches, torch=torch)
            return fixed + self.dense_readout(self.features(patches))

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
        if cfg.visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER:
            return MultiscaleHashEncoder()
        if cfg.visual_encoder == MULTISCALE_RESIDUAL_VISUAL_ENCODER:
            return TrainableMultiscaleResidualEncoder()
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
            if cfg.architecture != POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
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
            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                # The page encoder has already produced and exposed the learned
                # sparse local graph. Correspondence stages only exchange
                # cross-page evidence, avoiding a second hidden graph owner.
                source_local = source_states
                target_local = target_states
            else:
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
            if cfg.architecture != POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                self.relation_compatibility = nn.Parameter(
                    torch.eye(relation_count)
                )
            self.association_layers = nn.ModuleList(
                ContextualRefinementLayer()
                for _ in range(cfg.association_layers)
            )
            self.matchability_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1),
            )

            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                self.sparse_graph_learners = nn.ModuleList(
                    PointConditionedGraphLearner()
                    for _ in range(cfg.local_graph_layers)
                )
                self.sparse_graph_updates = nn.ModuleList(
                    SparseGraphUpdate() for _ in range(cfg.local_graph_layers)
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
            if cfg.architecture != POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
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
            if cfg.architecture != POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                self.correspondence_evidence = nn.Linear(1, 1, bias=False)
            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                self.graph_prior_scale = nn.Parameter(torch.tensor(1.0))

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
            if cfg.architecture != POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
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
            learned_graph: dict[str, Any] | None = None,
        ) -> tuple[Any, Any, Any, Any, Any]:
            count = states.shape[0]
            if learned_graph is not None:
                available = learned_graph["locality_mask"]
                priority = learned_graph["edge_scores"]
                full_selection_weights = learned_graph["edge_weights"]
                active = available & full_selection_weights.gt(0.0)
                kept = max(1, int(active.sum(dim=1).max().item()))
                ranking_scores = priority.masked_fill(~active, -1e4)
                selected_slots = ranking_scores.topk(kept, dim=1).indices
                indices = relations.neighbor_indices.gather(1, selected_slots)
                mask = active.gather(1, selected_slots)
                selected_scores = priority.gather(1, selected_slots)
                selection_weights = full_selection_weights.gather(
                    1, selected_slots
                )
                relation_values = relations.features.gather(
                    1,
                    selected_slots.unsqueeze(-1).expand(
                        -1,
                        -1,
                        relations.features.shape[-1],
                    ),
                )
            else:
                available = relations[..., 17].gt(0.0)
                available = available & ~torch.eye(
                    count,
                    dtype=torch.bool,
                    device=relations.device,
                )
            if learned_graph is None and cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION:
                queries = self.neighbor_query(states)
                keys = self.neighbor_key(states)
                priority = (queries @ keys.T) / math.sqrt(
                    float(cfg.relation_hidden_dim)
                )
                priority = priority + self.neighbor_relation_score(
                    relations
                ).squeeze(-1)
            elif learned_graph is None:
                priority = relations[..., 1:9].abs().sum(dim=-1)
                priority = priority + 1.0 - 0.5 * (
                    relations[..., 11] + relations[..., 12]
                ).clamp(max=2.0)
            if learned_graph is None:
                priority = priority.masked_fill(~available, -1e4)
                kept = count if max_neighbours == 0 else min(max_neighbours, count)
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
            if learned_graph is None:
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
            else:
                selection_weights = selection_weights * mask.to(
                    selection_weights.dtype
                )
            if (
                self.training
                and learned_graph is None
                and cfg.neighbor_selection == LEARNED_NEIGHBOR_SELECTION
                and max_neighbours > 0
            ):
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
            source_graph_weights: Any,
            target_relations: Any,
            target_mask: Any,
            target_neighbor_indices: Any,
            target_graph_weights: Any,
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
            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                tiny = torch.finfo(evidence.dtype).eps
                graph_log_prior = (
                    source_graph_weights[:, None, :, None].clamp_min(tiny).log()
                    + target_graph_weights[None, :, None, :]
                    .clamp_min(tiny)
                    .log()
                )
                evidence = evidence + self.graph_prior_scale * graph_log_prior
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

        def _sparse_graph_match(
            self,
            pair_features: Any,
            soft_correspondence: Any,
            source_graph: dict[str, Any],
            source_edge_tokens: Any,
            target_graph: dict[str, Any],
            target_edge_tokens: Any,
        ) -> tuple[Any, Any, Any, Any]:
            """Align two explicit learned graphs without enumerating edge pairs."""

            source_edges = (
                source_graph["edge_weights"].unsqueeze(-1) * source_edge_tokens
            )
            target_edges = (
                target_graph["edge_weights"].unsqueeze(-1) * target_edge_tokens
            )
            source_neighbor_correspondence = soft_correspondence[
                source_graph["neighbor_indices"]
            ]
            source_aligned = torch.einsum(
                "ikr,ikm->imr",
                source_edges,
                source_neighbor_correspondence,
            )
            source_support = torch.einsum(
                "ik,ikm->im",
                source_graph["edge_weights"],
                source_neighbor_correspondence,
            )
            relation_consistency = torch.zeros(
                pair_features.shape[0],
                pair_features.shape[1],
                source_edge_tokens.shape[-1],
                dtype=pair_features.dtype,
                device=pair_features.device,
            )
            graph_support = torch.zeros_like(soft_correspondence)
            for slot in range(target_graph["neighbor_indices"].shape[1]):
                target_indices = target_graph["neighbor_indices"][:, slot]
                relation_consistency = relation_consistency + (
                    source_aligned[:, target_indices, :]
                    * target_edges[:, slot, :].unsqueeze(0)
                )
                graph_support = graph_support + (
                    source_support[:, target_indices]
                    * target_graph["edge_weights"][:, slot].unsqueeze(0)
                )
            source_mass = source_graph["edge_weights"].sum(dim=1)
            target_mass = target_graph["edge_weights"].sum(dim=1)
            available_mass = source_mass[:, None] * target_mass[None, :]
            stable_mass = 1e-4
            normalized_support = graph_support / available_mass.clamp_min(stable_mass)
            normalized_consistency = relation_consistency / graph_support.clamp_min(
                stable_mass
            ).unsqueeze(-1)
            normalized_consistency = torch.where(
                graph_support.unsqueeze(-1).gt(stable_mass),
                normalized_consistency,
                torch.zeros_like(normalized_consistency),
            )
            normalized_support = torch.where(
                available_mass.gt(stable_mass),
                normalized_support,
                torch.zeros_like(normalized_support),
            )
            query = self.relation_query(pair_features)
            scaled_consistency = self.graph_prior_scale * normalized_consistency
            context = self.local_pair(
                torch.cat(
                    (
                        query,
                        scaled_consistency,
                        torch.abs(query - scaled_consistency),
                        query * scaled_consistency,
                        graph_support.unsqueeze(-1),
                        normalized_support.unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )
            support = torch.stack((graph_support, normalized_support), dim=-1)
            return context, graph_support, graph_support, support

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
            states = base_states
            learned_graph = None
            learned_graph_by_layer = []
            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                for graph_learner, graph_update in zip(
                    self.sparse_graph_learners,
                    self.sparse_graph_updates,
                    strict=True,
                ):
                    learned_graph = graph_learner(states, relations, numeric)
                    learned_graph_by_layer.append(learned_graph)
                    states = graph_update(
                        states,
                        relations,
                        learned_graph["edge_weights"],
                    )
            (
                relation_tokens,
                relation_mask,
                relation_neighbor_indices,
                neighbor_selection_scores,
                neighbor_selection_weights,
            ) = self._relations(states, relations, learned_graph)
            page_hidden, _ = self._page(states)
            state_embedding = self._state_embedding(states)
            encoded = {
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
                "relation_bases": (
                    None
                    if cfg.architecture
                    == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE
                    else typed_relation_bases(
                        relations,
                        numeric,
                        feature_schema_id=cfg.feature_schema_id,
                    )
                ),
                "page_hidden": page_hidden,
                "page_embedding": state_embedding,
                # Compatibility aliases: this model publishes one state
                # embedding until stable/active invariance has real supervision.
                "stable_state_embedding": state_embedding,
                "active_state_embedding": state_embedding,
            }
            if learned_graph is not None:
                local_graph_edge_tokens = self.relation_encoder(
                    torch.cat(
                        (
                            states[relations.neighbor_indices],
                            relations.features,
                        ),
                        dim=-1,
                    )
                )
                encoded.update(
                    {
                        "local_graph": learned_graph,
                        "local_graph_by_layer": tuple(learned_graph_by_layer),
                        "local_graph_edge_tokens": local_graph_edge_tokens,
                        "locality_mask": learned_graph["locality_mask"],
                        "local_graph_edge_scores": learned_graph["edge_scores"],
                        "local_graph_edge_weights": learned_graph["edge_weights"],
                        "local_graph_null_weights": learned_graph["null_weights"],
                        "local_graph_bandwidths": learned_graph["bandwidths"],
                    }
                )
            return encoded

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
                    (
                        None
                        if cfg.architecture
                        == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE
                        else self.relation_compatibility
                    ),
                )
                pair_features = self._pair_features(source_states, target_states)
                soft_correspondence = self._soft_correspondence(logits)
                soft_correspondences.append(soft_correspondence)
                if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                    (
                        local_context,
                        local_weights,
                        local_anchor_confidence,
                        local_support,
                    ) = self._sparse_graph_match(
                        pair_features,
                        soft_correspondence,
                        source_page["local_graph"],
                        source_page["local_graph_edge_tokens"],
                        target_page["local_graph"],
                        target_page["local_graph_edge_tokens"],
                    )
                else:
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
                        source_page["neighbor_selection_weights"],
                        target_page["relation_tokens"],
                        target_page["relation_mask"],
                        target_page["relation_neighbor_indices"],
                        target_page["neighbor_selection_weights"],
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
            output = {
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
            if cfg.architecture == POINT_CONDITIONED_SPARSE_GRAPH_ARCHITECTURE:
                output.update(
                    {
                        "source_local_graph": source_page["local_graph"],
                        "target_local_graph": target_page["local_graph"],
                        "source_local_graph_by_layer": source_page[
                            "local_graph_by_layer"
                        ],
                        "target_local_graph_by_layer": target_page[
                            "local_graph_by_layer"
                        ],
                    }
                )
            return output

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
