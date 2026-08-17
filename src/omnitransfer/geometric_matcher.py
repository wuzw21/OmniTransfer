"""The single trainable OmniTransfer geometric-v9 model."""

from __future__ import annotations

import math
from typing import Any


def build_geometric_matcher(config: Any) -> Any:
    """Build geometric-v9 while preserving the released checkpoint layout."""

    from omnitransfer.learned_matcher import (
        ALIGNMENT_RELATION_FEATURE_INDICES,
        ALL_NODE_CANDIDATE_POLICY,
        DIRECT_PAIR_EVIDENCE_NAMES,
        DIRECT_SEMANTIC_EVIDENCE_NAMES,
        DIRECT_TEXT_EVIDENCE_ENCODER,
        LEARNED_TOKEN_LOOKUP_ENCODER,
        NODE_DESCRIPTOR_DIM,
        OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
        RELATION_FEATURE_DIM,
        TEXT_DESCRIPTOR_DIM,
        TYPED_RELATION_NAMES,
        VISUAL_DESCRIPTOR_DIM,
        XML_DESCRIPTOR_DIM,
        XML_NODE_FEATURE_DIM,
        local_semantic_context_score,
        mutual_log_assignment,
        relational_consensus_features,
        relative_geometry_consensus_features,
        semantic_anchor_relation_vote,
        typed_relation_bases,
        _require_torch,
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
    if cfg.visual_canvas_size < cfg.visual_patch_size:
        raise ValueError("visual_canvas_size must cover one visual patch")
    if cfg.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if cfg.assignment_head != "partial_assignment":
        raise ValueError("geometric-v9 requires partial assignment")

    torch = _require_torch()
    nn = torch.nn

    class SymmetricUnaryHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual = nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim * 2),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

        def forward(self, source_states: Any, target_states: Any) -> Any:
            normalized_source = torch.nn.functional.normalize(source_states, dim=-1)
            normalized_target = torch.nn.functional.normalize(target_states, dim=-1)
            source = source_states[:, None, :].expand(-1, target_states.shape[0], -1)
            target = target_states[None, :, :].expand(source_states.shape[0], -1, -1)
            symmetric_pair = torch.cat(
                (torch.abs(source - target), source * target),
                dim=-1,
            )
            return (
                self.logit_scale.exp().clamp(max=100.0)
                * (normalized_source @ normalized_target.T)
                + self.residual(symmetric_pair).squeeze(-1)
            )

    class AssociationGraphLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.message_projection = nn.Linear(
                cfg.association_dim,
                cfg.association_dim,
                bias=False,
            )
            self.message_norm = nn.LayerNorm(cfg.association_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.association_dim, cfg.association_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.association_dim * 2, cfg.association_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.association_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            pair_states: Any,
            anchor_weights: Any,
            source_bases: Any,
            target_bases: Any,
            relation_compatibility: Any,
        ) -> Any:
            anchored_states = pair_states * anchor_weights.unsqueeze(-1)
            source_messages = torch.einsum(
                "rik,kjd->rijd",
                source_bases,
                anchored_states,
            )
            typed_messages = torch.einsum(
                "rq,rijd->qijd",
                relation_compatibility,
                source_messages,
            )
            messages = torch.einsum(
                "qjl,qild->ijd",
                target_bases,
                typed_messages,
            )
            pair_states = self.message_norm(
                pair_states + self.dropout(self.message_projection(messages))
            )
            return self.output_norm(
                pair_states + self.dropout(self.feed_forward(pair_states))
            )

    class GeometricAlignmentMatcher(nn.Module):
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
            self.missing_text = nn.Parameter(torch.zeros(TEXT_DESCRIPTOR_DIM))
            self.missing_visual = nn.Parameter(torch.zeros(VISUAL_DESCRIPTOR_DIM))
            self.descriptor_fusion = nn.Sequential(
                nn.LayerNorm(NODE_DESCRIPTOR_DIM),
                nn.Linear(NODE_DESCRIPTOR_DIM, NODE_DESCRIPTOR_DIM),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(NODE_DESCRIPTOR_DIM, NODE_DESCRIPTOR_DIM),
            )
            self.descriptor_norm = nn.LayerNorm(NODE_DESCRIPTOR_DIM)
            self.descriptor_projection = nn.Linear(
                NODE_DESCRIPTOR_DIM,
                cfg.hidden_dim,
                bias=False,
            )
            self.input_norm = nn.LayerNorm(cfg.hidden_dim)
            self.unary_head = SymmetricUnaryHead()
            self.relation_compatibility = nn.Parameter(
                torch.eye(len(TYPED_RELATION_NAMES))
            )
            self.voting_strengths = nn.Parameter(torch.zeros(cfg.num_layers))
            pair_feature_dim = NODE_DESCRIPTOR_DIM * 2 + 5
            self.association_pair_encoder = nn.Sequential(
                nn.LayerNorm(pair_feature_dim),
                nn.Linear(pair_feature_dim, cfg.association_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.association_dim, cfg.association_dim),
            )
            self.association_layers = nn.ModuleList(
                AssociationGraphLayer() for _ in range(cfg.association_layers)
            )
            self.association_score = nn.Linear(cfg.association_dim, 1, bias=False)
            nn.init.zeros_(self.association_score.weight)
            self.direct_evidence_head = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.direct_evidence_head.weight)
            self.anchor_voting_strength = nn.Parameter(
                torch.tensor(math.log(math.expm1(2.0)))
            )
            self.context_voting_strength = nn.Parameter(
                torch.tensor(math.log(math.expm1(0.5)))
            )
            feature_dim = (
                len(DIRECT_PAIR_EVIDENCE_NAMES)
                + 16
                + len(TYPED_RELATION_NAMES) ** 2
                + len(ALIGNMENT_RELATION_FEATURE_INDICES)
            )
            self.local_alignment_score = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, 1, bias=False),
            )
            nn.init.zeros_(self.local_alignment_score[-1].weight)

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            if mode:
                for module in self.children():
                    module.eval()
                self.local_alignment_score.train()
            return self

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
                self.missing_visual.unsqueeze(0).expand(visual_states.shape[0], -1)
            )
            text_mask = token_ids.ne(0).any(dim=1, keepdim=True).to(
                dtype=visual_states.dtype
            )
            if pooled_tokens is not None:
                text_states = self.text_projection(pooled_tokens)
            else:
                text_states = self.present_text.unsqueeze(0).expand(
                    token_ids.shape[0],
                    -1,
                )
            text_states = text_mask * text_states + (1.0 - text_mask) * (
                self.missing_text.unsqueeze(0).expand(text_states.shape[0], -1)
            )
            xml_states = self.xml_projection(numeric_features)
            raw_descriptor = torch.cat(
                (text_states, visual_states, xml_states),
                dim=-1,
            )
            descriptor = self.descriptor_norm(
                raw_descriptor + self.descriptor_fusion(raw_descriptor)
            )
            states = self.input_norm(self.descriptor_projection(descriptor))
            return states, descriptor, raw_descriptor

        @staticmethod
        def _symmetric_pair_features(
            source_values: Any,
            target_values: Any,
            source_available: Any,
            target_available: Any,
        ) -> tuple[Any, Any, Any]:
            source = source_values[:, None, :]
            target = target_values[None, :, :]
            both_available = source_available[:, None] * target_available[None, :]
            exactly_one_available = torch.abs(
                source_available[:, None] - target_available[None, :]
            )
            features = torch.cat(
                (torch.abs(source - target), source * target),
                dim=-1,
            )
            return (
                features * both_available.unsqueeze(-1),
                both_available,
                exactly_one_available,
            )

        def _association_pair_states(
            self,
            source_modalities: Any,
            target_modalities: Any,
            source_text_available: Any,
            target_text_available: Any,
            source_visual_available: Any,
            target_visual_available: Any,
            anchor_weights: Any,
        ) -> Any:
            source_text, source_visual, source_xml = torch.split(
                source_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            target_text, target_visual, target_xml = torch.split(
                target_modalities,
                (TEXT_DESCRIPTOR_DIM, VISUAL_DESCRIPTOR_DIM, XML_DESCRIPTOR_DIM),
                dim=-1,
            )
            text, text_both, text_one = self._symmetric_pair_features(
                source_text,
                target_text,
                source_text_available,
                target_text_available,
            )
            visual, visual_both, visual_one = self._symmetric_pair_features(
                source_visual,
                target_visual,
                source_visual_available,
                target_visual_available,
            )
            xml, _, _ = self._symmetric_pair_features(
                source_xml,
                target_xml,
                torch.ones_like(source_text_available),
                torch.ones_like(target_text_available),
            )
            availability = torch.stack(
                (text_both, text_one, visual_both, visual_one, anchor_weights),
                dim=-1,
            )
            return self.association_pair_encoder(
                torch.cat((text, visual, xml, availability), dim=-1)
            )

        def _relation_vote(
            self,
            soft_assignment: Any,
            source_bases: Any,
            target_bases: Any,
        ) -> Any:
            compatibility = 0.5 * (
                self.relation_compatibility + self.relation_compatibility.T
            )
            source_messages = torch.einsum(
                "rik,kl->ril",
                source_bases,
                soft_assignment,
            )
            typed_messages = torch.einsum(
                "rq,ril->qil",
                compatibility,
                source_messages,
            )
            vote = torch.einsum(
                "qil,qjl->ij",
                typed_messages,
                target_bases,
            )
            centered = vote - vote.mean()
            return centered / centered.square().mean().clamp_min(
                torch.finfo(vote.dtype).eps
            ).sqrt()

        @staticmethod
        def _standardize(values: Any) -> Any:
            centered = values - values.mean(dim=(0, 1), keepdim=True)
            return centered / centered.square().mean(
                dim=(0, 1),
                keepdim=True,
            ).clamp_min(torch.finfo(centered.dtype).eps).sqrt()

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
            source_states, source_descriptors, source_modalities = self.encode_nodes(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            target_states, target_descriptors, target_modalities = self.encode_nodes(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            unary_affinity = self.unary_head(source_states, target_states)
            source_bases = typed_relation_bases(source_relations, source_numeric)
            target_bases = typed_relation_bases(target_relations, target_numeric)
            soft_assignment = torch.exp(mutual_log_assignment(unary_affinity))
            assignments = []
            affinities = []
            relation_votes = []
            soft_assignments = []
            attention_diagnostics = []
            for round_index in range(cfg.num_layers):
                relation_vote = self._relation_vote(
                    soft_assignment,
                    source_bases,
                    target_bases,
                )
                affinity = unary_affinity + torch.nn.functional.softplus(
                    self.voting_strengths[round_index]
                ) * relation_vote
                assignment = mutual_log_assignment(affinity)
                soft_assignment = torch.exp(mutual_log_assignment(affinity))
                assignments.append(assignment)
                affinities.append(affinity)
                relation_votes.append(relation_vote)
                soft_assignments.append(soft_assignment)
                attention_diagnostics.append(
                    {
                        "soft_assignment": soft_assignment,
                        "relation_vote": relation_vote,
                    }
                )
            base_affinity = affinities[-1]
            if direct_pair_evidence.shape != (
                base_affinity.shape[0],
                base_affinity.shape[1],
                RELATION_FEATURE_DIM,
            ):
                raise ValueError("direct pair evidence must align with candidate pairs")
            direct_evidence_residual = self.direct_evidence_head(
                direct_pair_evidence[
                    ..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)
                ].sum(dim=-1, keepdim=True)
            ).squeeze(-1)
            anchor_weights = torch.exp(
                mutual_log_assignment(base_affinity + direct_evidence_residual)
            )
            pair_states = self._association_pair_states(
                source_modalities,
                target_modalities,
                source_token_ids.ne(0).any(dim=1).to(base_affinity.dtype),
                target_token_ids.ne(0).any(dim=1).to(base_affinity.dtype),
                source_visual_mask.squeeze(-1).to(base_affinity.dtype),
                target_visual_mask.squeeze(-1).to(base_affinity.dtype),
                anchor_weights,
            )
            compatibility = 0.5 * (
                self.relation_compatibility + self.relation_compatibility.T
            )
            association_states = []
            for layer in self.association_layers:
                pair_states = layer(
                    pair_states,
                    anchor_weights,
                    source_bases,
                    target_bases,
                    compatibility,
                )
                association_states.append(pair_states)
            association_residual = self.association_score(pair_states).squeeze(-1)
            anchor_relation_vote = semantic_anchor_relation_vote(
                direct_pair_evidence,
                source_bases,
                target_bases,
                compatibility,
            )
            anchor_vote_residual = torch.nn.functional.softplus(
                self.anchor_voting_strength
            ) * anchor_relation_vote
            context_relation_score = local_semantic_context_score(
                direct_pair_evidence,
                source_bases,
                target_bases,
                source_numeric,
                target_numeric,
            )
            context_vote_residual = torch.nn.functional.softplus(
                self.context_voting_strength
            ) * context_relation_score
            affinity = (
                base_affinity
                + association_residual
                + direct_evidence_residual
                + anchor_vote_residual
                + context_vote_residual
            )
            assignment = mutual_log_assignment(affinity)
            score_components = torch.stack(
                (
                    unary_affinity,
                    base_affinity,
                    association_residual,
                    direct_evidence_residual,
                    anchor_vote_residual,
                    context_relation_score,
                    context_vote_residual,
                    affinity,
                ),
                dim=-1,
            )
            local_consensus = self._standardize(
                relational_consensus_features(
                    assignment,
                    source_bases,
                    target_bases,
                )
            )
            geometry_consensus = self._standardize(
                relative_geometry_consensus_features(
                    assignment,
                    source_relations,
                    target_relations,
                )
            )
            local_features = torch.cat(
                (
                    direct_pair_evidence,
                    torch.tanh(score_components / 5.0),
                    self._standardize(score_components),
                    local_consensus,
                    geometry_consensus,
                ),
                dim=-1,
            )
            local_residual = self.local_alignment_score(local_features).squeeze(-1)
            final_affinity = affinity + local_residual
            final_assignment = mutual_log_assignment(final_affinity)
            return {
                "logits_ab": final_assignment,
                "logits_ba": final_assignment.T,
                "affinity": final_affinity,
                "unary_affinity": unary_affinity,
                "base_affinity": base_affinity,
                "association_residual": association_residual,
                "direct_evidence_residual": direct_evidence_residual,
                "anchor_relation_vote": anchor_relation_vote,
                "anchor_vote_residual": anchor_vote_residual,
                "context_relation_score": context_relation_score,
                "context_vote_residual": context_vote_residual,
                "direct_pair_evidence": direct_pair_evidence,
                "v9_affinity": affinity,
                "local_alignment_residual": local_residual,
                "local_alignment_base_residual": local_residual,
                "geometry_alignment_residual": torch.zeros_like(local_residual),
                "local_alignment_consensus": local_consensus,
                "local_alignment_geometry": geometry_consensus,
                "assignment_scores_by_layer": (final_assignment,),
                "assignment_loss_weights": (1.0,),
                "affinities_by_layer": (final_affinity,),
                "base_assignment_scores_by_layer": tuple(assignments),
                "base_affinities_by_layer": tuple(affinities),
                "relation_votes_by_layer": tuple(relation_votes),
                "soft_assignments_by_layer": tuple(soft_assignments),
                "attention_by_layer": tuple(attention_diagnostics),
                "association_states_by_layer": tuple(association_states),
                "source_relation_bases": source_bases,
                "target_relation_bases": target_bases,
                "source_descriptors": source_descriptors,
                "target_descriptors": target_descriptors,
                "source_states": source_states,
                "target_states": target_states,
            }

    return GeometricAlignmentMatcher()
