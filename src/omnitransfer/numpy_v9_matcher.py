"""NumPy inference for the frozen OmniTransfer geometric-alignment v9 model."""

from __future__ import annotations

import json
import math
from io import BytesIO
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from omnitransfer.learned_matcher import (
    ALIGNMENT_RELATION_FEATURE_INDICES,
    ALL_NODE_CANDIDATE_POLICY,
    DIRECT_SEMANTIC_EVIDENCE_NAMES,
    LearnedMatch,
    MatcherConfig,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    OMNITRANSFER_V7_FEATURE_SCHEMA_ID,
    RELATION_FEATURE_DIM,
    TEXT_DESCRIPTOR_DIM,
    TYPED_RELATION_NAMES,
    VISUAL_DESCRIPTOR_DIM,
    XML_DESCRIPTOR_DIM,
    cross_relation_features,
    encode_graph,
    is_actionable,
    spatial_xml_alignment_choice,
)
from omnitransfer.numpy_matcher import (
    _decode,
    _encode,
    _gelu,
    _log_softmax,
    _require_numpy,
    _sigmoid,
    _softmax,
    _visual_inputs,
    np_float,
)
from omnitransfer.ui_graph import UIGraph, local_context_graph

NUMPY_GEOMETRIC_V9_SCHEMA = "omnitransfer_numpy_geometric_alignment_v9_v1"


class NumpyGeometricAlignmentMatcher:
    """Inference-only v9 matcher for the Android Python/NumPy runtime."""

    feature_schema_id = OMNITRANSFER_V7_FEATURE_SCHEMA_ID
    backend = "numpy-v9"

    def __init__(self, weights: Mapping[str, Any], *, config: MatcherConfig) -> None:
        if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
            raise ValueError("NumPy v9 requires geometric-alignment architecture")
        self.weights = dict(weights)
        self.config = config

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NumpyGeometricAlignmentMatcher":
        np = _require_numpy()
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            if _decode(checkpoint["__schema_version__"]) != NUMPY_GEOMETRIC_V9_SCHEMA:
                raise ValueError("checkpoint is not an OmniTransfer NumPy v9 matcher")
            config = MatcherConfig(
                **json.loads(_decode(checkpoint["__config_json__"]))
            )
            weights = {
                name: checkpoint[name].astype(np.float32, copy=True)
                for name in checkpoint.files
                if not name.startswith("__")
            }
        return cls(weights, config=config)

    def predict(
        self,
        source: UIGraph,
        target: UIGraph,
        *,
        source_node_id: str,
        candidate_node_ids: Iterable[str] | None = None,
        min_probability: float = 0.0,
        min_margin: float = 0.0,
    ) -> LearnedMatch:
        source_node = next(
            (node for node in source.nodes if node.node_id == source_node_id),
            None,
        )
        if source_node is None:
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        if (
            self.config.candidate_policy != ALL_NODE_CANDIDATE_POLICY
            and not is_actionable(source_node)
        ):
            return LearnedMatch(None, 0.0, 0.0, "source_node_not_actionable", ())
        if (
            self.config.candidate_policy != ALL_NODE_CANDIDATE_POLICY
            and len(source.nodes) > self.config.source_context_nodes
        ):
            source = local_context_graph(
                source,
                anchor_node_id=source_node_id,
                max_nodes=self.config.source_context_nodes,
            )
        source_index = next(
            (
                index
                for index, node in enumerate(source.nodes)
                if node.node_id == source_node_id
            ),
            None,
        )
        if source_index is None:
            raise AssertionError("source context dropped its anchor node")
        allowed = set(candidate_node_ids or (node.node_id for node in target.nodes))
        candidate_indices = [
            index
            for index, node in enumerate(target.nodes)
            if node.node_id in allowed
            and (
                self.config.candidate_policy == ALL_NODE_CANDIDATE_POLICY
                or is_actionable(node)
            )
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())

        output = self._forward(source, target)
        selected_logits = output["logits_ab"][source_index][candidate_indices].copy()
        selected_affinity = output["affinity"][source_index][candidate_indices]
        choice = spatial_xml_alignment_choice(
            selected_logits,
            source_node=source.nodes[source_index],
            target_nodes=tuple(target.nodes[index] for index in candidate_indices),
            source_graph=source,
            target_graph=target,
        )
        if choice is not None:
            model_position = min(
                range(len(selected_logits)),
                key=lambda position: (
                    -float(selected_logits[position]),
                    target.nodes[candidate_indices[position]].node_id,
                ),
            )
            selected_logits[model_position], selected_logits[choice] = (
                selected_logits[choice],
                selected_logits[model_position],
            )
        rank_probabilities = _softmax(selected_logits, axis=0)
        match_probabilities = _sigmoid(selected_affinity)
        ranked = sorted(
            (
                (target.nodes[index].node_id, float(rank_probabilities[position]))
                for position, index in enumerate(candidate_indices)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        best_position = next(
            position
            for position, index in enumerate(candidate_indices)
            if target.nodes[index].node_id == best_id
        )
        match_probability = float(match_probabilities[best_position])
        second_probability = max((score for _, score in ranked[1:]), default=0.0)
        margin = best_probability - second_probability
        scores = tuple(ranked)
        if match_probability < min_probability or margin < min_margin:
            return LearnedMatch(
                None,
                match_probability,
                margin,
                "learned_low_confidence",
                scores,
            )
        target_node = next(node for node in target.nodes if node.node_id == best_id)
        return LearnedMatch(target_node, match_probability, margin, "learned_match", scores)

    def _forward(self, source: UIGraph, target: UIGraph) -> dict[str, Any]:
        np = _require_numpy()
        source_encoded = encode_graph(
            source,
            config=self.config,
            feature_schema_id=self.feature_schema_id,
        )
        target_encoded = encode_graph(
            target,
            config=self.config,
            feature_schema_id=self.feature_schema_id,
        )
        source_tokens = np.asarray(source_encoded.token_ids, dtype=np.int64)
        target_tokens = np.asarray(target_encoded.token_ids, dtype=np.int64)
        source_numeric = np.asarray(source_encoded.numeric_features, dtype=np.float32)
        target_numeric = np.asarray(target_encoded.numeric_features, dtype=np.float32)
        source_relations = np.asarray(
            source_encoded.relation_features,
            dtype=np.float32,
        )
        target_relations = np.asarray(
            target_encoded.relation_features,
            dtype=np.float32,
        )
        direct_evidence = np.asarray(
            cross_relation_features(
                source,
                target,
                feature_schema_id=self.feature_schema_id,
            ),
            dtype=np.float32,
        )
        source_visual, source_visual_mask = _visual_inputs(
            source,
            patch_size=self.config.visual_patch_size,
            canvas_size=self.config.visual_canvas_size,
        )
        target_visual, target_visual_mask = _visual_inputs(
            target,
            patch_size=self.config.visual_patch_size,
            canvas_size=self.config.visual_canvas_size,
        )
        source_states, source_modalities = self._encode_nodes(
            source_tokens,
            source_numeric,
            source_visual,
            source_visual_mask,
        )
        target_states, target_modalities = self._encode_nodes(
            target_tokens,
            target_numeric,
            target_visual,
            target_visual_mask,
        )
        unary_affinity = self._unary(source_states, target_states)
        source_bases = _typed_relation_bases(source_relations, source_numeric)
        target_bases = _typed_relation_bases(target_relations, target_numeric)
        soft_assignment = np.exp(_mutual_log_assignment(unary_affinity))
        affinity = unary_affinity
        for round_index in range(self.config.num_layers):
            relation_vote = _relation_vote(
                soft_assignment,
                source_bases,
                target_bases,
                self.weights["relation_compatibility"],
            )
            strength = _softplus(self.weights["voting_strengths"][round_index])
            affinity = unary_affinity + strength * relation_vote
            soft_assignment = np.exp(_mutual_log_assignment(affinity))
        base_affinity = affinity
        semantic_sum = direct_evidence[
            ..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)
        ].sum(axis=-1, dtype=np.float32)
        direct_residual = (
            semantic_sum * self.weights["direct_evidence_head.weight"].reshape(-1)[0]
        )
        anchor_weights = np.exp(_mutual_log_assignment(base_affinity + direct_residual))
        pair_states = self._association_pair_states(
            source_modalities,
            target_modalities,
            (source_tokens != 0).any(axis=1).astype(np.float32),
            (target_tokens != 0).any(axis=1).astype(np.float32),
            source_visual_mask[:, 0],
            target_visual_mask[:, 0],
            anchor_weights,
        )
        compatibility = np_float(0.5) * (
            self.weights["relation_compatibility"]
            + self.weights["relation_compatibility"].T
        )
        for layer_index in range(self.config.association_layers):
            pair_states = self._association_layer(
                pair_states,
                anchor_weights,
                source_bases,
                target_bases,
                compatibility,
                layer_index,
            )
        association_residual = self._linear(
            pair_states,
            "association_score",
            bias=False,
        ).squeeze(-1)
        anchor_vote = _semantic_anchor_relation_vote(
            direct_evidence,
            source_bases,
            target_bases,
            compatibility,
        )
        anchor_vote_residual = _softplus(
            self.weights["anchor_voting_strength"]
        ) * anchor_vote
        context_score = _local_semantic_context_score(
            direct_evidence,
            source_bases,
            target_bases,
            source_numeric,
            target_numeric,
        )
        context_vote_residual = _softplus(
            self.weights["context_voting_strength"]
        ) * context_score
        affinity = (
            base_affinity
            + association_residual
            + direct_residual
            + anchor_vote_residual
            + context_vote_residual
        )
        assignment = _partial_assignment(
            affinity,
            source_numeric,
            target_numeric,
            all_nodes=self.config.candidate_policy == ALL_NODE_CANDIDATE_POLICY,
        )
        score_components = np.stack(
            (
                unary_affinity,
                base_affinity,
                association_residual,
                direct_residual,
                anchor_vote_residual,
                context_score,
                context_vote_residual,
                affinity,
            ),
            axis=-1,
        )
        standardized_scores = _standardize(score_components)
        consensus = _standardize(
            _relational_consensus_features(assignment, source_bases, target_bases)
        )
        geometry = _standardize(
            _relative_geometry_consensus_features(
                assignment,
                source_relations,
                target_relations,
            )
        )
        local_features = np.concatenate(
            (
                direct_evidence,
                np.tanh(score_components / np_float(5.0)),
                standardized_scores,
                consensus,
                geometry,
            ),
            axis=-1,
        )
        local_residual = self._layer_norm(local_features, "local_alignment_score.0")
        local_residual = _gelu(
            self._linear(local_residual, "local_alignment_score.1")
        )
        local_residual = self._linear(
            local_residual,
            "local_alignment_score.3",
            bias=False,
        ).squeeze(-1)
        final_affinity = affinity + local_residual
        final_assignment = _partial_assignment(
            final_affinity,
            source_numeric,
            target_numeric,
            all_nodes=self.config.candidate_policy == ALL_NODE_CANDIDATE_POLICY,
        )
        return {"logits_ab": final_assignment, "affinity": final_affinity}

    def _encode_nodes(
        self,
        token_ids: Any,
        numeric: Any,
        visual_patches: Any,
        visual_mask: Any,
    ) -> tuple[Any, Any]:
        np = _require_numpy()
        visual = visual_patches
        for index in (0, 2, 4):
            visual = _gelu(
                self._conv2d(visual, f"visual_encoder.{index}", stride=2)
            )
        visual = visual.mean(axis=(2, 3), dtype=np.float32)
        visual = (
            visual_mask * visual
            + (np_float(1.0) - visual_mask) * self.weights["missing_visual"][None]
        )
        text_mask = (token_ids != 0).any(axis=1, keepdims=True).astype(np.float32)
        present = np.broadcast_to(
            self.weights["present_text"],
            (len(token_ids), TEXT_DESCRIPTOR_DIM),
        )
        missing = np.broadcast_to(
            self.weights["missing_text"],
            (len(token_ids), TEXT_DESCRIPTOR_DIM),
        )
        text = text_mask * present + (np_float(1.0) - text_mask) * missing
        xml = self._layer_norm(numeric, "xml_projection.0")
        xml = _gelu(self._linear(xml, "xml_projection.1"))
        xml = self._linear(xml, "xml_projection.3")
        raw = np.concatenate((text, visual, xml), axis=-1)
        fused = self._layer_norm(raw, "descriptor_fusion.0")
        fused = _gelu(self._linear(fused, "descriptor_fusion.1"))
        fused = self._linear(fused, "descriptor_fusion.4")
        descriptor = self._layer_norm(raw + fused, "descriptor_norm")
        states = self._linear(descriptor, "descriptor_projection", bias=False)
        states = self._layer_norm(states, "input_norm")
        return states, raw

    def _unary(self, source: Any, target: Any) -> Any:
        np = _require_numpy()
        source_norm = source / np.maximum(
            np.linalg.norm(source, axis=-1, keepdims=True),
            np_float(1e-12),
        )
        target_norm = target / np.maximum(
            np.linalg.norm(target, axis=-1, keepdims=True),
            np_float(1e-12),
        )
        cosine = source_norm @ target_norm.T
        pair = np.concatenate(
            (
                np.abs(source[:, None, :] - target[None, :, :]),
                source[:, None, :] * target[None, :, :],
            ),
            axis=-1,
        )
        residual = self._layer_norm(pair, "unary_head.residual.0")
        residual = _gelu(self._linear(residual, "unary_head.residual.1"))
        residual = self._linear(residual, "unary_head.residual.4").squeeze(-1)
        scale = min(100.0, math.exp(float(self.weights["unary_head.logit_scale"])))
        return np_float(scale) * cosine + residual

    def _association_pair_states(
        self,
        source: Any,
        target: Any,
        source_text_available: Any,
        target_text_available: Any,
        source_visual_available: Any,
        target_visual_available: Any,
        anchor_weights: Any,
    ) -> Any:
        np = _require_numpy()
        splits = (TEXT_DESCRIPTOR_DIM, TEXT_DESCRIPTOR_DIM + VISUAL_DESCRIPTOR_DIM)
        source_text, source_visual, source_xml = np.split(source, splits, axis=-1)
        target_text, target_visual, target_xml = np.split(target, splits, axis=-1)
        text, text_both, text_one = _symmetric_pair_features(
            source_text,
            target_text,
            source_text_available,
            target_text_available,
        )
        visual, visual_both, visual_one = _symmetric_pair_features(
            source_visual,
            target_visual,
            source_visual_available,
            target_visual_available,
        )
        xml, _, _ = _symmetric_pair_features(
            source_xml,
            target_xml,
            np.ones_like(source_text_available),
            np.ones_like(target_text_available),
        )
        availability = np.stack(
            (text_both, text_one, visual_both, visual_one, anchor_weights),
            axis=-1,
        )
        pair = np.concatenate((text, visual, xml, availability), axis=-1)
        pair = self._layer_norm(pair, "association_pair_encoder.0")
        pair = _gelu(self._linear(pair, "association_pair_encoder.1"))
        return self._linear(pair, "association_pair_encoder.4")

    def _association_layer(
        self,
        pair_states: Any,
        anchor_weights: Any,
        source_bases: Any,
        target_bases: Any,
        compatibility: Any,
        layer_index: int,
    ) -> Any:
        np = _require_numpy()
        prefix = f"association_layers.{layer_index}"
        anchored = pair_states * anchor_weights[..., None]
        source_messages = np.einsum(
            "rik,kjd->rijd",
            source_bases,
            anchored,
            optimize=True,
        )
        typed = np.einsum(
            "rq,rijd->qijd",
            compatibility,
            source_messages,
            optimize=True,
        )
        messages = np.einsum(
            "qjl,qild->ijd",
            target_bases,
            typed,
            optimize=True,
        )
        projected = self._linear(
            messages,
            f"{prefix}.message_projection",
            bias=False,
        )
        pair_states = self._layer_norm(
            pair_states + projected,
            f"{prefix}.message_norm",
        )
        feed_forward = _gelu(
            self._linear(pair_states, f"{prefix}.feed_forward.0")
        )
        feed_forward = self._linear(
            feed_forward,
            f"{prefix}.feed_forward.3",
        )
        return self._layer_norm(
            pair_states + feed_forward,
            f"{prefix}.output_norm",
        )

    def _conv2d(self, values: Any, prefix: str, *, stride: int) -> Any:
        np = _require_numpy()
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights[f"{prefix}.bias"]
        kernel_height, kernel_width = weight.shape[-2:]
        padded = np.pad(
            values,
            (
                (0, 0),
                (0, 0),
                (kernel_height // 2, kernel_height // 2),
                (kernel_width // 2, kernel_width // 2),
            ),
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (kernel_height, kernel_width),
            axis=(2, 3),
        )[:, :, ::stride, ::stride]
        return (
            np.einsum("nchwkl,ockl->nohw", windows, weight, optimize=True)
            + bias[None, :, None, None]
        )

    def _linear(self, values: Any, prefix: str, *, bias: bool = True) -> Any:
        result = values @ self.weights[f"{prefix}.weight"].T
        if bias:
            result = result + self.weights[f"{prefix}.bias"]
        return result

    def _layer_norm(self, values: Any, prefix: str) -> Any:
        np = _require_numpy()
        mean = values.mean(axis=-1, keepdims=True, dtype=np.float32)
        variance = ((values - mean) ** 2).mean(
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        normalized = (values - mean) / np.sqrt(variance + np_float(1e-5))
        return (
            normalized * self.weights[f"{prefix}.weight"]
            + self.weights[f"{prefix}.bias"]
        )


def save_numpy_geometric_v9_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, Any],
    *,
    config: MatcherConfig,
) -> None:
    """Export the frozen v9 weights as a deterministic, pickle-free NPZ."""

    if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
        raise ValueError("only geometric-alignment v9 can use this exporter")
    np = _require_numpy()
    arrays: dict[str, Any] = {
        "__schema_version__": _encode(NUMPY_GEOMETRIC_V9_SCHEMA),
        "__config_json__": _encode(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        ),
    }
    for name, value in state_dict.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value, dtype=np.float32)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = ZipInfo(f"{name}.npy", date_time=(1981, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)


def _symmetric_pair_features(
    source: Any,
    target: Any,
    source_available: Any,
    target_available: Any,
) -> tuple[Any, Any, Any]:
    np = _require_numpy()
    both = source_available[:, None] * target_available[None, :]
    one = np.abs(source_available[:, None] - target_available[None, :])
    features = np.concatenate(
        (
            np.abs(source[:, None, :] - target[None, :, :]),
            source[:, None, :] * target[None, :, :],
        ),
        axis=-1,
    )
    return features * both[..., None], both, one


def _typed_relation_bases(relations: Any, numeric: Any) -> Any:
    np = _require_numpy()
    identity = np.clip(relations[..., 0], 0.0, 1.0)
    non_identity = np_float(1.0) - identity
    local = np.clip(relations[..., 17], 0.0, 1.0) * non_identity
    same_row = np.clip(relations[..., 6], 0.0, 1.0) * local
    same_column = np.clip(relations[..., 7], 0.0, 1.0) * local
    delta_x = relations[..., 9]
    delta_y = relations[..., 10]
    actionable = (numeric[:, :3].sum(axis=1) > 0.0) & (numeric[:, 3] > 0.0)
    context = ~actionable
    control_to_context = (
        actionable[:, None].astype(np.float32)
        * context[None, :].astype(np.float32)
        * local
    )
    context_to_control = (
        context[:, None].astype(np.float32)
        * actionable[None, :].astype(np.float32)
        * local
    )
    bases = np.stack(
        (
            identity,
            np.clip(relations[..., 1], 0.0, 1.0),
            np.clip(relations[..., 2], 0.0, 1.0),
            np.clip(relations[..., 3], 0.0, 1.0),
            np.clip(relations[..., 4], 0.0, 1.0),
            np.clip(relations[..., 5], 0.0, 1.0),
            same_row,
            same_column,
            np.clip(relations[..., 8], 0.0, 1.0) * non_identity,
            (delta_x < 0.0).astype(np.float32) * same_row,
            (delta_x > 0.0).astype(np.float32) * same_row,
            (delta_y < 0.0).astype(np.float32) * same_column,
            (delta_y > 0.0).astype(np.float32) * same_column,
            local,
            control_to_context,
            context_to_control,
        ),
        axis=0,
    )
    normalizer = bases.sum(axis=-1, keepdims=True, dtype=np.float32)
    return np.where(normalizer > 0.0, bases / np.maximum(normalizer, 1e-12), bases)


def _mutual_log_assignment(affinity: Any) -> Any:
    return np_float(0.5) * (
        _log_softmax(affinity, axis=1) + _log_softmax(affinity, axis=0)
    )


def _partial_assignment(
    affinity: Any,
    source_numeric: Any,
    target_numeric: Any,
    *,
    all_nodes: bool,
) -> Any:
    np = _require_numpy()
    if all_nodes:
        source_indices = np.arange(len(source_numeric))
        target_indices = np.arange(len(target_numeric))
    else:
        source_indices = np.flatnonzero(
            (source_numeric[:, :3].sum(axis=1) > 0.0)
            & (source_numeric[:, 3] > 0.0)
        )
        target_indices = np.flatnonzero(
            (target_numeric[:, :3].sum(axis=1) > 0.0)
            & (target_numeric[:, 3] > 0.0)
        )
    assignment = np.full_like(affinity, np_float(-1e4))
    if not len(source_indices) or not len(target_indices):
        return assignment
    selected = affinity[np.ix_(source_indices, target_indices)]
    assignment[np.ix_(source_indices, target_indices)] = _mutual_log_assignment(selected)
    return assignment


def _relation_vote(
    soft_assignment: Any,
    source_bases: Any,
    target_bases: Any,
    relation_compatibility: Any,
) -> Any:
    np = _require_numpy()
    compatibility = np_float(0.5) * (
        relation_compatibility + relation_compatibility.T
    )
    source_messages = np.einsum(
        "rik,kl->ril",
        source_bases,
        soft_assignment,
        optimize=True,
    )
    typed = np.einsum(
        "rq,ril->qil",
        compatibility,
        source_messages,
        optimize=True,
    )
    vote = np.einsum("qil,qjl->ij", typed, target_bases, optimize=True)
    centered = vote - vote.mean(dtype=np.float32)
    rms = np.sqrt(np.maximum((centered**2).mean(dtype=np.float32), np.finfo(np.float32).eps))
    return centered / rms


def _semantic_anchor_relation_vote(
    evidence: Any,
    source_bases: Any,
    target_bases: Any,
    compatibility: Any,
) -> Any:
    np = _require_numpy()
    semantic = evidence[..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)].sum(axis=-1)
    source_best_targets = semantic.argmax(axis=1)
    source_best_scores = semantic[np.arange(semantic.shape[0]), source_best_targets]
    target_best_sources = semantic.argmax(axis=0)
    target_best_scores = semantic[target_best_sources, np.arange(semantic.shape[1])]
    source_margins = source_best_scores.copy()
    target_margins = target_best_scores.copy()
    if semantic.shape[1] > 1:
        source_top = np.partition(semantic, -2, axis=1)[:, -2:]
        source_margins = source_top.max(axis=1) - source_top.min(axis=1)
    if semantic.shape[0] > 1:
        target_top = np.partition(semantic, -2, axis=0)[-2:]
        target_margins = target_top.max(axis=0) - target_top.min(axis=0)
    source_indices = np.arange(semantic.shape[0])
    retained = (
        (target_best_sources[source_best_targets] == source_indices)
        & (source_best_scores >= 1.0)
        & (source_margins >= 0.5)
        & (target_margins[source_best_targets] >= 0.5)
    )
    source_anchors = source_indices[retained]
    if not len(source_anchors):
        return np.zeros_like(semantic, dtype=np.float32)
    target_anchors = source_best_targets[source_anchors]
    source_binary = (source_bases > 0).astype(np.float32)
    target_binary = (target_bases > 0).astype(np.float32)
    source_out = source_binary[:, :, source_anchors]
    target_out = target_binary[:, :, target_anchors]
    source_in = source_binary[:, source_anchors, :].transpose(0, 2, 1)
    target_in = target_binary[:, target_anchors, :].transpose(0, 2, 1)
    vote = np.einsum(
        "ria,rq,qja->ij", source_out, compatibility, target_out, optimize=True
    ) + np.einsum(
        "ria,rq,qja->ij", source_in, compatibility, target_in, optimize=True
    )
    support = (
        np.einsum("ria,qja->ija", source_out, target_out, optimize=True) > 0
    ).sum(axis=-1) + (
        np.einsum("ria,qja->ija", source_in, target_in, optimize=True) > 0
    ).sum(axis=-1)
    return vote / np.maximum(support.astype(np.float32), np_float(1.0))


def _local_semantic_context_score(
    evidence: Any,
    source_bases: Any,
    target_bases: Any,
    source_numeric: Any,
    target_numeric: Any,
) -> Any:
    np = _require_numpy()
    semantic = evidence[..., : len(DIRECT_SEMANTIC_EVIDENCE_NAMES)].sum(axis=-1)
    hierarchy = [TYPED_RELATION_NAMES.index(name) for name in (
        "parent", "child", "sibling", "ancestor", "descendant"
    )]
    source_actionable = (
        (source_numeric[:, :3].sum(axis=1) > 0.0) & (source_numeric[:, 3] > 0.0)
    )
    target_actionable = (
        (target_numeric[:, :3].sum(axis=1) > 0.0) & (target_numeric[:, 3] > 0.0)
    )
    source_links = (source_bases[hierarchy] > 0).any(axis=0)
    target_links = (target_bases[hierarchy] > 0).any(axis=0)
    source_links = (source_links | source_links.T) & (~source_actionable)[None, :]
    target_links = (target_links | target_links.T) & (~target_actionable)[None, :]
    source_to_target = np.where(
        target_links[None, :, :], semantic[:, None, :], np_float(0.0)
    ).max(axis=2)
    source_context_to_target = np.where(
        source_links[:, :, None], semantic[None, :, :], np_float(0.0)
    ).max(axis=1)
    context_to_context = np.where(
        target_links[None, :, :],
        source_context_to_target[:, None, :],
        np_float(0.0),
    ).max(axis=2)
    return np.maximum(np.maximum(source_to_target, source_context_to_target), context_to_context)


def _relational_consensus_features(
    assignment: Any,
    source_bases: Any,
    target_bases: Any,
) -> Any:
    np = _require_numpy()
    anchors = np.exp(assignment)
    source = source_bases.copy()
    target = target_bases.copy()
    source[0] = 0.0
    target[0] = 0.0
    outgoing = np.einsum(
        "ria,ab,qjb->ijrq", source, anchors, target, optimize=True
    )
    incoming = np.einsum(
        "rai,ab,qbj->ijrq", source, anchors, target, optimize=True
    )
    return (outgoing + incoming).reshape(
        assignment.shape[0], assignment.shape[1], len(TYPED_RELATION_NAMES) ** 2
    )


def _relative_geometry_consensus_features(
    assignment: Any,
    source_relations: Any,
    target_relations: Any,
) -> Any:
    np = _require_numpy()
    indices = list(ALIGNMENT_RELATION_FEATURE_INDICES)
    source = source_relations[..., indices].copy()
    target = target_relations[..., indices].copy()
    source[np.arange(len(source)), np.arange(len(source))] = 0.0
    target[np.arange(len(target)), np.arange(len(target))] = 0.0
    anchors = np.exp(assignment)
    outgoing = np.einsum("iak,ab,jbk->ijk", source, anchors, target, optimize=True)
    incoming = np.einsum("aik,ab,bjk->ijk", source, anchors, target, optimize=True)
    return outgoing + incoming


def _standardize(values: Any) -> Any:
    np = _require_numpy()
    centered = values - values.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
    rms = np.sqrt(
        np.maximum(
            (centered**2).mean(axis=(0, 1), keepdims=True, dtype=np.float32),
            np.finfo(np.float32).eps,
        )
    )
    return centered / rms


def _softplus(value: Any) -> Any:
    np = _require_numpy()
    return np.logaddexp(np_float(0.0), value)
