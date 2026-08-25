"""NumPy inference for the unified OmniTransfer association model."""

from __future__ import annotations

import base64
import json
import math
from io import BytesIO
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import zlib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from omnitransfer.learned_matcher import (
    ALIGNMENT_RELATION_FEATURE_INDICES,
    DETERMINISTIC_ICON_VISUAL_ENCODER,
    DIRECT_PAIR_EVIDENCE_NAMES,
    GEOMETRIC_FEATURE_SCHEMA_ID,
    LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID,
    LearnedMatch,
    LEGACY_GLOBAL_POOL_VISUAL_ENCODER,
    MatcherConfig,
    MULTISCALE_HASH_VISUAL_ENCODER,
    OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE,
    TEXT_DESCRIPTOR_DIM,
    TYPED_RELATION_NAMES,
    VISUAL_DESCRIPTOR_DIM,
    encode_graph,
    SPATIAL_CNN_VISUAL_ENCODER,
)
from omnitransfer.unified_alignment import (
    ALIGNMENT_STRATEGY_SCHEMA_ID,
    UNIFIED_ASSOCIATION_SCHEMA_ID,
)
from omnitransfer.ui_graph import (
    UIGraph,
    local_context_graph,
    multi_anchor_context_graph,
    visual_bbox_fraction,
)
from omnitransfer.visual_descriptor import (
    deterministic_icon_descriptor_numpy,
    multiscale_hash_descriptor_numpy,
)

NUMPY_UNIFIED_ASSOCIATION_SCHEMA = "omnitransfer_numpy_unified_association_v1"


class NumpyGeometricAlignmentMatcher:
    """Inference-only unified association matcher for the runtime."""

    feature_schema_id = LEGACY_GEOMETRIC_FEATURE_SCHEMA_ID
    backend = "numpy-unified-association-v1"

    def __init__(self, weights: Mapping[str, Any], *, config: MatcherConfig) -> None:
        if config.architecture != OMNITRANSFER_GEOMETRIC_ALIGNMENT_ARCHITECTURE:
            raise ValueError("NumPy v9 requires geometric-alignment architecture")
        self.weights = dict(weights)
        self.config = config

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NumpyGeometricAlignmentMatcher":
        np = _require_numpy()
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            if _decode(checkpoint["__schema_version__"]) != NUMPY_UNIFIED_ASSOCIATION_SCHEMA:
                raise ValueError(
                    "checkpoint is not an OmniTransfer unified association matcher"
                )
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
        if not any(node.node_id == source_node_id for node in source.nodes):
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        source_context = local_context_graph(
            source,
            anchor_node_id=source_node_id,
            max_nodes=self.config.source_context_nodes,
        )
        allowed = set(
            candidate_node_ids or (node.node_id for node in target.nodes)
        ) & {node.node_id for node in target.nodes}
        if not allowed:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        target_context = multi_anchor_context_graph(
            target,
            anchor_node_ids=allowed,
            max_nodes=max(self.config.target_context_nodes, len(allowed)),
        )
        source_index = next(
            index
            for index, node in enumerate(source_context.nodes)
            if node.node_id == source_node_id
        )
        candidate_indices = self._candidate_indices(target_context, allowed)
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        return self._predict_from_output(
            source_context,
            target_context,
            source_index=source_index,
            candidate_indices=candidate_indices,
            output=self._forward(source_context, target_context),
            min_probability=min_probability,
            min_margin=min_margin,
        )

    def predict_many(
        self,
        source: UIGraph,
        target: UIGraph,
        *,
        source_node_ids: Iterable[str],
        candidate_node_ids: Iterable[str] | None = None,
        min_probability: float = 0.0,
        min_margin: float = 0.0,
    ) -> dict[str, LearnedMatch]:
        """Predict several source nodes while encoding one page pair once."""

        requested_source_ids = tuple(dict.fromkeys(source_node_ids))
        valid_source_ids = tuple(
            node_id
            for node_id in requested_source_ids
            if any(node.node_id == node_id for node in source.nodes)
        )
        allowed = set(
            candidate_node_ids or (node.node_id for node in target.nodes)
        ) & {node.node_id for node in target.nodes}
        if not allowed:
            return {
                source_node_id: LearnedMatch(
                    None, 0.0, 0.0, "target_candidates_missing", ()
                )
                for source_node_id in requested_source_ids
            }
        if not valid_source_ids:
            return {
                source_node_id: LearnedMatch(
                    None, 0.0, 0.0, "source_node_missing", ()
                )
                for source_node_id in requested_source_ids
            }
        source_context = multi_anchor_context_graph(
            source,
            anchor_node_ids=valid_source_ids,
            max_nodes=max(self.config.source_context_nodes, len(valid_source_ids)),
        )
        target_context = multi_anchor_context_graph(
            target,
            anchor_node_ids=allowed,
            max_nodes=max(self.config.target_context_nodes, len(allowed)),
        )
        source_indices = {
            node.node_id: index for index, node in enumerate(source_context.nodes)
        }
        candidate_indices = self._candidate_indices(target_context, allowed)
        if not candidate_indices:
            return {
                source_node_id: LearnedMatch(
                    None, 0.0, 0.0, "target_candidates_missing", ()
                )
                for source_node_id in requested_source_ids
            }
        output = self._forward(source_context, target_context)
        predictions = {}
        for source_node_id in requested_source_ids:
            source_index = source_indices.get(source_node_id)
            if source_index is None:
                predictions[source_node_id] = LearnedMatch(
                    None, 0.0, 0.0, "source_node_missing", ()
                )
                continue
            predictions[source_node_id] = self._predict_from_output(
                source_context,
                target_context,
                source_index=source_index,
                candidate_indices=candidate_indices,
                output=output,
                min_probability=min_probability,
                min_margin=min_margin,
            )
        return predictions

    def page_embedding(self, page: UIGraph) -> Any:
        """Return the normalized contextual configuration readout for one page."""

        if not page.nodes:
            raise ValueError("page embedding requires at least one node")
        return self._forward(page, page)["source_config_embedding"].copy()

    @staticmethod
    def _candidate_indices(
        target: UIGraph,
        candidate_node_ids: Iterable[str] | None,
    ) -> list[int]:
        allowed = set(candidate_node_ids or (node.node_id for node in target.nodes))
        return [
            index
            for index, node in enumerate(target.nodes)
            if node.node_id in allowed
        ]

    def _predict_from_output(
        self,
        source: UIGraph,
        target: UIGraph,
        *,
        source_index: int,
        candidate_indices: list[int],
        output: dict[str, Any],
        min_probability: float,
        min_margin: float,
    ) -> LearnedMatch:
        assignment_layers = tuple(output.get("assignment_scores_by_layer") or ())
        affinity_layers = tuple(output.get("affinities_by_layer") or ())
        selected_layer = len(assignment_layers) - 1
        selected_logits = output["logits_ab"][source_index][candidate_indices].copy()
        if len(assignment_layers) >= 2 and len(candidate_indices) >= 2:
            previous_logits = assignment_layers[-2][source_index][candidate_indices]
            previous_probability = _softmax(previous_logits, axis=0)
            final_probability = _softmax(selected_logits, axis=0)
            previous_top = sorted(previous_probability, reverse=True)[:2]
            final_top = sorted(final_probability, reverse=True)[:2]
            if previous_top[0] - previous_top[1] > final_top[0] - final_top[1]:
                selected_layer = len(assignment_layers) - 2
                selected_logits = previous_logits.copy()
        selected_affinity_matrix = (
            affinity_layers[selected_layer] if affinity_layers else output["affinity"]
        )
        selected_affinity = selected_affinity_matrix[source_index][candidate_indices]
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
        direct_evidence = np.zeros(
            (
                len(source.nodes),
                len(target.nodes),
                len(DIRECT_PAIR_EVIDENCE_NAMES),
            ),
            dtype=np.float32,
        )
        source_visual, source_visual_mask = _visual_inputs(
            source,
            patch_size=self.config.visual_patch_size,
            canvas_size=self.config.visual_canvas_size,
            visual_encoder=self.config.visual_encoder,
            context_scale=self.config.visual_context_scale,
        )
        target_visual, target_visual_mask = _visual_inputs(
            target,
            patch_size=self.config.visual_patch_size,
            canvas_size=self.config.visual_canvas_size,
            visual_encoder=self.config.visual_encoder,
            context_scale=self.config.visual_context_scale,
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
        source_bases = _typed_relation_bases(source_relations, source_numeric)
        target_bases = _typed_relation_bases(target_relations, target_numeric)
        unary_affinity, _, _ = self._affinity(source_states, target_states)
        assignments = []
        affinities = []
        for layer_index in range(self.config.association_layers):
            source_states, target_states = self._contextual_layer(
                source_states,
                target_states,
                source_bases,
                target_bases,
                layer_index,
            )
            layer_affinity, _, _ = self._affinity(source_states, target_states)
            affinities.append(layer_affinity)
            assignments.append(_mutual_log_assignment(layer_affinity))
        association_score, source_matchability, target_matchability = self._affinity(
            source_states, target_states
        )
        final_assignment = _mutual_log_assignment(association_score)
        return {
            "logits_ab": final_assignment,
            "affinity": association_score,
            "association_score": association_score,
            "direct_pair_evidence": direct_evidence,
            "unary_affinity": unary_affinity,
            "assignment_scores_by_layer": tuple(assignments),
            "affinities_by_layer": tuple(affinities),
            "source_matchability": source_matchability,
            "target_matchability": target_matchability,
            "source_config_embedding": self._page_embedding(source_states),
            "target_config_embedding": self._page_embedding(target_states),
            "alignment_strategy_schema": ALIGNMENT_STRATEGY_SCHEMA_ID,
            "association_schema": UNIFIED_ASSOCIATION_SCHEMA_ID,
        }

    def _encode_nodes(
        self,
        token_ids: Any,
        numeric: Any,
        visual_patches: Any,
        visual_mask: Any,
    ) -> tuple[Any, Any]:
        np = _require_numpy()
        visual = visual_patches
        if self.config.visual_encoder == LEGACY_GLOBAL_POOL_VISUAL_ENCODER:
            for index in (0, 2, 4):
                visual = _gelu(
                    self._conv2d(visual, f"visual_encoder.{index}", stride=2)
                )
            visual = visual.mean(axis=(2, 3), dtype=np.float32)
        elif self.config.visual_encoder == SPATIAL_CNN_VISUAL_ENCODER:
            for index in (0, 2, 4, 6):
                visual = _gelu(
                    self._conv2d(visual, f"visual_encoder.{index}", stride=1 if index in (0, 6) else 2)
                )
            visual = visual.reshape(visual.shape[0], -1)
            visual = self._linear(visual, "visual_encoder.9")
        elif self.config.visual_encoder == DETERMINISTIC_ICON_VISUAL_ENCODER:
            visual = deterministic_icon_descriptor_numpy(visual)
        elif self.config.visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER:
            visual = multiscale_hash_descriptor_numpy(visual)
        else:
            raise ValueError(f"unsupported visual encoder: {self.config.visual_encoder}")
        visual = (
            visual_mask * visual
            + (np_float(1.0) - visual_mask) * self.weights["missing_visual"][None]
        )
        text_mask = (token_ids != 0).any(axis=1, keepdims=True).astype(np.float32)
        missing = np.broadcast_to(
            self.weights["missing_text"],
            (len(token_ids), TEXT_DESCRIPTOR_DIM),
        )
        if self.config.text_encoder == "learned_token_lookup":
            token_vectors = self.weights["token_embedding.weight"][token_ids]
            token_mask = (token_ids != 0).astype(np.float32)[..., None]
            pooled = (token_vectors * token_mask).sum(axis=1)
            pooled /= np.maximum(token_mask.sum(axis=1), 1.0)
            text = self._linear(pooled, "text_projection")
        elif self.config.text_encoder == "direct_text_evidence":
            present = np.broadcast_to(
                self.weights["present_text"],
                (len(token_ids), TEXT_DESCRIPTOR_DIM),
            )
            text = present
        else:
            raise ValueError(f"unsupported text encoder: {self.config.text_encoder}")
        text = text_mask * text + (np_float(1.0) - text_mask) * missing
        xml = self._layer_norm(numeric, "xml_projection.0")
        xml = _gelu(self._linear(xml, "xml_projection.1"))
        xml = self._linear(xml, "xml_projection.3")
        raw = np.concatenate((text, visual, xml), axis=-1)
        modalities = np.stack(
            (
                self._linear(text, "text_to_hidden"),
                self._linear(visual, "visual_to_hidden"),
                self._linear(xml, "xml_to_hidden"),
            ),
            axis=1,
        ) + self.weights["modality_type"][None]
        available = np.concatenate(
            (text_mask, visual_mask.astype(np.float32), np.ones_like(text_mask)),
            axis=1,
        )
        modality_logits = self._layer_norm(modalities, "modality_score.0")
        modality_logits = self._linear(
            modality_logits, "modality_score.1", bias=False
        ).squeeze(-1)
        modality_logits = np.where(available > 0.0, modality_logits, np_float(-1e4))
        modality_weights = _softmax(modality_logits / np_float(0.25), axis=1)
        fused = (modality_weights[..., None] * modalities).sum(
            axis=1, dtype=np.float32
        )
        states = self._layer_norm(fused, "input_norm")
        feed_forward = _gelu(
            self._linear(states, "input_feed_forward.0")
        )
        feed_forward = self._linear(feed_forward, "input_feed_forward.3")
        states = self._layer_norm(
            states + feed_forward, "input_output_norm"
        )
        return states, raw

    def _affinity(self, source: Any, target: Any) -> tuple[Any, Any, Any]:
        np = _require_numpy()
        source_norm = source / np.maximum(
            np.linalg.norm(source, axis=-1, keepdims=True),
            np_float(1e-12),
        )
        target_norm = target / np.maximum(
            np.linalg.norm(target, axis=-1, keepdims=True),
            np_float(1e-12),
        )
        source_matchability = self._linear(
            self._layer_norm(source, "matchability_head.0"),
            "matchability_head.1",
        ).squeeze(-1)
        target_matchability = self._linear(
            self._layer_norm(target, "matchability_head.0"),
            "matchability_head.1",
        ).squeeze(-1)
        scale = min(100.0, math.exp(float(self.weights["logit_scale"])))
        affinity = (
            np_float(scale) * (source_norm @ target_norm.T)
            + np_float(0.5)
            * (source_matchability[:, None] + target_matchability[None, :])
        )
        return affinity, source_matchability, target_matchability

    def _local_graph_layer(
        self,
        states: Any,
        relation_bases: Any,
        *,
        prefix: str,
    ) -> Any:
        np = _require_numpy()
        compatibility = _softmax(
            np_float(5.0) * self.weights["relation_compatibility"],
            axis=-1,
        )
        mixed_bases = np.einsum(
            "rq,qij->rij", compatibility, relation_bases, optimize=True
        )
        values = self._linear(states, f"{prefix}.value", bias=False)
        relation_messages = np.einsum(
            "rij,jd->rid", mixed_bases, values, optimize=True
        )
        messages = relation_messages.transpose(1, 0, 2).reshape(
            len(states), -1
        )
        projected = self._linear(
            messages, f"{prefix}.message_projection", bias=False
        )
        states = self._layer_norm(
            states + projected, f"{prefix}.message_norm"
        )
        feed_forward = _gelu(
            self._linear(states, f"{prefix}.feed_forward.0")
        )
        feed_forward = self._linear(
            feed_forward, f"{prefix}.feed_forward.3"
        )
        return self._layer_norm(
            states + feed_forward, f"{prefix}.output_norm"
        )

    def _cross_context(
        self,
        query_states: Any,
        key_states: Any,
        *,
        prefix: str,
    ) -> Any:
        query = self._linear(query_states, f"{prefix}.query", bias=False)
        key = self._linear(key_states, f"{prefix}.key", bias=False)
        scores = (query @ key.T) / np_float(math.sqrt(self.config.hidden_dim))
        values = self._linear(key_states, f"{prefix}.value", bias=False)
        return _softmax(scores, axis=-1) @ values

    def _cross_update(self, states: Any, context: Any, *, prefix: str) -> Any:
        evidence = _require_numpy().concatenate(
            (states, context, abs(states - context), states * context), axis=-1
        )
        update = self._layer_norm(evidence, f"{prefix}.cross_update.0")
        update = _gelu(self._linear(update, f"{prefix}.cross_update.1"))
        update = self._linear(update, f"{prefix}.cross_update.4")
        return self._layer_norm(states + update, f"{prefix}.cross_norm")

    def _contextual_layer(
        self,
        source: Any,
        target: Any,
        source_bases: Any,
        target_bases: Any,
        layer_index: int,
    ) -> tuple[Any, Any]:
        prefix = f"association_layers.{layer_index}"
        source_local = self._local_graph_layer(
            source, source_bases, prefix=f"{prefix}.local"
        )
        target_local = self._local_graph_layer(
            target, target_bases, prefix=f"{prefix}.local"
        )
        source_context = self._cross_context(
            source_local, target_local, prefix=prefix
        )
        target_context = self._cross_context(
            target_local, source_local, prefix=prefix
        )
        return (
            self._cross_update(source_local, source_context, prefix=prefix),
            self._cross_update(target_local, target_context, prefix=prefix),
        )

    def _page_embedding(self, states: Any) -> Any:
        np = _require_numpy()
        logits = self._linear(
            self._layer_norm(states, "page_attention.0"),
            "page_attention.1",
            bias=False,
        ).squeeze(-1)
        embedding = (_softmax(logits, axis=0)[:, None] * states).sum(
            axis=0, dtype=np.float32
        )
        return embedding / np.maximum(
            np.linalg.norm(embedding), np_float(1e-12)
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


def save_numpy_unified_association_checkpoint(
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
        "__schema_version__": _encode(NUMPY_UNIFIED_ASSOCIATION_SCHEMA),
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
    kinship_proximity = (np_float(1.0) - np.clip(relations[..., 16], 0.0, 1.0)) * local
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
            kinship_proximity,
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

def _visual_inputs(
    graph: UIGraph,
    *,
    patch_size: int,
    canvas_size: int = 0,
    visual_encoder: str = DETERMINISTIC_ICON_VISUAL_ENCODER,
    context_scale: float = 3.0,
) -> tuple[Any, Any]:
    np = _require_numpy()
    channels = 6 if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER else 3
    patches = np.zeros(
        (len(graph.nodes), channels, patch_size, patch_size),
        dtype=np.float32,
    )
    mask = np.zeros((len(graph.nodes), 1), dtype=np.float32)
    image = _graph_rgb(graph, canvas_size=canvas_size)
    if image is None:
        return patches, mask
    screenshot_path = str(graph.metadata.get("screenshot_path") or "")
    screenshot_signature = None
    if screenshot_path and not isinstance(graph.metadata.get("visual_rgb"), Mapping):
        try:
            stat = Path(screenshot_path).stat()
            screenshot_signature = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            screenshot_signature = None
    visual_height, visual_width = image.shape[:2]
    for index, node in enumerate(graph.nodes):
        if bool(node.metadata.get("visual_disabled")):
            continue
        bbox = visual_bbox_fraction(
            graph,
            node,
            image_width=visual_width,
            image_height=visual_height,
        )
        if bbox is None:
            continue
        context_bbox = _expanded_bbox_fraction(bbox, context_scale)
        if screenshot_signature is not None:
            patch = _cached_screenshot_patch(
                screenshot_path,
                int(canvas_size),
                screenshot_signature[0],
                screenshot_signature[1],
                tuple(float(value) for value in bbox),
                int(patch_size),
            )
            context_patch = (
                _cached_screenshot_patch(
                    screenshot_path,
                    int(canvas_size),
                    screenshot_signature[0],
                    screenshot_signature[1],
                    tuple(float(value) for value in context_bbox),
                    int(patch_size),
                )
                if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER
                else None
            )
        else:
            patch = _sample_rgb_bbox(image, bbox=bbox, size=patch_size)
            context_patch = (
                _sample_rgb_bbox(image, bbox=context_bbox, size=patch_size)
                if visual_encoder == MULTISCALE_HASH_VISUAL_ENCODER
                else None
            )
        patches[index, :3] = patch.transpose(2, 0, 1)
        if context_patch is not None:
            patches[index, 3:] = context_patch.transpose(2, 0, 1)
        mask[index, 0] = np_float(1.0)
    return patches, mask


def _expanded_bbox_fraction(
    bbox: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    scale = max(1.0, float(scale))
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    half_width = (right - left) * scale * 0.5
    half_height = (bottom - top) * scale * 0.5
    return (
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(1.0, center_x + half_width),
        min(1.0, center_y + half_height),
    )


def _graph_rgb(graph: UIGraph, *, canvas_size: int) -> Any | None:
    np = _require_numpy()
    payload = graph.metadata.get("visual_rgb")
    if isinstance(payload, Mapping):
        try:
            width = int(payload["width"])
            height = int(payload["height"])
            raw = base64.b64decode(str(payload["data_base64"]), validate=True)
            if str(payload.get("compression") or "") == "zlib":
                raw = zlib.decompress(raw)
            image = np.frombuffer(raw, dtype=np.uint8)
            if image.size != width * height * 3:
                return None
            image = image.reshape(height, width, 3)
        except (KeyError, TypeError, ValueError, zlib.error):
            return None
    else:
        screenshot_path = str(graph.metadata.get("screenshot_path") or "")
        if not screenshot_path:
            return None
        try:
            stat = Path(screenshot_path).stat()
            return _cached_screenshot_rgb(
                screenshot_path,
                int(canvas_size),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except (OSError, ValueError):
            return None
    if canvas_size <= 0 or max(image.shape[:2]) <= canvas_size:
        return image
    try:
        from PIL import Image
    except ImportError:
        return None
    scale = canvas_size / max(image.shape[:2])
    resized = Image.fromarray(image).resize(
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        resample=Image.Resampling.BILINEAR,
    )
    return np.array(resized, dtype=np.uint8, copy=True)


@lru_cache(maxsize=32)
def _cached_screenshot_rgb(
    screenshot_path: str,
    canvas_size: int,
    modified_ns: int,
    file_size: int,
) -> Any:
    """Decode immutable observation screenshots once per runtime process."""

    del modified_ns, file_size
    np = _require_numpy()
    try:
        from PIL import Image

        with Image.open(screenshot_path) as loaded:
            image = np.array(loaded.convert("RGB"), dtype=np.uint8, copy=True)
        if canvas_size > 0 and max(image.shape[:2]) > canvas_size:
            scale = canvas_size / max(image.shape[:2])
            resized = Image.fromarray(image).resize(
                (
                    max(1, round(image.shape[1] * scale)),
                    max(1, round(image.shape[0] * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
            image = np.array(resized, dtype=np.uint8, copy=True)
        image.setflags(write=False)
        return image
    except (ImportError, OSError, ValueError):
        return None


def _sample_rgb_bbox(
    image: Any,
    *,
    bbox: tuple[float, float, float, float],
    size: int,
) -> Any:
    np = _require_numpy()
    source_height, source_width = image.shape[:2]
    left, top, right, bottom = bbox
    ys = np.linspace(
        top * max(0, source_height - 1),
        bottom * max(0, source_height - 1),
        size,
        dtype=np.float32,
    )
    xs = np.linspace(
        left * max(0, source_width - 1),
        right * max(0, source_width - 1),
        size,
        dtype=np.float32,
    )
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = (
        image[y0[:, None], x0[None, :]].astype(np.float32) * (np_float(1.0) - wx)
        + image[y0[:, None], x1[None, :]].astype(np.float32) * wx
    )
    bottom = (
        image[y1[:, None], x0[None, :]].astype(np.float32) * (np_float(1.0) - wx)
        + image[y1[:, None], x1[None, :]].astype(np.float32) * wx
    )
    sampled = top * (np_float(1.0) - wy) + bottom * wy
    return sampled.astype(np.float32, copy=False) / np_float(255.0)


@lru_cache(maxsize=4096)
def _cached_screenshot_patch(
    screenshot_path: str,
    canvas_size: int,
    modified_ns: int,
    file_size: int,
    bbox: tuple[float, float, float, float],
    size: int,
) -> Any:
    image = _cached_screenshot_rgb(
        screenshot_path,
        canvas_size,
        modified_ns,
        file_size,
    )
    if image is None:
        return _require_numpy().zeros((size, size, 3), dtype=_require_numpy().float32)
    patch = _sample_rgb_bbox(image, bbox=bbox, size=size)
    patch.setflags(write=False)
    return patch


def _gelu(values: Any) -> Any:
    np = _require_numpy()
    scaled = values * np_float(1.0 / math.sqrt(2.0))
    sign = np.sign(scaled)
    absolute = np.abs(scaled)
    factor = np_float(1.0) / (np_float(1.0) + np_float(0.3275911) * absolute)
    polynomial = (
        ((((np_float(1.061405429) * factor - np_float(1.453152027)) * factor
           + np_float(1.421413741)) * factor - np_float(0.284496736)) * factor
         + np_float(0.254829592)) * factor
    )
    error_function = sign * (
        np_float(1.0) - polynomial * np.exp(-(absolute * absolute))
    )
    return np_float(0.5) * values * (np_float(1.0) + error_function)


def _softmax(values: Any, *, axis: int) -> Any:
    np = _require_numpy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _log_softmax(values: Any, *, axis: int) -> Any:
    np = _require_numpy()
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _sigmoid(values: Any) -> Any:
    np = _require_numpy()
    return np_float(1.0) / (np_float(1.0) + np.exp(-values))


def _encode(value: str) -> Any:
    np = _require_numpy()
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode(value: Any) -> str:
    np = _require_numpy()
    return value.astype(np.uint8, copy=False).tobytes().decode("utf-8")


def np_float(value: float) -> Any:
    return _require_numpy().float32(value)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for OmniTransfer runtime") from exc
    return np
