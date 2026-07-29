"""Training and evaluation utilities for the learned widget matcher."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, Mapping

from omnitransfer.eval import RankingMetrics, ranking_metrics
from omnitransfer.learned_matcher import (
    MatcherConfig,
    RCAM_FEATURE_SCHEMA_ID,
    build_relation_aware_matcher,
    matcher_inputs,
    prepare_visual_asset,
)
from omnitransfer.outcome_learning import OutcomePreference, preference_ranking_loss
from omnitransfer.schema import Candidate, Prediction, Query
from omnitransfer.self_supervised import (
    AugmentConfig,
    CorrespondencePair,
    make_training_pair,
    matching_loss,
)
from omnitransfer.ui_graph import (
    UIGraph,
    UINode,
    graph_from_record,
    local_context_graph,
    multi_anchor_context_graph,
)


@dataclass(frozen=True)
class FineTuneResult:
    """Learned matcher plus its supervised optimization history."""

    model: Any
    history: tuple[dict[str, float], ...]
    train_queries: int
    train_correspondence_pairs: int


@dataclass(frozen=True)
class LatencySummary:
    """Compute latency gate plus separately reported image preparation."""

    samples: int
    budget_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    under_budget_rate: float
    budget_met: bool
    p95_graph_ms: float
    p95_input_ms: float
    p95_model_ms: float
    p95_postprocess_ms: float
    p50_asset_prepare_ms: float
    p95_asset_prepare_ms: float
    max_asset_prepare_ms: float


def split_queries_by_group(
    queries: Iterable[Query],
    *,
    seed: int = 17,
    train_percent: int = 70,
    dev_percent: int = 15,
) -> dict[str, list[Query]]:
    """Create deterministic app-disjoint splits when app metadata is available."""

    if train_percent <= 0 or dev_percent < 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave non-empty train and test ranges")
    splits = {"train": [], "dev": [], "test": []}
    for query in queries:
        group = _query_group(query)
        digest = hashlib.blake2b(
            f"{seed}:{group}".encode("utf-8"),
            digest_size=8,
        ).digest()
        bucket = int.from_bytes(digest, "big") % 100
        if bucket < train_percent:
            split = "train"
        elif bucket < train_percent + dev_percent:
            split = "dev"
        else:
            split = "test"
        splits[split].append(query)
    return splits


def partition_queries_by_declared_split(
    queries: Iterable[Query],
) -> dict[str, list[Query]]:
    """Use benchmark-authored train/dev/test labels without re-partitioning."""

    splits: dict[str, list[Query]] = {"train": [], "dev": [], "test": []}
    for query in queries:
        split = str(query.metadata.get("split") or "").strip().lower()
        if split not in splits:
            raise ValueError(
                f"query {query.query_id} has no supported declared split: {split!r}"
            )
        splits[split].append(query)
    empty = [name for name, values in splits.items() if not values]
    if empty:
        raise ValueError(f"declared benchmark split is empty: {', '.join(empty)}")
    return splits


def query_graphs(
    query: Query,
    *,
    max_source_nodes: int = 48,
    max_target_nodes: int = 64,
) -> tuple[UIGraph, UIGraph, str]:
    """Convert one canonical candidate-ranking query to learned graph inputs."""

    source_graph, source_node_id = _source_graph(query)
    source_graph = local_context_graph(
        source_graph,
        anchor_node_id=source_node_id,
        max_nodes=max_source_nodes,
    )
    target_graph = _target_graph(
        query,
        graph_id=f"{query.query_id}:target",
        screenshot_path=str(query.metadata.get("target_screenshot_path") or ""),
    )
    target_graph = multi_anchor_context_graph(
        target_graph,
        anchor_node_ids=(candidate.candidate_id for candidate in query.target_candidates),
        max_nodes=max(max_target_nodes, len(query.target_candidates)),
    )
    return source_graph, target_graph, source_node_id


def fine_tune_matcher(
    queries: list[Query],
    *,
    model: Any | None = None,
    config: MatcherConfig | None = None,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 17,
    device: str = "cpu",
    outcome_preferences: Mapping[str, tuple[OutcomePreference, ...]] | None = None,
    preference_weight: float = 0.2,
    self_supervised_graphs: list[UIGraph] | None = None,
    self_supervised_weight: float = 0.0,
    self_supervised_interval: int = 4,
    augment_config: AugmentConfig | None = None,
    correspondence_pairs: list[CorrespondencePair] | None = None,
    correspondence_weight: float = 1.0,
    confidence_weight: float = 1.0,
    context_mask_probability: float = 0.0,
    feature_schema_id: str = RCAM_FEATURE_SCHEMA_ID,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
) -> FineTuneResult:
    """Fine-tune mutual ranking and absolute pair confidence."""

    torch = _require_torch()
    cfg = config or MatcherConfig()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    matcher = (model or build_relation_aware_matcher(cfg)).to(device)
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if self_supervised_weight < 0.0:
        raise ValueError("self_supervised_weight must be non-negative")
    if self_supervised_interval <= 0:
        raise ValueError("self_supervised_interval must be positive")
    if correspondence_weight < 0.0:
        raise ValueError("correspondence_weight must be non-negative")
    if confidence_weight < 0.0:
        raise ValueError("confidence_weight must be non-negative")
    if not 0.0 <= context_mask_probability <= 1.0:
        raise ValueError("context_mask_probability must be between zero and one")
    training_queries = [query for query in queries if query.target_candidates]
    weak_pairs = list(correspondence_pairs or ())
    auxiliary_graphs = [
        graph for graph in (self_supervised_graphs or ()) if len(graph.nodes) >= 4
    ]
    auxiliary_config = augment_config or AugmentConfig()
    history: list[dict[str, float]] = []
    for epoch in range(max(1, int(epochs))):
        matcher.train()
        rng.shuffle(training_queries)
        losses: list[float] = []
        preference_losses: list[float] = []
        self_supervised_losses: list[float] = []
        negative_queries = 0
        confidence_losses: list[float] = []
        preference_pairs = 0
        self_supervised_unmatched_labels = 0
        self_supervised_positive_labels = 0
        self_supervised_pairs = 0
        correspondence_losses: list[float] = []
        final_assignment_losses: list[float] = []
        intermediate_assignment_losses: list[float] = []
        context_masked_queries = 0
        rng.shuffle(auxiliary_graphs)
        rng.shuffle(weak_pairs)
        auxiliary_index = 0
        for query_index, query in enumerate(training_queries):
            source_graph, target_graph, source_node_id = query_graphs(
                query,
                max_source_nodes=cfg.source_context_nodes,
                max_target_nodes=cfg.target_context_nodes,
            )
            source_index = next(
                index
                for index, node in enumerate(source_graph.nodes)
                if node.node_id == source_node_id
            )
            masked_source_indices = (
                (source_index,)
                if rng.random() < context_mask_probability
                else ()
            )
            context_masked_queries += int(bool(masked_source_indices))
            output = matcher(
                *matcher_inputs(
                    source_graph,
                    target_graph,
                    config=cfg,
                    device=device,
                    source_context_mask_indices=masked_source_indices,
                    feature_schema_id=feature_schema_id,
                )
            )
            layer_scores = tuple(output.get("assignment_scores_by_layer") or ())
            if not layer_scores:
                layer_scores = (output["logits_ab"],)
            candidate_layer_scores = tuple(
                _candidate_values(
                    scores[source_index],
                    target_graph,
                    query,
                    torch=torch,
                    device=device,
                )
                for scores in layer_scores
            )
            logits = candidate_layer_scores[-1]
            confidence_logits = _candidate_values(
                output["affinity"][source_index],
                target_graph,
                query,
                torch=torch,
                device=device,
            )
            gold_ids = set(query.acceptable_gold_candidate_ids())
            gold_indices = [
                index
                for index, candidate in enumerate(query.target_candidates)
                if candidate.candidate_id in gold_ids
            ]
            if gold_ids and not gold_indices:
                raise ValueError(
                    f"query {query.query_id} has gold ids absent from target candidates"
                )
            confidence_loss = _balanced_pair_confidence_loss(
                confidence_logits,
                gold_indices,
                torch=torch,
            )
            confidence_losses.append(float(confidence_loss.detach().cpu()))
            if gold_indices:
                assignment_losses = tuple(
                    _set_valued_assignment_loss(
                        layer_logits,
                        gold_indices,
                        torch=torch,
                        device=device,
                    )
                    for layer_logits in candidate_layer_scores
                )
                loss = torch.stack(assignment_losses).mean()
                final_assignment_losses.append(
                    float(assignment_losses[-1].detach().cpu())
                )
                intermediate_assignment_losses.append(
                    float(
                        (
                            torch.stack(assignment_losses[:-1]).mean()
                            if len(assignment_losses) > 1
                            else loss.new_zeros(())
                        )
                        .detach()
                        .cpu()
                    )
                )
            else:
                loss = confidence_loss.new_zeros(())
                negative_queries += 1
            loss = loss + float(confidence_weight) * confidence_loss
            query_preferences = tuple((outcome_preferences or {}).get(query.query_id, ()))
            if query_preferences and preference_weight > 0.0:
                ranking_loss = preference_ranking_loss(
                    logits,
                    tuple(
                        candidate.candidate_id for candidate in query.target_candidates
                    ),
                    query_preferences,
                )
                if ranking_loss is not None:
                    loss = loss + float(preference_weight) * ranking_loss
                    preference_losses.append(float(ranking_loss.detach().cpu()))
                    preference_pairs += len(query_preferences)
            if (
                auxiliary_graphs
                and self_supervised_weight > 0.0
                and query_index % self_supervised_interval == 0
            ):
                graph = auxiliary_graphs[auxiliary_index % len(auxiliary_graphs)]
                auxiliary_index += 1
                pair = make_training_pair(
                    graph,
                    rng=rng,
                    config=auxiliary_config,
                    matcher_config=cfg,
                )
                if pair is not None:
                    auxiliary_loss = matching_loss(
                        matcher,
                        pair,
                        device=device,
                        matcher_config=cfg,
                    )
                    loss = loss + float(self_supervised_weight) * auxiliary_loss
                    self_supervised_losses.append(
                        float(auxiliary_loss.detach().cpu())
                    )
                    labels = (*pair.targets_a_to_b, *pair.targets_b_to_a)
                    self_supervised_positive_labels += sum(
                        target >= 0 for target in labels
                    )
                    self_supervised_unmatched_labels += sum(
                        target < 0 for target in labels
                    )
                    self_supervised_pairs += 1
            correspondence_start = (
                query_index * len(weak_pairs) // len(training_queries)
            )
            correspondence_end = (
                (query_index + 1) * len(weak_pairs) // len(training_queries)
            )
            step_correspondence_losses = [
                matching_loss(
                    matcher,
                    pair,
                    device=device,
                    matcher_config=cfg,
                )
                for pair in weak_pairs[correspondence_start:correspondence_end]
            ]
            if step_correspondence_losses:
                correspondence_loss = torch.stack(step_correspondence_losses).mean()
                loss = loss + float(correspondence_weight) * correspondence_loss
                correspondence_losses.extend(
                    float(value.detach().cpu())
                    for value in step_correspondence_losses
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_metrics = {
                "epoch": float(epoch + 1),
                "loss": sum(losses) / max(len(losses), 1),
                "queries": float(len(losses)),
                "context_masked_queries": float(context_masked_queries),
                "context_mask_rate": (
                    context_masked_queries / max(len(losses), 1)
                ),
                "final_assignment_loss": (
                    sum(final_assignment_losses) / len(final_assignment_losses)
                    if final_assignment_losses
                    else 0.0
                ),
                "intermediate_assignment_loss": (
                    sum(intermediate_assignment_losses)
                    / len(intermediate_assignment_losses)
                    if intermediate_assignment_losses
                    else 0.0
                ),
                "supervised_layers": float(len(layer_scores)),
                "negative_queries": float(negative_queries),
                "confidence_loss": (
                    sum(confidence_losses) / len(confidence_losses)
                    if confidence_losses
                    else 0.0
                ),
                "preference_loss": (
                    sum(preference_losses) / len(preference_losses)
                    if preference_losses
                    else 0.0
                ),
                "preference_pairs": float(preference_pairs),
                "self_supervised_loss": (
                    sum(self_supervised_losses) / len(self_supervised_losses)
                    if self_supervised_losses
                    else 0.0
                ),
                "self_supervised_pairs": float(self_supervised_pairs),
                "self_supervised_positive_labels": float(
                    self_supervised_positive_labels
                ),
                "self_supervised_unmatched_labels": float(
                    self_supervised_unmatched_labels
                ),
                "correspondence_loss": (
                    sum(correspondence_losses) / len(correspondence_losses)
                    if correspondence_losses
                    else 0.0
                ),
                "correspondence_pairs": float(len(correspondence_losses)),
            }
        history.append(epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(dict(epoch_metrics))
    matcher.eval()
    return FineTuneResult(
        model=matcher,
        history=tuple(history),
        train_queries=len(training_queries),
        train_correspondence_pairs=len(weak_pairs),
    )


def predict_queries(
    model: Any,
    queries: Iterable[Query],
    *,
    config: MatcherConfig | None = None,
    device: str = "cpu",
    min_probability: float = 0.0,
    min_margin: float = 0.0,
    feature_schema_id: str = RCAM_FEATURE_SCHEMA_ID,
) -> list[Prediction]:
    """Predict candidates or NULL with no coordinate-passthrough fallback."""

    torch = _require_torch()
    cfg = config or MatcherConfig()
    model.to(device)
    model.eval()
    predictions: list[Prediction] = []
    with torch.no_grad():
        for query in queries:
            start = time.perf_counter()
            if not query.target_candidates:
                elapsed = (time.perf_counter() - start) * 1000.0
                predictions.append(
                    Prediction(
                        query.query_id,
                        None,
                        {},
                        metadata={
                            "mapping_status": "transfer_failure",
                            "graph_ms": 0.0,
                            "input_ms": 0.0,
                            "model_ms": 0.0,
                            "postprocess_ms": 0.0,
                            "latency_ms": elapsed,
                        },
                    )
                )
                continue
            graph_start = time.perf_counter()
            source_graph, target_graph, source_node_id = query_graphs(
                query,
                max_source_nodes=cfg.source_context_nodes,
                max_target_nodes=cfg.target_context_nodes,
            )
            source_index = next(
                index
                for index, node in enumerate(source_graph.nodes)
                if node.node_id == source_node_id
            )
            graph_end = time.perf_counter()
            model_inputs = matcher_inputs(
                source_graph,
                target_graph,
                config=cfg,
                device=device,
                feature_schema_id=feature_schema_id,
            )
            _synchronize(device, torch)
            input_end = time.perf_counter()
            output = model(*model_inputs)
            logits = _candidate_values(
                output["logits_ab"][source_index],
                target_graph,
                query,
                torch=torch,
                device=device,
            )
            confidence_logits = _candidate_values(
                output["affinity"][source_index],
                target_graph,
                query,
                torch=torch,
                device=device,
            )
            _synchronize(device, torch)
            model_end = time.perf_counter()
            probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            match_probabilities = (
                torch.sigmoid(confidence_logits).detach().cpu().tolist()
            )
            candidate_scores = {
                candidate.candidate_id: float(probabilities[index])
                for index, candidate in enumerate(query.target_candidates)
            }
            ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
            best_id, best_probability = ranked[0]
            best_index = next(
                index
                for index, candidate in enumerate(query.target_candidates)
                if candidate.candidate_id == best_id
            )
            match_confidence = float(match_probabilities[best_index])
            second_probability = max(
                (score for _, score in ranked[1:]),
                default=0.0,
            )
            margin = best_probability - second_probability
            selected = (
                best_id
                if match_confidence >= min_probability
                and margin >= min_margin
                else None
            )
            postprocess_end = time.perf_counter()
            predictions.append(
                Prediction(
                    query_id=query.query_id,
                    selected_candidate_id=selected,
                    scores=candidate_scores,
                    metadata={
                        "match_confidence": match_confidence,
                        "top1_probability": best_probability,
                        "margin": margin,
                        "mapping_status": "mapped" if selected else "transfer_failure",
                        "graph_ms": (graph_end - graph_start) * 1000.0,
                        "input_ms": (input_end - graph_end) * 1000.0,
                        "model_ms": (model_end - input_end) * 1000.0,
                        "postprocess_ms": (postprocess_end - model_end) * 1000.0,
                        "latency_ms": (postprocess_end - start) * 1000.0,
                        "source_nodes": len(source_graph.nodes),
                        "target_nodes": len(target_graph.nodes),
                    },
                )
            )
    return predictions


def benchmark_image_latency(
    model: Any,
    queries: Iterable[Query],
    *,
    config: MatcherConfig | None = None,
    device: str = "cpu",
    budget_ms: float = 50.0,
    max_images: int = 100,
    feature_schema_id: str = RCAM_FEATURE_SCHEMA_ID,
) -> LatencySummary:
    """Measure one query per decoded image pair after model-only warmup."""

    unique_queries: list[Query] = []
    seen_images: set[tuple[str, str]] = set()
    for query in queries:
        image_key = (
            str(
                query.metadata.get("source_screenshot_path")
                or query.metadata.get("source_screen")
                or f"{query.query_id}:source"
            ),
            str(
                query.metadata.get("target_screenshot_path")
                or query.metadata.get("target_screen")
                or f"{query.query_id}:target"
            ),
        )
        if image_key in seen_images:
            continue
        seen_images.add(image_key)
        unique_queries.append(query)
        if max_images > 0 and len(unique_queries) >= max_images:
            break
    if not unique_queries:
        return latency_summary((), budget_ms=budget_ms)
    cfg = config or MatcherConfig()
    prepared_paths: set[str] = set()
    asset_prepare_latencies: list[float] = []
    for query in unique_queries:
        prepare_start = time.perf_counter()
        for key in ("source_screenshot_path", "target_screenshot_path"):
            screenshot_path = str(query.metadata.get(key) or "")
            if not screenshot_path or screenshot_path in prepared_paths:
                continue
            prepare_visual_asset(
                screenshot_path,
                canvas_size=cfg.visual_canvas_size,
            )
            prepared_paths.add(screenshot_path)
        asset_prepare_latencies.append((time.perf_counter() - prepare_start) * 1000.0)
    warmup_query = Query(
        query_id=f"{unique_queries[0].query_id}:model_warmup",
        source=unique_queries[0].source,
        target_candidates=unique_queries[0].target_candidates,
        gold_candidate_id=unique_queries[0].gold_candidate_id,
        metadata={
            **unique_queries[0].metadata,
            "source_screenshot_path": "",
            "target_screenshot_path": "",
        },
    )
    predict_queries(
        model,
        (warmup_query,),
        config=config,
        device=device,
        feature_schema_id=feature_schema_id,
    )
    predictions = predict_queries(
        model,
        unique_queries,
        config=config,
        device=device,
        feature_schema_id=feature_schema_id,
    )
    return latency_summary(
        predictions,
        budget_ms=budget_ms,
        asset_prepare_latencies=asset_prepare_latencies,
    )


def latency_summary(
    predictions: Iterable[Prediction],
    *,
    budget_ms: float = 50.0,
    asset_prepare_latencies: Iterable[float] = (),
) -> LatencySummary:
    """Summarize total and stage latency with a strict all-samples gate."""

    rows = [
        prediction
        for prediction in predictions
        if prediction.metadata.get("latency_ms") is not None
    ]
    latencies = [float(row.metadata["latency_ms"]) for row in rows]
    preparation = [float(value) for value in asset_prepare_latencies]
    under_budget = sum(latency < budget_ms for latency in latencies)
    return LatencySummary(
        samples=len(latencies),
        budget_ms=float(budget_ms),
        p50_latency_ms=_percentile(latencies, 50.0),
        p95_latency_ms=_percentile(latencies, 95.0),
        max_latency_ms=max(latencies, default=0.0),
        under_budget_rate=under_budget / len(latencies) if latencies else 0.0,
        budget_met=bool(latencies) and under_budget == len(latencies),
        p95_graph_ms=_metadata_percentile(rows, "graph_ms", 95.0),
        p95_input_ms=_metadata_percentile(rows, "input_ms", 95.0),
        p95_model_ms=_metadata_percentile(rows, "model_ms", 95.0),
        p95_postprocess_ms=_metadata_percentile(rows, "postprocess_ms", 95.0),
        p50_asset_prepare_ms=_percentile(preparation, 50.0),
        p95_asset_prepare_ms=_percentile(preparation, 95.0),
        max_asset_prepare_ms=max(preparation, default=0.0),
    )


def evaluate_slices(
    queries: Iterable[Query],
    predictions: Iterable[Prediction],
    *,
    slice_keys: tuple[str, ...] = ("widget_type", "app", "mapping_cardinality"),
) -> dict[str, RankingMetrics]:
    """Evaluate the overall benchmark and diagnostic robustness slices."""

    query_list = list(queries)
    prediction_list = list(predictions)
    output = {"all": ranking_metrics(query_list, prediction_list)}
    prediction_by_id = {prediction.query_id: prediction for prediction in prediction_list}
    for key in slice_keys:
        groups: dict[str, list[Query]] = defaultdict(list)
        for query in query_list:
            value = _slice_value(query, key)
            groups[value].append(query)
        for value, group in sorted(groups.items()):
            group_predictions = [
                prediction_by_id[query.query_id]
                for query in group
                if query.query_id in prediction_by_id
            ]
            output[f"{key}:{value}"] = ranking_metrics(group, group_predictions)
    return output


def _source_graph(query: Query) -> tuple[UIGraph, str]:
    xml_path = str(query.metadata.get("source_xml_path") or "")
    graph: UIGraph | None = None
    if xml_path:
        if not Path(xml_path).is_file():
            raise FileNotFoundError(f"source XML is missing: {xml_path}")
        coordinate_size = _coordinate_size(query.metadata, "source")
        parsed = _xml_graph_from_path(
            str(Path(xml_path).resolve()),
            width=coordinate_size[0] if coordinate_size is not None else 0.0,
            height=coordinate_size[1] if coordinate_size is not None else 0.0,
        )
        graph = UIGraph(
            graph_id=f"{query.query_id}:source",
            nodes=parsed.nodes,
            width=parsed.width,
            height=parsed.height,
            metadata={
                **parsed.metadata,
                "screenshot_path": str(query.metadata.get("source_screenshot_path") or ""),
            },
        )
    descriptor = _node_from_payload(
        query.source,
        node_id="__source__",
        origin_id="__source__",
    )
    if graph is not None:
        matched = _locate_source_node(graph, descriptor, query.source)
        if matched is not None:
            return (
                UIGraph(
                    graph_id=graph.graph_id,
                    nodes=graph.nodes,
                    width=graph.width,
                    height=graph.height,
                    metadata={**graph.metadata, "source_anchor_bound": True},
                ),
                matched.node_id,
            )
        raise ValueError(
            f"query {query.query_id} source anchor does not bind to its XML graph"
        )
    width, height = _graph_size((descriptor,))
    return (
        UIGraph(
            graph_id=f"{query.query_id}:source",
            nodes=(descriptor,),
            width=width,
            height=height,
            metadata={
                "screenshot_path": str(query.metadata.get("source_screenshot_path") or ""),
            },
        ),
        descriptor.node_id,
    )


def _target_graph(
    query: Query,
    *,
    graph_id: str,
    screenshot_path: str = "",
) -> UIGraph:
    candidates = query.target_candidates
    xml_path = _target_xml_path(query)
    if xml_path:
        path = Path(xml_path)
        if not path.is_file():
            raise FileNotFoundError(f"target XML is missing: {xml_path}")
        parsed_base = _xml_graph_from_path(
            str(path.resolve()),
            width=0.0,
            height=0.0,
        )
        parsed = UIGraph(
            graph_id=graph_id,
            nodes=parsed_base.nodes,
            width=parsed_base.width,
            height=parsed_base.height,
            metadata=parsed_base.metadata,
        )
        return _bind_target_candidates(
            parsed,
            candidates,
            screenshot_path=screenshot_path,
        )
    nodes = tuple(
        _node_from_payload(
            {**candidate.metadata, "bounds": candidate.bbox},
            node_id=candidate.candidate_id,
            origin_id=candidate.candidate_id,
        )
        for candidate in candidates
    )
    width, height = _graph_size(nodes)
    return UIGraph(
        graph_id=graph_id,
        nodes=nodes,
        width=width,
        height=height,
        metadata={"screenshot_path": screenshot_path},
    )


def _bind_target_candidates(
    graph: UIGraph,
    candidates: tuple[Candidate, ...],
    *,
    screenshot_path: str,
) -> UIGraph:
    candidates_by_index: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        node_index = _candidate_node_index(candidate, graph)
        if node_index is None:
            raise ValueError(
                f"candidate {candidate.candidate_id} does not bind to target XML"
            )
        candidates_by_index[node_index].append(candidate)

    renamed_ids = {
        node.node_id: candidates_by_index[index][0].candidate_id
        if index in candidates_by_index
        else node.node_id
        for index, node in enumerate(graph.nodes)
    }
    nodes: list[UINode] = []
    for index, node in enumerate(graph.nodes):
        node_candidates = candidates_by_index.get(index)
        if node_candidates is None:
            node_candidates = []
        if not node_candidates:
            nodes.append(
                replace(
                    node,
                    parent_id=renamed_ids.get(node.parent_id, node.parent_id),
                    child_ids=tuple(
                        renamed_ids.get(child_id, child_id) for child_id in node.child_ids
                    ),
                )
            )
            continue
        for candidate_offset, candidate in enumerate(node_candidates):
            descriptor = _node_from_payload(
                {**candidate.metadata, "bounds": candidate.bbox},
                node_id=candidate.candidate_id,
                origin_id=node.origin_id,
            )
            node = replace(
                node,
                node_id=candidate.candidate_id,
                text=descriptor.text or node.text,
                content_desc=descriptor.content_desc or node.content_desc,
                resource_id=descriptor.resource_id or node.resource_id,
                class_name=descriptor.class_name or node.class_name,
                bbox=descriptor.bbox or node.bbox,
                clickable=descriptor.clickable,
                editable=descriptor.editable,
                scrollable=descriptor.scrollable,
                enabled=descriptor.enabled,
                metadata={
                    **node.metadata,
                    **descriptor.metadata,
                    "is_candidate": True,
                    "candidate_id": candidate.candidate_id,
                    "xml_node_index": index,
                },
            )
            nodes.append(
                replace(
                    node,
                    parent_id=renamed_ids.get(node.parent_id, node.parent_id),
                    child_ids=tuple(
                        renamed_ids.get(child_id, child_id) for child_id in node.child_ids
                    )
                    if candidate_offset == 0
                    else (),
                )
            )
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=tuple(nodes),
        width=graph.width,
        height=graph.height,
        metadata={
            **graph.metadata,
            "screenshot_path": screenshot_path,
            "candidate_node_ids": tuple(candidate.candidate_id for candidate in candidates),
            "target_candidates_bound": len(candidates),
        },
    )


def _candidate_node_index(
    candidate: Candidate,
    graph: UIGraph,
) -> int | None:
    raw_index = candidate.metadata.get("node_index")
    try:
        node_index = int(raw_index) if raw_index is not None else None
    except (TypeError, ValueError):
        node_index = None
    if (
        node_index is not None
        and 0 <= node_index < len(graph.nodes)
    ):
        return node_index
    descriptor = _node_from_payload(
        {**candidate.metadata, "bounds": candidate.bbox},
        node_id=candidate.candidate_id,
        origin_id=candidate.candidate_id,
    )
    point = _source_point(candidate.metadata, descriptor.bbox)
    ranked = sorted(
        (
            (_source_anchor_score(descriptor, node, point=point), index)
            for index, node in enumerate(graph.nodes)
            if node.bbox is not None
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if ranked and ranked[0][0] >= 0.5:
        return ranked[0][1]
    return None


def _target_xml_path(query: Query) -> str:
    value = query.metadata.get("target_xml_path")
    if value:
        return str(value)
    for candidate in query.target_candidates:
        value = candidate.metadata.get("target_xml_path") or candidate.metadata.get("xml_path")
        if value:
            return str(value)
    return ""


def _coordinate_size(metadata: Mapping[str, Any], prefix: str) -> tuple[float, float] | None:
    width = metadata.get(f"{prefix}_coordinate_width")
    height = metadata.get(f"{prefix}_coordinate_height")
    try:
        width_value = float(width)
        height_value = float(height)
    except (TypeError, ValueError):
        return None
    if width_value <= 0.0 or height_value <= 0.0:
        return None
    return width_value, height_value


def _scale_graph(graph: UIGraph, *, width: float, height: float) -> UIGraph:
    if not graph.width or not graph.height:
        raise ValueError(f"graph {graph.graph_id} has no coordinate extent")
    scale_x = width / float(graph.width)
    scale_y = height / float(graph.height)
    return UIGraph(
        graph_id=graph.graph_id,
        nodes=tuple(
            replace(
                node,
                bbox=(
                    node.bbox[0] * scale_x,
                    node.bbox[1] * scale_y,
                    node.bbox[2] * scale_x,
                    node.bbox[3] * scale_y,
                )
                if node.bbox is not None
                else None,
            )
            for node in graph.nodes
        ),
        width=width,
        height=height,
        metadata={
            **graph.metadata,
            "coordinate_scale": (scale_x, scale_y),
        },
    )


@lru_cache(maxsize=4096)
def _xml_graph_from_path(path: str, *, width: float, height: float) -> UIGraph:
    parsed = graph_from_record(
        {"xml": Path(path).read_text(encoding="utf-8")},
        graph_id=path,
    )
    if width > 0.0 and height > 0.0:
        return _scale_graph(parsed, width=width, height=height)
    return parsed


def _candidate_values(
    logits: Any,
    target_graph: UIGraph,
    query: Query,
    *,
    torch: Any,
    device: str,
) -> Any:
    node_indices = {node.node_id: index for index, node in enumerate(target_graph.nodes)}
    missing = [
        candidate.candidate_id
        for candidate in query.target_candidates
        if candidate.candidate_id not in node_indices
    ]
    if missing:
        raise ValueError(f"target graph dropped candidate nodes: {missing!r}")
    selected_indices = [
        node_indices[candidate.candidate_id] for candidate in query.target_candidates
    ]
    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
    return logits.index_select(0, index_tensor)


def _set_valued_assignment_loss(
    logits: Any,
    gold_indices: list[int],
    *,
    torch: Any,
    device: str,
) -> Any:
    log_probabilities = torch.log_softmax(logits, dim=0)
    selected = torch.tensor(gold_indices, dtype=torch.long, device=device)
    return -torch.logsumexp(log_probabilities[selected], dim=0)


def _balanced_pair_confidence_loss(
    logits: Any,
    gold_indices: list[int],
    *,
    torch: Any,
) -> Any:
    labels = torch.zeros_like(logits)
    if gold_indices:
        labels[torch.tensor(gold_indices, dtype=torch.long, device=logits.device)] = 1.0
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    parts = []
    if torch.any(positive_mask):
        parts.append(losses[positive_mask].mean())
    if torch.any(negative_mask):
        parts.append(losses[negative_mask].mean())
    return torch.stack(parts).mean()


def _node_from_payload(
    payload: dict[str, Any],
    *,
    node_id: str,
    origin_id: str,
) -> UINode:
    bbox = _bbox(payload.get("bounds", payload.get("bbox")))
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "action_type",
        "parent_text",
        "sibling_texts",
        "nearby_texts",
        "screen_region",
    ):
        if key in payload:
            metadata[key] = payload[key]
    return UINode(
        node_id=node_id,
        origin_id=origin_id,
        text=str(payload.get("text") or payload.get("label") or ""),
        content_desc=str(
            payload.get("content_desc") or payload.get("content-desc") or ""
        ),
        resource_id=str(
            payload.get("resource_id") or payload.get("resource-id") or ""
        ),
        class_name=str(payload.get("class_name") or payload.get("class") or ""),
        bbox=bbox,
        clickable=_optional_bool(payload.get("clickable")),
        editable=_optional_bool(payload.get("editable")),
        scrollable=_optional_bool(payload.get("scrollable")),
        enabled=payload.get("enabled") is not False,
        metadata=metadata,
    )


def _locate_source_node(
    graph: UIGraph,
    descriptor: UINode,
    payload: dict[str, Any],
) -> UINode | None:
    point = _source_point(payload, descriptor.bbox)
    candidates = [node for node in graph.nodes if node.bbox is not None]
    if descriptor.bbox is not None:
        ranked = sorted(
            (
                (
                    _source_anchor_score(descriptor, node, point=point),
                    node,
                )
                for node in candidates
            ),
            key=lambda item: (-item[0], item[1].node_id),
        )
        if ranked and ranked[0][0] >= 0.5:
            return ranked[0][1]
    if point is not None and not any(
        (descriptor.text, descriptor.content_desc, descriptor.resource_id, descriptor.class_name)
    ):
        containing = [node for node in candidates if _contains(node.bbox, point)]
        if containing:
            return min(containing, key=lambda node: _area(node.bbox))
    return None


def _source_anchor_score(
    source: UINode,
    candidate: UINode,
    *,
    point: tuple[float, float] | None,
) -> float:
    score = _iou(source.bbox, candidate.bbox) if source.bbox is not None else 0.0
    if source.resource_id and source.resource_id == candidate.resource_id:
        score += 0.30
    if source.text and source.text.strip().lower() == candidate.text.strip().lower():
        score += 0.20
    if source.content_desc and source.content_desc.strip().lower() == candidate.content_desc.strip().lower():
        score += 0.20
    if source.class_name and _class_tail(source.class_name) == _class_tail(candidate.class_name):
        score += 0.10
    if point is not None and _contains(candidate.bbox, point):
        score += 0.10
    return score


def _class_tail(value: str) -> str:
    return str(value or "").rsplit(".", 1)[-1].lower()


def _query_group(query: Query) -> str:
    source_metadata = query.source.get("metadata") or {}
    for value in (
        query.metadata.get("app"),
        source_metadata.get("app"),
        query.metadata.get("package"),
        source_metadata.get("package"),
    ):
        if str(value or "").strip():
            return str(value)
    return query.query_id


def _slice_value(query: Query, key: str) -> str:
    if key == "mapping_cardinality":
        count = len(query.acceptable_gold_candidate_ids())
        return "null" if count == 0 else "one" if count == 1 else "many"
    source_metadata = query.source.get("metadata") or {}
    return str(query.metadata.get(key) or source_metadata.get(key) or "unknown")


def _graph_size(nodes: tuple[UINode, ...]) -> tuple[float, float]:
    right = max((node.bbox or (0.0, 0.0, 1.0, 1.0))[2] for node in nodes)
    bottom = max((node.bbox or (0.0, 0.0, 1.0, 1.0))[3] for node in nodes)
    return max(float(right), 1.0), max(float(bottom), 1.0)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = map(float, value)
    except (TypeError, ValueError):
        return None
    return (left, top, right, bottom) if right > left and bottom > top else None


def _source_point(
    payload: dict[str, Any],
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    try:
        if payload.get("x") is not None and payload.get("y") is not None:
            return float(payload["x"]), float(payload["y"])
    except (TypeError, ValueError):
        pass
    if bbox is None:
        return None
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _optional_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _contains(
    bbox: tuple[float, float, float, float] | None,
    point: tuple[float, float],
) -> bool:
    return bool(
        bbox is not None
        and bbox[0] <= point[0] <= bbox[2]
        and bbox[1] <= point[1] <= bbox[3]
    )


def _area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return math.inf
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float] | None,
) -> float:
    if second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _synchronize(device: str, torch: Any) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _metadata_percentile(
    predictions: list[Prediction],
    key: str,
    percentile: float,
) -> float:
    values = [
        float(prediction.metadata[key])
        for prediction in predictions
        if prediction.metadata.get(key) is not None
    ]
    return _percentile(values, percentile)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(round((len(ordered) - 1) * percentile / 100.0)),
        ),
    )
    return float(ordered[index])


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for learned matcher fine-tuning. Install omnitransfer[train]."
        ) from exc
    return torch
