#!/usr/bin/env python3
"""Retrieval evaluation metrics for RAG evaluation.

Provides:
- Traditional retrieval metrics: MRR, NDCG@K, Hit Rate@K
- RAGAS ID-based metrics: Context Precision, Context Recall (deterministic, no LLM)
"""

import math
from collections import defaultdict


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of retrieved items in top-k that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank: 1/rank of first relevant item, 0 if none found."""
    for i, cid in enumerate(retrieved):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: relevant=1, non-relevant=0.
    """
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-based

    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant item in top-k, 0.0 otherwise."""
    top_k = retrieved[:k]
    return 1.0 if any(cid in relevant for cid in top_k) else 0.0


def ragas_context_precision(retrieved: list[str], relevant: set[str]) -> float:
    """RAGAS IDBasedContextPrecision.

    Precision = |retrieved ∩ relevant| / |retrieved|
    Measures: of all retrieved chunks, what fraction are relevant.
    """
    if not retrieved:
        return 0.0
    retrieved_set = set(retrieved)
    return len(retrieved_set & relevant) / len(retrieved_set)


def ragas_context_recall(retrieved: list[str], relevant: set[str]) -> float:
    """RAGAS IDBasedContextRecall.

    Recall = |retrieved ∩ relevant| / |relevant|
    Measures: of all relevant chunks, what fraction were retrieved.
    """
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved)
    return len(retrieved_set & relevant) / len(relevant)


def compute_metrics_for_query(
    retrieved: list[str],
    relevant_ids: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Compute all metrics for a single query."""
    relevant = set(relevant_ids)
    result = {
        "mrr": reciprocal_rank(retrieved, relevant),
        "ragas_id_precision": ragas_context_precision(retrieved, relevant),
        "ragas_id_recall": ragas_context_recall(retrieved, relevant),
    }
    for k in ks:
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        result[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        result[f"precision@{k}"] = _precision_at_k(retrieved, relevant, k)
        result[f"recall@{k}"] = _recall_at_k(retrieved, relevant, k)
    return result


def _average_metrics(all_metrics: list[dict]) -> dict:
    """Average a list of per-query metric dicts."""
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}


def compute_all_metrics(
    results: list[dict],
) -> dict:
    """Compute overall and per-pack metrics.

    Args:
        results: list of dicts, each with:
            - query: str
            - retrieved_ids: list[str]
            - relevant_ids: list[str]
            - pack_id: str (optional, for per-pack breakdown)

    Returns:
        {overall: {...}, per_pack: {pack_id: {...}}}
    """
    all_query_metrics = []
    per_pack_metrics: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        retrieved = r["retrieved_ids"]
        relevant = r["relevant_ids"]
        pack_id = r.get("pack_id", "unknown")

        query_metrics = compute_metrics_for_query(retrieved, relevant)
        all_query_metrics.append(query_metrics)
        per_pack_metrics[pack_id].append(query_metrics)

    overall = _average_metrics(all_query_metrics)
    overall["total_queries"] = len(results)

    per_pack = {
        pack_id: {**_average_metrics(metrics), "total_queries": len(metrics)}
        for pack_id, metrics in per_pack_metrics.items()
    }

    return {"overall": overall, "per_pack": per_pack}
