#!/usr/bin/env python3
"""Run RAG retrieval evaluation.

Loads benchmark dataset, runs retrieval for each query, computes metrics,
and saves timestamped results.

Usage:
    python tools/rag_eval/run_eval.py \
        --dataset benchmark/dataset.json \
        --packs assets/rag/packs \
        --model assets/models/minilm.onnx \
        --vocab assets/models/vocab.txt

    # Quick smoke test with a single pack:
    python tools/rag_eval/run_eval.py \
        --dataset benchmark/dataset.json \
        --packs assets/rag/packs \
        --model assets/models/minilm.onnx \
        --vocab assets/models/vocab.txt \
        --filter-pack bleeding
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from metrics import compute_all_metrics
from retrieval import RetrievalEngine


def run_evaluation(args):
    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = (project_root / args.dataset).resolve()
    packs_dir = (project_root / args.packs).resolve()
    model_path = (project_root / args.model).resolve()
    vocab_path = (project_root / args.vocab).resolve()
    results_dir = (project_root / "benchmark" / "results").resolve()

    # Validate inputs
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not vocab_path.exists():
        print(f"Error: vocab not found: {vocab_path}", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries.")

    # Optionally filter to single pack
    if args.filter_pack:
        queries = [q for q in queries if q["pack_id"] == args.filter_pack]
        print(f"Filtered to pack '{args.filter_pack}': {len(queries)} queries.")

    if not queries:
        print("No queries to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Initialize retrieval engine
    engine = RetrievalEngine(
        packs_dir=str(packs_dir),
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
    )

    # Run retrieval for each query
    print(f"\nRunning retrieval ({args.mode} mode)...")
    results = []
    start_time = time.time()

    for q in tqdm(queries, desc="Retrieval", unit="q"):
        retrieved = engine.search(q["query"], mode=args.mode, top_k=args.top_k)
        results.append({
            "query": q["query"],
            "query_id": q["query_id"],
            "retrieved_ids": retrieved,
            "relevant_ids": q["relevant_chunk_ids"],
            "pack_id": q["pack_id"],
        })

    elapsed = time.time() - start_time
    print(f"Retrieval complete: {len(queries)} queries in {elapsed:.1f}s ({len(queries)/elapsed:.0f} q/s)")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(results)

    # Assemble output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "retrieval_mode": args.mode,
            "rrf_k": args.rrf_k,
            "vector_top_k": args.vector_top_k,
            "bm25_top_k": args.bm25_top_k,
            "output_top_k": args.top_k,
        },
        "overall": metrics["overall"],
        "per_pack": metrics["per_pack"],
    }

    # Save results
    if args.output:
        results_path = (project_root / args.output).resolve()
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        results_path = results_dir / f"{ts}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    o = output["overall"]
    print(f"\n{'='*50}")
    print(f"  Overall Metrics ({o['total_queries']} queries)")
    print(f"{'='*50}")
    print(f"  MRR:              {o.get('mrr', 0):.4f}")
    print(f"  NDCG@5:           {o.get('ndcg@5', 0):.4f}")
    print(f"  NDCG@10:          {o.get('ndcg@10', 0):.4f}")
    print(f"  Hit Rate@1:       {o.get('hit_rate@1', 0):.4f}")
    print(f"  Hit Rate@3:       {o.get('hit_rate@3', 0):.4f}")
    print(f"  Hit Rate@5:       {o.get('hit_rate@5', 0):.4f}")
    print(f"  RAGAS Precision:  {o.get('ragas_id_precision', 0):.4f}")
    print(f"  RAGAS Recall:     {o.get('ragas_id_recall', 0):.4f}")
    print(f"{'='*50}")

    print_metrics_glossary(args.top_k)

    return output


def print_metrics_glossary(output_top_k: int):
    """Print a human-readable explanation of each metric after the summary.

    This benchmark has exactly ONE gold/relevant chunk per query, which caps
    some metrics mechanically (recall@k == hit_rate@k; precision is bounded by
    1/k). The notes below read the numbers against those real ceilings.
    """
    precision_cap = 1.0 / output_top_k if output_top_k else 0.0
    print(f"\n{'='*50}")
    print("  Metric Guide  (range, direction, what to expect)")
    print(f"{'='*50}")
    print("  NOTE: one gold chunk per query. So recall@k == hit_rate@k,")
    print(f"        and set-based precision is capped at 1/k "
          f"(RAGAS Precision ceiling = {precision_cap:.2f}).")
    print()
    print("  MRR (Mean Reciprocal Rank)")
    print("    Avg of 1/rank of the first relevant chunk. Range 0-1, higher better.")
    print("    1.0 = gold always ranked #1; 0.5 ~ around rank 2. Good: ~0.6-0.8.")
    print()
    print("  NDCG@k (Normalized Discounted Cumulative Gain)")
    print("    Ranking quality in top-k, position-discounted, normalized. Range 0-1,")
    print("    higher better. Rises with k. Good retrievers ~0.6-0.8.")
    print()
    print("  Hit Rate@k  (== Recall@k here)")
    print("    Fraction of queries with the gold chunk in top-k. Range 0-1, higher")
    print("    better. @1 = top result correct; @10 = answer somewhere in top 10.")
    print()
    print("  RAGAS Precision (IDBasedContextPrecision)")
    print("    |retrieved ∩ relevant| / |retrieved| over all returned chunks.")
    print(f"    Range 0-1 but capped at {precision_cap:.2f} (one gold chunk over "
          f"{output_top_k} slots).")
    print("    A low absolute value is structural dilution, not a bad retriever;")
    print("    it tracks RAGAS Recall / output_top_k. Higher better.")
    print()
    print("  RAGAS Recall (IDBasedContextRecall)")
    print("    |retrieved ∩ relevant| / |relevant| over all returned chunks. Range")
    print("    0-1, higher better. The headline 'did we retrieve the answer at all'.")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation")
    parser.add_argument(
        "--dataset", default="benchmark/dataset.json",
        help="Path to benchmark dataset JSON"
    )
    parser.add_argument(
        "--packs", default="assets/rag/packs",
        help="Directory containing pack JSONs"
    )
    parser.add_argument(
        "--model", default="assets/models/minilm.onnx",
        help="Path to ONNX embedding model"
    )
    parser.add_argument(
        "--vocab", default="assets/models/vocab.txt",
        help="Path to BERT vocab file"
    )
    parser.add_argument(
        "--mode", default="hybrid", choices=["hybrid", "vector", "bm25"],
        help="Retrieval mode"
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF fusion constant k"
    )
    parser.add_argument(
        "--vector-top-k", type=int, default=5,
        help="Number of vector search candidates"
    )
    parser.add_argument(
        "--bm25-top-k", type=int, default=10,
        help="Number of BM25 search candidates"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of final results per query"
    )
    parser.add_argument(
        "--filter-pack", type=str, default=None,
        help="Evaluate only queries from this pack (for smoke testing)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (relative to project root). If omitted, writes a "
             "timestamped file to benchmark/results/."
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
