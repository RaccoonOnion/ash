#!/usr/bin/env python3
"""Cross-relevance mining to expand single-positive gold labels to multi-positive.

For each query in the benchmark dataset, this script finds additional relevant chunks
beyond the source chunk using TWO signals and takes their INTERSECTION:
1. Embedding similarity: chunks with high cosine similarity to both query and source
2. LLM-judge: a local model judges "Does chunk X answer query Q?"

Usage:
    python tools/rag_eval/cross_relevance_mining.py \
        --dataset benchmark/dataset.json \
        --packs assets/rag/packs \
        --model assets/models/minilm.onnx \
        --vocab assets/models/vocab.txt \
        --output benchmark/dataset_multi_positive.json \
        --similarity-threshold 0.8
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent dir to import from rag_preprocessor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_preprocessor import create_embedder, embed_text, BertTokenizer
from retrieval import RetrievalEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. LLM-judge signal will be disabled.", file=sys.stderr)


def _llm_judge_chunk(query: str, chunk_text: str, model, tokenizer, device="cuda") -> bool:
    """Ask a local LLM: does this chunk answer the query? Returns True if yes."""
    if not TORCH_AVAILABLE:
        return False

    # Simple prompt format - adjust based on model
    prompt = f"""Question: {query}

Context: {chunk_text[:500]}

Answer the question with ONLY "yes" or "no" - does the context above answer the question?
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special=True).strip().lower()

    # Check for positive responses
    return any(word in response for word in ['yes', 'yeah', 'yep', 'correct'])


def mine_cross_relevance(
    dataset_path: str,
    packs_dir: str,
    model_path: str,
    vocab_path: str,
    output_path: str,
    similarity_threshold: float = 0.8,
    use_llm_judge: bool = True,
    judge_model: str = "google/gemma-2b",
    top_candidates: int = 20,
    sample_audit_size: int = 200,
):
    """Expand single-positive gold to multi-positive using cross-relevance mining."""

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries")

    # Load retrieval engine for embeddings
    print("Loading retrieval engine (for embeddings)...")
    engine = RetrievalEngine(
        packs_dir=str(packs_dir),
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        vector_top_k=50,  # Get more candidates
        bm25_top_k=50,
        rrf_k=60,
    )

    # Load LLM judge if requested
    llm_model = None
    llm_tokenizer = None
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    if use_llm_judge and TORCH_AVAILABLE:
        print(f"Loading LLM judge: {judge_model}")
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(judge_model)
            llm_model = AutoModelForCausalLM.from_pretrained(
                judge_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                llm_model = llm_model.to("cpu")
            print(f"LLM judge loaded on {device}")
        except Exception as e:
            print(f"Failed to load LLM judge: {e}", file=sys.stderr)
            print("Proceeding with embedding signal only.", file=sys.stderr)
            use_llm_judge = False

    # Process each query
    expanded_queries = []
    stats = {
        "total_queries": len(queries),
        "total_expansions": 0,
        "embedding_only_hits": 0,
        "llm_judge_hits": 0,
        "final_multi_positive": 0,
    }

    # Sample for audit
    audit_sample = []
    import random
    random.seed(42)
    audit_indices = set(random.sample(range(len(queries)), min(sample_audit_size, len(queries))))

    for idx, q in enumerate(tqdm(queries, desc="Mining cross-relevance")):
        query_text = q["query"]
        source_chunk_id = q["relevant_chunk_ids"][0]  # Single-positive assumption

        # Get source chunk embedding
        source_idx = engine.chunk_id_to_idx[source_chunk_id]
        source_embedding = engine.embeddings[source_idx]

        # Get query embedding
        query_emb = np.array(
            embed_text(engine.session, engine.tokenizer, query_text),
            dtype=np.float32
        )
        query_emb = query_emb / np.linalg.norm(query_emb)

        # Signal 1: Embedding similarity to query AND source
        # Compute similarities to all chunks
        query_sim = engine.embeddings @ query_emb
        source_sim = engine.embeddings @ source_embedding

        # Threshold: must be similar to BOTH query and source
        mask = (query_sim >= similarity_threshold) & (source_sim >= similarity_threshold)
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            # No candidates
            expanded_queries.append(q)
            continue

        stats["embedding_only_hits"] += len(candidate_indices)

        # Signal 2: LLM judge (if enabled) - only judge top candidates
        final_chunk_ids = [source_chunk_id]  # Always include source

        if use_llm_judge and llm_model is not None:
            # Sort by query similarity and take top N
            top_indices = candidate_indices[np.argsort(query_sim[candidate_indices])[::-1][:top_candidates]]

            for candidate_idx in top_indices:
                if candidate_idx == source_idx:
                    continue  # Skip source (already included)

                candidate_chunk_id = engine.chunks[candidate_idx]["chunkId"]
                candidate_text = engine.chunks[candidate_idx]["text"]

                # Ask LLM judge
                is_relevant = _llm_judge_chunk(
                    query_text, candidate_text, llm_model, llm_tokenizer, device
                )

                if is_relevant:
                    final_chunk_ids.append(candidate_chunk_id)

            stats["llm_judge_hits"] += len(final_chunk_ids) - 1
        else:
            # No LLM judge - use all embedding candidates
            for candidate_idx in candidate_indices:
                if candidate_idx == source_idx:
                    continue
                final_chunk_ids.append(engine.chunks[candidate_idx]["chunkId"])

        # Dedupe while preserving order
        seen = set()
        unique_chunk_ids = []
        for cid in final_chunk_ids:
            if cid not in seen:
                seen.add(cid)
                unique_chunk_ids.append(cid)

        # Update query
        expanded_q = q.copy()
        expanded_q["relevant_chunk_ids"] = unique_chunk_ids

        if len(unique_chunk_ids) > 1:
            stats["total_expansions"] += 1
            stats["final_multi_positive"] += len(unique_chunk_ids)

        expanded_queries.append(expanded_q)

        # Add to audit sample if selected
        if idx in audit_indices:
            audit_sample.append({
                "query": query_text,
                "source_chunk_id": source_chunk_id,
                "expanded_chunk_ids": unique_chunk_ids,
                "num_expanded": len(unique_chunk_ids) - 1
            })

    # Create output dataset
    output_dataset = {
        "version": "1.1",
        "generated_at": dataset.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "generator_model": dataset.get("generator_model", "unknown"),
        "expanded_at": datetime.now(timezone.utc).isoformat(),
        "expansion_config": {
            "similarity_threshold": similarity_threshold,
            "use_llm_judge": use_llm_judge,
            "judge_model": judge_model if use_llm_judge else "none",
            "top_candidates": top_candidates,
        },
        "total_queries": len(expanded_queries),
        "statistics": stats,
        "queries": expanded_queries,
    }

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nOutput saved to: {output_path}")
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save audit sample
    audit_path = output_path.replace(".json", "_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_sample, f, indent=2, ensure_ascii=False)
    print(f"\nAudit sample ({len(audit_sample)} pairs) saved to: {audit_path}")
    print("Please manually audit this sample to estimate expansion precision.")


def main():
    parser = argparse.ArgumentParser(description="Cross-relevance mining for multi-positive gold")
    parser.add_argument("--dataset", default="benchmark/dataset.json", help="Input dataset path")
    parser.add_argument("--packs", default="assets/rag/packs", help="Packs directory")
    parser.add_argument("--model", default="assets/models/minilm.onnx", help="Embedding model path")
    parser.add_argument("--vocab", default="assets/models/vocab.txt", help="BERT vocab path")
    parser.add_argument("--output", default="benchmark/dataset_multi_positive.json", help="Output dataset path")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Cosine similarity threshold")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM judge signal")
    parser.add_argument("--judge-model", default="google/gemma-2b", help="Local LLM for judging")
    parser.add_argument("--top-candidates", type=int, default=20, help="Top candidates to judge with LLM")
    parser.add_argument("--sample-audit-size", type=int, default=200, help="Sample size for manual audit")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    args.dataset = str((project_root / args.dataset).resolve())
    args.packs = str((project_root / args.packs).resolve())
    args.model = str((project_root / args.model).resolve())
    args.vocab = str((project_root / args.vocab).resolve())
    args.output = str((project_root / args.output).resolve())

    mine_cross_relevance(
        dataset_path=args.dataset,
        packs_dir=args.packs,
        model_path=args.model,
        vocab_path=args.vocab,
        output_path=args.output,
        similarity_threshold=args.similarity_threshold,
        use_llm_judge=not args.no_llm_judge,
        judge_model=args.judge_model,
        top_candidates=args.top_candidates,
        sample_audit_size=args.sample_audit_size,
    )


if __name__ == "__main__":
    main()
