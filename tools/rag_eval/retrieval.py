#!/usr/bin/env python3
"""Offline retrieval engine for RAG evaluation.

Replicates the Dart hybrid retrieval logic (BM25 + vector search + RRF fusion)
in pure Python so we can evaluate retrieval quality without running the app.

Reuses BertTokenizer, embed_text, and create_embedder from rag_preprocessor.py.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent dir so we can import from rag_preprocessor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_preprocessor import BertTokenizer, create_embedder, embed_text

# BM25 parameters — matches lib/services/bm25_search.dart
_K1 = 1.5
_B = 0.75

# ~30 common English stop words — matches Dart implementation
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as", "was", "are",
    "be", "been", "has", "have", "had", "this", "that", "not", "no",
    "do", "if", "we", "you", "can", "will", "what", "how",
})

# RRF constant — matches lib/services/gemma_inference_service.dart
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, filter stop words."""
    cleaned = text.lower()
    # Strip punctuation
    for ch in cleaned:
        if not ch.isalnum() and not ch.isspace():
            cleaned = cleaned.replace(ch, " ")
    tokens = cleaned.split()
    return [t for t in tokens if t and t not in _STOP_WORDS]


class RetrievalEngine:
    """BM25 + vector search + RRF fusion for offline evaluation."""

    def __init__(
        self,
        packs_dir: str,
        model_path: str,
        vocab_path: str,
        vector_top_k: int = 5,
        bm25_top_k: int = 10,
        rrf_k: int = 60,
    ):
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

        # Load ONNX embedder
        print("Loading ONNX model...")
        self.session = create_embedder(model_path)
        self.tokenizer = BertTokenizer(vocab_path)
        print("Model loaded.")

        # Load all packs
        self.chunks: list[dict] = []  # [{chunkId, text, embedding, packId, ...}]
        self.embeddings: np.ndarray | None = None  # shape (N, 384)
        self.chunk_id_to_idx: dict[str, int] = {}

        # BM25 index structures
        self._bm25_index: dict[str, list[tuple[int, int]]] = {}  # token → [(docIdx, tf)]
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0

        self._load_packs(packs_dir)
        self._build_bm25_index()

    def _load_packs(self, packs_dir: str):
        """Load all pack JSONs from directory."""
        packs_path = Path(packs_dir)
        all_embeddings = []

        pack_files = sorted(packs_path.glob("*.json"))
        # Skip registry file
        pack_files = [f for f in pack_files if f.name != "packs_registry.json"]

        for pf in tqdm(pack_files, desc="Loading packs", unit="pack"):
            with open(pf, "r", encoding="utf-8") as f:
                pack = json.load(f)
            pack_id = pack["packId"]
            for chunk in pack["chunks"]:
                idx = len(self.chunks)
                self.chunk_id_to_idx[chunk["chunkId"]] = idx
                self.chunks.append({
                    "chunkId": chunk["chunkId"],
                    "text": chunk["text"],
                    "packId": pack_id,
                    "sectionPath": chunk.get("sectionPath", ""),
                })
                all_embeddings.append(chunk["embedding"])

        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize rows (should already be normalized, but ensure)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        print(f"Loaded {len(self.chunks)} chunks from {len(pack_files)} packs.")

    def _build_bm25_index(self):
        """Build BM25 inverted index from chunk text."""
        temp_index: dict[str, list[tuple[int, int]]] = {}

        for i, chunk in enumerate(tqdm(self.chunks, desc="Building BM25 index", unit="chunk")):
            tokens = _tokenize(chunk["text"])
            self._doc_lengths.append(len(tokens))

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for token, count in tf.items():
                if token not in temp_index:
                    temp_index[token] = []
                temp_index[token].append((i, count))

        self._bm25_index = temp_index
        self._avg_dl = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 0.0
        )
        print(f"BM25 index built: {len(self._bm25_index)} unique tokens.")

    def search_vector(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Vector search using cosine similarity. Returns [(chunkId, score)]."""
        if top_k is None:
            top_k = self.vector_top_k

        query_emb = np.array(
            embed_text(self.session, self.tokenizer, query), dtype=np.float32
        )
        # Normalize query embedding
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        # Cosine similarity (both normalized → dot product)
        similarities = self.embeddings @ query_emb  # shape (N,)

        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            (self.chunks[idx]["chunkId"], float(similarities[idx]))
            for idx in top_indices
        ]

    def search_bm25(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """BM25 search. Returns [(chunkId, score)]."""
        if top_k is None:
            top_k = self.bm25_top_k

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self.chunks)
        scores: dict[int, float] = {}

        for qt in query_tokens:
            postings = self._bm25_index.get(qt)
            if postings is None:
                continue

            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, tf in postings:
                dl = self._doc_lengths[doc_idx]
                numerator = tf * (_K1 + 1)
                denominator = tf + _K1 * (1 - _B + _B * dl / self._avg_dl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (self.chunks[idx]["chunkId"], score)
            for idx, score in ranked[:top_k]
        ]

    def _fuse_rrf(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of vector and BM25 results."""
        # Build rank maps (1-based)
        vector_ranks = {cid: i + 1 for i, (cid, _) in enumerate(vector_results)}
        bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(bm25_results)}

        all_ids = set(vector_ranks) | set(bm25_ranks)

        scored: list[tuple[str, float]] = []
        for cid in all_ids:
            rrf = 0.0
            if cid in vector_ranks:
                rrf += 1.0 / (self.rrf_k + vector_ranks[cid])
            if cid in bm25_ranks:
                rrf += 1.0 / (self.rrf_k + bm25_ranks[cid])
            scored.append((cid, rrf))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> list[str]:
        """Run retrieval and return ranked chunk IDs.

        mode: 'hybrid' (RRF fusion), 'vector', or 'bm25'
        """
        if mode == "vector":
            results = self.search_vector(query, top_k=top_k)
            return [cid for cid, _ in results]
        elif mode == "bm25":
            results = self.search_bm25(query, top_k=top_k)
            return [cid for cid, _ in results]
        else:  # hybrid
            vector_results = self.search_vector(query)
            bm25_results = self.search_bm25(query)
            fused = self._fuse_rrf(vector_results, bm25_results, top_k)
            return [cid for cid, _ in fused]
