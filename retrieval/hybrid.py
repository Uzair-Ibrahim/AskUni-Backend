"""
retrieval/hybrid.py
====================
Hybrid retrieval engine combining:
  1. Dense retrieval (FAISS + fastembed)
  2. Sparse retrieval (BM25 keyword matching)
  3. Reciprocal Rank Fusion (RRF) to merge results

This is the core retrieval improvement over the original system.
"""

import os
from typing import List, Optional, Tuple, Dict
from knowledge.chunker import Chunk
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store


# ─── CONFIGURATION ───────────────────────────────────────────────────────────

DENSE_TOP_K = int(os.getenv("RAG_DENSE_TOP_K", "15"))
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "15"))
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "6"))
RRF_K = 60  # Reciprocal Rank Fusion constant


# ─── RECIPROCAL RANK FUSION ──────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Chunk, float]]],
    k: int = RRF_K,
) -> List[Tuple[Chunk, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document d across N rankings:
        RRF(d) = Σ  1 / (k + rank_i(d))

    This gives each ranking equal influence and is robust to score scale differences.
    """
    chunk_scores: Dict[int, Tuple[Chunk, float]] = {}

    for results in result_lists:
        for rank, (chunk, _orig_score) in enumerate(results, start=1):
            chunk_id = id(chunk)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = (chunk, 0.0)

            existing_chunk, existing_score = chunk_scores[chunk_id]
            chunk_scores[chunk_id] = (existing_chunk, existing_score + 1.0 / (k + rank))

    # Sort by RRF score descending
    merged = list(chunk_scores.values())
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


# ─── HYBRID RETRIEVER ────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval with RRF fusion.

    Pipeline:
        Query → [Dense Search] + [BM25 Search]
              → RRF Fusion
              → Top-K final chunks
    """

    def __init__(self, chunks: List[Chunk]):
        print("  [Hybrid] Initializing hybrid retriever ...")
        self._dense = VectorStore(chunks)
        self._bm25 = BM25Store(chunks)
        self._chunks = chunks
        print(f"  [Hybrid] Ready — {len(chunks)} chunks indexed in both dense + BM25")

    def retrieve(
        self,
        query: str,
        search_queries: Optional[List[str]] = None,
        domain_filter: Optional[str] = None,
        top_k: int = FINAL_TOP_K,
    ) -> List[Chunk]:
        """
        Hybrid retrieval pipeline.

        Args:
            query: Primary user query
            search_queries: Additional expanded queries (from query understanding)
            domain_filter: Limit results to a specific domain (faculty, admission, etc.)
            top_k: Number of final chunks to return

        Returns:
            List of Chunk objects, ranked by relevance
        """
        all_queries = [query]
        if search_queries:
            all_queries.extend(q for q in search_queries if q != query)

        # ── Stage 1: Gather candidates from both indices ──────────────────────
        dense_results = []
        bm25_results = []

        for q in all_queries:
            dense_results.extend(
                self._dense.search(q, top_k=DENSE_TOP_K, domain_filter=domain_filter)
            )
            bm25_results.extend(
                self._bm25.search(q, top_k=BM25_TOP_K, domain_filter=domain_filter)
            )

        # Deduplicate within each list
        dense_results = _deduplicate(dense_results)
        bm25_results = _deduplicate(bm25_results)

        # ── Stage 2: Reciprocal Rank Fusion ───────────────────────────────────
        fused = reciprocal_rank_fusion([dense_results, bm25_results])

        if not fused:
            # Fallback: try without domain filter
            if domain_filter:
                return self.retrieve(
                    query=query,
                    search_queries=search_queries,
                    domain_filter=None,
                    top_k=top_k,
                )
            return []

        # ── Stage 3: Return top-k by RRF score ─────────────────────────────────
        return [chunk for chunk, _score in fused[:top_k]]


def _deduplicate(results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
    """Remove duplicate chunks, keeping the highest-scored occurrence."""
    seen = {}
    deduped = []
    for chunk, score in results:
        chunk_id = id(chunk)
        if chunk_id not in seen or score > seen[chunk_id]:
            if chunk_id not in seen:
                deduped.append((chunk, score))
            else:
                # Update score for existing entry
                for i, (c, s) in enumerate(deduped):
                    if id(c) == chunk_id:
                        deduped[i] = (c, score)
                        break
            seen[chunk_id] = score
    return deduped
