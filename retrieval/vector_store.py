"""
retrieval/vector_store.py
==========================
FAISS-based vector store with metadata support.
Stores chunk embeddings and allows filtered search.
"""

import os
from typing import List, Optional, Tuple
from knowledge.chunker import Chunk


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"   # smaller, faster to download/load
)


def _import_deps():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        return SentenceTransformer, faiss, np
    except ImportError as e:
        raise ImportError(
            f"RAG dependency missing: {e}\n"
            "Run: pip install sentence-transformers faiss-cpu numpy"
        ) from e


class VectorStore:
    """
    In-memory FAISS index with metadata-aware search.

    Each chunk is stored with its full metadata dict,
    enabling filtered search by domain, faculty name, etc.
    """

    def __init__(self, chunks: List[Chunk], model_name: str = EMBED_MODEL):
        SentenceTransformer, faiss, np = _import_deps()

        print(f"  [VectorStore] Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._chunks = chunks
        self._np = np
        self._faiss = faiss

        texts = [c.text for c in chunks]
        print(f"  [VectorStore] Embedding {len(texts)} chunks ...")
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine via inner product
            batch_size=16,
        )

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"  [VectorStore] Index ready — {self._index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = 15,
        domain_filter: Optional[str] = None,
        min_score: float = 0.20,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for chunks relevant to the query.

        Args:
            query: The search query text
            top_k: Number of results to return
            domain_filter: If set, only return chunks from this domain
            min_score: Minimum cosine similarity threshold

        Returns:
            List of (Chunk, score) tuples, sorted by score descending
        """
        # If domain filter is active, we search more candidates then filter
        search_k = top_k * 3 if domain_filter else top_k

        q_vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores, indices = self._index.search(q_vec, min(search_k, len(self._chunks)))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1 or score < min_score:
                continue

            chunk = self._chunks[idx]

            # Apply domain filter
            if domain_filter and chunk.metadata.get("domain") != domain_filter:
                continue

            results.append((chunk, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_multiple(
        self,
        queries: List[str],
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        min_score: float = 0.20,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search with multiple query variants and merge results.
        Useful when query expansion generates several search strings.
        """
        seen = set()
        all_results = []

        for query in queries:
            results = self.search(query, top_k=top_k, domain_filter=domain_filter, min_score=min_score)
            for chunk, score in results:
                chunk_id = id(chunk)
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    all_results.append((chunk, score))

        # Sort by score descending
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def encode_query(self, query: str):
        """Encode a query for external use (e.g., hybrid retrieval)."""
        return self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
