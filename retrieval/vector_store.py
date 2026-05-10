"""
retrieval/vector_store.py
==========================
FAISS-based vector store with metadata support.
Stores chunk embeddings and allows filtered search.
Uses FastEmbed (ONNX) to avoid PyTorch memory bloat (~400MB saved).
"""

import os
from typing import List, Optional, Tuple
from knowledge.chunker import Chunk


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv(
    "RAG_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def _import_deps():
    try:
        from fastembed import TextEmbedding
        import faiss
        import numpy as np
        return TextEmbedding, faiss, np
    except ImportError as e:
        raise ImportError(
            f"RAG dependency missing: {e}\n"
            "Run: pip install fastembed faiss-cpu numpy"
        ) from e


class VectorStore:
    """
    In-memory FAISS index with metadata-aware search.

    Each chunk is stored with its full metadata dict,
    enabling filtered search by domain, faculty name, etc.
    """

    def __init__(self, chunks: List[Chunk], model_name: str = EMBED_MODEL):
        TextEmbedding, faiss, np = _import_deps()

        print(f"  [VectorStore] Loading FastEmbed model (ONNX): {model_name}")
        self._model = TextEmbedding(model_name=model_name, threads=1)
        self._chunks = chunks
        self._np = np
        self._faiss = faiss

        texts = [c.text for c in chunks]
        print(f"  [VectorStore] Embedding {len(texts)} chunks ...")
        
        # fastembed returns a generator of numpy arrays
        # Use a small batch_size to prevent memory spikes on 1GB RAM
        embeddings_list = list(self._model.embed(texts, batch_size=16))
        
        # Stack into a single 2D matrix
        embeddings = np.vstack(embeddings_list).astype(np.float32)

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

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
        """
        search_k = top_k * 3 if domain_filter else top_k

        # Embed single query
        q_vec = list(self._model.embed([query]))[0].astype(self._np.float32)
        q_vec = q_vec.reshape(1, -1)
        
        # Normalize
        norm = self._np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

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
        seen = set()
        all_results = []

        for query in queries:
            results = self.search(query, top_k=top_k, domain_filter=domain_filter, min_score=min_score)
            for chunk, score in results:
                chunk_id = id(chunk)
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    all_results.append((chunk, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def encode_query(self, query: str):
        """Encode a query for external use."""
        q_vec = list(self._model.embed([query]))[0].astype(self._np.float32)
        norm = self._np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return q_vec
