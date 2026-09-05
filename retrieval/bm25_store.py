"""
retrieval/bm25_store.py
========================
BM25 keyword-based retrieval index.
Complements dense (vector) search for exact keyword / name matches.

Uses the lightweight `rank_bm25` library.
"""

import re
import math
from typing import List, Optional, Tuple
from knowledge.chunker import Chunk


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    # Keep email-like tokens intact
    tokens = re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]+|[a-z0-9]+", text)
    return tokens


class BM25Store:
    """
    In-memory BM25 index over text chunks.
    Provides keyword-based retrieval that catches exact matches
    where dense/semantic search fails.
    """

    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._corpus = [_tokenize(c.text) for c in chunks]

        # Compute BM25 statistics
        self._avg_dl = sum(len(doc) for doc in self._corpus) / max(len(self._corpus), 1)
        self._doc_count = len(self._corpus)

        # Inverse document frequency
        self._idf = {}
        df = {}
        for doc in self._corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1
        for term, freq in df.items():
            self._idf[term] = math.log((self._doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

        print(f"  [BM25] Index ready — {len(chunks)} documents, {len(self._idf)} unique terms")

    def search(
        self,
        query: str,
        top_k: int = 15,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for chunks using BM25 scoring.

        Args:
            query: The search query text
            top_k: Number of results to return
            domain_filter: If set, only return chunks from this domain

        Returns:
            List of (Chunk, score) tuples, sorted by score descending
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores = []

        for idx, doc_tokens in enumerate(self._corpus):
            # Apply domain filter early to skip unnecessary computation
            if domain_filter and self._chunks[idx].metadata.get("domain") != domain_filter:
                continue

            doc_len = len(doc_tokens)
            score = 0.0

            # Count term frequencies in this document
            tf_map = {}
            for token in doc_tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for q_term in query_tokens:
                if q_term not in self._idf:
                    continue
                tf = tf_map.get(q_term, 0)
                if tf == 0:
                    continue
                idf = self._idf[q_term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append((self._chunks[idx], score))

        return results
