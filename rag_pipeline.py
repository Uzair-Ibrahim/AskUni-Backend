"""
rag_pipeline.py
===============
Loads NUCES_KHI_Complete.md, chunks it intelligently, and provides
a single ask() function that retrieves relevant context and generates
a grounded answer using the LLM.

Does NOT touch any timetable / SQL logic.
"""

import os
import re
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MD_FILE        = os.getenv("RAG_MD_FILE", "NUCES_KHI_Complete.md")
CHUNK_SIZE     = int(os.getenv("RAG_CHUNK_SIZE", "600"))      # characters
CHUNK_OVERLAP  = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))   # characters
TOP_K          = int(os.getenv("RAG_TOP_K", "4"))             # chunks to retrieve
EMBED_MODEL    = os.getenv("RAG_EMBED_MODEL",
                           "sentence-transformers/all-MiniLM-L6-v2")


# ─── LAZY IMPORTS (so missing packages give a clear error only when RAG is used)

def _import_rag_deps():
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


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_markdown(path: str) -> str:
    """Read the markdown knowledge-base file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Knowledge base not found: {path}\n"
            "Run scraper.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Normalize whitespace and remove markdown noise."""
    # Remove HTML-style comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove bare horizontal rules
    text = re.sub(r"\n---+\n", "\n", text)
    # Strip leading/trailing whitespace per line
    lines = [l.rstrip() for l in text.splitlines()]
    return "\n".join(lines).strip()


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def _split_by_faculty(text: str) -> List[Tuple[str, str]]:
    """
    Split the document on '### Faculty:' sections.
    Returns list of (section_header, section_body).
    Each faculty profile stays together as one semantic unit.
    """
    # Pattern: lines starting with ### Faculty:
    pattern = re.compile(r"(^### Faculty:.+)$", re.MULTILINE)
    parts   = pattern.split(text)

    chunks = []
    # parts = [pre, header1, body1, header2, body2, ...]
    if parts[0].strip():
        chunks.append(("intro", parts[0].strip()))

    it = iter(parts[1:])
    for header in it:
        body = next(it, "")
        chunks.append((header.strip(), body.strip()))

    return chunks


def _split_by_section(text: str, size: int, overlap: int) -> List[str]:
    """
    Generic sliding-window character splitter, respecting paragraph boundaries.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks, current = [], ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from previous
            if current and overlap > 0:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def build_chunks(text: str) -> List[str]:
    """
    Two-pass chunking strategy:
    1. Faculty sections → each profile is one chunk (preserves entity coherence).
    2. Non-faculty sections → sliding-window split.
    """
    cleaned = clean_text(text)

    # Separate faculty section from everything else
    faculty_start = cleaned.find("### Faculty:")
    if faculty_start == -1:
        # No faculty sections – just do generic split
        return _split_by_section(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

    preamble       = cleaned[:faculty_start]
    faculty_block  = cleaned[faculty_start:]

    # Generic chunks from preamble (university info, admissions, etc.)
    preamble_chunks = _split_by_section(preamble, CHUNK_SIZE, CHUNK_OVERLAP)

    # One chunk per faculty profile
    faculty_pairs   = _split_by_faculty(faculty_block)
    faculty_chunks  = []
    for header, body in faculty_pairs:
        combined = f"{header}\n\n{body}".strip()
        if len(combined) > CHUNK_SIZE * 3:
            # Very long profile – sub-split
            faculty_chunks.extend(
                _split_by_section(combined, CHUNK_SIZE, CHUNK_OVERLAP)
            )
        else:
            faculty_chunks.append(combined)

    all_chunks = [c for c in preamble_chunks + faculty_chunks if c.strip()]
    return all_chunks


# ─── VECTOR STORE ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS index over text chunks.
    Rebuilt once on startup; no disk persistence needed for this scale.
    (~50-200 chunks from the markdown file loads in < 10 seconds)
    """

    def __init__(self, chunks: List[str], model_name: str = EMBED_MODEL):
        SentenceTransformer, faiss, np = _import_rag_deps()

        print(f"  [RAG] Loading embedding model: {model_name}")
        self._model  = SentenceTransformer(model_name)
        self._chunks = chunks
        self._np     = np
        self._faiss  = faiss

        print(f"  [RAG] Embedding {len(chunks)} chunks …")
        embeddings = self._model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via inner product
        )

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner Product on L2-normed = cosine
        self._index.add(embeddings)
        print(f"  [RAG] Index ready — {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = TOP_K) -> List[str]:
        """Return top_k most relevant chunks for the query."""
        q_vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores, indices = self._index.search(q_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score > 0.15:   # minimum similarity threshold
                results.append(self._chunks[idx])
        return results


# ─── RAG PIPELINE ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Single entry point for knowledge-base queries.
    Usage:
        rag = RAGPipeline(llm)
        answer = rag.ask("What is the email of Dr XYZ?", chat_history)
    """

    def __init__(self, llm):
        self._llm   = llm
        self._store = None

    def _ensure_loaded(self):
        if self._store is not None:
            return
        print("  [RAG] Loading knowledge base …")
        raw    = load_markdown(MD_FILE)
        chunks = build_chunks(raw)
        self._store = VectorStore(chunks)
        print(f"  [RAG] Pipeline ready ({len(chunks)} chunks)")

    def ask(self, query: str, chat_history: str = "", language_hint: str = "") -> str:
        """
        Retrieve relevant chunks and ask the LLM to answer grounded on them.
        Returns the LLM's answer string.
        """
        self._ensure_loaded()

        # Retrieve context
        relevant_chunks = self._store.search(query, top_k=TOP_K)

        if not relevant_chunks:
            return (
                "Mujhe is sawal ka jawab knowledge base mein nahi mila. "
                "Please verify the faculty name or topic and try again."
            )

        context = "\n\n---\n\n".join(relevant_chunks)

        system_prompt = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.
Answer ONLY using the provided CONTEXT. Do not hallucinate or add information not present in context.

Rules:
1. If the answer is clearly in the context, give a concise, helpful response.
2. If context does not contain the answer, say: "I don't have that information in my knowledge base."
3. For faculty queries: always include Name, Email, Designation if available.
    4. Reply in English if the user writes English; otherwise reply in Roman Urdu using Latin letters only (no Urdu/Arabic script).
5. Keep answers short and structured (use bullet points where helpful).
6. Never make up emails, phone numbers, or designations.
"""

        full_prompt = f"""{system_prompt}

    Language preference: {language_hint or "auto"} (English or Roman Urdu in Latin letters only).

--- CHAT HISTORY ---
{chat_history if chat_history else "(no previous conversation)"}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""

        try:
            if hasattr(self._llm, "invoke"):
                response = self._llm.invoke(full_prompt)
            else:
                response = self._llm(full_prompt)

            # LangChain LLMs return an AIMessage; .content gives the string
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"[RAG Error] LLM call failed: {e}"