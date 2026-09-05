"""
rag_pipeline.py  (v2 — REWRITTEN)
====================================
Redesigned RAG pipeline for FAST NUCES knowledge-base queries.

Key improvements over v1:
  1. Domain-aware semantic chunking (faculty, admissions, programs)
  2. Rich metadata per chunk (name, email, department, domain)
  3. Hybrid retrieval: Dense (FAISS) + Sparse (BM25)
  4. Reciprocal Rank Fusion for merging results
  5. Cross-encoder re-ranking for final precision
  6. Query understanding with intent detection and entity extraction
  7. Domain-specific prompt templates
  8. Better context assembly

Does NOT touch any timetable / SQL logic — that remains in chatbot.py / main.py.
"""

import os
import re
from typing import List, Optional
from enum import Enum, auto

from dotenv import load_dotenv

load_dotenv()


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MD_FILE = os.getenv("RAG_MD_FILE", os.path.join("knowledge", "NUCES_KHI_Complete.md"))
MD_FILE = os.path.join(BASE_DIR, MD_FILE)


# ─── KNOWLEDGE SUB-TYPES ─────────────────────────────────────────────────────

class KnowledgeSubType(Enum):
    FACULTY = auto()
    ADMISSION = auto()
    PROGRAM = auto()
    FEE = auto()
    GENERAL = auto()


# ─── QUERY UNDERSTANDING ─────────────────────────────────────────────────────

# Patterns for detecting query intent
_FACULTY_PATTERN = re.compile(
    r"\b(sir|dr\.?|prof\.?|professor|miss|ms\.?|mr\.?|madam)\s+"
    r"([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})",
    re.IGNORECASE,
)

_FACULTY_KEYWORDS = {
    "who is", "email", "faculty", "professor", "teacher",
    "contact", "phone", "designation", "research", "courses taught",
    "taught by", "teaches", "extension",
}

_ADMISSION_KEYWORDS = {
    "admission", "admissions", "apply", "how to apply", "application",
    "eligibility", "eligible", "criteria", "merit", "merit list",
    "last date", "deadline", "test", "entry test", "test pattern",
    "schedule", "admission schedule", "sat", "gre", "nat", "nts",
    "selected candidates",
}

_FEE_KEYWORDS = {
    "fee", "fees", "fee structure", "tuition", "scholarship",
    "scholarships", "financial aid", "refund", "payment",
    "challan", "bank",
}

_PROGRAM_KEYWORDS = {
    "program", "programmes", "degree", "bs", "ms", "mba", "phd",
    "bba", "bachelor", "master", "doctoral",
    "credit hours", "cgpa", "award of degree",
    "career opportunities", "course", "curriculum",
}

_PROGRAM_PATTERN = re.compile(
    r"\b(BS|MS|MBA|PhD|BBA|B\.?S\.?|M\.?S\.?)\s*[\(\[]?\s*"
    r"([A-Z]{2,}(?:\s+[A-Z]{2,})?)\s*[\)\]]?",
    re.IGNORECASE,
)


def understand_query(query: str) -> dict:
    """
    Analyze a user query to extract intent, entities, domain filter,
    and generate expanded search queries.

    Returns:
        {
            "original": str,
            "sub_type": KnowledgeSubType,
            "domain_filter": str | None,
            "search_queries": List[str],
            "entities": dict,
        }
    """
    result = {
        "original": query,
        "sub_type": KnowledgeSubType.GENERAL,
        "domain_filter": None,
        "search_queries": [query],
        "entities": {},
    }

    lower = query.lower()

    # ── Faculty detection ──
    faculty_match = _FACULTY_PATTERN.search(query)
    is_faculty_query = (
        faculty_match
        or any(kw in lower for kw in _FACULTY_KEYWORDS)
    )

    if is_faculty_query:
        result["sub_type"] = KnowledgeSubType.FACULTY
        result["domain_filter"] = "faculty"

        if faculty_match:
            name = faculty_match.group(0).strip()
            result["entities"]["faculty_name"] = name

            # Add name-focused search queries
            clean_name = re.sub(
                r"\b(sir|dr\.?|prof\.?|professor|miss|ms\.?|mr\.?|madam)\b",
                "", name, flags=re.IGNORECASE,
            ).strip()

            result["search_queries"].extend([
                f"Faculty: {name}",
                clean_name,
                f"### Faculty: {clean_name}",
            ])

        return result

    # ── Fee detection (check before admission — "fee" is also in admission pages) ──
    if any(kw in lower for kw in _FEE_KEYWORDS):
        result["sub_type"] = KnowledgeSubType.FEE
        result["domain_filter"] = "fee"
        # Also search admission domain since fee info overlaps
        result["search_queries"].append(f"fee structure {query}")
        return result

    # ── Admission detection ──
    if any(kw in lower for kw in _ADMISSION_KEYWORDS):
        result["sub_type"] = KnowledgeSubType.ADMISSION
        result["domain_filter"] = "admission"
        result["search_queries"].append(f"admission {query}")
        return result

    # ── Program detection ──
    program_match = _PROGRAM_PATTERN.search(query)
    is_program_query = (
        program_match
        or any(kw in lower for kw in _PROGRAM_KEYWORDS)
    )

    if is_program_query:
        result["sub_type"] = KnowledgeSubType.PROGRAM
        result["domain_filter"] = "program"

        if program_match:
            prog = program_match.group(0).strip()
            result["entities"]["program"] = prog
            result["search_queries"].append(f"Program: {prog}")
            # Also search admission for eligibility
            result["search_queries"].append(f"{prog} eligibility criteria")

        return result

    # ── General — no domain filter, search all ──
    return result


# ─── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

_FACULTY_PROMPT = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.

TASK: Answer the user's question about a FAST NUCES faculty member.

RULES:
1. Answer ONLY from the CONTEXT below. If the answer is not in the context, say exactly:
   "I don't have information about this faculty member in my knowledge base."
2. For faculty queries, always format your response with these fields if available:
   - **Name:** [full name]
   - **Email:** [email]
   - **Designation:** [designation]
   - **Department:** [department]
   - **Extension:** [extension number]
   - [Any additional info the user asked for — biography, courses, research, etc.]
3. NEVER invent or guess emails, phone numbers, extensions, or designations.
4. If multiple faculty members appear in context, pick the EXACT one asked by the user.
5. CRITICAL: If the user asks for a specific name (e.g., "Abdullah Shaikh") and the context only has a completely different person with the same first name (e.g., "Abdullah Siddiqui"), DO NOT assume it's a typo. You MUST state that you do not have information about the requested person.
6. TYPO HANDLING: If the requested name and the name in the context are clearly the same person with a slight spelling mistake (e.g., "Tala Shahid" vs "Talha Shahid", or "Shaikh" vs "Sheikh"), you MAY assume it is a typo and provide the information for the person in the context. However, NEVER mix up completely different last names.

{language_instruction}

--- CHAT HISTORY ---
{chat_history}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""


_ADMISSION_PROMPT = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.

TASK: Answer the user's question about FAST NUCES admissions, eligibility, or application process.

RULES:
1. Answer ONLY from the CONTEXT below. Do not add information not present.
2. For eligibility questions, include ALL criteria: marks, test options, weightages, and any notes.
3. If there are tables in the context, reproduce them accurately in your response.
4. For schedule questions, include all relevant dates.
5. Use bullet points and tables for structured data — make it easy to scan.
6. If you cannot find the answer, say exactly:
   "I don't have this specific information in my knowledge base. Please check nu.edu.pk for the latest details."
7. NEVER make up dates, percentages, or requirements.

{language_instruction}

--- CHAT HISTORY ---
{chat_history}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""


_FEE_PROMPT = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.

TASK: Answer the user's question about FAST NUCES fee structure, scholarships, or payments.

RULES:
1. Answer ONLY from the CONTEXT below.
2. For fee questions, include the complete fee table or breakdown if available.
3. For scholarship questions, include eligibility criteria and coverage details.
4. For payment questions, include all available payment methods.
5. Include any important notes about withholding tax, refund policies, etc.
6. NEVER make up fee amounts or scholarship percentages.

{language_instruction}

--- CHAT HISTORY ---
{chat_history}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""


_PROGRAM_PROMPT = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.

TASK: Answer the user's question about FAST NUCES degree programs.

RULES:
1. Answer ONLY from the CONTEXT below.
2. For program details, include: mission, career opportunities, credit hours, and degree requirements.
3. For "which programs" questions, list all matching programs with their campuses.
4. For prerequisite questions, list all required courses.
5. NEVER make up credit hour requirements, CGPA requirements, or course names.

{language_instruction}

--- CHAT HISTORY ---
{chat_history}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""


_GENERAL_PROMPT = """You are AskUni, the official AI assistant for FAST NUCES Karachi campus.

TASK: Answer the user's question about FAST University.

RULES:
1. Answer ONLY from the CONTEXT below. Do not hallucinate or add information not present.
2. If the answer is clearly in the context, give a concise, helpful response.
3. If the context does not contain the answer, say exactly:
   "I don't have that information in my knowledge base. Please visit nu.edu.pk for more details."
4. Use bullet points where helpful.
5. Keep answers structured and scannable.
6. NEVER make up facts, emails, phone numbers, or statistics.

{language_instruction}

--- CHAT HISTORY ---
{chat_history}

--- CONTEXT FROM KNOWLEDGE BASE ---
{context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---"""


_PROMPT_MAP = {
    KnowledgeSubType.FACULTY: _FACULTY_PROMPT,
    KnowledgeSubType.ADMISSION: _ADMISSION_PROMPT,
    KnowledgeSubType.FEE: _FEE_PROMPT,
    KnowledgeSubType.PROGRAM: _PROGRAM_PROMPT,
    KnowledgeSubType.GENERAL: _GENERAL_PROMPT,
}


# ─── RAG PIPELINE ────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Redesigned RAG pipeline with hybrid retrieval.

    Usage (same interface as v1 — drop-in replacement):
        rag = RAGPipeline(llm)
        answer = rag.ask("What is the email of Dr XYZ?", chat_history)
    """

    def __init__(self, llm):
        self._llm = llm
        self._retriever = None
        self._store = None   # kept for backward compat with health endpoint

    def _ensure_loaded(self):
        if self._retriever is not None:
            return

        print("  [RAG v2] Loading knowledge base ...")

        # Step 1: Load raw markdown
        if not os.path.exists(MD_FILE):
            raise FileNotFoundError(
                f"Knowledge base not found: {MD_FILE}\n"
                "Run scraper.py first to generate it."
            )
        with open(MD_FILE, "r", encoding="utf-8") as f:
            raw = f.read()

        # Step 2: Preprocess into structured documents
        from knowledge.preprocessor import split_into_documents
        documents = split_into_documents(raw)
        print(f"  [RAG v2] Extracted {len(documents)} documents")

        # Log domain distribution
        domains = {}
        for doc in documents:
            d = doc.metadata.get("domain", "unknown")
            domains[d] = domains.get(d, 0) + 1
        print(f"  [RAG v2] Domain distribution: {domains}")

        # Step 3: Chunk documents
        from knowledge.chunker import chunk_documents
        chunks = chunk_documents(documents)
        print(f"  [RAG v2] Created {len(chunks)} chunks")

        # Step 4: Build hybrid retriever
        from retrieval.hybrid import HybridRetriever
        self._retriever = HybridRetriever(chunks)

        # Mark as loaded for health endpoint
        self._store = True

        print(f"  [RAG v2] Pipeline ready! ({len(chunks)} chunks in hybrid index)")

    def ask(self, query: str, chat_history: str = "", language_hint: str = "") -> str:
        """
        Retrieve relevant chunks and ask the LLM to answer grounded on them.
        Returns the LLM's answer string.

        This method has the same signature as v1 — drop-in replacement.
        """
        self._ensure_loaded()

        # ── Step 1: Understand the query ──────────────────────────────────────
        q_info = understand_query(query)

        # ── Step 2: Retrieve relevant chunks ──────────────────────────────────
        chunks = self._retriever.retrieve(
            query=query,
            search_queries=q_info["search_queries"],
            domain_filter=q_info["domain_filter"],
        )

        if not chunks:
            return (
                "Mujhe is sawal ka jawab knowledge base mein nahi mila. "
                "Please verify the name or topic and try again."
            )

        # ── Step 3: Assemble context ──────────────────────────────────────────
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Add source info for grounding
            source = chunk.metadata.get("page_title", "")
            dept = chunk.metadata.get("department", "")
            label = f"[Source: {source}]" if source else ""
            if dept:
                label = f"[Department: {dept}] {label}"

            context_parts.append(f"--- Chunk {i} {label} ---\n{chunk.text}")

        context = "\n\n".join(context_parts)

        # ── Step 4: Select prompt template ────────────────────────────────────
        template = _PROMPT_MAP.get(q_info["sub_type"], _GENERAL_PROMPT)

        language_instruction = _get_language_instruction(language_hint)

        full_prompt = template.format(
            language_instruction=language_instruction,
            chat_history=chat_history if chat_history else "(no previous conversation)",
            context=context,
            query=query,
        )

        # ── Step 5: Call LLM ──────────────────────────────────────────────────
        try:
            if hasattr(self._llm, "invoke"):
                response = self._llm.invoke(full_prompt)
            else:
                response = self._llm(full_prompt)

            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            return f"[RAG Error] LLM call failed: {e}"


def _get_language_instruction(hint: str) -> str:
    """Generate the language instruction based on detected language."""
    if hint == "roman_urdu":
        return (
            "LANGUAGE: The user is writing in Roman Urdu. Reply in Roman Urdu "
            "using Latin/English letters only (NO Urdu/Arabic script). "
            "You may use English technical terms where appropriate."
        )
    return (
        "LANGUAGE: Reply in English. If the user switches to Roman Urdu, "
        "switch to Roman Urdu using Latin letters only."
    )