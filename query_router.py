"""
query_router.py
===============
Classifies an incoming user query into one of three categories:
    - TIMETABLE   → existing SQL agent handles it
    - FREE_SLOT   → free_slot_service handles it
    - KNOWLEDGE   → RAG pipeline handles it

Uses keyword matching (fast, zero-cost, no LLM call needed for routing).
"""

import re
from enum import Enum, auto


class QueryType(Enum):
    TIMETABLE  = auto()   # class schedule queries
    FREE_SLOT  = auto()   # free/available slot queries
    KNOWLEDGE  = auto()   # faculty info, admissions, fees, etc.


# ─── KEYWORD SETS ─────────────────────────────────────────────────────────────

_TIMETABLE_KEYWORDS = {
    # English
    "schedule", "timetable", "time table", "class", "classes",
    "lecture", "lectures", "room", "subject", "subjects",
    "when is", "which room", "lab", "slot",
    # Roman Urdu
    "timetable", "schedule", "class kab", "kab hai", "kitne baje",
    "room number", "class room",
}

_FREE_SLOT_KEYWORDS = {
    # English
    "free", "available", "free slot", "free time", "no class",
    "empty slot", "when is free", "when free", "availability",
    "off period", "break", "gap",
    # Roman Urdu
    "free kab", "khali", "free hai", "available hai",
    "class nahi", "chhutti",
    # Common typos / short forms
    "fre", "fr", "freee",
}

_FREE_SLOT_REGEX = re.compile(
    r"\b(fre|free|freee|available|khali|class\s+nahi|free\s+kab|kab\s+free)\b",
    re.IGNORECASE,
)

_KNOWLEDGE_KEYWORDS = {
    # Faculty / people
    "who is", "email", "faculty", "professor", "dr.", "sir",
    "contact", "phone", "designation", "department", "research",
    # Admissions / programs
    "admission", "apply", "how to apply", "eligibility",
    "fee", "fees", "scholarship", "program", "programs",
    "degree", "bs", "ms", "phd", "merit", "last date",
    "test", "entry test", "nust", "ecat", "fast", "nuces",
    "university", "campus",
}


def _contains_any(text: str, keywords: set) -> bool:
    """Case-insensitive keyword match against the full query text."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ─── ENTITY EXTRACTORS ────────────────────────────────────────────────────────

_DAY_PATTERN = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|"
    r"mon|tue|wed|thu|fri|sat|kal|aaj|tomorrow|today)\b",
    re.IGNORECASE,
)

_SECTION_PATTERN = re.compile(
    r"\b(BS[A-Z]{1,3}|BBA|MBA|MS[A-Z]{0,3})\s*[-–]?\s*\d[A-Z]?\b",
    re.IGNORECASE,
)

_TEACHER_PREFIXES = re.compile(
    r"\b(sir|dr\.?|miss|ms\.?|mr\.?|prof\.?|professor)\b\s+(\w+)",
    re.IGNORECASE,
)


def extract_day(query: str) -> str:
    """Extract the day mentioned in the query, or empty string."""
    match = _DAY_PATTERN.search(query)
    return match.group(0) if match else ""


def extract_section(query: str) -> str:
    """Extract a section identifier like 'BSCS-5A'."""
    match = _SECTION_PATTERN.search(query)
    return match.group(0).strip() if match else ""


def extract_teacher(query: str) -> str:
    """
    Extract teacher name — looks for prefix (Sir/Dr/Miss) + next word,
    or falls back to returning the whole query for fuzzy DB search.
    """
    match = _TEACHER_PREFIXES.search(query)
    if match:
        # Return prefix + name, e.g. "Sir Shakeel"
        return match.group(0).strip()
    return ""


# ─── ROUTER ───────────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Stateless router.  Call route(query) to get a QueryType.

    Priority order (highest first):
        1. FREE_SLOT   — must check before TIMETABLE (overlapping words like 'slot')
        2. TIMETABLE
        3. KNOWLEDGE   — default fallback
    """

    def route(self, query: str) -> QueryType:
        if _contains_any(query, _FREE_SLOT_KEYWORDS) or _FREE_SLOT_REGEX.search(query):
            return QueryType.FREE_SLOT

        if _contains_any(query, _TIMETABLE_KEYWORDS):
            return QueryType.TIMETABLE

        # Knowledge is the safe fallback — RAG will say "I don't know" if needed
        return QueryType.KNOWLEDGE

    def describe(self, query: str) -> str:
        qt = self.route(query)
        labels = {
            QueryType.TIMETABLE:  "📅 TIMETABLE (SQL agent)",
            QueryType.FREE_SLOT:  "🟢 FREE SLOT (free_slot_service)",
            QueryType.KNOWLEDGE:  "📚 KNOWLEDGE (RAG pipeline)",
        }
        return labels[qt]