"""
knowledge/preprocessor.py
==========================
Splits the monolithic NUCES_KHI_Complete.md into structured documents
with domain labels before chunking happens.

Does NOT modify the source file — works purely in memory.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Document:
    """A semantically coherent document extracted from the Markdown KB."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── SCRAPER ARTIFACT CLEANING ───────────────────────────────────────────────

# Patterns left behind by the scraper that waste embedding space
_SCRAPER_NOISE = [
    re.compile(r"PAGE WRAPPER.*?\.page-title-content", re.IGNORECASE),
    re.compile(r"BUTTON BACK TO TOP", re.IGNORECASE),
    re.compile(r"Wrapper For Slides.*?End of Wrapper For Slides", re.DOTALL | re.IGNORECASE),
    re.compile(r"<div[^>]*>.*?</div>", re.DOTALL),
    re.compile(r"<[^>]+>"),                          # any remaining HTML tags
    re.compile(r"<!--.*?-->", re.DOTALL),             # HTML comments
    re.compile(r"\bLeft Control.*?Right Control.*?\n", re.IGNORECASE),
    re.compile(r"(?:End of )?(?:Slide|COEE Slider|bootstrap-touch-slider).*?\n", re.IGNORECASE),
    re.compile(r"Or use any fixed height\s*", re.IGNORECASE),
    re.compile(r"Slide Background\s*", re.IGNORECASE),
    re.compile(r"Slide Text Layer\s*", re.IGNORECASE),
    re.compile(r"(?:1st|2nd|3rd|4th|5th|6th) Slide\s*", re.IGNORECASE),
    re.compile(r"PANEL\s+(?:NEWS|TOP COURSES).*?\n", re.IGNORECASE),
    re.compile(r"Old Content (?:Starting|Ending) Here\s*", re.IGNORECASE),
    re.compile(r"MS\([A-Z]{2,}\) New Eligibility Criteria (?:Starting|Ending) Here\s*", re.IGNORECASE),
    re.compile(r"MAIN CONTENT CONTENT\s*", re.IGNORECASE),
    re.compile(r"SLIDER BANNER Documentary\s*", re.IGNORECASE),
    re.compile(r"‹\s*›\s*", re.IGNORECASE),
    re.compile(r"Previous\s+Next\s*", re.IGNORECASE),
    re.compile(r"Indicators\s*$", re.MULTILINE | re.IGNORECASE),
]


def clean_markdown(text: str) -> str:
    """Remove scraper artifacts, HTML remnants, and normalize whitespace."""
    for pattern in _SCRAPER_NOISE:
        text = pattern.sub("", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ─── DOMAIN SPLITTING ────────────────────────────────────────────────────────

def _extract_url(text: str) -> str:
    """Pull the **URL:** line from the start of a page section."""
    match = re.search(r"\*\*URL:\*\*\s*(https?://\S+)", text)
    return match.group(1) if match else ""


def _classify_page(title: str, url: str) -> str:
    """Classify a page into a domain based on its title and URL."""
    t = title.lower()
    u = url.lower()
    if any(kw in t for kw in ("admission", "how to apply", "eligibility", "schedule")):
        return "admission"
    if "fee" in t or "scholarship" in t:
        return "fee"
    if "degree program" in t or "program" in t:
        return "program"
    if "test pattern" in t:
        return "admission"
    if "admissions" in u:
        return "admission"
    return "general"


def split_into_documents(raw_markdown: str) -> List[Document]:
    """
    Split the monolithic Markdown into structured documents.

    Strategy:
      1. Split into Part 1 (university info) and Part 2 (departments & faculty)
      2. Part 1 → split by '## Page:' headers
      3. Part 2 → split by '### Faculty:' headers (each = one doc)
         Also extract department sections
    """
    cleaned = clean_markdown(raw_markdown)
    docs: List[Document] = []

    # ── Split Part 1 vs Part 2 ────────────────────────────────────────────────
    part2_marker = "# Part 2: Departments & Faculty Profiles"
    part2_idx = cleaned.find(part2_marker)

    if part2_idx == -1:
        # No Part 2 — treat everything as Part 1
        part1 = cleaned
        part2 = ""
    else:
        part1 = cleaned[:part2_idx]
        part2 = cleaned[part2_idx:]

    # ── Part 1: Split by '## Page:' ──────────────────────────────────────────
    page_pattern = re.compile(r"^## Page:\s*(.+)$", re.MULTILINE)
    page_splits = page_pattern.split(part1)

    # page_splits = [preamble, title1, body1, title2, body2, ...]
    # The preamble (before first ## Page:) is the header
    if page_splits[0].strip():
        docs.append(Document(
            content=page_splits[0].strip(),
            metadata={
                "domain": "general",
                "page_title": "University Overview",
                "section_title": "",
                "source_url": "",
                "content_type": "narrative",
            }
        ))

    it = iter(page_splits[1:])
    for title in it:
        body = next(it, "")
        title = title.strip()
        full_content = f"## {title}\n\n{body.strip()}"
        url = _extract_url(body)
        domain = _classify_page(title, url)

        # Detect if page is table-heavy
        table_count = body.count("| --- |")
        content_type = "table_heavy" if table_count >= 2 else "narrative"

        docs.append(Document(
            content=full_content,
            metadata={
                "domain": domain,
                "page_title": title,
                "section_title": "",
                "source_url": url,
                "content_type": content_type,
            }
        ))

    # ── Part 2: Departments and Faculty ──────────────────────────────────────
    if part2:
        _extract_faculty_documents(part2, docs)

    return docs


def _extract_faculty_documents(part2_text: str, docs: List[Document]):
    """
    Extract faculty profiles and department overviews from Part 2.
    Each '### Faculty:' block becomes one document.
    Department headers and overviews become separate documents.
    """
    current_department = ""
    current_dept_url = ""

    # Split by department headers
    dept_pattern = re.compile(r"^## Department:\s*(.+)$", re.MULTILINE)
    dept_splits = dept_pattern.split(part2_text)

    # dept_splits = [preamble, dept1_name, dept1_body, dept2_name, dept2_body, ...]
    it = iter(dept_splits[1:])
    for dept_name in it:
        dept_body = next(it, "")
        dept_name = dept_name.strip()
        dept_url_match = re.search(r"\*\*URL:\*\*\s*(https?://\S+)", dept_body)
        dept_url = dept_url_match.group(1) if dept_url_match else ""

        # Extract department overview (text before first ### Faculty:)
        faculty_start = dept_body.find("### Faculty:")
        if faculty_start > 0:
            overview = dept_body[:faculty_start].strip()
            if len(overview) > 100:  # Only if there's meaningful content
                docs.append(Document(
                    content=f"## Department: {dept_name}\n\n{overview}",
                    metadata={
                        "domain": "department",
                        "page_title": f"Department of {dept_name}",
                        "section_title": "Department Overview",
                        "source_url": dept_url,
                        "content_type": "narrative",
                        "department": dept_name,
                    }
                ))

        # Extract individual faculty profiles
        faculty_pattern = re.compile(r"(^### Faculty:\s*.+)$", re.MULTILINE)
        faculty_splits = faculty_pattern.split(dept_body)

        fit = iter(faculty_splits[1:] if faculty_splits else [])
        for fac_header in fit:
            fac_body = next(fit, "")
            fac_header = fac_header.strip()

            # Extract faculty name from header
            name_match = re.match(r"### Faculty:\s*(.+)", fac_header)
            name = name_match.group(1).strip() if name_match else "Unknown"

            full_text = f"{fac_header}\n\n{fac_body.strip()}"

            # Extract email
            email_match = re.search(
                r"\*\*Email:\*\*\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                full_text
            )
            email = email_match.group(1).lower() if email_match else ""

            # Extract designation
            desig_match = re.search(r"\*\*Designation:\*\*\s*(.+)", full_text)
            designation = desig_match.group(1).strip() if desig_match else ""

            # Extract extension
            ext_match = re.search(r"\*\*Extension:\*\*\s*(\d+)", full_text)
            extension = ext_match.group(1) if ext_match else ""

            # Build searchable keywords from the profile
            keywords = _extract_faculty_keywords(name, full_text)

            docs.append(Document(
                content=full_text,
                metadata={
                    "domain": "faculty",
                    "page_title": f"Faculty: {name}",
                    "section_title": "",
                    "source_url": "",
                    "content_type": "profile",
                    "faculty_name": name,
                    "faculty_email": email,
                    "department": dept_name,
                    "designation": designation,
                    "extension": extension,
                    "keywords": keywords,
                }
            ))


def _extract_faculty_keywords(name: str, text: str) -> List[str]:
    """Extract searchable keywords from a faculty profile."""
    keywords = []

    # Name variants
    name_parts = re.sub(r"\b(Dr\.|Prof\.|PhD|Engr\.?)\b", "", name, flags=re.IGNORECASE)
    name_parts = name_parts.strip().strip(",").strip()
    keywords.append(name_parts.lower())
    for part in name_parts.split():
        if len(part) > 2:
            keywords.append(part.lower())

    # Research areas
    research_match = re.search(
        r"(?:research\s+interest|area|specializ)[^.]*?(?:include|are|in)[:\s]*([^.]+)",
        text, re.IGNORECASE
    )
    if research_match:
        keywords.extend(
            w.strip().lower() for w in research_match.group(1).split(",") if len(w.strip()) > 2
        )

    # Courses taught
    courses_section = re.search(r"Courses Taught\s*\n((?:- .+\n?)+)", text)
    if courses_section:
        for line in courses_section.group(1).splitlines():
            course = line.lstrip("- ").strip()
            if course:
                keywords.append(course.lower())

    return list(set(keywords))
