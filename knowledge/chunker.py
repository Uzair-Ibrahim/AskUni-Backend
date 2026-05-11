"""
knowledge/chunker.py
=====================
Domain-aware semantic chunking that respects Markdown structure.

Key rules:
  - Faculty profiles → one chunk each (never split)
  - Tables → keep whole with header context (never split mid-row)
  - Narrative → split by ### sub-headers, preserve parent heading
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from knowledge.preprocessor import Document


@dataclass
class Chunk:
    """A retrieval-ready text chunk with rich metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── CHUNKING PARAMETERS ─────────────────────────────────────────────────────

MAX_CHUNK_CHARS = 2000      # Maximum characters per chunk
MIN_CHUNK_CHARS = 150       # Minimum — below this, merge with neighbor
TARGET_CHUNK_CHARS = 1200   # Ideal target for narrative splits


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Chunk]:
    """
    Convert structured documents into retrieval-ready chunks.

    Strategy varies by domain / content type:
      - faculty profiles → keep as one chunk
      - table-heavy pages → split preserving complete tables
      - narrative pages → split by sub-headers
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        domain = doc.metadata.get("domain", "general")
        content_type = doc.metadata.get("content_type", "narrative")

        if domain == "faculty":
            chunks = _chunk_faculty(doc)
        elif content_type == "table_heavy":
            chunks = _chunk_table_heavy(doc)
        else:
            chunks = _chunk_narrative(doc)

        all_chunks.extend(chunks)

    # Filter out empty / too-short chunks, but NEVER filter out faculty profiles
    filtered_chunks = []
    for c in all_chunks:
        if c.metadata.get("domain") == "faculty":
            filtered_chunks.append(c)
        elif len(c.text.strip()) >= MIN_CHUNK_CHARS:
            filtered_chunks.append(c)

    return filtered_chunks


# ─── FACULTY CHUNKING ────────────────────────────────────────────────────────

def _chunk_faculty(doc: Document) -> List[Chunk]:
    """
    Faculty profile → one chunk.
    Most profiles are 500-2000 chars, well within limits.
    Only split if extraordinarily long (>3000 chars, rare).
    """
    text = doc.content.strip()

    if len(text) <= MAX_CHUNK_CHARS * 1.5:
        # Keep entire profile as one chunk
        return [Chunk(text=text, metadata=dict(doc.metadata))]

    # Exceptionally long profile — split by ##### sub-headers
    # but prepend the core info (name, email, designation) to each chunk
    core_info = _extract_faculty_core(text)
    sections = re.split(r"\n(?=##### )", text)

    chunks = []
    current = core_info
    for section in sections:
        if section.startswith("### Faculty:") or section.startswith("- **Email"):
            current = section
            continue
        if len(current) + len(section) <= MAX_CHUNK_CHARS:
            current += "\n\n" + section
        else:
            if current.strip():
                chunks.append(Chunk(
                    text=current.strip(),
                    metadata=dict(doc.metadata)
                ))
            current = core_info + "\n\n" + section

    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata=dict(doc.metadata)))

    return chunks if chunks else [Chunk(text=text[:MAX_CHUNK_CHARS], metadata=dict(doc.metadata))]


def _extract_faculty_core(text: str) -> str:
    """Extract the first few lines of a faculty profile (name, email, designation)."""
    lines = text.splitlines()
    core_lines = []
    for line in lines:
        core_lines.append(line)
        if line.startswith("- **Designation:"):
            break
        if len(core_lines) > 6:
            break
    return "\n".join(core_lines)


# ─── TABLE-HEAVY CHUNKING ────────────────────────────────────────────────────

def _chunk_table_heavy(doc: Document) -> List[Chunk]:
    """
    For pages with tables (eligibility, fees, programs).
    Split by ### sub-headers, keeping each table intact with its heading.
    """
    text = doc.content.strip()
    sections = _split_by_markdown_headers(text, level=3)

    chunks = []
    parent_heading = doc.metadata.get("page_title", "")

    for section_title, section_body in sections:
        full_text = section_body.strip()

        # If section has a table, keep it whole even if large
        has_table = "| --- |" in full_text

        if has_table:
            # Keep entire section (table + surrounding text) as one chunk
            chunk_text = f"[{parent_heading}]\n\n{full_text}" if parent_heading else full_text
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "section_title": section_title,
                    "content_type": "table",
                }
            ))
        elif len(full_text) <= MAX_CHUNK_CHARS:
            chunk_text = f"[{parent_heading}]\n\n{full_text}" if parent_heading else full_text
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "section_title": section_title,
                }
            ))
        else:
            # Long narrative sub-section — split by paragraphs
            sub_chunks = _split_by_paragraphs(full_text, parent_heading, section_title, doc.metadata)
            chunks.extend(sub_chunks)

    return chunks


# ─── NARRATIVE CHUNKING ──────────────────────────────────────────────────────

def _chunk_narrative(doc: Document) -> List[Chunk]:
    """
    For narrative pages (home, how to apply, general info).
    Split by ### sub-headers, then by paragraphs if still too long.
    """
    text = doc.content.strip()
    sections = _split_by_markdown_headers(text, level=3)

    chunks = []
    parent_heading = doc.metadata.get("page_title", "")

    for section_title, section_body in sections:
        full_text = section_body.strip()

        if len(full_text) <= MAX_CHUNK_CHARS:
            chunk_text = f"[{parent_heading}]\n\n{full_text}" if parent_heading else full_text
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "section_title": section_title,
                }
            ))
        else:
            # Further split by ###### sub-sub-headers or paragraphs
            sub_sections = _split_by_markdown_headers(full_text, level=6)
            if len(sub_sections) > 1:
                for sub_title, sub_body in sub_sections:
                    if len(sub_body.strip()) < MIN_CHUNK_CHARS:
                        continue
                    chunk_text = f"[{parent_heading} > {section_title}]\n\n{sub_body.strip()}"
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "section_title": f"{section_title} > {sub_title}",
                        }
                    ))
            else:
                sub_chunks = _split_by_paragraphs(
                    full_text, parent_heading, section_title, doc.metadata
                )
                chunks.extend(sub_chunks)

    return chunks


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _split_by_markdown_headers(text: str, level: int = 3) -> List[tuple]:
    """
    Split text by Markdown headers of the given level (e.g., ### for level=3).
    Returns list of (header_text, section_body) tuples.
    The content before the first header gets header_text = "".
    """
    # Match headers at the specified level or below
    prefix = "#" * level
    # Pattern: match lines starting with exactly `level` #s followed by a space
    pattern = re.compile(rf"^({prefix}{{1,3}}\s+.+)$", re.MULTILINE)
    parts = pattern.split(text)

    sections = []
    if parts[0].strip():
        sections.append(("", parts[0].strip()))

    it = iter(parts[1:])
    for header in it:
        body = next(it, "")
        header_text = re.sub(r"^#+\s+", "", header.strip())
        sections.append((header_text, f"{header}\n\n{body}".strip()))

    return sections if sections else [("", text)]


def _split_by_paragraphs(
    text: str,
    parent_heading: str,
    section_title: str,
    base_metadata: Dict[str, Any],
) -> List[Chunk]:
    """
    Split long text by paragraph boundaries, merging small paragraphs together.
    Each chunk gets the parent heading prepended for context.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= TARGET_CHUNK_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunk_text = f"[{parent_heading} > {section_title}]\n\n{current}"
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "section_title": section_title,
                    }
                ))
            current = para

    if current:
        chunk_text = f"[{parent_heading} > {section_title}]\n\n{current}"
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **base_metadata,
                "section_title": section_title,
            }
        ))

    return chunks
