"""
test_rag_pipeline.py
=====================
Quick validation script to test the RAG v2 preprocessing, chunking,
and retrieval without needing LLM API keys.

Usage:
    python test_rag_pipeline.py
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    print("=" * 60)
    print("  AskUni RAG v2 -- Pipeline Validation")
    print("=" * 60)

    # -- Step 1: Load Markdown
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md_file = os.getenv("RAG_MD_FILE", os.path.join("knowledge", "NUCES_KHI_Complete.md"))
    md_file = os.path.join(base_dir, md_file)
    if not os.path.exists(md_file):
        print(f"[X] Knowledge base not found: {md_file}")
        sys.exit(1)

    with open(md_file, "r", encoding="utf-8") as f:
        raw = f.read()
    print(f"\n[OK] Loaded {md_file} -- {len(raw):,} characters, {raw.count(chr(10)):,} lines")

    # -- Step 2: Preprocess into documents
    from knowledge.preprocessor import split_into_documents
    documents = split_into_documents(raw)
    print(f"\n[OK] Extracted {len(documents)} documents")

    # Domain distribution
    domains = {}
    for doc in documents:
        d = doc.metadata.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print(f"   Domain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"     {domain}: {count}")

    # Show a few examples
    print(f"\n   Sample documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"\n  [{i+1}] {doc.metadata.get('page_title', 'N/A')}")
        print(f"      Domain: {doc.metadata.get('domain')}")
        print(f"      Content: {doc.content[:120]}...")

    # Show a faculty document
    faculty_docs = [d for d in documents if d.metadata.get("domain") == "faculty"]
    if faculty_docs:
        fac = faculty_docs[0]
        print(f"\n  [Faculty] {fac.metadata.get('faculty_name', 'N/A')}")
        print(f"      Email: {fac.metadata.get('faculty_email', 'N/A')}")
        print(f"      Department: {fac.metadata.get('department', 'N/A')}")
        print(f"      Designation: {fac.metadata.get('designation', 'N/A')}")
        print(f"      Content: {fac.content[:150]}...")

    # -- Step 3: Chunk documents
    from knowledge.chunker import chunk_documents
    chunks = chunk_documents(documents)
    print(f"\n[OK] Created {len(chunks)} chunks")

    # Size distribution
    sizes = [len(c.text) for c in chunks]
    print(f"   Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    # Domain distribution of chunks
    chunk_domains = {}
    for c in chunks:
        d = c.metadata.get("domain", "unknown")
        chunk_domains[d] = chunk_domains.get(d, 0) + 1
    print(f"   Chunk domain distribution:")
    for domain, count in sorted(chunk_domains.items()):
        print(f"     {domain}: {count}")

    # -- Step 4: Test query understanding
    print(f"\n--- Testing query understanding ---")
    from app.services.rag_pipeline import understand_query

    test_queries = [
        "Who is Dr. Shakil Ahmed?",
        "What is the email of Sir Talha Shahid?",
        "What is the eligibility criteria for BS CS?",
        "How to apply for admission?",
        "What is the fee structure?",
        "How many programs does FAST offer?",
        "Tell me about FAST University",
        "What campuses are available?",
        "Research interests of Dr. Fahad Samad",
        "What is the FAST entry test pattern?",
    ]

    for q in test_queries:
        info = understand_query(q)
        entities = info.get('entities', {})
        ent_str = f" | Entities: {entities}" if entities else ""
        print(f"  Q: {q}")
        print(f"     -> {info['sub_type'].name} (filter: {info.get('domain_filter', 'None')}){ent_str}")

    # -- Step 5: Test BM25 retrieval (no embedding model needed)
    print(f"\n--- Testing BM25 retrieval ---")
    from retrieval.bm25_store import BM25Store
    bm25 = BM25Store(chunks)

    bm25_queries = [
        ("Dr. Fahad Samad", "faculty"),
        ("eligibility criteria BS", None),
        ("fee structure", None),
        ("nadeem.kafi@nu.edu.pk", "faculty"),
        ("How to apply", None),
    ]

    for q, domain in bm25_queries:
        results = bm25.search(q, top_k=3, domain_filter=domain)
        print(f"\n  Q: '{q}' (domain={domain})")
        for i, (chunk, score) in enumerate(results):
            title = chunk.metadata.get("page_title", "N/A")
            print(f"    [{i+1}] score={score:.3f} | {title} | {chunk.text[:80]}...")

    # -- Step 6: Test section regex fix
    print(f"\n--- Testing section regex ---")
    from app.api.query_router import extract_section
    section_tests = [
        "bcs4b monday ko kab free ha",
        "bscs04b monday ko kab free ha",
        "BCS-4E ki classes",
        "BSCS-5A ka schedule",
        "bcy4a ki class kab hai",
    ]
    for q in section_tests:
        print(f"  '{q}' -> '{extract_section(q)}'")

    print(f"\n{'=' * 60}")
    print(f"  [OK] RAG v2 pipeline validation COMPLETE")
    print(f"  Next: run the full server to test with embedding model + LLM")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
