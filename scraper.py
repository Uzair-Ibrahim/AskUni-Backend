"""
NUCES FAST Karachi Campus — Incremental Single-File Scraper
============================================================
Only re-scrapes pages whose content has changed since the last run.
Uses a local cache file (nuces_cache.json) to store page hashes.

Usage:
    pip install requests beautifulsoup4
    python nuces_khi_scraper.py

    # Force full re-scrape (ignore cache):
    python nuces_khi_scraper.py --force

Output:
    NUCES_KHI_Complete.md   — full compiled output (always regenerated)
    NUCES_KHI_Changes.md    — what changed vs last run (if anything did)
    nuces_cache.json        — hash + scraped content cache per URL
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import os
import re
import json
import hashlib
import sys
from datetime import datetime
from urllib.parse import urljoin

# ---------- CONFIG ----------
OUTPUT_FILE   = "NUCES_KHI_Complete.md"
CHANGES_FILE  = "NUCES_KHI_Changes.md"
CACHE_FILE    = "nuces_cache.json"
DELAY_MIN     = 1.2
DELAY_MAX     = 2.8
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         "https://khi.nu.edu.pk/",
}

GENERAL_URLS = [
    ("NUCES Karachi Campus Home",   "https://khi.nu.edu.pk/"),
    ("Degree Programs",             "https://nu.edu.pk/Degree-Programs"),
    ("NUCES Home",                  "https://nu.edu.pk/Home"),
    ("Admissions Schedule",         "https://nu.edu.pk/Admissions/Schedule"),
    ("How To Apply",                "https://nu.edu.pk/Admissions/HowToApply"),
    ("Eligibility Criteria",        "https://nu.edu.pk/Admissions/EligibilityCriteria"),
    ("Scholarships",                "https://nu.edu.pk/Admissions/Scholarship"),
    ("Test Pattern",                "https://nu.edu.pk/Admissions/TestPattern"),
    ("Fee Structure",               "https://nu.edu.pk/Admissions/FeeStructure"),
]

DEPARTMENT_URLS = [
    ("Computer Science",        "https://khi.nu.edu.pk/faculty-php/"),
    ("Cyber Security",          "https://khi.nu.edu.pk/department-of-cyber-security/"),
    ("Artificial Intelligence", "https://khi.nu.edu.pk/department-of-artificial-intelligence/"),
    ("Software Engineering",    "https://khi.nu.edu.pk/department-of-software-engineering/"),
    ("Electrical Engineering",  "https://khi.nu.edu.pk/department-of-electrical-engineering/"),
    ("Management Sciences",     "https://khi.nu.edu.pk/department-of-management-sciences/"),
    ("Sciences & Humanities",   "https://khi.nu.edu.pk/department-of-sciences-humanities/"),
]
# ----------------------------

FORCE_RESCRAPE = "--force" in sys.argv


# ─── CACHE ───────────────────────────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def page_hash(raw_html):
    """
    Hash only meaningful text content — ignores dynamic elements like
    timestamps, session tokens, visitor counters to avoid false positives.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(r"(counter|visitor|clock|datetime|nonce)", re.I)):
        tag.decompose()
    clean = soup.get_text(separator=" ", strip=True)
    clean = re.sub(r"\s+", " ", clean).strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


# ─── HTTP ────────────────────────────────────────────────────────────────────

def fetch_raw(url, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            if len(resp.text) < 300:
                time.sleep(3 * (attempt + 1))
                continue
            return resp.text, BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1} failed ({url}): {e}")
            if attempt < retries:
                time.sleep(2 + attempt)
    return None, None


# ─── HTML → MARKDOWN ─────────────────────────────────────────────────────────

def parse_table(table_tag):
    rows = table_tag.find_all("tr")
    if not rows:
        return ""
    md_rows = []
    for i, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        data = [c.get_text(separator=" ", strip=True).replace("|", "/") for c in cells]
        if not any(data):
            continue
        md_rows.append("| " + " | ".join(data) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(data)) + " |")
    return "\n" + "\n".join(md_rows) + "\n" if md_rows else ""

CHROME = re.compile(
    r"(nav|menu|sidebar|footer|header|breadcrumb|social|share|cookie|"
    r"popup|modal|banner|advertisement|widget-area|wp-caption)", re.I
)
STRIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas"}

def element_to_md(element, heading_offset=0):
    if element is None:
        return ""
    lines = []

    def _walk(node):
        if isinstance(node, str):
            t = node.strip()
            if t:
                lines.append(t + " ")
            return
        tag = node.name
        if not tag:
            return
        if tag in STRIP_TAGS:
            return
        classes = " ".join(node.get("class", []))
        node_id = node.get("id", "")
        if CHROME.search(classes) or CHROME.search(node_id):
            return
        if tag == "table":
            lines.append("\n" + parse_table(node) + "\n")
            return
        if tag in ("h1","h2","h3","h4","h5","h6"):
            level = min(int(tag[1]) + 1 + heading_offset, 6)
            text = node.get_text(strip=True)
            if text:
                lines.append(f"\n\n{'#'*level} {text}\n\n")
            return
        if tag == "p":
            text = node.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"\n{text}\n")
            return
        if tag in ("ul","ol"):
            lines.append("\n")
            for c in node.children: _walk(c)
            lines.append("\n")
            return
        if tag == "li":
            text = node.get_text(separator=" ", strip=True)
            if text: lines.append(f"- {text}\n")
            return
        if tag == "br":
            lines.append("\n"); return
        if tag == "hr":
            lines.append("\n---\n"); return
        if tag == "a":
            text = node.get_text(strip=True)
            href = node.get("href","").strip()
            if text and href and not href.startswith(("#","javascript")):
                lines.append(f"[{text}]({href}) ")
            elif text:
                lines.append(text + " ")
            return
        if tag in ("strong","b"):
            text = node.get_text(strip=True)
            if text: lines.append(f"**{text}** ")
            return
        if tag in ("em","i"):
            text = node.get_text(strip=True)
            if text: lines.append(f"*{text}* ")
            return
        for c in node.children: _walk(c)

    _walk(element)
    text = "".join(lines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def get_main_content(soup):
    for sel in ["main",
                {"id": re.compile(r"(content|main|primary)", re.I)},
                {"class_": re.compile(r"(entry-content|post-content|page-content|main-content)", re.I)},
                "article",
                {"class_": re.compile(r"(gdlr-core-pbf-wrapper|site-content)", re.I)}]:
        el = soup.find(sel) if isinstance(sel, str) else soup.find(**sel)
        if el:
            return el
    return soup.body


# ─── CHANGE TRACKER ──────────────────────────────────────────────────────────

class ChangeTracker:
    def __init__(self):
        self.added     = []
        self.updated   = []
        self.removed   = []
        self.unchanged = []

    def summary(self):
        total   = len(self.added) + len(self.updated) + len(self.unchanged)
        changed = len(self.added) + len(self.updated)
        return (f"{changed}/{total} pages changed  "
                f"({len(self.added)} new, {len(self.updated)} updated, "
                f"{len(self.unchanged)} unchanged, {len(self.removed)} removed)")

    def build_changes_md(self, run_ts):
        lines = [
            f"# NUCES KHI — Change Report\n",
            f"**Run date:** {run_ts}\n",
            f"**Summary:** {self.summary()}\n\n---\n",
        ]
        if self.added:
            lines.append("\n## New Pages / Faculty\n")
            for label, url in self.added:
                lines.append(f"- **{label}** — {url}")
        if self.updated:
            lines.append("\n## Updated Pages\n")
            for label, url in self.updated:
                lines.append(f"- **{label}** — {url}")
        if self.removed:
            lines.append("\n## Removed Pages / Faculty\n")
            for label, url in self.removed:
                lines.append(f"- **{label}** — {url}")
        if self.unchanged:
            lines.append("\n## Unchanged Pages\n")
            for label, url in self.unchanged:
                lines.append(f"- {label}")
        return "\n".join(lines)


# ─── SCRAPE-OR-CACHE ─────────────────────────────────────────────────────────

def scrape_or_cache(url, label, scrape_fn, cache, tracker):
    """
    1. Fetch page → hash its content.
    2. Hash matches cache → return cached markdown (skip scraping).
    3. Hash differs or no cache entry → call scrape_fn, update cache.
    4. Network failure → fall back to cache if available.
    """
    raw, soup = fetch_raw(url)

    if raw is None:
        if url in cache:
            print("  ⚠️  Network failure — using cached version")
            tracker.unchanged.append((label, url))
            return cache[url]["markdown"]
        print(f"  ❌  No cache and network failure")
        tracker.unchanged.append((label + " [FAILED]", url))
        return f"> ⚠️ Could not load: {url}\n\n"

    h = page_hash(raw)

    if not FORCE_RESCRAPE and url in cache and cache[url]["hash"] == h:
        print("  ✅  Unchanged (using cache)")
        tracker.unchanged.append((label, url))
        return cache[url]["markdown"]

    is_new = url not in cache
    md = scrape_fn(soup, raw)

    cache[url] = {
        "hash":       h,
        "markdown":   md,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "label":      label,
    }

    if is_new:
        print("  🆕  New — scraped")
        tracker.added.append((label, url))
    else:
        print("  ✏️   Changed — re-scraped")
        tracker.updated.append((label, url))

    return md


# ─── SCRAPE FUNCTIONS ────────────────────────────────────────────────────────

def scrape_general_fn(soup, raw):
    for tag in soup.find_all(["nav","footer","script","style","noscript","iframe"]):
        tag.decompose()
    main = get_main_content(soup)
    content = element_to_md(main)
    content = re.sub(r"\[(?:Home|Skip to[^\]]*)\]\([^)]+\)\s*", "", content, flags=re.I)
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    return content


def scrape_faculty_profile_fn(soup, raw, name, designation="", email_hint="-", extension_hint="-"):
    content_div = (
        soup.find("div", class_="gdlr-core-pbf-wrapper")
        or soup.find("article")
        or soup.find("main")
        or soup.body
    )
    raw_text = content_div.get_text(separator=" ") if content_div else ""

    email = "-"
    for e in re.findall(r"[a-zA-Z0-9._%+-]+@nu\.edu\.pk", raw_text, re.IGNORECASE):
        if e.lower() != "info@nu.edu.pk":
            email = e.lower(); break

    extension = "-"
    m = re.search(r"(?:Ext\.?|Extension|Intercom|Phone)[\s.:-]*(\d{3,4})\b", raw_text, re.IGNORECASE)
    if m:
        extension = m.group(1)

    if email == "-" and email_hint and email_hint != "-":
        email = email_hint
    if extension == "-" and extension_hint and extension_hint != "-":
        extension = extension_hint

    if not designation:
        d_el = content_div.find(class_=re.compile(r"desig|position|sub-title|gdlr-core-title", re.I)) if content_div else None
        if d_el:
            designation = d_el.get_text(strip=True)

    structured = element_to_md(content_div, heading_offset=1) if content_div else ""
    structured = re.sub(r"info@nu\.edu\.pk", "", structured, flags=re.IGNORECASE)
    if email != "-":
        structured = re.sub(re.escape(email), "", structured, flags=re.IGNORECASE)
    if extension != "-":
        structured = re.sub(r"\b" + re.escape(extension) + r"\b", "", structured, count=1)
    structured = re.sub(
        r"#{2,}\s*(?:Email|E-mail|Extension|Ext\.?|Contact\s*Details?)\s*\n+.*?(?=\n#{2,}|\Z)",
        "", structured, flags=re.IGNORECASE | re.DOTALL
    )
    structured = re.sub(r"\b(?:Ext\.?|Extension|Intercom)[\s.:-]*", "", structured, flags=re.IGNORECASE)
    structured = re.sub(r"^#+\s*" + re.escape(name) + r"\s*\n", "", structured, flags=re.IGNORECASE)
    structured = re.sub(r"\n{3,}", "\n\n", structured).strip()

    out  = f"### Faculty: {name}\n\n"
    out += f"- **Email:** {email}\n"
    if extension != "-":
        out += f"- **Extension:** {extension}\n"
    if designation:
        out += f"- **Designation:** {designation}\n"
    if structured:
        out += "\n" + structured + "\n"
    out += "\n---\n\n"
    return out


def get_faculty_links_from_page(soup, base_url):
    links, seen = [], set()
    for a in soup.find_all("a", string=lambda t: t and "more detail" in t.lower()):
        href = a.get("href","").strip()
        if not href: continue
        full_url = urljoin(base_url, href.rstrip("/") + "/")
        if full_url in seen: continue
        seen.add(full_url)
        card = a.find_parent("div", class_=lambda c: c and "faculty" in c.lower()) or a.find_parent("div")
        name_tag = card.find(re.compile(r"h[234]")) if card else None
        name = name_tag.get_text(strip=True) if name_tag else "Unknown"
        name = re.sub(r"\s*\(?\s*on\s+leave\s*\)?\s*", "", name, flags=re.I).strip()
        desig = ""
        d_el = card.find(class_=re.compile(r"desig|position|sub-title", re.I)) if card else None
        if d_el: desig = d_el.get_text(strip=True)
        email = "-"
        extension = "-"
        if card:
            card_text = card.get_text(separator=" ", strip=True)
            mailto = card.find("a", href=re.compile(r"^mailto:", re.I))
            if mailto:
                email = re.sub(r"^mailto:", "", mailto.get("href", ""), flags=re.I).strip()
            if email == "-":
                m = re.search(r"[a-zA-Z0-9._%+-]+@nu\.edu\.pk", card_text, re.IGNORECASE)
                if m:
                    email = m.group(0).lower()
            ext_match = re.search(r"(?:Ext\.?|Extension|Intercom|Phone)[\s.:-]*(\d{3,4})\b", card_text, re.IGNORECASE)
            if ext_match:
                extension = ext_match.group(1)
            else:
                fallback_ext = re.search(r"\b(\d{3,4})\b", card_text)
                if fallback_ext:
                    extension = fallback_ext.group(1)

        links.append({"name": name, "url": full_url, "designation": desig, "email": email, "extension": extension})
    return links


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cache   = load_cache()
    tracker = ChangeTracker()

    if FORCE_RESCRAPE:
        print("⚡ --force flag: ignoring all cached content\n")

    active_urls = set()
    sections = []
    sections.append(
        f"# NUCES FAST University — Karachi Campus\n"
        f"## Complete Information Guide\n\n"
        f"*Last updated: {run_ts}*\n\n"
        f"---\n\n"
    )

    # ── Part 1: General pages ──
    print("=" * 60)
    print("PART 1: General University Pages")
    print("=" * 60)
    sections.append("# Part 1: University & Admissions Information\n\n")

    for title, url in GENERAL_URLS:
        active_urls.add(url)
        print(f"\n📄 {title}")
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        def _general(soup, raw, _title=title, _url=url):
            return (
                f"## Page: {_title}\n\n**URL:** {_url}\n\n"
                + scrape_general_fn(soup, raw)
                + "\n\n---\n\n"
            )

        sections.append(scrape_or_cache(url, title, _general, cache, tracker))

    # ── Part 2: Departments & Faculty ──
    print("\n" + "=" * 60)
    print("PART 2: Departments & Faculty Profiles")
    print("=" * 60)
    sections.append("\n\n# Part 2: Departments & Faculty Profiles\n\n")

    for dept_name, dept_url in DEPARTMENT_URLS:
        active_urls.add(dept_url)
        print(f"\n📂 Department: {dept_name}")
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        raw_dept, soup_dept = fetch_raw(dept_url)

        dept_section = f"## Department: {dept_name}\n\n**URL:** {dept_url}\n\n"

        # Dept overview intro
        if soup_dept:
            for tag in soup_dept.find_all(["nav","footer","script","style"]):
                tag.decompose()
            main_el = get_main_content(soup_dept)
            if main_el:
                intro = []
                for p in main_el.find_all("p"):
                    t = p.get_text(strip=True)
                    if t and len(t) > 50 and "more detail" not in t.lower():
                        intro.append(t)
                    if len(intro) >= 6: break
                if intro:
                    dept_section += "### Department Overview\n\n" + "\n\n".join(intro) + "\n\n"

        faculty_links = get_faculty_links_from_page(soup_dept, dept_url) if soup_dept else []
        print(f"  🔗 Found {len(faculty_links)} faculty members")

        # Detect removed faculty
        current_faculty_urls = {f["url"] for f in faculty_links}
        for cached_url, cached_data in list(cache.items()):
            if (cached_url.startswith(dept_url)
                    and cached_url != dept_url
                    and cached_url not in current_faculty_urls):
                label = cached_data.get("label", cached_url)
                tracker.removed.append((label, cached_url))
                print(f"  ❌  Removed from site: {label}")

        if faculty_links:
            dept_section += f"### Faculty Members ({len(faculty_links)} total)\n\n"
            for i, info in enumerate(faculty_links, 1):
                name = info["name"]
                furl = info["url"]
                desig = info["designation"]
                email_hint = info.get("email", "-")
                extension_hint = info.get("extension", "-")
                active_urls.add(furl)
                print(f"  [{i}/{len(faculty_links)}] {name} ...", end=" ", flush=True)
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

                def _faculty(soup, raw, _name=name, _desig=desig, _email=email_hint, _ext=extension_hint):
                    return scrape_faculty_profile_fn(soup, raw, _name, _desig, _email, _ext)

                dept_section += scrape_or_cache(furl, f"Faculty: {name}", _faculty, cache, tracker)

        sections.append(dept_section + "\n\n")

    # Catch any cached URLs no longer visited at all
    for cached_url, cached_data in cache.items():
        if cached_url not in active_urls:
            label = cached_data.get("label", cached_url)
            if (label, cached_url) not in tracker.removed:
                tracker.removed.append((label, cached_url))

    # ── Write outputs ──
    final = "\n".join(sections)
    final = re.sub(r"\n{5,}", "\n\n\n", final)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final)

    with open(CHANGES_FILE, "w", encoding="utf-8") as f:
        f.write(tracker.build_changes_md(run_ts))

    save_cache(cache)

    kb = os.path.getsize(OUTPUT_FILE) // 1024
    print(f"\n{'='*60}")
    print(f"📊 {tracker.summary()}")
    print(f"📄 Output   → {OUTPUT_FILE} ({kb} KB)")
    print(f"📋 Changes  → {CHANGES_FILE}")
    print(f"💾 Cache    → {CACHE_FILE}")
    if FORCE_RESCRAPE:
        print("⚡ Full re-scrape was forced (--force)")
    print("="*60)


if __name__ == "__main__":
    main()