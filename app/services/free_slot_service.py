"""
free_slot_service.py
====================
Computes FREE time slots for teachers and sections by subtracting
occupied slots (from PostgreSQL) from the fixed daily time grid.

No data is hardcoded — everything comes from the DB at runtime.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ─── FIXED DAILY TIME GRID ────────────────────────────────────────────────────
# Adjust these if your timetable uses different periods.

ALL_SLOTS: List[str] = [
    "08:00 – 08:50",
    "8:55 – 9:45",
    "9:50 – 10:40",
    "10:45 – 11:35",
    "11:40 – 12:30",
    "12:35 – 1:25",
    "01:30 – 02:20",
    "02:25 – 03:15",
    "03:20 – 04:10",
]

# Day name aliases so queries like "Mon", "monday", "MONDAY" all work
DAY_ALIASES: Dict[str, str] = {
    "mon":       "Monday",
    "monday":    "Monday",
    "tue":       "Tuesday",
    "tuesday":   "Tuesday",
    "wed":       "Wednesday",
    "wednesday": "Wednesday",
    "thu":       "Thursday",
    "thursday":  "Thursday",
    "fri":       "Friday",
    "friday":    "Friday",
    "sat":       "Saturday",
    "saturday":  "Saturday",
    "monady":    "Monday",
    "moday":     "Monday",
    "mondy":     "Monday",
    "tuesay":    "Tuesday",
    "tusday":    "Tuesday",
    "wednsday":  "Wednesday",
    "thurday":   "Thursday",
    "frday":     "Friday",
    "satday":    "Saturday",
}

# Map profile/full names to timetable teacher names
TEACHER_ALIASES: Dict[str, str] = {
    "atif luqman": "Atif",
    "atif": "Atif",
}


def normalize_day(day_str: str) -> Optional[str]:
    """Convert any day string variant to a canonical day name."""
    return DAY_ALIASES.get(day_str.strip().lower())


# ─── DB HELPER ────────────────────────────────────────────────────────────────

def _get_connection():
    """Return a raw psycopg2 connection using DATABASE_URL from .env"""
    import psycopg2
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in .env")
    return psycopg2.connect(db_url)


def _normalize_slot(slot: str) -> str:
    return re.sub(r"\s+", " ", slot.replace("–", "-").replace("—", "-")).strip()


def _parse_time_range(slot: str) -> Optional[tuple]:
    cleaned = _normalize_slot(slot)
    match = re.search(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", cleaned)
    if not match:
        return None
    start_h, start_m, end_h, end_m = map(int, match.groups())
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    return start, end


def _slot_sort_key(slot: str) -> int:
    match = re.search(r"(\d{1,2}):(\d{2})", slot)
    if not match:
        return 10**9
    hour = int(match.group(1))
    minute = int(match.group(2))
    return hour * 60 + minute


def _fetch_occupied_slots(
    where_clause: str,
    params: tuple,
    day: str,
) -> Optional[List[str]]:
    """
    Generic query: returns list of time-slot strings occupied
    for a given entity (teacher or section) on a given day.

    Assumes the timetable table has at minimum:
        - a 'day' column  (e.g. 'Monday')
        - a 'time_slot' column (e.g. '08:30 – 10:00')

    Adjust column names below if your schema differs.
    """
    # ── Column name constants — change here if your schema is different ──
    TABLE    = "university_timetable"
    DAY_COL  = "day"        # column holding day name
    TIME_COL = "time"       # column holding slot string like '08:30 – 10:00'

    sql = f"""
        SELECT DISTINCT {TIME_COL}
        FROM {TABLE}
        WHERE LOWER({DAY_COL}) = LOWER(%s)
          AND {where_clause}
    """
    try:
        conn = _get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (day, *params))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [row[0].strip() for row in rows if row[0]]
    except Exception as e:
        print(f"  [FreeSlot] DB error: {e}")
        return None


# ─── FREE SLOT COMPUTATION ────────────────────────────────────────────────────

def _compute_free(all_slots: List[str], occupied: List[str]) -> List[str]:
    """Subtract occupied slots from the daily grid."""
    occupied_ranges = []
    for s in occupied:
        parsed = _parse_time_range(s)
        if parsed:
            occupied_ranges.append(parsed)

    free_slots = []
    for slot in all_slots:
        parsed = _parse_time_range(slot)
        if not parsed:
            continue
        slot_start, slot_end = parsed
        overlaps = False
        for occ_start, occ_end in occupied_ranges:
            if slot_start < occ_end and occ_start < slot_end:
                overlaps = True
                break
        if not overlaps:
            free_slots.append(slot)
    return free_slots


def _fetch_day_slots(day: str) -> Optional[List[str]]:
    """Return distinct time slots for a given day from the timetable."""
    DAY_COL = "day"
    TIME_COL = "time"
    TABLE = "university_timetable"

    sql = f"""
        SELECT DISTINCT {TIME_COL}
        FROM {TABLE}
        WHERE LOWER({DAY_COL}) = LOWER(%s)
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(sql, (day,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        slots = [_normalize_slot(row[0]) for row in rows if row and row[0]]
        unique = sorted(set(slots), key=_slot_sort_key)
        return unique
    except Exception as e:
        print(f"  [FreeSlot] DB error: {e}")
        return None


def _resolve_teacher_name(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None

    TABLE = "university_timetable"
    TEACHER_COL = "teacher_name"

    # Strategy 1: AND all tokens — e.g. "Atif Luqman" → ILIKE '%Atif%' AND ILIKE '%Luqman%'
    clauses = " AND ".join([f"{TEACHER_COL} ILIKE %s" for _ in tokens])
    sql = f"""
        SELECT DISTINCT {TEACHER_COL}
        FROM {TABLE}
        WHERE {clauses}
        LIMIT 5
    """

    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(sql, tuple(f"%{t}%" for t in tokens))
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  [FreeSlot] DB error: {e}")
        return None

    rows = _filter_teacher_candidates(rows)
    if len(rows) == 1:
        return rows[0]

    # Strategy 2: Fallback — try each token individually (OR), then pick the
    # best match.  Handles cases where the timetable stores only a first name
    # (e.g. "Atif") but the knowledge base / user says "Atif Luqman".
    if not rows and len(tokens) > 1:
        or_clauses = " OR ".join([f"{TEACHER_COL} ILIKE %s" for _ in tokens])
        sql_or = f"""
            SELECT DISTINCT {TEACHER_COL}
            FROM {TABLE}
            WHERE {or_clauses}
            LIMIT 10
        """
        try:
            conn = _get_connection()
            cur = conn.cursor()
            cur.execute(sql_or, tuple(f"%{t}%" for t in tokens))
            or_rows = [row[0] for row in cur.fetchall() if row and row[0]]
            cur.close()
            conn.close()
        except Exception as e:
            print(f"  [FreeSlot] DB error (fallback): {e}")
            return None

        or_rows = _filter_teacher_candidates(or_rows)
        if len(or_rows) == 1:
            return or_rows[0]

        # If multiple matches, prefer exact first-name match (shortest name)
        # to disambiguate e.g. "Atif" vs "Dr. Atif Saleem"
        if or_rows:
            # Filter: keep only names where the first meaningful token matches
            first_token = tokens[0].lower()
            exact_first = [
                r for r in or_rows
                if r.strip().lower() == first_token
                or r.strip().lower().split()[-1] == first_token
                or r.strip().lower().startswith(first_token)
            ]
            if len(exact_first) == 1:
                return exact_first[0]

    return None


def _filter_teacher_candidates(names: List[str]) -> List[str]:
    if not names:
        return []
    cleaned = []
    for name in names:
        lower = name.lower().strip()
        if not lower:
            continue
        if "reserved for" in lower or "lab exam" in lower or "exam" in lower:
            continue
        cleaned.append(name)
    return cleaned


def _find_teacher_candidates(tokens: List[str]) -> List[str]:
    if not tokens:
        return []

    TABLE = "university_timetable"
    TEACHER_COL = "teacher_name"

    clauses = " AND ".join([f"{TEACHER_COL} ILIKE %s" for _ in tokens])
    sql_and = f"""
        SELECT DISTINCT {TEACHER_COL}
        FROM {TABLE}
        WHERE {clauses}
        LIMIT 10
    """

    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(sql_and, tuple(f"%{t}%" for t in tokens))
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
        cur.close()
        conn.close()
    except Exception:
        rows = []

    if rows:
        return _filter_teacher_candidates(rows)

    if len(tokens) > 1:
        or_clauses = " OR ".join([f"{TEACHER_COL} ILIKE %s" for _ in tokens])
        sql_or = f"""
            SELECT DISTINCT {TEACHER_COL}
            FROM {TABLE}
            WHERE {or_clauses}
            LIMIT 10
        """
        try:
            conn = _get_connection()
            cur = conn.cursor()
            cur.execute(sql_or, tuple(f"%{t}%" for t in tokens))
            rows = [row[0] for row in cur.fetchall() if row and row[0]]
            cur.close()
            conn.close()
        except Exception:
            rows = []

    return _filter_teacher_candidates(rows)


def _pick_default_candidate(tokens: List[str], candidates: List[str]) -> Optional[str]:
    if len(tokens) != 1 or not candidates:
        return None
    token = tokens[0].lower()
    exact = [c for c in candidates if c.strip().lower() == token]
    if exact:
        return exact[0]
    return sorted(candidates, key=lambda c: (len(c), c.lower()))[0]


import re  # needed by _compute_free; placed here to keep top of file clean


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def get_teacher_free_slots(teacher_name: str, day: str, language_hint: str = "english") -> str:
    """
    Returns a human-readable string listing the teacher's free slots on day.

    Args:
        teacher_name: partial or full name (case-insensitive ILIKE search)
        day:          canonical day string e.g. 'Monday'
    """
    canonical_day = normalize_day(day)
    if not canonical_day:
        if language_hint == "roman_urdu":
            return f"'{day}' koi theek din nahi hai. Monday se Saturday tak likhen."
        return f"'{day}' is not a recognized day. Please use Monday–Saturday."

    # ── Column name for teacher — adjust if yours is 'instructor', 'faculty', etc.
    TEACHER_COL = "teacher_name"

    cleaned_teacher = re.sub(r"\b(sir|dr\.?|mr\.?|ms\.?|prof\.?|professor)\b", "", teacher_name, flags=re.IGNORECASE)
    cleaned_teacher = re.sub(r"\s+", " ", cleaned_teacher).strip()
    alias_key = cleaned_teacher.lower()
    alias_hit = False
    if alias_key in TEACHER_ALIASES:
        cleaned_teacher = TEACHER_ALIASES[alias_key]
        alias_hit = True
    tokens = [t for t in cleaned_teacher.split(" ") if t]

    teacher_query = cleaned_teacher or teacher_name
    default_notice = ""
    use_exact_match = False
    resolved = _resolve_teacher_name(tokens)
    if resolved:
        teacher_query = resolved
    elif not alias_hit:
        if len(tokens) == 1:
            token = tokens[0].lower()
            TABLE = "university_timetable"
            TEACHER_COL = "teacher_name"
            sql = f"""
                SELECT DISTINCT {TEACHER_COL}
                FROM {TABLE}
                WHERE LOWER({TEACHER_COL}) = LOWER(%s)
                LIMIT 1
            """
            try:
                conn = _get_connection()
                cur = conn.cursor()
                cur.execute(sql, (token,))
                row = cur.fetchone()
                cur.close()
                conn.close()
                if row and row[0]:
                    teacher_query = row[0]
                    tokens = []
                    use_exact_match = True
            except Exception:
                pass
        candidates = _find_teacher_candidates(tokens)
        if len(candidates) == 1:
            teacher_query = candidates[0]
        elif len(candidates) > 1:
            default_pick = _pick_default_candidate(tokens, candidates)
            if default_pick:
                teacher_query = default_pick
                if len(tokens) == 1 and default_pick.strip().lower() == tokens[0].lower():
                    use_exact_match = True
                names = ", ".join(candidates[:5])
                if language_hint == "roman_urdu":
                    default_notice = (
                        "\nAgar aap kisi aur teacher ka pooch rahe hain to full name likhen. "
                        f"Misal: {names}"
                    )
                else:
                    default_notice = (
                        "\nIf you meant a different teacher, please provide the full name. "
                        f"Examples: {names}"
                    )
            else:
                names = ", ".join(candidates[:5])
                if language_hint == "roman_urdu":
                    return (
                        "Is naam se multiple teachers mil rahe hain. Full name likhen. "
                        f"Misal: {names}"
                    )
                return (
                    "This name matches multiple teachers. Please provide the full name. "
                    f"Examples: {names}"
                )

    if alias_hit or use_exact_match:
        occupied = _fetch_occupied_slots(
            where_clause=f"LOWER({TEACHER_COL}) = LOWER(%s)",
            params=(teacher_query,),
            day=canonical_day,
        )
    else:
        occupied = _fetch_occupied_slots(
            where_clause=f"{TEACHER_COL} ILIKE %s",
            params=(f"%{teacher_query}%",),
            day=canonical_day,
        )

    if occupied is None:  # DB error already printed
        if language_hint == "roman_urdu":
            return "Schedule nahi mil saka. Baad mein dobara try karein."
        return "Could not retrieve schedule. Please try again later."

    if occupied is not None and not occupied:
        if language_hint == "roman_urdu":
            return f"Is naam ka koi timetable record nahi mila: {teacher_name}"
        return f"No timetable records found for {teacher_name}."

    free = _compute_free(ALL_SLOTS, occupied or [])

    if not free:
        if language_hint == "roman_urdu":
            return (
                f"**{teacher_name}** ka **{canonical_day}** ko koi free slot nahi — "
                f"sab periods occupied hain."
            )
        return (
            f"**{teacher_name}** has no free slots on **{canonical_day}** — "
            f"all periods are occupied."
        )

    slots_str = "\n".join(f"  • {s}" for s in free)
    if language_hint == "roman_urdu":
        return (
            f"**{teacher_name}** **{canonical_day}** ko free hai:\n{slots_str}{default_notice}"
        )
    return (
        f"**{teacher_name}** is free on **{canonical_day}** during:\n{slots_str}{default_notice}"
    )


def get_section_free_slots(section: str, day: str, language_hint: str = "english") -> str:
    """
    Returns a human-readable string listing the section's free slots on day.

    Args:
        section: section identifier e.g. 'BSCS-5A' (ILIKE search)
        day:     canonical day string e.g. 'Tuesday'
    """
    canonical_day = normalize_day(day)
    if not canonical_day:
        if language_hint == "roman_urdu":
            return f"'{day}' koi theek din nahi hai. Monday se Saturday tak likhen."
        return f"'{day}' is not a recognized day. Please use Monday–Saturday."

    # ── Column name for section — adjust if yours is 'batch', 'group', etc.
    SECTION_COL = "section"

    occupied = _fetch_occupied_slots(
        where_clause=f"{SECTION_COL} ILIKE %s",
        params=(f"%{section}%",),
        day=canonical_day,
    )

    if occupied is None:
        if language_hint == "roman_urdu":
            return "Schedule nahi mil saka. Baad mein dobara try karein."
        return "Could not retrieve schedule. Please try again later."

    free = _compute_free(ALL_SLOTS, occupied or [])

    if not free:
        if language_hint == "roman_urdu":
            return (
                f"**{section}** ka **{canonical_day}** ko koi free slot nahi — "
                f"sab periods scheduled hain."
            )
        return (
            f"**{section}** has no free slots on **{canonical_day}** — "
            f"all periods are scheduled."
        )

    slots_str = "\n".join(f"  • {s}" for s in free)
    if language_hint == "roman_urdu":
        return (
            f"**{section}** ke **{canonical_day}** ko free slots hain:\n{slots_str}"
        )
    return (
        f"**{section}** has free slots on **{canonical_day}**:\n{slots_str}"
    )