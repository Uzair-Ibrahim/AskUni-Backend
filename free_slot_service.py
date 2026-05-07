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

    if len(rows) == 1:
        return rows[0]
    return None


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
    tokens = [t for t in cleaned_teacher.split(" ") if t]

    teacher_query = cleaned_teacher or teacher_name
    resolved = _resolve_teacher_name(tokens)
    if resolved:
        teacher_query = resolved

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
            f"**{teacher_name}** **{canonical_day}** ko free hai:\n{slots_str}"
        )
    return (
        f"**{teacher_name}** is free on **{canonical_day}** during:\n{slots_str}"
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