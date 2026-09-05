"""
import_seating.py
=================
Dual-purpose import script for:
  1. Exam seating plans (default)
  2. Class timetables   (--timetable flag)

Both modes DELETE old data and INSERT fresh rows from the PDF.

─── SEATING PLAN ───────────────────────────────────────────────────────────────
Usage:
    python import_seating.py "Seating_Plan.pdf"
    python import_seating.py "Seating_Plan.pdf" "Final Fall 2026"
    python import_seating.py "Seating_Plan.pdf" "Final Fall 2026" Karachi --replace-all

─── TIMETABLE ──────────────────────────────────────────────────────────────────
Usage:
    python import_seating.py --timetable "Timetable.pdf"
    python import_seating.py --timetable "Timetable.pdf" Karachi

    # Dry run (parse only, no DB changes):
    python import_seating.py --timetable "Timetable.pdf" --dry-run
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Base

# Create tables if they don't exist yet
if engine:
    Base.metadata.create_all(bind=engine)
    print("✅ Tables verified/created")
else:
    print("[!] DATABASE_URL not set -- DB operations unavailable")

# ── Parse flags ───────────────────────────────────────────────────────────────

args = sys.argv[1:]
is_timetable = "--timetable" in args
is_dry_run   = "--dry-run" in args
replace_all  = "--replace-all" in args

# Remove flags from args list
for flag in ("--timetable", "--dry-run", "--replace-all"):
    while flag in args:
        args.remove(flag)


# =============================================================================
# TIMETABLE MODE
# =============================================================================
if is_timetable:
    from app.services.exam_seating import parse_timetable_pdf, replace_timetable_from_pdf

    pdf_path = args[0] if len(args) > 0 else "FAST-NUCES, Karachi Sections_Timetables FALL 2026 Version 2..pdf"
    campus   = args[1] if len(args) > 1 else "Karachi"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"\n📄 PDF:     {pdf_path}")
    print(f"🏫 Campus:  {campus}")
    print(f"📋 Mode:    TIMETABLE {'(dry run)' if is_dry_run else ''}")
    print("\nStarting import ...\n")

    if is_dry_run:
        # Parse only — show stats, no DB changes
        records = parse_timetable_pdf(pdf_path, campus)
        sections = set(r.section for r in records)
        teachers = set(r.teacher_name for r in records if r.teacher_name)
        days     = set(r.day for r in records)
        print(f"\n📊 Dry-run results:")
        print(f"   Total rows:  {len(records)}")
        print(f"   Sections:    {len(sections)} → {sorted(sections)[:10]}...")
        print(f"   Teachers:    {len(teachers)}")
        print(f"   Days:        {sorted(days)}")
        print("\n   Sample records (first 10):")
        for r in records[:10]:
            print(f"     {r.section} | {r.day} | {r.time} | {r.subject} | {r.teacher_name} | {r.room_number}")
    else:
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            count = replace_timetable_from_pdf(
                session=session,
                pdf_path=pdf_path,
                campus=campus,
            )
            print(f"\n🎉 Done! {count} timetable records imported.")
            print("   Old timetable data has been deleted and replaced.")
        except Exception as e:
            session.rollback()
            print(f"\n❌ Import failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            session.close()


# =============================================================================
# SEATING PLAN MODE (original behavior — unchanged)
# =============================================================================
else:
    from app.services.exam_seating import replace_exam_seating_from_pdf

    pdf_path     = args[0] if len(args) > 0 else "Student Seating Plan Final Examination Spring 2026"
    exam_session = args[1] if len(args) > 1 else "Final Examination Spring 2026"
    campus       = args[2] if len(args) > 2 else "Karachi"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"\n📄 PDF:          {pdf_path}")
    print(f"📅 Exam Session: {exam_session}")
    print(f"🏫 Campus:       {campus}")
    print("\nStarting import ...\n")

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        count = replace_exam_seating_from_pdf(
            session=session,
            pdf_path=pdf_path,
            session_name=exam_session,
            campus=campus,
            replace_all=replace_all,
        )
        print(f"\n🎉 Done! {count} records imported into database.")

    except Exception as e:
        session.rollback()
        print(f"\n❌ Import failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        session.close()