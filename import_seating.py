"""
import_seating.py
=================
One-time script to parse the seating plan PDF and push to PostgreSQL.
Run this once every exam season when you get a new PDF.

Usage:
    python import_seating.py "Seating_Plan_Sessional_II_Spring_2026.pdf"

    # Or with custom session name:
    python import_seating.py "Seating_Plan_Final_Fall_2026.pdf" "Final Fall 2026"

    # Delete all exam seating rows before insert:
    python import_seating.py "Seating_Plan_Final_Fall_2026.pdf" "Final Fall 2026" Karachi --replace-all
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Base, ExamSeating
from app.services.exam_seating import replace_exam_seating_from_pdf

# Create table if it doesn't exist yet
Base.metadata.create_all(bind=engine)
print("✅ Tables verified/created")

# Args
args = sys.argv[1:]
replace_all = "--replace-all" in args
if replace_all:
    args.remove("--replace-all")

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