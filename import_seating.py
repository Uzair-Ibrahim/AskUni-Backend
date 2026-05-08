"""
import_seating.py
=================
One-time script to parse the seating plan PDF and push to PostgreSQL.
Run this once every exam season when you get a new PDF.

Usage:
    python import_seating.py "Seating_Plan_Sessional_II_Spring_2026.pdf"

    # Or with custom session name:
    python import_seating.py "Seating_Plan_Final_Fall_2026.pdf" "Final Fall 2026"
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
pdf_path     = sys.argv[1] if len(sys.argv) > 1 else "Seating_Plan_Sessional_II_Spring_2026.pdf"
exam_session = sys.argv[2] if len(sys.argv) > 2 else "Sessional II Spring 2026"
campus       = sys.argv[3] if len(sys.argv) > 3 else "Karachi"

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
    )
    print(f"\n🎉 Done! {count} records imported into database.")

except Exception as e:
    session.rollback()
    print(f"\n❌ Import failed: {e}")
    import traceback; traceback.print_exc()
finally:
    session.close()