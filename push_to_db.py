import pandas as pd
from sqlalchemy import create_engine
from database.models import Base, Timetable
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL missing in environment variables")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def replace_timetable_from_csv(csv_path="cleaned_timetable.csv"):
    Base.metadata.create_all(bind=engine)
    df = pd.read_csv(csv_path).fillna("Unknown")

    session = SessionLocal()
    try:
        session.query(Timetable).delete(synchronize_session=False)

        for _, row in df.iterrows():
            new_entry = Timetable(
                day=str(row["Day"]),
                time=f"{row['Start_Time']} - {row['End_Time']}",
                subject=str(row["Subject"]),
                teacher_name=str(row["Teacher"]),
                room_number=str(row["Room"]),
                section=str(row["Section"]),
                campus=str(row["Campus"]),
            )
            session.add(new_entry)

        session.commit()
        return len(df)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    try:
        print("✅ Database se connection successful!")
        rows = replace_timetable_from_csv("cleaned_timetable.csv")
        print(f"⏳ {rows} rows database mein ja rahi hain...")
        print("🚀 KAMYABI! Main aur City campus dono ka data successfully Godaam mein save ho gaya.")
    except Exception as e:
        print(f"❌ Masla aa gaya: {e}")