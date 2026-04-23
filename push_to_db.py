import pandas as pd
from sqlalchemy import create_engine
from database.models import Base, Timetable
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

try:
    print("✅ Database se connection successful!")
    
    # 🧹 ZAROORI STEP: Purana table aur uski kharab settings delete kar do
    Base.metadata.drop_all(bind=engine)
    
    # 🏗️ Naya saaf table banao
    Base.metadata.create_all(bind=engine)
    
    # CSV se saaf data load karo
    df = pd.read_csv("cleaned_timetable.csv")
    df = df.fillna("Unknown") 

    print(f"⏳ {len(df)} rows database mein ja rahi hain...")

    for index, row in df.iterrows():
        new_entry = Timetable(
            day=str(row['Day']),
            time=f"{row['Start_Time']} - {row['End_Time']}",
            subject=str(row['Subject']),
            teacher_name=str(row['Teacher']),
            room_number=str(row['Room']),
            section=str(row['Section']),
            campus=str(row['Campus'])
        )
        session.add(new_entry)

    session.commit()
    print("🚀 KAMYABI! Main aur City campus dono ka data successfully Godaam mein save ho gaya.")

except Exception as e:
    session.rollback()
    print(f"❌ Masla aa gaya: {e}")
finally:
    session.close()