import pandas as pd
import re
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Timetable, Base 

print("⏳ Deep Data Parsing aur Database insertion shuru ho rahi hai...")

Base.metadata.create_all(bind=engine)

df = pd.read_csv("cleaned_timetable.csv")

Session = sessionmaker(bind=engine)
session = Session()

try:
    session.query(Timetable).delete()
    
    for index, row in df.iterrows():
        full_info = str(row['Class_Info'])
        
        parts = full_info.split('|')
        subject_section_part = parts[0].strip()
        teacher = parts[1].strip() if len(parts) > 1 else "Unknown"
        
        sub_parts = subject_section_part.rsplit(' ', 1)
        
        course_code_name = sub_parts[0].strip() 
        section = sub_parts[1].strip() if len(sub_parts) > 1 else "N/A"
        
        course_code = course_code_name.split('-')[0].strip()
        
        new_entry = Timetable(
            day="Monday",
            time=f"{row['Start_Time']} - {row['End_Time']}",
            subject=f"{course_code_name} [{section}]", 
            teacher_name=teacher,
            room_number=row['Room']
        )
        
        session.add(new_entry)
    
    session.commit()
    print(f"✅ Mubarak ho! {len(df)} rows ko smart parsing ke saath save kar liya gaya hai.")

except Exception as e:
    session.rollback()
    print(f"❌ Masla aa gaya: {e}")
finally:
    session.close()