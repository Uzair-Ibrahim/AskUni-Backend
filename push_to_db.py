import pandas as pd
import re
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
# DHYAN DEIN: Yahan Base ko bhi import karna zaroori hai (jahan bhi aapne Base define kiya hai, usually models ya sql_db mein hota hai)
from database.models import Timetable, Base 

print("⏳ Deep Data Parsing aur Database insertion shuru ho rahi hai...")

# 🌟 YEH LINE ADD KI HAI: Yeh check karegi ke table exist karti hai ya nahi. Agar nahi, toh bana degi!
Base.metadata.create_all(bind=engine)

# 1. CSV Load karo
df = pd.read_csv("cleaned_timetable.csv")

Session = sessionmaker(bind=engine)
session = Session()

try:
    session.query(Timetable).delete() # Purana data saaf karo
    
    for index, row in df.iterrows():
        full_info = str(row['Class_Info'])
        
        # Step A: Pehle Teacher aur Subject ko '|' se alag karo
        parts = full_info.split('|')
        subject_section_part = parts[0].strip() # Example: "CS3009-SE BCS-6D"
        teacher = parts[1].strip() if len(parts) > 1 else "Unknown"
        
        # Step B: Subject aur Section ko alag karne ke liye logic
        sub_parts = subject_section_part.rsplit(' ', 1)
        
        course_code_name = sub_parts[0].strip() # CS3009-SE
        section = sub_parts[1].strip() if len(sub_parts) > 1 else "N/A"
        
        # Step C: Course Code nikalna
        course_code = course_code_name.split('-')[0].strip()

        # 2. Database mein Entry
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