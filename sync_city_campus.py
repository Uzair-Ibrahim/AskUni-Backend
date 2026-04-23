import pandas as pd
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Timetable

print("⏳ City Campus ka data process ho raha hai...")

file_path = "data/timetable.xlsx"
sheet_name = "BS City Campus Classes" 

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    total_inserted = 0
    
    for index, row in df.iterrows():
        if pd.isna(row['Code']):
            continue
            
        day_time_str = str(row['Days & Timing']).strip()
        
        if "(" in day_time_str and ")" in day_time_str:
            parts = day_time_str.split('(')
            day = parts[0].strip() # "Saturday"
            time = parts[1].replace(')', '').strip() 
        else:
            day = "Unknown"
            time = day_time_str
            
        course_name = str(row['Course Names']).strip()
        course_code = str(row['Code']).strip()
        section = str(row['Section']).strip()
        
        formatted_subject = f"{course_name} ({course_code}) [{section}]"
        
        new_entry = Timetable(
            day=day.capitalize(),
            time=time,
            subject=formatted_subject,
            teacher_name=str(row['Name of Teacher']).strip(),
            room_number="City Campus" 
        )
        
        session.add(new_entry)
        total_inserted += 1
        
    session.commit()
    print(f"✅ Behtareen! City Campus ki {total_inserted} classes successfully Godaam mein save ho gayin!")
    
except Exception as e:
    print(f"❌ Masla aa gaya: {e}")