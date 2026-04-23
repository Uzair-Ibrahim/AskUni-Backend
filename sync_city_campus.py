import pandas as pd
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Timetable

print("⏳ City Campus ka data process ho raha hai...")

# 1. Excel se City Campus wali specific sheet uthao
file_path = "data/timetable.xlsx"
sheet_name = "BS City Campus Classes" # Bhai, apni Excel file mein exact sheet ka naam check kar lena agar error aaye

try:
    # Header 2 isliye kyunke pehli do lines mein FAST ka naam aur semester likha hai
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    total_inserted = 0
    
    for index, row in df.iterrows():
        # Agar Code khali hai toh is row ko ignore karo
        if pd.isna(row['Code']):
            continue
            
        # 2. Day aur Time ko todna (e.g., "Saturday (11:00 - 01:00)")
        day_time_str = str(row['Days & Timing']).strip()
        
        if "(" in day_time_str and ")" in day_time_str:
            parts = day_time_str.split('(')
            day = parts[0].strip() # "Saturday"
            time = parts[1].replace(')', '').strip() # "11:00 - 01:00"
        else:
            day = "Unknown"
            time = day_time_str
            
        # 3. Subject ka naam properly format karna
        course_name = str(row['Course Names']).strip()
        course_code = str(row['Code']).strip()
        section = str(row['Section']).strip()
        
        formatted_subject = f"{course_name} ({course_code}) [{section}]"
        
        # 4. Database mein add karna
        new_entry = Timetable(
            day=day.capitalize(),
            time=time,
            subject=formatted_subject,
            teacher_name=str(row['Name of Teacher']).strip(),
            room_number="City Campus" # Room nahi likha tha, toh default value de di
        )
        
        session.add(new_entry)
        total_inserted += 1
        
    session.commit()
    print(f"✅ Behtareen! City Campus ki {total_inserted} classes successfully Godaam mein save ho gayin!")
    
except Exception as e:
    print(f"❌ Masla aa gaya: {e}")