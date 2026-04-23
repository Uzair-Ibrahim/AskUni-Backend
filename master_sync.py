import pandas as pd
import re
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Timetable, Base

print("⏳ Poore Hafte ka data Excel se nikal kar Godaam mein ja raha hai...")

# Table ensure karo
Base.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

file_path = "data/timetable.xlsx"

try:
    # Purana kachra saaf karo taake double data na ho jaye
    session.query(Timetable).delete() 
    total_inserted = 0
    
    # 🧠 JADOO: sheet_name=None ka matlab hai poori Excel file ek saath load kar lo!
    all_sheets = pd.read_excel(file_path, sheet_name=None, header=2)
    
    valid_days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]

    # Har sheet (din) ke liye loop
    for sheet_name, df in all_sheets.items():
        
        # Space hata kar check karo ke kya yeh valid din hai?
        day_name = sheet_name.strip().upper()
        if day_name not in valid_days:
            continue
            
        print(f"📅 Process ho raha hai: {day_name}...")
        
        # Pehle column ka naam theek karo
        df.rename(columns={df.columns[0]: 'Room'}, inplace=True)
        time_slots = df.columns[1:].tolist()
        
        for index, row in df.iterrows():
            room = row['Room']
            if pd.isna(room): continue # Agar room khali hai toh chhor do
                
            for i in range(len(time_slots)):
                cell_value = row[time_slots[i]]
                
                if pd.notna(cell_value) and str(cell_value).strip() != "":
                    
                    # Time logic
                    times = str(time_slots[i]).split('-')
                    start_time = times[0].strip()
                    end_time = times[1].strip() if len(times) > 1 else start_time
                    
                    # Lab/Workshop Logic (3 ghante wala)
                    if "Lab" in str(cell_value) or "Workshop" in str(cell_value):
                        end_index = min(i + 2, len(time_slots) - 1)
                        end_time_parts = str(time_slots[end_index]).split('-')
                        end_time = end_time_parts[1].strip() if len(end_time_parts) > 1 else end_time
                    
                    # Text todne (Parsing) ki Logic
                    full_info = str(cell_value).strip().replace('\n', ' | ')
                    parts = full_info.split('|')
                    subject_section_part = parts[0].strip()
                    teacher = parts[1].strip() if len(parts) > 1 else "Unknown"
                    
                    sub_parts = subject_section_part.rsplit(' ', 1)
                    course_code_name = sub_parts[0].strip()
                    section = sub_parts[1].strip() if len(sub_parts) > 1 else "N/A"
                    
                    # Database Entry
                    new_entry = Timetable(
                        day=day_name.capitalize(), # "MONDAY" ko "Monday" bana dega
                        time=f"{start_time} - {end_time}",
                        subject=f"{course_code_name} [{section}]",
                        teacher_name=teacher,
                        room_number=str(room).strip()
                    )
                    session.add(new_entry)
                    total_inserted += 1

    # Saari entries Database mein bhej do
    session.commit()
    print(f"\n✅ Mashallah! Poore hafte ki {total_inserted} classes successfully Godaam mein save ho gayin!")

except Exception as e:
    session.rollback()
    print(f"❌ Error aa gaya: {e}")
finally:
    session.close()