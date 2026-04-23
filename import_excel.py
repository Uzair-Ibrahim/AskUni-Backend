import pandas as pd
import re

file_path = "data/timetable.xlsx"
print("⏳ Advance Data Parsing shuru ho rahi hai...")

def parse_class_info(cell_text):
    text = str(cell_text).strip()
    section = "Unknown"
    teacher = "Unknown"
    
    if '\n' in text:
        parts = text.split('\n')
        teacher = parts[-1].strip()
        subject_sec_text = " ".join(parts[:-1]).strip()
    else:
        teacher_match = re.search(r'\b(Sir|Dr\.|Mr\.|Ms\.|Miss|Engr\.|Prof\.)\s+(.*)', text, re.IGNORECASE)
        if teacher_match:
            teacher = teacher_match.group(0).strip()
            subject_sec_text = text[:teacher_match.start()].strip()
        else:
            subject_sec_text = text

    sec_pattern = r'([A-Z]{2,4}-\d[A-Z0-9]+(?:\s*\(.*?\))?)'
    sec_match = re.search(sec_pattern, subject_sec_text)
    
    if sec_match:
        section = sec_match.group(1).strip()
        subject = subject_sec_text.replace(section, '').strip()
    else:
        sec_bracket_match = re.search(r'\((.*?)\)', subject_sec_text)
        if sec_bracket_match:
            section = sec_bracket_match.group(1).strip()
            subject = re.sub(r'\(.*?\)', '', subject_sec_text).strip()
        else:
            subject = subject_sec_text

    return subject, section, teacher

try:
    df = pd.read_excel(file_path, sheet_name="MONDAY", header=2)
    df.rename(columns={df.columns[0]: 'Room'}, inplace=True)
    time_slots = df.columns[1:].tolist()
    clean_data = [] 
    
    for index, row in df.iterrows():
        room = row['Room']
        if pd.isna(room):
            continue
            
        for i in range(len(time_slots)):
            cell_value = row[time_slots[i]]
            
            if pd.notna(cell_value) and str(cell_value).strip() != "":
                times = str(time_slots[i]).split('-')
                start_time = times[0].strip()
                end_time = times[1].strip() if len(times) > 1 else start_time
    
                if "Lab" in str(cell_value) or "Workshop" in str(cell_value):
                    end_index = min(i + 2, len(time_slots) - 1)
                    end_time_parts = str(time_slots[end_index]).split('-')
                    end_time = end_time_parts[1].strip() if len(end_time_parts) > 1 else end_time
    
                subject, section, teacher = parse_class_info(cell_value)
                
                clean_data.append({
                    "Room": str(room).strip(),
                    "Start_Time": start_time,
                    "End_Time": end_time,
                    "Subject": subject,
                    "Section": section,
                    "Teacher": teacher
                })

    df_clean = pd.DataFrame(clean_data)
    df_clean.to_csv("cleaned_timetable.csv", index=False)
    
    print("\n✅ Jadoo ho gaya! Subject, Section, aur Teacher alag alag ho gaye hain.")
    print("👉 'cleaned_timetable.csv' khol kar check karo!")
    
except Exception as e:
    print(f"❌ Error aa gaya: {e}")