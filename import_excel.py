import pandas as pd

file_path = "data/timetable.xlsx"
print("⏳ Advance Data Parsing shuru ho rahi hai...")

try:
    # 1. File load karo aur pehle column ka naam theek karo
    df = pd.read_excel(file_path, sheet_name="MONDAY", header=2)
    df.rename(columns={df.columns[0]: 'Room'}, inplace=True)
    
    # Time slots ki list (maslan ['08:00-8:50', '08:55-09:45', ...])
    time_slots = df.columns[1:].tolist()
    
    clean_data = [] # Yahan hum apna saaf data rakhenge
    
    # 2. Har room ki line (row) ko bari bari check karo
    for index, row in df.iterrows():
        room = row['Room']
        
        # Agar room ka dabba hi khali hai toh is line ko chhor do
        if pd.isna(room):
            continue
            
        # 3. Har time slot (column) ko bari bari check karo
        for i in range(len(time_slots)):
            cell_value = row[time_slots[i]]
            
            # Agar dabba khali nahi hai (Yani koi Class ya Lab hai)
            if pd.notna(cell_value) and str(cell_value).strip() != "":
                
                # Time ko split karo (Start aur End nikalne ke liye)
                # Maslan '08:00-8:50' toot kar ['08:00', '8:50'] ban jayega
                times = str(time_slots[i]).split('-')
                start_time = times[0].strip()
                end_time = times[1].strip() if len(times) > 1 else start_time
                
                # 🧠 YEH HAI ASLI LOGIC: LAB CHECK
                # Agar text mein 'Lab' ya 'Workshop' likha hai, toh end_time aage ka lo
                if "Lab" in str(cell_value) or "Workshop" in str(cell_value):
                    # 2 slots aage dekho (agar aage itne slots hain)
                    end_index = min(i + 2, len(time_slots) - 1)
                    end_time_parts = str(time_slots[end_index]).split('-')
                    end_time = end_time_parts[1].strip() if len(end_time_parts) > 1 else end_time
                    
                # Data ko apni list mein save kar lo
                clean_data.append({
                    "Room": str(room).strip(),
                    "Start_Time": start_time,
                    "End_Time": end_time,
                    "Class_Info": str(cell_value).strip().replace('\n', ' | ') # Lines ko saaf karne ke liye
                })

    # 4. Apni list ko wapis Pandas DataFrame mein badal kar save kar lo
    df_clean = pd.DataFrame(clean_data)
    
    # Nayi CSV file banao
    df_clean.to_csv("cleaned_timetable.csv", index=False)
    
    print("\n✅ Jadoo ho gaya! Smart logic se data nikal liya gaya hai.")
    print("👉 Apne folder mein 'cleaned_timetable.csv' khol kar dekho!")
    
except Exception as e:
    print(f"❌ Error aa gaya: {e}")