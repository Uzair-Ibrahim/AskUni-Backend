import pandas as pd
import re


def parse_main_class_info(cell_text):
    text = str(cell_text).strip()
    section, teacher = "Unknown", "Unknown"

    if "\n" in text:
        parts = text.split("\n")
        teacher = parts[-1].strip()
        subject_sec_text = " ".join(parts[:-1]).strip()
    else:
        teacher_match = re.search(
            r"\b(Sir|Dr\.|Mr\.|Ms\.|Miss|Engr\.|Prof\.)\s+(.*)", text, re.IGNORECASE
        )
        if teacher_match:
            teacher = teacher_match.group(0).strip()
            subject_sec_text = text[:teacher_match.start()].strip()
        else:
            subject_sec_text = text

    sec_pattern = r"([A-Z]{2,4}-\d[A-Z0-9]+(?:\s*\(.*?\))?)"
    sec_match = re.search(sec_pattern, subject_sec_text)

    if sec_match:
        section = sec_match.group(1).strip()
        subject = subject_sec_text.replace(section, "").strip()
    else:
        sec_bracket_match = re.search(r"\((.*?)\)", subject_sec_text)
        if sec_bracket_match:
            section = sec_bracket_match.group(1).strip()
            subject = re.sub(r"\(.*?\)", "", subject_sec_text).strip()
        else:
            subject = subject_sec_text

    return subject, section, teacher


def normalize_time_value(time_text):
    return str(time_text).strip().rstrip(":")


def extract_timetable_to_csv(file_path="data/timetable.xlsx", output_csv_path="cleaned_timetable.csv"):
    xl = pd.ExcelFile(file_path)
    clean_data = []

    for sheet in xl.sheet_names:
        sheet_upper = sheet.strip().upper()

        # Main Campus (Matrix format)
        if sheet_upper in ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]:
            print(f"📖 Reading Main Campus: {sheet}")
            df = pd.read_excel(file_path, sheet_name=sheet, header=2)
            df.rename(columns={df.columns[0]: "Room"}, inplace=True)
            time_slots = df.columns[1:].tolist()

            for _, row in df.iterrows():
                room = row["Room"]
                if pd.isna(room):
                    continue

                for i in range(len(time_slots)):
                    cell_value = row[time_slots[i]]
                    if pd.notna(cell_value) and str(cell_value).strip() != "":
                        times = str(time_slots[i]).split("-")
                        start_time = normalize_time_value(times[0])
                        end_time = normalize_time_value(times[1]) if len(times) > 1 else start_time

                        if "Lab" in str(cell_value) or "Workshop" in str(cell_value):
                            end_index = min(i + 2, len(time_slots) - 1)
                            end_time_parts = str(time_slots[end_index]).split("-")
                            end_time = (
                                normalize_time_value(end_time_parts[1])
                                if len(end_time_parts) > 1
                                else end_time
                            )

                        subject, section, teacher = parse_main_class_info(cell_value)

                        if pd.isna(subject) or "Course Name" in str(subject) or "Teacher" in str(subject):
                            continue

                        clean_data.append(
                            {
                                "Campus": "Main",
                                "Day": sheet.strip().capitalize(),
                                "Room": str(room).strip(),
                                "Start_Time": start_time,
                                "End_Time": end_time,
                                "Subject": subject,
                                "Section": section,
                                "Teacher": teacher,
                            }
                        )

        # City Campus (List format)
        elif "CITY CAMPUS" in sheet_upper:
            print(f"🏙️ Reading City Campus: {sheet}")
            df_city = pd.read_excel(file_path, sheet_name=sheet, header=2)

            for _, row in df_city.iterrows():
                if pd.isna(row.get("Code")):
                    continue

                day_time_str = str(row.get("Days & Timing", "")).strip()

                if "(" in day_time_str and ")" in day_time_str:
                    parts = day_time_str.split("(")
                    day = parts[0].strip()
                    time_range = parts[1].replace(")", "").strip()
                else:
                    day = "Unknown"
                    time_range = day_time_str

                time_parts = time_range.split("-")
                start_time = normalize_time_value(time_parts[0]) if len(time_parts) > 0 else time_range
                end_time = normalize_time_value(time_parts[1]) if len(time_parts) > 1 else time_range

                course_name = str(row.get("Course Names", "")).strip()
                course_code = str(row.get("Code", "")).strip()

                clean_data.append(
                    {
                        "Campus": "City",
                        "Day": day.capitalize(),
                        "Room": "City Campus",
                        "Start_Time": start_time,
                        "End_Time": end_time,
                        "Subject": f"{course_code} - {course_name}",
                        "Section": str(row.get("Section", "")).strip(),
                        "Teacher": str(row.get("Name of Teacher", "")).strip(),
                    }
                )

    df_result = pd.DataFrame(clean_data)
    df_result.to_csv(output_csv_path, index=False)
    return len(df_result)


if __name__ == "__main__":
    print("⏳ Master Data Extraction Shuru: Main + City Campus...")
    try:
        total_rows = extract_timetable_to_csv()
        print(f"\n🎉 MUBARAK HO! Total {total_rows} classes (Main + City) extract ho gayin.")
    except Exception as e:
        print(f"❌ Error: {e}")