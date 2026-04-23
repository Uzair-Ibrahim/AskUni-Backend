from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker
from database.sql_db import engine
from database.models import Timetable
from typing import Optional

app = FastAPI()
Session = sessionmaker(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Welcome to AskUni Smart Search! 🚀"}

@app.get("/search")
def search_timetable(
    room: Optional[str] = None,
    teacher: Optional[str] = None,
    subject: Optional[str] = None,
    day: Optional[str] = None
):
    session = Session()
    try:
        query = session.query(Timetable)
        
        if room: query = query.filter(Timetable.room_number.ilike(f"%{room}%"))
        if teacher: query = query.filter(Timetable.teacher_name.ilike(f"%{teacher}%"))
        if subject: query = query.filter(Timetable.subject.ilike(f"%{subject}%"))
        if day: query = query.filter(Timetable.day.ilike(f"%{day}%"))
            
        results = query.all()
        
        if not results:
            return {"message": "Bhai, is search par koi class nahi mili."}
            
        # 🧠 ASLI JADOO: Data ko Din (Day) ke hisab se group karna
        organized_schedule = {}
        
        for c in results:
            din = c.day # Monday, Tuesday etc.
            
            # Agar is din ka khana (key) nahi bana, toh pehle banao
            if din not in organized_schedule:
                organized_schedule[din] = []
                
            # Us din ke andar class ki detail daalo
            organized_schedule[din].append({
                "time": c.time,
                "subject": c.subject,
                "teacher": c.teacher_name,
                "room": c.room_number
            })
            
        return {
            "total_results": len(results),
            "schedule_by_day": organized_schedule # Ab data grouped aayega
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()