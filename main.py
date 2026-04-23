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
    day: Optional[str] = None,
    section: Optional[str] = None
):
    session = Session()
    try:
        query = session.query(Timetable)
        
        if room: query = query.filter(Timetable.room_number.ilike(f"%{room}%"))
        if teacher: query = query.filter(Timetable.teacher_name.ilike(f"%{teacher}%"))
        if subject: query = query.filter(Timetable.subject.ilike(f"%{subject}%"))
        if day: query = query.filter(Timetable.day.ilike(f"%{day}%"))
        if section: query = query.filter(Timetable.section.ilike(f"%{section}%")) # 👈 Section filter
            
        results = query.all()
        
        if not results:
            return {"message": "Bhai, is search par koi class nahi mili."}
            
        organized_schedule = {}
        
        for c in results:
            din = c.day 
            if din not in organized_schedule:
                organized_schedule[din] = []
                
            organized_schedule[din].append({
                "time": c.time,
                "subject": c.subject,
                "section": c.section,
                "teacher": c.teacher_name,
                "room": c.room_number
            })
            
        return {
            "total_results": len(results),
            "schedule_by_day": organized_schedule 
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()