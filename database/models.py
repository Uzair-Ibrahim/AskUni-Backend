from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Timetable(Base):
    
    __tablename__ = "university_timetable"

    id = Column(Integer, primary_key=True, index=True)
    day = Column(String)
    time = Column(String)
    subject = Column(String)       # 👈 Check karo ye String hi ho
    teacher_name = Column(String)
    room_number = Column(String)   # 👈 Ye lazmi String hona chahiye (kyunke isme "E-1" wagera aata hai)
    section = Column(String)
    campus = Column(String)