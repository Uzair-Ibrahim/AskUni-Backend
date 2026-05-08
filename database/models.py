from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Timetable(Base):
    
    __tablename__ = "university_timetable"

    id = Column(Integer, primary_key=True, index=True)
    day = Column(String, index=True)
    time = Column(String)
    subject = Column(String)
    teacher_name = Column(String, index=True)
    room_number = Column(String, index=True)
    section = Column(String, index=True)
    campus = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_day_section', 'day', 'section'),
        Index('idx_campus_section', 'campus', 'section'),
        Index('idx_day_time', 'day', 'time'),
    )


class ExamSeating(Base):
    """
    One row = one exam slot for one student.
    A student with 6 courses will have 6 rows.
    """
    __tablename__ = "exam_seating"
 
    id           = Column(Integer, primary_key=True, index=True)
    roll_no      = Column(String, index=True, nullable=False)   # e.g. 24K-0030
    student_name = Column(String, index=True)
    course_code  = Column(String, index=True)                   # e.g. CS2006
    section      = Column(String, index=True)                   # e.g. BCS-4F
    course_name  = Column(String)
    day          = Column(String)                               # e.g. 09-Apr-26
    time         = Column(String)                               # e.g. 08:30-09:30
    teacher      = Column(String)
    seat         = Column(String)                               # e.g. AB2RoomB 11
    exam_session = Column(String, index=True)                   # e.g. Sessional II Spring 2026
    campus       = Column(String, index=True, default="Karachi")
    created_at   = Column(DateTime, default=datetime.utcnow)
 
    __table_args__ = (
        Index("idx_exam_roll",        "roll_no"),
        Index("idx_exam_session",     "exam_session"),
        Index("idx_exam_roll_session","roll_no", "exam_session"),
        Index("idx_exam_course",      "course_code"),
    )