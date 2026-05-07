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