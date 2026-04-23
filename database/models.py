from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

# Yeh hamara 'Sancha' (Mould/Blueprint) hai
Base = declarative_base()

# Yeh hamari Timetable ki Almari (Table) hai
class Timetable(Base):
    # Database mein is table ka naam kya hoga:
    __tablename__ = "university_timetable"

    # Ab hum columns (khanay) bana rahe hain
    id = Column(Integer, primary_key=True, index=True) # Yeh har row ka unique number hoga (1, 2, 3...)
    day = Column(String)                               # Din (e.g., "Monday")
    time = Column(String)                              # Waqt (e.g., "10:00 AM")
    subject = Column(String)                           # Subject ka naam (e.g., "Software Engineering")
    teacher_name = Column(String)                      # Teacher ka naam
    room_number = Column(String)                       # Kamra number (e.g., "Room 302")