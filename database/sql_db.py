from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

engine = None
if DB_URL:
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as connection:
            print("[DB] PostgreSQL connection successful.")
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
else:
    print("[DB] Warning: DATABASE_URL not set in environment.")