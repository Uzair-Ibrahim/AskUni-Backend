from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

try:
    engine = create_engine(DB_URL)
    with engine.connect() as connection:
        print("✅ Mashallah! PostgreSQL se connection successful ho gaya hai!")
except Exception as e:
    print(f"❌ Connection fail ho gaya. Error: {e}")