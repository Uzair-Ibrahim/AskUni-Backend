"""
main.py — AskUni FastAPI Backend
==================================
Two sets of endpoints:

  EXISTING (unchanged):
    GET  /              → welcome message
    GET  /search        → timetable search by room/teacher/subject/day/section

  NEW:
    POST /api/chat      → AI chatbot (timetable + free slots + RAG knowledge)
    GET  /api/health    → health check
    GET  /api/clear-history → reset chat memory for a session
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

# ── Existing database imports (unchanged) ─────────────────────────────────────
from database.sql_db import engine
from database.models import Timetable

# ── LLM imports ───────────────────────────────────────────────────────────────
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── New modules ───────────────────────────────────────────────────────────────
from query_router import QueryRouter, QueryType, extract_day, extract_teacher, extract_section
from rag_pipeline import RAGPipeline
from free_slot_service import get_teacher_free_slots, get_section_free_slots

load_dotenv()

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AskUni API",
    description="FAST NUCES Karachi — Smart Search + AI Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # React dev server
        "http://localhost:5173",       # Vite dev server
        "http://localhost:8080",       # Vue dev server
        os.getenv("FRONTEND_URL", ""), # production frontend URL from .env
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── EXISTING: SQLAlchemy session (unchanged) ─────────────────────────────────

Session = sessionmaker(bind=engine)

# ─── NEW: LLM + Pipelines (initialized once at startup) ──────────────────────

print("\n⏳ AskUni API initializing …")

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)
llm_gpt = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
    temperature=0,
)
llm_groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0,
)

def ask_llm(prompt):
    """Fallback mechanism: Gemini -> GPT -> Groq"""
    try:
        return llm_gemini.invoke(prompt)
    except:
        try:
            return llm_gpt.invoke(prompt)
        except:
            return llm_groq.invoke(prompt)

db_langchain   = SQLDatabase.from_uri(DATABASE_URL)
agent_executor = create_sql_agent(
    llm=llm_gemini,
    db=db_langchain,
    agent_type="tool-calling",
    verbose=False,
)

rag    = RAGPipeline(ask_llm)
router = QueryRouter()

print("✅ AskUni API Ready!\n")

# ─── SYSTEM CONTEXT (unchanged from chatbot.py) ───────────────────────────────

TIMETABLE_SYSTEM_CONTEXT = """
You are the official AskUni AI for FAST NUCES.
You have access to the chat history provided in the prompt, so you can understand
follow-up questions like 'why', 'who', or 'more details'.

1. Language: Detect user language (English or Roman Urdu) and reply in the same.
2. Search: Always use ILIKE or regex (~*) for names/sections/campus.
3. If no data is found: Explain that "The database currently has no records for this."
4. Follow-ups: If a user asks 'why' after no data was found, explain that you can
   only provide information present in the database.
5. Handling Offensive Input:
   - If the user says anything unrelated or offensive, politely say:
     "Maazrat, main sirf FAST University ke timetable se mutaliq sawalaat ke jawab de sakta hoon."
   - If information is not in context, strictly say I don't know.
"""

# ─── SESSION MEMORY STORE ─────────────────────────────────────────────────────

session_store: dict = {}
MAX_HISTORY_CHARS = 2000

def get_history(session_id: str) -> str:
    return session_store.get(session_id, "")

def update_history(session_id: str, question: str, answer: str):
    history = session_store.get(session_id, "")
    history += f"User: {question}\nBot: {answer}\n"
    if len(history) > MAX_HISTORY_CHARS:
        history = history[-MAX_HISTORY_CHARS:]
    session_store[session_id] = history

# ─── FREE SLOT HANDLER (unchanged from chatbot.py) ────────────────────────────

def handle_free_slot(query: str) -> str:
    day     = extract_day(query)
    teacher = extract_teacher(query)
    section = extract_section(query)

    if not day:
        return (
            "Kaunsa din? Please specify a day like Monday, Tuesday, etc.\n"
            "Example: 'Sir Shakeel free kab hai Monday ko?'"
        )
    if section:
        return get_section_free_slots(section, day)
    if teacher:
        return get_teacher_free_slots(teacher, day)
    return (
        "Please mention a teacher name (e.g., 'Sir Shakeel') "
        "or a section (e.g., 'BSCS-5A') along with the day."
    )

# =============================================================================
# EXISTING ENDPOINTS (completely unchanged)
# =============================================================================

@app.get("/")
def read_root():
    return {"message": "Welcome to AskUni Smart Search! 🚀"}


@app.get("/search")
def search_timetable(
    room: Optional[str] = None,
    teacher: Optional[str] = None,
    subject: Optional[str] = None,
    day: Optional[str] = None,
    section: Optional[str] = None  # 👈 Naya Parameter
):
    session = Session()
    try:
        query = session.query(Timetable)

        if room:    query = query.filter(Timetable.room_number.ilike(f"%{room}%"))
        if teacher: query = query.filter(Timetable.teacher_name.ilike(f"%{teacher}%"))
        if subject: query = query.filter(Timetable.subject.ilike(f"%{subject}%"))
        if day:     query = query.filter(Timetable.day.ilike(f"%{day}%"))
        if section: query = query.filter(Timetable.section.ilike(f"%{section}%"))  # 👈 Section filter

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
                "section": c.section,  # 👈 API response mein section
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


# =============================================================================
# NEW ENDPOINTS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # frontend sends this to maintain history


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    query_type: str  # "TIMETABLE" | "FREE_SLOT" | "KNOWLEDGE"


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Frontend sends { message, session_id } and receives { reply, session_id, query_type }.
    On first message, omit session_id — a new one will be returned; save it for follow-ups.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id   = req.session_id or str(uuid.uuid4())
    chat_history = get_history(session_id)
    question     = req.message.strip()

    try:
        query_type = router.route(question)

        # ── TIMETABLE ──────────────────────────────────────────────────────────
        if query_type == QueryType.TIMETABLE:
            full_prompt = (
                f"{TIMETABLE_SYSTEM_CONTEXT}\n\n"
                f"--- CHAT HISTORY ---\n{chat_history}\n\n"
                f"--- NEW QUESTION ---\nUser: {question}"
            )
            response  = agent_executor.invoke({"input": full_prompt})
            bot_reply = response["output"]

        # ── FREE SLOT ──────────────────────────────────────────────────────────
        elif query_type == QueryType.FREE_SLOT:
            bot_reply = handle_free_slot(question)

        # ── KNOWLEDGE (RAG) ────────────────────────────────────────────────────
        else:
            bot_reply = rag.ask(question, chat_history)

        update_history(session_id, question, bot_reply)

        return ChatResponse(
            reply=bot_reply,
            session_id=session_id,
            query_type=query_type.name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clear-history")
async def clear_history(session_id: str):
    """Reset chat memory for a given session."""
    if session_id in session_store:
        del session_store[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/health")
async def health():
    """Health check — frontend or load balancer can ping this."""
    return {
        "status":          "ok",
        "service":         "AskUni API",
        "rag_loaded":      rag._store is not None,
        "active_sessions": len(session_store),
    }