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
import traceback
import threading
from typing import Optional, List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_

# ── Existing database imports (unchanged) ─────────────────────────────────────
from database.sql_db import engine
from database.models import Base, Timetable, ExamSeating

# ── LLM imports ───────────────────────────────────────────────────────────────
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── New modules ───────────────────────────────────────────────────────────────
from app.api.query_router import QueryRouter, QueryType, extract_day, extract_teacher, extract_section
from app.services.rag_pipeline import RAGPipeline
from app.services.free_slot_service import get_teacher_free_slots, get_section_free_slots, normalize_day
from app.services.exam_seating import replace_exam_seating_from_pdf, get_seating_by_roll
from app.services.exam_seating import get_seating_by_roll, format_seating_response
import re
import asyncio


load_dotenv()


async def run_blocking_with_timeout(func, *args, timeout: float = 12.0, **kwargs):
    """Run a blocking function in the default executor with an asyncio timeout.

    Returns the function result or raises asyncio.TimeoutError.
    """
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(*args, **kwargs)),
            timeout,
        )
    except asyncio.TimeoutError:
        raise

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
Base.metadata.create_all(bind=engine)

# ─── NEW: LLM + Pipelines (initialized once at startup) ──────────────────────

print("\n⏳ AskUni API initializing …")

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

llm_gemini = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    timeout=10,
    max_retries=1,
)
llm_gpt = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com",
    temperature=0,
    request_timeout=10,
    max_retries=1,
)
llm_groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0,
    request_timeout=10,
    max_retries=1,
)

def ask_llm(prompt):
    """Fallback mechanism: Groq -> GPT -> Gemini"""
    try:
        return llm_groq.invoke(prompt)
    except Exception:
        try:
            return llm_gpt.invoke(prompt)
        except Exception:
            return llm_gemini.invoke(prompt)

db_langchain   = SQLDatabase.from_uri(DATABASE_URL)
def build_sql_agent(llm):
    return create_sql_agent(
        llm=llm,
        db=db_langchain,
        agent_type="tool-calling",
        verbose=True,
        max_iterations=10,
        max_execution_time=20.0,
    )

agent_executor = build_sql_agent(llm_gpt)
agent_executor_fallback = None
agent_executor_fallback_groq = None

rag    = RAGPipeline(ask_llm)
router = QueryRouter()

print("✅ AskUni API Ready!\n")


# ─── EAGER RAG LOADING AT STARTUP ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_preload_rag():
    """
    Pre-load the RAG pipeline at startup to avoid first-request delay.
    Set RAG_PRELOAD=false to disable.
    """
    if os.getenv("RAG_PRELOAD", "true").lower() == "false":
        print("ℹ️ RAG preload disabled — will lazy-load on first RAG query.")
        return

    print("⏳ Pre-loading RAG pipeline at startup (in background) ...")
    _start_rag_loading()

# ─── SYSTEM CONTEXT (unchanged from chatbot.py) ───────────────────────────────

TIMETABLE_SYSTEM_CONTEXT = """
You are the official AskUni AI for FAST NUCES.
You have access to the chat history provided in the prompt, so you can understand
follow-up questions like 'why', 'who', or 'more details'.

1. Language: Reply in English if the user writes English; otherwise reply in Roman Urdu using Latin letters only (no Urdu/Arabic script).
2. Search: Always use ILIKE or regex (~*) for names/sections/campus.
3. If no data is found: Explain that "The database currently has no records for this."
4. If user asks for timetable or classes or class:
   - If section is provided, search in that section for whole week
   - If teacher is provided, search in that teacher for whole week
   - If both are provided, search in that teacher and section for whole week
   - If day is provided, search in that day only
   - If day and section are provided, search in that day and section only
   - If day and teacher are provided, search in that day and teacher only
   - If day and teacher and section are provided, search in that day and teacher and section only
5. Follow-ups: If a user asks 'why' after no data was found, explain that you can
   only provide information present in the database.
6. Handling Offensive Input:
   - If the user says anything unrelated or offensive, politely say:
     "Maazrat, main sirf FAST University ke timetable se mutaliq sawalaat ke jawab de sakta hoon."
   - If information is not in context, strictly say I don't know.
7. Use a single concise SQL query whenever possible. Avoid multi-step tool loops.
"""

# ─── SESSION MEMORY STORE ─────────────────────────────────────────────────────

session_store: dict = {}
pending_disambiguation: dict = {}
exam_import_status: dict = {
    "running": False,
    "status": "idle",
    "rows": 0,
    "error": "",
    "pdf_path": "",
    "exam_session": "",
    "campus": "",
}
MAX_HISTORY_CHARS = 2000

ROMAN_URDU_HINTS = re.compile(
    r"\b(kya|ky|kaise|kaisa|kaun|kis|ki|ka|ke|ko|hai|hain|ho|hoon|haan|nahi|kyun|kab|kahan|"
    r"sir|madam|teacher|ustad|batao|batayen|btado|plz|please|classes)\b",
    re.IGNORECASE,
)
URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
GREETING_PATTERN = re.compile(r"\b(hi|hello|hey|assalam|salam|aoa)\b", re.IGNORECASE)
CLASS_QUERY_PATTERN = re.compile(r"\b(class|classes|classe|clase|lecture|lectures|period|periods)\b", re.IGNORECASE)
FREE_SLOT_HINT_PATTERN = re.compile(r"\b(free|available|khali|slot|freee|fre)\b", re.IGNORECASE)
DISAMBIGUATION_LIST_PATTERN = re.compile(r"\b(Misal|Examples?)\s*:\s*(.+)$", re.IGNORECASE)
TEACHER_FULL_PATTERN = re.compile(
    r"\b(sir|dr\.?|mr\.?|ms\.?|prof\.?|professor)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})",
    re.IGNORECASE,
)
ROLL_NO_PATTERN = re.compile(r"\b(\d{2}[A-Za-z]-\d{4})\b")
TEACHER_STOP_WORDS = {
    "ki", "ka", "ke", "ko", "se", "mein", "me", "par", "pe",
    "kab", "konsi", "kon", "kaun", "classes", "class", "lecture",
    "lectures", "period", "periods", "hain", "hai", "ho",
    "unka", "unki", "unke",
}
TEACHER_ALIASES = {
    "atif luqman": "Atif",
    "atif": "Atif",
}
DAY_TYPO_MAP = {
    "moday": "monday",
    "mondy": "monday",
    "mondoy": "monday",
    "tuesay": "tuesday",
    "tusday": "tuesday",
    "wednsday": "wednesday",
    "thurday": "thursday",
    "frday": "friday",
    "satday": "saturday",
}
DAY_TOKENS = {
    "mon", "monday", "tue", "tuesday", "wed", "wednesday",
    "thu", "thursday", "fri", "friday", "sat", "saturday",
}

rag_loading = False
rag_lock = threading.Lock()


def detect_language(query: str) -> str:
    if URDU_SCRIPT.search(query):
        return "roman_urdu"
    if ROMAN_URDU_HINTS.search(query):
        return "roman_urdu"
    return "english"


def extract_teacher_full(query: str) -> str:
    match = TEACHER_FULL_PATTERN.search(query)
    if not match:
        return ""

    prefix = match.group(1)
    name_part = match.group(2)
    tokens = [t for t in name_part.split() if t]
    while tokens and (tokens[-1].lower() in TEACHER_STOP_WORDS or tokens[-1].lower() in DAY_TOKENS):
        tokens.pop()

    if not tokens:
        return ""
    return f"{prefix} {' '.join(tokens)}".strip()


def detect_day_in_text(query: str) -> str:
    lower = query.lower()
    for typo, fixed in DAY_TYPO_MAP.items():
        if typo in lower:
            return fixed
    match = re.search(r"\b(mon|monday|tue|tuesday|wed|wednesday|thu|thursday|fri|friday|sat|saturday)\b", lower)
    return match.group(0) if match else ""


def _extract_candidate_list(reply: str) -> List[str]:
    match = DISAMBIGUATION_LIST_PATTERN.search(reply)
    if not match:
        return []
    raw = match.group(2)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _store_disambiguation(session_id: str, intent: str, day_token: str, reply: str):
    candidates = _extract_candidate_list(reply)
    if not candidates:
        return
    pending_disambiguation[session_id] = {
        "intent": intent,
        "day": day_token,
        "candidates": candidates,
    }


def _try_resolve_disambiguation(session_id: str, question: str, language_hint: str) -> Optional[str]:
    pending = pending_disambiguation.get(session_id)
    if not pending:
        return None

    lower = question.strip().lower()
    if not lower:
        return None

    matches = [c for c in pending["candidates"] if lower in c.lower()]
    if len(matches) == 1:
        teacher_name = matches[0]
        day_token = pending.get("day", "")
        if not day_token:
            if language_hint == "roman_urdu":
                return "Kaunsa din? Monday, Tuesday waghera likhen."
            return "Which day? Please specify a day like Monday, Tuesday, etc."

        intent = pending["intent"]
        del pending_disambiguation[session_id]
        if intent == "FREE_SLOT":
            return get_teacher_free_slots(teacher_name, day_token, language_hint)
        if intent == "TIMETABLE":
            return fetch_teacher_classes(teacher_name, day_token, language_hint)
        return None

    if len(matches) > 1:
        names = ", ".join(matches[:5])
        if language_hint == "roman_urdu":
            return (
                "Is naam se multiple teachers mil rahe hain. Full name likhen. "
                f"Misal: {names}"
            )
        return (
            "This name matches multiple teachers. Please provide the full name. "
            f"Examples: {names}"
        )

    return None


def fetch_teacher_classes(teacher_name: str, day_token: str, language_hint: str) -> str:
    canonical_day = normalize_day(day_token)
    if not canonical_day:
        if language_hint == "roman_urdu":
            return f"'{day_token}' koi theek din nahi hai. Monday se Saturday tak likhen."
        return f"'{day_token}' is not a recognized day. Please use Monday–Saturday."

    cleaned_teacher = re.sub(r"\b(sir|dr\.?|mr\.?|ms\.?|prof\.?|professor)\b", "", teacher_name, flags=re.IGNORECASE)
    cleaned_teacher = re.sub(r"\s+", " ", cleaned_teacher).strip()
    alias_key = cleaned_teacher.lower()
    alias_hit = False
    if alias_key in TEACHER_ALIASES:
        cleaned_teacher = TEACHER_ALIASES[alias_key]
        alias_hit = True
    tokens = [t for t in cleaned_teacher.split(" ") if t]
    teacher_query = cleaned_teacher or teacher_name

    session = Session()
    default_notice = ""
    use_exact_match = False
    try:
        candidates = []
        if tokens and not alias_hit:
            if len(tokens) == 1:
                token = tokens[0].lower()
                exact_row = (
                    session.query(Timetable.teacher_name)
                    .filter(Timetable.teacher_name.ilike(token))
                    .first()
                )
                if exact_row:
                    teacher_query = exact_row[0]
                    tokens = []
                    use_exact_match = True
            query = session.query(Timetable.teacher_name).distinct()
            for token in tokens:
                query = query.filter(Timetable.teacher_name.ilike(f"%{token}%"))
            candidates = [row[0] for row in query.limit(5).all()]

            if len(candidates) == 1:
                teacher_query = candidates[0]
            elif len(candidates) > 1:
                if len(tokens) == 1:
                    token = tokens[0].lower()
                    exact = [c for c in candidates if c.strip().lower() == token]
                    if exact:
                        teacher_query = exact[0]
                        use_exact_match = True
                    else:
                        teacher_query = sorted(candidates, key=lambda c: (len(c), c.lower()))[0]
                    names = ", ".join(candidates[:5])
                    if language_hint == "roman_urdu":
                        default_notice = (
                            "\nAgar aap kisi aur teacher ka pooch rahe hain to full name likhen. "
                            f"Misal: {names}"
                        )
                    else:
                        default_notice = (
                            "\nIf you meant a different teacher, please provide the full name. "
                            f"Examples: {names}"
                        )
                else:
                    if language_hint == "roman_urdu":
                        return "Ye naam multiple teachers se match ho raha hai. Full name likhen."
                    return "This name matches multiple teachers. Please provide the full name."
            elif len(tokens) > 1:
                or_query = session.query(Timetable.teacher_name).distinct()
                or_filters = [Timetable.teacher_name.ilike(f"%{t}%") for t in tokens]
                or_query = or_query.filter(or_(*or_filters))
                or_candidates = [row[0] for row in or_query.limit(10).all()]
                if len(or_candidates) == 1:
                    teacher_query = or_candidates[0]
                elif len(or_candidates) > 1:
                    names = ", ".join(or_candidates[:5])
                    if language_hint == "roman_urdu":
                        return (
                            "Is naam se multiple teachers mil rahe hain. Full name likhen. "
                            f"Misal: {names}"
                        )
                    return (
                        "This name matches multiple teachers. Please provide the full name. "
                        f"Examples: {names}"
                    )

        query_base = session.query(Timetable.subject, Timetable.time, Timetable.room_number, Timetable.section)
        query_base = query_base.filter(Timetable.day.ilike(canonical_day))
        if alias_hit or use_exact_match:
            query_base = query_base.filter(Timetable.teacher_name.ilike(teacher_query))
        else:
            query_base = query_base.filter(Timetable.teacher_name.ilike(f"%{teacher_query}%"))
        results = query_base.order_by(Timetable.time).all()
    except Exception:
        if language_hint == "roman_urdu":
            return "Schedule nahi mil saka. Baad mein dobara try karein."
        return "Could not retrieve schedule. Please try again later."
    finally:
        session.close()

    if not results:
        if language_hint == "roman_urdu":
            return "Database mein is naam ke liye record nahi mila."
        return "The database currently has no records for this."

    grouped = {}
    for subject, time_val, room, section in results:
        key = subject or "Unknown"
        grouped.setdefault(key, []).append((time_val, room, section))

    if language_hint == "roman_urdu":
        header = f"{teacher_query} ki {canonical_day} ko classes yeh hain:"
    else:
        header = f"Classes for {teacher_query} on {canonical_day}:"

    lines = [header, ""]
    idx = 1
    for subject, items in grouped.items():
        lines.append(f"{idx}. **{subject}**")
        for time_val, room, section in items:
            lines.append(f"   - Time: {time_val}")
            if room:
                lines.append(f"   - Room: {room}")
            if section:
                lines.append(f"   - Section: {section}")
        lines.append("")
        idx += 1

    return "\n".join(lines).strip() + default_notice


def _start_rag_loading():
    global rag_loading
    with rag_lock:
        if rag_loading or rag._retriever is not None:
            return
        rag_loading = True

    def _load():
        global rag_loading
        try:
            rag._ensure_loaded()
        finally:
            rag_loading = False

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()

def get_history(session_id: str) -> str:
    return session_store.get(session_id, "")

def update_history(session_id: str, question: str, answer: str):
    history = session_store.get(session_id, "")
    history += f"User: {question}\nBot: {answer}\n"
    if len(history) > MAX_HISTORY_CHARS:
        history = history[-MAX_HISTORY_CHARS:]
    session_store[session_id] = history

# ─── FREE SLOT HANDLER (unchanged from chatbot.py) ────────────────────────────

def handle_free_slot(query: str, language_hint: str) -> str:
    day     = extract_day(query)
    teacher = extract_teacher(query)
    section = extract_section(query)

    if not day:
        if language_hint == "english":
            return (
                "Which day? Please specify a day like Monday, Tuesday, etc.\n"
                "Example: 'Is Sir Shakeel free on Monday?'"
            )
        return (
            "Kaunsa din? Monday, Tuesday waghera likhen.\n"
            "Example: 'Sir Shakeel Monday ko free kab hai?'"
        )
    if section:
        return get_section_free_slots(section, day, language_hint)
    if teacher:
        return get_teacher_free_slots(teacher, day, language_hint)
    if language_hint == "english":
        return (
            "Please mention a teacher name (e.g., 'Sir Shakeel') "
            "or a section (e.g., 'BSCS-5A') along with the day."
        )
    return (
        "Teacher name (e.g., 'Sir Shakeel') ya section (e.g., 'BSCS-5A') aur din bhi likhen."
    )

def handle_seating_query(question: str, language_hint: str) -> Optional[str]:
    """
    If user's message contains a roll number, fetch and return their seating plan.
    Returns None if no roll number found (so routing continues normally).
 
    Example queries handled:
        "my roll number is 24K-0030"
        "seating plan for 21K-3600"
        "24K-0516 ka exam plan batao"
        "tell me my seat, roll no 24K-0030"
    """
    match = ROLL_NO_PATTERN.search(question)
    if not match:
        return None
 
    roll_no = match.group(1)
    db_session = Session()
    try:
        records = get_seating_by_roll(db_session, roll_no)
        return format_seating_response(records, language_hint)
    finally:
        db_session.close()

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


class ExamImportRequest(BaseModel):
    pdf_path: str
    exam_session: str = "Sessional/Final"
    campus: str = "Karachi"


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

    global agent_executor_fallback
    global agent_executor_fallback_groq

    try:
        language_hint = detect_language(question)
        has_day = bool(extract_day(question)) or bool(detect_day_in_text(question))
        has_section = bool(extract_section(question))
        has_free_hint = bool(FREE_SLOT_HINT_PATTERN.search(question))
        has_class_hint = bool(CLASS_QUERY_PATTERN.search(question))

        if session_id in pending_disambiguation and not (has_day or has_section or has_free_hint or has_class_hint):
            pending_intent = pending_disambiguation[session_id].get("intent", "TIMETABLE")
            resolved_reply = _try_resolve_disambiguation(session_id, question, language_hint)
            if resolved_reply:
                update_history(session_id, question, resolved_reply)
                return ChatResponse(
                    reply=resolved_reply,
                    session_id=session_id,
                    query_type=pending_intent,
                )

        seating_reply = handle_seating_query(question, language_hint)
        if seating_reply is not None:
            update_history(session_id, question, seating_reply)
            return ChatResponse(
                reply=seating_reply,
                session_id=session_id,
                query_type="SEATING",
            )
        query_type = router.route(question)

        if GREETING_PATTERN.search(question) and len(question.split()) <= 3:
            if language_hint == "english":
                bot_reply = "Hi! I am AskUni for FAST NUCES Karachi. How can I help?"
            else:
                bot_reply = "Salam! Main AskUni hoon FAST NUCES Karachi ke liye. Aap ki madad kar sakta hoon."
            update_history(session_id, question, bot_reply)
            return ChatResponse(
                reply=bot_reply,
                session_id=session_id,
                query_type="GREETING",
            )

        # ── TIMETABLE ──────────────────────────────────────────────────────────
        if query_type == QueryType.TIMETABLE:
            teacher_in_query = extract_teacher_full(question) or extract_teacher(question)
            day_token = detect_day_in_text(question)
            if teacher_in_query and day_token:
                bot_reply = fetch_teacher_classes(teacher_in_query, day_token, language_hint)
                _store_disambiguation(session_id, "TIMETABLE", day_token, bot_reply)
                update_history(session_id, question, bot_reply)
                return ChatResponse(
                    reply=bot_reply,
                    session_id=session_id,
                    query_type="TIMETABLE",
                )
            full_prompt = (
                f"{TIMETABLE_SYSTEM_CONTEXT}\n\n"
                f"Language preference: {language_hint} (English or Roman Urdu in Latin letters only).\n\n"
                f"--- CHAT HISTORY ---\n{chat_history}\n\n"
                f"--- NEW QUESTION ---\nUser: {question}"
            )
            try:
                try:
                    response = await run_blocking_with_timeout(agent_executor.invoke, {"input": full_prompt}, timeout=8.0)
                    bot_reply = response["output"] if isinstance(response, dict) and "output" in response else response
                except Exception:
                    if agent_executor_fallback is None:
                        agent_executor_fallback = build_sql_agent(llm_gemini)
                    try:
                        response = await run_blocking_with_timeout(agent_executor_fallback.invoke, {"input": full_prompt}, timeout=8.0)
                        bot_reply = response["output"] if isinstance(response, dict) and "output" in response else response
                    except Exception:
                        if agent_executor_fallback_groq is None:
                            agent_executor_fallback_groq = build_sql_agent(llm_groq)
                        try:
                            response = await run_blocking_with_timeout(agent_executor_fallback_groq.invoke, {"input": full_prompt}, timeout=8.0)
                            bot_reply = response["output"] if isinstance(response, dict) and "output" in response else response
                        except asyncio.TimeoutError:
                            bot_reply = "Model timeout. Please try again later."
                        except Exception:
                            bot_reply = "Model error. Please try again later."
            except asyncio.TimeoutError:
                bot_reply = "Model timeout. Please try again later."

        # ── FREE SLOT ──────────────────────────────────────────────────────────
        elif query_type == QueryType.FREE_SLOT:
            bot_reply = handle_free_slot(question, language_hint)
            day_token = extract_day(question) or detect_day_in_text(question)
            _store_disambiguation(session_id, "FREE_SLOT", day_token, bot_reply)

        # ── KNOWLEDGE (RAG) ────────────────────────────────────────────────────
        else:
            if rag._retriever is None:
                _start_rag_loading()
                if language_hint == "english":
                    bot_reply = "Knowledge base is loading. Please try again in a few seconds."
                else:
                    bot_reply = "Knowledge base load ho raha hai. Thori dair baad dobara try karein."
            else:
                try:
                    bot_reply = await run_blocking_with_timeout(rag.ask, question, chat_history, language_hint, timeout=12.0)
                except asyncio.TimeoutError:
                    bot_reply = "Knowledge query timed out. Please try again later."
                except Exception:
                    bot_reply = "Knowledge backend error. Please try again later."

        update_history(session_id, question, bot_reply)

        return ChatResponse(
            reply=bot_reply,
            session_id=session_id,
            query_type=query_type.name,
        )

    except Exception as e:
        traceback.print_exc()
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


@app.post("/exam/import")
def import_exam_seating(payload: ExamImportRequest):
    if exam_import_status["running"]:
        return {
            "status": "running",
            "message": "Import is already in progress.",
            "details": exam_import_status,
        }

    exam_import_status.update({
        "running": True,
        "status": "running",
        "rows": 0,
        "error": "",
        "pdf_path": payload.pdf_path,
        "exam_session": payload.exam_session,
        "campus": payload.campus,
    })

    def _worker():
        try:
            count = replace_exam_seating_from_pdf(
                pdf_path=payload.pdf_path,
                session_name=payload.exam_session,
                campus=payload.campus,
            )
            exam_import_status.update({
                "running": False,
                "status": "completed",
                "rows": count,
                "error": "",
            })
        except Exception as e:
            exam_import_status.update({
                "running": False,
                "status": "failed",
                "rows": 0,
                "error": str(e),
            })

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    return {
        "status": "accepted",
        "message": "Import started in background. Check /exam/import/status",
        "details": {
            "pdf_path": payload.pdf_path,
            "exam_session": payload.exam_session,
            "campus": payload.campus,
        },
    }


@app.get("/exam/import/status")
def get_exam_import_status():
    return exam_import_status


@app.get("/exam/seat/{roll_no}")
def get_exam_seat(roll_no: str, exam_session: Optional[str] = None):
    db_session = Session()
    try:
        rows = get_seating_by_roll(db_session, roll_no)  # ← session add kiya
        if exam_session:
            rows = [r for r in rows if r.get("exam_session") == exam_session]
        if not rows:
            return {"message": "No seating plan found for this roll number."}

        return {"count": len(rows), "records": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()  # ← session close karna zaroori hai