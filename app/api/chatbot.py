"""
chatbot.py  (UPDATED — backward compatible)
============================================
AskUni chatbot with three integrated capabilities:

  1. 📅 TIMETABLE queries  → LangChain SQL agent (EXISTING, unchanged)
  2. 🟢 FREE SLOT queries  → free_slot_service  (NEW)
  3. 📚 KNOWLEDGE queries  → RAG pipeline        (NEW)

The existing SQL agent logic is preserved exactly as it was.
New modules are injected around it via the QueryRouter.
"""

import os
import re
from dotenv import load_dotenv

# ── Existing imports (unchanged) ──────────────────────────────────────────────
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ── New imports ───────────────────────────────────────────────────────────────
from app.api.query_router import QueryRouter, QueryType, extract_day, extract_teacher, extract_section
from app.services.rag_pipeline import RAGPipeline
from app.services.free_slot_service import get_teacher_free_slots, get_section_free_slots, normalize_day

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ─── SYSTEM CONTEXT (existing, unchanged) ─────────────────────────────────────

TIMETABLE_SYSTEM_CONTEXT = """
You are the official AskUni AI for FAST NUCES.
You have access to the chat history provided in the prompt, so you can understand
follow-up questions like 'why', 'who', or 'more details'.

1. Language: Reply in English if the user writes English; otherwise reply in Roman Urdu using Latin letters only (no Urdu/Arabic script).
2. Search: Always use ILIKE or regex (~*) for names/sections/campus.
3. If no data is found: Explain that "The database currently has no records for this."
4. Follow-ups: If a user asks 'why' after no data was found, explain that you can
   only provide information present in the database.
5. Handling Offensive Input:
   - If the user says anything unrelated or offensive, politely say:
     "Maazrat, main sirf FAST University ke timetable se mutaliq sawalaat ke jawab de sakta hoon."
   - If information is not in context, strictly say I don't know.
"""


# ─── FREE SLOT HANDLER ────────────────────────────────────────────────────────

ROMAN_URDU_HINTS = re.compile(
    r"\b(kya|ky|kaise|kaisa|kaun|kis|ki|ka|ke|ko|hai|hain|ho|hoon|haan|nahi|kyun|kab|kahan|"
    r"sir|madam|teacher|ustad|batao|batayen|btado|plz|please|classes)\b",
    re.IGNORECASE,
)
URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")
GREETING_PATTERN = re.compile(r"\b(hi|hello|hey|assalam|salam|aoa)\b", re.IGNORECASE)
PRONOUN_FOLLOWUP = re.compile(r"\b(unke|unki|unka|uska|uski|uske|us|un|inhon|inki|inke)\b", re.IGNORECASE)
FACULTY_NAME_PATTERN = re.compile(
    r"\b(sir|dr\.?|mr\.?|ms\.?|prof\.?|professor)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,3})",
    re.IGNORECASE,
)
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


def detect_language(query: str) -> str:
    if URDU_SCRIPT.search(query):
        return "roman_urdu"
    if ROMAN_URDU_HINTS.search(query):
        return "roman_urdu"
    return "english"


def extract_faculty_name(query: str) -> str:
    match = FACULTY_NAME_PATTERN.search(query)
    if not match:
        return ""
    prefix = match.group(1).strip()
    name = match.group(2).strip()
    return f"{prefix} {name}".strip()


def detect_day_in_text(query: str) -> str:
    lower = query.lower()
    for typo, fixed in DAY_TYPO_MAP.items():
        if typo in lower:
            return fixed
    match = re.search(r"\b(mon|monday|tue|tuesday|wed|wednesday|thu|thursday|fri|friday|sat|saturday)\b", lower)
    return match.group(0) if match else ""


def fetch_teacher_classes(teacher_name: str, day_token: str, language_hint: str) -> str:
    import psycopg2

    canonical_day = normalize_day(day_token)
    if not canonical_day:
        if language_hint == "roman_urdu":
            return f"'{day_token}' koi theek din nahi hai. Monday se Saturday tak likhen."
        return f"'{day_token}' is not a recognized day. Please use Monday–Saturday."

    cleaned_teacher = re.sub(r"\b(sir|dr\.?|mr\.?|ms\.?|prof\.?|professor)\b", "", teacher_name, flags=re.IGNORECASE)
    cleaned_teacher = re.sub(r"\s+", " ", cleaned_teacher).strip()
    teacher_query = cleaned_teacher or teacher_name
    tokens = [t for t in teacher_query.split(" ") if t]

    resolved_teacher = None
    if tokens:
        clauses = " AND ".join(["teacher_name ILIKE %s" for _ in tokens])
        resolve_sql = f"""
            SELECT DISTINCT teacher_name
            FROM university_timetable
            WHERE {clauses}
            LIMIT 5
        """
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cur = conn.cursor()
            cur.execute(resolve_sql, tuple(f"%{t}%" for t in tokens))
            candidates = [row[0] for row in cur.fetchall() if row and row[0]]
            cur.close()
            conn.close()
        except Exception:
            candidates = []

        if len(candidates) == 1:
            resolved_teacher = candidates[0]
        elif len(candidates) > 1:
            if language_hint == "roman_urdu":
                return "Ye naam multiple teachers se match ho raha hai. Full name likhen."
            return "This name matches multiple teachers. Please provide the full name."
        elif len(tokens) > 1:
            or_clauses = " OR ".join(["teacher_name ILIKE %s" for _ in tokens])
            resolve_sql_or = f"""
                SELECT DISTINCT teacher_name
                FROM university_timetable
                WHERE {or_clauses}
                LIMIT 10
            """
            try:
                conn = psycopg2.connect(os.getenv("DATABASE_URL"))
                cur = conn.cursor()
                cur.execute(resolve_sql_or, tuple(f"%{t}%" for t in tokens))
                or_candidates = [row[0] for row in cur.fetchall() if row and row[0]]
                cur.close()
                conn.close()
            except Exception:
                or_candidates = []

            if len(or_candidates) == 1:
                resolved_teacher = or_candidates[0]
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

    if resolved_teacher:
        teacher_query = resolved_teacher

    sql = """
        SELECT subject, time, room_number, section
        FROM university_timetable
        WHERE LOWER(day) = LOWER(%s)
          AND teacher_name ILIKE %s
        ORDER BY time
    """

    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute(sql, (canonical_day, f"%{teacher_query}%"))
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        if language_hint == "roman_urdu":
            return "Schedule nahi mil saka. Baad mein dobara try karein."
        return "Could not retrieve schedule. Please try again later."

    if not rows:
        if language_hint == "roman_urdu":
            return "The database currently has no records for this."
        return "The database currently has no records for this."

    grouped = {}
    for subject, time_val, room, section in rows:
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
            if language_hint == "roman_urdu":
                lines.append(f"   - Time: {time_val}")
            else:
                lines.append(f"   - Time: {time_val}")
            if room:
                lines.append(f"   - Room: {room}")
            if section:
                lines.append(f"   - Section: {section}")
        lines.append("")
        idx += 1

    return "\n".join(lines).strip()


def handle_free_slot(query: str, language_hint: str) -> str:
    """
    Determine whether the free-slot query is for a teacher or a section,
    extract the relevant name and day, then call the service.
    """
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

    # Section query takes priority if a section pattern is found
    if section:
        return get_section_free_slots(section, day, language_hint)

    if teacher:
        return get_teacher_free_slots(teacher, day, language_hint)

    # Nothing extracted clearly — ask the user
    if language_hint == "english":
        return (
            "Please mention a teacher name (e.g., 'Sir Shakeel') "
            "or a section (e.g., 'BSCS-5A') along with the day."
        )
    return (
        "Teacher name (e.g., 'Sir Shakeel') ya section (e.g., 'BSCS-5A') aur din bhi likhen."
    )


# ─── MAIN BOT ─────────────────────────────────────────────────────────────────

try:
    print("\n⏳ 1. Connection setup ho raha hai...")

    # ── 1. MODEL CONFIGURATION (with fallback) ────────────────────────────────
    llm_gemini = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )
    llm_gpt = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=GITHUB_TOKEN,
        base_url="https://models.inference.ai.azure.com",
        temperature=0
    )
    llm_groq = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    def ask_llm(prompt):
        """Fallback mechanism: Gemini -> GPT -> Groq"""
        try:
            return llm_groq.invoke(prompt)
        except:
            try:
                return llm_gpt.invoke(prompt)
            except:
                return llm_gemini.invoke(prompt)

    def ensure_roman_urdu(text: str) -> str:
        if not URDU_SCRIPT.search(text):
            return text
        try:
            rewrite_prompt = (
                "Rewrite the following in Roman Urdu using Latin letters only. "
                "Do not add new facts.\n\n"
                f"Text:\n{text}"
            )
            rewritten = ask_llm(rewrite_prompt)
            if hasattr(rewritten, "content"):
                return rewritten.content.strip()
            return str(rewritten).strip()
        except Exception:
            return text

    # ── 2. Existing SQL agent (unchanged) — Gemini primary, fallback available ─
    db = SQLDatabase.from_uri(DATABASE_URL)

    def build_sql_agent(llm):
        return create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=False
        )

    agent_executor = build_sql_agent(llm_gemini)   # Gemini is primary for SQL
    agent_executor_fallback = None

    # ── 3. RAG pipeline — receives ask_llm so it uses the same fallback chain ──
    rag = RAGPipeline(ask_llm)

    # ── 4. Query router ───────────────────────────────────────────────────────
    router = QueryRouter()

    print("✅ AskUni Bot Ready with Custom Memory!")
    print("="*50)

    # 🧠 HUMARI CUSTOM MEMORY VARIABLE (existing, unchanged)
    chat_history = ""
    last_faculty_name = ""
    pending_free_slot = False
    last_free_slot_teacher = ""
    last_free_slot_section = ""
    last_timetable_teacher = ""
    last_timetable_section = ""

    while True:
        question = input("\n🗣️ Aap: ")

        exit_words = ['exit', 'quit', 'bye', 'khuda hafiz', 'allah hafiz', 'khatam']
        if any(word in question.lower() for word in exit_words):
            print("\n🤖 AskUni Bot: Allah Hafiz! Apna khayal rakhna.")
            break

        if not question.strip(): continue

        try:
            print("⏳ Bot soch raha hai...")

            language_hint = detect_language(question)
            query_type = router.route(question)
            day_in_query = extract_day(question)
            teacher_in_query = extract_teacher(question)
            section_in_query = extract_section(question)
            day_token = day_in_query or detect_day_in_text(question)

            faculty_name = extract_faculty_name(question)
            if faculty_name:
                last_faculty_name = faculty_name

            if GREETING_PATTERN.search(question) and len(question.split()) <= 3:
                if language_hint == "english":
                    print("\n🤖 AskUni Bot: Hi! I am AskUni for FAST NUCES Karachi. How can I help?")
                else:
                    print("\n🤖 AskUni Bot: Salam! Main AskUni hoon FAST NUCES Karachi ke liye. Aap ki madad kar sakta hoon.")
                continue

            if pending_free_slot and day_in_query and not teacher_in_query and not section_in_query:
                if last_free_slot_section:
                    bot_reply = get_section_free_slots(last_free_slot_section, day_in_query, language_hint)
                else:
                    bot_reply = get_teacher_free_slots(last_free_slot_teacher, day_in_query, language_hint)
                pending_free_slot = False
                print(f"\n🤖 AskUni Bot: {bot_reply}")
                chat_history += f"User: {question}\nBot: {bot_reply}\n"
                if len(chat_history) > 2000:
                    chat_history = chat_history[-2000:]
                continue

            # ── TIMETABLE ──────────────────────────────────────────────────────
            if query_type == QueryType.TIMETABLE:
                if teacher_in_query:
                    last_timetable_teacher = teacher_in_query
                if section_in_query:
                    last_timetable_section = section_in_query

                if ("class" in question.lower() or "classes" in question.lower()) and (teacher_in_query or last_timetable_teacher) and day_token:
                    teacher_target = teacher_in_query or last_timetable_teacher
                    bot_reply = fetch_teacher_classes(teacher_target, day_token, language_hint)
                else:
                    teacher_context = ""
                    if not teacher_in_query and last_timetable_teacher:
                        teacher_context = f"Teacher: {last_timetable_teacher}\n"
                    section_context = ""
                    if not section_in_query and last_timetable_section:
                        section_context = f"Section: {last_timetable_section}\n"

                    # Agent ko history aur naya sawal ek sath bhej rahe hain (existing logic)
                    full_prompt = (
                        f"{TIMETABLE_SYSTEM_CONTEXT}\n\n"
                        f"Language preference: {language_hint} (English or Roman Urdu in Latin letters only).\n\n"
                        f"{teacher_context}{section_context}"
                        f"--- CHAT HISTORY ---\n{chat_history}\n\n"
                        f"--- NEW QUESTION ---\nUser: {question}"
                    )
                    try:
                        response = agent_executor.invoke({"input": full_prompt})
                        bot_reply = response["output"]
                    except Exception:
                        if agent_executor_fallback is None:
                            agent_executor_fallback = build_sql_agent(llm_gpt)
                        response = agent_executor_fallback.invoke({"input": full_prompt})
                        bot_reply = response["output"]

            # ── FREE SLOT ──────────────────────────────────────────────────────
            elif query_type == QueryType.FREE_SLOT:
                if teacher_in_query:
                    last_free_slot_teacher = teacher_in_query
                    last_free_slot_section = ""
                if section_in_query:
                    last_free_slot_section = section_in_query
                    last_free_slot_teacher = ""
                pending_free_slot = not day_in_query
                bot_reply = handle_free_slot(question, language_hint)

            # ── KNOWLEDGE (RAG) ────────────────────────────────────────────────
            else:
                rag_query = question
                if PRONOUN_FOLLOWUP.search(question) and last_faculty_name:
                    rag_query = f"{question}\nFaculty: {last_faculty_name}"
                bot_reply = rag.ask(rag_query, chat_history, language_hint)

            if language_hint == "roman_urdu":
                bot_reply = ensure_roman_urdu(bot_reply)

            print(f"\n🤖 AskUni Bot: {bot_reply}")

            # 🔄 Jawab aane ke baad, usko memory mein save kar lo (existing logic)
            chat_history += f"User: {question}\nBot: {bot_reply}\n"

            # Memory ko limits mein rakhna (existing logic)
            if len(chat_history) > 2000:
                chat_history = chat_history[-2000:]

        except Exception as inner_e:
            print(f"\n❌ Error: {inner_e}")

except Exception as e:
    print(f"\n❌ BARI GHALTI: {e}")