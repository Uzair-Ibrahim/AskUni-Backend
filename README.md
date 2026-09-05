# AskUni-Backend

**Intelligent FAST University Timetable Management & AI Chatbot Backend**

A comprehensive backend system that manages university timetables, provides intelligent chatbot interactions, and retrieves FAST University information. Automatically syncs data from Google Sheets to PostgreSQL and exposes powerful REST APIs.

## ✨ Key Features

- 📊 **Auto-Sync Timetable**: Automatically updates timetable from Google Sheets to PostgreSQL every 5 minutes
- 🤖 **AI Chatbot**: Natural language chatbot for timetable queries (English & Roman Urdu)
- 📅 **Smart Search**: Filter by section, teacher, room, day, campus
- ⏰ **Current Class Detection**: Find which class is running right now
- 🔓 **Free Slots Analysis**: Calculate available time slots for sections/teachers
- 👨‍🏫 **Teacher Schedule**: View all classes for a specific teacher
- 🏫 **Campus Info**: Get overview of all sections, teachers, and days
- 🌐 **FAST Website Scraper**: Retrieve latest news and announcements
- 🔄 **Real-time Sync**: Manual sync endpoint with force refresh option
- 📝 **Full REST API**: Comprehensive endpoints with error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Google Sheets (for timetable source)
- Git

### Installation

1. **Clone and Setup**
```bash
cd AskUni-Backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
Create `.env` file:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/askuni_db

# API Keys
GITHUB_TOKEN=your_github_token_for_openai
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# Timetable Source
TIMETABLE_SOURCE_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit?usp=sharing

# Optional: Sync Configuration
TIMETABLE_SYNC_INTERVAL_SECONDS=300
```

4. **Run Server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: **http://localhost:8000**

Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📚 API Endpoints Overview

### Core Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API welcome message |
| GET | `/sync/status` | Check timetable sync status |
| POST | `/sync/now` | Force immediate sync |
| GET | `/search` | Search classes by criteria |
| GET | `/current-class` | Get running class now |
| GET | `/free-slots` | Get available time slots |
| GET | `/teacher-schedule` | View teacher's schedule |
| GET | `/section-schedule` | View section's schedule |
| GET | `/campus-info` | Get campus overview |
| GET | `/fast-news` | Scrape FAST news |

### Chatbot Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chatbot/chat` | Chat with AskUni bot |
| GET | `/chatbot/health` | Check bot status |
| DELETE | `/chatbot/session/{session_id}` | Clear chat history |
| GET | `/chatbot/sessions` | List active sessions |

---

## 💡 Usage Examples

### Search Classes by Section
```bash
curl "http://localhost:8000/search?section=CS-1A&day=Monday"
```

### Get Current Running Class
```bash
curl "http://localhost:8000/current-class?section=CS-1A"
```

### Find Free Slots
```bash
curl "http://localhost:8000/free-slots?day=Tuesday&section=CS-1A"
```

### Chat with Bot
```bash
curl -X POST "http://localhost:8000/chatbot/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the schedule for CS-1A?",
    "session_id": "user_123"
  }'
```

### Get Teacher Schedule
```bash
curl "http://localhost:8000/teacher-schedule?teacher_name=Dr.%20Ahmed"
```

### Check Sync Status
```bash
curl "http://localhost:8000/sync/status"
```

---

## 🔄 Auto-Sync Flow

```
Google Sheets
    ↓
(Hash Check - Changed?)
    ↓
Extract to CSV
    ↓
Parse & Clean Data
    ↓
Insert to PostgreSQL
    ↓
Update State File
    ↓
Done ✅
```

**Schedule**: Every 5 minutes (configurable via `TIMETABLE_SYNC_INTERVAL_SECONDS`)

---

## 📋 Database Schema

### university_timetable Table
```sql
- id (Primary Key)
- day (Monday-Saturday)
- time (HH:MM - HH:MM format)
- subject (Course name)
- section (e.g., CS-1A)
- teacher_name
- room_number (e.g., A-101)
- campus (Main or City)
- created_at (Timestamp)
- updated_at (Timestamp)
```

**Indexes**: 
- `idx_day_section` - Fast section queries by day
- `idx_campus_section` - Campus + section filtering
- `idx_day_time` - Time-based queries

---

## 🤖 Chatbot Features

### Capabilities
- ✅ Understands natural language (English & Roman Urdu)
- ✅ Maintains conversation context with session IDs
- ✅ Queries timetable data using SQL agent
- ✅ Handles follow-up questions
- ✅ Filters unrelated/offensive queries

### Example Queries
- "What is CS-1A's schedule on Monday?"
- "اہل کے لیے کون کون سی کلاسز ہیں؟" (Roman Urdu)
- "Who teaches Data Structures?"
- "Show me free slots for CS-1A"
- "Is there any class running right now?"

---

## 📊 Configuration

### Environment Variables

#### Required
- `DATABASE_URL` - PostgreSQL connection string
- `GITHUB_TOKEN` - For OpenAI API access

#### Optional
- `GEMINI_API_KEY` - Google Gemini API key
- `GROQ_API_KEY` - Groq API key
- `TIMETABLE_SOURCE_URL` - Google Sheets URL
- `TIMETABLE_SYNC_INTERVAL_SECONDS` - Sync frequency (default: 300)
- `TIMETABLE_SOURCE_PATH` - Local xlsx path
- `TIMETABLE_CSV_PATH` - Generated CSV path

---

## 🐛 Troubleshooting

### Issue: Database connection fails
**Solution**: Verify `DATABASE_URL` is correct and PostgreSQL is running
```bash
psql -U postgres -d askuni_db -c "SELECT COUNT(*) FROM university_timetable;"
```

### Issue: Google Sheets sync fails
**Solution**: 
1. Ensure Google Sheet is shared with "Viewer" access
2. Check `TIMETABLE_SOURCE_URL` is correct
3. Verify internet connectivity

### Issue: Chatbot returns "not ready"
**Solution**: 
1. Check API keys in `.env`
2. Verify `DATABASE_URL` is accessible
3. Check server logs for errors

---

## 📁 Project Structure

```
AskUni-Backend/
├── main.py                    # FastAPI app with core endpoints
├── chatbot.py                 # CLI chatbot (legacy)
├── chatbot_api.py             # REST API chatbot endpoints
├── timetable_sync.py          # Sync logic
├── import_excel.py            # Excel parsing
├── push_to_db.py              # Database operations
├── database/
│   ├── models.py              # SQLAlchemy models
│   └── sql_db.py              # DB connection
├── data/
│   ├── timetable.xlsx         # Source file
│   └── .timetable_sync_state.json
├── cleaned_timetable.csv      # Processed data
├── requirements.txt           # Dependencies
├── .env                       # Configuration (not in git)
├── API_DOCUMENTATION.md       # Full API docs
└── README.md                  # This file
```

---

## 🔒 Security Notes

**Current State** (Development):
- No authentication required
- Public API access

**Production TODO**:
- [ ] Implement JWT authentication
- [ ] Add rate limiting
- [ ] Use environment-based secrets
- [ ] Enable CORS with specific origins
- [ ] Add request logging
- [ ] Implement caching with Redis

---

## 🧪 Testing

### Manual Testing with curl
```bash
# Test API health
curl http://localhost:8000/

# Test search
curl "http://localhost:8000/search?section=CS-1A"

# Test chatbot
curl -X POST http://localhost:8000/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test"}'
```

### Using Swagger UI
Navigate to: http://localhost:8000/docs

---

## 📈 Performance Optimization

- **Indexes**: Database queries are optimized with multi-column indexes
- **Async Sync**: Background timetable sync doesn't block API
- **Session Management**: Chatbot maintains context per session
- **CSV Caching**: Extracted data cached before DB insertion

---

## 🚀 Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Notification system (email/SMS alerts)
- [ ] Mobile app authentication flow
- [ ] Advanced export (PDF/Excel timetables)
- [ ] Calendar integration (Google Calendar sync)
- [ ] Multi-language support expansion
- [ ] Analytics dashboard
- [ ] Student feedback system

---

## 📝 Git Commit Trailer

All commits include:
```
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## 📞 Support

**API Documentation**: See `API_DOCUMENTATION.md` for detailed endpoint specs

**Common Issues**: Check troubleshooting section above

**Questions**: Review code comments and docstrings in main.py

---

## 📄 License

This project is part of FAST University's official infrastructure.

---

**Last Updated**: April 27, 2024  
**API Version**: 2.0  
**Python Version**: 3.8+  
**Database**: PostgreSQL 12+