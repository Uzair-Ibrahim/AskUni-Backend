import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env se API key load karega
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Google API ko configure karega
genai.configure(api_key=api_key)

print("🔍 Google API se available models ki list nikaal raha hoon...\n")

# List fetch karega
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available Model Name: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")