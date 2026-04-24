import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

try:
    print("\n⏳ 1. Connection setup ho raha hai...")
    db = SQLDatabase.from_uri(DATABASE_URL)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=GITHUB_TOKEN,
        base_url="https://models.inference.ai.azure.com",
        temperature=0
    )

    # Agent waisa hi rahega (baghair memory module ke)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=False
    )
    
    system_context = """
    You are the official AskUni AI for FAST NUCES. 
    You have access to the chat history provided in the prompt, so you can understand follow-up questions like 'why', 'who', or 'more details'.

    1. Language: Detect user language (English or Roman Urdu) and reply in the same.
    2. Search: Always use ILIKE or regex (~*) for names/sections/campus.
    3. If no data is found: Explain that "The database currently has no records for this." 
    4. Follow-ups: If a user asks 'why' after no data was found, explain that you can only provide information present in the database.
    5. Handling Offensive Input: If the user says anything unrelated or offensive, politely say: "Maazrat, main sirf FAST University ke timetable se mutaliq sawalaat ke jawab de sakta hoon."
    """

    print("✅ AskUni Bot Ready with Custom Memory!")
    print("="*50)
    
    # 🧠 HUMARI CUSTOM MEMORY VARIABLE
    chat_history = ""

    while True:
        question = input("\n🗣️ Aap: ")
        
        exit_words = ['exit', 'quit', 'bye', 'khuda hafiz', 'allah hafiz', 'khatam']
        if any(word in question.lower() for word in exit_words):
            print("\n🤖 AskUni Bot: Allah Hafiz! Apna khayal rakhna.")
            break
            
        if not question.strip(): continue
            
        try:
            print("⏳ Bot soch raha hai...")
            
            # Agent ko history aur naya sawal ek sath bhej rahe hain
            full_prompt = f"{system_context}\n\n--- CHAT HISTORY ---\n{chat_history}\n\n--- NEW QUESTION ---\nUser: {question}"
            
            response = agent_executor.invoke({"input": full_prompt})
            bot_reply = response['output']
            
            print(f"\n🤖 AskUni Bot: {bot_reply}")
            
            # 🔄 Jawab aane ke baad, usko memory mein save kar lo
            chat_history += f"User: {question}\nBot: {bot_reply}\n"
            
            # Memory ko limits mein rakhna (Sirf aakhri thori baatein yaad rakhe taake limit hit na ho)
            if len(chat_history) > 2000:
                chat_history = chat_history[-2000:]

        except Exception as inner_e:
            print(f"\n❌ Error: {inner_e}")

except Exception as e:
    print(f"\n❌ BARI GHALTI: {e}")