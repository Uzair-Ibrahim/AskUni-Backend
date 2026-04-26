import os
import asyncio
import shutil
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler

# Warning bypass
os.environ["USER_AGENT"] = "AskUniBot/1.0"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. MODELLING SETUP ---
# Temperature 0 zaroori hai taake AI "tukkay" na mare
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
llm_groq = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)
llm_gpt = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("GITHUB_TOKEN"), base_url="https://models.inference.ai.azure.com", temperature=0)

def custom_fallback_llm(prompt_value):
    # Gemini ko pehle rakha hai kyunke iski token limit bari hai, memory ke liye best hai
    try: return llm_gemini.invoke(prompt_value)
    except Exception:
        try: return llm_gpt.invoke(prompt_value)
        except Exception: return llm_groq.invoke(prompt_value)

# --- 2. DATA LOADING & VECTORSTORE ---
static_info_pages = [
    "https://nu.edu.pk/Admissions/Schedule",
    "https://nu.edu.pk/Admissions/FeeStructure",
    "https://nu.edu.pk/Admissions/Scholarships/",
    "https://nu.edu.pk/Admissions/TestPattern/"
]

dept_pages = [
    "https://nu.edu.pk/University/PhDFaculty/",
    "https://khi.nu.edu.pk/faculty-php/",
    "https://khi.nu.edu.pk/department-of-cyber-security/",
    "https://khi.nu.edu.pk/department-of-artificial-intelligence/",
    "https://khi.nu.edu.pk/department-of-software-engineering/",
    "https://khi.nu.edu.pk/department-of-electrical-engineering/",
    "https://khi.nu.edu.pk/department-of-management-sciences/",
    "https://khi.nu.edu.pk/department-of-sciences-humanities/"
]

async def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        print("🚀 Fresh data mojood hai! Bot start ho raha hai...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("🛰️ Discovery Mode Active: Website crawl ho rahi hai (Stealth Mode)...")
    
    # Browser ko real dikhane ke liye settings
    async with AsyncWebCrawler(verbose=True) as crawler:
        all_markdown_data = []
        target_urls = static_info_pages + dept_pages

        for url in target_urls:
            try:
                # Page load hone ka wait barha diya
                result = await crawler.arun(url=url, wait_for=5, bypass_cache=True)
                if result and result.markdown:
                    all_markdown_data.append(result.markdown)
                    
                    faculty_links = [l['href'] for l in result.links.get("internal", []) if "/personnel/" in l['href']]
                    
                    if faculty_links:
                        print(f"📡 {len(faculty_links)} profiles mili hain. Aram se utha raha hoon...")
                        
                        # BATCH SIZE 1 kar diya taake IP block na ho
                        for f_link in faculty_links:
                            full_f_url = f_link if f_link.startswith("http") else f"https://khi.nu.edu.pk{f_link}"
                            try:
                                # Har profile ke baad 3-5 second ka gap (Human-like)
                                await asyncio.sleep(3) 
                                f_result = await crawler.arun(url=full_f_url, wait_for=3)
                                
                                if f_result and f_result.markdown:
                                    all_markdown_data.append(f_result.markdown)
                                    print(f"✅ Saved: {full_f_url.split('/')[-2]}")
                            except Exception:
                                print(f"⚠️ Skip ho gaya (Server busy): {full_f_url}")
                                continue # Error aaye toh agle par jao, ruko mat
            except Exception as e:
                print(f"❌ Error in {url}: {e}")

        # --- CHUNKING LOGIC (Yehi hai jo aap dhoond rahe thay) ---
        full_text = "\n\n---\n\n".join([str(d) for d in all_markdown_data if d])
        
        # Isse data mix nahi hota
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, 
            chunk_overlap=200, 
            separators=["---", "\n\n", "\n"]
        )
        chunks = text_splitter.split_text(full_text)
        
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(index_path)
        print("🎉 Database successfully updated!")
        return vectorstore

# --- 3. MAIN RUN WITH MEMORY ---
async def main():
    vectorstore = await setup_vectorstore()
    if not vectorstore: return
    
    # Retriever with MMR for better variety
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 7, 'fetch_k': 30})

    # Memory Store
    chat_history = []

    # System Prompt for Contextualizing (Memory Logic)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    # Main QA Prompt
    qa_template = """You are the official FAST NUCES Karachi Expert Assistant.

    STRICT RULES:
    1. **Identity Integrity:** Kisi bhi person (Teacher/Staff) ka data dusre se mix na karein. Agar multiple log ek hi naam ke hon, toh context se unka department aur designation verify karein.
    2. **Context Only:** Sirf provided context se jawab dein. Agar information nahi hai, toh seedha kahein "Mujhe iska record nahi mila."
    3. **Zero Hallucination:** Apne paas se emails, roll numbers, ya policies ijaad na karein.
    4. **Direct Response:** Roman Urdu/English mix mein jawab dein. User ko kisi faculty ke naam (e.g. Sir Atif) se address na karein unless user khud wo teacher ho.
    5. **Memory:** Agar user "unka", "wo", "email address" jese words use kare, toh history dekh kar samjhein kiski baat ho rahi hai.

    Context: {context}
    Question: {question}
    Answer:"""
    
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    print("\n🎓 AskUni Ready! (Type 'exit' to stop, 'refresh' to update data)")
    
    while True:
        user_input = input("\n🗣️ Aap: ").strip()
        if user_input.lower() in ['exit', 'quit', 'stop']: break
        
        if user_input.lower() == 'refresh':
            if os.path.exists("faiss_index"): shutil.rmtree("faiss_index")
            print("♻️ Index deleted. Restart to re-crawl.")
            break

        if not user_input: continue

        # --- STEP A: Contextualize Question (Memory Power) ---
        search_query = user_input
        if chat_history:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
            context_q_input = f"{contextualize_q_system_prompt}\n\nHistory:\n{history_str}\n\nQuestion: {user_input}"
            # Rewriting the question to be standalone
            rewritten_q = custom_fallback_llm(context_q_input)
            search_query = rewritten_q.content if hasattr(rewritten_q, 'content') else str(rewritten_q)

        # --- STEP B: Retrieval ---
        docs = retriever.invoke(search_query)
        context_text = "\n\n".join([d.page_content for d in docs])

        # --- STEP C: Generate Answer ---
        final_input = qa_prompt.format(context=context_text, question=user_input)
        response = custom_fallback_llm(final_input)
        answer = response.content if hasattr(response, 'content') else str(response)

        print(f"🤖 Bot: {answer}")

        # --- STEP D: Update Memory ---
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "bot", "content": answer})
        if len(chat_history) > 6: chat_history = chat_history[-6:] # Keep it short

if __name__ == "__main__":
    asyncio.run(main())