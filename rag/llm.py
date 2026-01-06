from langchain_google_genai import ChatGoogleGenerativeAI
import os

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Make sure .env is loaded!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)
