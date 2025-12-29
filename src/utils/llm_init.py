from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def llm() -> ChatGoogleGenerativeAI:
    """Initializes and returns a ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
