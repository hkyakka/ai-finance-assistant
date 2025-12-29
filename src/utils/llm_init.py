from __future__ import annotations

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def llm() -> ChatOpenAI:
    """Initializes and returns a ChatOpenAI instance."""
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
