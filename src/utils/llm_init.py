from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.core.config import SETTINGS

load_dotenv()

def llm() -> ChatGoogleGenerativeAI:
    """Initializes and returns a ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(model=SETTINGS.llm_model, temperature=SETTINGS.llm_temperature)
