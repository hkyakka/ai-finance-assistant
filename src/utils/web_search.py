from __future__ import annotations

import os
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from src.core.config import SETTINGS
from src.utils.logging import get_logger

logger = get_logger("web_search")

def search_web_tavily(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Searches the web using Tavily API and returns the top_k results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not found in environment variables.")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=top_k)
        return [{"url": r.get("url"), "title": r.get("title")} for r in response["results"]]
    except Exception as e:
        logger.error(f"Error calling Tavily API: {e}")
        return []


def fetch_url_content(url: str) -> str:
    """Fetches and parses the text content of a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return " ".join(t.strip() for t in soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return ""

def summarize_with_gemini(content: str, query: str) -> str:
    """Summarizes the given content using the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return "Error: Gemini API key not configured."

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = (
            f"Please summarize the following content to answer the query: '{query}'.\n\n"
            f"Content:\n{content}\n\n"
            "Summary:"
        )

        response = client.models.generate_content(
            model=SETTINGS.llm_model,
            contents=prompt,
            config={"temperature": float(SETTINGS.llm_temperature)},
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"Error summarizing content: {e}"


def web_search_and_summarize(query: str, top_k: int = 5) -> str:
    """Performs a web search, fetches content, and summarizes it."""
    search_results = search_web_tavily(query, top_k=top_k)
    if not search_results:
        return "Could not perform web search."

    content = []
    for result in search_results:
        page_content = fetch_url_content(result["url"])
        if page_content:
            content.append(f"Title: {result['title']}\nContent: {page_content}")

    if not content:
        return "Could not fetch content from web search results."

    combined_content = "\n\n".join(content)
    summary = summarize_with_gemini(combined_content, query)
    return summary
