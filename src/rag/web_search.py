from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger("rag.web_search")


@dataclass
class WebSearchHit:
    title: str
    url: str
    snippet: str = ""
    score: Optional[float] = None
    provider: Optional[str] = None


class WebSearchClient:
    """
    Optional web search helper for agents.

    Supported providers (set via env):
      - WEB_SEARCH_PROVIDER=tavily   (requires TAVILY_API_KEY)
      - WEB_SEARCH_PROVIDER=serper   (requires SERPER_API_KEY)
      - WEB_SEARCH_PROVIDER=none     (disable)

    If WEB_SEARCH_PROVIDER is not set, it will auto-detect:
      - tavily if TAVILY_API_KEY is present
      - serper if SERPER_API_KEY is present
      - otherwise disabled
    """

    def __init__(self, timeout_seconds: int = 12) -> None:
        self.timeout_seconds = timeout_seconds
        self.provider = (os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower()
        if not self.provider:
            if (os.getenv("TAVILY_API_KEY") or "").strip():
                self.provider = "tavily"
            elif (os.getenv("SERPER_API_KEY") or "").strip():
                self.provider = "serper"
            else:
                self.provider = "none"

    def search(self, query: str, max_results: int = 5) -> List[WebSearchHit]:
        q = (query or "").strip()
        if not q or self.provider in ("none", "off", "disabled", "false", "0"):
            return []

        try:
            if self.provider == "tavily":
                return self._search_tavily(q, max_results=max_results)
            if self.provider == "serper":
                return self._search_serper(q, max_results=max_results)

            logger.warning("Unknown WEB_SEARCH_PROVIDER=%s. Web search disabled.", self.provider)
            return []
        except Exception as e:
            logger.warning("Web search failed provider=%s err=%s", self.provider, e)
            return []

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)

    def _search_tavily(self, query: str, max_results: int = 5) -> List[WebSearchHit]:
        api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
        if not api_key:
            return []

        # Tavily API: https://docs.tavily.com/
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": int(max_results),
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        out = self._post_json("https://api.tavily.com/search", payload)
        results = out.get("results") or []
        hits: List[WebSearchHit] = []
        for r in results[:max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or r.get("snippet") or "").strip()
            score = r.get("score")
            if title and url:
                hits.append(WebSearchHit(title=title, url=url, snippet=snippet, score=score, provider="tavily"))
        return hits

    def _search_serper(self, query: str, max_results: int = 5) -> List[WebSearchHit]:
        api_key = (os.getenv("SERPER_API_KEY") or "").strip()
        if not api_key:
            return []

        # Serper API: https://serper.dev/
        payload = {"q": query, "num": int(max_results)}
        headers = {"X-API-KEY": api_key}
        out = self._post_json("https://google.serper.dev/search", payload, headers=headers)

        organic = out.get("organic") or []
        hits: List[WebSearchHit] = []
        for r in organic[:max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if title and url:
                hits.append(WebSearchHit(title=title, url=url, snippet=snippet, score=None, provider="serper"))
        return hits
