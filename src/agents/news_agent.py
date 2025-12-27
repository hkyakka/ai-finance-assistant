from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

import requests

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope

NEWS_DISCLAIMER = "News summaries are informational, not investment advice."


def _news_doc_id(url: str, title: str) -> str:
    base = (url or title or "").encode("utf-8", errors="ignore")
    h = hashlib.sha1(base).hexdigest()[:12]
    return f"news:{h}"


class NewsAgent(BaseAgent):
    name = "news_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            mp = getattr(req, "market_payload", None)
            query = None
            if isinstance(mp, dict):
                query = mp.get("query")
            if not query:
                query = (req.user_text or "").strip()

            if not query:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Tell me what news you want (e.g., 'Nifty 50 today', 'AAPL earnings').",
                    data={},
                    citations=[],
                    warnings=["EMPTY_QUERY"],
                    confidence="low",
                )

            api_key = os.getenv("NEWSAPI_KEY")
            if not api_key:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md=(
                        f"{NEWS_DISCLAIMER}\n\n"
                        "To enable news, set `NEWSAPI_KEY` in `.env`.\n"
                        f"Query: **{query}**"
                    ),
                    data={},
                    citations=[],
                    warnings=["NEWSAPI_KEY_MISSING"],
                    confidence="low",
                )

            url = "https://newsapi.org/v2/everything"
            r = requests.get(
                url,
                params={"q": query, "pageSize": 5, "sortBy": "publishedAt"},
                headers={"X-Api-Key": api_key},
                timeout=20,
            )
            r.raise_for_status()
            payload = r.json()
            arts = (payload.get("articles") or [])[:5]

            if not arts:
                from src.utils.web_search import web_search_and_summarize

                summary = web_search_and_summarize(query)
                answer_body = (
                    "I couldn't find any recent news articles for your query, but here's a summary from the web:\n\n"
                    + summary
                )
                answer_md = f"{NEWS_DISCLAIMER}\n\n{answer_body}"
                return AgentResponse(
                    agent_name=self.name,
                    answer_md=answer_md,
                    data={"web_summary": summary},
                    citations=[],
                    warnings=["WEB_SEARCH_FALLBACK"],
                    confidence="medium",
                )

            from src.utils.web_search import summarize_with_gemini

            content_to_summarize = []
            for art in arts:
                content_to_summarize.append(f"Title: {art.get('title')}\nContent: {art.get('content') or art.get('description')}")

            summary = summarize_with_gemini("\n\n".join(content_to_summarize), query)
            answer_md = f"{NEWS_DISCLAIMER}\n\n## Summary of news for: **{query}**\n\n{summary}"

            citations: List[Dict[str, Any]] = []
            for a in arts:
                citations.append(
                    {
                        "doc_id": _news_doc_id(a.get("url"), a.get("title")),
                        "title": a.get("title"),
                        "url": a.get("url"),
                        "snippet": a.get("description"),
                        "score": None,
                    }
                )

            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data={"articles": arts, "summary": summary},
                citations=citations,
                warnings=[],
                confidence="medium",
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="NewsAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
