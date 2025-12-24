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

            lines = [NEWS_DISCLAIMER, "", f"## Top news for: **{query}**"]
            citations: List[Dict[str, Any]] = []
            for i, a in enumerate(arts, start=1):
                title = a.get("title") or "Untitled"
                link = a.get("url") or ""
                source = (a.get("source") or {}).get("name") or ""
                desc = a.get("description") or ""
                lines.append(f"{i}. [{title}]({link}) — *{source}*")
                if desc:
                    lines.append(f"   - {desc[:200]}{'…' if len(desc) > 200 else ''}")

                citations.append(
                    {
                        "doc_id": _news_doc_id(link, title),
                        "title": title,
                        "url": link,
                        "snippet": desc,
                        "score": None,
                    }
                )

            confidence = "medium" if arts else "low"

            return AgentResponse(
                agent_name=self.name,
                answer_md="\n".join(lines),
                data={"articles": arts},
                citations=citations,
                warnings=[],
                confidence=confidence,
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
