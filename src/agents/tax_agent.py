from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.core.config import SETTINGS
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.rag.retriever import Retriever
from src.rag.web_search import WebSearchClient
from src.utils.answer_format import format_citations_md

DISCLAIMER = (
    "**Disclaimer:** I’m not a tax professional. This is general education only. "
    "Tax rules vary by country/state and can change; verify with official guidance or a qualified advisor."
)


def _chunk_to_dict(ch: Any) -> Dict[str, Any]:
    if ch is None:
        return {}
    if hasattr(ch, "model_dump"):
        try:
            return ch.model_dump()
        except Exception:
            pass
    if hasattr(ch, "dict"):
        try:
            return ch.dict()
        except Exception:
            pass
    if isinstance(ch, dict):
        return ch
    return {k: getattr(ch, k) for k in dir(ch) if not k.startswith("_")}


def _chunks_to_citations(chunks: List[Any]) -> List[Dict[str, Any]]:
    cites: List[Dict[str, Any]] = []
    for ch in chunks or []:
        d = _chunk_to_dict(ch)
        doc_id = d.get("doc_id") or d.get("id") or d.get("chunk_id") or "unknown"
        title = d.get("title") or "Untitled"
        url = d.get("source_url") or d.get("url") or ""
        snippet = d.get("snippet") or d.get("text") or ""
        score = d.get("score")
        cites.append(
            {
                "doc_id": str(doc_id),
                "title": str(title),
                "url": str(url),
                "snippet": str(snippet),
                "score": score,
                "provider": d.get("provider") or d.get("source") or "rag",
            }
        )
    return cites


def _web_hits_to_citations(hits: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        d = _chunk_to_dict(h)
        title = d.get("title") or "Web result"
        url = d.get("url") or ""
        snippet = d.get("snippet") or ""
        score = d.get("score")
        out.append(
            {
                "doc_id": url or title,
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": score,
                "provider": d.get("provider") or "web",
            }
        )
    return out


def _build_context(citations: List[Dict[str, Any]], max_items: int = 8) -> str:
    """
    Small, LLM-friendly context block.
    """
    lines: List[str] = []
    for i, c in enumerate(citations[:max_items], start=1):
        title = (c.get("title") or "").strip()
        url = (c.get("url") or "").strip()
        snippet = (c.get("snippet") or "").strip()
        provider = (c.get("provider") or "").strip()
        if not (title or snippet):
            continue
        header = f"[{i}] {title}"
        if provider:
            header += f" ({provider})"
        if url:
            header += f" - {url}"
        lines.append(header)
        if snippet:
            # keep short
            snippet = snippet.replace("\n", " ").strip()
            lines.append(f"    {snippet[:500]}{'…' if len(snippet) > 500 else ''}")
    return "\n".join(lines).strip()


def _build_prompt(user_text: str, context: str) -> str:
    """
    Prompt is deliberately structured so:
      - it answers even when context is empty,
      - it asks for missing info when required,
      - it avoids giving personalized filing advice.
    """
    return (
        "You are a tax education assistant.\n"
        "You can answer from general knowledge, and you should use the provided context when it helps.\n\n"
        "Safety / quality rules:\n"
        "1) Do NOT give personalized filing instructions or tell the user exactly what to put on a tax return.\n"
        "2) If the question depends on missing facts (country/state, tax year, filing status, residency, cost basis, "
        "holding period, ordinary-income bracket, whether it's profit vs withdrawal), ask concise follow-ups.\n"
        "3) If you estimate anything, label it clearly as an assumption and give a range or a formula.\n"
        "4) Prefer plain language. Use headings + bullets. Keep calculations simple.\n\n"
        "Output format (markdown):\n"
        "## Answer\n"
        "(1–3 sentences)\n\n"
        "## Details\n"
        "- bullets\n\n"
        "## Example\n"
        "- one short example\n\n"
        "## What to confirm\n"
        "- bullets with missing info + what to verify in official guidance\n\n"
        f"User question: {user_text.strip()}\n\n"
        f"Context (may be empty):\n{context if context else '(no retrieved context)'}\n"
    )


def _choose_confidence(citations: List[Dict[str, Any]], llm_used: bool) -> str:
    if citations and llm_used:
        return "high"
    if llm_used:
        return "medium"
    return "low"


class TaxAgent(BaseAgent):
    name = "TaxAgent"

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.web_search = WebSearchClient()

    def _retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Returns (citations, warnings). Retrieval is best-effort.
        """
        warnings: List[str] = []
        citations: List[Dict[str, Any]] = []

        # 1) Local RAG (if index exists)
        try:
            rag = self.retriever.retrieve(
                query,
                top_k=SETTINGS.rag_top_k,
                use_mmr=SETTINGS.rag_use_mmr,
                min_score=SETTINGS.rag_min_score,
            )
            citations.extend(_chunks_to_citations(getattr(rag, "chunks", []) or []))
        except FileNotFoundError:
            warnings.append("NO_LOCAL_RAG_INDEX")
        except Exception:
            warnings.append("LOCAL_RAG_FAILED")

        # 2) Optional web search (if configured)
        # Heuristic: if we didn't retrieve anything useful, or user asked a very general question.
        if len(citations) < 2:
            hits = self.web_search.search(query=query + " tax", max_results=5)
            web_cites = _web_hits_to_citations(hits)
            if web_cites:
                citations.extend(web_cites)
            else:
                warnings.append("NO_WEB_RESULTS_OR_DISABLED")

        return citations, warnings

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            query = (req.user_text or "").strip()
            if not query:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md=f"{DISCLAIMER}\n\nPlease enter a tax question.",
                    data={},
                    citations=[],
                    warnings=["EMPTY_QUERY"],
                    confidence="low",
                )

            citations, warnings = self._retrieve(query)
            context = _build_context(citations)

            answer_body = ""
            llm_used = False
            try:
                from src.core.llm_client import LLMClient

                llm = LLMClient()
                prompt = _build_prompt(query, context)
                answer_body = (llm.generate(prompt).text or "").strip()
                llm_used = True
            except Exception:
                warnings.append("LLM_UNAVAILABLE")
                if citations:
                    answer_body = (
                        f"I found some potentially relevant sources for: **{query}**.\n\n"
                        "If you tell me your **country/state**, **tax year**, and the **type of income/asset**, "
                        "I can summarize how the rules usually apply.\n"
                    )
                else:
                    answer_body = (
                        "I can't access an LLM right now, and no sources were retrieved. "
                        "Please configure GEMINI_API_KEY (or web search keys) and try again."
                    )

            answer_md = f"{DISCLAIMER}\n\n{answer_body}{format_citations_md(citations)}"
            confidence = _choose_confidence(citations, llm_used)

            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data={
                    "query": query,
                    "used_llm": llm_used,
                    "context_len": len(context),
                    "citations_count": len(citations),
                },
                citations=citations,
                warnings=warnings,
                confidence=confidence,
            )
        except Exception as e:
            return AgentResponse(
                agent_name=self.name,
                answer_md="TaxAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
