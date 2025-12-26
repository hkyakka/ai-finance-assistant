from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.rag.retriever import Retriever
from src.utils.answer_format import format_citations_md

DISCLAIMER = (
    "**Disclaimer:** I’m not a tax professional. This is general education only. "
    "Tax rules vary by country and can change; verify with an official source or a qualified advisor."
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
                "score": float(score) if score is not None else None,
            }
        )
    return cites


class TaxAgent(BaseAgent):
    name = "tax_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            query = (req.user_text or "").strip()
            if not query:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md=DISCLAIMER + "\n\nPlease ask a tax-related question (e.g. *STCG vs LTCG*).",
                    data={},
                    citations=[],
                    warnings=["EMPTY_QUERY"],
                    confidence="low",
                )

            retriever = Retriever()
            rag = retriever.retrieve(query=query)

            chunks = rag.get("chunks", []) if isinstance(rag, dict) else getattr(rag, "chunks", [])
            citations = _chunks_to_citations(chunks)

            # If retriever has low confidence, fall back to web search
            use_web_fallback = not citations or (citations[0].get("score") or 0.0) < 0.3
            if use_web_fallback:
                from src.utils.web_search import web_search_and_summarize

                summary = web_search_and_summarize(query)
                answer_body = (
                    "I couldn't find a confident answer in my knowledge base, but here's a summary from the web:\n\n"
                    + summary
                )
                answer_md = DISCLAIMER + "\n\n" + answer_body
                return AgentResponse(
                    agent_name=self.name,
                    answer_md=answer_md,
                    data={"web_summary": summary},
                    citations=[],
                    warnings=["WEB_SEARCH_FALLBACK"],
                    confidence="medium",
                )

            answer_body = ""
            llm_used = False
            try:
                from src.core.llm_client import LLMClient

                llm = LLMClient()
                context_lines = []
                for c in citations[:5]:
                    ctx = (c.get("snippet") or "").strip()
                    if ctx:
                        context_lines.append(f"- [{c['title']}]({c['url']}): {ctx}")
                context = "\n".join(context_lines).strip()

                prompt = (
                    "You are a tax education assistant.\n"
                    "Rules:\n"
                    "1) Use ONLY the provided context.\n"
                    "2) Avoid personalized filing advice.\n"
                    "3) If context is insufficient, ask for the jurisdiction (India/US/etc.) and what asset/type.\n\n"
                    f"User question: {query}\n\n"
                    f"Context:\n{context if context else '(no context found)'}\n\n"
                    "Return markdown with: definition, how it applies, 1 short example, and what to verify."
                )
                answer_body = llm.generate(prompt).text
                llm_used = True
            except Exception:
                if citations:
                    answer_body = (
                        f"General tax education related to: **{query}**.\n\n"
                        "- Tell me your country/jurisdiction (India/US/etc.).\n"
                        "- Review the sources below for definitions and examples."
                    )
                else:
                    answer_body = (
                        "I couldn't find tax notes for that query in the KB. "
                        "Try keywords like STCG, LTCG, capital gains, dividend tax."
                    )

            answer_md = DISCLAIMER + "\n\n" + answer_body + format_citations_md(citations)

            warnings: List[str] = []
            if not citations:
                warnings.append("NO_RAG_SOURCES")
            if not llm_used:
                warnings.append("LLM_NOT_USED_FALLBACK")

            confidence = "high" if citations else "low"

            data_chunks = []
            for ch in (chunks or [])[:5]:
                data_chunks.append(_chunk_to_dict(ch))

            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_md,
                data={"rag_chunks": data_chunks},
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
