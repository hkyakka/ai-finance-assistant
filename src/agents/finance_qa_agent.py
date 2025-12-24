from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.rag.retriever import Retriever
from src.utils.answer_format import format_citations_md


def _chunk_to_dict(ch: Any) -> Dict[str, Any]:
    """
    Retriever returns RagChunk (Pydantic) objects in most cases.
    Older code assumed dict-like chunks. This helper normalizes either form.
    """
    if ch is None:
        return {}
    # Pydantic v2
    if hasattr(ch, "model_dump"):
        try:
            return ch.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(ch, "dict"):
        try:
            return ch.dict()
        except Exception:
            pass
    if isinstance(ch, dict):
        return ch
    # Best-effort fallback
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
        # Ensure required fields exist for Citation schema
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


class FinanceQAAgent(BaseAgent):
    name = "finance_qa_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            query = (req.user_text or "").strip()
            if not query:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Please ask a finance question (e.g. *What are ETFs?*).",
                    data={},
                    citations=[],
                    warnings=["EMPTY_QUERY"],
                    confidence="low",
                )

            retriever = Retriever()
            rag = retriever.retrieve(query=query)

            chunks = rag.get("chunks", []) if isinstance(rag, dict) else getattr(rag, "chunks", [])
            citations = _chunks_to_citations(chunks)

            answer_body = ""
            llm_used = False
            try:
                from src.core.llm_client import LLMClient

                llm = LLMClient()

                # Ground the LLM: only snippets from citations
                context_lines = []
                for c in citations[:5]:
                    ctx = (c.get("snippet") or "").strip()
                    if ctx:
                        context_lines.append(f"- [{c['title']}]({c['url']}): {ctx}")
                context = "\n".join(context_lines).strip()

                prompt = (
                    "You are a finance tutor.\n"
                    "Rules:\n"
                    "1) Use ONLY the provided context.\n"
                    "2) If context is insufficient, say what is missing and ask 1 clarifying question.\n"
                    "3) Do not do numeric calculations beyond simple, single-step examples.\n\n"
                    f"User question: {query}\n\n"
                    f"Context:\n{context if context else '(no context found)'}\n\n"
                    "Return markdown, 6-12 lines, simple language."
                )
                answer_body = llm.generate(prompt).text
                llm_used = True
            except Exception:
                # Deterministic fallback
                if citations:
                    answer_body = (
                        f"I found relevant notes for: **{query}**.\n\n"
                        "I can explain it using the sources below. "
                        "Tell me your goal (learning / investing / exam) and your country (India/US/etc.) so I can tailor it."
                    )
                else:
                    answer_body = (
                        "I couldn't find relevant notes in the knowledge base for that query. "
                        "Try rephrasing with simpler keywords (e.g., 'ETF meaning', 'diversification')."
                    )

            answer_md = answer_body + format_citations_md(citations)

            warnings: List[str] = []
            if not citations:
                warnings.append("NO_RAG_SOURCES")
            if not llm_used:
                warnings.append("LLM_NOT_USED_FALLBACK")

            confidence = "high" if citations else "low"

            # Avoid returning raw RagChunk objects in data (can be non-serializable in some runtimes)
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
                answer_md="FinanceQAAgent failed.",
                data={},
                citations=[],
                warnings=["AGENT_FAILED"],
                confidence="low",
                error=ErrorEnvelope(code="AGENT_FAILED", message=str(e)).model_dump(),
            )
