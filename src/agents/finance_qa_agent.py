"""Finance QA agent (education / concepts) grounded via RAG.

Behavior:
- If KB context is available: answer primarily using it and cite inline like [1], [2].
- If KB context is NOT available: still answer using general finance knowledge and clearly label as
  "Not sourced from KB:" so the UI can show missing citations without blocking the response.

Note: Keep math minimal; do not compute numerically here. Use Quant tools for computations.
"""

from __future__ import annotations

from typing import List

from src.agents.base_agent import BaseAgent
from src.core.llm_client import LLMClient
from src.core.schemas import AgentRequest, AgentResponse, RagResult
from src.rag.retriever import Retriever


def _build_prompt(user_text: str, rag: RagResult) -> str:
    chunks = rag.chunks or []

    # Build a compact context block with numbered citations.
    ctx_lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        ctx_lines.append(f"[{i}] {c.title}\nURL: {c.url}\nSnippet: {c.snippet}\n")
    context = "\n".join(ctx_lines)

    if chunks:
        return (
            "You are a finance education assistant.\n"
            "Use the provided context as the primary source.\n"
            "When you use information from the context, cite it inline like [1], [2].\n"
            "If the question is not covered by the context, you MAY answer using general knowledge, "
            "but you MUST clearly label that portion as 'Not sourced from KB'.\n"
            "Keep the answer short and practical.\n\n"
            f"USER QUESTION: {user_text}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Write a short, clear answer in markdown.\n"
        )

    # No retrieved context: still answer, but disclose lack of KB grounding.
    return (
        "You are a finance education assistant.\n"
        "No KB context was retrieved for this question.\n"
        "Answer using general finance knowledge.\n"
        "Start your answer with exactly: 'Not sourced from KB:'\n"
        "Keep it short, clear, and in markdown.\n"
        "Ask at most one follow-up question only if it materially helps.\n\n"
        f"USER QUESTION: {user_text}\n"
    )


class FinanceQAAgent(BaseAgent):
    name = "FinanceQAAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        # RAG can be precomputed by the LangGraph tool node.
        rag = req.rag_result
        if rag is None:
            retriever = Retriever()
            rag = retriever.retrieve(
                query=req.user_text,
                top_k=5,
                use_mmr=True,
                mmr_lambda=0.5,
                min_score=0.05,
            )

        prompt = _build_prompt(req.user_text, rag)

        llm = LLMClient()
        llm_resp = llm.generate(prompt)

        # Build citations from RagChunk fields.
        citations = []
        for c in rag.chunks or []:
            citations.append(
                {
                    "doc_id": c.doc_id,
                    "chunk_id": getattr(c, "chunk_id", None),
                    "title": c.title,
                    "url": c.url,
                    "snippet": c.snippet,
                    "score": getattr(c, "score", None),
                }
            )

        answer_md = (getattr(llm_resp, "text", None) or "").strip()
        if not answer_md:
            # If LLM returned nothing, provide a graceful fallback.
            answer_md = (
                "Not sourced from KB: I couldn't generate an answer just now. "
                "Please try again or rephrase the question."
            )

        warnings = []
        if not citations:
            warnings.append("NO_KB_MATCH")

        return AgentResponse(
            agent_name=self.name,
            answer_md=answer_md,
            citations=citations,
            warnings=warnings,
            confidence="medium" if citations else "low",
        )
