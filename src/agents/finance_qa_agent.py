"""Finance QA agent (education / concepts) grounded via RAG.

Note: Keep math minimal; do not compute numerically here. Use Quant tools for computations.
"""

from __future__ import annotations

from typing import List

from src.agents.base_agent import BaseAgent
from src.core.llm_client import LLMClient
from src.core.schemas import AgentRequest, AgentResponse, RagResult, RagChunk
from src.rag.retriever import Retriever


def _build_prompt(user_text: str, rag: RagResult) -> str:
    ctx_lines: List[str] = []
    for i, c in enumerate(rag.chunks or [], start=1):
        ctx_lines.append(
            f"[{i}] {c.title}\nURL: {c.url}\nSnippet: {c.snippet}\n"
        )
    context = "\n".join(ctx_lines) if ctx_lines else "(no retrieved context)"

    return (
        "You are a finance education assistant.\n"
        "Answer the user using ONLY the provided context.\n"
        "If the context is insufficient, say so and ask a brief follow-up.\n\n"
        f"USER QUESTION: {user_text}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write a short, clear answer in markdown.\n"
    )


class FinanceQAAgent(BaseAgent):
    name = "FinanceQAAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        # Retrieve context (patched in tests)
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

        # Build citations from RagChunk fields. We intentionally pass dicts to avoid
        # importing a citation class name that might vary across schema versions.
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

        answer_md = (llm_resp.text or "").strip()
        if not answer_md:
            answer_md = "I couldn't generate an answer from the provided context."

        warnings = []
        if not citations:
            warnings.append("NO_CITATIONS")

        return AgentResponse(
            agent_name=self.name,
            answer_md=answer_md,
            citations=citations,
            warnings=warnings,
            confidence="medium" if citations else "low",
        )
