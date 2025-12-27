from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse, ErrorEnvelope
from src.utils.web_search import web_search_and_summarize

class TaxAgent(BaseAgent):
    name = "tax_agent"

    def run(self, req: AgentRequest) -> AgentResponse:
        try:
            query = (req.user_text or "").strip()
            if not query:
                return AgentResponse(
                    agent_name=self.name,
                    answer_md="Please ask a tax-related question.",
                    data={},
                    citations=[],
                    warnings=["EMPTY_QUERY"],
                    confidence="low",
                )

            from src.core.llm_client import LLMClient

            llm = LLMClient()

            # Use the web search function to get information
            search_summary = web_search_and_summarize(query)

            prompt = (
                "You are an expert tax advisor. Your task is to provide a clear, accurate, and personalized tax answer based on the user's question and the provided web search summary.\n\n"
                "Instructions:\n"
                "1. Analyze the user's question to understand their specific tax situation.\n"
                "2. Use the web search summary as your primary source of information to formulate the answer.\n"
                "3. If the user's question involves a calculation (e.g., 'what is the tax on a $10,000 withdrawal'), provide a step-by-step calculation.\n"
                "4. If the provided summary is insufficient to answer the question, state that you were unable to find enough information to provide a specific answer.\n"
                "5. Your final answer should be a comprehensive and easy-to-understand explanation.\n\n"
                f"User question: {query}\n\n"
                f"Web Search Summary:\n{search_summary}\n\n"
                "Your expert tax answer:"
            )

            answer_body = llm.generate(prompt).text

            return AgentResponse(
                agent_name=self.name,
                answer_md=answer_body,
                data={"web_summary": search_summary},
                citations=[],
                warnings=[],
                confidence="high",
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
