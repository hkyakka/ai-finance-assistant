"""Finance QA agent (education / concepts) grounded via RAG.

Note: Keep math minimal; do not compute numerically here. Use Quant tools for computations.
"""

from __future__ import annotations

import os
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest, AgentResponse
from src.utils.llm_init import llm

# Initialize the LLM
llm = llm()

def format_docs(docs: List[Dict]) -> str:
    """Format the retrieved documents for the prompt."""
    ctx_lines: List[str] = []
    for i, doc in enumerate(docs, start=1):
        ctx_lines.append(
            f"[{i}] {doc.metadata.get('title', '')}\n"
            f"URL: {doc.metadata.get('url', '')}\n"
            f"Snippet: {doc.page_content}\n"
        )
    return "\n".join(ctx_lines) if ctx_lines else "(no retrieved context)"


class FinanceQAAgent(BaseAgent):
    name = "FinanceQAAgent"

    def run(self, req: AgentRequest) -> AgentResponse:
        # Load the FAISS index
        embeddings = OpenAIEmbeddings()
        index_dir = os.getenv("KB_INDEX_DIR", "data/kb/index")
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()

        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a finance education assistant.\n"
                    "Answer the user using ONLY the provided context.\n"
                    "If the context is insufficient, say so and ask a brief follow-up.\n\n"
                    "CONTEXT:\n{context}\n\n",
                ),
                ("human", "{question}"),
            ]
        )

        # Create a chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(
                answer=(
                    RunnableLambda(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            )
        )

        # Invoke the chain
        result = rag_chain.invoke(req.user_text)
        answer_md = result["answer"]

        # Build citations from the retrieved documents
        citations = []
        for doc in result["context"]:
            citations.append(
                {
                    "doc_id": doc.metadata.get("doc_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "title": doc.metadata.get("title"),
                    "url": doc.metadata.get("url"),
                    "snippet": doc.page_content,
                    "score": doc.metadata.get("score"),
                }
            )

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
