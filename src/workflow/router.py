
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm_init import llm

# Initialize the LLM
llm = llm()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["FINANCE_QA", "TAX", "NEWS", "MARKET", "GOAL", "PORTFOLIO"] = Field(
        ...,
        description="Given a user question, which datasource would be most relevant for answering their question.",
    )

# Create a prompt template for routing
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert at routing a user question to the appropriate data source."),
        ("human", "Given the user query, route it to the most relevant data source. Query: {query}"),
    ]
)

# Create a structured LLM chain for routing
structured_llm = llm.with_structured_output(RouteQuery)
router = prompt | structured_llm

def route_query(state):
    """
    Route a user query to the appropriate data source.
    """
    query = state.get("user_text")
    source = router.invoke({"query": query})
    return source.datasource
