from src.core.llm_client import LLMClient

llm = LLMClient()
out = llm.generate("Explain diversification in 2 lines.")
print(out.text)
