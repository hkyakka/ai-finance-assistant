from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.retriever import Retriever

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print("Usage: python scripts/query_kb.py <your question>")
        raise SystemExit(2)

    r = Retriever()
    out = r.retrieve(q, top_k=5, use_mmr=True)
    print(f"Query: {out.query}")
    for c in out.chunks:
        print(f"- [{c.score:.3f}] {c.title} ({c.doc_id}) -> {c.url}")
        print(f"  {c.snippet}")
