from __future__ import annotations
from typing import List, Dict, Any

def format_citations_md(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = ["\n\n---\n### Sources"]
    for i, c in enumerate(citations, start=1):
        title = c.get("title") or c.get("doc_id") or f"Source {i}"
        url = c.get("url") or ""
        snippet = (c.get("snippet") or "").strip()
        score = c.get("score")
        score_txt = f" (score={score:.3f})" if isinstance(score, (float, int)) else ""
        if url:
            lines.append(f"{i}. [{title}]({url}){score_txt}")
        else:
            lines.append(f"{i}. {title}{score_txt}")
        if snippet:
            lines.append(f"   - {snippet[:220]}{'…' if len(snippet) > 220 else ''}")
    return "\n".join(lines)
