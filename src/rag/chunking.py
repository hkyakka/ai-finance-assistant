from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str


def _clean_markdown(md: str) -> str:
    # Remove code fences (keep code, but strip backticks)
    md = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"```[a-zA-Z0-9_-]*\n|\n```", "", m.group(0)), md)
    # Collapse excessive newlines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_size_chars: int = 1400,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Simple, robust chunking that works even without tokenizers.
    - Uses character windows with overlap
    - Respects paragraph boundaries when possible
    """
    text = _clean_markdown(text)
    if not text:
        return []

    # Prefer splitting by paragraphs first
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for p in paras:
        # If paragraph is huge, fall back to hard slicing
        if len(p) > chunk_size_chars:
            flush()
            for i in range(0, len(p), chunk_size_chars - overlap_chars):
                chunks.append(p[i:i + chunk_size_chars].strip())
            continue

        if len(cur) + len(p) + 2 <= chunk_size_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            flush()
            cur = p

    flush()

    out: List[Chunk] = []
    for i, c in enumerate(chunks):
        out.append(Chunk(chunk_id=f"{doc_id}::c{i:03d}", text=c))
    return out
