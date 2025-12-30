from __future__ import annotations

from typing import List

def chunk_text(text: str, *, chunk_size_chars: int = 1400, overlap_chars: int = 200) -> List[str]:
    """Backward-compatible helper. Prefer using build_index/_split_text."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=overlap_chars)
        return splitter.split_text(text or "")
    except Exception:
        t = (text or "").strip()
        if not t:
            return []
        chunks=[]
        step=max(1, chunk_size_chars-overlap_chars)
        i=0
        while i < len(t):
            chunks.append(t[i:i+chunk_size_chars])
            i += step
        return chunks
