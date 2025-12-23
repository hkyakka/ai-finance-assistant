from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


class BaseEmbedder:
    name: str

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0:1]


@dataclass
class HashEmbedder(BaseEmbedder):
    """
    Deterministic, dependency-free embedder.
    Not semantic like transformer embeddings, but works for Stage-4 plumbing and tests.

    Uses hashing trick over tokens -> dense vector.
    """
    dim: int = 768
    name: str = "hash"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # Basic tokenization
            tokens = [tok for tok in _simple_tokens(t) if tok]
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
                mat[i, idx] += sign
        return _normalize_rows(mat)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Optional semantic embedder. Requires: sentence-transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence_transformers:{model_name}"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def _simple_tokens(text: str) -> List[str]:
    text = text.lower()
    # keep letters/numbers
    out = []
    cur = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def get_embedder() -> BaseEmbedder:
    """
    Chooses embedder based on env var:
      RAG_EMBEDDER=hash | sentence_transformers
      RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
    Falls back to HashEmbedder if sentence-transformers is not installed.
    """
    choice = (os.getenv("RAG_EMBEDDER") or "hash").strip().lower()
    if choice in ("st", "sentence", "sentence_transformers", "sbert"):
        model = os.getenv("RAG_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
        try:
            return SentenceTransformerEmbedder(model_name=model)
        except Exception:
            # fall back
            return HashEmbedder()
    return HashEmbedder()
