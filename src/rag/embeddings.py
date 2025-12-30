from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

# langchain-core provides the Embeddings interface in v0.2+
try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover
    Embeddings = object  # type: ignore


@dataclass(frozen=True)
class HashEmbeddings(Embeddings):
    """Deterministic, offline embeddings.

    This is intentionally simple so tests run without downloading models or calling APIs.
    """

    dim: int = 384

    def _vec(self, text: str) -> List[float]:
        # Expand sha256 digest to dim floats in [0, 1].
        data = (text or "").encode("utf-8", errors="ignore")
        digest = hashlib.sha256(data).digest()
        out: List[float] = []
        i = 0
        while len(out) < self.dim:
            b = digest[i % len(digest)]
            out.append(float(b) / 255.0)
            i += 1
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)
