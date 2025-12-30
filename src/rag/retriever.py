from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.schemas import RagChunk, RagResult
from src.core.langchain_factory import get_embeddings
from src.utils.logging import get_logger

logger = get_logger("rag.retriever")


def _default_index_dir() -> str:
    return os.getenv("KB_INDEX_DIR", "data/kb/index")


def _as_similarity(score: float) -> float:
    # Convert distance or similarity to [0,1].
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    if s > 1.2:
        return 1.0 / (1.0 + s)
    return min(1.0, max(0.0, s))


class Retriever:
    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.index_dir = Path(index_dir or _default_index_dir())
        self._embeddings = get_embeddings()

        # Backends (lazy)
        self._vs: Any = None  # LangChain vectorstore (FAISS)
        self._mat = None      # numpy matrix for SIMPLE_INDEX
        self._metas: List[Dict[str, Any]] = []

    def load(self) -> None:
        if self._vs is not None or self._mat is not None:
            return

        # 1) Try FAISS saved by LangChain
        try:
            from langchain_community.vectorstores import FAISS  # type: ignore

            self._vs = FAISS.load_local(
                str(self.index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return
        except Exception as e:
            logger.info("FAISS index not available in %s (%s).", self.index_dir, e)

        # 2) Try SIMPLE_INDEX (numpy cosine)
        try:
            import numpy as np

            if (self.index_dir / "SIMPLE_INDEX").exists() and (self.index_dir / "embeddings.npy").exists():
                self._mat = np.load(self.index_dir / "embeddings.npy")
                # metas are stored as a list of dicts
                metas_path = self.index_dir / "metas.json"
                if metas_path.exists():
                    self._metas = json.loads(metas_path.read_text(encoding="utf-8"))
                else:
                    self._metas = [{} for _ in range(int(self._mat.shape[0]))]
                return
        except Exception as e:
            logger.warning("Failed to load SIMPLE_INDEX from %s: %s", self.index_dir, e)

        raise FileNotFoundError(
            f"No usable vector index found in {self.index_dir}. "
            "Run scripts/build_kb_index.py (or set KB_INDEX_DIR)."
        )

    def retrieve(
        self,
        *,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        min_score: float = 0.2,
    ) -> RagResult:
        self.load()
        q = (query or "").strip()
        if not q:
            return RagResult(query=query, chunks=[])

        # FAISS backend
        if self._vs is not None:
            results: List[Tuple[Any, float]] = []
            try:
                if use_mmr and hasattr(self._vs, "max_marginal_relevance_search_with_score"):
                    results = self._vs.max_marginal_relevance_search_with_score(
                        q,
                        k=int(top_k),
                        fetch_k=max(20, int(top_k) * 4),
                        lambda_mult=float(mmr_lambda),
                    )
                elif hasattr(self._vs, "similarity_search_with_score"):
                    results = self._vs.similarity_search_with_score(q, k=int(top_k))
                else:
                    docs = self._vs.similarity_search(q, k=int(top_k))
                    results = [(d, 1.0) for d in docs]
            except Exception as e:
                logger.warning("Vector search failed: %s", e)
                results = []

            chunks: List[RagChunk] = []
            for doc, raw_score in results or []:
                meta: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
                text = getattr(doc, "page_content", "") or ""
                sim = _as_similarity(raw_score)
                if sim < float(min_score):
                    continue
                chunks.append(
                    RagChunk(
                        doc_id=str(meta.get("doc_id") or ""),
                        chunk_id=str(meta.get("chunk_id") or ""),
                        title=str(meta.get("title") or ""),
                        url=str(meta.get("url") or ""),
                        snippet=(text[:280] + "…") if len(text) > 280 else text,
                        score=sim,
                        metadata={
                            "category": meta.get("category"),
                            "sub_category": meta.get("sub_category"),
                            "local_path": meta.get("local_path"),
                        },
                    )
                )
            return RagResult(query=query, chunks=chunks)

        # SIMPLE_INDEX backend
        import numpy as np

        mat = self._mat
        if mat is None:
            return RagResult(query=query, chunks=[])

        qv = np.asarray(self._embeddings.embed_query(q), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = mat @ qv  # cosine sim because rows are normalized
        top_idx = np.argsort(-sims)[: int(top_k) * 3]  # extra for filtering/MMR

        # Optional crude MMR: greedily pick diverse chunks
        selected: List[int] = []
        if use_mmr:
            lam = float(mmr_lambda)
            cand = list(map(int, top_idx))
            while cand and len(selected) < int(top_k):
                if not selected:
                    best = cand.pop(0)
                    selected.append(best)
                    continue
                # score = lam*sim(q, d) - (1-lam)*max_sim(d, selected)
                best_i = None
                best_score = -1e9
                for i in cand:
                    rel = float(sims[i])
                    div = max(float(mat[i] @ mat[j]) for j in selected)
                    score = lam * rel - (1.0 - lam) * div
                    if score > best_score:
                        best_score = score
                        best_i = i
                if best_i is None:
                    break
                cand.remove(best_i)
                selected.append(best_i)
        else:
            selected = list(map(int, top_idx[: int(top_k)]))

        chunks: List[RagChunk] = []
        for i in selected:
            sim = float(sims[i])
            if sim < float(min_score):
                continue
            meta = self._metas[i] if i < len(self._metas) else {}
            text = str(meta.get("text") or "")  # not stored
            # For SIMPLE_INDEX we didn't persist page_content; keep snippet empty unless present.
            snippet = str(meta.get("snippet") or "")
            chunks.append(
                RagChunk(
                    doc_id=str(meta.get("doc_id") or ""),
                    chunk_id=str(meta.get("chunk_id") or ""),
                    title=str(meta.get("title") or ""),
                    url=str(meta.get("url") or ""),
                    snippet=snippet,
                    score=sim,
                    metadata={k: meta.get(k) for k in ("category", "sub_category", "local_path") if k in meta},
                )
            )

        return RagResult(query=query, chunks=chunks)
