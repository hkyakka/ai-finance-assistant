from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config import SETTINGS
from src.core.schemas import RagChunk, RagResult
from src.rag.embeddings import get_embedder
from src.utils.logging import get_logger

logger = get_logger("rag.retriever")


def _default_index_dir() -> str:
    return os.getenv("KB_INDEX_DIR", "data/kb/index")


@dataclass
class _ChunkMeta:
    doc_id: str
    chunk_id: str
    title: str
    url: str
    category: str
    sub_category: str
    text: str


def _load_meta(meta_path: Path) -> List[_ChunkMeta]:
    metas: List[_ChunkMeta] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            metas.append(_ChunkMeta(
                doc_id=m.get("doc_id",""),
                chunk_id=m.get("chunk_id",""),
                title=m.get("title",""),
                url=m.get("url",""),
                category=m.get("category","general"),
                sub_category=m.get("sub_category","basics"),
                text=m.get("text",""),
            ))
    return metas


def _snippet(text: str, max_chars: int = 280) -> str:
    s = " ".join(text.split())
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (1,D), b: (N,D) assumed normalized
    return (a @ b.T).astype(np.float32)


def mmr_select(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_indices: List[int],
    *,
    top_k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """
    Max Marginal Relevance selection.
    Assumes vectors are L2-normalized so dot product = cosine similarity.
    Returns indices into the FULL embedding matrix (not 0..len(cands)).
    """
    if len(cand_indices) <= top_k:
        return cand_indices

    # Similarity query->candidate
    q_sims = (query_vec @ cand_vecs.T).flatten()  # (C,)
    selected: List[int] = []
    selected_c: List[int] = []

    # pick best first
    first = int(np.argmax(q_sims))
    selected.append(cand_indices[first])
    selected_c.append(first)

    while len(selected) < top_k:
        best_score = -1e9
        best_c = None

        for c in range(len(cand_indices)):
            if c in selected_c:
                continue
            sim_to_query = float(q_sims[c])

            # max similarity to already selected
            if selected_c:
                sel_vecs = cand_vecs[selected_c]  # (S,D)
                sim_to_sel = float(np.max(sel_vecs @ cand_vecs[c:c+1].T))
            else:
                sim_to_sel = 0.0

            mmr = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_sel
            if mmr > best_score:
                best_score = mmr
                best_c = c

        if best_c is None:
            break
        selected.append(cand_indices[best_c])
        selected_c.append(best_c)

    return selected


class Retriever:
    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.index_dir = Path(index_dir or _default_index_dir())
        self._embeddings: Optional[np.ndarray] = None
        self._metas: Optional[List[_ChunkMeta]] = None
        self._faiss = None
        self._faiss_index = None
        self._embedder = get_embedder()

    def load(self) -> None:
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "meta.jsonl"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index not found in {self.index_dir}. Run: python -m src.rag.ingest")

        self._embeddings = np.load(emb_path).astype(np.float32)
        self._metas = _load_meta(meta_path)

        # optional faiss
        faiss_path = self.index_dir / "faiss.index"
        try:
            import faiss  # type: ignore
            if faiss_path.exists():
                self._faiss = faiss
                self._faiss_index = faiss.read_index(str(faiss_path))
        except Exception:
            self._faiss = None
            self._faiss_index = None

        if len(self._metas) != self._embeddings.shape[0]:
            raise RuntimeError("Index meta and embeddings length mismatch.")

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        min_score: float = 0.0,
        category: Optional[str] = None,
    ) -> RagResult:
        if self._embeddings is None or self._metas is None:
            self.load()

        assert self._embeddings is not None
        assert self._metas is not None

        q = query.strip()
        if not q:
            return RagResult(query="", chunks=[])

        q_vec = self._embedder.embed_query(q).astype(np.float32)
        # Ensure normalized (some embedders already normalize)
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
        q_vec = (q_vec / q_norm).astype(np.float32)

        # Candidate selection
        emb = self._embeddings
        metas = self._metas

        # apply category filter by masking indices
        valid_idx = list(range(len(metas)))
        if category:
            cat = category.strip().lower()
            valid_idx = [i for i, m in enumerate(metas) if (m.category or "").strip().lower() == cat]

        if not valid_idx:
            return RagResult(query=q, chunks=[])

        cand_vecs = emb[valid_idx]

        # Use FAISS if available and no category filter (or build temp subset index)
        if self._faiss_index is not None and len(valid_idx) == emb.shape[0]:
            D, I = self._faiss_index.search(q_vec, k=max(top_k * 5, top_k))
            sims = D[0].tolist()
            idxs = I[0].tolist()
            # filter invalid (-1)
            ranked = [(idxs[i], float(sims[i])) for i in range(len(idxs)) if idxs[i] >= 0]
        else:
            sims = (q_vec @ cand_vecs.T).flatten()  # (C,)
            ranked_local = np.argsort(-sims)[: max(top_k * 5, top_k)]
            ranked = [(valid_idx[int(i)], float(sims[int(i)])) for i in ranked_local]

        # Score filtering
        ranked = [(i, s) for (i, s) in ranked if s >= float(min_score)]

        if not ranked:
            return RagResult(query=q, chunks=[])

        # MMR selection over top candidates
        cand_n = min(len(ranked), max(top_k * 5, top_k))
        cand_indices = [ranked[i][0] for i in range(cand_n)]
        cand_scores = {ranked[i][0]: ranked[i][1] for i in range(cand_n)}

        if use_mmr and len(cand_indices) > top_k:
            cand_vecs2 = emb[cand_indices]
            selected = mmr_select(q_vec, cand_vecs2, cand_indices, top_k=top_k, lambda_param=mmr_lambda)
        else:
            selected = cand_indices[:top_k]

        chunks: List[RagChunk] = []
        for idx in selected:
            m = metas[idx]
            score = float(cand_scores.get(idx, 0.0))
            chunks.append(RagChunk(
                doc_id=m.doc_id,
                chunk_id=m.chunk_id,
                title=m.title,
                url=m.url,
                snippet=_snippet(m.text),
                score=score,
            ))

        # Sort by score descending for stable output
        chunks.sort(key=lambda c: c.score, reverse=True)
        return RagResult(query=q, chunks=chunks)
           