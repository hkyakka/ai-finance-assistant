from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.core.config import SETTINGS
from src.utils.logging import get_logger
from src.utils.kb_loader import load_manifest
from src.core.langchain_factory import get_embeddings

logger = get_logger("rag.ingest")


def _default_index_dir() -> str:
    return os.getenv("KB_INDEX_DIR", "data/kb/index")


def _split_text(text: str, *, chunk_size_chars: int, overlap_chars: int) -> List[str]:
    # Prefer LangChain splitter when available; fall back to a tiny local splitter.
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text or "")
    except Exception:
        t = (text or "").strip()
        if not t:
            return []
        chunks: List[str] = []
        i = 0
        step = max(1, chunk_size_chars - overlap_chars)
        while i < len(t):
            chunks.append(t[i : i + chunk_size_chars])
            i += step
        return chunks


def build_index(
    manifest_path: Optional[str] = None,
    docs_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    *,
    chunk_size_chars: int = 1400,
    overlap_chars: int = 200,
    force: bool = False,
) -> str:
    """Build a local vector index using LangChain's VectorStore.

    Persists under KB_INDEX_DIR (default data/kb/index).

    This function keeps the existing public API used by tests and the app.
    """
    manifest_path = manifest_path or SETTINGS.kb_manifest
    docs_dir = docs_dir or SETTINGS.kb_docs_dir
    index_dir = index_dir or _default_index_dir()

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # If already built and not forcing, keep it.
    if not force:
        # FAISS uses index.faiss + index.pkl by default; keep checks broad.
        if any((index_path / n).exists() for n in ("index.faiss", "index.pkl", "chroma.sqlite3", "SIMPLE_INDEX")):
            return str(index_path)

    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Knowledge base manifest is empty.")

    try:
        from langchain_core.documents import Document
    except Exception as e:
        raise ImportError("Missing dependency: langchain-core") from e

    documents: List[Document] = []

    docs_dir_path = Path(docs_dir)
    docs_dir_norm = docs_dir_path.as_posix().rstrip("/")

    def _resolve_doc_path(local_path: str) -> Path:
        """Resolve a manifest local_path robustly.

        The shipped manifest may store either:
          - a filename relative to docs_dir (e.g. 'kb-0001-...md')
          - a path already rooted at docs_dir (e.g. 'data/kb/docs/kb-0001-...md')
          - an absolute path.

        We normalize separators and avoid duplicating docs_dir.
        """
        lp = (local_path or "").strip().replace("\\", "/")
        lp = lp.lstrip("./")
        if not lp:
            return Path("")

        p0 = Path(lp)
        if p0.is_absolute():
            return p0

        # If manifest already includes docs_dir prefix, use it as-is.
        if docs_dir_norm and lp.startswith(docs_dir_norm + "/"):
            return Path(lp)

        # Otherwise treat it as relative to docs_dir.
        return docs_dir_path / lp
    for r in rows:
        local_path = (r.local_path or "").strip()
        if not local_path:
            continue
        p = _resolve_doc_path(local_path)
        if not p.exists():
            logger.warning("KB doc not found: %s", p)
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for j, chunk in enumerate(_split_text(text, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)):
            meta: Dict[str, Any] = {
                "doc_id": r.doc_id,
                "chunk_id": f"{r.doc_id}:{j}",
                "title": r.title,
                "category": r.category,
                "sub_category": r.sub_category,
                "url": r.source_url,
                "local_path": str(p),
            }
            documents.append(Document(page_content=chunk, metadata=meta))

    if not documents:
        raise RuntimeError("No documents found to index. Check manifest paths and docs_dir.")

    embeddings = get_embeddings()

    # Prefer FAISS; fall back to a simple persisted cosine index if deps aren't available.
    try:
        from langchain_community.vectorstores import FAISS  # type: ignore
        vs = FAISS.from_documents(documents, embeddings)
        vs.save_local(str(index_path))
        logger.info("Built FAISS index: %s (docs=%s)", index_path, len(documents))
        return str(index_path)
    except Exception as e:
        logger.warning("FAISS unavailable (%s). Falling back to simple persisted index.", e)

        # Simple persisted index: embeddings.npy + metas.json
        import json
        import numpy as np

        texts = [d.page_content for d in documents]
        metas = []
        for d in documents:
            meta = dict(d.metadata or {})
            text = d.page_content or ""
            meta["snippet"] = (text[:280] + "…") if len(text) > 280 else text
            metas.append(meta)

        vecs = embeddings.embed_documents(texts)
        arr = np.asarray(vecs, dtype=np.float32)
        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms

        import numpy as np
        np.save(index_path / "embeddings.npy", arr)
        (index_path / "metas.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
        (index_path / "SIMPLE_INDEX").write_text("1", encoding="utf-8")
        logger.info("Built simple index: %s (docs=%s)", index_path, len(documents))
        return str(index_path)

