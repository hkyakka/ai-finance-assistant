from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.core.config import SETTINGS
from src.utils.logging import get_logger
from src.rag.chunking import chunk_text
from src.rag.embeddings import get_embedder
from src.utils.kb_loader import load_manifest
from src.utils.kb_models import KBManifestRow

logger = get_logger("rag.ingest")


@dataclass
class IndexMeta:
    embedder_name: str
    dim: int
    created_at: str
    chunk_size_chars: int
    overlap_chars: int
    total_chunks: int


def _default_index_dir() -> str:
    return os.getenv("KB_INDEX_DIR", "data/kb/index")


def build_index(
    manifest_path: Optional[str] = None,
    docs_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    *,
    chunk_size_chars: int = 1400,
    overlap_chars: int = 200,
    force: bool = False,
) -> str:
    """
    Build a vector index from KB markdown notes and persist to disk.

    Output files in <index_dir>/:
      - embeddings.npy           (float32, NxD, L2-normalized)
      - meta.jsonl              (one JSON per chunk: doc_id, title, url, text, category, chunk_id...)
      - index_meta.json         (embedder info, dims, params)
      - faiss.index (optional)  (if faiss is available)
    """
    manifest_path = manifest_path or SETTINGS.kb_manifest
    docs_dir = docs_dir or SETTINGS.kb_docs_dir
    index_dir = index_dir or _default_index_dir()

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "meta.jsonl"
    index_meta_path = out_dir / "index_meta.json"
    faiss_path = out_dir / "faiss.index"

    if (not force) and emb_path.exists() and meta_path.exists() and index_meta_path.exists():
        logger.info(f"Index already exists at {out_dir}. Use --force to rebuild.")
        return str(out_dir)

    rows: List[KBManifestRow] = load_manifest(manifest_path)
    docs_base = Path(docs_dir)

    embedder = get_embedder()

    texts: List[str] = []
    metas: List[Dict] = []

    t0 = time.time()
    for r in rows:
        # Resolve doc file path robustly
        manifest_path_str = r.local_path.replace("\\", "/")
        candidate = Path(manifest_path_str)

        # 1) try as-is relative to repo root
        if not candidate.is_absolute():
            p = Path(".") / candidate
        else:
            p = candidate

        # 2) fallback to docs_dir/<filename>
        if not p.exists():
            p = docs_base / Path(manifest_path_str).name

        if not p.exists():
            logger.warning(f"Missing doc file for doc_id={r.doc_id}: {r.local_path}")
            continue

        md = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(md, doc_id=r.doc_id, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)

        for c in chunks:
            texts.append(c.text)
            metas.append({
                "doc_id": r.doc_id,
                "chunk_id": c.chunk_id,
                "title": r.title,
                "category": r.category,
                "sub_category": r.sub_category,
                "url": r.source_url,
                "local_path": r.local_path,
                "text": c.text,
            })

    if not texts:
        raise RuntimeError("No texts found to index. Check manifest paths and docs_dir.")

    logger.info(f"Embedding {len(texts)} chunks with embedder={embedder.name}")
    vecs = embedder.embed_texts(texts)
    vecs = np.asarray(vecs, dtype=np.float32)

    # Persist
    np.save(emb_path, vecs)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    meta = IndexMeta(
        embedder_name=embedder.name,
        dim=int(vecs.shape[1]),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
        total_chunks=len(metas),
    )
    index_meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    # Optional FAISS for speed
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(int(vecs.shape[1]))
        index.add(vecs)
        faiss.write_index(index, str(faiss_path))
        logger.info("Saved FAISS index.")
    except Exception as e:
        logger.warning(f"FAISS not available or failed to write index: {e}")

    logger.info(f"Index built in {time.time()-t0:.2f}s at {out_dir}")
    return str(out_dir)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build KB vector index")
    p.add_argument("--manifest", default=None)
    p.add_argument("--docs_dir", default=None)
    p.add_argument("--index_dir", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--chunk_size_chars", type=int, default=1400)
    p.add_argument("--overlap_chars", type=int, default=200)
    args = p.parse_args()

    build_index(
        manifest_path=args.manifest,
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        chunk_size_chars=args.chunk_size_chars,
        overlap_chars=args.overlap_chars,
        force=args.force,
    )


if __name__ == "__main__":
    main()
        