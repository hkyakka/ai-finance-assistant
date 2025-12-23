from __future__ import annotations

import os
from pathlib import Path

from src.rag.ingest import build_index
from src.rag.retriever import Retriever


def test_rag_build_and_retrieve(tmp_path: Path, monkeypatch):
    # Use deterministic hash embedder in tests (no heavy model downloads)
    monkeypatch.setenv("RAG_EMBEDDER", "hash")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create two tiny docs
    (docs_dir / "kb-0001.md").write_text(
        "# What are stocks?\n\nStocks represent ownership in a company.\n\n## Key ideas\n- ownership\n\n## Simple example\n...\n\n## Source\nhttps://example.com/stocks\n",
        encoding="utf-8",
    )
    (docs_dir / "kb-0002.md").write_text(
        "# What are bonds?\n\nBonds are loans to issuers.\n\n## Key ideas\n- fixed income\n\n## Simple example\n...\n\n## Source\nhttps://example.com/bonds\n",
        encoding="utf-8",
    )

    # Create manifest
    manifest = tmp_path / "knowledge_base_manifest.csv"
    manifest.write_text(
        "doc_id,title,category,sub_category,source_name,source_url,language,license_or_usage_notes,created_at,updated_at,local_path,summary,tags\n"
        f"kb-0001,What are stocks?,stocks,basics,test,https://example.com/stocks,en,,2025-01-01,2025-01-01,{docs_dir/'kb-0001.md'},,\n"
        f"kb-0002,What are bonds?,bonds,basics,test,https://example.com/bonds,en,,2025-01-01,2025-01-01,{docs_dir/'kb-0002.md'},,\n",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    build_index(manifest_path=str(manifest), docs_dir=str(docs_dir), index_dir=str(index_dir), force=True)

    r = Retriever(index_dir=str(index_dir))
    out = r.retrieve("Explain stock ownership", top_k=2, use_mmr=True)

    assert out.chunks, "Expected at least 1 retrieved chunk"
    # Ensure citations fields are present
    c = out.chunks[0]
    assert c.doc_id
    assert c.title
    assert c.url.startswith("http")
    assert c.snippet
