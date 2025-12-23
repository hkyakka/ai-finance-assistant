from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.utils.kb_loader import load_manifest, load_glossary
from src.utils.kb_models import ValidationReport


_URL_RE = re.compile(r"https?://\S+")


def validate_kb(manifest_path: str, docs_dir: str, glossary_path: str, *, min_docs: int = 50, min_terms: int = 50) -> ValidationReport:
    report = ValidationReport(ok=True)

    # Manifest
    try:
        rows = load_manifest(manifest_path)
    except Exception as e:
        report.add_error(str(e), location=manifest_path)
        return report.finalize()

    if len(rows) < min_docs:
        report.add_warning(f"Manifest has {len(rows)} rows; expected at least {min_docs}.", location=manifest_path)

    # Doc existence + basic structure checks
    docs_base = Path(docs_dir)
    if not docs_base.exists():
        report.add_error("Docs directory not found.", location=str(docs_base))
        return report.finalize()

    seen_ids = set()
    for r in rows:
        if r.doc_id in seen_ids:
            report.add_error(f"Duplicate doc_id in manifest: {r.doc_id}", location=manifest_path)
        seen_ids.add(r.doc_id)

        p = Path(r.local_path)
        if not p.is_absolute():
            # Resolve relative to repo
            p2 = Path(".") / p
        else:
            p2 = p

        # If file doesn't exist, try under docs_dir by filename
        if not p2.exists():
            p2 = docs_base / Path(r.local_path).name

        if not p2.exists():
            report.add_error(f"Missing doc file for doc_id={r.doc_id}: {r.local_path}", location=str(p2))
            continue

        txt = p2.read_text(encoding="utf-8", errors="ignore")
        if "## Key ideas" not in txt:
            report.add_warning(f"Doc missing section '## Key ideas' ({r.doc_id})", location=str(p2))
        if "## Simple example" not in txt:
            report.add_warning(f"Doc missing section '## Simple example' ({r.doc_id})", location=str(p2))
        if "## Source" not in txt:
            report.add_warning(f"Doc missing section '## Source' ({r.doc_id})", location=str(p2))
        if not _URL_RE.search(txt):
            report.add_warning(f"Doc has no URL in Source section ({r.doc_id})", location=str(p2))

    # Glossary
    try:
        terms = load_glossary(glossary_path)
        if len(terms) < min_terms:
            report.add_warning(f"Glossary has {len(terms)} terms; expected at least {min_terms}.", location=glossary_path)
    except Exception as e:
        report.add_warning(f"Glossary check skipped/failed: {e}", location=glossary_path)

    return report.finalize()
