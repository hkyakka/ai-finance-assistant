from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.utils.kb_models import KBManifestRow, GlossaryTerm

REQUIRED_MANIFEST_COLS = [
    "doc_id","title","category","sub_category","source_name","source_url",
    "language","license_or_usage_notes","created_at","updated_at",
    "local_path","summary","tags"
]

REQUIRED_GLOSSARY_COLS = ["term","definition","category","examples"]


def load_manifest(manifest_path: str) -> List[KBManifestRow]:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"KB manifest not found: {p}")

    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in REQUIRED_MANIFEST_COLS if c not in cols]
        if missing:
            raise ValueError(f"KB manifest missing columns: {missing}. Found: {cols}")

        rows: List[KBManifestRow] = []
        for r in reader:
            rows.append(KBManifestRow(**r))
        return rows


def load_glossary(glossary_path: str) -> List[GlossaryTerm]:
    p = Path(glossary_path)
    if not p.exists():
        raise FileNotFoundError(f"Glossary not found: {p}")

    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in REQUIRED_GLOSSARY_COLS if c not in cols]
        if missing:
            raise ValueError(f"Glossary missing columns: {missing}. Found: {cols}")

        out: List[GlossaryTerm] = []
        for r in reader:
            out.append(GlossaryTerm(**r))
        return out
