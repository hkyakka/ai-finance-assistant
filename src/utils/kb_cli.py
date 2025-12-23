from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.core.config import SETTINGS
from src.utils.validators import validate_kb


def cmd_validate(args: argparse.Namespace) -> int:
    rep = validate_kb(
        manifest_path=args.manifest or SETTINGS.kb_manifest,
        docs_dir=args.docs_dir or SETTINGS.kb_docs_dir,
        glossary_path=args.glossary or SETTINGS.kb_glossary,
        min_docs=args.min_docs,
        min_terms=args.min_terms,
    )

    if args.json:
        print(rep.model_dump_json(indent=2))
    else:
        if rep.errors:
            print("❌ KB Validation FAILED")
            for e in rep.errors:
                print(f"ERROR: {e.message} ({e.location or ''})")
        else:
            print("✅ KB Validation OK (no errors)")

        for w in rep.warnings:
            print(f"WARN: {w.message} ({w.location or ''})")

    return 0 if rep.ok else 2


def main() -> None:
    p = argparse.ArgumentParser(prog="kb_cli", description="Knowledge base utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate KB completeness")
    v.add_argument("--manifest", default=None)
    v.add_argument("--docs_dir", default=None)
    v.add_argument("--glossary", default=None)
    v.add_argument("--min_docs", type=int, default=50)
    v.add_argument("--min_terms", type=int, default=50)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
