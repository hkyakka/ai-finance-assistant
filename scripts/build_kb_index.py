from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ingest import build_index

if __name__ == "__main__":
    # Uses config.yaml paths by default
    build_index(force="--force" in sys.argv)
    print("✅ KB index built. Files saved to data/kb/index (or KB_INDEX_DIR env var).")
