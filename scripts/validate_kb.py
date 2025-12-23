from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path even when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.kb_cli import main

if __name__ == "__main__":
    main()
