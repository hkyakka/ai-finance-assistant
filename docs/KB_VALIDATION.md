
# KB Validation

Run validation:

```bash
python -m src.utils.kb_cli validate
# or
python scripts/validate_kb.py validate
```

What it checks:
- `data/kb/knowledge_base_manifest.csv` exists and has required columns
- Every `local_path` exists
- Each markdown note has: `# Title`, `## Key ideas`, `## Simple example`, `## Source` with at least one URL
- `data/kb/glossary.csv` exists and has required columns and recommended minimum size
