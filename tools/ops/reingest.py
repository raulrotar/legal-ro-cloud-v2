#!/usr/bin/env python3
"""Drop all chunks/gazettes and re-ingest every PDF under laws/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_processing.pipeline import process_gazette

settings = load_settings()
db = get_db(settings)

print("Dropping existing chunks, gazettes, acts, laws...")
db.chunks.drop()
db.gazettes.drop()
db.acts.drop()
db.laws.drop()

laws_dir = Path(__file__).parent.parent / "laws"
pdfs = sorted(laws_dir.rglob("*.pdf"))
print(f"Found {len(pdfs)} PDFs to ingest.")

for i, pdf in enumerate(pdfs, 1):
    print(f"[{i}/{len(pdfs)}] {pdf.name} ...", end=" ", flush=True)
    try:
        result = process_gazette(str(pdf), settings)
        if result.status == "skipped":
            print("skipped")
        elif result.status == "failed":
            print(f"FAILED: {result.warnings}")
        else:
            warn = f" [{'; '.join(result.warnings)}]" if result.warnings else ""
            print(f"acts={result.acts_segmented} chunks={result.chunks_created}{warn}")
    except Exception as e:
        print(f"ERROR: {e}")

total = db.chunks.count_documents({})
print(f"\nIngestion complete. Total chunks: {total}")

print("\nRebuilding indexes...")
from setup_indexes import ensure_indexes  # noqa: E402
ensure_indexes(db, settings)
