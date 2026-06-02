"""Re-ingest (and optionally re-extract) specific MO issues in the STAGE DB.

Two modes per target:
  "reingest"   — delete old DB chunks, re-ingest from the existing extracted JSON.
                 Use when only the ingest-layer code changed (e.g. cotizatii table fix).
  "reextract"  — delete old DB chunks AND the cached JSON, re-run OCR+segmentation
                 from the PDF, then re-ingest.
                 Use when segment.py or gazette_extractor changed.

Usage:
    cd legal-Ro-poc-cloud-v2
    MONGODB_URI="mongodb+srv://..." uv run python scripts/reingest_targeted_mos.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "processing" / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db


TARGETS = [
    # ── MO_820_2007: already correctly split by previous run; just re-ingest
    #    (keeping this so the script remains idempotent if run again)
    {
        "mode": "reingest",
        "source_issue_id": "PI_820_2007",
        "json_path": "extracted/2007/12/03/MO_PI_820_2007-12-03.json",
        "label": "MO_PI_820_2007 (ANCEX Ord 346 + MIRA taxi Ord 356 + HG 1448)",
    },
    # ── MO_2_1989: segment.py fix (merge 38-char fragment) → must re-extract from PDF
    {
        "mode": "reextract",
        "source_issue_id": "PI_2_1989",
        "pdf_path": "laws/1989/12/25/MO_PI_2_1989-12-25.pdf",
        "label": "MO_PI_2_1989 (merge 38-char column-break fragment — Q34/Q35)",
    },
    # ── MO_311_2026: cotizatii table fix in ingest layer → re-ingest only
    {
        "mode": "reingest",
        "source_issue_id": "PI_311_2026",
        "json_path": "extracted/2026/04/20/MO_PI_311_2026-04-20.json",
        "label": "MO_PI_311_2026 (cotizatii table restructure — Q56)",
    },
]


def reingest_issue(source_issue_id: str, json_path: str, label: str, settings) -> None:
    """Delete existing DB entries and re-ingest from an existing JSON file."""
    from legalro_processing.ingest_module import run_ingestion
    from legalro_processing.extract.gazette_extractor import load_gazette

    db = get_db(settings)

    deleted = db.chunks.delete_many({"source_issue_id": source_issue_id})
    print(f"  Deleted {deleted.deleted_count} chunks for {source_issue_id}")

    gazette = load_gazette(Path(json_path))
    db.gazettes.delete_many({"sha256": gazette.sha256})
    db.gazettes.delete_many({"filename": gazette.filename})

    result = run_ingestion(json_path, settings)
    print(f"  Ingested: {result['chunks_created']} chunks, {result['acts_ingested']} acts — status={result['status']}")


def reextract_and_reingest(source_issue_id: str, pdf_path: str, label: str, settings) -> None:
    """Delete cached JSON + DB entries, re-run OCR+segmentation from PDF, then re-ingest.

    Use this when segment.py changed and the cached JSON must be regenerated.
    """
    from legalro_processing.extract_module import run_extraction, _expected_json_path
    from legalro_processing.ingest_module import run_ingestion
    from legalro_processing.extract.gazette_extractor import load_gazette

    db = get_db(settings)
    root = Path.cwd()
    extracted_dir = root / "extracted"

    # Delete existing DB entries
    deleted = db.chunks.delete_many({"source_issue_id": source_issue_id})
    print(f"  Deleted {deleted.deleted_count} chunks for {source_issue_id}")

    # Locate and delete cached JSON so run_extraction re-OCRs
    pdf_full = root / pdf_path
    json_path = _expected_json_path(pdf_full, extracted_dir)
    if json_path and json_path.exists():
        gazette = load_gazette(json_path)
        db.gazettes.delete_many({"sha256": gazette.sha256})
        db.gazettes.delete_many({"filename": gazette.filename})
        json_path.unlink()
        print(f"  Deleted cached JSON: {json_path}")
    else:
        print(f"  No cached JSON found (or already absent)")

    # Re-extract from PDF (applies updated segment.py logic)
    print(f"  Re-extracting {pdf_path} …")
    new_json = run_extraction(str(pdf_full), settings, extracted_dir)
    print(f"  Re-extracted → {new_json}")

    # Re-ingest (applies updated ingest_module.py cotizatii fix too)
    result = run_ingestion(str(new_json), settings)
    print(f"  Ingested: {result['chunks_created']} chunks, {result['acts_ingested']} acts — status={result['status']}")


def main():
    root = Path(__file__).parent.parent
    os.chdir(root)

    settings = load_settings("config/cloud.yaml")
    db = get_db(settings)
    total_before = db.chunks.count_documents({})
    print(f"\nSTAGE DB chunks before: {total_before}")

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Processing: {target['label']}  [mode={target['mode']}]")

        mode = target["mode"]

        if mode == "reingest":
            json_full = root / target["json_path"]
            if not json_full.exists():
                print(f"  ERROR: JSON not found at {json_full}")
                continue
            reingest_issue(
                source_issue_id=target["source_issue_id"],
                json_path=str(json_full),
                label=target["label"],
                settings=settings,
            )

        elif mode == "reextract":
            pdf_full = root / target["pdf_path"]
            if not pdf_full.exists():
                print(f"  ERROR: PDF not found at {pdf_full}")
                continue
            reextract_and_reingest(
                source_issue_id=target["source_issue_id"],
                pdf_path=target["pdf_path"],
                label=target["label"],
                settings=settings,
            )

        else:
            print(f"  ERROR: unknown mode '{mode}'")

    total_after = db.chunks.count_documents({})
    print(f"\n{'='*60}")
    print(f"STAGE DB chunks after:  {total_after}  (delta={total_after - total_before:+d})")

    # Spot-checks
    print("\nVerifying MO_2_1989 chunks:")
    for chunk in db.chunks.find(
        {"source_issue_id": "PI_2_1989"},
        {"act_number": 1, "text": 1},
    ):
        print(f"  act_number={chunk.get('act_number')} | text[:150]={chunk.get('text','')[:150]}")

    print("\nVerifying PI_311_2026 Oltenilor chunk prefix:")
    for chunk in db.chunks.find(
        {"source_issue_id": "PI_311_2026", "text": {"$regex": "STRUCTURA TABEL", "$options": "i"}},
        {"text": 1},
    ):
        print(f"  text[:300]={chunk.get('text','')[:300]}")
        break

    print("\nVerifying MO_820_2007 act_number groups:")
    from pymongo import ASCENDING
    for doc in db.chunks.aggregate([
        {"$match": {"source_issue_id": "PI_820_2007"}},
        {"$group": {"_id": "$act_number", "count": {"$sum": 1}, "auth": {"$first": "$issuing_authority"}}},
        {"$sort": {"_id": ASCENDING}},
    ]):
        print(f"  act_number={doc['_id']} count={doc['count']} auth={doc['auth'][:50] if doc['auth'] else ''}")


if __name__ == "__main__":
    main()
