"""Re-ingest specific MO issues that have wrong metadata or poor chunk structure in STAGE DB.

Target MOs:
  - MO_PI_820_2007: has wrong act metadata (all chunks show ANCEX title, not MIRA taxi title).
    The extracted JSON already has correct separation: ANCEX Ord 346, MIRA Ord 356.
  - MO_PI_2_1989: has "Revoluția a învins" buried after MO header in first chunk.
    Re-ingesting from current extracted JSON puts it at the start of the chunk.

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
    {
        "source_issue_id": "PI_820_2007",
        "json_path": "extracted/2007/12/03/MO_PI_820_2007-12-03.json",
        "label": "MO_PI_820_2007 (ANCEX Ord 346 + MIRA taxi Ord 356 + HG 1448/1449/1450)",
    },
    {
        "source_issue_id": "PI_2_1989",
        "json_path": "extracted/1989/12/25/MO_PI_2_1989-12-25.json",
        "label": "MO_PI_2_1989 (FSN communicat — Revoluția a învins)",
    },
]


def reingest_issue(source_issue_id: str, json_path: str, label: str, settings) -> None:
    from legalro_processing.ingest_module import run_ingestion
    from legalro_processing.extract.gazette_extractor import load_gazette

    db = get_db(settings)

    # Delete existing chunks for this issue
    deleted = db.chunks.delete_many({"source_issue_id": source_issue_id})
    print(f"  Deleted {deleted.deleted_count} chunks for {source_issue_id}")

    # Delete the gazette record so ingestion doesn't skip (sha256 check)
    gazette = load_gazette(Path(json_path))
    db.gazettes.delete_many({"sha256": gazette.sha256})
    db.gazettes.delete_many({"filename": gazette.filename})

    # Re-ingest
    result = run_ingestion(json_path, settings)
    print(f"  Ingested: {result['chunks_created']} chunks, {result['acts_ingested']} acts — status={result['status']}")


def main():
    root = Path(__file__).parent.parent
    os.chdir(root)

    from legalro_core.config import load_settings
    settings = load_settings("config/cloud.yaml")

    db = get_db(settings)
    total_before = db.chunks.count_documents({})
    print(f"\nSTAGE DB chunks before: {total_before}")

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Processing: {target['label']}")
        print(f"  JSON: {target['json_path']}")

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

    total_after = db.chunks.count_documents({})
    print(f"\n{'='*60}")
    print(f"STAGE DB chunks after:  {total_after}  (delta={total_after - total_before:+d})")
    print("\nVerifying new metadata for MO_820_2007:")
    for chunk in db.chunks.find(
        {"source_issue_id": "PI_820_2007", "act_number": {"$in": ["346", "356"]}},
        {"act_number": 1, "title": 1, "issuing_authority": 1, "text": 1},
    ):
        print(f"  act_number={chunk.get('act_number')} | auth={chunk.get('issuing_authority','')[:50]}")
        print(f"    title={chunk.get('title','')[:80]}")
        print(f"    text[:80]={chunk.get('text','')[:80]}")
        break  # just show one example per act_number

    print("\nVerifying MO_2_1989 chunks:")
    for chunk in db.chunks.find(
        {"source_issue_id": "PI_2_1989"},
        {"act_number": 1, "text": 1},
    ):
        print(f"  act_number={chunk.get('act_number')} | text[:120]={chunk.get('text','')[:120]}")


if __name__ == "__main__":
    main()
