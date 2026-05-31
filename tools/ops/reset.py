#!/usr/bin/env python3
"""
Full project reset: drop DB, delete extracted JSON cache, re-extract, re-ingest,
rebuild search indexes.

Usage:
    uv run python scripts/reset.py [--laws-dir laws/] [--config config/local.yaml]
    uv run python scripts/reset.py --skip-extract   # keep extracted/ JSONs, only re-ingest
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legalro_core.config import load_settings
from legalro_core.store import get_db
from legalro_processing.pipeline import process_gazette


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full LegalRo project reset")
    p.add_argument("--laws-dir", default="laws", help="Directory containing PDF files (default: laws/)")
    p.add_argument("--config", default="config/local.yaml", help="Settings file (default: config/local.yaml)")
    p.add_argument("--skip-extract", action="store_true", help="Keep existing extracted/ JSONs; only drop DB and re-ingest")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    return p.parse_args()


def confirm(msg: str, yes: bool) -> None:
    if yes:
        print(msg)
        return
    answer = input(f"{msg} [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def drop_db(db) -> None:
    print("── Dropping MongoDB collections ────────────────────────────────")
    for coll in ("chunks", "gazettes", "acts"):
        result = db[coll].drop()
        print(f"   dropped: {coll}")
    print()


def delete_extracted(root: Path) -> None:
    extracted_dir = root / "extracted"
    if not extracted_dir.exists():
        print("   extracted/ not found — nothing to delete")
        return
    json_files = list(extracted_dir.rglob("*.json"))
    print(f"   deleting {len(json_files)} JSON files from extracted/ …")
    shutil.rmtree(extracted_dir)
    extracted_dir.mkdir()
    print()


def ingest_all(pdfs: list[Path], settings) -> dict:
    total = len(pdfs)
    counts = {"ok": 0, "skipped": 0, "error": 0}
    for i, pdf in enumerate(pdfs, 1):
        prefix = f"[{i:>2}/{total}]"
        print(f"{prefix} {pdf.name} … ", end="", flush=True)
        try:
            result = process_gazette(str(pdf), settings)
            if result.status == "skipped":
                print("skipped (already ingested)")
                counts["skipped"] += 1
            else:
                warn = f"  ⚠ {'; '.join(result.warnings)}" if result.warnings else ""
                print(f"✓  acts={result.acts_segmented}  chunks={result.chunks_created}{warn}")
                counts["ok"] += 1
        except Exception as exc:
            print(f"✗  ERROR: {exc}")
            counts["error"] += 1
    return counts


def setup_indexes(db, dims: int) -> None:
    print("── Rebuilding search indexes ───────────────────────────────────")

    # Standard B-tree indexes
    db.gazettes.create_index("filename", unique=True, background=True)
    db.gazettes.create_index("year", background=True)
    db.chunks.create_index([("source_issue_id", 1), ("act_index_in_issue", 1)], background=True)
    db.chunks.create_index("law_id", background=True)
    db.chunks.create_index("document_type", background=True)

    existing = {idx["name"]: idx for idx in db.chunks.list_search_indexes()}

    # Vector search index
    vector_fields = [
        {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
        {"type": "filter", "path": "law_id"},
        {"type": "filter", "path": "chunk_type"},
        {"type": "filter", "path": "document_type"},
    ]
    if "chunks_vector" in existing:
        db.chunks.update_search_index("chunks_vector", {"fields": vector_fields})
        print("   updated chunks_vector index")
    else:
        db.chunks.create_search_index({
            "name": "chunks_vector",
            "type": "vectorSearch",
            "definition": {"fields": vector_fields},
        })
        print("   created chunks_vector index")

    # Full-text search index with Romanian analyzer
    text_definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string", "analyzer": "ro_analyzer"},
                "text_normalized": {"type": "string", "analyzer": "lucene.romanian"},
                "title": {"type": "string", "analyzer": "ro_analyzer"},
            },
        },
        "analyzers": [{
            "name": "ro_analyzer",
            "charFilters": [],
            "tokenizer": {"type": "standard"},
            "tokenFilters": [
                {"type": "lowercase"},
                {"type": "icuFolding"},
                {
                    "type": "stopword",
                    "tokens": [
                        "și", "de", "la", "în", "cu", "pe", "pentru",
                        "din", "care", "este", "sau", "nu", "se", "prin",
                        "acest", "această", "ale", "cel", "cea",
                        "si", "in", "pentru", "din",
                    ],
                },
                {"type": "snowballStemming", "stemmerName": "romanian"},
            ],
        }],
    }
    if "chunks_search_ro" in existing:
        db.chunks.update_search_index("chunks_search_ro", text_definition)
        print("   updated chunks_search_ro index")
    else:
        db.chunks.create_search_index({
            "name": "chunks_search_ro",
            "type": "search",
            "definition": text_definition,
        })
        print("   created chunks_search_ro index")
    print()


def wait_for_indexes(db, timeout: int = 120) -> None:
    print("── Waiting for indexes to become READY ─────────────────────────")
    deadline = time.time() + timeout
    target = {"chunks_vector", "chunks_search_ro"}
    while time.time() < deadline:
        statuses = {
            idx["name"]: idx.get("status", "unknown")
            for idx in db.chunks.list_search_indexes()
            if idx["name"] in target
        }
        ready = {name for name, st in statuses.items() if st == "READY"}
        pending = target - ready
        status_str = "  ".join(f"{n}={s}" for n, s in sorted(statuses.items()))
        print(f"   {status_str}", end="\r", flush=True)
        if not pending:
            print(f"\n   All indexes READY ✓")
            return
        time.sleep(5)
    print(f"\n   ⚠  Timed out after {timeout}s — indexes may still be building.")
    print("   Run a test query in a minute; they usually finish shortly after.")
    print()


def main() -> None:
    args = parse_args()

    root = Path(__file__).parent.parent
    laws_dir = (root / args.laws_dir).resolve()
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    pdfs = sorted(laws_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {laws_dir}")
        sys.exit(1)

    action = "Re-ingest only (keep extracted/ JSONs)" if args.skip_extract else "Full reset (drop DB + delete extracted/ JSONs + re-ingest)"
    print("══════════════════════════════════════════════════════════════════")
    print(f"  LegalRo Reset")
    print(f"  Mode    : {action}")
    print(f"  Laws dir: {laws_dir}  ({len(pdfs)} PDFs)")
    print(f"  Config  : {config_path}")
    print("══════════════════════════════════════════════════════════════════")
    confirm("\nProceed?", args.yes)
    print()

    settings = load_settings(str(config_path))
    db = get_db(settings)

    # 1. Drop DB
    drop_db(db)

    # 2. Delete extracted JSONs (unless --skip-extract)
    if not args.skip_extract:
        print("── Deleting extracted/ JSON cache ──────────────────────────────")
        delete_extracted(root)

    # 3. Ingest all PDFs (Phase 1 extract + Phase 2 embed/store)
    print("── Ingesting PDFs ──────────────────────────────────────────────")
    t0 = time.time()
    counts = ingest_all(pdfs, settings)
    elapsed = time.time() - t0
    print()
    print(f"   Ingestion done in {elapsed:.0f}s  —  ✓ {counts['ok']}  skipped {counts['skipped']}  ✗ {counts['error']}")
    print(f"   Total chunks in DB: {db.chunks.count_documents({})}")
    print()

    # 4. Rebuild search indexes
    setup_indexes(db, settings.embeddings.dimensions)

    # 5. Wait for Atlas Search indexes to become ready
    wait_for_indexes(db)

    print("══════════════════════════════════════════════════════════════════")
    print("  Reset complete. Start the LLM server and run QA:")
    print("    uv run legalro start")
    print("    uv run python test_questions.py")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
