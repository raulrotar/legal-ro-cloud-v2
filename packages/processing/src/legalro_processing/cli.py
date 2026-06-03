"""Processing CLI — the batch entrypoint that runs on the VPS.

    legalro-process extract --root laws/ --out out/      # Stage A: PDF -> bundle
    legalro-process load    --root out/  --mongo "$URI"  # Stage B: bundle -> Mongo

Stage A (extract) is fully wired below; Stage B (load) lives in loader.py.
"""
from __future__ import annotations

from pathlib import Path

import typer
from legalro_core.config import _load_dotenv

# Load .env at import time so every command sees MONGODB_URI / API keys from .env
_load_dotenv()

app = typer.Typer(help="LegalRo processing pipeline (Stage A extract+embed, Stage B load).")


@app.command()
def extract(
    root: Path = typer.Option(..., help="Directory of source PDFs (searched recursively)."),
    out: Path = typer.Option(Path("out"), help="Bundle output root."),
    extracted_dir: Path = typer.Option(Path("extracted"), help="Where to save/cache GazetteDocument JSONs."),
    config: str = typer.Option(None, help="Path to a config yaml (defaults to env/cloud.yaml)."),
    embed: bool = typer.Option(True, help="Compute bge-m3 embeddings (off = fast structural run)."),
    gzip_chunks: bool = typer.Option(True, help="gzip chunks.jsonl in the bundle."),
    limit: int = typer.Option(0, help="Process at most N PDFs (0 = all). Useful for the pilot."),
):
    """Stage A: PDF -> structured JSON -> chunk -> embed -> on-disk bundle."""
    from legalro_core.config import load_settings
    from legalro_processing.extract_module import run_extraction
    from legalro_processing.extract.gazette_extractor import load_gazette
    from legalro_processing.prepare.build import build_issue_docs
    from legalro_processing.bundle_writer import emit_doc

    settings = load_settings(config)
    out.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(root.rglob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        typer.echo(f"No PDFs found under {root}")
        raise typer.Exit(code=1)

    typer.echo(f"[extract] {len(pdfs)} PDF(s) -> {out}  extracted_dir={extracted_dir}  (embed={embed})")
    ok, failed, cov_min = 0, 0, 1.0
    for i, pdf in enumerate(pdfs, 1):
        try:
            json_path = run_extraction(pdf, settings, extracted_dir=extracted_dir)
            gazette = load_gazette(json_path)
            issue_id, sha, gazette_doc, chunks, coverage = build_issue_docs(
                gazette, pdf, settings, embed=embed
            )
            emit_doc(out, issue_id, sha, coverage, gazette_doc, chunks,
                     gzip_chunks=gzip_chunks)
            cov_min = min(cov_min, coverage)
            ok += 1
            typer.echo(f"  [{i}/{len(pdfs)}] {issue_id}: {len(chunks)} chunks, coverage={coverage:.3f}")
        except Exception as exc:  # noqa: BLE001 — never let one bad PDF kill the batch
            failed += 1
            typer.echo(f"  [{i}/{len(pdfs)}] FAILED {pdf.name}: {exc}")

    typer.echo(f"[extract] done: {ok} ok, {failed} failed, coverage_min={cov_min:.3f}")
    typer.echo(f"[extract] manifest: {out / 'manifest.jsonl'}")


@app.command()
def extract_json(
    root: Path = typer.Option(..., help="Directory of source PDFs (searched recursively)."),
    extracted_dir: Path = typer.Option(Path("extracted"), help="Output directory for GazetteDocument JSONs."),
    config: str = typer.Option(None, help="Path to a config yaml."),
    limit: int = typer.Option(0, help="Process at most N PDFs (0 = all)."),
):
    """PDF → GazetteDocument JSON only. No embeddings, no MongoDB.

    Use this to test/iterate on extraction quality (including LLM extraction)
    without running the full pipeline. JSONs are saved under --extracted-dir.

    Example — run with LLM extraction into a separate folder:
      legalro-process extract-json \\
        --root laws/ \\
        --extracted-dir extracted_llm/ \\
        --config config/local.yaml
    """
    from legalro_core.config import load_settings
    from legalro_processing.extract_module import run_extraction

    settings = load_settings(config)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(root.rglob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        typer.echo(f"No PDFs found under {root}")
        raise typer.Exit(code=1)

    llm_on = getattr(getattr(settings, "extraction_llm", None), "enabled", False)
    typer.echo(f"[extract-json] {len(pdfs)} PDF(s) -> {extracted_dir}  llm={llm_on}")

    ok = failed = 0
    for i, pdf in enumerate(pdfs, 1):
        try:
            json_path = run_extraction(pdf, settings, extracted_dir=extracted_dir)
            typer.echo(f"  [{i}/{len(pdfs)}] {pdf.name} -> {json_path}")
            ok += 1
        except Exception as exc:
            failed += 1
            typer.echo(f"  [{i}/{len(pdfs)}] FAILED {pdf.name}: {exc}")

    typer.echo(f"[extract-json] done: {ok} ok, {failed} failed")


@app.command()
def load(
    root: Path = typer.Option(Path("out"), help="Bundle root produced by `extract`."),
    mongo: str = typer.Option(None, help="MongoDB URI. Defaults to $MONGODB_URI env var."),
    db: str = typer.Option("legalro", help="Database name."),
    resume: bool = typer.Option(True, help="Skip docs already loaded (.load_state.json)."),
):
    """Stage B: idempotent bulk-upsert of a bundle into MongoDB."""
    import os
    from legalro_processing.loader import load_bundle

    uri = mongo or os.environ.get("MONGODB_URI", "")
    if not uri:
        typer.echo("ERROR: provide --mongo or set $MONGODB_URI", err=True)
        raise typer.Exit(code=1)

    result = load_bundle(root, uri, db_name=db, resume=resume)
    typer.echo(f"[load] loaded={result['loaded']} skipped={result['skipped']}")


@app.command()
def setup_indexes(
    mongo: str = typer.Option(None, help="MongoDB URI. Defaults to $MONGODB_URI env var."),
    db: str = typer.Option("legalro", help="Database name."),
    wait: bool = typer.Option(True, help="Wait up to 120s for Atlas Search indexes to become READY."),
):
    """Create/update MongoDB Atlas Search + vector indexes for v2 chunk schema."""
    import os, time
    from pymongo import MongoClient

    uri = mongo or os.environ.get("MONGODB_URI", "")
    if not uri:
        typer.echo("ERROR: provide --mongo or set $MONGODB_URI", err=True)
        raise typer.Exit(code=1)

    database = MongoClient(uri)[db]
    _ensure_indexes(database)

    if wait:
        _wait_for_indexes(database)


@app.command()
def reset_db(
    mongo: str = typer.Option(None, help="MongoDB URI. Defaults to $MONGODB_URI env var."),
    db: str = typer.Option("legalro", help="Database name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Drop all LegalRo collections from MongoDB (data wipe — use before a clean reingest)."""
    import os
    from pymongo import MongoClient
    from legalro_core import schema

    uri = mongo or os.environ.get("MONGODB_URI", "")
    if not uri:
        typer.echo("ERROR: provide --mongo or set $MONGODB_URI", err=True)
        raise typer.Exit(code=1)

    if not yes:
        typer.confirm(f"Drop ALL data in database '{db}'?", abort=True)

    database = MongoClient(uri)[db]
    for coll in (schema.COLL_GAZETTES, schema.COLL_CHUNKS, schema.COLL_EDGES, schema.COLL_RUNS):
        database[coll].drop()
        typer.echo(f"  dropped: {coll}")
    typer.echo("[reset-db] done")


# ── index helpers (shared by setup_indexes + reset_db flow) ──────────────────

def _ensure_indexes(database) -> None:
    from legalro_core import schema

    database[schema.COLL_GAZETTES].create_index("filename", unique=True, background=True)
    database[schema.COLL_GAZETTES].create_index("year", background=True)
    database[schema.COLL_GAZETTES].create_index("sha256", background=True)
    database[schema.COLL_CHUNKS].create_index(
        [("source_issue_id", 1), ("act_index_in_issue", 1)], background=True)
    database[schema.COLL_CHUNKS].create_index("law_id", background=True)
    database[schema.COLL_CHUNKS].create_index("document_type", background=True)

    existing = {idx["name"]: idx for idx in database[schema.COLL_CHUNKS].list_search_indexes()}

    # v2 vector index — adds modality/publication_date/act_year/embedding_version filters
    vector_fields = [
        {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
        {"type": "filter", "path": "document_type"},
        {"type": "filter", "path": "law_id"},
        {"type": "filter", "path": "chunk_type"},
        {"type": "filter", "path": "modality"},
        {"type": "filter", "path": "publication_date"},
        {"type": "filter", "path": "act_year"},
        {"type": "filter", "path": "embedding_version"},
    ]
    if "chunks_vector" in existing:
        database[schema.COLL_CHUNKS].update_search_index("chunks_vector", {"fields": vector_fields})
        typer.echo("  updated: chunks_vector")
    else:
        database[schema.COLL_CHUNKS].create_search_index({
            "name": "chunks_vector", "type": "vectorSearch",
            "definition": {"fields": vector_fields},
        })
        typer.echo("  created: chunks_vector")

    text_definition = {
        "mappings": {"dynamic": False, "fields": {
            "text":            {"type": "string", "analyzer": "ro_analyzer"},
            "text_normalized": {"type": "string", "analyzer": "lucene.romanian"},
            "title":           {"type": "string", "analyzer": "ro_analyzer"},
        }},
        "analyzers": [{"name": "ro_analyzer", "charFilters": [],
            "tokenizer": {"type": "standard"},
            "tokenFilters": [
                {"type": "lowercase"},
                {"type": "icuFolding"},
                {"type": "stopword", "tokens": [
                    "și","de","la","în","cu","pe","pentru","din","care","este",
                    "sau","nu","se","prin","acest","această","ale","cel","cea",
                    "si","in",
                ]},
                {"type": "snowballStemming", "stemmerName": "romanian"},
            ],
        }],
    }
    if "chunks_search_ro" in existing:
        database[schema.COLL_CHUNKS].update_search_index("chunks_search_ro", text_definition)
        typer.echo("  updated: chunks_search_ro")
    else:
        database[schema.COLL_CHUNKS].create_search_index({
            "name": "chunks_search_ro", "type": "search",
            "definition": text_definition,
        })
        typer.echo("  created: chunks_search_ro")

    typer.echo("[setup-indexes] done")


def _wait_for_indexes(database, timeout: int = 120) -> None:
    import time
    from legalro_core import schema

    target = {"chunks_vector", "chunks_search_ro"}
    deadline = time.time() + timeout
    typer.echo(f"[setup-indexes] waiting for READY (up to {timeout}s)...")
    while time.time() < deadline:
        statuses = {
            idx["name"]: idx.get("status", "unknown")
            for idx in database[schema.COLL_CHUNKS].list_search_indexes()
            if idx["name"] in target
        }
        ready = {n for n, s in statuses.items() if s == "READY"}
        if ready == target:
            typer.echo("[setup-indexes] indexes READY ✓")
            return
        time.sleep(5)
    typer.echo("[setup-indexes] timed out — indexes may still be building (check Atlas UI)")


if __name__ == "__main__":
    app()
