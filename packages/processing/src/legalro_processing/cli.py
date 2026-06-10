"""Processing CLI — local extraction + cloud push.

    legalro-process extract --root laws/ --out db/bundle_bge-m3/      # Stage A: PDF -> bundle (local)
    legalro-process load    --root db/bundle_bge-m3/  --mongo "$URI"  # Stage B: push bundle -> Mongo (cloud)

Extraction and processing run fully locally.  Only `load` touches the remote server.
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
    out: Path = typer.Option(Path("db/bundle_bge-m3"), help="Bundle output root."),
    extracted_dir: Path = typer.Option(Path("db/extracted"), help="Where to save/cache GazetteDocument JSONs."),
    config: str = typer.Option(None, help="Path to a config yaml (defaults to config/local.yaml)."),
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

    # ── Fallback merge pass ───────────────────────────────────────────────────
    # Collect any JSONs that still have VALIDATE:* error tags after the inline
    # retries.  Re-extract each from its source PDF using the regex pipeline
    # (no LLM) and merge the results field-by-field into the primary JSON.
    # The merged bundle slice is then re-written so the load step gets
    # the best possible data.
    from legalro_processing.fallback_merge import collect_flagged, run_fallback_merge, find_source_pdf

    flagged = collect_flagged(extracted_dir)
    if flagged:
        typer.echo(f"\n[fallback] {len(flagged)} flagged JSON(s) — running regex merge pass")
        merged_ok = merged_failed = 0
        for json_path in flagged:
            pdf_path = find_source_pdf(json_path, root)
            if pdf_path is None:
                typer.echo(f"  SKIP {json_path.name} — source PDF not found under {root}")
                merged_failed += 1
                continue
            try:
                merged_path = run_fallback_merge(json_path, pdf_path, settings)
                # Rebuild bundle slice from merged JSON
                gazette = load_gazette(merged_path)
                issue_id, sha, gazette_doc, chunks, coverage = build_issue_docs(
                    gazette, pdf_path, settings, embed=embed
                )
                emit_doc(out, issue_id, sha, coverage, gazette_doc, chunks,
                         gzip_chunks=gzip_chunks)
                typer.echo(f"  merged {json_path.name}: {len(chunks)} chunks")
                merged_ok += 1
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"  FAILED {json_path.name}: {exc}")
                merged_failed += 1
        typer.echo(f"[fallback] done: {merged_ok} merged, {merged_failed} failed")
    else:
        typer.echo("[fallback] no flagged JSONs — all clean")

    # Release Ollama models from RAM now that the full batch is finished.
    if getattr(settings.llm, "provider", "") == "ollama":
        _ollama_unload(settings.llm.base_url, settings.llm.model)
        if getattr(settings.embeddings, "provider", "") == "ollama":
            _ollama_unload(settings.llm.base_url, settings.embeddings.model)


@app.command()
def extract_json(
    root: Path = typer.Option(..., help="Directory of source PDFs (searched recursively)."),
    extracted_dir: Path = typer.Option(Path("db/extracted"), help="Output directory for GazetteDocument JSONs."),
    config: str = typer.Option(None, help="Path to a config yaml."),
    limit: int = typer.Option(0, help="Process at most N PDFs (0 = all)."),
):
    """PDF → GazetteDocument JSON only. No embeddings, no MongoDB.

    Use this to test/iterate on extraction quality (including LLM extraction)
    without running the full pipeline. JSONs are saved under --extracted-dir.

    Example — run with LLM extraction into a separate folder:
      legalro-process extract-json \\
        --root laws/ \\
        --extracted-dir db/extracted/ \\
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

    # Fallback merge pass — same as extract command, JSON-only (no bundle rebuild)
    from legalro_processing.fallback_merge import collect_flagged, run_fallback_merge, find_source_pdf

    flagged = collect_flagged(extracted_dir)
    if flagged:
        typer.echo(f"\n[fallback] {len(flagged)} flagged JSON(s) — running regex merge pass")
        merged_ok = merged_failed = 0
        for json_path in flagged:
            pdf_path = find_source_pdf(json_path, root)
            if pdf_path is None:
                typer.echo(f"  SKIP {json_path.name} — source PDF not found under {root}")
                merged_failed += 1
                continue
            try:
                run_fallback_merge(json_path, pdf_path, settings)
                typer.echo(f"  merged {json_path.name}")
                merged_ok += 1
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"  FAILED {json_path.name}: {exc}")
                merged_failed += 1
        typer.echo(f"[fallback] done: {merged_ok} merged, {merged_failed} failed")
    else:
        typer.echo("[fallback] no flagged JSONs — all clean")

    if getattr(settings.llm, "provider", "") == "ollama":
        _ollama_unload(settings.llm.base_url, settings.llm.model)


@app.command()
def load(
    root: Path = typer.Option(Path("db/bundle_bge-m3"), help="Bundle root produced by `extract`."),
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
    import os
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


@app.command()
def validate_extractions(
    extracted_dir: Path = typer.Option(Path("db/extracted"), help="Directory of GazetteDocument JSONs to validate."),
    laws_dir: Path = typer.Option(Path("laws"), help="PDF source directory (for re-extraction)."),
    out: Path = typer.Option(Path("db/bundle_bge-m3"), help="Bundle output root (for re-extraction)."),
    config: str = typer.Option(None, help="Path to a config yaml."),
    reextract: bool = typer.Option(False, "--reextract", help="Re-extract files with ERROR-level issues."),
    embed: bool = typer.Option(True, help="Compute embeddings during re-extraction."),
    severity: str = typer.Option("WARNING", help="Minimum severity to report: ERROR, WARNING, INFO."),
    report: Path = typer.Option(None, help="Write JSON report to this path."),
):
    """Validate extracted JSONs and optionally re-extract files with issues.

    Scans every GazetteDocument JSON under --extracted-dir, runs quality
    checks, and prints a report.  With --reextract, files with ERROR-level
    issues are deleted and re-extracted from the original PDF using the
    current pipeline (including the draft-then-verify LLM stage).

    Example — validate only:
      legalro-process validate-extractions --extracted-dir db/extracted/

    Example — validate + auto-fix:
      legalro-process validate-extractions \\
        --extracted-dir db/extracted/ \\
        --laws-dir laws/ \\
        --out db/bundle_bge-m3/ \\
        --reextract \\
        --config config/local.yaml
    """
    import json
    from legalro_processing.extraction_validator import (
        validate_directory, group_by_file, Severity,
    )

    min_sev = Severity(severity.upper())
    sev_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    min_sev_rank = sev_order[min_sev]

    typer.echo(f"[validate] scanning {extracted_dir} …")
    all_issues = validate_directory(extracted_dir)

    # Filter by minimum severity
    shown = [i for i in all_issues if sev_order[i.severity] <= min_sev_rank]

    # Print report
    by_file = group_by_file(shown)
    errors = [i for i in all_issues if i.severity == Severity.ERROR]
    warnings = [i for i in all_issues if i.severity == Severity.WARNING]
    infos = [i for i in all_issues if i.severity == Severity.INFO]

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Validation summary — {len(all_issues)} issues across "
               f"{len(group_by_file(all_issues))} files")
    typer.echo(f"  ERROR: {len(errors)}  WARNING: {len(warnings)}  INFO: {len(infos)}")
    typer.echo(f"{'='*60}\n")

    for json_path, file_issues in sorted(by_file.items()):
        gazette_id = file_issues[0].gazette_id
        has_error  = any(i.severity == Severity.ERROR for i in file_issues)
        marker = "❌" if has_error else "⚠️ "
        typer.echo(f"{marker} {gazette_id}  ({json_path})")
        for issue in file_issues:
            act_tag = f"act[{issue.act_index}]" if issue.act_index is not None else "gazette"
            typer.echo(f"    [{issue.severity.value:7}] {issue.check:28} {act_tag}: {issue.message}")
        typer.echo("")

    # JSON report
    if report:
        report_data = [
            {
                "check":      i.check,
                "severity":   i.severity.value,
                "gazette_id": i.gazette_id,
                "json_path":  str(i.json_path),
                "act_index":  i.act_index,
                "act_number": i.act_number,
                "doc_type":   i.doc_type,
                "message":    i.message,
            }
            for i in all_issues
        ]
        report.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[validate] report written to {report}")

    # Re-extraction
    if not reextract:
        error_files = {i.json_path for i in errors}
        if error_files:
            typer.echo(f"[validate] {len(error_files)} file(s) with ERROR-level issues. "
                       f"Run with --reextract to fix them.")
        return

    # Identify files to re-extract (ERROR-level issues only)
    error_file_issues = {
        json_path: file_issues
        for json_path, file_issues in group_by_file(all_issues).items()
        if any(i.severity == Severity.ERROR for i in file_issues)
    }
    if not error_file_issues:
        typer.echo("[validate] no ERROR-level issues — nothing to re-extract.")
        return

    from legalro_core.config import load_settings
    from legalro_processing.extract_module import run_extraction
    from legalro_processing.extract.gazette_extractor import load_gazette
    from legalro_processing.prepare.build import build_issue_docs
    from legalro_processing.bundle_writer import emit_doc

    settings = load_settings(config)
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\n[validate] re-extracting {len(error_file_issues)} file(s) …\n")
    ok = failed = 0
    for json_path, file_issues in error_file_issues.items():
        gazette_id = file_issues[0].gazette_id

        # Find the original PDF — mirror the extracted path back to laws/
        pdf_path = _find_source_pdf(json_path, extracted_dir, laws_dir)
        if pdf_path is None:
            typer.echo(f"  SKIP {gazette_id}: source PDF not found under {laws_dir}")
            failed += 1
            continue

        issues_summary = ", ".join(
            f"{i.check}(act[{i.act_index}])" for i in file_issues if i.severity == Severity.ERROR
        )
        typer.echo(f"  re-extracting {gazette_id}  issues: {issues_summary}")

        try:
            # Delete stale JSON so run_extraction re-extracts it
            json_path.unlink(missing_ok=True)
            new_json_path = run_extraction(pdf_path, settings, extracted_dir=extracted_dir)

            # Validate the freshly extracted file
            fresh_issues = [
                i for i in validate_directory(extracted_dir)  # scoped to this file only
                if i.json_path == new_json_path and i.severity == Severity.ERROR
            ]

            # Rebuild bundle slice
            gazette = load_gazette(new_json_path)
            issue_id, sha, gazette_doc, chunks, coverage = build_issue_docs(
                gazette, pdf_path, settings, embed=embed
            )
            emit_doc(out, issue_id, sha, coverage, gazette_doc, chunks, gzip_chunks=True)

            remaining = len(fresh_issues)
            status = "✓ fixed" if remaining == 0 else f"⚠ {remaining} ERROR(s) remain"
            typer.echo(f"    {status}  ({len(chunks)} chunks, coverage={coverage:.3f})")
            ok += 1
        except Exception as exc:
            typer.echo(f"    FAILED: {exc}")
            failed += 1

    typer.echo(f"\n[validate] re-extraction done: {ok} ok, {failed} failed")


def _find_source_pdf(json_path: Path, extracted_dir: Path, laws_dir: Path) -> Path | None:
    """Map an extracted JSON path back to the source PDF under laws_dir."""
    try:
        # extracted/2007/12/03/MO_PI_820_2007-12-03.json
        # → laws/2007/12/03/MO_PI_820_2007-12-03.pdf
        rel = json_path.relative_to(extracted_dir)
        candidate = laws_dir / rel.with_suffix(".pdf")
        if candidate.exists():
            return candidate
    except ValueError:
        pass

    # Fallback: search by filename
    stem = json_path.stem  # e.g. MO_PI_820_2007-12-03
    matches = list(laws_dir.rglob(f"{stem}.pdf"))
    return matches[0] if matches else None


def _ollama_unload(base_url: str, model: str) -> None:
    """Ask Ollama to immediately evict a model from RAM (keep_alive=0)."""
    try:
        import httpx
        # Ollama's native endpoint accepts keep_alive; use it directly
        ollama_base = base_url.rstrip("/").removesuffix("/v1")
        httpx.post(
            f"{ollama_base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
        typer.echo(f"[ollama] unloaded {model} from RAM")
    except Exception as exc:
        typer.echo(f"[ollama] unload skipped ({exc})")


if __name__ == "__main__":
    app()
