"""Main ingestion orchestrator: PDF → GazetteDocument JSON → MongoDB.

Delegates to two independent modules:
  extract_module.run_extraction  — PDF → GazetteDocument JSON (no DB)
  ingest_module.run_ingestion    — JSON → chunk → embed → MongoDB (no PDF)

The JSON is cached in extracted/ and is human-editable: fix metadata or OCR
errors there and re-run without re-OCRing the PDF.
"""
from pathlib import Path

from legalro_core.config import Settings
from legalro_core.models import Era, GazetteResult


def _get_extracted_dir(settings: Settings) -> Path:
    raw = getattr(getattr(settings, "ingestion", None), "extracted_dir", None)
    return Path(raw) if raw else Path("extracted")


def process_gazette(pdf_path: str, settings: Settings) -> GazetteResult:
    """PDF → MongoDB. Orchestrates extraction then ingestion."""
    from legalro_processing.extract_module import run_extraction
    from legalro_processing.ingest_module import run_ingestion

    extracted_dir = _get_extracted_dir(settings)
    json_path = run_extraction(pdf_path, settings, extracted_dir)
    result = run_ingestion(json_path, settings)

    # Resolve era from the ingested JSON for the result object
    from legalro_processing.extract.gazette_extractor import load_gazette
    gazette = load_gazette(json_path)

    return GazetteResult(
        gazette_id=result["gazette_id"],
        era=Era(gazette.era),
        acts_segmented=result["acts_ingested"],
        chunks_created=result["chunks_created"],
        status=result["status"],
        warnings=gazette.extraction_warnings,
    )
