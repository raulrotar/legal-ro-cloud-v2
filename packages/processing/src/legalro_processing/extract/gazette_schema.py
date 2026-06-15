"""
Structured intermediate representation for a gazette PDF.

A gazette (Monitorul Oficial) is extracted to this schema ONCE from the PDF,
written to extracted/{year}/{month}/{day}/{filename}.json, and used as the
canonical source for all downstream ingestion.  You can edit the JSON files
directly to fix OCR errors or wrong metadata before re-ingesting.

Eras:
  SCANNED      – fully scanned image (1989, most pre-2001)
  BROKEN_2002  – text PDF but broken Mac-Roman / cp1250 encoding (2001-2006)
  BROKEN_2007  – similar encoding issues, slightly different font set (2007-2012)
  HYBRID       – mix of real text and facsimile pages (some 2012-2016)
  MODERN       – clean UTF-8 text PDF (2016-present)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SumarEntry:
    """One line from the gazette table of contents."""
    act_number: str          # "1.642/2016" or "115" (prime-minister decision)
    doc_type: str            # normalised: ORDIN / DECIZIE / HOTARARE / DECRET / LEGE etc.
    title: str               # full summary description from sumar
    page_start: int          # gazette-relative page where the act starts
    page_end: Optional[int]  # gazette-relative last page (None if single-page)
    category: str            # section header above this entry e.g.
                             # "ACTE ALE ORGANELOR DE SPECIALITATE"


@dataclass
class Article:
    """A single article / articol within a legal act."""
    article_number: str           # "1", "2", "unic" etc.
    title: Optional[str]          # optional heading on the same line
    alineate: list[str]           # each "(1)", "(2)" paragraph as a string
    raw_text: str                 # full verbatim text of the article


@dataclass
class Annex:
    """An annex / anexa attached to an act."""
    annex_number: str        # "1", "2", "Nr. 3" etc.
    title: Optional[str]
    raw_text: str            # full text (may be a table)


@dataclass
class Table:
    """A financial / tabular region extracted from a gazette page.

    Table-dense pages (e.g. AEP party-financing reports) are diverted from
    the act segmenter to avoid phantom acts.  Each table is stored verbatim
    as a Markdown pipe-table and ingested as a retrievable chunk with
    chunk_type='financial_table'.
    """
    markdown: str            # verbatim pipe-table markdown
    page: int                # 0-based PDF page index (best guess from page-break markers)
    title: str               # nearest preceding heading line, or ""
    n_rows: int              # number of data rows (excludes separator row)

    # ── HTML-table feature (Phase 1, flag-gated) ─────────────────────────
    # All default to "" / 0 so the ~50k cached JSONs deserialize via
    # Table(**t) without these keys present.  HTML is the LLM/display view
    # (chunk.act_full_text); text_flat is the tag-free search/embedding view.
    html: str = ""           # single-line flat <table>…</table>, HTML-escaped
    text_flat: str = ""      # tab/space-joined tag-free cell text, source order
    n_cols: int = 0          # column count of the (flat) header band


@dataclass
class LegalAct:
    """
    One legal act extracted from a gazette issue.

    An act maps 1-to-1 with a sumar entry (when the sumar is parseable) or
    to a page-boundary-detected segment (fallback).
    """
    act_index: int                    # 0-based position in this gazette
    doc_type: str                     # ORDIN / DECIZIE / HOTARARE / DECRET / LEGE / COMUNICAT / RECTIFICARE …
    act_number: str                   # "1.642" or "576" or "" for unnumbered acts
    act_year: Optional[int]           # year from the act number, e.g. 2016 for "1.642/2016"
    issuing_authority: str            # "ANCPI" / "GUVERNUL ROMÂNIEI" / "CURTEA CONSTITUȚIONALĂ" …
    title: str                        # full title of the act
    locality: Optional[str]          # county / city when act is locality-specific

    # Textual body
    preamble: str                     # text before Art. 1 (temeiul legal, considerente)
    articles: list[Article]
    annexes: list[Annex]
    full_text: str                    # verbatim concatenated text of the whole act

    # Source coordinates within the gazette
    page_start: int                   # 0-based page index in the PDF
    page_end: int                     # 0-based page index (inclusive)

    # Signatories
    signed_by: list[str]             # e.g. ["Călin Popescu-Tăriceanu"]
    countersigned_by: list[str]      # ministerial countersignatures

    # Validation flags (set by the extraction checker)
    extraction_warnings: list[str] = field(default_factory=list)


@dataclass
class GazetteDocument:
    """
    Top-level structured representation of one gazette PDF.

    Written to extracted/.../MO_PI_76_2017-01-30.json after extraction.
    The file is human-editable; downstream ingestion reads from it, not
    from the original PDF.
    """
    # ── Gazette identity ──────────────────────────────────────────────
    filename: str                  # original PDF filename
    gazette_id: str                # "PI_76_2017" or "PI_294Bis_2026"
    part: str                      # "I", "II", "IV" …
    issue_number: int
    issue_year: int
    issue_date: str                # ISO 8601: "2017-01-30"
    era: str                       # Era enum value as string

    # ── Header info from page 0 ───────────────────────────────────────
    year_label: Optional[str]      # "Anul 185 (XXIX)"
    weekday: Optional[str]         # "Luni" / "Marți" …
    pdf_page_count: int
    sha256: str

    # ── Table of contents ─────────────────────────────────────────────
    sumar: list[SumarEntry]        # parsed from page 0; empty for SCANNED eras
    sumar_raw: str                 # verbatim sumar text for debugging / re-parse

    # ── Acts ──────────────────────────────────────────────────────────
    acts: list[LegalAct]

    # ── Extraction metadata ───────────────────────────────────────────
    extraction_version: str        # semver of the extractor, e.g. "1.0.0"
    extracted_at: str              # ISO 8601 datetime
    is_bis: bool = False           # True for "294Bis" variant issues; default for old JSONs
    extraction_warnings: list[str] = field(default_factory=list)

    # ── Tables (table-dense pages diverted from act segmenter) ────────
    tables: list["Table"] = field(default_factory=list)
