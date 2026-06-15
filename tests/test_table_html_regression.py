"""QA §3 — HTML-table extraction regression tests (offline, deterministic).

Validates the Phase-1 HTML-table feature against the worst-case 294Bis
Nomenclator (a transposed multi-level-header table with 90°-rotated cells).
All checks run the deterministic PyMuPDF find_tables annex path — NO Ollama,
NO services.  Spec: docs/qa_html_tables_2026-06-15.md §1 (oracle) + §3.

§3.1 (no text-bleed), §3.4 (oracle cells), §3.5 (tag-free flat view),
§3.6 (coverage not inflated) must pass.  §3.2 (true colspan) and §3.3
(hierarchy via spans) are xfail — true geometric/Docling spans are Phase 2.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from legalro_processing.extract.annex_tables import (
    _to_flat_text,
    _to_html,
    extract_annex_tables,
)

_PDF = Path(__file__).resolve().parents[1] / "laws/2026/04/14/MO_PI_294Bis_2026-04-14.pdf"

pytestmark = pytest.mark.skipif(
    not _PDF.exists(), reason="294Bis source PDF not present in this checkout"
)


@pytest.fixture(scope="module")
def nomenclator():
    """The 294Bis Nomenclator first table (page index 2 = PDF page 3),
    freshly fitz-extracted.  Table.page is 0-based.

    rebuild_cells=True mirrors the html_tables_annex flag ON — this suite
    validates the Phase-1 feature, which is what enables the rotated-cell
    repair / html / text_flat fields it asserts on.
    """
    tables = extract_annex_tables(str(_PDF), rebuild_cells=True)
    assert tables, "annex extractor returned no tables"
    return tables[0]


# ── §3.1 — no merged-header text-bleed; cell tokens in source order ──────────
_BLEED_BIGRAMS = [
    "anorganice mediului",          # was: "...și protecția anorganice mediului"
    "și (TIC) comunicațiilor",      # was: "...informației și (TIC) comunicațiilor"
]


def test_no_text_bleed_bigrams(nomenclator):
    flat = nomenclator.text_flat
    for bigram in _BLEED_BIGRAMS:
        assert bigram not in flat, f"text-bleed bigram leaked into cells: {bigram!r}"
    # the correctly-ordered forms must be present instead
    assert "anorganice și protecția mediului" in flat
    assert "comunicațiilor (TIC)" in flat


# ── §3.2 — true colspan on the domain header band (Phase 2) ─────────────────
@pytest.mark.xfail(reason="true colspan/rowspan spans = Phase 2", strict=False)
def test_header_colspan_preserved(nomenclator):
    # Phase 1 emits a flat grid (the band "Matematică" repeats as 3 leaf <th>).
    # Phase 2 will collapse it into <th colspan="3">Matematică</th>.
    assert 'colspan="3"' in nomenclator.html
    assert re.search(r'<th colspan="\d+">\s*Matematică', nomenclator.html)


# ── §3.3 — hierarchy recoverable from spans (Phase 2) ───────────────────────
@pytest.mark.xfail(reason="hierarchy via true spans = Phase 2", strict=False)
def test_hierarchy_specialization_to_domain(nomenclator):
    # "Inteligență artificială" → domain "Informatică", 180 credite.
    # Needs span-encoded hierarchy; the flat grid loses the grouping.
    html = nomenclator.html
    assert "Inteligență artificială" in html and "Informatică" in html
    # Real capability check: resolve the specialization's domain via the colspan
    # on the domain header band.  Phase 1 emits a flat grid (no colspan), so this
    # fails today and the test xfails; once Phase 2 emits true spans the leaf
    # column under "Inteligență artificială" maps back to the "Informatică"
    # band header and this XPASSes — the signal that Phase 2 landed.
    m = re.search(
        r'<th colspan="(\d+)">\s*Informatică\s*</th>', html)
    assert m, "domain 'Informatică' not encoded as a spanning header (no colspan)"
    assert int(m.group(1)) > 1, "Informatică band must span >1 specialization column"


# ── §3.4 — oracle cells exact (from §1) ─────────────────────────────────────
# Transposed grid: attribute rows × specialization columns.  Each oracle row is
# (specializare, cod ISCED de specializare, credite, cod S).  The grid carries:
#   row "Cod ISCED - 2013 F", row "Număr de credite", row "Specializarea (S)",
#   row "Cod S" — we locate the column by specialization name and read across.
_ORACLE = [
    ("Matematică", "0541", "180", "10"),
    ("Matematici aplicate", "0541", "180", "20"),
    ("Informatică", "0613", "180", "10"),
    ("Inteligență artificială", "0619", "180", "30"),
    ("Securitate informatică și știința datelor", "0613", "180", "40"),
    ("Fizică", "0533", "180", "10"),
]


def _grid_from_flat(text_flat: str) -> list[list[str]]:
    return [line.split("\t") for line in text_flat.splitlines()]


def _row_by_label(grid, label):
    for r in grid:
        if r and r[0].strip().startswith(label):
            return r
    return None


def test_oracle_cells_recoverable(nomenclator):
    grid = _grid_from_flat(nomenclator.text_flat)
    isced = _row_by_label(grid, "Cod ISCED")
    credite = _row_by_label(grid, "Număr de credite")
    spec = _row_by_label(grid, "Specializarea")
    cod_s = _row_by_label(grid, "Cod S")
    assert isced and credite and spec and cod_s, "oracle attribute rows missing"

    for name, want_isced, want_credite, want_cod_s in _ORACLE:
        # find the column whose specialization cell exactly matches
        cols = [i for i, c in enumerate(spec) if c.strip() == name]
        assert cols, f"specialization not found in grid: {name!r}"
        col = cols[0]
        assert isced[col].strip() == want_isced, (
            f"{name}: ISCED {isced[col]!r} != {want_isced!r}")
        assert credite[col].strip() == want_credite, (
            f"{name}: credite {credite[col]!r} != {want_credite!r}")
        assert cod_s[col].strip() == want_cod_s, (
            f"{name}: Cod S {cod_s[col]!r} != {want_cod_s!r}")


# ── §3.5 — flattened view is clean & tag-free; HTML separate ────────────────
def test_flat_view_tag_free(nomenclator):
    flat = nomenclator.text_flat
    for bad in ("<", ">", "colspan"):
        assert bad not in flat, f"tag/attr leaked into flat view: {bad!r}"
    # HTML view, by contrast, IS tagged and carries the same cell data
    assert nomenclator.html.startswith("<table><tr><th>")
    assert "0619" in nomenclator.html


def test_text_embedded_prefix_is_tag_free():
    # build.py composes text_embedded as "[TABLE …] " + flat text; the flat
    # renderer must never emit tags so bge-m3 / BM25 stay tag-free.
    flat = _to_flat_text(["Specializarea", "Credite"], [["Fizică", "180"]])
    # no table markup tags are ever introduced by the flat renderer
    assert "<table>" not in flat and "<th>" not in flat and "<td>" not in flat
    assert "Fizică" in flat and "180" in flat
    # HTML renderer escapes structural chars in cell text
    html = _to_html(["A & B", "C <x>"], [["1", "2"]])
    assert "&amp;" in html and "&lt;x&gt;" in html


# ── §3.6 — coverage not inflated (HTML must not enter coverage payload) ──────
def test_coverage_payload_is_flat_not_html(nomenclator):
    # The coverage metric counts tbl.text_flat (or markdown), never tbl.html.
    # Tags would inflate char-overlap against the raw PDF text; assert the flat
    # view shares no angle-bracket noise with HTML.
    assert "<table>" not in nomenclator.text_flat
    assert "<th>" not in nomenclator.markdown
    # html length > flat length (tags add bytes) — proves they are distinct views
    assert len(nomenclator.html) > len(nomenclator.text_flat)
