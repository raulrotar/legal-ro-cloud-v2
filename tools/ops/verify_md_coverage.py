"""Verify MD cache files against source PDFs.

Calculates:
- Text coverage: how much PDF text appears in the MD
- Token-level precision/recall
- Table detection
- Era-level breakdowns
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import fitz  # pymupdf


ROOT = Path(__file__).parent.parent.parent
LAWS_DIR = ROOT / "laws"
MD_CACHE_DIR = ROOT / "db" / "md_cache"
REPORTS_DIR = ROOT / "reports"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from PDF using pymupdf."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def load_md(md_path: Path) -> str:
    """Load MD, stripping legalro header comments."""
    text = md_path.read_text(encoding="utf-8")
    # Strip HTML comment headers
    text = re.sub(r'<!--legalro:[^>]+-->\n?', '', text)
    # Strip markdown formatting characters for text comparison
    return text


def tokenize(text: str) -> list[str]:
    """Split text into comparable tokens (lowercase words, 3+ chars)."""
    tokens = re.findall(r'\b[a-zăâîșțA-ZĂÂÎȘȚ]{3,}\b', text.lower())
    return tokens


# Matches the colophon footer line that contains ISSN (always present in the footer)
# or a line that is purely a price/page-count token (standalone "lei" / "pagini" lines)
_COLOPHON_RE = re.compile(
    r'(?im)^[^\n]*\bI\.?S\.?S\.?N\.?\b[^\n]*$\n?'
)


def normalize_ro(text: str) -> str:
    """Normalize Romanian diacritics variants, including legacy symbol-font encodings."""
    replacements = {
        'ş': 'ș', 'ţ': 'ț',   # cedilla → comma-below
        'Ş': 'Ș', 'Ţ': 'Ț',
        # Legacy symbol-font PDFs (Jan-2007 batch): diacritics stored as non-letter symbols
        '˛': 'ț',          # ˛ OGONEK → ț
        '∫': 'ș',          # ∫ INTEGRAL SIGN → ș
        '˚': 'ț',          # ˚ RING ABOVE (another variant seen) → ț
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def strip_colophon(text: str) -> str:
    """Remove colophon/footer lines (ISSN, price, page count) before tokenizing."""
    return _COLOPHON_RE.sub('', text)


def token_overlap(pdf_text: str, md_text: str) -> dict:
    """Calculate token-level overlap between PDF and MD."""
    pdf_norm = strip_colophon(normalize_ro(pdf_text))
    md_norm = strip_colophon(normalize_ro(md_text))

    pdf_tokens = set(tokenize(pdf_norm))
    md_tokens = set(tokenize(md_norm))

    if not pdf_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pdf_tokens": 0, "md_tokens": 0, "overlap": 0}

    overlap = pdf_tokens & md_tokens
    precision = len(overlap) / len(md_tokens) if md_tokens else 0.0
    recall = len(overlap) / len(pdf_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pdf_tokens": len(pdf_tokens),
        "md_tokens": len(md_tokens),
        "overlap": len(overlap),
    }


def check_tables(pdf_path: Path, md_text: str) -> dict:
    """Check if tables in PDF are represented in MD."""
    doc = fitz.open(str(pdf_path))
    pdf_table_count = 0
    for page in doc:
        tabs = page.find_tables()
        pdf_table_count += len(tabs.tables)
    doc.close()

    md_table_count = len(re.findall(r'^\|', md_text, re.MULTILINE))
    md_has_tables = md_table_count > 0

    return {
        "pdf_table_rows_detected": pdf_table_count,
        "md_pipe_lines": md_table_count,
        "tables_represented": md_has_tables if pdf_table_count > 0 else None,
    }


def check_structure(md_text: str) -> dict:
    """Check MD structural quality."""
    headings = re.findall(r'^#{1,6} .+', md_text, re.MULTILINE)
    paragraphs = [p for p in md_text.split('\n\n') if p.strip()]
    return {
        "headings": len(headings),
        "paragraphs": len(paragraphs),
        "total_chars": len(md_text),
    }


@dataclass
class FileResult:
    gazette_id: str
    era: str
    pdf_path: Path
    md_path: Path | None
    has_md: bool
    pdf_char_count: int = 0
    md_char_count: int = 0
    token_recall: float = 0.0
    token_precision: float = 0.0
    f1: float = 0.0
    table_info: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    error: str = ""


def infer_era(pdf_path: Path) -> str:
    m = re.search(r'/(\d{4})/', str(pdf_path))
    if m:
        year = int(m.group(1))
        if year < 2000:
            return "communist_era"
        elif year < 2010:
            return "transition"
        elif year < 2020:
            return "modern"
        else:
            return "recent"
    return "unknown"


def analyze_pair(pdf_path: Path, md_cache_dir: Path) -> FileResult:
    gazette_id = pdf_path.stem
    era = infer_era(pdf_path)

    # Find corresponding MD
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', gazette_id)
    if m:
        md_path = md_cache_dir / m.group(1) / m.group(2) / m.group(3) / f"{gazette_id}.md"
    else:
        md_path = None

    has_md = md_path is not None and md_path.exists()

    result = FileResult(
        gazette_id=gazette_id,
        era=era,
        pdf_path=pdf_path,
        md_path=md_path,
        has_md=has_md,
    )

    try:
        pdf_text = extract_pdf_text(pdf_path)
        result.pdf_char_count = len(pdf_text)

        if has_md:
            md_text = load_md(md_path)
            result.md_char_count = len(md_text)

            overlap = token_overlap(pdf_text, md_text)
            result.token_recall = overlap["recall"]
            result.token_precision = overlap["precision"]
            result.f1 = overlap["f1"]

            result.table_info = check_tables(pdf_path, md_text)
            result.structure = check_structure(md_text)
    except Exception as e:
        result.error = str(e)

    return result


def run_verification(laws_dir: Path, md_cache_dir: Path) -> list[FileResult]:
    pdfs = sorted(laws_dir.rglob("*.pdf"))
    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.stem}...", flush=True)
        r = analyze_pair(pdf, md_cache_dir)
        results.append(r)
        status = f"recall={r.token_recall:.1%} precision={r.token_precision:.1%}" if r.has_md else "NO MD"
        print(f"         → {status}", flush=True)
    return results


def generate_report(results: list[FileResult], out_path: Path) -> None:
    lines = ["# MD Coverage Verification Report\n"]
    lines.append(f"**Date:** 2026-06-06  \n**PDFs analyzed:** {len(results)}\n")

    with_md = [r for r in results if r.has_md and not r.error]
    without_md = [r for r in results if not r.has_md]
    errored = [r for r in results if r.error]

    lines.append(f"**MDs present:** {len(with_md)}/{len(results)}  ")
    lines.append(f"**Missing MDs:** {len(without_md)}  ")
    lines.append(f"**Errors:** {len(errored)}\n")

    # Overall metrics
    if with_md:
        avg_recall = sum(r.token_recall for r in with_md) / len(with_md)
        avg_precision = sum(r.token_precision for r in with_md) / len(with_md)
        avg_f1 = sum(r.f1 for r in with_md) / len(with_md)
        lines.append("## Overall Metrics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Avg Token Recall (PDF→MD coverage) | **{avg_recall:.1%}** |")
        lines.append(f"| Avg Token Precision (MD faithfulness) | **{avg_precision:.1%}** |")
        lines.append(f"| Avg F1 | **{avg_f1:.1%}** |")
        lines.append(f"| MD coverage (files present) | **{len(with_md)/len(results):.1%}** |")
        lines.append("")

    # Era breakdown
    lines.append("## By Era\n")
    lines.append("| Era | Files | MD Present | Avg Recall | Avg Precision | Avg F1 |")
    lines.append("|-----|-------|-----------|-----------|--------------|--------|")
    era_groups = defaultdict(list)
    for r in results:
        era_groups[r.era].append(r)
    for era, group in sorted(era_groups.items()):
        ok = [r for r in group if r.has_md and not r.error]
        avg_rec = sum(r.token_recall for r in ok) / len(ok) if ok else 0
        avg_prec = sum(r.token_precision for r in ok) / len(ok) if ok else 0
        avg_f1 = sum(r.f1 for r in ok) / len(ok) if ok else 0
        lines.append(f"| {era} | {len(group)} | {len(ok)}/{len(group)} | {avg_rec:.1%} | {avg_prec:.1%} | {avg_f1:.1%} |")
    lines.append("")

    # Per-file detail
    lines.append("## Per-File Results\n")
    lines.append("| Gazette | Era | MD? | Recall | Precision | F1 | PDF chars | MD chars | Tables (PDF/MD) |")
    lines.append("|---------|-----|-----|--------|-----------|-----|----------|---------|----------------|")
    for r in sorted(results, key=lambda x: x.gazette_id):
        md_str = "✓" if r.has_md else "✗"
        rec = f"{r.token_recall:.1%}" if r.has_md else "—"
        prec = f"{r.token_precision:.1%}" if r.has_md else "—"
        f1 = f"{r.f1:.1%}" if r.has_md else "—"
        pdf_c = f"{r.pdf_char_count:,}" if r.pdf_char_count else "—"
        md_c = f"{r.md_char_count:,}" if r.md_char_count else "—"
        if r.table_info:
            tab = f"{r.table_info.get('pdf_table_rows_detected', 0)}/{r.table_info.get('md_pipe_lines', 0)}"
        else:
            tab = "—"
        lines.append(f"| {r.gazette_id} | {r.era} | {md_str} | {rec} | {prec} | {f1} | {pdf_c} | {md_c} | {tab} |")
    lines.append("")

    # Missing MDs
    if without_md:
        lines.append("## Missing MDs\n")
        for r in without_md:
            lines.append(f"- `{r.gazette_id}` ({r.era})")
        lines.append("")

    # Low coverage files
    low = [r for r in with_md if r.token_recall < 0.5]
    if low:
        lines.append("## Low Coverage Files (recall < 50%)\n")
        for r in sorted(low, key=lambda x: x.token_recall):
            lines.append(f"- `{r.gazette_id}` recall={r.token_recall:.1%} — possible OCR gap or scanned PDF")
        lines.append("")

    # Errors
    if errored:
        lines.append("## Errors\n")
        for r in errored:
            lines.append(f"- `{r.gazette_id}`: {r.error}")
        lines.append("")

    # Notes on methodology
    lines.append("## Methodology Notes\n")
    lines.append("- **Token Recall**: fraction of unique word tokens (3+ chars) from PDF that appear in MD")
    lines.append("- **Token Precision**: fraction of MD tokens that also appear in the PDF")
    lines.append("- For scanned PDFs (1989 era), pymupdf extracts little/no raw text — recall will be low by design (OCR is the ground truth)")
    lines.append("- Table detection uses pymupdf's `find_tables()` for PDF side; MD side counts `|`-delimited lines")
    lines.append("- Diacritics normalized (cedilla→comma-below) before comparison")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--laws", default=str(LAWS_DIR))
    p.add_argument("--md-cache", default=str(MD_CACHE_DIR))
    p.add_argument("--out", default=str(REPORTS_DIR / "md_coverage_report.md"))
    args = p.parse_args()

    laws_dir = Path(args.laws)
    md_cache_dir = Path(args.md_cache)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Verifying MDs in {md_cache_dir} against PDFs in {laws_dir}...")
    results = run_verification(laws_dir, md_cache_dir)
    generate_report(results, out_path)
