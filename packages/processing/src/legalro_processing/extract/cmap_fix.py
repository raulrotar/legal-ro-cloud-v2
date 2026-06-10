"""Repair broken ToUnicode CMaps in QuarkXPress-era gazette PDFs (2000–2008).

The era's "hacked" Romanian DTP fonts draw diacritic glyph SHAPES in
punctuation slots; the embedded ToUnicode CMap declares the punctuation
identity, so every text extractor sees mojibake:

    „ (U+201E) where the page shows ă        ∫ (U+222B) → ș
    ‚ (U+201A) → â        ˛ (U+02DB) → ț     Ó (U+00D3) → î   …

normalize.py already repairs this at TEXT level (NORMALIZATION_TABLES), but
Docling's parser folds typographic punctuation to ASCII lookalikes BEFORE the
table can run („→' and ‚→,), which collide with real punctuation and leave
~7% of words corrupted (urm'tor, Rom,niei).

This module fixes the problem at the SOURCE: rewrite each font's ToUnicode
bfchar targets so the text layer itself yields correct diacritics.  Every
downstream consumer — Docling, PyMuPDF, the secondary analyzer — then sees
clean text, and the fix is exact because it operates per font codepoint,
not on flattened text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from legalro_core.models import Era
from legalro_core.normalize import NORMALIZATION_TABLES


_TOUNICODE_REF = re.compile(r"/ToUnicode\s+(\d+)\s+0\s+R")
_ENCODING_REF = re.compile(r"/Encoding\s+(\d+)\s+0\s+R")
_BFCHAR_BLOCK = re.compile(r"(beginbfchar)(.*?)(endbfchar)", re.DOTALL)

# Docling's parser ignores ToUnicode and resolves glyphs via the Encoding
# /Differences glyph NAMES, so those must be rewritten too.  Names follow the
# hacked-font convention (the ș shape lives in the /integral slot, etc.);
# targets use the unambiguous uniXXXX form of the Adobe Glyph List algorithm.
_GLYPH_NAME_REMAP: dict[Era, dict[str, str]] = {
    Era.BROKEN_2007: {
        "quotedblbase": "uni0103",   # „ slot draws ă
        "quotesinglbase": "uni00E2", # ‚ slot draws â
        "integral": "uni0219",       # ∫ slot draws ș
        "ogonek": "uni021B",         # ˛ slot draws ț
        "Oacute": "uni00EE",         # Ó slot draws î
        "OE": "uni00CE",             # Œ slot draws Î
        "radical": "uni0102",        # √ slot draws Ă
        "logicalnot": "uni00C2",     # ¬ slot draws Â
        "trademark": "uni0218",      # ™ slot draws Ș
        "fi": "uni021A",             # ﬁ slot draws Ț
        "caron": "uni021A",          # ˇ slot draws Ț
    },
}


def _rewrite_bfchar_block(block: str, remap: dict[int, int]) -> str:
    """Rewrite `<code> <target>` lines whose target unicode is in remap."""

    def _sub(m: re.Match) -> str:
        target = int(m.group(2), 16)
        if target in remap:
            return f"{m.group(1)}<{remap[target]:04X}>"
        return m.group(0)

    # one source code, one 4-hex target (multi-char targets left untouched)
    return re.sub(r"(<[0-9a-fA-F]{2,4}>\s*)<([0-9a-fA-F]{4})>", _sub, block)


def fix_tounicode(
    pdf_path: str | Path,
    era: Era,
    out_path: str | Path | None = None,
) -> Path:
    """Write a copy of the PDF with corrected ToUnicode CMaps.

    Only bfchar entries are touched (bfrange targets are ASCII runs here and
    must not be rewritten).  Returns the path of the fixed PDF; if no font
    needed fixing, the original path is returned unchanged.
    """
    import fitz

    table = NORMALIZATION_TABLES.get(era)
    if not table:
        return Path(pdf_path)
    # char→char table to codepoint→codepoint (single-char entries only)
    remap = {ord(k): ord(v) for k, v in table.items() if len(k) == 1 and len(v) == 1}

    name_remap = _GLYPH_NAME_REMAP.get(era, {})
    name_re = (
        re.compile(r"/(" + "|".join(map(re.escape, name_remap)) + r")(?=[\s/\]>])")
        if name_remap else None
    )

    doc = fitz.open(str(pdf_path))
    seen: set[int] = set()
    n_fixed = 0

    def _rewrite_differences(font_obj: str, font_xref: int) -> bool:
        """Rewrite hacked glyph names in the font's Encoding /Differences."""
        if name_re is None:
            return False
        m = _ENCODING_REF.search(font_obj)
        if m:  # indirect Encoding object
            enc_xref = int(m.group(1))
            enc_obj = doc.xref_object(enc_xref) or ""
            fixed = name_re.sub(lambda g: "/" + name_remap[g.group(1)], enc_obj)
            if fixed != enc_obj:
                doc.update_object(enc_xref, fixed)
                return True
        elif "/Differences" in font_obj:  # inline Encoding dict
            fixed = name_re.sub(lambda g: "/" + name_remap[g.group(1)], font_obj)
            if fixed != font_obj:
                doc.update_object(font_xref, fixed)
                return True
        return False

    for page in doc:
        for font in page.get_fonts(full=True):
            xref = font[0]
            if xref in seen:
                continue
            seen.add(xref)
            font_obj = doc.xref_object(xref) or ""
            changed = _rewrite_differences(font_obj, xref)

            m = _TOUNICODE_REF.search(font_obj)
            if m:
                tu_xref = int(m.group(1))
                raw = doc.xref_stream(tu_xref)
                if raw is not None:
                    text = raw.decode("latin-1")
                    fixed = _BFCHAR_BLOCK.sub(
                        lambda b: b.group(1) + _rewrite_bfchar_block(b.group(2), remap) + b.group(3),
                        text,
                    )
                    if fixed != text:
                        doc.update_stream(tu_xref, fixed.encode("latin-1"))
                        changed = True
            if changed:
                n_fixed += 1

    if n_fixed == 0:
        doc.close()
        return Path(pdf_path)

    if out_path is None:
        out_path = Path("db/pdf_fixed") / Path(pdf_path).name
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=1, deflate=True)
    doc.close()
    print(
        f"[cmap-fix] {Path(pdf_path).name}: rewrote ToUnicode for {n_fixed} font(s) → {out_path}",
        file=sys.stderr, flush=True,
    )
    return out_path
