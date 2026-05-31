"""Detect PDF era via font introspection and text density."""
import re
import fitz
from legalro_core.models import Era


# Mojibake signature characters unique to each broken encoding.
SIG_2007 = set("∫˛√¬™‚ÓŒ")
SIG_2002 = set("ãºþªÞÃ")


def detect_era(pdf_path: str) -> Era:
    doc = fitz.open(pdf_path)
    total_chars = 0
    fonts: set[str] = set()
    text = ""

    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        page_text = page.get_text("text")
        text += page_text
        total_chars += len(page_text.strip())
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        fonts.add(span["font"])
    doc.close()

    if total_chars < 50:
        return Era.SCANNED

    # Content-based detection takes priority over fonts: real files mislabel
    # their era by font (e.g. 2007-Quark mojibake inside SwitzBun-font PDFs).
    sig_2007 = sum(text.count(c) for c in SIG_2007)
    sig_2002 = sum(text.count(c) for c in SIG_2002)
    if sig_2007 >= 5 and sig_2007 >= sig_2002:
        return Era.BROKEN_2007
    if sig_2002 >= 5:
        return Era.BROKEN_2002

    if any("Quark" in f for f in fonts):
        return Era.BROKEN_2007
    if any("SwitzBun" in f for f in fonts):
        return Era.BROKEN_2002
    if any(re.match(r'^[A-Z]{6}\+', f) for f in fonts):
        return Era.HYBRID
    return Era.MODERN
