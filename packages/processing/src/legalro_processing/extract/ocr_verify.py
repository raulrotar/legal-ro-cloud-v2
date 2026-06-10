"""Verification harness for VLM-OCR output (scanned era).

VLM OCR (glm-ocr via Ollama) has two silent failure modes observed on the
1989 gazettes:
  1. Omission — generation aborts early ("token repeat limit", ollama#14117)
     or the image is clipped (>2048 px, ollama#14114), dropping most of a page.
  2. Repetition — the model loops, emitting the same block 3+ times.

Tesseract (`-l ron`) is the oracle: it produces noisy text on hard spots but
NEVER silently drops half a page and never hallucinates whole passages.  A VLM
page that captures far fewer words than Tesseract, or fails to contain most of
Tesseract's confident words, has lost content and must be retried/flagged.

Pure functions here; the glm-ocr loop in md_extractor.py calls verify_page().
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass, field


# ── Text normalization for fuzzy comparison ──────────────────────────────────

def _fold(text: str) -> str:
    """Lowercase + strip diacritics so Tesseract/VLM diacritic disagreements
    (ş vs ș, OCR'd ã vs ă) don't count as mismatches."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_WORD_RE = re.compile(r"[a-zăâîșțşţ]{4,}", re.IGNORECASE)


def _confident_words(text: str) -> list[str]:
    """Alphabetic words of 4+ chars — short tokens and digits are too noisy
    to serve as oracle evidence."""
    return _WORD_RE.findall(_fold(text))


# ── Tesseract oracle ─────────────────────────────────────────────────────────

def tesseract_page_text(png_bytes: bytes, lang: str = "ron", psm: int = 1,
                        timeout: int = 120) -> str:
    """OCR a rendered page image with Tesseract. Returns plain text.

    psm 1 = automatic page segmentation with OSD (handles two-column layouts).
    Raises FileNotFoundError if tesseract is not installed (caller decides
    whether verification is mandatory).
    """
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", lang, "--psm", str(psm)],
        input=png_bytes,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout.decode("utf-8", errors="replace")


# ── Repetition-loop detection ────────────────────────────────────────────────

def repeated_shingle(
    text: str,
    size: int = 40,
    min_repeats: int = 3,
    min_period: int = 20,
    max_period: int = 2000,
) -> str | None:
    """Detect a TANDEM repetition loop: the same block repeated back-to-back
    >= min_repeats times.  Returns the start of the repeated unit, or None.

    Gazette pages legitimately repeat formulaic blocks (the presidential
    signature appears after every decree), but those copies are separated by
    unique decree bodies — a simple occurrence count would false-positive.
    A VLM loop emits copies ADJACENTLY, i.e. text has period L over a span of
    >= min_repeats × L chars; that is what we test for.

    Whitespace is collapsed first so reflowed duplicates still match.  The
    repeated unit must contain >= 15 letters, so dot leaders and numeric
    table columns don't trigger it.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    n = len(flat)
    if n < min_period * min_repeats:
        return None
    seen: dict[str, int] = {}
    for i in range(n - size + 1):
        sh = flat[i:i + size]
        p = seen.setdefault(sh, i)
        if p == i:
            continue
        period = i - p
        if period < min_period or period > max_period:
            continue
        # extend the matching span: flat has period `period` from p to j
        j = p
        while j + period < n and flat[j + period] == flat[j]:
            j += 1
        # 0.9 factor: tolerates a stripped trailing char and loops whose last
        # copy was cut off mid-emission (common when generation aborts)
        if j - p >= (min_repeats - 1) * period * 0.9:
            unit = flat[p:p + period]
            if sum(c.isalpha() for c in unit) >= 15:
                return unit[:size]
    return None


def collapse_repeated_line_blocks(
    text: str,
    max_block: int = 8,
    min_letters: int = 15,
) -> str:
    """Drop ADJACENT duplicate line blocks (VLM emission loops).

    Comparison is whitespace-, case- and diacritic-insensitive so reflowed or
    SHOUTING-case copies still collapse.  Only blocks with >= min_letters are
    deduped, so dot leaders / blank lines / numeric rows are never touched.
    The first copy is kept verbatim.
    """
    def _norm(line: str) -> str:
        return re.sub(r"\s+", "", _fold(line))

    lines = text.splitlines()
    changed = True
    while changed:
        changed = False
        for b in range(max_block, 0, -1):
            i = 0
            while i + 2 * b <= len(lines):
                a = [_norm(x) for x in lines[i:i + b]]
                if a == [_norm(x) for x in lines[i + b:i + 2 * b]] and \
                        sum(c.isalpha() for c in "".join(a)) >= min_letters:
                    del lines[i + b:i + 2 * b]
                    changed = True
                else:
                    i += 1
    return "\n".join(lines)


# ── Coverage gate ────────────────────────────────────────────────────────────

@dataclass
class PageVerification:
    page_index: int
    vlm_words: int = 0
    oracle_words: int = 0
    word_ratio: float = 1.0       # vlm_words / oracle_words
    coverage: float = 1.0         # fraction of oracle confident words present in VLM text
    repeated: str | None = None   # detected repetition shingle, if any
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "page": self.page_index + 1,
            "vlm_words": self.vlm_words,
            "oracle_words": self.oracle_words,
            "word_ratio": round(self.word_ratio, 3),
            "coverage": round(self.coverage, 3),
            "repeated": self.repeated,
            "passed": self.passed,
            "reasons": self.reasons,
        }


def verify_page(
    vlm_text: str,
    oracle_text: str,
    page_index: int = 0,
    *,
    min_word_ratio: float = 0.75,
    max_word_ratio: float = 2.5,
    min_coverage: float = 0.85,
    blank_oracle_words: int = 12,
) -> PageVerification:
    """Gate a VLM-OCR'd page against the Tesseract oracle text.

    Fails when the VLM produced far fewer words than the oracle saw, when
    most of the oracle's confident words are absent from the VLM output, or
    when the VLM output contains a repetition loop.

    A near-blank oracle page (< blank_oracle_words) passes automatically —
    back covers and separator pages legitimately have almost no text.
    """
    v = PageVerification(page_index=page_index)
    oracle_words = _confident_words(oracle_text)
    vlm_folded = _fold(vlm_text)
    v.vlm_words = len(_confident_words(vlm_text))
    v.oracle_words = len(oracle_words)

    v.repeated = repeated_shingle(vlm_text)
    if v.repeated:
        v.passed = False
        v.reasons.append(f"repetition loop: {v.repeated[:40]!r}")

    if v.oracle_words < blank_oracle_words:
        return v  # blank-ish page; only the repetition check applies

    v.word_ratio = v.vlm_words / v.oracle_words
    if v.word_ratio < min_word_ratio:
        v.passed = False
        v.reasons.append(f"word ratio {v.word_ratio:.2f} < {min_word_ratio}")
    elif v.word_ratio > max_word_ratio:
        # far MORE words than the oracle saw — emission loop / hallucination
        v.passed = False
        v.reasons.append(f"word ratio {v.word_ratio:.2f} > {max_word_ratio} (over-emission)")

    found = sum(1 for w in oracle_words if w in vlm_folded)
    v.coverage = found / len(oracle_words)
    if v.coverage < min_coverage:
        v.passed = False
        v.reasons.append(f"oracle word coverage {v.coverage:.2f} < {min_coverage}")

    return v
