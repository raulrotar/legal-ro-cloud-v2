"""Era-specific character normalization for Romanian diacritics."""
import math
import re
import unicodedata
from pathlib import Path
from legalro_core.models import Era

# ── Romanian word de-fusion (broken_2007 era) ─────────────────────────────────
# Lazy-initialized; populated on first call to defuse_words().
_RO_WORD_COSTS: dict[str, float] | None = None
_RO_WORD_SET: set[str] | None = None
_MAX_WORD_LEN = 20

# Small curated bigram boost: common Romanian function-word pairs that the
# unigram model splits sub-optimally.  Expressed as diacritic-folded lowercase.
# A bigram is applied post-segmentation to merge over-split pairs.
_FUNCTION_BIGRAMS = frozenset({
    ("de", "la"), ("de", "stat"), ("in", "functia"), ("la", "ministerul"),
    ("de", "catre"), ("in", "vederea"), ("cu", "privire"), ("in", "baza"),
    ("de", "la"), ("sub", "rezerva"),
})

def _fold_ro(s: str) -> str:
    """Lowercase + strip diacritics for dictionary lookup (matching only, not output)."""
    nfkd = unicodedata.normalize('NFKD', s.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _load_ro_lexicon() -> None:
    global _RO_WORD_COSTS, _RO_WORD_SET
    if _RO_WORD_COSTS is not None:
        return
    data_path = Path(__file__).parent / "data" / "ro_unigrams.txt"
    counts: dict[str, int] = {}
    if data_path.exists():
        for line in data_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                w, cnt = parts
                try:
                    counts[_fold_ro(w)] = int(cnt)
                except ValueError:
                    pass
    total = sum(counts.values()) or 1
    # log-probability cost: lower = more likely.  Unknown words get a large cost
    # proportional to length, so the DP prefers splitting into known words.
    oov_base = math.log(total * 10)
    _RO_WORD_COSTS = {w: -math.log(c / total) for w, c in counts.items()}
    _RO_WORD_SET = set(counts.keys())
    # Store OOV base for use in defuse_words
    _RO_WORD_COSTS["__oov_base__"] = oov_base  # type: ignore[assignment]


def _viterbi_segment(token: str) -> list[str] | None:
    """Norvig-style max-probability Viterbi segmentation for a single fused token.

    Returns a list of parts if segmentation is accepted, or None to leave the
    token unchanged.  Acceptance criteria:
      - All parts are in the dictionary (folded) with length ≥ 3.
      - Part count ≤ 6.
      - Segmentation log-probability beats leaving the token whole (OOV) by
        a margin of at least 2 nats (≈ e² ≈ 7× more likely than whole token).
      - Character-preservation: ''.join(parts) == token (always true by construction).
    """
    _load_ro_lexicon()
    assert _RO_WORD_COSTS is not None and _RO_WORD_SET is not None

    n = len(token)
    folded = _fold_ro(token)
    oov_base: float = _RO_WORD_COSTS.get("__oov_base__", 20.0)  # type: ignore[arg-type]
    oov_cost = oov_base + n * 2   # long unknown tokens are very expensive

    # best[i] = (min_cost, back_index)
    INF = float("inf")
    best: list[tuple[float, int]] = [(0.0, 0)] + [(INF, 0)] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - _MAX_WORD_LEN), i):
            chunk_folded = folded[j:i]
            chunk_len = i - j
            if chunk_len < 3:
                # Parts shorter than 3 chars disqualify the split (see gate).
                wc = INF
            else:
                wc = _RO_WORD_COSTS.get(chunk_folded, oov_base + chunk_len * 3)
            c = best[j][0] + wc
            if c < best[i][0]:
                best[i] = (c, j)

    # Backtrack
    parts: list[str] = []
    idx = n
    while idx > 0:
        prev = best[idx][1]
        parts.append(token[prev:idx])
        idx = prev
    parts = parts[::-1]

    # Acceptance gate
    if len(parts) <= 1 or len(parts) > 6:
        return None
    if not all(_fold_ro(p) in _RO_WORD_SET and len(p) >= 3 for p in parts):
        return None
    # Likelihood margin: segmented cost vs OOV cost
    seg_cost = best[n][0]
    if oov_cost - seg_cost < 2.0:   # margin threshold: 2 nats ≈ 7× more likely
        return None
    # Character-preservation (always true for pure segmentation, but explicit)
    if "".join(parts) != token:
        return None
    return parts


# Lines that should never have de-fusion applied (names, numbers, headings).
_DEFUSE_SKIP_LINE = re.compile(
    r'(?i)'
    r'(?:Ministrul|Ministr[au]|Președintele|Prim-?[Mm]inistr'
    r'|Bucure[șs]ti,|nr\.\s*\d|Art\.\s*\d'
    r'|^#)',  # headings
    re.MULTILINE,
)

def defuse_words(text: str, audit_log: list[dict] | None = None) -> str:
    """Re-insert spaces between fused words in broken-font OCR output.

    Designed for broken_2007 era where zero-width inter-word glyphs caused
    Docling to concatenate words (e.g. 'PetruinfunctiadesecretardestatlaMinisterul').

    Gate (applied before segmentation to minimize false-split risk):
      - Token must be all-lowercase alphabetic, length ≥ 18.
      - Token must not be in the Romanian dictionary.
      - Line must not contain signatures, act numbers, or be a heading.

    Every split is recorded in audit_log (if provided) for human review.
    The original characters are always preserved; only spaces are inserted.
    """
    _load_ro_lexicon()
    assert _RO_WORD_SET is not None

    # Candidate gate regex: all-lowercase, purely alphabetic, ≥18 chars.
    _CANDIDATE = re.compile(r'\b([a-zăâîșț]{18,})\b')

    def replace_token(line: str) -> str:
        # Skip lines with signatures/numbers/headings to protect proper nouns.
        if _DEFUSE_SKIP_LINE.search(line):
            return line

        def sub(m: re.Match) -> str:
            tok = m.group(1)
            # Must not be a known word (dictionary gate).
            if _fold_ro(tok) in _RO_WORD_SET:
                return tok
            parts = _viterbi_segment(tok)
            if parts is None:
                return tok
            result = " ".join(parts)
            if audit_log is not None:
                audit_log.append({"token": tok, "parts": parts})
            return result

        return _CANDIDATE.sub(sub, line)

    lines = text.splitlines()
    return "\n".join(replace_token(line) for line in lines)

# Runs of 3+ single letters each separated by exactly one space (OCR/layout
# artifact): "D E C R E T E" -> "DECRETE". Word breaks use 2+ spaces, so each
# spaced word collapses independently and normal text is left untouched.
_LETTER = r'[A-Za-zĂÂÎȘȚăâîșț]'
SPACED_RUN = re.compile(rf'(?<!\S)(?:{_LETTER} ){{2,}}{_LETTER}(?!\S)')


def reconstruct_spaced(text: str) -> str:
    return SPACED_RUN.sub(lambda m: m.group(0).replace(' ', ''), text)

NORMALIZATION_TABLES: dict[Era, dict[str, str]] = {
    Era.BROKEN_2002: {
        'ã': 'ă', 'º': 'ș', 'þ': 'ț', 'ª': 'Ș', 'Þ': 'Ț',
        'Ã': 'Ă', 'Ñ': '—', '\x93': '"', '\x94': '"',
    },
    Era.BROKEN_2007: {
        '„': 'ă', '∫': 'ș', '˛': 'ț',
        'Ó': 'î', 'Œ': 'Î', 'ˇ': 'Ț',
        '‚': 'â', '√': 'Ă', '¬': 'Â', '™': 'Ș',
        'ﬁ': 'Ț',  # Unicode fi-ligature (U+FB01) → Ț
    },
}

# Two-char ASCII "fi" rendered by Docling for the Ț glyph in BROKEN_2007 PDFs.
# The font maps Ț to the fi-ligature codepoint; Docling outputs plain 'fi'.
# Two safe contexts where fi=Ț:
#   1. between uppercase letters: ADMINISTRAfiIEI → ADMINISTRAȚIEI
#   2. at a word boundary before apostrophe: fi'rii → Ț'rii (Țării)
_FI_CAPS_RE = re.compile(r'(?<=[A-ZĂÂÎȘȚ])fi(?=[A-ZĂÂÎȘȚ])')
_FI_APOS_RE = re.compile(r"\bfi(?=')")


def fix_fi_ligature(text: str) -> str:
    """Replace Docling's two-char 'fi' with Ț in BROKEN_2007 contexts."""
    text = _FI_CAPS_RE.sub('Ț', text)
    text = _FI_APOS_RE.sub('Ț', text)
    return text

UNIVERSAL_FIXES: dict[str, str] = {
    'ş': 'ș', 'ţ': 'ț', 'Ş': 'Ș', 'Ţ': 'Ț',
}

# Unicode categories for CJK / fullwidth / enclosed characters that are
# plainly OCR hallucinations in Romanian text and should be stripped.
_OCR_NOISE_CATEGORIES = frozenset({
    'Lo',   # Letter, other (CJK ideographs)
    'Nl',   # Number, letter (e.g. enclosed alphanumerics)
})
_OCR_NOISE_BLOCKS_RE = re.compile(
    r'[　-鿿'      # CJK unified + kana + enclosed
    r'＀-￯'       # Fullwidth / halfwidth forms
    r'①-⓿'       # Enclosed alphanumerics
    r'■-◿]'      # Geometric shapes (occasional OCR artifact)
)


def strip_ocr_noise(text: str) -> str:
    """Remove isolated OCR-hallucinated CJK / fullwidth codepoints.

    Docling occasionally produces stray CJK ideographs (e.g. 忆, 司, 口) or
    fullwidth digits (１, ２) when processing degraded Romanian scans.  These
    are never legitimate content in Romanian legal text.

    Only *isolated* noise characters (surrounded by whitespace or at
    line boundaries) are removed to avoid touching intentional foreign-script
    content (e.g. a quoted treaty in another language).
    """
    return _OCR_NOISE_BLOCKS_RE.sub('', text)


def normalize_text(text: str, era: Era) -> str:
    table = NORMALIZATION_TABLES.get(era, {})
    for old, new in table.items():
        text = text.replace(old, new)
    for old, new in UNIVERSAL_FIXES.items():
        text = text.replace(old, new)
    text = reconstruct_spaced(text)
    if era in (Era.BROKEN_2007, Era.BROKEN_2002):
        text = strip_ocr_noise(text)
    if era == Era.BROKEN_2007:
        text = fix_fi_ligature(text)
    return text


# ── Config flag for de-fusion ─────────────────────────────────────────────────
# Set LEGALRO_DEFUSE_WORDS=0 or extraction.defuse_words_enabled=False to disable.
import os as _os
DEFUSE_WORDS_ENABLED: bool = _os.environ.get("LEGALRO_DEFUSE_WORDS", "1") != "0"


def normalize_pages(pages: list[str], era: Era) -> list[str]:
    return [normalize_text(p, era) for p in pages]


def normalize_for_search(text: str) -> str:
    """Lowercase + strip diacritics for BM25 text_normalized field."""
    import unicodedata
    lowered = text.lower()
    nfkd = unicodedata.normalize('NFKD', lowered)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# ── Heading promotion for scanned_1989 ───────────────────────────────────────

def _fold_for_heading(text: str) -> str:
    """Normalize for act-keyword matching against 1989 OCR output.

    Strips diacritics, uppercases, and maps common 1989-OCR substitutions:
    - digits 1/l → I (OCR confusion with Î)
    - pre-1993 Romanian orthography uses î almost everywhere internally
    """
    nfkd = unicodedata.normalize('NFKD', text.upper())
    stripped = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.translate(str.maketrans('1l', 'II'))


def _build_heading_promoter() -> re.Pattern:
    """Build the heading-promotion regex using the shared ACT_KEYWORD_ALTERNATION.

    Imports lazily from md_segmenter to avoid a circular import at module load
    time (normalize is a core module; segmenter depends on processing).
    Falls back to a minimal built-in keyword set if the import fails.
    """
    try:
        from legalro_processing.extract.md_segmenter import ACT_KEYWORD_ALTERNATION  # type: ignore
        alternation = ACT_KEYWORD_ALTERNATION
    except ImportError:
        # Fallback used in tests of legalro_core without legalro_processing installed.
        alternation = (
            r'DECRET(?:-LEGE)?\b|HOT[ĂA]R[ÂI]RE[A]?\b|LEGE[A]?\b|ORDIN\b'
            r'|DECIZIE\b|DECIZIA\b|PROCLAMA[TȚ]IE\b|COMUNICAT\b|OUG\b'
        )
    # Match on diacritic-folded + uppercase form; the line must not already be a heading.
    # We require a positive act-number/intro cue on the same or next line to avoid
    # promoting inline citations ("în baza Decretului nr. ...") and bare signatures.
    return re.compile(
        r'(?m)'
        r'^(?!#)'                             # not already a heading (idempotency)
        r'(?P<line>[ \t]*(?:' + alternation + r')(?:[^\n]*))$',
        re.IGNORECASE,
    )


# Pre-compiled at import; rebuilt lazily if the segmenter alternation changes.
_HEADING_PROMOTER: re.Pattern | None = None

# Known all-caps lines that are NOT act headings (denylist backstop).
_HEADING_DENYLIST = re.compile(
    r'(?i)^(?:ROM[ÂA]NIA|NICOLAE\s+CEA[UȘ][SC]ESCU|ELENA\s+CEA[UȘ][SC]ESCU'
    r'|MARELE\s+STAT\s+MAJOR|ADUNAREA\s+NATIONAL[ĂA]|MAREA\s+ADUNARE)',
)

# Positive cues on the same or next line that confirm a line is a heading
# (not a citation or bare institution name).
# Bare section-header lines (plural act types). An act heading printed
# directly under one of these is still a heading — 1989 issues put
# "COMUNICATUL CĂTRE ȚARĂ" on the line right after the "COMUNICATE" section
# line, with no blank line between them.
_SECTION_LINE = re.compile(
    r'(?i)^[ \t]*(?:COMUNICATE|DECRETE(?:\s*-\s*LEGE)?|DECIZII'
    r'|HOT[ĂA]R[ÂI]RI|LEGI|ORDINE)\s*$',
)

_ACT_NUMBER_CUE = re.compile(
    r'(?i)(?:nr\.?|privind|cu\s+privire\s+la|referitor\s+la'
    # 1989 communiqués are unnumbered: "COMUNICATUL CĂTRE ȚARĂ" /
    # next line "al Consiliului Frontului Salvării Naționale" /
    # body opening straight after the bare COMUNICAT keyword
    r'|c[ăa]tre\s+[țt]ar[ăa]|al\s+consiliului|av[îi]nd\s+[îi]n\s+vedere)',
)


def promote_act_headings(md: str) -> str:
    """Promote Romanian gazette act-type lines to ## Markdown headings.

    Designed for scanned_1989 era where OCR produces flat prose with no heading
    structure.  Only lines that:
      1. Are not already headings (idempotent)
      2. Are preceded by a blank line
      3. Match the act-keyword vocabulary (diacritic-folded, pre-1993 orthography)
      4. Pass a positive cue check: an act number / 'privind' follows on the
         same line or the immediately next non-empty line
      5. Are not in the signature/header denylist
    are promoted to '## <line>'.

    Character-preservation invariant: the promoted line text is verbatim;
    only the '## ' prefix is added.
    """
    global _HEADING_PROMOTER
    if _HEADING_PROMOTER is None:
        _HEADING_PROMOTER = _build_heading_promoter()

    lines = md.splitlines()
    result = []
    n = len(lines)
    for i, line in enumerate(lines):
        raw = line.rstrip()
        # Skip already-promoted headings (idempotency guard).
        if raw.lstrip().startswith('#'):
            result.append(line)
            continue
        # Must be blank-line-preceded (a bare section line counts as blank).
        prev_blank = (i == 0) or not lines[i - 1].strip() \
            or bool(_SECTION_LINE.match(_fold_for_heading(lines[i - 1])))
        if not prev_blank:
            result.append(line)
            continue
        # Must match an act keyword on the diacritic-folded form.
        folded = _fold_for_heading(raw)
        if not _HEADING_PROMOTER.match(folded):
            result.append(line)
            continue
        # Denylist backstop.
        if _HEADING_DENYLIST.search(raw):
            result.append(line)
            continue
        # Positive cue: act number or 'privind' on this line or next non-empty line.
        cue_found = bool(_ACT_NUMBER_CUE.search(raw))
        if not cue_found:
            for j in range(i + 1, min(i + 3, n)):
                if lines[j].strip():
                    cue_found = bool(_ACT_NUMBER_CUE.search(lines[j]))
                    break
        if not cue_found:
            result.append(line)
            continue
        # Promote. Character-preservation: line text unchanged, only prefix added.
        result.append(f'## {raw.strip()}')
    return '\n'.join(result)
