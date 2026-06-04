from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from legalro_processing.extract.blocks import Block, is_full_width, PAGE_WIDTH


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def strip_letterspacing(text: str) -> str:
    """
    Remove letter-spacing.
    'L E G E' → 'LEGE'
    'H O T Ă R Â R E' → 'HOTĂRÂRE'
    'Președintele României d e c r e t e a z ă:' → 'Președintele României decretează:'
    """
    t = text.strip()
    # Case 1: entire string is letter-spaced (all non-empty tokens ≤ 2 chars)
    tokens = [tok for tok in t.split(' ') if tok]
    if len(tokens) >= 3 and all(len(tok) <= 2 for tok in tokens):
        return ''.join(tokens)
    # Case 2: partial letter-spacing — collapse runs of ≥3 consecutive single-char tokens
    # e.g. "d e c r e t e a z ă" → "decretează" within a longer sentence
    return re.sub(
        r'(?<!\S)(\S{1,2})(?:\s(\S{1,2})){2,}(?=\s|$|[,;:.\-])',
        lambda m: m.group(0).replace(' ', ''),
        t,
    )


def collapse_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Legacy encoding repair (M2 eras)
# ---------------------------------------------------------------------------

LEGACY_CODEPOINT_MAP = {
    "Ã": "Ă", "ã": "ă",
    "Ñ": "—",
    "ª": "Ş", "º": "ş",
    "Þ": "Ţ", "þ": "ţ",
    "Ò": "”",
    "Ð": "Đ",
    "√": "Ă", "¬": "Â", "™": "Ş",
}

_REPAIR_GUARD = set("ÃÑªºÞþÒÐ√¬™")


def repair_legacy_encoding(text: str) -> str:
    if not any(ch in text for ch in _REPAIR_GUARD):
        return text
    out = text
    for k, v in sorted(LEGACY_CODEPOINT_MAP.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(k, v)
    return unicodedata.normalize("NFC", out)


# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

ISSUER_TAXONOMY: set[str] = {
    "PARLAMENTUL ROMÂNIEI",
    "CAMERA DEPUTAȚILOR",
    "SENATUL ROMÂNIEI",
    "PREȘEDINTELE ROMÂNIEI",
    "PREŞEDINTELE ROMÂNIEI",
    "GUVERNUL ROMÂNIEI",
    "PRIM-MINISTRUL ROMÂNIEI",
    "CURTEA CONSTITUȚIONALĂ",
    "CURTEA CONSTITUŢIONALĂ",
    "MINISTERUL INTERNELOR ȘI REFORMEI ADMINISTRATIVE",
    "MINISTERUL INTERNELOR",
    "MINISTERUL ADMINISTRAȚIEI ȘI INTERNELOR",
    "MINISTERUL APĂRĂRII NAȚIONALE",
    "MINISTERUL ECONOMIEI",
    "MINISTERUL FINANȚELOR PUBLICE",
    "MINISTERUL FINANŢELOR PUBLICE",
    "MINISTERUL EDUCAȚIEI",
    "MINISTERUL EDUCAŢIEI",
    "MINISTERUL SĂNĂTĂȚII",
    "MINISTERUL SĂNĂTĂŢII PUBLICE",
    "MINISTERUL JUSTIȚIEI",
    "MINISTERUL JUSTIŢIEI",
    "MINISTERUL MUNCII",
    "MINISTERUL TRANSPORTURILOR",
    "MINISTERUL CULTURII",
    "MINISTERUL AGRICULTURII",
    "MINISTERUL MEDIULUI",
    "MINISTERUL ENERGIEI",
    "BANCA NAȚIONALĂ A ROMÂNIEI",
    "BANCA NAŢIONALĂ A ROMÂNIEI",
    "AGENȚIA NAȚIONALĂ DE CADASTRU ȘI PUBLICITATE IMOBILIARĂ",
    "AGENŢIA NAŢIONALĂ DE CADASTRU ŞI PUBLICITATE IMOBILIARĂ",
    "AGENȚIA NAȚIONALĂ PENTRU RESURSE MINERALE",
    "AGENȚIA NAȚIONALĂ DE CONTROL AL EXPORTURILOR",
    "AUTORITATEA NAȚIONALĂ PENTRU REGLEMENTARE ÎN COMUNICAȚII ȘI TEHNOLOGIA INFORMAȚIEI",
    "AUTORITATEA ELECTORALĂ PERMANENTĂ",
    "CONSILIUL NAȚIONAL AL AUDIOVIZUALULUI",
    "ÎNALTA CURTE DE CASAȚIE ȘI JUSTIȚIE",
    "PARCHETUL DE PE LÂNGĂ ÎNALTA CURTE",
}

ACT_TYPE_TAXONOMY: set[str] = {
    "LEGE", "LEGEA",
    "DECRET", "DECRETUL",
    "HOTĂRÂRE", "HOTĂRÂREA",
    "HOTĂRÎRE", "HOTĂRÎREA",
    "ORDIN", "ORDINUL",
    "ORDONANȚĂ", "ORDONANȚA",
    "ORDONANŢĂ", "ORDONANŢA",
    "DECIZIE", "DECIZIA",
    "DECRET-LEGE",
    "REGULAMENT", "REGULAMENTUL",
    "NORMĂ", "NORMA",
    "INSTRUCȚIUNE", "INSTRUCŢIUNE",
    "COMUNICAT",
    "RECTIFICARE",
    "ORDONANȚĂ DE URGENȚĂ",
    "ORDONANŢĂ DE URGENŢĂ",
}

SECTION_BANNERS: dict[str, str] = {
    "lege_decrete":      r"^L\s*E\s*G\s*I\s*(?:Ş|Ș)\s*I\s*D\s*E\s*C\s*R\s*E\s*T\s*E$",
    "decrete":           r"^D\s*E\s*C\s*R\s*E\s*T\s*E$",
    "hg":                r"^H\s*O\s*T\s*[ĂA]\s*R[ÂA]?R?\s*I\s+A\s*L\s*E\s+G\s*U\s*V\s*E\s*R\s*N\s*U\s*L\s*U\s*I",
    "ccr":               r"^D\s*E\s*C\s*I\s*Z\s*I\s*I\s+A\s*L\s*E\s+C\s*U\s*R\s*[ȚT]\s*I\s*I\s+C\s*O\s*N\s*S\s*T\s*I\s*T\s*U\s*[ȚT]\s*I\s*O\s*N\s*A\s*L\s*E",
    "acte_specialitate": r"^ACTE\s+ALE\s+ORGANELOR\s+DE\s+SPECIALITATE",
    "decizii_pm":        r"^D\s*E\s*C\s*I\s*Z\s*I\s*I\s+A\s*L\s*E\s+P\s*R\s*I\s*M\s*-?\s*M\s*I\s*N\s*I\s*S\s*T\s*R\s*U\s*L\s*U\s*I",
    "legi":              r"^L\s*E\s*G\s*I$",
}

_COMPILED_BANNERS: dict[str, re.Pattern] = {
    k: re.compile(v, re.IGNORECASE) for k, v in SECTION_BANNERS.items()
}

OPERATIVE_PHRASES: set[str] = {
    "Parlamentul României adoptă prezenta lege:",
    "Parlamentul României adoptă prezenta lege :",
    "Guvernul României adoptă prezenta hotărâre:",
    "Guvernul României adoptă prezenta hotărâre :",
    "Guvernul României adoptă prezenta ordonanță:",
    "Guvernul României adoptă prezenta ordonanță de urgență:",
    "Președintele României decretează:",
    "Preşedintele României decretează:",
    "adoptă prezenta lege:",
    "adoptă prezenta hotărâre:",
    "decretează:",
    "emite prezentul ordin:",
    "emite prezenta decizie:",
    "decide:",
    "dispune:",
}

SIGNATURE_ROLE_TAXONOMY: set[str] = {
    "PRIM-MINISTRU",
    "PRIM - MINISTRU",
    "VICEPRIM-MINISTRU",
    "MINISTRUL INTERNELOR",
    "MINISTRUL AFACERILOR INTERNE",
    "MINISTRUL ECONOMIEI",
    "MINISTRUL FINANȚELOR PUBLICE",
    "MINISTRUL FINANŢELOR PUBLICE",
    "MINISTRUL EDUCAȚIEI",
    "MINISTRUL EDUCAŢIEI",
    "MINISTRUL SĂNĂTĂȚII",
    "MINISTRUL JUSTIȚIEI",
    "MINISTRUL MUNCII",
    "MINISTRUL TRANSPORTURILOR",
    "MINISTRUL APĂRĂRII NAȚIONALE",
    "MINISTRUL MEDIULUI",
    "MINISTRUL AGRICULTURII",
    "MINISTRUL CULTURII",
    "MINISTRUL ENERGIEI",
    "DIRECTORUL GENERAL",
    "DIRECTORUL GENERAL AL",
    "PREȘEDINTELE",
    "PREŞEDINTELE",
    "GUVERNATORUL BĂNCII NAȚIONALE",
    "GUVERNATORUL BĂNCII NAŢIONALE",
    "Contrasemnează:",
    "Contrasemneaza:",
    "CONTRASEMNEAZĂ:",
}

# ---------------------------------------------------------------------------
# Date parsing constants (imported by metadata.py)
# ---------------------------------------------------------------------------

DATE_RE = re.compile(
    r"(?P<dom>\d{1,2})\s+(?P<mon>"
    r"ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie"
    r")\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

MONTHS_RO = {
    "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4,
    "mai": 5, "iunie": 6, "iulie": 7, "august": 8,
    "septembrie": 9, "octombrie": 10, "noiembrie": 11, "decembrie": 12,
}

# ---------------------------------------------------------------------------
# Context and helpers
# ---------------------------------------------------------------------------

@dataclass
class PageContext:
    page_index: int
    page_w: float
    current_section: str = ""
    last_role: str = ""


def is_centered(block: Block, page_w: float, tol: float = 30.0) -> bool:
    cx = (block.bbox[0] + block.bbox[2]) / 2
    return abs(cx - page_w / 2) < tol


def is_caps_name(text: str) -> bool:
    """All-uppercase 2+ word line — a person name in a signature block."""
    words = text.strip().split()
    if len(words) < 2:
        return False
    return all(
        w.isupper() or w in {"-", "—"} or (len(w) > 1 and w[0].isupper() and w[1:].isupper())
        for w in words if len(w) > 1
    )


_ACT_NUMBER_CCR_RE = re.compile(r"^D\s*E\s*C\s*I\s*Z\s*I\s*A\s+Nr\.", re.IGNORECASE)
_ACT_NUMBER_RE = re.compile(r"^Nr\.\s*\d+\.?$", re.IGNORECASE)
_DATE_SUBLINE_RE = re.compile(r"^din\s+\d{1,2}\s+\w+\s+\d{4}$", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^(Art\.\s*\d+|Articol\s+unic)\.?\s*[—–-]")
_PLACE_DATE_RE = re.compile(r"^(Bucure[șşs]ti|Cluj|Iași|Constanța),\s+\d", re.IGNORECASE)

_CLOSING_STARTS = (
    "Această lege",
    "Această ordonanță",
    "Această hotărâre",
)

_PREAMBLE_STARTS = (
    "În temeiul",
    "Având în vedere",
    "Văzând",
    "Ținând seama",
)

_COVER_SUMAR_RE = re.compile(r"SUMAR", re.IGNORECASE)
_COVER_ISSUE_RE = re.compile(r"Nr\.\s*\d+", re.IGNORECASE)
_COVER_DATE_RE = re.compile(
    r"(luni|marți|miercuri|joi|vineri|sâmbătă|duminică),\s+\d{1,2}\s+\w+\s+\d{4}",
    re.IGNORECASE,
)
_COVER_SUMAR_CHR_RE = re.compile(r"^(Pag\.|Nr\.\s+crt\.)", re.IGNORECASE)
_COVER_SUMAR_SECT_RE = re.compile(r"^[IVX]+\.", re.IGNORECASE)


def _normalize(text: str) -> str:
    return collapse_spaces(strip_letterspacing(text))


def classify_blocks(
    blocks: list[Block],
    page_index: int,
    page_w: float,
) -> list[Block]:
    ctx = PageContext(page_index=page_index, page_w=page_w)

    for b in blocks:
        if b.role not in {"unknown"}:
            ctx.last_role = b.role
            continue

        norm = _normalize(b.text)
        norm_upper = norm.upper()

        # --- Page 0 cover classification ---
        if page_index == 0:
            role = _classify_cover(b, norm, norm_upper, ctx, page_w)
            if role:
                b.role = role
                ctx.last_role = role
                continue

        # --- Section banners ---
        # Section banners may be full-width OR centered (e.g. in short-decree gazettes
        # where the banner fits in the center rather than spanning both columns).
        # IMPORTANT: check raw block text too — the patterns contain \s* between chars
        # to match letter-spaced text, but after strip_letterspacing word spaces are lost
        # (e.g. "HOTĂRÂRIALEGUVERNULUIROMÂNIEI" no longer matches the pattern).
        first_line = b.text.split('\n')[0].strip()
        first_norm = collapse_spaces(strip_letterspacing(first_line))
        if b.font_size >= 12 and (is_full_width(b, page_w) or is_centered(b, page_w, tol=80)):
            for section_key, pat in _COMPILED_BANNERS.items():
                if pat.search(first_norm) or pat.search(norm) or pat.search(b.text):
                    b.role = "section_banner"
                    ctx.current_section = section_key
                    ctx.last_role = b.role
                    break
            if b.role != "unknown":
                continue

        # --- Issuer ---
        # Context-aware: "PREȘEDINTELE ROMÂNIEI" can be either an issuer (act header)
        # or a signature (act footer). Distinguish by ctx.last_role:
        # after body/article/operative roles → it's a signature, not a new issuer.
        _BODY_ROLES = {"article", "preamble", "operative_phrase", "closing_disposition",
                       "signature_role", "signature_name", "signature_contrasemneaza"}
        if 9.0 <= b.font_size <= 11.5 and is_centered(b, page_w):
            if norm_upper in {s.upper() for s in ISSUER_TAXONOMY}:
                if ctx.last_role in _BODY_ROLES:
                    b.role = "signature_role"
                else:
                    b.role = "issuer"
                ctx.last_role = b.role
                continue

        # --- Act type ---
        # Check first line only: act_type and title often share a single fitz block
        # e.g. "D E C R E T\npentru numirea unui judecător"
        first_line_norm_upper = collapse_spaces(strip_letterspacing(first_line)).upper()
        if 11.0 <= b.font_size <= 13.5:
            if first_line_norm_upper in {s.upper() for s in ACT_TYPE_TAXONOMY} \
                    or norm_upper in {s.upper() for s in ACT_TYPE_TAXONOMY}:
                b.role = "act_type"
                ctx.last_role = b.role
                continue

        # --- Act number line ---
        if _ACT_NUMBER_CCR_RE.match(norm):
            b.role = "act_number"
            ctx.last_role = b.role
            continue
        if _ACT_NUMBER_RE.match(norm):
            b.role = "act_act_number"
            ctx.last_role = b.role
            continue

        # --- Date subline ---
        if _DATE_SUBLINE_RE.match(norm):
            b.role = "act_subdate"
            ctx.last_role = b.role
            continue

        # --- Operative phrase ---
        for phrase in OPERATIVE_PHRASES:
            if norm.startswith(phrase) or norm.startswith(phrase.rstrip(":")):
                b.role = "operative_phrase"
                break
        if b.role != "unknown":
            ctx.last_role = b.role
            continue

        # --- Article ---
        if _ARTICLE_RE.match(norm):
            b.role = "article"
            ctx.last_role = b.role
            continue

        # --- Closing disposition ---
        if any(norm.startswith(s) for s in _CLOSING_STARTS):
            b.role = "closing_disposition"
            ctx.last_role = b.role
            continue

        # --- Signatures ---
        norm_stripped = norm.strip().rstrip(":")
        sig_match = False
        for sig in SIGNATURE_ROLE_TAXONOMY:
            if norm.startswith(sig) or norm_stripped.upper() == sig.upper():
                if sig.lower().startswith("contrasemn"):
                    b.role = "signature_contrasemneaza"
                else:
                    b.role = "signature_role"
                sig_match = True
                break
        if sig_match:
            ctx.last_role = b.role
            continue

        if ctx.last_role in {"signature_role", "signature_contrasemneaza", "signature_name"} and is_caps_name(norm):
            b.role = "signature_name"
            ctx.last_role = b.role
            continue

        # --- Place and date ---
        if _PLACE_DATE_RE.match(norm):
            b.role = "place_and_date"
            ctx.last_role = b.role
            continue

        # --- Preamble ---
        if any(norm.startswith(s) for s in _PREAMBLE_STARTS):
            b.role = "preamble"
            ctx.last_role = b.role
            continue

        b.role = "unknown"
        ctx.last_role = b.role

    return blocks


def _classify_cover(
    b: Block,
    norm: str,
    norm_upper: str,
    ctx: PageContext,
    page_w: float,
) -> str:
    if b.font_size >= 18 and "PARTEA" in norm_upper and is_full_width(b, page_w):
        return "cover_partea"
    if _COVER_SUMAR_RE.fullmatch(norm.strip()):
        return "cover_sumar_label"
    if _COVER_ISSUE_RE.search(norm) and b.bbox[1] < PAGE_WIDTH * 0.15:
        return "cover_issue_line"
    if _COVER_DATE_RE.search(norm):
        return "cover_date_line"
    if _COVER_SUMAR_CHR_RE.match(norm):
        return "cover_sumar_colhdr"
    if _COVER_SUMAR_SECT_RE.match(norm):
        return "cover_sumar_section"
    if ctx.last_role in {"cover_sumar_section", "cover_sumar_entry", "cover_sumar_colhdr"}:
        return "cover_sumar_entry"
    return ""
