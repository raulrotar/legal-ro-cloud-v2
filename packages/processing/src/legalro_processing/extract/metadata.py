"""Extract structured metadata from act text — strong per-act classification."""
import re
from legalro_processing.extract.segment import RawAct

try:
    from legalro_processing.extract.roles import strip_letterspacing, DATE_RE, MONTHS_RO
except ImportError:
    def strip_letterspacing(text): return text  # type: ignore[misc]
    DATE_RE = None
    MONTHS_RO = {}

# Act type headers — more specific types first, matched only in the first 300 chars.
ACT_TYPE_HEADERS = [
    # Plural/articulated/OCR-split forms: 1989 gazettes print DECRETE-LEGE,
    # COMUNICATUL CĂTRE ȚARĂ, and glm-ocr sometimes splits "DE CRET"
    ("DECRET_LEGE", re.compile(r'^\s*DE\s?CRETE?\s*-\s*LEGE\b', re.IGNORECASE | re.MULTILINE)),
    ("OUG",         re.compile(r'^\s*ORDONAN[ȚTŢ][ĂA]\s+DE\s+URGEN[ȚTŢ][ĂA]\b', re.IGNORECASE | re.MULTILINE)),
    ("ORDONANȚĂ",   re.compile(r'^\s*ORDONAN[ȚTŢ][ĂA]\b', re.IGNORECASE | re.MULTILINE)),
    ("HG",          re.compile(r'^\s*HOT[ĂA]R[ÂA]RE\b', re.IGNORECASE | re.MULTILINE)),
    ("DECRET",      re.compile(r'^\s*DE\s?CRETE?\b', re.IGNORECASE | re.MULTILINE)),
    # DCC only when CCR is the issuing body; other DECIZIE acts are PM/agency decisions.
    ("DCC",         re.compile(r'CURTEA\s+CONSTITU[ȚT]IONAL[ĂA]', re.IGNORECASE)),
    ("DECIZIE",     re.compile(r'^\s*DECIZ(?:IE|IA?)\b', re.IGNORECASE | re.MULTILINE)),
    ("ORDIN",       re.compile(r'^\s*ORDIN(?:UL)?\b', re.IGNORECASE | re.MULTILINE)),
    ("LEGE",        re.compile(r'^\s*LEGEA?\b', re.IGNORECASE | re.MULTILINE)),
    ("COMUNICAT",   re.compile(r'^\s*COMUNICAT(?:UL|E)?\b', re.IGNORECASE | re.MULTILINE)),
    ("RAPORT",      re.compile(r'^\s*R\s*A\s*P\s*O\s*R\s*T\b', re.IGNORECASE | re.MULTILINE)),
    ("RECTIFICARE", re.compile(r'^\s*RECTIFIC[ĂA]RI?\b', re.IGNORECASE | re.MULTILINE)),
]

# Closing signature block — the act's own date and number.
# Allows up to 100 chars (incl. newlines) between the date and Nr. line
# to handle both single and double newline separators in OCR output.
CLOSING_BLOCK = re.compile(
    r'Bucure[șs]ti,\s+\d{1,2}\s+\w+\s+(\d{4})\.[\s\S]{0,100}?Nr\.\s*([\d.]+)\.',
)

# Fallback: bare Nr. match (for very old acts without standard closing).
BARE_NR = re.compile(r'\bNr\.\s*([\d.]+)')

# Patterns indicating a referenced act number inside an abrogation or citation
# clause — NOT the number of the act being extracted.
# E.g. "se abrogă Ordinul … nr. 275/2003" or "Ordinul … nr. X din YYYY se abrogă"
_ABROGATION_RE = re.compile(
    r'(?:se\s+abroga|abroga|abrogat[ăa]?|se\s+modifica|modifica)\s'
    r'|(?:Ordinul|Hotararea|Legea|Decizia|Decretul)\s+.*?[Nn]r\.',
    re.IGNORECASE,
)

# Authority patterns: (display_name, short_tag, regex).
# Order matters — most specific first.
AUTHORITY_PATTERNS = [
    ("Curtea Constituțională",
     "ccr",
     re.compile(r'CURTEA\s+CONSTITU[ȚT]IONAL[ĂA]', re.IGNORECASE)),

    ("Agenția Națională de Cadastru și Publicitate Imobiliară",
     "ancpi",
     # Match both nominative "Agenția" and genitive "Agenției"
     re.compile(r'Agen[țt]i(?:a|ei)\s+Na[țt]ional[ăa]\s+de\s+Cadastru', re.IGNORECASE)),

    ("Agenția Națională de Control al Exporturilor",
     "ance",
     re.compile(r'Agen[țt]ia\s+Na[țt]ional[ăa]\s+de\s+Control\s+al\s+Export', re.IGNORECASE)),

    ("Agenția Națională pentru Resurse Minerale",
     "anrm",
     re.compile(r'Agen[țt]ia\s+Na[țt]ional[ăa]\s+pentru\s+Resurse\s+Minerale', re.IGNORECASE)),

    ("Autoritatea Electorală Permanentă",
     "aep",
     re.compile(r'AUTORITATEA\s+ELECTORAL[ĂA]\s+PERMANENT[ĂA]', re.IGNORECASE)),

    ("Autoritatea Națională Sanitară Veterinară și pentru Siguranța Alimentelor",
     "ansvsa",
     re.compile(r'Autorit[aă]tea\s+Na[țt]ional[ăa]\s+Sanitar[ăa]\s+Veterinar[ăa]', re.IGNORECASE)),

    ("Banca Națională a României",
     "bnr",
     re.compile(r'BANCA\s+NA[ȚT]IONAL[ĂA]\s+A\s+ROM[ÂA]NIEI', re.IGNORECASE)),

    ("Parlamentul României",
     "parl",
     # Exclude the newspaper footer "PARLAMENTUL ROMÂNIEI — CAMERA DEPUTAȚILOR"
     re.compile(r'PARLAMENTUL\s+ROM[ÂA]NIEI(?!\s*[—–-]\s*CAMERA)', re.IGNORECASE)),

    ("Președintele României",
     "pres",
     re.compile(r'PRE[ȘS]EDINTELE\s+ROM[ÂA]NIEI', re.IGNORECASE)),

    ("Guvernul României",
     "gov",
     # Match only when GUVERNUL appears as an issuing authority (standalone header),
     # not inside a reference like "Hotărârea Guvernului nr. 1288/2012".
     re.compile(r'^GUVERNUL\s+ROM[ÂA]NIEI', re.IGNORECASE | re.MULTILINE)),

    ("Ministerul Internelor",
     "mai",
     re.compile(r'ministr\w+\s+internelor', re.IGNORECASE)),

    ("Ministerul Educației",
     "medu",
     re.compile(r'ministr\w+\s+educa[țt]iei', re.IGNORECASE)),

    ("Ministerul Finanțelor",
     "mfin",
     re.compile(r'ministr\w+\s+finan[țt]elor', re.IGNORECASE)),

    ("Ministerul Sănătății",
     "msan",
     re.compile(r'ministr\w+\s+s[ăa]n[ăa]t[ăa][țt]ii', re.IGNORECASE)),

    ("Minister",
     "min",
     re.compile(r'\bMINISTER(?:UL)?\b', re.IGNORECASE)),
]

# County/locality — used by ANCPI orders and similar.
COUNTY_PATTERN = re.compile(
    r'jude[țt]ul[ui]?\s+([A-ZĂÂÎȘȚ][a-zăâîșț-]{2,}(?:\s+[A-ZĂÂÎȘȚ][a-zăâîșț-]+)?)',
    re.IGNORECASE,
)


def extract_metadata(raw_act: RawAct, gazette_year: int) -> dict:
    text = raw_act.text
    header = text[:800]
    # Apply strip_letterspacing line-by-line: whole-header normalization fails because
    # normal body-text lines break the "all tokens ≤ 2 chars" single-char heuristic.
    header_norm = "\n".join(strip_letterspacing(line) for line in header.split("\n"))

    # --- Document type ---
    doc_type = "UNKNOWN"
    earliest = len(header) + 1
    for atype, pattern in ACT_TYPE_HEADERS:
        for search_target in (header, header_norm):
            m = pattern.search(search_target)
            if m and m.start() < earliest:
                earliest = m.start()
                doc_type = atype
    # ANCPI orders start with the agency name, not "ORDIN" — infer from authority
    if doc_type == "UNKNOWN" and "CADASTRU" in header[:400]:
        doc_type = "ORDIN"

    # --- Number + year from closing signature ---
    act_number, act_year = _extract_number_and_year(text, gazette_year)

    # --- Issuing authority ---
    # Search the whole text: authority may appear in the closing signature
    # (e.g. "Directorul general al ANCPI") rather than the opening header.
    search_zone = text
    authority_name = ""
    authority_tag = ""
    for name, tag, pattern in AUTHORITY_PATTERNS:
        if pattern.search(search_zone):
            authority_name = name
            authority_tag = tag
            break

    # --- Locality (county) ---
    locality = _extract_locality(text)

    # --- Title ---
    title = raw_act.title or _extract_title(text, doc_type)

    # --- law_id ---
    type_slug = doc_type.lower().replace("_", "_")
    if authority_tag:
        law_id = f"{type_slug}_{authority_tag}_{act_number}_{act_year}_v1"
    else:
        law_id = f"{type_slug}_{act_number}_{act_year}_v1"

    return {
        "doc_type": doc_type,
        "act_number": act_number,
        "act_year": act_year,
        "issuing_authority": authority_name,
        "authority_tag": authority_tag,
        "locality": locality,
        "title": title,
        "law_id": law_id,
    }


def _extract_number_and_year(text: str, gazette_year: int) -> tuple[str, int]:
    """Extract act number and ACTUAL signing year (not publication year).

    Search for the closing signature block 'București, DD YYYY. Nr. NNN.'
    The year here is the signing year, which may differ from gazette_year
    (e.g. act signed Dec 2006, published Jan 2007).
    """
    matches = list(CLOSING_BLOCK.finditer(text))
    if matches:
        # Use the LAST closing block (v1 behaviour). An act's own signature is
        # always the final "București, DD YYYY. Nr. NNN." in its text. Earlier
        # matches belong to the tail of a prior act that bled in before
        # _split_by_closing separated it. Annexes that were already split off
        # have no closing block at all, so this path isn't reached for them.
        m = matches[-1]
        year = int(m.group(1))
        number = m.group(2).replace(".", "")
        return number, year

    # Fallback: try to find a date using DATE_RE in the last 600 chars
    # (where 'București, DD luna YYYY.' typically appears)
    if DATE_RE is not None:
        tail = text[-600:] if len(text) > 600 else text
        date_matches = list(DATE_RE.finditer(tail))
        if date_matches:
            dm = date_matches[-1]
            try:
                year = int(dm.group("year"))
                # Sanity: year must be within a reasonable range
                if 1989 <= year <= gazette_year + 1:
                    # Also try to get the number from nearby text
                    bare = list(BARE_NR.finditer(tail))
                    number = bare[-1].group(1).replace(".", "") if bare else "0"
                    return number, year
            except (ValueError, IndexError):
                pass

    # Last resort: bare Nr. for number, gazette year for year.
    # Skip matches that appear inside an abrogation/reference clause
    # (e.g. "se abrogă Ordinul nr. 275/2003") — those are the numbers of OTHER
    # acts being referenced, not the number of the act being extracted.
    bare = list(BARE_NR.finditer(text))
    for m in reversed(bare):
        # Check a window of 120 chars before this match for abrogation signals
        window_start = max(0, m.start() - 120)
        window = text[window_start:m.start()]
        if _ABROGATION_RE.search(window):
            continue  # skip — this Nr. belongs to a referenced/abrogated act
        number = m.group(1).replace(".", "")
        return number, gazette_year

    # All bare Nr. matches were in abrogation clauses — fall back to "0"
    return "0", gazette_year


def _extract_locality(text: str) -> str:
    """Return the first-mentioned county name, or empty string."""
    # Skip the first 50 chars (act type header) to avoid false matches in titles.
    m = COUNTY_PATTERN.search(text, 50)
    if m:
        county = m.group(1).strip()
        # Exclude generic words that sometimes follow "județului"
        if county.lower() not in {"toate", "fiecare", "același", "același"}:
            return county
    return ""


def _extract_title(text: str, doc_type: str) -> str:
    lines = text.split('\n')
    title_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if doc_type.lower() in stripped.lower() or re.match(r'^nr\.', stripped, re.IGNORECASE):
                started = True
            continue
        if re.match(r'^Art\.\s*\d+', stripped) or re.match(r'^Articolul\s+\d+', stripped):
            break
        if stripped:
            title_lines.append(stripped)
        if len(title_lines) >= 3:
            break
    return " ".join(title_lines)[:300]
