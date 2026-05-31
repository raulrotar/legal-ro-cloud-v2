"""Era-specific character normalization for Romanian diacritics."""
import re
from legalro_core.models import Era

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
        'ﬁ': 'Ț',
    },
}

UNIVERSAL_FIXES: dict[str, str] = {
    'ş': 'ș', 'ţ': 'ț', 'Ş': 'Ș', 'Ţ': 'Ț',
}


def normalize_text(text: str, era: Era) -> str:
    table = NORMALIZATION_TABLES.get(era, {})
    for old, new in table.items():
        text = text.replace(old, new)
    for old, new in UNIVERSAL_FIXES.items():
        text = text.replace(old, new)
    text = reconstruct_spaced(text)
    return text


def normalize_pages(pages: list[str], era: Era) -> list[str]:
    return [normalize_text(p, era) for p in pages]


def normalize_for_search(text: str) -> str:
    """Lowercase + strip diacritics for BM25 text_normalized field."""
    import unicodedata
    lowered = text.lower()
    nfkd = unicodedata.normalize('NFKD', lowered)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))
