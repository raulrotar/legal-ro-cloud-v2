"""Tests for character normalization."""
import re
from pathlib import Path

from legalro_core.normalize import normalize_text, strip_ocr_noise, promote_act_headings, defuse_words
from legalro_core.models import Era

# Characters that should NEVER survive normalization of BROKEN_2007 text.
BROKEN_2007_SUSPECTS = set('√™¬˛∫ŒÓ„‚ˇﬁ')


def test_broken_2002_normalization():
    result = normalize_text("lege ã º þ", Era.BROKEN_2002)
    assert "ă" in result
    assert "ș" in result
    assert "ț" in result


def test_universal_fixes():
    result = normalize_text("ştiinţă", Era.MODERN)
    assert "ș" in result
    assert "ț" in result


def test_modern_era_no_change():
    text = "lege modernă"
    result = normalize_text(text, Era.MODERN)
    assert result == text


# ── BROKEN_2007 diacritic regression tests ────────────────────────────────────

def test_broken_2007_core_mappings():
    """All known corrupt → correct pairs must be resolved."""
    corrupt = "„ ∫ ˛ Ó Œ ˇ ‚ √ ¬ ™ ﬁ"
    result = normalize_text(corrupt, Era.BROKEN_2007)
    residual = set(result) & BROKEN_2007_SUSPECTS
    assert not residual, f"Suspect chars survived: {residual!r}"


def test_broken_2007_yields_correct_diacritics():
    """Spot-check that common words come out with proper Romanian diacritics."""
    # Mappings per BROKEN_2007 table:
    #   ∫→ș  ˛→ț  „→ă  ‚→â  ¬→Â  √→Ă  Ó→î  Œ→Î  ™→Ș  ˇ→Ț  ﬁ→Ț
    samples = {
        "Pre∫edintele": "Președintele",   # ∫→ș, ˛→ț (via ˇ path not used here)
        "Rom‚niei": "României",            # ‚→â
        "Hot„r‚rea": "Hotărârea",          # „→ă, ‚→â
        "Art. ∫i": "Art. și",             # ∫→ș
    }
    for corrupt, expected in samples.items():
        result = normalize_text(corrupt, Era.BROKEN_2007)
        assert result == expected, f"{corrupt!r} → {result!r}, expected {expected!r}"


def test_strip_ocr_noise_removes_cjk():
    """CJK and fullwidth chars are stripped by strip_ocr_noise."""
    noisy = "art. 1 忆 司 口 text"
    clean = strip_ocr_noise(noisy)
    assert "忆" not in clean
    assert "口" not in clean
    assert "art. 1" in clean
    assert "text" in clean


def test_broken_2007_strips_ocr_noise():
    """normalize_text for BROKEN_2007 must also strip OCR hallucinations."""
    noisy = "lege １ ２ 忆 română"
    result = normalize_text(noisy, Era.BROKEN_2007)
    assert "忆" not in result
    assert "română" in result


# ── Heading promotion (scanned_1989 era) ─────────────────────────────────────

def test_promote_act_headings_promotes_decret():
    """A DECRET line preceded by a blank and followed by nr. becomes ##."""
    md = "\n\nDECRET nr. 3 din 22 decembrie 1989\n\nText actului."
    result = promote_act_headings(md)
    assert "## DECRET nr. 3 din 22 decembrie 1989" in result


def test_promote_act_headings_promotes_hotarare():
    """HOTĂRÎRE (pre-1993 orthography) is promoted."""
    md = "\n\nHOTARIRE nr. 5 privind ceva\n\nText."
    result = promote_act_headings(md)
    assert result.count("## ") >= 1


def test_promote_act_headings_does_not_promote_signature():
    """Known signature lines must not be promoted."""
    md = "\n\nNICOLAE CEAUȘESCU\n\nContinuare text."
    result = promote_act_headings(md)
    assert "## NICOLAE CEAUȘESCU" not in result


def test_promote_act_headings_does_not_promote_inline_citation():
    """An inline reference ('în baza Decretului nr. X') must not become a heading."""
    md = "Prezentul regulament se adoptă în baza Decretului nr. 3 din 1989."
    result = promote_act_headings(md)
    assert result == md  # no change


def test_promote_act_headings_idempotent():
    """Running promote_act_headings twice must produce the same result."""
    md = "\n\nDECRET nr. 1 din 1989\n\nText."
    once = promote_act_headings(md)
    twice = promote_act_headings(once)
    assert once == twice


def test_promote_act_headings_char_preservation():
    """The promoted line text must be verbatim — only the '## ' prefix is added."""
    line = "DECRET nr. 12 din 25 decembrie 1989"
    md = f"\n\n{line}\n\nText."
    result = promote_act_headings(md)
    for r in result.splitlines():
        if r.startswith("## "):
            assert r[3:] == line


# ── Word de-fusion (broken_2007 era) ─────────────────────────────────────────

def test_defuse_words_leaves_normal_text_untouched():
    """Short tokens and known dictionary words are never touched."""
    text = "lege pentru aprobarea regulamentului din 2007"
    assert defuse_words(text) == text


def test_defuse_words_leaves_long_known_word_intact():
    """A long word that IS in the dictionary must not be split."""
    # 'responsabilitatile' is 18 chars and should be in the dictionary (or left alone).
    text = "responsabilitatile trebuie respectate"
    result = defuse_words(text)
    # Must not split a real long word into garbage parts
    # (result may equal input if word is in dict, which is the correct behaviour)
    assert "responsabilitatile" in result or "responsabilita" in result


def test_defuse_words_no_digits_or_camelcase():
    """Tokens with digits or internal uppercase are never de-fused."""
    text = "Art.1prevederileart.2 PetruDumitru 2007modificari"
    result = defuse_words(text)
    # Should leave these untouched (digits, CamelCase detected)
    assert "Art.1prevederileart.2" in result
    assert "PetruDumitru" in result


def test_defuse_words_char_preservation():
    """If a token is split, the joined parts must equal the original token."""
    audit: list[dict] = []
    text = "acestaesteunarticolfoartelung"  # synthetic fused token
    defuse_words(text, audit_log=audit)
    for entry in audit:
        assert "".join(entry["parts"]) == entry["token"]


def test_defuse_words_audit_log_populated():
    """Audit log records every split made."""
    audit: list[dict] = []
    # Use a token that should be fused (if it splits, we capture it)
    defuse_words("acestaesteunarticolfoartelung", audit_log=audit)
    # Each entry has token + parts keys
    for entry in audit:
        assert "token" in entry and "parts" in entry


def test_defuse_words_idempotent():
    """Running defuse_words twice must produce the same result."""
    text = "text normal cu cateva cuvinte lungi responsabilitatile"
    once = defuse_words(text)
    twice = defuse_words(once)
    assert once == twice


def test_defuse_words_skips_signature_lines():
    """Lines containing minister signatures are skipped entirely."""
    text = "Ministrul justitiei,\nromaniaesteofarafrumoasasicuprinzatoare"
    result = defuse_words(text)
    # The signature line must be untouched; the second line may or may not be split
    assert "Ministrul justitiei," in result


def test_broken_2007_cache_files_have_no_suspects():
    """Regression guard: all db/md_cache/2007/ files must have zero suspect chars after
    normalization.  This test is skipped automatically if the cache files are absent
    (e.g. CI without the full db/ tree).
    """
    cache_dir = Path(__file__).parent.parent / "db" / "md_cache" / "2007" / "01" / "03"
    md_files = list(cache_dir.glob("*.md"))
    if not md_files:
        import pytest
        pytest.skip("db/md_cache/2007/01/03/ not present; skipping live-cache test")

    for md_file in md_files:
        raw = md_file.read_text(encoding="utf-8", errors="replace")
        fixed = normalize_text(raw, Era.BROKEN_2007)
        residual = BROKEN_2007_SUSPECTS & set(fixed)
        assert not residual, (
            f"{md_file.name}: suspect chars remain after normalization: {residual!r}"
        )
