"""Tests for character normalization."""
from legalro_core.normalize import normalize_text
from legalro_core.models import Era


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
