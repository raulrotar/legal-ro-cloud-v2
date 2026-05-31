"""Tests for era detection."""
from legalro_core.models import Era


def test_era_values():
    assert Era.SCANNED.value == "scanned"
    assert Era.MODERN.value == "modern"
    assert Era.BROKEN_2002.value == "broken_2002"
