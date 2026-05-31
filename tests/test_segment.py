"""Tests for act segmentation."""
from legalro_processing.extract.segment import segment_acts, _segment_by_delimiter
from legalro_processing.extract.sumar import SumarBoundary as SumarEntry
from legalro_core.models import Era


def test_segment_by_delimiter():
    text = "act one«act two«act three"
    acts = _segment_by_delimiter(text, "«")
    assert len(acts) == 3
    assert acts[0].text == "act one"
    assert acts[1].text == "act two"


def test_segment_by_sumar():
    pages = ["ignored"] + [f"page {i}" for i in range(1, 5)]
    entries = [
        SumarEntry(title="Lege nr. 1", page_number=1),
        SumarEntry(title="OUG nr. 2", page_number=3),
    ]
    acts = segment_acts(pages, entries, Era.MODERN)
    assert len(acts) == 2
    assert acts[0].title == "Lege nr. 1"


def test_segment_fallback_returns_at_least_one():
    pages = ["some text without headers"]
    acts = segment_acts(pages, [], Era.MODERN)
    assert len(acts) >= 1
