"""Tests for adaptive chunking."""
from legalro_processing.prepare.chunk import chunk_act, count_tokens
from legalro_core.models import ChunkStrategy


def test_whole_act_strategy_short_text():
    text = "Lege nr. 1/2024 privind ceva."
    chunks = chunk_act(text, "LEGE", "PARLAMENTUL ROMÂNIEI")
    assert len(chunks) == 1
    assert chunks[0].strategy == ChunkStrategy.WHOLE_ACT


def test_article_chunking():
    text = "Prezenta lege reglementează...\nArt. 1 Dispoziții generale.\nArt. 2 Definiții.\n" * 10
    chunks = chunk_act(text, "LEGE", "PARLAMENTUL ROMÂNIEI")
    assert len(chunks) >= 1
    assert all(c.strategy in list(ChunkStrategy) for c in chunks)


def test_token_window_fallback():
    long_text = "cuvânt " * 2000
    chunks = chunk_act(long_text, "UNKNOWN", "")
    assert len(chunks) > 1
    assert chunks[0].strategy == ChunkStrategy.TOKEN_WINDOW


def test_count_tokens():
    assert count_tokens("hello world") > 0
