"""Tests for the VLM-OCR verification harness and page tiler."""
from __future__ import annotations

import numpy as np
import pytest

from legalro_processing.extract.ocr_verify import (
    collapse_repeated_line_blocks,
    repeated_shingle,
    verify_page,
)
from legalro_processing.extract.page_tiles import tile_page


FOOTER = (
    "EDITOR: CONSILIUL FRONTULUI SALVĂRII NATIONALE "
    "Adresa pentru publicitate: Combinatul poligrafic București – Biroul de "
    "publicitate și difuzare pentru Monitorul Oficial. "
)

# Varied legal prose — repeats short boilerplate phrases but no 40-char block
BODY = " ".join(
    f"Art. {i}. — Se numește în funcția de ministru al departamentului "
    f"numărul {i} persoana desemnată prin hotărârea numărul {i * 7} "
    f"din anul o mie nouă sute optzeci și nouă."
    for i in range(1, 12)
)


class TestRepeatedShingle:
    def test_detects_triple_repetition(self):
        # MO_PI_3_1989 failure mode: footer block emitted 3×
        assert repeated_shingle(FOOTER * 3) is not None

    def test_normal_text_passes(self):
        assert repeated_shingle(BODY) is None
        # legal boilerplate repeats short phrases but not 40-char blocks 3×
        unique_text = (
            "Decretul nr. 5 privind numirea viceprim-miniștrilor guvernului. "
            "Decretul nr. 6 privind înființarea Ministerului Economiei Naționale. "
            "Decretul nr. 7 privind eliberarea din funcție a prim-adjunctului. "
            "Decretul nr. 8 privind numirea ministrului economiei naționale. "
        )
        assert repeated_shingle(unique_text) is None

    def test_short_text_passes(self):
        assert repeated_shingle("Nr. 3. București 1989") is None


class TestVerifyPage:
    def test_complete_page_passes(self):
        v = verify_page(BODY, BODY, page_index=0)
        assert v.passed
        assert v.coverage == 1.0

    def test_omission_fails(self):
        # VLM kept ~10% of the page (the MO_PI_6 page-1 failure)
        v = verify_page(BODY[:60], BODY, page_index=0)
        assert not v.passed
        assert any("word ratio" in r or "coverage" in r for r in v.reasons)

    def test_repetition_fails(self):
        v = verify_page(FOOTER * 3, FOOTER, page_index=0)
        assert not v.passed
        assert any("repetition" in r for r in v.reasons)

    def test_blank_page_passes(self):
        v = verify_page("", "  \n 12 \n", page_index=0)
        assert v.passed

    def test_diacritics_disagreement_tolerated(self):
        vlm = "înaintarea în gradul următor a unui general din ministerul apărării"
        oracle = "inaintarea in gradul urmator a unui general din ministerul apararii"
        v = verify_page(vlm, oracle, page_index=0)
        assert v.passed


class TestCollapseRepeatedLineBlocks:
    FOOTER_BLOCK = (
        "EDITOR : CONSILIUL FRONTULUI SALVĂRII NATIONALE\n"
        "Adresa pentru publicitate: Combinatul poligrafic București\n"
        "Prețul 1,50 lei 40.816\n"
    )

    def test_collapses_duplicate_footer(self):
        text = self.FOOTER_BLOCK + self.FOOTER_BLOCK
        out = collapse_repeated_line_blocks(text)
        assert out.count("EDITOR") == 1

    def test_collapses_case_variant_copy(self):
        # MO_PI_3 page 2: third copy emitted in SHOUTING case
        shouted = self.FOOTER_BLOCK.upper()
        out = collapse_repeated_line_blocks(self.FOOTER_BLOCK + shouted)
        assert out.count("EDITOR") == 1

    def test_keeps_distinct_content(self):
        text = (
            "Art. 1. — Se numește în funcția de ministru persoana întâi.\n"
            "Art. 2. — Se numește în funcția de ministru persoana a doua.\n"
        )
        assert collapse_repeated_line_blocks(text) == text.rstrip("\n")

    def test_dot_leaders_untouched(self):
        text = ". . . . . . . . . .\n. . . . . . . . . .\n"
        assert collapse_repeated_line_blocks(text).count(". .") >= 2


class TestOverEmission:
    def test_over_emission_fails(self):
        oracle = " ".join(f"cuvântul{i} unic{i} aici{i}" for i in range(8))
        vlm = oracle + " " + " ".join(f"halucinație{i} suplimentară{i} text{i}" for i in range(30))
        v = verify_page(vlm, oracle, page_index=0)
        assert not v.passed
        assert any("over-emission" in r for r in v.reasons)


class TestTilePage:
    def _synthetic_page(self, h=2340, w=1650):
        """White page with black text-line stripes (200 DPI gazette shape)."""
        page = np.full((h, w), 255, dtype=np.uint8)
        for y in range(100, h - 150, 26):
            page[y : y + 11, 90 : w - 90] = 0
        return page

    def test_page_splits_into_two_full_width_tiles(self):
        tiles = tile_page(self._synthetic_page(), max_px=2000)
        assert len(tiles) == 2
        for t in tiles:
            assert t.height <= 2000
            assert t.width > 1400  # full width, never column-split

    def test_cut_lands_on_blank_row(self):
        page = self._synthetic_page()
        tiles = tile_page(page, max_px=2000)
        ink = page < 160
        boundary = tiles[0].y1
        # the cut row itself must be blank — no text line split
        assert ink[boundary, :].sum() == 0

    def test_small_page_single_tile(self):
        tiles = tile_page(self._synthetic_page(h=1800), max_px=2000)
        assert len(tiles) == 1

    def test_blank_page_yields_no_tiles(self):
        page = np.full((2340, 1650), 255, dtype=np.uint8)
        assert tile_page(page) == []

    def test_no_tiny_trailing_tile(self):
        # content ends just past the cut point → remnant merges into previous
        tiles = tile_page(self._synthetic_page(h=2080), max_px=2000)
        for t in tiles:
            assert t.height >= 120
