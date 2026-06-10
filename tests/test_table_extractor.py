"""Unit tests for md_table_extractor.find_table_regions."""
import pytest
from legalro_processing.extract.md_table_extractor import find_table_regions

_TABLE_MD = """\
# ORDIN Nr. 384

Ceva text introductiv.

## Situația finanțărilor partidelor politice

| Denumirea | Cuantum |
| --- | --- |
| Partidul Alpha | 180 |
| Partidul Beta | 240 |
| Partidul Gamma | 95 |
| Partidul Delta | 310 |
| Partidul Epsilon | 55 |

Text după tabel.
"""

_ACT_ONLY_MD = """\
# ORDIN Nr. 1639

privind aprobarea unor normative.

**București, 25 ianuarie 2017.**

**Nr. 1639.**
"""

_SMALL_TABLE_MD = """\
# DECIZIE

| Col A | Col B |
| --- | --- |
| x | y |
| z | w |

text
"""


def test_table_region_detected():
    tables, masked = find_table_regions(_TABLE_MD)
    assert len(tables) == 1
    tbl = tables[0]
    assert tbl.n_rows >= 5
    assert "Partidul Alpha" in tbl.markdown
    assert "180" in tbl.markdown
    # Table placeholder in masked output; original rows gone
    assert "Partidul Alpha" not in masked
    assert "TABELUL EXTRAS" in masked


def test_act_only_no_tables():
    tables, masked = find_table_regions(_ACT_ONLY_MD)
    assert tables == []
    assert masked == _ACT_ONLY_MD


def test_small_table_not_extracted():
    # Only 2 data rows — below _MIN_TABLE_ROWS threshold
    tables, masked = find_table_regions(_SMALL_TABLE_MD)
    assert tables == []
    # Original content preserved
    assert "Col A" in masked


def test_title_captured():
    tables, _ = find_table_regions(_TABLE_MD)
    assert tables[0].title == "Situația finanțărilor partidelor politice"


def test_page_break_tracking():
    md = (
        "<!-- legalro:page-break -->\n"
        "<!-- legalro:page-break -->\n"
        "| A | B |\n| --- | --- |\n"
        + "| x | y |\n" * 5
    )
    tables, _ = find_table_regions(md)
    assert len(tables) == 1
    assert tables[0].page == 2
