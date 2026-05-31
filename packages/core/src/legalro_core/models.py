from dataclasses import dataclass, field
from enum import Enum
from datetime import date
from typing import Optional


class Era(Enum):
    SCANNED = "scanned"
    BROKEN_2002 = "broken_2002"
    BROKEN_2007 = "broken_2007"
    HYBRID = "hybrid"
    MODERN = "modern"


class ChunkStrategy(Enum):
    WHOLE_ACT = "whole_act"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    TOKEN_WINDOW = "token_window"


@dataclass
class Gazette:
    issue_number: int
    part: str
    date: date
    year: int
    era: Era
    filename: str
    sha256: str
    page_count: int
    act_count: int = 0
    status: str = "pending"


@dataclass
class Act:
    gazette_id: str
    act_type: str
    number: str
    year: int
    canonical_id: str
    issuing_authority: str
    title: str
    full_text: str
    page_range: list[int]
    chunk_count: int = 0
    token_count: int = 0


@dataclass
class Chunk:
    act_id: str
    gazette_id: str
    text: str
    embedding: list[float]
    chunk_index: int
    chunk_strategy: ChunkStrategy
    article_number: Optional[str] = None
    hierarchy_path: Optional[str] = None
    token_count: int = 0
    act_type: str = ""
    act_year: int = 0
    act_canonical_id: str = ""


@dataclass
class GazetteResult:
    gazette_id: str
    era: Era
    acts_segmented: int
    chunks_created: int
    status: str
    warnings: list[str] = field(default_factory=list)
