"""Adaptive chunking strategies for legal acts."""
import re
from dataclasses import dataclass
from legalro_core.models import ChunkStrategy
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    strategy: ChunkStrategy
    article_number: str | None = None
    hierarchy_path: str | None = None
    token_count: int = 0


ARTICLE_PATTERN = re.compile(r'^(?:Art\.|Articolul)\s*(\d+)', re.MULTILINE)
ALINEAT_PATTERN = re.compile(r'^\s*\((\d+)\)', re.MULTILINE)
PARAGRAPH_PATTERN = re.compile(r'^\s*(\d+)\.\s', re.MULTILINE)

# Sizes follow the 2026 legal-RAG evidence (docs/EMBEDDINGS_PLAN.md):
# bge-m3 dense quality peaks <=512 tokens and legal benchmarks favor
# article-unit chunks, so the cap drops 1024→512 and the merge target
# 800→450.  Overlap stays only for the fallback window splitter —
# structural chunks are self-contained and overlap is wasted index.
TARGET_TOKENS = 450
MIN_TOKENS = 50
MAX_TOKENS = 512
OVERLAP_TOKENS = 80


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def chunk_act(act_text: str, act_type: str, authority: str) -> list[TextChunk]:
    strategy = _select_strategy(act_text, act_type, authority)

    if strategy == ChunkStrategy.WHOLE_ACT:
        chunks = [TextChunk(
            text=act_text, chunk_index=0,
            strategy=strategy, token_count=count_tokens(act_text)
        )]
    elif strategy == ChunkStrategy.ARTICLE:
        chunks = _chunk_by_article(act_text)
    elif strategy == ChunkStrategy.PARAGRAPH:
        chunks = _chunk_by_paragraph(act_text)
    else:
        chunks = _chunk_by_window(act_text)

    return _enforce_max_size(chunks)


def _enforce_max_size(chunks: list[TextChunk]) -> list[TextChunk]:
    """Window-split any chunk above MAX_TOKENS. Structural chunkers (article/
    paragraph) can emit oversized chunks when markers are sparse (tables,
    annexes); without this, content past the embedding truncation limit becomes
    unsearchable."""
    result = []
    for chunk in chunks:
        if chunk.token_count <= MAX_TOKENS:
            result.append(chunk)
            continue
        for sub in _chunk_by_window(chunk.text):
            result.append(TextChunk(
                text=sub.text, chunk_index=len(result),
                strategy=chunk.strategy,
                article_number=chunk.article_number,
                hierarchy_path=chunk.hierarchy_path,
                token_count=sub.token_count,
            ))
    for i, chunk in enumerate(result):
        chunk.chunk_index = i
    return result


def _select_strategy(text: str, act_type: str, authority: str) -> ChunkStrategy:
    tokens = count_tokens(text)
    if tokens <= 512:
        return ChunkStrategy.WHOLE_ACT
    if act_type in ("LEGE", "OUG", "HG", "HOTĂRÂRE", "ORDIN", "ORDONANȚĂ", "DECRET_LEGE"):
        if ARTICLE_PATTERN.search(text):
            return ChunkStrategy.ARTICLE
    if act_type == "DECIZIE" and "Curtea Constituțională" in authority:
        return ChunkStrategy.PARAGRAPH
    return ChunkStrategy.TOKEN_WINDOW


def _chunk_by_article(text: str) -> list[TextChunk]:
    splits = ARTICLE_PATTERN.split(text)

    chunks = []
    preamble = splits[0].strip()
    if preamble and count_tokens(preamble) >= MIN_TOKENS:
        chunks.append(TextChunk(
            text=preamble, chunk_index=0,
            strategy=ChunkStrategy.ARTICLE,
            hierarchy_path="preamble",
            token_count=count_tokens(preamble)
        ))

    buffer_text = ""
    buffer_articles = []

    for i in range(1, len(splits), 2):
        art_num = splits[i]
        art_text = splits[i + 1] if i + 1 < len(splits) else ""
        full_article = f"Art. {art_num}{art_text}"
        art_tokens = count_tokens(full_article)

        if art_tokens > MAX_TOKENS:
            if buffer_text:
                chunks.append(_make_article_chunk(buffer_text, buffer_articles, len(chunks)))
                buffer_text = ""
                buffer_articles = []
            chunks.extend(_split_article_by_alineat(full_article, art_num, len(chunks)))
        elif count_tokens(buffer_text + full_article) > TARGET_TOKENS:
            if buffer_text:
                chunks.append(_make_article_chunk(buffer_text, buffer_articles, len(chunks)))
            buffer_text = full_article
            buffer_articles = [art_num]
        else:
            buffer_text += "\n" + full_article if buffer_text else full_article
            buffer_articles.append(art_num)

    if buffer_text:
        chunks.append(_make_article_chunk(buffer_text, buffer_articles, len(chunks)))

    return chunks


def _make_article_chunk(text: str, articles: list[str], index: int) -> TextChunk:
    path = f"art_{articles[0]}" if len(articles) == 1 else f"art_{articles[0]}-{articles[-1]}"
    return TextChunk(
        text=text.strip(), chunk_index=index,
        strategy=ChunkStrategy.ARTICLE,
        article_number=articles[0],
        hierarchy_path=path,
        token_count=count_tokens(text)
    )


def _split_article_by_alineat(text: str, art_num: str, start_index: int) -> list[TextChunk]:
    parts = ALINEAT_PATTERN.split(text)
    chunks = []
    buffer = parts[0]

    for i in range(1, len(parts), 2):
        alin_num = parts[i]
        alin_text = parts[i + 1] if i + 1 < len(parts) else ""
        piece = f"({alin_num}){alin_text}"

        if count_tokens(buffer + piece) > TARGET_TOKENS and buffer.strip():
            chunks.append(TextChunk(
                text=buffer.strip(),
                chunk_index=start_index + len(chunks),
                strategy=ChunkStrategy.ARTICLE,
                article_number=art_num,
                hierarchy_path=f"art_{art_num}.alin_{alin_num}",
                token_count=count_tokens(buffer)
            ))
            buffer = piece
        else:
            buffer += piece

    if buffer.strip():
        chunks.append(TextChunk(
            text=buffer.strip(),
            chunk_index=start_index + len(chunks),
            strategy=ChunkStrategy.ARTICLE,
            article_number=art_num,
            token_count=count_tokens(buffer)
        ))

    return chunks


def _chunk_by_paragraph(text: str) -> list[TextChunk]:
    parts = PARAGRAPH_PATTERN.split(text)
    chunks = []
    buffer = parts[0] if parts[0].strip() else ""

    for i in range(1, len(parts), 2):
        para_num = parts[i]
        para_text = parts[i + 1] if i + 1 < len(parts) else ""
        piece = f"{para_num}. {para_text}"

        if count_tokens(buffer + piece) > TARGET_TOKENS and buffer.strip():
            chunks.append(TextChunk(
                text=buffer.strip(), chunk_index=len(chunks),
                strategy=ChunkStrategy.PARAGRAPH,
                token_count=count_tokens(buffer)
            ))
            buffer = piece
        else:
            buffer += "\n" + piece if buffer else piece

    if buffer.strip():
        chunks.append(TextChunk(
            text=buffer.strip(), chunk_index=len(chunks),
            strategy=ChunkStrategy.PARAGRAPH,
            token_count=count_tokens(buffer)
        ))

    return chunks or [TextChunk(text=text, chunk_index=0, strategy=ChunkStrategy.PARAGRAPH, token_count=count_tokens(text))]


def _chunk_by_window(text: str) -> list[TextChunk]:
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        # window must respect MAX_TOKENS — _enforce_max_size re-splits via
        # this function, so any slack here permanently leaks oversized chunks
        end = min(start + TARGET_TOKENS, start + MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(TextChunk(
            text=chunk_text.strip(), chunk_index=len(chunks),
            strategy=ChunkStrategy.TOKEN_WINDOW,
            token_count=len(chunk_tokens)
        ))

        start = end - OVERLAP_TOKENS if end < len(tokens) else end

    return chunks
