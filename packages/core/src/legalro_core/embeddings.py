"""Embedding provider — sentence-transformers (cloud) or MLX (local)."""
from legalro_core.config import Settings

_st_model = None
_mlx_model = None


# ── sentence-transformers (cloud / cross-platform) ────────────────────────────

def _get_st_model(settings: Settings):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(settings.embeddings.model)
    return _st_model


# ── MLX (local Apple Silicon only) ───────────────────────────────────────────

def _patch_tokenizer(tokenizer):
    if not hasattr(tokenizer, "batch_encode_plus"):
        tokenizer.batch_encode_plus = tokenizer.__call__
    return tokenizer


def _get_mlx_model(settings: Settings):
    global _mlx_model
    if _mlx_model is None:
        from mlx_embedding_models.embedding import EmbeddingModel
        _mlx_model = EmbeddingModel.from_registry(settings.embeddings.model)
        _patch_tokenizer(_mlx_model.tokenizer)
    return _mlx_model


_MLX_MAX_TOKENS = 510
_ST_MAX_SEQ_LEN = 8192
_ST_MAX_CHARS = 30000

# BGE-M3 query instruction in Romanian — improves asymmetric retrieval quality.
# Applied only at query time (embed_texts with is_query=True), not at ingest time.
_BGE_M3_QUERY_INSTRUCTION = "Reprezintă această interogare pentru căutarea documentelor juridice românești: "
_BGE_M3_MODEL = "BAAI/bge-m3"


def _truncate_to_tokens(text: str, model) -> str:
    enc = model.tokenizer(text, truncation=True, max_length=_MLX_MAX_TOKENS, return_tensors=None)
    return model.tokenizer.decode(enc["input_ids"], skip_special_tokens=True)


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], settings: Settings, is_query: bool = False) -> list[list[float]]:
    if settings.embeddings.provider == "sentence-transformers":
        model = _get_st_model(settings)
        model.max_seq_length = _ST_MAX_SEQ_LEN
        if is_query and settings.embeddings.model == _BGE_M3_MODEL:
            texts = [_BGE_M3_QUERY_INSTRUCTION + t for t in texts]
        truncated = [t[:_ST_MAX_CHARS] for t in texts]
        return model.encode(
            truncated,
            batch_size=settings.embeddings.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
    elif settings.embeddings.provider == "mlx":
        model = _get_mlx_model(settings)
        truncated = [_truncate_to_tokens(t, model) for t in texts]
        return model.encode(truncated).tolist()
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embeddings.provider}")


def embed_batch(texts: list[str], settings: Settings) -> list[list[float]]:
    all_embeddings = []
    batch_size = settings.embeddings.batch_size
    for i in range(0, len(texts), batch_size):
        all_embeddings.extend(embed_texts(texts[i:i + batch_size], settings))
    return all_embeddings
