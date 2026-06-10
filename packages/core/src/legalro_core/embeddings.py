"""Embedding provider — Ollama (local), sentence-transformers (cross-platform), or MLX (Apple Silicon)."""
from legalro_core.config import Settings

_st_model = None
_mlx_model = None


# ── sentence-transformers (cloud / cross-platform) ────────────────────────────

def _patch_auto_processor_for_text_models() -> None:
    """Workaround for sentence-transformers ≥5.5 calling AutoProcessor on text-only models.

    sentence-transformers 5.5.x tries to load an AutoProcessor for every model.
    Text-only models like BAAI/bge-m3 have no processor config, so transformers
    raises ValueError. We patch AutoProcessor.from_pretrained to return None for
    those models instead of raising, which sentence-transformers handles gracefully.
    """
    try:
        from transformers import AutoProcessor
        _orig_fn = AutoProcessor.from_pretrained.__func__

        @classmethod  # type: ignore[misc]
        def _safe_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            try:
                return _orig_fn(cls, pretrained_model_name_or_path, *args, **kwargs)
            except ValueError as exc:
                if "Unrecognized processing class" in str(exc):
                    return None
                raise

        AutoProcessor.from_pretrained = _safe_from_pretrained
    except Exception:
        pass


def _get_st_model(settings: Settings):
    global _st_model
    if _st_model is None:
        import sentence_transformers as _st_pkg
        print(f"[embeddings] loading {settings.embeddings.model} (sentence-transformers {_st_pkg.__version__})", flush=True)
        _patch_auto_processor_for_text_models()
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(settings.embeddings.model)
        print(f"[embeddings] model loaded OK, dim={_st_model.get_sentence_embedding_dimension()}", flush=True)
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


# ── Ollama (local server — manages its own model loading/unloading) ───────────

def _embed_ollama(texts: list[str], settings: Settings, is_query: bool = False) -> list[list[float]]:
    """Embed via Ollama's OpenAI-compatible /v1/embeddings endpoint.

    Ollama keeps bge-m3 loaded while requests arrive and unloads it after
    keep_alive (default 5 min), so there is no persistent in-process RAM cost.
    """
    import httpx

    base_url = getattr(settings.llm, "base_url", "http://localhost:11434/v1")
    model = settings.embeddings.model
    if is_query and model == "bge-m3":
        texts = [_BGE_M3_QUERY_INSTRUCTION + t for t in texts]

    truncated = [t[:_ST_MAX_CHARS] for t in texts]
    resp = httpx.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": truncated},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], settings: Settings, is_query: bool = False) -> list[list[float]]:
    if settings.embeddings.provider == "ollama":
        return _embed_ollama(texts, settings, is_query=is_query)
    elif settings.embeddings.provider == "sentence-transformers":
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
