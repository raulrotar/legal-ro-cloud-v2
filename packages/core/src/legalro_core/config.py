from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class ExtractionLLMConfig(BaseSettings):
    """Configuration for the optional LLM-based extraction stage.

    All fields default to False/empty so the stage is completely opt-in.
    When ``enabled=False`` (the default) the pipeline behaves exactly as it
    did before this feature was introduced — no LLM calls, no latency change.

    Provider resolution:
      - If ``base_url`` / ``model`` / ``api_key`` are empty, the extraction
        stage inherits the values from ``Settings.llm`` at call time.
      - Set them explicitly to route extraction to a different endpoint (e.g.
        a local vLLM server) while keeping generation on cloud Gemini.

    Phases:
      1. ``metadata_enabled`` — LLM reads per-act OCR text, returns structured
         fields (doc_type, act_number, issuing_authority, title, …).  The
         verbatim ``full_text`` is NEVER in the LLM output path.
      2. ``segmentation_enabled`` — LLM returns act boundaries as character
         offsets into the concatenated OCR text; Python slices ``RawAct.text``
         from the verbatim source.  Only active for the SCANNED era.
      3. ``vlm_enabled`` — Vision-language model on page images (SCANNED era
         only).  Emits the same metadata/boundary DTO as phase 1; text still
         comes from OCR.  Requires a vision-capable ``model``.
    """
    enabled: bool = False               # master toggle — default OFF = exact legacy behaviour
    # ── Option B: metadata-only mode ─────────────────────────────────────────
    metadata_enabled: bool = False      # phase 1: LLM metadata extraction
    segmentation_enabled: bool = False  # phase 2: LLM segmentation via offsets
    vlm_enabled: bool = False           # phase 3: VLM on page images (SCANNED only)
    # ── Option C: Docling→MD→LLM→JSON mode ───────────────────────────────────
    mode: str = "metadata_only"         # "metadata_only" (B) | "md_llm" (C)
    md_cache_dir: str = "md_cache"      # where to save/load intermediate .md files
    edit_distance_threshold: float = 0.15  # hallucination guard: max allowed edit ratio
    # ── Shared provider config ────────────────────────────────────────────────
    base_url: str = ""                  # empty → inherit Settings.llm.base_url
    model: str = ""                     # empty → inherit Settings.llm.model
    api_key: str = Field(default="", alias="api_key")
    temperature: float = 0.0           # 0.0 recommended for structured extraction
    max_tokens: int = 2048
    max_retries: int = 2
    # ── Fallback model (used on validation-triggered retries) ─────────────────
    # When validation finds ERROR-level issues after the primary extraction,
    # the pipeline retries using the fallback model instead of the same one.
    # Leave all fallback fields empty to retry with the same model (default).
    # Example: primary = Gemini (fast), fallback = local Llama 3.1 (thorough).
    fallback_base_url: str = ""         # empty → same as base_url
    fallback_model: str = ""            # empty → same as model (no escalation)
    fallback_api_key: str = Field(default="", alias="fallback_api_key")
    fallback_max_tokens: int = 0        # 0 → same as max_tokens


class LLMConfig(BaseSettings):
    provider: str = "gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = Field(default="", alias="api_key")
    model: str = "gemini-2.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.0
    agentic_max_tokens: int = 4096
    agentic_timeout: float = 90.0


class EmbeddingsConfig(BaseSettings):
    provider: str = "sentence-transformers"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 768
    batch_size: int = 32


class OCRConfig(BaseSettings):
    provider: str = "docling"
    language: str = "ro"
    llama_cloud_api_key: str = Field(default="", alias="llama_cloud_api_key")
    mistral_api_key: str = Field(default="", alias="mistral_api_key")


class MongoDBConfig(BaseSettings):
    uri: str = Field(default="", alias="uri")
    database: str = "legalro"


class SearchConfig(BaseSettings):
    use_rank_fusion: bool = False
    vector_weight: float = 0.3
    text_weight: float = 0.7
    rrf_k: int = 60
    num_candidates: int = 200
    limit: int = 10
    parent_doc_top_n: int = 3
    max_parent_chars: int = 8000


class Settings(BaseSettings):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    extraction_llm: ExtractionLLMConfig = Field(default_factory=ExtractionLLMConfig)


def load_settings(config_path: Path | str | None = None) -> Settings:
    import os

    # Load .env file if present (does not override already-set env vars)
    _load_dotenv()

    if config_path is None:
        env = os.getenv("LEGALRO_ENV", "").lower()
        if env == "staging":
            config_path = Path("config/staging.yaml")
        else:
            config_path = Path("config/cloud.yaml")

    path = Path(config_path)
    if not path.exists():
        path = Path("config/local.yaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Env var overrides (take precedence over yaml values)
    env = os.getenv("LEGALRO_ENV", "").lower()
    if env == "staging" and os.getenv("MONGODB_URI_STAGING"):
        data.setdefault("mongodb", {})["uri"] = os.environ["MONGODB_URI_STAGING"]
    elif os.getenv("MONGODB_URI"):
        data.setdefault("mongodb", {})["uri"] = os.environ["MONGODB_URI"]
    if os.getenv("LLAMA_CLOUD_API_KEY"):
        data.setdefault("ocr", {})["llama_cloud_api_key"] = os.environ["LLAMA_CLOUD_API_KEY"]
    if os.getenv("MISTRAL_API_KEY"):
        data.setdefault("ocr", {})["mistral_api_key"] = os.environ["MISTRAL_API_KEY"]
    provider = data.get("llm", {}).get("provider", "gemini")
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["GROQ_API_KEY"]
    elif os.getenv("GEMINI_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["GEMINI_API_KEY"]
    elif os.getenv("GROQ_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["GROQ_API_KEY"]

    # Extraction LLM overrides — master toggle and optional separate key
    if os.getenv("EXTRACTION_LLM_ENABLED", "").lower() in ("1", "true", "yes"):
        data.setdefault("extraction_llm", {})["enabled"] = True
        data["extraction_llm"].setdefault("metadata_enabled", True)
    if os.getenv("EXTRACTION_LLM_API_KEY"):
        data.setdefault("extraction_llm", {})["api_key"] = os.environ["EXTRACTION_LLM_API_KEY"]
    if os.getenv("EXTRACTION_LLM_BASE_URL"):
        data.setdefault("extraction_llm", {})["base_url"] = os.environ["EXTRACTION_LLM_BASE_URL"]
    if os.getenv("EXTRACTION_LLM_MODEL"):
        data.setdefault("extraction_llm", {})["model"] = os.environ["EXTRACTION_LLM_MODEL"]

    return Settings(**data)


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Minimal .env loader — no extra dependencies required."""
    import os
    if not dotenv_path.exists():
        return
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't override already-set vars
                os.environ[key] = value
