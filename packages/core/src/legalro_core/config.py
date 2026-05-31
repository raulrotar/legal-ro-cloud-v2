from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class LLMConfig(BaseSettings):
    provider: str = "gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = Field(default="", alias="api_key")
    model: str = "gemini-2.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.1
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
