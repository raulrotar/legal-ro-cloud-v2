"""LLM provider — routes to MLX server locally or cloud API."""
import httpx
from legalro_core.config import Settings


def call_llm(
    messages: list[dict],
    settings: Settings,
    stream: bool = False,
    max_tokens: int | None = None,
) -> str:
    max_tokens = max_tokens or settings.llm.max_tokens

    response = httpx.post(
        f"{settings.llm.base_url}/chat/completions",
        json={
            "model": settings.llm.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": settings.llm.temperature,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
