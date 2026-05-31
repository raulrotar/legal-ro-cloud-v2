"""Legal RAG query engine — hybrid agentic + single-turn fallback."""
import asyncio
import httpx
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
# pydantic-ai ≥ 1.x: GeminiModel/GoogleProvider renamed to GoogleModel/GoogleProvider.
# Use GoogleModel (not the deprecated GeminiModel stub) so the Agent's streaming
# protocol works correctly.
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from legalro_core.config import Settings
from legalro_core.retrieval.search import hybrid_search
from legalro_core.retrieval.context import assemble_context


def _print_usage(usage: dict, label: str = "") -> None:
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt + completion)
    tag = f" [{label}]" if label else ""
    print(f"\n─── token usage{tag}: input={prompt}  output={completion}  total={total} ───", flush=True)


def run_query_hybrid(question: str, settings: Settings) -> str:
    """Hybrid: agentic+thinking first (bounded by timeout + max_tokens),
    falling back to single-turn no-thinking RAG on any failure.

    Agentic mode requires tool-calling support. MLX provider skips it
    because mlx_lm's OpenAI-compat server doesn't reliably support function
    calling — attempting it always produces a ModelHTTPError.
    """
    # MLX's OpenAI-compat server doesn't support function calling reliably
    if settings.llm.provider == "mlx":
        return run_query(question, settings)

    agent = create_agent(settings)
    agentic_timeout = settings.llm.agentic_timeout
    agentic_max_tokens = settings.llm.agentic_max_tokens

    async def _run_agentic():
        result = await agent.run(
            question,
            model_settings={"max_tokens": agentic_max_tokens},
        )
        usage = result.usage  # property in pydantic-ai ≥ 1.x (was a method before)
        _print_usage(
            {"prompt_tokens": usage.request_tokens, "completion_tokens": usage.response_tokens,
             "total_tokens": usage.total_tokens},
            label="agentic",
        )
        return result.output

    try:
        output = asyncio.run(
            asyncio.wait_for(_run_agentic(), timeout=agentic_timeout)
        )
        return output
    except (asyncio.TimeoutError, UnexpectedModelBehavior, httpx.HTTPError, Exception) as exc:
        trigger = type(exc).__name__
        msg = str(exc)
        if "token limit" in msg or "exceeded before" in msg:
            trigger = "TokenLimit"
        elif isinstance(exc, asyncio.TimeoutError):
            trigger = "Timeout"
        # Include HTTP status code if available
        status = getattr(getattr(exc, "response", None), "status_code", None)
        status_str = f" HTTP {status}" if status else ""
        detail = msg[:300].replace("\n", " ") if msg else ""
        print(f"[hybrid] stage-A failed ({trigger}{status_str}): {detail}", flush=True)
        print("[hybrid] falling back to single-turn", flush=True)
        return run_query(question, settings)


def run_query(question: str, settings: Settings, _retries: int = 3) -> str:
    """Single-turn RAG: search → assemble context → one LLM call, thinking off."""
    import time
    context = assemble_context(hybrid_search(question, settings), settings)
    user_msg = f"Documente relevante:\n{context}\n\nÎntrebare: {question}"
    api_key = settings.llm.api_key or "local"
    for attempt in range(_retries):
        try:
            resp = httpx.post(
                f"{settings.llm.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": settings.llm.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 2048,
                    "temperature": settings.llm.temperature,
                    **({"chat_template_kwargs": {"enable_thinking": False}} if settings.llm.provider == "mlx" else {}),
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            _print_usage(data.get("usage", {}), label="single-turn")
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 500, 502, 503, 504) and attempt < _retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"[retry] {exc.response.status_code} — retrying in {wait}s ({attempt + 1}/{_retries - 1})", flush=True)
                time.sleep(wait)
            else:
                raise


def create_agent(settings: Settings) -> Agent:
    """Agentic mode with search_law tool."""
    http_timeout = httpx.Timeout(settings.llm.agentic_timeout + 10)
    api_key = settings.llm.api_key or "local"
    if settings.llm.provider == "gemini":
        model = GoogleModel(
            model_name=settings.llm.model,
            provider=GoogleProvider(
                api_key=api_key,
                http_client=httpx.AsyncClient(timeout=http_timeout),
            ),
        )
    else:
        model = OpenAIModel(
            model_name=settings.llm.model,
            provider=OpenAIProvider(
                base_url=settings.llm.base_url,
                api_key=api_key,
                http_client=httpx.AsyncClient(timeout=http_timeout),
            ),
        )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        model_settings={"max_tokens": settings.llm.agentic_max_tokens},
    )

    @agent.tool_plain
    def search_law(query: str, act_type: str = "") -> str:
        """Search Romanian legal documents."""
        chunks = hybrid_search(
            query=query,
            settings=settings,
            act_type=act_type or None,
        )
        return assemble_context(chunks, settings)

    return agent


SYSTEM_PROMPT = """Ești un asistent juridic specializat în legislația românească.
Răspunzi DOAR pe baza documentelor găsite prin căutare. Nu inventa informații.

Reguli:
1. Citează sursa: tipul actului, numărul, anul, și articolul specific.
2. Dacă nu găsești informația, spune clar că nu ai găsit-o.
3. Folosește limba română.
4. Fii precis și concis.
5. Documentele recuperate prin căutare sunt sursa de adevăr și sunt întotdeauna actuale. \
Răspunde exclusiv pe baza lor, indiferent de data actului — inclusiv acte din 2025 sau 2026. \
Nu refuza și nu ezita să răspunzi invocând limita cunoștințelor tale. \
Dacă documentul există în rezultate, el este real și valid.

Format citare: [LEGE nr. 123/2024, Art. 5 alin. (2)]"""
