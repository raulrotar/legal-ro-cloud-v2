# Pydantic-AI Agent Baseline — Generation Layer

**Date recorded:** 2026-05-23  
**QA score when active:** 22/28 CORECT, 3 PARTIAL, 0 GRESIT, 3 EROARE  
**Model:** `mlx-community/Qwen3.5-9B-4bit` (local, MLX, thinking enabled)  
**Config:** `max_tokens: 4096`, `temperature: 0.1`

---

## Why this approach worked

The pydantic-ai Agent gave the model a `search_law` tool it could call with any query string.  
The agent loop runs **two LLM turns**:

1. **Turn 1** — model sees the user question, decides to call `search_law(query=..., act_type=...)` with a targeted query (it can rewrite or refine the question before searching).
2. **Turn 2** — model sees the original question + the tool result (retrieved chunks as formatted text), thinks, and produces the final answer.

The **thinking mode** (Qwen3's `<think>` block) was active in both turns. This let the model:
- Disambiguate queries (e.g. two different ORDIN 346/2007 documents in the corpus)
- Reason about conflicting retrieved chunks
- Produce more precise citations

Questions that benefited most from the agentic loop: **Q18** (Dan Cristian Georgescu), **Q11** (taxi norme — partial disambiguation), and all questions requiring multi-hop reasoning.

---

## Code — exact implementation that produced 22/28

```python
# src/legalro/generation/agent.py

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from legalro.config import Settings
from legalro.retrieval.search import hybrid_search
from legalro.retrieval.context import assemble_context


def create_agent(settings: Settings) -> Agent:
    model = OpenAIModel(
        model_name=settings.llm.model,
        provider=OpenAIProvider(base_url=settings.llm.base_url, api_key="local"),
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        model_settings={"max_tokens": settings.llm.max_tokens},
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
```

### CLI call site (`src/legalro/cli/app.py`)

```python
@app.command()
def query(question: str, act_type: str = "", year: int = 0):
    """Ask a single question."""
    from legalro.config import load_settings
    from legalro.generation.agent import create_agent

    settings = load_settings()
    agent = create_agent(settings)
    result = agent.run_sync(question)
    typer.echo(result.output)
```

---

## Config (`config/local.yaml`) at time of best run

```yaml
llm:
  provider: "mlx"
  base_url: "http://localhost:8080/v1"
  model: "mlx-community/Qwen3.5-9B-4bit"
  max_tokens: 4096
  temperature: 0.1
```

### LLM server start command (`src/legalro/cli/app.py`)

```python
["mlx_lm.server", "--model", settings.llm.model, "--port", "8080"]
# No --max-tokens flag — server default applies
# Thinking mode: ON (Qwen3 default)
```

---

## Known failure modes with this approach

| Issue | Affected questions | Root cause |
|-------|--------------------|------------|
| `UnexpectedModelBehavior: Model token limit (N) exceeded` | Q11, Q23, Q26 | Thinking block > max_tokens before any answer generated. Large acts (50k+ chars) retrieved → model thinks very long (>4096 tokens). |
| Per-question subprocess timeout (120s) | Q11, Q23, Q26 | Same root cause — combined thinking + generation exceeds 120s per question. |

**The 3 ERROAs (Q11, Q23, Q26) were NOT wrong answers — they were system crashes before the model produced output.** The actual content in the corpus is correct for these questions; it's purely a token budget problem.

---

## Why it was replaced

The agentic approach was replaced with single-turn RAG (pre-fetch + one httpx call, thinking OFF) to eliminate the 3 EROARE crashes. The trade-off:

| Metric | Agentic + thinking | Single-turn RAG (current) |
|--------|-------------------|--------------------------|
| CORECT | 22 | 22 |
| PARTIAL | 3 | 4 |
| GRESIT | 0 | 2 |
| EROARE | 3 | 0 |

Q18 (Dan Cristian Georgescu) regressed CORECT → GRESIT without the agentic loop.

---

## How to restore the agentic approach

1. Replace `run_query` in `src/legalro/generation/agent.py` with the `create_agent` implementation above.
2. Update `src/legalro/cli/app.py` query command to call `agent.run_sync(question)`.
3. Set `max_tokens: 4096` in `config/local.yaml`.
4. Start the server **without** `--max-tokens` flag (or with a large value like `--max-tokens 16384`).
5. Increase per-question subprocess timeout in `test_questions.py` to ≥300s.

To fix the 3 ERROAs without losing the agentic accuracy, the recommended next step is a **thinking budget** cap — once mlx_lm or Qwen3 tokenizer support `thinking_budget` in `apply_chat_template`, pass it via `--chat-template-args '{"enable_thinking": true, "thinking_budget": 2048}'` at server start.
