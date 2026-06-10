"""Shared OpenAI-compatible HTTP LLM client — used by serving and processing.

Provides a thin, provider-agnostic HTTP wrapper around any endpoint that speaks
the OpenAI /chat/completions API: cloud Gemini, local MLX, vLLM on a rented GPU.
The caller selects the backend by passing base_url + model + api_key; no import-
time coupling to any specific provider.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx


# ── Lenient JSON extractor ────────────────────────────────────────────────────

def _escape_control_chars_in_strings(raw: str) -> str:
    """Escape bare control characters that appear inside JSON string literals.

    Models like Llama emit literal newlines/tabs inside long string values
    (e.g. full_text_corrected) without the required JSON backslash escaping.
    This scans the raw text character-by-character and escapes control chars
    that appear between unescaped double-quote pairs.
    """
    result = []
    in_string = False
    escape_next = False
    _CTRL_MAP = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch in _CTRL_MAP:
            result.append(_CTRL_MAP[ch])
            continue
        result.append(ch)
    return ''.join(result)


def loads_lenient(raw: str) -> Any:
    """Parse JSON from an LLM response that may contain extra tokens or fences.

    Handles:
    - EOS tokens appended by local servers (Ollama, vLLM): ``<|im_end|>``, ``<|eot_id|>``, ``</s>``
    - Markdown code fences: ```json … ``` or ``` … ```
    - Leading/trailing whitespace
    - Multiple JSON objects on one line (takes the first balanced ``{…}``)

    Raises ``json.JSONDecodeError`` if no valid JSON object can be extracted.
    """
    # Strip known EOS markers
    raw = re.sub(r'<\|im_end\|>|<\|eot_id\|>|</s>', '', raw).strip()

    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Fast path: if the whole string parses cleanly, return it
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Sanitize unescaped control characters inside JSON string values.
    # Llama (and other models) sometimes emit literal \n, \t, \r inside
    # string values without escaping them, causing "Invalid control character".
    # Strategy: escape bare control chars that appear between JSON quotes.
    try:
        sanitized = _escape_control_chars_in_strings(raw)
        return json.loads(sanitized)
    except (json.JSONDecodeError, Exception):
        pass

    # Slow path: extract the first balanced {…} object
    start = raw.find('{')
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise json.JSONDecodeError("Unbalanced JSON object", raw, start)


def call_llm(
    messages: list[dict[str, Any]],
    base_url: str,
    model: str,
    api_key: str = "local",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = False,
    timeout: float = 60.0,
    max_retries: int = 2,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint.

    Parameters
    ----------
    messages:
        Standard OpenAI messages list. Each message may include a ``content``
        that is a list of content parts (text + image_url) for vision models.
    base_url:
        Root of the OpenAI-compatible API, e.g.
        ``"https://generativelanguage.googleapis.com/v1beta/openai/"``.
    model:
        Model identifier, e.g. ``"gemini-3.1-flash-lite"``.
    api_key:
        Bearer token.  Pass ``"local"`` for unauthenticated local servers.
    temperature:
        Sampling temperature.  Use 0.0 for structured extraction tasks.
    max_tokens:
        Maximum tokens in the completion.
    json_mode:
        When True, sets ``response_format={"type":"json_object"}`` so the model
        is constrained to emit valid JSON.  Not all providers support this flag;
        the prompt should still explicitly ask for JSON as a fallback.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of additional attempts on transient 429/5xx errors (exponential
        back-off: 5s, 10s, …).

    Returns
    -------
    str
        The assistant message content string.

    Raises
    ------
    httpx.HTTPStatusError
        On a non-2xx status that is not retried (e.g. 400, 401).
    httpx.TimeoutException
        If the request times out after all retries.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key or 'local'}"}

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra_body:
        payload.update(extra_body)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s …
                time.sleep(wait)
                last_exc = exc
                continue
            raise
        except httpx.TimeoutException as exc:
            if attempt < max_retries:
                time.sleep(5)
                last_exc = exc
                continue
            raise

    # Should not reach here, but satisfy the type checker.
    assert last_exc is not None
    raise last_exc
