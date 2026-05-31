"""OCR provider — Docling (local), LlamaParse, or Mistral OCR."""
import time
import random
from legalro_core.config import Settings


def ocr_pdf(pdf_path: str, settings: Settings) -> list[str]:
    """Return list of page texts extracted from a PDF."""
    if settings.ocr.provider == "llamaparse":
        return _extract_llamaparse(pdf_path, settings)
    elif settings.ocr.provider == "mistral":
        return _extract_mistral(pdf_path, settings)
    elif settings.ocr.provider == "docling":
        return _extract_docling(pdf_path)
    elif settings.ocr.provider == "ocrmac":
        return _extract_ocrmac(pdf_path, settings)
    raise ValueError(f"Unknown OCR provider: {settings.ocr.provider}")


def ocr_page(pdf_path: str, page_number: int, settings: Settings) -> str:
    """Return text for a single page (used by some extraction paths)."""
    pages = ocr_pdf(pdf_path, settings)
    if page_number < len(pages):
        return pages[page_number]
    return ""


def _extract_docling(pdf_path: str) -> list[str]:
    """Extract text via Docling — handles both digital and scanned PDFs."""
    from docling.document_converter import DocumentConverter
    result = DocumentConverter().convert(pdf_path)
    text = result.document.export_to_markdown()
    return [text]


def _extract_ocrmac(pdf_path: str, settings: Settings) -> list[str]:
    """macOS Vision OCR — local dev only."""
    from ocrmac import ocrmac as ocr_module
    annotations = ocr_module.OCR(
        pdf_path,
        recognition_level="accurate",
        language_preference=[settings.ocr.language],
    ).recognize()
    return ["\n".join(text for text, _, _ in annotations)]


def _retry_sleep(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter; returns the sleep duration."""
    delay = min(base * (2 ** attempt) + random.uniform(0, 1), cap)
    time.sleep(delay)
    return delay


def _extract_llamaparse(pdf_path: str, settings: Settings) -> list[str]:
    """Submit PDF to LlamaParse, poll for completion, return markdown as list[str].

    Throttle: 1 req/3s (≤20/min).
    Retry: 429 → exponential backoff (base 2s, cap 60s, 6 retries).
    402 (credits exhausted) → fatal stop.
    """
    import httpx

    api_key = settings.ocr.llama_cloud_api_key
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY not set — cannot use llamaparse provider")

    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = "https://api.cloud.llamaindex.ai/api/parsing"

    # Upload
    max_retries = 6
    for attempt in range(max_retries):
        try:
            with open(pdf_path, "rb") as f:
                resp = httpx.post(
                    f"{base_url}/upload",
                    headers=headers,
                    files={"file": (pdf_path, f, "application/pdf")},
                    data={"result_type": "markdown", "language": "ro"},
                    timeout=120,
                )
            if resp.status_code == 402:
                raise RuntimeError(
                    "LlamaParse credits exhausted (HTTP 402). "
                    "Re-run after your monthly reset or switch to mistral provider."
                )
            if resp.status_code == 429:
                wait = _retry_sleep(attempt)
                print(f"[llamaparse] 429 — sleeping {wait:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
                continue
            resp.raise_for_status()
            job_id = resp.json()["id"]
            break
        except httpx.HTTPStatusError:
            if attempt < max_retries - 1:
                _retry_sleep(attempt)
            else:
                raise
    else:
        raise RuntimeError("LlamaParse upload failed after max retries")

    # Poll for result
    print(f"[llamaparse] job {job_id} submitted, polling…", flush=True)
    for _ in range(120):  # up to ~4 min
        time.sleep(2)
        status_resp = httpx.get(
            f"{base_url}/job/{job_id}",
            headers=headers,
            timeout=30,
        )
        status_resp.raise_for_status()
        state = status_resp.json().get("status", "")
        if state == "SUCCESS":
            break
        if state == "ERROR":
            raise RuntimeError(f"LlamaParse job {job_id} failed: {status_resp.json()}")
        print(f"[llamaparse] status={state}", end="\r", flush=True)
    else:
        raise RuntimeError(f"LlamaParse job {job_id} timed out after polling")

    # Fetch markdown result
    result_resp = httpx.get(
        f"{base_url}/job/{job_id}/result/markdown",
        headers=headers,
        timeout=60,
    )
    result_resp.raise_for_status()
    markdown = result_resp.json().get("markdown", "")

    # Throttle: respect ≤20 req/min by sleeping 3s after every upload
    time.sleep(3)

    from legalro_core.md_normalize import normalize_llamaparse_markdown
    return normalize_llamaparse_markdown(markdown)


def _extract_mistral(pdf_path: str, settings: Settings) -> list[str]:
    """Extract text via Mistral OCR API (pixtral-12b-latest).

    Free tier: ≤2 req/min — throttle with 30s sleep after every call.
    Retry: 429 → exponential backoff.
    """
    import base64
    import httpx

    api_key = settings.ocr.mistral_api_key
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set — cannot use mistral provider")

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    max_retries = 6
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                "https://api.mistral.ai/v1/ocr",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
                timeout=300,
            )
            if resp.status_code == 429:
                wait = _retry_sleep(attempt)
                print(f"[mistral-ocr] 429 — sleeping {wait:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPStatusError:
            if attempt < max_retries - 1:
                _retry_sleep(attempt)
            else:
                raise
    else:
        raise RuntimeError("Mistral OCR failed after max retries")

    # Throttle: ≤2 req/min on the free tier
    time.sleep(30)

    from legalro_core.md_normalize import normalize_llamaparse_markdown

    # Mistral returns per-page markdown objects
    pages = data.get("pages", [])
    if pages:
        page_texts = [page.get("markdown", "") for page in pages]
        # Normalize each page independently (already split by Mistral)
        result = []
        for pt in page_texts:
            normalized = normalize_llamaparse_markdown(pt)
            result.extend(normalized if normalized else [pt])
        return result
    # Fallback: whole-doc text
    return normalize_llamaparse_markdown(data.get("text", ""))
