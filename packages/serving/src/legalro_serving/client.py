"""Thin HTTP client for the LegalRo cloud API.

Used by the CLI when LEGALRO_API_URL is set and --local is not passed.
"""
from __future__ import annotations

import os
import time
import httpx


def _base_url() -> str:
    return os.environ.get("LEGALRO_API_URL", "").rstrip("/")


def _headers() -> dict:
    headers = {}
    # HF user token — authenticates with the HF Spaces proxy (private spaces)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    # API token — authenticates with our FastAPI app
    api_token = os.environ.get("LEGALRO_API_TOKEN", "")
    if api_token:
        headers["X-API-Token"] = api_token
    return headers


def query(question: str, act_type: str = "") -> str:
    url = _base_url()
    if not url:
        raise RuntimeError("LEGALRO_API_URL is not set")
    resp = httpx.post(
        f"{url}/query",
        json={"question": question, "act_type": act_type},
        headers=_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def ingest(pdf_path: str, poll_interval: float = 5.0) -> dict:
    """Upload a PDF for ingestion and poll until done. Returns the final job dict."""
    url = _base_url()
    if not url:
        raise RuntimeError("LEGALRO_API_URL is not set")

    with open(pdf_path, "rb") as f:
        resp = httpx.post(
            f"{url}/ingest",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            headers=_headers(),
            timeout=60,
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  job {job_id} submitted — polling…", flush=True)

    while True:
        time.sleep(poll_interval)
        status_resp = httpx.get(
            f"{url}/jobs/{job_id}",
            headers=_headers(),
            timeout=30,
        )
        status_resp.raise_for_status()
        job = status_resp.json()
        print(f"  status: {job['status']}", end="\r", flush=True)
        if job["status"] in ("done", "error"):
            print()
            return job


def health() -> dict:
    url = _base_url()
    if not url:
        raise RuntimeError("LEGALRO_API_URL is not set")
    resp = httpx.get(f"{url}/health", headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()
