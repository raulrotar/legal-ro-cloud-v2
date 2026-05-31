# Backend Needs — LegalRo Frontend Integration

> Analysis of what the deployed API (`https://rraul99-legalro.hf.space`) currently exposes
> vs. what the frontend needs to replace all hardcoded data.

---

## Current API Surface

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | none | MongoDB ping |
| `POST` | `/query` | `X-API-Token` | `{question, act_type}` → `{answer: string}` |
| `POST` | `/ingest` | `X-API-Token` | Upload PDF → background job |
| `GET` | `/jobs/{id}` | `X-API-Token` | Poll ingest job status |
| `POST` | `/extract` | `X-API-Token` | PDF → structured JSON (no DB write) |
| `POST` | `/ingest-json` | `X-API-Token` | JSON → MongoDB (background) |

Auth: `X-API-Token: <API_TOKEN>` header on every protected request.
HF proxy additionally requires `Authorization: Bearer <HF_TOKEN>` if the Space is private.

---

## Gaps by Page

### 1. Dashboard — act listing table

**Status: no endpoint exists.**

The data lives in MongoDB's `chunks` collection (one document per chunk, not per act)
and the `gazettes` collection (one document per Monitorul Oficial issue).
There is no endpoint that returns a list of acts.

#### Required: `GET /acts`

Paginated, filterable list of acts for the dashboard table.

**Query parameters:**

| Param | Type | Description |
|---|---|---|
| `type` | `string` (comma-separated) | Filter by `document_type`: `LEGE,ORDIN,DECRET,HOTĂRÂRE,DECIZIE` |
| `year_from` | `int` | Filter acts with `act_year >= year_from` |
| `year_to` | `int` | Filter acts with `act_year <= year_to` |
| `q` | `string` | Substring search on `title`, `act_number`, `issuing_authority` |
| `page` | `int` | 1-based page number (default: 1) |
| `limit` | `int` | Page size (default: 50, max: 200) |
| `sort` | `string` | `date_desc` (default) \| `date_asc` \| `type` |

**Response shape (per item):**

```json
{
  "total": 1840,
  "page": 1,
  "limit": 50,
  "items": [
    {
      "id": "ORDIN_1642_2016",
      "document_type": "ORDIN",
      "act_number": "1.642",
      "act_year": 2016,
      "title": "Ordin pentru închiderea vechilor evidențe...",
      "issuing_authority": "ANCPI",
      "source_issue_id": "P1_76_2017",
      "gazette_date": "2017-01-30",
      "status": "în vigoare",
      "signed_by": "Radu-Codruț Ștefănescu",
      "pages": 1
    }
  ]
}
```

**Implementation note:** The best approach is a MongoDB aggregation on `chunks` filtered
to `position_in_law: 0` (first chunk = one record per act) with a `$group` stage.
A dedicated `acts` collection written at ingest time would be faster for this query.

**Missing fields that need to be added during ingestion:**

| Field | Status | Notes |
|---|---|---|
| `status` | ❌ not stored | "în vigoare" / "abrogat" — needs rule-based extraction from act text |
| `signed_by` | ❌ not stored | Partially present in raw text; needs structured extraction |
| `pages` | ❌ not per-act | Only stored per-gazette (`page_count`); `page_range` is on the Act model but not written to `chunks` |
| `gazette_date` | ⚠️ derivable | Can be reconstructed from `source_issue_id` + gazettes collection lookup |
| `summary` | ❌ not stored | No short summary field; `act_full_text` is the entire raw text |

---

### 2. Act detail panel

**Status: partially derivable, no dedicated endpoint.**

#### Required: `GET /acts/{id}`

Returns a single act record for the slide-out detail panel.
`id` = `law_id` field on chunks (e.g. `ORDIN_1642_2016`).

**Response shape:**

```json
{
  "id": "ORDIN_1642_2016",
  "document_type": "ORDIN",
  "act_number": "1.642",
  "act_year": 2016,
  "title": "...",
  "issuing_authority": "ANCPI",
  "full_authority": "Agenția Națională de Cadastru și Publicitate Imobiliară",
  "source_issue_id": "P1_76_2017",
  "gazette_date": "2017-01-30",
  "status": "în vigoare",
  "signed_by": "Radu-Codruț Ștefănescu",
  "pages": 1,
  "summary": "Se închid vechile evidențe de cadastru..."
}
```

---

### 3. Search — Q&A page

**Status: `/query` exists but response is incomplete.**

The endpoint works for the answer text, but the frontend also needs:
- A structured list of source acts cited in the answer
- Latency information
- Year-range filtering

#### Required changes to `POST /query`

**Extended request body:**

```json
{
  "question": "Ce județe sunt vizate de ordinele ANCPI...",
  "act_type": "ORDIN",
  "year_from": 2000,
  "year_to": 2026
}
```

`year_from` / `year_to` must be passed into `hybrid_search()` as an additional
`vector_filter` on `act_year` and also as a `$match` pre-filter on the BM25 pipeline.

**Extended response body:**

```json
{
  "answer": "Cele două ordine vizează județe diferite...",
  "sources": [
    {
      "document_type": "ORDIN",
      "act_number": "1.642",
      "act_year": 2016,
      "title": "Ordin pentru închiderea vechilor evidențe...",
      "issuing_authority": "ANCPI",
      "source_issue_id": "P1_76_2017",
      "gazette_date": "2017-01-30",
      "rrf_score": 0.031,
      "excerpt": "Se închid vechile evidențe de cadastru..."
    }
  ],
  "latency_ms": 3800,
  "chunks_used": 12
}
```

The `sources` data is already assembled inside `assemble_context()` — the
`hybrid_search()` results contain all the required fields. They just need to be
returned alongside the answer instead of being discarded.

---

### 4. AI Chat page

> **⏸ Deferred — scheduled for future implementation. Very low priority.**
> The chat UI is built in the frontend as a prototype only. No backend work
> should be planned or started until the dashboard and search pages are fully
> wired to real data.

**Status: no `/chat` endpoint exists.** Notes kept here for when the feature is picked up.

The current `/query` is fully stateless (single-turn). The chat UI will require
multi-turn conversation with memory of prior exchanges.

#### Future requirement: `POST /chat`

```json
{
  "messages": [
    {"role": "user",      "content": "Ce act se aplică la închiderea vechilor evidențe?"},
    {"role": "assistant", "content": "Ordinul ANCPI nr. 1.642/2016..."},
    {"role": "user",      "content": "Care sunt cele două UAT-uri vizate?"}
  ],
  "scope": {
    "act_types": ["LEGE", "ORDIN"],
    "year_from": 1989,
    "year_to": 2026
  }
}
```

**Response:**

```json
{
  "answer": "Cele două UAT-uri vizate...",
  "sources": [
    {
      "document_type": "ORDIN",
      "act_number": "1.642",
      "act_year": 2016,
      "title": "...",
      "issuing_authority": "ANCPI",
      "source_issue_id": "P1_76_2017",
      "excerpt": "..."
    }
  ],
  "latency_ms": 1400
}
```

**When ready:** The pydantic-ai `Agent` in `create_agent()` supports multi-turn via
`message_history` injection. Prior turns should be passed as context so the agent
can reference earlier answers without re-searching.

---

## Priority Summary

| # | Need | Effort | Unblocks |
|---|---|---|---|
| **1** | `/query` returns `sources[]` alongside answer | Low — already in `hybrid_search()` results, just expose them | Search page source cards |
| ~~**2**~~ | ~~`POST /chat` with message history~~ | ~~Medium~~ | **Deferred** — future release, very low priority |
| **3** | `GET /acts` paginated + filtered | Medium — MongoDB aggregation on `chunks` grouped by `law_id` | Dashboard table |
| **4** | Year-range filter on `/query` (`year_from`, `year_to`) | Low — add to `QueryRequest`, pass to `vector_filter` | Search page year filter |
| **5** | `GET /acts/{id}` single-act detail | Low — single `chunks` lookup once `GET /acts` exists | Dashboard detail panel |
| **6** | `act_status` field at ingest time | Medium — rule-based extraction ("abrogat" keyword detection) | Dashboard status column |
| **7** | `gazette_date` per act | Low — gazette date is in `gazettes` collection, join on `gazette_id` | Dashboard date column |
| **8** | `signed_by` field at ingest time | Medium — present in raw text, needs structured extraction | Dashboard / detail panel |
| **9** | `summary` per act | High — no extraction today; could be first 300 chars of `act_full_text` as a cheap proxy | Dashboard / detail panel |

Items 1, 2, and 4 can be implemented entirely inside `app.py` with no schema
migrations or re-ingestion required.
