# Learning Resources — LegalRo Tech Stack

## Technologies Used

### Language & Runtime
- **Python 3.12** — async/await, type hints, dataclasses
- **uv** — fast Python package/project manager (replaces pip + venv)

### API Layer
- **FastAPI** — async REST API framework
- **Uvicorn** — ASGI server that runs FastAPI
- **httpx** — async HTTP client (used for LLM API calls)

### CLI
- **Typer** — builds the `legalro` CLI from Python functions
- **Rich** — terminal formatting (colors, tables, progress bars)

### Data Validation & Config
- **Pydantic v2** — data models, validation, serialization
- **pydantic-settings** — loads config from YAML + `.env` files
- **PyYAML** — parses `config/cloud.yaml` / `config/local.yaml`

### PDF Processing & OCR
- **PyMuPDF (fitz)** — fast PDF text extraction for modern/broken PDFs
- **Mistral AI OCR API** — cloud OCR for scanned (image-only) PDFs
- **Docling** — structured document parsing (tables, headings)
- **LlamaParse** — alternative PDF parser (via LlamaIndex)
- **ocrmac** *(local only)* — Apple Silicon native OCR

### Embeddings & NLP
- **sentence-transformers** — loads `nomic-embed-text-v1.5` for 768-dim embeddings
- **tiktoken** — token counting (used for chunking budget)
- **numpy** — vector math
- **mlx-embedding-models** *(local only)* — Apple Silicon GPU embeddings

### Database & Search
- **MongoDB Atlas** — document store + vector index + BM25 full-text search
- **pymongo** — Python driver for MongoDB
- **Atlas `$vectorSearch`** — ANN cosine similarity search (768-dim)
- **Atlas `$search`** — BM25 full-text search with Romanian analyzer
- **RRF (Reciprocal Rank Fusion)** — manual re-ranking algorithm combining both

### LLM / Agent Layer
- **Google Gemini API** — cloud LLM (`gemini-2.5-flash-lite`)
- **pydantic-ai** — agent framework with tool calling; wraps Gemini + OpenAI APIs
- **mlx-lm** *(local only)* — runs quantized LLMs on Apple Silicon

### Deployment & Infrastructure
- **Docker** — containerizes the app (`Dockerfile`, `docker-compose.yml`)
- **Hugging Face Spaces** — hosts the Docker container; free GPU/CPU cloud
- **GitHub Actions** — CI/CD: pushes to HF Spaces on merge to `main`

### Testing
- **pytest** — unit tests for chunking, normalization, era detection, segmentation

---

## Knowledge Map — What You Need to Learn

### 1. Python Foundations

| Topic | Resource |
|---|---|
| Python async/await | [Real Python — Async IO](https://realpython.com/async-io-python/) |
| Type hints & Pydantic | [Pydantic v2 docs](https://docs.pydantic.dev/latest/) |
| uv package manager | [uv docs](https://docs.astral.sh/uv/) |

### 2. FastAPI + REST APIs

| Resource | Link |
|---|---|
| FastAPI official tutorial | https://fastapi.tiangolo.com/tutorial/ |
| Full FastAPI course (freeCodeCamp, YouTube) | https://www.youtube.com/watch?v=0sOvCWFmrtA |

### 3. PDF Processing & OCR

| Topic | Resource |
|---|---|
| PyMuPDF (fitz) | [PyMuPDF docs](https://pymupdf.readthedocs.io/en/latest/) |
| Mistral OCR API | [Mistral OCR docs](https://docs.mistral.ai/capabilities/document/) |
| Docling | [Docling GitHub](https://github.com/DS4SD/docling) |

### 4. Text Chunking & NLP

| Topic | Resource |
|---|---|
| tiktoken | [tiktoken GitHub](https://github.com/openai/tiktoken) |
| Chunking strategies (RAG) | [LangChain chunking guide](https://python.langchain.com/docs/concepts/text_splitters/) |

### 5. Vector Embeddings

| Topic | Resource |
|---|---|
| sentence-transformers | [SBERT docs](https://www.sbert.net/) |
| Embeddings concept (video) | [3Blue1Brown — Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) |
| Nomic Embed model | [Nomic blog post](https://blog.nomic.ai/posts/nomic-embed-text-v1) |

### 6. MongoDB Atlas (Vector + BM25 Search)

| Topic | Resource |
|---|---|
| MongoDB basics (CRUD) | [MongoDB University M001](https://learn.mongodb.com/learning-paths/introduction-to-mongodb) |
| Atlas Vector Search | [Atlas Vector Search quickstart](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/) |
| Atlas Search (BM25) | [Atlas Search docs](https://www.mongodb.com/docs/atlas/atlas-search/) |
| Aggregation pipelines | [Aggregation pipeline tutorial](https://www.mongodb.com/docs/manual/core/aggregation-pipeline/) |
| pymongo driver | [pymongo docs](https://pymongo.readthedocs.io/en/stable/) |

### 7. RAG (Retrieval-Augmented Generation)

| Topic | Resource |
|---|---|
| RAG concept (paper) | [Original RAG paper (arXiv)](https://arxiv.org/abs/2005.11401) |
| RAG practical guide | [LlamaIndex RAG guide](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) |
| Hybrid search + RRF | [Pinecone hybrid search blog](https://www.pinecone.io/learn/hybrid-search-intro/) |
| BM25 explained | [Elastic BM25 post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) |

### 8. LLM APIs & Agentic AI

| Topic | Resource |
|---|---|
| Google Gemini API | [Gemini API docs](https://ai.google.dev/gemini-api/docs) |
| pydantic-ai (agent framework) | [pydantic-ai docs](https://ai.pydantic.dev/) |
| Tool calling / function calling concept | [Anthropic tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) |
| Agentic AI patterns | [LangChain agents intro](https://python.langchain.com/docs/concepts/agents/) |

### 9. Docker & Containerization

| Topic | Resource |
|---|---|
| Docker fundamentals | [Docker official get-started](https://docs.docker.com/get-started/) |
| Docker for Python apps | [Real Python — Docker](https://realpython.com/docker-in-action-faq/) |
| docker-compose | [Compose docs](https://docs.docker.com/compose/) |

### 10. Hugging Face Spaces + CI/CD

| Topic | Resource |
|---|---|
| HF Spaces Docker SDK | [HF Spaces Docker docs](https://huggingface.co/docs/hub/spaces-sdks-docker) |
| GitHub Actions basics | [GitHub Actions quickstart](https://docs.github.com/en/actions/quickstart) |

---

## Suggested Learning Order

1. **Python async + type hints** — foundation for everything
2. **FastAPI** — build the server first
3. **Pydantic v2** — used everywhere for data models
4. **MongoDB basics → Atlas Vector Search → Atlas Search** — the hardest piece, takes most time
5. **Embeddings + sentence-transformers** — understand the math conceptually, then the library
6. **RAG pattern + hybrid search + RRF** — the intellectual core of the app
7. **Gemini API + pydantic-ai** — add the generation layer last
8. **PDF / OCR tools** — read PyMuPDF and Mistral OCR docs as needed
9. **Docker + HF Spaces** — deploy once everything works locally
