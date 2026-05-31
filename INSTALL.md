# LegalRo — Installation Guide

Two ways to run LegalRo:

| Mode | When to use |
|---|---|
| **Remote client** (recommended) | Use the live cloud deployment — no local models or database |
| **Local dev** | Run everything on macOS Apple Silicon for offline development |

---

## Remote client (recommended)

The CLI talks to the HF Space. All processing runs in the cloud.

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Credentials (`.env` file — see below)

### Steps

```bash
# 1. Clone
git clone https://github.com/raulrotar/legalRo-cloud.git
cd legalRo-cloud

# 2. Install (lightweight — no local model deps)
uv sync --no-dev

# 3. Configure
cp .env.example .env
```

Edit `.env` and fill in:

```dotenv
LEGALRO_API_URL=https://rraul99-legalro.hf.space
LEGALRO_API_TOKEN=<token>        # get this from the project owner
HF_TOKEN=<hf_read_token>         # generate at huggingface.co/settings/tokens → Read
```

```bash
# 4. Verify connection
uv run legalro status

# 5. Use the app
uv run legalro query "Ce drepturi are un salariat?"
uv run legalro chat
uv run legalro ingest path/to/gazette.pdf
```

That's it. No API keys, no local models, no Docker.

---

## Local dev mode (macOS Apple Silicon)

Runs everything locally: MLX LLM, local MongoDB, Apple Vision OCR.

### System requirements

| Requirement | Notes |
|---|---|
| macOS 13 Ventura+ | Apple Silicon (M1/M2/M3/M4) |
| Python 3.12+ | Managed by uv |
| RAM 16 GB | ~6 GB used by LLM at runtime |
| Disk 10 GB free | ~5 GB model weights |
| Docker Desktop | For local MongoDB |
| Xcode Command Line Tools | Required by ocrmac (Vision framework) |

### Step 1 — Xcode CLT

```bash
xcode-select --install
```

### Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 3 — Docker Desktop

Download from docker.com/products/docker-desktop, install, and start it.

### Step 4 — Clone and install

```bash
git clone https://github.com/raulrotar/legalRo-cloud.git
cd legalRo-cloud

# Install with local extras (ocrmac, MLX)
uv sync --no-dev --extra local
```

### Step 5 — Configure

```bash
cp .env.example .env
```

Fill in `.env`:

```dotenv
MONGODB_URI=mongodb://localhost:27018/?directConnection=true
GEMINI_API_KEY=your_gemini_key
```

### Step 6 — Start services

```bash
uv run legalro start
```

This starts MongoDB (Docker) and the MLX LLM server as a background process. On first run, model weights (~4 GB) are downloaded automatically.

### Step 7 — Create search indexes (first time only)

```bash
uv run python scripts/setup_indexes.py
```

### Step 8 — Verify

```bash
uv run legalro status
```

Expected output:
```
┌───────────┬─────────────┬───────────────────────────────────────────┐
│ Component │ Status      │ Details                                   │
├───────────┼─────────────┼───────────────────────────────────────────┤
│ MongoDB   │ ✓ Connected │ 0 gazettes, 0 acts, 0 chunks              │
│ LLM       │ ✓ Running   │ mlx-community/Qwen3.5-9B-4bit (pid XXXX) │
└───────────┴─────────────┴───────────────────────────────────────────┘
```

### Step 9 — Ingest and query

```bash
# Place PDFs under laws/ following naming: MO_PI_{issue}_{YYYY-MM-DD}.pdf
uv run legalro ingest laws/ --local
uv run legalro query "..." --local
uv run legalro chat --local
```

### Stopping

```bash
uv run legalro stop
```

Kills the LLM server and stops MongoDB. Data is preserved in the Docker volume.

---

## Troubleshooting

**MongoDB: "No replica set members match selector"**
Add `?directConnection=true` to `MONGODB_URI`. The local container runs a replica set.

**MongoDB: port 27018 already in use**
```bash
docker compose down
docker compose up -d
```

**LLM server slow to start**
Model is loading. Check `/tmp/legalro_mlx.log`. First run downloads ~4 GB.

**ocrmac fails on scanned PDFs**
Run `xcode-select -p` — Xcode CLT must be installed. The Vision framework is macOS-only.

**"Filename doesn't match expected pattern"**
Rename to `MO_PI_{issue}_{YYYY-MM-DD}.pdf`. The `Bis` suffix (e.g. `294Bis`) is supported.

**401 Unauthorized from HF Space**
- Check `LEGALRO_API_TOKEN` in `.env` matches `API_TOKEN` in HF Space secrets exactly.
- Check `HF_TOKEN` is a valid read token for the Space owner's account.
