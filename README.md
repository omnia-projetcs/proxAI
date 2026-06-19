# proxAI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenAI API](https://img.shields.io/badge/API-OpenAI%20compatible-412991)](https://platform.openai.com/docs/api-reference)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Uvicorn](https://img.shields.io/badge/Server-Uvicorn-000000?logo=uvicorn)](https://www.uvicorn.org/)

**Transparent OpenAI-compatible proxy** for Ollama, vLLM, **NVIDIA NIM**, **Groq**, **Mistral**, and other cloud providers.

One endpoint for all your LLM clients (OpenAI SDK, Cursor, LangChain, Continue, etc.).

---

## GitHub topics

Suggested tags for this repository:

`llm` `proxy` `openai-api` `ollama` `vllm` `nvidia-nim` `groq` `mistral` `fastapi` `python` `ai-gateway` `llm-router`

---

## Installation

**Requirements:** Python 3.10+

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd proxAI
```

### 2. Create and activate a virtual environment (venv)

A venv isolates proxAI dependencies from your system Python.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (cmd)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your shell prompt. To leave the venv later:

```bash
deactivate
```

### 3. Install proxAI

**Option A — `requirements.txt` (recommended)**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B — `pyproject.toml` (editable install)**

```bash
pip install --upgrade pip
pip install -e .
```

> Always activate the venv before running `pip install` or `proxai`.

### Dependencies

| Package | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP API server |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server (multi-worker) |
| [HTTPX](https://www.python-httpx.org/) | Async HTTP client to backends |
| [Pydantic](https://docs.pydantic.dev/) | Request/config validation |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` file loading |

---

## Quick start

```bash
source .venv/bin/activate      # activate venv if not already active
cp .env_example .env           # add your API keys
proxai
```

Server runs at **http://localhost:8080** by default.

Alternative run (without CLI entry point):

```bash
source .venv/bin/activate
python -m proxai.main --port 8080
```

---

## Configuration

Copy `.env_example` to `.env` and set your keys:

```env
# Default model when client sends model=default
DEFAULT_MODEL=ollama/llama3.2

# Local backends
OLLAMA_BASE_URL=http://localhost:11434
VLLM_BASE_URL=http://localhost:8000

# Cloud providers (fill only what you use)
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...
```

| Variable | Description |
|---|---|
| `DEFAULT_MODEL` | Model used for `default` / `auto` requests |
| `WORKERS` | Number of uvicorn worker processes (default: `4`) |
| `MAX_CONCURRENT_REQUESTS` | Max parallel requests per worker |
| `ENABLED_PROVIDERS` | Limit model listing, e.g. `ollama,groq,mistral,nvidia` |

---

## Usage

### Point any OpenAI client to proxAI

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=dummy    # not required for Ollama / vLLM
```

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# Use .env default model
client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Explicit provider routing
client.chat.completions.create(
    model="groq/llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
)

client.chat.completions.create(
    model="mistral/mistral-large-latest",
    messages=[{"role": "user", "content": "Hello!"}],
)

client.chat.completions.create(
    model="nvidia/meta/llama3-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### curl

```bash
# Chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}]}'

# List all models
curl http://localhost:8080/v1/models

# Health check
curl http://localhost:8080/health
```

---

## Model routing

| Format | Example | Backend |
|---|---|---|
| `provider/model` | `ollama/llama3.2` | Ollama |
| `provider/model` | `groq/llama-3.3-70b-versatile` | Groq |
| `provider/model` | `mistral/mistral-large-latest` | Mistral |
| `provider/model` | `nvidia/meta/llama3-8b-instruct` | NVIDIA NIM |
| `provider/model` | `openai/gpt-4o` | OpenAI |
| `default` | — | Uses `DEFAULT_MODEL` from `.env` |

Auto-detection (no prefix) works for common names: `gpt-*` → OpenAI, `claude-*` → Anthropic, `mistral-*` → Mistral, `llama-3*` → Groq, `meta/llama*` → NVIDIA.

---

## Supported providers

| Provider | Type | API key |
|---|---|---|
| Ollama | Local | No |
| vLLM | Local | No |
| **NVIDIA NIM** | Cloud / local | `NVIDIA_API_KEY` |
| **Groq** | Cloud | `GROQ_API_KEY` |
| **Mistral** | Cloud | `MISTRAL_API_KEY` |
| OpenAI | Cloud | `OPENAI_API_KEY` |
| Anthropic | Cloud | `ANTHROPIC_API_KEY` |
| Google Gemini | Cloud | `GOOGLE_API_KEY` |
| DeepSeek | Cloud | `DEEPSEEK_API_KEY` |
| Together AI | Cloud | `TOGETHER_API_KEY` |
| OpenRouter | Cloud | `OPENROUTER_API_KEY` |
| Azure OpenAI | Cloud | `AZURE_API_KEY` |
| Cohere | Cloud | `COHERE_API_KEY` |
| Fireworks AI | Cloud | `FIREWORKS_API_KEY` |
| Perplexity | Cloud | `PERPLEXITY_API_KEY` |
| xAI (Grok) | Cloud | `XAI_API_KEY` |

### NVIDIA NIM

- **Cloud**: `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`
- **Local NIM**: `NVIDIA_BASE_URL=http://localhost:8000/v1` (no key required if running locally)

### Groq

Fast inference for open models. Example models: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

### Mistral

Official Mistral API. Example models: `mistral-large-latest`, `mistral-small-latest`, `codestral-latest`.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status |
| `GET` | `/v1/models` | List all available models |
| `GET` | `/v1/models/{id}` | Get model details |
| `POST` | `/v1/chat/completions` | Chat (streaming supported) |
| `POST` | `/v1/embeddings` | Embeddings |

---

## CLI options

```bash
proxai                          # start server
proxai --port 9000 --workers 8  # custom port and workers
proxai --reload                 # dev mode (auto-reload, 1 worker)
```

---

## License

MIT