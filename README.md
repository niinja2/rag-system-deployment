# Production RAG System

A retrieval-augmented generation system built with FastAPI, FAISS, and cross-encoder reranking. Supports multiple LLM backends (local via Ollama, cloud via OpenRouter).

## Scope

This repository demonstrates a complete RAG pipeline end to end: retrieval, reranking, pluggable LLM backends, an API and a UI. It is meant for running and inspecting the pipeline, not for serving users: beyond request validation there is no error handling, rate limiting or test suite. A service-oriented counterpart with input validation, error handling and tests is [document-qa](https://github.com/niinja2/document-qa).

## Architecture

```
Query → FAISS Retrieval (top 20) → Cross-Encoder Reranking (top 5) → LLM Generation → Response
```

- **Retrieval**: Sentence-transformers (BGE-base) encodes the query, FAISS searches 100K MS MARCO passages
- **Reranking**: Cross-encoder rescores retrieved chunks for relevance
- **LLM**: Pluggable backends — Ollama (local), OpenRouter (cloud), or mock for testing
- **API**: FastAPI with Pydantic validation
- **UI**: Streamlit frontend with provider switching

## Data Generation

The deployment uses a pre-built FAISS index — the embedding pipeline runs offline, separate from the API.

**Source:** MS MARCO passage corpus (~8.84M documents)

**Process:**
1. Raw passages loaded from JSONL
2. Encoded into 768-dimensional vectors using `BAAI/bge-base-en-v1.5` (SentenceTransformer)
3. GPU-accelerated inference on AMD RX 6700 XT (12GB VRAM) via DirectML
4. Vectors L2-normalized (cosine similarity = inner product)
5. Processed in 100K batches for fault tolerance (up to ~330 docs/sec)
6. Indexed with FAISS IndexFlatIP

```
MS MARCO corpus (8.84M passages)
        │
        ▼
  SentenceTransformer (BGE-base, 768-dim)
        │
        ▼
  GPU embedding (DirectML, batched)
        │
        ▼
  Normalized vectors → FAISS IndexFlatIP
        │
        ▼
  Deployed index (100K subset served by API)
```

The deployment serves a 100K-document subset. The full embedding pipeline, model benchmarks, and evaluation code live in the companion research repository.

## Data

The FAISS index and passage corpus are not included in this repository. Download them from Hugging Face before running the API:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='niinja2/rag-system-data', repo_type='dataset', local_dir='.', allow_patterns=['data/*'])"
```

This will populate:

```
data/index/     # FAISS index (bge_base_100k.index)
data/ids/       # Chunk ID mappings
data/processed/ # MS MARCO passages (msmarco_100k.jsonl)
```

Without these files the API does not start: it loads the index at startup and stops with a file-not-found error. The UI then shows "API not reachable".

## Project Structure

```
api/app.py              # FastAPI application and endpoints
llm/rag_chain.py        # RAG pipeline orchestrator
llm/prompt_builder.py   # Prompt construction
llm/ollama_llm.py       # Ollama backend (local)
llm/openai_llm.py       # OpenRouter backend (cloud)
llm/mock_llm.py         # Mock backend (testing)
retrieval/query.py      # FAISS retriever
reranking/cross_encoder.py  # Cross-encoder reranker
prompts/                # System prompt templates
data/index/             # FAISS index
data/ids/               # Chunk ID mappings
data/processed/         # MS MARCO passages
ui.py                   # Streamlit frontend
Dockerfile              # Image used by both containers
docker-compose.yml      # API + UI
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/niinja2/rag-system-deployment.git
cd rag-system-deployment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Download data

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='niinja2/rag-system-data', repo_type='dataset', local_dir='.', allow_patterns=['data/*'])"
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 4. Run the API

```bash
python -m uvicorn api.app:app --reload
```

API available at `http://localhost:8000/docs`

### 5. Run the UI (optional)

In a separate terminal:

```bash
python -m streamlit run ui.py
```

UI available at `http://localhost:8501`

## Docker

Runs the API and the UI as two containers. Before starting, download the data (step 2 above) and create `.env` (step 3 above); the `data/` folder is mounted into the API container, not copied into the image.

```bash
docker compose up --build
```

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

Notes:
- The image is about 3.5 GB (CPU-only PyTorch); the build needs additional space for its cache.
- The embedding and reranker models are downloaded into the image at build time, so no download happens at runtime.
- The API loads the index and both models at startup. Until it is ready, the UI shows an "API loading..." spinner.
- In Docker use the `openrouter` or `mock` provider. The `ollama` provider expects Ollama on `localhost`, which inside a container is the container itself, so it works only when the API runs outside Docker.

## LLM Providers

| Provider | Name in API | Model | Requires |
|----------|------------|-------|----------|
| Mock | `mock` | - | Nothing |
| Ollama | `ollama` | gemma4:e2b | Ollama running locally, model pulled with `ollama pull gemma4:e2b` |
| OpenRouter | `openrouter` | mistral-small-3.1-24b | `OPENROUTER_API_KEY` in `.env` |

To use the `ollama` provider, install [Ollama](https://ollama.com), start it, and pull the model once:

```bash
ollama pull gemma4:e2b
```

The model name is the default in `llm/ollama_llm.py`; change it there to use another local model. Larger local models answer noticeably slower on a machine without a strong GPU. The UI selects `ollama` first in its provider list, so pick `mock` or `openrouter` if Ollama is not set up.

## API Usage

```bash
POST /query
{
  "query": "What is a black hole?",
  "llm_provider": "openrouter",
  "use_reranker": true,
  "top_k": 5,
  "retrieve_k": 20,
  "min_score": 0.3
}
```

Response:

```json
{
  "query": "What is a black hole?",
  "answer": "A black hole is...",
  "sources": ["89735", "89737"],
  "contexts": ["Black Holes. Don't let the name fool you..."],
  "model": "mistralai/mistral-small-3.1-24b-instruct",
  "server": "cloud"
}
```

## Companion Repository

Research, benchmarks, and the GPU embedding pipeline: [rag-system-research](https://github.com/niinja2/rag-system-research)
