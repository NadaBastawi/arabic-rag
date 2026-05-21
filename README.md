# Ar-RAG — Arabic Retrieval-Augmented Generation System

An end-to-end Arabic RAG system that combines multi-stage retrieval, morphology-aware preprocessing, and LLM-powered generation to answer questions over Arabic document collections. The system handles the linguistic complexity of Arabic (root patterns, diacritics, morphological variation) through specialized preprocessing, retrieves relevant documents using FAISS vector search with learned reranking, and generates answers via Google Gemini. Packaged as a FastAPI backend with a Gradio chat interface and deployed via Docker.


## Features

- **Multi-stage Arabic retrieval** — Dense embedding retrieval followed by learned reranking for relevance
- **Arabic morphology-aware preprocessing** — Lemmatization, diacritic handling, and root-based normalization via camel-tools
- **FAISS vector store with NumPy fallback** — Fast approximate nearest neighbor search with graceful degradation for offline/testing scenarios
- **FastAPI backend** — Full REST API for indexing documents and querying the RAG pipeline
- **Gradio chat UI** — Interactive web interface for conversational document Q&A
- **Dockerized deployment** — Single-command deployment with docker-compose
- **Document ingestion** — Supports PDF and plain text uploads; automatic chunking and embedding

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  Arabic Preprocessing    │  (lemmatization, normalization)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Text Embedding          │  (multilingual-e5-large)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  FAISS Retrieval         │  (top-k dense search)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Multi-Stage Reranking   │  (learned relevance ranking)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Context Assembly        │  (selected docs + query)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  LLM Generation          │  (Google Gemini API)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Answer + Citations      │  (response + source refs)
└──────────────────────────┘
```

### Pipeline Components

- **arabic_preprocessing.py** — Text normalization (whitespace, punctuation, encoding)
- **arabic_morphology.py** — Lemmatization and morphological analysis using camel-tools
- **embedding_service.py** — Loads and caches multilingual-e5-large embeddings; includes fallback embeddings for offline environments
- **faiss_store.py** — FAISS index management with NumPy fallback vector store for development and testing
- **multi_stage_retrieval.py** — Orchestrates dense retrieval → reranking → top-k result filtering
- **rag_pipeline.py** — End-to-end RAG orchestration (preprocessing → retrieval → generation)
- **llm_service.py** — Google Gemini API integration with configurable temperature and token limits
- **main.py (FastAPI)** — REST API endpoints for indexing, querying, and document management
- **app.py (Gradio)** — Chat interface for interactive question-answering over documents

## Tech Stack

Python | FastAPI | Gradio | FAISS | LangChain | Google Gemini | intfloat/multilingual-e5-large | camel-tools | Docker

## Project Structure

```
app/
  __init__.py
  main.py                              # FastAPI application
  app.py                               # Gradio chat UI
  config.py                            # Environment-based configuration
  build_index.py                       # Offline batch indexing utility
  eval.py                              # Retrieval evaluation and analysis
  services/
    __init__.py
    embedding_service.py               # Text embedding + caching
    faiss_store.py                     # FAISS index + NumPy fallback
    llm_service.py                     # LLM API integration (Gemini)
    multi_stage_retrieval.py           # Dense retrieval + reranking
    rag_pipeline.py                    # Full RAG pipeline orchestration
    cache.py                           # Caching layer for embeddings
    vector_store.py                    # Abstract vector store interface
    app/
      api.py                           # API route definitions
  utils/
    __init__.py
    arabic_preprocessing.py            # Text normalization for Arabic
    arabic_morphology.py               # Morphological lemmatization
    document_loader.py                 # PDF/TXT document ingestion
    errors.py                          # Custom exception classes
    logging_config.py                  # Structured logging configuration

data/
  index/
    index.faiss                        # FAISS index file
    metadata.json                      # Document metadata

tests/
  __init__.py
  test_rag_pipeline.py                 # End-to-end pipeline tests
  test_integration.py                  # Integration tests

docker-compose.yml                     # Multi-container orchestration
Dockerfile                             # Production image definition
requirements.txt                       # Python dependencies
.env.example                           # Configuration template
.gitignore                             # Git ignore rules
README.md                              # This file
```

## Quick Start — Local

### Prerequisites
- Python 3.10+
- pip
- (Optional) GPU with CUDA for faster embeddings

### Setup

1. **Clone and create virtual environment:**
   ```bash
   git clone https://github.com/NadaBastawi/arabic-rag.git
   cd arabic-rag
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and set LLM_API_KEY (Google Gemini API key)
   ```

4. **Run the API (Terminal 1):**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

5. **Run the UI (Terminal 2):**
   ```bash
   python -m gradio app.app
   ```
   - Chat UI: http://localhost:7860

## Quick Start — Docker

```bash
docker compose up --build
```

- **API:** http://localhost:8000/docs
- **Chat UI:** http://localhost:7860

Ensure `.env` is in the project root with `LLM_API_KEY` set.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/index` | Add raw text documents (plain text + metadata) |
| POST | `/query` | Query the RAG pipeline; returns answer + retrieved documents |
| POST | `/documents/upload` | Upload PDF or TXT files for automatic ingestion |
| GET | `/documents` | List all indexed documents with metadata |
| DELETE | `/documents/{document_id}` | Delete a specific document from the index |
| POST | `/documents/clear` | Clear all documents and reset the index |
| GET | `/stats` | Retrieve index size, cache hit rate, and retrieval statistics |
| GET | `/health` | Health check endpoint |

### Example: Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما هي الخطوات الأساسية لبناء نموذج تعلم آلي؟",
    "top_k": 5,
    "return_docs": true
  }'
```

## Configuration

All configuration is environment-based. Create a `.env` file (or copy `.env.example`) with these variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | LLM model name (Gemini) | `gemini-2.0-flash` |
| `LLM_API_KEY` | Google Gemini API key | (required) |
| `LLM_TEMPERATURE` | Response creativity (0–1) | `0.3` |
| `LLM_MAX_TOKENS` | Max tokens per response | `1024` |
| `EMBEDDING_MODEL` | Text embedding model | `intfloat/multilingual-e5-large` |
| `EMBEDDING_CACHE_SIZE` | Embedding cache entries | `10000` |
| `RETRIEVAL_MODE` | Retrieval strategy | `multi_stage` |
| `TOP_K_RETRIEVAL` | Number of documents to retrieve | `5` |
| `RERANKER_TOP_K` | Number of documents after reranking | `3` |
| `INDEX_PATH` | Path to FAISS index | `data/index/` |
| `CHUNK_SIZE` | Document chunk size (tokens) | `512` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |

## Running Tests

```bash
python -m pytest -q
```

**Note:** On Windows with OneDrive, if you encounter permission errors on `.pyc` files:
```bash
set PYTHONDONTWRITEBYTECODE=1
python -m pytest -q
```

## Design Decisions

**Why FAISS with NumPy fallback?**
FAISS provides fast approximate nearest neighbor search at scale (millions of vectors in milliseconds). The NumPy fallback ensures the system works offline and during development/testing without external dependencies, trading speed for reliability.

**Why multilingual-e5-large for Arabic embeddings?**
This model is multilingual and trained on 100+ languages including Arabic. It outperforms language-specific models on cross-lingual tasks and handles code-switching (common in social media Arabic). The large variant provides strong semantic understanding without excessive latency.

**Why multi-stage retrieval + reranking instead of single-pass?**
Two-stage retrieval maximizes recall (dense search casts a wide net) while reranking optimizes precision (learned ranking identifies genuinely relevant documents). This approach prevents missing relevant documents while ensuring the LLM receives only high-confidence context, improving answer quality and reducing token usage.

## Roadmap

- **Session-based conversation memory** — Maintain multi-turn conversation context with session persistence
- **API authentication layer** — JWT-based authentication for multi-user deployments and usage tracking
- **Source attribution** — Automatic in-line citations with document references and chunk indices
- **Observability and monitoring** — Request logging, latency metrics, embedding cache performance, retrieval quality scores
- **Evaluation framework** — Automated evaluation suite for answer quality, retrieval F1, and semantic similarity

## License

MIT
