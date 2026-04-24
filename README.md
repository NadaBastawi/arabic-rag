# Arabic RAG (Production Starter)

Arabic RAG system with:
- FastAPI backend (`app/main.py`)
- Gradio chatbot UI (`app/app.py`)
- FAISS/NumPy vector store (`app/services/faiss_store.py`)
- Multi-stage Arabic retrieval and reranking

This README is the single source of truth for setup and usage.

## 1) Quick Start (Local)

### Prerequisites
- Python 3.10+ (3.11 recommended)
- `pip`
- Optional GPU (CPU works)

### Install
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Configure
Create `.env` (or copy `.env.example`) and set at least:
```env
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=your_real_key_here
EMBEDDING_MODEL=intfloat/multilingual-e5-large
RETRIEVAL_MODE=multi_stage
```

### Run API
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open:
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Run Chat UI
In another terminal:
```bash
python -m gradio app.app
```
Default UI port: `7860`

## 2) Quick Start (Docker)

```bash
docker compose up --build
```

Open:
- API: `http://localhost:8000`
- UI: `http://localhost:7860`

## 3) Minimal Test Commands

Run tests:
```bash
python -m pytest -q
```

If OneDrive blocks `.pyc` writes:
```bash
set PYTHONDONTWRITEBYTECODE=1
python -m pytest -q
```

## 4) API Endpoints You Will Use Most

- `POST /index` add raw text documents
- `POST /query` ask question + get answer + retrieved docs
- `POST /documents/upload` upload `pdf/txt`
- `GET /documents` list indexed documents
- `DELETE /documents/{document_id}` delete one document
- `POST /documents/clear` clear all
- `GET /stats` index/cache stats

## 5) Project Structure (Keep This)

```text
app/
  main.py                    # FastAPI app (production API)
  app.py                     # Gradio chatbot UI
  config.py                  # env-based config
  services/
    embedding_service.py
    llm_service.py
    rag_pipeline.py
    faiss_store.py
    multi_stage_retrieval.py
    cache.py
  utils/
    document_loader.py
    arabic_morphology.py
    arabic_preprocessing.py
    errors.py
    logging_config.py
tests/
  test_rag_pipeline.py
  test_integration.py
requirements.txt
Dockerfile
docker-compose.yml
.env.example
README.md
```

## 6) Optional Utilities (Can Keep or Remove)

- `app/build_index.py` (offline indexing utility)
- `app/eval.py` (retrieval evaluation utility)

If your team does not use them, delete both to simplify the repo.

## 7) Common Production Notes

- `EmbeddingService` includes fallback embeddings for offline/dev environments.
- `FAISSVectorStore` can fall back to NumPy backend if FAISS is unavailable.
- Keep `INDEX_PATH` on persistent storage in production.
- Never commit real API keys.

## 8) Next Steps to Make It a Real Testable Chatbot Product

1. Create a stable test dataset of real Arabic documents (50-200 docs).
2. Add source citation in final answer text (document id + chunk snippet).
3. Add authentication to API (`/query`, `/documents/*`).
4. Add conversation memory (session id + message history store).
5. Add observability (request logs, latency, token usage, retrieval scores).
6. Run load test (concurrent users) and define SLAs.
7. Package one-click launch (Docker + `.env` template + sample docs).
8. Add human evaluation loop for answer quality before release.

## 9) Troubleshooting

- Model download blocked/network issue: fallback embeddings are used automatically.
- Missing `fastapi` or `gradio`: reinstall with `pip install -r requirements.txt`.
- OneDrive permission errors on `__pycache__`: use `PYTHONDONTWRITEBYTECODE=1` while testing.
