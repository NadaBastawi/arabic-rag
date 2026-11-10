# DALIL | دليل

Arabic document investigation with a FastAPI backend, FAISS retrieval, and an Arabic-first web interface. OpenAI generation is optional for local retrieval testing and requires a server-side key for generated answers.

> **Not production-ready.** Production identity, durable storage, safe concurrent writes, and a real OpenAI end-to-end test have not been verified. The application refuses startup on Azure App Service or when `APP_ENV=production`. Do not expose a development instance publicly.

## Local development

Use Python 3.12, install the project's pinned dependencies, and copy `.env.example` to `.env`. Set `OPENAI_API_KEY` only in the server environment, never in a browser or Git. No key is required to exercise retrieval-only behavior.

For local testing, bind to loopback and explicitly enable both development flags:

```bash
export LOCAL_DEV_MODE=true
export PRIVATE_DEV_MODE=true
export AUTH_WORKSPACE_ID=local-user
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. In Codespaces, only use these flags with a **private** forwarded port. The flags and shared development bearer token are not production identity. Keep `ALLOW_EMBEDDING_FALLBACK=false` except for isolated tests; hashed vectors are not interchangeable with model embeddings.

## Embeddings and index safety

The default model is `distiluse-base-multilingual-cased-v2`, with a 128-token model sequence limit and 512-dimensional embeddings. The loader uses the active model's fast tokenizer, includes model prompts and special tokens when enforcing the limit, and keeps PDF page numbers. It checks source-character coverage for each usable page. The application fails closed if an existing nonempty index has no model manifest, or if its model, vector dimension, or schema differs.

An older local configuration may name `intfloat/multilingual-e5-large`; its 1,024-dimensional index has no manifest, so the model that created it cannot be verified from that index. E5 is blocked until an explicit migration implements and evaluates its distinct `query:` and `passage:` inputs. Do not point DistilUSE at that index.

## Stage and evaluate a PDF index

Create a **new** directory outside the configured index and documents directories. This does not switch the live index:

```bash
python -m app.reindex_pdf \
  --pdf /path/to/document.pdf \
  --output /path/to/new-staging-directory \
  --workspace local-user
```

The staged directory contains `index/` and `documents/`; the latter contains a copy of the source PDF under the workspace and generated document ID. Validate its manifest, page metadata, retrieval quality, and evidence viewer before considering a switch. A switch requires setting **both** `INDEX_PATH` and `DOCS_PATH` to those staged paths. Preserve the old directories and do not activate an index built for a different workspace.

PDF text extraction can be imperfect. The extraction report identifies pages that require OCR; additional pages can contain corrupted characters. Citations are retrieved passages and should be checked against the original PDF.

## Tests

```bash
python -m pytest -q
```

Tests that use mock embeddings or a NumPy fallback do not validate the real model, FAISS, an external LLM, cloud persistence, or production authentication. Keep those checks separate when reporting results.

## API

The main endpoints are `GET /health`, `GET /documents`, `POST /documents/upload`, `GET /documents/{id}/file`, `POST /query`, and `DELETE /documents/{id}`. The development API docs are at `/docs`. Uploads are scoped to the configured development workspace; this is not multi-user production authentication.
