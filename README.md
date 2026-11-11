# DALIL | دليل

Arabic document investigation with a FastAPI backend, FAISS retrieval, and an Arabic-first web interface. OpenAI generation is optional for local retrieval testing and requires a server-side key for generated answers.

> **Azure pilot is not yet verified.** The app fails closed unless its App Service identity and mounted storage checks pass. The Azure mount, real sign-in, backup recovery, and real OpenAI end-to-end test must be verified on the actual deployment before relying on it. Do not expose a development instance publicly.

## Local development

Use Python 3.12.3, create a virtual environment, and install the pinned CPU dependencies. The first model load needs network access unless the model is already cached.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set `OPENAI_API_KEY` only in the server environment, never in a browser or Git. No key is required to exercise retrieval-only behavior. The lock is for the FastAPI/HTML application; the legacy Gradio UI, Gemini provider, and full CAMeL analyzer need their optional packages separately and are not part of the tested release environment.

For local testing, bind to loopback and explicitly enable both development flags:

```bash
export LOCAL_DEV_MODE=true
export PRIVATE_DEV_MODE=true
export AUTH_WORKSPACE_ID=local-user
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. In Codespaces, only use these flags with a **private** forwarded port. The flags and shared development bearer token are not production identity. Keep `ALLOW_EMBEDDING_FALLBACK=false` except for isolated tests; hashed vectors are not interchangeable with model embeddings.

## Embeddings and index safety

The default model is `distiluse-base-multilingual-cased-v2`, with a 128-token model sequence limit and 512-dimensional embeddings. Its default revision is pinned to `bfe45d0732ca50787611c0fe107ba278c7f3f889`. The loader uses the active model's fast tokenizer, includes model prompts and special tokens when enforcing the limit, and keeps PDF page numbers. It checks source-character coverage for each usable page. The application fails closed if an existing nonempty index has no model manifest, or if its model, pinned revision, vector dimension, or schema differs.

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

To evaluate any staged document, keep a local JSON case file outside the repository. Each entry identifies the staged document and expected **PDF** page numbers:

```json
[{"id":"example", "document_id":"DOCUMENT_ID", "question":"ما موضوع المستند؟", "expected_pages":[1]}]
```

```bash
python -m app.evaluate_index --index /path/to/staging/index \
  --documents /path/to/staging/documents --workspace local-user --cases /path/to/cases.json
```

The evaluator uses real configured embeddings, checks whether an expected page appears in the top results, and compares the retrieved passage offsets with the original PDF extraction. It calls no LLM and never activates the index. A passing small test set does not establish answer grounding or general retrieval quality.

## Tests

```bash
python -m pytest -q
```

The CI workflow installs the same locked requirements on Python 3.12.3. Docker uses the CPU lock and excludes local data and secrets from its build context; a container build must still be verified on the deployment host.

Tests that use mock embeddings or a NumPy fallback do not validate the real model, FAISS, an external LLM, cloud persistence, or production authentication. Keep those checks separate when reporting results.

## Private Azure pilot

The pilot uses a Linux custom container, exactly one App Service instance and one Uvicorn worker. Configure App Service authentication to require Microsoft Entra ID sign-in for every path and HTTPS, with no excluded paths. Restrict the application registration to the chosen tenant and restrict access to the pilot's approved user object IDs. The application checks both the tenant and user ID from App Service's injected identity claims; it derives a separate workspace from those IDs. Never forward identity headers from a separately exposed reverse proxy.

Mount an Azure Files share at a dedicated mount point (for example `/mounts/dalil`) and configure `APP_ENV=production`, `DALIL_DATA_MOUNT=/mounts/dalil`, `INDEX_PATH=/mounts/dalil/index`, and `DOCS_PATH=/mounts/dalil/documents`. Set `DALIL_TENANT_ID` and comma-separated `DALIL_ALLOWED_USER_IDS` to the tenant ID and approved Entra user **object IDs**, respectively. Set `LOCAL_DEV_MODE=false`, `PRIVATE_DEV_MODE=false`, `ALLOW_EMBEDDING_FALLBACK=false`, and inject `OPENAI_API_KEY` through an Azure app setting or Key Vault reference; never commit the key. App Service supplies `WEBSITE_AUTH_ENABLED`, `WEBSITE_AUTH_V2_CONFIG_JSON`, and `WEBSITE_HOSTNAME`.

The app verifies the mounted path and authentication settings during startup, holds an exclusive filesystem lock for its lifetime, and saves index updates to a new generation before switching the `CURRENT` pointer. Earlier generations are retained as local recovery copies; configure separate Azure Files backups and test restoration. The lock, flush and rename behavior must be tested on the actual Azure Files mount, and the service must not be scaled to multiple instances or workers. Do not switch an old index or test document into production without an explicit migration and access review.

## API

The main endpoints are `GET /health`, `GET /documents`, `POST /documents/upload`, `GET /documents/{id}/file`, `POST /query`, and `DELETE /documents/{id}`. The development API docs are at `/docs`. Uploads are scoped to the configured development workspace; this is not multi-user production authentication.
