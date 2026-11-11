"""Primary FastAPI application for the Arabic RAG service.

Production and staging use bearer authentication. Tokenless access is limited
to an explicitly enabled private development environment.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Lock
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSVectorStore, sync_directory
from app.services.llm_service import LLMService
from app.services.production_security import (
    single_writer, validate_browser_write, validate_production_config, workspace_for_request,
)
from app.services.rag_pipeline import RAGPipeline
from app.utils.document_loader import DocumentLoader
from app.utils.errors import InputValidator, ValidationError

logger = logging.getLogger(__name__)

embedding_service: Optional[EmbeddingService] = None
llm_service: Optional[LLMService] = None
vector_store: Optional[FAISSVectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
document_loader: Optional[DocumentLoader] = None

security = HTTPBearer(auto_error=False)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_pdf_render_lock = Lock()


class IndexRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class AddDocumentRequest(BaseModel):
    text: str
    document_id: Optional[str] = None
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Optional[str] = None
    document_id: Optional[str] = None


class DocumentInfo(BaseModel):
    document_id: str
    source: str
    chunk_count: int
    added_at: str


def _services() -> tuple[EmbeddingService, LLMService, FAISSVectorStore, RAGPipeline]:
    global embedding_service, llm_service, vector_store, rag_pipeline, document_loader
    if embedding_service is None:
        embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
            revision=config.effective_embedding_revision,
            allow_fallback=config.ALLOW_EMBEDDING_FALLBACK,
        )
    if llm_service is None:
        llm_service = LLMService(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            api_key=config.effective_llm_api_key,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    if vector_store is None:
        vector_store = FAISSVectorStore(
            index_path=config.INDEX_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            schema_version=config.INDEX_SCHEMA_VERSION,
            embedding_revision=config.effective_embedding_revision,
            atomic=config.is_production,
        )
        vector_store.ensure_embedding_dimension(
            embedding_service.get_embedding_dimension(), reset_on_mismatch=False
        )
    if document_loader is None:
        document_loader = DocumentLoader(chunk_size=500, chunk_overlap=50, **embedding_service.chunking_config())
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(
            embedding_service=embedding_service,
            llm_service=llm_service,
            vector_store=vector_store,
            top_k=config.RETRIEVAL_TOP_K,
            retrieval_mode=config.RETRIEVAL_MODE,
            initial_top_k=config.RETRIEVAL_INITIAL_K,
            use_adaptive=config.USE_ADAPTIVE,
        )
    return embedding_service, llm_service, vector_store, rag_pipeline


def get_services() -> tuple[EmbeddingService, LLMService, FAISSVectorStore, RAGPipeline]:
    """Backward-compatible service factory used by integrations and tests."""
    return _services()


def _current_workspace(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Authenticate a request and return its isolated workspace.

    An empty token is never accepted when authentication is enabled.  This
    avoids silently running a production deployment without an auth boundary.
    """
    if config.is_production:
        if config.LOCAL_DEV_MODE or config.PRIVATE_DEV_MODE or os.getenv("WEBSITE_AUTH_ENABLED", "").lower() != "true":
            raise HTTPException(status_code=503, detail="Production authentication is not configured")
        validate_browser_write(request)
        return workspace_for_request(request)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", config.AUTH_WORKSPACE_ID):
        raise HTTPException(status_code=500, detail="Authentication workspace is misconfigured")
    if config.LOCAL_DEV_MODE and config.PRIVATE_DEV_MODE:
        return config.AUTH_WORKSPACE_ID
    if not config.AUTH_ENABLED:
        return config.AUTH_WORKSPACE_ID
    if not config.AUTH_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured",
        )
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required")
    supplied = credentials.credentials.encode("utf-8")
    expected = config.AUTH_TOKEN.encode("utf-8")
    if not hmac.compare_digest(
        hashlib.sha256(supplied).digest(), hashlib.sha256(expected).digest()
    ):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return config.AUTH_WORKSPACE_ID


def _safe_source(filename: str) -> str:
    name = Path(filename or "upload").name
    if name in {".", ".."} or not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def _registry(workspace_id: str) -> dict[str, DocumentInfo]:
    assert vector_store is not None
    grouped: dict[str, DocumentInfo] = {}
    for item in vector_store.metadata:
        if item.get("workspace_id") != workspace_id:
            continue
        doc_id = str(item.get("document_id", ""))
        if not doc_id:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = DocumentInfo(
                document_id=doc_id,
                source=str(item.get("source", "unknown")),
                chunk_count=0,
                added_at=str(item.get("added_at", "")),
            )
        grouped[doc_id].chunk_count += 1
    return grouped


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting Arabic RAG API")
    if config.is_production:
        mount = validate_production_config(config)
        with single_writer(mount):
            get_services()
            yield
        logger.info("Shutting down Arabic RAG API")
        return
    get_services()
    yield
    logger.info("Shutting down Arabic RAG API")


app = FastAPI(title="Arabic RAG API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, _exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Invalid input"})


if Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


def _owned_document_path(document_id: str, workspace_id: str) -> Path:
    """Resolve a source only after validating the caller's workspace."""
    get_services()
    try:
        InputValidator.validate_document_id(document_id)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID") from exc
    matches = [
        item for item in vector_store.metadata
        if item.get("document_id") == document_id
        and item.get("workspace_id") == workspace_id
    ] if vector_store else []
    if not matches:
        raise HTTPException(status_code=404, detail="Document not found")
    suffix = Path(str(matches[0].get("source", ""))).suffix.lower()
    target = Path(config.DOCS_PATH, workspace_id, f"{document_id}{suffix}")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Document file is unavailable")
    return target


@app.get("/documents/{document_id}/file")
async def document_file(document_id: str, workspace_id: str = Depends(_current_workspace)) -> FileResponse:
    target = _owned_document_path(document_id, workspace_id)
    return FileResponse(target, media_type="application/pdf" if target.suffix.lower() == ".pdf" else None,
                        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/documents/{document_id}/pages")
async def document_pages(document_id: str, workspace_id: str = Depends(_current_workspace)) -> dict[str, int]:
    target = _owned_document_path(document_id, workspace_id)
    if target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Page viewing requires a PDF")
    try:
        import pypdfium2 as pdfium
        with _pdf_render_lock:
            document = pdfium.PdfDocument(str(target))
            try:
                return {"page_count": len(document)}
            finally:
                document.close()
    except Exception as exc:
        logger.warning("Unable to open a PDF for page viewing")
        raise HTTPException(status_code=422, detail="Unable to open PDF") from exc


@app.get("/documents/{document_id}/pages/{page_number}.png")
async def document_page_image(document_id: str, page_number: int,
                              workspace_id: str = Depends(_current_workspace)) -> Response:
    target = _owned_document_path(document_id, workspace_id)
    if target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Page viewing requires a PDF")
    try:
        import pypdfium2 as pdfium
        with _pdf_render_lock:
            document = pdfium.PdfDocument(str(target))
            try:
                if page_number < 1 or page_number > len(document):
                    raise HTTPException(status_code=404, detail="PDF page not found")
                page = document[page_number - 1]
                try:
                    width, height = page.get_size()
                    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                        raise HTTPException(status_code=422, detail="Unable to render PDF page")
                    scale = min(2.0, 1600.0 / max(width, height))
                    bitmap = page.render(scale=scale)
                    try:
                        output = BytesIO()
                        bitmap.to_pil().save(output, format="PNG")
                        content = output.getvalue()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            finally:
                document.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Unable to render a PDF page")
        raise HTTPException(status_code=422, detail="Unable to render PDF page") from exc
    return Response(content, media_type="image/png",
                    headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index")
async def index_texts(
    request: IndexRequest,
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    embedder, _, store, _ = get_services()
    texts = [InputValidator.validate_document(text) for text in request.texts]
    ids = [f"doc-{uuid.uuid4().hex}" for _ in texts]
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    store.add_documents(
        texts,
        embedder.embed(texts),
        document_ids=ids,
        sources=["api"] * len(texts),
        metadata_list=[{"workspace_id": workspace_id, "added_at": now} for _ in texts],
    )
    return {"indexed_count": len(texts), "total_documents": len(_registry(workspace_id))}


@app.post("/documents/add")
async def add_document(
    request: AddDocumentRequest,
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    embedder, _, store, _ = get_services()
    text = InputValidator.validate_document(request.text)
    requested_id = request.document_id.strip() if request.document_id else ""
    document_id = requested_id or f"doc-{uuid.uuid4().hex}"
    if requested_id:
        try:
            InputValidator.validate_document_id(requested_id)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if requested_id in _registry(workspace_id):
            raise HTTPException(status_code=409, detail="Document ID already exists")
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    protected = {"faiss_id", "text", "document_id", "source", "workspace_id", "added_at", "content_sha256"}
    metadata = {k: v for k, v in request.metadata.items() if k not in protected}
    metadata.update({"workspace_id": workspace_id, "added_at": now})
    store.add_documents(
        [text], embedder.embed([text]), [document_id], [Path(request.source).name], [metadata]
    )
    return {"document_id": document_id, "chunk_count": 1}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    source = _safe_source(file.filename or "")
    suffix = Path(source).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF, DOCX, and TXT files are supported")

    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Invalid PDF file")
    if suffix == ".docx" and not data.startswith(b"PK\x03\x04"):
        raise HTTPException(status_code=415, detail="Invalid DOCX file")
    embedder, _, store, _ = get_services()
    digest = hashlib.sha256(data).hexdigest()
    for item in store.metadata:
        if item.get("workspace_id") == workspace_id and item.get("content_sha256") == digest:
            existing_id = str(item["document_id"])
            existing = _registry(workspace_id)[existing_id]
            return {"document_id": existing_id, "source": existing.source,
                    "chunk_count": existing.chunk_count, "duplicate": True, "extraction": {}}
    Path(config.DOCS_PATH, workspace_id).mkdir(parents=True, exist_ok=True)
    document_id = uuid.uuid4().hex
    target = Path(config.DOCS_PATH, workspace_id, f"{document_id}{suffix}")
    with target.open("xb") as source_file:
        source_file.write(data)
        source_file.flush()
        os.fsync(source_file.fileno())
    if config.is_production:
        sync_directory(str(target.parent))
    try:
        chunks = document_loader.load_file(str(target)) if document_loader else []
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from the document")
        embedder, _, store, _ = get_services()
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        store.add_documents(
            [chunk.text for chunk in chunks],
            embedder.embed([chunk.text for chunk in chunks]),
            [document_id] * len(chunks),
            [source] * len(chunks),
            [
                {
                    "workspace_id": workspace_id,
                    "content_sha256": digest,
                    "chunk_id": chunk.chunk_id,
                    "page_num": chunk.page_num,
                    "added_at": now,
                    **{key: chunk.metadata[key] for key in
                       ("char_start", "char_end", "extraction_quality")
                       if key in chunk.metadata},
                }
                for chunk in chunks
            ],
        )
        report = document_loader.last_report if document_loader else {}
        return {
            "document_id": document_id,
            "source": source,
            "chunk_count": len(chunks),
            "extraction": {
                "page_count": report.get("page_count"),
                "indexed_page_count": report.get("indexed_page_count"),
                "ocr_required_pages": report.get("ocr_required_pages", []),
            },
        }
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception:
        # A failure after publishing a snapshot can still leave a committed
        # document. Keep its source whenever the recovered index references it.
        if not any(item.get("document_id") == document_id and item.get("workspace_id") == workspace_id
                   for item in store.metadata):
            target.unlink(missing_ok=True)
        logger.exception("Document upload processing failed")
        raise HTTPException(status_code=500, detail="Document processing failed")


@app.get("/documents")
async def list_documents(
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    documents = list(_registry(workspace_id).values())
    return {"documents": [item.model_dump() for item in documents], "total_count": len(documents)}


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    _, _, store, _ = get_services()
    try:
        InputValidator.validate_document_id(document_id)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not store.delete_document(document_id, workspace_id=workspace_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "document_id": document_id}


@app.post("/documents/clear")
async def clear_documents(workspace_id: str = Depends(_current_workspace)) -> dict[str, Any]:
    _, _, store, _ = get_services()
    ids = list(_registry(workspace_id))
    for document_id in ids:
        store.delete_document(document_id, workspace_id=workspace_id)
    return {"deleted_count": len(ids)}


@app.post("/query")
async def query(
    request: QueryRequest,
    workspace_id: str = Depends(_current_workspace),
) -> dict[str, Any]:
    _, _, store, pipeline = get_services()
    question = InputValidator.validate_query(request.question)
    docs = []
    if request.document_id:
        if request.document_id not in _registry(workspace_id):
            raise HTTPException(status_code=404, detail="Document not found")
        if pipeline.multi_stage_retriever and request.retrieval_mode != "basic":
            docs = pipeline.multi_stage_retriever.retrieve(
                question, top_k=request.top_k, workspace_id=workspace_id,
                document_id=request.document_id,
            )
        else:
            # Filter after searching all workspace vectors so other documents
            # cannot crowd the selected document out of the results.
            candidates = store.search(
                pipeline.embedding_service.embed(question),
                top_k=len(store.metadata), workspace_id=workspace_id,
            )
            docs = [d for d in candidates if d.get("document_id") == request.document_id][:request.top_k]
    elif request.retrieval_mode == "basic" or not pipeline.multi_stage_retriever:
        docs = store.search(
            pipeline.embedding_service.embed(question),
            top_k=request.top_k,
            workspace_id=workspace_id,
        )
    else:
        docs = pipeline.multi_stage_retriever.retrieve(
            question, top_k=request.top_k, workspace_id=workspace_id
        )
    if not docs:
        return {
            "answer": "لا توجد معلومات ذات صلة في المستندات المرفوعة.",
            "retrieved_documents": [],
        }
    answer = pipeline.answer(question, workspace_id=workspace_id, retrieved_docs=docs)
    citations = [
        {
            "document_id": doc.get("document_id"),
            "source": doc.get("source"),
            "page_num": doc.get("metadata", {}).get("page_num"),
            "chunk_id": doc.get("metadata", {}).get("chunk_id"),
            "snippet": doc.get("text", "")[:240],
            "score": doc.get("score"),
        }
        for doc in docs
    ]
    safe_docs = [
        {
            "text": doc.get("text", ""),
            "document_id": doc.get("document_id"),
            "source": doc.get("source"),
            "score": doc.get("score"),
            "metadata": {
                key: value
                for key, value in doc.get("metadata", {}).items()
                if key not in {"workspace_id", "added_at", "content_sha256"}
            },
        }
        for doc in docs
    ]
    return {"answer": answer, "retrieved_documents": safe_docs, "citations": citations}


@app.get("/stats")
async def stats(workspace_id: str = Depends(_current_workspace)) -> dict[str, Any]:
    get_services()
    documents = _registry(workspace_id)
    return {"workspace_documents": len(documents),
            "workspace_chunks": sum(doc.chunk_count for doc in documents.values())}
