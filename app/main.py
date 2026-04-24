"""
FastAPI application for Arabic RAG system.

Provides REST API endpoints for:
- Health check
- Document indexing
- Query answering
- Retrieval evaluation
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
import os
import tempfile

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.faiss_store import FAISSVectorStore
from app.services.rag_pipeline import RAGPipeline
from app.services.multi_stage_retrieval import ArabicReRanker, MultiStageRetriever, AdaptiveRetrieval
from app.services.cache import EmbeddingCache, QueryResultCache
from app.utils.document_loader import DocumentLoader
from app.utils.errors import InputValidator, ErrorHandler, ValidationError, DocumentError
from app.utils.logging_config import setup_system_logging

# Setup logging
os.makedirs("./logs", exist_ok=True)
setup_system_logging(log_dir="./logs", level=config.LOG_LEVEL, console=True)
logger = logging.getLogger(__name__)

embedding_service: Optional[EmbeddingService] = None
llm_service: Optional[LLMService] = None
vector_store: Optional[FAISSVectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
embedding_cache: Optional[EmbeddingCache] = None
query_cache: Optional[QueryResultCache] = None
document_loader: Optional[DocumentLoader] = None
document_registry: dict = {}  # Maps document_id to metadata


def _build_reranker() -> ArabicReRanker:
    """Create re-ranker from configured weights with safe defaults."""
    weights = config.RE_RANKER_WEIGHTS
    return ArabicReRanker(
        semantic_weight=weights.get("semantic", 0.30),
        lexical_weight=weights.get("lexical", 0.20),
        root_weight=weights.get("root", 0.15),
        bm25_weight=weights.get("bm25", 0.15),
        ngram_weight=weights.get("ngram", 0.10),
        exact_match_weight=weights.get("exact_match", 0.10),
    )


def get_services():
    """Initialize and return all services."""
    global embedding_service, llm_service, vector_store, rag_pipeline, embedding_cache, query_cache, document_loader
    
    if embedding_service is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
    
    if llm_service is None:
        logger.info(f"Initializing LLM service: {config.LLM_MODEL}")
        llm_service = LLMService(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
    
    if vector_store is None:
        logger.info(f"Initializing persistent FAISS store: {config.INDEX_PATH}")
        vector_store = FAISSVectorStore(index_path=config.INDEX_PATH)
    
    if embedding_cache is None:
        embedding_cache = EmbeddingCache(max_size=10000, ttl_hours=24)
    
    if query_cache is None:
        query_cache = QueryResultCache(max_size=1000, ttl_hours=6)
    
    if document_loader is None:
        document_loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(
            embedding_service=embedding_service,
            llm_service=llm_service,
            vector_store=vector_store,
            top_k=config.RETRIEVAL_TOP_K,
            retrieval_mode=config.RETRIEVAL_MODE,
            initial_top_k=config.RETRIEVAL_INITIAL_K,
            use_adaptive=config.USE_ADAPTIVE
        )
    
    return embedding_service, llm_service, vector_store, rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Arabic RAG API...")
    get_services()
    logger.info("Arabic RAG API ready")
    yield
    logger.info("Shutting down Arabic RAG API...")


app = FastAPI(
    title="Arabic RAG API",
    description="Multi-stage retrieval system for Arabic RAG",
    version="1.0.0",
    lifespan=lifespan
)


class IndexRequest(BaseModel):
    """Request model for document indexing."""
    texts: List[str] = Field(..., description="List of Arabic documents to index")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")


class AddDocumentRequest(BaseModel):
    """Request model for adding a document."""
    text: str = Field(..., description="Document text")
    document_id: str = Field(..., description="Unique document identifier")
    source: Optional[str] = Field(default="api", description="Document source")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class DeleteDocumentRequest(BaseModel):
    """Request model for deleting a document."""
    document_id: str = Field(..., description="Document ID to delete")


class DocumentInfo(BaseModel):
    """Model for document information."""
    document_id: str
    source: str
    chunk_count: int
    added_at: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentInfo]
    total_count: int


class IndexResponse(BaseModel):
    """Response model for indexing."""
    indexed_count: int
    total_documents: int
    message: str


class QueryRequest(BaseModel):
    """Request model for query answering."""
    question: str = Field(..., description="Arabic question to answer")
    retrieval_mode: Optional[str] = Field(default=None, description="Override retrieval mode")
    top_k: Optional[int] = Field(default=None, description="Override top_k")


class QueryResponse(BaseModel):
    """Response model for query answering."""
    answer: str
    retrieved_documents: List[dict]
    retrieval_mode: str


class RetrieveRequest(BaseModel):
    """Request model for retrieval only."""
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    retrieval_mode: Optional[str] = Field(default=None, description="Override retrieval mode")


class RetrieveResponse(BaseModel):
    """Response model for retrieval."""
    documents: List[dict]
    retrieval_mode: str


class ScoreRequest(BaseModel):
    """Request model for getting detailed scores."""
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of documents")


class ScoreResponse(BaseModel):
    """Response model for detailed scores."""
    documents: List[dict]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Arabic RAG API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE
    }


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Index Arabic documents into the vector store.
    
    Args:
        request: IndexRequest with texts and batch_size
        
    Returns:
        IndexResponse with indexed count
    """
    try:
        embed_svc, _, vs, _ = get_services()
        
        texts = request.texts
        total = len(texts)
        
        logger.info(f"Indexing {total} documents...")
        
        embeddings = embed_svc.embed(texts)
        vs.add_documents(texts=texts, embeddings=embeddings)
        
        logger.info(f"Successfully indexed {total} documents")
        
        return IndexResponse(
            indexed_count=total,
            total_documents=vs.get_stats().get("total_documents", total),
            message=f"Successfully indexed {total} documents"
        )
        
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.
    
    Args:
        request: QueryRequest with question, optional retrieval_mode and top_k
        
    Returns:
        QueryResponse with answer and retrieved documents
    """
    try:
        embed_svc, llm_svc, vs, pipeline = get_services()

        question = request.question.strip() if request.question else ""
        if not question:
            raise ValueError("Question must be a non-empty string")

        retrieval_mode = request.retrieval_mode or pipeline.retrieval_mode
        top_k = request.top_k or pipeline.top_k

        if retrieval_mode == "multi_stage":
            if config.USE_ADAPTIVE:
                retriever = AdaptiveRetrieval(embed_svc, vs)
            else:
                retriever = MultiStageRetriever(
                    embed_svc,
                    vs,
                    initial_top_k=config.RETRIEVAL_INITIAL_K,
                    reranker=_build_reranker()
                )
            retrieved_docs = retriever.retrieve(question, top_k=top_k)
        else:
            q_emb = embed_svc.embed(question)
            retrieved_docs = vs.search(q_emb, top_k=top_k)

        if not retrieved_docs:
            context = "لا توجد معلومات متاحة في قاعدة المعرفة."
        else:
            context = "\n\n".join(
                f"المستند {idx + 1}: {doc.get('text', '')}"
                for idx, doc in enumerate(retrieved_docs)
            )

        prompt = pipeline._construct_prompt(question, context)
        answer = llm_svc.generate(prompt)

        return QueryResponse(
            answer=answer,
            retrieved_documents=retrieved_docs,
            retrieval_mode=retrieval_mode
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Retrieve relevant documents without generating answer.
    
    Args:
        request: RetrieveRequest with query and top_k
        
    Returns:
        RetrieveResponse with retrieved documents
    """
    try:
        embed_svc, _, vs, _ = get_services()
        
        retrieval_mode = request.retrieval_mode or config.RETRIEVAL_MODE
        
        if retrieval_mode == "multi_stage":
            if config.USE_ADAPTIVE:
                retriever = AdaptiveRetrieval(embed_svc, vs)
            else:
                reranker = _build_reranker()
                retriever = MultiStageRetriever(
                    embed_svc, vs,
                    reranker=reranker
                )
            documents = retriever.retrieve(request.query, top_k=request.top_k)
        else:
            q_emb = embed_svc.embed(request.query)
            documents = vs.search(q_emb, top_k=request.top_k)
        
        return RetrieveResponse(
            documents=documents,
            retrieval_mode=retrieval_mode
        )
        
    except Exception as e:
        logger.error(f"Retrieve error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scores", response_model=ScoreResponse)
async def get_scores(request: ScoreRequest):
    """
    Get detailed score breakdown for retrieval.
    
    Only works with multi_stage retrieval mode.
    
    Args:
        request: ScoreRequest with query and top_k
        
    Returns:
        ScoreResponse with detailed scores
    """
    try:
        _, _, _, pipeline = get_services()
        
        if pipeline.retrieval_mode != "multi_stage":
            raise HTTPException(
                status_code=400,
                detail="Score endpoint requires multi_stage retrieval mode"
            )
        
        documents = pipeline.retrieve_with_scores(request.query)
        
        return ScoreResponse(documents=documents[:request.top_k])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scores error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive values)."""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "use_adaptive": config.USE_ADAPTIVE,
        "re_ranker_weights": config.RE_RANKER_WEIGHTS,
        "top_k": config.RETRIEVAL_TOP_K,
        "initial_k": config.RETRIEVAL_INITIAL_K
    }


@app.post("/documents/add")
async def add_document(request: AddDocumentRequest):
    """
    Add a single document to the index.
    
    Args:
        request: AddDocumentRequest with document text and metadata
        
    Returns:
        Success message with document ID
    """
    try:
        # Validate input
        text = InputValidator.validate_document(request.text)
        doc_id = InputValidator.validate_document_id(request.document_id)
        
        embed_svc, _, vs, _ = get_services()
        
        # Generate embedding
        embedding = embed_svc.embed(text)
        
        # Add to vector store
        vs.add_documents(
            texts=[text],
            embeddings=[embedding],
            document_ids=[doc_id],
            sources=[request.source],
            metadata_list=[request.metadata or {}]
        )
        
        # Update document registry
        from datetime import datetime
        document_registry[doc_id] = {
            "source": request.source,
            "added_at": datetime.now().isoformat(),
            "chunk_count": 1
        }
        
        logger.info(f"Added document: {doc_id}")
        
        return {
            "status": "success",
            "document_id": doc_id,
            "message": f"Document {doc_id} added successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, "/documents/add")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF or text document.
    
    Args:
        file: PDF or text file to upload
        
    Returns:
        Success message with chunk count
    """
    try:
        embed_svc, _, vs, _ = get_services()
        loader = document_loader
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load and chunk document
            chunks = loader.load_file(tmp_path)
            
            if not chunks:
                raise DocumentError(f"Could not extract text from {file.filename}")
            
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = embed_svc.embed(texts)
            
            # Add to vector store
            doc_id = os.path.splitext(file.filename)[0]
            sources = [chunk.source for chunk in chunks]
            
            vs.add_documents(
                texts=texts,
                embeddings=embeddings,
                document_ids=[doc_id] * len(chunks),
                sources=sources
            )
            
            # Update registry
            from datetime import datetime
            document_registry[doc_id] = {
                "source": file.filename,
                "added_at": datetime.now().isoformat(),
                "chunk_count": len(chunks)
            }
            
            logger.info(f"Uploaded document: {doc_id} with {len(chunks)} chunks")
            
            return {
                "status": "success",
                "document_id": doc_id,
                "chunk_count": len(chunks),
                "message": f"Document {file.filename} indexed with {len(chunks)} chunks"
            }
            
        finally:
            os.unlink(tmp_path)
            
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, "/documents/upload")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the index.
    
    Args:
        document_id: ID of document to delete
        
    Returns:
        Success message
    """
    try:
        _, _, vs, _ = get_services()
        
        if document_id not in document_registry:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        deleted = vs.delete_document(document_id)
        
        if deleted:
            del document_registry[document_id]
            logger.info(f"Deleted document: {document_id}")
            return {"status": "success", "message": f"Document {document_id} deleted"}
        else:
            raise DocumentError(f"Could not delete document {document_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, f"/documents/{document_id}")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])


@app.get("/documents")
async def list_documents():
    """
    List all indexed documents.
    
    Returns:
        DocumentListResponse with document list
    """
    try:
        documents = []
        for doc_id, info in document_registry.items():
            documents.append({
                "document_id": doc_id,
                "source": info["source"],
                "chunk_count": info["chunk_count"],
                "added_at": info["added_at"]
            })
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }
        
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, "/documents")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])


@app.post("/documents/clear")
async def clear_documents():
    """
    Clear all indexed documents.
    
    Returns:
        Success message
    """
    try:
        global vector_store, rag_pipeline, document_registry

        embed_svc, llm_svc, _, _ = get_services()

        # Clear vector store by reinitializing and rewire pipeline.
        vector_store = FAISSVectorStore(index_path=config.INDEX_PATH)
        rag_pipeline = RAGPipeline(
            embedding_service=embed_svc,
            llm_service=llm_svc,
            vector_store=vector_store,
            top_k=config.RETRIEVAL_TOP_K,
            retrieval_mode=config.RETRIEVAL_MODE,
            initial_top_k=config.RETRIEVAL_INITIAL_K,
            use_adaptive=config.USE_ADAPTIVE
        )
        document_registry.clear()
        
        logger.info("Cleared all documents")
        
        return {"status": "success", "message": "All documents cleared"}
        
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, "/documents/clear")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])


@app.get("/stats")
async def get_statistics():
    """
    Get system statistics.
    
    Returns:
        Statistics about vector store and caches
    """
    try:
        _, _, vs, _ = get_services()
        
        return {
            "vector_store": vs.get_stats(),
            "embedding_cache": embedding_cache.get_stats() if embedding_cache else {},
            "query_cache": query_cache.get_stats() if query_cache else {},
            "total_documents": len(document_registry)
        }
        
    except Exception as e:
        error_resp = ErrorHandler.handle_api_error(e, "/stats")
        raise HTTPException(status_code=error_resp["status_code"], detail=error_resp["message"])
