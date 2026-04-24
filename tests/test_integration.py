"""Integration and API tests for the Arabic RAG system."""

import hashlib
import tempfile
from typing import List, Union

import numpy as np
import pytest
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from app import main as api_main
from app.services.cache import EmbeddingCache, QueryResultCache
from app.services.faiss_store import FAISSVectorStore
from app.services.rag_pipeline import RAGPipeline
from app.utils.document_loader import DocumentChunk, DocumentLoader


class MockEmbeddingService:
    """Deterministic local embedding service for integration tests."""

    def __init__(self, dim: int = 48):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = text.split() or [text]
        for token in tokens:
            idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self._embed_one(texts)
        return np.vstack([self._embed_one(text) for text in texts])


class MockLLMService:
    def generate(self, prompt: str) -> str:
        return "mock-answer"


@pytest.fixture
def embedding_service():
    return MockEmbeddingService()


@pytest.fixture
def vector_store():
    with tempfile.TemporaryDirectory() as tmp:
        yield FAISSVectorStore(index_path=tmp)


def test_document_loader_chunks_text_with_overlap():
    loader = DocumentLoader(chunk_size=80, chunk_overlap=20, min_chunk_size=10)
    long_text = "\n\n".join(
        [
            "paragraph one has enough words for chunking",
            "paragraph two continues with additional content",
            "paragraph three ensures split and overlap",
            "paragraph four keeps the text long enough",
        ]
        * 4
    )

    chunks = loader._chunk_text(long_text, "doc1", "memory")

    assert len(chunks) > 1
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_vector_store_add_search_delete_and_persist(embedding_service):
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(index_path=tmp)

        texts = ["?????? ?????????", "?????? ?????", "?????? ????? ???????"]
        embeddings = embedding_service.embed(texts)
        store.add_documents(texts, embeddings, document_ids=["d1", "d2", "d3"])

        results = store.search(embedding_service.embed("??????"), top_k=2)
        assert len(results) >= 1
        assert "text" in results[0]

        assert store.delete_document("d2")
        assert all(item["document_id"] != "d2" for item in store.metadata)

        reloaded = FAISSVectorStore(index_path=tmp)
        assert len(reloaded.metadata) == len(store.metadata)


def test_embedding_and_query_cache():
    emb_cache = EmbeddingCache(max_size=2, ttl_hours=1)
    q_cache = QueryResultCache(max_size=2, ttl_hours=1)

    emb_cache.set("??", [0.1, 0.2])
    assert emb_cache.get("??") == [0.1, 0.2]
    assert emb_cache.get("?????") is None

    q_cache.set("????", 3, [{"text": "????"}])
    assert q_cache.get("????", 3) == [{"text": "????"}]
    assert q_cache.get("????", 2) is None


@pytest.fixture
def api_client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        emb = MockEmbeddingService()
        llm = MockLLMService()
        vs = FAISSVectorStore(index_path=tmp)
        pipeline = RAGPipeline(
            embedding_service=emb,
            llm_service=llm,
            vector_store=vs,
            top_k=3,
            retrieval_mode="basic",
        )
        loader = DocumentLoader(chunk_size=120, chunk_overlap=20, min_chunk_size=10)

        api_main.embedding_service = emb
        api_main.llm_service = llm
        api_main.vector_store = vs
        api_main.rag_pipeline = pipeline
        api_main.embedding_cache = EmbeddingCache(max_size=100, ttl_hours=1)
        api_main.query_cache = QueryResultCache(max_size=100, ttl_hours=1)
        api_main.document_loader = loader
        api_main.document_registry = {}

        def fake_get_services():
            return emb, llm, vs, pipeline

        monkeypatch.setattr(api_main, "get_services", fake_get_services)

        with TestClient(api_main.app) as client:
            yield client


def test_api_index_and_query(api_client):
    response = api_client.post(
        "/index",
        json={"texts": ["?????? ????????? ???", "?????? ????? ???"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed_count"] == 2

    query_response = api_client.post(
        "/query",
        json={"question": "?? ?? ?????? ??????????", "retrieval_mode": "basic", "top_k": 2},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["answer"] == "mock-answer"
    assert isinstance(query_payload["retrieved_documents"], list)


def test_api_document_add_and_list(api_client):
    add_response = api_client.post(
        "/documents/add",
        json={
            "text": "????? ????? ????? ????????.",
            "document_id": "doc_test",
            "source": "unit-test",
        },
    )
    assert add_response.status_code == 200

    list_response = api_client.get("/documents")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total_count"] == 1
    assert payload["documents"][0]["document_id"] == "doc_test"
