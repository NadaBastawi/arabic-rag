"""Unit tests for the RAG pipeline."""

import hashlib
from typing import List, Union

import numpy as np
import pytest

from app.services.rag_pipeline import RAGPipeline
from app.services.llm_service import LLMService
from app.services.vector_store import SimpleVectorStore


class MockEmbeddingService:
    """Deterministic local embedding service for tests."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = text.split() or [text]
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self._embed_one(texts)
        return np.vstack([self._embed_one(text) for text in texts])


class MockLLMService:
    def generate(self, prompt: str) -> str:
        assert "??????:" in prompt
        return "????? ???????"


class QuotaExceededLLMService:
    def generate(self, prompt: str) -> str:
        return LLMService.LLM_QUOTA_EXCEEDED


def test_documents_can_be_added_and_searched():
    vector_store = SimpleVectorStore()
    embedding_service = MockEmbeddingService()

    documents = [
        "?????? ????????? ?? ??? ?? ???? ???????",
        "?????? ????? ????? ??? ????????"
    ]

    embeddings = embedding_service.embed(documents)
    vector_store.add_documents(documents, embeddings)

    results = vector_store.search(embedding_service.embed("?????? ?????????"), top_k=1)

    assert len(vector_store.texts) == 2
    assert vector_store.embeddings is not None
    assert len(results) == 1
    assert "text" in results[0]


def test_pipeline_returns_non_empty_answer():
    embedding_service = MockEmbeddingService()
    llm_service = MockLLMService()
    vector_store = SimpleVectorStore()

    docs = [
        "?????? ????????? ?????? ?? ????",
        "?????? ?????? ??? ?? ?????? ?????",
    ]
    vector_store.add_documents(docs, embedding_service.embed(docs))

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        top_k=2,
        retrieval_mode="basic",
    )

    answer = pipeline.answer("?? ?? ?????? ??????????")

    assert isinstance(answer, str)
    assert answer.strip() == "????? ???????"


def test_pipeline_rejects_empty_question():
    embedding_service = MockEmbeddingService()
    llm_service = MockLLMService()
    vector_store = SimpleVectorStore()

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        pipeline.answer("")


def test_pipeline_falls_back_when_llm_quota_exceeded():
    embedding_service = MockEmbeddingService()
    llm_service = QuotaExceededLLMService()
    vector_store = SimpleVectorStore()

    docs = [
        "doc about arabic morphology and root extraction",
        "doc about retrieval and ranking",
    ]
    vector_store.add_documents(docs, embedding_service.embed(docs))

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        top_k=2,
        retrieval_mode="basic",
    )

    answer = pipeline.answer("what does the doc say about retrieval?")
    assert "quota/rate limit reached" in answer.lower()
    assert "quick evidence from your docs" in answer.lower()
