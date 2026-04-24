"""
RAG pipeline for question answering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.multi_stage_retrieval import (
    AdaptiveRetrieval,
    ArabicReRanker,
    MultiStageRetriever,
)


class VectorStore(Protocol):
    """Protocol that any vector-store implementation must satisfy."""

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        ...


class RAGPipeline:
    """RAG pipeline that combines retrieval with LLM generation."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: VectorStore,
        top_k: int = 5,
        retrieval_mode: str = "basic",
        initial_top_k: int = 20,
        use_adaptive: bool = False,
    ):
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        self.multi_stage_retriever = None
        if retrieval_mode == "multi_stage":
            if use_adaptive:
                self.multi_stage_retriever = AdaptiveRetrieval(embedding_service, vector_store)
            else:
                self.multi_stage_retriever = MultiStageRetriever(
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                    initial_top_k=initial_top_k,
                    reranker=ArabicReRanker(),
                )

    def answer(self, question: str) -> str:
        """Answer a question using retrieved knowledge-base context."""
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        try:
            if self.retrieval_mode == "multi_stage" and self.multi_stage_retriever:
                retrieved_docs = self.multi_stage_retriever.retrieve(question, top_k=self.top_k)
            else:
                question_embedding = self.embedding_service.embed(question)
                retrieved_docs = self.vector_store.search(question_embedding, top_k=self.top_k)

            if not retrieved_docs:
                context = "?? ???? ??????? ????? ?? ????? ???????."
            else:
                context_parts = [
                    f"??????? {idx + 1}: {doc.get('text', '')}"
                    for idx, doc in enumerate(retrieved_docs)
                ]
                context = "\n\n".join(context_parts)

            prompt = self._construct_prompt(question, context)
            return self.llm_service.generate(prompt)
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to generate answer: {str(exc)}") from exc

    def _construct_prompt(self, question: str, context: str) -> str:
        return (
            "?????? ????????? ??????? ?? ????? ??????? ??????? ??? ??????. "
            "??? ?? ??? ??????? ?? ????????? ???????? ?? ??? ?? ????.\n\n"
            f"????????? ?? ????? ???????:\n{context}\n\n"
            f"??????: {question}\n\n"
            "???????:"
        )

    def retrieve_with_scores(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve documents with score breakdown (multi-stage mode only)."""
        if self.retrieval_mode != "multi_stage" or not self.multi_stage_retriever:
            raise RuntimeError("retrieve_with_scores requires multi_stage retrieval mode")
        return self.multi_stage_retriever.retrieve(question, self.top_k)
