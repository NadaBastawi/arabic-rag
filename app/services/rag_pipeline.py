"""
RAG pipeline for question answering.
"""

from __future__ import annotations

import re
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
            use_fast_retrieval = (
                self.retrieval_mode == "multi_stage"
                and hasattr(self.llm_service, "is_quota_backoff_active")
                and self.llm_service.is_quota_backoff_active()
            )

            if self.retrieval_mode == "multi_stage" and self.multi_stage_retriever and not use_fast_retrieval:
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
            answer = self.llm_service.generate(prompt)

            if answer == LLMService.LLM_QUOTA_EXCEEDED:
                return self._build_retrieval_fallback(
                    question,
                    retrieved_docs,
                    reason=(
                        "LLM quota/rate limit reached, so I am returning a quick extract from your documents."
                    ),
                )
            if answer == LLMService.LLM_API_UNAVAILABLE:
                return self._build_retrieval_fallback(
                    question,
                    retrieved_docs,
                    reason=(
                        "LLM API is unavailable or API key is missing, so I am returning a quick extract from your documents."
                    ),
                )
            if answer.startswith("Error generating response:"):
                return self._build_retrieval_fallback(
                    question,
                    retrieved_docs,
                    reason="LLM generation failed, so I am returning a quick extract from your documents.",
                )

            return answer
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to generate answer: {str(exc)}") from exc

    @staticmethod
    def _build_retrieval_fallback(
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        reason: str,
    ) -> str:
        """Return a retrieval-only answer when generation is unavailable."""
        if not retrieved_docs:
            return (
                f"{reason}\n\n"
                "No relevant passages were found in the indexed documents for this question."
            )

        terms = RAGPipeline._query_terms(question)
        candidates: List[Dict[str, Any]] = []
        seen = set()

        for doc in retrieved_docs:
            raw_text = doc.get("text", "") or ""
            snippet = RAGPipeline._clean_snippet(raw_text)
            if len(snippet) < 40:
                continue

            dedupe_key = snippet[:180]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            snippet_terms = RAGPipeline._query_terms(snippet)
            overlap = len(terms.intersection(snippet_terms)) if terms else 0
            score = float(doc.get("score", 0.0))
            candidates.append(
                {
                    "snippet": snippet,
                    "overlap": overlap,
                    "score": score,
                    "source": doc.get("source", "unknown"),
                }
            )

        if not candidates:
            return f"{reason}\n\nNo clean passages were extracted from the retrieved chunks."

        candidates.sort(key=lambda item: (item["overlap"], item["score"]), reverse=True)

        lines = [reason, "", "Quick evidence from your docs:"]
        for idx, item in enumerate(candidates[:2], 1):
            lines.append(f"{idx}. {item['snippet'][:180]}")
        return "\n".join(lines)

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        tokens = re.findall(r"[\w\u0600-\u06FF]+", (text or "").lower(), flags=re.UNICODE)
        return {token for token in tokens if len(token) >= 3}

    @staticmethod
    def _clean_snippet(text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\S*[\\/]\S*\b", " ", cleaned)
        cleaned = re.sub(r"\b[^\s]{20,}\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

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
