"""
Embedding service for generating text embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generate embeddings with sentence-transformers when available.

    If model loading fails (for example in offline environments), the service
    falls back to a deterministic hashed embedding implementation so the
    pipeline remains operational and testable.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        fallback_dimension: int = 384,
    ):
        self.model_name = model_name
        self.device = device
        self.fallback_dimension = fallback_dimension

        self.model = None
        self._using_fallback = False
        self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformers model or enable fallback mode."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._using_fallback = True
            logger.warning(
                "sentence-transformers is not installed; using hashed fallback embeddings"
            )
            return

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._using_fallback = False
        except Exception as exc:
            self._using_fallback = True
            logger.warning(
                "Failed to load embedding model '%s' (%s). Using hashed fallback embeddings.",
                self.model_name,
                str(exc),
            )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return tokens if tokens else [text.strip().lower()]

    def _hash_token_to_index(self, token: str) -> int:
        token_hash = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(token_hash, 16) % self.fallback_dimension

    def _fallback_embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dimension, dtype=np.float32)

        for token in self._tokenize(text):
            if not token:
                continue
            vector[self._hash_token_to_index(token)] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._fallback_embed_text(text) for text in texts]).astype(
            np.float32
        )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for a single text or list of texts."""
        if not texts:
            raise ValueError("Input texts cannot be empty")

        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        if not isinstance(texts, list):
            raise ValueError("Input must be a string or list of strings")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts list must be strings")

        embeddings: np.ndarray
        if self.model is not None and not self._using_fallback:
            try:
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning(
                    "Model embedding failed (%s). Falling back to hashed embeddings.",
                    str(exc),
                )
                self._using_fallback = True
                embeddings = self._fallback_embed(texts)
        else:
            embeddings = self._fallback_embed(texts)

        if single_text:
            return embeddings[0]
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension for active backend."""
        if self.model is not None and not self._using_fallback:
            return self.model.get_sentence_embedding_dimension()
        return self.fallback_dimension
