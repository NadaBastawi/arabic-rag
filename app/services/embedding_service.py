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

    A deterministic hashed embedding implementation is available only when
    explicitly enabled for local development and tests.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        fallback_dimension: int = 384,
        allow_fallback: bool = False,
    ):
        # E5 requires distinct query/passage prefixes. Refuse it until an
        # explicit migration implements and evaluates both paths.
        if re.match(r"(?:multilingual-)?e5-", model_name.rsplit("/", 1)[-1], re.IGNORECASE):
            raise ValueError("E5 requires an explicit migration with query and passage prefixes")
        self.model_name = model_name
        self.device = self._normalize_device(device)
        self.fallback_dimension = fallback_dimension
        self.allow_fallback = allow_fallback

        self.model = None
        self._using_fallback = False
        self._load_model()

    @staticmethod
    def _normalize_device(device: str | None) -> str | None:
        if device is None:
            return None

        normalized = str(device).split("#", 1)[0].strip().lower()
        if not normalized or normalized in {"auto", "none", "default"}:
            return None
        return normalized

    def _load_model(self) -> None:
        """Load sentence-transformers model or enable fallback mode."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Could not import the embedding runtime; check the pinned dependencies"
                ) from exc
            self._using_fallback = True
            logger.warning(
                "Embedding runtime could not be imported; using hashed fallback embeddings"
            )
            return

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._using_fallback = False
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}'"
                ) from exc
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

    def chunking_config(self) -> dict:
        """Return the actual model tokenizer and length used by encode()."""
        if self.model is None or self._using_fallback:
            return {}
        tokenizer = self.model.tokenizer
        model_limit = int(self.model.max_seq_length)
        tokenizer_limit = int(tokenizer.model_max_length)
        limit = min(model_limit, tokenizer_limit)
        prompt_name = getattr(self.model, "default_prompt_name", None)
        prefix = (getattr(self.model, "prompts", {}) or {}).get(prompt_name, "") if prompt_name else ""
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer with offset mappings is required for safe chunking")
        return {"tokenizer": tokenizer, "max_tokens": limit, "token_prefix": prefix}

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
                cfg = self.chunking_config()
                for text in texts:
                    encoded = cfg["tokenizer"](cfg["token_prefix"] + text,
                                                add_special_tokens=True, truncation=False)
                    if len(encoded["input_ids"]) > cfg["max_tokens"]:
                        raise ValueError("Embedding input exceeds the model token limit")
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
            except ValueError:
                raise
            except Exception as exc:
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Embedding model '{self.model_name}' failed to encode input"
                    ) from exc
                logger.warning(
                    "Model embedding failed (%s). Falling back to hashed embeddings.",
                    str(exc),
                )
                self._using_fallback = True
                embeddings = self._fallback_embed(texts)
        else:
            embeddings = self._fallback_embed(texts)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        expected_dim = self.get_embedding_dimension()
        if embeddings.shape != (len(texts), expected_dim):
            raise RuntimeError("Embedding model returned an unexpected vector shape")
        if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) == 0):
            raise RuntimeError("Embedding model returned invalid vectors")
        if single_text:
            return embeddings[0]
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension for active backend."""
        if self.model is not None and not self._using_fallback:
            return self.model.get_sentence_embedding_dimension()
        return self.fallback_dimension
