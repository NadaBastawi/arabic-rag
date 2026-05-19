"""
Persistent vector store implementation with FAISS + NumPy fallback.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Persistent vector store using FAISS when available.

    If FAISS is not installed, the store falls back to a NumPy in-memory index
    while preserving the same API.
    """

    def __init__(self, index_path: str = "./data/index"):
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.metadata_path = os.path.join(index_path, "metadata.json")
        self.numpy_embeddings_path = os.path.join(index_path, "embeddings.npy")

        os.makedirs(index_path, exist_ok=True)

        self.index = None
        self.metadata: List[Dict] = []
        self._embeddings_np: Optional[np.ndarray] = None
        self._embedding_dim: Optional[int] = None
        self._next_id: int = 0

        self._backend = "numpy"
        self._faiss = None

        self._load()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _init_backend(self) -> None:
        try:
            import faiss

            self._faiss = faiss
            self._backend = "faiss"
        except ImportError:
            self._faiss = None
            self._backend = "numpy"
            logger.warning("FAISS not installed; using NumPy fallback backend")

    def _ensure_index(self, embedding_dim: int) -> None:
        if self._embedding_dim is None:
            self._embedding_dim = embedding_dim
        elif self._embedding_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {embedding_dim}"
            )

        if self._backend == "faiss":
            if self.index is None:
                base_index = self._faiss.IndexFlatIP(embedding_dim)
                self.index = self._faiss.IndexIDMap2(base_index)
        else:
            if self._embeddings_np is None:
                self._embeddings_np = np.empty((0, embedding_dim), dtype=np.float32)

    def clear(self) -> None:
        """Clear all vectors and persisted index artifacts."""
        self.index = None
        self.metadata = []
        self._embeddings_np = None
        self._embedding_dim = None
        self._next_id = 0

        for path in (self.index_file, self.metadata_path, self.numpy_embeddings_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Could not remove index artifact %s: %s", path, str(exc))

    def ensure_embedding_dimension(
        self,
        embedding_dim: int,
        reset_on_mismatch: bool = False,
    ) -> None:
        """
        Ensure vector-store dimension matches the active embedding model.

        When ``reset_on_mismatch`` is enabled and an existing persisted index
        has a different dimension, the stale index is cleared so the caller can
        re-index with the current model.
        """
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be a positive integer")

        if self._embedding_dim is None:
            self._ensure_index(embedding_dim)
            return

        if self._embedding_dim == embedding_dim:
            self._ensure_index(embedding_dim)
            return

        mismatch_message = (
            f"Embedding dimension mismatch: store={self._embedding_dim}, model={embedding_dim}"
        )
        if not reset_on_mismatch:
            raise ValueError(mismatch_message)

        logger.warning(
            "%s. Clearing stale index at %s and reinitializing.",
            mismatch_message,
            self.index_path,
        )
        self.clear()
        self._ensure_index(embedding_dim)
        self._save()

    def _metadata_map(self) -> Dict[int, Dict]:
        return {int(item["faiss_id"]): item for item in self.metadata}

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        document_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict]] = None,
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        if not texts:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = self._normalize(embeddings)
        self._ensure_index(embeddings.shape[1])

        ids = np.arange(self._next_id, self._next_id + len(texts), dtype=np.int64)

        if self._backend == "faiss":
            self.index.add_with_ids(embeddings, ids)
        else:
            self._embeddings_np = np.vstack([self._embeddings_np, embeddings])

        for idx, text in enumerate(texts):
            entry = {
                "faiss_id": int(ids[idx]),
                "text": text,
                "document_id": document_ids[idx] if document_ids else f"doc_{int(ids[idx])}",
                "source": sources[idx] if sources else "unknown",
            }
            if metadata_list and idx < len(metadata_list):
                extra = metadata_list[idx] or {}
                entry.update(extra)
            self.metadata.append(entry)

        self._next_id += len(texts)
        self._save()
        logger.info("Added %d documents to %s backend", len(texts), self._backend)

    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        """Compatibility alias for older code paths."""
        self.add_documents(texts=texts, embeddings=embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if top_k < 1:
            return []
        if not self.metadata:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if self._embedding_dim is None:
            return []
        if query.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._embedding_dim}, got {query.shape[1]}"
            )

        query = self._normalize(query)
        metadata_by_id = self._metadata_map()

        results: List[Dict] = []

        if self._backend == "faiss":
            if self.index is None or self.index.ntotal == 0:
                return []

            scores, ids = self.index.search(query, min(top_k, self.index.ntotal))
            score_row = scores[0]
            id_row = ids[0]

            for score, idx in zip(score_row, id_row):
                if idx < 0:
                    continue
                meta = metadata_by_id.get(int(idx))
                if not meta:
                    continue
                results.append(self._format_result(meta, float(score)))
        else:
            if self._embeddings_np is None or len(self._embeddings_np) == 0:
                return []

            scores = np.dot(self._embeddings_np, query[0])
            top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
            metadata_sorted = sorted(self.metadata, key=lambda item: int(item["faiss_id"]))

            for pos in top_indices:
                meta = metadata_sorted[int(pos)]
                results.append(self._format_result(meta, float(scores[int(pos)])))

        return results

    @staticmethod
    def _format_result(meta: Dict, score: float) -> Dict:
        excluded = {"text", "source", "document_id", "faiss_id"}
        return {
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "document_id": meta.get("document_id", ""),
            "score": score,
            "metadata": {k: v for k, v in meta.items() if k not in excluded},
        }

    def delete_document(self, document_id: str) -> bool:
        ids_to_remove = [
            int(item["faiss_id"]) for item in self.metadata if item.get("document_id") == document_id
        ]
        if not ids_to_remove:
            return False

        id_set = set(ids_to_remove)

        if self._backend == "faiss" and self.index is not None:
            remove_ids = np.array(ids_to_remove, dtype=np.int64)
            self.index.remove_ids(remove_ids)
        elif self._embeddings_np is not None and len(self._embeddings_np) > 0:
            metadata_sorted = sorted(self.metadata, key=lambda item: int(item["faiss_id"]))
            keep_positions = [
                pos for pos, item in enumerate(metadata_sorted) if int(item["faiss_id"]) not in id_set
            ]
            self._embeddings_np = self._embeddings_np[keep_positions]

        self.metadata = [item for item in self.metadata if int(item["faiss_id"]) not in id_set]
        self._save()
        logger.info("Deleted %d chunks for document_id=%s", len(ids_to_remove), document_id)
        return True

    def _save(self) -> None:
        if self._backend == "faiss" and self.index is not None:
            self._faiss.write_index(self.index, self.index_file)
        elif self._embeddings_np is not None:
            np.save(self.numpy_embeddings_path, self._embeddings_np)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        self._init_backend()

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as exc:
                logger.warning("Could not load metadata: %s", str(exc))
                self.metadata = []

        if self.metadata:
            self._next_id = max(int(item.get("faiss_id", -1)) for item in self.metadata) + 1
        else:
            self._next_id = 0

        if self._backend == "faiss":
            if os.path.exists(self.index_file):
                try:
                    self.index = self._faiss.read_index(self.index_file)
                    self._embedding_dim = self.index.d
                except Exception as exc:
                    logger.warning("Could not load FAISS index, starting fresh: %s", str(exc))
                    self.index = None
            elif self.metadata:
                logger.warning("Metadata exists but FAISS index is missing; starting empty index")
        else:
            if os.path.exists(self.numpy_embeddings_path):
                try:
                    self._embeddings_np = np.load(self.numpy_embeddings_path)
                    if self._embeddings_np.ndim == 1:
                        self._embeddings_np = self._embeddings_np.reshape(1, -1)
                    if len(self._embeddings_np) > 0:
                        self._embedding_dim = int(self._embeddings_np.shape[1])
                except Exception as exc:
                    logger.warning("Could not load NumPy embeddings, starting fresh: %s", str(exc))
                    self._embeddings_np = None

            if self._embeddings_np is None and self.metadata:
                logger.warning("Metadata exists but embedding matrix is missing; clearing stale metadata")
                self.metadata = []
                self._next_id = 0

    def get_stats(self) -> Dict:
        if self._backend == "faiss":
            total_vectors = int(self.index.ntotal) if self.index is not None else 0
        else:
            total_vectors = int(len(self._embeddings_np)) if self._embeddings_np is not None else 0

        return {
            "total_documents": len(self.metadata),
            "total_vectors": total_vectors,
            "index_type": "FAISS" if self._backend == "faiss" else "NumPy",
            "embedding_dim": self._embedding_dim,
            "index_path": self.index_path,
        }
