"""
Simple in-memory vector store implementation.

This module provides a basic vector store for storing and searching document
embeddings using cosine similarity. Suitable for development and small-scale
deployments.
"""

from typing import List
import numpy as np


class SimpleVectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    
    This implementation stores embeddings and texts in memory and performs
    similarity search using cosine similarity. Suitable for development and
    small datasets. For production use with large datasets, consider using
    FAISS, ChromaDB, or other specialized vector databases.
    
    Attributes:
        texts: List of document texts.
        embeddings: Numpy array of document embeddings.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.texts: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> None:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of document texts to add.
            embeddings: Numpy array of embeddings with shape (num_texts, embedding_dim).
                       Each row corresponds to a text in the texts list.
        
        Raises:
            ValueError: If the number of texts doesn't match the number of embeddings.
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        if len(texts) == 0:
            return
        
        # Add texts
        self.texts.extend(texts)
        
        # Add embeddings
        if self.embeddings is None:
            self.embeddings = embeddings.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector with shape (embedding_dim,).
            top_k: Number of top results to return. Defaults to 5.
            
        Returns:
            List of dictionaries, each containing:
            - 'text': The document text
            - 'score': Cosine similarity score (between -1 and 1)
            Results are sorted by similarity score in descending order.
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []
        
        # Ensure query_embedding is 1D
        query_embedding = query_embedding.flatten()
        
        # Normalize query embedding for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        query_embedding = query_embedding / query_norm
        
        # Normalize stored embeddings for cosine similarity
        # Compute norms for all embeddings
        embedding_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        embedding_norms = np.where(embedding_norms == 0, 1, embedding_norms)
        normalized_embeddings = self.embeddings / embedding_norms
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        # Get top k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results list
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'score': float(similarities[idx])
            })
        
        return results

