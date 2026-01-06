"""
Embedding service for generating text embeddings.

This module provides the EmbeddingService class for generating embeddings
using multilingual sentence transformers models.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Service for generating text embeddings using multilingual models.
    
    This class handles the loading and usage of embedding models, specifically
    designed to work with multilingual-e5-large for Arabic and other languages.
    The service is designed to be easily extensible to support other embedding
    models in the future.
    
    Attributes:
        model_name: Name or path of the sentence transformer model.
        model: Loaded SentenceTransformer model instance.
        device: Device on which the model runs (cuda/cpu).
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = None
    ):
        """
        Initialize the EmbeddingService with a specified model.
        
        Args:
            model_name: Name or path of the sentence transformer model.
                       Defaults to "intfloat/multilingual-e5-large".
            device: Device to run the model on ('cuda', 'cpu', or None for auto).
                    If None, automatically selects the best available device.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the sentence transformer model.
        
        This method is called during initialization to load the model.
        The model is loaded lazily to avoid unnecessary memory usage.
        """
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {str(e)}"
            ) from e
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        This method takes a single text string or a list of text strings
        and returns their corresponding embeddings as a numpy array.
        For a single text, returns a 1D array. For multiple texts, returns
        a 2D array where each row is an embedding vector.
        
        Args:
            texts: Single text string or list of text strings to embed.
            
        Returns:
            Numpy array of embeddings. Shape is (embedding_dim,) for single
            text or (num_texts, embedding_dim) for multiple texts.
            
        Raises:
            ValueError: If texts is empty or invalid.
            RuntimeError: If embedding generation fails.
            
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.embed("مرحبا بك")
            >>> embeddings = service.embed(["مرحبا", "كيف حالك"])
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
        
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Validate input
        if not isinstance(texts, list):
            raise ValueError("Input must be a string or list of strings")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts list must be strings")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Return 1D array for single text, 2D for multiple
            if single_text:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embeddings: {str(e)}"
            ) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer representing the embedding dimension.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.get_sentence_embedding_dimension()

