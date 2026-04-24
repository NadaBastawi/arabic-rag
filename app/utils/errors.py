"""
Error handling and validation utilities.

Provides custom exceptions and validators for input data and operations.
"""

import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class ValidationError(RAGException):
    """Validation error."""
    pass


class DocumentError(RAGException):
    """Error related to document processing."""
    pass


class EmbeddingError(RAGException):
    """Error related to embedding generation."""
    pass


class RetrievalError(RAGException):
    """Error related to document retrieval."""
    pass


class GenerationError(RAGException):
    """Error related to text generation."""
    pass


class RateLimitError(RAGException):
    """Rate limit exceeded error."""
    pass


class InputValidator:
    """Validator for input data."""
    
    # Arabic text pattern (basic check)
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
    
    @staticmethod
    def validate_query(query: str, max_length: int = 1000) -> str:
        """
        Validate and clean query text.
        
        Args:
            query: Query text to validate.
            max_length: Maximum allowed length.
            
        Returns:
            Cleaned query text.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        query = query.strip()
        
        if len(query) > max_length:
            raise ValidationError(f"Query exceeds maximum length of {max_length} characters")
        
        if len(query) < 2:
            raise ValidationError("Query must be at least 2 characters")
        
        return query
    
    @staticmethod
    def validate_document(text: str, max_length: int = 100000, min_length: int = 10) -> str:
        """
        Validate and clean document text.
        
        Args:
            text: Document text to validate.
            max_length: Maximum allowed length.
            min_length: Minimum allowed length.
            
        Returns:
            Cleaned text.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not text:
            raise ValidationError("Document cannot be empty")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(f"Document must be at least {min_length} characters")
        
        if len(text) > max_length:
            raise ValidationError(f"Document exceeds maximum length of {max_length} characters")
        
        return text
    
    @staticmethod
    def validate_document_id(doc_id: str) -> str:
        """
        Validate document ID.
        
        Args:
            doc_id: Document ID to validate.
            
        Returns:
            Validated ID.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not doc_id:
            raise ValidationError("Document ID cannot be empty")
        
        doc_id = doc_id.strip()
        
        if len(doc_id) > 256:
            raise ValidationError("Document ID cannot exceed 256 characters")
        
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', doc_id):
            raise ValidationError(
                "Document ID can only contain alphanumeric characters, underscores, hyphens, and dots"
            )
        
        return doc_id
    
    @staticmethod
    def validate_top_k(top_k: int, max_k: int = 100) -> int:
        """
        Validate top_k parameter.
        
        Args:
            top_k: Number of results to retrieve.
            max_k: Maximum allowed value.
            
        Returns:
            Validated top_k value.
            
        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(top_k, int):
            raise ValidationError("top_k must be an integer")
        
        if top_k < 1:
            raise ValidationError("top_k must be at least 1")
        
        if top_k > max_k:
            raise ValidationError(f"top_k cannot exceed {max_k}")
        
        return top_k
    
    @staticmethod
    def validate_embeddings(embeddings: List, dimension: Optional[int] = None) -> None:
        """
        Validate embeddings list.
        
        Args:
            embeddings: List of embedding vectors.
            dimension: Expected embedding dimension (optional).
            
        Raises:
            ValidationError: If validation fails.
        """
        if not embeddings:
            raise ValidationError("Embeddings list cannot be empty")
        
        if not isinstance(embeddings, list):
            raise ValidationError("Embeddings must be a list")
        
        if dimension is not None:
            for i, emb in enumerate(embeddings):
                if len(emb) != dimension:
                    raise ValidationError(
                        f"Embedding {i} has dimension {len(emb)}, expected {dimension}"
                    )


class ErrorHandler:
    """Handles errors and logs them appropriately."""
    
    @staticmethod
    def handle_api_error(error: Exception, endpoint: str) -> dict:
        """
        Handle API error and return appropriate response.
        
        Args:
            error: Exception that occurred.
            endpoint: API endpoint where error occurred.
            
        Returns:
            Error response dict.
        """
        logger.error(f"Error in {endpoint}: {str(error)}", exc_info=True)
        
        if isinstance(error, ValidationError):
            return {
                "error": "VALIDATION_ERROR",
                "message": str(error),
                "status_code": 422
            }
        elif isinstance(error, DocumentError):
            return {
                "error": "DOCUMENT_ERROR",
                "message": str(error),
                "status_code": 400
            }
        elif isinstance(error, EmbeddingError):
            return {
                "error": "EMBEDDING_ERROR",
                "message": str(error),
                "status_code": 500
            }
        elif isinstance(error, RetrievalError):
            return {
                "error": "RETRIEVAL_ERROR",
                "message": str(error),
                "status_code": 500
            }
        elif isinstance(error, GenerationError):
            return {
                "error": "GENERATION_ERROR",
                "message": str(error),
                "status_code": 500
            }
        elif isinstance(error, RateLimitError):
            return {
                "error": "RATE_LIMIT_EXCEEDED",
                "message": str(error),
                "status_code": 429
            }
        else:
            return {
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "status_code": 500
            }
