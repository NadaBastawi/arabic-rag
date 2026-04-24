"""
Caching layer for RAG system.

Implements LRU caching for embeddings and query results to improve performance.
"""

import hashlib
import json
import logging
from typing import Optional, Dict, List, Any
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings with TTL support."""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings.
            ttl_hours: Time-to-live in hours.
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Dict] = {}
    
    def _get_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for.
            
        Returns:
            Embedding vector or None if not cached or expired.
        """
        key = self._get_key(text)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        entry["hits"] += 1
        return entry["embedding"]
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Original text.
            embedding: Embedding vector.
        """
        if len(self.cache) >= self.max_size:
            # Remove least recently used (lowest hits)
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k]["hits"])
            del self.cache[lru_key]
        
        key = self._get_key(text)
        self.cache[key] = {
            "embedding": embedding,
            "expires_at": datetime.now() + self.ttl,
            "hits": 0,
            "created_at": datetime.now().isoformat()
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(v["hits"] for v in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "avg_hits": total_hits / len(self.cache) if self.cache else 0
        }


class QueryResultCache:
    """Cache for query results."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 6):
        """
        Initialize query result cache.
        
        Args:
            max_size: Maximum number of cached queries.
            ttl_hours: Time-to-live in hours.
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Dict] = {}
    
    def _get_key(self, query: str, top_k: int) -> str:
        """Generate cache key from query parameters."""
        key_str = f"{query}:{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, top_k: int) -> Optional[List[Dict]]:
        """
        Get cached query results.
        
        Args:
            query: Query text.
            top_k: Number of results requested.
            
        Returns:
            Cached results or None.
        """
        key = self._get_key(query, top_k)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        entry["hits"] += 1
        logger.debug(f"Query result cache hit for: {query[:50]}...")
        return entry["results"]
    
    def set(self, query: str, top_k: int, results: List[Dict]) -> None:
        """
        Cache query results.
        
        Args:
            query: Query text.
            top_k: Number of results.
            results: Query results.
        """
        if len(self.cache) >= self.max_size:
            # Remove entry with oldest expiration
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k]["expires_at"])
            del self.cache[lru_key]
        
        key = self._get_key(query, top_k)
        self.cache[key] = {
            "results": results,
            "expires_at": datetime.now() + self.ttl,
            "hits": 0,
            "created_at": datetime.now().isoformat()
        }
        
        logger.debug(f"Cached query results for: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(v["hits"] for v in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "avg_hits": total_hits / len(self.cache) if self.cache else 0
        }
