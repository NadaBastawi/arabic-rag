"""
Configuration for Arabic RAG system.

This module contains all configuration values that can be tuned
without modifying code.
"""

from os import getenv
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


class Config:
    """Configuration class for Arabic RAG system."""

    EMBEDDING_MODEL: str = getenv("EMBEDDING_MODEL", "distiluse-base-multilingual-cased-v2")
    EMBEDDING_DEVICE: Optional[str] = getenv("EMBEDDING_DEVICE", None)

    LLM_MODEL: str = getenv("LLM_MODEL", "gemini-2.0-flash")
    LLM_API_KEY: str = getenv("LLM_API_KEY", "")
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    VECTOR_STORE_TYPE: str = getenv("VECTOR_STORE_TYPE", "faiss")
    INDEX_PATH: str = getenv("INDEX_PATH", "./data/index")
    DOCS_PATH: str = getenv("DOCS_PATH", "./data/documents")

    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_INITIAL_K: int = 20
    RETRIEVAL_MODE: str = getenv("RETRIEVAL_MODE", "multi_stage")
    USE_ADAPTIVE: bool = getenv("USE_ADAPTIVE", "true").lower() == "true"

    RE_RANKER_WEIGHTS: dict = {
        "semantic": 0.30,
        "lexical": 0.20,
        "root": 0.15,
        "bm25": 0.15,
        "ngram": 0.10,
        "exact_match": 0.10
    }

    ARABIC_NORMALIZE: bool = True
    ARABIC_REMOVE_DIACRITICS: bool = True

    CAMEL_DB: str = getenv("CAMEL_DB", "arablmsr")

    API_HOST: str = getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(getenv("API_PORT", "8000"))

    GRADIO_PORT: int = int(getenv("GRADIO_PORT", "7860"))
    GRADIO_SHARE: bool = getenv("GRADIO_SHARE", "false").lower() == "true"

    LOG_LEVEL: str = getenv("LOG_LEVEL", "INFO")


config = Config()
