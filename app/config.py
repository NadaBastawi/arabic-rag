"""
Configuration for Arabic RAG system.

This module contains all configuration values that can be tuned
without modifying code.
"""

import logging
import os
from typing import Optional

from dotenv import dotenv_values, load_dotenv


logger = logging.getLogger(__name__)


def _environment_value(name: str, default: str = "") -> str:
    """Return the effective value, preserving shell-over-.env precedence."""
    return os.getenv(name, default)


def _warn_on_environment_conflicts() -> None:
    """Report configuration conflicts without revealing values or secrets."""
    env_file = dotenv_values(".env")
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY"):
        shell_value = os.getenv(name)
        file_value = env_file.get(name)
        if shell_value and file_value and shell_value != file_value:
            logger.warning(
                "Configuration conflict for %s: shell environment takes precedence over .env",
                name,
            )


load_dotenv()
_warn_on_environment_conflicts()


class Config:
    """Configuration class for Arabic RAG system."""

    EMBEDDING_MODEL: str = _environment_value(
        "EMBEDDING_MODEL", "distiluse-base-multilingual-cased-v2"
    )
    EMBEDDING_DEVICE: Optional[str] = _environment_value("EMBEDDING_DEVICE") or None
    ALLOW_EMBEDDING_FALLBACK: bool = (
        _environment_value("ALLOW_EMBEDDING_FALLBACK", "false").lower() == "true"
    )
    INDEX_SCHEMA_VERSION: int = int(_environment_value("INDEX_SCHEMA_VERSION", "1"))

    LLM_PROVIDER: str = _environment_value("LLM_PROVIDER", "openai").lower()
    LLM_MODEL: str = _environment_value("LLM_MODEL", "gpt-4.1-mini")
    OPENAI_API_KEY: str = _environment_value("OPENAI_API_KEY")
    GEMINI_API_KEY: str = _environment_value("GEMINI_API_KEY")
    # Legacy compatibility is opt-in to prevent stale shell credentials winning.
    LLM_API_KEY: str = _environment_value("LLM_API_KEY")
    LLM_LEGACY_COMPAT: bool = _environment_value("LLM_LEGACY_COMPAT", "false").lower() == "true"
    LLM_TEMPERATURE: float = float(_environment_value("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(_environment_value("LLM_MAX_TOKENS", "2048"))

    VECTOR_STORE_TYPE: str = _environment_value("VECTOR_STORE_TYPE", "faiss")
    INDEX_PATH: str = _environment_value("INDEX_PATH", "./data/index")
    DOCS_PATH: str = _environment_value("DOCS_PATH", "./data/documents")

    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_INITIAL_K: int = 20
    RETRIEVAL_MODE: str = _environment_value("RETRIEVAL_MODE", "multi_stage")
    USE_ADAPTIVE: bool = _environment_value("USE_ADAPTIVE", "true").lower() == "true"

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

    CAMEL_DB: str = _environment_value("CAMEL_DB", "arablmsr")

    API_HOST: str = _environment_value("API_HOST", "0.0.0.0")
    API_PORT: int = int(_environment_value("API_PORT", "8000"))

    GRADIO_PORT: int = int(_environment_value("GRADIO_PORT", "7860"))
    GRADIO_SHARE: bool = _environment_value("GRADIO_SHARE", "false").lower() == "true"

    LOG_LEVEL: str = _environment_value("LOG_LEVEL", "INFO")

    APP_ENV: str = _environment_value("APP_ENV", "development").lower()

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production" or bool(os.getenv("WEBSITE_SITE_NAME"))

    # A configured bearer token is a local/staging authentication mechanism.
    # Production deployments should replace this boundary with Entra ID/JWT
    # validation rather than disabling authentication.
    AUTH_ENABLED: bool = _environment_value("AUTH_ENABLED", "true").lower() == "true"
    AUTH_TOKEN: str = _environment_value("AUTH_TOKEN")
    AUTH_WORKSPACE_ID: str = _environment_value("AUTH_WORKSPACE_ID", "default")
    LOCAL_DEV_MODE: bool = _environment_value("LOCAL_DEV_MODE", "false").lower() == "true"
    PRIVATE_DEV_MODE: bool = _environment_value("PRIVATE_DEV_MODE", "false").lower() == "true"
    MAX_UPLOAD_BYTES: int = int(_environment_value("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

    @property
    def effective_llm_api_key(self) -> str:
        """Return the selected provider's key, with safe legacy compatibility."""
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY or (
                self.LLM_API_KEY if self.LLM_LEGACY_COMPAT else ""
            )
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_API_KEY or (
                self.LLM_API_KEY if self.LLM_LEGACY_COMPAT else ""
            )
        return ""

    def llm_diagnostic(self) -> dict[str, object]:
        """Return safe LLM configuration information for diagnostics."""
        return {
            "provider": self.LLM_PROVIDER,
            "model": self.LLM_MODEL,
            "credential_configured": bool(self.effective_llm_api_key),
            "provider_key_configured": bool(
                self.OPENAI_API_KEY if self.LLM_PROVIDER == "openai" else self.GEMINI_API_KEY
            ),
            "legacy_key_configured": bool(self.LLM_API_KEY and self.LLM_LEGACY_COMPAT),
            "legacy_key_ignored": bool(self.LLM_API_KEY and not self.LLM_LEGACY_COMPAT),
        }

    def validate_llm(self) -> None:
        """Validate provider and detect credentials intended for another provider."""
        if self.LLM_PROVIDER not in {"openai", "gemini"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'gemini'")
        key = self.effective_llm_api_key
        if not key:
            raise ValueError(
                f"No credentials configured for {self.LLM_PROVIDER}; set the provider-specific API key"
            )
        if self.LLM_PROVIDER == "openai" and key.startswith("AIza"):
            raise ValueError("OpenAI provider received a Gemini credential")
        if self.LLM_PROVIDER == "gemini" and key.startswith("sk-"):
            raise ValueError("Gemini provider received an OpenAI credential")


config = Config()


if __name__ == "__main__":
    import json

    print(json.dumps(config.llm_diagnostic(), sort_keys=True))
