"""Focused tests for authentication and tenant isolation."""

import tempfile
from starlette.requests import Request

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import main
from app.services.faiss_store import FAISSVectorStore


def test_auth_rejects_missing_or_invalid_token(monkeypatch):
    monkeypatch.setattr(main.config, "AUTH_ENABLED", True)
    monkeypatch.setattr(main.config, "AUTH_TOKEN", "correct-token")
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", False)
    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", False)

    request = Request({"type": "http", "client": ("10.0.0.2", 8000), "headers": []})
    with pytest.raises(HTTPException) as missing:
        main._current_workspace(request, None)
    assert missing.value.status_code == 401

    invalid = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")
    with pytest.raises(HTTPException) as rejected:
        main._current_workspace(request, invalid)
    assert rejected.value.status_code == 401


def test_auth_returns_configured_workspace(monkeypatch):
    monkeypatch.setattr(main.config, "AUTH_ENABLED", True)
    monkeypatch.setattr(main.config, "AUTH_TOKEN", "correct-token")
    monkeypatch.setattr(main.config, "AUTH_WORKSPACE_ID", "workspace-a")
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", False)
    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", False)
    request = Request({"type": "http", "client": ("10.0.0.2", 8000), "headers": []})

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="correct-token"
    )
    assert main._current_workspace(request, credentials) == "workspace-a"


def test_local_dev_mode_requires_loopback_and_skips_token(monkeypatch):
    monkeypatch.setattr(main.config, "AUTH_ENABLED", True)
    monkeypatch.setattr(main.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", True)
    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", True)
    request = Request({"type": "http", "client": ("127.0.0.1", 8000), "headers": []})

    assert main._current_workspace(request, None) == main.config.AUTH_WORKSPACE_ID

    remote_request = Request({"type": "http", "client": ("10.0.0.2", 8000), "headers": []})
    assert main._current_workspace(remote_request, None) == main.config.AUTH_WORKSPACE_ID

    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", False)
    with pytest.raises(HTTPException) as rejected:
        main._current_workspace(remote_request, None)
    assert rejected.value.status_code == 503


def test_production_mode_rejects_unauthorized_http_request(monkeypatch):
    monkeypatch.setattr(main.config, "AUTH_ENABLED", True)
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", False)
    monkeypatch.setattr(main.config, "AUTH_TOKEN", "correct-token")

    from fastapi.testclient import TestClient

    response = TestClient(main.app).get("/documents")
    assert response.status_code == 401


def test_vector_search_and_delete_are_workspace_scoped():
    with tempfile.TemporaryDirectory() as path:
        store = FAISSVectorStore(path)
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        store.add_documents(
            ["مستند أ", "مستند ب"],
            embeddings,
            document_ids=["same-name-a", "same-name-b"],
            metadata_list=[
                {"workspace_id": "a"},
                {"workspace_id": "b"},
            ],
        )

        results = store.search(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=10,
            workspace_id="a",
        )
        assert [item["document_id"] for item in results] == ["same-name-a"]
        assert store.delete_document("same-name-b", workspace_id="a") is False
        assert store.delete_document("same-name-a", workspace_id="a") is True
        assert any(item["document_id"] == "same-name-b" for item in store.metadata)


def test_production_never_accepts_development_bypass(monkeypatch):
    monkeypatch.setattr(main.config, "APP_ENV", "production")
    monkeypatch.setattr(main.config, "AUTH_ENABLED", False)
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", True)
    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", True)
    request = Request({"type": "http", "client": ("127.0.0.1", 8000), "headers": []})
    with pytest.raises(HTTPException) as rejected:
        main._current_workspace(request, None)
    assert rejected.value.status_code == 503


def test_azure_environment_cannot_use_development_bypass(monkeypatch):
    monkeypatch.setenv("WEBSITE_SITE_NAME", "example")
    monkeypatch.setattr(main.config, "APP_ENV", "development")
    monkeypatch.setattr(main.config, "AUTH_ENABLED", False)
    request = Request({"type": "http", "client": ("127.0.0.1", 8000), "headers": []})
    with pytest.raises(HTTPException) as rejected:
        main._current_workspace(request, None)
    assert rejected.value.status_code == 503
