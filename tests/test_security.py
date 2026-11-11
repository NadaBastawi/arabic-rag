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


def test_entra_identity_is_tenant_scoped_allowlisted_and_requires_same_origin(monkeypatch):
    import base64
    import json
    from app.services.production_security import workspace_for_request, validate_browser_write

    tenant = "11111111-1111-4111-8111-111111111111"
    user = "22222222-2222-4222-8222-222222222222"
    monkeypatch.setenv("DALIL_TENANT_ID", tenant)
    monkeypatch.setenv("DALIL_ALLOWED_USER_IDS", user)
    monkeypatch.setenv("WEBSITE_HOSTNAME", "pilot.azurewebsites.net")
    def request_for(oid, origin="https://pilot.azurewebsites.net", marker="1"):
        payload = {"auth_typ": "aad", "claims": [{"typ": "tid", "val": tenant}, {"typ": "oid", "val": oid}]}
        encoded = base64.b64encode(json.dumps(payload).encode())
        return Request({"type": "http", "method": "POST", "headers": [
            (b"x-ms-client-principal", encoded), (b"origin", origin.encode()),
            (b"x-dalil-request", marker.encode())]})
    good = request_for(user)
    assert len(workspace_for_request(good)) == 64
    validate_browser_write(good)
    with pytest.raises(HTTPException):
        workspace_for_request(request_for("33333333-3333-4333-8333-333333333333"))
    with pytest.raises(HTTPException):
        validate_browser_write(request_for(user, origin="https://evil.example"))
    with pytest.raises(HTTPException):
        validate_browser_write(request_for(user, marker=""))


def test_production_configuration_fails_closed(monkeypatch, tmp_path):
    import json
    from app.services.production_security import validate_production_config
    monkeypatch.setattr(main.config, "LOCAL_DEV_MODE", False)
    monkeypatch.setattr(main.config, "PRIVATE_DEV_MODE", False)
    monkeypatch.setattr(main.config, "ALLOW_EMBEDDING_FALLBACK", False)
    monkeypatch.setattr(main.config, "INDEX_PATH", str(tmp_path / "index"))
    monkeypatch.setattr(main.config, "DOCS_PATH", str(tmp_path / "documents"))
    monkeypatch.setenv("WEBSITE_SITE_NAME", "dalil")
    monkeypatch.setenv("WEBSITE_AUTH_ENABLED", "true")
    monkeypatch.setenv("WEBSITE_HOSTNAME", "pilot.azurewebsites.net")
    monkeypatch.setenv("DALIL_TENANT_ID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("DALIL_ALLOWED_USER_IDS", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv("DALIL_DATA_MOUNT", str(tmp_path))
    monkeypatch.setattr("app.services.production_security.os.path.ismount", lambda _: True)
    auth = {"platform": {"enabled": True}, "globalValidation": {"requireAuthentication": True,
        "unauthenticatedClientAction": "RedirectToLoginPage", "excludedPaths": []},
        "identityProviders": {"azureActiveDirectory": {"enabled": True}},
        "httpSettings": {"requireHttps": True}}
    monkeypatch.setenv("WEBSITE_AUTH_V2_CONFIG_JSON", json.dumps(auth))
    assert validate_production_config(main.config) == tmp_path.resolve()
    auth["globalValidation"]["requireAuthentication"] = False
    monkeypatch.setenv("WEBSITE_AUTH_V2_CONFIG_JSON", json.dumps(auth))
    with pytest.raises(RuntimeError):
        validate_production_config(main.config)
    auth["globalValidation"]["requireAuthentication"] = True
    monkeypatch.setenv("WEBSITE_AUTH_V2_CONFIG_JSON", json.dumps(auth))
    monkeypatch.setattr("app.services.production_security.os.path.ismount", lambda _: False)
    with pytest.raises(RuntimeError):
        validate_production_config(main.config)


def test_single_writer_excludes_another_process(tmp_path):
    import os
    import subprocess
    import sys
    if os.name == "nt":
        pytest.skip("Azure App Service pilot uses Linux file locks")
    from app.services.production_security import single_writer
    code = ("from pathlib import Path; from app.services.production_security import single_writer; "
            "import sys; lock=single_writer(Path(sys.argv[1])); lock.__enter__()")
    with single_writer(tmp_path):
        attempted = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
        assert attempted.returncode != 0
        assert "Another DALIL instance holds the data lock" in attempted.stderr
    released = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert released.returncode == 0
