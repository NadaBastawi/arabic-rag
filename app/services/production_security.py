"""Fail-closed boundary for a single Azure App Service pilot instance."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from threading import Event, Thread

from fastapi import HTTPException, Request


def _uuid(value: str) -> str:
    return str(uuid.UUID(value))


def validate_production_config(config) -> Path:
    """Check platform-injected auth config and require a mounted data root."""
    if not os.getenv("WEBSITE_SITE_NAME") or os.getenv("WEBSITE_AUTH_ENABLED", "").lower() != "true":
        raise RuntimeError("Azure App Service authentication is required")
    if config.LOCAL_DEV_MODE or config.PRIVATE_DEV_MODE or config.ALLOW_EMBEDDING_FALLBACK:
        raise RuntimeError("Development and embedding fallback modes must be disabled")
    try:
        raw = json.loads(os.environ["WEBSITE_AUTH_V2_CONFIG_JSON"])
        settings = raw.get("properties", raw)
        validation = settings["globalValidation"]
        aad = settings["identityProviders"]["azureActiveDirectory"]
        if not (settings["platform"]["enabled"] and validation["requireAuthentication"]
                and validation["unauthenticatedClientAction"] in ("RedirectToLoginPage", "Return401", "Return403")
                and not validation.get("excludedPaths") and aad["enabled"]
                and settings["httpSettings"]["requireHttps"]):
            raise ValueError("insecure authentication settings")
        _uuid(os.environ["DALIL_TENANT_ID"])
        users = [_uuid(item.strip()) for item in os.environ["DALIL_ALLOWED_USER_IDS"].split(",") if item.strip()]
        if not users:
            raise ValueError("empty allowlist")
        if not os.environ["DALIL_STORAGE_CONNECTION_STRING"]:
            raise ValueError("missing storage credential")
        container = os.environ["DALIL_CONTROL_CONTAINER"]
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])", container) or "--" in container:
            raise ValueError("invalid control container")
        host = os.environ["WEBSITE_HOSTNAME"]
        if not host or "/" in host or ":" in host:
            raise ValueError("invalid hostname")
        root = Path(os.environ["DALIL_DATA_MOUNT"]).resolve(strict=True)
        if not os.path.ismount(root):
            raise ValueError("data root is not a mount")
        for path in (config.INDEX_PATH, config.DOCS_PATH):
            target = Path(path).resolve()
            if target == root or not target.is_relative_to(root):
                raise ValueError("data paths must be inside the mount")
        index = Path(config.INDEX_PATH).resolve()
        docs = Path(config.DOCS_PATH).resolve()
        if index == docs or index.is_relative_to(docs) or docs.is_relative_to(index):
            raise ValueError("index and documents paths must be separate")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Azure identity or durable storage configuration is invalid") from exc
    return root


def workspace_for_request(request: Request) -> str:
    try:
        encoded = request.headers["x-ms-client-principal"]
        if len(encoded) > 32768:
            raise ValueError("principal too large")
        principal = json.loads(base64.b64decode(encoded, validate=True))
        if principal.get("auth_typ", "").lower() != "aad":
            raise ValueError("unexpected provider")
        claims = principal["claims"]
        def claim(*types):
            values = {_uuid(c["val"]) for c in claims if c.get("typ") in types}
            if len(values) != 1:
                raise ValueError("missing or conflicting identity claim")
            return values.pop()
        tenant = claim("tid", "http://schemas.microsoft.com/identity/claims/tenantid")
        user = claim("oid", "http://schemas.microsoft.com/identity/claims/objectidentifier")
        allowed = {_uuid(item.strip()) for item in os.environ["DALIL_ALLOWED_USER_IDS"].split(",") if item.strip()}
        if tenant != _uuid(os.environ["DALIL_TENANT_ID"]) or user not in allowed:
            raise ValueError("identity is not allowed")
    except (KeyError, TypeError, ValueError, binascii.Error, json.JSONDecodeError):
        raise HTTPException(status_code=403, detail="Authenticated user is not authorized")
    return hashlib.sha256(f"{tenant}:{user}".encode("ascii")).hexdigest()


def validate_browser_write(request: Request) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    expected = f"https://{os.environ.get('WEBSITE_HOSTNAME', '').lower()}"
    if request.headers.get("origin", "").lower() != expected or request.headers.get("x-dalil-request") != "1":
        raise HTTPException(status_code=403, detail="Invalid request origin")


class AzureIndexLease:
    """A renewable, service-enforced lease on the index generation pointer."""
    def __init__(self):
        self._stop = Event()
        self._lost = Event()
        self._thread = None
        self._blob = None
        self._lease = None

    @property
    def healthy(self) -> bool:
        return self._lease is not None and not self._lost.is_set()

    def __enter__(self):
        from azure.core.exceptions import HttpResponseError, ResourceExistsError
        from azure.storage.blob import BlobClient
        try:
            self._blob = BlobClient.from_connection_string(
                os.environ["DALIL_STORAGE_CONNECTION_STRING"],
                container_name=os.environ["DALIL_CONTROL_CONTAINER"], blob_name="CURRENT",
                retry_total=0, connection_timeout=5, read_timeout=5,
            )
            if not self._blob.exists():
                try:
                    self._blob.upload_blob(b"", overwrite=False)
                except ResourceExistsError:
                    pass
            deadline = time.monotonic() + 75
            while True:
                try:
                    self._lease = self._blob.acquire_lease(lease_duration=60)
                    break
                except HttpResponseError as exc:
                    if exc.status_code != 409 or time.monotonic() >= deadline:
                        raise
                    time.sleep(3)
        except Exception as exc:
            raise RuntimeError("Unable to acquire the Azure index lease") from exc
        self._thread = Thread(target=self._renew, name="dalil-index-lease", daemon=True)
        self._thread.start()
        return self

    def _renew(self):
        while not self._stop.wait(20):
            try:
                self._lease.renew()
            except Exception:
                self._lost.set()
                return

    def read(self) -> str:
        return self._blob.download_blob().readall().decode("ascii").strip()

    def write(self, generation: str) -> None:
        if not self.healthy:
            raise RuntimeError("Azure index lease is unavailable")
        try:
            # Azure rejects the write if the lease has expired or changed.
            self._blob.upload_blob((generation + "\n").encode("ascii"), overwrite=True, lease=self._lease)
        except Exception as exc:
            self._lost.set()
            raise RuntimeError("Azure rejected the index commit") from exc

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        try:
            if self._lease:
                self._lease.release()
        except Exception:
            # A fixed lease expires shortly after an ungraceful shutdown.
            pass
        self._lost.set()
