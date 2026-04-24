"""Deprecated entrypoint.

Use `app.main:app` as the production API entrypoint.
This module is kept only for backward compatibility.
"""

from app.main import app

__all__ = ["app"]
