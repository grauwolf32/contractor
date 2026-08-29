"""Compatibility import for the Runtime Agent ASGI application factory."""

from contractor_runtime.server import create_app

__all__ = ["create_app"]
