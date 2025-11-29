"""
Main entry point for the FastAPI application
This file is an alias to server.py for compatibility with ASGI servers
that expect 'main:app' as the application location.
"""

from server import app

__all__ = ["app"]

