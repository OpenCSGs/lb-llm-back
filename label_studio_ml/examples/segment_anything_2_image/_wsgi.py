"""Compatibility WSGI entry point for existing deployments."""

from label_studio_ml.aggregate_backend._wsgi import app

__all__ = ['app']
