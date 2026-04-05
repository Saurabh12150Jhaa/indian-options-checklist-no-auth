"""Facade – delegates to services.news_service for backward compatibility."""
from services.news_service import fetch_news  # noqa: F401
