"""
Application-specific exceptions for structured error handling.

Using custom exceptions instead of bare ``None`` returns makes error
boundaries explicit and lets callers decide how to degrade gracefully.
"""


class AppError(Exception):
    """Base exception for the application."""


class DataFetchError(AppError):
    """Raised when a data source fails to return data."""

    def __init__(self, source: str, detail: str = ""):
        self.source = source
        self.detail = detail
        msg = f"Failed to fetch from {source}: {detail}" if detail else f"Failed to fetch from {source}"
        super().__init__(msg)


class NSEConnectionError(DataFetchError):
    """Raised when NSE API is unreachable or rate-limiting."""

    def __init__(self, detail: str = ""):
        super().__init__("NSE API", detail)


class ConfigurationError(AppError):
    """Raised for invalid or missing configuration."""
