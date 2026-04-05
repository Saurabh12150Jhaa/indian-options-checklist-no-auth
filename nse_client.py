"""Facade – delegates to services.nse_client for backward compatibility."""
from services.nse_client import NSEClient, get_nse_client  # noqa: F401
