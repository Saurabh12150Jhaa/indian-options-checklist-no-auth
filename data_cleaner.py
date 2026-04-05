"""Facade – delegates to services.bhavcopy.cleaner for backward compatibility."""
from services.bhavcopy.cleaner import (  # noqa: F401
    mark_files_accessed,
    cleanup_date_range,
    get_cache_stats,
)
