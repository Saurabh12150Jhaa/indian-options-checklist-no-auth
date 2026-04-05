"""Facade – delegates to services.oi_tracker for backward compatibility."""
from services.oi_tracker import (  # noqa: F401
    save_oi_snapshot,
    get_oi_timeline,
    get_aggregate_oi_timeline,
    get_tracked_dates,
    cleanup_old_data,
)
