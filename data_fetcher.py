"""Facade – delegates to services.data_fetcher for backward compatibility."""
from services.data_fetcher import (  # noqa: F401
    fetch_global_cues,
    fetch_india_vix,
    fetch_spot_data,
    fetch_historical,
    fetch_intraday,
    fetch_nse_option_chain,
    fetch_fii_dii,
)
