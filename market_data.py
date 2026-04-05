"""Facade – delegates to services.market_data for backward compatibility."""
from services.market_data import (  # noqa: F401
    fetch_stock_option_chain,
    fetch_top_gainers,
    fetch_top_losers,
    fetch_advances_declines,
    fetch_market_turnover,
    fetch_live_index,
    fetch_pre_open,
)
