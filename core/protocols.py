"""
Service protocols (interfaces) following the Dependency Inversion Principle.

Components depend on these abstractions rather than concrete implementations,
making it straightforward to swap data sources or add new ones without
modifying consuming code (Open/Closed Principle).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    """Interface for spot / historical / intraday market data."""

    def fetch_spot(self, symbol: str) -> tuple[Optional[dict], datetime]: ...
    def fetch_historical(self, symbol: str, period: str = "1y") -> tuple[Optional[pd.DataFrame], datetime]: ...
    def fetch_intraday(self, symbol: str, interval: str = "15m") -> tuple[Optional[pd.DataFrame], datetime]: ...


@runtime_checkable
class OptionChainProvider(Protocol):
    """Interface for options chain data retrieval."""

    def fetch_option_chain(self, symbol: str) -> tuple[Optional[dict], datetime]: ...


@runtime_checkable
class GlobalDataProvider(Protocol):
    """Interface for global / macro market data."""

    def fetch_global_cues(self) -> tuple[pd.DataFrame, datetime]: ...
    def fetch_india_vix(self) -> tuple[Optional[float], datetime]: ...
    def fetch_fii_dii(self) -> tuple[Optional[pd.DataFrame], datetime]: ...


@runtime_checkable
class NewsProvider(Protocol):
    """Interface for news retrieval and classification."""

    def fetch_news(self, index_name: str, max_items: int = 50) -> tuple[pd.DataFrame, datetime]: ...
