"""Streamlit-cached data loaders wrapping service-layer fetchers."""

import streamlit as st

from data_fetcher import (
    fetch_global_cues,
    fetch_india_vix,
    fetch_spot_data,
    fetch_historical,
    fetch_intraday,
    fetch_nse_option_chain,
    fetch_fii_dii,
)
from news_fetcher import fetch_news


@st.cache_data(ttl=180, show_spinner=False)
def _global_cues():
    return fetch_global_cues()


@st.cache_data(ttl=120, show_spinner=False)
def _india_vix():
    return fetch_india_vix()


@st.cache_data(ttl=60, show_spinner=False)
def _spot_data(idx):
    return fetch_spot_data(idx)


@st.cache_data(ttl=120, show_spinner=False)
def _historical(idx):
    return fetch_historical(idx)


@st.cache_data(ttl=60, show_spinner=False)
def _intraday(idx, interval):
    return fetch_intraday(idx, interval)


@st.cache_data(ttl=90, show_spinner=False)
def _nse_option_chain(idx):
    return fetch_nse_option_chain(idx)


@st.cache_data(ttl=300, show_spinner=False)
def _fii_dii():
    return fetch_fii_dii()


@st.cache_data(ttl=120, show_spinner=False)
def _news(idx):
    return fetch_news(idx)
