"""
Data adapter: transforms NSE F&O bhavcopy data into
the DataFrame format expected by BacktestEngine.

Supports both CSV formats:

Old format (up to ~mid-2024):
    INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
    OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
    OPEN_INT, CHG_IN_OI, TIMESTAMP

New format (2024+):
    TradDt, BizDt, Sgmt, Src, FinInstrmTp, FinInstrmId, ISIN,
    TckrSymb, SctySrs, XpryDt, FininstrmActlXpryDt, StrkPric,
    OptnTp, FinInstrmNm, OpnPric, HghPric, LwPric, ClsPric,
    LastPric, PrvsClsgPric, UndrlygPric, SttlmPric, OpnIntrst,
    ChngInOpnIntrst, TtlTradgVol, TtlTrfVal, TtlNbOfTxsExctd,
    SsnId, NewBrdLotQty, Rmks, Rsvd1, Rsvd2, Rsvd3, Rsvd4

Output DataFrame columns:
    date, symbol, expiry, strike, option_type,
    open, high, low, close, settle, oi, chg_oi,
    underlying_close
"""

import logging
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parse_bhavcopy_csv(filepath: str | Path) -> Optional[pd.DataFrame]:
    """
    Parse a single NSE F&O bhavcopy CSV file (or .csv.zip).
    Returns standardised DataFrame.
    """
    filepath = Path(filepath)
    try:
        if filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_name:
                    return None
                with zf.open(csv_name[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(filepath)
    except Exception as exc:
        logger.error("Failed to read bhavcopy: %s – %s", filepath, exc)
        return None

    return _standardise(df)


def parse_bhavcopy_bytes(data: bytes) -> Optional[pd.DataFrame]:
    """Parse bhavcopy from raw bytes (ZIP or CSV)."""
    try:
        bio = BytesIO(data)
        try:
            with zipfile.ZipFile(bio) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_name:
                    return None
                with zf.open(csv_name[0]) as f:
                    df = pd.read_csv(f)
        except zipfile.BadZipFile:
            bio.seek(0)
            df = pd.read_csv(bio)
    except Exception as exc:
        logger.error("Failed to parse bhavcopy bytes: %s", exc)
        return None

    return _standardise(df)


def _standardise(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map bhavcopy column names to standard engine format.
    Auto-detects old vs new NSE CSV format."""
    df.columns = [c.strip() for c in df.columns]

    # Detect format by checking for new-format columns
    upper_cols = {c.upper() for c in df.columns}
    is_new_format = "TCKRSYMB" in upper_cols or "TRADDT" in upper_cols

    if is_new_format:
        col_map = {
            "TRADDT": "date",
            "TCKRSYMB": "symbol",
            "XPRYDT": "expiry",
            "STRKPRIC": "strike",
            "OPTNTP": "option_type",
            "OPNPRIC": "open",
            "HGHPRIC": "high",
            "LWPRIC": "low",
            "CLSPRIC": "close",
            "STTLMPRIC": "settle",
            "OPNINTRST": "oi",
            "CHNGINOPNINTRST": "chg_oi",
            "FININSTRMTP": "instrument",
            "TTLTRADGVOL": "contracts",
            "TTLTRFVAL": "value_lakhs",
            "UNDRLYGPRIC": "underlying_close",
        }
    else:
        col_map = {
            "INSTRUMENT": "instrument",
            "SYMBOL": "symbol",
            "EXPIRY_DT": "expiry",
            "STRIKE_PR": "strike",
            "OPTION_TYP": "option_type",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "SETTLE_PR": "settle",
            "CONTRACTS": "contracts",
            "VAL_INLAKH": "value_lakhs",
            "OPEN_INT": "oi",
            "CHG_IN_OI": "chg_oi",
            "TIMESTAMP": "date",
        }

    # Case-insensitive rename
    rename = {}
    for src, dst in col_map.items():
        for col in df.columns:
            if col.upper() == src:
                rename[col] = dst
                break
    df = df.rename(columns=rename)

    # Filter to options only
    if is_new_format and "instrument" in df.columns:
        # New format: STO = stock options, IDO = index options
        df = df[df["instrument"].isin(["STO", "IDO"])].copy()
    elif "instrument" in df.columns:
        # Old format: OPTIDX, OPTSTK
        df = df[df["instrument"].str.contains("OPT", case=False, na=False)].copy()

    # Filter valid option types
    if "option_type" in df.columns:
        df = df[df["option_type"].isin(["CE", "PE"])].copy()

    # Parse dates — new format uses YYYY-MM-DD, old uses dd-MMM-yyyy
    for date_col in ["date", "expiry"]:
        if date_col in df.columns:
            if is_new_format:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
            else:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    # Ensure numeric columns
    for num_col in ["strike", "open", "high", "low", "close", "settle", "oi", "chg_oi",
                     "contracts", "value_lakhs", "underlying_close"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

    return df if not df.empty else None


def merge_underlying_prices(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join underlying daily close prices onto the options DataFrame.
    
    underlying_df should have columns: date, close (or symbol, date, close).
    Result adds 'underlying_close' column.
    """
    if underlying_df.empty or options_df.empty:
        return options_df

    underlying_df = underlying_df.copy()
    if "date" in underlying_df.columns:
        underlying_df["date"] = pd.to_datetime(underlying_df["date"])
    
    # Rename to avoid column conflicts
    price_col = "close" if "close" in underlying_df.columns else "Close"
    underlying_df = underlying_df.rename(columns={price_col: "underlying_close"})

    options_df = options_df.copy()
    if "date" in options_df.columns:
        options_df["date"] = pd.to_datetime(options_df["date"])

    result = options_df.merge(
        underlying_df[["date", "underlying_close"]].drop_duplicates("date"),
        on="date",
        how="left",
    )
    return result


def load_multiple_bhavcopies(
    filepaths: list[str | Path],
    symbol_filter: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate multiple bhavcopy files.
    Optionally filter to a specific symbol (e.g., 'NIFTY', 'BANKNIFTY').
    """
    frames = []
    for fp in filepaths:
        df = parse_bhavcopy_csv(fp)
        if df is not None:
            if symbol_filter and "symbol" in df.columns:
                df = df[df["symbol"] == symbol_filter]
            if not df.empty:
                frames.append(df)

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["date", "expiry", "strike"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def build_backtest_dataset(
    options_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Full pipeline: filter symbol, merge underlying, prepare for engine.
    """
    # Filter to symbol
    if "symbol" in options_df.columns:
        options_df = options_df[options_df["symbol"] == symbol].copy()

    # Merge underlying
    result = merge_underlying_prices(options_df, underlying_df)

    # Ensure date columns are date objects (not datetime)
    for col in ["date", "expiry"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col]).dt.date

    return result
