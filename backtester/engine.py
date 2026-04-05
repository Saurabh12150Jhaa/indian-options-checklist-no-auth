"""
Backtesting engine for Indian F&O options strategies.
Works with NSE bhavcopy-style daily options data.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Single trade result."""
    entry_date: date
    exit_date: date
    strategy_name: str
    legs: list[dict]  # each: {type, strike, option_type, entry_price, exit_price, qty}
    entry_premium: float  # net position value at entry: BUY legs positive, SELL legs negative
    exit_premium: float
    pnl: float
    pnl_pct: float
    lot_size: int
    total_pnl: float  # pnl * lot_size
    dte_at_entry: int
    underlying_entry: float
    underlying_exit: float
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str
    lot_size: int
    start_date: date
    end_date: date
    initial_capital: float = 500_000.0
    dte_min: int = 7
    dte_max: int = 45
    exit_dte: int = 0  # 0 = hold to expiry
    stop_loss_pct: Optional[float] = None  # e.g. 50 means exit if loss > 50% of premium
    target_pct: Optional[float] = None  # e.g. 50 means exit if profit > 50% of premium
    max_open_trades: int = 1
    entry_days: list[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    slippage_pct: float = 0.5


@dataclass 
class BacktestReport:
    """Aggregated backtest results."""
    strategy_name: str
    symbol: str
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_single_win: float
    max_single_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    initial_capital: float
    final_capital: float
    return_pct: float
    trades: list[TradeResult]
    equity_curve: pd.DataFrame  # date, equity


class BacktestEngine:
    """
    Core backtesting engine. Takes a strategy callable and options data,
    simulates trades, and produces a BacktestReport.
    """

    def __init__(self, config: BacktestConfig, data: pd.DataFrame):
        """
        Args:
            config: Backtest configuration
            data: DataFrame with columns:
                date, symbol, expiry, strike, option_type (CE/PE),
                open, high, low, close, settle, oi, chg_oi,
                underlying_close (joined from index/equity daily data)
        """
        self.config = config
        self.data = data.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Pre-process and index the data for fast lookups."""
        df = self.data
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
        df["dte"] = df.apply(
            lambda r: (r["expiry"] - r["date"]).days if pd.notna(r.get("expiry")) and pd.notna(r.get("date")) else 0,
            axis=1,
        )
        # Mid price as fair value (close is settlement, use as proxy)
        df["mid"] = df["close"]
        self.data = df
        self.trading_dates = sorted(df["date"].unique())

    def get_chain_on_date(self, dt: date, expiry: Optional[date] = None) -> pd.DataFrame:
        """Get the option chain snapshot for a given date and optionally a specific expiry."""
        mask = self.data["date"] == dt
        if expiry:
            mask &= self.data["expiry"] == expiry
        return self.data[mask].copy()

    def find_nearest_expiry(self, dt: date, dte_min: int, dte_max: int) -> Optional[date]:
        """Find the nearest expiry within DTE range for a given date."""
        chain = self.get_chain_on_date(dt)
        if chain.empty:
            return None
        expiries = chain[
            (chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)
        ]["expiry"].unique()
        if len(expiries) == 0:
            return None
        return min(expiries)

    def find_strike_near_spot(
        self, chain: pd.DataFrame, spot: float,
        option_type: str, offset_pct: float = 0.0,
    ) -> Optional[float]:
        """
        Find strike nearest to spot * (1 + offset_pct/100).
        option_type: 'CE' or 'PE'
        offset_pct: positive = OTM for calls, negative = OTM for puts
        """
        sub = chain[chain["option_type"] == option_type]
        if sub.empty:
            return None
        target = spot * (1 + offset_pct / 100)
        idx = (sub["strike"] - target).abs().idxmin()
        return float(sub.loc[idx, "strike"])

    def get_option_price(
        self, dt: date, expiry: date, strike: float, option_type: str,
    ) -> Optional[float]:
        """Get the close/settle price for a specific option on a date."""
        mask = (
            (self.data["date"] == dt)
            & (self.data["expiry"] == expiry)
            & (self.data["strike"] == strike)
            & (self.data["option_type"] == option_type)
        )
        rows = self.data[mask]
        if rows.empty:
            return None
        return float(rows.iloc[0]["close"])

    def get_underlying_price(self, dt: date) -> Optional[float]:
        """Get underlying close price on a date."""
        rows = self.data[self.data["date"] == dt]
        if rows.empty:
            return None
        # Support both column names (new-format maps to underlying_close,
        # but underlying_price may appear if data_adapter version differs)
        for col in ("underlying_close", "underlying_price"):
            if col in rows.columns:
                val = rows.iloc[0][col]
                if pd.notna(val) and float(val) > 0:
                    return float(val)
        return None

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to a trade price."""
        slip = price * (self.config.slippage_pct / 100)
        return price + slip if is_buy else price - slip

    def run(self, strategy_fn) -> BacktestReport:
        """
        Run a backtest with the given strategy function.
        
        strategy_fn(engine, dt, chain, spot, expiry) -> list[dict] or None
            Each dict: {option_type: 'CE'/'PE', strike: float, action: 'BUY'/'SELL', qty: int}
            Returns None if no trade should be taken on this date.
        """
        config = self.config
        trades: list[TradeResult] = []
        equity = config.initial_capital
        equity_points: list[dict] = []
        open_positions: list[dict] = []  # tracks positions awaiting exit
        
        day_names = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}

        for dt in self.trading_dates:
            if dt < config.start_date or dt > config.end_date:
                continue

            # Check exit conditions for open positions
            positions_to_close = []
            for i, pos in enumerate(open_positions):
                should_exit = False
                exit_reason = ""

                # Expiry exit
                if dt >= pos["expiry"] or (config.exit_dte > 0 and (pos["expiry"] - dt).days <= config.exit_dte):
                    should_exit = True
                    exit_reason = "expiry/dte"

                # Stop loss / target check
                if not should_exit:
                    current_value = 0.0
                    for leg in pos["legs"]:
                        price = self.get_option_price(dt, pos["expiry"], leg["strike"], leg["option_type"])
                        if price is None:
                            continue
                        multiplier = 1 if leg["action"] == "BUY" else -1
                        current_value += price * multiplier * leg["qty"]
                    
                    entry_value = pos["entry_premium"]
                    # Position value convention: BUY=+, SELL=-
                    # P&L = change in position value = current - entry
                    # Credit (entry_value<0): options decay → current less negative → positive P&L
                    # Debit (entry_value>0): options gain → current more positive → positive P&L
                    unrealised_pnl = current_value - entry_value
                    
                    if config.stop_loss_pct and unrealised_pnl < 0:
                        loss_pct = abs(unrealised_pnl / abs(entry_value)) * 100 if entry_value != 0 else 0
                        if loss_pct >= config.stop_loss_pct:
                            should_exit = True
                            exit_reason = "stop_loss"
                    
                    if config.target_pct and unrealised_pnl > 0:
                        profit_pct = (unrealised_pnl / abs(entry_value)) * 100 if entry_value != 0 else 0
                        if profit_pct >= config.target_pct:
                            should_exit = True
                            exit_reason = "target"

                if should_exit:
                    positions_to_close.append(i)
                    # Calculate exit
                    exit_premium = 0.0
                    leg_results = []
                    for leg in pos["legs"]:
                        exit_price = self.get_option_price(dt, pos["expiry"], leg["strike"], leg["option_type"])
                        if exit_price is None:
                            # Option expired worthless or no data
                            exit_price = 0.0
                        is_buy_to_close = leg["action"] == "SELL"
                        exit_price = self.apply_slippage(exit_price, is_buy_to_close)
                        multiplier = 1 if leg["action"] == "BUY" else -1
                        exit_premium += exit_price * multiplier * leg["qty"]
                        leg_results.append({
                            "type": leg["action"],
                            "strike": leg["strike"],
                            "option_type": leg["option_type"],
                            "entry_price": leg["entry_price"],
                            "exit_price": exit_price,
                            "qty": leg["qty"],
                        })

                    entry_prem = pos["entry_premium"]
                    # PnL = change in position value (exit - entry)
                    # Credit (entry<0): options decay → exit less negative → positive PnL
                    # Debit (entry>0): options gain → exit more positive → positive PnL
                    pnl = exit_premium - entry_prem
                    total_pnl = pnl * config.lot_size
                    equity += total_pnl

                    underlying_exit = self.get_underlying_price(dt) or pos["underlying_entry"]
                    
                    trades.append(TradeResult(
                        entry_date=pos["entry_date"],
                        exit_date=dt,
                        strategy_name=pos["strategy_name"],
                        legs=leg_results,
                        entry_premium=entry_prem,
                        exit_premium=exit_premium,
                        pnl=round(pnl, 2),
                        pnl_pct=round((pnl / abs(entry_prem)) * 100, 2) if entry_prem != 0 else 0,
                        lot_size=config.lot_size,
                        total_pnl=round(total_pnl, 2),
                        dte_at_entry=pos["dte_at_entry"],
                        underlying_entry=pos["underlying_entry"],
                        underlying_exit=underlying_exit,
                    ))

            # Remove closed positions (reverse order to preserve indices)
            for i in sorted(positions_to_close, reverse=True):
                open_positions.pop(i)

            # Check if we should enter new trades
            day_name = day_names.get(dt.weekday(), "")
            if (
                day_name in config.entry_days
                and len(open_positions) < config.max_open_trades
            ):
                expiry = self.find_nearest_expiry(dt, config.dte_min, config.dte_max)
                if expiry:
                    chain = self.get_chain_on_date(dt, expiry)
                    spot = self.get_underlying_price(dt)
                    if spot and not chain.empty:
                        signal = strategy_fn(self, dt, chain, spot, expiry)
                        if signal:
                            # Build position
                            entry_premium = 0.0
                            legs = []
                            valid = True
                            for leg_spec in signal:
                                price = self.get_option_price(
                                    dt, expiry, leg_spec["strike"], leg_spec["option_type"],
                                )
                                if price is None or price <= 0:
                                    valid = False
                                    break
                                is_buy = leg_spec["action"] == "BUY"
                                adj_price = self.apply_slippage(price, is_buy)
                                multiplier = 1 if is_buy else -1
                                qty = leg_spec.get("qty", 1)
                                entry_premium += adj_price * multiplier * qty
                                legs.append({
                                    **leg_spec,
                                    "entry_price": adj_price,
                                })

                            if valid and legs:
                                dte = (expiry - dt).days
                                strat_name = legs[0].get("strategy_name", "Custom")
                                open_positions.append({
                                    "entry_date": dt,
                                    "expiry": expiry,
                                    "strategy_name": strat_name,
                                    "legs": legs,
                                    "entry_premium": entry_premium,
                                    "dte_at_entry": dte,
                                    "underlying_entry": spot,
                                })

            equity_points.append({"date": dt, "equity": round(equity, 2)})

        # Build report
        equity_df = pd.DataFrame(equity_points)
        return self._build_report(trades, equity_df, config)

    def _build_report(
        self, trades: list[TradeResult], equity_df: pd.DataFrame, config: BacktestConfig,
    ) -> BacktestReport:
        """Compute aggregate statistics from trade results."""
        total = len(trades)
        if total == 0:
            return BacktestReport(
                strategy_name="N/A", symbol=config.symbol,
                period=f"{config.start_date} to {config.end_date}",
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_pnl_per_trade=0,
                max_single_win=0, max_single_loss=0,
                max_drawdown=0, max_drawdown_pct=0,
                sharpe_ratio=0, profit_factor=0,
                initial_capital=config.initial_capital,
                final_capital=config.initial_capital,
                return_pct=0, trades=[], equity_curve=equity_df,
            )

        pnls = [t.total_pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(winners) / total * 100

        # Max drawdown from equity curve
        if not equity_df.empty:
            peak = equity_df["equity"].cummax()
            drawdown = equity_df["equity"] - peak
            max_dd = float(drawdown.min())
            max_dd_pct = (max_dd / peak[drawdown.idxmin()]) * 100 if peak[drawdown.idxmin()] > 0 else 0
        else:
            max_dd = 0
            max_dd_pct = 0

        # Sharpe ratio (annualised, assuming ~252 trading days)
        if len(pnls) > 1:
            returns = np.array(pnls) / config.initial_capital
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / max(total, 1)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        final_capital = config.initial_capital + total_pnl

        return BacktestReport(
            strategy_name=trades[0].strategy_name if trades else "N/A",
            symbol=config.symbol,
            period=f"{config.start_date} to {config.end_date}",
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 1),
            total_pnl=round(total_pnl, 2),
            avg_pnl_per_trade=round(total_pnl / total, 2),
            max_single_win=round(max(pnls), 2) if pnls else 0,
            max_single_loss=round(min(pnls), 2) if pnls else 0,
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            initial_capital=config.initial_capital,
            final_capital=round(final_capital, 2),
            return_pct=round((total_pnl / config.initial_capital) * 100, 2),
            trades=trades,
            equity_curve=equity_df,
        )
