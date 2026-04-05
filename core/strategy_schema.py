"""
Comprehensive Strategy Schema — parse, validate, and convert rich JSON
strategy definitions into the app's CustomStrategy format.

Handles the full strategy JSON format including:
- strategy_meta (name, market, instruments, capital, broker requirements)
- market_hours (entry/exit windows, avoid windows)
- pre_market_analysis (global cues, VIX thresholds, OI analysis)
- strategies (entry rules, exit rules, strike selection, adjustments)
- risk_management (daily/weekly/monthly limits, per-trade rules, costs)
- greeks_management (delta, theta, vega, gamma guidance)
- technical_indicators (primary + secondary with weights)
- day_classification (trending/range/volatile/expiry mappings)
- lot_sizes
- trade_journal_template
- automation_config
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  FULL STRATEGY CONFIG — parsed from rich JSON
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class TimeWindow:
    """A trading time window (IST)."""
    start: str  # "09:20"
    end: str  # "09:30"
    label: str = ""

    def contains(self, time_str: str) -> bool:
        return self.start <= time_str <= self.end


@dataclass
class ExitRules:
    """Comprehensive exit rules."""
    target_pct: list[float] = field(default_factory=lambda: [50.0])
    stop_loss_pct: float = 30.0
    trailing_stop: Optional[dict] = None  # {activation: 15, trail_percent: 10}
    time_based_exit_minutes: Optional[int] = None
    max_holding_minutes: Optional[int] = None
    partial_booking: Optional[dict] = None  # {target_1: 50, target_2: 30, trail_rest: true}
    mandatory_exit_time: str = "15:15"


@dataclass
class PositionSizing:
    """Position sizing rules."""
    max_capital_per_trade_pct: float = 5.0
    max_lots: int = 2
    scale_in: bool = False
    max_margin_usage_pct: float = 60.0


@dataclass
class AdjustmentRules:
    """Position adjustment rules."""
    trigger: str = ""  # when to adjust
    actions: list[str] = field(default_factory=list)
    max_adjustments: int = 3
    hedge_trigger: str = ""  # when to add hedge


@dataclass
class StrategyEntry:
    """A complete strategy definition from the rich JSON."""
    name: str
    strategy_type: str  # directional_buying / premium_selling / premium_selling_hedged
    description: str
    best_conditions: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    entry_window: Optional[TimeWindow] = None
    entry_rules: dict = field(default_factory=dict)
    exit_rules: Optional[ExitRules] = None
    position_sizing: Optional[PositionSizing] = None
    adjustment_rules: Optional[AdjustmentRules] = None
    strike_selection: dict = field(default_factory=dict)
    legs: list[dict] = field(default_factory=list)
    margin_required: dict = field(default_factory=dict)
    custom_strategy_config: Optional[dict] = None  # converted format for CustomStrategy


@dataclass
class RiskManagement:
    """Full risk management config."""
    max_daily_loss_pct: float = 2.0
    max_daily_loss_inr: float = 20000.0
    max_daily_profit_target_pct: float = 3.0
    stop_after_consecutive_losses: int = 3
    max_trades_per_day: int = 6
    max_weekly_loss_pct: float = 5.0
    max_monthly_drawdown_pct: float = 10.0
    max_risk_per_trade_pct: float = 1.5
    max_capital_per_trade_pct: float = 10.0
    max_concurrent_positions: int = 3
    no_averaging: bool = True
    no_doubling_down: bool = True
    slippage_pct: float = 0.5
    total_round_trip_cost_pct: float = 0.15


@dataclass
class FullStrategyConfig:
    """Complete parsed strategy configuration."""
    name: str
    version: str = "1.0"
    market: str = "NSE"
    instruments: list[str] = field(default_factory=list)
    style: str = "Intraday"
    capital_required_min: float = 500000.0
    capital_recommended: float = 2000000.0
    strategies: list[StrategyEntry] = field(default_factory=list)
    risk_management: Optional[RiskManagement] = None
    day_classification: dict = field(default_factory=dict)
    greeks_guidance: dict = field(default_factory=dict)
    technical_indicators: dict = field(default_factory=dict)
    lot_sizes: dict = field(default_factory=dict)
    market_hours: dict = field(default_factory=dict)
    automation_config: dict = field(default_factory=dict)
    raw_json: dict = field(default_factory=dict)  # preserve original


# ══════════════════════════════════════════════════════════════════════════
#  PARSER — rich JSON → FullStrategyConfig
# ══════════════════════════════════════════════════════════════════════════


def parse_strategy_json(raw: dict) -> FullStrategyConfig:
    """
    Parse a comprehensive strategy JSON (like the Indian Options Intraday
    Master Strategy) into a FullStrategyConfig with all sub-components.
    """
    meta = raw.get("strategy_meta", {})
    config = FullStrategyConfig(
        name=meta.get("name", "Imported Strategy"),
        version=meta.get("version", "1.0"),
        market=meta.get("market", "NSE"),
        instruments=meta.get("instruments", []),
        style=meta.get("style", "Intraday"),
        capital_required_min=meta.get("capital_required_min_inr", 500000),
        capital_recommended=meta.get("capital_recommended_inr", 2000000),
        market_hours=raw.get("market_hours", {}),
        lot_sizes=raw.get("lot_sizes", {}),
        greeks_guidance=raw.get("greeks_management", {}),
        technical_indicators=raw.get("technical_indicators", {}),
        day_classification=raw.get("day_classification", {}),
        automation_config=raw.get("automation_config", {}),
        raw_json=raw,
    )

    # Parse risk management
    rm = raw.get("risk_management", {})
    if rm:
        daily = rm.get("daily_limits", {})
        weekly = rm.get("weekly_limits", {})
        monthly = rm.get("monthly_limits", {})
        per_trade = rm.get("per_trade_rules", {})
        costs = rm.get("slippage_and_costs", {})
        config.risk_management = RiskManagement(
            max_daily_loss_pct=daily.get("max_daily_loss_percent_of_capital", 2.0),
            max_daily_loss_inr=daily.get("max_daily_loss_inr", 20000),
            max_daily_profit_target_pct=daily.get("max_daily_profit_target_percent", 3.0),
            stop_after_consecutive_losses=daily.get("stop_trading_after_consecutive_losses", 3),
            max_trades_per_day=daily.get("max_trades_per_day", 6),
            max_weekly_loss_pct=weekly.get("max_weekly_loss_percent", 5.0),
            max_monthly_drawdown_pct=monthly.get("max_monthly_drawdown_percent", 10.0),
            max_risk_per_trade_pct=per_trade.get("max_risk_per_trade_percent", 1.5),
            max_capital_per_trade_pct=per_trade.get("max_capital_in_single_trade_percent", 10.0),
            max_concurrent_positions=per_trade.get("max_concurrent_positions", 3),
            no_averaging=per_trade.get("no_averaging_losing_positions", True),
            no_doubling_down=per_trade.get("no_doubling_down", True),
            slippage_pct=costs.get("expected_slippage_percent", 0.5),
            total_round_trip_cost_pct=costs.get("total_round_trip_cost_estimate_percent", 0.15),
        )

    # Parse strategies
    strategies_raw = raw.get("strategies", {})
    for key, strat in strategies_raw.items():
        entry = _parse_strategy_entry(key, strat)
        config.strategies.append(entry)

    return config


def _parse_strategy_entry(key: str, strat: dict) -> StrategyEntry:
    """Parse a single strategy entry from the JSON."""
    entry = StrategyEntry(
        name=strat.get("name", key),
        strategy_type=strat.get("type", "unknown"),
        description=strat.get("description", ""),
        best_conditions=strat.get("best_conditions", []),
        instruments=strat.get("instruments", []),
        entry_rules=strat.get("entry_rules", {}),
        strike_selection=strat.get("strike_selection", {}),
        margin_required=strat.get("margin_required_approximate", {}),
    )

    # Parse entry window from entry_rules.time if present
    time_str = strat.get("entry_rules", {}).get("time", "")
    if time_str and "-" in time_str:
        parts = time_str.replace(" IST", "").split(" - ")
        if len(parts) == 2:
            entry.entry_window = TimeWindow(start=parts[0].strip(), end=parts[1].strip(), label="Entry Window")

    # Parse exit rules
    exit_raw = strat.get("exit_rules", {})
    if exit_raw:
        targets = exit_raw.get("target_percent", [])
        if isinstance(targets, (int, float)):
            targets = [targets]
        elif isinstance(targets, str):
            # Parse strings like "50% of total premium collected"
            import re
            nums = re.findall(r'(\d+(?:\.\d+)?)', str(targets))
            targets = [float(n) for n in nums] if nums else [50.0]

        sl = exit_raw.get("stop_loss_percent", exit_raw.get("stop_loss_option_premium_max_percent", 30.0))
        if isinstance(sl, str):
            import re
            nums = re.findall(r'(\d+(?:\.\d+)?)', sl)
            sl = float(nums[0]) if nums else 30.0

        entry.exit_rules = ExitRules(
            target_pct=targets if targets else [50.0],
            stop_loss_pct=float(sl),
            trailing_stop=exit_raw.get("trailing_stop"),
            max_holding_minutes=exit_raw.get("max_holding_time_minutes"),
            mandatory_exit_time=exit_raw.get("time_exit", "15:15"),
        )

        # Partial booking
        partial = exit_raw.get("partial_booking")
        if partial:
            entry.exit_rules.partial_booking = {"description": partial}

    # Parse position sizing
    ps_raw = strat.get("position_sizing", {})
    if ps_raw:
        entry.position_sizing = PositionSizing(
            max_capital_per_trade_pct=ps_raw.get("max_capital_per_trade_percent", 5.0),
            max_lots=ps_raw.get("max_lots", ps_raw.get("max_lots_nifty", 2)),
            scale_in=ps_raw.get("scale_in", False),
            max_margin_usage_pct=ps_raw.get("max_margin_usage_percent", 60.0),
        )

    # Parse adjustment rules
    adj_raw = strat.get("adjustment_rules", {})
    if adj_raw:
        actions = []
        for k, v in adj_raw.items():
            if k.startswith("action") and isinstance(v, str):
                actions.append(v)
        entry.adjustment_rules = AdjustmentRules(
            trigger=adj_raw.get("when", ""),
            actions=actions,
            max_adjustments=adj_raw.get("max_adjustments", 3),
            hedge_trigger=adj_raw.get("hedge_addition", ""),
        )

    # Parse legs if explicitly defined
    legs_raw = strat.get("legs", [])
    if legs_raw:
        entry.legs = legs_raw

    # Convert to CustomStrategy-compatible config
    entry.custom_strategy_config = _convert_to_custom_strategy(entry)

    return entry


def _convert_to_custom_strategy(entry: StrategyEntry) -> dict:
    """
    Convert a StrategyEntry into a config dict that can be fed
    to CustomStrategy(config) from backtester/custom_strategy.py.

    Maps the rich JSON parameters to available conditions and leg templates.
    """
    config = {"name": entry.name}
    conditions = []
    legs = []

    # Derive conditions from entry_rules and best_conditions
    rules = entry.entry_rules
    if isinstance(rules, dict):
        # Technical confirmations
        tech_rules = rules.get("technical_confirmation", [])
        if isinstance(tech_rules, list):
            for rule in tech_rules:
                rule_lower = rule.lower()
                if "vwap" in rule_lower:
                    pos = "above" if "above" in rule_lower else "below"
                    conditions.append({"type": "vwap", "params": {"position": pos}})
                elif "rsi" in rule_lower:
                    if "above" in rule_lower or "crosses above" in rule_lower:
                        conditions.append({"type": "rsi", "params": {"period": 14, "operator": ">", "value": 60}})
                    elif "below" in rule_lower:
                        conditions.append({"type": "rsi", "params": {"period": 14, "operator": "<", "value": 40}})
                    elif "divergence" in rule_lower:
                        conditions.append({"type": "rsi", "params": {"period": 14, "operator": "<", "value": 35}})
                elif "ema" in rule_lower or "moving average" in rule_lower:
                    if "above" in rule_lower:
                        conditions.append({"type": "price_vs_ema", "params": {"period": 20, "position": "above"}})
                    elif "below" in rule_lower:
                        conditions.append({"type": "price_vs_ema", "params": {"period": 20, "position": "below"}})
                    elif "cross" in rule_lower:
                        direction = "above" if "above" in rule_lower else "below"
                        conditions.append({"type": "ema_cross", "params": {"fast": 9, "slow": 21, "direction": direction}})
                elif "supertrend" in rule_lower:
                    signal = "bullish" if "bullish" in rule_lower or "above" in rule_lower else "bearish"
                    conditions.append({"type": "supertrend", "params": {"period": 10, "multiplier": 3.0, "signal": signal}})

        # OI confirmation
        oi_confirm = rules.get("oi_confirmation", "")
        if oi_confirm:
            if "call" in oi_confirm.lower() and "build" in oi_confirm.lower():
                conditions.append({"type": "oi_change", "params": {"buildup": "long_buildup"}})
            elif "put" in oi_confirm.lower():
                conditions.append({"type": "oi_change", "params": {"buildup": "put_writing"}})

        # Deviation / gap checks
        deviation = rules.get("deviation", "")
        if deviation and "vwap" in deviation.lower():
            conditions.append({"type": "vwap", "params": {"position": "below"}})

        # Reversal candle
        reversal = rules.get("reversal_candle", "")
        if reversal:
            conditions.append({"type": "candle_pattern", "params": {"pattern": "engulfing", "direction": "BULLISH"}})

        # Strategy type (short straddle / strangle)
        strategy_type = rules.get("strategy_type", "")
        if strategy_type == "short_strangle":
            conditions.append({"type": "day_of_week", "params": {"days": ["thu"]}})

    # Derive conditions from best_conditions
    for bc in entry.best_conditions:
        bc_lower = bc.lower()
        if "vix" in bc_lower:
            import re
            nums = re.findall(r'(\d+)', bc)
            if len(nums) >= 1:
                val = float(nums[-1])
                if "<" in bc_lower:
                    conditions.append({"type": "iv_rank", "params": {"operator": "<", "value": val}})
        if "trending" in bc_lower:
            conditions.append({"type": "trend_strength", "params": {"period": 14, "operator": ">", "value": 25}})
        if "range" in bc_lower:
            conditions.append({"type": "trend_strength", "params": {"period": 14, "operator": "<", "value": 20}})
        if "expiry" in bc_lower or "thursday" in bc_lower:
            conditions.append({"type": "day_of_week", "params": {"days": ["thu"]}})

    # Remove duplicate conditions (same type)
    seen_types = set()
    deduped = []
    for c in conditions:
        key = (c["type"], str(c["params"]))
        if key not in seen_types:
            seen_types.add(key)
            deduped.append(c)
    conditions = deduped[:5]  # max 5 conditions

    # Convert legs
    if entry.legs:
        for leg_raw in entry.legs:
            action = leg_raw.get("action", "BUY")
            otype = leg_raw.get("type", "CE")
            lots = leg_raw.get("lots", 1)
            strike = leg_raw.get("strike", "ATM")

            # Parse strike into template or offset
            strike_str = str(strike).upper().strip()
            if strike_str == "ATM":
                template = f"atm_{otype.lower()}"
                legs.append({"template": template, "action": action, "qty": lots})
            elif "ATM" in strike_str:
                # Parse offsets like "ATM + 200 (Nifty)" or "ATM - 500 (BankNifty)"
                import re
                match = re.search(r'ATM\s*([+-])\s*(\d+)', strike_str)
                if match:
                    sign = 1 if match.group(1) == "+" else -1
                    points = int(match.group(2)) * sign
                    legs.append({
                        "option_type": otype, "offset_points": points,
                        "action": action, "qty": lots,
                    })
                else:
                    legs.append({"template": f"atm_{otype.lower()}", "action": action, "qty": lots})
            else:
                # Fallback to ATM
                legs.append({"template": f"atm_{otype.lower()}", "action": action, "qty": lots})
    else:
        # Infer legs from strategy type
        if entry.strategy_type == "directional_buying":
            legs.append({"template": "atm_call", "action": "BUY", "qty": 1})
        elif entry.strategy_type == "premium_selling":
            legs.extend([
                {"template": "atm_call", "action": "SELL", "qty": 1},
                {"template": "atm_put", "action": "SELL", "qty": 1},
            ])
        elif entry.strategy_type == "premium_selling_hedged":
            legs.extend([
                {"template": "otm_call_1", "action": "SELL", "qty": 1},
                {"template": "otm_call_2", "action": "BUY", "qty": 1},
                {"template": "otm_put_1", "action": "SELL", "qty": 1},
                {"template": "otm_put_2", "action": "BUY", "qty": 1},
            ])

    config["conditions"] = conditions
    config["condition_logic"] = "AND"
    config["legs"] = legs

    return config


# ══════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """Result of strategy JSON validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategies_found: int = 0
    conditions_mapped: int = 0
    conditions_unmapped: int = 0


def validate_strategy_json(raw: dict) -> ValidationResult:
    """Validate a comprehensive strategy JSON and report issues."""
    result = ValidationResult(valid=True)

    # Check required top-level keys
    if "strategy_meta" not in raw:
        result.warnings.append("Missing 'strategy_meta' — strategy name will default to 'Imported Strategy'")
    if "strategies" not in raw:
        result.errors.append("Missing 'strategies' key — no strategies to import")
        result.valid = False
        return result

    strategies = raw.get("strategies", {})
    if not isinstance(strategies, dict):
        result.errors.append("'strategies' must be a dictionary of strategy definitions")
        result.valid = False
        return result

    result.strategies_found = len(strategies)
    if result.strategies_found == 0:
        result.errors.append("No strategies defined in 'strategies' dictionary")
        result.valid = False
        return result

    # Validate each strategy
    for key, strat in strategies.items():
        if not isinstance(strat, dict):
            result.errors.append(f"Strategy '{key}' is not a dictionary")
            continue
        if "name" not in strat and "description" not in strat:
            result.warnings.append(f"Strategy '{key}' has no name or description")
        if "type" not in strat:
            result.warnings.append(f"Strategy '{key}' has no 'type' field — will default to 'unknown'")

        # Check if we can map entry rules to conditions
        entry = _parse_strategy_entry(key, strat)
        if entry.custom_strategy_config:
            conds = entry.custom_strategy_config.get("conditions", [])
            result.conditions_mapped += len(conds)
            if not conds:
                result.warnings.append(f"Strategy '{entry.name}' — could not map any entry rules to conditions")
                result.conditions_unmapped += 1
            legs_converted = entry.custom_strategy_config.get("legs", [])
            if not legs_converted:
                result.warnings.append(f"Strategy '{entry.name}' — no legs could be derived")

    # Validate risk management
    rm = raw.get("risk_management", {})
    if rm:
        daily = rm.get("daily_limits", {})
        if daily.get("max_daily_loss_percent_of_capital", 0) > 5:
            result.warnings.append("Daily loss limit > 5% is very aggressive")

    return result


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════════


def summarize_strategy_config(config: FullStrategyConfig) -> str:
    """Generate a human-readable summary of the full strategy config."""
    parts = []
    parts.append(f"## {config.name} (v{config.version})")
    parts.append(f"**Market:** {config.market} | **Style:** {config.style}")
    parts.append(f"**Instruments:** {', '.join(config.instruments) if config.instruments else 'Not specified'}")
    parts.append(f"**Capital:** Min INR {config.capital_required_min:,.0f} / Recommended INR {config.capital_recommended:,.0f}")
    parts.append("")

    # Strategies
    parts.append(f"### Strategies ({len(config.strategies)})")
    for i, strat in enumerate(config.strategies, 1):
        risk_emoji = {"low": "LOW", "medium": "MED", "high": "HIGH"}.get(strat.strategy_type, "")
        parts.append(f"**{i}. {strat.name}** ({strat.strategy_type})")
        parts.append(f"   {strat.description}")
        if strat.best_conditions:
            parts.append(f"   Best when: {', '.join(strat.best_conditions)}")
        if strat.entry_window:
            parts.append(f"   Entry: {strat.entry_window.start}-{strat.entry_window.end} IST")
        if strat.exit_rules:
            parts.append(f"   Target: {strat.exit_rules.target_pct}% | SL: {strat.exit_rules.stop_loss_pct}%")
        if strat.legs:
            leg_strs = []
            for leg in strat.legs:
                leg_strs.append(f"{leg.get('action', '?')} {leg.get('type', '?')} @ {leg.get('strike', 'ATM')}")
            parts.append(f"   Legs: {', '.join(leg_strs)}")
        parts.append("")

    # Risk management
    if config.risk_management:
        rm = config.risk_management
        parts.append("### Risk Management")
        parts.append(f"- Daily loss limit: {rm.max_daily_loss_pct}% (INR {rm.max_daily_loss_inr:,.0f})")
        parts.append(f"- Max trades/day: {rm.max_trades_per_day}")
        parts.append(f"- Stop after {rm.stop_after_consecutive_losses} consecutive losses")
        parts.append(f"- Max risk/trade: {rm.max_risk_per_trade_pct}%")
        parts.append(f"- Max concurrent positions: {rm.max_concurrent_positions}")
        parts.append(f"- Weekly limit: {rm.max_weekly_loss_pct}% | Monthly: {rm.max_monthly_drawdown_pct}%")
        parts.append("")

    # Day classification
    if config.day_classification:
        parts.append("### Day Classification Rules")
        for day_type, rules in config.day_classification.items():
            if isinstance(rules, dict):
                signs = rules.get("signs", [])
                preferred = rules.get("preferred_strategies", [])
                avoid = rules.get("avoid_strategies", [])
                parts.append(f"**{day_type.replace('_', ' ').title()}**")
                if signs:
                    parts.append(f"  Signs: {', '.join(signs)}")
                if preferred:
                    parts.append(f"  Use: {', '.join(preferred)}")
                if avoid:
                    parts.append(f"  Avoid: {', '.join(avoid)}")
        parts.append("")

    return "\n".join(parts)


def load_strategy_from_json_string(json_string: str) -> tuple[Optional[FullStrategyConfig], ValidationResult]:
    """Load and validate a strategy from a JSON string."""
    try:
        raw = json.loads(json_string)
    except json.JSONDecodeError as e:
        return None, ValidationResult(valid=False, errors=[f"Invalid JSON: {e}"])

    validation = validate_strategy_json(raw)
    if not validation.valid:
        return None, validation

    config = parse_strategy_json(raw)
    return config, validation


def load_strategy_from_file(file_content: bytes) -> tuple[Optional[FullStrategyConfig], ValidationResult]:
    """Load and validate a strategy from uploaded file content."""
    try:
        raw = json.loads(file_content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, ValidationResult(valid=False, errors=[f"Failed to parse file: {e}"])

    validation = validate_strategy_json(raw)
    if not validation.valid:
        return None, validation

    config = parse_strategy_json(raw)
    return config, validation
