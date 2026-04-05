"""
Analysis package – technical indicators, NSE option-chain analytics,
signal aggregation, and market-hours awareness.

Re-exports every public name so that ``from analysis import X`` continues
to work after the split into sub-modules.
"""

from analysis.technical import (  # noqa: F401
    _ema,
    _rsi,
    _macd,
    compute_emas,
    compute_rsi,
    compute_macd,
    compute_pivot_points,
    compute_fibonacci_levels,
    run_technical_analysis,
)

from analysis.options import (  # noqa: F401
    parse_nse_option_chain,
    get_expiry_dates,
    get_underlying_value,
    compute_pcr,
    compute_pcr_volume,
    compute_max_pain,
    compute_highest_oi_strikes,
    compute_oi_buildup,
    compute_iv_summary,
    compute_atm_straddle,
    run_options_analysis,
)

from analysis.signals import (  # noqa: F401
    market_phase,
    generate_checklist,
    compute_overall_bias,
    BULLISH,
    BEARISH,
    NEUTRAL,
)
