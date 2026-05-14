"""Historical and parametric stress testing / scenario analysis for portfolio risk."""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, str]] = {
    "2008_financial_crisis": {
        "start": "2008-09-01",
        "end": "2009-03-31",
        "description": "Lehman collapse",
    },
    "2020_covid_crash": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 market crash",
    },
    "2022_rate_shock": {
        "start": "2022-01-01",
        "end": "2022-10-31",
        "description": "Fed rate hike cycle",
    },
    "2000_dotcom_bust": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "description": "Dot-com bust",
    },
    "2018_q4_selloff": {
        "start": "2018-10-01",
        "end": "2018-12-31",
        "description": "Q4 2018 selloff",
    },
    "1987_black_monday": {
        "start": "1987-10-14",
        "end": "1987-10-20",
        "description": "Black Monday",
    },
}

# Standard parametric shocks: (name, equity_shock_pct, rate_shock_bps, vol_shock_pct)
_STANDARD_SHOCKS: list[tuple[str, float, float, float]] = [
    ("mild_recession", -0.15, 50, 50),
    ("severe_recession", -0.35, 100, 150),
    ("rate_spike_200bps", -0.10, 200, 75),
    ("flash_crash", -0.20, -50, 200),
    ("stagflation", -0.25, 300, 100),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ScenarioResult(BaseModel):
    scenario: str
    description: str
    portfolio_return: float
    max_drawdown: float
    worst_day: float
    best_day: float
    volatility: float
    holding_returns: dict[str, float]


class ParametricShockResult(BaseModel):
    shock_name: str
    equity_shock_pct: float
    rate_shock_bps: float
    vol_shock_pct: float
    portfolio_pnl: float
    portfolio_pnl_pct: float


class StressTestReport(BaseModel):
    portfolio_value: float
    historical_scenarios: list[ScenarioResult]
    parametric_shocks: list[ParametricShockResult]
    worst_scenario: str
    worst_loss_pct: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_drawdown(cum_returns: pd.Series) -> float:
    """Compute max drawdown from a series of cumulative returns (starting at 0)."""
    nav = (1 + cum_returns).cumprod()
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    return float(drawdown.min())


def _slice_returns(
    returns_df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    mask = (returns_df.index >= pd.Timestamp(start)) & (
        returns_df.index <= pd.Timestamp(end)
    )
    return returns_df.loc[mask]


# ---------------------------------------------------------------------------
# Historical scenario
# ---------------------------------------------------------------------------


def run_historical_scenario(
    holdings: dict[str, float],
    portfolio_value: float,
    returns_df: pd.DataFrame,
    scenario_key: str,
) -> ScenarioResult:
    """Apply historical scenario returns to current portfolio weights.

    Uses cumulative returns over the scenario window for each ticker to
    compute portfolio return, max drawdown, daily extremes, and annualised vol.
    Tickers missing from returns_df are skipped (weight redistributed).
    """
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key!r}. Valid: {list(SCENARIOS)}")

    cfg = SCENARIOS[scenario_key]
    description = cfg["description"]
    logger.info(
        "run_historical_scenario",
        scenario=scenario_key,
        start=cfg["start"],
        end=cfg["end"],
    )

    scenario_returns = _slice_returns(returns_df, cfg["start"], cfg["end"])

    available_tickers = [t for t in holdings if t in scenario_returns.columns]
    if not available_tickers:
        logger.warning("no_tickers_in_scenario_window", scenario=scenario_key)
        return ScenarioResult(
            scenario=scenario_key,
            description=description,
            portfolio_return=0.0,
            max_drawdown=0.0,
            worst_day=0.0,
            best_day=0.0,
            volatility=0.0,
            holding_returns={},
        )

    weights = pd.Series({t: holdings[t] for t in available_tickers})
    weights = weights / weights.sum()  # renormalise for missing tickers

    # Cumulative return per ticker over the window
    holding_returns: dict[str, float] = {}
    for ticker in available_tickers:
        cum = float((1 + scenario_returns[ticker].dropna()).prod() - 1)
        holding_returns[ticker] = cum

    # Daily portfolio returns
    daily_port = scenario_returns[available_tickers].dot(weights)
    portfolio_return = float((1 + daily_port).prod() - 1)

    worst_day = float(daily_port.min()) if len(daily_port) else 0.0
    best_day = float(daily_port.max()) if len(daily_port) else 0.0
    volatility = float(daily_port.std() * np.sqrt(252)) if len(daily_port) > 1 else 0.0
    max_dd = _max_drawdown(daily_port) if len(daily_port) > 1 else 0.0

    return ScenarioResult(
        scenario=scenario_key,
        description=description,
        portfolio_return=portfolio_return,
        max_drawdown=max_dd,
        worst_day=worst_day,
        best_day=best_day,
        volatility=volatility,
        holding_returns=holding_returns,
    )


# ---------------------------------------------------------------------------
# Parametric shock
# ---------------------------------------------------------------------------


def run_parametric_shock(
    holdings: dict[str, float],
    portfolio_value: float,
    equity_beta: dict[str, float],
    shock_name: str,
    equity_shock_pct: float = -0.20,
    rate_shock_bps: float = 200,
    vol_shock_pct: float = 100,
) -> ParametricShockResult:
    """Apply parametric shocks: equity down X%, rates up Y bps, vol up Z%.

    For each holding the P&L is approximated as:
        pnl_i = weight_i * portfolio_value * beta_i * equity_shock_pct

    Rate and vol shocks are applied as portfolio-level overlays using standard
    duration/vega heuristics (equity-only portfolios use a flat 0 duration,
    0 vega):
        rate_pnl  = -0.05 * (rate_shock_bps / 100) * portfolio_value   # ~5yr dur proxy
        vol_pnl   = 0  (equity only; extend for options positions)
    """
    logger.info(
        "run_parametric_shock",
        shock=shock_name,
        equity_shock_pct=equity_shock_pct,
        rate_shock_bps=rate_shock_bps,
        vol_shock_pct=vol_shock_pct,
    )

    weights = pd.Series(holdings)
    weights = weights / weights.sum()

    equity_pnl = 0.0
    for ticker, weight in weights.items():
        beta = equity_beta.get(ticker, 1.0)
        equity_pnl += weight * portfolio_value * beta * equity_shock_pct

    # Simplified rate sensitivity: assume ~5-year equivalent duration for a
    # blended equity/bond portfolio proxy; pure equity → near-zero duration
    # but a moderate rate shock still has a second-order effect via discount rates.
    # Using 0.05 as a calibrated factor (basis-point value per unit portfolio).
    rate_pnl = -0.05 * (rate_shock_bps / 100.0) * portfolio_value

    # Vol shock: no direct P&L impact for long-only equity; placeholder 0.
    vol_pnl = 0.0

    total_pnl = equity_pnl + rate_pnl + vol_pnl
    total_pnl_pct = total_pnl / portfolio_value if portfolio_value != 0 else 0.0

    return ParametricShockResult(
        shock_name=shock_name,
        equity_shock_pct=equity_shock_pct,
        rate_shock_bps=rate_shock_bps,
        vol_shock_pct=vol_shock_pct,
        portfolio_pnl=float(total_pnl),
        portfolio_pnl_pct=float(total_pnl_pct),
    )


# ---------------------------------------------------------------------------
# Full stress test runner
# ---------------------------------------------------------------------------


def run_stress_test(
    holdings: dict[str, float],
    portfolio_value: float,
    returns_df: pd.DataFrame,
    equity_betas: dict[str, float] | None = None,
) -> StressTestReport:
    """Run all historical scenarios + standard parametric shocks.

    equity_betas defaults to 1.0 for all holdings if not provided.
    Scenarios missing from returns_df (e.g. 1987 for a data set starting 2000)
    are skipped with a warning rather than raising.
    """
    logger.info(
        "run_stress_test",
        tickers=list(holdings.keys()),
        portfolio_value=portfolio_value,
        num_scenarios=len(SCENARIOS),
    )

    if equity_betas is None:
        equity_betas = {t: 1.0 for t in holdings}

    # Historical scenarios
    historical_results: list[ScenarioResult] = []
    for scenario_key in SCENARIOS:
        try:
            result = run_historical_scenario(
                holdings=holdings,
                portfolio_value=portfolio_value,
                returns_df=returns_df,
                scenario_key=scenario_key,
            )
            historical_results.append(result)
        except Exception as exc:
            logger.warning(
                "scenario_skipped", scenario=scenario_key, error=str(exc)
            )

    # Parametric shocks
    parametric_results: list[ParametricShockResult] = []
    for shock_name, eq_shock, rate_shock, vol_shock in _STANDARD_SHOCKS:
        result = run_parametric_shock(
            holdings=holdings,
            portfolio_value=portfolio_value,
            equity_beta=equity_betas,
            shock_name=shock_name,
            equity_shock_pct=eq_shock,
            rate_shock_bps=rate_shock,
            vol_shock_pct=vol_shock,
        )
        parametric_results.append(result)

    # Worst scenario (historical)
    worst_scenario = "none"
    worst_loss_pct = 0.0
    if historical_results:
        worst = min(historical_results, key=lambda r: r.portfolio_return)
        worst_scenario = worst.scenario
        worst_loss_pct = worst.portfolio_return

    # Also check parametric shocks
    if parametric_results:
        worst_param = min(parametric_results, key=lambda r: r.portfolio_pnl_pct)
        if worst_param.portfolio_pnl_pct < worst_loss_pct:
            worst_scenario = worst_param.shock_name
            worst_loss_pct = worst_param.portfolio_pnl_pct

    return StressTestReport(
        portfolio_value=portfolio_value,
        historical_scenarios=historical_results,
        parametric_shocks=parametric_results,
        worst_scenario=worst_scenario,
        worst_loss_pct=worst_loss_pct,
    )
