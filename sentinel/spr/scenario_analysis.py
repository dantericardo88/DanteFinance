"""Macro scenario analysis studio — portfolio P&L under user-defined macro shocks."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

MACRO_FACTORS: dict[str, str] = {
    "equity_shock_pct": "S&P 500 return shock (%)",
    "rate_shock_bps": "10Y Treasury rate shock (basis points)",
    "credit_spread_bps": "IG credit spread widening (basis points)",
    "usd_shock_pct": "USD index change (%, + = USD strengthening)",
    "oil_shock_pct": "WTI crude price change (%)",
    "vix_shock_pts": "VIX level change (points)",
    "gold_shock_pct": "Gold price change (%)",
}

# Factor proxy ETFs: SPY=equity, UUP=DXY proxy, USO=WTI, GLD=gold
_FACTOR_PROXIES = ["SPY", "UUP", "USO", "GLD"]

SCENARIO_TEMPLATES: dict[str, dict[str, float]] = {
    "2008_crisis": {
        "equity_shock_pct": -37,
        "rate_shock_bps": -200,
        "credit_spread_bps": 400,
        "oil_shock_pct": -50,
        "vix_shock_pts": 40,
    },
    "covid_crash": {
        "equity_shock_pct": -34,
        "rate_shock_bps": -150,
        "credit_spread_bps": 300,
        "oil_shock_pct": -65,
        "vix_shock_pts": 65,
    },
    "rate_hike_200bps": {
        "equity_shock_pct": -15,
        "rate_shock_bps": 200,
        "credit_spread_bps": 50,
        "usd_shock_pct": 5,
    },
    "soft_landing": {
        "equity_shock_pct": 10,
        "rate_shock_bps": -50,
        "credit_spread_bps": -20,
        "oil_shock_pct": -5,
    },
    "stagflation": {
        "equity_shock_pct": -20,
        "rate_shock_bps": 150,
        "credit_spread_bps": 100,
        "oil_shock_pct": 40,
        "gold_shock_pct": 15,
    },
    "china_taiwan": {
        "equity_shock_pct": -25,
        "rate_shock_bps": -100,
        "credit_spread_bps": 200,
        "usd_shock_pct": 10,
        "oil_shock_pct": 30,
    },
    "usd_crash": {
        "equity_shock_pct": 5,
        "rate_shock_bps": 100,
        "usd_shock_pct": -15,
        "gold_shock_pct": 20,
        "oil_shock_pct": 15,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MacroShock(BaseModel):
    equity_shock_pct: float = 0.0
    rate_shock_bps: float = 0.0
    credit_spread_bps: float = 0.0
    usd_shock_pct: float = 0.0
    oil_shock_pct: float = 0.0
    vix_shock_pts: float = 0.0
    gold_shock_pct: float = 0.0
    scenario_name: str = "Custom"


class AssetSensitivity(BaseModel):
    ticker: str
    equity_beta: Optional[float] = None
    duration_years: float = 0.0
    fx_sensitivity: Optional[float] = None
    oil_sensitivity: Optional[float] = None
    gold_sensitivity: Optional[float] = None


class AssetScenarioResult(BaseModel):
    ticker: str
    weight: float
    estimated_return_pct: float
    pnl_usd: float
    sensitivity: AssetSensitivity
    dominant_factor: str


class ScenarioResult(BaseModel):
    scenario: MacroShock
    portfolio_value: float
    tickers: list[str]
    weights: list[float]
    asset_results: list[AssetScenarioResult]
    total_pnl_usd: float
    total_return_pct: float
    best_asset: str
    worst_asset: str
    warnings: list[str] = Field(default_factory=list)
    as_of: str


class MultiScenarioComparison(BaseModel):
    tickers: list[str]
    weights: list[float]
    portfolio_value: float
    scenarios: list[ScenarioResult]
    most_resilient_scenario: str
    worst_scenario: str
    as_of: str


# ---------------------------------------------------------------------------
# Data fetching (sync — run via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_returns_sync(tickers: list[str], lookback_days: int = 90) -> dict[str, "np.ndarray"]:
    """Fetch daily close prices for all tickers and return daily pct returns."""
    import yfinance as yf  # lazy import

    end = date.today()
    start = end - timedelta(days=lookback_days)
    combined = tickers
    logger.info("fetching_yfinance_data", tickers=combined, start=str(start), end=str(end))

    data = yf.download(
        combined,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex when multiple tickers
    if len(combined) == 1:
        closes = data["Close"].to_frame(name=combined[0])
    else:
        closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    returns: dict[str, np.ndarray] = {}
    for ticker in combined:
        if ticker not in closes.columns:
            logger.warning("ticker_missing_from_download", ticker=ticker)
            continue
        pct = closes[ticker].dropna().pct_change().dropna().values.astype(float)
        if len(pct) >= 10:
            returns[ticker] = pct
    return returns


def _fetch_duration_sync(ticker: str) -> float:
    """Try to read duration from yfinance info (bond ETFs only). Returns 0.0 on failure."""
    import yfinance as yf  # lazy import

    try:
        info = yf.Ticker(ticker).info
        dur = info.get("duration", None) or info.get("effectiveDuration", None)
        if dur is not None:
            return float(dur)
    except Exception as exc:
        logger.debug("duration_fetch_failed", ticker=ticker, error=str(exc))
    return 0.0


# ---------------------------------------------------------------------------
# OLS beta estimation
# ---------------------------------------------------------------------------


def _ols_beta(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """Simple OLS slope of y on x (no intercept in beta, but we include a constant)."""
    n = min(len(y), len(x))
    if n < 10:
        return None
    y_ = y[-n:]
    x_ = x[-n:]
    X = np.column_stack([np.ones(n), x_])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
        return float(coeffs[1])
    except np.linalg.LinAlgError:
        return None


def _estimate_sensitivity(
    asset_returns: np.ndarray,
    factor_returns: dict[str, np.ndarray],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (equity_beta, fx_sensitivity, oil_sensitivity, gold_sensitivity)."""
    equity_beta = _ols_beta(asset_returns, factor_returns.get("SPY", np.array([])))
    fx_sens = _ols_beta(asset_returns, factor_returns.get("UUP", np.array([])))
    oil_sens = _ols_beta(asset_returns, factor_returns.get("USO", np.array([])))
    gold_sens = _ols_beta(asset_returns, factor_returns.get("GLD", np.array([])))
    return equity_beta, fx_sens, oil_sens, gold_sens


# ---------------------------------------------------------------------------
# Per-asset beta estimation (async, one task per ticker)
# ---------------------------------------------------------------------------


async def _estimate_asset_sensitivity(
    ticker: str,
    all_returns: dict[str, np.ndarray],
    warnings: list[str],
) -> AssetSensitivity:
    """Compute factor sensitivities for a single ticker from pre-fetched returns."""
    factor_returns = {p: all_returns[p] for p in _FACTOR_PROXIES if p in all_returns}
    asset_ret = all_returns.get(ticker)

    if asset_ret is None or len(asset_ret) < 10:
        msg = f"{ticker}: insufficient return data — using defaults (equity_beta=1.0)"
        warnings.append(msg)
        logger.warning("beta_estimation_fallback", ticker=ticker)
        return AssetSensitivity(ticker=ticker, equity_beta=1.0, duration_years=0.0)

    equity_beta, fx_sens, oil_sens, gold_sens = _estimate_sensitivity(asset_ret, factor_returns)

    if equity_beta is None:
        msg = f"{ticker}: equity beta OLS failed — defaulting to 1.0"
        warnings.append(msg)
        equity_beta = 1.0

    # Fetch duration in thread (bond ETF check)
    duration = await asyncio.to_thread(_fetch_duration_sync, ticker)

    logger.info(
        "sensitivity_estimated",
        ticker=ticker,
        equity_beta=round(equity_beta, 4) if equity_beta else None,
        duration=duration,
        fx_sens=round(fx_sens, 4) if fx_sens else None,
        oil_sens=round(oil_sens, 4) if oil_sens else None,
        gold_sens=round(gold_sens, 4) if gold_sens else None,
    )

    return AssetSensitivity(
        ticker=ticker,
        equity_beta=equity_beta,
        duration_years=duration,
        fx_sensitivity=fx_sens,
        oil_sensitivity=oil_sens,
        gold_sensitivity=gold_sens,
    )


# ---------------------------------------------------------------------------
# P&L calculation
# ---------------------------------------------------------------------------


def _compute_asset_return(sens: AssetSensitivity, shock: MacroShock) -> tuple[float, str]:
    """Compute estimated return for an asset under a macro shock.

    Returns (estimated_return_pct, dominant_factor_name).
    """
    contributions: dict[str, float] = {}

    eq_beta = sens.equity_beta if sens.equity_beta is not None else 1.0
    contributions["equity"] = eq_beta * shock.equity_shock_pct / 100.0

    contributions["rates"] = sens.duration_years * -(shock.rate_shock_bps / 10_000.0)

    fx_sens = sens.fx_sensitivity if sens.fx_sensitivity is not None else 0.0
    contributions["fx"] = fx_sens * -(shock.usd_shock_pct / 100.0)

    oil_sens = sens.oil_sensitivity if sens.oil_sensitivity is not None else 0.0
    contributions["oil"] = oil_sens * shock.oil_shock_pct / 100.0

    gold_sens = sens.gold_sensitivity if sens.gold_sensitivity is not None else 0.0
    contributions["gold"] = gold_sens * shock.gold_shock_pct / 100.0

    total_return = sum(contributions.values())
    dominant = max(contributions, key=lambda k: abs(contributions[k]))

    return total_return, dominant


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------


async def _run_scenario_internal(
    tickers: list[str],
    weights: list[float],
    shock: MacroShock,
    portfolio_value: float,
    all_returns: dict[str, np.ndarray],
) -> ScenarioResult:
    """Internal: run a single scenario given pre-fetched returns."""
    warnings: list[str] = []
    as_of = datetime.utcnow().date().isoformat()

    # Estimate sensitivities in parallel across tickers
    tasks = [
        _estimate_asset_sensitivity(ticker, all_returns, warnings)
        for ticker in tickers
    ]
    sensitivities: list[AssetSensitivity] = await asyncio.gather(*tasks)

    asset_results: list[AssetScenarioResult] = []
    for ticker, weight, sens in zip(tickers, weights, sensitivities):
        estimated_return, dominant = _compute_asset_return(sens, shock)
        pnl_usd = weight * estimated_return * portfolio_value
        asset_results.append(
            AssetScenarioResult(
                ticker=ticker,
                weight=weight,
                estimated_return_pct=round(estimated_return * 100, 4),
                pnl_usd=round(pnl_usd, 2),
                sensitivity=sens,
                dominant_factor=dominant,
            )
        )

    total_pnl = sum(r.pnl_usd for r in asset_results)
    total_return_pct = total_pnl / portfolio_value * 100 if portfolio_value != 0 else 0.0

    best = max(asset_results, key=lambda r: r.estimated_return_pct)
    worst = min(asset_results, key=lambda r: r.estimated_return_pct)

    logger.info(
        "scenario_complete",
        scenario=shock.scenario_name,
        total_pnl=round(total_pnl, 2),
        total_return_pct=round(total_return_pct, 4),
    )

    return ScenarioResult(
        scenario=shock,
        portfolio_value=portfolio_value,
        tickers=tickers,
        weights=weights,
        asset_results=asset_results,
        total_pnl_usd=round(total_pnl, 2),
        total_return_pct=round(total_return_pct, 4),
        best_asset=best.ticker,
        worst_asset=worst.ticker,
        warnings=warnings,
        as_of=as_of,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def run_scenario(
    tickers: list[str],
    weights: list[float] | None = None,
    shock: MacroShock | None = None,
    scenario_name: str | None = None,
    portfolio_value: float = 1_000_000,
) -> ScenarioResult:
    """Run a single macro scenario against a portfolio.

    Provide either ``shock`` directly or ``scenario_name`` to use a template.
    If both are provided, ``shock`` takes precedence.
    Equal-weight across tickers if ``weights`` is None.
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    # Resolve shock
    if shock is None:
        if scenario_name is None:
            raise ValueError("Provide either shock or scenario_name")
        if scenario_name not in SCENARIO_TEMPLATES:
            raise ValueError(
                f"Unknown scenario_name {scenario_name!r}. "
                f"Valid: {list(SCENARIO_TEMPLATES)}"
            )
        shock = MacroShock(**SCENARIO_TEMPLATES[scenario_name], scenario_name=scenario_name)

    # Resolve weights
    if weights is None:
        w = 1.0 / len(tickers)
        weights = [w] * len(tickers)

    if len(weights) != len(tickers):
        raise ValueError("weights length must match tickers length")

    # Normalise weights to sum=1
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("weights must sum to a positive number")
    weights = [w / total_w for w in weights]

    logger.info(
        "run_scenario",
        scenario=shock.scenario_name,
        tickers=tickers,
        portfolio_value=portfolio_value,
    )

    # Fetch all required returns in one threaded call
    all_tickers = list(dict.fromkeys(tickers + _FACTOR_PROXIES))
    all_returns = await asyncio.to_thread(_fetch_returns_sync, all_tickers)

    return await _run_scenario_internal(tickers, weights, shock, portfolio_value, all_returns)


async def run_multi_scenario(
    tickers: list[str],
    weights: list[float] | None = None,
    scenario_names: list[str] | None = None,
    portfolio_value: float = 1_000_000,
) -> MultiScenarioComparison:
    """Run multiple template scenarios and compare results.

    Defaults to all 7 built-in templates. Returns a ranked comparison with
    most-resilient and worst-case scenario names.
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    names = scenario_names if scenario_names is not None else list(SCENARIO_TEMPLATES)
    for name in names:
        if name not in SCENARIO_TEMPLATES:
            raise ValueError(
                f"Unknown scenario_name {name!r}. Valid: {list(SCENARIO_TEMPLATES)}"
            )

    if weights is None:
        w = 1.0 / len(tickers)
        weights = [w] * len(tickers)

    if len(weights) != len(tickers):
        raise ValueError("weights length must match tickers length")

    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("weights must sum to a positive number")
    weights = [w / total_w for w in weights]

    logger.info(
        "run_multi_scenario",
        tickers=tickers,
        scenarios=names,
        portfolio_value=portfolio_value,
    )

    # Single data fetch reused across all scenarios
    all_tickers = list(dict.fromkeys(tickers + _FACTOR_PROXIES))
    all_returns = await asyncio.to_thread(_fetch_returns_sync, all_tickers)

    # Run all scenarios in parallel
    shocks = [
        MacroShock(**SCENARIO_TEMPLATES[name], scenario_name=name)
        for name in names
    ]
    tasks = [
        _run_scenario_internal(tickers, weights, shock, portfolio_value, all_returns)
        for shock in shocks
    ]
    results: list[ScenarioResult] = await asyncio.gather(*tasks)

    as_of = datetime.utcnow().date().isoformat()
    best_result = max(results, key=lambda r: r.total_return_pct)
    worst_result = min(results, key=lambda r: r.total_return_pct)

    return MultiScenarioComparison(
        tickers=tickers,
        weights=weights,
        portfolio_value=portfolio_value,
        scenarios=results,
        most_resilient_scenario=best_result.scenario.scenario_name,
        worst_scenario=worst_result.scenario.scenario_name,
        as_of=as_of,
    )
