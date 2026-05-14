"""
Brinson-Hood-Beebower (BHB) portfolio attribution model.

Decomposes active return (portfolio minus benchmark) into:
  - Allocation effect: did we over/under-weight the right sectors?
  - Selection effect: did we pick better stocks within each sector?
  - Interaction effect: combined impact of allocation and selection decisions.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf  # type: ignore
    _YF = True
except ImportError:
    _YF = False


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class SectorAttribution(BaseModel):
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float      # (wp - wb) * (rb - R_b)
    selection_effect: float       # wb * (rp - rb)
    interaction_effect: float     # (wp - wb) * (rp - rb)
    total_effect: float           # allocation + selection + interaction


class AttributionResult(BaseModel):
    period_start: date
    period_end: date
    portfolio_return: float
    benchmark_return: float
    active_return: float          # portfolio - benchmark
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    total_active_return: float    # should ≈ active_return (BHB identity check)
    sector_attributions: list[SectorAttribution]
    benchmark: str                # "SPY", "QQQ", etc.
    r_squared: Optional[float]    # how well attribution explains active return
    generated_at: datetime


# ---------------------------------------------------------------------------
# Sector map — 50+ major tickers
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "QCOM": "Technology",
    "TXN": "Technology",
    "MU": "Technology",
    "AMAT": "Technology",
    "KLAC": "Technology",
    "LRCX": "Technology",
    "ADI": "Technology",
    "MRVL": "Technology",
    "NOW": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "SAP": "Technology",
    "IBM": "Technology",
    "HPQ": "Technology",
    "DELL": "Technology",
    "PANW": "Technology",
    "CRWD": "Technology",
    "ZS": "Technology",
    "SNOW": "Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "TMUS": "Communication Services",
    "CHTR": "Communication Services",
    "TTWO": "Communication Services",
    "EA": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",
    "CL": "Consumer Staples",
    "GIS": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    # Healthcare
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "PFE": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "DHR": "Healthcare",
    "BMY": "Healthcare",
    "AMGN": "Healthcare",
    "GILD": "Healthcare",
    "ISRG": "Healthcare",
    "SYK": "Healthcare",
    "BSX": "Healthcare",
    "REGN": "Healthcare",
    "VRTX": "Healthcare",
    "HUM": "Healthcare",
    "CI": "Healthcare",
    "CVS": "Healthcare",
    # Financials
    "BRK-B": "Financials",
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SPGI": "Financials",
    "MCO": "Financials",
    "AXP": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "PYPL": "Financials",
    "COF": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    "TFC": "Financials",
    "MET": "Financials",
    "PRU": "Financials",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "RTX": "Industrials",
    "LMT": "Industrials",
    "DE": "Industrials",
    "BA": "Industrials",
    "MMM": "Industrials",
    "GD": "Industrials",
    "NSC": "Industrials",
    "UNP": "Industrials",
    "FDX": "Industrials",
    "ETN": "Industrials",
    "EMR": "Industrials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "PXD": "Energy",
    "OXY": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "MPC": "Energy",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "NEM": "Materials",
    "FCX": "Materials",
    "ALB": "Materials",
    "ECL": "Materials",
    "PPG": "Materials",
    "SHW": "Materials",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "EXC": "Utilities",
    "XEL": "Utilities",
    "PCG": "Utilities",
    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    "O": "Real Estate",
    "WELL": "Real Estate",
    "SPG": "Real Estate",
    "AVB": "Real Estate",
    "EQR": "Real Estate",
}


def _infer_sector(ticker: str) -> str:
    """Return the GICS sector for a ticker from the hardcoded map.

    Falls back to "Other" for unknown tickers.
    """
    return _SECTOR_MAP.get(ticker.upper(), "Other")


# ---------------------------------------------------------------------------
# Benchmark sector weights (approximate GICS, as of early 2025)
# ---------------------------------------------------------------------------

_BENCHMARK_SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "SPY": {
        "Technology": 0.290,
        "Communication Services": 0.090,
        "Consumer Discretionary": 0.105,
        "Consumer Staples": 0.060,
        "Healthcare": 0.130,
        "Financials": 0.130,
        "Industrials": 0.085,
        "Energy": 0.035,
        "Materials": 0.025,
        "Utilities": 0.025,
        "Real Estate": 0.025,
    },
    "QQQ": {
        "Technology": 0.510,
        "Communication Services": 0.165,
        "Consumer Discretionary": 0.185,
        "Consumer Staples": 0.045,
        "Healthcare": 0.060,
        "Financials": 0.010,
        "Industrials": 0.015,
        "Energy": 0.005,
        "Materials": 0.003,
        "Utilities": 0.002,
        "Real Estate": 0.000,
    },
    "IWM": {
        "Technology": 0.175,
        "Communication Services": 0.040,
        "Consumer Discretionary": 0.110,
        "Consumer Staples": 0.035,
        "Healthcare": 0.160,
        "Financials": 0.155,
        "Industrials": 0.170,
        "Energy": 0.055,
        "Materials": 0.040,
        "Utilities": 0.035,
        "Real Estate": 0.025,
    },
    "DIA": {
        "Technology": 0.195,
        "Communication Services": 0.045,
        "Consumer Discretionary": 0.090,
        "Consumer Staples": 0.080,
        "Healthcare": 0.185,
        "Financials": 0.215,
        "Industrials": 0.140,
        "Energy": 0.020,
        "Materials": 0.015,
        "Utilities": 0.010,
        "Real Estate": 0.005,
    },
}


def _benchmark_sector_weights(benchmark: str) -> dict[str, float]:
    """Return approximate GICS sector weights for the given benchmark ETF.

    Falls back to SPY weights for unknown benchmarks.
    """
    bm = benchmark.upper()
    if bm not in _BENCHMARK_SECTOR_WEIGHTS:
        logger.warning("unknown_benchmark_using_spy", benchmark=benchmark)
        bm = "SPY"
    return _BENCHMARK_SECTOR_WEIGHTS[bm].copy()


# ---------------------------------------------------------------------------
# Sector-level aggregation
# ---------------------------------------------------------------------------


def _compute_sector_aggregates(
    holdings: dict[str, float],
    returns: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate individual holdings into sector-level weights and returns.

    Args:
        holdings: {ticker: portfolio_weight}  (should sum to ~1)
        returns: {ticker: total_return_over_period}

    Returns:
        (sector_weights, sector_returns) where sector_returns is the
        value-weighted average return within each sector.
    """
    sector_weights: dict[str, float] = {}
    sector_weighted_returns: dict[str, float] = {}

    for ticker, weight in holdings.items():
        sector = _infer_sector(ticker)
        ret = returns.get(ticker, 0.0)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        # Accumulate weight * return for weighted-average calculation
        sector_weighted_returns[sector] = (
            sector_weighted_returns.get(sector, 0.0) + weight * ret
        )

    # Convert to weighted-average return within sector
    sector_returns: dict[str, float] = {}
    for sector, total_weight in sector_weights.items():
        if total_weight > 0:
            sector_returns[sector] = sector_weighted_returns[sector] / total_weight
        else:
            sector_returns[sector] = 0.0

    return sector_weights, sector_returns


# ---------------------------------------------------------------------------
# Price / return fetching
# ---------------------------------------------------------------------------


def _fetch_ticker_returns_sync(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, float]:
    """Fetch total price return for each ticker over [start, end] via yfinance.

    Returns {ticker: total_return} where total_return = (P_end / P_start) - 1.
    Missing tickers are returned with return = 0.0.
    """
    if not _YF:
        logger.warning("yfinance_not_installed_attribution")
        return {t: 0.0 for t in tickers}

    # Add one day buffer on each end for alignment
    start_str = str(start - _dt.timedelta(days=5))
    end_str = str(end + _dt.timedelta(days=2))

    logger.info("fetching_ticker_returns", tickers=tickers, start=str(start), end=str(end))
    try:
        raw = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("attribution_yf_download_failed", error=str(exc))
        return {t: 0.0 for t in tickers}

    if raw.empty:
        return {t: 0.0 for t in tickers}

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    if len(tickers) == 1:
        close = close.rename(columns={"Close": tickers[0]}) if "Close" in close.columns else close

    result: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in close.columns:
            logger.warning("ticker_missing_from_attribution", ticker=ticker)
            result[ticker] = 0.0
            continue

        series = close[ticker].dropna()
        if len(series) < 2:
            result[ticker] = 0.0
            continue

        p_start = float(series.iloc[0])
        p_end = float(series.iloc[-1])
        if p_start <= 0:
            result[ticker] = 0.0
        else:
            result[ticker] = round((p_end / p_start) - 1.0, 8)

    return result


async def _fetch_sector_returns(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, float]:
    """Async wrapper to fetch total returns for each ticker over the period."""
    return await asyncio.to_thread(_fetch_ticker_returns_sync, tickers, start, end)


# ---------------------------------------------------------------------------
# BHB attribution formula
# ---------------------------------------------------------------------------


def _bhb_attribution(
    portfolio_sector_weights: dict[str, float],
    portfolio_sector_returns: dict[str, float],
    benchmark_sector_weights: dict[str, float],
    benchmark_sector_returns: dict[str, float],
    benchmark_total_return: float,
) -> list[SectorAttribution]:
    """Apply Brinson-Hood-Beebower formula across all sectors.

    BHB effects per sector i:
      Allocation  = (wp_i - wb_i) * (rb_i - R_b)
      Selection   = wb_i * (rp_i - rb_i)
      Interaction = (wp_i - wb_i) * (rp_i - rb_i)

    where R_b is the total benchmark return.
    """
    all_sectors = sorted(
        set(portfolio_sector_weights) | set(benchmark_sector_weights)
    )
    R_b = benchmark_total_return
    attributions: list[SectorAttribution] = []

    for sector in all_sectors:
        wp = portfolio_sector_weights.get(sector, 0.0)
        wb = benchmark_sector_weights.get(sector, 0.0)
        rp = portfolio_sector_returns.get(sector, 0.0)
        rb = benchmark_sector_returns.get(sector, 0.0)

        allocation = (wp - wb) * (rb - R_b)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        total = allocation + selection + interaction

        attributions.append(
            SectorAttribution(
                sector=sector,
                portfolio_weight=round(wp, 6),
                benchmark_weight=round(wb, 6),
                portfolio_return=round(rp, 6),
                benchmark_return=round(rb, 6),
                allocation_effect=round(allocation, 8),
                selection_effect=round(selection, 8),
                interaction_effect=round(interaction, 8),
                total_effect=round(total, 8),
            )
        )

    return attributions


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def compute_attribution(
    holdings: dict[str, float],
    benchmark: str = "SPY",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> AttributionResult:
    """Compute BHB attribution for a portfolio against a benchmark ETF.

    Args:
        holdings: {ticker: weight} — portfolio positions (should sum to ~1).
                  Weights are normalised internally if they don't sum to 1.
        benchmark: ETF ticker to use as benchmark ("SPY", "QQQ", "IWM", "DIA").
        start_date: start of attribution period (default: 3 months ago).
        end_date: end of attribution period (default: today).

    Returns:
        AttributionResult with BHB attribution decomposition by sector.
    """
    today = _dt.date.today()
    if end_date is None:
        end_date = today
    if start_date is None:
        start_date = end_date - _dt.timedelta(days=91)  # ~3 months

    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be before end_date {end_date}")

    # Normalise holding weights to sum to 1
    total_weight = sum(holdings.values())
    if total_weight <= 0:
        raise ValueError("Holdings weights must sum to a positive number")
    norm_holdings = {t: w / total_weight for t, w in holdings.items()}

    portfolio_tickers = list(norm_holdings.keys())
    all_tickers = portfolio_tickers + [benchmark]

    logger.info(
        "compute_attribution",
        tickers=portfolio_tickers,
        benchmark=benchmark,
        start=str(start_date),
        end=str(end_date),
    )

    # Fetch returns for all tickers (portfolio + benchmark) in one call
    all_returns = await _fetch_sector_returns(all_tickers, start_date, end_date)

    benchmark_total_return = all_returns.get(benchmark, 0.0)
    holding_returns = {t: all_returns.get(t, 0.0) for t in portfolio_tickers}

    # Portfolio total return (weighted sum)
    portfolio_total_return = sum(
        norm_holdings[t] * holding_returns[t] for t in portfolio_tickers
    )

    active_return = portfolio_total_return - benchmark_total_return

    # Sector-level aggregation for portfolio
    port_sector_weights, port_sector_returns = _compute_sector_aggregates(
        norm_holdings, holding_returns
    )

    # Benchmark sector weights and implied benchmark sector returns
    bm_sector_weights = _benchmark_sector_weights(benchmark)

    # Benchmark sector returns: we approximate by using the benchmark total return
    # for all sectors (since we don't have sector ETF decomposition here).
    # For better accuracy with known SPDR sector ETFs, the caller could provide
    # sector returns explicitly. Here we use the benchmark total return as a
    # uniform approximation for benchmark sector returns.
    bm_sector_returns: dict[str, str | float] = {
        sector: benchmark_total_return for sector in bm_sector_weights
    }
    # Override with portfolio sector returns where available, scaled down to
    # reflect that the benchmark's sector returns differ — use zero alpha assumption:
    # rb_i ≈ R_b (benchmark total) for all i (conservative, avoids fabricating data)
    bm_sector_returns_float: dict[str, float] = {
        sector: float(benchmark_total_return) for sector in bm_sector_weights
    }

    # BHB computation
    sector_attributions = _bhb_attribution(
        portfolio_sector_weights=port_sector_weights,
        portfolio_sector_returns=port_sector_returns,
        benchmark_sector_weights=bm_sector_weights,
        benchmark_sector_returns=bm_sector_returns_float,
        benchmark_total_return=benchmark_total_return,
    )

    total_allocation = sum(s.allocation_effect for s in sector_attributions)
    total_selection = sum(s.selection_effect for s in sector_attributions)
    total_interaction = sum(s.interaction_effect for s in sector_attributions)
    total_active = total_allocation + total_selection + total_interaction

    # R² of attribution model: how much of active return is explained
    # by the BHB decomposition (should be close to 1 if sectors are complete)
    r_squared: Optional[float] = None
    if abs(active_return) > 1e-10:
        unexplained = active_return - total_active
        ss_total = active_return**2
        ss_resid = unexplained**2
        r_squared = round(float(1.0 - ss_resid / ss_total), 6)

    result = AttributionResult(
        period_start=start_date,
        period_end=end_date,
        portfolio_return=round(portfolio_total_return, 8),
        benchmark_return=round(benchmark_total_return, 8),
        active_return=round(active_return, 8),
        total_allocation_effect=round(total_allocation, 8),
        total_selection_effect=round(total_selection, 8),
        total_interaction_effect=round(total_interaction, 8),
        total_active_return=round(total_active, 8),
        sector_attributions=sector_attributions,
        benchmark=benchmark.upper(),
        r_squared=r_squared,
        generated_at=datetime.utcnow(),
    )

    logger.info(
        "attribution_computed",
        portfolio_return=portfolio_total_return,
        benchmark_return=benchmark_total_return,
        active_return=active_return,
        total_allocation=total_allocation,
        total_selection=total_selection,
        total_interaction=total_interaction,
        n_sectors=len(sector_attributions),
    )

    return result
