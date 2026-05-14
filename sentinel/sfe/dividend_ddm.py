"""Dividend Discount Model (Gordon, H-Model, 3-stage) + sustainability scoring — Dimension 28 enhancement."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RISK_FREE_RATE: float = 0.045        # 10Y US Treasury as of 2026-05
_EQUITY_RISK_PREMIUM: float = 0.055   # Damodaran ERP
_TERMINAL_GROWTH_RATE: float = 0.025  # Long-run nominal GDP proxy
_SECTOR_FALLBACK_GROWTH: float = 0.04 # Fallback 5Y growth when history is thin
_MAX_HISTORY_YEARS: int = 10
_H_MODEL_HALF_LIFE: float = 5.0       # Transition half-life for H-Model

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DDMInputs(BaseModel):
    """Consolidated inputs for all three DDM variants."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    current_dividend_per_share: float
    dividend_growth_rate_5y: Optional[float] = None
    terminal_growth_rate: float = _TERMINAL_GROWTH_RATE
    cost_of_equity: float = 0.09
    payout_ratio: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None
    earnings_per_share: Optional[float] = None
    dividend_history: list[tuple[str, float]] = Field(default_factory=list)


class GordonGrowthResult(BaseModel):
    """Gordon Growth Model (constant-growth perpetuity) result."""
    model_config = ConfigDict(frozen=True)

    intrinsic_value: float
    implied_upside_pct: float
    required_return: float
    growth_rate: float
    is_valid: bool   # False when g >= ke — model undefined


class HModelResult(BaseModel):
    """Fuller & Hsia (1984) H-Model — linear growth fade."""
    model_config = ConfigDict(frozen=True)

    intrinsic_value: float
    implied_upside_pct: float
    initial_growth: float
    terminal_growth: float
    half_life_years: float
    is_valid: bool   # False when g_long >= ke


class ThreeStageDDMResult(BaseModel):
    """Three-stage DDM: high-growth → fade → perpetuity."""
    model_config = ConfigDict(frozen=True)

    intrinsic_value: float
    implied_upside_pct: float
    stage1_pv: float
    stage2_pv: float
    terminal_pv: float
    stage1_years: int
    stage2_years: int


class DividendSustainability(BaseModel):
    """Dividend safety / sustainability scorecard."""
    model_config = ConfigDict(frozen=True)

    payout_ratio: Optional[float]
    fcf_coverage: Optional[float]        # FCF per share / DPS
    debt_dividend_ratio: Optional[float]
    consecutive_growth_years: int
    is_aristocrat: bool                  # 25+ consecutive growth years
    is_king: bool                        # 50+ consecutive growth years
    sustainability_score: float          # 0–10
    at_risk_flags: list[str]
    verdict: str


class DividendDDMResult(BaseModel):
    """Aggregated DDM result for a single ticker."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    spot_price: float
    gordon_growth: GordonGrowthResult
    h_model: HModelResult
    three_stage: ThreeStageDDMResult
    sustainability: DividendSustainability
    consensus_value: Optional[float]     # mean of valid DDM intrinsic values
    consensus_upside_pct: Optional[float]
    quality_tier: str
    as_of: str
    warnings: list[str]


class DividendScreenRow(BaseModel):
    """Single row in a dividend quality screen."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    dps: float
    yield_pct: float
    sustainability_score: float
    quality_tier: str
    consensus_value: Optional[float]
    upside_pct: Optional[float]
    consecutive_years: int


class DividendScreen(BaseModel):
    """Batch dividend quality screen result."""
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    results: list[DividendScreenRow]
    high_yield_count: int                # tickers with yield >= 3%
    aristocrats: list[str]
    kings: list[str]
    at_risk: list[str]
    avg_sustainability_score: float
    as_of: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------


async def _fetch_dividend_data(ticker: str) -> dict:
    """
    Fetch all yfinance data needed for DDM via asyncio.to_thread.
    Returns a comprehensive dict; missing fields set to None rather than raising.
    """
    import yfinance as yf  # noqa: PLC0415 — lazy import keeps module importable without yf

    def _sync_fetch() -> dict:
        t = yf.Ticker(ticker)
        info: dict = {}
        try:
            info = t.info or {}
        except Exception as exc:
            logger.warning("yf.Ticker.info failed", ticker=ticker, error=str(exc))

        spot: Optional[float] = None
        try:
            spot = t.fast_info.last_price
        except Exception:
            spot = info.get("currentPrice") or info.get("regularMarketPrice")

        div_rate: Optional[float] = (
            info.get("dividendRate") or info.get("trailingAnnualDividendRate")
        )
        payout_ratio: Optional[float] = info.get("payoutRatio")
        fcf: Optional[float] = info.get("freeCashflow")
        eps: Optional[float] = info.get("earningsPerShare") or info.get("trailingEps")
        beta: Optional[float] = info.get("beta")
        shares_out: Optional[float] = info.get("sharesOutstanding")

        dividends_series = None
        try:
            dividends_series = t.dividends  # pd.Series with DatetimeIndex
        except Exception as exc:
            logger.warning("Dividend history fetch failed", ticker=ticker, error=str(exc))

        fcf_per_share: Optional[float] = None
        if fcf is not None and shares_out and shares_out > 0:
            fcf_per_share = fcf / shares_out

        return {
            "ticker": ticker,
            "spot": spot,
            "div_rate": div_rate,
            "payout_ratio": payout_ratio,
            "fcf": fcf,
            "fcf_per_share": fcf_per_share,
            "eps": eps,
            "beta": beta,
            "shares_out": shares_out,
            "dividends_series": dividends_series,
            "info": info,
        }

    result = await asyncio.to_thread(_sync_fetch)
    logger.info("Dividend data fetched", ticker=ticker, spot=result.get("spot"))
    return result


# ---------------------------------------------------------------------------
# Annual DPS Series Helpers
# ---------------------------------------------------------------------------


def _compute_annual_dps_series(dividends_series) -> list[tuple[str, float]]:
    """
    Aggregate a quarterly dividend pd.Series (DatetimeIndex) into annual totals.
    Returns sorted list of (year_str, annual_dps) for up to _MAX_HISTORY_YEARS.
    Only years with at least one positive payment are included.
    """
    if dividends_series is None or len(dividends_series) == 0:
        return []
    try:
        import pandas as pd  # noqa: PLC0415 — lazy, called inside to_thread path

        if not isinstance(dividends_series.index, pd.DatetimeIndex):
            dividends_series.index = pd.to_datetime(dividends_series.index)
        annual: pd.Series = (
            dividends_series.groupby(dividends_series.index.year).sum()
        )
        annual = annual[annual > 0].tail(_MAX_HISTORY_YEARS)
        return [(str(year), float(dps)) for year, dps in annual.items()]
    except Exception as exc:
        logger.warning("Annual DPS computation failed", error=str(exc))
        return []


def _consecutive_growth_years(annual_dps: list[tuple[str, float]]) -> int:
    """
    Count consecutive years of DPS growth working backwards from the most recent.
    Returns 0 if fewer than 2 data points or if the streak is immediately broken.
    """
    if len(annual_dps) < 2:
        return 0
    values = [v for _, v in sorted(annual_dps, key=lambda x: x[0])]
    count = 0
    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break
    return count


def _dividend_growth_5y(annual_dps: list[tuple[str, float]]) -> Optional[float]:
    """
    5-year CAGR of DPS: (end / start) ^ (1 / years) - 1.
    Returns None if fewer than 3 data points or if starting DPS is zero.
    """
    if len(annual_dps) < 3:
        return None
    window = sorted(annual_dps, key=lambda x: x[0])
    window = window[-6:] if len(window) >= 6 else window
    start_dps, end_dps = window[0][1], window[-1][1]
    years = len(window) - 1
    if start_dps <= 0 or years < 1:
        return None
    return float((end_dps / start_dps) ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Cost of Equity (CAPM)
# ---------------------------------------------------------------------------


def _cost_of_equity(
    beta: float,
    risk_free: float = _RISK_FREE_RATE,
    erp: float = _EQUITY_RISK_PREMIUM,
) -> float:
    """
    CAPM: Ke = Rf + beta × ERP. Clamped to [6%, 25%] to avoid degenerate DDM inputs.
    Defaults: Rf=4.5% (10Y Treasury), ERP=5.5% (Damodaran 2025).
    """
    return float(np.clip(risk_free + beta * erp, 0.06, 0.25))


# ---------------------------------------------------------------------------
# DDM Model Functions
# ---------------------------------------------------------------------------


def gordon_growth(
    d1: float,
    g: float,
    ke: float,
    spot: float = 0.0,
) -> GordonGrowthResult:
    """
    Constant-Growth DDM (Gordon Growth Model).
    P = D1 / (ke - g). Invalid when g >= ke or d1 <= 0.
    Caller is responsible for computing D1 = D0 × (1 + g).
    """
    if g >= ke or d1 <= 0:
        return GordonGrowthResult(
            intrinsic_value=float("nan"),
            implied_upside_pct=float("nan"),
            required_return=ke,
            growth_rate=g,
            is_valid=False,
        )
    price = d1 / (ke - g)
    upside = (price - spot) / spot if spot > 0 else float("nan")
    logger.debug("Gordon Growth computed", d1=round(d1, 4), g=f"{g:.2%}", ke=f"{ke:.2%}", price=round(price, 2))
    return GordonGrowthResult(
        intrinsic_value=round(price, 2),
        implied_upside_pct=round(upside * 100, 2) if not np.isnan(upside) else float("nan"),
        required_return=ke,
        growth_rate=g,
        is_valid=True,
    )


def h_model(
    d0: float,
    g_short: float,
    g_long: float,
    ke: float,
    half_life: float = _H_MODEL_HALF_LIFE,
    spot: float = 0.0,
) -> HModelResult:
    """
    Fuller & Hsia (1984) H-Model — linear growth fade.
    P = D0 × [(1 + g_long) + H × (g_short - g_long)] / (ke - g_long)
    where H = half_life / 2. Invalid when g_long >= ke.
    """
    if g_long >= ke or d0 <= 0:
        return HModelResult(
            intrinsic_value=float("nan"),
            implied_upside_pct=float("nan"),
            initial_growth=g_short,
            terminal_growth=g_long,
            half_life_years=half_life,
            is_valid=False,
        )
    h = half_life / 2.0
    price = d0 * ((1.0 + g_long) + h * (g_short - g_long)) / (ke - g_long)
    upside = (price - spot) / spot if spot > 0 else float("nan")
    logger.debug("H-Model computed", d0=round(d0, 4), g_short=f"{g_short:.2%}", g_long=f"{g_long:.2%}", price=round(price, 2))
    return HModelResult(
        intrinsic_value=round(price, 2),
        implied_upside_pct=round(upside * 100, 2) if not np.isnan(upside) else float("nan"),
        initial_growth=g_short,
        terminal_growth=g_long,
        half_life_years=half_life,
        is_valid=True,
    )


def three_stage_ddm(
    d0: float,
    g1: float,
    g2: float,
    g3: float,
    ke: float,
    n1: int = 5,
    n2: int = 5,
    spot: float = 0.0,
) -> ThreeStageDDMResult:
    """
    Three-stage Dividend Discount Model.
    Stage 1 (years 1..n1):       constant high growth g1.
    Stage 2 (years n1+1..n1+n2): growth fades linearly from g1 → g3.
    Stage 3 (terminal):          Gordon Growth perpetuity at g3.
    Vectorized numpy discount factors used for Stage 1.
    g2 is accepted for API symmetry but the fade path is g1 → g3.
    """
    if d0 <= 0:
        return ThreeStageDDMResult(
            intrinsic_value=0.0, implied_upside_pct=float("nan"),
            stage1_pv=0.0, stage2_pv=0.0, terminal_pv=0.0,
            stage1_years=n1, stage2_years=n2,
        )

    # Stage 1 — vectorized
    t1 = np.arange(1, n1 + 1, dtype=float)
    stage1_pv = float(np.sum(d0 * (1.0 + g1) ** t1 / (1.0 + ke) ** t1))
    d_end1 = d0 * (1.0 + g1) ** n1

    # Stage 2 — linear fade g1 → g3
    stage2_pvs: list[float] = []
    d_prev = d_end1
    d_end2 = d_end1
    for k in range(1, n2 + 1):
        g_k = g1 + (k / n2) * (g3 - g1)
        d_k = d_prev * (1.0 + g_k)
        stage2_pvs.append(d_k / (1.0 + ke) ** (n1 + k))
        d_prev = d_k
        d_end2 = d_k
    stage2_pv = float(sum(stage2_pvs))

    # Stage 3 — terminal (Gordon Growth)
    t_term = n1 + n2
    if g3 < ke:
        tv = d_end2 * (1.0 + g3) / (ke - g3)
        terminal_pv = float(tv / (1.0 + ke) ** t_term)
    else:
        logger.warning("3-stage DDM: g3 >= ke; applying 20x terminal multiple", g3=f"{g3:.2%}", ke=f"{ke:.2%}")
        terminal_pv = float(d_end2 * 20.0 / (1.0 + ke) ** t_term)

    intrinsic = stage1_pv + stage2_pv + terminal_pv
    upside = (intrinsic - spot) / spot if spot > 0 else float("nan")
    logger.debug(
        "3-stage DDM computed", d0=round(d0, 4), g1=f"{g1:.2%}", g3=f"{g3:.2%}", ke=f"{ke:.2%}",
        s1=round(stage1_pv, 2), s2=round(stage2_pv, 2), tv=round(terminal_pv, 2), iv=round(intrinsic, 2),
    )
    return ThreeStageDDMResult(
        intrinsic_value=round(intrinsic, 2),
        implied_upside_pct=round(upside * 100, 2) if not np.isnan(upside) else float("nan"),
        stage1_pv=round(stage1_pv, 2),
        stage2_pv=round(stage2_pv, 2),
        terminal_pv=round(terminal_pv, 2),
        stage1_years=n1,
        stage2_years=n2,
    )


# ---------------------------------------------------------------------------
# Sustainability Helpers
# ---------------------------------------------------------------------------


def _payout_sustainability(
    payout_ratio: Optional[float],
    fcf_per_share: Optional[float],
    dps: float,
) -> tuple[float, list[str]]:
    """
    Compute a payout-based sustainability contribution [0, 10] and at-risk flags.
    FCF coverage = fcf_per_share / dps when both are available.
    """
    flags: list[str] = []
    if dps <= 0:
        return 5.0, []

    payout_score: Optional[float] = None
    if payout_ratio is not None:
        if payout_ratio > 0.80:
            flags.append(f"Payout ratio {payout_ratio:.0%} exceeds 80% — elevated risk")
        payout_score = 10.0 * (1.0 - min(1.0, max(0.0, payout_ratio)))

    fcf_score: Optional[float] = None
    if fcf_per_share is not None:
        cov = fcf_per_share / dps
        if cov < 1.0:
            flags.append(f"FCF coverage {cov:.2f}x < 1.0x — dividend exceeds free cash flow")
        fcf_score = 10.0 * min(1.0, max(0.0, (cov - 0.5) / 1.5))
    elif payout_ratio is not None and payout_ratio > 0.60:
        flags.append("No FCF data; payout ratio > 60% warrants caution")

    if payout_score is not None and fcf_score is not None:
        blended = (payout_score + fcf_score) / 2.0
    elif payout_score is not None:
        blended = payout_score
    elif fcf_score is not None:
        blended = fcf_score
    else:
        blended = 5.0
    return float(blended), flags


def _sustainability_score(
    payout_ratio: Optional[float],
    fcf_coverage: Optional[float],
    consecutive_years: int,
    debt_div_ratio: Optional[float],
) -> float:
    """
    Composite dividend sustainability score [0–10].
    Weights: payout=0.35, fcf=0.35, growth=0.20, debt=0.10.
    Neutral value (5.0) substituted when a component lacks data.
    """
    payout_c = 10.0 * max(0.0, 1.0 - payout_ratio / 0.90) if payout_ratio is not None else 5.0
    fcf_c = 10.0 * min(1.0, max(0.0, fcf_coverage / 2.0)) if fcf_coverage is not None else 5.0
    growth_c = min(10.0, float(consecutive_years) / 3.0)
    debt_c = 10.0 * max(0.0, 1.0 - debt_div_ratio / 10.0) if debt_div_ratio is not None else 5.0
    raw = 0.35 * payout_c + 0.35 * fcf_c + 0.20 * growth_c + 0.10 * debt_c
    return float(np.clip(raw, 0.0, 10.0))


def _quality_tier(consecutive_years: int, sustainability_score: float, dps: float) -> str:
    """Map consecutive growth years and sustainability score to a quality label."""
    if dps <= 0:
        return "No Dividend"
    if consecutive_years >= 50:
        return "Dividend King"
    if consecutive_years >= 25:
        return "Dividend Aristocrat"
    if sustainability_score >= 7.0:
        return "High Quality"
    if sustainability_score >= 5.0:
        return "Moderate"
    return "At Risk"


def _build_sustainability(
    payout_ratio: Optional[float],
    fcf_per_share: Optional[float],
    dps: float,
    consecutive_years: int,
    debt_div_ratio: Optional[float],
) -> DividendSustainability:
    """Assemble the full DividendSustainability scorecard."""
    _, flags = _payout_sustainability(payout_ratio, fcf_per_share, dps)

    fcf_coverage: Optional[float] = (
        fcf_per_share / dps if (fcf_per_share is not None and dps > 0) else None
    )
    score = _sustainability_score(payout_ratio, fcf_coverage, consecutive_years, debt_div_ratio)
    is_aristocrat = consecutive_years >= 25
    is_king = consecutive_years >= 50

    if dps <= 0:
        verdict = "No dividend paid; DDM not applicable."
    elif is_king:
        verdict = f"Dividend King — {consecutive_years} consecutive growth years; score {score:.1f}/10."
    elif is_aristocrat:
        verdict = f"Dividend Aristocrat — {consecutive_years} consecutive growth years; score {score:.1f}/10."
    elif score >= 7.0:
        verdict = f"Highly sustainable — strong FCF coverage and modest payout; score {score:.1f}/10."
    elif score >= 5.0:
        verdict = f"Moderate sustainability — some risk present; score {score:.1f}/10."
    elif flags:
        verdict = f"At risk — {flags[0].lower()}"
    else:
        verdict = f"Low sustainability — score {score:.1f}/10; review fundamentals."

    return DividendSustainability(
        payout_ratio=payout_ratio,
        fcf_coverage=round(fcf_coverage, 3) if fcf_coverage is not None else None,
        debt_dividend_ratio=round(debt_div_ratio, 3) if debt_div_ratio is not None else None,
        consecutive_growth_years=consecutive_years,
        is_aristocrat=is_aristocrat,
        is_king=is_king,
        sustainability_score=round(score, 2),
        at_risk_flags=flags,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Main Entry Point — Single Ticker
# ---------------------------------------------------------------------------


async def get_dividend_ddm(ticker: str) -> DividendDDMResult:
    """
    Full DDM pipeline for a single ticker.

    Steps:
      1. Fetch live market data via yfinance (asyncio.to_thread).
      2. Derive DDM inputs: d0, 5Y growth CAGR, cost of equity (CAPM).
      3. Run Gordon Growth, H-Model, and 3-stage DDM.
      4. Compute consensus value (mean of valid model outputs).
      5. Score dividend sustainability and assign quality tier.
    """
    warnings: list[str] = []
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        raw = await _fetch_dividend_data(ticker)
    except Exception as exc:
        logger.error("Data fetch failed", ticker=ticker, error=str(exc))
        warnings.append(f"Data fetch error: {exc}")
        raw = {
            "ticker": ticker, "spot": None, "div_rate": None, "payout_ratio": None,
            "fcf": None, "fcf_per_share": None, "eps": None, "beta": None,
            "shares_out": None, "dividends_series": None, "info": {},
        }

    spot: float = raw.get("spot") or 0.0
    div_rate: Optional[float] = raw.get("div_rate")
    payout_ratio: Optional[float] = raw.get("payout_ratio")
    fcf_per_share: Optional[float] = raw.get("fcf_per_share")
    beta: float = raw.get("beta") or 1.0
    dividends_series = raw.get("dividends_series")

    if spot == 0.0:
        warnings.append("Spot price unavailable; upside calculations will be NaN.")

    annual_dps = _compute_annual_dps_series(dividends_series)

    if div_rate and div_rate > 0:
        d0 = float(div_rate)
    elif annual_dps:
        d0 = annual_dps[-1][1]
    else:
        d0 = 0.0
        warnings.append("No dividend data found; DDM models will return zero/invalid.")

    g5 = _dividend_growth_5y(annual_dps)
    if g5 is None:
        warnings.append(
            f"Insufficient history for 5Y CAGR; using sector fallback {_SECTOR_FALLBACK_GROWTH:.1%}."
        )
        g5 = _SECTOR_FALLBACK_GROWTH

    g_near = float(np.clip(g5, -0.05, 0.25))
    g_terminal = _TERMINAL_GROWTH_RATE
    ke = _cost_of_equity(beta)
    consecutive_years = _consecutive_growth_years(annual_dps)

    # Debt / annual-dividend ratio (total debt / total annual dividends)
    debt_div_ratio: Optional[float] = None
    total_debt = raw.get("info", {}).get("totalDebt")
    shares_out = raw.get("shares_out")
    if total_debt and shares_out and d0 > 0 and shares_out > 0:
        annual_div_total = d0 * shares_out
        if annual_div_total > 0:
            debt_div_ratio = float(total_debt / annual_div_total)

    logger.info(
        "DDM inputs resolved", ticker=ticker, d0=round(d0, 4),
        g_near=f"{g_near:.2%}", g_terminal=f"{g_terminal:.2%}",
        ke=f"{ke:.2%}", consecutive_years=consecutive_years,
    )

    d1 = d0 * (1.0 + g_near)
    gg_result = gordon_growth(d1=d1, g=g_near, ke=ke, spot=spot)
    hm_result = h_model(d0=d0, g_short=g_near, g_long=g_terminal, ke=ke,
                        half_life=_H_MODEL_HALF_LIFE, spot=spot)
    ts_result = three_stage_ddm(d0=d0, g1=g_near, g2=g_near, g3=g_terminal,
                                ke=ke, n1=5, n2=5, spot=spot)

    valid_values: list[float] = []
    if gg_result.is_valid and not np.isnan(gg_result.intrinsic_value):
        valid_values.append(gg_result.intrinsic_value)
    if hm_result.is_valid and not np.isnan(hm_result.intrinsic_value):
        valid_values.append(hm_result.intrinsic_value)
    if ts_result.intrinsic_value > 0 and not np.isnan(ts_result.intrinsic_value):
        valid_values.append(ts_result.intrinsic_value)

    consensus_value: Optional[float] = None
    consensus_upside: Optional[float] = None
    if valid_values:
        consensus_value = round(float(np.mean(valid_values)), 2)
        if spot > 0:
            consensus_upside = round((consensus_value - spot) / spot * 100.0, 2)

    sustainability = _build_sustainability(
        payout_ratio=payout_ratio, fcf_per_share=fcf_per_share, dps=d0,
        consecutive_years=consecutive_years, debt_div_ratio=debt_div_ratio,
    )
    quality = _quality_tier(consecutive_years, sustainability.sustainability_score, d0)

    logger.info(
        "DDM complete", ticker=ticker, consensus=consensus_value,
        upside_pct=consensus_upside, quality_tier=quality,
        sustainability_score=sustainability.sustainability_score,
    )
    return DividendDDMResult(
        ticker=ticker.upper(),
        spot_price=round(spot, 2),
        gordon_growth=gg_result,
        h_model=hm_result,
        three_stage=ts_result,
        sustainability=sustainability,
        consensus_value=consensus_value,
        consensus_upside_pct=consensus_upside,
        quality_tier=quality,
        as_of=as_of,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Batch Screener Entry Point
# ---------------------------------------------------------------------------


async def screen_dividend_quality(
    tickers: list[str],
    min_yield_pct: float = 1.0,
) -> DividendScreen:
    """
    Screen a list of tickers by dividend quality.

    Steps:
      1. Run get_dividend_ddm for all tickers concurrently via asyncio.gather.
      2. Recover DPS from stored model parameters; compute yield.
      3. Filter to dividend yield >= min_yield_pct.
      4. Classify aristocrats, kings, at-risk tickers.
      5. Sort by sustainability_score descending, yield descending.
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    screen_warnings: list[str] = []

    logger.info("Starting dividend quality screen", tickers=tickers, min_yield_pct=min_yield_pct)

    raw_results: list[DividendDDMResult | BaseException] = await asyncio.gather(
        *[get_dividend_ddm(t) for t in tickers], return_exceptions=True
    )

    rows: list[DividendScreenRow] = []
    aristocrats: list[str] = []
    kings: list[str] = []
    at_risk: list[str] = []
    high_yield_count = 0

    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            screen_warnings.append(f"{ticker}: DDM failed — {res}")
            logger.warning("DDM failed for ticker", ticker=ticker, error=str(res))
            continue

        ddm: DividendDDMResult = res  # type: ignore[assignment]

        # Recover D0 from stored Gordon Growth parameters (invert P = D1 / (ke - g))
        if ddm.gordon_growth.is_valid:
            ke_r = ddm.gordon_growth.required_return
            g_r = ddm.gordon_growth.growth_rate
            d1_r = ddm.gordon_growth.intrinsic_value * (ke_r - g_r)
            dps_r = d1_r / (1.0 + g_r) if (1.0 + g_r) > 0 else 0.0
        elif ddm.h_model.is_valid:
            g_l = ddm.h_model.terminal_growth
            g_s = ddm.h_model.initial_growth
            h = ddm.h_model.half_life_years / 2.0
            ke_r = ddm.gordon_growth.required_return
            denom = ke_r - g_l
            nf = (1.0 + g_l) + h * (g_s - g_l)
            dps_r = ddm.h_model.intrinsic_value * denom / nf if (denom > 0 and nf > 0) else 0.0
        else:
            dps_r = 0.0

        yield_pct = dps_r / ddm.spot_price * 100.0 if ddm.spot_price > 0 else 0.0

        if yield_pct < min_yield_pct and dps_r <= 0:
            continue

        if yield_pct >= 3.0:
            high_yield_count += 1
        if ddm.sustainability.is_king:
            kings.append(ddm.ticker)
        if ddm.sustainability.is_aristocrat:
            aristocrats.append(ddm.ticker)
        if ddm.quality_tier == "At Risk":
            at_risk.append(ddm.ticker)

        rows.append(DividendScreenRow(
            ticker=ddm.ticker,
            dps=round(dps_r, 4),
            yield_pct=round(yield_pct, 2),
            sustainability_score=ddm.sustainability.sustainability_score,
            quality_tier=ddm.quality_tier,
            consensus_value=ddm.consensus_value,
            upside_pct=ddm.consensus_upside_pct,
            consecutive_years=ddm.sustainability.consecutive_growth_years,
        ))

    rows.sort(key=lambda r: (r.sustainability_score, r.yield_pct), reverse=True)
    avg_score = float(np.mean([r.sustainability_score for r in rows])) if rows else 0.0

    logger.info(
        "Dividend screen complete", screened=len(tickers), passed=len(rows),
        aristocrats=len(aristocrats), kings=len(kings), at_risk=len(at_risk),
        avg_score=round(avg_score, 2),
    )
    return DividendScreen(
        tickers_screened=len(tickers),
        results=rows,
        high_yield_count=high_yield_count,
        aristocrats=sorted(set(aristocrats)),
        kings=sorted(set(kings)),
        at_risk=sorted(set(at_risk)),
        avg_sustainability_score=round(avg_score, 2),
        as_of=as_of,
        warnings=screen_warnings,
    )
