"""Liquidity analytics — Amihud, Roll, Corwin-Schultz, Kyle's Lambda, turnover, ADV."""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LiquidityMetrics(BaseModel):
    ticker: str
    period_days: int
    amihud_illiquidity: Optional[float] = None       # daily avg × 1e6 (scaled)
    amihud_annualized: Optional[float] = None        # × 252
    roll_spread_pct: Optional[float] = None          # % of price
    corwin_schultz_spread_pct: Optional[float] = None
    kyle_lambda: Optional[float] = None              # price impact per $M volume
    turnover_ratio_annualized: Optional[float] = None
    adv_20d_usd: Optional[float] = None
    adv_60d_usd: Optional[float] = None
    volume_spike_days: int = 0
    liquidity_score: float
    liquidity_label: str
    warnings: list[str] = Field(default_factory=list)


class PortfolioLiquidity(BaseModel):
    tickers: list[str]
    weights: list[float]
    metrics: list[LiquidityMetrics]
    portfolio_adv_usd: Optional[float] = None
    portfolio_liquidity_score: float
    days_to_liquidate_90pct: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    as_of: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_yf_data(ticker: str, period_days: int) -> tuple[dict, dict]:
    """Sync yfinance fetch — run via asyncio.to_thread."""
    import yfinance as yf  # noqa: PLC0415

    obj = yf.Ticker(ticker)
    hist = obj.history(period=f"{period_days + 10}d", auto_adjust=True)
    info: dict = {}
    try:
        info = obj.info or {}
    except Exception:
        pass
    return hist, info


def _amihud(closes: np.ndarray, volumes: np.ndarray) -> tuple[float, float]:
    """Return (daily_avg_scaled, annualized). Scaled by 1e6 for readability."""
    returns = np.abs(np.diff(np.log(closes)))
    dollar_vol = volumes[1:] * closes[1:]
    mask = dollar_vol > 0
    if mask.sum() < 5:
        raise ValueError("Insufficient data for Amihud")
    illiq = returns[mask] / dollar_vol[mask]
    daily_avg = float(np.mean(illiq)) * 1e6
    return daily_avg, daily_avg * 252


def _roll_spread(closes: np.ndarray, mean_price: float) -> float:
    """Roll (1984) bid-ask spread estimator as % of price. Returns 0 if cov >= 0."""
    dp = np.diff(closes)
    if len(dp) < 4:
        raise ValueError("Insufficient data for Roll spread")
    cov = float(np.cov(dp[:-1], dp[1:])[0, 1])
    if cov >= 0:
        return 0.0
    roll_dollar = 2.0 * np.sqrt(-cov)
    return (roll_dollar / mean_price) * 100.0


def _corwin_schultz(highs: np.ndarray, lows: np.ndarray) -> float:
    """Corwin-Schultz (2012) spread estimator. Returns % of price, clipped to 0."""
    n = len(highs)
    if n < 2:
        raise ValueError("Insufficient data for Corwin-Schultz")

    log_hl = np.log(highs / lows)
    beta_vals = log_hl[:-1] ** 2 + log_hl[1:] ** 2
    beta = float(np.mean(beta_vals))

    h2 = np.maximum(highs[:-1], highs[1:])
    l2 = np.minimum(lows[:-1], lows[1:])
    gamma_vals = np.log(h2 / l2) ** 2
    gamma = float(np.mean(gamma_vals))

    sqrt2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return max(0.0, float(spread) * 100.0)


def _kyle_lambda(closes: np.ndarray, volumes: np.ndarray) -> float:
    """OLS regression of ΔP on signed-volume. λ in price units per $M dollar volume."""
    dp = np.diff(closes)
    ret_sign = np.sign(dp)
    dollar_vol = volumes[1:] * closes[1:]
    signed_vol = dollar_vol * ret_sign / 1e6  # scale to $M

    if len(signed_vol) < 10:
        raise ValueError("Insufficient data for Kyle lambda")

    A = np.column_stack([signed_vol, np.ones(len(signed_vol))])
    result = np.linalg.lstsq(A, dp, rcond=None)
    return float(result[0][0])


def _turnover(volumes: np.ndarray, closes: np.ndarray, shares_outstanding: Optional[float]) -> float:
    """Daily average turnover annualized. Raises if shares_outstanding unavailable."""
    if shares_outstanding is None or shares_outstanding <= 0:
        raise ValueError("Shares outstanding unavailable")
    daily_turnover = volumes / shares_outstanding
    return float(np.mean(daily_turnover)) * 252.0 * 100.0  # as %


def _adv(closes: np.ndarray, volumes: np.ndarray, window: int) -> Optional[float]:
    """Average daily dollar volume over last `window` days."""
    dv = closes * volumes
    if len(dv) < window:
        return float(np.mean(dv)) if len(dv) > 0 else None
    return float(np.mean(dv[-window:]))


def _volume_spikes(volumes: np.ndarray, window: int = 20) -> int:
    """Count days where volume > 2× the trailing 20-day average."""
    if len(volumes) <= window:
        return 0
    count = 0
    for i in range(window, len(volumes)):
        avg = np.mean(volumes[i - window:i])
        if volumes[i] > 2.0 * avg:
            count += 1
    return count


def _liquidity_score(amihud_annualized: Optional[float], cs_spread: Optional[float]) -> tuple[float, str]:
    """Score 1-10 from Amihud + C-S spread. Falls back gracefully if either is None."""
    score = 5.0

    if amihud_annualized is not None and amihud_annualized > 0:
        log_a = np.log10(amihud_annualized + 1e-12)
        # Empirical range: mega-cap ~-2, micro-cap ~4
        amihud_score = np.clip(10.0 - (log_a + 2.0) * (9.0 / 6.0), 1.0, 10.0)
        score = float(amihud_score)

    if cs_spread is not None:
        # Spread ranges: ~0.01% (liquid ETF) to ~3%+ (illiquid)
        spread_score = np.clip(10.0 - cs_spread * 4.0, 1.0, 10.0)
        if amihud_annualized is not None:
            score = 0.6 * score + 0.4 * float(spread_score)
        else:
            score = float(spread_score)

    score = float(np.clip(score, 1.0, 10.0))

    if score >= 8.5:
        label = "high"
    elif score >= 6.0:
        label = "medium"
    elif score >= 3.5:
        label = "low"
    else:
        label = "illiquid"

    return score, label


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def get_liquidity_metrics(ticker: str, period_days: int = 63) -> LiquidityMetrics:
    """Compute all liquidity metrics for a single ticker over period_days trading days."""
    log = logger.bind(ticker=ticker, period_days=period_days)
    warnings: list[str] = []

    hist, info = await asyncio.to_thread(_fetch_yf_data, ticker, period_days)

    if hist is None or len(hist) == 0:
        warnings.append("No price data returned from yfinance")
        score, label = _liquidity_score(None, None)
        return LiquidityMetrics(
            ticker=ticker,
            period_days=period_days,
            liquidity_score=score,
            liquidity_label=label,
            warnings=warnings,
        )

    closes = hist["Close"].to_numpy(dtype=float)
    highs = hist["High"].to_numpy(dtype=float)
    lows = hist["Low"].to_numpy(dtype=float)
    volumes = hist["Volume"].to_numpy(dtype=float)

    # Guard against zero prices / volumes corrupting metrics
    valid = (closes > 0) & (volumes > 0) & (highs > 0) & (lows > 0)
    closes = closes[valid]
    highs = highs[valid]
    lows = lows[valid]
    volumes = volumes[valid]

    n = len(closes)
    log.debug("data loaded", rows=n)

    if n < 5:
        warnings.append(f"Only {n} valid trading days — most metrics skipped")
        score, label = _liquidity_score(None, None)
        return LiquidityMetrics(
            ticker=ticker,
            period_days=period_days,
            liquidity_score=score,
            liquidity_label=label,
            warnings=warnings,
        )

    # --- Amihud ---
    amihud_daily: Optional[float] = None
    amihud_ann: Optional[float] = None
    try:
        amihud_daily, amihud_ann = _amihud(closes, volumes)
    except Exception as exc:
        warnings.append(f"Amihud skipped: {exc}")

    # --- Roll ---
    roll: Optional[float] = None
    if n >= 20:
        try:
            roll = _roll_spread(closes, float(np.mean(closes)))
        except Exception as exc:
            warnings.append(f"Roll spread skipped: {exc}")
    else:
        warnings.append("Roll spread skipped: fewer than 20 observations")

    # --- Corwin-Schultz ---
    cs: Optional[float] = None
    try:
        cs = _corwin_schultz(highs, lows)
    except Exception as exc:
        warnings.append(f"Corwin-Schultz skipped: {exc}")

    # --- Kyle's Lambda ---
    kyle: Optional[float] = None
    if n >= 20:
        try:
            kyle = _kyle_lambda(closes, volumes)
        except Exception as exc:
            warnings.append(f"Kyle lambda skipped: {exc}")
    else:
        warnings.append("Kyle lambda skipped: fewer than 20 observations")

    # --- Turnover ---
    turnover: Optional[float] = None
    shares_out: Optional[float] = info.get("sharesOutstanding")
    if shares_out is None:
        mktcap = info.get("marketCap")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(closes[-1]) if n > 0 else None)
        if mktcap and price and price > 0:
            shares_out = mktcap / price
            warnings.append("Shares outstanding estimated from marketCap/price")
    try:
        turnover = _turnover(volumes, closes, shares_out)
    except Exception as exc:
        warnings.append(f"Turnover skipped: {exc}")

    # --- ADV ---
    adv20 = _adv(closes, volumes, 20)
    adv60 = _adv(closes, volumes, 60)

    # --- Volume spikes ---
    spikes = _volume_spikes(volumes)

    # --- Score ---
    score, label = _liquidity_score(amihud_ann, cs)

    log.info("liquidity metrics computed", score=score, label=label)

    return LiquidityMetrics(
        ticker=ticker,
        period_days=period_days,
        amihud_illiquidity=amihud_daily,
        amihud_annualized=amihud_ann,
        roll_spread_pct=roll,
        corwin_schultz_spread_pct=cs,
        kyle_lambda=kyle,
        turnover_ratio_annualized=turnover,
        adv_20d_usd=adv20,
        adv_60d_usd=adv60,
        volume_spike_days=spikes,
        liquidity_score=score,
        liquidity_label=label,
        warnings=warnings,
    )


async def get_portfolio_liquidity(
    tickers: list[str],
    weights: list[float] | None = None,
    period_days: int = 63,
) -> PortfolioLiquidity:
    """Portfolio-level liquidity: aggregated metrics + liquidation horizon."""
    if not tickers:
        raise ValueError("tickers must be non-empty")

    n = len(tickers)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match tickers length")
        total = sum(weights)
        weights = [w / total for w in weights]

    warnings: list[str] = []

    all_metrics: list[LiquidityMetrics] = list(
        await asyncio.gather(*[get_liquidity_metrics(t, period_days) for t in tickers])
    )

    # Portfolio ADV = weighted sum of individual ADVs
    portfolio_adv: Optional[float] = None
    adv_vals = [m.adv_20d_usd for m in all_metrics]
    if all(v is not None for v in adv_vals):
        portfolio_adv = sum(w * v for w, v in zip(weights, adv_vals))  # type: ignore[arg-type]
    elif any(v is not None for v in adv_vals):
        available = [(w, v) for w, v in zip(weights, adv_vals) if v is not None]
        w_sum = sum(w for w, _ in available)
        portfolio_adv = sum(w * v for w, v in available) / w_sum if w_sum > 0 else None
        warnings.append("Portfolio ADV estimated from subset of tickers with available ADV data")

    # Days to liquidate 90% assumes $10M portfolio, 20% of ADV per day
    days_liq: Optional[float] = None
    if portfolio_adv and portfolio_adv > 0:
        portfolio_value = 10_000_000.0
        warnings.append("days_to_liquidate_90pct assumes $10M portfolio value")
        days_liq = (0.9 * portfolio_value) / (0.20 * portfolio_adv)

    port_score = float(
        sum(w * m.liquidity_score for w, m in zip(weights, all_metrics))
    )

    return PortfolioLiquidity(
        tickers=tickers,
        weights=weights,
        metrics=all_metrics,
        portfolio_adv_usd=portfolio_adv,
        portfolio_liquidity_score=port_score,
        days_to_liquidate_90pct=days_liq,
        warnings=warnings,
        as_of=str(date.today()),
    )
