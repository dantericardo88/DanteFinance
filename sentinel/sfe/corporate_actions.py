"""Dividend analytics and corporate actions intelligence — Bloomberg DVDS parity.

Provides:
- Full dividend history analysis with CAGR, consistency, and quality scoring
- DDM (Gordon Growth Model) intrinsic value estimation
- Corporate action detection: splits, reverse splits, special dividends, cuts, initiations
- Multi-ticker dividend screener with asyncio.gather parallelism
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_ERP = 0.05           # equity risk premium
_FALLBACK_RF = 0.045  # if FRED unreachable


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DividendHistory(BaseModel):
    date: str
    amount: float
    frequency: str  # "quarterly" | "monthly" | "annual" | "special"


class CorporateAction(BaseModel):
    date: str
    action_type: str  # "split"|"reverse_split"|"special_dividend"|"dividend_cut"|"dividend_initiation"
    description: str
    value: Optional[float] = None


class DividendAnalytics(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    current_yield_pct: Optional[float] = None
    annual_dividend_usd: Optional[float] = None
    ex_dividend_date: Optional[str] = None
    pay_date: Optional[str] = None
    payout_ratio_pct: Optional[float] = None
    dividend_growth_rate_5y: Optional[float] = None   # CAGR %
    dividend_consistency_pct: Optional[float] = None  # % quarters with divs (5y)
    five_year_avg_yield: Optional[float] = None
    trailing_12m_dividends: Optional[float] = None
    dividend_quality_score: float
    dividend_quality_label: str   # "high" | "medium" | "low" | "none"
    ddm_intrinsic_value: Optional[float] = None
    ddm_verdict: str   # "undervalued" | "fairly valued" | "overvalued" | "N/A"
    recent_history: list[DividendHistory] = Field(default_factory=list)
    corporate_actions: list[CorporateAction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    as_of: str


class DividendScreenResult(BaseModel):
    tickers: list[str]
    analytics: list[DividendAnalytics]
    avg_yield_pct: Optional[float] = None
    avg_quality_score: Optional[float] = None
    top_yielder: Optional[str] = None
    highest_quality: Optional[str] = None
    as_of: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val: object) -> float | None:
    """Safe float: None for NaN/inf/non-numeric."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _unix_to_iso(ts: object) -> str | None:
    v = _sf(ts)
    if v is None:
        return None
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _tz_index(s):
    """Ensure a pandas Series has a UTC-aware DatetimeIndex."""
    if s.index.tzinfo is None:
        return s.tz_localize("UTC")
    return s


async def _fetch_risk_free_rate() -> float:
    """Fetch FEDFUNDS from FRED CSV; fall back to _FALLBACK_RF on any error."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_FRED_BASE}?id=FEDFUNDS")
            resp.raise_for_status()
        for line in reversed(resp.text.strip().splitlines()):
            if line.startswith("DATE") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) == 2:
                v = _sf(parts[1])
                if v is not None:
                    return v / 100.0
    except Exception as exc:
        logger.warning("FRED FEDFUNDS fetch failed", error=str(exc))
    return _FALLBACK_RF


# ---------------------------------------------------------------------------
# Core synchronous analytics (runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------

def _fetch_raw(ticker: str) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    info, divs, splits = {}, None, None
    try:
        info = tk.info or {}
    except Exception as e:
        logger.warning("yfinance info failed", ticker=ticker, error=str(e))
    try:
        divs = tk.dividends
    except Exception as e:
        logger.warning("yfinance dividends failed", ticker=ticker, error=str(e))
    try:
        splits = tk.splits
    except Exception as e:
        logger.warning("yfinance splits failed", ticker=ticker, error=str(e))
    return {"info": info, "dividends": divs, "splits": splits}


def _classify_freq(gaps: list[float]) -> str:
    if not gaps:
        return "annual"
    med = float(np.median(gaps))
    if med <= 45:   return "monthly"
    if med <= 105:  return "quarterly"
    if med <= 200:  return "semi-annual"
    return "annual"


def _build_history(divs) -> tuple[list[DividendHistory], list[float]]:
    if divs is None or divs.empty:
        return [], []
    sorted_d = divs.sort_index()
    dts, amts = list(sorted_d.index), list(sorted_d.values)
    gaps = [float((dts[i] - dts[i - 1]).days) for i in range(1, len(dts))]
    freq = _classify_freq(gaps)
    history = []
    for dt, amt in zip(dts, amts):
        try:
            d = dt.date() if hasattr(dt, "date") else dt
            history.append(DividendHistory(date=d.isoformat(), amount=round(float(amt), 6), frequency=freq))
        except Exception:
            pass
    return history, gaps


def _trailing_12m(divs) -> float | None:
    if divs is None or divs.empty:
        return None
    try:
        import pandas as pd
        s = _tz_index(divs)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)
        recent = s[s.index >= cutoff]
        return round(float(recent.sum()), 4) if not recent.empty else None
    except Exception:
        return None


def _dgr_5y(divs) -> float | None:
    if divs is None or divs.empty:
        return None
    try:
        import pandas as pd
        now = pd.Timestamp.now(tz="UTC")
        s = pd.Series(_tz_index(divs).values, index=_tz_index(divs).index)
        five = s[s.index >= now - pd.DateOffset(years=5)]
        if len(five) < 4:
            return None
        annual = five.resample("YE").sum()
        annual = annual[annual > 0]
        if len(annual) < 2:
            return None
        cagr = (float(annual.iloc[-1]) / float(annual.iloc[0])) ** (1.0 / (len(annual) - 1)) - 1
        return round(cagr * 100, 2)
    except Exception:
        return None


def _consistency_5y(divs) -> float | None:
    if divs is None or divs.empty:
        return None
    try:
        import pandas as pd
        now = pd.Timestamp.now(tz="UTC")
        s = pd.Series(_tz_index(divs).values, index=_tz_index(divs).index)
        five = s[s.index >= now - pd.DateOffset(years=5)]
        if five.empty:
            return None
        q = five.resample("QE").sum()
        return round(float((q > 0).sum()) / len(q) * 100, 1)
    except Exception:
        return None


def _detect_actions(divs, splits) -> list[CorporateAction]:
    actions: list[CorporateAction] = []
    try:
        import pandas as pd
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - pd.DateOffset(years=3)

        # Splits / reverse splits
        if splits is not None and not splits.empty:
            s = _tz_index(splits)
            for dt, ratio in s[s.index >= cutoff].items():
                d = (dt.date() if hasattr(dt, "date") else dt).isoformat()
                r = float(ratio)
                if r > 1:
                    actions.append(CorporateAction(date=d, action_type="split",
                        description=f"{r:.0f}-for-1 stock split", value=r))
                elif 0 < r < 1:
                    actions.append(CorporateAction(date=d, action_type="reverse_split",
                        description=f"1-for-{1/r:.0f} reverse stock split", value=r))

        if divs is None or divs.empty:
            return actions

        full = pd.Series(_tz_index(divs).values, index=_tz_index(divs).index)
        recent = full[full.index >= cutoff]
        trailing4_avg = float(full[full.index < cutoff].iloc[-4:].mean()) if len(full[full.index < cutoff]) >= 4 else float(full.mean())
        qall = full.resample("QE").sum()

        # Special dividends
        for dt, amt in recent.items():
            a = float(amt)
            if trailing4_avg > 0 and a > 2.0 * trailing4_avg:
                d = (dt.date() if hasattr(dt, "date") else dt).isoformat()
                actions.append(CorporateAction(date=d, action_type="special_dividend",
                    description=f"Special dividend ${a:.4f} (>2x avg ${trailing4_avg:.4f})", value=a))

        # Cuts and initiations from quarterly series
        for q_date, q_amt in qall[qall.index >= cutoff].items():
            d = (q_date.date() if hasattr(q_date, "date") else q_date).isoformat()
            # Initiation: positive after 4+ empty quarters
            prev = qall[(qall.index >= q_date - pd.DateOffset(months=12)) & (qall.index < q_date)]
            if q_amt > 0 and (prev.empty or (prev == 0).all()):
                very_old = qall[qall.index < q_date - pd.DateOffset(months=12)]
                if not very_old.empty and (very_old > 0).any():
                    actions.append(CorporateAction(date=d, action_type="dividend_initiation",
                        description="Dividend reinitiated after 4+ quarters with none", value=float(q_amt)))

            # Cut: >10% below same quarter prior year
            pyear = qall[(qall.index >= q_date - pd.DateOffset(months=15)) &
                         (qall.index <= q_date - pd.DateOffset(months=9))]
            if not pyear.empty and q_amt > 0:
                prior = float(pyear.iloc[-1])
                if prior > 0 and q_amt < prior * 0.90:
                    actions.append(CorporateAction(date=d, action_type="dividend_cut",
                        description=f"Dividend cut: ${q_amt:.4f} vs ${prior:.4f} year-ago", value=float(q_amt)))

    except Exception as exc:
        logger.warning("Corporate action detection failed", error=str(exc))

    # Deduplicate and sort descending
    seen: set[tuple] = set()
    out: list[CorporateAction] = []
    for a in sorted(actions, key=lambda x: x.date, reverse=True):
        key = (a.date, a.action_type)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _quality_score(consistency, dgr_5y, payout_ratio, current_yield, rf: float) -> tuple[float, str]:
    score = 0.0
    if consistency is not None:
        score += 3.0 if consistency >= 95 else (2.0 if consistency >= 75 else (1.0 if consistency >= 50 else 0.0))
    if dgr_5y is not None:
        score += 3.0 if dgr_5y >= 10 else (2.0 if dgr_5y >= 5 else (1.0 if dgr_5y >= 0 else 0.0))
    if payout_ratio is not None:
        score += 2.0 if payout_ratio < 60 else (1.0 if payout_ratio < 80 else 0.0)
    treasury_10y_pct = (rf + 0.01) * 100
    if current_yield is not None:
        spread = current_yield - treasury_10y_pct
        score += 2.0 if spread >= 2.0 else (1.0 if spread >= 0 else 0.0)
    label = "none" if score == 0 else ("low" if score < 4 else ("medium" if score < 7 else "high"))
    return round(score, 1), label


def _ddm(annual_div, dgr_5y, payout_ratio, current_price, rf: float) -> tuple[float | None, str]:
    if not annual_div or annual_div <= 0 or not current_price or current_price <= 0:
        return None, "N/A"
    dgr = (dgr_5y / 100.0) if dgr_5y is not None else 0.03
    if payout_ratio is not None and payout_ratio < 100:
        g = min(dgr, 0.12 * (1.0 - payout_ratio / 100.0))
    else:
        g = min(dgr, 0.04)
    g = max(0.0, min(g, 0.07))
    r = rf + _ERP
    if r <= g:
        return None, "N/A"
    intrinsic = annual_div * (1.0 + g) / (r - g)
    verdict = "undervalued" if intrinsic > current_price * 1.15 else ("overvalued" if intrinsic < current_price * 0.85 else "fairly valued")
    return round(intrinsic, 2), verdict


def _build_analytics(ticker: str, raw: dict, rf: float) -> DividendAnalytics:
    ticker = ticker.upper()
    info, divs, splits = raw["info"], raw["dividends"], raw["splits"]
    warns: list[str] = []

    current_yield = _sf(info.get("dividendYield"))
    if current_yield is not None:
        current_yield = round(current_yield * 100, 4)
    annual_div = _sf(info.get("dividendRate"))
    payout_ratio = _sf(info.get("payoutRatio"))
    if payout_ratio is not None:
        payout_ratio = round(payout_ratio * 100, 2)
    current_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))

    history, _ = _build_history(divs)
    t12m = _trailing_12m(divs)
    dgr = _dgr_5y(divs)
    cons = _consistency_5y(divs)
    if annual_div is None and t12m is not None:
        annual_div = t12m

    corp = _detect_actions(divs, splits)

    has_div = (divs is not None and not divs.empty) or bool(annual_div and annual_div > 0)
    if has_div:
        score, label = _quality_score(cons, dgr, payout_ratio, current_yield, rf)
    else:
        score, label = 0.0, "none"

    ddm_val, ddm_verdict = _ddm(annual_div, dgr, payout_ratio, current_price, rf)

    if not info:
        warns.append("yfinance returned no info — ticker may be invalid")

    logger.info("Dividend analytics built", ticker=ticker, score=score, label=label,
                yield_pct=current_yield, ddm_verdict=ddm_verdict)

    return DividendAnalytics(
        ticker=ticker,
        company_name=info.get("longName") or info.get("shortName") or None,
        current_yield_pct=current_yield,
        annual_dividend_usd=round(annual_div, 4) if annual_div is not None else None,
        ex_dividend_date=_unix_to_iso(info.get("exDividendDate")),
        pay_date=_unix_to_iso(info.get("payDate")),
        payout_ratio_pct=payout_ratio,
        dividend_growth_rate_5y=dgr,
        dividend_consistency_pct=cons,
        five_year_avg_yield=_sf(info.get("fiveYearAvgDividendYield")),
        trailing_12m_dividends=t12m,
        dividend_quality_score=score,
        dividend_quality_label=label,
        ddm_intrinsic_value=ddm_val,
        ddm_verdict=ddm_verdict,
        recent_history=history[-8:],
        corporate_actions=corp,
        warnings=warns,
        as_of=date.today().isoformat(),
    )


# ---------------------------------------------------------------------------
# Public async entry points
# ---------------------------------------------------------------------------

async def get_dividend_analytics(ticker: str) -> DividendAnalytics:
    """Full dividend and corporate action analytics for a single ticker."""
    logger.info("get_dividend_analytics called", ticker=ticker.upper())
    raw, rf = await asyncio.gather(
        asyncio.to_thread(_fetch_raw, ticker),
        _fetch_risk_free_rate(),
    )
    return _build_analytics(ticker, raw, rf)


async def screen_dividends(
    tickers: list[str],
    min_yield_pct: float | None = None,
    min_quality_score: float | None = None,
    exclude_no_dividend: bool = True,
) -> DividendScreenResult:
    """Screen multiple tickers for dividend quality and yield via asyncio.gather."""
    if not tickers:
        return DividendScreenResult(tickers=[], analytics=[], as_of=date.today().isoformat())

    logger.info("screen_dividends called", tickers=tickers, min_yield=min_yield_pct,
                min_score=min_quality_score, exclude_none=exclude_no_dividend)

    results = await asyncio.gather(*[get_dividend_analytics(t) for t in tickers],
                                   return_exceptions=True)

    analytics: list[DividendAnalytics] = []
    for ticker, res in zip(tickers, results):
        if isinstance(res, Exception):
            logger.warning("Ticker failed in screen", ticker=ticker, error=str(res))
        else:
            analytics.append(res)

    if exclude_no_dividend:
        analytics = [a for a in analytics if a.dividend_quality_label != "none"]
    if min_yield_pct is not None:
        analytics = [a for a in analytics if a.current_yield_pct is not None and a.current_yield_pct >= min_yield_pct]
    if min_quality_score is not None:
        analytics = [a for a in analytics if a.dividend_quality_score >= min_quality_score]

    yields = [a.current_yield_pct for a in analytics if a.current_yield_pct is not None]
    scores = [a.dividend_quality_score for a in analytics]

    return DividendScreenResult(
        tickers=[t.upper() for t in tickers],
        analytics=analytics,
        avg_yield_pct=round(float(np.mean(yields)), 4) if yields else None,
        avg_quality_score=round(float(np.mean(scores)), 2) if scores else None,
        top_yielder=max(analytics, key=lambda a: a.current_yield_pct or 0.0).ticker if yields else None,
        highest_quality=max(analytics, key=lambda a: a.dividend_quality_score).ticker if scores else None,
        as_of=date.today().isoformat(),
    )
