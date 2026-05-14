"""Earnings quality analytics via EDGAR XBRL: accruals, cash conversion, operating leverage — Dimension 17/19 enhancement."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Optional

import numpy as np
import httpx
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# -- Constants ----------------------------------------------------------------

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_CIK_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&company={ticker}&type=10-K&dateb=&owner=include&count=5&search_text=&output=atom"
)
_HEADERS = {"User-Agent": "SENTINEL/1.0 research@sentinel.ai"}
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB cap on XBRL facts payload
_ANNUAL_FORMS = {"10-K", "10-K405"}

# -- Pydantic models ----------------------------------------------------------


class AnnualAccruals(BaseModel):
    model_config = ConfigDict(frozen=True)

    fiscal_year: str
    net_income: Optional[float] = None
    cfo: Optional[float] = None
    total_assets: Optional[float] = None
    accruals_ratio: Optional[float] = None
    sloan_ratio: Optional[float] = None
    cash_conversion: Optional[float] = None
    operating_leverage: Optional[float] = None
    quality_flag: str = "moderate"


class EarningsQualityProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: Optional[str] = None
    periods: list[AnnualAccruals] = Field(default_factory=list)
    avg_accruals_ratio: Optional[float] = None
    avg_sloan_ratio: Optional[float] = None
    avg_cash_conversion: Optional[float] = None
    quality_score: float = 0.0
    quality_tier: str = "Very Low Quality"
    trend: str = "stable"
    risk_flags: list[str] = Field(default_factory=list)
    data_quality: str = "estimated"
    as_of: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    warnings: list[str] = Field(default_factory=list)


class QualitySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    quality_score: float
    quality_tier: str
    avg_accruals_ratio: Optional[float] = None
    avg_cash_conversion: Optional[float] = None
    trend: str


class EarningsQualityScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_quality_filter: float
    results: list[QualitySummary] = Field(default_factory=list)
    high_quality_count: int = 0
    low_quality_count: int = 0
    best_quality: Optional[str] = None
    worst_quality: Optional[str] = None
    avg_score: Optional[float] = None
    as_of: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    warnings: list[str] = Field(default_factory=list)


# -- EDGAR network helpers ----------------------------------------------------


async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit CIK via EDGAR ATOM feed."""
    url = EDGAR_CIK_URL.format(ticker=ticker.upper())
    try:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        m = re.search(r"CIK=(\d+)", resp.text)
        if m:
            return m.group(1).zfill(10)
        logger.warning("CIK not found in ATOM feed", ticker=ticker)
        return None
    except Exception as exc:
        logger.error("CIK resolution failed", ticker=ticker, error=str(exc))
        return None


async def _fetch_xbrl_facts(cik: str, client: httpx.AsyncClient) -> dict:
    """Fetch EDGAR XBRL company facts JSON; cap at 2 MB to protect memory."""
    url = EDGAR_FACTS_URL.format(cik=cik)
    try:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        return json.loads(resp.content[:_MAX_RESPONSE_BYTES])
    except Exception as exc:
        logger.error("XBRL facts fetch failed", cik=cik, error=str(exc))
        return {}


# -- XBRL series extraction ---------------------------------------------------


def _extract_annual_series(facts: dict, tag_name: str, unit_key: str = "USD") -> dict[str, float]:
    """Extract annual (10-K) observations for a US-GAAP tag → {fiscal_year: value}.

    Deduplicates per fiscal year by keeping the most-recently-filed entry.
    Fiscal year sourced from 'frame' (CY20XX) then falls back to 'end' date year.
    """
    try:
        observations: list[dict] = facts["facts"]["us-gaap"][tag_name]["units"][unit_key]
    except (KeyError, TypeError):
        return {}

    best: dict[str, tuple[str, float]] = {}  # fy → (filed, value)
    for obs in observations:
        if obs.get("form", "") not in _ANNUAL_FORMS:
            continue
        val = obs.get("val")
        if val is None:
            continue
        filed = obs.get("filed", "")
        frame = obs.get("frame", "")
        if frame and re.match(r"CY\d{4}$", frame):
            fy = frame[2:]
        else:
            end_str = obs.get("end", "")
            if len(end_str) < 4:
                continue
            fy = end_str[:4]
        prev_filed, _ = best.get(fy, ("", 0.0))
        if filed >= prev_filed:
            best[fy] = (filed, float(val))

    return {fy: val for fy, (_, val) in best.items()}


# -- Computation helpers ------------------------------------------------------


def _accruals_ratio(net_income: float, cfo: float, avg_assets: float) -> Optional[float]:
    """Sloan (1996): (NI − CFO) / avg_assets.  Returns None when avg_assets is zero."""
    if avg_assets == 0:
        return None
    return (net_income - cfo) / avg_assets


def _cash_conversion(cfo: float, net_income: float) -> Optional[float]:
    """CFO / NI, capped at [−5, 5] to suppress sign-flip outliers."""
    if net_income == 0:
        return None
    return float(np.clip(cfo / net_income, -5.0, 5.0))


def _operating_leverage(
    rev_t: float, rev_tm1: float, opinc_t: float, opinc_tm1: float
) -> Optional[float]:
    """DOL = (ΔOpInc/OpInc_t-1) / (ΔRev/Rev_t-1), clamped [−50, 50].  None if denominator zero."""
    if rev_tm1 == 0 or opinc_tm1 == 0:
        return None
    rev_g = (rev_t - rev_tm1) / rev_tm1
    if rev_g == 0:
        return None
    opinc_g = (opinc_t - opinc_tm1) / opinc_tm1
    return float(np.clip(opinc_g / rev_g, -50.0, 50.0))


def _quality_flag(accruals_ratio: Optional[float], cash_conversion: Optional[float]) -> str:
    """Single-year quality classification: 'high quality' | 'moderate' | 'low quality'."""
    if accruals_ratio is None and cash_conversion is None:
        return "moderate"
    ar = accruals_ratio if accruals_ratio is not None else 0.0
    cc = cash_conversion if cash_conversion is not None else 1.0
    if ar < -0.05 and cc > 1.0:
        return "high quality"
    if ar > 0.05 or cc < 0.7:
        return "low quality"
    return "moderate"


def _quality_score(periods: list[AnnualAccruals]) -> float:
    """Composite score [0,10]: accruals_component×0.6 + cc_component×0.4.

    accruals_component: 10×(1 − clip((avg_ar+0.15)/0.30, 0, 1))  — lower accruals → higher.
    cc_component:       10×clip(avg_cc/1.5, 0, 1)                  — higher CFO/NI → higher.
    """
    if not periods:
        return 0.0
    ar_vals = [p.accruals_ratio for p in periods if p.accruals_ratio is not None]
    cc_vals = [float(np.clip(p.cash_conversion, 0.0, 2.0)) for p in periods if p.cash_conversion is not None]
    avg_ar = float(np.mean(ar_vals)) if ar_vals else 0.0
    avg_cc = float(np.mean(cc_vals)) if cc_vals else 0.5
    ar_norm = float(np.clip((avg_ar + 0.15) / 0.30, 0.0, 1.0))
    accruals_component = 10.0 * (1.0 - ar_norm)
    cc_component = 10.0 * float(np.clip(avg_cc / 1.5, 0.0, 1.0))
    return float(np.clip(accruals_component * 0.6 + cc_component * 0.4, 0.0, 10.0))


def _quality_tier(score: float) -> str:
    """Map score to named tier: AAA Quality / High Quality / Moderate / Low / Very Low."""
    if score >= 8:
        return "AAA Quality"
    if score >= 6:
        return "High Quality"
    if score >= 4:
        return "Moderate Quality"
    if score >= 2:
        return "Low Quality"
    return "Very Low Quality"


def _trend(periods: list[AnnualAccruals]) -> str:
    """'improving' | 'deteriorating' | 'stable' from first-half vs second-half accruals_ratio comparison (±0.02 threshold)."""
    ar_series = [p.accruals_ratio for p in periods if p.accruals_ratio is not None]
    if len(ar_series) < 2:
        return "stable"
    mid = len(ar_series) // 2
    first_avg = float(np.mean(ar_series[:mid] or ar_series[:1]))
    second_avg = float(np.mean(ar_series[mid:] or ar_series[-1:]))
    if second_avg < first_avg - 0.02:
        return "improving"
    if second_avg > first_avg + 0.02:
        return "deteriorating"
    return "stable"


# -- Risk flags ---------------------------------------------------------------


def _build_risk_flags(periods: list[AnnualAccruals]) -> list[str]:
    """Flag 'high accruals trend' (majority AR>0.05), 'poor cash conversion' (majority CC<0.7),
    'revenue/income divergence' (DOL std>5 or min<−3).
    """
    flags: list[str] = []
    if not periods:
        return flags
    ar_vals = [p.accruals_ratio for p in periods if p.accruals_ratio is not None]
    cc_vals = [p.cash_conversion for p in periods if p.cash_conversion is not None]
    ol_vals = [p.operating_leverage for p in periods if p.operating_leverage is not None]
    if ar_vals and sum(v > 0.05 for v in ar_vals) > len(ar_vals) / 2:
        flags.append("high accruals trend")
    if cc_vals and sum(v < 0.7 for v in cc_vals) > len(cc_vals) / 2:
        flags.append("poor cash conversion")
    if len(ol_vals) >= 2:
        ol_arr = np.array(ol_vals, dtype=float)
        if float(np.std(ol_arr)) > 5.0 or float(np.min(ol_arr)) < -3.0:
            flags.append("revenue/income divergence")
    return flags


# -- Helpers: data-quality label, revenue tag, period builder ----------------


def _data_quality_label(n_years: int) -> str:
    if n_years >= 3:
        return "xbrl"
    if n_years >= 1:
        return "partial"
    return "estimated"


def _extract_revenue(facts: dict) -> dict[str, float]:
    """Try primary Revenues tag; fall back to ASC 606 contract-based tag."""
    series = _extract_annual_series(facts, "Revenues")
    if not series:
        series = _extract_annual_series(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    return series


def _build_periods(
    fiscal_years: list[str],
    net_income_map: dict[str, float],
    cfo_map: dict[str, float],
    assets_map: dict[str, float],
    revenue_map: dict[str, float],
    opinc_map: dict[str, float],
    warnings: list[str],
) -> list[AnnualAccruals]:
    """Build AnnualAccruals for each year with net_income, cfo, and assets present.

    Uses prior-year assets for Sloan avg-assets denominator when available; falls
    back to current-year assets.  Operating leverage requires two consecutive years.
    """
    periods: list[AnnualAccruals] = []
    sorted_years = sorted(fiscal_years)
    for i, fy in enumerate(sorted_years):
        ni = net_income_map.get(fy)
        cfo = cfo_map.get(fy)
        assets_t = assets_map.get(fy)
        if ni is None or cfo is None or assets_t is None:
            warnings.append(f"Missing core XBRL data for {fy}; period skipped")
            continue
        prior_fy = sorted_years[i - 1] if i > 0 else None
        assets_tm1 = assets_map.get(prior_fy) if prior_fy else None
        avg_assets = (assets_t + assets_tm1) / 2.0 if assets_tm1 is not None else assets_t
        ar = _accruals_ratio(ni, cfo, avg_assets)
        cc = _cash_conversion(cfo, ni)
        ol: Optional[float] = None
        if prior_fy is not None:
            rev_t, rev_tm1 = revenue_map.get(fy), revenue_map.get(prior_fy)
            opinc_t, opinc_tm1 = opinc_map.get(fy), opinc_map.get(prior_fy)
            if None not in (rev_t, rev_tm1, opinc_t, opinc_tm1):
                ol = _operating_leverage(rev_t, rev_tm1, opinc_t, opinc_tm1)  # type: ignore[arg-type]
        periods.append(
            AnnualAccruals(
                fiscal_year=fy,
                net_income=ni,
                cfo=cfo,
                total_assets=assets_t,
                accruals_ratio=round(ar, 6) if ar is not None else None,
                sloan_ratio=round(ar, 6) if ar is not None else None,
                cash_conversion=round(cc, 6) if cc is not None else None,
                operating_leverage=round(ol, 4) if ol is not None else None,
                quality_flag=_quality_flag(ar, cc),
            )
        )
    return periods


# -- Public entry points ------------------------------------------------------


async def get_earnings_quality(ticker: str, years: int = 5) -> EarningsQualityProfile:
    """Resolve CIK, fetch EDGAR XBRL company facts, and compute earnings quality analytics.

    Returns an EarningsQualityProfile with per-year AnnualAccruals, aggregate ratios,
    quality score/tier/trend, risk flags, and data-quality label.  All network failures
    and missing tags are captured in *warnings* rather than raised as exceptions.
    """
    warnings: list[str] = []
    ticker = ticker.upper().strip()

    async with httpx.AsyncClient(timeout=30, headers=_HEADERS) as client:
        cik = await _resolve_cik(ticker, client)
        if not cik:
            warnings.append(f"Could not resolve CIK for {ticker}")
            return EarningsQualityProfile(ticker=ticker, warnings=warnings)

        facts = await _fetch_xbrl_facts(cik, client)

    if not facts:
        warnings.append(f"XBRL facts unavailable for CIK {cik}")
        return EarningsQualityProfile(ticker=ticker, cik=cik, warnings=warnings)

    net_income_map = _extract_annual_series(facts, "NetIncomeLoss")
    cfo_map = _extract_annual_series(facts, "NetCashProvidedByUsedInOperatingActivities")
    assets_map = _extract_annual_series(facts, "Assets")
    revenue_map = _extract_revenue(facts)
    opinc_map = _extract_annual_series(facts, "OperatingIncomeLoss")

    if not net_income_map:
        warnings.append("NetIncomeLoss absent from XBRL facts")
    if not cfo_map:
        warnings.append("NetCashProvidedByUsedInOperatingActivities absent from XBRL facts")
    if not assets_map:
        warnings.append("Assets absent from XBRL facts")
    if not revenue_map:
        warnings.append("Revenue tags absent; operating leverage unavailable")
    if not opinc_map:
        warnings.append("OperatingIncomeLoss absent; operating leverage unavailable")

    candidate_years: set[str] = set(net_income_map) & set(cfo_map) & set(assets_map)
    if not candidate_years:
        warnings.append("No overlapping fiscal years across core XBRL tags")
        return EarningsQualityProfile(ticker=ticker, cik=cik, warnings=warnings)

    # Cap to requested window; include one extra prior year for avg-assets / DOL denominators
    selected_years: list[str] = sorted(candidate_years, reverse=True)[:years]
    all_years = sorted(set(selected_years) | set(sorted(candidate_years, reverse=True)[: years + 1]))

    periods = _build_periods(
        fiscal_years=all_years,
        net_income_map=net_income_map,
        cfo_map=cfo_map,
        assets_map=assets_map,
        revenue_map=revenue_map,
        opinc_map=opinc_map,
        warnings=warnings,
    )

    # Restrict to the requested window, sorted ascending for trend analysis
    selected_set = set(selected_years)
    periods = sorted([p for p in periods if p.fiscal_year in selected_set], key=lambda p: p.fiscal_year)

    ar_vals = [p.accruals_ratio for p in periods if p.accruals_ratio is not None]
    sl_vals = [p.sloan_ratio for p in periods if p.sloan_ratio is not None]
    cc_vals = [p.cash_conversion for p in periods if p.cash_conversion is not None]
    avg_ar = round(float(np.mean(ar_vals)), 6) if ar_vals else None
    avg_sl = round(float(np.mean(sl_vals)), 6) if sl_vals else None
    avg_cc = round(float(np.mean(cc_vals)), 6) if cc_vals else None

    score = round(_quality_score(periods), 4)
    tier, trend = _quality_tier(score), _trend(periods)
    risk_flags, dq = _build_risk_flags(periods), _data_quality_label(len(periods))
    logger.info("Earnings quality computed", ticker=ticker, cik=cik, years=len(periods),
                score=score, tier=tier, trend=trend, risk_flags=risk_flags)

    return EarningsQualityProfile(
        ticker=ticker,
        cik=cik,
        periods=periods,
        avg_accruals_ratio=avg_ar,
        avg_sloan_ratio=avg_sl,
        avg_cash_conversion=avg_cc,
        quality_score=score,
        quality_tier=tier,
        trend=trend,
        risk_flags=risk_flags,
        data_quality=dq,
        warnings=warnings,
    )


async def screen_earnings_quality(
    tickers: list[str],
    min_quality_score: float = 5.0,
) -> EarningsQualityScreen:
    """Concurrently screen *tickers* for earnings quality; return ranked EarningsQualityScreen.

    Profiles are fetched via asyncio.gather.  Only tickers with quality_score >=
    *min_quality_score* appear in *results*, sorted descending.  Aggregate stats
    (high/low quality counts, best/worst ticker, avg_score) are always populated.
    """
    screen_warnings: list[str] = []

    if not tickers:
        return EarningsQualityScreen(
            tickers_screened=0,
            min_quality_filter=min_quality_score,
            warnings=["No tickers provided"],
        )

    profiles: list[EarningsQualityProfile] = await asyncio.gather(
        *[get_earnings_quality(t) for t in tickers]
    )

    for profile in profiles:  # bubble per-ticker warnings to screen level (deduplicated)
        for w in profile.warnings:
            msg = f"[{profile.ticker}] {w}"
            if msg not in screen_warnings:
                screen_warnings.append(msg)

    summaries: list[QualitySummary] = [
        QualitySummary(
            ticker=p.ticker,
            quality_score=p.quality_score,
            quality_tier=p.quality_tier,
            avg_accruals_ratio=p.avg_accruals_ratio,
            avg_cash_conversion=p.avg_cash_conversion,
            trend=p.trend,
        )
        for p in profiles
    ]

    summaries_sorted = sorted(summaries, key=lambda s: s.quality_score, reverse=True)
    passing = [s for s in summaries_sorted if s.quality_score >= min_quality_score]
    high_quality_count = sum(1 for s in summaries if s.quality_score >= 6.0)
    low_quality_count = sum(1 for s in summaries if s.quality_score < 4.0)
    best = summaries_sorted[0].ticker if summaries_sorted else None
    worst = summaries_sorted[-1].ticker if summaries_sorted else None
    score_vals = [s.quality_score for s in summaries]
    avg_score: Optional[float] = round(float(np.mean(score_vals)), 4) if score_vals else None

    logger.info(
        "Earnings quality screen complete",
        tickers_screened=len(tickers), passing=len(passing),
        min_quality_score=min_quality_score, best=best, worst=worst, avg_score=avg_score,
    )

    return EarningsQualityScreen(
        tickers_screened=len(tickers),
        min_quality_filter=min_quality_score,
        results=passing,
        high_quality_count=high_quality_count,
        low_quality_count=low_quality_count,
        best_quality=best,
        worst_quality=worst,
        avg_score=avg_score,
        warnings=screen_warnings,
    )
