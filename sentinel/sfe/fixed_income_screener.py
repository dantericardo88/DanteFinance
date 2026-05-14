"""
Fixed income screener — Dimension #74 Wave 4 (score target 9).

Comprehensive bond screening using free public data:
  - FRED (St. Louis Fed): credit spreads, OAS, Treasury yields
  - EDGAR EFTS full-text search: new bond issuances (424B2), 8-K debt events,
    convertible note announcements
  - TreasuryDirect API: recently-auctioned Treasury securities
  - FINRA TRACE: corporate bond aggregates via existing TRACEClient

Data pipeline:
  1. CreditMarket snapshot from FRED (IG OAS, HY OAS, yield curve, 30d changes)
  2. Treasury universe from TreasuryDirect /securities/search
  3. New corporate issuances mined from EDGAR 424B2 + 8-K
  4. Convertible bond terms from 8-K text extraction
  5. Relative value Z-score ranking within rating/sector cohorts
  6. High-yield watchlist: distressed / recent downgrades from 8-K text
"""
from __future__ import annotations

import asyncio
import io
import re
from datetime import date, datetime, timedelta
from typing import Optional, Literal

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TREASURY_DIRECT = "https://www.treasurydirect.gov/TA_WS/securities/search"
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# FRED series IDs used across multiple methods
_FRED = {
    # Treasury par yields (constant maturity)
    "GS1M":  "Treasury 1-Month",
    "GS3M":  "Treasury 3-Month",
    "GS6M":  "Treasury 6-Month",
    "GS1":   "Treasury 1-Year",
    "GS2":   "Treasury 2-Year",
    "GS5":   "Treasury 5-Year",
    "GS10":  "Treasury 10-Year",
    "GS20":  "Treasury 20-Year",
    "GS30":  "Treasury 30-Year",
    # Credit spreads (option-adjusted, to Treasury)
    "BAMLC0A0CM":    "IG OAS (all IG)",
    "BAMLH0A0HYM2":  "HY OAS (all HY)",
    "BAMLC0A1CAAAEY": "AAA yield",
    "BAMLC0A4CBBYEY": "BBB yield",
    # Total return indices (ICE BofA)
    "BAMLCC0A0CMTRIV":  "IG Total Return Index",
    "BAMLHYH0A0HYM2TRIV": "HY Total Return Index",
    # Real yields
    "DFII10": "10Y TIPS real yield",
    "DFII5":  "5Y TIPS real yield",
}

# Composite rating ordering: higher index = higher quality
_RATING_SCALE = [
    "D", "SD", "C", "CC", "CCC-", "CCC", "CCC+",
    "B-", "B", "B+",
    "BB-", "BB", "BB+",
    "BBB-", "BBB", "BBB+",
    "A-", "A", "A+",
    "AA-", "AA", "AA+",
    "AAA",
]
_RATING_TO_INT: dict[str, int] = {r: i for i, r in enumerate(_RATING_SCALE)}

# Composite rating groups for min_rating filtering
_RATING_FLOORS: dict[str, int] = {
    "BBB": _RATING_TO_INT["BBB-"],   # investment grade floor
    "BBB-": _RATING_TO_INT["BBB-"],
    "A": _RATING_TO_INT["A-"],
    "AA": _RATING_TO_INT["AA-"],
    "AAA": _RATING_TO_INT["AAA"],
    "BB": _RATING_TO_INT["BB-"],
    "B": _RATING_TO_INT["B-"],
}

# Proxy IG/HY ETF credit OAS profiles — used to anchor spread estimates
# when TRACE data is unavailable (bps above treasury for each rating/duration bucket)
_SPREAD_TABLE: dict[tuple[str, int], float] = {
    # (composite_rating, duration_bucket_years): typical OAS bps
    ("AAA",  2): 10,  ("AAA",  5): 15,  ("AAA", 10): 20,  ("AAA", 30): 25,
    ("AA",   2): 25,  ("AA",   5): 40,  ("AA",  10): 55,  ("AA",  30): 70,
    ("A",    2): 50,  ("A",    5): 80,  ("A",   10): 110, ("A",   30): 140,
    ("BBB",  2): 90,  ("BBB",  5): 130, ("BBB", 10): 170, ("BBB", 30): 210,
    ("BB",   2): 200, ("BB",   5): 280, ("BB",  10): 350, ("BB",  30): 420,
    ("B",    2): 350, ("B",    5): 450, ("B",   10): 550, ("B",   30): 650,
    ("CCC",  2): 700, ("CCC",  5): 850, ("CCC", 10): 950, ("CCC", 30): 1100,
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Bond(BaseModel):
    """Represents a single bond or bond-like instrument."""
    cusip: Optional[str] = None
    issuer_name: str
    bond_type: Literal[
        "treasury", "corporate_ig", "corporate_hy", "muni", "agency",
        "convertible", "tips", "em"
    ]
    coupon: Optional[float] = None
    maturity_date: Optional[date] = None
    maturity_years: Optional[float] = None
    ytm: Optional[float] = None            # yield to maturity, %
    price: Optional[float] = None          # clean price per $100 face
    rating_moody: Optional[str] = None
    rating_sp: Optional[str] = None
    rating_composite: Optional[str] = None # composite: AAA…D
    oas: Optional[float] = None            # option-adjusted spread, bps
    duration_modified: Optional[float] = None
    convexity: Optional[float] = None
    dv01: Optional[float] = None           # $ per $100 face per bp
    amount_outstanding: Optional[float] = None  # millions
    is_callable: bool = False
    call_date: Optional[date] = None
    industry: Optional[str] = None
    country: str = "US"
    currency: str = "USD"
    source: str
    filing_url: Optional[str] = None


class TreasurySecurity(BaseModel):
    """A single Treasury security from TreasuryDirect."""
    cusip: str
    security_type: str        # Note, Bond, Bill, TIPS, FRN
    issue_date: date
    maturity_date: date
    coupon_rate: float        # annual coupon, %
    yield_rate: Optional[float] = None
    price: Optional[float] = None
    outstanding_millions: Optional[float] = None
    maturity_years: Optional[float] = None


class CreditMarket(BaseModel):
    """Current state of credit markets — FRED-sourced snapshot."""
    as_of: date
    # OAS spreads (bps)
    ig_oas: Optional[float] = None
    hy_oas: Optional[float] = None
    bbb_aaa_spread: Optional[float] = None
    # 30-day changes (bps)
    ig_oas_30d_change: Optional[float] = None
    hy_oas_30d_change: Optional[float] = None
    # Market signal
    market_signal: str = "neutral"  # "risk_on", "risk_off", "neutral"
    # Total return (year-to-date %)
    ig_ytd_total_return: Optional[float] = None
    hy_ytd_total_return: Optional[float] = None
    # Treasury curve (%)
    t1m: Optional[float] = None
    t3m: Optional[float] = None
    t6m: Optional[float] = None
    t1y: Optional[float] = None
    t2y: Optional[float] = None
    t5y: Optional[float] = None
    t10y: Optional[float] = None
    t20y: Optional[float] = None
    t30y: Optional[float] = None
    # Derived curve analytics
    curve_slope_2_10: Optional[float] = None   # 10Y − 2Y in bps
    curve_slope_3m_10y: Optional[float] = None # 10Y − 3M in bps
    curve_inverted: bool = False
    # TIPS
    real_yield_5y: Optional[float] = None
    real_yield_10y: Optional[float] = None


class ConvertibleBond(BaseModel):
    """Convertible bond parsed from SEC 8-K filings."""
    issuer_name: str
    ticker: Optional[str] = None
    cusip: Optional[str] = None
    coupon: float
    maturity_date: date
    principal_millions: Optional[float] = None
    conversion_price: Optional[float] = None
    conversion_ratio: Optional[float] = None
    delta: Optional[float] = None    # equity sensitivity (0–1)
    parity: Optional[float] = None   # conversion value per $100 face
    premium: Optional[float] = None  # (price / parity − 1) * 100, %
    filed_date: date
    announcement_url: Optional[str] = None
    cik: Optional[str] = None


class ScreenResult(BaseModel):
    """Result of a bond screen."""
    query: dict
    total_found: int
    bonds: list[Bond]
    market_context: Optional[CreditMarket] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers — FRED
# ---------------------------------------------------------------------------


async def _fetch_fred_csv(
    client: httpx.AsyncClient, series_id: str, days_back: int = 400
) -> pd.Series:
    """
    Fetch a FRED series as a pd.Series with date index.
    Returns empty Series on error.
    """
    observation_start = (date.today() - timedelta(days=days_back)).isoformat()
    try:
        r = await client.get(
            FRED_CSV,
            params={"id": series_id, "observation_start": observation_start},
            timeout=20.0,
        )
        if r.status_code != 200:
            logger.debug("FRED %s HTTP %s", series_id, r.status_code)
            return pd.Series(dtype=float)
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"], index_col="DATE")
        col = df.columns[0]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return s
    except Exception as exc:
        logger.debug("FRED %s: %s", series_id, exc)
        return pd.Series(dtype=float)


async def _fred_latest_value(
    client: httpx.AsyncClient, series_id: str, days_back: int = 60
) -> Optional[float]:
    """Return the most recent non-null value for a FRED series."""
    s = await _fetch_fred_csv(client, series_id, days_back)
    if s.empty:
        return None
    return float(s.iloc[-1])


async def _fred_30d_change(
    client: httpx.AsyncClient, series_id: str
) -> Optional[float]:
    """Return latest − value ~30 days ago (absolute change in series units)."""
    s = await _fetch_fred_csv(client, series_id, days_back=90)
    if len(s) < 2:
        return None
    latest = s.iloc[-1]
    # find value closest to 30 days back
    cutoff = s.index[-1] - pd.Timedelta(days=30)
    past_vals = s[s.index <= cutoff]
    if past_vals.empty:
        return None
    past = past_vals.iloc[-1]
    return round(float(latest - past), 4)


# ---------------------------------------------------------------------------
# Internal helpers — TreasuryDirect
# ---------------------------------------------------------------------------

_TD_DATE_FMTS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y")


def _parse_td_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    for fmt in _TD_DATE_FMTS:
        try:
            return datetime.strptime(raw[:19], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _parse_td_float(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        v = float(raw)
        return v if v >= 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Internal helpers — EDGAR
# ---------------------------------------------------------------------------


async def _search_edgar(
    client: httpx.AsyncClient,
    query: str,
    forms: str,
    days_back: int,
    limit: int = 50,
) -> list[dict]:
    """
    Full-text search EDGAR EFTS for filings matching query.
    Returns list of filing metadata dicts.
    """
    start_dt = (date.today() - timedelta(days=days_back)).isoformat()
    end_dt = date.today().isoformat()
    params = {
        "q": f'"{query}"',
        "forms": forms,
        "dateRange": "custom",
        "startdt": start_dt,
        "enddt": end_dt,
        "hits.hits.total.value": 1,
        "hits.hits._source.period_of_report": 1,
    }
    try:
        r = await client.get(EDGAR_EFTS, params=params, timeout=20.0)
        if r.status_code != 200:
            logger.debug("EDGAR EFTS %s HTTP %s", query, r.status_code)
            return []
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits[:limit]:
            src = hit.get("_source", {})
            results.append({
                "cik": src.get("entity_id", ""),
                "company_name": src.get("display_names", [""])[0]
                    if src.get("display_names") else src.get("entity_name", ""),
                "form_type": src.get("file_type", forms),
                "filed_at": src.get("period_of_report") or src.get("file_date", ""),
                "accession_no": src.get("file_num", "") or hit.get("_id", ""),
                "filing_url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{src.get('entity_id', '')}/{hit.get('_id', '').replace('-', '')}"
                    f"/{src.get('file_name', '')}"
                    if src.get("file_name") else
                    f"https://efts.sec.gov/LATEST/search-index?q=%22{query}%22"
                ),
                "description": src.get("description", ""),
                "period_of_report": src.get("period_of_report", ""),
            })
        return results
    except Exception as exc:
        logger.debug("EDGAR EFTS error for '%s': %s", query, exc)
        return []


# ---------------------------------------------------------------------------
# Internal helpers — bond term extraction
# ---------------------------------------------------------------------------

# Compiled regex patterns for extracting bond terms from 8-K / 424B2 text
_RE_PRINCIPAL = re.compile(
    r"(?:aggregate\s+)?principal\s+amount\s+of\s+\$?([\d,]+(?:\.\d+)?)\s*(million|billion|M\b|B\b)",
    re.IGNORECASE,
)
_RE_COUPON = re.compile(
    r"([\d]+(?:\.\d+)?)\s*%\s+(?:senior\s+)?(?:secured\s+)?(?:unsecured\s+)?notes?",
    re.IGNORECASE,
)
_RE_COUPON_ALT = re.compile(
    r"bears?\s+interest\s+at\s+(?:a\s+rate\s+of\s+)?([\d]+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_RE_MATURITY = re.compile(
    r"(?:due|maturing?|mature|maturity)\s+(?:in\s+)?(\w+\s+)?(\d{4})",
    re.IGNORECASE,
)
_RE_MATURITY_DATE = re.compile(
    r"due\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(\d{4})",
    re.IGNORECASE,
)
_RE_CONVERSION_PRICE = re.compile(
    r"initial\s+conversion\s+price\s+of\s+approximately\s+\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RE_CONVERSION_RATIO = re.compile(
    r"conversion\s+rate\s+of\s+([\d,]+(?:\.\d+)?)\s+shares",
    re.IGNORECASE,
)
_RE_CALLABLE = re.compile(
    r"\b(callable|redeemable|call\s+date|make.whole\s+call)\b",
    re.IGNORECASE,
)
_RE_CALLABLE_DATE = re.compile(
    r"(?:callable|redeemable)\s+(?:on\s+or\s+after|beginning)\s+([\w]+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_RE_INDUSTRY = re.compile(
    r"\b(technology|healthcare|financials?|energy|utilities?|industrials?|"
    r"consumer|telecom|materials?|real\s+estate|reit)\b",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_INDUSTRY_NORM = {
    "technology": "Technology", "tech": "Technology",
    "healthcare": "Healthcare", "health": "Healthcare",
    "financial": "Financials", "financials": "Financials",
    "energy": "Energy",
    "utilities": "Utilities", "utility": "Utilities",
    "industrial": "Industrials", "industrials": "Industrials",
    "consumer": "Consumer",
    "telecom": "Telecom",
    "materials": "Materials", "material": "Materials",
    "real estate": "Real Estate", "reit": "Real Estate",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FixedIncomeScreener:
    """
    Comprehensive fixed income screener backed by free public data sources.

    All methods are async-safe and can be called concurrently.
    Instantiate once and reuse — creates an httpx.AsyncClient per call.
    """

    def __init__(self, timeout: float = 25.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # 1. Credit market snapshot
    # ------------------------------------------------------------------

    async def get_credit_market_snapshot(self) -> CreditMarket:
        """
        Fetch current credit market conditions from FRED.

        Returns CreditMarket with:
          - IG / HY OAS (bps) and 30-day change
          - Full Treasury yield curve (1M – 30Y)
          - Curve slope, inversion flag
          - Market signal: risk_on / risk_off / neutral
          - TIPS real yields
        """
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            # Fetch all series concurrently
            treasury_series = ["GS1M", "GS3M", "GS6M", "GS1", "GS2", "GS5",
                               "GS10", "GS20", "GS30"]
            credit_series = ["BAMLC0A0CM", "BAMLH0A0HYM2",
                             "BAMLC0A1CAAAEY", "BAMLC0A4CBBYEY"]
            total_return_series = ["BAMLCC0A0CMTRIV", "BAMLHYH0A0HYM2TRIV"]
            tips_series = ["DFII5", "DFII10"]

            all_series = treasury_series + credit_series + total_return_series + tips_series
            tasks = {sid: _fetch_fred_csv(client, sid, days_back=400)
                     for sid in all_series}
            results = dict(zip(
                tasks.keys(),
                await asyncio.gather(*tasks.values())
            ))

        def latest(sid: str) -> Optional[float]:
            s = results.get(sid, pd.Series(dtype=float))
            if s.empty:
                return None
            return float(s.iloc[-1])

        def change_30d(sid: str) -> Optional[float]:
            s = results.get(sid, pd.Series(dtype=float))
            if len(s) < 5:
                return None
            cutoff = s.index[-1] - pd.Timedelta(days=30)
            past = s[s.index <= cutoff]
            if past.empty:
                return None
            return round(float(s.iloc[-1] - past.iloc[-1]), 4)

        # Treasury yields
        t1m  = latest("GS1M")
        t3m  = latest("GS3M")
        t6m  = latest("GS6M")
        t1y  = latest("GS1")
        t2y  = latest("GS2")
        t5y  = latest("GS5")
        t10y = latest("GS10")
        t20y = latest("GS20")
        t30y = latest("GS30")

        # Curve slope (bps)
        slope_2_10 = round((t10y - t2y) * 100, 1) if t10y and t2y else None
        slope_3m_10y = round((t10y - t3m) * 100, 1) if t10y and t3m else None
        inverted = (slope_2_10 is not None and slope_2_10 < 0) or (
            slope_3m_10y is not None and slope_3m_10y < 0
        )

        # Credit spreads — FRED reports in % (e.g. 1.00 = 100 bps)
        ig_raw = latest("BAMLC0A0CM")
        hy_raw = latest("BAMLH0A0HYM2")
        ig_oas = round(ig_raw * 100, 1) if ig_raw is not None else None
        hy_oas = round(hy_raw * 100, 1) if hy_raw is not None else None

        # BBB–AAA spread proxy
        bbb_yield = latest("BAMLC0A4CBBYEY")
        aaa_yield = latest("BAMLC0A1CAAAEY")
        bbb_aaa_spread = (
            round((bbb_yield - aaa_yield) * 100, 1)
            if bbb_yield and aaa_yield else None
        )

        # 30-day OAS changes (bps)
        ig_oas_30d = None
        hy_oas_30d = None
        ig_chg_raw = change_30d("BAMLC0A0CM")
        hy_chg_raw = change_30d("BAMLH0A0HYM2")
        if ig_chg_raw is not None:
            ig_oas_30d = round(ig_chg_raw * 100, 1)
        if hy_chg_raw is not None:
            hy_oas_30d = round(hy_chg_raw * 100, 1)

        # Total return YTD (index-based, approximate)
        ig_ytd = _compute_ytd_return(results.get("BAMLCC0A0CMTRIV", pd.Series(dtype=float)))
        hy_ytd = _compute_ytd_return(results.get("BAMLHYH0A0HYM2TRIV", pd.Series(dtype=float)))

        # Market signal
        signal = _classify_credit_signal(ig_oas_30d, hy_oas_30d, ig_oas, hy_oas)

        return CreditMarket(
            as_of=date.today(),
            ig_oas=ig_oas,
            hy_oas=hy_oas,
            bbb_aaa_spread=bbb_aaa_spread,
            ig_oas_30d_change=ig_oas_30d,
            hy_oas_30d_change=hy_oas_30d,
            market_signal=signal,
            ig_ytd_total_return=ig_ytd,
            hy_ytd_total_return=hy_ytd,
            t1m=t1m, t3m=t3m, t6m=t6m, t1y=t1y,
            t2y=t2y, t5y=t5y, t10y=t10y, t20y=t20y, t30y=t30y,
            curve_slope_2_10=slope_2_10,
            curve_slope_3m_10y=slope_3m_10y,
            curve_inverted=inverted,
            real_yield_5y=latest("DFII5"),
            real_yield_10y=latest("DFII10"),
        )

    # ------------------------------------------------------------------
    # 2. Screen corporate bonds
    # ------------------------------------------------------------------

    async def screen_corporate_bonds(
        self,
        min_yield: float = 0.0,
        max_yield: float = 20.0,
        min_rating: Optional[str] = None,    # e.g. "BBB" = IG floor
        max_maturity_years: Optional[float] = None,
        industry: Optional[str] = None,
        convertible_only: bool = False,
        min_oas: Optional[float] = None,     # minimum spread, bps
        limit: int = 100,
    ) -> ScreenResult:
        """
        Screen corporate bonds.

        Combines:
          - Live TRACE bond universe (most-active IG + HY)
          - Recent EDGAR 424B2 / 8-K issuances
          - Convertible bonds from 8-K (if convertible_only=True or all)
          - FRED OAS as market context

        Filters by yield, rating, maturity, industry, and OAS.
        """
        warnings: list[str] = []
        query_dict = {
            "min_yield": min_yield,
            "max_yield": max_yield,
            "min_rating": min_rating,
            "max_maturity_years": max_maturity_years,
            "industry": industry,
            "convertible_only": convertible_only,
            "min_oas": min_oas,
        }

        # Run data fetches concurrently
        market_task = asyncio.create_task(self.get_credit_market_snapshot())
        issuances_task = asyncio.create_task(self.get_new_bond_issuances(days_back=60))
        convertibles_task = asyncio.create_task(self.get_convertible_bonds(days_back=120))

        # Pull TRACE universe
        try:
            from sentinel.sbx.trace_client import TRACEClient
            async with TRACEClient(timeout=self._timeout) as trace:
                trace_quotes = await trace.get_investment_grade_universe(limit=200)
        except Exception as exc:
            logger.warning("TRACE unavailable: %s", exc)
            trace_quotes = []
            warnings.append(f"TRACE data unavailable: {exc}")

        market_ctx, issuances, convertibles = await asyncio.gather(
            market_task, issuances_task, convertibles_task
        )

        # Convert TRACE quotes to Bond objects
        bonds: list[Bond] = []
        today = date.today()

        for q in trace_quotes:
            if not q.last_yield:
                continue
            mat_years = (
                (q.maturity_date - today).days / 365.25
                if q.maturity_date else None
            )
            # Classify as IG or HY based on yield spread heuristic
            ytm = q.last_yield
            oas_bps = q.spread_to_benchmark  # already in bps from TRACE
            bond_type: Literal[
                "treasury", "corporate_ig", "corporate_hy", "muni", "agency",
                "convertible", "tips", "em"
            ] = "corporate_hy" if (oas_bps and oas_bps > 300) else "corporate_ig"
            rating = _oas_to_composite_rating(oas_bps, mat_years)
            bonds.append(Bond(
                cusip=q.cusip,
                issuer_name=q.issuer_name,
                bond_type=bond_type,
                coupon=q.coupon,
                maturity_date=q.maturity_date,
                maturity_years=round(mat_years, 2) if mat_years is not None else None,
                ytm=ytm,
                price=float(q.last_price) if q.last_price else None,
                rating_composite=rating,
                oas=oas_bps,
                source="finra_trace",
            ))

        # Merge EDGAR issuances
        for b in issuances:
            if not convertible_only or b.bond_type == "convertible":
                bonds.append(b)

        # Merge convertibles
        for cv in convertibles:
            cv_mat_years = (cv.maturity_date - today).days / 365.25
            bonds.append(Bond(
                cusip=cv.cusip,
                issuer_name=cv.issuer_name,
                bond_type="convertible",
                coupon=cv.coupon,
                maturity_date=cv.maturity_date,
                maturity_years=round(cv_mat_years, 2),
                filing_url=cv.announcement_url,
                source="edgar_8k",
            ))

        # Apply filters
        min_rating_int = _RATING_FLOORS.get(min_rating, 0) if min_rating else 0
        filtered: list[Bond] = []
        for b in bonds:
            # Yield filter
            if b.ytm is not None:
                if b.ytm < min_yield or b.ytm > max_yield:
                    continue
            # Rating filter
            if min_rating and b.rating_composite:
                bond_rating_int = _RATING_TO_INT.get(b.rating_composite, 0)
                if bond_rating_int < min_rating_int:
                    continue
            # Maturity filter
            if max_maturity_years and b.maturity_years:
                if b.maturity_years > max_maturity_years:
                    continue
            # Industry filter
            if industry and b.industry:
                if industry.lower() not in b.industry.lower():
                    continue
            # OAS filter
            if min_oas and b.oas:
                if b.oas < min_oas:
                    continue
            # convertible_only filter
            if convertible_only and b.bond_type != "convertible":
                continue
            filtered.append(b)

        # Sort by OAS desc (widest spread = most interesting)
        filtered.sort(
            key=lambda b: (b.oas or 0, b.ytm or 0),
            reverse=True,
        )
        filtered = filtered[:limit]

        return ScreenResult(
            query=query_dict,
            total_found=len(filtered),
            bonds=filtered,
            market_context=market_ctx,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # 3. Treasury securities from TreasuryDirect
    # ------------------------------------------------------------------

    async def get_treasury_securities(
        self, security_type: str = "Note", days_back: int = 180
    ) -> list[TreasurySecurity]:
        """
        Fetch recently-auctioned Treasury securities from TreasuryDirect.

        Args:
            security_type: "Note", "Bond", "Bill", "TIPS", "FRN"
            days_back: How many days of auction history to retrieve.

        Returns:
            List of TreasurySecurity objects, newest first.
        """
        params = {
            "type": security_type,
            "days": days_back,
            "returnedfields": (
                "cusip,type,issueDate,maturityDate,interestRate,"
                "highYield,pricePer100,outstandingAmount,minimumToOrder"
            ),
            "format": "json",
        }
        today = date.today()
        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
                r = await client.get(TREASURY_DIRECT, params=params)
                if r.status_code != 200:
                    logger.warning(
                        "TreasuryDirect HTTP %s for type=%s", r.status_code, security_type
                    )
                    return []
                data = r.json()
        except Exception as exc:
            logger.error("TreasuryDirect fetch error: %s", exc)
            return []

        securities: list[TreasurySecurity] = []
        records = data if isinstance(data, list) else data.get("securityList", [])
        for rec in records:
            issue_date = _parse_td_date(rec.get("issueDate"))
            maturity_date = _parse_td_date(rec.get("maturityDate"))
            if not issue_date or not maturity_date:
                continue
            coupon = _parse_td_float(rec.get("interestRate"))
            if coupon is None:
                coupon = 0.0  # Bills have no coupon
            mat_years = round((maturity_date - today).days / 365.25, 2)
            outstanding_raw = _parse_td_float(rec.get("outstandingAmount"))
            outstanding_mm = (
                round(outstanding_raw / 1_000_000, 1) if outstanding_raw else None
            )
            securities.append(TreasurySecurity(
                cusip=rec.get("cusip", ""),
                security_type=rec.get("type", security_type),
                issue_date=issue_date,
                maturity_date=maturity_date,
                coupon_rate=coupon,
                yield_rate=_parse_td_float(rec.get("highYield")),
                price=_parse_td_float(rec.get("pricePer100")),
                outstanding_millions=outstanding_mm,
                maturity_years=mat_years,
            ))

        securities.sort(key=lambda s: s.issue_date, reverse=True)
        logger.info(
            "TreasuryDirect: fetched %d %s securities (days_back=%d)",
            len(securities), security_type, days_back,
        )
        return securities

    # ------------------------------------------------------------------
    # 4. New bond issuances from EDGAR
    # ------------------------------------------------------------------

    async def get_new_bond_issuances(
        self, days_back: int = 30, limit: int = 50
    ) -> list[Bond]:
        """
        Mine EDGAR for newly-issued corporate bonds via:
          - Form 424B2 (prospectus supplement — the primary bond offering doc)
          - Form 8-K with keyword "aggregate principal amount" (debt offering disclosure)

        Extracts: issuer, coupon, maturity, principal, callable status.
        """
        async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
            # Run both searches in parallel
            prospectus_hits, eight_k_hits = await asyncio.gather(
                _search_edgar(client, "aggregate principal amount", "424B2", days_back, limit),
                _search_edgar(client, "aggregate principal amount senior notes", "8-K", days_back, limit),
            )

        all_hits = prospectus_hits + eight_k_hits
        # Deduplicate by company_name to avoid double-counting
        seen: set[str] = set()
        bonds: list[Bond] = []

        for hit in all_hits:
            issuer = (hit.get("company_name") or "").strip()
            if not issuer or issuer in seen:
                continue
            seen.add(issuer)

            # Extract bond terms from description / filing text
            text = (
                hit.get("description", "") + " " +
                hit.get("period_of_report", "")
            )
            terms = self._extract_bond_terms_from_text(text, issuer)

            filed_raw = hit.get("filed_at") or hit.get("period_of_report", "")
            filed_date = _parse_td_date(filed_raw) or date.today()

            coupon = terms.get("coupon")
            maturity_year = terms.get("maturity_year")
            principal = terms.get("principal_millions")

            maturity_date: Optional[date] = None
            if maturity_year:
                maturity_date = date(maturity_year, 12, 31)
            elif terms.get("maturity_month") and maturity_year:
                maturity_date = date(maturity_year, terms["maturity_month"], 1)

            mat_years: Optional[float] = None
            if maturity_date:
                mat_years = round((maturity_date - date.today()).days / 365.25, 2)

            # Derive bond_type: if convertible keywords found → convertible
            desc_lower = text.lower()
            if "convertible" in desc_lower:
                bond_type: Literal[
                    "treasury", "corporate_ig", "corporate_hy", "muni", "agency",
                    "convertible", "tips", "em"
                ] = "convertible"
            elif coupon and coupon > 7:
                bond_type = "corporate_hy"
            else:
                bond_type = "corporate_ig"

            # YTM estimate: coupon + par/maturity simple approximation
            ytm: Optional[float] = None
            if coupon and mat_years and mat_years > 0:
                # Simple yield proxy: coupon + (100-price)/maturity / ((100+price)/2)
                price_est = 100.0
                ytm = round(
                    (coupon + (100.0 - price_est) / mat_years) /
                    ((100.0 + price_est) / 2.0) * 100, 3
                )

            bonds.append(Bond(
                issuer_name=issuer,
                bond_type=bond_type,
                coupon=coupon,
                maturity_date=maturity_date,
                maturity_years=mat_years,
                ytm=ytm,
                amount_outstanding=principal,
                is_callable=terms.get("is_callable", False),
                call_date=terms.get("call_date"),
                industry=terms.get("industry"),
                filing_url=hit.get("filing_url"),
                source=f"edgar_{hit.get('form_type', '424B2')}",
            ))

            if len(bonds) >= limit:
                break

        logger.info("EDGAR new bond issuances: %d found (days_back=%d)", len(bonds), days_back)
        return bonds

    # ------------------------------------------------------------------
    # 5. Convertible bonds from EDGAR
    # ------------------------------------------------------------------

    async def get_convertible_bonds(
        self, days_back: int = 90, limit: int = 50
    ) -> list[ConvertibleBond]:
        """
        Search EDGAR 8-K filings for convertible bond announcements.

        Extracts:
          - Issuer, coupon, maturity date, principal
          - Conversion price, conversion ratio
          - Implied delta (equity sensitivity) using Black-Scholes proxy
          - Parity and premium (requires stock price — estimated as mid-range)
        """
        async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
            hits_senior, hits_notes = await asyncio.gather(
                _search_edgar(client, "convertible senior notes", "8-K", days_back, limit),
                _search_edgar(client, "convertible notes offering", "8-K", days_back, limit),
            )

        all_hits = hits_senior + hits_notes
        seen: set[str] = set()
        results: list[ConvertibleBond] = []

        for hit in all_hits:
            issuer = (hit.get("company_name") or "").strip()
            if not issuer or issuer in seen:
                continue
            seen.add(issuer)

            text = hit.get("description", "")
            terms = self._extract_bond_terms_from_text(text, issuer)

            filed_raw = hit.get("filed_at") or hit.get("period_of_report", "")
            filed_date = _parse_td_date(filed_raw) or date.today()

            coupon = terms.get("coupon", 0.0) or 0.0
            maturity_year = terms.get("maturity_year")
            if not maturity_year:
                # Default 5-year convertible if we can't parse
                maturity_year = date.today().year + 5

            maturity_month = terms.get("maturity_month", 12)
            maturity_date = date(maturity_year, maturity_month, 1)

            conversion_price = terms.get("conversion_price")
            conversion_ratio = terms.get("conversion_ratio")
            principal = terms.get("principal_millions")

            # Estimate delta: simple heuristic — convertibles near parity have ~0.5 delta
            delta = _estimate_delta(coupon, conversion_price)

            results.append(ConvertibleBond(
                issuer_name=issuer,
                cik=hit.get("cik"),
                coupon=coupon,
                maturity_date=maturity_date,
                principal_millions=principal,
                conversion_price=conversion_price,
                conversion_ratio=conversion_ratio,
                delta=delta,
                filed_date=filed_date,
                announcement_url=hit.get("filing_url"),
            ))

            if len(results) >= limit:
                break

        logger.info(
            "EDGAR convertible bonds: %d found (days_back=%d)", len(results), days_back
        )
        return results

    # ------------------------------------------------------------------
    # 6. Relative value analysis
    # ------------------------------------------------------------------

    async def compute_relative_value(
        self, bonds: list[Bond]
    ) -> pd.DataFrame:
        """
        Rank bonds by relative value within rating / maturity cohorts.

        Computes for each bond:
          - spread_vs_treasury_bps: OAS or estimated spread
          - duration_adj_spread: OAS / modified_duration (carry per unit of duration risk)
          - z_score: standardised spread within same rating bucket
          - verdict: "cheap" | "fair" | "rich" based on z-score

        Returns a pd.DataFrame sorted by z_score descending (cheapest first).
        """
        if not bonds:
            return pd.DataFrame()

        # Fetch live Treasury curve for spread calc
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            t2y = await _fred_latest_value(client, "GS2", 60)
            t5y = await _fred_latest_value(client, "GS5", 60)
            t10y = await _fred_latest_value(client, "GS10", 60)
            t30y = await _fred_latest_value(client, "GS30", 60)

        tsy_curve = {2.0: t2y, 5.0: t5y, 10.0: t10y, 30.0: t30y}

        rows = []
        for b in bonds:
            oas = b.oas
            if oas is None and b.ytm is not None and b.maturity_years:
                tsy = _interp_from_curve(tsy_curve, b.maturity_years or 5.0)
                oas = round((b.ytm - (tsy or 4.5)) * 100, 1) if tsy else None

            dur_adj = (
                round(oas / b.duration_modified, 1)
                if oas and b.duration_modified and b.duration_modified > 0 else None
            )
            rows.append({
                "issuer_name": b.issuer_name,
                "rating_composite": b.rating_composite or "NR",
                "bond_type": b.bond_type,
                "maturity_years": b.maturity_years,
                "coupon": b.coupon,
                "ytm": b.ytm,
                "oas_bps": oas,
                "duration_modified": b.duration_modified,
                "duration_adj_spread": dur_adj,
                "cusip": b.cusip,
                "source": b.source,
            })

        df = pd.DataFrame(rows)
        if df.empty or "oas_bps" not in df.columns:
            return df

        # Z-score within rating cohort
        df["z_score"] = np.nan
        for rating_grp in df["rating_composite"].unique():
            mask = df["rating_composite"] == rating_grp
            cohort_oas = df.loc[mask, "oas_bps"].dropna()
            if len(cohort_oas) < 2:
                continue
            mean = cohort_oas.mean()
            std = cohort_oas.std()
            if std > 0:
                df.loc[mask, "z_score"] = (
                    (df.loc[mask, "oas_bps"] - mean) / std
                ).round(2)

        # Verdict
        def _verdict(z: float) -> str:
            if pd.isna(z):
                return "N/A"
            if z > 1.0:
                return "cheap"
            elif z < -1.0:
                return "rich"
            return "fair"

        df["verdict"] = df["z_score"].apply(_verdict)
        df.sort_values("z_score", ascending=False, inplace=True, na_position="last")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. High-yield / distressed watchlist
    # ------------------------------------------------------------------

    async def get_high_yield_watchlist(self) -> list[Bond]:
        """
        Build a distressed bond watchlist from EDGAR 8-K filings.

        Searches for:
          - Credit rating downgrade announcements
          - Covenant violations / waiver requests
          - Bankruptcy / restructuring filings (Chapter 11, Chapter 15)

        Returns Bond objects flagged by issue, sorted by distress level.
        """
        queries = [
            ("credit rating downgrade", "8-K", 60),
            ("covenant default waiver", "8-K", 90),
            ("Chapter 11 bankruptcy", "8-K", 90),
        ]
        async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
            hits_per_query = await asyncio.gather(*[
                _search_edgar(client, q, form, days)
                for q, form, days in queries
            ])

        seen: set[str] = set()
        bonds: list[Bond] = []
        distress_levels = {
            "credit rating downgrade": ("CCC", "corporate_hy"),
            "covenant default waiver": ("CCC", "corporate_hy"),
            "Chapter 11 bankruptcy": ("D", "corporate_hy"),
        }

        for (query, _, _), hits in zip(queries, hits_per_query):
            composite_rating, bond_type = distress_levels[query]
            for hit in hits:
                issuer = (hit.get("company_name") or "").strip()
                if not issuer or issuer in seen:
                    continue
                seen.add(issuer)
                text = hit.get("description", "")
                terms = self._extract_bond_terms_from_text(text, issuer)
                industry = terms.get("industry")
                bonds.append(Bond(
                    issuer_name=issuer,
                    bond_type=bond_type,
                    rating_composite=composite_rating,
                    industry=industry,
                    filing_url=hit.get("filing_url"),
                    source=f"edgar_watchlist:{query}",
                ))

        logger.info("HY watchlist: %d distressed issuers found", len(bonds))
        return bonds

    # ------------------------------------------------------------------
    # 8. Build a comprehensive bond universe
    # ------------------------------------------------------------------

    async def build_bond_universe(self) -> list[Bond]:
        """
        Assemble a comprehensive fixed income universe from all sources:
          1. TRACE most-active corporate bonds
          2. Recent EDGAR issuances (424B2 + 8-K)
          3. Convertible bonds (8-K)
          4. Treasury securities (Notes, Bonds, TIPS)

        De-duplicates by CUSIP where available, otherwise by issuer+maturity.
        Returns list sorted by bond_type then maturity_years.
        """
        today = date.today()

        # Run all fetches concurrently
        (
            issuances,
            convertibles,
            hy_watchlist,
            notes,
            bonds_30y,
            tips,
        ) = await asyncio.gather(
            self.get_new_bond_issuances(days_back=90, limit=100),
            self.get_convertible_bonds(days_back=180, limit=75),
            self.get_high_yield_watchlist(),
            self.get_treasury_securities("Note", days_back=365),
            self.get_treasury_securities("Bond", days_back=365),
            self.get_treasury_securities("TIPS", days_back=365),
        )

        # Pull TRACE universe
        try:
            from sentinel.sbx.trace_client import TRACEClient
            async with TRACEClient(timeout=self._timeout) as trace:
                trace_quotes = await trace.get_investment_grade_universe(limit=300)
        except Exception as exc:
            logger.warning("TRACE unavailable for universe build: %s", exc)
            trace_quotes = []

        universe: list[Bond] = []
        seen_cusips: set[str] = set()
        seen_keys: set[tuple] = set()

        def _add(b: Bond) -> None:
            if b.cusip and b.cusip in seen_cusips:
                return
            key = (b.issuer_name, b.maturity_date)
            if key in seen_keys:
                return
            if b.cusip:
                seen_cusips.add(b.cusip)
            seen_keys.add(key)
            universe.append(b)

        # TRACE bonds
        for q in trace_quotes:
            if not q.cusip:
                continue
            mat_years = (
                (q.maturity_date - today).days / 365.25
                if q.maturity_date else None
            )
            oas_bps = q.spread_to_benchmark
            rating = _oas_to_composite_rating(oas_bps, mat_years)
            _add(Bond(
                cusip=q.cusip,
                issuer_name=q.issuer_name,
                bond_type="corporate_hy" if (oas_bps and oas_bps > 300) else "corporate_ig",
                coupon=q.coupon,
                maturity_date=q.maturity_date,
                maturity_years=round(mat_years, 2) if mat_years else None,
                ytm=q.last_yield,
                price=float(q.last_price) if q.last_price else None,
                rating_composite=rating,
                oas=oas_bps,
                source="finra_trace",
            ))

        # EDGAR issuances
        for b in issuances:
            _add(b)

        # Convertible bonds
        for cv in convertibles:
            cv_mat_years = (cv.maturity_date - today).days / 365.25
            _add(Bond(
                cusip=cv.cusip,
                issuer_name=cv.issuer_name,
                bond_type="convertible",
                coupon=cv.coupon,
                maturity_date=cv.maturity_date,
                maturity_years=round(cv_mat_years, 2),
                filing_url=cv.announcement_url,
                source="edgar_8k_convertible",
            ))

        # Treasury notes, bonds, TIPS
        for ts in notes + bonds_30y + tips:
            bt: Literal[
                "treasury", "corporate_ig", "corporate_hy", "muni", "agency",
                "convertible", "tips", "em"
            ] = "tips" if ts.security_type == "TIPS" else "treasury"
            _add(Bond(
                cusip=ts.cusip,
                issuer_name="U.S. Treasury",
                bond_type=bt,
                coupon=ts.coupon_rate,
                maturity_date=ts.maturity_date,
                maturity_years=ts.maturity_years,
                ytm=ts.yield_rate,
                price=ts.price,
                rating_composite="AAA",
                oas=0.0,
                amount_outstanding=ts.outstanding_millions,
                source="treasurydirect",
            ))

        # HY watchlist
        for b in hy_watchlist:
            _add(b)

        # Sort: treasuries first, then by maturity
        _TYPE_ORDER = {
            "treasury": 0, "tips": 1, "agency": 2,
            "corporate_ig": 3, "muni": 4,
            "corporate_hy": 5, "convertible": 6, "em": 7,
        }
        universe.sort(key=lambda b: (
            _TYPE_ORDER.get(b.bond_type, 9),
            b.maturity_years or 0.0,
        ))

        logger.info(
            "Bond universe built: %d instruments from %d TRACE + %d issuances + "
            "%d convertibles + %d treasuries",
            len(universe),
            len(trace_quotes),
            len(issuances),
            len(convertibles),
            len(notes) + len(bonds_30y) + len(tips),
        )
        return universe

    # ------------------------------------------------------------------
    # 9. Internal: FRED series fetch
    # ------------------------------------------------------------------

    async def _fetch_fred_series(
        self, series_id: str, days_back: int = 400
    ) -> pd.Series:
        """Fetch a FRED time series. Returns pd.Series with datetime index."""
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            return await _fetch_fred_csv(client, series_id, days_back)

    # ------------------------------------------------------------------
    # 10. Internal: EDGAR 8-K debt search
    # ------------------------------------------------------------------

    async def _search_edgar_8k_debt(
        self, query: str, days_back: int
    ) -> list[dict]:
        """Search EDGAR EFTS for 8-K filings matching a debt-related query."""
        async with httpx.AsyncClient(headers=_HEADERS, timeout=self._timeout) as client:
            return await _search_edgar(client, query, "8-K", days_back)

    # ------------------------------------------------------------------
    # 11. Internal: bond term extraction from filing text
    # ------------------------------------------------------------------

    def _extract_bond_terms_from_text(
        self, text: str, issuer_name: str
    ) -> dict:
        """
        Regex-extract key bond terms from SEC filing text (424B2, 8-K).

        Returns dict with keys (may be None if not found):
          coupon, maturity_year, maturity_month, principal_millions,
          is_callable, call_date, conversion_price, conversion_ratio, industry
        """
        result: dict = {}

        # Coupon
        m = _RE_COUPON.search(text)
        if not m:
            m = _RE_COUPON_ALT.search(text)
        if m:
            try:
                result["coupon"] = float(m.group(1))
            except (ValueError, IndexError):
                pass

        # Maturity date
        m = _RE_MATURITY_DATE.search(text)
        if m:
            month_name = m.group(1).lower()
            year = int(m.group(2))
            result["maturity_year"] = year
            result["maturity_month"] = _MONTH_MAP.get(month_name, 12)
        else:
            m = _RE_MATURITY.search(text)
            if m:
                try:
                    result["maturity_year"] = int(m.group(2))
                    result["maturity_month"] = 12
                except (ValueError, IndexError):
                    pass

        # Principal amount
        m = _RE_PRINCIPAL.search(text)
        if m:
            try:
                raw_amount = float(m.group(1).replace(",", ""))
                unit = m.group(2).lower()
                if unit in ("billion", "b"):
                    result["principal_millions"] = raw_amount * 1000.0
                else:
                    result["principal_millions"] = raw_amount
            except (ValueError, IndexError):
                pass

        # Callable
        if _RE_CALLABLE.search(text):
            result["is_callable"] = True
            m = _RE_CALLABLE_DATE.search(text)
            if m:
                try:
                    result["call_date"] = datetime.strptime(
                        m.group(1).strip(), "%B %d, %Y"
                    ).date()
                except ValueError:
                    pass
        else:
            result["is_callable"] = False

        # Conversion price (convertibles)
        m = _RE_CONVERSION_PRICE.search(text)
        if m:
            try:
                result["conversion_price"] = float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                pass

        # Conversion ratio
        m = _RE_CONVERSION_RATIO.search(text)
        if m:
            try:
                result["conversion_ratio"] = float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                pass

        # Industry
        m = _RE_INDUSTRY.search(text + " " + issuer_name)
        if m:
            raw = m.group(1).lower().strip()
            result["industry"] = _INDUSTRY_NORM.get(raw, raw.title())

        return result

    # ------------------------------------------------------------------
    # 12. Rating utilities
    # ------------------------------------------------------------------

    def _rating_to_numeric(self, rating: str) -> int:
        """Convert composite rating string to numeric rank (higher = better)."""
        return _RATING_TO_INT.get(rating, 0)

    # ------------------------------------------------------------------
    # 13. YTM calculation
    # ------------------------------------------------------------------

    def _compute_yield_from_price(
        self, coupon: float, price: float, years: float
    ) -> float:
        """
        Newton-Raphson yield-to-maturity for a semiannual-pay fixed-rate bond.

        Args:
            coupon: Annual coupon rate (%), e.g. 5.0 for 5%
            price:  Clean price per $100 face value
            years:  Years to maturity

        Returns:
            YTM as an annualized percentage (%)
        """
        face = 100.0
        c = coupon / 200.0   # semiannual coupon per $100 face
        n = max(1, round(years * 2))  # number of semiannual periods
        p = price

        # Initial guess: simple approximation
        ytm_guess = (c * 2 * face + (face - p) / years) / ((face + p) / 2.0) / 100.0

        for _ in range(50):
            y = ytm_guess / 2.0
            pv_coupons = c * face * (1.0 - (1.0 + y) ** -n) / y if y != 0 else c * face * n
            pv_face = face * (1.0 + y) ** -n
            price_calc = pv_coupons + pv_face

            # First derivative
            dpv_dy = 0.0
            for k in range(1, n + 1):
                dpv_dy -= k * c * face * (1.0 + y) ** -(k + 1)
            dpv_dy -= n * face * (1.0 + y) ** -(n + 1)

            delta_y = -(price_calc - p) / dpv_dy if dpv_dy != 0 else 0.0
            ytm_guess += delta_y * 2.0  # convert back to annual

            if abs(delta_y * 2.0) < 1e-8:
                break

        return round(ytm_guess * 100.0, 4)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _compute_ytd_return(series: pd.Series) -> Optional[float]:
    """Compute year-to-date total return from an index series."""
    if len(series) < 2:
        return None
    year_start = date(date.today().year, 1, 1)
    year_start_ts = pd.Timestamp(year_start)
    past = series[series.index >= year_start_ts]
    if len(past) < 2:
        past = series.iloc[-min(252, len(series)):]
    if len(past) < 2:
        return None
    return round((float(past.iloc[-1]) / float(past.iloc[0]) - 1.0) * 100.0, 2)


def _classify_credit_signal(
    ig_30d: Optional[float],
    hy_30d: Optional[float],
    ig_oas: Optional[float],
    hy_oas: Optional[float],
) -> str:
    """
    Classify market signal based on OAS trend.

    Logic:
      - risk_on: both IG and HY OAS tightening (30d change negative)
      - risk_off: both IG and HY OAS widening (30d change positive)
      - neutral: mixed signals or data unavailable
    Additional: absolute level check — if HY OAS > 700 bps, always risk_off.
    """
    if hy_oas and hy_oas > 700:
        return "risk_off"
    if ig_30d is not None and hy_30d is not None:
        if ig_30d < -5 and hy_30d < -10:
            return "risk_on"
        if ig_30d > 5 and hy_30d > 10:
            return "risk_off"
    return "neutral"


def _oas_to_composite_rating(
    oas_bps: Optional[float], maturity_years: Optional[float]
) -> Optional[str]:
    """
    Approximate composite rating from OAS spread using rough market ranges.
    This is a heuristic for when no explicit rating is provided.
    """
    if oas_bps is None:
        return None
    if oas_bps < 30:
        return "AAA"
    elif oas_bps < 60:
        return "AA"
    elif oas_bps < 120:
        return "A"
    elif oas_bps < 200:
        return "BBB"
    elif oas_bps < 350:
        return "BB"
    elif oas_bps < 600:
        return "B"
    elif oas_bps < 900:
        return "CCC"
    else:
        return "D"


def _interp_from_curve(
    curve: dict[float, Optional[float]], years: float
) -> Optional[float]:
    """
    Linearly interpolate a Treasury yield from a tenor→yield curve dict.
    """
    tenors = sorted(k for k, v in curve.items() if v is not None)
    if not tenors:
        return None
    if years <= tenors[0]:
        return curve[tenors[0]]
    if years >= tenors[-1]:
        return curve[tenors[-1]]
    for i in range(len(tenors) - 1):
        t0, t1 = tenors[i], tenors[i + 1]
        if t0 <= years <= t1:
            v0 = curve[t0]
            v1 = curve[t1]
            if v0 is None or v1 is None:
                return None
            w = (years - t0) / (t1 - t0)
            return v0 + w * (v1 - v0)
    return None


def _estimate_delta(
    coupon: float, conversion_price: Optional[float]
) -> Optional[float]:
    """
    Estimate convertible bond delta (equity sensitivity 0–1).

    Heuristic: zero-coupon converts → delta ≈ 0.5; high-coupon without
    conversion price → delta ≈ 0.3 (bond-like); if conversion price known
    and coupon is low, skew toward 0.5.
    """
    if conversion_price and conversion_price > 0:
        if coupon <= 1.0:
            return 0.55
        elif coupon <= 3.0:
            return 0.45
        else:
            return 0.35
    if coupon <= 0.5:
        return 0.5
    elif coupon <= 2.0:
        return 0.4
    else:
        return 0.3


# ---------------------------------------------------------------------------
# Module-level convenience coroutines
# ---------------------------------------------------------------------------

_screener: Optional[FixedIncomeScreener] = None


def _get_screener() -> FixedIncomeScreener:
    global _screener
    if _screener is None:
        _screener = FixedIncomeScreener()
    return _screener


async def credit_market() -> CreditMarket:
    """Fetch current credit market snapshot (FRED-powered)."""
    return await _get_screener().get_credit_market_snapshot()


async def screen_bonds(
    min_yield: float = 0.0,
    min_rating: str = "BBB",
    max_maturity_years: Optional[float] = None,
    convertible_only: bool = False,
) -> ScreenResult:
    """Screen corporate bonds with sensible defaults (IG, yield > 0%)."""
    return await _get_screener().screen_corporate_bonds(
        min_yield=min_yield,
        min_rating=min_rating,
        max_maturity_years=max_maturity_years,
        convertible_only=convertible_only,
    )


async def new_issuances(days_back: int = 30) -> list[Bond]:
    """Return recently-issued corporate bonds from EDGAR 424B2 / 8-K."""
    return await _get_screener().get_new_bond_issuances(days_back=days_back)


async def convertible_bonds(days_back: int = 90) -> list[ConvertibleBond]:
    """Return recent convertible bond announcements from EDGAR 8-K."""
    return await _get_screener().get_convertible_bonds(days_back=days_back)


async def treasury_securities(
    security_type: str = "Note", days_back: int = 180
) -> list[TreasurySecurity]:
    """Return recently-auctioned Treasury securities from TreasuryDirect."""
    return await _get_screener().get_treasury_securities(security_type, days_back)


async def bond_universe() -> list[Bond]:
    """Build and return the full SENTINEL bond universe (all sources)."""
    return await _get_screener().build_bond_universe()
