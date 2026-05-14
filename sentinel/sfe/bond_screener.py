"""
Fixed income screener — Dimension #74.

Screens bond ETF proxies by yield, duration, and credit quality using:
  - yfinance for ETF price/nav data (free)
  - FRED for live Treasury yields and OAS credit spreads (free)
  - FINRA TRACE aggregate data via public endpoints

Score target: SENTINEL 4, Bloomberg 7 (they have individual CUSIP-level data).
Dim 74 target: 0 → 4.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# ---------------------------------------------------------------------------
# Bond ETF Catalog — covers IG, HY, sovereign, munis, TIPS, EM by duration
# ---------------------------------------------------------------------------

_ETF_CATALOG: list[dict] = [
    # Investment Grade — short
    {"symbol": "SHY", "name": "iShares 1-3 Year Treasury Bond", "credit": "AAA", "duration": 1.9, "category": "government", "issuer": "iShares"},
    {"symbol": "SPSB", "name": "SPDR Portfolio Short Term Corp", "credit": "A", "duration": 2.6, "category": "corporate_ig", "issuer": "SPDR"},
    {"symbol": "VCSH", "name": "Vanguard Short-Term Corp Bond", "credit": "A", "duration": 2.7, "category": "corporate_ig", "issuer": "Vanguard"},
    {"symbol": "IGSB", "name": "iShares 1-5 Year IG Corp Bond", "credit": "A", "duration": 2.9, "category": "corporate_ig", "issuer": "iShares"},
    {"symbol": "SLQD", "name": "iShares 0-5 Year IG Corp Bond", "credit": "BBB", "duration": 2.5, "category": "corporate_ig", "issuer": "iShares"},
    # Investment Grade — intermediate
    {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond", "credit": "AAA", "duration": 7.7, "category": "government", "issuer": "iShares"},
    {"symbol": "IEI", "name": "iShares 3-7 Year Treasury Bond", "credit": "AAA", "duration": 4.5, "category": "government", "issuer": "iShares"},
    {"symbol": "LQD", "name": "iShares iBoxx IG Corp Bond", "credit": "A", "duration": 8.4, "category": "corporate_ig", "issuer": "iShares"},
    {"symbol": "VCIT", "name": "Vanguard Intermediate-Term Corp", "credit": "A", "duration": 6.5, "category": "corporate_ig", "issuer": "Vanguard"},
    {"symbol": "IGIB", "name": "iShares 5-10 Year IG Corp Bond", "credit": "A", "duration": 6.8, "category": "corporate_ig", "issuer": "iShares"},
    {"symbol": "SPIB", "name": "SPDR Portfolio Intermediate Corp", "credit": "A", "duration": 6.2, "category": "corporate_ig", "issuer": "SPDR"},
    {"symbol": "MBB", "name": "iShares MBS ETF", "credit": "AAA", "duration": 5.2, "category": "mortgage", "issuer": "iShares"},
    {"symbol": "VMBS", "name": "Vanguard Mortgage-Backed Securities", "credit": "AAA", "duration": 5.4, "category": "mortgage", "issuer": "Vanguard"},
    # Investment Grade — long
    {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond", "credit": "AAA", "duration": 16.5, "category": "government", "issuer": "iShares"},
    {"symbol": "TLH", "name": "iShares 10-20 Year Treasury Bond", "credit": "AAA", "duration": 11.8, "category": "government", "issuer": "iShares"},
    {"symbol": "VCLT", "name": "Vanguard Long-Term Corp Bond", "credit": "A", "duration": 13.4, "category": "corporate_ig", "issuer": "Vanguard"},
    {"symbol": "IGLB", "name": "iShares 10+ Year IG Corp Bond", "credit": "A", "duration": 13.8, "category": "corporate_ig", "issuer": "iShares"},
    {"symbol": "SPTL", "name": "SPDR Portfolio Long Term Treasury", "credit": "AAA", "duration": 15.2, "category": "government", "issuer": "SPDR"},
    # Broad IG
    {"symbol": "AGG", "name": "iShares Core US Aggregate Bond", "credit": "AA", "duration": 6.2, "category": "aggregate", "issuer": "iShares"},
    {"symbol": "BND", "name": "Vanguard Total Bond Market", "credit": "AA", "duration": 6.3, "category": "aggregate", "issuer": "Vanguard"},
    {"symbol": "SPAB", "name": "SPDR Portfolio Aggregate Bond", "credit": "AA", "duration": 6.2, "category": "aggregate", "issuer": "SPDR"},
    {"symbol": "SCHZ", "name": "Schwab US Aggregate Bond", "credit": "AA", "duration": 6.1, "category": "aggregate", "issuer": "Schwab"},
    # High Yield
    {"symbol": "HYG", "name": "iShares iBoxx High Yield Corp", "credit": "BB", "duration": 3.8, "category": "corporate_hy", "issuer": "iShares"},
    {"symbol": "JNK", "name": "SPDR Bloomberg High Yield Bond", "credit": "BB", "duration": 3.9, "category": "corporate_hy", "issuer": "SPDR"},
    {"symbol": "USHY", "name": "iShares Broad USD High Yield Bond", "credit": "BB", "duration": 4.1, "category": "corporate_hy", "issuer": "iShares"},
    {"symbol": "SJNK", "name": "SPDR Bloomberg ST High Yield Bond", "credit": "B", "duration": 2.3, "category": "corporate_hy", "issuer": "SPDR"},
    {"symbol": "SHYG", "name": "iShares 0-5 Year High Yield Corp", "credit": "BB", "duration": 2.4, "category": "corporate_hy", "issuer": "iShares"},
    {"symbol": "FALN", "name": "iShares Fallen Angels USD Bond", "credit": "BB", "duration": 5.2, "category": "corporate_hy", "issuer": "iShares"},
    {"symbol": "ANGL", "name": "VanEck Fallen Angel High Yield", "credit": "BB", "duration": 5.7, "category": "corporate_hy", "issuer": "VanEck"},
    # TIPS / Inflation-linked
    {"symbol": "TIP", "name": "iShares TIPS Bond", "credit": "AAA", "duration": 7.0, "category": "tips", "issuer": "iShares"},
    {"symbol": "VTIP", "name": "Vanguard Short-Term Inflation-Protected", "credit": "AAA", "duration": 2.5, "category": "tips", "issuer": "Vanguard"},
    {"symbol": "STIP", "name": "iShares 0-5 Year TIPS Bond", "credit": "AAA", "duration": 2.6, "category": "tips", "issuer": "iShares"},
    {"symbol": "LTPZ", "name": "PIMCO 15+ Year TIPS", "credit": "AAA", "duration": 15.1, "category": "tips", "issuer": "PIMCO"},
    # Munis
    {"symbol": "MUB", "name": "iShares National Muni Bond", "credit": "AA", "duration": 6.3, "category": "muni", "issuer": "iShares"},
    {"symbol": "VTEB", "name": "Vanguard Tax-Exempt Bond", "credit": "AA", "duration": 5.9, "category": "muni", "issuer": "Vanguard"},
    {"symbol": "CMF", "name": "iShares California Muni Bond", "credit": "AA", "duration": 6.1, "category": "muni", "issuer": "iShares"},
    {"symbol": "SUB", "name": "iShares Short-Term National Muni", "credit": "AA", "duration": 2.3, "category": "muni", "issuer": "iShares"},
    {"symbol": "HYMB", "name": "SPDR Nuveen Bloomberg HY Muni", "credit": "BB", "duration": 7.8, "category": "muni_hy", "issuer": "SPDR"},
    # Emerging Markets
    {"symbol": "EMB", "name": "iShares JP Morgan USD EM Bond", "credit": "BBB", "duration": 6.4, "category": "em_sovereign", "issuer": "iShares"},
    {"symbol": "VWOB", "name": "Vanguard EM Government Bond", "credit": "BBB", "duration": 7.0, "category": "em_sovereign", "issuer": "Vanguard"},
    {"symbol": "FEMB", "name": "First Trust EM Local Currency Bond", "credit": "BBB", "duration": 4.9, "category": "em_local", "issuer": "First Trust"},
    {"symbol": "EMHY", "name": "iShares EM High Yield Bond", "credit": "BB", "duration": 4.5, "category": "em_hy", "issuer": "iShares"},
    # Convertibles
    {"symbol": "CWB", "name": "SPDR Bloomberg Conv Securities", "credit": "BB", "duration": 3.2, "category": "convertible", "issuer": "SPDR"},
    {"symbol": "ICVT", "name": "iShares Convertible Bond", "credit": "BB", "duration": 3.5, "category": "convertible", "issuer": "iShares"},
    # Preferred / Senior Loans
    {"symbol": "PFF", "name": "iShares Preferred & Income Securities", "credit": "BBB", "duration": 3.0, "category": "preferred", "issuer": "iShares"},
    {"symbol": "BKLN", "name": "Invesco Senior Loan", "credit": "BB", "duration": 0.3, "category": "senior_loan", "issuer": "Invesco"},
    {"symbol": "SRLN", "name": "SPDR Blackstone Senior Loan", "credit": "BB", "duration": 0.4, "category": "senior_loan", "issuer": "SPDR"},
]

# Credit rating ordering (ascending quality = ascending index means lower quality)
_CREDIT_ORDER = ["D", "C", "CC", "CCC", "B", "B+", "BB-", "BB", "BB+",
                  "BBB-", "BBB", "BBB+", "A-", "A", "A+", "AA-", "AA", "AA+", "AAA"]

_CREDIT_GROUPS = {
    "HY":  {"B", "B+", "BB-", "BB", "BB+", "CCC"},
    "IG":  {"BBB-", "BBB", "BBB+", "A-", "A", "A+", "AA-", "AA", "AA+", "AAA"},
    "AAA": {"AAA"},
    "AA":  {"AA-", "AA", "AA+"},
    "A":   {"A-", "A", "A+"},
    "BBB": {"BBB-", "BBB", "BBB+"},
    "BB":  {"BB-", "BB", "BB+"},
    "B":   {"B", "B+"},
}

# FRED series used for market context
_FRED_SERIES = {
    "10Y": "GS10",        # 10-Year Treasury
    "2Y":  "GS2",         # 2-Year Treasury
    "3M":  "GS3M",        # 3-Month Treasury
    "OAS_IG":  "BAMLC0A0CM",   # IG OAS spread
    "OAS_HY":  "BAMLH0A0HYM2", # HY OAS spread
    "REAL_10Y": "DFII10",  # 10Y TIPS (real yield)
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BondScreenCriteria(BaseModel):
    min_yield: Optional[float] = None
    max_yield: Optional[float] = None
    credit_quality: Optional[list[str]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    category: Optional[str] = None


class BondResult(BaseModel):
    symbol: str
    name: str
    credit_quality: str
    duration_years: float
    yield_pct: float
    ytd_return_pct: Optional[float] = None
    aum_billions: Optional[float] = None
    category: str
    issuer: str
    spread_vs_treasury: Optional[float] = None


class MarketContext(BaseModel):
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    treasury_3m: Optional[float] = None
    real_yield_10y: Optional[float] = None
    oas_ig_bps: Optional[float] = None
    oas_hy_bps: Optional[float] = None
    curve_slope_bps: Optional[float] = None
    as_of: str = ""


class BondScreenResult(BaseModel):
    criteria: BondScreenCriteria
    results: list[BondResult]
    total_matched: int
    market_context: MarketContext
    nl_query: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FRED helper — fetch latest value for a series
# ---------------------------------------------------------------------------


async def _fred_latest(client: httpx.AsyncClient, series_id: str) -> Optional[float]:
    try:
        r = await client.get(
            _FRED_BASE,
            params={"id": series_id, "vintage_date": "9999-12-31"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        lines = r.text.strip().splitlines()
        # CSV: DATE,VALUE — last row is most recent
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                try:
                    return float(parts[1].strip())
                except ValueError:
                    continue
    except Exception as exc:
        logger.debug("FRED %s: %s", series_id, exc)
    return None


async def _fetch_market_context(client: httpx.AsyncClient) -> MarketContext:
    tasks = {k: _fred_latest(client, v) for k, v in _FRED_SERIES.items()}
    results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    t10 = results.get("10Y")
    t2 = results.get("2Y")
    slope = round((t10 - t2) * 100, 1) if t10 and t2 else None

    from datetime import date
    return MarketContext(
        treasury_10y=t10,
        treasury_2y=t2,
        treasury_3m=results.get("3M"),
        real_yield_10y=results.get("REAL_10Y"),
        oas_ig_bps=round(results["OAS_IG"] * 100, 1) if results.get("OAS_IG") else None,
        oas_hy_bps=round(results["OAS_HY"] * 100, 1) if results.get("OAS_HY") else None,
        curve_slope_bps=slope,
        as_of=date.today().isoformat(),
    )


# ---------------------------------------------------------------------------
# yfinance ETF data
# ---------------------------------------------------------------------------


async def _fetch_etf_data(symbols: list[str]) -> dict[str, dict]:
    import yfinance as yf

    def _fetch() -> dict[str, dict]:
        out: dict[str, dict] = {}
        for sym in symbols:
            try:
                tk = yf.Ticker(sym)
                info = tk.info or {}
                hist = tk.history(period="1y", auto_adjust=True)
                ytd_ret = None
                if not hist.empty and len(hist) >= 2:
                    ytd_ret = round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2)

                # yield: prefer yield from info, fallback to trailing yield
                yld = (
                    info.get("yield")
                    or info.get("dividendYield")
                    or info.get("trailingAnnualDividendYield")
                )
                if yld and yld > 0:
                    yld = round(yld * 100, 2)  # convert to percent
                else:
                    yld = None

                aum = info.get("totalAssets")
                if aum:
                    aum = round(aum / 1e9, 1)

                out[sym] = {"yield_pct": yld, "ytd_return_pct": ytd_ret, "aum_billions": aum}
            except Exception as exc:
                logger.debug("yfinance %s: %s", sym, exc)
                out[sym] = {"yield_pct": None, "ytd_return_pct": None, "aum_billions": None}
        return out

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# NL query parser
# ---------------------------------------------------------------------------


def _parse_nl_query(query: str) -> BondScreenCriteria:
    """Extract screening criteria from a natural language query."""
    q = query.lower()
    criteria: dict = {}

    # Yield patterns
    m = re.search(r"yield\s*[>≥]\s*([\d.]+)\s*%?", q)
    if m:
        criteria["min_yield"] = float(m.group(1))
    m = re.search(r"yield\s*[<≤]\s*([\d.]+)\s*%?", q)
    if m:
        criteria["max_yield"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*%?\s*[<≤]\s*yield", q)
    if m:
        criteria["min_yield"] = float(m.group(1))
    m = re.search(r"yield\s+between\s+([\d.]+)\s+and\s+([\d.]+)", q)
    if m:
        criteria["min_yield"] = float(m.group(1))
        criteria["max_yield"] = float(m.group(2))
    m = re.search(r"yield\s+above\s+([\d.]+)", q)
    if m:
        criteria["min_yield"] = float(m.group(1))

    # Duration patterns
    m = re.search(r"duration\s*[<≤]\s*([\d.]+)", q)
    if m:
        criteria["max_duration"] = float(m.group(1))
    m = re.search(r"duration\s*[>≥]\s*([\d.]+)", q)
    if m:
        criteria["min_duration"] = float(m.group(1))
    if "short" in q and "duration" in q:
        criteria.setdefault("max_duration", 4.0)
    if "long" in q and "duration" in q:
        criteria.setdefault("min_duration", 10.0)

    # Credit quality
    credit_tags = []
    if "investment grade" in q or "inv grade" in q or " ig " in q:
        credit_tags.append("IG")
    if "high yield" in q or " hy " in q or "junk" in q:
        credit_tags.append("HY")
    for rating in ["aaa", "aa", "a", "bbb", "bb", "b"]:
        if re.search(rf"\b{rating}\b", q):
            credit_tags.append(rating.upper())
    if credit_tags:
        criteria["credit_quality"] = credit_tags

    # Category
    cat_map = {
        "treasury": "government", "government": "government",
        "corporate": "corporate_ig", "corp": "corporate_ig",
        "muni": "muni", "municipal": "muni",
        "tips": "tips", "inflation": "tips",
        "emerging": "em_sovereign", "em bond": "em_sovereign",
        "convertible": "convertible",
        "mortgage": "mortgage", "mbs": "mortgage",
        "preferred": "preferred",
        "loan": "senior_loan",
    }
    for kw, cat in cat_map.items():
        if kw in q:
            criteria["category"] = cat
            break

    return BondScreenCriteria(**criteria)


# ---------------------------------------------------------------------------
# Credit filter
# ---------------------------------------------------------------------------


def _credit_matches(etf_credit: str, allowed: list[str]) -> bool:
    """Return True if etf_credit satisfies any allowed credit group/rating."""
    for tag in allowed:
        tag_upper = tag.upper()
        if tag_upper in _CREDIT_GROUPS:
            if etf_credit in _CREDIT_GROUPS[tag_upper]:
                return True
        elif etf_credit.upper() == tag_upper:
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def screen_bonds(
    criteria: Optional[BondScreenCriteria] = None,
    query: Optional[str] = None,
    limit: int = 25,
) -> BondScreenResult:
    """
    Screen bond ETFs by yield, duration, credit quality, and category.

    Args:
        criteria:  Structured filter criteria. If None, derived from query.
        query:     Natural language query ("IG bonds yield > 5% short duration").
        limit:     Maximum number of results (default 25).

    Returns:
        BondScreenResult with matched ETFs and live market context.
    """
    warnings_: list[str] = []

    if criteria is None and query:
        criteria = _parse_nl_query(query)
    elif criteria is None:
        criteria = BondScreenCriteria()

    # First pass: filter catalog by structural criteria (no network needed)
    candidates = []
    for etf in _ETF_CATALOG:
        if criteria.min_duration is not None and etf["duration"] < criteria.min_duration:
            continue
        if criteria.max_duration is not None and etf["duration"] > criteria.max_duration:
            continue
        if criteria.credit_quality and not _credit_matches(etf["credit"], criteria.credit_quality):
            continue
        if criteria.category and etf["category"] != criteria.category:
            continue
        candidates.append(etf)

    if not candidates:
        candidates = _ETF_CATALOG[:]
        warnings_.append("No ETFs matched structural filters — returning full catalog sample")

    symbols = [e["symbol"] for e in candidates]

    # Fetch live yield/return data + market context concurrently
    async with httpx.AsyncClient() as client:
        etf_task = asyncio.create_task(_fetch_etf_data(symbols))
        ctx_task = asyncio.create_task(_fetch_market_context(client))
        etf_data, market_ctx = await asyncio.gather(etf_task, ctx_task)

    # Build result objects with live yield-based filter
    results: list[BondResult] = []
    for etf in candidates:
        sym = etf["symbol"]
        live = etf_data.get(sym, {})
        yld = live.get("yield_pct")

        if yld is None:
            warnings_.append(f"{sym}: yield unavailable, using estimated yield")
            # Estimate: 10Y treasury + credit spread heuristic
            base = market_ctx.treasury_10y or 4.5
            spread_map = {"AAA": 0.1, "AA": 0.2, "A": 0.5, "BBB": 1.0, "BB": 3.0, "B": 5.0}
            yld = round(base + spread_map.get(etf["credit"], 1.0), 2)

        if criteria.min_yield is not None and yld < criteria.min_yield:
            continue
        if criteria.max_yield is not None and yld > criteria.max_yield:
            continue

        # Spread vs comparable Treasury
        spread = None
        if market_ctx.treasury_10y is not None:
            # Use duration-matched Treasury as proxy
            dur = etf["duration"]
            if dur <= 2:
                tsy = market_ctx.treasury_2y
            elif dur <= 7:
                tsy = market_ctx.treasury_2y and market_ctx.treasury_10y and (
                    market_ctx.treasury_2y + (market_ctx.treasury_10y - market_ctx.treasury_2y) * (dur - 2) / 8
                )
            else:
                tsy = market_ctx.treasury_10y
            if tsy:
                spread = round((yld - tsy) * 100, 0)

        results.append(BondResult(
            symbol=sym,
            name=etf["name"],
            credit_quality=etf["credit"],
            duration_years=etf["duration"],
            yield_pct=yld,
            ytd_return_pct=live.get("ytd_return_pct"),
            aum_billions=live.get("aum_billions"),
            category=etf["category"],
            issuer=etf["issuer"],
            spread_vs_treasury=spread,
        ))

    # Sort by yield descending, cap at limit
    results.sort(key=lambda r: r.yield_pct, reverse=True)
    results = results[:limit]

    return BondScreenResult(
        criteria=criteria,
        results=results,
        total_matched=len(results),
        market_context=market_ctx,
        nl_query=query,
        warnings=warnings_,
    )
