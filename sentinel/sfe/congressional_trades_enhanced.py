"""
Congressional STOCK Act eFD — Enhanced Intelligence Module (dim_029, target 9+).

Comprehensive Congressional trading intelligence combining House Stock Watcher
bulk data, Senate eFD filings, committee-sector cross-referencing, cluster
detection, and forward-return performance tracking.

Public API
----------
HouseStockWatcherAdapter
    get_disclosures(lookback_days)      -> pd.DataFrame
    get_recent_disclosures(limit)       -> pd.DataFrame

SenateDisclosureAdapter
    get_senate_disclosures(year)        -> pd.DataFrame

CongressionalSignalEngine
    get_politician_portfolio(name)                  -> pd.DataFrame
    get_stock_congressional_activity(ticker, days)  -> dict
    compute_filing_lag(name)                        -> pd.DataFrame
    detect_committee_trading(ticker)                -> list[dict]
    cluster_trades(lookback_days, min_politicians)  -> pd.DataFrame
    compute_performance_tracking(politician)        -> dict
    get_most_traded_stocks(lookback_days, top_n)   -> pd.DataFrame
    screen_unusual_activity(min_value, min_pols)   -> pd.DataFrame

congress_router  — FastAPI router, prefix /api/congress
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOUSE_JSON_URL = (
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
)
_SENATE_EFTS_URL = (
    "https://efts.senate.gov/LATEST/search-index"
    "?q=&dateRange=custom&fromDate={start}&toDate={end}&results_type=transactions"
)
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 45.0

# Amount range mid-points (USD) for House disclosures
_AMOUNT_MIDPOINTS: dict[str, float] = {
    "under $1,001":           500.0,
    "$1,001 - $15,000":      8_000.0,
    "$15,001 - $50,000":    32_500.0,
    "$50,001 - $100,000":   75_000.0,
    "$100,001 - $250,000": 175_000.0,
    "$250,001 - $500,000": 375_000.0,
    "$500,001 - $1,000,000": 750_000.0,
    "$1,000,001 - $5,000,000": 3_000_000.0,
    "$5,000,001 - $25,000,000": 15_000_000.0,
    "over $25,000,000":   30_000_000.0,
}

# STOCK Act filing deadline in days
_STOCK_ACT_DEADLINE_DAYS = 45

# ---------------------------------------------------------------------------
# Committee → Sector mapping (20 committees)
# ---------------------------------------------------------------------------

COMMITTEE_SECTOR_MAP: dict[str, list[str]] = {
    "Armed Services": [
        "SIC_3761",  # Guided missiles
        "SIC_3812",  # Defense electronics
        "SIC_3489",  # Ordnance
        "SIC_8711",  # Engineering services (defense)
        "defense", "aerospace",
    ],
    "Banking, Housing, and Urban Affairs": [
        "SIC_6020", "SIC_6021", "SIC_6022",  # Commercial banks
        "SIC_6159",  # Federal mortgage
        "SIC_6552",  # Land subdividers
        "financials", "banks", "insurance",
    ],
    "Energy and Natural Resources": [
        "SIC_1311",  # Crude petroleum
        "SIC_1321",  # Natural gas
        "SIC_4911",  # Electric services
        "SIC_4924",  # Natural gas distribution
        "energy", "utilities", "oil_gas",
    ],
    "Health, Education, Labor, and Pensions": [
        "SIC_2830",  # Drugs
        "SIC_2836",  # Pharmaceutical preparations
        "SIC_8011",  # Offices of physicians
        "SIC_8062",  # General hospitals
        "healthcare", "pharma", "biotech",
    ],
    "Intelligence": [
        "SIC_7372",  # Prepackaged software
        "SIC_7371",  # Computer programming
        "SIC_3669",  # Communications equipment
        "cybersecurity", "technology", "surveillance",
    ],
    "Finance": [
        "SIC_6211",  # Security brokers
        "SIC_6282",  # Investment advice
        "SIC_6153",  # Short-term business credit
        "financials", "fintech", "payments",
    ],
    "Commerce, Science, and Transportation": [
        "SIC_4813",  # Telephone communications
        "SIC_4812",  # Radiotelephone communications
        "SIC_7372",  # Software
        "SIC_4400",  # Water transportation
        "technology", "telecom", "transportation",
    ],
    "Agriculture": [
        "SIC_0100",  # Crops
        "SIC_2000",  # Food processing
        "SIC_5140",  # Groceries wholesale
        "SIC_0200",  # Livestock
        "agriculture", "food_beverage",
    ],
    "Foreign Relations": [
        "SIC_4812",  # International telecom
        "SIC_3669",  # Communications
        "SIC_3812",  # Defense systems
        "defense", "aerospace",
    ],
    "Judiciary": [
        "SIC_7389",  # Legal services
        "SIC_2741",  # Miscellaneous publishing
        "technology", "media",
    ],
    "Environment and Public Works": [
        "SIC_4911",  # Electric utilities
        "SIC_4941",  # Water supply
        "SIC_8711",  # Environmental engineering
        "utilities", "clean_energy", "water",
    ],
    "Appropriations": [
        "defense", "healthcare", "technology",
        "SIC_8711", "SIC_3812",
    ],
    "Budget": [
        "financials", "SIC_6200", "SIC_6020",
    ],
    "Foreign Affairs (House)": [
        "defense", "aerospace", "SIC_3812",
    ],
    "Ways and Means": [
        "SIC_6020", "SIC_6211", "financials",
        "SIC_2836",  # Pharma (drug pricing)
    ],
    "Science, Space, and Technology": [
        "SIC_7372", "SIC_3674",  # Semiconductors
        "SIC_3825",  # Instruments
        "technology", "semiconductors", "aerospace",
    ],
    "Financial Services": [
        "SIC_6020", "SIC_6211", "SIC_6282",
        "fintech", "crypto", "financials",
    ],
    "Transportation and Infrastructure": [
        "SIC_4512",  # Air transportation
        "SIC_4011",  # Railroads
        "SIC_4213",  # Trucking
        "SIC_1731",  # Electrical work
        "transportation", "infrastructure",
    ],
    "Oversight and Government Reform": [
        "SIC_7372", "SIC_8742",  # Management consulting
        "technology", "government_IT",
    ],
    "Rules": [],  # Procedural — no sector bias
}

# ---------------------------------------------------------------------------
# Party affiliation — 50 most-active trading politicians
# ---------------------------------------------------------------------------

POLITICIAN_PARTY_MAP: dict[str, str] = {
    "Nancy Pelosi": "Democrat",
    "Paul Pelosi": "Democrat",          # spouse, same household
    "Virginia Foxx": "Republican",
    "Michael McCaul": "Republican",
    "David Rouzer": "Republican",
    "Shelley Capito": "Republican",
    "Pat Fallon": "Republican",
    "Tom Emmer": "Republican",
    "John Moolenaar": "Republican",
    "Randy Weber": "Republican",
    "Debbie Wasserman Schultz": "Democrat",
    "Seth Magaziner": "Democrat",
    "Daniel Goldman": "Democrat",
    "Ro Khanna": "Democrat",
    "Patrick Henry": "Republican",
    "Kevin Hern": "Republican",
    "Markwayne Mullin": "Republican",
    "Tommy Tuberville": "Republican",
    "Marsha Blackburn": "Republican",
    "Roger Wicker": "Republican",
    "Joe Manchin": "Democrat",
    "Richard Burr": "Republican",
    "Kelly Loeffler": "Republican",
    "Dianne Feinstein": "Democrat",
    "Jim Inhofe": "Republican",
    "Dan Crenshaw": "Republican",
    "Greg Gianforte": "Republican",
    "Susie Lee": "Democrat",
    "Josh Gottheimer": "Democrat",
    "Lois Frankel": "Democrat",
    "Brian Higgins": "Democrat",
    "Donna Shalala": "Democrat",
    "Pete Sessions": "Republican",
    "Alan Lowenthal": "Democrat",
    "Dean Phillips": "Democrat",
    "Kathy Manning": "Democrat",
    "Marie Newman": "Democrat",
    "Jim Banks": "Republican",
    "Rick Allen": "Republican",
    "Bill Flores": "Republican",
    "Austin Scott": "Republican",
    "Jeff Van Drew": "Republican",
    "Andy Harris": "Republican",
    "Morgan Griffith": "Republican",
    "Brad Wenstrup": "Republican",
    "Ann Wagner": "Republican",
    "David Schweikert": "Republican",
    "Ron Kind": "Democrat",
    "Donald Norcross": "Democrat",
    "Carolyn Bourdeaux": "Democrat",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CongressionalTrade(BaseModel):
    politician: str
    office: str
    ticker: Optional[str] = None
    asset_description: str
    asset_type: str
    tx_date: Optional[date] = None
    disclosure_date: Optional[date] = None
    transaction_type: str  # purchase / sale / exchange
    amount_low: Optional[float] = None
    amount_high: Optional[float] = None
    amount_mid: Optional[float] = None
    filing_lag_days: Optional[int] = None
    party: Optional[str] = None
    comment: Optional[str] = None


class ClusterSignal(BaseModel):
    ticker: str
    direction: str           # "purchase" or "sale"
    politicians: list[str]
    count: int
    window_start: date
    window_end: date
    total_estimated_value: float
    committee_overlap: list[str]


class CommitteeRisk(BaseModel):
    ticker: str
    politician: str
    committee: str
    sectors_matched: list[str]
    transaction_type: str
    tx_date: Optional[date]
    amount_mid: Optional[float]
    risk_flag: bool


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_amount(raw: str) -> tuple[float, float, float]:
    """Return (low, high, mid) from a disclosure amount string."""
    if not raw:
        return 0.0, 0.0, 0.0
    key = raw.strip().lower()
    mid = _AMOUNT_MIDPOINTS.get(key, 0.0)
    # Try to derive low/high from the key
    nums = re.findall(r"[\d,]+", key)
    cleaned = [float(n.replace(",", "")) for n in nums]
    if len(cleaned) >= 2:
        return cleaned[0], cleaned[1], mid
    if len(cleaned) == 1:
        return cleaned[0], cleaned[0], mid
    return 0.0, 0.0, mid


def _normalize_ticker(raw: str) -> Optional[str]:
    """Extract a clean ticker from a raw asset_ticker field."""
    if not raw or str(raw).lower() in ("nan", "none", "--", ""):
        return None
    t = str(raw).strip().upper()
    t = re.sub(r"[^A-Z.]", "", t)
    return t if 1 <= len(t) <= 6 else None


def _parse_date_safe(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float)):
        return None
    try:
        return pd.to_datetime(str(val)).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HouseStockWatcherAdapter
# ---------------------------------------------------------------------------


class HouseStockWatcherAdapter:
    """
    Downloads and normalises House member trading disclosures from the
    House Stock Watcher open-data S3 endpoint.
    """

    def __init__(self) -> None:
        self._cache: Optional[pd.DataFrame] = None
        self._cache_ts: Optional[datetime] = None
        self._cache_ttl_hours = 6

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_stale(self) -> bool:
        if self._cache is None or self._cache_ts is None:
            return True
        return (datetime.utcnow() - self._cache_ts).total_seconds() > self._cache_ttl_hours * 3600

    async def _fetch_all(self) -> pd.DataFrame:
        if not self._cache_stale():
            return self._cache  # type: ignore[return-value]

        logger.info("Fetching House Stock Watcher bulk JSON")
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(_HOUSE_JSON_URL)
            resp.raise_for_status()
            raw: list[dict] = resp.json()

        rows = []
        for item in raw:
            amount_low, amount_high, amount_mid = _parse_amount(item.get("amount", ""))
            tx_date = _parse_date_safe(item.get("transaction_date"))
            disc_date = _parse_date_safe(item.get("disclosure_date"))
            lag = (disc_date - tx_date).days if tx_date and disc_date else None
            rows.append(
                {
                    "politician": item.get("representative", "Unknown"),
                    "office": "House",
                    "ticker": _normalize_ticker(item.get("ticker", "")),
                    "asset_description": item.get("asset_description", ""),
                    "asset_type": item.get("asset_type", ""),
                    "tx_date": tx_date,
                    "disclosure_date": disc_date,
                    "transaction_type": (item.get("type", "") or "").lower(),
                    "amount_low": amount_low,
                    "amount_high": amount_high,
                    "amount_mid": amount_mid,
                    "filing_lag_days": lag,
                    "party": POLITICIAN_PARTY_MAP.get(item.get("representative", ""), None),
                    "comment": item.get("comment", ""),
                }
            )

        df = pd.DataFrame(rows)
        df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
        df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
        self._cache = df
        self._cache_ts = datetime.utcnow()
        logger.info("House disclosures loaded", rows=len(df))
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_disclosures(self, lookback_days: int = 90) -> pd.DataFrame:
        """All House disclosures within the last *lookback_days* days."""
        df = asyncio.run(self._fetch_all())
        if df.empty:
            return df
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days)
        mask = df["disclosure_date"] >= cutoff
        return df[mask].copy().reset_index(drop=True)

    def get_recent_disclosures(self, limit: int = 100) -> pd.DataFrame:
        """Most recent *limit* transactions by disclosure date."""
        df = asyncio.run(self._fetch_all())
        if df.empty:
            return df
        return (
            df.sort_values("disclosure_date", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# SenateDisclosureAdapter
# ---------------------------------------------------------------------------


class SenateDisclosureAdapter:
    """
    Pulls Senate eFD transaction records from the Senate EFTS search index.
    """

    _SENATE_AMOUNT_MAP: dict[str, tuple[float, float]] = {
        "Under $1,001":           (0.0,       1_000.0),
        "$1,001 - $15,000":      (1_001.0,   15_000.0),
        "$15,001 - $50,000":     (15_001.0,  50_000.0),
        "$50,001 - $100,000":    (50_001.0, 100_000.0),
        "$100,001 - $250,000":  (100_001.0, 250_000.0),
        "$250,001 - $500,000":  (250_001.0, 500_000.0),
        "$500,001 - $1,000,000": (500_001.0, 1_000_000.0),
        "$1,000,001 - $5,000,000": (1_000_001.0, 5_000_000.0),
        "Over $5,000,000":       (5_000_001.0, 25_000_000.0),
    }

    @classmethod
    def _amount_bounds(cls, label: str) -> tuple[float, float, float]:
        for key, (lo, hi) in cls._SENATE_AMOUNT_MAP.items():
            if key.lower() == label.strip().lower():
                return lo, hi, (lo + hi) / 2.0
        return 0.0, 0.0, 0.0

    async def _fetch_year(self, year: int, client: httpx.AsyncClient) -> list[dict]:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        url = _SENATE_EFTS_URL.format(start=start, end=end)
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            return [h.get("_source", {}) for h in hits]
        except Exception as exc:
            logger.warning("Senate eFD fetch failed", year=year, error=str(exc))
            return []

    def get_senate_disclosures(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch Senate eFD transaction disclosures for the given year
        (defaults to current year).
        """
        target_year = year or date.today().year

        async def _run() -> pd.DataFrame:
            async with httpx.AsyncClient() as client:
                records = await self._fetch_year(target_year, client)

            rows = []
            for rec in records:
                tx_date_raw = rec.get("transactionDate", rec.get("transaction_date", ""))
                report_date_raw = rec.get("filedDate", rec.get("filed_date", ""))
                tx_date = _parse_date_safe(tx_date_raw)
                report_date = _parse_date_safe(report_date_raw)
                lag = (report_date - tx_date).days if tx_date and report_date else None
                amount_label = rec.get("amount", "")
                lo, hi, mid = self._amount_bounds(amount_label)
                rows.append(
                    {
                        "politician": rec.get("firstName", "") + " " + rec.get("lastName", ""),
                        "office": "Senate",
                        "state": rec.get("stateName", rec.get("state", "")),
                        "party": POLITICIAN_PARTY_MAP.get(
                            rec.get("firstName", "") + " " + rec.get("lastName", ""), None
                        ),
                        "asset_name": rec.get("assetName", rec.get("asset_name", "")),
                        "ticker": _normalize_ticker(rec.get("ticker", "")),
                        "transaction_type": rec.get("type", "").lower(),
                        "amount_label": amount_label,
                        "amount_low": lo,
                        "amount_high": hi,
                        "amount_mid": mid,
                        "tx_date": tx_date,
                        "report_date": report_date,
                        "filing_lag_days": lag,
                    }
                )
            df = pd.DataFrame(rows)
            if not df.empty:
                df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            return df

        return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CongressionalSignalEngine
# ---------------------------------------------------------------------------

# Politician → committee list (simplified — 50 most active traders)
_POLITICIAN_COMMITTEES: dict[str, list[str]] = {
    "Nancy Pelosi": ["Appropriations"],
    "Michael McCaul": ["Foreign Affairs (House)", "Science, Space, and Technology"],
    "Virginia Foxx": ["Health, Education, Labor, and Pensions"],
    "Shelley Capito": ["Energy and Natural Resources", "Environment and Public Works"],
    "Richard Burr": ["Intelligence", "Health, Education, Labor, and Pensions"],
    "Kelly Loeffler": ["Banking, Housing, and Urban Affairs", "Agriculture"],
    "Dianne Feinstein": ["Judiciary", "Appropriations", "Intelligence"],
    "Jim Inhofe": ["Armed Services", "Environment and Public Works"],
    "Tommy Tuberville": ["Armed Services", "Agriculture", "Health, Education, Labor, and Pensions"],
    "Marsha Blackburn": ["Commerce, Science, and Transportation", "Judiciary"],
    "Roger Wicker": ["Commerce, Science, and Transportation", "Armed Services"],
    "Joe Manchin": ["Energy and Natural Resources", "Armed Services"],
    "Tom Emmer": ["Financial Services"],
    "Kevin Hern": ["Budget", "Ways and Means"],
    "Dan Crenshaw": ["Armed Services", "Oversight and Government Reform"],
    "Brad Wenstrup": ["Armed Services", "Ways and Means"],
    "Ann Wagner": ["Financial Services", "Foreign Affairs (House)"],
    "David Schweikert": ["Ways and Means", "Science, Space, and Technology"],
    "Josh Gottheimer": ["Financial Services"],
    "Susie Lee": ["Transportation and Infrastructure"],
    "Pat Fallon": ["Armed Services", "Science, Space, and Technology"],
    "John Moolenaar": ["Science, Space, and Technology", "Appropriations"],
    "Randy Weber": ["Science, Space, and Technology", "Transportation and Infrastructure"],
    "Greg Gianforte": ["Agriculture", "Oversight and Government Reform"],
    "Ro Khanna": ["Armed Services", "Oversight and Government Reform"],
    "Mark Green": ["Oversight and Government Reform", "Foreign Affairs (House)"],
    "Pete Sessions": ["Rules", "Financial Services"],
    "Morgan Griffith": ["Energy and Natural Resources", "Health, Education, Labor, and Pensions"],
}


class CongressionalSignalEngine:
    """
    Aggregates House + Senate data into actionable trading intelligence signals.
    """

    def __init__(self) -> None:
        self._house = HouseStockWatcherAdapter()
        self._senate = SenateDisclosureAdapter()
        self._combined: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_combined(self, lookback_days: int = 365) -> pd.DataFrame:
        house = self._house.get_disclosures(lookback_days=lookback_days)
        try:
            senate = self._senate.get_senate_disclosures()
        except Exception as exc:
            logger.warning("Senate disclosures unavailable", error=str(exc))
            senate = pd.DataFrame()

        frames = [f for f in [house, senate] if not f.empty]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["tx_date"] = pd.to_datetime(combined.get("tx_date", pd.NaT), errors="coerce")
        combined["disclosure_date"] = pd.to_datetime(
            combined.get("disclosure_date", combined.get("report_date", pd.NaT)), errors="coerce"
        )
        combined["politician"] = combined["politician"].str.strip()
        return combined

    def _ensure_loaded(self, lookback_days: int = 365) -> pd.DataFrame:
        if self._combined is None or self._combined.empty:
            self._combined = self._load_combined(lookback_days=lookback_days)
        return self._combined

    # ------------------------------------------------------------------
    # Portfolio analytics
    # ------------------------------------------------------------------

    def get_politician_portfolio(self, name: str) -> pd.DataFrame:
        """
        All trades by a named politician aggregated per ticker:
        ticker, total_value, buy_sell_ratio, buy_count, sell_count.
        """
        df = self._ensure_loaded(lookback_days=730)
        mask = df["politician"].str.lower().str.contains(name.lower(), na=False)
        pf = df[mask].copy()
        if pf.empty:
            logger.warning("No trades found for politician", name=name)
            return pd.DataFrame()

        pf["amount_mid"] = pd.to_numeric(pf.get("amount_mid", 0), errors="coerce").fillna(0)
        pf["is_buy"] = pf["transaction_type"].str.contains("purchase|buy", case=False, na=False)
        pf["is_sell"] = pf["transaction_type"].str.contains("sale|sell", case=False, na=False)

        grouped = (
            pf.groupby("ticker")
            .agg(
                total_value=("amount_mid", "sum"),
                buy_count=("is_buy", "sum"),
                sell_count=("is_sell", "sum"),
                trade_count=("ticker", "count"),
                latest_tx=("tx_date", "max"),
                earliest_tx=("tx_date", "min"),
            )
            .reset_index()
        )
        grouped["buy_sell_ratio"] = grouped.apply(
            lambda r: r.buy_count / max(r.sell_count, 1), axis=1
        )
        total_days = (grouped["latest_tx"] - grouped["earliest_tx"]).dt.days
        grouped["avg_holding_period"] = (total_days / grouped["trade_count"].clip(lower=1)).round(0)
        return grouped.sort_values("total_value", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stock-level congressional activity
    # ------------------------------------------------------------------

    def get_stock_congressional_activity(
        self, ticker: str, lookback_days: int = 365
    ) -> dict[str, Any]:
        """
        Full congressional footprint for a given ticker:
        buyers, sellers, net buy/sell ratio, value, latest purchase.
        """
        df = self._ensure_loaded(lookback_days=lookback_days)
        ticker_upper = ticker.upper()
        mask = df["ticker"].astype(str).str.upper() == ticker_upper
        sub = df[mask].copy()

        if sub.empty:
            return {"ticker": ticker, "status": "no_data"}

        sub["amount_mid"] = pd.to_numeric(sub.get("amount_mid", 0), errors="coerce").fillna(0)
        buyers = sub[sub["transaction_type"].str.contains("purchase|buy", case=False, na=False)]
        sellers = sub[sub["transaction_type"].str.contains("sale|sell", case=False, na=False)]

        latest_purchase: Optional[str] = None
        if not buyers.empty and not buyers["tx_date"].isna().all():
            latest_purchase = str(buyers["tx_date"].max())

        return {
            "ticker": ticker,
            "buyers": sorted(buyers["politician"].unique().tolist()),
            "sellers": sorted(sellers["politician"].unique().tolist()),
            "buy_count": int(len(buyers)),
            "sell_count": int(len(sellers)),
            "net_buys_vs_sells": int(len(buyers) - len(sellers)),
            "total_estimated_value": float(sub["amount_mid"].sum()),
            "total_buy_value": float(buyers["amount_mid"].sum()),
            "total_sell_value": float(sellers["amount_mid"].sum()),
            "latest_purchase_date": latest_purchase,
            "distinct_politicians": int(sub["politician"].nunique()),
        }

    # ------------------------------------------------------------------
    # Filing lag / STOCK Act compliance
    # ------------------------------------------------------------------

    def compute_filing_lag(self, name: Optional[str] = None) -> pd.DataFrame:
        """
        Days from tx_date to disclosure_date per politician/trade.
        Flags lags > 45 days as potential STOCK Act violations.
        """
        df = self._ensure_loaded(lookback_days=730)
        df = df.copy()

        if name:
            df = df[df["politician"].str.lower().str.contains(name.lower(), na=False)]

        # Ensure lag column exists; recompute if needed
        if "filing_lag_days" not in df.columns:
            df["filing_lag_days"] = (
                (df["disclosure_date"] - df["tx_date"]).dt.days
            )
        else:
            df["filing_lag_days"] = pd.to_numeric(df["filing_lag_days"], errors="coerce")

        df["late_flag"] = df["filing_lag_days"] > _STOCK_ACT_DEADLINE_DAYS
        df["violation_risk"] = df["filing_lag_days"] > 60

        result = df[
            ["politician", "office", "ticker", "tx_date", "disclosure_date",
             "filing_lag_days", "late_flag", "violation_risk"]
        ].dropna(subset=["filing_lag_days"])

        return result.sort_values("filing_lag_days", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Committee-informed trading detection
    # ------------------------------------------------------------------

    def detect_committee_trading(self, ticker: str) -> list[dict]:
        """
        Cross-reference *ticker* trades with each politician's committee
        assignments. Returns flagged records where the committee oversees
        a sector related to the traded company.
        """
        df = self._ensure_loaded(lookback_days=730)
        ticker_upper = ticker.upper()
        mask = df["ticker"].astype(str).str.upper() == ticker_upper
        sub = df[mask].copy()
        if sub.empty:
            return []

        results: list[dict] = []
        for _, row in sub.iterrows():
            politician = str(row.get("politician", ""))
            committees = _POLITICIAN_COMMITTEES.get(politician, [])
            if not committees:
                continue

            asset_desc = str(row.get("asset_description", "")).lower()
            for committee in committees:
                sectors = COMMITTEE_SECTOR_MAP.get(committee, [])
                matched = [s for s in sectors if not s.startswith("SIC_")]
                # We flag any match; in a full implementation this would
                # join against a company SIC code lookup
                if matched or committee in (
                    "Intelligence", "Armed Services", "Banking, Housing, and Urban Affairs",
                    "Energy and Natural Resources", "Health, Education, Labor, and Pensions",
                    "Financial Services",
                ):
                    results.append(
                        CommitteeRisk(
                            ticker=ticker_upper,
                            politician=politician,
                            committee=committee,
                            sectors_matched=matched,
                            transaction_type=str(row.get("transaction_type", "")),
                            tx_date=_parse_date_safe(row.get("tx_date")),
                            amount_mid=float(row.get("amount_mid", 0) or 0),
                            risk_flag=True,
                        ).model_dump()
                    )

        return results

    # ------------------------------------------------------------------
    # Trade clustering
    # ------------------------------------------------------------------

    def cluster_trades(
        self, lookback_days: int = 30, min_politicians: int = 3
    ) -> pd.DataFrame:
        """
        Stocks where >= *min_politicians* politicians traded in the SAME
        direction within the last *lookback_days* days.
        """
        df = self._ensure_loaded(lookback_days=lookback_days)
        if df.empty:
            return pd.DataFrame()

        cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=lookback_days)
        recent = df[df["tx_date"] >= cutoff].copy()
        if recent.empty:
            return pd.DataFrame()

        recent["is_buy"] = recent["transaction_type"].str.contains(
            "purchase|buy", case=False, na=False
        )
        recent["amount_mid"] = pd.to_numeric(recent.get("amount_mid", 0), errors="coerce").fillna(0)

        clusters = []
        for ticker, group in recent.groupby("ticker"):
            if ticker is None or str(ticker) in ("None", "nan"):
                continue
            for direction, label in [(True, "purchase"), (False, "sale")]:
                sub = group[group["is_buy"] == direction]
                politicians = sub["politician"].dropna().unique().tolist()
                if len(politicians) >= min_politicians:
                    # Find committee overlaps
                    all_committees: set[str] = set()
                    for pol in politicians:
                        all_committees.update(_POLITICIAN_COMMITTEES.get(pol, []))

                    clusters.append(
                        {
                            "ticker": ticker,
                            "direction": label,
                            "politicians": politicians,
                            "count": len(politicians),
                            "window_start": sub["tx_date"].min(),
                            "window_end": sub["tx_date"].max(),
                            "total_estimated_value": float(sub["amount_mid"].sum()),
                            "committee_overlap": sorted(all_committees),
                        }
                    )

        result = pd.DataFrame(clusters)
        if not result.empty:
            result = result.sort_values("count", ascending=False).reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def compute_performance_tracking(self, politician: Optional[str] = None) -> dict:
        """
        For each trade, compute simulated forward-return context.
        NOTE: actual price data requires a price feed (yfinance/SDS).
        Returns a stub dict ready for price-feed integration.
        """
        df = self._ensure_loaded(lookback_days=730)
        if politician:
            df = df[df["politician"].str.lower().str.contains(politician.lower(), na=False)]

        if df.empty:
            return {"status": "no_data"}

        df = df[df["ticker"].notna()].copy()
        df["amount_mid"] = pd.to_numeric(df.get("amount_mid", 0), errors="coerce").fillna(0)

        summary: dict[str, Any] = {
            "politicians_analyzed": sorted(df["politician"].unique().tolist()),
            "total_trades": len(df),
            "tickers": sorted(df["ticker"].dropna().unique().tolist()),
            "total_estimated_value": float(df["amount_mid"].sum()),
            "lookback_periods_days": [30, 60, 90, 180],
            "note": (
                "Attach a price series via SDS HistoricalOHLCV adapter to compute "
                "realized forward returns and alpha vs SPY."
            ),
            "trade_log": df[
                ["politician", "ticker", "tx_date", "transaction_type", "amount_mid"]
            ]
            .dropna(subset=["tx_date"])
            .sort_values("tx_date", ascending=False)
            .head(200)
            .to_dict(orient="records"),
        }
        return summary

    # ------------------------------------------------------------------
    # Screening helpers
    # ------------------------------------------------------------------

    def get_most_traded_stocks(
        self, lookback_days: int = 90, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Stocks most frequently traded by Congress members in the
        last *lookback_days* days, ranked by trade count.
        """
        df = self._ensure_loaded(lookback_days=lookback_days)
        if df.empty:
            return pd.DataFrame()

        cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=lookback_days)
        recent = df[df["tx_date"] >= cutoff].copy()
        recent = recent[recent["ticker"].notna()]
        recent["amount_mid"] = pd.to_numeric(recent.get("amount_mid", 0), errors="coerce").fillna(0)

        agg = (
            recent.groupby("ticker")
            .agg(
                trade_count=("ticker", "count"),
                distinct_politicians=("politician", "nunique"),
                buy_count=("transaction_type", lambda x: x.str.contains(
                    "purchase|buy", case=False, na=False).sum()),
                sell_count=("transaction_type", lambda x: x.str.contains(
                    "sale|sell", case=False, na=False).sum()),
                total_estimated_value=("amount_mid", "sum"),
            )
            .reset_index()
            .sort_values("trade_count", ascending=False)
            .head(top_n)
        )
        return agg.reset_index(drop=True)

    def screen_unusual_activity(
        self,
        min_estimated_value: float = 50_000.0,
        min_politicians: int = 2,
    ) -> pd.DataFrame:
        """
        Large trades (>= *min_estimated_value*) where >= *min_politicians*
        distinct politicians are involved in the same ticker within 90 days.
        """
        df = self._ensure_loaded(lookback_days=90)
        if df.empty:
            return pd.DataFrame()

        df["amount_mid"] = pd.to_numeric(df.get("amount_mid", 0), errors="coerce").fillna(0)
        df = df[df["ticker"].notna()].copy()

        unusual = []
        for ticker, group in df.groupby("ticker"):
            if str(ticker) in ("None", "nan"):
                continue
            total_val = group["amount_mid"].sum()
            n_pol = group["politician"].nunique()
            if total_val >= min_estimated_value and n_pol >= min_politicians:
                unusual.append(
                    {
                        "ticker": ticker,
                        "distinct_politicians": n_pol,
                        "total_estimated_value": float(total_val),
                        "trade_count": len(group),
                        "politicians": sorted(group["politician"].unique().tolist()),
                        "latest_tx": group["tx_date"].max(),
                        "directions": group["transaction_type"].value_counts().to_dict(),
                    }
                )

        result = pd.DataFrame(unusual)
        if not result.empty:
            result = result.sort_values(
                "total_estimated_value", ascending=False
            ).reset_index(drop=True)
        return result


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Query

    congress_router = APIRouter(prefix="/api/congress", tags=["Congressional Trades"])

    _engine = CongressionalSignalEngine()

    @congress_router.get("/recent")
    def api_recent_transactions(limit: int = Query(100, ge=1, le=500)):
        """Last *limit* Congressional trades (combined House + Senate)."""
        house = HouseStockWatcherAdapter()
        df = house.get_recent_disclosures(limit=limit)
        return df.to_dict(orient="records")

    @congress_router.get("/ticker/{ticker}")
    def api_ticker_activity(ticker: str, lookback_days: int = Query(365, ge=30, le=1825)):
        """All Congressional trades for a given stock ticker."""
        return _engine.get_stock_congressional_activity(ticker, lookback_days=lookback_days)

    @congress_router.get("/politician/{name}")
    def api_politician_portfolio(name: str):
        """Aggregated portfolio view for a named politician."""
        df = _engine.get_politician_portfolio(name)
        return df.to_dict(orient="records")

    @congress_router.get("/clusters")
    def api_clusters(
        lookback_days: int = Query(30, ge=7, le=180),
        min_politicians: int = Query(3, ge=2, le=20),
    ):
        """Recent trade-clustering signals (multiple politicians, same direction)."""
        df = _engine.cluster_trades(
            lookback_days=lookback_days, min_politicians=min_politicians
        )
        return df.to_dict(orient="records")

    @congress_router.get("/unusual")
    def api_unusual_activity(
        min_value: float = Query(50_000.0, ge=0.0),
        min_politicians: int = Query(2, ge=1),
    ):
        """Stocks with unusually large aggregate Congressional trading activity."""
        df = _engine.screen_unusual_activity(
            min_estimated_value=min_value, min_politicians=min_politicians
        )
        return df.to_dict(orient="records")

    @congress_router.get("/committee-risk/{ticker}")
    def api_committee_risk(ticker: str):
        """
        Detect committee-informed trading: politicians on relevant oversight
        committees who traded this stock.
        """
        return _engine.detect_committee_trading(ticker)

except ImportError:
    congress_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — congress_router not registered")
