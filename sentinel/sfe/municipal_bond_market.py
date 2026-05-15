"""
Municipal bond market data via MSRB EMMA (Electronic Municipal Market Access).
Free public API: https://emma.msrb.org
Covers bond search, trade data, disclosures, credit analysis.

Dimension: dim_037 — Municipal bond market (MSRB EMMA)
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMMA_BASE = "https://emma.msrb.org"
EMMA_API = "https://emma.msrb.org/api/IssueView"
EMMA_TRADE_API = "https://www.msrb.org/msrb1/tradedata.asp"
EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
TREASURY_DIRECT = "https://www.treasurydirect.gov/TA_WS/securities/search"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

CACHE_DB = Path("sentinel_muni_cache.db")
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.5",
}

# Historical default rates by sector (MSRB/Moody's Municipal Bond Default Studies)
SECTOR_DEFAULT_RATES: Dict[str, float] = {
    "general_obligation": 0.0018,    # 0.18% 10yr cumulative
    "revenue_utility":    0.0089,    # 0.89%
    "revenue_water":      0.0042,    # 0.42%
    "revenue_hospital":   0.0156,    # 1.56%
    "revenue_airport":    0.0031,    # 0.31%
    "revenue_highway":    0.0025,    # 0.25%
    "revenue_school":     0.0011,    # 0.11%
    "housing":            0.0203,    # 2.03%
    "industrial_dev":     0.0412,    # 4.12%
    "tobacco":            0.0610,    # 6.10%
    "other_revenue":      0.0098,    # 0.98%
}

# State abbreviation -> full name
STATES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "Virgin Islands",
}

# State income tax rates (top marginal, approximate 2025)
STATE_TAX_RATES: Dict[str, float] = {
    "AL": 0.050, "AK": 0.000, "AZ": 0.025, "AR": 0.049, "CA": 0.133,
    "CO": 0.044, "CT": 0.069, "DE": 0.066, "FL": 0.000, "GA": 0.055,
    "HI": 0.110, "ID": 0.058, "IL": 0.049, "IN": 0.030, "IA": 0.060,
    "KS": 0.057, "KY": 0.045, "LA": 0.030, "ME": 0.075, "MD": 0.058,
    "MA": 0.090, "MI": 0.043, "MN": 0.098, "MS": 0.047, "MO": 0.054,
    "MT": 0.069, "NE": 0.068, "NV": 0.000, "NH": 0.000, "NJ": 0.109,
    "NM": 0.059, "NY": 0.109, "NC": 0.049, "ND": 0.025, "OH": 0.040,
    "OK": 0.047, "OR": 0.099, "PA": 0.031, "RI": 0.060, "SC": 0.070,
    "SD": 0.000, "TN": 0.000, "TX": 0.000, "UT": 0.047, "VT": 0.088,
    "VA": 0.057, "WA": 0.000, "WV": 0.065, "WI": 0.076, "WY": 0.000,
    "DC": 0.109, "PR": 0.000, "GU": 0.000, "VI": 0.000,
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MuniBond(BaseModel):
    cusip: str
    issuer_name: str = ""
    description: str = ""
    state: str = ""
    sector: str = "general_obligation"
    issuer_type: Literal["GO", "Revenue", "Other"] = "GO"
    coupon_rate: float = 0.0
    maturity_date: str = ""
    dated_date: str = ""
    principal_amount: float = 0.0
    tax_status: Literal["non-AMT", "AMT", "taxable"] = "non-AMT"
    rating_sp: str = "NR"
    rating_moodys: str = "NR"
    rating_fitch: str = "NR"
    call_date: Optional[str] = None
    call_price: float = 100.0
    use_of_proceeds: str = ""
    maturity_bucket: Literal["short", "medium", "long"] = "medium"
    last_trade_price: Optional[float] = None
    last_trade_yield: Optional[float] = None
    last_trade_date: Optional[str] = None
    par_traded_30d: float = 0.0
    outstanding_par: float = 0.0


class MuniTrade(BaseModel):
    cusip: str
    trade_date: str
    settlement_date: str
    par_amount: float
    price: float
    yield_pct: float
    trade_type: Literal["customer_buy", "customer_sell", "interdealer"]
    dealer_id: str = ""
    market_type: Literal["secondary", "primary"] = "secondary"


class TaxEquivResult(BaseModel):
    cusip: str
    muni_yield: float
    federal_tax_rate: float
    state_tax_rate: float
    state: str
    tax_equiv_yield: float
    treasury_yield_same_maturity: float
    spread_to_treasury_bps: float
    after_tax_corporate_rate: Optional[float] = None
    after_tax_treasury_rate: Optional[float] = None
    muni_advantage_vs_corporate_bps: Optional[float] = None
    muni_advantage_vs_treasury_bps: Optional[float] = None
    breakeven_tax_rate: float


class CreditAnalysis(BaseModel):
    state: str
    issuer_type: str
    sector: str
    dscr: Optional[float] = None               # Debt service coverage ratio
    fund_balance_ratio: Optional[float] = None  # Fund balance / expenditures
    pension_liability_pct: Optional[float] = None  # Unfunded pension / budget
    revenue_coverage: Optional[float] = None    # Revenue / debt service
    historical_default_rate: float = 0.0
    credit_score: float = 0.0                   # 0-100 composite
    credit_rating_implied: str = "BBB"
    risk_factors: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


class ScreenerResult(BaseModel):
    bonds: List[MuniBond]
    total: int
    filters_applied: Dict[str, Any]
    yield_stats: Dict[str, float]


class MuniYieldCurve(BaseModel):
    state: str
    as_of_date: str
    tenors: List[float]
    yields: List[float]
    aaa_yields: List[float]
    aa_yields: List[float]
    a_yields: List[float]
    bbb_yields: List[float]
    treasury_yields: List[float]
    muni_treasury_ratios: List[float]


# ---------------------------------------------------------------------------
# SQLite cache helper
# ---------------------------------------------------------------------------

class _CacheDB:
    """Simple SQLite cache with TTL for HTTP responses."""

    def __init__(self, db_path: Path = CACHE_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key=? AND expires_at>?", (key, now)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str, ttl: int = CACHE_TTL_SECONDS) -> None:
        expires = time.time() + ttl
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, value, expires_at) VALUES(?,?,?)",
                (key, value, expires),
            )

    def purge_expired(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE expires_at<?", (time.time(),))


_cache = _CacheDB()


# ---------------------------------------------------------------------------
# EMMAAdapter
# ---------------------------------------------------------------------------

class EMMAAdapter:
    """
    MSRB EMMA public data adapter.
    Fetches bond details, trade history, and disclosure data.
    Falls back to EDGAR for municipal disclosures when EMMA is unavailable.
    """

    TIMEOUT = 20.0
    RATE_LIMIT_SLEEP = 0.5  # seconds between requests

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def _get(self, url: str, params: Optional[Dict] = None, cache_key: Optional[str] = None) -> str:
        """GET with caching."""
        ck = cache_key or f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        cached = _cache.get(ck)
        if cached:
            return cached
        time.sleep(self.RATE_LIMIT_SLEEP)
        resp = self.session.get(url, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        text = resp.text
        _cache.set(ck, text)
        return text

    def _get_json(self, url: str, params: Optional[Dict] = None, cache_key: Optional[str] = None) -> Any:
        text = self._get(url, params, cache_key)
        return json.loads(text)

    def fetch_bond_details_html(self, cusip: str) -> Dict[str, Any]:
        """
        Scrape EMMA security details page for a CUSIP.
        Returns parsed dict with issuer, coupon, maturity, rating etc.
        """
        cusip = cusip.upper().replace("-", "")
        url = f"{EMMA_BASE}/SecurityView/SecurityDetails"
        ck = f"emma_bond_{cusip}"
        try:
            html = self._get(url, {"cusip": cusip}, cache_key=ck)
        except Exception as exc:
            logger.warning("EMMA HTML fetch failed for %s: %s", cusip, exc)
            return self._fetch_bond_edgar_fallback(cusip)

        return self._parse_security_details_html(html, cusip)

    def _parse_security_details_html(self, html: str, cusip: str) -> Dict[str, Any]:
        """Parse EMMA security details page HTML."""
        soup = BeautifulSoup(html, "html.parser")
        result: Dict[str, Any] = {"cusip": cusip}

        def extract_text(label: str) -> str:
            # Find label spans/tds and get sibling value
            for el in soup.find_all(string=re.compile(label, re.I)):
                parent = el.parent
                if parent:
                    nxt = parent.find_next_sibling()
                    if nxt:
                        return nxt.get_text(strip=True)
            return ""

        # Extract key fields from typical EMMA layout
        for tag in soup.find_all(["td", "th", "span", "div"], string=re.compile(r"issuer name", re.I)):
            sib = tag.find_next_sibling()
            if sib:
                result["issuer_name"] = sib.get_text(strip=True)
                break

        result["issuer_name"] = result.get("issuer_name") or extract_text("issuer")
        result["coupon_rate"] = self._parse_float(extract_text("coupon"))
        result["maturity_date"] = extract_text("maturity")
        result["dated_date"] = extract_text("dated date")
        result["principal_amount"] = self._parse_dollar(extract_text("principal"))
        result["state"] = self._extract_state(result.get("issuer_name", ""))

        # Rating cells
        rating_text = extract_text("S&P") or extract_text("standard")
        result["rating_sp"] = rating_text[:10] if rating_text else "NR"
        result["rating_moodys"] = extract_text("moody") or "NR"

        # Tax status
        tax_text = (extract_text("tax") or "").lower()
        if "amt" in tax_text:
            result["tax_status"] = "AMT"
        elif "taxable" in tax_text:
            result["tax_status"] = "taxable"
        else:
            result["tax_status"] = "non-AMT"

        result["use_of_proceeds"] = extract_text("use of proceeds") or extract_text("purpose")
        result["sector"] = self._classify_sector(
            result.get("use_of_proceeds", "") + " " + result.get("issuer_name", "")
        )
        result["issuer_type"] = "Revenue" if "revenue" in result["sector"] else "GO"
        result["maturity_bucket"] = self._maturity_bucket(result.get("maturity_date", ""))

        return result

    def fetch_trade_history(self, cusip: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch trade history from MSRB trade data endpoint.
        Falls back to synthetic structure if unavailable.
        """
        cusip = cusip.upper().replace("-", "")
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        ck = f"emma_trades_{cusip}_{since_date}"

        # Try EMMA/MSRB trade data API
        try:
            url = f"{EMMA_BASE}/api/IssueView/GetTradeData"
            params = {"cusip9": cusip, "startDate": since_date}
            data = self._get_json(url, params, ck)
            if isinstance(data, list):
                return [self._normalize_trade(t, cusip) for t in data]
        except Exception as exc:
            logger.debug("EMMA trade API failed: %s", exc)

        # Fallback: MSRB legacy endpoint
        try:
            url = EMMA_TRADE_API
            params = {"cusip": cusip, "format": "json", "startdate": since_date}
            text = self._get(url, params, ck + "_legacy")
            data = json.loads(text)
            if isinstance(data, (list, dict)):
                trades = data if isinstance(data, list) else data.get("trades", [])
                return [self._normalize_trade(t, cusip) for t in trades]
        except Exception as exc:
            logger.debug("MSRB legacy trade endpoint failed: %s", exc)

        return []

    def _normalize_trade(self, raw: Dict, cusip: str) -> Dict[str, Any]:
        """Normalize raw trade dict to standard schema."""
        return {
            "cusip": cusip,
            "trade_date": str(raw.get("tradeDate") or raw.get("trade_date") or ""),
            "settlement_date": str(raw.get("settlementDate") or raw.get("settlement_date") or ""),
            "par_amount": float(raw.get("parAmount") or raw.get("par_amount") or 0),
            "price": float(raw.get("price") or 0),
            "yield_pct": float(raw.get("yield") or raw.get("yield_pct") or 0),
            "trade_type": self._classify_trade_type(raw),
            "dealer_id": str(raw.get("dealerId") or raw.get("dealer_id") or ""),
        }

    def _classify_trade_type(self, raw: Dict) -> str:
        side = str(raw.get("buySell") or raw.get("trade_type") or "").upper()
        if side in ("B", "BUY", "CUSTOMER_BUY"):
            return "customer_buy"
        if side in ("S", "SELL", "CUSTOMER_SELL"):
            return "customer_sell"
        return "interdealer"

    def fetch_disclosures_edgar(self, issuer_name: str) -> List[Dict[str, Any]]:
        """
        Search SEC EDGAR for municipal official statements (OS/AOS forms).
        Returns list of filing metadata.
        """
        params = {
            "q": f'"{issuer_name}" municipal',
            "dateRange": "custom",
            "startdt": (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d"),
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "forms": "OS,AOS",
            "_source": "hits.hits._source",
        }
        ck = f"edgar_muni_{issuer_name[:40]}"
        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            data = self._get_json(url, params, ck)
            hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
            return [
                {
                    "filing_id": h.get("_id", ""),
                    "entity_name": h.get("_source", {}).get("entity_name", ""),
                    "file_date": h.get("_source", {}).get("file_date", ""),
                    "form_type": h.get("_source", {}).get("form_type", ""),
                    "display_date_filed": h.get("_source", {}).get("display_date_filed", ""),
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("EDGAR municipal search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_float(self, s: str) -> float:
        """Extract first float from a string."""
        m = re.search(r"[\d.]+", s or "")
        return float(m.group()) if m else 0.0

    def _parse_dollar(self, s: str) -> float:
        """Parse dollar amount string like '$1,250,000' -> 1250000."""
        s = re.sub(r"[$,\s]", "", s or "")
        m = re.search(r"[\d.]+", s)
        return float(m.group()) if m else 0.0

    def _extract_state(self, text: str) -> str:
        """Try to extract state abbreviation from issuer name."""
        text = text.upper()
        for abbr in STATES:
            patterns = [f" {abbr} ", f" {abbr},", f", {abbr}", f"({abbr})"]
            for p in patterns:
                if p in text:
                    return abbr
        return ""

    def _classify_sector(self, text: str) -> str:
        """Classify muni sector from description/issuer text."""
        t = text.lower()
        if any(w in t for w in ["hospital", "health", "medical", "clinic"]):
            return "revenue_hospital"
        if any(w in t for w in ["airport", "aviation", "air"]):
            return "revenue_airport"
        if any(w in t for w in ["water", "sewer", "wastewater"]):
            return "revenue_water"
        if any(w in t for w in ["utility", "electric", "power", "energy"]):
            return "revenue_utility"
        if any(w in t for w in ["highway", "toll", "bridge", "turnpike"]):
            return "revenue_highway"
        if any(w in t for w in ["school", "education", "university", "college"]):
            return "revenue_school"
        if any(w in t for w in ["housing", "mortgage", "hfa"]):
            return "housing"
        if any(w in t for w in ["tobacco", "settlement"]):
            return "tobacco"
        if any(w in t for w in ["industrial", "development", "ida"]):
            return "industrial_dev"
        if any(w in t for w in ["revenue", "revenue bond"]):
            return "other_revenue"
        return "general_obligation"

    def _maturity_bucket(self, maturity_date_str: str) -> str:
        """Classify maturity bucket from maturity date string."""
        if not maturity_date_str:
            return "medium"
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"):
            try:
                mat = datetime.strptime(maturity_date_str, fmt).date()
                years = (mat - date.today()).days / 365.25
                if years < 3:
                    return "short"
                if years <= 10:
                    return "medium"
                return "long"
            except ValueError:
                continue
        return "medium"

    def _fetch_bond_edgar_fallback(self, cusip: str) -> Dict[str, Any]:
        """Search EDGAR EFTS for CUSIP-tagged filings as fallback."""
        params = {"q": cusip, "forms": "OS,AOS,8-K", "_source": "hits.hits._source"}
        ck = f"edgar_cusip_{cusip}"
        try:
            data = self._get_json("https://efts.sec.gov/LATEST/search-index", params, ck)
            hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
            if hits:
                src = hits[0].get("_source", {})
                return {
                    "cusip": cusip,
                    "issuer_name": src.get("entity_name", ""),
                    "state": "",
                    "sector": "general_obligation",
                    "issuer_type": "GO",
                    "coupon_rate": 0.0,
                    "maturity_date": "",
                    "tax_status": "non-AMT",
                    "rating_sp": "NR",
                    "rating_moodys": "NR",
                    "maturity_bucket": "medium",
                }
        except Exception as exc:
            logger.debug("EDGAR CUSIP fallback failed: %s", exc)
        return {"cusip": cusip, "issuer_name": "", "sector": "general_obligation"}


# ---------------------------------------------------------------------------
# MuniBondUniverse
# ---------------------------------------------------------------------------

class MuniBondUniverse:
    """
    Universe of actively traded municipal bonds.
    Builds and maintains a representative sample across states, sectors, ratings,
    maturity buckets, and tax status classifications.
    """

    def __init__(self):
        self.adapter = EMMAAdapter()
        self._bonds: Dict[str, MuniBond] = {}

    def add_bond(self, bond: MuniBond) -> None:
        self._bonds[bond.cusip] = bond

    def get_bond(self, cusip: str) -> Optional[MuniBond]:
        return self._bonds.get(cusip.upper())

    def all_bonds(self) -> List[MuniBond]:
        return list(self._bonds.values())

    def fetch_and_register(self, cusip: str) -> Optional[MuniBond]:
        """Fetch bond from EMMA and add to universe."""
        try:
            raw = self.adapter.fetch_bond_details_html(cusip)
            bond = MuniBond(
                cusip=cusip,
                issuer_name=raw.get("issuer_name", ""),
                description=raw.get("description", ""),
                state=raw.get("state", ""),
                sector=raw.get("sector", "general_obligation"),
                issuer_type=raw.get("issuer_type", "GO"),
                coupon_rate=float(raw.get("coupon_rate") or 0),
                maturity_date=raw.get("maturity_date", ""),
                dated_date=raw.get("dated_date", ""),
                principal_amount=float(raw.get("principal_amount") or 0),
                tax_status=raw.get("tax_status", "non-AMT"),
                rating_sp=raw.get("rating_sp", "NR"),
                rating_moodys=raw.get("rating_moodys", "NR"),
                use_of_proceeds=raw.get("use_of_proceeds", ""),
                maturity_bucket=raw.get("maturity_bucket", "medium"),
            )
            self.add_bond(bond)
            return bond
        except Exception as exc:
            logger.error("Failed to fetch bond %s: %s", cusip, exc)
            return None

    def filter_by(
        self,
        state: Optional[str] = None,
        sector: Optional[str] = None,
        issuer_type: Optional[str] = None,
        maturity_bucket: Optional[str] = None,
        tax_status: Optional[str] = None,
        min_rating_sp: Optional[str] = None,
    ) -> List[MuniBond]:
        """Filter universe bonds by multiple criteria."""
        _RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                         "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
                         "B+", "B", "B-", "CCC", "CC", "C", "D", "NR"]

        def rating_rank(r: str) -> int:
            r = r.upper().strip()
            return _RATING_ORDER.index(r) if r in _RATING_ORDER else len(_RATING_ORDER)

        results = list(self._bonds.values())
        if state:
            results = [b for b in results if b.state.upper() == state.upper()]
        if sector:
            results = [b for b in results if b.sector == sector]
        if issuer_type:
            results = [b for b in results if b.issuer_type == issuer_type]
        if maturity_bucket:
            results = [b for b in results if b.maturity_bucket == maturity_bucket]
        if tax_status:
            results = [b for b in results if b.tax_status == tax_status]
        if min_rating_sp:
            min_rank = rating_rank(min_rating_sp)
            results = [b for b in results if rating_rank(b.rating_sp) <= min_rank]
        return results

    def summary_stats(self) -> Dict[str, Any]:
        """Return universe summary statistics."""
        bonds = list(self._bonds.values())
        if not bonds:
            return {"count": 0}
        states = {}
        sectors = {}
        for b in bonds:
            states[b.state] = states.get(b.state, 0) + 1
            sectors[b.sector] = sectors.get(b.sector, 0) + 1
        yields = [b.last_trade_yield for b in bonds if b.last_trade_yield]
        return {
            "count": len(bonds),
            "by_state": states,
            "by_sector": sectors,
            "avg_yield": sum(yields) / len(yields) if yields else None,
            "maturity_distribution": {
                "short": sum(1 for b in bonds if b.maturity_bucket == "short"),
                "medium": sum(1 for b in bonds if b.maturity_bucket == "medium"),
                "long": sum(1 for b in bonds if b.maturity_bucket == "long"),
            },
        }


# ---------------------------------------------------------------------------
# MuniYieldCalculator
# ---------------------------------------------------------------------------

class MuniYieldCalculator:
    """
    Yield analytics for municipal bonds.
    Provides tax-equivalent yield, spread to Treasury, and after-tax comparisons.
    """

    # Federal marginal tax brackets 2025 (simplified top brackets)
    FEDERAL_TAX_BRACKETS = [
        (0.37,    609350),
        (0.35,    243725),
        (0.32,    191950),
        (0.24,    100525),
        (0.22,     47150),
        (0.12,     11925),
        (0.10,         0),
    ]

    def __init__(self):
        self._fred_cache: Dict[str, List[Tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Core yield calculations
    # ------------------------------------------------------------------

    def tax_equivalent_yield(
        self,
        muni_yield: float,
        federal_tax_rate: float,
        state_tax_rate: float = 0.0,
        state: str = "",
    ) -> float:
        """
        Tax-equivalent yield = muni_yield / (1 - federal_tax - state_tax).
        Assumes muni is exempt from both federal and state taxes (home-state bond).
        """
        if state and not state_tax_rate:
            state_tax_rate = STATE_TAX_RATES.get(state.upper(), 0.0)
        combined_rate = federal_tax_rate + state_tax_rate - (federal_tax_rate * state_tax_rate)
        combined_rate = min(combined_rate, 0.95)  # cap at 95%
        if combined_rate >= 1.0:
            return float("inf")
        return muni_yield / (1.0 - combined_rate)

    def tax_equivalent_yield_out_of_state(
        self,
        muni_yield: float,
        federal_tax_rate: float,
        state_tax_rate: float,
    ) -> float:
        """
        Out-of-state muni: exempt from federal only, not state.
        TEY = muni_yield / (1 - federal_tax_rate).
        """
        if federal_tax_rate >= 1.0:
            return float("inf")
        return muni_yield / (1.0 - federal_tax_rate)

    def after_tax_yield(self, gross_yield: float, tax_rate: float) -> float:
        """After-tax yield for taxable instrument: gross * (1 - tax_rate)."""
        return gross_yield * (1.0 - tax_rate)

    def breakeven_tax_rate(self, muni_yield: float, taxable_yield: float) -> float:
        """
        Tax rate at which muni and taxable yields are equivalent.
        breakeven = 1 - (muni_yield / taxable_yield)
        """
        if taxable_yield <= 0:
            return 0.0
        return max(0.0, 1.0 - (muni_yield / taxable_yield))

    def federal_tax_rate_for_income(self, annual_income: float) -> float:
        """Estimate marginal federal tax rate for given income."""
        for rate, threshold in self.FEDERAL_TAX_BRACKETS:
            if annual_income >= threshold:
                return rate
        return 0.10

    def spread_to_treasury(
        self,
        muni_yield: float,
        maturity_years: float,
        treasury_curve: Optional[Dict[float, float]] = None,
    ) -> float:
        """Spread in percentage points: muni_yield - treasury_yield."""
        tsy_yield = self._interpolate_treasury(maturity_years, treasury_curve)
        return muni_yield - tsy_yield

    def muni_treasury_ratio(self, muni_yield: float, maturity_years: float,
                             treasury_curve: Optional[Dict[float, float]] = None) -> float:
        """Muni/Treasury ratio as percentage: muni_yield / treasury_yield * 100."""
        tsy = self._interpolate_treasury(maturity_years, treasury_curve)
        if tsy <= 0:
            return 0.0
        return (muni_yield / tsy) * 100.0

    def compare_after_tax(
        self,
        muni_yield: float,
        corporate_yield: float,
        treasury_yield: float,
        federal_tax_rate: float,
        state_tax_rate: float = 0.0,
        state: str = "",
        in_state_bond: bool = True,
    ) -> Dict[str, float]:
        """
        Compare after-tax returns across muni, corporate, and Treasury.
        Returns dict with after-tax yields and spreads.
        """
        if state and not state_tax_rate:
            state_tax_rate = STATE_TAX_RATES.get(state.upper(), 0.0)

        # Muni: typically exempt from federal + state (if in-state)
        muni_after_tax = muni_yield  # munis not taxed at federal level
        if not in_state_bond:
            muni_after_tax = muni_yield  # still exempt from federal

        # Corporate: taxed at federal + state
        corp_combined = federal_tax_rate + state_tax_rate - (federal_tax_rate * state_tax_rate)
        corp_after_tax = self.after_tax_yield(corporate_yield, corp_combined)

        # Treasury: taxed at federal only (state-exempt)
        tsy_after_tax = self.after_tax_yield(treasury_yield, federal_tax_rate)

        tey_vs_corp = self.tax_equivalent_yield(muni_yield, federal_tax_rate, state_tax_rate, state)
        breakeven = self.breakeven_tax_rate(muni_yield, corporate_yield)

        return {
            "muni_after_tax": round(muni_after_tax, 4),
            "corporate_after_tax": round(corp_after_tax, 4),
            "treasury_after_tax": round(tsy_after_tax, 4),
            "muni_vs_corporate_bps": round((muni_after_tax - corp_after_tax) * 100, 2),
            "muni_vs_treasury_bps": round((muni_after_tax - tsy_after_tax) * 100, 2),
            "tax_equivalent_yield": round(tey_vs_corp, 4),
            "breakeven_tax_rate": round(breakeven, 4),
        }

    def duration_approx(
        self,
        coupon_rate: float,
        ytm: float,
        maturity_years: float,
        freq: int = 2,
    ) -> Tuple[float, float]:
        """
        Compute modified duration and convexity analytically.
        Returns (modified_duration, convexity).
        """
        n = int(round(maturity_years * freq))
        if n <= 0:
            return 0.0, 0.0
        c = coupon_rate / freq / 100.0
        y = ytm / freq / 100.0

        if abs(y) < 1e-12:
            mac = maturity_years
            return mac, 0.0

        # Sum PV-weighted times for Macaulay duration
        mac_num = 0.0
        total_pv = 0.0
        conv_num = 0.0

        for t in range(1, n + 1):
            df = (1.0 + y) ** t
            pv_t = c / df
            mac_num += t * pv_t
            conv_num += t * (t + 1) * pv_t
            total_pv += pv_t

        pv_face = 1.0 / (1.0 + y) ** n
        mac_num += n * pv_face
        conv_num += n * (n + 1) * pv_face
        total_pv += pv_face

        macaulay = (mac_num / total_pv) / freq
        modified = macaulay / (1.0 + y)
        convexity = (conv_num / total_pv) / ((1.0 + y) ** 2 * freq ** 2)
        return modified, convexity

    def _interpolate_treasury(
        self, maturity_years: float, curve: Optional[Dict[float, float]] = None
    ) -> float:
        """Linear interpolation on Treasury curve."""
        if not curve:
            # Default approximate US Treasury curve (2025)
            curve = {0.25: 5.20, 0.5: 5.18, 1.0: 5.10, 2.0: 4.85,
                     5.0: 4.50, 7.0: 4.45, 10.0: 4.40, 20.0: 4.65, 30.0: 4.55}
        tenors = sorted(curve.keys())
        yields = [curve[t] for t in tenors]

        if maturity_years <= tenors[0]:
            return yields[0]
        if maturity_years >= tenors[-1]:
            return yields[-1]

        for i in range(len(tenors) - 1):
            if tenors[i] <= maturity_years <= tenors[i + 1]:
                t0, t1 = tenors[i], tenors[i + 1]
                y0, y1 = yields[i], yields[i + 1]
                w = (maturity_years - t0) / (t1 - t0)
                return y0 + w * (y1 - y0)
        return yields[-1]

    def yield_curve_by_rating(
        self,
        state: str = "",
        tenors: Optional[List[float]] = None,
    ) -> Dict[str, List[float]]:
        """
        Return approximate muni yield curves by rating category.
        Based on historical muni/Treasury ratio averages.
        """
        if tenors is None:
            tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        # Approximate Treasury yields (2025 baseline)
        tsy_curve = {0.25: 5.20, 0.5: 5.18, 1.0: 5.10, 2.0: 4.85,
                     5.0: 4.50, 7.0: 4.45, 10.0: 4.40, 20.0: 4.65, 30.0: 4.55}

        # Muni/Treasury ratios by rating (typical)
        ratios = {
            "AAA": {1: 0.62, 5: 0.68, 10: 0.72, 30: 0.84},
            "AA":  {1: 0.65, 5: 0.72, 10: 0.77, 30: 0.88},
            "A":   {1: 0.71, 5: 0.79, 10: 0.85, 30: 0.96},
            "BBB": {1: 0.85, 5: 0.95, 10: 1.05, 30: 1.18},
        }

        result: Dict[str, List[float]] = {}
        for rating, ratio_map in ratios.items():
            yields_list = []
            for t in tenors:
                tsy = self._interpolate_treasury(t, tsy_curve)
                # Interpolate ratio
                ratio_tenors = sorted(ratio_map.keys())
                if t <= ratio_tenors[0]:
                    ratio = ratio_map[ratio_tenors[0]]
                elif t >= ratio_tenors[-1]:
                    ratio = ratio_map[ratio_tenors[-1]]
                else:
                    for i in range(len(ratio_tenors) - 1):
                        if ratio_tenors[i] <= t <= ratio_tenors[i + 1]:
                            r0 = ratio_map[ratio_tenors[i]]
                            r1 = ratio_map[ratio_tenors[i + 1]]
                            w = (t - ratio_tenors[i]) / (ratio_tenors[i + 1] - ratio_tenors[i])
                            ratio = r0 + w * (r1 - r0)
                            break
                    else:
                        ratio = 0.75
                yields_list.append(round(tsy * ratio / 100.0, 4))
            result[rating] = yields_list

        result["treasury"] = [
            round(self._interpolate_treasury(t, tsy_curve) / 100.0, 4) for t in tenors
        ]
        return result


# ---------------------------------------------------------------------------
# MuniCreditAnalyzer
# ---------------------------------------------------------------------------

class MuniCreditAnalyzer:
    """
    Credit analysis for municipal bonds.
    Analyzes DSCR, fund balances, pension liabilities, revenue coverage.
    Uses EDGAR CAFR disclosures where available.
    """

    def __init__(self):
        self.adapter = EMMAAdapter()

    def analyze_go_bond(
        self,
        state: str,
        issuer_name: str,
        fund_balance: Optional[float] = None,
        annual_expenditures: Optional[float] = None,
        unfunded_pension: Optional[float] = None,
        total_budget: Optional[float] = None,
        debt_outstanding: Optional[float] = None,
        tax_base_value: Optional[float] = None,
    ) -> CreditAnalysis:
        """
        Credit analysis for general obligation bonds.
        Computes fund balance ratio, pension overhang, debt ratios.
        """
        risk_factors: List[str] = []
        strengths: List[str] = []
        score = 60.0  # Start at 60/100

        fund_balance_ratio = None
        if fund_balance is not None and annual_expenditures and annual_expenditures > 0:
            fund_balance_ratio = fund_balance / annual_expenditures
            if fund_balance_ratio >= 0.20:
                score += 15
                strengths.append(f"Strong fund balance ratio: {fund_balance_ratio:.1%}")
            elif fund_balance_ratio >= 0.10:
                score += 5
                strengths.append(f"Adequate fund balance ratio: {fund_balance_ratio:.1%}")
            elif fund_balance_ratio < 0.05:
                score -= 20
                risk_factors.append(f"Very low fund balance ratio: {fund_balance_ratio:.1%}")
            else:
                score -= 5
                risk_factors.append(f"Below-average fund balance ratio: {fund_balance_ratio:.1%}")

        pension_liability_pct = None
        if unfunded_pension is not None and total_budget and total_budget > 0:
            pension_liability_pct = unfunded_pension / total_budget
            if pension_liability_pct > 2.0:
                score -= 25
                risk_factors.append(f"Severe pension overhang: {pension_liability_pct:.1%} of budget")
            elif pension_liability_pct > 1.0:
                score -= 15
                risk_factors.append(f"Significant pension liability: {pension_liability_pct:.1%} of budget")
            elif pension_liability_pct < 0.3:
                score += 10
                strengths.append(f"Low pension burden: {pension_liability_pct:.1%} of budget")

        debt_to_tax_base = None
        if debt_outstanding is not None and tax_base_value and tax_base_value > 0:
            debt_to_tax_base = debt_outstanding / tax_base_value
            if debt_to_tax_base > 0.10:
                score -= 10
                risk_factors.append(f"High debt burden vs tax base: {debt_to_tax_base:.1%}")
            elif debt_to_tax_base < 0.03:
                score += 8
                strengths.append(f"Low debt burden vs tax base: {debt_to_tax_base:.1%}")

        # State-level adjustments
        high_tax_states = {"CA", "NY", "IL", "NJ", "CT", "MA"}
        if state.upper() in high_tax_states:
            score -= 5
            risk_factors.append("High-tax state: increased fiscal pressure risk")

        # Historical default rate adjustment
        default_rate = SECTOR_DEFAULT_RATES.get("general_obligation", 0.0018)
        if default_rate < 0.002:
            score += 5
            strengths.append("Low historical default rate for GO bonds")

        score = max(0.0, min(100.0, score))
        implied_rating = self._score_to_rating(score)

        return CreditAnalysis(
            state=state,
            issuer_type="GO",
            sector="general_obligation",
            fund_balance_ratio=fund_balance_ratio,
            pension_liability_pct=pension_liability_pct,
            historical_default_rate=default_rate,
            credit_score=round(score, 1),
            credit_rating_implied=implied_rating,
            risk_factors=risk_factors,
            strengths=strengths,
        )

    def analyze_revenue_bond(
        self,
        state: str,
        sector: str,
        pledged_revenues: Optional[float] = None,
        annual_debt_service: Optional[float] = None,
        operating_expenses: Optional[float] = None,
        reserve_fund: Optional[float] = None,
    ) -> CreditAnalysis:
        """
        Credit analysis for revenue bonds.
        Focuses on DSCR and revenue coverage ratios.
        """
        risk_factors: List[str] = []
        strengths: List[str] = []
        score = 55.0

        dscr = None
        revenue_coverage = None
        if pledged_revenues is not None and annual_debt_service and annual_debt_service > 0:
            dscr = pledged_revenues / annual_debt_service
            revenue_coverage = dscr
            if dscr >= 2.0:
                score += 20
                strengths.append(f"Excellent DSCR: {dscr:.2f}x")
            elif dscr >= 1.50:
                score += 12
                strengths.append(f"Strong DSCR: {dscr:.2f}x")
            elif dscr >= 1.25:
                score += 5
                strengths.append(f"Adequate DSCR: {dscr:.2f}x")
            elif dscr < 1.10:
                score -= 25
                risk_factors.append(f"Weak DSCR: {dscr:.2f}x — near break-even")
            else:
                score -= 10
                risk_factors.append(f"Below-average DSCR: {dscr:.2f}x")

        if reserve_fund is not None and annual_debt_service and annual_debt_service > 0:
            reserve_ratio = reserve_fund / annual_debt_service
            if reserve_ratio >= 1.0:
                score += 8
                strengths.append(f"Strong debt service reserve: {reserve_ratio:.2f}x annual DS")
            elif reserve_ratio < 0.5:
                score -= 5
                risk_factors.append("Below-average reserve fund coverage")

        # Sector-specific adjustments
        sector_adjustments = {
            "revenue_utility": +8,
            "revenue_water": +10,
            "revenue_airport": +5,
            "revenue_highway": +5,
            "revenue_hospital": -5,
            "revenue_school": +8,
            "housing": -3,
            "tobacco": -20,
            "industrial_dev": -15,
        }
        score += sector_adjustments.get(sector, 0)
        if sector in ("tobacco", "industrial_dev"):
            risk_factors.append(f"High-risk sector: {sector}")
        elif sector in ("revenue_water", "revenue_utility"):
            strengths.append(f"Essential service sector: {sector}")

        default_rate = SECTOR_DEFAULT_RATES.get(sector, 0.01)
        score = max(0.0, min(100.0, score))
        implied_rating = self._score_to_rating(score)

        return CreditAnalysis(
            state=state,
            issuer_type="Revenue",
            sector=sector,
            dscr=dscr,
            revenue_coverage=revenue_coverage,
            historical_default_rate=default_rate,
            credit_score=round(score, 1),
            credit_rating_implied=implied_rating,
            risk_factors=risk_factors,
            strengths=strengths,
        )

    def state_fiscal_summary(self, state: str) -> Dict[str, Any]:
        """
        Return high-level fiscal summary for a state.
        Uses hardcoded structural data from public Pew/NASBO reports.
        """
        # Pension funding ratios (approximate, 2024 data from Pew Charitable Trusts)
        pension_funding = {
            "WI": 0.99, "SD": 0.96, "TN": 0.95, "ND": 0.93, "ID": 0.91,
            "WY": 0.88, "OR": 0.84, "FL": 0.82, "GA": 0.80, "WA": 0.78,
            "TX": 0.76, "OH": 0.75, "VA": 0.74, "NC": 0.73, "CO": 0.72,
            "MO": 0.70, "NY": 0.68, "CA": 0.66, "MA": 0.64, "PA": 0.63,
            "CT": 0.52, "IL": 0.44, "NJ": 0.38, "KY": 0.36,
        }

        # Credit ratings (S&P, approximate 2025)
        state_ratings = {
            "ND": "AAA", "GA": "AAA", "MD": "AAA", "MO": "AAA", "TN": "AAA",
            "UT": "AAA", "VA": "AAA", "FL": "AA+", "TX": "AA", "CA": "AA-",
            "NY": "AA", "PA": "A+", "IL": "BBB+", "NJ": "BBB+",
            "CT": "A-", "MA": "AA", "WA": "AA+", "CO": "AA",
        }

        pension_ratio = pension_funding.get(state.upper(), 0.72)
        rating = state_ratings.get(state.upper(), "AA")
        state_full = STATES.get(state.upper(), state)

        risk_level = "low" if pension_ratio > 0.85 else "medium" if pension_ratio > 0.65 else "high"
        default_rate = SECTOR_DEFAULT_RATES["general_obligation"]

        return {
            "state": state.upper(),
            "state_name": state_full,
            "sp_rating": rating,
            "pension_funding_ratio": pension_ratio,
            "pension_risk_level": risk_level,
            "top_marginal_income_tax": STATE_TAX_RATES.get(state.upper(), 0.05),
            "historical_go_default_rate": default_rate,
            "recommended_sectors": self._recommend_sectors(state.upper(), pension_ratio),
        }

    def _recommend_sectors(self, state: str, pension_ratio: float) -> List[str]:
        """Recommend muni sectors based on state fiscal health."""
        recs = ["revenue_water", "revenue_utility"]
        if pension_ratio > 0.80:
            recs.append("general_obligation")
        if state in ("TX", "FL", "GA", "TN", "ND"):
            recs.append("revenue_school")
        return recs

    def _score_to_rating(self, score: float) -> str:
        """Convert credit score 0-100 to implied rating."""
        if score >= 90:
            return "AAA"
        if score >= 82:
            return "AA+"
        if score >= 75:
            return "AA"
        if score >= 68:
            return "AA-"
        if score >= 62:
            return "A+"
        if score >= 56:
            return "A"
        if score >= 50:
            return "A-"
        if score >= 44:
            return "BBB+"
        if score >= 38:
            return "BBB"
        if score >= 32:
            return "BBB-"
        if score >= 26:
            return "BB+"
        if score >= 20:
            return "BB"
        return "B"


# ---------------------------------------------------------------------------
# MuniTradeAnalyzer
# ---------------------------------------------------------------------------

class MuniTradeAnalyzer:
    """
    Trade flow analysis for municipal bonds.
    Analyzes volume, buy/sell imbalance, liquidity, and odd-lot premiums.
    """

    def __init__(self):
        self.adapter = EMMAAdapter()

    def analyze_trades(self, trades: List[MuniTrade]) -> Dict[str, Any]:
        """Comprehensive trade analysis from a list of trades."""
        if not trades:
            return {"error": "No trades provided"}

        total_par = sum(t.par_amount for t in trades)
        cust_buy_par = sum(t.par_amount for t in trades if t.trade_type == "customer_buy")
        cust_sell_par = sum(t.par_amount for t in trades if t.trade_type == "customer_sell")
        interdealer_par = sum(t.par_amount for t in trades if t.trade_type == "interdealer")

        buy_sell_imbalance = 0.0
        if cust_buy_par + cust_sell_par > 0:
            buy_sell_imbalance = (cust_buy_par - cust_sell_par) / (cust_buy_par + cust_sell_par)

        prices = [t.price for t in trades]
        yields = [t.yield_pct for t in trades if t.yield_pct > 0]

        odd_lot_trades = [t for t in trades if t.par_amount < 100_000]
        regular_trades = [t for t in trades if t.par_amount >= 100_000]
        odd_lot_premium = 0.0
        if odd_lot_trades and regular_trades:
            avg_odd_price = sum(t.price for t in odd_lot_trades) / len(odd_lot_trades)
            avg_reg_price = sum(t.price for t in regular_trades) / len(regular_trades)
            odd_lot_premium = avg_odd_price - avg_reg_price

        daily_volumes: Dict[str, float] = {}
        for t in trades:
            d = t.trade_date[:10] if t.trade_date else "unknown"
            daily_volumes[d] = daily_volumes.get(d, 0) + t.par_amount

        return {
            "total_par_traded": total_par,
            "trade_count": len(trades),
            "customer_buy_par": cust_buy_par,
            "customer_sell_par": cust_sell_par,
            "interdealer_par": interdealer_par,
            "buy_sell_imbalance": round(buy_sell_imbalance, 4),
            "imbalance_signal": "net buyer pressure" if buy_sell_imbalance > 0.1 else
                                "net seller pressure" if buy_sell_imbalance < -0.1 else "balanced",
            "avg_price": round(sum(prices) / len(prices), 4) if prices else None,
            "price_range": {"min": min(prices), "max": max(prices)} if prices else None,
            "avg_yield": round(sum(yields) / len(yields), 4) if yields else None,
            "odd_lot_count": len(odd_lot_trades),
            "odd_lot_premium_pts": round(odd_lot_premium, 4),
            "daily_volume": daily_volumes,
            "largest_trade_par": max(t.par_amount for t in trades),
        }

    def liquidity_score(
        self,
        par_traded_30d: float,
        outstanding_par: float,
        trade_count_30d: int,
    ) -> Dict[str, Any]:
        """
        Compute liquidity score based on turnover and trade frequency.
        Returns score 0-100 and liquidity tier.
        """
        turnover = 0.0
        if outstanding_par > 0:
            turnover = par_traded_30d / outstanding_par

        score = 0.0
        # Turnover component (0-60 points)
        if turnover >= 0.10:
            score += 60
        elif turnover >= 0.05:
            score += 40
        elif turnover >= 0.02:
            score += 25
        elif turnover >= 0.005:
            score += 10

        # Trade frequency component (0-40 points)
        if trade_count_30d >= 20:
            score += 40
        elif trade_count_30d >= 10:
            score += 28
        elif trade_count_30d >= 5:
            score += 15
        elif trade_count_30d >= 2:
            score += 8

        if score >= 75:
            tier = "liquid"
        elif score >= 45:
            tier = "semi-liquid"
        elif score >= 20:
            tier = "illiquid"
        else:
            tier = "very illiquid"

        return {
            "liquidity_score": round(score, 1),
            "liquidity_tier": tier,
            "30d_turnover": round(turnover, 4),
            "trade_count_30d": trade_count_30d,
            "par_traded_30d": par_traded_30d,
        }

    def volume_by_state(self, trades: List[MuniTrade], universe: "MuniBondUniverse") -> Dict[str, float]:
        """Aggregate trade volume by state using universe bond data."""
        result: Dict[str, float] = {}
        for t in trades:
            bond = universe.get_bond(t.cusip)
            state = bond.state if bond else "Unknown"
            result[state] = result.get(state, 0) + t.par_amount
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def volume_by_sector(self, trades: List[MuniTrade], universe: "MuniBondUniverse") -> Dict[str, float]:
        """Aggregate trade volume by sector."""
        result: Dict[str, float] = {}
        for t in trades:
            bond = universe.get_bond(t.cusip)
            sector = bond.sector if bond else "unknown"
            result[sector] = result.get(sector, 0) + t.par_amount
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# MuniScreener
# ---------------------------------------------------------------------------

class MuniScreenerParams(BaseModel):
    min_yield: float = 0.0
    max_yield: float = 20.0
    min_rating_sp: str = "D"
    states: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    maturity_min_years: float = 0.0
    maturity_max_years: float = 50.0
    tax_status: Optional[Literal["non-AMT", "AMT", "taxable"]] = None
    min_par_amount: float = 0.0
    max_par_amount: float = 1e12
    issuer_type: Optional[Literal["GO", "Revenue", "Other"]] = None
    sort_by: Literal["yield", "maturity", "rating", "par_amount"] = "yield"
    sort_desc: bool = True
    limit: int = 50


class MuniScreener:
    """
    Screen municipal bonds by multiple criteria.
    Returns ranked, filtered results from universe.
    """

    _RATING_ORDER = [
        "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
        "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
        "B+", "B", "B-", "CCC", "CC", "C", "D", "NR",
    ]

    def __init__(self, universe: MuniBondUniverse, calc: MuniYieldCalculator):
        self.universe = universe
        self.calc = calc

    def _rating_rank(self, rating: str) -> int:
        r = rating.upper().strip()
        if r in self._RATING_ORDER:
            return self._RATING_ORDER.index(r)
        return len(self._RATING_ORDER)

    def _maturity_years(self, bond: MuniBond) -> float:
        if not bond.maturity_date:
            return 10.0
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"):
            try:
                mat = datetime.strptime(bond.maturity_date, fmt).date()
                return max(0.0, (mat - date.today()).days / 365.25)
            except ValueError:
                continue
        return 10.0

    def screen(self, params: MuniScreenerParams) -> ScreenerResult:
        """Apply screening filters and return ranked bonds."""
        bonds = self.universe.all_bonds()
        min_rank = self._rating_rank(params.min_rating_sp)

        filtered = []
        for b in bonds:
            yield_val = b.last_trade_yield or 0.0
            if not (params.min_yield <= yield_val <= params.max_yield):
                continue
            if self._rating_rank(b.rating_sp) > min_rank:
                continue
            if params.states and b.state.upper() not in [s.upper() for s in params.states]:
                continue
            if params.sectors and b.sector not in params.sectors:
                continue
            mat_yrs = self._maturity_years(b)
            if not (params.maturity_min_years <= mat_yrs <= params.maturity_max_years):
                continue
            if params.tax_status and b.tax_status != params.tax_status:
                continue
            if not (params.min_par_amount <= b.outstanding_par <= params.max_par_amount):
                continue
            if params.issuer_type and b.issuer_type != params.issuer_type:
                continue
            filtered.append((b, mat_yrs))

        # Sort
        def sort_key(item: Tuple[MuniBond, float]) -> float:
            b, mat = item
            if params.sort_by == "yield":
                return b.last_trade_yield or 0.0
            if params.sort_by == "maturity":
                return mat
            if params.sort_by == "rating":
                return -self._rating_rank(b.rating_sp)  # higher rating = better sort
            if params.sort_by == "par_amount":
                return b.outstanding_par
            return 0.0

        filtered.sort(key=sort_key, reverse=params.sort_desc)
        result_bonds = [item[0] for item in filtered[: params.limit]]

        yields = [b.last_trade_yield for b in result_bonds if b.last_trade_yield]
        yield_stats: Dict[str, float] = {}
        if yields:
            yield_stats = {
                "min": round(min(yields), 4),
                "max": round(max(yields), 4),
                "avg": round(sum(yields) / len(yields), 4),
                "median": round(sorted(yields)[len(yields) // 2], 4),
            }

        return ScreenerResult(
            bonds=result_bonds,
            total=len(filtered),
            filters_applied=params.model_dump(),
            yield_stats=yield_stats,
        )

    def tax_aware_rank(
        self,
        federal_tax_rate: float,
        state: str,
        bonds: Optional[List[MuniBond]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank bonds by tax-equivalent yield for given investor tax profile.
        """
        if bonds is None:
            bonds = self.universe.all_bonds()

        state_tax = STATE_TAX_RATES.get(state.upper(), 0.0)
        results = []
        for b in bonds:
            if not b.last_trade_yield:
                continue
            tey = self.calc.tax_equivalent_yield(b.last_trade_yield, federal_tax_rate, state_tax, state)
            mat_yrs = self._maturity_years(b)
            tsy_spread = self.calc.spread_to_treasury(b.last_trade_yield, mat_yrs)
            results.append({
                "cusip": b.cusip,
                "issuer_name": b.issuer_name,
                "state": b.state,
                "sector": b.sector,
                "muni_yield": b.last_trade_yield,
                "tax_equivalent_yield": round(tey, 4),
                "maturity_years": round(mat_yrs, 2),
                "rating_sp": b.rating_sp,
                "tax_status": b.tax_status,
                "spread_to_treasury_bps": round(tsy_spread * 100, 2),
            })

        results.sort(key=lambda x: x["tax_equivalent_yield"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

muni_router = APIRouter(prefix="/muni", tags=["Municipal Bonds"])

_universe = MuniBondUniverse()
_adapter = EMMAAdapter()
_calc = MuniYieldCalculator()
_credit = MuniCreditAnalyzer()
_trade_analyzer = MuniTradeAnalyzer()
_screener = MuniScreener(_universe, _calc)


@muni_router.get("/search")
def search_munis(
    issuer: str = Query("", description="Issuer name keyword"),
    state: str = Query("", description="State abbreviation (e.g. CA)"),
    sector: str = Query("", description="Sector: general_obligation, revenue_hospital, etc."),
    maturity_bucket: str = Query("", description="short/medium/long"),
    tax_status: str = Query("", description="non-AMT/AMT/taxable"),
    limit: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    """Search municipal bond universe by various criteria."""
    bonds = _universe.filter_by(
        state=state or None,
        sector=sector or None,
        maturity_bucket=maturity_bucket or None,
        tax_status=tax_status or None,
    )
    if issuer:
        kw = issuer.lower()
        bonds = [b for b in bonds if kw in b.issuer_name.lower()]
    bonds = bonds[:limit]
    return {
        "count": len(bonds),
        "bonds": [b.model_dump() for b in bonds],
        "universe_size": len(_universe.all_bonds()),
    }


@muni_router.get("/bond/{cusip}")
def get_bond_detail(cusip: str) -> Dict[str, Any]:
    """
    Fetch detailed bond data from MSRB EMMA for a given CUSIP.
    Returns bond details, recent trades, and credit context.
    """
    cusip = cusip.upper().replace("-", "")

    # Check universe first
    bond = _universe.get_bond(cusip)
    if bond is None:
        bond = _universe.fetch_and_register(cusip)
    if bond is None:
        raise HTTPException(status_code=404, detail=f"Bond {cusip} not found on EMMA")

    # Fetch trades
    raw_trades = _adapter.fetch_trade_history(cusip, days_back=30)
    trades = [MuniTrade(**t) for t in raw_trades if len(raw_trades) > 0 and isinstance(t, dict)]

    trade_analysis = _trade_analyzer.analyze_trades(trades) if trades else {}

    return {
        "bond": bond.model_dump(),
        "trade_analysis": trade_analysis,
        "recent_trades": [t.model_dump() for t in trades[:10]],
    }


@muni_router.get("/tax-equivalent")
def tax_equivalent_yield(
    cusip: str = Query("", description="CUSIP (optional, uses bond yield if known)"),
    muni_yield: float = Query(..., description="Municipal bond yield as decimal (e.g. 0.035 = 3.5%)"),
    federal_tax_rate: float = Query(..., description="Federal marginal tax rate (e.g. 0.37)"),
    state: str = Query("", description="State abbreviation for state tax lookup"),
    state_tax_rate: float = Query(0.0, description="Override state tax rate"),
    maturity_years: float = Query(10.0, description="Years to maturity for Treasury comparison"),
    corporate_yield: float = Query(0.0, description="Corporate bond yield for comparison"),
    treasury_yield: float = Query(0.0, description="Treasury yield (0 = auto-interpolate)"),
    in_state_bond: bool = Query(True, description="Is this an in-state bond?"),
) -> TaxEquivResult:
    """Compute tax-equivalent yield and after-tax comparisons."""
    eff_state_tax = state_tax_rate or STATE_TAX_RATES.get(state.upper(), 0.0)
    tey = _calc.tax_equivalent_yield(muni_yield * 100, federal_tax_rate, eff_state_tax, state)

    tsy_yield = treasury_yield or (_calc._interpolate_treasury(maturity_years) / 100.0)
    spread_bps = (muni_yield - tsy_yield) * 10000  # convert to bps

    breakeven = _calc.breakeven_tax_rate(muni_yield, tsy_yield)

    corp_after_tax = None
    muni_vs_corp_bps = None
    if corporate_yield > 0:
        combined = federal_tax_rate + eff_state_tax - federal_tax_rate * eff_state_tax
        corp_after_tax = corporate_yield * (1 - combined)
        muni_vs_corp_bps = (muni_yield - corp_after_tax) * 10000

    tsy_after_tax = tsy_yield * (1 - federal_tax_rate)
    muni_vs_tsy_bps = (muni_yield - tsy_after_tax) * 10000

    return TaxEquivResult(
        cusip=cusip,
        muni_yield=round(muni_yield, 6),
        federal_tax_rate=federal_tax_rate,
        state_tax_rate=eff_state_tax,
        state=state.upper(),
        tax_equiv_yield=round(tey / 100.0, 6),
        treasury_yield_same_maturity=round(tsy_yield, 6),
        spread_to_treasury_bps=round(spread_bps, 2),
        after_tax_corporate_rate=round(corp_after_tax, 6) if corp_after_tax else None,
        after_tax_treasury_rate=round(tsy_after_tax, 6),
        muni_advantage_vs_corporate_bps=round(muni_vs_corp_bps, 2) if muni_vs_corp_bps else None,
        muni_advantage_vs_treasury_bps=round(muni_vs_tsy_bps, 2),
        breakeven_tax_rate=round(breakeven, 4),
    )


@muni_router.post("/screener")
def screen_munis(params: MuniScreenerParams) -> ScreenerResult:
    """Screen munis by yield, rating, state, sector, maturity, and tax status."""
    return _screener.screen(params)


@muni_router.get("/credit-analysis/{state}")
def credit_analysis_by_state(
    state: str,
    sector: str = Query("general_obligation"),
    fund_balance: Optional[float] = Query(None),
    annual_expenditures: Optional[float] = Query(None),
    unfunded_pension: Optional[float] = Query(None),
    total_budget: Optional[float] = Query(None),
    pledged_revenues: Optional[float] = Query(None),
    annual_debt_service: Optional[float] = Query(None),
) -> Dict[str, Any]:
    """
    Credit analysis for a state's municipal bonds.
    Returns fiscal health metrics, implied ratings, and sector recommendations.
    """
    state = state.upper()
    fiscal = _credit.state_fiscal_summary(state)

    if sector == "general_obligation":
        analysis = _credit.analyze_go_bond(
            state=state,
            issuer_name=STATES.get(state, state),
            fund_balance=fund_balance,
            annual_expenditures=annual_expenditures,
            unfunded_pension=unfunded_pension,
            total_budget=total_budget,
        )
    else:
        analysis = _credit.analyze_revenue_bond(
            state=state,
            sector=sector,
            pledged_revenues=pledged_revenues,
            annual_debt_service=annual_debt_service,
        )

    return {
        "state_fiscal_summary": fiscal,
        "credit_analysis": analysis.model_dump(),
        "sector_default_rates": SECTOR_DEFAULT_RATES,
    }


@muni_router.get("/yield-curve")
def muni_yield_curve(
    state: str = Query("", description="State for state-specific curve adjustment"),
    rating: str = Query("AA", description="Rating category: AAA/AA/A/BBB"),
) -> MuniYieldCurve:
    """
    Return municipal yield curve for a given rating and state.
    Compares to Treasury curve and provides muni/Treasury ratios.
    """
    tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    curves = _calc.yield_curve_by_rating(state=state, tenors=tenors)

    tsy_curve = {0.25: 5.20, 0.5: 5.18, 1.0: 5.10, 2.0: 4.85,
                 5.0: 4.50, 7.0: 4.45, 10.0: 4.40, 20.0: 4.65, 30.0: 4.55}
    tsy_yields = [round(_calc._interpolate_treasury(t, tsy_curve) / 100.0, 4) for t in tenors]

    selected = curves.get(rating.upper(), curves.get("AA", []))
    ratios = [
        round(selected[i] / tsy_yields[i] * 100.0, 2) if tsy_yields[i] > 0 else 0.0
        for i in range(len(tenors))
    ]

    return MuniYieldCurve(
        state=state.upper(),
        as_of_date=date.today().isoformat(),
        tenors=tenors,
        yields=selected,
        aaa_yields=curves.get("AAA", []),
        aa_yields=curves.get("AA", []),
        a_yields=curves.get("A", []),
        bbb_yields=curves.get("BBB", []),
        treasury_yields=tsy_yields,
        muni_treasury_ratios=ratios,
    )


@muni_router.get("/tax-aware-rank")
def tax_aware_rank(
    federal_tax_rate: float = Query(..., description="Federal marginal tax rate"),
    state: str = Query("NY", description="State abbreviation"),
    min_yield: float = Query(0.0),
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Rank universe bonds by tax-equivalent yield for investor tax profile."""
    params = MuniScreenerParams(min_yield=min_yield * 100, limit=1000)
    screened = _screener.screen(params)
    ranked = _screener.tax_aware_rank(federal_tax_rate, state, screened.bonds)
    return {
        "federal_tax_rate": federal_tax_rate,
        "state": state.upper(),
        "state_tax_rate": STATE_TAX_RATES.get(state.upper(), 0.0),
        "ranked_bonds": ranked[:limit],
    }


@muni_router.get("/disclosures/{issuer_name}")
def get_disclosures(issuer_name: str) -> Dict[str, Any]:
    """Fetch EDGAR official statement disclosures for a municipal issuer."""
    filings = _adapter.fetch_disclosures_edgar(issuer_name)
    return {
        "issuer": issuer_name,
        "filing_count": len(filings),
        "filings": filings,
        "source": "SEC EDGAR EFTS (OS/AOS forms)",
    }


@muni_router.get("/sector-defaults")
def sector_default_rates() -> Dict[str, Any]:
    """Return historical municipal bond default rates by sector."""
    return {
        "source": "MSRB / Moody's Municipal Bond Default Studies",
        "horizon": "10-year cumulative default rates",
        "as_of": "2024",
        "rates": SECTOR_DEFAULT_RATES,
        "note": "General obligation bonds have significantly lower default rates than revenue bonds",
    }


@muni_router.get("/state-fiscal/{state}")
def state_fiscal_health(state: str) -> Dict[str, Any]:
    """Return fiscal health summary for a US state."""
    return _credit.state_fiscal_summary(state.upper())
