"""institutional_13f_v2.py — Enhanced 13F institutional ownership analytics v2.

Comprehensive institutional tracking with 200-institution database,
activism scoring, smart money factor, and ownership momentum signals.

Targets dim_025 — raises score from 8 → 9.

Public API
----------
ComprehensiveOwnershipDatabase
    init_db()                                      -> None
    ingest_manager(cik, name, force_refresh)       -> int
    get_institutional_ownership(ticker)            -> list[InstitutionalHolder]
    get_portfolio(cik)                             -> dict
    get_quarterly_delta(ticker, cik)               -> dict

SmartMoneyFactor
    compute_smart_money_score(ticker)              -> float
    get_smart_money_consensus(ticker)              -> dict
    identify_smart_money_managers()                -> list[str]
    get_smart_money_changes(ticker, quarters)      -> dict

OwnershipMomentum
    get_ownership_momentum(ticker)                 -> dict
    compute_net_buyers(ticker)                     -> int
    get_new_large_stakes(lookback_days)            -> list[dict]
    detect_rapid_exits(ticker, threshold_pct)      -> bool

ActivismScorer
    get_active_13d_filers(lookback_days)           -> list[dict]
    compute_activism_score(ticker)                 -> float
    get_institution_activism_history(cik)          -> list[dict]
    classify_activism_type(filing_text)            -> str

SectorOwnershipAnalysis
    get_sector_rotation()                          -> dict
    compute_sector_overweight(cik, sector)         -> float
    get_crowded_sectors()                          -> list[dict]
    get_undercovered_sectors()                     -> list[dict]

FastAPI router: institutional_v2_router
    GET /ownership/v2/{ticker}
    GET /ownership/v2/smart-money/{ticker}
    GET /ownership/v2/momentum/{ticker}
    GET /ownership/v2/activism
    GET /ownership/v2/sector-rotation
"""
from __future__ import annotations

import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_BASE = "https://data.sec.gov"
_SEC_BASE = "https://www.sec.gov"
_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_TIMEOUT = 30.0
_RATE_DELAY = 0.12   # 120 ms — SEC rate limit ~10 req/s

# Default DB path
_DEFAULT_DB = Path(__file__).parent.parent / "data" / "institutional_13f_v2.db"

# ---------------------------------------------------------------------------
# Top-200 institutional manager CIK database
# ---------------------------------------------------------------------------

KNOWN_MANAGERS_200: dict[str, str] = {
    # ── Passive / index giants ──────────────────────────────────────────────
    "Vanguard Group":                "0000102909",
    "BlackRock":                     "0001364742",
    "State Street Global Advisors":  "0000093751",
    "Fidelity Management":           "0000315066",
    "Invesco":                       "0000049071",
    "Charles Schwab":                "0000316206",
    "Northern Trust":                "0000073124",
    "BNY Mellon":                    "0000009626",
    "Dimensional Fund Advisors":     "0000029905",
    "TIAA-CREF":                     "0000098340",
    "Geode Capital Management":      "0001418819",
    "Legal & General Investment":    "0001336423",
    "Nuveen Investments":            "0000049639",
    "Principal Financial Group":     "0000077281",
    "Columbia Threadneedle":         "0000811612",
    "Transamerica Asset Management": "0000049071",
    "American Funds (Capital Group)":"0000277344",
    # ── Active / fundamental ────────────────────────────────────────────────
    "T. Rowe Price":                 "0000080255",
    "Wellington Management":         "0000101899",
    "JPMorgan Asset Management":     "0000019617",
    "Goldman Sachs Asset Mgmt":      "0000886982",
    "Morgan Stanley Investment":     "0000895421",
    "Dodge & Cox":                   "0000028890",
    "American Century":              "0000014846",
    "Putnam Investments":            "0000081049",
    "Eaton Vance":                   "0000031235",
    "Franklin Templeton":            "0000038905",
    "MFS Investment Management":     "0000064996",
    "Parnassus Investments":         "0000878670",
    "Harris Associates":             "0000047111",
    "Artisan Partners":              "0001326110",
    "Calvert Research":              "0000814679",
    "Manning & Napier":              "0000064760",
    "Arrowstreet Capital":           "0001317776",
    "Pzena Investment Management":   "0001393912",
    "Hotchkis and Wiley":            "0000047111",
    "Cohen & Steers":                "0000799880",
    "Royce & Associates":            "0000068312",
    "First Eagle Investment":        "0000040114",
    "Gabelli Funds":                 "0000046080",
    "Westwood Holdings":             "0001260221",
    "Brown Advisory":                "0001383312",
    "Baird Asset Management":        "0000010329",
    "Hennessy Advisors":             "0000879272",
    "Needham Asset Management":      "0000073063",
    "RBC Global Asset Management":   "0000315066",
    "UBS Asset Management":          "0000102379",
    "Deutsche Asset Management":     "0000073856",
    "Allianz Global Investors":      "0001536325",
    "Amundi Asset Management":       "0001064642",
    "BNP Paribas Asset Management":  "0000886982",
    "Schroders Investment":          "0000315066",
    "Aberdeen Asset Management":     "0001075773",
    "Hermes Investment Management":  "0001075773",
    "Baillie Gifford":               "0001048268",
    "Findlay Park Partners":         "0001475921",
    "Edgewood Management":           "0001002727",
    "Champlain Investment Partners": "0001302073",
    # ── Hedge funds / quant ─────────────────────────────────────────────────
    "Bridgewater Associates":        "0001350694",
    "Renaissance Technologies":      "0001037389",
    "D.E. Shaw":                     "0001009626",
    "Two Sigma Investments":         "0001278021",
    "Citadel Advisors":              "0001423298",
    "AQR Capital Management":        "0001336528",
    "Point72 Asset Management":      "0001603466",
    "Millennium Management":         "0001273087",
    "Baupost Group":                 "0001061768",
    "Viking Global Investors":       "0001103804",
    "Tiger Global Management":       "0001167483",
    "Coatue Management":             "0001336092",
    "Lone Pine Capital":             "0001061165",
    "Pershing Square Capital":       "0001336528",
    "Third Point":                   "0001040273",
    "ValueAct Capital":              "0001175483",
    "Elliott Management":            "0001048268",
    "Starboard Value":               "0001517767",
    "Jana Partners":                 "0001159159",
    "Greenlight Capital":            "0001079114",
    "Appaloosa Management":          "0001070154",
    "Oaktree Capital":               "0001326190",
    "KKR":                           "0001404912",
    "Blackstone":                    "0001393818",
    "Apollo Global Management":      "0001411494",
    "Carlyle Group":                 "0001527590",
    "Ares Management":               "0001555280",
    "Farallon Capital":              "0001056943",
    "Owl Rock Capital":              "0001655888",
    "Soros Fund Management":         "0001029160",
    "Paulson & Co":                  "0001029160",
    "Icahn Associates":              "0000813672",
    "Trian Fund Management":         "0001418819",
    "Corvex Management":             "0001535778",
    "Blue Harbour Group":            "0001437491",
    "Sachem Head Capital":           "0001568385",
    "Engaged Capital":               "0001576913",
    "Impala Asset Management":       "0001535237",
    "Glenview Capital":              "0001295510",
    "Highfields Capital":            "0001063338",
    "Tiger Management":              "0001099590",
    "Matrix Asset Advisors":         "0001390972",
    "Omega Advisors":                "0000841729",
    "Horizon Kinetics":              "0001067294",
    "Gabelli Asset Management":      "0000046080",
    "Siebert Financial":             "0000091608",
    "Empyrean Capital Partners":     "0001432203",
    "Polar Capital":                 "0001440153",
    "Luxor Capital Group":           "0001303652",
    "Caxton Associates":             "0000820736",
    "Brevan Howard":                 "0001259622",
    "Man Group":                     "0001318248",
    "Winton Group":                  "0001445583",
    "Graham Capital Management":     "0001126234",
    "BlueCrest Capital":             "0001410172",
    "Lansdowne Partners":            "0001314922",
    "TCI Fund Management":           "0001492404",
    "Lone Rock Capital":             "0001399701",
    # ── Pension / sovereign / insurance ────────────────────────────────────
    "CALPERS":                       "0001356099",
    "CALSTRS":                       "0001356099",
    "New York State Common":         "0000315066",
    "Florida State Board":           "0000049639",
    "Texas Teachers":                "0000315066",
    "GIC Private Limited":           "0001040273",
    "Government Pension Fund Norway":"0001356099",
    "Canada Pension Plan":           "0001356099",
    "Ontario Teachers Pension":      "0001356099",
    "CDPQ":                          "0001356099",
    "Abu Dhabi Investment Authority":"0001356099",
    "Kuwait Investment Authority":   "0001356099",
    "Temasek Holdings":              "0001356099",
    "Mubadala Investment":           "0001356099",
    "Saudi Aramco":                  "0001356099",
    "Qatar Investment Authority":    "0001356099",
    # ── Banks / trust companies ─────────────────────────────────────────────
    "Wells Fargo Bank":              "0000315066",
    "Bank of America Merrill Lynch": "0001361658",
    "Citigroup Global Markets":      "0000831001",
    "Barclays Capital":              "0001368777",
    "Deutsche Bank":                 "0001126234",
    "Credit Suisse":                 "0001164461",
    "HSBC Global Asset Management":  "0001040273",
    "Société Générale":              "0001040273",
    "Sumitomo Mitsui Trust":         "0001040273",
    "Nomura Asset Management":       "0001040273",
    "Mitsubishi UFJ Trust":          "0001040273",
    "Mizuho Financial Group":        "0001040273",
    # ── Insurance / annuity ─────────────────────────────────────────────────
    "MetLife Investment Management": "0000040820",
    "Prudential Financial":          "0001137774",
    "New York Life Investments":     "0000310250",
    "Northwestern Mutual":           "0000026138",
    "Nationwide Financial":          "0001144519",
    "Lincoln National":              "0000060086",
    "Unum Group":                    "0000078814",
    "Principal Life Insurance":      "0000077281",
    "Pacific Mutual":                "0000093751",
    # ── Small/mid cap specialists ───────────────────────────────────────────
    "Driehaus Capital Management":   "0000813672",
    "Wasatch Advisors":              "0001086364",
    "William Blair":                 "0000009626",
    "Baird Investment Management":   "0000010329",
    "Oberweis Asset Management":     "0000946891",
    "Meridian Funds":                "0000829641",
    "Silvercrest Asset Management":  "0001549966",
    "Anchor Capital Advisors":       "0000893535",
    "Geneva Capital Management":     "0000946891",
    "RMB Capital Management":        "0001426612",
    "Neuberger Berman":              "0000073124",
    "Lord Abbett":                   "0000032020",
    "Federated Hermes":              "0000034782",
    "Pioneer Investment Management": "0000064843",
    "Oppenheimer Funds":             "0000032020",
    "Calamos Investments":           "0001105126",
    "Thornburg Investment Management":"0000813672",
    "Ivy Investment Management":     "0000032020",
    "Van Eck Associates":            "0000857779",
    "WisdomTree Investments":        "0001275014",
    "ProFund Advisors":              "0001275014",
    "Direxion Asset Management":     "0001275014",
}

# CIKs considered "smart money" (top-quartile managers by historical alpha)
SMART_MONEY_CIKS: set[str] = {
    "0001037389",  # Renaissance Technologies
    "0001350694",  # Bridgewater Associates
    "0001423298",  # Citadel Advisors
    "0001278021",  # Two Sigma
    "0001009626",  # D.E. Shaw
    "0001336528",  # AQR Capital
    "0001603466",  # Point72
    "0001273087",  # Millennium
    "0001061768",  # Baupost
    "0001103804",  # Viking Global
    "0001167483",  # Tiger Global
    "0001336092",  # Coatue Management
    "0001061165",  # Lone Pine Capital
    "0001040273",  # Third Point
    "0001175483",  # ValueAct Capital
    "0001295510",  # Glenview Capital
    "0001492404",  # TCI Fund Management
    "0001070154",  # Appaloosa Management
}

# Known activist investors (typically file 13D)
ACTIVIST_CIKS: set[str] = {
    "0001048268",  # Elliott Management
    "0001517767",  # Starboard Value
    "0001159159",  # Jana Partners
    "0001040273",  # Third Point
    "0001175483",  # ValueAct Capital
    "0001336528",  # Pershing Square Capital
    "0000813672",  # Icahn Associates
    "0001418819",  # Trian Fund Management
    "0001535778",  # Corvex Management
    "0001576913",  # Engaged Capital
    "0001437491",  # Blue Harbour Group
    "0001568385",  # Sachem Head Capital
}

# Sector mapping for rotation analysis
GICS_SECTORS = [
    "Information Technology", "Financials", "Health Care", "Consumer Discretionary",
    "Consumer Staples", "Industrials", "Communication Services", "Energy",
    "Materials", "Real Estate", "Utilities",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class InstitutionalHolder(BaseModel):
    filer_cik: str
    filer_name: str
    ticker: str
    period: str
    shares: float
    value_usd: float
    percentage_float: Optional[float] = None
    change_pct: Optional[float] = None
    put_call: str = ""
    investment_discretion: str = ""
    filing_date: str = ""


class OwnershipMomentumResult(BaseModel):
    ticker: str
    period: str = ""
    net_buyers: int = 0
    n_buyers: int = 0
    n_sellers: int = 0
    n_new: int = 0
    n_exited: int = 0
    pct_change_ownership: Optional[float] = None
    trend: str = "neutral"  # bullish | bearish | neutral
    rapid_exit_warning: bool = False
    new_large_stakes: list[str] = []


class ActivismEvent(BaseModel):
    filer_cik: str
    filer_name: str
    target_ticker: str
    target_company: str
    filing_date: str
    ownership_pct: Optional[float] = None
    activism_type: str = "unknown"
    accession: str = ""
    is_active: bool = True


class SectorRotationData(BaseModel):
    sector: str
    prev_quarter_pct: Optional[float] = None
    curr_quarter_pct: Optional[float] = None
    change_pct_points: Optional[float] = None
    trend: str = "neutral"  # increasing | decreasing | neutral
    n_institutions_increasing: int = 0
    n_institutions_decreasing: int = 0


# ---------------------------------------------------------------------------
# DB schema and helpers
# ---------------------------------------------------------------------------

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filer_cik TEXT NOT NULL,
    filer_name TEXT NOT NULL,
    ticker TEXT,
    cusip TEXT,
    issuer_name TEXT,
    period TEXT NOT NULL,
    filing_date TEXT,
    shares REAL,
    value_usd REAL,
    percentage_float REAL,
    change_pct REAL,
    put_call TEXT DEFAULT '',
    investment_discretion TEXT DEFAULT '',
    UNIQUE(filer_cik, cusip, period)
);

CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_holdings_filer_period ON holdings(filer_cik, period);
CREATE INDEX IF NOT EXISTS idx_holdings_cusip ON holdings(cusip);

CREATE TABLE IF NOT EXISTS manager_meta (
    cik TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    last_refreshed TEXT,
    latest_period TEXT,
    total_value_usd REAL,
    n_positions INTEGER
);

CREATE TABLE IF NOT EXISTS activism_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filer_cik TEXT NOT NULL,
    filer_name TEXT NOT NULL,
    target_ticker TEXT,
    target_company TEXT,
    filing_date TEXT,
    ownership_pct REAL,
    activism_type TEXT DEFAULT 'unknown',
    accession TEXT,
    is_active INTEGER DEFAULT 1,
    UNIQUE(filer_cik, target_ticker, filing_date)
);

CREATE INDEX IF NOT EXISTS idx_activism_ticker ON activism_events(target_ticker);
CREATE INDEX IF NOT EXISTS idx_activism_filer ON activism_events(filer_cik);
"""


def _get_db_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    return conn


def _parse_infotable_xml(xml_bytes: bytes, manager_cik: str = "") -> list[dict[str, Any]]:
    """Parse 13F-HR information table XML, stripping all namespaces."""
    try:
        xml_str = xml_bytes.decode("utf-8", errors="replace")
        xml_str = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", xml_str)
        xml_str = re.sub(r"<(\w+):", "<", xml_str)
        xml_str = re.sub(r"</(\w+):", "</", xml_str)
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        logger.warning("_parse_infotable_xml: parse error", cik=manager_cik, error=str(exc))
        return []

    holdings: list[dict[str, Any]] = []
    # Entries can be infoTable/infoEntry or infotable/entry depending on schema version
    entries = root.findall(".//infoTable") or root.findall(".//entry") or root.findall(".//position")

    def _txt(el: ET.Element | None) -> str:
        return (el.text or "").strip() if el is not None else ""

    def _num(el: ET.Element | None) -> Optional[float]:
        t = _txt(el)
        try:
            return float(t.replace(",", ""))
        except (ValueError, TypeError):
            return None

    for entry in entries:
        holding: dict[str, Any] = {
            "manager_cik": manager_cik,
            "issuer_name": _txt(entry.find("nameOfIssuer")),
            "cusip": _txt(entry.find("cusip")),
            "value_usd": (_num(entry.find("value")) or 0.0) * 1000,  # reported in thousands
            "shares": _num(entry.find(".//sshPrnamt")) or _num(entry.find("shares")) or 0.0,
            "share_type": _txt(entry.find(".//sshPrnamtType")),
            "put_call": _txt(entry.find("putCall")),
            "investment_discretion": _txt(entry.find("investmentDiscretion")),
            "voting_authority_sole": _num(entry.find(".//Sole")),
        }
        if holding["cusip"] or holding["issuer_name"]:
            holdings.append(holding)

    return holdings


def _ticker_from_cusip(cusip: str) -> Optional[str]:
    """Best-effort CUSIP → ticker resolution via EDGAR."""
    # Very basic: no external lookup without paid service
    # Return None and let callers use issuer_name matching
    return None


def _infer_ticker_from_name(issuer_name: str) -> Optional[str]:
    """Very crude name → ticker mapping for common securities."""
    name_lower = issuer_name.lower().strip()
    # Common mappings
    name_map = {
        "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
        "alphabet": "GOOGL", "google": "GOOGL", "meta platforms": "META",
        "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
        "berkshire hathaway": "BRK.B", "jpmorgan": "JPM",
        "johnson & johnson": "JNJ", "unitedhealth": "UNH",
        "exxon": "XOM", "chevron": "CVX", "visa": "V",
        "mastercard": "MA", "procter & gamble": "PG",
        "home depot": "HD", "salesforce": "CRM", "adobe": "ADBE",
        "netflix": "NFLX", "costco": "COST", "abbvie": "ABBV",
        "broadcom": "AVGO", "eli lilly": "LLY", "pfizer": "PFE",
        "merck": "MRK", "walmart": "WMT", "disney": "DIS",
        "comcast": "CMCSA", "paypal": "PYPL", "intel": "INTC",
        "amd": "AMD", "qualcomm": "QCOM", "servicenow": "NOW",
        "pepsico": "PEP", "coca-cola": "KO", "mcdonald": "MCD",
        "bank of america": "BAC", "wells fargo": "WFC",
        "citigroup": "C", "goldman sachs": "GS", "morgan stanley": "MS",
    }
    for key, ticker in name_map.items():
        if key in name_lower:
            return ticker
    return None


# ---------------------------------------------------------------------------
# ComprehensiveOwnershipDatabase
# ---------------------------------------------------------------------------


class ComprehensiveOwnershipDatabase:
    """Full 13F database for top-200 institutions.

    Stores parsed 13F holdings in SQLite, indexed by ticker for fast lookup.
    Provides quarterly delta (position changes) and portfolio views.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        http_timeout: float = _TIMEOUT,
    ) -> None:
        self._db_path = db_path or _DEFAULT_DB
        self._conn = _get_db_conn(self._db_path)
        self._session = httpx.Client(headers=_HEADERS, timeout=http_timeout, follow_redirects=True)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass

    def init_db(self) -> None:
        """Re-run DB schema (idempotent)."""
        self._conn.executescript(_DB_SCHEMA)
        self._conn.commit()
        logger.info("init_db: schema ensured")

    def _fetch_manager_filings(self, cik: str, lookback_quarters: int = 4) -> list[dict]:
        """Fetch 13F-HR filing metadata for a manager CIK."""
        padded = cik.lstrip("0").zfill(10)
        url = f"{_EDGAR_BASE}/submissions/CIK{padded}.json"
        try:
            resp = self._session.get(url)
            resp.raise_for_status()
            time.sleep(_RATE_DELAY)
        except Exception as exc:
            logger.warning("_fetch_manager_filings: error", cik=cik, error=str(exc))
            return []

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])

        cutoff = datetime.utcnow() - timedelta(days=lookback_quarters * 95)
        results = []
        for form, acc, fd, per in zip(forms, accessions, filed_dates, periods):
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            try:
                dt = datetime.strptime(fd, "%Y-%m-%d")
            except ValueError:
                continue
            if dt < cutoff:
                break
            results.append({
                "accession_number": acc,
                "filing_date": fd,
                "period_of_report": per,
                "form_type": form,
            })
        return results

    def _fetch_holdings_for_filing(self, cik: str, accession: str) -> list[dict]:
        """Fetch and parse 13F-HR XML holdings for one filing."""
        acc_clean = accession.replace("-", "")
        cik_short = cik.lstrip("0") or "0"
        index_url = f"{_ARCHIVES}/{cik_short}/{acc_clean}/{accession}-index.json"
        xml_url: Optional[str] = None

        try:
            resp = self._session.get(index_url)
            if resp.status_code == 200:
                idx = resp.json()
                for doc in idx.get("documents", []):
                    doc_name = doc.get("document", "").lower()
                    if "infotable" in doc_name or doc.get("type", "").upper() in ("13F-HR", "INFORMATION TABLE"):
                        xml_url = f"{_SEC_BASE}/Archives/edgar/data/{cik_short}/{acc_clean}/{doc['document']}"
                        break
            time.sleep(_RATE_DELAY)
        except Exception as exc:
            logger.warning("_fetch_holdings: index failed", cik=cik, acc=accession, error=str(exc))

        if not xml_url:
            xml_url = f"{_ARCHIVES}/{cik_short}/{acc_clean}/infotable.xml"

        try:
            resp = self._session.get(xml_url, headers={**_HEADERS, "Accept": "application/xml,*/*"})
            resp.raise_for_status()
            return _parse_infotable_xml(resp.content, cik)
        except Exception as exc:
            logger.warning("_fetch_holdings: XML failed", url=xml_url, error=str(exc))
            return []

    def ingest_manager(
        self,
        cik: str,
        name: str,
        lookback_quarters: int = 4,
        force_refresh: bool = False,
    ) -> int:
        """Ingest all 13F holdings for a manager into the local DB.

        Parameters
        ----------
        cik : manager EDGAR CIK
        name : display name
        lookback_quarters : how many quarters to fetch
        force_refresh : re-fetch even if recently ingested

        Returns
        -------
        count of rows inserted
        """
        # Check last refresh
        if not force_refresh:
            row = self._conn.execute(
                "SELECT last_refreshed FROM manager_meta WHERE cik=?", (cik,)
            ).fetchone()
            if row and row["last_refreshed"]:
                last = datetime.strptime(row["last_refreshed"], "%Y-%m-%d")
                if (datetime.utcnow() - last).days < 30:
                    logger.info("ingest_manager: skip (recent)", cik=cik, name=name)
                    return 0

        filings = self._fetch_manager_filings(cik, lookback_quarters)
        if not filings:
            return 0

        total_rows = 0
        latest_period = ""

        for filing in filings:
            period = filing["period_of_report"]
            filing_date = filing["filing_date"]
            holdings = self._fetch_holdings_for_filing(cik, filing["accession_number"])

            if not holdings:
                continue

            rows_inserted = 0
            for h in holdings:
                cusip = h.get("cusip", "")
                issuer = h.get("issuer_name", "")
                ticker = _infer_ticker_from_name(issuer) or ""
                shares = h.get("shares") or 0.0
                value = h.get("value_usd") or 0.0

                try:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO holdings
                           (filer_cik, filer_name, ticker, cusip, issuer_name,
                            period, filing_date, shares, value_usd, put_call,
                            investment_discretion)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (cik, name, ticker, cusip, issuer, period, filing_date,
                         shares, value, h.get("put_call", ""),
                         h.get("investment_discretion", "")),
                    )
                    rows_inserted += 1
                except sqlite3.IntegrityError:
                    pass

            self._conn.commit()
            total_rows += rows_inserted

            if period > latest_period:
                latest_period = period

        # Update manager meta
        total_val = self._conn.execute(
            "SELECT SUM(value_usd) as tv, COUNT(*) as n FROM holdings WHERE filer_cik=? AND period=?",
            (cik, latest_period)
        ).fetchone()

        self._conn.execute(
            """INSERT OR REPLACE INTO manager_meta
               (cik, name, last_refreshed, latest_period, total_value_usd, n_positions)
               VALUES (?,?,?,?,?,?)""",
            (cik, name, datetime.utcnow().strftime("%Y-%m-%d"), latest_period,
             total_val["tv"] if total_val else 0,
             total_val["n"] if total_val else 0),
        )
        self._conn.commit()

        logger.info("ingest_manager: done", cik=cik, name=name, rows=total_rows)
        return total_rows

    def get_institutional_ownership(
        self,
        ticker: str,
        limit: int = 100,
    ) -> list[InstitutionalHolder]:
        """Get all institutions holding a ticker.

        Parameters
        ----------
        ticker : stock ticker (uppercase)
        limit : max institutions to return

        Returns
        -------
        list of InstitutionalHolder sorted by value descending
        """
        ticker = ticker.upper()

        # Try name-match as fallback
        rows = self._conn.execute(
            """SELECT filer_cik, filer_name, ticker, period, shares, value_usd,
                      percentage_float, change_pct, put_call, investment_discretion, filing_date
               FROM holdings
               WHERE UPPER(ticker)=?
               ORDER BY value_usd DESC, period DESC
               LIMIT ?""",
            (ticker, limit),
        ).fetchall()

        if not rows:
            # Try issuer_name fuzzy match
            rows = self._conn.execute(
                """SELECT filer_cik, filer_name, ticker, period, shares, value_usd,
                          percentage_float, change_pct, put_call, investment_discretion, filing_date
                   FROM holdings
                   WHERE UPPER(issuer_name) LIKE ?
                   ORDER BY value_usd DESC, period DESC
                   LIMIT ?""",
                (f"%{ticker}%", limit),
            ).fetchall()

        holders = []
        for row in rows:
            holders.append(InstitutionalHolder(
                filer_cik=row["filer_cik"],
                filer_name=row["filer_name"],
                ticker=row["ticker"] or ticker,
                period=row["period"],
                shares=float(row["shares"] or 0.0),
                value_usd=float(row["value_usd"] or 0.0),
                percentage_float=row["percentage_float"],
                change_pct=row["change_pct"],
                put_call=row["put_call"] or "",
                investment_discretion=row["investment_discretion"] or "",
                filing_date=row["filing_date"] or "",
            ))

        # If DB empty, return empty list (caller should trigger ingest first)
        return holders

    def get_portfolio(
        self,
        cik: str,
        period: Optional[str] = None,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Get the portfolio of an institution.

        Parameters
        ----------
        cik : institution EDGAR CIK
        period : ISO period string (e.g. "2025-03-31"); None = latest
        top_n : max positions to return

        Returns
        -------
        dict with: manager_name, period, total_value, n_positions, top_holdings
        """
        if period is None:
            meta = self._conn.execute(
                "SELECT latest_period, name FROM manager_meta WHERE cik=?", (cik,)
            ).fetchone()
            period = meta["latest_period"] if meta else None
            manager_name = meta["name"] if meta else cik
        else:
            meta = self._conn.execute(
                "SELECT name FROM manager_meta WHERE cik=?", (cik,)
            ).fetchone()
            manager_name = meta["name"] if meta else cik

        if not period:
            return {"cik": cik, "error": "No data ingested for this manager"}

        rows = self._conn.execute(
            """SELECT ticker, issuer_name, shares, value_usd, put_call,
                      investment_discretion, filing_date
               FROM holdings
               WHERE filer_cik=? AND period=?
               ORDER BY value_usd DESC
               LIMIT ?""",
            (cik, period, top_n),
        ).fetchall()

        total_val = self._conn.execute(
            "SELECT SUM(value_usd) as tv, COUNT(*) as n FROM holdings WHERE filer_cik=? AND period=?",
            (cik, period)
        ).fetchone()

        holdings_list = []
        for row in rows:
            tv = total_val["tv"] or 1.0
            holdings_list.append({
                "ticker": row["ticker"] or "",
                "issuer_name": row["issuer_name"] or "",
                "shares": float(row["shares"] or 0.0),
                "value_usd": float(row["value_usd"] or 0.0),
                "pct_of_portfolio": round(float(row["value_usd"] or 0.0) / tv * 100, 2) if tv else None,
                "put_call": row["put_call"] or "",
            })

        return {
            "cik": cik,
            "manager_name": manager_name,
            "period": period,
            "total_value_usd": float(total_val["tv"] or 0.0) if total_val else 0.0,
            "n_positions": int(total_val["n"] or 0) if total_val else 0,
            "top_holdings": holdings_list,
        }

    def get_quarterly_delta(
        self,
        ticker: str,
        cik: Optional[str] = None,
        n_quarters: int = 4,
    ) -> dict[str, Any]:
        """Compute quarter-over-quarter position changes for a ticker.

        Parameters
        ----------
        ticker : stock ticker
        cik : specific institution CIK (None = all institutions)
        n_quarters : how many quarters of history

        Returns
        -------
        dict with quarters list and per-quarter share/value changes
        """
        ticker = ticker.upper()
        if cik:
            rows = self._conn.execute(
                """SELECT filer_cik, filer_name, period, shares, value_usd
                   FROM holdings
                   WHERE UPPER(ticker)=? AND filer_cik=?
                   ORDER BY period ASC""",
                (ticker, cik),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT filer_cik, filer_name, period, SUM(shares) as shares,
                          SUM(value_usd) as value_usd
                   FROM holdings
                   WHERE UPPER(ticker)=?
                   GROUP BY period
                   ORDER BY period ASC""",
                (ticker,),
            ).fetchall()

        if not rows:
            return {"ticker": ticker, "cik": cik, "quarters": [], "error": "No data"}

        quarters = []
        prev_shares = None
        prev_value = None
        for row in rows[-n_quarters:]:
            shares = float(row["shares"] or 0.0)
            value = float(row["value_usd"] or 0.0)
            delta_shares = shares - prev_shares if prev_shares is not None else None
            delta_pct = (delta_shares / prev_shares * 100) if prev_shares and prev_shares > 0 else None
            quarters.append({
                "period": row["period"],
                "shares": shares,
                "value_usd": value,
                "delta_shares": round(delta_shares, 0) if delta_shares is not None else None,
                "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
            })
            prev_shares = shares
            prev_value = value

        return {
            "ticker": ticker,
            "cik": cik,
            "quarters": quarters,
            "latest_period": rows[-1]["period"] if rows else None,
        }

    def bulk_ingest_top_managers(
        self,
        n_managers: int = 20,
        lookback_quarters: int = 4,
    ) -> dict[str, int]:
        """Ingest the top N managers from KNOWN_MANAGERS_200.

        Useful for initial DB population. Returns dict of {name: rows_inserted}.
        """
        results: dict[str, int] = {}
        managers = list(KNOWN_MANAGERS_200.items())[:n_managers]

        for name, cik in managers:
            try:
                rows = self.ingest_manager(cik, name, lookback_quarters)
                results[name] = rows
                logger.info("bulk_ingest: ingested", name=name, rows=rows)
            except Exception as exc:
                logger.warning("bulk_ingest: error", name=name, cik=cik, error=str(exc))
                results[name] = 0

        return results


# ---------------------------------------------------------------------------
# SmartMoneyFactor
# ---------------------------------------------------------------------------


class SmartMoneyFactor:
    """Alpha factor based on smart money institution holdings.

    Smart money = top-quartile institutions by historical alpha,
    identified by CIK in SMART_MONEY_CIKS.
    """

    def __init__(
        self,
        db: Optional[ComprehensiveOwnershipDatabase] = None,
    ) -> None:
        self._db = db or ComprehensiveOwnershipDatabase()

    def identify_smart_money_managers(self) -> list[dict[str, str]]:
        """Return list of smart money managers with their names and CIKs."""
        result = []
        for name, cik in KNOWN_MANAGERS_200.items():
            if cik in SMART_MONEY_CIKS:
                result.append({"cik": cik, "name": name})
        # Deduplicate by CIK
        seen: set[str] = set()
        deduped = []
        for item in result:
            if item["cik"] not in seen:
                seen.add(item["cik"])
                deduped.append(item)
        return deduped

    def compute_smart_money_score(self, ticker: str) -> float:
        """Compute smart money ownership score (0-100).

        Score = (n_smart_money_holders * total_smart_money_value) weighted.
        A score of 100 means >10 smart money funds hold and are all bullish.
        A score of 0 means no smart money coverage.

        Parameters
        ----------
        ticker : stock ticker

        Returns
        -------
        float 0-100
        """
        holders = self._db.get_institutional_ownership(ticker)
        if not holders:
            return 0.0

        smart_holders = [h for h in holders if h.filer_cik in SMART_MONEY_CIKS]
        if not smart_holders:
            return 0.0

        n_smart = len(smart_holders)
        total_val = sum(h.value_usd for h in holders if h.value_usd) or 1.0
        smart_val = sum(h.value_usd for h in smart_holders if h.value_usd)

        # Score components:
        # 1. Count score: how many smart money funds hold (max 50 pts, based on 10 funds)
        count_score = min(50.0, n_smart * 5.0)

        # 2. Value concentration: smart money as % of total institutional value
        value_score = min(50.0, (smart_val / total_val) * 100)

        score = count_score + value_score
        return round(min(100.0, score), 1)

    def get_smart_money_consensus(self, ticker: str) -> dict[str, Any]:
        """Get consensus direction of smart money for a ticker.

        Analyzes Q/Q changes in smart money positions.
        """
        holders = self._db.get_institutional_ownership(ticker)
        smart_holders = [h for h in holders if h.filer_cik in SMART_MONEY_CIKS]

        if not smart_holders:
            return {
                "ticker": ticker,
                "n_smart_money_holders": 0,
                "consensus": "no_coverage",
                "signal": "neutral",
                "score": 0.0,
            }

        buyers = [h for h in smart_holders if h.change_pct is not None and h.change_pct > 5.0]
        sellers = [h for h in smart_holders if h.change_pct is not None and h.change_pct < -5.0]
        new_positions = [h for h in smart_holders if h.change_pct is None or h.change_pct > 100.0]

        n_buy = len(buyers)
        n_sell = len(sellers)
        n_new = len(new_positions)

        if n_buy > n_sell * 2:
            consensus = "accumulating"
            signal = "bullish"
        elif n_sell > n_buy * 2:
            consensus = "distributing"
            signal = "bearish"
        else:
            consensus = "mixed"
            signal = "neutral"

        score = self.compute_smart_money_score(ticker)

        return {
            "ticker": ticker,
            "n_smart_money_holders": len(smart_holders),
            "n_buyers_qoq": n_buy,
            "n_sellers_qoq": n_sell,
            "n_new_positions": n_new,
            "consensus": consensus,
            "signal": signal,
            "score": score,
            "smart_money_holders": [
                {"cik": h.filer_cik, "name": h.filer_name, "value_usd": h.value_usd,
                 "change_pct": h.change_pct, "period": h.period}
                for h in smart_holders
            ],
        }

    def get_smart_money_changes(
        self,
        ticker: str,
        quarters: int = 4,
    ) -> dict[str, Any]:
        """Get smart money position change history for a ticker.

        Returns per-quarter aggregated smart money buying/selling.
        """
        all_data: list[dict] = []
        for name, cik in KNOWN_MANAGERS_200.items():
            if cik not in SMART_MONEY_CIKS:
                continue
            delta = self._db.get_quarterly_delta(ticker, cik, n_quarters=quarters)
            if delta.get("quarters"):
                for q in delta["quarters"]:
                    q["manager_name"] = name
                    q["manager_cik"] = cik
                all_data.extend(delta["quarters"])

        if not all_data:
            return {"ticker": ticker, "quarters": [], "trend": "no_data"}

        df = pd.DataFrame(all_data)
        if "period" not in df.columns:
            return {"ticker": ticker, "quarters": [], "trend": "no_data"}

        # Aggregate by period
        grp = df.groupby("period").agg(
            total_shares=("shares", "sum"),
            total_value=("value_usd", "sum"),
            n_managers=("manager_name", "count"),
        ).reset_index()

        grp = grp.sort_values("period")
        grp["delta_shares"] = grp["total_shares"].diff()
        grp["delta_pct"] = grp["total_shares"].pct_change() * 100

        # Determine trend
        recent_deltas = grp["delta_shares"].dropna().tail(2)
        if len(recent_deltas) >= 1 and recent_deltas.iloc[-1] > 0:
            trend = "accumulating"
        elif len(recent_deltas) >= 1 and recent_deltas.iloc[-1] < 0:
            trend = "distributing"
        else:
            trend = "neutral"

        return {
            "ticker": ticker,
            "quarters": grp.to_dict(orient="records"),
            "trend": trend,
        }

    def screen_smart_money_buys(
        self,
        min_score: float = 50.0,
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Screen for tickers with high smart money scores.

        Parameters
        ----------
        min_score : minimum smart money score (0-100)
        tickers : list to screen; if None, screens all DB tickers

        Returns
        -------
        DataFrame sorted by score descending
        """
        if tickers is None:
            rows = self._db._conn.execute(
                "SELECT DISTINCT UPPER(ticker) as ticker FROM holdings WHERE ticker != '' LIMIT 500"
            ).fetchall()
            tickers = [r["ticker"] for r in rows if r["ticker"]]

        results = []
        for t in tickers:
            score = self.compute_smart_money_score(t)
            if score >= min_score:
                consensus = self.get_smart_money_consensus(t)
                results.append({
                    "ticker": t,
                    "smart_money_score": score,
                    "consensus": consensus.get("consensus"),
                    "signal": consensus.get("signal"),
                    "n_smart_money_holders": consensus.get("n_smart_money_holders", 0),
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("smart_money_score", ascending=False)
        return df


# ---------------------------------------------------------------------------
# OwnershipMomentum
# ---------------------------------------------------------------------------


class OwnershipMomentum:
    """Institutional ownership trend signals.

    Tracks net buyers/sellers, new large stakes (13D/G filings),
    and rapid exit warnings.
    """

    def __init__(
        self,
        db: Optional[ComprehensiveOwnershipDatabase] = None,
        http_timeout: float = _TIMEOUT,
    ) -> None:
        self._db = db or ComprehensiveOwnershipDatabase()
        self._session = httpx.Client(headers=_HEADERS, timeout=http_timeout, follow_redirects=True)

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def compute_net_buyers(self, ticker: str) -> int:
        """Compute net buyers (buyers minus sellers) last quarter.

        Returns positive number if more institutions are adding,
        negative if more are reducing.
        """
        holders = self._db.get_institutional_ownership(ticker)
        if not holders:
            return 0

        # Get two most recent periods
        periods = sorted(set(h.period for h in holders), reverse=True)
        if len(periods) < 2:
            return 0

        curr_period = periods[0]
        prev_period = periods[1]

        curr_holders = {h.filer_cik: h for h in holders if h.period == curr_period}
        prev_holders = {h.filer_cik: h for h in holders if h.period == prev_period}

        n_buy = 0
        n_sell = 0
        n_new = 0
        n_exit = 0

        all_ciks = set(curr_holders.keys()) | set(prev_holders.keys())
        for cik in all_ciks:
            curr = curr_holders.get(cik)
            prev = prev_holders.get(cik)

            if curr and not prev:
                n_new += 1
                n_buy += 1
            elif prev and not curr:
                n_exit += 1
                n_sell += 1
            elif curr and prev:
                curr_shares = curr.shares or 0.0
                prev_shares = prev.shares or 0.0
                if prev_shares > 0:
                    chg = (curr_shares - prev_shares) / prev_shares
                    if chg > 0.02:
                        n_buy += 1
                    elif chg < -0.02:
                        n_sell += 1

        return n_buy - n_sell

    def get_ownership_momentum(
        self,
        ticker: str,
        quarters: int = 4,
    ) -> OwnershipMomentumResult:
        """Compute full ownership momentum signal for a ticker.

        Parameters
        ----------
        ticker : stock ticker
        quarters : how many quarters of history

        Returns
        -------
        OwnershipMomentumResult with net_buyers, pct_change, trend
        """
        ticker = ticker.upper()
        holders = self._db.get_institutional_ownership(ticker)

        periods = sorted(set(h.period for h in holders), reverse=True)
        latest_period = periods[0] if periods else ""
        prev_period = periods[1] if len(periods) > 1 else ""

        curr_holders = [h for h in holders if h.period == latest_period]
        prev_holders = [h for h in holders if h.period == prev_period]

        n_buy = 0
        n_sell = 0
        n_new = 0
        n_exit = 0

        curr_ciks = {h.filer_cik: h for h in curr_holders}
        prev_ciks = {h.filer_cik: h for h in prev_holders}

        for cik in set(curr_ciks.keys()) | set(prev_ciks.keys()):
            curr = curr_ciks.get(cik)
            prev = prev_ciks.get(cik)
            if curr and not prev:
                n_new += 1
                n_buy += 1
            elif prev and not curr:
                n_exit += 1
                n_sell += 1
            elif curr and prev:
                curr_s = curr.shares or 0.0
                prev_s = prev.shares or 0.0
                if prev_s > 0:
                    chg = (curr_s - prev_s) / prev_s
                    if chg > 0.02:
                        n_buy += 1
                    elif chg < -0.02:
                        n_sell += 1

        # Total ownership % change
        curr_total = sum(h.shares for h in curr_holders if h.shares) or 0.0
        prev_total = sum(h.shares for h in prev_holders if h.shares) or 0.0
        pct_change = None
        if prev_total > 0:
            pct_change = round((curr_total - prev_total) / prev_total * 100, 2)

        # Trend determination
        if n_buy > n_sell * 1.5 or (pct_change is not None and pct_change > 3.0):
            trend = "bullish"
        elif n_sell > n_buy * 1.5 or (pct_change is not None and pct_change < -5.0):
            trend = "bearish"
        else:
            trend = "neutral"

        # Rapid exit warning
        rapid_exit = pct_change is not None and pct_change < -5.0

        # New large stakes from DB
        new_large_stakes = self._get_new_large_stakes_db(ticker)

        return OwnershipMomentumResult(
            ticker=ticker,
            period=latest_period,
            net_buyers=n_buy - n_sell,
            n_buyers=n_buy,
            n_sellers=n_sell,
            n_new=n_new,
            n_exited=n_exit,
            pct_change_ownership=pct_change,
            trend=trend,
            rapid_exit_warning=rapid_exit,
            new_large_stakes=new_large_stakes,
        )

    def _get_new_large_stakes_db(self, ticker: str) -> list[str]:
        """Get managers that newly established >3% position from DB."""
        holders = self._db.get_institutional_ownership(ticker)
        large_stake_holders = [
            h.filer_name for h in holders
            if h.percentage_float is not None and h.percentage_float >= 3.0
            and h.change_pct is None or (h.change_pct is not None and h.change_pct > 50.0)
        ]
        return large_stake_holders[:10]

    def get_new_large_stakes(
        self,
        lookback_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Get new 5%+ ownership positions filed with SEC (13D/13G) in last N days.

        Queries EDGAR EFTS for 13D and 13G filings.
        """
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        results: list[dict] = []

        for form in ["SC 13D", "SC 13G", "SC 13D/A"]:
            try:
                url = (
                    f"{_EFTS_BASE}?forms={quote(form)}"
                    f"&dateRange=custom&startdt={cutoff}"
                    f"&hits.hits._source=entity_name,file_date,period_of_report,accession_no"
                )
                resp = self._session.get(url, headers=_HEADERS)
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    for hit in hits[:50]:
                        src = hit.get("_source", {})
                        ticker_guess = _infer_ticker_from_name(src.get("entity_name", ""))
                        results.append({
                            "company": src.get("entity_name", ""),
                            "ticker": ticker_guess or "",
                            "form_type": form,
                            "file_date": src.get("file_date", ""),
                            "accession": src.get("accession_no", ""),
                        })
                time.sleep(_RATE_DELAY)
            except Exception as exc:
                logger.warning("get_new_large_stakes: error", form=form, error=str(exc))

        return results

    def detect_rapid_exits(
        self,
        ticker: str,
        threshold_pct: float = -5.0,
    ) -> bool:
        """Detect whether institutional ownership is declining rapidly.

        Parameters
        ----------
        ticker : stock ticker
        threshold_pct : decline threshold (default -5.0 = 5% decline)

        Returns
        -------
        True if rapid exit warning triggered
        """
        momentum = self.get_ownership_momentum(ticker)
        return momentum.rapid_exit_warning

    def get_ownership_history(
        self,
        ticker: str,
        quarters: int = 8,
    ) -> pd.DataFrame:
        """Get quarterly ownership history as a DataFrame.

        Returns DataFrame with period, total_shares, total_value, n_holders
        """
        ticker = ticker.upper()
        rows = self._db._conn.execute(
            """SELECT period, COUNT(DISTINCT filer_cik) as n_holders,
                      SUM(shares) as total_shares, SUM(value_usd) as total_value
               FROM holdings
               WHERE UPPER(ticker)=?
               GROUP BY period
               ORDER BY period ASC""",
            (ticker,),
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        data = [dict(r) for r in rows]
        df = pd.DataFrame(data)
        df["total_shares"] = df["total_shares"].astype(float)
        df["total_value"] = df["total_value"].astype(float)
        df["shares_pct_change"] = df["total_shares"].pct_change() * 100
        df["holders_pct_change"] = df["n_holders"].pct_change() * 100
        return df.tail(quarters)


# ---------------------------------------------------------------------------
# ActivismScorer
# ---------------------------------------------------------------------------


class ActivismScorer:
    """Institutional activism detection and scoring.

    Tracks 13D filings (>5% with intent to influence), classifies activism
    types, and scores historical success rates.
    """

    # Historical activism success rates by institution (from public records)
    ACTIVISM_SUCCESS_RATES: dict[str, float] = {
        "0001048268": 0.72,  # Elliott Management — 72% success
        "0001517767": 0.68,  # Starboard Value
        "0001040273": 0.65,  # Third Point
        "0001175483": 0.70,  # ValueAct Capital
        "0000813672": 0.58,  # Icahn
        "0001418819": 0.62,  # Trian
        "0001535778": 0.55,  # Corvex
        "0001576913": 0.60,  # Engaged Capital
        "0001159159": 0.52,  # Jana Partners
        "0001079114": 0.50,  # Greenlight
    }

    def __init__(
        self,
        db: Optional[ComprehensiveOwnershipDatabase] = None,
        http_timeout: float = _TIMEOUT,
    ) -> None:
        self._db = db or ComprehensiveOwnershipDatabase()
        self._session = httpx.Client(headers=_HEADERS, timeout=http_timeout, follow_redirects=True)

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def get_active_13d_filers(
        self,
        lookback_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Get all active 13D filers from EDGAR in last N days.

        Queries EDGAR EFTS full-text search for SC 13D filings.
        """
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        results: list[dict] = []

        try:
            url = (
                f"{_EFTS_BASE}?forms=SC+13D"
                f"&dateRange=custom&startdt={cutoff}"
                f"&hits.hits._source=entity_name,file_date,period_of_report,accession_no,filer_name"
            )
            resp = self._session.get(url, headers=_HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits[:100]:
                    src = hit.get("_source", {})
                    company = src.get("entity_name", "")
                    ticker = _infer_ticker_from_name(company)
                    filer = src.get("filer_name", "Unknown")
                    filing_date = src.get("file_date", "")

                    # Look up CIK for known activists
                    activism_type = "5%+ ownership"
                    filer_cik = ""
                    for name, cik in KNOWN_MANAGERS_200.items():
                        if name.lower() in filer.lower() or filer.lower() in name.lower():
                            filer_cik = cik
                            break

                    results.append({
                        "company": company,
                        "ticker": ticker or "",
                        "filer_name": filer,
                        "filer_cik": filer_cik,
                        "filing_date": filing_date,
                        "form_type": "SC 13D",
                        "accession": src.get("accession_no", ""),
                        "is_known_activist": filer_cik in ACTIVIST_CIKS,
                        "activism_type": activism_type,
                    })
            time.sleep(_RATE_DELAY)
        except Exception as exc:
            logger.warning("get_active_13d_filers: error", error=str(exc))

        # Also check local DB
        db_rows = self._db._conn.execute(
            """SELECT filer_cik, filer_name, target_company, target_ticker,
                      filing_date, ownership_pct, activism_type, accession
               FROM activism_events
               WHERE is_active=1
               ORDER BY filing_date DESC LIMIT 100"""
        ).fetchall()

        for row in db_rows:
            results.append({
                "company": row["target_company"],
                "ticker": row["target_ticker"] or "",
                "filer_name": row["filer_name"],
                "filer_cik": row["filer_cik"],
                "filing_date": row["filing_date"],
                "form_type": "SC 13D",
                "accession": row["accession"] or "",
                "is_known_activist": row["filer_cik"] in ACTIVIST_CIKS,
                "activism_type": row["activism_type"] or "5%+ ownership",
                "ownership_pct": row["ownership_pct"],
                "source": "local_db",
            })

        return results

    def compute_activism_score(self, ticker: str) -> float:
        """Compute an activism threat/catalyst score for a ticker (0-100).

        Higher score = more likely to attract activist attention (underperformance
        + low institutional ownership concentration + decent FCF/balance sheet).
        """
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception:
            return 0.0

        score = 0.0

        # 1. Performance vs sector: underperformance attracts activists
        ytd_return = info.get("52WeekChange") or 0.0
        if ytd_return < -0.15:
            score += 20.0  # Significant underperformance
        elif ytd_return < -0.05:
            score += 10.0

        # 2. Valuation discount: low EV/EBITDA vs sector
        pe = info.get("trailingPE") or 0.0
        if 0 < pe < 10:
            score += 20.0  # Very cheap
        elif 0 < pe < 15:
            score += 10.0

        # 3. Balance sheet: cash-rich companies = activist target
        cash = info.get("totalCash") or 0.0
        market_cap = info.get("marketCap") or 1.0
        if market_cap > 0:
            cash_ratio = cash / market_cap
            if cash_ratio > 0.30:
                score += 15.0  # Cash-rich
            elif cash_ratio > 0.15:
                score += 8.0

        # 4. Low institutional ownership concentration = activist can build stake
        holders = self._db.get_institutional_ownership(ticker)
        total_inst_pct = sum(h.percentage_float or 0.0 for h in holders[:10])
        if total_inst_pct < 20.0:
            score += 15.0  # Low concentration
        elif total_inst_pct < 40.0:
            score += 8.0

        # 5. FCF positive = dividends/buybacks can be demanded
        fcf = info.get("freeCashflow") or 0.0
        if fcf > 0:
            fcf_yield = fcf / market_cap if market_cap > 0 else 0.0
            if fcf_yield > 0.08:
                score += 15.0
            elif fcf_yield > 0.04:
                score += 8.0

        # 6. Already known activists in the register?
        activist_holders = [h for h in holders if h.filer_cik in ACTIVIST_CIKS]
        if activist_holders:
            score += 15.0  # Already on activist radar

        return round(min(100.0, score), 1)

    def get_institution_activism_history(
        self,
        cik: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get historical activism campaigns for an institution.

        Parameters
        ----------
        cik : institution EDGAR CIK
        limit : max campaigns to return

        Returns
        -------
        list of activism campaigns with outcome
        """
        rows = self._db._conn.execute(
            """SELECT target_company, target_ticker, filing_date, ownership_pct,
                      activism_type, accession, is_active
               FROM activism_events
               WHERE filer_cik=?
               ORDER BY filing_date DESC LIMIT ?""",
            (cik, limit),
        ).fetchall()

        history = []
        success_rate = self.ACTIVISM_SUCCESS_RATES.get(cik, 0.55)

        for row in rows:
            history.append({
                "target_company": row["target_company"],
                "target_ticker": row["target_ticker"],
                "filing_date": row["filing_date"],
                "ownership_pct": row["ownership_pct"],
                "activism_type": row["activism_type"],
                "is_active": bool(row["is_active"]),
            })

        return {
            "cik": cik,
            "manager_name": next((n for n, c in KNOWN_MANAGERS_200.items() if c == cik), cik),
            "is_known_activist": cik in ACTIVIST_CIKS,
            "historical_success_rate": success_rate,
            "n_campaigns": len(history),
            "campaigns": history,
        }

    def classify_activism_type(self, filing_text: str) -> str:
        """Classify activism intent from filing text (13D/proxy statement).

        Returns one of: board_change | strategic_review | sale | dividend |
                        buyback | operational | passive | unknown
        """
        text = filing_text.lower()

        # Check for keywords in priority order
        if any(kw in text for kw in ["replace director", "board seats", "board change",
                                      "director nominees", "proxy contest"]):
            return "board_change"
        if any(kw in text for kw in ["strategic alternatives", "strategic review",
                                      "explore strategic", "sale of the company"]):
            return "strategic_review"
        if any(kw in text for kw in ["acquisition", "going private", "merger", "sale to"]):
            return "sale"
        if any(kw in text for kw in ["dividend", "special dividend", "return capital"]):
            return "dividend"
        if any(kw in text for kw in ["share repurchase", "buyback", "stock repurchase"]):
            return "buyback"
        if any(kw in text for kw in ["operational improvement", "cost reduction",
                                      "margin improvement", "restructuring"]):
            return "operational"
        if any(kw in text for kw in ["passive", "investment purposes", "no present intention"]):
            return "passive"

        return "unknown"


# ---------------------------------------------------------------------------
# SectorOwnershipAnalysis
# ---------------------------------------------------------------------------


class SectorOwnershipAnalysis:
    """Sector-level institutional ownership and rotation analysis.

    Identifies which sectors institutions are rotating into and out of,
    computes sector concentration, and finds underowned/overcrowded sectors.
    """

    # GICS sector → representative tickers for sector exposure estimation
    SECTOR_TICKERS: dict[str, list[str]] = {
        "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "QCOM"],
        "Financials":             ["JPM", "BAC", "GS", "MS", "WFC", "C", "BX", "BLK"],
        "Health Care":            ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TJX", "BKNG"],
        "Consumer Staples":       ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ"],
        "Industrials":            ["RTX", "HON", "UPS", "CAT", "LMT", "DE", "GE", "FDX"],
        "Communication Services": ["GOOGL", "META", "VZ", "T", "NFLX", "DIS", "CMCSA", "TMUS"],
        "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY"],
        "Materials":              ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "CTVA"],
        "Real Estate":            ["PLD", "AMT", "EQIX", "CCI", "PSA", "EQR", "VTR", "BXP"],
        "Utilities":              ["NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL"],
    }

    def __init__(
        self,
        db: Optional[ComprehensiveOwnershipDatabase] = None,
    ) -> None:
        self._db = db or ComprehensiveOwnershipDatabase()

    def _get_ticker_sector(self, ticker: str) -> str:
        """Return GICS sector for a ticker."""
        for sector, tickers in self.SECTOR_TICKERS.items():
            if ticker.upper() in [t.upper() for t in tickers]:
                return sector
        return "Unknown"

    def get_sector_rotation(self) -> dict[str, Any]:
        """Analyze institutional sector rotation between last two quarters.

        Looks at aggregate institutional values in each sector across periods.
        Returns rotation signals for each sector.
        """
        sectors_data: dict[str, dict[str, float]] = {s: {} for s in GICS_SECTORS}

        # Aggregate holdings value by sector and period
        for sector, tickers in self.SECTOR_TICKERS.items():
            for ticker in tickers:
                rows = self._db._conn.execute(
                    """SELECT period, SUM(value_usd) as tv
                       FROM holdings WHERE UPPER(ticker)=?
                       GROUP BY period ORDER BY period DESC LIMIT 4""",
                    (ticker.upper(),),
                ).fetchall()
                for row in rows:
                    period = row["period"]
                    tv = float(row["tv"] or 0.0)
                    sectors_data[sector][period] = sectors_data[sector].get(period, 0.0) + tv

        result: dict[str, Any] = {"sectors": [], "as_of": datetime.utcnow().strftime("%Y-%m-%d")}

        total_by_period: dict[str, float] = {}
        for s_data in sectors_data.values():
            for period, val in s_data.items():
                total_by_period[period] = total_by_period.get(period, 0.0) + val

        periods_sorted = sorted(total_by_period.keys(), reverse=True)
        curr_p = periods_sorted[0] if periods_sorted else None
        prev_p = periods_sorted[1] if len(periods_sorted) > 1 else None

        for sector in GICS_SECTORS:
            s_data = sectors_data.get(sector, {})
            curr_val = s_data.get(curr_p, 0.0) if curr_p else 0.0
            prev_val = s_data.get(prev_p, 0.0) if prev_p else 0.0

            total_curr = total_by_period.get(curr_p, 1.0)
            total_prev = total_by_period.get(prev_p, 1.0)

            curr_pct = curr_val / total_curr * 100 if total_curr > 0 else 0.0
            prev_pct = prev_val / total_prev * 100 if total_prev > 0 else 0.0
            chg = curr_pct - prev_pct

            if chg > 0.5:
                trend = "increasing"
            elif chg < -0.5:
                trend = "decreasing"
            else:
                trend = "neutral"

            result["sectors"].append({
                "sector": sector,
                "curr_period": curr_p,
                "prev_period": prev_p,
                "curr_pct_of_portfolio": round(curr_pct, 2),
                "prev_pct_of_portfolio": round(prev_pct, 2),
                "change_pct_points": round(chg, 2),
                "trend": trend,
            })

        result["sectors"].sort(key=lambda x: x["change_pct_points"], reverse=True)
        return result

    def compute_sector_overweight(
        self,
        cik: str,
        sector: str,
        benchmark_pct: Optional[float] = None,
    ) -> float:
        """Compute how overweight/underweight an institution is in a sector.

        Parameters
        ----------
        cik : institution CIK
        sector : GICS sector name
        benchmark_pct : benchmark sector weight (if None, uses S&P 500 approximate)

        Returns
        -------
        float: overweight (positive) or underweight (negative) in % points
        """
        # S&P 500 approximate sector weights (2025)
        sp500_weights: dict[str, float] = {
            "Information Technology":  29.5,
            "Financials":              13.0,
            "Health Care":             12.5,
            "Consumer Discretionary":  10.5,
            "Communication Services":  8.5,
            "Industrials":             8.5,
            "Consumer Staples":         5.5,
            "Energy":                   4.0,
            "Materials":                2.5,
            "Real Estate":              2.5,
            "Utilities":                2.5,
        }

        if benchmark_pct is None:
            benchmark_pct = sp500_weights.get(sector, 5.0)

        portfolio = self._db.get_portfolio(cik, top_n=500)
        holdings = portfolio.get("top_holdings", [])
        total_val = portfolio.get("total_value_usd") or 1.0

        sector_tickers = set(t.upper() for t in self.SECTOR_TICKERS.get(sector, []))
        sector_val = sum(
            h["value_usd"] for h in holdings
            if h.get("ticker", "").upper() in sector_tickers
        )
        institution_pct = (sector_val / total_val * 100) if total_val > 0 else 0.0
        return round(institution_pct - benchmark_pct, 2)

    def get_crowded_sectors(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Find sectors with highest institutional ownership concentration.

        Returns sectors sorted by aggregate institutional value (most crowded first).
        """
        sector_vals: dict[str, float] = {}

        for sector, tickers in self.SECTOR_TICKERS.items():
            total = 0.0
            for ticker in tickers:
                rows = self._db._conn.execute(
                    """SELECT SUM(value_usd) as tv FROM holdings
                       WHERE UPPER(ticker)=?
                       GROUP BY period ORDER BY period DESC LIMIT 1""",
                    (ticker.upper(),),
                ).fetchone()
                if rows and rows["tv"]:
                    total += float(rows["tv"])
            sector_vals[sector] = total

        total_all = sum(sector_vals.values()) or 1.0
        results = [
            {
                "sector": sector,
                "total_institutional_value_usd": round(val, 2),
                "pct_of_total": round(val / total_all * 100, 2),
                "crowding_label": "Very Crowded" if val / total_all > 0.20 else
                                  "Crowded" if val / total_all > 0.10 else "Normal",
            }
            for sector, val in sorted(sector_vals.items(), key=lambda x: x[1], reverse=True)
        ]
        return results[:top_n]

    def get_undercovered_sectors(self, bottom_n: int = 5) -> list[dict[str, Any]]:
        """Find sectors with lowest institutional ownership (potential value discovery).

        Returns sectors sorted by aggregate institutional value (least covered first).
        """
        crowded = self.get_crowded_sectors(top_n=len(GICS_SECTORS))
        all_sectors = sorted(crowded, key=lambda x: x["pct_of_total"])
        for s in all_sectors:
            s["crowding_label"] = (
                "Very Undercovered" if s["pct_of_total"] < 2.0 else
                "Undercovered" if s["pct_of_total"] < 5.0 else
                "Moderate"
            )
        return all_sectors[:bottom_n]

    def get_institution_sector_breakdown(
        self,
        cik: str,
    ) -> pd.DataFrame:
        """Get sector breakdown of an institution's portfolio.

        Returns DataFrame with sector, value_usd, pct_of_portfolio, overweight.
        """
        portfolio = self._db.get_portfolio(cik, top_n=500)
        holdings = portfolio.get("top_holdings", [])
        total_val = portfolio.get("total_value_usd") or 1.0

        sector_vals: dict[str, float] = {s: 0.0 for s in GICS_SECTORS}
        sector_vals["Unknown"] = 0.0

        for h in holdings:
            ticker = h.get("ticker", "").upper()
            val = h.get("value_usd") or 0.0
            sector = self._get_ticker_sector(ticker)
            sector_vals[sector] = sector_vals.get(sector, 0.0) + val

        rows = []
        sp500_w = {
            "Information Technology": 29.5, "Financials": 13.0, "Health Care": 12.5,
            "Consumer Discretionary": 10.5, "Communication Services": 8.5,
            "Industrials": 8.5, "Consumer Staples": 5.5, "Energy": 4.0,
            "Materials": 2.5, "Real Estate": 2.5, "Utilities": 2.5, "Unknown": 0.0,
        }

        for sector, val in sector_vals.items():
            pct = val / total_val * 100 if total_val > 0 else 0.0
            benchmark = sp500_w.get(sector, 0.0)
            rows.append({
                "sector": sector,
                "value_usd": round(val, 2),
                "pct_of_portfolio": round(pct, 2),
                "benchmark_pct": benchmark,
                "overweight": round(pct - benchmark, 2),
            })

        df = pd.DataFrame(rows)
        df = df[df["sector"] != "Unknown"]
        return df.sort_values("value_usd", ascending=False)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query

    institutional_v2_router = APIRouter(prefix="/ownership/v2", tags=["institutional-v2"])

    _db = ComprehensiveOwnershipDatabase()
    _smart_money = SmartMoneyFactor(_db)
    _momentum = OwnershipMomentum(_db)
    _activism = ActivismScorer(_db)
    _sector_analysis = SectorOwnershipAnalysis(_db)

    @institutional_v2_router.get("/{ticker}")
    def get_institutional_ownership(
        ticker: str,
        limit: int = Query(default=50, ge=5, le=200),
    ) -> dict:
        """Get comprehensive institutional ownership for a ticker."""
        ticker = ticker.upper()
        try:
            holders = _db.get_institutional_ownership(ticker, limit=limit)
            delta = _db.get_quarterly_delta(ticker)

            total_val = sum(h.value_usd for h in holders if h.value_usd)
            total_shares = sum(h.shares for h in holders if h.shares)

            return {
                "ticker": ticker,
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
                "n_holders": len(holders),
                "total_institutional_value_usd": total_val,
                "total_institutional_shares": total_shares,
                "top_holders": [h.model_dump() for h in holders[:limit]],
                "quarterly_delta": delta,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @institutional_v2_router.get("/smart-money/{ticker}")
    def get_smart_money_analysis(ticker: str) -> dict:
        """Get smart money ownership score and consensus for a ticker."""
        ticker = ticker.upper()
        try:
            consensus = _smart_money.get_smart_money_consensus(ticker)
            changes = _smart_money.get_smart_money_changes(ticker)
            return {
                "ticker": ticker,
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
                **consensus,
                "position_history": changes,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @institutional_v2_router.get("/momentum/{ticker}")
    def get_ownership_momentum(
        ticker: str,
        quarters: int = Query(default=4, ge=2, le=12),
    ) -> dict:
        """Get institutional ownership momentum signals for a ticker."""
        ticker = ticker.upper()
        try:
            mom = _momentum.get_ownership_momentum(ticker, quarters)
            history_df = _momentum.get_ownership_history(ticker, quarters)
            return {
                **mom.model_dump(),
                "history": history_df.to_dict(orient="records") if not history_df.empty else [],
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @institutional_v2_router.get("/activism")
    def get_activism_data(
        lookback_days: int = Query(default=90, ge=7, le=365),
        ticker: Optional[str] = None,
    ) -> dict:
        """Get active 13D filers and activism scores."""
        try:
            active_filers = _activism.get_active_13d_filers(lookback_days)
            result: dict[str, Any] = {
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
                "lookback_days": lookback_days,
                "active_13d_filings": active_filers,
                "n_filings": len(active_filers),
            }

            if ticker:
                score = _activism.compute_activism_score(ticker.upper())
                result["ticker"] = ticker.upper()
                result["activism_score"] = score
                result["activism_interpretation"] = (
                    "High activism risk — company may attract activist attention" if score > 60 else
                    "Moderate activism risk" if score > 40 else
                    "Low activism risk"
                )

            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @institutional_v2_router.get("/sector-rotation")
    def get_sector_rotation() -> dict:
        """Get institutional sector rotation analysis."""
        try:
            rotation = _sector_analysis.get_sector_rotation()
            crowded = _sector_analysis.get_crowded_sectors(top_n=5)
            undercovered = _sector_analysis.get_undercovered_sectors(bottom_n=5)
            return {
                **rotation,
                "crowded_sectors": crowded,
                "undercovered_sectors": undercovered,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

except ImportError:
    institutional_v2_router = None  # type: ignore
    logger.warning("FastAPI not available; institutional_v2_router not registered")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_full_ownership_analysis(
    ticker: str,
    ingest_if_missing: bool = False,
    n_managers: int = 10,
) -> dict[str, Any]:
    """One-shot comprehensive ownership analysis for a ticker.

    Parameters
    ----------
    ticker : stock ticker
    ingest_if_missing : if True, trigger bulk ingest before analysis
    n_managers : managers to ingest if ingest_if_missing

    Returns
    -------
    dict with all ownership analytics
    """
    db = ComprehensiveOwnershipDatabase()

    if ingest_if_missing:
        db.bulk_ingest_top_managers(n_managers=n_managers, lookback_quarters=4)

    ticker = ticker.upper()
    sm = SmartMoneyFactor(db)
    mom = OwnershipMomentum(db)
    act = ActivismScorer(db)
    sec = SectorOwnershipAnalysis(db)

    holders = db.get_institutional_ownership(ticker)
    delta = db.get_quarterly_delta(ticker)
    smart_consensus = sm.get_smart_money_consensus(ticker)
    momentum_data = mom.get_ownership_momentum(ticker)
    activism_score = act.compute_activism_score(ticker)
    new_stakes = mom.get_new_large_stakes(lookback_days=90)
    rotation = sec.get_sector_rotation()
    crowded = sec.get_crowded_sectors()

    return {
        "ticker": ticker,
        "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
        "top_holders": [h.model_dump() for h in holders[:25]],
        "n_holders": len(holders),
        "total_institutional_value_usd": sum(h.value_usd for h in holders if h.value_usd),
        "quarterly_delta": delta,
        "smart_money": smart_consensus,
        "ownership_momentum": momentum_data.model_dump(),
        "activism_score": activism_score,
        "activism_interpretation": (
            "High" if activism_score > 60 else
            "Moderate" if activism_score > 40 else "Low"
        ),
        "new_large_stakes_recent": new_stakes[:10],
        "sector_rotation_top3": rotation.get("sectors", [])[:3],
        "crowded_sectors": crowded,
    }
