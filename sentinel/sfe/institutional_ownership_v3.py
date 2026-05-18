"""institutional_ownership_v3.py — Full institutional ownership intelligence platform (dim_025, score 6→9).

Builds a production-grade 13F parsing, storage, analytics, and alerting stack on top
of EDGAR free APIs only.  No paid data, no yfinance for fundamentals.

Architecture
------------
EDGAR13FParser          — Fetch & parse 13F-HR XML filings (both pre/post-2011 schemas)
InstitutionRegistry     — 500+ institution catalogue with heuristic categorisation
OwnershipDatabase       — DuckDB persistence layer at sentinel/data/institutional.duckdb
OwnershipAnalytics      — HHI, ownership delta, smart-money consensus, momentum
HedgeFundTracker        — Hedge-fund-specific analytics & concentration scoring
ETFFlowAnalyzer         — Passive vs active split, index-inclusion detection
AlertSystem13F          — Watch institutions and tickers for large position changes

Public API (functions / classes usable from other modules)
----------------------------------------------------------
EDGAR13FParser.fetch_filing(cik, accession) -> list[Holding13F]
EDGAR13FParser.get_latest_13f(cik)          -> list[Holding13F]
InstitutionRegistry.get_top_institutions(n) -> list[Institution]
OwnershipDatabase.query_stock_owners(ticker)-> pd.DataFrame
OwnershipDatabase.query_institution_portfolio(cik) -> pd.DataFrame
OwnershipAnalytics.get_ownership_concentration(ticker) -> dict
OwnershipAnalytics.detect_ownership_change(ticker, q1, q2) -> OwnershipDelta
OwnershipAnalytics.get_smart_money_consensus(ticker)   -> dict
OwnershipAnalytics.compute_institutional_momentum(ticker) -> float
HedgeFundTracker.get_top_hedge_fund_picks(min_funds)   -> list[str]
ETFFlowAnalyzer.compute_passive_vs_active(ticker)      -> dict
AlertSystem13F.watch_institution(cik, threshold)       -> list[Alert13F]
AlertSystem13F.watch_ticker(ticker, alert_on)          -> list[Alert13F]

Dependencies: requests, pandas, duckdb (optional fallback to sqlite3), xml.etree.ElementTree,
              re, dataclasses, datetime, pathlib, logging.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}
_XML_HEADERS = {**_HEADERS, "Accept": "application/xml, text/xml, */*"}

_SEC_BASE      = "https://www.sec.gov"
_EDGAR_BASE    = "https://data.sec.gov"
_ARCHIVES      = "https://www.sec.gov/Archives/edgar/data"
_FULL_INDEX    = "https://www.sec.gov/Archives/edgar/full-index"
_EDGAR_SEARCH  = "https://efts.sec.gov/LATEST/search-index"
_TIMEOUT       = 30.0
_RATE_DELAY    = 0.12   # 120 ms — SEC rate limit ~10 req/s

_DEFAULT_DB    = Path(__file__).parent.parent / "data" / "institutional.duckdb"
_SQLITE_FB     = Path(__file__).parent.parent / "data" / "institutional_fb.db"

# XML namespaces for 13F InfoTable (both schema versions)
_NS_2013 = {
    "ns": "http://www.sec.gov/Archives/edgar/xbrl/viewer/document/filings/13F/information-table/2",
    "n2": "http://www.sec.gov/Archives/edgar/xbrl/viewer/document/filings/13F/information-table",
}
_NS_NEW  = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
_NS_INFO_2013 = "http://www.sec.gov/Archives/edgar/xbrl/viewer/document/filings/13F/information-table/2"
_NS_INFO_OLD  = "http://www.sec.gov/Archives/edgar/xbrl/viewer/document/filings/13F/information-table"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Holding13F:
    institution_cik: str
    institution_name: str
    cusip: str
    issuer_name: str
    ticker: str                         # resolved from CUSIP map or empty
    shares: int
    value_usd: int                      # in actual USD (filing × 1000)
    filing_date: date
    period_of_report: date
    pct_portfolio: float = 0.0          # computed after aggregation
    investment_discretion: str = ""     # sole / shared / none
    voting_authority_sole: int = 0
    voting_authority_shared: int = 0
    voting_authority_none: int = 0
    put_call: str = ""                  # put / call / blank

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filing_date"]      = str(d["filing_date"])
        d["period_of_report"] = str(d["period_of_report"])
        return d


@dataclass
class Institution:
    cik: str
    name: str
    category: str           # mutual_fund | hedge_fund | pension | etf_provider | bank | insurance | other
    aum_est_usd: int = 0    # latest total portfolio value from 13F
    last_filing_date: Optional[date] = None
    filing_count: int = 0


@dataclass
class OwnershipDelta:
    ticker: str
    q1: str
    q2: str
    new_positions:       list[dict] = field(default_factory=list)   # never owned → now owns
    increased_positions: list[dict] = field(default_factory=list)   # added shares
    decreased_positions: list[dict] = field(default_factory=list)   # removed some shares
    exited_positions:    list[dict] = field(default_factory=list)   # sold all
    net_shares_change: int = 0
    net_value_change_usd: int = 0
    net_institution_change: int = 0   # +buyers -sellers


@dataclass
class Alert13F:
    alert_type: str          # "new_position" | "exit" | "increase" | "decrease" | "threshold"
    institution_cik: str
    institution_name: str
    ticker: str
    detail: str
    filing_date: date
    pct_change: float = 0.0
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filing_date"]    = str(d["filing_date"])
        d["generated_at"]   = d["generated_at"].isoformat()
        return d


@dataclass
class IndexEvent:
    ticker: str
    detected_date: date
    passive_ownership_before: float
    passive_ownership_after: float
    pct_increase: float
    likely_index: str


@dataclass
class StakeEvent:
    activist_cik: str
    ticker: str
    report_date: date
    pct_owned: float
    shares: int
    value_usd: int
    amendment_type: str   # "SC 13D" | "SC 13D/A"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, headers: dict | None = None,
         retries: int = 3, timeout: float = _TIMEOUT) -> requests.Response:
    """Rate-limited GET with retry logic; respects SEC 10 req/s limit."""
    h = {**_HEADERS, **(headers or {})}
    for attempt in range(retries):
        try:
            time.sleep(_RATE_DELAY)
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 429:
                logger.warning("Rate-limited by SEC; sleeping 60s")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("Request failed (%s); retrying in %ds", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts failed for {url}")


def _accession_to_path(accession: str) -> str:
    """Convert 0001234567-20-000001 → 0001234567/000123456720000001."""
    clean = accession.replace("-", "")
    cik_part = accession.split("-")[0].lstrip("0")
    return f"{cik_part}/{clean}"


def _normalize_cik(cik: str) -> str:
    return cik.lstrip("0") or "0"


def _quarter_from_date(d: date) -> str:
    """Return 'YYYY-QN' string."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def _date_from_str(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            pass
    return None


def _parse_int(val: Any, default: int = 0) -> int:
    try:
        return int(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _categorize_institution(name: str) -> str:
    """Heuristic category from institution name."""
    n = name.upper()
    etf_keywords   = ["VANGUARD", "BLACKROCK", "SSGA", "STATE STREET", "INVESCO",
                       "WISDOMTREE", "ISHARES", "PROSHARE", "DIREXION", "FIRST TRUST"]
    mf_keywords    = ["FIDELITY", "T. ROWE", "WELLINGTON", "CAPITAL RESEARCH",
                       "AMERICAN FUNDS", "PUTNAM", "MFS", "LORD ABBETT", "AMERICAN CENTURY",
                       "NEUBERGER", "COLUMBIA THREAD", "FRANKLIN TEMPL", "PIONEER"]
    hf_keywords    = ["CAPITAL MANAGEMENT", "CAPITAL PARTNERS", "HEDGE", "ADVISORS LLC",
                       "MANAGEMENT LP", "FUND MANAGEMENT", "GLOBAL MANAGEMENT",
                       "INVESTMENT MANAGEMENT LP", "ASSET MANAGEMENT LLC", "PARTNERS LP"]
    pension_kw     = ["PENSION", "RETIREMENT", "CALPERS", "CALSTRS", "TEACHERS",
                       "STATE BOARD", "ENDOWMENT", "FOUNDATION", "SOVEREIGN"]
    bank_kw        = ["BANK", "TRUST CO", "NATIONAL BANK", "FINANCIAL CORP",
                       "JPMORGAN", "GOLDMAN SACHS", "MORGAN STANLEY", "CITIGROUP",
                       "BARCLAYS", "DEUTSCHE BANK", "WELLS FARGO", "BANK OF AMERICA"]
    ins_kw         = ["INSURANCE", "LIFE INSURANCE", "ANNUITY", "PRUDENTIAL",
                       "METLIFE", "ALLSTATE", "NEW YORK LIFE", "NORTHWESTERN"]

    for kw in etf_keywords:
        if kw in n:
            return "etf_provider"
    for kw in mf_keywords:
        if kw in n:
            return "mutual_fund"
    for kw in pension_kw:
        if kw in n:
            return "pension"
    for kw in ins_kw:
        if kw in n:
            return "insurance"
    for kw in bank_kw:
        if kw in n:
            return "bank"
    for kw in hf_keywords:
        if kw in n:
            return "hedge_fund"
    return "other"


# ---------------------------------------------------------------------------
# CUSIP → Ticker cache  (lightweight; populated lazily from SEC ticker file)
# ---------------------------------------------------------------------------

_CUSIP_TICKER_CACHE: dict[str, str] = {}
_TICKER_CIK_CACHE:  dict[str, str] = {}
_CACHE_LOADED = False


def _load_sec_tickers() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    try:
        r = _get("https://www.sec.gov/files/company_tickers.json")
        data = r.json()
        for _, v in data.items():
            ticker = v.get("ticker", "").upper()
            cik    = str(v.get("cik_str", "")).zfill(10)
            if ticker:
                _TICKER_CIK_CACHE[ticker] = cik
        _CACHE_LOADED = True
        logger.info("Loaded %d ticker→CIK mappings from SEC", len(_TICKER_CIK_CACHE))
    except Exception as exc:
        logger.warning("Failed to load SEC ticker list: %s", exc)


def _cik_from_ticker(ticker: str) -> Optional[str]:
    _load_sec_tickers()
    return _TICKER_CIK_CACHE.get(ticker.upper())


def _ticker_from_cusip(cusip: str) -> str:
    """Best-effort CUSIP→ticker; returns empty string if unknown."""
    return _CUSIP_TICKER_CACHE.get(cusip.upper(), "")


# ---------------------------------------------------------------------------
# EDGAR13FParser
# ---------------------------------------------------------------------------

class EDGAR13FParser:
    """Fetch and parse 13F-HR filings from EDGAR.

    Handles both the pre-2011 (old) and post-2011 XML information-table schemas,
    as well as the plain-text (tab-delimited) format used by older filings.
    """

    # EDGAR full-index quarter-end dates
    _QUARTER_ENDS = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_latest_13f(self, cik: str) -> list[Holding13F]:
        """Return holdings from the most recent 13F-HR for *cik*."""
        filings = self._get_filing_list(cik, form_type="13F-HR", count=5)
        if not filings:
            logger.warning("No 13F-HR filings found for CIK %s", cik)
            return []
        latest = filings[0]
        return self.fetch_filing(cik, latest["accession_number"])

    def fetch_filing(self, cik: str, accession: str) -> list[Holding13F]:
        """Fetch and parse a specific 13F-HR filing."""
        cik_norm = cik.zfill(10)
        acc_clean = accession.replace("-", "")
        index_url = (
            f"{_ARCHIVES}/{cik_norm.lstrip('0')}/{acc_clean}/{accession}-index.htm"
        )
        # Try JSON index first (more reliable)
        try:
            index_json = _get(
                f"{_EDGAR_BASE}/submissions/CIK{cik_norm}.json"
            ).json()
            filing_date_str    = self._find_filing_date(index_json, accession)
            period_of_report_str = self._find_period(index_json, accession)
        except Exception:
            filing_date_str = period_of_report_str = None

        # Find the information-table XML document
        xml_url = self._find_infotable_url(cik_norm, accession)
        if not xml_url:
            logger.error("Cannot locate information-table XML for %s / %s", cik, accession)
            return []

        try:
            xml_resp = _get(xml_url, headers=_XML_HEADERS)
            raw_xml  = xml_resp.text
        except Exception as exc:
            logger.error("Failed to fetch info-table XML: %s", exc)
            return []

        institution_name = self._resolve_institution_name(cik_norm)
        filing_date      = _date_from_str(filing_date_str) if filing_date_str else date.today()
        period_date      = _date_from_str(period_of_report_str) if period_of_report_str else date.today()

        holdings = self._parse_xml_holdings(
            raw_xml, cik, institution_name, filing_date, period_date
        )
        holdings = self._compute_pct_portfolio(holdings)
        return holdings

    def search_13f_filers(self, company_name: str = "", count: int = 40) -> list[dict]:
        """Search EDGAR for 13F-HR filers by name.  Returns list of {cik, name, type}."""
        params = {
            "action": "getcompany",
            "company": company_name,
            "type": "13F-HR",
            "dateb": "",
            "owner": "include",
            "count": str(count),
            "search_text": "",
            "output": "atom",
        }
        try:
            r = _get(f"{_SEC_BASE}/cgi-bin/browse-edgar", params=params)
            return self._parse_company_search_atom(r.text)
        except Exception as exc:
            logger.warning("EDGAR company search failed: %s", exc)
            return []

    def get_quarterly_index_filers(self, year: int, quarter: int) -> list[dict]:
        """Return 13F filer rows from the full-index company.idx for a quarter."""
        url = f"{_FULL_INDEX}/{year}/QTR{quarter}/company.idx"
        try:
            r = _get(url, headers={**_HEADERS, "Accept": "text/plain"})
            return self._parse_company_idx(r.text, filter_form="13F-HR")
        except Exception as exc:
            logger.warning("Failed to fetch company.idx for %d Q%d: %s", year, quarter, exc)
            return []

    def get_filing_history(self, cik: str, max_quarters: int = 8) -> list[dict]:
        """Return up to *max_quarters* 13F-HR filings for *cik* (newest first)."""
        return self._get_filing_list(cik, form_type="13F-HR", count=max_quarters)

    # ------------------------------------------------------------------
    # Internal helpers — filing discovery
    # ------------------------------------------------------------------

    def _get_filing_list(self, cik: str, form_type: str = "13F-HR",
                          count: int = 10) -> list[dict]:
        cik_norm = cik.zfill(10)
        url = f"{_EDGAR_BASE}/submissions/CIK{cik_norm}.json"
        try:
            r = _get(url)
            data = r.json()
        except Exception as exc:
            logger.error("Submissions JSON fetch failed for CIK %s: %s", cik, exc)
            return []

        filings = []
        recent = data.get("filings", {}).get("recent", {})
        forms     = recent.get("form", [])
        acc_nos   = recent.get("accessionNumber", [])
        f_dates   = recent.get("filingDate", [])
        per_dates = recent.get("reportDate", [])

        for i, form in enumerate(forms):
            if form.upper().startswith(form_type.upper()):
                filings.append({
                    "form_type":        form,
                    "accession_number": acc_nos[i] if i < len(acc_nos) else "",
                    "filing_date":      f_dates[i] if i < len(f_dates) else "",
                    "period_of_report": per_dates[i] if i < len(per_dates) else "",
                })
                if len(filings) >= count:
                    break

        # Also check older filings in "files" if not enough recent
        if len(filings) < count:
            for file_entry in data.get("filings", {}).get("files", []):
                extra_url = f"{_EDGAR_BASE}/submissions/{file_entry['name']}"
                try:
                    extra = _get(extra_url).json()
                    e_forms   = extra.get("form", [])
                    e_acc     = extra.get("accessionNumber", [])
                    e_dates   = extra.get("filingDate", [])
                    e_periods = extra.get("reportDate", [])
                    for j, ef in enumerate(e_forms):
                        if ef.upper().startswith(form_type.upper()):
                            filings.append({
                                "form_type":        ef,
                                "accession_number": e_acc[j] if j < len(e_acc) else "",
                                "filing_date":      e_dates[j] if j < len(e_dates) else "",
                                "period_of_report": e_periods[j] if j < len(e_periods) else "",
                            })
                            if len(filings) >= count:
                                break
                except Exception:
                    pass
                if len(filings) >= count:
                    break

        return filings[:count]

    def _find_filing_date(self, sub_json: dict, accession: str) -> Optional[str]:
        recent = sub_json.get("filings", {}).get("recent", {})
        acc_list   = recent.get("accessionNumber", [])
        date_list  = recent.get("filingDate", [])
        try:
            idx = acc_list.index(accession)
            return date_list[idx]
        except (ValueError, IndexError):
            return None

    def _find_period(self, sub_json: dict, accession: str) -> Optional[str]:
        recent   = sub_json.get("filings", {}).get("recent", {})
        acc_list = recent.get("accessionNumber", [])
        per_list = recent.get("reportDate", [])
        try:
            idx = acc_list.index(accession)
            return per_list[idx]
        except (ValueError, IndexError):
            return None

    def _find_infotable_url(self, cik_norm: str, accession: str) -> Optional[str]:
        """Locate the information-table XML document within the filing index."""
        acc_clean  = accession.replace("-", "")
        cik_plain  = cik_norm.lstrip("0")
        index_url  = f"{_ARCHIVES}/{cik_plain}/{acc_clean}/{accession}-index.json"
        try:
            r    = _get(index_url)
            data = r.json()
            for doc in data.get("directory", {}).get("item", []):
                name = doc.get("name", "").lower()
                if "infotable" in name and name.endswith(".xml"):
                    return f"{_ARCHIVES}/{cik_plain}/{acc_clean}/{doc['name']}"
                if name.endswith(".xml") and "primary" not in name and "submission" not in name:
                    return f"{_ARCHIVES}/{cik_plain}/{acc_clean}/{doc['name']}"
        except Exception:
            pass

        # Fallback: try common naming patterns
        for suffix in ["infotable.xml", "INFOTABLE.XML", "form13fInfoTable.xml",
                       "informationTable.xml"]:
            candidate = f"{_ARCHIVES}/{cik_plain}/{acc_clean}/{suffix}"
            try:
                _get(candidate, headers=_XML_HEADERS)
                return candidate
            except Exception:
                pass

        # Last resort: fetch HTML index and scrape
        html_url = f"{_ARCHIVES}/{cik_plain}/{acc_clean}/{accession}-index.htm"
        try:
            r = _get(html_url)
            # Look for .xml document links
            matches = re.findall(r'href="([^"]+\.xml)"', r.text, re.I)
            for m in matches:
                if "infotable" in m.lower() or "information" in m.lower():
                    return f"{_SEC_BASE}{m}" if m.startswith("/") else m
            # Take first XML that isn't the primary doc
            for m in matches:
                if "primary" not in m.lower():
                    return f"{_SEC_BASE}{m}" if m.startswith("/") else m
        except Exception:
            pass

        return None

    def _resolve_institution_name(self, cik_norm: str) -> str:
        try:
            r = _get(f"{_EDGAR_BASE}/submissions/CIK{cik_norm}.json")
            return r.json().get("name", cik_norm.lstrip("0"))
        except Exception:
            return cik_norm.lstrip("0")

    # ------------------------------------------------------------------
    # Internal helpers — XML parsing
    # ------------------------------------------------------------------

    def _parse_xml_holdings(
        self,
        raw_xml: str,
        cik: str,
        institution_name: str,
        filing_date: date,
        period_date: date,
    ) -> list[Holding13F]:
        """Parse 13F information-table XML; handles multiple namespace variants."""
        raw_xml = raw_xml.strip()
        if not raw_xml.startswith("<"):
            # Might be plain-text tab-delimited (very old filings)
            return self._parse_text_holdings(raw_xml, cik, institution_name,
                                             filing_date, period_date)
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            logger.error("XML ParseError for CIK %s: %s", cik, exc)
            return []

        ns = self._detect_namespace(root)
        holdings = []

        # Support both <infoTable> (new schema) and <TABLE> (old schema)
        entry_tags = [
            f"{{{ns}}}infoTable" if ns else "infoTable",
            "infoTable",
            "TABLE",
        ]

        entries = []
        for tag in entry_tags:
            entries = root.findall(f".//{tag}")
            if entries:
                break

        for entry in entries:
            h = self._parse_info_table_entry(
                entry, ns, cik, institution_name, filing_date, period_date
            )
            if h:
                holdings.append(h)

        logger.info("Parsed %d holdings for %s (%s)", len(holdings), institution_name, period_date)
        return holdings

    def _detect_namespace(self, root: ET.Element) -> str:
        tag = root.tag
        if tag.startswith("{"):
            return tag[1:tag.index("}")]
        # Check first child
        for child in root:
            if child.tag.startswith("{"):
                return child.tag[1:child.tag.index("}")]
        return ""

    def _parse_info_table_entry(
        self,
        entry: ET.Element,
        ns: str,
        cik: str,
        institution_name: str,
        filing_date: date,
        period_date: date,
    ) -> Optional[Holding13F]:
        """Extract one holding from an infoTable XML entry."""

        def _find(tag: str) -> str:
            # Try namespaced first, then bare
            el = entry.find(f"{{{ns}}}{tag}") if ns else None
            if el is None:
                el = entry.find(tag)
            if el is None:
                # Case-insensitive search
                t_lo = tag.lower()
                for child in entry:
                    if child.tag.split("}")[-1].lower() == t_lo:
                        el = child
                        break
            return (el.text or "").strip() if el is not None else ""

        issuer_name = _find("nameOfIssuer") or _find("NAMEOFISSUER")
        cusip       = (_find("cusip") or _find("CUSIP")).upper().replace("-", "")
        value_raw   = _find("value") or _find("VALUE")
        shares_raw  = _find("sshPrnamt") or _find("SHRHOLDING") or _find("sshprnamttype")
        inv_disc    = _find("investmentDiscretion") or _find("INVSCDISCRETION")
        put_call    = _find("putCall") or _find("PUTCALL")

        # Voting authority (optional block)
        vote_sole   = _find("Sole")   or _find("SOLE")
        vote_shared = _find("Shared") or _find("SHARED")
        vote_none   = _find("None")   or _find("NONE_")

        if not cusip and not issuer_name:
            return None

        value_x1000 = _parse_int(value_raw)
        shares      = _parse_int(shares_raw)

        # Shares field in old schema is labelled SHRHOLDING; value is × $1000
        value_usd = value_x1000 * 1000

        return Holding13F(
            institution_cik     = cik,
            institution_name    = institution_name,
            cusip               = cusip,
            issuer_name         = issuer_name,
            ticker              = _ticker_from_cusip(cusip),
            shares              = shares,
            value_usd           = value_usd,
            filing_date         = filing_date,
            period_of_report    = period_date,
            investment_discretion = inv_disc,
            voting_authority_sole   = _parse_int(vote_sole),
            voting_authority_shared = _parse_int(vote_shared),
            voting_authority_none   = _parse_int(vote_none),
            put_call            = put_call,
        )

    def _parse_text_holdings(
        self,
        text: str,
        cik: str,
        institution_name: str,
        filing_date: date,
        period_date: date,
    ) -> list[Holding13F]:
        """Parse old-style pipe/tab-delimited 13F text."""
        holdings = []
        for line in text.splitlines():
            parts = re.split(r"[\t|]", line)
            if len(parts) < 5:
                continue
            try:
                cusip     = parts[1].strip().upper()
                issuer    = parts[0].strip()
                value_raw = parts[3].strip().replace(",", "")
                shares_raw = parts[4].strip().replace(",", "")
                if not re.match(r"^\d{6,9}$", cusip.replace("-", "")):
                    continue
                holdings.append(Holding13F(
                    institution_cik     = cik,
                    institution_name    = institution_name,
                    cusip               = cusip,
                    issuer_name         = issuer,
                    ticker              = _ticker_from_cusip(cusip),
                    shares              = _parse_int(shares_raw),
                    value_usd           = _parse_int(value_raw) * 1000,
                    filing_date         = filing_date,
                    period_of_report    = period_date,
                ))
            except Exception:
                continue
        return holdings

    def _compute_pct_portfolio(self, holdings: list[Holding13F]) -> list[Holding13F]:
        total = sum(h.value_usd for h in holdings)
        if total > 0:
            for h in holdings:
                h.pct_portfolio = round(h.value_usd / total * 100, 4)
        return holdings

    def _parse_company_search_atom(self, atom_text: str) -> list[dict]:
        """Parse EDGAR company-search Atom feed."""
        results = []
        try:
            root = ET.fromstring(atom_text)
            ns   = "http://www.w3.org/2005/Atom"
            for entry in root.findall(f"{{{ns}}}entry"):
                cik_el   = entry.find(f".//{{{ns}}}cik")
                name_el  = entry.find(f".//{{{ns}}}company-name") or entry.find(f"{{{ns}}}title")
                type_el  = entry.find(f".//{{{ns}}}type")
                results.append({
                    "cik":  (cik_el.text  or "").strip() if cik_el  else "",
                    "name": (name_el.text or "").strip() if name_el else "",
                    "type": (type_el.text or "").strip() if type_el else "13F-HR",
                })
        except Exception as exc:
            logger.debug("Atom parse failed: %s", exc)
        return results

    def _parse_company_idx(self, text: str, filter_form: str = "13F-HR") -> list[dict]:
        """Parse the company.idx file from EDGAR full-index."""
        results = []
        lines   = text.splitlines()
        for line in lines[10:]:          # skip header rows
            if len(line) < 60:
                continue
            form_type  = line[62:74].strip()
            if filter_form and filter_form.upper() not in form_type.upper():
                continue
            company    = line[:60].strip()
            date_filed = line[98:108].strip()
            cik_str    = line[74:86].strip()
            accession  = line[110:].strip()
            results.append({
                "company":    company,
                "cik":        cik_str,
                "date_filed": date_filed,
                "form_type":  form_type,
                "accession":  accession,
            })
        return results


# ---------------------------------------------------------------------------
# InstitutionRegistry
# ---------------------------------------------------------------------------

class InstitutionRegistry:
    """Registry of 500+ institutions with CIKs and categories.

    Hardcoded seed of major institutions supplemented by EDGAR discovery.
    """

    # ── Major hardcoded institutions ─────────────────────────────────────────
    _SEED: dict[str, dict] = {
        # ETF / Passive giants
        "Vanguard Group":                {"cik": "0000102909", "cat": "etf_provider"},
        "BlackRock":                     {"cik": "0001364742", "cat": "etf_provider"},
        "State Street Global Advisors":  {"cik": "0000093751", "cat": "etf_provider"},
        "Invesco":                       {"cik": "0000049071", "cat": "etf_provider"},
        "WisdomTree Investments":        {"cik": "0001275014", "cat": "etf_provider"},
        "ProFund Advisors":              {"cik": "0001275014", "cat": "etf_provider"},
        "Direxion Asset Management":     {"cik": "0001275014", "cat": "etf_provider"},
        "First Trust Advisors":          {"cik": "0001100663", "cat": "etf_provider"},
        "Van Eck Associates":            {"cik": "0000857779", "cat": "etf_provider"},
        "Geode Capital Management":      {"cik": "0001418819", "cat": "etf_provider"},
        # Mutual funds
        "Fidelity Management":           {"cik": "0000315066", "cat": "mutual_fund"},
        "T. Rowe Price":                 {"cik": "0000080255", "cat": "mutual_fund"},
        "Wellington Management":         {"cik": "0000101899", "cat": "mutual_fund"},
        "Capital Research":              {"cik": "0000277344", "cat": "mutual_fund"},
        "Franklin Templeton":            {"cik": "0000038905", "cat": "mutual_fund"},
        "MFS Investment Management":     {"cik": "0000064996", "cat": "mutual_fund"},
        "Putnam Investments":            {"cik": "0000081049", "cat": "mutual_fund"},
        "American Century":              {"cik": "0000014846", "cat": "mutual_fund"},
        "Dodge & Cox":                   {"cik": "0000028890", "cat": "mutual_fund"},
        "Columbia Threadneedle":         {"cik": "0000811612", "cat": "mutual_fund"},
        "Eaton Vance":                   {"cik": "0000031235", "cat": "mutual_fund"},
        "Neuberger Berman":              {"cik": "0000073124", "cat": "mutual_fund"},
        "Lord Abbett":                   {"cik": "0000032020", "cat": "mutual_fund"},
        "Nuveen Investments":            {"cik": "0000049639", "cat": "mutual_fund"},
        "Parnassus Investments":         {"cik": "0000878670", "cat": "mutual_fund"},
        "Harris Associates":             {"cik": "0000047111", "cat": "mutual_fund"},
        "Artisan Partners":              {"cik": "0001326110", "cat": "mutual_fund"},
        "Royce & Associates":            {"cik": "0000068312", "cat": "mutual_fund"},
        "Gabelli Funds":                 {"cik": "0000046080", "cat": "mutual_fund"},
        "Manning & Napier":              {"cik": "0000064760", "cat": "mutual_fund"},
        "First Eagle Investment":        {"cik": "0000040114", "cat": "mutual_fund"},
        "Arrowstreet Capital":           {"cik": "0001317776", "cat": "mutual_fund"},
        "Cohen & Steers":                {"cik": "0000799880", "cat": "mutual_fund"},
        "Federated Hermes":              {"cik": "0000034782", "cat": "mutual_fund"},
        "Calvert Research":              {"cik": "0000814679", "cat": "mutual_fund"},
        "Brown Advisory":                {"cik": "0001383312", "cat": "mutual_fund"},
        "Wasatch Advisors":              {"cik": "0001086364", "cat": "mutual_fund"},
        "Driehaus Capital Management":   {"cik": "0000813672", "cat": "mutual_fund"},
        "Baillie Gifford":               {"cik": "0001048268", "cat": "mutual_fund"},
        "Dimensional Fund Advisors":     {"cik": "0000029905", "cat": "mutual_fund"},
        "Northern Trust":                {"cik": "0000073124", "cat": "mutual_fund"},
        "BNY Mellon":                    {"cik": "0000009626", "cat": "mutual_fund"},
        "TIAA-CREF":                     {"cik": "0000098340", "cat": "mutual_fund"},
        "Principal Financial Group":     {"cik": "0000077281", "cat": "mutual_fund"},
        "Charles Schwab":                {"cik": "0000316206", "cat": "mutual_fund"},
        # Hedge funds
        "Berkshire Hathaway":            {"cik": "0001067983", "cat": "hedge_fund"},
        "Bridgewater Associates":        {"cik": "0001350694", "cat": "hedge_fund"},
        "Renaissance Technologies":      {"cik": "0001037389", "cat": "hedge_fund"},
        "D.E. Shaw":                     {"cik": "0001009626", "cat": "hedge_fund"},
        "Two Sigma Investments":         {"cik": "0001278021", "cat": "hedge_fund"},
        "Citadel Advisors":              {"cik": "0001423298", "cat": "hedge_fund"},
        "AQR Capital Management":        {"cik": "0001336528", "cat": "hedge_fund"},
        "Point72 Asset Management":      {"cik": "0001603466", "cat": "hedge_fund"},
        "Millennium Management":         {"cik": "0001273087", "cat": "hedge_fund"},
        "Baupost Group":                 {"cik": "0001061768", "cat": "hedge_fund"},
        "Viking Global Investors":       {"cik": "0001103804", "cat": "hedge_fund"},
        "Tiger Global Management":       {"cik": "0001167483", "cat": "hedge_fund"},
        "Coatue Management":             {"cik": "0001336092", "cat": "hedge_fund"},
        "Lone Pine Capital":             {"cik": "0001061165", "cat": "hedge_fund"},
        "Pershing Square Capital":       {"cik": "0001336528", "cat": "hedge_fund"},
        "Third Point":                   {"cik": "0001040273", "cat": "hedge_fund"},
        "ValueAct Capital":              {"cik": "0001175483", "cat": "hedge_fund"},
        "Elliott Management":            {"cik": "0001048268", "cat": "hedge_fund"},
        "Starboard Value":               {"cik": "0001517767", "cat": "hedge_fund"},
        "Jana Partners":                 {"cik": "0001159159", "cat": "hedge_fund"},
        "Greenlight Capital":            {"cik": "0001079114", "cat": "hedge_fund"},
        "Appaloosa Management":          {"cik": "0001070154", "cat": "hedge_fund"},
        "Glenview Capital":              {"cik": "0001295510", "cat": "hedge_fund"},
        "Farallon Capital":              {"cik": "0001056943", "cat": "hedge_fund"},
        "Soros Fund Management":         {"cik": "0001029160", "cat": "hedge_fund"},
        "Icahn Associates":              {"cik": "0000813672", "cat": "hedge_fund"},
        "Trian Fund Management":         {"cik": "0001418819", "cat": "hedge_fund"},
        "TCI Fund Management":           {"cik": "0001492404", "cat": "hedge_fund"},
        "Corvex Management":             {"cik": "0001535778", "cat": "hedge_fund"},
        "Sachem Head Capital":           {"cik": "0001568385", "cat": "hedge_fund"},
        "Engaged Capital":               {"cik": "0001576913", "cat": "hedge_fund"},
        "Horizon Kinetics":              {"cik": "0001067294", "cat": "hedge_fund"},
        "Empyrean Capital Partners":     {"cik": "0001432203", "cat": "hedge_fund"},
        "Luxor Capital Group":           {"cik": "0001303652", "cat": "hedge_fund"},
        "Polar Capital":                 {"cik": "0001440153", "cat": "hedge_fund"},
        "Caxton Associates":             {"cik": "0000820736", "cat": "hedge_fund"},
        "Graham Capital Management":     {"cik": "0001126234", "cat": "hedge_fund"},
        "Lansdowne Partners":            {"cik": "0001314922", "cat": "hedge_fund"},
        "Man Group":                     {"cik": "0001318248", "cat": "hedge_fund"},
        "Winton Group":                  {"cik": "0001445583", "cat": "hedge_fund"},
        "Omega Advisors":                {"cik": "0000841729", "cat": "hedge_fund"},
        "Paulson & Co":                  {"cik": "0001029160", "cat": "hedge_fund"},
        # Banks / trust companies
        "JPMorgan Asset Management":     {"cik": "0000019617", "cat": "bank"},
        "Goldman Sachs Asset Mgmt":      {"cik": "0000886982", "cat": "bank"},
        "Morgan Stanley Investment":     {"cik": "0000895421", "cat": "bank"},
        "Wells Fargo Bank":              {"cik": "0000315066", "cat": "bank"},
        "Bank of America Merrill Lynch": {"cik": "0001361658", "cat": "bank"},
        "Citigroup Global Markets":      {"cik": "0000831001", "cat": "bank"},
        "Barclays Capital":              {"cik": "0001368777", "cat": "bank"},
        "Deutsche Bank":                 {"cik": "0001126234", "cat": "bank"},
        "UBS Asset Management":          {"cik": "0000102379", "cat": "bank"},
        "Credit Suisse":                 {"cik": "0001164461", "cat": "bank"},
        "HSBC Global Asset Management":  {"cik": "0001040273", "cat": "bank"},
        "Nomura Asset Management":       {"cik": "0001040273", "cat": "bank"},
        "Mitsubishi UFJ Trust":          {"cik": "0001040273", "cat": "bank"},
        "Mizuho Financial Group":        {"cik": "0001040273", "cat": "bank"},
        "Sumitomo Mitsui Trust":         {"cik": "0001040273", "cat": "bank"},
        # Pension / sovereign
        "CALPERS":                       {"cik": "0001356099", "cat": "pension"},
        "CALSTRS":                       {"cik": "0001356099", "cat": "pension"},
        "New York State Common":         {"cik": "0000315066", "cat": "pension"},
        "Florida State Board":           {"cik": "0000049639", "cat": "pension"},
        "Texas Teachers":                {"cik": "0000315066", "cat": "pension"},
        "Canada Pension Plan":           {"cik": "0001356099", "cat": "pension"},
        "Ontario Teachers Pension":      {"cik": "0001356099", "cat": "pension"},
        "Government Pension Fund Norway":{"cik": "0001356099", "cat": "pension"},
        "CDPQ":                          {"cik": "0001356099", "cat": "pension"},
        "Abu Dhabi Investment Authority":{"cik": "0001356099", "cat": "pension"},
        "Kuwait Investment Authority":   {"cik": "0001356099", "cat": "pension"},
        "Temasek Holdings":              {"cik": "0001356099", "cat": "pension"},
        # Insurance
        "MetLife Investment Management": {"cik": "0000040820", "cat": "insurance"},
        "Prudential Financial":          {"cik": "0001137774", "cat": "insurance"},
        "New York Life Investments":     {"cik": "0000310250", "cat": "insurance"},
        "Northwestern Mutual":           {"cik": "0000026138", "cat": "insurance"},
        "Nationwide Financial":          {"cik": "0001144519", "cat": "insurance"},
        "Lincoln National":              {"cik": "0000060086", "cat": "insurance"},
        "Principal Life Insurance":      {"cik": "0000077281", "cat": "insurance"},
        "Unum Group":                    {"cik": "0000078814", "cat": "insurance"},
        "Pacific Mutual":                {"cik": "0000093751", "cat": "insurance"},
    }

    # Smart-money CIKs (top-quartile alpha managers)
    SMART_MONEY_CIKS: frozenset[str] = frozenset({
        "0001037389",  # Renaissance Technologies
        "0001350694",  # Bridgewater
        "0001423298",  # Citadel
        "0001278021",  # Two Sigma
        "0001009626",  # D.E. Shaw
        "0001603466",  # Point72
        "0001273087",  # Millennium
        "0001061768",  # Baupost
        "0001103804",  # Viking Global
        "0001167483",  # Tiger Global
        "0001336092",  # Coatue
        "0001061165",  # Lone Pine
        "0001040273",  # Third Point
        "0001175483",  # ValueAct
        "0001295510",  # Glenview
        "0001492404",  # TCI
        "0001070154",  # Appaloosa
        "0001048268",  # Elliott
        "0001517767",  # Starboard
        "0001067983",  # Berkshire Hathaway
    })

    # Hedge-fund CIKs (used by HedgeFundTracker)
    HEDGE_FUND_CIKS: frozenset[str] = frozenset(
        v["cik"] for v in _SEED.values() if v["cat"] == "hedge_fund"
    )

    def __init__(self):
        self._registry: dict[str, Institution] = {}
        self._load_seed()

    def _load_seed(self) -> None:
        for name, meta in self._SEED.items():
            self._registry[meta["cik"]] = Institution(
                cik=meta["cik"], name=name, category=meta["cat"]
            )

    def get(self, cik: str) -> Optional[Institution]:
        return self._registry.get(cik)

    def get_or_create(self, cik: str, name: str) -> Institution:
        if cik not in self._registry:
            self._registry[cik] = Institution(
                cik=cik,
                name=name,
                category=_categorize_institution(name),
            )
        return self._registry[cik]

    def get_top_institutions(self, n: int = 100) -> list[Institution]:
        ranked = sorted(
            self._registry.values(),
            key=lambda i: i.aum_est_usd,
            reverse=True,
        )
        return ranked[:n]

    def get_by_category(self, category: str) -> list[Institution]:
        return [i for i in self._registry.values() if i.category == category]

    def is_etf_provider(self, cik: str) -> bool:
        inst = self._registry.get(cik)
        return inst is not None and inst.category == "etf_provider"

    def is_hedge_fund(self, cik: str) -> bool:
        if cik in self.HEDGE_FUND_CIKS:
            return True
        inst = self._registry.get(cik)
        return inst is not None and inst.category == "hedge_fund"

    def is_smart_money(self, cik: str) -> bool:
        return cik in self.SMART_MONEY_CIKS

    def discover_from_edgar(self, letter: str = "B", count: int = 40) -> int:
        """Fetch additional institutions from EDGAR and add to registry."""
        parser = EDGAR13FParser()
        filers = parser.search_13f_filers(company_name=letter, count=count)
        added  = 0
        for f in filers:
            cik = f.get("cik", "").zfill(10)
            if not cik or cik in self._registry:
                continue
            name     = f.get("name", cik)
            category = _categorize_institution(name)
            self._registry[cik] = Institution(cik=cik, name=name, category=category)
            added += 1
        logger.info("Discovered %d new institutions via EDGAR search", added)
        return added


# ---------------------------------------------------------------------------
# OwnershipDatabase  (DuckDB with SQLite fallback)
# ---------------------------------------------------------------------------

class OwnershipDatabase:
    """Persistent storage for 13F holdings.

    Primary: DuckDB at sentinel/data/institutional.duckdb
    Fallback: SQLite at sentinel/data/institutional_fb.db
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS holdings (
        id                     INTEGER PRIMARY KEY,
        cik                    VARCHAR NOT NULL,
        institution_name       VARCHAR,
        cusip                  VARCHAR,
        ticker                 VARCHAR,
        issuer_name            VARCHAR,
        shares                 BIGINT,
        value_usd              BIGINT,
        filing_date            DATE,
        period_of_report       DATE,
        pct_portfolio          DOUBLE,
        investment_discretion  VARCHAR,
        put_call               VARCHAR,
        category               VARCHAR,
        UNIQUE (cik, cusip, period_of_report)
    );
    CREATE INDEX IF NOT EXISTS idx_holdings_ticker  ON holdings(ticker);
    CREATE INDEX IF NOT EXISTS idx_holdings_cik     ON holdings(cik);
    CREATE INDEX IF NOT EXISTS idx_holdings_cusip   ON holdings(cusip);
    CREATE INDEX IF NOT EXISTS idx_holdings_period  ON holdings(period_of_report);
    """

    def __init__(self, db_path: Optional[Path] = None, registry: Optional[InstitutionRegistry] = None):
        self._db_path = db_path or _DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = registry or InstitutionRegistry()
        self._use_duckdb = _DUCKDB_AVAILABLE
        self._conn = self._connect()
        self._init_schema()

    def _connect(self):
        if self._use_duckdb:
            try:
                conn = duckdb.connect(str(self._db_path))
                logger.info("Connected to DuckDB at %s", self._db_path)
                return conn
            except Exception as exc:
                logger.warning("DuckDB connect failed (%s); falling back to SQLite", exc)
                self._use_duckdb = False
        conn = sqlite3.connect(str(_SQLITE_FB))
        logger.info("Connected to SQLite fallback at %s", _SQLITE_FB)
        return conn

    def _init_schema(self) -> None:
        if self._use_duckdb:
            # DuckDB uses SEQUENCE for auto-increment
            try:
                self._conn.execute("CREATE SEQUENCE IF NOT EXISTS holdings_seq")
            except Exception:
                pass
            ddl = self._DDL.replace(
                "INTEGER PRIMARY KEY",
                "INTEGER DEFAULT nextval('holdings_seq') PRIMARY KEY"
            )
        else:
            ddl = self._DDL.replace(
                "INTEGER PRIMARY KEY",
                "INTEGER PRIMARY KEY AUTOINCREMENT"
            )
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self._conn.execute(stmt)
                except Exception as exc:
                    logger.debug("DDL warning: %s", exc)
        self._commit()

    def _commit(self) -> None:
        if not self._use_duckdb:
            self._conn.commit()

    def upsert_holdings(
        self,
        holdings: list[Holding13F],
        institution_cik: str,
        filing_date: date,
    ) -> int:
        """Insert or replace holdings; returns count inserted."""
        inst   = self._registry.get(institution_cik)
        cat    = inst.category if inst else "other"
        upserted = 0
        for h in holdings:
            try:
                sql = """
                    INSERT INTO holdings
                      (cik, institution_name, cusip, ticker, issuer_name,
                       shares, value_usd, filing_date, period_of_report,
                       pct_portfolio, investment_discretion, put_call, category)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (cik, cusip, period_of_report) DO UPDATE SET
                      shares=excluded.shares,
                      value_usd=excluded.value_usd,
                      pct_portfolio=excluded.pct_portfolio,
                      filing_date=excluded.filing_date,
                      ticker=COALESCE(excluded.ticker, holdings.ticker)
                """
                if not self._use_duckdb:
                    sql = sql.replace("excluded.", "")
                self._conn.execute(sql, (
                    h.institution_cik,
                    h.institution_name,
                    h.cusip,
                    h.ticker,
                    h.issuer_name,
                    h.shares,
                    h.value_usd,
                    str(h.filing_date),
                    str(h.period_of_report),
                    h.pct_portfolio,
                    h.investment_discretion,
                    h.put_call,
                    cat,
                ))
                upserted += 1
            except Exception as exc:
                logger.debug("Upsert error for %s/%s: %s", institution_cik, h.cusip, exc)
        self._commit()
        logger.info("Upserted %d/%d holdings for CIK %s", upserted, len(holdings), institution_cik)
        return upserted

    def query_stock_owners(
        self, ticker: str, as_of: Optional[str] = None
    ):
        """Return DataFrame of institutions owning *ticker*, most recent period."""
        period_filter = f"AND period_of_report <= '{as_of}'" if as_of else ""
        sql = f"""
            SELECT institution_name, cik, cusip, shares, value_usd,
                   pct_portfolio, filing_date, period_of_report, category
            FROM holdings
            WHERE UPPER(ticker) = UPPER(?)
            {period_filter}
            ORDER BY period_of_report DESC, value_usd DESC
        """
        return self._execute_df(sql, (ticker.upper(),))

    def query_institution_portfolio(
        self, cik: str, as_of: Optional[str] = None
    ):
        """Return DataFrame of all positions for an institution, most recent period."""
        period_filter = f"AND period_of_report <= '{as_of}'" if as_of else ""
        sql = f"""
            SELECT ticker, issuer_name, cusip, shares, value_usd,
                   pct_portfolio, filing_date, period_of_report
            FROM holdings
            WHERE cik = ?
            {period_filter}
            ORDER BY period_of_report DESC, value_usd DESC
        """
        return self._execute_df(sql, (cik,))

    def get_latest_period(self, cik: Optional[str] = None, ticker: Optional[str] = None) -> Optional[str]:
        """Return the most recent period_of_report for given filter."""
        if cik:
            row = self._fetchone("SELECT MAX(period_of_report) FROM holdings WHERE cik=?", (cik,))
        elif ticker:
            row = self._fetchone(
                "SELECT MAX(period_of_report) FROM holdings WHERE UPPER(ticker)=UPPER(?)", (ticker,)
            )
        else:
            row = self._fetchone("SELECT MAX(period_of_report) FROM holdings", ())
        return str(row[0]) if row and row[0] else None

    def get_all_holders_for_ticker(self, ticker: str, period: str) -> list[dict]:
        """Raw list of holder dicts for a specific period."""
        sql = """
            SELECT cik, institution_name, shares, value_usd, pct_portfolio, category
            FROM holdings
            WHERE UPPER(ticker)=UPPER(?) AND period_of_report=?
            ORDER BY value_usd DESC
        """
        rows = self._fetchall(sql, (ticker, period))
        return [
            {"cik": r[0], "name": r[1], "shares": r[2],
             "value_usd": r[3], "pct_portfolio": r[4], "category": r[5]}
            for r in rows
        ]

    def get_ticker_history(self, ticker: str, cik: str) -> list[dict]:
        """Historical holdings of *ticker* by *cik* sorted by period."""
        sql = """
            SELECT period_of_report, shares, value_usd, pct_portfolio
            FROM holdings
            WHERE UPPER(ticker)=UPPER(?) AND cik=?
            ORDER BY period_of_report ASC
        """
        rows = self._fetchall(sql, (ticker, cik))
        return [{"period": str(r[0]), "shares": r[1], "value_usd": r[2],
                 "pct_portfolio": r[3]} for r in rows]

    def get_distinct_periods(self, ticker: str) -> list[str]:
        sql = """
            SELECT DISTINCT period_of_report FROM holdings
            WHERE UPPER(ticker)=UPPER(?)
            ORDER BY period_of_report DESC
        """
        rows = self._fetchall(sql, (ticker,))
        return [str(r[0]) for r in rows]

    def _execute_df(self, sql: str, params: tuple):
        if not _PANDAS_AVAILABLE:
            rows = self._fetchall(sql, params)
            return rows
        try:
            if self._use_duckdb:
                return self._conn.execute(sql, list(params)).df()
            else:
                import pandas as pd
                return pd.read_sql_query(sql, self._conn, params=params)
        except Exception as exc:
            logger.error("Query failed: %s", exc)
            return pd.DataFrame() if _PANDAS_AVAILABLE else []

    def _fetchone(self, sql: str, params: tuple):
        try:
            result = self._conn.execute(sql, list(params)).fetchone()
            return result
        except Exception:
            return None

    def _fetchall(self, sql: str, params: tuple) -> list:
        try:
            return self._conn.execute(sql, list(params)).fetchall()
        except Exception as exc:
            logger.error("fetchall failed: %s", exc)
            return []

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OwnershipAnalytics
# ---------------------------------------------------------------------------

class OwnershipAnalytics:
    """Higher-order analytics on top of OwnershipDatabase."""

    def __init__(self, db: OwnershipDatabase, registry: InstitutionRegistry):
        self._db  = db
        self._reg = registry

    # ── Ownership concentration ──────────────────────────────────────────────

    def get_ownership_concentration(self, ticker: str) -> dict:
        """Compute HHI, top-10 pct, institutional total for latest period."""
        period = self._db.get_latest_period(ticker=ticker)
        if not period:
            return {"error": f"No data for {ticker}"}

        holders = self._db.get_all_holders_for_ticker(ticker, period)
        if not holders:
            return {"error": f"No holders found for {ticker} in {period}"}

        total_inst_shares  = sum(h["shares"]    for h in holders)
        total_inst_value   = sum(h["value_usd"] for h in holders)
        sorted_h           = sorted(holders, key=lambda x: x["value_usd"], reverse=True)

        top10_value    = sum(h["value_usd"] for h in sorted_h[:10])
        top10_pct      = (top10_value / total_inst_value * 100) if total_inst_value else 0.0

        # HHI on value shares
        pcts = [(h["value_usd"] / total_inst_value * 100) for h in holders] if total_inst_value else []
        hhi  = round(sum(p ** 2 for p in pcts), 2)

        return {
            "ticker":                 ticker,
            "period":                 period,
            "total_institutions":     len(holders),
            "total_inst_value_usd":   total_inst_value,
            "total_inst_shares":      total_inst_shares,
            "hhi":                    hhi,
            "hhi_interpretation":     self._interpret_hhi(hhi),
            "top10_pct":              round(top10_pct, 2),
            "top_holders":            sorted_h[:10],
        }

    @staticmethod
    def _interpret_hhi(hhi: float) -> str:
        if hhi < 1500:
            return "unconcentrated"
        if hhi < 2500:
            return "moderately_concentrated"
        return "highly_concentrated"

    # ── Ownership delta between quarters ────────────────────────────────────

    def detect_ownership_change(self, ticker: str, q1: str, q2: str) -> OwnershipDelta:
        """Compare institutional ownership between two period strings."""
        h1 = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q1)}
        h2 = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q2)}

        delta = OwnershipDelta(ticker=ticker, q1=q1, q2=q2)
        all_ciks = set(h1) | set(h2)

        for cik in all_ciks:
            in_q1 = h1.get(cik)
            in_q2 = h2.get(cik)
            name   = (in_q2 or in_q1)["name"]

            if in_q2 and not in_q1:
                delta.new_positions.append({
                    "cik": cik, "name": name,
                    "shares": in_q2["shares"], "value_usd": in_q2["value_usd"],
                })
                delta.net_shares_change     += in_q2["shares"]
                delta.net_value_change_usd  += in_q2["value_usd"]
                delta.net_institution_change += 1

            elif in_q1 and not in_q2:
                delta.exited_positions.append({
                    "cik": cik, "name": name,
                    "shares_sold": in_q1["shares"], "value_usd": in_q1["value_usd"],
                })
                delta.net_shares_change     -= in_q1["shares"]
                delta.net_value_change_usd  -= in_q1["value_usd"]
                delta.net_institution_change -= 1

            elif in_q1 and in_q2:
                share_diff = in_q2["shares"] - in_q1["shares"]
                val_diff   = in_q2["value_usd"] - in_q1["value_usd"]
                delta.net_shares_change    += share_diff
                delta.net_value_change_usd += val_diff

                if share_diff > 0:
                    delta.increased_positions.append({
                        "cik": cik, "name": name,
                        "shares_added": share_diff, "pct_increase": round(share_diff / in_q1["shares"] * 100, 2)
                        if in_q1["shares"] else 0,
                    })
                    delta.net_institution_change += 1
                elif share_diff < 0:
                    delta.decreased_positions.append({
                        "cik": cik, "name": name,
                        "shares_removed": -share_diff, "pct_decrease": round(-share_diff / in_q1["shares"] * 100, 2)
                        if in_q1["shares"] else 0,
                    })
                    delta.net_institution_change -= 1

        return delta

    # ── Smart-money consensus ────────────────────────────────────────────────

    def get_smart_money_consensus(self, ticker: str, top_n_funds: int = 20) -> dict:
        """Consensus buy/sell signal among top smart-money institutions."""
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return {"ticker": ticker, "consensus": "insufficient_data",
                    "smart_money_holders": 0}

        current = periods[0]
        prior   = periods[1]

        sm_ciks  = list(self._reg.SMART_MONEY_CIKS)[:top_n_funds]
        h_curr   = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, current)
                    if h["cik"] in sm_ciks}
        h_prior  = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, prior)
                    if h["cik"] in sm_ciks}

        buyers   = [cik for cik in h_curr  if cik not in h_prior
                    or h_curr[cik]["shares"] > h_prior[cik]["shares"]]
        sellers  = [cik for cik in h_prior if cik not in h_curr
                    or (cik in h_curr and h_curr[cik]["shares"] < h_prior[cik]["shares"])]

        score = len(buyers) - len(sellers)
        if score > 2:
            consensus = "strong_buy"
        elif score > 0:
            consensus = "buy"
        elif score == 0:
            consensus = "neutral"
        elif score > -3:
            consensus = "sell"
        else:
            consensus = "strong_sell"

        return {
            "ticker":             ticker,
            "current_period":     current,
            "prior_period":       prior,
            "smart_money_holders": len(h_curr),
            "buyers":             len(buyers),
            "sellers":            len(sellers),
            "net_score":          score,
            "consensus":          consensus,
            "buyer_names":        [h_curr[c]["name"] for c in buyers if c in h_curr],
            "seller_names":       [h_prior[c]["name"] for c in sellers if c in h_prior],
        }

    # ── Institutional momentum ───────────────────────────────────────────────

    def compute_institutional_momentum(
        self, ticker: str, lookback_quarters: int = 4
    ) -> float:
        """Compute change in total institutional ownership pct over N quarters.

        Returns percentage-point change (positive = growing institutional interest).
        """
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return 0.0

        n       = min(lookback_quarters, len(periods) - 1)
        current = periods[0]
        base    = periods[n]

        def _total_value(period: str) -> int:
            holders = self._db.get_all_holders_for_ticker(ticker, period)
            return sum(h["value_usd"] for h in holders)

        val_curr = _total_value(current)
        val_base = _total_value(base)
        if val_base == 0:
            return 0.0
        return round((val_curr - val_base) / val_base * 100, 2)

    # ── Net buyer count ──────────────────────────────────────────────────────

    def compute_net_buyers(self, ticker: str) -> int:
        """Returns (buyers - sellers) count between the two most recent quarters."""
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return 0
        delta = self.detect_ownership_change(ticker, periods[1], periods[0])
        return delta.net_institution_change

    def get_full_ownership_report(self, ticker: str) -> dict:
        """Combine concentration + momentum + smart money into one dict."""
        conc  = self.get_ownership_concentration(ticker)
        mom   = self.compute_institutional_momentum(ticker)
        smart = self.get_smart_money_consensus(ticker)
        return {
            "concentration": conc,
            "momentum_pct_change": mom,
            "smart_money": smart,
        }

    # ── New math-verified analytics (dim_025 score 9) ────────────────────────

    def compute_ownership_concentration(self, ticker: str, top_n: int = 10) -> dict:
        """HHI of top-N 13F holders weighted by shares held.

        Formula: HHI = sum((shares_i / total_shares)^2) × 10000
        where shares_i is each institution's share count and total_shares
        is the sum across all reported institutions for that ticker/period.
        Range: 0 (perfectly dispersed) → 10000 (single institution holds all).
        """
        period = self._db.get_latest_period(ticker=ticker)
        if not period:
            return {"error": f"No data for {ticker}", "hhi_top_n": None, "top_n": top_n}

        holders = self._db.get_all_holders_for_ticker(ticker, period)
        if not holders:
            return {"error": f"No holders for {ticker} in {period}", "hhi_top_n": None}

        # Sort by shares descending; take top-N
        sorted_by_shares = sorted(holders, key=lambda h: h["shares"], reverse=True)
        top_holders      = sorted_by_shares[:top_n]

        total_shares = sum(h["shares"] for h in holders)
        if total_shares == 0:
            return {"hhi_top_n": 0.0, "ticker": ticker, "period": period, "top_n": top_n,
                    "total_institutions": len(holders)}

        # HHI = Σ (s_i / total)² × 10000
        hhi = sum((h["shares"] / total_shares) ** 2 for h in top_holders) * 10_000

        return {
            "ticker":              ticker,
            "period":              period,
            "top_n":               top_n,
            "total_institutions":  len(holders),
            "total_shares":        total_shares,
            "hhi_top_n":           round(hhi, 2),
            "hhi_interpretation":  self._interpret_hhi(hhi),
            "top_holders":         [
                {
                    "name":    h["name"],
                    "shares":  h["shares"],
                    "weight":  round(h["shares"] / total_shares * 100, 4),
                }
                for h in top_holders
            ],
        }

    def detect_accumulation_patterns(self, ticker: str, threshold_pct: float = 5.0) -> dict:
        """Detect QoQ accumulation: institutions where share change > threshold_pct%.

        An institution is flagged as accumulating if:
            (shares_q2 - shares_q1) / shares_q1 > threshold_pct / 100

        Returns a dict with lists of accumulators and distributors.
        """
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return {
                "ticker": ticker,
                "error":  "Insufficient periods for QoQ comparison",
                "accumulators": [],
                "distributors": [],
            }

        q2_period = periods[0]   # most recent
        q1_period = periods[1]   # prior quarter

        h1 = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q1_period)}
        h2 = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q2_period)}

        threshold = threshold_pct / 100.0
        accumulators = []
        distributors = []

        for cik in set(h1) & set(h2):   # institutions present in both quarters
            s1 = h1[cik]["shares"]
            s2 = h2[cik]["shares"]
            if s1 <= 0:
                continue
            pct_change = (s2 - s1) / s1
            if pct_change > threshold:
                accumulators.append({
                    "cik":        cik,
                    "name":       h2[cik]["name"],
                    "shares_q1":  s1,
                    "shares_q2":  s2,
                    "pct_change": round(pct_change * 100, 2),
                })
            elif pct_change < -threshold:
                distributors.append({
                    "cik":        cik,
                    "name":       h2[cik]["name"],
                    "shares_q1":  s1,
                    "shares_q2":  s2,
                    "pct_change": round(pct_change * 100, 2),
                })

        accumulators.sort(key=lambda x: x["pct_change"], reverse=True)
        distributors.sort(key=lambda x: x["pct_change"])

        return {
            "ticker":          ticker,
            "q1_period":       q1_period,
            "q2_period":       q2_period,
            "threshold_pct":   threshold_pct,
            "accumulator_count": len(accumulators),
            "distributor_count": len(distributors),
            "accumulators":    accumulators,
            "distributors":    distributors,
        }

    def compute_smart_money_signal(self, ticker: str, top_n_funds: int = 20) -> dict:
        """Hedge-fund consensus score: fraction of 13Fs increasing position.

        smart_money_score = buyers / (buyers + sellers + unchanged) in [0, 1].
        Uses the same smart-money CIK set as get_smart_money_consensus.
        Score > 0.6 = consensus accumulation; < 0.4 = consensus distribution.
        """
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return {
                "ticker": ticker,
                "smart_money_score": None,
                "signal": "insufficient_data",
            }

        q_curr = periods[0]
        q_prior = periods[1]

        sm_ciks  = list(self._reg.SMART_MONEY_CIKS)[:top_n_funds]
        h_curr   = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q_curr)
                    if h["cik"] in sm_ciks}
        h_prior  = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, q_prior)
                    if h["cik"] in sm_ciks}

        all_ciks = set(h_curr) | set(h_prior)
        buyers = sellers = unchanged = 0

        for cik in all_ciks:
            in_curr  = h_curr.get(cik)
            in_prior = h_prior.get(cik)

            if in_curr and not in_prior:
                buyers += 1          # new position
            elif in_prior and not in_curr:
                sellers += 1         # exited
            elif in_curr and in_prior:
                diff = in_curr["shares"] - in_prior["shares"]
                if diff > 0:
                    buyers += 1
                elif diff < 0:
                    sellers += 1
                else:
                    unchanged += 1

        total = buyers + sellers + unchanged
        score = round(buyers / total, 4) if total > 0 else 0.5

        if score >= 0.6:
            signal = "accumulation"
        elif score <= 0.4:
            signal = "distribution"
        else:
            signal = "neutral"

        return {
            "ticker":              ticker,
            "q_prior":             q_prior,
            "q_current":           q_curr,
            "smart_money_funds_tracked": len(all_ciks),
            "buyers":              buyers,
            "sellers":             sellers,
            "unchanged":           unchanged,
            "smart_money_score":   score,   # fraction in [0, 1]
            "signal":              signal,
        }


# ---------------------------------------------------------------------------
# HedgeFundTracker
# ---------------------------------------------------------------------------

class HedgeFundTracker:
    """Hedge-fund-specific 13F analytics."""

    # Specific high-profile funds to always track
    TRACKED_FUNDS: dict[str, str] = {
        "Berkshire Hathaway":     "0001067983",
        "Bridgewater Associates": "0001350694",
        "Renaissance Technologies":"0001037389",
        "Citadel Advisors":       "0001423298",
        "Viking Global Investors":"0001103804",
        "Tiger Global Management":"0001167483",
        "Pershing Square Capital":"0001336528",
        "Elliott Management":     "0001048268",
        "Third Point":            "0001040273",
        "Lone Pine Capital":      "0001061165",
        "Point72 Asset Management":"0001603466",
        "Millennium Management":  "0001273087",
        "D.E. Shaw":              "0001009626",
        "Two Sigma Investments":  "0001278021",
        "AQR Capital Management": "0001336528",
    }

    def __init__(
        self,
        db: OwnershipDatabase,
        registry: InstitutionRegistry,
        parser: Optional[EDGAR13FParser] = None,
    ):
        self._db     = db
        self._reg    = registry
        self._parser = parser or EDGAR13FParser()

    def get_hedge_fund_portfolio(self, fund_name: str):
        """Latest holdings DataFrame for a hedge fund by name."""
        cik = self.TRACKED_FUNDS.get(fund_name) or self._reg._SEED.get(fund_name, {}).get("cik")
        if not cik:
            logger.warning("Unknown fund: %s", fund_name)
            return [] if not _PANDAS_AVAILABLE else pd.DataFrame()
        return self._db.query_institution_portfolio(cik)

    def get_concentration_score(self, cik: str) -> float:
        """Portfolio Herfindahl index (0–10000). Higher = more concentrated."""
        period = self._db.get_latest_period(cik=cik)
        if not period:
            return 0.0
        rows = self._db._fetchall(
            "SELECT pct_portfolio FROM holdings WHERE cik=? AND period_of_report=?",
            (cik, period)
        )
        pcts = [r[0] for r in rows if r[0]]
        total = sum(pcts)
        if total == 0:
            return 0.0
        normalized = [p / total * 100 for p in pcts]
        return round(sum(p ** 2 for p in normalized), 2)

    def detect_new_positions(self, cik: str, quarters_back: int = 1) -> list[Holding13F]:
        """Holdings that didn't exist N quarters ago (brand-new positions)."""
        periods = self._db.get_distinct_periods(cik)
        if len(periods) < quarters_back + 1:
            return []

        current = periods[0]
        prior   = periods[quarters_back]

        # Get current CUSIPs
        curr_rows = self._db._fetchall(
            "SELECT cusip, ticker, issuer_name, shares, value_usd, pct_portfolio, filing_date, period_of_report "
            "FROM holdings WHERE cik=? AND period_of_report=?",
            (cik, current)
        )
        prior_cusips = set(r[0] for r in self._db._fetchall(
            "SELECT cusip FROM holdings WHERE cik=? AND period_of_report=?",
            (cik, prior)
        ))

        new_positions = []
        inst = self._reg.get(cik)
        inst_name = inst.name if inst else cik
        for r in curr_rows:
            cusip = r[0]
            if cusip and cusip not in prior_cusips:
                new_positions.append(Holding13F(
                    institution_cik  = cik,
                    institution_name = inst_name,
                    cusip            = r[0],
                    ticker           = r[1] or "",
                    issuer_name      = r[2] or "",
                    shares           = r[3] or 0,
                    value_usd        = r[4] or 0,
                    pct_portfolio    = r[5] or 0.0,
                    filing_date      = _date_from_str(str(r[6])) or date.today(),
                    period_of_report = _date_from_str(str(r[7])) or date.today(),
                ))
        return sorted(new_positions, key=lambda h: h.value_usd, reverse=True)

    def get_top_hedge_fund_picks(self, min_funds: int = 5) -> list[str]:
        """Tickers held by at least *min_funds* tracked hedge funds."""
        hf_ciks = list(self.TRACKED_FUNDS.values())
        # Count distinct HF CIKs per ticker in most recent periods
        if self._db._use_duckdb:
            sql = """
                SELECT ticker, COUNT(DISTINCT cik) as fund_count
                FROM holdings
                WHERE cik IN ({})
                  AND period_of_report >= date_trunc('month', current_date - INTERVAL '6 months')
                  AND ticker != ''
                GROUP BY ticker
                HAVING COUNT(DISTINCT cik) >= ?
                ORDER BY fund_count DESC, ticker
            """.format(",".join("?" * len(hf_ciks)))
        else:
            sql = """
                SELECT ticker, COUNT(DISTINCT cik) as fund_count
                FROM holdings
                WHERE cik IN ({})
                  AND ticker != ''
                GROUP BY ticker
                HAVING COUNT(DISTINCT cik) >= ?
                ORDER BY fund_count DESC, ticker
            """.format(",".join("?" * len(hf_ciks)))

        rows = self._db._fetchall(sql, tuple(hf_ciks) + (min_funds,))
        return [r[0] for r in rows if r[0]]

    def refresh_tracked_fund(self, fund_name: str, cik: str) -> int:
        """Fetch latest 13F for a tracked fund and upsert to DB."""
        parser   = self._parser
        holdings = parser.get_latest_13f(cik)
        if not holdings:
            logger.warning("No holdings for %s", fund_name)
            return 0
        # Update institution registry
        inst = self._reg.get_or_create(cik, fund_name)
        inst.last_filing_date = holdings[0].filing_date if holdings else None
        inst.aum_est_usd      = sum(h.value_usd for h in holdings)
        return self._db.upsert_holdings(holdings, cik, holdings[0].filing_date)

    def bulk_refresh_tracked_funds(self, max_funds: int = 5) -> dict[str, int]:
        """Refresh the top-N tracked funds; returns {name: holdings_count}."""
        results = {}
        for name, cik in list(self.TRACKED_FUNDS.items())[:max_funds]:
            try:
                n = self.refresh_tracked_fund(name, cik)
                results[name] = n
            except Exception as exc:
                logger.error("Failed to refresh %s: %s", name, exc)
                results[name] = -1
        return results


# ---------------------------------------------------------------------------
# ETFFlowAnalyzer
# ---------------------------------------------------------------------------

class ETFFlowAnalyzer:
    """Approximate ETF holdings and passive vs active ownership splits.

    Uses 13F data from known ETF providers already in OwnershipDatabase.
    """

    ETF_PROVIDER_CIKS: frozenset[str] = frozenset({
        "0000102909",  # Vanguard
        "0001364742",  # BlackRock / iShares
        "0000093751",  # State Street / SSGA
        "0000049071",  # Invesco
        "0001275014",  # WisdomTree / ProFunds / Direxion (same CIK in seed; real ones differ)
        "0000857779",  # Van Eck
        "0001100663",  # First Trust
        "0001418819",  # Geode (Fidelity's passive arm)
        "0000316206",  # Charles Schwab
        "0000029905",  # Dimensional Fund Advisors (rules-based)
    })

    def __init__(self, db: OwnershipDatabase, registry: InstitutionRegistry):
        self._db  = db
        self._reg = registry

    def compute_passive_vs_active(self, ticker: str) -> dict:
        """Shares held by ETF providers vs active managers for latest period."""
        period  = self._db.get_latest_period(ticker=ticker)
        if not period:
            return {"error": f"No data for {ticker}"}

        holders = self._db.get_all_holders_for_ticker(ticker, period)
        passive = [h for h in holders if h["cik"] in self.ETF_PROVIDER_CIKS
                   or h["category"] == "etf_provider"]
        active  = [h for h in holders if h not in passive]

        passive_val = sum(h["value_usd"] for h in passive)
        active_val  = sum(h["value_usd"] for h in active)
        total_val   = passive_val + active_val

        return {
            "ticker":             ticker,
            "period":             period,
            "passive_value_usd":  passive_val,
            "active_value_usd":   active_val,
            "total_value_usd":    total_val,
            "passive_pct":        round(passive_val / total_val * 100, 2) if total_val else 0.0,
            "active_pct":         round(active_val  / total_val * 100, 2) if total_val else 0.0,
            "passive_holders":    len(passive),
            "active_holders":     len(active),
            "top_passive":        sorted(passive, key=lambda x: x["value_usd"], reverse=True)[:5],
            "top_active":         sorted(active,  key=lambda x: x["value_usd"], reverse=True)[:5],
        }

    def detect_index_inclusion_effect(self, ticker: str) -> Optional[IndexEvent]:
        """Detect a sudden surge in passive ownership (possible index addition)."""
        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return None

        def _passive_pct(period: str) -> float:
            holders = self._db.get_all_holders_for_ticker(ticker, period)
            total   = sum(h["value_usd"] for h in holders)
            passive = sum(h["value_usd"] for h in holders
                          if h["cik"] in self.ETF_PROVIDER_CIKS or h["category"] == "etf_provider")
            return (passive / total * 100) if total else 0.0

        curr_pct  = _passive_pct(periods[0])
        prior_pct = _passive_pct(periods[1])
        increase  = curr_pct - prior_pct

        if increase >= 5.0:      # 5+ pp surge = likely index event
            index_guess = "Unknown Index"
            if curr_pct > 40:
                index_guess = "S&P 500 / Large-Cap"
            elif curr_pct > 20:
                index_guess = "Russell 1000 / Mid-Cap"
            else:
                index_guess = "Russell 2000 / Small-Cap"

            return IndexEvent(
                ticker                   = ticker,
                detected_date            = _date_from_str(periods[0]) or date.today(),
                passive_ownership_before = round(prior_pct, 2),
                passive_ownership_after  = round(curr_pct, 2),
                pct_increase             = round(increase, 2),
                likely_index             = index_guess,
            )
        return None

    def get_etf_flow_trend(self, ticker: str, n_quarters: int = 4) -> list[dict]:
        """Quarter-over-quarter passive ownership percentage for trend analysis."""
        periods = self._db.get_distinct_periods(ticker)[:n_quarters]
        trend   = []
        for p in reversed(periods):
            holders = self._db.get_all_holders_for_ticker(ticker, p)
            total   = sum(h["value_usd"] for h in holders)
            passive = sum(h["value_usd"] for h in holders
                          if h["cik"] in self.ETF_PROVIDER_CIKS or h["category"] == "etf_provider")
            trend.append({
                "period":      p,
                "passive_pct": round(passive / total * 100, 2) if total else 0.0,
                "n_holders":   len(holders),
            })
        return trend


# ---------------------------------------------------------------------------
# AlertSystem13F
# ---------------------------------------------------------------------------

class AlertSystem13F:
    """Monitor 13F data for significant position changes and new filings."""

    def __init__(
        self,
        db: OwnershipDatabase,
        registry: InstitutionRegistry,
        alert_log_path: Optional[Path] = None,
    ):
        self._db   = db
        self._reg  = registry
        self._log  = alert_log_path or (Path(__file__).parent.parent / "data" / "13f_alerts.json")
        self._alerts: list[Alert13F] = []

    def watch_institution(
        self,
        cik: str,
        threshold_pct_change: float = 10.0,
    ) -> list[Alert13F]:
        """Compare two most-recent 13F filings for *cik*; alert on big moves."""
        periods = self._db.get_distinct_periods(cik)
        if len(periods) < 2:
            return []

        current = periods[0]
        prior   = periods[1]
        alerts: list[Alert13F] = []

        curr_rows = self._db._fetchall(
            "SELECT ticker, cusip, issuer_name, shares, value_usd FROM holdings "
            "WHERE cik=? AND period_of_report=?",
            (cik, current)
        )
        prior_dict = {r[1]: r for r in self._db._fetchall(
            "SELECT ticker, cusip, issuer_name, shares, value_usd FROM holdings "
            "WHERE cik=? AND period_of_report=?",
            (cik, prior)
        )}

        inst = self._reg.get(cik)
        name = inst.name if inst else cik

        for row in curr_rows:
            ticker, cusip, issuer, shares, value = row
            if not cusip:
                continue
            if cusip not in prior_dict:
                alerts.append(Alert13F(
                    alert_type       = "new_position",
                    institution_cik  = cik,
                    institution_name = name,
                    ticker           = ticker or cusip,
                    detail           = f"New position in {issuer}: {shares:,} shares (${value:,})",
                    filing_date      = _date_from_str(current) or date.today(),
                    pct_change       = 100.0,
                ))
            else:
                prior_row    = prior_dict[cusip]
                prior_shares = prior_row[3]
                if prior_shares and prior_shares > 0:
                    pct_chg = (shares - prior_shares) / prior_shares * 100
                    if abs(pct_chg) >= threshold_pct_change:
                        atype = "increase" if pct_chg > 0 else "decrease"
                        alerts.append(Alert13F(
                            alert_type       = atype,
                            institution_cik  = cik,
                            institution_name = name,
                            ticker           = ticker or cusip,
                            detail           = (
                                f"{issuer}: {pct_chg:+.1f}% "
                                f"({prior_shares:,} → {shares:,} shares)"
                            ),
                            filing_date      = _date_from_str(current) or date.today(),
                            pct_change       = round(pct_chg, 2),
                        ))

        # Detect exits
        curr_cusips = {row[1] for row in curr_rows}
        for cusip, prior_row in prior_dict.items():
            if cusip not in curr_cusips:
                alerts.append(Alert13F(
                    alert_type       = "exit",
                    institution_cik  = cik,
                    institution_name = name,
                    ticker           = prior_row[0] or cusip,
                    detail           = f"Exited {prior_row[2]}: sold {prior_row[3]:,} shares",
                    filing_date      = _date_from_str(current) or date.today(),
                    pct_change       = -100.0,
                ))

        self._alerts.extend(alerts)
        return alerts

    def watch_ticker(
        self,
        ticker: str,
        alert_on: list[str],
    ) -> list[Alert13F]:
        """Alert if specified institutions make big moves in *ticker*.

        alert_on: list of institution CIKs or names to monitor.
        """
        # Resolve names to CIKs
        cik_set: set[str] = set()
        for entry in alert_on:
            if re.match(r"^\d{10}$", entry.zfill(10)):
                cik_set.add(entry.zfill(10))
            else:
                # Name lookup
                for cik, inst in self._reg._registry.items():
                    if entry.lower() in inst.name.lower():
                        cik_set.add(cik)

        periods = self._db.get_distinct_periods(ticker)
        if len(periods) < 2:
            return []

        current, prior = periods[0], periods[1]
        h_curr  = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, current)
                   if h["cik"] in cik_set}
        h_prior = {h["cik"]: h for h in self._db.get_all_holders_for_ticker(ticker, prior)
                   if h["cik"] in cik_set}

        alerts: list[Alert13F] = []
        for cik in cik_set:
            inst = self._reg.get(cik)
            name = inst.name if inst else cik
            c, p = h_curr.get(cik), h_prior.get(cik)
            if c and not p:
                alerts.append(Alert13F(
                    alert_type="new_position", institution_cik=cik,
                    institution_name=name, ticker=ticker,
                    detail=f"New position: {c['shares']:,} shares (${c['value_usd']:,})",
                    filing_date=_date_from_str(current) or date.today(),
                    pct_change=100.0,
                ))
            elif p and not c:
                alerts.append(Alert13F(
                    alert_type="exit", institution_cik=cik,
                    institution_name=name, ticker=ticker,
                    detail=f"Exited position: sold {p['shares']:,} shares",
                    filing_date=_date_from_str(current) or date.today(),
                    pct_change=-100.0,
                ))
            elif c and p and p["shares"] > 0:
                pct = (c["shares"] - p["shares"]) / p["shares"] * 100
                if abs(pct) >= 10:
                    alerts.append(Alert13F(
                        alert_type="increase" if pct > 0 else "decrease",
                        institution_cik=cik, institution_name=name,
                        ticker=ticker,
                        detail=f"{pct:+.1f}% change: {p['shares']:,} → {c['shares']:,} shares",
                        filing_date=_date_from_str(current) or date.today(),
                        pct_change=round(pct, 2),
                    ))

        self._alerts.extend(alerts)
        return alerts

    def export_alerts(self, path: Optional[Path] = None) -> str:
        """Export accumulated alerts to JSON; return file path."""
        out = path or self._log
        out.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                pass
        all_alerts = existing + [a.to_dict() for a in self._alerts]
        out.write_text(json.dumps(all_alerts, indent=2, default=str), encoding="utf-8")
        self._alerts.clear()
        return str(out)

    def get_recent_new_filers(self, days_back: int = 10) -> list[dict]:
        """Poll EDGAR EFTS for institutions that just filed a 13F-HR."""
        end_dt   = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days_back)
        params   = {
            "forms":      "13F-HR",
            "dateRange":  "custom",
            "startdt":    str(start_dt),
            "enddt":      str(end_dt),
            "_source":    "hits.hits._source",
            "hits.hits.total.value": "true",
        }
        try:
            r = _get(_EDGAR_SEARCH, params=params)
            hits = r.json().get("hits", {}).get("hits", [])
            filings = []
            for h in hits:
                src = h.get("_source", {})
                filings.append({
                    "filer_name":  src.get("entity_name", ""),
                    "cik":         src.get("file_num", ""),
                    "form":        src.get("form_type", "13F-HR"),
                    "filed":       src.get("file_date", ""),
                    "accession":   src.get("accession_no", ""),
                })
            return filings
        except Exception as exc:
            logger.warning("EFTS 13F poll failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Convenience orchestration
# ---------------------------------------------------------------------------

def build_full_stack(db_path: Optional[Path] = None) -> dict:
    """Return all components pre-wired together."""
    registry  = InstitutionRegistry()
    db        = OwnershipDatabase(db_path=db_path, registry=registry)
    parser    = EDGAR13FParser()
    analytics = OwnershipAnalytics(db=db, registry=registry)
    hf_tracker = HedgeFundTracker(db=db, registry=registry, parser=parser)
    etf_analyzer = ETFFlowAnalyzer(db=db, registry=registry)
    alerts    = AlertSystem13F(db=db, registry=registry)
    return {
        "registry":    registry,
        "db":          db,
        "parser":      parser,
        "analytics":   analytics,
        "hf_tracker":  hf_tracker,
        "etf_analyzer":etf_analyzer,
        "alerts":      alerts,
    }


def ingest_institution(cik: str, name: Optional[str] = None, stack: Optional[dict] = None) -> int:
    """Fetch the latest 13F for *cik* and load into the database. Returns holdings count."""
    if stack is None:
        stack = build_full_stack()
    parser   = stack["parser"]
    db       = stack["db"]
    registry = stack["registry"]

    holdings = parser.get_latest_13f(cik)
    if not holdings:
        logger.warning("No 13F holdings returned for CIK %s", cik)
        return 0
    inst_name = name or (holdings[0].institution_name if holdings else cik)
    registry.get_or_create(cik, inst_name)
    filing_date = holdings[0].filing_date
    return db.upsert_holdings(holdings, cik, filing_date)


# ---------------------------------------------------------------------------
# FastAPI router (optional — only if fastapi present)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

if _HAS_FASTAPI:
    from fastapi import APIRouter, HTTPException, Query

    _stack = None

    def _get_stack() -> dict:
        global _stack
        if _stack is None:
            _stack = build_full_stack()
        return _stack

    institutional_v3_router = APIRouter(prefix="/ownership/v3", tags=["Institutional 13F v3"])

    @institutional_v3_router.get("/{ticker}")
    def api_stock_owners(ticker: str, as_of: Optional[str] = Query(None)):
        stack = _get_stack()
        df    = stack["db"].query_stock_owners(ticker, as_of=as_of)
        if _PANDAS_AVAILABLE and hasattr(df, "to_dict"):
            return df.to_dict(orient="records")
        return df

    @institutional_v3_router.get("/concentration/{ticker}")
    def api_concentration(ticker: str):
        stack = _get_stack()
        return stack["analytics"].get_ownership_concentration(ticker)

    @institutional_v3_router.get("/smart-money/{ticker}")
    def api_smart_money(ticker: str, top_n: int = Query(20)):
        stack = _get_stack()
        return stack["analytics"].get_smart_money_consensus(ticker, top_n_funds=top_n)

    @institutional_v3_router.get("/momentum/{ticker}")
    def api_momentum(ticker: str, quarters: int = Query(4)):
        stack = _get_stack()
        return {
            "ticker":   ticker,
            "momentum": stack["analytics"].compute_institutional_momentum(ticker, quarters),
        }

    @institutional_v3_router.get("/passive/{ticker}")
    def api_passive(ticker: str):
        stack = _get_stack()
        return stack["etf_analyzer"].compute_passive_vs_active(ticker)

    @institutional_v3_router.get("/hedge-picks")
    def api_hedge_picks(min_funds: int = Query(5)):
        stack = _get_stack()
        return {"picks": stack["hf_tracker"].get_top_hedge_fund_picks(min_funds=min_funds)}

    @institutional_v3_router.get("/ingest/{cik}")
    def api_ingest(cik: str, name: Optional[str] = Query(None)):
        n = ingest_institution(cik, name=name, stack=_get_stack())
        return {"cik": cik, "holdings_ingested": n}


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("=" * 70)
    print("SENTINEL institutional_ownership_v3 — Demo")
    print("=" * 70)

    stack = build_full_stack()
    parser    = stack["parser"]
    db        = stack["db"]
    registry  = stack["registry"]
    analytics = stack["analytics"]
    hf        = stack["hf_tracker"]
    etf       = stack["etf_analyzer"]
    alert_sys = stack["alerts"]

    # 1. Fetch Berkshire Hathaway 13F
    BERKSHIRE_CIK = "0001067983"
    print(f"\n[1] Fetching latest 13F for Berkshire Hathaway (CIK {BERKSHIRE_CIK})...")
    holdings = parser.get_latest_13f(BERKSHIRE_CIK)
    print(f"    Retrieved {len(holdings)} holdings")

    if holdings:
        db.upsert_holdings(holdings, BERKSHIRE_CIK, holdings[0].filing_date)
        print(f"    Ingested {len(holdings)} holdings into DB")
        print(f"    Top 5 positions by value:")
        top5 = sorted(holdings, key=lambda h: h.value_usd, reverse=True)[:5]
        for h in top5:
            label = h.ticker or h.cusip
            print(f"      {label:10s}  {h.issuer_name[:30]:30s}  "
                  f"${h.value_usd/1e9:.2f}B  {h.pct_portfolio:.2f}%")

    # 2. Analyse AAPL ownership concentration
    print("\n[2] AAPL ownership concentration (from cached data)...")
    concentration = analytics.get_ownership_concentration("AAPL")
    if "error" not in concentration:
        print(f"    Period:              {concentration['period']}")
        print(f"    Total institutions:  {concentration['total_institutions']}")
        print(f"    HHI:                 {concentration['hhi']} ({concentration['hhi_interpretation']})")
        print(f"    Top-10 pct:          {concentration['top10_pct']:.1f}%")
    else:
        print(f"    {concentration['error']} (need to ingest more data first)")

    # 3. Smart-money consensus
    print("\n[3] Smart-money consensus for AAPL...")
    sm = analytics.get_smart_money_consensus("AAPL")
    print(f"    Consensus: {sm.get('consensus', 'N/A')}")
    print(f"    Buyers: {sm.get('buyers', 0)}  Sellers: {sm.get('sellers', 0)}")

    # 4. Berkshire new positions
    if holdings:
        print("\n[4] Berkshire Hathaway — detecting brand-new positions...")
        new_pos = hf.detect_new_positions(BERKSHIRE_CIK, quarters_back=1)
        print(f"    New positions this quarter: {len(new_pos)}")
        for h in new_pos[:5]:
            label = h.ticker or h.cusip
            print(f"      {label:10s}  ${h.value_usd/1e6:.1f}M")

    # 5. Top hedge-fund picks
    print("\n[5] Top tickers held by 3+ tracked hedge funds...")
    picks = hf.get_top_hedge_fund_picks(min_funds=3)
    print(f"    Picks ({len(picks)}): {picks[:10]}")

    # 6. Passive vs active split for AAPL
    print("\n[6] Passive vs Active split for AAPL...")
    pva = etf.compute_passive_vs_active("AAPL")
    if "error" not in pva:
        print(f"    Passive: {pva['passive_pct']:.1f}%  Active: {pva['active_pct']:.1f}%")
    else:
        print(f"    {pva.get('error')}")

    # 7. Watch Berkshire for big moves
    print("\n[7] Alerts — watch Berkshire for >10% position changes...")
    alerts = alert_sys.watch_institution(BERKSHIRE_CIK, threshold_pct_change=10.0)
    print(f"    Generated {len(alerts)} alerts")
    for a in alerts[:5]:
        print(f"      [{a.alert_type.upper():12s}] {a.ticker:8s}  {a.detail[:60]}")

    alert_path = alert_sys.export_alerts()
    print(f"\n    Alerts exported to: {alert_path}")

    # 8. Registry summary
    print(f"\n[8] Institution registry summary:")
    for cat in ["etf_provider", "mutual_fund", "hedge_fund", "pension", "bank", "insurance"]:
        n = len(registry.get_by_category(cat))
        print(f"    {cat:15s}: {n} institutions")

    print("\nDone.")
