"""ownership_screener_v3.py — Ownership-based screening platform (dim_072, score 6→9).

Combines EDGAR Form 4 insider transactions with 13F institutional ownership data
to produce multi-factor ownership signals for stock screening.

Architecture
------------
Form4Parser                  — Fetch and parse EDGAR Form 4 XML filings
InsiderSignalEngine          — Compute insider buying/selling scores
InstitutionalOwnershipAnalyzer — Analyze 13F data for ownership signals
OwnershipScreener            — Multi-factor screening with preset screens
OwnershipChangeMonitor       — Filing-date monitoring of ownership changes
OwnershipScreenerEngine      — Orchestrator: full picture → screen → export

Data sources
------------
  EDGAR EFTS (Form 4 search): https://efts.sec.gov/LATEST/search-index
  EDGAR Full-text search (Form 4 XML): SEC EDGAR Archives
  EDGAR 13F-HR: institutional_ownership_v3 (imported if available)
  EDGAR company search: CIK resolution via company_tickers.json

Public API
----------
Form4Parser.fetch_form4_filings(ticker, days) -> list[Form4Filing]
InsiderSignalEngine.compute_insider_score(ticker, days) -> InsiderScore
InsiderSignalEngine.get_cluster_buys(universe, days) -> pd.DataFrame
OwnershipScreener.screen(universe, criteria) -> pd.DataFrame
OwnershipScreener.run_preset(preset_name, universe) -> pd.DataFrame
OwnershipScreenerEngine.get_full_ownership_picture(ticker) -> OwnershipProfile
OwnershipScreenerEngine.run_all_screens(universe) -> pd.DataFrame
OwnershipScreenerEngine.rank_by_ownership_quality(universe) -> pd.DataFrame

Dependencies: requests, pandas (optional), duckdb (optional),
              xml.etree.ElementTree, dataclasses, datetime, logging
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _PANDAS = False

try:
    import duckdb
    _DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore
    _DUCKDB = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
# Paths & constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DUCKDB_PATH = _DATA_DIR / "ownership.duckdb"
_SQLITE_PATH = _DATA_DIR / "ownership_fallback.db"

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_EDGAR_BASE = "https://www.sec.gov"
_EFTS_BASE = "https://efts.sec.gov"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_RATE_DELAY = 0.4  # seconds between SEC requests

# Insider title patterns
_CEO_PATTERNS = re.compile(
    r"(chief\s+executive|ceo|president\s+and\s+ceo|c\.e\.o)", re.I
)
_CFO_PATTERNS = re.compile(
    r"(chief\s+financial|cfo|chief\s+fin\.|c\.f\.o)", re.I
)
_DIRECTOR_PATTERNS = re.compile(r"(director|board)", re.I)

# CIK resolution cache
_CIK_CACHE: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Form4Filing:
    ticker: str
    issuer_cik: str
    owner_name: str
    owner_cik: str
    owner_title: str
    is_ceo: bool
    is_cfo: bool
    is_director: bool
    is_officer: bool
    is_ten_pct_owner: bool
    transaction_type: str        # P, S, A, D, F, G, M, …
    transaction_code: str        # raw acquistionOrDisposition code
    transaction_date: str        # YYYY-MM-DD
    shares_transacted: float
    price_per_share: float
    total_value: float
    shares_after: float
    is_direct: bool              # D = direct, I = indirect
    is_plan: bool                # Rule 10b5-1 automatic plan flag
    accession_number: str = ""
    filing_date: str = ""

@dataclass
class InsiderScore:
    ticker: str
    as_of_date: str
    net_purchase_value: float = 0.0      # positive = net buyer
    net_purchase_shares: float = 0.0
    n_unique_buyers: int = 0
    n_unique_sellers: int = 0
    ceo_bought: bool = False
    cfo_bought: bool = False
    cluster_buy: bool = False            # ≥3 insiders buying in 30 days
    cluster_sell: bool = False
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    score: float = 0.0                   # composite 0–10

@dataclass
class OwnershipChanges:
    ticker: str
    quarter: str
    new_positions: List[Dict] = field(default_factory=list)
    eliminated_positions: List[Dict] = field(default_factory=list)
    increased_positions: List[Dict] = field(default_factory=list)
    decreased_positions: List[Dict] = field(default_factory=list)
    net_institutional_change_pct: float = 0.0
    total_institutional_pct: float = 0.0
    hhi: float = 0.0

@dataclass
class OwnershipCriteria:
    min_insider_score: float = 0.0
    require_cluster_buy: bool = False
    require_ceo_buy: bool = False
    min_institutional_change_pct: float = 0.0
    max_hhi: float = 1.0
    min_insider_ownership_pct: float = 0.0
    require_smart_money: bool = False
    max_institutional_pct: float = 100.0  # for "neglect" screen

@dataclass
class OwnershipProfile:
    ticker: str
    as_of_date: str
    insider_score: Optional[InsiderScore] = None
    institutional: Optional[OwnershipChanges] = None
    insider_ownership_pct: float = 0.0
    institutional_pct: float = 0.0
    hhi: float = 0.0
    smart_money_signal: float = 0.0
    composite_ownership_score: float = 0.0
    flags: List[str] = field(default_factory=list)

@dataclass
class OwnershipSignalDashboard:
    as_of_date: str
    universe: List[str] = field(default_factory=list)
    cluster_buys: List[str] = field(default_factory=list)     # tickers with cluster buys
    cluster_sells: List[str] = field(default_factory=list)
    ceo_buys: List[str] = field(default_factory=list)
    smart_money_accumulating: List[str] = field(default_factory=list)
    new_13f_filers: List[str] = field(default_factory=list)
    insider_sell_alerts: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers() -> Dict[str, str]:
    return {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _safe_get(url: str, params: Optional[Dict] = None,
              timeout: int = 20) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.debug(f"HTTP GET failed: {url} — {exc}")
        return None


def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit EDGAR CIK."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    resp = _safe_get("https://www.sec.gov/files/company_tickers.json")
    if resp:
        try:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
        except Exception:
            pass
    return None


def _parse_float(text: Optional[str]) -> float:
    """Parse float from text, return 0.0 on failure."""
    if not text:
        return 0.0
    try:
        return float(text.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _text(element: Optional[ET.Element], path: str) -> Optional[str]:
    """Extract text from XML element at path."""
    if element is None:
        return None
    el = element.find(path)
    return el.text.strip() if el is not None and el.text else None


def _is_ceo(title: str) -> bool:
    return bool(_CEO_PATTERNS.search(title))


def _is_cfo(title: str) -> bool:
    return bool(_CFO_PATTERNS.search(title))


def _is_director(title: str) -> bool:
    return bool(_DIRECTOR_PATTERNS.search(title))


# ---------------------------------------------------------------------------
# Form4Parser
# ---------------------------------------------------------------------------

class Form4Parser:
    """Fetch and parse EDGAR Form 4 filings for insider transaction data.

    Uses EDGAR EFTS (full-text search) to find filing accession numbers,
    then fetches the primary XML document from SEC Archives.
    """

    _EFTS_SEARCH = f"{_EFTS_BASE}/LATEST/search-index"
    _EFTS_SEARCH_ALT = f"{_EFTS_BASE}/LATEST/search-index?q={{ticker}}&forms=4"

    def fetch_form4_filings(self, ticker: str,
                            days: int = 180) -> List[Form4Filing]:
        """Fetch Form 4 filings for a ticker over the past N days."""
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        # Resolve CIK so we can search by entity
        cik = _resolve_cik(ticker)
        logger.info(f"Fetching Form 4 filings for {ticker} (CIK={cik}), {start_str}→{end_str}")

        accessions = self._search_efts(ticker, cik, start_str, end_str)
        if not accessions:
            logger.debug(f"No Form 4 accessions found for {ticker}")
            return []

        filings: List[Form4Filing] = []
        for acc_no, filing_date in accessions[:50]:  # cap at 50 filings
            time.sleep(_RATE_DELAY)
            xml_content = self._fetch_form4_xml(cik or "0", acc_no)
            if xml_content:
                parsed = self.parse_form4_xml(xml_content)
                if parsed:
                    parsed.accession_number = acc_no
                    parsed.filing_date = filing_date
                    if not parsed.ticker:
                        parsed.ticker = ticker
                    filings.append(parsed)

        logger.info(f"Parsed {len(filings)} Form 4 filings for {ticker}")
        return filings

    def _search_efts(self, ticker: str, cik: Optional[str],
                     start_str: str, end_str: str) -> List[Tuple[str, str]]:
        """Search EDGAR EFTS for Form 4 accession numbers."""
        # Try entity search with ticker symbol
        params = {
            "q": f'"{ticker}"',
            "forms": "4",
            "dateRange": "custom",
            "startdt": start_str,
            "enddt": end_str,
        }
        resp = _safe_get(self._EFTS_SEARCH, params=params)
        results = []
        if resp:
            results = self._parse_efts_response(resp)

        # If CIK is known and we got few results, also search by CIK
        if cik and len(results) < 5:
            time.sleep(_RATE_DELAY)
            params2 = {
                "q": "",
                "forms": "4",
                "dateRange": "custom",
                "startdt": start_str,
                "enddt": end_str,
                "entity": ticker,
            }
            resp2 = _safe_get(self._EFTS_SEARCH, params=params2)
            if resp2:
                extra = self._parse_efts_response(resp2)
                seen = {r[0] for r in results}
                results += [r for r in extra if r[0] not in seen]

        return results

    def _parse_efts_response(self, resp: requests.Response) -> List[Tuple[str, str]]:
        """Extract (accession_number, filing_date) pairs from EFTS response."""
        accessions: List[Tuple[str, str]] = []
        try:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits:
                src = hit.get("_source", {})
                acc = src.get("accession_no", "")
                filing_date = src.get("file_date", "")
                if acc:
                    accessions.append((acc.replace("-", ""), filing_date))
        except Exception as exc:
            logger.debug(f"EFTS parse error: {exc}")
        return accessions

    def _fetch_form4_xml(self, cik: str, accession_no: str) -> Optional[str]:
        """Fetch the primary Form 4 XML document from SEC Archives."""
        # Normalize accession: strip hyphens, then re-format as XX-XXXXXXXX-XXXXXXXXXX
        acc_clean = accession_no.replace("-", "")
        if len(acc_clean) != 18:
            return None
        acc_formatted = f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        acc_path = acc_formatted.replace("-", "")
        cik_clean = cik.lstrip("0") or "0"

        # Try to fetch filing index to find the primary XML
        index_url = (f"{_EDGAR_BASE}/cgi-bin/browse-edgar?"
                     f"action=getcompany&CIK={cik_clean}&type=4&dateb=&owner=include&count=1")
        # Direct approach: construct the archive URL
        archive_url = (f"{_ARCHIVES_BASE}/{cik_clean}/{acc_clean}/")
        idx_resp = _safe_get(f"{_EDGAR_BASE}/Archives/edgar/data/{cik_clean}/{acc_clean}/{acc_formatted}-index.htm")
        if idx_resp and idx_resp.text:
            # Extract .xml link from the index
            xml_link = self._extract_xml_link(idx_resp.text, acc_formatted)
            if xml_link:
                xml_resp = _safe_get(xml_link)
                if xml_resp:
                    return xml_resp.text

        # Fallback: try common naming pattern
        for suffix in [".xml", "-primary.xml", "xbrl.xml"]:
            url = f"{_ARCHIVES_BASE}/{cik_clean}/{acc_clean}/{acc_formatted}{suffix}"
            resp = _safe_get(url)
            if resp and resp.text.startswith("<?xml"):
                return resp.text

        return None

    def _extract_xml_link(self, html_text: str, acc_formatted: str) -> Optional[str]:
        """Extract the Form 4 XML file link from the filing index HTML."""
        # Look for .xml links in the index
        matches = re.findall(r'href="(/Archives/edgar/data/[^"]+\.xml)"', html_text, re.I)
        for m in matches:
            if "form4" in m.lower() or acc_formatted.lower() in m.lower() or "0000" in m:
                return f"{_EDGAR_BASE}{m}"
        if matches:
            return f"{_EDGAR_BASE}{matches[0]}"
        return None

    def parse_form4_xml(self, xml_content: str) -> Optional[Form4Filing]:
        """Parse Form 4 XML into a Form4Filing dataclass (first transaction only).

        For filings with multiple transactions, each should be parsed separately.
        This method returns the primary/first non-derivative transaction.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            logger.debug(f"Form 4 XML parse error: {exc}")
            return None

        # Issuer info
        ticker = _text(root, ".//issuerTradingSymbol") or ""
        issuer_cik = _text(root, ".//issuerCik") or ""

        # Reporting owner info
        owner_el = root.find(".//reportingOwner")
        if owner_el is None:
            return None

        owner_name = _text(owner_el, ".//rptOwnerName") or "Unknown"
        owner_cik = _text(owner_el, ".//rptOwnerCik") or ""
        is_director = _text(owner_el, ".//isDirector") == "1"
        is_officer = _text(owner_el, ".//isOfficer") == "1"
        is_ten_pct = _text(owner_el, ".//isTenPercentOwner") == "1"
        officer_title = _text(owner_el, ".//officerTitle") or ""

        is_ceo = _is_ceo(officer_title)
        is_cfo = _is_cfo(officer_title)

        # Look for non-derivative transactions first
        for tx_el in root.findall(".//nonDerivativeTransaction"):
            filing = self._parse_nonderivative_tx(
                tx_el, ticker, issuer_cik, owner_name, owner_cik,
                officer_title, is_director, is_officer, is_ten_pct, is_ceo, is_cfo
            )
            if filing:
                return filing

        # Fall back to derivative transactions
        for tx_el in root.findall(".//derivativeTransaction"):
            filing = self._parse_derivative_tx(
                tx_el, ticker, issuer_cik, owner_name, owner_cik,
                officer_title, is_director, is_officer, is_ten_pct, is_ceo, is_cfo
            )
            if filing:
                return filing

        return None

    def parse_form4_xml_all(self, xml_content: str) -> List[Form4Filing]:
        """Parse all transactions from a Form 4 XML."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return []

        ticker = _text(root, ".//issuerTradingSymbol") or ""
        issuer_cik = _text(root, ".//issuerCik") or ""

        owner_el = root.find(".//reportingOwner")
        if owner_el is None:
            return []

        owner_name = _text(owner_el, ".//rptOwnerName") or "Unknown"
        owner_cik = _text(owner_el, ".//rptOwnerCik") or ""
        is_director = _text(owner_el, ".//isDirector") == "1"
        is_officer = _text(owner_el, ".//isOfficer") == "1"
        is_ten_pct = _text(owner_el, ".//isTenPercentOwner") == "1"
        officer_title = _text(owner_el, ".//officerTitle") or ""
        is_ceo = _is_ceo(officer_title)
        is_cfo = _is_cfo(officer_title)

        filings: List[Form4Filing] = []
        for tx_el in root.findall(".//nonDerivativeTransaction"):
            f = self._parse_nonderivative_tx(
                tx_el, ticker, issuer_cik, owner_name, owner_cik,
                officer_title, is_director, is_officer, is_ten_pct, is_ceo, is_cfo
            )
            if f:
                filings.append(f)

        for tx_el in root.findall(".//derivativeTransaction"):
            f = self._parse_derivative_tx(
                tx_el, ticker, issuer_cik, owner_name, owner_cik,
                officer_title, is_director, is_officer, is_ten_pct, is_ceo, is_cfo
            )
            if f:
                filings.append(f)

        return filings

    def _parse_nonderivative_tx(self, tx_el: ET.Element, ticker: str, issuer_cik: str,
                                 owner_name: str, owner_cik: str, title: str,
                                 is_director: bool, is_officer: bool, is_ten_pct: bool,
                                 is_ceo: bool, is_cfo: bool) -> Optional[Form4Filing]:
        tx_code = _text(tx_el, ".//transactionCode") or ""
        tx_date = _text(tx_el, ".//transactionDate/value") or ""
        shares = _parse_float(_text(tx_el, ".//transactionShares/value"))
        price = _parse_float(_text(tx_el, ".//transactionPricePerShare/value"))
        acq_disp = _text(tx_el, ".//transactionAcquiredDisposedCode/value") or ""
        shares_after = _parse_float(_text(tx_el, ".//sharesOwnedFollowingTransaction/value"))
        ownership_form = _text(tx_el, ".//ownershipNatureCode") or "D"  # D=direct
        is_plan = bool(re.search(r"10b5-1|plan", _text(tx_el, ".//footnoteId") or "", re.I))

        # Map to buy/sell
        if acq_disp == "A":
            transaction_type = "P"  # purchase/acquisition
        elif acq_disp == "D":
            transaction_type = "S"  # sale/disposition
        else:
            transaction_type = tx_code

        total_value = shares * price

        if shares == 0 and total_value == 0:
            return None

        return Form4Filing(
            ticker=ticker.upper(),
            issuer_cik=issuer_cik,
            owner_name=owner_name,
            owner_cik=owner_cik,
            owner_title=title,
            is_ceo=is_ceo,
            is_cfo=is_cfo,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_pct_owner=is_ten_pct,
            transaction_type=transaction_type,
            transaction_code=tx_code,
            transaction_date=tx_date,
            shares_transacted=shares,
            price_per_share=price,
            total_value=total_value,
            shares_after=shares_after,
            is_direct=(ownership_form == "D"),
            is_plan=is_plan,
        )

    def _parse_derivative_tx(self, tx_el: ET.Element, ticker: str, issuer_cik: str,
                               owner_name: str, owner_cik: str, title: str,
                               is_director: bool, is_officer: bool, is_ten_pct: bool,
                               is_ceo: bool, is_cfo: bool) -> Optional[Form4Filing]:
        tx_code = _text(tx_el, ".//transactionCode") or ""
        tx_date = _text(tx_el, ".//transactionDate/value") or ""
        shares = _parse_float(_text(tx_el, ".//transactionShares/value"))
        price = _parse_float(_text(tx_el, ".//exercisePrice/value"))
        acq_disp = _text(tx_el, ".//transactionAcquiredDisposedCode/value") or ""
        shares_after = _parse_float(_text(tx_el, ".//sharesOwnedFollowingTransaction/value"))

        transaction_type = "A" if acq_disp == "A" else "D"  # award vs derivative disposition
        total_value = shares * price

        return Form4Filing(
            ticker=ticker.upper(),
            issuer_cik=issuer_cik,
            owner_name=owner_name,
            owner_cik=owner_cik,
            owner_title=title,
            is_ceo=is_ceo,
            is_cfo=is_cfo,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_pct_owner=is_ten_pct,
            transaction_type=transaction_type,
            transaction_code=tx_code,
            transaction_date=tx_date,
            shares_transacted=shares,
            price_per_share=price,
            total_value=total_value,
            shares_after=shares_after,
            is_direct=True,
            is_plan=False,
        )

    def get_insider_universe(self, ticker: str, years: int = 2) -> List[str]:
        """Get all insider names who filed Form 4 for this ticker in the last N years."""
        days = years * 365
        filings = self.fetch_form4_filings(ticker, days=days)
        names: Set[str] = set()
        for f in filings:
            if f.owner_name and f.owner_name != "Unknown":
                names.add(f.owner_name)
        return sorted(names)


# ---------------------------------------------------------------------------
# InsiderSignalEngine
# ---------------------------------------------------------------------------

class InsiderSignalEngine:
    """Compute insider buying/selling composite signals from Form 4 data."""

    def __init__(self):
        self._parser = Form4Parser()
        self._db = OwnershipDatabase()

    def compute_insider_score(self, ticker: str, days: int = 90) -> InsiderScore:
        """Compute comprehensive insider score for a ticker.

        Scoring rules:
          - Net purchases > 0: positive
          - Each unique buyer: +1 point
          - CEO buy: +2 bonus
          - CFO buy: +1.5 bonus
          - Cluster buy (3+ insiders): +3 bonus
          - Exclude: awards (A/D codes), 10b5-1 plans
        """
        as_of = datetime.today().strftime("%Y-%m-%d")

        # Try to load from DB first
        cached = self._db.get_insider_score(ticker)
        if cached and cached.as_of_date == as_of:
            return cached

        filings = self._parser.fetch_form4_filings(ticker, days=days)

        # Filter: exclude awards and automatic plans
        open_market = [
            f for f in filings
            if f.transaction_type in ("P", "S")
            and f.transaction_code not in ("A", "D", "G", "L", "M", "X")
            and not f.is_plan
        ]

        buyers: Set[str] = set()
        sellers: Set[str] = set()
        total_buy = 0.0
        total_sell = 0.0
        ceo_bought = False
        cfo_bought = False

        for f in open_market:
            if f.transaction_type == "P":
                buyers.add(f.owner_name)
                total_buy += f.total_value
                if f.is_ceo:
                    ceo_bought = True
                if f.is_cfo:
                    cfo_bought = True
            elif f.transaction_type == "S":
                sellers.add(f.owner_name)
                total_sell += f.total_value

        # Cluster buy detection: ≥3 unique insiders buying within 30 days
        cluster_buy = self._detect_cluster(open_market, min_insiders=3, window_days=30, tx_type="P")
        cluster_sell = self._detect_cluster(open_market, min_insiders=3, window_days=30, tx_type="S")

        net_value = total_buy - total_sell
        net_shares = sum(
            (f.shares_transacted if f.transaction_type == "P" else -f.shares_transacted)
            for f in open_market
        )

        # Composite score (0–10 scale)
        score = 0.0
        if net_value > 0:
            score += min(3.0, math.log10(max(1, net_value)) / 2)  # 0–3 for purchase size
        score += min(2.0, len(buyers) * 0.5)                       # 0–2 for buyer count
        if ceo_bought:
            score += 2.0
        if cfo_bought:
            score += 1.5
        if cluster_buy:
            score += 3.0
        if cluster_sell and net_value < 0:
            score -= 2.0
        score = max(0.0, min(10.0, score))

        result = InsiderScore(
            ticker=ticker,
            as_of_date=as_of,
            net_purchase_value=net_value,
            net_purchase_shares=net_shares,
            n_unique_buyers=len(buyers),
            n_unique_sellers=len(sellers),
            ceo_bought=ceo_bought,
            cfo_bought=cfo_bought,
            cluster_buy=cluster_buy,
            cluster_sell=cluster_sell,
            total_buy_value=total_buy,
            total_sell_value=total_sell,
            score=score,
        )
        self._db.store_insider_score(result)
        return result

    def _detect_cluster(self, filings: List[Form4Filing], min_insiders: int,
                        window_days: int, tx_type: str) -> bool:
        """Detect if min_insiders unique people transacted within window_days."""
        type_filings = [f for f in filings if f.transaction_type == tx_type and f.transaction_date]
        if len(type_filings) < min_insiders:
            return False

        # Convert dates
        dated: List[Tuple[datetime, str]] = []
        for f in type_filings:
            try:
                dt = datetime.strptime(f.transaction_date[:10], "%Y-%m-%d")
                dated.append((dt, f.owner_name))
            except ValueError:
                continue

        if len(dated) < min_insiders:
            return False

        # Sliding window
        dated.sort(key=lambda x: x[0])
        for i, (dt_i, _) in enumerate(dated):
            window_end = dt_i + timedelta(days=window_days)
            window_insiders = {name for dt, name in dated if dt_i <= dt <= window_end}
            if len(window_insiders) >= min_insiders:
                return True

        return False

    def get_cluster_buys(self, universe: List[str], days: int = 60) -> "pd.DataFrame":
        """Return DataFrame of tickers with recent cluster buying events."""
        if not _PANDAS:
            return pd.DataFrame()

        rows = []
        for ticker in universe:
            try:
                score = self.compute_insider_score(ticker, days=days)
                if score.cluster_buy:
                    rows.append({
                        "ticker": ticker,
                        "cluster_buy": True,
                        "n_buyers": score.n_unique_buyers,
                        "total_buy_value": score.total_buy_value,
                        "ceo_bought": score.ceo_bought,
                        "score": score.score,
                    })
                time.sleep(0.2)
            except Exception as exc:
                logger.debug(f"Cluster buy check failed for {ticker}: {exc}")

        if not rows:
            return pd.DataFrame(columns=["ticker", "cluster_buy", "n_buyers",
                                          "total_buy_value", "ceo_bought", "score"])
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    def get_cluster_sells(self, universe: List[str], days: int = 60) -> "pd.DataFrame":
        """Return DataFrame of tickers with recent cluster selling events."""
        if not _PANDAS:
            return pd.DataFrame()

        rows = []
        for ticker in universe:
            try:
                score = self.compute_insider_score(ticker, days=days)
                if score.cluster_sell:
                    rows.append({
                        "ticker": ticker,
                        "cluster_sell": True,
                        "n_sellers": score.n_unique_sellers,
                        "total_sell_value": score.total_sell_value,
                        "net_purchase_value": score.net_purchase_value,
                    })
                time.sleep(0.2)
            except Exception as exc:
                logger.debug(f"Cluster sell check failed for {ticker}: {exc}")

        if not rows:
            return pd.DataFrame(columns=["ticker", "cluster_sell", "n_sellers",
                                          "total_sell_value", "net_purchase_value"])
        return pd.DataFrame(rows).reset_index(drop=True)

    def compute_insider_momentum(self, ticker: str) -> float:
        """Direction of insider activity trend: positive = increasing buying.

        Compares rolling 30-day net purchases vs 30–90 day net purchases.
        """
        filings_90 = self._parser.fetch_form4_filings(ticker, days=90)
        open_mkt = [f for f in filings_90
                    if f.transaction_type in ("P", "S") and not f.is_plan
                    and f.transaction_code not in ("A", "D")]

        cutoff = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent, prior = [], []
        for f in open_mkt:
            val = f.total_value if f.transaction_type == "P" else -f.total_value
            if f.transaction_date >= cutoff:
                recent.append(val)
            else:
                prior.append(val)

        recent_net = sum(recent)
        prior_net = sum(prior)
        if abs(prior_net) < 1:
            return 1.0 if recent_net > 0 else -1.0 if recent_net < 0 else 0.0
        return (recent_net - prior_net) / abs(prior_net)

    def compute_ceo_confidence_index(self, universe: List[str]) -> float:
        """Aggregate CEO net buying direction across universe (−1 to +1)."""
        signals: List[float] = []
        for ticker in universe:
            try:
                score = self.compute_insider_score(ticker, days=90)
                if score.ceo_bought:
                    signals.append(1.0)
                elif score.n_unique_sellers > 0 and score.total_sell_value > score.total_buy_value:
                    signals.append(-1.0)
                else:
                    signals.append(0.0)
                time.sleep(0.15)
            except Exception:
                pass
        if not signals:
            return 0.0
        return sum(signals) / len(signals)


# ---------------------------------------------------------------------------
# InstitutionalOwnershipAnalyzer
# ---------------------------------------------------------------------------

class InstitutionalOwnershipAnalyzer:
    """Analyze 13F institutional ownership data.

    Imports from institutional_ownership_v3 when available;
    falls back to direct EDGAR 13F XML parsing.
    """

    def __init__(self):
        self._inst_module = None
        try:
            from sentinel.sfe import institutional_ownership_v3 as _inst
            self._inst_module = _inst
            logger.info("Using institutional_ownership_v3 for 13F data")
        except ImportError:
            logger.info("institutional_ownership_v3 not available — using fallback 13F parser")

    def get_institutional_changes(self, ticker: str,
                                  quarter: Optional[str] = None) -> OwnershipChanges:
        """Fetch institutional ownership changes for the latest quarter."""
        if quarter is None:
            quarter = self._latest_quarter()

        if self._inst_module:
            try:
                return self._from_inst_module(ticker, quarter)
            except Exception as exc:
                logger.debug(f"institutional_ownership_v3 error: {exc}")

        return self._fallback_13f(ticker, quarter)

    def _from_inst_module(self, ticker: str, quarter: str) -> OwnershipChanges:
        """Delegate to institutional_ownership_v3."""
        analytics = self._inst_module.OwnershipAnalytics()
        delta = analytics.detect_ownership_change(ticker, quarter, quarter)
        new_pos = []
        elim_pos = []
        if hasattr(delta, "new_positions"):
            new_pos = [{"fund": p} for p in (delta.new_positions or [])]
        if hasattr(delta, "eliminated_positions"):
            elim_pos = [{"fund": p} for p in (delta.eliminated_positions or [])]

        conc = analytics.get_ownership_concentration(ticker)
        hhi = conc.get("hhi", 0.0) if conc else 0.0
        total_pct = conc.get("total_pct", 0.0) if conc else 0.0

        return OwnershipChanges(
            ticker=ticker,
            quarter=quarter,
            new_positions=new_pos,
            eliminated_positions=elim_pos,
            hhi=hhi,
            total_institutional_pct=total_pct,
        )

    def _fallback_13f(self, ticker: str, quarter: str) -> OwnershipChanges:
        """Direct EDGAR 13F-HR parsing as fallback."""
        cik = _resolve_cik(ticker)
        if not cik:
            return OwnershipChanges(ticker=ticker, quarter=quarter)

        # Find the latest 13F-HR filing for this ticker from filers
        # (full 13F universe scraping is expensive — return empty with note)
        logger.debug(f"13F fallback: limited data available for {ticker} without inst_module")
        return OwnershipChanges(ticker=ticker, quarter=quarter)

    def compute_ownership_concentration(self, ticker: str) -> float:
        """Compute HHI of top-10 institutional holders."""
        if self._inst_module:
            try:
                analytics = self._inst_module.OwnershipAnalytics()
                conc = analytics.get_ownership_concentration(ticker)
                return float(conc.get("hhi", 0.0))
            except Exception:
                pass
        return 0.0

    def detect_new_positions(self, ticker: str) -> List[Dict]:
        """Funds that initiated a new position this quarter."""
        changes = self.get_institutional_changes(ticker)
        return changes.new_positions

    def detect_eliminated_positions(self, ticker: str) -> List[Dict]:
        """Funds that sold out completely this quarter."""
        changes = self.get_institutional_changes(ticker)
        return changes.eliminated_positions

    def get_smart_money_signal(self, ticker: str, top_n_funds: int = 20) -> float:
        """Net buying direction from 'smart money' (top 20 funds by implied alpha).

        Smart money proxy: funds with largest AUM are used as a heuristic.
        Returns −1.0 (selling) to +1.0 (buying).
        """
        if self._inst_module:
            try:
                analytics = self._inst_module.OwnershipAnalytics()
                result = analytics.get_smart_money_consensus(ticker)
                return float(result.get("consensus_direction", 0.0))
            except Exception:
                pass
        return 0.0

    def compute_short_interest_vs_institutional(self, ticker: str) -> Dict[str, float]:
        """Compare institutional ownership vs short interest (proxy from yfinance)."""
        result: Dict[str, float] = {
            "institutional_pct": 0.0,
            "short_interest_ratio": 0.0,
            "divergence": 0.0,
        }
        changes = self.get_institutional_changes(ticker)
        result["institutional_pct"] = changes.total_institutional_pct

        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            short_pct = info.get("shortPercentOfFloat", 0.0) or 0.0
            result["short_interest_ratio"] = float(short_pct)
            result["divergence"] = changes.total_institutional_pct - float(short_pct) * 100
        except Exception:
            pass

        return result

    @staticmethod
    def _latest_quarter() -> str:
        """Return the most recently completed fiscal quarter (e.g., '2025Q1')."""
        today = datetime.today()
        q = (today.month - 1) // 3  # 0=Q1, 1=Q2, 2=Q3, 3=Q4
        if q == 0:
            return f"{today.year - 1}Q4"
        return f"{today.year}Q{q}"


# ---------------------------------------------------------------------------
# OwnershipDatabase
# ---------------------------------------------------------------------------

class OwnershipDatabase:
    """DuckDB (SQLite fallback) persistence for ownership signals."""

    def __init__(self):
        self._conn = None
        self._use_duckdb = _DUCKDB
        self._init_db()

    def _get_conn(self):
        if self._conn is not None:
            return self._conn
        if self._use_duckdb:
            try:
                self._conn = duckdb.connect(str(_DUCKDB_PATH))
                return self._conn
            except Exception:
                self._use_duckdb = False
        self._conn = sqlite3.connect(str(_SQLITE_PATH), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        ddl_insider = """
        CREATE TABLE IF NOT EXISTS insider_scores (
            ticker       VARCHAR NOT NULL,
            as_of_date   VARCHAR NOT NULL,
            net_value    DOUBLE,
            n_buyers     INTEGER,
            n_sellers    INTEGER,
            ceo_bought   BOOLEAN,
            cluster_buy  BOOLEAN,
            score        DOUBLE,
            PRIMARY KEY  (ticker, as_of_date)
        )
        """
        ddl_form4 = """
        CREATE TABLE IF NOT EXISTS form4_filings (
            ticker           VARCHAR,
            filing_date      VARCHAR,
            owner_name       VARCHAR,
            owner_title      VARCHAR,
            transaction_type VARCHAR,
            transaction_date VARCHAR,
            shares_transacted DOUBLE,
            price_per_share  DOUBLE,
            total_value      DOUBLE,
            is_ceo           BOOLEAN,
            is_cfo           BOOLEAN,
            is_plan          BOOLEAN
        )
        """
        ddl_watch = """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker VARCHAR PRIMARY KEY,
            added_date VARCHAR
        )
        """
        try:
            for ddl in [ddl_insider, ddl_form4, ddl_watch]:
                conn.execute(ddl)
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception as exc:
            logger.debug(f"OwnershipDatabase DDL error: {exc}")

    def store_insider_score(self, score: InsiderScore):
        conn = self._get_conn()
        sql = ("INSERT OR REPLACE INTO insider_scores "
               "(ticker, as_of_date, net_value, n_buyers, n_sellers, ceo_bought, cluster_buy, score) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
        try:
            conn.execute(sql, (
                score.ticker, score.as_of_date, score.net_purchase_value,
                score.n_unique_buyers, score.n_unique_sellers,
                score.ceo_bought, score.cluster_buy, score.score
            ))
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception:
            pass

    def get_insider_score(self, ticker: str) -> Optional[InsiderScore]:
        conn = self._get_conn()
        sql = ("SELECT ticker, as_of_date, net_value, n_buyers, n_sellers, "
               "ceo_bought, cluster_buy, score FROM insider_scores WHERE ticker=? "
               "ORDER BY as_of_date DESC LIMIT 1")
        try:
            if self._use_duckdb:
                row = conn.execute(sql, [ticker]).fetchone()
            else:
                row = conn.execute(sql, (ticker,)).fetchone()
            if row:
                return InsiderScore(
                    ticker=row[0], as_of_date=row[1],
                    net_purchase_value=row[2] or 0.0,
                    n_unique_buyers=row[3] or 0,
                    n_unique_sellers=row[4] or 0,
                    ceo_bought=bool(row[5]),
                    cluster_buy=bool(row[6]),
                    score=row[7] or 0.0,
                )
        except Exception:
            pass
        return None

    def store_form4(self, filing: Form4Filing):
        conn = self._get_conn()
        sql = ("INSERT INTO form4_filings "
               "(ticker, filing_date, owner_name, owner_title, transaction_type, "
               "transaction_date, shares_transacted, price_per_share, total_value, "
               "is_ceo, is_cfo, is_plan) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        try:
            conn.execute(sql, (
                filing.ticker, filing.filing_date, filing.owner_name, filing.owner_title,
                filing.transaction_type, filing.transaction_date, filing.shares_transacted,
                filing.price_per_share, filing.total_value, filing.is_ceo, filing.is_cfo,
                filing.is_plan
            ))
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception:
            pass

    def get_watchlist(self) -> List[str]:
        conn = self._get_conn()
        try:
            if self._use_duckdb:
                rows = conn.execute("SELECT ticker FROM watchlist").fetchall()
            else:
                rows = conn.execute("SELECT ticker FROM watchlist").fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def add_to_watchlist(self, ticker: str):
        conn = self._get_conn()
        today = datetime.today().strftime("%Y-%m-%d")
        try:
            conn.execute("INSERT OR REPLACE INTO watchlist (ticker, added_date) VALUES (?, ?)",
                         (ticker, today) if not self._use_duckdb else [ticker, today])
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OwnershipScreener
# ---------------------------------------------------------------------------

class OwnershipScreener:
    """Multi-factor ownership screening system with preset screens.

    Presets
    -------
    cluster_buy             : 3+ insiders buying in 60 days
    ceo_buy                 : CEO purchased shares in last 90 days
    smart_money_accumulate  : top institutional funds increasing position
    high_insider_ownership  : insiders own >15%
    institutional_neglect   : <10% institutional ownership (undiscovered)
    insider_sell_alert      : heavy insider selling (risk flag)
    """

    PRESETS: Dict[str, OwnershipCriteria] = {
        "cluster_buy": OwnershipCriteria(require_cluster_buy=True),
        "ceo_buy": OwnershipCriteria(require_ceo_buy=True),
        "smart_money_accumulate": OwnershipCriteria(
            require_smart_money=True,
            min_institutional_change_pct=2.0
        ),
        "high_insider_ownership": OwnershipCriteria(min_insider_ownership_pct=15.0),
        "institutional_neglect": OwnershipCriteria(
            max_institutional_pct=10.0,
            min_insider_score=3.0
        ),
        "insider_sell_alert": OwnershipCriteria(min_insider_score=-100.0),  # catch negatives
    }

    def __init__(self):
        self._insider_engine = InsiderSignalEngine()
        self._inst_analyzer = InstitutionalOwnershipAnalyzer()

    def screen(self, universe: List[str],
               criteria: OwnershipCriteria) -> "pd.DataFrame":
        """Screen universe against ownership criteria, return matching tickers."""
        if not _PANDAS:
            return pd.DataFrame()

        rows = []
        for ticker in universe:
            try:
                row = self._evaluate_ticker(ticker, criteria)
                if row is not None:
                    rows.append(row)
                time.sleep(0.25)
            except Exception as exc:
                logger.debug(f"Screen eval failed for {ticker}: {exc}")

        if not rows:
            return pd.DataFrame(columns=["ticker", "score", "cluster_buy",
                                          "ceo_bought", "smart_money", "flags"])
        df = pd.DataFrame(rows)
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    def _evaluate_ticker(self, ticker: str,
                         criteria: OwnershipCriteria) -> Optional[Dict]:
        """Evaluate one ticker against criteria. Return dict if passes, else None."""
        score_obj = self._insider_engine.compute_insider_score(ticker, days=90)
        changes = self._inst_analyzer.get_institutional_changes(ticker)
        smart_money = self._inst_analyzer.get_smart_money_signal(ticker)

        # Apply criteria filters
        if criteria.require_cluster_buy and not score_obj.cluster_buy:
            return None
        if criteria.require_ceo_buy and not score_obj.ceo_bought:
            return None
        if criteria.min_insider_score > 0 and score_obj.score < criteria.min_insider_score:
            return None
        if (criteria.min_institutional_change_pct > 0
                and changes.net_institutional_change_pct < criteria.min_institutional_change_pct):
            return None
        if criteria.max_hhi < 1.0 and changes.hhi > criteria.max_hhi:
            return None
        if (criteria.require_smart_money and smart_money <= 0):
            return None
        if (criteria.max_institutional_pct < 100.0
                and changes.total_institutional_pct > criteria.max_institutional_pct):
            return None

        flags = []
        if score_obj.cluster_buy:
            flags.append("CLUSTER_BUY")
        if score_obj.ceo_bought:
            flags.append("CEO_BUY")
        if score_obj.cfo_bought:
            flags.append("CFO_BUY")
        if score_obj.cluster_sell:
            flags.append("CLUSTER_SELL")
        if changes.new_positions:
            flags.append(f"NEW_INST({len(changes.new_positions)})")
        if smart_money > 0:
            flags.append("SMART_MONEY_BUYING")

        return {
            "ticker": ticker,
            "score": score_obj.score,
            "net_buy_value": score_obj.net_purchase_value,
            "n_buyers": score_obj.n_unique_buyers,
            "n_sellers": score_obj.n_unique_sellers,
            "cluster_buy": score_obj.cluster_buy,
            "ceo_bought": score_obj.ceo_bought,
            "smart_money": smart_money,
            "inst_hhi": changes.hhi,
            "inst_pct": changes.total_institutional_pct,
            "new_positions": len(changes.new_positions),
            "flags": "; ".join(flags),
        }

    def run_preset(self, preset_name: str,
                   universe: Optional[List[str]] = None) -> "pd.DataFrame":
        """Run a named preset screen on the universe."""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                             f"Available: {list(self.PRESETS.keys())}")

        if universe is None:
            universe = self._default_universe()

        criteria = self.PRESETS[preset_name]
        logger.info(f"Running preset '{preset_name}' on {len(universe)} tickers")
        return self.screen(universe, criteria)

    @staticmethod
    def _default_universe() -> List[str]:
        """Default S&P 500 proxy universe."""
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
            "JPM", "V", "UNH", "XOM", "COST", "MA", "HD", "PG", "JNJ", "ORCL",
            "BAC", "ABBV", "MRK", "KO", "CVX", "CRM", "NFLX", "AMD", "PEP", "ADBE",
            "TMO", "WMT", "LIN", "ACN", "MCD", "CSCO", "ABT", "PM", "DHR", "CAT",
            "TXN", "INTC", "AMGN", "INTU", "WFC", "HON", "IBM", "GS", "SPGI", "BX",
        ]


# ---------------------------------------------------------------------------
# OwnershipChangeMonitor
# ---------------------------------------------------------------------------

class OwnershipChangeMonitor:
    """Real-time (filing-date) monitoring of ownership changes.

    Watches a list of tickers and institutions for new Form 4 and 13F filings.
    """

    def __init__(self):
        self._parser = Form4Parser()
        self._db = OwnershipDatabase()
        self._watchlist: Set[str] = set(self._db.get_watchlist())

    def check_new_13f_filings(self, since: str) -> List[Dict]:
        """Find new 13F-HR filings submitted since a given date."""
        url = (f"{_EDGAR_BASE}/cgi-bin/browse-edgar?"
               f"action=getcompany&type=13F-HR&dateb=&owner=include&count=40&search_text=")
        params = {
            "action": "getcurrent",
            "type": "13F-HR",
            "dateb": "",
            "owner": "include",
            "count": "40",
            "search_text": "",
            "start": "0",
        }
        resp = _safe_get(f"{_EDGAR_BASE}/cgi-bin/browse-edgar", params=params)
        filings: List[Dict] = []
        if resp:
            try:
                # Parse filing list from EDGAR HTML
                filing_blocks = re.findall(
                    r'<td class="normal">(.*?)</td>.*?'
                    r'href="(/cgi-bin/browse-edgar[^"]+)".*?'
                    r'(\d{4}-\d{2}-\d{2})',
                    resp.text, re.S
                )
                for filer_name, link, filing_date in filing_blocks[:20]:
                    if filing_date >= since:
                        filings.append({
                            "filer": filer_name.strip(),
                            "link": f"{_EDGAR_BASE}{link}",
                            "filing_date": filing_date,
                        })
            except Exception:
                pass
        return filings

    def check_new_form4_filings(self, tickers: List[str],
                                since: str) -> List[Form4Filing]:
        """Check for new Form 4 filings for a list of tickers since a given date."""
        all_filings: List[Form4Filing] = []
        for ticker in tickers:
            try:
                since_dt = datetime.strptime(since, "%Y-%m-%d")
                days = (datetime.today() - since_dt).days + 1
                filings = self._parser.fetch_form4_filings(ticker, days=days)
                new = [f for f in filings if f.filing_date >= since]
                all_filings.extend(new)
                time.sleep(0.3)
            except Exception as exc:
                logger.debug(f"Form 4 check failed for {ticker}: {exc}")
        return all_filings

    def get_latest_signals(self, universe: List[str]) -> OwnershipSignalDashboard:
        """Compute ownership signals dashboard for a universe."""
        as_of = datetime.today().strftime("%Y-%m-%d")
        dashboard = OwnershipSignalDashboard(as_of_date=as_of, universe=universe)

        insider_engine = InsiderSignalEngine()
        for ticker in universe:
            try:
                score = insider_engine.compute_insider_score(ticker, days=60)
                if score.cluster_buy:
                    dashboard.cluster_buys.append(ticker)
                if score.cluster_sell:
                    dashboard.cluster_sells.append(ticker)
                if score.ceo_bought:
                    dashboard.ceo_buys.append(ticker)
                if score.total_sell_value > score.total_buy_value * 3 and score.n_unique_sellers >= 2:
                    dashboard.insider_sell_alerts.append(ticker)
                time.sleep(0.2)
            except Exception as exc:
                logger.debug(f"Signal check failed for {ticker}: {exc}")

        return dashboard

    def subscribe_ticker(self, ticker: str):
        """Add a ticker to the monitoring watchlist."""
        self._watchlist.add(ticker.upper())
        self._db.add_to_watchlist(ticker.upper())
        logger.info(f"Subscribed to {ticker} ownership monitoring")

    def generate_daily_alert_report(self) -> str:
        """Generate a plain-text daily alert report for the watchlist."""
        watchlist = list(self._watchlist)
        if not watchlist:
            return "Watchlist is empty. Use subscribe_ticker(ticker) to add stocks."

        lines = [
            "=" * 65,
            "SENTINEL OWNERSHIP MONITOR — Daily Alert Report",
            f"Date: {datetime.today().strftime('%Y-%m-%d %H:%M')}",
            f"Watchlist: {len(watchlist)} tickers",
            "=" * 65,
        ]

        dashboard = self.get_latest_signals(watchlist)

        if dashboard.cluster_buys:
            lines.append(f"\nCLUSTER BUYS: {', '.join(dashboard.cluster_buys)}")
        if dashboard.ceo_buys:
            lines.append(f"CEO BUYS:     {', '.join(dashboard.ceo_buys)}")
        if dashboard.cluster_sells:
            lines.append(f"CLUSTER SELLS:{', '.join(dashboard.cluster_sells)}")
        if dashboard.insider_sell_alerts:
            lines.append(f"SELL ALERTS:  {', '.join(dashboard.insider_sell_alerts)}")

        if not any([dashboard.cluster_buys, dashboard.ceo_buys,
                    dashboard.cluster_sells, dashboard.insider_sell_alerts]):
            lines.append("\nNo significant ownership changes detected today.")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OwnershipScreenerEngine (Orchestrator)
# ---------------------------------------------------------------------------

class OwnershipScreenerEngine:
    """Full ownership analysis pipeline: profile → screen → rank → export."""

    def __init__(self):
        self._insider_engine = InsiderSignalEngine()
        self._inst_analyzer = InstitutionalOwnershipAnalyzer()
        self._screener = OwnershipScreener()
        self._monitor = OwnershipChangeMonitor()
        self._db = OwnershipDatabase()

    def get_full_ownership_picture(self, ticker: str) -> OwnershipProfile:
        """Compute comprehensive ownership profile for a single ticker."""
        as_of = datetime.today().strftime("%Y-%m-%d")
        logger.info(f"Building ownership profile for {ticker}")

        insider_score = self._insider_engine.compute_insider_score(ticker, days=90)
        institutional = self._inst_analyzer.get_institutional_changes(ticker)
        smart_money = self._inst_analyzer.get_smart_money_signal(ticker)
        hhi = self._inst_analyzer.compute_ownership_concentration(ticker)

        # Composite score: blend insider + institutional signals
        composite = 0.0
        composite += min(5.0, insider_score.score * 0.5)        # insider: up to 5 pts
        if institutional.net_institutional_change_pct > 2.0:
            composite += 2.0
        elif institutional.net_institutional_change_pct > 0:
            composite += 1.0
        if smart_money > 0:
            composite += 2.0
        if len(institutional.new_positions) > 0:
            composite += min(1.0, len(institutional.new_positions) * 0.25)
        composite = min(10.0, composite)

        flags: List[str] = []
        if insider_score.cluster_buy:
            flags.append("CLUSTER_BUY")
        if insider_score.ceo_bought:
            flags.append("CEO_BUY")
        if insider_score.cluster_sell:
            flags.append("CLUSTER_SELL")
        if smart_money > 0:
            flags.append("SMART_MONEY")
        if hhi < 0.1:
            flags.append("DIVERSE_OWNERSHIP")
        if insider_score.net_purchase_value < -1_000_000:
            flags.append("HEAVY_SELL")

        return OwnershipProfile(
            ticker=ticker,
            as_of_date=as_of,
            insider_score=insider_score,
            institutional=institutional,
            insider_ownership_pct=0.0,  # requires proxy statement parsing (Form DEF14A)
            institutional_pct=institutional.total_institutional_pct,
            hhi=hhi,
            smart_money_signal=smart_money,
            composite_ownership_score=composite,
            flags=flags,
        )

    def run_all_screens(self, universe: List[str]) -> "pd.DataFrame":
        """Run all preset screens and return combined results."""
        if not _PANDAS:
            return pd.DataFrame()

        preset_results: Dict[str, "pd.DataFrame"] = {}
        for preset_name in OwnershipScreener.PRESETS:
            logger.info(f"Running preset: {preset_name}")
            df = self._screener.run_preset(preset_name, universe)
            if not df.empty:
                df["screen"] = preset_name
                preset_results[preset_name] = df

        if not preset_results:
            return pd.DataFrame()

        combined = pd.concat(preset_results.values(), ignore_index=True)
        return combined

    def rank_by_ownership_quality(self, universe: List[str]) -> "pd.DataFrame":
        """Compute composite ownership quality score for each ticker and rank."""
        if not _PANDAS:
            return pd.DataFrame()

        rows = []
        for ticker in universe:
            try:
                profile = self.get_full_ownership_picture(ticker)
                rows.append({
                    "ticker": ticker,
                    "composite_score": profile.composite_ownership_score,
                    "insider_score": profile.insider_score.score if profile.insider_score else 0.0,
                    "smart_money": profile.smart_money_signal,
                    "hhi": profile.hhi,
                    "institutional_pct": profile.institutional_pct,
                    "flags": "; ".join(profile.flags),
                    "as_of": profile.as_of_date,
                })
                time.sleep(0.3)
            except Exception as exc:
                logger.debug(f"Profile failed for {ticker}: {exc}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def export_signals(self, path: str):
        """Export ranking for default universe to CSV."""
        universe = OwnershipScreener._default_universe()
        df = self.rank_by_ownership_quality(universe)
        if not df.empty:
            df.to_csv(path, index=False)
            logger.info(f"Ownership signals exported to {path}")
        else:
            logger.warning("No data to export")

    def get_dashboard(self, universe: Optional[List[str]] = None) -> OwnershipSignalDashboard:
        """Get real-time signal dashboard."""
        if universe is None:
            universe = OwnershipScreener._default_universe()
        return self._monitor.get_latest_signals(universe)


# ---------------------------------------------------------------------------
# Additional utilities
# ---------------------------------------------------------------------------

def fetch_finra_short_interest(ticker: str) -> Optional[float]:
    """Attempt to fetch short interest ratio from FINRA Reg SHO data.

    FINRA publishes short interest data at:
    https://www.finra.org/investors/learn-to-invest/advanced-investing/short-sale-data

    Returns: short interest as % of float (or None if unavailable).
    Note: Requires FINRA API key for programmatic access — returns None otherwise.
    """
    # FINRA short interest is not freely downloadable via API without registration.
    # We return None and note this limitation.
    logger.debug(f"FINRA short interest for {ticker}: not available via free API")
    return None


def compute_form4_stats(filings: List[Form4Filing]) -> Dict[str, Any]:
    """Compute aggregate statistics from a list of Form 4 filings."""
    open_market = [f for f in filings
                   if f.transaction_type in ("P", "S") and not f.is_plan
                   and f.transaction_code not in ("A", "D", "G")]
    buys = [f for f in open_market if f.transaction_type == "P"]
    sells = [f for f in open_market if f.transaction_type == "S"]

    return {
        "total_filings": len(filings),
        "open_market_transactions": len(open_market),
        "n_buys": len(buys),
        "n_sells": len(sells),
        "total_buy_value": sum(f.total_value for f in buys),
        "total_sell_value": sum(f.total_value for f in sells),
        "net_value": sum(f.total_value for f in buys) - sum(f.total_value for f in sells),
        "unique_buyers": len({f.owner_name for f in buys}),
        "unique_sellers": len({f.owner_name for f in sells}),
        "ceo_transactions": [f for f in open_market if f.is_ceo],
        "cfo_transactions": [f for f in open_market if f.is_cfo],
        "date_range": (
            min((f.transaction_date for f in open_market if f.transaction_date), default=""),
            max((f.transaction_date for f in open_market if f.transaction_date), default=""),
        ),
    }


def summarize_form4_filings(filings: List[Form4Filing], ticker: str) -> str:
    """Plain-text summary of Form 4 filings for a ticker."""
    stats = compute_form4_stats(filings)
    lines = [
        f"Form 4 Summary: {ticker}",
        f"  Total filings: {stats['total_filings']}",
        f"  Open-market transactions: {stats['open_market_transactions']}",
        f"  Buys: {stats['n_buys']} (${stats['total_buy_value']:,.0f})",
        f"  Sells: {stats['n_sells']} (${stats['total_sell_value']:,.0f})",
        f"  Net purchase value: ${stats['net_value']:,.0f}",
        f"  Unique buyers: {stats['unique_buyers']}",
        f"  Unique sellers: {stats['unique_sellers']}",
        f"  CEO transactions: {len(stats['ceo_transactions'])}",
        f"  CFO transactions: {len(stats['cfo_transactions'])}",
    ]
    if stats["date_range"][0]:
        lines.append(f"  Date range: {stats['date_range'][0]} → {stats['date_range'][1]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SENTINEL Ownership Screener v3 — Live Demo")
    print("=" * 70)

    DEMO_TICKERS = ["AAPL", "MSFT", "NVDA"]
    SP500_SAMPLE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "BAC", "GS", "V", "MA", "WFC", "BX", "SPGI",
        "XOM", "CVX", "LLY", "JNJ", "UNH", "PFE", "ABBV", "MRK",
        "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD",
    ]

    # ---- Demo 1: Form 4 filings ----
    print("\n--- Fetching Form 4 filings ---")
    parser = Form4Parser()
    insider_engine = InsiderSignalEngine()

    for ticker in DEMO_TICKERS:
        print(f"\n{ticker}:")
        try:
            filings = parser.fetch_form4_filings(ticker, days=90)
            print(f"  Filings retrieved: {len(filings)}")
            if filings:
                print(summarize_form4_filings(filings, ticker))
            time.sleep(1.0)
        except Exception as e:
            print(f"  Error: {e}")

    # ---- Demo 2: Insider scores ----
    print("\n--- Computing insider scores ---")
    for ticker in DEMO_TICKERS:
        try:
            score = insider_engine.compute_insider_score(ticker, days=90)
            print(f"  {ticker}: score={score.score:.1f}, buyers={score.n_unique_buyers}, "
                  f"sellers={score.n_unique_sellers}, cluster_buy={score.cluster_buy}, "
                  f"ceo_buy={score.ceo_bought}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {ticker}: Error — {e}")

    # ---- Demo 3: Cluster buy screen ----
    print(f"\n--- Running cluster_buy screen on {len(SP500_SAMPLE)}-ticker universe ---")
    screener = OwnershipScreener()
    try:
        cluster_results = screener.run_preset("cluster_buy", SP500_SAMPLE)
        if _PANDAS and not cluster_results.empty:
            print(f"  Cluster buys found: {len(cluster_results)} tickers")
            print(cluster_results[["ticker", "score", "n_buyers", "ceo_bought", "flags"]].head(10).to_string())
        else:
            print("  No cluster buys detected in sample (or no Form 4 data available)")
    except Exception as e:
        print(f"  Screen error: {e}")

    # ---- Demo 4: CEO buy screen ----
    print("\n--- Running ceo_buy screen ---")
    try:
        ceo_results = screener.run_preset("ceo_buy", SP500_SAMPLE[:15])
        if _PANDAS and not ceo_results.empty:
            print(f"  CEO buys: {len(ceo_results)} tickers")
            if "ticker" in ceo_results.columns:
                print("  Tickers:", ceo_results["ticker"].tolist())
        else:
            print("  No CEO buys found in sample")
    except Exception as e:
        print(f"  CEO screen error: {e}")

    # ---- Demo 5: Monitor setup ----
    print("\n--- Monitor watchlist setup ---")
    monitor = OwnershipChangeMonitor()
    for t in DEMO_TICKERS:
        monitor.subscribe_ticker(t)
    print(f"  Watchlist: {DEMO_TICKERS}")
    report = monitor.generate_daily_alert_report()
    print(report[:300])

    # ---- Demo 6: Full ownership picture for AAPL ----
    print("\n--- Full ownership picture for AAPL ---")
    engine = OwnershipScreenerEngine()
    try:
        profile = engine.get_full_ownership_picture("AAPL")
        print(f"  Composite score:  {profile.composite_ownership_score:.1f}/10")
        print(f"  Insider score:    {profile.insider_score.score:.1f}/10" if profile.insider_score else "  Insider score: N/A")
        print(f"  Smart money:      {profile.smart_money_signal:.2f}")
        print(f"  Institutional %:  {profile.institutional_pct:.1f}%")
        print(f"  Flags:            {', '.join(profile.flags) or 'None'}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nOwnership Screener v3 — demo complete.")
    print("Run OwnershipScreenerEngine().run_all_screens(universe) for full pipeline.")
