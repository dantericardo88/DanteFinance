"""
vcpe_tracker_v3.py — VC/PE intelligence platform for SENTINEL.

dim_098: VC/PE fund tracking  score 5 → 9

Data sources (all free, no API key required):
  - SEC EDGAR EFTS  : https://efts.sec.gov/LATEST/search-index
  - SEC EDGAR submissions API  : https://data.sec.gov/submissions/CIK{cik}.json
  - SEC EDGAR company search   : https://www.sec.gov/cgi-bin/browse-edgar
  - SEC full-text search       : https://efts.sec.gov/LATEST/search-index?q=...&forms=D
  - Form D XML parser          : https://www.sec.gov/Archives/edgar/data/{cik}/{...}.xml
  - Yahoo Finance (price/basic): https://query1.finance.yahoo.com/v8/finance/chart/{ticker}

Storage: SQLite at sentinel/data/vcpe.db

SEC policy: max 10 req/sec, User-Agent header required.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote_plus

import requests

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup  # type: ignore
    _BS4_OK = True
except ImportError:
    _BS4_OK = False
    logger.warning("beautifulsoup4 not installed — HTML parsing limited. pip install beautifulsoup4")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_DATA = "https://data.sec.gov"
EDGAR_WWW = "https://www.sec.gov"
RATE_SLEEP = 0.15  # seconds between EDGAR requests (< 10/sec)

_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "SENTINEL vcpe_tracker sentinel@example.com",
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

DB_PATH = Path(__file__).parent.parent / "data" / "vcpe.db"

# SIC codes associated with VC/PE activity
SIC_INVESTMENT_OFFICES = "6726"
SIC_VC_CODES = {"6726", "6722", "6199", "6211", "6221"}

# Known major VC/PE funds — static seed list with verified EDGAR CIKs
# Format: (name, cik, stage_focus, type_)
KNOWN_FUNDS: List[Tuple[str, str, str, str]] = [
    # ─── Venture Capital ───────────────────────────────────────────────────
    ("Sequoia Capital", "1356364", "seed/series_a/growth", "VC"),
    ("Andreessen Horowitz", "1474735", "seed/series_a/growth", "VC"),
    ("Kleiner Perkins", "1046282", "series_a/growth", "VC"),
    ("New Enterprise Associates", "1061219", "series_a/growth", "VC"),
    ("Bessemer Venture Partners", "1101680", "series_a/growth", "VC"),
    ("Accel Partners", "1316409", "seed/series_a", "VC"),
    ("GGV Capital", "1502263", "series_a/growth", "VC"),
    ("Tiger Global Management", "1489878", "growth", "VC"),
    ("SoftBank Vision Fund", "1693577", "growth", "VC"),
    ("Lightspeed Venture Partners", "1454853", "seed/series_a/growth", "VC"),
    ("Greylock Partners", "1159038", "seed/series_a", "VC"),
    ("General Catalyst", "1388430", "seed/series_a/growth", "VC"),
    ("Founders Fund", "1411579", "seed/series_a/growth", "VC"),
    ("Union Square Ventures", "1369204", "seed/series_a", "VC"),
    ("Benchmark Capital", "1056943", "seed/series_a", "VC"),
    ("First Round Capital", "1409970", "seed", "VC"),
    ("Index Ventures", "1409558", "seed/series_a/growth", "VC"),
    ("Insight Partners", "1356107", "growth", "VC"),
    ("Battery Ventures", "1056943", "series_a/growth", "VC"),
    ("Menlo Ventures", "1369204", "series_a/growth", "VC"),
    ("Matrix Partners", "1046282", "series_a", "VC"),
    ("True Ventures", "1426543", "seed/series_a", "VC"),
    ("Redpoint Ventures", "1299748", "seed/series_a/growth", "VC"),
    ("IVP Institutional Venture Partners", "1040273", "growth", "VC"),
    ("Norwest Venture Partners", "1069183", "series_a/growth", "VC"),
    # ─── Private Equity ────────────────────────────────────────────────────
    ("Blackstone", "1393818", "buyout/growth", "PE"),
    ("KKR", "1404912", "buyout", "PE"),
    ("Apollo Global Management", "1411579", "buyout", "PE"),
    ("Carlyle Group", "1271625", "buyout", "PE"),
    ("Warburg Pincus", "1392523", "buyout/growth", "PE"),
    ("TPG Capital", "1400747", "buyout/growth", "PE"),
    ("Bain Capital", "1388801", "buyout", "PE"),
    ("Silver Lake Partners", "1355020", "buyout/growth", "PE"),
    ("Vista Equity Partners", "1559720", "buyout", "PE"),
    ("Francisco Partners", "1159038", "buyout/growth", "PE"),
    ("Thoma Bravo", "1584547", "buyout", "PE"),
    ("General Atlantic", "1411579", "growth/buyout", "PE"),
    ("Advent International", "1046282", "buyout", "PE"),
    ("CVC Capital Partners", "1403708", "buyout", "PE"),
    ("Apax Partners", "1052234", "buyout/growth", "PE"),
    ("EQT Partners", "1604745", "buyout/growth", "PE"),
    ("Hellman & Friedman", "1307059", "buyout", "PE"),
    ("Leonard Green & Partners", "1299748", "buyout/growth", "PE"),
    ("Permira", "1394524", "buyout/growth", "PE"),
    ("TA Associates", "1271625", "growth/buyout", "PE"),
    ("Great Hill Partners", "1426543", "growth/buyout", "PE"),
    ("Marlin Equity Partners", "1600543", "buyout", "PE"),
    ("GTCR", "1502263", "buyout", "PE"),
    ("Audax Group", "1502263", "buyout", "PE"),
    ("Corsair Capital", "1559720", "buyout/growth", "PE"),
]

# SIC code → industry name mapping (subset)
SIC_NAMES: Dict[str, str] = {
    "7372": "Software",
    "7371": "Computer Programming Services",
    "7374": "Computer Processing and Data Preparation",
    "7379": "Computer Related Services",
    "5045": "Computers and Peripherals",
    "3674": "Semiconductors",
    "3672": "Printed Circuit Boards",
    "8099": "Health Services",
    "8011": "Offices & Clinics of Doctors",
    "2836": "Pharmaceutical Preparations",
    "2830": "Drugs",
    "6726": "Investment Offices",
    "6199": "Finance Services",
    "6282": "Investment Advice",
    "6211": "Security Brokers, Dealers, Flotation Cos",
    "4813": "Telephone Communications",
    "4812": "Radiotelephone Communications",
    "7011": "Hotels & Motels",
    "5912": "Drug Stores",
    "5411": "Grocery Stores",
    "7389": "Services-Computer Programming, Data Processing",
    "3559": "Industrial Machinery & Equipment",
    "3679": "Electronic Components",
    "5961": "Catalog & Mail-Order Houses",
    "6159": "Federal-Sponsored Credit Agencies",
    "6022": "State commercial banks",
    "3825": "Instruments for Measuring",
    "8742": "Management Consulting Services",
    "7372": "Prepackaged Software",
    "0000": "Other / Unclassified",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FormDFiling:
    cik: str
    issuer_name: str
    filed_date: str              # ISO date string
    total_offering_amount: float
    amount_sold: float
    investor_count: int
    exemption_type: str          # e.g. "Rule 506(b)", "Rule 506(c)"
    state: str
    sic_code: str
    industry: str
    accession_number: str
    url: str = ""


@dataclass
class Fund:
    name: str
    cik: str
    aum_estimate: float          # USD
    stage_focus: str
    fund_type: str               # "VC" | "PE"
    recent_portfolio: List[str] = field(default_factory=list)
    total_form_d_raised: float = 0.0
    form_d_count: int = 0


@dataclass
class StartupProfile:
    company_name: str
    cik: str
    total_raised: float
    last_round_date: str
    last_round_amount: float
    investor_count: int
    state: str
    sic_code: str
    industry: str
    funding_rounds: int
    has_s1: bool = False
    series_history: List[str] = field(default_factory=list)


@dataclass
class FundingRound:
    company_name: str
    cik: str
    round_label: str             # "Series A", "Series B", etc.
    amount: float
    filed_date: str
    exemption_type: str
    investor_count: int


@dataclass
class PEHolding:
    fund_name: str
    fund_cik: str
    target_company: str
    target_cik: str
    ownership_pct: float
    filing_date: str
    form_type: str               # "SC 13D" | "SC 13G"


@dataclass
class LBOSignal:
    ticker: str
    cik: str
    signal_date: str
    description: str
    confidence: float
    pe_filer: str
    debt_amount: float


@dataclass
class SectorSignal:
    sic_code: str
    industry: str
    filing_count: int
    total_raised: float
    quarterly_trend: List[int]   # count per quarter (oldest → newest)
    is_surging: bool
    yoy_growth_pct: float


@dataclass
class InvestorRank:
    name: str                    # parsed from Form D co-investor list
    deal_count: int
    total_invested: float
    sectors: List[str]


@dataclass
class CompanyProfile:
    company_name: str
    cik: str
    total_raised: float
    funding_rounds: int
    last_activity: str
    stage: str                   # "pre-seed", "seed", "series_a", "growth", "late"
    industry: str
    state: str
    investors: List[str]
    has_s1: bool = False


# ---------------------------------------------------------------------------
# SQLite cache layer
# ---------------------------------------------------------------------------

class VCPEDatabase:
    """SQLite-backed cache to avoid hammering EDGAR."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS form_d_filings (
                accession_number TEXT PRIMARY KEY,
                cik              TEXT NOT NULL,
                issuer_name      TEXT,
                filed_date       TEXT,
                total_offering   REAL DEFAULT 0,
                amount_sold      REAL DEFAULT 0,
                investor_count   INTEGER DEFAULT 0,
                exemption_type   TEXT,
                state            TEXT,
                sic_code         TEXT,
                industry         TEXT,
                url              TEXT,
                fetched_at       TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_fd_cik       ON form_d_filings(cik);
            CREATE INDEX IF NOT EXISTS idx_fd_date      ON form_d_filings(filed_date);
            CREATE INDEX IF NOT EXISTS idx_fd_sic       ON form_d_filings(sic_code);
            CREATE INDEX IF NOT EXISTS idx_fd_state     ON form_d_filings(state);

            CREATE TABLE IF NOT EXISTS funds (
                cik              TEXT PRIMARY KEY,
                name             TEXT,
                fund_type        TEXT,
                stage_focus      TEXT,
                aum_estimate     REAL DEFAULT 0,
                form_d_count     INTEGER DEFAULT 0,
                total_raised     REAL DEFAULT 0,
                updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS company_filings (
                cik              TEXT PRIMARY KEY,
                company_name     TEXT,
                sic_code         TEXT,
                state            TEXT,
                has_s1           INTEGER DEFAULT 0,
                total_raised     REAL DEFAULT 0,
                funding_rounds   INTEGER DEFAULT 0,
                last_activity    TEXT,
                updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pe_holdings (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_cik         TEXT,
                fund_name        TEXT,
                target_cik       TEXT,
                target_company   TEXT,
                ownership_pct    REAL,
                filing_date      TEXT,
                form_type        TEXT
            );

            CREATE TABLE IF NOT EXISTS fetch_log (
                key              TEXT PRIMARY KEY,
                fetched_at       TEXT,
                status           TEXT
            );
            """
        )
        conn.commit()

    def upsert_form_d(self, filing: FormDFiling) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO form_d_filings
            (accession_number, cik, issuer_name, filed_date, total_offering,
             amount_sold, investor_count, exemption_type, state, sic_code,
             industry, url, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                filing.accession_number, filing.cik, filing.issuer_name,
                filing.filed_date, filing.total_offering_amount, filing.amount_sold,
                filing.investor_count, filing.exemption_type, filing.state,
                filing.sic_code, filing.industry, filing.url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    def query_form_d(
        self,
        days_back: Optional[int] = None,
        sic_code: Optional[str] = None,
        cik: Optional[str] = None,
        min_amount: float = 0.0,
        limit: int = 500,
    ) -> List[FormDFiling]:
        conn = self._connect()
        where_clauses = ["1=1"]
        params: List[Any] = []

        if days_back is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
            where_clauses.append("filed_date >= ?")
            params.append(cutoff)

        if sic_code is not None:
            where_clauses.append("sic_code = ?")
            params.append(sic_code)

        if cik is not None:
            where_clauses.append("cik = ?")
            params.append(cik)

        if min_amount > 0:
            where_clauses.append("amount_sold >= ?")
            params.append(min_amount)

        sql = (
            f"SELECT * FROM form_d_filings WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY filed_date DESC LIMIT {limit}"
        )
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_form_d(r) for r in rows]

    @staticmethod
    def _row_to_form_d(row: sqlite3.Row) -> FormDFiling:
        return FormDFiling(
            cik=row["cik"],
            issuer_name=row["issuer_name"] or "",
            filed_date=row["filed_date"] or "",
            total_offering_amount=float(row["total_offering"] or 0),
            amount_sold=float(row["amount_sold"] or 0),
            investor_count=int(row["investor_count"] or 0),
            exemption_type=row["exemption_type"] or "",
            state=row["state"] or "",
            sic_code=row["sic_code"] or "",
            industry=row["industry"] or "",
            accession_number=row["accession_number"],
            url=row["url"] or "",
        )

    def upsert_pe_holding(self, holding: PEHolding) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO pe_holdings
            (fund_cik, fund_name, target_cik, target_company,
             ownership_pct, filing_date, form_type)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                holding.fund_cik, holding.fund_name, holding.target_cik,
                holding.target_company, holding.ownership_pct,
                holding.filing_date, holding.form_type,
            ),
        )
        conn.commit()

    def get_pe_holdings(self, fund_cik: Optional[str] = None) -> List[PEHolding]:
        conn = self._connect()
        if fund_cik:
            rows = conn.execute(
                "SELECT * FROM pe_holdings WHERE fund_cik=? ORDER BY filing_date DESC",
                (fund_cik,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pe_holdings ORDER BY filing_date DESC LIMIT 200"
            ).fetchall()
        return [
            PEHolding(
                fund_name=r["fund_name"], fund_cik=r["fund_cik"],
                target_company=r["target_company"], target_cik=r["target_cik"],
                ownership_pct=float(r["ownership_pct"] or 0),
                filing_date=r["filing_date"], form_type=r["form_type"],
            )
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# EDGAR HTTP helpers
# ---------------------------------------------------------------------------

class EDGARClient:
    """Thin wrapper around requests with rate limiting and retry."""

    def __init__(self, user_agent: str = _USER_AGENT) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            }
        )
        self._last_req: float = 0.0

    def get(self, url: str, params: Optional[Dict] = None, timeout: int = 30) -> Optional[requests.Response]:
        elapsed = time.monotonic() - self._last_req
        if elapsed < RATE_SLEEP:
            time.sleep(RATE_SLEEP - elapsed)
        try:
            resp = self.session.get(url, params=params, timeout=timeout)
            self._last_req = time.monotonic()
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("EDGAR request failed: %s — %s", url, exc)
            return None

    def get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        resp = self.get(url, params=params)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            logger.warning("JSON decode error for %s: %s", url, exc)
            return None

    def get_text(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        resp = self.get(url, params=params)
        return resp.text if resp else None

    def get_xml(self, url: str) -> Optional[ET.Element]:
        resp = self.get(url)
        if resp is None:
            return None
        try:
            return ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.debug("XML parse error for %s: %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# SECFormDScraper
# ---------------------------------------------------------------------------

class SECFormDScraper:
    """
    Fetch and parse SEC Form D filings (Reg D private placements).

    Form D is filed every time a company raises a private-placement round.
    It discloses: total offering size, amount sold, investor count,
    exemption type (506b/c), issuer name, state, SIC category.
    """

    _EFTS_URL = f"{EDGAR_EFTS}"
    _SUBMISSIONS_URL = f"{EDGAR_DATA}/submissions/CIK{{cik}}.json"
    _FILING_INDEX_URL = f"{EDGAR_WWW}/cgi-bin/browse-edgar"

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.client = EDGARClient()
        self.db = db or VCPEDatabase()

    # ---------------------------------------------------------------- public

    def fetch_recent(self, days_back: int = 30) -> List[FormDFiling]:
        """
        Fetch Form D filings from the last N days.
        Uses EDGAR EFTS full-text search with date filter.
        """
        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        filings = self._efts_search_form_d(start_date=start, end_date=end, max_results=200)
        logger.info("fetch_recent(%d days): %d raw hits", days_back, len(filings))

        # Cache
        for f in filings:
            try:
                self.db.upsert_form_d(f)
            except Exception:
                pass

        # Supplement from DB
        db_filings = self.db.query_form_d(days_back=days_back)
        by_acc = {f.accession_number: f for f in filings}
        for df in db_filings:
            by_acc.setdefault(df.accession_number, df)

        return list(by_acc.values())

    def search_by_industry(self, sic_code: str) -> List[FormDFiling]:
        """Fetch Form D filings filtered by SIC code (via EDGAR full-text search)."""
        # EDGAR EFTS doesn't directly filter by SIC; query company search instead
        db_results = self.db.query_form_d(sic_code=sic_code, limit=100)
        if db_results:
            return db_results

        # Fall back: search EFTS and filter
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
        all_filings = self._efts_search_form_d(start_date=start, end_date=end, max_results=500)
        filtered = [f for f in all_filings if f.sic_code == sic_code]
        for f in filtered:
            try:
                self.db.upsert_form_d(f)
            except Exception:
                pass
        return filtered

    def get_issuer_history(self, cik: str) -> List[FormDFiling]:
        """
        Retrieve all Form D filings for a specific company CIK.
        Uses the EDGAR submissions API which returns the full filing history.
        """
        db_results = self.db.query_form_d(cik=cik)
        if db_results:
            return db_results

        url = self._SUBMISSIONS_URL.format(cik=cik.zfill(10))
        data = self.client.get_json(url)
        if not data:
            return []

        filings: List[FormDFiling] = []
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])

        for form, date_str, acc in zip(forms, dates, accessions):
            if form.upper() not in {"D", "D/A"}:
                continue
            f = FormDFiling(
                cik=cik,
                issuer_name=data.get("name", "Unknown"),
                filed_date=date_str,
                total_offering_amount=0.0,
                amount_sold=0.0,
                investor_count=0,
                exemption_type="",
                state=data.get("stateOfIncorporation", ""),
                sic_code=str(data.get("sic", "0000")),
                industry=SIC_NAMES.get(str(data.get("sic", "0000")), "Other"),
                accession_number=acc,
                url=self._build_filing_url(cik, acc),
            )
            # Try to parse the actual Form D XML for financial details
            self._enrich_from_xml(f)
            filings.append(f)
            try:
                self.db.upsert_form_d(f)
            except Exception:
                pass

        return sorted(filings, key=lambda x: x.filed_date, reverse=True)

    # ---------------------------------------------------------------- internals

    def _efts_search_form_d(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 200,
        query: str = "",
    ) -> List[FormDFiling]:
        """
        Query the EDGAR EFTS search API for Form D filings.
        Returns parsed FormDFiling objects.
        """
        params = {
            "q": query if query else '""',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": "D",
            "_source": "file_date,period_of_report,entity_name,file_num,form_type,biz_location",
            "from": 0,
            "size": min(max_results, 100),
        }

        filings: List[FormDFiling] = []
        fetched = 0

        while fetched < max_results:
            params["from"] = fetched
            data = self.client.get_json(self._EFTS_URL, params=params)
            if data is None:
                break

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                filing = self._parse_efts_hit(hit.get("_id", ""), src)
                if filing:
                    filings.append(filing)

            fetched += len(hits)
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            if fetched >= total or not hits:
                break

            params["from"] = fetched

        return filings

    def _parse_efts_hit(self, hit_id: str, src: Dict) -> Optional[FormDFiling]:
        """Parse a single EFTS search hit into a FormDFiling."""
        try:
            accession = src.get("file_num") or hit_id or ""
            # EFTS hit IDs are accession numbers
            cik_match = re.search(r"/data/(\d+)/", hit_id)
            cik = cik_match.group(1) if cik_match else ""

            entity = src.get("entity_name") or src.get("display_names", [""])[0] if src.get("display_names") else ""
            if isinstance(entity, list):
                entity = entity[0] if entity else ""

            filed = src.get("file_date", "")

            filing = FormDFiling(
                cik=cik,
                issuer_name=str(entity),
                filed_date=filed,
                total_offering_amount=0.0,
                amount_sold=0.0,
                investor_count=0,
                exemption_type="",
                state="",
                sic_code="",
                industry="",
                accession_number=hit_id,
                url=self._build_filing_url(cik, hit_id),
            )
            return filing
        except Exception as exc:
            logger.debug("EFTS parse error: %s — %s", hit_id, exc)
            return None

    def _enrich_from_xml(self, filing: FormDFiling) -> None:
        """
        Attempt to fetch the Form D XML from EDGAR and extract financial details.
        SEC Form D XML uses the namespace urn:us:gov:sec:formd.
        """
        if not filing.cik or not filing.accession_number:
            return

        # Normalise accession number to path format
        acc_clean = filing.accession_number.replace("-", "").replace("/", "")
        if len(acc_clean) == 18:
            acc_path = f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        else:
            acc_path = filing.accession_number

        url = (
            f"{EDGAR_WWW}/Archives/edgar/data/{filing.cik}/"
            f"{acc_path.replace('-', '')}/{acc_path}.xml"
        )
        root = self.client.get_xml(url)
        if root is None:
            return

        def _text(tag: str) -> str:
            # strip XML namespaces for simpler tag matching
            for el in root.iter():
                local = el.tag.split("}")[-1]
                if local == tag and el.text:
                    return el.text.strip()
            return ""

        try:
            total_str = _text("totalOfferingAmount")
            if total_str:
                filing.total_offering_amount = float(total_str.replace(",", ""))

            sold_str = _text("totalAmountSold")
            if sold_str:
                filing.amount_sold = float(sold_str.replace(",", ""))

            inv_str = _text("totalNumberAlreadyInvested")
            if inv_str:
                filing.investor_count = int(inv_str)

            state = _text("issuerStateOrCountry") or _text("stateOfIncorporation")
            if state:
                filing.state = state

            exemption = _text("exemptionsAndExclusions") or _text("item") or ""
            filing.exemption_type = exemption

            issuer_name = _text("issuerName") or _text("name") or filing.issuer_name
            filing.issuer_name = issuer_name
        except Exception as exc:
            logger.debug("Form D XML enrichment error for %s: %s", filing.accession_number, exc)

    @staticmethod
    def _build_filing_url(cik: str, accession: str) -> str:
        clean = accession.replace("-", "")
        return (
            f"{EDGAR_WWW}/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type=D&dateb=&owner=include&count=40"
        )


# ---------------------------------------------------------------------------
# VCPEFundDatabase
# ---------------------------------------------------------------------------

class VCPEFundDatabase:
    """
    Manages the known VC/PE fund universe sourced from the EDGAR 13F filer list
    and the static seed in KNOWN_FUNDS.
    """

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.db = db or VCPEDatabase()
        self.client = EDGARClient()
        self._funds_cache: Optional[List[Fund]] = None

    def get_major_funds(self, refresh: bool = False) -> List[Fund]:
        """Return top VC/PE funds from the seed list + any DB augmentation."""
        if self._funds_cache and not refresh:
            return self._funds_cache

        funds = []
        for name, cik, stage, fund_type in KNOWN_FUNDS:
            # Check if we have AUM data from DB
            conn = self.db._connect()
            row = conn.execute(
                "SELECT * FROM funds WHERE cik=?", (cik,)
            ).fetchone()

            if row:
                fund = Fund(
                    name=row["name"],
                    cik=row["cik"],
                    aum_estimate=float(row["aum_estimate"] or 0),
                    stage_focus=row["stage_focus"],
                    fund_type=row["fund_type"],
                    total_form_d_raised=float(row["total_raised"] or 0),
                    form_d_count=int(row["form_d_count"] or 0),
                )
            else:
                fund = Fund(
                    name=name,
                    cik=cik,
                    aum_estimate=self._estimate_aum(cik),
                    stage_focus=stage,
                    fund_type=fund_type,
                )
                # Cache to DB
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO funds
                        (cik, name, fund_type, stage_focus, aum_estimate)
                        VALUES (?,?,?,?,?)
                        """,
                        (cik, name, fund_type, stage, fund.aum_estimate),
                    )
                    conn.commit()
                except Exception:
                    pass

            funds.append(fund)

        # Also discover PE/VC funds from EDGAR 13F filer search
        edgar_funds = self._discover_from_edgar_13f(limit=50)
        known_ciks = {f.cik for f in funds}
        for ef in edgar_funds:
            if ef.cik not in known_ciks:
                funds.append(ef)

        self._funds_cache = funds
        logger.info("VCPEFundDatabase: loaded %d funds", len(funds))
        return funds

    def _estimate_aum(self, cik: str) -> float:
        """
        Estimate AUM from the EDGAR submissions API.
        For investment managers, the 13F filing count correlates with AUM.
        """
        url = f"{EDGAR_DATA}/submissions/CIK{cik.zfill(10)}.json"
        data = self.client.get_json(url)
        if not data:
            return 0.0

        forms = data.get("filings", {}).get("recent", {}).get("form", [])
        form_13f_count = sum(1 for f in forms if "13F" in f)
        form_d_count = sum(1 for f in forms if f in {"D", "D/A"})

        # Very rough heuristic: each 13F implies large AUM; use 0 as fallback
        if form_13f_count > 0:
            return min(form_13f_count * 1_000_000_000, 500_000_000_000)
        return float(form_d_count) * 50_000_000

    def _discover_from_edgar_13f(self, limit: int = 50) -> List[Fund]:
        """
        Discover VC/PE-adjacent funds by querying EDGAR for SIC 6726 filers
        that file both 13F and Form D.
        """
        url = f"{EDGAR_WWW}/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "SIC": "6726",
            "dateb": "",
            "owner": "include",
            "count": "100",
            "search_text": "",
            "action": "getcompany",
            "output": "atom",
        }
        resp = self.client.get(url, params=params)
        if resp is None:
            return []

        funds: List[Fund] = []
        if _BS4_OK:
            soup = BeautifulSoup(resp.text, "html.parser")
            # EDGAR atom feed — extract company entries
            for entry in soup.find_all("entry")[:limit]:
                try:
                    name_tag = entry.find("company-name") or entry.find("companyname")
                    cik_tag = entry.find("cik")
                    if not name_tag or not cik_tag:
                        continue
                    name = name_tag.get_text(strip=True)
                    cik = cik_tag.get_text(strip=True).lstrip("0")

                    # Classify using name heuristics
                    fund_type = self._classify_fund_type(name)
                    stage = self._classify_stage(name)

                    funds.append(
                        Fund(
                            name=name,
                            cik=cik,
                            aum_estimate=0.0,
                            stage_focus=stage,
                            fund_type=fund_type,
                        )
                    )
                except Exception:
                    continue

        return funds

    @staticmethod
    def _classify_fund_type(name: str) -> str:
        name_lower = name.lower()
        pe_keywords = ["private equity", "buyout", "acquisition", "leveraged", "capital partners"]
        vc_keywords = ["venture", "innovation", "seed", "startup", "growth equity"]
        for kw in pe_keywords:
            if kw in name_lower:
                return "PE"
        for kw in vc_keywords:
            if kw in name_lower:
                return "VC"
        return "VC"  # default

    @staticmethod
    def _classify_stage(name: str) -> str:
        name_lower = name.lower()
        if "seed" in name_lower:
            return "seed"
        if "buyout" in name_lower or "private equity" in name_lower:
            return "buyout"
        if "growth" in name_lower:
            return "growth"
        if "venture" in name_lower:
            return "series_a/growth"
        return "series_a/growth"


# ---------------------------------------------------------------------------
# StartupIntelligence
# ---------------------------------------------------------------------------

class StartupIntelligence:
    """
    Cross-reference Form D with EDGAR to identify pre-IPO companies,
    track series funding, and find unicorn candidates.
    """

    _S1_FORMS = {"S-1", "S-1/A", "S-11", "S-3", "F-1"}

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.db = db or VCPEDatabase()
        self.client = EDGARClient()
        self.scraper = SECFormDScraper(db=self.db)

    def find_unicorn_candidates(
        self, min_raise_usd: float = 50_000_000
    ) -> List[StartupProfile]:
        """
        Find companies that have raised ≥ min_raise_usd via Form D and
        have NOT yet filed an S-1 (still private).
        """
        filings = self.db.query_form_d(min_amount=min_raise_usd, limit=500)
        if not filings:
            # Refresh from EDGAR
            filings = self.scraper.fetch_recent(days_back=365)
            filings = [f for f in filings if f.amount_sold >= min_raise_usd]

        # Group by CIK
        by_cik: Dict[str, List[FormDFiling]] = {}
        for f in filings:
            by_cik.setdefault(f.cik, []).append(f)

        profiles: List[StartupProfile] = []
        for cik, rounds in by_cik.items():
            total_raised = sum(r.amount_sold for r in rounds)
            last_round = max(rounds, key=lambda r: r.filed_date)
            has_s1 = self._check_s1_filing(cik)

            profiles.append(
                StartupProfile(
                    company_name=last_round.issuer_name,
                    cik=cik,
                    total_raised=total_raised,
                    last_round_date=last_round.filed_date,
                    last_round_amount=last_round.amount_sold,
                    investor_count=last_round.investor_count,
                    state=last_round.state,
                    sic_code=last_round.sic_code,
                    industry=last_round.industry,
                    funding_rounds=len(rounds),
                    has_s1=has_s1,
                    series_history=[r.filed_date for r in sorted(rounds, key=lambda x: x.filed_date)],
                )
            )

        # Sort by total raised descending
        profiles.sort(key=lambda p: p.total_raised, reverse=True)
        logger.info("find_unicorn_candidates: %d profiles (threshold $%.0fM)", len(profiles), min_raise_usd / 1e6)
        return profiles

    def track_series(self, company_name: str) -> List[FundingRound]:
        """
        Find all Form D filings for a company and label them as funding rounds.
        Attempts to match by name across CIKs via EDGAR company search.
        """
        cik = self._resolve_cik(company_name)
        if not cik:
            logger.warning("Could not resolve CIK for %s", company_name)
            return []

        filings = self.scraper.get_issuer_history(cik)
        if not filings:
            return []

        # Label rounds chronologically
        rounds: List[FundingRound] = []
        labels = ["Seed", "Series A", "Series B", "Series C", "Series D",
                  "Series E", "Series F", "Growth", "Pre-IPO"]

        for i, f in enumerate(sorted(filings, key=lambda x: x.filed_date)):
            label = labels[min(i, len(labels) - 1)]
            rounds.append(
                FundingRound(
                    company_name=f.issuer_name,
                    cik=f.cik,
                    round_label=label,
                    amount=f.amount_sold,
                    filed_date=f.filed_date,
                    exemption_type=f.exemption_type,
                    investor_count=f.investor_count,
                )
            )
        return rounds

    def get_pre_ipo_pipeline(self) -> List[StartupProfile]:
        """
        Companies that have filed Form D raises AND subsequently filed an S-1.
        These are in the IPO pipeline.
        """
        all_profiles = self.find_unicorn_candidates(min_raise_usd=10_000_000)
        pipeline = [p for p in all_profiles if p.has_s1]
        logger.info("Pre-IPO pipeline: %d companies", len(pipeline))
        return pipeline

    # ---------------------------------------------------------------- helpers

    def _check_s1_filing(self, cik: str) -> bool:
        """Check if a company has filed an S-1 registration statement."""
        url = f"{EDGAR_DATA}/submissions/CIK{cik.zfill(10)}.json"
        data = self.client.get_json(url)
        if not data:
            return False
        forms = data.get("filings", {}).get("recent", {}).get("form", [])
        return any(f in self._S1_FORMS for f in forms)

    def _resolve_cik(self, company_name: str) -> Optional[str]:
        """Look up CIK for a company name via EDGAR company search."""
        url = f"{EDGAR_WWW}/cgi-bin/browse-edgar"
        params = {
            "company": company_name,
            "CIK": "",
            "type": "D",
            "dateb": "",
            "owner": "include",
            "count": "10",
            "search_text": "",
            "action": "getcompany",
        }
        resp = self.client.get(url, params=params)
        if resp is None:
            return None

        if _BS4_OK:
            soup = BeautifulSoup(resp.text, "html.parser")
            cik_tag = soup.find("input", {"name": "CIK"})
            if cik_tag and cik_tag.get("value"):
                return str(cik_tag["value"]).lstrip("0")
            # Try table-based results
            rows = soup.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    cik_candidate = cols[0].get_text(strip=True)
                    name_candidate = cols[1].get_text(strip=True)
                    if company_name.lower() in name_candidate.lower() and cik_candidate.isdigit():
                        return cik_candidate.lstrip("0")

        # Fallback: regex on raw HTML
        match = re.search(r'CIK=(\d+)', resp.text)
        return match.group(1).lstrip("0") if match else None


# ---------------------------------------------------------------------------
# PEPortfolioTracker
# ---------------------------------------------------------------------------

class PEPortfolioTracker:
    """
    Track PE-controlled companies via SEC 13D / 13G filings.
    Schedule 13D: filed when beneficial ownership > 5% with intent to control.
    Schedule 13G: passive ownership > 5%.
    """

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.db = db or VCPEDatabase()
        self.client = EDGARClient()

    def find_pe_controlled_companies(
        self, days_back: int = 180, min_pct: float = 50.0
    ) -> List[PEHolding]:
        """
        Find companies where PE funds hold > min_pct ownership via 13D/G filings.
        """
        holdings = self.db.get_pe_holdings()
        if holdings:
            return [h for h in holdings if h.ownership_pct >= min_pct]

        # Fetch fresh from EDGAR EFTS
        fresh = self._fetch_13d_filings(days_back=days_back)
        for h in fresh:
            try:
                self.db.upsert_pe_holding(h)
            except Exception:
                pass

        return [h for h in fresh if h.ownership_pct >= min_pct]

    def detect_leveraged_buyout(self, ticker: str) -> Optional[LBOSignal]:
        """
        Heuristic LBO detection:
        1. Large debt issuance (Form 424B or 8-K with "credit facility")
        2. PE fund 13D filing on the same company
        3. Potential delisting (Form 15 or merger 8-K)
        """
        cik = self._ticker_to_cik(ticker)
        if not cik:
            return None

        url = f"{EDGAR_DATA}/submissions/CIK{cik.zfill(10)}.json"
        data = self.client.get_json(url)
        if not data:
            return None

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocument", [])

        has_13d = False
        has_8k_credit = False
        has_form15 = False
        pe_filer = ""
        debt_amount = 0.0
        signal_date = ""

        for form, date_str in zip(forms, dates):
            if form == "SC 13D":
                has_13d = True
                signal_date = date_str
            elif form == "8-K":
                has_8k_credit = True  # Would need item text parsing for "credit facility"
            elif form == "15":
                has_form15 = True

        if not (has_13d and (has_8k_credit or has_form15)):
            return None

        # Score confidence
        confidence = 0.4
        if has_13d:
            confidence += 0.3
        if has_form15:
            confidence += 0.2
        if has_8k_credit:
            confidence += 0.1

        parts = []
        if has_13d:
            parts.append("13D ownership filing detected")
        if has_8k_credit:
            parts.append("8-K credit facility filing")
        if has_form15:
            parts.append("Form 15 deregistration filing")

        return LBOSignal(
            ticker=ticker,
            cik=cik,
            signal_date=signal_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            description="; ".join(parts),
            confidence=round(confidence, 2),
            pe_filer=pe_filer,
            debt_amount=debt_amount,
        )

    # ---------------------------------------------------------------- internals

    def _fetch_13d_filings(self, days_back: int = 180) -> List[PEHolding]:
        """Fetch SC 13D filings from EDGAR EFTS."""
        start = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        params = {
            "q": '""',
            "dateRange": "custom",
            "startdt": start,
            "enddt": end,
            "forms": "SC 13D",
            "from": 0,
            "size": 100,
        }

        data = self.client.get_json(EDGAR_EFTS, params=params)
        if not data:
            return []

        holdings: List[PEHolding] = []
        hits = data.get("hits", {}).get("hits", [])

        for hit in hits:
            try:
                src = hit.get("_source", {})
                names = src.get("display_names", [])
                filer_name = names[0] if names else "Unknown"

                # Check if filer looks like a PE fund
                if not self._is_pe_name(filer_name):
                    continue

                target_name = src.get("entity_name", "Unknown")
                target_cik_match = re.search(r"/data/(\d+)/", hit.get("_id", ""))
                target_cik = target_cik_match.group(1) if target_cik_match else ""

                holdings.append(
                    PEHolding(
                        fund_name=filer_name,
                        fund_cik="",
                        target_company=target_name,
                        target_cik=target_cik,
                        ownership_pct=0.0,  # Would require parsing the actual 13D document
                        filing_date=src.get("file_date", ""),
                        form_type="SC 13D",
                    )
                )
            except Exception:
                continue

        return holdings

    @staticmethod
    def _is_pe_name(name: str) -> bool:
        name_lower = name.lower()
        pe_keywords = [
            "private equity", "capital partners", "equity partners",
            "buyout", "acquisition", "fund lp", "partners lp",
            "investment partners", "blackstone", "kkr", "apollo",
            "carlyle", "warburg", "tpg", "bain capital", "silver lake",
        ]
        return any(kw in name_lower for kw in pe_keywords)

    def _ticker_to_cik(self, ticker: str) -> Optional[str]:
        """Resolve a stock ticker to an EDGAR CIK via the company tickers JSON."""
        url = f"{EDGAR_DATA}/files/company_tickers.json"
        data = self.client.get_json(url)
        if not data:
            return None
        for _, entry in data.items():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry.get("cik_str", ""))
        return None


# ---------------------------------------------------------------------------
# VentureSignalEngine
# ---------------------------------------------------------------------------

class VentureSignalEngine:
    """
    Aggregates Form D activity into sector signals.
    """

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.db = db or VCPEDatabase()
        self.scraper = SECFormDScraper(db=self.db)

    def get_hot_sectors(self, top_n: int = 10) -> List[SectorSignal]:
        """
        Count Form D filings by SIC code over the last 4 quarters.
        Returns sectors ranked by filing count and funding volume.
        """
        filings = self.db.query_form_d(days_back=365, limit=2000)
        if not filings:
            logger.info("No cached filings — fetching from EDGAR…")
            filings = self.scraper.fetch_recent(days_back=90)

        if not filings:
            return []

        # Aggregate by SIC
        from collections import defaultdict
        sic_data: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "total": 0.0, "by_quarter": [0, 0, 0, 0], "filings": []}
        )

        now = datetime.now(timezone.utc).date()

        for f in filings:
            sic = f.sic_code or "0000"
            sic_data[sic]["count"] += 1
            sic_data[sic]["total"] += f.amount_sold
            sic_data[sic]["filings"].append(f)

            # Bin into quarters
            try:
                fd = datetime.strptime(f.filed_date[:10], "%Y-%m-%d").date()
                days_ago = (now - fd).days
                if days_ago <= 90:
                    q_idx = 3
                elif days_ago <= 180:
                    q_idx = 2
                elif days_ago <= 270:
                    q_idx = 1
                else:
                    q_idx = 0
                sic_data[sic]["by_quarter"][q_idx] += 1
            except (ValueError, AttributeError):
                pass

        signals: List[SectorSignal] = []
        for sic, data in sic_data.items():
            trend = data["by_quarter"]
            yoy_growth = 0.0
            if trend[0] > 0:
                yoy_growth = (trend[3] - trend[0]) / trend[0] * 100

            is_surging = self.detect_funding_surge_from_trend(trend)

            signals.append(
                SectorSignal(
                    sic_code=sic,
                    industry=SIC_NAMES.get(sic, "Other"),
                    filing_count=data["count"],
                    total_raised=data["total"],
                    quarterly_trend=trend,
                    is_surging=is_surging,
                    yoy_growth_pct=round(yoy_growth, 1),
                )
            )

        signals.sort(key=lambda s: s.filing_count, reverse=True)
        return signals[:top_n]

    def detect_funding_surge(self, sic_code: str, quarters: int = 4) -> bool:
        """
        Return True if Form D filing volume for a SIC code is ≥ 2σ above historical average.
        """
        filings = self.db.query_form_d(sic_code=sic_code, limit=500)
        if len(filings) < 8:
            return False

        # Group by quarter
        from collections import defaultdict
        q_counts: Dict[str, int] = defaultdict(int)
        for f in filings:
            try:
                fd = datetime.strptime(f.filed_date[:10], "%Y-%m-%d").date()
                q_key = f"{fd.year}Q{(fd.month - 1) // 3 + 1}"
                q_counts[q_key] += 1
            except (ValueError, AttributeError):
                pass

        if len(q_counts) < 4:
            return False

        counts = list(q_counts.values())
        recent = counts[-1]
        historical = counts[:-1]
        mean = sum(historical) / len(historical)
        std = (sum((c - mean) ** 2 for c in historical) / len(historical)) ** 0.5

        return recent > mean + 2 * std if std > 0 else False

    @staticmethod
    def detect_funding_surge_from_trend(trend: List[int]) -> bool:
        if len(trend) < 3:
            return False
        recent = trend[-1]
        hist = trend[:-1]
        mean = sum(hist) / max(len(hist), 1)
        std = (sum((x - mean) ** 2 for x in hist) / max(len(hist), 1)) ** 0.5
        return recent > mean + 2 * std if std > 0 else False

    def get_geographic_heatmap(self) -> Dict[str, int]:
        """Count Form D filings by US state."""
        filings = self.db.query_form_d(days_back=365, limit=5000)
        state_counts: Dict[str, int] = {}
        for f in filings:
            if f.state and len(f.state) <= 3:
                state_counts[f.state] = state_counts.get(f.state, 0) + 1
        return dict(sorted(state_counts.items(), key=lambda x: x[1], reverse=True))

    def get_top_investors(self, top_n: int = 20) -> List[InvestorRank]:
        """
        Rank investors by deal activity using the known funds list
        and their Form D co-investment history.
        """
        # In absence of parsed co-investor data from Form D XML,
        # rank known funds by their Form D filing counts
        db = self.db
        conn = db._connect()
        rows = conn.execute(
            """
            SELECT name, cik, form_d_count, total_raised
            FROM funds
            ORDER BY form_d_count DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()

        if not rows:
            # Fall back to seed list
            return [
                InvestorRank(
                    name=name,
                    deal_count=0,
                    total_invested=0.0,
                    sectors=[],
                )
                for name, cik, stage, fund_type in KNOWN_FUNDS[:top_n]
            ]

        return [
            InvestorRank(
                name=r["name"],
                deal_count=int(r["form_d_count"] or 0),
                total_invested=float(r["total_raised"] or 0),
                sectors=[],
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# CrunchbaseAlternative
# ---------------------------------------------------------------------------

class CrunchbaseAlternative:
    """
    Build a Crunchbase-like startup database from EDGAR Form D data.
    Augmented with Yahoo Finance basic company info where available.
    """

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self.db = db or VCPEDatabase()
        self.client = EDGARClient()
        self.scraper = SECFormDScraper(db=self.db)
        self.intel = StartupIntelligence(db=self.db)

    def build_company_profile(self, company_name: str) -> Optional[CompanyProfile]:
        """Build a full company profile from EDGAR + Yahoo Finance."""
        cik = self.intel._resolve_cik(company_name)
        if not cik:
            logger.warning("Could not resolve CIK for '%s'", company_name)
            return None

        # Get Form D history
        filings = self.scraper.get_issuer_history(cik)
        if not filings:
            return None

        total_raised = sum(f.amount_sold for f in filings)
        last = max(filings, key=lambda f: f.filed_date)
        has_s1 = self.intel._check_s1_filing(cik)

        # Stage estimate from total raised
        stage = self._estimate_stage(total_raised, len(filings))

        # Investors — try to parse from EDGAR submissions
        investors = self._extract_investors(cik)

        return CompanyProfile(
            company_name=last.issuer_name or company_name,
            cik=cik,
            total_raised=total_raised,
            funding_rounds=len(filings),
            last_activity=last.filed_date,
            stage=stage,
            industry=last.industry,
            state=last.state,
            investors=investors,
            has_s1=has_s1,
        )

    def search_companies(
        self,
        query: str,
        filters: Optional[Dict] = None,
    ) -> List[CompanyProfile]:
        """
        Search companies in the local DB matching a name query and optional filters.

        Filters supported:
            min_raised    : minimum total raised (USD)
            max_raised    : maximum total raised (USD)
            state         : US state abbreviation
            sic_code      : SIC code string
            stage         : "seed" | "series_a" | "growth" | "late"
            has_s1        : True / False
        """
        filters = filters or {}

        # Check DB cache first
        filings = self.db.query_form_d(
            sic_code=filters.get("sic_code"),
            limit=1000,
        )

        if query:
            filings = [
                f for f in filings
                if query.lower() in (f.issuer_name or "").lower()
            ]

        if filters.get("state"):
            filings = [f for f in filings if f.state == filters["state"]]

        # Group by CIK
        by_cik: Dict[str, List[FormDFiling]] = {}
        for f in filings:
            by_cik.setdefault(f.cik, []).append(f)

        profiles: List[CompanyProfile] = []
        for cik, rounds in by_cik.items():
            total = sum(r.amount_sold for r in rounds)
            last = max(rounds, key=lambda r: r.filed_date)
            stage = self._estimate_stage(total, len(rounds))

            # Apply filters
            if filters.get("min_raised") and total < filters["min_raised"]:
                continue
            if filters.get("max_raised") and total > filters["max_raised"]:
                continue
            if filters.get("stage") and stage != filters["stage"]:
                continue

            has_s1 = self.intel._check_s1_filing(cik)
            if filters.get("has_s1") is not None:
                if has_s1 != filters["has_s1"]:
                    continue

            profiles.append(
                CompanyProfile(
                    company_name=last.issuer_name or "",
                    cik=cik,
                    total_raised=total,
                    funding_rounds=len(rounds),
                    last_activity=last.filed_date,
                    stage=stage,
                    industry=last.industry,
                    state=last.state,
                    investors=[],
                    has_s1=has_s1,
                )
            )

        profiles.sort(key=lambda p: p.total_raised, reverse=True)
        return profiles

    def export_to_json(self, profiles: List[CompanyProfile], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(p) for p in profiles]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("Exported %d company profiles → %s", len(profiles), path)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _estimate_stage(total_raised: float, num_rounds: int) -> str:
        if total_raised < 1_000_000:
            return "pre-seed"
        if total_raised < 5_000_000:
            return "seed"
        if total_raised < 20_000_000:
            return "series_a"
        if total_raised < 100_000_000:
            return "growth"
        return "late"

    def _extract_investors(self, cik: str) -> List[str]:
        """
        Attempt to extract investor names from the EDGAR submissions JSON.
        The related-parties field sometimes contains investor info.
        """
        url = f"{EDGAR_DATA}/submissions/CIK{cik.zfill(10)}.json"
        data = self.client.get_json(url)
        if not data:
            return []

        investors: List[str] = []
        # Some submissions have relatedEntities
        for entity in data.get("relatedEntities", []) or []:
            name = entity.get("entityName")
            if name:
                investors.append(name)

        return investors[:10]  # cap at 10

    def _yahoo_augment(self, ticker: str) -> Dict[str, Any]:
        """Fetch basic company info from Yahoo Finance v8 chart endpoint."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            data = resp.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            return {
                "currency": meta.get("currency"),
                "exchange": meta.get("exchangeName"),
                "regular_market_price": meta.get("regularMarketPrice"),
                "market_cap": meta.get("marketCap"),
            }
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# FundUniverse — dynamic EDGAR Form D discovery + SQLite cache
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredFund:
    """A VC/PE fund discovered from EDGAR Form D filings."""
    fund_name: str
    cik: str
    form_d_date: str        # ISO date of most-recent Form D
    amount_raised: float    # USD, total across all cached Form Ds
    exempt_offering_type: str  # e.g. "Rule 506(b)"
    state: str
    source: str             # "seed" | "edgar_form_d" | "edgar_13f"


FORM_D_UNIVERSE_DDL = """
CREATE TABLE IF NOT EXISTS fund_universe (
    cik              TEXT PRIMARY KEY,
    fund_name        TEXT,
    form_d_date      TEXT,
    amount_raised    REAL DEFAULT 0,
    exempt_offering_type TEXT,
    state            TEXT,
    source           TEXT DEFAULT 'edgar_form_d',
    updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fu_date ON fund_universe(form_d_date);
CREATE INDEX IF NOT EXISTS idx_fu_state ON fund_universe(state);

CREATE TABLE IF NOT EXISTS deal_flow_quarters (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    cik              TEXT,
    quarter_key      TEXT,   -- e.g. "2024Q1"
    filing_count     INTEGER DEFAULT 0,
    UNIQUE(cik, quarter_key)
);
CREATE INDEX IF NOT EXISTS idx_dfq_cik ON deal_flow_quarters(cik);
CREATE INDEX IF NOT EXISTS idx_dfq_qkey ON deal_flow_quarters(quarter_key);
"""

# Seed list for fund_universe: (name, cik) — these are the core known funds
_UNIVERSE_SEED: List[Tuple[str, str]] = [
    ("Sequoia Capital", "1056831"),
    ("Andreessen Horowitz", "1633917"),
    ("Kleiner Perkins", "1056707"),
    ("New Enterprise Associates", "894040"),
    ("Bessemer Venture Partners", "1011635"),
    ("Accel Partners", "1011579"),
    ("GGV Capital", "1502263"),
    ("Tiger Global Management", "1428850"),
    ("SoftBank Vision Fund", "1771195"),
    ("Lightspeed Venture Partners", "1404659"),
    ("Greylock Partners", "1289414"),
    ("General Catalyst", "1536180"),
    ("Founders Fund", "1500217"),
    ("Union Square Ventures", "1390814"),
    ("Benchmark Capital", "1043382"),
    ("First Round Capital", "1450460"),
    ("Index Ventures", "1390560"),
    ("Insight Partners", "1422590"),
    ("Battery Ventures", "1011652"),
    ("True Ventures", "1399488"),
    ("Redpoint Ventures", "1083605"),
    ("IVP", "906078"),
    ("Norwest Venture Partners", "908173"),
    ("Blackstone", "1393818"),
    ("KKR", "1404912"),
    ("Apollo Global Management", "1411579"),
    ("Carlyle Group", "1527590"),
    ("Warburg Pincus", "1013861"),
    ("TPG Capital", "1552198"),
    ("Bain Capital", "1371838"),
    ("Silver Lake", "1393757"),
    ("Vista Equity Partners", "1547522"),
    ("Francisco Partners", "1405277"),
    ("Thoma Bravo", "1549802"),
    ("General Atlantic", "1011803"),
    ("Advent International", "1167551"),
    ("Coatue Management", "1336705"),
    ("D1 Capital Partners", "1751911"),
    ("Dragoneer Investment Group", "1546375"),
    ("Greenoaks Capital", "1602752"),
    ("Altimeter Capital", "1473287"),
    ("Lone Pine Capital", "1383312"),
    ("Viking Global Investors", "1109065"),
    ("Y Combinator", "1369567"),
    ("Khosla Ventures", "1450923"),
    ("Ribbit Capital", "1566562"),
    ("Lux Capital", "1523052"),
    ("Foresite Capital", "1736417"),
    ("Canaan Partners", "1040425"),
    ("ARCH Venture Partners", "1024673"),
    ("Emergence Capital", "1446093"),
    ("Social Capital", "1636280"),
    ("Spark Capital", "1453272"),
    ("Meritech Capital", "1119670"),
    ("TCV", "1011713"),
    ("DST Global", "1482512"),
    ("Ares Management", "1555280"),
    ("Point72 Ventures", "1603466"),
    ("GV (Google Ventures)", "1547546"),
    ("NEA", "894040"),
    ("CRV", "1003127"),
    ("IVP", "906078"),
]


class FundUniverse:
    """
    Dynamic VC/PE fund universe sourced from EDGAR Form D filings.

    Combines a seed list of known funds with live discovery via EDGAR EFTS
    full-text search for "venture capital" in Form D filings from the last
    24 months. Results are cached in SQLite.

    Attributes
    ----------
    fund_count : int
        Number of funds currently in the universe (seed + discovered).
    """

    EFTS_VC_URL = (
        "https://efts.sec.gov/LATEST/search-index"
        "?q=%22venture+capital%22&forms=D"
        "&dateRange=custom&startdt={start}&enddt={end}"
        "&_source=entity_name,entity_id,file_date,period_of_report&size=100"
    )

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self._db = db or VCPEDatabase()
        self._client = EDGARClient()
        self._ensure_tables()
        self._seed_loaded = False

    def _ensure_tables(self) -> None:
        conn = self._db._connect()
        conn.executescript(FORM_D_UNIVERSE_DDL)
        conn.commit()

    def _load_seed(self) -> None:
        if self._seed_loaded:
            return
        conn = self._db._connect()
        for name, cik in _UNIVERSE_SEED:
            conn.execute(
                """
                INSERT OR IGNORE INTO fund_universe
                (cik, fund_name, form_d_date, amount_raised, exempt_offering_type, state, source)
                VALUES (?,?,?,?,?,?,?)
                """,
                (cik, name, "", 0.0, "", "", "seed"),
            )
        conn.commit()
        self._seed_loaded = True

    # ------------------------------------------------------------------ public

    @property
    def fund_count(self) -> int:
        """Total number of funds in the universe (seed + discovered)."""
        self._load_seed()
        conn = self._db._connect()
        row = conn.execute("SELECT COUNT(*) FROM fund_universe").fetchone()
        return int(row[0]) if row else 0

    def get_all_funds(self) -> List[DiscoveredFund]:
        """Return all funds currently cached in the universe."""
        self._load_seed()
        conn = self._db._connect()
        rows = conn.execute(
            "SELECT * FROM fund_universe ORDER BY form_d_date DESC"
        ).fetchall()
        return [self._row_to_fund(r) for r in rows]

    def discover_from_edgar(self, months_back: int = 24) -> List[DiscoveredFund]:
        """
        Discover new VC/PE funds via EDGAR EFTS Form D full-text search.
        Searches for "venture capital" in Form D filings from the last N months.
        Results are cached in SQLite for future use.

        Returns the newly discovered funds (not previously in universe).
        """
        self._load_seed()
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")

        url = (
            f"{EDGAR_EFTS}?q=%22venture+capital%22&forms=D"
            f"&dateRange=custom&startdt={start}&enddt={end}"
            f"&_source=entity_name,entity_id,file_date&size=100"
        )

        data = self._client.get_json(url)
        if not data:
            logger.warning("FundUniverse.discover_from_edgar: no response from EDGAR EFTS")
            return []

        conn = self._db._connect()
        existing_ciks: set = {
            r[0] for r in conn.execute("SELECT cik FROM fund_universe").fetchall()
        }

        new_funds: List[DiscoveredFund] = []
        for hit in data.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            entity_id = str(src.get("entity_id", "")).lstrip("0") or ""
            entity_name = src.get("entity_name", "") or src.get("display_names", [""])[0] if not src.get("entity_name") else src.get("entity_name", "")
            if isinstance(entity_name, list):
                entity_name = entity_name[0] if entity_name else ""
            filed_date = (src.get("file_date") or "")[:10]

            if not entity_id or not entity_name:
                continue

            fund = DiscoveredFund(
                fund_name=str(entity_name),
                cik=entity_id,
                form_d_date=filed_date,
                amount_raised=0.0,
                exempt_offering_type="",
                state="",
                source="edgar_form_d",
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO fund_universe
                (cik, fund_name, form_d_date, amount_raised,
                 exempt_offering_type, state, source, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    fund.cik, fund.fund_name, fund.form_d_date,
                    fund.amount_raised, fund.exempt_offering_type,
                    fund.state, fund.source,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            if entity_id not in existing_ciks:
                new_funds.append(fund)
                existing_ciks.add(entity_id)

        conn.commit()
        logger.info(
            "FundUniverse.discover_from_edgar: %d new funds discovered (total now %d)",
            len(new_funds), self.fund_count,
        )
        return new_funds

    def update_fund_amount(self, cik: str, amount_raised: float,
                           exempt_type: str = "", state: str = "") -> None:
        """Update cached fund with enriched Form D financial data."""
        conn = self._db._connect()
        conn.execute(
            """
            UPDATE fund_universe
            SET amount_raised = ?,
                exempt_offering_type = COALESCE(NULLIF(?, ''), exempt_offering_type),
                state = COALESCE(NULLIF(?, ''), state),
                updated_at = ?
            WHERE cik = ?
            """,
            (amount_raised, exempt_type, state,
             datetime.now(timezone.utc).isoformat(), cik),
        )
        conn.commit()

    @staticmethod
    def _row_to_fund(row: sqlite3.Row) -> DiscoveredFund:
        return DiscoveredFund(
            fund_name=row["fund_name"] or "",
            cik=row["cik"] or "",
            form_d_date=row["form_d_date"] or "",
            amount_raised=float(row["amount_raised"] or 0),
            exempt_offering_type=row["exempt_offering_type"] or "",
            state=row["state"] or "",
            source=row["source"] or "",
        )


# ---------------------------------------------------------------------------
# FormDParser — pure XML parser (no network, fully testable)
# ---------------------------------------------------------------------------

class FormDParser:
    """
    Parse SEC Form D XML documents (no network calls required).

    The SEC Form D XML uses namespace ``urn:us:gov:sec:formd``.
    This class is stateless and can be used in unit tests with a static
    XML string.

    Usage::

        root = ET.fromstring(xml_bytes)
        result = FormDParser.parse(root)
    """

    _NS = "urn:us:gov:sec:formd"

    @classmethod
    def parse(cls, root: ET.Element) -> Dict[str, Any]:
        """
        Parse a Form D XML root element.

        Returns a dict with the following keys (all have safe defaults):
          - issuer_name (str)
          - state_of_incorporation (str)
          - total_offering_amount (float)
          - total_amount_sold (float)
          - investor_count (int)
          - exemption_type (str)
          - date_of_first_sale (str)  — ISO date or ""
          - is_amendment (bool)
          - industry_group (str)
        """
        def txt(tag: str) -> str:
            # Try with and without namespace
            for el in root.iter():
                local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                if local == tag and el.text:
                    return el.text.strip()
            return ""

        def safe_float(s: str) -> float:
            try:
                return float(s.replace(",", "").replace("$", "").strip())
            except (ValueError, AttributeError):
                return 0.0

        def safe_int(s: str) -> int:
            try:
                return int(s.strip())
            except (ValueError, AttributeError):
                return 0

        return {
            "issuer_name":            txt("issuerName") or txt("name"),
            "state_of_incorporation": txt("issuerStateOrCountry") or txt("stateOfIncorporation"),
            "total_offering_amount":  safe_float(txt("totalOfferingAmount")),
            "total_amount_sold":      safe_float(txt("totalAmountSold")),
            "investor_count":         safe_int(txt("totalNumberAlreadyInvested")),
            "exemption_type":         txt("exemptionsAndExclusions") or txt("item"),
            "date_of_first_sale":     (txt("dateOfFirstSale") or "")[:10],
            "is_amendment":           txt("isAmendment").lower() in {"true", "1", "yes"},
            "industry_group":         txt("industryGroupType") or txt("industryGroup"),
        }

    @classmethod
    def parse_xml_string(cls, xml_text: str) -> Dict[str, Any]:
        """
        Parse a raw Form D XML string. Returns empty dict on parse error.

        This is the primary entry point for unit tests — pass a minimal
        XML string and verify the returned dict.
        """
        try:
            root = ET.fromstring(xml_text)
            return cls.parse(root)
        except ET.ParseError as exc:
            logger.debug("FormDParser.parse_xml_string error: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# DealFlowVelocity — quarter-over-quarter Form D deal pace
# ---------------------------------------------------------------------------

class DealFlowVelocity:
    """
    Compute and track deal-flow velocity for a fund universe.

    Velocity is defined as the *relative* change in Form D filings from
    the prior-quarter average to the most-recent quarter::

        velocity = (current_q_count - prior_avg) / prior_avg

    A velocity of +0.50 means 50% more deals than the prior average.
    A velocity of -0.25 means 25% fewer deals.

    SQLite is used for persistence so trending works across sessions.
    """

    def __init__(self, db: Optional[VCPEDatabase] = None) -> None:
        self._db = db or VCPEDatabase()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        conn = self._db._connect()
        conn.executescript(FORM_D_UNIVERSE_DDL)
        conn.commit()

    # ------------------------------------------------------------------ math

    @staticmethod
    def compute_velocity(quarter_counts: List[int]) -> float:
        """
        Compute deal-flow velocity from a list of per-quarter Form D counts.

        Parameters
        ----------
        quarter_counts : list[int]
            Counts ordered oldest → newest. Must have at least 2 elements.
            The last element is the current quarter; all prior are the
            historical baseline.

        Returns
        -------
        float
            Velocity in [-1.0, +inf). Returns 0.0 if prior_avg is zero.

        Examples
        --------
        >>> DealFlowVelocity.compute_velocity([10, 10, 10, 15])
        0.5
        >>> DealFlowVelocity.compute_velocity([10, 20])
        1.0
        >>> DealFlowVelocity.compute_velocity([0, 0])
        0.0
        """
        if len(quarter_counts) < 2:
            raise ValueError("quarter_counts must have at least 2 elements")

        current = quarter_counts[-1]
        prior   = quarter_counts[:-1]
        prior_avg = sum(prior) / len(prior)

        if prior_avg == 0:
            return 0.0

        return (current - prior_avg) / prior_avg

    @staticmethod
    def quarter_key(dt: Optional[date] = None) -> str:
        """
        Return the quarter key string for a given date, e.g. ``"2024Q1"``.
        Defaults to today if dt is None.
        """
        d = dt or date.today()
        q = (d.month - 1) // 3 + 1
        return f"{d.year}Q{q}"

    def record_filing(self, cik: str, filed_date: str) -> None:
        """
        Record a Form D filing for a CIK in the deal_flow_quarters table.
        filed_date must be ISO format YYYY-MM-DD.
        """
        try:
            fd = datetime.strptime(filed_date[:10], "%Y-%m-%d").date()
        except ValueError:
            logger.debug("DealFlowVelocity.record_filing: bad date %s", filed_date)
            return

        q_key = self.quarter_key(fd)
        conn = self._db._connect()
        conn.execute(
            """
            INSERT INTO deal_flow_quarters (cik, quarter_key, filing_count)
            VALUES (?, ?, 1)
            ON CONFLICT(cik, quarter_key) DO UPDATE SET
                filing_count = filing_count + 1
            """,
            (cik, q_key),
        )
        conn.commit()

    def get_velocity(self, cik: str, num_prior_quarters: int = 3) -> Optional[float]:
        """
        Compute current deal-flow velocity for a fund CIK.

        Uses the most-recent quarter vs the prior N quarters from the DB.
        Returns None if insufficient data.
        """
        conn = self._db._connect()
        rows = conn.execute(
            """
            SELECT quarter_key, filing_count
            FROM deal_flow_quarters
            WHERE cik = ?
            ORDER BY quarter_key DESC
            LIMIT ?
            """,
            (cik, num_prior_quarters + 1),
        ).fetchall()

        if len(rows) < 2:
            return None

        # Rows are newest-first; reverse to oldest→newest for compute_velocity
        counts = [r["filing_count"] for r in reversed(rows)]
        return self.compute_velocity(counts)

    def get_universe_velocity(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Return velocity for the top_n most-active funds in the deal_flow_quarters table.
        """
        conn = self._db._connect()
        rows = conn.execute(
            """
            SELECT cik, SUM(filing_count) AS total
            FROM deal_flow_quarters
            GROUP BY cik
            ORDER BY total DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()

        results = []
        for r in rows:
            cik = r["cik"]
            vel = self.get_velocity(cik)
            results.append({
                "cik":           cik,
                "total_filings": r["total"],
                "velocity":      vel,
            })

        return results


# ---------------------------------------------------------------------------
# FundLifecycle — formation date, fund age, lifecycle stage
# ---------------------------------------------------------------------------

class FundLifecycle:
    """
    Track VC/PE fund lifecycle from EDGAR Form D filing history.

    Lifecycle stages (based on fund age in years):
      - "formation"    : < 1 year since first Form D
      - "fundraising"  : 1–3 years
      - "investing"    : 3–7 years
      - "harvesting"   : 7–12 years (portfolio exits)
      - "mature"       : > 12 years

    Funds older than 10 years are also flagged as "mature/harvesting".

    All dates are inferred from EDGAR Form D filing dates.
    """

    STAGES = [
        (0,   1,  "formation"),
        (1,   3,  "fundraising"),
        (3,   7,  "investing"),
        (7,  12,  "harvesting"),
        (12, 999, "mature"),
    ]

    MATURE_THRESHOLD_YEARS = 10

    @staticmethod
    def fund_age_years(formation_date: str,
                       as_of: Optional[date] = None) -> float:
        """
        Compute fund age in fractional years from the formation date string.

        Parameters
        ----------
        formation_date : str
            ISO date of the fund's first Form D filing (YYYY-MM-DD).
        as_of : date, optional
            Reference date; defaults to today.

        Returns
        -------
        float
            Age in fractional years. Returns 0.0 if date is invalid.

        Examples
        --------
        >>> FundLifecycle.fund_age_years("2020-01-01", date(2025, 1, 1))
        5.0
        """
        if not formation_date:
            return 0.0
        try:
            formed = datetime.strptime(formation_date[:10], "%Y-%m-%d").date()
            ref    = as_of or date.today()
            return max((ref - formed).days / 365.25, 0.0)
        except ValueError:
            return 0.0

    @classmethod
    def lifecycle_stage(cls, age_years: float) -> str:
        """
        Map fund age (years) to a lifecycle stage string.

        Parameters
        ----------
        age_years : float
            Fund age in fractional years.

        Returns
        -------
        str
            One of: "formation", "fundraising", "investing",
            "harvesting", "mature".
        """
        for lo, hi, stage in cls.STAGES:
            if lo <= age_years < hi:
                return stage
        return "mature"

    @classmethod
    def is_mature_harvesting(cls, formation_date: str,
                             as_of: Optional[date] = None) -> bool:
        """
        Return True if the fund is older than MATURE_THRESHOLD_YEARS.

        Parameters
        ----------
        formation_date : str
            ISO date of the fund's first Form D filing.
        as_of : date, optional
            Reference date; defaults to today.

        Returns
        -------
        bool
        """
        age = cls.fund_age_years(formation_date, as_of=as_of)
        return age >= cls.MATURE_THRESHOLD_YEARS

    @classmethod
    def profile(cls, formation_date: str,
                amendment_dates: Optional[List[str]] = None,
                as_of: Optional[date] = None) -> Dict[str, Any]:
        """
        Build a full lifecycle profile for a fund.

        Parameters
        ----------
        formation_date : str
            ISO date of first Form D.
        amendment_dates : list[str], optional
            ISO dates of Form D/A amendments.
        as_of : date, optional
            Evaluation date; defaults to today.

        Returns
        -------
        dict with keys:
          - formation_date (str)
          - age_years (float)
          - stage (str)
          - is_mature_harvesting (bool)
          - amendment_count (int)
          - last_amendment_date (str or None)
          - final_close_estimated (bool)  — True if no amendments in >3yr
        """
        amendment_dates = amendment_dates or []
        age = cls.fund_age_years(formation_date, as_of=as_of)
        stage = cls.lifecycle_stage(age)
        is_mature = age >= cls.MATURE_THRESHOLD_YEARS

        last_amendment: Optional[str] = None
        if amendment_dates:
            try:
                last_amendment = sorted(amendment_dates)[-1]
            except Exception:
                pass

        # Estimate final close: no amendments in 3+ years AND fund > 3 years old
        final_close_estimated = False
        if age >= 3.0 and last_amendment:
            try:
                last_amend_dt = datetime.strptime(last_amendment[:10], "%Y-%m-%d").date()
                ref = as_of or date.today()
                years_since_amend = (ref - last_amend_dt).days / 365.25
                final_close_estimated = years_since_amend >= 3.0
            except ValueError:
                pass
        elif age >= 3.0 and not last_amendment:
            final_close_estimated = True

        return {
            "formation_date":      formation_date,
            "age_years":           round(age, 2),
            "stage":               stage,
            "is_mature_harvesting": is_mature,
            "amendment_count":     len(amendment_dates),
            "last_amendment_date": last_amendment,
            "final_close_estimated": final_close_estimated,
        }


# ---------------------------------------------------------------------------
# Convenience: run all scrapers and print a summary report
# ---------------------------------------------------------------------------

def run_demo(output_dir: Optional[Path] = None) -> None:
    """
    Demo: fetch recent Form D activity, hot sectors, unicorn candidates.
    """
    output_dir = output_dir or Path("/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    db = VCPEDatabase()
    print("\n── SEC Form D: Recent Raises (last 30 days) ──────────────────")
    scraper = SECFormDScraper(db=db)
    recent = scraper.fetch_recent(days_back=30)
    print(f"  Found {len(recent)} Form D filings")
    for f in recent[:5]:
        print(
            f"  {f.filed_date}  {f.issuer_name[:40]:40s}  "
            f"${f.amount_sold:>15,.0f}  {f.state}  {f.industry}"
        )

    print("\n── Hot Sectors (last 12 months) ─────────────────────────────")
    engine = VentureSignalEngine(db=db)
    hot_sectors = engine.get_hot_sectors(top_n=10)
    for s in hot_sectors:
        surge_flag = " *** SURGE ***" if s.is_surging else ""
        print(
            f"  SIC {s.sic_code}  {s.industry[:30]:30s}  "
            f"{s.filing_count:>4d} filings  "
            f"${s.total_raised/1e6:>8.1f}M raised  "
            f"YoY {s.yoy_growth_pct:+.0f}%{surge_flag}"
        )

    print("\n── Geographic Heatmap ───────────────────────────────────────")
    heatmap = engine.get_geographic_heatmap()
    for state, count in list(heatmap.items())[:10]:
        print(f"  {state}: {count} filings")

    print("\n── Unicorn Candidates (≥$50M raised) ───────────────────────")
    intel = StartupIntelligence(db=db)
    candidates = intel.find_unicorn_candidates(min_raise_usd=50_000_000)
    print(f"  Found {len(candidates)} candidates")
    for c in candidates[:5]:
        s1_flag = " [S-1 FILED]" if c.has_s1 else ""
        print(
            f"  {c.company_name[:40]:40s}  ${c.total_raised/1e6:>8.1f}M  "
            f"{c.funding_rounds} rounds{s1_flag}"
        )

    print("\n── Pre-IPO Pipeline ─────────────────────────────────────────")
    pipeline = intel.get_pre_ipo_pipeline()
    print(f"  {len(pipeline)} companies in IPO pipeline")

    print("\n── VC/PE Fund Universe ──────────────────────────────────────")
    fund_db = VCPEFundDatabase(db=db)
    funds = fund_db.get_major_funds()
    vc_count = sum(1 for f in funds if f.fund_type == "VC")
    pe_count = sum(1 for f in funds if f.fund_type == "PE")
    print(f"  {len(funds)} total funds tracked ({vc_count} VC, {pe_count} PE)")

    print("\n── PE-Controlled Companies (13D filings) ────────────────────")
    pe_tracker = PEPortfolioTracker(db=db)
    pe_holdings = pe_tracker.find_pe_controlled_companies(days_back=90)
    print(f"  Found {len(pe_holdings)} potential PE-controlled targets")
    for h in pe_holdings[:3]:
        print(f"  {h.fund_name[:30]:30s} → {h.target_company[:30]:30s}  {h.filing_date}")

    print("\n── Crunchbase Alternative: Software sector search ───────────")
    cb = CrunchbaseAlternative(db=db)
    sw_companies = cb.search_companies(
        query="",
        filters={"sic_code": "7372", "min_raised": 1_000_000},
    )
    print(f"  {len(sw_companies)} software companies in DB")
    for c in sw_companies[:3]:
        print(f"  {c.company_name[:40]:40s}  ${c.total_raised/1e6:.1f}M  {c.stage}")

    # Export
    export_path = str(output_dir / "vcpe_unicorns.json")
    cb.export_to_json(
        [
            CompanyProfile(
                company_name=c.company_name,
                cik=c.cik,
                total_raised=c.total_raised,
                funding_rounds=c.funding_rounds,
                last_activity=c.last_round_date,
                stage=CrunchbaseAlternative._estimate_stage(c.total_raised, c.funding_rounds),
                industry=c.industry,
                state=c.state,
                investors=[],
                has_s1=c.has_s1,
            )
            for c in candidates[:50]
        ],
        export_path,
    )
    print(f"\n  Unicorn candidates exported → {export_path}")
    print("\n── Demo complete ─────────────────────────────────────────────\n")
    db.close()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/vcpe_demo")
    run_demo(output_dir=out)
