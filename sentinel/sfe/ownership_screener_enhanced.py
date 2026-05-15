"""
Institutional + insider ownership screener: smart money tracking,
cluster buy signals, hedge fund overlap, conviction change detection.
13F + Form 4 + Schedule 13D/G via EDGAR.

dim_072 — Ownership-based screener (13F + Form 4) (target: 9)

Builds on sentinel/sfe/institutional_ownership_enhanced.py (ThirteenFAdapter,
InstitutionalOwnershipAnalyzer) with much richer signal generation:
  • SmartMoneyTracker         — 30 tier-1 funds, portfolio overlap, consensus positions
  • InsiderClusterSignal      — cluster buys (3+ insiders / 30 days), CEO/CFO 2x weight
  • Schedule13DGMonitor       — activist (SC 13D) + passive (SC 13G) large holders
  • OwnershipConcentrationAnalyzer — institutional %, short interest, float turnover
  • OwnershipScreener         — multi-criteria screener with conviction ranking
  • CrowdingMonitor           — crowding index, squeeze candidates, uncrowded value
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import pandas as pd
import requests

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_SEC_BASE = "https://www.sec.gov"
_EDGAR_DATA = "https://data.sec.gov"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_TIMEOUT = 30
_RATE_DELAY = 0.15   # 150 ms per request — EDGAR rate limit ~8 req/s

_DB_PATH = Path(__file__).parent.parent / "data" / "ownership_screener.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS smart_money_holdings (
            manager_name     TEXT NOT NULL,
            manager_cik      TEXT NOT NULL,
            ticker           TEXT NOT NULL,
            cusip            TEXT,
            shares           REAL,
            value_usd        REAL,
            pct_portfolio    REAL,
            period           TEXT,
            filing_date      TEXT,
            PRIMARY KEY (manager_cik, ticker, period)
        );
        CREATE TABLE IF NOT EXISTS insider_transactions (
            acc_number       TEXT PRIMARY KEY,
            ticker           TEXT,
            issuer_cik       TEXT,
            owner_name       TEXT,
            owner_cik        TEXT,
            role             TEXT,
            transaction_date TEXT,
            shares           REAL,
            price_per_share  REAL,
            total_value      REAL,
            tx_code          TEXT,
            is_10b51         INTEGER DEFAULT 0,
            filing_date      TEXT
        );
        CREATE TABLE IF NOT EXISTS activist_filings (
            acc_number       TEXT PRIMARY KEY,
            form_type        TEXT,
            filer_name       TEXT,
            subject_ticker   TEXT,
            subject_cik      TEXT,
            ownership_pct    REAL,
            filing_date      TEXT,
            period_of_report TEXT,
            is_activist      INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS ownership_concentration (
            ticker           TEXT PRIMARY KEY,
            institutional_pct REAL,
            insider_pct      REAL,
            short_pct_float  REAL,
            days_to_cover    REAL,
            float_shares     REAL,
            net_inst_change_q REAL,
            as_of            TEXT
        );
        CREATE TABLE IF NOT EXISTS cluster_buy_signals (
            ticker           TEXT NOT NULL,
            signal_date      TEXT NOT NULL,
            n_insiders       INTEGER,
            cluster_strength REAL,
            total_value_usd  REAL,
            ceo_cfo_included INTEGER,
            PRIMARY KEY (ticker, signal_date)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# XML / HTTP helpers
# ---------------------------------------------------------------------------


def _strip_ns(xml_bytes: bytes) -> bytes:
    """Strip XML namespaces for simple XPath."""
    text = xml_bytes.decode("utf-8", errors="replace")
    text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", text)
    text = re.sub(r"<(/?)[\w]+:([\w]+)", r"<\1\2", text)
    return text.encode("utf-8")


def _safe_decimal(val: Any) -> Optional[Decimal]:
    if not val:
        return None
    try:
        return Decimal(str(val).strip().replace(",", ""))
    except InvalidOperation:
        return None


def _get(url: str, headers: Optional[Dict] = None, timeout: int = _TIMEOUT) -> Optional[requests.Response]:
    """GET with rate limiting and error logging."""
    h = {**_HEADERS, **(headers or {})}
    try:
        resp = requests.get(url, headers=h, timeout=timeout)
        time.sleep(_RATE_DELAY)
        if resp.status_code == 200:
            return resp
        logger.warning("_get non-200", url=url, status=resp.status_code)
        return None
    except Exception as exc:
        logger.warning("_get failed", url=url, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# TIER 1 INVESTOR TABLE
# ---------------------------------------------------------------------------

TIER1_INVESTORS: Dict[str, str] = {
    # CIK → manager name
    "0001067983": "Berkshire Hathaway",
    "0001037389": "Renaissance Technologies",
    "0001350694": "Bridgewater Associates",
    "0001423298": "Citadel Advisors",
    "0001278021": "Two Sigma Investments",
    "0001009626": "D.E. Shaw",
    "0001336528": "AQR Capital Management",
    "0001603466": "Point72 Asset Management",
    "0001273087": "Millennium Management",
    "0001061768": "Baupost Group",
    "0001103804": "Viking Global Investors",
    "0001167483": "Tiger Global Management",
    "0001336092": "Coatue Management",
    "0001061165": "Lone Pine Capital",
    "0001040273": "Third Point",
    "0001175483": "ValueAct Capital",
    "0001048268": "Elliott Management",
    "0001517767": "Starboard Value",
    "0001159159": "Jana Partners",
    "0001079114": "Greenlight Capital",
    "0001070154": "Appaloosa Management",
    "0001326190": "Oaktree Capital",
    "0001404912": "KKR",
    "0001393818": "Blackstone",
    "0000102909": "Vanguard Group",
    "0001364742": "BlackRock",
    "0000093751": "State Street Global Advisors",
    "0000315066": "Fidelity Management",
    "0000080255": "T. Rowe Price",
    "0000277344": "Capital Group",
}

# CIK reverse lookup: name → CIK
TIER1_BY_NAME: Dict[str, str] = {v: k for k, v in TIER1_INVESTORS.items()}

# Insider role weighting for cluster signal
_ROLE_WEIGHTS: Dict[str, float] = {
    "ceo": 2.0,
    "president": 2.0,
    "cfo": 2.0,
    "coo": 1.8,
    "director": 1.5,
    "officer": 1.2,
    "10pct": 1.5,
    "other": 1.0,
}

# -----------------------------------------------------------------------
# Representative universe for screening
# -----------------------------------------------------------------------

_SCREEN_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "UNH", "XOM", "COST", "MA", "HD", "PG", "JNJ", "ORCL",
    "BAC", "ABBV", "MRK", "KO", "CVX", "CRM", "NFLX", "AMD", "PEP", "ADBE",
    "TMO", "WMT", "LIN", "ACN", "MCD", "CSCO", "ABT", "PM", "DHR", "CAT",
]


# ---------------------------------------------------------------------------
# SmartMoneyTracker
# ---------------------------------------------------------------------------


class SmartMoneyTracker:
    """Track holdings and signals for top 30 Tier-1 institutional investors.

    Pulls 13F-HR filings from EDGAR, computes portfolio overlap,
    identifies consensus positions and new/exited positions.
    """

    def __init__(self) -> None:
        self._db = _get_db()
        self._holdings_cache: Dict[str, pd.DataFrame] = {}  # cik → holdings df

    # ------------------------------------------------------------------
    # 13F data retrieval
    # ------------------------------------------------------------------

    def get_manager_13f(
        self,
        manager_cik: str,
        lookback_quarters: int = 4,
    ) -> List[Dict[str, Any]]:
        """Return 13F filing metadata for a manager."""
        cik_stripped = manager_cik.lstrip("0") or "0"
        padded = cik_stripped.zfill(10)
        url = f"{_EDGAR_DATA}/submissions/CIK{padded}.json"
        resp = _get(url)
        if not resp:
            return []

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])

        cutoff = datetime.utcnow() - timedelta(days=lookback_quarters * 92)
        results: List[Dict[str, Any]] = []
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
                "manager_cik": manager_cik,
            })
        return results

    def parse_13f_holdings(self, accession_number: str, manager_cik: str) -> pd.DataFrame:
        """Parse 13F infotable XML, return DataFrame of holdings."""
        cik_stripped = manager_cik.lstrip("0") or "0"
        acc_clean = accession_number.replace("-", "")
        index_url = f"{_ARCHIVES}/{cik_stripped}/{acc_clean}/{accession_number}-index.json"

        xml_url: Optional[str] = None
        resp = _get(index_url)
        if resp:
            try:
                idx = resp.json()
                for doc in idx.get("documents", []):
                    docname = doc.get("document", "").lower()
                    doctype = doc.get("type", "").upper()
                    if "infotable" in docname or doctype in ("13F-HR", "INFORMATION TABLE"):
                        xml_url = f"{_SEC_BASE}/Archives/edgar/data/{cik_stripped}/{acc_clean}/{doc['document']}"
                        break
            except Exception:
                pass

        if not xml_url:
            xml_url = f"{_ARCHIVES}/{cik_stripped}/{acc_clean}/infotable.xml"

        resp2 = _get(xml_url, headers={"Accept": "application/xml, text/xml, */*"})
        if not resp2:
            return pd.DataFrame()

        xml_bytes = _strip_ns(resp2.content)
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            logger.warning("parse_13f_holdings: XML parse error", acc=accession_number, error=str(exc))
            return pd.DataFrame()

        tables: List[ET.Element] = []
        for tag in ("infoTable", "infotable", "InfoTable"):
            tables = root.findall(f".//{tag}")
            if tables:
                break

        rows: List[Dict[str, Any]] = []
        for tbl in tables:
            def _txt(t: str) -> str:
                el = tbl.find(t) or tbl.find(t.lower())
                return el.text.strip() if el is not None and el.text else ""

            issuer = _txt("nameOfIssuer") or _txt("nameofissuer")
            cusip = _txt("cusip")
            val_str = _txt("value") or "0"
            val_dec = _safe_decimal(val_str)
            value_usd = float(val_dec * 1000) if val_dec else 0.0

            shrs_el = tbl.find("shrsOrPrnAmt") or tbl.find("shrsorprnamt")
            shares: Optional[float] = None
            if shrs_el is not None:
                raw = (shrs_el.findtext("sshPrnamt") or shrs_el.findtext("sshprnamt") or "").strip()
                d = _safe_decimal(raw)
                shares = float(d) if d else None

            put_call = _txt("putCall").strip() or _txt("putcall").strip() or None

            if not issuer and not cusip:
                continue
            rows.append({
                "issuer_name": issuer[:200],
                "cusip": cusip[:9] or None,
                "value_usd": value_usd,
                "shares": shares,
                "put_call": put_call,
                "manager_cik": manager_cik,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Manager holdings with caching
    # ------------------------------------------------------------------

    def get_current_holdings(self, manager_cik: str) -> pd.DataFrame:
        """Return current (latest) 13F holdings for a manager."""
        if manager_cik in self._holdings_cache:
            return self._holdings_cache[manager_cik]

        filings = self.get_manager_13f(manager_cik, lookback_quarters=2)
        if not filings:
            return pd.DataFrame()

        df = self.parse_13f_holdings(filings[0]["accession_number"], manager_cik)
        df["period_of_report"] = filings[0].get("period_of_report")
        df["filing_date"] = filings[0].get("filing_date")
        if not df.empty:
            self._holdings_cache[manager_cik] = df
        return df

    def get_prior_holdings(self, manager_cik: str) -> pd.DataFrame:
        """Return prior quarter 13F holdings for a manager."""
        filings = self.get_manager_13f(manager_cik, lookback_quarters=3)
        if len(filings) < 2:
            return pd.DataFrame()
        return self.parse_13f_holdings(filings[1]["accession_number"], manager_cik)

    # ------------------------------------------------------------------
    # Portfolio overlap — Jaccard similarity
    # ------------------------------------------------------------------

    def portfolio_overlap(
        self,
        cik_a: str,
        cik_b: str,
        by: str = "cusip",
    ) -> Optional[float]:
        """Compute Jaccard similarity between two managers' portfolios.

        Jaccard = |A ∩ B| / |A ∪ B|

        Parameters
        ----------
        by: "cusip" for exact match or "issuer_name" for name-based

        Returns
        -------
        Jaccard similarity [0, 1] or None if data unavailable
        """
        df_a = self.get_current_holdings(cik_a)
        df_b = self.get_current_holdings(cik_b)
        if df_a.empty or df_b.empty:
            return None

        col = by if by in df_a.columns and by in df_b.columns else "issuer_name"
        set_a: Set[str] = set(df_a[col].dropna().astype(str).str.upper())
        set_b: Set[str] = set(df_b[col].dropna().astype(str).str.upper())

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return round(intersection / union, 3)

    def compute_overlap_matrix(
        self,
        cik_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute pairwise Jaccard overlap for all Tier-1 managers.

        Returns symmetric DataFrame (CIK × CIK) with similarity scores.
        """
        ciks = cik_list or list(TIER1_INVESTORS.keys())[:10]  # cap to avoid rate limits
        names = [TIER1_INVESTORS.get(c, c) for c in ciks]
        n = len(ciks)
        matrix = pd.DataFrame(index=names, columns=names, dtype=float)

        for i, (cik_i, name_i) in enumerate(zip(ciks, names)):
            matrix.loc[name_i, name_i] = 1.0
            for j, (cik_j, name_j) in enumerate(zip(ciks, names)):
                if j <= i:
                    continue
                overlap = self.portfolio_overlap(cik_i, cik_j)
                matrix.loc[name_i, name_j] = overlap
                matrix.loc[name_j, name_i] = overlap

        return matrix

    # ------------------------------------------------------------------
    # Consensus holdings
    # ------------------------------------------------------------------

    def get_consensus_holdings(
        self,
        min_managers: int = 5,
        cik_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return stocks held by at least min_managers Tier-1 investors.

        Returns
        -------
        DataFrame: issuer_name, n_managers, manager_names, total_value_usd,
        new_buyers_this_q, exited_this_q — sorted by n_managers desc
        """
        ciks = cik_list or list(TIER1_INVESTORS.keys())
        # Count holdings across managers
        ticker_counts: Dict[str, Dict[str, Any]] = {}

        for cik in ciks[:15]:  # cap for rate limits
            manager_name = TIER1_INVESTORS.get(cik, cik)
            df = self.get_current_holdings(cik)
            if df.empty:
                continue
            for _, row in df.iterrows():
                name = str(row.get("issuer_name", "") or "").upper().strip()
                if not name:
                    continue
                if name not in ticker_counts:
                    ticker_counts[name] = {
                        "managers": [],
                        "total_value_usd": 0.0,
                        "cusip": row.get("cusip"),
                    }
                ticker_counts[name]["managers"].append(manager_name)
                ticker_counts[name]["total_value_usd"] += float(row.get("value_usd") or 0)

        rows: List[Dict[str, Any]] = []
        for issuer, data in ticker_counts.items():
            n_mgrs = len(data["managers"])
            if n_mgrs >= min_managers:
                rows.append({
                    "issuer_name": issuer,
                    "cusip": data["cusip"],
                    "n_managers": n_mgrs,
                    "manager_names": ", ".join(sorted(set(data["managers"]))),
                    "total_value_usd_mn": round(data["total_value_usd"] / 1e6, 1),
                })

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows).sort_values("n_managers", ascending=False)
        logger.info("get_consensus_holdings", n_consensus=len(df_out), min_mgrs=min_managers)
        return df_out

    # ------------------------------------------------------------------
    # New / exited positions
    # ------------------------------------------------------------------

    def get_new_positions(
        self,
        manager_cik: str,
    ) -> Tuple[List[str], List[str]]:
        """Return (new_positions, exited_positions) vs prior quarter for a manager."""
        current_df = self.get_current_holdings(manager_cik)
        prior_df = self.get_prior_holdings(manager_cik)

        if current_df.empty:
            return [], []

        cur_names: Set[str] = set(current_df["issuer_name"].dropna().str.upper())
        prior_names: Set[str] = set(prior_df["issuer_name"].dropna().str.upper()) if not prior_df.empty else set()

        new_pos = sorted(cur_names - prior_names)
        exited = sorted(prior_names - cur_names)
        return new_pos, exited

    def get_tier1_new_positions(
        self,
        cik_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate new positions across Tier-1 managers.

        Returns issuers that appeared as NEW positions in the latest quarter
        for multiple managers, sorted by new-buyer count.
        """
        ciks = cik_list or list(TIER1_INVESTORS.keys())[:15]
        new_pos_count: Dict[str, Dict[str, Any]] = {}

        for cik in ciks:
            manager_name = TIER1_INVESTORS.get(cik, cik)
            new_pos, _ = self.get_new_positions(cik)
            for issuer in new_pos:
                if issuer not in new_pos_count:
                    new_pos_count[issuer] = {"buyers": [], "count": 0}
                new_pos_count[issuer]["buyers"].append(manager_name)
                new_pos_count[issuer]["count"] += 1

        rows = [
            {
                "issuer_name": issuer,
                "new_buyer_count": data["count"],
                "new_buyers": ", ".join(data["buyers"]),
            }
            for issuer, data in new_pos_count.items()
            if data["count"] >= 2
        ]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("new_buyer_count", ascending=False)


# ---------------------------------------------------------------------------
# InsiderClusterSignal
# ---------------------------------------------------------------------------


class InsiderClusterSignal:
    """Detect clusters of insider buying as a strong alpha signal.

    STRONG signal: 3+ insiders buying same ticker within 30 days.
    CEO/CFO weighted 2x. Excludes 10b5-1 plans and option exercises.
    """

    _FORM4_SEARCH = (
        "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
        "&forms=4&dateRange=custom&startdt={start}&enddt={end}"
        "&hits.hits._source=period_of_report,entity_name,file_num"
    )

    def __init__(self) -> None:
        self._db = _get_db()

    # ------------------------------------------------------------------
    # Form 4 fetching from EDGAR
    # ------------------------------------------------------------------

    def fetch_recent_form4(
        self,
        ticker: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Fetch recent Form 4 filings for a ticker from EDGAR EFTS.

        Returns list of filing metadata dicts.
        """
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        url = (
            f"{_EFTS_BASE}?q=%22{quote(ticker)}%22"
            f"&forms=4"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
        )
        resp = _get(url)
        if not resp:
            return []

        try:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            results = []
            for hit in hits[:50]:  # cap
                src = hit.get("_source", {})
                results.append({
                    "acc_number": hit.get("_id", ""),
                    "filing_date": src.get("file_date", ""),
                    "period_of_report": src.get("period_of_report", ""),
                    "entity_name": src.get("entity_name", ""),
                    "ticker": ticker,
                })
            return results
        except Exception as exc:
            logger.warning("fetch_recent_form4 parse error", ticker=ticker, error=str(exc))
            return []

    def parse_form4_xml(
        self,
        acc_number: str,
        issuer_cik: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Parse a Form 4 XML filing.

        Returns dict with owner info and open-market transactions.
        Excludes 10b5-1 plans (footnotes) and option exercises (code M/A).
        """
        acc_clean = acc_number.replace("-", "")
        cik_num = issuer_cik.lstrip("0") or "0"

        # Try primary document URL pattern
        url = f"{_ARCHIVES}/{cik_num}/{acc_clean}/{acc_number}.xml"
        resp = _get(url, headers={"Accept": "application/xml, text/xml, */*"})
        if not resp:
            # Try index for actual document filename
            idx_url = f"{_ARCHIVES}/{cik_num}/{acc_clean}/{acc_number}-index.json"
            idx_resp = _get(idx_url)
            if idx_resp:
                try:
                    docs = idx_resp.json().get("documents", [])
                    for doc in docs:
                        if doc.get("type", "").upper() in ("4", "OWNERSHIP"):
                            doc_url = f"{_SEC_BASE}/Archives/edgar/data/{cik_num}/{acc_clean}/{doc['document']}"
                            resp = _get(doc_url, headers={"Accept": "application/xml, */*"})
                            break
                except Exception:
                    pass

        if not resp:
            return None

        try:
            xml_bytes = _strip_ns(resp.content)
            root = ET.fromstring(xml_bytes)
        except Exception as exc:
            logger.debug("parse_form4_xml: XML parse error", acc=acc_number, error=str(exc))
            return None

        def _txt(el: ET.Element, path: str) -> str:
            found = el.find(path)
            return found.text.strip() if found is not None and found.text else ""

        # Issuer
        issuer_el = root.find(".//issuer")
        issuer_name = _txt(root, ".//issuerName") if issuer_el is None else _txt(issuer_el, "issuerName")
        ticker = _txt(root, ".//issuerTradingSymbol")

        # Reporting owner
        owner_el = root.find(".//reportingOwner")
        if owner_el is None:
            return None
        owner_name = _txt(owner_el, ".//rptOwnerName")
        owner_cik_val = _txt(owner_el, ".//rptOwnerCik")
        is_director = _txt(owner_el, ".//isDirector") == "1"
        is_officer = _txt(owner_el, ".//isOfficer") == "1"
        is_ten_pct = _txt(owner_el, ".//isTenPercentOwner") == "1"
        officer_title = _txt(owner_el, ".//officerTitle").lower()

        # Determine role
        role = self._infer_role(is_director, is_officer, is_ten_pct, officer_title)
        role_weight = _ROLE_WEIGHTS.get(role, 1.0)

        # Non-derivative transactions (open market buys/sells)
        open_market_buys: List[Dict[str, Any]] = []
        for tx in root.findall(".//nonDerivativeTransaction"):
            code_el = tx.find(".//transactionCode")
            code = code_el.text.strip().upper() if code_el is not None and code_el.text else ""
            # Only open-market purchases (code P); skip exercises (M, A, S, etc.)
            if code != "P":
                continue

            # Check for 10b5-1 plan indicator
            deemed_el = tx.find(".//transactionDeemedExecution")
            is_10b51 = deemed_el is not None and deemed_el.text is not None and "1" in str(deemed_el.text)

            shares_el = tx.find(".//transactionShares/value")
            price_el = tx.find(".//transactionPricePerShare/value")
            date_el = tx.find(".//transactionDate/value")

            shares = float(_safe_decimal(shares_el.text) or 0) if shares_el is not None else 0.0
            price = float(_safe_decimal(price_el.text) or 0) if price_el is not None else 0.0
            tx_date = date_el.text.strip() if date_el is not None and date_el.text else ""

            if shares > 0:
                open_market_buys.append({
                    "acc_number": acc_number,
                    "issuer_name": issuer_name,
                    "ticker": ticker,
                    "owner_name": owner_name,
                    "owner_cik": owner_cik_val,
                    "role": role,
                    "role_weight": role_weight,
                    "transaction_date": tx_date,
                    "shares": shares,
                    "price_per_share": price,
                    "total_value": shares * price,
                    "tx_code": code,
                    "is_10b51": is_10b51,
                })

        return {
            "ticker": ticker,
            "issuer_name": issuer_name,
            "owner_name": owner_name,
            "owner_cik": owner_cik_val,
            "role": role,
            "role_weight": role_weight,
            "open_market_buys": open_market_buys,
        }

    def _infer_role(
        self,
        is_director: bool,
        is_officer: bool,
        is_ten_pct: bool,
        officer_title: str,
    ) -> str:
        """Classify insider role for weighting."""
        title_l = officer_title.lower()
        if "chief executive" in title_l or title_l.startswith("ceo"):
            return "ceo"
        if "president" in title_l and "vice" not in title_l:
            return "president"
        if "chief financial" in title_l or title_l.startswith("cfo"):
            return "cfo"
        if "chief operating" in title_l or title_l.startswith("coo"):
            return "coo"
        if is_ten_pct:
            return "10pct"
        if is_director:
            return "director"
        if is_officer:
            return "officer"
        return "other"

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def detect_cluster_buy(
        self,
        ticker: str,
        days: int = 30,
        min_insiders: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Detect if 3+ insiders bought ticker stock within the last N days.

        Parameters
        ----------
        ticker: equity ticker
        days: rolling window for cluster detection
        min_insiders: minimum insider buyers to trigger cluster signal

        Returns
        -------
        dict with cluster details, or None if no cluster detected
        """
        filings = self.fetch_recent_form4(ticker, days=max(days + 30, 90))
        if not filings:
            return None

        window_start = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        buys_in_window: List[Dict[str, Any]] = []

        for filing in filings[:20]:  # check up to 20 filings
            acc = filing.get("acc_number", "")
            if not acc:
                continue
            parsed = self.parse_form4_xml(acc)
            if not parsed:
                continue
            for buy in parsed.get("open_market_buys", []):
                if buy.get("is_10b51"):
                    continue  # exclude 10b5-1 plans
                tx_date = buy.get("transaction_date", "")
                if tx_date >= window_start:
                    buys_in_window.append(buy)

        if not buys_in_window:
            return None

        # Group by owner to deduplicate (one insider may file multiple forms)
        seen_owners: Set[str] = set()
        unique_buys: List[Dict[str, Any]] = []
        for buy in buys_in_window:
            owner_key = buy.get("owner_cik") or buy.get("owner_name", "")
            if owner_key not in seen_owners:
                seen_owners.add(owner_key)
                unique_buys.append(buy)

        n_unique_insiders = len(unique_buys)
        if n_unique_insiders < min_insiders:
            return None

        # Compute cluster strength: sum(role_weight × log(dollar_value))
        cluster_strength = 0.0
        total_value = 0.0
        ceo_cfo_included = False

        for buy in unique_buys:
            val = buy.get("total_value", 0) or 0
            weight = buy.get("role_weight", 1.0) or 1.0
            if val > 0:
                cluster_strength += weight * math.log(val + 1)
                total_value += val
            if buy.get("role") in ("ceo", "cfo", "president"):
                ceo_cfo_included = True

        signal = {
            "ticker": ticker,
            "n_insiders": n_unique_insiders,
            "cluster_strength": round(cluster_strength, 2),
            "total_value_usd": round(total_value, 0),
            "ceo_cfo_included": ceo_cfo_included,
            "window_days": days,
            "signal_date": datetime.utcnow().date().isoformat(),
            "insiders": [
                {
                    "owner_name": b.get("owner_name"),
                    "role": b.get("role"),
                    "shares": b.get("shares"),
                    "total_value": b.get("total_value"),
                    "transaction_date": b.get("transaction_date"),
                }
                for b in unique_buys
            ],
        }

        # Persist to DB
        try:
            self._db.execute("""
                INSERT OR REPLACE INTO cluster_buy_signals
                (ticker, signal_date, n_insiders, cluster_strength, total_value_usd, ceo_cfo_included)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                signal["signal_date"],
                n_unique_insiders,
                cluster_strength,
                total_value,
                int(ceo_cfo_included),
            ))
            self._db.commit()
        except Exception as exc:
            logger.warning("detect_cluster_buy persist failed", error=str(exc))

        logger.info("detect_cluster_buy", ticker=ticker, n_insiders=n_unique_insiders,
                    strength=cluster_strength)
        return signal

    def screen_cluster_buys(
        self,
        tickers: Optional[List[str]] = None,
        days: int = 30,
        min_insiders: int = 3,
    ) -> pd.DataFrame:
        """Screen universe for cluster insider buying signals.

        Returns
        -------
        DataFrame sorted by cluster_strength descending
        """
        universe = tickers or _SCREEN_UNIVERSE[:20]
        results: List[Dict[str, Any]] = []

        for ticker in universe:
            signal = self.detect_cluster_buy(ticker, days=days, min_insiders=min_insiders)
            if signal:
                results.append({
                    "ticker": signal["ticker"],
                    "n_insiders": signal["n_insiders"],
                    "cluster_strength": signal["cluster_strength"],
                    "total_value_usd": signal["total_value_usd"],
                    "ceo_cfo_included": signal["ceo_cfo_included"],
                })

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).sort_values("cluster_strength", ascending=False)
        df = df.set_index("ticker")
        logger.info("screen_cluster_buys", n_signals=len(df))
        return df

    def get_insider_history(self, ticker: str) -> pd.DataFrame:
        """Return recent insider transactions for a ticker (all open-market buys/sells)."""
        filings = self.fetch_recent_form4(ticker, days=180)
        all_buys: List[Dict[str, Any]] = []
        for f in filings[:15]:
            parsed = self.parse_form4_xml(f.get("acc_number", ""))
            if parsed:
                for buy in parsed.get("open_market_buys", []):
                    buy["ticker"] = ticker
                    all_buys.append(buy)

        if not all_buys:
            return pd.DataFrame()
        df = pd.DataFrame(all_buys)
        df = df.sort_values("transaction_date", ascending=False)
        return df


# ---------------------------------------------------------------------------
# Schedule13DGMonitor
# ---------------------------------------------------------------------------


class Schedule13DGMonitor:
    """Monitor Schedule 13D (activist) and 13G (passive large holder) filings from EDGAR."""

    _SC_FORMS = ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"]

    def __init__(self) -> None:
        self._db = _get_db()

    # ------------------------------------------------------------------
    # EDGAR EFTS search for 13D/G filings
    # ------------------------------------------------------------------

    def search_13dg_filings(
        self,
        ticker: str,
        days: int = 90,
        form_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search EDGAR for SC 13D/13G filings mentioning the ticker.

        Parameters
        ----------
        ticker: subject company ticker
        days: lookback window
        form_types: list of form types; None = all 13D and 13G variants

        Returns
        -------
        list of {acc_number, form_type, filer_name, filing_date, period_of_report}
        """
        forms_filter = form_types or ["SC 13D", "SC 13G"]
        form_str = "%2C".join(quote(f) for f in forms_filter)

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        url = (
            f"{_EFTS_BASE}?q=%22{quote(ticker)}%22"
            f"&forms={form_str}"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
        )
        resp = _get(url)
        if not resp:
            return []

        try:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            results: List[Dict[str, Any]] = []
            for hit in hits[:30]:
                src = hit.get("_source", {})
                results.append({
                    "acc_number": hit.get("_id", ""),
                    "form_type": src.get("form_type", ""),
                    "filer_name": src.get("entity_name", ""),
                    "filing_date": src.get("file_date", ""),
                    "period_of_report": src.get("period_of_report", ""),
                    "subject_ticker": ticker,
                })
            logger.info("search_13dg_filings", ticker=ticker, n=len(results))
            return results
        except Exception as exc:
            logger.warning("search_13dg_filings parse error", ticker=ticker, error=str(exc))
            return []

    def parse_13dg_xml(
        self,
        acc_number: str,
        filer_cik: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Parse a Schedule 13D or 13G XML filing for ownership percentage and intent.

        Returns
        -------
        dict with: filer_name, subject_company, ownership_pct, is_activist,
        filing_date, amend_type
        """
        if not acc_number:
            return None

        acc_clean = acc_number.replace("-", "")
        cik_num = filer_cik.lstrip("0") or "0"
        url = f"{_ARCHIVES}/{cik_num}/{acc_clean}/{acc_number}.xml"
        resp = _get(url, headers={"Accept": "application/xml, */*"})

        if not resp:
            # Try alternate: fetch filing index
            idx_url = f"{_ARCHIVES}/{cik_num}/{acc_clean}/{acc_number}-index.json"
            idx_resp = _get(idx_url)
            if idx_resp:
                try:
                    docs = idx_resp.json().get("documents", [])
                    for doc in docs:
                        dtype = doc.get("type", "").upper()
                        if "13D" in dtype or "13G" in dtype or dtype == "PRIMARY DOCUMENT":
                            doc_url = f"{_SEC_BASE}/Archives/edgar/data/{cik_num}/{acc_clean}/{doc['document']}"
                            resp = _get(doc_url, headers={"Accept": "application/xml, */*"})
                            if resp:
                                break
                except Exception:
                    pass

        if not resp:
            return None

        try:
            xml_bytes = _strip_ns(resp.content)
            root = ET.fromstring(xml_bytes)
        except Exception:
            return None

        def _find_text(*paths: str) -> str:
            for path in paths:
                el = root.find(f".//{path}")
                if el is not None and el.text:
                    return el.text.strip()
            return ""

        filer_name = _find_text("reporterName", "filerName", "nameOfReportingPerson")
        subject_company = _find_text("nameOfIssuer", "subjectCompany")
        ownership_pct_str = _find_text("percentOfClassRepresented", "ownershipPercent")
        ownership_pct: Optional[float] = None
        d = _safe_decimal(re.sub(r"[%\s]", "", ownership_pct_str))
        if d:
            ownership_pct = float(d)

        # 13D is activist (intent to change control); 13G is passive
        form_type = _find_text("formType") or ""
        is_activist = "13D" in form_type.upper() and "/A" not in form_type.upper()

        # Check for 13G → 13D conversion language
        purpose_text = _find_text("purposeOfTransaction")
        if purpose_text and any(w in purpose_text.lower() for w in ["change", "board", "merger", "acquisition"]):
            is_activist = True

        return {
            "filer_name": filer_name or "Unknown",
            "subject_company": subject_company,
            "ownership_pct": ownership_pct,
            "is_activist": is_activist,
            "form_type": form_type,
            "purpose_text": purpose_text[:200] if purpose_text else None,
        }

    # ------------------------------------------------------------------
    # Activist screening
    # ------------------------------------------------------------------

    def get_activist_filings(
        self,
        ticker: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Return activist 13D filings for a ticker within N days.

        An SC 13D filing signals an activist (>5% stake with intent to engage management).

        Returns
        -------
        list of activist filing dicts with ownership % and purpose
        """
        filings = self.search_13dg_filings(ticker, days=days, form_types=["SC 13D", "SC 13D/A"])
        results: List[Dict[str, Any]] = []

        for f in filings[:10]:
            acc = f.get("acc_number", "")
            parsed = self.parse_13dg_xml(acc)
            if parsed:
                entry = {**f, **parsed}
                entry["is_activist"] = True
                results.append(entry)

        logger.info("get_activist_filings", ticker=ticker, n=len(results))
        return results

    def monitor_universe_activists(
        self,
        tickers: Optional[List[str]] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """Screen universe for recent activist 13D filings.

        Returns
        -------
        DataFrame of stocks with recent activist filings, sorted by filing date
        """
        universe = tickers or _SCREEN_UNIVERSE[:20]
        rows: List[Dict[str, Any]] = []

        for ticker in universe:
            activist_filings = self.get_activist_filings(ticker, days=days)
            for f in activist_filings:
                rows.append({
                    "ticker": ticker,
                    "filer_name": f.get("filer_name", ""),
                    "ownership_pct": f.get("ownership_pct"),
                    "form_type": f.get("form_type", ""),
                    "filing_date": f.get("filing_date", ""),
                    "is_activist": f.get("is_activist", False),
                    "purpose": f.get("purpose_text", ""),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("filing_date", ascending=False)
        logger.info("monitor_universe_activists", n_stocks=len(df))
        return df

    def detect_13g_to_13d_conversion(
        self,
        ticker: str,
        days: int = 180,
    ) -> bool:
        """Detect if a passive 13G holder converted to activist 13D in the period.

        Passive → Activist conversion is a very strong event-driven signal.

        Returns
        -------
        True if conversion detected, False otherwise
        """
        all_filings = self.search_13dg_filings(ticker, days=days)
        # Check if same filer has both a 13G (older) and 13D (newer)
        filer_forms: Dict[str, List[str]] = {}
        for f in all_filings:
            fname = f.get("filer_name", "").lower()
            ftype = f.get("form_type", "").upper()
            if fname not in filer_forms:
                filer_forms[fname] = []
            filer_forms[fname].append(ftype)

        for fname, forms in filer_forms.items():
            has_13g = any("13G" in f and "13D" not in f for f in forms)
            has_13d = any("13D" in f for f in forms)
            if has_13g and has_13d:
                logger.info("detect_13g_to_13d_conversion: conversion found",
                            ticker=ticker, filer=fname)
                return True
        return False


# ---------------------------------------------------------------------------
# OwnershipConcentrationAnalyzer
# ---------------------------------------------------------------------------


class OwnershipConcentrationAnalyzer:
    """Compute ownership concentration metrics: institutional %, short interest, float turnover."""

    def __init__(self) -> None:
        self._db = _get_db()

    def _fetch_info(self, ticker: str) -> Dict[str, Any]:
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            info = t.info or {}
            time.sleep(0.3)
            return info
        except Exception as exc:
            logger.warning("_fetch_info failed", ticker=ticker, error=str(exc))
            return {}

    def compute_concentration_metrics(self, ticker: str) -> Dict[str, Any]:
        """Compute full ownership concentration profile for a stock.

        Metrics
        -------
        institutional_ownership_pct  : % float held by institutions
        insider_ownership_pct        : % float held by insiders
        short_pct_float              : short interest as % of float
        days_to_cover                : short interest / avg daily volume
        float_turnover               : volume / float (monthly)
        supply_demand_imbalance      : net institutional Q/Q change (proxy)
        net_buyer_count_1q           : new institutional buyers - sellers
        """
        info = self._fetch_info(ticker)
        if not info:
            return {"ticker": ticker, "error": "No data available"}

        # Institutional ownership
        inst_pct = info.get("heldPercentInstitutions")
        inst_pct_f = round(float(inst_pct) * 100, 2) if inst_pct is not None else None

        # Insider ownership
        insider_pct = info.get("heldPercentInsiders")
        insider_pct_f = round(float(insider_pct) * 100, 2) if insider_pct is not None else None

        # Short interest
        short_pct = info.get("shortPercentOfFloat")
        short_pct_f = round(float(short_pct) * 100, 2) if short_pct is not None else None

        # Days to cover
        short_ratio = info.get("shortRatio")
        dtc = round(float(short_ratio), 1) if short_ratio else None

        # Float shares
        float_shares = info.get("floatShares") or 0
        avg_volume = info.get("averageVolume") or 0

        # Float turnover (30-day): avg_volume * 30 / float
        float_turnover: Optional[float] = None
        if float_shares > 0 and avg_volume > 0:
            float_turnover = round(avg_volume * 30 / float_shares * 100, 1)

        # Net institutional buyers (proxy: institutionsCount Q/Q)
        inst_now = info.get("institutionsCount") or 0
        inst_prev = info.get("institutionsCountPreviousQuarter") or 0
        net_buyers = inst_now - inst_prev if inst_now and inst_prev else None

        # Supply/demand imbalance score
        supply_demand = None
        if net_buyers is not None and inst_prev > 0:
            supply_demand = round((net_buyers / inst_prev) * 100, 1)

        result = {
            "ticker": ticker,
            "institutional_ownership_pct": inst_pct_f,
            "insider_ownership_pct": insider_pct_f,
            "short_pct_float": short_pct_f,
            "days_to_cover": dtc,
            "float_shares_mn": round(float_shares / 1e6, 1) if float_shares else None,
            "avg_volume_mn": round(avg_volume / 1e6, 1) if avg_volume else None,
            "float_turnover_pct_30d": float_turnover,
            "net_buyer_count_1q": net_buyers,
            "supply_demand_imbalance_pct": supply_demand,
            "institutions_count": inst_now,
            "as_of": datetime.utcnow().date().isoformat(),
        }

        # Persist to DB
        try:
            self._db.execute("""
                INSERT OR REPLACE INTO ownership_concentration
                (ticker, institutional_pct, insider_pct, short_pct_float, days_to_cover,
                 float_shares, net_inst_change_q, as_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, inst_pct_f, insider_pct_f, short_pct_f, dtc,
                float_shares, net_buyers, result["as_of"],
            ))
            self._db.commit()
        except Exception as exc:
            logger.warning("persist concentration metrics failed", ticker=ticker, error=str(exc))

        return result

    def compute_ownership_herfindahl(self, ticker: str) -> Optional[float]:
        """Herfindahl index of top-10 institutional holder concentration.

        HHI = sum(pct_i ^ 2). Higher = more concentrated ownership risk.
        """
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            holders = t.institutional_holders
            info = t.info or {}
            time.sleep(0.3)
        except Exception:
            return None

        if holders is None or holders.empty:
            return None

        float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 0
        if float_shares == 0:
            return None

        shares_col = next((c for c in holders.columns if "share" in c.lower()), None)
        if not shares_col:
            return None

        top10 = holders.head(10)
        hhi = sum(
            (float(row[shares_col] or 0) / float_shares) ** 2
            for _, row in top10.iterrows()
        )
        return round(hhi, 4)


# ---------------------------------------------------------------------------
# OwnershipScreener (enhanced)
# ---------------------------------------------------------------------------


class OwnershipScreener:
    """Multi-criteria ownership screener combining 13F, Form 4, and 13D/G signals.

    Screening criteria:
    - min/max institutional ownership %
    - has_activist (recent SC 13D)
    - min_insider_buy_value (cluster buy dollar value)
    - max_short_pct (short interest ceiling)
    - smart_money_count (number of Tier-1 funds holding)
    - recent_new_position (appeared in Tier-1 13F last quarter)

    Ranking by conviction score: change in position size × portfolio weight.
    """

    def __init__(self) -> None:
        self._smart_money = SmartMoneyTracker()
        self._insider = InsiderClusterSignal()
        self._activist = Schedule13DGMonitor()
        self._concentration = OwnershipConcentrationAnalyzer()
        self._db = _get_db()

    def screen(
        self,
        tickers: Optional[List[str]] = None,
        min_institutional_pct: float = 20.0,
        max_institutional_pct: float = 95.0,
        has_activist: bool = False,
        min_insider_cluster_value: float = 0.0,
        max_short_pct: float = 50.0,
        min_smart_money_count: int = 0,
        require_recent_new_position: bool = False,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Run multi-criteria ownership screen.

        Parameters
        ----------
        tickers: universe to screen; None = default universe
        min_institutional_pct: floor for institutional ownership % (default 20%)
        max_institutional_pct: ceiling for institutional ownership % (default 95%; >95% = overcrowded)
        has_activist: require recent SC 13D activist filing (default False)
        min_insider_cluster_value: minimum cluster buy total value in USD
        max_short_pct: maximum short interest % of float (default 50%)
        min_smart_money_count: minimum Tier-1 managers holding
        require_recent_new_position: require Tier-1 manager added position last quarter
        top_n: return top N results

        Returns
        -------
        DataFrame ranked by conviction score with full ownership metrics
        """
        universe = tickers or _SCREEN_UNIVERSE
        rows: List[Dict[str, Any]] = []

        # Pre-compute consensus holdings for smart money count
        consensus_df = self._smart_money.get_consensus_holdings(min_managers=1)
        consensus_issuers: Set[str] = set()
        issuer_mgr_count: Dict[str, int] = {}
        if not consensus_df.empty:
            for _, row in consensus_df.iterrows():
                name = str(row.get("issuer_name", "")).upper()
                consensus_issuers.add(name)
                issuer_mgr_count[name] = int(row.get("n_managers", 0))

        # Pre-compute Tier-1 new positions for "recent new position" filter
        new_pos_df = pd.DataFrame()
        new_pos_issuers: Set[str] = set()
        if require_recent_new_position:
            new_pos_df = self._smart_money.get_tier1_new_positions()
            if not new_pos_df.empty:
                new_pos_issuers = set(new_pos_df["issuer_name"].str.upper())

        for ticker in universe:
            try:
                # Concentration metrics
                metrics = self._concentration.compute_concentration_metrics(ticker)

                inst_pct = metrics.get("institutional_ownership_pct") or 0
                short_pct = metrics.get("short_pct_float") or 0

                # Filter: institutional ownership range
                if not (min_institutional_pct <= inst_pct <= max_institutional_pct):
                    continue

                # Filter: short interest ceiling
                if short_pct > max_short_pct:
                    continue

                # Smart money count (match ticker to issuer name in consensus holdings)
                ticker_upper = ticker.upper()
                sm_count = 0
                for issuer, count in issuer_mgr_count.items():
                    if ticker_upper in issuer or issuer in ticker_upper:
                        sm_count = count
                        break

                if sm_count < min_smart_money_count:
                    continue

                # Recent new position filter
                if require_recent_new_position:
                    is_new_pos = any(ticker_upper in iss or iss in ticker_upper
                                    for iss in new_pos_issuers)
                    if not is_new_pos:
                        continue

                # Activist filter
                has_recent_activist = False
                if has_activist:
                    activist_filings = self._activist.get_activist_filings(ticker, days=90)
                    has_recent_activist = len(activist_filings) > 0
                    if not has_recent_activist:
                        continue
                else:
                    # Still check but don't require
                    activist_filings = self._activist.get_activist_filings(ticker, days=90)
                    has_recent_activist = len(activist_filings) > 0

                # Insider cluster signal
                cluster_signal = self._insider.detect_cluster_buy(ticker, days=30, min_insiders=3)
                cluster_value = cluster_signal.get("total_value_usd", 0) if cluster_signal else 0
                cluster_strength = cluster_signal.get("cluster_strength", 0) if cluster_signal else 0

                if cluster_value < min_insider_cluster_value:
                    if min_insider_cluster_value > 0:
                        continue

                # Conviction score: composite of signals
                conviction = self._compute_conviction_score(
                    inst_pct=inst_pct,
                    short_pct=short_pct,
                    sm_count=sm_count,
                    cluster_strength=cluster_strength,
                    has_activist=has_recent_activist,
                    net_buyers=metrics.get("net_buyer_count_1q") or 0,
                )

                rows.append({
                    "ticker": ticker,
                    "institutional_pct": inst_pct,
                    "insider_pct": metrics.get("insider_ownership_pct"),
                    "short_pct_float": short_pct,
                    "days_to_cover": metrics.get("days_to_cover"),
                    "smart_money_count": sm_count,
                    "has_activist": has_recent_activist,
                    "cluster_buy_value": cluster_value,
                    "cluster_strength": cluster_strength,
                    "net_buyer_count_1q": metrics.get("net_buyer_count_1q"),
                    "float_turnover_pct": metrics.get("float_turnover_pct_30d"),
                    "conviction_score": conviction,
                })

            except Exception as exc:
                logger.debug("OwnershipScreener.screen skip", ticker=ticker, error=str(exc))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")
        df = df.sort_values("conviction_score", ascending=False).head(top_n)
        logger.info("OwnershipScreener.screen complete", n_results=len(df))
        return df

    def _compute_conviction_score(
        self,
        inst_pct: float,
        short_pct: float,
        sm_count: int,
        cluster_strength: float,
        has_activist: bool,
        net_buyers: int,
    ) -> float:
        """Multi-factor conviction score (0–100)."""
        score = 0.0

        # Smart money count (0–30 points)
        score += min(sm_count * 5, 30)

        # Net institutional buying (0–20 points)
        if net_buyers > 0:
            score += min(net_buyers * 0.5, 20)

        # Insider cluster buy (0–25 points)
        score += min(cluster_strength * 2, 25)

        # Activist (10 points bonus)
        if has_activist:
            score += 10

        # Institutional ownership in sweet spot 40–75% (0–10 points)
        if 40 <= inst_pct <= 75:
            score += 10
        elif 20 <= inst_pct < 40:
            score += 5

        # Short interest — moderate short = potential squeeze catalyst (0–5 points)
        if 5 <= short_pct <= 20:
            score += 5

        return round(min(score, 100), 1)

    def get_top_holdings_by_manager(
        self,
        manager_name: str,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return top holdings for a named Tier-1 manager."""
        cik = TIER1_BY_NAME.get(manager_name)
        if not cik:
            # Try fuzzy match
            for name, cik_val in TIER1_BY_NAME.items():
                if manager_name.lower() in name.lower():
                    cik = cik_val
                    break
        if not cik:
            logger.warning("get_top_holdings_by_manager: unknown manager", name=manager_name)
            return pd.DataFrame()

        df = self._smart_money.get_current_holdings(cik)
        if df.empty:
            return df

        # Compute portfolio weights
        total_val = df["value_usd"].sum()
        df["pct_portfolio"] = df["value_usd"].apply(
            lambda v: round(float(v or 0) / total_val * 100, 2) if total_val else None
        )
        df["manager_name"] = manager_name

        val_col = "value_usd"
        df = df.nlargest(top_n, val_col)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CrowdingMonitor
# ---------------------------------------------------------------------------


class CrowdingMonitor:
    """Identify crowded and uncrowded trades among Tier-1 hedge funds.

    Crowded positions: held by >10 Tier-1 funds — higher unwind risk.
    Uncrowded quality: high quality (ROE > 15%, low debt) + low institutional ownership.
    Short squeeze candidates: high short interest + low institutional + recent insider buy.
    """

    def __init__(self) -> None:
        self._smart_money = SmartMoneyTracker()
        self._concentration = OwnershipConcentrationAnalyzer()
        self._insider = InsiderClusterSignal()

    def compute_crowding_index(
        self,
        tickers: Optional[List[str]] = None,
        cik_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute hedge fund crowding index for each ticker.

        Crowding index = (n_tier1_funds_holding / total_tier1_funds) × 100

        Returns
        -------
        DataFrame: ticker, crowding_index, n_funds_holding, crowding_label
        """
        ciks = cik_list or list(TIER1_INVESTORS.keys())[:15]
        universe = tickers or _SCREEN_UNIVERSE

        # Count Tier-1 fund ownership per issuer name
        issuer_count: Dict[str, Dict[str, Any]] = {}
        for cik in ciks:
            manager_name = TIER1_INVESTORS.get(cik, cik)
            df = self._smart_money.get_current_holdings(cik)
            if df.empty:
                continue
            total_val = df["value_usd"].sum() or 1

            for _, row in df.iterrows():
                issuer = str(row.get("issuer_name", "") or "").upper()
                if not issuer:
                    continue
                val = float(row.get("value_usd") or 0)
                if issuer not in issuer_count:
                    issuer_count[issuer] = {"n_funds": 0, "total_value": 0.0, "funds": []}
                issuer_count[issuer]["n_funds"] += 1
                issuer_count[issuer]["total_value"] += val
                issuer_count[issuer]["funds"].append(manager_name)

        total_funds = len(ciks)
        rows: List[Dict[str, Any]] = []

        # Match universe tickers to issuer names
        for ticker in universe:
            ticker_up = ticker.upper()
            best_match: Optional[str] = None
            for issuer in issuer_count.keys():
                if ticker_up in issuer or issuer[:len(ticker_up)] == ticker_up:
                    best_match = issuer
                    break

            if best_match:
                data = issuer_count[best_match]
                n_funds = data["n_funds"]
                crowding_idx = round(n_funds / total_funds * 100, 1)
                if n_funds >= 10:
                    label = "very_crowded"
                elif n_funds >= 5:
                    label = "crowded"
                elif n_funds >= 2:
                    label = "moderate"
                else:
                    label = "uncrowded"

                rows.append({
                    "ticker": ticker,
                    "crowding_index": crowding_idx,
                    "n_tier1_funds": n_funds,
                    "crowding_label": label,
                    "fund_holders": ", ".join(set(data["funds"])),
                })
            else:
                rows.append({
                    "ticker": ticker,
                    "crowding_index": 0.0,
                    "n_tier1_funds": 0,
                    "crowding_label": "uncrowded",
                    "fund_holders": "",
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")
        df = df.sort_values("crowding_index", ascending=False)
        logger.info("compute_crowding_index", n_tickers=len(df))
        return df

    def find_uncrowded_quality(
        self,
        tickers: Optional[List[str]] = None,
        max_inst_pct: float = 60.0,
        min_roe: float = 15.0,
    ) -> pd.DataFrame:
        """Find high-quality stocks with low institutional ownership = potential value discovery.

        Criteria:
        - Institutional ownership < max_inst_pct (not yet crowded)
        - ROE > min_roe %
        - Positive free cash flow

        Returns
        -------
        DataFrame: ticker, inst_pct, roe, fcf_margin, crowding_index, quality_score
        """
        universe = tickers or _SCREEN_UNIVERSE
        rows: List[Dict[str, Any]] = []

        for ticker in universe:
            try:
                metrics = self._concentration.compute_concentration_metrics(ticker)
                inst_pct = metrics.get("institutional_ownership_pct") or 100.0

                if inst_pct > max_inst_pct:
                    continue

                # Fetch fundamentals for quality check
                import yfinance as yf  # type: ignore
                info = yf.Ticker(ticker).info or {}
                time.sleep(0.3)

                roe = (info.get("returnOnEquity") or 0) * 100
                if roe < min_roe:
                    continue

                fcf = info.get("freeCashflow") or 0
                if fcf <= 0:
                    continue

                gm = (info.get("grossMargins") or 0) * 100
                rev = info.get("totalRevenue") or 1
                fcf_margin = fcf / rev * 100

                quality_score = min(roe / 2 + (60 - inst_pct) * 0.5 + fcf_margin, 100)

                rows.append({
                    "ticker": ticker,
                    "institutional_pct": inst_pct,
                    "roe": round(roe, 1),
                    "gross_margin_pct": round(gm, 1),
                    "fcf_margin_pct": round(fcf_margin, 1),
                    "quality_score": round(quality_score, 1),
                })
            except Exception as exc:
                logger.debug("find_uncrowded_quality skip", ticker=ticker, error=str(exc))

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("ticker").sort_values("quality_score", ascending=False)
        logger.info("find_uncrowded_quality", n_results=len(df))
        return df

    def find_short_squeeze_candidates(
        self,
        tickers: Optional[List[str]] = None,
        min_short_pct: float = 15.0,
        max_inst_pct: float = 70.0,
        require_insider_buy: bool = True,
    ) -> pd.DataFrame:
        """Identify potential short squeeze candidates.

        Classic squeeze setup:
        - High short interest (min_short_pct %)
        - Not excessively institutionally owned (max_inst_pct %)
        - Recent insider buying (optional but significantly strengthens signal)
        - Positive catalyst: activist filing or cluster insider buy

        Returns
        -------
        DataFrame: ticker, short_pct, days_to_cover, inst_pct, insider_buy, squeeze_score
        """
        universe = tickers or _SCREEN_UNIVERSE
        rows: List[Dict[str, Any]] = []

        for ticker in universe:
            try:
                metrics = self._concentration.compute_concentration_metrics(ticker)
                short_pct = metrics.get("short_pct_float") or 0.0
                inst_pct = metrics.get("institutional_ownership_pct") or 100.0
                dtc = metrics.get("days_to_cover") or 0

                if short_pct < min_short_pct:
                    continue
                if inst_pct > max_inst_pct:
                    continue

                # Check for insider buying
                cluster = self._insider.detect_cluster_buy(ticker, days=60, min_insiders=2)
                insider_buy = cluster is not None
                cluster_strength = cluster.get("cluster_strength", 0) if cluster else 0

                if require_insider_buy and not insider_buy:
                    continue

                # Squeeze score: higher = stronger catalyst
                squeeze_score = (
                    min(short_pct * 2, 40)  # short interest component (max 40)
                    + min(dtc * 3, 20)       # days to cover (max 20)
                    + (20 if insider_buy else 0)  # insider buy catalyst (20)
                    + min(cluster_strength, 20)   # cluster strength (max 20)
                )

                rows.append({
                    "ticker": ticker,
                    "short_pct_float": short_pct,
                    "days_to_cover": dtc,
                    "institutional_pct": inst_pct,
                    "insider_cluster_buy": insider_buy,
                    "cluster_strength": round(cluster_strength, 2),
                    "squeeze_score": round(min(squeeze_score, 100), 1),
                })
            except Exception as exc:
                logger.debug("find_short_squeeze_candidates skip", ticker=ticker, error=str(exc))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker").sort_values("squeeze_score", ascending=False)
        logger.info("find_short_squeeze_candidates", n_candidates=len(df))
        return df

    def crowding_risk_report(
        self,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a crowding risk summary for the portfolio/universe.

        Returns
        -------
        {very_crowded, crowded, moderate, uncrowded} grouped tickers with counts
        and risk narrative
        """
        df = self.compute_crowding_index(tickers)
        if df.empty:
            return {"error": "No crowding data available"}

        groups: Dict[str, List[str]] = {
            "very_crowded": [],
            "crowded": [],
            "moderate": [],
            "uncrowded": [],
        }
        for ticker, row in df.iterrows():
            label = row.get("crowding_label", "uncrowded")
            groups[label].append(str(ticker))

        avg_crowding = float(df["crowding_index"].mean())
        risk_level = (
            "HIGH" if avg_crowding > 40
            else "MODERATE" if avg_crowding > 20
            else "LOW"
        )

        return {
            "universe_size": len(df),
            "avg_crowding_index": round(avg_crowding, 1),
            "crowding_risk_level": risk_level,
            "groups": groups,
            "most_crowded": df.head(5).reset_index()[["ticker", "crowding_index", "n_tier1_funds"]].to_dict(orient="records"),
            "least_crowded": df.tail(5).reset_index()[["ticker", "crowding_index", "n_tier1_funds"]].to_dict(orient="records"),
            "as_of": datetime.utcnow().date().isoformat(),
        }


# ---------------------------------------------------------------------------
# FastAPI Router — /api/ownership/v2
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query as QParam

    ownership_router = APIRouter(prefix="/api/ownership/v2", tags=["Ownership Screener Enhanced"])

    _smart_money = SmartMoneyTracker()
    _insider_signal = InsiderClusterSignal()
    _activist_monitor = Schedule13DGMonitor()
    _concentration_analyzer = OwnershipConcentrationAnalyzer()
    _screener = OwnershipScreener()
    _crowding = CrowdingMonitor()

    @ownership_router.get("/smart-money", summary="Tier-1 smart money consensus holdings")
    def get_smart_money_consensus(
        min_managers: int = QParam(5, ge=2, le=20, description="Min Tier-1 managers holding"),
    ) -> Dict[str, Any]:
        """Return stocks held by multiple Tier-1 smart money managers."""
        try:
            df = _smart_money.get_consensus_holdings(min_managers=min_managers)
            if df.empty:
                return {"n_results": 0, "holdings": []}
            return {
                "n_results": len(df),
                "min_managers": min_managers,
                "holdings": df.where(pd.notnull(df), other=None).to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/smart-money/new-positions", summary="Recent new Tier-1 fund positions")
    def get_new_positions() -> Dict[str, Any]:
        """Return issuers where multiple Tier-1 managers opened new positions last quarter."""
        try:
            df = _smart_money.get_tier1_new_positions()
            if df.empty:
                return {"n_results": 0, "new_positions": []}
            return {
                "n_results": len(df),
                "new_positions": df.where(pd.notnull(df), other=None).to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/smart-money/overlap", summary="Portfolio overlap between two Tier-1 managers")
    def get_portfolio_overlap(
        cik_a: str = QParam(..., description="CIK of first manager"),
        cik_b: str = QParam(..., description="CIK of second manager"),
    ) -> Dict[str, Any]:
        """Compute Jaccard portfolio similarity between two Tier-1 managers."""
        try:
            overlap = _smart_money.portfolio_overlap(cik_a, cik_b)
            name_a = TIER1_INVESTORS.get(cik_a, cik_a)
            name_b = TIER1_INVESTORS.get(cik_b, cik_b)
            return {
                "manager_a": name_a,
                "manager_b": name_b,
                "jaccard_overlap": overlap,
                "interpretation": (
                    "high overlap (similar positions)" if (overlap or 0) > 0.3
                    else "moderate overlap" if (overlap or 0) > 0.1
                    else "low overlap (different strategies)"
                ),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/cluster-buys", summary="Insider cluster buy screen")
    def get_cluster_buys(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = universe"),
        days: int = QParam(30, ge=7, le=90, description="Rolling window in days"),
        min_insiders: int = QParam(3, ge=2, le=10, description="Minimum insider buyer count"),
    ) -> Dict[str, Any]:
        """Screen for stocks with cluster insider buying signals."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _insider_signal.screen_cluster_buys(universe, days=days, min_insiders=min_insiders)
            if df.empty:
                return {"n_results": 0, "cluster_signals": []}
            df = df.where(pd.notnull(df), other=None).reset_index()
            return {
                "n_results": len(df),
                "window_days": days,
                "min_insiders": min_insiders,
                "cluster_signals": df.to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/insider/{ticker}", summary="Insider transaction history")
    def get_insider_history(ticker: str) -> Dict[str, Any]:
        """Return recent open-market insider transactions for a ticker."""
        try:
            df = _insider_signal.get_insider_history(ticker.upper())
            if df.empty:
                return {"ticker": ticker.upper(), "n_transactions": 0, "transactions": []}
            df = df.where(pd.notnull(df), other=None)
            return {
                "ticker": ticker.upper(),
                "n_transactions": len(df),
                "transactions": df.to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/activists", summary="Recent activist 13D filings screen")
    def get_activists(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = universe"),
        days: int = QParam(90, ge=30, le=365, description="Lookback window in days"),
    ) -> Dict[str, Any]:
        """Screen for stocks with recent activist SC 13D filings."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _activist_monitor.monitor_universe_activists(universe, days=days)
            if df.empty:
                return {"n_results": 0, "activist_filings": []}
            df = df.where(pd.notnull(df), other=None)
            return {
                "n_results": len(df),
                "lookback_days": days,
                "activist_filings": df.to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/activists/{ticker}", summary="Activist filings for a specific ticker")
    def get_ticker_activists(
        ticker: str,
        days: int = QParam(180, ge=30, le=365),
    ) -> Dict[str, Any]:
        """Return SC 13D/13G filings for a specific ticker."""
        try:
            filings = _activist_monitor.search_13dg_filings(ticker.upper(), days=days)
            is_conversion = _activist_monitor.detect_13g_to_13d_conversion(ticker.upper(), days=days)
            return {
                "ticker": ticker.upper(),
                "n_filings": len(filings),
                "passive_to_activist_conversion": is_conversion,
                "filings": filings,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/concentration/{ticker}", summary="Ownership concentration metrics")
    def get_concentration(ticker: str) -> Dict[str, Any]:
        """Return full ownership concentration profile: institutional %, short %, float turnover."""
        try:
            metrics = _concentration_analyzer.compute_concentration_metrics(ticker.upper())
            hhi = _concentration_analyzer.compute_ownership_herfindahl(ticker.upper())
            metrics["herfindahl_index"] = hhi
            return metrics
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/screener", summary="Multi-criteria ownership screen")
    def run_screener(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = default universe"),
        min_institutional_pct: float = QParam(20.0, ge=0, le=100),
        max_institutional_pct: float = QParam(95.0, ge=0, le=100),
        has_activist: bool = QParam(False),
        max_short_pct: float = QParam(50.0, ge=0, le=100),
        min_smart_money_count: int = QParam(0, ge=0),
        top_n: int = QParam(20, ge=1, le=50),
    ) -> Dict[str, Any]:
        """Run multi-criteria ownership screener and return top ranked stocks."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _screener.screen(
                tickers=universe,
                min_institutional_pct=min_institutional_pct,
                max_institutional_pct=max_institutional_pct,
                has_activist=has_activist,
                max_short_pct=max_short_pct,
                min_smart_money_count=min_smart_money_count,
                top_n=top_n,
            )
            if df.empty:
                return {"n_results": 0, "results": []}
            df = df.where(pd.notnull(df), other=None).reset_index()
            return {
                "n_results": len(df),
                "results": df.to_dict(orient="records"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/crowding", summary="Hedge fund crowding index")
    def get_crowding(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers; None = default universe"),
    ) -> Dict[str, Any]:
        """Return hedge fund crowding index and risk summary for the universe."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            report = _crowding.crowding_risk_report(universe)
            df = _crowding.compute_crowding_index(universe)
            if not df.empty:
                report["crowding_table"] = df.where(pd.notnull(df), other=None).reset_index().to_dict(orient="records")
            return report
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/crowding/squeeze-candidates", summary="Short squeeze candidates")
    def get_squeeze_candidates(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers"),
        min_short_pct: float = QParam(15.0, ge=5, le=80),
        require_insider_buy: bool = QParam(True),
    ) -> Dict[str, Any]:
        """Identify short squeeze candidates: high short + low institutional + insider buy."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _crowding.find_short_squeeze_candidates(
                universe, min_short_pct=min_short_pct, require_insider_buy=require_insider_buy
            )
            if df.empty:
                return {"n_results": 0, "candidates": []}
            df = df.where(pd.notnull(df), other=None).reset_index()
            return {"n_results": len(df), "candidates": df.to_dict(orient="records")}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/crowding/uncrowded-quality", summary="Uncrowded quality stocks")
    def get_uncrowded_quality(
        tickers: Optional[str] = QParam(None, description="Comma-separated tickers"),
        max_inst_pct: float = QParam(60.0, ge=10, le=90),
        min_roe: float = QParam(15.0, ge=5, le=50),
    ) -> Dict[str, Any]:
        """Find high-quality stocks not yet crowded by institutional ownership."""
        try:
            universe = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
            df = _crowding.find_uncrowded_quality(universe, max_inst_pct=max_inst_pct, min_roe=min_roe)
            if df.empty:
                return {"n_results": 0, "results": []}
            df = df.where(pd.notnull(df), other=None).reset_index()
            return {"n_results": len(df), "results": df.to_dict(orient="records")}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @ownership_router.get("/tier1-managers", summary="List all Tier-1 managers and CIKs")
    def list_tier1_managers() -> Dict[str, Any]:
        """Return the list of 30 Tier-1 institutional managers tracked by this module."""
        return {
            "n_managers": len(TIER1_INVESTORS),
            "managers": [
                {"cik": cik, "name": name}
                for cik, name in TIER1_INVESTORS.items()
            ],
        }

    @ownership_router.get("/manager/{cik}/holdings", summary="Tier-1 manager top holdings")
    def get_manager_holdings(
        cik: str,
        top_n: int = QParam(20, ge=1, le=100),
    ) -> Dict[str, Any]:
        """Return current 13F holdings for a Tier-1 manager by CIK."""
        try:
            manager_name = TIER1_INVESTORS.get(cik, cik)
            df = _smart_money.get_current_holdings(cik)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No holdings found for CIK {cik}")

            total_val = df["value_usd"].sum() or 1
            df["pct_portfolio"] = df["value_usd"].apply(
                lambda v: round(float(v or 0) / total_val * 100, 2)
            )
            df = df.nlargest(top_n, "value_usd")
            df = df.where(pd.notnull(df), other=None)

            return {
                "manager_cik": cik,
                "manager_name": manager_name,
                "n_positions": len(df),
                "portfolio_value_bn": round(total_val / 1e9, 2),
                "holdings": df.to_dict(orient="records"),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

except ImportError:
    ownership_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available; ownership_router not registered")
