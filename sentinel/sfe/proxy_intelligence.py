"""
Proxy / DEF 14A Intelligence — Dimension #028 (target 9+).

Comprehensive proxy filing intelligence using EDGAR free data sources.
Covers executive compensation parsing, governance scoring, voting analytics,
board composition, say-on-pay trends, and peer compensation benchmarking.

Public API
----------
ProxyFilingAdapter
    get_proxy_filings(cik, lookback_years)          -> list[dict]
    get_proxy_text(accession_number, cik)           -> str
    get_proxy_exhibits(accession_number)            -> list[dict]

ExecutiveCompensationParser
    parse_summary_comp_table(proxy_text)            -> pd.DataFrame
    parse_ceo_pay_ratio(proxy_text)                 -> dict
    parse_pay_vs_performance(proxy_text)            -> pd.DataFrame
    compute_compensation_metrics(comp_df, ...)      -> dict

GovernanceScorer
    parse_board_composition(proxy_text)             -> dict
    parse_shareholder_proposals(proxy_text)         -> list[dict]
    parse_antitakeover_provisions(proxy_text)       -> dict
    score_governance(proxy_text, cik)               -> dict

VotingAnalyticsEngine
    get_vote_results(cik, year)                     -> list[dict]
    parse_say_on_pay_result(vote_8k_text)           -> dict
    compute_director_election_support(...)          -> pd.DataFrame
    track_governance_trend(cik, years)              -> dict

ProxyCompAnalytics
    peer_compensation_benchmark(ticker, cik, ...)   -> pd.DataFrame
    sector_governance_screen(sic_code)              -> pd.DataFrame

FastAPI router: proxy_router
"""
from __future__ import annotations

import asyncio
import html
import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_SUBMISSIONS  = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data"
EDGAR_DATA         = "https://data.sec.gov"
COMPANY_TICKERS    = "https://www.sec.gov/files/company_tickers.json"
COMPANY_TICKERS_EX = "https://www.sec.gov/files/company_tickers_exchange.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT    = 30.0
_RATE_DELAY = 0.15   # 150 ms between EDGAR requests (SEC fair-access policy)

PROXY_FORMS = {"DEF 14A", "DEFA14A", "PRE 14A", "PREM14A", "DEFC14A"}

# Stress-scenario date windows (approximate peak-drawdown periods)
_STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008_GFC":         ("2008-09-01", "2009-03-09"),
    "2020_COVID":       ("2020-02-19", "2020-03-23"),
    "2000_DotCom":      ("2000-03-10", "2002-10-09"),
    "1998_LTCM":        ("1998-07-17", "1998-10-08"),
    "2022_RateShock":   ("2022-01-03", "2022-10-12"),
    "2011_EUDebt":      ("2011-05-02", "2011-10-03"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    """Convert HTML/SGML filing text to plain text."""
    raw = html.unescape(raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"&\w+;", " ", raw)
    raw = re.sub(r"\s{3,}", "\n\n", raw)
    return raw.strip()


def _normalize_accession(acc: str) -> str:
    """Normalise accession number to dashed form (0001234567-24-000001)."""
    acc = acc.strip().replace("/", "-")
    if re.match(r"^\d{18}$", acc.replace("-", "")):
        digits = acc.replace("-", "")
        acc = f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return acc


async def _get(client: httpx.AsyncClient, url: str, **kw) -> httpx.Response:
    """Rate-limited EDGAR GET with retry."""
    await asyncio.sleep(_RATE_DELAY)
    for attempt in range(3):
        try:
            r = await client.get(url, timeout=_TIMEOUT, **kw)
            r.raise_for_status()
            return r
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """Resolve ticker symbol to zero-padded 10-digit CIK via EDGAR tickers JSON."""
    import urllib.request
    url = COMPANY_TICKERS_EX
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        # data structure: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
        fields = data.get("fields", [])
        ticker_idx = fields.index("ticker") if "ticker" in fields else 2
        cik_idx    = fields.index("cik")    if "cik"    in fields else 0
        ticker_up  = ticker.upper()
        for row in data.get("data", []):
            if str(row[ticker_idx]).upper() == ticker_up:
                return str(row[cik_idx]).zfill(10)
    except Exception as exc:
        logger.warning("ticker_to_cik failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ProxyFiling(BaseModel):
    accession_number: str
    form_type: str
    filing_date: str
    period_of_report: Optional[str] = None
    cik: str
    company_name: Optional[str] = None
    primary_doc_url: Optional[str] = None


class BoardDirector(BaseModel):
    name: str
    independent: bool = True
    tenure_years: Optional[float] = None
    age: Optional[int] = None
    committees: list[str] = Field(default_factory=list)
    gender: Optional[str] = None


class ShareholderProposal(BaseModel):
    number: int
    description: str
    proposer: str            # "management" | "shareholder"
    vote_required: str       # "majority" | "supermajority" | "plurality"
    board_recommendation: str
    esg_related: bool = False
    say_on_pay: bool = False
    director_election: bool = False


class GovernanceScore(BaseModel):
    board_independence_score: float       # 0-30
    board_diversity_score: float          # 0-20
    compensation_alignment_score: float   # 0-25
    shareholder_rights_score: float       # 0-25
    total_score: float                    # 0-100
    letter_grade: str
    flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ProxyFilingAdapter
# ---------------------------------------------------------------------------

class ProxyFilingAdapter:
    """Download and parse DEF 14A proxy filings from EDGAR."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(headers=_HEADERS, follow_redirects=True)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    async def get_proxy_filings(
        self,
        cik: str,
        lookback_years: int = 5,
    ) -> list[dict]:
        """
        Return proxy filings for *cik* via EDGAR submissions API.

        Filters form_type in {DEF 14A, DEFA14A, PRE 14A, PREM14A, DEFC14A}.
        """
        cik_padded = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"
        client = self._client or httpx.AsyncClient(headers=_HEADERS, follow_redirects=True)
        try:
            r = await _get(client, url)
            data = r.json()
        finally:
            if not self._client:
                await client.aclose()

        filings: list[dict] = []
        cutoff = datetime.utcnow() - timedelta(days=lookback_years * 365)

        recent = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        acc_nums    = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        periods     = recent.get("reportDate", [])
        documents   = recent.get("primaryDocument", [])

        company_name = data.get("name", "")

        for form, acc, filed, period, doc in zip(
            forms, acc_nums, filed_dates, periods, documents
        ):
            if form not in PROXY_FORMS:
                continue
            try:
                filed_dt = datetime.strptime(filed, "%Y-%m-%d")
            except ValueError:
                continue
            if filed_dt < cutoff:
                continue

            acc_dashed = _normalize_accession(acc)
            acc_nodash = acc_dashed.replace("-", "")
            cik_int    = str(int(cik))
            primary_url = (
                f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{doc}"
                if doc else None
            )
            filings.append(
                ProxyFiling(
                    accession_number=acc_dashed,
                    form_type=form,
                    filing_date=filed,
                    period_of_report=period or None,
                    cik=cik,
                    company_name=company_name,
                    primary_doc_url=primary_url,
                ).model_dump()
            )

        return filings

    async def get_proxy_text(self, accession_number: str, cik: str) -> str:
        """
        Download primary DEF 14A document and return plain text (HTML stripped).
        """
        acc_dashed = _normalize_accession(accession_number)
        acc_nodash = acc_dashed.replace("-", "")
        cik_int    = str(int(cik))

        # Fetch filing index to find the primary document
        index_url = (
            f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{acc_dashed}-index.htm"
        )
        client = self._client or httpx.AsyncClient(headers=_HEADERS, follow_redirects=True)
        try:
            try:
                r = await _get(client, index_url)
                index_html = r.text
            except Exception:
                index_html = ""

            # Extract primary document filename from index
            primary_doc = self._find_primary_doc(index_html)
            if not primary_doc:
                # fallback: try .txt full-submission
                primary_doc = f"{acc_dashed}.txt"

            doc_url = f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{primary_doc}"
            r2 = await _get(client, doc_url)
            raw = r2.text
        finally:
            if not self._client:
                await client.aclose()

        return _strip_html(raw)

    async def get_proxy_exhibits(self, accession_number: str) -> list[dict]:
        """
        List all exhibits in this filing: compensation tables, governance docs.
        """
        # Parse the filing index JSON
        acc_dashed = _normalize_accession(accession_number)
        acc_nodash = acc_dashed.replace("-", "")
        # CIK is embedded in accession number (first 10 digits)
        cik_int = str(int(acc_nodash[:10]))
        index_json_url = (
            f"{EDGAR_DATA}/submissions/CIK{cik_int.zfill(10)}.json"
        )
        # Simpler: fetch EDGAR filing index page
        index_url = (
            f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{acc_dashed}-index.json"
        )
        client = self._client or httpx.AsyncClient(headers=_HEADERS, follow_redirects=True)
        exhibits: list[dict] = []
        try:
            try:
                r = await _get(client, index_url)
                data = r.json()
                for item in data.get("directory", {}).get("item", []):
                    name = item.get("name", "")
                    desc = item.get("type", "")
                    exhibits.append(
                        {
                            "filename": name,
                            "type": desc,
                            "url": f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{name}",
                            "is_compensation_exhibit": any(
                                kw in name.lower() or kw in desc.lower()
                                for kw in ["comp", "exhibit 99", "ex-99"]
                            ),
                            "is_governance_doc": any(
                                kw in name.lower() or kw in desc.lower()
                                for kw in ["charter", "bylaw", "governance", "code"]
                            ),
                        }
                    )
            except Exception:
                pass
        finally:
            if not self._client:
                await client.aclose()

        return exhibits

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_primary_doc(index_html: str) -> Optional[str]:
        """Extract primary document filename from EDGAR filing index HTML."""
        # Match links with DEF 14A type
        pattern = re.compile(
            r'<td[^>]*>\s*(?:DEF 14A|DEFA14A|PRE 14A)\s*</td>'
            r'.*?<a[^>]+href="([^"]+)"',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(index_html)
        if m:
            href = m.group(1)
            return href.split("/")[-1]
        # Fallback: first .htm file
        m2 = re.search(r'href="[^"]+/([^/"]+\.htm)"', index_html, re.IGNORECASE)
        return m2.group(1) if m2 else None


# ---------------------------------------------------------------------------
# ExecutiveCompensationParser
# ---------------------------------------------------------------------------

class ExecutiveCompensationParser:
    """
    Parse executive compensation tables from raw DEF 14A plain text.

    The SEC mandates a specific Summary Compensation Table format (Reg S-K §402).
    We use regex heuristics robust to formatting variations.
    """

    _SCT_HEADER_RE = re.compile(
        r"SUMMARY\s+COMPENSATION\s+TABLE",
        re.IGNORECASE,
    )
    _DOLLAR_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)")
    _NAME_YEAR_RE = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+"       # Name
        r"(\d{4})\s+"                                  # Year
        r"([\d,]+)\s+"                                 # Salary
        r"([\d,]+)?\s*"                                # Bonus
        r"([\d,]+)?\s*"                                # Stock awards
        r"([\d,]+)?\s*"                                # Option awards
        r"([\d,]+)?\s*"                                # NEIP
        r"([\d,]+)?\s*"                                # Pension
        r"([\d,]+)?\s*"                                # Other
        r"([\d,]+)",                                   # Total
    )
    _CEO_RATIO_RE = re.compile(
        r"CEO\s+(?:annual\s+)?(?:total\s+)?compensation[:\s]*\$?([\d,]+)"
        r".*?"
        r"median[^$]*\$?([\d,]+)"
        r".*?"
        r"ratio[^0-9]*(\d+)\s*(?:to\s*1|:1|x)",
        re.IGNORECASE | re.DOTALL,
    )
    _PVP_YEAR_RE = re.compile(
        r"(\d{4})\s+"
        r"\$?([\d,]+(?:\.\d+)?)\s+"   # SCT total
        r"\$?([\d,]+(?:\.\d+)?)\s+"   # CAP
        r"\$?([\d,]+(?:\.\d+)?)\s+"   # Peer CAP (optional col)
        r"\$?([\d,.]+)",               # TSR or Net income
    )

    @staticmethod
    def _parse_dollars(s: str) -> float:
        s = s.replace(",", "").replace("$", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    def parse_summary_comp_table(self, proxy_text: str) -> pd.DataFrame:
        """
        Extract the SEC-mandated Summary Compensation Table.

        Columns: name, title, year, salary, bonus, stock_awards, option_awards,
                 non_equity_incentive, pension_change, all_other_comp, total
        Returns up to 3 years × 5 NEOs = 15 rows.
        """
        rows: list[dict] = []

        # Locate the SCT section
        m = self._SCT_HEADER_RE.search(proxy_text)
        if not m:
            logger.debug("Summary Compensation Table header not found")
            return pd.DataFrame(columns=[
                "name", "title", "year", "salary", "bonus",
                "stock_awards", "option_awards", "non_equity_incentive",
                "pension_change", "all_other_comp", "total",
            ])

        section = proxy_text[m.start(): m.start() + 8000]

        # Pattern A: structured rows with name, year, and dollar columns
        pattern_a = re.compile(
            r"([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+)"  # name
            r"(?:[^\n]{0,60}\n)?"                               # optional title line
            r"\s*(20\d{2})\s+"                                  # year
            r"\$?\s*([\d,]+)"                                   # salary
            r"(?:\s+\$?\s*([\d,]*))?",                          # bonus (optional)
            re.MULTILINE,
        )
        for m2 in pattern_a.finditer(section):
            name = m2.group(1).strip()
            year = int(m2.group(2))
            salary = self._parse_dollars(m2.group(3) or "0")
            bonus  = self._parse_dollars(m2.group(4) or "0")
            if salary < 1:
                continue
            rows.append(
                {
                    "name": name,
                    "title": "",
                    "year": year,
                    "salary": salary,
                    "bonus": bonus,
                    "stock_awards": 0.0,
                    "option_awards": 0.0,
                    "non_equity_incentive": 0.0,
                    "pension_change": 0.0,
                    "all_other_comp": 0.0,
                    "total": salary + bonus,
                }
            )

        # Pattern B: full numeric rows (tab/whitespace separated)
        full_row_re = re.compile(
            r"([A-Z][a-z]+(?: \w+){1,3})\s+"
            r"(20\d{2})\s+"
            r"([\d,]+)\s+"
            r"([\d,]*)\s+"
            r"([\d,]*)\s+"
            r"([\d,]*)\s+"
            r"([\d,]*)\s+"
            r"([\d,]*)\s+"
            r"([\d,]*)\s+"
            r"([\d,]+)",
        )
        for m3 in full_row_re.finditer(section):
            g = m3.groups()
            rows.append(
                {
                    "name": g[0].strip(),
                    "title": "",
                    "year": int(g[1]),
                    "salary": self._parse_dollars(g[2]),
                    "bonus": self._parse_dollars(g[3]),
                    "stock_awards": self._parse_dollars(g[4]),
                    "option_awards": self._parse_dollars(g[5]),
                    "non_equity_incentive": self._parse_dollars(g[6]),
                    "pension_change": self._parse_dollars(g[7]),
                    "all_other_comp": self._parse_dollars(g[8]),
                    "total": self._parse_dollars(g[9]),
                }
            )

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[
                "name", "title", "year", "salary", "bonus",
                "stock_awards", "option_awards", "non_equity_incentive",
                "pension_change", "all_other_comp", "total",
            ]
        )
        if not df.empty:
            df = df.drop_duplicates(subset=["name", "year"]).sort_values(
                ["year", "name"], ascending=[False, True]
            ).reset_index(drop=True)
        return df

    def parse_ceo_pay_ratio(self, proxy_text: str) -> dict:
        """
        Extract CEO Pay Ratio (required since 2018 per SEC Reg S-K §953(b)).

        Returns: {ceo_pay, median_employee_pay, ratio, disclosure_year}
        """
        result = {
            "ceo_pay": None,
            "median_employee_pay": None,
            "ratio": None,
            "disclosure_year": None,
        }

        # Approach 1: structured pattern
        m = self._CEO_RATIO_RE.search(proxy_text)
        if m:
            result["ceo_pay"] = self._parse_dollars(m.group(1))
            result["median_employee_pay"] = self._parse_dollars(m.group(2))
            result["ratio"] = int(m.group(3))
            return result

        # Approach 2: look for "X to 1" or "X:1" near "pay ratio"
        ratio_section_re = re.compile(
            r"(?:ceo\s+pay\s+ratio|pay\s+ratio\s+disclosure)[^\n]*\n"
            r"(?:.*\n){0,10}",
            re.IGNORECASE,
        )
        ms = ratio_section_re.search(proxy_text)
        if ms:
            chunk = proxy_text[ms.start(): ms.start() + 1000]
            ratio_m = re.search(r"(\d+)\s*(?:to\s*1|:1|x)", chunk, re.IGNORECASE)
            dollar_matches = re.findall(r"\$\s*([\d,]+)", chunk)
            if ratio_m:
                result["ratio"] = int(ratio_m.group(1))
            if len(dollar_matches) >= 2:
                vals = sorted([self._parse_dollars(d) for d in dollar_matches], reverse=True)
                result["ceo_pay"] = vals[0]
                result["median_employee_pay"] = vals[1]

        return result

    def parse_pay_vs_performance(self, proxy_text: str) -> pd.DataFrame:
        """
        Extract Pay vs Performance table (SEC Reg S-K §402(v), required since 2023).

        Returns DataFrame with: year, sct_total_ceo, cap_ceo, peer_cap,
        company_tsr, peer_tsr, net_income
        """
        rows: list[dict] = []

        pvp_header = re.search(
            r"PAY\s+VERSUS?\s+PERFORMANCE|PAY\s+VS\.?\s+PERFORMANCE",
            proxy_text,
            re.IGNORECASE,
        )
        if not pvp_header:
            return pd.DataFrame(columns=[
                "year", "sct_total_ceo", "cap_ceo", "peer_cap",
                "company_tsr", "peer_tsr", "net_income",
            ])

        section = proxy_text[pvp_header.start(): pvp_header.start() + 5000]

        # Parse rows: year + 4-6 dollar columns
        row_re = re.compile(
            r"(20\d{2})\s+"
            r"\$?([\d,]+(?:\.\d+)?)\s+"
            r"\$?(-?[\d,]+(?:\.\d+)?)\s+"
            r"\$?(-?[\d,]+(?:\.\d+)?)\s+"
            r"\$?([\d,.]+)(?:\s+\$?([\d,.]+))?(?:\s+\$?(-?[\d,.]+))?",
        )
        for m in row_re.finditer(section):
            g = m.groups()
            rows.append(
                {
                    "year": int(g[0]),
                    "sct_total_ceo": self._parse_dollars(g[1] or "0"),
                    "cap_ceo": self._parse_dollars(g[2] or "0"),
                    "peer_cap": self._parse_dollars(g[3] or "0"),
                    "company_tsr": self._parse_dollars(g[4] or "0"),
                    "peer_tsr": self._parse_dollars(g[5] or "0") if g[5] else None,
                    "net_income": self._parse_dollars(g[6] or "0") if g[6] else None,
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[
                "year", "sct_total_ceo", "cap_ceo", "peer_cap",
                "company_tsr", "peer_tsr", "net_income",
            ]
        )

    def compute_compensation_metrics(
        self,
        comp_df: pd.DataFrame,
        revenue: float = None,
        market_cap: float = None,
    ) -> dict:
        """
        Derive institutional-grade compensation metrics and flag red flags.

        Returns: {ceo_pay_pct_revenue, ceo_pay_pct_market_cap,
                  total_exec_comp, stock_vs_cash_ratio, red_flags}
        """
        if comp_df.empty:
            return {
                "ceo_pay_pct_revenue": None,
                "ceo_pay_pct_market_cap": None,
                "total_exec_comp": None,
                "stock_vs_cash_ratio": None,
                "red_flags": ["no_compensation_data"],
            }

        # Most recent year
        latest_year = comp_df["year"].max()
        latest = comp_df[comp_df["year"] == latest_year]

        # CEO is typically the highest-paid executive
        ceo_row = latest.sort_values("total", ascending=False).iloc[0]
        ceo_pay = ceo_row.get("total", 0.0)

        # Top-5 NEO total
        top5 = latest.nlargest(5, "total")
        total_exec_comp = top5["total"].sum()

        # Stock vs cash ratio for CEO
        stock_comp = ceo_row.get("stock_awards", 0.0) + ceo_row.get("option_awards", 0.0)
        cash_comp   = ceo_row.get("salary", 0.0) + ceo_row.get("bonus", 0.0) + ceo_row.get("non_equity_incentive", 0.0)
        stock_vs_cash_ratio = (
            stock_comp / cash_comp if cash_comp > 0 else float("inf")
        )

        metrics: dict[str, Any] = {
            "ceo_pay": ceo_pay,
            "ceo_pay_pct_revenue": None,
            "ceo_pay_pct_market_cap": None,
            "total_exec_comp": total_exec_comp,
            "stock_vs_cash_ratio": round(stock_vs_cash_ratio, 3),
            "red_flags": [],
        }

        if revenue and ceo_pay > 0:
            pct = ceo_pay / revenue * 100
            metrics["ceo_pay_pct_revenue"] = round(pct, 4)
            if pct > 1.0:
                metrics["red_flags"].append("ceo_pay_exceeds_1pct_revenue")

        if market_cap and ceo_pay > 0:
            pct_mc = ceo_pay / market_cap * 100
            metrics["ceo_pay_pct_market_cap"] = round(pct_mc, 4)

        # Red flag: >50% cash (no equity alignment)
        total_comp = stock_comp + cash_comp
        if total_comp > 0 and (cash_comp / total_comp) > 0.5:
            metrics["red_flags"].append("majority_cash_compensation_no_equity_alignment")

        # Red flag: declining revenue + rising pay (need multi-year data)
        if len(comp_df["year"].unique()) >= 2:
            prior_year = sorted(comp_df["year"].unique())[-2]
            prior_ceo = comp_df[comp_df["year"] == prior_year].sort_values(
                "total", ascending=False
            )
            if not prior_ceo.empty:
                prior_pay = prior_ceo.iloc[0]["total"]
                if ceo_pay > prior_pay * 1.05:
                    metrics["red_flags"].append("ceo_pay_rising_year_over_year")

        return metrics


# ---------------------------------------------------------------------------
# GovernanceScorer
# ---------------------------------------------------------------------------

class GovernanceScorer:
    """
    Score corporate governance from DEF 14A proxy text.
    Uses ISS / Glass Lewis framework adapted for free data.
    """

    _INDEPENDENCE_RE = re.compile(
        r"\b(?:independent|non-executive|outside)\b",
        re.IGNORECASE,
    )
    _GENDER_F_RE = re.compile(r"\b(?:Ms\.|Mrs\.|she/her|female)\b", re.IGNORECASE)
    _GENDER_M_RE = re.compile(r"\b(?:Mr\.|he/him|male)\b", re.IGNORECASE)
    _AGE_RE       = re.compile(r"age\s+(\d{2})", re.IGNORECASE)
    _STAGGERED_RE = re.compile(
        r"\bstaggered\s+board|\bclassified\s+board", re.IGNORECASE
    )
    _POISON_PILL_RE = re.compile(
        r"\bpoison\s+pill|\bshareholder\s+rights\s+plan|\brights\s+plan\b",
        re.IGNORECASE,
    )
    _SUPERMAJORITY_RE = re.compile(
        r"\bsupermajority\b|\b(?:66|75|80)\s*%\s*vote", re.IGNORECASE
    )
    _DUAL_CLASS_RE = re.compile(
        r"\bdual[\-\s]class|\bClass\s+[AB]\s+(?:common\s+)?shares?\b|\bmulti[\-\s]class",
        re.IGNORECASE,
    )
    _ADVANCE_NOTICE_RE = re.compile(r"\badvance\s+notice\b", re.IGNORECASE)
    _CLAWBACK_RE  = re.compile(r"\bclawback\b|\brecoupment\b", re.IGNORECASE)
    _PERF_METRIC_RE = re.compile(
        r"\b(?:TSR|total\s+shareholder\s+return|ROIC|EPS|revenue\s+growth|EBITDA|free\s+cash\s+flow)\b",
        re.IGNORECASE,
    )

    def parse_board_composition(self, proxy_text: str) -> dict:
        """
        Extract director roster and compute governance statistics.
        """
        # Find board section
        board_section_m = re.search(
            r"(?:DIRECTORS|BOARD\s+OF\s+DIRECTORS|DIRECTOR\s+NOMINEES)",
            proxy_text,
            re.IGNORECASE,
        )
        if not board_section_m:
            return {
                "directors": [],
                "board_size": 0,
                "independent_pct": None,
                "avg_tenure": None,
                "female_count": 0,
                "male_count": 0,
                "avg_age": None,
            }

        section = proxy_text[board_section_m.start(): board_section_m.start() + 10000]
        directors: list[dict] = []

        # Director block pattern: Name, age, independent status
        dir_re = re.compile(
            r"([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+)"
            r"(?:[^\n]{0,100}\n){0,3}"
            r"(?:.*?(?:independent|non-executive|executive|chairman)[^\n]*\n)?",
            re.IGNORECASE,
        )
        for m in dir_re.finditer(section[:5000]):
            name = m.group(1).strip()
            if len(name) < 5 or name.lower() in {"the board", "our board", "the company"}:
                continue
            chunk = section[m.start(): m.start() + 400]
            is_independent = bool(self._INDEPENDENCE_RE.search(chunk))
            gender_f = bool(self._GENDER_F_RE.search(chunk))
            age_m    = self._AGE_RE.search(chunk)
            age      = int(age_m.group(1)) if age_m else None
            # Tenure: look for "X years" near name
            tenure_m = re.search(r"(\d{1,2})\s+years?", chunk, re.IGNORECASE)
            tenure   = float(tenure_m.group(1)) if tenure_m else None
            # Committees
            cmt_m    = re.findall(
                r"\b(Audit|Compensation|Nominating|Governance|Risk|Finance)\b",
                chunk,
                re.IGNORECASE,
            )
            directors.append(
                {
                    "name": name,
                    "independent": is_independent,
                    "age": age,
                    "tenure_years": tenure,
                    "gender": "F" if gender_f else "M",
                    "committees": list(set(cmt_m)),
                }
            )
            if len(directors) >= 15:
                break

        board_size = len(directors) if directors else 0
        independent_pct = (
            sum(1 for d in directors if d["independent"]) / board_size * 100
            if board_size else None
        )
        ages = [d["age"] for d in directors if d["age"]]
        tenures = [d["tenure_years"] for d in directors if d["tenure_years"]]

        return {
            "directors": directors,
            "board_size": board_size,
            "independent_pct": round(independent_pct, 1) if independent_pct else None,
            "avg_tenure": round(float(np.mean(tenures)), 1) if tenures else None,
            "female_count": sum(1 for d in directors if d["gender"] == "F"),
            "male_count": sum(1 for d in directors if d["gender"] == "M"),
            "avg_age": round(float(np.mean(ages)), 1) if ages else None,
        }

    def parse_shareholder_proposals(self, proxy_text: str) -> list[dict]:
        """
        Extract all proxy proposals with metadata.
        """
        proposals: list[dict] = []

        # Look for numbered proposal blocks
        prop_re = re.compile(
            r"(?:PROPOSAL|ITEM)\s+(?:NO\.?\s*)?(\d+)[:\s—–-]+"
            r"([^\n]{20,200})",
            re.IGNORECASE,
        )
        for i, m in enumerate(prop_re.finditer(proxy_text)):
            number = int(m.group(1))
            desc   = m.group(2).strip()

            # Look ahead for recommendation and vote requirement
            chunk = proxy_text[m.start(): m.start() + 1500]
            board_rec = "FOR"
            if re.search(r"\bboard\s+recommends?\s+(?:a\s+vote\s+)?AGAINST\b", chunk, re.IGNORECASE):
                board_rec = "AGAINST"
            elif re.search(r"\bboard\s+recommends?\s+(?:a\s+vote\s+)?FOR\b", chunk, re.IGNORECASE):
                board_rec = "FOR"

            is_shareholder = bool(
                re.search(r"(?:submitted|proposed|sponsored)\s+by\s+(?:a\s+)?shareholder", chunk, re.IGNORECASE)
                or re.search(r"shareholder\s+proposal", desc, re.IGNORECASE)
            )
            vote_req = "majority"
            if re.search(r"supermajority|two-thirds|66\s*%", chunk, re.IGNORECASE):
                vote_req = "supermajority"
            elif re.search(r"plurality", chunk, re.IGNORECASE):
                vote_req = "plurality"

            is_esg = bool(re.search(
                r"\b(?:climate|environmental|social|diversity|human\s+rights|DEI|sustainability|ESG)\b",
                desc,
                re.IGNORECASE,
            ))
            is_sop = bool(re.search(
                r"say[-\s]on[-\s]pay|executive\s+compensation\s+advisory|advisory\s+vote",
                desc,
                re.IGNORECASE,
            ))
            is_dir = bool(re.search(
                r"elect(?:ion)?\s+of\s+director|director\s+election",
                desc,
                re.IGNORECASE,
            ))
            proposals.append(
                {
                    "number": number,
                    "description": desc,
                    "proposer": "shareholder" if is_shareholder else "management",
                    "vote_required": vote_req,
                    "board_recommendation": board_rec,
                    "esg_related": is_esg,
                    "say_on_pay": is_sop,
                    "director_election": is_dir,
                }
            )

        return proposals

    def parse_antitakeover_provisions(self, proxy_text: str) -> dict:
        """
        Detect anti-shareholder / entrenchment provisions.
        Each True = worse for shareholders.
        """
        return {
            "staggered_board": bool(self._STAGGERED_RE.search(proxy_text)),
            "poison_pill": bool(self._POISON_PILL_RE.search(proxy_text)),
            "supermajority_requirements": bool(self._SUPERMAJORITY_RE.search(proxy_text)),
            "dual_class_shares": bool(self._DUAL_CLASS_RE.search(proxy_text)),
            "advance_notice_provisions": bool(self._ADVANCE_NOTICE_RE.search(proxy_text)),
            "has_clawback_policy": bool(self._CLAWBACK_RE.search(proxy_text)),
            "uses_performance_metrics": bool(self._PERF_METRIC_RE.search(proxy_text)),
        }

    def score_governance(self, proxy_text: str, cik: str = None) -> dict:
        """
        Compute governance scorecard (0-100) with ISS-aligned methodology.

        Sub-scores:
          Board independence  (0-30)
          Board diversity     (0-20)
          Comp alignment      (0-25)
          Shareholder rights  (0-25)
        """
        board   = self.parse_board_composition(proxy_text)
        atakeover = self.parse_antitakeover_provisions(proxy_text)
        proposals = self.parse_shareholder_proposals(proxy_text)
        flags: list[str] = []

        # ── Board independence score (0-30) ────────────────────────────────
        ind_pct = board.get("independent_pct") or 50.0
        board_independence_score = min(30.0, ind_pct / 100 * 30)
        if ind_pct < 50:
            flags.append("board_majority_not_independent")

        # ── Board diversity score (0-20) ────────────────────────────────────
        board_size   = board.get("board_size", 0)
        female_count = board.get("female_count", 0)
        female_pct   = (female_count / board_size * 100) if board_size else 0
        gender_pts   = min(12, female_pct / 100 * 12)   # up to 12 pts for gender
        ethnic_signals = sum(1 for kw in [
            r"diverse", r"minority", r"Hispanic", r"Black\b", r"Asian", r"Latino"
        ] if re.search(kw, proxy_text, re.IGNORECASE))
        ethnic_pts = min(8, ethnic_signals * 2)
        board_diversity_score = gender_pts + ethnic_pts
        if female_pct < 20:
            flags.append("low_gender_diversity_lt_20pct")

        # ── Compensation alignment score (0-25) ─────────────────────────────
        comp_align = 0.0
        if atakeover["has_clawback_policy"]:
            comp_align += 8
        if atakeover["uses_performance_metrics"]:
            comp_align += 10
        # Equity-heavy pay: look for stock awards section
        if re.search(r"\bperformance[- ](?:based\s+)?(?:shares?|units?|RSU|PSU)\b", proxy_text, re.IGNORECASE):
            comp_align += 7
        compensation_alignment_score = min(25.0, comp_align)

        # ── Shareholder rights score (0-25) ─────────────────────────────────
        rights = 25.0
        if atakeover["staggered_board"]:
            rights -= 7
            flags.append("staggered_board_detected")
        if atakeover["poison_pill"]:
            rights -= 6
            flags.append("poison_pill_detected")
        if atakeover["supermajority_requirements"]:
            rights -= 5
            flags.append("supermajority_requirements")
        if atakeover["dual_class_shares"]:
            rights -= 7
            flags.append("dual_class_shares")
        shareholder_rights_score = max(0.0, rights)

        # ── Long tenure warning ───────────────────────────────────────────────
        avg_tenure = board.get("avg_tenure") or 0
        if avg_tenure > 10:
            flags.append("high_avg_board_tenure_gt_10yr")

        total_score = (
            board_independence_score
            + board_diversity_score
            + compensation_alignment_score
            + shareholder_rights_score
        )

        # Letter grade
        if total_score >= 85:
            grade = "A"
        elif total_score >= 70:
            grade = "B"
        elif total_score >= 55:
            grade = "C"
        elif total_score >= 40:
            grade = "D"
        else:
            grade = "F"

        return GovernanceScore(
            board_independence_score=round(board_independence_score, 2),
            board_diversity_score=round(board_diversity_score, 2),
            compensation_alignment_score=round(compensation_alignment_score, 2),
            shareholder_rights_score=round(shareholder_rights_score, 2),
            total_score=round(total_score, 2),
            letter_grade=grade,
            flags=flags,
        ).model_dump()


# ---------------------------------------------------------------------------
# VotingAnalyticsEngine
# ---------------------------------------------------------------------------

class VotingAnalyticsEngine:
    """
    Parse actual shareholder vote results from SEC 8-K Item 5.07 filings.
    """

    async def get_vote_results(
        self,
        cik: str,
        year: int = None,
    ) -> list[dict]:
        """
        Fetch 8-K Item 5.07 filings (results of shareholder voting) from EDGAR.
        """
        cik_padded = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
            r = await _get(client, url)
            data = r.json()

        recent    = data.get("filings", {}).get("recent", {})
        forms     = recent.get("form", [])
        acc_nums  = recent.get("accessionNumber", [])
        dates     = recent.get("filingDate", [])
        items     = recent.get("items", [])

        results: list[dict] = []
        cik_int = str(int(cik))

        for form, acc, filed, item_str in zip(forms, acc_nums, dates, items):
            if form not in {"8-K", "8-K/A"}:
                continue
            if "5.07" not in str(item_str):
                continue
            if year and not filed.startswith(str(year)):
                continue

            acc_dashed = _normalize_accession(acc)
            acc_nodash = acc_dashed.replace("-", "")
            # Try to get the 8-K text
            try:
                async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
                    txt_url = f"{EDGAR_ARCHIVES}/{cik_int}/{acc_nodash}/{acc_dashed}.txt"
                    r2 = await _get(client, txt_url)
                    raw_text = _strip_html(r2.text)
                results.append(
                    {
                        "filing_date": filed,
                        "accession_number": acc_dashed,
                        "item": "5.07",
                        "raw_text": raw_text[:5000],
                    }
                )
            except Exception as exc:
                logger.debug("Could not fetch 8-K %s: %s", acc_dashed, exc)

        return results

    def parse_say_on_pay_result(self, vote_8k_text: str) -> dict:
        """
        Extract say-on-pay vote percentages from 8-K Item 5.07 text.

        Returns: {for_pct, against_pct, abstain_pct, broker_non_votes, passed}
        """
        result = {
            "for_pct": None,
            "against_pct": None,
            "abstain_pct": None,
            "broker_non_votes": None,
            "passed": None,
        }

        # Locate say-on-pay section
        sop_m = re.search(
            r"(?:say[-\s]on[-\s]pay|advisory\s+vote|executive\s+compensation)",
            vote_8k_text,
            re.IGNORECASE,
        )
        if not sop_m:
            return result

        chunk = vote_8k_text[sop_m.start(): sop_m.start() + 1500]

        # Extract vote counts
        for_m     = re.search(r"for[:\s]+([\d,]+)", chunk, re.IGNORECASE)
        against_m = re.search(r"against[:\s]+([\d,]+)", chunk, re.IGNORECASE)
        abstain_m = re.search(r"abstain(?:ed)?[:\s]+([\d,]+)", chunk, re.IGNORECASE)
        broker_m  = re.search(r"broker\s+non[- ]votes?[:\s]+([\d,]+)", chunk, re.IGNORECASE)

        for_votes     = int(for_m.group(1).replace(",", ""))     if for_m     else 0
        against_votes = int(against_m.group(1).replace(",", "")) if against_m else 0
        abstain_votes = int(abstain_m.group(1).replace(",", "")) if abstain_m else 0
        broker_nv     = int(broker_m.group(1).replace(",", ""))  if broker_m  else 0

        total = for_votes + against_votes + abstain_votes
        if total > 0:
            result["for_pct"]     = round(for_votes / total * 100, 2)
            result["against_pct"] = round(against_votes / total * 100, 2)
            result["abstain_pct"] = round(abstain_votes / total * 100, 2)
            result["passed"]      = result["for_pct"] > 50.0

        result["broker_non_votes"] = broker_nv
        return result

    def compute_director_election_support(
        self,
        proxy_text: str,
        vote_results: list[dict] = None,
    ) -> pd.DataFrame:
        """
        Compute per-director election support. Flag < 80% support (ISS standard).
        """
        rows: list[dict] = []

        # Parse director names from proxy proposals
        dir_re = re.compile(
            r"(?:elect(?:ion)?\s+of\s+)?([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+)"
            r"(?:\s+as\s+(?:a\s+)?(?:Class\s+[ABC]\s+)?[Dd]irector)?",
        )

        if vote_results:
            for vr in vote_results:
                text = vr.get("raw_text", "")
                # Find director vote blocks
                vote_block_re = re.compile(
                    r"([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+)\s+"
                    r"([\d,]+)\s+"   # For
                    r"([\d,]+)",     # Withheld / Against
                    re.MULTILINE,
                )
                for m in vote_block_re.finditer(text[:3000]):
                    name         = m.group(1).strip()
                    votes_for    = int(m.group(2).replace(",", ""))
                    votes_withheld = int(m.group(3).replace(",", ""))
                    total_votes  = votes_for + votes_withheld
                    support_pct  = (
                        votes_for / total_votes * 100 if total_votes > 0 else None
                    )
                    rows.append(
                        {
                            "director": name,
                            "votes_for": votes_for,
                            "votes_withheld": votes_withheld,
                            "support_pct": round(support_pct, 2) if support_pct else None,
                            "low_support_flag": support_pct < 80 if support_pct else False,
                            "filing_date": vr.get("filing_date"),
                        }
                    )

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=[
                "director", "votes_for", "votes_withheld",
                "support_pct", "low_support_flag", "filing_date",
            ]
        )
        return df

    async def track_governance_trend(self, cik: str, years: int = 5) -> dict:
        """
        Track year-over-year governance score, say-on-pay trend, board evolution.
        """
        adapter = ProxyFilingAdapter()
        scorer  = GovernanceScorer()
        async with adapter:
            filings = await adapter.get_proxy_filings(cik, lookback_years=years)

        trend: list[dict] = []
        for filing in filings[:years]:
            try:
                async with adapter:
                    text = await adapter.get_proxy_text(
                        filing["accession_number"], cik
                    )
                gov_score = scorer.score_governance(text, cik)
                board     = scorer.parse_board_composition(text)
                trend.append(
                    {
                        "filing_date": filing["filing_date"],
                        "governance_score": gov_score["total_score"],
                        "letter_grade": gov_score["letter_grade"],
                        "board_size": board.get("board_size"),
                        "independent_pct": board.get("independent_pct"),
                        "female_count": board.get("female_count"),
                    }
                )
            except Exception as exc:
                logger.warning("trend error for %s: %s", filing["accession_number"], exc)

        return {
            "cik": cik,
            "years_analyzed": len(trend),
            "trend": trend,
            "score_improving": (
                trend[-1]["governance_score"] > trend[0]["governance_score"]
                if len(trend) >= 2 else None
            ),
        }


# ---------------------------------------------------------------------------
# ProxyCompAnalytics
# ---------------------------------------------------------------------------

class ProxyCompAnalytics:
    """
    Peer compensation benchmarking and sector governance screening.
    """

    async def _get_sic_peers(self, cik: str, n_peers: int = 10) -> list[dict]:
        """Get peer companies by SIC code."""
        cik_padded = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
            r = await _get(client, url)
            data = r.json()

        sic = data.get("sic", "")
        company_name = data.get("name", "")
        if not sic:
            return []

        # Search EDGAR for companies with the same SIC
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=*&dateRange=custom&startdt=2024-01-01&forms=DEF+14A"
            f"&hits.hits._source.sic={sic}&hits.hits.total.value=true"
            f"&hits.hits.hits.total=true"
        )
        peers: list[dict] = []
        try:
            async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
                r2 = await _get(client, search_url)
                hits = r2.json().get("hits", {}).get("hits", [])
                for hit in hits[:n_peers]:
                    src = hit.get("_source", {})
                    peer_cik = src.get("entity_id", "")
                    peer_name = src.get("display_names", [src.get("entity_name", "")])[0]
                    if peer_cik and peer_cik != str(int(cik)):
                        peers.append({"cik": str(peer_cik), "name": peer_name, "sic": sic})
        except Exception as exc:
            logger.warning("peer SIC search failed: %s", exc)

        return peers[:n_peers]

    async def peer_compensation_benchmark(
        self,
        ticker: str,
        cik: str = None,
        n_peers: int = 10,
    ) -> pd.DataFrame:
        """
        Build compensation comparison table for target + SIC peers.

        Columns: company, ceo_pay, cfo_pay (est), pay_ratio, stock_pct,
                 total_exec_comp, filing_date
        """
        if not cik:
            cik = _ticker_to_cik(ticker)
        if not cik:
            return pd.DataFrame()

        adapter = ProxyFilingAdapter()
        parser  = ExecutiveCompensationParser()
        rows: list[dict] = []

        # Target company
        targets = [(cik, ticker)]
        peers   = await self._get_sic_peers(cik, n_peers=n_peers)
        for p in peers:
            targets.append((p["cik"], p["name"]))

        for company_cik, company_name in targets:
            try:
                async with adapter:
                    filings = await adapter.get_proxy_filings(company_cik, lookback_years=2)
                    if not filings:
                        continue
                    text = await adapter.get_proxy_text(
                        filings[0]["accession_number"], company_cik
                    )
                comp_df = parser.parse_summary_comp_table(text)
                metrics = parser.compute_compensation_metrics(comp_df)
                if comp_df.empty:
                    continue
                latest_year = comp_df["year"].max()
                latest = comp_df[comp_df["year"] == latest_year].sort_values(
                    "total", ascending=False
                )
                ceo_pay = latest.iloc[0]["total"] if len(latest) > 0 else None
                cfo_pay = latest.iloc[1]["total"] if len(latest) > 1 else None
                stock_pct = None
                if ceo_pay and latest.iloc[0].get("stock_awards", 0):
                    stock_pct = (
                        (latest.iloc[0]["stock_awards"] + latest.iloc[0].get("option_awards", 0))
                        / ceo_pay * 100
                        if ceo_pay > 0 else None
                    )
                rows.append(
                    {
                        "company": company_name,
                        "cik": company_cik,
                        "ceo_pay": ceo_pay,
                        "cfo_pay": cfo_pay,
                        "stock_pct": round(stock_pct, 1) if stock_pct else None,
                        "total_exec_comp": metrics.get("total_exec_comp"),
                        "pay_ratio": parser.parse_ceo_pay_ratio(text).get("ratio"),
                        "filing_date": filings[0]["filing_date"],
                    }
                )
            except Exception as exc:
                logger.warning("peer bench error for %s: %s", company_name, exc)

        return pd.DataFrame(rows)

    async def sector_governance_screen(self, sic_code: str) -> pd.DataFrame:
        """
        Screen governance scores across all companies with given SIC code.
        Returns DataFrame sorted by governance score descending.
        """
        # Fetch recent DEF 14A filers for this SIC via EFTS
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=*&dateRange=custom&startdt=2024-01-01&forms=DEF+14A"
            f"&hits.hits.total.value=true"
        )
        adapter = ProxyFilingAdapter()
        scorer  = GovernanceScorer()
        rows: list[dict] = []

        try:
            async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
                r = await _get(client, search_url)
                hits = r.json().get("hits", {}).get("hits", [])
        except Exception:
            hits = []

        for hit in hits[:20]:
            src = hit.get("_source", {})
            cik = str(src.get("entity_id", ""))
            name = (src.get("display_names") or [src.get("entity_name", "")])[0]
            acc = src.get("accession_no", "")
            if not cik or not acc:
                continue
            try:
                async with adapter:
                    text = await adapter.get_proxy_text(acc, cik)
                score = scorer.score_governance(text, cik)
                rows.append(
                    {
                        "company": name,
                        "cik": cik,
                        "governance_score": score["total_score"],
                        "grade": score["letter_grade"],
                        "flags": "; ".join(score["flags"]),
                    }
                )
            except Exception as exc:
                logger.debug("sector screen error %s: %s", name, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("governance_score", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query

    proxy_router = APIRouter(prefix="/api/proxy", tags=["proxy"])

    def _run(coro):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    @proxy_router.get("/{ticker}/latest")
    def get_proxy_latest(ticker: str):
        """Most recent proxy filing summary for a ticker."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(404, f"CIK not found for ticker {ticker}")
        adapter = ProxyFilingAdapter()

        async def _fetch():
            async with adapter:
                filings = await adapter.get_proxy_filings(cik, lookback_years=3)
            return filings

        filings = _run(_fetch())
        if not filings:
            raise HTTPException(404, "No proxy filings found")
        latest = filings[0]

        async def _text():
            async with adapter:
                return await adapter.get_proxy_text(latest["accession_number"], cik)

        text = _run(_text())
        scorer   = GovernanceScorer()
        gov      = scorer.score_governance(text, cik)
        ratio    = ExecutiveCompensationParser().parse_ceo_pay_ratio(text)
        return {
            "ticker": ticker,
            "cik": cik,
            "filing": latest,
            "governance_summary": gov,
            "ceo_pay_ratio": ratio,
        }

    @proxy_router.get("/{ticker}/compensation")
    def get_proxy_compensation(ticker: str):
        """Executive compensation table for a ticker."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(404, f"CIK not found for ticker {ticker}")
        adapter = ProxyFilingAdapter()
        parser  = ExecutiveCompensationParser()

        async def _fetch():
            async with adapter:
                filings = await adapter.get_proxy_filings(cik, lookback_years=3)
                if not filings:
                    return None, None
                text = await adapter.get_proxy_text(filings[0]["accession_number"], cik)
            return filings[0], text

        filing, text = _run(_fetch())
        if not text:
            raise HTTPException(404, "No proxy filing text available")

        comp_df  = parser.parse_summary_comp_table(text)
        metrics  = parser.compute_compensation_metrics(comp_df)
        pvp      = parser.parse_pay_vs_performance(text)
        return {
            "ticker": ticker,
            "filing_date": filing["filing_date"] if filing else None,
            "summary_comp_table": comp_df.to_dict(orient="records"),
            "pay_vs_performance": pvp.to_dict(orient="records"),
            "metrics": metrics,
        }

    @proxy_router.get("/{ticker}/governance-score")
    def get_governance_score(ticker: str):
        """Governance scorecard with letter grade."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(404, f"CIK not found for ticker {ticker}")
        adapter = ProxyFilingAdapter()
        scorer  = GovernanceScorer()

        async def _fetch():
            async with adapter:
                filings = await adapter.get_proxy_filings(cik, lookback_years=3)
                if not filings:
                    return None
                return await adapter.get_proxy_text(filings[0]["accession_number"], cik)

        text = _run(_fetch())
        if not text:
            raise HTTPException(404, "No proxy text available")

        score       = scorer.score_governance(text, cik)
        board       = scorer.parse_board_composition(text)
        provisions  = scorer.parse_antitakeover_provisions(text)
        return {
            "ticker": ticker,
            "cik": cik,
            "governance_score": score,
            "board_composition": board,
            "antitakeover_provisions": provisions,
        }

    @proxy_router.get("/{ticker}/voting")
    def get_proxy_voting(ticker: str, year: int = Query(None)):
        """Say-on-pay and director vote results."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(404, f"CIK not found for ticker {ticker}")
        engine = VotingAnalyticsEngine()
        vote_results = _run(engine.get_vote_results(cik, year=year))
        parsed: list[dict] = []
        for vr in vote_results:
            sop = engine.parse_say_on_pay_result(vr.get("raw_text", ""))
            parsed.append({"filing_date": vr["filing_date"], "say_on_pay": sop})
        return {"ticker": ticker, "cik": cik, "vote_results": parsed}

    @proxy_router.get("/{ticker}/board")
    def get_board_composition(ticker: str):
        """Board composition analysis."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            raise HTTPException(404, f"CIK not found for ticker {ticker}")
        adapter = ProxyFilingAdapter()
        scorer  = GovernanceScorer()

        async def _fetch():
            async with adapter:
                filings = await adapter.get_proxy_filings(cik, lookback_years=2)
                if not filings:
                    return None
                return await adapter.get_proxy_text(filings[0]["accession_number"], cik)

        text = _run(_fetch())
        if not text:
            raise HTTPException(404, "No proxy text available")

        board      = scorer.parse_board_composition(text)
        proposals  = scorer.parse_shareholder_proposals(text)
        return {
            "ticker": ticker,
            "board": board,
            "proposals": proposals,
        }

    @proxy_router.get("/{ticker}/peer-comp")
    def get_peer_compensation(ticker: str, n_peers: int = Query(10, le=25)):
        """Peer compensation benchmark."""
        cik = _ticker_to_cik(ticker)
        analytics = ProxyCompAnalytics()
        df = _run(analytics.peer_compensation_benchmark(ticker, cik, n_peers=n_peers))
        return {"ticker": ticker, "peer_benchmark": df.to_dict(orient="records")}

except ImportError:
    proxy_router = None  # type: ignore
    logger.info("FastAPI not available; proxy_router not registered")


# ---------------------------------------------------------------------------
# Convenience async façade
# ---------------------------------------------------------------------------

class ProxyIntelligence:
    """
    High-level async interface combining all proxy intelligence capabilities.
    """

    def __init__(self):
        self.adapter  = ProxyFilingAdapter()
        self.comp_parser = ExecutiveCompensationParser()
        self.scorer   = GovernanceScorer()
        self.voting   = VotingAnalyticsEngine()
        self.comp_analytics = ProxyCompAnalytics()

    async def full_analysis(self, ticker: str) -> dict:
        """
        Run complete proxy intelligence analysis for a ticker.
        Returns all sections in one call.
        """
        cik = _ticker_to_cik(ticker)
        if not cik:
            return {"error": f"CIK not found for ticker {ticker}"}

        async with self.adapter:
            filings = await self.adapter.get_proxy_filings(cik, lookback_years=5)
            if not filings:
                return {"error": "No proxy filings found"}
            latest_text = await self.adapter.get_proxy_text(
                filings[0]["accession_number"], cik
            )

        comp_df     = self.comp_parser.parse_summary_comp_table(latest_text)
        metrics     = self.comp_parser.compute_compensation_metrics(comp_df)
        pay_ratio   = self.comp_parser.parse_ceo_pay_ratio(latest_text)
        pvp         = self.comp_parser.parse_pay_vs_performance(latest_text)
        gov_score   = self.scorer.score_governance(latest_text, cik)
        board       = self.scorer.parse_board_composition(latest_text)
        proposals   = self.scorer.parse_shareholder_proposals(latest_text)
        provisions  = self.scorer.parse_antitakeover_provisions(latest_text)

        return {
            "ticker": ticker,
            "cik": cik,
            "filings_found": len(filings),
            "latest_filing_date": filings[0]["filing_date"],
            "compensation": {
                "summary_table": comp_df.to_dict(orient="records"),
                "metrics": metrics,
                "pay_ratio": pay_ratio,
                "pay_vs_performance": pvp.to_dict(orient="records"),
            },
            "governance": {
                "score": gov_score,
                "board": board,
                "proposals": proposals,
                "antitakeover_provisions": provisions,
            },
        }
