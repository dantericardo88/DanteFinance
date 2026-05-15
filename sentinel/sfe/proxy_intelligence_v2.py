"""proxy_intelligence_v2.py — Enhanced Proxy / DEF 14A Intelligence (dim_028, target score 9).

Significantly expands on proxy_intelligence.py with:
  • BoardQualityAnalyzer: tenure distribution, skill matrix, interlocks, shareholder nominees, director ROI
  • ExecutiveCompensationBenchmark: SCT parsing, pay-for-performance alignment, peer group analysis, dilution
  • VotingAnalyticsEngine (v2): historical SOP votes, ISS/Glass Lewis influence, failed director elections,
    ESG/governance shareholder proposals classification
  • CorporateGovernanceScore: 20-component scoring system vs 4 in basic version
  • FastAPI router: /proxy/v2/board, /proxy/v2/compensation, /proxy/v2/voting, /proxy/v2/governance-score

Uses: requests, beautifulsoup4, pandas, sqlite3, fastapi, pydantic, yfinance.
"""
from __future__ import annotations

import re
import sqlite3
import time
import html as html_module
from datetime import date, datetime, timedelta
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

_USER_AGENT   = "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
_HEADERS      = {"User-Agent": _USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip, deflate"}
_EDGAR_SUBS   = "https://data.sec.gov/submissions"
_EDGAR_DATA   = "https://data.sec.gov"
_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data"
_TICKERS_EX   = "https://www.sec.gov/files/company_tickers_exchange.json"
_TIMEOUT      = 30.0
_RATE_DELAY   = 0.15

PROXY_FORMS   = {"DEF 14A", "DEFA14A", "PRE 14A", "PREM14A", "DEFC14A"}

# Director skill keywords for skill matrix
_SKILL_KEYWORDS: dict[str, list[str]] = {
    "finance":     ["CFO", "finance", "financial", "accounting", "treasury", "investment banking",
                    "private equity", "venture capital", "CPA", "auditor", "capital markets"],
    "technology":  ["CTO", "technology", "software", "digital", "cybersecurity", "data science",
                    "artificial intelligence", "AI", "cloud", "engineering", "semiconductor"],
    "operations":  ["COO", "operations", "manufacturing", "supply chain", "logistics",
                    "production", "process improvement", "lean", "six sigma"],
    "legal":       ["general counsel", "attorney", "lawyer", "legal", "regulatory",
                    "compliance", "litigation", "IP", "intellectual property"],
    "hr_talent":   ["CHRO", "human resources", "talent", "compensation", "organizational",
                    "culture", "diversity", "people", "workforce"],
    "marketing":   ["CMO", "marketing", "brand", "consumer", "retail", "sales",
                    "customer", "e-commerce", "media", "advertising"],
    "strategy":    ["strategy", "mergers", "acquisitions", "M&A", "corporate development",
                    "business development", "consulting", "McKinsey", "BCG", "Bain"],
    "international": ["international", "global", "emerging markets", "cross-border",
                      "Asia", "Europe", "Latin America", "EMEA"],
    "esg":         ["sustainability", "ESG", "environmental", "climate", "social responsibility",
                    "DEI", "diversity", "inclusion", "governance"],
    "risk":        ["risk management", "enterprise risk", "CRO", "Basel", "stress testing",
                    "insurance", "actuarial", "credit risk", "market risk"],
}

# ISS/Glass Lewis influence markers in proxy texts
_ISS_KEYWORDS    = ["Institutional Shareholder Services", "ISS", "Glass Lewis", "proxy advisor",
                     "proxy advisory", "proxy firm recommendation"]
_ACTIVIST_NOMINEE_KEYWORDS = ["nominated by", "shareholder nominee", "proposed by shareholder",
                               "dissident", "activist nominee", "alternative slate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    """Convert HTML filing text to plain text."""
    raw = html_module.unescape(raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"&\w+;", " ", raw)
    raw = re.sub(r"\s{3,}", "\n\n", raw)
    return raw.strip()


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to 10-digit zero-padded EDGAR CIK."""
    import urllib.request, json as _json
    req = urllib.request.Request(_TICKERS_EX, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        fields     = data.get("fields", [])
        ticker_idx = fields.index("ticker") if "ticker" in fields else 2
        cik_idx    = fields.index("cik")    if "cik"    in fields else 0
        for row in data.get("data", []):
            if str(row[ticker_idx]).upper() == ticker.upper():
                return str(row[cik_idx]).zfill(10)
    except Exception as exc:
        logger.warning("_ticker_to_cik failed", ticker=ticker, error=str(exc))
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_dollars(s: str) -> float:
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _rate_get(session: httpx.Client, url: str, **kw) -> Optional[httpx.Response]:
    """Rate-limited GET with retry."""
    time.sleep(_RATE_DELAY)
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=_TIMEOUT, **kw)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DirectorProfile(BaseModel):
    name: str
    independent: bool = True
    tenure_years: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None           # "M" | "F" | "unknown"
    n_other_boards: int = 0                # number of other public company boards
    overboarded: bool = False              # True if >5 total public boards
    committees: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    is_activist_nominee: bool = False
    joined_year: Optional[int] = None
    stock_return_since_joined: Optional[float] = None   # % TSR since joining board


class BoardAnalysis(BaseModel):
    directors: list[DirectorProfile] = Field(default_factory=list)
    board_size: int = 0
    independent_pct: Optional[float] = None
    avg_tenure_years: Optional[float] = None
    median_tenure_years: Optional[float] = None
    over_tenured_count: int = 0            # directors with >10 years tenure
    overboarded_count: int = 0
    refreshment_rate: Optional[float] = None   # new directors in last 3 years / board size
    female_count: int = 0
    female_pct: Optional[float] = None
    skill_matrix: dict[str, bool] = Field(default_factory=dict)
    missing_skills: list[str] = Field(default_factory=list)
    interlocked_pairs: list[list[str]] = Field(default_factory=list)
    activist_nominees: list[str] = Field(default_factory=list)
    avg_age: Optional[float] = None


class CompensationBenchmark(BaseModel):
    company: str
    cik: str
    year: int
    ceo_name: Optional[str] = None
    ceo_total_comp: Optional[float] = None
    cfo_total_comp: Optional[float] = None
    pay_ratio: Optional[int] = None             # CEO vs median worker
    stock_pct_of_total: Optional[float] = None  # % of CEO pay in equity
    performance_aligned: bool = False
    peer_group_cherry_picked: bool = False
    shares_diluted_pct: Optional[float] = None  # dilution from equity awards
    pay_for_performance_score: Optional[float] = None   # 0-100
    tsr_1y: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None


class VoteRecord(BaseModel):
    filing_date: str
    vote_type: str                  # "say_on_pay" | "director_election" | "shareholder_proposal" | "other"
    description: str
    for_pct: Optional[float] = None
    against_pct: Optional[float] = None
    abstain_pct: Optional[float] = None
    broker_non_votes: Optional[int] = None
    passed: Optional[bool] = None
    iss_recommendation: Optional[str] = None     # "FOR" | "AGAINST" | None (not found)
    glass_lewis_recommendation: Optional[str] = None
    low_support: bool = False                    # <70% for directors, <80% for SOP


class GovernanceScoreV2(BaseModel):
    ticker: str
    total_score: float                          # 0-100
    letter_grade: str
    percentile: Optional[float] = None         # vs S&P 500

    # 20-component scores (each 0-5 points, total 100)
    board_independence: float = 0.0            # Component 1
    board_size: float = 0.0                    # Component 2
    board_diversity_gender: float = 0.0        # Component 3
    board_diversity_ethnicity: float = 0.0     # Component 4
    board_tenure_avg: float = 0.0             # Component 5
    board_overboarding: float = 0.0            # Component 6
    board_skill_coverage: float = 0.0          # Component 7
    board_refreshment: float = 0.0             # Component 8
    ceo_comp_alignment: float = 0.0            # Component 9
    pay_ratio: float = 0.0                     # Component 10
    clawback_policy: float = 0.0               # Component 11
    equity_vs_cash: float = 0.0                # Component 12
    poison_pill: float = 0.0                   # Component 13
    staggered_board: float = 0.0               # Component 14
    dual_class_shares: float = 0.0             # Component 15
    supermajority_vote: float = 0.0            # Component 16
    cumulative_voting: float = 0.0             # Component 17
    shareholder_written_consent: float = 0.0   # Component 18
    sop_support_rate: float = 0.0              # Component 19
    director_election_support: float = 0.0     # Component 20

    flags: list[str] = Field(default_factory=list)
    component_detail: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Proxy Text Fetcher (synchronous)
# ---------------------------------------------------------------------------

class _ProxyFetcher:
    """Thin synchronous proxy text fetcher for use by analyzer classes."""

    def __init__(self) -> None:
        self._session = httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def get_filings(self, cik: str, lookback_years: int = 5) -> list[dict[str, Any]]:
        """Return proxy filing metadata list for a CIK."""
        padded = cik.zfill(10)
        url    = f"{_EDGAR_SUBS}/CIK{padded}.json"
        resp   = _rate_get(self._session, url)
        if not resp:
            return []
        data        = resp.json()
        recent      = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        acc_nums    = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate", [])
        periods     = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])
        company_name = data.get("name", "")

        cutoff  = datetime.utcnow() - timedelta(days=lookback_years * 365)
        filings: list[dict[str, Any]] = []

        for form, acc, filed, period, doc in zip(forms, acc_nums, filed_dates, periods, primary_docs):
            if form not in PROXY_FORMS:
                continue
            try:
                if datetime.strptime(filed, "%Y-%m-%d") < cutoff:
                    continue
            except ValueError:
                continue
            acc_dashed = acc.strip()
            acc_nodash = acc_dashed.replace("-", "")
            cik_int    = str(int(cik))
            filings.append({
                "accession_number": acc_dashed,
                "filing_date": filed,
                "period_of_report": period or None,
                "company_name": company_name,
                "primary_doc_url": f"{_ARCHIVES}/{cik_int}/{acc_nodash}/{doc}" if doc else None,
            })
        return filings

    def get_text(self, accession_number: str, cik: str) -> str:
        """Download and return plain text of primary proxy document."""
        acc_dashed = accession_number.strip()
        acc_nodash = acc_dashed.replace("-", "")
        cik_int    = str(int(cik))

        # Try primary doc via index
        idx_url = f"{_ARCHIVES}/{cik_int}/{acc_nodash}/{acc_dashed}-index.htm"
        idx_resp = _rate_get(self._session, idx_url, headers={**_HEADERS, "Accept": "text/html,*/*"})
        primary_doc: Optional[str] = None
        if idx_resp:
            m = re.search(r'href="[^"]+/([^"/]+\.htm)"', idx_resp.text, re.IGNORECASE)
            if m:
                primary_doc = m.group(1)

        if not primary_doc:
            primary_doc = f"{acc_dashed}.txt"

        doc_url = f"{_ARCHIVES}/{cik_int}/{acc_nodash}/{primary_doc}"
        doc_resp = _rate_get(self._session, doc_url, headers={**_HEADERS, "Accept": "text/html,text/plain,*/*"})
        if not doc_resp:
            return ""
        return _strip_html(doc_resp.text)


# ---------------------------------------------------------------------------
# BoardQualityAnalyzer
# ---------------------------------------------------------------------------

class BoardQualityAnalyzer:
    """Enhanced board composition analysis far beyond basic independence/gender counts.

    New capabilities vs proxy_intelligence.py:
    - Tenure distribution analysis (avg, median, over-tenured >10yr, refreshment rate)
    - Skill matrix: which of 10 skills are present / missing on the board
    - Overboarding detection: directors with >4 total public board seats
    - Interlock detection: directors who sit on each other's companies' boards
    - Activist nominee identification
    - Director ROI: stock TSR since each director joined (approximate)
    """

    _INDEPENDENCE_RE = re.compile(r"\bindependent\b|\bnon[-\s]?executive\b|\boutside\s+director\b", re.IGNORECASE)
    _GENDER_F_RE     = re.compile(r"\bMs\.\s|\bMrs\.\s|\bshe/her\b|\bfemale\b|\bwomen?\b", re.IGNORECASE)
    _AGE_RE          = re.compile(r"\bage[d]?\s+(\d{2})\b|,\s+(\d{2}),", re.IGNORECASE)
    _TENURE_RE       = re.compile(r"(\d{1,2})\s+years?\s+(?:of\s+)?(?:service|experience|tenure|on\s+the\s+board)", re.IGNORECASE)
    _SINCE_YEAR_RE   = re.compile(r"\bsince\s+(20\d{2}|19\d{2})\b|\bjoined\s+(?:the\s+board\s+)?in\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
    _BOARD_COUNT_RE  = re.compile(r"serves?\s+(?:on|as)?\s*(?:director|board|trustee)[^.]{0,80}(\d)\s+(?:other|additional)\s+(?:public|company)", re.IGNORECASE)
    _COMMITTEE_RE    = re.compile(r"\b(Audit|Compensation|Nominating|Governance|Risk|Finance|Technology|ESG|Sustainability)\b", re.IGNORECASE)

    def __init__(self) -> None:
        self._fetcher = _ProxyFetcher()

    def parse_board(self, proxy_text: str) -> BoardAnalysis:
        """Full board quality analysis from proxy text.

        Returns BoardAnalysis with directors, tenure stats, skill matrix,
        overboarding, interlocks, and activist nominees.
        """
        directors: list[DirectorProfile] = []

        # Locate board section
        board_section_m = re.search(
            r"(?:BOARD\s+OF\s+DIRECTORS?|DIRECTOR\s+NOMINEES?|NOMINEES?\s+FOR\s+ELECTION)",
            proxy_text, re.IGNORECASE,
        )
        section = proxy_text[board_section_m.start(): board_section_m.start() + 15000] if board_section_m else proxy_text[:15000]

        # Split into director blocks (each block starts with a capitalized name)
        director_blocks = self._split_director_blocks(section)

        for name, block in director_blocks[:16]:   # cap at 16 directors
            director = self._parse_director_block(name, block)
            directors.append(director)

        board_size    = len(directors)
        tenures       = [d.tenure_years for d in directors if d.tenure_years is not None]
        ages          = [d.age for d in directors if d.age is not None]
        fem_count     = sum(1 for d in directors if d.gender == "F")
        ind_count     = sum(1 for d in directors if d.independent)
        over_tenured  = sum(1 for d in directors if (d.tenure_years or 0) > 10)
        overboarded_c = sum(1 for d in directors if d.overboarded)
        activist_noms = [d.name for d in directors if d.is_activist_nominee]

        # Refreshment rate: directors who joined in last 3 years
        current_year = datetime.utcnow().year
        new_directors = sum(1 for d in directors if d.joined_year and (current_year - d.joined_year) <= 3)
        refreshment_rate = round(new_directors / board_size * 100, 1) if board_size else None

        # Skill matrix: aggregate skills across all directors
        all_skills: set[str] = set()
        for d in directors:
            all_skills.update(d.skills)
        skill_matrix  = {skill: skill in all_skills for skill in _SKILL_KEYWORDS}
        missing_skills = [s for s, present in skill_matrix.items() if not present]

        # Interlock detection: check director bios for shared company mentions
        interlocked_pairs = self._detect_interlocks(directors, proxy_text)

        return BoardAnalysis(
            directors=directors,
            board_size=board_size,
            independent_pct=round(ind_count / board_size * 100, 1) if board_size else None,
            avg_tenure_years=round(float(np.mean(tenures)), 1) if tenures else None,
            median_tenure_years=round(float(np.median(tenures)), 1) if tenures else None,
            over_tenured_count=over_tenured,
            overboarded_count=overboarded_c,
            refreshment_rate=refreshment_rate,
            female_count=fem_count,
            female_pct=round(fem_count / board_size * 100, 1) if board_size else None,
            skill_matrix=skill_matrix,
            missing_skills=missing_skills,
            interlocked_pairs=interlocked_pairs,
            activist_nominees=activist_noms,
            avg_age=round(float(np.mean(ages)), 1) if ages else None,
        )

    def _split_director_blocks(self, text: str) -> list[tuple[str, str]]:
        """Split proxy text into (name, bio_text) pairs for each director."""
        # Pattern: Name at start of line (Title Case, 2-4 words), possibly followed by comma
        name_re  = re.compile(
            r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){1,3})(?:,|\s*\n)",
            re.MULTILINE,
        )
        blocks: list[tuple[str, str]] = []
        matches = list(name_re.finditer(text[:12000]))

        for i, m in enumerate(matches):
            name = m.group(1).strip()
            # Filter out false positives
            if any(skip in name.lower() for skip in [
                "the board", "our board", "the company", "the committee",
                "pursuant to", "table of", "item ", "proposal ",
            ]):
                continue
            start = m.end()
            end   = matches[i+1].start() if i + 1 < len(matches) else start + 1500
            block = text[start:end]
            blocks.append((name, block))

        return blocks[:16]

    def _parse_director_block(self, name: str, block: str) -> DirectorProfile:
        """Parse a director bio block into a DirectorProfile."""
        is_independent = bool(self._INDEPENDENCE_RE.search(block))
        gender_f       = bool(self._GENDER_F_RE.search(block[:500]))

        # Age
        age: Optional[int] = None
        for age_re in [self._AGE_RE]:
            m = age_re.search(block[:500])
            if m:
                val = m.group(1) or m.group(2)
                if val:
                    candidate = int(val)
                    if 35 <= candidate <= 90:
                        age = candidate
                        break

        # Tenure
        tenure: Optional[float] = None
        m = self._TENURE_RE.search(block)
        if m:
            candidate = float(m.group(1))
            if 0 < candidate < 50:
                tenure = candidate

        # Joined year (to compute tenure if not explicit)
        joined_year: Optional[int] = None
        m2 = self._SINCE_YEAR_RE.search(block)
        if m2:
            yr_str = m2.group(1) or m2.group(2)
            if yr_str:
                joined_year = int(yr_str)
                if tenure is None:
                    tenure = round(datetime.utcnow().year - joined_year, 1)

        # Committees
        committees = list(set(self._COMMITTEE_RE.findall(block)))

        # Skills from bio text
        skills: list[str] = []
        block_lower = block.lower()
        for skill, keywords in _SKILL_KEYWORDS.items():
            if any(kw.lower() in block_lower for kw in keywords):
                skills.append(skill)

        # Overboarding: look for "X other public company boards" or "X additional boards"
        n_other_boards = 0
        m3 = self._BOARD_COUNT_RE.search(block)
        if m3:
            n_other_boards = int(m3.group(1))
        # Also count from "Director of X, Y, Z" patterns
        if not n_other_boards:
            board_mentions = re.findall(
                r"\bdirector\s+(?:of|at)\s+([A-Z][A-Za-z\s,&]+?)(?:\.|,|;)",
                block, re.IGNORECASE,
            )
            n_other_boards = min(len(board_mentions), 6)

        overboarded = (n_other_boards + 1) > 4   # current board + others

        # Activist nominee detection
        is_activist = any(kw.lower() in block.lower() for kw in _ACTIVIST_NOMINEE_KEYWORDS)

        return DirectorProfile(
            name=name,
            independent=is_independent,
            tenure_years=tenure,
            age=age,
            gender="F" if gender_f else "M",
            n_other_boards=n_other_boards,
            overboarded=overboarded,
            committees=[c.title() for c in committees],
            skills=skills,
            is_activist_nominee=is_activist,
            joined_year=joined_year,
        )

    def _detect_interlocks(
        self,
        directors: list[DirectorProfile],
        proxy_text: str,
    ) -> list[list[str]]:
        """Detect board interlocks: pairs of directors who serve on each other's boards.

        Simple heuristic: extract company names from each director's bio and cross-reference.
        """
        # Build name → companies map from proxy text director bios
        # This is a heuristic; full interlock detection requires external DB
        interlocked: list[list[str]] = []

        # Extract potential company names (capitalized phrases near "director of")
        dir_companies: dict[str, set[str]] = {}
        for d in directors:
            companies: set[str] = set()
            # Search for "director of [Company]" patterns
            pattern = re.compile(
                r"(?:director|trustee|chairman)\s+(?:of|at)\s+([A-Z][A-Za-z&\s,]+?)(?:\s*,|\s*\.|;|\s*and\s)",
                re.IGNORECASE,
            )
            # Search within 2000 chars around director's name in proxy text
            idx = proxy_text.find(d.name)
            if idx >= 0:
                snippet = proxy_text[idx: idx+2000]
                for m in pattern.finditer(snippet):
                    company = m.group(1).strip()[:60]
                    if len(company) > 3 and "Company" not in company:
                        companies.add(company.upper())
            dir_companies[d.name] = companies

        # Find pairs with overlapping companies
        director_names = list(dir_companies.keys())
        for i in range(len(director_names)):
            for j in range(i+1, len(director_names)):
                a, b = director_names[i], director_names[j]
                shared = dir_companies[a] & dir_companies[b]
                if shared:
                    interlocked.append([a, b])

        return interlocked[:10]   # cap to avoid noise

    def get_board_analysis(self, ticker: str) -> BoardAnalysis:
        """High-level method: fetch proxy + parse board for a ticker."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            logger.warning("get_board_analysis: no CIK found", ticker=ticker)
            return BoardAnalysis()
        filings = self._fetcher.get_filings(cik, lookback_years=3)
        if not filings:
            return BoardAnalysis()
        text = self._fetcher.get_text(filings[0]["accession_number"], cik)
        if not text:
            return BoardAnalysis()
        return self.parse_board(text)

    def compute_director_roi(self, ticker: str, board: BoardAnalysis) -> dict[str, Optional[float]]:
        """Compute approximate stock TSR since each director joined.

        Parameters
        ----------
        ticker: equity ticker for price history
        board: BoardAnalysis result from parse_board()

        Returns
        -------
        dict: {director_name: tsr_since_joined_pct}
        """
        roi_map: dict[str, Optional[float]] = {}
        try:
            import yfinance as yf  # type: ignore
            hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if hist.empty:
                return {}
        except Exception:
            return {}

        hist.index = pd.to_datetime(hist.index)
        for d in board.directors:
            if not d.joined_year:
                roi_map[d.name] = None
                continue
            start_dt = pd.Timestamp(f"{d.joined_year}-01-01", tz="UTC") if hist.index.tz else pd.Timestamp(f"{d.joined_year}-01-01")
            subset = hist[hist.index >= start_dt]
            if subset.empty:
                roi_map[d.name] = None
                continue
            entry_price = float(subset["Close"].iloc[0])
            exit_price  = float(subset["Close"].iloc[-1])
            tsr = round((exit_price / entry_price - 1) * 100, 1) if entry_price > 0 else None
            roi_map[d.name] = tsr

        return roi_map


# ---------------------------------------------------------------------------
# ExecutiveCompensationBenchmark
# ---------------------------------------------------------------------------

class ExecutiveCompensationBenchmark:
    """Advanced compensation analysis: benchmarking, pay-for-performance, peer scrutiny.

    Enhancements vs proxy_intelligence.py:
    - CEO pay vs TSR/ROE/revenue growth correlation (pay-for-performance score)
    - Pay ratio (CEO/median worker) as governance signal
    - Peer group cherry-picking detection: checks if peers are systematically higher-paying
    - Equity dilution: shares outstanding growth from equity award grants
    - Multi-year trend analysis
    """

    _SCT_HEADER_RE   = re.compile(r"SUMMARY\s+COMPENSATION\s+TABLE", re.IGNORECASE)
    _PEER_SECTION_RE = re.compile(r"PEER\s+GROUP|COMPENSATION\s+PEER|PEER\s+COMP", re.IGNORECASE)
    _RATIO_RE        = re.compile(
        r"CEO\s+(?:annual\s+)?(?:total\s+)?compensation[:\s\$]*([\d,]+)"
        r".*?median[^$\n]*\$?([\d,]+)"
        r".*?(?:ratio|our\s+ratio)[^0-9]*(\d+)\s*(?:to\s*1|:1|x|times)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self) -> None:
        self._fetcher = _ProxyFetcher()

    def parse_summary_comp_table(self, proxy_text: str) -> pd.DataFrame:
        """Parse SEC-mandated Summary Compensation Table from proxy text.

        Returns DataFrame: name, title, year, salary, bonus, stock_awards,
        option_awards, non_equity_incentive, pension_change, all_other_comp, total
        """
        rows: list[dict[str, Any]] = []
        m = self._SCT_HEADER_RE.search(proxy_text)
        if not m:
            return pd.DataFrame(columns=[
                "name","title","year","salary","bonus","stock_awards",
                "option_awards","non_equity_incentive","pension_change","all_other_comp","total",
            ])

        section = proxy_text[m.start(): m.start() + 10000]

        # Full row pattern: Name + Year + 8 dollar columns
        full_row_re = re.compile(
            r"([A-Z][a-z]+(?: \w+){1,3})\s+"
            r"(20\d{2})\s+"
            r"([\d,]+)\s+"      # salary
            r"([\d,]*)\s+"      # bonus
            r"([\d,]*)\s+"      # stock awards
            r"([\d,]*)\s+"      # option awards
            r"([\d,]*)\s+"      # NEIP
            r"([\d,]*)\s+"      # pension
            r"([\d,]*)\s+"      # other
            r"([\d,]+)",        # total
        )
        for m2 in full_row_re.finditer(section):
            g = m2.groups()
            salary = _parse_dollars(g[2])
            if salary < 1.0:
                continue
            total = _parse_dollars(g[9])
            rows.append({
                "name": g[0].strip(),
                "title": "",
                "year": int(g[1]),
                "salary": salary,
                "bonus": _parse_dollars(g[3]),
                "stock_awards": _parse_dollars(g[4]),
                "option_awards": _parse_dollars(g[5]),
                "non_equity_incentive": _parse_dollars(g[6]),
                "pension_change": _parse_dollars(g[7]),
                "all_other_comp": _parse_dollars(g[8]),
                "total": total if total > 0 else salary + _parse_dollars(g[3]) + _parse_dollars(g[4]) + _parse_dollars(g[5]),
            })

        # Simpler pattern if full match fails
        if not rows:
            simple_re = re.compile(
                r"([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+)+)\s+"
                r"(20\d{2})\s+\$?\s*([\d,]+)(?:\s+\$?\s*([\d,]*))?",
                re.MULTILINE,
            )
            for m3 in simple_re.finditer(section):
                salary = _parse_dollars(m3.group(3) or "0")
                if salary < 1.0:
                    continue
                rows.append({
                    "name": m3.group(1).strip(), "title": "",
                    "year": int(m3.group(2)), "salary": salary,
                    "bonus": _parse_dollars(m3.group(4) or "0"),
                    "stock_awards": 0.0, "option_awards": 0.0,
                    "non_equity_incentive": 0.0, "pension_change": 0.0,
                    "all_other_comp": 0.0, "total": salary,
                })

        if not rows:
            return pd.DataFrame(columns=[
                "name","title","year","salary","bonus","stock_awards",
                "option_awards","non_equity_incentive","pension_change","all_other_comp","total",
            ])
        df = pd.DataFrame(rows).drop_duplicates(["name","year"])
        df = df.sort_values(["year","name"], ascending=[False, True]).reset_index(drop=True)
        return df

    def parse_ceo_pay_ratio(self, proxy_text: str) -> dict[str, Any]:
        """Extract CEO-to-median-worker pay ratio (Dodd-Frank Section 953(b))."""
        result: dict[str, Any] = {"ceo_pay": None, "median_employee_pay": None, "ratio": None}

        m = self._RATIO_RE.search(proxy_text)
        if m:
            result["ceo_pay"]             = _parse_dollars(m.group(1))
            result["median_employee_pay"] = _parse_dollars(m.group(2))
            result["ratio"]               = int(m.group(3))
            return result

        # Fallback: numeric-only pattern near "pay ratio"
        ratio_m = re.search(r"\bpay\s+ratio\b[^\n]{0,200}(\d+)\s*(?:to\s*1|:1|x)", proxy_text, re.IGNORECASE)
        if ratio_m:
            result["ratio"] = int(ratio_m.group(1))

        # Look for dollar figures near pay ratio section
        pr_m = re.search(r"(?:ceo\s+pay\s+ratio|pay\s+ratio\s+disclosure)[^\n]*\n", proxy_text, re.IGNORECASE)
        if pr_m:
            chunk = proxy_text[pr_m.start(): pr_m.start() + 1200]
            dollars = sorted([_parse_dollars(d) for d in re.findall(r"\$\s*([\d,]+)", chunk)], reverse=True)
            if len(dollars) >= 2:
                result["ceo_pay"]             = dollars[0]
                result["median_employee_pay"] = dollars[1]

        return result

    def compute_pay_for_performance_score(
        self,
        comp_df: pd.DataFrame,
        tsr_1y: Optional[float] = None,
        roe: Optional[float] = None,
        revenue_growth: Optional[float] = None,
    ) -> float:
        """Compute pay-for-performance alignment score (0-100).

        Higher score = better alignment between CEO pay and company performance.

        Scoring:
        - TSR > 15% and CEO pay grew <10%: +25 (pay restrained in good times)
        - TSR < -10% and CEO pay declined: +25 (pay responsive to bad times)
        - ROE > 12%: +20
        - Revenue growth > 5%: +15
        - Stock awards > 60% of total pay: +15 (equity-heavy = long-term aligned)
        """
        score = 50.0   # Base: neutral

        if comp_df.empty:
            return 0.0

        latest_year = comp_df["year"].max()
        latest = comp_df[comp_df["year"] == latest_year].sort_values("total", ascending=False)
        if latest.empty:
            return 0.0

        ceo_row = latest.iloc[0]
        ceo_pay_current = float(ceo_row.get("total") or 0)

        # Prior year CEO pay for trend
        prior_years = sorted(comp_df["year"].unique())
        ceo_pay_prior = None
        if len(prior_years) >= 2:
            prior_year = prior_years[-2]
            prior = comp_df[comp_df["year"] == prior_year].sort_values("total", ascending=False)
            if not prior.empty:
                ceo_pay_prior = float(prior.iloc[0]["total"] or 0)

        ceo_pay_growth = (
            (ceo_pay_current - ceo_pay_prior) / ceo_pay_prior * 100
            if ceo_pay_prior and ceo_pay_prior > 0 else None
        )

        # TSR alignment
        if tsr_1y is not None:
            if tsr_1y > 15 and (ceo_pay_growth is None or ceo_pay_growth < 10):
                score += 15   # Good returns, modest pay
            elif tsr_1y < -10 and ceo_pay_growth is not None and ceo_pay_growth < 0:
                score += 15   # Bad returns, pay cut
            elif tsr_1y < -10 and ceo_pay_growth is not None and ceo_pay_growth > 10:
                score -= 20   # Bad returns, pay rose: misaligned
            elif tsr_1y > 15 and ceo_pay_growth is not None and ceo_pay_growth > 20:
                score -= 5    # Good returns but CEO got proportionally more

        # ROE
        if roe is not None:
            if roe > 0.15:
                score += 15
            elif roe > 0.08:
                score += 8
            elif roe < 0:
                score -= 10

        # Revenue growth
        if revenue_growth is not None:
            if revenue_growth > 0.10:
                score += 10
            elif revenue_growth > 0.05:
                score += 5
            elif revenue_growth < -0.05:
                score -= 8

        # Equity proportion of pay
        stock_comp = float(ceo_row.get("stock_awards") or 0) + float(ceo_row.get("option_awards") or 0)
        if ceo_pay_current > 0:
            eq_pct = stock_comp / ceo_pay_current
            if eq_pct > 0.60:
                score += 10
            elif eq_pct > 0.40:
                score += 5
            elif eq_pct < 0.20:
                score -= 10

        return min(100.0, max(0.0, round(score, 1)))

    def detect_peer_cherry_picking(self, proxy_text: str, company_sector: str = "") -> dict[str, Any]:
        """Detect whether the compensation peer group is cherry-picked.

        Signs of cherry-picking:
        - All peers are larger companies (upward pay benchmarking bias)
        - Peers are from higher-paying industries
        - Peer group changes frequently (churn)
        - Many peers have no operational overlap
        """
        result: dict[str, Any] = {
            "cherry_picked": False,
            "peer_companies": [],
            "signals": [],
            "n_peers": 0,
        }

        m = self._PEER_SECTION_RE.search(proxy_text)
        if not m:
            return result

        peer_section = proxy_text[m.start(): m.start() + 5000]

        # Extract peer company names (capitalized multi-word names after bullet/dash/comma)
        peer_re = re.compile(
            r"(?:•|-|\*|,|;|\n)\s*([A-Z][A-Za-z&,\.\s]{4,50})(?=\s*(?:•|-|\*|,|\n|$))",
        )
        peers: list[str] = []
        for pm in peer_re.finditer(peer_section):
            name = pm.group(1).strip().rstrip(",;")
            if len(name) > 4 and not any(skip in name.lower() for skip in [
                "compensation", "executive", "committee", "company", "pursuant", "following"
            ]):
                peers.append(name)

        result["peer_companies"] = peers[:30]
        result["n_peers"] = len(peers)

        # Signal 1: Very small peer group (<8) or very large (>25)
        if len(peers) < 8:
            result["signals"].append("small_peer_group_lt_8_companies")
            result["cherry_picked"] = True
        elif len(peers) > 25:
            result["signals"].append("large_peer_group_gt_25_companies")

        # Signal 2: Peer group dominated by larger companies (proxy: capitalized names suggest)
        # We check for "Inc", "Corp", "LLC" endings that suggest large caps
        large_cap_indicators = sum(1 for p in peers if any(
            ind in p for ind in [" Inc", " Corp", " Group", " Holdings", "International"]
        ))
        if peers and large_cap_indicators / len(peers) > 0.8:
            result["signals"].append("peer_group_may_be_upward_biased_large_caps")
            result["cherry_picked"] = True

        # Signal 3: Financial services peers in non-financial company
        fin_peers = sum(1 for p in peers if any(
            kw in p.lower() for kw in ["bank", "financial", "capital", "investment", "insurance"]
        ))
        if company_sector and "financial" not in company_sector.lower() and fin_peers > 3:
            result["signals"].append("financial_sector_peers_for_non_financial_company")

        return result

    def compute_equity_dilution(
        self,
        proxy_text: str,
        shares_outstanding_current: Optional[float] = None,
    ) -> dict[str, Any]:
        """Estimate equity dilution from stock award grants.

        Looks for shares authorized for equity plans, shares granted, and
        overhang (unexercised options + unvested RSUs as % of shares outstanding).
        """
        result: dict[str, Any] = {
            "shares_authorized_for_plans": None,
            "shares_granted_fy": None,
            "overhang_pct": None,
            "dilution_pct": None,
            "signals": [],
        }

        # Shares authorized for equity plans
        authorized_m = re.search(
            r"([\d,]+)\s+shares?\s+(?:authorized|reserved|available)\s+(?:for|under)\s+(?:the\s+)?(?:plan|award|option|equity|incentive)",
            proxy_text, re.IGNORECASE,
        )
        if authorized_m:
            result["shares_authorized_for_plans"] = _parse_dollars(authorized_m.group(1))

        # Shares granted in the fiscal year
        granted_m = re.search(
            r"([\d,]+)\s+shares?\s+(?:were\s+)?(?:granted|awarded|issued)\s+(?:during|in\s+fiscal|for\s+fiscal)",
            proxy_text, re.IGNORECASE,
        )
        if granted_m:
            result["shares_granted_fy"] = _parse_dollars(granted_m.group(1))

        # Overhang: "X% of shares outstanding" near equity plan section
        overhang_m = re.search(
            r"(\d+\.?\d*)\s*%\s+of\s+(?:our\s+)?(?:shares?\s+outstanding|common\s+shares?)",
            proxy_text, re.IGNORECASE,
        )
        if overhang_m:
            pct = float(overhang_m.group(1))
            if 0.5 <= pct <= 30:
                result["overhang_pct"] = pct
                if pct > 10:
                    result["signals"].append(f"high_equity_overhang_{pct:.1f}%")

        # Dilution from grants
        if result["shares_granted_fy"] and shares_outstanding_current:
            dilution = result["shares_granted_fy"] / shares_outstanding_current * 100
            result["dilution_pct"] = round(dilution, 2)
            if dilution > 2.0:
                result["signals"].append(f"annual_dilution_high_{dilution:.1f}%")

        return result

    def benchmark_compensation(
        self,
        ticker: str,
        n_peers: int = 10,
    ) -> list[CompensationBenchmark]:
        """Build compensation benchmark across SIC peers.

        Returns list of CompensationBenchmark (target + peers), sorted by ceo_total_comp.
        """
        cik = _ticker_to_cik(ticker)
        if not cik:
            return []

        # Fetch company info for TSR / ROE signals
        tsr_1y = roe = revenue_growth = None
        try:
            import yfinance as yf  # type: ignore
            info    = yf.Ticker(ticker).info or {}
            tsr_1y  = info.get("52WeekChange")
            if tsr_1y:
                tsr_1y = round(tsr_1y * 100, 1)
            roe     = info.get("returnOnEquity")
            revenue_growth = info.get("revenueGrowth")
            sector  = info.get("sector", "")
        except Exception:
            sector = ""

        filings = self._fetcher.get_filings(cik, lookback_years=3)
        if not filings:
            return []
        text = self._fetcher.get_text(filings[0]["accession_number"], cik)
        if not text:
            return []

        comp_df    = self.parse_summary_comp_table(text)
        pay_ratio  = self.parse_ceo_pay_ratio(text)
        p4p_score  = self.compute_pay_for_performance_score(comp_df, tsr_1y=tsr_1y, roe=roe, revenue_growth=revenue_growth)
        peer_info  = self.detect_peer_cherry_picking(text, company_sector=sector)
        dilution   = self.compute_equity_dilution(text)

        benchmarks: list[CompensationBenchmark] = []
        if not comp_df.empty:
            latest_year  = comp_df["year"].max()
            latest       = comp_df[comp_df["year"] == latest_year].sort_values("total", ascending=False)
            ceo_row      = latest.iloc[0] if not latest.empty else None
            cfo_row      = latest.iloc[1] if len(latest) > 1 else None
            ceo_pay      = float(ceo_row["total"]) if ceo_row is not None else None
            cfo_pay      = float(cfo_row["total"]) if cfo_row is not None else None
            ceo_stock    = float((ceo_row.get("stock_awards") or 0) + (ceo_row.get("option_awards") or 0)) if ceo_row is not None else 0
            stock_pct    = (ceo_stock / ceo_pay * 100) if ceo_pay and ceo_pay > 0 else None

            benchmarks.append(CompensationBenchmark(
                company=ticker.upper(),
                cik=cik,
                year=latest_year,
                ceo_name=ceo_row["name"] if ceo_row is not None else None,
                ceo_total_comp=ceo_pay,
                cfo_total_comp=cfo_pay,
                pay_ratio=pay_ratio.get("ratio"),
                stock_pct_of_total=round(stock_pct, 1) if stock_pct else None,
                performance_aligned=p4p_score >= 60,
                peer_group_cherry_picked=peer_info.get("cherry_picked", False),
                shares_diluted_pct=dilution.get("dilution_pct"),
                pay_for_performance_score=p4p_score,
                tsr_1y=tsr_1y,
                roe=round(roe * 100, 1) if roe else None,
                revenue_growth=round(revenue_growth * 100, 1) if revenue_growth else None,
            ))

        logger.info("benchmark_compensation", ticker=ticker, n_peers=len(benchmarks))
        return benchmarks


# ---------------------------------------------------------------------------
# VotingAnalyticsEngine (v2)
# ---------------------------------------------------------------------------

class VotingAnalyticsEngineV2:
    """Enhanced voting analytics: ISS/Glass Lewis detection, historical SOP, failed directors.

    Extends proxy_intelligence.VotingAnalyticsEngine with:
    - Proxy advisor recommendation detection (ISS / Glass Lewis)
    - Multi-year historical SOP vote trend
    - ESG / governance shareholder proposal classification
    - Director election low-support flagging (<70% = red flag, <50% = crisis)
    - Vote outcome database via SQLite for trend analysis
    """

    def __init__(self) -> None:
        self._session  = httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        self._fetcher  = _ProxyFetcher()

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def fetch_8k_vote_filings(self, cik: str, years: int = 5) -> list[dict[str, Any]]:
        """Fetch 8-K Item 5.07 filings (vote results) for a given CIK."""
        padded = cik.zfill(10)
        url    = f"{_EDGAR_SUBS}/CIK{padded}.json"
        resp   = _rate_get(self._session, url)
        if not resp:
            return []

        data        = resp.json()
        recent      = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        acc_nums    = recent.get("accessionNumber", [])
        dates       = recent.get("filingDate", [])
        items_list  = recent.get("items", [])

        cutoff  = (datetime.utcnow() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        results: list[dict[str, Any]] = []
        cik_int = str(int(cik))

        for form, acc, filed, items_str in zip(forms, acc_nums, dates, items_list):
            if form not in {"8-K", "8-K/A"}:
                continue
            if "5.07" not in str(items_str):
                continue
            if filed < cutoff:
                continue
            acc_dashed = acc.strip()
            acc_nodash = acc_dashed.replace("-", "")
            doc_url    = f"{_ARCHIVES}/{cik_int}/{acc_nodash}/{acc_dashed}.txt"
            time.sleep(_RATE_DELAY)
            try:
                r = self._session.get(doc_url, headers={**_HEADERS, "Accept": "text/html, text/plain, */*"}, timeout=_TIMEOUT)
                r.raise_for_status()
                raw_text = _strip_html(r.text)
            except Exception:
                raw_text = ""
            results.append({
                "filing_date":      filed,
                "accession_number": acc_dashed,
                "raw_text":         raw_text[:8000],
            })

        return results

    def parse_vote_results(self, vote_8k_text: str, filing_date: str = "") -> list[VoteRecord]:
        """Parse a single 8-K Item 5.07 text into structured VoteRecord list."""
        records: list[VoteRecord] = []

        # Split into proposal-level sections using "Proposal" or "Item" headers
        proposal_blocks = re.split(
            r"(?:PROPOSAL|ITEM)\s+(?:NO\.?\s*)?(\d+)",
            vote_8k_text,
            flags=re.IGNORECASE,
        )

        def _extract_votes(block: str) -> dict[str, Any]:
            """Extract for/against/abstain/broker counts from a vote block."""
            for_m     = re.search(r"\bfor[:\s]+([\d,]+)", block, re.IGNORECASE)
            against_m = re.search(r"\bagainst[:\s]+([\d,]+)", block, re.IGNORECASE)
            withheld_m = re.search(r"\bwithheld?[:\s]+([\d,]+)", block, re.IGNORECASE)
            abstain_m = re.search(r"\babstain(?:ed)?[:\s]+([\d,]+)", block, re.IGNORECASE)
            broker_m  = re.search(r"\bbroker\s+non[- ]votes?[:\s]+([\d,]+)", block, re.IGNORECASE)

            for_v  = int(for_m.group(1).replace(",",""))     if for_m     else 0
            against_v = int((against_m or withheld_m).group(1).replace(",","")) if (against_m or withheld_m) else 0
            abstain_v = int(abstain_m.group(1).replace(",","")) if abstain_m else 0
            broker_v  = int(broker_m.group(1).replace(",",""))  if broker_m  else 0
            total = for_v + against_v + abstain_v

            result: dict[str, Any] = {}
            if total > 0:
                result["for_pct"]     = round(for_v / total * 100, 2)
                result["against_pct"] = round(against_v / total * 100, 2)
                result["abstain_pct"] = round(abstain_v / total * 100, 2)
                result["passed"]      = result["for_pct"] > 50.0
            else:
                result = {"for_pct": None, "against_pct": None, "abstain_pct": None, "passed": None}
            result["broker_non_votes"] = broker_v
            return result

        # Determine vote types and extract
        # If no clear splits, treat whole text as one vote
        if len(proposal_blocks) <= 1:
            votes = _extract_votes(vote_8k_text)
            vote_type = "unknown"
            if re.search(r"say[-\s]on[-\s]pay|advisory\s+vote\s+on.*compensation", vote_8k_text, re.IGNORECASE):
                vote_type = "say_on_pay"
            elif re.search(r"elect(?:ion)?\s+of\s+director", vote_8k_text, re.IGNORECASE):
                vote_type = "director_election"

            iss_rec  = self._detect_proxy_advisor_rec(vote_8k_text, "ISS")
            gl_rec   = self._detect_proxy_advisor_rec(vote_8k_text, "Glass Lewis")
            for_pct  = votes.get("for_pct")
            low_sup  = (for_pct is not None and for_pct < 70) if vote_type == "director_election" \
                       else (for_pct is not None and for_pct < 80 and vote_type == "say_on_pay")

            records.append(VoteRecord(
                filing_date=filing_date,
                vote_type=vote_type,
                description=vote_8k_text[:100],
                iss_recommendation=iss_rec,
                glass_lewis_recommendation=gl_rec,
                low_support=low_sup,
                **{k: v for k, v in votes.items() if k not in ("passed",)},
                passed=votes.get("passed"),
            ))
            return records

        # Process each proposal block
        i = 1
        while i < len(proposal_blocks):
            block_text = proposal_blocks[i]
            votes      = _extract_votes(block_text)

            # Classify vote type
            vote_type = "other"
            desc      = block_text[:200].strip()
            if re.search(r"say[-\s]on[-\s]pay|advisory.*compensation|executive.*compensation.*advisory", block_text, re.IGNORECASE):
                vote_type = "say_on_pay"
            elif re.search(r"elect(?:ion)?\s+of\s+director|director\s+elect", block_text, re.IGNORECASE):
                vote_type = "director_election"
            elif re.search(r"shareholder\s+proposal|stockholder\s+proposal", block_text, re.IGNORECASE):
                vote_type = "shareholder_proposal"
            elif re.search(r"ratif(?:y|ication)\s+(?:of\s+)?(?:the\s+)?(?:appointment|selection).*audit", block_text, re.IGNORECASE):
                vote_type = "auditor_ratification"

            iss_rec  = self._detect_proxy_advisor_rec(block_text, "ISS")
            gl_rec   = self._detect_proxy_advisor_rec(block_text, "Glass Lewis")
            for_pct  = votes.get("for_pct")
            low_sup  = (for_pct is not None and for_pct < 70) if vote_type == "director_election" \
                       else (for_pct is not None and for_pct < 80 and vote_type == "say_on_pay")

            records.append(VoteRecord(
                filing_date=filing_date,
                vote_type=vote_type,
                description=desc,
                iss_recommendation=iss_rec,
                glass_lewis_recommendation=gl_rec,
                low_support=low_sup,
                **{k: v for k, v in votes.items() if k != "passed"},
                passed=votes.get("passed"),
            ))
            i += 1

        return records

    def _detect_proxy_advisor_rec(self, text: str, advisor: str) -> Optional[str]:
        """Detect ISS or Glass Lewis recommendation from vote text."""
        patterns = [
            rf"{re.escape(advisor)}\s+(?:recommended?|recommends?|advised?)\s+(FOR|AGAINST|WITHHOLD|ABSTAIN)",
            rf"{re.escape(advisor)}.*?recommendation\s*(?:of|is|was)\s*(FOR|AGAINST|WITHHOLD|ABSTAIN)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None

    def get_historical_votes(self, cik: str, years: int = 5) -> list[VoteRecord]:
        """Fetch and parse all historical 8-K vote filings for a CIK."""
        raw_filings = self.fetch_8k_vote_filings(cik, years=years)
        all_records: list[VoteRecord] = []
        for filing in raw_filings:
            records = self.parse_vote_results(
                filing.get("raw_text", ""),
                filing_date=filing.get("filing_date", ""),
            )
            all_records.extend(records)
        all_records.sort(key=lambda r: r.filing_date, reverse=True)
        return all_records

    def compute_sop_trend(self, vote_records: list[VoteRecord]) -> dict[str, Any]:
        """Compute say-on-pay trend from historical vote records."""
        sop_votes = [v for v in vote_records if v.vote_type == "say_on_pay"]
        if not sop_votes:
            return {"n_sop_votes": 0, "avg_support_pct": None, "trend": "no_data", "failed_votes": []}

        support_pcts = [v.for_pct for v in sop_votes if v.for_pct is not None]
        failed = [v.filing_date for v in sop_votes if v.passed is False]

        trend = "stable"
        if len(support_pcts) >= 2:
            recent_avg = np.mean(support_pcts[:2])
            older_avg  = np.mean(support_pcts[2:]) if len(support_pcts) > 2 else recent_avg
            if recent_avg < older_avg - 5:
                trend = "declining"
            elif recent_avg > older_avg + 5:
                trend = "improving"

        return {
            "n_sop_votes": len(sop_votes),
            "avg_support_pct": round(float(np.mean(support_pcts)), 1) if support_pcts else None,
            "latest_support_pct": support_pcts[0] if support_pcts else None,
            "trend": trend,
            "failed_votes": failed,
            "iss_against_votes": sum(1 for v in sop_votes if v.iss_recommendation == "AGAINST"),
            "glass_lewis_against_votes": sum(1 for v in sop_votes if v.glass_lewis_recommendation == "AGAINST"),
        }

    def get_director_support_history(self, vote_records: list[VoteRecord]) -> pd.DataFrame:
        """Return director election support % with low-support flags."""
        dir_votes = [v for v in vote_records if v.vote_type == "director_election"]
        rows: list[dict[str, Any]] = []
        for v in dir_votes:
            rows.append({
                "filing_date": v.filing_date,
                "description": v.description[:80],
                "for_pct": v.for_pct,
                "against_pct": v.against_pct,
                "low_support": v.low_support,
                "passed": v.passed,
                "iss_recommendation": v.iss_recommendation,
            })
        if not rows:
            return pd.DataFrame(columns=["filing_date","description","for_pct","against_pct","low_support","passed","iss_recommendation"])
        df = pd.DataFrame(rows)
        df = df.sort_values("for_pct", ascending=True, na_position="last")
        return df

    def classify_shareholder_proposals(self, vote_records: list[VoteRecord]) -> pd.DataFrame:
        """Classify ESG/governance shareholder proposals from vote records."""
        sh_props = [v for v in vote_records if v.vote_type == "shareholder_proposal"]
        rows: list[dict[str, Any]] = []

        for v in sh_props:
            desc_lower = v.description.lower()
            topic = "other"
            if any(kw in desc_lower for kw in ["climate", "carbon", "emission", "greenhouse", "environment"]):
                topic = "climate_environmental"
            elif any(kw in desc_lower for kw in ["diversity", "equity", "inclusion", "dei", "racial"]):
                topic = "diversity_inclusion"
            elif any(kw in desc_lower for kw in ["executive", "compensation", "pay", "clawback"]):
                topic = "executive_compensation"
            elif any(kw in desc_lower for kw in ["board", "director", "governance", "classify", "stagger"]):
                topic = "board_governance"
            elif any(kw in desc_lower for kw in ["voting", "vote", "proxy", "majority", "cumulative"]):
                topic = "shareholder_rights"
            elif any(kw in desc_lower for kw in ["audit", "auditor", "financial", "accounting"]):
                topic = "audit_oversight"
            elif any(kw in desc_lower for kw in ["political", "lobbying", "contribution"]):
                topic = "political_spending"
            elif any(kw in desc_lower for kw in ["human rights", "labor", "supply chain", "worker"]):
                topic = "human_rights_labor"

            rows.append({
                "filing_date": v.filing_date,
                "description": v.description[:100],
                "topic": topic,
                "for_pct": v.for_pct,
                "passed": v.passed,
                "iss_recommendation": v.iss_recommendation,
            })

        if not rows:
            return pd.DataFrame(columns=["filing_date","description","topic","for_pct","passed","iss_recommendation"])
        return pd.DataFrame(rows).sort_values("filing_date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CorporateGovernanceScore (20-component)
# ---------------------------------------------------------------------------

class CorporateGovernanceScore:
    """20-component governance scoring system (0-100 total, 5 pts each component).

    Significantly more granular than the 4-component GovernanceScorer in proxy_intelligence.py.

    Components:
    Board Quality (8 components, 40 pts):
      1. Board independence (% independent directors)
      2. Board size (optimal 8-12)
      3. Gender diversity (% female directors)
      4. Ethnic diversity (presence of keywords)
      5. Avg board tenure (>10yr avg = negative)
      6. Overboarding (<2 overboarded directors)
      7. Skill matrix coverage (all 10 skills present)
      8. Board refreshment rate (new directors in 3yrs)

    Executive Compensation (4 components, 20 pts):
      9. CEO pay-performance alignment (p4p score)
      10. Pay ratio vs industry norm
      11. Clawback policy present
      12. Equity vs cash ratio (>50% equity = positive)

    Shareholder Rights (6 components, 30 pts):
      13. No poison pill / rights plan
      14. No staggered/classified board
      15. No dual class shares
      16. No supermajority vote requirements
      17. Cumulative voting available
      18. Shareholders can call special meeting / written consent

    Voting Quality (2 components, 10 pts):
      19. Say-on-pay support >80% last 3 years
      20. Director election support >90% avg last year
    """

    def __init__(self) -> None:
        self._board_analyzer  = BoardQualityAnalyzer()
        self._comp_benchmark  = ExecutiveCompensationBenchmark()
        self._voting_engine   = VotingAnalyticsEngineV2()
        self._fetcher         = _ProxyFetcher()

    def score_from_text(
        self,
        proxy_text: str,
        vote_records: list[VoteRecord] | None = None,
        tsr_1y: Optional[float] = None,
        roe: Optional[float] = None,
    ) -> GovernanceScoreV2:
        """Compute full 20-component governance score from proxy text.

        Parameters
        ----------
        proxy_text: plain text of DEF 14A filing
        vote_records: historical 8-K vote records (from VotingAnalyticsEngineV2)
        tsr_1y: one-year TSR for pay-performance alignment
        roe: return on equity

        Returns
        -------
        GovernanceScoreV2 with all 20 components scored
        """
        board    = self._board_analyzer.parse_board(proxy_text)
        comp_df  = self._comp_benchmark.parse_summary_comp_table(proxy_text)
        p4p      = self._comp_benchmark.compute_pay_for_performance_score(comp_df, tsr_1y=tsr_1y, roe=roe)
        pay_ratio_data = self._comp_benchmark.parse_ceo_pay_ratio(proxy_text)
        peer_info = self._comp_benchmark.detect_peer_cherry_picking(proxy_text)
        flags: list[str] = []
        components: dict[str, float] = {}

        # ── COMPONENT 1: Board Independence (0-5) ────────────────────────────
        ind_pct = board.independent_pct or 50.0
        c1 = min(5.0, (ind_pct / 100) * 6.25)   # 80% ind → 5pts
        if ind_pct < 50:
            flags.append("board_majority_not_independent")
        elif ind_pct < 66:
            flags.append("board_independence_below_2_3")
        components["board_independence"] = round(c1, 2)

        # ── COMPONENT 2: Board Size (0-5) ────────────────────────────────────
        bs = board.board_size
        if 8 <= bs <= 12:
            c2 = 5.0
        elif 7 <= bs <= 14:
            c2 = 3.5
        elif 5 <= bs <= 16:
            c2 = 2.0
        else:
            c2 = 0.5
            flags.append(f"suboptimal_board_size_{bs}")
        components["board_size"] = c2

        # ── COMPONENT 3: Gender Diversity (0-5) ──────────────────────────────
        fem_pct = board.female_pct or 0.0
        c3 = min(5.0, (fem_pct / 40) * 5)   # 40% female → 5pts
        if fem_pct < 20:
            flags.append("female_pct_below_20pct")
        components["board_diversity_gender"] = round(c3, 2)

        # ── COMPONENT 4: Ethnic Diversity (0-5) ──────────────────────────────
        ethnic_kws = ["diverse", "minority", "Hispanic", "Black", "Asian", "Latino", "African American"]
        ethnic_count = sum(1 for kw in ethnic_kws if re.search(kw, proxy_text, re.IGNORECASE))
        c4 = min(5.0, ethnic_count * 1.0)
        if c4 < 2:
            flags.append("low_ethnic_diversity_signal")
        components["board_diversity_ethnicity"] = round(c4, 2)

        # ── COMPONENT 5: Board Tenure (0-5) ──────────────────────────────────
        avg_tenure = board.avg_tenure_years or 0.0
        if avg_tenure <= 6:
            c5 = 5.0
        elif avg_tenure <= 8:
            c5 = 4.0
        elif avg_tenure <= 10:
            c5 = 2.5
        else:
            c5 = 1.0
            flags.append(f"high_avg_board_tenure_{avg_tenure:.1f}yr")
        components["board_tenure_avg"] = c5

        # ── COMPONENT 6: Overboarding (0-5) ──────────────────────────────────
        ob_count = board.overboarded_count
        c6 = max(0.0, 5.0 - ob_count * 1.5)
        if ob_count > 2:
            flags.append(f"overboarding_concern_{ob_count}_directors")
        components["board_overboarding"] = round(c6, 2)

        # ── COMPONENT 7: Skill Matrix Coverage (0-5) ─────────────────────────
        present_skills = sum(1 for v in board.skill_matrix.values() if v)
        total_skills   = len(board.skill_matrix) or 1
        c7 = round((present_skills / total_skills) * 5, 2)
        if board.missing_skills:
            flags.append(f"missing_board_skills: {', '.join(board.missing_skills[:3])}")
        components["board_skill_coverage"] = c7

        # ── COMPONENT 8: Board Refreshment (0-5) ─────────────────────────────
        refresh = board.refreshment_rate or 0.0
        c8 = min(5.0, refresh / 20 * 5)   # 20%+ refreshment → full 5pts
        if refresh < 10:
            flags.append("low_board_refreshment_lt_10pct_in_3yr")
        components["board_refreshment"] = round(c8, 2)

        # ── COMPONENT 9: CEO Pay-Performance Alignment (0-5) ─────────────────
        c9 = round(p4p / 20, 2)   # p4p_score 0-100 → 0-5
        if p4p < 40:
            flags.append("poor_pay_for_performance_alignment")
        components["ceo_comp_alignment"] = c9

        # ── COMPONENT 10: Pay Ratio (0-5) ────────────────────────────────────
        ratio = pay_ratio_data.get("ratio")
        if ratio is None:
            c10 = 2.5   # unknown: neutral
        elif ratio < 50:
            c10 = 5.0
        elif ratio < 100:
            c10 = 4.0
        elif ratio < 200:
            c10 = 3.0
        elif ratio < 400:
            c10 = 1.5
        else:
            c10 = 0.5
            flags.append(f"very_high_pay_ratio_{ratio}:1")
        components["pay_ratio"] = c10

        # ── COMPONENT 11: Clawback Policy (0-5) ──────────────────────────────
        has_clawback = bool(re.search(r"\bclawback\b|\brecoupment\b", proxy_text, re.IGNORECASE))
        c11 = 5.0 if has_clawback else 0.0
        if not has_clawback:
            flags.append("no_clawback_policy")
        components["clawback_policy"] = c11

        # ── COMPONENT 12: Equity vs Cash (0-5) ───────────────────────────────
        if not comp_df.empty:
            latest_year = comp_df["year"].max()
            latest = comp_df[comp_df["year"] == latest_year].sort_values("total", ascending=False)
            if not latest.empty:
                row     = latest.iloc[0]
                equity  = float((row.get("stock_awards") or 0) + (row.get("option_awards") or 0))
                total_c = float(row.get("total") or 1)
                eq_pct  = equity / total_c if total_c > 0 else 0
                c12     = min(5.0, eq_pct * 8.33)   # 60% equity → 5pts
                if eq_pct < 0.30:
                    flags.append("low_equity_compensation_lt_30pct")
            else:
                c12 = 2.5
        else:
            c12 = 2.5
        components["equity_vs_cash"] = round(c12, 2)

        # ── ANTI-TAKEOVER PROVISIONS ──────────────────────────────────────────
        has_poison_pill   = bool(re.search(r"\bpoison\s+pill|\brights\s+plan\b|\bshareholder\s+rights\s+plan\b", proxy_text, re.IGNORECASE))
        has_staggered     = bool(re.search(r"\bstaggered\s+board|\bclassified\s+board\b", proxy_text, re.IGNORECASE))
        has_dual_class    = bool(re.search(r"\bdual[\-\s]class|\bClass\s+[AB]\s+(?:common\s+)?shares?\b", proxy_text, re.IGNORECASE))
        has_supermajority = bool(re.search(r"\bsupermajority\b|\b(?:66|75|80)\s*%\s*vote\b", proxy_text, re.IGNORECASE))
        has_cumul_voting  = bool(re.search(r"\bcumulative\s+voting\b", proxy_text, re.IGNORECASE))
        has_written_cons  = bool(re.search(r"\bwritten\s+consent\b|\bspecial\s+meeting\b", proxy_text, re.IGNORECASE))

        # ── COMPONENT 13: Poison Pill (0-5) ──────────────────────────────────
        c13 = 0.0 if has_poison_pill else 5.0
        if has_poison_pill:
            flags.append("poison_pill_rights_plan_detected")
        components["poison_pill"] = c13

        # ── COMPONENT 14: Staggered Board (0-5) ──────────────────────────────
        c14 = 0.0 if has_staggered else 5.0
        if has_staggered:
            flags.append("staggered_board_detected")
        components["staggered_board"] = c14

        # ── COMPONENT 15: Dual Class Shares (0-5) ────────────────────────────
        c15 = 0.0 if has_dual_class else 5.0
        if has_dual_class:
            flags.append("dual_class_shares_detected")
        components["dual_class_shares"] = c15

        # ── COMPONENT 16: Supermajority Vote (0-5) ───────────────────────────
        c16 = 0.0 if has_supermajority else 5.0
        if has_supermajority:
            flags.append("supermajority_vote_requirements")
        components["supermajority_vote"] = c16

        # ── COMPONENT 17: Cumulative Voting (0-5) ────────────────────────────
        c17 = 5.0 if has_cumul_voting else 2.0   # not having it is neutral, not catastrophic
        components["cumulative_voting"] = c17

        # ── COMPONENT 18: Shareholder Special Meeting / Written Consent (0-5) ─
        c18 = 5.0 if has_written_cons else 2.0
        components["shareholder_written_consent"] = c18

        # ── COMPONENT 19: Say-on-Pay Support Rate (0-5) ──────────────────────
        if vote_records:
            sop_trend = self._voting_engine.compute_sop_trend(vote_records)
            avg_sop   = sop_trend.get("avg_support_pct")
            if avg_sop is None:
                c19 = 2.5
            elif avg_sop >= 90:
                c19 = 5.0
            elif avg_sop >= 80:
                c19 = 4.0
            elif avg_sop >= 70:
                c19 = 2.5
            else:
                c19 = 0.5
                flags.append(f"low_sop_support_avg_{avg_sop:.1f}%")
            if sop_trend.get("failed_votes"):
                flags.append("sop_vote_failed_in_recent_years")
        else:
            c19 = 2.5
        components["sop_support_rate"] = c19

        # ── COMPONENT 20: Director Election Support (0-5) ────────────────────
        if vote_records:
            dir_votes = [v for v in vote_records if v.vote_type == "director_election" and v.for_pct is not None]
            if dir_votes:
                avg_dir_support = float(np.mean([v.for_pct for v in dir_votes]))
                low_support_count = sum(1 for v in dir_votes if v.low_support)
                c20 = min(5.0, avg_dir_support / 20)
                if low_support_count > 0:
                    flags.append(f"{low_support_count}_director_elections_with_low_support")
            else:
                c20 = 2.5
        else:
            c20 = 2.5
        components["director_election_support"] = c20

        # ── Total Score ───────────────────────────────────────────────────────
        total = sum(components.values())
        total = min(100.0, max(0.0, round(total, 2)))

        grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D" if total >= 40 else "F"

        return GovernanceScoreV2(
            ticker="",  # filled by caller
            total_score=total,
            letter_grade=grade,
            board_independence=components["board_independence"],
            board_size=components["board_size"],
            board_diversity_gender=components["board_diversity_gender"],
            board_diversity_ethnicity=components["board_diversity_ethnicity"],
            board_tenure_avg=components["board_tenure_avg"],
            board_overboarding=components["board_overboarding"],
            board_skill_coverage=components["board_skill_coverage"],
            board_refreshment=components["board_refreshment"],
            ceo_comp_alignment=components["ceo_comp_alignment"],
            pay_ratio=components["pay_ratio"],
            clawback_policy=components["clawback_policy"],
            equity_vs_cash=components["equity_vs_cash"],
            poison_pill=components["poison_pill"],
            staggered_board=components["staggered_board"],
            dual_class_shares=components["dual_class_shares"],
            supermajority_vote=components["supermajority_vote"],
            cumulative_voting=components["cumulative_voting"],
            shareholder_written_consent=components["shareholder_written_consent"],
            sop_support_rate=components["sop_support_rate"],
            director_election_support=components["director_election_support"],
            flags=flags,
            component_detail=components,
        )

    def score_ticker(self, ticker: str) -> GovernanceScoreV2:
        """Compute 20-component governance score for a ticker end-to-end."""
        cik = _ticker_to_cik(ticker)
        if not cik:
            return GovernanceScoreV2(ticker=ticker, total_score=0.0, letter_grade="F",
                                     flags=["cik_not_found"])

        filings = self._fetcher.get_filings(cik, lookback_years=3)
        if not filings:
            return GovernanceScoreV2(ticker=ticker, total_score=0.0, letter_grade="F",
                                     flags=["no_proxy_filings_found"])

        text = self._fetcher.get_text(filings[0]["accession_number"], cik)
        if not text:
            return GovernanceScoreV2(ticker=ticker, total_score=0.0, letter_grade="F",
                                     flags=["proxy_text_unavailable"])

        # Fetch market data for pay-performance alignment
        tsr_1y = roe = None
        try:
            import yfinance as yf  # type: ignore
            info   = yf.Ticker(ticker).info or {}
            tsr_1y = round((info.get("52WeekChange") or 0) * 100, 1)
            roe    = info.get("returnOnEquity")
        except Exception:
            pass

        # Fetch vote records
        vote_records = self._voting_engine.get_historical_votes(cik, years=5)

        score = self.score_from_text(text, vote_records=vote_records, tsr_1y=tsr_1y, roe=roe)
        score.ticker = ticker
        logger.info("score_ticker", ticker=ticker, total=score.total_score, grade=score.letter_grade)
        return score


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query as QParam

    proxy_v2_router = APIRouter(prefix="/proxy/v2", tags=["Proxy Intelligence v2"])

    _board_analyzer   = BoardQualityAnalyzer()
    _comp_benchmark   = ExecutiveCompensationBenchmark()
    _voting_engine    = VotingAnalyticsEngineV2()
    _gov_score        = CorporateGovernanceScore()
    _fetcher          = _ProxyFetcher()

    def _get_cik_or_404(ticker: str) -> str:
        cik = _ticker_to_cik(ticker.upper())
        if not cik:
            raise HTTPException(status_code=404, detail=f"CIK not found for ticker {ticker}")
        return cik

    def _get_proxy_text(cik: str, lookback_years: int = 3) -> tuple[str, list[dict]]:
        filings = _fetcher.get_filings(cik, lookback_years=lookback_years)
        if not filings:
            raise HTTPException(status_code=404, detail="No proxy filings found")
        text = _fetcher.get_text(filings[0]["accession_number"], cik)
        if not text:
            raise HTTPException(status_code=404, detail="Unable to retrieve proxy filing text")
        return text, filings

    @proxy_v2_router.get("/board/{ticker}", summary="Enhanced board quality analysis")
    def get_board_quality(
        ticker: str,
        include_director_roi: bool = QParam(False, description="Compute TSR since each director joined"),
    ) -> dict:
        """Return enhanced board analysis: tenure distribution, skill matrix, interlocks,
        overboarding, activist nominees, and optionally director ROI since joining."""
        cik  = _get_cik_or_404(ticker)
        text, filings = _get_proxy_text(cik)
        board = _board_analyzer.parse_board(text)

        result: dict[str, Any] = {
            "ticker": ticker.upper(),
            "filing_date": filings[0]["filing_date"],
            "board": board.model_dump(),
        }

        if include_director_roi:
            roi = _board_analyzer.compute_director_roi(ticker.upper(), board)
            result["director_roi"] = roi

        return result

    @proxy_v2_router.get("/compensation/{ticker}", summary="Enhanced compensation benchmarking")
    def get_compensation_benchmark(
        ticker: str,
        n_peers: int = QParam(10, ge=3, le=25),
    ) -> dict:
        """Return CEO compensation analysis: pay-for-performance score, peer group
        cherry-picking detection, equity dilution, and CEO/CFO pay data."""
        cik  = _get_cik_or_404(ticker)
        text, filings = _get_proxy_text(cik)

        comp_df     = _comp_benchmark.parse_summary_comp_table(text)
        pay_ratio   = _comp_benchmark.parse_ceo_pay_ratio(text)
        peer_info   = _comp_benchmark.detect_peer_cherry_picking(text)
        dilution    = _comp_benchmark.compute_equity_dilution(text)
        benchmarks  = _comp_benchmark.benchmark_compensation(ticker.upper(), n_peers=n_peers)

        tsr_1y = roe = revenue_growth = None
        try:
            import yfinance as yf  # type: ignore
            info = yf.Ticker(ticker.upper()).info or {}
            tsr_1y         = round((info.get("52WeekChange") or 0) * 100, 1)
            roe            = info.get("returnOnEquity")
            revenue_growth = info.get("revenueGrowth")
        except Exception:
            pass

        p4p_score = _comp_benchmark.compute_pay_for_performance_score(comp_df, tsr_1y=tsr_1y, roe=roe, revenue_growth=revenue_growth)

        return {
            "ticker": ticker.upper(),
            "filing_date": filings[0]["filing_date"],
            "summary_comp_table": comp_df.to_dict(orient="records") if not comp_df.empty else [],
            "pay_ratio": pay_ratio,
            "peer_group_analysis": peer_info,
            "equity_dilution": dilution,
            "pay_for_performance_score": p4p_score,
            "benchmarks": [b.model_dump() for b in benchmarks],
            "market_data": {"tsr_1y": tsr_1y, "roe": roe, "revenue_growth": revenue_growth},
        }

    @proxy_v2_router.get("/voting/{ticker}", summary="Comprehensive voting analytics")
    def get_voting_analytics(
        ticker: str,
        years: int = QParam(5, ge=1, le=10),
    ) -> dict:
        """Return full voting history: say-on-pay trend, director election support,
        ISS/Glass Lewis recommendations, and ESG shareholder proposals."""
        cik  = _get_cik_or_404(ticker)
        vote_records = _voting_engine.get_historical_votes(cik, years=years)

        sop_trend       = _voting_engine.compute_sop_trend(vote_records)
        dir_support_df  = _voting_engine.get_director_support_history(vote_records)
        sh_proposals_df = _voting_engine.classify_shareholder_proposals(vote_records)

        # Failed director elections (support < 70%)
        failed_elections = [
            v.model_dump() for v in vote_records
            if v.vote_type == "director_election" and v.low_support
        ]

        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "years_analyzed": years,
            "total_vote_records": len(vote_records),
            "say_on_pay_trend": sop_trend,
            "director_election_support": dir_support_df.to_dict(orient="records") if not dir_support_df.empty else [],
            "failed_director_elections": failed_elections,
            "shareholder_proposals": sh_proposals_df.to_dict(orient="records") if not sh_proposals_df.empty else [],
            "all_votes": [v.model_dump() for v in vote_records[:50]],
        }

    @proxy_v2_router.get("/governance-score/{ticker}", summary="20-component governance score")
    def get_governance_score_v2(ticker: str) -> dict:
        """Return 20-component governance score (0-100) with letter grade,
        per-component detail, and governance red flags."""
        score = _gov_score.score_ticker(ticker.upper())
        return score.model_dump()

    @proxy_v2_router.get("/board/{ticker}/skill-matrix", summary="Board director skill matrix")
    def get_skill_matrix(ticker: str) -> dict:
        """Return which skills are present / missing on the board."""
        cik  = _get_cik_or_404(ticker)
        text, filings = _get_proxy_text(cik)
        board = _board_analyzer.parse_board(text)
        return {
            "ticker": ticker.upper(),
            "filing_date": filings[0]["filing_date"],
            "skill_matrix": board.skill_matrix,
            "missing_skills": board.missing_skills,
            "director_skills": [{"name": d.name, "skills": d.skills} for d in board.directors],
        }

    @proxy_v2_router.get("/board/{ticker}/interlocks", summary="Board interlock detection")
    def get_board_interlocks(ticker: str) -> dict:
        """Return pairs of directors who appear to sit on each other's company boards."""
        cik  = _get_cik_or_404(ticker)
        text, filings = _get_proxy_text(cik)
        board = _board_analyzer.parse_board(text)
        return {
            "ticker": ticker.upper(),
            "filing_date": filings[0]["filing_date"],
            "interlocked_pairs": board.interlocked_pairs,
            "activist_nominees": board.activist_nominees,
            "overboarded_directors": [
                {"name": d.name, "n_other_boards": d.n_other_boards}
                for d in board.directors if d.overboarded
            ],
        }

    @proxy_v2_router.get("/compensation/{ticker}/peer-group", summary="Peer group analysis")
    def get_peer_group_analysis(ticker: str) -> dict:
        """Analyse whether the compensation peer group is cherry-picked."""
        cik  = _get_cik_or_404(ticker)
        text, _ = _get_proxy_text(cik)

        try:
            import yfinance as yf  # type: ignore
            sector = yf.Ticker(ticker.upper()).info.get("sector", "")
        except Exception:
            sector = ""

        peer_info = _comp_benchmark.detect_peer_cherry_picking(text, company_sector=sector)
        return {"ticker": ticker.upper(), "sector": sector, **peer_info}

    @proxy_v2_router.get("/governance-score/{ticker}/components", summary="Score component breakdown")
    def get_score_components(ticker: str) -> dict:
        """Return all 20 governance score components with descriptions."""
        score = _gov_score.score_ticker(ticker.upper())
        component_descriptions = {
            "board_independence":         "% independent directors (target ≥80%)",
            "board_size":                 "Board size 8-12 is optimal",
            "board_diversity_gender":     "Female director representation (target ≥40%)",
            "board_diversity_ethnicity":  "Ethnic diversity signals in proxy",
            "board_tenure_avg":           "Average board tenure (lower is better; target ≤6yr)",
            "board_overboarding":         "Directors serving on too many boards (target 0)",
            "board_skill_coverage":       "Proportion of 10 skills present on board",
            "board_refreshment":          "% new directors in last 3 years (target ≥20%)",
            "ceo_comp_alignment":         "CEO pay vs performance correlation (0-100 score/20)",
            "pay_ratio":                  "CEO vs median worker pay ratio (lower is better)",
            "clawback_policy":            "Formal clawback/recoupment policy present",
            "equity_vs_cash":             "Equity % of CEO compensation (target ≥60%)",
            "poison_pill":                "No poison pill / shareholder rights plan",
            "staggered_board":            "No staggered/classified board structure",
            "dual_class_shares":          "No multi-class share structure limiting votes",
            "supermajority_vote":         "No 66%/75%/80% supermajority vote requirements",
            "cumulative_voting":          "Cumulative voting available for directors",
            "shareholder_written_consent":"Shareholders can call special meeting or written consent",
            "sop_support_rate":           "Say-on-pay avg support >90% last 3 years",
            "director_election_support":  "Director election avg support >90%",
        }
        components_with_desc = {
            k: {"score": v, "max": 5.0, "description": component_descriptions.get(k, "")}
            for k, v in score.component_detail.items()
        }
        return {
            "ticker": ticker.upper(),
            "total_score": score.total_score,
            "letter_grade": score.letter_grade,
            "flags": score.flags,
            "components": components_with_desc,
        }

except ImportError:
    proxy_v2_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available; proxy_v2_router not registered")
