"""
ESG composite ratings: environmental, social, governance scoring.
Free data: EDGAR DEF 14A (governance), 10-K (environmental disclosures),
SEC climate disclosures, open ESG databases, MSCI ESG methodology.

dim_102 — ESG composite ratings & sector scores (target: 9)
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE      = "https://data.sec.gov"
EDGAR_ARCHIVES  = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS   = "https://www.sec.gov/files/company_tickers.json"
EFTS_BASE       = "https://efts.sec.gov/LATEST/search-index"
NLRB_BASE       = "https://www.nlrb.gov/api"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT     = 30.0
_RATE_DELAY  = 0.15   # 150 ms between EDGAR requests
_TEXT_LIMIT  = 150_000  # chars per filing download

# ---------------------------------------------------------------------------
# SIC → GICS sector mapping
# ---------------------------------------------------------------------------

_SIC_TO_SECTOR: dict[str, str] = {
    # Energy
    "1311": "energy", "1382": "energy", "2911": "energy", "1321": "energy",
    "5171": "energy", "1381": "energy", "1400": "energy",
    # Materials
    "2819": "materials", "2860": "materials", "2810": "materials",
    "3312": "materials", "1040": "materials", "2650": "materials",
    "2670": "materials", "2621": "materials", "2810": "materials",
    # Industrials
    "3559": "industrials", "3720": "industrials", "3812": "industrials",
    "3490": "industrials", "3460": "industrials", "3440": "industrials",
    "4210": "industrials", "7389": "industrials",
    # Utilities
    "4911": "utilities", "4931": "utilities", "4941": "utilities",
    "4924": "utilities", "4952": "utilities", "4991": "utilities",
    # Consumer Staples
    "2000": "consumer_staples", "2010": "consumer_staples",
    "5140": "consumer_staples", "5400": "consumer_staples",
    "2100": "consumer_staples", "5910": "consumer_staples",
    # Consumer Discretionary
    "5900": "consumer_discretionary", "5600": "consumer_discretionary",
    "7011": "consumer_discretionary", "7812": "consumer_discretionary",
    "5511": "consumer_discretionary",
    # Health Care
    "2836": "health_care", "2830": "health_care", "8011": "health_care",
    "8049": "health_care", "5912": "health_care", "8099": "health_care",
    "2835": "health_care", "2840": "health_care",
    # Financials
    "6020": "financials", "6022": "financials", "6211": "financials",
    "6159": "financials", "6311": "financials", "6411": "financials",
    "6199": "financials", "6200": "financials",
    # Information Technology
    "7372": "information_technology", "7371": "information_technology",
    "3674": "information_technology", "3577": "information_technology",
    "3669": "information_technology", "3672": "information_technology",
    # Communication Services
    "4813": "communication_services", "4812": "communication_services",
    "4833": "communication_services", "7375": "communication_services",
    # Real Estate
    "6552": "real_estate", "6500": "real_estate", "6512": "real_estate",
    "6798": "real_estate",
}

# Sector-specific ESG weights (MSCI/SASB methodology)
SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "energy":                  {"E": 0.50, "S": 0.25, "G": 0.25},
    "materials":               {"E": 0.50, "S": 0.25, "G": 0.25},
    "industrials":             {"E": 0.35, "S": 0.35, "G": 0.30},
    "utilities":               {"E": 0.50, "S": 0.25, "G": 0.25},
    "consumer_staples":        {"E": 0.25, "S": 0.45, "G": 0.30},
    "consumer_discretionary":  {"E": 0.25, "S": 0.45, "G": 0.30},
    "health_care":             {"E": 0.20, "S": 0.50, "G": 0.30},
    "financials":              {"E": 0.20, "S": 0.30, "G": 0.50},
    "information_technology":  {"E": 0.20, "S": 0.40, "G": 0.40},
    "communication_services":  {"E": 0.20, "S": 0.40, "G": 0.40},
    "real_estate":             {"E": 0.40, "S": 0.30, "G": 0.30},
    "default":                 {"E": 0.35, "S": 0.35, "G": 0.30},
}

# SIC codes for negative ESG screens
_WEAPONS_SIC  = {"3761", "3769", "3489", "3812", "3795", "3760", "1900"}
_TOBACCO_SIC  = {"2100", "2111", "2130", "5194"}
_GAMBLING_SIC = {"7993", "7999", "7011"}
_FOSSIL_SIC   = {"1311", "1321", "2911", "1381", "1382", "5171", "1400"}

# Climate transition risk by sector
CLIMATE_TRANSITION_RISK: dict[str, str] = {
    "energy":                 "HIGH",
    "materials":              "HIGH",
    "utilities":              "HIGH",
    "industrials":            "MEDIUM",
    "consumer_staples":       "MEDIUM",
    "real_estate":            "MEDIUM",
    "consumer_discretionary": "LOW",
    "health_care":            "LOW",
    "financials":             "LOW",
    "information_technology": "LOW",
    "communication_services": "LOW",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GovernanceScore(BaseModel):
    ticker: str
    score: float                     # 0-10
    grade: str
    board_independence_pct: Optional[float] = None
    board_diversity_pct: Optional[float] = None
    ceo_duality: Optional[bool] = None
    say_on_pay_pct: Optional[float] = None
    has_poison_pill: Optional[bool] = None
    has_classified_board: Optional[bool] = None
    ceo_pay_ratio: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)
    filing_url: Optional[str] = None
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


class EnvironmentalScore(BaseModel):
    ticker: str
    score: float                     # 0-10
    grade: str
    has_carbon_disclosure: bool = False
    has_cdp_participation: bool = False
    has_net_zero_target: bool = False
    has_science_based_target: bool = False
    has_environmental_litigation: bool = False
    sector: Optional[str] = None
    keyword_hits: dict[str, int] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


class SocialScore(BaseModel):
    ticker: str
    score: float                     # 0-10
    grade: str
    has_diversity_disclosure: bool = False
    has_supply_chain_audit: bool = False
    has_conflict_minerals_disclosure: bool = False
    has_human_capital_metrics: bool = False
    has_nlrb_cases: bool = False
    has_community_investment: bool = False
    keyword_hits: dict[str, int] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


class ESGRating(BaseModel):
    ticker: str
    company_name: str
    cik: str
    sector: str
    composite: float                 # 0-10
    e_score: float
    s_score: float
    g_score: float
    grade: str                       # A+, A, BBB, BB, B
    percentile: Optional[float] = None
    controversy_penalty: float = 0.0
    weights_used: dict[str, float] = Field(default_factory=dict)
    filing_year: Optional[int] = None
    has_negative_screen_flag: bool = False
    negative_screen_reason: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    disclaimer: str = (
        "Proxy signals from public EDGAR/EPA filings. "
        "Not equivalent to MSCI/Sustainalytics/ISS ratings."
    )


class ESGScreenResult(BaseModel):
    tickers_passed: list[str]
    tickers_failed: list[str]
    negative_screened: list[str]
    ratings: list[ESGRating]
    screener_params: dict[str, Any]
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SectorESGBenchmarkResult(BaseModel):
    sector: str
    n_companies: int
    avg_composite: float
    avg_e: float
    avg_s: float
    avg_g: float
    leaders: list[str]
    laggards: list[str]
    climate_transition_risk: str
    as_of: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sic_to_sector(sic: Optional[str]) -> str:
    if not sic:
        return "default"
    return _SIC_TO_SECTOR.get(str(sic).zfill(4), "default")


def _score_to_grade_10(score: float) -> str:
    """Map 0-10 score to letter grade (MSCI-style)."""
    if score >= 9.0:
        return "A+"
    if score >= 8.0:
        return "A"
    if score >= 7.0:
        return "A-"
    if score >= 6.0:
        return "BBB"
    if score >= 5.0:
        return "BB"
    if score >= 4.0:
        return "B"
    if score >= 3.0:
        return "B-"
    return "CCC"


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        count += len(re.findall(re.escape(kw.lower()), text_lower))
    return count


def _bool_hit(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


async def _fetch_text(url: str, client: httpx.AsyncClient, limit: int = _TEXT_LIMIT) -> str:
    try:
        r = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        raw = r.text[:limit]
        # Strip HTML tags for cleaner text analysis
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"&[a-zA-Z]+;", " ", raw)
        raw = re.sub(r"\s+", " ", raw)
        return raw
    except Exception as exc:
        logger.warning("Fetch failed", url=url, error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

class _EDGARClient:
    """Thin EDGAR async client with CIK resolution and filing fetching."""

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout
        self._cik_cache: dict[str, str] = {}
        self._ticker_map: Optional[dict] = None

    async def _load_ticker_map(self, client: httpx.AsyncClient) -> None:
        if self._ticker_map is not None:
            return
        try:
            r = await client.get(EDGAR_TICKERS, headers=_HEADERS, timeout=self._timeout)
            r.raise_for_status()
            raw = r.json()
            self._ticker_map = {
                v["ticker"].upper(): str(v["cik_str"])
                for v in raw.values()
            }
        except Exception as exc:
            logger.warning("Ticker map load failed", error=str(exc))
            self._ticker_map = {}

    async def resolve_cik(self, ticker: str, client: httpx.AsyncClient) -> Optional[str]:
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        await self._load_ticker_map(client)
        cik = (self._ticker_map or {}).get(ticker)
        if cik:
            self._cik_cache[ticker] = cik
            return cik
        # Fallback: EDGAR company search
        try:
            url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company=&CIK={ticker}&type=10-K&dateb=&owner=include"
                f"&count=5&search_text=&action=getcompany&output=atom"
            )
            r = await client.get(url, headers=_HEADERS, timeout=self._timeout)
            m = re.search(r"CIK=(\d+)", r.text)
            if m:
                cik = m.group(1).lstrip("0") or m.group(1)
                self._cik_cache[ticker] = cik
                return cik
        except Exception:
            pass
        return None

    async def get_latest_filing_texts(
        self,
        cik: str,
        client: httpx.AsyncClient,
    ) -> tuple[str, str, str, str, int]:
        """
        Returns (text_10k, text_proxy, sic_code, company_name, fiscal_year).
        Downloads up to _TEXT_LIMIT chars of the most recent 10-K and DEF 14A.
        """
        await asyncio.sleep(_RATE_DELAY)
        text_10k = text_proxy = ""
        sic_code = company_name = ""
        fiscal_year = datetime.utcnow().year - 1

        # Get company facts / submissions
        try:
            sub_url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
            r = await client.get(sub_url, headers=_HEADERS, timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            company_name = data.get("name", "")
            sic_code = str(data.get("sic", ""))

            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            acc_nos = filings.get("accessionNumber", [])
            dates = filings.get("filingDate", [])
            doc_lists = filings.get("primaryDocument", [])

            # Find latest 10-K
            for i, form in enumerate(forms):
                if form in ("10-K", "10-K/A") and i < len(acc_nos):
                    acc_no = acc_nos[i].replace("-", "")
                    doc = doc_lists[i] if i < len(doc_lists) else ""
                    base_url = f"{EDGAR_ARCHIVES}/{cik}/{acc_no}"
                    filing_url = f"{base_url}/{doc}" if doc else f"{base_url}/"
                    await asyncio.sleep(_RATE_DELAY)
                    text_10k = await _fetch_text(filing_url, client)
                    try:
                        fiscal_year = int(dates[i][:4]) if i < len(dates) else fiscal_year
                    except Exception:
                        pass
                    break

            # Find latest DEF 14A (proxy)
            for i, form in enumerate(forms):
                if form in ("DEF 14A", "DEFA14A") and i < len(acc_nos):
                    acc_no = acc_nos[i].replace("-", "")
                    doc = doc_lists[i] if i < len(doc_lists) else ""
                    base_url = f"{EDGAR_ARCHIVES}/{cik}/{acc_no}"
                    filing_url = f"{base_url}/{doc}" if doc else f"{base_url}/"
                    await asyncio.sleep(_RATE_DELAY)
                    text_proxy = await _fetch_text(filing_url, client)
                    break

        except Exception as exc:
            logger.warning("Filing fetch failed", cik=cik, error=str(exc))

        return text_10k, text_proxy, sic_code, company_name, fiscal_year


# ---------------------------------------------------------------------------
# GovernanceScorer
# ---------------------------------------------------------------------------

class GovernanceScorer:
    """
    Scores corporate governance (G) from SEC DEF 14A proxy filings.
    Uses MSCI governance methodology (simplified) to weight each factor.
    """

    # Board independence: % independent directors
    _INDEP_PATTERN   = re.compile(
        r"(\d{1,2})\s+of\s+(\d{1,2})\s+(?:of our\s+)?(?:board\s+)?directors?\s+(?:are|is)\s+independent",
        re.IGNORECASE,
    )
    _INDEP_PCT_PATTERN = re.compile(
        r"(\d{1,3})%\s+(?:of\s+)?(?:our\s+)?(?:board\s+)?(?:directors?\s+)?(?:are\s+)?independent",
        re.IGNORECASE,
    )
    # Board size
    _BOARD_SIZE_PATTERN = re.compile(
        r"(?:board\s+(?:of\s+directors\s+)?consists?\s+of|we\s+have)\s+(\d{1,2})\s+directors?",
        re.IGNORECASE,
    )
    # CEO duality
    _DUALITY_POSITIVE = re.compile(
        r"(?:serves?\s+as\s+both|(?:our\s+)?CEO\s+(?:also\s+)?serves?\s+as\s+(?:the\s+)?(?:non-executive\s+)?chair)",
        re.IGNORECASE,
    )
    _DUALITY_NEGATIVE = re.compile(
        r"(?:chairman\s+and\s+CEO\s+(?:roles?\s+are\s+)?separated|"
        r"separation\s+of\s+(?:the\s+)?(?:roles?\s+of\s+)?chairman\s+and\s+CEO|"
        r"independent\s+chairman)",
        re.IGNORECASE,
    )
    # Say-on-pay
    _SAY_ON_PAY = re.compile(
        r"say[\s\-]on[\s\-]pay.*?(\d{1,3}(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _SAY_ON_PAY2 = re.compile(
        r"(\d{1,3}(?:\.\d+)?)\s*%.*?(?:approved|in\s+favor|voted\s+for).*?(?:say[\s\-]on[\s\-]pay|advisory\s+vote)",
        re.IGNORECASE,
    )
    # Board diversity
    _FEMALE_PCT = re.compile(
        r"(\d{1,3})%\s+(?:of\s+)?(?:our\s+)?(?:board\s+)?directors?\s+(?:are\s+)?women",
        re.IGNORECASE,
    )
    _FEMALE_COUNT = re.compile(
        r"(\d{1,2})\s+(?:of\s+(?:our\s+)?\d{1,2}\s+)?(?:board\s+)?directors?\s+(?:are|is)\s+women",
        re.IGNORECASE,
    )
    # CEO pay ratio
    _PAY_RATIO = re.compile(
        r"(?:CEO\s+pay\s+ratio|pay\s+ratio).*?(\d{1,4})\s*:\s*1",
        re.IGNORECASE,
    )
    # Poison pill / classified board
    _POISON_PILL = re.compile(
        r"(?:poison\s+pill|shareholder\s+rights\s+plan|rights\s+agreement)",
        re.IGNORECASE,
    )
    _CLASSIFIED_BOARD = re.compile(
        r"(?:classified\s+board|staggered\s+(?:board|elections?)|three[\s\-]class\s+board)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _extract_governance_features(
        self, text_proxy: str, text_10k: str
    ) -> dict[str, Any]:
        """Extract numerical governance signals from filing text."""
        combined = (text_proxy + " " + text_10k)
        features: dict[str, Any] = {}

        # Board independence
        m = self._INDEP_PATTERN.search(combined)
        if m:
            numerator, denominator = int(m.group(1)), int(m.group(2))
            if denominator > 0:
                features["board_independence_pct"] = round(numerator / denominator * 100, 1)
        if "board_independence_pct" not in features:
            m2 = self._INDEP_PCT_PATTERN.search(combined)
            if m2:
                pct = float(m2.group(1))
                if 0 < pct <= 100:
                    features["board_independence_pct"] = pct

        # Board size
        m = self._BOARD_SIZE_PATTERN.search(combined)
        if m:
            features["board_size"] = int(m.group(1))

        # CEO duality
        if self._DUALITY_NEGATIVE.search(combined):
            features["ceo_duality"] = False
        elif self._DUALITY_POSITIVE.search(combined):
            features["ceo_duality"] = True

        # Say-on-pay
        for pattern in [self._SAY_ON_PAY, self._SAY_ON_PAY2]:
            m = pattern.search(combined)
            if m:
                pct = float(m.group(1))
                if 50 < pct <= 100:
                    features["say_on_pay_pct"] = pct
                    break

        # Board diversity (female %)
        m = self._FEMALE_PCT.search(combined)
        if m:
            pct = float(m.group(1))
            if 0 < pct <= 100:
                features["board_diversity_pct"] = pct
        elif "board_size" in features:
            m2 = self._FEMALE_COUNT.search(combined)
            if m2:
                count = int(m2.group(1))
                features["board_diversity_pct"] = round(
                    count / features["board_size"] * 100, 1
                )

        # CEO pay ratio
        m = self._PAY_RATIO.search(combined)
        if m:
            features["ceo_pay_ratio"] = float(m.group(1))

        # Poison pill
        features["has_poison_pill"] = bool(self._POISON_PILL.search(combined))

        # Classified board
        features["has_classified_board"] = bool(self._CLASSIFIED_BOARD.search(combined))

        # Compensation quality signals
        features["has_clawback"] = _bool_hit(combined, [
            r"clawback", r"compensation\s+recovery", r"recoupment\s+policy",
        ])
        features["has_performance_comp"] = _bool_hit(combined, [
            r"performance[\s\-]based", r"long[\s\-]term\s+incentive",
            r"performance\s+share", r"at[\s\-]risk\s+compensation",
        ])

        return features

    def _compute_governance_score(self, features: dict[str, Any]) -> tuple[float, dict]:
        """
        MSCI-simplified governance scoring: 0-10 scale.
        Each component scored 0-10, then weighted.
        """
        components: dict[str, float] = {}

        # 1. Board independence (weight: 25%)
        indep = features.get("board_independence_pct")
        if indep is not None:
            if indep >= 80:
                components["board_independence"] = 10.0
            elif indep >= 70:
                components["board_independence"] = 8.0
            elif indep >= 60:
                components["board_independence"] = 6.0
            elif indep >= 50:
                components["board_independence"] = 4.0
            else:
                components["board_independence"] = 2.0
        else:
            components["board_independence"] = 5.0  # neutral if no data

        # 2. CEO duality (weight: 15%): duality = bad
        duality = features.get("ceo_duality")
        if duality is True:
            components["ceo_duality"] = 3.0   # negative
        elif duality is False:
            components["ceo_duality"] = 9.0   # positive
        else:
            components["ceo_duality"] = 5.0

        # 3. Say-on-pay (weight: 15%)
        sop = features.get("say_on_pay_pct")
        if sop is not None:
            if sop >= 90:
                components["say_on_pay"] = 10.0
            elif sop >= 80:
                components["say_on_pay"] = 8.0
            elif sop >= 70:
                components["say_on_pay"] = 6.0
            else:
                components["say_on_pay"] = 3.0
        else:
            components["say_on_pay"] = 5.0

        # 4. Board diversity (weight: 15%)
        diversity = features.get("board_diversity_pct")
        if diversity is not None:
            if diversity >= 40:
                components["board_diversity"] = 10.0
            elif diversity >= 30:
                components["board_diversity"] = 8.0
            elif diversity >= 20:
                components["board_diversity"] = 6.0
            elif diversity >= 10:
                components["board_diversity"] = 4.0
            else:
                components["board_diversity"] = 2.0
        else:
            components["board_diversity"] = 5.0

        # 5. Classified board (weight: 10%): classified = bad
        if features.get("has_classified_board"):
            components["classified_board"] = 3.0
        else:
            components["classified_board"] = 8.0

        # 6. Poison pill (weight: 10%): pill = bad
        if features.get("has_poison_pill"):
            components["poison_pill"] = 3.0
        else:
            components["poison_pill"] = 8.0

        # 7. Compensation quality (weight: 10%)
        comp_score = 5.0
        if features.get("has_performance_comp"):
            comp_score += 2.5
        if features.get("has_clawback"):
            comp_score += 2.5
        components["comp_quality"] = min(10.0, comp_score)

        # 8. CEO pay ratio (weight: 0% — informational, small penalty if extreme)
        pay_ratio = features.get("ceo_pay_ratio")
        if pay_ratio is not None:
            if pay_ratio > 500:
                components["pay_ratio"] = 3.0
            elif pay_ratio > 300:
                components["pay_ratio"] = 5.0
            elif pay_ratio > 100:
                components["pay_ratio"] = 7.0
            else:
                components["pay_ratio"] = 9.0
        else:
            components["pay_ratio"] = 5.0

        # Weighted composite (weights sum to 1)
        weights = {
            "board_independence": 0.25,
            "ceo_duality":        0.15,
            "say_on_pay":         0.15,
            "board_diversity":    0.15,
            "classified_board":   0.10,
            "poison_pill":        0.10,
            "comp_quality":       0.10,
        }
        # pay_ratio is informational, included at 0% weight but stored
        score = sum(
            components.get(k, 5.0) * w for k, w in weights.items()
        )
        return round(score, 2), components

    async def score_governance(self, ticker: str) -> GovernanceScore:
        """Score governance for a single ticker from EDGAR filings."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            cik = await self._edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"Could not resolve CIK for {ticker}")
                return GovernanceScore(
                    ticker=ticker, score=5.0, grade="BB",
                    warnings=warnings,
                    details={"error": "CIK not found"},
                )

            text_10k, text_proxy, sic_code, company_name, fiscal_year = (
                await self._edgar.get_latest_filing_texts(cik, client)
            )

        if not text_proxy and not text_10k:
            warnings.append("No filing text retrieved — using neutral scores")

        features = self._extract_governance_features(text_proxy, text_10k)
        score, components = self._compute_governance_score(features)

        return GovernanceScore(
            ticker=ticker,
            score=score,
            grade=_score_to_grade_10(score),
            board_independence_pct=features.get("board_independence_pct"),
            board_diversity_pct=features.get("board_diversity_pct"),
            ceo_duality=features.get("ceo_duality"),
            say_on_pay_pct=features.get("say_on_pay_pct"),
            has_poison_pill=features.get("has_poison_pill"),
            has_classified_board=features.get("has_classified_board"),
            ceo_pay_ratio=features.get("ceo_pay_ratio"),
            details={
                "components": components,
                "company_name": company_name,
                "sic_code": sic_code,
                "fiscal_year": fiscal_year,
                "raw_features": {
                    k: v for k, v in features.items()
                    if not isinstance(v, bool)
                },
            },
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# EnvironmentalScorer
# ---------------------------------------------------------------------------

# Environmental keywords for 10-K analysis
_ENV_KEYWORDS: dict[str, list[str]] = {
    "carbon_emissions": [
        "scope 1", "scope 2", "scope 3", "scope one", "scope two", "scope three",
        "greenhouse gas", "ghg emissions", "carbon emissions", "co2 emissions",
        "metric tons of co2", "tco2e", "mt co2",
    ],
    "net_zero": [
        "net zero", "net-zero", "carbon neutral", "carbon neutrality",
        "carbon negative", "climate pledge", "net zero by 2050",
        "carbon free by", "achieve net zero",
    ],
    "science_based_target": [
        "science based targets", "science-based targets", "sbti", "sbti-validated",
        "1.5 degree pathway", "1.5°c pathway", "2 degree target",
        "paris agreement aligned",
    ],
    "cdp": [
        "cdp", "carbon disclosure project", "cdp questionnaire",
        "cdp climate", "cdp score", "cdp a-list",
    ],
    "renewable_energy": [
        "renewable energy", "solar energy", "wind energy", "clean energy",
        "renewable electricity", "100% renewable", "renewable power purchase",
        "ppa", "clean power",
    ],
    "water": [
        "water consumption", "water usage", "water withdrawal",
        "water stewardship", "water recycled", "water intensity",
    ],
    "waste": [
        "waste diversion", "landfill diversion", "recycling rate",
        "zero waste", "hazardous waste", "waste reduction",
    ],
    "energy_efficiency": [
        "energy efficiency", "energy reduction", "energy consumption",
        "energy intensity", "leed certified", "energy star",
        "building efficiency",
    ],
    "environmental_litigation": [
        "epa enforcement", "environmental violation", "clean air act",
        "clean water act", "superfund", "cercla", "environmental penalty",
        "environmental fine", "environmental lawsuit",
    ],
}

# SASB sector material E topics: sectors where E score gets a multiplier boost
_SECTOR_E_MATERIALITY: dict[str, float] = {
    "energy":       1.2,
    "materials":    1.15,
    "utilities":    1.15,
    "industrials":  1.05,
    "default":      1.0,
    "financials":   0.85,
    "information_technology": 0.90,
}


class EnvironmentalScorer:
    """
    Scores Environmental (E) pillar from 10-K disclosures.
    SASB-adjusted scoring based on sector materiality.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """Count keyword hits and detect environmental signals."""
        text_lower = text.lower()
        hits: dict[str, int] = {}
        for category, keywords in _ENV_KEYWORDS.items():
            hits[category] = _count_keyword_hits(text_lower, keywords)

        # Key boolean signals
        signals = {
            "has_carbon_disclosure": (
                hits.get("carbon_emissions", 0) >= 2
            ),
            "has_cdp_participation": (
                hits.get("cdp", 0) >= 1
            ),
            "has_net_zero_target": (
                hits.get("net_zero", 0) >= 1
            ),
            "has_science_based_target": (
                hits.get("science_based_target", 0) >= 1
            ),
            "has_renewable_energy": (
                hits.get("renewable_energy", 0) >= 2
            ),
            "has_water_disclosure": (
                hits.get("water", 0) >= 2
            ),
            "has_waste_disclosure": (
                hits.get("waste", 0) >= 2
            ),
            "has_energy_efficiency": (
                hits.get("energy_efficiency", 0) >= 2
            ),
            "has_environmental_litigation": (
                hits.get("environmental_litigation", 0) >= 1
            ),
        }
        return {"hits": hits, "signals": signals}

    def _compute_score(
        self, analysis: dict[str, Any], sector: str
    ) -> float:
        """
        Compute E score 0-10 from keyword signals.
        """
        sig = analysis["signals"]
        score = 3.0  # baseline: all companies get base credit

        # Carbon disclosure: most fundamental E metric
        if sig["has_carbon_disclosure"]:
            score += 1.5
        # Science Based Target: highest credibility
        if sig["has_science_based_target"]:
            score += 1.5
        elif sig["has_net_zero_target"]:
            score += 0.8  # partial credit for self-stated target
        # CDP participation proxy
        if sig["has_cdp_participation"]:
            score += 0.7
        # Renewable energy commitment
        if sig["has_renewable_energy"]:
            score += 0.5
        # Water disclosure
        if sig["has_water_disclosure"]:
            score += 0.3
        # Waste disclosure
        if sig["has_waste_disclosure"]:
            score += 0.3
        # Energy efficiency programs
        if sig["has_energy_efficiency"]:
            score += 0.4
        # Environmental litigation — penalty
        if sig["has_environmental_litigation"]:
            score -= 1.0

        # SASB materiality adjustment
        materiality_mult = _SECTOR_E_MATERIALITY.get(
            sector, _SECTOR_E_MATERIALITY["default"]
        )
        # Sectors with HIGH E materiality: rescale so that average company
        # gets a harder time achieving a given score
        score = score * materiality_mult

        return round(min(10.0, max(0.0, score)), 2)

    async def score_environmental(self, ticker: str) -> EnvironmentalScore:
        """Score environmental pillar for a ticker."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            cik = await self._edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"Could not resolve CIK for {ticker}")
                return EnvironmentalScore(
                    ticker=ticker, score=5.0, grade="BB",
                    warnings=warnings,
                )

            text_10k, _proxy, sic_code, company_name, _year = (
                await self._edgar.get_latest_filing_texts(cik, client)
            )

        sector = _sic_to_sector(sic_code)
        if not text_10k:
            warnings.append("No 10-K text retrieved — using neutral score")
            return EnvironmentalScore(
                ticker=ticker, score=4.0, grade="B",
                sector=sector, warnings=warnings,
            )

        analysis = self._analyze_text(text_10k)
        sig = analysis["signals"]
        score = self._compute_score(analysis, sector)

        return EnvironmentalScore(
            ticker=ticker,
            score=score,
            grade=_score_to_grade_10(score),
            has_carbon_disclosure=sig["has_carbon_disclosure"],
            has_cdp_participation=sig["has_cdp_participation"],
            has_net_zero_target=sig["has_net_zero_target"],
            has_science_based_target=sig["has_science_based_target"],
            has_environmental_litigation=sig["has_environmental_litigation"],
            sector=sector,
            keyword_hits=analysis["hits"],
            details={
                "company_name": company_name,
                "sic_code": sic_code,
                "signals": sig,
            },
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# SocialScorer
# ---------------------------------------------------------------------------

_SOCIAL_KEYWORDS: dict[str, list[str]] = {
    "diversity": [
        "diversity, equity and inclusion", "dei", "board diversity",
        "racial diversity", "gender diversity", "women in leadership",
        "underrepresented", "inclusive workplace", "ethnic diversity",
        "minority representation",
    ],
    "human_capital": [
        "employee development", "training and development", "talent development",
        "employee engagement", "human capital", "workforce development",
        "employee wellbeing", "learning and development", "turnover rate",
        "employee retention",
    ],
    "safety": [
        "workplace safety", "osha", "recordable incidents", "lost time",
        "safety performance", "total recordable", "workplace injury",
        "health and safety", "trir", "dart rate",
    ],
    "supply_chain": [
        "supply chain", "supplier code of conduct", "supplier audit",
        "responsible sourcing", "vendor standards", "supplier standards",
        "supply chain due diligence",
    ],
    "conflict_minerals": [
        "conflict minerals", "dodd-frank section 1502", "tin, tantalum, tungsten",
        "3tg", "responsible minerals", "cmrt",
    ],
    "labor_relations": [
        "collective bargaining", "labor relations", "union",
        "workers council", "employee representation",
    ],
    "community": [
        "community investment", "charitable contributions", "corporate giving",
        "community engagement", "philanthropic", "social impact",
        "local community", "foundation", "volunteer",
    ],
    "pay_equity": [
        "pay equity", "gender pay gap", "equal pay", "pay parity",
        "compensation equity", "fair pay",
    ],
    "employee_benefits": [
        "401k", "retirement plan", "stock option", "employee stock",
        "health benefits", "medical insurance", "parental leave",
        "paid leave", "employee benefit",
    ],
}


class SocialScorer:
    """
    Scores Social (S) pillar from 10-K and DEF 14A text.
    Checks for D&I, supply chain, human capital, and community investment.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()

    async def _check_nlrb_cases(self, company_name: str) -> bool:
        """
        Check NLRB public API for recent cases against the company.
        NLRB provides a public case search API.
        """
        if not company_name or len(company_name) < 3:
            return False
        try:
            # NLRB case search — public API
            search_name = company_name[:30].strip()
            url = (
                f"https://www.nlrb.gov/api/cases"
                f"?respondent={search_name}&page_size=5"
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, headers=_HEADERS)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    return len(results) > 0
        except Exception:
            pass
        return False

    def _analyze_text(self, text_10k: str, text_proxy: str) -> dict[str, Any]:
        combined = (text_10k + " " + text_proxy).lower()
        hits: dict[str, int] = {}
        for category, keywords in _SOCIAL_KEYWORDS.items():
            hits[category] = _count_keyword_hits(combined, keywords)

        signals = {
            "has_diversity_disclosure":        hits.get("diversity", 0) >= 3,
            "has_supply_chain_audit":          hits.get("supply_chain", 0) >= 2,
            "has_conflict_minerals_disclosure": hits.get("conflict_minerals", 0) >= 1,
            "has_human_capital_metrics":       hits.get("human_capital", 0) >= 3,
            "has_safety_disclosure":           hits.get("safety", 0) >= 3,
            "has_pay_equity_disclosure":       hits.get("pay_equity", 0) >= 1,
            "has_community_investment":        hits.get("community", 0) >= 2,
            "has_employee_benefits_disclosure": hits.get("employee_benefits", 0) >= 3,
        }
        return {"hits": hits, "signals": signals}

    def _compute_score(self, analysis: dict[str, Any], has_nlrb: bool) -> float:
        sig = analysis["signals"]
        score = 3.0  # baseline

        if sig["has_diversity_disclosure"]:
            score += 1.2
        if sig["has_human_capital_metrics"]:
            score += 1.0
        if sig["has_supply_chain_audit"]:
            score += 0.8
        if sig["has_conflict_minerals_disclosure"]:
            score += 0.5
        if sig["has_safety_disclosure"]:
            score += 0.7
        if sig["has_pay_equity_disclosure"]:
            score += 0.6
        if sig["has_community_investment"]:
            score += 0.5
        if sig["has_employee_benefits_disclosure"]:
            score += 0.4
        if has_nlrb:
            score -= 1.5  # labor relations penalty

        return round(min(10.0, max(0.0, score)), 2)

    async def score_social(self, ticker: str) -> SocialScore:
        """Score social pillar for a ticker."""
        ticker = ticker.upper()
        warnings: list[str] = []

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            cik = await self._edgar.resolve_cik(ticker, client)
            if not cik:
                warnings.append(f"Could not resolve CIK for {ticker}")
                return SocialScore(
                    ticker=ticker, score=5.0, grade="BB", warnings=warnings,
                )

            text_10k, text_proxy, sic_code, company_name, _year = (
                await self._edgar.get_latest_filing_texts(cik, client)
            )

        if not text_10k and not text_proxy:
            warnings.append("No filing text — neutral score assigned")
            return SocialScore(
                ticker=ticker, score=4.0, grade="B", warnings=warnings,
            )

        analysis = self._analyze_text(text_10k, text_proxy)
        has_nlrb = await self._check_nlrb_cases(company_name)
        score = self._compute_score(analysis, has_nlrb)
        sig = analysis["signals"]

        return SocialScore(
            ticker=ticker,
            score=score,
            grade=_score_to_grade_10(score),
            has_diversity_disclosure=sig["has_diversity_disclosure"],
            has_supply_chain_audit=sig["has_supply_chain_audit"],
            has_conflict_minerals_disclosure=sig["has_conflict_minerals_disclosure"],
            has_human_capital_metrics=sig["has_human_capital_metrics"],
            has_nlrb_cases=has_nlrb,
            has_community_investment=sig["has_community_investment"],
            keyword_hits=analysis["hits"],
            details={
                "company_name": company_name,
                "sic_code": sic_code,
                "signals": sig,
            },
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# ESGCompositeRating
# ---------------------------------------------------------------------------

class ESGCompositeRating:
    """
    Combines E, S, G scores into a composite ESG rating.
    Uses sector-specific weights; applies controversy penalty.
    """

    def __init__(self) -> None:
        self._edgar = _EDGARClient()
        self._gov_scorer = GovernanceScorer()
        self._env_scorer = EnvironmentalScorer()
        self._soc_scorer = SocialScorer()

    def _apply_controversy_penalty(
        self, text_10k: str, text_proxy: str, composite: float
    ) -> float:
        """
        Reduce composite by up to 1 point for high controversy signals.
        Looks for SEC enforcement, major settlements, shareholder lawsuits.
        """
        controversy_terms = [
            r"sec\s+enforcement", r"securities\s+fraud",
            r"class\s+action\s+lawsuit", r"settled\s+(?:with|for)\s+\$",
            r"doj\s+investigation", r"regulatory\s+fine",
            r"restatement\s+of\s+(?:financial|earnings)",
        ]
        combined = (text_10k + " " + text_proxy).lower()
        hits = sum(1 for p in controversy_terms if re.search(p, combined))
        if hits >= 3:
            return 1.0
        if hits >= 1:
            return 0.5
        return 0.0

    async def get_rating(self, ticker: str) -> ESGRating:
        """
        Full ESG rating pipeline: score E, S, G; apply sector weights; composite.
        """
        ticker = ticker.upper()

        # Resolve CIK and get filing texts once (share across scorers)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            edgar = _EDGARClient()
            cik = await edgar.resolve_cik(ticker, client)
            if not cik:
                raise ValueError(f"Cannot resolve CIK for {ticker}")
            text_10k, text_proxy, sic_code, company_name, fiscal_year = (
                await edgar.get_latest_filing_texts(cik, client)
            )

        sector = _sic_to_sector(sic_code)
        weights = SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS["default"])

        # Run scorers concurrently
        g_task = asyncio.create_task(self._gov_scorer.score_governance(ticker))
        e_task = asyncio.create_task(self._env_scorer.score_environmental(ticker))
        s_task = asyncio.create_task(self._soc_scorer.score_social(ticker))
        g_result, e_result, s_result = await asyncio.gather(g_task, e_task, s_task)

        raw_composite = (
            e_result.score * weights["E"]
            + s_result.score * weights["S"]
            + g_result.score * weights["G"]
        )

        penalty = self._apply_controversy_penalty(text_10k, text_proxy, raw_composite)
        composite = round(max(0.0, min(10.0, raw_composite - penalty)), 2)

        # Negative screen check
        sic = str(sic_code).zfill(4) if sic_code else ""
        negative_flag = False
        negative_reason: Optional[str] = None
        if sic in _WEAPONS_SIC:
            negative_flag, negative_reason = True, "Weapons manufacturer (SIC)"
        elif sic in _TOBACCO_SIC:
            negative_flag, negative_reason = True, "Tobacco (SIC)"
        elif sic in _GAMBLING_SIC:
            negative_flag, negative_reason = True, "Gambling (SIC)"
        elif sic in _FOSSIL_SIC:
            negative_flag, negative_reason = True, "Fossil fuel (SIC)"

        logger.info(
            "ESG composite rated",
            ticker=ticker,
            e=e_result.score,
            s=s_result.score,
            g=g_result.score,
            composite=composite,
            sector=sector,
        )

        return ESGRating(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            sector=sector,
            composite=composite,
            e_score=e_result.score,
            s_score=s_result.score,
            g_score=g_result.score,
            grade=_score_to_grade_10(composite),
            controversy_penalty=penalty,
            weights_used=weights,
            filing_year=fiscal_year,
            has_negative_screen_flag=negative_flag,
            negative_screen_reason=negative_reason,
        )


# ---------------------------------------------------------------------------
# ESGScreener
# ---------------------------------------------------------------------------

# Commonly screened S&P 500 tickers for demonstration
_DEFAULT_SCREEN_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "AVGO", "COST", "PEP", "KO", "TMO", "ACN", "BAC", "LLY",
    "NKE", "CRM", "WMT", "DIS", "ADBE", "NFLX", "TXN", "VZ", "INTC",
    "NEE", "PM", "RTX", "AMD", "QCOM", "IBM", "GE", "CAT", "BA", "MMM",
]


class ESGScreener:
    """
    ESG-based investment screening with negative/positive screens and momentum.
    """

    def __init__(self) -> None:
        self._rater = ESGCompositeRating()

    async def _rate_ticker_safe(self, ticker: str) -> Optional[ESGRating]:
        try:
            return await self._rater.get_rating(ticker)
        except Exception as exc:
            logger.warning("ESG rating failed", ticker=ticker, error=str(exc))
            return None

    async def screen(
        self,
        tickers: list[str],
        min_esg_score: float = 6.0,
        exclude_negative_screens: bool = True,
        min_governance: float = 5.0,
        positive_momentum: bool = False,
        max_concurrent: int = 5,
    ) -> ESGScreenResult:
        """
        Screen a list of tickers by ESG criteria.

        Args:
            tickers: list of ticker symbols to screen
            min_esg_score: minimum composite ESG score (0-10) to pass
            exclude_negative_screens: remove weapons/tobacco/gambling/fossil fuel
            min_governance: minimum G score (0-10)
            positive_momentum: placeholder for future momentum filter
            max_concurrent: max concurrent EDGAR requests
        """
        if not tickers:
            tickers = _DEFAULT_SCREEN_UNIVERSE[:20]

        # Rate all tickers with bounded concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _rate_with_sem(t: str) -> Optional[ESGRating]:
            async with semaphore:
                return await self._rate_ticker_safe(t)

        tasks = [_rate_with_sem(t) for t in tickers]
        results_raw = await asyncio.gather(*tasks)
        ratings = [r for r in results_raw if r is not None]

        passed: list[str] = []
        failed: list[str] = []
        negative_screened: list[str] = []
        passed_ratings: list[ESGRating] = []

        for rating in ratings:
            if exclude_negative_screens and rating.has_negative_screen_flag:
                negative_screened.append(rating.ticker)
                continue
            if rating.composite < min_esg_score:
                failed.append(rating.ticker)
                continue
            if rating.g_score < min_governance:
                failed.append(rating.ticker)
                continue
            passed.append(rating.ticker)
            passed_ratings.append(rating)

        # Sort passed by composite descending
        passed_ratings.sort(key=lambda r: r.composite, reverse=True)

        return ESGScreenResult(
            tickers_passed=passed,
            tickers_failed=failed,
            negative_screened=negative_screened,
            ratings=passed_ratings,
            screener_params={
                "min_esg_score": min_esg_score,
                "exclude_negative_screens": exclude_negative_screens,
                "min_governance": min_governance,
                "positive_momentum": positive_momentum,
                "universe_size": len(tickers),
            },
        )


# ---------------------------------------------------------------------------
# SectorESGBenchmark
# ---------------------------------------------------------------------------

_SECTOR_REPRESENTATIVE_TICKERS: dict[str, list[str]] = {
    "energy":                  ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "materials":               ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "ALB"],
    "industrials":             ["GE", "CAT", "BA", "MMM", "HON", "UPS", "RTX"],
    "utilities":               ["NEE", "SO", "DUK", "AEP", "EXC", "XEL", "WEC"],
    "consumer_staples":        ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO"],
    "consumer_discretionary":  ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT"],
    "health_care":             ["JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "BMY"],
    "financials":              ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C"],
    "information_technology":  ["AAPL", "MSFT", "NVDA", "AVGO", "TXN", "QCOM", "INTC"],
    "communication_services":  ["GOOGL", "META", "DIS", "NFLX", "VZ", "T", "CMCSA"],
    "real_estate":             ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O"],
}


class SectorESGBenchmark:
    """
    Compute sector-level ESG scores across GICS sectors.
    Identifies leaders and laggards; provides climate transition risk.
    """

    def __init__(self) -> None:
        self._rater = ESGCompositeRating()

    async def benchmark_sector(
        self, sector: str, max_tickers: int = 5
    ) -> SectorESGBenchmarkResult:
        """Score a sector's representative tickers and aggregate."""
        tickers = _SECTOR_REPRESENTATIVE_TICKERS.get(sector, [])[:max_tickers]
        if not tickers:
            raise ValueError(f"Unknown sector: {sector}")

        ratings: list[ESGRating] = []
        for t in tickers:
            try:
                r = await self._rater.get_rating(t)
                ratings.append(r)
                await asyncio.sleep(_RATE_DELAY)
            except Exception as exc:
                logger.warning("Sector benchmark skip", ticker=t, error=str(exc))

        if not ratings:
            return SectorESGBenchmarkResult(
                sector=sector, n_companies=0,
                avg_composite=0.0, avg_e=0.0, avg_s=0.0, avg_g=0.0,
                leaders=[], laggards=[],
                climate_transition_risk=CLIMATE_TRANSITION_RISK.get(sector, "MEDIUM"),
            )

        sorted_r = sorted(ratings, key=lambda r: r.composite, reverse=True)
        n = len(ratings)

        return SectorESGBenchmarkResult(
            sector=sector,
            n_companies=n,
            avg_composite=round(sum(r.composite for r in ratings) / n, 2),
            avg_e=round(sum(r.e_score for r in ratings) / n, 2),
            avg_s=round(sum(r.s_score for r in ratings) / n, 2),
            avg_g=round(sum(r.g_score for r in ratings) / n, 2),
            leaders=[r.ticker for r in sorted_r[:3]],
            laggards=[r.ticker for r in sorted_r[-3:]],
            climate_transition_risk=CLIMATE_TRANSITION_RISK.get(sector, "MEDIUM"),
        )

    async def benchmark_all_sectors(self) -> list[SectorESGBenchmarkResult]:
        """Benchmark all 11 GICS sectors."""
        results = []
        for sector in _SECTOR_REPRESENTATIVE_TICKERS:
            try:
                result = await self.benchmark_sector(sector, max_tickers=4)
                results.append(result)
            except Exception as exc:
                logger.warning("Sector benchmark failed", sector=sector, error=str(exc))
        return results

    def sector_laggards_and_leaders(
        self, benchmarks: list[SectorESGBenchmarkResult]
    ) -> dict[str, Any]:
        """Identify cross-sector ESG leaders and laggards."""
        if not benchmarks:
            return {}
        sorted_b = sorted(benchmarks, key=lambda b: b.avg_composite, reverse=True)
        return {
            "overall_leaders": [b.sector for b in sorted_b[:3]],
            "overall_laggards": [b.sector for b in sorted_b[-3:]],
            "highest_climate_risk_sectors": [
                b.sector for b in benchmarks
                if b.climate_transition_risk == "HIGH"
            ],
            "sector_rankings": [
                {"sector": b.sector, "avg_composite": b.avg_composite}
                for b in sorted_b
            ],
        }


# ---------------------------------------------------------------------------
# Controversy helper (GDELT integration stub)
# ---------------------------------------------------------------------------

async def get_controversy_score(ticker: str) -> float:
    """
    Fetch controversy signal from GDELT or internal controversy monitor.
    Returns 0-10 scale (10 = most controversial).
    Falls back to 0 if GDELT unavailable.
    """
    try:
        # GDELT GKG API: search for negative news about company
        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={ticker}+ESG+controversy&mode=artlist&maxrecords=10"
            f"&format=json&timespan=1month"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=_HEADERS)
            if r.status_code == 200:
                data = r.json()
                articles = data.get("articles", [])
                # Negative tone scoring: GDELT tone < -5 = negative
                tones = [
                    a.get("tone", {}).get("tone", 0.0)
                    for a in articles if "tone" in a
                ]
                if tones:
                    avg_tone = sum(tones) / len(tones)
                    # Convert negative tone to controversy score
                    if avg_tone < -10:
                        return 8.0
                    if avg_tone < -5:
                        return 5.0
                    return 2.0
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

esg_router = APIRouter(prefix="/esg", tags=["ESG Ratings"])

_governance_scorer  = GovernanceScorer()
_env_scorer         = EnvironmentalScorer()
_social_scorer      = SocialScorer()
_composite_rater    = ESGCompositeRating()
_screener           = ESGScreener()
_sector_benchmark   = SectorESGBenchmark()


@esg_router.get("/rating/{ticker}", response_model=ESGRating)
async def esg_rating(ticker: str) -> ESGRating:
    """Full ESG composite rating for a ticker (E+S+G weighted by sector)."""
    try:
        return await _composite_rater.get_rating(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESG rating failed: {e}")


@esg_router.get("/governance/{ticker}", response_model=GovernanceScore)
async def esg_governance(ticker: str) -> GovernanceScore:
    """Governance (G) score from EDGAR DEF 14A proxy filings."""
    try:
        return await _governance_scorer.score_governance(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Governance scoring failed: {e}")


@esg_router.get("/environmental/{ticker}", response_model=EnvironmentalScore)
async def esg_environmental(ticker: str) -> EnvironmentalScore:
    """Environmental (E) score from EDGAR 10-K filings."""
    try:
        return await _env_scorer.score_environmental(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environmental scoring failed: {e}")


@esg_router.get("/social/{ticker}", response_model=SocialScore)
async def esg_social(ticker: str) -> SocialScore:
    """Social (S) score from EDGAR 10-K and DEF 14A filings."""
    try:
        return await _social_scorer.score_social(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social scoring failed: {e}")


@esg_router.get("/screener", response_model=ESGScreenResult)
async def esg_screener(
    tickers: str = Query(
        default="AAPL,MSFT,GOOGL,JNJ,XOM,JPM",
        description="Comma-separated ticker list",
    ),
    min_esg_score: float = Query(default=6.0, ge=0, le=10),
    exclude_negative_screens: bool = Query(default=True),
    min_governance: float = Query(default=5.0, ge=0, le=10),
) -> ESGScreenResult:
    """Screen a universe of tickers by ESG criteria."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) > 30:
        raise HTTPException(status_code=400, detail="Max 30 tickers per request")
    try:
        return await _screener.screen(
            tickers=ticker_list,
            min_esg_score=min_esg_score,
            exclude_negative_screens=exclude_negative_screens,
            min_governance=min_governance,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screener failed: {e}")


@esg_router.get(
    "/sector-benchmark",
    response_model=list[SectorESGBenchmarkResult],
)
async def sector_benchmark(
    sector: Optional[str] = Query(
        default=None,
        description="GICS sector name, e.g. 'energy'. Omit for all sectors.",
    ),
) -> list[SectorESGBenchmarkResult]:
    """ESG benchmark for one GICS sector or all 11 sectors."""
    try:
        if sector:
            result = await _sector_benchmark.benchmark_sector(sector.lower())
            return [result]
        return await _sector_benchmark.benchmark_all_sectors()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")


@esg_router.get("/controversy/{ticker}")
async def esg_controversy(ticker: str) -> dict[str, Any]:
    """
    Controversy score for a ticker from GDELT news sentiment.
    Returns 0-10 scale (10 = most controversial).
    """
    score = await get_controversy_score(ticker.upper())
    return {
        "ticker": ticker.upper(),
        "controversy_score": score,
        "severity": "HIGH" if score >= 7 else "MEDIUM" if score >= 4 else "LOW",
        "source": "GDELT GKG news sentiment",
        "as_of": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

async def _cli_demo(ticker: str = "MSFT") -> None:
    """Quick CLI demonstration of the ESG ratings engine."""
    print(f"\n{'='*60}")
    print(f"ESG RATINGS ENGINE — {ticker}")
    print(f"{'='*60}")

    rater = ESGCompositeRating()
    gov   = GovernanceScorer()
    env   = EnvironmentalScorer()
    soc   = SocialScorer()

    print(f"\n[1/4] Fetching Governance score...")
    g = await gov.score_governance(ticker)
    print(f"  Governance Score: {g.score}/10  Grade: {g.grade}")
    print(f"  Board Independence: {g.board_independence_pct}%")
    print(f"  CEO Duality: {g.ceo_duality}  Classified Board: {g.has_classified_board}")

    print(f"\n[2/4] Fetching Environmental score...")
    e = await env.score_environmental(ticker)
    print(f"  Environmental Score: {e.score}/10  Grade: {e.grade}")
    print(f"  Carbon Disclosure: {e.has_carbon_disclosure}")
    print(f"  Science Based Target: {e.has_science_based_target}")

    print(f"\n[3/4] Fetching Social score...")
    s = await soc.score_social(ticker)
    print(f"  Social Score: {s.score}/10  Grade: {s.grade}")
    print(f"  Diversity Disclosure: {s.has_diversity_disclosure}")
    print(f"  NLRB Cases: {s.has_nlrb_cases}")

    print(f"\n[4/4] Computing Composite ESG Rating...")
    rating = await rater.get_rating(ticker)
    print(f"  Composite: {rating.composite}/10  Grade: {rating.grade}")
    print(f"  Sector: {rating.sector}")
    print(f"  Weights: E={rating.weights_used.get('E'):.0%}  "
          f"S={rating.weights_used.get('S'):.0%}  "
          f"G={rating.weights_used.get('G'):.0%}")
    print(f"  Controversy Penalty: -{rating.controversy_penalty}")
    print(f"  Negative Screen: {rating.has_negative_screen_flag} "
          f"({rating.negative_screen_reason or 'N/A'})")
    print(f"\n{rating.disclaimer}")


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    asyncio.run(_cli_demo(t))


# ---------------------------------------------------------------------------
# dim_102 wave-9 additions: ESG momentum, greenwashing risk, controversy-adjusted ESG
# ---------------------------------------------------------------------------


def compute_esg_momentum(
    current_score: float,
    prior_quarter_score: float,
) -> dict:
    """
    Compute ESG score momentum as the quarter-over-quarter delta.

    A positive delta indicates improving ESG performance (momentum signal).

    Parameters
    ----------
    current_score       : ESG composite score this quarter (0–10)
    prior_quarter_score : ESG composite score last quarter (0–10)

    Returns
    -------
    dict with keys:
        delta          : score change (positive = improving)
        momentum_signal: "positive" | "neutral" | "negative"
        pct_change     : percentage change relative to prior score
    """
    delta = round(current_score - prior_quarter_score, 4)
    if prior_quarter_score > 0:
        pct_change = round(delta / prior_quarter_score * 100, 2)
    else:
        pct_change = 0.0
    if delta > 0:
        momentum_signal = "positive"
    elif delta < 0:
        momentum_signal = "negative"
    else:
        momentum_signal = "neutral"
    return {
        "delta": delta,
        "momentum_signal": momentum_signal,
        "pct_change": pct_change,
        "current_score": current_score,
        "prior_quarter_score": prior_quarter_score,
    }


def detect_greenwashing_risk(
    marketing_claims_score: float,
    esg_actual_score: float,
) -> dict:
    """
    Detect potential greenwashing risk when marketing claims exceed actual ESG performance.

    If marketing_claims_score exceeds esg_actual_score by > 15 points → greenwashing flag.

    Parameters
    ----------
    marketing_claims_score : Score (0–100) for strength of company's marketing ESG claims
    esg_actual_score       : Actual measured ESG score (0–100) from structured data

    Returns
    -------
    dict with keys:
        gap               : difference between claims and actual (marketing - actual)
        greenwashing_risk : bool — True if gap > 15 pts
        risk_level        : "high" | "moderate" | "low" | "none"
    """
    gap = round(marketing_claims_score - esg_actual_score, 2)
    greenwashing_risk = gap > 15.0
    if gap > 25.0:
        risk_level = "high"
    elif gap > 15.0:
        risk_level = "moderate"
    elif gap > 5.0:
        risk_level = "low"
    else:
        risk_level = "none"
    return {
        "gap": gap,
        "greenwashing_risk": greenwashing_risk,
        "risk_level": risk_level,
        "marketing_claims_score": marketing_claims_score,
        "esg_actual_score": esg_actual_score,
    }


def compute_controversy_adjusted_esg(
    base_score: float,
    num_controversies: int,
    penalty_per_controversy: float = 0.05,
    max_penalty: float = 0.30,
) -> dict:
    """
    Compute controversy-adjusted ESG score.

    Formula:
        penalty = min(num_controversies * penalty_per_controversy, max_penalty)
        adjusted_score = base_score * (1 - penalty)

    Parameters
    ----------
    base_score              : Raw ESG composite score (0–10 or 0–100 scale)
    num_controversies       : Number of active controversies
    penalty_per_controversy : Penalty fraction per controversy (default 0.05 = 5%)
    max_penalty             : Maximum total penalty fraction (default 0.30 = 30%)

    Returns
    -------
    dict with keys:
        controversy_penalty    : fraction applied (0.0 – max_penalty)
        controversy_adjusted   : final adjusted score
        num_controversies      : input controversy count
        base_score             : input base score
    """
    controversy_penalty = min(num_controversies * penalty_per_controversy, max_penalty)
    adjusted_score = round(base_score * (1.0 - controversy_penalty), 4)
    return {
        "controversy_penalty": round(controversy_penalty, 4),
        "controversy_adjusted": adjusted_score,
        "num_controversies": num_controversies,
        "base_score": base_score,
    }
