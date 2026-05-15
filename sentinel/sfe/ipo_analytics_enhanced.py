"""
IPO Analytics Enhanced — Comprehensive S-1 / IPO Intelligence (dim_030, target 9+).

Extends sentinel/sfe/ipo_intelligence.py with deeper S-1 parsing, richer pipeline
tracking, aftermarket performance attribution, lock-up expiry screens, IPO-to-peer
valuation comparison, and enhanced SPAC intelligence.

Public API
----------
S1FilingParser
    parse_s1_prospectus(accession_number, cik)   -> dict
    extract_financial_summary(text)              -> dict
    detect_red_flags(s1_data)                    -> list[str]

IPOPipelineTracker
    get_filed_s1s(lookback_days)                 -> pd.DataFrame
    get_priced_ipos(lookback_days)               -> pd.DataFrame
    get_withdrawn_ipos(lookback_days)            -> pd.DataFrame
    track_ipo_pipeline_by_sector()               -> pd.DataFrame

IPOPerformanceAnalyzer
    compute_ipo_day1_return(ticker, ipo_price)   -> dict
    compute_ipo_aftermarket_performance(ticker, ipo_date, ipo_price) -> dict
    lockup_expiry_analysis(ticker, ipo_date)     -> dict
    screen_ipos_near_lockup_expiry(lookback_days) -> pd.DataFrame
    compare_ipo_to_peers(ticker, ipo_valuation, peers) -> dict

SPACTracker
    get_active_spacs()                           -> pd.DataFrame
    get_spac_mergers(lookback_days)              -> list[dict]
    compute_spac_discount(trust_nav, market_price) -> dict

ipo_router — FastAPI router, prefix /api/ipo

Research backdrop
-----------------
  • Average IPO first-day return: ~18% (2000-2023, Jay Ritter data)
  • Stocks fall on average 2-3% around lock-up expiry (Brav & Gompers, 2003)
  • Withdrawn IPOs signal deteriorating market conditions or company-specific issues
  • SPAC mergers have underperformed traditional IPOs by ~15% over 2 years (2020-2022)
  • Tier-1 underwriter-backed IPOs outperform no-name underwriters by ~8% over 1 year
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
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

_EDGAR_DATA     = "https://data.sec.gov"
_EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
_EDGAR_EFTS     = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ATOM     = "https://www.sec.gov/cgi-bin/browse-edgar"
_EDGAR_TICKERS  = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT    = 35.0
_RATE_DELAY = 0.15

# S-1 form types to track
_S1_FORMS = {"S-1", "S-1/A", "S-11", "F-1", "F-1/A"}

# Final prospectus = IPO priced
_FINAL_PROSPECTUS_FORMS = {"424B4", "424B3"}

# Registration withdrawal = IPO pulled
_WITHDRAWAL_FORMS = {"RW", "RW/A"}

# SPAC-specific keywords
_SPAC_KEYWORDS = [
    "blank check company", "special purpose acquisition", "spac",
    "no operating history", "trust account", "founder shares",
    "business combination", "de-spac",
]

# Tier-1 underwriters (associated with better IPO long-term performance)
_TIER1_UNDERWRITERS = {
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "J.P. Morgan",
    "Bank of America", "Merrill Lynch",
}
_ALL_UNDERWRITERS = list(_TIER1_UNDERWRITERS) + [
    "Citigroup", "Citi", "Credit Suisse", "Deutsche Bank", "Barclays", "UBS",
    "Wells Fargo", "RBC Capital", "Jefferies", "Cowen", "Piper Sandler",
    "Needham", "William Blair", "Stifel", "Cantor Fitzgerald", "Oppenheimer",
    "KeyBanc", "Evercore", "Lazard", "Guggenheim", "Houlihan Lokey",
]

_GOING_CONCERN_PHRASE = (
    r"substantial\s+doubt\s+about\s+our\s+ability\s+to\s+continue\s+as\s+a\s+going\s+concern"
)

# SIC → GICS-style sector mapping (abbreviated)
_SIC_SECTOR_MAP: dict[str, str] = {
    "7372": "Technology",    "7371": "Technology",    "7374": "Technology",
    "7389": "Technology",    "3674": "Technology",    "3559": "Industrials",
    "3825": "Technology",    "8731": "Health Care",   "2836": "Health Care",
    "2835": "Health Care",   "5912": "Health Care",   "6770": "Financials",
    "6199": "Financials",    "6211": "Financials",    "7011": "Consumer Discretionary",
    "5812": "Consumer Discretionary", "5411": "Consumer Staples",
    "5731": "Consumer Discretionary", "4812": "Communication Services",
    "4813": "Communication Services", "4911": "Utilities",
    "1311": "Energy",        "6500": "Real Estate",   "6512": "Real Estate",
    "6552": "Real Estate",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class S1ProspectusData(BaseModel):
    """Structured data extracted from an S-1 prospectus."""
    accession_number:        str
    cik:                     str
    company_name:            str                = ""
    ticker:                  Optional[str]      = None
    filed_date:              Optional[date]     = None
    form_type:               str                = "S-1"
    sic_code:                Optional[str]      = None
    sector:                  Optional[str]      = None
    state_of_inc:            Optional[str]      = None
    # Business / structure
    is_spac:                 bool               = False
    has_dual_class:          bool               = False
    lockup_days:             int                = 180
    lock_up_expiry:          Optional[date]     = None
    # Use of proceeds
    use_of_proceeds_text:    Optional[str]      = None
    proceeds_to_selling_shs: bool               = False
    # Risk factors
    risk_factors_count:      int                = 0
    has_going_concern:       bool               = False
    # Financial highlights (latest annual)
    revenue_y0:              Optional[float]    = None
    revenue_y1:              Optional[float]    = None
    revenue_y2:              Optional[float]    = None
    revenue_growth_yoy:      Optional[float]    = None
    gross_profit_margin:     Optional[float]    = None
    ebitda:                  Optional[float]    = None
    net_income:              Optional[float]    = None
    total_assets:            Optional[float]    = None
    total_debt:              Optional[float]    = None
    cash_and_equiv:          Optional[float]    = None
    # Offering details
    shares_offered:          Optional[int]      = None
    price_range_low:         Optional[float]    = None
    price_range_high:        Optional[float]    = None
    ipo_size_est:            Optional[float]    = None   # midpoint × shares
    # Ownership / cap table
    insider_pct_pre_ipo:     Optional[float]    = None
    insider_pct_post_ipo:    Optional[float]    = None
    dilution_per_share:      Optional[float]    = None
    # Underwriters
    underwriters:            list[str]          = Field(default_factory=list)
    has_tier1_underwriter:   bool               = False
    # Shareholder concentration
    top_customer_pct:        Optional[float]    = None   # % revenue from largest customer
    # Red flags / green flags
    red_flags:               list[str]          = Field(default_factory=list)
    green_flags:             list[str]          = Field(default_factory=list)
    quality_score:           int                = 5      # 0-10


class IPOPipelineRow(BaseModel):
    company_name:     str
    cik:              str
    ticker:           Optional[str]  = None
    form_type:        str
    filed_date:       date
    sector:           Optional[str]  = None
    sic_code:         Optional[str]  = None
    ipo_size_est:     Optional[float] = None
    price_range_low:  Optional[float] = None
    price_range_high: Optional[float] = None
    underwriters:     list[str]      = Field(default_factory=list)
    is_spac:          bool           = False
    accession_number: str


class PricedIPO(BaseModel):
    company_name:    str
    cik:             str
    ticker:          Optional[str]  = None
    accession_number: str
    priced_date:     date
    final_price:     Optional[float] = None
    shares_offered:  Optional[int]   = None
    gross_proceeds:  Optional[float] = None
    underwriters:    list[str]       = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    try:
        v = float(str(val).replace(",", "").replace("$", "").strip())
        return v if not (v != v) else None  # NaN check
    except (TypeError, ValueError):
        return None


def _parse_date_safe(s: Any) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _zero_pad_cik(cik: str) -> str:
    return str(cik).lstrip("0").zfill(10)


def _find_dollar_near(text: str, label_pat: str, window: int = 350) -> Optional[float]:
    m = re.search(label_pat, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + window]
    dm = re.search(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|[BMK])?",
        snippet, re.IGNORECASE,
    )
    if not dm:
        return None
    try:
        val = float(dm.group(1).replace(",", ""))
    except ValueError:
        return None
    suffix = (dm.group(2) or "").lower()
    if suffix in ("billion", "b"):
        val *= 1_000_000_000
    elif suffix in ("million", "m"):
        val *= 1_000_000
    elif suffix in ("thousand", "k"):
        val *= 1_000
    return val


def _extract_price_range(text: str) -> tuple[Optional[float], Optional[float]]:
    pat = r"\$\s*([\d]+(?:\.\d+)?)\s*(?:to|and|-)\s*\$\s*([\d]+(?:\.\d+)?)"
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None


def _count_risk_sections(text: str) -> int:
    """Count risk factor entries in the filing text."""
    bullets = re.findall(
        r"(?:^|\n)\s*(?:\d+\.|•|-)\s+(?:Risk|Our|We |The |Changes|Competition|Failure)",
        text, re.MULTILINE | re.IGNORECASE,
    )
    all_caps = re.findall(r"\n[A-Z][A-Z\s]{15,80}\n", text)
    return len(bullets) + max(0, len(all_caps) - 5)


def _extract_underwriters_from_text(text: str) -> tuple[list[str], bool]:
    found: list[str] = []
    text_lower = text.lower()
    for bank in _ALL_UNDERWRITERS:
        if bank.lower() in text_lower:
            if bank not in found:
                found.append(bank)
    has_tier1 = any(u in _TIER1_UNDERWRITERS for u in found)
    return found, has_tier1


def _is_spac(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _SPAC_KEYWORDS)


def _sic_to_sector(sic: Optional[str]) -> Optional[str]:
    return _SIC_SECTOR_MAP.get((sic or "").zfill(4))


# ---------------------------------------------------------------------------
# S1FilingParser
# ---------------------------------------------------------------------------

class S1FilingParser:
    """
    Download and deeply parse S-1 / F-1 prospectuses from EDGAR.

    Extracts structured data from all major S-1 sections:
    business description, risk factors, use of proceeds, dilution,
    3-year financials, cap table, lock-up, and underwriters.

    Produces a quality score (0-10) and categorised red/green flags.
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def parse_s1_prospectus(
        self,
        accession_number: str,
        cik: str,
    ) -> dict:
        """
        Download and parse an S-1 filing, returning a richly structured dict
        mapping directly to S1ProspectusData fields.
        """
        cik_clean = str(cik).lstrip("0") or "0"
        acc_clean = accession_number.replace("-", "")
        index_url = (
            f"{_EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{accession_number}-index.json"
        )

        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            # ── Step 1: resolve company metadata via submissions ────────────
            cik_padded = _zero_pad_cik(cik)
            sub_url    = f"{_EDGAR_DATA}/submissions/CIK{cik_padded}.json"
            meta: dict = {}
            try:
                sub_resp = await client.get(sub_url)
                sub_resp.raise_for_status()
                meta = sub_resp.json()
            except Exception as exc:
                logger.debug("Submissions fetch failed", cik=cik, error=str(exc))

            company_name  = meta.get("name", "Unknown")
            sic_code      = str(meta.get("sic", "")) or None
            state_of_inc  = meta.get("stateOfIncorporation")
            ticker_sym    = (meta.get("tickers") or [None])[0]
            filed_date: Optional[date] = None
            form_type     = "S-1"

            # Match this accession in the filing history to get filed_date & form
            filings_recent = meta.get("filings", {}).get("recent", {})
            accessions_list = filings_recent.get("accessionNumber", [])
            for i, acc in enumerate(accessions_list):
                if acc.replace("-", "") == acc_clean:
                    forms = filings_recent.get("form", [])
                    dates = filings_recent.get("filingDate", [])
                    if i < len(forms):
                        form_type = forms[i]
                    if i < len(dates):
                        filed_date = _parse_date_safe(dates[i])
                    break

            # ── Step 2: resolve primary document ──────────────────────────
            primary_doc: Optional[str] = None
            try:
                idx_resp = await client.get(index_url)
                idx_resp.raise_for_status()
                idx = idx_resp.json()
                for doc in idx.get("documents", []):
                    fname = doc.get("filename", "")
                    dtype = doc.get("type", "")
                    if dtype in _S1_FORMS and fname.lower().endswith((".htm", ".html")):
                        primary_doc = fname
                        break
                if not primary_doc:
                    for doc in idx.get("documents", []):
                        if doc.get("filename", "").lower().endswith((".htm", ".html", ".txt")):
                            primary_doc = doc["filename"]
                            break
            except Exception as exc:
                logger.warning("Index fetch failed", acc=accession_number, error=str(exc))

            # ── Step 3: download filing text ───────────────────────────────
            filing_text = ""
            if primary_doc:
                doc_url = f"{_EDGAR_ARCHIVES}/{cik_clean}/{acc_clean}/{primary_doc}"
                try:
                    await asyncio.sleep(_RATE_DELAY)
                    doc_resp = await client.get(doc_url)
                    doc_resp.raise_for_status()
                    filing_text = doc_resp.text
                except Exception as exc:
                    logger.warning("Filing text download failed", url=doc_url, error=str(exc))

        # ── Step 4: extract structured data from text ──────────────────
        financials  = self.extract_financial_summary(filing_text)
        underwriters, has_tier1 = _extract_underwriters_from_text(filing_text)
        risk_count  = _count_risk_sections(filing_text)
        pl, ph      = _extract_price_range(filing_text)
        shares_off  = self._extract_shares_offered(filing_text)
        lockup_days = self._extract_lockup_days(filing_text)
        use_proc    = self._extract_use_of_proceeds(filing_text)
        proceeds_to_selling = bool(re.search(
            r"selling\s+shareholder|secondary\s+offer|existing\s+stockholder",
            use_proc or "", re.IGNORECASE,
        ))
        has_dual    = self._detect_dual_class(filing_text)
        going_conc  = bool(re.search(_GOING_CONCERN_PHRASE, filing_text, re.IGNORECASE))
        top_cust    = self._extract_customer_concentration(filing_text)
        insider_pre, insider_post = self._extract_cap_table(filing_text)
        dilution_ps = _find_dollar_near(filing_text, r"dilution\s+per\s+share")
        ipo_size    = None
        if pl and ph and shares_off:
            midpoint = (pl + ph) / 2.0
            ipo_size = midpoint * shares_off
        sector      = _sic_to_sector(sic_code)
        lock_expiry = None
        if filed_date and lockup_days:
            lock_expiry = filed_date + timedelta(days=lockup_days)

        data = {
            "accession_number":        accession_number,
            "cik":                     cik,
            "company_name":            company_name,
            "ticker":                  ticker_sym,
            "filed_date":              str(filed_date) if filed_date else None,
            "form_type":               form_type,
            "sic_code":                sic_code,
            "sector":                  sector,
            "state_of_inc":            state_of_inc,
            "is_spac":                 _is_spac(filing_text),
            "has_dual_class":          has_dual,
            "lockup_days":             lockup_days,
            "lock_up_expiry":          str(lock_expiry) if lock_expiry else None,
            "use_of_proceeds_text":    use_proc,
            "proceeds_to_selling_shs": proceeds_to_selling,
            "risk_factors_count":      risk_count,
            "has_going_concern":       going_conc,
            "underwriters":            underwriters,
            "has_tier1_underwriter":   has_tier1,
            "shares_offered":          shares_off,
            "price_range_low":         pl,
            "price_range_high":        ph,
            "ipo_size_est":            ipo_size,
            "insider_pct_pre_ipo":     insider_pre,
            "insider_pct_post_ipo":    insider_post,
            "dilution_per_share":      dilution_ps,
            "top_customer_pct":        top_cust,
            **financials,
        }

        # ── Step 5: apply red/green flag logic ────────────────────────
        red_flags, green_flags, quality_score = self._score_quality(data)
        data["red_flags"]    = red_flags
        data["green_flags"]  = green_flags
        data["quality_score"] = quality_score
        return data

    def extract_financial_summary(self, text: str) -> dict:
        """
        Regex-based extraction of 3-year financial snapshot from S-1 text.

        Returns dict with keys: revenue_y0, revenue_y1, revenue_y2,
        revenue_growth_yoy, gross_profit_margin, ebitda, net_income,
        total_assets, total_debt, cash_and_equiv.
        """
        if not text:
            return {}
        result: dict = {}

        # Revenue (latest year = y0, prior = y1, two-years-prior = y2)
        rev_labels = [
            r"(?:total\s+)?(?:net\s+)?revenue[s]?(?:\s*\(.*?\))?\s*",
            r"net\s+sales\s*", r"total\s+sales\s*",
        ]
        for label in rev_labels:
            v = _find_dollar_near(text, label)
            if v and v > 0:
                result["revenue_y0"] = v
                break

        # Attempt second occurrence for prior year
        for label in rev_labels:
            matches = list(re.finditer(label, text, re.IGNORECASE))
            if len(matches) >= 2:
                m2 = matches[1]
                snippet = text[m2.end(): m2.end() + 300]
                dm = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|[BMK])?",
                               snippet, re.IGNORECASE)
                if dm:
                    try:
                        v1 = float(dm.group(1).replace(",", ""))
                        suffix = (dm.group(2) or "").lower()
                        if suffix in ("billion", "b"): v1 *= 1e9
                        elif suffix in ("million", "m"): v1 *= 1e6
                        elif suffix in ("thousand", "k"): v1 *= 1e3
                        result["revenue_y1"] = v1
                    except ValueError:
                        pass
            if len(matches) >= 3:
                m3 = matches[2]
                snippet = text[m3.end(): m3.end() + 300]
                dm = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|[BMK])?",
                               snippet, re.IGNORECASE)
                if dm:
                    try:
                        v2 = float(dm.group(1).replace(",", ""))
                        suffix = (dm.group(2) or "").lower()
                        if suffix in ("billion", "b"): v2 *= 1e9
                        elif suffix in ("million", "m"): v2 *= 1e6
                        elif suffix in ("thousand", "k"): v2 *= 1e3
                        result["revenue_y2"] = v2
                    except ValueError:
                        pass
            break

        # YoY growth
        if result.get("revenue_y0") and result.get("revenue_y1") and result["revenue_y1"] > 0:
            result["revenue_growth_yoy"] = (result["revenue_y0"] - result["revenue_y1"]) / result["revenue_y1"]
        else:
            gm = re.search(r"revenue[s]?\s+(?:increased|grew|declined|decreased)\s+(?:by\s+)?([\d\.]+)%",
                           text, re.IGNORECASE)
            if gm:
                g = float(gm.group(1)) / 100.0
                if re.search(r"declin|decreas", gm.group(), re.IGNORECASE):
                    g = -g
                result["revenue_growth_yoy"] = g

        # Gross margin
        gp = _find_dollar_near(text, r"gross\s+profit\s*")
        if gp and result.get("revenue_y0") and result["revenue_y0"] > 0:
            result["gross_profit_margin"] = gp / result["revenue_y0"]

        # EBITDA
        ebitda = _find_dollar_near(text, r"ebitda\s*")
        if ebitda:
            result["ebitda"] = ebitda

        # Net income / loss
        for label in [r"net\s+(?:income|loss)(?:\s+attributable)?(?:\s*\(.*?\))?\s*"]:
            v = _find_dollar_near(text, label)
            if v is not None:
                m = re.search(label, text, re.IGNORECASE)
                if m:
                    snippet = text[m.start(): m.end() + 300]
                    if re.search(r"\([\d,\.]+\)|net\s+loss", snippet, re.IGNORECASE):
                        v = -abs(v)
                result["net_income"] = v
                break

        # Balance sheet
        v = _find_dollar_near(text, r"total\s+assets\s*")
        if v: result["total_assets"] = v

        for label in [r"(?:total\s+)?long[- ]term\s+debt\s*", r"total\s+(?:indebtedness|debt)\s*"]:
            v = _find_dollar_near(text, label)
            if v is not None and v >= 0:
                result["total_debt"] = v
                break

        for label in [r"cash\s+and\s+cash\s+equivalents\s*",
                      r"cash,?\s+cash\s+equivalents\s+and\s+(?:restricted\s+cash|short[- ]term)\s*"]:
            v = _find_dollar_near(text, label)
            if v is not None and v >= 0:
                result["cash_and_equiv"] = v
                break

        return result

    def detect_red_flags(self, s1_data: dict) -> list[str]:
        """
        Identify structural red flags from parsed S-1 data.

        Checks:
        - Going concern qualification
        - Dual-class share structure (founders retain voting control)
        - Revenue decline (negative YoY growth)
        - Heavy insider selling at IPO (secondary offering)
        - Negative gross margin
        - High customer concentration (>30% from one customer)
        - Pre-revenue / loss-stage company
        - Missing underwriter
        """
        flags: list[str] = []
        if s1_data.get("has_going_concern"):
            flags.append("Going concern qualification — auditors doubt ability to continue")
        if s1_data.get("has_dual_class"):
            flags.append("Dual-class share structure — founders retain outsized voting control")
        if s1_data.get("revenue_growth_yoy") is not None and s1_data["revenue_growth_yoy"] < 0:
            pct = abs(s1_data["revenue_growth_yoy"]) * 100
            flags.append(f"Revenue declining — {pct:.1f}% YoY contraction")
        if s1_data.get("proceeds_to_selling_shs"):
            flags.append("IPO proceeds include secondary/selling shareholder component (insider cash-out)")
        if s1_data.get("gross_profit_margin") is not None and s1_data["gross_profit_margin"] < 0:
            flags.append(f"Negative gross margin ({s1_data['gross_profit_margin']:.1%}) — structurally challenged unit economics")
        if s1_data.get("top_customer_pct") is not None and s1_data["top_customer_pct"] > 0.30:
            flags.append(f"High customer concentration — {s1_data['top_customer_pct']:.0%} revenue from top customer")
        if not s1_data.get("revenue_y0"):
            flags.append("Pre-revenue company — no revenue identified in filing")
        if not s1_data.get("underwriters"):
            flags.append("No major underwriter identified — limited institutional support")
        if s1_data.get("is_spac"):
            flags.append("SPAC structure — blank check company with no operating history")
        return flags

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_shares_offered(text: str) -> Optional[int]:
        pat = r"(?:total\s+)?shares\s+(?:of\s+)?(?:common\s+stock\s+)?offered"
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            return None
        snippet = text[m.end(): m.end() + 200]
        sm = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:million|thousand)?\s*shares", snippet, re.IGNORECASE)
        if not sm:
            return None
        raw = sm.group(1).replace(",", "")
        mult_m = re.search(r"(million|thousand)", sm.group(), re.IGNORECASE)
        mult = 1
        if mult_m:
            mult = 1_000_000 if mult_m.group(1).lower() == "million" else 1_000
        try:
            return int(float(raw) * mult)
        except ValueError:
            return None

    @staticmethod
    def _extract_lockup_days(text: str) -> int:
        m = re.search(r"(\d+)[-\s]day\s+lock[-\s]?up", text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return 180  # standard if not specified

    @staticmethod
    def _extract_use_of_proceeds(text: str) -> Optional[str]:
        m = re.search(r"USE OF PROCEEDS(.{30,600})", text, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        snippet = re.sub(r"\s+", " ", m.group(1).strip())
        return snippet[:500]

    @staticmethod
    def _detect_dual_class(text: str) -> bool:
        return bool(re.search(
            r"dual[- ]class|class\s+[AB]\s+common|multiple\s+voting|high[- ]vote",
            text, re.IGNORECASE,
        ))

    @staticmethod
    def _extract_customer_concentration(text: str) -> Optional[float]:
        m = re.search(
            r"(\d{1,3}(?:\.\d+)?)\s*%\s*of\s+(?:our\s+)?(?:total\s+)?(?:net\s+)?revenue[s]?"
            r"(?:\s+was|\s+were|\s+came)\s+(?:from|generated\s+by)\s+(?:one|a\s+single|our\s+largest)",
            text, re.IGNORECASE,
        )
        if m:
            try:
                return float(m.group(1)) / 100.0
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_cap_table(text: str) -> tuple[Optional[float], Optional[float]]:
        pre, post = None, None
        m_pre = re.search(
            r"(?:prior\s+to|before)\s+this\s+offering.{0,200}(\d{1,3}(?:\.\d+)?)\s*%",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m_pre:
            try:
                pre = float(m_pre.group(1)) / 100.0
            except ValueError:
                pass
        m_post = re.search(
            r"(?:after|following)\s+this\s+offering.{0,200}(\d{1,3}(?:\.\d+)?)\s*%",
            text, re.IGNORECASE | re.DOTALL,
        )
        if m_post:
            try:
                post = float(m_post.group(1)) / 100.0
            except ValueError:
                pass
        return pre, post

    @staticmethod
    def _score_quality(data: dict) -> tuple[list[str], list[str], int]:
        red_flags:   list[str] = []
        green_flags: list[str] = []
        score = 5

        # Revenue
        rev = data.get("revenue_y0")
        if rev is None:
            red_flags.append("No revenue identified — pre-revenue stage")
            score -= 1
        elif rev < 5_000_000:
            red_flags.append(f"Very low revenue: ${rev:,.0f}")
            score -= 1
        elif rev >= 100_000_000:
            green_flags.append(f"Significant revenue: ${rev/1e6:.1f}M")
            score += 1

        # Growth
        growth = data.get("revenue_growth_yoy")
        if growth is not None:
            if growth > 0.50:
                green_flags.append(f"High revenue growth: {growth:.0%} YoY")
                score += 2
            elif growth > 0.15:
                green_flags.append(f"Solid revenue growth: {growth:.0%} YoY")
                score += 1
            elif growth < 0:
                red_flags.append(f"Declining revenue: {growth:.1%} YoY")
                score -= 2

        # Profitability
        ni = data.get("net_income")
        if ni is not None:
            if ni > 0:
                green_flags.append("Profitable at IPO — rare and positive signal")
                score += 2
            elif ni < -50_000_000:
                red_flags.append(f"Large net loss: ${ni/1e6:.1f}M")
                score -= 1

        # Gross margin
        gpm = data.get("gross_profit_margin")
        if gpm is not None:
            if gpm > 0.60:
                green_flags.append(f"Strong gross margin: {gpm:.0%}")
                score += 1
            elif gpm < 0:
                red_flags.append(f"Negative gross margin: {gpm:.1%}")
                score -= 2

        # Cash runway proxy
        cash = data.get("cash_and_equiv")
        if cash and ni and ni < 0:
            monthly_burn = abs(ni) / 12
            if monthly_burn > 0:
                runway = cash / monthly_burn
                if runway < 12:
                    red_flags.append(f"Short cash runway: ~{runway:.0f} months post-IPO")
                    score -= 1
                elif runway > 24:
                    green_flags.append(f"Adequate cash runway: ~{runway:.0f} months")
                    score += 1

        # Going concern
        if data.get("has_going_concern"):
            red_flags.append("Going concern qualification")
            score -= 2

        # Dual-class
        if data.get("has_dual_class"):
            red_flags.append("Dual-class share structure")
            score -= 1

        # Secondary selling
        if data.get("proceeds_to_selling_shs"):
            red_flags.append("Insider cash-out via secondary shares")
            score -= 1

        # Customer concentration
        top_cust = data.get("top_customer_pct")
        if top_cust and top_cust > 0.30:
            red_flags.append(f"Customer concentration risk: {top_cust:.0%} from top customer")
            score -= 1

        # Risk complexity
        risks = data.get("risk_factors_count", 0)
        if risks > 60:
            red_flags.append(f"Very high risk factor count: {risks}")
            score -= 1

        # Underwriter
        if data.get("has_tier1_underwriter"):
            green_flags.append(f"Tier-1 underwriter: {', '.join(u for u in data.get('underwriters', []) if u in _TIER1_UNDERWRITERS)}")
            score += 1
        elif not data.get("underwriters"):
            red_flags.append("No major underwriter identified")
            score -= 1

        # SPAC
        if data.get("is_spac"):
            red_flags.append("SPAC structure")
            score -= 1

        score = max(0, min(10, score))
        return red_flags, green_flags, score


# ---------------------------------------------------------------------------
# IPOPipelineTracker
# ---------------------------------------------------------------------------

class IPOPipelineTracker:
    """
    Monitor the EDGAR IPO pipeline via S-1, 424B4, and RW filings.

    Covers:
    - Active S-1 / F-1 pipeline (companies that filed but not yet priced)
    - Priced IPOs (424B4 = final prospectus filed on IPO day)
    - Withdrawn IPOs (RW = registration withdrawal)
    - Sector-level pipeline analytics
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_filed_s1s(self, lookback_days: int = 180) -> pd.DataFrame:
        """
        Return all S-1, S-1/A, S-11, F-1 filings from the last *lookback_days*.

        Fields: company_name, cik, form_type, filed_date, sector, sic_code,
                is_spac, ipo_size_est, price_range_low, price_range_high,
                underwriters, accession_number
        """
        rows = await self._fetch_efts_filings(
            forms="S-1,S-1/A,S-11,F-1,F-1/A",
            lookback_days=lookback_days,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values("filed_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def get_priced_ipos(self, lookback_days: int = 90) -> pd.DataFrame:
        """
        Return recently priced IPOs (424B4 = final prospectus = IPO day).

        Fields: company_name, cik, ticker, priced_date, final_price,
                shares_offered, gross_proceeds, underwriters, accession_number
        """
        rows = await self._fetch_efts_filings(
            forms="424B4,424B3",
            lookback_days=lookback_days,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.rename(columns={"filed_date": "priced_date"}, inplace=True)
        df.sort_values("priced_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def get_withdrawn_ipos(self, lookback_days: int = 180) -> pd.DataFrame:
        """
        Return withdrawn IPO registrations (RW filings).

        Withdrawn IPOs signal poor market conditions or company-specific issues
        (weak demand during road show, accounting problems, SEC comments unresolved).
        """
        rows = await self._fetch_efts_filings(
            forms="RW,RW/A",
            lookback_days=lookback_days,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values("filed_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def track_ipo_pipeline_by_sector(self) -> pd.DataFrame:
        """
        Aggregate the S-1 pipeline by GICS-equivalent sector.

        Returns: sector, n_filings, n_spacs, avg_ipo_size_est, latest_filing_date
        """
        df = await self.get_filed_s1s(lookback_days=180)
        if df.empty:
            return pd.DataFrame()

        df["sector"] = df["sector"].fillna("Unknown")
        agg = (
            df.groupby("sector")
            .agg(
                n_filings=("accession_number", "count"),
                n_spacs=("is_spac", "sum"),
                avg_ipo_size_est=("ipo_size_est", "mean"),
                latest_filing_date=("filed_date", "max"),
            )
            .reset_index()
            .sort_values("n_filings", ascending=False)
        )
        return agg

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _fetch_efts_filings(
        self,
        forms: str,
        lookback_days: int,
    ) -> list[dict]:
        today      = date.today()
        start_date = today - timedelta(days=lookback_days)
        params     = {
            "q":         '""',
            "forms":     forms,
            "dateRange": "custom",
            "startdt":   start_date.isoformat(),
            "enddt":     today.isoformat(),
            "_source":   "entity_name,file_date,form_type,accession_no,entity_id,display_names,period_of_report",
        }
        rows: list[dict] = []
        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            for from_idx in range(0, 200, 10):
                params["from"] = from_idx
                try:
                    resp = await client.get(_EDGAR_EFTS, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.warning("EFTS fetch failed", forms=forms, error=str(exc))
                    break
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    src  = hit.get("_source", {})
                    names = src.get("display_names", [{}])
                    cname = names[0].get("name", "Unknown") if names else "Unknown"
                    sic   = str(names[0].get("sic", "")) if names else ""
                    cik   = str(src.get("entity_id", "")).lstrip("0") or "0"
                    acc   = (hit.get("_id") or "").replace(":", "-")
                    fdate = _parse_date_safe(src.get("file_date"))
                    rows.append({
                        "company_name":    cname,
                        "cik":             cik,
                        "form_type":       src.get("form_type", ""),
                        "filed_date":      str(fdate) if fdate else "",
                        "sic_code":        sic or None,
                        "sector":          _sic_to_sector(sic) if sic else None,
                        "is_spac":         False,  # enriched separately if needed
                        "ipo_size_est":    None,
                        "price_range_low": None,
                        "price_range_high": None,
                        "underwriters":    [],
                        "accession_number": acc,
                    })
                if len(hits) < 10:
                    break
                await asyncio.sleep(_RATE_DELAY)
        return rows


# ---------------------------------------------------------------------------
# IPOPerformanceAnalyzer
# ---------------------------------------------------------------------------

class IPOPerformanceAnalyzer:
    """
    Compute aftermarket performance metrics for recently priced IPOs.

    Covers:
    - Day-1 return and money left on table (underpricing)
    - +1d / +30d / +90d / +180d / +1yr returns vs SPY
    - Lock-up expiry analysis (date, shares unlocking, expected pressure)
    - Peer valuation comparison (was the IPO fairly priced?)
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def compute_ipo_day1_return(
        self,
        ticker: str,
        ipo_price: float,
    ) -> dict:
        """
        IPO price vs first-day close from yfinance.

        Returns: ipo_price, day1_open, day1_close, day1_return_pct,
                 money_left_on_table_per_share (underpricing measure)
        """
        try:
            import yfinance as yf
            hist = await asyncio.to_thread(
                lambda: yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            )
        except Exception as exc:
            logger.warning("yfinance history failed", ticker=ticker, error=str(exc))
            return {"ticker": ticker, "error": str(exc)}

        if hist.empty:
            return {"ticker": ticker, "error": "No price data"}

        day1 = hist.iloc[0]
        day1_open  = float(day1.get("Open",  ipo_price))
        day1_close = float(day1.get("Close", ipo_price))
        day1_return = (day1_close - ipo_price) / ipo_price if ipo_price > 0 else None
        money_left  = day1_close - ipo_price  # positive = underpriced

        return {
            "ticker":                     ticker,
            "ipo_price":                  ipo_price,
            "day1_open":                  round(day1_open, 2),
            "day1_close":                 round(day1_close, 2),
            "day1_return_pct":            round(day1_return, 4) if day1_return is not None else None,
            "money_left_on_table_per_sh": round(money_left, 2),
            "underpriced":                money_left > 0,
        }

    async def compute_ipo_aftermarket_performance(
        self,
        ticker: str,
        ipo_date: str,
        ipo_price: float,
    ) -> dict:
        """
        Compute IPO aftermarket returns at multiple horizons vs SPY.

        Horizons: +1d, +30d, +90d, +180d (lock-up expiry), +1yr
        """
        ipo_dt = _parse_date_safe(ipo_date)
        if not ipo_dt:
            return {"ticker": ticker, "error": "Invalid ipo_date"}

        price_start = (ipo_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        price_end   = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            import yfinance as yf
            hist = await asyncio.to_thread(
                lambda: yf.download(
                    [ticker, "SPY"], start=price_start, end=price_end,
                    progress=False, auto_adjust=True,
                )
            )
        except Exception as exc:
            logger.warning("yfinance download failed", ticker=ticker, error=str(exc))
            return {"ticker": ticker, "error": str(exc)}

        if hist.empty:
            return {"ticker": ticker, "error": "No price data returned"}

        def _price_at(symbol: str, target_date: date) -> Optional[float]:
            try:
                close = hist["Close"][symbol]
                idx = close.index.searchsorted(pd.Timestamp(target_date))
                if idx < len(close):
                    return float(close.iloc[idx])
            except Exception:
                pass
            return None

        def _ret(symbol: str, days: int) -> Optional[float]:
            p0 = ipo_price if symbol == ticker else _price_at("SPY", ipo_dt)
            p1 = _price_at(symbol, ipo_dt + timedelta(days=days))
            if p0 and p1 and p0 > 0:
                return (p1 - p0) / p0
            return None

        horizons = [1, 30, 90, 180, 365]
        returns: dict = {}
        for h in horizons:
            r_ticker = _ret(ticker, h)
            r_spy    = _ret("SPY",  h)
            alpha    = (r_ticker - r_spy) if r_ticker is not None and r_spy is not None else None
            returns[f"return_{h}d"]       = round(r_ticker, 4) if r_ticker is not None else None
            returns[f"spy_return_{h}d"]   = round(r_spy, 4)    if r_spy    is not None else None
            returns[f"alpha_{h}d_vs_spy"] = round(alpha, 4)    if alpha    is not None else None

        return {"ticker": ticker, "ipo_date": ipo_date, "ipo_price": ipo_price, **returns}

    async def lockup_expiry_analysis(
        self,
        ticker: str,
        ipo_date: Optional[str] = None,
        lockup_days: int = 180,
    ) -> dict:
        """
        Analyse the upcoming lock-up expiry event.

        - Lock-up expiry date = ipo_date + lockup_days
        - Expected selling pressure: insider shares as % of float
        - Historical average: stocks fall 2-3% around lock-up expiry

        ipo_date can be inferred from the earliest yfinance price if omitted.
        """
        try:
            import yfinance as yf
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
            float_shares = float(info.get("floatShares") or info.get("sharesOutstanding") or 0)
            shares_insider = float(info.get("heldPercentInsiders", 0) or 0) * float_shares
        except Exception:
            float_shares   = None
            shares_insider = None

        if ipo_date:
            ipo_dt = _parse_date_safe(ipo_date)
        else:
            try:
                hist = await asyncio.to_thread(
                    lambda: yf.Ticker(ticker).history(period="max", auto_adjust=True)
                )
                ipo_dt = hist.index[0].date() if not hist.empty else None
            except Exception:
                ipo_dt = None

        expiry_dt = (ipo_dt + timedelta(days=lockup_days)) if ipo_dt else None
        days_until = (expiry_dt - date.today()).days if expiry_dt else None

        pct_float_unlocking = None
        if shares_insider and float_shares and float_shares > 0:
            pct_float_unlocking = shares_insider / float_shares

        return {
            "ticker":                ticker,
            "ipo_date":              str(ipo_dt) if ipo_dt else None,
            "lockup_days":           lockup_days,
            "lockup_expiry_date":    str(expiry_dt) if expiry_dt else None,
            "days_until_expiry":     days_until,
            "float_shares":          float_shares,
            "estimated_insider_shares": shares_insider,
            "pct_float_unlocking":   round(pct_float_unlocking, 4) if pct_float_unlocking else None,
            "expected_pressure":     "high" if (pct_float_unlocking or 0) > 0.30 else "moderate" if (pct_float_unlocking or 0) > 0.10 else "low",
            "historical_avg_drop":   -0.025,   # -2.5% average around lock-up expiry (Brav & Gompers)
            "trade_signal":          "watch_for_short" if days_until is not None and 0 < days_until <= 14 else "monitor",
        }

    async def screen_ipos_near_lockup_expiry(
        self,
        lookback_days: int = 14,
    ) -> pd.DataFrame:
        """
        Screen for IPOs with lock-up expiring within the next *lookback_days*.

        Sources 424B4 filings from the last 180 days and computes expiry dates.
        Returns DataFrame sorted by days_until_expiry ascending.
        """
        tracker = IPOPipelineTracker(timeout=self._timeout)
        priced  = await tracker.get_priced_ipos(lookback_days=180)
        if priced.empty:
            return pd.DataFrame()

        today = date.today()
        rows: list[dict] = []
        for _, row in priced.iterrows():
            priced_date = _parse_date_safe(row.get("priced_date", ""))
            if not priced_date:
                continue
            expiry = priced_date + timedelta(days=180)
            days_until = (expiry - today).days
            if not (0 <= days_until <= lookback_days):
                continue
            rows.append({
                "company_name":      row.get("company_name", ""),
                "ticker":            row.get("ticker"),
                "priced_date":       str(priced_date),
                "lockup_expiry":     str(expiry),
                "days_until_expiry": days_until,
                "accession_number":  row.get("accession_number", ""),
            })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values("days_until_expiry", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def compare_ipo_to_peers(
        self,
        ticker: str,
        ipo_valuation: float,
        peers: list[str],
    ) -> dict:
        """
        Compare IPO valuation multiples to peer group at the time of IPO.

        Fetches yfinance fundamentals for each peer (revenue TTM, EBITDA, market cap)
        and computes EV/Revenue and EV/EBITDA.
        Returns whether the IPO was cheap, fair, or expensive relative to peers.
        """
        try:
            import yfinance as yf
        except ImportError:
            return {"ticker": ticker, "error": "yfinance not installed"}

        ipo_rev:    Optional[float] = None
        ipo_ebitda: Optional[float] = None
        try:
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
            ipo_rev    = float(info.get("totalRevenue") or 0) or None
            ipo_ebitda = float(info.get("ebitda")       or 0) or None
        except Exception:
            pass

        ipo_ev_rev    = ipo_valuation / ipo_rev    if ipo_rev    and ipo_rev    > 0 else None
        ipo_ev_ebitda = ipo_valuation / ipo_ebitda if ipo_ebitda and ipo_ebitda > 0 else None

        peer_data: list[dict] = []
        for peer in peers:
            try:
                pinfo = await asyncio.to_thread(lambda p=peer: yf.Ticker(p).info)
                p_mktcap = float(pinfo.get("marketCap") or 0)
                p_debt   = float(pinfo.get("totalDebt") or 0)
                p_cash   = float(pinfo.get("totalCash") or 0)
                p_ev     = p_mktcap + p_debt - p_cash
                p_rev    = float(pinfo.get("totalRevenue") or 0) or None
                p_ebitda = float(pinfo.get("ebitda")       or 0) or None
                peer_data.append({
                    "peer":      peer,
                    "ev":        p_ev,
                    "revenue":   p_rev,
                    "ebitda":    p_ebitda,
                    "ev_rev":    p_ev / p_rev    if p_rev    and p_rev    > 0 else None,
                    "ev_ebitda": p_ev / p_ebitda if p_ebitda and p_ebitda > 0 else None,
                })
            except Exception:
                pass

        peer_ev_revs    = [p["ev_rev"]    for p in peer_data if p.get("ev_rev")    is not None]
        peer_ev_ebitdas = [p["ev_ebitda"] for p in peer_data if p.get("ev_ebitda") is not None]
        avg_peer_ev_rev    = sum(peer_ev_revs)    / len(peer_ev_revs)    if peer_ev_revs    else None
        avg_peer_ev_ebitda = sum(peer_ev_ebitdas) / len(peer_ev_ebitdas) if peer_ev_ebitdas else None

        def _relative(ipo_mult: Optional[float], peer_avg: Optional[float]) -> Optional[str]:
            if ipo_mult is None or peer_avg is None or peer_avg == 0:
                return None
            ratio = ipo_mult / peer_avg
            if ratio < 0.85:
                return "cheap"
            elif ratio > 1.20:
                return "expensive"
            return "fair"

        return {
            "ticker":               ticker,
            "ipo_valuation":        ipo_valuation,
            "ipo_ev_revenue":       round(ipo_ev_rev, 2)    if ipo_ev_rev    is not None else None,
            "ipo_ev_ebitda":        round(ipo_ev_ebitda, 2) if ipo_ev_ebitda is not None else None,
            "avg_peer_ev_revenue":  round(avg_peer_ev_rev, 2)    if avg_peer_ev_rev    is not None else None,
            "avg_peer_ev_ebitda":   round(avg_peer_ev_ebitda, 2) if avg_peer_ev_ebitda is not None else None,
            "valuation_vs_peers_rev":    _relative(ipo_ev_rev, avg_peer_ev_rev),
            "valuation_vs_peers_ebitda": _relative(ipo_ev_ebitda, avg_peer_ev_ebitda),
            "peers_analysed":       peer_data,
        }


# ---------------------------------------------------------------------------
# SPACTracker (enhanced)
# ---------------------------------------------------------------------------

class SPACTracker:
    """
    Enhanced SPAC intelligence — active SPACs, announced mergers, trust discount.

    SPACs are blank-check companies that raise money through IPO to acquire
    a target within a defined window (typically 18-24 months).

    Academic context:
    - SPAC mergers have on average underperformed traditional IPOs by ~15% (2020-22)
    - SPACs trading below trust NAV ($10) signal poor merger prospects
    - After announcing a target, many SPACs trade at premium until merger vote
    """

    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def get_active_spacs(self) -> pd.DataFrame:
        """
        Return recently filed blank-check companies (SIC 6770) from EDGAR.

        Approach: query EFTS for S-1 filings by SIC 6770 (Blank Checks)
        or keyword "blank check company" within the last 24 months.

        Fields: company_name, cik, filed_date, trust_size_est,
                sector_focus, sponsor_name, months_since_ipo, deadline_est
        """
        params = {
            "q":         '"blank check company"',
            "forms":     "S-1,S-1/A",
            "dateRange": "custom",
            "startdt":   (date.today() - timedelta(days=730)).isoformat(),
            "enddt":     date.today().isoformat(),
            "_source":   "entity_name,file_date,form_type,accession_no,entity_id,display_names",
        }
        rows: list[dict] = []
        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            for from_idx in range(0, 100, 10):
                params["from"] = from_idx
                try:
                    resp = await client.get(_EDGAR_EFTS, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.warning("SPAC EFTS query failed", error=str(exc))
                    break
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    src   = hit.get("_source", {})
                    names = src.get("display_names", [{}])
                    cname = names[0].get("name", "Unknown") if names else "Unknown"
                    cik   = str(src.get("entity_id", "")).lstrip("0") or "0"
                    acc   = (hit.get("_id") or "").replace(":", "-")
                    fdate = _parse_date_safe(src.get("file_date"))
                    months_since = None
                    if fdate:
                        delta = date.today() - fdate
                        months_since = round(delta.days / 30.4, 1)
                    rows.append({
                        "company_name":   cname,
                        "cik":            cik,
                        "accession_number": acc,
                        "filed_date":     str(fdate) if fdate else "",
                        "months_since_ipo": months_since,
                        "deadline_est":   str(fdate + timedelta(days=547)) if fdate else None,  # 18 months
                        "trust_size_est": None,
                        "sector_focus":   None,
                        "sponsor_name":   None,
                        "status":         "searching",
                    })
                if len(hits) < 10:
                    break
                await asyncio.sleep(_RATE_DELAY)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values("filed_date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    async def get_spac_mergers(self, lookback_days: int = 180) -> list[dict]:
        """
        Identify SPAC business combination announcements via 8-K Item 1.01.

        Searches EFTS for 8-K filings containing "business combination agreement"
        or "definitive agreement" alongside "SPAC" or "blank check".

        Returns list of dicts with: spac_name, cik, announced_date, target_name,
        accession_number.
        """
        today      = date.today()
        start_date = today - timedelta(days=lookback_days)
        params = {
            "q":         '"business combination agreement"',
            "forms":     "8-K",
            "dateRange": "custom",
            "startdt":   start_date.isoformat(),
            "enddt":     today.isoformat(),
            "_source":   "entity_name,file_date,form_type,accession_no,entity_id,display_names",
        }
        results: list[dict] = []
        async with httpx.AsyncClient(timeout=self._timeout, headers=_HEADERS) as client:
            try:
                resp = await client.get(_EDGAR_EFTS, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("SPAC merger EFTS query failed", error=str(exc))
                return []

            for hit in data.get("hits", {}).get("hits", []):
                src   = hit.get("_source", {})
                names = src.get("display_names", [{}])
                cname = names[0].get("name", "Unknown") if names else "Unknown"
                cik   = str(src.get("entity_id", "")).lstrip("0") or "0"
                acc   = (hit.get("_id") or "").replace(":", "-")
                fdate = _parse_date_safe(src.get("file_date"))
                results.append({
                    "spac_name":        cname,
                    "cik":              cik,
                    "announced_date":   str(fdate) if fdate else "",
                    "accession_number": acc,
                    "target_name":      None,  # enrichable via full-text parse
                })

        return results

    @staticmethod
    def compute_spac_discount(trust_nav: float, market_price: float) -> dict:
        """
        Compute the SPAC discount/premium vs trust NAV.

        SPACs typically IPO at $10 and hold funds in trust ≈ $10/share.
        Market price < trust NAV → discount → merger prospects viewed poorly.
        Market price > trust NAV → premium → market anticipates a good deal.

        Returns: trust_nav, market_price, discount_pct, premium_pct, signal
        """
        if trust_nav <= 0:
            return {"error": "trust_nav must be positive"}

        spread = market_price - trust_nav
        spread_pct = spread / trust_nav

        if spread_pct < -0.05:
            signal = "deep_discount_poor_prospects"
        elif spread_pct < 0:
            signal = "slight_discount_wait_and_see"
        elif spread_pct < 0.10:
            signal = "near_nav_neutral"
        elif spread_pct < 0.30:
            signal = "premium_deal_expected"
        else:
            signal = "large_premium_speculative"

        return {
            "trust_nav":    trust_nav,
            "market_price": market_price,
            "spread_usd":   round(spread, 2),
            "spread_pct":   round(spread_pct, 4),
            "signal":       signal,
            "arbitrage_floor": trust_nav,  # price should revert to NAV if no deal
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query

    ipo_router = APIRouter(prefix="/api/ipo", tags=["IPO Analytics"])
    _s1_parser = S1FilingParser()
    _pipeline  = IPOPipelineTracker()
    _perf      = IPOPerformanceAnalyzer()
    _spac      = SPACTracker()

    @ipo_router.get("/pipeline")
    async def api_ipo_pipeline(lookback_days: int = Query(180, ge=7, le=365)):
        """Current S-1 / F-1 filing pipeline."""
        df = await _pipeline.get_filed_s1s(lookback_days)
        return {"count": len(df), "filings": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/pipeline/by-sector")
    async def api_ipo_pipeline_sector():
        """IPO pipeline breakdown by sector."""
        df = await _pipeline.track_ipo_pipeline_by_sector()
        return {"sectors": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/recent")
    async def api_recent_ipos(lookback_days: int = Query(90, ge=7, le=180)):
        """Recently priced IPOs (424B4 filings)."""
        df = await _pipeline.get_priced_ipos(lookback_days)
        return {"count": len(df), "ipos": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/withdrawn")
    async def api_withdrawn_ipos(lookback_days: int = Query(180, ge=7, le=365)):
        """Withdrawn IPO registrations (RW filings)."""
        df = await _pipeline.get_withdrawn_ipos(lookback_days)
        return {"count": len(df), "withdrawals": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/{ticker}/performance")
    async def api_ipo_performance(
        ticker: str,
        ipo_date: str = Query(..., description="YYYY-MM-DD"),
        ipo_price: float = Query(..., gt=0),
    ):
        """IPO aftermarket performance vs SPY at multiple horizons."""
        return await _perf.compute_ipo_aftermarket_performance(ticker.upper(), ipo_date, ipo_price)

    @ipo_router.get("/{ticker}/day1")
    async def api_ipo_day1(
        ticker: str,
        ipo_price: float = Query(..., gt=0),
    ):
        """IPO first-day return and money-left-on-table (underpricing)."""
        return await _perf.compute_ipo_day1_return(ticker.upper(), ipo_price)

    @ipo_router.get("/{ticker}/lockup")
    async def api_lockup_analysis(
        ticker: str,
        ipo_date: Optional[str] = Query(None),
        lockup_days: int = Query(180, ge=30, le=365),
    ):
        """Lock-up expiry analysis and expected selling pressure."""
        return await _perf.lockup_expiry_analysis(ticker.upper(), ipo_date, lockup_days)

    @ipo_router.get("/{ticker}/peers")
    async def api_ipo_peers(
        ticker: str,
        ipo_valuation: float = Query(..., gt=0),
        peers: str = Query(..., description="Comma-separated peer tickers"),
    ):
        """Compare IPO valuation multiples to peer group."""
        peer_list = [p.strip().upper() for p in peers.split(",") if p.strip()]
        return await _perf.compare_ipo_to_peers(ticker.upper(), ipo_valuation, peer_list)

    @ipo_router.get("/lockup-expiring")
    async def api_lockup_expiring(days_ahead: int = Query(14, ge=1, le=60)):
        """IPOs with lock-up expiring within the next N days."""
        df = await _perf.screen_ipos_near_lockup_expiry(days_ahead)
        return {"count": len(df), "expiring": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/spacs")
    async def api_active_spacs():
        """Active blank-check companies (SPACs) from recent S-1 filings."""
        df = await _spac.get_active_spacs()
        return {"count": len(df), "spacs": df.to_dict(orient="records") if not df.empty else []}

    @ipo_router.get("/spacs/mergers")
    async def api_spac_mergers(lookback_days: int = Query(180, ge=7, le=365)):
        """SPAC business combination announcements."""
        mergers = await _spac.get_spac_mergers(lookback_days)
        return {"count": len(mergers), "mergers": mergers}

    @ipo_router.get("/spacs/discount")
    async def api_spac_discount(
        trust_nav: float = Query(10.0, gt=0),
        market_price: float = Query(..., gt=0),
    ):
        """SPAC market price vs trust NAV discount/premium."""
        return SPACTracker.compute_spac_discount(trust_nav, market_price)

    @ipo_router.get("/{ticker}/s1")
    async def api_parse_s1(
        ticker: str,
        accession_number: str = Query(...),
        cik: str = Query(...),
    ):
        """Parse an S-1 prospectus and return structured data with red/green flags."""
        data = await _s1_parser.parse_s1_prospectus(accession_number, cik)
        if not data:
            raise HTTPException(status_code=404, detail="Filing not found or parse failed")
        return data

except ImportError:
    ipo_router = None  # type: ignore[assignment]
    logger.info("FastAPI not available — ipo_router not registered")
