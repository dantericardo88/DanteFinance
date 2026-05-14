"""esg_parser.py — ESG proxy signals from free EDGAR filings.

Derives E, S, G scores from:
  E: Environmental keywords in 10-K Item 1A Risk Factors
  S: CEO pay ratio + board gender diversity from DEF 14A
  G: Governance policy mentions from DEF 14A + 10-K

Proxy signals only — NOT equivalent to MSCI/Sustainalytics ratings.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PayRatioData(BaseModel):
    ceo_pay: Optional[float] = None
    median_employee_pay: Optional[float] = None
    pay_ratio: Optional[float] = None
    year: Optional[int] = None
    source: str = "DEF14A"


class BoardComposition(BaseModel):
    total_directors: Optional[int] = None
    independent_directors: Optional[int] = None
    female_directors: Optional[int] = None
    underrepresented_minorities: Optional[int] = None
    average_tenure_years: Optional[float] = None
    independence_pct: Optional[float] = None
    gender_diversity_pct: Optional[float] = None


class EnvironmentalSignals(BaseModel):
    climate_mention_count: int = 0
    carbon_mention_count: int = 0
    net_zero_mentioned: bool = False
    scope1_disclosed: bool = False
    scope2_disclosed: bool = False
    tcfd_mentioned: bool = False
    sustainability_report_mentioned: bool = False
    environmental_fine_mentioned: bool = False
    risk_score: int = 0  # 0-10


class GovernanceSignals(BaseModel):
    whistleblower_policy: bool = False
    code_of_ethics_filed: bool = False
    insider_trading_policy: bool = False
    related_party_transactions: bool = False
    accounting_restatement_mentioned: bool = False
    governance_score: int = 0  # 0-10


class ESGProfile(BaseModel):
    ticker: str
    cik: Optional[str] = None
    fiscal_year: Optional[int] = None
    environmental_score: float
    social_score: float
    governance_score: float
    composite_score: float
    pay_ratio: PayRatioData = Field(default_factory=PayRatioData)
    board: BoardComposition = Field(default_factory=BoardComposition)
    environmental: EnvironmentalSignals = Field(default_factory=EnvironmentalSignals)
    governance: GovernanceSignals = Field(default_factory=GovernanceSignals)
    data_completeness: float = 0.0
    disclaimer: str = (
        "Proxy signals from public EDGAR filings. "
        "Not equivalent to MSCI/Sustainalytics ratings."
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

async def _resolve_cik(ticker: str) -> Optional[str]:
    headers = {**_HEADERS, "Host": "www.sec.gov"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(EDGAR_TICKERS_URL, headers=headers)
            resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.error("CIK resolution error", ticker=ticker, error=str(exc))
    return None


async def _fetch_submissions(cik: str) -> dict:
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers={**_HEADERS, "Host": "data.sec.gov"})
        resp.raise_for_status()
        return resp.json()


def _latest_filing(submissions: dict, form_type: str) -> Optional[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == form_type:
            return {
                "accession": recent.get("accessionNumber", [])[i] if i < len(recent.get("accessionNumber", [])) else None,
                "primary_doc": recent.get("primaryDocument", [])[i] if i < len(recent.get("primaryDocument", [])) else None,
                "period": recent.get("reportDate", [])[i] if i < len(recent.get("reportDate", [])) else None,
            }
    return None


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")]:
        text = text.replace(ent, rep)
    return re.sub(r"\s{3,}", "  ", text).strip()


async def _get_filing_text(cik: str, accession: str, doc: str, limit: int) -> str:
    acc_clean = accession.replace("-", "")
    url = f"{EDGAR_ARCHIVES}/{cik.lstrip('0') or '0'}/{acc_clean}/{doc}"
    await asyncio.sleep(0.5)
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()
            raw = resp.text
        if "<" in raw[:500]:
            raw = _strip_html(raw)
        return re.sub(r"\s+", " ", raw)[:limit]
    except Exception as exc:
        logger.error("Filing fetch failed", url=url, error=str(exc))
        return ""


def _extract_section(text: str, starts: list[str], ends: list[str]) -> str:
    start_pos = next(
        (m.start() for pat in starts for m in [re.search(pat, text, re.IGNORECASE)] if m),
        None,
    )
    if start_pos is None:
        return ""
    end_pos = next(
        (start_pos + 100 + m.start()
         for pat in ends for m in [re.search(pat, text[start_pos + 100:], re.IGNORECASE)] if m),
        len(text),
    )
    return text[start_pos:end_pos]


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

async def fetch_def14a_text(cik: str) -> str:
    """Fetch most recent DEF 14A proxy statement text (up to 20,000 chars)."""
    cik = cik.zfill(10)
    try:
        sub = await _fetch_submissions(cik)
    except Exception as exc:
        logger.error("Submissions fetch failed", cik=cik, error=str(exc))
        return ""
    filing = _latest_filing(sub, "DEF 14A")
    if not filing or not filing.get("accession") or not filing.get("primary_doc"):
        logger.warning("No DEF 14A found", cik=cik)
        return ""
    return await _get_filing_text(cik, filing["accession"], filing["primary_doc"], 20_000)


async def fetch_10k_text(cik: str) -> str:
    """Fetch 10-K Item 1A Risk Factors section text (up to 15,000 chars)."""
    cik = cik.zfill(10)
    try:
        sub = await _fetch_submissions(cik)
    except Exception as exc:
        logger.error("Submissions fetch failed", cik=cik, error=str(exc))
        return ""
    filing = _latest_filing(sub, "10-K")
    if not filing or not filing.get("accession") or not filing.get("primary_doc"):
        logger.warning("No 10-K found", cik=cik)
        return ""
    raw = await _get_filing_text(cik, filing["accession"], filing["primary_doc"], 80_000)
    section = _extract_section(
        raw,
        starts=[r"ITEM\s+1A[\.\s]+RISK\s+FACTORS", r"RISK\s+FACTORS"],
        ends=[r"ITEM\s+1B[\.\s]", r"ITEM\s+2[\.\s]", r"UNRESOLVED\s+STAFF\s+COMMENTS"],
    )
    return (section or raw)[:15_000]


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _re_first(patterns: list[str], text: str, group: int = 1) -> Optional[str]:
    """Return the first regex match group across a list of patterns."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return m.group(group)
            except IndexError:
                continue
    return None


def _parse_money(raw: str, unit_word: str) -> float:
    num = float(raw.replace(",", ""))
    uw = unit_word.lower()
    if uw in ("million", "m"):
        return num * 1_000_000
    if uw in ("billion", "b"):
        return num * 1_000_000_000
    if uw == "thousand":
        return num * 1_000
    return num if num > 100_000 else num * 1_000  # bare table number heuristic


def extract_pay_ratio(proxy_text: str) -> PayRatioData:
    """Regex extraction of CEO pay ratio from DEF 14A."""
    ratio_val = _re_first([
        r"(?:ceo[\s\-]to[\s\-]median|pay\s+ratio)[^\d]*?([\d,]+)\s*(?::|to|-to-)\s*1",
        r"ratio\s+of\s+([\d,]+)\s*(?::|to)\s*1",
        r"approximately\s+([\d,]+)\s*(?::|to|-to-)\s*1",
        r"([\d,]+)-to-1\s+ratio",
        r"([\d,]+)\s*(?::|to|-to-)\s*1\s+(?:ceo[\s\-]to[\s\-]median|pay\s+ratio)",
    ], proxy_text)
    pay_ratio = float(ratio_val.replace(",", "")) if ratio_val else None

    # CEO pay
    ceo_pay: Optional[float] = None
    m = re.search(
        r"(?:chief\s+executive\s+officer|ceo)[^$\n]{0,200}\$([\d,.]+)\s*(million|billion|thousand)?",
        proxy_text, re.IGNORECASE
    )
    if m:
        try:
            ceo_pay = _parse_money(m.group(1), m.group(2) or "")
        except ValueError:
            pass

    # Median employee pay
    median_pay: Optional[float] = None
    m2 = re.search(
        r"median\s+(?:annual\s+)?(?:total\s+)?(?:employee\s+)?compensation[^$\n]{0,100}"
        r"\$([\d,.]+)\s*(million|billion|thousand)?",
        proxy_text, re.IGNORECASE
    )
    if m2:
        try:
            median_pay = _parse_money(m2.group(1), m2.group(2) or "")
        except ValueError:
            pass

    if pay_ratio is None and ceo_pay and median_pay and median_pay > 0:
        pay_ratio = round(ceo_pay / median_pay, 1)

    yr = _re_first([r"\b(20\d{2})\b"], proxy_text[:2000])
    return PayRatioData(
        ceo_pay=round(ceo_pay, 2) if ceo_pay else None,
        median_employee_pay=round(median_pay, 2) if median_pay else None,
        pay_ratio=pay_ratio,
        year=int(yr) if yr else None,
    )


def extract_board_composition(proxy_text: str) -> BoardComposition:
    """Extract board diversity data from DEF 14A."""
    total_s = _re_first([
        r"board[^.]{0,40}(?:consists?\s+of|comprises?|composed\s+of)\s+(\d+)\s+(?:members|directors)",
        r"(\d+)\s+(?:current\s+)?(?:members|directors)\s+(?:of\s+)?(?:the\s+)?board",
    ], proxy_text)
    total = int(total_s) if total_s else None

    indep_s = _re_first([
        r"(\d+)\s+(?:of\s+(?:our\s+)?(?:\d+\s+)?)?directors?\s+(?:are\s+)?(?:qualified\s+as\s+)?independent",
        r"(\d+)\s+independent\s+directors?",
    ], proxy_text)
    independent = int(indep_s) if indep_s else None

    indep_pct_s = _re_first([r"([\d.]+)%\s+(?:of\s+(?:our\s+)?directors?\s+(?:are\s+)?)?independent"], proxy_text)
    indep_pct = float(indep_pct_s) if indep_pct_s else (
        round(100.0 * independent / total, 1) if independent and total else None
    )

    female_s = _re_first([
        r"(\d+)\s+(?:of\s+(?:our\s+)?(?:\d+\s+)?)?directors?\s+(?:are\s+)?women",
        r"(\d+)\s+(?:female|women)\s+directors?",
    ], proxy_text)
    female = int(female_s) if female_s else None

    female_pct_s = _re_first([r"([\d.]+)%\s+(?:of\s+(?:our\s+)?directors?\s+(?:are\s+)?)?(?:women|female)"], proxy_text)
    female_pct = float(female_pct_s) if female_pct_s else (
        round(100.0 * female / total, 1) if female and total else None
    )

    urm_s = _re_first([
        r"(\d+)\s+(?:directors?\s+)?(?:identify\s+as\s+)?(?:racially|ethnically)\s+diverse",
        r"(\d+)\s+(?:directors?\s+are\s+)?(?:members?\s+of\s+)?underrepresented\s+(?:minority|minorities)",
    ], proxy_text)
    urm = int(urm_s) if urm_s else None

    tenure_s = _re_first([
        r"average\s+(?:director\s+)?tenure\s+of\s+([\d.]+)\s+years?",
        r"([\d.]+)\s+years?\s+(?:of\s+)?average\s+(?:director\s+)?tenure",
    ], proxy_text)
    tenure = float(tenure_s) if tenure_s else None

    return BoardComposition(
        total_directors=total,
        independent_directors=independent,
        female_directors=female,
        underrepresented_minorities=urm,
        average_tenure_years=tenure,
        independence_pct=indep_pct,
        gender_diversity_pct=female_pct,
    )


def extract_environmental_signals(text_10k: str) -> EnvironmentalSignals:
    """Count environmental keyword mentions in 10-K text."""
    tl = text_10k.lower()
    climate_count = sum(tl.count(w) for w in ["climate", "global warming", "greenhouse", "weather", "flood", "drought"])
    carbon_count = sum(tl.count(w) for w in ["carbon", "co2", "ghg", "emission", "methane"])

    net_zero = bool(re.search(r"net[\s\-]zero|carbon[\s\-]neutral", tl))
    scope1 = bool(re.search(r"scope\s*1\b|scope\s+one\b", tl))
    scope2 = bool(re.search(r"scope\s*2\b|scope\s+two\b", tl))
    tcfd = bool(re.search(r"\btcfd\b|task\s+force\s+on\s+climate", tl))
    sustain = bool(re.search(r"sustainability\s+report|esg\s+report|corporate\s+responsibility\s+report", tl))
    env_fine = bool(re.search(r"epa\s+(?:fine|penalty|violation)|environmental\s+(?:fine|penalty|violation)", tl))

    disclosure_bonus = sum([scope1, scope2, tcfd, sustain, net_zero])
    raw_risk = min(10, (climate_count + carbon_count) // 3)
    risk_score = max(0, min(10, raw_risk - disclosure_bonus + (2 if env_fine else 0)))

    return EnvironmentalSignals(
        climate_mention_count=climate_count,
        carbon_mention_count=carbon_count,
        net_zero_mentioned=net_zero,
        scope1_disclosed=scope1,
        scope2_disclosed=scope2,
        tcfd_mentioned=tcfd,
        sustainability_report_mentioned=sustain,
        environmental_fine_mentioned=env_fine,
        risk_score=risk_score,
    )


def extract_governance_signals(proxy_text: str, text_10k: str) -> GovernanceSignals:
    """Look for governance policy mentions across proxy and 10-K text."""
    combined = (proxy_text + " " + text_10k).lower()
    whistleblower = bool(re.search(r"whistleblower|speak\s*up\s*(?:hotline|policy|program)", combined))
    code_ethics = bool(re.search(r"code\s+of\s+(?:business\s+)?(?:ethics|conduct)|ethics\s+code", combined))
    insider_policy = bool(re.search(r"insider\s+trading\s+policy|securities\s+trading\s+policy|blackout\s+period", combined))
    related_party = bool(re.search(r"related[\s\-]party\s+transaction|related\s+person\s+transaction", combined))
    restatement = bool(re.search(r"restate(?:ment|d)|material\s+weakness|accounting\s+error|error\s+correction", combined))

    score = (2 if whistleblower else 0) + (2 if code_ethics else 0) + (2 if insider_policy else 0) + \
            (1 if related_party else 0) + (3 if not restatement else 0)

    return GovernanceSignals(
        whistleblower_policy=whistleblower,
        code_of_ethics_filed=code_ethics,
        insider_trading_policy=insider_policy,
        related_party_transactions=related_party,
        accounting_restatement_mentioned=restatement,
        governance_score=min(10, score),
    )


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_esg_scores(
    pay_ratio: PayRatioData,
    board: BoardComposition,
    env: EnvironmentalSignals,
    gov: GovernanceSignals,
) -> tuple[float, float, float]:
    """Compute E, S, G scores 0-10. Returns (e, s, g)."""
    # Environmental: base 5 + disclosure bonus - fine penalty +/- risk adjustment
    disclosure_bonus = sum([env.scope1_disclosed, env.scope2_disclosed, env.tcfd_mentioned,
                            env.sustainability_report_mentioned, env.net_zero_mentioned]) * 0.8
    e_score = max(0.0, min(10.0,
        5.0 + disclosure_bonus - (2.0 if env.environmental_fine_mentioned else 0.0)
        - (env.risk_score - 5) * 0.2
    ))

    # Social: baseline 3 + pay ratio (0-3) + gender diversity (0-2) + URM (0-1)
    s_score = 3.0
    if pay_ratio.pay_ratio is not None:
        s_score += 3.0 if pay_ratio.pay_ratio < 100 else (2.0 if pay_ratio.pay_ratio < 300 else 0.5)

    gd_pct = board.gender_diversity_pct
    if gd_pct is None and board.female_directors and board.total_directors:
        gd_pct = 100.0 * board.female_directors / board.total_directors
    if gd_pct is not None:
        s_score += 2.0 if gd_pct >= 30 else (1.0 if gd_pct >= 20 else 0.0)

    if board.underrepresented_minorities and board.underrepresented_minorities >= 2:
        s_score += 1.0
    s_score = max(0.0, min(10.0, s_score))

    # Governance: gov.governance_score (0-10) + independent board bonus
    g_score = float(gov.governance_score)
    indep_pct = board.independence_pct
    if indep_pct is None and board.independent_directors and board.total_directors:
        indep_pct = 100.0 * board.independent_directors / board.total_directors
    if indep_pct is not None:
        g_score = min(10.0, g_score + (2.0 if indep_pct >= 75 else (1.0 if indep_pct >= 50 else 0.0)))
    g_score = max(0.0, min(10.0, g_score))

    return round(e_score, 1), round(s_score, 1), round(g_score, 1)


def _data_completeness(pay_ratio: PayRatioData, board: BoardComposition,
                        env: EnvironmentalSignals, gov: GovernanceSignals) -> float:
    checks = [
        pay_ratio.pay_ratio is not None, pay_ratio.ceo_pay is not None,
        board.total_directors is not None, board.independent_directors is not None,
        board.female_directors is not None, board.independence_pct is not None,
        env.climate_mention_count > 0, env.carbon_mention_count > 0,
        gov.whistleblower_policy, gov.code_of_ethics_filed,
    ]
    return round(sum(checks) / len(checks), 2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def get_esg_profile(ticker: str, cik: Optional[str] = None) -> ESGProfile:
    """Full pipeline: resolve CIK → fetch DEF 14A + 10-K → extract signals → compute scores."""
    warnings: list[str] = []

    if not cik:
        cik = await _resolve_cik(ticker)
    if not cik:
        warnings.append(f"Could not resolve CIK for {ticker}")
        return ESGProfile(ticker=ticker, environmental_score=0.0, social_score=0.0,
                          governance_score=0.0, composite_score=0.0, warnings=warnings)

    cik = cik.zfill(10)
    proxy_text, text_10k = await asyncio.gather(fetch_def14a_text(cik), fetch_10k_text(cik))

    if not proxy_text:
        warnings.append("DEF 14A not available; social/governance signals may be incomplete")
    if not text_10k:
        warnings.append("10-K not available; environmental signals may be incomplete")

    pay_ratio = extract_pay_ratio(proxy_text) if proxy_text else PayRatioData()
    board = extract_board_composition(proxy_text) if proxy_text else BoardComposition()
    env = extract_environmental_signals(text_10k) if text_10k else EnvironmentalSignals(risk_score=0)
    gov = extract_governance_signals(proxy_text, text_10k)

    e_score, s_score, g_score = compute_esg_scores(pay_ratio, board, env, gov)
    composite = round((e_score + s_score + g_score) / 3.0, 1)
    completeness = _data_completeness(pay_ratio, board, env, gov)

    fiscal_year: Optional[int] = pay_ratio.year
    if fiscal_year is None and text_10k:
        yr_m = re.search(r"\b(20\d{2})\b", text_10k[:1000])
        if yr_m:
            fiscal_year = int(yr_m.group(1))

    logger.info("ESG profile computed", ticker=ticker, e=e_score, s=s_score, g=g_score,
                completeness=completeness)

    return ESGProfile(
        ticker=ticker, cik=cik, fiscal_year=fiscal_year,
        environmental_score=e_score, social_score=s_score, governance_score=g_score,
        composite_score=composite, pay_ratio=pay_ratio, board=board,
        environmental=env, governance=gov, data_completeness=completeness, warnings=warnings,
    )
