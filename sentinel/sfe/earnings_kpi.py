"""earnings_kpi.py — Visible Alpha-style LLM KPI extraction from EDGAR 10-K/10-Q filings.

Fetches the MD&A section from EDGAR and uses Claude (or fallback regex) to extract
structured KPIs, management tone, and forward guidance.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "SENTINEL financial-terminal richard.porras@realempanada.com"
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KPIValue(BaseModel):
    name: str
    value: str
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    trend: Optional[str] = None
    is_guidance: bool = False


class ManagementTone(BaseModel):
    overall: str = "neutral"
    confidence_score: float = 0.5
    forward_looking_mentions: int = 0
    risk_mentions: int = 0
    guidance_provided: bool = False
    key_themes: list[str] = Field(default_factory=list)


class EarningsKPIResult(BaseModel):
    ticker: str
    cik: Optional[str] = None
    filing_type: str
    period: str
    filed_date: Optional[date] = None
    kpis: list[KPIValue] = Field(default_factory=list)
    management_tone: ManagementTone = Field(default_factory=ManagementTone)
    revenue_guidance: Optional[str] = None
    eps_guidance: Optional[str] = None
    margin_commentary: Optional[str] = None
    extraction_confidence: float = 0.0
    data_source: str = "EDGAR"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

async def resolve_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to zero-padded 10-digit CIK via SEC bulk ticker file."""
    headers = {**_HEADERS, "Host": "www.sec.gov"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(EDGAR_TICKERS_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        logger.warning("CIK not found for ticker", ticker=ticker)
        return None
    except Exception as exc:
        logger.error("CIK resolution error", ticker=ticker, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# EDGAR fetch helpers
# ---------------------------------------------------------------------------

async def _fetch_submissions(cik: str) -> dict:
    """Fetch submissions JSON for a CIK."""
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    headers = {**_HEADERS, "Host": "data.sec.gov"}
    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _find_latest_filing(submissions: dict, form_type: str) -> Optional[dict]:
    """Return the most recent filing of the given type from submissions JSON."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    periods = recent.get("reportDate", [])

    for i, form in enumerate(forms):
        if form == form_type:
            return {
                "form": form,
                "filing_date": dates[i] if i < len(dates) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
                "period": periods[i] if i < len(periods) else None,
            }
    return None


async def _fetch_filing_document(cik: str, accession: str, primary_doc: str) -> str:
    """Fetch the raw text of the primary filing document."""
    acc_clean = accession.replace("-", "")
    cik_stripped = cik.lstrip("0") or "0"
    url = f"{EDGAR_ARCHIVES}/{cik_stripped}/{acc_clean}/{primary_doc}"
    await asyncio.sleep(0.5)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        return resp.text


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def _extract_mda_section(full_text: str) -> str:
    """
    Extract the MD&A section from filing text.
    Looks for 'ITEM 2' header through to 'ITEM 3' or 'ITEM 4'.
    """
    # Normalise whitespace for matching
    text = re.sub(r"\s+", " ", full_text)

    # Try HTML-stripped version if needed
    if "<" in text[:500]:
        text = _strip_html(text)
        text = re.sub(r"\s+", " ", text)

    patterns_start = [
        r"ITEM\s+2[\.\s]+MANAGEMENT[\'’]?S\s+DISCUSSION",
        r"ITEM\s+2[\.\s]+MANAGEMENT'S\s+DISCUSSION",
        r"ITEM\s+2[\.\s]+MD&A",
        r"MANAGEMENT[\'’]?S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF",
    ]
    patterns_end = [
        r"ITEM\s+3[\.\s]",
        r"ITEM\s+4[\.\s]",
        r"QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
    ]

    start_pos = None
    for pat in patterns_start:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        logger.warning("MD&A section not found; using first 8000 chars of text")
        return text[:8000]

    end_pos = len(text)
    for pat in patterns_end:
        m = re.search(pat, text[start_pos + 200:], re.IGNORECASE)
        if m:
            end_pos = start_pos + 200 + m.start()
            break

    mda = text[start_pos:end_pos]
    return mda[:8000]


async def fetch_mda_text(cik: str, form_type: str = "10-Q") -> tuple[str, str, Optional[date]]:
    """
    Fetch MD&A section text from the most recent 10-K or 10-Q.
    Returns (mda_text, period_string, filed_date).
    """
    cik = cik.zfill(10)
    try:
        submissions = await _fetch_submissions(cik)
    except Exception as exc:
        logger.error("Submissions fetch failed", cik=cik, error=str(exc))
        return ("", "unknown", None)

    filing = _find_latest_filing(submissions, form_type)
    if not filing:
        # fall back to 10-K if 10-Q not found
        alt = "10-K" if form_type == "10-Q" else "10-Q"
        filing = _find_latest_filing(submissions, alt)
    if not filing:
        return ("", "unknown", None)

    accession = filing.get("accession", "")
    primary_doc = filing.get("primary_doc", "")
    period_str = filing.get("period", "unknown")
    filed_str = filing.get("filing_date")
    filed_date: Optional[date] = None
    if filed_str:
        try:
            filed_date = date.fromisoformat(filed_str)
        except ValueError:
            pass

    if not accession or not primary_doc:
        return ("", period_str, filed_date)

    try:
        raw_text = await _fetch_filing_document(cik, accession, primary_doc)
    except Exception as exc:
        logger.error("Filing document fetch failed", cik=cik, accession=accession, error=str(exc))
        return ("", period_str, filed_date)

    mda_text = _extract_mda_section(raw_text)
    logger.info("MD&A extracted", cik=cik, length=len(mda_text), period=period_str)
    return (mda_text, period_str, filed_date)


# ---------------------------------------------------------------------------
# Regex fallback KPI extraction
# ---------------------------------------------------------------------------

def _try_parse_float(s: str) -> Optional[float]:
    """Best-effort float parse, stripping commas and currency symbols."""
    cleaned = re.sub(r"[,$]", "", s).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_kpis_regex(mda_text: str, period: str) -> list[KPIValue]:
    """Regex-based KPI extraction fallback when Claude is unavailable."""
    kpis: list[KPIValue] = []

    # Revenue
    m = re.search(r"(?:revenue|net\s+revenue|total\s+revenue|sales)[^\n.]*?\$([\d,.]+)\s*(billion|million|B\b|M\b)",
                  mda_text, re.IGNORECASE)
    if m:
        unit = "USD_billions" if m.group(2).lower() in ("billion", "b") else "USD_millions"
        kpis.append(KPIValue(name="Revenue", value=f"${m.group(1)} {m.group(2)}",
                             numeric_value=_try_parse_float(m.group(1)), unit=unit, period=period))

    # Percentage changes (up to 8 items)
    pos_dirs = {"increase", "growth", "improvement"}
    for m in re.finditer(r"([\d.]+)\s*%\s*(increase|decrease|growth|decline|improvement)",
                         mda_text, re.IGNORECASE):
        direction = m.group(2).lower()
        pct = _try_parse_float(m.group(1))
        ctx = " ".join(mda_text[max(0, m.start() - 60):m.start()].split()[-4:]).title()
        kpis.append(KPIValue(
            name=ctx or "Metric", value=f"{m.group(1)}% {direction}",
            numeric_value=pct if direction in pos_dirs else -(pct or 0),
            unit="percent", period=period,
            trend="improving" if direction in pos_dirs else "declining",
        ))
        if len(kpis) >= 9:
            break

    # EPS
    m = re.search(r"(?:earnings\s+per\s+(?:diluted\s+)?share|EPS)[^$\n]*\$([\d.]+)", mda_text, re.IGNORECASE)
    if m:
        kpis.append(KPIValue(name="EPS", value=f"${m.group(1)}",
                             numeric_value=_try_parse_float(m.group(1)), unit="USD", period=period))

    # Operating margin
    m = re.search(r"operating\s+(?:income\s+)?margin[^%\n]*?([\d.]+)\s*%", mda_text, re.IGNORECASE)
    if m:
        kpis.append(KPIValue(name="Operating Margin", value=f"{m.group(1)}%",
                             numeric_value=_try_parse_float(m.group(1)), unit="percent", period=period))

    return kpis


def _extract_tone_regex(mda_text: str) -> ManagementTone:
    """Heuristic tone analysis from MD&A text."""
    tl = mda_text.lower()
    opt = sum(tl.count(w) for w in ["strong","robust","exceed","outperform","growth","record",
                                     "momentum","confident","positive","increase","improve"])
    caut = sum(tl.count(w) for w in ["uncertain","challenging","headwind","volatile","risk",
                                      "concern","pressure","decline","decrease","adverse","difficult"])
    fwd = sum(tl.count(w) for w in ["expect","anticipate","guidance","outlook","forecast",
                                     "we believe","we project","next quarter"])
    risk = sum(tl.count(w) for w in ["risk","uncertainty","adverse","loss","impairment",
                                      "litigation","regulatory","inflation","currency","supply chain"])

    if opt > caut * 1.5:
        overall, conf = "optimistic", min(0.9, 0.55 + opt * 0.01)
    elif caut > opt * 1.5:
        overall, conf = "cautious", max(0.2, 0.5 - caut * 0.01)
    elif caut > opt:
        overall, conf = "concerned", 0.4
    else:
        overall, conf = "neutral", 0.5

    theme_map = {
        "revenue growth": tl.count("revenue growth"),
        "operating efficiency": tl.count("operating efficiency") + tl.count("cost reduction"),
        "margin expansion": tl.count("margin"),
        "market share": tl.count("market share"),
        "innovation": tl.count("innovat"),
        "supply chain": tl.count("supply chain"),
        "interest rates": tl.count("interest rate"),
        "currency headwinds": tl.count("foreign currency") + tl.count("currency"),
    }
    top_themes = [k for k, v in sorted(theme_map.items(), key=lambda x: x[1], reverse=True) if v > 0][:5]
    guidance_provided = any(w in tl for w in ["guidance", "outlook", "we expect", "we anticipate"])

    return ManagementTone(overall=overall, confidence_score=round(conf, 2),
                          forward_looking_mentions=fwd, risk_mentions=risk,
                          guidance_provided=guidance_provided, key_themes=top_themes)


# ---------------------------------------------------------------------------
# Claude extraction
# ---------------------------------------------------------------------------

_CLAUDE_SYSTEM = (
    "You are a senior financial analyst extracting KPIs from a 10-K or 10-Q MD&A section. "
    "Extract all numerical KPIs, management guidance, and tone. "
    "Output ONLY valid JSON with no markdown or prose outside the JSON object."
)

_CLAUDE_USER_TEMPLATE = (
    "Extract KPIs and management tone from this MD&A for {ticker} ({period}).\n\n"
    "MD&A TEXT:\n{mda_text}\n\n"
    "Output ONLY this JSON structure:\n"
    '{{"kpis":[{{"name":"Revenue","value":"$5.2B","numeric_value":5.2,"unit":"USD_billions",'
    '"period":"{period}","trend":"improving","is_guidance":false}}],'
    '"tone":{{"overall":"optimistic","confidence_score":0.75,"forward_looking_mentions":12,'
    '"risk_mentions":5,"guidance_provided":true,"key_themes":["revenue growth","margin expansion"]}},'
    '"revenue_guidance":"...", "eps_guidance":"...", "margin_commentary":"..."}}\n\n'
    "Rules: extract ALL numerical KPIs; is_guidance=true for forward-looking statements; "
    "trend: improving|declining|stable|first_report|null; "
    "unit: USD_billions|USD_millions|USD|percent|units|other; null if not found."
)


async def extract_kpis(
    mda_text: str,
    ticker: str,
    period: str,
    anthropic_api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[list[KPIValue], ManagementTone]:
    """
    Use Claude to extract KPIs from MD&A text.
    Returns (kpis, management_tone).
    """
    import anthropic  # lazy import — optional dependency

    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    user_msg = _CLAUDE_USER_TEMPLATE.format(
        ticker=ticker,
        period=period,
        mda_text=mda_text[:7000],  # stay within context for haiku
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fencing
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Claude JSON parse error", error=str(exc))
        return _extract_kpis_regex(mda_text, period), _extract_tone_regex(mda_text)
    except Exception as exc:
        logger.error("Claude API error", error=str(exc))
        return _extract_kpis_regex(mda_text, period), _extract_tone_regex(mda_text)

    kpis: list[KPIValue] = []
    for raw_kpi in data.get("kpis", []):
        try:
            kpis.append(KPIValue(**raw_kpi))
        except Exception:
            continue

    raw_tone = data.get("tone", {})
    try:
        tone = ManagementTone(**raw_tone)
    except Exception:
        tone = _extract_tone_regex(mda_text)

    return kpis, tone


def _extract_guidance_regex(mda_text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract revenue guidance, EPS guidance, margin commentary via regex."""
    def _first(pat: str) -> Optional[str]:
        m = re.search(pat, mda_text, re.IGNORECASE)
        return m.group(0).strip()[:200] if m else None

    return (
        _first(r"(?:revenue|net\s+sales)\s+(?:guidance|outlook|expect|anticipate)[^.]*?\$[\d,.]+[^\.\n]{0,80}"),
        _first(r"(?:EPS|earnings\s+per\s+(?:diluted\s+)?share)\s+(?:guidance|expect|anticipate)[^.]*?\$[\d.]+[^\.\n]{0,80}"),
        _first(r"(?:operating|gross|EBITDA)\s+margin[^.]*?(?:expect|anticipate|expand|compress|improve)[^.]{0,120}\."),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def get_earnings_kpi(
    ticker: str,
    cik: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    form_type: str = "10-Q",
) -> EarningsKPIResult:
    """
    Full pipeline: resolve CIK → fetch MD&A → extract KPIs → return structured result.
    """
    warnings: list[str] = []

    # 1. Resolve CIK
    if not cik:
        cik = await resolve_cik(ticker)
    if not cik:
        warnings.append(f"Could not resolve CIK for {ticker}")
        return EarningsKPIResult(
            ticker=ticker,
            filing_type=form_type,
            period="unknown",
            warnings=warnings,
        )

    cik = cik.zfill(10)

    # 2. Fetch MD&A
    mda_text, period, filed_date = await fetch_mda_text(cik, form_type)
    if not mda_text:
        warnings.append("MD&A text not available; filing may use iXBRL inline format")
        return EarningsKPIResult(
            ticker=ticker,
            cik=cik,
            filing_type=form_type,
            period=period,
            filed_date=filed_date,
            extraction_confidence=0.0,
            warnings=warnings,
        )

    extraction_confidence = min(1.0, len(mda_text) / 5000)

    # 3. Extract KPIs
    if anthropic_api_key:
        try:
            kpis, tone = await extract_kpis(mda_text, ticker, period, anthropic_api_key)
            rev_guidance = next((k.value for k in kpis if k.is_guidance and "revenue" in k.name.lower()), None)
            eps_guidance = next((k.value for k in kpis if k.is_guidance and "eps" in k.name.lower()), None)
            _, _, margin_commentary = _extract_guidance_regex(mda_text)
        except Exception as exc:
            logger.warning("Claude extraction failed, falling back to regex", error=str(exc))
            warnings.append(f"Claude extraction failed: {exc}; using regex fallback")
            kpis = _extract_kpis_regex(mda_text, period)
            tone = _extract_tone_regex(mda_text)
            rev_guidance, eps_guidance, margin_commentary = _extract_guidance_regex(mda_text)
    else:
        warnings.append("No Anthropic API key; using regex extraction (reduced accuracy)")
        kpis = _extract_kpis_regex(mda_text, period)
        tone = _extract_tone_regex(mda_text)
        rev_guidance, eps_guidance, margin_commentary = _extract_guidance_regex(mda_text)

    logger.info("Earnings KPI extraction complete", ticker=ticker, kpi_count=len(kpis),
                period=period, confidence=extraction_confidence)

    return EarningsKPIResult(
        ticker=ticker, cik=cik, filing_type=form_type, period=period, filed_date=filed_date,
        kpis=kpis, management_tone=tone, revenue_guidance=rev_guidance,
        eps_guidance=eps_guidance, margin_commentary=margin_commentary,
        extraction_confidence=round(extraction_confidence, 2), warnings=warnings,
    )
