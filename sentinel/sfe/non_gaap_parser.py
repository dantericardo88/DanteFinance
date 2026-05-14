"""Non-GAAP metrics parser — extracts Adjusted EBITDA, FCF, Non-GAAP EPS, and
reconciliation tables from SEC EDGAR 8-K earnings releases."""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT = "SENTINEL financial-terminal richard.porras@realempanada.com"
_EDGAR_BASE = "https://data.sec.gov"
_SEC_BASE = "https://www.sec.gov"
_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class NonGAAPMetric(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None          # "millions", "billions", "per share"
    period: Optional[str] = None        # "Q1 2024", "FY 2023"
    gaap_equivalent: Optional[str] = None
    adjustment_items: list[str] = Field(default_factory=list)
    is_guidance: bool = False


class GaapReconciliation(BaseModel):
    gaap_metric: str
    gaap_value: Optional[float] = None
    adjustments: list[tuple[str, float]] = Field(default_factory=list)
    non_gaap_value: Optional[float] = None
    period: str = ""


class EarningsReleaseResult(BaseModel):
    ticker: str
    cik: Optional[str] = None
    period: str = ""
    filed_date: Optional[date] = None
    non_gaap_metrics: list[NonGAAPMetric] = Field(default_factory=list)
    reconciliations: list[GaapReconciliation] = Field(default_factory=list)
    management_guidance: list[NonGAAPMetric] = Field(default_factory=list)
    press_release_url: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter that preserves whitespace structure."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag == "script":
            self._in_script = True
        elif tag == "style":
            self._in_style = True
        elif tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")
        elif tag == "td":
            self._parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script":
            self._in_script = False
        elif tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if not self._in_script and not self._in_style:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
        return stripper.get_text()
    except Exception:
        # Fallback: crude tag removal
        return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# EDGAR network helpers
# ---------------------------------------------------------------------------

async def resolve_cik(ticker: str) -> Optional[str]:
    """Look up CIK from EDGAR company_tickers.json."""
    url = f"{_EDGAR_BASE}/files/company_tickers.json"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            data: dict = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                logger.info("CIK resolved", ticker=ticker, cik=cik)
                return cik
        logger.warning("CIK not found", ticker=ticker)
        return None
    except Exception as exc:
        logger.error("resolve_cik error", ticker=ticker, error=str(exc))
        return None


async def fetch_latest_8k(cik: str) -> Optional[dict]:
    """Return metadata for the most recent 8-K with an earnings-related description."""
    url = f"{_SEC_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=20&search_text=&output=atom"
    # Use submissions API — more reliable JSON
    submissions_url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    earnings_keywords = re.compile(
        r"earnings|results\s+of\s+operations|financial\s+results|quarterly\s+results"
        r"|revenue|income|press\s+release",
        re.IGNORECASE,
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(submissions_url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        form_types: list[str] = filings.get("form", [])
        accessions: list[str] = filings.get("accessionNumber", [])
        descriptions: list[str] = filings.get("primaryDocument", [])
        filing_dates: list[str] = filings.get("filingDate", [])
        items: list[str] = filings.get("items", [])

        for i, form in enumerate(form_types):
            if form != "8-K":
                continue
            desc = items[i] if i < len(items) else ""
            doc = descriptions[i] if i < len(descriptions) else ""
            if earnings_keywords.search(desc) or earnings_keywords.search(doc):
                return {
                    "accession": accessions[i],
                    "primary_doc": doc,
                    "filed_date": filing_dates[i] if i < len(filing_dates) else None,
                    "description": desc,
                }

        # Fallback: return first 8-K regardless of description
        for i, form in enumerate(form_types):
            if form == "8-K":
                return {
                    "accession": accessions[i],
                    "primary_doc": descriptions[i] if i < len(descriptions) else "",
                    "filed_date": filing_dates[i] if i < len(filing_dates) else None,
                    "description": items[i] if i < len(items) else "",
                }

        logger.warning("No 8-K found", cik=cik)
        return None
    except Exception as exc:
        logger.error("fetch_latest_8k error", cik=cik, error=str(exc))
        return None


async def fetch_8k_text(cik: str, accession: str, doc: str) -> str:
    """Fetch the 8-K HTML document and return plain text."""
    acc_clean = accession.replace("-", "")
    cik_stripped = cik.lstrip("0")
    url = f"{_SEC_BASE}/Archives/edgar/data/{cik_stripped}/{acc_clean}/{doc}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type or doc.endswith((".htm", ".html")):
                return _strip_html(resp.text)
            return resp.text
    except Exception as exc:
        logger.error("fetch_8k_text error", cik=cik, accession=accession, doc=doc, error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# Non-GAAP regex extraction
# ---------------------------------------------------------------------------

# (pattern, metric_name, gaap_equivalent)
_NON_GAAP_PATTERNS: list[tuple[re.Pattern, str, Optional[str]]] = [
    (
        re.compile(
            r"[Aa]djusted\s+EBITDA[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Adjusted EBITDA",
        "EBITDA",
    ),
    (
        re.compile(
            r"[Aa]djusted\s+(?:diluted\s+)?EPS[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "Adjusted EPS",
        "EPS Diluted",
    ),
    (
        re.compile(
            r"[Nn]on-GAAP\s+(?:diluted\s+)?(?:EPS|[Ee]arnings\s+[Pp]er\s+[Ss]hare)[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "Non-GAAP EPS",
        "EPS Diluted",
    ),
    (
        re.compile(
            r"[Ff]ree\s+[Cc]ash\s+[Ff]low[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Free Cash Flow",
        "Net Cash from Operations",
    ),
    (
        re.compile(
            r"[Aa]djusted\s+[Oo]perating\s+[Ii]ncome[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Adjusted Operating Income",
        "Operating Income",
    ),
    (
        re.compile(
            r"[Aa]djusted\s+[Nn]et\s+[Ii]ncome[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Adjusted Net Income",
        "Net Income",
    ),
    (
        re.compile(
            r"[Nn]on-GAAP\s+(?:gross\s+)?[Oo]perating\s+[Ii]ncome[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Non-GAAP Operating Income",
        "Operating Income",
    ),
    (
        re.compile(
            r"[Nn]on-GAAP\s+[Nn]et\s+[Ii]ncome[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Non-GAAP Net Income",
        "Net Income",
    ),
    (
        re.compile(
            r"[Aa]djusted\s+[Gg]ross\s+[Pp]rofit[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?",
            re.IGNORECASE,
        ),
        "Adjusted Gross Profit",
        "Gross Profit",
    ),
]

# Common adjustment items found near non-GAAP metrics
_ADJUSTMENT_TERMS = [
    "stock-based compensation",
    "stock based compensation",
    "share-based compensation",
    "amortization of intangibles",
    "amortization of acquired intangibles",
    "restructuring",
    "impairment",
    "acquisition-related",
    "merger-related",
    "litigation",
    "one-time",
    "non-recurring",
    "transaction costs",
    "depreciation and amortization",
]

_PERIOD_RE = re.compile(
    r"(?:(?:Q[1-4]|first|second|third|fourth)\s+(?:quarter|qtr)[\s\w,]*?(?:20\d{2}|19\d{2})"
    r"|(?:full[- ]year|fiscal\s+year|FY|year\s+ended)[^\n]{0,40}?(?:20\d{2}|19\d{2})"
    r"|(?:three|six|nine|twelve)\s+months\s+ended[^\n]{0,40}?(?:20\d{2}|19\d{2}))",
    re.IGNORECASE,
)

_GUIDANCE_RE = re.compile(
    r"(?:guidance|outlook|expects?|anticipates?|forecast)[^\n]{0,120}"
    r"(?:Adjusted\s+EBITDA|Adjusted\s+EPS|Non-GAAP|Free\s+Cash\s+Flow)[^\n]{0,120}?\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_value(raw: str) -> Optional[float]:
    """Convert comma-formatted numeric string to float."""
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _normalize_unit(unit_str: Optional[str]) -> Optional[str]:
    if not unit_str:
        return None
    u = unit_str.lower().strip()
    if u in ("million", "m"):
        return "millions"
    if u in ("billion", "b"):
        return "billions"
    return u


def _find_period(text: str, start: int, window: int = 400) -> Optional[str]:
    """Search a window around a match position for a period description."""
    snippet = text[max(0, start - window): start + window]
    m = _PERIOD_RE.search(snippet)
    return m.group(0).strip() if m else None


def _find_adjustments(text: str, start: int, window: int = 600) -> list[str]:
    """Find adjustment items mentioned near a non-GAAP metric."""
    snippet = text[max(0, start - window): start + window].lower()
    found = []
    for term in _ADJUSTMENT_TERMS:
        if term in snippet:
            found.append(term)
    return found


def extract_non_gaap_regex(text: str) -> list[NonGAAPMetric]:
    """Extract non-GAAP metrics from press-release plain text using regex."""
    metrics: list[NonGAAPMetric] = []
    seen: set[tuple[str, Optional[float]]] = set()

    for pattern, name, gaap_equiv in _NON_GAAP_PATTERNS:
        for match in pattern.finditer(text):
            raw_val = match.group(1)
            unit_raw = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            value = _parse_value(raw_val)
            unit = _normalize_unit(unit_raw)
            period = _find_period(text, match.start())
            adjustments = _find_adjustments(text, match.start())
            key = (name, value)
            if key in seen:
                continue
            seen.add(key)
            # Determine unit from context when not captured in the match
            if unit is None:
                ctx = text[max(0, match.start() - 80): match.end() + 80].lower()
                if "per share" in ctx or "diluted" in ctx:
                    unit = "per share"
                elif "million" in ctx or " m " in ctx:
                    unit = "millions"
                elif "billion" in ctx:
                    unit = "billions"
            metrics.append(NonGAAPMetric(
                name=name,
                value=value,
                unit=unit,
                period=period,
                gaap_equivalent=gaap_equiv,
                adjustment_items=adjustments,
                is_guidance=False,
            ))

    logger.info("Non-GAAP metrics extracted", count=len(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Reconciliation extraction
# ---------------------------------------------------------------------------

# Patterns for reconciliation blocks
_RECON_START_RE = re.compile(
    r"(?:GAAP\s+(?:net\s+income|operating\s+income|earnings)|reconciliation\s+of\s+(?:GAAP|non-GAAP))",
    re.IGNORECASE,
)

_ADJUSTMENT_LINE_RE = re.compile(
    r"^[ \t]*(.{5,60}?)\s{2,}(\(?\d[\d,]*(?:\.\d+)?\)?)\s*$",
    re.MULTILINE,
)

_GAAP_VALUE_RE = re.compile(
    r"GAAP\s+(?:net\s+income|operating\s+income|earnings)[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

_NON_GAAP_VALUE_RE = re.compile(
    r"(?:Non-GAAP|Adjusted)\s+(?:net\s+income|operating\s+income|earnings|EBITDA)[^\n]{0,80}?\$?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_reconciliation_regex(text: str) -> list[GaapReconciliation]:
    """Attempt to extract GAAP→non-GAAP reconciliation tables from press-release text."""
    reconciliations: list[GaapReconciliation] = []

    # Split text into candidate reconciliation blocks
    blocks: list[str] = []
    positions = [m.start() for m in _RECON_START_RE.finditer(text)]
    for pos in positions:
        block = text[pos: pos + 2000]
        blocks.append(block)

    if not blocks:
        # Try the entire text as one block
        blocks = [text]

    for block in blocks:
        gaap_m = _GAAP_VALUE_RE.search(block)
        non_gaap_m = _NON_GAAP_VALUE_RE.search(block)

        gaap_value = _parse_value(gaap_m.group(1)) if gaap_m else None
        non_gaap_value = _parse_value(non_gaap_m.group(1)) if non_gaap_m else None

        # Determine metric names
        if gaap_m:
            gaap_name_m = re.search(
                r"GAAP\s+(net\s+income|operating\s+income|earnings)", gaap_m.group(0), re.IGNORECASE
            )
            gaap_metric = gaap_name_m.group(0).strip() if gaap_name_m else "GAAP Net Income"
        else:
            gaap_metric = "GAAP Net Income"

        # Extract adjustment lines (label + dollar amount on same line)
        adjustments: list[tuple[str, float]] = []
        for line_m in _ADJUSTMENT_LINE_RE.finditer(block):
            label = line_m.group(1).strip()
            raw_amount = line_m.group(2).replace(",", "").replace("(", "-").replace(")", "")
            # Skip lines that look like the main GAAP/non-GAAP totals
            if re.search(r"(GAAP|Non-GAAP|Adjusted)\s+(net|operating|earnings|EBITDA)", label, re.IGNORECASE):
                continue
            try:
                amount = float(raw_amount)
                if label and abs(amount) > 0:
                    adjustments.append((label, amount))
            except ValueError:
                continue

        # Only record if we have at least GAAP or non-GAAP value
        if gaap_value is None and non_gaap_value is None:
            continue

        period = _find_period(text, text.find(block[:50])) or ""

        reconciliations.append(GaapReconciliation(
            gaap_metric=gaap_metric,
            gaap_value=gaap_value,
            adjustments=adjustments[:20],  # cap to avoid noise
            non_gaap_value=non_gaap_value,
            period=period,
        ))

    logger.info("Reconciliations extracted", count=len(reconciliations))
    return reconciliations


# ---------------------------------------------------------------------------
# Management guidance extraction
# ---------------------------------------------------------------------------

def _extract_guidance(text: str) -> list[NonGAAPMetric]:
    """Extract forward-looking non-GAAP guidance from press-release text."""
    guidance: list[NonGAAPMetric] = []
    seen: set[tuple[str, Optional[float]]] = set()

    guidance_section_re = re.compile(
        r"(?:financial\s+guidance|(?:full[- ]year|fiscal\s+(?:year|20\d{2}))\s+(?:outlook|guidance)"
        r"|(?:Q[1-4]\s+)?(?:20\d{2}\s+)?(?:outlook|guidance|targets?))[^\n]{0,200}",
        re.IGNORECASE,
    )

    guidance_value_re = re.compile(
        r"(Adjusted\s+EBITDA|Adjusted\s+EPS|Non-GAAP\s+EPS|Free\s+Cash\s+Flow|Adjusted\s+(?:Operating|Net)\s+Income)"
        r"[^\n]{0,120}?\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|per\s+share)?",
        re.IGNORECASE,
    )

    for section_m in guidance_section_re.finditer(text):
        window = text[section_m.start(): section_m.start() + 1500]
        for m in guidance_value_re.finditer(window):
            name = m.group(1).strip()
            value = _parse_value(m.group(2))
            unit = _normalize_unit(m.group(3))
            period = _find_period(text, section_m.start() + m.start())
            key = (name, value)
            if key in seen:
                continue
            seen.add(key)
            guidance.append(NonGAAPMetric(
                name=name,
                value=value,
                unit=unit,
                period=period,
                is_guidance=True,
            ))

    return guidance


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def get_non_gaap_metrics(
    ticker: str,
    cik: Optional[str] = None,
) -> EarningsReleaseResult:
    """Full pipeline: resolve CIK → fetch latest 8-K → parse non-GAAP metrics."""
    result = EarningsReleaseResult(ticker=ticker.upper())

    try:
        # 1. Resolve CIK
        if cik is None:
            cik = await resolve_cik(ticker)
        if cik is None:
            result.warnings.append(f"Could not resolve CIK for {ticker}")
            return result
        result.cik = cik

        # 2. Fetch latest 8-K metadata
        filing = await fetch_latest_8k(cik)
        if filing is None:
            result.warnings.append(f"No 8-K found for CIK {cik}")
            return result

        accession = filing["accession"]
        doc = filing.get("primary_doc", "")
        filed_str = filing.get("filed_date")
        if filed_str:
            try:
                result.filed_date = date.fromisoformat(filed_str)
            except ValueError:
                pass

        acc_clean = accession.replace("-", "")
        cik_stripped = cik.lstrip("0")
        result.press_release_url = (
            f"{_SEC_BASE}/Archives/edgar/data/{cik_stripped}/{acc_clean}/{doc}"
        )

        # 3. Fetch and strip 8-K text
        if not doc:
            result.warnings.append("No primary document found in 8-K filing")
            return result

        text = await fetch_8k_text(cik, accession, doc)
        if not text:
            result.warnings.append("Could not fetch 8-K document text")
            return result

        # 4. Extract non-GAAP metrics
        result.non_gaap_metrics = extract_non_gaap_regex(text)

        # 5. Extract reconciliations
        result.reconciliations = extract_reconciliation_regex(text)

        # 6. Extract guidance
        result.management_guidance = _extract_guidance(text)

        # 7. Infer period from best available period string
        all_periods = (
            [m.period for m in result.non_gaap_metrics if m.period]
            + [r.period for r in result.reconciliations if r.period]
        )
        if all_periods:
            result.period = all_periods[0]
        elif result.filed_date:
            result.period = result.filed_date.strftime("%Y")

        logger.info(
            "Non-GAAP parse complete",
            ticker=ticker,
            metrics=len(result.non_gaap_metrics),
            reconciliations=len(result.reconciliations),
            guidance=len(result.management_guidance),
        )

    except Exception as exc:
        logger.error("get_non_gaap_metrics unhandled error", ticker=ticker, error=str(exc))
        result.warnings.append(f"Unhandled error: {exc}")

    return result
