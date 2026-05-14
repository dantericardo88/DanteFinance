"""IFRS financials for non-US SEC filers (20-F / XBRL) — Dimension #21."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_UA = "SENTINEL/1.0 research@sentinel.ai"
_TIMEOUT = 15.0

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_HEADERS_SEC = {"User-Agent": _UA, "Accept": "application/json"}
_HEADERS_EFTS = {"User-Agent": _UA, "Accept": "application/json"}

# ---------------------------------------------------------------------------
# IFRS tag map:  canonical_name → (preferred_tag, fallback_tag | None)
# ---------------------------------------------------------------------------

_INCOME_TAGS: list[tuple[str, str, Optional[str]]] = [
    ("Revenue", "Revenue", "RevenueFromContractsWithCustomers"),
    ("GrossProfit", "GrossProfit", None),
    ("OperatingProfit", "ProfitLossFromOperatingActivities", None),
    ("ProfitLoss", "ProfitLoss", None),
    ("EarningsPerShare", "BasicEarningsLossPerShare", None),
]

_BALANCE_TAGS: list[tuple[str, str, Optional[str]]] = [
    ("TotalAssets", "Assets", None),
    ("TotalEquity", "Equity", None),
    ("TotalLiabilities", "Liabilities", None),
    ("CashAndEquivalents", "CashAndCashEquivalents", None),
]

_CASHFLOW_TAGS: list[tuple[str, str, Optional[str]]] = [
    ("OperatingCashFlow", "CashFlowsFromUsedInOperatingActivities", None),
    ("CapExpenditures", "PurchaseOfPropertyPlantAndEquipment", None),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IFRSLineItem(BaseModel):
    tag: str
    label: str
    values: list[dict]  # [{"period_end": "2023-12-31", "value": 12345, "form": "20-F"}]
    unit: str = "USD"


class IFRSStatement(BaseModel):
    income_statement: list[IFRSLineItem]
    balance_sheet: list[IFRSLineItem]
    cash_flow: list[IFRSLineItem]


class IFRSRatios(BaseModel):
    revenue_growth_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None


class IFRSFundamentals(BaseModel):
    ticker: str
    cik: str
    entity_name: str
    filing_standard: str  # "IFRS" | "US-GAAP" | "mixed"
    latest_period: str
    statements: IFRSStatement
    ratios: IFRSRatios
    fiscal_year_end: Optional[str] = None
    country_of_incorporation: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    as_of: str


# ---------------------------------------------------------------------------
# CIK resolution helpers
# ---------------------------------------------------------------------------

async def _cik_from_tickers_json(ticker: str, client: httpx.AsyncClient) -> Optional[str]:
    """Try the bulk company_tickers.json endpoint first — fastest path."""
    try:
        resp = await client.get(_TICKERS_URL, headers={**_HEADERS_SEC, "Host": "www.sec.gov"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ifrs.cik_bulk_load failed", ticker=ticker, error=str(exc))
        return None

    needle = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == needle:
            return str(entry["cik_str"]).zfill(10)
    return None


async def _cik_from_efts_search(ticker: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fall back to EFTS full-text search limited to 20-F filings."""
    params = {
        "q": f'"{ticker}"',
        "forms": "20-F",
        "hits.hits._source": "period_of_report,display_names,entity_id",
    }
    try:
        resp = await client.get(
            f"{_EDGAR_SEARCH}/LATEST/search-index",
            params=params,
            headers={**_HEADERS_EFTS, "Host": "efts.sec.gov"},
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
    except Exception as exc:
        logger.warning("ifrs.efts_search failed", ticker=ticker, error=str(exc))
        return None

    if not hits:
        return None

    entity_id = hits[0].get("_source", {}).get("entity_id") or hits[0].get("_id", "")
    if entity_id:
        return str(entity_id).zfill(10)
    return None


async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> str:
    cik = await _cik_from_tickers_json(ticker, client)
    if cik:
        return cik
    cik = await _cik_from_efts_search(ticker, client)
    if cik:
        return cik
    raise ValueError(
        f"Ticker '{ticker}' not found in SEC company tickers or EFTS 20-F search. "
        "Confirm the ticker is an SEC-registered foreign private issuer."
    )


# ---------------------------------------------------------------------------
# XBRL companyfacts fetch
# ---------------------------------------------------------------------------

async def _fetch_companyfacts(cik: str, client: httpx.AsyncClient) -> dict:
    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = await client.get(url, headers={**_HEADERS_SEC, "Host": "data.sec.gov"})
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("ifrs.companyfacts http error", cik=cik, status=exc.response.status_code)
        raise
    except Exception as exc:
        logger.warning("ifrs.companyfacts failed", cik=cik, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# IFRS fact extraction
# ---------------------------------------------------------------------------

def _extract_tag(
    facts_ns: dict,
    tag: str,
    fallback: Optional[str],
    label: str,
    periods: int,
    form_filter: str = "20-F",
) -> Optional[IFRSLineItem]:
    """
    Pull up to `periods` annual observations for an IFRS tag from one namespace dict.
    Deduplicates by period_end keeping highest accession (lexicographic).
    Returns None if no data found.
    """
    obs_list: list[dict] = []
    for try_tag in ([tag] + ([fallback] if fallback else [])):
        concept_data = facts_ns.get(try_tag)
        if concept_data:
            for unit, obs_arr in concept_data.get("units", {}).items():
                filtered = [o for o in obs_arr if o.get("form") == form_filter]
                if filtered:
                    obs_list = filtered
                    used_unit = unit
                    break
        if obs_list:
            break
    else:
        return None

    # Deduplicate by end date — keep highest accn string (latest filing)
    by_end: dict[str, dict] = {}
    for obs in obs_list:
        end = obs.get("end") or obs.get("instant")
        if not end:
            continue
        accn = obs.get("accn", "")
        if end not in by_end or accn > by_end[end].get("accn", ""):
            by_end[end] = obs

    sorted_obs = sorted(by_end.values(), key=lambda o: o.get("end") or o.get("instant", ""), reverse=True)
    top = sorted_obs[:periods]

    values = [
        {
            "period_end": o.get("end") or o.get("instant"),
            "value": o.get("val"),
            "form": o.get("form", form_filter),
        }
        for o in top
        if o.get("val") is not None
    ]

    if not values:
        return None

    return IFRSLineItem(tag=tag, label=label, values=values, unit=used_unit)


def _extract_statement(
    facts_ns: dict,
    tag_list: list[tuple[str, str, Optional[str]]],
    periods: int,
    form_filter: str = "20-F",
) -> list[IFRSLineItem]:
    items: list[IFRSLineItem] = []
    for label, tag, fallback in tag_list:
        item = _extract_tag(facts_ns, tag, fallback, label, periods, form_filter)
        if item:
            items.append(item)
    return items


def _detect_filing_standard(facts: dict) -> str:
    has_ifrs = bool(facts.get("ifrs-full"))
    has_gaap = bool(facts.get("us-gaap"))
    if has_ifrs and has_gaap:
        return "mixed"
    if has_ifrs:
        return "IFRS"
    if has_gaap:
        return "US-GAAP"
    return "unknown"


def _latest_period_from_items(items: list[IFRSLineItem]) -> str:
    """Return the most recent period_end string across all line items."""
    dates: list[str] = []
    for item in items:
        for v in item.values:
            p = v.get("period_end")
            if p:
                dates.append(p)
    return max(dates) if dates else ""


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def _first_val(item: Optional[IFRSLineItem], idx: int = 0) -> Optional[float]:
    if item is None or len(item.values) <= idx:
        return None
    v = item.values[idx].get("value")
    return float(v) if v is not None else None


def _compute_ratios(
    income: list[IFRSLineItem],
    balance: list[IFRSLineItem],
) -> IFRSRatios:
    by_label = {i.label: i for i in income + balance}

    rev = by_label.get("Revenue")
    gross = by_label.get("GrossProfit")
    net = by_label.get("ProfitLoss")
    equity = by_label.get("TotalEquity")
    liabilities = by_label.get("TotalLiabilities")

    rev0 = _first_val(rev, 0)
    rev1 = _first_val(rev, 1)
    gross0 = _first_val(gross, 0)
    net0 = _first_val(net, 0)
    equity0 = _first_val(equity, 0)
    liabilities0 = _first_val(liabilities, 0)

    revenue_growth_pct: Optional[float] = None
    if rev0 is not None and rev1 and rev1 != 0:
        revenue_growth_pct = round((rev0 - rev1) / abs(rev1) * 100.0, 2)

    gross_margin_pct: Optional[float] = None
    if gross0 is not None and rev0 and rev0 != 0:
        gross_margin_pct = round(gross0 / rev0 * 100.0, 2)

    net_margin_pct: Optional[float] = None
    if net0 is not None and rev0 and rev0 != 0:
        net_margin_pct = round(net0 / rev0 * 100.0, 2)

    roe_pct: Optional[float] = None
    if net0 is not None and equity0 and equity0 != 0:
        roe_pct = round(net0 / equity0 * 100.0, 2)

    debt_to_equity: Optional[float] = None
    if liabilities0 is not None and equity0 and equity0 != 0:
        debt_to_equity = round((liabilities0 - equity0) / equity0, 4)

    return IFRSRatios(
        revenue_growth_pct=revenue_growth_pct,
        gross_margin_pct=gross_margin_pct,
        net_margin_pct=net_margin_pct,
        roe_pct=roe_pct,
        debt_to_equity=debt_to_equity,
    )


# ---------------------------------------------------------------------------
# Submissions metadata helpers
# ---------------------------------------------------------------------------

async def _fetch_submissions_meta(cik: str, client: httpx.AsyncClient) -> dict:
    url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        resp = await client.get(url, headers={**_HEADERS_SEC, "Host": "data.sec.gov"})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("ifrs.submissions_meta failed", cik=cik, error=str(exc))
        return {}


def _extract_fiscal_year_end(submissions: dict) -> Optional[str]:
    """e.g. '1231' → '12-31'."""
    fye = submissions.get("fiscalYearEnd")
    if fye and len(fye) == 4:
        return f"{fye[:2]}-{fye[2:]}"
    return fye or None


def _extract_country(submissions: dict) -> Optional[str]:
    return submissions.get("stateOfIncorporation") or submissions.get("incorporationState")


def _has_20f_filings(submissions: dict) -> bool:
    forms = submissions.get("filings", {}).get("recent", {}).get("form", [])
    return "20-F" in forms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def get_ifrs_fundamentals(ticker: str, periods: int = 4) -> IFRSFundamentals:
    """Fetch IFRS financials for non-US SEC filer via 20-F XBRL."""
    ticker = ticker.upper().strip()
    warnings: list[str] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        cik = await _resolve_cik(ticker, client)
        logger.info("ifrs.cik_resolved", ticker=ticker, cik=cik)

        try:
            companyfacts = await _fetch_companyfacts(cik, client)
        except Exception as exc:
            raise ValueError(f"Failed to fetch XBRL companyfacts for {ticker} (CIK {cik}): {exc}") from exc

        submissions = await _fetch_submissions_meta(cik, client)

    if submissions and not _has_20f_filings(submissions):
        warnings.append(f"{ticker} has no 20-F filings in recent submissions history.")

    entity_name = companyfacts.get("entityName", ticker)
    raw_facts = companyfacts.get("facts", {})

    filing_standard = _detect_filing_standard(raw_facts)

    # Prefer ifrs-full namespace; fall back to us-gaap
    if filing_standard in ("IFRS", "mixed"):
        primary_ns = raw_facts.get("ifrs-full", {})
    else:
        primary_ns = raw_facts.get("us-gaap", {})
        warnings.append(
            f"{ticker} has no ifrs-full XBRL facts; returning US-GAAP data from 20-F instead."
        )

    if not primary_ns:
        warnings.append(f"{ticker}: no IFRS or US-GAAP facts found in XBRL data.")
        empty_stmt = IFRSStatement(income_statement=[], balance_sheet=[], cash_flow=[])
        return IFRSFundamentals(
            ticker=ticker,
            cik=cik,
            entity_name=entity_name,
            filing_standard=filing_standard,
            latest_period="",
            statements=empty_stmt,
            ratios=IFRSRatios(),
            fiscal_year_end=_extract_fiscal_year_end(submissions),
            country_of_incorporation=_extract_country(submissions),
            warnings=warnings,
            as_of=datetime.utcnow().date().isoformat(),
        )

    income = _extract_statement(primary_ns, _INCOME_TAGS, periods)
    balance = _extract_statement(primary_ns, _BALANCE_TAGS, periods)
    cash_flow = _extract_statement(primary_ns, _CASHFLOW_TAGS, periods)

    all_items = income + balance + cash_flow
    if not all_items:
        warnings.append(
            f"{ticker}: XBRL namespace present but no recognised IFRS tags matched "
            f"(namespace keys: {list(primary_ns.keys())[:10]})"
        )

    latest_period = _latest_period_from_items(all_items)

    has_ifrs_data = bool(income or balance or cash_flow)
    has_gaap_fallback = bool(raw_facts.get("us-gaap")) and not has_ifrs_data
    if has_gaap_fallback and filing_standard == "IFRS":
        gaap_ns = raw_facts.get("us-gaap", {})
        income = _extract_statement(gaap_ns, _INCOME_TAGS, periods)
        balance = _extract_statement(gaap_ns, _BALANCE_TAGS, periods)
        cash_flow = _extract_statement(gaap_ns, _CASHFLOW_TAGS, periods)
        filing_standard = "US-GAAP"
        warnings.append(f"{ticker}: ifrs-full namespace empty; fell back to us-gaap namespace.")
        latest_period = _latest_period_from_items(income + balance + cash_flow)

    logger.info(
        "ifrs.extracted",
        ticker=ticker,
        cik=cik,
        filing_standard=filing_standard,
        income_items=len(income),
        balance_items=len(balance),
        cashflow_items=len(cash_flow),
        latest_period=latest_period,
    )

    ratios = _compute_ratios(income, balance)

    return IFRSFundamentals(
        ticker=ticker,
        cik=cik,
        entity_name=entity_name,
        filing_standard=filing_standard,
        latest_period=latest_period,
        statements=IFRSStatement(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
        ),
        ratios=ratios,
        fiscal_year_end=_extract_fiscal_year_end(submissions),
        country_of_incorporation=_extract_country(submissions),
        warnings=warnings,
        as_of=datetime.utcnow().date().isoformat(),
    )
