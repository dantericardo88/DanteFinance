"""Supply chain concentration and customer/supplier risk analytics from EDGAR XBRL.

Dimension 16 enhancement: segment & geographic revenue breakdown (+1 score).
New capability: supply chain concentration risk scoring.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@sentinel.ai",
    "Accept": "application/json",
}

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_CIK_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&company={ticker}&type=10-K"
    "&dateb=&owner=include&count=5&search_text=&output=atom"
)
EFTS_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22major+customer%22+%22{ticker}%22"
    "&forms=10-K&dateRange=custom&startdt={startdt}"
    "&hits.hits.total.value=true"
)

_CONCENTRATION_TAGS = (
    "ConcentrationRiskPercentage1",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NumberOfMajorCustomers",
    "RevenueRemainingPerformanceObligation",
)
_GEO_TAGS = (
    "RevenueFromExternalCustomersByGeographicAreas",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
_GEO_AXES = frozenset({"us-gaap:GeographicAreasAxis", "srt:StatementGeographicalAxis"})
_CONCENTRATION_AXES = frozenset({
    "us-gaap:ConcentrationRiskByCustomerAxis",
    "us-gaap:ConcentrationRiskByTypeAxis",
    "srt:MajorCustomersAxis",
})

_PCT_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%?\s*of\s*(?:total\s+)?(?:net\s+)?(?:revenue|sales)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CustomerConcentration(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    revenue_pct: float | None
    is_major: bool


class GeographicSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    region: str
    revenue_usd: float | None
    revenue_pct: float | None


class SupplyChainProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str | None
    filing_date: str | None
    major_customers: list[CustomerConcentration]
    num_major_customers: int | None
    top_customer_pct: float | None
    geographic_segments: list[GeographicSegment]
    international_revenue_pct: float | None
    hhi_geographic: float | None
    concentration_score: float
    diversification_score: float
    supply_chain_risk_score: float
    risk_flags: list[str]
    data_quality: str
    as_of: str
    warnings: list[str]


class ConcentrationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    supply_chain_risk_score: float
    top_customer_pct: float | None
    num_major_customers: int | None
    international_revenue_pct: float | None


class ConcentrationScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    max_concentration_filter: float
    results: list[ConcentrationSummary]
    highest_risk: list[str]
    most_concentrated: str | None
    most_diversified: str | None
    avg_risk_score: float | None
    as_of: str
    warnings: list[str]


async def _resolve_cik(ticker: str, client: httpx.AsyncClient) -> str | None:
    url = EDGAR_CIK_URL.format(ticker=ticker.upper())
    try:
        resp = await client.get(url, headers={**_HEADERS, "Accept": "application/atom+xml"})
        resp.raise_for_status()
        m = re.search(r"CIK=(\d+)", resp.text)
        if m:
            return m.group(1).zfill(10)
    except Exception as exc:
        logger.warning("supply_chain.cik_resolve failed ticker=%s error=%s", ticker, exc)
    return None


async def _fetch_facts(cik: str, client: httpx.AsyncClient) -> dict:
    resp = await client.get(EDGAR_FACTS_URL.format(cik=cik.zfill(10)))
    resp.raise_for_status()
    return resp.json()


def _camel_split(s: str) -> str:
    s = s.replace("Member", "").replace("Segment", "").strip()
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s).strip()


def _axis_of(seg_raw: dict | list | None) -> str | None:
    if isinstance(seg_raw, dict):
        return seg_raw.get("dimension") or seg_raw.get("axis")
    if isinstance(seg_raw, list):
        for item in seg_raw:
            if isinstance(item, dict):
                d = item.get("dimension") or item.get("axis")
                if d:
                    return d
    return None


def _label_of(seg_raw: dict | list | None) -> str:
    if isinstance(seg_raw, dict):
        val = seg_raw.get("value") or seg_raw.get("label") or ""
        if ":" in val:
            val = val.split(":")[-1]
        return _camel_split(val)
    if isinstance(seg_raw, list) and seg_raw:
        return _label_of(seg_raw[0])
    return ""


def _parse_concentration_from_facts(facts: dict) -> tuple[list[CustomerConcentration], int | None, str | None]:
    """Return (customers, num_major, filing_date) from XBRL facts."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    customers: dict[str, float] = {}
    filing_date: str | None = None
    num_major: int | None = None

    # NumberOfMajorCustomers — scalar
    n_data = gaap.get("NumberOfMajorCustomers", {})
    pure_obs = [o for o in n_data.get("units", {}).get("pure", []) if o.get("segment") is None]
    if pure_obs:
        best = max(pure_obs, key=lambda o: o.get("filed", ""))
        num_major = int(best.get("val", 0)) or None
        filing_date = best.get("filed")

    # ConcentrationRiskPercentage1 — dimensional by customer axis
    conc_data = gaap.get("ConcentrationRiskPercentage1", {})
    for obs in conc_data.get("units", {}).get("pure", []):
        seg_raw = obs.get("segment")
        if seg_raw is None:
            continue
        axis = _axis_of(seg_raw)
        if not axis:
            continue
        norm_axis = axis.split(":")[-1] if ":" in axis else axis
        if norm_axis not in {a.split(":")[-1] for a in _CONCENTRATION_AXES}:
            continue
        val = obs.get("val")
        if val is None:
            continue
        pct = float(val)
        if pct > 1.0:
            pct /= 100.0
        label = _label_of(seg_raw) or "Customer"
        filed = obs.get("filed", "")
        if label not in customers or filed > filing_date:
            customers[label] = max(customers.get(label, 0.0), pct)
            if not filing_date or filed > filing_date:
                filing_date = filed

    customer_list = [
        CustomerConcentration(description=k, revenue_pct=v, is_major=(v >= 0.10))
        for k, v in sorted(customers.items(), key=lambda x: x[1], reverse=True)
    ]
    return customer_list, num_major, filing_date


def _parse_geo_from_facts(facts: dict) -> list[GeographicSegment]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    geo_rev: dict[str, float] = {}

    for tag in _GEO_TAGS:
        concept = gaap.get(tag, {})
        obs_list = concept.get("units", {}).get("USD", [])
        for obs in obs_list:
            seg_raw = obs.get("segment")
            if seg_raw is None:
                continue
            axis = _axis_of(seg_raw)
            if not axis:
                continue
            norm = axis.split(":")[-1] if ":" in axis else axis
            if norm not in {a.split(":")[-1] for a in _GEO_AXES}:
                continue
            val = obs.get("val")
            if val is None:
                continue
            region = _label_of(seg_raw) or "Unknown"
            geo_rev[region] = max(geo_rev.get(region, 0.0), float(val))

    if not geo_rev:
        return []

    total = sum(geo_rev.values())
    segments = []
    for region, rev in sorted(geo_rev.items(), key=lambda x: x[1], reverse=True):
        pct = rev / total if total > 0 else None
        segments.append(GeographicSegment(region=region, revenue_usd=rev, revenue_pct=pct))
    return segments


async def _efts_customer_pcts(ticker: str, client: httpx.AsyncClient) -> list[float]:
    startdt = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    url = EFTS_SEARCH_URL.format(ticker=ticker.upper(), startdt=startdt)
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        pcts: list[float] = []
        for hit in hits[:5]:
            excerpt = hit.get("_source", {}).get("file_date", "") + " " + str(hit.get("highlight", {}) or "")
            for m in _PCT_RE.finditer(excerpt):
                pct = float(m.group(1))
                if 0 < pct < 100:
                    pcts.append(pct / 100.0)
        return pcts
    except Exception as exc:
        logger.warning("supply_chain.efts_fallback ticker=%s error=%s", ticker, exc)
        return []


def _compute_hhi(segments: list[GeographicSegment]) -> float | None:
    shares = [s.revenue_pct for s in segments if s.revenue_pct is not None]
    if not shares:
        return None
    arr = np.array(shares, dtype=float)
    return float(np.sum(arr ** 2) * 10000)


def _concentration_score(customers: list[CustomerConcentration], fallback_pcts: list[float]) -> float:
    all_pcts = [c.revenue_pct for c in customers if c.revenue_pct is not None]
    if not all_pcts:
        all_pcts = fallback_pcts
    if not all_pcts:
        return 5.0  # unknown, neutral

    top = max(all_pcts)
    if top >= 0.30:
        base = 8.0 + min(2.0, (top - 0.30) * 10.0)
    elif top >= 0.10:
        base = 5.0 + (top - 0.10) / 0.20 * 3.0
    else:
        base = max(0.0, top / 0.10 * 5.0)

    weighted = sum(p * 1.0 for p in all_pcts)
    combined = base * 0.7 + min(10.0, weighted * 10.0) * 0.3
    return round(min(10.0, max(0.0, combined)), 2)


def _diversification_score(hhi: float | None) -> float:
    if hhi is None:
        return 5.0
    raw = 10.0 - (hhi / 1000.0)
    return round(min(10.0, max(0.0, raw)), 2)


def _risk_flags(
    customers: list[CustomerConcentration],
    international_pct: float | None,
    fallback_pcts: list[float],
) -> list[str]:
    flags: list[str]= []
    all_pcts = [c.revenue_pct for c in customers if c.revenue_pct is not None] or fallback_pcts
    if any(p >= 0.10 for p in all_pcts):
        flags.append("customer concentration risk")
    if len(all_pcts) >= 3 and sum(sorted(all_pcts, reverse=True)[:3]) >= 0.50:
        flags.append("critical customer dependency")
    if international_pct is not None and international_pct < 0.20:
        flags.append("domestic revenue concentration")
    return flags


async def get_supply_chain_risk(ticker: str) -> SupplyChainProfile:
    """Fetch EDGAR XBRL data and return a SupplyChainProfile for the given ticker."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []
    data_quality = "estimated"

    async with httpx.AsyncClient(timeout=30, headers=_HEADERS) as client:
        cik = await _resolve_cik(ticker, client)
        if not cik:
            warnings.append(f"CIK resolution failed for {ticker}; scores are estimated")
            return SupplyChainProfile(
                ticker=ticker, cik=None, filing_date=None,
                major_customers=[], num_major_customers=None, top_customer_pct=None,
                geographic_segments=[], international_revenue_pct=None, hhi_geographic=None,
                concentration_score=5.0, diversification_score=5.0, supply_chain_risk_score=5.0,
                risk_flags=[], data_quality="estimated", as_of=as_of, warnings=warnings,
            )

        facts: dict = {}
        try:
            facts = await _fetch_facts(cik, client)
            data_quality = "xbrl"
        except Exception as exc:
            warnings.append(f"XBRL fetch failed: {exc}; falling back to EFTS text")

        customers, num_major, filing_date = _parse_concentration_from_facts(facts)
        geo_segments = _parse_geo_from_facts(facts)

        fallback_pcts: list[float] = []
        if not customers and data_quality == "xbrl":
            fallback_pcts = await _efts_customer_pcts(ticker, client)
            if fallback_pcts:
                data_quality = "text_extract"
                customers = [
                    CustomerConcentration(
                        description=f"Customer {chr(65+i)}",
                        revenue_pct=p,
                        is_major=(p >= 0.10),
                    )
                    for i, p in enumerate(fallback_pcts)
                ]
                warnings.append("Customer data sourced from EFTS text extraction")
        elif not customers:
            data_quality = "estimated"
            warnings.append("No concentration data found; scores are estimated")

    # --- derive international revenue ---
    intl_pct: float | None = None
    if geo_segments:
        us_pct = next(
            (s.revenue_pct for s in geo_segments
             if s.region and "united states" in s.region.lower()),
            None,
        )
        if us_pct is not None:
            intl_pct = round(1.0 - us_pct, 4)

    hhi = _compute_hhi(geo_segments)
    conc_score = _concentration_score(customers, fallback_pcts)
    div_score = _diversification_score(hhi)
    risk_score = round(conc_score * 0.6 + (10.0 - div_score) * 0.4, 2)
    top_cust_pct = max((c.revenue_pct for c in customers if c.revenue_pct is not None), default=None)
    flags = _risk_flags(customers, intl_pct, fallback_pcts)

    logger.info(
        "supply_chain.profile ticker=%s cik=%s customers=%d geo=%d risk=%.2f quality=%s",
        ticker, cik, len(customers), len(geo_segments), risk_score, data_quality,
    )
    return SupplyChainProfile(
        ticker=ticker,
        cik=cik,
        filing_date=filing_date,
        major_customers=customers,
        num_major_customers=num_major,
        top_customer_pct=top_cust_pct,
        geographic_segments=geo_segments,
        international_revenue_pct=intl_pct,
        hhi_geographic=hhi,
        concentration_score=conc_score,
        diversification_score=div_score,
        supply_chain_risk_score=risk_score,
        risk_flags=flags,
        data_quality=data_quality,
        as_of=as_of,
        warnings=warnings,
    )


async def screen_concentration_risk(
    tickers: list[str],
    max_customer_concentration: float = 0.30,
) -> ConcentrationScreen:
    """Screen multiple tickers for supply chain concentration risk in parallel."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    screen_warnings: list[str] = []

    profiles = await asyncio.gather(
        *[get_supply_chain_risk(t) for t in tickers],
        return_exceptions=True,
    )

    summaries: list[ConcentrationSummary] = []
    for t, result in zip(tickers, profiles):
        if isinstance(result, Exception):
            screen_warnings.append(f"{t}: fetch error — {result}")
            summaries.append(ConcentrationSummary(
                ticker=t, supply_chain_risk_score=5.0,
                top_customer_pct=None, num_major_customers=None,
                international_revenue_pct=None,
            ))
        else:
            p: SupplyChainProfile = result
            summaries.append(ConcentrationSummary(
                ticker=t,
                supply_chain_risk_score=p.supply_chain_risk_score,
                top_customer_pct=p.top_customer_pct,
                num_major_customers=p.num_major_customers,
                international_revenue_pct=p.international_revenue_pct,
            ))

    summaries.sort(key=lambda s: s.supply_chain_risk_score, reverse=True)

    # Filter by max concentration
    filtered = [
        s for s in summaries
        if s.top_customer_pct is None or s.top_customer_pct <= max_customer_concentration
    ]

    highest_risk = [s.ticker for s in summaries if s.supply_chain_risk_score >= 7.0]

    concentrated = [s for s in summaries if s.top_customer_pct is not None]
    most_concentrated = (
        max(concentrated, key=lambda s: s.top_customer_pct).ticker  # type: ignore[arg-type]
        if concentrated else None
    )
    most_diversified = summaries[-1].ticker if summaries else None

    scores = [s.supply_chain_risk_score for s in summaries]
    avg_risk = round(float(np.mean(scores)), 2) if scores else None

    logger.info(
        "supply_chain.screen tickers=%d filtered=%d highest_risk=%d",
        len(tickers), len(filtered), len(highest_risk),
    )
    return ConcentrationScreen(
        tickers_screened=len(tickers),
        max_concentration_filter=max_customer_concentration,
        results=filtered,
        highest_risk=highest_risk,
        most_concentrated=most_concentrated,
        most_diversified=most_diversified,
        avg_risk_score=avg_risk,
        as_of=as_of,
        warnings=screen_warnings,
    )
