"""SEC EDGAR XBRL segment and geographic revenue breakdown parser — dimension #16."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# XBRL concept / axis constants

SEGMENT_CONCEPTS: list[str] = [
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
]

GEOGRAPHIC_CONCEPTS: list[str] = [
    "us-gaap:RevenueFromExternalCustomersByGeographicAreasTableTextBlock",
]

SEGMENT_AXES: list[str] = [
    "us-gaap:StatementBusinessSegmentsAxis",
    "us-gaap:GeographicAreasAxis",
    "us-gaap:ProductOrServiceAxis",
    "srt:StatementGeographicalAxis",
]

# Axes that indicate geographic segmentation
_GEO_AXES: frozenset[str] = frozenset(
    {
        "us-gaap:GeographicAreasAxis",
        "srt:StatementGeographicalAxis",
    }
)

# Revenue concept short names (no namespace prefix) for fast lookup
_REVENUE_CONCEPTS: frozenset[str] = frozenset(
    {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    }
)
_OPINCOME_CONCEPTS: frozenset[str] = frozenset(
    {
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    }
)

# Forms we care about
_ANNUAL_FORMS: frozenset[str] = frozenset({"10-K", "20-F"})
_QUARTERLY_FORMS: frozenset[str] = frozenset({"10-Q"})

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_HEADERS = {
    "User-Agent": "SENTINEL/1.0 research@example.com",
    "Accept": "application/json",
}

# Models


class SegmentRevenue(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str
    period_end: date
    period_type: str         # "annual" | "quarterly"
    filed_at: datetime
    segment_name: str        # e.g. "North America", "Cloud Services", "Consumer"
    segment_type: str        # "business" | "geographic" | "product"
    revenue: Decimal | None
    operating_income: Decimal | None
    revenue_pct: float | None  # % of total revenue
    currency: str = "USD"


class SegmentBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    period_end: date
    total_revenue: Decimal
    segments: list[SegmentRevenue]
    geographic_segments: list[SegmentRevenue]
    has_segment_data: bool


# Parsing helpers


def _norm_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits for EDGAR URLs."""
    return cik.strip().lstrip("0").zfill(10)


def _parse_filed(filed_str: str | None) -> datetime:
    if not filed_str:
        return datetime.min
    try:
        return datetime.fromisoformat(filed_str)
    except ValueError:
        try:
            return datetime.strptime(filed_str.strip(), "%Y-%m-%d")
        except ValueError:
            return datetime.min


def _period_type(obs: dict) -> str:
    form = obs.get("form", "")
    if form in _ANNUAL_FORMS:
        return "annual"
    if form in _QUARTERLY_FORMS:
        return "quarterly"
    # Heuristic: if start→end span ~12 months it's annual
    start = obs.get("start")
    end_str = obs.get("end") or obs.get("instant")
    if start and end_str:
        try:
            days = (date.fromisoformat(end_str) - date.fromisoformat(start)).days
            return "annual" if days >= 330 else "quarterly"
        except ValueError:
            pass
    return "annual"


def _segment_type_from_axis(axis: str) -> str:
    if axis in _GEO_AXES:
        return "geographic"
    if "Product" in axis or "Service" in axis:
        return "product"
    return "business"


def _label_from_segment_dict(seg: dict) -> str:
    """Human-readable name from EDGAR dimensional context dict or list."""
    if isinstance(seg, dict):
        val = seg.get("value") or seg.get("label") or ""
        if ":" in val:
            val = val.split(":")[-1]
        return _camel_to_words(val)
    return str(seg)


def _camel_to_words(s: str) -> str:
    """'NorthAmericaMember' → 'North America'."""
    import re
    s = s.replace("Member", "").replace("Segment", "").strip()
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s).strip()


def _axis_from_segment_dict(seg: dict | list) -> str | None:
    """Return the axis name from an EDGAR segment dimension annotation."""
    if isinstance(seg, dict):
        return seg.get("dimension") or seg.get("axis")
    if isinstance(seg, list):
        for item in seg:
            if isinstance(item, dict):
                dim = item.get("dimension") or item.get("axis")
                if dim:
                    return dim
    return None


# Core extraction


def extract_segments_from_facts(
    facts: dict, ticker: str, cik: str
) -> list[SegmentRevenue]:
    """Parse EDGAR companyfacts JSON for revenue/income obs with segment dimensions.

    Obs with a 'segment' key carry dimensional context (breakdown facts).
    Schema: facts['us-gaap'][concept]['units']['USD'][obs, ...]
    """
    segments: list[SegmentRevenue] = []
    gaap = facts.get("facts", {}).get("us-gaap", {})

    # key → {revenue, operating_income, meta} — joined by (period, name, type, form)
    seg_map: dict[tuple, dict] = {}

    for concept_name, concept_data in gaap.items():
        full_concept = f"us-gaap:{concept_name}"
        is_revenue = concept_name in _REVENUE_CONCEPTS
        is_opincome = concept_name in _OPINCOME_CONCEPTS

        if not (is_revenue or is_opincome):
            continue

        usd_obs: list[dict] = concept_data.get("units", {}).get("USD", [])

        for obs in usd_obs:
            seg_raw = obs.get("segment")
            if seg_raw is None:
                # Consolidated fact — not a segment breakdown
                continue

            axis = _axis_from_segment_dict(seg_raw)
            if axis is None:
                continue

            # Only process recognised segment axes
            matched_axis = next(
                (a for a in SEGMENT_AXES if a == axis or a.split(":")[-1] == axis.split(":")[-1]),
                None,
            )
            if matched_axis is None:
                continue

            end_str = obs.get("end") or obs.get("instant")
            if not end_str:
                continue
            try:
                period_end = date.fromisoformat(end_str)
            except ValueError:
                continue

            val = obs.get("val")
            if val is None:
                continue
            try:
                amount = Decimal(str(val))
            except InvalidOperation:
                continue

            filed_at = _parse_filed(obs.get("filed"))
            seg_label = _label_from_segment_dict(
                seg_raw if isinstance(seg_raw, dict) else (seg_raw[0] if seg_raw else {})
            )
            if not seg_label:
                continue

            seg_type = _segment_type_from_axis(matched_axis)
            pt = _period_type(obs)
            key = (period_end, seg_label, seg_type, pt)

            if key not in seg_map:
                seg_map[key] = {
                    "period_end": period_end,
                    "period_type": pt,
                    "filed_at": filed_at,
                    "segment_name": seg_label,
                    "segment_type": seg_type,
                    "revenue": None,
                    "operating_income": None,
                }
            entry = seg_map[key]
            # Keep latest filing if duplicate
            if filed_at > entry["filed_at"]:
                entry["filed_at"] = filed_at

            if is_revenue:
                if entry["revenue"] is None or filed_at >= entry["filed_at"]:
                    entry["revenue"] = amount
            elif is_opincome:
                if entry["operating_income"] is None or filed_at >= entry["filed_at"]:
                    entry["operating_income"] = amount

    for meta in seg_map.values():
        segments.append(
            SegmentRevenue(
                ticker=ticker,
                cik=cik,
                period_end=meta["period_end"],
                period_type=meta["period_type"],
                filed_at=meta["filed_at"] if meta["filed_at"] != datetime.min else datetime.utcnow(),
                segment_name=meta["segment_name"],
                segment_type=meta["segment_type"],
                revenue=meta["revenue"],
                operating_income=meta["operating_income"],
                revenue_pct=None,  # computed later with total
                currency="USD",
            )
        )

    logger.info(
        "segment_parser.extract ticker=%s segments_found=%d", ticker, len(segments)
    )
    return segments


def _compute_pcts(
    segments: list[SegmentRevenue], total: Decimal
) -> list[SegmentRevenue]:
    """Return new SegmentRevenue list with revenue_pct filled in."""
    if total == 0:
        return segments
    updated: list[SegmentRevenue] = []
    for s in segments:
        pct: float | None = None
        if s.revenue is not None:
            try:
                pct = float(s.revenue / total) * 100.0
            except (InvalidOperation, ZeroDivisionError):
                pass
        updated.append(s.model_copy(update={"revenue_pct": pct}))
    return updated


def _consolidated_revenue(facts: dict, period_end: date) -> Decimal:
    """Pull consolidated (non-dimensional) total revenue for a given period_end."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[Decimal] = []
    for concept_name in _REVENUE_CONCEPTS:
        usd_obs = gaap.get(concept_name, {}).get("units", {}).get("USD", [])
        for obs in usd_obs:
            if obs.get("segment") is not None:
                continue  # skip dimensional
            end_str = obs.get("end") or obs.get("instant")
            if not end_str:
                continue
            try:
                if date.fromisoformat(end_str) != period_end:
                    continue
            except ValueError:
                continue
            val = obs.get("val")
            if val is not None:
                try:
                    candidates.append(Decimal(str(val)))
                except InvalidOperation:
                    pass
    # Return the largest found (multiple concepts may overlap; biggest is safest)
    return max(candidates) if candidates else Decimal("0")


# HTTP layer


async def _fetch_company_facts(cik: str) -> dict:
    """Fetch raw EDGAR companyfacts JSON for a CIK with tenacity retries."""
    padded = _norm_cik(cik)
    url = EDGAR_FACTS_URL.format(cik=padded)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(
                headers=EDGAR_HEADERS, timeout=30.0, follow_redirects=True
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()


async def fetch_segment_filing(cik: str, accession_number: str) -> dict:
    """Fetch EDGAR XBRL companyfacts and return dimensional segment observations.

    accession_number is accepted for API compatibility; the full companyfacts
    endpoint is used because it aggregates all filings reliably.
    """
    facts = await _fetch_company_facts(cik)
    gaap = facts.get("facts", {}).get("us-gaap", {})

    dimensional_facts: dict[str, list[dict]] = {}
    for concept_name, concept_data in gaap.items():
        full_concept = f"us-gaap:{concept_name}"
        if full_concept not in SEGMENT_CONCEPTS:
            continue
        usd_obs = concept_data.get("units", {}).get("USD", [])
        dim_obs = [o for o in usd_obs if o.get("segment") is not None]
        if dim_obs:
            dimensional_facts[full_concept] = dim_obs

    logger.info(
        "fetch_segment_filing cik=%s accn=%s dimensional_concepts=%d",
        cik, accession_number, len(dimensional_facts),
    )
    return {"facts": facts, "dimensional": dimensional_facts}


# Public pipeline


async def get_segment_breakdown(
    ticker: str,
    cik: str,
    period_end: date | None = None,
) -> SegmentBreakdown:
    """Full pipeline: fetch companyfacts → extract segment revenue → compute %.

    If period_end is None, uses the most recent period with segment data.
    """
    logger.info("get_segment_breakdown ticker=%s cik=%s period=%s", ticker, cik, period_end)

    try:
        facts = await _fetch_company_facts(cik)
    except Exception as exc:
        logger.error("get_segment_breakdown fetch failed ticker=%s: %s", ticker, exc)
        return SegmentBreakdown(
            ticker=ticker,
            period_end=period_end or date.today(),
            total_revenue=Decimal("0"),
            segments=[],
            geographic_segments=[],
            has_segment_data=False,
        )

    all_segments = extract_segments_from_facts(facts, ticker, cik)

    if not all_segments:
        logger.info("get_segment_breakdown no segment data found ticker=%s", ticker)
        return SegmentBreakdown(
            ticker=ticker,
            period_end=period_end or date.today(),
            total_revenue=Decimal("0"),
            segments=[],
            geographic_segments=[],
            has_segment_data=False,
        )

    # Determine target period
    if period_end is None:
        # Pick most recent annual period that has revenue data
        annual_with_rev = [
            s for s in all_segments
            if s.period_type == "annual" and s.revenue is not None
        ]
        if annual_with_rev:
            period_end = max(s.period_end for s in annual_with_rev)
        else:
            period_end = max(s.period_end for s in all_segments)

    # Filter to target period (latest filing per segment to avoid duplicates)
    period_segs = [s for s in all_segments if s.period_end == period_end]

    best: dict[tuple, SegmentRevenue] = {}
    for s in period_segs:
        key = (s.segment_name, s.segment_type)
        if key not in best or s.filed_at > best[key].filed_at:
            best[key] = s
    period_segs = list(best.values())

    total_revenue = _consolidated_revenue(facts, period_end)
    if total_revenue == 0 and period_segs:
        biz = [s for s in period_segs if s.segment_type == "business" and s.revenue]
        if biz:
            total_revenue = sum(s.revenue for s in biz)  # type: ignore[misc]

    period_segs = _compute_pcts(period_segs, total_revenue)

    business_segs = sorted(
        [s for s in period_segs if s.segment_type in ("business", "product")],
        key=lambda s: s.revenue or Decimal("0"),
        reverse=True,
    )
    geo_segs = sorted(
        [s for s in period_segs if s.segment_type == "geographic"],
        key=lambda s: s.revenue or Decimal("0"),
        reverse=True,
    )

    logger.info(
        "get_segment_breakdown ticker=%s period=%s business=%d geo=%d total=%s",
        ticker, period_end, len(business_segs), len(geo_segs), total_revenue,
    )
    return SegmentBreakdown(
        ticker=ticker,
        period_end=period_end,
        total_revenue=total_revenue,
        segments=business_segs,
        geographic_segments=geo_segs,
        has_segment_data=bool(period_segs),
    )


async def get_geographic_breakdown(
    ticker: str, cik: str
) -> list[SegmentRevenue]:
    """Return only geographic-type segments from the latest annual period."""
    breakdown = await get_segment_breakdown(ticker, cik)
    return breakdown.geographic_segments
