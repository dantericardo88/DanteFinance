"""
SIL Natural-Language Screener — SENTINEL.

Translates free-text investment queries into structured DuckDB screening
criteria via Claude claude-haiku-4-5-20251001 tool-use, then executes them against the SSE
ScreenerEngine. Replaces the regex keyword hack in mcp_server.py.
"""
from __future__ import annotations

import asyncio
from typing import Any

import anthropic
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── Schema columns available in DuckDB screening universe ────────────────────
AVAILABLE_COLUMNS = [
    "ticker", "company_name", "sector", "industry",
    "market_cap", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "dividend_yield", "revenue_growth_yoy", "earnings_growth_yoy",
    "net_margin", "gross_margin", "roe", "roa",
    "debt_to_equity", "current_ratio", "quick_ratio",
    "rsi_14", "sma_20", "sma_50", "sma_200",
    "price_to_52w_high", "price_to_52w_low",
    "avg_volume_30d", "insider_buying_90d", "short_interest_pct",
    "institutional_ownership_pct", "beta",
]

# ── Tool definition for Claude ────────────────────────────────────────────────
SCREEN_TOOL: dict[str, Any] = {
    "name": "set_screen_criteria",
    "description": "Set quantitative screening criteria to filter stocks based on user request",
    "input_schema": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "description": "Column name from the schema",
                        },
                        "operator": {
                            "type": "string",
                            "enum": ["gt", "lt", "gte", "lte", "eq", "between"],
                        },
                        "value": {
                            "description": (
                                "Threshold value. For 'between', use [low, high] array."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this criterion captures the intent",
                        },
                    },
                    "required": ["field", "operator", "value"],
                },
            },
            "screen_name": {
                "type": "string",
                "description": "Short name for this screen",
            },
            "description": {
                "type": "string",
                "description": "What kind of stocks this finds",
            },
            "sort_by": {
                "type": "string",
                "description": "Field to sort results by",
            },
            "sort_asc": {
                "type": "boolean",
                "default": False,
            },
        },
        "required": ["criteria", "screen_name", "description"],
    },
}

# ── System prompt for Claude ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a quantitative equity screener for SENTINEL, a professional financial terminal.

Your job: translate natural-language investment queries into precise numeric screening criteria.

## Available schema columns (DuckDB)
ticker, company_name, sector, industry,
market_cap, pe_ratio, pb_ratio, ps_ratio, ev_ebitda,
dividend_yield, revenue_growth_yoy, earnings_growth_yoy,
net_margin, gross_margin, roe, roa,
debt_to_equity, current_ratio, quick_ratio,
rsi_14, sma_20, sma_50, sma_200,
price_to_52w_high, price_to_52w_low,
avg_volume_30d, insider_buying_90d, short_interest_pct,
institutional_ownership_pct, beta

## Operator semantics
- gt / gte: greater than / greater than or equal
- lt / lte: less than / less than or equal
- eq: exact match (use for string fields like sector, industry)
- between: range filter; value must be a 2-element array [low, high]

## Typical value ranges and concept mappings
### Valuation ("cheap" / "value")
- pe_ratio lt 15          → classic value threshold
- pb_ratio lt 2           → below book
- ps_ratio lt 2           → low price/sales
- ev_ebitda lt 10         → cheap enterprise value

### Profitability ("profitable" / "quality")
- net_margin gt 0.10      → >10% net profit margin
- gross_margin gt 0.40    → >40% gross margin
- roe gt 0.15             → >15% return on equity
- roa gt 0.08             → >8% return on assets

### Growth ("growth" / "hypergrowth")
- revenue_growth_yoy gt 0.20   → >20% YoY revenue growth
- earnings_growth_yoy gt 0.15  → >15% earnings growth
- revenue_growth_yoy gt 0.40   → hypergrowth threshold

### Size ("small cap" / "mid cap" / "large cap" / "mega cap")
- market_cap between [3e8, 2e9]    → small cap ($300M–$2B)
- market_cap between [2e9, 1e10]   → mid cap ($2B–$10B)
- market_cap gt 1e10               → large cap (>$10B)
- market_cap gt 2e11               → mega cap (>$200B)
- market_cap lt 3e8                → micro cap (<$300M)

### Momentum / Technical
- rsi_14 gt 50            → price momentum (above midpoint)
- rsi_14 gt 60            → strong momentum
- rsi_14 lt 35            → oversold / contrarian entry
- price_to_52w_high gt 0.85   → near 52-week high (momentum)
- price_to_52w_low gt 2.0     → well off lows

### Dividends ("dividend" / "income" / "yield")
- dividend_yield gt 0.03   → >3% yield (decent income)
- dividend_yield gt 0.05   → >5% yield (high income)
- dividend_yield between [0.02, 0.06]  → sustainable range

### Balance sheet / Safety ("low debt" / "fortress" / "safe")
- debt_to_equity lt 0.5    → conservative leverage
- debt_to_equity lt 1.0    → moderate leverage
- current_ratio gt 1.5     → liquid balance sheet
- quick_ratio gt 1.0       → true liquidity

### Insider / Ownership signals
- insider_buying_90d gt 0          → any insider buying
- insider_buying_90d gt 3          → meaningful insider conviction
- short_interest_pct lt 0.05       → low short interest
- institutional_ownership_pct gt 0.70  → institutional-grade

### Risk
- beta lt 0.8    → low volatility
- beta gt 1.3    → high beta / aggressive

## Rules
1. Always call the set_screen_criteria tool — do NOT explain in prose.
2. Choose 2–6 criteria; avoid over-constraining to the point of zero results.
3. Use exact column names from the schema above.
4. Provide a brief rationale per criterion explaining the logic.
5. Set sort_by to the most relevant ranking field for the query intent.
6. For sector/industry text filters, use eq operator and exact strings like
   "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary", etc.
"""


# ── Pydantic models ───────────────────────────────────────────────────────────

class ScreenCriterion(BaseModel):
    field: str
    operator: str  # "gt" | "lt" | "gte" | "lte" | "eq" | "between"
    value: Any
    rationale: str = ""


class NLScreenRequest(BaseModel):
    query: str
    max_results: int = 50
    anthropic_api_key: str


class NLScreenResult(BaseModel):
    query: str
    screen_name: str
    description: str
    criteria: list[ScreenCriterion]
    results: list[dict]
    result_count: int
    claude_rationale: str


# ── Example-based fallback screens ───────────────────────────────────────────

EXAMPLE_SCREENS: dict[str, list[dict[str, Any]]] = {
    "value": [
        {"field": "pe_ratio", "operator": "lt", "value": 15, "rationale": "Low P/E"},
        {"field": "pb_ratio", "operator": "lt", "value": 2, "rationale": "Below book value"},
        {"field": "net_margin", "operator": "gt", "value": 0.05, "rationale": "Minimally profitable"},
    ],
    "growth": [
        {"field": "revenue_growth_yoy", "operator": "gt", "value": 0.20, "rationale": ">20% revenue growth"},
        {"field": "earnings_growth_yoy", "operator": "gt", "value": 0.15, "rationale": ">15% earnings growth"},
        {"field": "net_margin", "operator": "gt", "value": 0.08, "rationale": "Margin positive"},
    ],
    "momentum": [
        {"field": "rsi_14", "operator": "gt", "value": 55, "rationale": "Above RSI midpoint"},
        {"field": "price_to_52w_high", "operator": "gt", "value": 0.85, "rationale": "Near 52-week high"},
        {"field": "avg_volume_30d", "operator": "gt", "value": 1_000_000, "rationale": "Liquid"},
    ],
    "quality": [
        {"field": "roe", "operator": "gt", "value": 0.20, "rationale": ">20% ROE"},
        {"field": "net_margin", "operator": "gt", "value": 0.12, "rationale": ">12% net margin"},
        {"field": "debt_to_equity", "operator": "lt", "value": 0.5, "rationale": "Low leverage"},
    ],
    "dividend": [
        {"field": "dividend_yield", "operator": "gt", "value": 0.035, "rationale": ">3.5% yield"},
        {"field": "net_margin", "operator": "gt", "value": 0.05, "rationale": "Earnings to support payout"},
        {"field": "debt_to_equity", "operator": "lt", "value": 1.0, "rationale": "Manageable debt"},
    ],
    "smallcap": [
        {"field": "market_cap", "operator": "between", "value": [3e8, 2e9], "rationale": "Small cap range"},
        {"field": "net_margin", "operator": "gt", "value": 0.05, "rationale": "Profitable"},
    ],
    "insider": [
        {"field": "insider_buying_90d", "operator": "gt", "value": 0, "rationale": "Insider conviction"},
        {"field": "net_margin", "operator": "gt", "value": 0.0, "rationale": "Not loss-making"},
    ],
}

_FALLBACK_KEYWORDS: dict[str, str] = {
    "value": "value",
    "cheap": "value",
    "undervalued": "value",
    "growth": "growth",
    "hypergrowth": "growth",
    "momentum": "momentum",
    "breakout": "momentum",
    "quality": "quality",
    "moat": "quality",
    "dividend": "dividend",
    "income": "dividend",
    "yield": "dividend",
    "small cap": "smallcap",
    "smallcap": "smallcap",
    "insider": "insider",
}


def fallback_screen(query: str) -> list[ScreenCriterion]:
    """
    Keyword-based fallback when no API key is available.
    Better than the old regex hack: maps concepts to validated threshold sets.
    Honest about its limitations — returns [] if no keyword matched.
    """
    q = query.lower()
    matched_key: str | None = None
    for keyword, screen_key in _FALLBACK_KEYWORDS.items():
        if keyword in q:
            matched_key = screen_key
            break

    if matched_key is None:
        logger.warning("nl_screener.fallback: no keyword matched", query=query)
        return []

    raw = EXAMPLE_SCREENS[matched_key]
    criteria = [ScreenCriterion(**c) for c in raw]
    logger.info(
        "nl_screener.fallback: matched screen",
        query=query,
        screen=matched_key,
        criteria_count=len(criteria),
    )
    return criteria


# ── Criteria → SSE screener dict translation ──────────────────────────────────

_OP_MAP: dict[str, str] = {
    "gt":  "_gt",
    "gte": "_gte",
    "lt":  "_lt",
    "lte": "_lte",
}

# Mapping from NL-screener column names to the legacy SSE screener dict keys
_FIELD_ALIAS: dict[str, str] = {
    "pe_ratio":                 "pe_ratio",
    "pb_ratio":                 "price_to_book",
    "ps_ratio":                 "price_to_sales",
    "ev_ebitda":                "ev_to_ebitda",
    "dividend_yield":           "dividend_yield",
    "revenue_growth_yoy":       "revenue_growth",
    "earnings_growth_yoy":      "earnings_growth",
    "net_margin":               "net_margin",
    "gross_margin":             "gross_margin",
    "roe":                      "roe",
    "roa":                      "roa",
    "debt_to_equity":           "debt_to_equity",
    "current_ratio":            "current_ratio",
    "quick_ratio":              "quick_ratio",
    "rsi_14":                   "rsi",
    "price_to_52w_high":        "price_vs_52w_high",
    "insider_buying_90d":       "insider_buying_30d",  # best available in SSE
    "short_interest_pct":       "short_interest",
    "market_cap":               "market_cap",
    "beta":                     "beta",
    "avg_volume_30d":           "avg_volume",
    "sector":                   "sectors",
    "institutional_ownership_pct": "institutional_pct",
}


def _criteria_to_sse_dict(
    criteria: list[ScreenCriterion],
    sort_by: str | None,
    sort_asc: bool,
    max_results: int,
) -> dict[str, Any]:
    """Convert ScreenCriterion list into the legacy SSE screener criteria dict."""
    d: dict[str, Any] = {}

    for c in criteria:
        raw_field = _FIELD_ALIAS.get(c.field, c.field)

        if c.operator == "between":
            if isinstance(c.value, (list, tuple)) and len(c.value) == 2:
                lo, hi = c.value
                # SSE uses _min / _max suffixes for range fields
                if raw_field == "market_cap":
                    d["market_cap_min"] = float(lo)
                    d["market_cap_max"] = float(hi)
                else:
                    d[f"{raw_field}_gt"] = float(lo)
                    d[f"{raw_field}_lt"] = float(hi)
            else:
                logger.warning(
                    "nl_screener: 'between' value must be [low, high]",
                    field=c.field,
                    value=c.value,
                )
        elif c.operator == "eq":
            # Sector / industry handled as list filter
            if raw_field == "sectors":
                existing = d.get("sectors", [])
                if isinstance(c.value, list):
                    existing.extend(c.value)
                else:
                    existing.append(str(c.value))
                d["sectors"] = existing
            else:
                d[raw_field] = c.value
        elif c.operator in _OP_MAP:
            suffix = _OP_MAP[c.operator]
            key = f"{raw_field}{suffix}"
            d[key] = float(c.value)
        else:
            logger.warning("nl_screener: unknown operator", operator=c.operator, field=c.field)

    # Sort / limit
    d["limit"] = max_results
    if sort_by:
        sse_sort = _FIELD_ALIAS.get(sort_by, sort_by)
        d["sort_by"] = sse_sort
        d["sort_desc"] = not sort_asc

    return d


# ── Claude tool-use translation ───────────────────────────────────────────────

async def translate_to_criteria(
    query: str,
    anthropic_api_key: str,
) -> tuple[str, str, list[ScreenCriterion], str, str | None, bool]:
    """
    Call Claude claude-haiku-4-5-20251001 with tool-use to translate a natural-language query
    into structured screening criteria.

    Returns:
        (screen_name, description, criteria_list, rationale_text, sort_by, sort_asc)
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    logger.info("nl_screener.translate: calling Claude", query=query)

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[SCREEN_TOOL],  # type: ignore[list-item]
        tool_choice={"type": "auto"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Translate this investment screening query into criteria:\n\n{query}"
                ),
            }
        ],
    )

    # Extract tool call from response
    tool_use_block = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "set_screen_criteria":
            tool_use_block = block
            break

    if tool_use_block is None:
        # Claude responded with text instead of a tool call — extract what we can
        text_response = " ".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        logger.warning(
            "nl_screener.translate: no tool call in response, using fallback",
            response_preview=text_response[:200],
        )
        fallback = fallback_screen(query)
        return (
            "Custom Screen",
            query,
            fallback,
            f"Claude did not return a tool call. Fallback used. Response: {text_response[:300]}",
        )

    args: dict[str, Any] = tool_use_block.input  # type: ignore[union-attr]

    raw_criteria: list[dict[str, Any]] = args.get("criteria", [])
    screen_name: str = args.get("screen_name", "Custom Screen")
    description: str = args.get("description", query)

    # Validate fields against known schema
    valid_criteria: list[ScreenCriterion] = []
    for c in raw_criteria:
        field = c.get("field", "")
        if field not in AVAILABLE_COLUMNS:
            logger.warning(
                "nl_screener.translate: unknown field, skipping",
                field=field,
            )
            continue
        valid_criteria.append(
            ScreenCriterion(
                field=field,
                operator=c.get("operator", "gt"),
                value=c.get("value"),
                rationale=c.get("rationale", ""),
            )
        )

    # Collect per-criterion rationales into a readable summary
    rationale_parts = [
        f"{sc.field} {sc.operator} {sc.value}: {sc.rationale}"
        for sc in valid_criteria
        if sc.rationale
    ]
    rationale = "\n".join(rationale_parts) if rationale_parts else description

    sort_by: str | None = args.get("sort_by")
    sort_asc: bool = args.get("sort_asc", False)

    logger.info(
        "nl_screener.translate: criteria extracted",
        screen_name=screen_name,
        criteria_count=len(valid_criteria),
        sort_by=sort_by,
    )
    return screen_name, description, valid_criteria, rationale, sort_by, sort_asc  # type: ignore[return-value]


# ── Full pipeline ─────────────────────────────────────────────────────────────

async def run_nl_screen(
    query: str,
    db_path: str,
    anthropic_api_key: str,
    max_results: int = 50,
) -> NLScreenResult:
    """
    Full pipeline: Claude tool-use → criteria → SSE screener execution → results.

    Args:
        query:             Natural-language investment query.
        db_path:           Path to DuckDB file (or ':memory:').
        anthropic_api_key: Anthropic API key. If empty string, uses fallback.
        max_results:       Max rows to return from screener.

    Returns:
        NLScreenResult with matched stocks and Claude's rationale.
    """
    from sentinel.sse.screener import ScreenerEngine

    # ── Step 1: Translate query to criteria ───────────────────────────────────
    sort_by: str | None = None
    sort_asc: bool = False

    if anthropic_api_key:
        try:
            result = await translate_to_criteria(query, anthropic_api_key)
            # translate_to_criteria returns 6-tuple when tool call succeeded
            screen_name, description, criteria, rationale, sort_by, sort_asc = result  # type: ignore[misc]
        except anthropic.APIError as exc:
            logger.error("nl_screener: Anthropic API error, using fallback", error=str(exc))
            criteria = fallback_screen(query)
            screen_name = "Fallback Screen"
            description = query
            rationale = f"API error — fallback keyword matching used: {exc}"
    else:
        criteria = fallback_screen(query)
        screen_name = "Fallback Screen"
        description = query
        rationale = "No API key provided — keyword-based fallback used."

    if not criteria:
        logger.warning("nl_screener: no criteria produced", query=query)
        return NLScreenResult(
            query=query,
            screen_name=screen_name,
            description=description,
            criteria=[],
            results=[],
            result_count=0,
            claude_rationale="No criteria could be generated for this query.",
        )

    # ── Step 2: Convert to SSE dict ───────────────────────────────────────────
    sse_criteria = _criteria_to_sse_dict(criteria, sort_by, sort_asc, max_results)
    logger.info("nl_screener: SSE criteria", criteria_dict=sse_criteria)

    # ── Step 3: Execute against SSE ScreenerEngine ────────────────────────────
    engine = ScreenerEngine(db_path=db_path)
    engine.initialize_tables()

    try:
        screen_results = engine.screen(sse_criteria)
    except Exception as exc:
        logger.error("nl_screener: screener execution error", error=str(exc))
        screen_results = []

    # ── Step 4: Serialise ScreenResult → plain dicts ──────────────────────────
    rows: list[dict] = []
    for sr in screen_results:
        row: dict[str, Any] = {
            "ticker": sr.ticker,
            "name": sr.name,
            "score": sr.score,
        }
        # Include fields dict if populated
        if sr.fields:
            row.update(sr.fields)
        rows.append(row)

    return NLScreenResult(
        query=query,
        screen_name=screen_name,
        description=description,
        criteria=criteria,
        results=rows,
        result_count=len(rows),
        claude_rationale=rationale,
    )


# ── Synchronous wrapper ───────────────────────────────────────────────────────

def run_nl_screen_sync(
    query: str,
    db_path: str,
    anthropic_api_key: str,
    max_results: int = 50,
) -> NLScreenResult:
    """Synchronous wrapper around run_nl_screen for non-async callers."""
    return asyncio.run(
        run_nl_screen(
            query=query,
            db_path=db_path,
            anthropic_api_key=anthropic_api_key,
            max_results=max_results,
        )
    )
