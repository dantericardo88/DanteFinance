"""
Query expansion for the SENTINEL RAG pipeline.
Dimension 56 — Smart synonym / query expansion (0→4).

Expands a raw user query with:
  1. Financial domain synonyms (hardcoded, instant)
  2. Ticker-to-company name mapping (EDGAR lookup)
  3. LLM expansion via Claude (optional, higher quality)
  4. HyDE — Hypothetical Document Embeddings (generate an ideal answer, embed that)

The expanded query improves both dense and sparse retrieval recall.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Financial synonym lexicon ─────────────────────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    # Fundamentals
    "revenue": ["sales", "net sales", "top line", "turnover"],
    "sales": ["revenue", "net revenue", "top line"],
    "earnings": ["profit", "net income", "net earnings", "bottom line", "EPS"],
    "eps": ["earnings per share", "diluted eps", "basic eps"],
    "ebitda": ["earnings before interest taxes depreciation amortization", "operating cash earnings"],
    "gross margin": ["gross profit margin", "gross profit percentage"],
    "operating margin": ["operating profit margin", "EBIT margin"],
    "free cash flow": ["FCF", "levered free cash flow", "unlevered FCF"],
    "capex": ["capital expenditures", "capital spending", "PP&E investment"],
    "debt": ["leverage", "borrowings", "long-term debt", "net debt"],
    "guidance": ["outlook", "forecast", "projection", "forward estimates"],
    # Valuation
    "pe ratio": ["price to earnings", "P/E", "earnings multiple", "price earnings ratio"],
    "ev/ebitda": ["enterprise value to EBITDA", "EV multiple"],
    "book value": ["net asset value", "NAV", "shareholders equity per share"],
    "fair value": ["intrinsic value", "DCF value", "target price"],
    "cheap": ["undervalued", "discount to peers", "low multiple", "value"],
    "expensive": ["overvalued", "premium", "high multiple", "priced to perfection"],
    # Risk
    "risk": ["uncertainty", "downside", "volatility", "exposure"],
    "downturn": ["recession", "contraction", "slowdown", "correction"],
    "default": ["bankruptcy", "insolvency", "credit event", "restructuring"],
    # Growth
    "growth": ["expansion", "acceleration", "ramp", "momentum"],
    "slowdown": ["deceleration", "headwinds", "normalization"],
    # Sectors
    "tech": ["technology", "software", "SaaS", "semiconductors"],
    "bank": ["financial institution", "lender", "depository"],
    "pharma": ["pharmaceutical", "biotech", "drug maker", "biopharma"],
    # Actions
    "buyback": ["share repurchase", "stock repurchase", "return of capital"],
    "dividend": ["distribution", "payout", "yield", "income"],
    "acquisition": ["M&A", "merger", "takeover", "deal"],
    "ipo": ["initial public offering", "listing", "going public"],
}

# Flip the map for reverse lookups too
_REVERSE_SYNONYMS: dict[str, str] = {}
for _primary, _alts in _SYNONYMS.items():
    for _alt in _alts:
        _REVERSE_SYNONYMS[_alt.lower()] = _primary


# ── Ticker → company name cache ───────────────────────────────────────────────

_TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "META": "Meta Platforms", "JPM": "JPMorgan Chase", "BRK.B": "Berkshire Hathaway",
    "V": "Visa", "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth",
    "WMT": "Walmart", "XOM": "ExxonMobil", "MA": "Mastercard",
    "PG": "Procter & Gamble", "HD": "Home Depot", "CVX": "Chevron",
    "LLY": "Eli Lilly", "ABBV": "AbbVie", "BAC": "Bank of America",
    "MRK": "Merck", "COST": "Costco", "AVGO": "Broadcom",
    "PEP": "PepsiCo", "ADBE": "Adobe", "CSCO": "Cisco",
    "KO": "Coca-Cola", "TMO": "Thermo Fisher", "ACN": "Accenture",
    "AMD": "Advanced Micro Devices", "INTC": "Intel",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
    "GLD": "Gold ETF", "TLT": "20+ Year Treasury ETF",
    "BTC": "Bitcoin", "ETH": "Ethereum",
}


# ── Models ────────────────────────────────────────────────────────────────────

class ExpandedQuery(BaseModel):
    original: str
    expanded: str                     # primary query string for retrieval
    synonyms_added: list[str]         # list of synonym phrases added
    ticker_expanded: bool = False     # whether ticker was expanded to company name
    company_names: list[str] = []     # detected company names
    hyde_document: str | None = None  # hypothetical ideal answer (for HyDE embedding)
    expansion_method: str = "lexical" # "lexical" | "llm" | "hyde"


# ── Core expansion functions ───────────────────────────────────────────────────

def _detect_tickers(query: str) -> list[str]:
    """Find capitalized 1-5 char words that look like tickers."""
    tokens = re.findall(r'\b[A-Z]{1,5}(?:\.[AB])?\b', query)
    return [t for t in tokens if t in _TICKER_NAMES]


def expand_lexical(query: str) -> ExpandedQuery:
    """
    Fast local synonym expansion. No API calls.

    Strategy:
    1. Detect any tickers and expand to company names
    2. Find financial terms in query and append synonyms
    3. Return expanded query string (original + synonym phrases)
    """
    q_lower = query.lower()
    synonyms_added: list[str] = []
    company_names: list[str] = []

    # Ticker expansion
    tickers = _detect_tickers(query)
    ticker_expanded = bool(tickers)
    for ticker in tickers:
        name = _TICKER_NAMES[ticker]
        company_names.append(name)
        if name.lower() not in q_lower:
            synonyms_added.append(name)

    # Synonym expansion — match primary terms
    for primary, alts in _SYNONYMS.items():
        if primary in q_lower:
            # Add up to 2 alternatives to keep query focused
            for alt in alts[:2]:
                if alt.lower() not in q_lower:
                    synonyms_added.append(alt)
            break  # one synonym expansion per query to avoid bloat

    # Reverse synonym check — if user typed an alt form, normalize to primary
    for alt, primary in _REVERSE_SYNONYMS.items():
        if alt in q_lower and primary not in q_lower:
            synonyms_added.append(primary)
            break

    # Build expanded query
    expansion_parts = [query]
    if company_names:
        expansion_parts.append(" ".join(company_names))
    if synonyms_added:
        expansion_parts.append(" ".join(synonyms_added[:4]))  # cap at 4 extras

    expanded = " ".join(expansion_parts)

    return ExpandedQuery(
        original=query,
        expanded=expanded,
        synonyms_added=synonyms_added,
        ticker_expanded=ticker_expanded,
        company_names=company_names,
        expansion_method="lexical",
    )


async def expand_with_llm(
    query: str,
    anthropic_api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    generate_hyde: bool = False,
) -> ExpandedQuery:
    """
    LLM-powered query expansion using Claude.

    Two modes:
    - Standard: Claude generates related financial terms and synonyms
    - HyDE: Claude writes a hypothetical ideal answer paragraph (embedded for retrieval)
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.warning("anthropic not installed — falling back to lexical expansion")
        return expand_lexical(query)

    client = AsyncAnthropic(api_key=anthropic_api_key)

    if generate_hyde:
        system = (
            "You are a financial analyst. Write a concise 2-3 sentence paragraph that would "
            "be the ideal answer to the following question, using specific financial terminology. "
            "Do not hedge or say you don't know — write as if you have full knowledge. "
            "This text will be used for semantic search, not shown to users."
        )
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        hyde_doc = response.content[0].text.strip()
        base = expand_lexical(query)
        return ExpandedQuery(
            original=query,
            expanded=f"{query} {hyde_doc}",
            synonyms_added=base.synonyms_added,
            ticker_expanded=base.ticker_expanded,
            company_names=base.company_names,
            hyde_document=hyde_doc,
            expansion_method="hyde",
        )
    else:
        system = (
            "You are a financial search query optimizer. Given a user query, output a JSON object "
            'with a single key "terms": a list of 4-6 additional financial terms, synonyms, '
            "or related concepts that would help find relevant documents. Keep each term short "
            "(1-4 words). Focus on financial jargon. Output only valid JSON."
        )
        response = await client.messages.create(
            model=model,
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        import json
        raw = response.content[0].text.strip()
        try:
            data = json.loads(raw)
            extra_terms: list[str] = data.get("terms", [])[:6]
        except (json.JSONDecodeError, KeyError):
            extra_terms = []

        base = expand_lexical(query)
        expanded = f"{query} {' '.join(extra_terms)}"
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            synonyms_added=base.synonyms_added + extra_terms,
            ticker_expanded=base.ticker_expanded,
            company_names=base.company_names,
            expansion_method="llm",
        )


async def expand_query(
    query: str,
    anthropic_api_key: Optional[str] = None,
    use_hyde: bool = False,
    use_llm: bool = True,
) -> ExpandedQuery:
    """
    Main entry point. Selects expansion strategy:
    - No API key → lexical only
    - API key + use_hyde → HyDE expansion (best for factual questions)
    - API key + use_llm → LLM synonym expansion (best for keyword queries)
    - Both false → lexical only
    """
    if not anthropic_api_key or (not use_llm and not use_hyde):
        return expand_lexical(query)

    return await expand_with_llm(
        query=query,
        anthropic_api_key=anthropic_api_key,
        generate_hyde=use_hyde,
    )


# ── MCP-compatible wrapper ────────────────────────────────────────────────────

async def expand_for_rag(
    query: str,
    settings=None,
    use_hyde: bool = False,
) -> str:
    """
    Convenience wrapper used by the RAG query pipeline.
    Returns the expanded query string ready for embedding + BM25.
    """
    if settings is None:
        from sentinel.core.config import get_settings
        settings = get_settings()

    api_key = getattr(settings, "anthropic_api_key", None)
    result = await expand_query(
        query=query,
        anthropic_api_key=api_key,
        use_hyde=use_hyde,
        use_llm=bool(api_key),
    )
    logger.debug(
        "Query expanded",
        original=result.original,
        method=result.expansion_method,
        synonyms=len(result.synonyms_added),
    )
    return result.expanded
