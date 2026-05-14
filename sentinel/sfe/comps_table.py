"""Comparable company (comps) table builder — EDGAR XBRL + yfinance.

Fetches standardised financial metrics for a target ticker and its sector
peers, assembling a side-by-side valuation and operating comparison table.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
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


class PeerMetric(BaseModel):
    ticker: str
    company_name: str = ""
    sector: Optional[str] = None
    market_cap_usd: Optional[float] = None
    # Valuation
    pe_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    # Growth
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    # Size
    revenue_ttm: Optional[float] = None
    ebitda_ttm: Optional[float] = None


class CompsTable(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    peers: list[PeerMetric] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    data_source: str = "EDGAR XBRL + yfinance"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sector peer map  (target → peer list)
# ---------------------------------------------------------------------------

_SECTOR_PEERS: dict[str, list[str]] = {
    # Technology — mega-cap
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "CRM"],
    "GOOGL": ["META", "MSFT", "AMZN", "SNAP"],
    "GOOG":  ["META", "MSFT", "AMZN", "SNAP"],
    "META":  ["GOOGL", "SNAP", "PINS", "RDDT"],
    "AMZN": ["MSFT", "GOOGL", "AAPL", "WMT"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM"],
    "AMD":  ["NVDA", "INTC", "AVGO", "QCOM"],
    "INTC": ["NVDA", "AMD", "AVGO", "TSM"],
    "CRM":  ["MSFT", "SAP", "NOW", "ORCL"],
    "ORCL": ["MSFT", "SAP", "CRM", "IBM"],
    "NOW":  ["CRM", "MSFT", "WDAY", "HUBS"],
    # Electric vehicles
    "TSLA": ["GM", "F", "RIVN", "NIO"],
    "RIVN": ["TSLA", "NIO", "GM", "F"],
    "NIO":  ["TSLA", "RIVN", "LI", "XPEV"],
    # Traditional auto
    "GM": ["F", "STLA", "TM", "TSLA"],
    "F":  ["GM", "STLA", "TM", "TSLA"],
    # Financials — banks
    "JPM": ["BAC", "WFC", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "USB"],
    "WFC": ["JPM", "BAC", "C", "USB"],
    "GS":  ["MS", "JPM", "BAC", "BX"],
    "MS":  ["GS", "JPM", "BAC", "BX"],
    "C":   ["JPM", "BAC", "WFC", "USB"],
    # Financials — asset managers / insurance
    "BX":  ["KKR", "APO", "CG", "BAM"],
    "AXP": ["V", "MA", "DFS", "COF"],
    "V":   ["MA", "AXP", "DFS", "PYPL"],
    "MA":  ["V", "AXP", "DFS", "PYPL"],
    # Healthcare — pharma
    "JNJ": ["PFE", "MRK", "ABBV", "LLY"],
    "PFE": ["JNJ", "MRK", "ABBV", "BMY"],
    "MRK": ["JNJ", "PFE", "ABBV", "LLY"],
    "ABBV": ["JNJ", "PFE", "BMY", "LLY"],
    "LLY": ["ABBV", "JNJ", "NVO", "AZN"],
    # Healthcare — biotech / devices
    "AMGN": ["GILD", "BIIB", "REGN", "MRNA"],
    "GILD": ["AMGN", "BIIB", "REGN", "AZN"],
    "UNH": ["CVS", "HUM", "CI", "CNC"],
    # Energy
    "XOM": ["CVX", "COP", "BP", "SHEL"],
    "CVX": ["XOM", "COP", "BP", "TTE"],
    "COP": ["XOM", "CVX", "PXD", "OXY"],
    "OXY": ["COP", "CVX", "XOM", "DVN"],
    # Consumer staples
    "PG":  ["KO", "PEP", "CL", "UL"],
    "KO":  ["PEP", "MDLZ", "PG", "STZ"],
    "PEP": ["KO", "MDLZ", "PG", "STZ"],
    "WMT": ["TGT", "COST", "AMZN", "HD"],
    "COST": ["WMT", "TGT", "BJ", "AMZN"],
    # Consumer discretionary
    "HD":  ["LOW", "TGT", "WMT", "AMZN"],
    "LOW": ["HD", "TGT", "WMT", "AMZN"],
    "MCD": ["SBUX", "YUM", "QSR", "DPZ"],
    "SBUX": ["MCD", "YUM", "DPZ", "DNKN"],
    # Industrials
    "GE":  ["HON", "RTX", "BA", "MMM"],
    "HON": ["GE", "MMM", "RTX", "ETN"],
    "BA":  ["AIR", "LMT", "RTX", "GE"],
    "CAT": ["DE", "CMI", "PCAR", "AGCO"],
    # Materials
    "LIN": ["APD", "PX", "DD", "DOW"],
    "APD": ["LIN", "DD", "DOW", "ECL"],
    "NEM": ["GOLD", "AEM", "WPM", "KGC"],
    # Utilities
    "NEE": ["DUK", "SO", "AEP", "EXC"],
    "DUK": ["NEE", "SO", "AEP", "PPL"],
    # Real estate
    "AMT": ["CCI", "EQIX", "SBAC", "DLR"],
    "PLD": ["DRE", "EGP", "FR", "REXR"],
    # Communication
    "T":   ["VZ", "TMUS", "CMCSA", "CHTR"],
    "VZ":  ["T", "TMUS", "CMCSA", "CHTR"],
    "TMUS": ["T", "VZ", "CMCSA", "CHTR"],
    "NFLX": ["DIS", "PARA", "WBD", "AMZN"],
    "DIS": ["NFLX", "PARA", "WBD", "CMCSA"],
}

# Generic S&P 500 mega-caps used as fallback peers when ticker not in map
_GENERIC_PEERS: list[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V"]


def _get_peer_tickers(ticker: str, sector: Optional[str] = None) -> list[str]:
    """Return peer tickers for a given ticker.

    Uses the hardcoded sector peer map if the ticker is present.
    Falls back to generic mega-cap peers when the ticker is unknown.
    Excludes the ticker itself from the peer list.
    """
    ticker_upper = ticker.upper()
    peers = _SECTOR_PEERS.get(ticker_upper)
    if peers:
        return [p for p in peers if p != ticker_upper]
    # Fallback: generic mega-caps, excluding the ticker itself
    return [p for p in _GENERIC_PEERS if p.upper() != ticker_upper][:6]


# ---------------------------------------------------------------------------
# CIK resolution (shared with non_gaap_parser, inlined to keep modules independent)
# ---------------------------------------------------------------------------

_CIK_CACHE: dict[str, Optional[str]] = {}


async def _resolve_cik(ticker: str) -> Optional[str]:
    """Resolve CIK from EDGAR company_tickers.json with a process-local cache."""
    ticker_upper = ticker.upper()
    if ticker_upper in _CIK_CACHE:
        return _CIK_CACHE[ticker_upper]

    url = f"{_EDGAR_BASE}/files/company_tickers.json"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            data: dict = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker_upper] = cik
                return cik

        _CIK_CACHE[ticker_upper] = None
        logger.warning("CIK not found", ticker=ticker)
        return None
    except Exception as exc:
        logger.error("_resolve_cik error", ticker=ticker, error=str(exc))
        _CIK_CACHE[ticker_upper] = None
        return None


# ---------------------------------------------------------------------------
# XBRL facts fetcher
# ---------------------------------------------------------------------------

# XBRL concept short names we want to extract
_XBRL_CONCEPTS: dict[str, str] = {
    # Revenue (multiple possible GAAP concepts)
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "SalesRevenueGoodsNet": "revenue",
    # Gross profit
    "GrossProfit": "gross_profit",
    # Operating income
    "OperatingIncomeLoss": "operating_income",
    # Net income
    "NetIncomeLoss": "net_income",
    # EPS
    "EarningsPerShareDiluted": "eps_diluted",
    "EarningsPerShareBasic": "eps_basic",
    # D&A
    "DepreciationAndAmortization": "da",
    "DepreciationDepletionAndAmortization": "da",
    # CapEx
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "CapitalExpenditures": "capex",
    # Shareholders equity
    "StockholdersEquity": "equity",
    # Shares out
    "CommonStockSharesOutstanding": "shares_outstanding",
}


def _get_quarterly_observations(concept_data: dict) -> list[dict]:
    """Extract quarterly (10-Q) observations from a single XBRL concept data block."""
    results: list[dict] = []
    for unit, obs_list in concept_data.get("units", {}).items():
        if unit not in ("USD", "shares"):
            continue
        for obs in obs_list:
            form = obs.get("form", "")
            # Only include 10-Q and 10-K observations; skip if no start date (point-in-time)
            if form in ("10-Q", "10-K") and obs.get("start") and obs.get("end"):
                results.append({**obs, "_unit": unit})
    return results


def _ttm_from_quarterly(obs_list: list[dict]) -> Optional[float]:
    """Sum the last 4 non-overlapping quarterly observations (TTM)."""
    if not obs_list:
        return None

    # Sort by period end descending
    sorted_obs = sorted(obs_list, key=lambda o: o.get("end", ""), reverse=True)

    # Prefer 10-Q observations; keep 10-K as fallback for annual
    quarterlies = [o for o in sorted_obs if o.get("form") == "10-Q"]
    if len(quarterlies) >= 4:
        last4 = quarterlies[:4]
    elif sorted_obs:
        # Fall back: use the most recent annual
        annual = next((o for o in sorted_obs if o.get("form") == "10-K"), None)
        if annual:
            try:
                return float(annual["val"])
            except (KeyError, ValueError):
                return None
        return None
    else:
        return None

    try:
        return sum(float(o["val"]) for o in last4)
    except (KeyError, ValueError):
        return None


def _yoy_growth(obs_list: list[dict]) -> Optional[float]:
    """Compute YoY growth from same-quarter observations."""
    if not obs_list:
        return None
    quarterlies = sorted(
        [o for o in obs_list if o.get("form") == "10-Q"],
        key=lambda o: o.get("end", ""),
        reverse=True,
    )
    if len(quarterlies) < 5:
        return None
    try:
        current = float(quarterlies[0]["val"])
        prior = float(quarterlies[4]["val"])
        if prior == 0:
            return None
        return (current - prior) / abs(prior)
    except (KeyError, ValueError, IndexError):
        return None


async def _fetch_xbrl_facts(ticker: str) -> dict:
    """Fetch EDGAR companyfacts JSON and extract standardised TTM metrics.

    Returns a dict with keys matching _XBRL_CONCEPTS values plus computed
    derived metrics (gross_margin, operating_margin, net_margin, roe,
    revenue_growth_yoy, eps_growth_yoy, ebitda_ttm, company_name).
    """
    cik = await _resolve_cik(ticker)
    if cik is None:
        logger.warning("No CIK for XBRL fetch", ticker=ticker)
        return {"_error": "no_cik"}

    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.error("_fetch_xbrl_facts HTTP error", ticker=ticker, error=str(exc))
        return {"_error": str(exc)}

    company_name: str = data.get("entityName", "")
    gaap: dict = data.get("facts", {}).get("us-gaap", {})

    # Collect TTM values for each concept, picking the first concept that resolves
    raw: dict[str, Optional[float]] = {}
    obs_bank: dict[str, list[dict]] = {}   # label → list of quarterly obs

    for concept_name, label in _XBRL_CONCEPTS.items():
        if concept_name not in gaap:
            continue
        if label in raw and raw[label] is not None:
            continue  # already resolved from a prior concept alias
        obs_list = _get_quarterly_observations(gaap[concept_name])
        ttm = _ttm_from_quarterly(obs_list)
        if ttm is not None:
            raw[label] = ttm
            obs_bank[label] = obs_list

    # Compute derived metrics
    computed: dict = {"company_name": company_name}
    computed.update(raw)

    # Margins
    rev = raw.get("revenue")
    gp = raw.get("gross_profit")
    oi = raw.get("operating_income")
    ni = raw.get("net_income")
    eq = raw.get("equity")

    computed["gross_margin"] = (gp / rev) if (gp is not None and rev and rev != 0) else None
    computed["operating_margin"] = (oi / rev) if (oi is not None and rev and rev != 0) else None
    computed["net_margin"] = (ni / rev) if (ni is not None and rev and rev != 0) else None
    computed["roe"] = (ni / eq) if (ni is not None and eq and eq != 0) else None

    # EBITDA TTM = operating income + D&A
    da = raw.get("da")
    if oi is not None and da is not None:
        computed["ebitda_ttm"] = oi + da
    elif oi is not None:
        computed["ebitda_ttm"] = oi  # rough proxy
    else:
        computed["ebitda_ttm"] = None

    # YoY growth
    computed["revenue_growth_yoy"] = _yoy_growth(obs_bank.get("revenue", []))
    computed["eps_growth_yoy"] = _yoy_growth(obs_bank.get("eps_diluted", []))

    computed["revenue_ttm"] = raw.get("revenue")
    return computed


# ---------------------------------------------------------------------------
# Price data via yfinance
# ---------------------------------------------------------------------------

async def _fetch_price_data(ticker: str) -> dict:
    """Fetch current market data from yfinance for the given ticker.

    Wrapped in try/except ImportError so the module degrades gracefully when
    yfinance is not installed.
    """
    try:
        import yfinance as yf  # noqa: PLC0415 — lazy import by design
    except ImportError:
        logger.warning("yfinance not installed — price data unavailable", ticker=ticker)
        return {"_error": "yfinance_not_installed"}

    try:
        # yfinance is synchronous; run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _get_info() -> dict:
            t = yf.Ticker(ticker)
            return t.info or {}

        info: dict = await loop.run_in_executor(None, _get_info)

        return {
            "company_name": info.get("longName") or info.get("shortName", ""),
            "sector": info.get("sector"),
            "market_cap_usd": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "enterprise_value": info.get("enterpriseValue"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
        }
    except Exception as exc:
        logger.error("_fetch_price_data error", ticker=ticker, error=str(exc))
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Margin computation from XBRL facts
# ---------------------------------------------------------------------------

def _compute_margins(facts: dict) -> dict:
    """Return margin dict from an already-fetched XBRL facts dict.

    This is a convenience wrapper over values already computed inside
    _fetch_xbrl_facts; it re-derives them in case callers supply a raw facts
    dict directly.
    """
    rev = facts.get("revenue")
    gp = facts.get("gross_profit")
    oi = facts.get("operating_income")
    ni = facts.get("net_income")
    eq = facts.get("equity")

    return {
        "gross_margin": (gp / rev) if (gp is not None and rev and rev != 0) else None,
        "operating_margin": (oi / rev) if (oi is not None and rev and rev != 0) else None,
        "net_margin": (ni / rev) if (ni is not None and rev and rev != 0) else None,
        "roe": (ni / eq) if (ni is not None and eq and eq != 0) else None,
    }


# ---------------------------------------------------------------------------
# PeerMetric builder
# ---------------------------------------------------------------------------

def _build_peer_metric(ticker: str, xbrl: dict, price: dict) -> PeerMetric:
    """Merge XBRL facts and price data into a PeerMetric."""
    margins = _compute_margins(xbrl)

    # EV/EBITDA: prefer yfinance ratio; fall back to derived
    ev_ebitda = price.get("ev_ebitda")
    if ev_ebitda is None:
        ev = price.get("enterprise_value")
        ebitda = xbrl.get("ebitda_ttm")
        if ev and ebitda and ebitda != 0:
            ev_ebitda = ev / ebitda

    return PeerMetric(
        ticker=ticker.upper(),
        company_name=price.get("company_name") or xbrl.get("company_name", ""),
        sector=price.get("sector"),
        market_cap_usd=price.get("market_cap_usd"),
        pe_ratio=price.get("pe_ratio"),
        ev_ebitda=ev_ebitda,
        price_to_book=price.get("price_to_book"),
        price_to_sales=price.get("price_to_sales"),
        revenue_growth_yoy=xbrl.get("revenue_growth_yoy"),
        eps_growth_yoy=xbrl.get("eps_growth_yoy"),
        gross_margin=margins["gross_margin"],
        operating_margin=margins["operating_margin"],
        net_margin=margins["net_margin"],
        roe=margins["roe"],
        revenue_ttm=xbrl.get("revenue_ttm"),
        ebitda_ttm=xbrl.get("ebitda_ttm"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def get_comps_table(
    ticker: str,
    include_peers: Optional[list[str]] = None,
) -> CompsTable:
    """Build a comparable company table for the given ticker.

    Args:
        ticker: The primary ticker to analyse.
        include_peers: Optional explicit list of peer tickers to add or
            override the default sector peer map.

    Returns:
        CompsTable with the target company and all resolved peers.
    """
    ticker_upper = ticker.upper()
    table = CompsTable(ticker=ticker_upper)

    try:
        # 1. Determine peer list
        auto_peers = _get_peer_tickers(ticker_upper)
        if include_peers:
            merged = dict.fromkeys(include_peers + auto_peers)  # include_peers first, deduped
            peer_list = [p.upper() for p in merged if p.upper() != ticker_upper]
        else:
            peer_list = [p.upper() for p in auto_peers if p.upper() != ticker_upper]

        all_tickers = [ticker_upper] + peer_list

        # 2. Fetch XBRL and price data concurrently for all tickers
        xbrl_tasks = [_fetch_xbrl_facts(t) for t in all_tickers]
        price_tasks = [_fetch_price_data(t) for t in all_tickers]

        xbrl_results: list[dict] = await asyncio.gather(*xbrl_tasks, return_exceptions=False)
        price_results: list[dict] = await asyncio.gather(*price_tasks, return_exceptions=False)

        # 3. Build PeerMetric for every ticker
        all_metrics: list[PeerMetric] = []
        for t, xbrl, price in zip(all_tickers, xbrl_results, price_results):
            if xbrl.get("_error"):
                table.warnings.append(f"XBRL fetch failed for {t}: {xbrl['_error']}")
            if price.get("_error"):
                table.warnings.append(f"Price fetch failed for {t}: {price['_error']}")
            try:
                metric = _build_peer_metric(t, xbrl, price)
                all_metrics.append(metric)
            except Exception as exc:
                logger.error("PeerMetric build error", ticker=t, error=str(exc))
                table.warnings.append(f"Could not build metric for {t}: {exc}")

        # 4. Separate target from peers
        if all_metrics:
            target_metric = all_metrics[0]
            table.company_name = target_metric.company_name or None
            table.sector = target_metric.sector
            # The full peers list includes the target itself for easy tabular display
            table.peers = all_metrics
        else:
            table.warnings.append("No metrics could be built for any ticker")

        logger.info(
            "Comps table built",
            ticker=ticker_upper,
            peers=len(peer_list),
            warnings=len(table.warnings),
        )

    except Exception as exc:
        logger.error("get_comps_table unhandled error", ticker=ticker, error=str(exc))
        table.warnings.append(f"Unhandled error: {exc}")

    return table
