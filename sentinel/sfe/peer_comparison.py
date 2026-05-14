"""peer_comparison.py — Auto-generating peer/competitor comparison tables.

Auto-discovers peers via GICS sector/industry classification from yfinance,
computes a comprehensive multi-metric comparison table, and ranks the subject
company versus its peers on each metric.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Peer discovery maps
# ---------------------------------------------------------------------------

INDUSTRY_PEERS: dict[str, list[str]] = {
    "Software—Application": ["MSFT", "ORCL", "CRM", "SAP", "NOW", "INTU", "ADSK"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT"],
    "Internet Retail": ["AMZN", "EBAY", "ETSY", "W", "CHWY", "OSTK"],
    "Biotechnology": ["AMGN", "GILD", "REGN", "BIIB", "VRTX", "MRNA", "BNTX"],
    "Pharmaceuticals": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AZN", "NVO"],
    "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "Insurance": ["BRK-B", "MET", "PRU", "AIG", "HIG", "TRV", "ALL", "CB"],
    "Oil & Gas E&P": ["XOM", "CVX", "COP", "EOG", "PXD", "OXY", "DVN", "MRO"],
    "Utilities—Regulated Electric": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE", "XEL"],
    "Retail—Defensive": ["WMT", "COST", "TGT", "KR", "DG", "DLTR", "BJ"],
    "Airlines": ["DAL", "UAL", "AAL", "LUV", "JBLU", "ALK", "SAVE"],
    "Automobiles": ["TSLA", "GM", "F", "RIVN", "NIO", "LCID", "STLA"],
    "Media—Diversified": ["DIS", "NFLX", "WBD", "PARA", "FOXA", "CMCSA"],
    "Telecom Services": ["T", "VZ", "TMUS", "DISH", "LUMN", "SHEN"],
    "REITs": ["PLD", "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "WELL"],
}

SECTOR_FALLBACK: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM"],
    "Healthcare": ["JNJ", "UNH", "ABBV", "MRK", "PFE", "TMO", "ABT"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SPGI"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "BKR"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL"],
    "Industrials": ["HON", "CAT", "UPS", "RTX", "LMT", "GE", "MMM"],
    "Materials": ["LIN", "APD", "FCX", "NEM", "NUE", "PKG", "IP"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "SPG", "O"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC"],
}

# ---------------------------------------------------------------------------
# Metric metadata: (field_name, lower_is_better)
# ---------------------------------------------------------------------------

_METRIC_CONFIGS: list[tuple[str, bool]] = [
    ("pe_ratio", True),
    ("forward_pe", True),
    ("ev_ebitda", True),
    ("ps_ratio", True),
    ("pb_ratio", True),
    ("revenue_growth_pct", False),
    ("gross_margin_pct", False),
    ("operating_margin_pct", False),
    ("net_margin_pct", False),
    ("roe_pct", False),
    ("debt_to_equity", True),
    ("dividend_yield_pct", False),
    ("beta", True),
    ("return_1y_pct", False),
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PeerMetrics(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ps_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    beta: Optional[float] = None
    return_1y_pct: Optional[float] = None


class PeerRank(BaseModel):
    metric: str
    subject_value: Optional[float] = None
    subject_rank: int              # 1 = best
    total_peers: int
    percentile: float              # 0-100, higher = better relative standing
    best_ticker: str
    worst_ticker: str


class PeerComparisonResult(BaseModel):
    subject_ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    peers: list[str]
    metrics_table: list[PeerMetrics]   # subject + peers, sorted by market cap desc
    rankings: list[PeerRank]
    overall_percentile: float          # avg percentile across all ranked metrics
    verdict: str                       # "top quartile" | "above average" | "below average" | "bottom quartile"
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Peer discovery
# ---------------------------------------------------------------------------


def _discover_peers(ticker: str, industry: Optional[str], sector: Optional[str]) -> list[str]:
    """Return candidate peer tickers using industry then sector fallback."""
    ticker_upper = ticker.upper()

    if industry and industry in INDUSTRY_PEERS:
        candidates = INDUSTRY_PEERS[industry]
    elif sector and sector in SECTOR_FALLBACK:
        candidates = SECTOR_FALLBACK[sector]
    else:
        # Generic large-cap fallback
        candidates = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM"]

    return [p for p in candidates if p.upper() != ticker_upper]


# ---------------------------------------------------------------------------
# yfinance data fetching (lazy, inside asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_info_sync(ticker: str) -> dict:
    """Synchronous yfinance info fetch — runs inside asyncio.to_thread."""
    import yfinance as yf  # noqa: PLC0415 — lazy import by design
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.error("yfinance info error", ticker=ticker, error=str(exc))
        return {"_error": str(exc)}
    return info


def _fetch_history_sync(ticker: str) -> Optional[float]:
    """Compute 52-week return from 1Y history — runs inside asyncio.to_thread."""
    import yfinance as yf  # noqa: PLC0415 — lazy import by design
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None
        return float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
    except Exception as exc:
        logger.error("yfinance history error", ticker=ticker, error=str(exc))
        return None


async def _fetch_ticker_data(ticker: str) -> tuple[dict, Optional[float]]:
    """Fetch info dict and 52W return concurrently for a single ticker."""
    info, ret = await asyncio.gather(
        asyncio.to_thread(_fetch_info_sync, ticker),
        asyncio.to_thread(_fetch_history_sync, ticker),
    )
    return info, ret


def _build_peer_metrics(ticker: str, info: dict, ret_1y: Optional[float]) -> PeerMetrics:
    """Assemble a PeerMetrics object from raw yfinance info."""

    def _pct(val: Optional[float]) -> Optional[float]:
        """Convert a decimal ratio to percent, returning None if missing."""
        return round(val * 100, 4) if val is not None else None

    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    fcf_yield: Optional[float] = None
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100

    return PeerMetrics(
        ticker=ticker.upper(),
        company_name=info.get("longName") or info.get("shortName"),
        market_cap=market_cap,
        pe_ratio=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        ev_ebitda=info.get("enterpriseToEbitda"),
        ps_ratio=info.get("priceToSalesTrailing12Months"),
        pb_ratio=info.get("priceToBook"),
        revenue_growth_pct=_pct(info.get("revenueGrowth")),
        gross_margin_pct=_pct(info.get("grossMargins")),
        operating_margin_pct=_pct(info.get("operatingMargins")),
        net_margin_pct=_pct(info.get("profitMargins")),
        roe_pct=_pct(info.get("returnOnEquity")),
        debt_to_equity=info.get("debtToEquity"),
        dividend_yield_pct=_pct(info.get("dividendYield")),
        beta=info.get("beta"),
        return_1y_pct=round(ret_1y, 4) if ret_1y is not None else None,
    )


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------


def _compute_rankings(
    subject_ticker: str,
    all_metrics: list[PeerMetrics],
) -> list[PeerRank]:
    """Rank subject vs peers for each metric; return list of PeerRank."""
    rankings: list[PeerRank] = []
    subject_ticker_upper = subject_ticker.upper()

    for field_name, lower_is_better in _METRIC_CONFIGS:
        # Collect (ticker, value) pairs where value is not None
        pairs = [
            (m.ticker, getattr(m, field_name))
            for m in all_metrics
            if getattr(m, field_name) is not None
        ]
        if len(pairs) < 2:
            continue

        subject_pair = next((p for p in pairs if p[0] == subject_ticker_upper), None)
        if subject_pair is None:
            continue

        # Sort: ascending if lower_is_better, descending otherwise
        sorted_pairs = sorted(pairs, key=lambda p: p[1], reverse=not lower_is_better)
        tickers_ranked = [p[0] for p in sorted_pairs]

        subject_rank = tickers_ranked.index(subject_ticker_upper) + 1  # 1-based
        n = len(tickers_ranked)

        # Percentile: rank 1 → 100%, rank N → 0%
        percentile = round((1 - (subject_rank - 1) / (n - 1)) * 100, 2) if n > 1 else 100.0

        rankings.append(
            PeerRank(
                metric=field_name,
                subject_value=subject_pair[1],
                subject_rank=subject_rank,
                total_peers=n,
                percentile=percentile,
                best_ticker=tickers_ranked[0],
                worst_ticker=tickers_ranked[-1],
            )
        )

    return rankings


def _verdict(percentile: float) -> str:
    if percentile >= 75:
        return "top quartile"
    if percentile >= 50:
        return "above average"
    if percentile >= 25:
        return "below average"
    return "bottom quartile"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def get_peer_comparison(
    ticker: str,
    custom_peers: list[str] | None = None,
    max_peers: int = 7,
) -> PeerComparisonResult:
    """Auto-discover peers and build comprehensive comparison table with rankings.

    Args:
        ticker:       The subject company ticker symbol.
        custom_peers: If provided, use these peer tickers instead of auto-discovery.
        max_peers:    Maximum number of peers to include (low-data peers pruned first).

    Returns:
        PeerComparisonResult with metrics table, per-metric rankings, and verdict.
    """
    ticker_upper = ticker.upper()
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Fetch subject info to determine sector/industry
    # ------------------------------------------------------------------
    try:
        subject_info_raw, subject_ret = await _fetch_ticker_data(ticker_upper)
    except Exception as exc:
        logger.error("Failed to fetch subject data", ticker=ticker_upper, error=str(exc))
        subject_info_raw, subject_ret = {}, None
        warnings.append(f"Subject data fetch failed: {exc}")

    if subject_info_raw.get("_error"):
        warnings.append(f"Subject yfinance error: {subject_info_raw['_error']}")

    sector: Optional[str] = subject_info_raw.get("sector")
    industry: Optional[str] = subject_info_raw.get("industry")

    logger.info(
        "Peer comparison started",
        ticker=ticker_upper,
        sector=sector,
        industry=industry,
    )

    # ------------------------------------------------------------------
    # Step 2: Build peer list
    # ------------------------------------------------------------------
    if custom_peers:
        peer_candidates = [p.upper() for p in custom_peers if p.upper() != ticker_upper]
    else:
        peer_candidates = _discover_peers(ticker_upper, industry, sector)

    # ------------------------------------------------------------------
    # Step 3: Fetch peer data in parallel
    # ------------------------------------------------------------------
    fetch_results: list[tuple[dict, Optional[float]]] = await asyncio.gather(
        *[_fetch_ticker_data(p) for p in peer_candidates],
        return_exceptions=True,
    )

    # ------------------------------------------------------------------
    # Step 4: Build PeerMetrics for each peer; prune low-data tickers
    # ------------------------------------------------------------------
    def _data_count(m: PeerMetrics) -> int:
        """Count non-None metric fields (excluding ticker, company_name, market_cap)."""
        fields = [
            m.pe_ratio, m.forward_pe, m.ev_ebitda, m.ps_ratio, m.pb_ratio,
            m.revenue_growth_pct, m.gross_margin_pct, m.operating_margin_pct,
            m.net_margin_pct, m.roe_pct, m.debt_to_equity, m.dividend_yield_pct,
            m.beta, m.return_1y_pct,
        ]
        return sum(1 for f in fields if f is not None)

    peer_metrics_raw: list[PeerMetrics] = []
    for peer_ticker, result in zip(peer_candidates, fetch_results):
        if isinstance(result, Exception):
            warnings.append(f"Fetch error for {peer_ticker}: {result}")
            continue
        info, ret = result
        if info.get("_error"):
            warnings.append(f"yfinance error for {peer_ticker}: {info['_error']}")
        try:
            pm = _build_peer_metrics(peer_ticker, info, ret)
            peer_metrics_raw.append(pm)
        except Exception as exc:
            logger.error("PeerMetrics build error", ticker=peer_ticker, error=str(exc))
            warnings.append(f"Could not build metrics for {peer_ticker}: {exc}")

    # Sort peers by data completeness (desc), then keep max_peers
    peer_metrics_raw.sort(key=_data_count, reverse=True)
    peer_metrics: list[PeerMetrics] = peer_metrics_raw[:max_peers]

    if len(peer_metrics) < len(peer_metrics_raw):
        pruned = [m.ticker for m in peer_metrics_raw[max_peers:]]
        warnings.append(f"Pruned peers (max_peers={max_peers}): {pruned}")

    # ------------------------------------------------------------------
    # Step 5: Build subject PeerMetrics
    # ------------------------------------------------------------------
    try:
        subject_metrics = _build_peer_metrics(ticker_upper, subject_info_raw, subject_ret)
    except Exception as exc:
        logger.error("Subject PeerMetrics build error", ticker=ticker_upper, error=str(exc))
        warnings.append(f"Could not build subject metrics: {exc}")
        subject_metrics = PeerMetrics(ticker=ticker_upper)

    # ------------------------------------------------------------------
    # Step 6: Assemble metrics table (subject first, sorted by market cap)
    # ------------------------------------------------------------------
    all_metrics: list[PeerMetrics] = [subject_metrics] + peer_metrics
    # Sort by market cap descending, keeping subject first only if tied
    all_metrics.sort(
        key=lambda m: (m.market_cap is None, -(m.market_cap or 0))
    )

    # ------------------------------------------------------------------
    # Step 7: Compute rankings using numpy for percentile validation
    # ------------------------------------------------------------------
    rankings = _compute_rankings(ticker_upper, all_metrics)

    if rankings:
        percentiles = np.array([r.percentile for r in rankings], dtype=float)
        overall_percentile = float(np.mean(percentiles))
    else:
        overall_percentile = 50.0
        warnings.append("No rankings could be computed — insufficient peer data")

    # ------------------------------------------------------------------
    # Step 8: Assemble result
    # ------------------------------------------------------------------
    peer_tickers = [m.ticker for m in peer_metrics]

    result = PeerComparisonResult(
        subject_ticker=ticker_upper,
        sector=sector,
        industry=industry,
        peers=peer_tickers,
        metrics_table=all_metrics,
        rankings=rankings,
        overall_percentile=round(overall_percentile, 2),
        verdict=_verdict(overall_percentile),
        as_of=date.today().isoformat(),
        warnings=warnings,
    )

    logger.info(
        "Peer comparison complete",
        ticker=ticker_upper,
        peers=len(peer_tickers),
        metrics_ranked=len(rankings),
        overall_percentile=result.overall_percentile,
        verdict=result.verdict,
    )

    return result
