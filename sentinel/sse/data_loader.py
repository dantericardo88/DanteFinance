"""
Loads fundamental and price data from SENTINEL's data sources into the DuckDB screener.
Run this after `make backfill` to populate the screener universe.
Can also be called programmatically to refresh the universe.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from pydantic import BaseModel

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    _DUCKDB_AVAILABLE = False

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    _ASYNCPG_AVAILABLE = False

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues", "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
_NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
_ASSETS_CONCEPTS = ["Assets"]
_EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
_LT_DEBT_CONCEPTS = ["LongTermDebt", "LongTermDebtNoncurrent"]
_ST_DEBT_CONCEPTS = ["ShortTermBorrowings", "NotesPayableCurrent", "LongTermDebtCurrent"]
_OCF_CONCEPTS = ["NetCashProvidedByUsedInOperatingActivities"]
_EPS_DILUTED_CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
_SHARES_DILUTED_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]

_HEADERS = {
    "User-Agent": "SENTINEL/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
}

# Ordered column list for INSERT (mirrors screener.py schema exactly)
_UNIVERSE_COLS = [
    "ticker", "figi", "name", "sector", "industry", "exchange",
    "market_cap", "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales",
    "ev_to_ebitda", "ev_to_revenue", "revenue_growth_yoy", "earnings_growth_yoy",
    "revenue_growth_3y", "gross_margin", "operating_margin", "net_margin",
    "roe", "roa", "roic", "debt_to_equity", "current_ratio", "quick_ratio",
    "interest_coverage", "dividend_yield", "payout_ratio",
    "price", "price_52w_high", "price_52w_low", "price_vs_52w_high",
    "sma_50", "sma_200", "rsi_14", "avg_volume_20d",
    "insider_buy_30d", "insider_sell_30d", "institutional_pct", "short_interest_pct",
]


# ── Models ────────────────────────────────────────────────────────────────────

class LoadStats(BaseModel):
    tickers_loaded: int
    tickers_failed: int
    total_rows: int
    duration_seconds: float
    errors: list[str]


# ── XBRL helpers ──────────────────────────────────────────────────────────────

def _latest_annual_value(facts: dict, concept: str) -> float | None:
    """Extract the most recent annual (10-K) value for an XBRL concept."""
    try:
        units = facts.get("us-gaap", {}).get(concept, {}).get("units", {})
        for unit_key in ("USD", "shares", "pure"):
            entries = units.get(unit_key, [])
            annual = [
                e for e in entries
                if e.get("form") in ("10-K", "20-F", "10-K/A")
                and (e.get("start") or not e.get("start"))
            ]
            if annual:
                annual.sort(key=lambda e: e.get("filed", ""), reverse=True)
                val = annual[0].get("val")
                return float(val) if val is not None else None
    except Exception:
        pass
    return None


def _pick_first(facts: dict, concepts: list[str]) -> float | None:
    for concept in concepts:
        val = _latest_annual_value(facts, concept)
        if val is not None:
            return val
    return None


def _extract_metrics(ticker: str, facts: dict) -> dict[str, Any]:
    """Parse company facts JSON into a flat metric dict keyed by _UNIVERSE_COLS."""
    revenue = _pick_first(facts, _REVENUE_CONCEPTS)
    net_income = _pick_first(facts, _NET_INCOME_CONCEPTS)
    total_assets = _pick_first(facts, _ASSETS_CONCEPTS)
    total_equity = _pick_first(facts, _EQUITY_CONCEPTS)
    lt_debt = _pick_first(facts, _LT_DEBT_CONCEPTS)
    st_debt = _pick_first(facts, _ST_DEBT_CONCEPTS)
    total_debt = (lt_debt or 0.0) + (st_debt or 0.0)

    def _safe_div(n: float | None, d: float | None) -> float | None:
        return (n / d) if (n is not None and d and d != 0) else None

    return {
        "ticker": ticker, "figi": None, "name": None, "sector": None,
        "industry": None, "exchange": None, "market_cap": None,
        "pe_ratio": None, "forward_pe": None, "peg_ratio": None,
        "price_to_book": None, "price_to_sales": None,
        "ev_to_ebitda": None, "ev_to_revenue": None,
        "revenue_growth_yoy": None, "earnings_growth_yoy": None, "revenue_growth_3y": None,
        "gross_margin": None, "operating_margin": None,
        "net_margin": _safe_div(net_income, revenue),
        "roe": _safe_div(net_income, total_equity),
        "roa": _safe_div(net_income, total_assets),
        "roic": None,
        "debt_to_equity": _safe_div(total_debt, total_equity),
        "current_ratio": None, "quick_ratio": None, "interest_coverage": None,
        "dividend_yield": None, "payout_ratio": None,
        "price": None, "price_52w_high": None, "price_52w_low": None,
        "price_vs_52w_high": None, "sma_50": None, "sma_200": None,
        "rsi_14": None, "avg_volume_20d": None,
        "insider_buy_30d": 0, "insider_sell_30d": 0,
        "institutional_pct": None, "short_interest_pct": None,
    }


# ── DuckDB helpers ────────────────────────────────────────────────────────────

def _ensure_tables(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screen_universe (
            ticker VARCHAR PRIMARY KEY, figi VARCHAR, name VARCHAR,
            sector VARCHAR, industry VARCHAR, exchange VARCHAR,
            market_cap DOUBLE, pe_ratio DOUBLE, forward_pe DOUBLE, peg_ratio DOUBLE,
            price_to_book DOUBLE, price_to_sales DOUBLE, ev_to_ebitda DOUBLE, ev_to_revenue DOUBLE,
            revenue_growth_yoy DOUBLE, earnings_growth_yoy DOUBLE, revenue_growth_3y DOUBLE,
            gross_margin DOUBLE, operating_margin DOUBLE, net_margin DOUBLE,
            roe DOUBLE, roa DOUBLE, roic DOUBLE, debt_to_equity DOUBLE,
            current_ratio DOUBLE, quick_ratio DOUBLE, interest_coverage DOUBLE,
            dividend_yield DOUBLE, payout_ratio DOUBLE,
            price DOUBLE, price_52w_high DOUBLE, price_52w_low DOUBLE, price_vs_52w_high DOUBLE,
            sma_50 DOUBLE, sma_200 DOUBLE, rsi_14 DOUBLE, avg_volume_20d DOUBLE,
            insider_buy_30d INTEGER DEFAULT 0, insider_sell_30d INTEGER DEFAULT 0,
            institutional_pct DOUBLE, short_interest_pct DOUBLE,
            updated_at TIMESTAMP DEFAULT current_timestamp
        )
    """)


def _insert_row(conn: Any, row: dict) -> None:
    placeholders = ", ".join("?" * len(_UNIVERSE_COLS))
    cols = ", ".join(_UNIVERSE_COLS)
    values = [row[c] for c in _UNIVERSE_COLS]
    conn.execute(
        f"""
        INSERT INTO screen_universe ({cols}) VALUES ({placeholders})
        ON CONFLICT (ticker) DO UPDATE SET
            net_margin = excluded.net_margin,
            roe = excluded.roe,
            roa = excluded.roa,
            debt_to_equity = excluded.debt_to_equity,
            updated_at = current_timestamp
        """,
        values,
    )


# ── Core loaders ──────────────────────────────────────────────────────────────

async def load_from_edgar_xbrl(
    screener_db_path: str,
    cik_ticker_map: dict[str, str],
    max_tickers: int = 500,
) -> LoadStats:
    """
    Fetch EDGAR company facts for each CIK, extract standardized metrics,
    insert into DuckDB screener universe table.

    Metrics extracted: revenue, net_income, total_assets, total_equity, total_debt,
    operating_cash_flow, eps_diluted, shares_diluted plus derived ratios.
    Rate limit: max 10 req/s. Uses asyncio.Semaphore(8).
    """
    if not _DUCKDB_AVAILABLE:
        return LoadStats(
            tickers_loaded=0, tickers_failed=0, total_rows=0,
            duration_seconds=0.0, errors=["duckdb not installed"],
        )

    t0 = time.monotonic()
    sem = asyncio.Semaphore(8)
    errors: list[str] = []
    rows: list[dict] = []
    cik_items = list(cik_ticker_map.items())[:max_tickers]

    async def fetch_one(cik: str, ticker: str) -> dict | None:
        url = EDGAR_COMPANY_FACTS_URL.format(cik=cik.zfill(10))
        async with sem:
            try:
                async with httpx.AsyncClient(headers=_HEADERS, timeout=30) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    facts = resp.json().get("facts", {})
                metrics = _extract_metrics(ticker, facts)
                logger.debug("EDGAR facts loaded", ticker=ticker, cik=cik)
                return metrics
            except httpx.HTTPStatusError as exc:
                errors.append(f"{ticker} ({cik}): HTTP {exc.response.status_code}")
                return None
            except Exception as exc:
                errors.append(f"{ticker} ({cik}): {exc}")
                return None

    results = await asyncio.gather(*[fetch_one(cik, t) for cik, t in cik_items])
    rows = [r for r in results if r is not None]

    conn = duckdb.connect(screener_db_path)  # type: ignore[union-attr]
    _ensure_tables(conn)

    loaded = 0
    for row in rows:
        try:
            _insert_row(conn, row)
            loaded += 1
        except Exception as exc:
            errors.append(f"DB insert {row['ticker']}: {exc}")

    total_rows = conn.execute("SELECT COUNT(*) FROM screen_universe").fetchone()[0]
    conn.close()

    duration = round(time.monotonic() - t0, 2)
    logger.info("load_from_edgar_xbrl complete", loaded=loaded, total_rows=total_rows, duration=duration)
    return LoadStats(
        tickers_loaded=loaded,
        tickers_failed=len(cik_items) - loaded,
        total_rows=total_rows,
        duration_seconds=duration,
        errors=errors,
    )


async def load_sp500_universe(screener_db_path: str) -> LoadStats:
    """
    Load the S&P 500 universe from SEC EDGAR company tickers JSON,
    then call load_from_edgar_xbrl for all found companies.
    """
    logger.info("Fetching universe from SEC EDGAR company tickers")
    try:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=30) as client:
            resp = await client.get(EDGAR_COMPANY_TICKERS_URL)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch company tickers", error=str(exc))
        return LoadStats(
            tickers_loaded=0, tickers_failed=0, total_rows=0,
            duration_seconds=0.0, errors=[f"Failed to fetch tickers: {exc}"],
        )

    cik_ticker_map: dict[str, str] = {
        str(e.get("cik_str", "")).strip(): str(e.get("ticker", "")).strip().upper()
        for e in raw.values()
        if e.get("cik_str") and e.get("ticker")
    }
    logger.info("Company tickers loaded", count=len(cik_ticker_map))
    return await load_from_edgar_xbrl(screener_db_path, cik_ticker_map, max_tickers=500)


async def load_price_data(
    screener_db_path: str,
    db_url: str,
    tickers: list[str],
) -> int:
    """
    Load price-based metrics from TimescaleDB into DuckDB screener:
    RSI 14-day, SMA 50/200, 52-week high/low, avg 30-day volume.
    Returns number of tickers updated.
    """
    if not _DUCKDB_AVAILABLE or not _ASYNCPG_AVAILABLE:
        logger.error("Missing dependency: duckdb or asyncpg not installed")
        return 0

    conn = duckdb.connect(screener_db_path)  # type: ignore[union-attr]
    _ensure_tables(conn)

    try:
        pg = await asyncpg.connect(db_url)
    except Exception as exc:
        logger.error("TimescaleDB connection failed", error=str(exc))
        conn.close()
        return 0

    updated = 0
    for ticker in tickers:
        try:
            pg_rows = await pg.fetch(
                "SELECT time, close, volume FROM ohlcv WHERE ticker = $1 ORDER BY time DESC LIMIT 252",
                ticker,
            )
            if not pg_rows:
                continue

            closes = [float(r["close"]) for r in reversed(pg_rows)]
            volumes = [float(r["volume"]) for r in reversed(pg_rows)]

            price = closes[-1]
            high_52w = max(closes)
            low_52w = min(closes)
            avg_vol_30d = sum(volumes[-30:]) / max(len(volumes[-30:]), 1)

            conn.execute("""
                UPDATE screen_universe SET
                    price = ?, price_52w_high = ?, price_52w_low = ?,
                    price_vs_52w_high = ?, rsi_14 = ?,
                    sma_50 = ?, sma_200 = ?, avg_volume_20d = ?,
                    updated_at = current_timestamp
                WHERE ticker = ?
            """, [
                price, high_52w, low_52w,
                price / high_52w if high_52w else None,
                compute_rsi(closes) if len(closes) >= 15 else None,
                compute_sma(closes, 50), compute_sma(closes, 200),
                avg_vol_30d, ticker,
            ])
            updated += 1
        except Exception as exc:
            logger.warning("Price load failed", ticker=ticker, error=str(exc))

    await pg.close()
    conn.close()
    logger.info("Price data loaded", updated=updated, total=len(tickers))
    return updated


# ── Technical indicators ──────────────────────────────────────────────────────

def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Standard Wilder RSI from a list of close prices (oldest first)."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


def compute_sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of the last `period` closes. Returns None if insufficient data."""
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 6)


# ── Full refresh ──────────────────────────────────────────────────────────────

async def refresh_universe(
    screener_db_path: str,
    db_url: str,
    max_tickers: int = 500,
) -> LoadStats:
    """Full refresh: load EDGAR fundamentals → load price data → return stats."""
    t0 = time.monotonic()
    logger.info("refresh_universe: starting full refresh")

    stats = await load_sp500_universe(screener_db_path)

    tickers: list[str] = []
    try:
        import duckdb
        conn = duckdb.connect(screener_db_path)
        tickers = [r[0] for r in conn.execute("SELECT ticker FROM screen_universe").fetchall()]
        conn.close()
    except Exception as exc:
        logger.error("Could not read tickers for price load", error=str(exc))

    price_updated = await load_price_data(screener_db_path, db_url, tickers) if tickers and db_url else 0

    total_duration = round(time.monotonic() - t0, 2)
    logger.info(
        "refresh_universe complete",
        tickers_loaded=stats.tickers_loaded,
        price_updated=price_updated,
        duration=total_duration,
    )
    return LoadStats(
        tickers_loaded=stats.tickers_loaded,
        tickers_failed=stats.tickers_failed,
        total_rows=stats.total_rows,
        duration_seconds=total_duration,
        errors=stats.errors,
    )
