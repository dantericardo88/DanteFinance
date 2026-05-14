"""SENTINEL Data API — Excel / Google Sheets plugin backend + TradingView UDF.

Serves data in formats consumable by:
  - Excel RTD (via PyXLL or XLL add-in calling these HTTP endpoints)
  - Google Sheets via =IMPORTDATA("http://localhost:8080/v1/price/AAPL")
    or =IMPORTDATA("http://localhost:8080/v1/ohlcv/AAPL?fmt=csv")
  - Python scripts / Jupyter notebooks
  - TradingView Lightweight Charts / Advanced Charts via the UDF protocol

Runs standalone on port 8080; separate from the main SENTINEL API (port 8000).

Usage
-----
    python -m sentinel.api.sheets_plugin
    # or
    from sentinel.api.sheets_plugin import run_server
    run_server(host="0.0.0.0", port=8080)
"""
from __future__ import annotations

import asyncio
import io
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# ── Optional SENTINEL adapters (gracefully absent in standalone mode) ─────────
try:
    from sentinel.sds.adapters.realtime_quotes_adapter import (
        RealtimeQuotesAdapter,
        Quote,
        MarketStatus,
    )
    _ADAPTER_OK = True
except ImportError:
    RealtimeQuotesAdapter = None  # type: ignore[assignment,misc]
    Quote = None  # type: ignore[assignment]
    MarketStatus = None  # type: ignore[assignment]
    _ADAPTER_OK = False

try:
    from sentinel.sds.cache.ohlcv_cache import OHLCVCache
    _CACHE_OK = True
except ImportError:
    OHLCVCache = None  # type: ignore[assignment,misc]
    _CACHE_OK = False

try:
    from sentinel.sfe.yield_curve import get_yield_curve
    _YIELD_OK = True
except ImportError:
    get_yield_curve = None  # type: ignore[assignment]
    _YIELD_OK = False

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SENTINEL Data API",
    description=(
        "REST API for Excel/Google Sheets integration and TradingView charting. "
        "Runs on port 8080 as a lightweight sidecar to the main SENTINEL API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level singletons (initialised on startup) ─────────────────────────

_quotes_adapter: Optional["RealtimeQuotesAdapter"] = None
_ohlcv_cache: Optional["OHLCVCache"] = None


@app.on_event("startup")
async def startup() -> None:
    global _quotes_adapter, _ohlcv_cache
    if _ADAPTER_OK:
        _quotes_adapter = RealtimeQuotesAdapter()
    if _CACHE_OK:
        _ohlcv_cache = OHLCVCache()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolution_to_yf_interval(resolution: str) -> str:
    """Map TradingView UDF resolution strings to yfinance interval strings.

    TradingView uses:  "1" "5" "15" "30" "60" (minutes), "D", "W", "M"
    yfinance uses:     "1m" "5m" "15m" "30m" "1h"        "1d" "1wk" "1mo"
    """
    _MAP = {
        "1":   "1m",
        "5":   "5m",
        "15":  "15m",
        "30":  "30m",
        "60":  "1h",
        "D":   "1d",
        "W":   "1wk",
        "M":   "1mo",
    }
    return _MAP.get(resolution, "1d")


def _df_to_ohlcv_dict(df: pd.DataFrame, ticker: str) -> dict:
    """Convert a yfinance OHLCV DataFrame to a plain dict with list columns."""
    if df is None or df.empty:
        return {}

    # Flatten MultiIndex columns (yfinance >= 0.2 multi-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    result: dict = {"ticker": ticker.upper()}
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            result[col.lower()] = df[col].where(pd.notna(df[col]), other=None).tolist()
    if hasattr(df.index, "to_pydatetime"):
        result["date"] = [d.strftime("%Y-%m-%d") for d in df.index]
    return result


def _df_to_csv(df: pd.DataFrame, ticker: str) -> str:
    """Return an OHLCV DataFrame as a CSV string suitable for IMPORTDATA()."""
    if df is None or df.empty:
        return "date,open,high,low,close,volume\n"

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index().copy()
    df.index.name = "date"

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Keep only the standard OHLCV columns
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue()


async def _yf_ohlcv_async(
    ticker: str,
    start: date,
    end: date,
    interval: str,
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance in a thread executor (non-blocking)."""
    loop = asyncio.get_event_loop()

    def _fetch() -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    return await loop.run_in_executor(None, _fetch)


async def _get_ohlcv(
    ticker: str,
    start: date,
    end: date,
    interval: str,
) -> pd.DataFrame:
    """Cache-first OHLCV fetch; falls back to yfinance when cache absent."""
    if _ohlcv_cache is not None:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: _ohlcv_cache.get_ohlcv(ticker, start, end, interval),
        )
        if df is not None and not df.empty:
            return df

    # Direct yfinance fallback
    return await _yf_ohlcv_async(ticker, start, end, interval)


# ─────────────────────────────────────────────────────────────────────────────
# Quote endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/quote/{ticker}",
    summary="Real-time quote (JSON / CSV / plain-text)",
    tags=["Quotes"],
)
async def get_quote(
    ticker: str,
    fmt: str = Query("json", enum=["json", "csv", "plain"]),
):
    """Return a best-available real-time (or delayed) quote.

    - **fmt=json** (default): full Quote object as JSON.
    - **fmt=csv**: single CSV row: `ticker,price,change_pct,volume\\n`
    - **fmt=plain**: just the last price followed by a newline —
      suitable for `=IMPORTDATA("http://localhost:8080/v1/quote/AAPL?fmt=plain")`.
    """
    ticker = ticker.upper()

    # Prefer the full adapter; fall back to yfinance fast_info
    if _quotes_adapter is not None:
        try:
            quote = await _quotes_adapter.get_quote(ticker)
            price = quote.last_price
            change_pct = quote.change_pct
            volume = quote.volume
            source = quote.source
            bid = quote.bid
            ask = quote.ask
            prev_close = quote.prev_close
            change = quote.change
            high = quote.high
            low = quote.low
            ts = quote.timestamp.isoformat()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Quote fetch failed: {exc}") from exc
    else:
        # Pure yfinance fallback
        loop = asyncio.get_event_loop()
        try:
            info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).fast_info)
            price = float(getattr(info, "last_price", None) or 0.0)
            prev_close = float(getattr(info, "previous_close", None) or 0.0)
            change = round(price - prev_close, 4) if prev_close else None
            change_pct = round((change / prev_close) * 100, 4) if (change and prev_close) else None
            volume = getattr(info, "three_month_average_volume", None)
            source = "yfinance"
            bid = ask = high = low = None
            ts = datetime.now(tz=timezone.utc).isoformat()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"yfinance fallback failed: {exc}") from exc

    if fmt == "plain":
        return PlainTextResponse(f"{price}\n")

    if fmt == "csv":
        cp = f"{change_pct:.2f}%" if change_pct is not None else ""
        vol = volume if volume is not None else ""
        return PlainTextResponse(
            f"ticker,price,change_pct,volume\n{ticker},{price},{cp},{vol}\n",
            media_type="text/csv",
        )

    # JSON (default)
    return {
        "ticker": ticker,
        "timestamp": ts,
        "last_price": price,
        "bid": bid,
        "ask": ask,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "high": high,
        "low": low,
        "source": source,
    }


@app.get(
    "/v1/quotes",
    summary="Batch quotes for multiple tickers",
    tags=["Quotes"],
)
async def get_quotes(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOG"),
):
    """Fetch quotes for multiple symbols in parallel.

    Returns a list of quote objects.  Tickers that fail are omitted
    rather than failing the entire request.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(ticker_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request")

    if _quotes_adapter is not None:
        try:
            snapshot = await _quotes_adapter.get_quotes(ticker_list)
            return {
                "quotes": [q.model_dump() for q in snapshot.quotes],
                "as_of": snapshot.as_of.isoformat(),
                "source": snapshot.source,
                "count": len(snapshot.quotes),
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # yfinance batch fallback
    loop = asyncio.get_event_loop()

    def _batch_yf() -> list[dict]:
        results = []
        for t in ticker_list:
            try:
                info = yf.Ticker(t).fast_info
                price = float(getattr(info, "last_price", None) or 0.0)
                prev_close = float(getattr(info, "previous_close", None) or 0.0)
                change = round(price - prev_close, 4) if prev_close else None
                change_pct = round((change / prev_close) * 100, 4) if (change and prev_close) else None
                results.append({
                    "ticker": t,
                    "last_price": price,
                    "prev_close": prev_close,
                    "change": change,
                    "change_pct": change_pct,
                    "source": "yfinance",
                })
            except Exception:
                pass
        return results

    quotes = await loop.run_in_executor(None, _batch_yf)
    return {
        "quotes": quotes,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
        "source": "yfinance",
        "count": len(quotes),
    }


@app.get(
    "/v1/price/{ticker}",
    response_class=PlainTextResponse,
    summary="Plain-text last price (Google Sheets IMPORTDATA shortcut)",
    tags=["Quotes"],
)
async def get_price(ticker: str) -> PlainTextResponse:
    """Return just the last price as plain text, one value per line.

    Ideal for Google Sheets: `=IMPORTDATA("http://localhost:8080/v1/price/AAPL")`
    """
    ticker = ticker.upper()

    if _quotes_adapter is not None:
        try:
            quote = await _quotes_adapter.get_quote(ticker)
            return PlainTextResponse(f"{quote.last_price}\n")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).fast_info)
        price = float(getattr(info, "last_price", None) or 0.0)
        return PlainTextResponse(f"{price}\n")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/ohlcv/{ticker}",
    summary="Historical OHLCV bars (JSON or CSV)",
    tags=["OHLCV"],
)
async def get_ohlcv(
    ticker: str,
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD (default: 1 year ago)"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD (default: today)"),
    interval: str = Query(
        "1d",
        enum=["1d", "1wk", "1mo", "1h", "30m", "15m", "5m", "1m"],
        description="Bar interval",
    ),
    fmt: str = Query("json", enum=["json", "csv"]),
):
    """Historical OHLCV data for a single ticker.

    - **fmt=json**: `{"ticker": "AAPL", "date": [...], "open": [...], ...}`
    - **fmt=csv**: raw CSV rows — use with `=IMPORTDATA()` in Google Sheets.

    Defaults to the last year of daily bars when start/end are omitted.
    """
    ticker = ticker.upper()

    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start) if start else end_date - timedelta(days=365)

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start must be before end")

    df = await _get_ohlcv(ticker, start_date, end_date, interval)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No OHLCV data for {ticker}")

    if fmt == "csv":
        return PlainTextResponse(_df_to_csv(df, ticker), media_type="text/csv")

    return _df_to_ohlcv_dict(df, ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental data endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/fundamentals/{ticker}",
    summary="Key ratios and fundamentals",
    tags=["Fundamentals"],
)
async def get_fundamentals(ticker: str):
    """Return key financial ratios and company fundamentals via yfinance.

    Includes P/E, P/B, EV/EBITDA, dividend yield, market cap, beta, and more.
    """
    ticker = ticker.upper()
    loop = asyncio.get_event_loop()

    def _fetch() -> dict:
        t = yf.Ticker(ticker)
        info = t.info or {}
        # Extract the most useful fields; omit nulls to keep response lean
        fields = [
            "shortName", "sector", "industry", "country",
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
            "priceToBook", "priceToSalesTrailing12Months",
            "enterpriseToEbitda", "enterpriseToRevenue",
            "trailingEps", "forwardEps", "bookValue",
            "dividendYield", "dividendRate", "payoutRatio",
            "beta", "52WeekChange", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "fiftyDayAverage", "twoHundredDayAverage",
            "returnOnEquity", "returnOnAssets",
            "operatingMargins", "profitMargins", "grossMargins",
            "revenueGrowth", "earningsGrowth",
            "totalDebt", "totalCash", "freeCashflow",
            "currentRatio", "quickRatio", "debtToEquity",
            "sharesOutstanding", "floatShares", "heldPercentInsiders",
            "heldPercentInstitutions",
            "shortRatio", "shortPercentOfFloat",
            "averageVolume", "averageVolume10days",
        ]
        result: dict = {"ticker": ticker}
        for f in fields:
            v = info.get(f)
            if v is not None:
                result[f] = v
        return result

    try:
        data = await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"yfinance error: {exc}") from exc

    if len(data) <= 1:
        raise HTTPException(status_code=404, detail=f"No fundamentals found for {ticker}")

    return data


@app.get(
    "/v1/financials/{ticker}",
    summary="Income statement / balance sheet / cash flow",
    tags=["Fundamentals"],
)
async def get_financials(
    ticker: str,
    statement: str = Query("income", enum=["income", "balance", "cashflow"]),
    period: str = Query("annual", enum=["annual", "quarterly"]),
    fmt: str = Query("json", enum=["json", "csv"]),
):
    """Return a financial statement as structured JSON or CSV.

    Suitable for pasting into Excel or loading with =IMPORTDATA().
    """
    ticker = ticker.upper()
    loop = asyncio.get_event_loop()

    def _fetch() -> pd.DataFrame:
        t = yf.Ticker(ticker)
        if statement == "income":
            return t.quarterly_income_stmt if period == "quarterly" else t.income_stmt
        if statement == "balance":
            return t.quarterly_balance_sheet if period == "quarterly" else t.balance_sheet
        return t.quarterly_cash_flow if period == "quarterly" else t.cash_flow

    try:
        df = await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"yfinance error: {exc}") from exc

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {statement} ({period}) data for {ticker}",
        )

    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    # JSON: transpose so rows=metrics, cols=dates
    df_t = df.T
    df_t.index = [str(i)[:10] for i in df_t.index]  # shorten timestamps
    return {
        "ticker": ticker,
        "statement": statement,
        "period": period,
        "dates": list(df_t.index),
        "data": df_t.where(pd.notna(df_t), other=None).to_dict(orient="index"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market data endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/market/status",
    summary="Current US equity market session",
    tags=["Market"],
)
async def market_status():
    """Return the current US equity market session and next open/close times.

    Sessions: pre (04:00–09:30 ET), regular (09:30–16:00 ET),
    post (16:00–20:00 ET), closed.
    """
    if _quotes_adapter is not None:
        status = _quotes_adapter.get_market_status()
        return status.model_dump()

    # Lightweight fallback without the adapter
    from zoneinfo import ZoneInfo
    from datetime import time as dtime

    _ET = ZoneInfo("America/New_York")
    now_et = datetime.now(tz=_ET)
    t = now_et.time()
    weekday = now_et.weekday()

    if weekday >= 5:
        session, is_open = "closed", False
    elif dtime(4, 0) <= t < dtime(9, 30):
        session, is_open = "pre", True
    elif dtime(9, 30) <= t < dtime(16, 0):
        session, is_open = "regular", True
    elif dtime(16, 0) <= t < dtime(20, 0):
        session, is_open = "post", True
    else:
        session, is_open = "closed", False

    return {
        "is_open": is_open,
        "session": session,
        "as_of": now_et.isoformat(),
    }


@app.get(
    "/v1/market/movers",
    summary="Top gainers, losers, and most active",
    tags=["Market"],
)
async def market_movers(n: int = Query(10, ge=1, le=50)):
    """Return top-N gainers, losers, and most-active stocks.

    Uses the configured quote adapter when available; otherwise falls back to
    a yfinance scan of a curated liquid-large-cap universe.
    """
    if _quotes_adapter is not None:
        try:
            movers = await _quotes_adapter.get_movers(n=n)
            return {
                "gainers": [q.model_dump() for q in movers["gainers"]],
                "losers": [q.model_dump() for q in movers["losers"]],
                "most_active": [q.model_dump() for q in movers["most_active"]],
                "as_of": datetime.now(tz=timezone.utc).isoformat(),
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # yfinance fallback — scan a modest universe
    _UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "JPM", "XOM", "JNJ", "V", "MA", "PG", "HD", "AVGO",
        "COST", "AMD", "NFLX", "BAC", "KO", "WMT", "ORCL", "QCOM",
    ]
    loop = asyncio.get_event_loop()

    def _scan() -> list[dict]:
        rows = []
        for ticker in _UNIVERSE:
            try:
                info = yf.Ticker(ticker).fast_info
                price = float(getattr(info, "last_price", None) or 0.0)
                prev = float(getattr(info, "previous_close", None) or 0.0)
                vol = getattr(info, "three_month_average_volume", None)
                change_pct = round((price - prev) / prev * 100, 2) if prev else None
                rows.append({"ticker": ticker, "last_price": price,
                             "change_pct": change_pct, "volume": vol})
            except Exception:
                pass
        return rows

    rows = await loop.run_in_executor(None, _scan)
    with_change = [r for r in rows if r.get("change_pct") is not None]
    gainers = sorted(with_change, key=lambda r: r["change_pct"], reverse=True)[:n]
    losers = sorted(with_change, key=lambda r: r["change_pct"])[:n]
    with_vol = [r for r in rows if r.get("volume") is not None]
    most_active = sorted(with_vol, key=lambda r: r["volume"], reverse=True)[:n]

    return {
        "gainers": gainers,
        "losers": losers,
        "most_active": most_active,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
        "source": "yfinance",
    }


@app.get(
    "/v1/rates/treasury",
    summary="US Treasury yield curve from FRED",
    tags=["Market"],
)
async def treasury_rates():
    """Return the current US Treasury yield curve.

    Uses the SENTINEL yield_curve module when available, otherwise fetches
    spot rates directly from the FRED public CSV API.
    """
    if _YIELD_OK and get_yield_curve is not None:
        try:
            result = await get_yield_curve()
            return result.model_dump()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yield curve error: {exc}") from exc

    # Fallback: fetch just the spot tenors directly from FRED
    _TENORS = [
        ("1M", "GS1M"), ("3M", "GS3M"), ("6M", "GS6M"),
        ("1Y", "GS1"),  ("2Y", "GS2"),  ("5Y", "GS5"),
        ("7Y", "GS7"),  ("10Y", "GS10"), ("20Y", "GS20"), ("30Y", "GS30"),
    ]
    _FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    async def _fetch_tenor(client: httpx.AsyncClient, label: str, series: str) -> dict:
        try:
            r = await client.get(_FRED_CSV, params={"id": series}, timeout=15.0)
            lines = [l for l in r.text.strip().splitlines()[1:] if l.split(",")[1] not in (".", "", "NA")]
            if lines:
                _, val = lines[-1].split(",")
                return {"tenor": label, "yield_pct": float(val)}
        except Exception:
            pass
        return {"tenor": label, "yield_pct": None}

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_fetch_tenor(client, lbl, sid) for lbl, sid in _TENORS])

    curve = [r for r in results if r["yield_pct"] is not None]
    return {
        "as_of": date.today().isoformat(),
        "curve": curve,
        "source": "FRED",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TradingView UDF (Universal Data Feed) endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/chart/config",
    summary="TradingView UDF config",
    tags=["Charting (TradingView UDF)"],
)
async def chart_config():
    """TradingView UDF config endpoint.

    Returns supported resolutions and feature flags consumed by the
    TradingView Charting Library when configured with a custom data feed.
    """
    return {
        "supported_resolutions": ["1", "5", "15", "30", "60", "D", "W", "M"],
        "supports_group_request": False,
        "supports_marks": False,
        "supports_search": True,
        "supports_timescale_marks": False,
        "exchanges": [{"value": "", "name": "All Exchanges", "desc": ""}],
        "symbols_types": [{"name": "All types", "value": ""}],
    }


@app.get(
    "/v1/chart/symbols",
    summary="TradingView UDF symbol_info",
    tags=["Charting (TradingView UDF)"],
)
async def chart_symbols(symbol: str = Query(..., description="Ticker symbol, e.g. AAPL")):
    """Return symbol metadata in TradingView UDF SymbolInfo format.

    Called by TradingView when resolving a symbol before requesting bars.
    """
    ticker = symbol.upper()
    loop = asyncio.get_event_loop()

    def _fetch_info() -> dict:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": ticker,
            "ticker": ticker,
            "description": info.get("shortName", ticker),
            "type": "stock",
            "session": "0930-1600",
            "timezone": "America/New_York",
            "exchange": info.get("exchange", ""),
            "listed_exchange": info.get("exchange", ""),
            "minmov": 1,
            "pricescale": 100,
            "has_intraday": True,
            "intraday_multipliers": ["1", "5", "15", "30", "60"],
            "has_weekly_and_monthly": True,
            "supported_resolutions": ["1", "5", "15", "30", "60", "D", "W", "M"],
            "volume_precision": 0,
            "data_status": "streaming",
            "currency_code": info.get("currency", "USD"),
        }

    try:
        return await loop.run_in_executor(None, _fetch_info)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {exc}") from exc


@app.get(
    "/v1/chart/search",
    summary="TradingView UDF symbol search",
    tags=["Charting (TradingView UDF)"],
)
async def chart_search(
    query: str = Query(..., description="Search query string"),
    limit: int = Query(10, ge=1, le=50),
    type: str = Query("", description="Symbol type filter (unused)"),
    exchange: str = Query("", description="Exchange filter (unused)"),
):
    """Return symbol search results in TradingView UDF format.

    Performs a best-effort symbol lookup using yfinance's search endpoint.
    Unrecognised queries return an empty list rather than an error.
    """
    loop = asyncio.get_event_loop()

    def _search() -> list[dict]:
        try:
            results = yf.Search(query, max_results=limit).quotes or []
            items = []
            for r in results[:limit]:
                sym = r.get("symbol", "")
                items.append({
                    "symbol": sym,
                    "full_name": sym,
                    "description": r.get("shortname", r.get("longname", sym)),
                    "exchange": r.get("exchange", ""),
                    "type": r.get("quoteType", "stock").lower(),
                })
            return items
        except Exception:
            return []

    results = await loop.run_in_executor(None, _search)
    return results


@app.get(
    "/v1/chart/{ticker}/bars",
    summary="TradingView UDF bars (OHLCV)",
    tags=["Charting (TradingView UDF)"],
)
async def chart_bars(
    ticker: str,
    resolution: str = Query(
        "D",
        description="Bar resolution: D=daily, W=weekly, M=monthly, 60=1h, 30=30m, 15=15m, 5=5m, 1=1m",
    ),
    from_ts: Optional[int] = Query(None, description="Start Unix timestamp (seconds)"),
    to_ts: Optional[int] = Query(None, description="End Unix timestamp (seconds)"),
    countback: Optional[int] = Query(None, description="Number of bars to return (overrides from_ts)"),
):
    """Return OHLCV bars in TradingView UDF format.

    Response shape::

        {
            "s": "ok",          // "ok" | "error" | "no_data"
            "t": [1710000000, ...],  // Unix timestamps (seconds)
            "o": [182.3, ...],
            "h": [185.1, ...],
            "l": [181.0, ...],
            "c": [184.5, ...],
            "v": [45000000, ...]
        }

    TradingView calls this endpoint repeatedly as the user pans/zooms the chart.
    The ``countback`` parameter is used during initial load to pre-fetch a fixed
    number of bars ending at ``to_ts``.
    """
    ticker = ticker.upper()
    interval = _resolution_to_yf_interval(resolution)

    # Determine date range from timestamps / countback
    now = datetime.now(tz=timezone.utc)
    end_dt = datetime.fromtimestamp(to_ts, tz=timezone.utc) if to_ts else now
    end_date = end_dt.date()

    if countback is not None and countback > 0:
        # Estimate how far back we need to go based on interval and countback
        if interval in ("1m",):
            delta = timedelta(minutes=countback + 120)  # buffer for market hours
        elif interval in ("5m",):
            delta = timedelta(minutes=countback * 5 + 120)
        elif interval in ("15m",):
            delta = timedelta(minutes=countback * 15 + 240)
        elif interval in ("30m",):
            delta = timedelta(minutes=countback * 30 + 480)
        elif interval in ("1h",):
            delta = timedelta(hours=countback + 48)
        elif interval == "1wk":
            delta = timedelta(weeks=countback + 4)
        elif interval == "1mo":
            delta = timedelta(days=countback * 31 + 90)
        else:  # 1d default
            delta = timedelta(days=countback + 30)
        start_date = (end_dt - delta).date()
    elif from_ts is not None:
        start_date = datetime.fromtimestamp(from_ts, tz=timezone.utc).date()
    else:
        start_date = end_date - timedelta(days=365)

    try:
        df = await _get_ohlcv(ticker, start_date, end_date, interval)
    except Exception as exc:
        return {"s": "error", "errmsg": str(exc)}

    if df is None or df.empty:
        return {"s": "no_data"}

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()

    # Apply from_ts / to_ts filter on the resulting DataFrame
    if from_ts is not None:
        from_dt = datetime.fromtimestamp(from_ts, tz=timezone.utc)
        df = df[df.index >= pd.Timestamp(from_dt).tz_localize(None)]
    if to_ts is not None:
        to_dt = datetime.fromtimestamp(to_ts, tz=timezone.utc)
        df = df[df.index <= pd.Timestamp(to_dt).tz_localize(None)]

    if countback is not None and len(df) > countback:
        df = df.iloc[-countback:]

    if df.empty:
        return {"s": "no_data"}

    def _ts(idx_val) -> int:
        """Convert DataFrame index entry to Unix timestamp (int seconds)."""
        if hasattr(idx_val, "timestamp"):
            return int(idx_val.timestamp())
        return int(pd.Timestamp(idx_val).timestamp())

    def _col(name: str) -> list:
        col = name.capitalize() if name.capitalize() in df.columns else name
        if col not in df.columns:
            return []
        return [
            round(float(v), 4) if pd.notna(v) else None
            for v in df[col]
        ]

    return {
        "s": "ok",
        "t": [_ts(i) for i in df.index],
        "o": _col("Open"),
        "h": _col("High"),
        "l": _col("Low"),
        "c": _col("Close"),
        "v": [int(v) if pd.notna(v) else 0 for v in df.get("Volume", [])],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sheets-friendly bulk / comparison endpoints
# ─────────────────────────────────────────────────────────────────────────────

# Sample universes — enough for demo / Sheets paste; real screeners use DB
_SAMPLE_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "LLY", "UNH", "JPM", "XOM", "JNJ", "V", "MA", "PG", "HD", "AVGO",
    "COST", "MRK",
]
_SAMPLE_NDX100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "QCOM", "PEP", "CSCO", "INTC", "INTU",
    "CMCSA", "TMUS", "TXN",
]


@app.get(
    "/v1/screener/csv",
    response_class=PlainTextResponse,
    summary="Universe screener as CSV (Google Sheets paste-ready)",
    tags=["Sheets"],
)
async def screener_csv(
    universe: str = Query(
        "sp500_sample",
        enum=["sp500_sample", "nasdaq100"],
        description="Pre-defined ticker universe",
    ),
    sort_by: str = Query(
        "market_cap",
        description="Sort column: market_cap | pe | price | div_yield | 52w_change",
    ),
):
    """Return key metrics for a universe of stocks as a plain CSV.

    Columns: ticker, name, price, pe, market_cap, div_yield, 52w_high, 52w_low,
             52w_change, volume

    Designed to be pasted into Google Sheets or loaded via =IMPORTDATA().
    Fetches in parallel using yfinance; expect ~5–15 s for 20 tickers.
    """
    tickers = _SAMPLE_SP500 if universe == "sp500_sample" else _SAMPLE_NDX100

    loop = asyncio.get_event_loop()

    def _row(ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info or {}
            fi = yf.Ticker(ticker).fast_info
            return {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "price": getattr(fi, "last_price", None),
                "pe": info.get("trailingPE"),
                "market_cap": info.get("marketCap"),
                "div_yield": round(info.get("dividendYield", 0) * 100, 2)
                if info.get("dividendYield") else None,
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "52w_change": round(info.get("52WeekChange", 0) * 100, 2)
                if info.get("52WeekChange") else None,
                "volume": info.get("averageVolume"),
            }
        except Exception:
            return {"ticker": ticker}

    rows = await loop.run_in_executor(None, lambda: [_row(t) for t in tickers])

    # Sort
    _sort_key_map = {
        "market_cap": "market_cap",
        "pe": "pe",
        "price": "price",
        "div_yield": "div_yield",
        "52w_change": "52w_change",
    }
    key = _sort_key_map.get(sort_by, "market_cap")
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key) or 0), reverse=True)

    # Build CSV
    cols = ["ticker", "name", "price", "pe", "market_cap", "div_yield",
            "52w_high", "52w_low", "52w_change", "volume"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    csv_body = "\n".join(lines) + "\n"

    return PlainTextResponse(csv_body, media_type="text/csv")


@app.get(
    "/v1/compare/csv",
    response_class=PlainTextResponse,
    summary="Side-by-side fundamental comparison as CSV",
    tags=["Sheets"],
)
async def compare_csv(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOG"),
):
    """Return a side-by-side fundamental comparison as CSV.

    Rows are metrics; columns are tickers.  Designed to be pasted directly
    into Google Sheets or Excel.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(ticker_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 tickers for comparison")

    loop = asyncio.get_event_loop()

    _METRICS = [
        ("name",                "shortName"),
        ("sector",              "sector"),
        ("market_cap",          "marketCap"),
        ("price",               "_price"),
        ("trailing_pe",         "trailingPE"),
        ("forward_pe",          "forwardPE"),
        ("price_to_book",       "priceToBook"),
        ("ev_ebitda",           "enterpriseToEbitda"),
        ("trailing_eps",        "trailingEps"),
        ("dividend_yield_pct",  "_divyield"),
        ("52w_high",            "fiftyTwoWeekHigh"),
        ("52w_low",             "fiftyTwoWeekLow"),
        ("beta",                "beta"),
        ("return_on_equity",    "returnOnEquity"),
        ("profit_margin",       "profitMargins"),
        ("revenue_growth",      "revenueGrowth"),
        ("debt_to_equity",      "debtToEquity"),
        ("current_ratio",       "currentRatio"),
        ("short_ratio",         "shortRatio"),
    ]

    def _fetch_all() -> dict[str, dict]:
        result = {}
        for t in ticker_list:
            try:
                info = yf.Ticker(t).info or {}
                fi = yf.Ticker(t).fast_info
                info["_price"] = getattr(fi, "last_price", None)
                dy = info.get("dividendYield")
                info["_divyield"] = round(dy * 100, 2) if dy else None
                result[t] = info
            except Exception:
                result[t] = {}
        return result

    infos = await loop.run_in_executor(None, _fetch_all)

    # Build CSV: metric, AAPL, MSFT, ...
    header = "metric," + ",".join(ticker_list)
    lines = [header]
    for label, key in _METRICS:
        vals = [str(infos.get(t, {}).get(key, "")) for t in ticker_list]
        lines.append(f"{label}," + ",".join(vals))

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Health / root
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """Liveness check."""
    return {
        "status": "ok",
        "adapter_loaded": _quotes_adapter is not None,
        "cache_loaded": _ohlcv_cache is not None,
        "yield_curve_module": _YIELD_OK,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/", tags=["Meta"])
async def root():
    """API root — links to docs and key endpoints."""
    return {
        "name": "SENTINEL Data API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "quote": "/v1/quote/{ticker}?fmt=json|csv|plain",
            "quotes_batch": "/v1/quotes?tickers=AAPL,MSFT",
            "price_plain": "/v1/price/{ticker}",
            "ohlcv": "/v1/ohlcv/{ticker}?start=YYYY-MM-DD&end=YYYY-MM-DD&interval=1d&fmt=json|csv",
            "fundamentals": "/v1/fundamentals/{ticker}",
            "financials": "/v1/financials/{ticker}?statement=income|balance|cashflow&period=annual|quarterly",
            "market_status": "/v1/market/status",
            "market_movers": "/v1/market/movers?n=10",
            "treasury_rates": "/v1/rates/treasury",
            "chart_config": "/v1/chart/config",
            "chart_symbols": "/v1/chart/symbols?symbol=AAPL",
            "chart_search": "/v1/chart/search?query=apple",
            "chart_bars": "/v1/chart/{ticker}/bars?resolution=D&from_ts=&to_ts=",
            "screener_csv": "/v1/screener/csv?universe=sp500_sample",
            "compare_csv": "/v1/compare/csv?tickers=AAPL,MSFT,GOOG",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Server runner
# ─────────────────────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    """Start the SENTINEL Data API with uvicorn.

    Args:
        host:   Bind address (default "0.0.0.0").
        port:   TCP port (default 8080; avoids clash with main API on 8000).
        reload: Enable hot-reload for development.
    """
    uvicorn.run(
        "sentinel.api.sheets_plugin:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
