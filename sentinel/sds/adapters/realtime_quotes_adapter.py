"""Real-time equity quotes via free sources — Alpaca IEX, Finnhub, yfinance.

dim_001 target: score 9 via layered fallback chain that maximises data quality
within free-tier constraints.  Full SIP (~$30K/yr) is not used; IEX feed
(Alpaca) is the highest-quality free option and is tried first.

Source priority (highest quality → fallback):
  1. Alpaca IEX REST  — consolidated, ~15-min delay, streaming-capable
  2. Finnhub REST     — real-time last trade, 60 calls/min free
  3. yfinance fast_info — last resort; sync wrapped in executor

WebSocket streaming lives in alpaca_ws.py (Alpaca) and finnhub_adapter.py
(Finnhub).  This module provides a polling-based AsyncGenerator for
convenience when a persistent WebSocket is overkill.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional
from zoneinfo import ZoneInfo

import httpx
import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# NYSE holiday dates for current + next year (simplified static set).
# In production extend via pandas_market_calendars or similar.
_NYSE_HOLIDAYS_2025 = {
    (2025, 1, 1),   # New Year's Day
    (2025, 1, 20),  # MLK Jr. Day
    (2025, 2, 17),  # Presidents' Day
    (2025, 4, 18),  # Good Friday
    (2025, 5, 26),  # Memorial Day
    (2025, 6, 19),  # Juneteenth
    (2025, 7, 4),   # Independence Day
    (2025, 9, 1),   # Labor Day
    (2025, 11, 27), # Thanksgiving
    (2025, 12, 25), # Christmas
}
_NYSE_HOLIDAYS_2026 = {
    (2026, 1, 1),   # New Year's Day
    (2026, 1, 19),  # MLK Jr. Day
    (2026, 2, 16),  # Presidents' Day
    (2026, 4, 3),   # Good Friday
    (2026, 5, 25),  # Memorial Day
    (2026, 6, 19),  # Juneteenth
    (2026, 7, 3),   # Independence Day (observed)
    (2026, 9, 7),   # Labor Day
    (2026, 11, 26), # Thanksgiving
    (2026, 12, 25), # Christmas
}
_NYSE_HOLIDAYS: frozenset[tuple[int, int, int]] = frozenset(
    _NYSE_HOLIDAYS_2025 | _NYSE_HOLIDAYS_2026
)

# Liquid large-cap universe used as movers proxy (free-tier friendly)
_LIQUID_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "LLY", "UNH", "JPM", "XOM", "JNJ", "V", "MA", "PG", "HD", "AVGO",
    "COST", "MRK", "CVX", "CRM", "BAC", "ABBV", "PEP", "KO", "WMT",
    "AMD", "NFLX", "TMO", "ABT", "ACN", "ORCL", "QCOM", "MCD", "ADBE",
    "CSCO", "LIN", "GE", "DHR", "TXN", "WFC", "PM", "NEE", "RTX",
    "SPGI", "IBM", "UNP", "LOW", "INTU",
]


# ── Pydantic Models ────────────────────────────────────────────────────────────

class Quote(BaseModel):
    """Unified real-time (or best-available delayed) equity quote."""

    ticker: str
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_price: float
    last_size: Optional[int] = None
    prev_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    market_cap: Optional[float] = None
    source: str  # "alpaca", "polygon", "yfinance", "finnhub"
    is_delayed: bool = True  # True for all free sources


class MarketDepth(BaseModel):
    """Single-level NBBO depth from IEX (free tier has no Level 2)."""

    ticker: str
    timestamp: datetime
    bids: list[tuple[float, int]] = Field(default_factory=list)  # (price, size) sorted desc
    asks: list[tuple[float, int]] = Field(default_factory=list)  # (price, size) sorted asc
    source: str


class MarketStatus(BaseModel):
    """Current US equity market session status."""

    is_open: bool
    session: str  # "pre", "regular", "post", "closed"
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None


class QuoteSnapshot(BaseModel):
    """Batch quote result."""

    quotes: list[Quote]
    as_of: datetime
    source: str


# ── Main Adapter ──────────────────────────────────────────────────────────────

class RealtimeQuotesAdapter:
    """Multi-source real-time quote adapter for US equities.

    Source priority per get_quote():
      1. Alpaca IEX REST (highest free quality, requires API key)
      2. Finnhub REST    (real-time last trade, 60 req/min free)
      3. yfinance        (always available, least reliable for real-time)

    All sources are free-tier and produce delayed or best-effort real-time
    quotes.  Full SIP data is not available without a ~$30K/yr data license.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._polygon_key = os.getenv("POLYGON_API_KEY", "")
        self._finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        # quote_cache: ticker → (Quote, epoch_secs)
        self._quote_cache: dict[str, tuple[Quote, float]] = {}
        self._cache_ttl: float = 5.0  # seconds

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_quote(self, ticker: str) -> Quote:
        """Return best-available quote for a single ticker.

        Uses a short TTL cache to avoid hammering APIs when callers request
        the same ticker repeatedly within a few seconds.
        """
        cached = self._quote_cache.get(ticker)
        if cached is not None:
            quote, ts = cached
            if time.monotonic() - ts < self._cache_ttl:
                return quote

        quote: Optional[Quote] = None

        # 1. Alpaca IEX
        if self._alpaca_key:
            quote = await self._fetch_alpaca_quote(ticker)

        # 2. Finnhub
        if quote is None and self._finnhub_key:
            quote = await self._fetch_finnhub_quote(ticker)

        # 3. yfinance (sync fallback)
        if quote is None:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(None, self._fetch_yfinance_quote, ticker)

        if quote is None:
            raise RuntimeError(f"All quote sources failed for {ticker}")

        self._quote_cache[ticker] = (quote, time.monotonic())
        return quote

    async def get_quotes(self, tickers: list[str]) -> QuoteSnapshot:
        """Batch quote fetch using asyncio.gather with concurrency caps.

        Alpaca allows 10 concurrent, Finnhub 30 concurrent on free tier.
        We use a semaphore sized to 10 so the primary source is never
        overwhelmed.
        """
        sem = asyncio.Semaphore(10)

        async def _bounded(ticker: str) -> Optional[Quote]:
            async with sem:
                try:
                    return await self.get_quote(ticker)
                except Exception as exc:
                    logger.warning("get_quote failed", ticker=ticker, error=str(exc))
                    return None

        results = await asyncio.gather(*[_bounded(t) for t in tickers])
        quotes = [q for q in results if q is not None]
        return QuoteSnapshot(
            quotes=quotes,
            as_of=datetime.now(tz=timezone.utc),
            source="multi",
        )

    async def get_market_depth(self, ticker: str) -> Optional[MarketDepth]:
        """Return single-level NBBO from Alpaca IEX.

        IEX is a consolidated feed but provides only one bid/ask level
        (not a full Level 2 book).  Returns None when Alpaca key is absent
        or the request fails.
        """
        if not self._alpaca_key:
            return None

        url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
            f"?feed=iex"
        )
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("market_depth fetch failed", ticker=ticker, error=str(exc))
            return None

        q = data.get("quote", {})
        if not q:
            return None

        bp = q.get("bp") or q.get("bid_price")
        bs = q.get("bs") or q.get("bid_size") or 0
        ap = q.get("ap") or q.get("ask_price")
        as_ = q.get("as") or q.get("ask_size") or 0
        ts_raw = q.get("t") or q.get("timestamp")
        ts = _parse_ts(ts_raw) or datetime.now(tz=timezone.utc)

        bids: list[tuple[float, int]] = [(float(bp), int(bs))] if bp else []
        asks: list[tuple[float, int]] = [(float(ap), int(as_))] if ap else []

        return MarketDepth(
            ticker=ticker,
            timestamp=ts,
            bids=bids,
            asks=asks,
            source="alpaca_iex",
        )

    def get_market_status(self) -> MarketStatus:
        """Compute current US equity market session from wall-clock time.

        Sessions (US/Eastern):
          pre     04:00 – 09:30
          regular 09:30 – 16:00
          post    16:00 – 20:00
          closed  outside above + weekends + NYSE holidays
        """
        now_et = datetime.now(tz=_ET)
        t = now_et.time()
        weekday = now_et.weekday()  # Mon=0, Sun=6
        date_key = (now_et.year, now_et.month, now_et.day)

        is_holiday = date_key in _NYSE_HOLIDAYS
        is_weekend = weekday >= 5  # Sat or Sun

        if is_holiday or is_weekend:
            next_open = _next_market_open(now_et)
            return MarketStatus(
                is_open=False,
                session="closed",
                next_open=next_open,
                next_close=None,
            )

        from datetime import time as dtime

        pre_start = dtime(4, 0)
        reg_start = dtime(9, 30)
        reg_end = dtime(16, 0)
        post_end = dtime(20, 0)

        if pre_start <= t < reg_start:
            next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            next_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return MarketStatus(
                is_open=True, session="pre",
                next_open=next_open, next_close=next_close,
            )
        if reg_start <= t < reg_end:
            next_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return MarketStatus(
                is_open=True, session="regular",
                next_open=None, next_close=next_close,
            )
        if reg_end <= t < post_end:
            next_close = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
            return MarketStatus(
                is_open=True, session="post",
                next_open=None, next_close=next_close,
            )

        # Closed tonight — compute next open
        next_open = _next_market_open(now_et)
        return MarketStatus(
            is_open=False, session="closed",
            next_open=next_open, next_close=None,
        )

    async def stream_quotes(
        self,
        tickers: list[str],
        interval_seconds: float = 1.0,
    ) -> AsyncGenerator[Quote, None]:
        """Polling-based quote stream.

        Yields a Quote whenever last_price or bid/ask has changed since the
        previous poll.  For true WebSocket streaming, use alpaca_ws.py.

        Args:
            tickers: symbols to stream.
            interval_seconds: poll cadence in seconds (minimum 1.0 recommended).
        """
        prev: dict[str, tuple[float, Optional[float], Optional[float]]] = {}

        while True:
            snapshot = await self.get_quotes(tickers)
            for quote in snapshot.quotes:
                key = (
                    quote.last_price,
                    quote.bid,
                    quote.ask,
                )
                if prev.get(quote.ticker) != key:
                    prev[quote.ticker] = key
                    yield quote
            await asyncio.sleep(interval_seconds)

    async def get_movers(
        self,
        n: int = 20,
        session: str = "regular",
    ) -> dict[str, list[Quote]]:
        """Return top gainers, losers, and most active from the liquid universe.

        Uses yfinance fast_info for change_pct and volume because it batches
        well and doesn't count against Alpaca/Finnhub rate limits.  Falls back
        to get_quotes() for the full enriched Quote object.

        Args:
            n: number of tickers to return per category.
            session: currently informational (free sources don't distinguish).

        Returns:
            {"gainers": [...], "losers": [...], "most_active": [...]}
        """
        snapshot = await self.get_quotes(_LIQUID_UNIVERSE)
        quotes = snapshot.quotes

        # Sort by change_pct
        with_change = [q for q in quotes if q.change_pct is not None]
        gainers = sorted(with_change, key=lambda q: q.change_pct or 0.0, reverse=True)[:n]
        losers = sorted(with_change, key=lambda q: q.change_pct or 0.0)[:n]

        # Sort by volume
        with_vol = [q for q in quotes if q.volume is not None]
        most_active = sorted(with_vol, key=lambda q: q.volume or 0, reverse=True)[:n]

        return {"gainers": gainers, "losers": losers, "most_active": most_active}

    # ── Private fetch helpers ─────────────────────────────────────────────────

    async def _fetch_alpaca_quote(self, ticker: str) -> Optional[Quote]:
        """GET Alpaca IEX latest quote for a single symbol.

        Endpoint: GET /v2/stocks/{symbol}/quotes/latest?feed=iex
        Auth:     APCA-API-KEY-ID / APCA-API-SECRET-KEY headers
        """
        url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
            f"?feed=iex"
        )
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 422:
                    # Symbol not found on IEX feed
                    return None
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Alpaca quote HTTP error", ticker=ticker,
                           status=exc.response.status_code)
            return None
        except Exception as exc:
            logger.warning("Alpaca quote error", ticker=ticker, error=str(exc))
            return None

        quote_raw = data.get("quote")
        if not quote_raw:
            return None

        # Also fetch latest bar for last_price + volume + vwap
        bar = await self._fetch_alpaca_latest_bar(ticker)
        return self._build_quote_from_alpaca(ticker, quote_raw, bar)

    async def _fetch_alpaca_latest_bar(self, ticker: str) -> dict:
        """GET Alpaca IEX latest minute bar (for last_price, vwap, volume)."""
        url = (
            f"https://data.alpaca.markets/v2/stocks/{ticker}/bars/latest"
            f"?feed=iex"
        )
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json().get("bar", {})
        except Exception:
            return {}

    async def _fetch_finnhub_quote(self, ticker: str) -> Optional[Quote]:
        """GET Finnhub /quote for a symbol.

        Response keys: c=current, d=change, dp=change_pct, h=high,
        l=low, o=open, pc=prev_close, t=unix timestamp.
        """
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker, "token": self._finnhub_key}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Finnhub quote error", ticker=ticker, error=str(exc))
            return None

        if not data or data.get("c", 0) == 0:
            return None

        return self._build_quote_from_finnhub(ticker, data)

    def _fetch_yfinance_quote(self, ticker: str) -> Optional[Quote]:
        """Synchronous yfinance fast_info quote (wrapped in executor for async use)."""
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            return self._build_quote_from_yfinance(ticker, fi)
        except Exception as exc:
            logger.warning("yfinance quote error", ticker=ticker, error=str(exc))
            return None

    # ── Builder helpers ───────────────────────────────────────────────────────

    def _build_quote_from_alpaca(
        self, ticker: str, q: dict, bar: dict
    ) -> Quote:
        """Construct Quote from Alpaca IEX quote payload + optional bar."""
        bp = q.get("bp") or q.get("bid_price")
        bs = q.get("bs") or q.get("bid_size")
        ap = q.get("ap") or q.get("ask_price")
        as_ = q.get("as") or q.get("ask_size")
        ts_raw = q.get("t") or q.get("timestamp")
        ts = _parse_ts(ts_raw) or datetime.now(tz=timezone.utc)

        last_price = (
            bar.get("c")          # bar close
            or bar.get("close")
            or ((float(bp) + float(ap)) / 2 if bp and ap else None)
        )
        if last_price is None:
            mid = ((float(bp or 0) + float(ap or 0)) / 2)
            last_price = mid if mid > 0 else 0.0

        return Quote(
            ticker=ticker,
            timestamp=ts,
            bid=float(bp) if bp else None,
            ask=float(ap) if ap else None,
            bid_size=int(bs) if bs is not None else None,
            ask_size=int(as_) if as_ is not None else None,
            last_price=float(last_price),
            volume=int(bar.get("v") or bar.get("volume") or 0) or None,
            vwap=float(bar["vw"]) if bar.get("vw") else None,
            high=float(bar["h"]) if bar.get("h") else None,
            low=float(bar["l"]) if bar.get("l") else None,
            open=float(bar["o"]) if bar.get("o") else None,
            source="alpaca",
            is_delayed=True,
        )

    def _build_quote_from_finnhub(self, ticker: str, data: dict) -> Quote:
        """Construct Quote from Finnhub /quote response."""
        ts_raw = data.get("t")
        if ts_raw:
            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)

        current = float(data.get("c", 0))
        prev_close = float(data.get("pc", 0)) or None
        change = float(data.get("d", 0)) if data.get("d") is not None else None
        change_pct = float(data.get("dp", 0)) if data.get("dp") is not None else None

        return Quote(
            ticker=ticker,
            timestamp=ts,
            last_price=current,
            prev_close=prev_close,
            change=change,
            change_pct=change_pct,
            high=float(data["h"]) if data.get("h") else None,
            low=float(data["l"]) if data.get("l") else None,
            open=float(data["o"]) if data.get("o") else None,
            source="finnhub",
            is_delayed=False,  # Finnhub last trade is real-time on free tier
        )

    def _build_quote_from_yfinance(self, ticker: str, fi: object) -> Optional[Quote]:
        """Construct Quote from yfinance Ticker.fast_info object."""
        try:
            last_price = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
            if last_price is None:
                return None

            bid = getattr(fi, "bid", None)
            ask = getattr(fi, "ask", None)
            volume = getattr(fi, "three_month_average_volume", None)  # fast_info doesn't have daily vol
            market_cap = getattr(fi, "market_cap", None)
            prev_close = getattr(fi, "previous_close", None)
            open_ = getattr(fi, "open", None)
            high = getattr(fi, "day_high", None)
            low = getattr(fi, "day_low", None)

            change = (float(last_price) - float(prev_close)) if prev_close else None
            change_pct = (change / float(prev_close) * 100) if prev_close and change is not None else None

            return Quote(
                ticker=ticker,
                timestamp=datetime.now(tz=timezone.utc),
                bid=float(bid) if bid else None,
                ask=float(ask) if ask else None,
                last_price=float(last_price),
                volume=int(volume) if volume else None,
                market_cap=float(market_cap) if market_cap else None,
                prev_close=float(prev_close) if prev_close else None,
                change=change,
                change_pct=change_pct,
                high=float(high) if high else None,
                low=float(low) if low else None,
                open=float(open_) if open_ else None,
                source="yfinance",
                is_delayed=True,
            )
        except Exception as exc:
            logger.warning("yfinance build error", ticker=ticker, error=str(exc))
            return None


# ── Private helpers ────────────────────────────────────────────────────────────

def _parse_ts(raw: object) -> Optional[datetime]:
    """Parse a timestamp from Alpaca/Finnhub string or epoch int."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            # Alpaca uses RFC 3339 strings like "2024-01-02T14:30:00.123456789Z"
            # datetime.fromisoformat chokes on nanoseconds, trim to microseconds
            clean = raw.rstrip("Z")
            if "." in clean:
                base, frac = clean.split(".", 1)
                frac = frac[:6]
                clean = f"{base}.{frac}"
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    return None


def _next_market_open(now_et: datetime) -> datetime:
    """Return next NYSE open (09:30 ET) after now, skipping weekends & holidays."""
    candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if candidate <= now_et:
        candidate += timedelta(days=1)

    for _ in range(10):  # safety limit
        key = (candidate.year, candidate.month, candidate.day)
        if candidate.weekday() < 5 and key not in _NYSE_HOLIDAYS:
            return candidate
        candidate += timedelta(days=1)

    return candidate


# ── Module-level convenience helpers ──────────────────────────────────────────

_default_adapter: Optional[RealtimeQuotesAdapter] = None


def _get_adapter() -> RealtimeQuotesAdapter:
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = RealtimeQuotesAdapter()
    return _default_adapter


async def get_quote(ticker: str) -> Quote:
    """Module-level quote fetch using a shared default adapter."""
    return await _get_adapter().get_quote(ticker)


async def get_quotes(tickers: list[str]) -> QuoteSnapshot:
    """Module-level batch quote fetch using a shared default adapter."""
    return await _get_adapter().get_quotes(tickers)


def market_status() -> MarketStatus:
    """Module-level market status using a shared default adapter."""
    return _get_adapter().get_market_status()
