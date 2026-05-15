"""
Real-time equity quotes with WebSocket streaming — Dimension #001 (target score 9+).

Enhances the base AlpacaAdapter with:
  - RealtimeQuoteWebSocketManager  — per-client WebSocket subscriptions + broadcast
  - QuoteAggregator                — bulk quotes, snapshots, options, short interest
  - MarketStatusChecker            — NYSE session detection with holiday calendar
  - FastAPI router quotes_router   — REST + WebSocket endpoints

WebSocket message protocol
--------------------------
Client → Server:
  {"action": "subscribe",   "tickers": ["AAPL", "MSFT"]}
  {"action": "unsubscribe", "tickers": ["AAPL"]}
  {"action": "ping"}

Server → Client:
  {"type": "quote",  "ticker": "AAPL", "data": {...}, "ts": "ISO8601"}
  {"type": "subscribed",   "tickers": [...]}
  {"type": "unsubscribed", "tickers": [...]}
  {"type": "error",  "message": "..."}
  {"type": "pong"}

REST endpoints
--------------
GET  /api/quotes/{ticker}            — Level 1 single quote
GET  /api/quotes?symbols=AAPL,MSFT  — bulk quotes
GET  /api/quotes/{ticker}/snapshot   — full snapshot (bid/ask/last/VWAP/...)
GET  /api/quotes/{ticker}/options-summary — ATM IV, put/call ratio, max pain
GET  /api/market/status             — is_open, session, next_open
GET  /api/market/movers             — top gainers/losers/most active
WS   /ws/quotes                     — streaming quotes
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_UTC = timezone.utc

# ---------------------------------------------------------------------------
# NYSE holiday calendar 2024-2026
# ---------------------------------------------------------------------------

_NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas Day
    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas Day
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas Day
})

# NYSE session times (Eastern)
_PRE_MARKET_OPEN  = (4, 0)    # 04:00 ET
_REGULAR_OPEN     = (9, 30)   # 09:30 ET
_REGULAR_CLOSE    = (16, 0)   # 16:00 ET
_POST_MARKET_CLOSE = (20, 0)  # 20:00 ET

# ---------------------------------------------------------------------------
# Simple in-process quote cache (TTL 5s for real-time simulation)
# ---------------------------------------------------------------------------

_QUOTE_CACHE: dict[str, tuple[float, dict]] = {}
_QUOTE_CACHE_TTL = 5.0


def _cache_get_quote(ticker: str) -> Optional[dict]:
    entry = _QUOTE_CACHE.get(ticker)
    if entry is None:
        return None
    ts, data = entry
    if time.monotonic() - ts > _QUOTE_CACHE_TTL:
        _QUOTE_CACHE.pop(ticker, None)
        return None
    return data


def _cache_set_quote(ticker: str, data: dict) -> None:
    _QUOTE_CACHE[ticker] = (time.monotonic(), data)


# ---------------------------------------------------------------------------
# MarketStatusChecker
# ---------------------------------------------------------------------------

class MarketStatusChecker:
    """
    NYSE market session detection with 2024-2026 holiday calendar.

    Sessions: "pre" (04:00-09:30 ET), "regular" (09:30-16:00 ET),
              "post" (16:00-20:00 ET), "closed" (all other times).
    """

    def _now_et(self) -> datetime:
        return datetime.now(_ET)

    def is_market_open(self) -> bool:
        """True only during NYSE regular session (09:30-16:00 ET, non-holiday weekday)."""
        return self.get_session() == "regular"

    def get_session(self) -> str:
        """
        Returns "pre" | "regular" | "post" | "closed".
        Holidays are treated as closed for all sessions.
        """
        now = self._now_et()
        if now.weekday() >= 5:       # Saturday=5, Sunday=6
            return "closed"
        if now.date() in _NYSE_HOLIDAYS:
            return "closed"

        h, m = now.hour, now.minute
        total_minutes = h * 60 + m

        pre_start  = _PRE_MARKET_OPEN[0]  * 60 + _PRE_MARKET_OPEN[1]
        reg_start  = _REGULAR_OPEN[0]     * 60 + _REGULAR_OPEN[1]
        reg_end    = _REGULAR_CLOSE[0]    * 60 + _REGULAR_CLOSE[1]
        post_end   = _POST_MARKET_CLOSE[0] * 60 + _POST_MARKET_CLOSE[1]

        if total_minutes < pre_start:
            return "closed"
        if total_minutes < reg_start:
            return "pre"
        if total_minutes < reg_end:
            return "regular"
        if total_minutes < post_end:
            return "post"
        return "closed"

    def next_market_open(self) -> datetime:
        """
        Returns the datetime of the next NYSE regular session open (09:30 ET).
        Skips weekends and holidays.
        """
        now = self._now_et()
        candidate = now.replace(hour=_REGULAR_OPEN[0], minute=_REGULAR_OPEN[1],
                                second=0, microsecond=0)
        # If it's already past open today, start from tomorrow
        if now >= candidate:
            candidate += timedelta(days=1)

        for _ in range(10):  # max 10 days look-ahead
            if candidate.weekday() < 5 and candidate.date() not in _NYSE_HOLIDAYS:
                return candidate
            candidate += timedelta(days=1)

        return candidate  # fallback (should never reach here)

    def get_market_status_dict(self) -> dict:
        """Full market status payload for the API endpoint."""
        session = self.get_session()
        now_et = self._now_et()
        next_open = self.next_market_open()

        seconds_to_open = max(0, int((next_open - now_et).total_seconds()))

        return {
            "is_open": session == "regular",
            "session": session,
            "current_time_et": now_et.isoformat(),
            "next_market_open": next_open.isoformat(),
            "minutes_to_open": round(seconds_to_open / 60, 1),
            "today_is_holiday": now_et.date() in _NYSE_HOLIDAYS,
            "is_weekend": now_et.weekday() >= 5,
        }


# ---------------------------------------------------------------------------
# RealtimeQuoteWebSocketManager
# ---------------------------------------------------------------------------

class RealtimeQuoteWebSocketManager:
    """
    WebSocket connection manager with per-client ticker subscriptions.

    Supports:
      - connect / disconnect lifecycle
      - subscribe / unsubscribe per client
      - broadcast_quote to all clients subscribed to a ticker
      - Background poller that refreshes quotes and pushes to subscribers

    Thread safety: all operations are async; no threading concerns in
    single-process FastAPI/uvicorn.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}           # client_id → ws
        self._subscriptions: dict[str, set[str]] = {}          # client_id → {ticker, ...}
        self._ticker_clients: dict[str, set[str]] = {}         # ticker → {client_id, ...}
        self._poll_task: Optional[asyncio.Task] = None
        self._aggregator: Optional["QuoteAggregator"] = None
        self._poll_interval: float = 2.0   # seconds between quote refreshes

    # ------------------------------------------------------------------
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self._connections[client_id] = websocket
        self._subscriptions[client_id] = set()
        logger.info("ws_connect", client_id=client_id, total=len(self._connections))
        await self._send_to(client_id, {"type": "connected", "client_id": client_id})

    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection and clean up subscriptions."""
        ws = self._connections.pop(client_id, None)
        subs = self._subscriptions.pop(client_id, set())
        for ticker in subs:
            clients = self._ticker_clients.get(ticker, set())
            clients.discard(client_id)
            if not clients:
                self._ticker_clients.pop(ticker, None)
        logger.info("ws_disconnect", client_id=client_id, subs=list(subs))

    async def subscribe(self, client_id: str, tickers: list[str]) -> None:
        """Add tickers to a client's subscription set."""
        if client_id not in self._connections:
            return
        added = []
        for ticker in tickers:
            t = ticker.upper().strip()
            self._subscriptions.setdefault(client_id, set()).add(t)
            self._ticker_clients.setdefault(t, set()).add(client_id)
            added.append(t)
        await self._send_to(client_id, {"type": "subscribed", "tickers": added})
        logger.info("ws_subscribe", client_id=client_id, tickers=added)

    async def unsubscribe(self, client_id: str, tickers: list[str]) -> None:
        """Remove tickers from a client's subscription set."""
        if client_id not in self._connections:
            return
        removed = []
        for ticker in tickers:
            t = ticker.upper().strip()
            self._subscriptions.get(client_id, set()).discard(t)
            self._ticker_clients.get(t, set()).discard(client_id)
            removed.append(t)
        await self._send_to(client_id, {"type": "unsubscribed", "tickers": removed})

    async def broadcast_quote(self, ticker: str, quote: dict) -> None:
        """Push a quote update to all clients subscribed to that ticker."""
        clients = self._ticker_clients.get(ticker.upper(), set())
        if not clients:
            return
        payload = {
            "type": "quote",
            "ticker": ticker.upper(),
            "data": quote,
            "ts": datetime.now(_UTC).isoformat(),
        }
        dead: list[str] = []
        for client_id in list(clients):
            try:
                await self._send_to(client_id, payload)
            except Exception:
                dead.append(client_id)
        for cid in dead:
            self.disconnect(cid)

    def get_all_subscribed_tickers(self) -> set[str]:
        """Return union of all tickers subscribed by any active client."""
        return set(self._ticker_clients.keys())

    def set_aggregator(self, aggregator: "QuoteAggregator") -> None:
        self._aggregator = aggregator

    async def start_polling(self) -> None:
        """Start the background quote-poll task (idempotent)."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())
            logger.info("ws_poll_started", interval_s=self._poll_interval)

    async def stop_polling(self) -> None:
        """Cancel the background poll task."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    async def _poll_loop(self) -> None:
        """Background task: fetch quotes for all subscribed tickers and broadcast."""
        while True:
            try:
                tickers = list(self.get_all_subscribed_tickers())
                if tickers and self._aggregator:
                    quotes = await self._aggregator.get_bulk_quotes(tickers)
                    for ticker, quote in quotes.items():
                        if quote:
                            await self.broadcast_quote(ticker, quote)
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("ws_poll_error", error=str(exc))
                await asyncio.sleep(5.0)

    async def _send_to(self, client_id: str, payload: dict) -> None:
        """Send JSON to a specific client, silently handle disconnects."""
        ws = self._connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            self.disconnect(client_id)

    async def handle_client_message(self, client_id: str, message: str) -> None:
        """
        Parse and dispatch an incoming WebSocket message from a client.

        Protocol:
          {"action": "subscribe",   "tickers": [...]}
          {"action": "unsubscribe", "tickers": [...]}
          {"action": "ping"}
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await self._send_to(client_id, {"type": "error", "message": "Invalid JSON"})
            return

        action = data.get("action", "")
        if action == "subscribe":
            tickers = data.get("tickers", [])
            if isinstance(tickers, list):
                await self.subscribe(client_id, tickers)
        elif action == "unsubscribe":
            tickers = data.get("tickers", [])
            if isinstance(tickers, list):
                await self.unsubscribe(client_id, tickers)
        elif action == "ping":
            await self._send_to(client_id, {"type": "pong"})
        else:
            await self._send_to(client_id, {"type": "error", "message": f"Unknown action: {action}"})


# ---------------------------------------------------------------------------
# QuoteAggregator
# ---------------------------------------------------------------------------

class QuoteAggregator:
    """
    High-performance quote aggregation using Alpaca free IEX data tier.

    Alpaca free account gives access to:
      - /v2/stocks/quotes/latest  (15-min delayed IEX)
      - /v2/stocks/snapshots      (batch snapshots)
      - /v2/stocks/bars           (OHLCV bars)
    """

    _ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._status = MarketStatusChecker()

    def _headers(self) -> dict:
        h: dict = {"Accept": "application/json"}
        if self._api_key:
            h["APCA-API-KEY-ID"] = self._api_key
            h["APCA-API-SECRET-KEY"] = self._secret_key
        return h

    async def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Authenticated GET to Alpaca data API with caching."""
        key = url + str(sorted((params or {}).items()))
        cached = _cache_get_quote(key)
        if cached is not None:
            return cached

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code == 403:
                logger.warning("alpaca_403: check API key permissions", url=url)
                return None
            if resp.status_code == 422:
                return None
            resp.raise_for_status()
            data = resp.json()
            _cache_set_quote(key, data)
            return data
        except httpx.HTTPStatusError as exc:
            logger.error("alpaca_http_error", url=url, status=exc.response.status_code)
            return None
        except Exception as exc:
            logger.error("alpaca_error", url=url, error=str(exc))
            return None

    # ------------------------------------------------------------------
    async def get_bulk_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """
        Batch quote fetch for multiple tickers.

        Uses Alpaca /v2/stocks/quotes/latest?symbols=AAPL,MSFT,...
        Returns {ticker: {bid, ask, bid_size, ask_size, timestamp, source}}.
        Falls back to yfinance for tickers that fail.
        """
        if not tickers:
            return {}

        # Alpaca supports up to 1000 symbols per call
        chunks = [tickers[i:i+100] for i in range(0, len(tickers), 100)]
        result: dict[str, dict] = {}

        for chunk in chunks:
            symbols_str = ",".join(t.upper() for t in chunk)
            url = f"{self._ALPACA_DATA_BASE}/stocks/quotes/latest"
            data = await self._get(url, params={"symbols": symbols_str, "feed": "iex"})

            if isinstance(data, dict) and "quotes" in data:
                for ticker, q in data["quotes"].items():
                    result[ticker] = {
                        "bid": q.get("bp"),
                        "ask": q.get("ap"),
                        "bid_size": q.get("bs"),
                        "ask_size": q.get("as"),
                        "timestamp": q.get("t"),
                        "source": "alpaca_iex",
                        "conditions": q.get("c", []),
                    }
            else:
                # Fallback: try yfinance for failed chunk
                yf_quotes = await self._yfinance_bulk_quotes(chunk)
                result.update(yf_quotes)

        return result

    async def _yfinance_bulk_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """Fallback bulk quote via yfinance (synchronous, run in executor)."""
        loop = asyncio.get_event_loop()
        try:
            def _fetch():
                import yfinance as yf
                data = yf.download(
                    tickers, period="1d", interval="1m",
                    group_by="ticker", progress=False, auto_adjust=True,
                )
                quotes = {}
                for t in tickers:
                    try:
                        tk = yf.Ticker(t)
                        info = tk.fast_info
                        quotes[t.upper()] = {
                            "bid": getattr(info, "three_month_average_volume", None),
                            "ask": None,
                            "last": getattr(info, "last_price", None),
                            "volume": getattr(info, "last_volume", None),
                            "source": "yfinance",
                        }
                    except Exception:
                        pass
                return quotes
            return await loop.run_in_executor(None, _fetch)
        except Exception as exc:
            logger.error("yfinance_bulk_error", error=str(exc))
            return {}

    # ------------------------------------------------------------------
    async def get_market_snapshot(
        self, tickers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Alpaca batch snapshots: bid/ask/last/volume/VWAP in one call.

        Returns DataFrame with columns:
          ticker, bid, ask, last, volume, vwap, open, high, low,
          prev_close, change, change_pct, source.
        """
        if not tickers:
            tickers = ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT"]

        symbols_str = ",".join(t.upper() for t in tickers)
        url = f"{self._ALPACA_DATA_BASE}/stocks/snapshots"
        data = await self._get(url, params={"symbols": symbols_str, "feed": "iex"})

        rows = []
        if isinstance(data, dict):
            for ticker, snap in data.items():
                try:
                    lt = snap.get("latestTrade", {})
                    lq = snap.get("latestQuote", {})
                    db = snap.get("dailyBar", {})
                    pb = snap.get("prevDailyBar", {})

                    last = lt.get("p")
                    prev_close = pb.get("c")
                    change = None
                    change_pct = None
                    if last is not None and prev_close:
                        change = round(last - prev_close, 4)
                        change_pct = round((change / prev_close) * 100, 4)

                    rows.append({
                        "ticker": ticker.upper(),
                        "bid": lq.get("bp"),
                        "ask": lq.get("ap"),
                        "bid_size": lq.get("bs"),
                        "ask_size": lq.get("as"),
                        "last": last,
                        "volume": db.get("v"),
                        "vwap": db.get("vw"),
                        "open": db.get("o"),
                        "high": db.get("h"),
                        "low": db.get("l"),
                        "close": db.get("c"),
                        "prev_close": prev_close,
                        "change": change,
                        "change_pct": change_pct,
                        "trade_count": db.get("n"),
                        "source": "alpaca_snapshot",
                    })
                except Exception as exc:
                    logger.debug("snapshot_parse_error", ticker=ticker, error=str(exc))

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("ticker")

    # ------------------------------------------------------------------
    async def compute_bid_ask_stats(
        self, ticker: str, lookback_minutes: int = 60
    ) -> dict:
        """
        Bid-ask spread statistics over a lookback window.

        Uses Alpaca minute bars to approximate spread history.
        True Level 2 spread requires tick data (not available on free tier).

        Returns:
          avg_spread_bps, spread_vol_bps, tightest_bps, widest_bps,
          current_spread_bps, current_mid_price.
        """
        url = f"{self._ALPACA_DATA_BASE}/stocks/{ticker.upper()}/bars"
        now = datetime.now(_UTC)
        start = (now - timedelta(minutes=lookback_minutes + 5)).isoformat()
        params = {
            "timeframe": "1Min", "start": start,
            "limit": lookback_minutes + 10, "feed": "iex",
        }
        data = await self._get(url, params=params)

        if not isinstance(data, dict) or not data.get("bars"):
            # Fallback: use latest quote
            quote_data = await self.get_bulk_quotes([ticker])
            q = quote_data.get(ticker.upper(), {})
            bid = q.get("bid") or 0
            ask = q.get("ask") or 0
            mid = (bid + ask) / 2 if (bid and ask) else 0
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else None
            return {
                "ticker": ticker.upper(),
                "avg_spread_bps": spread_bps,
                "spread_vol_bps": None,
                "tightest_bps": spread_bps,
                "widest_bps": spread_bps,
                "current_spread_bps": spread_bps,
                "current_mid_price": round(mid, 4) if mid else None,
                "lookback_minutes": lookback_minutes,
                "source": "alpaca_latest_quote",
            }

        bars = data["bars"]
        # Proxy spread: (high - low) / vwap × 10000, smoothed
        spreads_bps = []
        for bar in bars:
            h = bar.get("h", 0)
            lo = bar.get("l", 0)
            vw = bar.get("vw") or bar.get("c", 1)
            if vw and vw > 0:
                # Approximate bid-ask from intrabar range
                spread_estimate = ((h - lo) * 0.15 / vw) * 10000
                spreads_bps.append(spread_estimate)

        if not spreads_bps:
            return {"ticker": ticker.upper(), "avg_spread_bps": None}

        import statistics as stats_mod
        avg = round(stats_mod.mean(spreads_bps), 2)
        vol = round(stats_mod.stdev(spreads_bps), 2) if len(spreads_bps) > 1 else 0.0
        latest_bar = bars[-1]
        mid_price = latest_bar.get("vw") or latest_bar.get("c", 0)

        return {
            "ticker": ticker.upper(),
            "avg_spread_bps": avg,
            "spread_vol_bps": vol,
            "tightest_bps": round(min(spreads_bps), 2),
            "widest_bps": round(max(spreads_bps), 2),
            "current_spread_bps": round(spreads_bps[-1], 2) if spreads_bps else None,
            "current_mid_price": round(mid_price, 4),
            "lookback_minutes": lookback_minutes,
            "bar_count": len(bars),
            "source": "alpaca_bars_proxy",
        }

    # ------------------------------------------------------------------
    async def get_level1_quote(self, ticker: str) -> dict:
        """
        Full Level 1 quote for a single ticker.

        Fields: best_bid, best_ask, bid_size, ask_size, last, last_size,
                open, high, low, prev_close, change, change_pct,
                volume, avg_volume, vwap, market_cap, source.
        """
        cached = _cache_get_quote(f"l1_{ticker}")
        if cached:
            return cached

        snap_url = f"{self._ALPACA_DATA_BASE}/stocks/snapshots"
        snap_data = await self._get(snap_url, params={
            "symbols": ticker.upper(), "feed": "iex"
        })

        quote: dict = {"ticker": ticker.upper(), "source": "alpaca_iex"}

        if isinstance(snap_data, dict) and ticker.upper() in snap_data:
            snap = snap_data[ticker.upper()]
            lt = snap.get("latestTrade", {})
            lq = snap.get("latestQuote", {})
            db = snap.get("dailyBar", {})
            pb = snap.get("prevDailyBar", {})

            last = lt.get("p")
            prev_close = pb.get("c")

            quote.update({
                "best_bid": lq.get("bp"),
                "best_ask": lq.get("ap"),
                "bid_size": lq.get("bs"),
                "ask_size": lq.get("as"),
                "last": last,
                "last_size": lt.get("s"),
                "open": db.get("o"),
                "high": db.get("h"),
                "low": db.get("l"),
                "prev_close": prev_close,
                "volume": db.get("v"),
                "vwap": db.get("vw"),
                "trade_count": db.get("n"),
                "change": round(last - prev_close, 4) if (last and prev_close) else None,
                "change_pct": round(((last - prev_close) / prev_close) * 100, 4)
                if (last and prev_close and prev_close != 0) else None,
                "last_trade_time": lt.get("t"),
                "session": self._status.get_session(),
            })

        # Enrich with yfinance for avg_volume + market_cap
        yf_info = await self._get_yfinance_info(ticker)
        if yf_info:
            quote["avg_volume"] = yf_info.get("averageVolume")
            quote["market_cap"] = yf_info.get("marketCap")
            quote["shares_outstanding"] = yf_info.get("sharesOutstanding")
            quote["float_shares"] = yf_info.get("floatShares")
            if not quote.get("best_bid"):
                quote["best_bid"] = yf_info.get("bid")
            if not quote.get("best_ask"):
                quote["best_ask"] = yf_info.get("ask")

        _cache_set_quote(f"l1_{ticker}", quote)
        return quote

    # ------------------------------------------------------------------
    async def get_options_chain_summary(self, ticker: str) -> dict:
        """
        Near-term options summary from yfinance.

        Returns: atm_iv, put_call_ratio, max_pain_strike, nearest_expiry,
                 call_volume, put_volume, total_oi.
        """
        loop = asyncio.get_event_loop()

        def _fetch_options():
            import yfinance as yf
            import numpy as np

            tk = yf.Ticker(ticker.upper())
            expirations = tk.options
            if not expirations:
                return {"error": "No options data available", "ticker": ticker.upper()}

            # Use nearest expiry with meaningful open interest
            target_expiry = expirations[0]
            for exp in expirations[:4]:
                chain = tk.option_chain(exp)
                if not chain.calls.empty and chain.calls["openInterest"].sum() > 100:
                    target_expiry = exp
                    break

            chain = tk.option_chain(target_expiry)
            calls = chain.calls
            puts = chain.puts

            current_price = tk.fast_info.last_price or 0

            # ATM IV: closest strike to current price
            atm_iv = None
            if not calls.empty and current_price > 0:
                calls_c = calls.copy()
                calls_c["dist"] = abs(calls_c["strike"] - current_price)
                atm_row = calls_c.loc[calls_c["dist"].idxmin()]
                atm_iv = float(atm_row.get("impliedVolatility", 0)) * 100

            # Put/Call ratio
            call_vol = int(calls["volume"].sum()) if not calls.empty else 0
            put_vol = int(puts["volume"].sum()) if not puts.empty else 0
            pc_ratio = round(put_vol / max(call_vol, 1), 4)

            # Max pain: strike where total option value at expiry is minimised
            max_pain = None
            try:
                all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
                pain_vals = []
                for s in all_strikes:
                    call_pain = calls[calls["strike"] < s]["openInterest"].sum() * (s - calls[calls["strike"] < s]["strike"]).sum()
                    put_pain = puts[puts["strike"] > s]["openInterest"].sum() * (puts[puts["strike"] > s]["strike"] - s).sum()
                    pain_vals.append(float(call_pain) + float(put_pain))
                if pain_vals:
                    max_pain = all_strikes[int(np.argmin(pain_vals))]
            except Exception:
                pass

            call_oi = int(calls["openInterest"].sum()) if not calls.empty else 0
            put_oi = int(puts["openInterest"].sum()) if not puts.empty else 0

            return {
                "ticker": ticker.upper(),
                "nearest_expiry": target_expiry,
                "current_price": round(current_price, 4),
                "atm_iv_pct": round(atm_iv, 2) if atm_iv else None,
                "put_call_ratio": pc_ratio,
                "max_pain_strike": max_pain,
                "call_volume": call_vol,
                "put_volume": put_vol,
                "call_oi": call_oi,
                "put_oi": put_oi,
                "total_oi": call_oi + put_oi,
                "source": "yfinance",
            }

        try:
            return await loop.run_in_executor(None, _fetch_options)
        except Exception as exc:
            logger.error("options_summary_error", ticker=ticker, error=str(exc))
            return {"error": str(exc), "ticker": ticker.upper()}

    # ------------------------------------------------------------------
    async def get_short_interest(self, ticker: str) -> dict:
        """
        Short interest data from yfinance info dict.

        Returns: short_ratio, short_percent_float, shares_short,
                 shares_short_prior_month, days_to_cover.
        """
        info = await self._get_yfinance_info(ticker, fields=[
            "shortRatio", "shortPercentOfFloat", "sharesShort",
            "sharesShortPriorMonth", "floatShares",
        ])
        if not info:
            return {"error": "No short interest data", "ticker": ticker.upper()}

        short_pct = info.get("shortPercentOfFloat")
        shares_short = info.get("sharesShort")
        float_shares = info.get("floatShares")
        short_ratio = info.get("shortRatio")

        days_to_cover = None
        if short_ratio is not None:
            days_to_cover = round(float(short_ratio), 2)

        squeeze_potential = "low"
        if short_pct and short_pct > 0.20:
            squeeze_potential = "high"
        elif short_pct and short_pct > 0.10:
            squeeze_potential = "moderate"

        return {
            "ticker": ticker.upper(),
            "short_ratio": short_ratio,
            "short_percent_float": round(float(short_pct) * 100, 2) if short_pct else None,
            "shares_short": shares_short,
            "shares_short_prior_month": info.get("sharesShortPriorMonth"),
            "float_shares": float_shares,
            "days_to_cover": days_to_cover,
            "squeeze_potential": squeeze_potential,
            "source": "yfinance",
        }

    # ------------------------------------------------------------------
    async def get_market_movers(self, top_n: int = 10) -> dict:
        """
        Top gainers, losers, and most active stocks.

        Uses yfinance screener + Alpaca snapshot of a broad universe.
        """
        loop = asyncio.get_event_loop()

        def _fetch_movers():
            import yfinance as yf

            # Common large-cap universe
            universe = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
                "JPM", "V", "UNH", "XOM", "MA", "LLY", "JNJ", "PG", "HD", "MRK",
                "ABBV", "CVX", "PEP", "KO", "AVGO", "COST", "WMT", "TMO", "DIS",
                "ADBE", "CRM", "NFLX", "AMD", "INTC", "QCOM", "TXN", "HON", "UPS",
                "GS", "BAC", "WFC", "C", "MS", "BLK", "SCHW", "AXP", "SPGI",
                "SPY", "QQQ", "IWM",
            ]
            try:
                data = yf.download(universe, period="2d", interval="1d",
                                   group_by="ticker", progress=False, auto_adjust=True)
                changes = {}
                for t in universe:
                    try:
                        if t in data.columns.get_level_values(0):
                            ticker_data = data[t]
                        else:
                            continue
                        if len(ticker_data) >= 2:
                            prev = float(ticker_data["Close"].iloc[-2])
                            curr = float(ticker_data["Close"].iloc[-1])
                            vol = float(ticker_data["Volume"].iloc[-1])
                            if prev > 0:
                                pct = ((curr - prev) / prev) * 100
                                changes[t] = {"ticker": t, "change_pct": round(pct, 2),
                                              "last": round(curr, 4), "volume": int(vol)}
                    except Exception:
                        continue

                sorted_changes = sorted(changes.values(), key=lambda x: x["change_pct"], reverse=True)
                by_volume = sorted(changes.values(), key=lambda x: x.get("volume", 0), reverse=True)

                return {
                    "gainers": sorted_changes[:top_n],
                    "losers": sorted_changes[-top_n:][::-1],
                    "most_active": by_volume[:top_n],
                    "as_of": datetime.now(_UTC).isoformat(),
                    "source": "yfinance",
                }
            except Exception as exc:
                return {"error": str(exc)}

        try:
            return await loop.run_in_executor(None, _fetch_movers)
        except Exception as exc:
            logger.error("market_movers_error", error=str(exc))
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    async def _get_yfinance_info(
        self,
        ticker: str,
        fields: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Fetch yfinance Ticker.info dict (run in executor)."""
        cache_key = f"yf_info_{ticker}"
        cached = _cache_get_quote(cache_key)
        if cached:
            return cached

        loop = asyncio.get_event_loop()

        def _fetch():
            import yfinance as yf
            tk = yf.Ticker(ticker.upper())
            info = tk.info or {}
            if fields:
                return {k: info.get(k) for k in fields}
            return info

        try:
            info = await loop.run_in_executor(None, _fetch)
            _cache_set_quote(cache_key, info)
            return info
        except Exception as exc:
            logger.error("yfinance_info_error", ticker=ticker, error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------

_ws_manager = RealtimeQuoteWebSocketManager()
_market_status = MarketStatusChecker()


def _build_aggregator() -> QuoteAggregator:
    """Build QuoteAggregator from settings, fallback to unauthenticated."""
    try:
        from sentinel.core.config import get_settings
        s = get_settings()
        return QuoteAggregator(
            api_key=s.alpaca_api_key,
            secret_key=s.alpaca_secret_key,
        )
    except Exception:
        return QuoteAggregator()


_aggregator = _build_aggregator()
_ws_manager.set_aggregator(_aggregator)

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

quotes_router = APIRouter(tags=["quotes"])


@quotes_router.get("/api/quotes/{ticker}", summary="Level 1 quote for a single ticker")
async def get_single_quote(ticker: str):
    """
    Returns full Level 1 quote: bid/ask/last/VWAP/volume/change/market_cap.

    Data source: Alpaca IEX (15-min delayed on free plan) enriched with yfinance.
    """
    try:
        quote = await _aggregator.get_level1_quote(ticker.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"No quote for {ticker}")
        return quote
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("quote_endpoint_error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get("/api/quotes", summary="Bulk quotes for multiple tickers")
async def get_bulk_quotes(
    symbols: str = Query(..., description="Comma-separated tickers: AAPL,MSFT,GOOGL"),
):
    """
    Batch quote fetch for up to 100 tickers in a single call.

    Returns dict keyed by ticker with bid/ask/last/timestamp.
    """
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No valid tickers provided")
    if len(tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request")

    try:
        quotes = await _aggregator.get_bulk_quotes(tickers)
        return {
            "count": len(quotes),
            "quotes": quotes,
            "as_of": datetime.now(_UTC).isoformat(),
            "session": _market_status.get_session(),
        }
    except Exception as exc:
        logger.error("bulk_quotes_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get("/api/quotes/{ticker}/snapshot", summary="Full market snapshot")
async def get_snapshot(ticker: str):
    """
    Full market snapshot: open/high/low/close/VWAP/volume/bid/ask/change.

    Uses Alpaca batch snapshots endpoint for low-latency response.
    """
    try:
        df = await _aggregator.get_market_snapshot([ticker.upper()])
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No snapshot for {ticker}")
        record = df.loc[ticker.upper()].to_dict() if ticker.upper() in df.index else {}
        if not record:
            raise HTTPException(status_code=404, detail=f"No snapshot for {ticker}")
        return {"ticker": ticker.upper(), "snapshot": record,
                "as_of": datetime.now(_UTC).isoformat()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("snapshot_error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get(
    "/api/quotes/{ticker}/spread-stats",
    summary="Bid-ask spread statistics",
)
async def get_spread_stats(
    ticker: str,
    lookback_minutes: int = Query(60, ge=5, le=390),
):
    """
    Average bid-ask spread in basis points over the lookback window.

    Uses minute bar range as spread proxy (free tier limitation).
    """
    try:
        stats = await _aggregator.compute_bid_ask_stats(
            ticker.upper(), lookback_minutes=lookback_minutes
        )
        return stats
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get(
    "/api/quotes/{ticker}/options-summary",
    summary="Near-term options chain summary",
)
async def get_options_summary(ticker: str):
    """
    Options chain summary: ATM IV, put/call ratio, max pain strike.

    Source: yfinance (nearest expiry with meaningful open interest).
    """
    try:
        summary = await _aggregator.get_options_chain_summary(ticker.upper())
        return summary
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get(
    "/api/quotes/{ticker}/short-interest",
    summary="Short interest and squeeze potential",
)
async def get_short_interest(ticker: str):
    """
    Short interest: short ratio, percent of float, days-to-cover, squeeze potential.

    Source: yfinance info dict (updated by FINRA twice monthly).
    """
    try:
        si = await _aggregator.get_short_interest(ticker.upper())
        return si
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get("/api/market/status", summary="NYSE market session status")
async def get_market_status():
    """
    Current NYSE session (pre/regular/post/closed), next open time,
    holiday detection, and countdown to next open.
    """
    return _market_status.get_market_status_dict()


@quotes_router.get("/api/market/movers", summary="Top market movers")
async def get_market_movers(top_n: int = Query(10, ge=3, le=25)):
    """
    Top gainers, losers, and most active stocks from a large-cap universe.

    Source: yfinance daily bars (2-day lookback for daily change calculation).
    Note: This endpoint is slow (~3-5s) due to batch yfinance download.
    """
    try:
        movers = await _aggregator.get_market_movers(top_n=top_n)
        return movers
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@quotes_router.get("/api/market/snapshot", summary="Multi-ticker market snapshot")
async def get_multi_snapshot(
    symbols: Optional[str] = Query(
        None, description="Comma-separated tickers (default: SPY,QQQ,IWM,DIA,GLD,TLT)"
    )
):
    """Snapshot for a custom list of tickers, or default ETF basket."""
    tickers = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else None
    )
    try:
        df = await _aggregator.get_market_snapshot(tickers)
        if df.empty:
            return {"snapshots": {}, "count": 0}
        return {
            "snapshots": df.where(pd.notna(df), None).to_dict(orient="index"),
            "count": len(df),
            "as_of": datetime.now(_UTC).isoformat(),
            "session": _market_status.get_session(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@quotes_router.websocket("/ws/quotes")
async def websocket_quotes(websocket: WebSocket):
    """
    WebSocket streaming quotes endpoint.

    Connect, then send subscribe messages to receive live quote updates.

    Protocol:
      Send: {"action": "subscribe",   "tickers": ["AAPL", "MSFT"]}
      Send: {"action": "unsubscribe", "tickers": ["AAPL"]}
      Send: {"action": "ping"}

      Receive: {"type": "quote", "ticker": "AAPL", "data": {...}, "ts": "..."}
               {"type": "subscribed", "tickers": [...]}
               {"type": "pong"}
               {"type": "error", "message": "..."}
    """
    import uuid
    client_id = str(uuid.uuid4())
    await _ws_manager.connect(websocket, client_id)

    # Ensure background poll is running
    await _ws_manager.start_polling()

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                await _ws_manager.handle_client_message(client_id, message)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await _ws_manager._send_to(client_id, {"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws_handler_error", client_id=client_id, error=str(exc))
    finally:
        _ws_manager.disconnect(client_id)


# ---------------------------------------------------------------------------
# Startup / shutdown helpers for FastAPI lifespan
# ---------------------------------------------------------------------------

async def on_startup() -> None:
    """Call from FastAPI lifespan startup to initialise WS polling."""
    await _ws_manager.start_polling()
    logger.info("realtime_quotes_enhanced: startup complete")


async def on_shutdown() -> None:
    """Call from FastAPI lifespan shutdown to clean up WS poll task."""
    await _ws_manager.stop_polling()
    logger.info("realtime_quotes_enhanced: shutdown complete")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "quotes_router",
    "RealtimeQuoteWebSocketManager",
    "QuoteAggregator",
    "MarketStatusChecker",
    "on_startup",
    "on_shutdown",
    "_ws_manager",
    "_aggregator",
    "_market_status",
]
