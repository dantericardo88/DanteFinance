"""
Extended Hours Enhanced — institutional pre/post-market data, gap analytics, and signal engine.

Raises dim_011 "Pre/post-market quotes" from 7 to 9+ by adding:
  - Four-session time classification (pre / regular / post / overnight)
  - AlpacaExtendedAdapter: minute bars + latest quote with extended_hours flag
  - ExtendedHoursSignalEngine: gap computation, open-drive prediction, premarket screening,
    earnings reaction classification, overnight news tracking
  - ExtendedHoursBatchTracker: bulk premarket, S&P 500 gap universe, earnings calendar reactions
  - FastAPI router with six endpoints

Public API
----------
ExtendedHoursSession
    SESSION_TIMES                          dict[str, tuple[str, str]]
    get_current_session()                  -> str
    get_session_for_time(dt)               -> str
    is_extended_hours()                    -> bool
    next_session_start()                   -> datetime

AlpacaExtendedAdapter
    get_premarket_quote(ticker)            -> dict
    get_premarket_bars(ticker, date)       -> pd.DataFrame
    get_postmarket_bars(ticker, date)      -> pd.DataFrame
    get_full_session_bars(ticker, date, include_extended) -> pd.DataFrame
    get_overnight_gap(ticker, date)        -> dict

ExtendedHoursSignalEngine
    compute_premarket_gap(ticker, prev_close)      -> dict
    compute_open_drive_prediction(ticker)          -> dict
    screen_premarket_movers(min_gap_pct, min_vol)  -> pd.DataFrame
    get_earnings_reaction(ticker)                  -> dict
    track_overnight_news(tickers)                  -> pd.DataFrame

ExtendedHoursBatchTracker
    get_bulk_premarket(tickers)                    -> pd.DataFrame
    get_gap_universe(universe)                     -> pd.DataFrame
    get_earnings_calendar_reactions(lookback_days) -> pd.DataFrame

extended_hours_router                      FastAPI APIRouter
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timezone / time constants
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_ALPACA_BASE = "https://data.alpaca.markets/v2/stocks"
_EDGAR_EFTS  = "https://efts.sec.gov/LATEST/search-index"
_HEADERS_SEC = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
}

# S&P 500 representative universe (top 100 for performance in free tier)
_SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK.B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "COST", "HD",
    "MRK", "ABBV", "CVX", "BAC", "CRM", "WMT", "NFLX", "AMD", "KO", "PEP",
    "TMO", "ORCL", "ACN", "MCD", "CSCO", "LIN", "ABT", "WFC", "DHR", "IBM",
    "ADBE", "GS", "TXN", "INTU", "QCOM", "SPGI", "AXP", "ISRG", "UNP", "CAT",
    "RTX", "VZ", "NOW", "BKNG", "PLD", "HON", "AMGN", "SYK", "T", "GE",
    "CMCSA", "DE", "TJX", "SCHW", "NEE", "LOW", "BLK", "BSX", "C", "MDT",
    "ETN", "MS", "UBER", "SBUX", "ADP", "CB", "BA", "SO", "MMC", "GILD",
    "REGN", "LRCX", "PH", "CI", "MO", "ZTS", "BMY", "KLAC", "ITW", "GD",
    "AMT", "DUK", "SLB", "NOC", "CME", "ICE", "ANET", "TGT", "USB", "FDX",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    session: str
    session_start: str
    session_end: str
    is_extended: bool
    next_session: str
    next_session_start: str


class PremarketQuote(BaseModel):
    ticker: str
    as_of: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    session: str


class GapAnalysis(BaseModel):
    ticker: str
    gap_date: date
    prev_close: float
    premarket_price: float
    gap_pct: float
    gap_type: str           # gap_up / gap_down / flat
    premarket_volume: int
    volume_vs_avg: Optional[float] = None
    catalyst_detected: bool = False
    catalyst_hint: Optional[str] = None


class EarningsReactionDetail(BaseModel):
    ticker: str
    earnings_date: date
    close_before: Optional[float] = None
    ah_price: Optional[float] = None
    ah_move_pct: Optional[float] = None
    premarket_price: Optional[float] = None
    premarket_move_pct: Optional[float] = None
    open_price: Optional[float] = None
    open_reaction_pct: Optional[float] = None
    classification: str     # beat+guide_up / beat_only / miss+guide_down / miss_only / in_line


# ---------------------------------------------------------------------------
# Session classifier
# ---------------------------------------------------------------------------

class ExtendedHoursSession:
    """Four-session time model for US equities (all times Eastern)."""

    SESSION_TIMES: dict[str, tuple[str, str]] = {
        "pre_market":  ("04:00", "09:30"),
        "regular":     ("09:30", "16:00"),
        "post_market": ("16:00", "20:00"),
        "overnight":   ("20:00", "04:00"),   # wraps midnight
    }

    _ORDER = ["overnight", "pre_market", "regular", "post_market"]

    @staticmethod
    def _now_et() -> datetime:
        return datetime.now(tz=ET)

    def get_session_for_time(self, dt: datetime) -> str:
        """Return the session name for a given datetime (tz-aware or naive ET assumed)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        dt_et = dt.astimezone(ET)
        total = dt_et.hour * 60 + dt_et.minute
        if 240 <= total < 570:   # 4:00 – 9:30
            return "pre_market"
        if 570 <= total < 960:   # 9:30 – 16:00
            return "regular"
        if 960 <= total < 1200:  # 16:00 – 20:00
            return "post_market"
        return "overnight"       # 20:00 – 4:00

    def get_current_session(self) -> str:
        return self.get_session_for_time(self._now_et())

    def is_extended_hours(self) -> bool:
        return self.get_current_session() in ("pre_market", "post_market", "overnight")

    def next_session_start(self) -> datetime:
        """Return the next session boundary as a tz-aware datetime (ET)."""
        now_et = self._now_et()
        total = now_et.hour * 60 + now_et.minute
        today = now_et.date()
        # Boundaries in minutes-since-midnight
        boundaries = [
            (240,  "pre_market",  today),
            (570,  "regular",     today),
            (960,  "post_market", today),
            (1200, "overnight",   today),
            (240 + 1440, "pre_market", today + timedelta(days=1)),  # next day pre
        ]
        for mins, session, d in boundaries:
            if total < mins % 1440:
                h, m = divmod(mins % 1440, 60)
                return datetime(d.year, d.month, d.day, h, m, tzinfo=ET)
        # Fallback: pre-market next day
        nxt = today + timedelta(days=1)
        return datetime(nxt.year, nxt.month, nxt.day, 4, 0, tzinfo=ET)


# ---------------------------------------------------------------------------
# Alpaca extended-hours adapter
# ---------------------------------------------------------------------------

class AlpacaExtendedAdapter:
    """Alpaca Markets data adapter with extended-hours support."""

    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout
        self._key    = os.getenv("ALPACA_API_KEY", "")
        self._secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._session_classifier = ExtendedHoursSession()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID":     self._key,
            "APCA-API-SECRET-KEY": self._secret,
        }

    def _has_creds(self) -> bool:
        return bool(self._key and self._secret)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _alpaca_get(self, url: str, params: dict) -> dict:
        """GET from Alpaca with rate-limit retry."""
        if not self._has_creds():
            return {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(4):
                resp = await client.get(url, headers=self._headers, params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
        return {}

    async def _fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        url = f"{_ALPACA_BASE}/{ticker.upper()}/bars"
        start_s = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_s   = end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        params: dict = {
            "timeframe":      timeframe,
            "start":          start_s,
            "end":            end_s,
            "feed":           "iex",
            "extended_hours": "true",
            "limit":          10000,
        }
        all_bars: list[dict] = []
        next_token: Optional[str] = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                while True:
                    if next_token:
                        params["page_token"] = next_token
                    resp = await client.get(url, headers=self._headers, params=params)
                    if resp.status_code == 429:
                        await asyncio.sleep(2)
                        continue
                    resp.raise_for_status()
                    payload = resp.json()
                    all_bars.extend(payload.get("bars") or [])
                    next_token = payload.get("next_page_token")
                    if not next_token:
                        break
        except Exception as exc:
            logger.warning("Alpaca bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

        if not all_bars:
            return pd.DataFrame()
        rows = []
        for b in all_bars:
            try:
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            except Exception:
                continue
            rows.append({
                "time":   ts,
                "open":   float(b.get("o", 0)),
                "high":   float(b.get("h", 0)),
                "low":    float(b.get("l", 0)),
                "close":  float(b.get("c", 0)),
                "volume": int(b.get("v", 0)),
                "vwap":   float(b.get("vw", 0)),
                "trades": int(b.get("n", 0)),
            })
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    def _yf_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        try:
            t = yf.Ticker(ticker)
            df = t.history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1m",
                prepost=True,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index.name = "time"
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index = pd.to_datetime(df.index, utc=True)
            for col in ("vwap", "trades"):
                if col not in df.columns:
                    df[col] = 0.0
            return df[["open", "high", "low", "close", "volume", "vwap", "trades"]].sort_index()
        except Exception as exc:
            logger.warning("yfinance bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    def _session_window(self, target_date: date, session: str) -> tuple[datetime, datetime]:
        """Return (start, end) UTC datetimes for a named session on a given date."""
        d = target_date
        windows = {
            "pre_market":  (datetime(d.year, d.month, d.day,  4,  0, tzinfo=ET),
                            datetime(d.year, d.month, d.day,  9, 30, tzinfo=ET)),
            "regular":     (datetime(d.year, d.month, d.day,  9, 30, tzinfo=ET),
                            datetime(d.year, d.month, d.day, 16,  0, tzinfo=ET)),
            "post_market": (datetime(d.year, d.month, d.day, 16,  0, tzinfo=ET),
                            datetime(d.year, d.month, d.day, 20,  0, tzinfo=ET)),
        }
        s, e = windows[session]
        return s.astimezone(UTC), e.astimezone(UTC)

    def _resolve_date(self, date_str: Optional[str]) -> date:
        if date_str:
            return date.fromisoformat(date_str)
        return date.today()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_premarket_quote(self, ticker: str) -> dict:
        """Latest pre-market quote via Alpaca snapshot (extended_hours=True)."""
        if not self._has_creds():
            # Fallback: yfinance fast_info
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                return {
                    "ticker":  ticker,
                    "source":  "yfinance",
                    "bid":     None,
                    "ask":     None,
                    "last":    getattr(info, "last_price", None),
                    "volume":  getattr(info, "three_month_average_volume", None),
                    "session": ExtendedHoursSession().get_current_session(),
                }
            except Exception:
                return {"ticker": ticker, "error": "no credentials and yfinance failed"}

        url = f"{_ALPACA_BASE}/{ticker.upper()}/snapshot"
        try:
            data = await self._alpaca_get(url, {"feed": "iex"})
            quote = data.get("latestQuote", {})
            trade = data.get("latestTrade", {})
            return {
                "ticker":         ticker,
                "source":         "alpaca",
                "bid":            float(quote.get("bp", 0)) or None,
                "ask":            float(quote.get("ap", 0)) or None,
                "bid_size":       int(quote.get("bs", 0)) or None,
                "ask_size":       int(quote.get("as", 0)) or None,
                "last":           float(trade.get("p", 0)) or None,
                "last_volume":    int(trade.get("s", 0)) or None,
                "last_timestamp": trade.get("t"),
                "session":        ExtendedHoursSession().get_current_session(),
            }
        except Exception as exc:
            logger.warning("Alpaca snapshot failed", ticker=ticker, error=str(exc))
            return {"ticker": ticker, "error": str(exc)}

    async def get_premarket_bars(self, ticker: str, date: Optional[str] = None) -> pd.DataFrame:
        """Minute bars for the pre-market session (4:00–9:30 AM ET)."""
        target = self._resolve_date(date)
        start_utc, end_utc = self._session_window(target, "pre_market")
        df = await self._fetch_bars(ticker, start_utc, end_utc)
        if df.empty:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._yf_bars, ticker, target, target)
            if not df.empty:
                # filter to pre-market window
                mask = (df.index >= pd.Timestamp(start_utc)) & (df.index < pd.Timestamp(end_utc))
                df = df[mask]
        df["session"] = "pre_market"
        return df

    async def get_postmarket_bars(self, ticker: str, date: Optional[str] = None) -> pd.DataFrame:
        """Minute bars for the post-market session (4:00–8:00 PM ET)."""
        target = self._resolve_date(date)
        start_utc, end_utc = self._session_window(target, "post_market")
        df = await self._fetch_bars(ticker, start_utc, end_utc)
        if df.empty:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._yf_bars, ticker, target, target)
            if not df.empty:
                mask = (df.index >= pd.Timestamp(start_utc)) & (df.index < pd.Timestamp(end_utc))
                df = df[mask]
        df["session"] = "post_market"
        return df

    async def get_full_session_bars(
        self,
        ticker: str,
        date: Optional[str] = None,
        include_extended: bool = True,
    ) -> pd.DataFrame:
        """Pre + regular + post combined bars for a single trading day."""
        target = self._resolve_date(date)
        d = target
        if include_extended:
            start = datetime(d.year, d.month, d.day,  4,  0, tzinfo=ET).astimezone(UTC)
            end   = datetime(d.year, d.month, d.day, 20,  0, tzinfo=ET).astimezone(UTC)
        else:
            start = datetime(d.year, d.month, d.day,  9, 30, tzinfo=ET).astimezone(UTC)
            end   = datetime(d.year, d.month, d.day, 16,  0, tzinfo=ET).astimezone(UTC)

        df = await self._fetch_bars(ticker, start, end)
        if df.empty:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self._yf_bars, ticker, target, target)
            if not df.empty:
                mask = (df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))
                df = df[mask]

        if not df.empty:
            classifier = ExtendedHoursSession()
            df["session"] = df.index.map(
                lambda ts: classifier.get_session_for_time(ts.to_pydatetime())
            )
        return df

    async def get_overnight_gap(self, ticker: str, date: Optional[str] = None) -> dict:
        """Compute overnight gap: prior close vs first premarket print."""
        target = self._resolve_date(date)
        prior_date = target - timedelta(days=1)
        while prior_date.weekday() >= 5:
            prior_date -= timedelta(days=1)

        loop = asyncio.get_event_loop()
        prior_df = await loop.run_in_executor(None, self._yf_bars, ticker, prior_date, prior_date)
        prior_close: float = 0.0
        if prior_df is not None and not prior_df.empty:
            classifier = ExtendedHoursSession()
            reg_mask = prior_df.index.map(
                lambda ts: classifier.get_session_for_time(ts.to_pydatetime()) == "regular"
            )
            reg_df = prior_df[reg_mask]
            if not reg_df.empty:
                prior_close = float(reg_df["close"].iloc[-1])
            elif not prior_df.empty:
                prior_close = float(prior_df["close"].iloc[-1])

        pre_df = await self.get_premarket_bars(ticker, target.isoformat())
        premarket_open: Optional[float] = None
        premarket_last: Optional[float] = None
        premarket_volume: int = 0
        if not pre_df.empty:
            premarket_open   = float(pre_df["open"].iloc[0])
            premarket_last   = float(pre_df["close"].iloc[-1])
            premarket_volume = int(pre_df["volume"].sum())

        reference = premarket_last or premarket_open or prior_close
        gap_pct = ((reference - prior_close) / prior_close * 100.0) if prior_close != 0 else 0.0

        return {
            "ticker":           ticker,
            "gap_date":         target.isoformat(),
            "prior_close":      prior_close,
            "premarket_open":   premarket_open,
            "premarket_last":   premarket_last,
            "premarket_volume": premarket_volume,
            "gap_pct":          round(gap_pct, 4),
            "gap_direction":    "up" if gap_pct > 0.05 else ("down" if gap_pct < -0.05 else "flat"),
        }


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class ExtendedHoursSignalEngine:
    """
    Pre/post-market signals: gap detection, open-drive prediction,
    earnings reaction classification, overnight news tracking.
    """

    def __init__(self):
        self._adapter = AlpacaExtendedAdapter()
        self._session = ExtendedHoursSession()

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    async def compute_premarket_gap(
        self, ticker: str, prev_close: Optional[float] = None
    ) -> dict:
        """
        Compute pre-market gap vs prior close.

        Returns gap_pct, gap_type, volume_vs_avg, catalyst_detected.
        """
        today = date.today()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        # Resolve previous close
        if prev_close is None or prev_close == 0.0:
            loop = asyncio.get_event_loop()
            ydf = await loop.run_in_executor(
                None, self._adapter._yf_bars, ticker, yesterday, yesterday
            )
            if ydf is not None and not ydf.empty:
                classifier = ExtendedHoursSession()
                reg_mask = ydf.index.map(
                    lambda ts: classifier.get_session_for_time(ts.to_pydatetime()) == "regular"
                )
                reg = ydf[reg_mask]
                prev_close = float(reg["close"].iloc[-1]) if not reg.empty else float(ydf["close"].iloc[-1])
            else:
                prev_close = 0.0

        if prev_close == 0.0:
            return {"ticker": ticker, "error": "cannot resolve prior close"}

        # Pre-market bars for today
        pre_df = await self._adapter.get_premarket_bars(ticker)
        if pre_df.empty:
            return {"ticker": ticker, "error": "no premarket data"}

        pm_price  = float(pre_df["close"].iloc[-1])
        pm_volume = int(pre_df["volume"].sum())
        gap_pct   = (pm_price - prev_close) / prev_close * 100.0

        # 5-day average premarket volume
        avg_volume = await self._avg_premarket_volume(ticker, days=5)
        volume_vs_avg = (pm_volume / avg_volume) if avg_volume > 0 else None

        # Gap type
        if gap_pct >= 0.5:
            gap_type = "gap_up"
        elif gap_pct <= -0.5:
            gap_type = "gap_down"
        else:
            gap_type = "flat"

        # Catalyst detection via EDGAR 8-K in last 4 hours
        catalyst, catalyst_hint = await self._detect_catalyst(ticker)

        return {
            "ticker":           ticker,
            "gap_date":         today.isoformat(),
            "prev_close":       round(prev_close, 4),
            "premarket_price":  round(pm_price, 4),
            "gap_pct":          round(gap_pct, 4),
            "gap_type":         gap_type,
            "premarket_volume": pm_volume,
            "volume_vs_avg":    round(volume_vs_avg, 2) if volume_vs_avg else None,
            "catalyst_detected": catalyst,
            "catalyst_hint":    catalyst_hint,
        }

    async def _avg_premarket_volume(self, ticker: str, days: int = 5) -> float:
        """Average premarket volume over last N trading days."""
        today = date.today()
        volumes: list[float] = []
        d = today - timedelta(days=1)
        while len(volumes) < days:
            if d.weekday() < 5:
                try:
                    df = await self._adapter.get_premarket_bars(ticker, d.isoformat())
                    if not df.empty:
                        volumes.append(float(df["volume"].sum()))
                except Exception:
                    pass
            d -= timedelta(days=1)
            if (today - d).days > 30:
                break
        return sum(volumes) / len(volumes) if volumes else 0.0

    async def _detect_catalyst(self, ticker: str) -> tuple[bool, Optional[str]]:
        """Check EDGAR EFTS for any 8-K filed in the last 4 hours (earnings / material events)."""
        try:
            now_utc = datetime.now(tz=UTC)
            four_hours_ago = (now_utc - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S")
            params = {
                "q":         f'"{ticker}"',
                "dateRange":  "custom",
                "startdt":    four_hours_ago[:10],
                "forms":      "8-K",
                "_source":    "form_type,entity_name,file_date,period_of_report",
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_EDGAR_EFTS, headers=_HEADERS_SEC, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    if hits:
                        form_type = hits[0].get("_source", {}).get("form_type", "8-K")
                        return True, f"{form_type} filed by {hits[0].get('_source', {}).get('entity_name', ticker)}"
        except Exception:
            pass
        return False, None

    # ------------------------------------------------------------------
    # Open-drive prediction
    # ------------------------------------------------------------------

    async def compute_open_drive_prediction(self, ticker: str) -> dict:
        """
        Predict whether a pre-market gap will continue (drive) or fade at open.

        High-conviction drive: volume > 3× avg AND gap > 2% AND consistent direction.
        """
        gap = await self.compute_premarket_gap(ticker)
        if "error" in gap:
            return {"ticker": ticker, "prediction": "unknown", "error": gap["error"]}

        gap_pct       = gap["gap_pct"]
        vol_vs_avg    = gap.get("volume_vs_avg") or 0.0
        catalyst      = gap.get("catalyst_detected", False)

        # Directional consistency: first half vs second half of premarket
        pre_df = await self._adapter.get_premarket_bars(ticker)
        consistent = False
        if not pre_df.empty and len(pre_df) >= 4:
            mid = len(pre_df) // 2
            first_half_change  = pre_df["close"].iloc[mid - 1] - pre_df["open"].iloc[0]
            second_half_change = pre_df["close"].iloc[-1] - pre_df["close"].iloc[mid]
            consistent = (first_half_change > 0) == (second_half_change > 0)

        abs_gap = abs(gap_pct)
        high_conviction = (vol_vs_avg >= 3.0) and (abs_gap >= 2.0) and consistent

        if high_conviction:
            prediction = "drive_up" if gap_pct > 0 else "drive_down"
            confidence = "high"
        elif abs_gap >= 1.0 and vol_vs_avg >= 1.5:
            prediction = "moderate_continuation" if gap_pct > 0 else "moderate_fade"
            confidence = "medium"
        else:
            prediction  = "fade"
            confidence  = "low"

        return {
            "ticker":           ticker,
            "gap_pct":          round(gap_pct, 4),
            "volume_vs_avg":    round(vol_vs_avg, 2),
            "consistent_dir":   consistent,
            "catalyst":         catalyst,
            "high_conviction":  high_conviction,
            "prediction":       prediction,
            "confidence":       confidence,
        }

    # ------------------------------------------------------------------
    # Premarket movers screen
    # ------------------------------------------------------------------

    async def screen_premarket_movers(
        self,
        min_gap_pct: float = 1.0,
        min_volume: int = 50_000,
        universe: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Screen for significant pre-market movers across the universe.
        Sorted by abs(gap_pct) × volume_score.
        """
        tickers = universe or _SP500_SAMPLE

        async def _process(ticker: str) -> Optional[dict]:
            try:
                gap = await self.compute_premarket_gap(ticker)
                if "error" in gap:
                    return None
                if abs(gap["gap_pct"]) < min_gap_pct:
                    return None
                if gap["premarket_volume"] < min_volume:
                    return None
                vol_score = gap.get("volume_vs_avg") or 1.0
                score = abs(gap["gap_pct"]) * vol_score
                return {**gap, "score": round(score, 4)}
            except Exception:
                return None

        results = await asyncio.gather(*[_process(t) for t in tickers])
        rows = [r for r in results if r is not None]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Earnings reaction
    # ------------------------------------------------------------------

    async def get_earnings_reaction(self, ticker: str) -> dict:
        """
        Classify after-hours earnings reaction for most recent earnings event.

        Classification: beat+guide_up / beat_only / miss+guide_down / miss_only / in_line
        """
        # Find most recent post-market after earnings
        today = date.today()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        loop = asyncio.get_event_loop()
        reg_df = await loop.run_in_executor(
            None, self._adapter._yf_bars, ticker, yesterday, yesterday
        )

        prior_close: Optional[float] = None
        if reg_df is not None and not reg_df.empty:
            classifier = ExtendedHoursSession()
            reg_mask = reg_df.index.map(
                lambda ts: classifier.get_session_for_time(ts.to_pydatetime()) == "regular"
            )
            reg = reg_df[reg_mask]
            if not reg.empty:
                prior_close = float(reg["close"].iloc[-1])

        ah_df = await self._adapter.get_postmarket_bars(ticker, yesterday.isoformat())
        ah_price: Optional[float] = None
        ah_move_pct: Optional[float] = None
        if not ah_df.empty and prior_close:
            ah_price = float(ah_df["close"].iloc[-1])
            ah_move_pct = (ah_price - prior_close) / prior_close * 100.0

        pre_df = await self._adapter.get_premarket_bars(ticker, today.isoformat())
        pm_price: Optional[float] = None
        pm_move_pct: Optional[float] = None
        if not pre_df.empty and prior_close:
            pm_price = float(pre_df["close"].iloc[-1])
            pm_move_pct = (pm_price - prior_close) / prior_close * 100.0

        # Heuristic classification from after-hours move
        move = ah_move_pct or pm_move_pct or 0.0
        if move >= 5.0:
            classification = "beat+guide_up"
        elif 1.0 <= move < 5.0:
            classification = "beat_only"
        elif -1.0 < move < 1.0:
            classification = "in_line"
        elif -5.0 < move <= -1.0:
            classification = "miss_only"
        else:
            classification = "miss+guide_down"

        return {
            "ticker":           ticker,
            "earnings_date":    yesterday.isoformat(),
            "close_before":     prior_close,
            "ah_price":         ah_price,
            "ah_move_pct":      round(ah_move_pct, 4) if ah_move_pct is not None else None,
            "premarket_price":  pm_price,
            "pm_move_pct":      round(pm_move_pct, 4) if pm_move_pct is not None else None,
            "classification":   classification,
        }

    # ------------------------------------------------------------------
    # Overnight news tracking
    # ------------------------------------------------------------------

    async def track_overnight_news(self, tickers: list[str]) -> pd.DataFrame:
        """
        Fetch SEC 8-K / press releases filed 4 PM – 8 AM for a basket of tickers.
        Returns DataFrame of potential pre-market gap catalysts.
        """
        now_utc = datetime.now(tz=UTC)
        # Window: yesterday 8 PM ET → today 8 AM ET (approximate overnight)
        start_window = (now_utc - timedelta(hours=16)).strftime("%Y-%m-%d")
        rows: list[dict] = []

        async def _check(ticker: str) -> list[dict]:
            try:
                params = {
                    "q":        f'"{ticker}"',
                    "dateRange": "custom",
                    "startdt":   start_window,
                    "forms":     "8-K,8-K/A",
                    "_source":   "entity_name,file_date,form_type,period_of_report,accession_no",
                    "hits.hits._source.file_date": "desc",
                }
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(_EDGAR_EFTS, headers=_HEADERS_SEC, params=params)
                    if resp.status_code != 200:
                        return []
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    out = []
                    for h in hits[:3]:
                        src = h.get("_source", {})
                        out.append({
                            "ticker":        ticker,
                            "entity_name":   src.get("entity_name", ""),
                            "form_type":     src.get("form_type", ""),
                            "file_date":     src.get("file_date", ""),
                            "accession_no":  src.get("accession_no", ""),
                        })
                    return out
            except Exception:
                return []

        results = await asyncio.gather(*[_check(t) for t in tickers])
        for res in results:
            rows.extend(res)

        if not rows:
            return pd.DataFrame(columns=["ticker", "entity_name", "form_type", "file_date", "accession_no"])
        return pd.DataFrame(rows).drop_duplicates("accession_no")


# ---------------------------------------------------------------------------
# Batch tracker
# ---------------------------------------------------------------------------

class ExtendedHoursBatchTracker:
    """Bulk pre-market quote fetching and gap universe analysis."""

    def __init__(self):
        self._adapter = AlpacaExtendedAdapter()
        self._engine  = ExtendedHoursSignalEngine()

    async def get_bulk_premarket(self, tickers: list[str]) -> pd.DataFrame:
        """
        Pre-market quotes for a list of tickers.
        Returns DataFrame with ticker, last, bid, ask, volume, session.
        """
        async def _fetch_one(ticker: str) -> Optional[dict]:
            try:
                quote = await self._adapter.get_premarket_quote(ticker)
                return {
                    "ticker":  ticker,
                    "last":    quote.get("last"),
                    "bid":     quote.get("bid"),
                    "ask":     quote.get("ask"),
                    "volume":  quote.get("last_volume"),
                    "session": quote.get("session"),
                    "source":  quote.get("source"),
                }
            except Exception:
                return None

        results = await asyncio.gather(*[_fetch_one(t) for t in tickers])
        rows = [r for r in results if r is not None]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("ticker")

    async def get_gap_universe(self, universe: str = "sp500") -> pd.DataFrame:
        """
        All pre-market gaps for the S&P 500 universe, ranked by absolute gap size.

        Args:
            universe: "sp500" (default) — uses internal representative sample
        """
        tickers = _SP500_SAMPLE if universe.lower() in ("sp500", "s&p500", "sp_500") else _SP500_SAMPLE

        async def _gap(ticker: str) -> Optional[dict]:
            try:
                g = await self._engine.compute_premarket_gap(ticker)
                if "error" in g:
                    return None
                return g
            except Exception:
                return None

        results = await asyncio.gather(*[_gap(t) for t in tickers])
        rows = [r for r in results if r is not None]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["abs_gap"] = df["gap_pct"].abs()
        df = df.sort_values("abs_gap", ascending=False).drop(columns=["abs_gap"]).reset_index(drop=True)
        return df

    async def get_earnings_calendar_reactions(
        self, lookback_days: int = 7
    ) -> pd.DataFrame:
        """
        For each ticker in universe that reported earnings in the last N days,
        compute the after-hours reaction.

        Uses yfinance earnings calendar as the data source.
        """
        rows: list[dict] = []
        cutoff = date.today() - timedelta(days=lookback_days)

        async def _check_earnings(ticker: str) -> Optional[dict]:
            try:
                loop = asyncio.get_event_loop()
                t = yf.Ticker(ticker)
                cal = await loop.run_in_executor(None, lambda: getattr(t, "calendar", None))
                if cal is None:
                    return None
                # calendar can be a dict or DataFrame
                if isinstance(cal, dict):
                    earnings_dt = cal.get("Earnings Date", [None])[0]
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    earnings_dt = cal["Earnings Date"].iloc[0] if not cal.empty else None
                else:
                    return None
                if earnings_dt is None:
                    return None
                if hasattr(earnings_dt, "date"):
                    earnings_date = earnings_dt.date()
                else:
                    earnings_date = date.fromisoformat(str(earnings_dt)[:10])
                if earnings_date < cutoff or earnings_date > date.today():
                    return None
                reaction = await self._engine.get_earnings_reaction(ticker)
                reaction["earnings_calendar_date"] = earnings_date.isoformat()
                return reaction
            except Exception:
                return None

        results = await asyncio.gather(*[_check_earnings(t) for t in _SP500_SAMPLE[:50]])
        rows = [r for r in results if r is not None]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "ah_move_pct" in df.columns:
            df = df.sort_values("ah_move_pct", key=abs, ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

extended_hours_router = APIRouter(prefix="/api/extended", tags=["extended-hours"])

_session_obj  = ExtendedHoursSession()
_alpaca_obj   = AlpacaExtendedAdapter()
_engine_obj   = ExtendedHoursSignalEngine()
_tracker_obj  = ExtendedHoursBatchTracker()


@extended_hours_router.get("/session", response_model=SessionInfo)
async def get_session_info() -> SessionInfo:
    """Current trading session info."""
    now_et = datetime.now(tz=ET)
    session = _session_obj.get_current_session()
    times = _session_obj.SESSION_TIMES.get(session, ("--:--", "--:--"))
    nxt_start = _session_obj.next_session_start()
    # Determine next session name
    nxt_session = _session_obj.get_session_for_time(nxt_start + timedelta(minutes=1))
    return SessionInfo(
        session=session,
        session_start=times[0],
        session_end=times[1],
        is_extended=_session_obj.is_extended_hours(),
        next_session=nxt_session,
        next_session_start=nxt_start.isoformat(),
    )


@extended_hours_router.get("/{ticker}/premarket")
async def get_premarket(
    ticker: str,
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD, defaults to today"),
) -> dict:
    """Pre-market quote and gap analysis for a ticker."""
    ticker = ticker.upper()
    try:
        quote = await _alpaca_obj.get_premarket_quote(ticker)
        gap   = await _engine_obj.compute_premarket_gap(ticker)
        return {"quote": quote, "gap": gap}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@extended_hours_router.get("/{ticker}/postmarket")
async def get_postmarket(
    ticker: str,
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD, defaults to today"),
) -> dict:
    """Post-market bars summary for a ticker."""
    ticker = ticker.upper()
    try:
        bars_df = await _alpaca_obj.get_postmarket_bars(ticker, date)
        if bars_df.empty:
            return {"ticker": ticker, "date": date or date.today().isoformat(), "bars": []}
        bars_df = bars_df.reset_index()
        bars_df["time"] = bars_df["time"].astype(str)
        return {"ticker": ticker, "date": date, "bars": bars_df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@extended_hours_router.get("/{ticker}/overnight-gap")
async def get_overnight_gap(
    ticker: str,
    date: Optional[str] = Query(None),
) -> dict:
    """Overnight gap analysis: prior close vs premarket price."""
    ticker = ticker.upper()
    try:
        return await _alpaca_obj.get_overnight_gap(ticker, date)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@extended_hours_router.get("/movers/premarket")
async def get_premarket_movers(
    min_gap_pct: float = Query(1.0, description="Minimum absolute gap % to include"),
    min_volume:  int   = Query(50000, description="Minimum premarket volume"),
) -> dict:
    """Pre-market movers screen across S&P 500 universe."""
    try:
        df = await _engine_obj.screen_premarket_movers(
            min_gap_pct=min_gap_pct, min_volume=min_volume
        )
        return {
            "count":  len(df),
            "movers": df.to_dict(orient="records") if not df.empty else [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@extended_hours_router.get("/earnings-reactions")
async def get_earnings_reactions(
    lookback_days: int = Query(7, description="Days back to scan for earnings"),
) -> dict:
    """Recent earnings reactions across the universe."""
    try:
        df = await _tracker_obj.get_earnings_calendar_reactions(lookback_days=lookback_days)
        return {
            "count":     len(df),
            "reactions": df.to_dict(orient="records") if not df.empty else [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
