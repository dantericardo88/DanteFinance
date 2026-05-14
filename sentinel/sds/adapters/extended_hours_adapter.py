from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
logger = get_logger(__name__)

_PRE_START_H = 4
_PRE_END_H = 9
_PRE_END_M = 30
_POST_START_H = 16
_POST_END_H = 20
_ALPACA_BASE = "https://data.alpaca.markets/v2/stocks"
_POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"


class ExtendedHoursBar(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    session: str
    ticker: str


class OvernightGap(BaseModel):
    ticker: str
    gap_date: date
    prior_close: float
    ah_low: Optional[float] = None
    ah_high: Optional[float] = None
    premarket_open: Optional[float] = None
    gap_open: Optional[float] = None
    gap_pct: float
    gap_direction: str
    filled_by_noon: Optional[bool] = None


class EarningsReaction(BaseModel):
    ticker: str
    earnings_date: date
    ah_move_pct: Optional[float] = None
    premarket_move_pct: Optional[float] = None
    open_reaction_pct: Optional[float] = None
    intraday_reversal_pct: Optional[float] = None


class ExtendedHoursAdapter:
    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._polygon_key = os.getenv("POLYGON_API_KEY", "")

    def _classify_session(self, dt_utc: datetime) -> str:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC)
        dt_et = dt_utc.astimezone(ET)
        h, m = dt_et.hour, dt_et.minute
        total_min = h * 60 + m
        pre_start = _PRE_START_H * 60
        pre_end = _PRE_END_H * 60 + _PRE_END_M
        regular_end = _POST_START_H * 60
        post_end = _POST_END_H * 60
        if pre_start <= total_min < pre_end:
            return "pre"
        if pre_end <= total_min < regular_end:
            return "regular"
        if regular_end <= total_min < post_end:
            return "post"
        return "closed"

    async def _fetch_alpaca_bars(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not self._alpaca_key or not self._alpaca_secret:
            return pd.DataFrame()
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
        }
        start_iso = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{_ALPACA_BASE}/{ticker.upper()}/bars"
        params = {
            "timeframe": "1Min",
            "start": start_iso,
            "end": end_iso,
            "feed": "iex",
            "extended_hours": "true",
            "limit": 10000,
        }
        all_bars: list[dict] = []
        next_token: Optional[str] = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                while True:
                    if next_token:
                        params["page_token"] = next_token
                    resp = await client.get(url, headers=headers, params=params)
                    if resp.status_code == 429:
                        logger.warning("Alpaca rate limit", ticker=ticker)
                        await asyncio.sleep(2)
                        continue
                    resp.raise_for_status()
                    payload = resp.json()
                    bars = payload.get("bars") or []
                    all_bars.extend(bars)
                    next_token = payload.get("next_page_token")
                    if not next_token:
                        break
        except Exception as exc:
            logger.warning("Alpaca extended bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

        if not all_bars:
            return pd.DataFrame()

        rows = []
        for b in all_bars:
            ts_str = b.get("t", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            rows.append({
                "time": ts,
                "open": float(b.get("o", 0)),
                "high": float(b.get("h", 0)),
                "low": float(b.get("l", 0)),
                "close": float(b.get("c", 0)),
                "volume": int(b.get("v", 0)),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    def _fetch_yfinance_bars(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        try:
            t = yf.Ticker(ticker)
            end_dt = end + timedelta(days=1)
            df = t.history(
                start=start.isoformat(),
                end=end_dt.isoformat(),
                interval="1m",
                prepost=True,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df.index.name = "time"
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index = pd.to_datetime(df.index, utc=True)
            return df[["open", "high", "low", "close", "volume"]].sort_index()
        except Exception as exc:
            logger.warning("yfinance bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

    async def _fetch_polygon_bars(
        self, ticker: str, from_date: date, to_date: date
    ) -> pd.DataFrame:
        if not self._polygon_key:
            return pd.DataFrame()
        from_str = from_date.isoformat()
        to_str = to_date.isoformat()
        url = f"{_POLYGON_BASE}/{ticker.upper()}/range/1/minute/{from_str}/{to_str}"
        params: dict = {
            "adjusted": "true",
            "limit": 50000,
            "sort": "asc",
        }
        if self._polygon_key:
            params["apiKey"] = self._polygon_key
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Polygon bars failed", ticker=ticker, error=str(exc))
            return pd.DataFrame()

        results = data.get("results") or []
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            ts = datetime.utcfromtimestamp(r["t"] / 1000).replace(tzinfo=UTC)
            rows.append({
                "time": ts,
                "open": float(r.get("o", 0)),
                "high": float(r.get("h", 0)),
                "low": float(r.get("l", 0)),
                "close": float(r.get("c", 0)),
                "volume": int(r.get("v", 0)),
            })
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    async def _get_bars_with_fallback(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        df = await self._fetch_alpaca_bars(ticker, start, end)
        if df is not None and not df.empty:
            logger.info("Alpaca extended hours bars", ticker=ticker, rows=len(df))
            return df
        loop = asyncio.get_event_loop()
        start_d = start.astimezone(ET).date()
        end_d = end.astimezone(ET).date()
        df = await loop.run_in_executor(
            None, self._fetch_yfinance_bars, ticker, start_d, end_d
        )
        if df is not None and not df.empty:
            logger.info("yfinance extended hours bars", ticker=ticker, rows=len(df))
            return df
        df = await self._fetch_polygon_bars(ticker, start_d, end_d)
        if df is not None and not df.empty:
            logger.info("Polygon extended hours bars", ticker=ticker, rows=len(df))
            return df
        logger.warning("All sources exhausted for extended hours", ticker=ticker)
        return pd.DataFrame()

    def _df_to_bars(
        self, df: pd.DataFrame, ticker: str, session_filter: str
    ) -> list[ExtendedHoursBar]:
        bars: list[ExtendedHoursBar] = []
        for ts, row in df.iterrows():
            dt_utc = pd.Timestamp(ts).to_pydatetime()
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=UTC)
            sess = self._classify_session(dt_utc)
            if session_filter == "all":
                if sess not in ("pre", "post"):
                    continue
            elif sess != session_filter:
                continue
            bars.append(ExtendedHoursBar(
                time=dt_utc,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                session=sess,
                ticker=ticker,
            ))
        return bars

    async def get_extended_bars(
        self,
        ticker: str,
        session_date: date,
        session: str = "all",
    ) -> list[ExtendedHoursBar]:
        if session == "pre":
            start_et = datetime(session_date.year, session_date.month, session_date.day,
                                _PRE_START_H, 0, 0, tzinfo=ET)
            end_et = datetime(session_date.year, session_date.month, session_date.day,
                              _PRE_END_H, _PRE_END_M, 0, tzinfo=ET)
        elif session == "post":
            start_et = datetime(session_date.year, session_date.month, session_date.day,
                                _POST_START_H, 0, 0, tzinfo=ET)
            end_et = datetime(session_date.year, session_date.month, session_date.day,
                              _POST_END_H, 0, 0, tzinfo=ET)
        else:
            start_et = datetime(session_date.year, session_date.month, session_date.day,
                                _PRE_START_H, 0, 0, tzinfo=ET)
            end_et = datetime(session_date.year, session_date.month, session_date.day,
                              _POST_END_H, 0, 0, tzinfo=ET)

        start_utc = start_et.astimezone(UTC)
        end_utc = end_et.astimezone(UTC)
        df = await self._get_bars_with_fallback(ticker, start_utc, end_utc)
        if df.empty:
            return []
        return self._df_to_bars(df, ticker, session)

    async def get_overnight_gap(self, ticker: str, gap_date: date) -> OvernightGap:
        prior_date = gap_date - timedelta(days=1)
        while prior_date.weekday() >= 5:
            prior_date -= timedelta(days=1)

        prior_bars = await self.get_extended_bars(ticker, prior_date, session="all")

        prior_regular_start = datetime(prior_date.year, prior_date.month, prior_date.day,
                                       9, 30, 0, tzinfo=ET).astimezone(UTC)
        prior_regular_end = datetime(prior_date.year, prior_date.month, prior_date.day,
                                     16, 0, 0, tzinfo=ET).astimezone(UTC)
        loop = asyncio.get_event_loop()
        prior_regular_df = await loop.run_in_executor(
            None, self._fetch_yfinance_bars, ticker, prior_date, prior_date
        )

        prior_close: float = 0.0
        if prior_regular_df is not None and not prior_regular_df.empty:
            regular_mask = prior_regular_df.index.to_series().apply(
                lambda ts: self._classify_session(
                    ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=UTC)
                ) == "regular"
            )
            regular_df = prior_regular_df[regular_mask]
            if not regular_df.empty:
                prior_close = float(regular_df["close"].iloc[-1])

        if prior_close == 0.0 and prior_regular_df is not None and not prior_regular_df.empty:
            prior_close = float(prior_regular_df["close"].iloc[-1])

        ah_bars = [b for b in prior_bars if b.session == "post"]
        ah_low: Optional[float] = min(b.low for b in ah_bars) if ah_bars else None
        ah_high: Optional[float] = max(b.high for b in ah_bars) if ah_bars else None

        gap_bars = await self.get_extended_bars(ticker, gap_date, session="all")
        pre_bars = [b for b in gap_bars if b.session == "pre"]
        premarket_open: Optional[float] = pre_bars[0].open if pre_bars else None

        today_full_df = await loop.run_in_executor(
            None, self._fetch_yfinance_bars, ticker, gap_date, gap_date
        )
        gap_open: Optional[float] = None
        noon_close: Optional[float] = None
        if today_full_df is not None and not today_full_df.empty:
            regular_mask = today_full_df.index.to_series().apply(
                lambda ts: self._classify_session(
                    ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=UTC)
                ) == "regular"
            )
            regular_today = today_full_df[regular_mask]
            if not regular_today.empty:
                gap_open = float(regular_today["open"].iloc[0])
                noon_et = datetime(gap_date.year, gap_date.month, gap_date.day,
                                   12, 0, 0, tzinfo=ET).astimezone(UTC)
                noon_ts = pd.Timestamp(noon_et)
                before_noon = regular_today[regular_today.index <= noon_ts]
                if not before_noon.empty:
                    noon_close = float(before_noon["close"].iloc[-1])

        open_price = gap_open if gap_open is not None else (premarket_open or prior_close)
        if prior_close == 0.0:
            gap_pct = 0.0
        else:
            gap_pct = (open_price - prior_close) / prior_close * 100.0

        if abs(gap_pct) < 0.05:
            gap_direction = "flat"
        elif gap_pct > 0:
            gap_direction = "up"
        else:
            gap_direction = "down"

        filled_by_noon: Optional[bool] = None
        if noon_close is not None and prior_close != 0.0:
            if gap_direction == "up":
                filled_by_noon = noon_close <= prior_close
            elif gap_direction == "down":
                filled_by_noon = noon_close >= prior_close
            else:
                filled_by_noon = True

        return OvernightGap(
            ticker=ticker,
            gap_date=gap_date,
            prior_close=prior_close,
            ah_low=ah_low,
            ah_high=ah_high,
            premarket_open=premarket_open,
            gap_open=gap_open,
            gap_pct=gap_pct,
            gap_direction=gap_direction,
            filled_by_noon=filled_by_noon,
        )

    async def get_multi_day_extended(
        self,
        ticker: str,
        days_back: int = 30,
        session: str = "pre",
    ) -> pd.DataFrame:
        today = date.today()
        tasks = []
        dates_to_fetch: list[date] = []
        d = today
        count = 0
        while count < days_back:
            if d.weekday() < 5:
                dates_to_fetch.append(d)
                count += 1
            d -= timedelta(days=1)

        tasks = [self.get_extended_bars(ticker, sd, session) for sd in dates_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_bars: list[dict] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Multi-day fetch error", ticker=ticker, error=str(res))
                continue
            for bar in res:
                all_bars.append({
                    "time": bar.time,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "session": bar.session,
                })

        if not all_bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "session"])

        df = pd.DataFrame(all_bars)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time").sort_index()

    async def detect_earnings_reaction(
        self, ticker: str, earnings_date: date
    ) -> EarningsReaction:
        reaction_date = earnings_date + timedelta(days=1)
        while reaction_date.weekday() >= 5:
            reaction_date += timedelta(days=1)

        loop = asyncio.get_event_loop()
        earnings_df_task = loop.run_in_executor(
            None, self._fetch_yfinance_bars, ticker, earnings_date, earnings_date
        )
        reaction_df_task = loop.run_in_executor(
            None, self._fetch_yfinance_bars, ticker, reaction_date, reaction_date
        )
        earnings_df, reaction_df = await asyncio.gather(earnings_df_task, reaction_df_task)

        prior_close: Optional[float] = None
        if earnings_df is not None and not earnings_df.empty:
            regular_mask = earnings_df.index.to_series().apply(
                lambda ts: self._classify_session(
                    ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=UTC)
                ) == "regular"
            )
            regular_earnings = earnings_df[regular_mask]
            if not regular_earnings.empty:
                prior_close = float(regular_earnings["close"].iloc[-1])

        ah_bars = await self.get_extended_bars(ticker, earnings_date, session="post")
        ah_move_pct: Optional[float] = None
        if ah_bars and prior_close and prior_close != 0.0:
            ah_last = ah_bars[-1].close
            ah_move_pct = (ah_last - prior_close) / prior_close * 100.0

        pre_bars = await self.get_extended_bars(ticker, reaction_date, session="pre")
        premarket_move_pct: Optional[float] = None
        if pre_bars and prior_close and prior_close != 0.0:
            pm_last = pre_bars[-1].close
            premarket_move_pct = (pm_last - prior_close) / prior_close * 100.0

        open_reaction_pct: Optional[float] = None
        intraday_reversal_pct: Optional[float] = None
        if reaction_df is not None and not reaction_df.empty:
            regular_mask = reaction_df.index.to_series().apply(
                lambda ts: self._classify_session(
                    ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=UTC)
                ) == "regular"
            )
            regular_reaction = reaction_df[regular_mask]
            if not regular_reaction.empty and prior_close and prior_close != 0.0:
                reaction_open = float(regular_reaction["open"].iloc[0])
                reaction_close = float(regular_reaction["close"].iloc[-1])
                open_reaction_pct = (reaction_open - prior_close) / prior_close * 100.0
                if reaction_open != 0.0:
                    intraday_reversal_pct = (reaction_close - reaction_open) / reaction_open * 100.0

        return EarningsReaction(
            ticker=ticker,
            earnings_date=earnings_date,
            ah_move_pct=ah_move_pct,
            premarket_move_pct=premarket_move_pct,
            open_reaction_pct=open_reaction_pct,
            intraday_reversal_pct=intraday_reversal_pct,
        )

    async def scan_premarket_movers(
        self,
        tickers: list[str],
        min_move_pct: float = 3.0,
        min_volume: int = 10000,
    ) -> list[dict]:
        today = date.today()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        async def _process_ticker(ticker: str) -> Optional[dict]:
            loop = asyncio.get_event_loop()
            try:
                prior_df = await loop.run_in_executor(
                    None, self._fetch_yfinance_bars, ticker, yesterday, yesterday
                )
                prior_close: Optional[float] = None
                if prior_df is not None and not prior_df.empty:
                    regular_mask = prior_df.index.to_series().apply(
                        lambda ts: self._classify_session(
                            ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=UTC)
                        ) == "regular"
                    )
                    regular_prior = prior_df[regular_mask]
                    if not regular_prior.empty:
                        prior_close = float(regular_prior["close"].iloc[-1])

                if prior_close is None or prior_close == 0.0:
                    return None

                pre_bars = await self.get_extended_bars(ticker, today, session="pre")
                if not pre_bars:
                    return None

                total_volume = sum(b.volume for b in pre_bars)
                if total_volume < min_volume:
                    return None

                last_price = pre_bars[-1].close
                move_pct = (last_price - prior_close) / prior_close * 100.0

                if abs(move_pct) < min_move_pct:
                    return None

                return {
                    "ticker": ticker,
                    "premarket_move_pct": round(move_pct, 4),
                    "volume": total_volume,
                    "last_price": round(last_price, 4),
                }
            except Exception as exc:
                logger.warning("Premarket scan error", ticker=ticker, error=str(exc))
                return None

        results = await asyncio.gather(*[_process_ticker(t) for t in tickers])
        movers = [r for r in results if r is not None]
        movers.sort(key=lambda x: abs(x["premarket_move_pct"]), reverse=True)
        return movers


async def get_premarket_gap(ticker: str) -> OvernightGap:
    adapter = ExtendedHoursAdapter()
    return await adapter.get_overnight_gap(ticker, date.today())


async def scan_movers(tickers: list[str], min_pct: float = 3.0) -> list[dict]:
    adapter = ExtendedHoursAdapter()
    return await adapter.scan_premarket_movers(tickers, min_move_pct=min_pct)
