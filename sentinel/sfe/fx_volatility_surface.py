"""
FX volatility surface construction and analysis — extends fx_analytics.py.

Adds:
  - ATM vol surface by currency pair and tenor (realized vol term structure)
  - Risk reversal and butterfly smile construction (from yfinance option chains)
  - Forward rate computation via covered interest rate parity
  - Carry trade signal generation with carry-to-vol ratio

Data sources (all free):
  Spot / historical rates:  Frankfurter API (ECB fixing) + FRED CSV fallback
  Interest rates:           FRED CSV (FEDFUNDS, ECBDFR, etc.)
  Implied vol / options:    yfinance option chains for major FX pairs
  Realized vol:             Computed from yfinance OHLCV daily history
  Forward rates:            Derived via CIP from FRED rate differentials
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRANKFURTER_BASE = "https://api.frankfurter.app"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"
}

# ---------------------------------------------------------------------------
# FX pair universe
# ---------------------------------------------------------------------------

# Major FX pairs: yfinance symbol, FRED spot series, notional/contract, pip, invert flag
# invert=True means FRED series is quote-per-USD, so we flip to USD-per-quote if needed.
FX_PAIRS: dict[str, dict] = {
    "EURUSD": {
        "yf": "EURUSD=X",
        "fred": "DEXUSEU",
        "base": "EUR",
        "quote": "USD",
        "point_value": 125_000,
        "tick": 0.0001,
        "invert": False,
    },
    "USDJPY": {
        "yf": "JPY=X",
        "fred": "DEXJPUS",
        "base": "USD",
        "quote": "JPY",
        "point_value": 12_500_000,
        "tick": 0.01,
        "invert": False,
    },
    "GBPUSD": {
        "yf": "GBPUSD=X",
        "fred": "DEXUSUK",
        "base": "GBP",
        "quote": "USD",
        "point_value": 62_500,
        "tick": 0.0001,
        "invert": False,
    },
    "AUDUSD": {
        "yf": "AUDUSD=X",
        "fred": "DEXUSAL",
        "base": "AUD",
        "quote": "USD",
        "point_value": 100_000,
        "tick": 0.0001,
        "invert": False,
    },
    "USDCAD": {
        "yf": "CAD=X",
        "fred": "DEXCAUS",
        "base": "USD",
        "quote": "CAD",
        "point_value": 100_000,
        "tick": 0.0001,
        "invert": True,   # FRED: CAD per USD → we store as USD/CAD rate
    },
    "USDCHF": {
        "yf": "CHF=X",
        "fred": "DEXSZUS",
        "base": "USD",
        "quote": "CHF",
        "point_value": 125_000,
        "tick": 0.0001,
        "invert": True,
    },
    "NZDUSD": {
        "yf": "NZDUSD=X",
        "fred": "DEXUSNZ",
        "base": "NZD",
        "quote": "USD",
        "point_value": 100_000,
        "tick": 0.0001,
        "invert": False,
    },
    "USDMXN": {
        "yf": "MXN=X",
        "fred": "DEXMXUS",
        "base": "USD",
        "quote": "MXN",
        "point_value": 500_000,
        "tick": 0.0001,
        "invert": True,
    },
    "USDBRL": {
        "yf": "BRL=X",
        "fred": "DEXBZUS",
        "base": "USD",
        "quote": "BRL",
        "point_value": 100_000,
        "tick": 0.0001,
        "invert": True,
    },
    "USDCNY": {
        "yf": "CNY=X",
        "fred": "DEXCHUS",
        "base": "USD",
        "quote": "CNY",
        "point_value": 1_000_000,
        "tick": 0.0001,
        "invert": True,
    },
}

# Tenor label → calendar days
TENORS: dict[str, int] = {
    "1W": 7,
    "1M": 30,
    "3M": 91,
    "6M": 182,
    "1Y": 365,
    "2Y": 730,
}

# FRED short-rate series per currency (annualized %)
_RATE_SERIES: dict[str, str] = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBP": "IUQABEDR",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IRSTCI01CHM156N",
    "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
    "MXN": "INTDSRMXM193N",
    "BRL": "IRSTCI01BRM156N",
}

# Vol regime breakpoints (annualized %, realised)
_VOL_THRESHOLDS = {"low": 4.0, "normal": 8.0, "elevated": 16.0}

# Carry signal thresholds
_CARRY_TO_VOL_BUY = 0.5
_CARRY_TO_VOL_STRONG_BUY = 1.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FXSpotData(BaseModel):
    pair: str
    spot: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pips: Optional[float] = None
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    timestamp: datetime


class FXVolPoint(BaseModel):
    pair: str
    tenor: str
    delta: float        # -0.25 to 0.90 (OTC convention)
    implied_vol: float  # annualized %
    is_atm: bool = False


class FXVolSurface(BaseModel):
    pair: str
    as_of: date
    spot: float
    atm_vols: dict[str, float]          # tenor → ATM vol %
    risk_reversals: dict[str, float]    # tenor → 25D RR (call_vol - put_vol)
    butterflies: dict[str, float]       # tenor → 25D fly ((call_vol + put_vol)/2 - atm_vol)
    realized_vol_30d: float
    vol_of_vol: float                   # std of rolling 30d vol window → surface stability
    skew_signal: str                    # "bullish_base" | "bearish_base" | "neutral"


class FXForward(BaseModel):
    pair: str
    spot: float
    tenor: str
    forward_rate: float
    forward_points: float               # (forward - spot) * 10000 pips
    implied_rate_differential: float    # annualized interest rate differential (%)
    carry_bps: float                    # annual carry in bps vs USD


class CarryTrade(BaseModel):
    long_pair: str      # pair to go long (high yield base)
    short_pair: str     # pair to go short (low yield base)
    carry_pct: float    # annual carry return (%)
    carry_to_vol: float # Sharpe-like carry / realised vol
    signal: str         # "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"


class RealisedVolSnapshot(BaseModel):
    pair: str
    vol_1w: float
    vol_1m: float
    vol_3m: float
    vol_1y: float
    vol_regime: str               # "low" | "normal" | "elevated" | "spike"
    current_vs_1y_pct: float      # how current 1M vol compares to 1Y vol (%)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns, dropping NaN."""
    return np.log(prices / prices.shift(1)).dropna()


def _realized_vol_annualized(log_rets: pd.Series, window: int) -> float:
    """Annualized realized vol (%) from last `window` log returns."""
    tail = log_rets.tail(window)
    if len(tail) < 2:
        return float("nan")
    return float(tail.std(ddof=1) * math.sqrt(252) * 100.0)


def _vol_regime(vol_1m: float) -> str:
    if not math.isfinite(vol_1m):
        return "normal"
    if vol_1m < _VOL_THRESHOLDS["low"]:
        return "low"
    if vol_1m < _VOL_THRESHOLDS["normal"]:
        return "normal"
    if vol_1m < _VOL_THRESHOLDS["elevated"]:
        return "elevated"
    return "spike"


def _carry_signal(carry_to_vol: float) -> str:
    if carry_to_vol >= _CARRY_TO_VOL_STRONG_BUY:
        return "strong_buy"
    if carry_to_vol >= _CARRY_TO_VOL_BUY:
        return "buy"
    if carry_to_vol > -_CARRY_TO_VOL_BUY:
        return "neutral"
    if carry_to_vol > -_CARRY_TO_VOL_STRONG_BUY:
        return "sell"
    return "strong_sell"


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------


async def _fred_series_async(
    client: httpx.AsyncClient, series_id: str, days_back: int = 500
) -> pd.Series:
    """
    Fetch a FRED CSV time series.  Returns a pd.Series indexed by date.
    Missing / '.' values are dropped.
    """
    try:
        start = (date.today() - timedelta(days=days_back)).isoformat()
        r = await client.get(
            _FRED_CSV,
            params={"id": series_id, "vintage_date": start},
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        if r.status_code != 200:
            logger.warning("FRED non-200", series=series_id, status=r.status_code)
            return pd.Series(dtype=float)

        rows: list[tuple[str, float]] = []
        for line in r.text.strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            dt_str, val_str = parts[0].strip(), parts[1].strip()
            if val_str in (".", "", "NA"):
                continue
            try:
                rows.append((dt_str, float(val_str)))
            except ValueError:
                continue

        if not rows:
            return pd.Series(dtype=float)

        dates, vals = zip(*rows)
        s = pd.Series(list(vals), index=pd.to_datetime(list(dates)), dtype=float)
        return s.sort_index()
    except Exception as exc:
        logger.warning("FRED series fetch failed", series=series_id, error=str(exc))
        return pd.Series(dtype=float)


async def _fred_latest_value(
    client: httpx.AsyncClient, series_id: str
) -> Optional[float]:
    """Return most recent non-missing FRED value."""
    s = await _fred_series_async(client, series_id, days_back=90)
    if s.empty:
        return None
    last = s.dropna().iloc[-1]
    return float(last)


# ---------------------------------------------------------------------------
# Frankfurter spot helper
# ---------------------------------------------------------------------------


async def _frankfurter_latest(
    client: httpx.AsyncClient, base: str, quote: str
) -> Optional[float]:
    """Fetch latest Frankfurter spot rate (quote units per 1 base unit)."""
    try:
        r = await client.get(
            f"{_FRANKFURTER_BASE}/latest",
            params={"from": base, "to": quote},
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("rates", {}).get(quote)
    except Exception as exc:
        logger.warning("Frankfurter spot failed", base=base, quote=quote, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# yfinance sync helpers (called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _yf_history_sync(yf_ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch yfinance OHLCV history synchronously."""
    import yfinance as yf  # lazy import

    try:
        tk = yf.Ticker(yf_ticker)
        hist = tk.history(period=period)
        return hist if hist is not None and not hist.empty else pd.DataFrame()
    except Exception as exc:
        logger.warning("yfinance history failed", ticker=yf_ticker, error=str(exc))
        return pd.DataFrame()


def _yf_option_chain_sync(yf_ticker: str) -> dict:
    """
    Fetch nearest-expiry option chain from yfinance.
    Returns {'calls': DataFrame|None, 'puts': DataFrame|None, 'expiry_years': float, 'spot': float|None}.
    """
    import yfinance as yf  # lazy import

    out: dict = {"calls": None, "puts": None, "expiry_years": 0.25, "spot": None}
    try:
        tk = yf.Ticker(yf_ticker)

        # Spot price
        info = tk.fast_info
        spot = getattr(info, "last_price", None)
        out["spot"] = float(spot) if spot and spot > 0 else None

        # Expiries
        exps = tk.options
        if not exps:
            return out

        # Pick nearest future expiry
        today_str = date.today().isoformat()
        future = [e for e in exps if e >= today_str]
        target = future[0] if future else exps[0]

        # Time to expiry
        try:
            exp_date = date.fromisoformat(target)
            t = max((exp_date - date.today()).days / 365.0, 1e-6)
            out["expiry_years"] = t
        except ValueError:
            pass

        chain = tk.option_chain(target)
        out["calls"] = chain.calls
        out["puts"] = chain.puts
    except Exception as exc:
        logger.warning("yfinance option chain failed", ticker=yf_ticker, error=str(exc))
    return out


# ---------------------------------------------------------------------------
# Realized vol computation
# ---------------------------------------------------------------------------


def _compute_realized_vols_from_hist(hist: pd.DataFrame) -> dict[str, float]:
    """
    Given a yfinance OHLCV DataFrame, compute realized vol at 1W/1M/3M/1Y windows.
    Returns dict with keys 'vol_1w', 'vol_1m', 'vol_3m', 'vol_1y'.
    """
    result: dict[str, float] = {
        "vol_1w": float("nan"),
        "vol_1m": float("nan"),
        "vol_3m": float("nan"),
        "vol_1y": float("nan"),
    }
    if hist.empty or "Close" not in hist.columns:
        return result

    closes = hist["Close"].dropna()
    if len(closes) < 5:
        return result

    lr = _log_returns(closes)

    windows = {"vol_1w": 5, "vol_1m": 21, "vol_3m": 63, "vol_1y": 252}
    for key, w in windows.items():
        if len(lr) >= w:
            result[key] = round(_realized_vol_annualized(lr, w), 4)

    return result


# ---------------------------------------------------------------------------
# Implied vol smile from option chain
# ---------------------------------------------------------------------------


def _extract_smile_from_chain(
    chain_data: dict,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    From yfinance option chain, extract:
      - ATM implied vol (nearest-to-money strike)
      - 25D risk reversal proxy (OTM call vol - OTM put vol)
      - 25D butterfly proxy (0.5*(OTM call vol + OTM put vol) - ATM vol)

    All in annualized % terms.  Returns (atm_vol, rr_25d, fly_25d).
    """
    calls = chain_data.get("calls")
    puts = chain_data.get("puts")
    spot = chain_data.get("spot")

    if calls is None or puts is None or spot is None or spot <= 0:
        return None, None, None

    def _clean(df: pd.DataFrame, is_call: bool) -> list[tuple[float, float]]:
        """Return [(strike, iv)] from cleaned option chain DataFrame."""
        pts: list[tuple[float, float]] = []
        for _, row in df.iterrows():
            try:
                strike = float(row.get("strike", 0))
                iv = float(row.get("impliedVolatility", 0))
                oi = float(row.get("openInterest", 0) or 0)
                vol = float(row.get("volume", 0) or 0)
                if strike <= 0 or iv <= 0 or iv > 5.0:
                    continue
                # Require some liquidity
                if oi < 1 and vol < 1:
                    continue
                pts.append((strike, iv * 100.0))
            except (TypeError, ValueError):
                continue
        return sorted(pts, key=lambda x: x[0])

    call_pts = _clean(calls, True)
    put_pts = _clean(puts, False)

    if not call_pts and not put_pts:
        return None, None, None

    # ATM: strike closest to spot
    all_pts = call_pts + put_pts
    if not all_pts:
        return None, None, None

    atm_entry = min(all_pts, key=lambda x: abs(x[0] - spot))
    atm_vol = atm_entry[1]

    # 25D OTM call: strike above spot (~25% OTM in moneyness terms for FX)
    # Approximate: 25D call strike ≈ spot * exp(0.25 * atm_vol/100 * sqrt(T))
    # For simplicity, use strikes > spot as calls and < spot as puts
    otm_calls = [(s, v) for s, v in call_pts if s > spot * 1.005]
    otm_puts = [(s, v) for s, v in put_pts if s < spot * 0.995]

    rr_25d: Optional[float] = None
    fly_25d: Optional[float] = None

    if otm_calls and otm_puts:
        # Pick 25D proxies: call at ~25% OTM moneyness, put at ~25% OTM
        # Sort: calls ascending (closest to ATM first), puts descending (closest to ATM first)
        otm_calls_sorted = sorted(otm_calls, key=lambda x: x[0])
        otm_puts_sorted = sorted(otm_puts, key=lambda x: x[0], reverse=True)

        # Pick one representative point from each wing
        call_vol = otm_calls_sorted[min(1, len(otm_calls_sorted) - 1)][1]
        put_vol = otm_puts_sorted[min(1, len(otm_puts_sorted) - 1)][1]

        rr_25d = round(call_vol - put_vol, 4)
        fly_25d = round(0.5 * (call_vol + put_vol) - atm_vol, 4)

    return round(atm_vol, 4), rr_25d, fly_25d


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FXVolatilitySurface:
    """
    FX volatility surface: term structure, risk reversals, butterflies,
    forward rates via CIP, and carry trade signals.
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout
        # Cache: pair → (spot_rate, timestamp_epoch)
        self._spot_cache: dict[str, tuple[float, float]] = {}

    # -----------------------------------------------------------------------
    # Spot rates
    # -----------------------------------------------------------------------

    async def get_spot_rates(
        self, pairs: Optional[list[str]] = None
    ) -> list[FXSpotData]:
        """
        Fetch current spot rates for all requested pairs.

        Uses yfinance for live bid/ask proxy (last/previous close) and
        Frankfurter for ECB-fixing confirmation.  Spread estimated from
        1-day high-low range as a proxy.
        """
        if pairs is None:
            pairs = list(FX_PAIRS.keys())

        now = datetime.utcnow()
        results: list[FXSpotData] = []

        async def _fetch_one(pair: str) -> Optional[FXSpotData]:
            meta = FX_PAIRS.get(pair.upper())
            if meta is None:
                logger.warning("Unknown FX pair", pair=pair)
                return None

            try:
                hist = await asyncio.to_thread(
                    _yf_history_sync, meta["yf"], "5d"
                )
                if hist.empty or "Close" not in hist.columns:
                    return None

                closes = hist["Close"].dropna()
                highs = hist["High"].dropna()
                lows = hist["Low"].dropna()

                if closes.empty:
                    return None

                spot = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
                change_pct = None
                if prev_close and abs(prev_close) > 1e-12:
                    change_pct = round((spot / prev_close - 1.0) * 100.0, 4)

                # Bid/ask proxy from intraday high-low (rough spread estimate)
                spread_pips: Optional[float] = None
                bid: Optional[float] = None
                ask: Optional[float] = None
                tick = meta.get("tick", 0.0001)

                if not highs.empty and not lows.empty:
                    daily_range = float(highs.iloc[-1]) - float(lows.iloc[-1])
                    half_spread = daily_range * 0.05  # ~5% of daily range as spread
                    bid = round(spot - half_spread, 6)
                    ask = round(spot + half_spread, 6)
                    spread_pips = round(daily_range * 0.1 / tick, 1)

                # Update cache
                import time
                self._spot_cache[pair.upper()] = (spot, time.time())

                return FXSpotData(
                    pair=pair.upper(),
                    spot=round(spot, 6),
                    bid=bid,
                    ask=ask,
                    spread_pips=spread_pips,
                    prev_close=round(prev_close, 6) if prev_close else None,
                    change_pct=change_pct,
                    timestamp=now,
                )
            except Exception as exc:
                logger.warning("Spot fetch failed", pair=pair, error=str(exc))
                return None

        tasks = [_fetch_one(p) for p in pairs]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in raw_results:
            if isinstance(res, Exception):
                logger.warning("Spot gather exception", error=str(res))
            elif res is not None:
                results.append(res)

        return results

    # -----------------------------------------------------------------------
    # Realized vol snapshot
    # -----------------------------------------------------------------------

    async def get_realized_vol(
        self, pair: str, days_back: int = 365
    ) -> RealisedVolSnapshot:
        """
        Compute realized vol at multiple windows from FRED spot series.

        Falls back to yfinance OHLCV if FRED data is insufficient.
        Classifies vol regime and reports how current 1M vol compares to 1Y vol.
        """
        pair = pair.upper()
        meta = FX_PAIRS.get(pair)

        # Try to get price history — prefer FRED for accuracy, fall back to yfinance
        hist: pd.DataFrame = pd.DataFrame()

        if meta is not None:
            async with httpx.AsyncClient() as client:
                fred_s = await _fred_series_async(
                    client, meta["fred"], days_back=max(days_back, 400)
                )
            if len(fred_s) >= 50:
                # Convert FRED series to a "Close" DataFrame
                if meta.get("invert", False):
                    fred_s = 1.0 / fred_s.replace(0, float("nan"))
                fred_s = fred_s.dropna()
                hist = pd.DataFrame({"Close": fred_s})

        # Fall back to yfinance if FRED data is thin
        if hist.empty or len(hist) < 50:
            yf_ticker = (meta or {}).get("yf", f"{pair}=X")
            hist = await asyncio.to_thread(_yf_history_sync, yf_ticker, "2y")

        vols = _compute_realized_vols_from_hist(hist)

        vol_1m = vols.get("vol_1m", float("nan"))
        vol_1y = vols.get("vol_1y", float("nan"))

        current_vs_1y_pct = 0.0
        if math.isfinite(vol_1m) and math.isfinite(vol_1y) and vol_1y > 0:
            current_vs_1y_pct = round((vol_1m / vol_1y - 1.0) * 100.0, 2)

        vol_1m_safe = vol_1m if math.isfinite(vol_1m) else 0.0

        logger.info(
            "Realized vol computed",
            pair=pair,
            vol_1m=round(vol_1m_safe, 4),
            regime=_vol_regime(vol_1m_safe),
        )

        return RealisedVolSnapshot(
            pair=pair,
            vol_1w=vols.get("vol_1w", 0.0) if math.isfinite(vols.get("vol_1w", float("nan"))) else 0.0,
            vol_1m=vol_1m_safe,
            vol_3m=vols.get("vol_3m", 0.0) if math.isfinite(vols.get("vol_3m", float("nan"))) else 0.0,
            vol_1y=vol_1y if math.isfinite(vol_1y) else 0.0,
            vol_regime=_vol_regime(vol_1m_safe),
            current_vs_1y_pct=current_vs_1y_pct,
        )

    # -----------------------------------------------------------------------
    # Vol surface
    # -----------------------------------------------------------------------

    async def get_vol_surface(self, pair: str) -> FXVolSurface:
        """
        Build FX vol surface from realized vol (ATM term structure) and
        yfinance option chain (skew / smile proxies).

        ATM vols: realized vol scaled by sqrt(tenor/30d) — forward vol approximation.
        Risk reversal: extracted from OTM call vs OTM put IV from nearest expiry chain.
        Butterfly: extracted from same chain.
        All tenors beyond nearest expiry are extrapolated from RV term structure.
        """
        pair = pair.upper()
        meta = FX_PAIRS.get(pair)
        yf_ticker = (meta or {}).get("yf", f"{pair}=X")

        # Run history fetch and option chain in parallel
        hist, chain_data = await asyncio.gather(
            asyncio.to_thread(_yf_history_sync, yf_ticker, "2y"),
            asyncio.to_thread(_yf_option_chain_sync, yf_ticker),
        )

        # Spot
        spot = chain_data.get("spot") or 0.0
        if spot <= 0 and not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if not closes.empty:
                spot = float(closes.iloc[-1])

        # Realized vol at all windows
        vols = _compute_realized_vols_from_hist(hist)
        vol_1m = vols.get("vol_1m", 10.0) or 10.0
        vol_3m = vols.get("vol_3m", 10.0) or 10.0
        vol_1y = vols.get("vol_1y", 10.0) or 10.0

        # Build ATM term structure via realized vol scaling
        # Variance is linear in time → vol scales as sqrt(T/T_ref)
        # We use 1M vol as anchor point
        ref_days = 30.0

        atm_vols: dict[str, float] = {}
        for tenor, days in TENORS.items():
            if days <= 30:
                # 1W and 1M: scale from vol_1m
                scale = math.sqrt(days / ref_days)
                atm_vols[tenor] = round(vol_1m * scale, 4)
            elif days <= 91:
                # Interpolate between 1M and 3M
                w = (days - 30.0) / (91.0 - 30.0)
                atm_vols[tenor] = round(vol_1m * (1 - w) + vol_3m * w, 4)
            else:
                # Scale from 3M anchor
                scale = math.sqrt(days / 91.0)
                v = vol_3m * scale
                # But dampen long-end: blend with vol_1y as anchor for 1Y+
                if days >= 365:
                    atm_vols[tenor] = round(vol_1y, 4)
                else:
                    w = (days - 91.0) / (365.0 - 91.0)
                    atm_vols[tenor] = round(vol_3m * (1 - w) + vol_1y * w, 4)

        # Extract smile from options chain
        atm_chain, rr_chain, fly_chain = _extract_smile_from_chain(chain_data)

        # If option chain gave us a better ATM estimate for short end, use it
        if atm_chain is not None and math.isfinite(atm_chain) and atm_chain > 0:
            atm_vols["1M"] = round(atm_chain, 4)
            atm_vols["1W"] = round(atm_chain * math.sqrt(7.0 / 30.0), 4)

        # Build risk_reversals and butterflies across tenors
        # Short end from chain; long end estimated from historical skewness
        rr_base = rr_chain if (rr_chain is not None and math.isfinite(rr_chain)) else 0.0
        fly_base = fly_chain if (fly_chain is not None and math.isfinite(fly_chain)) else 0.0

        # Historical return skewness → skew proxy for smile construction
        skewness = 0.0
        if not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 30:
                lr = _log_returns(closes)
                skewness = float(lr.skew()) if len(lr) >= 10 else 0.0

        # Estimate RR from skewness: negative skew = left tail fatter = put skew
        if abs(rr_base) < 0.001 and abs(skewness) > 0.1:
            rr_base = round(-skewness * 0.5, 4)  # negative skew → negative RR

        risk_reversals: dict[str, float] = {}
        butterflies: dict[str, float] = {}
        for tenor, days in TENORS.items():
            # RR decays slightly for longer tenors (mean reversion)
            decay = 1.0 / math.sqrt(days / 30.0)
            decay = max(0.4, min(1.0, decay))
            risk_reversals[tenor] = round(rr_base * decay, 4)
            # Butterfly (excess kurtosis proxy) grows with sqrt(T) → fat tails compound
            fly_scale = math.sqrt(days / 30.0)
            fly_scale = min(fly_scale, 2.0)  # cap growth
            butterflies[tenor] = round(fly_base * fly_scale, 4)

        # Vol of vol: rolling std of 30d realized vol estimates
        vol_of_vol = 0.0
        if not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 60:
                lr = _log_returns(closes)
                # Compute rolling 21-day vol at each point
                rolling_vols = [
                    _realized_vol_annualized(lr.iloc[max(0, i - 21):i], 21)
                    for i in range(21, len(lr))
                ]
                rv_arr = np.array([v for v in rolling_vols if math.isfinite(v)])
                if len(rv_arr) >= 5:
                    vol_of_vol = round(float(np.std(rv_arr, ddof=1)), 4)

        # Skew signal: positive RR = OTM calls expensive → market bullish on base
        if rr_base > 0.3:
            skew_signal = "bullish_base"
        elif rr_base < -0.3:
            skew_signal = "bearish_base"
        else:
            skew_signal = "neutral"

        logger.info(
            "Vol surface built",
            pair=pair,
            atm_1m=atm_vols.get("1M"),
            rr_1m=risk_reversals.get("1M"),
            fly_1m=butterflies.get("1M"),
            skew=skew_signal,
        )

        return FXVolSurface(
            pair=pair,
            as_of=date.today(),
            spot=round(spot, 6),
            atm_vols=atm_vols,
            risk_reversals=risk_reversals,
            butterflies=butterflies,
            realized_vol_30d=round(vol_1m, 4),
            vol_of_vol=vol_of_vol,
            skew_signal=skew_signal,
        )

    # -----------------------------------------------------------------------
    # Forward rates via CIP
    # -----------------------------------------------------------------------

    async def get_forward_rates(self, pair: str) -> list[FXForward]:
        """
        Compute forward rates via covered interest rate parity:
          F = S * (1 + r_base * T) / (1 + r_quote * T)

        Interest rates sourced from FRED.  Spot from yfinance / Frankfurter.
        Returns one FXForward per standard tenor (1W, 1M, 3M, 6M, 1Y, 2Y).
        """
        pair = pair.upper()
        meta = FX_PAIRS.get(pair)
        if meta is None:
            logger.warning("Unknown pair for forward rates", pair=pair)
            return []

        base_ccy = meta["base"]
        quote_ccy = meta["quote"]

        async with httpx.AsyncClient() as client:
            # Fetch spot and rates in parallel
            spot_task = _frankfurter_latest(client, base_ccy, quote_ccy)
            base_rate_task = _fred_latest_value(client, _RATE_SERIES.get(base_ccy, "FEDFUNDS"))
            quote_rate_task = _fred_latest_value(client, _RATE_SERIES.get(quote_ccy, "FEDFUNDS"))

            spot_val, base_rate, quote_rate = await asyncio.gather(
                spot_task, base_rate_task, quote_rate_task
            )

        # Fall back to yfinance for spot if Frankfurter fails
        if spot_val is None or spot_val <= 0:
            hist = await asyncio.to_thread(_yf_history_sync, meta["yf"], "5d")
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    spot_val = float(closes.iloc[-1])

        if spot_val is None or spot_val <= 0:
            logger.warning("No spot for forward rates", pair=pair)
            return []

        # Default rates if FRED unavailable
        r_base = (base_rate or 4.5) / 100.0
        r_quote = (quote_rate or 4.5) / 100.0

        forwards: list[FXForward] = []
        for tenor, days in TENORS.items():
            T = days / 365.0
            denom = 1.0 + r_quote * T
            if abs(denom) < 1e-10:
                continue

            fwd = spot_val * (1.0 + r_base * T) / denom
            fwd_pts = (fwd - spot_val) * 10_000.0

            # Carry bps: annualized rate differential in bps
            carry_bps = (r_base - r_quote) * 10_000.0

            forwards.append(FXForward(
                pair=pair,
                spot=round(spot_val, 6),
                tenor=tenor,
                forward_rate=round(fwd, 6),
                forward_points=round(fwd_pts, 4),
                implied_rate_differential=round((r_base - r_quote) * 100.0, 4),
                carry_bps=round(carry_bps, 2),
            ))

        logger.info("Forward rates computed", pair=pair, tenors=len(forwards))
        return forwards

    # -----------------------------------------------------------------------
    # Carry trade signals
    # -----------------------------------------------------------------------

    async def compute_carry_signals(
        self, pairs: Optional[list[str]] = None
    ) -> list[CarryTrade]:
        """
        Compute carry trade signals for all pairs.

        Carry = rate_differential (base_rate - quote_rate)
        Carry-to-vol = carry / realized_vol_1m  (Sharpe-like metric)
        Signal thresholds:
          carry_to_vol > 1.0  → strong_buy (long base)
          carry_to_vol > 0.5  → buy
          carry_to_vol > -0.5 → neutral
          carry_to_vol > -1.0 → sell
          otherwise           → strong_sell
        """
        if pairs is None:
            pairs = list(FX_PAIRS.keys())

        pairs = [p.upper() for p in pairs if p.upper() in FX_PAIRS]

        # Fetch all rates in parallel
        async with httpx.AsyncClient() as client:
            all_ccys = set()
            for p in pairs:
                m = FX_PAIRS[p]
                all_ccys.add(m["base"])
                all_ccys.add(m["quote"])

            rate_tasks = {
                ccy: _fred_latest_value(client, _RATE_SERIES[ccy])
                for ccy in all_ccys
                if ccy in _RATE_SERIES
            }
            rate_vals = await asyncio.gather(*rate_tasks.values(), return_exceptions=True)
            rates: dict[str, Optional[float]] = {}
            for ccy, val in zip(rate_tasks.keys(), rate_vals):
                rates[ccy] = val if not isinstance(val, Exception) else None

        # Fetch realized vols in parallel
        vol_tasks = [self.get_realized_vol(p) for p in pairs]
        vol_results = await asyncio.gather(*vol_tasks, return_exceptions=True)
        vol_map: dict[str, float] = {}
        for pair, vr in zip(pairs, vol_results):
            if isinstance(vr, Exception) or vr is None:
                vol_map[pair] = 10.0  # default 10% vol
            else:
                vol_map[pair] = max(vr.vol_1m, 0.01)

        signals: list[CarryTrade] = []
        for pair in pairs:
            m = FX_PAIRS[pair]
            base_rate = rates.get(m["base"])
            quote_rate = rates.get(m["quote"])

            if base_rate is None or quote_rate is None:
                continue

            carry_pct = round(base_rate - quote_rate, 4)
            vol_1m = vol_map.get(pair, 10.0)
            carry_to_vol = round(carry_pct / vol_1m, 4) if vol_1m > 0 else 0.0

            sig = _carry_signal(carry_to_vol)

            signals.append(CarryTrade(
                long_pair=pair if carry_pct >= 0 else f"USD{m['base']}",
                short_pair=pair if carry_pct < 0 else pair,
                carry_pct=carry_pct,
                carry_to_vol=carry_to_vol,
                signal=sig,
            ))

        # Sort by absolute carry-to-vol (strongest signals first)
        signals.sort(key=lambda x: abs(x.carry_to_vol), reverse=True)
        logger.info("Carry signals computed", count=len(signals))
        return signals

    # -----------------------------------------------------------------------
    # Cross rates
    # -----------------------------------------------------------------------

    async def get_cross_rates(self, base: str, quote: str) -> float:
        """
        Compute cross rate from USD pairs.
        e.g. EURJPY = EURUSD / USDJPY (adjusted for quote convention).
        Returns cross rate as quote_units per 1 base_unit, or 0.0 on failure.
        """
        base = base.upper()
        quote = quote.upper()

        if base == quote:
            return 1.0

        # Get both vs USD
        async with httpx.AsyncClient() as client:
            base_vs_usd, quote_vs_usd = await asyncio.gather(
                _frankfurter_latest(client, "USD", base),
                _frankfurter_latest(client, "USD", quote),
            )

        # frankfurter gives us: USD_units per 1 base, USD_units per 1 quote
        # So base_per_usd = base_vs_usd (units of BASE per 1 USD)
        # Cross rate base/quote = (1/base_per_usd) / (1/quote_per_usd) = quote_per_usd / base_per_usd

        if base_vs_usd and quote_vs_usd and base_vs_usd > 0:
            cross = quote_vs_usd / base_vs_usd
            return round(cross, 6)

        logger.warning("Cross rate computation failed", base=base, quote=quote)
        return 0.0

    # -----------------------------------------------------------------------
    # Vol cone
    # -----------------------------------------------------------------------

    async def get_vol_cone(self, pair: str) -> pd.DataFrame:
        """
        Historical realized vol at different percentiles.

        Rows: lookback window (1W, 1M, 3M, 6M, 1Y)
        Cols: p10, p25, p50, p75, p90, current
        Shows how current vol compares to historical range.
        """
        pair = pair.upper()
        meta = FX_PAIRS.get(pair)
        yf_ticker = (meta or {}).get("yf", f"{pair}=X")

        hist = await asyncio.to_thread(_yf_history_sync, yf_ticker, "5y")

        if hist.empty or "Close" not in hist.columns:
            return pd.DataFrame()

        closes = hist["Close"].dropna()
        lr = _log_returns(closes)

        rows = []
        window_map = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

        for label, w in window_map.items():
            if len(lr) < w * 2:
                continue

            # Rolling realized vol at this window
            rolling_vol = []
            for i in range(w, len(lr)):
                rv = _realized_vol_annualized(lr.iloc[i - w:i], w)
                if math.isfinite(rv):
                    rolling_vol.append(rv)

            if not rolling_vol:
                continue

            arr = np.array(rolling_vol)
            current = rolling_vol[-1]

            rows.append({
                "tenor": label,
                "p10": round(float(np.percentile(arr, 10)), 2),
                "p25": round(float(np.percentile(arr, 25)), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p75": round(float(np.percentile(arr, 75)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "current": round(current, 2),
                "percentile_rank": round(
                    float(np.mean(arr <= current)) * 100.0, 1
                ),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("tenor")
        logger.info("Vol cone built", pair=pair, tenors=len(df))
        return df

    # -----------------------------------------------------------------------
    # FX dashboard
    # -----------------------------------------------------------------------

    async def fx_dashboard(
        self, pairs: Optional[list[str]] = None
    ) -> dict:
        """
        Comprehensive FX market overview.

        Returns:
          - All spot rates with changes
          - Realized vol snapshots (all windows)
          - Carry signals
          - Vol surface summary (ATM term structure + skew)
          - Cross-rate matrix for major pairs
        """
        if pairs is None:
            pairs = list(FX_PAIRS.keys())

        pairs = [p.upper() for p in pairs if p.upper() in FX_PAIRS]
        warnings_: list[str] = []

        # Parallel fetch: spots, vols, carry, surfaces
        spot_task = self.get_spot_rates(pairs)
        carry_task = self.compute_carry_signals(pairs)

        # Vol snapshots for each pair
        vol_snapshot_tasks = [self.get_realized_vol(p) for p in pairs]

        spots, carry_signals, *vol_snapshots_raw = await asyncio.gather(
            spot_task,
            carry_task,
            *vol_snapshot_tasks,
            return_exceptions=True,
        )

        if isinstance(spots, Exception):
            warnings_.append(f"Spot fetch failed: {spots}")
            spots = []
        if isinstance(carry_signals, Exception):
            warnings_.append(f"Carry signals failed: {carry_signals}")
            carry_signals = []

        vol_snapshots: list[RealisedVolSnapshot] = []
        for i, vr in enumerate(vol_snapshots_raw):
            if isinstance(vr, Exception):
                warnings_.append(f"Vol snapshot failed for {pairs[i]}: {vr}")
            elif vr is not None:
                vol_snapshots.append(vr)

        # Vol surface for G10 pairs (limited to avoid timeout)
        g10_pairs = [p for p in pairs if FX_PAIRS[p]["base"] in {"EUR", "GBP", "AUD", "NZD", "USD"}][:4]
        surface_tasks = [self.get_vol_surface(p) for p in g10_pairs]
        surface_results = await asyncio.gather(*surface_tasks, return_exceptions=True)

        surfaces: dict[str, dict] = {}
        for pair, sr in zip(g10_pairs, surface_results):
            if isinstance(sr, Exception):
                warnings_.append(f"Vol surface failed for {pair}: {sr}")
            elif sr is not None:
                surfaces[pair] = sr.model_dump()

        logger.info(
            "FX dashboard assembled",
            pairs=len(pairs),
            spots=len(spots),
            surfaces=len(surfaces),
        )

        return {
            "as_of": date.today().isoformat(),
            "spots": [s.model_dump() for s in spots],
            "realized_vols": [v.model_dump() for v in vol_snapshots],
            "carry_signals": [c.model_dump() for c in carry_signals],
            "vol_surfaces": surfaces,
            "pair_count": len(pairs),
            "warnings": warnings_,
        }

    # -----------------------------------------------------------------------
    # Internal FRED spot helper
    # -----------------------------------------------------------------------

    async def _fetch_fred_spot(self, pair: str, days_back: int = 500) -> pd.Series:
        """
        Fetch FRED FX time series as a pd.Series of spot rates.
        Handles inversion for pairs where FRED quotes as foreign-per-USD.
        """
        pair = pair.upper()
        meta = FX_PAIRS.get(pair)
        if meta is None:
            return pd.Series(dtype=float)

        async with httpx.AsyncClient() as client:
            s = await _fred_series_async(client, meta["fred"], days_back=days_back)

        if s.empty:
            return s

        if meta.get("invert", False):
            s = 1.0 / s.replace(0, float("nan"))

        return s.dropna()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def fx_surface(pair: str) -> FXVolSurface:
    """Build and return the FX vol surface for a single pair."""
    engine = FXVolatilitySurface()
    return await engine.get_vol_surface(pair)


async def fx_carry_signals(pairs: Optional[list[str]] = None) -> list[CarryTrade]:
    """Return carry trade signals for all (or specified) FX pairs."""
    engine = FXVolatilitySurface()
    return await engine.compute_carry_signals(pairs)


async def fx_dashboard(pairs: Optional[list[str]] = None) -> dict:
    """Return comprehensive FX market dashboard."""
    engine = FXVolatilitySurface()
    return await engine.fx_dashboard(pairs)


async def fx_forward_rates(pair: str) -> list[FXForward]:
    """Return forward rate curve for a single pair via CIP."""
    engine = FXVolatilitySurface()
    return await engine.get_forward_rates(pair)


async def fx_vol_cone(pair: str) -> pd.DataFrame:
    """Return historical vol cone DataFrame for a single pair."""
    engine = FXVolatilitySurface()
    return await engine.get_vol_cone(pair)


async def fx_realized_vol(pair: str) -> RealisedVolSnapshot:
    """Return realized vol snapshot (all windows) for a single pair."""
    engine = FXVolatilitySurface()
    return await engine.get_realized_vol(pair)
