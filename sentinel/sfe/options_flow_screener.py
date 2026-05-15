"""
Options flow screener: unusual activity detection, dark pool vs lit,
sweep detection, whale trades, put/call ratio by strike/expiry.
Free data: CBOE delayed, yfinance options, OCC.
"""
from __future__ import annotations

import math
import sqlite3
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent / "data" / "options_flow.db"
_CACHE_TTL_SECONDS = 900          # 15-minute TTL for options data
_RISK_FREE_RATE = 0.053           # approx Fed Funds / 3M T-bill
_DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "AMD", "NFLX", "BABA", "COIN", "MARA", "PLTR", "SOFI", "HOOD", "GME",
    "AMC", "RIVN", "LCID", "NIO", "F", "GM", "GS", "JPM", "BAC", "XOM",
    "CVX", "OXY", "DIS", "INTC", "CSCO", "ORCL", "CRM", "ADBE", "PYPL",
]

_UNUSUAL_VOL_OI_THRESHOLD = 0.5   # vol/OI > 0.5 is unusual
_SWEEP_BID_ASK_RATIO = 0.6        # (ask-bid)/ask > 0.6 suggests sweep
_WHALE_PREMIUM_USD = 1_000_000    # $1M+ single-leg premium
_OTM_DELTA_THRESHOLD = 0.30       # delta < 0.30 = OTM / speculative

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_db() -> None:
    """Create SQLite tables if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB_PATH)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS options_cache (
                ticker      TEXT NOT NULL,
                expiry      TEXT NOT NULL,
                option_type TEXT NOT NULL,   -- 'call' or 'put'
                cached_at   INTEGER NOT NULL,
                payload     BLOB NOT NULL,
                PRIMARY KEY (ticker, expiry, option_type)
            );

            CREATE TABLE IF NOT EXISTS pcr_history (
                ticker      TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                pcr_volume  REAL,
                pcr_oi      REAL,
                pcr_front   REAL,
                pcr_back    REAL,
                sentiment   TEXT,
                PRIMARY KEY (ticker, as_of)
            );

            CREATE TABLE IF NOT EXISTS unusual_flow_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                expiry      TEXT,
                strike      REAL,
                option_type TEXT,
                unusual_score REAL,
                volume      INTEGER,
                open_interest INTEGER,
                premium_usd REAL,
                signal      TEXT
            );

            CREATE TABLE IF NOT EXISTS max_pain_history (
                ticker      TEXT NOT NULL,
                as_of       TEXT NOT NULL,
                max_pain_strike REAL,
                spot_price  REAL,
                gex         REAL,
                PRIMARY KEY (ticker, as_of)
            );
        """)


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OptionRow(BaseModel):
    ticker: str
    expiry: str
    option_type: str          # 'call' or 'put'
    strike: float
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    # Derived
    mid: float
    spread_pct: float         # (ask - bid) / ask
    vol_oi_ratio: float       # volume / max(OI, 1)
    premium_usd: float        # mid * volume * 100 (dollar value of today's flow)
    moneyness: Optional[float] = None    # strike / spot
    delta_proxy: Optional[float] = None  # simplified BS delta
    unusual_score: float = 0.0
    days_to_expiry: int = 0
    is_0dte: bool = False


class OptionsChain(BaseModel):
    ticker: str
    spot_price: float
    as_of: str
    expiries: list[str]
    calls: list[OptionRow]
    puts: list[OptionRow]
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int


class PutCallRatios(BaseModel):
    ticker: str
    spot_price: float
    as_of: str
    pcr_volume: float          # total put vol / total call vol
    pcr_oi: float              # total put OI / total call OI
    pcr_near_money: float      # ±5% of spot
    pcr_far_otm: float         # >10% OTM
    pcr_front_month: float
    pcr_back_month: float
    sentiment: str             # "bullish" | "neutral" | "bearish"


class MaxPainResult(BaseModel):
    ticker: str
    spot_price: float
    as_of: str
    expiry: str
    max_pain_strike: float
    max_pain_distance_pct: float   # (max_pain - spot) / spot
    gex: float                      # gamma exposure in $bn
    gamma_flip_level: Optional[float] = None
    key_gamma_strike: float
    interpretation: str


class UnusualActivity(BaseModel):
    ticker: str
    expiry: str
    strike: float
    option_type: str
    volume: int
    open_interest: int
    vol_oi_ratio: float
    premium_usd: float
    unusual_score: float
    flags: list[str]           # ["sweep", "whale", "0DTE", "otm_spec", ...]
    last_price: float
    iv: float
    days_to_expiry: int


class ScreenerResult(BaseModel):
    as_of: str
    top_unusual: list[UnusualActivity]
    sweep_leaders: list[dict]
    whale_trades: list[dict]
    zero_dte_flow: list[dict]
    iv_term_structure_flags: list[dict]


class OptionsSignal(BaseModel):
    ticker: str
    spot_price: float
    as_of: str
    signal: str                     # "bullish" | "bearish" | "vol_expansion" | "vol_compression" | "neutral"
    confidence: float               # 0-100
    gamma_squeeze_risk: float        # 0-100
    reasons: list[str]
    pcr_volume: float
    pcr_oi: float
    max_pain_strike: Optional[float] = None
    gex: Optional[float] = None


# ---------------------------------------------------------------------------
# Helper: simplified Black-Scholes delta
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Approximate BS delta. Returns 0.0 on edge cases."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if is_call:
            return _norm_cdf(d1)
        else:
            return _norm_cdf(d1) - 1.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        return pdf_d1 / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# 1. OptionsChainCollector
# ---------------------------------------------------------------------------

class OptionsChainCollector:
    """
    Collect full options chains via yfinance.
    Caches each (ticker, expiry, option_type) tuple in SQLite for 15 minutes.
    """

    def __init__(self, cache_ttl: int = _CACHE_TTL_SECONDS) -> None:
        self.cache_ttl = cache_ttl
        _ensure_db()

    def get_chain(self, ticker: str) -> OptionsChain:
        """
        Fetch and return the full options chain for a ticker across all available expiries.
        Computes derived fields (vol/OI ratio, premium, moneyness, delta proxy,
        unusual score) for each row.
        """
        ticker = ticker.upper()
        try:
            yt = yf.Ticker(ticker)
            info = yt.info
            spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
            if spot <= 0:
                # Fallback: try fast_info
                fi = yt.fast_info
                spot = float(getattr(fi, "last_price", 0) or 0)
        except Exception as exc:
            logger.warning("options_chain spot fetch failed %s: %s", ticker, exc)
            spot = 0.0

        try:
            expiries: list[str] = list(yt.options)
        except Exception as exc:
            logger.warning("options list failed %s: %s", ticker, exc)
            expiries = []

        all_calls: list[OptionRow] = []
        all_puts: list[OptionRow] = []
        today = date.today()

        for expiry in expiries:
            try:
                calls_df, puts_df = self._fetch_expiry(ticker, expiry, yt, today)
            except Exception as exc:
                logger.debug("expiry %s %s failed: %s", ticker, expiry, exc)
                continue

            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            is_0dte = (dte == 0)

            for df, opt_type in ((calls_df, "call"), (puts_df, "put")):
                if df is None or df.empty:
                    continue
                is_call = (opt_type == "call")
                for _, row in df.iterrows():
                    strike = float(row.get("strike", 0))
                    if strike <= 0:
                        continue
                    last = float(row.get("lastPrice", 0) or 0)
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    vol = int(row.get("volume", 0) or 0)
                    oi = int(row.get("openInterest", 0) or 0)
                    iv = float(row.get("impliedVolatility", 0) or 0)

                    mid = (bid + ask) / 2 if (bid > 0 or ask > 0) else last
                    spread_pct = (ask - bid) / ask if ask > 0 else 0.0
                    vol_oi = vol / max(oi, 1)
                    premium_usd = mid * vol * 100

                    moneyness: Optional[float] = None
                    delta_proxy: Optional[float] = None
                    if spot > 0 and strike > 0:
                        moneyness = strike / spot
                        T = max(dte, 1) / 365.0
                        sigma = iv if iv > 0 else 0.3
                        delta_proxy = _bs_delta(spot, strike, T, _RISK_FREE_RATE, sigma, is_call)

                    opt_row = OptionRow(
                        ticker=ticker,
                        expiry=expiry,
                        option_type=opt_type,
                        strike=strike,
                        last_price=last,
                        bid=bid,
                        ask=ask,
                        volume=vol,
                        open_interest=oi,
                        implied_volatility=round(iv, 6),
                        mid=round(mid, 4),
                        spread_pct=round(spread_pct, 4),
                        vol_oi_ratio=round(vol_oi, 4),
                        premium_usd=round(premium_usd, 2),
                        moneyness=round(moneyness, 4) if moneyness else None,
                        delta_proxy=round(delta_proxy, 4) if delta_proxy is not None else None,
                        days_to_expiry=dte,
                        is_0dte=is_0dte,
                    )
                    if is_call:
                        all_calls.append(opt_row)
                    else:
                        all_puts.append(opt_row)

        return OptionsChain(
            ticker=ticker,
            spot_price=round(spot, 4),
            as_of=str(today),
            expiries=expiries,
            calls=all_calls,
            puts=all_puts,
            total_call_volume=sum(r.volume for r in all_calls),
            total_put_volume=sum(r.volume for r in all_puts),
            total_call_oi=sum(r.open_interest for r in all_calls),
            total_put_oi=sum(r.open_interest for r in all_puts),
        )

    def _fetch_expiry(
        self, ticker: str, expiry: str, yt: yf.Ticker, today: date
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch calls/puts for one expiry, with SQLite cache."""
        now_ts = int(time.time())
        calls_df: Optional[pd.DataFrame] = None
        puts_df: Optional[pd.DataFrame] = None

        with _db() as conn:
            for opt_type in ("call", "put"):
                row = conn.execute(
                    "SELECT payload, cached_at FROM options_cache "
                    "WHERE ticker=? AND expiry=? AND option_type=?",
                    (ticker, expiry, opt_type),
                ).fetchone()
                if row and (now_ts - row["cached_at"]) < self.cache_ttl:
                    try:
                        df = pd.read_json(row["payload"])
                        if opt_type == "call":
                            calls_df = df
                        else:
                            puts_df = df
                    except Exception:
                        pass

        # Fetch missing data from yfinance
        if calls_df is None or puts_df is None:
            chain = yt.option_chain(expiry)
            calls_fresh = chain.calls.copy() if hasattr(chain, "calls") else pd.DataFrame()
            puts_fresh = chain.puts.copy() if hasattr(chain, "puts") else pd.DataFrame()

            with _db() as conn:
                for opt_type, df in (("call", calls_fresh), ("put", puts_fresh)):
                    if not df.empty:
                        payload = df.to_json()
                        conn.execute(
                            "INSERT OR REPLACE INTO options_cache "
                            "(ticker, expiry, option_type, cached_at, payload) VALUES (?,?,?,?,?)",
                            (ticker, expiry, opt_type, now_ts, payload),
                        )

            if calls_df is None:
                calls_df = calls_fresh
            if puts_df is None:
                puts_df = puts_fresh

        return calls_df, puts_df


# ---------------------------------------------------------------------------
# 2. UnusualActivityDetector
# ---------------------------------------------------------------------------

class UnusualActivityDetector:
    """
    Score each options row for unusual activity using multiple heuristics.
    """

    def __init__(self, adv_lookup: Optional[dict[str, float]] = None) -> None:
        """
        adv_lookup: optional mapping ticker -> 30-day average dollar volume (stock).
        If not provided, the ADV-relative size check is skipped.
        """
        self.adv_lookup = adv_lookup or {}

    def score_activity(self, row: OptionRow, adv: Optional[float] = None) -> float:
        """
        Score one options row for unusualness. Returns 0-100.

        Scoring sub-components (each max 25 points):
          A. Volume/OI ratio
          B. Absolute dollar premium
          C. OTM speculative (call delta < 0.30 or put delta > -0.30 with high vol)
          D. Sweep indicator (wide spread + high volume)
        """
        score = 0.0

        # A. Volume/OI ratio
        if row.vol_oi_ratio > 2.0:
            score += 25.0
        elif row.vol_oi_ratio > 1.0:
            score += 18.0
        elif row.vol_oi_ratio > 0.5:
            score += 10.0
        elif row.vol_oi_ratio > 0.2:
            score += 4.0

        # B. Premium size
        if row.premium_usd >= _WHALE_PREMIUM_USD:
            score += 25.0
        elif row.premium_usd >= 500_000:
            score += 18.0
        elif row.premium_usd >= 100_000:
            score += 10.0
        elif row.premium_usd >= 25_000:
            score += 5.0

        # ADV-relative size check (bonus if we have it)
        effective_adv = adv or self.adv_lookup.get(row.ticker)
        if effective_adv and effective_adv > 0:
            rel = row.premium_usd / effective_adv
            if rel > 0.05:
                score = min(100, score + 10)
            elif rel > 0.01:
                score = min(100, score + 5)

        # C. OTM speculative bet
        if row.delta_proxy is not None:
            abs_delta = abs(row.delta_proxy)
            if abs_delta < _OTM_DELTA_THRESHOLD and row.volume > 100:
                score += 20.0
            elif abs_delta < 0.15 and row.volume > 50:
                score += 25.0

        # D. Sweep indicator: wide spread suggests taker aggressiveness
        if row.spread_pct > _SWEEP_BID_ASK_RATIO and row.volume > 200:
            score += 25.0
        elif row.spread_pct > 0.4 and row.volume > 100:
            score += 15.0
        elif row.spread_pct > 0.2 and row.volume > 50:
            score += 8.0

        # 0DTE bonus (very time-sensitive = speculative)
        if row.is_0dte and row.volume > 50:
            score = min(100, score + 10)

        return round(min(score, 100.0), 2)

    def score_chain(self, chain: OptionsChain) -> OptionsChain:
        """Score all rows in an OptionsChain and update unusual_score in place."""
        adv = self.adv_lookup.get(chain.ticker)
        for row in chain.calls + chain.puts:
            row.unusual_score = self.score_activity(row, adv)
        return chain

    def get_unusual_activity(
        self, chain: OptionsChain, min_score: float = 50.0, min_volume: int = 10
    ) -> list[UnusualActivity]:
        """Extract rows with unusual_score >= min_score as UnusualActivity objects."""
        results: list[UnusualActivity] = []
        for row in chain.calls + chain.puts:
            if row.unusual_score < min_score:
                continue
            if row.volume < min_volume:
                continue
            flags: list[str] = []
            if row.vol_oi_ratio > 0.5:
                flags.append("high_vol_oi")
            if row.premium_usd >= _WHALE_PREMIUM_USD:
                flags.append("whale")
            if row.is_0dte:
                flags.append("0DTE")
            if row.delta_proxy is not None and abs(row.delta_proxy) < _OTM_DELTA_THRESHOLD:
                flags.append("otm_spec")
            if row.spread_pct > _SWEEP_BID_ASK_RATIO and row.volume > 200:
                flags.append("sweep")
            results.append(UnusualActivity(
                ticker=chain.ticker,
                expiry=row.expiry,
                strike=row.strike,
                option_type=row.option_type,
                volume=row.volume,
                open_interest=row.open_interest,
                vol_oi_ratio=row.vol_oi_ratio,
                premium_usd=row.premium_usd,
                unusual_score=row.unusual_score,
                flags=flags,
                last_price=row.last_price,
                iv=row.implied_volatility,
                days_to_expiry=row.days_to_expiry,
            ))
        results.sort(key=lambda r: r.unusual_score, reverse=True)
        return results

    def detect_sweeps(self, chain: OptionsChain, top_n: int = 10) -> list[dict]:
        """
        Detect sweep-like trades: high volume + wide bid-ask spread.
        Returns list of dicts with key fields.
        """
        candidates = []
        for row in chain.calls + chain.puts:
            if row.volume < 100:
                continue
            # Sweep heuristic: taker pays the spread (bid-ask wide = aggressor)
            sweep_score = (row.spread_pct * row.volume * row.mid) / 1000
            if row.spread_pct > 0.2:
                candidates.append({
                    "ticker": chain.ticker,
                    "expiry": row.expiry,
                    "strike": row.strike,
                    "option_type": row.option_type,
                    "volume": row.volume,
                    "spread_pct": row.spread_pct,
                    "mid": row.mid,
                    "premium_usd": row.premium_usd,
                    "sweep_score": round(sweep_score, 2),
                    "days_to_expiry": row.days_to_expiry,
                })
        candidates.sort(key=lambda x: x["sweep_score"], reverse=True)
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# 3. PutCallRatioAnalyzer
# ---------------------------------------------------------------------------

class PutCallRatioAnalyzer:
    """Compute put/call ratios across various dimensions and persist history."""

    def analyze(self, chain: OptionsChain) -> PutCallRatios:
        """
        Compute PCR by volume, OI, moneyness bucket, and expiry term.
        Persists to pcr_history table.
        """
        spot = chain.spot_price
        today_str = chain.as_of

        # All-chain PCR
        total_call_vol = sum(r.volume for r in chain.calls)
        total_put_vol = sum(r.volume for r in chain.puts)
        total_call_oi = sum(r.open_interest for r in chain.calls)
        total_put_oi = sum(r.open_interest for r in chain.puts)

        pcr_volume = total_put_vol / max(total_call_vol, 1)
        pcr_oi = total_put_oi / max(total_call_oi, 1)

        # Near-money PCR (±5% of spot)
        near_lo, near_hi = spot * 0.95, spot * 1.05
        nm_calls_vol = sum(r.volume for r in chain.calls if near_lo <= r.strike <= near_hi)
        nm_puts_vol = sum(r.volume for r in chain.puts if near_lo <= r.strike <= near_hi)
        pcr_near_money = nm_puts_vol / max(nm_calls_vol, 1)

        # Far OTM PCR (>10% OTM)
        otm_calls_vol = sum(r.volume for r in chain.calls if r.strike > spot * 1.10)
        otm_puts_vol = sum(r.volume for r in chain.puts if r.strike < spot * 0.90)
        pcr_far_otm = otm_puts_vol / max(otm_calls_vol, 1)

        # Front vs back month: front = expiries within 30 days
        today_dt = date.fromisoformat(today_str)
        front_calls = sum(
            r.volume for r in chain.calls
            if (datetime.strptime(r.expiry, "%Y-%m-%d").date() - today_dt).days <= 30
        )
        front_puts = sum(
            r.volume for r in chain.puts
            if (datetime.strptime(r.expiry, "%Y-%m-%d").date() - today_dt).days <= 30
        )
        back_calls = sum(
            r.volume for r in chain.calls
            if (datetime.strptime(r.expiry, "%Y-%m-%d").date() - today_dt).days > 30
        )
        back_puts = sum(
            r.volume for r in chain.puts
            if (datetime.strptime(r.expiry, "%Y-%m-%d").date() - today_dt).days > 30
        )
        pcr_front = front_puts / max(front_calls, 1)
        pcr_back = back_puts / max(back_calls, 1)

        # Sentiment
        if pcr_volume > 1.2:
            sentiment = "bearish"
        elif pcr_volume < 0.7:
            sentiment = "bullish"
        else:
            sentiment = "neutral"

        # Persist
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pcr_history "
                "(ticker, as_of, pcr_volume, pcr_oi, pcr_front, pcr_back, sentiment) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    chain.ticker, today_str,
                    round(pcr_volume, 4), round(pcr_oi, 4),
                    round(pcr_front, 4), round(pcr_back, 4),
                    sentiment,
                ),
            )

        return PutCallRatios(
            ticker=chain.ticker,
            spot_price=spot,
            as_of=today_str,
            pcr_volume=round(pcr_volume, 4),
            pcr_oi=round(pcr_oi, 4),
            pcr_near_money=round(pcr_near_money, 4),
            pcr_far_otm=round(pcr_far_otm, 4),
            pcr_front_month=round(pcr_front, 4),
            pcr_back_month=round(pcr_back, 4),
            sentiment=sentiment,
        )

    def get_pcr_history(self, ticker: str, lookback_days: int = 30) -> pd.DataFrame:
        """Retrieve rolling PCR history from SQLite."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM pcr_history WHERE ticker=? AND as_of >= ? ORDER BY as_of",
                (ticker.upper(), cutoff),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# 4. MaxPainCalculator
# ---------------------------------------------------------------------------

class MaxPainCalculator:
    """
    Compute max pain (minimum aggregate option value) and gamma exposure (GEX).
    """

    def calculate(self, chain: OptionsChain, expiry: Optional[str] = None) -> MaxPainResult:
        """
        Calculate max pain for the given expiry (defaults to front-month).
        Also computes GEX across all expiries.
        """
        spot = chain.spot_price
        today_str = chain.as_of
        today_dt = date.fromisoformat(today_str)

        # Choose expiry
        if expiry is None:
            # Use front-month: first expiry >= 7 days out (avoid 0DTE)
            expiry = self._pick_front_expiry(chain.expiries, today_dt)

        # Filter chain to chosen expiry
        exp_calls = [r for r in chain.calls if r.expiry == expiry]
        exp_puts = [r for r in chain.puts if r.expiry == expiry]

        # All strikes in this expiry
        strikes = sorted(set(r.strike for r in exp_calls + exp_puts))

        if not strikes:
            return MaxPainResult(
                ticker=chain.ticker, spot_price=spot, as_of=today_str, expiry=expiry or "",
                max_pain_strike=spot, max_pain_distance_pct=0.0,
                gex=0.0, key_gamma_strike=spot,
                interpretation="Insufficient data",
            )

        # Build OI lookup
        call_oi: dict[float, int] = {r.strike: r.open_interest for r in exp_calls}
        put_oi: dict[float, int] = {r.strike: r.open_interest for r in exp_puts}

        # Max pain: for each potential expiry price S, sum total option holder losses
        # (which equals total option writer profits)
        min_pain = float("inf")
        max_pain_strike = strikes[len(strikes) // 2]

        pain_by_strike: dict[float, float] = {}
        for S in strikes:
            total_pain = 0.0
            for K in strikes:
                c_oi = call_oi.get(K, 0)
                p_oi = put_oi.get(K, 0)
                # Call value at expiry S: max(S-K, 0) per share
                total_pain += c_oi * max(S - K, 0.0) * 100
                # Put value at expiry S: max(K-S, 0) per share
                total_pain += p_oi * max(K - S, 0.0) * 100
            pain_by_strike[S] = total_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = S

        max_pain_dist = (max_pain_strike - spot) / spot if spot > 0 else 0.0

        # GEX: Gamma Exposure across ALL expiries
        # GEX (call) = gamma × OI × 100 × spot²/100 (in $)
        # Net GEX = GEX_calls - GEX_puts
        gex_total = 0.0
        gex_by_strike: dict[float, float] = {}

        for r in chain.calls + chain.puts:
            T = max(r.days_to_expiry, 1) / 365.0
            sigma = r.implied_volatility if r.implied_volatility > 0 else 0.3
            gamma = _bs_gamma(spot, r.strike, T, _RISK_FREE_RATE, sigma)
            gex_contribution = gamma * r.open_interest * 100 * (spot ** 2) / 100
            if r.option_type == "call":
                gex_total += gex_contribution
                gex_by_strike[r.strike] = gex_by_strike.get(r.strike, 0.0) + gex_contribution
            else:
                gex_total -= gex_contribution
                gex_by_strike[r.strike] = gex_by_strike.get(r.strike, 0.0) - gex_contribution

        # Key gamma strike: highest absolute GEX concentration
        if gex_by_strike:
            key_gamma_strike = max(gex_by_strike.keys(), key=lambda k: abs(gex_by_strike[k]))
        else:
            key_gamma_strike = spot

        # Gamma flip level: strike where GEX crosses zero (simplified)
        gamma_flip: Optional[float] = None
        sorted_strikes = sorted(gex_by_strike.keys())
        for i in range(len(sorted_strikes) - 1):
            g1 = gex_by_strike[sorted_strikes[i]]
            g2 = gex_by_strike[sorted_strikes[i + 1]]
            if g1 * g2 < 0:
                # Linear interpolation
                gamma_flip = sorted_strikes[i] + (sorted_strikes[i + 1] - sorted_strikes[i]) * (
                    -g1 / (g2 - g1)
                )
                break

        # Interpretation
        gex_bn = gex_total / 1e9
        if gex_bn > 1:
            interp = f"Positive GEX ${gex_bn:.2f}B — dealers long gamma, vol suppression expected"
        elif gex_bn < -1:
            interp = f"Negative GEX ${gex_bn:.2f}B — dealers short gamma, vol amplification risk"
        else:
            interp = f"Near-zero GEX ${gex_bn:.2f}B — neutral dealer positioning"

        if abs(max_pain_dist) > 0.03:
            interp += f". Max pain {abs(max_pain_dist)*100:.1f}% {'above' if max_pain_dist>0 else 'below'} spot."

        # Persist
        with _db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO max_pain_history "
                "(ticker, as_of, max_pain_strike, spot_price, gex) VALUES (?,?,?,?,?)",
                (chain.ticker, today_str, max_pain_strike, spot, round(gex_bn, 4)),
            )

        return MaxPainResult(
            ticker=chain.ticker,
            spot_price=spot,
            as_of=today_str,
            expiry=expiry,
            max_pain_strike=max_pain_strike,
            max_pain_distance_pct=round(max_pain_dist, 4),
            gex=round(gex_bn, 4),
            gamma_flip_level=round(gamma_flip, 2) if gamma_flip else None,
            key_gamma_strike=key_gamma_strike,
            interpretation=interp,
        )

    @staticmethod
    def _pick_front_expiry(expiries: list[str], today_dt: date) -> str:
        """Pick the nearest expiry that is at least 7 days out."""
        for exp in expiries:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
                if (d - today_dt).days >= 7:
                    return exp
            except ValueError:
                continue
        return expiries[0] if expiries else ""


# ---------------------------------------------------------------------------
# 5. OptionsFlowScreener
# ---------------------------------------------------------------------------

class OptionsFlowScreener:
    """
    Screen across a universe of tickers for notable options flow.
    Identifies: unusual activity, sweeps, whale trades, 0DTE flow, IV spikes.
    """

    def __init__(self, universe: Optional[list[str]] = None, max_workers: int = 6) -> None:
        self.universe = [t.upper() for t in (universe or _DEFAULT_UNIVERSE)]
        self.max_workers = max_workers
        self._collector = OptionsChainCollector()
        self._detector = UnusualActivityDetector()
        self._pcr = PutCallRatioAnalyzer()
        self._max_pain = MaxPainCalculator()

    def run(self, min_unusual_score: float = 55.0) -> ScreenerResult:
        """
        Screen the universe. Returns top unusual activity, sweep leaders,
        whale trades, 0DTE flow, and IV term-structure flags.
        """
        today_str = str(date.today())
        chains: list[OptionsChain] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_ticker = {
                pool.submit(self._safe_get_chain, t): t
                for t in self.universe
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    chain = future.result(timeout=30)
                    if chain:
                        chains.append(chain)
                except Exception as exc:
                    logger.debug("screener chain failed %s: %s", ticker, exc)

        # Score all chains
        for chain in chains:
            self._detector.score_chain(chain)

        # Top unusual activity across the universe
        all_unusual: list[UnusualActivity] = []
        all_sweeps: list[dict] = []
        all_whales: list[dict] = []
        all_0dte: list[dict] = []
        all_iv_flags: list[dict] = []

        for chain in chains:
            unusual = self._detector.get_unusual_activity(chain, min_score=min_unusual_score)
            all_unusual.extend(unusual)

            # Sweeps
            sweeps = self._detector.detect_sweeps(chain, top_n=5)
            all_sweeps.extend(sweeps)

            # Whale trades: premium >= $1M
            for row in chain.calls + chain.puts:
                if row.premium_usd >= _WHALE_PREMIUM_USD:
                    all_whales.append({
                        "ticker": chain.ticker,
                        "expiry": row.expiry,
                        "strike": row.strike,
                        "option_type": row.option_type,
                        "premium_usd": row.premium_usd,
                        "volume": row.volume,
                        "iv": row.implied_volatility,
                        "days_to_expiry": row.days_to_expiry,
                    })

                # 0DTE flow
                if row.is_0dte and row.volume >= 100:
                    all_0dte.append({
                        "ticker": chain.ticker,
                        "strike": row.strike,
                        "option_type": row.option_type,
                        "volume": row.volume,
                        "premium_usd": row.premium_usd,
                        "iv": row.implied_volatility,
                    })

            # IV term structure: front month IV vs back month
            iv_flag = self._check_iv_term_structure(chain)
            if iv_flag:
                all_iv_flags.append(iv_flag)

        # Sort and trim
        all_unusual.sort(key=lambda x: x.unusual_score, reverse=True)
        top_unusual = all_unusual[:20]

        all_sweeps.sort(key=lambda x: x["sweep_score"], reverse=True)
        sweep_leaders = all_sweeps[:10]

        all_whales.sort(key=lambda x: x["premium_usd"], reverse=True)
        whale_trades = all_whales[:20]

        all_0dte.sort(key=lambda x: x["premium_usd"], reverse=True)
        zero_dte_flow = all_0dte[:20]

        all_iv_flags.sort(key=lambda x: x.get("iv_ratio", 0), reverse=True)

        # Log unusual to DB
        self._log_unusual(top_unusual)

        return ScreenerResult(
            as_of=today_str,
            top_unusual=top_unusual,
            sweep_leaders=sweep_leaders,
            whale_trades=whale_trades,
            zero_dte_flow=zero_dte_flow,
            iv_term_structure_flags=all_iv_flags,
        )

    def _safe_get_chain(self, ticker: str) -> Optional[OptionsChain]:
        try:
            return self._collector.get_chain(ticker)
        except Exception as exc:
            logger.debug("safe_get_chain %s: %s", ticker, exc)
            return None

    @staticmethod
    def _check_iv_term_structure(chain: OptionsChain) -> Optional[dict]:
        """
        Check if front-month IV is elevated vs back-month (>30% higher = earnings/event play).
        """
        today_dt = date.fromisoformat(chain.as_of)

        # Front month: <= 45 days
        front_ivs = [
            r.implied_volatility for r in chain.calls + chain.puts
            if 0 < r.days_to_expiry <= 45 and r.implied_volatility > 0 and r.volume > 0
        ]
        back_ivs = [
            r.implied_volatility for r in chain.calls + chain.puts
            if r.days_to_expiry > 45 and r.implied_volatility > 0 and r.volume > 0
        ]

        if not front_ivs or not back_ivs:
            return None

        front_avg = float(np.median(front_ivs))
        back_avg = float(np.median(back_ivs))

        if back_avg <= 0:
            return None

        iv_ratio = front_avg / back_avg
        if iv_ratio > 1.30:
            return {
                "ticker": chain.ticker,
                "front_iv": round(front_avg, 4),
                "back_iv": round(back_avg, 4),
                "iv_ratio": round(iv_ratio, 4),
                "interpretation": "Elevated front-month IV — potential earnings/event catalyst",
            }
        return None

    @staticmethod
    def _log_unusual(items: list[UnusualActivity]) -> None:
        """Persist unusual flow to SQLite for historical reference."""
        now_str = datetime.utcnow().isoformat()
        with _db() as conn:
            for item in items:
                conn.execute(
                    "INSERT INTO unusual_flow_log "
                    "(recorded_at, ticker, expiry, strike, option_type, unusual_score, "
                    "volume, open_interest, premium_usd, signal) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        now_str, item.ticker, item.expiry, item.strike, item.option_type,
                        item.unusual_score, item.volume, item.open_interest,
                        item.premium_usd, ",".join(item.flags),
                    ),
                )


# ---------------------------------------------------------------------------
# 6. OptionsSignalEngine
# ---------------------------------------------------------------------------

class OptionsSignalEngine:
    """
    Generate trading signals from options flow data.
    Combines PCR, max pain, GEX, unusual activity, and term structure.
    """

    def __init__(self) -> None:
        self._collector = OptionsChainCollector()
        self._detector = UnusualActivityDetector()
        self._pcr_analyzer = PutCallRatioAnalyzer()
        self._max_pain_calc = MaxPainCalculator()

    def generate_signal(self, ticker: str) -> OptionsSignal:
        """
        Full signal pipeline for a single ticker.
        Returns an OptionsSignal with direction, confidence, gamma squeeze risk.
        """
        ticker = ticker.upper()

        # Fetch chain
        chain = self._collector.get_chain(ticker)
        self._detector.score_chain(chain)
        pcr = self._pcr_analyzer.analyze(chain)
        max_pain = self._max_pain_calc.calculate(chain)

        spot = chain.spot_price
        today_str = chain.as_of
        reasons: list[str] = []
        bull_score = 0.0
        bear_score = 0.0
        vol_buy_score = 0.0
        vol_sell_score = 0.0

        # --- Directional signals ---

        # PCR Volume: high = bearish sentiment (put buying)
        if pcr.pcr_volume > 1.2:
            bear_score += 20
            reasons.append(f"Bearish PCR (volume) = {pcr.pcr_volume:.2f}")
        elif pcr.pcr_volume < 0.7:
            bull_score += 20
            reasons.append(f"Bullish PCR (volume) = {pcr.pcr_volume:.2f}")

        # PCR OI: put OI accumulation
        if pcr.pcr_oi > 1.3:
            bear_score += 15
            reasons.append(f"Elevated put OI ratio = {pcr.pcr_oi:.2f}")
        elif pcr.pcr_oi < 0.8:
            bull_score += 15
            reasons.append(f"Call OI dominance = {pcr.pcr_oi:.2f}")

        # OTM call activity (bullish spec)
        otm_call_vol = sum(
            r.volume for r in chain.calls
            if r.delta_proxy is not None and r.delta_proxy < _OTM_DELTA_THRESHOLD
        )
        total_call_vol = sum(r.volume for r in chain.calls) or 1
        otm_call_frac = otm_call_vol / total_call_vol
        if otm_call_frac > 0.4:
            bull_score += 15
            reasons.append(f"OTM call fraction = {otm_call_frac:.1%} (speculative bullish bets)")

        # OTM put activity (defensive hedging = bearish)
        otm_put_vol = sum(
            r.volume for r in chain.puts
            if r.delta_proxy is not None and abs(r.delta_proxy) < _OTM_DELTA_THRESHOLD
        )
        total_put_vol = sum(r.volume for r in chain.puts) or 1
        otm_put_frac = otm_put_vol / total_put_vol
        if otm_put_frac > 0.4:
            bear_score += 15
            reasons.append(f"OTM put fraction = {otm_put_frac:.1%} (defensive hedging)")

        # Sweep direction: more call sweeps = bullish aggression
        call_sweeps = sum(
            1 for r in chain.calls if r.spread_pct > _SWEEP_BID_ASK_RATIO and r.volume > 200
        )
        put_sweeps = sum(
            1 for r in chain.puts if r.spread_pct > _SWEEP_BID_ASK_RATIO and r.volume > 200
        )
        if call_sweeps > put_sweeps * 2:
            bull_score += 20
            reasons.append(f"Call sweeps ({call_sweeps}) dominate put sweeps ({put_sweeps})")
        elif put_sweeps > call_sweeps * 2:
            bear_score += 20
            reasons.append(f"Put sweeps ({put_sweeps}) dominate call sweeps ({call_sweeps})")

        # Max pain: spot below max pain = upside pressure from pinning
        if spot > 0 and abs(max_pain.max_pain_distance_pct) > 0.02:
            if max_pain.max_pain_strike > spot:
                bull_score += 10
                reasons.append(
                    f"Max pain {max_pain.max_pain_strike:.2f} above spot "
                    f"({max_pain.max_pain_distance_pct*100:.1f}%) — upside pinning bias"
                )
            else:
                bear_score += 10
                reasons.append(
                    f"Max pain {max_pain.max_pain_strike:.2f} below spot — downside pinning bias"
                )

        # --- Volatility signals ---

        # Straddle buying: roughly balanced call/put volume near-money with high IV
        nm_lo, nm_hi = spot * 0.97, spot * 1.03
        nm_call_vol = sum(r.volume for r in chain.calls if nm_lo <= r.strike <= nm_hi)
        nm_put_vol = sum(r.volume for r in chain.puts if nm_lo <= r.strike <= nm_hi)
        nm_balance = min(nm_call_vol, nm_put_vol) / max(max(nm_call_vol, nm_put_vol), 1)
        nm_avg_iv = np.mean(
            [r.implied_volatility for r in chain.calls + chain.puts
             if nm_lo <= r.strike <= nm_hi and r.implied_volatility > 0]
        ) if any(nm_lo <= r.strike <= nm_hi for r in chain.calls + chain.puts) else 0.0
        if nm_balance > 0.7 and nm_avg_iv > 0.4:
            vol_buy_score += 30
            reasons.append(
                f"Near-money straddle buying detected (balance={nm_balance:.2f}, IV={nm_avg_iv:.2f})"
            )
        elif nm_balance < 0.3:
            vol_sell_score += 20
            reasons.append("Directional near-money flow (straddle selling signal)")

        # IV term structure (front vs back)
        iv_flag = OptionsFlowScreener._check_iv_term_structure(chain)
        if iv_flag:
            vol_buy_score += 20
            reasons.append(iv_flag["interpretation"])

        # --- Gamma squeeze risk ---
        # High when: GEX negative + significant OTM call OI above spot
        gamma_squeeze = 0.0
        if max_pain.gex < -0.5:
            gamma_squeeze += 40
            reasons.append(f"Negative GEX ({max_pain.gex:.2f}B) — dealer short gamma amplifies moves")
        above_spot_call_oi = sum(
            r.open_interest for r in chain.calls if r.strike > spot * 1.05
        )
        total_oi = sum(r.open_interest for r in chain.calls + chain.puts) or 1
        above_frac = above_spot_call_oi / total_oi
        if above_frac > 0.25:
            gamma_squeeze += 30
            reasons.append(
                f"Large above-spot call OI ({above_frac:.1%}) — squeeze fuel if stock moves up"
            )
        gamma_squeeze = min(gamma_squeeze, 100.0)

        # --- Determine primary signal ---
        net_dir = bull_score - bear_score
        net_vol = vol_buy_score - vol_sell_score

        if abs(net_vol) > 30 and abs(net_vol) > abs(net_dir):
            if net_vol > 0:
                signal = "vol_expansion"
                confidence = min(net_vol, 100.0)
            else:
                signal = "vol_compression"
                confidence = min(-net_vol, 100.0)
        elif net_dir > 20:
            signal = "bullish"
            confidence = min(net_dir, 100.0)
        elif net_dir < -20:
            signal = "bearish"
            confidence = min(-net_dir, 100.0)
        else:
            signal = "neutral"
            confidence = max(0, 50.0 - abs(net_dir))

        return OptionsSignal(
            ticker=ticker,
            spot_price=spot,
            as_of=today_str,
            signal=signal,
            confidence=round(confidence, 1),
            gamma_squeeze_risk=round(gamma_squeeze, 1),
            reasons=reasons,
            pcr_volume=pcr.pcr_volume,
            pcr_oi=pcr.pcr_oi,
            max_pain_strike=max_pain.max_pain_strike,
            gex=max_pain.gex,
        )


# ---------------------------------------------------------------------------
# 7. FastAPI Router
# ---------------------------------------------------------------------------

options_flow_router = APIRouter(prefix="/options", tags=["options-flow"])

# Module-level singletons (created on first import)
_collector = OptionsChainCollector()
_detector = UnusualActivityDetector()
_pcr_analyzer = PutCallRatioAnalyzer()
_max_pain_calc = MaxPainCalculator()
_screener = OptionsFlowScreener()
_signal_engine = OptionsSignalEngine()


@options_flow_router.get("/chain/{ticker}", response_model=OptionsChain)
def api_get_chain(ticker: str) -> OptionsChain:
    """
    Return the full options chain (all expiries, all strikes) for a ticker.
    Includes derived fields: vol/OI ratio, premium, moneyness, delta proxy, unusual score.
    Data is cached in SQLite for 15 minutes.
    """
    try:
        chain = _collector.get_chain(ticker.upper())
        _detector.score_chain(chain)
        return chain
    except Exception as exc:
        logger.error("api_get_chain %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/unusual", response_model=ScreenerResult)
def api_unusual(
    min_score: float = Query(default=55.0, ge=0, le=100),
) -> ScreenerResult:
    """
    Screen the default universe for unusual options activity.
    Returns top unusual flow, sweeps, whale trades, 0DTE, and IV flags.
    """
    try:
        return _screener.run(min_unusual_score=min_score)
    except Exception as exc:
        logger.error("api_unusual: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/pcr/{ticker}", response_model=PutCallRatios)
def api_pcr(ticker: str) -> PutCallRatios:
    """
    Put/call ratio analysis for a ticker.
    Returns PCR by volume, OI, near-money, far-OTM, front/back month.
    """
    try:
        chain = _collector.get_chain(ticker.upper())
        return _pcr_analyzer.analyze(chain)
    except Exception as exc:
        logger.error("api_pcr %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/max-pain/{ticker}", response_model=MaxPainResult)
def api_max_pain(
    ticker: str,
    expiry: Optional[str] = Query(default=None, description="YYYY-MM-DD expiry; defaults to front month"),
) -> MaxPainResult:
    """
    Max pain and GEX analysis for a ticker.
    Max pain = strike minimizing total option value at expiry.
    GEX = net gamma exposure across all expiries in $bn.
    """
    try:
        chain = _collector.get_chain(ticker.upper())
        return _max_pain_calc.calculate(chain, expiry=expiry)
    except Exception as exc:
        logger.error("api_max_pain %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/gex/{ticker}", response_model=MaxPainResult)
def api_gex(ticker: str) -> MaxPainResult:
    """
    Gamma exposure (GEX) for a ticker.
    Positive GEX = dealers long gamma (vol suppression).
    Negative GEX = dealers short gamma (vol amplification).
    """
    try:
        chain = _collector.get_chain(ticker.upper())
        return _max_pain_calc.calculate(chain)
    except Exception as exc:
        logger.error("api_gex %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/screener", response_model=ScreenerResult)
def api_screener(
    tickers: Optional[str] = Query(
        default=None,
        description="Comma-separated tickers to screen; defaults to built-in universe",
    ),
    min_score: float = Query(default=50.0, ge=0, le=100),
) -> ScreenerResult:
    """
    Full options flow screener across a configurable universe.
    """
    try:
        universe = [t.strip().upper() for t in tickers.split(",")] if tickers else None
        local_screener = OptionsFlowScreener(universe=universe)
        return local_screener.run(min_unusual_score=min_score)
    except Exception as exc:
        logger.error("api_screener: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@options_flow_router.get("/signals/{ticker}", response_model=OptionsSignal)
def api_signals(ticker: str) -> OptionsSignal:
    """
    Generate a directional/volatility signal from options flow for a ticker.
    Combines PCR, max pain, GEX, sweep detection, and term structure analysis.
    """
    try:
        return _signal_engine.generate_signal(ticker.upper())
    except Exception as exc:
        logger.error("api_signals %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_chain(ticker: str) -> OptionsChain:
    """Fetch the full options chain for a ticker."""
    chain = _collector.get_chain(ticker)
    _detector.score_chain(chain)
    return chain


def get_unusual_activity(ticker: str, min_score: float = 50.0) -> list[UnusualActivity]:
    """Get unusual options activity for a single ticker."""
    chain = get_chain(ticker)
    return _detector.get_unusual_activity(chain, min_score=min_score)


def get_pcr(ticker: str) -> PutCallRatios:
    """Get put/call ratios for a ticker."""
    chain = _collector.get_chain(ticker)
    return _pcr_analyzer.analyze(chain)


def get_max_pain(ticker: str, expiry: Optional[str] = None) -> MaxPainResult:
    """Get max pain and GEX for a ticker."""
    chain = _collector.get_chain(ticker)
    return _max_pain_calc.calculate(chain, expiry=expiry)


def get_signal(ticker: str) -> OptionsSignal:
    """Get options flow signal for a ticker."""
    return _signal_engine.generate_signal(ticker)


def screen_universe(universe: Optional[list[str]] = None) -> ScreenerResult:
    """Screen a universe of tickers for unusual options flow."""
    sc = OptionsFlowScreener(universe=universe)
    return sc.run()
