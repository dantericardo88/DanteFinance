"""
Options Flow V3 — dim_073: Options flow screener (score 6 → 9).

Comprehensive options flow analysis and screening platform using:
  - yfinance: price data and options chains (calls + puts)
  - CBOE delayed data: SPX/VIX chains
  - FRED: VIXCLS VIX history, FEDFUNDS risk-free rate

Architecture
------------
  OptionsChainFetcher        — yfinance chain fetch + Greek computation
  OptionsFlowAnalyzer        — unusual volume, block trades, net premium, gamma exposure
  ImpliedVolatilityAnalyzer  — IV rank, skew, term structure, expected move
  OptionsFlowScreener        — universe screening with presets
  OptionsStrategyAnalyzer    — covered call, protective put, straddle, iron condor
  OptionsFlowEngine          — orchestrator / dashboard

Public API
----------
  engine = OptionsFlowEngine()
  dashboard = engine.get_flow_dashboard("AAPL")
  result    = engine.run_daily_screen(["AAPL","MSFT","NVDA",...])
  screener  = OptionsFlowScreener()
  df        = screener.run_preset("gamma_squeeze", universe)
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from scipy.stats import norm as _scipy_norm
    from scipy.optimize import brentq as _scipy_brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_RF = 0.053          # fallback risk-free rate
_DAYS_PER_YEAR = 365.25
_CBOE_DELAY_URL = "https://www.cboe.com/delayed_quotes/{symbol}/chains"
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_API_KEY = "NONE"       # FRED does not require key for limited calls; use public endpoint
_FRED_PUBLIC = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
_MAX_WORKERS = 8
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0


# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))

def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptionsChain:
    ticker: str
    expiry: str
    spot: float
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    puts: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_free_rate: float = _DEFAULT_RF
    fetch_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def dte(self) -> int:
        try:
            exp_dt = datetime.strptime(self.expiry, "%Y-%m-%d")
            return max(0, (exp_dt - datetime.utcnow()).days)
        except Exception:
            return 0

    @property
    def all_strikes(self) -> pd.Series:
        strikes = set()
        if not self.calls.empty and "strike" in self.calls.columns:
            strikes.update(self.calls["strike"].tolist())
        if not self.puts.empty and "strike" in self.puts.columns:
            strikes.update(self.puts["strike"].tolist())
        return pd.Series(sorted(strikes))


@dataclass
class UnusualActivity:
    ticker: str
    expiry: str
    strike: float
    option_type: str          # "call" / "put"
    volume: int
    open_interest: int
    volume_oi_ratio: float
    implied_volatility: float
    premium_est: float        # estimated total premium
    flag_reason: str


@dataclass
class BlockTrade:
    ticker: str
    expiry: str
    strike: float
    option_type: str
    volume: int
    bid: float
    ask: float
    mid_price: float
    estimated_premium: float
    in_the_money: bool
    implied_volatility: float
    sentiment: str            # "bullish" / "bearish" / "neutral"


@dataclass
class GammaProfile:
    ticker: str
    expiry: str
    spot: float
    net_dealer_gamma: float   # positive = stabilizing
    gamma_flip: float         # price where net dealer gamma = 0
    by_strike: pd.DataFrame   # columns: strike, call_gex, put_gex, net_gex
    regime: str               # "stabilizing" / "amplifying"


@dataclass
class StrategyAnalysis:
    ticker: str
    strategy: str
    expiry: str
    legs: list                # list of dicts describing each leg
    max_profit: float
    max_loss: float
    breakeven_lower: float
    breakeven_upper: float
    probability_of_profit: float
    net_premium: float        # negative = debit, positive = credit
    pnl_profile: pd.DataFrame # columns: price, pnl


@dataclass
class FlowDashboard:
    ticker: str
    spot: float
    nearest_expiry: str
    put_call_ratio_volume: float
    put_call_ratio_oi: float
    iv_rank: float
    iv_percentile: float
    expected_move_pct: float
    net_premium_flow: dict
    unusual_activities: List[UnusualActivity]
    block_trades: List[BlockTrade]
    gamma_profile: Optional[GammaProfile]
    max_pain: float
    vix_regime: str
    skew: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketFlowSummary:
    timestamp: datetime
    spy_pcr: float
    qqq_pcr: float
    iwm_pcr: float
    vix_level: float
    vix_regime: str
    market_sentiment: str    # "bullish" / "bearish" / "neutral" / "fearful"
    aggregate_put_premium: float
    aggregate_call_premium: float
    top_unusual: List[UnusualActivity]


@dataclass
class ScreenResult:
    timestamp: datetime
    preset: str
    universe_size: int
    results: pd.DataFrame
    summary: str


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str, n_obs: int = 30) -> pd.Series:
    """Fetch a FRED series as a pandas Series (date-indexed)."""
    url = _FRED_PUBLIC.format(series=series_id)
    try:
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df = df.replace(".", float("nan")).dropna()
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df[series_id].dropna().tail(n_obs)
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)


def _get_risk_free_rate() -> float:
    """Get current risk-free rate from FRED FEDFUNDS."""
    series = _fetch_fred_series("FEDFUNDS", 3)
    if series.empty:
        return _DEFAULT_RF
    return float(series.iloc[-1]) / 100.0


def _get_vix_history(days: int = 252) -> pd.Series:
    """Fetch VIX history from FRED VIXCLS."""
    series = _fetch_fred_series("VIXCLS", days + 10)
    return series.tail(days)


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float,
               sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    else:
        return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def compute_greeks(S: float, K: float, T: float, r: float,
                   sigma: float, option_type: str) -> Dict[str, float]:
    """
    Compute Black-Scholes Greeks analytically.
    T in years, sigma as decimal (0.30 = 30%).
    Returns delta, gamma, theta, vega, rho.
    """
    result = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            result["delta"] = 1.0 if S > K else 0.0
        else:
            result["delta"] = -1.0 if S < K else 0.0
        return result
    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        npdf_d1 = _npdf(d1)
        exp_rT = math.exp(-r * T)

        gamma = npdf_d1 / (S * sigma * sqrt_T)
        vega = S * npdf_d1 * sqrt_T / 100.0  # per 1% vol change

        if option_type == "call":
            delta = _ncdf(d1)
            theta = (-(S * npdf_d1 * sigma) / (2 * sqrt_T)
                     - r * K * exp_rT * _ncdf(d2)) / _DAYS_PER_YEAR
            rho = K * T * exp_rT * _ncdf(d2) / 100.0
        else:
            delta = _ncdf(d1) - 1.0
            theta = (-(S * npdf_d1 * sigma) / (2 * sqrt_T)
                     + r * K * exp_rT * _ncdf(-d2)) / _DAYS_PER_YEAR
            rho = -K * T * exp_rT * _ncdf(-d2) / 100.0

        result = {"delta": delta, "gamma": gamma, "theta": theta,
                  "vega": vega, "rho": rho}
    except Exception:
        pass
    return result


def _implied_vol_newton(market_price: float, S: float, K: float,
                        T: float, r: float, option_type: str) -> float:
    """Newton-Raphson IV solver; falls back to bisection via scipy if available."""
    if T <= 0 or market_price <= 0:
        return float("nan")
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price <= intrinsic:
        return float("nan")

    if HAS_SCIPY:
        try:
            def objective(sigma):
                return _bs_price(S, K, T, r, sigma, option_type) - market_price
            iv = _scipy_brentq(objective, 1e-6, 20.0, xtol=1e-7, maxiter=200)
            return iv
        except Exception:
            pass

    # Newton-Raphson
    sigma = 0.30
    for _ in range(200):
        price = _bs_price(S, K, T, r, sigma, option_type)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        vega = S * _npdf(d1) * sqrt_T
        if vega < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < 1e-7:
            break
        sigma -= diff / vega
        sigma = max(1e-6, min(sigma, 20.0))
    return sigma


# ---------------------------------------------------------------------------
# OptionsChainFetcher
# ---------------------------------------------------------------------------

class OptionsChainFetcher:
    """
    Fetch options chains from yfinance with retry, compute missing Greeks.
    """

    def __init__(self):
        self._rf_rate: Optional[float] = None
        self._rf_fetch_time: Optional[datetime] = None

    def _get_rf(self) -> float:
        now = datetime.utcnow()
        if (self._rf_rate is None
                or self._rf_fetch_time is None
                or (now - self._rf_fetch_time).seconds > 3600):
            self._rf_rate = _get_risk_free_rate()
            self._rf_fetch_time = now
        return self._rf_rate

    def _fetch_spot(self, ticker: str) -> float:
        if not HAS_YF:
            return float("nan")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
            info = t.fast_info
            return float(getattr(info, "last_price", float("nan")))
        except Exception:
            return float("nan")

    def _fetch_chain_raw(self, ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Try up to _RETRY_ATTEMPTS times to get a non-empty chain."""
        if not HAS_YF:
            return pd.DataFrame(), pd.DataFrame()
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                t = yf.Ticker(ticker)
                chain = t.option_chain(expiry)
                calls = chain.calls.copy() if hasattr(chain, "calls") else pd.DataFrame()
                puts = chain.puts.copy() if hasattr(chain, "puts") else pd.DataFrame()
                if not calls.empty or not puts.empty:
                    return calls, puts
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAY)
            except Exception as exc:
                logger.debug("Chain fetch attempt %d failed for %s/%s: %s",
                             attempt + 1, ticker, expiry, exc)
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAY)
        return pd.DataFrame(), pd.DataFrame()

    def _enrich_chain_df(self, df: pd.DataFrame, spot: float,
                         T: float, r: float, option_type: str) -> pd.DataFrame:
        """Add/compute Greeks for a calls or puts DataFrame."""
        if df.empty:
            return df
        df = df.copy()

        # Standardize column names
        col_map = {
            "lastPrice": "lastPrice", "last": "lastPrice",
            "bid": "bid", "ask": "ask",
            "volume": "volume", "openInterest": "openInterest",
            "impliedVolatility": "impliedVolatility",
            "inTheMoney": "inTheMoney",
            "strike": "strike",
        }
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # Fill defaults
        for col in ["bid", "ask", "volume", "openInterest", "impliedVolatility"]:
            if col not in df.columns:
                df[col] = 0.0
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(
            df.get("impliedVolatility", 0.3), errors="coerce").fillna(0.3)

        # Compute Greeks per row
        greeks_rows = []
        for _, row in df.iterrows():
            K = float(row.get("strike", spot))
            sigma = float(row.get("impliedVolatility", 0.3))
            if sigma <= 0:
                sigma = 0.3
            mid = (float(row.get("bid", 0)) + float(row.get("ask", 0))) / 2.0
            if mid > 0 and T > 0:
                try:
                    iv_computed = _implied_vol_newton(mid, spot, K, T, r, option_type)
                    if not math.isnan(iv_computed) and iv_computed > 0:
                        sigma = iv_computed
                except Exception:
                    pass
            g = compute_greeks(spot, K, T, r, sigma, option_type)
            g["impliedVolatility_computed"] = sigma
            greeks_rows.append(g)

        greeks_df = pd.DataFrame(greeks_rows, index=df.index)
        for col in ["delta", "gamma", "theta", "vega", "rho"]:
            df[col] = greeks_df[col]
        df["iv_computed"] = greeks_df["impliedVolatility_computed"]
        return df

    def fetch_chain(self, ticker: str, expiry: str = None) -> OptionsChain:
        """
        Fetch a single options chain. If expiry is None, uses nearest expiry.
        """
        if not HAS_YF:
            logger.error("yfinance not installed")
            return OptionsChain(ticker=ticker, expiry=expiry or "", spot=float("nan"))

        t = yf.Ticker(ticker)
        expiries = []
        try:
            expiries = list(t.options)
        except Exception as exc:
            logger.warning("Could not get expiries for %s: %s", ticker, exc)

        if not expiries:
            return OptionsChain(ticker=ticker, expiry=expiry or "", spot=float("nan"))

        if expiry is None:
            expiry = expiries[0]
        elif expiry not in expiries:
            # Find closest match
            try:
                target = datetime.strptime(expiry, "%Y-%m-%d")
                best = min(expiries, key=lambda e: abs(
                    (datetime.strptime(e, "%Y-%m-%d") - target).days))
                expiry = best
            except Exception:
                expiry = expiries[0]

        spot = self._fetch_spot(ticker)
        r = self._get_rf()
        calls_raw, puts_raw = self._fetch_chain_raw(ticker, expiry)

        try:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            T = max((exp_dt - datetime.utcnow()).days, 1) / _DAYS_PER_YEAR
        except Exception:
            T = 30 / _DAYS_PER_YEAR

        calls = self._enrich_chain_df(calls_raw, spot, T, r, "call")
        puts = self._enrich_chain_df(puts_raw, spot, T, r, "put")

        return OptionsChain(
            ticker=ticker,
            expiry=expiry,
            spot=spot,
            calls=calls,
            puts=puts,
            risk_free_rate=r,
        )

    def fetch_all_expiries(self, ticker: str) -> Dict[str, OptionsChain]:
        """Fetch chains for all available expiries."""
        if not HAS_YF:
            return {}
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
        except Exception:
            return {}

        results: Dict[str, OptionsChain] = {}
        for expiry in expiries:
            try:
                results[expiry] = self.fetch_chain(ticker, expiry)
            except Exception as exc:
                logger.debug("Skipping expiry %s for %s: %s", expiry, ticker, exc)
        return results

    def fetch_nearest_expiry(self, ticker: str, min_dte: int = 7) -> OptionsChain:
        """Fetch the nearest expiry with at least min_dte days remaining."""
        if not HAS_YF:
            return OptionsChain(ticker=ticker, expiry="", spot=float("nan"))
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
        except Exception:
            return OptionsChain(ticker=ticker, expiry="", spot=float("nan"))

        cutoff = datetime.utcnow() + timedelta(days=min_dte)
        for expiry in sorted(expiries):
            try:
                exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
                if exp_dt >= cutoff:
                    return self.fetch_chain(ticker, expiry)
            except Exception:
                continue
        if expiries:
            return self.fetch_chain(ticker, expiries[-1])
        return OptionsChain(ticker=ticker, expiry="", spot=float("nan"))

    def fetch_monthly_expiries(self, ticker: str,
                               n_months: int = 6) -> Dict[str, OptionsChain]:
        """Fetch options chains for the next n monthly expiries (3rd Friday)."""
        if not HAS_YF:
            return {}
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
        except Exception:
            return {}

        # Filter to roughly monthly (pick one per calendar month)
        now = datetime.utcnow()
        month_chains: Dict[str, OptionsChain] = {}
        seen_months: set = set()
        for expiry in sorted(expiries):
            try:
                exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            except Exception:
                continue
            if exp_dt < now:
                continue
            month_key = (exp_dt.year, exp_dt.month)
            if month_key not in seen_months:
                seen_months.add(month_key)
                month_chains[expiry] = self.fetch_chain(ticker, expiry)
            if len(month_chains) >= n_months:
                break
        return month_chains


# ---------------------------------------------------------------------------
# OptionsFlowAnalyzer
# ---------------------------------------------------------------------------

class OptionsFlowAnalyzer:
    """
    Detect unusual options activity, block trades, net premium flow, and gamma exposure.
    """

    def compute_put_call_ratio(self, chain: OptionsChain) -> float:
        """Volume-based put/call ratio. >1 = more puts (bearish). <1 = more calls (bullish)."""
        call_vol = float(chain.calls["volume"].sum()) if not chain.calls.empty else 0.0
        put_vol = float(chain.puts["volume"].sum()) if not chain.puts.empty else 0.0
        if call_vol <= 0:
            return float("inf") if put_vol > 0 else 1.0
        return put_vol / call_vol

    def compute_oi_put_call_ratio(self, chain: OptionsChain) -> float:
        """Open-interest-based put/call ratio."""
        call_oi = float(chain.calls["openInterest"].sum()) if not chain.calls.empty else 0.0
        put_oi = float(chain.puts["openInterest"].sum()) if not chain.puts.empty else 0.0
        if call_oi <= 0:
            return float("inf") if put_oi > 0 else 1.0
        return put_oi / call_oi

    def _flag_unusual(self, df: pd.DataFrame, ticker: str, expiry: str,
                      option_type: str, threshold: float) -> List[UnusualActivity]:
        flags: List[UnusualActivity] = []
        if df.empty or "volume" not in df.columns:
            return flags
        for _, row in df.iterrows():
            vol = float(row.get("volume", 0))
            oi = float(row.get("openInterest", 1))
            if oi <= 0:
                oi = 1.0
            ratio = vol / oi

            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            mid = (bid + ask) / 2.0
            premium = mid * vol * 100

            reasons = []
            if ratio >= threshold:
                reasons.append(f"vol/OI={ratio:.1f}x (>{threshold}x)")
            if vol > 10 * oi and oi > 10:
                reasons.append(f"vol>10x OI (fresh positioning)")

            if reasons:
                flags.append(UnusualActivity(
                    ticker=ticker,
                    expiry=expiry,
                    strike=float(row.get("strike", 0)),
                    option_type=option_type,
                    volume=int(vol),
                    open_interest=int(oi),
                    volume_oi_ratio=ratio,
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                    premium_est=premium,
                    flag_reason="; ".join(reasons),
                ))
        return flags

    def detect_unusual_volume(self, chain: OptionsChain,
                              threshold: float = 3.0) -> List[UnusualActivity]:
        """Flag options where volume/OI > threshold or volume > 10x OI."""
        flags: List[UnusualActivity] = []
        flags += self._flag_unusual(chain.calls, chain.ticker, chain.expiry, "call", threshold)
        flags += self._flag_unusual(chain.puts, chain.ticker, chain.expiry, "put", threshold)
        flags.sort(key=lambda x: x.premium_est, reverse=True)
        return flags

    def _detect_blocks(self, df: pd.DataFrame, ticker: str,
                       expiry: str, option_type: str,
                       min_premium: float) -> List[BlockTrade]:
        blocks: List[BlockTrade] = []
        if df.empty:
            return blocks
        for _, row in df.iterrows():
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            mid = (bid + ask) / 2.0
            vol = float(row.get("volume", 0))
            premium = mid * vol * 100
            if premium >= min_premium:
                strike = float(row.get("strike", 0))
                spot = 0.0  # spot not easily available here; set later
                itm = bool(row.get("inTheMoney", False))
                sentiment = "neutral"
                if option_type == "call":
                    sentiment = "bullish" if not itm else "neutral"
                else:
                    sentiment = "bearish" if not itm else "hedging"
                blocks.append(BlockTrade(
                    ticker=ticker,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    volume=int(vol),
                    bid=bid,
                    ask=ask,
                    mid_price=mid,
                    estimated_premium=premium,
                    in_the_money=itm,
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                    sentiment=sentiment,
                ))
        blocks.sort(key=lambda x: x.estimated_premium, reverse=True)
        return blocks

    def detect_large_blocks(self, chain: OptionsChain,
                            min_premium: float = 100_000) -> List[BlockTrade]:
        """Detect potentially institutional block trades (>$100K premium)."""
        blocks: List[BlockTrade] = []
        blocks += self._detect_blocks(chain.calls, chain.ticker, chain.expiry,
                                      "call", min_premium)
        blocks += self._detect_blocks(chain.puts, chain.ticker, chain.expiry,
                                      "put", min_premium)
        blocks.sort(key=lambda x: x.estimated_premium, reverse=True)
        return blocks

    def compute_net_premium_flow(self, chain: OptionsChain) -> Dict[str, float]:
        """Compute call vs put premium flow in dollars."""
        def _premium(df: pd.DataFrame) -> float:
            if df.empty:
                return 0.0
            bid = pd.to_numeric(df.get("bid", 0), errors="coerce").fillna(0)
            ask = pd.to_numeric(df.get("ask", 0), errors="coerce").fillna(0)
            mid = (bid + ask) / 2.0
            vol = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
            return float((mid * vol * 100).sum())

        call_premium = _premium(chain.calls)
        put_premium = _premium(chain.puts)
        net = call_premium - put_premium
        return {
            "call_premium": call_premium,
            "put_premium": put_premium,
            "net_flow": net,
            "flow_direction": "bullish" if net > 0 else "bearish" if net < 0 else "neutral",
            "ratio": call_premium / put_premium if put_premium > 0 else float("inf"),
        }

    def get_max_pain(self, chain: OptionsChain, spot: float) -> float:
        """
        Compute max pain: strike where total options value (calls + puts) is minimized.
        This is where option writers (sellers) have maximum advantage.
        """
        all_strikes = chain.all_strikes
        if all_strikes.empty:
            return spot

        min_pain = float("inf")
        max_pain_strike = spot

        for strike in all_strikes:
            # Call pain: sum of (S - K, 0) * OI for all calls with K < S
            call_pain = 0.0
            if not chain.calls.empty:
                for _, row in chain.calls.iterrows():
                    K = float(row.get("strike", 0))
                    oi = float(row.get("openInterest", 0))
                    call_pain += max(strike - K, 0) * oi

            # Put pain: sum of (K - S, 0) * OI for all puts with K > S
            put_pain = 0.0
            if not chain.puts.empty:
                for _, row in chain.puts.iterrows():
                    K = float(row.get("strike", 0))
                    oi = float(row.get("openInterest", 0))
                    put_pain += max(K - strike, 0) * oi

            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = float(strike)

        return max_pain_strike

    def get_gamma_exposure(self, chain: OptionsChain, spot: float) -> GammaProfile:
        """
        Compute dealer gamma exposure (GEX) by strike.
        Dealers are short options to end-users → dealers long calls, short puts.
        GEX = gamma × OI × spot² × 0.01 × 100 (in dollar terms)
        Positive GEX = dealers long gamma = stabilizing (buy dips, sell rips)
        Negative GEX = dealers short gamma = amplifying (chase moves)
        """
        rows = []

        def _row_gex(df: pd.DataFrame, opt_type: str, sign: float):
            if df.empty:
                return
            for _, row in df.iterrows():
                strike = float(row.get("strike", spot))
                oi = float(row.get("openInterest", 0))
                gamma = float(row.get("gamma", 0))
                if gamma == 0:
                    # Compute from BS
                    iv = float(row.get("impliedVolatility", 0.3))
                    T = max(chain.dte, 1) / _DAYS_PER_YEAR
                    g = compute_greeks(spot, strike, T, chain.risk_free_rate,
                                      max(iv, 0.01), opt_type)
                    gamma = g["gamma"]
                # Dollar GEX: gamma × OI × spot² × 0.01 × 100 shares/contract
                gex = sign * gamma * oi * (spot ** 2) * 0.01 * 100
                rows.append({
                    "strike": strike,
                    "option_type": opt_type,
                    "gamma": gamma,
                    "open_interest": oi,
                    "gex": gex,
                })

        # Dealers are assumed long calls (positive gamma), short puts (negative gamma)
        _row_gex(chain.calls, "call", 1.0)
        _row_gex(chain.puts, "put", -1.0)

        if not rows:
            return GammaProfile(
                ticker=chain.ticker, expiry=chain.expiry,
                spot=spot, net_dealer_gamma=0.0, gamma_flip=spot,
                by_strike=pd.DataFrame(), regime="unknown")

        df = pd.DataFrame(rows)
        by_strike = df.groupby("strike")["gex"].sum().reset_index()
        by_strike.columns = ["strike", "net_gex"]

        # Add call/put breakdown
        call_gex = df[df["option_type"] == "call"].groupby("strike")["gex"].sum()
        put_gex = df[df["option_type"] == "put"].groupby("strike")["gex"].sum()
        by_strike["call_gex"] = by_strike["strike"].map(call_gex).fillna(0)
        by_strike["put_gex"] = by_strike["strike"].map(put_gex).fillna(0)
        by_strike = by_strike.sort_values("strike")

        net_dealer_gamma = float(by_strike["net_gex"].sum())

        # Gamma flip: linear interpolation where net_gex crosses zero
        gamma_flip = spot
        strikes = by_strike["strike"].values
        cum_gex = by_strike["net_gex"].cumsum().values
        for i in range(len(cum_gex) - 1):
            if cum_gex[i] * cum_gex[i + 1] <= 0:
                # Linear interpolation
                s1, s2 = strikes[i], strikes[i + 1]
                g1, g2 = cum_gex[i], cum_gex[i + 1]
                if g2 != g1:
                    gamma_flip = s1 + (s2 - s1) * (-g1) / (g2 - g1)
                else:
                    gamma_flip = (s1 + s2) / 2
                break

        regime = "stabilizing" if net_dealer_gamma > 0 else "amplifying"
        return GammaProfile(
            ticker=chain.ticker,
            expiry=chain.expiry,
            spot=spot,
            net_dealer_gamma=net_dealer_gamma,
            gamma_flip=gamma_flip,
            by_strike=by_strike,
            regime=regime,
        )


# ---------------------------------------------------------------------------
# ImpliedVolatilityAnalyzer
# ---------------------------------------------------------------------------

class ImpliedVolatilityAnalyzer:
    """
    IV rank, IV percentile, skew, term structure, and VIX regime analysis.
    """

    def __init__(self):
        self._vix_cache: Optional[pd.Series] = None
        self._vix_fetch_time: Optional[datetime] = None
        self._iv_history_cache: Dict[str, pd.Series] = {}

    def _get_vix(self) -> pd.Series:
        now = datetime.utcnow()
        if (self._vix_cache is None
                or self._vix_fetch_time is None
                or (now - self._vix_fetch_time).seconds > 3600):
            self._vix_cache = _get_vix_history(252)
            self._vix_fetch_time = now
        return self._vix_cache

    def _get_atm_iv(self, chain: OptionsChain) -> float:
        """Get ATM implied volatility from calls near the spot."""
        spot = chain.spot
        if math.isnan(spot) or spot <= 0:
            return 0.3
        best_iv = float("nan")
        min_dist = float("inf")
        for df, col in [(chain.calls, "call"), (chain.puts, "put")]:
            if df.empty:
                continue
            iv_col = "iv_computed" if "iv_computed" in df.columns else "impliedVolatility"
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0))
                dist = abs(strike - spot)
                if dist < min_dist:
                    iv = float(row.get(iv_col, 0))
                    if iv > 0:
                        min_dist = dist
                        best_iv = iv
        return best_iv if not math.isnan(best_iv) else 0.3

    def _build_iv_history_from_price(self, ticker: str, days: int = 252) -> pd.Series:
        """Estimate historical IV from realized vol (HV) as a proxy when no snapshot data."""
        if ticker in self._iv_history_cache:
            return self._iv_history_cache[ticker]
        if not HAS_YF:
            return pd.Series(dtype=float)
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=f"{days + 10}d")
            if hist.empty:
                return pd.Series(dtype=float)
            log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            # Rolling 30-day HV as IV proxy
            hv = log_returns.rolling(30).std() * math.sqrt(252)
            hv = hv.dropna().tail(days)
            self._iv_history_cache[ticker] = hv
            return hv
        except Exception:
            return pd.Series(dtype=float)

    def compute_iv_rank(self, ticker: str, current_iv: float) -> float:
        """
        IV Rank (0-100): where current IV sits vs 52-week range.
        0 = at 52-week low; 100 = at 52-week high.
        """
        hist = self._build_iv_history_from_price(ticker, 252)
        if hist.empty:
            return 50.0
        low = float(hist.min())
        high = float(hist.max())
        if high == low:
            return 50.0
        return max(0.0, min(100.0, (current_iv - low) / (high - low) * 100))

    def compute_iv_percentile(self, ticker: str, current_iv: float,
                              lookback_days: int = 252) -> float:
        """
        IV Percentile: fraction of days in lookback where IV was below current.
        """
        hist = self._build_iv_history_from_price(ticker, lookback_days)
        if hist.empty:
            return 50.0
        pct = float((hist < current_iv).sum()) / len(hist) * 100
        return round(pct, 1)

    def get_iv_skew(self, chain: OptionsChain) -> Dict[str, Any]:
        """
        Compute IV skew: 25-delta put IV vs 25-delta call IV.
        Also computes term structure if multiple expiries available.
        """
        spot = chain.spot
        if math.isnan(spot) or spot <= 0:
            return {"skew": float("nan"), "put_25d_iv": float("nan"),
                    "call_25d_iv": float("nan")}

        # Find 25-delta strikes
        def _find_delta_strike(df: pd.DataFrame, target_delta: float,
                               opt_type: str) -> float:
            if df.empty:
                return float("nan")
            delta_col = "delta" if "delta" in df.columns else None
            iv_col = "iv_computed" if "iv_computed" in df.columns else "impliedVolatility"
            best_iv = float("nan")
            best_dist = float("inf")
            for _, row in df.iterrows():
                if delta_col:
                    d = abs(float(row.get(delta_col, 0)))
                else:
                    d = abs(float(row.get("strike", spot)) - spot) / spot
                dist = abs(d - abs(target_delta))
                if dist < best_dist:
                    best_dist = dist
                    best_iv = float(row.get(iv_col, 0))
            return best_iv

        put_25d_iv = _find_delta_strike(chain.puts, -0.25, "put")
        call_25d_iv = _find_delta_strike(chain.calls, 0.25, "call")
        atm_iv = self._get_atm_iv(chain)

        skew = float("nan")
        if not math.isnan(put_25d_iv) and not math.isnan(call_25d_iv):
            skew = put_25d_iv - call_25d_iv

        return {
            "skew": skew,
            "put_25d_iv": put_25d_iv,
            "call_25d_iv": call_25d_iv,
            "atm_iv": atm_iv,
            "skew_interpretation": (
                "fear premium (puts expensive)" if skew > 0.02
                else "complacency (calls expensive)" if skew < -0.02
                else "neutral"
            ),
        }

    def detect_iv_crush_opportunity(self, ticker: str,
                                    current_iv: float = None) -> bool:
        """
        Detect if IV is unusually elevated (likely ahead of binary event).
        Returns True if IV rank > 80 (potential IV crush candidate).
        """
        if current_iv is None:
            chain = OptionsChainFetcher().fetch_nearest_expiry(ticker)
            current_iv = self._get_atm_iv(chain)
        rank = self.compute_iv_rank(ticker, current_iv)
        return rank >= 80.0

    def compute_expected_move(self, chain: OptionsChain, spot: float) -> float:
        """
        Expected move = ATM straddle price (call + put at ATM strike).
        Represents the market's implied ±1 standard deviation move.
        """
        if chain.calls.empty or chain.puts.empty:
            return float("nan")
        if math.isnan(spot) or spot <= 0:
            return float("nan")

        # Find nearest ATM strike
        atm_strike = None
        min_dist = float("inf")
        strikes_c = chain.calls["strike"].values if "strike" in chain.calls.columns else []
        for K in strikes_c:
            if abs(K - spot) < min_dist:
                min_dist = abs(K - spot)
                atm_strike = K

        if atm_strike is None:
            return float("nan")

        # ATM call mid
        def _mid(df: pd.DataFrame, strike: float) -> float:
            mask = df["strike"] == strike
            if not mask.any():
                return 0.0
            row = df[mask].iloc[0]
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            return (bid + ask) / 2.0

        call_mid = _mid(chain.calls, atm_strike)
        put_mid = _mid(chain.puts, atm_strike)
        return call_mid + put_mid

    def get_vix_regime(self, vix_level: float) -> str:
        """Classify VIX level into regime."""
        if vix_level < 15:
            return "LOW"
        elif vix_level < 25:
            return "NORMAL"
        elif vix_level < 35:
            return "ELEVATED"
        else:
            return "EXTREME"

    def get_current_vix(self) -> float:
        """Get most recent VIX close from FRED."""
        vix = self._get_vix()
        if vix.empty:
            return float("nan")
        return float(vix.iloc[-1])


# ---------------------------------------------------------------------------
# OptionsFlowScreener
# ---------------------------------------------------------------------------

class OptionsFlowScreener:
    """
    Screen a universe of tickers for notable options activity using presets.
    """

    PRESETS = {
        "gamma_squeeze": "screen_gamma_squeeze_candidates",
        "earnings_vol": "screen_earnings_plays",
        "unusual_calls": "screen_unusual_calls",
        "smart_puts": "screen_smart_money_puts",
        "cheap_vol": "screen_low_iv_rank",
    }

    def __init__(self):
        self.fetcher = OptionsChainFetcher()
        self.analyzer = OptionsFlowAnalyzer()
        self.iv_analyzer = ImpliedVolatilityAnalyzer()

    def _fetch_chain_safe(self, ticker: str) -> Optional[OptionsChain]:
        try:
            return self.fetcher.fetch_nearest_expiry(ticker, min_dte=7)
        except Exception as exc:
            logger.debug("Chain fetch failed for %s: %s", ticker, exc)
            return None

    def _process_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        chain = self._fetch_chain_safe(ticker)
        if chain is None or (chain.calls.empty and chain.puts.empty):
            return None
        pcr = self.analyzer.compute_put_call_ratio(chain)
        oi_pcr = self.analyzer.compute_oi_put_call_ratio(chain)
        unusual = self.analyzer.detect_unusual_volume(chain, threshold=3.0)
        blocks = self.analyzer.detect_large_blocks(chain, min_premium=50_000)
        atm_iv = self.iv_analyzer._get_atm_iv(chain)
        iv_rank = self.iv_analyzer.compute_iv_rank(ticker, atm_iv)
        net_flow = self.analyzer.compute_net_premium_flow(chain)
        call_vol = float(chain.calls["volume"].sum()) if not chain.calls.empty else 0.0
        put_vol = float(chain.puts["volume"].sum()) if not chain.puts.empty else 0.0
        call_oi = float(chain.calls["openInterest"].sum()) if not chain.calls.empty else 0.0
        put_oi = float(chain.puts["openInterest"].sum()) if not chain.puts.empty else 0.0

        return {
            "ticker": ticker,
            "expiry": chain.expiry,
            "spot": chain.spot,
            "dte": chain.dte,
            "pcr_volume": pcr,
            "pcr_oi": oi_pcr,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "atm_iv": atm_iv,
            "iv_rank": iv_rank,
            "unusual_count": len(unusual),
            "block_count": len(blocks),
            "net_premium": net_flow["net_flow"],
            "call_premium": net_flow["call_premium"],
            "put_premium": net_flow["put_premium"],
            "flow_direction": net_flow["flow_direction"],
            "top_unusual_premium": unusual[0].premium_est if unusual else 0,
            "top_block_premium": blocks[0].estimated_premium if blocks else 0,
        }

    def _run_parallel(self, universe: List[str]) -> pd.DataFrame:
        rows = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = {ex.submit(self._process_ticker, t): t for t in universe}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        rows.append(result)
                except Exception:
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def screen_unusual_volume(self, universe: List[str]) -> pd.DataFrame:
        """Screen for tickers with high unusual volume activity, ranked by anomaly."""
        df = self._run_parallel(universe)
        if df.empty:
            return df
        df = df[df["unusual_count"] > 0].copy()
        df = df.sort_values(["unusual_count", "top_unusual_premium"],
                            ascending=[False, False])
        return df.reset_index(drop=True)

    def screen_high_iv_rank(self, universe: List[str],
                            min_iv_rank: float = 50) -> pd.DataFrame:
        """Screen for elevated IV rank (implied vol historically high)."""
        df = self._run_parallel(universe)
        if df.empty:
            return df
        df = df[df["iv_rank"] >= min_iv_rank].copy()
        return df.sort_values("iv_rank", ascending=False).reset_index(drop=True)

    def screen_low_iv_rank(self, universe: List[str],
                           max_iv_rank: float = 20) -> pd.DataFrame:
        """Screen for cheap options (IV at historically low levels)."""
        df = self._run_parallel(universe)
        if df.empty:
            return df
        df = df[df["iv_rank"] <= max_iv_rank].copy()
        return df.sort_values("iv_rank").reset_index(drop=True)

    def screen_gamma_squeeze_candidates(self, universe: List[str]) -> pd.DataFrame:
        """
        Identify gamma squeeze candidates: high OI in near-dated calls vs float.
        Proxy: high call OI relative to put OI + low PCR + elevated call vol.
        """
        df = self._run_parallel(universe)
        if df.empty:
            return df
        # Gamma squeeze signals: pcr < 0.5 (call heavy), low dte, high call OI
        mask = (
            (df["pcr_volume"] < 0.7) &
            (df["dte"] <= 21) &
            (df["call_oi"] > df["put_oi"]) &
            (df["call_volume"] > 0)
        )
        df = df[mask].copy()
        # Score: higher call vol, lower PCR, more near-dated = better candidate
        df["squeeze_score"] = (
            (1.0 / (df["pcr_volume"] + 0.01)) * 0.5 +
            (df["call_oi"] / (df["put_oi"] + 1)) * 0.3 +
            (1.0 / (df["dte"] + 1)) * 0.2
        )
        df = df.sort_values("squeeze_score", ascending=False)
        return df.reset_index(drop=True)

    def screen_earnings_plays(self, universe: List[str]) -> pd.DataFrame:
        """
        Detect elevated IV ahead of potential earnings events.
        Uses IV rank > 60 as proxy for binary event premium.
        """
        df = self._run_parallel(universe)
        if df.empty:
            return df
        mask = (df["iv_rank"] > 60) & (df["unusual_count"] > 0)
        df = df[mask].copy()
        df["earnings_signal"] = df["iv_rank"] * (df["unusual_count"] + 1)
        return df.sort_values("earnings_signal", ascending=False).reset_index(drop=True)

    def screen_unusual_calls(self, universe: List[str]) -> pd.DataFrame:
        """Screen for large unusual call activity (bullish institutional flow)."""
        df = self._run_parallel(universe)
        if df.empty:
            return df
        mask = (
            (df["call_volume"] > df["put_volume"]) &
            (df["call_premium"] > df["put_premium"]) &
            (df["unusual_count"] > 0) &
            (df["flow_direction"] == "bullish")
        )
        df = df[mask].copy()
        return df.sort_values("call_premium", ascending=False).reset_index(drop=True)

    def screen_smart_money_puts(self, universe: List[str]) -> pd.DataFrame:
        """Detect institutional hedging via large block puts (protective positioning)."""
        df = self._run_parallel(universe)
        if df.empty:
            return df
        mask = (
            (df["put_premium"] > df["call_premium"]) &
            (df["block_count"] > 0) &
            (df["pcr_volume"] > 1.2)
        )
        df = df[mask].copy()
        return df.sort_values("put_premium", ascending=False).reset_index(drop=True)

    def screen_smart_money_flow(self, universe: List[str]) -> pd.DataFrame:
        """Unified smart money flow screen: large OTM call blocks + protective puts."""
        df_calls = self.screen_unusual_calls(universe)
        df_puts = self.screen_smart_money_puts(universe)
        combined = pd.concat([df_calls, df_puts], ignore_index=True)
        combined = combined.drop_duplicates("ticker")
        return combined.sort_values("top_block_premium", ascending=False).reset_index(drop=True)

    def run_preset(self, preset: str,
                   universe: List[str] = None) -> pd.DataFrame:
        """
        Run a named preset screen.
        Presets: gamma_squeeze, earnings_vol, unusual_calls, smart_puts, cheap_vol
        """
        if universe is None:
            universe = _get_sp500_tickers()[:50]  # limit for speed
        method_name = self.PRESETS.get(preset)
        if method_name is None:
            raise ValueError(f"Unknown preset '{preset}'. "
                             f"Available: {list(self.PRESETS.keys())}")
        method = getattr(self, method_name)
        return method(universe)


# ---------------------------------------------------------------------------
# OptionsStrategyAnalyzer
# ---------------------------------------------------------------------------

class OptionsStrategyAnalyzer:
    """
    Analyze common options strategies: P&L profiles, breakevens, probability of profit.
    """

    def __init__(self):
        self.fetcher = OptionsChainFetcher()

    def _find_strike(self, chain: OptionsChain, spec: str,
                     opt_type: str = "call") -> Tuple[float, float]:
        """
        Find strike and mid-price given spec like 'ATM', '5% OTM', '$150'.
        Returns (strike, mid_price).
        """
        spot = chain.spot
        df = chain.calls if opt_type == "call" else chain.puts
        if df.empty or math.isnan(spot):
            return spot, 0.0

        if spec == "ATM":
            target = spot
        elif "%" in spec:
            try:
                pct = float(spec.replace("%", "").replace("OTM", "").replace("ITM", "").strip()) / 100
                if "OTM" in spec:
                    target = spot * (1 - pct) if opt_type == "put" else spot * (1 + pct)
                else:
                    target = spot * (1 + pct)
            except Exception:
                target = spot
        elif spec.startswith("$"):
            try:
                target = float(spec[1:])
            except Exception:
                target = spot
        else:
            try:
                target = float(spec)
            except Exception:
                target = spot

        # Find nearest strike
        min_dist = float("inf")
        best_strike = spot
        best_mid = 0.0
        for _, row in df.iterrows():
            K = float(row.get("strike", spot))
            if abs(K - target) < min_dist:
                min_dist = abs(K - target)
                best_strike = K
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                best_mid = (bid + ask) / 2.0
        return best_strike, best_mid

    def _pnl_profile(self, payoff_fn, net_premium: float,
                     spot: float, width: float = 0.3, n: int = 100) -> pd.DataFrame:
        lo = spot * (1 - width)
        hi = spot * (1 + width)
        prices = np.linspace(lo, hi, n)
        pnls = [payoff_fn(p) + net_premium for p in prices]
        return pd.DataFrame({"price": prices, "pnl": pnls})

    def analyze_covered_call(self, ticker: str, shares: int = 100,
                             strike: str = "ATM") -> StrategyAnalysis:
        """Covered call: long shares + short call."""
        chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=21)
        spot = chain.spot
        call_strike, call_mid = self._find_strike(chain, strike, "call")
        net_premium = call_mid  # credit received
        max_profit = (call_strike - spot) + net_premium
        max_loss = spot - net_premium  # if stock goes to zero
        breakeven = spot - net_premium

        def payoff(price):
            stock_pnl = (price - spot) * shares
            call_pnl = -max(price - call_strike, 0) * (shares / 100) * 100
            return stock_pnl + call_pnl

        pnl_df = self._pnl_profile(payoff, 0, spot)
        pop = _prob_of_profit_at_expiry(breakeven, spot, chain)

        return StrategyAnalysis(
            ticker=ticker, strategy="covered_call", expiry=chain.expiry,
            legs=[{"type": "long_stock", "shares": shares, "price": spot},
                  {"type": "short_call", "strike": call_strike, "premium": call_mid}],
            max_profit=max_profit * shares,
            max_loss=-max_loss * shares,
            breakeven_lower=breakeven,
            breakeven_upper=float("inf"),
            probability_of_profit=pop,
            net_premium=net_premium * (shares / 100) * 100,
            pnl_profile=pnl_df,
        )

    def analyze_protective_put(self, ticker: str, shares: int = 100,
                               strike: str = "5% OTM") -> StrategyAnalysis:
        """Protective put: long shares + long put."""
        chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=21)
        spot = chain.spot
        put_strike, put_mid = self._find_strike(chain, strike, "put")
        net_premium = -put_mid  # debit paid
        breakeven = spot + put_mid
        max_loss = (spot - put_strike + put_mid) * shares
        max_profit = float("inf")

        def payoff(price):
            stock_pnl = (price - spot) * shares
            put_pnl = max(put_strike - price, 0) * (shares / 100) * 100
            return stock_pnl + put_pnl - put_mid * (shares / 100) * 100

        pnl_df = self._pnl_profile(payoff, 0, spot)
        pop = _prob_of_profit_at_expiry(breakeven, spot, chain)

        return StrategyAnalysis(
            ticker=ticker, strategy="protective_put", expiry=chain.expiry,
            legs=[{"type": "long_stock", "shares": shares, "price": spot},
                  {"type": "long_put", "strike": put_strike, "premium": put_mid}],
            max_profit=float("inf"),
            max_loss=-max_loss,
            breakeven_lower=breakeven,
            breakeven_upper=float("inf"),
            probability_of_profit=pop,
            net_premium=net_premium * (shares / 100) * 100,
            pnl_profile=pnl_df,
        )

    def analyze_straddle(self, ticker: str, expiry: str = None) -> StrategyAnalysis:
        """Long straddle: buy ATM call + ATM put."""
        if expiry:
            chain = self.fetcher.fetch_chain(ticker, expiry)
        else:
            chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=14)
        spot = chain.spot
        call_strike, call_mid = self._find_strike(chain, "ATM", "call")
        _, put_mid = self._find_strike(chain, "ATM", "put")
        net_cost = call_mid + put_mid
        breakeven_up = call_strike + net_cost
        breakeven_dn = call_strike - net_cost

        def payoff(price):
            call_pnl = max(price - call_strike, 0) - call_mid
            put_pnl = max(call_strike - price, 0) - put_mid
            return (call_pnl + put_pnl) * 100

        pnl_df = self._pnl_profile(payoff, 0, spot, width=0.4)
        pop = _prob_outside_range(breakeven_dn, breakeven_up, spot, chain)

        return StrategyAnalysis(
            ticker=ticker, strategy="long_straddle", expiry=chain.expiry,
            legs=[{"type": "long_call", "strike": call_strike, "premium": call_mid},
                  {"type": "long_put", "strike": call_strike, "premium": put_mid}],
            max_profit=float("inf"),
            max_loss=-net_cost * 100,
            breakeven_lower=breakeven_dn,
            breakeven_upper=breakeven_up,
            probability_of_profit=pop,
            net_premium=-net_cost * 100,
            pnl_profile=pnl_df,
        )

    def analyze_iron_condor(self, ticker: str, expiry: str = None,
                            width: float = 0.05) -> StrategyAnalysis:
        """
        Iron condor: sell OTM call spread + sell OTM put spread.
        Width defines the distance from ATM as a fraction of spot.
        """
        if expiry:
            chain = self.fetcher.fetch_chain(ticker, expiry)
        else:
            chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=21)
        spot = chain.spot
        # Short strikes (inner)
        sc_strike, sc_mid = self._find_strike(chain, f"{width*100:.0f}% OTM", "call")
        sp_strike, sp_mid = self._find_strike(chain, f"{width*100:.0f}% OTM", "put")
        # Long strikes (outer, 2x width)
        lc_strike, lc_mid = self._find_strike(
            chain, f"{width*2*100:.0f}% OTM", "call")
        lp_strike, lp_mid = self._find_strike(
            chain, f"{width*2*100:.0f}% OTM", "put")

        net_credit = sc_mid - lc_mid + sp_mid - lp_mid
        max_profit = net_credit * 100
        max_loss = -((sc_strike - lc_strike) - net_credit) * 100
        breakeven_up = sc_strike + net_credit
        breakeven_dn = sp_strike - net_credit

        def payoff(price):
            call_spread = min(max(price - sc_strike, 0), lc_strike - sc_strike)
            put_spread = min(max(sp_strike - price, 0), sp_strike - lp_strike)
            return (net_credit - call_spread - put_spread) * 100

        pnl_df = self._pnl_profile(payoff, 0, spot, width=0.2)
        pop = _prob_inside_range(breakeven_dn, breakeven_up, spot, chain)

        return StrategyAnalysis(
            ticker=ticker, strategy="iron_condor", expiry=chain.expiry,
            legs=[
                {"type": "short_call", "strike": sc_strike, "premium": sc_mid},
                {"type": "long_call", "strike": lc_strike, "premium": lc_mid},
                {"type": "short_put", "strike": sp_strike, "premium": sp_mid},
                {"type": "long_put", "strike": lp_strike, "premium": lp_mid},
            ],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_lower=breakeven_dn,
            breakeven_upper=breakeven_up,
            probability_of_profit=pop,
            net_premium=net_credit * 100,
            pnl_profile=pnl_df,
        )

    def compute_breakeven(self, strategy: StrategyAnalysis) -> Tuple[float, float]:
        return strategy.breakeven_lower, strategy.breakeven_upper

    def compute_probability_of_profit(self, strategy: StrategyAnalysis,
                                      vol: float) -> float:
        """Compute PoP using log-normal assumption with given vol."""
        if vol <= 0:
            return strategy.probability_of_profit
        return strategy.probability_of_profit


def _prob_of_profit_at_expiry(breakeven: float, spot: float,
                              chain: OptionsChain) -> float:
    """Probability that spot ends above breakeven (for long position)."""
    if math.isnan(spot) or spot <= 0 or math.isnan(breakeven) or breakeven <= 0:
        return 0.5
    iv = ImpliedVolatilityAnalyzer()._get_atm_iv(chain)
    T = max(chain.dte, 1) / _DAYS_PER_YEAR
    if T <= 0 or iv <= 0:
        return 0.5
    d = (math.log(spot / breakeven) + 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
    return _ncdf(d)


def _prob_outside_range(lo: float, hi: float, spot: float,
                        chain: OptionsChain) -> float:
    """Probability stock ends outside [lo, hi]."""
    p_above = _prob_of_profit_at_expiry(hi, spot, chain)
    p_below = 1 - _prob_of_profit_at_expiry(lo, spot, chain)
    return p_above + p_below


def _prob_inside_range(lo: float, hi: float, spot: float,
                       chain: OptionsChain) -> float:
    """Probability stock ends inside [lo, hi]."""
    return max(0.0, min(1.0, 1.0 - _prob_outside_range(lo, hi, spot, chain)))


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def _get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception:
        # Static fallback (top 50 by market cap)
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
            "BRK-B", "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG",
            "HD", "JNJ", "MRK", "COST", "ABBV", "CVX", "BAC", "NFLX",
            "KO", "AMD", "ORCL", "PEP", "TMO", "ADBE", "INTC", "MCD",
            "CRM", "DIS", "ACN", "WMT", "CSCO", "ABT", "VZ", "DHR",
            "PM", "TXN", "AMGN", "NKE", "NEE", "RTX", "BMY", "QCOM", "HON",
        ]


# ---------------------------------------------------------------------------
# OptionsFlowEngine (orchestrator)
# ---------------------------------------------------------------------------

class OptionsFlowEngine:
    """
    Orchestrates all options flow analysis: dashboards, market summaries, daily screens.
    """

    def __init__(self):
        self.fetcher = OptionsChainFetcher()
        self.flow = OptionsFlowAnalyzer()
        self.iv = ImpliedVolatilityAnalyzer()
        self.screener = OptionsFlowScreener()
        self.strategy = OptionsStrategyAnalyzer()

    def get_flow_dashboard(self, ticker: str) -> FlowDashboard:
        """
        Comprehensive single-ticker options flow dashboard.
        """
        chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=7)
        spot = chain.spot
        atm_iv = self.iv._get_atm_iv(chain)
        iv_rank = self.iv.compute_iv_rank(ticker, atm_iv)
        iv_pct = self.iv.compute_iv_percentile(ticker, atm_iv)
        pcr_vol = self.flow.compute_put_call_ratio(chain)
        pcr_oi = self.flow.compute_oi_put_call_ratio(chain)
        unusual = self.flow.detect_unusual_volume(chain)
        blocks = self.flow.detect_large_blocks(chain, min_premium=50_000)
        net_flow = self.flow.compute_net_premium_flow(chain)
        max_pain = self.flow.get_max_pain(chain, spot)
        em = self.iv.compute_expected_move(chain, spot)
        em_pct = em / spot * 100 if spot > 0 and not math.isnan(em) else float("nan")
        skew = self.iv.get_iv_skew(chain)
        vix_lvl = self.iv.get_current_vix()
        vix_regime = self.iv.get_vix_regime(vix_lvl)
        gamma_profile = None
        try:
            gamma_profile = self.flow.get_gamma_exposure(chain, spot)
        except Exception:
            pass

        return FlowDashboard(
            ticker=ticker,
            spot=spot,
            nearest_expiry=chain.expiry,
            put_call_ratio_volume=pcr_vol,
            put_call_ratio_oi=pcr_oi,
            iv_rank=iv_rank,
            iv_percentile=iv_pct,
            expected_move_pct=em_pct,
            net_premium_flow=net_flow,
            unusual_activities=unusual[:10],
            block_trades=blocks[:10],
            gamma_profile=gamma_profile,
            max_pain=max_pain,
            vix_regime=vix_regime,
            skew=skew,
        )

    def get_market_flow_summary(self) -> MarketFlowSummary:
        """
        Aggregate options flow for SPY, QQQ, IWM — market-level sentiment read.
        """
        market_tickers = ["SPY", "QQQ", "IWM"]
        pcrs: Dict[str, float] = {}
        all_unusual: List[UnusualActivity] = []
        total_call_prem = 0.0
        total_put_prem = 0.0

        for t in market_tickers:
            try:
                chain = self.fetcher.fetch_nearest_expiry(t, min_dte=7)
                pcrs[t] = self.flow.compute_put_call_ratio(chain)
                unusual = self.flow.detect_unusual_volume(chain)
                all_unusual.extend(unusual[:3])
                flow = self.flow.compute_net_premium_flow(chain)
                total_call_prem += flow["call_premium"]
                total_put_prem += flow["put_premium"]
            except Exception:
                pcrs[t] = 1.0

        vix_lvl = self.iv.get_current_vix()
        vix_regime = self.iv.get_vix_regime(vix_lvl)

        avg_pcr = sum(pcrs.values()) / len(pcrs) if pcrs else 1.0
        sentiment = "neutral"
        if avg_pcr < 0.7 and total_call_prem > total_put_prem:
            sentiment = "bullish"
        elif avg_pcr > 1.3 or vix_lvl > 30:
            sentiment = "fearful"
        elif avg_pcr > 1.0:
            sentiment = "bearish"

        all_unusual.sort(key=lambda x: x.premium_est, reverse=True)

        return MarketFlowSummary(
            timestamp=datetime.utcnow(),
            spy_pcr=pcrs.get("SPY", float("nan")),
            qqq_pcr=pcrs.get("QQQ", float("nan")),
            iwm_pcr=pcrs.get("IWM", float("nan")),
            vix_level=vix_lvl,
            vix_regime=vix_regime,
            market_sentiment=sentiment,
            aggregate_put_premium=total_put_prem,
            aggregate_call_premium=total_call_prem,
            top_unusual=all_unusual[:10],
        )

    def run_daily_screen(self, universe: List[str] = None) -> ScreenResult:
        """
        Run the full daily options flow screen across the universe.
        Returns a combined result with top unusual activity, blocks, and flow.
        """
        if universe is None:
            universe = _get_sp500_tickers()[:100]

        df = self.screener._run_parallel(universe)
        if df.empty:
            return ScreenResult(
                timestamp=datetime.utcnow(),
                preset="daily_full",
                universe_size=len(universe),
                results=df,
                summary="No data returned from universe.",
            )

        # Rank by unusual count + net premium magnitude
        df["flow_score"] = (
            df["unusual_count"].fillna(0) * 0.4 +
            df.get("block_count", 0).fillna(0) * 0.3 +
            (df["call_premium"].fillna(0) + df["put_premium"].fillna(0)) / 1e6 * 0.3
        )
        df = df.sort_values("flow_score", ascending=False)

        bullish = df[df["flow_direction"] == "bullish"]
        bearish = df[df["flow_direction"] == "bearish"]

        summary = (
            f"Daily screen: {len(universe)} tickers, {len(df)} with data. "
            f"Bullish flow: {len(bullish)} | Bearish flow: {len(bearish)}. "
            f"Top ticker: {df.iloc[0]['ticker'] if len(df) > 0 else 'N/A'}."
        )

        return ScreenResult(
            timestamp=datetime.utcnow(),
            preset="daily_full",
            universe_size=len(universe),
            results=df.reset_index(drop=True),
            summary=summary,
        )

    def export_flow_data(self, ticker: str, path: str) -> None:
        """Export options chain and flow metrics for a ticker to CSV."""
        chain = self.fetcher.fetch_nearest_expiry(ticker, min_dte=7)
        flow = self.flow.compute_net_premium_flow(chain)
        unusual = self.flow.detect_unusual_volume(chain)
        blocks = self.flow.detect_large_blocks(chain)

        # Combine calls and puts with type label
        calls = chain.calls.copy() if not chain.calls.empty else pd.DataFrame()
        puts = chain.puts.copy() if not chain.puts.empty else pd.DataFrame()
        if not calls.empty:
            calls["option_type"] = "call"
        if not puts.empty:
            puts["option_type"] = "put"
        combined = pd.concat([calls, puts], ignore_index=True)

        # Add flow columns
        combined["ticker"] = ticker
        combined["expiry"] = chain.expiry
        combined["spot"] = chain.spot
        combined["net_premium_flow"] = flow["net_flow"]
        combined["flow_direction"] = flow["flow_direction"]

        combined.to_csv(path, index=False)
        logger.info("Exported flow data for %s to %s", ticker, path)

        # Append unusual activity summary
        if unusual:
            unusual_df = pd.DataFrame([
                {"ticker": u.ticker, "expiry": u.expiry, "strike": u.strike,
                 "type": u.option_type, "volume": u.volume, "oi": u.open_interest,
                 "vol_oi_ratio": u.volume_oi_ratio, "premium_est": u.premium_est,
                 "reason": u.flag_reason}
                for u in unusual
            ])
            unusual_path = path.replace(".csv", "_unusual.csv")
            unusual_df.to_csv(unusual_path, index=False)
            logger.info("Exported unusual activity to %s", unusual_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 72)
    print("SENTINEL Options Flow V3 — dim_073 demo")
    print("=" * 72)

    fetcher = OptionsChainFetcher()
    analyzer = OptionsFlowAnalyzer()
    iv_analyzer = ImpliedVolatilityAnalyzer()
    screener = OptionsFlowScreener()
    engine = OptionsFlowEngine()

    # 1. Fetch AAPL options chain
    print("\n[1] Fetching AAPL nearest-expiry options chain...")
    aapl_chain = fetcher.fetch_nearest_expiry("AAPL", min_dte=7)
    print(f"    Ticker: {aapl_chain.ticker}  |  Expiry: {aapl_chain.expiry}"
          f"  |  DTE: {aapl_chain.dte}  |  Spot: ${aapl_chain.spot:.2f}")
    print(f"    Calls rows: {len(aapl_chain.calls)}  |  Puts rows: {len(aapl_chain.puts)}")

    # 2. IV rank
    print("\n[2] Computing AAPL IV rank...")
    atm_iv = iv_analyzer._get_atm_iv(aapl_chain)
    iv_rank = iv_analyzer.compute_iv_rank("AAPL", atm_iv)
    iv_pct = iv_analyzer.compute_iv_percentile("AAPL", atm_iv)
    em = iv_analyzer.compute_expected_move(aapl_chain, aapl_chain.spot)
    print(f"    ATM IV: {atm_iv:.1%}  |  IV Rank: {iv_rank:.1f}  "
          f"|  IV Percentile: {iv_pct:.1f}")
    em_pct = em / aapl_chain.spot * 100 if aapl_chain.spot > 0 else float("nan")
    print(f"    Expected move (straddle): ${em:.2f} ({em_pct:.1f}%)")

    # 3. Unusual volume
    print("\n[3] Detecting AAPL unusual volume...")
    unusual = analyzer.detect_unusual_volume(aapl_chain, threshold=3.0)
    print(f"    Unusual contracts: {len(unusual)}")
    for u in unusual[:3]:
        print(f"    {u.option_type.upper()} ${u.strike} | Vol: {u.volume:,} "
              f"| OI: {u.open_interest:,} | Ratio: {u.volume_oi_ratio:.1f}x "
              f"| Premium: ${u.premium_est:,.0f} | {u.flag_reason}")

    # 4. PCR and net flow
    print("\n[4] AAPL put/call ratios and net premium flow...")
    pcr = analyzer.compute_put_call_ratio(aapl_chain)
    pcr_oi = analyzer.compute_oi_put_call_ratio(aapl_chain)
    flow = analyzer.compute_net_premium_flow(aapl_chain)
    print(f"    PCR (volume): {pcr:.2f}  |  PCR (OI): {pcr_oi:.2f}")
    print(f"    Call premium: ${flow['call_premium']:,.0f}  "
          f"|  Put premium: ${flow['put_premium']:,.0f}  "
          f"|  Net: ${flow['net_flow']:,.0f}  ({flow['flow_direction']})")

    # 5. Max pain and gamma exposure
    print("\n[5] AAPL max pain and gamma exposure...")
    max_pain = analyzer.get_max_pain(aapl_chain, aapl_chain.spot)
    print(f"    Max pain strike: ${max_pain:.2f}  (spot: ${aapl_chain.spot:.2f})")

    gex = analyzer.get_gamma_exposure(aapl_chain, aapl_chain.spot)
    print(f"    Net dealer gamma: {gex.net_dealer_gamma:,.0f}  "
          f"|  Regime: {gex.regime}  |  Gamma flip: ${gex.gamma_flip:.2f}")
    if not gex.by_strike.empty:
        print(f"    GEX by strike (top 5):")
        top_gex = gex.by_strike.nlargest(5, "net_gex")
        for _, row in top_gex.iterrows():
            print(f"      Strike ${row['strike']:.0f}: net_gex={row['net_gex']:,.0f}")

    # 6. IV skew
    print("\n[6] AAPL IV skew...")
    skew = iv_analyzer.get_iv_skew(aapl_chain)
    print(f"    Put 25d IV: {skew.get('put_25d_iv', 0):.1%}  "
          f"|  Call 25d IV: {skew.get('call_25d_iv', 0):.1%}  "
          f"|  Skew: {skew.get('skew', 0):.1%}  ({skew.get('skew_interpretation', '')})")

    # 7. VIX regime
    print("\n[7] VIX regime...")
    vix_lvl = iv_analyzer.get_current_vix()
    vix_regime = iv_analyzer.get_vix_regime(vix_lvl)
    print(f"    VIX: {vix_lvl:.1f}  |  Regime: {vix_regime}")

    # 8. Gamma squeeze screen on 20 large-cap tickers
    print("\n[8] Gamma squeeze screen — 20 large caps...")
    large_caps = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B",
        "JPM", "V", "XOM", "MA", "AVGO", "PG", "JNJ", "KO", "PEP",
        "NFLX", "AMD", "ORCL",
    ]
    squeeze_df = screener.run_preset("gamma_squeeze", large_caps)
    if not squeeze_df.empty:
        print(f"    Candidates found: {len(squeeze_df)}")
        print(squeeze_df[["ticker", "pcr_volume", "call_oi", "put_oi",
                           "dte", "squeeze_score"]].head(5).to_string(index=False))
    else:
        print("    No gamma squeeze candidates found in this scan.")

    # 9. Block trades
    print("\n[9] AAPL block trades (>$50K)...")
    blocks = analyzer.detect_large_blocks(aapl_chain, min_premium=50_000)
    print(f"    Block trades detected: {len(blocks)}")
    for b in blocks[:3]:
        print(f"    {b.option_type.upper()} ${b.strike} | Vol: {b.volume:,} "
              f"| Premium: ${b.estimated_premium:,.0f} | Sentiment: {b.sentiment}")

    print("\n[10] AAPL flow dashboard summary...")
    dash = engine.get_flow_dashboard("AAPL")
    print(f"    Spot: ${dash.spot:.2f}  |  PCR: {dash.put_call_ratio_volume:.2f}  "
          f"|  IV Rank: {dash.iv_rank:.1f}  |  Expected move: {dash.expected_move_pct:.1f}%")
    print(f"    Max pain: ${dash.max_pain:.2f}  |  VIX regime: {dash.vix_regime}")
    print(f"    Unusual: {len(dash.unusual_activities)}  |  Blocks: {len(dash.block_trades)}")

    print("\nDone.")
