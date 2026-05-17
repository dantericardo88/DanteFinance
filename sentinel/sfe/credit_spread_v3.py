"""
Credit spread analysis v3 — Merton structural model, KMV Distance-to-Default,
CDS proxy construction, spread term structure, regime classification, and
sector-level credit analytics.

Dimension #039 — Credit spread analysis / Merton model (target: 9).

Free data sources only:
  - FRED CSV (no API key): ICE BofA OAS indices, treasury yields
  - EDGAR XBRL: per-company debt figures
  - yfinance: equity market cap, realized volatility

Architecture:
  MertonModel              — full Merton (1974) structural credit model
  KMVDistanceToDefault     — Moody's KMV extension, EDF classification
  CreditSpreadBuilder      — FRED spread curves, regime detection
  CDSSpreadsProxy          — synthetic CDS from bond/equity data
  CreditSectorAnalyzer     — sector-level OAS history and z-scores
  CreditRiskEngine         — orchestrator: issuer profiles, dashboards, screener
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Literal, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

# scipy guarded — graceful fallback to Newton iteration when unavailable
try:
    from scipy.optimize import fsolve as _scipy_fsolve
    from scipy.stats import norm as _norm
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
EDGAR_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"
EDGAR_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/3.0 richard.porras@realempanada.com",
    "Accept": "application/json, text/html, */*",
}

# FRED OAS series — all free via CSV endpoint
FRED_OAS_SERIES: Dict[str, str] = {
    "IG":    "BAMLC0A0CM",      # ICE BofA US Corporate Index OAS
    "AAA":   "BAMLC0A1CAAA",    # AAA OAS
    "AA":    "BAMLC0A2CAA",     # AA OAS
    "A":     "BAMLC0A3CA",      # A OAS
    "BBB":   "BAMLC0A4CBBB",    # BBB OAS
    "HY":    "BAMLH0A0HYM2",    # US High Yield OAS
    "BB":    "BAMLH0A1HYBB",    # BB OAS
    "B":     "BAMLH0A2HYB",     # B OAS
    "CCC":   "BAMLH0A3HYC",     # CCC OAS
    "EM":    "BAMLEMCBPIOAS",   # Emerging Market Corporate OAS
}

FRED_TERM_SERIES: Dict[str, str] = {
    "IG_1_3Y":    "BAMLC1A0C13Y",     # 1-3 year IG
    "IG_3_5Y":    "BAMLC2A0C35Y",     # 3-5 year IG
    "IG_5_7Y":    "BAMLC3A0C57Y",     # 5-7 year IG
    "IG_7_10Y":   "BAMLC4A0C710Y",    # 7-10 year IG
    "IG_10_15Y":  "BAMLC7A0C1015Y",   # 10-15 year IG
}

# Typical spread benchmarks (bps) by rating — fallback when FRED unavailable
TYPICAL_SPREADS_BPS: Dict[str, float] = {
    "AAA": 40,
    "AA":  55,
    "A":   80,
    "BBB": 140,
    "BB":  280,
    "B":   450,
    "CCC": 1000,
    "HY":  350,
    "IG":  120,
}

# KMV credit quality thresholds (Distance-to-Default)
DD_THRESHOLDS = {
    "INVESTMENT_GRADE":   5.0,
    "BBB_EQUIVALENT":     3.0,
    "SPECULATIVE":        2.0,
    "DISTRESSED":         1.0,
}

# Loss Given Default assumption
LGD_DEFAULT = 0.60   # 60% LGD (40% recovery)

# ---------------------------------------------------------------------------
# Normal distribution helpers (scipy-optional)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — uses scipy when available, math.erfc otherwise."""
    if _SCIPY:
        return float(_norm.cdf(x))
    # Abramowitz & Stegun approximation
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    if _SCIPY:
        return float(_norm.pdf(x))
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MertonResult:
    ticker: str
    equity_value: float          # E (market cap, $M)
    equity_vol: float            # σ_E (annualised)
    firm_value: float            # V (asset value, $M)
    asset_vol: float             # σ_V
    debt_face: float             # D (face value of debt, $M)
    time_horizon: float          # T (years)
    risk_free_rate: float        # r
    d1: float
    d2: float
    distance_to_default: float   # DD (physical)
    risk_neutral_pd: float       # N(-d2)
    physical_pd: float           # N(-DD)
    credit_spread_bps: float     # implied credit spread in bps
    credit_quality: str
    leverage_ratio: float        # D / V
    as_of: date = field(default_factory=date.today)
    error: Optional[str] = None


@dataclass
class KMVResult:
    ticker: str
    equity_value: float
    equity_vol: float
    debt_short: float
    debt_long: float
    default_point: float         # DP = STD + 0.5 × LTD
    firm_value: float
    asset_vol: float
    kmv_dd: float
    edf: float                   # Expected Default Frequency (%)
    credit_quality: str
    as_of: date = field(default_factory=date.today)
    error: Optional[str] = None


@dataclass
class IssuerCreditProfile:
    ticker: str
    company_name: str
    merton: Optional[MertonResult]
    kmv: Optional[KMVResult]
    implied_rating: str
    credit_spread_estimate_bps: float
    comparable_oas_bps: float        # FRED spread for implied rating
    spread_premium_bps: float        # issuer spread - index spread
    distress_flag: bool
    as_of: date = field(default_factory=date.today)


# ---------------------------------------------------------------------------
# Merton Structural Credit Model
# ---------------------------------------------------------------------------

class MertonModel:
    """
    Full Merton (1974) structural credit model.

    Equity is modelled as a European call option on firm assets:
        E = V·N(d1) - D·e^(-rT)·N(d2)
        d1 = [ln(V/D) + (r + σV²/2)·T] / (σV·√T)
        d2 = d1 - σV·√T

    Two unknowns (V, σV) are solved from two equations:
        1) E = Merton_call(V, D, T, r, σV)
        2) σE·E = N(d1)·σV·V   (Ito's lemma / equity vol constraint)
    """

    def _merton_call(
        self, V: float, D: float, T: float, r: float, sigma_V: float
    ) -> Tuple[float, float, float]:
        """Return (call_value, d1, d2)."""
        if T <= 0 or sigma_V <= 1e-9 or V <= 0:
            return 0.0, -99.0, -99.0
        d1 = (math.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
        d2 = d1 - sigma_V * math.sqrt(T)
        call = V * _norm_cdf(d1) - D * math.exp(-r * T) * _norm_cdf(d2)
        return call, d1, d2

    def _system_equations(
        self,
        params: Tuple[float, float],
        equity_value: float,
        equity_vol: float,
        debt_face: float,
        T: float,
        r: float,
    ) -> Tuple[float, float]:
        """System of two equations for (V, σV)."""
        V, sigma_V = params
        if V <= 0 or sigma_V <= 1e-9:
            return (1e9, 1e9)
        call, d1, _ = self._merton_call(V, debt_face, T, r, sigma_V)
        eq1 = call - equity_value
        eq2 = _norm_cdf(d1) * sigma_V * V - equity_vol * equity_value
        return (eq1, eq2)

    def solve_firm_value_vol(
        self,
        equity_value: float,
        equity_vol: float,
        debt_face: float,
        T: float = 1.0,
        r: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Solve for firm value V and asset volatility σV simultaneously.

        Returns (V, sigma_V).  Falls back to Newton iteration when scipy
        is not available.
        """
        if equity_value <= 0 or debt_face <= 0:
            raise ValueError("equity_value and debt_face must be positive")

        # Initial guess: V ≈ E + D, σV ≈ σE × E/(E+D)
        V0 = equity_value + debt_face
        sV0 = max(equity_vol * equity_value / V0, 0.01)
        x0 = [V0, sV0]

        if _SCIPY:
            def _f(x):
                return list(self._system_equations((x[0], x[1]), equity_value, equity_vol, debt_face, T, r))
            sol = _scipy_fsolve(_f, x0, full_output=True)
            V_sol, sV_sol = float(sol[0][0]), float(sol[0][1])
        else:
            # Simple Newton iteration (damped)
            V_sol, sV_sol = float(x0[0]), float(x0[1])
            for _ in range(200):
                f1, f2 = self._system_equations((V_sol, sV_sol), equity_value, equity_vol, debt_face, T, r)
                if abs(f1) < 1e-4 and abs(f2) < 1e-6:
                    break
                # Numerical Jacobian
                dV = max(V_sol * 1e-5, 1.0)
                ds = max(sV_sol * 1e-5, 1e-7)
                f1p, f2p = self._system_equations((V_sol + dV, sV_sol), equity_value, equity_vol, debt_face, T, r)
                f1q, f2q = self._system_equations((V_sol, sV_sol + ds), equity_value, equity_vol, debt_face, T, r)
                j11 = (f1p - f1) / dV
                j12 = (f1q - f1) / ds
                j21 = (f2p - f2) / dV
                j22 = (f2q - f2) / ds
                det = j11 * j22 - j12 * j21
                if abs(det) < 1e-30:
                    break
                dv = (f1 * j22 - f2 * j12) / det
                dsv = (f2 * j11 - f1 * j21) / det
                step = 1.0
                V_sol = max(V_sol - step * dv, equity_value * 0.5)
                sV_sol = max(sV_sol - step * dsv, 1e-4)

        # Sanity checks
        V_sol = max(V_sol, equity_value)
        sV_sol = max(min(sV_sol, 5.0), 0.001)
        return V_sol, sV_sol

    def compute_distance_to_default(
        self,
        V: float,
        D: float,
        mu: float,
        sigma_V: float,
        T: float = 1.0,
    ) -> float:
        """
        Physical Distance-to-Default.
        DD = [ln(V/D) + (μ - σV²/2)·T] / (σV·√T)
        μ is the expected return on assets (r + equity_premium).
        """
        if D <= 0 or sigma_V <= 0 or V <= 0 or T <= 0:
            return 0.0
        dd = (math.log(V / D) + (mu - 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
        return dd

    def compute_default_probability(
        self, dd: float, d2: float
    ) -> Tuple[float, float]:
        """
        Returns (risk_neutral_pd, physical_pd).
        Risk-neutral PD = N(-d2);  Physical PD = N(-DD).
        """
        rn_pd = _norm_cdf(-d2)
        phys_pd = _norm_cdf(-dd)
        return rn_pd, phys_pd

    def compute_credit_spread(
        self,
        V: float,
        D: float,
        T: float,
        r: float,
        sigma_V: float,
    ) -> float:
        """
        Merton-implied credit spread (annualised, in decimal).
        spread = -ln(N(d2) + (V/(D·e^(-rT)))·N(-d1)) / T  - r
        Simplified floor: spread ≥ PD × LGD / T.
        Returns spread in bps.
        """
        if T <= 0 or D <= 0 or V <= 0:
            return 0.0
        _, d1, d2 = self._merton_call(V, D, T, r, sigma_V)
        nd2 = _norm_cdf(d2)
        nd1_neg = _norm_cdf(-d1)
        ratio = V / (D * math.exp(-r * T))
        inner = nd2 + ratio * nd1_neg
        if inner <= 0:
            inner = 1e-10
        merton_spread = -math.log(inner) / T - r
        # Simplified PD×LGD floor
        rn_pd = _norm_cdf(-d2)
        simple_spread = rn_pd * LGD_DEFAULT / T
        spread = max(merton_spread, simple_spread, 0.0)
        return spread * 10_000  # bps

    def _classify_credit_quality(self, dd: float) -> str:
        if dd > DD_THRESHOLDS["INVESTMENT_GRADE"]:
            return "INVESTMENT_GRADE"
        if dd > DD_THRESHOLDS["BBB_EQUIVALENT"]:
            return "BBB_EQUIVALENT"
        if dd > DD_THRESHOLDS["SPECULATIVE"]:
            return "SPECULATIVE"
        if dd > DD_THRESHOLDS["DISTRESSED"]:
            return "DISTRESSED"
        return "DEFAULT_IMMINENT"

    def run_merton_analysis(self, ticker: str, T: float = 1.0, r: float = 0.05) -> MertonResult:
        """
        Full Merton analysis for a single ticker.
        Fetches equity market cap and vol from yfinance,
        debt from EDGAR XBRL.
        """
        base = MertonResult(
            ticker=ticker, equity_value=0, equity_vol=0, firm_value=0,
            asset_vol=0, debt_face=0, time_horizon=T, risk_free_rate=r,
            d1=0, d2=0, distance_to_default=0, risk_neutral_pd=0,
            physical_pd=0, credit_spread_bps=0, credit_quality="UNKNOWN",
            leverage_ratio=0,
        )
        try:
            eq_val, eq_vol = _fetch_equity_data(ticker)
            if eq_val is None or eq_vol is None:
                base.error = "yfinance data unavailable"
                return base
            debt = _fetch_edgar_debt(ticker)
            if debt is None or debt <= 0:
                # Fallback: estimate debt from balance sheet via yfinance info
                debt = _fetch_yf_debt(ticker)
            if debt is None or debt <= 0:
                base.error = "Debt data unavailable"
                return base

            eq_val_m = eq_val / 1e6   # convert to $M
            debt_m   = debt / 1e6

            V, sigma_V = self.solve_firm_value_vol(eq_val_m, eq_vol, debt_m, T, r)
            _, d1, d2 = self._merton_call(V, debt_m, T, r, sigma_V)
            mu = r + 0.05  # assume 5% equity premium on assets
            dd = self.compute_distance_to_default(V, debt_m, mu, sigma_V, T)
            rn_pd, phys_pd = self.compute_default_probability(dd, d2)
            cs_bps = self.compute_credit_spread(V, debt_m, T, r, sigma_V)
            quality = self._classify_credit_quality(dd)

            return MertonResult(
                ticker=ticker,
                equity_value=eq_val_m,
                equity_vol=eq_vol,
                firm_value=V,
                asset_vol=sigma_V,
                debt_face=debt_m,
                time_horizon=T,
                risk_free_rate=r,
                d1=d1,
                d2=d2,
                distance_to_default=dd,
                risk_neutral_pd=rn_pd,
                physical_pd=phys_pd,
                credit_spread_bps=cs_bps,
                credit_quality=quality,
                leverage_ratio=debt_m / V if V > 0 else 0,
            )
        except Exception as exc:
            logger.warning("Merton analysis failed for %s: %s", ticker, exc)
            base.error = str(exc)
            return base

    def run_universe_merton(self, tickers: List[str], T: float = 1.0, r: float = 0.05) -> pd.DataFrame:
        """Run Merton analysis on a list of tickers in parallel. Returns DataFrame."""
        results = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self.run_merton_analysis, t, T, r): t for t in tickers}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.warning("Universe Merton error for %s: %s", futures[fut], exc)
        if not results:
            return pd.DataFrame()
        rows = []
        for r_ in results:
            rows.append({
                "ticker": r_.ticker,
                "equity_value_m": round(r_.equity_value, 1),
                "firm_value_m": round(r_.firm_value, 1),
                "debt_face_m": round(r_.debt_face, 1),
                "asset_vol": round(r_.asset_vol, 4),
                "equity_vol": round(r_.equity_vol, 4),
                "d1": round(r_.d1, 4),
                "d2": round(r_.d2, 4),
                "distance_to_default": round(r_.distance_to_default, 4),
                "risk_neutral_pd": round(r_.risk_neutral_pd, 6),
                "physical_pd": round(r_.physical_pd, 6),
                "credit_spread_bps": round(r_.credit_spread_bps, 1),
                "leverage_ratio": round(r_.leverage_ratio, 4),
                "credit_quality": r_.credit_quality,
                "error": r_.error,
            })
        df = pd.DataFrame(rows).sort_values("distance_to_default", ascending=True)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KMV Distance-to-Default
# ---------------------------------------------------------------------------

class KMVDistanceToDefault:
    """
    Moody's KMV extension to Merton.

    Key difference: uses an empirically calibrated Default Point
        DP = STD + 0.5 × LTD
    instead of total face debt, reflecting KMV's finding that
    defaults typically cluster when asset value falls to this level.
    """

    _merton = MertonModel()

    def compute_kmv_dd(
        self,
        equity_value: float,
        equity_vol: float,
        debt_short: float,
        debt_long: float,
        r: float = 0.05,
        T: float = 1.0,
        ticker: str = "UNKNOWN",
    ) -> KMVResult:
        """
        Compute KMV Distance-to-Default.

        equity_value: market cap in $M
        equity_vol:   annualised equity volatility
        debt_short:   short-term debt in $M
        debt_long:    long-term debt in $M
        """
        dp = debt_short + 0.5 * debt_long   # KMV Default Point
        total_debt = debt_short + debt_long

        try:
            V, sigma_V = self._merton.solve_firm_value_vol(equity_value, equity_vol, dp, T, r)
            mu = r + 0.05
            kmv_dd = self._merton.compute_distance_to_default(V, dp, mu, sigma_V, T)
            edf = self.compute_expected_default_frequency(kmv_dd)
            quality = self.classify_credit_quality(kmv_dd)
            return KMVResult(
                ticker=ticker,
                equity_value=equity_value,
                equity_vol=equity_vol,
                debt_short=debt_short,
                debt_long=debt_long,
                default_point=dp,
                firm_value=V,
                asset_vol=sigma_V,
                kmv_dd=kmv_dd,
                edf=edf,
                credit_quality=quality,
            )
        except Exception as exc:
            return KMVResult(
                ticker=ticker,
                equity_value=equity_value,
                equity_vol=equity_vol,
                debt_short=debt_short,
                debt_long=debt_long,
                default_point=dp,
                firm_value=0,
                asset_vol=0,
                kmv_dd=0,
                edf=1.0,
                credit_quality="UNKNOWN",
                error=str(exc),
            )

    def compute_expected_default_frequency(self, dd: float) -> float:
        """
        EDF as fraction [0, 1].
        KMV empirical mapping: approximately N(-DD) with a floor for very
        high DD values.  Historical data shows EDF ≈ 0.1% for DD=4,
        EDF ≈ 5% for DD=2, EDF ≈ 20% for DD=1.
        """
        raw = _norm_cdf(-dd)
        # KMV empirical correction: EDF < N(-DD) for high credit quality
        # Simple linear adjustment based on public KMV research
        if dd > 4:
            return max(raw * 0.3, 0.0001)   # high quality: much lower than theoretical
        if dd > 2:
            return raw * 0.7
        return raw   # low DD: N(-DD) reasonable approximation

    def classify_credit_quality(self, dd: float) -> str:
        """
        Credit quality bands from DD value.
        DD > 5 → INVESTMENT_GRADE
        3-5    → BBB_EQUIVALENT
        2-3    → SPECULATIVE
        1-2    → DISTRESSED
        <1     → DEFAULT_IMMINENT
        """
        if dd > 5:
            return "INVESTMENT_GRADE"
        if dd > 3:
            return "BBB_EQUIVALENT"
        if dd > 2:
            return "SPECULATIVE"
        if dd > 1:
            return "DISTRESSED"
        return "DEFAULT_IMMINENT"

    def run_for_ticker(self, ticker: str, r: float = 0.05) -> KMVResult:
        """Convenience: fetch data from yfinance/EDGAR and compute KMV DD."""
        try:
            eq_val, eq_vol = _fetch_equity_data(ticker)
            if eq_val is None:
                return KMVResult(ticker=ticker, equity_value=0, equity_vol=0,
                                 debt_short=0, debt_long=0, default_point=0,
                                 firm_value=0, asset_vol=0, kmv_dd=0, edf=1.0,
                                 credit_quality="UNKNOWN", error="No equity data")
            eq_val_m = eq_val / 1e6
            std, ltd = _fetch_edgar_debt_split(ticker)
            if std is None:
                std, ltd = _fetch_yf_debt_split(ticker)
            std = (std or 0) / 1e6
            ltd = (ltd or 0) / 1e6
            return self.compute_kmv_dd(eq_val_m, eq_vol, std, ltd, r, ticker=ticker)
        except Exception as exc:
            return KMVResult(ticker=ticker, equity_value=0, equity_vol=0,
                             debt_short=0, debt_long=0, default_point=0,
                             firm_value=0, asset_vol=0, kmv_dd=0, edf=1.0,
                             credit_quality="UNKNOWN", error=str(exc))


# ---------------------------------------------------------------------------
# Credit Spread Builder
# ---------------------------------------------------------------------------

class CreditSpreadBuilder:
    """
    Build synthetic credit spread curves from FRED free data.
    Covers the full rating stack (AAA → CCC) and term structure (1-3Y → 10-15Y).
    """

    def __init__(self, cache_ttl_hours: int = 4):
        self._cache: Dict[str, Tuple[pd.Series, float]] = {}  # series_id → (series, ts)
        self._ttl = cache_ttl_hours * 3600

    def _fetch_fred_series(self, series_id: str) -> pd.Series:
        """Fetch a single FRED series via CSV (no API key needed)."""
        cached, ts = self._cache.get(series_id, (None, 0))
        if cached is not None and (time.time() - ts) < self._ttl:
            return cached
        url = f"{FRED_CSV_BASE}?id={series_id}"
        try:
            with httpx.Client(timeout=30, headers=_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            s.name = series_id
            self._cache[series_id] = (s, time.time())
            return s
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float, name=series_id)

    def fetch_fred_credit_spreads(self) -> pd.DataFrame:
        """
        Fetch all OAS series from FRED.
        Returns a DataFrame with columns = rating codes, index = date.
        """
        frames = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self._fetch_fred_series, sid): key
                       for key, sid in FRED_OAS_SERIES.items()}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    s = fut.result()
                    if not s.empty:
                        frames[key] = s
                except Exception as exc:
                    logger.warning("Spread fetch error for %s: %s", key, exc)
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames).sort_index()
        return df

    def fetch_term_structure(self) -> pd.DataFrame:
        """Fetch IG spread term structure (1-3Y, 3-5Y, 5-7Y, 7-10Y, 10-15Y)."""
        frames = {}
        for key, sid in FRED_TERM_SERIES.items():
            s = self._fetch_fred_series(sid)
            if not s.empty:
                frames[key] = s
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).sort_index()

    def build_rating_spread_curve(self, date_str: Optional[str] = None) -> Dict[str, float]:
        """
        Spread by rating (in bps) for a given date.
        Falls back to typical spreads when FRED data is unavailable.
        """
        df = self.fetch_fred_credit_spreads()
        if df.empty:
            logger.info("Using typical spread fallbacks (FRED unavailable)")
            return dict(TYPICAL_SPREADS_BPS)

        if date_str:
            target = pd.Timestamp(date_str)
            # Use most recent observation on or before target
            df = df[df.index <= target]

        if df.empty:
            return dict(TYPICAL_SPREADS_BPS)

        latest = df.iloc[-1]
        curve: Dict[str, float] = {}
        for rating, typical in TYPICAL_SPREADS_BPS.items():
            if rating in latest and not pd.isna(latest[rating]):
                curve[rating] = float(latest[rating])
            else:
                curve[rating] = typical

        # Fill any remaining gaps with interpolation
        ordered = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        for i, r in enumerate(ordered):
            if r not in curve or pd.isna(curve.get(r)):
                neighbors = [(j, ordered[j]) for j in [i - 1, i + 1] if 0 <= j < len(ordered)]
                vals = [curve[ordered[j]] for j, _ in neighbors if ordered[j] in curve]
                curve[r] = float(np.mean(vals)) if vals else TYPICAL_SPREADS_BPS.get(r, 100)
        return curve

    def compute_spread_percentile(
        self,
        rating: str,
        current_spread: float,
        lookback_years: int = 10,
    ) -> float:
        """
        Current spread percentile vs historical range.
        Returns value in [0, 100].
        """
        sid = FRED_OAS_SERIES.get(rating)
        if sid is None:
            return 50.0
        s = self._fetch_fred_series(sid)
        if s.empty:
            return 50.0
        cutoff = s.index[-1] - pd.DateOffset(years=lookback_years)
        hist = s[s.index >= cutoff].dropna()
        if hist.empty:
            return 50.0
        pct = float((hist < current_spread).mean() * 100)
        return round(pct, 1)

    def get_spread_z_score(self, rating: str, window_years: int = 5) -> Optional[float]:
        """Z-score of current spread vs rolling mean/std."""
        sid = FRED_OAS_SERIES.get(rating)
        if sid is None:
            return None
        s = self._fetch_fred_series(sid)
        if s.empty or len(s) < 252:
            return None
        cutoff = s.index[-1] - pd.DateOffset(years=window_years)
        hist = s[s.index >= cutoff].dropna()
        if len(hist) < 50:
            return None
        mu = hist.mean()
        sigma = hist.std()
        if sigma < 1e-9:
            return 0.0
        return float((s.iloc[-1] - mu) / sigma)

    def detect_spread_regime(self, ig_spread: float, hy_spread: float) -> str:
        """
        Classify current credit environment.
        TIGHT   → IG < 80bps,  HY < 250bps   (risk-on, credit bullish)
        NORMAL  → IG 80-150,   HY 250-450
        WIDE    → IG > 150,    HY > 450       (risk-off, recession fears)
        CRISIS  → IG > 300,    HY > 900       (GFC-style dislocation)
        """
        if ig_spread > 300 or hy_spread > 900:
            return "CRISIS"
        if ig_spread < 80 and hy_spread < 250:
            return "TIGHT"
        if ig_spread > 150 or hy_spread > 450:
            return "WIDE"
        return "NORMAL"

    def get_current_regime(self) -> Tuple[str, float, float]:
        """Returns (regime, ig_oas_bps, hy_oas_bps)."""
        ig_s = self._fetch_fred_series(FRED_OAS_SERIES["IG"])
        hy_s = self._fetch_fred_series(FRED_OAS_SERIES["HY"])
        ig_bps = float(ig_s.iloc[-1]) if not ig_s.empty else TYPICAL_SPREADS_BPS["IG"]
        hy_bps = float(hy_s.iloc[-1]) if not hy_s.empty else TYPICAL_SPREADS_BPS["HY"]
        regime = self.detect_spread_regime(ig_bps, hy_bps)
        return regime, ig_bps, hy_bps

    def get_credit_term_structure_snapshot(self) -> Dict[str, float]:
        """
        Latest IG spread by maturity bucket (bps).
        {1-3Y, 3-5Y, 5-7Y, 7-10Y, 10-15Y}
        """
        df = self.fetch_term_structure()
        if df.empty:
            return {k: TYPICAL_SPREADS_BPS["IG"] for k in FRED_TERM_SERIES}
        latest = df.iloc[-1]
        result = {}
        for key in FRED_TERM_SERIES:
            result[key] = float(latest[key]) if key in latest and not pd.isna(latest[key]) else float("nan")
        return result


# ---------------------------------------------------------------------------
# CDS Spread Proxy
# ---------------------------------------------------------------------------

class CDSSpreadsProxy:
    """
    Construct CDS spread proxies without paying for CDS data.
    Uses bond spreads, equity volatility, and Merton structural model.
    """

    _merton = MertonModel()

    def construct_cds_proxy_from_bond(
        self,
        bond_yield: float,
        treasury_yield: float,
        recovery_rate: float = 0.40,
    ) -> float:
        """
        Approximate CDS spread from bond z-spread / OAS.

        CDS_spread ≈ bond_spread × (1 - recovery_rate) / (1 - recovery_rate)
        ≈ bond_spread (for IG) with par CDS recovery convention.

        Per Blanco, Brennan & Marsh (2005): CDS-bond basis typically ±20 bps.
        For IG: CDS ≈ bond OAS × 0.97 (slight positive basis historically).
        For HY: CDS ≈ bond OAS × 1.10 (CDS typically wider for HY).

        Returns CDS spread in bps.
        """
        bond_spread = (bond_yield - treasury_yield) * 10_000  # convert to bps
        if bond_spread < 0:
            bond_spread = 0.0
        # Standard ISDA recovery = 40%; par-equivalent CDS pricing
        # CDS ≈ (1 - R) × hazard_rate = bond_spread for similar seniority
        cds_proxy = bond_spread * (1 - recovery_rate) / (1 - recovery_rate)  # = bond_spread
        # Adjust for par vs market value of bond
        # High yield bonds often trade at discount → wider effective spread
        if bond_spread > 300:   # HY territory
            cds_proxy *= 1.10
        else:                   # IG territory
            cds_proxy *= 0.97
        return round(cds_proxy, 1)

    def construct_cds_proxy_from_equity(
        self,
        ticker: str,
        T: float = 5.0,
        r: float = 0.05,
    ) -> float:
        """
        Merton-implied CDS spread (5Y conventional) from equity + debt structure.
        Returns spread in bps or 0.0 on failure.
        """
        try:
            result = self._merton.run_merton_analysis(ticker, T=T, r=r)
            if result.error:
                return 0.0
            return round(result.credit_spread_bps, 1)
        except Exception as exc:
            logger.warning("Equity-based CDS proxy failed for %s: %s", ticker, exc)
            return 0.0

    def compute_basis(self, cds_spread: float, bond_spread: float) -> float:
        """
        CDS-Bond basis = CDS spread - bond OAS spread.
        Positive basis → CDS more expensive (short bond + buy protection = arb).
        Negative basis → bonds cheap relative to CDS (buy bond, sell protection).
        Both in bps.
        """
        return round(cds_spread - bond_spread, 1)

    def basis_signal(self, basis: float) -> str:
        """
        Basis trade signal interpretation.
        """
        if basis > 20:
            return "POSITIVE_BASIS: Buy bond / buy CDS protection (convergence arb)"
        if basis < -20:
            return "NEGATIVE_BASIS: Buy bond / sell CDS protection (cheapness signal)"
        return "NEUTRAL_BASIS: No clear basis trade"

    def estimate_cds_curve(
        self,
        ticker: str,
        tenors: List[float] = [1.0, 3.0, 5.0, 7.0, 10.0],
    ) -> Dict[str, float]:
        """
        Estimate CDS spread at multiple tenors using Merton structural model.
        Returns {tenor_str: spread_bps}.
        """
        curve = {}
        for T in tenors:
            try:
                result = self._merton.run_merton_analysis(ticker, T=T)
                if not result.error:
                    curve[f"{int(T)}Y"] = round(result.credit_spread_bps, 1)
                else:
                    curve[f"{int(T)}Y"] = float("nan")
            except Exception:
                curve[f"{int(T)}Y"] = float("nan")
        return curve


# ---------------------------------------------------------------------------
# Credit Sector Analyzer
# ---------------------------------------------------------------------------

# Mapping sectors to FRED OAS sector series where available
# FRED has maturity-bucket IG indices; sector-specific limited
SECTOR_FRED_PROXIES: Dict[str, str] = {
    # Maturity-bucket proxies for now (FRED lacks free sector-specific OAS)
    "SHORT":   "BAMLC1A0C13Y",    # 1-3Y IG (proxy for short-duration sectors)
    "MEDIUM":  "BAMLC3A0C57Y",    # 5-7Y IG
    "LONG":    "BAMLC7A0C1015Y",  # 10-15Y IG
    "OVERALL": "BAMLC0A0CM",      # Broad IG
}

# Sector to maturity-bucket mapping (approximation)
SECTOR_MATURITY_BUCKET: Dict[str, str] = {
    "Financials":   "SHORT",
    "Technology":   "MEDIUM",
    "Healthcare":   "MEDIUM",
    "Energy":       "LONG",
    "Utilities":    "LONG",
    "Consumer":     "MEDIUM",
    "Industrials":  "MEDIUM",
    "Real Estate":  "LONG",
    "Materials":    "MEDIUM",
    "Telecom":      "LONG",
}

# Typical sector spread premium over IG index (bps)
SECTOR_SPREAD_PREMIUM: Dict[str, float] = {
    "Financials":  20,
    "Technology": -10,
    "Healthcare": -5,
    "Energy":      35,
    "Utilities":   15,
    "Consumer":    10,
    "Industrials": 10,
    "Real Estate": 25,
    "Materials":   20,
    "Telecom":     30,
}


class CreditSectorAnalyzer:
    """
    Sector-level credit analysis using FRED OAS indices and
    sector-specific spread premium adjustments.
    """

    def __init__(self):
        self._builder = CreditSpreadBuilder()

    @property
    def sectors(self) -> List[str]:
        return list(SECTOR_MATURITY_BUCKET.keys())

    def compute_sector_spread_history(
        self, sector: str, rating: str = "IG"
    ) -> pd.Series:
        """
        Approximate sector spread history.
        Uses the closest FRED maturity-bucket series + sector premium adjustment.
        """
        bucket = SECTOR_MATURITY_BUCKET.get(sector, "OVERALL")
        sid = SECTOR_FRED_PROXIES.get(bucket, FRED_OAS_SERIES["IG"])
        base = self._builder._fetch_fred_series(sid)
        if base.empty:
            return pd.Series(dtype=float, name=sector)
        premium = SECTOR_SPREAD_PREMIUM.get(sector, 0)
        sector_spread = base + premium
        sector_spread.name = sector
        return sector_spread

    def compute_sector_z_spread(self, sector: str, window_years: int = 5) -> float:
        """
        Z-score of current spread vs 5-year rolling mean (a proxy for z-spread richness).
        Positive → sector wide vs history (cheap); Negative → tight (expensive).
        """
        s = self.compute_sector_spread_history(sector)
        if s.empty or len(s) < 60:
            return 0.0
        cutoff = s.index[-1] - pd.DateOffset(years=window_years)
        hist = s[s.index >= cutoff].dropna()
        if len(hist) < 20:
            return 0.0
        mu = hist.mean()
        sigma = hist.std()
        if sigma < 1e-9:
            return 0.0
        return round(float((s.iloc[-1] - mu) / sigma), 3)

    def rank_sectors_by_value(self) -> pd.DataFrame:
        """
        Rank all sectors by z-score (cheapest → most expensive).
        Positive z-score = wide vs history = potentially attractive.
        """
        rows = []
        for sector in self.sectors:
            s = self.compute_sector_spread_history(sector)
            current = float(s.iloc[-1]) if not s.empty else float("nan")
            z = self.compute_sector_z_spread(sector)
            rows.append({
                "sector": sector,
                "current_spread_bps": round(current, 1),
                "z_score_5yr": z,
                "assessment": "CHEAP" if z > 1 else "EXPENSIVE" if z < -1 else "FAIR",
            })
        df = pd.DataFrame(rows).sort_values("z_score_5yr", ascending=False)
        return df.reset_index(drop=True)

    def detect_sector_stress(self, sector: str) -> str:
        """
        NORMAL / ELEVATED / STRESSED based on z-score.
        """
        z = self.compute_sector_z_spread(sector)
        if z > 2:
            return "STRESSED"
        if z > 1:
            return "ELEVATED"
        return "NORMAL"

    def get_sector_snapshot(self) -> pd.DataFrame:
        """Full sector table: spread, z-score, stress level, 1M change."""
        rows = []
        for sector in self.sectors:
            s = self.compute_sector_spread_history(sector)
            if s.empty:
                continue
            current = float(s.iloc[-1])
            # 1-month change
            month_ago_idx = s.index[-1] - pd.DateOffset(days=30)
            prev = s[s.index <= month_ago_idx]
            mom_chg = round(current - float(prev.iloc[-1]), 1) if not prev.empty else float("nan")
            z = self.compute_sector_z_spread(sector)
            rows.append({
                "sector": sector,
                "spread_bps": round(current, 1),
                "mom_change_bps": mom_chg,
                "z_score_5yr": round(z, 3),
                "stress": self.detect_sector_stress(sector),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Credit Risk Engine (Orchestrator)
# ---------------------------------------------------------------------------

class CreditRiskEngine:
    """
    Orchestrator for credit analytics.
    Combines Merton, KMV, FRED spreads, and sector analytics.
    """

    def __init__(self):
        self._merton = MertonModel()
        self._kmv = KMVDistanceToDefault()
        self._builder = CreditSpreadBuilder()
        self._cds = CDSSpreadsProxy()
        self._sector = CreditSectorAnalyzer()

    def analyze_issuer(self, ticker: str) -> IssuerCreditProfile:
        """
        Full credit profile for a single issuer.
        Merton + KMV + implied rating + comparable FRED spread.
        """
        merton_res = self._merton.run_merton_analysis(ticker)
        kmv_res = self._kmv.run_for_ticker(ticker)
        spread_curve = self._builder.build_rating_spread_curve()

        # Infer implied rating from KMV DD
        dd = kmv_res.kmv_dd if kmv_res.kmv_dd else merton_res.distance_to_default
        implied_rating = _dd_to_implied_rating(dd)
        comparable_oas = spread_curve.get(implied_rating, TYPICAL_SPREADS_BPS.get(implied_rating, 120))
        issuer_spread = merton_res.credit_spread_bps if not merton_res.error else comparable_oas
        premium = round(issuer_spread - comparable_oas, 1)

        distress_flag = (
            dd < DD_THRESHOLDS["DISTRESSED"] or
            merton_res.risk_neutral_pd > 0.05
        )

        name = _get_company_name(ticker)
        return IssuerCreditProfile(
            ticker=ticker,
            company_name=name,
            merton=merton_res if not merton_res.error else None,
            kmv=kmv_res if not kmv_res.error else None,
            implied_rating=implied_rating,
            credit_spread_estimate_bps=round(issuer_spread, 1),
            comparable_oas_bps=round(comparable_oas, 1),
            spread_premium_bps=premium,
            distress_flag=distress_flag,
        )

    def get_credit_market_dashboard(self) -> dict:
        """
        Macro credit market snapshot.
        IG OAS, HY OAS, regime, rating spread curve, z-scores.
        """
        regime, ig_bps, hy_bps = self._builder.get_current_regime()
        curve = self._builder.build_rating_spread_curve()
        term = self._builder.get_credit_term_structure_snapshot()

        # Z-scores vs 1yr, 3yr, 5yr
        ig_z_1y = self._builder.get_spread_z_score("IG", window_years=1)
        ig_z_3y = self._builder.get_spread_z_score("IG", window_years=3)
        ig_z_5y = self._builder.get_spread_z_score("IG", window_years=5)
        hy_z_5y = self._builder.get_spread_z_score("HY", window_years=5)

        # Percentiles
        ig_pct_10y = self._builder.compute_spread_percentile("IG", ig_bps, 10)
        hy_pct_10y = self._builder.compute_spread_percentile("HY", hy_bps, 10)

        return {
            "as_of": date.today().isoformat(),
            "ig_oas_bps": round(ig_bps, 1),
            "hy_oas_bps": round(hy_bps, 1),
            "ig_hy_ratio": round(hy_bps / ig_bps, 2) if ig_bps > 0 else None,
            "regime": regime,
            "ig_pct_10y": ig_pct_10y,
            "hy_pct_10y": hy_pct_10y,
            "ig_z_score_1y": round(ig_z_1y, 3) if ig_z_1y is not None else None,
            "ig_z_score_3y": round(ig_z_3y, 3) if ig_z_3y is not None else None,
            "ig_z_score_5y": round(ig_z_5y, 3) if ig_z_5y is not None else None,
            "hy_z_score_5y": round(hy_z_5y, 3) if hy_z_5y is not None else None,
            "rating_spread_curve_bps": curve,
            "ig_term_structure_bps": term,
            "credit_cycle": self.compute_credit_cycle_position(),
        }

    def screen_distressed(
        self,
        universe: List[str],
        dd_threshold: float = 2.0,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Distressed screener: tickers with KMV DD < threshold.
        Returns sorted DataFrame of distressed issuers.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._kmv.run_for_ticker, t): t for t in universe}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    kmv = fut.result()
                    if kmv.kmv_dd < dd_threshold:
                        results.append({
                            "ticker": ticker,
                            "kmv_dd": round(kmv.kmv_dd, 3),
                            "edf_pct": round(kmv.edf * 100, 2),
                            "credit_quality": kmv.credit_quality,
                            "default_point_m": round(kmv.default_point, 1),
                            "firm_value_m": round(kmv.firm_value, 1),
                            "asset_vol": round(kmv.asset_vol, 4),
                            "error": kmv.error,
                        })
                except Exception as exc:
                    logger.warning("Distressed screen error for %s: %s", ticker, exc)

        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results).sort_values("kmv_dd", ascending=True)
        return df.reset_index(drop=True)

    def compute_credit_cycle_position(self) -> str:
        """
        Classify current credit cycle position.
        EXPANSION → tight spreads, narrowing trend
        PEAK      → tight but starting to widen
        CONTRACTION → spreads widening, HY underperforming IG
        TROUGH    → historically wide, starting to tighten
        """
        ig_s = self._builder._fetch_fred_series(FRED_OAS_SERIES["IG"])
        hy_s = self._builder._fetch_fred_series(FRED_OAS_SERIES["HY"])
        if ig_s.empty or len(ig_s) < 60:
            return "UNKNOWN"

        ig_now = float(ig_s.iloc[-1])
        ig_3m_ago = float(ig_s.iloc[-63]) if len(ig_s) > 63 else ig_now
        ig_1y_avg = float(ig_s.iloc[-252:].mean()) if len(ig_s) > 252 else ig_now

        hy_now = float(hy_s.iloc[-1]) if not hy_s.empty else 350.0
        hy_3m_ago = float(hy_s.iloc[-63]) if (not hy_s.empty and len(hy_s) > 63) else hy_now

        ig_trend_widening = ig_now > ig_3m_ago * 1.05    # +5% = widening
        ig_trend_tightening = ig_now < ig_3m_ago * 0.95  # -5% = tightening
        hy_trend_widening = hy_now > hy_3m_ago * 1.05
        ig_historically_tight = ig_now < ig_1y_avg * 0.90

        if not ig_trend_widening and not ig_historically_tight:
            return "EXPANSION"
        if ig_historically_tight and ig_trend_widening:
            return "PEAK"
        if ig_trend_widening and hy_trend_widening:
            return "CONTRACTION"
        if ig_now > ig_1y_avg * 1.20 and ig_trend_tightening:
            return "TROUGH"
        return "EXPANSION"

    def get_full_credit_report(self, ticker: str) -> dict:
        """
        Comprehensive credit report: issuer profile + market context.
        """
        profile = self.analyze_issuer(ticker)
        dashboard = self.get_credit_market_dashboard()
        cds_curve = self._cds.estimate_cds_curve(ticker)
        sector_stress = self._sector.get_sector_snapshot()

        return {
            "ticker": ticker,
            "company": profile.company_name,
            "as_of": profile.as_of.isoformat(),
            "implied_rating": profile.implied_rating,
            "credit_spread_estimate_bps": profile.credit_spread_estimate_bps,
            "comparable_index_oas_bps": profile.comparable_oas_bps,
            "spread_premium_bps": profile.spread_premium_bps,
            "distress_flag": profile.distress_flag,
            "merton": {
                "distance_to_default": round(profile.merton.distance_to_default, 3) if profile.merton else None,
                "risk_neutral_pd_pct": round(profile.merton.risk_neutral_pd * 100, 3) if profile.merton else None,
                "physical_pd_pct": round(profile.merton.physical_pd * 100, 3) if profile.merton else None,
                "asset_vol": round(profile.merton.asset_vol, 4) if profile.merton else None,
                "leverage_ratio": round(profile.merton.leverage_ratio, 4) if profile.merton else None,
                "firm_value_m": round(profile.merton.firm_value, 1) if profile.merton else None,
            },
            "kmv": {
                "kmv_dd": round(profile.kmv.kmv_dd, 3) if profile.kmv else None,
                "edf_pct": round(profile.kmv.edf * 100, 3) if profile.kmv else None,
                "credit_quality": profile.kmv.credit_quality if profile.kmv else None,
                "default_point_m": round(profile.kmv.default_point, 1) if profile.kmv else None,
            },
            "cds_curve_bps": cds_curve,
            "market_context": {
                "ig_oas_bps": dashboard["ig_oas_bps"],
                "hy_oas_bps": dashboard["hy_oas_bps"],
                "regime": dashboard["regime"],
                "credit_cycle": dashboard["credit_cycle"],
                "ig_pct_10y": dashboard["ig_pct_10y"],
            },
        }


# ---------------------------------------------------------------------------
# Utility / Data-fetching helpers
# ---------------------------------------------------------------------------

def _fetch_equity_data(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (market_cap_usd, annualised_equity_vol).
    Uses 90-day realized vol.
    """
    if not _YF:
        return None, None
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        market_cap = getattr(info, "market_cap", None)
        if market_cap is None:
            info2 = t.info
            market_cap = info2.get("marketCap")
        if not market_cap:
            return None, None

        # 90-day daily returns for realized vol
        end = datetime.today()
        start = end - timedelta(days=130)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty or len(hist) < 20:
            return float(market_cap), 0.30  # fallback vol

        close = hist["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        vol_daily = float(log_ret.tail(90).std())
        vol_annual = vol_daily * math.sqrt(252)
        return float(market_cap), round(vol_annual, 6)
    except Exception as exc:
        logger.warning("yfinance equity data failed for %s: %s", ticker, exc)
        return None, None


def _fetch_edgar_debt(ticker: str) -> Optional[float]:
    """
    Fetch total debt (LTD + STD) from EDGAR XBRL company facts.
    Returns total debt in USD or None.
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None
        url = EDGAR_COMPANY_FACTS.format(cik=cik)
        with httpx.Client(timeout=30, headers=_HEADERS) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()

        us_gaap = data.get("facts", {}).get("us-gaap", {})
        ltd = _extract_latest_xbrl(us_gaap, "LongTermDebt")
        std = _extract_latest_xbrl(us_gaap, "ShortTermBorrowings")
        std2 = _extract_latest_xbrl(us_gaap, "DebtCurrent")
        notes = _extract_latest_xbrl(us_gaap, "LongTermNotesPayable")

        total = (ltd or 0) + (std or std2 or 0) + (notes or 0)
        return total if total > 0 else None
    except Exception as exc:
        logger.debug("EDGAR debt fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_edgar_debt_split(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Returns (short_term_debt, long_term_debt) in USD from EDGAR."""
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None, None
        url = EDGAR_COMPANY_FACTS.format(cik=cik)
        with httpx.Client(timeout=30, headers=_HEADERS) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                return None, None
            data = resp.json()

        us_gaap = data.get("facts", {}).get("us-gaap", {})
        ltd = _extract_latest_xbrl(us_gaap, "LongTermDebt") or 0
        std = (_extract_latest_xbrl(us_gaap, "ShortTermBorrowings") or
               _extract_latest_xbrl(us_gaap, "DebtCurrent") or 0)
        return std, ltd
    except Exception:
        return None, None


def _fetch_yf_debt(ticker: str) -> Optional[float]:
    """Fallback: total debt from yfinance info dict."""
    if not _YF:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        total_debt = info.get("totalDebt") or info.get("longTermDebt")
        return float(total_debt) if total_debt else None
    except Exception:
        return None


def _fetch_yf_debt_split(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Fallback: (STD, LTD) from yfinance balance sheet."""
    if not _YF:
        return None, None
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        if bs is None or bs.empty:
            return None, None
        latest = bs.iloc[:, 0]
        std = _get_bs_item(latest, ["Current Debt", "Short Long Term Debt", "Short Term Debt"])
        ltd = _get_bs_item(latest, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        return std, ltd
    except Exception:
        return None, None


def _get_bs_item(series: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in series.index:
            v = series[c]
            if pd.notna(v):
                return float(v)
    return None


def _get_cik(ticker: str) -> Optional[str]:
    """Map ticker → SEC CIK via EDGAR company_tickers.json."""
    try:
        with httpx.Client(timeout=15, headers=_HEADERS) as client:
            resp = client.get(EDGAR_TICKER_MAP)
            resp.raise_for_status()
            data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                return cik
        return None
    except Exception as exc:
        logger.debug("CIK lookup failed for %s: %s", ticker, exc)
        return None


def _extract_latest_xbrl(us_gaap: dict, concept: str) -> Optional[float]:
    """Extract the most recent annual value of an XBRL concept."""
    node = us_gaap.get(concept, {})
    units = node.get("units", {})
    usd_data = units.get("USD", [])
    # Filter to 10-K filings only
    annual = [e for e in usd_data if e.get("form") in ("10-K", "10-K/A", "20-F")]
    if not annual:
        annual = usd_data
    if not annual:
        return None
    # Take the most recently filed
    annual_sorted = sorted(annual, key=lambda x: x.get("filed", ""), reverse=True)
    return float(annual_sorted[0]["val"])


def _get_company_name(ticker: str) -> str:
    if not _YF:
        return ticker
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def _dd_to_implied_rating(dd: float) -> str:
    """Map Distance-to-Default to approximate credit rating."""
    if dd > 7:
        return "AAA"
    if dd > 6:
        return "AA"
    if dd > 5:
        return "A"
    if dd > 3:
        return "BBB"
    if dd > 2:
        return "BB"
    if dd > 1:
        return "B"
    return "CCC"


# ---------------------------------------------------------------------------
# Main — demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("=" * 70)
    print("SENTINEL Credit Analytics v3 — Merton / KMV / CDS / Sector")
    print("=" * 70)

    # 1. Merton analysis on a small universe
    tickers = ["AAPL", "TSLA", "F", "KHC"]
    print(f"\n[1] Running Merton structural model on {tickers}...")
    merton = MertonModel()
    df_merton = merton.run_universe_merton(tickers)
    if not df_merton.empty:
        cols = ["ticker", "distance_to_default", "risk_neutral_pd", "credit_spread_bps",
                "leverage_ratio", "credit_quality", "error"]
        print(df_merton[[c for c in cols if c in df_merton.columns]].to_string(index=False))
    else:
        print("  Merton data unavailable (check network/yfinance)")

    # 2. KMV Distance-to-Default
    print(f"\n[2] KMV Distance-to-Default...")
    kmv = KMVDistanceToDefault()
    for t in tickers:
        r = kmv.run_for_ticker(t)
        if not r.error:
            print(f"  {t}: DD={r.kmv_dd:.2f}, EDF={r.edf*100:.2f}%, Quality={r.credit_quality}")
        else:
            print(f"  {t}: Error — {r.error}")

    # 3. Rating spread curve from FRED
    print("\n[3] FRED Rating Spread Curve (bps)...")
    builder = CreditSpreadBuilder()
    curve = builder.build_rating_spread_curve()
    for rating, spread in sorted(curve.items(), key=lambda x: TYPICAL_SPREADS_BPS.get(x[0], 999)):
        print(f"  {rating:5s}: {spread:6.1f} bps")

    # 4. Credit market regime
    print("\n[4] Credit Market Regime...")
    regime, ig, hy = builder.get_current_regime()
    print(f"  IG OAS: {ig:.1f} bps | HY OAS: {hy:.1f} bps | Regime: {regime}")
    ig_pct = builder.compute_spread_percentile("IG", ig, lookback_years=10)
    print(f"  IG spread is at {ig_pct:.0f}th percentile vs 10-year history")

    # 5. Spread term structure
    print("\n[5] IG Term Structure (bps)...")
    term = builder.get_credit_term_structure_snapshot()
    for bucket, bps in term.items():
        print(f"  {bucket}: {bps:.1f} bps")

    # 6. Sector analysis
    print("\n[6] Sector Credit Ranking...")
    sector = CreditSectorAnalyzer()
    df_sector = sector.rank_sectors_by_value()
    print(df_sector.to_string(index=False))

    # 7. Distressed screener on S&P 500 sample
    print("\n[7] Distressed Screener (S&P 500 sample, KMV DD < 2.5)...")
    sp500_sample = ["AAPL", "TSLA", "F", "KHC", "GE", "BA", "M", "WBA", "ETSY", "DVN"]
    engine = CreditRiskEngine()
    df_distressed = engine.screen_distressed(sp500_sample, dd_threshold=2.5)
    if df_distressed.empty:
        print("  No distressed issuers found in sample (or data unavailable)")
    else:
        print(df_distressed.to_string(index=False))

    # 8. Credit market dashboard
    print("\n[8] Credit Market Dashboard...")
    dashboard = engine.get_credit_market_dashboard()
    print(f"  Regime:       {dashboard['regime']}")
    print(f"  Credit Cycle: {dashboard['credit_cycle']}")
    print(f"  IG OAS:       {dashboard['ig_oas_bps']} bps (10Y pct: {dashboard['ig_pct_10y']:.0f})")
    print(f"  HY OAS:       {dashboard['hy_oas_bps']} bps (10Y pct: {dashboard['hy_pct_10y']:.0f})")
    print(f"  IG Z-Score 5Y: {dashboard['ig_z_score_5y']}")
    print(f"  HY Z-Score 5Y: {dashboard['hy_z_score_5y']}")

    print("\nDone.")
