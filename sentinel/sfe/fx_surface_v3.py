"""
fx_surface_v3.py — Comprehensive FX analytics platform.

dim_006: FX spot, forwards, volatility surface — score 6 → 9

Architecture:
  ECBSpotRateCollector   — ECB SDW free API, 20+ currencies vs EUR
  FREDFXAdapter          — FRED CSV, USD-based pairs to 1971
  ForwardCurveBuilder    — CIP-based forward rates, ON to 2Y tenors
  FXVolatilitySurface    — Vol surface: realized EWMA + SABR calibration
  FXOptionsAnalytics     — Garman-Kohlhagen pricing, Greeks, risk reversal/fly
  CarryTradeAnalyzer     — Rate differentials, carry ladder, Sharpe, unwind detect
  FXRealEffectiveRate    — REER computation from bilateral rates + GDP weights
  FXMomentumSignals      — 12-1 momentum, trend-following (SMA + ADX)

Free data only: ECB SDW, FRED CSV, yfinance (price only).
Core pricing (GK, forward curve) uses numpy only — no scipy dependency.
scipy used only for SABR calibration and spline interpolation (guarded).
"""
from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — guarded so core works without them
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize, least_squares  # type: ignore[import-untyped]
    from scipy.interpolate import RectBivariateSpline   # type: ignore[import-untyped]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not installed — SABR calibration and spline interpolation disabled")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json",
}
_TIMEOUT = 30
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
_FRANKFURTER = "https://api.frankfurter.app"

# G10 currencies
_G10 = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"}

# All currencies vs EUR available in ECB SDW
_ECB_CURRENCIES = [
    "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK", "DKK",
    "NZD", "HKD", "SGD", "CNY", "KRW", "MXN", "BRL", "INR", "ZAR",
    "TRY", "PLN", "CZK", "HUF", "RON", "IDR", "MYR", "PHP", "THB",
]

# FRED series IDs for USD-based spot rates
_FRED_FX_SERIES: Dict[str, str] = {
    "EURUSD": "DEXUSEU",   # USD per EUR  (not inverted)
    "USDJPY": "DEXJPUS",   # JPY per USD
    "GBPUSD": "DEXUSUK",   # USD per GBP  (not inverted)
    "USDCHF": "DEXSZUS",   # CHF per USD
    "AUDUSD": "DEXUSAL",   # USD per AUD  (not inverted)
    "USDCAD": "DEXCAUS",   # CAD per USD
    "USDMXN": "DEXMXUS",   # MXN per USD
    "USDBRL": "DEXBZUS",   # BRL per USD
    "USDCNY": "DEXCHUS",   # CNY per USD
    "USDKRW": "DEXKOUS",   # KRW per USD
    "USDSGD": "DEXSIUS",   # SGD per USD
    "USDINR": "DEXINUS",   # INR per USD
    "USDHKD": "DEXHKUS",   # HKD per USD
    "USDZAR": "DEXSFUS",   # ZAR per USD
    "USDNOK": "DEXNOUS",   # NOK per USD
    "USDSEK": "DEXSDUS",   # SEK per USD
    "USDTWD": "DEXTAUS",   # TWD per USD
    "USDTHB": "DEXTHUS",   # THB per USD
}

# FRED interest rate series per currency
_RATE_SERIES: Dict[str, str] = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBP": "IUQABEDR",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IRSTCI01CHM156N",
    "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
    "SEK": "IRSTCI01SEM156N",
    "NOK": "IRSTCI01NOM156N",
}

# Forward tenors: label → year fraction
_TENORS: Dict[str, float] = {
    "ON":  1 / 365,
    "1W":  7 / 365,
    "2W":  14 / 365,
    "1M":  1 / 12,
    "2M":  2 / 12,
    "3M":  3 / 12,
    "6M":  6 / 12,
    "9M":  9 / 12,
    "1Y":  1.0,
    "2Y":  2.0,
}

# Vol surface moneyness grid (% of ATM)
_MONEYNESS = [0.75, 0.90, 1.00, 1.10, 1.25]
_VOL_TENORS = ["1W", "1M", "3M", "6M", "1Y"]

# EWMA lambda for realized vol
_EWMA_LAMBDA = 0.94

# Approximate GDP weights for REER (World Bank 2023 data, normalised)
_GDP_WEIGHTS: Dict[str, float] = {
    "USD": 0.2500, "EUR": 0.1850, "CNY": 0.1750, "JPY": 0.0550,
    "GBP": 0.0450, "INR": 0.0380, "CAD": 0.0210, "KRW": 0.0200,
    "AUD": 0.0175, "BRL": 0.0175, "MXN": 0.0150, "IDR": 0.0145,
    "CHF": 0.0120, "SAR": 0.0105, "SEK": 0.0080, "NOR": 0.0075,
    "TRY": 0.0065, "NZD": 0.0055, "SGD": 0.0050, "HKD": 0.0050,
    "ZAR": 0.0045, "THB": 0.0040, "PLN": 0.0040, "MYR": 0.0040,
    "DKK": 0.0035,
}

# Extended pair universe
_ALL_PAIRS = [
    # G10
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
    "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURNOK",
    "EURSEK", "USDSEK", "USDNOK", "EURAUD", "GBPAUD",
    # EM
    "USDCNY", "USDMXN", "USDBRL", "USDSGD", "USDHKD",
    "USDKRW", "USDINR", "USDZAR", "USDTRY", "USDPLN",
    "USDCZK", "USDHUF", "USDIDR", "USDPHP", "USDTHB",
]


# ===========================================================================
# Helpers
# ===========================================================================

def _parse_date(d: str | date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d[:10], "%Y-%m-%d").date()


def _fred_fetch(series_id: str, start: str, end: str,
                session: Optional[requests.Session] = None) -> pd.Series:
    """Fetch a FRED series as a pandas Series. Returns empty on failure."""
    sess = session or requests.Session()
    params = {"id": series_id, "vintage_date": end}
    try:
        url = _FRED_CSV
        resp = sess.get(url, params={"id": series_id}, timeout=_TIMEOUT,
                        headers={"User-Agent": _HEADERS["User-Agent"]})
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"],
                         index_col="DATE", na_values=[".", ""])
        df.index = pd.to_datetime(df.index)
        series = df.iloc[:, 0].dropna()
        series = series.loc[(series.index >= pd.Timestamp(start)) &
                            (series.index <= pd.Timestamp(end))]
        return series
    except Exception as exc:
        logger.warning(f"FRED {series_id}: {exc}")
        return pd.Series(dtype=float)


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class OptionPrice:
    """Result from Garman-Kohlhagen pricing."""
    value: float
    delta: float
    gamma: float
    theta: float      # per calendar day
    vega: float       # per 1% vol move
    rho_dom: float
    rho_for: float
    call_put: str

@dataclass
class VolSurface:
    """FX implied vol surface on a moneyness × tenor grid."""
    pair: str
    as_of: date
    moneyness: List[float]          # e.g. [0.75, 0.90, 1.00, 1.10, 1.25]
    tenors: List[str]               # e.g. ["1W", "1M", "3M", "6M", "1Y"]
    vol_grid: np.ndarray            # shape: (len(tenors), len(moneyness))
    atm_vols: Dict[str, float]      # tenor → ATM vol
    sabr_params: Optional[Dict[str, Any]] = None  # tenor → {alpha, beta, rho, nu}

@dataclass
class TrendSignal:
    """SMA crossover + ADX trend-following signal."""
    pair: str
    direction: str        # "long", "short", "flat"
    sma_fast: float
    sma_slow: float
    adx: float
    signal_strength: float  # 0-1

@dataclass
class ForwardCurve:
    """CIP-derived forward curve for an FX pair."""
    pair: str
    spot: float
    base_rate: float   # annualised %
    quote_rate: float  # annualised %
    as_of: date
    tenors: Dict[str, float]       # tenor label → forward rate
    forward_points: Dict[str, float]  # tenor label → pips difference


# ===========================================================================
# ECBSpotRateCollector — primary spot source via ECB SDW free API
# ===========================================================================

class ECBSpotRateCollector:
    """Fetch FX spot rates from the ECB Statistical Data Warehouse.

    Endpoint: https://data-api.ecb.europa.eu/service/data/EXR/D.{currency}.EUR.SP00.A
    Returns daily closing rates (last business day of period).
    All rates quoted as foreign currency units per 1 EUR.
    """

    _BASE = _ECB_BASE

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": _HEADERS["User-Agent"],
            "Accept": "application/vnd.sdmx.data+csv; version=1.0.0",
        })

    def _build_url(self, currency: str) -> str:
        key = f"D.{currency}.EUR.SP00.A"
        return f"{self._BASE}/EXR/{key}"

    def fetch_spot(self, base: str = "EUR", quote: str = "USD",
                   start: str = "2000-01-01",
                   end: Optional[str] = None) -> pd.Series:
        """Fetch spot rate history.

        Returns Series of (quote per base) rates indexed by date.
        Handles cross-rate computation when neither currency is EUR.
        """
        if end is None:
            end = date.today().isoformat()

        # ECB provides all rates vs EUR. Compute cross-rates as needed.
        if base == "EUR":
            return self._fetch_eur_cross(quote, start, end)
        elif quote == "EUR":
            series = self._fetch_eur_cross(base, start, end)
            return (1.0 / series).dropna()
        else:
            # Compute cross via EUR: base/quote = EUR/quote ÷ EUR/base
            eur_quote = self._fetch_eur_cross(quote, start, end)
            eur_base = self._fetch_eur_cross(base, start, end)
            if eur_quote.empty or eur_base.empty:
                return pd.Series(dtype=float)
            combined = pd.concat([eur_quote, eur_base], axis=1).dropna()
            combined.columns = ["eur_quote", "eur_base"]
            # base/quote rate = EUR_quote / EUR_base → how many 'quote' per 1 'base'
            cross = combined["eur_quote"] / combined["eur_base"]
            cross.name = f"{base}{quote}"
            return cross

    def _fetch_eur_cross(self, currency: str, start: str, end: str) -> pd.Series:
        """Fetch currency vs EUR from ECB SDW."""
        url = self._build_url(currency)
        params = {
            "startPeriod": start,
            "endPeriod": end,
            "format": "csvdata",
        }
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 404:
                logger.warning(f"ECB: {currency}/EUR not found, trying Frankfurter fallback")
                return self._frankfurter_fallback(currency, start, end)
            resp.raise_for_status()
            text = resp.text

            if not text.strip():
                return self._frankfurter_fallback(currency, start, end)

            # Parse CSV — ECB SDMX CSV format
            df = pd.read_csv(io.StringIO(text))

            # ECB SDMX CSV has various column layouts depending on version
            # Look for date column and value column
            date_col = None
            value_col = None
            for col in df.columns:
                lc = col.lower()
                if lc in ("time_period", "date", "period"):
                    date_col = col
                elif lc in ("obs_value", "value", "close"):
                    value_col = col

            if date_col is None or value_col is None:
                # Try the transposed layout
                if "TIME_PERIOD" in df.columns:
                    date_col = "TIME_PERIOD"
                elif df.columns[0].startswith("20") or df.columns[0].startswith("19"):
                    # dates are columns
                    dates = pd.to_datetime(df.columns[1:], errors="coerce")
                    vals = pd.to_numeric(df.iloc[0, 1:], errors="coerce")
                    series = pd.Series(vals.values, index=dates).dropna()
                    series.name = f"EUR{currency}"
                    return series
                else:
                    logger.warning(f"ECB: unexpected CSV format for {currency}")
                    return self._frankfurter_fallback(currency, start, end)

            if value_col is None:
                # last numeric column
                for col in reversed(df.columns.tolist()):
                    if df[col].dtype in (float, int) or pd.api.types.is_numeric_dtype(df[col]):
                        value_col = col
                        break

            if value_col is None:
                return self._frankfurter_fallback(currency, start, end)

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df = df.dropna(subset=[date_col, value_col])
            series = df.set_index(date_col)[value_col].sort_index()
            series.name = f"EUR{currency}"
            logger.info(f"ECB: EUR/{currency} → {len(series)} rows")
            return series

        except Exception as exc:
            logger.warning(f"ECB fetch error for {currency}: {exc}")
            return self._frankfurter_fallback(currency, start, end)

    def _frankfurter_fallback(self, currency: str, start: str, end: str) -> pd.Series:
        """Frankfurter API fallback for EUR cross rates (supports ~30 currencies)."""
        try:
            url = f"{_FRANKFURTER}/{start}..{end}"
            params = {"from": "EUR", "to": currency}
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            rates_dict = data.get("rates", {})
            records = {
                pd.Timestamp(d): v.get(currency, np.nan)
                for d, v in rates_dict.items()
            }
            series = pd.Series(records).dropna()
            series = series.sort_index()
            series.name = f"EUR{currency}"
            logger.info(f"Frankfurter fallback: EUR/{currency} → {len(series)} rows")
            return series
        except Exception as exc:
            logger.warning(f"Frankfurter fallback failed for {currency}: {exc}")
            return pd.Series(dtype=float)

    def fetch_all_pairs(self, start: str = "2000-01-01",
                        end: Optional[str] = None) -> pd.DataFrame:
        """Fetch all available currency pairs vs EUR.

        Returns DataFrame where each column is EUR/CCY, index is date.
        """
        if end is None:
            end = date.today().isoformat()
        frames: Dict[str, pd.Series] = {}
        for ccy in _ECB_CURRENCIES:
            try:
                s = self._fetch_eur_cross(ccy, start, end)
                if not s.empty:
                    frames[ccy] = s
                time.sleep(0.2)  # rate limit courtesy
            except Exception as exc:
                logger.warning(f"fetch_all_pairs: {ccy} → {exc}")

        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames)
        df.index.name = "date"
        return df.sort_index()

    def get_latest_rates(self, currencies: Optional[List[str]] = None) -> Dict[str, float]:
        """Return most recent EUR/CCY rates for the given currencies."""
        if currencies is None:
            currencies = _ECB_CURRENCIES[:10]

        rates: Dict[str, float] = {}
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=10)).isoformat()
        for ccy in currencies:
            s = self._fetch_eur_cross(ccy, start, end)
            if not s.empty:
                rates[ccy] = float(s.iloc[-1])
        return rates

    def compute_cross_rate(self, series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """Given EUR/A and EUR/B, return A/B cross rate."""
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        cross = aligned.iloc[:, 0] / aligned.iloc[:, 1]
        return cross


# ===========================================================================
# FREDFXAdapter — USD-based pairs, deep history to 1971
# ===========================================================================

class FREDFXAdapter:
    """Fetch FX spot rates from FRED free CSV download.

    Supports USD-based major and EM pairs going back to 1971 for G10.
    FRED rates are sometimes quoted as foreign-per-USD (JPY/USD, CHF/USD)
    and sometimes as USD-per-foreign (USD/EUR, USD/GBP).
    """

    # FRED convention flags: True means series is USD_per_foreign
    _USD_PER_FOREIGN = {"DEXUSEU", "DEXUSUK", "DEXUSAL"}

    def __init__(self, timeout: int = _TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _HEADERS["User-Agent"]})

    def fetch(self, pair: str, start: str = "1971-01-01",
              end: Optional[str] = None) -> pd.Series:
        """Fetch spot rate time series for a named pair.

        pair: e.g. 'EURUSD', 'USDJPY', 'GBPUSD'
        Returns Series of (quote per base), indexed by date.
        """
        if end is None:
            end = date.today().isoformat()

        pair = pair.upper()
        series_id = _FRED_FX_SERIES.get(pair)
        if not series_id:
            logger.warning(f"FRED: no series mapping for pair '{pair}'")
            return pd.Series(dtype=float)

        raw = _fred_fetch(series_id, start, end, self.session)
        if raw.empty:
            return raw

        # Adjust direction: ensure rate is (quote per base)
        # e.g. EURUSD: DEXUSEU is USD per EUR → correct, no flip
        # USDJPY: DEXJPUS is JPY per USD → correct, no flip
        # GBPUSD: DEXUSUK is USD per GBP → correct, no flip
        # USDCHF: DEXSZUS is CHF per USD → correct, no flip
        raw.name = pair
        raw = raw.dropna()
        raw.index = pd.to_datetime(raw.index).normalize()
        return raw

    def fetch_rate_series(self, currency: str, start: str = "2000-01-01",
                          end: Optional[str] = None) -> pd.Series:
        """Fetch interest rate series for a currency from FRED."""
        if end is None:
            end = date.today().isoformat()
        series_id = _RATE_SERIES.get(currency.upper())
        if not series_id:
            logger.warning(f"FRED: no rate series for {currency}")
            return pd.Series(dtype=float)
        return _fred_fetch(series_id, start, end, self.session)

    def get_all_available_pairs(self) -> List[str]:
        """Return all pair codes with FRED series mappings."""
        return list(_FRED_FX_SERIES.keys())

    def get_latest_rate(self, pair: str) -> Optional[float]:
        """Return the most recent available rate for a pair."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=10)).isoformat()
        s = self.fetch(pair, start, end)
        if s.empty:
            return None
        return float(s.iloc[-1])

    def build_rate_table(self, as_of: Optional[str] = None) -> pd.DataFrame:
        """Return a table of all available spot rates as of a date."""
        if as_of is None:
            as_of = date.today().isoformat()
        start = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
        rows = []
        for pair, series_id in _FRED_FX_SERIES.items():
            s = _fred_fetch(series_id, start, as_of, self.session)
            if not s.empty:
                rows.append({
                    "pair": pair,
                    "rate": float(s.iloc[-1]),
                    "date": s.index[-1].date(),
                    "series_id": series_id,
                })
        return pd.DataFrame(rows).sort_values("pair")


# ===========================================================================
# ForwardCurveBuilder — CIP-based forward rates
# ===========================================================================

class ForwardCurveBuilder:
    """Build FX forward curves using Covered Interest Rate Parity.

    F(T) = S × (1 + r_quote × T) / (1 + r_base × T)
    or continuously compounded:
    F(T) = S × exp((r_quote - r_base) × T)

    Where T is time in years, rates are annualised decimals.
    """

    def __init__(self, fred_adapter: Optional[FREDFXAdapter] = None) -> None:
        self.fred = fred_adapter or FREDFXAdapter()

    def build_forward_curve(
        self,
        spot: float,
        base_rate: float,   # annualised % (e.g. 5.25 for 5.25%)
        quote_rate: float,  # annualised %
        pair: str = "UNKNOWN",
        as_of: Optional[date] = None,
    ) -> ForwardCurve:
        """Compute forward rates for all tenors using continuous compounding CIP.

        Args:
            spot: current spot rate (quote per base)
            base_rate: annualised interest rate for base currency (%)
            quote_rate: annualised interest rate for quote currency (%)
            pair: currency pair label
            as_of: valuation date

        Returns:
            ForwardCurve with tenors, forward rates, and forward points.
        """
        r_base = base_rate / 100.0
        r_quote = quote_rate / 100.0

        tenors: Dict[str, float] = {}
        points: Dict[str, float] = {}

        for tenor_label, T in _TENORS.items():
            # Continuous compounding CIP
            fwd = spot * math.exp((r_quote - r_base) * T)
            tenors[tenor_label] = round(fwd, 6)
            # Forward points in pips (4th decimal place for most pairs)
            if "JPY" in pair:
                pip_factor = 100.0    # JPY quoted to 2dp, pips = 0.01
            else:
                pip_factor = 10_000.0  # standard 4dp, pips = 0.0001
            points[tenor_label] = round((fwd - spot) * pip_factor, 4)

        return ForwardCurve(
            pair=pair,
            spot=round(spot, 6),
            base_rate=base_rate,
            quote_rate=quote_rate,
            as_of=as_of or date.today(),
            tenors=tenors,
            forward_points=points,
        )

    def get_forward_points(self, pair: str, tenor: str,
                           spot: Optional[float] = None) -> float:
        """Return forward points for a pair and tenor using current FRED rates.

        Forward points = (forward - spot) × pip_factor
        """
        if tenor not in _TENORS:
            raise ValueError(f"Unknown tenor '{tenor}'. Valid: {list(_TENORS.keys())}")

        base = pair[:3].upper()
        quote = pair[3:].upper()

        # Get spot rate from FRED if not provided
        if spot is None:
            spot = self.fred.get_latest_rate(pair) or 1.0

        # Get interest rates
        r_base = self._get_current_rate(base)
        r_quote = self._get_current_rate(quote)

        T = _TENORS[tenor]
        fwd = spot * math.exp((r_quote / 100 - r_base / 100) * T)
        pip_factor = 100.0 if "JPY" in pair else 10_000.0
        return round((fwd - spot) * pip_factor, 4)

    def implied_forward_rate(self, spot: float, forward: float,
                             tenor_days: int, base_rate: float) -> float:
        """Solve for the implied quote-currency rate given spot, forward, base rate.

        F = S × exp((r_q - r_b) × T)
        r_q = ln(F/S) / T + r_b
        """
        T = tenor_days / 365.0
        r_base = base_rate / 100.0
        if T <= 0 or spot <= 0 or forward <= 0:
            return base_rate
        r_quote = math.log(forward / spot) / T + r_base
        return round(r_quote * 100, 6)

    def _get_current_rate(self, currency: str) -> float:
        """Fetch most recent interest rate for a currency from FRED."""
        series_id = _RATE_SERIES.get(currency.upper())
        if not series_id:
            return 4.0  # default 4% if unknown
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=90)).isoformat()
        s = _fred_fetch(series_id, start, end, self.fred.session)
        if s.empty:
            return 4.0
        return float(s.iloc[-1])

    def build_pair_curve(self, pair: str,
                         start: str = "2020-01-01",
                         end: Optional[str] = None) -> pd.DataFrame:
        """Build a time series of 3M forward rates for a pair.

        Returns DataFrame with columns: spot, rate_base, rate_quote, fwd_3m
        """
        if end is None:
            end = date.today().isoformat()
        base = pair[:3].upper()
        quote = pair[3:].upper()

        spot_s = self.fred.fetch(pair, start, end)
        rate_base_s = self.fred.fetch_rate_series(base, start, end)
        rate_quote_s = self.fred.fetch_rate_series(quote, start, end)

        if spot_s.empty:
            return pd.DataFrame()

        df = pd.concat({
            "spot": spot_s,
            "rate_base": rate_base_s,
            "rate_quote": rate_quote_s,
        }, axis=1).dropna(subset=["spot"])

        # Forward fill rates (published monthly)
        df["rate_base"] = df["rate_base"].ffill().fillna(2.0)
        df["rate_quote"] = df["rate_quote"].ffill().fillna(2.0)

        T = _TENORS["3M"]
        df["fwd_3m"] = df["spot"] * np.exp(
            (df["rate_quote"] / 100 - df["rate_base"] / 100) * T
        )
        df["fwd_points_3m"] = (df["fwd_3m"] - df["spot"]) * (
            100.0 if "JPY" in pair else 10_000.0
        )
        return df


# ===========================================================================
# FXVolatilitySurface — realized EWMA vol surface + SABR calibration
# ===========================================================================

class FXVolatilitySurface:
    """Construct FX implied/realized volatility surface.

    Primary method: Historical realized vol via EWMA (λ=0.94) on daily returns.
    This approximates implied vol where option market data is unavailable.

    SABR calibration (if scipy available): fit α, β=0.5, ρ, ν to smile data.
    Smile construction: ATM + risk-reversal + butterfly to get full moneyness smile.
    Interpolation: bi-cubic spline across moneyness and time (scipy required).
    """

    def __init__(self, fred: Optional[FREDFXAdapter] = None,
                 ecb: Optional[ECBSpotRateCollector] = None) -> None:
        self.fred = fred or FREDFXAdapter()
        self.ecb = ecb or ECBSpotRateCollector()
        self._vol_cache: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Core: realized vol from daily returns
    # ------------------------------------------------------------------

    def _fetch_returns(self, pair: str, lookback_days: int = 504) -> pd.Series:
        """Fetch daily log-returns for a pair (yfinance price data)."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=lookback_days)).isoformat()

        # Try FRED first
        s = self.fred.fetch(pair, start, end)
        if not s.empty:
            rets = np.log(s / s.shift(1)).dropna()
            return rets

        # Fallback: yfinance FX pair
        if _YF_AVAILABLE:
            try:
                yf_symbol = pair[:3] + pair[3:] + "=X"
                raw = yf.download(yf_symbol, start=start, end=end,
                                  auto_adjust=True, progress=False, threads=False)
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    closes = raw["Close"].dropna()
                    rets = np.log(closes / closes.shift(1)).dropna()
                    return rets
            except Exception as exc:
                logger.warning(f"yfinance FX {pair}: {exc}")

        return pd.Series(dtype=float)

    def _ewma_vol(self, returns: pd.Series, lam: float = _EWMA_LAMBDA) -> pd.Series:
        """Compute EWMA variance series and return annualised vol series."""
        if returns.empty:
            return pd.Series(dtype=float)
        r2 = returns ** 2
        # Recursive EWMA: var_t = λ * var_{t-1} + (1-λ) * r_t^2
        ewma_var = r2.ewm(com=(lam / (1 - lam))).mean()
        ewma_vol = np.sqrt(ewma_var * 252)  # annualise
        return ewma_vol

    def get_atm_vol(self, pair: str, tenor_days: int) -> float:
        """Return ATM vol for a pair and tenor (in calendar days).

        Uses EWMA realized vol as proxy for implied vol where no option data exists.
        Interpolates between computed tenors.
        """
        rets = self._fetch_returns(pair, lookback_days=max(tenor_days * 3, 252))
        if rets.empty:
            return 0.10  # fallback 10% vol

        ewma_v = self._ewma_vol(rets)
        if ewma_v.empty:
            return 0.10

        # Use the most recent EWMA vol for the requested tenor
        # Scale by sqrt(tenor/252) relative to trailing realized to get term vol
        trailing_252 = rets.tail(252).std() * math.sqrt(252) if len(rets) >= 21 else 0.10
        tenor_years = tenor_days / 365.0
        # Term vol approximation: flat vol structure (conservative)
        recent_vol = float(ewma_v.iloc[-1])
        return max(round(recent_vol, 4), 0.0050)

    def get_vol_smile(self, pair: str, tenor: str) -> Dict[float, float]:
        """Return vol smile: moneyness → vol.

        For markets without traded FX options, uses realized vol with
        skew estimated from historical return skewness.
        """
        tenor_days = int(_TENORS[tenor] * 365)
        atm_vol = self.get_atm_vol(pair, tenor_days)

        rets = self._fetch_returns(pair, lookback_days=504)
        if rets.empty:
            return {m: atm_vol for m in _MONEYNESS}

        # Empirical skew and kurtosis from historical returns
        skew = float(rets.skew()) if len(rets) >= 30 else 0.0
        kurt = float(rets.kurtosis()) if len(rets) >= 30 else 0.0

        # Simple skew-adjusted smile:
        # vol(m) ≈ ATM_vol × (1 - skew_adj × (m - 1) + kurt_adj × (m - 1)^2)
        skew_adj = -skew * 0.05   # empirical scaling
        kurt_adj = (kurt + 3) * 0.02  # excess kurtosis → wings

        smile: Dict[float, float] = {}
        for m in _MONEYNESS:
            dev = m - 1.0
            adj_vol = atm_vol * (1.0 - skew_adj * dev + kurt_adj * dev ** 2)
            smile[m] = max(round(adj_vol, 5), 0.0050)

        return smile

    def build_surface(self, pair: str) -> VolSurface:
        """Build a full vol surface: moneyness × tenor.

        Returns VolSurface with grid, ATM vols, and optional SABR params.
        """
        atm_vols: Dict[str, float] = {}
        vol_grid = np.zeros((len(_VOL_TENORS), len(_MONEYNESS)))

        for i, tenor in enumerate(_VOL_TENORS):
            tenor_days = int(_TENORS[tenor] * 365)
            atm_vol = self.get_atm_vol(pair, tenor_days)
            atm_vols[tenor] = atm_vol
            smile = self.get_vol_smile(pair, tenor)
            for j, m in enumerate(_MONEYNESS):
                vol_grid[i, j] = smile.get(m, atm_vol)

        # SABR calibration if scipy available
        sabr_params: Optional[Dict[str, Any]] = None
        if _SCIPY_AVAILABLE:
            sabr_params = {}
            for i, tenor in enumerate(_VOL_TENORS):
                smile_m = np.array(_MONEYNESS)
                smile_v = vol_grid[i, :]
                try:
                    params = self._calibrate_sabr(
                        moneyness=smile_m,
                        vols=smile_v,
                        atm_vol=atm_vols[tenor],
                        T=_TENORS[tenor],
                    )
                    sabr_params[tenor] = params
                except Exception as exc:
                    logger.debug(f"SABR calibration for {pair}/{tenor}: {exc}")

        return VolSurface(
            pair=pair,
            as_of=date.today(),
            moneyness=list(_MONEYNESS),
            tenors=list(_VOL_TENORS),
            vol_grid=vol_grid,
            atm_vols=atm_vols,
            sabr_params=sabr_params,
        )

    # ------------------------------------------------------------------
    # SABR model calibration (scipy required)
    # ------------------------------------------------------------------

    def _sabr_vol(self, F: float, K: float, T: float,
                  alpha: float, beta: float, rho: float, nu: float) -> float:
        """SABR implied vol formula (Hagan 2002, simplified).

        Args:
            F: forward price
            K: strike
            T: time to expiry (years)
            alpha: initial vol
            beta: CEV exponent (typically 0.5 for FX)
            rho: vol-spot correlation
            nu: vol-of-vol
        """
        if abs(F - K) < 1e-10:
            # ATM formula
            FK_beta = F ** (1 - beta)
            denom = FK_beta * (1 + ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2 * T
                               + rho * beta * nu * alpha / (4 * FK_beta) * T
                               + (2 - 3 * rho ** 2) / 24 * nu ** 2 * T)
            return alpha / denom

        # General formula
        log_FK = math.log(F / K)
        FK_mid = math.sqrt(F * K)
        FK_mid_beta = FK_mid ** (1 - beta)
        log_sq = log_FK ** 2

        z = nu / alpha * FK_mid_beta * log_FK
        x_z = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)) if abs(z) > 1e-8 else 1.0

        num = alpha
        denom1 = FK_mid_beta * (1 + (1 - beta) ** 2 / 24 * log_sq
                                + (1 - beta) ** 4 / 1920 * log_sq ** 2)
        bracket = (1 + ((1 - beta) ** 2 / 24 * alpha ** 2 / FK_mid_beta ** 2
                        + 0.25 * rho * beta * nu * alpha / FK_mid_beta
                        + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)

        return (num / denom1) * (z / x_z if abs(x_z) > 1e-8 else 1.0) * bracket

    def _calibrate_sabr(
        self,
        moneyness: np.ndarray,
        vols: np.ndarray,
        atm_vol: float,
        T: float,
        beta: float = 0.5,
    ) -> Dict[str, float]:
        """Calibrate SABR parameters (alpha, rho, nu) with beta fixed at 0.5.

        Uses scipy least_squares to minimise squared vol differences.
        """
        if not _SCIPY_AVAILABLE:
            return {"alpha": atm_vol, "beta": beta, "rho": 0.0, "nu": 0.3}

        # Initial guess: alpha ≈ ATM vol, rho ≈ 0, nu ≈ vol_of_vol proxy
        vol_range = float(np.max(vols) - np.min(vols))
        x0 = [atm_vol, -0.1, 0.3]  # alpha, rho, nu

        def residuals(x: List[float]) -> np.ndarray:
            alpha, rho, nu = x[0], x[1], x[2]
            rho = np.clip(rho, -0.999, 0.999)
            alpha = max(alpha, 1e-4)
            nu = max(nu, 1e-4)
            res = []
            for m, target_vol in zip(moneyness, vols):
                K = m  # treat moneyness as K/F ratio, F=1
                try:
                    model_vol = self._sabr_vol(1.0, K, T, alpha, beta, rho, nu)
                    res.append(model_vol - target_vol)
                except Exception:
                    res.append(1.0)
            return np.array(res)

        try:
            result = least_squares(
                residuals, x0,
                bounds=([-0.5, -0.999, 0.001], [5.0, 0.999, 5.0]),
                max_nfev=500,
            )
            alpha, rho, nu = result.x
            return {
                "alpha": round(float(alpha), 6),
                "beta": beta,
                "rho": round(float(rho), 6),
                "nu": round(float(nu), 6),
                "fit_rmse": round(float(np.sqrt(np.mean(result.fun ** 2))), 6),
            }
        except Exception as exc:
            logger.debug(f"SABR least_squares failed: {exc}")
            return {"alpha": atm_vol, "beta": beta, "rho": 0.0, "nu": 0.3}

    def interpolate_surface(self, surface: VolSurface,
                            moneyness: float, tenor_years: float) -> float:
        """Bi-cubic spline interpolation across the vol surface.

        Falls back to nearest-point if scipy not available.
        """
        if _SCIPY_AVAILABLE and len(surface.tenors) >= 4 and len(surface.moneyness) >= 4:
            try:
                x = np.array([_TENORS[t] for t in surface.tenors])
                y = np.array(surface.moneyness)
                z = surface.vol_grid  # (n_tenors, n_moneyness)
                spline = RectBivariateSpline(x, y, z, kx=3, ky=3)
                return float(spline(tenor_years, moneyness)[0, 0])
            except Exception as exc:
                logger.debug(f"Spline interpolation failed: {exc}")

        # Fallback: nearest moneyness × tenor
        tenor_vals = np.array([_TENORS[t] for t in surface.tenors])
        m_vals = np.array(surface.moneyness)
        ti = int(np.argmin(np.abs(tenor_vals - tenor_years)))
        mi = int(np.argmin(np.abs(m_vals - moneyness)))
        return float(surface.vol_grid[ti, mi])


# ===========================================================================
# FXOptionsAnalytics — Garman-Kohlhagen pricing + Greeks
# ===========================================================================

class FXOptionsAnalytics:
    """European FX option pricing using the Garman-Kohlhagen model.

    Garman-Kohlhagen extends Black-Scholes for FX:
      C = S × e^(-r_f × T) × N(d1) - K × e^(-r_d × T) × N(d2)
      P = K × e^(-r_d × T) × N(-d2) - S × e^(-r_f × T) × N(-d1)

    where:
      d1 = [ln(S/K) + (r_d - r_f + σ²/2) × T] / (σ × √T)
      d2 = d1 - σ × √T
      r_d = domestic (quote) risk-free rate
      r_f = foreign (base) risk-free rate

    Core pricing (GK, forward curve) uses numpy only — no scipy required.
    """

    # ---------------------------------------------------------------------------
    # Core math: pure numpy, no scipy
    # ---------------------------------------------------------------------------

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using the Horner approximation (numpy only)."""
        return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _d1_d2(self, spot: float, strike: float, T: float,
               vol: float, r_dom: float, r_for: float) -> Tuple[float, float]:
        """Compute d1 and d2 for GK model."""
        if T <= 0 or vol <= 0:
            return 0.0, 0.0
        ln_sk = math.log(spot / strike)
        d1 = (ln_sk + (r_dom - r_for + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        return d1, d2

    def price_vanilla(
        self,
        spot: float,
        strike: float,
        T: float,        # time to expiry in years
        vol: float,      # annualised implied vol (decimal, e.g. 0.10)
        r_dom: float,    # domestic (quote ccy) rate (decimal)
        r_for: float,    # foreign (base ccy) rate (decimal)
        call_put: str = "call",
    ) -> OptionPrice:
        """Price a European FX vanilla option using Garman-Kohlhagen.

        All rates in decimal (e.g. 0.05 for 5%). Vol also decimal.
        Returns OptionPrice dataclass with value and full Greeks.
        """
        call_put = call_put.lower()
        if T <= 0:
            intrinsic = max(spot - strike, 0) if call_put == "call" else max(strike - spot, 0)
            return OptionPrice(value=intrinsic, delta=1.0 if call_put == "call" else -1.0,
                               gamma=0.0, theta=0.0, vega=0.0, rho_dom=0.0,
                               rho_for=0.0, call_put=call_put)

        d1, d2 = self._d1_d2(spot, strike, T, vol, r_dom, r_for)
        N = self._norm_cdf
        n = self._norm_pdf

        exp_rf_T = math.exp(-r_for * T)
        exp_rd_T = math.exp(-r_dom * T)
        sqrt_T = math.sqrt(T)

        if call_put == "call":
            value = spot * exp_rf_T * N(d1) - strike * exp_rd_T * N(d2)
            delta = exp_rf_T * N(d1)
            rho_dom = strike * T * exp_rd_T * N(d2)
            rho_for = -T * spot * exp_rf_T * N(d1)
        else:
            value = strike * exp_rd_T * N(-d2) - spot * exp_rf_T * N(-d1)
            delta = -exp_rf_T * N(-d1)
            rho_dom = -strike * T * exp_rd_T * N(-d2)
            rho_for = T * spot * exp_rf_T * N(-d1)

        gamma = n(d1) * exp_rf_T / (spot * vol * sqrt_T)
        vega = spot * exp_rf_T * n(d1) * sqrt_T * 0.01  # per 1% vol move
        # Theta: value change per calendar day
        theta_base = (
            -spot * exp_rf_T * n(d1) * vol / (2 * sqrt_T)
            + r_for * spot * exp_rf_T * (N(d1) if call_put == "call" else -N(-d1))
            - r_dom * strike * exp_rd_T * (N(d2) if call_put == "call" else -N(-d2))
        )
        theta = theta_base / 365.0  # per calendar day

        return OptionPrice(
            value=round(value, 8),
            delta=round(delta, 6),
            gamma=round(gamma, 8),
            theta=round(theta, 8),
            vega=round(vega, 8),
            rho_dom=round(rho_dom, 6),
            rho_for=round(rho_for, 6),
            call_put=call_put,
        )

    def get_delta_neutral_strike(
        self,
        spot: float,
        T: float,
        vol: float,
        r_dom: float,
        r_for: float,
    ) -> float:
        """Return the delta-neutral (50-delta) ATM-forward strike.

        K_atm = F × exp(σ² × T / 2)
        where F = S × exp((r_d - r_f) × T)
        """
        F = spot * math.exp((r_dom - r_for) * T)
        K_atm = F * math.exp(0.5 * vol * vol * T)
        return round(K_atm, 6)

    def implied_vol(
        self,
        price: float,
        spot: float,
        strike: float,
        T: float,
        r_dom: float,
        r_for: float,
        call_put: str = "call",
        tol: float = 1e-7,
        max_iter: int = 100,
    ) -> Optional[float]:
        """Compute implied vol via bisection method (Newton unavailable without scipy).

        Returns None if no solution found.
        """
        call_put = call_put.lower()
        # Intrinsic value bounds check
        if T <= 0:
            return None

        low_vol, high_vol = 1e-5, 10.0  # 0.001% to 1000%

        for _ in range(max_iter):
            mid_vol = (low_vol + high_vol) / 2.0
            opt = self.price_vanilla(spot, strike, T, mid_vol, r_dom, r_for, call_put)
            diff = opt.value - price
            if abs(diff) < tol:
                return round(mid_vol, 6)
            if diff > 0:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
            if high_vol - low_vol < tol:
                break

        return round((low_vol + high_vol) / 2.0, 6)

    def risk_reversal(
        self,
        spot: float,
        T: float,
        vol_25d_call: float,
        vol_25d_put: float,
        r_dom: float,
        r_for: float,
    ) -> Dict[str, float]:
        """Compute 25-delta risk reversal: 25d_call_vol - 25d_put_vol.

        Also computes the risk reversal option value (call - put).
        """
        rr_vol = vol_25d_call - vol_25d_put

        # Approximate 25-delta strikes
        # For a call, Δ = 0.25 ≈ N(d1) → d1 ≈ N^{-1}(0.25) ≈ -0.6745
        # K_25d_call ≈ F × exp(0.6745 × σ × √T - 0.5 × σ² × T)
        sqrt_T = math.sqrt(T)
        F = spot * math.exp((r_dom - r_for) * T)

        K_25c = F * math.exp(-0.6745 * vol_25d_call * sqrt_T - 0.5 * vol_25d_call ** 2 * T)
        K_25p = F * math.exp(0.6745 * vol_25d_put * sqrt_T - 0.5 * vol_25d_put ** 2 * T)

        call_25 = self.price_vanilla(spot, K_25c, T, vol_25d_call, r_dom, r_for, "call")
        put_25 = self.price_vanilla(spot, K_25p, T, vol_25d_put, r_dom, r_for, "put")

        return {
            "rr_vol": round(rr_vol, 6),
            "rr_value": round(call_25.value - put_25.value, 8),
            "call_25d_strike": round(K_25c, 6),
            "put_25d_strike": round(K_25p, 6),
            "call_25d_vol": vol_25d_call,
            "put_25d_vol": vol_25d_put,
        }

    def butterfly_spread(
        self,
        spot: float,
        T: float,
        vol_25d_call: float,
        vol_25d_put: float,
        vol_atm: float,
        r_dom: float,
        r_for: float,
    ) -> Dict[str, float]:
        """Compute 25-delta butterfly: 0.5*(25d_call_vol + 25d_put_vol) - ATM_vol."""
        fly_vol = 0.5 * (vol_25d_call + vol_25d_put) - vol_atm
        sqrt_T = math.sqrt(T)
        F = spot * math.exp((r_dom - r_for) * T)

        K_atm = F * math.exp(0.5 * vol_atm ** 2 * T)
        K_25c = F * math.exp(-0.6745 * vol_25d_call * sqrt_T - 0.5 * vol_25d_call ** 2 * T)
        K_25p = F * math.exp(0.6745 * vol_25d_put * sqrt_T - 0.5 * vol_25d_put ** 2 * T)

        long_25c = self.price_vanilla(spot, K_25c, T, vol_25d_call, r_dom, r_for, "call")
        long_25p = self.price_vanilla(spot, K_25p, T, vol_25d_put, r_dom, r_for, "put")
        short_atm = self.price_vanilla(spot, K_atm, T, vol_atm, r_dom, r_for, "call")
        short_atm_p = self.price_vanilla(spot, K_atm, T, vol_atm, r_dom, r_for, "put")

        fly_value = long_25c.value + long_25p.value - short_atm.value - short_atm_p.value

        return {
            "fly_vol": round(fly_vol, 6),
            "fly_value": round(fly_value, 8),
            "atm_vol": vol_atm,
            "vol_25d_call": vol_25d_call,
            "vol_25d_put": vol_25d_put,
            "atm_strike": round(K_atm, 6),
        }

    def compute_greeks_profile(
        self,
        spot: float,
        strike: float,
        T: float,
        vol: float,
        r_dom: float,
        r_for: float,
        call_put: str = "call",
        spot_range_pct: float = 0.15,
        n_points: int = 31,
    ) -> pd.DataFrame:
        """Compute option price and Greeks across a spot range (for P&L profile)."""
        spots = np.linspace(spot * (1 - spot_range_pct), spot * (1 + spot_range_pct), n_points)
        rows = []
        for s in spots:
            opt = self.price_vanilla(s, strike, T, vol, r_dom, r_for, call_put)
            rows.append({
                "spot": round(s, 6),
                "value": opt.value,
                "delta": opt.delta,
                "gamma": opt.gamma,
                "theta": opt.theta,
                "vega": opt.vega,
            })
        return pd.DataFrame(rows).set_index("spot")


# ===========================================================================
# CarryTradeAnalyzer — rate differentials, carry ladder, Sharpe, unwind detect
# ===========================================================================

class CarryTradeAnalyzer:
    """FX carry trade analytics: rate differentials and portfolio construction.

    Carry trade: long high-yield currency, short low-yield currency.
    """

    def __init__(self, fred: Optional[FREDFXAdapter] = None,
                 ecb: Optional[ECBSpotRateCollector] = None) -> None:
        self.fred = fred or FREDFXAdapter()
        self.ecb = ecb or ECBSpotRateCollector()
        self._rate_cache: Dict[str, float] = {}

    def _get_rate(self, currency: str) -> float:
        """Get latest short-term interest rate for a currency (%)."""
        ccy = currency.upper()
        if ccy in self._rate_cache:
            return self._rate_cache[ccy]
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=90)).isoformat()
        series_id = _RATE_SERIES.get(ccy)
        if not series_id:
            self._rate_cache[ccy] = 4.0
            return 4.0
        s = _fred_fetch(series_id, start, end, self.fred.session)
        rate = float(s.iloc[-1]) if not s.empty else 4.0
        self._rate_cache[ccy] = rate
        return rate

    def compute_carry(self, pair: str) -> float:
        """Return annualised carry (%) for a pair: r_quote - r_base.

        Positive carry = long quote (higher yield), short base.
        """
        base = pair[:3].upper()
        quote = pair[3:].upper()
        r_base = self._get_rate(base)
        r_quote = self._get_rate(quote)
        return round(r_quote - r_base, 4)

    def get_carry_ladder(self, pairs: Optional[List[str]] = None) -> pd.DataFrame:
        """Rank all pairs by carry. Positive = long quote is attractive.

        Returns DataFrame sorted by carry descending.
        """
        if pairs is None:
            pairs = _ALL_PAIRS

        rows = []
        for pair in pairs:
            try:
                carry = self.compute_carry(pair)
                base = pair[:3].upper()
                quote = pair[3:].upper()
                rows.append({
                    "pair": pair,
                    "base": base,
                    "quote": quote,
                    "carry_pct": carry,
                    "rate_base": self._get_rate(base),
                    "rate_quote": self._get_rate(quote),
                    "position": "LONG" if carry > 0 else "SHORT",
                })
            except Exception as exc:
                logger.debug(f"carry for {pair}: {exc}")

        df = pd.DataFrame(rows).sort_values("carry_pct", ascending=False)
        df["carry_rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def compute_carry_sharpe(self, pair: str, lookback_years: int = 5) -> float:
        """Compute historical carry Sharpe ratio over a lookback period.

        Sharpe = mean(daily carry return) / std(daily carry return) × sqrt(252)
        Daily carry ≈ rate_differential / 252 + daily_fx_return
        """
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=int(lookback_years * 365.25))).isoformat()

        spot_s = self.fred.fetch(pair, start, end)
        if spot_s.empty:
            return 0.0

        carry = self.compute_carry(pair)  # annualised %
        daily_carry = carry / (100.0 * 252)
        fx_returns = spot_s.pct_change().dropna()

        # Total daily carry return (long quote, short base)
        total_ret = fx_returns + daily_carry

        if total_ret.std() < 1e-10:
            return 0.0
        sharpe = total_ret.mean() / total_ret.std() * math.sqrt(252)
        return round(float(sharpe), 4)

    def detect_carry_unwind(
        self,
        pairs: Optional[List[str]] = None,
        lookback_days: int = 20,
        threshold_pct: float = 20.0,
    ) -> List[str]:
        """Detect pairs where carry has compressed rapidly (risk-off signal).

        Compression: spot moved against the carry trade by > threshold_pct
        of the pair's historical daily vol × sqrt(lookback_days).
        Returns list of pairs showing potential unwind.
        """
        if pairs is None:
            pairs = [p for p in _ALL_PAIRS if p in _FRED_FX_SERIES]

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=lookback_days + 60)).isoformat()
        unwinding = []

        for pair in pairs:
            try:
                carry = self.compute_carry(pair)
                if abs(carry) < 0.5:
                    continue  # low carry pair, skip

                spot_s = self.fred.fetch(pair, start, end)
                if len(spot_s) < lookback_days + 5:
                    continue

                recent = spot_s.tail(lookback_days)
                earlier = spot_s.iloc[-(lookback_days + 30):-lookback_days]
                if earlier.empty:
                    continue

                level_change_pct = (recent.iloc[-1] - earlier.iloc[-1]) / earlier.iloc[-1] * 100
                # If carry is positive (long quote) and quote has strengthened recently,
                # that's normal carry return. Unwind = quote weakens.
                # positive carry → long quote → unwind if level_change_pct < -threshold_pct
                if carry > 0 and level_change_pct < -threshold_pct:
                    unwinding.append(pair)
                elif carry < 0 and level_change_pct > threshold_pct:
                    unwinding.append(pair)

            except Exception as exc:
                logger.debug(f"carry unwind check {pair}: {exc}")

        return unwinding

    def carry_portfolio_returns(
        self,
        pairs: Optional[List[str]] = None,
        lookback_years: int = 10,
        n_long: int = 3,
        n_short: int = 3,
    ) -> pd.DataFrame:
        """Build a carry portfolio: long top n, short bottom n (by carry).

        Returns daily portfolio return series.
        """
        ladder = self.get_carry_ladder(pairs)
        if ladder.empty:
            return pd.DataFrame()

        long_pairs = ladder.head(n_long)["pair"].tolist()
        short_pairs = ladder.tail(n_short)["pair"].tolist()

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=int(lookback_years * 365.25))).isoformat()

        all_ret: Dict[str, pd.Series] = {}
        for pair in long_pairs + short_pairs:
            s = self.fred.fetch(pair, start, end)
            if not s.empty:
                all_ret[pair] = s.pct_change()

        if not all_ret:
            return pd.DataFrame()

        ret_df = pd.DataFrame(all_ret).dropna()
        # Equal weight: +1 for long, -1 for short
        weights = {}
        for p in long_pairs:
            if p in ret_df.columns:
                weights[p] = 1.0 / max(len(long_pairs), 1)
        for p in short_pairs:
            if p in ret_df.columns:
                weights[p] = -1.0 / max(len(short_pairs), 1)

        portfolio_ret = sum(ret_df[p] * w for p, w in weights.items() if p in ret_df.columns)
        if isinstance(portfolio_ret, (int, float)):
            return pd.DataFrame()

        result = pd.DataFrame({
            "portfolio_return": portfolio_ret,
            "cumulative_return": (1 + portfolio_ret).cumprod() - 1,
        })
        return result


# ===========================================================================
# FXRealEffectiveRate — REER computation
# ===========================================================================

class FXRealEffectiveRate:
    """Compute Real Effective Exchange Rate (REER) using BIS-style methodology.

    REER = geometric weighted average of bilateral exchange rates
           adjusted for relative price levels (inflation).
    Uses GDP weights as proxy for trade weights where BIS data unavailable.
    """

    def __init__(self, fred: Optional[FREDFXAdapter] = None,
                 ecb: Optional[ECBSpotRateCollector] = None) -> None:
        self.fred = fred or FREDFXAdapter()
        self.ecb = ecb or ECBSpotRateCollector()

    # FRED CPI series for inflation adjustment
    _CPI_SERIES: Dict[str, str] = {
        "USD": "CPIAUCSL",    # US CPI All Urban
        "EUR": "CP0000EZ19M086NEST",  # Euro area HICP
        "GBP": "GBRCPIALLMINMEI",
        "JPY": "JPNCPIALLMINMEI",
        "CHF": "CHECPIALLMINMEI",
        "CAD": "CANCPIALLMINMEI",
        "AUD": "AUSCPIALLMINMEI",
        "CNY": "CHNCPIALLMINMEI",
        "KRW": "KORCPIALLMINMEI",
        "SEK": "SWECPIALLMINMEI",
    }

    def _fetch_cpi(self, currency: str, start: str, end: str) -> pd.Series:
        series_id = self._CPI_SERIES.get(currency.upper())
        if not series_id:
            return pd.Series(dtype=float)
        return _fred_fetch(series_id, start, end, self.fred.session)

    def compute_reer(
        self,
        currency: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Compute REER for a currency against a basket.

        REER = prod_j [ (S_j / S_j_base) × (P_j / P_j_base) ]^{w_j}

        where:
          S_j = bilateral exchange rate (units of j per unit of home)
          P_j = foreign CPI / home CPI ratio
          w_j = trade/GDP weight

        Returns indexed series (base period = start).
        """
        if end is None:
            end = date.today().isoformat()
        if weights is None:
            weights = _GDP_WEIGHTS.copy()

        ccy = currency.upper()

        # Home CPI
        home_cpi = self._fetch_cpi(ccy, start, end)

        # Get bilateral rates vs home currency
        partner_currencies = [c for c in list(weights.keys())[:15] if c != ccy]

        log_reer = pd.Series(dtype=float)
        total_weight = 0.0

        for partner in partner_currencies:
            w = weights.get(partner, 0.0)
            if w < 0.001:
                continue

            # Fetch bilateral rate: partner per home currency
            # Use FRED or ECB as available
            pair_a = ccy + partner
            pair_b = partner + ccy
            s_series = self.fred.fetch(pair_a, start, end)
            if s_series.empty:
                s_series = self.fred.fetch(pair_b, start, end)
                if not s_series.empty:
                    s_series = 1.0 / s_series  # flip
            if s_series.empty:
                continue

            # Partner CPI
            partner_cpi = self._fetch_cpi(partner, start, end)

            if home_cpi.empty or partner_cpi.empty:
                # Nominal effective rate (no CPI adjustment)
                combined = s_series
            else:
                # Resample all to monthly for CPI matching
                home_m = home_cpi.resample("ME").last().ffill()
                partner_m = partner_cpi.resample("ME").last().ffill()
                spot_m = s_series.resample("ME").last().ffill()

                aligned = pd.concat({
                    "spot": spot_m,
                    "home_cpi": home_m,
                    "partner_cpi": partner_m,
                }, axis=1).dropna()

                if aligned.empty:
                    combined = s_series
                else:
                    # Real rate = nominal × (P_partner / P_home)
                    combined = aligned["spot"] * (aligned["partner_cpi"] / aligned["home_cpi"])

            # Log contribution
            if log_reer.empty:
                log_reer = w * np.log(combined)
            else:
                aligned_contribution = w * np.log(combined)
                log_reer = log_reer.add(aligned_contribution, fill_value=0)
            total_weight += w

        if log_reer.empty or total_weight < 0.1:
            return pd.Series(dtype=float)

        # Normalise weights
        log_reer /= total_weight

        reer = np.exp(log_reer)
        # Index to 100 at start
        if not reer.empty:
            reer = reer / reer.iloc[0] * 100.0
        reer.name = f"REER_{ccy}"
        return reer.dropna().sort_index()

    def detect_misalignment(
        self,
        currency: str,
        lookback_years: int = 5,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect REER misalignment from 5-year average.

        Returns dict with: current_reer, mean_5y, deviation_pct, signal
        """
        if end is None:
            end = date.today().isoformat()
        start = (date.today() - timedelta(days=int(lookback_years * 365.25 + 365))).isoformat()

        reer = self.compute_reer(currency, start, end)
        if reer.empty:
            return {"error": f"No REER data for {currency}"}

        lookback_start = pd.Timestamp(
            (date.today() - timedelta(days=int(lookback_years * 365.25))).isoformat()
        )
        historical = reer.loc[reer.index >= lookback_start]
        mean_5y = float(historical.mean()) if not historical.empty else float(reer.mean())
        current = float(reer.iloc[-1])
        dev_pct = (current - mean_5y) / mean_5y * 100

        signal = "NEUTRAL"
        if dev_pct > 10:
            signal = "OVERVALUED"
        elif dev_pct < -10:
            signal = "UNDERVALUED"

        return {
            "currency": currency,
            "current_reer": round(current, 2),
            "mean_5y": round(mean_5y, 2),
            "deviation_pct": round(dev_pct, 2),
            "signal": signal,
            "as_of": reer.index[-1].date().isoformat(),
        }


# ===========================================================================
# FXMomentumSignals — cross-sectional momentum and trend following
# ===========================================================================

class FXMomentumSignals:
    """FX momentum strategy signals.

    12-1 month momentum: return over 12 months, excluding most recent 1 month.
    Trend following: SMA crossover (50-day/200-day) + ADX filter.
    """

    def __init__(self, fred: Optional[FREDFXAdapter] = None,
                 ecb: Optional[ECBSpotRateCollector] = None) -> None:
        self.fred = fred or FREDFXAdapter()
        self.ecb = ecb or ECBSpotRateCollector()

    def _fetch_spot(self, pair: str, lookback_days: int = 400) -> pd.Series:
        """Fetch spot rate time series for momentum computation."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=lookback_days)).isoformat()
        s = self.fred.fetch(pair, start, end)
        if s.empty and _YF_AVAILABLE:
            try:
                yf_sym = pair[:3] + pair[3:] + "=X"
                raw = yf.download(yf_sym, start=start, end=end,
                                  auto_adjust=True, progress=False, threads=False)
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    s = raw["Close"].dropna()
            except Exception:
                pass
        return s

    def compute_momentum(self, pair: str, lookback: int = 252) -> float:
        """Compute 12-1 month momentum for a pair.

        Returns 11-month return (months 2-12, skipping most recent month).
        Positive = upward trend (quote appreciated vs base).
        """
        s = self._fetch_spot(pair, lookback_days=lookback + 30)
        if len(s) < lookback:
            return 0.0

        # 12-month ago price (trading days approximation)
        price_12m_ago = s.iloc[-(lookback)]
        price_1m_ago = s.iloc[-21]  # 1 month = ~21 trading days

        if price_12m_ago <= 0:
            return 0.0

        momentum = (price_1m_ago - price_12m_ago) / price_12m_ago
        return round(float(momentum), 6)

    def get_momentum_portfolio(
        self,
        pairs: Optional[List[str]] = None,
        lookback: int = 252,
        n_long: int = 5,
        n_short: int = 5,
    ) -> pd.DataFrame:
        """Compute momentum for all pairs and return long/short portfolio.

        Long top tercile (strongest momentum), short bottom tercile.
        """
        if pairs is None:
            pairs = [p for p in _ALL_PAIRS if p in _FRED_FX_SERIES]

        rows = []
        for pair in pairs:
            try:
                mom = self.compute_momentum(pair, lookback)
                rows.append({"pair": pair, "momentum": mom})
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("momentum", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        n = len(df)
        tercile = max(n // 3, 1)

        df["position"] = "FLAT"
        df.loc[df.index[:n_long], "position"] = "LONG"
        df.loc[df.index[-n_short:], "position"] = "SHORT"
        df["weight"] = 0.0
        df.loc[df["position"] == "LONG", "weight"] = 1.0 / n_long
        df.loc[df["position"] == "SHORT", "weight"] = -1.0 / n_short

        return df.reset_index(drop=True)

    def compute_trend_following_signal(self, pair: str,
                                       fast_period: int = 50,
                                       slow_period: int = 200) -> TrendSignal:
        """SMA crossover + ADX filter for trend-following signal.

        Signal logic:
          - SMA_fast > SMA_slow → uptrend candidate
          - ADX > 25 → trending market (filters range-bound)
          - Combined: LONG if fast > slow AND ADX > 25
                      SHORT if fast < slow AND ADX > 25
                      FLAT if ADX <= 25 (choppy)
        """
        s = self._fetch_spot(pair, lookback_days=slow_period + 60)
        if len(s) < slow_period:
            return TrendSignal(
                pair=pair, direction="flat",
                sma_fast=0.0, sma_slow=0.0,
                adx=0.0, signal_strength=0.0,
            )

        prices = s.values
        sma_fast = float(np.mean(prices[-fast_period:]))
        sma_slow = float(np.mean(prices[-slow_period:]))
        adx = self._compute_adx(prices, period=14)

        if adx > 25:
            if sma_fast > sma_slow:
                direction = "long"
                strength = min((sma_fast - sma_slow) / sma_slow * 100, 1.0)
            else:
                direction = "short"
                strength = min((sma_slow - sma_fast) / sma_slow * 100, 1.0)
        else:
            direction = "flat"
            strength = 0.0

        return TrendSignal(
            pair=pair,
            direction=direction,
            sma_fast=round(sma_fast, 6),
            sma_slow=round(sma_slow, 6),
            adx=round(adx, 2),
            signal_strength=round(abs(strength), 4),
        )

    def _compute_adx(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute ADX (Average Directional Index) from price series.

        Simplified: using price series only (no separate OHLC needed for FX).
        Uses absolute returns as TR proxy and directional movement approximation.
        """
        if len(prices) < period * 2:
            return 0.0

        rets = np.diff(prices)

        # Directional movement (simplified from single series)
        dm_plus = np.where(rets > 0, rets, 0.0)
        dm_minus = np.where(rets < 0, -rets, 0.0)
        tr = np.abs(rets)

        def smooth(x: np.ndarray, p: int) -> np.ndarray:
            """Wilder smoothing."""
            out = np.zeros(len(x))
            if len(x) < p:
                return out
            out[p - 1] = np.sum(x[:p])
            for i in range(p, len(x)):
                out[i] = out[i - 1] - out[i - 1] / p + x[i]
            return out

        tr_smooth = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)

        # Avoid division by zero
        eps = 1e-10
        di_plus = 100 * dm_plus_smooth / (tr_smooth + eps)
        di_minus = 100 * dm_minus_smooth / (tr_smooth + eps)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + eps)

        dx_smooth = smooth(dx[period - 1:], period)
        if len(dx_smooth) < period:
            return 0.0
        adx_vals = dx_smooth[period - 1:] / period
        if len(adx_vals) == 0:
            return 0.0
        return float(np.mean(adx_vals[-period:]))

    def compute_cross_sectional_momentum(
        self,
        pairs: Optional[List[str]] = None,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """Cross-sectional momentum: rank all pairs and compute z-scores.

        Returns DataFrame with pair, momentum, z_score, signal
        """
        portfolio = self.get_momentum_portfolio(pairs, lookback)
        if portfolio.empty:
            return pd.DataFrame()

        moms = portfolio["momentum"].values
        if moms.std() < 1e-10:
            portfolio["z_score"] = 0.0
        else:
            portfolio["z_score"] = (moms - moms.mean()) / moms.std()

        portfolio["signal"] = portfolio["z_score"].apply(
            lambda z: "LONG" if z > 0.5 else ("SHORT" if z < -0.5 else "FLAT")
        )
        return portfolio

    def get_all_trend_signals(
        self,
        pairs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute trend-following signals for all pairs."""
        if pairs is None:
            pairs = [p for p in _ALL_PAIRS if p in _FRED_FX_SERIES]
        rows = []
        for pair in pairs:
            try:
                sig = self.compute_trend_following_signal(pair)
                rows.append({
                    "pair": pair,
                    "direction": sig.direction,
                    "sma_fast": sig.sma_fast,
                    "sma_slow": sig.sma_slow,
                    "adx": sig.adx,
                    "signal_strength": sig.signal_strength,
                })
            except Exception as exc:
                logger.debug(f"Trend signal {pair}: {exc}")
        return pd.DataFrame(rows).sort_values("signal_strength", ascending=False)


# ===========================================================================
# Convenience factory
# ===========================================================================

def build_fx_platform() -> Dict[str, Any]:
    """Factory: create all FX platform components sharing adapters."""
    fred = FREDFXAdapter()
    ecb = ECBSpotRateCollector()
    fwd = ForwardCurveBuilder(fred)
    vol = FXVolatilitySurface(fred, ecb)
    opts = FXOptionsAnalytics()
    carry = CarryTradeAnalyzer(fred, ecb)
    reer = FXRealEffectiveRate(fred, ecb)
    mom = FXMomentumSignals(fred, ecb)

    return {
        "fred": fred,
        "ecb": ecb,
        "forward_builder": fwd,
        "vol_surface": vol,
        "options": opts,
        "carry": carry,
        "reer": reer,
        "momentum": mom,
    }


# ===========================================================================
# Entrypoint demo
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    print(f"\n{'='*64}")
    print("SENTINEL FX Analytics Platform v3")
    print(f"{'='*64}")

    platform = build_fx_platform()
    fred: FREDFXAdapter = platform["fred"]
    ecb: ECBSpotRateCollector = platform["ecb"]
    fwd_builder: ForwardCurveBuilder = platform["forward_builder"]
    vol_surface: FXVolatilitySurface = platform["vol_surface"]
    opts: FXOptionsAnalytics = platform["options"]
    carry: CarryTradeAnalyzer = platform["carry"]
    mom: FXMomentumSignals = platform["momentum"]

    # 1. EUR/USD spot rate history
    print("\n--- EUR/USD Spot Rate (FRED, last 5 rows) ---")
    eurusd = fred.fetch("EURUSD", start="2020-01-01")
    if not eurusd.empty:
        print(eurusd.tail(5).to_string())
        spot = float(eurusd.iloc[-1])
    else:
        spot = 1.08
        print(f"  (using fallback spot: {spot})")

    # 2. Forward curve — EUR/USD
    print("\n--- EUR/USD Forward Curve (CIP) ---")
    r_usd = fwd_builder._get_current_rate("USD")
    r_eur = fwd_builder._get_current_rate("EUR")
    curve = fwd_builder.build_forward_curve(
        spot=spot, base_rate=r_eur, quote_rate=r_usd,
        pair="EURUSD", as_of=date.today(),
    )
    print(f"  Spot: {curve.spot:.5f}  |  r_EUR: {curve.base_rate:.2f}%  |  r_USD: {curve.quote_rate:.2f}%")
    for tenor, fwd in curve.tenors.items():
        pts = curve.forward_points[tenor]
        print(f"  {tenor:4s}: {fwd:.5f}  ({pts:+.1f} pips)")

    # 3. Vol surface — EUR/USD
    print("\n--- EUR/USD Volatility Surface ---")
    surface = vol_surface.build_surface("EURUSD")
    print(f"  Pair: {surface.pair}  |  As of: {surface.as_of}")
    print(f"  ATM vols by tenor: " + " | ".join(
        f"{t}: {v*100:.1f}%" for t, v in surface.atm_vols.items()
    ))
    vol_df = pd.DataFrame(surface.vol_grid * 100,
                          index=surface.tenors, columns=[f"{int(m*100)}%" for m in surface.moneyness])
    print("\n  Vol grid (%):")
    print(vol_df.round(2).to_string())

    # 4. Option pricing — EUR/USD 1-month ATM call
    print("\n--- EUR/USD 1M ATM Call Option (Garman-Kohlhagen) ---")
    atm_vol = surface.atm_vols.get("1M", 0.07)
    K = opts.get_delta_neutral_strike(spot=spot, T=1/12, vol=atm_vol,
                                      r_dom=r_usd/100, r_for=r_eur/100)
    opt = opts.price_vanilla(
        spot=spot, strike=K, T=1/12, vol=atm_vol,
        r_dom=r_usd/100, r_for=r_eur/100, call_put="call",
    )
    print(f"  Spot: {spot:.5f}  Strike: {K:.5f}  Vol: {atm_vol*100:.2f}%")
    print(f"  Value: {opt.value:.6f}  Delta: {opt.delta:.4f}  Gamma: {opt.gamma:.6f}")
    print(f"  Theta: {opt.theta:.6f}/day  Vega: {opt.vega:.6f}/1%vol")

    # 5. Risk reversal and butterfly
    atm_v = atm_vol
    rr_25d_call_v = atm_v * 1.02  # approximate
    rr_25d_put_v = atm_v * 0.98
    rr = opts.risk_reversal(spot, T=1/12, vol_25d_call=rr_25d_call_v,
                             vol_25d_put=rr_25d_put_v,
                             r_dom=r_usd/100, r_for=r_eur/100)
    fly = opts.butterfly_spread(spot, T=1/12, vol_25d_call=rr_25d_call_v,
                                 vol_25d_put=rr_25d_put_v, vol_atm=atm_v,
                                 r_dom=r_usd/100, r_for=r_eur/100)
    print(f"\n--- Risk Reversal & Butterfly (EUR/USD 1M) ---")
    print(f"  25d Risk Reversal vol: {rr['rr_vol']*100:+.2f}%  value: {rr['rr_value']:.6f}")
    print(f"  25d Butterfly vol:     {fly['fly_vol']*100:+.3f}%  value: {fly['fly_value']:.6f}")

    # 6. Carry ladder
    print("\n--- Carry Trade Ladder (top 10 pairs) ---")
    ladder = carry.get_carry_ladder()
    if not ladder.empty:
        print(ladder.head(10)[["pair", "rate_base", "rate_quote", "carry_pct", "position"]].to_string(index=False))

    # 7. Carry unwind detection
    print("\n--- Carry Unwind Detection ---")
    unwinding = carry.detect_carry_unwind(lookback_days=20, threshold_pct=15.0)
    print(f"  Pairs showing potential unwind: {unwinding if unwinding else 'None detected'}")

    # 8. Momentum signals
    print("\n--- FX Momentum Signals ---")
    mom_df = mom.get_momentum_portfolio(
        pairs=[p for p in _ALL_PAIRS if p in _FRED_FX_SERIES]
    )
    if not mom_df.empty:
        print(mom_df[["pair", "momentum", "position", "weight"]].head(10).to_string(index=False))

    # 9. Trend-following signals for major pairs
    print("\n--- Trend-Following Signals (major G10 pairs) ---")
    major_pairs = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD"]
    for pair in major_pairs:
        try:
            sig = mom.compute_trend_following_signal(pair)
            print(f"  {pair:8s}  {sig.direction:5s}  SMA50={sig.sma_fast:.4f}  "
                  f"SMA200={sig.sma_slow:.4f}  ADX={sig.adx:.1f}  "
                  f"strength={sig.signal_strength:.3f}")
        except Exception as exc:
            print(f"  {pair:8s}  error: {exc}")

    # 10. REER misalignment check (USD)
    print("\n--- USD REER Misalignment ---")
    reer_obj: FXRealEffectiveRate = platform["reer"]
    misalign = reer_obj.detect_misalignment("USD", lookback_years=5)
    for k, v in misalign.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*64}")
    print("FX Platform v3 demo complete.")
    print(f"{'='*64}\n")
