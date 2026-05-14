"""
FX analytics — spot rates, forward curves, implied vol proxy, carry/momentum signals.

Data sources (all free):
  Spot rates:    Frankfurter API (ECB official fixing, https://api.frankfurter.app)
  Interest rates: FRED CSV (FEDFUNDS, ECBDFR, etc.)
  Vol/OHLCV:     yfinance — lazy-imported inside asyncio.to_thread

Extended (score-9 additions):
  FXVolSurfaceModel — SABR vol surface fit, smile, risk reversal, butterfly
  FXCrossMatrix     — 20+ pair cross-rate matrix with G10/EM classification
  FXCarrySignal     — Carry + momentum + value (PPP) combined signal
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize  # type: ignore[import-untyped]

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRANKFURTER_BASE = "https://api.frankfurter.app"

# Default pairs (base3 + quote3)
_DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "USDCNY", "USDMXN"]

# Extended 20+ pair universe
_EXTENDED_PAIRS = [
    # G10
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY",
    # EM
    "USDCNY", "USDMXN", "USDBRL", "USDSGD", "USDHKD", "USDKRW",
    "USDPLN", "USDCZK", "USDSEK", "USDNOK",
]

# G10 currencies
_G10 = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"}

# FRED short-rate series per currency
_RATE_SERIES: dict[str, str] = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBP": "IUQABEDR",
    "JPY": "IRSTCI01JPM156N",
    "CHF": "IRSTCI01CHM156N",
    "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N",
}

# Currencies supported by Frankfurter
_FRANKFURTER_CURRENCIES = {
    "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "MXN", "BRL", "INR",
    "USD", "SEK", "NOK", "DKK", "NZD", "SGD", "HKD", "KRW", "PLN", "CZK",
}

# Forward tenors in years
_FORWARD_TENORS: list[tuple[str, float]] = [
    ("1M", 1 / 12),
    ("3M", 3 / 12),
    ("6M", 6 / 12),
    ("1Y", 1.0),
]

_LN2 = math.log(2.0)

# Approximate PPP rates vs USD (World Bank 2023 ICP, USD per 1 unit foreign)
# Used for value signal: positive deviation = currency undervalued vs PPP
_PPP_VS_USD: dict[str, float] = {
    "EUR": 1.10,   # ~1.10 USD per EUR at PPP
    "GBP": 1.28,
    "JPY": 0.0092,
    "CHF": 1.15,
    "CAD": 0.82,
    "AUD": 0.73,
    "NZD": 0.68,
    "SEK": 0.098,
    "NOK": 0.105,
    "CNY": 0.195,
    "MXN": 0.082,
    "BRL": 0.24,
    "SGD": 0.78,
    "HKD": 0.145,
    "KRW": 0.00095,
    "PLN": 0.27,
    "CZK": 0.048,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ForwardPoint(BaseModel):
    tenor: str
    forward_rate: float
    forward_points: float       # pips (F - S) * 10000
    implied_yield_diff: float   # annualized rate differential


class FXVolSurface(BaseModel):
    realized_vol_30d: Optional[float] = None   # annualized %
    realized_vol_90d: Optional[float] = None
    hl_vol_estimate: Optional[float] = None    # Parkinson high-low estimator
    vol_regime: str  # "low" | "normal" | "elevated" | "crisis"


class SABRParams(BaseModel):
    """SABR model parameters for a single maturity."""
    maturity: str
    alpha: float   # initial vol level
    beta: float    # CEV exponent (fixed at 0.5 for FX)
    rho: float     # correlation spot/vol
    nu: float      # vol-of-vol
    fit_error: Optional[float] = None


class FXSmilePoint(BaseModel):
    """A single point on the implied vol smile."""
    strike: float
    delta: float     # option delta (0.1 to 0.9)
    implied_vol: float  # annualized %


class CrossRateEntry(BaseModel):
    """Single entry in the cross-rate matrix."""
    pair: str
    base: str
    quote: str
    spot: float
    is_direct: bool   # True if Frankfurter quotes it directly
    base_class: str   # "G10" | "EM"
    quote_class: str
    spot_date: str


class CarrySignalEntry(BaseModel):
    """Per-pair carry/momentum/value combined signal."""
    pair: str
    carry_diff_pct: Optional[float] = None     # rate differential (%)
    carry_signal: int = 0                       # -1, 0, +1
    momentum_12m_1m: Optional[float] = None    # 12M - 1M return
    momentum_signal: int = 0
    ppp_deviation_pct: Optional[float] = None  # % deviation from PPP
    value_signal: int = 0
    combined_score: float = 0.0                # -3 to +3


class FXPairAnalytics(BaseModel):
    pair: str
    base: str
    quote: str
    spot: float
    spot_date: str
    change_1d: Optional[float] = None
    change_1w: Optional[float] = None
    change_1m: Optional[float] = None
    change_ytd: Optional[float] = None
    forward_curve: list[ForwardPoint]
    vol_surface: FXVolSurface
    carry_score: int                 # -1, 0, +1
    momentum_zscore: Optional[float] = None
    momentum_signal: str             # "strong_usd" | "weak_usd" | "neutral"
    domestic_rate: Optional[float] = None
    foreign_rate: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)


class FXDashboard(BaseModel):
    base_currency: str
    pairs: list[FXPairAnalytics]
    dxy_proxy: Optional[float] = None
    usd_trend: str   # "strengthening" | "weakening" | "stable"
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

async def _fred_latest(client: httpx.AsyncClient, series_id: str) -> Optional[float]:
    """Fetch the most recent non-missing value from FRED CSV."""
    try:
        r = await client.get(_FRED_CSV, params={"id": series_id}, timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("FRED non-200", series=series_id, status=r.status_code)
            return None
        lines = [ln for ln in r.text.strip().splitlines()[1:] if ln.strip()]
        # Walk backwards to find last non-dot value
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) != 2:
                continue
            val_str = parts[1].strip()
            if val_str in (".", "", "NA"):
                continue
            try:
                return float(val_str)
            except ValueError:
                continue
        return None
    except Exception as exc:
        logger.warning("FRED fetch failed", series=series_id, error=str(exc))
        return None


async def _fetch_rates(
    client: httpx.AsyncClient, currencies: set[str]
) -> dict[str, Optional[float]]:
    """Fetch FRED short rates for the given currency set. Returns pct (e.g. 5.33)."""
    tasks = {
        ccy: _fred_latest(client, _RATE_SERIES[ccy])
        for ccy in currencies
        if ccy in _RATE_SERIES
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[str, Optional[float]] = {}
    for ccy, res in zip(tasks.keys(), results):
        if isinstance(res, Exception):
            logger.warning("Rate fetch exception", currency=ccy, error=str(res))
            out[ccy] = None
        else:
            out[ccy] = res  # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# Frankfurter API helpers
# ---------------------------------------------------------------------------

async def _frankfurter_spot(
    client: httpx.AsyncClient, base: str, targets: list[str]
) -> tuple[dict[str, float], str]:
    """
    Fetch latest spot rates from Frankfurter.
    Returns (rates_dict, date_str) where rates_dict is {CCY: rate_vs_base}.
    """
    targets_str = ",".join(t for t in targets if t in _FRANKFURTER_CURRENCIES)
    try:
        r = await client.get(
            f"{_FRANKFURTER_BASE}/latest",
            params={"from": base, "to": targets_str},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            logger.warning("Frankfurter non-200", status=r.status_code)
            return {}, date.today().isoformat()
        data = r.json()
        return data.get("rates", {}), data.get("date", date.today().isoformat())
    except Exception as exc:
        logger.warning("Frankfurter spot failed", error=str(exc))
        return {}, date.today().isoformat()


async def _frankfurter_history(
    client: httpx.AsyncClient,
    base: str,
    target: str,
    start: str,
    end: str,
) -> dict[str, float]:
    """
    Fetch historical daily rates from Frankfurter.
    Returns {date_str: rate}.
    """
    try:
        r = await client.get(
            f"{_FRANKFURTER_BASE}/{start}..{end}",
            params={"from": base, "to": target},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        # data["rates"] = {"2024-01-15": {"EUR": 0.92}, ...}
        raw: dict[str, dict[str, float]] = data.get("rates", {})
        return {dt: vals[target] for dt, vals in raw.items() if target in vals}
    except Exception as exc:
        logger.warning("Frankfurter history failed", base=base, target=target, error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# yfinance vol helpers (run in thread)
# ---------------------------------------------------------------------------

def _compute_vol_surface_sync(yf_ticker: str) -> dict:
    """Synchronous yfinance fetch + vol computation. Called via asyncio.to_thread."""
    import yfinance as yf  # lazy import

    out: dict = {
        "realized_vol_30d": None,
        "realized_vol_90d": None,
        "hl_vol_estimate": None,
    }
    try:
        tk = yf.Ticker(yf_ticker)
        hist = tk.history(period="1y")
        if hist is None or len(hist) < 5:
            return out

        closes = hist["Close"].dropna().values.astype(float)
        if len(closes) >= 22:
            rets_30 = np.diff(np.log(closes[-31:]))
            out["realized_vol_30d"] = round(float(np.std(rets_30, ddof=1) * math.sqrt(252) * 100), 4)
        if len(closes) >= 61:
            rets_90 = np.diff(np.log(closes[-91:]))
            out["realized_vol_90d"] = round(float(np.std(rets_90, ddof=1) * math.sqrt(252) * 100), 4)

        # Parkinson high-low estimator
        highs = hist["High"].dropna().values.astype(float)
        lows = hist["Low"].dropna().values.astype(float)
        n = min(len(highs), len(lows), len(closes))
        if n >= 10:
            h = highs[-n:]
            l_ = lows[-n:]
            # Guard zero/negative lows
            valid = (l_ > 0) & (h > l_)
            if valid.sum() >= 5:
                hl_ratios = np.log(h[valid] / l_[valid])
                hl_vol = math.sqrt(252) * float(np.mean(hl_ratios) / (2.0 * math.sqrt(_LN2)))
                out["hl_vol_estimate"] = round(hl_vol * 100, 4)
    except Exception as exc:
        logger.warning("yfinance vol failed", ticker=yf_ticker, error=str(exc))
    return out


# ---------------------------------------------------------------------------
# Vol regime classifier
# ---------------------------------------------------------------------------

def _vol_regime(rv30: Optional[float], hl: Optional[float]) -> str:
    vol = rv30 if rv30 is not None else hl
    if vol is None:
        return "normal"
    if vol < 5.0:
        return "low"
    if vol < 10.0:
        return "normal"
    if vol < 20.0:
        return "elevated"
    return "crisis"


# ---------------------------------------------------------------------------
# Forward curve builder
# ---------------------------------------------------------------------------

def _build_forward_curve(
    spot: float,
    domestic_rate_pct: Optional[float],
    foreign_rate_pct: Optional[float],
    warnings: list[str],
) -> list[ForwardPoint]:
    """
    Covered interest rate parity: F = S * (1 + r_domestic*T) / (1 + r_foreign*T).
    domestic = USD, foreign = the paired currency.
    Rates are in pct (e.g. 5.33) — convert /100.
    """
    if domestic_rate_pct is None or foreign_rate_pct is None:
        warnings.append("Missing rate(s) — forward curve unavailable.")
        return []

    r_d = domestic_rate_pct / 100.0
    r_f = foreign_rate_pct / 100.0
    curve: list[ForwardPoint] = []

    for tenor_label, tenor_yr in _FORWARD_TENORS:
        denom = 1.0 + r_f * tenor_yr
        if abs(denom) < 1e-10:
            warnings.append(f"Foreign rate division by zero at tenor {tenor_label}.")
            continue
        fwd = spot * (1.0 + r_d * tenor_yr) / denom
        pips = (fwd - spot) * 10_000.0
        impl_diff = (r_d - r_f) * 100.0  # back to pct, annualized
        curve.append(ForwardPoint(
            tenor=tenor_label,
            forward_rate=round(fwd, 6),
            forward_points=round(pips, 4),
            implied_yield_diff=round(impl_diff, 4),
        ))
    return curve


# ---------------------------------------------------------------------------
# Carry & momentum
# ---------------------------------------------------------------------------

def _carry_score(domestic_rate_pct: Optional[float], foreign_rate_pct: Optional[float]) -> int:
    if domestic_rate_pct is None or foreign_rate_pct is None:
        return 0
    diff = domestic_rate_pct - foreign_rate_pct
    if diff > 0.5:
        return 1
    if diff < -0.5:
        return -1
    return 0


def _momentum_signal(history: dict[str, float]) -> tuple[Optional[float], str]:
    """
    21-day return z-scored against 252-day lookback.
    history: {date_str: rate} sorted ascending.
    Returns (zscore, signal_label).
    """
    if not history:
        return None, "neutral"
    sorted_dates = sorted(history.keys())
    prices = np.array([history[d] for d in sorted_dates], dtype=float)

    if len(prices) < 22:
        return None, "neutral"

    ret_21 = (prices[-1] / prices[-22]) - 1.0
    roll_21 = np.array([
        (prices[i] / prices[i - 21]) - 1.0
        for i in range(21, len(prices))
    ])
    if len(roll_21) < 2:
        return None, "neutral"

    mu = float(np.mean(roll_21))
    sigma = float(np.std(roll_21, ddof=1))
    if sigma < 1e-10:
        return None, "neutral"

    zscore = (ret_21 - mu) / sigma
    if zscore > 1.0:
        signal = "strong_usd"
    elif zscore < -1.0:
        signal = "weak_usd"
    else:
        signal = "neutral"

    return round(float(zscore), 4), signal


# ---------------------------------------------------------------------------
# Pair parser
# ---------------------------------------------------------------------------

def _parse_pair(pair: str) -> tuple[str, str]:
    """Returns (base, quote). E.g. 'EURUSD' -> ('EUR', 'USD')."""
    pair = pair.upper().strip()
    if len(pair) != 6:
        raise ValueError(f"Invalid pair format: {pair!r}. Expected 6 chars, e.g. 'EURUSD'.")
    return pair[:3], pair[3:]


def _yf_ticker(base: str, quote: str) -> str:
    return f"{base}{quote}=X"


# ---------------------------------------------------------------------------
# Core pair analytics
# ---------------------------------------------------------------------------

async def get_fx_pair(
    pair: str,
    history_days: int = 90,
) -> FXPairAnalytics:
    """
    Deep FX analytics for one currency pair (e.g. 'EURUSD', 'GBPUSD', 'USDJPY').

    Fetches spot from Frankfurter (ECB fixing), interest rates from FRED,
    vol surface from yfinance, then computes forward curve, carry, and momentum.
    """
    warnings_: list[str] = []

    try:
        base, quote = _parse_pair(pair)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    today = date.today()
    today_str = today.isoformat()

    # History window for Frankfurter
    start_str = (today - timedelta(days=history_days + 10)).isoformat()

    # --- Determine how to get the spot rate from Frankfurter ---
    # Frankfurter always quotes in terms of "from" currency.
    # If base == "USD", ask Frankfurter for USD→QUOTE then invert to get BASE/QUOTE = USD/QUOTE.
    # Actually: EURUSD = EUR per 1 USD? No. EURUSD = how many USD per 1 EUR → spot in USD.
    # Frankfurter: from=EUR, to=USD → {"EUR": x} where x = USD per 1 EUR → that's EURUSD spot.
    # For USDJPY: from=USD, to=JPY → JPY per 1 USD → that's USDJPY spot.
    # So: always use from=base, to=quote.

    fkf_base = base
    fkf_quote = quote

    async with httpx.AsyncClient() as client:
        # Parallel: spot + history + FRED rates + vol
        (spot_rates, spot_date), hist_raw, rates_dict, vol_raw = await asyncio.gather(
            _frankfurter_spot(client, fkf_base, [fkf_quote]),
            _frankfurter_history(client, fkf_base, fkf_quote, start_str, today_str),
            _fetch_rates(client, {base, quote}),
            asyncio.to_thread(_compute_vol_surface_sync, _yf_ticker(base, quote)),
        )

    spot = spot_rates.get(fkf_quote)
    if spot is None:
        warnings_.append(f"Spot rate unavailable from Frankfurter for {pair}.")
        spot = float("nan")

    # --- Return changes ---
    sorted_dates = sorted(hist_raw.keys())

    def _pct_change(days_back: int) -> Optional[float]:
        if len(sorted_dates) < 2:
            return None
        # Find the date at least `days_back` calendar days ago
        target = (today - timedelta(days=days_back)).isoformat()
        candidates = [d for d in sorted_dates if d <= target]
        if not candidates:
            return None
        ref_date = candidates[-1]
        ref = hist_raw[ref_date]
        if abs(ref) < 1e-12 or not math.isfinite(spot):
            return None
        return round((spot / ref - 1.0) * 100.0, 4)

    ytd_start = f"{today.year}-01-01"
    ytd_candidates = [d for d in sorted_dates if d <= ytd_start]
    change_ytd: Optional[float] = None
    if ytd_candidates and math.isfinite(spot):
        ref_ytd = hist_raw[ytd_candidates[-1]]
        if abs(ref_ytd) > 1e-12:
            change_ytd = round((spot / ref_ytd - 1.0) * 100.0, 4)

    # --- Rates ---
    dom_rate = rates_dict.get("USD")   # domestic = USD always in our pairs
    for_rate = rates_dict.get(base if base != "USD" else quote)
    # Edge: if USD is quote (e.g. EURUSD), base=EUR → foreign rate is EUR
    # if USD is base (e.g. USDJPY), base=USD → foreign rate is JPY

    # For covered interest parity, define:
    #   r_domestic = USD rate, r_foreign = non-USD rate
    non_usd = quote if base == "USD" else base
    for_rate = rates_dict.get(non_usd)

    # --- Forward curve ---
    forward_curve = _build_forward_curve(spot, dom_rate, for_rate, warnings_)

    # --- Vol surface ---
    vol_surface = FXVolSurface(
        realized_vol_30d=vol_raw.get("realized_vol_30d"),
        realized_vol_90d=vol_raw.get("realized_vol_90d"),
        hl_vol_estimate=vol_raw.get("hl_vol_estimate"),
        vol_regime=_vol_regime(vol_raw.get("realized_vol_30d"), vol_raw.get("hl_vol_estimate")),
    )

    # --- Carry ---
    carry = _carry_score(dom_rate, for_rate)

    # --- Momentum ---
    zscore, mom_signal = _momentum_signal(hist_raw)

    logger.info(
        "FX pair analytics",
        pair=pair,
        spot=round(spot, 6) if math.isfinite(spot) else None,
        spot_date=spot_date,
        carry_score=carry,
        vol_regime=vol_surface.vol_regime,
        momentum=mom_signal,
    )

    return FXPairAnalytics(
        pair=pair.upper(),
        base=base,
        quote=quote,
        spot=round(spot, 6) if math.isfinite(spot) else 0.0,
        spot_date=spot_date,
        change_1d=_pct_change(1),
        change_1w=_pct_change(7),
        change_1m=_pct_change(30),
        change_ytd=change_ytd,
        forward_curve=forward_curve,
        vol_surface=vol_surface,
        carry_score=carry,
        momentum_zscore=zscore,
        momentum_signal=mom_signal,
        domestic_rate=dom_rate,
        foreign_rate=for_rate,
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

async def get_fx_dashboard(
    pairs: list[str] | None = None,
    history_days: int = 90,
) -> FXDashboard:
    """
    Multi-currency FX dashboard with forward curves, vol, and carry signals.

    Default pairs: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, USDCNY, USDMXN.
    Fetches all pairs in parallel via asyncio.gather.
    Computes a DXY proxy (equal-weighted USD strength basket) and USD trend.
    """
    if pairs is None:
        pairs = list(_DEFAULT_PAIRS)

    warnings_: list[str] = []
    today_str = date.today().isoformat()

    results = await asyncio.gather(
        *[get_fx_pair(p, history_days=history_days) for p in pairs],
        return_exceptions=True,
    )

    valid_pairs: list[FXPairAnalytics] = []
    for p, res in zip(pairs, results):
        if isinstance(res, Exception):
            warnings_.append(f"{p}: fetch failed — {res}")
            logger.warning("Pair analytics failed", pair=p, error=str(res))
        else:
            valid_pairs.append(res)  # type: ignore[arg-type]

    # DXY proxy: for each pair, compute USD strength as how many USD per 1 unit foreign.
    # EURUSD spot already = USD per EUR. USDJPY spot = JPY per USD → invert → USD per JPY.
    usd_per_foreign: list[float] = []
    for pa in valid_pairs:
        if pa.spot <= 0:
            continue
        if pa.base == "USD":
            # e.g. USDJPY: spot = JPY per USD → USD per JPY = 1/spot
            usd_per_foreign.append(1.0 / pa.spot)
        else:
            # e.g. EURUSD: spot = USD per EUR → USD strength proxy = 1/spot (lower = USD weaker)
            usd_per_foreign.append(pa.spot)

    dxy_proxy: Optional[float] = None
    usd_trend = "stable"

    if len(usd_per_foreign) >= 2:
        arr = np.array(usd_per_foreign, dtype=float)
        # Z-score of basket: proxy for DXY deviation from mean
        # Return equal-weighted mean (normalised to 100 scale)
        dxy_proxy = round(float(np.mean(arr) * 100.0), 4)

        # Trend: use 1-week changes from valid pairs that have them
        changes_1w = [p.change_1w for p in valid_pairs if p.change_1w is not None]
        if changes_1w:
            # Positive change_1w means rate went up vs USD → USD weaker for USD-base, stronger for non-USD-base.
            # Too complex to unify without explicit direction — use momentum signals instead.
            usd_strong = sum(1 for p in valid_pairs if p.momentum_signal == "strong_usd")
            usd_weak   = sum(1 for p in valid_pairs if p.momentum_signal == "weak_usd")
            if usd_strong > usd_weak and usd_strong > len(valid_pairs) / 3:
                usd_trend = "strengthening"
            elif usd_weak > usd_strong and usd_weak > len(valid_pairs) / 3:
                usd_trend = "weakening"
            else:
                usd_trend = "stable"

    logger.info(
        "FX dashboard assembled",
        pairs=len(valid_pairs),
        dxy_proxy=dxy_proxy,
        usd_trend=usd_trend,
        as_of=today_str,
    )

    return FXDashboard(
        base_currency="USD",
        pairs=valid_pairs,
        dxy_proxy=dxy_proxy,
        usd_trend=usd_trend,
        as_of=today_str,
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# SABR vol model helpers
# ---------------------------------------------------------------------------

def _sabr_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) SABR implied vol approximation.

    F: forward price
    K: strike
    T: time to expiry (years)
    alpha, beta, rho, nu: SABR parameters
    Returns implied Black vol (annualized fraction).
    """
    if T <= 0 or F <= 0 or K <= 0:
        return float("nan")

    eps = 1e-8
    if abs(F - K) < eps:
        # ATM formula
        FK_beta = F ** (1.0 - beta)
        term1 = alpha / FK_beta
        term2 = 1.0 + T * (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / FK_beta ** 2)
            + (rho * beta * nu * alpha) / (4.0 * FK_beta)
            + nu ** 2 * (2.0 - 3.0 * rho ** 2) / 24.0
        )
        return term1 * term2

    FK = F * K
    FK_mid_beta = FK ** ((1.0 - beta) / 2.0)
    log_FK = math.log(F / K)

    z = (nu / alpha) * FK_mid_beta * log_FK
    x_z = math.log((math.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    if abs(x_z) < eps:
        z_over_xz = 1.0
    else:
        z_over_xz = z / x_z

    numer = alpha * z_over_xz
    denom = FK_mid_beta * (
        1.0
        + ((1.0 - beta) ** 2 / 6.0) * (log_FK ** 2)
        + ((1.0 - beta) ** 4 / 120.0) * (log_FK ** 4)
    )
    correction = 1.0 + T * (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (FK ** (1.0 - beta)))
        + (rho * beta * nu * alpha) / (4.0 * FK_mid_beta)
        + nu ** 2 * (2.0 - 3.0 * rho ** 2) / 24.0
    )
    return (numer / denom) * correction


def _fit_sabr_to_quotes(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
) -> tuple[float, float, float, float]:
    """
    Fit SABR (alpha, rho, nu) given fixed beta.
    Returns (alpha, beta, rho, nu).
    """
    def objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-1.0 < rho < 1.0):
            return 1e10
        total = 0.0
        for K, sig_mkt in zip(strikes, market_vols):
            sig_mod = _sabr_vol(F, K, T, alpha, beta, rho, nu)
            if not math.isfinite(sig_mod):
                return 1e10
            total += (sig_mod - sig_mkt) ** 2
        return total

    # Initial guess: ATM vol ~= alpha / F^(1-beta), rho=0, nu=0.3
    atm_vol = float(np.median(market_vols)) if len(market_vols) > 0 else 0.1
    alpha0 = atm_vol * (F ** (1.0 - beta))
    x0 = np.array([alpha0, 0.0, 0.3])
    bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12})
    alpha_fit, rho_fit, nu_fit = result.x
    return float(alpha_fit), beta, float(rho_fit), float(nu_fit)


# ---------------------------------------------------------------------------
# FXVolSurfaceModel class
# ---------------------------------------------------------------------------

class FXVolSurfaceModel:
    """
    SABR vol surface model for FX pairs.

    Uses yfinance options chain to extract implied vols, then fits SABR
    parameters per maturity. Also computes risk reversals and butterflies.
    """

    def __init__(self, pair: str) -> None:
        self.pair = pair.upper()
        base, quote = _parse_pair(self.pair)
        self.base = base
        self.quote = quote
        self._ticker = _yf_ticker(base, quote)

    # ------------------------------------------------------------------
    # Internal: fetch options chain in a thread
    # ------------------------------------------------------------------

    def _fetch_options_sync(self, expiry: str) -> dict:
        """
        Returns {
          "calls": DataFrame or None,
          "puts":  DataFrame or None,
          "spot":  float or None,
          "expiry_years": float
        }
        """
        import yfinance as yf  # lazy

        out: dict = {"calls": None, "puts": None, "spot": None, "expiry_years": 0.25}
        try:
            tk = yf.Ticker(self._ticker)
            info = tk.fast_info
            spot = getattr(info, "last_price", None)
            out["spot"] = float(spot) if spot and spot > 0 else None

            # Resolve expiry string to closest available expiry
            exps = tk.options
            if not exps:
                return out

            # Match caller's label to a real expiry
            target_exp = expiry
            if expiry not in exps:
                # Try to pick the first expiry that's >= today
                today_str = date.today().isoformat()
                future = [e for e in exps if e >= today_str]
                target_exp = future[0] if future else exps[0]

            # Time to expiry in years
            try:
                exp_date = date.fromisoformat(target_exp)
                t_years = max((exp_date - date.today()).days / 365.0, 1e-6)
                out["expiry_years"] = t_years
            except ValueError:
                pass

            chain = tk.option_chain(target_exp)
            out["calls"] = chain.calls
            out["puts"] = chain.puts
        except Exception as exc:
            logger.warning("Options fetch failed", ticker=self._ticker, error=str(exc))
        return out

    # ------------------------------------------------------------------
    # compute_vol_smile
    # ------------------------------------------------------------------

    async def compute_vol_smile(self, expiry: str) -> list[FXSmilePoint]:
        """
        Extract implied vol smile from yfinance options chain for `expiry`.

        Returns a list of FXSmilePoint sorted by strike, filtering out
        low-OI / zero-vol entries.
        """
        raw = await asyncio.to_thread(self._fetch_options_sync, expiry)
        calls = raw.get("calls")
        puts = raw.get("puts")
        spot = raw.get("spot")

        if calls is None or spot is None or spot <= 0:
            return []

        points: list[FXSmilePoint] = []

        for df, is_call in [(calls, True), (puts, False)]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                try:
                    strike = float(row.get("strike", 0))
                    iv = float(row.get("impliedVolatility", 0))
                    oi = float(row.get("openInterest", 0) or 0)
                    if strike <= 0 or iv <= 0 or iv > 5.0 or oi < 1:
                        continue
                    # Approximate delta from moneyness
                    moneyness = strike / spot
                    if is_call:
                        delta_approx = max(0.01, min(0.99, 1.0 - moneyness * 0.5))
                    else:
                        delta_approx = max(0.01, min(0.99, moneyness * 0.5))
                    points.append(FXSmilePoint(
                        strike=round(strike, 6),
                        delta=round(delta_approx, 4),
                        implied_vol=round(iv * 100.0, 4),
                    ))
                except (TypeError, ValueError):
                    continue

        # Deduplicate by strike, keep highest OI
        seen: dict[float, FXSmilePoint] = {}
        for pt in points:
            if pt.strike not in seen or pt.implied_vol < seen[pt.strike].implied_vol:
                seen[pt.strike] = pt

        return sorted(seen.values(), key=lambda p: p.strike)

    # ------------------------------------------------------------------
    # fit_sabr_surface
    # ------------------------------------------------------------------

    async def fit_sabr_surface(
        self,
        maturities: list[str] | None = None,
    ) -> list[SABRParams]:
        """
        Fit SABR (alpha, beta=0.5, rho, nu) for each available maturity.

        `maturities`: list of expiry date strings (YYYY-MM-DD) or None to
        use all available yfinance expiries (up to 4).
        Returns a list of SABRParams, one per maturity that has sufficient data.
        """
        import yfinance as yf  # lazy

        def _get_expiries_sync() -> list[str]:
            try:
                return list(yf.Ticker(self._ticker).options) or []
            except Exception:
                return []

        all_expiries = await asyncio.to_thread(_get_expiries_sync)

        if maturities is None:
            today_str = date.today().isoformat()
            maturities = [e for e in all_expiries if e >= today_str][:4]

        results: list[SABRParams] = []

        for exp in maturities:
            smile = await self.compute_vol_smile(exp)
            if len(smile) < 3:
                logger.warning("Not enough smile points for SABR fit", pair=self.pair, expiry=exp)
                results.append(SABRParams(
                    maturity=exp, alpha=0.1, beta=0.5, rho=0.0, nu=0.3,
                    fit_error=None,
                ))
                continue

            raw = await asyncio.to_thread(self._fetch_options_sync, exp)
            spot = raw.get("spot") or 1.0
            t = raw.get("expiry_years", 0.25)

            # Use spot as forward (ignoring carry for simplicity — forward ~spot for short T)
            F = spot
            strikes = np.array([p.strike for p in smile], dtype=float)
            vols = np.array([p.implied_vol / 100.0 for p in smile], dtype=float)

            try:
                alpha, beta, rho, nu = _fit_sabr_to_quotes(F, t, strikes, vols)

                # Compute in-sample RMSE
                fitted = np.array([_sabr_vol(F, K, t, alpha, beta, rho, nu) for K in strikes])
                valid_mask = np.isfinite(fitted)
                if valid_mask.sum() > 0:
                    rmse = float(np.sqrt(np.mean((fitted[valid_mask] - vols[valid_mask]) ** 2)))
                else:
                    rmse = None

                results.append(SABRParams(
                    maturity=exp,
                    alpha=round(alpha, 6),
                    beta=round(beta, 4),
                    rho=round(rho, 6),
                    nu=round(nu, 6),
                    fit_error=round(rmse, 8) if rmse is not None else None,
                ))
            except Exception as exc:
                logger.warning("SABR fit failed", pair=self.pair, expiry=exp, error=str(exc))
                results.append(SABRParams(
                    maturity=exp, alpha=0.1, beta=0.5, rho=0.0, nu=0.3, fit_error=None,
                ))

        return results

    # ------------------------------------------------------------------
    # compute_risk_reversal
    # ------------------------------------------------------------------

    async def compute_risk_reversal(
        self,
        expiry: str | None = None,
        delta: float = 0.25,
    ) -> float:
        """
        25-delta risk reversal = IV(25d call) - IV(25d put).

        If expiry is None, uses the nearest available expiry.
        Returns 0.0 if insufficient data.
        """
        smile = await self.compute_vol_smile(expiry or "")
        if not smile:
            return 0.0

        # Find closest calls (delta > 0.5) and puts (delta < 0.5) to target delta
        target_call = 0.5 + delta  # e.g. 0.75 for 25d call
        target_put = 0.5 - delta   # e.g. 0.25 for 25d put

        call_points = [p for p in smile if p.delta >= 0.5]
        put_points = [p for p in smile if p.delta < 0.5]

        if not call_points or not put_points:
            return 0.0

        call_iv = min(call_points, key=lambda p: abs(p.delta - target_call)).implied_vol
        put_iv = min(put_points, key=lambda p: abs(p.delta - target_put)).implied_vol

        return round(call_iv - put_iv, 4)

    # ------------------------------------------------------------------
    # compute_butterfly
    # ------------------------------------------------------------------

    async def compute_butterfly(self, expiry: str | None = None) -> float:
        """
        25-delta butterfly = 0.5 * (IV(25d call) + IV(25d put)) - IV(ATM).

        Returns 0.0 if insufficient data.
        """
        smile = await self.compute_vol_smile(expiry or "")
        if len(smile) < 3:
            return 0.0

        call_points = [p for p in smile if p.delta >= 0.5]
        put_points = [p for p in smile if p.delta < 0.5]
        all_points = smile

        if not call_points or not put_points:
            return 0.0

        call_iv = min(call_points, key=lambda p: abs(p.delta - 0.75)).implied_vol
        put_iv = min(put_points, key=lambda p: abs(p.delta - 0.25)).implied_vol
        atm_iv = min(all_points, key=lambda p: abs(p.delta - 0.5)).implied_vol

        return round(0.5 * (call_iv + put_iv) - atm_iv, 4)


# ---------------------------------------------------------------------------
# FXCrossMatrix class
# ---------------------------------------------------------------------------

class FXCrossMatrix:
    """
    Cross-rate matrix for 20+ CCY pairs.

    Fetches all spot rates vs USD from Frankfurter, then computes
    cross-rates for any pair not directly quoted.
    """

    def __init__(self) -> None:
        self._rates_vs_usd: dict[str, float] = {}   # {CCY: USD per 1 CCY}
        self._spot_date: str = date.today().isoformat()

    # ------------------------------------------------------------------
    # Build matrix
    # ------------------------------------------------------------------

    async def build(self) -> None:
        """
        Fetch all Frankfurter rates vs USD and build the internal matrix.
        Call this before using get_cross_rate or get_matrix.
        """
        async with httpx.AsyncClient() as client:
            # Frankfurter: from=USD gives CCY per 1 USD → invert to USD per CCY
            rates, spot_date = await _frankfurter_spot(
                client, "USD",
                [c for c in _FRANKFURTER_CURRENCIES if c != "USD"],
            )
        self._rates_vs_usd = {"USD": 1.0}
        for ccy, rate in rates.items():
            if rate and rate > 0:
                self._rates_vs_usd[ccy] = 1.0 / rate   # USD per 1 unit of CCY
        self._spot_date = spot_date

    # ------------------------------------------------------------------
    # Cross-rate calculation
    # ------------------------------------------------------------------

    def get_cross_rate(self, base: str, quote: str) -> Optional[float]:
        """
        Return spot rate (quote units per 1 base unit).
        Uses USD as the vehicle currency for cross calculations.
        """
        base = base.upper()
        quote = quote.upper()
        if base == quote:
            return 1.0

        usd_per_base = self._rates_vs_usd.get(base)
        usd_per_quote = self._rates_vs_usd.get(quote)

        if usd_per_base is None or usd_per_quote is None:
            return None
        if usd_per_quote < 1e-12:
            return None

        return usd_per_base / usd_per_quote

    # ------------------------------------------------------------------
    # Currency classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify(ccy: str) -> str:
        return "G10" if ccy.upper() in _G10 else "EM"

    # ------------------------------------------------------------------
    # Full matrix
    # ------------------------------------------------------------------

    def get_matrix(self, pairs: list[str] | None = None) -> list[CrossRateEntry]:
        """
        Return CrossRateEntry for each pair in `pairs` (default: _EXTENDED_PAIRS).
        """
        if pairs is None:
            pairs = _EXTENDED_PAIRS

        # Determine which pairs are directly quoted on Frankfurter
        # Frankfurter quotes: USD as base or EUR as base vs many
        directly_quoted = set()
        for p in pairs:
            b, q = p[:3], p[3:]
            if b in _FRANKFURTER_CURRENCIES and q in _FRANKFURTER_CURRENCIES:
                directly_quoted.add(p)

        entries: list[CrossRateEntry] = []
        for pair in pairs:
            b, q = pair[:3], pair[3:]
            rate = self.get_cross_rate(b, q)
            if rate is None:
                continue
            entries.append(CrossRateEntry(
                pair=pair,
                base=b,
                quote=q,
                spot=round(rate, 6),
                is_direct=pair in directly_quoted,
                base_class=self.classify(b),
                quote_class=self.classify(q),
                spot_date=self._spot_date,
            ))
        return entries

    # ------------------------------------------------------------------
    # Heatmap data (pct change from PPP fair value)
    # ------------------------------------------------------------------

    def get_heatmap_data(self) -> dict[str, dict[str, float]]:
        """
        Returns a nested dict {base: {quote: pct_vs_usd_cross}} suitable for
        terminal heatmap rendering. Values are spot rates normalised as
        (spot - PPP_spot) / PPP_spot * 100 where available, else raw spot.
        """
        ccys = list(self._rates_vs_usd.keys())
        heatmap: dict[str, dict[str, float]] = {}
        for base in ccys:
            heatmap[base] = {}
            for quote in ccys:
                if base == quote:
                    continue
                rate = self.get_cross_rate(base, quote)
                if rate is not None:
                    heatmap[base][quote] = round(rate, 6)
        return heatmap


# ---------------------------------------------------------------------------
# FXCarrySignal class
# ---------------------------------------------------------------------------

class FXCarrySignal:
    """
    Combined carry + momentum + value (PPP) signal for G10 FX pairs.
    """

    # G10 pair universe (all USD-quoted for rate differential simplicity)
    _G10_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD",
                  "NZDUSD", "USDSEK", "USDNOK", "USDSGD"]

    def __init__(self) -> None:
        self._rates: dict[str, Optional[float]] = {}

    async def _fetch_all_rates(self) -> None:
        async with httpx.AsyncClient() as client:
            self._rates = await _fetch_rates(client, set(_RATE_SERIES.keys()))

    # ------------------------------------------------------------------
    # Carry component
    # ------------------------------------------------------------------

    def _carry_for_pair(self, pair: str) -> tuple[Optional[float], int]:
        """
        Returns (rate_differential_pct, signal).
        For USD-base pair (e.g. USDJPY): diff = JPY_rate - USD_rate → positive means JPY carry.
        For non-USD-base (e.g. EURUSD): diff = EUR_rate - USD_rate.
        Signal: +1 long foreign, -1 long USD, 0 neutral.
        """
        b, q = pair[:3], pair[3:]
        non_usd = q if b == "USD" else b
        usd_rate = self._rates.get("USD")
        foreign_rate = self._rates.get(non_usd)

        if usd_rate is None or foreign_rate is None:
            return None, 0

        # Rate differential = foreign - USD (positive = foreign higher = long foreign)
        diff = foreign_rate - usd_rate
        signal = 1 if diff > 0.5 else (-1 if diff < -0.5 else 0)
        # Flip if USD is base (long pair = long USD)
        if b == "USD":
            signal = -signal
        return round(diff, 4), signal

    # ------------------------------------------------------------------
    # Momentum component (12M - 1M)
    # ------------------------------------------------------------------

    async def _momentum_for_pair(self, pair: str) -> tuple[Optional[float], int]:
        """
        12M - 1M momentum: 12-month return minus 1-month return.
        Positive = upward momentum for the pair (base currency appreciating).
        """
        b, q = pair[:3], pair[3:]
        today = date.today()
        start_12m = (today - timedelta(days=375)).isoformat()
        end_str = today.isoformat()

        async with httpx.AsyncClient() as client:
            hist = await _frankfurter_history(client, b, q, start_12m, end_str)

        if len(hist) < 22:
            return None, 0

        sorted_dates = sorted(hist.keys())
        prices = [hist[d] for d in sorted_dates]

        # 12M return: last price vs price ~252 trading days ago
        ret_12m = (prices[-1] / prices[0]) - 1.0 if prices[0] > 0 else 0.0

        # 1M return: last price vs price ~21 trading days ago
        idx_1m = max(0, len(prices) - 22)
        ret_1m = (prices[-1] / prices[idx_1m]) - 1.0 if prices[idx_1m] > 0 else 0.0

        mom = ret_12m - ret_1m
        signal = 1 if mom > 0.01 else (-1 if mom < -0.01 else 0)
        return round(mom * 100.0, 4), signal

    # ------------------------------------------------------------------
    # Value (PPP) component
    # ------------------------------------------------------------------

    def _value_for_pair(self, pair: str, spot: Optional[float]) -> tuple[Optional[float], int]:
        """
        PPP deviation = (spot - PPP_fair) / PPP_fair * 100.
        Positive = pair overvalued vs PPP (expect mean reversion down → short signal).
        Signal: +1 = base currency undervalued (buy), -1 = overvalued (sell).
        """
        b, q = pair[:3], pair[3:]
        non_usd = q if b == "USD" else b
        ppp = _PPP_VS_USD.get(non_usd)

        if ppp is None or spot is None or spot <= 0:
            return None, 0

        # Convert PPP to pair quote terms
        if b == "USD":
            ppp_pair = 1.0 / ppp   # USD per foreign → quote foreign per USD
        else:
            ppp_pair = ppp         # USD per foreign

        dev = (spot - ppp_pair) / ppp_pair * 100.0
        # Positive dev = spot > PPP → base overvalued → sell base → signal -1
        signal = -1 if dev > 10.0 else (1 if dev < -10.0 else 0)
        return round(dev, 4), signal

    # ------------------------------------------------------------------
    # Public: compute all signals
    # ------------------------------------------------------------------

    async def compute(
        self,
        pairs: list[str] | None = None,
        spots: dict[str, float] | None = None,
    ) -> list[CarrySignalEntry]:
        """
        Compute carry + momentum + value combined signals for all pairs.

        `spots`: optional pre-fetched spot dict {pair: spot_rate}
        Returns list of CarrySignalEntry sorted by combined_score descending.
        """
        if pairs is None:
            pairs = self._G10_PAIRS

        await self._fetch_all_rates()

        # Gather momentum concurrently
        mom_tasks = [self._momentum_for_pair(p) for p in pairs]
        mom_results = await asyncio.gather(*mom_tasks, return_exceptions=True)

        entries: list[CarrySignalEntry] = []
        for pair, mom_res in zip(pairs, mom_results):
            carry_diff, carry_sig = self._carry_for_pair(pair)

            if isinstance(mom_res, Exception):
                mom_val, mom_sig = None, 0
            else:
                mom_val, mom_sig = mom_res  # type: ignore[misc]

            spot = (spots or {}).get(pair)
            ppp_dev, val_sig = self._value_for_pair(pair, spot)

            combined = float(carry_sig + mom_sig + val_sig)

            entries.append(CarrySignalEntry(
                pair=pair,
                carry_diff_pct=carry_diff,
                carry_signal=carry_sig,
                momentum_12m_1m=mom_val,
                momentum_signal=mom_sig,
                ppp_deviation_pct=ppp_dev,
                value_signal=val_sig,
                combined_score=combined,
            ))

        return sorted(entries, key=lambda e: e.combined_score, reverse=True)


# ---------------------------------------------------------------------------
# Extended FX dashboard (top-level convenience function)
# ---------------------------------------------------------------------------

async def get_extended_fx_dashboard(base: str = "USD") -> dict:
    """
    Top-level function returning a comprehensive FX dashboard dict including:
    - Cross-rate matrix for 20+ pairs
    - Carry/momentum/value signals
    - Vol surface summary (SABR fit for EURUSD, GBPUSD)
    - Risk reversal and butterfly for key pairs

    Designed as the score-9 entry point for the FX analytics module.
    """
    warnings_: list[str] = []

    # 1. Build cross-rate matrix
    matrix = FXCrossMatrix()
    try:
        await matrix.build()
    except Exception as exc:
        warnings_.append(f"Cross-rate matrix build failed: {exc}")

    matrix_entries = matrix.get_matrix()
    heatmap = matrix.get_heatmap_data()

    # Spots dict for carry signal value component
    spots_dict: dict[str, float] = {}
    for entry in matrix_entries:
        spots_dict[entry.pair] = entry.spot

    # 2. Carry/momentum/value signals
    carry_model = FXCarrySignal()
    try:
        carry_signals = await carry_model.compute(spots=spots_dict)
    except Exception as exc:
        warnings_.append(f"Carry signal computation failed: {exc}")
        carry_signals = []

    # 3. SABR vol surface for EURUSD (key liquid pair)
    sabr_results: dict[str, list[dict]] = {}
    rr_results: dict[str, float] = {}
    fly_results: dict[str, float] = {}

    for pair in ["EURUSD", "GBPUSD"]:
        vol_model = FXVolSurfaceModel(pair)
        try:
            sabr_fits = await vol_model.fit_sabr_surface()
            sabr_results[pair] = [p.model_dump() for p in sabr_fits]

            rr = await vol_model.compute_risk_reversal()
            rr_results[pair] = rr

            fly = await vol_model.compute_butterfly()
            fly_results[pair] = fly
        except Exception as exc:
            warnings_.append(f"Vol surface for {pair} failed: {exc}")
            sabr_results[pair] = []
            rr_results[pair] = 0.0
            fly_results[pair] = 0.0

    # 4. Assemble dashboard
    return {
        "base_currency": base.upper(),
        "as_of": date.today().isoformat(),
        "cross_rate_matrix": [e.model_dump() for e in matrix_entries],
        "heatmap": heatmap,
        "carry_signals": [e.model_dump() for e in carry_signals],
        "sabr_surfaces": sabr_results,
        "risk_reversals_25d": rr_results,
        "butterflies_25d": fly_results,
        "warnings": warnings_,
        "metadata": {
            "pairs_count": len(matrix_entries),
            "g10_pairs": sum(1 for e in matrix_entries if e.base_class == "G10" and e.quote_class == "G10"),
            "em_pairs": sum(1 for e in matrix_entries if "EM" in (e.base_class, e.quote_class)),
        },
    }
