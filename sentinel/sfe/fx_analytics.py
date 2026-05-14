"""
FX analytics — spot rates, forward curves, implied vol proxy, carry/momentum signals.

Data sources (all free):
  Spot rates:    Frankfurter API (ECB official fixing, https://api.frankfurter.app)
  Interest rates: FRED CSV (FEDFUNDS, ECBDFR, etc.)
  Vol/OHLCV:     yfinance — lazy-imported inside asyncio.to_thread
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = 20.0
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRANKFURTER_BASE = "https://api.frankfurter.app"

# Default pairs (base3 + quote3)
_DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "USDCNY", "USDMXN"]

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
    hist_vals = [hist_raw[d] for d in sorted_dates]

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
            mean_chg = float(np.mean(changes_1w))
            # Positive change_1w means rate went up vs USD → USD weaker for USD-base, stronger for non-USD-base
            # Simplify: if most pairs show USD appreciating
            # For EURUSD: positive chg = EUR up / USD down → USD weaker
            # For USDJPY: positive chg = JPY up (USD up) → USD stronger
            # Too complex to unify without explicit direction — use momentum signals instead
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
