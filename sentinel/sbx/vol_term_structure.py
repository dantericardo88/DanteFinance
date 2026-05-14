"""Options volatility term structure and smile analytics — Dimension 15 enhancement."""
from __future__ import annotations

import asyncio
from datetime import datetime, date
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────

class ExpirationSlice(BaseModel):
    model_config = ConfigDict(frozen=True)
    expiration: str          # ISO date string
    dte: int                 # calendar days to expiration
    atm_iv: Optional[float]  # ATM implied vol: avg(nearest call IV, nearest put IV)
    iv_skew_25d: Optional[float]  # IV(0.75×spot) − atm_iv  — downside proxy
    iv_skew_10d: Optional[float]  # IV(0.90×spot) − atm_iv
    put_call_skew: Optional[float]  # avg OTM put IV − avg OTM call IV
    total_calls_oi: int
    total_puts_oi: int
    vol_of_vol: Optional[float]  # std of all valid mid-chain IVs in slice

class ForwardVol(BaseModel):
    model_config = ConfigDict(frozen=True)
    from_tenor: str    # near expiration ISO date
    to_tenor: str      # far expiration ISO date
    forward_vol: float  # annualised forward vol

class SVIParams(BaseModel):
    """Raw Heston SVI smile parameters: w(k) = a + b(ρ(k−m) + √((k−m)²+σ²))."""
    model_config = ConfigDict(frozen=True)
    a: float; b: float; rho: float; m: float; sigma: float
    expiration: str

class VolTermStructure(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    spot: float
    as_of: str
    slices: list[ExpirationSlice]      # sorted by DTE ascending
    forward_vols: list[ForwardVol]
    svi_fits: list[SVIParams]          # SVI smile fits per expiration (where fittable)
    term_slope: Optional[float]        # slope of ATM IV vs DTE; positive = contango
    contango: Optional[bool]
    atm_term_structure: list[tuple[int, float]]  # [(dte, atm_iv), ...]
    skew_summary: dict                 # avg_25d_skew, avg_10d_skew, skew_regime
    # Enhanced analytics
    front_back_ratio: Optional[float]  # front ATM IV / back ATM IV (backwardation indicator)
    iv_rank: Optional[float]           # IVR vs 52-week range [0, 100] — requires hist data
    iv_percentile_exact: Optional[float]  # % of hist days below current IV [0, 100]
    skew_zscore: Optional[float]       # (current_25d_skew − 30d_mean) / 30d_std
    warnings: list[str]

class VolSurfaceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    spot: float
    atm_iv_front: Optional[float]
    atm_iv_back: Optional[float]
    term_slope: Optional[float]
    front_back_ratio: Optional[float]  # front/back IV ratio
    skew_regime: str
    iv_percentile: Optional[float]     # legacy bucket-based percentile
    iv_rank: Optional[float]           # IVR [0, 100] when historical data available
    skew_zscore: Optional[float]       # skew z-score vs 30d baseline
    warnings: list[str]

class VolSurfaceScreen(BaseModel):
    model_config = ConfigDict(frozen=True)
    tickers_screened: list[str]
    results: dict[str, VolSurfaceSummary]
    elevated_skew: list[str]           # avg_25d_skew > 0.05
    inverted_term_structure: list[str] # contango == False
    as_of: str
    warnings: list[str]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_iv(iv) -> Optional[float]:
    """Return float IV if valid (0 < iv ≤ 5, not NaN), else None."""
    try:
        v = float(iv)
    except (TypeError, ValueError):
        return None
    return v if (0 < v <= 5.0 and not np.isnan(v)) else None

def _atm_iv(calls_df, puts_df, spot: float) -> Optional[float]:
    """Average of nearest-strike call IV and put IV."""
    ivs: list[float] = []
    for df in (calls_df, puts_df):
        if df is None or df.empty:
            continue
        try:
            idx = int(np.argmin(np.abs(df["strike"].values.astype(float) - spot)))
            iv = _valid_iv(df.iloc[idx]["impliedVolatility"])
            if iv is not None:
                ivs.append(iv)
        except Exception as exc:
            logger.debug("_atm_iv: skip", error=str(exc))
    return float(np.mean(ivs)) if ivs else None

def _skew_iv(df, spot: float, moneyness: float) -> Optional[float]:
    """IV of the option closest to spot × moneyness."""
    if df is None or df.empty:
        return None
    try:
        idx = int(np.argmin(np.abs(df["strike"].values.astype(float) - spot * moneyness)))
        return _valid_iv(df.iloc[idx]["impliedVolatility"])
    except Exception as exc:
        logger.debug("_skew_iv: error", moneyness=moneyness, error=str(exc))
        return None

def _analyze_expiration(calls_df, puts_df, spot: float, expiration: str) -> Optional[ExpirationSlice]:
    """Build ExpirationSlice for one expiration; returns None if expired or unparseable."""
    try:
        dte = (date.fromisoformat(expiration) - date.today()).days
    except ValueError:
        return None
    if dte <= 0:
        return None

    atm = _atm_iv(calls_df, puts_df, spot)
    iv_25 = _skew_iv(puts_df, spot, 0.75)
    iv_10 = _skew_iv(puts_df, spot, 0.90)
    iv_skew_25d = round(iv_25 - atm, 5) if (atm and iv_25 is not None) else None
    iv_skew_10d = round(iv_10 - atm, 5) if (atm and iv_10 is not None) else None

    put_call_skew: Optional[float] = None
    try:
        p_ivs = [_valid_iv(v) for v in puts_df[puts_df["strike"] < spot * 0.98]["impliedVolatility"]] if (puts_df is not None and not puts_df.empty) else []
        c_ivs = [_valid_iv(v) for v in calls_df[calls_df["strike"] > spot * 1.02]["impliedVolatility"]] if (calls_df is not None and not calls_df.empty) else []
        p_ivs = [v for v in p_ivs if v is not None]
        c_ivs = [v for v in c_ivs if v is not None]
        if p_ivs and c_ivs:
            put_call_skew = round(float(np.mean(p_ivs)) - float(np.mean(c_ivs)), 5)
    except Exception as exc:
        logger.debug("_analyze_expiration: put_call_skew", exp=expiration, error=str(exc))

    vol_of_vol: Optional[float] = None
    try:
        all_ivs: list[float] = []
        for df in (calls_df, puts_df):
            if df is None or df.empty:
                continue
            all_ivs += [v for v in (_valid_iv(x) for x in df["impliedVolatility"]) if v is not None]
        if len(all_ivs) >= 2:
            vol_of_vol = round(float(np.std(all_ivs, ddof=1)), 5)
    except Exception as exc:
        logger.debug("_analyze_expiration: vol_of_vol", exp=expiration, error=str(exc))

    total_calls_oi = 0
    total_puts_oi = 0
    try:
        if calls_df is not None and not calls_df.empty and "openInterest" in calls_df.columns:
            total_calls_oi = int(calls_df["openInterest"].fillna(0).sum())
        if puts_df is not None and not puts_df.empty and "openInterest" in puts_df.columns:
            total_puts_oi = int(puts_df["openInterest"].fillna(0).sum())
    except Exception as exc:
        logger.debug("_analyze_expiration: OI", exp=expiration, error=str(exc))

    return ExpirationSlice(
        expiration=expiration, dte=dte,
        atm_iv=round(atm, 5) if atm is not None else None,
        iv_skew_25d=iv_skew_25d, iv_skew_10d=iv_skew_10d,
        put_call_skew=put_call_skew,
        total_calls_oi=total_calls_oi, total_puts_oi=total_puts_oi,
        vol_of_vol=vol_of_vol,
    )

def _forward_vol(t1_iv: float, t1_dte: int, t2_iv: float, t2_dte: int) -> Optional[float]:
    """Forward variance: fwd_var = (σ₂²T₂ − σ₁²T₁)/(T₂−T₁).  Returns √fwd_var or None."""
    if t2_dte <= t1_dte:
        return None
    fwd_var = (t2_iv ** 2 * t2_dte - t1_iv ** 2 * t1_dte) / (t2_dte - t1_dte)
    return float(np.sqrt(max(0.0, fwd_var))) if fwd_var > 0 else None

def _term_slope(slices: list[ExpirationSlice]) -> Optional[float]:
    """Linear regression of ATM IV on DTE; requires ≥3 valid slices."""
    pts = [(s.dte, s.atm_iv) for s in slices if s.atm_iv is not None]
    if len(pts) < 3:
        return None
    try:
        slope, _ = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        return float(slope)
    except Exception:
        return None

def _skew_regime(slices: list[ExpirationSlice]) -> str:
    """Classify skew regime from average 25-delta downside skew."""
    vals = [s.iv_skew_25d for s in slices if s.iv_skew_25d is not None]
    if not vals:
        return "unknown"
    avg = float(np.mean(vals))
    if avg > 0.05:
        return "steep downside skew"
    if avg > 0.02:
        return "moderate skew"
    if avg < 0:
        return "inverted skew"
    return "flat"

def _iv_percentile(atm_iv_front: float) -> Optional[float]:
    """Rough market-regime IV percentile bucket from front ATM IV level."""
    if atm_iv_front < 0.15: return 10.0
    if atm_iv_front < 0.20: return 25.0
    if atm_iv_front < 0.25: return 40.0
    if atm_iv_front < 0.30: return 55.0
    if atm_iv_front < 0.40: return 70.0
    return 85.0


def _compute_iv_rank(
    atm_iv_front: float,
    historical_ivs: list[float],
) -> Optional[float]:
    """Compute IV rank (IVR) = (current − 52w_low) / (52w_high − 52w_low) × 100.

    historical_ivs should be a list of daily ATM IV observations over ~252 trading
    days.  Returns a value in [0, 100] or None if data is insufficient.
    """
    if not historical_ivs or len(historical_ivs) < 5:
        return None
    lo = min(historical_ivs)
    hi = max(historical_ivs)
    if hi <= lo:
        return None
    rank = (atm_iv_front - lo) / (hi - lo) * 100.0
    return round(float(np.clip(rank, 0.0, 100.0)), 2)


def _compute_iv_percentile_exact(
    atm_iv_front: float,
    historical_ivs: list[float],
) -> Optional[float]:
    """Compute IV percentile = fraction of historical days with IV below current × 100.

    Complements IVR — measures *how often* IV was below today's level.
    Returns value in [0, 100].
    """
    if not historical_ivs or len(historical_ivs) < 5:
        return None
    below = sum(1 for v in historical_ivs if v < atm_iv_front)
    return round(below / len(historical_ivs) * 100.0, 2)


def _compute_term_slope_front_back(
    slices: list[ExpirationSlice],
) -> Optional[float]:
    """Term structure slope = front ATM IV / back ATM IV.

    Ratio > 1 → backwardation (inverted); ratio < 1 → contango.
    Requires at least 2 slices with valid ATM IV.
    """
    valid = [(s.dte, s.atm_iv) for s in slices if s.atm_iv is not None]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda x: x[0])
    front_iv = valid[0][1]
    back_iv = valid[-1][1]
    if back_iv is None or back_iv <= 0:
        return None
    return round(float(front_iv) / float(back_iv), 6)


def _compute_skew_zscore(
    current_skew: Optional[float],
    historical_skews: list[float],
) -> Optional[float]:
    """Skew z-score = (current_skew − 30d_mean) / 30d_std.

    historical_skews should contain daily avg_25d skew observations for the
    trailing ~30 calendar days.  Returns None if data is insufficient (< 5 obs).
    """
    if current_skew is None or not historical_skews or len(historical_skews) < 5:
        return None
    arr = np.array(historical_skews, dtype=float)
    mu = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return None
    return round((current_skew - mu) / std, 4)

def _fit_svi(log_moneyness: np.ndarray, total_var: np.ndarray, expiration: str) -> Optional[SVIParams]:
    """Least-squares fit of raw SVI to (log-moneyness, total variance) data."""
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:
        return None

    def svi_w(p, k):
        a, b, rho, m, sigma = p
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def residuals(p, k, w):
        a, b, rho, m, sigma = p
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return np.full_like(w, 1e6)
        return svi_w(p, k) - w

    atm_var = float(np.median(total_var))
    try:
        res = least_squares(
            residuals, [atm_var * 0.8, 0.1, -0.3, 0.0, 0.1],
            args=(log_moneyness, total_var),
            bounds=([0.0, 1e-4, -0.999, -2.0, 1e-4], [10.0, 5.0, 0.999, 2.0, 5.0]),
            max_nfev=2000, ftol=1e-9, xtol=1e-9,
        )
    except Exception as exc:
        logger.debug("_fit_svi: optimise failed", exp=expiration, error=str(exc))
        return None

    a, b, rho, m, sigma = res.x
    if b * (1 + abs(rho)) > 4:  # butterfly arbitrage check
        return None
    return SVIParams(
        a=round(float(a), 6), b=round(float(b), 6), rho=round(float(rho), 6),
        m=round(float(m), 6), sigma=round(float(sigma), 6), expiration=expiration,
    )

def _build_svi_params(calls_df, puts_df, spot: float, expiration: str, dte: int) -> Optional[SVIParams]:
    """Collect smile points and call SVI fitter for one expiration."""
    if dte <= 0 or spot <= 0:
        return None
    T = dte / 365.0
    rows: list[tuple[float, float]] = []
    for df in (calls_df, puts_df):
        if df is None or df.empty:
            continue
        try:
            for _, row in df.iterrows():
                iv = _valid_iv(row["impliedVolatility"])
                k = row.get("strike", 0)
                if iv is None or float(k) <= 0:
                    continue
                rows.append((float(np.log(float(k) / spot)), iv ** 2 * T))
        except Exception:
            continue
    if len(rows) < 5:
        return None
    return _fit_svi(
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        expiration,
    )

# ── Data Fetching ─────────────────────────────────────────────────────────────

async def _fetch_options_chain(ticker: str) -> tuple[float, list[str], dict]:
    """Fetch yfinance options chain for up to 6 expirations.
    Returns (spot, expirations[:6], {exp: (calls_df, puts_df)}).
    All yfinance I/O is wrapped in asyncio.to_thread (sync library).
    """
    def _ticker_obj():
        import yfinance as yf
        return yf.Ticker(ticker)

    t = await asyncio.to_thread(_ticker_obj)

    def _spot():
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        if not price:
            slow = t.info
            price = slow.get("regularMarketPrice") or slow.get("previousClose")
        return float(price) if price else None

    spot = await asyncio.to_thread(_spot)
    if not spot or spot <= 0:
        raise ValueError(f"Cannot determine spot price for {ticker}")

    expirations = await asyncio.to_thread(lambda: list(t.options))
    if not expirations:
        return spot, [], {}

    target = expirations[:6]
    chains: dict = {}

    async def _fetch_one(exp: str):
        try:
            chain = await asyncio.to_thread(lambda: t.option_chain(exp))
            chains[exp] = (chain.calls, chain.puts)
        except Exception as exc:
            logger.warning("_fetch_options_chain: chain failed", ticker=ticker, exp=exp, error=str(exc))
            chains[exp] = (None, None)

    await asyncio.gather(*[_fetch_one(e) for e in target])
    return spot, target, chains

# ── Public API ────────────────────────────────────────────────────────────────

async def get_vol_term_structure(ticker: str) -> VolTermStructure:
    """Compute full volatility term structure for *ticker* using yfinance options data.

    Fetches up to 6 expirations, analyses each slice (ATM IV, skew, OI, vol-of-vol),
    computes adjacent forward vols, term slope via linear regression, and skew summary.
    """
    warnings: list[str] = []
    as_of = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    logger.info("get_vol_term_structure start", ticker=ticker)

    try:
        spot, expirations, chains = await _fetch_options_chain(ticker)
    except Exception as exc:
        warnings.append(f"Failed to fetch options chain: {exc}")
        logger.error("get_vol_term_structure: fetch failed", ticker=ticker, error=str(exc))
        return VolTermStructure(
            ticker=ticker, spot=0.0, as_of=as_of, slices=[], forward_vols=[],
            term_slope=None, contango=None, atm_term_structure=[],
            skew_summary={"avg_25d_skew": None, "avg_10d_skew": None, "skew_regime": "unknown"},
            warnings=warnings,
        )

    if not expirations:
        warnings.append(f"No options data available for {ticker}")

    slices: list[ExpirationSlice] = []
    for exp in expirations:
        calls_df, puts_df = chains.get(exp, (None, None))
        if calls_df is None and puts_df is None:
            warnings.append(f"No chain data for expiration {exp}")
            continue
        sl = _analyze_expiration(calls_df, puts_df, spot, exp)
        if sl is None:
            warnings.append(f"Expiration {exp} skipped (expired or parse error)")
            continue
        if sl.atm_iv is None:
            warnings.append(f"Illiquid slice at {exp} — no valid ATM IV")
        slices.append(sl)

    slices.sort(key=lambda s: s.dte)

    if len(slices) < 2:
        warnings.append("Insufficient expirations for term structure (need ≥2)")

    forward_vols: list[ForwardVol] = []
    for i in range(len(slices) - 1):
        s1, s2 = slices[i], slices[i + 1]
        if s1.atm_iv is None or s2.atm_iv is None:
            continue
        fv = _forward_vol(s1.atm_iv, s1.dte, s2.atm_iv, s2.dte)
        if fv is not None:
            forward_vols.append(ForwardVol(
                from_tenor=s1.expiration, to_tenor=s2.expiration,
                forward_vol=round(fv, 5),
            ))
        else:
            warnings.append(
                f"Negative forward variance {s1.expiration}→{s2.expiration} — possible calendar arb"
            )

    # SVI smile fits — attempted per slice; silently skipped if scipy absent or data sparse
    svi_fits: list[SVIParams] = []
    for sl in slices:
        calls_df, puts_df = chains.get(sl.expiration, (None, None))
        fit = _build_svi_params(calls_df, puts_df, spot, sl.expiration, sl.dte)
        if fit is not None:
            svi_fits.append(fit)
    if svi_fits:
        logger.info("get_vol_term_structure: SVI fits", ticker=ticker, fitted=len(svi_fits))
    else:
        warnings.append("SVI parameterization unavailable (scipy missing or insufficient smile data)")

    slope = _term_slope(slices)
    contango: Optional[bool] = (slope >= 0) if slope is not None else None
    atm_ts = [(s.dte, s.atm_iv) for s in slices if s.atm_iv is not None]

    vals_25d = [s.iv_skew_25d for s in slices if s.iv_skew_25d is not None]
    vals_10d = [s.iv_skew_10d for s in slices if s.iv_skew_10d is not None]
    skew_summary = {
        "avg_25d_skew": round(float(np.mean(vals_25d)), 5) if vals_25d else None,
        "avg_10d_skew": round(float(np.mean(vals_10d)), 5) if vals_10d else None,
        "skew_regime": _skew_regime(slices),
    }

    # ── Enhanced analytics ────────────────────────────────────────────────────

    # Front/back term structure slope ratio
    front_back_ratio = _compute_term_slope_front_back(slices)

    # IV rank and percentile — require caller to inject historical_ivs;
    # without historical data we fall back to the bucket-based legacy _iv_percentile.
    # These fields are populated to None here and enriched by callers that have
    # access to historical IV time series (e.g. from a DB cache).
    iv_rank: Optional[float] = None
    iv_percentile_exact: Optional[float] = None

    # Skew z-score — similarly requires a 30-day baseline; set to None by default,
    # enriched by callers with access to historical skew series.
    skew_zscore: Optional[float] = None

    # Provide helper stubs so callers can enrich the struct post-hoc:
    # e.g. vts = enrich_iv_rank(vts, historical_atm_ivs)
    # We expose the computation functions publicly for that purpose.

    logger.info(
        "get_vol_term_structure complete",
        ticker=ticker, spot=spot, slices=len(slices),
        forward_vols=len(forward_vols), contango=contango,
        regime=skew_summary["skew_regime"],
        front_back_ratio=front_back_ratio,
    )
    return VolTermStructure(
        ticker=ticker, spot=spot, as_of=as_of, slices=slices,
        forward_vols=forward_vols, svi_fits=svi_fits,
        term_slope=round(slope, 7) if slope is not None else None,
        contango=contango, atm_term_structure=atm_ts,
        skew_summary=skew_summary,
        front_back_ratio=front_back_ratio,
        iv_rank=iv_rank,
        iv_percentile_exact=iv_percentile_exact,
        skew_zscore=skew_zscore,
        warnings=warnings,
    )


def enrich_vol_term_structure(
    vts: VolTermStructure,
    historical_atm_ivs: list[float],
    historical_25d_skews: list[float],
) -> VolTermStructure:
    """Enrich a VolTermStructure with 52-week IV rank, exact IV percentile, and
    skew z-score using externally supplied historical data.

    Args:
        vts: The VolTermStructure to enrich.
        historical_atm_ivs: Daily ATM IV observations for ~252 trading days
            (52-week window). Used for IVR and IV percentile.
        historical_25d_skews: Daily avg_25d downside skew observations for
            ~30 calendar days. Used for skew z-score.

    Returns:
        A new VolTermStructure (frozen model) with the computed fields set.
        All other fields are identical to the input.
    """
    atm_iv_front = vts.atm_term_structure[0][1] if vts.atm_term_structure else None

    iv_rank: Optional[float] = None
    iv_percentile_exact: Optional[float] = None
    if atm_iv_front is not None:
        iv_rank = _compute_iv_rank(atm_iv_front, historical_atm_ivs)
        iv_percentile_exact = _compute_iv_percentile_exact(atm_iv_front, historical_atm_ivs)

    current_25d_skew = vts.skew_summary.get("avg_25d_skew")
    skew_zscore = _compute_skew_zscore(current_25d_skew, historical_25d_skews)

    return VolTermStructure(
        ticker=vts.ticker,
        spot=vts.spot,
        as_of=vts.as_of,
        slices=vts.slices,
        forward_vols=vts.forward_vols,
        svi_fits=vts.svi_fits,
        term_slope=vts.term_slope,
        contango=vts.contango,
        atm_term_structure=vts.atm_term_structure,
        skew_summary=vts.skew_summary,
        front_back_ratio=vts.front_back_ratio,
        iv_rank=iv_rank,
        iv_percentile_exact=iv_percentile_exact,
        skew_zscore=skew_zscore,
        warnings=vts.warnings,
    )


def vol_risk_summary(vts: VolTermStructure) -> dict:
    """Extract a concise risk-oriented view from a VolTermStructure.

    Returns a flat dict suitable for display, alerting, or downstream scoring.
    Fields:
      ticker, spot, as_of, dte_front, dte_back, atm_iv_front, atm_iv_back,
      front_fwd_vol, term_slope, contango, skew_regime,
      avg_25d_skew, avg_10d_skew, avg_put_call_skew,
      avg_vol_of_vol, total_oi_calls, total_oi_puts, svi_fit_count,
      iv_percentile, risk_flag.
    """
    slices = vts.slices
    atm_ts = vts.atm_term_structure

    dte_front = slices[0].dte if slices else None
    dte_back = slices[-1].dte if slices else None
    atm_iv_front = atm_ts[0][1] if atm_ts else None
    atm_iv_back = atm_ts[-1][1] if len(atm_ts) >= 2 else None

    front_fwd_vol: Optional[float] = None
    if vts.forward_vols:
        front_fwd_vol = vts.forward_vols[0].forward_vol

    avg_put_call = None
    pc_vals = [s.put_call_skew for s in slices if s.put_call_skew is not None]
    if pc_vals:
        avg_put_call = round(float(np.mean(pc_vals)), 5)

    avg_vov = None
    vov_vals = [s.vol_of_vol for s in slices if s.vol_of_vol is not None]
    if vov_vals:
        avg_vov = round(float(np.mean(vov_vals)), 5)

    total_oi_calls = sum(s.total_calls_oi for s in slices)
    total_oi_puts = sum(s.total_puts_oi for s in slices)
    iv_pct = _iv_percentile(atm_iv_front) if atm_iv_front is not None else None

    # Risk flag: elevated skew + inverted term structure is the most stress-indicative combo
    avg_25d = vts.skew_summary.get("avg_25d_skew")
    risk_flag = "elevated" if (avg_25d is not None and avg_25d > 0.05 and vts.contango is False) else "normal"

    return {
        "ticker": vts.ticker,
        "spot": vts.spot,
        "as_of": vts.as_of,
        "dte_front": dte_front,
        "dte_back": dte_back,
        "atm_iv_front": atm_iv_front,
        "atm_iv_back": atm_iv_back,
        "front_fwd_vol": front_fwd_vol,
        "term_slope": vts.term_slope,
        "front_back_ratio": vts.front_back_ratio,
        "contango": vts.contango,
        "skew_regime": vts.skew_summary.get("skew_regime"),
        "avg_25d_skew": avg_25d,
        "avg_10d_skew": vts.skew_summary.get("avg_10d_skew"),
        "avg_put_call_skew": avg_put_call,
        "avg_vol_of_vol": avg_vov,
        "total_oi_calls": total_oi_calls,
        "total_oi_puts": total_oi_puts,
        "svi_fit_count": len(vts.svi_fits),
        "iv_percentile": iv_pct,
        "iv_rank": vts.iv_rank,
        "iv_percentile_exact": vts.iv_percentile_exact,
        "skew_zscore": vts.skew_zscore,
        "risk_flag": risk_flag,
    }


async def screen_vol_surface(tickers: list[str]) -> VolSurfaceScreen:
    """Screen a list of tickers for notable vol surface characteristics.

    Runs get_vol_term_structure concurrently via asyncio.gather.
    Flags elevated downside skew (avg_25d > 0.05) and inverted term structure.
    """
    as_of = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    screen_warnings: list[str] = []
    logger.info("screen_vol_surface start", count=len(tickers))

    results_raw: list[VolTermStructure] = await asyncio.gather(
        *[get_vol_term_structure(t) for t in tickers],
        return_exceptions=False,
    )

    summaries: dict[str, VolSurfaceSummary] = {}
    elevated_skew: list[str] = []
    inverted_ts: list[str] = []

    for ticker, vts in zip(tickers, results_raw):
        atm_iv_front = vts.atm_term_structure[0][1] if vts.atm_term_structure else None
        atm_iv_back = vts.atm_term_structure[-1][1] if len(vts.atm_term_structure) >= 2 else None
        iv_pct = _iv_percentile(atm_iv_front) if atm_iv_front is not None else None
        regime = vts.skew_summary.get("skew_regime", "unknown")
        avg_25d = vts.skew_summary.get("avg_25d_skew")

        if avg_25d is not None and avg_25d > 0.05:
            elevated_skew.append(ticker)
        if vts.contango is False:
            inverted_ts.append(ticker)

        ticker_warns = vts.warnings[:]
        if vts.spot == 0.0:
            screen_warnings.append(f"{ticker}: fetch failed")

        summaries[ticker] = VolSurfaceSummary(
            ticker=ticker, spot=vts.spot,
            atm_iv_front=atm_iv_front, atm_iv_back=atm_iv_back,
            term_slope=vts.term_slope,
            front_back_ratio=vts.front_back_ratio,
            skew_regime=regime,
            iv_percentile=iv_pct,
            iv_rank=vts.iv_rank,
            skew_zscore=vts.skew_zscore,
            warnings=ticker_warns,
        )

    logger.info(
        "screen_vol_surface complete",
        screened=len(tickers), elevated_skew=elevated_skew, inverted=inverted_ts,
    )
    return VolSurfaceScreen(
        tickers_screened=tickers, results=summaries,
        elevated_skew=elevated_skew, inverted_term_structure=inverted_ts,
        as_of=as_of, warnings=screen_warnings,
    )
