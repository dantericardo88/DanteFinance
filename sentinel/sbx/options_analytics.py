"""Options market analytics — IV surface, skew, GEX, max pain, term structure, P/C ratio."""
from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Black-Scholes Greeks Engine ───────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — pure Python, no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict[str, Optional[float]]:
    """Compute the full Black-Scholes Greek set for a European option.

    Args:
        S: Underlying spot price
        K: Strike price
        T: Time to expiration in years (must be > 0)
        r: Risk-free rate (annualised, continuously compounded)
        sigma: Implied volatility (annualised)
        option_type: "call" or "put"

    Returns:
        dict with keys:
            delta   — dV/dS                  (directional sensitivity)
            gamma   — d²V/dS²                (convexity of delta)
            vega    — dV/dσ  (per 1-pt move) (vol sensitivity ×0.01 for 1%)
            theta   — dV/dt  (per calendar day, negative for long options)
            rho     — dV/dr  (per 1-pt rate move)
            charm   — dDelta/dt  (per calendar day; delta decay)
            vanna   — dDelta/dσ  (=d²V/dSdσ; cross-Greek)
            vomma   — d²V/dσ²   (vol of vol sensitivity)
            speed   — d³V/dS³   (gamma sensitivity to spot)
            color   — dGamma/dt  (gamma decay per calendar day)

    All values are None if inputs are invalid (T≤0, sigma≤0, S≤0, K≤0).
    Vega, theta, rho are expressed per 1-unit moves (not per 1% / 1bp).
    """
    result: dict[str, Optional[float]] = {
        k: None for k in ("delta", "gamma", "vega", "theta", "rho",
                           "charm", "vanna", "vomma", "speed", "color")
    }
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return result

    try:
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        nd1 = _norm_pdf(d1)
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)
        Nnd2 = _norm_cdf(-d2)

        disc = math.exp(-r * T)

        # ── The five standard Greeks ──────────────────────────────────────────
        if option_type == "call":
            delta = Nd1
            theta_daily = (
                -(S * nd1 * sigma) / (2 * sqrtT)
                - r * K * disc * Nd2
            ) / 365.0
            rho = K * T * disc * Nd2 / 100.0  # per 1% rate change
        else:
            delta = Nd1 - 1.0
            theta_daily = (
                -(S * nd1 * sigma) / (2 * sqrtT)
                + r * K * disc * Nnd2
            ) / 365.0
            rho = -K * T * disc * Nnd2 / 100.0

        gamma = nd1 / (S * sigma * sqrtT)
        vega = S * nd1 * sqrtT / 100.0  # per 1% vol change

        result["delta"] = round(delta, 6)
        result["gamma"] = round(gamma, 8)
        result["vega"] = round(vega, 6)
        result["theta"] = round(theta_daily, 6)
        result["rho"] = round(rho, 6)

        # ── Higher-order / cross Greeks ───────────────────────────────────────
        # Charm (delta decay): dDelta/dt per calendar day
        if option_type == "call":
            charm = -nd1 * (2 * r * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
        else:
            charm = -nd1 * (2 * r * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
        result["charm"] = round(charm / 365.0, 8)

        # Vanna: dDelta/dVol = d²V/(dS dσ) = vega × (1/S - d1/(sigma√T))
        # Equivalently: -nd1 × d2 / sigma
        vanna = -nd1 * d2 / sigma
        result["vanna"] = round(vanna, 8)

        # Vomma (volga): d²V/dσ² = vega × d1 × d2 / sigma
        vomma = vega * d1 * d2 / sigma
        result["vomma"] = round(vomma, 8)

        # Speed: dGamma/dS = -gamma/S × (d1/(sigma√T) + 1)
        speed = -gamma / S * (d1 / (sigma * sqrtT) + 1.0)
        result["speed"] = round(speed, 10)

        # Color (gamma decay): dGamma/dt per calendar day
        color = (
            -nd1 / (2 * S * T * sigma * sqrtT)
            * (2 * r * T + 1 + d1 * (2 * r * T - d2 * sigma * sqrtT) / (sigma * sqrtT))
        )
        result["color"] = round(color / 365.0, 10)

    except Exception as exc:
        logger.debug("bs_greeks: computation error", error=str(exc))

    return result


# ── Result Models ─────────────────────────────────────────────────────────────

class NormalizedContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    expiry: date
    strike: Decimal
    contract_type: str          # "call" | "put"
    iv: float
    open_interest: int
    volume: int
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    close_price: Optional[Decimal]
    underlying_price: float
    moneyness: float            # strike / underlying_price


class IVSurface(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    calls: dict[str, dict[str, float]]   # {expiry: {strike: iv}}
    puts: dict[str, dict[str, float]]    # {expiry: {strike: iv}}
    atm_iv: dict[str, float]             # expiry → ATM IV (avg call+put nearest strike)
    expiries: list[str]                  # sorted ISO date strings


class SkewMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    expiry: str
    dte: int
    risk_reversal_25d: Optional[float]  # 25d put IV − 25d call IV (positive = downside fear)
    put_spread_25d: Optional[float]     # 25d put IV − ATM put IV
    atm_iv: Optional[float]
    put_25d_iv: Optional[float]
    call_25d_iv: Optional[float]


class GEXResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    by_strike: dict[str, float]     # {strike_str: gex_dollars}
    net_gex: float                  # positive = dealers long gamma (suppressive)
    largest_positive_strike: Optional[float]
    largest_negative_strike: Optional[float]


class MaxPainResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    max_pain_strike: float
    pain_table: dict[str, float]    # {strike: total OI-weighted holder loss}
    underlying_price: float
    distance_pct: float             # (max_pain − spot) / spot * 100


class PCRatioResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    oi_ratio: float                 # put OI / call OI
    volume_ratio: float             # put vol / call vol
    total_put_oi: int
    total_call_oi: int
    total_put_volume: int
    total_call_volume: int
    by_expiry: dict[str, dict[str, float]]  # {expiry: {oi_ratio, volume_ratio, ...}}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dte(expiry_str: str) -> int:
    """Calendar days from today to expiry."""
    exp = date.fromisoformat(expiry_str)
    return max(0, (exp - date.today()).days)


def _expiry_str(expiry: date) -> str:
    return expiry.isoformat()


# ── Core Functions ────────────────────────────────────────────────────────────

def normalize_chain(raw_contracts: list[dict], underlying_price: float) -> list[NormalizedContract]:
    """
    Map raw Polygon option dicts to NormalizedContract.
    Filters out: zero/None IV, zero OI, non-positive strikes, missing expiry.
    """
    result: list[NormalizedContract] = []
    skipped = 0

    for raw in raw_contracts:
        try:
            iv = raw.get("implied_volatility")
            oi = raw.get("open_interest") or 0
            strike_raw = raw.get("strike_price")
            expiry_raw = raw.get("expiration_date")
            ctype = (raw.get("contract_type") or "").lower()

            # Hard filters
            if not iv or float(iv) <= 0:
                skipped += 1
                continue
            if int(oi) <= 0:
                skipped += 1
                continue
            if not strike_raw or float(strike_raw) <= 0:
                skipped += 1
                continue
            if not expiry_raw:
                skipped += 1
                continue
            if ctype not in ("call", "put"):
                skipped += 1
                continue

            strike = Decimal(str(strike_raw))
            moneyness = float(strike) / underlying_price

            delta_raw = raw.get("delta")
            gamma_raw = raw.get("gamma")
            theta_raw = raw.get("theta")
            vega_raw = raw.get("vega")
            close_raw = raw.get("close_price")

            result.append(NormalizedContract(
                ticker=raw.get("underlying_ticker", raw.get("ticker", "")),
                expiry=date.fromisoformat(str(expiry_raw)[:10]),
                strike=strike,
                contract_type=ctype,
                iv=float(iv),
                open_interest=int(oi),
                volume=int(raw.get("volume") or 0),
                delta=float(delta_raw) if delta_raw is not None else None,
                gamma=float(gamma_raw) if gamma_raw is not None else None,
                theta=float(theta_raw) if theta_raw is not None else None,
                vega=float(vega_raw) if vega_raw is not None else None,
                close_price=Decimal(str(close_raw)) if close_raw is not None else None,
                underlying_price=underlying_price,
                moneyness=moneyness,
            ))
        except Exception as exc:
            logger.warning("normalize_chain: skipping bad contract", error=str(exc))
            skipped += 1

    logger.info(
        "normalize_chain complete",
        total=len(raw_contracts),
        accepted=len(result),
        skipped=skipped,
    )
    return result


def build_iv_surface(contracts: list[NormalizedContract]) -> IVSurface:
    """
    Build IV surface: {expiry_date: {strike: iv}} for calls and puts separately.
    ATM IV per expiry = average of ATM call IV and ATM put IV (closest strike to spot).
    """
    ticker = contracts[0].ticker if contracts else ""
    calls: dict[str, dict[str, float]] = {}
    puts: dict[str, dict[str, float]] = {}

    for c in contracts:
        exp_str = _expiry_str(c.expiry)
        strike_str = str(c.strike)
        if c.contract_type == "call":
            calls.setdefault(exp_str, {})[strike_str] = c.iv
        else:
            puts.setdefault(exp_str, {})[strike_str] = c.iv

    expiries = sorted(set(list(calls.keys()) + list(puts.keys())))

    # ATM IV: for each expiry, find strike closest to underlying_price
    atm_iv: dict[str, float] = {}
    for exp_str in expiries:
        exp_contracts = [c for c in contracts if _expiry_str(c.expiry) == exp_str]
        if not exp_contracts:
            continue
        spot = exp_contracts[0].underlying_price
        atm_calls = [c for c in exp_contracts if c.contract_type == "call"]
        atm_puts = [c for c in exp_contracts if c.contract_type == "put"]

        ivs = []
        if atm_calls:
            nearest_call = min(atm_calls, key=lambda c: abs(float(c.strike) - spot))
            ivs.append(nearest_call.iv)
        if atm_puts:
            nearest_put = min(atm_puts, key=lambda c: abs(float(c.strike) - spot))
            ivs.append(nearest_put.iv)

        if ivs:
            atm_iv[exp_str] = sum(ivs) / len(ivs)

    logger.info("build_iv_surface", ticker=ticker, expiries=len(expiries))
    return IVSurface(
        ticker=ticker,
        calls=calls,
        puts=puts,
        atm_iv=atm_iv,
        expiries=expiries,
    )


def compute_skew(surface: IVSurface, underlying_price: float) -> list[SkewMetrics]:
    """Per expiry: risk reversal (25d put IV − 25d call IV) and put spread
    (25d put IV − ATM put IV). Uses moneyness bands: 25d ≈ 7.5% OTM."""
    results: list[SkewMetrics] = []

    for exp_str in surface.expiries:
        dte = _dte(exp_str)
        atm = surface.atm_iv.get(exp_str)

        # 25d put: moneyness ≈ 0.90–0.95 (5–10% OTM put)
        put_strikes = surface.puts.get(exp_str, {})
        call_strikes = surface.calls.get(exp_str, {})

        if not put_strikes and not call_strikes:
            continue

        # Find 25d put IV via moneyness (0.925 ± 0.075)
        target_put_m = 0.925
        target_call_m = 1.075

        def closest_iv(strike_iv: dict[str, float], target_m: float, band: float = 0.08) -> Optional[float]:
            candidates = {
                k: v for k, v in strike_iv.items()
                if abs(float(k) / underlying_price - target_m) <= band
            }
            if not candidates:
                return None
            best_k = min(candidates, key=lambda k: abs(float(k) / underlying_price - target_m))
            return candidates[best_k]

        put_25d_iv = closest_iv(put_strikes, target_put_m)
        call_25d_iv = closest_iv(call_strikes, target_call_m)

        risk_reversal = (
            put_25d_iv - call_25d_iv
            if put_25d_iv is not None and call_25d_iv is not None
            else None
        )

        # Put spread: 25d put IV − ATM put IV
        atm_put_iv: Optional[float] = None
        if put_strikes:
            best_atm_put = min(put_strikes, key=lambda k: abs(float(k) / underlying_price - 1.0))
            atm_put_iv = put_strikes[best_atm_put]

        put_spread = (
            put_25d_iv - atm_put_iv
            if put_25d_iv is not None and atm_put_iv is not None
            else None
        )

        results.append(SkewMetrics(
            expiry=exp_str,
            dte=dte,
            risk_reversal_25d=round(risk_reversal, 4) if risk_reversal is not None else None,
            put_spread_25d=round(put_spread, 4) if put_spread is not None else None,
            atm_iv=round(atm, 4) if atm is not None else None,
            put_25d_iv=round(put_25d_iv, 4) if put_25d_iv is not None else None,
            call_25d_iv=round(call_25d_iv, 4) if call_25d_iv is not None else None,
        ))

    results.sort(key=lambda s: s.dte)
    logger.info("compute_skew", expiries=len(results))
    return results


def compute_term_structure(surface: IVSurface, underlying_price: float = 0.0) -> list[dict]:  # noqa: ARG001
    """
    Returns list of {expiry, dte, atm_iv} sorted by DTE for vol term structure.
    Useful for contango/backwardation analysis and vol calendar spreads.
    underlying_price is accepted for API compatibility but not used internally
    (ATM IV is already embedded in the surface).
    """
    rows = []
    for exp_str in surface.expiries:
        atm = surface.atm_iv.get(exp_str)
        if atm is None:
            continue
        rows.append({
            "expiry": exp_str,
            "dte": _dte(exp_str),
            "atm_iv": round(atm, 4),
        })
    rows.sort(key=lambda r: r["dte"])
    logger.info("compute_term_structure", points=len(rows))
    return rows


def compute_gex(contracts: list[NormalizedContract], underlying_price: float) -> GEXResult:
    """GEX = gamma * OI * 100 * spot^2 * 0.01 per strike.
    Calls add positive GEX (dealers long gamma), puts subtract (dealers short gamma).
    Net positive GEX → mean-reversion / vol suppression near that strike cluster.
    Net negative GEX → trend-following / vol amplification."""
    ticker = contracts[0].ticker if contracts else ""
    gex_by_strike: dict[str, float] = {}

    for c in contracts:
        if c.gamma is None or c.gamma == 0:
            continue
        strike_str = str(c.strike)
        # Gamma exposure formula: gamma * OI * contract_multiplier * spot^2 * 1%
        raw_gex = c.gamma * c.open_interest * 100 * (underlying_price ** 2) * 0.01
        # Sign: calls add positive GEX, puts subtract (dealers short puts = short gamma)
        signed_gex = raw_gex if c.contract_type == "call" else -raw_gex
        gex_by_strike[strike_str] = gex_by_strike.get(strike_str, 0.0) + signed_gex

    net_gex = sum(gex_by_strike.values())

    pos = {k: v for k, v in gex_by_strike.items() if v > 0}
    neg = {k: v for k, v in gex_by_strike.items() if v < 0}
    largest_pos = float(max(pos, key=lambda k: pos[k])) if pos else None
    largest_neg = float(min(neg, key=lambda k: neg[k])) if neg else None

    # Round for readability (dollar-denominated)
    gex_rounded = {k: round(v, 2) for k, v in gex_by_strike.items()}
    logger.info(
        "compute_gex",
        ticker=ticker,
        strikes=len(gex_by_strike),
        net_gex=round(net_gex, 2),
    )
    return GEXResult(
        ticker=ticker,
        by_strike=gex_rounded,
        net_gex=round(net_gex, 2),
        largest_positive_strike=largest_pos,
        largest_negative_strike=largest_neg,
    )


def compute_max_pain(contracts: list[NormalizedContract]) -> MaxPainResult:
    """Max pain = strike minimizing total option-holder loss (writers' gain).
    pain(K) = Σ (strike-K)*OI for calls above K + Σ (K-strike)*OI for puts below K."""
    ticker = contracts[0].ticker if contracts else ""
    underlying_price = contracts[0].underlying_price if contracts else 0.0

    strikes = sorted(set(float(c.strike) for c in contracts))
    if not strikes:
        return MaxPainResult(
            ticker=ticker,
            max_pain_strike=underlying_price,
            pain_table={},
            underlying_price=underlying_price,
            distance_pct=0.0,
        )

    pain_table: dict[str, float] = {}
    for K in strikes:
        pain = 0.0
        for c in contracts:
            s = float(c.strike)
            if c.contract_type == "call" and s > K:
                pain += (s - K) * c.open_interest
            elif c.contract_type == "put" and s < K:
                pain += (K - s) * c.open_interest
        pain_table[str(K)] = round(pain, 2)

    max_pain_strike = float(min(pain_table, key=lambda k: pain_table[k]))
    distance_pct = (max_pain_strike - underlying_price) / underlying_price * 100

    logger.info(
        "compute_max_pain",
        ticker=ticker,
        max_pain=max_pain_strike,
        distance_pct=round(distance_pct, 2),
    )
    return MaxPainResult(
        ticker=ticker,
        max_pain_strike=max_pain_strike,
        pain_table=pain_table,
        underlying_price=underlying_price,
        distance_pct=round(distance_pct, 2),
    )


def compute_pc_ratio(contracts: list[NormalizedContract]) -> PCRatioResult:
    """Put/call OI and volume ratios — overall and per-expiry.
    Ratio > 1 signals elevated put demand (bearish/defensive positioning)."""
    ticker = contracts[0].ticker if contracts else ""
    total_put_oi = sum(c.open_interest for c in contracts if c.contract_type == "put")
    total_call_oi = sum(c.open_interest for c in contracts if c.contract_type == "call")
    total_put_vol = sum(c.volume for c in contracts if c.contract_type == "put")
    total_call_vol = sum(c.volume for c in contracts if c.contract_type == "call")

    oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else float("inf")
    vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else float("inf")

    # Per-expiry breakdown
    expiries = sorted(set(_expiry_str(c.expiry) for c in contracts))
    by_expiry: dict[str, dict[str, float]] = {}
    for exp_str in expiries:
        exp = [c for c in contracts if _expiry_str(c.expiry) == exp_str]
        p_oi = sum(c.open_interest for c in exp if c.contract_type == "put")
        c_oi = sum(c.open_interest for c in exp if c.contract_type == "call")
        p_vol = sum(c.volume for c in exp if c.contract_type == "put")
        c_vol = sum(c.volume for c in exp if c.contract_type == "call")
        by_expiry[exp_str] = {
            "oi_ratio": round(p_oi / c_oi, 4) if c_oi > 0 else None,
            "volume_ratio": round(p_vol / c_vol, 4) if c_vol > 0 else None,
            "put_oi": p_oi,
            "call_oi": c_oi,
        }

    logger.info(
        "compute_pc_ratio",
        ticker=ticker,
        oi_ratio=round(oi_ratio, 4),
        vol_ratio=round(vol_ratio, 4),
    )
    return PCRatioResult(
        ticker=ticker,
        oi_ratio=round(oi_ratio, 4),
        volume_ratio=round(vol_ratio, 4),
        total_put_oi=total_put_oi,
        total_call_oi=total_call_oi,
        total_put_volume=total_put_vol,
        total_call_volume=total_call_vol,
        by_expiry=by_expiry,
    )


# ── Top-Level Orchestrator ────────────────────────────────────────────────────

def get_options_summary(
    ticker: str,
    contracts_raw: list[dict],
    underlying_price: float,
) -> dict:
    """Full pipeline: normalize → surface → skew → term structure → GEX → max pain → P/C.
    Returns dict with keys: ticker, underlying_price, contract_count, surface, skew,
    term_structure, gex, max_pain, pc_ratio."""
    logger.info("get_options_summary start", ticker=ticker, raw_contracts=len(contracts_raw))

    contracts = normalize_chain(contracts_raw, underlying_price)
    if not contracts:
        logger.warning("get_options_summary: no valid contracts after normalization", ticker=ticker)
        return {
            "ticker": ticker,
            "underlying_price": underlying_price,
            "contract_count": 0,
            "surface": None,
            "skew": [],
            "term_structure": [],
            "gex": None,
            "max_pain": None,
            "pc_ratio": None,
        }

    surface = build_iv_surface(contracts)
    skew = compute_skew(surface, underlying_price)
    term_structure = compute_term_structure(surface, underlying_price)
    gex = compute_gex(contracts, underlying_price)
    max_pain = compute_max_pain(contracts)
    pc_ratio = compute_pc_ratio(contracts)

    logger.info(
        "get_options_summary complete",
        ticker=ticker,
        contracts=len(contracts),
        expiries=len(surface.expiries),
        net_gex=gex.net_gex,
        max_pain=max_pain.max_pain_strike,
    )

    return {
        "ticker": ticker,
        "underlying_price": underlying_price,
        "contract_count": len(contracts),
        "surface": surface,
        "skew": skew,
        "term_structure": term_structure,
        "gex": gex,
        "max_pain": max_pain,
        "pc_ratio": pc_ratio,
    }


# ── Full-Chain Greeks ─────────────────────────────────────────────────────────

class ContractGreeks(BaseModel):
    """All Greeks for a single option contract in the chain."""
    model_config = ConfigDict(frozen=True)

    expiry: str
    strike: float
    contract_type: str          # "call" | "put"
    iv: float
    dte: int
    moneyness: float            # strike / spot
    # Standard Greeks
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]       # per 1% vol change
    theta: Optional[float]      # per calendar day
    rho: Optional[float]        # per 1% rate change
    # Higher-order Greeks
    charm: Optional[float]      # dDelta/dt per calendar day
    vanna: Optional[float]      # dDelta/dVol
    vomma: Optional[float]      # d²V/dσ²
    speed: Optional[float]      # dGamma/dS
    color: Optional[float]      # dGamma/dt per calendar day


class ChainGreeks(BaseModel):
    """Full Greeks for every strike/expiry in the options chain."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    underlying_price: float
    risk_free_rate: float
    as_of: str
    calls: list[ContractGreeks]
    puts: list[ContractGreeks]
    total_contracts: int
    warnings: list[str]


def compute_chain_greeks(
    contracts: list[NormalizedContract],
    underlying_price: float,
    risk_free_rate: float = 0.05,
) -> ChainGreeks:
    """Compute full Black-Scholes Greeks for every contract in the chain.

    Covers all 5 standard Greeks (delta, gamma, vega, theta, rho) plus
    charm (dDelta/dt), vanna (dDelta/dVol), vomma (d²V/dσ²), speed, and color.

    Args:
        contracts: Normalised option contracts (from normalize_chain).
        underlying_price: Current spot price of the underlying.
        risk_free_rate: Annualised risk-free rate (default 5%).

    Returns:
        ChainGreeks with calls and puts lists sorted by expiry then strike.
    """
    from datetime import datetime as _dt
    as_of = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    calls: list[ContractGreeks] = []
    puts: list[ContractGreeks] = []

    for c in contracts:
        dte = max(0, (c.expiry - date.today()).days)
        T = dte / 365.0

        greeks = bs_greeks(
            S=underlying_price,
            K=float(c.strike),
            T=T,
            r=risk_free_rate,
            sigma=c.iv,
            option_type=c.contract_type,
        )

        cg = ContractGreeks(
            expiry=c.expiry.isoformat(),
            strike=float(c.strike),
            contract_type=c.contract_type,
            iv=round(c.iv, 6),
            dte=dte,
            moneyness=round(c.moneyness, 6),
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            vega=greeks["vega"],
            theta=greeks["theta"],
            rho=greeks["rho"],
            charm=greeks["charm"],
            vanna=greeks["vanna"],
            vomma=greeks["vomma"],
            speed=greeks["speed"],
            color=greeks["color"],
        )

        if c.contract_type == "call":
            calls.append(cg)
        else:
            puts.append(cg)

    # Sort: expiry ascending, then strike ascending
    calls.sort(key=lambda x: (x.expiry, x.strike))
    puts.sort(key=lambda x: (x.expiry, x.strike))

    total = len(calls) + len(puts)
    logger.info(
        "compute_chain_greeks complete",
        ticker=contracts[0].ticker if contracts else "",
        calls=len(calls),
        puts=len(puts),
    )

    return ChainGreeks(
        ticker=contracts[0].ticker if contracts else "",
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        as_of=as_of,
        calls=calls,
        puts=puts,
        total_contracts=total,
        warnings=warnings,
    )


# ── Options Chain Heatmap ─────────────────────────────────────────────────────

class HeatmapCell(BaseModel):
    """A single cell in the options chain heat map."""
    model_config = ConfigDict(frozen=True)

    strike: float
    expiry: str
    dte: int
    call_iv: Optional[float]
    put_iv: Optional[float]
    call_delta: Optional[float]
    put_delta: Optional[float]
    call_gamma: Optional[float]
    put_gamma: Optional[float]
    call_oi: Optional[int]
    put_oi: Optional[int]
    call_volume: Optional[int]
    put_volume: Optional[int]
    net_gex: Optional[float]       # call_gamma×call_oi − put_gamma×put_oi (× 100 × S²)
    moneyness: float               # strike / spot


class OptionsHeatmap(BaseModel):
    """Rectangular strike × expiry heatmap for terminal display."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    underlying_price: float
    strikes: list[float]           # sorted ascending
    expiries: list[str]            # sorted ascending (ISO date)
    cells: list[HeatmapCell]       # flat list; index by (strike, expiry)
    atm_strike: Optional[float]    # nearest listed strike to spot
    as_of: str


def build_options_heatmap(
    contracts: list[NormalizedContract],
    chain_greeks: ChainGreeks,
    underlying_price: float,
) -> OptionsHeatmap:
    """Build a strike × expiry heatmap data structure for terminal display.

    Merges contract data (IV, OI, volume) with pre-computed Greeks into a flat
    list of HeatmapCell objects.  The terminal renderer can pivot by (expiry,
    strike) to produce a 2-D grid.

    Args:
        contracts: Normalised contracts from normalize_chain.
        chain_greeks: Output of compute_chain_greeks for the same contracts.
        underlying_price: Current spot price.

    Returns:
        OptionsHeatmap with all unique (strike, expiry) pairs covered.
    """
    from datetime import datetime as _dt
    as_of = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Index contract data by (expiry_str, strike)
    call_data: dict[tuple[str, float], NormalizedContract] = {}
    put_data: dict[tuple[str, float], NormalizedContract] = {}
    for c in contracts:
        key = (c.expiry.isoformat(), float(c.strike))
        if c.contract_type == "call":
            call_data[key] = c
        else:
            put_data[key] = c

    # Index greeks by (expiry, strike)
    call_greeks: dict[tuple[str, float], ContractGreeks] = {
        (cg.expiry, cg.strike): cg for cg in chain_greeks.calls
    }
    put_greeks: dict[tuple[str, float], ContractGreeks] = {
        (cg.expiry, cg.strike): cg for cg in chain_greeks.puts
    }

    # Enumerate all unique (expiry, strike) pairs
    all_keys: set[tuple[str, float]] = (
        set(call_data.keys()) | set(put_data.keys())
    )

    strikes_set: set[float] = {k[1] for k in all_keys}
    expiries_set: set[str] = {k[0] for k in all_keys}
    sorted_strikes = sorted(strikes_set)
    sorted_expiries = sorted(expiries_set)

    # Find ATM strike
    atm_strike: Optional[float] = None
    if sorted_strikes:
        atm_strike = min(sorted_strikes, key=lambda s: abs(s - underlying_price))

    cells: list[HeatmapCell] = []
    for exp in sorted_expiries:
        dte = max(0, (_dte(exp)))
        for strike in sorted_strikes:
            key = (exp, strike)
            c_contract = call_data.get(key)
            p_contract = put_data.get(key)

            # Skip cells where neither call nor put exists
            if c_contract is None and p_contract is None:
                continue

            c_gk = call_greeks.get(key)
            p_gk = put_greeks.get(key)

            # Net GEX for this cell: (call_gamma - put_gamma) × avg_OI × 100 × S²
            net_gex: Optional[float] = None
            if c_gk and c_gk.gamma is not None and p_gk and p_gk.gamma is not None:
                c_oi = c_contract.open_interest if c_contract else 0
                p_oi = p_contract.open_interest if p_contract else 0
                net_gex = round(
                    (c_gk.gamma * c_oi - p_gk.gamma * p_oi) * 100 * underlying_price ** 2,
                    2,
                )

            moneyness = round(strike / underlying_price, 6) if underlying_price > 0 else 0.0

            cells.append(HeatmapCell(
                strike=strike,
                expiry=exp,
                dte=dte,
                call_iv=round(c_contract.iv, 6) if c_contract else None,
                put_iv=round(p_contract.iv, 6) if p_contract else None,
                call_delta=c_gk.delta if c_gk else None,
                put_delta=p_gk.delta if p_gk else None,
                call_gamma=c_gk.gamma if c_gk else None,
                put_gamma=p_gk.gamma if p_gk else None,
                call_oi=c_contract.open_interest if c_contract else None,
                put_oi=p_contract.open_interest if p_contract else None,
                call_volume=c_contract.volume if c_contract else None,
                put_volume=p_contract.volume if p_contract else None,
                net_gex=net_gex,
                moneyness=moneyness,
            ))

    logger.info(
        "build_options_heatmap complete",
        ticker=chain_greeks.ticker,
        cells=len(cells),
        strikes=len(sorted_strikes),
        expiries=len(sorted_expiries),
    )

    return OptionsHeatmap(
        ticker=chain_greeks.ticker,
        underlying_price=underlying_price,
        strikes=sorted_strikes,
        expiries=sorted_expiries,
        cells=cells,
        atm_strike=atm_strike,
        as_of=as_of,
    )


# ── Unified Options Analytics Dashboard ──────────────────────────────────────

def get_options_dashboard(
    ticker: str,
    contracts_raw: list[dict],
    underlying_price: float,
    risk_free_rate: float = 0.05,
) -> dict:
    """Unified options analytics dashboard — full pipeline including Greeks and heatmap.

    Extends get_options_summary with:
      - Full Greeks for all strikes/expiries (delta, gamma, vega, theta, rho,
        charm, vanna, vomma, speed, color)
      - Options chain heatmap (strike × expiry grid with Greeks, IV, OI, GEX)

    Args:
        ticker: Underlying ticker symbol.
        contracts_raw: Raw option contract dicts (Polygon format).
        underlying_price: Current spot price.
        risk_free_rate: Annualised risk-free rate (default 5%).

    Returns:
        Dict with all get_options_summary keys plus:
          chain_greeks: ChainGreeks — all Greeks per contract
          heatmap: OptionsHeatmap — 2-D grid ready for terminal rendering
    """
    logger.info("get_options_dashboard start", ticker=ticker, raw_contracts=len(contracts_raw))

    # Run the base pipeline
    base = get_options_summary(ticker, contracts_raw, underlying_price)

    contracts: list[NormalizedContract] = []
    if base.get("contract_count", 0) > 0 and base.get("surface") is not None:
        # Re-normalise to get typed contracts list (base pipeline returns the surface, not contracts)
        contracts = normalize_chain(contracts_raw, underlying_price)

    chain_greeks: Optional[ChainGreeks] = None
    heatmap: Optional[OptionsHeatmap] = None

    if contracts:
        chain_greeks = compute_chain_greeks(contracts, underlying_price, risk_free_rate)
        heatmap = build_options_heatmap(contracts, chain_greeks, underlying_price)

    logger.info(
        "get_options_dashboard complete",
        ticker=ticker,
        greeks_computed=chain_greeks.total_contracts if chain_greeks else 0,
        heatmap_cells=len(heatmap.cells) if heatmap else 0,
    )

    return {
        **base,
        "chain_greeks": chain_greeks,
        "heatmap": heatmap,
    }
