"""Options market analytics — IV surface, skew, GEX, max pain, term structure, P/C ratio."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


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


def compute_term_structure(surface: IVSurface, underlying_price: float) -> list[dict]:
    """
    Returns list of {expiry, dte, atm_iv} sorted by DTE for vol term structure.
    Useful for contango/backwardation analysis and vol calendar spreads.
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
