"""Convertible bond analytics engine — bond floor, parity, greeks, verdict, screener."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Pydantic Models ────────────────────────────────────────────────────────────

class ConvertibleTerms(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    face_value: float = 1000.0
    coupon_rate: float                  # annual coupon as fraction, e.g. 0.025 = 2.5%
    maturity_years: float               # remaining years to maturity
    conversion_ratio: float             # shares per bond
    current_stock_price: float          # live equity price
    straight_bond_yield: float          # y = risk-free + credit spread, e.g. 0.07
    market_price: Optional[float] = None  # actual CB market price if known
    implied_vol: float = 0.30           # equity vol assumption
    risk_free_rate: float = 0.045


class ConvertibleGreeks(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta: float     # equity sensitivity 0-1
    gamma: float     # delta rate of change
    theta: float     # time decay per year (negative)
    rho: float       # rate sensitivity


class ConvertibleAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    bond_floor: float              # straight bond PV
    parity: float                  # conversion_ratio × stock_price
    market_price: float            # estimated or input
    conversion_price: float        # face / conversion_ratio
    premium_pct: float             # (market_price - parity) / parity * 100
    investment_premium_pct: float  # (market_price - bond_floor) / bond_floor * 100
    breakeven_years: Optional[float]  # payback period for premium
    greeks: ConvertibleGreeks
    verdict: str                   # equity-like / balanced / bond-like
    coupon_income_annual: float
    as_of: str
    warnings: list[str]


class ConvertibleSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    parity: float
    premium_pct: float
    delta: float
    verdict: str
    bond_floor: float


class ConvertibleScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    results: list[ConvertibleSummary]
    equity_like: list[str]    # delta > 0.70
    balanced: list[str]       # 0.30 <= delta <= 0.70
    bond_like: list[str]      # delta < 0.30
    avg_delta: Optional[float]
    as_of: str
    warnings: list[str]


# ── Math helpers ───────────────────────────────────────────────────────────────

def _bond_floor(face: float, coupon_rate: float, maturity_years: float, y: float) -> float:
    """PV of coupon payments + PV of face at maturity (annual coupon frequency)."""
    n = int(round(maturity_years))
    if n < 1:
        # Treat as zero-coupon: just PV of face
        return face / ((1.0 + y) ** maturity_years)

    coupon = face * coupon_rate
    pv_coupons = sum(coupon / ((1.0 + y) ** t) for t in range(1, n + 1))
    pv_face = face / ((1.0 + y) ** n)
    return pv_coupons + pv_face


def _compute_greeks(
    S: float,       # stock price
    K: float,       # conversion price = face / conversion_ratio
    r: float,       # risk-free rate
    sigma: float,   # implied vol
    T: float,       # maturity years
    face: float,
) -> tuple[ConvertibleGreeks, list[str]]:
    """Black-Scholes greeks for the embedded call option."""
    from scipy.stats import norm  # lazy import inside function body

    warnings: list[str] = []

    if T < 0.01:
        warnings.append(f"maturity_years={T:.4f} < 0.01 — greeks set to zero")
        return (
            ConvertibleGreeks(delta=0.0, gamma=0.0, theta=0.0, rho=0.0),
            warnings,
        )

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    if sigma_sqrt_T < 0.0001:
        warnings.append(f"σ*√T={sigma_sqrt_T:.6f} < 0.0001 — greeks set to zero")
        return (
            ConvertibleGreeks(delta=0.0, gamma=0.0, theta=0.0, rho=0.0),
            warnings,
        )

    if S <= 0.0 or K <= 0.0:
        warnings.append("Non-positive stock or conversion price — greeks set to zero")
        return (
            ConvertibleGreeks(delta=0.0, gamma=0.0, theta=0.0, rho=0.0),
            warnings,
        )

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    delta = float(norm.cdf(d1))
    n_prime_d1 = float(norm.pdf(d1))

    gamma = n_prime_d1 / (S * sigma_sqrt_T)
    theta = -0.5 * S * sigma * n_prime_d1 / sqrt_T
    rho = K * T * math.exp(-r * T) * float(norm.cdf(d2))

    return (
        ConvertibleGreeks(
            delta=round(delta, 6),
            gamma=round(gamma, 8),
            theta=round(theta, 6),
            rho=round(rho, 6),
        ),
        warnings,
    )


def _verdict(delta: float) -> str:
    if delta > 0.70:
        return "equity-like"
    if delta < 0.30:
        return "bond-like"
    return "balanced"


# ── Entry Point 1: pure sync analytics ────────────────────────────────────────

def analyze_convertible(terms: ConvertibleTerms) -> ConvertibleAnalytics:
    """
    Pure-sync convertible bond analytics.  No I/O.

    Computes: bond floor (investment value), parity (conversion value), market price
    estimate, conversion premium, investment premium, breakeven, Black-Scholes greeks,
    and verdict (equity-like / balanced / bond-like).
    """
    warnings: list[str] = []
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    S = terms.current_stock_price
    y = terms.straight_bond_yield
    face = terms.face_value
    ratio = terms.conversion_ratio
    T = terms.maturity_years
    sigma = terms.implied_vol
    r = terms.risk_free_rate

    # 1. Straight bond value (bond floor / investment value)
    floor = _bond_floor(face, terms.coupon_rate, T, y)

    # 2. Conversion value (parity)
    parity = ratio * S

    # 3. Conversion price
    conversion_price = face / ratio if ratio > 0 else float("nan")

    # 4. Market price — use input if supplied, else estimate
    if terms.market_price is not None:
        mkt = terms.market_price
    else:
        mkt = max(floor, parity) * 1.05
        warnings.append("market_price not provided — estimated as max(bond_floor, parity) * 1.05")

    # 5. Conversion premium
    if parity > 0:
        premium_pct = (mkt - parity) / parity * 100.0
    else:
        premium_pct = 0.0
        warnings.append("parity is zero — conversion_premium set to 0")

    # 6. Investment premium
    if floor > 0:
        investment_premium_pct = (mkt - floor) / floor * 100.0
    else:
        investment_premium_pct = 0.0
        warnings.append("bond_floor is zero — investment_premium set to 0")

    # 7. Annual coupon income
    coupon_income_annual = face * terms.coupon_rate

    # 8. Breakeven (payback period) — years to recoup premium via coupon income
    premium_dollars = mkt - parity
    if coupon_income_annual > 0 and premium_dollars > 0:
        breakeven_years = premium_dollars / coupon_income_annual
    elif premium_dollars <= 0:
        breakeven_years = 0.0  # already at or below parity
    else:
        breakeven_years = None
        warnings.append("coupon_income_annual is zero — breakeven undefined")

    # 9. Greeks (Black-Scholes for embedded call)
    greeks, greek_warnings = _compute_greeks(S, conversion_price, r, sigma, T, face)
    warnings.extend(greek_warnings)

    # 10. Verdict
    verdict = _verdict(greeks.delta)

    logger.info(
        "analyze_convertible",
        ticker=terms.ticker,
        bond_floor=round(floor, 4),
        parity=round(parity, 4),
        market_price=round(mkt, 4),
        delta=greeks.delta,
        verdict=verdict,
    )

    return ConvertibleAnalytics(
        ticker=terms.ticker,
        bond_floor=round(floor, 4),
        parity=round(parity, 4),
        market_price=round(mkt, 4),
        conversion_price=round(conversion_price, 4),
        premium_pct=round(premium_pct, 4),
        investment_premium_pct=round(investment_premium_pct, 4),
        breakeven_years=round(breakeven_years, 4) if breakeven_years is not None else None,
        greeks=greeks,
        verdict=verdict,
        coupon_income_annual=round(coupon_income_annual, 4),
        as_of=as_of,
        warnings=warnings,
    )


# ── yfinance price fetch (sync, runs in thread) ────────────────────────────────

def _fetch_price_sync(ticker: str) -> Optional[float]:
    """Sync yfinance price fetch — intended to run inside asyncio.to_thread."""
    import yfinance as yf  # lazy import

    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            # Fallback: fast_info
            fast = yf.Ticker(ticker).fast_info
            price = getattr(fast, "last_price", None)
        return float(price) if price is not None else None
    except Exception as exc:
        logger.warning("_fetch_price_sync failed", ticker=ticker, error=str(exc))
        return None


# ── Entry Point 2: async screener ─────────────────────────────────────────────

async def screen_convertibles(tickers: list[str]) -> ConvertibleScreen:
    """
    Fetch live stock prices via yfinance (parallel) and run convertible analytics.

    Default CB terms per ticker:
      - face=1000, coupon=0.025, maturity_years=3.0
      - conversion_price = spot * 1.20  (20% premium at issuance)
      - conversion_ratio = 1000 / conversion_price
      - straight_bond_yield = 0.07
    """
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    screen_warnings: list[str] = []

    # Parallel price fetches
    price_tasks = [
        asyncio.to_thread(_fetch_price_sync, ticker)
        for ticker in tickers
    ]
    prices: list[Optional[float]] = await asyncio.gather(*price_tasks)

    results: list[ConvertibleSummary] = []
    equity_like: list[str] = []
    balanced: list[str] = []
    bond_like: list[str] = []
    deltas: list[float] = []

    for ticker, price in zip(tickers, prices):
        if price is None or price <= 0:
            screen_warnings.append(f"{ticker}: could not fetch price — skipped")
            logger.warning("screen_convertibles: no price", ticker=ticker)
            continue

        # Build default CB terms: conversion_price 20% above spot at issuance
        conversion_price = price * 1.20
        conversion_ratio = 1000.0 / conversion_price

        terms = ConvertibleTerms(
            ticker=ticker,
            face_value=1000.0,
            coupon_rate=0.025,
            maturity_years=3.0,
            conversion_ratio=conversion_ratio,
            current_stock_price=price,
            straight_bond_yield=0.07,
            market_price=None,
            implied_vol=0.30,
            risk_free_rate=0.045,
        )

        try:
            analytics = analyze_convertible(terms)
        except Exception as exc:
            screen_warnings.append(f"{ticker}: analytics error — {exc}")
            logger.warning("screen_convertibles: analytics failed", ticker=ticker, error=str(exc))
            continue

        summary = ConvertibleSummary(
            ticker=ticker,
            parity=analytics.parity,
            premium_pct=analytics.premium_pct,
            delta=analytics.greeks.delta,
            verdict=analytics.verdict,
            bond_floor=analytics.bond_floor,
        )
        results.append(summary)
        deltas.append(analytics.greeks.delta)

        if analytics.verdict == "equity-like":
            equity_like.append(ticker)
        elif analytics.verdict == "balanced":
            balanced.append(ticker)
        else:
            bond_like.append(ticker)

    avg_delta = float(np.mean(deltas)) if deltas else None

    logger.info(
        "screen_convertibles complete",
        screened=len(tickers),
        returned=len(results),
        equity_like=len(equity_like),
        balanced=len(balanced),
        bond_like=len(bond_like),
        avg_delta=round(avg_delta, 4) if avg_delta is not None else None,
    )

    return ConvertibleScreen(
        tickers_screened=len(tickers),
        results=results,
        equity_like=equity_like,
        balanced=balanced,
        bond_like=bond_like,
        avg_delta=round(avg_delta, 4) if avg_delta is not None else None,
        as_of=as_of,
        warnings=screen_warnings,
    )
