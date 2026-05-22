"""
High-frequency microstructure signals: roll yield, futures basis, VPIN,
realized variance components, bid-ask bounce correction, order flow imbalance.

Pure numpy — zero network calls, zero external data deps.

dim_132 — HF microstructure signals (target: 9)

Classes
-------
FuturesBasisMetrics
    Basis, basis%, roll yield, implied carry, market regime.

RealizedVolComponents
    Realized variance, bipower variation, jump/continuous decomposition.

FuturesMicrostructure
    .basis()               → FuturesBasisMetrics
    .roll_yield()          → annualized roll yield
    .term_structure_slope()→ linear fit slope across maturities
    .carry_signal()        → roll_yield / sigma (carry Sharpe)

RealizedVolAnalytics
    .realized_variance()   → sum(r^2)
    .bipower_variation()   → (pi/2) * sum(|r_t||r_{t-1}|)
    .decompose()           → RealizedVolComponents
    .jump_test()           → dict with jump_detected, jump_ratio, z_stat
    .realized_kernel()     → Barndorff-Nielsen kernel estimator

OrderFlowImbalance
    .ofi()                 → per-tick OFI array
    .ofi_lambda()          → OLS price impact coefficient
    .net_order_flow()      → signed volume array

HFSignalGenerator
    .vpin_signal()         → VPIN in [0,1]
    .momentum_signal()     → price momentum
    .mean_reversion_signal()→ z-score mean reversion
    .carry_signal()        → carry Sharpe
    .composite_signal()    → dict of all signals

Convenience functions
---------------------
futures_basis, roll_yield, realized_vol_decomposition, vpin, ofi_lambda
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FuturesBasisMetrics:
    """Futures basis and carry metrics."""

    basis: float            # futures_price - spot_price
    basis_pct: float        # basis / spot_price * 100
    roll_yield: float       # annualized roll yield (near/far - 1) * 365/days
    implied_carry: float    # annualized cost of carry log(F/S)/T
    market_regime: str      # 'backwardation' or 'contango'


@dataclass
class RealizedVolComponents:
    """Decomposition of realized variance into jump and continuous components."""

    realized_variance: float
    bipower_variation: float
    jump_component: float
    continuous_component: float
    jump_ratio: float
    realized_vol: float             # sqrt(RV)
    realized_vol_continuous: float  # sqrt(continuous)


# ---------------------------------------------------------------------------
# FuturesMicrostructure
# ---------------------------------------------------------------------------


class FuturesMicrostructure:
    """Futures-specific microstructure analytics."""

    # ------------------------------------------------------------------
    # Basis & carry
    # ------------------------------------------------------------------

    def basis(self, spot: float, futures: float, T: float = 0.25) -> FuturesBasisMetrics:
        """Compute full futures basis metrics.

        Parameters
        ----------
        spot    : current spot price
        futures : current futures price
        T       : time to expiry in years (used for implied carry)
        """
        b = futures - spot
        b_pct = b / spot * 100.0
        # Implied cost of carry: log(F/S) / T
        if T > 0 and spot > 0 and futures > 0:
            implied_carry = math.log(futures / spot) / T
        else:
            implied_carry = 0.0
        # Roll yield: not applicable here (needs near/far); use basis sign
        ry = 0.0  # roll_yield requires near/far, not spot/futures
        regime = "contango" if futures >= spot else "backwardation"
        return FuturesBasisMetrics(
            basis=b,
            basis_pct=b_pct,
            roll_yield=ry,
            implied_carry=implied_carry,
            market_regime=regime,
        )

    def roll_yield(self, near: float, far: float, days_to_roll: float) -> float:
        """Annualized roll yield.

        Positive in backwardation (near > far), negative in contango.

        Parameters
        ----------
        near         : near-contract price
        far          : far-contract price
        days_to_roll : calendar days until near expiry
        """
        if far <= 0 or days_to_roll <= 0:
            return 0.0
        return (near / far - 1.0) * (365.0 / days_to_roll)

    def term_structure_slope(
        self, prices: List[float], maturities: List[float]
    ) -> float:
        """Linear OLS slope of the futures term structure.

        A positive slope means contango; negative means backwardation.

        Parameters
        ----------
        prices     : list of futures prices across maturities
        maturities : list of time-to-expiry values (same length)
        """
        if len(prices) < 2 or len(prices) != len(maturities):
            return 0.0
        x = np.array(maturities, dtype=float)
        y = np.array(prices, dtype=float)
        # OLS slope
        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((x - xm) * (y - ym)) / denom)

    def carry_signal(
        self, near: float, far: float, days_to_roll: float, sigma: float
    ) -> float:
        """Carry Sharpe: annualized roll yield divided by volatility.

        Parameters
        ----------
        near         : near-contract price
        far          : far-contract price
        days_to_roll : calendar days until roll
        sigma        : annualized volatility
        """
        ry = self.roll_yield(near, far, days_to_roll)
        if sigma <= 0:
            return 0.0
        return ry / sigma


# ---------------------------------------------------------------------------
# RealizedVolAnalytics
# ---------------------------------------------------------------------------


class RealizedVolAnalytics:
    """Realized variance decomposition following Barndorff-Nielsen & Shephard."""

    def realized_variance(self, returns: np.ndarray) -> float:
        """Sum of squared returns (realized variance)."""
        r = np.asarray(returns, dtype=float)
        return float(np.sum(r ** 2))

    def bipower_variation(self, returns: np.ndarray) -> float:
        """Bipower variation — jump-robust estimator.

        BV = (pi/2) * sum(|r_t| * |r_{t-1}|)
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 2:
            return 0.0
        scaling = math.pi / 2.0
        return float(scaling * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))

    def decompose(self, returns: np.ndarray) -> RealizedVolComponents:
        """Full decomposition: RV, BV, jump/continuous components.

        Returns
        -------
        RealizedVolComponents
        """
        r = np.asarray(returns, dtype=float)
        rv = self.realized_variance(r)
        bv = self.bipower_variation(r)
        # Jump component = max(RV - BV, 0); continuous = min(BV, RV)
        # This guarantees: jump + continuous = RV exactly.
        jump = max(rv - bv, 0.0)
        cont = rv - jump   # = min(RV, BV) — ensures additive decomposition
        j_ratio = (jump / rv) if rv > 0 else 0.0
        return RealizedVolComponents(
            realized_variance=rv,
            bipower_variation=bv,
            jump_component=jump,
            continuous_component=cont,
            jump_ratio=j_ratio,
            realized_vol=math.sqrt(rv) if rv >= 0 else 0.0,
            realized_vol_continuous=math.sqrt(max(cont, 0.0)),
        )

    def jump_test(
        self, returns: np.ndarray, confidence: float = 0.99
    ) -> dict:
        """Barndorff-Nielsen & Shephard jump test.

        Returns
        -------
        dict with keys: 'jump_detected' (bool), 'jump_ratio' (float), 'z_stat' (float)
        """
        r = np.asarray(returns, dtype=float)
        n = len(r)
        if n < 4:
            return {"jump_detected": False, "jump_ratio": 0.0, "z_stat": 0.0}

        rv = self.realized_variance(r)
        bv = self.bipower_variation(r)
        jump = max(rv - bv, 0.0)
        j_ratio = jump / rv if rv > 0 else 0.0

        # Ratio test statistic: (RV - BV)/RV ~ N(0, variance)
        # Variance of ratio estimator (simplified BNS 2004)
        mu1 = math.sqrt(2.0 / math.pi)
        kappa3 = 0.0  # under Gaussian returns, third cumulant = 0
        # theta = ((pi/2)^2 + pi - 5) * mu1^{-4}
        theta = ((math.pi / 2) ** 2 + math.pi - 5) * (mu1 ** (-4))
        var_ratio = theta / n if n > 0 else 1.0
        z_stat = j_ratio / math.sqrt(max(var_ratio, 1e-12))

        # Critical value at 99% confidence (one-sided upper)
        # Z > 2.326 → jump detected
        critical = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(confidence, 2.326)
        jump_detected = bool(z_stat > critical)

        return {
            "jump_detected": jump_detected,
            "jump_ratio": j_ratio,
            "z_stat": z_stat,
        }

    def realized_kernel(self, returns: np.ndarray, bandwidth: int = 10) -> float:
        """Barndorff-Nielsen kernel-based realized variance.

        RK = RV + 2 * sum_{h=1}^{H} k(h/H) * gamma_h

        where gamma_h = sum_t r_t * r_{t-h} (h-th realized autocovariance)
        and k(x) = 1 - x is a Bartlett kernel.

        Parameters
        ----------
        returns   : high-frequency return array
        bandwidth : kernel bandwidth H
        """
        r = np.asarray(returns, dtype=float)
        n = len(r)
        rv = float(np.sum(r ** 2))
        correction = 0.0
        for h in range(1, bandwidth + 1):
            if h >= n:
                break
            # Bartlett kernel weight
            k = 1.0 - h / (bandwidth + 1.0)
            gamma_h = float(np.dot(r[h:], r[: n - h]))
            correction += k * gamma_h
        return rv + 2.0 * correction


# ---------------------------------------------------------------------------
# OrderFlowImbalance
# ---------------------------------------------------------------------------


class OrderFlowImbalance:
    """Order flow imbalance metrics following Cont, Kukanov & Stoikov (2014)."""

    def ofi(
        self,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> np.ndarray:
        """Per-tick order flow imbalance.

        OFI_t = dBidSize * I(bid_t >= bid_{t-1})
               - dAskSize * I(ask_t <= ask_{t-1})

        Parameters
        ----------
        bid_prices, ask_prices : arrays of shape (n,)
        bid_sizes, ask_sizes   : arrays of shape (n,)

        Returns
        -------
        ofi : np.ndarray of shape (n-1,)
        """
        bp = np.asarray(bid_prices, dtype=float)
        ap = np.asarray(ask_prices, dtype=float)
        bs = np.asarray(bid_sizes, dtype=float)
        as_ = np.asarray(ask_sizes, dtype=float)

        n = min(len(bp), len(ap), len(bs), len(as_))
        if n < 2:
            return np.array([])

        d_bid = bs[1:n] - bs[: n - 1]
        d_ask = as_[1:n] - as_[: n - 1]

        bid_up = (bp[1:n] >= bp[: n - 1]).astype(float)
        ask_down = (ap[1:n] <= ap[: n - 1]).astype(float)

        return d_bid * bid_up - d_ask * ask_down

    def ofi_lambda(
        self, price_changes: np.ndarray, ofi: np.ndarray
    ) -> float:
        """OLS estimate of price impact coefficient.

        price_change_t = lambda * OFI_t + epsilon

        Parameters
        ----------
        price_changes : array of mid-price changes
        ofi           : array of OFI values (same length)

        Returns
        -------
        lambda : float (OLS coefficient)
        """
        pc = np.asarray(price_changes, dtype=float)
        of = np.asarray(ofi, dtype=float)
        n = min(len(pc), len(of))
        if n < 2:
            return 0.0
        x = of[:n]
        y = pc[:n]
        xm = x.mean()
        denom = np.sum((x - xm) ** 2)
        if denom < 1e-15:
            return 0.0
        return float(np.sum((x - xm) * y) / denom)

    def net_order_flow(
        self, directions: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """Signed volume array.

        Parameters
        ----------
        directions : +1 for buy, -1 for sell
        volumes    : unsigned trade volumes

        Returns
        -------
        signed_volume : np.ndarray
        """
        d = np.asarray(directions, dtype=float)
        v = np.asarray(volumes, dtype=float)
        return d * v


# ---------------------------------------------------------------------------
# HFSignalGenerator
# ---------------------------------------------------------------------------


class HFSignalGenerator:
    """Combines microstructure signals into actionable trading signals."""

    def __init__(self) -> None:
        self._fm = FuturesMicrostructure()
        self._rv = RealizedVolAnalytics()
        self._ofi = OrderFlowImbalance()

    def vpin_signal(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bucket_size: float = 1000.0,
    ) -> float:
        """VPIN in [0, 1] — probability of informed trading."""
        return vpin(prices, volumes, bucket_size)

    def momentum_signal(self, prices: np.ndarray, window: int = 20) -> float:
        """Price momentum: (current - window-ago) / window-ago."""
        p = np.asarray(prices, dtype=float)
        if len(p) <= window:
            return 0.0
        return float((p[-1] - p[-window - 1]) / (p[-window - 1] + 1e-15))

    def mean_reversion_signal(
        self, prices: np.ndarray, window: int = 60
    ) -> float:
        """Z-score of price relative to rolling window (negative = buy signal)."""
        p = np.asarray(prices, dtype=float)
        if len(p) < window:
            return 0.0
        recent = p[-window:]
        mu = recent.mean()
        sigma = recent.std()
        if sigma < 1e-15:
            return 0.0
        return float(-(p[-1] - mu) / sigma)

    def carry_signal(
        self, near: float, far: float, days: float, sigma: float
    ) -> float:
        """Carry Sharpe (roll yield / sigma)."""
        return self._fm.carry_signal(near, far, days, sigma)

    def composite_signal(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        near_future: float,
        far_future: float,
        days: float,
    ) -> dict:
        """All signals combined into a single dict.

        Returns
        -------
        dict with keys: 'vpin', 'momentum', 'mean_reversion', 'carry',
                        'jump_ratio', 'realized_vol'
        """
        p = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(p + 1e-15))

        vol_comps = self._rv.decompose(returns) if len(returns) > 1 else None
        sigma = vol_comps.realized_vol if vol_comps else 1.0

        return {
            "vpin": self.vpin_signal(p, volumes, bucket_size=max(volumes.mean(), 1.0)),
            "momentum": self.momentum_signal(p),
            "mean_reversion": self.mean_reversion_signal(p),
            "carry": self.carry_signal(near_future, far_future, days, sigma),
            "jump_ratio": vol_comps.jump_ratio if vol_comps else 0.0,
            "realized_vol": vol_comps.realized_vol if vol_comps else 0.0,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def futures_basis(spot: float, futures: float, T: float = 0.25) -> FuturesBasisMetrics:
    """Compute futures basis metrics.

    Parameters
    ----------
    spot    : spot price
    futures : futures price
    T       : time to expiry in years
    """
    return FuturesMicrostructure().basis(spot, futures, T)


def roll_yield(near: float, far: float, days_to_roll: float) -> float:
    """Annualized roll yield between near and far futures contracts.

    Positive in backwardation, negative in contango.
    """
    return FuturesMicrostructure().roll_yield(near, far, days_to_roll)


def realized_vol_decomposition(returns: np.ndarray) -> RealizedVolComponents:
    """Decompose realized variance into jump and continuous components."""
    return RealizedVolAnalytics().decompose(returns)


def vpin(
    prices: np.ndarray, volumes: np.ndarray, bucket_size: float = 1000.0
) -> float:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Algorithm
    ---------
    1. Classify each trade as buy/sell using tick rule (price change sign).
    2. Aggregate into equal-volume buckets of size `bucket_size`.
    3. For each bucket: |V_buy - V_sell| / bucket_size.
    4. VPIN = rolling mean over all buckets.

    Returns
    -------
    float in [0, 1]
    """
    p = np.asarray(prices, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = min(len(p), len(v))
    if n < 2:
        return 0.0

    # Tick rule: direction = sign of price change; 0 → carry forward last
    diffs = np.diff(p[:n])
    directions = np.zeros(n, dtype=float)
    last_dir = 1.0
    for i, d in enumerate(diffs):
        if d > 0:
            last_dir = 1.0
        elif d < 0:
            last_dir = -1.0
        directions[i + 1] = last_dir
    directions[0] = directions[1] if n > 1 else 1.0

    buy_vol = np.where(directions > 0, v[:n], 0.0)
    sell_vol = np.where(directions < 0, v[:n], 0.0)

    # Build equal-volume buckets
    bucket_size = max(bucket_size, 1.0)
    buckets: list = []
    cum_buy = 0.0
    cum_sell = 0.0
    cum_vol = 0.0

    for i in range(n):
        bv_i = buy_vol[i]
        sv_i = sell_vol[i]
        remaining_buy = bv_i
        remaining_sell = sv_i
        remaining_vol = v[i]

        while cum_vol + remaining_vol >= bucket_size:
            fill = bucket_size - cum_vol
            # Proportional split
            frac = fill / (remaining_vol + 1e-15)
            cum_buy += frac * remaining_buy
            cum_sell += frac * remaining_sell
            remaining_buy -= frac * remaining_buy
            remaining_sell -= frac * remaining_sell
            remaining_vol -= fill
            buckets.append(abs(cum_buy - cum_sell) / bucket_size)
            cum_buy = 0.0
            cum_sell = 0.0
            cum_vol = 0.0

        cum_buy += remaining_buy
        cum_sell += remaining_sell
        cum_vol += remaining_vol

    # Flush last partial bucket if it has material content
    if cum_vol > 0 and buckets == []:
        # Only partial bucket — compute directly
        imb = abs(cum_buy - cum_sell) / (cum_vol + 1e-15)
        buckets.append(min(imb, 1.0))

    if not buckets:
        return 0.0

    return float(np.clip(np.mean(buckets), 0.0, 1.0))


def ofi_lambda(price_changes: np.ndarray, ofi: np.ndarray) -> float:
    """OLS price impact coefficient: price_change = lambda * OFI."""
    return OrderFlowImbalance().ofi_lambda(price_changes, ofi)
