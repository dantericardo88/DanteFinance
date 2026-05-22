"""
Trade Flow Analysis — VWAP/TWAP execution engines, trade classification,
VPIN, Kyle lambda, and futures basis analytics.
Pure numpy — zero network calls, zero external data deps.

dim_122 — Trade flow analysis (VWAP/TWAP/VPIN/Kyle lambda) (target: 9)

Classes
-------
TradeFlowMetrics
    Container for aggregate trade-flow statistics.

TradeClassifier
    .tick_rule()                  → signed direction array (+1/-1)
    .lee_ready()                  → Lee-Ready classification vs midquote
    .bulk_volume_classification() → BVC using standardised price changes

VWAPEngine
    .compute_vwap()              → float
    .vwap_schedule()             → participation weights array
    .tracking_error()            → bps deviation from market VWAP

TWAPEngine
    .compute_twap()              → float
    .twap_schedule()             → uniform-slice array
    .adaptive_twap()             → vol-weighted slice array (more when vol is low)

VPINCalculator
    .compute()                   → VPIN scalar
    .vpin_series()               → rolling VPIN array

FuturesBasis
    .basis()                     → future - spot
    .basis_bps()                 → (future - spot) / spot * 10000
    .roll_yield()                → annualised roll yield
    .implied_repo()              → implied repo rate

Module-level convenience functions
-----------------------------------
classify_trades_tick_rule, compute_vwap, compute_vpin, kyle_lambda, roll_yield
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TradeFlowMetrics:
    """Aggregate trade-flow statistics for a market session."""

    kyle_lambda: float
    vpin: float
    buy_sell_ratio: float
    net_order_flow: float
    price_impact_bps_per_M: float   # bps per $1 M of signed order flow
    informed_trading_probability: float  # simplified PIN estimate (0-1)


# ---------------------------------------------------------------------------
# Trade Classifier
# ---------------------------------------------------------------------------


class TradeClassifier:
    """Classify individual trades as buyer- or seller-initiated.

    All methods return arrays of +1 (buy) or -1 (sell).
    """

    def tick_rule(self, prices: np.ndarray) -> np.ndarray:
        """+1 if price rose vs last trade, -1 if price fell, carry forward if unchanged.

        The first observation is assumed to be a buy (+1).

        Parameters
        ----------
        prices : array-like, shape (N,)

        Returns
        -------
        directions : ndarray of int8, shape (N,)
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        directions = np.ones(n, dtype=np.int8)
        for i in range(1, n):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                directions[i] = 1
            elif diff < 0:
                directions[i] = -1
            else:
                directions[i] = directions[i - 1]  # carry forward
        return directions

    def lee_ready(self, prices: np.ndarray, quotes: np.ndarray) -> np.ndarray:
        """Lee-Ready (1991) algorithm: compare trade price to mid-quote.

        Parameters
        ----------
        prices : array-like, shape (N,) — trade prices
        quotes : array-like, shape (N,) — concurrent mid-quotes

        Returns
        -------
        directions : ndarray of int8, shape (N,) — +1 buy, -1 sell
        """
        prices = np.asarray(prices, dtype=float)
        quotes = np.asarray(quotes, dtype=float)
        above = prices > quotes
        below = prices < quotes
        directions = np.where(above, np.int8(1), np.where(below, np.int8(-1), np.int8(0)))
        # Apply tick rule for tie-breaking (at the quote)
        tick = self.tick_rule(prices)
        tie_mask = directions == 0
        directions = np.where(tie_mask, tick, directions).astype(np.int8)
        return directions

    def bulk_volume_classification(
        self,
        price_changes: np.ndarray,
        volumes: np.ndarray,
        z_threshold: float = 0.0,
    ) -> np.ndarray:
        """Bulk Volume Classification (Easley et al. 2012).

        Classify using the sign of the standardised price change.

        Parameters
        ----------
        price_changes : array-like, shape (N,)
        volumes : array-like, shape (N,)
        z_threshold : float
            Threshold below which trades are classified as neutral (mapped to +1).

        Returns
        -------
        directions : ndarray of int8, shape (N,)
        """
        price_changes = np.asarray(price_changes, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        std = float(np.std(price_changes)) if len(price_changes) > 1 else 1.0
        if std == 0:
            std = 1.0
        z = price_changes / std
        directions = np.where(z > z_threshold, np.int8(1), np.int8(-1))
        return directions


# ---------------------------------------------------------------------------
# VWAP Engine
# ---------------------------------------------------------------------------


class VWAPEngine:
    """VWAP computation and scheduling tools."""

    def compute_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Compute Volume-Weighted Average Price.

        Parameters
        ----------
        prices : array-like, shape (N,)
        volumes : array-like, shape (N,)

        Returns
        -------
        float
        """
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        total_vol = float(np.sum(volumes))
        if total_vol == 0:
            return float(np.mean(prices))
        return float(np.dot(prices, volumes) / total_vol)

    def vwap_schedule(self, historical_volumes: np.ndarray, n_intervals: int) -> np.ndarray:
        """Compute participation weights for a VWAP order schedule.

        Resamples historical intraday volume profile to *n_intervals* and
        normalises to sum to 1.

        Parameters
        ----------
        historical_volumes : array-like, shape (M,)
            Historical intraday volume profile (any granularity).
        n_intervals : int
            Number of execution intervals.

        Returns
        -------
        weights : ndarray, shape (n_intervals,) — sum to 1.0
        """
        historical_volumes = np.asarray(historical_volumes, dtype=float)
        if len(historical_volumes) == 0 or float(np.sum(historical_volumes)) == 0:
            return np.full(n_intervals, 1.0 / n_intervals)
        # Resample by splitting into n_intervals buckets and summing
        indices = np.linspace(0, len(historical_volumes), n_intervals + 1, dtype=int)
        indices = np.clip(indices, 0, len(historical_volumes))
        weights = np.zeros(n_intervals)
        for i in range(n_intervals):
            start, end = indices[i], indices[i + 1]
            if start < end:
                weights[i] = float(np.sum(historical_volumes[start:end]))
            else:
                weights[i] = 0.0
        total = float(np.sum(weights))
        if total == 0:
            return np.full(n_intervals, 1.0 / n_intervals)
        return weights / total

    def tracking_error(
        self,
        execution_prices: np.ndarray,
        execution_sizes: np.ndarray,
        market_vwap: float,
    ) -> float:
        """VWAP tracking error in bps.

        Parameters
        ----------
        execution_prices : array-like, shape (N,)
        execution_sizes : array-like, shape (N,) — shares per interval
        market_vwap : float

        Returns
        -------
        float — bps deviation (positive = filled at worse price than market VWAP)
        """
        execution_prices = np.asarray(execution_prices, dtype=float)
        execution_sizes = np.asarray(execution_sizes, dtype=float)
        our_vwap = self.compute_vwap(execution_prices, execution_sizes)
        if market_vwap == 0:
            return 0.0
        return (our_vwap - market_vwap) / market_vwap * 10_000.0


# ---------------------------------------------------------------------------
# TWAP Engine
# ---------------------------------------------------------------------------


class TWAPEngine:
    """TWAP computation and scheduling tools."""

    def compute_twap(self, prices: np.ndarray) -> float:
        """Compute Time-Weighted Average Price — simple mean of interval prices.

        Parameters
        ----------
        prices : array-like, shape (N,)

        Returns
        -------
        float
        """
        prices = np.asarray(prices, dtype=float)
        return float(np.mean(prices))

    def twap_schedule(self, total_size: float, n_intervals: int) -> np.ndarray:
        """Uniform TWAP schedule — equal slices per interval.

        Parameters
        ----------
        total_size : float
        n_intervals : int

        Returns
        -------
        ndarray, shape (n_intervals,) — each element = total_size / n_intervals
        """
        if n_intervals <= 0:
            return np.array([])
        return np.full(n_intervals, total_size / n_intervals)

    def adaptive_twap(
        self,
        prices: np.ndarray,
        volatility: np.ndarray,
        total_size: float,
    ) -> np.ndarray:
        """Volatility-adjusted TWAP: trade more when volatility is lower.

        The inverse-volatility weighting ensures lower-vol intervals receive
        proportionally larger slice allocations.

        Parameters
        ----------
        prices : array-like, shape (N,) — not used directly but required for interface
        volatility : array-like, shape (N,) — per-interval volatility estimate
        total_size : float — total order size

        Returns
        -------
        slices : ndarray, shape (N,) — sums to total_size
        """
        volatility = np.asarray(volatility, dtype=float)
        n = len(volatility)
        if n == 0:
            return np.array([])
        # Avoid division by zero: replace zero-vol with small positive
        vol_safe = np.where(volatility <= 0, 1e-10, volatility)
        inv_vol = 1.0 / vol_safe
        weights = inv_vol / float(np.sum(inv_vol))
        return weights * total_size


# ---------------------------------------------------------------------------
# VPIN Calculator
# ---------------------------------------------------------------------------


class VPINCalculator:
    """Volume-synchronised Probability of Informed Trading (VPIN).

    Uses bulk volume classification on equally-sized volume buckets.
    """

    def compute(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bucket_size: float,
    ) -> float:
        """Compute scalar VPIN over the entire sample.

        Parameters
        ----------
        prices : array-like, shape (N,)
        volumes : array-like, shape (N,)
        bucket_size : float — target volume per bucket

        Returns
        -------
        float in [0, 1]
        """
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        price_changes = np.diff(prices, prepend=prices[0])
        classifier = TradeClassifier()
        directions = classifier.bulk_volume_classification(price_changes, volumes)
        buy_vol = volumes * (directions == 1).astype(float)
        sell_vol = volumes * (directions == -1).astype(float)

        return self._vpin_from_classified(buy_vol, sell_vol, bucket_size)

    def _vpin_from_classified(
        self,
        buy_vol: np.ndarray,
        sell_vol: np.ndarray,
        bucket_size: float,
    ) -> float:
        """Build volume buckets and compute VPIN."""
        total_vol = float(np.sum(buy_vol + sell_vol))
        if total_vol == 0 or bucket_size <= 0:
            return 0.0
        n_buckets = max(1, int(total_vol / bucket_size))
        # Accumulate into buckets
        cum_buy = float(np.sum(buy_vol))
        cum_sell = float(np.sum(sell_vol))
        # Simple approximation: distribute proportionally across buckets
        if n_buckets == 1:
            imbalance = abs(cum_buy - cum_sell)
            return float(min(1.0, imbalance / total_vol)) if total_vol > 0 else 0.0
        # Chunked bucket construction
        all_vol = buy_vol + sell_vol
        cum_all = np.cumsum(all_vol)
        bucket_edges = np.arange(1, n_buckets + 1) * bucket_size
        imbalances = []
        prev_idx = 0
        for edge in bucket_edges:
            idx = int(np.searchsorted(cum_all, edge, side="right"))
            idx = min(idx, len(all_vol))
            b_buy = float(np.sum(buy_vol[prev_idx:idx]))
            b_sell = float(np.sum(sell_vol[prev_idx:idx]))
            bvol = b_buy + b_sell
            if bvol > 0:
                imbalances.append(abs(b_buy - b_sell) / bvol)
            prev_idx = idx
        if not imbalances:
            return 0.0
        return float(np.mean(imbalances))

    def vpin_series(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bucket_size: float,
        window: int = 50,
    ) -> np.ndarray:
        """Compute a rolling VPIN series.

        Parameters
        ----------
        prices : array-like, shape (N,)
        volumes : array-like, shape (N,)
        bucket_size : float
        window : int — number of observations per rolling window

        Returns
        -------
        ndarray, shape (N,) — NaN for early periods without enough data
        """
        prices = np.asarray(prices, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        n = len(prices)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            p_win = prices[i - window + 1: i + 1]
            v_win = volumes[i - window + 1: i + 1]
            result[i] = self.compute(p_win, v_win, bucket_size)
        return result


# ---------------------------------------------------------------------------
# Futures Basis
# ---------------------------------------------------------------------------


class FuturesBasis:
    """Futures basis and roll analytics."""

    def basis(self, spot: float, future: float) -> float:
        """Basis = future - spot."""
        return future - spot

    def basis_bps(self, spot: float, future: float) -> float:
        """Basis in basis points = (future - spot) / spot * 10000."""
        if spot == 0:
            return 0.0
        return (future - spot) / spot * 10_000.0

    def roll_yield(
        self, near_future: float, far_future: float, days_to_roll: float
    ) -> float:
        """Annualised roll yield.

        roll_yield = (near / far - 1) * 365 / days_to_roll

        A positive roll yield indicates the market is in backwardation.

        Parameters
        ----------
        near_future : float
        far_future : float
        days_to_roll : float — calendar days to roll date

        Returns
        -------
        float — annualised rate (e.g. 0.05 = 5 %)
        """
        if far_future == 0 or days_to_roll <= 0:
            return 0.0
        return (near_future / far_future - 1.0) * 365.0 / days_to_roll

    def implied_repo(
        self,
        spot: float,
        future: float,
        T: float,
        r_domestic: float = 0.0,
    ) -> float:
        """Implied repo rate from cash-and-carry.

        F = S * exp((r_implied - q) * T)  →  r_implied = log(F/S) / T + q

        Parameters
        ----------
        spot : float
        future : float
        T : float — time to expiry in years
        r_domestic : float — dividend/convenience yield q

        Returns
        -------
        float — implied financing rate
        """
        if spot <= 0 or T <= 0:
            return 0.0
        return math.log(future / spot) / T + r_domestic


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def classify_trades_tick_rule(prices: np.ndarray) -> np.ndarray:
    """Classify trades using the tick rule.

    Parameters
    ----------
    prices : array-like, shape (N,)

    Returns
    -------
    ndarray of int8, shape (N,) — +1 buy, -1 sell
    """
    return TradeClassifier().tick_rule(prices)


def compute_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price.

    Parameters
    ----------
    prices : array-like
    volumes : array-like

    Returns
    -------
    float
    """
    return VWAPEngine().compute_vwap(prices, volumes)


def compute_vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: float = 1_000.0,
) -> float:
    """Compute scalar VPIN.

    Parameters
    ----------
    prices : array-like
    volumes : array-like
    bucket_size : float

    Returns
    -------
    float in [0, 1]
    """
    return VPINCalculator().compute(prices, volumes, bucket_size)


def kyle_lambda(
    price_changes: np.ndarray,
    signed_volume: np.ndarray,
) -> float:
    """Estimate Kyle's lambda via OLS regression: delta_p = lambda * signed_vol + eps.

    Parameters
    ----------
    price_changes : array-like, shape (N,)
    signed_volume : array-like, shape (N,) — positive for buys, negative for sells

    Returns
    -------
    float — price impact per unit of signed volume (lambda >= 0 expected)
    """
    price_changes = np.asarray(price_changes, dtype=float)
    signed_volume = np.asarray(signed_volume, dtype=float)
    if len(price_changes) < 2:
        return 0.0
    # OLS: beta = Cov(dp, sv) / Var(sv)
    var_sv = float(np.var(signed_volume))
    if var_sv == 0:
        return 0.0
    cov = float(np.cov(price_changes, signed_volume, ddof=1)[0, 1])
    return float(cov / var_sv)


def roll_yield(near_price: float, far_price: float, days_to_roll: float) -> float:
    """Annualised futures roll yield.

    Parameters
    ----------
    near_price : float
    far_price : float
    days_to_roll : float

    Returns
    -------
    float — annualised roll yield
    """
    return FuturesBasis().roll_yield(near_price, far_price, days_to_roll)
