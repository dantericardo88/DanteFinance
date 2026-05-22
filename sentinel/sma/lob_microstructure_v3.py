"""
Limit Order Book (LOB) microstructure analytics.
Pure numpy — zero network calls, zero external data deps.

dim_119 — LOB microstructure / market microstructure analytics (target: 9)

Classes
-------
OrderBookLevel
    Single price level in the order book (price, size).

LOBSnapshot
    Full order-book snapshot with properties: best_bid, best_ask, midpoint,
    quoted_spread, spread_bps, imbalance (OBI), weighted_mid.

LOBAnalytics
    Analytics over a time-series of LOBSnapshots:
    .time_weighted_spread()    → average quoted spread
    .average_imbalance()       → average OBI
    .depth_profile()           → volume by level
    .market_impact()           → bps cost to fill an order
    .book_resilience()         → proxy resilience metric

MicrostructureMetrics
    Classic microstructure estimators:
    .effective_spread()        → 2 * |trade_price - midpoint|
    .realized_spread()         → signed realized spread
    .price_impact()            → Kyle price impact
    .kyle_lambda()             → OLS price impact coefficient
    .amihud_illiquidity()      → |R| / Volume average
    .vpin()                    → volume-sync probability of informed trading
    .roll_spread()             → Roll (1984) serial-covariance estimator
    .hasbrouck_information_share() → simple 2-market IS approximation

LOBSimulator
    Synthetic LOB data generator for testing.

Module-level convenience functions
-----------------------------------
quoted_spread, order_book_imbalance, market_depth,
kyle_lambda, amihud_illiquidity, vpin, roll_spread
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OrderBookLevel:
    """A single price/size level in the order book."""

    price: float
    size: float


@dataclass
class LOBSnapshot:
    """Full order-book snapshot at a single point in time.

    Parameters
    ----------
    bids : list of OrderBookLevel, sorted *descending* by price
    asks : list of OrderBookLevel, sorted *ascending* by price
    timestamp : float
    """

    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> float:
        if not self.bids:
            return float("nan")
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        if not self.asks:
            return float("nan")
        return self.asks[0].price

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def quoted_spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """Quoted spread in basis points relative to midpoint."""
        mid = self.midpoint
        if mid == 0.0:
            return float("nan")
        return self.quoted_spread / mid * 10_000.0

    @property
    def imbalance(self) -> float:
        """Order Book Imbalance (OBI) in [-1, 1].

        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Positive means buy pressure.
        """
        bid_vol = sum(lvl.size for lvl in self.bids)
        ask_vol = sum(lvl.size for lvl in self.asks)
        total = bid_vol + ask_vol
        if total == 0.0:
            return 0.0
        return (bid_vol - ask_vol) / total

    @property
    def weighted_mid(self) -> float:
        """Weighted mid-price (size-weighted):

        weighted_mid = (ask_size * bid_price + bid_size * ask_price)
                       / (bid_size + ask_size)

        Uses best bid/ask sizes only (Lee-Ready style).
        """
        if not self.bids or not self.asks:
            return self.midpoint
        bid_sz = self.bids[0].size
        ask_sz = self.asks[0].size
        denom = bid_sz + ask_sz
        if denom == 0.0:
            return self.midpoint
        return (ask_sz * self.best_bid + bid_sz * self.best_ask) / denom


# ---------------------------------------------------------------------------
# LOB analytics over a snapshot series
# ---------------------------------------------------------------------------


class LOBAnalytics:
    """Compute analytics over a time-series of LOBSnapshots."""

    def __init__(self, snapshots: List[LOBSnapshot]) -> None:
        if not snapshots:
            raise ValueError("snapshots list must be non-empty")
        self.snapshots = snapshots

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def time_weighted_spread(self) -> float:
        """Average quoted spread across snapshots (time-weighted assumes
        uniform spacing if no explicit timestamps, otherwise weighted by
        the time delta to the next snapshot)."""
        snaps = self.snapshots
        n = len(snaps)
        if n == 1:
            return snaps[0].quoted_spread

        # If all timestamps are 0 (uniform), simple average
        timestamps = [s.timestamp for s in snaps]
        if all(t == 0.0 for t in timestamps):
            return float(np.mean([s.quoted_spread for s in snaps]))

        # Time-weighted: weight = duration held (last snapshot gets weight 0)
        total_w = 0.0
        total_ws = 0.0
        for i in range(n - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0:
                dt = 1.0
            total_ws += snaps[i].quoted_spread * dt
            total_w += dt
        # Include last snapshot with dt=1
        total_ws += snaps[-1].quoted_spread
        total_w += 1.0
        return total_ws / total_w

    # ------------------------------------------------------------------
    # Imbalance
    # ------------------------------------------------------------------

    def average_imbalance(self) -> float:
        """Average order book imbalance (OBI) across snapshots."""
        return float(np.mean([s.imbalance for s in self.snapshots]))

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------

    def depth_profile(self, n_levels: int = 5) -> Dict[str, List[float]]:
        """Aggregate bid and ask volumes by level, averaged across snapshots.

        Returns
        -------
        dict with 'bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes' —
        each a list of length min(n_levels, available).
        """
        bid_acc: Dict[int, List[float]] = {}
        ask_acc: Dict[int, List[float]] = {}
        bid_px_acc: Dict[int, List[float]] = {}
        ask_px_acc: Dict[int, List[float]] = {}

        for snap in self.snapshots:
            for lvl_i, lvl in enumerate(snap.bids[:n_levels]):
                bid_acc.setdefault(lvl_i, []).append(lvl.size)
                bid_px_acc.setdefault(lvl_i, []).append(lvl.price)
            for lvl_i, lvl in enumerate(snap.asks[:n_levels]):
                ask_acc.setdefault(lvl_i, []).append(lvl.size)
                ask_px_acc.setdefault(lvl_i, []).append(lvl.price)

        max_bid = max(bid_acc.keys(), default=-1) + 1
        max_ask = max(ask_acc.keys(), default=-1) + 1

        return {
            "bid_prices": [float(np.mean(bid_px_acc[i])) for i in range(max_bid)],
            "bid_sizes": [float(np.mean(bid_acc[i])) for i in range(max_bid)],
            "ask_prices": [float(np.mean(ask_px_acc[i])) for i in range(max_ask)],
            "ask_sizes": [float(np.mean(ask_acc[i])) for i in range(max_ask)],
        }

    # ------------------------------------------------------------------
    # Market impact
    # ------------------------------------------------------------------

    def market_impact(self, order_size: float, side: str = "buy") -> float:
        """Estimate bps cost to fill *order_size* shares against the last snapshot.

        Walks the book exhausting level by level until the order is filled.
        Returns bps moved from midpoint to volume-weighted fill price.
        """
        snap = self.snapshots[-1]
        mid = snap.midpoint
        if mid == 0.0:
            return 0.0

        levels = snap.asks if side.lower() == "buy" else snap.bids
        remaining = order_size
        total_cost = 0.0
        total_filled = 0.0

        for lvl in levels:
            if remaining <= 0:
                break
            fill = min(remaining, lvl.size)
            total_cost += fill * lvl.price
            total_filled += fill
            remaining -= fill

        if total_filled == 0:
            return 0.0

        vwap_fill = total_cost / total_filled
        if side.lower() == "buy":
            impact_bps = (vwap_fill - mid) / mid * 10_000.0
        else:
            impact_bps = (mid - vwap_fill) / mid * 10_000.0

        return max(0.0, impact_bps)

    # ------------------------------------------------------------------
    # Resilience
    # ------------------------------------------------------------------

    def book_resilience(self) -> float:
        """Simple resilience proxy: 1 / variance(spread) — higher means
        the spread is more stable (resilient).  Returns 0 if variance = 0."""
        spreads = np.array([s.quoted_spread for s in self.snapshots])
        var = float(np.var(spreads))
        if var == 0.0:
            return float("inf")
        return 1.0 / var


# ---------------------------------------------------------------------------
# Classic microstructure estimators
# ---------------------------------------------------------------------------


class MicrostructureMetrics:
    """Classic market microstructure estimators, all pure-numpy."""

    # ------------------------------------------------------------------
    # Spread decomposition
    # ------------------------------------------------------------------

    def effective_spread(
        self,
        trade_prices: np.ndarray,
        midpoints: np.ndarray,
        directions: np.ndarray,
    ) -> float:
        """Effective spread = 2 * |trade_price - midpoint|, averaged.

        directions: +1 buyer-initiated, -1 seller-initiated (used for sign,
        here we take absolute value so directions are not strictly needed).
        """
        trade_prices = np.asarray(trade_prices, dtype=float)
        midpoints = np.asarray(midpoints, dtype=float)
        return float(np.mean(2.0 * np.abs(trade_prices - midpoints)))

    def realized_spread(
        self,
        trade_prices: np.ndarray,
        midpoints: np.ndarray,
        future_midpoints: np.ndarray,
        directions: np.ndarray,
    ) -> float:
        """Realized spread = 2 * d * (trade_price - future_midpoint).

        Measures the spread component retained by the dealer.
        """
        d = np.asarray(directions, dtype=float)
        tp = np.asarray(trade_prices, dtype=float)
        fmid = np.asarray(future_midpoints, dtype=float)
        return float(np.mean(2.0 * d * (tp - fmid)))

    def price_impact(
        self,
        midpoints: np.ndarray,
        directions: np.ndarray,
    ) -> float:
        """Price impact = d * (future_midpoint - current_midpoint).

        Uses consecutive midpoints: future = midpoint[i+1], current = midpoint[i].
        """
        mids = np.asarray(midpoints, dtype=float)
        d = np.asarray(directions, dtype=float)
        n = min(len(mids) - 1, len(d))
        if n <= 0:
            return 0.0
        delta_mid = mids[1 : n + 1] - mids[:n]
        return float(np.mean(d[:n] * delta_mid))

    # ------------------------------------------------------------------
    # Kyle's Lambda
    # ------------------------------------------------------------------

    def kyle_lambda(
        self,
        price_changes: np.ndarray,
        signed_volumes: np.ndarray,
    ) -> float:
        """Kyle (1985) price-impact coefficient via OLS.

        delta_p = lambda * signed_volume + epsilon
        lambda = Cov(delta_p, signed_vol) / Var(signed_vol)
        """
        return kyle_lambda(price_changes, signed_volumes)

    # ------------------------------------------------------------------
    # Amihud illiquidity
    # ------------------------------------------------------------------

    def amihud_illiquidity(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """Amihud (2002) illiquidity ratio.

        ILLIQ = mean(|R_t| / Volume_t)
        """
        return amihud_illiquidity(returns, volumes)

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------

    def vpin(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bucket_size: float = 1000.0,
    ) -> float:
        """VPIN (Easley et al. 2012): volume-synchronised PIN estimate."""
        return vpin(prices, volumes, bucket_size)

    # ------------------------------------------------------------------
    # Roll spread
    # ------------------------------------------------------------------

    def roll_spread(self, prices: np.ndarray) -> float:
        """Roll (1984) implicit spread estimator.

        spread = 2 * sqrt(-Cov(delta_p_t, delta_p_{t-1}))  if Cov < 0
        """
        return roll_spread(prices)

    # ------------------------------------------------------------------
    # Hasbrouck information share
    # ------------------------------------------------------------------

    def hasbrouck_information_share(
        self,
        price_series_1: np.ndarray,
        price_series_2: np.ndarray,
    ) -> float:
        """Simplified 2-market Hasbrouck information share for market 1.

        Full Hasbrouck (1995) requires a VECM; this provides an OLS-based
        approximation used in practice when a full VAR/VECM is unavailable:

        IS_1 = sigma_1^2 / (sigma_1^2 + sigma_2^2)

        where sigma_i is the std of the innovation in the price series,
        computed as the std of the first-difference residuals.

        Returns a value in (0, 1). Returns 0.5 if series are identical.
        """
        p1 = np.asarray(price_series_1, dtype=float)
        p2 = np.asarray(price_series_2, dtype=float)
        # Innovations ≈ first differences
        d1 = np.diff(p1)
        d2 = np.diff(p2)
        var1 = float(np.var(d1))
        var2 = float(np.var(d2))
        total = var1 + var2
        if total == 0.0:
            return 0.5
        return var1 / total


# ---------------------------------------------------------------------------
# LOB Simulator
# ---------------------------------------------------------------------------


class LOBSimulator:
    """Generate synthetic LOB snapshots for testing and benchmarking.

    Prices follow a random walk around *mid_price*.  Each snapshot has
    *depth_levels* levels on each side with exponentially increasing size.
    """

    def __init__(
        self,
        mid_price: float = 100.0,
        spread_bps: float = 5.0,
        depth_levels: int = 5,
        seed: int = 42,
    ) -> None:
        self.mid_price = mid_price
        self.spread_bps = spread_bps
        self.depth_levels = depth_levels
        self.rng = np.random.default_rng(seed)
        self._current_mid = mid_price

    def _half_spread(self) -> float:
        return self.mid_price * self.spread_bps / 2.0 / 10_000.0

    def generate_snapshot(self) -> LOBSnapshot:
        """Generate one LOB snapshot and advance the mid-price."""
        # Random walk step
        vol = self.mid_price * 5.0 / 10_000.0  # 0.5 bps vol per step
        self._current_mid += self.rng.normal(0, vol)
        mid = self._current_mid

        # Half-spread (add small noise)
        base_hs = self.mid_price * self.spread_bps / 2.0 / 10_000.0
        hs = max(base_hs * (1.0 + 0.2 * self.rng.standard_normal()), 1e-6)

        bid_price = mid - hs
        ask_price = mid + hs

        # Build levels with increasing offsets and exponentially growing sizes
        tick = hs * 0.5  # level spacing
        bids = []
        asks = []
        for i in range(self.depth_levels):
            size_b = 100.0 * (1.5 ** i) * (1.0 + 0.3 * abs(self.rng.standard_normal()))
            size_a = 100.0 * (1.5 ** i) * (1.0 + 0.3 * abs(self.rng.standard_normal()))
            bids.append(OrderBookLevel(bid_price - i * tick, round(size_b, 2)))
            asks.append(OrderBookLevel(ask_price + i * tick, round(size_a, 2)))

        return LOBSnapshot(bids=bids, asks=asks, timestamp=float(self.rng.uniform(0, 1)))

    def generate_sequence(self, n: int = 100) -> List[LOBSnapshot]:
        """Generate a time-ordered sequence of n LOB snapshots."""
        t = 0.0
        snapshots = []
        for _ in range(n):
            snap = self.generate_snapshot()
            snap.timestamp = t
            t += 1.0
            snapshots.append(snap)
        return snapshots

    def simulate_trade_flow(self, n_trades: int = 500) -> Dict[str, np.ndarray]:
        """Simulate a realistic trade flow.

        Returns
        -------
        dict with:
          'prices'     : trade prices (n_trades,)
          'volumes'    : trade sizes  (n_trades,)
          'directions' : +1 buy / -1 sell (n_trades,)
          'midpoints'  : midpoint at trade time (n_trades,)
        """
        prices = np.empty(n_trades)
        volumes = np.empty(n_trades)
        directions = np.empty(n_trades, dtype=int)
        midpoints = np.empty(n_trades)

        # Reset state
        self._current_mid = self.mid_price
        hs = self._half_spread()
        vol_mid = self.mid_price * 5.0 / 10_000.0

        for i in range(n_trades):
            # Advance midpoint
            self._current_mid += self.rng.normal(0, vol_mid)
            mid = self._current_mid
            midpoints[i] = mid

            # Direction: biased by recent OBI proxy (random here)
            d = 1 if self.rng.random() > 0.5 else -1
            directions[i] = d

            # Trade price = midpoint +/- half-spread +/- noise
            noise = self.rng.uniform(0, hs * 0.5)
            if d == 1:
                prices[i] = mid + hs * 0.5 + noise
            else:
                prices[i] = mid - hs * 0.5 - noise

            # Volume: log-normal
            volumes[i] = self.rng.lognormal(mean=5.0, sigma=1.0)

        return {
            "prices": prices,
            "volumes": volumes,
            "directions": directions.astype(float),
            "midpoints": midpoints,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def quoted_spread(lob: LOBSnapshot) -> float:
    """Quoted bid-ask spread for a single LOBSnapshot."""
    return lob.quoted_spread


def order_book_imbalance(lob: LOBSnapshot) -> float:
    """Order Book Imbalance (OBI) for a single LOBSnapshot."""
    return lob.imbalance


def market_depth(lob: LOBSnapshot, bps: float = 10.0) -> Dict[str, float]:
    """Total volume within *bps* basis points of midpoint on each side.

    Returns
    -------
    dict: {'bid_depth': float, 'ask_depth': float, 'total_depth': float}
    """
    mid = lob.midpoint
    threshold = mid * bps / 10_000.0

    bid_depth = sum(
        lvl.size for lvl in lob.bids if (mid - lvl.price) <= threshold
    )
    ask_depth = sum(
        lvl.size for lvl in lob.asks if (lvl.price - mid) <= threshold
    )
    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": bid_depth + ask_depth,
    }


def kyle_lambda(
    price_changes: np.ndarray,
    signed_volumes: np.ndarray,
) -> float:
    """Kyle (1985) Lambda — OLS price impact coefficient.

    lambda = Cov(delta_p, signed_vol) / Var(signed_vol)

    Returns 0.0 if variance of signed volume is zero.
    """
    dp = np.asarray(price_changes, dtype=float).ravel()
    sv = np.asarray(signed_volumes, dtype=float).ravel()
    n = min(len(dp), len(sv))
    if n < 2:
        return 0.0
    dp, sv = dp[:n], sv[:n]

    # Remove NaN
    mask = np.isfinite(dp) & np.isfinite(sv)
    dp, sv = dp[mask], sv[mask]
    if len(dp) < 2:
        return 0.0

    var_sv = float(np.var(sv))
    if var_sv == 0.0:
        return 0.0
    cov = float(np.cov(dp, sv)[0, 1])
    return cov / var_sv


def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
) -> float:
    """Amihud (2002) illiquidity ratio.

    ILLIQ = mean(|R_t| / Volume_t)

    Volumes of zero are skipped. Returns 0.0 if all volumes are zero.
    """
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(volumes, dtype=float).ravel()
    n = min(len(r), len(v))
    r, v = r[:n], v[:n]

    mask = np.isfinite(r) & np.isfinite(v) & (v > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(r[mask]) / v[mask]))


def vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: float = 1000.0,
) -> float:
    """Volume-Synchronized Probability of Informed Trading (VPIN).

    Algorithm (Easley, Lopez de Prado & O'Hara, 2012):
    1. Classify each trade as buy/sell using the tick rule:
       price_t > price_{t-1} → buy; price_t < price_{t-1} → sell;
       price_t == price_{t-1} → same as previous.
    2. Aggregate into equal-volume buckets of size *bucket_size*.
    3. VPIN = mean(|V_buy - V_sell|) / bucket_size.

    Returns a value in [0, 1].  Returns 0.5 on insufficient data.
    """
    px = np.asarray(prices, dtype=float).ravel()
    vol = np.asarray(volumes, dtype=float).ravel()
    n = min(len(px), len(vol))
    if n < 2:
        return 0.5

    px, vol = px[:n], vol[:n]

    # Tick-rule classification
    dp = np.diff(px)
    directions = np.zeros(n, dtype=float)
    last_dir = 1.0
    for i in range(1, n):
        if dp[i - 1] > 0:
            directions[i] = 1.0
            last_dir = 1.0
        elif dp[i - 1] < 0:
            directions[i] = -1.0
            last_dir = -1.0
        else:
            directions[i] = last_dir
    directions[0] = directions[1] if n > 1 else 1.0

    buy_vol = vol * (directions > 0).astype(float)
    sell_vol = vol * (directions < 0).astype(float)

    if bucket_size <= 0:
        bucket_size = float(np.sum(vol)) / max(10, n // 10)

    # Fill buckets
    buckets_buy: List[float] = []
    buckets_sell: List[float] = []
    cum_buy = 0.0
    cum_sell = 0.0
    cum_total = 0.0

    for i in range(n):
        cum_buy += buy_vol[i]
        cum_sell += sell_vol[i]
        cum_total += vol[i]

        while cum_total >= bucket_size:
            # Proportion filled by this bucket
            frac = bucket_size / cum_total
            buckets_buy.append(cum_buy * frac)
            buckets_sell.append(cum_sell * frac)
            cum_buy *= 1.0 - frac
            cum_sell *= 1.0 - frac
            cum_total -= bucket_size

    if not buckets_buy:
        return 0.5

    imbalances = np.abs(
        np.array(buckets_buy) - np.array(buckets_sell)
    )
    vpin_val = float(np.mean(imbalances)) / bucket_size
    # Clip to [0, 1] for safety (numerical precision)
    return float(np.clip(vpin_val, 0.0, 1.0))


def roll_spread(prices: np.ndarray) -> float:
    """Roll (1984) implicit spread estimator.

    spread = 2 * sqrt(-Cov(delta_p_t, delta_p_{t-1}))

    If the serial covariance is non-negative (prices are not mean-reverting),
    returns 0.0 (spread is indeterminate from this estimator alone).
    """
    px = np.asarray(prices, dtype=float).ravel()
    if len(px) < 3:
        return 0.0
    dp = np.diff(px)
    if len(dp) < 2:
        return 0.0
    cov = float(np.cov(dp[:-1], dp[1:])[0, 1])
    if cov >= 0:
        return 0.0
    return 2.0 * math.sqrt(-cov)
