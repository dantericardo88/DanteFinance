"""
Transaction Cost Analysis (TCA) — full implementation shortfall suite.
Pure numpy — zero network calls, zero external data deps.

dim_120 — Transaction cost analysis (TCA / implementation shortfall) (target: 9)

Classes
-------
Order
    Single-order container with all fields required for IS decomposition.

ISDecomposition
    IS breakdown: delay, execution, opportunity, commissions — all in bps.

MarketImpactEstimate
    Square-root model impact decomposition: permanent, temporary, spread, timing.

ImplementationShortfallAnalyzer
    .decompose()         → ISDecomposition for one order
    .batch_decompose()   → list of ISDecomposition
    .aggregate_stats()   → dict of summary statistics

BenchmarkAnalyzer
    .vwap_slippage()             → bps vs VWAP benchmark
    .twap_slippage()             → bps vs TWAP benchmark
    .arrival_slippage()          → bps vs arrival price
    .participation_weighted_price() → PWP float

MarketImpactModel
    Square-root market impact model (Almgren et al.).
    .estimate()                    → MarketImpactEstimate
    .optimal_participation_rate()  → h* = sqrt(lam / (2*eta*sigma^2))

BrokerAnalytics
    .by_broker()     → dict broker → metrics
    .rank_brokers()  → sorted list of (broker, score)
    .best_broker()   → name of best broker

TCAReport
    .summary()            → aggregate stats dict
    .by_direction()       → buy/sell split
    .time_of_day_analysis() → bucketed metrics by hour

Module-level convenience functions
-----------------------------------
compute_is, vwap, twap, market_impact
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """Single execution order with all data needed for TCA."""

    order_id: str
    direction: int          # +1 buy, -1 sell
    target_shares: float
    decision_price: float
    arrival_price: float
    avg_fill_price: float
    shares_filled: float
    close_price: float
    commission_per_share: float = 0.01
    broker: str = "default"
    timestamp: float = 0.0  # unix seconds or hour-of-day for ToD analysis

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")
        if self.target_shares <= 0:
            raise ValueError("target_shares must be positive")
        if self.shares_filled < 0:
            raise ValueError("shares_filled must be non-negative")


@dataclass
class ISDecomposition:
    """Implementation Shortfall decomposition in basis points.

    Attributes
    ----------
    delay_cost_bps : float
        Cost of delay from decision to arrival.
    execution_cost_bps : float
        Cost of execution relative to arrival price.
    opportunity_cost_bps : float
        Cost of unfilled portion, measured vs close.
    commission_bps : float
        Commissions paid in bps.
    total_is_bps : float
        Sum of all components.
    fill_rate : float
        shares_filled / target_shares.
    """

    delay_cost_bps: float
    execution_cost_bps: float
    opportunity_cost_bps: float
    commission_bps: float
    total_is_bps: float
    fill_rate: float

    def is_efficient(self, threshold_bps: float = 10.0) -> bool:
        """Return True if total IS is within *threshold_bps* of zero."""
        return abs(self.total_is_bps) <= threshold_bps

    def as_dict(self) -> Dict[str, float]:
        return {
            "delay_cost_bps": self.delay_cost_bps,
            "execution_cost_bps": self.execution_cost_bps,
            "opportunity_cost_bps": self.opportunity_cost_bps,
            "commission_bps": self.commission_bps,
            "total_is_bps": self.total_is_bps,
            "fill_rate": self.fill_rate,
        }


@dataclass
class MarketImpactEstimate:
    """Square-root model market impact decomposition in basis points."""

    permanent_bps: float
    temporary_bps: float
    spread_bps: float
    timing_risk_bps: float
    total_bps: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "permanent_bps": self.permanent_bps,
            "temporary_bps": self.temporary_bps,
            "spread_bps": self.spread_bps,
            "timing_risk_bps": self.timing_risk_bps,
            "total_bps": self.total_bps,
        }


# ---------------------------------------------------------------------------
# Implementation Shortfall Analyzer
# ---------------------------------------------------------------------------


class ImplementationShortfallAnalyzer:
    """Decompose execution orders into IS components.

    IS = delay_cost + execution_cost + opportunity_cost + commissions

    All quantities normalised by (decision_price * target_shares) → bps.
    """

    def decompose(self, order: Order) -> ISDecomposition:
        """Decompose a single order into IS components."""
        dp = order.decision_price
        ap = order.arrival_price
        fp = order.avg_fill_price
        cp = order.close_price
        d = order.direction
        q_filled = order.shares_filled
        q_target = order.target_shares
        q_unfilled = q_target - q_filled
        notional = dp * q_target  # normalisation base

        # Delay cost: (arrival - decision) * direction * filled
        delay_cost = (ap - dp) * d * q_filled
        # Execution cost: (fill - arrival) * direction * filled
        execution_cost = (fp - ap) * d * q_filled
        # Opportunity cost: (close - decision) * direction * unfilled
        opportunity_cost = (cp - dp) * d * q_unfilled
        # Commissions: always positive drag
        commission = order.commission_per_share * q_filled

        total_is = delay_cost + execution_cost + opportunity_cost + commission

        scale = 10_000.0 / notional if notional != 0 else 0.0

        fill_rate = q_filled / q_target if q_target != 0 else 0.0

        return ISDecomposition(
            delay_cost_bps=delay_cost * scale,
            execution_cost_bps=execution_cost * scale,
            opportunity_cost_bps=opportunity_cost * scale,
            commission_bps=commission * scale,
            total_is_bps=total_is * scale,
            fill_rate=fill_rate,
        )

    def batch_decompose(self, orders: List[Order]) -> List[ISDecomposition]:
        """Decompose a list of orders."""
        return [self.decompose(o) for o in orders]

    def aggregate_stats(self, decompositions: List[ISDecomposition]) -> dict:
        """Compute summary statistics across a batch of decompositions."""
        if not decompositions:
            return {}
        total_is = [d.total_is_bps for d in decompositions]
        fill_rates = [d.fill_rate for d in decompositions]
        delays = [d.delay_cost_bps for d in decompositions]
        execs = [d.execution_cost_bps for d in decompositions]
        opps = [d.opportunity_cost_bps for d in decompositions]
        commissions = [d.commission_bps for d in decompositions]
        return {
            "count": len(decompositions),
            "avg_total_is_bps": float(np.mean(total_is)),
            "std_total_is_bps": float(np.std(total_is, ddof=1)) if len(total_is) > 1 else 0.0,
            "avg_fill_rate": float(np.mean(fill_rates)),
            "avg_delay_cost_bps": float(np.mean(delays)),
            "avg_execution_cost_bps": float(np.mean(execs)),
            "avg_opportunity_cost_bps": float(np.mean(opps)),
            "avg_commission_bps": float(np.mean(commissions)),
            "efficiency_rate": float(
                sum(1 for d in decompositions if d.is_efficient()) / len(decompositions)
            ),
        }


# ---------------------------------------------------------------------------
# Benchmark Analyzer
# ---------------------------------------------------------------------------


class BenchmarkAnalyzer:
    """Compute execution quality relative to standard benchmarks."""

    def vwap_slippage(
        self,
        order: Order,
        market_prices: np.ndarray,
        market_volumes: np.ndarray,
    ) -> float:
        """Slippage vs VWAP in bps.

        Positive = fill was worse than VWAP (bought above / sold below).
        """
        vwap_price = vwap(market_prices, market_volumes)
        return (order.avg_fill_price - vwap_price) * order.direction / vwap_price * 10_000.0

    def twap_slippage(self, order: Order, interval_prices: np.ndarray) -> float:
        """Slippage vs TWAP in bps."""
        twap_price = twap(interval_prices)
        return (order.avg_fill_price - twap_price) * order.direction / twap_price * 10_000.0

    def arrival_slippage(self, order: Order) -> float:
        """Slippage vs arrival price in bps."""
        ap = order.arrival_price
        return (order.avg_fill_price - ap) * order.direction / ap * 10_000.0

    def participation_weighted_price(
        self,
        order: Order,
        market_volumes: np.ndarray,
        fill_volumes: np.ndarray,
    ) -> float:
        """Participation Weighted Price (PWP): market VWAP over the participation window.

        Weights market prices by fill volume distribution.
        Returns price in same units as order prices.
        """
        if market_volumes.shape != fill_volumes.shape:
            raise ValueError("market_volumes and fill_volumes must have the same shape")
        total_fill = float(np.sum(fill_volumes))
        if total_fill == 0:
            return float(order.arrival_price)
        weights = fill_volumes / total_fill
        # PWP uses market mid-prices weighted by order's own fill pattern
        # Here market_volumes proxy for market mid-prices bins; caller passes appropriate data
        return float(np.sum(weights * market_volumes))


# ---------------------------------------------------------------------------
# Market Impact Model (square-root)
# ---------------------------------------------------------------------------


class MarketImpactModel:
    """Almgren (2005) square-root market impact model.

    Parameters
    ----------
    gamma : float
        Permanent impact coefficient (~0.5 typical).
    eta : float
        Temporary impact coefficient (~0.1 typical).
    """

    def __init__(self, gamma: float = 0.5, eta: float = 0.1) -> None:
        self.gamma = gamma
        self.eta = eta

    def estimate(
        self,
        order_size: float,
        adv: float,
        sigma: float,
        bid_ask_spread_pct: float,
        T_days: float,
    ) -> MarketImpactEstimate:
        """Estimate market impact components.

        Parameters
        ----------
        order_size : float
            Order size in shares.
        adv : float
            Average daily volume in shares.
        sigma : float
            Daily return volatility (e.g. 0.02 = 2 %).
        bid_ask_spread_pct : float
            Bid-ask spread as fraction of price (e.g. 0.0005 = 5 bps).
        T_days : float
            Execution horizon in trading days.

        Returns
        -------
        MarketImpactEstimate in bps.
        """
        # Participation rate Q/ADV
        pov = order_size / adv if adv > 0 else 0.0
        # Single-trade size for temporary: order_size / n_trades where n_trades~T
        n_trades = max(T_days, 1.0)
        q_per_trade = order_size / n_trades
        adv_daily = adv

        # Square-root permanent impact (in bps)
        permanent_bps = self.gamma * sigma * math.sqrt(pov) * 10_000.0
        # Square-root temporary impact per trade (in bps)
        temporary_bps = self.eta * sigma * math.sqrt(q_per_trade / adv_daily) * 10_000.0
        # Spread cost (one-way): half spread in bps
        spread_bps = 0.5 * bid_ask_spread_pct * 10_000.0
        # Timing risk: sigma * sqrt(T) in bps
        timing_risk_bps = sigma * math.sqrt(T_days) * 10_000.0

        total_bps = permanent_bps + temporary_bps + spread_bps

        return MarketImpactEstimate(
            permanent_bps=permanent_bps,
            temporary_bps=temporary_bps,
            spread_bps=spread_bps,
            timing_risk_bps=timing_risk_bps,
            total_bps=total_bps,
        )

    def optimal_participation_rate(self, sigma: float, lam_risk: float) -> float:
        """Optimal participation rate from risk-aversion trade-off.

        h* = sqrt(lam_risk / (2 * eta * sigma^2))

        Parameters
        ----------
        sigma : float
            Volatility.
        lam_risk : float
            Risk aversion coefficient.

        Returns
        -------
        float
            Participation rate h* (fraction of ADV per day).
        """
        denom = 2.0 * self.eta * sigma ** 2
        if denom <= 0:
            return 0.0
        return float(math.sqrt(lam_risk / denom))


# ---------------------------------------------------------------------------
# Broker Analytics
# ---------------------------------------------------------------------------


class BrokerAnalytics:
    """Analyse execution quality by broker.

    Parameters
    ----------
    orders : list of Order
    """

    def __init__(self, orders: List[Order]) -> None:
        self.orders = orders
        self._analyzer = ImplementationShortfallAnalyzer()
        self._decomps: Dict[str, List[ISDecomposition]] = {}
        for o in orders:
            d = self._analyzer.decompose(o)
            self._decomps.setdefault(o.broker, []).append(d)

    def by_broker(self) -> Dict[str, dict]:
        """Return per-broker aggregate metrics."""
        result: Dict[str, dict] = {}
        for broker, decomps in self._decomps.items():
            stats = self._analyzer.aggregate_stats(decomps)
            result[broker] = stats
        return result

    def rank_brokers(self, metric: str = "total_is_bps") -> List[Tuple[str, float]]:
        """Rank brokers by *metric* (ascending = lower IS is better).

        Returns
        -------
        list of (broker_name, metric_value) sorted ascending.
        """
        stats = self.by_broker()
        key_map = {
            "total_is_bps": "avg_total_is_bps",
            "fill_rate": "avg_fill_rate",
            "commission": "avg_commission_bps",
        }
        stat_key = key_map.get(metric, f"avg_{metric}")
        rows = []
        for broker, s in stats.items():
            val = s.get(stat_key, s.get(metric, float("inf")))
            rows.append((broker, float(val)))
        # Lower IS = better → ascending
        reverse = metric == "fill_rate"
        rows.sort(key=lambda x: x[1], reverse=reverse)
        return rows

    def best_broker(self, metric: str = "total_is_bps") -> str:
        """Return name of the best (lowest IS or highest fill rate) broker."""
        ranked = self.rank_brokers(metric)
        if not ranked:
            return ""
        return ranked[0][0]


# ---------------------------------------------------------------------------
# TCA Report
# ---------------------------------------------------------------------------


class TCAReport:
    """Full TCA report over a basket of orders.

    Parameters
    ----------
    orders : list of Order
    adv : float
        Average daily volume (for market impact estimation).
    sigma : float
        Daily return volatility.
    """

    def __init__(
        self,
        orders: List[Order],
        adv: float = 1_000_000.0,
        sigma: float = 0.02,
    ) -> None:
        self.orders = orders
        self.adv = adv
        self.sigma = sigma
        self._analyzer = ImplementationShortfallAnalyzer()
        self._decomps = self._analyzer.batch_decompose(orders)

    def summary(self) -> dict:
        """High-level aggregate statistics."""
        return self._analyzer.aggregate_stats(self._decomps)

    def by_direction(self) -> dict:
        """Split summary statistics by buy vs sell."""
        buy_orders = [o for o in self.orders if o.direction == 1]
        sell_orders = [o for o in self.orders if o.direction == -1]
        buy_decomps = self._analyzer.batch_decompose(buy_orders)
        sell_decomps = self._analyzer.batch_decompose(sell_orders)
        return {
            "buy": self._analyzer.aggregate_stats(buy_decomps),
            "sell": self._analyzer.aggregate_stats(sell_decomps),
        }

    def time_of_day_analysis(self) -> dict:
        """Bucket orders by hour-of-day (timestamp % 86400 // 3600).

        Returns dict mapping hour (int) → aggregate stats.
        """
        buckets: Dict[int, List[ISDecomposition]] = {}
        for o, d in zip(self.orders, self._decomps):
            hour = int(o.timestamp % 86400 // 3600)
            buckets.setdefault(hour, []).append(d)
        return {
            hour: self._analyzer.aggregate_stats(ds)
            for hour, ds in sorted(buckets.items())
        }

    def broker_report(self) -> dict:
        """Per-broker breakdown."""
        return BrokerAnalytics(self.orders).by_broker()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def compute_is(order: Order) -> ISDecomposition:
    """Decompose a single order into IS components."""
    return ImplementationShortfallAnalyzer().decompose(order)


def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price.

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


def twap(prices: np.ndarray) -> float:
    """Time-Weighted Average Price — simple mean of interval mid-prices.

    Parameters
    ----------
    prices : array-like, shape (N,)

    Returns
    -------
    float
    """
    prices = np.asarray(prices, dtype=float)
    return float(np.mean(prices))


def market_impact(
    order_size: float,
    adv: float,
    sigma: float,
    spread_pct: float,
    T_days: float,
    gamma: float = 0.5,
    eta: float = 0.1,
) -> MarketImpactEstimate:
    """Estimate market impact using the square-root model.

    Parameters
    ----------
    order_size : float
        Order size in shares.
    adv : float
        Average daily volume.
    sigma : float
        Daily return volatility.
    spread_pct : float
        Bid-ask spread as fraction of price.
    T_days : float
        Execution horizon in days.
    gamma : float
        Permanent impact coefficient.
    eta : float
        Temporary impact coefficient.

    Returns
    -------
    MarketImpactEstimate
    """
    model = MarketImpactModel(gamma=gamma, eta=eta)
    return model.estimate(order_size, adv, sigma, spread_pct, T_days)
