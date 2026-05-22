"""
Order book / Level 2 market depth analytics.
Pure numpy — zero network calls, zero external data deps.

dim_010 — Order book depth analytics (target: 9)

Focuses on depth-aggregation, visualization metrics, and depth-based signals.
Complements dim_119 (LOB microstructure / Kyle-lambda) which covers trade-flow
microstructure; this module covers multi-level depth-weighted pricing, cumulative
depth profiles, price impact via book-walking, liquidity maps, and wall detection.

Classes
-------
OrderBookLevel
    Single price level: price, size, n_orders.

OrderBook
    Full snapshot with bids (desc) and asks (asc).
    Properties: best_bid, best_ask, spread, midpoint,
                total_bid_depth, total_ask_depth.

DepthAnalytics
    Rich analytics over a single OrderBook snapshot.
    .bid_vwap(levels)             → volume-weighted avg bid price (top N)
    .ask_vwap(levels)             → volume-weighted avg ask price (top N)
    .depth_weighted_mid(levels)   → (bid_vwap + ask_vwap) / 2
    .price_impact_buy(qty)        → bps cost to buy Q shares walking the book
    .price_impact_sell(qty)       → bps proceeds shortfall selling Q shares
    .cumulative_bid_depth(n)      → (prices, cum_sizes) arrays
    .cumulative_ask_depth(n)      → (prices, cum_sizes) arrays
    .book_skew(levels)            → bid_depth_pct - ask_depth_pct
    .aggregate_imbalance(levels)  → proximity-weighted imbalance ∈ [-1, 1]
    .bid_wall(threshold)          → price of largest bid wall or None
    .ask_wall(threshold)          → price of lowest ask wall or None
    .depth_map(n_levels)          → dict with bid/ask prices, sizes, cum depths

OrderBookSimulator
    Synthetic order book generator.
    .generate(spread_bps)           → balanced OrderBook
    .generate_skewed(buy_pressure)  → skewed OrderBook (buy_pressure > 0 = more bids)

DepthSignalGenerator
    Signal generation from depth data.
    .imbalance_signal(book, levels)          → float ∈ [-1, 1]
    .wall_signal(book)                       → dict with bid/ask wall info
    .impact_cost_signal(book, trade_size)    → float (bps)

Module-level convenience functions
-----------------------------------
depth_weighted_mid, price_impact_bps, book_imbalance, detect_walls
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
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    size: float
    n_orders: int = 1

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")


@dataclass
class OrderBook:
    """
    Full order book snapshot.

    Parameters
    ----------
    bids : list of OrderBookLevel, sorted descending by price
    asks : list of OrderBookLevel, sorted ascending by price
    timestamp : float, unix epoch seconds (default 0.0)
    """
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.bids:
            raise ValueError("bids must not be empty")
        if not self.asks:
            raise ValueError("asks must not be empty")
        # Ensure sorted
        self.bids = sorted(self.bids, key=lambda l: -l.price)
        self.asks = sorted(self.asks, key=lambda l: l.price)

    @property
    def best_bid(self) -> float:
        """Highest bid price."""
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        """Lowest ask price."""
        return self.asks[0].price

    @property
    def spread(self) -> float:
        """Quoted spread = best_ask - best_bid."""
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float:
        """Simple midpoint = (best_bid + best_ask) / 2."""
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def total_bid_depth(self) -> float:
        """Total shares available on the bid side."""
        return sum(level.size for level in self.bids)

    @property
    def total_ask_depth(self) -> float:
        """Total shares available on the ask side."""
        return sum(level.size for level in self.asks)


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

class DepthAnalytics:
    """
    Rich depth analytics for a single OrderBook snapshot.

    All methods are pure — they do not mutate the book.
    """

    def __init__(self, book: OrderBook) -> None:
        self.book = book

    # ------------------------------------------------------------------
    # Volume-weighted pricing
    # ------------------------------------------------------------------

    def bid_vwap(self, levels: int = 5) -> float:
        """
        Volume-weighted average price of the top *levels* bid levels.

        bid_vwap = sum(price_i * size_i) / sum(size_i)
        """
        top = self.book.bids[:levels]
        total_size = sum(l.size for l in top)
        if total_size == 0:
            return self.book.best_bid
        return sum(l.price * l.size for l in top) / total_size

    def ask_vwap(self, levels: int = 5) -> float:
        """
        Volume-weighted average price of the top *levels* ask levels.

        ask_vwap = sum(price_i * size_i) / sum(size_i)
        """
        top = self.book.asks[:levels]
        total_size = sum(l.size for l in top)
        if total_size == 0:
            return self.book.best_ask
        return sum(l.price * l.size for l in top) / total_size

    def depth_weighted_mid(self, levels: int = 5) -> float:
        """
        Depth-weighted mid-price.

        depth_mid = (bid_vwap + ask_vwap) / 2
        """
        return (self.bid_vwap(levels) + self.ask_vwap(levels)) / 2.0

    # ------------------------------------------------------------------
    # Price impact (book-walking)
    # ------------------------------------------------------------------

    def price_impact_buy(self, quantity: float) -> float:
        """
        Estimated cost in bps to BUY *quantity* shares by walking up the ask side.

        cost_bps = (avg_fill_price / best_ask - 1) * 10_000
        Returns 0 bps if the book has insufficient liquidity (partial fill
        at last level is allowed).
        """
        if quantity <= 0:
            return 0.0
        best_ask = self.book.best_ask
        filled = 0.0
        cost = 0.0
        for level in self.book.asks:
            if filled >= quantity:
                break
            take = min(level.size, quantity - filled)
            cost += level.price * take
            filled += take
        if filled == 0:
            return 0.0
        avg_price = cost / filled
        return (avg_price / best_ask - 1.0) * 10_000.0

    def price_impact_sell(self, quantity: float) -> float:
        """
        Estimated shortfall in bps from selling *quantity* shares walking down bids.

        shortfall_bps = (best_bid / avg_fill_price - 1) * 10_000
        Positive value = proceeds below best bid (negative market impact for seller).
        """
        if quantity <= 0:
            return 0.0
        best_bid = self.book.best_bid
        filled = 0.0
        proceeds = 0.0
        for level in self.book.bids:
            if filled >= quantity:
                break
            take = min(level.size, quantity - filled)
            proceeds += level.price * take
            filled += take
        if filled == 0:
            return 0.0
        avg_price = proceeds / filled
        return (best_bid / avg_price - 1.0) * 10_000.0

    # ------------------------------------------------------------------
    # Cumulative depth profiles
    # ------------------------------------------------------------------

    def cumulative_bid_depth(self, n_levels: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cumulative bid depth profile over the top *n_levels* bid levels.

        Returns
        -------
        prices : np.ndarray — bid prices, descending
        cum_sizes : np.ndarray — cumulative sizes (best bid to depth)
        """
        levels = self.book.bids[:n_levels]
        prices = np.array([l.price for l in levels])
        sizes = np.array([l.size for l in levels])
        cum_sizes = np.cumsum(sizes)
        return prices, cum_sizes

    def cumulative_ask_depth(self, n_levels: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cumulative ask depth profile over the top *n_levels* ask levels.

        Returns
        -------
        prices : np.ndarray — ask prices, ascending
        cum_sizes : np.ndarray — cumulative sizes (best ask outward)
        """
        levels = self.book.asks[:n_levels]
        prices = np.array([l.price for l in levels])
        sizes = np.array([l.size for l in levels])
        cum_sizes = np.cumsum(sizes)
        return prices, cum_sizes

    # ------------------------------------------------------------------
    # Skew and imbalance
    # ------------------------------------------------------------------

    def book_skew(self, levels: int = 5) -> float:
        """
        Order book skew: bid_depth_pct - ask_depth_pct.

        Positive → more size on bid side (buy pressure).
        Negative → more size on ask side (sell pressure).
        Returns 0 if total depth is zero.
        """
        bid_depth = sum(l.size for l in self.book.bids[:levels])
        ask_depth = sum(l.size for l in self.book.asks[:levels])
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total

    def aggregate_imbalance(self, levels: int = 5) -> float:
        """
        Proximity-weighted order-book imbalance ∈ [-1, 1].

        level_imbalance_i = (bid_i - ask_i) / (bid_i + ask_i)
        weight_i = 1 / i  (i = 1 for best level)
        aggregate = weighted_sum / sum(weights)

        Positive → buy pressure, Negative → sell pressure.
        """
        bid_levels = self.book.bids[:levels]
        ask_levels = self.book.asks[:levels]
        n = min(len(bid_levels), len(ask_levels), levels)
        if n == 0:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for i in range(n):
            bid_sz = bid_levels[i].size
            ask_sz = ask_levels[i].size
            denom = bid_sz + ask_sz
            if denom == 0:
                continue
            imb = (bid_sz - ask_sz) / denom
            w = 1.0 / (i + 1)
            weighted_sum += w * imb
            weight_total += w

        if weight_total == 0:
            return 0.0
        return weighted_sum / weight_total

    # ------------------------------------------------------------------
    # Wall detection
    # ------------------------------------------------------------------

    def bid_wall(self, threshold_multiple: float = 2.0) -> Optional[float]:
        """
        Detect the shallowest bid 'wall' — a level whose size exceeds
        *threshold_multiple* × average bid size.

        Returns the price of the highest-price (closest to mid) wall, or None.
        """
        if not self.book.bids:
            return None
        sizes = [l.size for l in self.book.bids]
        avg_size = np.mean(sizes)
        if avg_size == 0:
            return None
        threshold = threshold_multiple * avg_size
        # Return the wall closest to the best bid (highest price)
        for level in self.book.bids:
            if level.size > threshold:
                return level.price
        return None

    def ask_wall(self, threshold_multiple: float = 2.0) -> Optional[float]:
        """
        Detect the shallowest ask 'wall' — a level whose size exceeds
        *threshold_multiple* × average ask size.

        Returns the price of the lowest-price (closest to mid) wall, or None.
        """
        if not self.book.asks:
            return None
        sizes = [l.size for l in self.book.asks]
        avg_size = np.mean(sizes)
        if avg_size == 0:
            return None
        threshold = threshold_multiple * avg_size
        for level in self.book.asks:
            if level.size > threshold:
                return level.price
        return None

    # ------------------------------------------------------------------
    # Depth map (liquidity map)
    # ------------------------------------------------------------------

    def depth_map(self, n_levels: int = 10) -> Dict:
        """
        Full liquidity map for visualisation.

        Returns
        -------
        dict with keys:
            bid_prices, bid_sizes, ask_prices, ask_sizes,
            cum_bid, cum_ask
        """
        bid_prices, cum_bid = self.cumulative_bid_depth(n_levels)
        ask_prices, cum_ask = self.cumulative_ask_depth(n_levels)
        bid_sizes = np.array([l.size for l in self.book.bids[:n_levels]])
        ask_sizes = np.array([l.size for l in self.book.asks[:n_levels]])
        return {
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
            "cum_bid": cum_bid,
            "cum_ask": cum_ask,
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class OrderBookSimulator:
    """
    Synthetic order book generator for testing and backtesting.

    Parameters
    ----------
    mid_price : float, default 100.0
    n_levels  : int, default 10
    seed      : int, default 42
    """

    def __init__(
        self,
        mid_price: float = 100.0,
        n_levels: int = 10,
        seed: int = 42,
    ) -> None:
        self.mid_price = mid_price
        self.n_levels = n_levels
        self.rng = np.random.default_rng(seed)

    def generate(self, spread_bps: float = 5.0) -> OrderBook:
        """
        Generate a balanced synthetic order book.

        Prices are spaced at approximately 1 tick (spread_bps / 2) from mid.
        Sizes follow a log-normal distribution.
        """
        half_spread = self.mid_price * spread_bps / 20_000.0  # half spread in $

        bid_prices = [
            self.mid_price - half_spread * (1 + i)
            for i in range(self.n_levels)
        ]
        ask_prices = [
            self.mid_price + half_spread * (1 + i)
            for i in range(self.n_levels)
        ]

        bid_sizes = np.round(
            self.rng.lognormal(mean=5.0, sigma=0.6, size=self.n_levels) * 10
        )
        ask_sizes = np.round(
            self.rng.lognormal(mean=5.0, sigma=0.6, size=self.n_levels) * 10
        )
        # Ensure minimum size of 1
        bid_sizes = np.maximum(bid_sizes, 1)
        ask_sizes = np.maximum(ask_sizes, 1)

        bids = [OrderBookLevel(price=p, size=s) for p, s in zip(bid_prices, bid_sizes)]
        asks = [OrderBookLevel(price=p, size=s) for p, s in zip(ask_prices, ask_sizes)]
        return OrderBook(bids=bids, asks=asks)

    def generate_skewed(self, buy_pressure: float = 0.3) -> OrderBook:
        """
        Generate a skewed order book.

        Parameters
        ----------
        buy_pressure : float ∈ (-1, 1)
            Positive → more size on bids (buy pressure).
            Negative → more size on asks (sell pressure).
        """
        base = self.generate()

        # Scale factor: shift bid/ask volume proportional to pressure
        # buy_pressure = 0.5 → bid sizes multiplied by 1.5, ask by 0.5
        bid_factor = 1.0 + abs(buy_pressure) if buy_pressure >= 0 else 1.0 - abs(buy_pressure)
        ask_factor = 1.0 - abs(buy_pressure) if buy_pressure >= 0 else 1.0 + abs(buy_pressure)
        bid_factor = max(bid_factor, 0.1)
        ask_factor = max(ask_factor, 0.1)

        bids = [
            OrderBookLevel(price=l.price, size=max(1.0, l.size * bid_factor), n_orders=l.n_orders)
            for l in base.bids
        ]
        asks = [
            OrderBookLevel(price=l.price, size=max(1.0, l.size * ask_factor), n_orders=l.n_orders)
            for l in base.asks
        ]
        return OrderBook(bids=bids, asks=asks)


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

class DepthSignalGenerator:
    """
    Generate trading signals from order book depth data.
    """

    def imbalance_signal(self, book: OrderBook, levels: int = 5) -> float:
        """
        Aggregate proximity-weighted imbalance signal ∈ [-1, 1].
        Positive → buy signal (more bids), Negative → sell signal (more asks).
        """
        analytics = DepthAnalytics(book)
        return analytics.aggregate_imbalance(levels)

    def wall_signal(self, book: OrderBook) -> Dict:
        """
        Summarise detected price walls.

        Returns
        -------
        dict with keys:
            bid_wall (float or None), ask_wall (float or None),
            bid_wall_detected (bool), ask_wall_detected (bool),
            wall_gap (float or None) — distance between walls
        """
        analytics = DepthAnalytics(book)
        bw = analytics.bid_wall()
        aw = analytics.ask_wall()
        gap = (aw - bw) if (bw is not None and aw is not None) else None
        return {
            "bid_wall": bw,
            "ask_wall": aw,
            "bid_wall_detected": bw is not None,
            "ask_wall_detected": aw is not None,
            "wall_gap": gap,
        }

    def impact_cost_signal(
        self,
        book: OrderBook,
        trade_size: float = 1000.0,
    ) -> float:
        """
        Estimated round-trip impact cost in bps for *trade_size* shares.

        impact_bps = (buy_impact + sell_impact) / 2
        """
        analytics = DepthAnalytics(book)
        buy_bps = analytics.price_impact_buy(trade_size)
        sell_bps = analytics.price_impact_sell(trade_size)
        return (buy_bps + sell_bps) / 2.0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def depth_weighted_mid(book: OrderBook, levels: int = 5) -> float:
    """Depth-weighted mid-price from the top *levels* of each side."""
    return DepthAnalytics(book).depth_weighted_mid(levels)


def price_impact_bps(book: OrderBook, quantity: float, side: str = "buy") -> float:
    """
    Estimated price impact in bps for a market order.

    Parameters
    ----------
    side : {'buy', 'sell'}
    """
    analytics = DepthAnalytics(book)
    if side.lower() == "buy":
        return analytics.price_impact_buy(quantity)
    elif side.lower() == "sell":
        return analytics.price_impact_sell(quantity)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def book_imbalance(book: OrderBook, levels: int = 5) -> float:
    """Proximity-weighted aggregate imbalance ∈ [-1, 1]."""
    return DepthAnalytics(book).aggregate_imbalance(levels)


def detect_walls(book: OrderBook, threshold: float = 2.0) -> Dict:
    """
    Detect bid and ask walls.

    Returns
    -------
    dict with 'bid_wall' and 'ask_wall' (float or None)
    """
    analytics = DepthAnalytics(book)
    return {
        "bid_wall": analytics.bid_wall(threshold),
        "ask_wall": analytics.ask_wall(threshold),
    }
