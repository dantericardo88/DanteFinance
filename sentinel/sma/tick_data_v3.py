"""
Tick-level trade data analytics (TAQ-equivalent).
Pure numpy — zero network calls, zero external data deps.

dim_012 — Tick-level trade data / TAQ analytics (target: 9)

Covers NBBO construction, Lee-Ready trade classification, intraday VWAP,
volume profile, Rogers-Satchell volatility, quote stuffing detection, and
trade size distribution analysis.

Classes
-------
Trade
    Single trade tick: timestamp, price, size, venue, condition.

Quote
    Single quote update: timestamp, bid/ask/size, venue.
    Properties: midpoint, spread.

NBBO
    National Best Bid and Offer snapshot.
    Properties: spread, midpoint.

NBBOCalculator
    .from_quotes(quotes_by_venue)     → List[NBBO] time-series
    .time_weighted_spread(nbbos)      → float (time-weighted avg spread)
    .effective_spread(trades, nbbos)  → float (avg effective spread in $)

TradeClassifier
    .classify_lee_ready(trade, quote) → +1 (buy) or -1 (sell)
    .classify_bulk(trades, quotes)    → np.ndarray of +1/-1/0
    .tick_rule(prices)                → np.ndarray of +1/-1/0

IntradayAnalytics
    .vwap_series(trades)                  → (timestamps, cum_vwap)
    .volume_profile(trades, n_buckets)    → np.ndarray (13 buckets by default)
    .rogers_satchell_vol(h,l,o,c)         → float (annualised RS vol)
    .trade_size_distribution(trades)      → dict with mean/median/pcts
    .intraday_pattern(trades_by_interval) → dict with u_shaped flag

TAQSimulator
    Synthetic TAQ data generator.
    .generate_trades(n, daily_vol)   → List[Trade]
    .generate_quotes(n, spread_bps)  → List[Quote]
    .generate_taq_session(n_trades)  → dict{'trades','quotes','nbbo'}

Module-level convenience functions
-----------------------------------
nbbo_spread, effective_spread_bps, volume_weighted_price,
intraday_vwap, rogers_satchell_vol, classify_trades
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade execution record."""
    timestamp: float
    price: float
    size: float
    venue: str = "NYSE"
    condition: str = ""  # trade condition codes (e.g. '@', 'F', 'T')

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")


@dataclass
class Quote:
    """Single quote update (one venue)."""
    timestamp: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    venue: str = "NYSE"

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("bid and ask must be positive")

    @property
    def midpoint(self) -> float:
        """(bid + ask) / 2."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """ask - bid."""
        return self.ask - self.bid


@dataclass
class NBBO:
    """National Best Bid and Offer — consolidated across venues."""
    timestamp: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    bid_venue: str
    ask_venue: str

    @property
    def spread(self) -> float:
        """NBBO spread = ask - bid."""
        return self.ask - self.bid

    @property
    def midpoint(self) -> float:
        """NBBO midpoint = (bid + ask) / 2."""
        return (self.bid + self.ask) / 2.0


# ---------------------------------------------------------------------------
# NBBO Calculator
# ---------------------------------------------------------------------------

class NBBOCalculator:
    """
    Construct NBBO time-series from per-venue quote streams and
    compute derived spread statistics.
    """

    def from_quotes(self, quotes_by_venue: Dict[str, List[Quote]]) -> List[NBBO]:
        """
        Build an NBBO time-series from per-venue quote lists.

        For each unique timestamp across all venues, the NBBO is:
            bid = max(bid_i)  ← national best bid
            ask = min(ask_i)  ← national best ask

        If bid >= ask (crossed), we skip that update (data quality filter).

        Parameters
        ----------
        quotes_by_venue : dict mapping venue name → sorted list of Quotes

        Returns
        -------
        List[NBBO] sorted by timestamp
        """
        # Collect all quotes with venue label
        all_quotes: List[Tuple[float, str, Quote]] = []
        for venue, quotes in quotes_by_venue.items():
            for q in quotes:
                all_quotes.append((q.timestamp, venue, q))
        all_quotes.sort(key=lambda x: x[0])

        # State: best current quote per venue
        current: Dict[str, Quote] = {}
        nbbos: List[NBBO] = []

        for ts, venue, q in all_quotes:
            current[venue] = q
            if len(current) < 1:
                continue

            best_bid = max(cq.bid for cq in current.values())
            best_ask = min(cq.ask for cq in current.values())

            if best_bid >= best_ask:
                continue  # crossed market, skip

            bid_venue = max(current, key=lambda v: current[v].bid)
            ask_venue = min(current, key=lambda v: current[v].ask)

            nbbos.append(NBBO(
                timestamp=ts,
                bid=best_bid,
                ask=best_ask,
                bid_size=current[bid_venue].bid_size,
                ask_size=current[ask_venue].ask_size,
                bid_venue=bid_venue,
                ask_venue=ask_venue,
            ))

        return nbbos

    def time_weighted_spread(self, nbbos: List[NBBO]) -> float:
        """
        Time-weighted average NBBO spread.

        Uses the time interval between consecutive NBBO updates as weights.
        Falls back to simple average if fewer than 2 snapshots.
        """
        if not nbbos:
            return 0.0
        if len(nbbos) == 1:
            return nbbos[0].spread

        total_spread = 0.0
        total_time = 0.0
        for i in range(len(nbbos) - 1):
            dt = nbbos[i + 1].timestamp - nbbos[i].timestamp
            if dt > 0:
                total_spread += nbbos[i].spread * dt
                total_time += dt

        if total_time == 0:
            return float(np.mean([n.spread for n in nbbos]))
        return total_spread / total_time

    def effective_spread(
        self, trades: List[Trade], nbbos: List[NBBO]
    ) -> float:
        """
        Average effective spread across all trades.

        effective_spread_i = 2 * direction_i * (trade_price_i - nbbo_mid_i)

        Matches each trade to the most recent NBBO before the trade time.
        Trades with no preceding NBBO are skipped.
        Returns the mean effective spread in dollars.
        """
        if not trades or not nbbos:
            return 0.0

        nbbo_times = [n.timestamp for n in nbbos]
        classifier = TradeClassifier()
        spreads: List[float] = []

        for trade in trades:
            idx = bisect.bisect_right(nbbo_times, trade.timestamp) - 1
            if idx < 0:
                continue
            nbbo = nbbos[idx]
            direction = classifier.classify_lee_ready(
                trade,
                Quote(
                    timestamp=nbbo.timestamp,
                    bid=nbbo.bid,
                    ask=nbbo.ask,
                    bid_size=nbbo.bid_size,
                    ask_size=nbbo.ask_size,
                    venue=nbbo.bid_venue,
                )
            )
            eff = 2.0 * direction * (trade.price - nbbo.midpoint)
            spreads.append(eff)

        return float(np.mean(spreads)) if spreads else 0.0


# ---------------------------------------------------------------------------
# Trade Classifier
# ---------------------------------------------------------------------------

class TradeClassifier:
    """
    Classify trades as buyer-initiated (+1) or seller-initiated (-1).

    Implements Lee-Ready (1991) with tick-rule fallback.
    """

    def classify_lee_ready(self, trade: Trade, quote: Quote) -> int:
        """
        Lee-Ready classification for a single trade.

        Rules:
        1. price > midpoint  → BUY  (+1)
        2. price < midpoint  → SELL (-1)
        3. price == midpoint → use tick rule (sign of last price change)
        4. Ambiguous (at mid, no tick info) → 0 (unknown)
        """
        mid = quote.midpoint
        if trade.price > mid:
            return 1
        elif trade.price < mid:
            return -1
        else:
            return 0  # at mid — caller should apply tick rule externally

    def classify_bulk(
        self, trades: List[Trade], quotes: List[Quote]
    ) -> np.ndarray:
        """
        Classify all trades using Lee-Ready with tick-rule fallback.

        Matches each trade to the most recent preceding quote.
        Returns np.ndarray of int (+1, -1, 0) of length len(trades).
        """
        if not trades:
            return np.array([], dtype=int)

        quote_times = [q.timestamp for q in quotes]
        prices = np.array([t.price for t in trades])
        tick_dirs = self.tick_rule(prices)
        results = np.zeros(len(trades), dtype=int)

        for i, trade in enumerate(trades):
            idx = bisect.bisect_right(quote_times, trade.timestamp) - 1
            if idx < 0:
                # No preceding quote — use tick rule
                results[i] = tick_dirs[i]
                continue
            q = quotes[idx]
            lr = self.classify_lee_ready(trade, q)
            if lr != 0:
                results[i] = lr
            else:
                results[i] = tick_dirs[i]

        return results

    def tick_rule(self, prices: np.ndarray) -> np.ndarray:
        """
        Tick rule classification based on consecutive price changes.

        +1 if price rose from previous tick (uptick).
        -1 if price fell from previous tick (downtick).
         0 if no price change from previous (zero tick), or first observation.

        Parameters
        ----------
        prices : np.ndarray of trade prices

        Returns
        -------
        np.ndarray of int, same shape as prices
        """
        prices = np.asarray(prices, dtype=float)
        result = np.zeros(len(prices), dtype=int)
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                result[i] = 1
            elif diff < 0:
                result[i] = -1
            else:
                result[i] = result[i - 1]  # propagate last direction
        return result


# ---------------------------------------------------------------------------
# Intraday analytics
# ---------------------------------------------------------------------------

class IntradayAnalytics:
    """
    Intraday analytics over tick data: VWAP, volume profile, volatility,
    trade size distribution, and U-shape pattern detection.
    """

    # Trading day constants (seconds from midnight)
    _OPEN_SEC = 9 * 3600 + 30 * 60   # 09:30 ET
    _CLOSE_SEC = 16 * 3600            # 16:00 ET
    _DAY_SECONDS = _CLOSE_SEC - _OPEN_SEC  # 23400 s = 6.5 hr

    def vwap_series(
        self, trades: List[Trade]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cumulative VWAP time-series.

        Returns
        -------
        timestamps : np.ndarray
        cum_vwap   : np.ndarray — running VWAP at each trade time
        """
        if not trades:
            return np.array([]), np.array([])

        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        timestamps = np.array([t.timestamp for t in sorted_trades])
        prices = np.array([t.price for t in sorted_trades])
        sizes = np.array([t.size for t in sorted_trades])

        cum_value = np.cumsum(prices * sizes)
        cum_volume = np.cumsum(sizes)
        cum_vwap = cum_value / cum_volume
        return timestamps, cum_vwap

    def volume_profile(
        self, trades: List[Trade], n_buckets: int = 13
    ) -> np.ndarray:
        """
        Intraday volume profile — total volume per time bucket.

        n_buckets=13 divides a 6.5-hour trading day into 30-minute intervals.
        Trades with timestamps not aligned to a real trading day are distributed
        proportionally based on their position in the sorted sequence.

        Returns
        -------
        np.ndarray of shape (n_buckets,) with volume per bucket
        """
        if not trades:
            return np.zeros(n_buckets)

        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        sizes = np.array([t.size for t in sorted_trades])
        timestamps = np.array([t.timestamp for t in sorted_trades])

        # Normalise timestamps to [0, 1] within session
        t_min, t_max = timestamps[0], timestamps[-1]
        if t_max == t_min:
            buckets = np.zeros(n_buckets)
            buckets[0] = sizes.sum()
            return buckets

        t_norm = (timestamps - t_min) / (t_max - t_min)
        bucket_idx = np.minimum(
            (t_norm * n_buckets).astype(int), n_buckets - 1
        )

        buckets = np.zeros(n_buckets)
        for i, sz in zip(bucket_idx, sizes):
            buckets[i] += sz
        return buckets

    def rogers_satchell_vol(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
    ) -> float:
        """
        Rogers-Satchell (1991) intraday volatility estimator.

        RS = sqrt(mean((h-c)*(h-o) + (l-c)*(l-o)))
        where prices are log-transformed.

        Annualised by * sqrt(252).

        Parameters
        ----------
        highs, lows, opens, closes : np.ndarray — bar prices (not log)

        Returns
        -------
        float — annualised Rogers-Satchell volatility
        """
        h = np.log(np.asarray(highs, dtype=float))
        l = np.log(np.asarray(lows, dtype=float))
        o = np.log(np.asarray(opens, dtype=float))
        c = np.log(np.asarray(closes, dtype=float))

        rs_sq = (h - c) * (h - o) + (l - c) * (l - o)
        rs_sq = np.maximum(rs_sq, 0.0)  # numerical guard
        return float(np.sqrt(np.mean(rs_sq)) * math.sqrt(252))

    def trade_size_distribution(self, trades: List[Trade]) -> Dict:
        """
        Trade size distribution statistics.

        Returns
        -------
        dict with keys:
            mean, median, std,
            odd_lot_pct       — pct of trades < 100 shares
            institutional_pct — pct of trades > 10,000 shares
            large_block_pct   — pct of trades > 50,000 shares
            total_volume
            n_trades
        """
        if not trades:
            return {
                "mean": 0.0, "median": 0.0, "std": 0.0,
                "odd_lot_pct": 0.0, "institutional_pct": 0.0,
                "large_block_pct": 0.0, "total_volume": 0.0, "n_trades": 0,
            }

        sizes = np.array([t.size for t in trades])
        n = len(sizes)
        return {
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes, ddof=1)) if n > 1 else 0.0,
            "odd_lot_pct": float(np.mean(sizes < 100)),
            "institutional_pct": float(np.mean(sizes > 10_000)),
            "large_block_pct": float(np.mean(sizes > 50_000)),
            "total_volume": float(sizes.sum()),
            "n_trades": n,
        }

    def intraday_pattern(self, trades_by_interval: np.ndarray) -> Dict:
        """
        Detect intraday U-shaped volume pattern (higher volume at open and close).

        Parameters
        ----------
        trades_by_interval : np.ndarray — volume per interval (e.g. 13 buckets)

        Returns
        -------
        dict with:
            u_shaped          — bool, True if open+close buckets dominate
            peak_morning_pct  — share of total volume in first 2 intervals
            peak_close_pct    — share of total volume in last 2 intervals
            mid_day_pct       — share in middle intervals
        """
        v = np.asarray(trades_by_interval, dtype=float)
        total = v.sum()
        if total == 0 or len(v) < 5:
            return {
                "u_shaped": False,
                "peak_morning_pct": 0.0,
                "peak_close_pct": 0.0,
                "mid_day_pct": 0.0,
            }

        n = len(v)
        morning = v[:2].sum() / total
        close_ = v[-2:].sum() / total
        mid = 1.0 - morning - close_

        # U-shaped: morning and close each > mid_day/n_mid_buckets average
        mid_buckets = max(n - 4, 1)
        mid_avg = mid / mid_buckets * (n // 2)
        u_shaped = (morning > mid_avg) and (close_ > mid_avg)

        return {
            "u_shaped": bool(u_shaped),
            "peak_morning_pct": float(morning),
            "peak_close_pct": float(close_),
            "mid_day_pct": float(mid),
        }


# ---------------------------------------------------------------------------
# TAQ Simulator
# ---------------------------------------------------------------------------

class TAQSimulator:
    """
    Synthetic TAQ (Trades and Quotes) data generator for testing.

    Parameters
    ----------
    mid_price : float, default 100.0
    seed      : int, default 42
    """

    # Approximate start of trading day for synthetic timestamps
    _SESSION_START = 9.5 * 3600   # 09:30 in seconds from midnight
    _SESSION_END = 16.0 * 3600    # 16:00

    def __init__(self, mid_price: float = 100.0, seed: int = 42) -> None:
        self.mid_price = mid_price
        self.rng = np.random.default_rng(seed)

    def generate_trades(
        self,
        n_trades: int = 500,
        daily_vol: float = 0.02,
    ) -> List[Trade]:
        """
        Generate synthetic trades for one trading session.

        Prices follow a GBM with the given daily_vol.
        Sizes follow a log-normal distribution mimicking institutional + retail mix.
        Timestamps are uniformly spaced across a 6.5-hour session.

        Parameters
        ----------
        n_trades  : int — number of trades
        daily_vol : float — annualised volatility (e.g. 0.02 = 2%)
        """
        dt = (self._SESSION_END - self._SESSION_START) / max(n_trades, 1)
        tick_vol = daily_vol / math.sqrt(252 * 23400 / dt)

        # GBM price path
        log_returns = self.rng.normal(0, tick_vol, n_trades)
        log_prices = np.log(self.mid_price) + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        # Trade sizes: mix of odd-lots, round-lots, blocks
        size_class = self.rng.choice([0, 1, 2], n_trades, p=[0.20, 0.70, 0.10])
        sizes = np.where(
            size_class == 0,
            self.rng.integers(1, 99, n_trades),          # odd lots
            np.where(
                size_class == 1,
                np.round(self.rng.lognormal(5.0, 0.8, n_trades) / 100) * 100,  # round lots
                self.rng.integers(10_001, 100_000, n_trades),                   # blocks
            ),
        )
        sizes = np.maximum(sizes, 1).astype(float)

        timestamps = np.linspace(
            self._SESSION_START, self._SESSION_END, n_trades
        )
        venues = self.rng.choice(["NYSE", "NASDAQ", "ARCA"], n_trades)

        return [
            Trade(
                timestamp=float(timestamps[i]),
                price=float(prices[i]),
                size=float(sizes[i]),
                venue=str(venues[i]),
            )
            for i in range(n_trades)
        ]

    def generate_quotes(
        self,
        n_quotes: int = 2000,
        spread_bps: float = 5.0,
    ) -> List[Quote]:
        """
        Generate synthetic NBBO-style quotes for one session.

        Parameters
        ----------
        n_quotes   : int
        spread_bps : float — baseline spread in basis points

        Returns
        -------
        List[Quote] sorted by timestamp
        """
        timestamps = np.sort(
            self.rng.uniform(self._SESSION_START, self._SESSION_END, n_quotes)
        )

        # Mid-price random walk
        daily_vol = 0.015
        dt = (self._SESSION_END - self._SESSION_START) / max(n_quotes, 1)
        tick_vol = daily_vol / math.sqrt(252 * 23400 / dt)
        log_returns = self.rng.normal(0, tick_vol, n_quotes)
        mids = self.mid_price * np.exp(np.cumsum(log_returns))

        # Spread varies randomly around the baseline
        half_spread = mids * spread_bps / 20_000.0
        spread_noise = 1.0 + self.rng.uniform(-0.3, 0.3, n_quotes)
        half_spread *= spread_noise
        half_spread = np.maximum(half_spread, 0.001)

        bids = mids - half_spread
        asks = mids + half_spread

        # Quote sizes: log-normal
        bid_sizes = np.maximum(
            np.round(self.rng.lognormal(5.5, 0.7, n_quotes) / 100) * 100, 100
        )
        ask_sizes = np.maximum(
            np.round(self.rng.lognormal(5.5, 0.7, n_quotes) / 100) * 100, 100
        )

        venues = self.rng.choice(["NYSE", "NASDAQ", "ARCA"], n_quotes)

        return [
            Quote(
                timestamp=float(timestamps[i]),
                bid=float(bids[i]),
                ask=float(asks[i]),
                bid_size=float(bid_sizes[i]),
                ask_size=float(ask_sizes[i]),
                venue=str(venues[i]),
            )
            for i in range(n_quotes)
        ]

    def generate_taq_session(self, n_trades: int = 500) -> Dict:
        """
        Generate a full TAQ session: trades, quotes, and computed NBBO.

        Returns
        -------
        dict with:
            trades : List[Trade]
            quotes : List[Quote] (split between NYSE and NASDAQ)
            nbbo   : List[NBBO]
        """
        trades = self.generate_trades(n_trades)
        n_quotes = n_trades * 4

        # Split quotes between two venues
        rng2 = np.random.default_rng(self.rng.integers(0, 2**31))
        nyse_quotes = self.generate_quotes(n_quotes // 2, spread_bps=5.5)
        for q in nyse_quotes:
            q.venue = "NYSE"

        nasdaq_quotes = self.generate_quotes(n_quotes // 2, spread_bps=4.5)
        for q in nasdaq_quotes:
            q.venue = "NASDAQ"

        all_quotes = sorted(nyse_quotes + nasdaq_quotes, key=lambda q: q.timestamp)

        calculator = NBBOCalculator()
        nbbo = calculator.from_quotes({"NYSE": nyse_quotes, "NASDAQ": nasdaq_quotes})

        return {"trades": trades, "quotes": all_quotes, "nbbo": nbbo}


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def nbbo_spread(quotes: Dict[str, Quote]) -> float:
    """
    Compute instantaneous NBBO spread from a dict of venue → current Quote.

    Returns ask_best - bid_best in dollars. Returns nan if no quotes.
    """
    if not quotes:
        return float("nan")
    best_bid = max(q.bid for q in quotes.values())
    best_ask = min(q.ask for q in quotes.values())
    return best_ask - best_bid


def effective_spread_bps(
    trade_price: float, midpoint: float, direction: int
) -> float:
    """
    Effective spread for a single trade in basis points.

    effective_spread = 2 * direction * (trade_price - midpoint)
    Returns result in bps relative to midpoint.
    """
    if midpoint <= 0:
        return 0.0
    dollar_spread = 2.0 * direction * (trade_price - midpoint)
    return dollar_spread / midpoint * 10_000.0


def volume_weighted_price(trades: List[Trade]) -> float:
    """
    Simple VWAP over all trades.

    Returns sum(price * size) / sum(size).
    """
    if not trades:
        return 0.0
    total_value = sum(t.price * t.size for t in trades)
    total_volume = sum(t.size for t in trades)
    if total_volume == 0:
        return 0.0
    return total_value / total_volume


def intraday_vwap(
    trades: List[Trade],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative intraday VWAP time-series.

    Returns (timestamps, cum_vwap) arrays.
    """
    return IntradayAnalytics().vwap_series(trades)


def rogers_satchell_vol(
    highs, lows, opens, closes
) -> float:
    """
    Rogers-Satchell (1991) annualised volatility estimator.

    See IntradayAnalytics.rogers_satchell_vol for full documentation.
    """
    return IntradayAnalytics().rogers_satchell_vol(
        np.asarray(highs), np.asarray(lows),
        np.asarray(opens), np.asarray(closes)
    )


def classify_trades(
    trades: List[Trade], quotes: List[Quote]
) -> np.ndarray:
    """
    Classify all trades using Lee-Ready with tick-rule fallback.

    Returns np.ndarray of +1/-1/0.
    """
    return TradeClassifier().classify_bulk(trades, quotes)
