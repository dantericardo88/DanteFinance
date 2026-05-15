"""
Event-driven backtesting engine v2 — Dimension #062 (target score 9).

Builds on event_driven_backtest.py with:
  - AdvancedOrderTypes: bracket orders, trailing stops, TIF, TWAP/VWAP, hidden
  - MarketMicrostructureModel: bid-ask spread, Almgren-Chriss impact, partial fills
  - PortfolioConstraintEngine: gross/net exposure, sector limits, beta neutrality,
    correlation caps, VaR budget
  - MultiStrategyBacktester: capital allocation, regime rotation, combined attribution
  - BacktestAnalyticsV2: monthly heatmap, rolling Sharpe/drawdown, streaks, best/worst

FastAPI router: backtest_v2_router
  POST /backtest/v2/run
  POST /backtest/v2/multi-strategy
  GET  /backtest/v2/result/{run_id}
  POST /backtest/v2/analyze
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import math
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Iterator, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sentinel.core.logging import get_logger

# Re-export base classes so callers only need this module
from sentinel.sbx.event_driven_backtest import (
    BaseStrategy,
    BuyAndHoldStrategy,
    Event,
    EventQueue,
    EventType,
    FillEvent,
    MarketDataEvent,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossStrategy,
    OrderEvent,
    Portfolio,
    PortfolioSnapshot,
    SignalEvent,
    _PRIORITY,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Extended order type literals
# ---------------------------------------------------------------------------

OrderTypeV2 = Literal[
    "MARKET", "LIMIT", "STOP", "STOP_LIMIT",
    "BRACKET", "TRAILING_STOP_PCT", "TRAILING_STOP_ATR",
    "TWAP", "VWAP", "HIDDEN",
]

TimeInForce = Literal["DAY", "GTC", "IOC", "FOK"]


# ---------------------------------------------------------------------------
# AdvancedOrderTypes
# ---------------------------------------------------------------------------

@dataclass
class BracketOrder:
    """Atomic bracket: entry + take-profit + stop-loss."""
    entry_order_id: str
    ticker: str
    direction: Literal["BUY", "SELL"]
    quantity: float
    entry_price: float           # limit or None → market
    take_profit_price: float
    stop_loss_price: float
    tif: TimeInForce = "GTC"
    filled_quantity: float = 0.0
    status: Literal["PENDING", "ENTRY_FILLED", "TP_FILLED", "SL_FILLED", "CANCELLED"] = "PENDING"
    bracket_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_complete(self) -> bool:
        return self.status in ("TP_FILLED", "SL_FILLED", "CANCELLED")


@dataclass
class TrailingStopOrder:
    """Trailing stop: percent-based or ATR-based."""
    order_id: str
    ticker: str
    direction: Literal["BUY", "SELL"]   # direction of the trailing stop exit
    quantity: float
    trail_type: Literal["PCT", "ATR"]
    trail_value: float                  # pct (e.g. 0.05 = 5%) or ATR multiplier
    atr_period: int = 14                # used when trail_type == "ATR"
    activation_price: float = 0.0       # only activate when price moves this far
    peak_price: float = 0.0             # tracks high (long) or low (short)
    stop_price: float = 0.0             # computed trailing stop level
    active: bool = False
    filled: bool = False
    tif: TimeInForce = "GTC"
    order_type: str = "TRAILING_STOP"


@dataclass
class AlgoSliceOrder:
    """TWAP / VWAP algorithmic execution split into child slices."""
    order_id: str
    ticker: str
    direction: Literal["BUY", "SELL"]
    total_quantity: float
    algo_type: Literal["TWAP", "VWAP"]
    start_time: datetime
    end_time: datetime
    n_slices: int = 10
    filled_quantity: float = 0.0
    child_orders: list[OrderEvent] = field(default_factory=list)
    completed: bool = False

    def remaining(self) -> float:
        return max(0.0, self.total_quantity - self.filled_quantity)

    def slice_size(self) -> float:
        return self.total_quantity / max(self.n_slices, 1)


class AdvancedOrderTypes:
    """
    Factory and manager for extended order types.

    Maintains pending bracket orders, trailing stops, and algo slices.
    On each bar, updates trailing stop levels and checks for trigger.
    """

    def __init__(self) -> None:
        self._brackets: dict[str, BracketOrder] = {}      # bracket_id → BracketOrder
        self._trailing_stops: dict[str, TrailingStopOrder] = {}  # order_id → TSO
        self._algo_slices: dict[str, AlgoSliceOrder] = {}  # order_id → AlgoSliceOrder
        self._atr_cache: dict[str, float] = {}            # ticker → last ATR value

    # ------------------------------------------------------------------
    # Bracket orders
    # ------------------------------------------------------------------

    def create_bracket(
        self,
        ticker: str,
        direction: Literal["BUY", "SELL"],
        quantity: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        tif: TimeInForce = "GTC",
    ) -> BracketOrder:
        bo = BracketOrder(
            entry_order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            tif=tif,
        )
        self._brackets[bo.bracket_id] = bo
        return bo

    def update_bracket_on_fill(self, bracket_id: str, fill_price: float, fill_qty: float) -> None:
        bo = self._brackets.get(bracket_id)
        if bo is None:
            return
        if bo.status == "PENDING":
            bo.filled_quantity += fill_qty
            if bo.filled_quantity >= bo.quantity * 0.99:
                bo.status = "ENTRY_FILLED"

    def check_bracket_exits(
        self, bracket_id: str, current_high: float, current_low: float
    ) -> Optional[Literal["TP", "SL"]]:
        """Check if TP or SL was hit. Returns which one (or None)."""
        bo = self._brackets.get(bracket_id)
        if bo is None or bo.status != "ENTRY_FILLED":
            return None
        if bo.direction == "BUY":
            if current_high >= bo.take_profit_price:
                bo.status = "TP_FILLED"
                return "TP"
            if current_low <= bo.stop_loss_price:
                bo.status = "SL_FILLED"
                return "SL"
        else:  # short
            if current_low <= bo.take_profit_price:
                bo.status = "TP_FILLED"
                return "TP"
            if current_high >= bo.stop_loss_price:
                bo.status = "SL_FILLED"
                return "SL"
        return None

    def cancel_bracket(self, bracket_id: str) -> None:
        bo = self._brackets.get(bracket_id)
        if bo:
            bo.status = "CANCELLED"

    # ------------------------------------------------------------------
    # Trailing stops
    # ------------------------------------------------------------------

    def create_trailing_stop(
        self,
        ticker: str,
        direction: Literal["BUY", "SELL"],
        quantity: float,
        trail_type: Literal["PCT", "ATR"],
        trail_value: float,
        current_price: float,
        atr_period: int = 14,
        tif: TimeInForce = "GTC",
    ) -> TrailingStopOrder:
        tso = TrailingStopOrder(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            trail_type=trail_type,
            trail_value=trail_value,
            atr_period=atr_period,
            peak_price=current_price,
            tif=tif,
        )
        tso.active = True
        tso.stop_price = self._compute_trail_stop(tso, current_price)
        self._trailing_stops[tso.order_id] = tso
        return tso

    def _compute_trail_stop(self, tso: TrailingStopOrder, current_price: float) -> float:
        if tso.trail_type == "PCT":
            if tso.direction == "SELL":  # stop for long position
                return current_price * (1 - tso.trail_value)
            else:  # stop for short position
                return current_price * (1 + tso.trail_value)
        else:  # ATR-based
            atr = self._atr_cache.get(tso.ticker, current_price * 0.01)
            if tso.direction == "SELL":
                return current_price - tso.trail_value * atr
            else:
                return current_price + tso.trail_value * atr

    def update_atr(self, ticker: str, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        """Compute and cache ATR for trailing stop calculations."""
        if len(highs) < period + 1:
            atr = float((highs - lows).mean()) if len(highs) > 0 else 0.0
        else:
            h = highs.values
            l = lows.values
            c = closes.values
            tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
            atr = float(pd.Series(tr).ewm(span=period, adjust=False).mean().iloc[-1])
        self._atr_cache[ticker] = atr
        return atr

    def on_bar(self, ticker: str, bar_open: float, bar_high: float, bar_low: float, bar_close: float) -> list[TrailingStopOrder]:
        """Update trailing stops and return any that triggered."""
        triggered = []
        for tso in list(self._trailing_stops.values()):
            if tso.ticker != ticker or not tso.active or tso.filled:
                continue

            # Update peak price
            if tso.direction == "SELL":  # trailing stop for long (sell if price drops)
                if bar_high > tso.peak_price:
                    tso.peak_price = bar_high
                    tso.stop_price = self._compute_trail_stop(tso, tso.peak_price)
                if bar_low <= tso.stop_price:
                    tso.filled = True
                    triggered.append(tso)
            else:  # trailing stop for short (buy to cover if price rises)
                if bar_low < tso.peak_price:
                    tso.peak_price = bar_low
                    tso.stop_price = self._compute_trail_stop(tso, tso.peak_price)
                if bar_high >= tso.stop_price:
                    tso.filled = True
                    triggered.append(tso)

        return triggered

    # ------------------------------------------------------------------
    # TWAP / VWAP slicing
    # ------------------------------------------------------------------

    def create_algo_order(
        self,
        ticker: str,
        direction: Literal["BUY", "SELL"],
        total_quantity: float,
        algo_type: Literal["TWAP", "VWAP"],
        start_time: datetime,
        end_time: datetime,
        n_slices: int = 10,
    ) -> AlgoSliceOrder:
        aso = AlgoSliceOrder(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            direction=direction,
            total_quantity=total_quantity,
            algo_type=algo_type,
            start_time=start_time,
            end_time=end_time,
            n_slices=n_slices,
        )
        self._algo_slices[aso.order_id] = aso
        return aso

    def get_next_slice(self, order_id: str, current_time: datetime, volume: float = 0.0) -> Optional[OrderEvent]:
        """Get the next child slice to submit, if applicable."""
        aso = self._algo_slices.get(order_id)
        if aso is None or aso.completed:
            return None
        if current_time < aso.start_time or current_time > aso.end_time:
            return None

        remaining = aso.remaining()
        if remaining <= 0:
            aso.completed = True
            return None

        # TWAP: equal time slices
        if aso.algo_type == "TWAP":
            slice_qty = min(aso.slice_size(), remaining)
        else:  # VWAP: weight by volume
            # Simplified: use volume fraction, capped at 2x TWAP slice
            if volume > 0:
                vol_weight = min(volume / max(aso.total_quantity, 1), 2.0)
                slice_qty = min(aso.slice_size() * vol_weight, remaining)
            else:
                slice_qty = min(aso.slice_size(), remaining)

        if slice_qty < 1.0:
            slice_qty = remaining  # clean up

        slice_qty = math.floor(slice_qty)
        if slice_qty <= 0:
            return None

        order = OrderEvent(
            timestamp=current_time,
            event_type="ORDER",
            ticker=aso.ticker,
            order_type="MARKET",
            direction=aso.direction,
            quantity=slice_qty,
        )
        aso.filled_quantity += slice_qty
        if aso.filled_quantity >= aso.total_quantity * 0.99:
            aso.completed = True

        return order

    def get_active_brackets(self) -> list[BracketOrder]:
        return [b for b in self._brackets.values() if not b.is_complete()]

    def get_active_trailing_stops(self) -> list[TrailingStopOrder]:
        return [t for t in self._trailing_stops.values() if t.active and not t.filled]


# ---------------------------------------------------------------------------
# MarketMicrostructureModel
# ---------------------------------------------------------------------------

@dataclass
class MicrostructureParams:
    base_spread_bps: float = 5.0        # baseline spread in bps
    spread_adv_multiplier: float = 1.0  # scales 1/sqrt(ADV) term
    temp_impact_eta: float = 0.1        # Almgren-Chriss η (temporary impact)
    perm_impact_gamma: float = 0.1      # Almgren-Chriss γ (permanent impact)
    daily_liquidity_factor: float = 0.05  # max pct of ADV per order
    hidden_order_discount: float = 0.5  # hidden orders pay 50% less market impact
    min_fill_probability: float = 0.3   # min fill probability for limit orders


class MarketMicrostructureModel:
    """
    Realistic execution model based on market microstructure theory.

    Models:
    - Bid-ask spread: proportional to 1/sqrt(ADV)
    - Market impact: Almgren-Chriss temporary + permanent
    - Partial fills: large orders receive partial fills
    - Limit order fill probability: based on price vs mid
    - Hidden orders: reduced market impact
    """

    def __init__(self, params: MicrostructureParams | None = None) -> None:
        self.params = params or MicrostructureParams()
        self._adv_cache: dict[str, float] = {}    # ticker → avg daily volume
        self._perm_impact: dict[str, float] = {}  # accumulated permanent impact per ticker

    def update_adv(self, ticker: str, volume_series: pd.Series) -> float:
        """Update average daily volume estimate from recent volume data."""
        adv = float(volume_series.rolling(20, min_periods=5).mean().iloc[-1]) if len(volume_series) >= 5 else float(volume_series.mean())
        self._adv_cache[ticker] = max(adv, 1.0)
        return adv

    def compute_spread(self, ticker: str, mid_price: float) -> tuple[float, float]:
        """Compute bid and ask prices.

        Spread model: spread = base_bps + multiplier / sqrt(ADV / reference_volume)
        reference_volume is normalised to 1M shares as unit.
        """
        adv = self._adv_cache.get(ticker, 1_000_000)
        spread_bps = self.params.base_spread_bps + self.params.spread_adv_multiplier * 100 / math.sqrt(max(adv / 1_000_000, 0.001))
        half_spread = mid_price * spread_bps / 20_000  # half-spread
        return mid_price - half_spread, mid_price + half_spread

    def compute_market_impact(
        self,
        ticker: str,
        direction: Literal["BUY", "SELL"],
        quantity: float,
        mid_price: float,
        hidden: bool = False,
    ) -> tuple[float, float]:
        """
        Almgren-Chriss market impact model.

        Returns (temporary_impact_per_share, permanent_impact_per_share).

        Temporary impact: η × σ × (Q / ADV)^0.6
        Permanent impact: γ × σ × (Q / ADV)

        where σ is estimated as spread * 50 (rough proxy for daily vol).
        """
        adv = self._adv_cache.get(ticker, 1_000_000)
        spread = mid_price * self.params.base_spread_bps / 10_000
        sigma = spread * 50  # rough daily vol proxy from spread

        participation_rate = quantity / max(adv, 1.0)

        # Temporary impact (price reverts after trade)
        temp_impact = self.params.temp_impact_eta * sigma * (participation_rate ** 0.6)

        # Permanent impact (price does not revert)
        perm_impact = self.params.perm_impact_gamma * sigma * participation_rate

        if hidden:
            temp_impact *= self.params.hidden_order_discount
            perm_impact *= self.params.hidden_order_discount

        # Accumulate permanent impact
        sign = 1 if direction == "BUY" else -1
        self._perm_impact[ticker] = self._perm_impact.get(ticker, 0.0) + sign * perm_impact

        return temp_impact, perm_impact

    def compute_partial_fill(
        self,
        ticker: str,
        quantity: float,
        available_liquidity: float | None = None,
    ) -> float:
        """
        Determine how much of an order gets filled.

        Large orders relative to ADV get partial fills.
        Returns filled quantity (<= quantity).
        """
        adv = self._adv_cache.get(ticker, 1_000_000)
        if available_liquidity is None:
            available_liquidity = adv * self.params.daily_liquidity_factor

        # If order > available liquidity, partially fill
        if quantity <= available_liquidity:
            return quantity
        else:
            # Fill available_liquidity plus a random partial fraction of the rest
            extra_pct = np.random.uniform(0.1, 0.4)
            filled = available_liquidity + extra_pct * (quantity - available_liquidity)
            return math.floor(min(filled, quantity))

    def limit_fill_probability(self, limit_price: float, mid_price: float, direction: Literal["BUY", "SELL"]) -> float:
        """
        Probability of a limit order filling based on price vs mid.

        BUY limit: if limit >= ask → prob=1.0; if limit << mid → low prob
        SELL limit: if limit <= bid → prob=1.0; if limit >> mid → low prob
        """
        bid, ask = self.compute_spread("_generic", mid_price)
        if direction == "BUY":
            if limit_price >= ask:
                return 1.0
            elif limit_price >= mid_price:
                return 0.7
            elif limit_price >= bid:
                return 0.4
            else:
                edge = max(0.0, mid_price - limit_price)
                return max(self.params.min_fill_probability, math.exp(-edge / mid_price * 100))
        else:  # SELL
            if limit_price <= bid:
                return 1.0
            elif limit_price <= mid_price:
                return 0.7
            elif limit_price <= ask:
                return 0.4
            else:
                edge = max(0.0, limit_price - mid_price)
                return max(self.params.min_fill_probability, math.exp(-edge / mid_price * 100))

    def simulate_fill_v2(
        self,
        order: OrderEvent,
        next_bar: dict,
        order_type_v2: OrderTypeV2 = "MARKET",
        hidden: bool = False,
        commission_pct: float = 0.001,
    ) -> tuple[FillEvent | None, float]:
        """
        Fill simulation with microstructure effects.

        Returns (FillEvent | None, actual_filled_quantity).
        Partial fills return FillEvent with reduced quantity.
        """
        if not next_bar or order.quantity <= 0:
            return None, 0.0

        mid_price = next_bar.get("open", next_bar.get("close", 0.0))
        if mid_price <= 0:
            return None, 0.0

        ticker = order.ticker

        # Determine filled quantity (partial fill for large orders)
        volume = next_bar.get("volume", self._adv_cache.get(ticker, 1_000_000))
        self._adv_cache.setdefault(ticker, volume)
        available = volume * self.params.daily_liquidity_factor
        filled_qty = self.compute_partial_fill(ticker, order.quantity, available)

        if filled_qty < 1.0:
            return None, 0.0

        # Limit order fill probability check
        if order.order_type == "LIMIT" and order.limit_price is not None:
            prob = self.limit_fill_probability(order.limit_price, mid_price, order.direction)
            if np.random.random() > prob:
                return None, 0.0
            mid_price = order.limit_price  # fill at limit

        # Bid-ask spread effect
        bid, ask = self.compute_spread(ticker, mid_price)
        if order.direction == "BUY":
            base_fill = ask
        else:
            base_fill = bid

        # Market impact
        temp_impact, perm_impact = self.compute_market_impact(
            ticker, order.direction, filled_qty, mid_price, hidden=hidden
        )

        # Direction of impact on fill price
        impact_sign = 1 if order.direction == "BUY" else -1
        fill_price = base_fill + impact_sign * (temp_impact + perm_impact)

        commission = fill_price * filled_qty * commission_pct

        ts = next_bar.get("timestamp", order.timestamp + timedelta(days=1))
        fill = FillEvent(
            timestamp=ts,
            event_type="FILL",
            ticker=ticker,
            direction=order.direction,
            quantity=filled_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(fill_price - mid_price) * filled_qty,
            order_id=order.order_id,
        )
        return fill, filled_qty


# ---------------------------------------------------------------------------
# PortfolioConstraintEngine
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    max_gross_exposure: float = 2.0       # e.g. 2.0 = 200% gross
    max_net_exposure: float = 1.0         # e.g. 1.0 = 100% net long
    max_sector_concentration: float = 0.30  # max 30% in any sector
    beta_target: float = 0.0             # 0.0 = not enforced, else target beta
    beta_tolerance: float = 0.2          # ±0.2 around target beta
    max_position_correlation: float = 0.8  # max allowed pairwise correlation
    max_daily_var_pct: float = 0.02       # 2% portfolio VaR at 99% confidence
    var_confidence: float = 0.99
    max_single_position_pct: float = 0.15 # 15% max in single position


class PortfolioConstraintEngine:
    """
    Enforces portfolio-level constraints before orders are placed.

    Checks:
    - Gross / net exposure limits
    - Sector concentration limits
    - Beta neutrality
    - Position correlation caps
    - Daily VaR budget
    """

    def __init__(self, config: ConstraintConfig | None = None) -> None:
        self.config = config or ConstraintConfig()
        self._sector_map: dict[str, str] = {}         # ticker → sector
        self._beta_map: dict[str, float] = {}         # ticker → beta vs market
        self._returns_cache: dict[str, pd.Series] = {}  # ticker → recent returns

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        self._sector_map = sector_map

    def set_beta_map(self, beta_map: dict[str, float]) -> None:
        self._beta_map = beta_map

    def update_returns(self, ticker: str, returns: pd.Series) -> None:
        self._returns_cache[ticker] = returns.tail(60)

    def _compute_gross_net(self, positions: dict[str, float], prices: dict[str, float], equity: float) -> tuple[float, float]:
        if equity <= 0:
            return 0.0, 0.0
        long_val = sum(max(q, 0) * prices.get(t, 0) for t, q in positions.items())
        short_val = sum(abs(min(q, 0)) * prices.get(t, 0) for t, q in positions.items())
        gross = (long_val + short_val) / equity
        net = (long_val - short_val) / equity
        return gross, net

    def _compute_sector_concentration(self, positions: dict[str, float], prices: dict[str, float], equity: float) -> dict[str, float]:
        sector_values: dict[str, float] = {}
        for ticker, qty in positions.items():
            sector = self._sector_map.get(ticker, "Unknown")
            val = abs(qty) * prices.get(ticker, 0)
            sector_values[sector] = sector_values.get(sector, 0.0) + val
        if equity <= 0:
            return {}
        return {s: v / equity for s, v in sector_values.items()}

    def _compute_portfolio_beta(self, positions: dict[str, float], prices: dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total_beta = 0.0
        for ticker, qty in positions.items():
            weight = qty * prices.get(ticker, 0) / equity
            beta = self._beta_map.get(ticker, 1.0)
            total_beta += weight * beta
        return total_beta

    def _compute_var(self, positions: dict[str, float], prices: dict[str, float], equity: float) -> float:
        """Historical simulation VaR using cached returns."""
        if equity <= 0 or not positions:
            return 0.0
        weights: dict[str, float] = {}
        for ticker, qty in positions.items():
            weights[ticker] = qty * prices.get(ticker, 0) / equity

        tickers_with_data = [t for t in weights if t in self._returns_cache and len(self._returns_cache[t]) >= 20]
        if not tickers_with_data:
            return 0.0

        # Build return matrix
        min_len = min(len(self._returns_cache[t]) for t in tickers_with_data)
        ret_matrix = np.column_stack([self._returns_cache[t].values[-min_len:] for t in tickers_with_data])
        w = np.array([weights[t] for t in tickers_with_data])
        portfolio_returns = ret_matrix @ w

        var_pct = float(np.percentile(portfolio_returns, (1 - self.config.var_confidence) * 100))
        return abs(var_pct)  # Return as positive number (loss)

    def _max_pairwise_correlation(self, positions: dict[str, float]) -> float:
        tickers = list(positions.keys())
        if len(tickers) < 2:
            return 0.0
        pairs_checked = 0
        max_corr = 0.0
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                r1 = self._returns_cache.get(t1)
                r2 = self._returns_cache.get(t2)
                if r1 is None or r2 is None or len(r1) < 10 or len(r2) < 10:
                    continue
                min_len = min(len(r1), len(r2))
                corr = abs(float(np.corrcoef(r1.values[-min_len:], r2.values[-min_len:])[0, 1]))
                max_corr = max(max_corr, corr)
                pairs_checked += 1
        return max_corr

    def check_order(
        self,
        signal: SignalEvent,
        proposed_qty: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> tuple[float, list[str]]:
        """
        Validate a proposed order against all constraints.

        Returns (allowed_quantity, [list of constraint violations]).
        allowed_quantity may be reduced (or 0) to satisfy constraints.
        """
        violations: list[str] = []
        prices = dict(portfolio._last_prices)
        positions = dict(portfolio.positions)
        equity = portfolio._compute_equity()

        if equity <= 0:
            return proposed_qty, []

        # Simulate new position
        if signal.signal_type == "LONG":
            test_positions = dict(positions)
            test_positions[signal.ticker] = positions.get(signal.ticker, 0) + proposed_qty
        elif signal.signal_type == "SHORT":
            test_positions = dict(positions)
            test_positions[signal.ticker] = positions.get(signal.ticker, 0) - proposed_qty
        else:
            return proposed_qty, []

        test_prices = dict(prices)
        test_prices[signal.ticker] = current_price

        # --- Gross / net exposure ---
        gross, net = self._compute_gross_net(test_positions, test_prices, equity)
        if gross > self.config.max_gross_exposure:
            violations.append(f"Gross exposure {gross:.2f}x > max {self.config.max_gross_exposure:.2f}x")
            proposed_qty = 0.0

        if abs(net) > self.config.max_net_exposure:
            violations.append(f"Net exposure {net:.2f}x > max {self.config.max_net_exposure:.2f}x")
            proposed_qty = 0.0

        # --- Single position size ---
        pos_val = proposed_qty * current_price / equity
        if pos_val > self.config.max_single_position_pct:
            max_qty = math.floor(equity * self.config.max_single_position_pct / current_price)
            violations.append(f"Position size {pos_val:.1%} > max {self.config.max_single_position_pct:.1%}; reduced to {max_qty}")
            proposed_qty = max_qty

        # --- Sector concentration ---
        if self._sector_map:
            sector_conc = self._compute_sector_concentration(test_positions, test_prices, equity)
            for sector, conc in sector_conc.items():
                if conc > self.config.max_sector_concentration:
                    violations.append(f"Sector {sector} concentration {conc:.1%} > max {self.config.max_sector_concentration:.1%}")
                    proposed_qty = 0.0

        # --- Beta neutrality ---
        if self.config.beta_target != 0.0 and self._beta_map:
            port_beta = self._compute_portfolio_beta(test_positions, test_prices, equity)
            if abs(port_beta - self.config.beta_target) > self.config.beta_tolerance:
                violations.append(
                    f"Portfolio beta {port_beta:.2f} deviates from target {self.config.beta_target:.2f} "
                    f"by more than tolerance {self.config.beta_tolerance:.2f}"
                )

        # --- Correlation cap ---
        if positions:
            max_corr = self._max_pairwise_correlation(test_positions)
            if max_corr > self.config.max_position_correlation:
                violations.append(f"Max pairwise correlation {max_corr:.2f} > {self.config.max_position_correlation:.2f}")

        # --- VaR budget ---
        port_var = self._compute_var(test_positions, test_prices, equity)
        if port_var > self.config.max_daily_var_pct:
            violations.append(f"Portfolio VaR {port_var:.2%} > budget {self.config.max_daily_var_pct:.2%}")
            proposed_qty = 0.0

        return max(0.0, math.floor(proposed_qty)), violations

    def get_portfolio_risk_report(self, portfolio: Portfolio) -> dict:
        """Generate a current risk snapshot."""
        prices = dict(portfolio._last_prices)
        positions = dict(portfolio.positions)
        equity = portfolio._compute_equity()

        gross, net = self._compute_gross_net(positions, prices, equity)
        sector_conc = self._compute_sector_concentration(positions, prices, equity)
        port_beta = self._compute_portfolio_beta(positions, prices, equity) if self._beta_map else None
        port_var = self._compute_var(positions, prices, equity)
        max_corr = self._max_pairwise_correlation(positions)

        return {
            "equity": round(equity, 2),
            "n_positions": len(positions),
            "gross_exposure": round(gross, 4),
            "net_exposure": round(net, 4),
            "sector_concentration": {s: round(v, 4) for s, v in sector_conc.items()},
            "portfolio_beta": round(port_beta, 4) if port_beta is not None else None,
            "daily_var_99": round(port_var, 4),
            "max_pairwise_correlation": round(max_corr, 4),
        }


# ---------------------------------------------------------------------------
# MultiStrategyBacktester
# ---------------------------------------------------------------------------

@dataclass
class StrategyAllocation:
    strategy: BaseStrategy
    capital_weight: float       # fraction of total capital (sum must ≤ 1.0)
    regime_filter: Optional[Callable[[str], bool]] = None   # regime → bool (run in this regime?)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.strategy.name


@dataclass
class MultiStrategyResult:
    """Combined result from multi-strategy backtest."""
    combined_equity_curve: pd.Series
    strategy_equity_curves: dict[str, pd.Series]
    strategy_metrics: dict[str, dict]
    combined_metrics: dict
    capital_allocation: dict[str, float]
    regime_performance: dict[str, dict]
    correlation_matrix: pd.DataFrame
    attribution: pd.DataFrame  # date × strategy P&L


def _compute_regime(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Classify market regime from a returns series.

    Regimes: 'bull', 'bear', 'sideways', 'crisis'
    Uses rolling return + volatility to classify.
    """
    roll_ret = returns.rolling(window).mean() * 252        # annualised rolling mean
    roll_vol = returns.rolling(window).std() * math.sqrt(252)  # annualised vol

    regime = pd.Series("sideways", index=returns.index)
    crisis_vol_threshold = roll_vol.quantile(0.85) if len(roll_vol.dropna()) > 10 else 0.3
    bull_mask = (roll_ret > 0.05) & (roll_vol < crisis_vol_threshold)
    bear_mask = (roll_ret < -0.05) & (roll_vol < crisis_vol_threshold)
    crisis_mask = roll_vol >= crisis_vol_threshold

    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"
    regime[crisis_mask] = "crisis"
    return regime


class MultiStrategyBacktester:
    """
    Run multiple strategies simultaneously with capital allocation.

    Features:
    - Per-strategy capital pools (isolated P&L)
    - Strategy correlation monitoring
    - Regime-based strategy activation
    - Combined P&L and attribution
    """

    def __init__(
        self,
        total_capital: float = 1_000_000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        rebalance_frequency: int = 21,   # bars between rebalance
    ) -> None:
        self.total_capital = total_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.rebalance_frequency = rebalance_frequency
        self._allocations: list[StrategyAllocation] = []
        self._market_returns: pd.Series | None = None  # benchmark returns for regime

    def add_strategy(self, allocation: StrategyAllocation) -> None:
        self._allocations.append(allocation)

    def set_market_returns(self, returns: pd.Series) -> None:
        """Set market benchmark returns for regime detection."""
        self._market_returns = returns

    def _build_bar_timeline(self, data: dict[str, pd.DataFrame], start: str, end: str) -> list[tuple]:
        timeline = []
        s_dt = pd.Timestamp(start)
        e_dt = pd.Timestamp(end)
        for ticker, df in data.items():
            sub = df.loc[s_dt:e_dt]
            for ts, row in sub.iterrows():
                bar = row.to_dict()
                bar["timestamp"] = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                timeline.append((bar["timestamp"], ticker, bar))
        timeline.sort(key=lambda x: (x[0], x[1]))
        return timeline

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> MultiStrategyResult:
        """
        Run all strategies over the period and aggregate results.
        """
        if not self._allocations:
            raise RuntimeError("No strategies added. Call add_strategy() first.")

        # Validate weights
        total_weight = sum(a.capital_weight for a in self._allocations)
        if total_weight > 1.01:
            raise ValueError(f"Total capital weights {total_weight:.2f} > 1.0")

        # Build per-strategy portfolios
        strategy_portfolios: dict[str, Portfolio] = {}
        for alloc in self._allocations:
            cap = self.total_capital * alloc.capital_weight
            strategy_portfolios[alloc.name] = Portfolio(
                initial_capital=cap,
                position_sizing="equal_weight",
                max_positions=20,
            )

        timeline = self._build_bar_timeline(data, start_date, end_date)
        logger.info("MultiStrategy backtest: %d bars, %d strategies", len(timeline), len(self._allocations))

        # Per-strategy equity tracking
        strat_equity: dict[str, list[tuple[datetime, float]]] = {a.name: [] for a in self._allocations}
        combined_equity: list[tuple[datetime, float]] = []

        from sentinel.sbx.event_driven_backtest import ExecutionHandler
        execution = ExecutionHandler(self.commission_pct, self.slippage_pct)

        last_bar_per_ticker: dict[str, dict] = {}

        def bars_after(ticker: str, after_ts: datetime) -> dict | None:
            df = data.get(ticker)
            if df is None:
                return None
            future = df.loc[pd.Timestamp(after_ts):]
            for ts in future.index:
                if ts.to_pydatetime() > after_ts:
                    row = future.loc[ts].to_dict()
                    row["timestamp"] = ts.to_pydatetime()
                    return row
            return None

        # Build market regime series if available
        market_regime: pd.Series | None = None
        if self._market_returns is not None:
            market_regime = _compute_regime(self._market_returns)

        def get_regime(ts: datetime) -> str:
            if market_regime is None:
                return "unknown"
            pt = pd.Timestamp(ts)
            if pt in market_regime.index:
                return market_regime.loc[pt]
            # Nearest
            idx = market_regime.index.get_indexer([pt], method="nearest")[0]
            return market_regime.iloc[idx] if idx >= 0 else "unknown"

        attribution_rows: list[dict] = []
        bar_count = 0

        for ts, ticker, bar in timeline:
            last_bar_per_ticker[ticker] = bar
            current_regime = get_regime(ts)

            mde = MarketDataEvent(
                timestamp=ts,
                event_type="MARKET_DATA",
                ticker=ticker,
                open=bar.get("open", 0),
                high=bar.get("high", 0),
                low=bar.get("low", 0),
                close=bar.get("close", 0),
                volume=bar.get("volume", 0),
            )

            row_pnl: dict[str, float] = {"timestamp": ts}
            combined_val = 0.0

            for alloc in self._allocations:
                # Regime filter
                if alloc.regime_filter is not None and not alloc.regime_filter(current_regime):
                    portfolio = strategy_portfolios[alloc.name]
                    portfolio.update_on_bar({ticker: bar})
                    val = portfolio._compute_equity()
                    row_pnl[alloc.name] = 0.0
                    combined_val += val
                    strat_equity[alloc.name].append((ts, val))
                    continue

                portfolio = strategy_portfolios[alloc.name]
                portfolio.update_on_bar({ticker: bar})

                try:
                    alloc.strategy._register_bar(mde)
                    snap = portfolio.compute_snapshot(ts)
                    signals = alloc.strategy.on_bar(mde, snap) or []
                except Exception as exc:
                    logger.debug("Strategy %s error: %s", alloc.name, exc)
                    signals = []

                for signal in signals:
                    cur_price = bar.get("close", 0)
                    qty = portfolio.size_order(signal, cur_price, len(signals))
                    if qty <= 0:
                        continue
                    direction = "BUY" if signal.signal_type == "LONG" else "SELL"
                    if signal.signal_type == "EXIT":
                        qty = abs(portfolio.positions.get(signal.ticker, 0))
                        direction = "SELL"
                    if qty <= 0:
                        continue

                    order = OrderEvent(
                        timestamp=ts,
                        event_type="ORDER",
                        ticker=signal.ticker,
                        order_type="MARKET",
                        direction=direction,
                        quantity=qty,
                    )
                    nb = bars_after(ticker, ts)
                    if nb is None:
                        continue
                    fill = execution.simulate_fill(order, nb)
                    if fill:
                        portfolio.update_on_fill(fill)
                        try:
                            alloc.strategy.on_fill(fill)
                        except Exception:
                            pass

                val = portfolio._compute_equity()
                strat_equity[alloc.name].append((ts, val))
                row_pnl[alloc.name] = val
                combined_val += val

            combined_equity.append((ts, combined_val))
            attribution_rows.append(row_pnl)
            bar_count += 1

        # Build output series
        strategy_curves: dict[str, pd.Series] = {}
        for name, eq_list in strat_equity.items():
            if eq_list:
                idx = pd.DatetimeIndex([t for t, _ in eq_list])
                vals = [v for _, v in eq_list]
                strategy_curves[name] = pd.Series(vals, index=idx, name=name).resample("D").last().dropna()

        if combined_equity:
            comb_idx = pd.DatetimeIndex([t for t, _ in combined_equity])
            comb_vals = [v for _, v in combined_equity]
            combined_curve = pd.Series(comb_vals, index=comb_idx, name="combined").resample("D").last().dropna()
        else:
            combined_curve = pd.Series(dtype=float)

        # Compute per-strategy metrics
        strategy_metrics: dict[str, dict] = {}
        for name, curve in strategy_curves.items():
            m = self._compute_curve_metrics(name, curve, self.total_capital)
            strategy_metrics[name] = m

        combined_metrics = self._compute_curve_metrics("combined", combined_curve, self.total_capital)

        # Correlation matrix
        corr_df = self._compute_correlation_matrix(strategy_curves)

        # Regime performance breakdown
        regime_perf = self._compute_regime_performance(strategy_curves, market_regime)

        # Attribution dataframe
        attr_df = pd.DataFrame(attribution_rows).set_index("timestamp") if attribution_rows else pd.DataFrame()

        return MultiStrategyResult(
            combined_equity_curve=combined_curve,
            strategy_equity_curves=strategy_curves,
            strategy_metrics=strategy_metrics,
            combined_metrics=combined_metrics,
            capital_allocation={a.name: a.capital_weight for a in self._allocations},
            regime_performance=regime_perf,
            correlation_matrix=corr_df,
            attribution=attr_df,
        )

    @staticmethod
    def _compute_curve_metrics(name: str, curve: pd.Series, initial_capital: float) -> dict:
        if curve.empty or len(curve) < 2:
            return {"strategy": name, "error": "insufficient data"}
        rets = curve.pct_change().dropna()
        years = (curve.index[-1] - curve.index[0]).days / 365.25 or 1
        total_ret = (curve.iloc[-1] / curve.iloc[0]) - 1
        cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
        roll_max = curve.cummax()
        drawdown = (curve - roll_max) / roll_max
        max_dd = float(drawdown.min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        return {
            "strategy": name,
            "total_return_pct": round(total_ret * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "years": round(years, 2),
        }

    @staticmethod
    def _compute_correlation_matrix(curves: dict[str, pd.Series]) -> pd.DataFrame:
        if len(curves) < 2:
            return pd.DataFrame()
        rets = pd.DataFrame({name: curve.pct_change() for name, curve in curves.items()}).dropna()
        return rets.corr().round(4)

    @staticmethod
    def _compute_regime_performance(
        curves: dict[str, pd.Series],
        regime: pd.Series | None,
    ) -> dict[str, dict]:
        if regime is None or not curves:
            return {}
        result: dict[str, dict] = {}
        for reg in ["bull", "bear", "sideways", "crisis"]:
            reg_idx = regime[regime == reg].index
            if len(reg_idx) == 0:
                continue
            result[reg] = {}
            for name, curve in curves.items():
                sub = curve[curve.index.isin(reg_idx)]
                if len(sub) < 5:
                    result[reg][name] = {"sharpe": None, "total_return": None}
                    continue
                rets = sub.pct_change().dropna()
                sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
                total_ret = float(sub.iloc[-1] / sub.iloc[0] - 1) if sub.iloc[0] > 0 else 0.0
                result[reg][name] = {"sharpe": round(sharpe, 3), "total_return_pct": round(total_ret * 100, 2)}
        return result


# ---------------------------------------------------------------------------
# BacktestAnalyticsV2
# ---------------------------------------------------------------------------

class BacktestAnalyticsV2:
    """
    Enhanced analytics suite for backtest results.

    Provides:
    - Monthly / quarterly P&L heatmap data
    - Rolling Sharpe ratio and max drawdown series
    - Win/loss streaks
    - Best/worst trades (top 10 each)
    - Holding period distribution
    """

    def __init__(self, window_sharpe: int = 63, window_dd: int = 21, ann: int = 252) -> None:
        self.window_sharpe = window_sharpe
        self.window_dd = window_dd
        self.ann = ann

    def monthly_pnl_heatmap(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Build monthly returns matrix: rows = years, cols = months (1-12).

        Returns a DataFrame suitable for heatmap rendering.
        """
        if equity_curve.empty or len(equity_curve) < 5:
            return pd.DataFrame()

        ec = equity_curve.sort_index().resample("ME").last().dropna()
        monthly_rets = ec.pct_change().dropna()

        df = monthly_rets.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ][:len(pivot.columns)]
        pivot["Annual"] = pivot.sum(axis=1)
        return pivot.round(4)

    def quarterly_pnl(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Quarterly return aggregation."""
        if equity_curve.empty:
            return pd.DataFrame()
        ec = equity_curve.sort_index().resample("QE").last().dropna()
        rets = ec.pct_change().dropna()
        df = rets.to_frame("quarterly_return")
        df["year"] = df.index.year
        df["quarter"] = df.index.quarter
        df["label"] = df["year"].astype(str) + " Q" + df["quarter"].astype(str)
        return df[["label", "quarterly_return"]].reset_index(drop=True)

    def rolling_sharpe(self, equity_curve: pd.Series) -> pd.Series:
        """Rolling annualised Sharpe ratio over window_sharpe bars."""
        if equity_curve.empty or len(equity_curve) < self.window_sharpe:
            return pd.Series(dtype=float)
        rets = equity_curve.pct_change().dropna()
        roll_mean = rets.rolling(self.window_sharpe).mean()
        roll_std = rets.rolling(self.window_sharpe).std()
        sharpe = roll_mean / roll_std * math.sqrt(self.ann)
        sharpe.name = "rolling_sharpe"
        return sharpe.dropna()

    def rolling_max_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Rolling maximum drawdown over a trailing window."""
        if equity_curve.empty or len(equity_curve) < self.window_dd:
            return pd.Series(dtype=float)
        roll_max = equity_curve.rolling(self.window_dd, min_periods=5).max()
        drawdown = (equity_curve - roll_max) / roll_max
        drawdown.name = "rolling_max_drawdown"
        return drawdown.dropna()

    def win_loss_streaks(self, trades: pd.DataFrame) -> dict:
        """
        Compute win and loss streaks from the trades DataFrame.

        Expects a 'pnl' column.
        Returns dict with max_win_streak, max_loss_streak, current_streak, streak_type.
        """
        if trades.empty or "pnl" not in trades.columns:
            return {"max_win_streak": 0, "max_loss_streak": 0, "current_streak": 0, "streak_type": "none"}

        pnl = trades.sort_values("entry_ts")["pnl"] if "entry_ts" in trades.columns else trades["pnl"]
        wins = (pnl > 0).astype(int)

        max_win = max_loss = cur_streak = 0
        streak_type = "none"
        win_count = loss_count = 0

        for i, w in enumerate(wins):
            if w == 1:
                loss_count = 0
                win_count += 1
                max_win = max(max_win, win_count)
            else:
                win_count = 0
                loss_count += 1
                max_loss = max(max_loss, loss_count)

        if wins.iloc[-1] == 1:
            cur_streak = win_count
            streak_type = "win"
        else:
            cur_streak = loss_count
            streak_type = "loss"

        return {
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "current_streak": cur_streak,
            "streak_type": streak_type,
        }

    def best_worst_trades(self, trades: pd.DataFrame, n: int = 10) -> dict:
        """Return top N best and worst trades by P&L."""
        if trades.empty or "pnl" not in trades.columns:
            return {"best": [], "worst": []}

        sorted_trades = trades.sort_values("pnl", ascending=False)
        cols = [c for c in ["ticker", "entry_ts", "exit_ts", "entry_price", "exit_price", "quantity", "pnl", "holding_days"] if c in trades.columns]
        best = sorted_trades.head(n)[cols].to_dict(orient="records")
        worst = sorted_trades.tail(n)[cols].to_dict(orient="records")
        worst.reverse()  # worst first
        return {"best": best, "worst": worst}

    def holding_period_distribution(self, trades: pd.DataFrame, bins: int = 20) -> dict:
        """
        Compute distribution of trade holding periods.

        Returns dict with histogram data (bin_edges, counts, percentiles).
        """
        if trades.empty or "holding_days" not in trades.columns:
            return {"bins": [], "counts": [], "percentiles": {}}

        hp = trades["holding_days"].dropna()
        if len(hp) == 0:
            return {"bins": [], "counts": [], "percentiles": {}}

        counts, bin_edges = np.histogram(hp, bins=min(bins, len(hp)))
        percentiles = {
            "p10": round(float(np.percentile(hp, 10)), 1),
            "p25": round(float(np.percentile(hp, 25)), 1),
            "p50": round(float(np.percentile(hp, 50)), 1),
            "p75": round(float(np.percentile(hp, 75)), 1),
            "p90": round(float(np.percentile(hp, 90)), 1),
            "mean": round(float(hp.mean()), 1),
            "std": round(float(hp.std()), 1),
        }
        return {
            "bins": [round(float(e), 1) for e in bin_edges],
            "counts": counts.tolist(),
            "percentiles": percentiles,
        }

    def full_report(self, equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
        """Generate all analytics in one call."""
        monthly_heatmap = self.monthly_pnl_heatmap(equity_curve)
        quarterly = self.quarterly_pnl(equity_curve)
        rolling_sh = self.rolling_sharpe(equity_curve)
        rolling_mdd = self.rolling_max_drawdown(equity_curve)
        streaks = self.win_loss_streaks(trades)
        bw = self.best_worst_trades(trades)
        hp_dist = self.holding_period_distribution(trades)

        return {
            "monthly_heatmap": monthly_heatmap.to_dict() if not monthly_heatmap.empty else {},
            "quarterly_returns": quarterly.to_dict(orient="records") if not quarterly.empty else [],
            "rolling_sharpe": {
                "dates": [str(d)[:10] for d in rolling_sh.index],
                "values": rolling_sh.round(3).tolist(),
            } if not rolling_sh.empty else {},
            "rolling_max_drawdown": {
                "dates": [str(d)[:10] for d in rolling_mdd.index],
                "values": rolling_mdd.round(4).tolist(),
            } if not rolling_mdd.empty else {},
            "streaks": streaks,
            "best_trades": bw["best"],
            "worst_trades": bw["worst"],
            "holding_period_distribution": hp_dist,
        }


# ---------------------------------------------------------------------------
# Enhanced EventDrivenBacktesterV2
# ---------------------------------------------------------------------------

class EventDrivenBacktesterV2:
    """
    Enhanced backtester with microstructure, advanced orders, and constraints.

    Wraps the base backtester with the three new engines:
      - MarketMicrostructureModel for realistic fills
      - AdvancedOrderTypes for brackets, trailing stops, algo execution
      - PortfolioConstraintEngine for risk-based order gating
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_pct: float = 0.001,
        microstructure_params: MicrostructureParams | None = None,
        constraint_config: ConstraintConfig | None = None,
        sector_map: dict[str, str] | None = None,
        beta_map: dict[str, float] | None = None,
        max_positions: int = 20,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self._raw_data: dict[str, pd.DataFrame] = {}
        self._strategies: list[BaseStrategy] = []

        self.microstructure = MarketMicrostructureModel(microstructure_params)
        self.constraints = PortfolioConstraintEngine(constraint_config)
        self.advanced_orders = AdvancedOrderTypes()
        self.analytics = BacktestAnalyticsV2()

        if sector_map:
            self.constraints.set_sector_map(sector_map)
        if beta_map:
            self.constraints.set_beta_map(beta_map)

        self._portfolio = Portfolio(
            initial_capital=initial_capital,
            position_sizing="equal_weight",
            max_positions=max_positions,
        )

    def load_data(self, ticker_data: dict[str, pd.DataFrame]) -> None:
        for ticker, df in ticker_data.items():
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{ticker} missing columns: {missing}")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            self._raw_data[ticker] = df.sort_index()

            # Warm up microstructure ADV cache
            self.microstructure.update_adv(ticker, df["volume"])

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)

    def _bars_after(self, ticker: str, after_ts: datetime) -> dict | None:
        df = self._raw_data.get(ticker)
        if df is None:
            return None
        for ts in df.index:
            if ts.to_pydatetime() > after_ts:
                row = df.loc[ts].to_dict()
                row["timestamp"] = ts.to_pydatetime()
                return row
        return None

    def run(self, start_date: str, end_date: str, use_microstructure: bool = True) -> dict:
        """
        Run enhanced backtest.

        Returns dict with:
          - equity_curve (pd.Series)
          - trades (pd.DataFrame)
          - metrics (dict)
          - analytics (dict) — full BacktestAnalyticsV2 report
          - risk_snapshots (list[dict])
        """
        if not self._strategies:
            raise RuntimeError("No strategies added.")
        if not self._raw_data:
            raise RuntimeError("No data loaded.")

        timeline = []
        s_dt = pd.Timestamp(start_date)
        e_dt = pd.Timestamp(end_date)
        for ticker, df in self._raw_data.items():
            sub = df.loc[s_dt:e_dt]
            for ts, row in sub.iterrows():
                bar = row.to_dict()
                bar["timestamp"] = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                timeline.append((bar["timestamp"], ticker, bar))
        timeline.sort(key=lambda x: (x[0], x[1]))

        logger.info("BacktesterV2: %d bars, %d strategies", len(timeline), len(self._strategies))

        all_fills: list[FillEvent] = []
        portfolio = self._portfolio
        risk_snapshots: list[dict] = []
        bar_num = 0

        for ts, ticker, bar in timeline:
            portfolio.update_on_bar({ticker: bar})

            # Update ATR for trailing stops
            hist = list(self._raw_data[ticker].loc[:pd.Timestamp(ts)].tail(20).itertuples())
            if len(hist) >= 5:
                highs = pd.Series([r.high for r in hist])
                lows = pd.Series([r.low for r in hist])
                closes = pd.Series([r.close for r in hist])
                self.advanced_orders.update_atr(ticker, highs, lows, closes)

            # Update returns cache for constraints
            ret_series = self._raw_data[ticker]["close"].pct_change().dropna()
            self.constraints.update_returns(ticker, ret_series)

            # Check trailing stop triggers
            triggered_tsos = self.advanced_orders.on_bar(
                ticker, bar.get("open", 0), bar.get("high", 0),
                bar.get("low", 0), bar.get("close", 0)
            )
            for tso in triggered_tsos:
                # Convert triggered trailing stop to exit fill
                nb = self._bars_after(ticker, ts)
                if nb:
                    exit_price = tso.stop_price if tso.stop_price > 0 else nb.get("open", bar.get("close", 0))
                    fill = FillEvent(
                        timestamp=nb.get("timestamp", ts + timedelta(days=1)),
                        event_type="FILL",
                        ticker=ticker,
                        direction="SELL" if tso.direction == "SELL" else "BUY",
                        quantity=tso.quantity,
                        fill_price=exit_price,
                        commission=exit_price * tso.quantity * self.commission_pct,
                        slippage=0.0,
                        order_id=tso.order_id,
                    )
                    portfolio.update_on_fill(fill)
                    all_fills.append(fill)

            # Check active bracket exits
            for bracket in self.advanced_orders.get_active_brackets():
                if bracket.ticker == ticker:
                    trigger = self.advanced_orders.check_bracket_exits(
                        bracket.bracket_id, bar.get("high", 0), bar.get("low", 0)
                    )
                    if trigger:
                        exit_price = bracket.take_profit_price if trigger == "TP" else bracket.stop_loss_price
                        nb = self._bars_after(ticker, ts)
                        fill_ts = nb.get("timestamp", ts + timedelta(days=1)) if nb else ts + timedelta(days=1)
                        fill = FillEvent(
                            timestamp=fill_ts,
                            event_type="FILL",
                            ticker=ticker,
                            direction="SELL",
                            quantity=bracket.quantity,
                            fill_price=exit_price,
                            commission=exit_price * bracket.quantity * self.commission_pct,
                            slippage=0.0,
                            order_id=bracket.entry_order_id,
                        )
                        portfolio.update_on_fill(fill)
                        all_fills.append(fill)

            # Strategy signals
            snap = portfolio.compute_snapshot(ts)
            mde = MarketDataEvent(
                timestamp=ts,
                event_type="MARKET_DATA",
                ticker=ticker,
                open=bar.get("open", 0),
                high=bar.get("high", 0),
                low=bar.get("low", 0),
                close=bar.get("close", 0),
                volume=bar.get("volume", 0),
            )

            signals: list[SignalEvent] = []
            for strategy in self._strategies:
                try:
                    strategy._register_bar(mde)
                    sigs = strategy.on_bar(mde, snap)
                    signals.extend(sigs or [])
                except Exception as exc:
                    logger.debug("Strategy error %s/%s: %s", ticker, ts, exc)

            for signal in signals:
                cur_price = bar.get("close", 0)
                if cur_price <= 0:
                    continue

                if signal.signal_type == "EXIT":
                    qty = abs(portfolio.positions.get(signal.ticker, 0))
                    direction = "SELL"
                else:
                    qty = portfolio.size_order(signal, cur_price, max(len(signals), 1))
                    direction = "BUY" if signal.signal_type == "LONG" else "SELL"

                if qty <= 0:
                    continue

                # Constraint check
                allowed_qty, viols = self.constraints.check_order(signal, qty, cur_price, portfolio)
                if viols:
                    logger.debug("Constraint violations for %s: %s", signal.ticker, viols)
                if allowed_qty <= 0:
                    continue

                order = OrderEvent(
                    timestamp=ts,
                    event_type="ORDER",
                    ticker=signal.ticker,
                    order_type="MARKET",
                    direction=direction,
                    quantity=allowed_qty,
                )

                nb = self._bars_after(signal.ticker, ts)
                if nb is None:
                    continue

                if use_microstructure:
                    fill, _ = self.microstructure.simulate_fill_v2(
                        order, nb, commission_pct=self.commission_pct
                    )
                else:
                    from sentinel.sbx.event_driven_backtest import ExecutionHandler
                    eh = ExecutionHandler(self.commission_pct, 0.0005)
                    fill = eh.simulate_fill(order, nb)

                if fill:
                    portfolio.update_on_fill(fill)
                    all_fills.append(fill)
                    for strategy in self._strategies:
                        try:
                            strategy.on_fill(fill)
                        except Exception:
                            pass

            # Risk snapshot every 20 bars
            bar_num += 1
            if bar_num % 20 == 0:
                risk_snapshots.append(self.constraints.get_portfolio_risk_report(portfolio))

        # Build results
        equity_data = portfolio.equity_series
        if equity_data:
            eq_series = pd.Series(
                {ts: eq for ts, eq in equity_data},
                name="equity",
            )
            eq_series.index = pd.to_datetime(eq_series.index)
        else:
            eq_series = pd.Series(dtype=float, name="equity")

        trades_df = self._build_trades_df(all_fills)
        strategy_names = ", ".join(s.name for s in self._strategies)

        from sentinel.sbx.event_driven_backtest import BacktestResult
        result = BacktestResult(
            strategy_name=strategy_names,
            initial_capital=self.initial_capital,
            equity_curve=eq_series,
            trades=trades_df,
        )
        metrics = result.compute_metrics()
        analytics_report = self.analytics.full_report(eq_series, trades_df)

        return {
            "strategy": strategy_names,
            "equity_curve": eq_series,
            "trades": trades_df,
            "metrics": metrics,
            "analytics": analytics_report,
            "risk_snapshots": risk_snapshots,
            "n_fills": len(all_fills),
        }

    def _build_trades_df(self, fills: list[FillEvent]) -> pd.DataFrame:
        """Match BUY→SELL fills into round trips."""
        if not fills:
            return pd.DataFrame(columns=[
                "ticker", "direction", "entry_price", "exit_price",
                "quantity", "pnl", "holding_days", "entry_ts", "exit_ts", "commission",
            ])
        buys: dict[str, list[FillEvent]] = {}
        rows = []
        for fill in fills:
            if fill.direction == "BUY":
                buys.setdefault(fill.ticker, []).append(fill)
            else:
                queue = buys.get(fill.ticker, [])
                if not queue:
                    continue
                entry_fill = queue.pop(0)
                pnl = (fill.fill_price - entry_fill.fill_price) * fill.quantity
                pnl -= fill.commission + entry_fill.commission
                holding = (fill.timestamp - entry_fill.timestamp).days
                rows.append({
                    "ticker": fill.ticker,
                    "direction": "LONG",
                    "entry_price": round(entry_fill.fill_price, 4),
                    "exit_price": round(fill.fill_price, 4),
                    "quantity": fill.quantity,
                    "pnl": round(pnl, 2),
                    "holding_days": holding,
                    "entry_ts": entry_fill.timestamp,
                    "exit_ts": fill.timestamp,
                    "commission": round(fill.commission + entry_fill.commission, 2),
                    "entry_slippage": round(entry_fill.slippage, 2),
                    "exit_slippage": round(fill.slippage, 2),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# In-memory result store (for API run_id retrieval)
# ---------------------------------------------------------------------------

_RESULT_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel as PydanticModel, Field as PydanticField

    backtest_v2_router = APIRouter(prefix="/backtest/v2", tags=["Backtest V2"])

    class BacktestV2Request(PydanticModel):
        tickers: list[str] = PydanticField(default_factory=list)
        start_date: str = "2020-01-01"
        end_date: str = "2024-12-31"
        initial_capital: float = 1_000_000
        commission_pct: float = 0.001
        strategy: str = "momentum"   # momentum | mean_reversion | ma_cross | buy_hold
        use_microstructure: bool = True
        max_positions: int = 20
        max_gross_exposure: float = 2.0
        max_sector_concentration: float = 0.30
        max_daily_var_pct: float = 0.02

    class MultiStrategyRequest(PydanticModel):
        tickers: list[str] = PydanticField(default_factory=list)
        start_date: str = "2020-01-01"
        end_date: str = "2024-12-31"
        total_capital: float = 1_000_000
        commission_pct: float = 0.001
        strategy_weights: dict[str, float] = PydanticField(
            default_factory=lambda: {"momentum": 0.4, "mean_reversion": 0.3, "ma_cross": 0.3}
        )

    class AnalyzeRequest(PydanticModel):
        run_id: str
        window_sharpe: int = 63
        window_drawdown: int = 21

    def _strategy_factory(name: str) -> BaseStrategy:
        mapping = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "ma_cross": MovingAverageCrossStrategy,
            "buy_hold": BuyAndHoldStrategy,
        }
        cls = mapping.get(name.lower(), BuyAndHoldStrategy)
        return cls()

    def _fetch_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data via yfinance."""
        import yfinance as yf
        result = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                result[ticker] = df
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", ticker, exc)
        return result

    @backtest_v2_router.post("/run")
    async def api_run_backtest_v2(req: BacktestV2Request):
        """Run a single-strategy enhanced backtest."""
        if not req.tickers:
            raise HTTPException(status_code=422, detail="tickers required")

        data = _fetch_data(req.tickers, req.start_date, req.end_date)
        if not data:
            raise HTTPException(status_code=400, detail="Could not fetch data for any ticker")

        constraint_cfg = ConstraintConfig(
            max_gross_exposure=req.max_gross_exposure,
            max_sector_concentration=req.max_sector_concentration,
            max_daily_var_pct=req.max_daily_var_pct,
        )

        backtester = EventDrivenBacktesterV2(
            initial_capital=req.initial_capital,
            commission_pct=req.commission_pct,
            constraint_config=constraint_cfg,
            max_positions=req.max_positions,
        )
        backtester.load_data(data)
        backtester.add_strategy(_strategy_factory(req.strategy))

        try:
            result = backtester.run(req.start_date, req.end_date, use_microstructure=req.use_microstructure)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        run_id = str(uuid.uuid4())[:12]
        # Serialize for storage (equity curve as list)
        serialised = {
            "strategy": result["strategy"],
            "metrics": result["metrics"],
            "n_fills": result["n_fills"],
            "analytics": {
                k: v for k, v in result["analytics"].items()
                if k not in ("monthly_heatmap",)  # skip non-JSON-serializable complex dicts
            },
            "equity_curve_dates": [str(d)[:10] for d in result["equity_curve"].index],
            "equity_curve_values": result["equity_curve"].tolist(),
            "risk_snapshots": result["risk_snapshots"][:5],  # first 5
        }
        _RESULT_STORE[run_id] = serialised

        return {"run_id": run_id, **serialised}

    @backtest_v2_router.post("/multi-strategy")
    async def api_multi_strategy(req: MultiStrategyRequest):
        """Run multiple strategies simultaneously with capital allocation."""
        if not req.tickers:
            raise HTTPException(status_code=422, detail="tickers required")

        data = _fetch_data(req.tickers, req.start_date, req.end_date)
        if not data:
            raise HTTPException(status_code=400, detail="Could not fetch data")

        ms_backtester = MultiStrategyBacktester(
            total_capital=req.total_capital,
            commission_pct=req.commission_pct,
        )

        for name, weight in req.strategy_weights.items():
            strategy = _strategy_factory(name)
            alloc = StrategyAllocation(strategy=strategy, capital_weight=weight, name=name)
            ms_backtester.add_strategy(alloc)

        try:
            result = ms_backtester.run(data, req.start_date, req.end_date)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        run_id = str(uuid.uuid4())[:12]
        out = {
            "run_id": run_id,
            "combined_metrics": result.combined_metrics,
            "strategy_metrics": result.strategy_metrics,
            "capital_allocation": result.capital_allocation,
            "regime_performance": result.regime_performance,
            "correlation_matrix": result.correlation_matrix.to_dict() if not result.correlation_matrix.empty else {},
            "combined_equity_dates": [str(d)[:10] for d in result.combined_equity_curve.index],
            "combined_equity_values": result.combined_equity_curve.tolist(),
        }
        _RESULT_STORE[run_id] = out
        return out

    @backtest_v2_router.get("/result/{run_id}")
    async def api_get_result(run_id: str):
        """Retrieve a previously computed backtest result."""
        result = _RESULT_STORE.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Result {run_id} not found")
        return result

    @backtest_v2_router.post("/analyze")
    async def api_analyze(req: AnalyzeRequest):
        """Run enhanced analytics on an existing backtest result."""
        result = _RESULT_STORE.get(req.run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Result {req.run_id} not found")

        if "equity_curve_dates" not in result or "equity_curve_values" not in result:
            raise HTTPException(status_code=400, detail="Result has no equity curve data")

        dates = pd.to_datetime(result["equity_curve_dates"])
        vals = result["equity_curve_values"]
        equity_curve = pd.Series(vals, index=dates)

        analytics = BacktestAnalyticsV2(
            window_sharpe=req.window_sharpe,
            window_dd=req.window_drawdown,
        )

        rolling_sh = analytics.rolling_sharpe(equity_curve)
        rolling_mdd = analytics.rolling_max_drawdown(equity_curve)
        monthly = analytics.monthly_pnl_heatmap(equity_curve)
        quarterly = analytics.quarterly_pnl(equity_curve)

        return {
            "run_id": req.run_id,
            "rolling_sharpe": {
                "dates": [str(d)[:10] for d in rolling_sh.index],
                "values": rolling_sh.round(3).tolist(),
            },
            "rolling_max_drawdown": {
                "dates": [str(d)[:10] for d in rolling_mdd.index],
                "values": rolling_mdd.round(4).tolist(),
            },
            "monthly_heatmap": monthly.round(4).to_dict() if not monthly.empty else {},
            "quarterly_returns": quarterly.to_dict(orient="records") if not quarterly.empty else [],
        }

except ImportError:
    backtest_v2_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available — backtest_v2_router not registered")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def create_bracket_strategy_example(
    ticker: str,
    capital: float = 100_000,
    take_profit_pct: float = 0.05,
    stop_loss_pct: float = 0.02,
) -> dict:
    """
    Example demonstrating bracket order creation.

    Returns a dict with order details and expected TP/SL prices at hypothetical entry.
    """
    entry_price = 100.0  # hypothetical
    tp_price = entry_price * (1 + take_profit_pct)
    sl_price = entry_price * (1 - stop_loss_pct)
    qty = math.floor(capital * 0.1 / entry_price)

    mgr = AdvancedOrderTypes()
    bracket = mgr.create_bracket(
        ticker=ticker,
        direction="BUY",
        quantity=qty,
        entry_price=entry_price,
        take_profit_price=tp_price,
        stop_loss_price=sl_price,
        tif="GTC",
    )

    return {
        "bracket_id": bracket.bracket_id,
        "ticker": ticker,
        "quantity": qty,
        "entry_price": entry_price,
        "take_profit_price": round(tp_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "max_gain_per_share": round(tp_price - entry_price, 2),
        "max_loss_per_share": round(entry_price - sl_price, 2),
        "risk_reward_ratio": round(take_profit_pct / stop_loss_pct, 2),
    }


def quick_backtest_v2(
    ticker_data: dict[str, pd.DataFrame],
    start: str,
    end: str,
    strategy: BaseStrategy | None = None,
    capital: float = 1_000_000,
    use_microstructure: bool = True,
) -> dict:
    """
    One-liner enhanced backtest.

    Defaults to BuyAndHoldStrategy with microstructure simulation.
    """
    strategy = strategy or BuyAndHoldStrategy()
    bt = EventDrivenBacktesterV2(
        initial_capital=capital,
        microstructure_params=MicrostructureParams(base_spread_bps=5.0),
    )
    bt.load_data(ticker_data)
    bt.add_strategy(strategy)
    return bt.run(start, end, use_microstructure=use_microstructure)
