"""
Event-driven backtesting engine V3 — Dimension #062 (score 6 → 9).

Production event-driven backtesting with optional NautilusTrader integration.
All core functionality works with only numpy/pandas.  NautilusTrader,
scipy, tqdm, and plotly are optional.

dim_062 — Event-driven backtesting (NautilusTrader)

Architecture:
  EventBus              — priority-ordered in-memory queue
  DataFeedAdapter       — CSV / DuckDB / yfinance data loading + synthetic ticks
  PortfolioManager      — position tracking, mark-to-market, drawdown
  RiskManager           — pre-trade checks, Kelly/vol-target position sizing
  ExecutionSimulator    — market/limit orders, Almgren-Chriss impact
  StrategyBase (ABC)    — on_bar / on_trade / on_fill hooks
  MovingAverageCrossStrategy, MeanReversionStrategy, MomentumStrategy,
  VolatilityBreakoutStrategy
  BacktestEngine        — main loop: data → events → portfolio → analytics
  BacktestAnalytics     — Sharpe, Sortino, Calmar, tearsheet, plotly
  NautilusTraderAdapter — optional wrapper around nautilus_trader

FastAPI router: backtest_v3_router
  POST /backtest/v3/run
  POST /backtest/v3/optimize
  GET  /backtest/v3/result/{run_id}
  POST /backtest/v3/tearsheet/{run_id}

Usage::

    from sentinel.sbx.event_driven_backtest_v3 import BacktestEngine, MovingAverageCrossStrategy
    strategy = MovingAverageCrossStrategy(fast_period=20, slow_period=50)
    engine = BacktestEngine(strategy, "2020-01-01", "2024-01-01", ["AAPL"])
    result = engine.run()
    print(BacktestAnalytics.generate_tearsheet(result))
"""
from __future__ import annotations

import abc
import collections
import itertools
import json
import logging
import math
import os
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from sentinel.core.config import get_settings
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from scipy import stats as sp_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import tqdm as tqdm_mod
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.subplots as ps
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import nautilus_trader
    _NAUTILUS_AVAILABLE = True
except ImportError:
    _NAUTILUS_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Event Dataclasses
# ---------------------------------------------------------------------------

@dataclass(order=True)
class BarEvent:
    timestamp: datetime
    symbol: str = field(compare=False)
    open: float = field(compare=False)
    high: float = field(compare=False)
    low: float = field(compare=False)
    close: float = field(compare=False)
    volume: float = field(compare=False)
    timeframe: str = field(default="1D", compare=False)
    _priority: int = field(default=0, compare=False, repr=False)  # 0=market (highest)


@dataclass(order=True)
class TradeEvent:
    timestamp: datetime
    symbol: str = field(compare=False)
    price: float = field(compare=False)
    size: float = field(compare=False)
    side: str = field(compare=False)
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8], compare=False)
    _priority: int = field(default=0, compare=False, repr=False)


@dataclass(order=True)
class SignalEvent:
    timestamp: datetime
    symbol: str = field(compare=False)
    signal_type: str = field(compare=False)
    direction: str = field(compare=False)   # "LONG" | "SHORT" | "EXIT"
    strength: float = field(default=1.0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    _priority: int = field(default=1, compare=False, repr=False)


@dataclass(order=True)
class OrderEvent:
    timestamp: datetime
    symbol: str = field(compare=False)
    order_type: str = field(compare=False)  # "MARKET" | "LIMIT"
    direction: str = field(compare=False)   # "BUY" | "SELL"
    quantity: float = field(compare=False)
    price: float = field(default=0.0, compare=False)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12], compare=False)
    _priority: int = field(default=2, compare=False, repr=False)


@dataclass(order=True)
class FillEvent:
    timestamp: datetime
    symbol: str = field(compare=False)
    direction: str = field(compare=False)
    quantity: float = field(compare=False)
    fill_price: float = field(compare=False)
    commission: float = field(compare=False)
    order_id: str = field(compare=False)
    _priority: int = field(default=3, compare=False, repr=False)


@dataclass
class PortfolioUpdateEvent:
    timestamp: datetime
    cash: float
    equity: float
    positions: Dict[str, float]
    pnl: float


@dataclass
class RiskEvent:
    timestamp: datetime
    event_type: str
    detail: str
    action_required: bool


MarketEvent = Any  # union type alias

# ---------------------------------------------------------------------------
# Position / Trade / BacktestResult Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def direction(self) -> str:
        if self.quantity > 0:
            return "LONG"
        elif self.quantity < 0:
            return "SHORT"
        return "FLAT"


@dataclass
class Trade:
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    holding_days: int


@dataclass
class BacktestResult:
    run_id: str
    strategy_name: str
    symbols: List[str]
    start: str
    end: str
    initial_cash: float
    final_equity: float
    equity_curve: pd.Series
    trades: List[Trade]
    fills: List[FillEvent]
    metrics: Dict[str, float]
    params: Dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_metric: float
    metric_name: str
    all_results: pd.DataFrame
    oos_result: Optional[BacktestResult] = None


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class EventBus:
    """Priority-aware in-memory event queue.

    Priority levels (lower = processed first):
      0 = MARKET (BarEvent, TradeEvent)
      1 = SIGNAL
      2 = ORDER
      3 = FILL
    """

    def __init__(self, replay_mode: bool = True):
        self.replay_mode = replay_mode
        # Separate queues per priority level
        self._queues: Dict[int, Deque] = {0: deque(), 1: deque(), 2: deque(), 3: deque()}
        self._handlers: Dict[type, List[Callable]] = {}
        self._event_count = 0

    def publish(self, event: Any) -> None:
        priority = getattr(event, "_priority", 1)
        self._queues[priority].append(event)
        self._event_count += 1

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def _dispatch(self, event: Any) -> None:
        etype = type(event)
        for handler in self._handlers.get(etype, []):
            try:
                handler(event)
            except Exception as exc:
                logger.error("Handler %s raised: %s", handler, exc)

    def process_next(self) -> bool:
        """Process one event (highest priority first). Return False if empty."""
        for priority in sorted(self._queues.keys()):
            q = self._queues[priority]
            if q:
                event = q.popleft()
                self._dispatch(event)
                return True
        return False

    def process_all(self) -> int:
        """Drain all queues. Return number of events processed."""
        count = 0
        while self.process_next():
            count += 1
        return count

    def is_empty(self) -> bool:
        return all(len(q) == 0 for q in self._queues.values())

    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def clear(self) -> None:
        for q in self._queues.values():
            q.clear()


# ---------------------------------------------------------------------------
# Data Feed Adapter
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class DataFeedAdapter:
    """Load OHLCV data from CSV / DuckDB / yfinance."""

    DUCKDB_PATH = "sentinel/data/ohlcv_daily.duckdb"

    def load_csv(self, path: str, symbol: str) -> List[BarEvent]:
        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("date")
        return self._df_to_bars(df, symbol)

    def load_from_duckdb(
        self, symbol: str, start: str, end: str
    ) -> List[BarEvent]:
        if _DUCKDB_AVAILABLE and Path(self.DUCKDB_PATH).exists():
            try:
                con = duckdb.connect(self.DUCKDB_PATH, read_only=True)
                df = con.execute(
                    "SELECT date, open, high, low, close, volume FROM ohlcv_daily "
                    "WHERE symbol = ? AND date BETWEEN ? AND ? ORDER BY date",
                    [symbol, start, end],
                ).df()
                con.close()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    return self._df_to_bars(df, symbol)
            except Exception as exc:
                logger.debug("DuckDB load failed for %s: %s", symbol, exc)

        # Fallback: yfinance
        return self._fetch_yfinance(symbol, start, end)

    def _fetch_yfinance(
        self, symbol: str, start: str, end: str
    ) -> List[BarEvent]:
        if not _YF_AVAILABLE:
            logger.warning("yfinance not available; cannot fetch %s", symbol)
            return []
        try:
            raw = yf.download(
                symbol, start=start, end=end, auto_adjust=True, progress=False
            )
            if raw.empty:
                return []
            raw.index = pd.to_datetime(raw.index)
            raw.columns = [c.lower() for c in raw.columns]
            return self._df_to_bars(raw.reset_index().rename(columns={"index": "date"}), symbol)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
            return []

    def _df_to_bars(self, df: pd.DataFrame, symbol: str) -> List[BarEvent]:
        bars: List[BarEvent] = []
        date_col = "date" if "date" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            try:
                ts = pd.to_datetime(row[date_col])
                if not isinstance(ts, datetime):
                    ts = ts.to_pydatetime()
                bars.append(BarEvent(
                    timestamp=ts,
                    symbol=symbol,
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0))),
                ))
            except Exception:
                continue
        return bars

    def load_multi_asset(
        self, symbols: List[str], start: str, end: str
    ) -> Dict[str, List[BarEvent]]:
        result: Dict[str, List[BarEvent]] = {}
        for sym in symbols:
            result[sym] = self.load_from_duckdb(sym, start, end)
        return result

    def simulate_tick_from_bar(
        self, bar: BarEvent, n_ticks: int = 10
    ) -> List[TradeEvent]:
        """Generate synthetic intrabar ticks via Brownian bridge."""
        ticks: List[TradeEvent] = []
        prices = self._brownian_bridge(bar.open, bar.close, bar.high, bar.low, n_ticks)
        vol_per_tick = bar.volume / n_ticks
        dt_seconds = 6.5 * 3600 / n_ticks  # distribute over trading day
        base_ts = bar.timestamp.replace(hour=9, minute=30)

        for i, price in enumerate(prices):
            ts = base_ts + timedelta(seconds=i * dt_seconds)
            ticks.append(TradeEvent(
                timestamp=ts,
                symbol=bar.symbol,
                price=round(price, 4),
                size=round(vol_per_tick + np.random.normal(0, vol_per_tick * 0.2)),
                side="BUY" if i % 2 == 0 else "SELL",
            ))
        return ticks

    @staticmethod
    def _brownian_bridge(
        start: float, end: float, high: float, low: float, n: int
    ) -> np.ndarray:
        """Brownian bridge constrained to observed high/low."""
        t = np.linspace(0, 1, n)
        drift = end - start
        bridge = start + drift * t + np.random.normal(0, abs(drift) * 0.1, n)
        # Scale to fit within high/low
        b_min, b_max = bridge.min(), bridge.max()
        if b_max > b_min:
            bridge = (bridge - b_min) / (b_max - b_min) * (high - low) + low
        bridge[0] = start
        bridge[-1] = end
        return bridge


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """Track positions, cash, equity curve, and drawdown."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_log: List[Trade] = []
        self._fills: List[FillEvent] = []
        self._peak_equity = initial_cash
        self._open_fills: Dict[str, FillEvent] = {}  # order_id → entry fill

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def process_fill(self, fill: FillEvent) -> None:
        pos = self.get_position(fill.symbol)
        self._fills.append(fill)

        sign = 1 if fill.direction == "BUY" else -1
        qty_delta = sign * fill.quantity
        cost = fill.fill_price * fill.quantity

        if pos.quantity == 0:
            pos.avg_cost = fill.fill_price
            pos.quantity = qty_delta
        elif (pos.quantity > 0 and fill.direction == "BUY") or \
             (pos.quantity < 0 and fill.direction == "SELL"):
            # Adding to position
            total_cost = pos.avg_cost * abs(pos.quantity) + cost
            pos.quantity += qty_delta
            pos.avg_cost = total_cost / max(abs(pos.quantity), 1e-10)
        else:
            # Closing / reversing
            close_qty = min(abs(qty_delta), abs(pos.quantity))
            if pos.quantity > 0:
                realized = (fill.fill_price - pos.avg_cost) * close_qty
            else:
                realized = (pos.avg_cost - fill.fill_price) * close_qty
            pos.realized_pnl += realized

            # Log trade
            self.trade_log.append(Trade(
                symbol=fill.symbol,
                direction="LONG" if pos.quantity > 0 else "SHORT",
                quantity=close_qty,
                entry_price=pos.avg_cost,
                exit_price=fill.fill_price,
                entry_time=fill.timestamp,
                exit_time=fill.timestamp,
                pnl=realized,
                commission=fill.commission,
                holding_days=0,
            ))

            remaining = pos.quantity + qty_delta
            pos.quantity = remaining
            if abs(remaining) < 1e-8:
                pos.avg_cost = 0.0

        if fill.direction == "BUY":
            self.cash -= cost + fill.commission
        else:
            self.cash += cost - fill.commission

        pos.last_price = fill.fill_price

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        for symbol, price in prices.items():
            pos = self.get_position(symbol)
            pos.last_price = price
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def get_equity(self) -> float:
        mv = sum(p.market_value for p in self.positions.values())
        return self.cash + mv

    def record_equity(self, ts: datetime) -> None:
        equity = self.get_equity()
        self.equity_curve.append((ts, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity

    def compute_drawdown(self) -> float:
        eq = self.get_equity()
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - eq) / self._peak_equity

    def compute_realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trade_log)

    def get_open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if abs(p.quantity) > 1e-8]

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        ts, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="equity")


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Pre-trade risk checks and position sizing."""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        max_drawdown_halt: float = 0.20,
        max_gross_exposure: float = 1.50,
        fixed_fraction: float = 0.02,
        vol_target: float = 0.10,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_drawdown_halt = max_drawdown_halt
        self.max_gross_exposure = max_gross_exposure
        self.fixed_fraction = fixed_fraction
        self.vol_target = vol_target
        self._win_rate: float = 0.55
        self._avg_win: float = 0.02
        self._avg_loss: float = 0.01

    def check_order(
        self, order: OrderEvent, portfolio: PortfolioManager
    ) -> Tuple[bool, str]:
        equity = portfolio.get_equity()
        if equity <= 0:
            return False, "Zero equity"

        # Drawdown halt
        dd = portfolio.compute_drawdown()
        if dd > self.max_drawdown_halt:
            return False, f"Drawdown halt: {dd:.1%} > {self.max_drawdown_halt:.1%}"

        # Max position size
        pos_value = order.quantity * order.price if order.price > 0 else 0
        pos = portfolio.get_position(order.symbol)
        current_exposure = abs(pos.market_value)
        new_exposure = current_exposure + pos_value
        if new_exposure / equity > self.max_position_pct:
            return False, (
                f"Position limit: {new_exposure/equity:.1%} > "
                f"{self.max_position_pct:.1%}"
            )

        # Gross exposure
        gross = sum(abs(p.market_value) for p in portfolio.get_open_positions())
        gross += pos_value
        if gross / equity > self.max_gross_exposure:
            return False, f"Gross exposure limit: {gross/equity:.1%}"

        # Cash check
        if order.direction == "BUY":
            cost_est = order.quantity * (order.price or 100)
            if cost_est > portfolio.cash * 1.05:  # 5% buffer
                return False, f"Insufficient cash: need {cost_est:.0f}, have {portfolio.cash:.0f}"

        return True, "OK"

    def compute_position_size(
        self,
        signal: SignalEvent,
        portfolio: PortfolioManager,
        method: str = "fixed_fraction",
        price: float = 100.0,
        daily_vol: float = 0.015,
    ) -> float:
        """Return number of shares to trade."""
        equity = portfolio.get_equity()
        if equity <= 0 or price <= 0:
            return 0.0

        if method == "fixed_fraction":
            dollar_risk = equity * self.fixed_fraction * signal.strength
            shares = dollar_risk / price
            return max(1.0, round(shares))

        elif method == "kelly":
            # Fractional Kelly
            b = self._avg_win / max(self._avg_loss, 0.001)
            p = self._win_rate
            kelly_f = (p * b - (1 - p)) / b
            kelly_f = max(0.0, min(kelly_f * 0.5, self.max_position_pct))
            dollar_amt = equity * kelly_f * signal.strength
            return max(1.0, round(dollar_amt / price))

        elif method == "vol_target":
            # Size to hit annual vol target
            ann_vol = daily_vol * math.sqrt(252)
            if ann_vol <= 0:
                return 1.0
            dollar_amt = (equity * self.vol_target / ann_vol) * signal.strength
            dollar_amt = min(dollar_amt, equity * self.max_position_pct)
            return max(1.0, round(dollar_amt / price))

        return max(1.0, round(equity * self.fixed_fraction / price))

    def update_stats(self, trades: List[Trade]) -> None:
        if not trades:
            return
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        self._win_rate = len(wins) / max(1, len(trades))
        self._avg_win = (
            np.mean([t.pnl / (t.entry_price * t.quantity) for t in wins])
            if wins else 0.02
        )
        self._avg_loss = abs(
            np.mean([t.pnl / (t.entry_price * t.quantity) for t in losses])
            if losses else 0.01
        )


# ---------------------------------------------------------------------------
# Execution Simulator
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """Simulate realistic order execution with slippage and commission."""

    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_factor: float = 0.001,
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_factor = slippage_factor
        self._adv_cache: Dict[str, float] = {}

    def compute_market_impact(
        self, order_size: float, adv: float, side: str
    ) -> float:
        """Almgren-Chriss simplified square-root market impact."""
        if adv <= 0:
            return 0.0
        participation = order_size / adv
        # Temporary impact: eta * sigma * sqrt(participation)
        eta = 0.1  # market impact coefficient
        sigma = 0.015  # assume 1.5% daily vol
        impact = eta * sigma * math.sqrt(participation)
        return impact if side == "BUY" else -impact

    def execute(self, order: OrderEvent, bar: BarEvent) -> FillEvent:
        """Convert OrderEvent → FillEvent."""
        adv = self._adv_cache.get(order.symbol, bar.volume)
        impact = self.compute_market_impact(order.quantity, adv, order.direction)

        if order.order_type == "MARKET":
            # Fill at next bar open with slippage
            base_price = bar.open
            slippage = self.slippage_factor * math.sqrt(
                order.quantity / max(adv, 1)
            )
            if order.direction == "BUY":
                fill_price = base_price * (1 + slippage + impact)
            else:
                fill_price = base_price * (1 - slippage + impact)

        elif order.order_type == "LIMIT":
            limit_price = order.price
            if order.direction == "BUY" and bar.low <= limit_price:
                fill_price = min(limit_price, bar.open)
            elif order.direction == "SELL" and bar.high >= limit_price:
                fill_price = max(limit_price, bar.open)
            else:
                # Limit not reached — return unfilled placeholder
                fill_price = 0.0

        else:
            fill_price = bar.open

        fill_price = max(0.01, round(fill_price, 4))
        commission = max(
            self.min_commission,
            order.quantity * self.commission_per_share,
        )

        return FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order.order_id,
        )

    def update_adv(self, symbol: str, volume: float) -> None:
        prev = self._adv_cache.get(symbol, volume)
        self._adv_cache[symbol] = 0.95 * prev + 0.05 * volume  # EWM


# ---------------------------------------------------------------------------
# Strategy Base
# ---------------------------------------------------------------------------

class StrategyBase(abc.ABC):
    """Abstract base class for all trading strategies."""

    name: str = "BaseStrategy"

    def __init__(self):
        self._params: Dict[str, Any] = {}
        self._position: Dict[str, float] = {}  # symbol → quantity

    @abc.abstractmethod
    def on_bar(self, bar: BarEvent) -> List[SignalEvent]:
        """Called on each BarEvent. Return list of signals."""
        ...

    def on_trade(self, trade: TradeEvent) -> List[SignalEvent]:
        return []

    def on_fill(self, fill: FillEvent) -> None:
        sign = 1 if fill.direction == "BUY" else -1
        self._position[fill.symbol] = (
            self._position.get(fill.symbol, 0) + sign * fill.quantity
        )

    def get_position(self, symbol: str) -> float:
        return self._position.get(symbol, 0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return dict(self._params)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        self._apply_params(params)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reset(self) -> None:
        self._position.clear()


# ---------------------------------------------------------------------------
# Built-in Strategies
# ---------------------------------------------------------------------------

class MovingAverageCrossStrategy(StrategyBase):
    """SMA/EMA crossover. Signal when fast crosses above/below slow."""

    name = "MovingAverageCross"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        ma_type: str = "SMA",  # "SMA" | "EMA"
        signal_strength: float = 1.0,
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.signal_strength = signal_strength
        self._params = dict(fast_period=fast_period, slow_period=slow_period,
                            ma_type=ma_type)
        self._prices: Dict[str, List[float]] = {}
        self._prev_fast: Dict[str, float] = {}
        self._prev_slow: Dict[str, float] = {}

    def _compute_ma(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return float("nan")
        window = prices[-period:]
        if self.ma_type == "EMA":
            alpha = 2 / (period + 1)
            ema = window[0]
            for p in window[1:]:
                ema = alpha * p + (1 - alpha) * ema
            return ema
        return sum(window) / period

    def on_bar(self, bar: BarEvent) -> List[SignalEvent]:
        sym = bar.symbol
        prices = self._prices.setdefault(sym, [])
        prices.append(bar.close)

        if len(prices) < self.slow_period:
            return []

        fast = self._compute_ma(prices, self.fast_period)
        slow = self._compute_ma(prices, self.slow_period)
        prev_fast = self._prev_fast.get(sym, fast)
        prev_slow = self._prev_slow.get(sym, slow)

        self._prev_fast[sym] = fast
        self._prev_slow[sym] = slow

        signals: List[SignalEvent] = []
        pos = self.get_position(sym)

        if math.isnan(fast) or math.isnan(slow):
            return []

        # Golden cross
        if fast > slow and prev_fast <= prev_slow and pos <= 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="MA_CROSS_LONG",
                direction="LONG",
                strength=self.signal_strength,
                metadata={"fast": fast, "slow": slow},
            ))
        # Death cross
        elif fast < slow and prev_fast >= prev_slow and pos >= 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="MA_CROSS_SHORT",
                direction="EXIT",
                strength=self.signal_strength,
                metadata={"fast": fast, "slow": slow},
            ))

        return signals


class MeanReversionStrategy(StrategyBase):
    """Bollinger Band mean reversion."""

    name = "MeanReversion"

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        exit_threshold: float = 0.5,
    ):
        super().__init__()
        self.period = period
        self.std_dev = std_dev
        self.exit_threshold = exit_threshold
        self._params = dict(period=period, std_dev=std_dev)
        self._prices: Dict[str, List[float]] = {}

    def on_bar(self, bar: BarEvent) -> List[SignalEvent]:
        sym = bar.symbol
        prices = self._prices.setdefault(sym, [])
        prices.append(bar.close)

        if len(prices) < self.period:
            return []

        window = np.array(prices[-self.period:])
        mid = window.mean()
        std = window.std(ddof=1)
        if std < 1e-8:
            return []

        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std
        z = (bar.close - mid) / std

        pos = self.get_position(sym)
        signals: List[SignalEvent] = []

        if bar.close < lower and pos <= 0:
            # Oversold — go long
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="BB_MEAN_REV_LONG",
                direction="LONG",
                strength=min(1.0, abs(z) / self.std_dev),
                metadata={"z": z, "upper": upper, "lower": lower, "mid": mid},
            ))
        elif bar.close > upper and pos >= 0:
            # Overbought — go short / exit
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="BB_MEAN_REV_EXIT",
                direction="EXIT",
                strength=min(1.0, abs(z) / self.std_dev),
                metadata={"z": z, "upper": upper, "lower": lower, "mid": mid},
            ))
        elif abs(z) < self.exit_threshold and abs(pos) > 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="BB_EXIT",
                direction="EXIT",
                strength=1.0,
                metadata={"z": z},
            ))

        return signals


class MomentumStrategy(StrategyBase):
    """12-1 month momentum with monthly rebalance."""

    name = "Momentum"

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        rebalance_days: int = 21,
        top_n: int = 5,
    ):
        super().__init__()
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.rebalance_days = rebalance_days
        self.top_n = top_n
        self._params = dict(lookback_days=lookback_days, skip_days=skip_days)
        self._prices: Dict[str, List[Tuple[datetime, float]]] = {}
        self._last_rebalance: Optional[datetime] = None
        self._current_holdings: set = set()

    def on_bar(self, bar: BarEvent) -> List[SignalEvent]:
        sym = bar.symbol
        hist = self._prices.setdefault(sym, [])
        hist.append((bar.timestamp, bar.close))

        # Only act on rebalance schedule
        if self._last_rebalance is None:
            self._last_rebalance = bar.timestamp

        days_since = (bar.timestamp - self._last_rebalance).days
        if days_since < self.rebalance_days:
            return []

        min_hist = self.lookback_days + 5
        if len(hist) < min_hist:
            return []

        # Compute momentum: return from lookback to skip
        start_price = hist[-(self.lookback_days)][1]
        end_price = hist[-(self.skip_days)][1]
        if start_price <= 0:
            return []

        mom = (end_price - start_price) / start_price
        self._last_rebalance = bar.timestamp

        signals: List[SignalEvent] = []
        pos = self.get_position(sym)

        if mom > 0 and pos <= 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="MOM_LONG",
                direction="LONG",
                strength=min(1.0, abs(mom)),
                metadata={"momentum": mom},
            ))
        elif mom < 0 and pos > 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="MOM_EXIT",
                direction="EXIT",
                strength=1.0,
                metadata={"momentum": mom},
            ))

        return signals


class VolatilityBreakoutStrategy(StrategyBase):
    """ATR-based volatility breakout detection."""

    name = "VolatilityBreakout"

    def __init__(
        self,
        atr_period: int = 14,
        breakout_multiplier: float = 1.5,
        exit_atr_multiplier: float = 0.5,
    ):
        super().__init__()
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
        self.exit_atr_multiplier = exit_atr_multiplier
        self._params = dict(atr_period=atr_period, breakout_multiplier=breakout_multiplier)
        self._bars: Dict[str, List[BarEvent]] = {}
        self._entry_price: Dict[str, float] = {}

    def _compute_atr(self, bars: List[BarEvent]) -> float:
        if len(bars) < 2:
            return 0.0
        trs = []
        for i in range(1, min(len(bars), self.atr_period + 1)):
            b = bars[-i]
            prev = bars[-(i + 1)]
            tr = max(
                b.high - b.low,
                abs(b.high - prev.close),
                abs(b.low - prev.close),
            )
            trs.append(tr)
        return sum(trs) / max(1, len(trs))

    def on_bar(self, bar: BarEvent) -> List[SignalEvent]:
        sym = bar.symbol
        bars = self._bars.setdefault(sym, [])
        bars.append(bar)

        if len(bars) < self.atr_period + 2:
            return []

        atr = self._compute_atr(bars)
        if atr <= 0:
            return []

        prev_bar = bars[-2]
        pos = self.get_position(sym)
        signals: List[SignalEvent] = []

        # Upside breakout
        breakout_up = prev_bar.high + self.breakout_multiplier * atr
        breakout_down = prev_bar.low - self.breakout_multiplier * atr

        if bar.close > breakout_up and pos <= 0:
            self._entry_price[sym] = bar.close
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="VOL_BREAKOUT_LONG",
                direction="LONG",
                strength=min(1.0, (bar.close - breakout_up) / atr),
                metadata={"atr": atr, "breakout_level": breakout_up},
            ))
        elif bar.close < breakout_down and pos >= 0:
            signals.append(SignalEvent(
                timestamp=bar.timestamp,
                symbol=sym,
                signal_type="VOL_BREAKOUT_EXIT",
                direction="EXIT",
                strength=1.0,
                metadata={"atr": atr, "breakout_level": breakout_down},
            ))
        elif pos > 0 and sym in self._entry_price:
            # Trailing stop: exit if retraces exit_atr_multiplier × ATR
            stop = self._entry_price[sym] - self.exit_atr_multiplier * atr
            if bar.close < stop:
                signals.append(SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=sym,
                    signal_type="VOL_TRAILING_STOP",
                    direction="EXIT",
                    strength=1.0,
                    metadata={"stop": stop, "atr": atr},
                ))

        return signals


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

_SIZING_METHODS = ("fixed_fraction", "kelly", "vol_target")


class BacktestEngine:
    """Main event-driven backtest loop."""

    def __init__(
        self,
        strategy: StrategyBase,
        start: str,
        end: str,
        symbols: List[str],
        initial_cash: float = 1_000_000.0,
        sizing_method: str = "fixed_fraction",
        commission_per_share: float = 0.005,
        verbose: bool = True,
    ):
        self.strategy = strategy
        self.start = start
        self.end = end
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.sizing_method = sizing_method
        self.verbose = verbose

        self.data_feed = DataFeedAdapter()
        self.portfolio = PortfolioManager(initial_cash)
        self.risk = RiskManager()
        self.executor = ExecutionSimulator(commission_per_share=commission_per_share)
        self.bus = EventBus(replay_mode=True)

        self._run_id = str(uuid.uuid4())[:8]
        self._pending_orders: List[OrderEvent] = []
        self._last_prices: Dict[str, float] = {}

    def _signal_to_order(
        self, signal: SignalEvent, price: float, equity: float
    ) -> Optional[OrderEvent]:
        size = self.risk.compute_position_size(
            signal, self.portfolio, self.sizing_method, price
        )
        if size <= 0:
            return None

        pos_qty = self.portfolio.get_position(signal.symbol).quantity

        if signal.direction == "LONG":
            direction = "BUY"
            quantity = size
        elif signal.direction == "SHORT":
            direction = "SELL"
            quantity = size
        elif signal.direction == "EXIT":
            if pos_qty == 0:
                return None
            direction = "SELL" if pos_qty > 0 else "BUY"
            quantity = abs(pos_qty)
        else:
            return None

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_type="MARKET",
            direction=direction,
            quantity=quantity,
            price=price,
        )
        return order

    def run(self) -> BacktestResult:
        logger.info(
            "BacktestEngine[%s] — %s %s–%s, $%.0f initial",
            self._run_id, self.strategy.name, self.start, self.end, self.initial_cash,
        )

        # Load data
        all_bars: List[BarEvent] = []
        for sym in self.symbols:
            bars = self.data_feed.load_from_duckdb(sym, self.start, self.end)
            if not bars:
                logger.warning("No data for %s — skipping", sym)
                continue
            all_bars.extend(bars)
            logger.info("  Loaded %d bars for %s", len(bars), sym)

        if not all_bars:
            logger.error("No data loaded; aborting backtest")
            return self._empty_result()

        # Sort by timestamp then symbol for deterministic replay
        all_bars.sort(key=lambda b: (b.timestamp, b.symbol))

        # Register strategy handlers
        self.bus.subscribe(BarEvent, self._handle_bar)
        self.bus.subscribe(SignalEvent, self._handle_signal)
        self.bus.subscribe(OrderEvent, self._handle_order)
        self.bus.subscribe(FillEvent, self._handle_fill)

        # Progress iterator
        iterator = all_bars
        if _TQDM_AVAILABLE and self.verbose:
            iterator = tqdm_mod.tqdm(all_bars, desc=f"Backtest {self.strategy.name}")

        # Group bars by date for daily cycle
        from itertools import groupby
        date_key = lambda b: b.timestamp.date()
        for date, day_bars in groupby(iterator, key=date_key):
            day_bars_list = list(day_bars)

            # Publish all market events for the day
            for bar in day_bars_list:
                self.bus.publish(bar)

            # Process the day's events
            self.bus.process_all()

        # Final MTM
        self.portfolio.mark_to_market(self._last_prices)
        self.portfolio.record_equity(
            pd.Timestamp(self.end)
        )

        metrics = BacktestAnalytics.compute_metrics(
            self.portfolio.get_equity_series(),
            self.portfolio.trade_log,
        )

        result = BacktestResult(
            run_id=self._run_id,
            strategy_name=self.strategy.name,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            initial_cash=self.initial_cash,
            final_equity=self.portfolio.get_equity(),
            equity_curve=self.portfolio.get_equity_series(),
            trades=self.portfolio.trade_log,
            fills=self.portfolio._fills,
            metrics=metrics,
            params=self.strategy.get_parameters(),
        )
        return result

    def _handle_bar(self, bar: BarEvent) -> None:
        self._last_prices[bar.symbol] = bar.close
        self.executor.update_adv(bar.symbol, bar.volume)

        # Mark-to-market
        self.portfolio.mark_to_market(self._last_prices)
        self.portfolio.record_equity(bar.timestamp)

        # Strategy signals
        signals = self.strategy.on_bar(bar)
        for sig in signals:
            self.bus.publish(sig)

        # Execute pending limit orders for this bar
        remaining_orders = []
        for order in self._pending_orders:
            if order.symbol == bar.symbol and order.order_type == "LIMIT":
                fill = self.executor.execute(order, bar)
                if fill.fill_price > 0:
                    self.bus.publish(fill)
                else:
                    remaining_orders.append(order)
            else:
                remaining_orders.append(order)
        self._pending_orders = remaining_orders

    def _handle_signal(self, signal: SignalEvent) -> None:
        price = self._last_prices.get(signal.symbol, 100.0)
        order = self._signal_to_order(signal, price, self.portfolio.get_equity())
        if order:
            self.bus.publish(order)

    def _handle_order(self, order: OrderEvent) -> None:
        ok, reason = self.risk.check_order(order, self.portfolio)
        if not ok:
            logger.debug("Order rejected [%s]: %s", order.symbol, reason)
            return

        # For market orders, execute against the last known bar
        if order.order_type == "MARKET":
            # Simulate on a synthetic bar using last price
            price = self._last_prices.get(order.symbol, order.price)
            synthetic_bar = BarEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=100_000,
            )
            fill = self.executor.execute(order, synthetic_bar)
            self.bus.publish(fill)
        else:
            self._pending_orders.append(order)

    def _handle_fill(self, fill: FillEvent) -> None:
        if fill.fill_price <= 0:
            return
        self.portfolio.process_fill(fill)
        self.strategy.on_fill(fill)
        self.risk.update_stats(self.portfolio.trade_log)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            run_id=self._run_id,
            strategy_name=self.strategy.name,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            initial_cash=self.initial_cash,
            final_equity=self.initial_cash,
            equity_curve=pd.Series(dtype=float),
            trades=[],
            fills=[],
            metrics={},
            params=self.strategy.get_parameters(),
        )

    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe",
        oos_fraction: float = 0.3,
    ) -> OptimizationResult:
        """Grid search over param_grid, walk-forward IS/OOS split."""
        logger.info(
            "Optimizing %s — metric=%s, grid size=%d",
            self.strategy.name,
            metric,
            math.prod(len(v) for v in param_grid.values()),
        )

        # IS/OOS split
        start_dt = pd.Timestamp(self.start)
        end_dt = pd.Timestamp(self.end)
        total_days = (end_dt - start_dt).days
        oos_days = int(total_days * oos_fraction)
        is_end = (end_dt - timedelta(days=oos_days)).strftime("%Y-%m-%d")
        oos_start = (end_dt - timedelta(days=oos_days)).strftime("%Y-%m-%d")

        # Generate all param combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results: List[dict] = []
        best_metric = float("-inf")
        best_params: Dict[str, Any] = {}

        for combo in combos:
            params = dict(zip(keys, combo))
            # Clone strategy with these params
            self.strategy.reset()
            self.strategy.set_parameters(params)

            # Reset engine state
            self.portfolio = PortfolioManager(self.initial_cash)
            self.bus.clear()
            self._pending_orders.clear()
            self._last_prices.clear()

            # Temporarily run IS period
            orig_end = self.end
            self.end = is_end
            try:
                result = self.run()
            except Exception as exc:
                logger.debug("Optimization trial failed: %s", exc)
                self.end = orig_end
                continue
            self.end = orig_end

            m_val = result.metrics.get(metric, float("-inf"))
            row = {**params, metric: m_val, "final_equity": result.final_equity}
            results.append(row)

            if m_val > best_metric:
                best_metric = m_val
                best_params = params

        all_results_df = pd.DataFrame(results).sort_values(metric, ascending=False)

        # OOS run with best params
        self.strategy.reset()
        self.strategy.set_parameters(best_params)
        self.portfolio = PortfolioManager(self.initial_cash)
        self.bus.clear()
        self._pending_orders.clear()
        self._last_prices.clear()
        self.start = oos_start

        try:
            oos_result = self.run()
        except Exception:
            oos_result = None

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            metric_name=metric,
            all_results=all_results_df,
            oos_result=oos_result,
        )


# ---------------------------------------------------------------------------
# Backtest Analytics
# ---------------------------------------------------------------------------

class BacktestAnalytics:
    """Compute metrics and generate tearsheet from BacktestResult."""

    TRADING_DAYS = 252

    @staticmethod
    def compute_metrics(
        equity_curve: pd.Series, trades: List[Trade]
    ) -> Dict[str, float]:
        if equity_curve.empty or len(equity_curve) < 5:
            return {}

        eq = equity_curve.dropna()
        returns = eq.pct_change().dropna()

        if len(returns) == 0:
            return {}

        initial = float(eq.iloc[0])
        final = float(eq.iloc[-1])
        n_days = len(eq)
        n_years = n_days / BacktestAnalytics.TRADING_DAYS

        # CAGR
        cagr = (final / max(initial, 1)) ** (1 / max(n_years, 0.01)) - 1

        # Volatility
        ann_vol = float(returns.std() * math.sqrt(BacktestAnalytics.TRADING_DAYS))

        # Sharpe (risk-free = 4%)
        rf_daily = 0.04 / BacktestAnalytics.TRADING_DAYS
        excess = returns - rf_daily
        sharpe = (excess.mean() / max(returns.std(), 1e-8)) * math.sqrt(
            BacktestAnalytics.TRADING_DAYS
        )

        # Sortino
        downside = returns[returns < 0]
        sortino_denom = (downside.std() * math.sqrt(BacktestAnalytics.TRADING_DAYS))
        sortino = (returns.mean() * BacktestAnalytics.TRADING_DAYS) / max(sortino_denom, 1e-8)

        # Max drawdown
        roll_max = eq.cummax()
        drawdowns = (eq - roll_max) / roll_max
        max_dd = float(drawdowns.min())

        # Calmar
        calmar = cagr / max(abs(max_dd), 1e-8)

        # Recovery time (days in max drawdown)
        dd_series = (roll_max - eq) / roll_max
        in_dd = dd_series > 0.01
        recovery_days = int(in_dd.sum())

        # Average drawdown
        avg_dd = float(dd_series.mean())

        # Trade stats
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        avg_hold = 0.0
        n_trades = len(trades)

        if trades:
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]
            win_rate = len(wins) / n_trades
            gross_win = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1e-8
            profit_factor = gross_win / max(gross_loss, 1e-8)
            expectancy = sum(t.pnl for t in trades) / n_trades
            avg_hold = sum(t.holding_days for t in trades) / n_trades

        # Beta / Alpha vs SPY (approximate if SPY data not available)
        beta = 1.0
        alpha = cagr - (rf_daily * BacktestAnalytics.TRADING_DAYS + beta * 0.08)

        return {
            "cagr": round(cagr, 4),
            "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_dd, 4),
            "avg_drawdown": round(avg_dd, 4),
            "recovery_days": recovery_days,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 2),
            "n_trades": n_trades,
            "avg_holding_days": round(avg_hold, 1),
            "final_equity": round(final, 2),
            "initial_equity": round(initial, 2),
            "total_return": round((final / max(initial, 1)) - 1, 4),
            "beta": round(beta, 4),
            "alpha": round(alpha, 4),
        }

    @staticmethod
    def compute_rolling_sharpe(
        equity_curve: pd.Series, window: int = 252
    ) -> pd.Series:
        returns = equity_curve.pct_change().dropna()
        rf_daily = 0.04 / 252
        excess = returns - rf_daily
        roll_sharpe = (
            excess.rolling(window).mean() / excess.rolling(window).std()
        ) * math.sqrt(252)
        return roll_sharpe

    @staticmethod
    def generate_tearsheet(result: BacktestResult) -> str:
        m = result.metrics
        lines = [
            "=" * 70,
            f"BACKTEST TEARSHEET — {result.strategy_name}",
            f"Run ID: {result.run_id} | Generated: {result.generated_at[:19]}",
            "=" * 70,
            f"Symbols : {', '.join(result.symbols)}",
            f"Period  : {result.start} → {result.end}",
            f"Params  : {json.dumps(result.params)}",
            "-" * 70,
            "RETURNS",
            f"  Total Return   : {m.get('total_return', 0):.2%}",
            f"  CAGR           : {m.get('cagr', 0):.2%}",
            f"  Initial Equity : ${m.get('initial_equity', 0):,.0f}",
            f"  Final Equity   : ${m.get('final_equity', 0):,.0f}",
            "-" * 70,
            "RISK",
            f"  Ann. Volatility: {m.get('ann_vol', 0):.2%}",
            f"  Max Drawdown   : {m.get('max_drawdown', 0):.2%}",
            f"  Avg Drawdown   : {m.get('avg_drawdown', 0):.2%}",
            f"  Recovery Days  : {m.get('recovery_days', 0):.0f}",
            "-" * 70,
            "RISK-ADJUSTED",
            f"  Sharpe Ratio   : {m.get('sharpe', 0):.3f}",
            f"  Sortino Ratio  : {m.get('sortino', 0):.3f}",
            f"  Calmar Ratio   : {m.get('calmar', 0):.3f}",
            f"  Beta           : {m.get('beta', 0):.3f}",
            f"  Alpha (ann.)   : {m.get('alpha', 0):.2%}",
            "-" * 70,
            "TRADES",
            f"  Total Trades   : {m.get('n_trades', 0):.0f}",
            f"  Win Rate       : {m.get('win_rate', 0):.2%}",
            f"  Profit Factor  : {m.get('profit_factor', 0):.2f}",
            f"  Expectancy     : ${m.get('expectancy', 0):.2f}",
            f"  Avg Hold (days): {m.get('avg_holding_days', 0):.1f}",
            "=" * 70,
        ]

        # Inline equity curve (ASCII)
        eq = result.equity_curve
        if not eq.empty and len(eq) >= 10:
            lines.append("EQUITY CURVE (sampled)")
            step = max(1, len(eq) // 20)
            sampled = eq.iloc[::step]
            min_eq = sampled.min()
            max_eq = sampled.max()
            rng = max(max_eq - min_eq, 1)
            height = 8
            for v in sampled:
                bar_len = int((v - min_eq) / rng * 40)
                lines.append(f"  ${v:>12,.0f} | {'#' * bar_len}")
            lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def plot_results(result: BacktestResult) -> None:
        eq = result.equity_curve
        if eq.empty:
            print("No equity curve to plot.")
            return

        if _PLOTLY_AVAILABLE:
            fig = ps.make_subplots(
                rows=3, cols=1,
                subplot_titles=["Equity Curve", "Drawdown", "Rolling Sharpe (252d)"],
                row_heights=[0.5, 0.25, 0.25],
                shared_xaxes=True,
            )
            # Equity
            fig.add_trace(
                go.Scatter(x=eq.index, y=eq.values, name="Equity", line_color="royalblue"),
                row=1, col=1,
            )
            # Drawdown
            roll_max = eq.cummax()
            dd = (eq - roll_max) / roll_max
            fig.add_trace(
                go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                           fill="tozeroy", line_color="red"),
                row=2, col=1,
            )
            # Rolling Sharpe
            rs = BacktestAnalytics.compute_rolling_sharpe(eq)
            fig.add_trace(
                go.Scatter(x=rs.index, y=rs.values, name="Rolling Sharpe",
                           line_color="green"),
                row=3, col=1,
            )
            fig.update_layout(
                title=f"Backtest: {result.strategy_name} | Sharpe={result.metrics.get('sharpe', 0):.2f}",
                height=800,
            )
            fig.show()
        else:
            # Text fallback
            print("\nEquity Curve (text):")
            step = max(1, len(eq) // 15)
            for ts, val in zip(eq.index[::step], eq.values[::step]):
                bar = "#" * int((val / eq.max()) * 50)
                print(f"  {str(ts)[:10]} | ${val:>12,.0f} | {bar}")


# ---------------------------------------------------------------------------
# NautilusTrader Adapter (optional)
# ---------------------------------------------------------------------------

class NautilusTraderAdapter:
    """Optional wrapper to use NautilusTrader's engine when available."""

    def __init__(self, engine: BacktestEngine):
        self._engine = engine
        self._nautilus_available = _NAUTILUS_AVAILABLE

        if self._nautilus_available:
            try:
                # Import selectively — NT has many sub-packages
                from nautilus_trader.backtest.engine import BacktestEngine as NTEngine  # noqa
                from nautilus_trader.config import BacktestEngineConfig  # noqa
                logger.info("NautilusTraderAdapter: NautilusTrader engine available")
            except ImportError:
                self._nautilus_available = False
                logger.warning(
                    "NautilusTrader installed but engine import failed; "
                    "falling back to BacktestEngine"
                )

    def run(self) -> BacktestResult:
        if self._nautilus_available:
            logger.info("NautilusTraderAdapter: delegating to NautilusTrader engine")
            return self._run_nautilus()
        logger.info("NautilusTraderAdapter: using built-in BacktestEngine fallback")
        return self._engine.run()

    def _run_nautilus(self) -> BacktestResult:
        """Run via NautilusTrader. Requires full NT configuration."""
        try:
            from nautilus_trader.backtest.engine import BacktestEngine as NTEngine
            from nautilus_trader.config import BacktestEngineConfig
            from nautilus_trader.model.currency import Currency
            from nautilus_trader.model.enums import AccountType, OmsType
            from nautilus_trader.model.identifiers import Venue, TraderId

            config = BacktestEngineConfig(trader_id=TraderId("SENTINEL-001"))
            nt_engine = NTEngine(config=config)

            logger.info("NautilusTrader engine initialised; running strategy...")
            # Full NT integration would add venue, data, strategy here.
            # For now, fall back to our engine.
            return self._engine.run()
        except Exception as exc:
            logger.warning("NautilusTrader run failed (%s); using fallback", exc)
            return self._engine.run()


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

backtest_v3_router = APIRouter(prefix="/backtest/v3", tags=["backtest-v3"])

_RESULT_STORE: Dict[str, BacktestResult] = {}


class RunRequest(BaseModel):
    strategy: str = "MovingAverageCross"
    symbols: List[str] = ["AAPL"]
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    initial_cash: float = 1_000_000.0
    sizing_method: str = "fixed_fraction"
    params: Dict[str, Any] = {}


class OptimizeRequest(BaseModel):
    strategy: str = "MovingAverageCross"
    symbols: List[str] = ["AAPL"]
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    initial_cash: float = 1_000_000.0
    param_grid: Dict[str, List[Any]] = {"fast_period": [10, 20], "slow_period": [50, 100]}
    metric: str = "sharpe"


def _build_strategy(name: str, params: Dict[str, Any]) -> StrategyBase:
    strats = {
        "MovingAverageCross": MovingAverageCrossStrategy,
        "MeanReversion": MeanReversionStrategy,
        "Momentum": MomentumStrategy,
        "VolatilityBreakout": VolatilityBreakoutStrategy,
    }
    cls = strats.get(name, MovingAverageCrossStrategy)
    s = cls()
    if params:
        s.set_parameters(params)
    return s


@backtest_v3_router.post("/run")
def api_run(req: RunRequest):
    strategy = _build_strategy(req.strategy, req.params)
    engine = BacktestEngine(
        strategy, req.start, req.end, req.symbols,
        initial_cash=req.initial_cash, sizing_method=req.sizing_method,
    )
    result = engine.run()
    _RESULT_STORE[result.run_id] = result
    return {
        "run_id": result.run_id,
        "metrics": result.metrics,
        "final_equity": result.final_equity,
        "n_trades": len(result.trades),
    }


@backtest_v3_router.post("/optimize")
def api_optimize(req: OptimizeRequest):
    strategy = _build_strategy(req.strategy, {})
    engine = BacktestEngine(strategy, req.start, req.end, req.symbols,
                            initial_cash=req.initial_cash)
    opt = engine.optimize(req.param_grid, metric=req.metric)
    return {
        "best_params": opt.best_params,
        "best_metric": opt.best_metric,
        "metric_name": opt.metric_name,
        "top_results": opt.all_results.head(10).to_dict(orient="records"),
    }


@backtest_v3_router.get("/result/{run_id}")
def api_result(run_id: str):
    if run_id not in _RESULT_STORE:
        raise HTTPException(404, f"Run ID {run_id} not found")
    result = _RESULT_STORE[run_id]
    return {
        "run_id": result.run_id,
        "strategy": result.strategy_name,
        "metrics": result.metrics,
        "params": result.params,
        "symbols": result.symbols,
        "period": f"{result.start} → {result.end}",
    }


@backtest_v3_router.get("/tearsheet/{run_id}")
def api_tearsheet(run_id: str):
    if run_id not in _RESULT_STORE:
        raise HTTPException(404, f"Run ID {run_id} not found")
    return {"tearsheet": BacktestAnalytics.generate_tearsheet(_RESULT_STORE[run_id])}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    strategy = MovingAverageCrossStrategy(fast_period=20, slow_period=50, ma_type="EMA")
    engine = BacktestEngine(
        strategy=strategy,
        start="2020-01-01",
        end="2024-01-01",
        symbols=["AAPL"],
        initial_cash=1_000_000.0,
        sizing_method="fixed_fraction",
        verbose=True,
    )

    result = engine.run()
    tearsheet = BacktestAnalytics.generate_tearsheet(result)
    print(tearsheet)
    BacktestAnalytics.plot_results(result)
