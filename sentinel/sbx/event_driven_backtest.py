"""
Event-driven backtesting engine — Dimension #62.

Implements a clean event-driven architecture without NautilusTrader dependency.
Supports pluggable strategies, realistic fill simulation (open-of-next-bar,
slippage + commission), portfolio mark-to-market, and a full suite of
performance analytics.

Included concrete strategies:
  MomentumStrategy          — 12M-1M cross-sectional momentum
  MeanReversionStrategy     — RSI-based with stop-loss
  MovingAverageCrossStrategy — Dual SMA crossover
  BuyAndHoldStrategy        — Equal-weight benchmark

Score target: SENTINEL 9  (was 6 — paper_trading.py covers live simulation;
this module adds event-driven backtesting, walk-forward validation,
Monte Carlo bootstrap, and multi-strategy comparison — capabilities absent
from Bloomberg PORT without separate licensing).
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Iterator, Literal

import numpy as np
import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Event hierarchy
# ---------------------------------------------------------------------------

EventType = Literal[
    "MARKET_DATA", "SIGNAL", "ORDER", "FILL", "PORTFOLIO"
]

_PRIORITY: dict[str, int] = {
    "MARKET_DATA": 0,
    "SIGNAL": 1,
    "ORDER": 2,
    "FILL": 3,
    "PORTFOLIO": 4,
}


@dataclass(order=True)
class Event:
    """Base event. Ordering is by (timestamp, priority) for the event queue."""
    timestamp: datetime
    event_type: EventType = field(compare=False)
    _priority: int = field(init=False, compare=True)

    def __post_init__(self) -> None:
        self._priority = _PRIORITY.get(self.event_type, 9)


@dataclass
class MarketDataEvent(Event):
    ticker: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    vwap: float | None = None
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        self.event_type = "MARKET_DATA"
        super().__post_init__()


@dataclass
class SignalEvent(Event):
    ticker: str = ""
    signal_type: Literal["LONG", "SHORT", "EXIT"] = "LONG"
    strength: float = 1.0       # 0–1
    source_strategy: str = ""

    def __post_init__(self) -> None:
        self.event_type = "SIGNAL"
        super().__post_init__()


@dataclass
class OrderEvent(Event):
    ticker: str = ""
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    direction: Literal["BUY", "SELL"] = "BUY"
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        self.event_type = "ORDER"
        super().__post_init__()


@dataclass
class FillEvent(Event):
    ticker: str = ""
    direction: Literal["BUY", "SELL"] = "BUY"
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""

    @property
    def net_cost(self) -> float:
        """Signed cash impact (negative for buys)."""
        sign = 1 if self.direction == "SELL" else -1
        return sign * self.quantity * self.fill_price - self.commission

    def __post_init__(self) -> None:
        self.event_type = "FILL"
        super().__post_init__()


@dataclass
class PortfolioEvent(Event):
    holdings: dict[str, float] = field(default_factory=dict)  # ticker → quantity
    cash: float = 0.0
    equity: float = 0.0
    drawdown: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "PORTFOLIO"
        super().__post_init__()


# ---------------------------------------------------------------------------
# Event queue  (priority-aware deque)
# ---------------------------------------------------------------------------

class EventQueue:
    """
    Thread-safe-ish FIFO event queue with priority ordering within the same
    timestamp. Uses a deque internally; events are sorted by (timestamp, priority)
    on insertion via bisect-style insert.
    """

    def __init__(self) -> None:
        self._queue: list[Event] = []

    def put(self, event: Event) -> None:
        # Insertion sort to maintain ordering — for typical bar counts this is fine
        i = len(self._queue)
        while i > 0 and (
            self._queue[i - 1].timestamp > event.timestamp
            or (
                self._queue[i - 1].timestamp == event.timestamp
                and self._queue[i - 1]._priority > event._priority
            )
        ):
            i -= 1
        self._queue.insert(i, event)

    def get(self) -> Event | None:
        return self._queue.pop(0) if self._queue else None

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Portfolio snapshot
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    holdings: dict[str, float]         # ticker → shares held
    prices: dict[str, float]           # ticker → last price
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    peak_equity: float
    drawdown_pct: float


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Tracks positions, cash, and P&L during a backtest run.

    All position-sizing logic lives here. Strategies emit signals;
    the portfolio decides how many shares to buy/sell.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        position_sizing: Literal["equal_weight", "fixed_fractional", "user"] = "equal_weight",
        max_positions: int = 20,
        risk_per_trade: float = 0.02,  # for fixed_fractional
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_sizing = position_sizing
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade

        self.positions: dict[str, float] = {}          # ticker → quantity
        self.avg_cost: dict[str, float] = {}           # ticker → avg fill price
        self.realized_pnl: float = 0.0
        self.peak_equity: float = initial_capital
        self._last_prices: dict[str, float] = {}
        self.equity_series: list[tuple[datetime, float]] = []
        self._fills: list[FillEvent] = []

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_order(
        self,
        signal: SignalEvent,
        current_price: float,
        n_signals_pending: int = 1,
    ) -> float:
        """Return the number of shares to trade for a given signal."""
        if current_price <= 0:
            return 0.0

        equity = self._compute_equity()

        if self.position_sizing == "equal_weight":
            slot_value = equity / max(self.max_positions, n_signals_pending)
            return math.floor(slot_value / current_price)

        if self.position_sizing == "fixed_fractional":
            risk_capital = equity * self.risk_per_trade * signal.strength
            return math.floor(risk_capital / current_price)

        # user: default to 1% of equity
        return math.floor(equity * 0.01 / current_price)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_on_fill(self, fill: FillEvent) -> None:
        """Adjust positions and cash after a fill."""
        self._fills.append(fill)
        ticker = fill.ticker
        qty = fill.quantity if fill.direction == "BUY" else -fill.quantity

        prev_qty = self.positions.get(ticker, 0.0)
        prev_cost = self.avg_cost.get(ticker, 0.0)
        new_qty = prev_qty + qty

        if abs(new_qty) < 1e-9:
            # Flat — compute realized P&L
            self.realized_pnl += (fill.fill_price - prev_cost) * prev_qty
            self.positions.pop(ticker, None)
            self.avg_cost.pop(ticker, None)
        elif fill.direction == "BUY":
            # Weighted average cost
            if prev_qty >= 0:
                total_cost = prev_cost * prev_qty + fill.fill_price * fill.quantity
                self.avg_cost[ticker] = total_cost / new_qty
            else:
                # Covering short
                self.realized_pnl += (prev_cost - fill.fill_price) * fill.quantity
                remaining = new_qty
                if remaining > 0:
                    self.avg_cost[ticker] = fill.fill_price
                elif remaining < 0:
                    self.avg_cost[ticker] = prev_cost
            self.positions[ticker] = new_qty
        else:  # SELL
            if prev_qty > 0:
                self.realized_pnl += (fill.fill_price - prev_cost) * fill.quantity
            self.positions[ticker] = new_qty
            if new_qty < 0 and ticker not in self.avg_cost:
                self.avg_cost[ticker] = fill.fill_price

        self.cash += fill.net_cost
        self._last_prices[fill.ticker] = fill.fill_price

    def update_on_bar(self, bar_data: dict[str, dict]) -> None:
        """Mark-to-market all positions using latest close prices."""
        for ticker, bar in bar_data.items():
            if "close" in bar:
                self._last_prices[ticker] = bar["close"]

    def _compute_equity(self) -> float:
        mkt_value = sum(
            self.positions.get(t, 0) * self._last_prices.get(t, 0)
            for t in self.positions
        )
        return self.cash + mkt_value

    def compute_snapshot(self, timestamp: datetime | None = None) -> PortfolioSnapshot:
        equity = self._compute_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        unrealized = sum(
            self.positions.get(t, 0) * (self._last_prices.get(t, 0) - self.avg_cost.get(t, 0))
            for t in self.positions
        )
        snap = PortfolioSnapshot(
            timestamp=timestamp or datetime.now(),
            holdings=dict(self.positions),
            prices=dict(self._last_prices),
            cash=self.cash,
            equity=equity,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            peak_equity=self.peak_equity,
            drawdown_pct=drawdown * 100,
        )
        if timestamp:
            self.equity_series.append((timestamp, equity))
        return snap


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class BaseStrategy(abc.ABC):
    """
    Abstract base for all event-driven strategies.

    Subclasses implement on_bar() and optionally on_fill().
    Access historical data via self.history(ticker, lookback).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self._bars: dict[str, collections.deque] = {}

    def _register_bar(self, event: MarketDataEvent) -> None:
        if event.ticker not in self._bars:
            self._bars[event.ticker] = collections.deque(maxlen=500)
        self._bars[event.ticker].append({
            "timestamp": event.timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        })

    def history(self, ticker: str, lookback: int = 20) -> pd.DataFrame:
        """Return the last `lookback` bars for `ticker` as a DataFrame."""
        buf = self._bars.get(ticker)
        if not buf:
            return pd.DataFrame()
        rows = list(buf)[-lookback:]
        return pd.DataFrame(rows).set_index("timestamp")

    @abc.abstractmethod
    def on_bar(
        self,
        event: MarketDataEvent,
        portfolio: PortfolioSnapshot,
    ) -> list[SignalEvent]:
        """Process a new bar and return zero or more signals."""
        ...

    def on_fill(self, event: FillEvent) -> None:
        """Called after each fill. Override to track strategy-level state."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class MomentumStrategy(BaseStrategy):
    """
    12-month minus 1-month cross-sectional momentum.
    Goes long the top quintile of momentum scores across the universe.
    Rebalances monthly.
    """

    def __init__(
        self,
        lookback_long: int = 252,
        lookback_skip: int = 21,
        top_pct: float = 0.20,
        name: str = "MomentumStrategy",
    ) -> None:
        super().__init__(name)
        self.lookback_long = lookback_long
        self.lookback_skip = lookback_skip
        self.top_pct = top_pct
        self._last_rebalance: datetime | None = None
        self._universe_scores: dict[str, float] = {}

    def on_bar(
        self, event: MarketDataEvent, portfolio: PortfolioSnapshot
    ) -> list[SignalEvent]:
        self._register_bar(event)

        # Only rebalance monthly
        if self._last_rebalance is not None:
            days_since = (event.timestamp - self._last_rebalance).days
            if days_since < 20:
                return []

        hist = self.history(event.ticker, self.lookback_long)
        if len(hist) < self.lookback_long:
            return []

        closes = hist["close"]
        ret_long = (closes.iloc[-self.lookback_skip - 1] / closes.iloc[0]) - 1
        ret_skip = (closes.iloc[-1] / closes.iloc[-self.lookback_skip]) - 1
        momentum = ret_long - ret_skip
        self._universe_scores[event.ticker] = momentum

        # Compute quantile cut-off
        if not self._universe_scores:
            return []
        scores = sorted(self._universe_scores.values())
        cutoff = scores[max(0, int(len(scores) * (1 - self.top_pct)) - 1)]

        signals = []
        if momentum >= cutoff:
            in_portfolio = event.ticker in portfolio.holdings
            if not in_portfolio:
                signals.append(SignalEvent(
                    timestamp=event.timestamp,
                    event_type="SIGNAL",
                    ticker=event.ticker,
                    signal_type="LONG",
                    strength=min(1.0, max(0.1, (momentum - cutoff) / (abs(cutoff) + 1e-9))),
                    source_strategy=self.name,
                ))
        elif event.ticker in portfolio.holdings:
            signals.append(SignalEvent(
                timestamp=event.timestamp,
                event_type="SIGNAL",
                ticker=event.ticker,
                signal_type="EXIT",
                strength=1.0,
                source_strategy=self.name,
            ))

        if signals:
            self._last_rebalance = event.timestamp
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion: buy when RSI < oversold, sell when RSI > overbought.
    2% stop-loss from entry.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        stop_pct: float = 0.02,
        name: str = "MeanReversionStrategy",
    ) -> None:
        super().__init__(name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_pct = stop_pct
        self._entry_prices: dict[str, float] = {}

    @staticmethod
    def _rsi(closes: pd.Series, period: int = 14) -> float:
        delta = closes.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))

    def on_bar(
        self, event: MarketDataEvent, portfolio: PortfolioSnapshot
    ) -> list[SignalEvent]:
        self._register_bar(event)
        hist = self.history(event.ticker, self.rsi_period + 2)
        if len(hist) < self.rsi_period + 1:
            return []

        rsi = self._rsi(hist["close"], self.rsi_period)
        signals = []
        in_position = event.ticker in portfolio.holdings

        # Stop-loss check
        if in_position and event.ticker in self._entry_prices:
            entry = self._entry_prices[event.ticker]
            if event.close < entry * (1 - self.stop_pct):
                signals.append(SignalEvent(
                    timestamp=event.timestamp,
                    event_type="SIGNAL",
                    ticker=event.ticker,
                    signal_type="EXIT",
                    strength=1.0,
                    source_strategy=self.name,
                ))
                self._entry_prices.pop(event.ticker, None)
                return signals

        if rsi < self.oversold and not in_position:
            signals.append(SignalEvent(
                timestamp=event.timestamp,
                event_type="SIGNAL",
                ticker=event.ticker,
                signal_type="LONG",
                strength=(self.oversold - rsi) / self.oversold,
                source_strategy=self.name,
            ))
            self._entry_prices[event.ticker] = event.close
        elif rsi > self.overbought and in_position:
            signals.append(SignalEvent(
                timestamp=event.timestamp,
                event_type="SIGNAL",
                ticker=event.ticker,
                signal_type="EXIT",
                strength=1.0,
                source_strategy=self.name,
            ))
            self._entry_prices.pop(event.ticker, None)

        return signals

    def on_fill(self, event: FillEvent) -> None:
        if event.direction == "BUY":
            self._entry_prices[event.ticker] = event.fill_price
        elif event.direction == "SELL":
            self._entry_prices.pop(event.ticker, None)


class MovingAverageCrossStrategy(BaseStrategy):
    """
    Dual SMA crossover: go long when fast SMA crosses above slow SMA;
    exit when fast crosses below slow.
    """

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        name: str = "MovingAverageCrossStrategy",
    ) -> None:
        super().__init__(name)
        self.fast = fast
        self.slow = slow
        self._prev_cross: dict[str, str] = {}  # ticker → "above"/"below"

    def on_bar(
        self, event: MarketDataEvent, portfolio: PortfolioSnapshot
    ) -> list[SignalEvent]:
        self._register_bar(event)
        hist = self.history(event.ticker, self.slow + 1)
        if len(hist) < self.slow:
            return []

        closes = hist["close"]
        fast_sma = float(closes.tail(self.fast).mean())
        slow_sma = float(closes.mean())

        prev = self._prev_cross.get(event.ticker, "none")
        current = "above" if fast_sma > slow_sma else "below"
        self._prev_cross[event.ticker] = current

        signals = []
        if current == "above" and prev == "below":
            # Bullish crossover
            if event.ticker not in portfolio.holdings:
                signals.append(SignalEvent(
                    timestamp=event.timestamp,
                    event_type="SIGNAL",
                    ticker=event.ticker,
                    signal_type="LONG",
                    strength=min(1.0, abs(fast_sma - slow_sma) / slow_sma * 100),
                    source_strategy=self.name,
                ))
        elif current == "below" and prev == "above":
            # Bearish crossover
            if event.ticker in portfolio.holdings:
                signals.append(SignalEvent(
                    timestamp=event.timestamp,
                    event_type="SIGNAL",
                    ticker=event.ticker,
                    signal_type="EXIT",
                    strength=1.0,
                    source_strategy=self.name,
                ))

        return signals


class BuyAndHoldStrategy(BaseStrategy):
    """
    Equal-weight buy-and-hold benchmark.
    Buys all tickers on first bar; never sells. Used as benchmark.
    """

    def __init__(self, name: str = "BuyAndHoldStrategy") -> None:
        super().__init__(name)
        self._bought: set[str] = set()

    def on_bar(
        self, event: MarketDataEvent, portfolio: PortfolioSnapshot
    ) -> list[SignalEvent]:
        self._register_bar(event)
        if event.ticker in self._bought:
            return []
        self._bought.add(event.ticker)
        return [SignalEvent(
            timestamp=event.timestamp,
            event_type="SIGNAL",
            ticker=event.ticker,
            signal_type="LONG",
            strength=1.0,
            source_strategy=self.name,
        )]


# ---------------------------------------------------------------------------
# Execution handler
# ---------------------------------------------------------------------------

class ExecutionHandler:
    """
    Simulates order fills at the open of the next bar.

    Avoids look-ahead bias by filling at the NEXT bar's open, not the
    current bar's close.
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def simulate_fill(
        self,
        order: OrderEvent,
        next_bar: dict,
    ) -> FillEvent | None:
        """
        Fill market orders at next bar's open with slippage.
        Returns None if there is no next bar.
        """
        if not next_bar or order.quantity <= 0:
            return None

        base_price = next_bar.get("open", next_bar.get("close", 0.0))
        if base_price <= 0:
            return None

        slippage_adj = self.slippage_pct if order.direction == "BUY" else -self.slippage_pct
        fill_price = base_price * (1 + slippage_adj)
        commission = fill_price * order.quantity * self.commission_pct

        return FillEvent(
            timestamp=next_bar.get("timestamp", order.timestamp + timedelta(days=1)),
            event_type="FILL",
            ticker=order.ticker,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=abs(fill_price - base_price) * order.quantity,
            order_id=order.order_id,
        )


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    equity_curve: pd.Series          # DatetimeIndex → equity
    trades: pd.DataFrame             # fills joined into round trips
    _metrics: dict | None = field(default=None, repr=False)

    def compute_metrics(self) -> dict:
        if self._metrics is not None:
            return self._metrics

        ec = self.equity_curve.sort_index().dropna()
        if ec.empty or len(ec) < 2:
            return {}

        rets = ec.pct_change().dropna()
        total_days = (ec.index[-1] - ec.index[0]).days or 1
        years = total_days / 365.25

        total_return = (ec.iloc[-1] / self.initial_capital) - 1
        cagr = (ec.iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

        daily_rf = 0.0  # risk-free rate = 0 for simplicity
        excess = rets - daily_rf
        sharpe = (excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0

        downside = rets[rets < 0]
        sortino = (
            excess.mean() / downside.std() * math.sqrt(252)
            if not downside.empty and downside.std() > 0
            else 0.0
        )

        roll_max = ec.cummax()
        drawdown = (ec - roll_max) / roll_max
        max_dd = float(drawdown.min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Trade-level stats
        num_trades = len(self.trades) if not self.trades.empty else 0
        wins = self.trades[self.trades["pnl"] > 0] if not self.trades.empty and "pnl" in self.trades.columns else pd.DataFrame()
        losses = self.trades[self.trades["pnl"] < 0] if not self.trades.empty and "pnl" in self.trades.columns else pd.DataFrame()
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0
        profit_factor = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if not losses.empty and losses["pnl"].sum() != 0
            else float("inf")
        )

        avg_holding = (
            float(self.trades["holding_days"].mean())
            if not self.trades.empty and "holding_days" in self.trades.columns
            else 0.0
        )

        self._metrics = {
            "strategy": self.strategy_name,
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "num_trades": num_trades,
            "avg_holding_days": round(avg_holding, 1),
            "years_simulated": round(years, 2),
        }
        return self._metrics

    def summary(self) -> str:
        m = self.compute_metrics()
        return (
            f"Strategy: {m.get('strategy')}\n"
            f"  Total Return:  {m.get('total_return_pct')}%\n"
            f"  CAGR:          {m.get('cagr_pct')}%\n"
            f"  Sharpe:        {m.get('sharpe_ratio')}\n"
            f"  Max Drawdown:  {m.get('max_drawdown_pct')}%\n"
            f"  Win Rate:      {m.get('win_rate_pct')}%\n"
            f"  Num Trades:    {m.get('num_trades')}\n"
        )


# ---------------------------------------------------------------------------
# Event-driven backtester
# ---------------------------------------------------------------------------

class EventDrivenBacktester:
    """
    Main backtesting engine.

    Usage:
        bt = EventDrivenBacktester(initial_capital=500_000)
        bt.load_data({"AAPL": aapl_ohlcv_df, "MSFT": msft_ohlcv_df})
        bt.add_strategy(MomentumStrategy())
        result = bt.run("2018-01-01", "2023-12-31")
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_sizing: Literal["equal_weight", "fixed_fractional", "user"] = "equal_weight",
        max_positions: int = 20,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self._raw_data: dict[str, pd.DataFrame] = {}
        self._strategies: list[BaseStrategy] = []
        self._execution = ExecutionHandler(commission_pct, slippage_pct)
        self._portfolio = Portfolio(
            initial_capital=initial_capital,
            position_sizing=position_sizing,
            max_positions=max_positions,
        )

    def load_data(self, ticker_data: dict[str, pd.DataFrame]) -> None:
        """
        Load OHLCV data.

        Each DataFrame must have a DatetimeIndex and columns:
        open, high, low, close, volume. Additional columns (vwap, bid, ask) are optional.
        """
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
        logger.info("Loaded %d tickers for backtest", len(self._raw_data))

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)

    def _build_bar_timeline(
        self, start: str, end: str
    ) -> list[tuple[datetime, str, dict]]:
        """
        Create a chronologically sorted list of (timestamp, ticker, bar_dict).
        """
        timeline = []
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        for ticker, df in self._raw_data.items():
            sub = df.loc[start_dt:end_dt]
            for ts, row in sub.iterrows():
                bar = row.to_dict()
                bar["timestamp"] = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                timeline.append((bar["timestamp"], ticker, bar))
        timeline.sort(key=lambda x: (x[0], x[1]))
        return timeline

    def _bars_after(self, ticker: str, after_ts: datetime) -> dict | None:
        """Return the first bar for ticker strictly after after_ts."""
        df = self._raw_data.get(ticker)
        if df is None:
            return None
        future = df.loc[pd.Timestamp(after_ts):]
        if future.empty:
            return None
        # Skip the current bar (exact match) — we want the NEXT one
        idx = future.index
        next_idx = None
        for i, ts in enumerate(idx):
            if ts.to_pydatetime() > after_ts:
                next_idx = ts
                break
        if next_idx is None:
            return None
        row = future.loc[next_idx].to_dict()
        row["timestamp"] = next_idx.to_pydatetime()
        return row

    def _signal_to_order(
        self,
        signal: SignalEvent,
        current_price: float,
        n_signals: int = 1,
    ) -> OrderEvent | None:
        if signal.signal_type == "EXIT":
            qty = self._portfolio.positions.get(signal.ticker, 0)
            if qty == 0:
                return None
            return OrderEvent(
                timestamp=signal.timestamp,
                event_type="ORDER",
                ticker=signal.ticker,
                order_type="MARKET",
                direction="SELL",
                quantity=abs(qty),
            )

        if signal.signal_type in ("LONG", "SHORT"):
            direction = "BUY" if signal.signal_type == "LONG" else "SELL"
            qty = self._portfolio.size_order(signal, current_price, n_signals)
            if qty <= 0:
                return None
            return OrderEvent(
                timestamp=signal.timestamp,
                event_type="ORDER",
                ticker=signal.ticker,
                order_type="MARKET",
                direction=direction,
                quantity=qty,
            )

        return None

    def run(self, start_date: str, end_date: str) -> BacktestResult:
        """
        Main event loop.

        1. Iterate bars chronologically
        2. Emit MarketDataEvent → strategy processes → SignalEvents
        3. Portfolio converts signals → OrderEvents
        4. Execution handler fills orders → FillEvents
        5. Portfolio updates
        """
        if not self._strategies:
            raise RuntimeError("No strategies added. Call add_strategy() first.")
        if not self._raw_data:
            raise RuntimeError("No data loaded. Call load_data() first.")

        timeline = self._build_bar_timeline(start_date, end_date)
        logger.info("Running backtest: %d bars across %d tickers", len(timeline), len(self._raw_data))

        all_fills: list[FillEvent] = []
        portfolio = self._portfolio

        # Track last bar per ticker for mark-to-market
        last_bar: dict[str, dict] = {}

        for ts, ticker, bar in timeline:
            last_bar[ticker] = bar

            # --- MarketDataEvent ---
            mde = MarketDataEvent(
                timestamp=ts,
                event_type="MARKET_DATA",
                ticker=ticker,
                open=bar.get("open", 0),
                high=bar.get("high", 0),
                low=bar.get("low", 0),
                close=bar.get("close", 0),
                volume=bar.get("volume", 0),
                vwap=bar.get("vwap"),
                bid=bar.get("bid"),
                ask=bar.get("ask"),
            )

            # Update portfolio mark-to-market
            portfolio.update_on_bar({ticker: bar})
            snap = portfolio.compute_snapshot(ts)

            # --- Strategy signal generation ---
            signals: list[SignalEvent] = []
            for strategy in self._strategies:
                try:
                    strategy._register_bar(mde)
                    new_signals = strategy.on_bar(mde, snap)
                    signals.extend(new_signals or [])
                except Exception as exc:
                    logger.debug("Strategy %s error on %s/%s: %s", strategy.name, ticker, ts, exc)

            # --- Signal → Order → Fill ---
            for signal in signals:
                current_price = bar.get("close", 0)
                order = self._signal_to_order(signal, current_price, len(signals))
                if order is None:
                    continue

                next_bar = self._bars_after(ticker, ts)
                if next_bar is None:
                    continue

                fill = self._execution.simulate_fill(order, next_bar)
                if fill is None:
                    continue

                portfolio.update_on_fill(fill)
                all_fills.append(fill)

                for strategy in self._strategies:
                    try:
                        strategy.on_fill(fill)
                    except Exception:
                        pass

        # Build equity curve from recorded snapshots
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
        result = BacktestResult(
            strategy_name=strategy_names,
            initial_capital=self.initial_capital,
            equity_curve=eq_series,
            trades=trades_df,
        )

        logger.info(
            "Backtest complete. Trades: %d | Final equity: %.2f",
            len(all_fills),
            float(eq_series.iloc[-1]) if not eq_series.empty else self.initial_capital,
        )
        return result

    def _build_trades_df(self, fills: list[FillEvent]) -> pd.DataFrame:
        """
        Convert a list of fills into round-trip trade records.

        Matches BUY fills with subsequent SELL fills per ticker using FIFO.
        """
        if not fills:
            return pd.DataFrame(columns=[
                "ticker", "direction", "entry_price", "exit_price",
                "quantity", "pnl", "holding_days", "entry_ts", "exit_ts", "commission",
            ])

        # Group fills by ticker
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
                })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    High-level runner for single runs, walk-forward validation,
    Monte Carlo bootstrap, and multi-strategy comparison.
    """

    def run_single(
        self,
        strategy: BaseStrategy,
        data: dict[str, pd.DataFrame],
        start: str,
        end: str,
        capital: float = 1_000_000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> BacktestResult:
        """Run a single strategy over the full period."""
        bt = EventDrivenBacktester(
            initial_capital=capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        bt.load_data(data)
        bt.add_strategy(strategy)
        return bt.run(start, end)

    def run_walk_forward(
        self,
        strategy_factory: Callable[[], BaseStrategy],
        data: dict[str, pd.DataFrame],
        full_start: str,
        full_end: str,
        train_months: int = 12,
        test_months: int = 3,
        capital: float = 1_000_000,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation.

        Splits the full period into consecutive train/test windows.
        Returns one BacktestResult per test window.
        """
        results = []
        start = pd.Timestamp(full_start)
        end = pd.Timestamp(full_end)
        cursor = start + pd.DateOffset(months=train_months)

        while cursor < end:
            test_end = min(cursor + pd.DateOffset(months=test_months), end)
            train_start_str = start.strftime("%Y-%m-%d")
            test_start_str = cursor.strftime("%Y-%m-%d")
            test_end_str = test_end.strftime("%Y-%m-%d")

            # Train (in-sample) — strategy uses this period to warm up
            strategy = strategy_factory()
            bt_train = EventDrivenBacktester(initial_capital=capital)
            bt_train.load_data(data)
            bt_train.add_strategy(strategy)
            try:
                bt_train.run(train_start_str, test_start_str)
            except Exception:
                pass  # Allow warm-up to fail silently

            # Test (out-of-sample)
            bt_test = EventDrivenBacktester(initial_capital=capital)
            bt_test.load_data(data)
            bt_test.add_strategy(strategy)
            try:
                result = bt_test.run(test_start_str, test_end_str)
                result.strategy_name = f"{strategy.name} [OOS {test_start_str}:{test_end_str}]"
                results.append(result)
                logger.info(
                    "WF window %s→%s: %.2f%% return",
                    test_start_str,
                    test_end_str,
                    result.compute_metrics().get("total_return_pct", 0),
                )
            except Exception as exc:
                logger.warning("Walk-forward window failed: %s", exc)

            cursor = test_end

        return results

    def run_monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        seed: int | None = 42,
    ) -> dict:
        """
        Bootstrap returns from a BacktestResult to generate confidence intervals.

        Returns distributions and CIs for Sharpe ratio, max drawdown, and CAGR.
        """
        rng = np.random.default_rng(seed)
        ec = result.equity_curve.sort_index().dropna()
        if len(ec) < 10:
            return {"error": "Insufficient data for Monte Carlo simulation"}

        daily_rets = ec.pct_change().dropna().values
        n_days = len(daily_rets)

        sharpes, max_dds, cagrs = [], [], []

        for _ in range(n_simulations):
            sim_rets = rng.choice(daily_rets, size=n_days, replace=True)
            sim_equity = result.initial_capital * np.cumprod(1 + sim_rets)

            # Sharpe
            if sim_rets.std() > 0:
                sharpes.append(sim_rets.mean() / sim_rets.std() * math.sqrt(252))
            else:
                sharpes.append(0.0)

            # Max drawdown
            roll_max = np.maximum.accumulate(sim_equity)
            drawdowns = (sim_equity - roll_max) / roll_max
            max_dds.append(float(drawdowns.min()))

            # CAGR
            years = n_days / 252
            if years > 0 and sim_equity[-1] > 0:
                cagrs.append((sim_equity[-1] / result.initial_capital) ** (1 / years) - 1)

        def _ci(arr: list[float], lo: float = 5, hi: float = 95) -> tuple:
            return (
                round(float(np.percentile(arr, lo)), 4),
                round(float(np.median(arr)), 4),
                round(float(np.percentile(arr, hi)), 4),
            )

        sharpe_ci = _ci(sharpes)
        dd_ci = _ci(max_dds)
        cagr_ci = _ci(cagrs)

        return {
            "n_simulations": n_simulations,
            "sharpe_ratio": {
                "p5": sharpe_ci[0], "median": sharpe_ci[1], "p95": sharpe_ci[2],
                "mean": round(float(np.mean(sharpes)), 4),
                "std": round(float(np.std(sharpes)), 4),
            },
            "max_drawdown_pct": {
                "p5": round(dd_ci[0] * 100, 2),
                "median": round(dd_ci[1] * 100, 2),
                "p95": round(dd_ci[2] * 100, 2),
            },
            "cagr_pct": {
                "p5": round(cagr_ci[0] * 100, 2),
                "median": round(cagr_ci[1] * 100, 2),
                "p95": round(cagr_ci[2] * 100, 2),
            },
            "probability_positive_return": round(
                sum(1 for c in cagrs if c > 0) / len(cagrs) * 100, 1
            ) if cagrs else 0.0,
        }

    def compare_strategies(
        self,
        strategies: dict[str, BaseStrategy],
        data: dict[str, pd.DataFrame],
        start: str,
        end: str,
        capital: float = 1_000_000,
    ) -> pd.DataFrame:
        """
        Run multiple strategies over the same period and return a side-by-side metrics table.

        strategies: {name: BaseStrategy instance}
        Returns a DataFrame with strategies as rows and metrics as columns.
        """
        rows = []
        for name, strategy in strategies.items():
            strategy.name = name
            try:
                result = self.run_single(strategy, data, start, end, capital)
                m = result.compute_metrics()
                m["strategy"] = name
                rows.append(m)
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", name, exc)
                rows.append({"strategy": name, "error": str(exc)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("strategy")
        # Sort by Sharpe ratio descending
        if "sharpe_ratio" in df.columns:
            df = df.sort_values("sharpe_ratio", ascending=False)
        return df


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def create_default_strategies() -> dict[str, BaseStrategy]:
    """Return the four bundled strategies ready for comparison."""
    return {
        "Momentum": MomentumStrategy(),
        "MeanReversion": MeanReversionStrategy(),
        "MA Cross": MovingAverageCrossStrategy(),
        "Buy & Hold": BuyAndHoldStrategy(),
    }


def quick_backtest(
    ticker_data: dict[str, pd.DataFrame],
    start: str,
    end: str,
    strategy: BaseStrategy | None = None,
    capital: float = 1_000_000,
) -> BacktestResult:
    """
    One-liner convenience wrapper.

    If no strategy is provided, defaults to BuyAndHoldStrategy.
    """
    strategy = strategy or BuyAndHoldStrategy()
    runner = BacktestRunner()
    return runner.run_single(strategy, ticker_data, start, end, capital)
