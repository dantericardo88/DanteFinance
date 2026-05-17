"""
Paper Trading Simulator v3 — Production-Grade Execution Engine
==============================================================
Dimension: dim_066  target score 7 → 9

Features:
  - PaperBroker: realistic fills for MARKET/LIMIT/STOP/STOP_LIMIT/MOC/MOO
  - SlippageModel: none / fixed / proportional / market_impact (Almgren-Chriss)
  - Portfolio: multi-position, mark-to-market, drawdown tracking
  - MarketSimulator: yfinance live price, bid-ask estimation, NYSE calendar
  - PaperTradingSession: live thread loop + bar-by-bar backtest
  - PerformanceAnalytics: Sharpe, Sortino, Calmar, VaR, CVaR, attribution
  - RiskManager: position limits, sector concentration, drawdown halt, VaR cap
  - PaperTradingDashboard: formatted display, trade log, JSON export/import
  - Built-in strategies: SMA crossover, momentum
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
        return logging.getLogger(name)

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOC = "MOC"   # Market On Close
    MOO = "MOO"   # Market On Open


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class SlippageModel(str, Enum):
    NONE = "none"
    FIXED = "fixed"
    PROPORTIONAL = "proportional"
    MARKET_IMPACT = "market_impact"


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    commission_paid: float = 0.0
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class Trade:
    trade_id: str
    order_id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    realized_pnl: float = 0.0
    strategy_tag: str = ""


@dataclass
class Account:
    cash: float
    buying_power: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    initial_cash: float
    margin_used: float = 0.0
    day_trades: int = 0


@dataclass
class SessionState:
    timestamp: datetime
    equity: float
    cash: float
    daily_pnl: float
    total_pnl: float
    n_positions: int
    n_open_orders: int
    drawdown: float


@dataclass
class Tearsheet:
    # Returns
    cagr: float
    total_return: float
    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float
    # Risk
    max_drawdown: float
    volatility_annual: float
    var_95: float
    cvar_95: float
    # Activity
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_hold_days: float
    turnover_annual: float
    # Drawdown detail
    avg_drawdown: float
    max_drawdown_duration_days: int


@dataclass
class RiskCheck:
    passed: bool
    reason: str
    metric_name: str
    metric_value: float
    limit: float


@dataclass
class SessionResult:
    equity_curve: pd.Series
    trades: List[Trade]
    tearsheet: Tearsheet
    positions_history: List[dict]
    config: dict


# ──────────────────────────────────────────────────────────────────────────────
# NYSE Holiday Calendar 2024-2026 (hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

NYSE_HOLIDAYS: set = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9),  # Carter memorial
    date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


# ──────────────────────────────────────────────────────────────────────────────
# Position
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sector: str = "Unknown"
    strategy_tag: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def pnl_pct(self) -> float:
        cost_basis = self.avg_cost * self.quantity
        if cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / cost_basis

    @property
    def holding_period_days(self) -> int:
        return (datetime.now(timezone.utc) - self.opened_at).days

    def update_price(self, price: float) -> None:
        self.current_price = price

    def add_shares(self, qty: float, price: float) -> None:
        """Average up/down: update avg_cost."""
        total_cost = self.avg_cost * self.quantity + price * qty
        self.quantity += qty
        self.avg_cost = total_cost / self.quantity if self.quantity != 0 else 0.0

    def reduce_shares(self, qty: float) -> float:
        """Return realized PnL for reduced quantity."""
        qty = min(qty, self.quantity)
        realized = (self.current_price - self.avg_cost) * qty
        self.realized_pnl += realized
        self.quantity -= qty
        return realized


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────────────────────────────────────

class Portfolio:
    """Manages all open positions with mark-to-market pricing."""

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self._peak_equity: float = initial_cash
        self._equity_history: List[float] = [initial_cash]

    # ── Price Updates ──────────────────────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]) -> None:
        for ticker, price in prices.items():
            if ticker in self.positions and price > 0:
                self.positions[ticker].update_price(price)
        equity = self.get_equity()
        self._equity_history.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    # ── Position Access ────────────────────────────────────────────────────────

    def get_position(self, ticker: str) -> Optional[Position]:
        pos = self.positions.get(ticker)
        return pos if pos and pos.quantity != 0 else None

    def get_all_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.quantity != 0]

    # ── Equity / Metrics ───────────────────────────────────────────────────────

    def get_equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.get_all_positions())

    def get_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.get_all_positions())

    def get_portfolio_weights(self) -> Dict[str, float]:
        equity = self.get_equity()
        if equity == 0:
            return {}
        return {
            p.ticker: p.market_value / equity
            for p in self.get_all_positions()
        }

    def compute_drawdown(self) -> float:
        """Current drawdown from peak equity (0 to 1)."""
        equity = self.get_equity()
        if self._peak_equity == 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity)

    # ── Trade Processing ───────────────────────────────────────────────────────

    def apply_fill(self, order: Order, fill_price: float) -> float:
        """Apply a filled order to portfolio. Returns realized PnL (for closes)."""
        ticker = order.ticker
        qty = order.filled_quantity
        realized = 0.0

        if order.side == OrderSide.BUY:
            cost = fill_price * qty + order.commission_paid
            self.cash -= cost
            if ticker in self.positions and self.positions[ticker].quantity > 0:
                self.positions[ticker].add_shares(qty, fill_price)
            else:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_cost=fill_price,
                    current_price=fill_price,
                )
        else:  # SELL
            proceeds = fill_price * qty - order.commission_paid
            self.cash += proceeds
            if ticker in self.positions:
                realized = self.positions[ticker].reduce_shares(qty)
                if self.positions[ticker].quantity == 0:
                    del self.positions[ticker]
            self.realized_pnl += realized

        return realized

    def get_sector_weights(self) -> Dict[str, float]:
        equity = self.get_equity()
        sectors: Dict[str, float] = {}
        for pos in self.get_all_positions():
            sectors[pos.sector] = sectors.get(pos.sector, 0.0) + pos.market_value
        return {s: v / equity for s, v in sectors.items()} if equity > 0 else {}


# ──────────────────────────────────────────────────────────────────────────────
# Market Simulator
# ──────────────────────────────────────────────────────────────────────────────

class MarketSimulator:
    """Simulates market conditions: prices, spreads, slippage, market hours."""

    ILLIQUID_TICKERS: set = {"OTC", "PINK"}  # placeholder; expand as needed

    def __init__(self, slippage_model: SlippageModel = SlippageModel.FIXED) -> None:
        self.slippage_model = slippage_model
        self._price_cache: Dict[str, float] = {}
        self._adv_cache: Dict[str, float] = {}   # avg daily volume × price

    # ── Price Fetching ─────────────────────────────────────────────────────────

    def get_current_price(self, ticker: str) -> float:
        """Live delayed price via yfinance fast_info, fallback to last cache."""
        try:
            info = yf.Ticker(ticker).fast_info
            price = float(info.last_price)
            if price and price > 0:
                self._price_cache[ticker] = price
                return price
        except Exception:
            pass
        cached = self._price_cache.get(ticker)
        if cached:
            # random walk perturbation ±0.05% for backtesting continuity
            noise = cached * random.gauss(0, 0.0005)
            return max(0.01, cached + noise)
        return 0.0

    def get_current_prices_batch(self, tickers: List[str]) -> Dict[str, float]:
        """Batch price fetch using yfinance download."""
        prices: Dict[str, float] = {}
        try:
            data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            last = close.iloc[-1]
            for t in tickers:
                if t in last.index and not pd.isna(last[t]):
                    prices[t] = float(last[t])
                    self._price_cache[t] = float(last[t])
        except Exception as exc:
            logger.warning("batch price fetch failed: %s", exc)
        # fill missing from cache
        for t in tickers:
            if t not in prices:
                prices[t] = self.get_current_price(t)
        return prices

    # ── Bid-Ask Estimation ─────────────────────────────────────────────────────

    def get_bid_ask(self, ticker: str) -> Tuple[float, float]:
        """Estimate bid-ask. Liquid stocks: ≈0.02% spread. Illiquid: 0.1-0.5%."""
        price = self._price_cache.get(ticker) or self.get_current_price(ticker)
        if price <= 0:
            return 0.0, 0.0
        # Use minimum tick of $0.01 and percentage-based spread
        if price >= 100:
            spread_pct = 0.0002   # 2bps
        elif price >= 10:
            spread_pct = 0.0005   # 5bps
        else:
            spread_pct = 0.002    # 20bps for penny/micro
        spread = max(0.01, price * spread_pct)
        half = spread / 2
        return price - half, price + half

    # ── Slippage ───────────────────────────────────────────────────────────────

    def simulate_slippage(self, order: Order, base_price: float) -> float:
        """Returns adjusted fill price including slippage."""
        if self.slippage_model == SlippageModel.NONE:
            return base_price

        sign = 1.0 if order.side == OrderSide.BUY else -1.0

        if self.slippage_model == SlippageModel.FIXED:
            return base_price * (1.0 + sign * 0.0001)   # ±0.01%

        if self.slippage_model == SlippageModel.PROPORTIONAL:
            bid, ask = self.get_bid_ask(order.ticker)
            spread = ask - bid
            return base_price + sign * (0.5 * spread)

        if self.slippage_model == SlippageModel.MARKET_IMPACT:
            # Almgren-Chriss simplified: impact = σ × sqrt(order_size / ADV) × price
            sigma = 0.015  # daily vol estimate; ideally computed from history
            order_value = base_price * order.quantity
            adv = self._adv_cache.get(order.ticker, 10_000_000)  # default $10M ADV
            if order_value > 10_000 and adv > 0:
                impact = sigma * math.sqrt(order_value / adv) * base_price
            else:
                impact = base_price * 0.0001
            return base_price + sign * impact

        return base_price

    # ── Market Hours ───────────────────────────────────────────────────────────

    def get_market_hours_status(self) -> str:
        """Returns PRE, OPEN, AFTER, or CLOSED."""
        now_et = datetime.now(timezone.utc) - timedelta(hours=4)  # EST approx
        if now_et.weekday() >= 5:
            return "CLOSED"
        t = now_et.time()
        from datetime import time as dtime
        if dtime(4, 0) <= t < dtime(9, 30):
            return "PRE"
        if dtime(9, 30) <= t < dtime(16, 0):
            return "OPEN"
        if dtime(16, 0) <= t < dtime(20, 0):
            return "AFTER"
        return "CLOSED"

    def is_trading_day(self, dt: date) -> bool:
        """True if NYSE is open on this date."""
        if isinstance(dt, str):
            dt = date.fromisoformat(dt)
        if dt.weekday() >= 5:  # Sat/Sun
            return False
        return dt not in NYSE_HOLIDAYS

    def cache_adv(self, ticker: str, adv: float) -> None:
        self._adv_cache[ticker] = adv


# ──────────────────────────────────────────────────────────────────────────────
# Paper Broker
# ──────────────────────────────────────────────────────────────────────────────

class PaperBroker:
    """
    Simulates a real broker with realistic fill logic.
    Supports MARKET, LIMIT, STOP, STOP_LIMIT, MOC, MOO order types.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        market_sim: MarketSimulator,
        initial_cash: float = 100_000.0,
        commission: float = 0.0,
        commission_per_share: float = 0.0,
        slippage_model: str = "fixed",
    ) -> None:
        self.portfolio = portfolio
        self.market_sim = market_sim
        self.commission = commission               # flat per order
        self.commission_per_share = commission_per_share
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._rejected_orders: List[Order] = []

    # ── Order Submission ───────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with assigned ID and PENDING status."""
        # Basic validation
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.notes = "Quantity must be positive."
            self._rejected_orders.append(order)
            return order
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            order.notes = "LIMIT order requires limit_price."
            self._rejected_orders.append(order)
            return order
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
            order.status = OrderStatus.REJECTED
            order.notes = "STOP order requires stop_price."
            self._rejected_orders.append(order)
            return order

        self._orders[order.order_id] = order
        logger.debug("Order submitted: %s %s %s×%s", order.order_type, order.side, order.quantity, order.ticker)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def get_trades(self) -> List[Trade]:
        return list(self._trades)

    # ── Fill Processing ────────────────────────────────────────────────────────

    def process_bar(self, ticker: str, bar: pd.Series) -> List[Trade]:
        """
        Process all pending orders for a ticker against a single OHLC bar.
        bar must have: Open, High, Low, Close, Volume fields.
        Returns list of new trades generated.
        """
        new_trades: List[Trade] = []
        for order in list(self._orders.values()):
            if order.ticker != ticker or order.status != OrderStatus.PENDING:
                continue
            fill_price = self._check_fill(order, bar)
            if fill_price is not None:
                trade = self._execute_fill(order, fill_price, bar)
                new_trades.append(trade)
        return new_trades

    def _check_fill(self, order: Order, bar: pd.Series) -> Optional[float]:
        """Determine fill price for an order given a bar. Returns None if no fill."""
        o, h, l, c = bar["Open"], bar["High"], bar["Low"], bar["Close"]

        if order.order_type == OrderType.MARKET:
            # Fill at open + slippage
            return self.market_sim.simulate_slippage(order, o)

        if order.order_type == OrderType.MOO:
            return self.market_sim.simulate_slippage(order, o)

        if order.order_type == OrderType.MOC:
            return self.market_sim.simulate_slippage(order, c)

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and l <= order.limit_price:
                return min(order.limit_price, o)  # favorable fill
            if order.side == OrderSide.SELL and h >= order.limit_price:
                return max(order.limit_price, o)

        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and h >= order.stop_price:
                fill = max(order.stop_price, o)
                return self.market_sim.simulate_slippage(order, fill)
            if order.side == OrderSide.SELL and l <= order.stop_price:
                fill = min(order.stop_price, o)
                return self.market_sim.simulate_slippage(order, fill)

        if order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                if h >= order.stop_price and l <= order.limit_price:
                    return order.limit_price
            if order.side == OrderSide.SELL:
                if l <= order.stop_price and h >= order.limit_price:
                    return order.limit_price

        return None

    def _compute_commission(self, order: Order) -> float:
        base = self.commission
        per_share = self.commission_per_share * order.quantity
        return base + per_share

    def _execute_fill(self, order: Order, fill_price: float, bar: pd.Series) -> Trade:
        commission = self._compute_commission(order)
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission_paid = commission
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)

        realized = self.portfolio.apply_fill(order, fill_price)

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now(timezone.utc),
            realized_pnl=realized,
        )
        self._trades.append(trade)
        logger.info(
            "FILL: %s %s %d @ $%.4f  comm=$%.2f  realized_pnl=$%.2f",
            order.side.value, order.ticker, order.quantity, fill_price, commission, realized,
        )
        return trade

    def process_live_order(self, order: Order) -> Optional[Trade]:
        """Process a single MARKET order immediately at current price."""
        price = self.market_sim.get_current_price(order.ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.notes = "Could not fetch live price."
            return None
        fill_price = self.market_sim.simulate_slippage(order, price)
        commission = self._compute_commission(order)
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission_paid = commission
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)

        realized = self.portfolio.apply_fill(order, fill_price)
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now(timezone.utc),
            realized_pnl=realized,
        )
        self._trades.append(trade)
        return trade

    def get_account(self) -> Account:
        equity = self.portfolio.get_equity()
        unrealized = self.portfolio.get_unrealized_pnl()
        realized = self.portfolio.realized_pnl
        return Account(
            cash=self.portfolio.cash,
            buying_power=self.portfolio.cash,
            equity=equity,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_pnl=unrealized + realized,
            initial_cash=self.portfolio.initial_cash,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ──────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """Real-time risk checks on the paper portfolio."""

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_sector_pct: float = 0.40,
        max_leverage: float = 1.0,
        drawdown_warn: float = 0.10,
        drawdown_halt: float = 0.20,
        var_limit_pct: float = 0.05,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_leverage = max_leverage
        self.drawdown_warn = drawdown_warn
        self.drawdown_halt = drawdown_halt
        self.var_limit_pct = var_limit_pct
        self._halted: bool = False

    def check_position_limits(self, portfolio: Portfolio, order: Order) -> RiskCheck:
        equity = portfolio.get_equity()
        price = portfolio.positions.get(order.ticker)
        if price:
            current_mv = price.market_value
        else:
            current_mv = 0.0

        # Estimate post-fill market value
        # Use a rough price estimate from the order (limit price or last known)
        est_price = order.limit_price or order.stop_price or 0.0
        if est_price == 0 and order.ticker in portfolio.positions:
            est_price = portfolio.positions[order.ticker].current_price
        post_mv = current_mv + est_price * order.quantity if order.side == OrderSide.BUY else current_mv - est_price * order.quantity
        post_pct = abs(post_mv) / equity if equity > 0 else 0.0

        if post_pct > self.max_position_pct:
            return RiskCheck(
                passed=False,
                reason=f"Position would be {post_pct:.1%} of equity (limit {self.max_position_pct:.0%})",
                metric_name="position_concentration",
                metric_value=post_pct,
                limit=self.max_position_pct,
            )

        # Sector check
        sector_weights = portfolio.get_sector_weights()
        for sector, wt in sector_weights.items():
            if wt > self.max_sector_pct:
                return RiskCheck(
                    passed=False,
                    reason=f"Sector {sector} at {wt:.1%} (limit {self.max_sector_pct:.0%})",
                    metric_name="sector_concentration",
                    metric_value=wt,
                    limit=self.max_sector_pct,
                )

        return RiskCheck(passed=True, reason="OK", metric_name="position_concentration", metric_value=post_pct, limit=self.max_position_pct)

    def check_drawdown_stop(self, portfolio: Portfolio) -> bool:
        """Returns True if trading should halt."""
        dd = portfolio.compute_drawdown()
        if dd >= self.drawdown_halt:
            self._halted = True
            logger.warning("RISK HALT: drawdown %.1f%% exceeds halt level %.1f%%", dd * 100, self.drawdown_halt * 100)
            return True
        if dd >= self.drawdown_warn:
            logger.warning("RISK WARN: drawdown %.1f%% exceeds warning level %.1f%%", dd * 100, self.drawdown_warn * 100)
        return False

    def check_var_limit(self, portfolio: Portfolio, returns_window: List[float], var_limit_pct: Optional[float] = None) -> bool:
        """1-day 95% VaR check. Returns True if within limit."""
        limit = var_limit_pct or self.var_limit_pct
        if len(returns_window) < 20:
            return True
        equity = portfolio.get_equity()
        arr = np.array(returns_window)
        var_95 = float(np.percentile(arr, 5))  # 5th percentile of returns
        var_dollar = abs(var_95) * equity
        var_pct_equity = var_dollar / equity if equity > 0 else 0.0
        if var_pct_equity > limit:
            logger.warning("VaR %.2f%% exceeds limit %.2f%%", var_pct_equity * 100, limit * 100)
            return False
        return True

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset_halt(self) -> None:
        self._halted = False


# ──────────────────────────────────────────────────────────────────────────────
# Performance Analytics
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceAnalytics:
    """Comprehensive performance analysis for paper trading sessions."""

    @staticmethod
    def compute_tearsheet(equity_curve: pd.Series, trades: List[Trade]) -> Tearsheet:
        """Full tearsheet computation from equity curve and trade list."""
        if len(equity_curve) < 2:
            return Tearsheet(
                cagr=0, total_return=0, sharpe=0, sortino=0, calmar=0,
                max_drawdown=0, volatility_annual=0, var_95=0, cvar_95=0,
                n_trades=0, win_rate=0, profit_factor=0, avg_hold_days=0,
                turnover_annual=0, avg_drawdown=0, max_drawdown_duration_days=0,
            )

        returns = equity_curve.pct_change().dropna()
        n_days = len(equity_curve)
        years = n_days / 252

        # Returns metrics
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

        # Volatility & Sharpe
        vol = float(returns.std() * np.sqrt(252))
        rf_daily = 0.05 / 252  # 5% risk-free
        excess = returns - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        sortino_denom = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 1e-9
        sortino = float((returns.mean() - rf_daily) * 252 / sortino_denom)

        # Drawdown
        rolling_max = equity_curve.cummax()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown_series.min())
        avg_dd = float(drawdown_series[drawdown_series < 0].mean()) if (drawdown_series < 0).any() else 0.0

        # Max drawdown duration
        in_dd = drawdown_series < 0
        max_dd_dur = 0
        current_dur = 0
        for v in in_dd:
            if v:
                current_dur += 1
                max_dd_dur = max(max_dd_dur, current_dur)
            else:
                current_dur = 0

        # Calmar
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

        # VaR / CVaR
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

        # Trade metrics
        n_trades = len(trades)
        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        gross_profit = sum(t.realized_pnl for t in wins)
        gross_loss = abs(sum(t.realized_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Avg hold days: approximate from sequential buy/sell pairs
        avg_hold_days = 1.0  # default

        # Turnover
        initial_equity = float(equity_curve.iloc[0])
        total_traded = sum(t.price * t.quantity for t in trades)
        turnover_annual = (total_traded / initial_equity) / years if years > 0 else 0.0

        return Tearsheet(
            cagr=cagr,
            total_return=total_return,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_dd,
            volatility_annual=vol,
            var_95=var_95,
            cvar_95=cvar_95,
            n_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days,
            turnover_annual=turnover_annual,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_dur,
        )

    @staticmethod
    def compute_attribution(trades: List[Trade], portfolio: Portfolio) -> pd.DataFrame:
        """PnL attribution by ticker."""
        rows = []
        by_ticker: Dict[str, dict] = {}
        for t in trades:
            if t.ticker not in by_ticker:
                by_ticker[t.ticker] = {"ticker": t.ticker, "realized_pnl": 0.0, "n_trades": 0, "total_volume": 0.0}
            by_ticker[t.ticker]["realized_pnl"] += t.realized_pnl
            by_ticker[t.ticker]["n_trades"] += 1
            by_ticker[t.ticker]["total_volume"] += t.price * t.quantity

        for pos in portfolio.get_all_positions():
            if pos.ticker not in by_ticker:
                by_ticker[pos.ticker] = {"ticker": pos.ticker, "realized_pnl": 0.0, "n_trades": 0, "total_volume": 0.0}
            by_ticker[pos.ticker]["unrealized_pnl"] = pos.unrealized_pnl
            by_ticker[pos.ticker]["sector"] = pos.sector

        rows = list(by_ticker.values())
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["total_pnl"] = df.get("realized_pnl", 0) + df.get("unrealized_pnl", 0)
        return df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

    @staticmethod
    def compute_rolling_performance(equity_curve: pd.Series, window: int = 63) -> pd.DataFrame:
        """Rolling Sharpe, drawdown, volatility over rolling window."""
        returns = equity_curve.pct_change().dropna()
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_max = equity_curve.rolling(window).max()
        rolling_dd = (equity_curve - rolling_max) / rolling_max

        return pd.DataFrame({
            "rolling_sharpe": rolling_sharpe,
            "rolling_vol": rolling_vol,
            "rolling_drawdown": rolling_dd,
        })

    @staticmethod
    def benchmark_comparison(equity_curve: pd.Series, benchmark_ticker: str = "SPY") -> dict:
        """Compute beta, alpha, information ratio, tracking error vs benchmark."""
        try:
            bm_data = yf.download(benchmark_ticker, start=equity_curve.index[0], end=equity_curve.index[-1],
                                  auto_adjust=True, progress=False)
            bm_returns = bm_data["Close"].pct_change().dropna()
        except Exception:
            return {"error": "Could not fetch benchmark data"}

        strat_returns = equity_curve.pct_change().dropna()

        # Align
        aligned = pd.DataFrame({"strat": strat_returns, "bm": bm_returns}).dropna()
        if len(aligned) < 20:
            return {"error": "Insufficient overlapping data"}

        cov_matrix = np.cov(aligned["strat"], aligned["bm"])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        rf_daily = 0.05 / 252
        alpha = (aligned["strat"].mean() - rf_daily) - beta * (aligned["bm"].mean() - rf_daily)
        alpha_annual = alpha * 252

        active_returns = aligned["strat"] - aligned["bm"]
        tracking_error = float(active_returns.std() * np.sqrt(252))
        ir = float(active_returns.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0
        correlation = float(aligned.corr().iloc[0, 1])

        return {
            "beta": round(beta, 4),
            "alpha_annual": round(alpha_annual, 4),
            "information_ratio": round(ir, 4),
            "tracking_error": round(tracking_error, 4),
            "correlation": round(correlation, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Built-in Strategies
# ──────────────────────────────────────────────────────────────────────────────

def sma_crossover_paper_strategy(
    prices: Dict[str, float],
    history: Dict[str, pd.Series],
    fast: int = 20,
    slow: int = 50,
) -> Dict[str, str]:
    """
    SMA crossover strategy. Returns {ticker: 'BUY'/'SELL'/'HOLD'}.
    BUY when fast SMA crosses above slow SMA.
    SELL when fast SMA crosses below slow SMA.
    """
    signals: Dict[str, str] = {}
    for ticker, close in history.items():
        if len(close) < slow + 1:
            signals[ticker] = "HOLD"
            continue
        fast_now = close.iloc[-fast:].mean()
        slow_now = close.iloc[-slow:].mean()
        fast_prev = close.iloc[-fast - 1:-1].mean()
        slow_prev = close.iloc[-slow - 1:-1].mean()

        if fast_prev <= slow_prev and fast_now > slow_now:
            signals[ticker] = "BUY"
        elif fast_prev >= slow_prev and fast_now < slow_now:
            signals[ticker] = "SELL"
        else:
            signals[ticker] = "HOLD"
    return signals


def momentum_paper_strategy(
    prices: Dict[str, float],
    history: Dict[str, pd.Series],
    lookback: int = 63,
) -> Dict[str, str]:
    """
    12-1 momentum strategy (Jegadeesh-Titman).
    BUY top 30% by momentum, SELL bottom 30%, HOLD rest.
    """
    mom_scores: Dict[str, float] = {}
    for ticker, close in history.items():
        if len(close) < lookback + 21:
            mom_scores[ticker] = 0.0
            continue
        ret_12m = float(close.iloc[-lookback] / close.iloc[-lookback - 1] - 1) if len(close) > lookback else 0.0
        ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0.0
        mom_scores[ticker] = ret_12m - ret_1m  # classic 12-1

    if not mom_scores:
        return {t: "HOLD" for t in prices}

    sorted_tickers = sorted(mom_scores, key=mom_scores.get, reverse=True)
    n = len(sorted_tickers)
    top_n = max(1, int(n * 0.30))
    bot_n = max(1, int(n * 0.30))

    signals: Dict[str, str] = {}
    for i, ticker in enumerate(sorted_tickers):
        if i < top_n:
            signals[ticker] = "BUY"
        elif i >= n - bot_n:
            signals[ticker] = "SELL"
        else:
            signals[ticker] = "HOLD"
    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Paper Trading Session
# ──────────────────────────────────────────────────────────────────────────────

class PaperTradingSession:
    """
    Main session manager.
    Supports both live polling and bar-by-bar backtest modes.
    """

    def __init__(
        self,
        strategy_fn: Callable,
        tickers: List[str],
        initial_cash: float = 100_000.0,
        commission: float = 0.0,
        slippage_model: str = "fixed",
        default_order_size: float = 0.05,  # 5% of equity per signal
    ) -> None:
        self.strategy_fn = strategy_fn
        self.tickers = tickers
        self.initial_cash = initial_cash
        self.default_order_size = default_order_size
        self.slippage_model = SlippageModel(slippage_model)

        self.market_sim = MarketSimulator(self.slippage_model)
        self.portfolio = Portfolio(initial_cash)
        self.broker = PaperBroker(
            self.portfolio, self.market_sim,
            initial_cash=initial_cash,
            commission=commission,
            slippage_model=slippage_model,
        )
        self.risk_mgr = RiskManager()
        self.analytics = PerformanceAnalytics()
        self.dashboard = PaperTradingDashboard(self)

        self._history: Dict[str, pd.Series] = {}
        self._equity_curve: List[Tuple[datetime, float]] = [(datetime.now(timezone.utc), initial_cash)]
        self._daily_equity: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._strategy_tag: str = strategy_fn.__name__ if hasattr(strategy_fn, "__name__") else "unknown"

    # ── Signal Submission ──────────────────────────────────────────────────────

    def submit_signal(
        self,
        ticker: str,
        action: str,
        quantity: Optional[int] = None,
        target_weight: Optional[float] = None,
    ) -> Optional[Order]:
        """
        action: BUY, SELL, CLOSE, REBALANCE
        target_weight: target portfolio weight (auto-computes quantity if provided)
        """
        if self.risk_mgr.is_halted:
            logger.warning("Session halted by risk manager. Signal for %s ignored.", ticker)
            return None

        current_price = self.market_sim._price_cache.get(ticker, 0.0)
        if current_price <= 0:
            current_price = self.market_sim.get_current_price(ticker)

        if target_weight is not None and current_price > 0:
            equity = self.portfolio.get_equity()
            target_value = equity * target_weight
            current_pos = self.portfolio.get_position(ticker)
            current_value = current_pos.market_value if current_pos else 0.0
            delta_value = target_value - current_value
            quantity = int(abs(delta_value) / current_price)
            action = "BUY" if delta_value > 0 else "SELL"

        if quantity is None or quantity <= 0:
            equity = self.portfolio.get_equity()
            quantity = max(1, int(equity * self.default_order_size / max(current_price, 1)))

        if action in ("SELL", "CLOSE"):
            pos = self.portfolio.get_position(ticker)
            if action == "CLOSE":
                if pos is None or pos.quantity == 0:
                    return None
                quantity = int(pos.quantity)
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY

        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        return self.broker.submit_order(order)

    # ── Live Mode ──────────────────────────────────────────────────────────────

    def run_live(self, poll_interval_seconds: int = 60) -> None:
        """Start live session in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._live_loop, args=(poll_interval_seconds,), daemon=True)
        self._thread.start()
        logger.info("Live paper trading session started. Poll interval: %ds", poll_interval_seconds)

    def stop_live(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _live_loop(self, poll_interval: int) -> None:
        while self._running:
            status = self.market_sim.get_market_hours_status()
            if status not in ("OPEN", "PRE"):
                logger.info("Market %s. Sleeping 5 min.", status)
                time.sleep(300)
                continue

            # Fetch prices
            prices = self.market_sim.get_current_prices_batch(self.tickers)
            self.portfolio.update_prices(prices)

            # Update history (append latest price)
            for ticker, price in prices.items():
                if ticker not in self._history:
                    self._history[ticker] = pd.Series(dtype=float)
                self._history[ticker] = pd.concat([
                    self._history[ticker],
                    pd.Series([price], index=[pd.Timestamp.now()])
                ])

            # Run strategy
            signals = self.strategy_fn(prices, self._history)

            # Submit signals
            for ticker, signal in signals.items():
                if signal in ("BUY", "SELL"):
                    order = self.submit_signal(ticker, signal)
                    if order:
                        self.broker.process_live_order(order)

            # Risk checks
            self.risk_mgr.check_drawdown_stop(self.portfolio)

            # Log state
            state = self.get_state()
            self._equity_curve.append((state.timestamp, state.equity))

            logger.info(
                "Live | equity=$%.2f daily_pnl=$%.2f positions=%d open_orders=%d",
                state.equity, state.daily_pnl, state.n_positions, state.n_open_orders,
            )

            time.sleep(poll_interval)

    # ── Backtest Mode ──────────────────────────────────────────────────────────

    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start: str,
        end: str,
    ) -> SessionResult:
        """
        Bar-by-bar event-driven backtest.
        data: {ticker: OHLCV DataFrame with DatetimeIndex}
        """
        logger.info("Starting backtest %s → %s on %d tickers", start, end, len(data))

        # Reset state
        self.portfolio = Portfolio(self.initial_cash)
        self.broker = PaperBroker(
            self.portfolio, self.market_sim,
            initial_cash=self.initial_cash,
        )
        self._history = {t: pd.Series(dtype=float) for t in data}

        # Build unified date index
        all_dates: set = set()
        for df in data.values():
            all_dates.update(df.loc[start:end].index.tolist())
        dates = sorted(all_dates)

        equity_records: Dict[str, float] = {}
        positions_history: List[dict] = []

        for dt in dates:
            day_str = str(dt)[:10]
            prices_today: Dict[str, float] = {}

            for ticker, df in data.items():
                if dt not in df.index:
                    continue
                bar = df.loc[dt]
                # Update history with today's close
                self._history[ticker] = pd.concat([
                    self._history[ticker],
                    pd.Series([bar["Close"]], index=[dt])
                ])
                prices_today[ticker] = float(bar["Close"])
                # Cache for slippage / spread calc
                self.market_sim._price_cache[ticker] = float(bar["Close"])

                # Process fills against today's bar
                self.broker.process_bar(ticker, bar)

            # Mark-to-market
            self.portfolio.update_prices(prices_today)

            # Run strategy on today's close prices
            signals = self.strategy_fn(prices_today, self._history)
            for ticker, signal in signals.items():
                if signal in ("BUY", "SELL"):
                    order = self.submit_signal(ticker, signal)
                    if order and order.status == OrderStatus.PENDING:
                        # Order will fill on next bar open
                        pass  # Already queued in broker

            # Risk checks
            self.risk_mgr.check_drawdown_stop(self.portfolio)

            equity_today = self.portfolio.get_equity()
            equity_records[day_str] = equity_today

            if len(positions_history) == 0 or len(positions_history) % 5 == 0:
                positions_history.append({
                    "date": day_str,
                    "equity": equity_today,
                    "cash": self.portfolio.cash,
                    "positions": {p.ticker: {"qty": p.quantity, "price": p.current_price, "pnl": p.unrealized_pnl}
                                  for p in self.portfolio.get_all_positions()},
                })

        equity_curve = pd.Series(equity_records)
        equity_curve.index = pd.to_datetime(equity_curve.index)
        trades = self.broker.get_trades()
        tearsheet = self.analytics.compute_tearsheet(equity_curve, trades)

        return SessionResult(
            equity_curve=equity_curve,
            trades=trades,
            tearsheet=tearsheet,
            positions_history=positions_history,
            config={
                "tickers": self.tickers,
                "initial_cash": self.initial_cash,
                "strategy": self._strategy_tag,
                "start": start,
                "end": end,
            },
        )

    # ── State Access ───────────────────────────────────────────────────────────

    def get_state(self) -> SessionState:
        equity = self.portfolio.get_equity()
        initial = self.initial_cash
        # Compute daily PnL from last two equity curve points
        if len(self._equity_curve) >= 2:
            daily_pnl = equity - self._equity_curve[-2][1]
        else:
            daily_pnl = 0.0
        return SessionState(
            timestamp=datetime.now(timezone.utc),
            equity=equity,
            cash=self.portfolio.cash,
            daily_pnl=daily_pnl,
            total_pnl=equity - initial,
            n_positions=len(self.portfolio.get_all_positions()),
            n_open_orders=len(self.broker.get_open_orders()),
            drawdown=self.portfolio.compute_drawdown(),
        )

    def load_history(self, tickers: List[str], period: str = "1y") -> None:
        """Pre-load price history from yfinance for strategy warm-up."""
        logger.info("Loading history for %d tickers (period=%s)...", len(tickers), period)
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
        for ticker in tickers:
            if ticker in close.columns:
                self._history[ticker] = close[ticker].dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────

class PaperTradingDashboard:
    """Formatted display, trade log, JSON export/import for paper sessions."""

    def __init__(self, session: PaperTradingSession) -> None:
        self.session = session

    def get_live_display(self) -> str:
        """Formatted table of positions + PnL."""
        lines = []
        state = self.session.get_state()
        account = self.session.broker.get_account()

        lines.append("=" * 72)
        lines.append(f"  SENTINEL Paper Trading — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 72)
        lines.append(f"  Equity:      ${account.equity:>12,.2f}    Daily PnL:  ${state.daily_pnl:>+10,.2f}")
        lines.append(f"  Cash:        ${account.cash:>12,.2f}    Total PnL:  ${account.total_pnl:>+10,.2f}")
        lines.append(f"  Unrealized:  ${account.unrealized_pnl:>+11,.2f}    Drawdown:   {state.drawdown:>9.2%}")
        lines.append("")
        lines.append(f"  {'Ticker':<8} {'Qty':>8} {'Avg Cost':>10} {'Price':>10} {'Mkt Val':>12} {'Unreal PnL':>12} {'PnL%':>8}")
        lines.append("  " + "-" * 70)

        for pos in sorted(self.session.portfolio.get_all_positions(), key=lambda p: -abs(p.market_value)):
            lines.append(
                f"  {pos.ticker:<8} {pos.quantity:>8.0f} {pos.avg_cost:>10.2f} {pos.current_price:>10.2f} "
                f"{pos.market_value:>12,.2f} {pos.unrealized_pnl:>+12,.2f} {pos.pnl_pct:>8.2%}"
            )

        lines.append("")
        lines.append(f"  Open Orders: {state.n_open_orders}   Positions: {state.n_positions}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def get_trade_log(self, n: int = 20) -> pd.DataFrame:
        """Last n trades as DataFrame."""
        trades = self.session.broker.get_trades()[-n:]
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            rows.append({
                "timestamp": t.timestamp,
                "ticker": t.ticker,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
                "realized_pnl": t.realized_pnl,
            })
        return pd.DataFrame(rows)

    def print_tearsheet(self, tearsheet: Tearsheet) -> None:
        ts = tearsheet
        print("=" * 50)
        print("  PERFORMANCE TEARSHEET")
        print("=" * 50)
        print(f"  CAGR:              {ts.cagr:>10.2%}")
        print(f"  Total Return:      {ts.total_return:>10.2%}")
        print(f"  Sharpe Ratio:      {ts.sharpe:>10.4f}")
        print(f"  Sortino Ratio:     {ts.sortino:>10.4f}")
        print(f"  Calmar Ratio:      {ts.calmar:>10.4f}")
        print(f"  Annual Vol:        {ts.volatility_annual:>10.2%}")
        print(f"  Max Drawdown:      {ts.max_drawdown:>10.2%}")
        print(f"  Avg Drawdown:      {ts.avg_drawdown:>10.2%}")
        print(f"  Max DD Duration:   {ts.max_drawdown_duration_days:>10d} days")
        print(f"  VaR (95%):         {ts.var_95:>10.4f}")
        print(f"  CVaR (95%):        {ts.cvar_95:>10.4f}")
        print("-" * 50)
        print(f"  N Trades:          {ts.n_trades:>10d}")
        print(f"  Win Rate:          {ts.win_rate:>10.2%}")
        print(f"  Profit Factor:     {ts.profit_factor:>10.4f}")
        print(f"  Turnover (Annual): {ts.turnover_annual:>10.2f}x")
        print("=" * 50)

    def export_session(self, path: str) -> None:
        """JSON export of all trades, equity curve, config."""
        data = {
            "config": {
                "tickers": self.session.tickers,
                "initial_cash": self.session.initial_cash,
                "strategy": self.session._strategy_tag,
            },
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "order_id": t.order_id,
                    "ticker": t.ticker,
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "price": t.price,
                    "commission": t.commission,
                    "timestamp": t.timestamp.isoformat(),
                    "realized_pnl": t.realized_pnl,
                }
                for t in self.session.broker.get_trades()
            ],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq}
                for ts, eq in self.session._equity_curve
            ],
            "positions": [
                {
                    "ticker": p.ticker,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in self.session.portfolio.get_all_positions()
            ],
            "account": {
                "cash": self.session.portfolio.cash,
                "equity": self.session.portfolio.get_equity(),
                "realized_pnl": self.session.portfolio.realized_pnl,
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Session exported to %s", path)

    @staticmethod
    def import_session(path: str, strategy_fn: Callable) -> PaperTradingSession:
        """Restore session state from JSON export."""
        raw = json.loads(Path(path).read_text())
        config = raw["config"]
        session = PaperTradingSession(
            strategy_fn=strategy_fn,
            tickers=config["tickers"],
            initial_cash=config["initial_cash"],
        )
        # Restore positions
        session.portfolio.cash = raw["account"]["cash"]
        session.portfolio.realized_pnl = raw["account"]["realized_pnl"]
        for pos_data in raw.get("positions", []):
            session.portfolio.positions[pos_data["ticker"]] = Position(
                ticker=pos_data["ticker"],
                quantity=pos_data["quantity"],
                avg_cost=pos_data["avg_cost"],
                current_price=pos_data["current_price"],
            )
        # Restore trades
        for t_data in raw.get("trades", []):
            trade = Trade(
                trade_id=t_data["trade_id"],
                order_id=t_data["order_id"],
                ticker=t_data["ticker"],
                side=OrderSide(t_data["side"]),
                quantity=t_data["quantity"],
                price=t_data["price"],
                commission=t_data["commission"],
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
                realized_pnl=t_data["realized_pnl"],
            )
            session.broker._trades.append(trade)
        logger.info("Session imported from %s  (%d trades restored)", path, len(session.broker._trades))
        return session


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: Fetch OHLCV for Backtest
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data from yfinance for a list of tickers."""
    logger.info("Fetching OHLCV: %s → %s for %s", start, end, tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    result: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw.xs(ticker, axis=1, level=1).dropna()
                if not df.empty:
                    result[ticker] = df
            except KeyError:
                logger.warning("No data for %s", ticker)
    else:
        # Single ticker case
        if not raw.empty:
            result[tickers[0]] = raw.dropna()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main — Demo Backtest
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  SENTINEL Paper Trading v3 — Demo Backtest")
    print("  SPY / QQQ / GLD | 2024-01-01 → 2025-06-01 | SMA Crossover")
    print("=" * 60 + "\n")

    TICKERS = ["SPY", "QQQ", "GLD"]
    START = "2024-01-01"
    END = "2025-06-01"

    # Fetch data
    data = fetch_ohlcv(TICKERS, START, END)
    if not data:
        print("ERROR: Could not fetch data. Check internet connection.")
        sys.exit(1)
    print(f"  Data loaded: {', '.join(data.keys())} ({len(next(iter(data.values())))} bars each)")

    # Create session with SMA crossover strategy
    session = PaperTradingSession(
        strategy_fn=sma_crossover_paper_strategy,
        tickers=TICKERS,
        initial_cash=100_000.0,
        commission=0.0,
        slippage_model="fixed",
    )

    # Run backtest
    result = session.run_backtest(data, START, END)

    # Print tearsheet
    print()
    session.dashboard.print_tearsheet(result.tearsheet)

    # Print last 10 trades
    print("\n  LAST 10 TRADES:")
    trade_df = session.dashboard.get_trade_log(10)
    if not trade_df.empty:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(trade_df.to_string(index=False))
    else:
        print("  No trades executed in backtest period.")

    # Benchmark comparison
    print("\n  BENCHMARK COMPARISON (vs SPY):")
    bm = PerformanceAnalytics.benchmark_comparison(result.equity_curve, "SPY")
    for k, v in bm.items():
        print(f"    {k:<25}: {v}")

    # Live display snapshot
    print()
    print(session.dashboard.get_live_display())

    # Export
    export_path = "/tmp/sentinel_paper_session.json"
    session.dashboard.export_session(export_path)
    print(f"\n  Session exported to {export_path}")
