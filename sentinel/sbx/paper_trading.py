"""Paper trading simulator with full order management, portfolio tracking, and P&L attribution.

Dimension targeted:
  dim_066 — Paper trading simulator  score 5 → 9

Persistence: SQLite at ~/.sentinel/paper_trading.db (auto-created on first use).
Price data: yfinance (free, no API key required).
Commission model: zero-commission (Robinhood / IBKR Lite style).

Schema:
  accounts       — id, name, cash, initial_capital, created_at
  orders         — all Order fields (limit/stop orders pending until triggered)
  trades         — executed fills with realized P&L
  equity_history — daily equity snapshots for Sharpe / drawdown computation
"""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Literal, Optional

import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

DB_PATH = Path.home() / ".sentinel" / "paper_trading.db"

# ── Pydantic models ───────────────────────────────────────────────────────────

class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id: str
    ticker: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: Literal["pending", "filled", "partial", "cancelled", "rejected"] = "pending"
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    commission: float = 0.0
    notes: Optional[str] = None


class Position(BaseModel):
    account_id: str
    ticker: str
    quantity: float             # positive = long, negative = short
    avg_cost: float             # average cost basis per share
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    realized_pnl: float = 0.0
    opened_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class Portfolio(BaseModel):
    account_id: str
    account_name: str
    cash: float
    initial_capital: float
    positions: list[Position] = Field(default_factory=list)
    total_market_value: Optional[float] = None
    total_equity: Optional[float] = None
    total_pnl: Optional[float] = None
    total_pnl_pct: Optional[float] = None
    day_pnl: Optional[float] = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class Trade(BaseModel):
    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id: str
    ticker: str
    side: str
    quantity: float
    price: float
    commission: float
    executed_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    order_id: str
    realized_pnl: Optional[float] = None


class PerformanceReport(BaseModel):
    account_id: str
    period_start: date
    period_end: date
    initial_equity: float
    final_equity: float
    total_return: float                 # decimal, e.g. 0.15 = 15%
    annualized_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    best_trade: Optional[Trade] = None
    worst_trade: Optional[Trade] = None


# ── PaperTradingEngine ────────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Full paper trading simulator backed by SQLite.

    Usage:
        engine = PaperTradingEngine()
        account_id = engine.create_account("My Strategy", initial_capital=100_000)
        order = await engine.submit_order(account_id, "AAPL", "buy", 10)
        pf = await engine.get_portfolio(account_id)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("PaperTradingEngine initialized", db=str(self._db))

    # ── DB plumbing ───────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a SQLite connection with row_factory set."""
        conn = sqlite3.connect(str(self._db))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create all tables if they don't already exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id              TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    cash            REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    created_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id       TEXT PRIMARY KEY,
                    account_id     TEXT NOT NULL,
                    ticker         TEXT NOT NULL,
                    side           TEXT NOT NULL,
                    order_type     TEXT NOT NULL,
                    quantity       REAL NOT NULL,
                    limit_price    REAL,
                    stop_price     REAL,
                    status         TEXT NOT NULL DEFAULT 'pending',
                    submitted_at   TEXT NOT NULL,
                    filled_at      TEXT,
                    filled_price   REAL,
                    filled_qty     REAL NOT NULL DEFAULT 0,
                    commission     REAL NOT NULL DEFAULT 0,
                    notes          TEXT,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id     TEXT PRIMARY KEY,
                    account_id   TEXT NOT NULL,
                    ticker       TEXT NOT NULL,
                    side         TEXT NOT NULL,
                    quantity     REAL NOT NULL,
                    price        REAL NOT NULL,
                    commission   REAL NOT NULL DEFAULT 0,
                    executed_at  TEXT NOT NULL,
                    order_id     TEXT NOT NULL,
                    realized_pnl REAL,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS positions (
                    account_id   TEXT NOT NULL,
                    ticker       TEXT NOT NULL,
                    quantity     REAL NOT NULL,
                    avg_cost     REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    opened_at    TEXT NOT NULL,
                    PRIMARY KEY (account_id, ticker),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS equity_history (
                    account_id   TEXT NOT NULL,
                    date         TEXT NOT NULL,
                    equity_value REAL NOT NULL,
                    PRIMARY KEY (account_id, date),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id);
                CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);
                CREATE INDEX IF NOT EXISTS idx_equity_account ON equity_history(account_id);
            """)

    # ── Account management ────────────────────────────────────────────────────

    def create_account(
        self,
        account_name: str,
        initial_capital: float = 100_000.0,
    ) -> str:
        """Create a new paper trading account. Returns the new account_id."""
        account_id = str(uuid.uuid4())[:12]
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO accounts (id, name, cash, initial_capital, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (account_id, account_name, initial_capital, initial_capital, now),
            )
        logger.info(
            "Paper trading account created",
            account_id=account_id,
            name=account_name,
            capital=initial_capital,
        )
        return account_id

    def list_accounts(self) -> list[dict]:
        """Return all accounts with id, name, cash, initial_capital, and created_at."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM accounts ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def _get_account(self, account_id: str) -> dict:
        """Fetch a single account row or raise ValueError if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Account not found: {account_id}")
        return dict(row)

    # ── Price fetching ────────────────────────────────────────────────────────

    async def get_quote(self, ticker: str) -> float:
        """Fetch current price via yfinance. Returns the last close (or fast_info price)."""
        try:
            price = await asyncio.to_thread(self._yf_price, ticker.upper())
            if price and price > 0:
                return price
            raise ValueError(f"Zero or None price returned for {ticker}")
        except Exception as exc:
            raise ValueError(f"Price fetch failed for {ticker}: {exc}") from exc

    @staticmethod
    def _yf_price(ticker: str) -> float:
        """Synchronous yfinance price lookup (runs in thread executor)."""
        t = yf.Ticker(ticker)
        # fast_info.last_price is most current during market hours
        try:
            price = t.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        # Fallback: last close from 5-day history
        hist = t.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
        raise ValueError(f"No price data from yfinance for {ticker}")

    # ── Order submission ──────────────────────────────────────────────────────

    async def submit_order(
        self,
        account_id: str,
        ticker: str,
        side: Literal["buy", "sell"],
        quantity: float,
        order_type: Literal["market", "limit", "stop", "stop_limit"] = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Submit a paper order.

        Market orders are filled immediately at the current price.
        Limit/stop orders are stored as pending and filled via fill_pending_orders().
        Validates:
          - Buy: sufficient cash (quantity × price + commission)
          - Sell: sufficient shares held
        """
        ticker_up = ticker.upper()
        account = self._get_account(account_id)

        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        order = Order(
            account_id=account_id,
            ticker=ticker_up,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        if order_type == "market":
            current_price = await self.get_quote(ticker_up)
            order = await self._fill_order(order, current_price, account)
        else:
            # Validate and store pending
            if side == "buy" and limit_price:
                required_cash = quantity * limit_price
                if account["cash"] < required_cash:
                    order = order.model_copy(update={
                        "status": "rejected",
                        "notes": f"Insufficient cash: need {required_cash:.2f}, have {account['cash']:.2f}",
                    })
            elif side == "sell":
                position = self._get_position(account_id, ticker_up)
                available = position["quantity"] if position else 0.0
                if available < quantity:
                    order = order.model_copy(update={
                        "status": "rejected",
                        "notes": f"Insufficient shares: need {quantity}, have {available}",
                    })

            if order.status != "rejected":
                self._save_order(order)
                logger.info(
                    "Pending order stored",
                    order_id=order.order_id,
                    ticker=ticker_up,
                    side=side,
                    order_type=order_type,
                )

        return order

    async def _fill_order(
        self, order: Order, price: float, account: dict
    ) -> Order:
        """Execute a fill at the given price; update cash, positions, and trades."""
        commission = self._compute_commission(order.quantity, price)
        now = datetime.now(tz=timezone.utc)

        if order.side == "buy":
            total_cost = order.quantity * price + commission
            if account["cash"] < total_cost:
                filled = order.model_copy(update={
                    "status": "rejected",
                    "notes": f"Insufficient cash: need {total_cost:.2f}, have {account['cash']:.2f}",
                })
                self._save_order(filled)
                return filled
        else:
            # Sell: check shares
            position = self._get_position(order.account_id, order.ticker)
            available = position["quantity"] if position else 0.0
            if available < order.quantity:
                filled = order.model_copy(update={
                    "status": "rejected",
                    "notes": f"Insufficient shares: need {order.quantity}, have {available}",
                })
                self._save_order(filled)
                return filled

        # All good — execute
        realized_pnl = self._update_position(
            order.account_id, order.ticker, order.side, order.quantity, price
        )

        # Update cash
        if order.side == "buy":
            new_cash = account["cash"] - order.quantity * price - commission
        else:
            new_cash = account["cash"] + order.quantity * price - commission

        with self._conn() as conn:
            conn.execute(
                "UPDATE accounts SET cash = ? WHERE id = ?",
                (new_cash, order.account_id),
            )

        trade = Trade(
            account_id=order.account_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=commission,
            executed_at=now,
            order_id=order.order_id,
            realized_pnl=realized_pnl,
        )
        self._save_trade(trade)

        filled_order = order.model_copy(update={
            "status": "filled",
            "filled_at": now,
            "filled_price": price,
            "filled_qty": order.quantity,
            "commission": commission,
        })
        self._save_order(filled_order)

        logger.info(
            "Order filled",
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            qty=order.quantity,
            price=price,
            realized_pnl=realized_pnl,
        )
        return filled_order

    # ── Pending order fills ───────────────────────────────────────────────────

    async def fill_pending_orders(self, account_id: str) -> list[Order]:
        """
        Check all pending limit/stop orders against current market prices.
        Fill any that have triggered. Returns the list of newly filled orders.
        """
        pending = self.get_open_orders(account_id)
        if not pending:
            return []

        # Fetch unique tickers
        tickers = list({o.ticker for o in pending})
        price_tasks = [self.get_quote(t) for t in tickers]
        price_results = await asyncio.gather(*price_tasks, return_exceptions=True)
        prices: dict[str, float] = {}
        for t, result in zip(tickers, price_results):
            if isinstance(result, (int, float)):
                prices[t] = result

        account = self._get_account(account_id)
        newly_filled: list[Order] = []

        for order in pending:
            current_price = prices.get(order.ticker)
            if current_price is None:
                continue

            should_fill = False
            fill_price = current_price

            if order.order_type == "limit":
                # Buy limit: fill if current price ≤ limit
                # Sell limit: fill if current price ≥ limit
                if order.side == "buy" and order.limit_price and current_price <= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price  # best price guarantee
                elif order.side == "sell" and order.limit_price and current_price >= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price

            elif order.order_type == "stop":
                # Buy stop: fill if price ≥ stop (breakout buy)
                # Sell stop: fill if price ≤ stop (stop-loss)
                if order.side == "buy" and order.stop_price and current_price >= order.stop_price:
                    should_fill = True
                elif order.side == "sell" and order.stop_price and current_price <= order.stop_price:
                    should_fill = True

            elif order.order_type == "stop_limit":
                # Stop triggers, then fills at limit price
                stop_triggered = False
                if order.side == "buy" and order.stop_price and current_price >= order.stop_price:
                    stop_triggered = True
                elif order.side == "sell" and order.stop_price and current_price <= order.stop_price:
                    stop_triggered = True
                if stop_triggered and order.limit_price:
                    if order.side == "buy" and current_price <= order.limit_price:
                        should_fill = True
                        fill_price = order.limit_price
                    elif order.side == "sell" and current_price >= order.limit_price:
                        should_fill = True
                        fill_price = order.limit_price

            if should_fill:
                filled = await self._fill_order(order, fill_price, account)
                newly_filled.append(filled)
                # Refresh account cash for next iteration
                account = self._get_account(account_id)

        return newly_filled

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def get_portfolio(self, account_id: str) -> Portfolio:
        """
        Load all positions, fetch current prices, and compute unrealized P&L.
        Records an equity snapshot for performance tracking.
        """
        account = self._get_account(account_id)
        raw_positions = self._get_all_positions(account_id)

        if not raw_positions:
            equity = account["cash"]
            self.record_equity_snapshot(account_id, equity)
            return Portfolio(
                account_id=account_id,
                account_name=account["name"],
                cash=account["cash"],
                initial_capital=account["initial_capital"],
                total_market_value=0.0,
                total_equity=equity,
                total_pnl=equity - account["initial_capital"],
                total_pnl_pct=(equity - account["initial_capital"]) / account["initial_capital"]
                if account["initial_capital"] else 0.0,
                day_pnl=0.0,
            )

        # Fetch current prices for all tickers in parallel
        tickers = list({p["ticker"] for p in raw_positions})
        price_tasks = [self.get_quote(t) for t in tickers]
        price_results = await asyncio.gather(*price_tasks, return_exceptions=True)
        prices: dict[str, float] = {}
        for t, result in zip(tickers, price_results):
            if isinstance(result, (int, float)) and result > 0:
                prices[t] = result

        positions: list[Position] = []
        total_market_value = 0.0

        for p in raw_positions:
            ticker = p["ticker"]
            qty = p["quantity"]
            avg_cost = p["avg_cost"]
            realized_pnl = p["realized_pnl"]
            opened_at_str = p["opened_at"]
            try:
                opened_at = datetime.fromisoformat(opened_at_str)
            except Exception:
                opened_at = datetime.now(tz=timezone.utc)

            current_price = prices.get(ticker)
            market_value: Optional[float] = None
            unrealized_pnl: Optional[float] = None
            unrealized_pnl_pct: Optional[float] = None

            if current_price is not None:
                market_value = round(qty * current_price, 4)
                cost_basis = qty * avg_cost
                unrealized_pnl = round(market_value - cost_basis, 4)
                unrealized_pnl_pct = (
                    round(unrealized_pnl / abs(cost_basis), 6) if cost_basis != 0 else 0.0
                )
                total_market_value += market_value

            positions.append(Position(
                account_id=account_id,
                ticker=ticker,
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                realized_pnl=realized_pnl,
                opened_at=opened_at,
            ))

        total_equity = round(account["cash"] + total_market_value, 4)
        total_pnl = round(total_equity - account["initial_capital"], 4)
        total_pnl_pct = (
            round(total_pnl / account["initial_capital"], 6)
            if account["initial_capital"] else 0.0
        )

        # Day P&L: compare to yesterday's equity snapshot
        day_pnl = self._compute_day_pnl(account_id, total_equity)

        self.record_equity_snapshot(account_id, total_equity)

        return Portfolio(
            account_id=account_id,
            account_name=account["name"],
            cash=round(account["cash"], 4),
            initial_capital=account["initial_capital"],
            positions=positions,
            total_market_value=round(total_market_value, 4),
            total_equity=total_equity,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            day_pnl=day_pnl,
        )

    def _compute_day_pnl(self, account_id: str, current_equity: float) -> Optional[float]:
        """Return equity change vs yesterday's snapshot, or None if no history."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT equity_value FROM equity_history WHERE account_id = ? AND date = ?",
                (account_id, yesterday),
            ).fetchone()
        if row:
            return round(current_equity - row["equity_value"], 4)
        return None

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_open_orders(self, account_id: str) -> list[Order]:
        """Return all pending orders for this account."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE account_id = ? AND status = 'pending' "
                "ORDER BY submitted_at DESC",
                (account_id,),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled, False if not found/fillable."""
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = ? AND status = 'pending'",
                (order_id,),
            )
        cancelled = result.rowcount > 0
        if cancelled:
            logger.info("Order cancelled", order_id=order_id)
        return cancelled

    def get_trade_history(self, account_id: str, days_back: int = 30) -> list[Trade]:
        """Return all filled trades from the last N days, newest first."""
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE account_id = ? AND executed_at >= ? "
                "ORDER BY executed_at DESC",
                (account_id, since),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # ── Equity snapshot ───────────────────────────────────────────────────────

    def record_equity_snapshot(self, account_id: str, equity: float) -> None:
        """Upsert today's equity value for performance tracking."""
        today = date.today().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_history (account_id, date, equity_value) VALUES (?, ?, ?) "
                "ON CONFLICT(account_id, date) DO UPDATE SET equity_value = excluded.equity_value",
                (account_id, today, round(equity, 4)),
            )

    # ── Performance report ────────────────────────────────────────────────────

    async def generate_report(
        self,
        account_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> PerformanceReport:
        """
        Compute full performance analytics from trade history and equity snapshots.
        Metrics: total return, annualized return, Sharpe ratio, max drawdown,
                 win rate, avg win/loss, profit factor.
        """
        end_date = end_date or date.today()
        # Find earliest trade date if start not specified
        if start_date is None:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT MIN(executed_at) as first FROM trades WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
            if row and row["first"]:
                try:
                    start_date = datetime.fromisoformat(row["first"]).date()
                except Exception:
                    start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=30)

        account = self._get_account(account_id)
        trades = self.get_trade_history(account_id, days_back=(end_date - start_date).days + 1)
        trades_in_period = [
            t for t in trades
            if start_date <= t.executed_at.date() <= end_date
        ]

        # Equity series from history table
        with self._conn() as conn:
            eq_rows = conn.execute(
                "SELECT date, equity_value FROM equity_history WHERE account_id = ? "
                "AND date BETWEEN ? AND ? ORDER BY date ASC",
                (account_id, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()

        equity_series = [r["equity_value"] for r in eq_rows]
        initial_equity = equity_series[0] if equity_series else account["initial_capital"]
        final_equity = equity_series[-1] if equity_series else account["initial_capital"]

        total_return = (final_equity - initial_equity) / initial_equity if initial_equity else 0.0
        n_days = max(1, (end_date - start_date).days)
        annualized_return = (1 + total_return) ** (365.0 / n_days) - 1

        # Sharpe ratio (annualized, risk-free = 0)
        sharpe: Optional[float] = None
        if len(equity_series) >= 5:
            try:
                daily_returns = [
                    (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
                    for i in range(1, len(equity_series))
                    if equity_series[i - 1] > 0
                ]
                if daily_returns:
                    n = len(daily_returns)
                    mean_r = sum(daily_returns) / n
                    variance = sum((r - mean_r) ** 2 for r in daily_returns) / max(1, n - 1)
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    sharpe = round((mean_r / std) * math.sqrt(252), 4) if std > 0 else None
            except Exception:
                pass

        # Maximum drawdown
        max_drawdown = 0.0
        if equity_series:
            peak = equity_series[0]
            for v in equity_series:
                peak = max(peak, v)
                dd = (peak - v) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)

        # Win/loss stats from realized P&L in trades
        pnl_values = [t.realized_pnl for t in trades_in_period if t.realized_pnl is not None]
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]

        n_trades = len(pnl_values)
        win_rate = len(wins) / n_trades if n_trades else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0

        # Best / worst trade
        best_trade: Optional[Trade] = None
        worst_trade: Optional[Trade] = None
        if trades_in_period:
            ranked = sorted(
                [t for t in trades_in_period if t.realized_pnl is not None],
                key=lambda t: t.realized_pnl or 0,
            )
            if ranked:
                worst_trade = ranked[0]
                best_trade = ranked[-1]

        return PerformanceReport(
            account_id=account_id,
            period_start=start_date,
            period_end=end_date,
            initial_equity=round(initial_equity, 2),
            final_equity=round(final_equity, 2),
            total_return=round(total_return, 6),
            annualized_return=round(annualized_return, 6),
            sharpe_ratio=sharpe,
            max_drawdown=round(max_drawdown, 6),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 4),
            n_trades=n_trades,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    # ── Commission ────────────────────────────────────────────────────────────

    def _compute_commission(self, quantity: float, price: float) -> float:
        """Zero-commission model (Robinhood / IBKR Lite style)."""
        return 0.0

    # ── Position management ───────────────────────────────────────────────────

    def _update_position(
        self,
        account_id: str,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Optional[float]:
        """
        Update (or create) a position in the DB.
        Returns realized P&L when a sell reduces or closes a long position,
        or None for buys / partial-reduction logic that isn't closed.
        """
        realized_pnl: Optional[float] = None
        existing = self._get_position(account_id, ticker)
        now = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            if existing is None:
                # New position
                new_qty = quantity if side == "buy" else -quantity
                conn.execute(
                    "INSERT INTO positions (account_id, ticker, quantity, avg_cost, realized_pnl, opened_at) "
                    "VALUES (?, ?, ?, ?, 0, ?)",
                    (account_id, ticker, new_qty, price, now),
                )
            else:
                old_qty = existing["quantity"]
                old_avg = existing["avg_cost"]
                old_realized = existing["realized_pnl"]

                if side == "buy":
                    # Increase long (or reduce short)
                    if old_qty >= 0:
                        # Adding to long: update average cost
                        new_qty = old_qty + quantity
                        new_avg = (old_qty * old_avg + quantity * price) / new_qty
                    else:
                        # Covering short
                        cover_qty = min(quantity, abs(old_qty))
                        realized_pnl = cover_qty * (old_avg - price)  # short profit = cost - cover
                        remaining_short = abs(old_qty) - cover_qty
                        excess_long = quantity - cover_qty
                        new_qty = -remaining_short + excess_long
                        new_avg = price if new_qty > 0 else old_avg
                        old_realized += realized_pnl
                else:  # sell
                    if old_qty > 0:
                        # Reducing long
                        sell_qty = min(quantity, old_qty)
                        realized_pnl = sell_qty * (price - old_avg)
                        new_qty = old_qty - quantity
                        new_avg = old_avg  # avg_cost unchanged on reduction
                        old_realized += realized_pnl
                    else:
                        # Increasing short
                        new_qty = old_qty - quantity
                        new_avg = (abs(old_qty) * old_avg + quantity * price) / abs(new_qty)

                if abs(new_qty) < 1e-9:
                    # Position closed — remove row
                    conn.execute(
                        "DELETE FROM positions WHERE account_id = ? AND ticker = ?",
                        (account_id, ticker),
                    )
                else:
                    conn.execute(
                        "UPDATE positions SET quantity = ?, avg_cost = ?, realized_pnl = ? "
                        "WHERE account_id = ? AND ticker = ?",
                        (new_qty, new_avg, old_realized, account_id, ticker),
                    )

        return realized_pnl

    def _get_position(self, account_id: str, ticker: str) -> Optional[dict]:
        """Return a position row as a dict, or None if not held."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE account_id = ? AND ticker = ?",
                (account_id, ticker),
            ).fetchone()
        return dict(row) if row else None

    def _get_all_positions(self, account_id: str) -> list[dict]:
        """Return all open positions for an account."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE account_id = ? AND ABS(quantity) > 1e-9",
                (account_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── DB serialization helpers ──────────────────────────────────────────────

    def _save_order(self, order: Order) -> None:
        """Insert or replace an order row."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders "
                "(order_id, account_id, ticker, side, order_type, quantity, "
                "limit_price, stop_price, status, submitted_at, filled_at, "
                "filled_price, filled_qty, commission, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    order.order_id, order.account_id, order.ticker, order.side,
                    order.order_type, order.quantity, order.limit_price, order.stop_price,
                    order.status,
                    order.submitted_at.isoformat() if order.submitted_at else None,
                    order.filled_at.isoformat() if order.filled_at else None,
                    order.filled_price, order.filled_qty, order.commission, order.notes,
                ),
            )

    def _save_trade(self, trade: Trade) -> None:
        """Insert a trade row."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trades "
                "(trade_id, account_id, ticker, side, quantity, price, commission, "
                "executed_at, order_id, realized_pnl) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    trade.trade_id, trade.account_id, trade.ticker, trade.side,
                    trade.quantity, trade.price, trade.commission,
                    trade.executed_at.isoformat(),
                    trade.order_id, trade.realized_pnl,
                ),
            )

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        d = dict(row)
        for dt_field in ("submitted_at", "filled_at"):
            if d.get(dt_field):
                try:
                    d[dt_field] = datetime.fromisoformat(d[dt_field])
                except (ValueError, TypeError):
                    d[dt_field] = None
        return Order(**d)

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> Trade:
        d = dict(row)
        if d.get("executed_at"):
            try:
                d["executed_at"] = datetime.fromisoformat(d["executed_at"])
            except (ValueError, TypeError):
                d["executed_at"] = datetime.now(tz=timezone.utc)
        return Trade(**d)


# ── Module-level convenience helpers ─────────────────────────────────────────

_default_engine: Optional[PaperTradingEngine] = None


def _get_engine() -> PaperTradingEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = PaperTradingEngine()
    return _default_engine


def _resolve_account(account_id: str) -> str:
    """Resolve 'default' → the first account in the DB, or raise if none exist."""
    if account_id != "default":
        return account_id
    engine = _get_engine()
    accounts = engine.list_accounts()
    if not accounts:
        raise ValueError("No paper trading accounts exist. Call PaperTradingEngine().create_account() first.")
    return accounts[0]["id"]


async def buy(ticker: str, quantity: float, account_id: str = "default") -> Order:
    """Convenience: submit a market buy order on the default account."""
    acct = _resolve_account(account_id)
    return await _get_engine().submit_order(acct, ticker, "buy", quantity, order_type="market")


async def sell(ticker: str, quantity: float, account_id: str = "default") -> Order:
    """Convenience: submit a market sell order on the default account."""
    acct = _resolve_account(account_id)
    return await _get_engine().submit_order(acct, ticker, "sell", quantity, order_type="market")


async def portfolio(account_id: str = "default") -> Portfolio:
    """Convenience: fetch the current portfolio for the default account."""
    acct = _resolve_account(account_id)
    return await _get_engine().get_portfolio(acct)
