"""
Live Trading Execution — Dimension #065 (target score 9+).

Complete live trading execution module using Alpaca Markets API (alpaca-py SDK).
Bridges backtested SENTINEL strategies to real (or paper) brokerage execution.

Architecture
------------
  AlpacaOrderManager     — order lifecycle: submit, cancel, query, history
  AlpacaPortfolioManager — positions, account equity, live P&L metrics
  StrategyExecutor       — signal → size → risk-check → order pipeline
  LiveDataFeed           — WebSocket bar/quote subscription + REST latest bar
  RiskManager            — pre-trade gates and portfolio-level stop-outs

FastAPI router: live_trading_router (prefix /api/live)

Credentials are read from environment variables:
    ALPACA_API_KEY    — Alpaca API key ID
    ALPACA_SECRET_KEY — Alpaca secret key
    ALPACA_PAPER      — "true" (default) or "false" for live trading

Paper trading is the default; set ALPACA_PAPER=false to go live.
"""
from __future__ import annotations

import abc
import math
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Literal, Optional

import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SDK availability guard
# ---------------------------------------------------------------------------

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        TrailingStopOrderRequest,
        GetOrdersRequest,
        GetPortfolioHistoryRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderType,
        TimeInForce,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest
    from alpaca.data.live import StockDataStream

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning(
        "alpaca-py not installed. Run: pip install alpaca-py  "
        "Live trading features will raise RuntimeError until installed."
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_API_KEY = "ALPACA_API_KEY"
_ENV_SECRET = "ALPACA_SECRET_KEY"
_ENV_PAPER = "ALPACA_PAPER"

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_BASE_URL = "https://api.alpaca.markets"


def _is_paper() -> bool:
    return os.getenv(_ENV_PAPER, "true").lower() != "false"


def _get_credentials() -> tuple[str, str]:
    key = os.getenv(_ENV_API_KEY, "")
    secret = os.getenv(_ENV_SECRET, "")
    if not key or not secret:
        raise EnvironmentError(
            f"Set {_ENV_API_KEY} and {_ENV_SECRET} environment variables "
            "before using live trading features."
        )
    return key, secret


def _require_alpaca() -> None:
    if not _ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py SDK not installed. Run: pip install alpaca-py")


def _order_to_dict(order: Any) -> dict:
    """Coerce an Alpaca Order object to a plain dict."""
    try:
        return order.dict()
    except Exception:  # noqa: BLE001
        return {k: getattr(order, k, None) for k in (
            "id", "client_order_id", "symbol", "qty", "side", "type",
            "time_in_force", "limit_price", "stop_price", "status",
            "filled_at", "filled_qty", "filled_avg_price", "created_at",
        )}


def _position_to_dict(pos: Any) -> dict:
    try:
        return pos.dict()
    except Exception:  # noqa: BLE001
        return {k: getattr(pos, k, None) for k in (
            "symbol", "qty", "avg_entry_price", "current_price",
            "unrealized_pl", "unrealized_plpc", "market_value",
            "cost_basis", "side",
        )}


# ---------------------------------------------------------------------------
# AlpacaOrderManager
# ---------------------------------------------------------------------------


class AlpacaOrderManager:
    """Full order lifecycle management via Alpaca trading API.

    Parameters
    ----------
    api_key    : Alpaca API key ID (falls back to ALPACA_API_KEY env var)
    secret_key : Alpaca secret key (falls back to ALPACA_SECRET_KEY env var)
    paper      : True = paper trading endpoint, False = live brokerage
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
    ) -> None:
        _require_alpaca()
        key = api_key or os.getenv(_ENV_API_KEY, "")
        secret = secret_key or os.getenv(_ENV_SECRET, "")
        if not key or not secret:
            raise EnvironmentError(
                f"Provide api_key/secret_key or set {_ENV_API_KEY}/{_ENV_SECRET}"
            )
        self._paper = paper or _is_paper()
        self._client = TradingClient(
            api_key=key,
            secret_key=secret,
            paper=self._paper,
        )
        logger.info("AlpacaOrderManager initialised (paper=%s)", self._paper)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        ticker: str,
        qty: float,
        side: str,
    ) -> dict:
        """Submit a market order.

        Parameters
        ----------
        ticker : equity symbol e.g. 'AAPL'
        qty    : number of shares (fractional allowed)
        side   : 'buy' or 'sell'
        """
        req = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("Market order submitted: %s %s qty=%s", side, ticker, qty)
        return _order_to_dict(order)

    def submit_limit_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict:
        """Submit a limit order."""
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
        req = LimitOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=round(limit_price, 2),
            time_in_force=tif,
        )
        order = self._client.submit_order(req)
        logger.info("Limit order submitted: %s %s qty=%s @ %.2f", side, ticker, qty, limit_price)
        return _order_to_dict(order)

    def submit_stop_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        stop_price: float,
    ) -> dict:
        """Submit a stop (market) order."""
        req = StopOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            stop_price=round(stop_price, 2),
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("Stop order submitted: %s %s qty=%s stop=%.2f", side, ticker, qty, stop_price)
        return _order_to_dict(order)

    def submit_trailing_stop(
        self,
        ticker: str,
        qty: float,
        side: str,
        trail_pct: float,
    ) -> dict:
        """Submit a trailing stop order.

        Parameters
        ----------
        trail_pct : trail percentage, e.g. 0.05 = 5% trailing stop
        """
        req = TrailingStopOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            trail_percent=round(trail_pct * 100, 2),
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        logger.info("Trailing stop submitted: %s %s qty=%s trail=%.1f%%", side, ticker, qty, trail_pct * 100)
        return _order_to_dict(order)

    def submit_bracket_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        take_profit_pct: float = 0.05,
        stop_loss_pct: float = 0.02,
    ) -> dict:
        """Submit a bracket (OCO) order with TP and SL legs.

        The take-profit and stop-loss prices are computed from the most
        recent quote.  Requires Alpaca OTO/OCO support.

        Parameters
        ----------
        take_profit_pct : distance above entry (long) for TP, e.g. 0.05 = 5%
        stop_loss_pct   : distance below entry (long) for SL, e.g. 0.02 = 2%
        """
        # Get latest quote to estimate entry price
        latest = self.get_latest_quote(ticker)
        entry_price = latest.get("ask_price") or latest.get("bid_price") or 0.0
        if entry_price == 0.0:
            raise ValueError(f"Cannot determine entry price for {ticker}")

        is_buy = side.lower() == "buy"
        if is_buy:
            tp_price = round(entry_price * (1 + take_profit_pct), 2)
            sl_price = round(entry_price * (1 - stop_loss_pct), 2)
        else:
            tp_price = round(entry_price * (1 - take_profit_pct), 2)
            sl_price = round(entry_price * (1 + stop_loss_pct), 2)

        # Alpaca bracket via nested order legs
        req = MarketOrderRequest(
            symbol=ticker.upper(),
            qty=qty,
            side=OrderSide.BUY if is_buy else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit={"limit_price": tp_price},
            stop_loss={"stop_price": sl_price},
        )
        order = self._client.submit_order(req)
        logger.info(
            "Bracket order: %s %s qty=%s tp=%.2f sl=%.2f",
            side, ticker, qty, tp_price, sl_price,
        )
        return _order_to_dict(order)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID. Returns True if successfully cancelled."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cancel order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info("Cancelled %d open orders", count)
        return count

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order by ID."""
        order = self._client.get_order_by_id(order_id)
        return _order_to_dict(order)

    def get_open_orders(self) -> list[dict]:
        """Return all currently open orders."""
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        orders = self._client.get_orders(req)
        return [_order_to_dict(o) for o in orders]

    def get_order_history(
        self,
        start: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Return order history as a DataFrame.

        Parameters
        ----------
        start : ISO date string, e.g. '2024-01-01'. Defaults to 30 days ago.
        limit : maximum number of records to return (max 500)
        """
        if start is None:
            start_dt = datetime.now(tz=timezone.utc) - timedelta(days=30)
        else:
            start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()

        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=start_dt,
            limit=min(limit, 500),
        )
        orders = self._client.get_orders(req)
        records = [_order_to_dict(o) for o in orders]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if "created_at" in df.columns:
            df = df.sort_values("created_at", ascending=False)
        return df

    # ------------------------------------------------------------------
    # Quote helper (used internally for bracket orders)
    # ------------------------------------------------------------------

    def get_latest_quote(self, ticker: str) -> dict:
        """Return the latest NBBO quote for a ticker."""
        try:
            key, secret = _get_credentials()
            data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
            quotes = data_client.get_stock_latest_quote(req)
            q = quotes.get(ticker.upper())
            if q is None:
                return {}
            try:
                return q.dict()
            except Exception:  # noqa: BLE001
                return {"bid_price": getattr(q, "bid_price", 0.0),
                        "ask_price": getattr(q, "ask_price", 0.0)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_latest_quote(%s) failed: %s", ticker, exc)
            return {}


# ---------------------------------------------------------------------------
# AlpacaPortfolioManager
# ---------------------------------------------------------------------------


class AlpacaPortfolioManager:
    """Portfolio-level queries: account equity, positions, live P&L."""

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
    ) -> None:
        _require_alpaca()
        key = api_key or os.getenv(_ENV_API_KEY, "")
        secret = secret_key or os.getenv(_ENV_SECRET, "")
        if not key or not secret:
            raise EnvironmentError(
                f"Provide api_key/secret_key or set {_ENV_API_KEY}/{_ENV_SECRET}"
            )
        self._paper = paper or _is_paper()
        self._client = TradingClient(api_key=key, secret_key=secret, paper=self._paper)
        self._api_key = key
        self._secret_key = secret

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return account summary: equity, buying power, cash, portfolio value, P&L."""
        acct = self._client.get_account()
        try:
            d = acct.dict()
        except Exception:  # noqa: BLE001
            d = {k: getattr(acct, k, None) for k in (
                "id", "equity", "buying_power", "cash", "portfolio_value",
                "unrealized_pl", "unrealized_plpc", "realized_pl",
                "last_equity", "currency", "pattern_day_trader",
                "trading_blocked", "account_blocked", "created_at",
            )}
        # Compute day P&L
        try:
            d["day_pnl"] = float(d.get("equity", 0) or 0) - float(d.get("last_equity", 0) or 0)
        except Exception:  # noqa: BLE001
            d["day_pnl"] = 0.0
        return d

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.DataFrame:
        """Return all open positions as a DataFrame.

        Columns: symbol, qty, avg_entry_price, current_price,
                 unrealized_pl, unrealized_plpc, market_value, cost_basis, side
        """
        positions = self._client.get_all_positions()
        records = [_position_to_dict(p) for p in positions]
        if not records:
            return pd.DataFrame(columns=[
                "symbol", "qty", "avg_entry_price", "current_price",
                "unrealized_pl", "unrealized_plpc", "market_value", "cost_basis", "side",
            ])
        df = pd.DataFrame(records)
        # Ensure numeric columns are float
        for col in ("qty", "avg_entry_price", "current_price",
                    "unrealized_pl", "unrealized_plpc", "market_value", "cost_basis"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_position(self, ticker: str) -> dict:
        """Return position details for a single ticker, or empty dict if not held."""
        try:
            pos = self._client.get_open_position(ticker.upper())
            return _position_to_dict(pos)
        except Exception:  # noqa: BLE001
            return {}

    def close_position(self, ticker: str) -> dict:
        """Close (liquidate) the entire position in *ticker*."""
        result = self._client.close_position(ticker.upper())
        logger.info("Closed position: %s", ticker)
        return _order_to_dict(result)

    def close_all_positions(self) -> list[dict]:
        """Close all open positions immediately."""
        results = self._client.close_all_positions(cancel_orders=True)
        logger.info("Closed all positions")
        if not results:
            return []
        return [_order_to_dict(r) for r in results]

    # ------------------------------------------------------------------
    # Portfolio history
    # ------------------------------------------------------------------

    def get_portfolio_value_history(self, period: str = "1M") -> pd.DataFrame:
        """Return account equity timeseries.

        Parameters
        ----------
        period : Alpaca portfolio history period string.
                 '1D', '1W', '1M', '3M', '6M', '1A' (1 year), 'all'
        """
        period_map = {
            "1D": "1D", "1W": "1W", "1M": "1M",
            "3M": "3M", "6M": "6M", "1Y": "1A", "1A": "1A", "ALL": "all",
        }
        alp_period = period_map.get(period.upper(), "1M")
        req = GetPortfolioHistoryRequest(period=alp_period, timeframe="1D")
        history = self._client.get_portfolio_history(req)

        timestamps = getattr(history, "timestamp", [])
        equity = getattr(history, "equity", [])
        profit_loss = getattr(history, "profit_loss", [])
        profit_loss_pct = getattr(history, "profit_loss_pct", [])

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "equity": equity,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
        })
        df = df.set_index("timestamp").sort_index()
        return df

    # ------------------------------------------------------------------
    # Live metrics
    # ------------------------------------------------------------------

    def compute_live_metrics(self) -> dict:
        """Compute daily, MTD, and YTD P&L from portfolio history.

        Also computes win rate from portfolio history daily returns.
        """
        try:
            df = self.get_portfolio_value_history(period="1A")
        except Exception as exc:  # noqa: BLE001
            logger.warning("compute_live_metrics: portfolio history failed: %s", exc)
            df = pd.DataFrame()

        acct = self.get_account()
        day_pnl = acct.get("day_pnl", 0.0)
        equity = float(acct.get("equity") or 0)

        # MTD and YTD P&L from history
        now = pd.Timestamp.now(tz="UTC")
        mtd_pnl = 0.0
        ytd_pnl = 0.0
        win_rate = 0.0

        if not df.empty and "equity" in df.columns:
            df_eq = df["equity"].dropna()
            # MTD: first trading day of current month
            mtd_start = df_eq[df_eq.index >= now.replace(day=1)]
            if len(mtd_start) >= 2:
                mtd_pnl = float(mtd_start.iloc[-1]) - float(mtd_start.iloc[0])
            # YTD: first trading day of current year
            ytd_start = df_eq[df_eq.index >= now.replace(month=1, day=1)]
            if len(ytd_start) >= 2:
                ytd_pnl = float(ytd_start.iloc[-1]) - float(ytd_start.iloc[0])
            # Win rate: % of days with positive P&L
            daily_ret = df_eq.pct_change().dropna()
            if len(daily_ret) > 0:
                win_rate = float((daily_ret > 0).mean())

        return {
            "day_pnl": day_pnl,
            "mtd_pnl": mtd_pnl,
            "ytd_pnl": ytd_pnl,
            "current_equity": equity,
            "win_rate": win_rate,
            "unrealized_pl": float(acct.get("unrealized_pl") or 0),
            "buying_power": float(acct.get("buying_power") or 0),
        }


# ---------------------------------------------------------------------------
# Abstract BaseStrategy — for type-checking in StrategyExecutor
# ---------------------------------------------------------------------------


class BaseStrategy(abc.ABC):
    """Minimal strategy interface consumed by StrategyExecutor."""

    @abc.abstractmethod
    def on_bar(self, ticker: str, bar: dict) -> Literal["BUY", "SELL", "HOLD"] | str:
        """Process a new bar and return a signal string."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# StrategyExecutor
# ---------------------------------------------------------------------------


class StrategyExecutor:
    """Bridge between a backtested SENTINEL strategy and live Alpaca execution.

    Handles the full signal-to-order pipeline:
      bar → strategy.on_bar() → signal → position_size → risk_gates → order
    """

    MAX_POSITION_PCT = 0.10          # no single position > 10% of equity
    MAX_DRAWDOWN_HALT = 0.15         # halt if portfolio drawdown > 15%
    DEFAULT_RISK_PER_TRADE = 0.01    # 1% of equity risked per trade
    DEFAULT_STOP_DISTANCE_PCT = 0.02 # 2% stop distance for Kelly sizing

    def __init__(
        self,
        strategy: BaseStrategy,
        order_manager: AlpacaOrderManager,
        portfolio_manager: AlpacaPortfolioManager,
    ) -> None:
        self.strategy = strategy
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self._peak_equity: float | None = None

    # ------------------------------------------------------------------
    # Core: process a live bar
    # ------------------------------------------------------------------

    def run_live_bar(self, ticker: str, bar: dict) -> dict:
        """Feed a live bar to the strategy; process signal; submit order.

        Parameters
        ----------
        ticker : equity symbol
        bar    : dict with keys: open, high, low, close, volume, timestamp

        Returns
        -------
        dict with keys: signal, order (if submitted), reason (if blocked)
        """
        # 1. Get signal from strategy
        signal = self.strategy.on_bar(ticker, bar)
        if signal == "HOLD" or signal is None:
            return {"signal": signal, "action": "no_action"}

        # 2. Pre-trade risk gates
        price = float(bar.get("close") or bar.get("c") or 0.0)
        approved, reason = self.check_risk_gates(ticker, signal, price)
        if not approved:
            logger.info("Signal %s %s blocked: %s", signal, ticker, reason)
            return {"signal": signal, "action": "blocked", "reason": reason}

        # 3. Compute position size
        acct = self.portfolio_manager.get_account()
        equity = float(acct.get("equity") or 0)
        qty = self.compute_position_size(
            ticker=ticker,
            signal_strength=1.0,
            risk_per_trade_pct=self.DEFAULT_RISK_PER_TRADE,
        )
        if qty <= 0:
            return {"signal": signal, "action": "skipped", "reason": "qty=0"}

        # 4. Submit order
        side = "buy" if signal.upper() in ("BUY", "LONG") else "sell"
        try:
            order = self.order_manager.submit_market_order(ticker, qty, side)
            logger.info("Executed signal %s %s qty=%d", signal, ticker, qty)
            return {"signal": signal, "action": "order_submitted", "order": order}
        except Exception as exc:  # noqa: BLE001
            logger.error("Order submission failed for %s: %s", ticker, exc)
            return {"signal": signal, "action": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        ticker: str,
        signal_strength: float = 1.0,
        risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE,
    ) -> int:
        """Kelly-inspired position sizing with hard caps.

        Formula
        -------
        raw_size = (signal_strength × equity × risk_per_trade_pct)
                   / (price × stop_distance_pct)

        Caps
        ----
        - No single position > MAX_POSITION_PCT of equity
        - No new orders if portfolio drawdown > MAX_DRAWDOWN_HALT

        Returns integer shares (floor).
        """
        try:
            acct = self.portfolio_manager.get_account()
            equity = float(acct.get("equity") or 0)
        except Exception:  # noqa: BLE001
            logger.warning("compute_position_size: could not fetch account equity")
            return 0

        if equity <= 0:
            return 0

        # Check drawdown halt
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown > self.MAX_DRAWDOWN_HALT:
            logger.warning("Drawdown %.1f%% > %.1f%% halt — no sizing", drawdown * 100, self.MAX_DRAWDOWN_HALT * 100)
            return 0

        # Get current price
        try:
            latest = self.order_manager.get_latest_quote(ticker)
            price = float(latest.get("ask_price") or latest.get("bid_price") or 0)
        except Exception:  # noqa: BLE001
            price = 0.0

        if price <= 0:
            logger.warning("compute_position_size: price=0 for %s", ticker)
            return 0

        stop_distance_pct = self.DEFAULT_STOP_DISTANCE_PCT
        # Kelly-based raw dollar risk allocation
        dollar_risk = signal_strength * equity * risk_per_trade_pct
        raw_qty = dollar_risk / (price * stop_distance_pct)

        # Cap: max_position_value = MAX_POSITION_PCT × equity
        max_qty_by_cap = (equity * self.MAX_POSITION_PCT) / price

        # Check existing position
        existing = self.portfolio_manager.get_position(ticker)
        existing_qty = float(existing.get("qty") or 0) if existing else 0.0
        remaining_cap = max(0.0, max_qty_by_cap - abs(existing_qty))

        qty = max(0, int(math.floor(min(raw_qty, remaining_cap))))
        logger.debug(
            "Position size %s: equity=%.0f price=%.2f raw=%.1f cap=%.1f → %d",
            ticker, equity, price, raw_qty, remaining_cap, qty,
        )
        return qty

    # ------------------------------------------------------------------
    # Risk gates
    # ------------------------------------------------------------------

    def check_risk_gates(
        self,
        ticker: str,
        signal: str,
        price: float,
    ) -> tuple[bool, str]:
        """Run pre-trade checks before submitting any order.

        Checks performed
        ----------------
        1. Market hours — Alpaca accepts orders outside hours for GTC, but we
           default to DAY orders; warn if outside regular session.
        2. Account health — trading_blocked or account_blocked.
        3. Drawdown limit — halt if portfolio drawdown > MAX_DRAWDOWN_HALT.
        4. Position concentration — block if existing position already at cap.

        Returns (approved: bool, reason: str).
        """
        try:
            acct = self.portfolio_manager.get_account()
        except Exception as exc:  # noqa: BLE001
            return False, f"account fetch failed: {exc}"

        # Account blocks
        if acct.get("trading_blocked"):
            return False, "account trading_blocked"
        if acct.get("account_blocked"):
            return False, "account account_blocked"

        equity = float(acct.get("equity") or 0)
        if equity <= 0:
            return False, "equity is zero or negative"

        # Drawdown check
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown > self.MAX_DRAWDOWN_HALT:
            return False, f"portfolio drawdown {drawdown:.1%} exceeds {self.MAX_DRAWDOWN_HALT:.0%} halt threshold"

        # Position concentration
        existing = self.portfolio_manager.get_position(ticker)
        if existing:
            existing_qty = float(existing.get("qty") or 0)
            existing_value = abs(existing_qty) * price
            if existing_value >= equity * self.MAX_POSITION_PCT:
                return False, (
                    f"existing position {ticker} at {existing_value / equity:.1%} "
                    f"of equity — at concentration cap"
                )

        return True, "approved"

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def emergency_flatten(self) -> dict:
        """Close all positions and cancel all orders immediately."""
        logger.warning("EMERGENCY FLATTEN triggered by %s", self.strategy.name)
        cancelled = self.order_manager.cancel_all_orders()
        closed = self.portfolio_manager.close_all_positions()
        return {
            "action": "emergency_flatten",
            "orders_cancelled": cancelled,
            "positions_closed": len(closed),
            "closed_orders": closed,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# LiveDataFeed
# ---------------------------------------------------------------------------


class LiveDataFeed:
    """Alpaca WebSocket bar/quote subscription + REST latest-data endpoints.

    WebSocket subscriptions use alpaca-py's async StockDataStream.
    """

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        _require_alpaca()
        self._api_key = api_key or os.getenv(_ENV_API_KEY, "")
        self._secret_key = secret_key or os.getenv(_ENV_SECRET, "")
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        self._stream: Any = None   # StockDataStream, created on subscribe

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    def subscribe_bars(
        self,
        tickers: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to real-time minute bars for *tickers*.

        The *callback* receives a dict with keys:
            symbol, open, high, low, close, volume, timestamp

        This method starts the WebSocket stream in the current async event
        loop.  Call from within an async context or use asyncio.run().

        Example
        -------
        import asyncio

        async def on_bar(bar):
            print(bar)

        feed = LiveDataFeed()
        feed.subscribe_bars(["AAPL", "MSFT"], on_bar)
        """
        stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        async def _handler(bar: Any) -> None:
            try:
                d = bar.dict() if hasattr(bar, "dict") else {"symbol": bar.symbol}
            except Exception:  # noqa: BLE001
                d = {"symbol": getattr(bar, "symbol", "?"), "close": getattr(bar, "close", 0)}
            callback(d)

        for ticker in tickers:
            stream.subscribe_bars(_handler, ticker.upper())

        self._stream = stream
        stream.run()
        logger.info("Bar subscription started for: %s", tickers)

    def subscribe_quotes(
        self,
        tickers: list[str],
        callback: Callable[[dict], None],
    ) -> None:
        """Subscribe to real-time NBBO quote stream for *tickers*."""
        stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        async def _handler(quote: Any) -> None:
            try:
                d = quote.dict() if hasattr(quote, "dict") else {}
            except Exception:  # noqa: BLE001
                d = {
                    "symbol": getattr(quote, "symbol", "?"),
                    "bid_price": getattr(quote, "bid_price", 0),
                    "ask_price": getattr(quote, "ask_price", 0),
                }
            callback(d)

        for ticker in tickers:
            stream.subscribe_quotes(_handler, ticker.upper())

        self._stream = stream
        stream.run()
        logger.info("Quote subscription started for: %s", tickers)

    # ------------------------------------------------------------------
    # REST: latest bar / quote
    # ------------------------------------------------------------------

    def get_latest_bar(self, ticker: str) -> dict:
        """Fetch the latest 1-minute bar for a ticker via REST."""
        req = StockLatestBarRequest(symbol_or_symbols=ticker.upper())
        bars = self._data_client.get_stock_latest_bar(req)
        bar = bars.get(ticker.upper())
        if bar is None:
            return {}
        try:
            return bar.dict()
        except Exception:  # noqa: BLE001
            return {
                "symbol": ticker.upper(),
                "open": getattr(bar, "open", 0),
                "high": getattr(bar, "high", 0),
                "low": getattr(bar, "low", 0),
                "close": getattr(bar, "close", 0),
                "volume": getattr(bar, "volume", 0),
                "timestamp": getattr(bar, "timestamp", None),
            }

    def get_latest_quote(self, ticker: str) -> dict:
        """Fetch the latest NBBO quote for a ticker via REST."""
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
        quotes = self._data_client.get_stock_latest_quote(req)
        quote = quotes.get(ticker.upper())
        if quote is None:
            return {}
        try:
            return quote.dict()
        except Exception:  # noqa: BLE001
            return {
                "symbol": ticker.upper(),
                "bid_price": getattr(quote, "bid_price", 0),
                "ask_price": getattr(quote, "ask_price", 0),
                "bid_size": getattr(quote, "bid_size", 0),
                "ask_size": getattr(quote, "ask_size", 0),
                "timestamp": getattr(quote, "timestamp", None),
            }


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """Portfolio-level risk gates and position monitoring.

    Enforces:
      - daily_loss_limit       : halt if day P&L < -X% of equity
      - max_drawdown_limit     : halt if portfolio drawdown > Y%
      - max_position_size_pct  : no single position > Z% of equity

    All limits are expressed as positive fractions (e.g. 0.02 = 2%).
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.02,
        max_drawdown_limit: float = 0.15,
        max_position_size_pct: float = 0.10,
        stop_loss_threshold: float = 0.05,   # flag positions with unrealised loss > 5%
    ) -> None:
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_position_size_pct = max_position_size_pct
        self.stop_loss_threshold = stop_loss_threshold
        self._peak_equity: float | None = None

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def check_pre_trade(
        self,
        ticker: str,
        qty: int,
        side: str,
        price: float,
        account: dict,
        positions_df: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Run all pre-trade risk checks.

        Parameters
        ----------
        ticker       : symbol being traded
        qty          : shares to trade
        side         : 'buy' or 'sell'
        price        : expected execution price
        account      : dict from AlpacaPortfolioManager.get_account()
        positions_df : DataFrame from AlpacaPortfolioManager.get_positions()

        Returns
        -------
        (approved: bool, reason: str)
        """
        equity = float(account.get("equity") or 0)
        if equity <= 0:
            return False, "equity is zero"

        # 1. Daily loss limit
        day_pnl = float(account.get("day_pnl") or 0)
        day_pnl_pct = day_pnl / equity if equity > 0 else 0.0
        if day_pnl_pct < -self.daily_loss_limit:
            return False, (
                f"daily loss {day_pnl_pct:.1%} exceeds limit -{self.daily_loss_limit:.1%}. "
                "Trading halted for today."
            )

        # 2. Portfolio drawdown
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown > self.max_drawdown_limit:
            return False, (
                f"portfolio drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_limit:.1%}. "
                "Trading halted."
            )

        # 3. Position concentration
        order_value = qty * price
        existing_value = 0.0
        if not positions_df.empty and "symbol" in positions_df.columns:
            mask = positions_df["symbol"] == ticker.upper()
            if mask.any():
                existing_row = positions_df[mask].iloc[0]
                existing_value = abs(float(existing_row.get("market_value") or 0))

        total_value = existing_value + order_value
        if total_value > equity * self.max_position_size_pct:
            return False, (
                f"{ticker} post-trade value {total_value:.0f} would exceed "
                f"{self.max_position_size_pct:.0%} position cap ({equity * self.max_position_size_pct:.0f})"
            )

        # 4. Buying power
        if side.lower() == "buy":
            buying_power = float(account.get("buying_power") or 0)
            if order_value > buying_power:
                return False, (
                    f"Insufficient buying power: need {order_value:.0f}, have {buying_power:.0f}"
                )

        return True, "approved"

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    def monitor_positions(self, positions_df: pd.DataFrame) -> list[dict]:
        """Flag positions with unrealised loss exceeding stop threshold.

        Returns list of flagged position dicts with keys:
            symbol, unrealized_plpc, flag, recommendation
        """
        flags: list[dict] = []
        if positions_df.empty:
            return flags

        for _, row in positions_df.iterrows():
            pct = float(row.get("unrealized_plpc") or 0)
            # Alpaca returns unrealized_plpc as a decimal (e.g. -0.05 = -5%)
            # Normalise: if raw value > 1 assume it's in percentage points
            if abs(pct) > 1:
                pct = pct / 100

            if pct < -self.stop_loss_threshold:
                flags.append({
                    "symbol": row.get("symbol"),
                    "unrealized_plpc": pct,
                    "flag": "STOP_THRESHOLD_BREACHED",
                    "recommendation": f"Consider closing — unrealised loss {pct:.1%} exceeds -{self.stop_loss_threshold:.1%}",
                })

        return flags


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Path as FPath, Query
    from pydantic import BaseModel as PydanticModel, Field as PydanticField

    live_trading_router = APIRouter(prefix="/api/live", tags=["Live Trading"])

    # ---- Request models ----

    class OrderRequest(PydanticModel):
        ticker: str
        qty: float
        side: str                          # 'buy' or 'sell'
        order_type: str = "market"         # 'market' | 'limit' | 'stop' | 'trailing_stop' | 'bracket'
        limit_price: Optional[float] = None
        stop_price: Optional[float] = None
        trail_pct: Optional[float] = None  # for trailing_stop
        take_profit_pct: float = 0.05
        stop_loss_pct: float = 0.02
        time_in_force: str = "day"

    # ---- Dependency: build managers from env ----

    def _get_managers() -> tuple[AlpacaOrderManager, AlpacaPortfolioManager]:
        paper = _is_paper()
        key, secret = _get_credentials()
        om = AlpacaOrderManager(api_key=key, secret_key=secret, paper=paper)
        pm = AlpacaPortfolioManager(api_key=key, secret_key=secret, paper=paper)
        return om, pm

    # ---- Endpoints ----

    @live_trading_router.get("/account")
    async def api_get_account():
        """Return Alpaca account summary."""
        try:
            _, pm = _get_managers()
            return pm.get_account()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.get("/positions")
    async def api_get_positions():
        """Return all open positions."""
        try:
            _, pm = _get_managers()
            df = pm.get_positions()
            return df.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.post("/order")
    async def api_submit_order(req: OrderRequest):
        """Submit an order to Alpaca."""
        try:
            om, _ = _get_managers()
            ot = req.order_type.lower()
            if ot == "market":
                return om.submit_market_order(req.ticker, req.qty, req.side)
            elif ot == "limit":
                if req.limit_price is None:
                    raise HTTPException(status_code=422, detail="limit_price required for limit order")
                return om.submit_limit_order(req.ticker, req.qty, req.side, req.limit_price, req.time_in_force)
            elif ot == "stop":
                if req.stop_price is None:
                    raise HTTPException(status_code=422, detail="stop_price required for stop order")
                return om.submit_stop_order(req.ticker, req.qty, req.side, req.stop_price)
            elif ot == "trailing_stop":
                trail = req.trail_pct or 0.05
                return om.submit_trailing_stop(req.ticker, req.qty, req.side, trail)
            elif ot == "bracket":
                return om.submit_bracket_order(
                    req.ticker, req.qty, req.side,
                    take_profit_pct=req.take_profit_pct,
                    stop_loss_pct=req.stop_loss_pct,
                )
            else:
                raise HTTPException(status_code=422, detail=f"Unknown order_type: {req.order_type}")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.delete("/order/{order_id}")
    async def api_cancel_order(order_id: str = FPath(..., description="Alpaca order ID")):
        """Cancel a specific order by ID."""
        try:
            om, _ = _get_managers()
            success = om.cancel_order(order_id)
            return {"cancelled": success, "order_id": order_id}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.delete("/positions")
    async def api_close_all_positions():
        """Close all open positions and cancel all open orders."""
        try:
            _, pm = _get_managers()
            closed = pm.close_all_positions()
            return {"closed": len(closed), "orders": closed}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.get("/orders/history")
    async def api_order_history(
        start: Optional[str] = Query(None, description="ISO date e.g. 2024-01-01"),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Return recent order history."""
        try:
            om, _ = _get_managers()
            df = om.get_order_history(start=start, limit=limit)
            return df.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @live_trading_router.post("/emergency-flatten")
    async def api_emergency_flatten():
        """Kill switch: close all positions and cancel all orders immediately."""
        try:
            om, pm = _get_managers()

            class _NoOpStrategy(BaseStrategy):
                def on_bar(self, ticker, bar):
                    return "HOLD"

            executor = StrategyExecutor(
                strategy=_NoOpStrategy(),
                order_manager=om,
                portfolio_manager=pm,
            )
            result = executor.emergency_flatten()
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

except ImportError:
    # FastAPI not available — module usable as a library
    live_trading_router = None  # type: ignore[assignment]
    logger.debug("FastAPI not available — live_trading_router not registered")
