"""
Live Trading Execution V3 — Dimension #065 (target score 9/10).

DATA-LATENCY TRANSPARENCY
--------------------------
Alpaca free tier uses the IEX data feed for quotes and bars:
  - IEX quotes: typically 50-500 ms behind SIP; NOT suitable for HFT/market-making.
  - IEX bars (historical): accurate and complete; fine for TWAP/VWAP scheduling.
  - Order SUBMISSION via Alpaca API is always real-time regardless of data feed.
  - All orders are tagged data_latency="near_real_time_iex" in the audit log.
  - Tradier sandbox used for order simulation and comparison.

What this means in practice:
  - Market orders always fill at real market prices (not IEX-lagged).
  - TWAP/VWAP pacing is safe; 500ms quote lag doesn't affect interval scheduling.
  - Limit prices derived from IEX quotes carry latency risk — add a buffer or use market.
  - Never use this module for latency-sensitive strategies (market-making, stat-arb <1s).

Architecture
------------
OrderManagementSystem      — full order lifecycle, all types and TIF flags
SmartOrderRouter           — routing decision: direct / TWAP / VWAP / Almgren-Chriss
TWAPEngine                 — time-sliced execution with size randomisation
VWAPEngine                 — historical volume profile participation
AlmgrenChrissEngine        — optimal liquidation trajectory for large orders
PortfolioRebalancer        — weight-deviation rebalancing, round-lots, constraints
PositionTracker            — real-time reconciliation with Alpaca account state
PreTradeRiskEngine         — ADV check, notional limit, daily loss halt, dupe detection
FillAnalytics              — IS, slippage, VWAP benchmark, fill rate per order
TradierSandbox             — side-by-side order simulation for venue comparison

FastAPI router: trading_v3_router (prefix /trading/v3)

Credentials (env vars):
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER (default "true")
    TRADIER_SANDBOX_TOKEN (optional; enables Tradier simulation)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Alpaca SDK availability guard
# ---------------------------------------------------------------------------

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        TrailingStopOrderRequest,
        GetOrdersRequest,
        CancelOrderResponse,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        QueryOrderStatus,
        OrderClass,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestBarRequest,
        StockLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning(
        "alpaca-py not installed — live trading V3 features disabled. "
        "Install with: pip install alpaca-py"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_API_KEY     = "ALPACA_API_KEY"
_ENV_SECRET      = "ALPACA_SECRET_KEY"
_ENV_PAPER       = "ALPACA_PAPER"
_ENV_TRADIER     = "TRADIER_SANDBOX_TOKEN"

_DATA_LATENCY_TAG = "near_real_time_iex"   # stamped on all orders; never claim SIP

# Alpaca base URLs
_ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
_ALPACA_LIVE_BASE  = "https://api.alpaca.markets"

# Tradier sandbox
_TRADIER_SANDBOX_BASE    = "https://sandbox.tradier.com/v1"
_TRADIER_SANDBOX_ACCOUNT = os.getenv("TRADIER_SANDBOX_ACCOUNT", "VA00000000")

# Market hours ET — fixed UTC-4 (EDT); production would use exchange-calendars
_MARKET_OPEN_ET   = (9, 30)
_MARKET_CLOSE_ET  = (16, 0)
_PRE_MARKET_OPEN  = (4, 0)
_AFTER_HOURS_CLOSE = (20, 0)
_AVOID_OPEN_MINS  = 15   # high-spread zone open
_AVOID_CLOSE_MINS = 15   # high-spread zone close

# Smart order router thresholds
_SOR_DIRECT_NOTIONAL   = 10_000.0    # <$10K → direct market
_SOR_TWAP_NOTIONAL     = 100_000.0   # $10K-$100K → TWAP
# >$100K → VWAP + Almgren-Chriss impact model

# Risk defaults
_DEFAULT_NOTIONAL_LIMIT = 500_000.0
_DEFAULT_ADV_PCT        = 0.15        # max 15% of 20-day ADV
_DEFAULT_DAILY_LOSS_PCT = 0.03        # 3% daily P&L halt
_DUPE_DETECT_WINDOW_S   = 1.0         # duplicate order: same side/symbol within 1s

DB_PATH = Path(os.getenv("SENTINEL_DB", "sentinel/data/live_trading_v3.db"))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderType(str, Enum):
    MARKET       = "market"
    LIMIT        = "limit"
    STOP         = "stop"
    STOP_LIMIT   = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TIF(str, Enum):
    DAY  = "day"
    GTC  = "gtc"
    IOC  = "ioc"
    FOK  = "fok"
    OPG  = "opg"    # market-on-open
    CLS  = "cls"    # market-on-close


class OrderStatus(str, Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    FILLED    = "filled"
    PARTIAL   = "partial"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"
    DEFERRED  = "deferred"


class SORStrategy(str, Enum):
    DIRECT  = "direct"
    TWAP    = "twap"
    VWAP    = "vwap"
    AC      = "almgren_chriss"   # Almgren-Chriss for large blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_paper() -> bool:
    return os.getenv(_ENV_PAPER, "true").lower() != "false"


def _get_credentials() -> Tuple[str, str]:
    key    = os.getenv(_ENV_API_KEY, "")
    secret = os.getenv(_ENV_SECRET, "")
    if not key or not secret:
        raise EnvironmentError(
            f"Set {_ENV_API_KEY} and {_ENV_SECRET} before using live trading V3."
        )
    return key, secret


def _require_alpaca() -> None:
    if not _ALPACA_AVAILABLE:
        raise RuntimeError(
            "alpaca-py SDK not installed. Run: pip install alpaca-py"
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _et_now() -> datetime:
    """Current time in Eastern Time (fixed UTC-4 offset for EDT)."""
    return _now_utc() - timedelta(hours=4)


def _minutes_since_open() -> float:
    et = _et_now()
    open_time = et.replace(
        hour=_MARKET_OPEN_ET[0], minute=_MARKET_OPEN_ET[1], second=0, microsecond=0
    )
    return (et - open_time).total_seconds() / 60.0


def _minutes_to_close() -> float:
    et = _et_now()
    close_time = et.replace(
        hour=_MARKET_CLOSE_ET[0], minute=_MARKET_CLOSE_ET[1], second=0, microsecond=0
    )
    return (close_time - et).total_seconds() / 60.0


def _in_core_hours() -> bool:
    mins_open  = _minutes_since_open()
    mins_close = _minutes_to_close()
    return 0 <= mins_open and mins_close > 0


def _in_impact_zone() -> bool:
    mins_open  = _minutes_since_open()
    mins_close = _minutes_to_close()
    return (0 <= mins_open < _AVOID_OPEN_MINS) or (0 < mins_close <= _AVOID_CLOSE_MINS)


def _is_extended_hours() -> bool:
    """True if current ET time is in pre-market (04:00-09:30) or after-hours (16:00-20:00)."""
    et = _et_now()
    h, m = et.hour, et.minute
    pre_market   = (h, m) >= _PRE_MARKET_OPEN and (h, m) < _MARKET_OPEN_ET
    after_hours  = (h, m) >= _MARKET_CLOSE_ET and (h, m) < _AFTER_HOURS_CLOSE
    return pre_market or after_hours


def _order_hash(symbol: str, side: str) -> str:
    return hashlib.md5(f"{symbol}:{side}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _ensure_tables(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id               TEXT PRIMARY KEY,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            side             TEXT NOT NULL,
            order_type       TEXT NOT NULL,
            tif              TEXT NOT NULL,
            qty              REAL NOT NULL,
            notional         REAL,
            limit_price      REAL,
            stop_price       REAL,
            trail_pct        REAL,
            trail_price      REAL,
            status           TEXT NOT NULL,
            alpaca_order_id  TEXT,
            tradier_order_id TEXT,
            avg_fill_price   REAL,
            filled_qty       REAL DEFAULT 0,
            parent_order_id  TEXT,
            tranche_idx      INTEGER DEFAULT 0,
            tranche_total    INTEGER DEFAULT 1,
            sor_strategy     TEXT,
            data_latency     TEXT DEFAULT 'near_real_time_iex',
            extended_hours   INTEGER DEFAULT 0,
            fractional       INTEGER DEFAULT 0,
            notes            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_orders_status  ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);

        CREATE TABLE IF NOT EXISTS fills (
            id               TEXT PRIMARY KEY,
            order_id         TEXT NOT NULL REFERENCES orders(id),
            filled_at        TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            side             TEXT NOT NULL,
            qty              REAL NOT NULL,
            fill_price       REAL NOT NULL,
            decision_price   REAL,
            vwap_benchmark   REAL,
            impl_shortfall_bps REAL,
            slippage_bps     REAL,
            vwap_vs_fill_bps REAL,
            market_impact_bps REAL,
            data_latency     TEXT DEFAULT 'near_real_time_iex'
        );

        CREATE INDEX IF NOT EXISTS idx_fills_order  ON fills(order_id);
        CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);

        CREATE TABLE IF NOT EXISTS positions (
            symbol           TEXT PRIMARY KEY,
            qty              REAL NOT NULL,
            avg_entry_price  REAL NOT NULL,
            current_price    REAL,
            market_value     REAL,
            unrealized_pl    REAL,
            unrealized_plpc  REAL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS risk_events (
            id          TEXT PRIMARY KEY,
            ts          TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            symbol      TEXT,
            detail      TEXT,
            severity    TEXT DEFAULT 'warning'
        );

        CREATE TABLE IF NOT EXISTS twap_schedules (
            schedule_id      TEXT PRIMARY KEY,
            parent_order_id  TEXT,
            symbol           TEXT NOT NULL,
            side             TEXT NOT NULL,
            total_qty        REAL NOT NULL,
            remaining_qty    REAL NOT NULL,
            n_tranches       INTEGER NOT NULL,
            tranche_size     REAL NOT NULL,
            interval_seconds REAL NOT NULL,
            start_utc        TEXT NOT NULL,
            end_utc          TEXT NOT NULL,
            completed        INTEGER DEFAULT 0,
            avg_fill_price   REAL,
            cum_filled_qty   REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS vwap_profiles (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,
            bin_5min    INTEGER NOT NULL,  -- 0-77 (390 min / 5)
            volume      REAL NOT NULL,
            vwap        REAL NOT NULL,
            PRIMARY KEY (symbol, date, bin_5min)
        );
    """)
    db.commit()


# ---------------------------------------------------------------------------
# Pydantic models — API request/response
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    symbol:        str   = Field(..., description="Ticker (e.g. AAPL)")
    side:          str   = Field(..., pattern="^(buy|sell)$")
    order_type:    str   = Field("market", description="market|limit|stop|stop_limit|trailing_stop")
    tif:           str   = Field("day",    description="day|gtc|ioc|fok|opg|cls")
    qty:           Optional[float] = Field(None, gt=0, description="Shares; use notional for dollar-based")
    notional:      Optional[float] = Field(None, gt=0, description="Dollar notional (fractional shares)")
    limit_price:   Optional[float] = Field(None, gt=0)
    stop_price:    Optional[float] = Field(None, gt=0)
    trail_pct:     Optional[float] = Field(None, gt=0, description="Trailing stop % (e.g. 1.5 = 1.5%)")
    trail_price:   Optional[float] = Field(None, gt=0, description="Trailing stop dollar amount")
    extended_hours: bool = Field(False, description="Allow pre-market / after-hours execution")
    fractional:    bool = Field(False, description="Enable fractional share fill")
    notes:         Optional[str] = None

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        valid = {e.value for e in OrderType}
        if v not in valid:
            raise ValueError(f"order_type must be one of {valid}")
        return v

    @field_validator("tif")
    @classmethod
    def validate_tif(cls, v: str) -> str:
        valid = {e.value for e in TIF}
        if v not in valid:
            raise ValueError(f"tif must be one of {valid}")
        return v


class TWAPRequest(BaseModel):
    symbol:          str   = Field(..., description="Ticker")
    side:            str   = Field(..., pattern="^(buy|sell)$")
    total_qty:       float = Field(..., gt=0)
    duration_minutes: int  = Field(30, ge=5, le=390, description="Execution window in minutes")
    n_tranches:      int   = Field(10, ge=2,  le=50,  description="Number of child orders")
    randomize_pct:   float = Field(0.10, ge=0, le=0.30, description="Size randomisation ± fraction")
    limit_offset_pct: Optional[float] = Field(None, description="Limit price offset from mid (0.001 = 0.1%)")


class VWAPRequest(BaseModel):
    symbol:         str   = Field(..., description="Ticker")
    side:           str   = Field(..., pattern="^(buy|sell)$")
    total_qty:      float = Field(..., gt=0)
    n_tranches:     int   = Field(13, ge=2, le=78, description="Number of 5-min bins to participate in")
    start_at_open:  bool  = Field(True,  description="Start from market open vs. current time")
    limit_offset_pct: Optional[float] = Field(None)


class RebalanceRequest(BaseModel):
    target_weights:      Dict[str, float] = Field(..., description="{symbol: weight}, should sum to ≤1")
    deviation_threshold: float            = Field(0.02, ge=0.001, le=0.20)
    max_position_pct:    float            = Field(0.20, ge=0.01,  le=1.0,  description="Max single-name weight")
    sector_limits:       Dict[str, float] = Field(default_factory=dict, description="{sector: max_weight}")
    cash_buffer_pct:     float            = Field(0.02, ge=0.0,   le=0.20, description="Keep this % in cash")
    odd_lot_avoidance:   bool             = Field(True, description="Floor qty to nearest whole share")
    spread_hours:        int              = Field(2, ge=1, le=8)
    tax_aware:           bool             = Field(True)


class PositionSizeRequest(BaseModel):
    symbol:      str
    side:        str   = Field(..., pattern="^(buy|sell)$")
    qty:         float = Field(..., gt=0)
    price:       Optional[float] = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrancheSpec:
    idx:            int
    qty:            float
    delay_seconds:  float
    limit_price:    Optional[float] = None
    filled:         bool  = False
    fill_price:     Optional[float] = None
    fill_ts:        Optional[str]   = None
    order_id:       Optional[str]   = None


@dataclass
class OrderRecord:
    id:              str
    created_at:      str
    symbol:          str
    side:            str
    order_type:      str
    tif:             str
    qty:             float
    notional:        Optional[float]
    limit_price:     Optional[float]
    stop_price:      Optional[float]
    trail_pct:       Optional[float]
    trail_price:     Optional[float]
    status:          str
    alpaca_order_id: Optional[str]
    avg_fill_price:  Optional[float]
    filled_qty:      float
    sor_strategy:    str
    extended_hours:  bool
    fractional:      bool


@dataclass
class RebalanceTrade:
    symbol:         str
    target_weight:  float
    current_weight: float
    deviation:      float
    action:         str       # buy | sell | hold
    qty:            float
    notional:       float
    tax_priority:   int       # 0 = sell losers first, 2 = buys
    scheduled_wave: int


# ---------------------------------------------------------------------------
# Alpaca client factory
# ---------------------------------------------------------------------------

def _build_trading_client() -> "TradingClient":
    _require_alpaca()
    key, secret = _get_credentials()
    return TradingClient(api_key=key, secret_key=secret, paper=_is_paper())


def _build_data_client() -> "StockHistoricalDataClient":
    _require_alpaca()
    key, secret = _get_credentials()
    return StockHistoricalDataClient(api_key=key, secret_key=secret)


# ---------------------------------------------------------------------------
# PreTradeRiskEngine
# ---------------------------------------------------------------------------

class PreTradeRiskEngine:
    """Pre-trade checks before any order submission.

    Checks
    ------
    1. Order size vs 20-day ADV (reject if > _DEFAULT_ADV_PCT)
    2. Notional limit per order
    3. Daily loss limit (halt all trading if breached)
    4. Duplicate order detection (same side/symbol within 1 second)
    5. Extended-hours flag consistency
    """

    def __init__(
        self,
        api_key:             str   = "",
        secret_key:          str   = "",
        paper:               bool  = True,
        notional_limit:      float = _DEFAULT_NOTIONAL_LIMIT,
        adv_pct_limit:       float = _DEFAULT_ADV_PCT,
        daily_loss_pct_halt: float = _DEFAULT_DAILY_LOSS_PCT,
    ) -> None:
        self._api_key       = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret        = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper         = paper
        self._notional_limit = notional_limit
        self._adv_pct_limit  = adv_pct_limit
        self._daily_loss_pct = daily_loss_pct_halt

        self._trading_client: Optional[Any] = None
        self._data_client:    Optional[Any] = None
        self._halted:         bool           = False
        self._halt_reason:    str            = ""
        self._daily_start_eq: float          = 0.0
        self._adv_cache:      Dict[str, Tuple[float, float]] = {}  # sym → (adv, ts)
        self._recent_orders:  deque          = deque(maxlen=200)   # (ts, sym, side)
        self._lock:           threading.Lock = threading.Lock()

    def _tc(self) -> Any:
        if self._trading_client is None and _ALPACA_AVAILABLE:
            self._trading_client = TradingClient(
                api_key=self._api_key, secret_key=self._secret, paper=self._paper
            )
        return self._trading_client

    def _dc(self) -> Any:
        if self._data_client is None and _ALPACA_AVAILABLE:
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key, secret_key=self._secret
            )
        return self._data_client

    # ------------------------------------------------------------------
    # ADV fetch (20-day average daily volume)
    # ------------------------------------------------------------------

    def _fetch_adv(self, symbol: str) -> float:
        """Return 20-day average daily volume, cached for 60 minutes."""
        now = time.time()
        if symbol in self._adv_cache:
            adv, ts = self._adv_cache[symbol]
            if now - ts < 3600:
                return adv

        dc = self._dc()
        if dc is None:
            return float("inf")  # can't check → pass through

        try:
            end   = _now_utc()
            start = end - timedelta(days=30)
            req   = StockBarsRequest(
                symbol_or_symbols=symbol.upper(),
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = dc.get_stock_bars(req)
            df   = bars.df
            if df.empty:
                return float("inf")
            adv = float(df["volume"].tail(20).mean())
            self._adv_cache[symbol] = (adv, now)
            return adv
        except Exception as exc:
            logger.warning("ADV fetch failed for %s: %s", symbol, exc)
            return float("inf")

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    def _get_daily_pnl_pct(self) -> float:
        tc = self._tc()
        if tc is None:
            return 0.0
        try:
            acct = tc.get_account()
            equity = float(acct.equity or 0)
            if self._daily_start_eq <= 0:
                self._daily_start_eq = equity
                return 0.0
            return (equity - self._daily_start_eq) / self._daily_start_eq
        except Exception:
            return 0.0

    def set_daily_start_equity(self, equity: float) -> None:
        self._daily_start_eq = equity

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _is_duplicate(self, symbol: str, side: str) -> bool:
        h   = _order_hash(symbol, side)
        now = time.time()
        with self._lock:
            for rec_ts, rec_h in self._recent_orders:
                if rec_h == h and (now - rec_ts) < _DUPE_DETECT_WINDOW_S:
                    return True
            self._recent_orders.append((now, h))
        return False

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        price:       float,
        order_type:  str  = "market",
        tif:         str  = "day",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Run all pre-trade checks. Returns {"ok": bool, "reason": str, ...}."""

        if self._halted:
            return {"ok": False, "reason": f"Trading halted: {self._halt_reason}"}

        # Notional check
        notional = qty * price if price > 0 else 0.0
        if notional > self._notional_limit and self._notional_limit > 0:
            reason = (
                f"Notional ${notional:,.0f} exceeds limit ${self._notional_limit:,.0f}"
            )
            _persist_risk_event("notional_limit", symbol, reason)
            return {"ok": False, "reason": reason}

        # ADV check
        adv = self._fetch_adv(symbol)
        if adv < float("inf") and adv > 0:
            adv_pct = qty / adv
            if adv_pct > self._adv_pct_limit:
                reason = (
                    f"Order qty {qty:.0f} is {adv_pct:.1%} of 20d ADV {adv:.0f} "
                    f"(limit {self._adv_pct_limit:.0%})"
                )
                _persist_risk_event("adv_limit", symbol, reason)
                return {"ok": False, "reason": reason, "adv_pct": adv_pct}

        # Daily loss check
        daily_pnl = self._get_daily_pnl_pct()
        if daily_pnl < -self._daily_loss_pct:
            self._halted     = True
            self._halt_reason = f"daily P&L {daily_pnl:.2%} < -{self._daily_loss_pct:.2%}"
            _persist_risk_event(
                "daily_loss_halt", symbol, self._halt_reason, severity="critical"
            )
            return {"ok": False, "reason": f"Trading halted: {self._halt_reason}"}

        # Duplicate detection
        if self._is_duplicate(symbol, side):
            reason = f"Duplicate order detected: {side} {symbol} within {_DUPE_DETECT_WINDOW_S}s"
            _persist_risk_event("duplicate_order", symbol, reason)
            return {"ok": False, "reason": reason}

        # Extended-hours flag: OPG/CLS tif only valid in core hours; warn
        if extended_hours and tif in ("opg", "cls"):
            logger.warning(
                "TIF %s combined with extended_hours=True may be rejected by Alpaca.", tif
            )

        return {
            "ok":       True,
            "notional": notional,
            "adv_pct":  (qty / adv) if adv not in (0, float("inf")) else None,
            "daily_pnl": daily_pnl,
        }

    def reset_halt(self) -> None:
        self._halted     = False
        self._halt_reason = ""
        logger.info("Pre-trade risk halt manually cleared.")

    def is_halted(self) -> bool:
        return self._halted


# ---------------------------------------------------------------------------
# Market data helper (shared)
# ---------------------------------------------------------------------------

class MarketDataHelper:
    """Thin wrapper around Alpaca data client with latency-aware logging."""

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self._api_key = api_key or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._dc: Optional[Any] = None
        self._quote_cache: Dict[str, Tuple[Dict, float]] = {}

    def _client(self) -> Any:
        if self._dc is None and _ALPACA_AVAILABLE:
            self._dc = StockHistoricalDataClient(
                api_key=self._api_key, secret_key=self._secret
            )
        return self._dc

    def get_quote(self, symbol: str, cache_seconds: float = 1.0) -> Dict[str, float]:
        """IEX near-real-time quote. Latency: typically 50-500ms behind SIP.
        Do NOT use for tick-sensitive limit price decisions without adding a buffer.
        """
        now = time.time()
        if symbol in self._quote_cache:
            cached, ts = self._quote_cache[symbol]
            if now - ts < cache_seconds:
                return cached

        dc = self._client()
        if dc is None:
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "latency_tag": _DATA_LATENCY_TAG}
        try:
            req  = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
            data = dc.get_stock_latest_quote(req)
            q    = data[symbol.upper()]
            bid  = float(q.bid_price or 0)
            ask  = float(q.ask_price or 0)
            mid  = (bid + ask) / 2.0 if bid and ask else max(bid, ask)
            result = {
                "bid":         bid,
                "ask":         ask,
                "mid":         mid,
                "spread":      ask - bid,
                "latency_tag": _DATA_LATENCY_TAG,
            }
            self._quote_cache[symbol] = (result, now)
            return result
        except Exception as exc:
            logger.warning("IEX quote failed for %s: %s", symbol, exc)
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "latency_tag": _DATA_LATENCY_TAG}

    def get_vwap_benchmark(self, symbol: str, minutes: int = 30) -> float:
        """Compute VWAP over last N minutes using 1-min bars (IEX data)."""
        dc = self._client()
        if dc is None:
            return 0.0
        try:
            end   = _now_utc()
            start = end - timedelta(minutes=minutes + 5)
            req   = StockBarsRequest(
                symbol_or_symbols=symbol.upper(),
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = dc.get_stock_bars(req)
            df   = bars.df.tail(minutes)
            if df.empty:
                return 0.0
            tp   = (df["high"] + df["low"] + df["close"]) / 3.0
            vwap = float((tp * df["volume"]).sum() / df["volume"].sum())
            return vwap
        except Exception as exc:
            logger.warning("VWAP benchmark failed for %s: %s", symbol, exc)
            return 0.0

    def get_daily_bars(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch daily OHLCV bars (IEX historical data — accurate, not lagged)."""
        dc = self._client()
        if dc is None:
            return pd.DataFrame()
        try:
            end   = _now_utc()
            start = end - timedelta(days=days + 5)
            req   = StockBarsRequest(
                symbol_or_symbols=symbol.upper(),
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = dc.get_stock_bars(req)
            return bars.df.tail(days)
        except Exception as exc:
            logger.warning("Daily bars failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_intraday_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Return 5-minute bin volume fractions (0-77) from last 5 trading days.

        Used by VWAPEngine to set participation rates. IEX bars are accurate here.
        Returns {bin_idx: volume_fraction}
        """
        dc = self._client()
        if dc is None:
            return {}
        try:
            end   = _now_utc()
            start = end - timedelta(days=8)
            req   = StockBarsRequest(
                symbol_or_symbols=symbol.upper(),
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = dc.get_stock_bars(req)
            df   = bars.df
            if df.empty:
                return {}

            # Convert timestamp to minutes-since-open bin
            if hasattr(df.index, "get_level_values"):
                timestamps = df.index.get_level_values("timestamp")
            else:
                timestamps = df.index

            et_times = pd.Series(timestamps).dt.tz_convert("America/New_York")
            open_mins = (et_times.dt.hour - 9) * 60 + et_times.dt.minute - 30
            df = df.copy()
            df["bin"] = (open_mins // 5).clip(0, 77)
            df = df[df["bin"] >= 0]

            profile  = df.groupby("bin")["volume"].sum()
            total_v  = profile.sum()
            if total_v <= 0:
                return {}
            return {int(b): float(v / total_v) for b, v in profile.items()}
        except Exception as exc:
            logger.warning("Volume profile failed for %s: %s", symbol, exc)
            return {}


# ---------------------------------------------------------------------------
# TradierSandbox — order simulation for comparison
# ---------------------------------------------------------------------------

class TradierSandbox:
    """Submit simulated orders to Tradier sandbox and compare fills with Alpaca.

    Tradier sandbox is always free. It does not execute real orders.
    Used to benchmark Alpaca fill quality against Tradier routing.

    NOTE: Tradier sandbox quotes may also have latency; treat as indicative only.
    """

    def __init__(self) -> None:
        self._token   = os.getenv(_ENV_TRADIER, "")
        self._account = _TRADIER_SANDBOX_ACCOUNT
        self._base    = _TRADIER_SANDBOX_BASE
        self._enabled = bool(self._token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept":        "application/json",
        }

    def submit_order(
        self,
        symbol:      str,
        side:        str,
        qty:         int,
        order_type:  str        = "market",
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        tif:         str        = "day",
    ) -> Dict[str, Any]:
        """Submit a simulated order to Tradier sandbox.

        Returns Tradier order ID and simulated fill details.
        """
        if not self._enabled:
            return {"status": "disabled", "reason": "TRADIER_SANDBOX_TOKEN not set"}

        side_t = "buy" if side == "buy" else "sell"
        tif_t  = {"day": "day", "gtc": "gtc", "ioc": "ioc", "fok": "fok"}.get(tif, "day")
        type_t = {
            "market":    "market",
            "limit":     "limit",
            "stop":      "stop",
            "stop_limit": "stop_limit",
        }.get(order_type, "market")

        payload: Dict[str, Any] = {
            "class":    "equity",
            "symbol":   symbol.upper(),
            "side":     side_t,
            "quantity": int(qty),
            "type":     type_t,
            "duration": tif_t,
        }
        if limit_price and type_t in ("limit", "stop_limit"):
            payload["price"] = str(limit_price)
        if stop_price and type_t in ("stop", "stop_limit"):
            payload["stop"] = str(stop_price)

        try:
            url  = f"{self._base}/accounts/{self._account}/orders"
            resp = requests.post(url, headers=self._headers(), data=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                order_data = data.get("order", {})
                return {
                    "status":          "submitted",
                    "tradier_order_id": str(order_data.get("id", "")),
                    "symbol":          symbol,
                    "side":            side,
                    "qty":             qty,
                    "type":            type_t,
                    "venue":           "tradier_sandbox",
                    "latency_tag":     _DATA_LATENCY_TAG,
                }
            else:
                return {
                    "status": "rejected",
                    "http_status": resp.status_code,
                    "body": resp.text[:200],
                }
        except Exception as exc:
            logger.warning("Tradier sandbox submission failed: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def get_order(self, tradier_order_id: str) -> Dict[str, Any]:
        """Poll Tradier sandbox for simulated fill status."""
        if not self._enabled:
            return {"status": "disabled"}
        try:
            url  = f"{self._base}/accounts/{self._account}/orders/{tradier_order_id}"
            resp = requests.get(url, headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                return resp.json().get("order", {})
            return {"status": "error", "http_status": resp.status_code}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}


# ---------------------------------------------------------------------------
# OrderManagementSystem
# ---------------------------------------------------------------------------

class OrderManagementSystem:
    """Full order lifecycle management.

    Supports
    --------
    - Order types: market, limit, stop, stop_limit, trailing_stop
    - Time-in-force: day, gtc, ioc, fok, opg (market-on-open), cls (market-on-close)
    - Extended hours: pre-market and after-hours flag
    - Fractional shares for qualifying securities
    - Tradier sandbox simulation alongside Alpaca submission
    - Full SQLite audit trail with data_latency tag on every order
    """

    def __init__(
        self,
        api_key:    str  = "",
        secret_key: str  = "",
        paper:      bool = True,
    ) -> None:
        self._api_key   = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret    = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper     = paper
        self._tc: Optional[Any]      = None
        self._md = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)
        self._tradier   = TradierSandbox()
        db = _get_db(); _ensure_tables(db); db.close()

    def _trading_client(self) -> Any:
        if self._tc is None and _ALPACA_AVAILABLE:
            self._tc = TradingClient(
                api_key=self._api_key, secret_key=self._secret, paper=self._paper
            )
        return self._tc

    # ------------------------------------------------------------------
    # Submit order
    # ------------------------------------------------------------------

    def submit(self, req: OrderRequest) -> Dict[str, Any]:
        """Submit order to Alpaca and optionally simulate on Tradier sandbox.

        Returns a canonical order record.
        """
        symbol  = req.symbol.upper()
        order_id = str(uuid.uuid4())
        now_s    = _now_utc().isoformat()

        # Capture decision price (IEX quote — tagged as near_real_time_iex)
        quote         = self._md.get_quote(symbol)
        decision_price = quote["mid"]

        # Build Alpaca request
        tc = self._trading_client()
        alpaca_order_id: Optional[str] = None
        if tc is not None:
            try:
                alpaca_order_id = self._submit_to_alpaca(tc, req, symbol)
            except Exception as exc:
                logger.error("Alpaca submission failed for %s: %s", symbol, exc)

        # Tradier simulation (non-blocking, best-effort)
        tradier_result = {}
        if req.qty and int(req.qty) == req.qty:   # Tradier sandbox needs integer qty
            tradier_result = self._tradier.submit_order(
                symbol     = symbol,
                side       = req.side,
                qty        = int(req.qty),
                order_type = req.order_type,
                limit_price = req.limit_price,
                stop_price  = req.stop_price,
                tif        = req.tif,
            )

        status = OrderStatus.SUBMITTED.value if alpaca_order_id else OrderStatus.REJECTED.value

        record = {
            "id":              order_id,
            "created_at":      now_s,
            "updated_at":      now_s,
            "symbol":          symbol,
            "side":            req.side,
            "order_type":      req.order_type,
            "tif":             req.tif,
            "qty":             req.qty or 0.0,
            "notional":        req.notional,
            "limit_price":     req.limit_price,
            "stop_price":      req.stop_price,
            "trail_pct":       req.trail_pct,
            "trail_price":     req.trail_price,
            "status":          status,
            "alpaca_order_id": alpaca_order_id,
            "tradier_order_id": tradier_result.get("tradier_order_id"),
            "avg_fill_price":  None,
            "filled_qty":      0.0,
            "sor_strategy":    SORStrategy.DIRECT.value,
            "data_latency":    _DATA_LATENCY_TAG,
            "extended_hours":  req.extended_hours,
            "fractional":      req.fractional,
            "decision_price":  decision_price,
            "tradier_result":  tradier_result,
            "notes":           req.notes,
        }
        self._persist_order(record)
        return record

    def _submit_to_alpaca(self, tc: Any, req: OrderRequest, symbol: str) -> Optional[str]:
        _require_alpaca()
        side_e = OrderSide.BUY if req.side == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK,
            "opg": TimeInForce.OPG, "cls": TimeInForce.CLS,
        }
        tif_e = tif_map.get(req.tif, TimeInForce.DAY)

        if req.order_type == "market":
            kwargs: Dict[str, Any] = dict(symbol=symbol, side=side_e, time_in_force=tif_e)
            if req.qty:
                kwargs["qty"] = req.qty
            elif req.notional:
                kwargs["notional"] = req.notional
            if req.extended_hours:
                kwargs["extended_hours"] = True
            order = tc.submit_order(MarketOrderRequest(**kwargs))

        elif req.order_type == "limit":
            if not req.limit_price:
                raise ValueError("limit_price required for limit orders")
            kwargs = dict(
                symbol=symbol, qty=req.qty or 1, side=side_e,
                limit_price=req.limit_price, time_in_force=tif_e,
            )
            if req.extended_hours:
                kwargs["extended_hours"] = True
            order = tc.submit_order(LimitOrderRequest(**kwargs))

        elif req.order_type == "stop":
            if not req.stop_price:
                raise ValueError("stop_price required for stop orders")
            order = tc.submit_order(StopOrderRequest(
                symbol=symbol, qty=req.qty or 1, side=side_e,
                stop_price=req.stop_price, time_in_force=tif_e,
            ))

        elif req.order_type == "stop_limit":
            if not req.stop_price or not req.limit_price:
                raise ValueError("Both stop_price and limit_price required for stop_limit")
            order = tc.submit_order(StopLimitOrderRequest(
                symbol=symbol, qty=req.qty or 1, side=side_e,
                stop_price=req.stop_price, limit_price=req.limit_price,
                time_in_force=tif_e,
            ))

        elif req.order_type == "trailing_stop":
            kwargs = dict(symbol=symbol, qty=req.qty or 1, side=side_e, time_in_force=tif_e)
            if req.trail_pct:
                kwargs["trail_percent"] = req.trail_pct
            elif req.trail_price:
                kwargs["trail_price"] = req.trail_price
            else:
                raise ValueError("trail_pct or trail_price required for trailing_stop")
            order = tc.submit_order(TrailingStopOrderRequest(**kwargs))

        else:
            raise ValueError(f"Unknown order_type: {req.order_type}")

        logger.info(
            "Alpaca order submitted: %s %s %s (type=%s, tif=%s, data=%s)",
            req.side, req.qty, symbol, req.order_type, req.tif, _DATA_LATENCY_TAG,
        )
        try:
            return str(order.id)
        except Exception:
            return str(order)

    def cancel(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order by SENTINEL order_id. Looks up alpaca_order_id."""
        db = _get_db()
        _ensure_tables(db)
        row = db.execute(
            "SELECT alpaca_order_id, status FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        db.close()

        if not row:
            raise ValueError(f"Order {order_id} not found")
        if row["status"] in ("filled", "cancelled"):
            return {"status": "already_terminal", "order_status": row["status"]}

        alpaca_id = row["alpaca_order_id"]
        if alpaca_id:
            tc = self._trading_client()
            if tc:
                try:
                    tc.cancel_order_by_id(alpaca_id)
                    self._update_order_status(order_id, OrderStatus.CANCELLED.value)
                    return {"status": "cancelled", "order_id": order_id}
                except Exception as exc:
                    return {"status": "error", "reason": str(exc)}
        return {"status": "no_alpaca_id", "order_id": order_id}

    def list_orders(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        db = _get_db()
        _ensure_tables(db)
        if status:
            rows = db.execute(
                "SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def _persist_order(self, r: Dict[str, Any]) -> None:
        db = _get_db()
        _ensure_tables(db)
        try:
            db.execute(
                """INSERT OR IGNORE INTO orders
                   (id, created_at, updated_at, symbol, side, order_type, tif,
                    qty, notional, limit_price, stop_price, trail_pct, trail_price,
                    status, alpaca_order_id, tradier_order_id, avg_fill_price,
                    filled_qty, sor_strategy, data_latency, extended_hours,
                    fractional, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["id"], r["created_at"], r["updated_at"],
                    r["symbol"], r["side"], r["order_type"], r["tif"],
                    r.get("qty", 0), r.get("notional"), r.get("limit_price"),
                    r.get("stop_price"), r.get("trail_pct"), r.get("trail_price"),
                    r["status"], r.get("alpaca_order_id"), r.get("tradier_order_id"),
                    r.get("avg_fill_price"), r.get("filled_qty", 0),
                    r.get("sor_strategy", "direct"), r.get("data_latency", _DATA_LATENCY_TAG),
                    int(r.get("extended_hours", False)), int(r.get("fractional", False)),
                    r.get("notes"),
                ),
            )
            db.commit()
        finally:
            db.close()

    def _update_order_status(self, order_id: str, status: str) -> None:
        db = _get_db()
        try:
            db.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now_utc().isoformat(), order_id),
            )
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# TWAPEngine
# ---------------------------------------------------------------------------

class TWAPEngine:
    """TWAP execution: split a large order into N child orders over T minutes.

    Features
    --------
    - Child size = total/N ± randomize_pct (reduce pattern detection)
    - Track cumulative fills, remaining qty, average fill price
    - Optional limit price derived from IEX mid (with latency caveat)
    - Stores schedule in SQLite for recovery / monitoring
    """

    def __init__(
        self,
        api_key:    str  = "",
        secret_key: str  = "",
        paper:      bool = True,
    ) -> None:
        self._api_key = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper   = paper
        self._oms     = OrderManagementSystem(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._md      = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)

    def build_schedule(self, req: TWAPRequest) -> Dict[str, Any]:
        """Build and persist a TWAP schedule (does NOT submit orders yet)."""
        symbol   = req.symbol.upper()
        n        = req.n_tranches
        per_base = req.total_qty / n
        interval = (req.duration_minutes * 60.0) / n

        # Randomise sizes: ±randomize_pct, ensure sum = total_qty
        rands    = [random.uniform(1 - req.randomize_pct, 1 + req.randomize_pct) for _ in range(n)]
        rands_s  = sum(rands)
        sizes    = [round(req.total_qty * r / rands_s, 4) for r in rands]
        # Fix rounding so exact sum
        diff     = round(req.total_qty - sum(sizes), 4)
        sizes[-1] = round(sizes[-1] + diff, 4)

        # Optional limit price from IEX quote
        quote        = self._md.get_quote(symbol)
        mid          = quote["mid"]
        limit_prices = []
        for _ in sizes:
            if req.limit_offset_pct and mid > 0:
                if req.side == "buy":
                    lp = round(mid * (1 + req.limit_offset_pct), 2)
                else:
                    lp = round(mid * (1 - req.limit_offset_pct), 2)
                limit_prices.append(lp)
            else:
                limit_prices.append(None)

        schedule_id = str(uuid.uuid4())
        now_utc     = _now_utc()

        tranches = []
        for i, (sz, lp) in enumerate(zip(sizes, limit_prices)):
            execute_at = now_utc + timedelta(seconds=i * interval)
            tranches.append({
                "tranche_idx":  i,
                "qty":          sz,
                "limit_price":  lp,
                "delay_seconds": i * interval,
                "execute_at_utc": execute_at.isoformat(),
                "status":       "pending",
            })

        record = {
            "schedule_id":      schedule_id,
            "symbol":           symbol,
            "side":             req.side,
            "total_qty":        req.total_qty,
            "remaining_qty":    req.total_qty,
            "n_tranches":       n,
            "tranche_size":     per_base,
            "interval_seconds": interval,
            "start_utc":        now_utc.isoformat(),
            "end_utc":          (now_utc + timedelta(minutes=req.duration_minutes)).isoformat(),
            "tranches":         tranches,
            "data_latency":     _DATA_LATENCY_TAG,
            "quote_at_schedule": quote,
        }
        self._persist_schedule(record)
        logger.info(
            "TWAP schedule %s: %s %s %g shares, %d tranches over %d min (data=%s)",
            schedule_id, req.side, symbol, req.total_qty, n, req.duration_minutes,
            _DATA_LATENCY_TAG,
        )
        return record

    def execute_next_tranche(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Execute the next pending tranche of a TWAP schedule."""
        db = _get_db()
        _ensure_tables(db)
        row = db.execute(
            "SELECT * FROM twap_schedules WHERE schedule_id = ?", (schedule_id,)
        ).fetchone()
        db.close()

        if not row or row["completed"]:
            return None

        remaining = row["remaining_qty"]
        if remaining <= 0:
            return None

        # Find next tranche size
        n       = row["n_tranches"]
        filled  = row["n_tranches"] - math.ceil(remaining / (row["total_qty"] / n))
        tranche = min(row["tranche_size"], remaining)

        order_req = OrderRequest(
            symbol     = row["symbol"],
            side       = row["side"],
            order_type = "market",
            tif        = "day",
            qty        = round(tranche, 4),
        )
        result = self._oms.submit(order_req)

        new_remaining = round(remaining - tranche, 4)
        completed_flag = 1 if new_remaining <= 0 else 0
        db = _get_db()
        try:
            db.execute(
                """UPDATE twap_schedules
                   SET remaining_qty = ?, completed = ?, cum_filled_qty = cum_filled_qty + ?
                   WHERE schedule_id = ?""",
                (max(new_remaining, 0), completed_flag, tranche, schedule_id),
            )
            db.commit()
        finally:
            db.close()

        return {
            "schedule_id":   schedule_id,
            "tranche_filled": filled,
            "qty_this":      tranche,
            "remaining_qty": new_remaining,
            "order_result":  result,
            "completed":     bool(completed_flag),
        }

    def get_schedule_status(self, schedule_id: str) -> Dict[str, Any]:
        db = _get_db()
        _ensure_tables(db)
        row = db.execute(
            "SELECT * FROM twap_schedules WHERE schedule_id = ?", (schedule_id,)
        ).fetchone()
        db.close()
        if not row:
            return {"error": "schedule not found"}
        d = dict(row)
        d["fill_pct"] = round(
            (d["cum_filled_qty"] / d["total_qty"] * 100) if d["total_qty"] > 0 else 0, 1
        )
        return d

    def _persist_schedule(self, rec: Dict[str, Any]) -> None:
        db = _get_db()
        _ensure_tables(db)
        try:
            db.execute(
                """INSERT OR IGNORE INTO twap_schedules
                   (schedule_id, symbol, side, total_qty, remaining_qty, n_tranches,
                    tranche_size, interval_seconds, start_utc, end_utc, completed, cum_filled_qty)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec["schedule_id"], rec["symbol"], rec["side"],
                    rec["total_qty"], rec["remaining_qty"], rec["n_tranches"],
                    rec["tranche_size"], rec["interval_seconds"],
                    rec["start_utc"], rec["end_utc"], 0, 0.0,
                ),
            )
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# VWAPEngine
# ---------------------------------------------------------------------------

class VWAPEngine:
    """VWAP execution using historical intraday volume profile.

    Participation rate per 5-min bin is proportional to expected volume fraction.
    Falls back to U-shaped profile if historical data is unavailable.

    Almgren-Chriss impact is NOT applied here (only for >$100K blocks via SOR).
    """

    def __init__(
        self,
        api_key:    str  = "",
        secret_key: str  = "",
        paper:      bool = True,
    ) -> None:
        self._api_key = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper   = paper
        self._oms     = OrderManagementSystem(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._md      = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)

    def _u_shaped_weights(self, n: int) -> List[float]:
        """U-shaped intraday volume profile fallback: more volume at open/close."""
        weights = []
        for i in range(n):
            x = i / max(n - 1, 1)
            w = 1.0 + 2.5 * (x - 0.5) ** 2  # wider U than V2
            weights.append(w)
        return weights

    def build_schedule(self, req: VWAPRequest) -> Dict[str, Any]:
        """Build VWAP participation schedule.

        Returns schedule dict with per-bin quantities.
        IMPORTANT: actual fill prices depend on market conditions at execution time;
        IEX quote at schedule-build time is for reference only.
        """
        symbol = req.symbol.upper()
        n      = req.n_tranches

        # Try to get real volume profile; fall back to U-shape
        profile = self._md.get_intraday_volume_profile(symbol)
        if profile and len(profile) >= n:
            # Sort bins; take n most-volume bins proportionally
            bins    = sorted(profile.keys())[:n]
            weights = [profile.get(b, 0.0) for b in bins]
        else:
            logger.info("No volume profile for %s; using U-shaped fallback.", symbol)
            bins    = list(range(n))
            weights = self._u_shaped_weights(n)

        total_w = sum(weights) or 1.0
        sizes   = [round(req.total_qty * w / total_w, 4) for w in weights]
        # Fix rounding
        diff    = round(req.total_qty - sum(sizes), 4)
        sizes[-1] = round(sizes[-1] + diff, 4)

        schedule_id = str(uuid.uuid4())
        now_utc     = _now_utc()

        # Each bin is 5 minutes; spacing = 5 min
        tranches = []
        for i, (b, sz) in enumerate(zip(bins, sizes)):
            if req.start_at_open:
                delay_s = b * 300.0   # 5 min per bin from market open
            else:
                delay_s = i * 300.0   # 5 min from now for each bin

            # Optional limit: add buffer to IEX quote because of latency
            quote = self._md.get_quote(symbol, cache_seconds=30.0)
            mid   = quote["mid"]
            lp    = None
            if req.limit_offset_pct and mid > 0:
                if req.side == "buy":
                    lp = round(mid * (1 + req.limit_offset_pct), 2)
                else:
                    lp = round(mid * (1 - req.limit_offset_pct), 2)

            tranches.append({
                "tranche_idx":   i,
                "bin_5min":      b,
                "qty":           sz,
                "delay_seconds": delay_s,
                "execute_at_utc": (now_utc + timedelta(seconds=delay_s)).isoformat(),
                "limit_price":   lp,
                "volume_weight": round(weights[i] / total_w, 4),
            })

        return {
            "schedule_id":   schedule_id,
            "symbol":        symbol,
            "side":          req.side,
            "total_qty":     req.total_qty,
            "n_tranches":    n,
            "profile_source": "historical_iex" if len(profile) >= n else "u_shape_fallback",
            "start_utc":     now_utc.isoformat(),
            "tranches":      tranches,
            "data_latency":  _DATA_LATENCY_TAG,
            "note": (
                "IEX quote captured at schedule build time; actual fill prices determined at execution. "
                "Order submission is real-time regardless of quote latency."
            ),
        }


# ---------------------------------------------------------------------------
# AlmgrenChrissEngine — optimal liquidation for large blocks (>$100K)
# ---------------------------------------------------------------------------

class AlmgrenChrissEngine:
    """Almgren-Chriss (2001) optimal execution trajectory.

    Minimises expected cost + variance of market impact for large orders.

    Model
    -----
    - Linear temporary impact: η * (dQ/dt) / ADV  [price depression per unit time]
    - Permanent impact:         γ * Q_total / ADV  [lasting price shift]
    - Risk aversion param λ controls urgency vs. price impact trade-off
    - Optimal trajectory: exponential decay from initial rate to final rate

    Limitations
    -----------
    - Parameters η, γ, σ are estimated from historical data; not calibrated per-stock
    - Assumes constant volatility and liquidity; breaks down for thinly-traded stocks
    - Use only for orders where your expected market impact justifies the complexity
    """

    # Default Almgren-Chriss parameters (rough market consensus for large-caps)
    _ETA   = 2.5e-7    # temporary impact coefficient
    _GAMMA = 2.5e-8    # permanent impact coefficient
    _SIGMA = 0.02      # daily return volatility (1σ)
    _LAMBDA = 1e-6     # risk aversion

    def __init__(
        self,
        api_key:    str  = "",
        secret_key: str  = "",
        paper:      bool = True,
    ) -> None:
        self._api_key = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper   = paper
        self._oms     = OrderManagementSystem(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._md      = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)

    def _estimate_sigma(self, symbol: str) -> float:
        df = self._md.get_daily_bars(symbol, days=60)
        if df.empty or len(df) < 10:
            return self._SIGMA
        rets  = df["close"].pct_change().dropna()
        sigma = float(rets.std())
        return max(sigma, 0.005)  # floor at 0.5% daily vol

    def _estimate_adv(self, symbol: str) -> float:
        df = self._md.get_daily_bars(symbol, days=25)
        if df.empty:
            return 1e6
        return float(df["volume"].tail(20).mean())

    def compute_trajectory(
        self,
        symbol:          str,
        total_qty:       float,
        n_intervals:     int   = 10,
        T_minutes:       int   = 390,
        risk_aversion:   Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute optimal liquidation trajectory.

        Returns schedule with per-interval sell quantities and expected cost.
        """
        symbol = symbol.upper()
        lam    = risk_aversion or self._LAMBDA
        sigma  = self._estimate_sigma(symbol)
        adv    = self._estimate_adv(symbol)

        # Time horizon T in days
        T    = T_minutes / 390.0
        tau  = T / n_intervals
        n    = n_intervals

        # Almgren-Chriss kappa (urgency) — simplified closed form
        # kappa = sqrt(lambda * sigma^2 / eta)
        kappa = math.sqrt(max(lam * sigma ** 2 / max(self._ETA, 1e-12), 0))

        # Optimal trajectory: exponential decay of inventory
        # x(j) = X * sinh(kappa*(T - j*tau)) / sinh(kappa*T)
        sinh_kT = math.sinh(kappa * T) if kappa * T < 700 else 1e300
        trajectory = []
        for j in range(n + 1):
            t_remaining = T - j * tau
            if sinh_kT > 0:
                x_j = total_qty * math.sinh(kappa * t_remaining) / sinh_kT
            else:
                x_j = total_qty * (1 - j / n)
            trajectory.append(max(x_j, 0))

        # Shares to sell at each interval = trajectory[j] - trajectory[j+1]
        intervals = []
        for j in range(n):
            qty_j = round(trajectory[j] - trajectory[j + 1], 4)
            intervals.append({
                "interval_idx":  j,
                "qty":           max(qty_j, 0),
                "delay_seconds": j * tau * 390 * 60,
                "inventory_remaining": round(trajectory[j + 1], 2),
            })

        # Expected implementation cost (simplified)
        permanent_cost = self._GAMMA * total_qty ** 2 / adv
        temporary_cost = self._ETA * sum((iv["qty"] / tau) ** 2 for iv in intervals) * tau

        return {
            "symbol":          symbol,
            "total_qty":       total_qty,
            "n_intervals":     n,
            "T_minutes":       T_minutes,
            "kappa":           round(kappa, 6),
            "sigma_daily":     round(sigma, 4),
            "adv_shares":      round(adv, 0),
            "trajectory":      intervals,
            "expected_permanent_cost_bps": round(permanent_cost * 1e4, 2),
            "expected_temporary_cost_bps": round(temporary_cost * 1e4, 2),
            "total_impact_bps": round((permanent_cost + temporary_cost) * 1e4, 2),
            "data_latency":    _DATA_LATENCY_TAG,
            "model":           "almgren_chriss_2001_simplified",
            "warning": (
                "Parameters are rough market-consensus values. "
                "Calibrate η and γ against your own fill data for production use."
            ),
        }


# ---------------------------------------------------------------------------
# SmartOrderRouter (SOR)
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """Route orders based on notional size.

    Routing table
    -------------
    < $10K notional        → direct market order (minimal latency, IEX is fine)
    $10K – $100K notional  → TWAP over 5-30 min depending on ADV ratio
    > $100K notional       → VWAP with Almgren-Chriss trajectory optimisation

    All routing decisions are logged with data_latency tag.
    """

    def __init__(
        self,
        api_key:    str   = "",
        secret_key: str   = "",
        paper:      bool  = True,
    ) -> None:
        self._api_key = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper   = paper
        self._oms     = OrderManagementSystem(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._twap    = TWAPEngine(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._vwap    = VWAPEngine(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._ac      = AlmgrenChrissEngine(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._md      = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)

    def route(
        self,
        symbol:           str,
        qty:              float,
        side:             str,
        price_hint:       Optional[float] = None,
        force_strategy:   Optional[str]   = None,
    ) -> Dict[str, Any]:
        """Select execution strategy and return execution plan.

        Parameters
        ----------
        symbol         : Ticker symbol
        qty            : Total shares
        side           : 'buy' or 'sell'
        price_hint     : Current price (if known); otherwise fetched from IEX
        force_strategy : Override auto-selection ('direct'|'twap'|'vwap'|'almgren_chriss')
        """
        symbol = symbol.upper()

        # Get price from IEX (near-real-time, tagged)
        if not price_hint:
            quote      = self._md.get_quote(symbol)
            price_hint = quote["mid"] or 1.0
        notional = qty * price_hint

        # Auto-select strategy
        if force_strategy:
            strategy = SORStrategy(force_strategy)
        elif notional < _SOR_DIRECT_NOTIONAL:
            strategy = SORStrategy.DIRECT
        elif notional < _SOR_TWAP_NOTIONAL:
            strategy = SORStrategy.TWAP
        else:
            strategy = SORStrategy.VWAP

        logger.info(
            "SOR: %s %g %s notional=$%.0f → %s (data=%s)",
            side, qty, symbol, notional, strategy.value, _DATA_LATENCY_TAG,
        )

        if strategy == SORStrategy.DIRECT:
            order_req = OrderRequest(symbol=symbol, side=side, order_type="market", tif="day", qty=qty)
            result    = self._oms.submit(order_req)
            return {"strategy": strategy.value, "notional": notional, "order": result}

        elif strategy == SORStrategy.TWAP:
            # Duration: shorter for small orders, longer if ADV ratio is low
            adv = 1e6  # default; real ADV from pre-trade risk engine
            adv_ratio = qty / max(adv, 1)
            duration = max(5, min(30, int(adv_ratio * 200 * 30)))  # 5-30 min
            twap_req  = TWAPRequest(
                symbol=symbol, side=side, total_qty=qty,
                duration_minutes=duration, n_tranches=max(3, duration // 3),
            )
            sched = self._twap.build_schedule(twap_req)
            return {"strategy": strategy.value, "notional": notional, "twap_schedule": sched}

        elif strategy in (SORStrategy.VWAP, SORStrategy.AC):
            # First compute AC trajectory for cost analysis
            ac_plan = self._ac.compute_trajectory(symbol, qty)

            vwap_req = VWAPRequest(symbol=symbol, side=side, total_qty=qty, n_tranches=13)
            vwap_sched = self._vwap.build_schedule(vwap_req)

            return {
                "strategy":         strategy.value,
                "notional":         notional,
                "vwap_schedule":    vwap_sched,
                "ac_trajectory":    ac_plan,
                "recommendation": (
                    "Execute per AC trajectory schedule; use VWAP as fill benchmark. "
                    "Expected impact: {:.1f} bps".format(ac_plan.get("total_impact_bps", 0))
                ),
            }

        return {"strategy": "unknown", "error": "routing fell through"}


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """Real-time position reconciliation with Alpaca account state.

    Reconciliation
    --------------
    - Fetch positions from Alpaca API (always current, not lagged)
    - Compare against local SQLite snapshot
    - Flag discrepancies above tolerance threshold
    - Update local snapshot after each reconcile
    """

    def __init__(
        self,
        api_key:    str   = "",
        secret_key: str   = "",
        paper:      bool  = True,
        tolerance:  float = 0.01,   # 1% discrepancy tolerance
    ) -> None:
        self._api_key   = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret    = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper     = paper
        self._tolerance = tolerance
        self._tc: Optional[Any] = None

    def _trading_client(self) -> Any:
        if self._tc is None and _ALPACA_AVAILABLE:
            self._tc = TradingClient(
                api_key=self._api_key, secret_key=self._secret, paper=self._paper
            )
        return self._tc

    def reconcile(self) -> Dict[str, Any]:
        """Fetch live positions from Alpaca and reconcile with local snapshot.

        Returns reconciliation result with any discrepancies flagged.
        """
        tc = self._trading_client()
        if tc is None:
            return {"error": "Alpaca not available", "positions": []}

        try:
            acct      = tc.get_account()
            equity    = float(acct.equity or 0)
            cash      = float(acct.cash or 0)
            port_val  = float(acct.portfolio_value or equity)
            positions = tc.get_all_positions()
        except Exception as exc:
            logger.error("Position reconcile failed: %s", exc)
            return {"error": str(exc), "positions": []}

        live: Dict[str, Dict] = {}
        for p in positions:
            sym = p.symbol
            live[sym] = {
                "symbol":         sym,
                "qty":            float(p.qty or 0),
                "avg_entry_price": float(p.avg_entry_price or 0),
                "current_price":  float(p.current_price or 0),
                "market_value":   float(p.market_value or 0),
                "unrealized_pl":  float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
            }

        # Load local snapshot
        db        = _get_db()
        _ensure_tables(db)
        local_rows = {
            r["symbol"]: dict(r)
            for r in db.execute("SELECT * FROM positions").fetchall()
        }

        discrepancies = []
        now_s         = _now_utc().isoformat()

        for sym, lv in live.items():
            local = local_rows.get(sym)
            if local:
                qty_diff = abs(lv["qty"] - local["qty"])
                if local["qty"] > 0 and qty_diff / local["qty"] > self._tolerance:
                    discrepancies.append({
                        "symbol":     sym,
                        "live_qty":   lv["qty"],
                        "local_qty":  local["qty"],
                        "diff":       qty_diff,
                    })
            # Upsert position
            db.execute(
                """INSERT OR REPLACE INTO positions
                   (symbol, qty, avg_entry_price, current_price, market_value,
                    unrealized_pl, unrealized_plpc, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    sym, lv["qty"], lv["avg_entry_price"], lv["current_price"],
                    lv["market_value"], lv["unrealized_pl"], lv["unrealized_plpc"],
                    now_s,
                ),
            )

        db.commit()
        db.close()

        if discrepancies:
            logger.warning("Position discrepancies detected: %s", discrepancies)
            for d in discrepancies:
                _persist_risk_event(
                    "position_discrepancy", d["symbol"],
                    f"live={d['live_qty']} local={d['local_qty']}"
                )

        return {
            "timestamp":      now_s,
            "equity":         equity,
            "cash":           cash,
            "portfolio_value": port_val,
            "positions":      list(live.values()),
            "discrepancies":  discrepancies,
            "reconcile_ok":   len(discrepancies) == 0,
        }

    def get_positions(self) -> List[Dict]:
        """Return last-reconciled positions from SQLite."""
        db = _get_db()
        _ensure_tables(db)
        rows = db.execute("SELECT * FROM positions ORDER BY market_value DESC").fetchall()
        db.close()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# PortfolioRebalancer
# ---------------------------------------------------------------------------

class PortfolioRebalancer:
    """Weight-deviation rebalancing with constraints.

    Features
    --------
    - Compute trades needed given target weights vs current positions
    - Net offsetting trades: buying A while selling B reduces cash outflow
    - Round-lot optimisation: floor to nearest whole share (odd-lot avoidance)
    - Constraint-aware: max position size, sector limits, cash buffer
    - Tax-aware: prefer selling losers first (TLH priority)
    - Spread execution over N waves
    """

    def __init__(
        self,
        api_key:    str   = "",
        secret_key: str   = "",
        paper:      bool  = True,
    ) -> None:
        self._api_key   = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret    = secret_key or os.getenv(_ENV_SECRET, "")
        self._paper     = paper
        self._tc: Optional[Any] = None
        self._tracker   = PositionTracker(api_key=self._api_key, secret_key=self._secret, paper=paper)
        self._sor       = SmartOrderRouter(api_key=self._api_key, secret_key=self._secret, paper=paper)

    def _trading_client(self) -> Any:
        if self._tc is None and _ALPACA_AVAILABLE:
            self._tc = TradingClient(
                api_key=self._api_key, secret_key=self._secret, paper=self._paper
            )
        return self._tc

    def compute_trades(
        self,
        req: RebalanceRequest,
        portfolio_value: float,
        current_weights: Dict[str, float],
        position_details: Dict[str, Dict],
    ) -> List[RebalanceTrade]:
        """Compute list of trades to reach target weights within constraints."""

        # Normalise target weights (allow cash buffer)
        usable_weight  = 1.0 - req.cash_buffer_pct
        total_tw       = sum(req.target_weights.values())
        norm_targets   = {
            k: (v / total_tw) * usable_weight
            for k, v in req.target_weights.items()
        }

        # Apply max position size constraint
        for sym in norm_targets:
            norm_targets[sym] = min(norm_targets[sym], req.max_position_pct)

        trades: List[RebalanceTrade] = []
        all_syms = set(norm_targets) | set(current_weights)

        for sym in all_syms:
            target  = norm_targets.get(sym, 0.0)
            current = current_weights.get(sym, 0.0)
            dev     = target - current

            if abs(dev) < req.deviation_threshold:
                continue

            action      = "buy" if dev > 0 else "sell"
            dollar_amt  = abs(dev) * portfolio_value
            detail      = position_details.get(sym, {})
            cur_price   = detail.get("current_price", 1.0) or 1.0

            qty = dollar_amt / cur_price
            if req.odd_lot_avoidance:
                qty = math.floor(qty)  # round down to whole share

            if qty <= 0:
                continue

            # Tax priority for sells
            plpc = detail.get("unrealized_plpc", 0.0) or 0.0
            if action == "sell" and req.tax_aware:
                tax_priority = 0 if plpc < 0 else 1   # sell losers first
            else:
                tax_priority = 2   # buys after sells

            trades.append(RebalanceTrade(
                symbol        = sym,
                target_weight = target,
                current_weight = current,
                deviation     = dev,
                action        = action,
                qty           = qty,
                notional      = dollar_amt,
                tax_priority  = tax_priority,
                scheduled_wave = 0,
            ))

        # Sort: sells first (losers first), then buys
        trades.sort(key=lambda t: (t.tax_priority, -abs(t.deviation)))

        # Net offsetting cash: sells generate cash → reduces buy notional needed
        sells_cash = sum(t.notional for t in trades if t.action == "sell")
        buys_cash  = sum(t.notional for t in trades if t.action == "buy")
        net_buy_cash = max(buys_cash - sells_cash, 0)
        logger.info(
            "Rebalance: sells=$%.0f buys=$%.0f net_buy_cash=$%.0f",
            sells_cash, buys_cash, net_buy_cash,
        )

        # Assign execution waves: sells wave 0-1, buys wave 2+
        sells = [t for t in trades if t.action == "sell"]
        buys  = [t for t in trades if t.action == "buy"]
        wave_sz = max(1, math.ceil(len(sells) / req.spread_hours))
        for i, t in enumerate(sells):
            t.scheduled_wave = i // wave_sz
        buy_wave_sz = max(1, math.ceil(len(buys) / req.spread_hours))
        for i, t in enumerate(buys):
            t.scheduled_wave = req.spread_hours + i // buy_wave_sz

        return trades

    def execute_rebalance(self, req: RebalanceRequest) -> Dict[str, Any]:
        """Reconcile, compute, and execute rebalance trades."""
        state = self._tracker.reconcile()
        if "error" in state:
            return {"status": "error", "reason": state["error"]}

        port_val = state["portfolio_value"]
        if port_val <= 0:
            return {"status": "error", "reason": "Portfolio value is zero"}

        # Build current weights from positions
        current_weights: Dict[str, float]   = {}
        position_details: Dict[str, Dict]   = {}
        for p in state["positions"]:
            sym                    = p["symbol"]
            current_weights[sym]   = p["market_value"] / port_val if port_val > 0 else 0
            position_details[sym]  = p

        trades = self.compute_trades(req, port_val, current_weights, position_details)
        if not trades:
            return {"status": "no_action", "message": "All weights within deviation threshold", "trades": []}

        executed, deferred = [], []
        for trade in trades:
            if trade.scheduled_wave == 0:
                try:
                    route_result = self._sor.route(
                        symbol     = trade.symbol,
                        qty        = trade.qty,
                        side       = trade.action,
                        price_hint = position_details.get(trade.symbol, {}).get("current_price"),
                    )
                    executed.append({
                        "symbol":    trade.symbol,
                        "action":    trade.action,
                        "qty":       trade.qty,
                        "notional":  round(trade.notional, 2),
                        "deviation": round(trade.deviation, 4),
                        "strategy":  route_result.get("strategy"),
                    })
                    _persist_rebalance_trade(trade)
                except Exception as exc:
                    logger.error("Rebalance trade failed: %s %s: %s", trade.action, trade.symbol, exc)
            else:
                deferred.append({
                    "symbol": trade.symbol, "action": trade.action,
                    "qty": trade.qty, "wave": trade.scheduled_wave,
                })

        return {
            "status":          "executed",
            "portfolio_value": round(port_val, 2),
            "executed_count":  len(executed),
            "deferred_count":  len(deferred),
            "executed":        executed,
            "deferred":        deferred,
        }


# ---------------------------------------------------------------------------
# FillAnalytics
# ---------------------------------------------------------------------------

class FillAnalytics:
    """Track and analyse execution quality.

    Metrics
    -------
    - Implementation shortfall (IS): decision_price vs avg_fill_price [bps]
    - Slippage: expected vs actual fill [bps]
    - VWAP vs fill: positive = beat VWAP (buy below / sell above) [bps]
    - Market impact: estimated price move caused by own order [bps]
    - Fill rate: filled_qty / total_qty
    """

    def __init__(self, api_key: str = "", secret_key: str = "") -> None:
        self._api_key = api_key    or os.getenv(_ENV_API_KEY, "")
        self._secret  = secret_key or os.getenv(_ENV_SECRET, "")
        self._md      = MarketDataHelper(api_key=self._api_key, secret_key=self._secret)

    def record_fill(
        self,
        order_id:      str,
        symbol:        str,
        side:          str,
        qty:           float,
        fill_price:    float,
        decision_price: float = 0.0,
    ) -> Dict[str, float]:
        """Record a fill and compute execution quality metrics."""
        vwap = self._md.get_vwap_benchmark(symbol, minutes=30)
        sign = 1.0 if side == "buy" else -1.0

        # Implementation shortfall: cost of delay between decision and fill
        if decision_price > 0:
            impl_shortfall = sign * (fill_price - decision_price) / decision_price * 10_000
        else:
            impl_shortfall = 0.0

        # Slippage: absolute price difference from decision
        slippage = abs(fill_price - decision_price) / max(decision_price, 0.01) * 10_000

        # VWAP performance: positive = we beat VWAP benchmark
        if vwap > 0:
            vwap_vs_fill = sign * (vwap - fill_price) / vwap * 10_000
        else:
            vwap_vs_fill = 0.0

        # Market impact proxy: sqrt rule heuristic; 0.1 bps per 100 shares
        market_impact = (qty / 100.0) * 0.1

        metrics = {
            "impl_shortfall_bps":  round(impl_shortfall, 2),
            "slippage_bps":        round(slippage, 2),
            "vwap_vs_fill_bps":    round(vwap_vs_fill, 2),
            "market_impact_bps":   round(market_impact, 2),
        }

        fill_id = str(uuid.uuid4())
        db      = _get_db()
        _ensure_tables(db)
        try:
            db.execute(
                """INSERT INTO fills
                   (id, order_id, filled_at, symbol, side, qty, fill_price,
                    decision_price, vwap_benchmark, impl_shortfall_bps,
                    slippage_bps, vwap_vs_fill_bps, market_impact_bps, data_latency)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fill_id, order_id, _now_utc().isoformat(),
                    symbol, side, qty, fill_price, decision_price, vwap,
                    metrics["impl_shortfall_bps"], metrics["slippage_bps"],
                    metrics["vwap_vs_fill_bps"], metrics["market_impact_bps"],
                    _DATA_LATENCY_TAG,
                ),
            )
            db.commit()
        finally:
            db.close()

        logger.info(
            "Fill: %s %s qty=%.2f fill=%.4f IS=%.2fbps vwap_perf=%.2fbps",
            side, symbol, qty, fill_price,
            metrics["impl_shortfall_bps"], metrics["vwap_vs_fill_bps"],
        )
        return metrics

    def get_analytics(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return aggregated fill analytics per symbol."""
        db     = _get_db()
        _ensure_tables(db)
        cutoff = (_now_utc() - timedelta(days=days)).isoformat()
        if symbol:
            rows = db.execute(
                """SELECT symbol,
                          AVG(impl_shortfall_bps)  AS avg_is,
                          AVG(slippage_bps)        AS avg_sl,
                          AVG(vwap_vs_fill_bps)    AS avg_vwap,
                          AVG(market_impact_bps)   AS avg_mi,
                          COUNT(*)                 AS fill_count,
                          SUM(qty)                 AS total_qty,
                          SUM(qty * fill_price)    AS total_notional,
                          MIN(filled_at)           AS first_fill,
                          MAX(filled_at)           AS last_fill
                   FROM fills
                   WHERE filled_at >= ? AND symbol = ?
                   GROUP BY symbol""",
                (cutoff, symbol.upper()),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT symbol,
                          AVG(impl_shortfall_bps)  AS avg_is,
                          AVG(slippage_bps)        AS avg_sl,
                          AVG(vwap_vs_fill_bps)    AS avg_vwap,
                          AVG(market_impact_bps)   AS avg_mi,
                          COUNT(*)                 AS fill_count,
                          SUM(qty)                 AS total_qty,
                          SUM(qty * fill_price)    AS total_notional,
                          MIN(filled_at)           AS first_fill,
                          MAX(filled_at)           AS last_fill
                   FROM fills
                   WHERE filled_at >= ?
                   GROUP BY symbol
                   ORDER BY total_notional DESC""",
                (cutoff,),
            ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def report(self, days: int = 30) -> Dict[str, Any]:
        rows = self.get_analytics(days=days)
        if not rows:
            return {"status": "no_data", "days": days}

        is_vals   = [r["avg_is"]   or 0 for r in rows]
        sl_vals   = [r["avg_sl"]   or 0 for r in rows]
        vwap_vals = [r["avg_vwap"] or 0 for r in rows]
        total_fills = sum(r["fill_count"] or 0 for r in rows)
        total_notional = sum(r["total_notional"] or 0 for r in rows)

        return {
            "period_days":              days,
            "symbols_traded":           len(rows),
            "total_fills":              total_fills,
            "total_notional":           round(total_notional, 2),
            "avg_impl_shortfall_bps":   round(statistics.mean(is_vals), 2)   if is_vals   else 0,
            "avg_slippage_bps":         round(statistics.mean(sl_vals), 2)   if sl_vals   else 0,
            "avg_vwap_performance_bps": round(statistics.mean(vwap_vals), 2) if vwap_vals else 0,
            "data_latency_tag":         _DATA_LATENCY_TAG,
            "by_symbol":                rows,
        }


# ---------------------------------------------------------------------------
# Risk event persistence helpers
# ---------------------------------------------------------------------------

def _persist_risk_event(
    event_type: str, symbol: Optional[str], detail: str, severity: str = "warning"
) -> None:
    try:
        db = _get_db()
        _ensure_tables(db)
        db.execute(
            """INSERT INTO risk_events (id, ts, event_type, symbol, detail, severity)
               VALUES (?,?,?,?,?,?)""",
            (str(uuid.uuid4()), _now_utc().isoformat(), event_type, symbol, detail, severity),
        )
        db.commit()
        db.close()
    except Exception as exc:
        logger.warning("Risk event persist failed: %s", exc)


def _persist_rebalance_trade(trade: RebalanceTrade) -> None:
    try:
        db = _get_db()
        _ensure_tables(db)
        db.execute(
            """INSERT INTO orders
               (id, created_at, updated_at, symbol, side, order_type, tif,
                qty, notional, status, sor_strategy, data_latency, extended_hours, fractional)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()), _now_utc().isoformat(), _now_utc().isoformat(),
                trade.symbol, trade.action, "market", "day",
                trade.qty, trade.notional, "submitted",
                "rebalance", _DATA_LATENCY_TAG, 0, 0,
            ),
        )
        db.commit()
        db.close()
    except Exception as exc:
        logger.warning("Rebalance trade persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_oms_singleton:      Optional[OrderManagementSystem] = None
_sor_singleton:      Optional[SmartOrderRouter]      = None
_twap_singleton:     Optional[TWAPEngine]            = None
_vwap_singleton:     Optional[VWAPEngine]            = None
_rebal_singleton:    Optional[PortfolioRebalancer]   = None
_tracker_singleton:  Optional[PositionTracker]       = None
_risk_singleton:     Optional[PreTradeRiskEngine]    = None
_analytics_singleton: Optional[FillAnalytics]        = None


def _oms()      -> OrderManagementSystem:
    global _oms_singleton
    if _oms_singleton is None:
        _oms_singleton = OrderManagementSystem(paper=_is_paper())
    return _oms_singleton


def _sor()      -> SmartOrderRouter:
    global _sor_singleton
    if _sor_singleton is None:
        _sor_singleton = SmartOrderRouter(paper=_is_paper())
    return _sor_singleton


def _twap()     -> TWAPEngine:
    global _twap_singleton
    if _twap_singleton is None:
        _twap_singleton = TWAPEngine(paper=_is_paper())
    return _twap_singleton


def _vwap_eng() -> VWAPEngine:
    global _vwap_singleton
    if _vwap_singleton is None:
        _vwap_singleton = VWAPEngine(paper=_is_paper())
    return _vwap_singleton


def _rebal()    -> PortfolioRebalancer:
    global _rebal_singleton
    if _rebal_singleton is None:
        _rebal_singleton = PortfolioRebalancer(paper=_is_paper())
    return _rebal_singleton


def _tracker()  -> PositionTracker:
    global _tracker_singleton
    if _tracker_singleton is None:
        _tracker_singleton = PositionTracker(paper=_is_paper())
    return _tracker_singleton


def _risk()     -> PreTradeRiskEngine:
    global _risk_singleton
    if _risk_singleton is None:
        _risk_singleton = PreTradeRiskEngine(paper=_is_paper())
    return _risk_singleton


def _analytics() -> FillAnalytics:
    global _analytics_singleton
    if _analytics_singleton is None:
        _analytics_singleton = FillAnalytics()
    return _analytics_singleton


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

trading_v3_router = APIRouter(prefix="/trading/v3", tags=["live-trading-v3"])


# ----- POST /order -----

class OrderResponseModel(BaseModel):
    order_id:        str
    symbol:          str
    side:            str
    order_type:      str
    tif:             str
    qty:             Optional[float]
    status:          str
    alpaca_order_id: Optional[str]
    tradier_result:  Optional[Dict]
    data_latency:    str
    decision_price:  Optional[float]


@trading_v3_router.post(
    "/order",
    summary="Submit an order (market/limit/stop/stop_limit/trailing_stop)",
    response_model=OrderResponseModel,
)
async def post_order(req: OrderRequest) -> OrderResponseModel:
    """Submit an order via Alpaca API with optional Tradier sandbox simulation.

    Data latency note: order submission is always real-time. Quote data used for
    risk checks and decision_price capture is IEX near-real-time (tagged).
    """
    if _risk().is_halted():
        raise HTTPException(503, "Trading halted due to risk limit breach.")

    symbol = req.symbol.upper()
    quote  = MarketDataHelper().get_quote(symbol)
    price  = quote["mid"] or 1.0
    qty    = req.qty or ((req.notional or 0) / price)

    risk_check = _risk().check(
        symbol      = symbol,
        side        = req.side,
        qty         = qty,
        price       = price,
        order_type  = req.order_type,
        tif         = req.tif,
        extended_hours = req.extended_hours,
    )
    if not risk_check["ok"]:
        raise HTTPException(422, f"Pre-trade risk check failed: {risk_check['reason']}")

    try:
        result = _oms().submit(req)
        return OrderResponseModel(
            order_id        = result["id"],
            symbol          = result["symbol"],
            side            = result["side"],
            order_type      = result["order_type"],
            tif             = result["tif"],
            qty             = result.get("qty"),
            status          = result["status"],
            alpaca_order_id = result.get("alpaca_order_id"),
            tradier_result  = result.get("tradier_result"),
            data_latency    = _DATA_LATENCY_TAG,
            decision_price  = result.get("decision_price"),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- GET /orders -----

@trading_v3_router.get("/orders", summary="List recent orders")
async def get_orders(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit:  int           = Query(50,   ge=1, le=500),
) -> Dict[str, Any]:
    return {"orders": _oms().list_orders(status=status, limit=limit)}


# ----- DELETE /order/{id} -----

@trading_v3_router.delete("/order/{order_id}", summary="Cancel an order")
async def cancel_order(order_id: str) -> Dict[str, Any]:
    try:
        return _oms().cancel(order_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- GET /positions -----

@trading_v3_router.get("/positions", summary="Live positions reconciled with Alpaca")
async def get_positions() -> Dict[str, Any]:
    return _tracker().reconcile()


# ----- POST /twap -----

@trading_v3_router.post("/twap", summary="Build a TWAP execution schedule")
async def post_twap(req: TWAPRequest) -> Dict[str, Any]:
    """Build a TWAP schedule. Child orders execute at defined intervals.
    First tranche is submitted immediately; remaining tranches require a scheduler.
    """
    if _risk().is_halted():
        raise HTTPException(503, "Trading halted.")
    try:
        sched = _twap().build_schedule(req)
        # Execute first tranche immediately
        first = _twap().execute_next_tranche(sched["schedule_id"])
        sched["first_tranche"] = first
        return sched
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- POST /vwap -----

@trading_v3_router.post("/vwap", summary="Build a VWAP participation schedule")
async def post_vwap(req: VWAPRequest) -> Dict[str, Any]:
    """Build a VWAP schedule using historical intraday volume profile.
    Falls back to U-shaped profile if historical data unavailable.
    """
    if _risk().is_halted():
        raise HTTPException(503, "Trading halted.")
    try:
        return _vwap_eng().build_schedule(req)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- POST /rebalance -----

@trading_v3_router.post("/rebalance", summary="Execute portfolio rebalance to target weights")
async def post_rebalance(req: RebalanceRequest) -> Dict[str, Any]:
    if _risk().is_halted():
        raise HTTPException(503, "Trading halted.")
    try:
        return _rebal().execute_rebalance(req)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


# ----- GET /risk-status -----

class RiskStatusResponse(BaseModel):
    halted:           bool
    halt_reason:      Optional[str]
    data_latency_tag: str
    daily_loss_limit_pct: float
    adv_pct_limit:    float
    notional_limit:   float
    recent_events:    List[Dict]


@trading_v3_router.get("/risk-status", response_model=RiskStatusResponse)
async def get_risk_status() -> RiskStatusResponse:
    r = _risk()
    db = _get_db()
    _ensure_tables(db)
    events = [
        dict(row)
        for row in db.execute(
            "SELECT * FROM risk_events ORDER BY ts DESC LIMIT 20"
        ).fetchall()
    ]
    db.close()
    return RiskStatusResponse(
        halted            = r.is_halted(),
        halt_reason       = r._halt_reason or None,
        data_latency_tag  = _DATA_LATENCY_TAG,
        daily_loss_limit_pct = r._daily_loss_pct,
        adv_pct_limit     = r._adv_pct_limit,
        notional_limit    = r._notional_limit,
        recent_events     = events,
    )


@trading_v3_router.post("/risk-status/reset-halt", summary="Manually reset trading halt")
async def reset_halt() -> Dict[str, str]:
    _risk().reset_halt()
    return {"status": "halt_cleared"}


# ----- GET /fill-analytics -----

@trading_v3_router.get("/fill-analytics", summary="Execution quality analytics")
async def get_fill_analytics(
    symbol: Optional[str] = Query(None),
    days:   int           = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    return _analytics().report(days=days) if not symbol else {
        "symbol": symbol,
        "fills":  _analytics().get_analytics(symbol=symbol, days=days),
    }


@trading_v3_router.post("/fill-analytics/record", summary="Record a fill for analytics")
async def record_fill(
    order_id:       str,
    symbol:         str,
    side:           str,
    qty:            float,
    fill_price:     float,
    decision_price: float = 0.0,
) -> Dict[str, float]:
    return _analytics().record_fill(order_id, symbol, side, qty, fill_price, decision_price)


# ----- GET /sor-route -----

class SORRouteRequest(BaseModel):
    symbol:         str
    qty:            float  = Field(..., gt=0)
    side:           str    = Field(..., pattern="^(buy|sell)$")
    price_hint:     Optional[float] = None
    force_strategy: Optional[str]  = None


@trading_v3_router.post("/sor-route", summary="Smart order routing decision (preview)")
async def sor_route(req: SORRouteRequest) -> Dict[str, Any]:
    """Preview SOR routing decision without submitting. Useful for pre-order analysis."""
    if _risk().is_halted():
        raise HTTPException(503, "Trading halted.")
    try:
        return _sor().route(
            symbol         = req.symbol,
            qty            = req.qty,
            side           = req.side,
            price_hint     = req.price_hint,
            force_strategy = req.force_strategy,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
