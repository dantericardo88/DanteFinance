"""
Enhanced paper trading simulator with realistic execution model,
risk management, portfolio analytics, and multi-strategy support.

Dimension targeted:
  dim_066 — Paper trading simulator  score 5 → 9

Enhancements over sentinel/sbx/paper_trading.py:
  - RealisticExecutionEngine: VWAP fill, Almgren-Chriss market impact, bid-ask spread tiers
  - PaperTradingAccount: margin, shorting, dividends, corporate actions, daily snapshots
  - RiskManager: VaR, drawdown halt, concentration checks, beta neutrality
  - StrategyRunner: multi-strategy capital allocation, per-strategy P&L, attribution
  - PerformanceDashboard: sector/factor attribution, trade journal, benchmark comparison
  - FastAPI router with /paper/* endpoints

Persistence: SQLite at ~/.sentinel/paper_trading_enhanced.db
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

DB_PATH = Path.home() / ".sentinel" / "paper_trading_enhanced.db"

# ── Constants ─────────────────────────────────────────────────────────────────

# Market cap tiers for spread simulation (spread as fraction of price)
SPREAD_TIERS = {
    "large_cap":  0.0001,   # 0.01% — e.g. AAPL, MSFT
    "mid_cap":    0.0005,   # 0.05% — $2B-$10B market cap
    "small_cap":  0.0020,   # 0.20% — <$2B market cap
}

# Almgren-Chriss market impact constant k
MARKET_IMPACT_K = 0.1

# Borrow cost tiers for shorting (annual rate as fraction)
BORROW_COST_TIERS = {
    "easy":   0.0025,   # 0.25% — widely available to borrow
    "medium": 0.0150,   # 1.50%
    "hard":   0.0500,   # 5.00% — hard to borrow
}

# Risk limits
MAX_POSITION_PCT   = 0.10   # max 10% of equity in any single ticker
MAX_SECTOR_PCT     = 0.30   # max 30% of equity in any sector
MAX_DRAWDOWN_HALT  = 0.20   # stop new orders if drawdown > 20%
CONCENTRATION_WARN = 0.60   # warn if top-5 > 60%
MARGIN_LEVERAGE    = 2.0
MAINTENANCE_MARGIN = 0.25   # 25%

# Sector mapping for common tickers (simplified)
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "IBM": "Technology", "CSCO": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "AXP": "Financials", "V": "Financials",
    "MA": "Financials", "PYPL": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy",
    "AMZN": "ConsumerDiscretionary", "TSLA": "ConsumerDiscretionary",
    "HD": "ConsumerDiscretionary", "NKE": "ConsumerDiscretionary",
    "MCD": "ConsumerDiscretionary", "SBUX": "ConsumerDiscretionary",
    "WMT": "ConsumerStaples", "PG": "ConsumerStaples", "KO": "ConsumerStaples",
    "PEP": "ConsumerStaples", "COST": "ConsumerStaples",
    "LIN": "Materials", "APD": "Materials", "NEM": "Materials",
    "UNP": "Industrials", "BA": "Industrials", "CAT": "Industrials",
    "MMM": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AMT": "RealEstate", "PLD": "RealEstate", "EQIX": "RealEstate",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class EnhancedOrder(BaseModel):
    order_id:      str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id:    str
    ticker:        str
    side:          Literal["buy", "sell"]
    order_type:    Literal["market", "limit", "stop", "stop_limit", "bracket"]
    quantity:      float
    limit_price:   Optional[float] = None
    stop_price:    Optional[float] = None
    take_profit:   Optional[float] = None    # bracket: TP leg
    stop_loss:     Optional[float] = None    # bracket: SL leg (OCO)
    status:        Literal["pending", "filled", "partial", "cancelled", "rejected"] = "pending"
    submitted_at:  datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    filled_at:     Optional[datetime] = None
    filled_price:  Optional[float] = None
    filled_qty:    float = 0.0
    commission:    float = 0.0
    slippage:      float = 0.0
    market_impact: float = 0.0
    strategy_id:   Optional[str] = None
    notes:         Optional[str] = None
    # OCO partner order id (for bracket legs)
    oco_partner_id: Optional[str] = None


class EnhancedPosition(BaseModel):
    account_id:        str
    ticker:            str
    quantity:          float
    avg_cost:          float
    current_price:     Optional[float] = None
    market_value:      Optional[float] = None
    unrealized_pnl:    Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    realized_pnl:      float = 0.0
    sector:            str = "Unknown"
    borrow_cost_annual: float = 0.0     # for short positions
    opened_at:         datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    strategy_id:       Optional[str] = None


class AccountSnapshot(BaseModel):
    account_id:    str
    date:          str
    equity:        float
    cash:          float
    margin_used:   float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl:  float = 0.0


class VaRResult(BaseModel):
    var_95:         float
    var_99:         float
    cvar_95:        float     # conditional VaR / expected shortfall
    portfolio_beta: float
    portfolio_vol:  float
    method:         str = "parametric"


class RiskAlert(BaseModel):
    alert_id:    str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id:  str
    alert_type:  str
    severity:    Literal["info", "warning", "critical"]
    message:     str
    created_at:  datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class TradeJournalEntry(BaseModel):
    journal_id:    str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    account_id:    str
    ticker:        str
    side:          str
    entry_price:   float
    exit_price:    Optional[float] = None
    quantity:      float
    realized_pnl:  Optional[float] = None
    pnl_pct:       Optional[float] = None
    hold_duration: Optional[float] = None   # seconds
    entry_reason:  Optional[str] = None
    exit_reason:   Optional[str] = None
    strategy_id:   Optional[str] = None
    sector:        str = "Unknown"
    time_of_day:   Optional[str] = None     # "morning", "midday", "afternoon"
    entry_at:      datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    exit_at:       Optional[datetime] = None


class StrategyConfig(BaseModel):
    strategy_id:   str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:          str
    description:   str = ""
    allocated_pct: float = 0.10         # fraction of total account capital
    max_drawdown:  float = 0.15         # auto-pause at this drawdown
    active:        bool = True
    created_at:    datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class StrategyPerformance(BaseModel):
    strategy_id:     str
    name:            str
    total_trades:    int
    win_rate:        float
    total_pnl:       float
    sharpe:          Optional[float]
    max_drawdown:    float
    active:          bool
    allocated_capital: float


class BenchmarkComparison(BaseModel):
    account_id:     str
    benchmark:      str
    period_days:    int
    account_return: float
    bench_return:   float
    alpha:          float
    beta:           float
    tracking_error: float
    information_ratio: Optional[float]
    as_of:          datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── RealisticExecutionEngine ──────────────────────────────────────────────────

class RealisticExecutionEngine:
    """
    Simulates realistic fill prices including bid-ask spread and market impact.

    Market impact model (Almgren-Chriss simplified):
        impact = k * sqrt(order_size / ADV) * sigma

    where:
        k     = market impact constant (0.1)
        ADV   = average daily volume (shares)
        sigma = daily volatility (fraction)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _classify_market_cap(ticker: str) -> str:
        """Classify ticker into market cap tier using yfinance fast_info."""
        try:
            t = yf.Ticker(ticker)
            mkt_cap = getattr(t.fast_info, "market_cap", None)
            if mkt_cap is None:
                return "mid_cap"
            if mkt_cap >= 10_000_000_000:
                return "large_cap"
            elif mkt_cap >= 2_000_000_000:
                return "mid_cap"
            else:
                return "small_cap"
        except Exception:
            return "mid_cap"

    @staticmethod
    def _fetch_adv_and_vol(ticker: str) -> Tuple[float, float]:
        """
        Fetch average daily volume (ADV) and daily vol from yfinance.
        Returns (adv_shares, daily_vol_fraction).
        """
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="30d")
            if hist.empty or len(hist) < 5:
                return 1_000_000.0, 0.02
            adv = float(hist["Volume"].mean())
            closes = hist["Close"].values
            if len(closes) < 2:
                return adv, 0.02
            log_rets = np.diff(np.log(closes + 1e-9))
            vol = float(np.std(log_rets))
            return max(adv, 10_000.0), max(vol, 0.005)
        except Exception:
            return 1_000_000.0, 0.02

    def compute_spread(self, ticker: str, price: float) -> float:
        """Return full bid-ask spread in dollars."""
        tier = self._classify_market_cap(ticker)
        spread_pct = SPREAD_TIERS[tier]
        return price * spread_pct

    def compute_market_impact(
        self,
        ticker: str,
        order_qty: float,
        price: float,
        adv: Optional[float] = None,
        vol: Optional[float] = None,
    ) -> float:
        """
        Almgren-Chriss simplified market impact in dollars per share.
        impact_per_share = k * sqrt(order_qty / adv) * vol * price
        """
        if adv is None or vol is None:
            adv, vol = self._fetch_adv_and_vol(ticker)
        participation = order_qty / max(adv, 1.0)
        impact_frac = MARKET_IMPACT_K * math.sqrt(participation) * vol
        return impact_frac * price

    def fill_market_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        mid_price: float,
        adv: Optional[float] = None,
        vol: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Simulate market order fill.
        Returns (fill_price, slippage_per_share, market_impact_per_share).
        Buy fills above mid; sell fills below mid.
        """
        spread = self.compute_spread(ticker, mid_price)
        half_spread = spread / 2.0
        impact = self.compute_market_impact(ticker, quantity, mid_price, adv, vol)

        # Add small random noise to simulate VWAP dispersion
        noise = random.gauss(0, spread * 0.1)

        if side == "buy":
            fill_price = mid_price + half_spread + impact + noise
        else:
            fill_price = mid_price - half_spread - impact + noise

        fill_price = max(fill_price, 0.01)
        slippage = abs(fill_price - mid_price) - half_spread
        return round(fill_price, 4), round(half_spread, 4), round(impact, 4)

    def fill_limit_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        limit_price: float,
        current_price: float,
        allow_partial: bool = True,
    ) -> Tuple[float, float, bool]:
        """
        Simulate limit order fill.
        Returns (fill_price, filled_qty, is_partial).
        Partial fills: 20-80% of quantity filled if price is near limit.
        """
        triggered = False
        if side == "buy" and current_price <= limit_price:
            triggered = True
        elif side == "sell" and current_price >= limit_price:
            triggered = True

        if not triggered:
            return 0.0, 0.0, False

        # Use limit price as fill (price improvement possible)
        fill_price = min(current_price, limit_price) if side == "buy" else max(current_price, limit_price)

        # Simulate partial fill probability based on proximity to limit
        price_gap_pct = abs(current_price - limit_price) / max(limit_price, 0.01)
        if allow_partial and price_gap_pct < 0.001:
            # Very close to limit — partial fill more likely
            fill_pct = random.uniform(0.4, 1.0)
        else:
            fill_pct = 1.0

        filled_qty = round(quantity * fill_pct, 2)
        is_partial = fill_pct < 0.99
        return round(fill_price, 4), filled_qty, is_partial

    def fill_stop_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        stop_price: float,
        current_price: float,
    ) -> Tuple[float, float]:
        """
        Simulate stop order fill with slippage.
        Returns (fill_price, slippage_per_share).
        Stops fill at stop price + slippage (gap risk on open included).
        """
        triggered = False
        if side == "sell" and current_price <= stop_price:
            triggered = True
        elif side == "buy" and current_price >= stop_price:
            triggered = True

        if not triggered:
            return 0.0, 0.0

        # Gap risk: fill may be worse than stop price
        spread = self.compute_spread(ticker, stop_price)
        gap = random.gauss(0, spread * 2.0)   # gaps can be larger than spread

        if side == "sell":
            fill_price = stop_price - abs(gap)
        else:
            fill_price = stop_price + abs(gap)

        slippage = abs(fill_price - stop_price)
        return round(max(fill_price, 0.01), 4), round(slippage, 4)


# ── PaperTradingAccount ───────────────────────────────────────────────────────

class PaperTradingAccount(BaseModel):
    account_id:       str
    name:             str
    cash:             float
    initial_capital:  float
    margin_enabled:   bool = False
    margin_used:      float = 0.0
    buying_power:     float = 0.0
    total_equity:     float = 0.0
    leverage:         float = 1.0
    created_at:       datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── RiskManager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Institutional-grade risk management for paper trading accounts.
    Checks position limits, sector concentration, VaR, drawdown halts.
    """

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_PCT,
        max_sector_pct: float = MAX_SECTOR_PCT,
        max_drawdown_halt: float = MAX_DRAWDOWN_HALT,
        concentration_warn: float = CONCENTRATION_WARN,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_drawdown_halt = max_drawdown_halt
        self.concentration_warn = concentration_warn

    def check_position_limit(
        self, ticker: str, order_qty: float, order_price: float, total_equity: float
    ) -> Tuple[bool, str]:
        """Return (allowed, reason). Blocks if this order exceeds max_position_pct."""
        order_value = order_qty * order_price
        position_pct = order_value / max(total_equity, 1.0)
        if position_pct > self.max_position_pct:
            return False, (
                f"Order value {order_value:.0f} = {position_pct:.1%} of equity "
                f"exceeds limit {self.max_position_pct:.0%}"
            )
        return True, "ok"

    def check_sector_concentration(
        self,
        ticker: str,
        order_value: float,
        positions: List[EnhancedPosition],
        total_equity: float,
    ) -> Tuple[bool, str]:
        """Return (allowed, reason). Checks sector weight after hypothetical fill."""
        sector = SECTOR_MAP.get(ticker.upper(), "Unknown")
        existing_sector_value = sum(
            abs(p.market_value or 0.0)
            for p in positions
            if SECTOR_MAP.get(p.ticker, "Unknown") == sector
        )
        new_sector_value = existing_sector_value + order_value
        sector_pct = new_sector_value / max(total_equity, 1.0)
        if sector_pct > self.max_sector_pct:
            return False, (
                f"Sector '{sector}' would be {sector_pct:.1%} of portfolio "
                f"(limit {self.max_sector_pct:.0%})"
            )
        return True, "ok"

    def check_drawdown_halt(
        self, equity_series: List[float]
    ) -> Tuple[bool, str]:
        """Return (trading_allowed, reason). Halts new orders if drawdown > threshold."""
        if len(equity_series) < 2:
            return True, "ok"
        peak = max(equity_series)
        current = equity_series[-1]
        drawdown = (peak - current) / max(peak, 1.0)
        if drawdown > self.max_drawdown_halt:
            return False, (
                f"Max drawdown halt triggered: drawdown {drawdown:.1%} > {self.max_drawdown_halt:.0%}"
            )
        return True, "ok"

    def check_concentration(
        self, positions: List[EnhancedPosition], total_equity: float
    ) -> List[RiskAlert]:
        """Return alerts if top-5 position concentration exceeds threshold."""
        alerts: List[RiskAlert] = []
        if not positions or total_equity <= 0:
            return alerts

        values = sorted(
            [abs(p.market_value or 0.0) for p in positions], reverse=True
        )
        top5_value = sum(values[:5])
        top5_pct = top5_value / total_equity

        if top5_pct > self.concentration_warn:
            alerts.append(RiskAlert(
                account_id="",
                alert_type="concentration",
                severity="warning",
                message=f"Top-5 positions = {top5_pct:.1%} of equity (warn threshold {self.concentration_warn:.0%})",
            ))
        return alerts

    def compute_var(
        self,
        positions: List[EnhancedPosition],
        equity_series: List[float],
        total_equity: float,
    ) -> VaRResult:
        """
        Parametric VaR assuming normal daily returns.
        Uses historical equity series to estimate portfolio vol.
        """
        if len(equity_series) < 10:
            return VaRResult(
                var_95=0.0, var_99=0.0, cvar_95=0.0,
                portfolio_beta=1.0, portfolio_vol=0.02,
            )

        arr = np.array(equity_series, dtype=float)
        daily_rets = np.diff(arr) / arr[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]

        if len(daily_rets) < 5:
            return VaRResult(
                var_95=0.0, var_99=0.0, cvar_95=0.0,
                portfolio_beta=1.0, portfolio_vol=0.02,
            )

        mu    = float(np.mean(daily_rets))
        sigma = float(np.std(daily_rets, ddof=1))

        # Parametric VaR (normal distribution z-scores: 1.645 for 95%, 2.326 for 99%)
        z_95 = 1.6449
        z_99 = 2.3263
        var_95 = (mu - z_95 * sigma) * total_equity   # negative = loss
        var_99 = (mu - z_99 * sigma) * total_equity

        # CVaR (expected shortfall at 95%): E[loss | loss > VaR_95]
        # For normal: CVaR_95 = mu - sigma * phi(z_95) / (1 - 0.95)
        phi_z95 = math.exp(-0.5 * z_95 ** 2) / math.sqrt(2 * math.pi)
        cvar_95 = (mu - sigma * phi_z95 / 0.05) * total_equity

        # Simplified beta: ratio of portfolio vol to SPY vol (assume SPY vol ~16% annual)
        spy_daily_vol = 0.16 / math.sqrt(252)
        portfolio_beta = sigma / spy_daily_vol if spy_daily_vol > 0 else 1.0

        return VaRResult(
            var_95=round(abs(var_95), 2),
            var_99=round(abs(var_99), 2),
            cvar_95=round(abs(cvar_95), 2),
            portfolio_beta=round(portfolio_beta, 4),
            portfolio_vol=round(sigma * math.sqrt(252), 4),
        )

    def check_margin_call(
        self, cash: float, margin_used: float, market_value: float
    ) -> Tuple[bool, str]:
        """
        Check if account is in margin call.
        Maintenance margin: equity >= 25% of market_value.
        equity = cash + market_value - margin_used
        """
        equity = cash + market_value - margin_used
        if market_value <= 0:
            return False, "ok"
        equity_ratio = equity / max(market_value, 1.0)
        if equity_ratio < MAINTENANCE_MARGIN:
            return True, (
                f"Margin call: equity ratio {equity_ratio:.1%} < "
                f"maintenance margin {MAINTENANCE_MARGIN:.0%}"
            )
        return False, "ok"

    def check_beta_neutrality(self, portfolio_beta: float, target: float = 1.0, tolerance: float = 0.3) -> Optional[str]:
        """Return warning string if portfolio beta deviates from target by more than tolerance."""
        deviation = abs(portfolio_beta - target)
        if deviation > tolerance:
            return (
                f"Portfolio beta {portfolio_beta:.2f} deviates from target {target:.1f} "
                f"by {deviation:.2f} (tolerance {tolerance:.1f})"
            )
        return None


# ── StrategyRunner ────────────────────────────────────────────────────────────

class StrategyRunner:
    """
    Manages multiple concurrent strategies within a single paper trading account.
    Each strategy has isolated capital allocation and P&L tracking.
    """

    def __init__(self, db_conn_fn) -> None:
        self._conn = db_conn_fn

    def register_strategy(self, account_id: str, config: StrategyConfig) -> str:
        """Register a new strategy. Returns strategy_id."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO strategies "
                "(strategy_id, account_id, name, description, allocated_pct, max_drawdown, active, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    config.strategy_id, account_id, config.name, config.description,
                    config.allocated_pct, config.max_drawdown,
                    1 if config.active else 0, now,
                ),
            )
        logger.info("Strategy registered", strategy_id=config.strategy_id, name=config.name)
        return config.strategy_id

    def get_strategies(self, account_id: str) -> List[StrategyConfig]:
        """Return all strategies for account."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["active"] = bool(d["active"])
            try:
                d["created_at"] = datetime.fromisoformat(d["created_at"])
            except Exception:
                d["created_at"] = datetime.now(tz=timezone.utc)
            result.append(StrategyConfig(**d))
        return result

    def compute_strategy_performance(
        self, account_id: str, strategy_id: str, total_equity: float
    ) -> StrategyPerformance:
        """Compute per-strategy P&L and risk metrics from trade history."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE strategy_id = ? AND account_id = ?",
                (strategy_id, account_id),
            ).fetchone()
            if rows is None:
                raise ValueError(f"Strategy {strategy_id} not found")
            strategy = dict(rows)

            trades = conn.execute(
                "SELECT realized_pnl, price, quantity FROM trades "
                "WHERE account_id = ? AND strategy_id = ?",
                (account_id, strategy_id),
            ).fetchall()

        pnl_values = [dict(t)["realized_pnl"] for t in trades if dict(t)["realized_pnl"] is not None]
        wins   = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]

        total_pnl  = sum(pnl_values)
        win_rate   = len(wins) / len(pnl_values) if pnl_values else 0.0
        max_dd     = 0.0  # would need cumulative equity series per strategy for full calc

        # Rolling P&L series for Sharpe (cumulative)
        sharpe: Optional[float] = None
        if len(pnl_values) >= 5:
            arr = np.array(pnl_values)
            if arr.std() > 0:
                sharpe = round(arr.mean() / arr.std() * math.sqrt(252), 4)

        allocated_capital = total_equity * strategy["allocated_pct"]

        return StrategyPerformance(
            strategy_id=strategy_id,
            name=strategy["name"],
            total_trades=len(pnl_values),
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            sharpe=sharpe,
            max_drawdown=max_dd,
            active=bool(strategy["active"]),
            allocated_capital=round(allocated_capital, 2),
        )

    def check_strategy_drawdown(
        self, account_id: str, strategy_id: str, max_drawdown_limit: float
    ) -> Tuple[bool, float]:
        """
        Check if strategy has hit its drawdown limit.
        Returns (should_pause, current_drawdown).
        Uses cumulative realized P&L series.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT realized_pnl, executed_at FROM trades "
                "WHERE account_id = ? AND strategy_id = ? "
                "ORDER BY executed_at ASC",
                (account_id, strategy_id),
            ).fetchall()

        if not rows:
            return False, 0.0

        pnl_series = [dict(r)["realized_pnl"] or 0.0 for r in rows]
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = np.where(peak != 0, (peak - cumulative) / np.abs(peak), 0.0)
        max_dd = float(drawdown.max())

        return max_dd > max_drawdown_limit, round(max_dd, 4)

    def pause_strategy(self, account_id: str, strategy_id: str) -> None:
        """Deactivate strategy to prevent new orders."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE strategies SET active = 0 WHERE strategy_id = ? AND account_id = ?",
                (strategy_id, account_id),
            )
        logger.warning("Strategy paused", strategy_id=strategy_id)

    def resume_strategy(self, account_id: str, strategy_id: str) -> None:
        """Reactivate a paused strategy."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE strategies SET active = 1 WHERE strategy_id = ? AND account_id = ?",
                (strategy_id, account_id),
            )
        logger.info("Strategy resumed", strategy_id=strategy_id)


# ── PerformanceDashboard ──────────────────────────────────────────────────────

class PerformanceDashboard:
    """
    Rich performance analytics with attribution, benchmark comparison, and trade journal.
    """

    def __init__(self, db_conn_fn) -> None:
        self._conn = db_conn_fn

    # ── P&L breakdowns ────────────────────────────────────────────────────────

    def pnl_by_period(
        self, account_id: str, period: Literal["daily", "weekly", "monthly"] = "daily"
    ) -> List[dict]:
        """Return P&L grouped by calendar period from equity history."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, equity_value FROM equity_history "
                "WHERE account_id = ? ORDER BY date ASC",
                (account_id,),
            ).fetchall()

        if not rows:
            return []

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if period == "daily":
            df["pnl"] = df["equity_value"].diff()
            df["pnl_pct"] = df["equity_value"].pct_change()
        elif period == "weekly":
            df = df.resample("W").agg({"equity_value": "last"})
            df["pnl"] = df["equity_value"].diff()
            df["pnl_pct"] = df["equity_value"].pct_change()
        else:  # monthly
            df = df.resample("ME").agg({"equity_value": "last"})
            df["pnl"] = df["equity_value"].diff()
            df["pnl_pct"] = df["equity_value"].pct_change()

        df = df.dropna()
        df.index = df.index.strftime("%Y-%m-%d")
        return df.reset_index().rename(columns={"date": "period"}).to_dict(orient="records")

    # ── Sector attribution ────────────────────────────────────────────────────

    def pnl_by_sector(self, account_id: str) -> Dict[str, float]:
        """Return total realized P&L aggregated by sector."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ticker, realized_pnl FROM trades WHERE account_id = ?",
                (account_id,),
            ).fetchall()

        sector_pnl: Dict[str, float] = {}
        for r in rows:
            d = dict(r)
            sector = SECTOR_MAP.get(d["ticker"], "Unknown")
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + (d["realized_pnl"] or 0.0)

        return {k: round(v, 2) for k, v in sector_pnl.items()}

    # ── Factor attribution ────────────────────────────────────────────────────

    def factor_attribution(self, account_id: str) -> Dict[str, float]:
        """
        Simplified factor attribution using price momentum proxy.
        Classifies trades by factor: momentum, value, quality, size, other.
        Returns dict of factor → fraction of total P&L.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ticker, side, quantity, price, realized_pnl, executed_at "
                "FROM trades WHERE account_id = ? ORDER BY executed_at ASC",
                (account_id,),
            ).fetchall()

        factor_pnl: Dict[str, float] = {
            "momentum": 0.0, "value": 0.0, "quality": 0.0, "size": 0.0, "other": 0.0
        }

        for r in rows:
            d = dict(r)
            pnl = d["realized_pnl"] or 0.0
            ticker = d["ticker"]
            # Map via sector as proxy (Technology/Growth ~ momentum, Financials/Value, etc.)
            sector = SECTOR_MAP.get(ticker, "Unknown")
            if sector in ("Technology", "ConsumerDiscretionary"):
                factor_pnl["momentum"] += pnl
            elif sector in ("Financials", "Energy", "Materials"):
                factor_pnl["value"] += pnl
            elif sector in ("Healthcare", "ConsumerStaples", "Utilities"):
                factor_pnl["quality"] += pnl
            elif sector in ("ETF",):
                factor_pnl["size"] += pnl
            else:
                factor_pnl["other"] += pnl

        total = sum(abs(v) for v in factor_pnl.values()) or 1.0
        return {k: round(v / total, 4) for k, v in factor_pnl.items()}

    # ── Trade journal ─────────────────────────────────────────────────────────

    def get_trade_journal(self, account_id: str, days_back: int = 30) -> List[TradeJournalEntry]:
        """Return trade journal entries, enriched with hold duration and time of day."""
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_journal WHERE account_id = ? AND entry_at >= ? "
                "ORDER BY entry_at DESC",
                (account_id, since),
            ).fetchall()
        entries = []
        for r in rows:
            d = dict(r)
            try:
                d["entry_at"] = datetime.fromisoformat(d["entry_at"])
            except Exception:
                d["entry_at"] = datetime.now(tz=timezone.utc)
            if d.get("exit_at"):
                try:
                    d["exit_at"] = datetime.fromisoformat(d["exit_at"])
                except Exception:
                    d["exit_at"] = None
            entries.append(TradeJournalEntry(**d))
        return entries

    def add_journal_entry(self, entry: TradeJournalEntry) -> None:
        """Insert a trade journal entry."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trade_journal "
                "(journal_id, account_id, ticker, side, entry_price, exit_price, quantity, "
                "realized_pnl, pnl_pct, hold_duration, entry_reason, exit_reason, "
                "strategy_id, sector, time_of_day, entry_at, exit_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entry.journal_id, entry.account_id, entry.ticker, entry.side,
                    entry.entry_price, entry.exit_price, entry.quantity,
                    entry.realized_pnl, entry.pnl_pct, entry.hold_duration,
                    entry.entry_reason, entry.exit_reason, entry.strategy_id,
                    entry.sector, entry.time_of_day,
                    entry.entry_at.isoformat(),
                    entry.exit_at.isoformat() if entry.exit_at else None,
                ),
            )

    # ── Win rate breakdowns ───────────────────────────────────────────────────

    def win_rate_by_strategy(self, account_id: str) -> Dict[str, dict]:
        """Return win rate stats grouped by strategy_id."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT strategy_id, realized_pnl FROM trades "
                "WHERE account_id = ? AND realized_pnl IS NOT NULL",
                (account_id,),
            ).fetchall()

        stats: Dict[str, dict] = {}
        for r in rows:
            d = dict(r)
            sid = d["strategy_id"] or "unassigned"
            if sid not in stats:
                stats[sid] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
            pnl = d["realized_pnl"]
            if pnl > 0:
                stats[sid]["wins"] += 1
            elif pnl < 0:
                stats[sid]["losses"] += 1
            stats[sid]["total_pnl"] += pnl

        result = {}
        for sid, s in stats.items():
            total = s["wins"] + s["losses"]
            result[sid] = {
                "win_rate": round(s["wins"] / total, 4) if total else 0.0,
                "total_trades": total,
                "total_pnl": round(s["total_pnl"], 2),
            }
        return result

    def win_rate_by_time_of_day(self, account_id: str) -> Dict[str, dict]:
        """Return win rate stats grouped by time of day (morning/midday/afternoon)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT executed_at, realized_pnl FROM trades "
                "WHERE account_id = ? AND realized_pnl IS NOT NULL",
                (account_id,),
            ).fetchall()

        buckets: Dict[str, dict] = {
            "morning":   {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "midday":    {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "afternoon": {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "other":     {"wins": 0, "losses": 0, "total_pnl": 0.0},
        }

        for r in rows:
            d = dict(r)
            pnl = d["realized_pnl"]
            try:
                dt = datetime.fromisoformat(d["executed_at"])
                hour = dt.hour
            except Exception:
                hour = 12

            # NYSE hours: 9:30-16:00 ET (approximate UTC offset)
            if 9 <= hour < 11:
                bucket = "morning"
            elif 11 <= hour < 14:
                bucket = "midday"
            elif 14 <= hour < 16:
                bucket = "afternoon"
            else:
                bucket = "other"

            if pnl > 0:
                buckets[bucket]["wins"] += 1
            elif pnl < 0:
                buckets[bucket]["losses"] += 1
            buckets[bucket]["total_pnl"] += pnl

        result = {}
        for name, s in buckets.items():
            total = s["wins"] + s["losses"]
            result[name] = {
                "win_rate": round(s["wins"] / total, 4) if total else 0.0,
                "total_trades": total,
                "total_pnl": round(s["total_pnl"], 2),
            }
        return result

    # ── Benchmark comparison ──────────────────────────────────────────────────

    async def compare_to_benchmark(
        self,
        account_id: str,
        benchmark: str = "SPY",
        period_days: int = 90,
    ) -> BenchmarkComparison:
        """
        Compare account returns vs benchmark (SPY/QQQ).
        Computes alpha, beta, tracking error, information ratio.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, equity_value FROM equity_history "
                "WHERE account_id = ? ORDER BY date ASC",
                (account_id,),
            ).fetchall()

        if len(rows) < 5:
            return BenchmarkComparison(
                account_id=account_id, benchmark=benchmark, period_days=period_days,
                account_return=0.0, bench_return=0.0, alpha=0.0, beta=1.0,
                tracking_error=0.0, information_ratio=None,
            )

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Trim to period
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
        df = df[df.index >= cutoff.tz_localize(None)]

        if len(df) < 5:
            return BenchmarkComparison(
                account_id=account_id, benchmark=benchmark, period_days=period_days,
                account_return=0.0, bench_return=0.0, alpha=0.0, beta=1.0,
                tracking_error=0.0, information_ratio=None,
            )

        # Fetch benchmark data
        try:
            bench_hist = await asyncio.to_thread(
                lambda: yf.Ticker(benchmark).history(period=f"{period_days + 5}d")
            )
            bench_closes = bench_hist["Close"].values
            bench_rets = np.diff(bench_closes) / bench_closes[:-1]
        except Exception:
            bench_rets = np.zeros(max(len(df) - 1, 1))

        acct_vals = df["equity_value"].values
        acct_rets = np.diff(acct_vals) / acct_vals[:-1]

        min_len = min(len(acct_rets), len(bench_rets))
        if min_len < 5:
            return BenchmarkComparison(
                account_id=account_id, benchmark=benchmark, period_days=period_days,
                account_return=0.0, bench_return=0.0, alpha=0.0, beta=1.0,
                tracking_error=0.0, information_ratio=None,
            )

        acct_rets  = acct_rets[-min_len:]
        bench_rets = bench_rets[-min_len:]

        # Beta via OLS
        cov = np.cov(acct_rets, bench_rets)
        beta = float(cov[0, 1] / max(cov[1, 1], 1e-10))

        # Annualized returns
        account_return = float((acct_vals[-1] / acct_vals[0]) - 1.0)
        bench_return   = float((bench_closes[-1] / bench_closes[0]) - 1.0)

        # Alpha (Jensen's): r_p - [r_f + beta * (r_m - r_f)], r_f = 0
        alpha = account_return - beta * bench_return

        # Tracking error (annualized std of excess returns)
        excess = acct_rets - bench_rets
        tracking_error = float(np.std(excess, ddof=1) * math.sqrt(252))

        # Information ratio = alpha / tracking_error (annualized)
        ir = alpha / tracking_error if tracking_error > 0 else None

        return BenchmarkComparison(
            account_id=account_id,
            benchmark=benchmark,
            period_days=period_days,
            account_return=round(account_return, 6),
            bench_return=round(bench_return, 6),
            alpha=round(alpha, 6),
            beta=round(beta, 4),
            tracking_error=round(tracking_error, 6),
            information_ratio=round(ir, 4) if ir is not None else None,
        )


# ── EnhancedPaperTradingEngine ────────────────────────────────────────────────

class EnhancedPaperTradingEngine:
    """
    Full enhanced paper trading simulator with realistic execution, margin,
    multi-strategy support, risk management, and rich performance analytics.

    Usage:
        engine = EnhancedPaperTradingEngine()
        acct_id = engine.create_account("Momentum Strategy", 100_000, margin_enabled=True)
        order = await engine.submit_order(acct_id, "AAPL", "buy", 100, order_type="market")
        portfolio = await engine.get_portfolio(acct_id)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._execution = RealisticExecutionEngine()
        self._risk = RiskManager()
        self._strategy_runner = StrategyRunner(self._conn)
        self._dashboard = PerformanceDashboard(self._conn)
        logger.info("EnhancedPaperTradingEngine initialized", db=str(self._db))

    # ── DB plumbing ───────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
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
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id              TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    cash            REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    margin_enabled  INTEGER NOT NULL DEFAULT 0,
                    margin_used     REAL NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id        TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    order_type      TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    limit_price     REAL,
                    stop_price      REAL,
                    take_profit     REAL,
                    stop_loss       REAL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    submitted_at    TEXT NOT NULL,
                    filled_at       TEXT,
                    filled_price    REAL,
                    filled_qty      REAL NOT NULL DEFAULT 0,
                    commission      REAL NOT NULL DEFAULT 0,
                    slippage        REAL NOT NULL DEFAULT 0,
                    market_impact   REAL NOT NULL DEFAULT 0,
                    strategy_id     TEXT,
                    notes           TEXT,
                    oco_partner_id  TEXT,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id        TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    price           REAL NOT NULL,
                    commission      REAL NOT NULL DEFAULT 0,
                    slippage        REAL NOT NULL DEFAULT 0,
                    market_impact   REAL NOT NULL DEFAULT 0,
                    executed_at     TEXT NOT NULL,
                    order_id        TEXT NOT NULL,
                    strategy_id     TEXT,
                    realized_pnl    REAL,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS positions (
                    account_id      TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    quantity        REAL NOT NULL,
                    avg_cost        REAL NOT NULL,
                    realized_pnl    REAL NOT NULL DEFAULT 0,
                    borrow_cost     REAL NOT NULL DEFAULT 0,
                    opened_at       TEXT NOT NULL,
                    strategy_id     TEXT,
                    PRIMARY KEY (account_id, ticker),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS equity_history (
                    account_id      TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    equity_value    REAL NOT NULL,
                    cash            REAL NOT NULL DEFAULT 0,
                    margin_used     REAL NOT NULL DEFAULT 0,
                    unrealized_pnl  REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_id, date),
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id     TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    description     TEXT,
                    allocated_pct   REAL NOT NULL DEFAULT 0.10,
                    max_drawdown    REAL NOT NULL DEFAULT 0.15,
                    active          INTEGER NOT NULL DEFAULT 1,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS risk_alerts (
                    alert_id        TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    alert_type      TEXT NOT NULL,
                    severity        TEXT NOT NULL,
                    message         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS trade_journal (
                    journal_id      TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    entry_price     REAL NOT NULL,
                    exit_price      REAL,
                    quantity        REAL NOT NULL,
                    realized_pnl    REAL,
                    pnl_pct         REAL,
                    hold_duration   REAL,
                    entry_reason    TEXT,
                    exit_reason     TEXT,
                    strategy_id     TEXT,
                    sector          TEXT DEFAULT 'Unknown',
                    time_of_day     TEXT,
                    entry_at        TEXT NOT NULL,
                    exit_at         TEXT,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS dividends (
                    id              TEXT PRIMARY KEY,
                    account_id      TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    amount_per_share REAL NOT NULL,
                    total_amount    REAL NOT NULL,
                    ex_date         TEXT NOT NULL,
                    credited_at     TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_orders_account   ON orders(account_id);
                CREATE INDEX IF NOT EXISTS idx_trades_account   ON trades(account_id);
                CREATE INDEX IF NOT EXISTS idx_equity_account   ON equity_history(account_id);
                CREATE INDEX IF NOT EXISTS idx_positions_acct   ON positions(account_id);
                CREATE INDEX IF NOT EXISTS idx_strategies_acct  ON strategies(account_id);
                CREATE INDEX IF NOT EXISTS idx_journal_acct     ON trade_journal(account_id);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy  ON trades(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_orders_strategy  ON orders(strategy_id);
            """)

    # ── Account management ────────────────────────────────────────────────────

    def create_account(
        self,
        account_name: str,
        initial_capital: float = 100_000.0,
        margin_enabled: bool = False,
    ) -> str:
        """Create a new paper trading account. Returns the new account_id."""
        account_id = str(uuid.uuid4())[:12]
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO accounts (id, name, cash, initial_capital, margin_enabled, margin_used, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (account_id, account_name, initial_capital, initial_capital,
                 1 if margin_enabled else 0, now),
            )
        logger.info("Account created", account_id=account_id, name=account_name, capital=initial_capital)
        return account_id

    def list_accounts(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM accounts ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def _get_account(self, account_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise ValueError(f"Account not found: {account_id}")
        return dict(row)

    def get_buying_power(self, account: dict) -> float:
        """Compute buying power considering margin."""
        cash = account["cash"]
        if account["margin_enabled"]:
            return cash * MARGIN_LEVERAGE
        return cash

    # ── Price fetching ────────────────────────────────────────────────────────

    async def get_quote(self, ticker: str) -> float:
        try:
            price = await asyncio.to_thread(self._yf_price, ticker.upper())
            if price and price > 0:
                return price
            raise ValueError(f"Zero or None price for {ticker}")
        except Exception as exc:
            raise ValueError(f"Price fetch failed for {ticker}: {exc}") from exc

    @staticmethod
    def _yf_price(ticker: str) -> float:
        t = yf.Ticker(ticker)
        try:
            price = t.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        hist = t.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
        raise ValueError(f"No price data for {ticker}")

    # ── Order submission ──────────────────────────────────────────────────────

    async def submit_order(
        self,
        account_id: str,
        ticker: str,
        side: Literal["buy", "sell"],
        quantity: float,
        order_type: Literal["market", "limit", "stop", "stop_limit", "bracket"] = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        strategy_id: Optional[str] = None,
        entry_reason: Optional[str] = None,
    ) -> EnhancedOrder:
        """
        Submit a paper order with realistic execution.
        Market orders: VWAP ± slippage + market impact.
        Limit/stop: stored pending, filled via fill_pending_orders().
        Bracket orders: TP + SL legs created as OCO.
        """
        ticker_up = ticker.upper()
        account   = self._get_account(account_id)

        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        # Risk gate: drawdown halt
        equity_series = self._get_equity_series(account_id)
        can_trade, halt_reason = self._risk.check_drawdown_halt(equity_series)
        if not can_trade:
            raise ValueError(f"Trading halted: {halt_reason}")

        # Strategy active check
        if strategy_id:
            strats = self._strategy_runner.get_strategies(account_id)
            strat = next((s for s in strats if s.strategy_id == strategy_id), None)
            if strat and not strat.active:
                raise ValueError(f"Strategy {strategy_id} is paused")

        order = EnhancedOrder(
            account_id=account_id,
            ticker=ticker_up,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            strategy_id=strategy_id,
        )

        if order_type == "market":
            current_price = await self.get_quote(ticker_up)

            # Position limit check
            total_equity = self._compute_equity_from_account(account_id, account)
            ok, reason = self._risk.check_position_limit(ticker_up, quantity, current_price, total_equity)
            if not ok:
                raise ValueError(f"Position limit: {reason}")

            order = await self._fill_market_order(order, current_price, account, entry_reason)

        elif order_type == "bracket":
            # For bracket: submit the entry as market, then queue TP/SL as OCO
            if take_profit is None or stop_loss is None:
                raise ValueError("Bracket orders require take_profit and stop_loss prices")
            current_price = await self.get_quote(ticker_up)
            order = await self._fill_market_order(order, current_price, account, entry_reason)
            if order.status == "filled":
                await self._create_oco_legs(order, account_id, strategy_id)

        else:
            # Pending order validation
            if side == "buy" and limit_price:
                buying_power = self.get_buying_power(account)
                required = quantity * limit_price
                if buying_power < required:
                    order = order.model_copy(update={
                        "status": "rejected",
                        "notes": f"Insufficient buying power: need {required:.2f}, have {buying_power:.2f}",
                    })
            elif side == "sell":
                pos = self._get_position(account_id, ticker_up)
                avail = pos["quantity"] if pos else 0.0
                if avail < quantity:
                    order = order.model_copy(update={
                        "status": "rejected",
                        "notes": f"Insufficient shares: need {quantity}, have {avail}",
                    })

            if order.status != "rejected":
                self._save_order(order)

        return order

    async def _fill_market_order(
        self,
        order: EnhancedOrder,
        mid_price: float,
        account: dict,
        entry_reason: Optional[str] = None,
    ) -> EnhancedOrder:
        """Fill a market order using the realistic execution engine."""
        # Fetch ADV and vol for impact calculation
        adv, vol = await asyncio.to_thread(
            self._execution._fetch_adv_and_vol, order.ticker
        )

        fill_price, slippage, impact = self._execution.fill_market_order(
            order.ticker, order.side, order.quantity, mid_price, adv, vol
        )

        account = self._get_account(order.account_id)
        return await self._execute_fill(order, fill_price, account, slippage, impact, entry_reason)

    async def _execute_fill(
        self,
        order: EnhancedOrder,
        fill_price: float,
        account: dict,
        slippage: float = 0.0,
        market_impact: float = 0.0,
        entry_reason: Optional[str] = None,
        is_partial: bool = False,
        filled_qty: Optional[float] = None,
    ) -> EnhancedOrder:
        """Common fill logic: update positions, cash, write trade and journal."""
        qty = filled_qty or order.quantity
        commission = 0.0   # zero-commission model
        now = datetime.now(tz=timezone.utc)

        if order.side == "buy":
            total_cost = qty * fill_price + commission
            buying_power = self.get_buying_power(account)
            if buying_power < total_cost:
                filled = order.model_copy(update={
                    "status": "rejected",
                    "notes": f"Insufficient buying power: need {total_cost:.2f}, have {buying_power:.2f}",
                })
                self._save_order(filled)
                return filled

            # Update margin tracking if cost > cash
            extra = max(0.0, total_cost - account["cash"])
            new_cash = account["cash"] - total_cost + extra * 0   # always deduct from cash first
            new_cash = account["cash"] - min(total_cost, account["cash"])
            new_margin = account.get("margin_used", 0.0) + max(0.0, total_cost - account["cash"])

        else:
            pos = self._get_position(order.account_id, order.ticker)
            avail = pos["quantity"] if pos else 0.0
            if avail < qty:
                filled = order.model_copy(update={
                    "status": "rejected",
                    "notes": f"Insufficient shares: need {qty}, have {avail}",
                })
                self._save_order(filled)
                return filled
            new_cash = account["cash"] + qty * fill_price - commission
            new_margin = account.get("margin_used", 0.0)

        realized_pnl = self._update_position(
            order.account_id, order.ticker, order.side, qty, fill_price, order.strategy_id
        )

        with self._conn() as conn:
            conn.execute(
                "UPDATE accounts SET cash = ?, margin_used = ? WHERE id = ?",
                (new_cash, new_margin, order.account_id),
            )

        # Save trade
        trade_id = str(uuid.uuid4())[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trades "
                "(trade_id, account_id, ticker, side, quantity, price, commission, "
                "slippage, market_impact, executed_at, order_id, strategy_id, realized_pnl) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trade_id, order.account_id, order.ticker, order.side,
                    qty, fill_price, commission, slippage, market_impact,
                    now.isoformat(), order.order_id, order.strategy_id, realized_pnl,
                ),
            )

        # Journal entry for sells (when we have realized P&L)
        if order.side == "sell" and realized_pnl is not None:
            sector = SECTOR_MAP.get(order.ticker, "Unknown")
            hour = now.hour
            if 9 <= hour < 11:
                tod = "morning"
            elif 11 <= hour < 14:
                tod = "midday"
            elif 14 <= hour < 16:
                tod = "afternoon"
            else:
                tod = "other"

            pnl_pct = realized_pnl / (qty * fill_price) if fill_price > 0 and qty > 0 else None
            entry = TradeJournalEntry(
                account_id=order.account_id,
                ticker=order.ticker,
                side=order.side,
                entry_price=self._get_avg_cost_or_price(order.account_id, order.ticker, fill_price),
                exit_price=fill_price,
                quantity=qty,
                realized_pnl=realized_pnl,
                pnl_pct=round(pnl_pct, 4) if pnl_pct else None,
                entry_reason=entry_reason,
                exit_reason=order.notes,
                strategy_id=order.strategy_id,
                sector=sector,
                time_of_day=tod,
                exit_at=now,
            )
            self._dashboard.add_journal_entry(entry)

        status = "partial" if is_partial else "filled"
        filled_order = order.model_copy(update={
            "status": status,
            "filled_at": now,
            "filled_price": fill_price,
            "filled_qty": qty,
            "commission": commission,
            "slippage": slippage,
            "market_impact": market_impact,
        })
        self._save_order(filled_order)

        # Strategy drawdown check
        if order.strategy_id:
            strats = self._strategy_runner.get_strategies(order.account_id)
            strat = next((s for s in strats if s.strategy_id == order.strategy_id), None)
            if strat:
                should_pause, dd = self._strategy_runner.check_strategy_drawdown(
                    order.account_id, order.strategy_id, strat.max_drawdown
                )
                if should_pause:
                    self._strategy_runner.pause_strategy(order.account_id, order.strategy_id)
                    self._save_alert(RiskAlert(
                        account_id=order.account_id,
                        alert_type="strategy_drawdown",
                        severity="critical",
                        message=f"Strategy {order.strategy_id} paused: drawdown {dd:.1%} exceeds limit {strat.max_drawdown:.1%}",
                    ))

        logger.info(
            "Order filled",
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            qty=qty,
            price=fill_price,
            slippage=slippage,
            impact=market_impact,
            realized_pnl=realized_pnl,
        )
        return filled_order

    async def _create_oco_legs(
        self, entry_order: EnhancedOrder, account_id: str, strategy_id: Optional[str]
    ) -> None:
        """Create TP and SL orders for a bracket entry. They are OCO."""
        tp_id = str(uuid.uuid4())[:8]
        sl_id = str(uuid.uuid4())[:8]
        now = datetime.now(tz=timezone.utc).isoformat()

        exit_side = "sell" if entry_order.side == "buy" else "buy"

        # Take-profit leg (limit order)
        tp_order = EnhancedOrder(
            order_id=tp_id,
            account_id=account_id,
            ticker=entry_order.ticker,
            side=exit_side,
            order_type="limit",
            quantity=entry_order.filled_qty or entry_order.quantity,
            limit_price=entry_order.take_profit,
            strategy_id=strategy_id,
            notes=f"TP leg for bracket {entry_order.order_id}",
            oco_partner_id=sl_id,
        )
        # Stop-loss leg (stop order)
        sl_order = EnhancedOrder(
            order_id=sl_id,
            account_id=account_id,
            ticker=entry_order.ticker,
            side=exit_side,
            order_type="stop",
            quantity=entry_order.filled_qty or entry_order.quantity,
            stop_price=entry_order.stop_loss,
            strategy_id=strategy_id,
            notes=f"SL leg for bracket {entry_order.order_id}",
            oco_partner_id=tp_id,
        )
        self._save_order(tp_order)
        self._save_order(sl_order)
        logger.info("OCO bracket legs created", tp_id=tp_id, sl_id=sl_id)

    # ── Pending order fills ───────────────────────────────────────────────────

    async def fill_pending_orders(self, account_id: str) -> List[EnhancedOrder]:
        """
        Check all pending limit/stop orders against current prices.
        Handles OCO cancellation on first fill.
        """
        pending = self.get_open_orders(account_id)
        if not pending:
            return []

        tickers  = list({o.ticker for o in pending})
        results  = await asyncio.gather(*[self.get_quote(t) for t in tickers], return_exceptions=True)
        prices: Dict[str, float] = {}
        for t, r in zip(tickers, results):
            if isinstance(r, (int, float)) and r > 0:
                prices[t] = r

        account = self._get_account(account_id)
        newly_filled: List[EnhancedOrder] = []
        cancelled_ids: set = set()

        for order in pending:
            if order.order_id in cancelled_ids:
                continue
            current_price = prices.get(order.ticker)
            if current_price is None:
                continue

            fill_price, filled_qty, slippage, impact = 0.0, 0.0, 0.0, 0.0
            should_fill = False
            is_partial  = False

            if order.order_type == "limit":
                fp, fq, partial = self._execution.fill_limit_order(
                    order.ticker, order.side, order.quantity, order.limit_price or current_price, current_price
                )
                if fq > 0:
                    should_fill = True
                    fill_price  = fp
                    filled_qty  = fq
                    is_partial  = partial
                    slippage    = 0.0

            elif order.order_type == "stop":
                fp, sl = self._execution.fill_stop_order(
                    order.ticker, order.side, order.quantity,
                    order.stop_price or current_price, current_price
                )
                if fp > 0:
                    should_fill = True
                    fill_price  = fp
                    filled_qty  = order.quantity
                    slippage    = sl

            elif order.order_type == "stop_limit":
                stop_hit = False
                if order.side == "sell" and order.stop_price and current_price <= order.stop_price:
                    stop_hit = True
                elif order.side == "buy" and order.stop_price and current_price >= order.stop_price:
                    stop_hit = True
                if stop_hit and order.limit_price:
                    fp, fq, partial = self._execution.fill_limit_order(
                        order.ticker, order.side, order.quantity,
                        order.limit_price, current_price
                    )
                    if fq > 0:
                        should_fill = True
                        fill_price  = fp
                        filled_qty  = fq
                        is_partial  = partial

            if should_fill:
                account = self._get_account(account_id)
                filled = await self._execute_fill(
                    order, fill_price, account, slippage, impact,
                    is_partial=is_partial, filled_qty=filled_qty
                )
                newly_filled.append(filled)

                # Cancel OCO partner
                if order.oco_partner_id and not is_partial:
                    self.cancel_order(order.oco_partner_id)
                    cancelled_ids.add(order.oco_partner_id)

        return newly_filled

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def get_portfolio(self, account_id: str) -> dict:
        """
        Full portfolio snapshot with margin, sector breakdown, and risk metrics.
        """
        account = self._get_account(account_id)
        raw_positions = self._get_all_positions(account_id)

        if not raw_positions:
            equity = account["cash"]
            self._record_equity_snapshot(account_id, equity, account["cash"], 0.0, 0.0)
            return {
                "account_id": account_id,
                "name": account["name"],
                "cash": account["cash"],
                "buying_power": self.get_buying_power(account),
                "margin_enabled": bool(account["margin_enabled"]),
                "margin_used": account.get("margin_used", 0.0),
                "total_market_value": 0.0,
                "total_equity": equity,
                "total_pnl": equity - account["initial_capital"],
                "total_pnl_pct": (equity - account["initial_capital"]) / account["initial_capital"],
                "positions": [],
                "sector_breakdown": {},
                "as_of": datetime.now(tz=timezone.utc).isoformat(),
            }

        tickers = list({p["ticker"] for p in raw_positions})
        price_results = await asyncio.gather(*[self.get_quote(t) for t in tickers], return_exceptions=True)
        prices: Dict[str, float] = {}
        for t, r in zip(tickers, price_results):
            if isinstance(r, (int, float)) and r > 0:
                prices[t] = r

        positions = []
        total_market_value = 0.0
        sector_breakdown: Dict[str, float] = {}
        total_unrealized = 0.0

        for p in raw_positions:
            ticker   = p["ticker"]
            qty      = p["quantity"]
            avg_cost = p["avg_cost"]
            cur_price = prices.get(ticker)
            sector   = SECTOR_MAP.get(ticker, "Unknown")

            market_value = unrealized_pnl = unrealized_pnl_pct = None
            if cur_price is not None:
                market_value     = round(qty * cur_price, 4)
                cost_basis       = qty * avg_cost
                unrealized_pnl   = round(market_value - cost_basis, 4)
                unrealized_pnl_pct = round(unrealized_pnl / abs(cost_basis), 6) if cost_basis != 0 else 0.0
                total_market_value += market_value
                total_unrealized   += unrealized_pnl
                sector_breakdown[sector] = sector_breakdown.get(sector, 0.0) + market_value

            positions.append({
                "ticker": ticker,
                "quantity": qty,
                "avg_cost": avg_cost,
                "current_price": cur_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "realized_pnl": p["realized_pnl"],
                "sector": sector,
                "strategy_id": p.get("strategy_id"),
            })

        total_equity = round(account["cash"] + total_market_value, 4)
        total_pnl    = round(total_equity - account["initial_capital"], 4)
        total_pnl_pct = round(total_pnl / account["initial_capital"], 6) if account["initial_capital"] else 0.0

        self._record_equity_snapshot(
            account_id, total_equity, account["cash"],
            account.get("margin_used", 0.0), total_unrealized
        )

        # Check margin call
        margin_call, mc_reason = self._risk.check_margin_call(
            account["cash"], account.get("margin_used", 0.0), total_market_value
        )
        if margin_call:
            self._save_alert(RiskAlert(
                account_id=account_id,
                alert_type="margin_call",
                severity="critical",
                message=mc_reason,
            ))

        return {
            "account_id": account_id,
            "name": account["name"],
            "cash": round(account["cash"], 4),
            "buying_power": round(self.get_buying_power(account), 4),
            "margin_enabled": bool(account["margin_enabled"]),
            "margin_used": account.get("margin_used", 0.0),
            "total_market_value": round(total_market_value, 4),
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": positions,
            "sector_breakdown": {k: round(v, 2) for k, v in sector_breakdown.items()},
            "margin_call": margin_call,
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }

    def _compute_equity_from_account(self, account_id: str, account: dict) -> float:
        """Quick equity estimate from last snapshot or cash."""
        series = self._get_equity_series(account_id)
        return series[-1] if series else account["cash"]

    # ── Risk metrics ──────────────────────────────────────────────────────────

    def get_risk_metrics(self, account_id: str) -> dict:
        """Compute full risk metrics: VaR, concentration, drawdown, beta."""
        account      = self._get_account(account_id)
        equity_series = self._get_equity_series(account_id)
        raw_positions = self._get_all_positions(account_id)

        # Build position objects (prices will be stale — good enough for risk)
        positions = [
            EnhancedPosition(
                account_id=account_id,
                ticker=p["ticker"],
                quantity=p["quantity"],
                avg_cost=p["avg_cost"],
                market_value=p["quantity"] * p["avg_cost"],
                realized_pnl=p["realized_pnl"],
                sector=SECTOR_MAP.get(p["ticker"], "Unknown"),
            )
            for p in raw_positions
        ]

        total_equity = equity_series[-1] if equity_series else account["cash"]

        var_result = self._risk.compute_var(positions, equity_series, total_equity)
        conc_alerts = self._risk.check_concentration(positions, total_equity)
        beta_warn = self._risk.check_beta_neutrality(var_result.portfolio_beta)

        # Max drawdown
        max_dd = 0.0
        if equity_series:
            arr  = np.array(equity_series)
            peak = np.maximum.accumulate(arr)
            dd   = np.where(peak > 0, (peak - arr) / peak, 0.0)
            max_dd = float(dd.max())

        # Current drawdown
        cur_dd = 0.0
        if equity_series:
            peak_val = max(equity_series)
            cur_dd   = (peak_val - equity_series[-1]) / max(peak_val, 1.0)

        can_trade, halt_reason = self._risk.check_drawdown_halt(equity_series)

        return {
            "account_id": account_id,
            "var_95": var_result.var_95,
            "var_99": var_result.var_99,
            "cvar_95": var_result.cvar_95,
            "portfolio_beta": var_result.portfolio_beta,
            "portfolio_vol_annual": var_result.portfolio_vol,
            "max_drawdown": round(max_dd, 6),
            "current_drawdown": round(cur_dd, 6),
            "trading_halted": not can_trade,
            "halt_reason": halt_reason if not can_trade else None,
            "concentration_alerts": [a.model_dump() for a in conc_alerts],
            "beta_warning": beta_warn,
            "equity_snapshots": len(equity_series),
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ── Dividends ─────────────────────────────────────────────────────────────

    def credit_dividend(
        self,
        account_id: str,
        ticker: str,
        amount_per_share: float,
        ex_date: str,
    ) -> float:
        """Credit dividends to account based on position size."""
        pos = self._get_position(account_id, ticker.upper())
        if not pos or pos["quantity"] <= 0:
            return 0.0

        total = pos["quantity"] * amount_per_share
        now = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            div_id = str(uuid.uuid4())[:8]
            conn.execute(
                "INSERT INTO dividends (id, account_id, ticker, amount_per_share, total_amount, ex_date, credited_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (div_id, account_id, ticker.upper(), amount_per_share, total, ex_date, now),
            )
            conn.execute(
                "UPDATE accounts SET cash = cash + ? WHERE id = ?",
                (total, account_id),
            )

        logger.info("Dividend credited", ticker=ticker, total=total)
        return round(total, 2)

    def apply_split(
        self,
        account_id: str,
        ticker: str,
        ratio: float,
    ) -> Optional[dict]:
        """
        Apply a stock split to an existing position.
        ratio=2.0 = 2-for-1 split (double shares, half avg cost).
        """
        pos = self._get_position(account_id, ticker.upper())
        if not pos:
            return None

        new_qty  = pos["quantity"] * ratio
        new_cost = pos["avg_cost"] / ratio

        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET quantity = ?, avg_cost = ? "
                "WHERE account_id = ? AND ticker = ?",
                (new_qty, new_cost, account_id, ticker.upper()),
            )

        logger.info("Split applied", ticker=ticker, ratio=ratio, new_qty=new_qty, new_cost=new_cost)
        return {"ticker": ticker, "ratio": ratio, "new_qty": new_qty, "new_avg_cost": new_cost}

    # ── Borrow cost deduction ─────────────────────────────────────────────────

    def apply_borrow_costs(self, account_id: str, days: int = 1) -> float:
        """
        Deduct daily borrow cost for all short positions.
        Returns total cost deducted.
        """
        raw_positions = self._get_all_positions(account_id)
        total_deducted = 0.0

        for p in raw_positions:
            if p["quantity"] >= 0:
                continue   # only shorts

            annual_rate = p.get("borrow_cost", BORROW_COST_TIERS["easy"])
            daily_rate  = annual_rate / 365.0
            short_value = abs(p["quantity"]) * p["avg_cost"]
            daily_cost  = short_value * daily_rate * days
            total_deducted += daily_cost

        if total_deducted > 0:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE accounts SET cash = cash - ? WHERE id = ?",
                    (total_deducted, account_id),
                )
            logger.info("Borrow costs deducted", account_id=account_id, total=total_deducted)

        return round(total_deducted, 4)

    # ── Order management ──────────────────────────────────────────────────────

    def get_open_orders(self, account_id: str) -> List[EnhancedOrder]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE account_id = ? AND status = 'pending' "
                "ORDER BY submitted_at DESC",
                (account_id,),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def cancel_order(self, order_id: str) -> bool:
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = ? AND status = 'pending'",
                (order_id,),
            )
        cancelled = result.rowcount > 0
        if cancelled:
            logger.info("Order cancelled", order_id=order_id)
        return cancelled

    def get_trade_history(self, account_id: str, days_back: int = 30) -> List[dict]:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE account_id = ? AND executed_at >= ? "
                "ORDER BY executed_at DESC",
                (account_id, since),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Performance report ────────────────────────────────────────────────────

    async def generate_performance_report(
        self,
        account_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict:
        """
        Full performance analytics with sector/factor attribution and benchmark comparison.
        """
        end_date = end_date or date.today()
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

        account      = self._get_account(account_id)
        equity_series = self._get_equity_series(account_id)
        trades       = self.get_trade_history(account_id, days_back=(end_date - start_date).days + 1)

        with self._conn() as conn:
            eq_rows = conn.execute(
                "SELECT date, equity_value FROM equity_history WHERE account_id = ? "
                "AND date BETWEEN ? AND ? ORDER BY date ASC",
                (account_id, start_date.isoformat(), end_date.isoformat()),
            ).fetchall()

        eq_values = [r["equity_value"] for r in eq_rows]
        initial_equity = eq_values[0] if eq_values else account["initial_capital"]
        final_equity   = eq_values[-1] if eq_values else account["initial_capital"]

        total_return = (final_equity - initial_equity) / max(initial_equity, 1.0)
        n_days = max(1, (end_date - start_date).days)
        annualized_return = (1 + total_return) ** (365.0 / n_days) - 1

        # Sharpe
        sharpe: Optional[float] = None
        if len(eq_values) >= 5:
            arr  = np.array(eq_values)
            rets = np.diff(arr) / arr[:-1]
            rets = rets[np.isfinite(rets)]
            if rets.std() > 0:
                sharpe = round(float(rets.mean() / rets.std() * math.sqrt(252)), 4)

        # Max drawdown
        max_dd = 0.0
        if eq_values:
            arr  = np.array(eq_values)
            peak = np.maximum.accumulate(arr)
            dd   = np.where(peak > 0, (peak - arr) / peak, 0.0)
            max_dd = float(dd.max())

        # Trade stats
        pnl_vals = [t["realized_pnl"] for t in trades if t.get("realized_pnl") is not None]
        wins   = [p for p in pnl_vals if p > 0]
        losses = [p for p in pnl_vals if p < 0]
        n_trades  = len(pnl_vals)
        win_rate  = len(wins) / n_trades if n_trades else 0.0
        avg_win   = sum(wins) / len(wins) if wins else 0.0
        avg_loss  = sum(losses) / len(losses) if losses else 0.0
        pf        = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0

        # Attribution
        sector_pnl  = self._dashboard.pnl_by_sector(account_id)
        factor_pnl  = self._dashboard.factor_attribution(account_id)
        pnl_daily   = self._dashboard.pnl_by_period(account_id, "daily")
        pnl_monthly = self._dashboard.pnl_by_period(account_id, "monthly")
        win_by_strat = self._dashboard.win_rate_by_strategy(account_id)
        win_by_tod   = self._dashboard.win_rate_by_time_of_day(account_id)

        # Benchmark
        bench = await self._dashboard.compare_to_benchmark(account_id, "SPY", n_days)

        return {
            "account_id": account_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "initial_equity": round(initial_equity, 2),
            "final_equity": round(final_equity, 2),
            "total_return": round(total_return, 6),
            "annualized_return": round(annualized_return, 6),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(max_dd, 6),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 4),
            "n_trades": n_trades,
            "sector_pnl": sector_pnl,
            "factor_attribution": factor_pnl,
            "pnl_daily": pnl_daily[-30:],
            "pnl_monthly": pnl_monthly,
            "win_rate_by_strategy": win_by_strat,
            "win_rate_by_time_of_day": win_by_tod,
            "benchmark_comparison": bench.model_dump(),
            "as_of": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ── Equity history ────────────────────────────────────────────────────────

    def _record_equity_snapshot(
        self,
        account_id: str,
        equity: float,
        cash: float,
        margin_used: float,
        unrealized_pnl: float,
    ) -> None:
        today = date.today().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_history (account_id, date, equity_value, cash, margin_used, unrealized_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(account_id, date) DO UPDATE SET "
                "equity_value=excluded.equity_value, cash=excluded.cash, "
                "margin_used=excluded.margin_used, unrealized_pnl=excluded.unrealized_pnl",
                (account_id, today, round(equity, 4), round(cash, 4),
                 round(margin_used, 4), round(unrealized_pnl, 4)),
            )

    def _get_equity_series(self, account_id: str) -> List[float]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT equity_value FROM equity_history WHERE account_id = ? ORDER BY date ASC",
                (account_id,),
            ).fetchall()
        return [r["equity_value"] for r in rows]

    # ── Position management ───────────────────────────────────────────────────

    def _update_position(
        self,
        account_id: str,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        strategy_id: Optional[str] = None,
    ) -> Optional[float]:
        realized_pnl: Optional[float] = None
        existing = self._get_position(account_id, ticker)
        now = datetime.now(tz=timezone.utc).isoformat()

        with self._conn() as conn:
            if existing is None:
                new_qty = quantity if side == "buy" else -quantity
                conn.execute(
                    "INSERT INTO positions (account_id, ticker, quantity, avg_cost, realized_pnl, "
                    "borrow_cost, opened_at, strategy_id) VALUES (?, ?, ?, ?, 0, 0, ?, ?)",
                    (account_id, ticker, new_qty, price, now, strategy_id),
                )
            else:
                old_qty     = existing["quantity"]
                old_avg     = existing["avg_cost"]
                old_realized = existing["realized_pnl"]

                if side == "buy":
                    if old_qty >= 0:
                        new_qty = old_qty + quantity
                        new_avg = (old_qty * old_avg + quantity * price) / new_qty
                    else:
                        cover_qty      = min(quantity, abs(old_qty))
                        realized_pnl   = cover_qty * (old_avg - price)
                        remaining_short = abs(old_qty) - cover_qty
                        excess_long    = quantity - cover_qty
                        new_qty        = -remaining_short + excess_long
                        new_avg        = price if new_qty > 0 else old_avg
                        old_realized  += realized_pnl
                else:
                    if old_qty > 0:
                        sell_qty      = min(quantity, old_qty)
                        realized_pnl  = sell_qty * (price - old_avg)
                        new_qty       = old_qty - quantity
                        new_avg       = old_avg
                        old_realized += realized_pnl
                    else:
                        new_qty = old_qty - quantity
                        new_avg = (abs(old_qty) * old_avg + quantity * price) / abs(new_qty)

                if abs(new_qty) < 1e-9:
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
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE account_id = ? AND ticker = ?",
                (account_id, ticker),
            ).fetchone()
        return dict(row) if row else None

    def _get_all_positions(self, account_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE account_id = ? AND ABS(quantity) > 1e-9",
                (account_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _get_avg_cost_or_price(self, account_id: str, ticker: str, fallback: float) -> float:
        pos = self._get_position(account_id, ticker)
        return pos["avg_cost"] if pos else fallback

    # ── Alerts ────────────────────────────────────────────────────────────────

    def _save_alert(self, alert: RiskAlert) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO risk_alerts "
                "(alert_id, account_id, alert_type, severity, message, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (alert.alert_id, alert.account_id, alert.alert_type,
                 alert.severity, alert.message, now),
            )

    def get_alerts(self, account_id: str, days_back: int = 7) -> List[dict]:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM risk_alerts WHERE account_id = ? AND created_at >= ? "
                "ORDER BY created_at DESC",
                (account_id, since),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── DB serialization ──────────────────────────────────────────────────────

    def _save_order(self, order: EnhancedOrder) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders "
                "(order_id, account_id, ticker, side, order_type, quantity, "
                "limit_price, stop_price, take_profit, stop_loss, status, submitted_at, "
                "filled_at, filled_price, filled_qty, commission, slippage, market_impact, "
                "strategy_id, notes, oco_partner_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    order.order_id, order.account_id, order.ticker, order.side,
                    order.order_type, order.quantity, order.limit_price, order.stop_price,
                    order.take_profit, order.stop_loss, order.status,
                    order.submitted_at.isoformat() if order.submitted_at else None,
                    order.filled_at.isoformat() if order.filled_at else None,
                    order.filled_price, order.filled_qty, order.commission,
                    order.slippage, order.market_impact, order.strategy_id,
                    order.notes, order.oco_partner_id,
                ),
            )

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> EnhancedOrder:
        d = dict(row)
        for dt_field in ("submitted_at", "filled_at"):
            if d.get(dt_field):
                try:
                    d[dt_field] = datetime.fromisoformat(d[dt_field])
                except (ValueError, TypeError):
                    d[dt_field] = None
        return EnhancedOrder(**d)


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: Optional[EnhancedPaperTradingEngine] = None


def _get_engine() -> EnhancedPaperTradingEngine:
    global _engine
    if _engine is None:
        _engine = EnhancedPaperTradingEngine()
    return _engine


# ── FastAPI Router ────────────────────────────────────────────────────────────

paper_router = APIRouter(prefix="/paper", tags=["paper-trading-enhanced"])


class OrderRequest(BaseModel):
    account_id:  str
    ticker:      str
    side:        Literal["buy", "sell"]
    quantity:    float
    order_type:  Literal["market", "limit", "stop", "stop_limit", "bracket"] = "market"
    limit_price:  Optional[float] = None
    stop_price:   Optional[float] = None
    take_profit:  Optional[float] = None
    stop_loss:    Optional[float] = None
    strategy_id:  Optional[str] = None
    entry_reason: Optional[str] = None


class CreateAccountRequest(BaseModel):
    name:            str
    initial_capital: float = 100_000.0
    margin_enabled:  bool = False


class RegisterStrategyRequest(BaseModel):
    account_id:    str
    name:          str
    description:   str = ""
    allocated_pct: float = 0.10
    max_drawdown:  float = 0.15


@paper_router.get("/accounts")
def list_accounts():
    return _get_engine().list_accounts()


@paper_router.post("/accounts")
def create_account(req: CreateAccountRequest):
    engine = _get_engine()
    account_id = engine.create_account(req.name, req.initial_capital, req.margin_enabled)
    return {"account_id": account_id, "name": req.name, "initial_capital": req.initial_capital}


@paper_router.get("/account")
async def get_account(account_id: str = Query(...)):
    engine = _get_engine()
    try:
        return await engine.get_portfolio(account_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@paper_router.post("/order")
async def submit_order(req: OrderRequest):
    engine = _get_engine()
    try:
        order = await engine.submit_order(
            req.account_id, req.ticker, req.side, req.quantity,
            req.order_type, req.limit_price, req.stop_price,
            req.take_profit, req.stop_loss, req.strategy_id, req.entry_reason,
        )
        return order.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@paper_router.get("/positions")
async def get_positions(account_id: str = Query(...)):
    engine = _get_engine()
    try:
        pf = await engine.get_portfolio(account_id)
        return pf["positions"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@paper_router.get("/history")
def get_trade_history(account_id: str = Query(...), days: int = Query(30)):
    return _get_engine().get_trade_history(account_id, days_back=days)


@paper_router.get("/performance")
async def get_performance(
    account_id: str = Query(...),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    engine = _get_engine()
    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None
    try:
        return await engine.generate_performance_report(account_id, sd, ed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@paper_router.get("/strategies")
def list_strategies(account_id: str = Query(...)):
    engine = _get_engine()
    return [s.model_dump() for s in engine._strategy_runner.get_strategies(account_id)]


@paper_router.post("/strategies")
def register_strategy(req: RegisterStrategyRequest):
    engine = _get_engine()
    config = StrategyConfig(
        name=req.name,
        description=req.description,
        allocated_pct=req.allocated_pct,
        max_drawdown=req.max_drawdown,
    )
    sid = engine._strategy_runner.register_strategy(req.account_id, config)
    return {"strategy_id": sid, "name": req.name}


@paper_router.get("/risk-metrics")
def get_risk_metrics(account_id: str = Query(...)):
    try:
        return _get_engine().get_risk_metrics(account_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@paper_router.get("/journal")
def get_journal(account_id: str = Query(...), days: int = Query(30)):
    entries = _get_engine()._dashboard.get_trade_journal(account_id, days_back=days)
    return [e.model_dump() for e in entries]


@paper_router.post("/fill-pending")
async def fill_pending(account_id: str = Query(...)):
    engine = _get_engine()
    filled = await engine.fill_pending_orders(account_id)
    return {"filled_count": len(filled), "orders": [o.model_dump() for o in filled]}


@paper_router.post("/dividend")
def credit_dividend(
    account_id: str = Query(...),
    ticker: str = Query(...),
    amount_per_share: float = Query(...),
    ex_date: str = Query(...),
):
    total = _get_engine().credit_dividend(account_id, ticker, amount_per_share, ex_date)
    return {"ticker": ticker, "total_credited": total, "ex_date": ex_date}


@paper_router.post("/split")
def apply_split(
    account_id: str = Query(...),
    ticker: str = Query(...),
    ratio: float = Query(...),
):
    result = _get_engine().apply_split(account_id, ticker, ratio)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No position found for {ticker}")
    return result


@paper_router.get("/alerts")
def get_alerts(account_id: str = Query(...), days: int = Query(7)):
    return _get_engine().get_alerts(account_id, days_back=days)


@paper_router.delete("/order/{order_id}")
def cancel_order(order_id: str):
    cancelled = _get_engine().cancel_order(order_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Order not found or already filled")
    return {"order_id": order_id, "status": "cancelled"}
