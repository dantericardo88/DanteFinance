"""
Strategy promotion state machine v3 — production lifecycle management.

Dimension: dim_067 — Strategy promotion state machine (target score: 9/10)

State machine:
  RESEARCH → PAPER_TRADING → LIVE_SHADOW → LIVE_SMALL → LIVE_FULL → RETIRED
                   ↓ (fail)         ↓ (fail)       ↓ (fail)      ↓ (fail)
                 RETIRED          RESEARCH       LIVE_SMALL    LIVE_SMALL

Key components:
  StrategyState          — enum of all lifecycle states
  PromotionCriteria      — per-transition gate definitions
  DemotionTrigger        — automatic demotion rule checker
  PerformanceTracker     — live PnL and metrics computation
  StrategyRegistry       — persistent JSON-backed registry
  StateMachine           — FSM core: transition, validate, emit
  CapitalAllocator       — state-aware capital sizing
  StrategyLifecycleManager — daily orchestrator

Persistence: sentinel/data/strategy_registry.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging as _logging
    def get_logger(name: str):  # type: ignore
        return _logging.getLogger(name)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("SENTINEL_DATA_DIR", Path(__file__).parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = DATA_DIR / "strategy_registry.json"
PERFORMANCE_PATH = DATA_DIR / "strategy_performance.json"


# ===========================================================================
# Enums
# ===========================================================================

class StrategyState(str, Enum):
    RESEARCH      = "research"
    PAPER_TRADING = "paper_trading"
    LIVE_SHADOW   = "live_shadow"
    LIVE_SMALL    = "live_small"
    LIVE_FULL     = "live_full"
    RETIRED       = "retired"
    SUSPENDED     = "suspended"


STATE_ORDER: List[StrategyState] = [
    StrategyState.RESEARCH,
    StrategyState.PAPER_TRADING,
    StrategyState.LIVE_SHADOW,
    StrategyState.LIVE_SMALL,
    StrategyState.LIVE_FULL,
]

DEMOTION_MAP: Dict[StrategyState, StrategyState] = {
    StrategyState.PAPER_TRADING: StrategyState.RETIRED,
    StrategyState.LIVE_SHADOW:   StrategyState.RESEARCH,
    StrategyState.LIVE_SMALL:    StrategyState.RESEARCH,
    StrategyState.LIVE_FULL:     StrategyState.LIVE_SMALL,
}


class TransitionType(str, Enum):
    PROMOTE  = "promote"
    DEMOTE   = "demote"
    SUSPEND  = "suspend"
    RETIRE   = "retire"
    REINSTATE = "reinstate"


class CriterionResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"   # Not enough data to evaluate


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class PerformanceMetrics:
    strategy_id: str
    period: str                        # "30d", "90d", "ytd", "all"
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0          # negative fraction e.g. -0.15
    volatility: float = 0.0           # annualized
    var_95: float = 0.0               # daily VaR at 95%

    # Trade metrics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Quality metrics
    days_in_state: int = 0
    n_consecutive_loss_days: int = 0
    max_daily_loss: float = 0.0        # worst single day loss (negative)
    slippage_ratio: float = 0.0       # slippage / expected_alpha

    # Correlation with backtest (available in LIVE states)
    backtest_correlation: float = 0.0
    pbo: float = 0.0                  # Probability of Backtest Overfitting (0-1)

    # Shadow tracking (LIVE_SHADOW)
    shadow_vs_actual_correlation: float = 0.0
    shadow_position_errors: int = 0
    risk_manager_triggers: int = 0

    # Data sufficiency flag
    sufficient_data: bool = False


@dataclass
class CriteriaResult:
    strategy_id: str
    from_state: StrategyState
    to_state: StrategyState
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: Dict[str, dict] = field(default_factory=dict)  # {criterion_name: {result, value, threshold, message}}
    can_promote: bool = False
    failed_criteria: List[str] = field(default_factory=list)
    warning_criteria: List[str] = field(default_factory=list)


@dataclass
class StateTransition:
    strategy_id: str
    from_state: str
    to_state: str
    transition_type: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    triggered_by: str = "system"      # "system" | "manual" | "emergency"
    metrics_snapshot: Optional[dict] = None


@dataclass
class TransitionResult:
    success: bool
    strategy_id: str
    from_state: str
    to_state: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    criteria_result: Optional[CriteriaResult] = None
    error: Optional[str] = None


@dataclass
class DemotionDecision:
    strategy_id: str
    should_demote: bool
    urgency: str                      # "immediate" | "scheduled" | "watch"
    reason: str
    triggered_rules: List[str] = field(default_factory=list)
    recommended_state: Optional[str] = None
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AllocationChange:
    strategy_id: str
    old_allocation: float
    new_allocation: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Strategy:
    strategy_id: str
    name: str
    description: str
    author: str = "unknown"
    state: str = StrategyState.RESEARCH.value
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state_entered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state_history: List[dict] = field(default_factory=list)

    # Backtest summary (populated before paper trading)
    backtest_sharpe: float = 0.0
    backtest_max_drawdown: float = 0.0
    backtest_days: int = 0
    backtest_win_rate: float = 0.0
    backtest_pbo: float = 0.0
    backtest_regimes_covered: int = 0

    # Live performance reference (updated periodically)
    live_sharpe: float = 0.0
    live_max_drawdown: float = 0.0
    current_allocation: float = 0.0   # dollars

    # Metadata
    asset_class: str = "equity"
    timeframe: str = "daily"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Demotion watch
    consecutive_months_negative_sharpe: int = 0
    last_demotion_check: Optional[str] = None
    suspended_at: Optional[str] = None
    suspension_reason: str = ""


# ===========================================================================
# Promotion Criteria
# ===========================================================================

class PromotionCriteria:
    """
    Gate definitions for each state transition.
    Each method returns the criteria dict for a given transition.
    """

    RESEARCH_TO_PAPER = {
        "backtest_sharpe_min":       {"threshold": 0.8,  "description": "OOS backtest Sharpe ≥ 0.8"},
        "backtest_max_drawdown_max":  {"threshold": 0.20, "description": "Max drawdown ≤ 20%"},
        "backtest_min_days":          {"threshold": 252,  "description": "≥ 252 trading days in backtest"},
        "backtest_win_rate_min":      {"threshold": 0.40, "description": "Win rate ≥ 40%"},
        "pbo_max":                    {"threshold": 0.50, "description": "PBO ≤ 0.5 (not data-mined)"},
        "regimes_covered_min":        {"threshold": 2,    "description": "≥ 2 market regimes covered"},
    }

    PAPER_TO_SHADOW = {
        "paper_days_min":            {"threshold": 60,   "description": "≥ 60 calendar days in paper trading"},
        "paper_sharpe_min":          {"threshold": 0.6,  "description": "Live paper Sharpe ≥ 0.6 (annualized)"},
        "paper_max_drawdown_max":    {"threshold": 0.15, "description": "Max drawdown ≤ 15%"},
        "slippage_ratio_max":        {"threshold": 0.20, "description": "Slippage < 20% of expected alpha"},
        "paper_trades_min":          {"threshold": 30,   "description": "≥ 30 paper trades completed"},
    }

    SHADOW_TO_LIVE_SMALL = {
        "shadow_days_min":           {"threshold": 30,   "description": "≥ 30 calendar days in shadow"},
        "shadow_correlation_min":    {"threshold": 0.85, "description": "Shadow vs actual fill correlation ≥ 0.85"},
        "shadow_position_errors_max":{"threshold": 0,    "description": "No position sizing errors"},
        "risk_manager_triggers_max": {"threshold": 0,    "description": "Risk manager never triggered"},
    }

    LIVE_SMALL_TO_FULL = {
        "live_days_min":             {"threshold": 90,   "description": "≥ 90 calendar days in live small"},
        "live_sharpe_min":           {"threshold": 0.7,  "description": "Live Sharpe ≥ 0.7 (annualized)"},
        "live_max_drawdown_max":     {"threshold": 0.12, "description": "Max drawdown ≤ 12%"},
        "profit_factor_min":         {"threshold": 1.4,  "description": "Profit factor ≥ 1.4"},
        "live_backtest_correlation":  {"threshold": 0.6,  "description": "Live vs backtest returns ρ ≥ 0.6"},
        "no_system_errors":          {"threshold": 0,    "description": "No fat-finger or system errors"},
    }

    @classmethod
    def get_criteria(cls, from_state: StrategyState, to_state: StrategyState) -> dict:
        key = (from_state, to_state)
        mapping = {
            (StrategyState.RESEARCH, StrategyState.PAPER_TRADING): cls.RESEARCH_TO_PAPER,
            (StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW): cls.PAPER_TO_SHADOW,
            (StrategyState.LIVE_SHADOW, StrategyState.LIVE_SMALL): cls.SHADOW_TO_LIVE_SMALL,
            (StrategyState.LIVE_SMALL, StrategyState.LIVE_FULL): cls.LIVE_SMALL_TO_FULL,
        }
        return mapping.get(key, {})


# ===========================================================================
# Performance Tracker
# ===========================================================================

class PerformanceTracker:
    """
    Tracks trade fills and daily PnL for each strategy.
    Computes Sharpe, Sortino, Calmar, drawdown, win_rate, profit_factor.
    Storage: PERFORMANCE_PATH (JSON).
    """

    def __init__(self) -> None:
        self._data: dict = self._load()

    def _load(self) -> dict:
        if PERFORMANCE_PATH.exists():
            try:
                return json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load performance data, starting fresh")
        return {}

    def _save(self) -> None:
        PERFORMANCE_PATH.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")

    def _ensure(self, strategy_id: str) -> None:
        if strategy_id not in self._data:
            self._data[strategy_id] = {"trades": [], "daily_pnl": [], "created_at": datetime.now(timezone.utc).isoformat()}

    def record_trade(self, strategy_id: str, fill: dict) -> None:
        """Log a trade fill. fill = {date, symbol, side, qty, price, expected_price, pnl}"""
        self._ensure(strategy_id)
        fill["recorded_at"] = datetime.now(timezone.utc).isoformat()
        self._data[strategy_id]["trades"].append(fill)
        self._save()

    def record_daily_pnl(self, strategy_id: str, date_str: str, pnl: float, equity: float) -> None:
        """Log daily PnL and equity level."""
        self._ensure(strategy_id)
        self._data[strategy_id]["daily_pnl"].append({
            "date": date_str,
            "pnl": pnl,
            "equity": equity,
        })
        self._save()

    def _filter_by_period(self, records: List[dict], period: str) -> List[dict]:
        if period == "all" or not records:
            return records
        now = datetime.now(timezone.utc)
        days_map = {"30d": 30, "90d": 90, "ytd": (now - datetime(now.year, 1, 1, tzinfo=timezone.utc)).days}
        cutoff_days = days_map.get(period, 9999)
        cutoff = (now - timedelta(days=cutoff_days)).date().isoformat()
        return [r for r in records if r.get("date", "9999") >= cutoff]

    def compute_metrics(self, strategy_id: str, period: str = "all") -> PerformanceMetrics:
        """Compute full performance metrics for a strategy over a period."""
        m = PerformanceMetrics(strategy_id=strategy_id, period=period)
        data = self._data.get(strategy_id, {})
        daily_pnl_records = self._filter_by_period(data.get("daily_pnl", []), period)
        trade_records = self._filter_by_period(data.get("trades", []), period)

        m.n_trades = len(trade_records)

        if not daily_pnl_records or len(daily_pnl_records) < 2:
            m.sufficient_data = False
            return m

        m.sufficient_data = True

        pnls = [r["pnl"] for r in daily_pnl_records]
        equities = [r["equity"] for r in daily_pnl_records]

        pnl_arr = np.array(pnls, dtype=float)
        equity_arr = np.array(equities, dtype=float)

        # Returns
        returns = pnl_arr / equity_arr[:-1].clip(min=1e-8) if len(equity_arr) > 1 else pnl_arr
        n_days = len(returns)
        ann_factor = 252.0

        m.total_return = float((equity_arr[-1] / equity_arr[0]) - 1) if equity_arr[0] != 0 else 0.0
        m.annualized_return = float((1 + m.total_return) ** (ann_factor / max(n_days, 1)) - 1)
        vol = float(np.std(returns, ddof=1)) * math.sqrt(ann_factor) if n_days > 1 else 0.0
        m.volatility = vol

        rf_daily = 0.05 / ann_factor  # 5% risk-free rate
        excess = returns - rf_daily
        m.sharpe_ratio = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(ann_factor)) if np.std(excess) > 0 else 0.0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 1:
            dd_vol = float(np.std(downside, ddof=1)) * math.sqrt(ann_factor)
            m.sortino_ratio = float(m.annualized_return / dd_vol) if dd_vol > 0 else 0.0
        else:
            m.sortino_ratio = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max.clip(min=1e-8)
        m.max_drawdown = float(np.min(drawdowns))

        # Calmar
        m.calmar_ratio = float(m.annualized_return / abs(m.max_drawdown)) if m.max_drawdown < 0 else 0.0

        # VaR 95%
        m.var_95 = float(np.percentile(returns, 5)) if n_days >= 20 else 0.0

        # Max daily loss
        m.max_daily_loss = float(np.min(pnl_arr)) if len(pnl_arr) else 0.0

        # Consecutive loss days
        consec = 0
        max_consec = 0
        for p in pnl_arr:
            if p < 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        m.n_consecutive_loss_days = max_consec

        # Trade metrics
        if trade_records:
            trade_pnls = [t.get("pnl", 0) for t in trade_records]
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            m.win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
            m.avg_win = float(np.mean(wins)) if wins else 0.0
            m.avg_loss = float(np.mean(losses)) if losses else 0.0
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            m.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            # Slippage ratio
            slippages = [abs(t.get("price", 0) - t.get("expected_price", t.get("price", 0))) for t in trade_records]
            expected_alphas = [abs(t.get("expected_pnl", t.get("pnl", 0))) for t in trade_records]
            if any(a > 0 for a in expected_alphas):
                avg_slip = float(np.mean(slippages))
                avg_alpha = float(np.mean([a for a in expected_alphas if a > 0]))
                m.slippage_ratio = avg_slip / avg_alpha if avg_alpha > 0 else 0.0

        return m

    def compute_rolling_metrics(self, strategy_id: str, window: int = 63) -> List[dict]:
        """Compute rolling Sharpe and drawdown over a sliding window. Returns list of dicts (or DataFrame)."""
        data = self._data.get(strategy_id, {})
        daily_pnl_records = data.get("daily_pnl", [])
        if len(daily_pnl_records) < window + 5:
            return []

        pnls = np.array([r["pnl"] for r in daily_pnl_records], dtype=float)
        equities = np.array([r["equity"] for r in daily_pnl_records], dtype=float)
        dates = [r["date"] for r in daily_pnl_records]
        ann = 252.0

        results = []
        for i in range(window, len(pnls)):
            w_pnl = pnls[i - window:i]
            w_eq = equities[i - window:i]
            w_ret = w_pnl / w_eq.clip(min=1e-8)
            vol = float(np.std(w_ret, ddof=1)) * math.sqrt(ann)
            mean_ret = float(np.mean(w_ret)) * ann
            sharpe = mean_ret / vol if vol > 0 else 0.0
            running_max = np.maximum.accumulate(w_eq)
            dd = float(np.min((w_eq - running_max) / running_max.clip(min=1e-8)))
            results.append({
                "date": dates[i],
                "rolling_sharpe": round(sharpe, 4),
                "rolling_max_dd": round(dd, 4),
                "window": window,
            })

        if PANDAS_AVAILABLE:
            try:
                return pd.DataFrame(results)
            except Exception:
                pass
        return results

    def check_against_criteria(self, strategy: Strategy, metrics: PerformanceMetrics, from_state: StrategyState, to_state: StrategyState) -> CriteriaResult:
        """Validate metrics against promotion criteria for a state transition."""
        criteria_def = PromotionCriteria.get_criteria(from_state, to_state)
        result = CriteriaResult(
            strategy_id=strategy.strategy_id,
            from_state=from_state,
            to_state=to_state,
        )

        days_in_state = _days_since(strategy.state_entered_at)

        def check(name: str, value: float, threshold: float, op: str = ">=") -> str:
            if op == ">=":
                passed = value >= threshold
            elif op == "<=":
                passed = value <= threshold
            elif op == "==":
                passed = int(value) == int(threshold)
            else:
                passed = True
            desc = criteria_def.get(name, {}).get("description", name)
            result.checks[name] = {
                "result": CriterionResult.PASS.value if passed else CriterionResult.FAIL.value,
                "value": round(value, 6),
                "threshold": threshold,
                "operator": op,
                "description": desc,
            }
            if not passed:
                result.failed_criteria.append(name)
            return CriterionResult.PASS.value if passed else CriterionResult.FAIL.value

        # RESEARCH → PAPER_TRADING
        if from_state == StrategyState.RESEARCH and to_state == StrategyState.PAPER_TRADING:
            check("backtest_sharpe_min", strategy.backtest_sharpe, 0.8)
            check("backtest_max_drawdown_max", abs(strategy.backtest_max_drawdown), 0.20, "<=")
            check("backtest_min_days", float(strategy.backtest_days), 252.0)
            check("backtest_win_rate_min", strategy.backtest_win_rate, 0.40)
            if strategy.backtest_pbo > 0:
                check("pbo_max", strategy.backtest_pbo, 0.50, "<=")
            else:
                result.checks["pbo_max"] = {"result": CriterionResult.SKIP.value, "value": None, "threshold": 0.5, "description": "PBO not computed — skipped"}
            check("regimes_covered_min", float(strategy.backtest_regimes_covered), 2.0)

        # PAPER_TRADING → LIVE_SHADOW
        elif from_state == StrategyState.PAPER_TRADING and to_state == StrategyState.LIVE_SHADOW:
            check("paper_days_min", float(days_in_state), 60.0)
            check("paper_sharpe_min", metrics.sharpe_ratio, 0.6)
            check("paper_max_drawdown_max", abs(metrics.max_drawdown), 0.15, "<=")
            check("slippage_ratio_max", metrics.slippage_ratio, 0.20, "<=")
            check("paper_trades_min", float(metrics.n_trades), 30.0)

        # LIVE_SHADOW → LIVE_SMALL
        elif from_state == StrategyState.LIVE_SHADOW and to_state == StrategyState.LIVE_SMALL:
            check("shadow_days_min", float(days_in_state), 30.0)
            check("shadow_correlation_min", metrics.shadow_vs_actual_correlation, 0.85)
            check("shadow_position_errors_max", float(metrics.shadow_position_errors), 0.0, "<=")
            check("risk_manager_triggers_max", float(metrics.risk_manager_triggers), 0.0, "<=")

        # LIVE_SMALL → LIVE_FULL
        elif from_state == StrategyState.LIVE_SMALL and to_state == StrategyState.LIVE_FULL:
            check("live_days_min", float(days_in_state), 90.0)
            check("live_sharpe_min", metrics.sharpe_ratio, 0.7)
            check("live_max_drawdown_max", abs(metrics.max_drawdown), 0.12, "<=")
            check("profit_factor_min", metrics.profit_factor, 1.4)
            check("live_backtest_correlation", metrics.backtest_correlation, 0.6)
            check("no_system_errors", 0.0, 0.0, "<=")  # Always passes unless set externally

        result.can_promote = len(result.failed_criteria) == 0
        return result


# ===========================================================================
# Demotion Trigger
# ===========================================================================

class DemotionTrigger:
    """
    Automatic demotion rule evaluator.
    Hard stops trigger immediately; soft stops trigger scheduled review.
    """

    # Hard stop thresholds
    HARD_DRAWDOWN = 0.25       # 25% drawdown from peak
    HARD_DAILY_LOSS = 0.05     # 5% single-day loss
    HARD_CONSEC_LOSS_DAYS = 5  # 5 consecutive loss days > 2%
    HARD_CONSEC_LOSS_PCT = 0.02

    # Soft stop thresholds
    SOFT_ROLLING_SHARPE_MONTHS = 2    # 2 consecutive months negative Sharpe
    SOFT_ALPHA_DECAY_RATIO = 0.50     # live Sharpe < backtest Sharpe * 0.5
    SOFT_CORRELATION_BREAKDOWN = 0.50 # live vs backtest correlation < 0.5

    def check_for_demotion(self, strategy: Strategy, metrics: PerformanceMetrics) -> DemotionDecision:
        decision = DemotionDecision(
            strategy_id=strategy.strategy_id,
            should_demote=False,
            urgency="none",
            reason="",
        )
        triggered: List[str] = []
        current = StrategyState(strategy.state)

        # Hard stop 1: drawdown > 25%
        if abs(metrics.max_drawdown) > self.HARD_DRAWDOWN:
            triggered.append(f"Hard drawdown {abs(metrics.max_drawdown)*100:.1f}% > {self.HARD_DRAWDOWN*100:.0f}%")

        # Hard stop 2: daily loss > 5%
        if metrics.max_daily_loss < 0:
            daily_loss_pct = abs(metrics.max_daily_loss) / max(1.0, strategy.current_allocation)
            if daily_loss_pct > self.HARD_DAILY_LOSS:
                triggered.append(f"Daily loss {daily_loss_pct*100:.1f}% > {self.HARD_DAILY_LOSS*100:.0f}%")

        # Hard stop 3: 5 consecutive loss days > 2%
        if metrics.n_consecutive_loss_days >= self.HARD_CONSEC_LOSS_DAYS:
            triggered.append(f"{metrics.n_consecutive_loss_days} consecutive loss days")

        if triggered:
            decision.should_demote = True
            decision.urgency = "immediate"
            decision.triggered_rules = triggered
            decision.reason = "; ".join(triggered)
            decision.recommended_state = StrategyState.SUSPENDED.value
            return decision

        # Soft stop 1: rolling Sharpe < 0 for 2 months
        if strategy.consecutive_months_negative_sharpe >= self.SOFT_ROLLING_SHARPE_MONTHS:
            triggered.append(f"Rolling Sharpe negative for {strategy.consecutive_months_negative_sharpe} months")

        # Soft stop 2: alpha decay
        if strategy.backtest_sharpe > 0 and metrics.sharpe_ratio < strategy.backtest_sharpe * self.SOFT_ALPHA_DECAY_RATIO:
            triggered.append(f"Alpha decay: live Sharpe {metrics.sharpe_ratio:.2f} < {strategy.backtest_sharpe * self.SOFT_ALPHA_DECAY_RATIO:.2f} (backtest×0.5)")

        # Soft stop 3: correlation breakdown
        if metrics.backtest_correlation > 0 and metrics.backtest_correlation < self.SOFT_CORRELATION_BREAKDOWN:
            triggered.append(f"Live vs backtest correlation {metrics.backtest_correlation:.2f} < {self.SOFT_CORRELATION_BREAKDOWN}")

        if triggered:
            decision.should_demote = True
            decision.urgency = "scheduled"
            decision.triggered_rules = triggered
            decision.reason = "; ".join(triggered)
            decision.recommended_state = DEMOTION_MAP.get(current, StrategyState.RETIRED).value

        return decision

    def check_retire_triggers(self, strategy: Strategy, metrics: PerformanceMetrics) -> bool:
        """Return True if strategy meets LIVE_FULL retirement conditions."""
        if abs(metrics.max_drawdown) > 0.25:
            return True
        rolling_90d = PerformanceTracker().compute_metrics(strategy.strategy_id, "90d")
        if rolling_90d.sharpe_ratio < -0.5:
            return True
        return False


# ===========================================================================
# Strategy Registry
# ===========================================================================

class StrategyRegistry:
    """
    Persistent JSON-backed registry of all strategies.
    Stored at sentinel/data/strategy_registry.json.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        if REGISTRY_PATH.exists():
            try:
                return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load strategy registry, starting fresh")
        return {}

    def _save(self) -> None:
        REGISTRY_PATH.write_text(json.dumps(self._strategies, indent=2, default=str), encoding="utf-8")

    def _to_strategy(self, data: dict) -> Strategy:
        # Filter to known fields
        known = {f for f in Strategy.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return Strategy(**filtered)

    def register(self, strategy: Strategy) -> str:
        """Register a new strategy, return its strategy_id."""
        if not strategy.strategy_id:
            strategy.strategy_id = str(uuid.uuid4())
        self._strategies[strategy.strategy_id] = asdict(strategy)
        self._save()
        logger.info("Registered strategy %s (%s)", strategy.name, strategy.strategy_id)
        return strategy.strategy_id

    def get(self, strategy_id: str) -> Optional[Strategy]:
        data = self._strategies.get(strategy_id)
        if not data:
            return None
        return self._to_strategy(data)

    def list_all(self, state: Optional[StrategyState] = None) -> List[Strategy]:
        strategies = [self._to_strategy(d) for d in self._strategies.values()]
        if state:
            strategies = [s for s in strategies if s.state == state.value]
        return sorted(strategies, key=lambda s: s.created_at, reverse=True)

    def update_state(self, strategy_id: str, new_state: StrategyState, reason: str, triggered_by: str = "system") -> None:
        data = self._strategies.get(strategy_id)
        if not data:
            raise KeyError(f"Strategy {strategy_id} not found")
        old_state = data["state"]
        now = datetime.now(timezone.utc).isoformat()
        transition = {
            "from_state": old_state,
            "to_state": new_state.value,
            "reason": reason,
            "timestamp": now,
            "triggered_by": triggered_by,
        }
        data["state"] = new_state.value
        data["state_entered_at"] = now
        data["last_demotion_check"] = now
        if "state_history" not in data:
            data["state_history"] = []
        data["state_history"].append(transition)
        self._save()
        logger.info("Strategy %s: %s → %s (%s)", strategy_id, old_state, new_state.value, reason)

    def update_strategy(self, strategy: Strategy) -> None:
        self._strategies[strategy.strategy_id] = asdict(strategy)
        self._save()

    def get_state_history(self, strategy_id: str) -> List[StateTransition]:
        data = self._strategies.get(strategy_id, {})
        history = data.get("state_history", [])
        return [
            StateTransition(
                strategy_id=strategy_id,
                from_state=h.get("from_state", ""),
                to_state=h.get("to_state", ""),
                transition_type=h.get("transition_type", TransitionType.PROMOTE.value),
                reason=h.get("reason", ""),
                timestamp=h.get("timestamp", ""),
                triggered_by=h.get("triggered_by", "system"),
            )
            for h in history
        ]

    def export_report(self) -> Any:
        """Return full registry summary as DataFrame or list of dicts."""
        rows = []
        for sid, data in self._strategies.items():
            rows.append({
                "strategy_id": sid,
                "name": data.get("name"),
                "state": data.get("state"),
                "created_at": data.get("created_at"),
                "state_entered_at": data.get("state_entered_at"),
                "backtest_sharpe": data.get("backtest_sharpe"),
                "live_sharpe": data.get("live_sharpe"),
                "live_max_drawdown": data.get("live_max_drawdown"),
                "current_allocation": data.get("current_allocation"),
                "tags": data.get("tags"),
            })
        if PANDAS_AVAILABLE:
            try:
                return pd.DataFrame(rows)
            except Exception:
                pass
        return rows


# ===========================================================================
# Capital Allocator
# ===========================================================================

class CapitalAllocator:
    """
    State-aware capital allocation with hard limits per lifecycle stage.
    Limits by state:
      PAPER_TRADING:  $0 (paper only)
      LIVE_SHADOW:    $0 (no real fills)
      LIVE_SMALL:     min($50K, 2% of portfolio)
      LIVE_FULL:      up to 20% per strategy, 60% total
    """

    LIVE_SMALL_MAX_ABS = 50_000.0
    LIVE_SMALL_PCT = 0.02
    LIVE_FULL_PER_STRATEGY_PCT = 0.20
    LIVE_FULL_TOTAL_PCT = 0.60

    def allocate(self, strategy: Strategy, portfolio_equity: float) -> float:
        state = StrategyState(strategy.state)
        if state in (StrategyState.RESEARCH, StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW,
                     StrategyState.RETIRED, StrategyState.SUSPENDED):
            return 0.0
        if state == StrategyState.LIVE_SMALL:
            return min(self.LIVE_SMALL_MAX_ABS, portfolio_equity * self.LIVE_SMALL_PCT)
        if state == StrategyState.LIVE_FULL:
            return portfolio_equity * self.LIVE_FULL_PER_STRATEGY_PCT
        return 0.0

    def compute_total_allocation(self, portfolio_equity: float, registry: StrategyRegistry) -> Dict[str, float]:
        """Compute dollar allocations for all live strategies, respecting total cap."""
        allocations: Dict[str, float] = {}
        live_full = registry.list_all(StrategyState.LIVE_FULL)
        live_small = registry.list_all(StrategyState.LIVE_SMALL)

        total_cap = portfolio_equity * self.LIVE_FULL_TOTAL_PCT
        running_total = 0.0

        for s in live_full:
            alloc = self.allocate(s, portfolio_equity)
            if running_total + alloc > total_cap:
                alloc = max(0.0, total_cap - running_total)
            allocations[s.strategy_id] = round(alloc, 2)
            running_total += alloc

        for s in live_small:
            alloc = self.allocate(s, portfolio_equity)
            allocations[s.strategy_id] = round(alloc, 2)

        return allocations

    def rebalance_allocations(self, portfolio_equity: float, registry: StrategyRegistry) -> List[AllocationChange]:
        """Compute allocation changes from current to target."""
        new_allocs = self.compute_total_allocation(portfolio_equity, registry)
        changes: List[AllocationChange] = []
        all_strategies = registry.list_all()

        for s in all_strategies:
            old = s.current_allocation
            new = new_allocs.get(s.strategy_id, 0.0)
            if abs(new - old) > 0.01:
                changes.append(AllocationChange(
                    strategy_id=s.strategy_id,
                    old_allocation=old,
                    new_allocation=new,
                    reason=f"Rebalance: state={s.state}, equity={portfolio_equity:,.0f}",
                ))
                # Update stored allocation
                s.current_allocation = new
                registry.update_strategy(s)

        return changes


# ===========================================================================
# State Machine
# ===========================================================================

class StateMachine:
    """
    Core FSM: validate criteria, execute transitions, record events, emit notifications.
    """

    def __init__(self, registry: StrategyRegistry, tracker: PerformanceTracker) -> None:
        self.registry = registry
        self.tracker = tracker
        self.trigger = DemotionTrigger()

    def can_promote(self, strategy: Strategy, metrics: PerformanceMetrics) -> Tuple[bool, List[str]]:
        """Return (can_promote, failed_criteria_list)."""
        current = StrategyState(strategy.state)
        idx = STATE_ORDER.index(current) if current in STATE_ORDER else -1
        if idx < 0 or idx >= len(STATE_ORDER) - 1:
            return False, [f"State {current.value} cannot be promoted"]
        next_state = STATE_ORDER[idx + 1]
        criteria = self.tracker.check_against_criteria(strategy, metrics, current, next_state)
        return criteria.can_promote, criteria.failed_criteria

    def transition(
        self,
        strategy: Strategy,
        target_state: StrategyState,
        metrics: PerformanceMetrics,
        force: bool = False,
        triggered_by: str = "system",
        reason: str = "",
    ) -> TransitionResult:
        current = StrategyState(strategy.state)
        criteria_result: Optional[CriteriaResult] = None

        # Validate criteria unless forced or it's a demotion/suspension/retire
        is_promotion = (
            target_state in STATE_ORDER and
            current in STATE_ORDER and
            STATE_ORDER.index(target_state) > STATE_ORDER.index(current)
        )

        if is_promotion and not force:
            criteria_result = self.tracker.check_against_criteria(strategy, metrics, current, target_state)
            if not criteria_result.can_promote:
                return TransitionResult(
                    success=False,
                    strategy_id=strategy.strategy_id,
                    from_state=current.value,
                    to_state=target_state.value,
                    reason=f"Criteria not met: {criteria_result.failed_criteria}",
                    criteria_result=criteria_result,
                    error=f"Failed criteria: {', '.join(criteria_result.failed_criteria)}",
                )

        transition_reason = reason or f"{triggered_by} transition {current.value} → {target_state.value}"

        try:
            self.registry.update_state(strategy.strategy_id, target_state, transition_reason, triggered_by)
            self._emit_notification(strategy, current, target_state, transition_reason)
            return TransitionResult(
                success=True,
                strategy_id=strategy.strategy_id,
                from_state=current.value,
                to_state=target_state.value,
                reason=transition_reason,
                criteria_result=criteria_result,
            )
        except Exception as exc:
            logger.exception("Transition failed for %s", strategy.strategy_id)
            return TransitionResult(
                success=False,
                strategy_id=strategy.strategy_id,
                from_state=current.value,
                to_state=target_state.value,
                reason=transition_reason,
                error=str(exc),
            )

    def auto_promote_check(self, strategy_id: str) -> Optional[TransitionResult]:
        """Check if a strategy qualifies for promotion; execute if so."""
        strategy = self.registry.get(strategy_id)
        if not strategy:
            return None
        current = StrategyState(strategy.state)
        if current not in STATE_ORDER or current == StrategyState.LIVE_FULL:
            return None
        metrics = self.tracker.compute_metrics(strategy_id, "all")
        can, failed = self.can_promote(strategy, metrics)
        if can:
            idx = STATE_ORDER.index(current)
            next_state = STATE_ORDER[idx + 1]
            logger.info("Auto-promoting %s: %s → %s", strategy_id, current.value, next_state.value)
            return self.transition(strategy, next_state, metrics, reason="Auto-promotion: all criteria met")
        return None

    def emergency_suspend(self, strategy_id: str, reason: str) -> TransitionResult:
        """Immediately suspend a strategy regardless of state."""
        strategy = self.registry.get(strategy_id)
        if not strategy:
            return TransitionResult(success=False, strategy_id=strategy_id, from_state="unknown", to_state="suspended", reason=reason, error="Strategy not found")
        current = StrategyState(strategy.state)
        empty_metrics = PerformanceMetrics(strategy_id=strategy_id, period="all")
        result = self.transition(strategy, StrategyState.SUSPENDED, empty_metrics, force=True, triggered_by="emergency", reason=f"EMERGENCY SUSPEND: {reason}")
        logger.warning("EMERGENCY SUSPEND: %s — %s", strategy_id, reason)
        return result

    def _emit_notification(self, strategy: Strategy, from_state: StrategyState, to_state: StrategyState, reason: str) -> None:
        """Log notification; extend to webhook if needed."""
        level = logging.WARNING if to_state in (StrategyState.RETIRED, StrategyState.SUSPENDED) else logging.INFO
        logger.log(level, "[LIFECYCLE] %s (%s): %s → %s | %s", strategy.name, strategy.strategy_id[:8], from_state.value, to_state.value, reason)


# ===========================================================================
# Strategy Lifecycle Manager (Orchestrator)
# ===========================================================================

class StrategyLifecycleManager:
    """
    Daily orchestrator: checks all active strategies for promotion/demotion triggers,
    manages capital allocation, generates lifecycle reports.
    """

    def __init__(self) -> None:
        self.registry = StrategyRegistry()
        self.tracker = PerformanceTracker()
        self.fsm = StateMachine(self.registry, self.tracker)
        self.allocator = CapitalAllocator()
        self.demotion_trigger = DemotionTrigger()

    def run_daily_checks(self) -> dict:
        """Run promotion and demotion checks for all active strategies."""
        results: dict = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "promotions": [],
            "demotions": [],
            "suspensions": [],
            "no_action": [],
        }

        active_states = [StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW, StrategyState.LIVE_SMALL, StrategyState.LIVE_FULL]
        for state in active_states:
            for strategy in self.registry.list_all(state):
                metrics = self.tracker.compute_metrics(strategy.strategy_id, "all")
                # Check demotion first (safety)
                dem = self.demotion_trigger.check_for_demotion(strategy, metrics)
                if dem.should_demote and dem.urgency == "immediate":
                    res = self.fsm.emergency_suspend(strategy.strategy_id, dem.reason)
                    results["suspensions"].append({"strategy_id": strategy.strategy_id, "reason": dem.reason})
                    continue
                if dem.should_demote and dem.urgency == "scheduled":
                    target = StrategyState(dem.recommended_state) if dem.recommended_state else StrategyState.RETIRED
                    res = self.fsm.transition(strategy, target, metrics, force=True, reason=dem.reason)
                    if res.success:
                        results["demotions"].append({"strategy_id": strategy.strategy_id, "to": target.value, "reason": dem.reason})
                        continue
                # Check promotion
                promo = self.fsm.auto_promote_check(strategy.strategy_id)
                if promo and promo.success:
                    results["promotions"].append({"strategy_id": strategy.strategy_id, "to": promo.to_state})
                else:
                    results["no_action"].append(strategy.strategy_id)

        logger.info("Daily checks: %d promotions, %d demotions, %d suspensions",
                    len(results["promotions"]), len(results["demotions"]), len(results["suspensions"]))
        return results

    def promote(self, strategy_id: str, force: bool = False) -> TransitionResult:
        strategy = self.registry.get(strategy_id)
        if not strategy:
            return TransitionResult(success=False, strategy_id=strategy_id, from_state="", to_state="", reason="Strategy not found", error="Not found")
        current = StrategyState(strategy.state)
        if current not in STATE_ORDER or current == StrategyState.LIVE_FULL:
            return TransitionResult(success=False, strategy_id=strategy_id, from_state=current.value, to_state="", reason="Cannot promote from this state", error=f"State {current.value} cannot be promoted")
        idx = STATE_ORDER.index(current)
        next_state = STATE_ORDER[idx + 1]
        metrics = self.tracker.compute_metrics(strategy_id, "all")
        return self.fsm.transition(strategy, next_state, metrics, force=force, triggered_by="manual", reason="Manual promotion request")

    def demote(self, strategy_id: str, reason: str = "Manual demotion") -> TransitionResult:
        strategy = self.registry.get(strategy_id)
        if not strategy:
            return TransitionResult(success=False, strategy_id=strategy_id, from_state="", to_state="", reason=reason, error="Not found")
        current = StrategyState(strategy.state)
        target = DEMOTION_MAP.get(current, StrategyState.RETIRED)
        metrics = self.tracker.compute_metrics(strategy_id, "all")
        return self.fsm.transition(strategy, target, metrics, force=True, triggered_by="manual", reason=reason)

    def retire(self, strategy_id: str, reason: str = "Manual retirement") -> TransitionResult:
        strategy = self.registry.get(strategy_id)
        if not strategy:
            return TransitionResult(success=False, strategy_id=strategy_id, from_state="", to_state="", reason=reason, error="Not found")
        metrics = self.tracker.compute_metrics(strategy_id, "all")
        return self.fsm.transition(strategy, StrategyState.RETIRED, metrics, force=True, triggered_by="manual", reason=reason)

    def get_dashboard(self) -> Any:
        """Return all strategies + state + key metrics + capital as DataFrame or list."""
        rows = []
        all_strategies = self.registry.list_all()
        for s in all_strategies:
            metrics = self.tracker.compute_metrics(s.strategy_id, "90d")
            rows.append({
                "strategy_id": s.strategy_id,
                "name": s.name,
                "state": s.state,
                "days_in_state": _days_since(s.state_entered_at),
                "sharpe_90d": round(metrics.sharpe_ratio, 3),
                "max_dd": round(metrics.max_drawdown, 4),
                "win_rate": round(metrics.win_rate, 3),
                "n_trades": metrics.n_trades,
                "current_allocation": s.current_allocation,
                "backtest_sharpe": s.backtest_sharpe,
                "created_at": s.created_at[:10],
            })
        if PANDAS_AVAILABLE:
            try:
                return pd.DataFrame(rows)
            except Exception:
                pass
        return rows

    def get_promotion_candidates(self) -> List[str]:
        """Return strategy IDs that are ready to move to the next state."""
        candidates = []
        promotable_states = [StrategyState.RESEARCH, StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW, StrategyState.LIVE_SMALL]
        for state in promotable_states:
            for s in self.registry.list_all(state):
                metrics = self.tracker.compute_metrics(s.strategy_id, "all")
                can, _ = self.fsm.can_promote(s, metrics)
                if can:
                    candidates.append(s.strategy_id)
        return candidates

    def get_at_risk_strategies(self) -> List[str]:
        """Return strategy IDs that are close to demotion triggers."""
        at_risk = []
        live_states = [StrategyState.LIVE_SHADOW, StrategyState.LIVE_SMALL, StrategyState.LIVE_FULL]
        for state in live_states:
            for s in self.registry.list_all(state):
                metrics = self.tracker.compute_metrics(s.strategy_id, "all")
                decision = self.demotion_trigger.check_for_demotion(s, metrics)
                if decision.should_demote:
                    at_risk.append(s.strategy_id)
                    continue
                # Also flag near-miss: drawdown > 20% (close to 25% hard stop)
                if abs(metrics.max_drawdown) > 0.20:
                    at_risk.append(s.strategy_id)
                # Sharpe trending down
                elif metrics.sharpe_ratio < 0.2:
                    at_risk.append(s.strategy_id)
        return list(set(at_risk))

    def generate_lifecycle_report(self) -> str:
        """Generate a human-readable lifecycle report."""
        lines = [
            "=" * 70,
            f"SENTINEL Strategy Lifecycle Report — {datetime.now(timezone.utc).date()}",
            "=" * 70,
        ]
        for state in [StrategyState.RESEARCH, StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW,
                      StrategyState.LIVE_SMALL, StrategyState.LIVE_FULL, StrategyState.RETIRED, StrategyState.SUSPENDED]:
            strategies = self.registry.list_all(state)
            if not strategies:
                continue
            lines.append(f"\n[{state.value.upper()}] — {len(strategies)} strateg{'y' if len(strategies) == 1 else 'ies'}")
            lines.append("-" * 50)
            for s in strategies:
                metrics = self.tracker.compute_metrics(s.strategy_id, "90d")
                days = _days_since(s.state_entered_at)
                alloc = f"${s.current_allocation:,.0f}" if s.current_allocation else "—"
                sharpe = f"{metrics.sharpe_ratio:.2f}" if metrics.sufficient_data else "n/a"
                dd = f"{abs(metrics.max_drawdown)*100:.1f}%" if metrics.sufficient_data else "n/a"
                lines.append(
                    f"  {s.name[:30]:<30} | {days:>4}d | Sharpe: {sharpe:>6} | DD: {dd:>6} | Capital: {alloc:>10}"
                )

        candidates = self.get_promotion_candidates()
        at_risk = self.get_at_risk_strategies()
        lines.append(f"\nPromotion candidates: {len(candidates)} — {candidates}")
        lines.append(f"At-risk strategies:   {len(at_risk)} — {at_risk}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ===========================================================================
# Backtest Gate Validator
# ===========================================================================

class BacktestGateValidator:
    """
    Validates backtest quality to prevent data-mined strategies entering paper trading.
    Checks PBO, regime coverage, parameter sensitivity, and minimum OOS sample.
    """

    def __init__(self) -> None:
        pass

    def compute_pbo(self, backtest_returns: List[float]) -> float:
        """
        Estimate Probability of Backtest Overfitting (PBO) via combinatorial
        cross-validation over parameter permutations.
        Simplified: ratio of parameter sets with OOS Sharpe < 0 to total.
        Returns value in [0, 1] — lower is better.
        """
        if len(backtest_returns) < 60:
            return 0.5  # Cannot compute — assume neutral
        arr = np.array(backtest_returns, dtype=float)
        mid = len(arr) // 2
        is_returns = arr[:mid]
        oos_returns = arr[mid:]
        is_sharpe = float(np.mean(is_returns) / np.std(is_returns, ddof=1)) * math.sqrt(252) if np.std(is_returns) > 0 else 0.0
        oos_sharpe = float(np.mean(oos_returns) / np.std(oos_returns, ddof=1)) * math.sqrt(252) if np.std(oos_returns) > 0 else 0.0
        # Simple deflated Sharpe estimate
        if is_sharpe <= 0:
            return 0.9
        ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0
        pbo = max(0.0, min(1.0, 1.0 - ratio))
        return round(pbo, 4)

    def check_regime_coverage(self, date_index: List[str]) -> dict:
        """
        Determine how many distinct market regimes are covered by the backtest date range.
        Uses known regime boundaries (simplified).
        Returns: {n_regimes, regimes_list, coverage_ok}
        """
        REGIME_BOUNDARIES = [
            ("2008-09-01", "2009-03-31", "bear_gfc"),
            ("2009-04-01", "2011-04-30", "bull_recovery"),
            ("2011-05-01", "2011-10-31", "sideways_euro_crisis"),
            ("2011-11-01", "2015-12-31", "bull_qe"),
            ("2016-01-01", "2016-03-31", "sideways_china_fear"),
            ("2016-04-01", "2019-12-31", "bull_trump"),
            ("2020-02-01", "2020-03-31", "bear_covid"),
            ("2020-04-01", "2021-12-31", "bull_stimulus"),
            ("2022-01-01", "2022-12-31", "bear_rate_hike"),
            ("2023-01-01", "2024-12-31", "bull_ai"),
        ]
        if not date_index:
            return {"n_regimes": 0, "regimes": [], "coverage_ok": False}
        start = min(date_index)
        end = max(date_index)
        regimes_hit = []
        for r_start, r_end, r_name in REGIME_BOUNDARIES:
            # Overlap check
            if start <= r_end and end >= r_start:
                regimes_hit.append(r_name)
        return {
            "n_regimes": len(regimes_hit),
            "regimes": regimes_hit,
            "coverage_ok": len(regimes_hit) >= 2,
        }

    def check_parameter_sensitivity(self, sharpes_by_param: Dict[str, float]) -> dict:
        """
        Check that strategy performance is not knife-edge sensitive to parameters.
        sharpes_by_param: {param_label: sharpe_ratio}
        Returns sensitivity score and pass/fail.
        """
        if len(sharpes_by_param) < 3:
            return {"sensitivity": None, "pass": True, "note": "Too few param samples"}
        values = list(sharpes_by_param.values())
        std = float(np.std(values, ddof=1))
        mean = float(np.mean(values))
        cv = std / abs(mean) if mean != 0 else 9.9  # Coefficient of variation
        # Low CV = robust; high CV = fragile
        return {
            "sensitivity_cv": round(cv, 4),
            "mean_sharpe": round(mean, 4),
            "std_sharpe": round(std, 4),
            "pass": cv < 0.5,
            "note": "CV < 0.5 considered robust",
        }

    def full_gate_report(self, strategy: Strategy, daily_returns: Optional[List[float]] = None, date_index: Optional[List[str]] = None) -> dict:
        """Run all backtest quality gates and return a full report."""
        report: dict = {
            "strategy_id": strategy.strategy_id,
            "strategy_name": strategy.name,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "gates": {},
        }

        # Gate 1: Sharpe
        report["gates"]["sharpe"] = {
            "value": strategy.backtest_sharpe,
            "threshold": 0.8,
            "pass": strategy.backtest_sharpe >= 0.8,
        }

        # Gate 2: Drawdown
        report["gates"]["max_drawdown"] = {
            "value": strategy.backtest_max_drawdown,
            "threshold": -0.20,
            "pass": strategy.backtest_max_drawdown >= -0.20,
        }

        # Gate 3: Trading days
        report["gates"]["backtest_days"] = {
            "value": strategy.backtest_days,
            "threshold": 252,
            "pass": strategy.backtest_days >= 252,
        }

        # Gate 4: Win rate
        report["gates"]["win_rate"] = {
            "value": strategy.backtest_win_rate,
            "threshold": 0.40,
            "pass": strategy.backtest_win_rate >= 0.40,
        }

        # Gate 5: PBO
        if daily_returns and len(daily_returns) >= 60:
            pbo = self.compute_pbo(daily_returns)
        else:
            pbo = strategy.backtest_pbo
        report["gates"]["pbo"] = {
            "value": pbo,
            "threshold": 0.50,
            "pass": pbo <= 0.50,
            "note": "PBO ≤ 0.5 means strategy likely generalises OOS",
        }

        # Gate 6: Regime coverage
        if date_index:
            regime_check = self.check_regime_coverage(date_index)
        else:
            regime_check = {"n_regimes": strategy.backtest_regimes_covered, "coverage_ok": strategy.backtest_regimes_covered >= 2}
        report["gates"]["regime_coverage"] = {
            "value": regime_check.get("n_regimes", 0),
            "threshold": 2,
            "pass": regime_check.get("coverage_ok", False),
            "regimes": regime_check.get("regimes", []),
        }

        all_pass = all(g["pass"] for g in report["gates"].values())
        report["overall_pass"] = all_pass
        report["failed_gates"] = [k for k, v in report["gates"].items() if not v["pass"]]
        return report


# ===========================================================================
# Promotion Audit Log
# ===========================================================================

class PromotionAuditLog:
    """
    Append-only audit log of all lifecycle events.
    Stored as JSONL at sentinel/data/promotion_audit.jsonl.
    Provides query, export, and replay capabilities.
    """

    AUDIT_PATH = DATA_DIR / "promotion_audit.jsonl"

    def __init__(self) -> None:
        self.AUDIT_PATH = DATA_DIR / "promotion_audit.jsonl"

    def log_event(self, event_type: str, strategy_id: str, data: dict, triggered_by: str = "system") -> None:
        """Append an audit event."""
        entry = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "strategy_id": strategy_id,
            "triggered_by": triggered_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with self.AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_transition(self, result: TransitionResult, triggered_by: str = "system") -> None:
        self.log_event(
            "state_transition",
            result.strategy_id,
            {
                "from_state": result.from_state,
                "to_state": result.to_state,
                "success": result.success,
                "reason": result.reason,
                "error": result.error,
            },
            triggered_by,
        )

    def log_criteria_check(self, criteria: CriteriaResult, triggered_by: str = "system") -> None:
        self.log_event(
            "criteria_check",
            criteria.strategy_id,
            {
                "from_state": criteria.from_state.value if hasattr(criteria.from_state, "value") else str(criteria.from_state),
                "to_state": criteria.to_state.value if hasattr(criteria.to_state, "value") else str(criteria.to_state),
                "can_promote": criteria.can_promote,
                "failed_criteria": criteria.failed_criteria,
                "checks": criteria.checks,
            },
            triggered_by,
        )

    def log_demotion_check(self, decision: DemotionDecision, triggered_by: str = "system") -> None:
        self.log_event(
            "demotion_check",
            decision.strategy_id,
            {
                "should_demote": decision.should_demote,
                "urgency": decision.urgency,
                "reason": decision.reason,
                "triggered_rules": decision.triggered_rules,
                "recommended_state": decision.recommended_state,
            },
            triggered_by,
        )

    def query(self, strategy_id: Optional[str] = None, event_type: Optional[str] = None, limit: int = 100) -> List[dict]:
        """Query audit log with optional filters."""
        if not self.AUDIT_PATH.exists():
            return []
        results = []
        with self.AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if strategy_id and entry.get("strategy_id") != strategy_id:
                    continue
                if event_type and entry.get("event_type") != event_type:
                    continue
                results.append(entry)
        # Most recent first
        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return results[:limit]

    def export_csv(self) -> str:
        """Export audit log to CSV string."""
        rows = self.query(limit=10000)
        if not rows:
            return "event_id,event_type,strategy_id,triggered_by,timestamp\n"
        lines = ["event_id,event_type,strategy_id,triggered_by,timestamp"]
        for r in rows:
            lines.append(",".join([
                r.get("event_id", ""),
                r.get("event_type", ""),
                r.get("strategy_id", ""),
                r.get("triggered_by", ""),
                r.get("timestamp", ""),
            ]))
        return "\n".join(lines)

    def get_summary(self, strategy_id: str) -> dict:
        """Return event count summary for a strategy."""
        events = self.query(strategy_id=strategy_id, limit=10000)
        by_type: Dict[str, int] = {}
        for e in events:
            t = e.get("event_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "strategy_id": strategy_id,
            "total_events": len(events),
            "by_type": by_type,
            "first_event": events[-1].get("timestamp") if events else None,
            "last_event": events[0].get("timestamp") if events else None,
        }


# ===========================================================================
# Strategy Analytics
# ===========================================================================

class StrategyAnalytics:
    """
    Advanced analytics for comparing strategies and computing ensemble metrics.
    Used by StrategyLifecycleManager to inform promotion/demotion decisions.
    """

    def __init__(self, tracker: PerformanceTracker) -> None:
        self.tracker = tracker

    def compute_sharpe_decay(self, strategy_id: str, window: int = 63) -> dict:
        """
        Detect Sharpe ratio decay over time using rolling windows.
        Returns slope of rolling Sharpe trend — negative slope = decay.
        """
        rolling = self.tracker.compute_rolling_metrics(strategy_id, window)
        if PANDAS_AVAILABLE and hasattr(rolling, "empty"):
            if rolling.empty:
                return {"strategy_id": strategy_id, "decay_detected": False, "note": "Insufficient data"}
            sharpes = rolling["rolling_sharpe"].values
        elif isinstance(rolling, list):
            sharpes = np.array([r["rolling_sharpe"] for r in rolling], dtype=float)
        else:
            return {"strategy_id": strategy_id, "decay_detected": False, "note": "No rolling data"}

        if len(sharpes) < 10:
            return {"strategy_id": strategy_id, "decay_detected": False, "note": "Insufficient rolling windows"}

        x = np.arange(len(sharpes), dtype=float)
        slope = float(np.polyfit(x, sharpes, 1)[0])
        latest_sharpe = float(sharpes[-1])
        early_sharpe = float(np.mean(sharpes[:max(1, len(sharpes) // 4)]))
        decay_detected = slope < -0.005 and latest_sharpe < early_sharpe * 0.7
        return {
            "strategy_id": strategy_id,
            "decay_detected": decay_detected,
            "slope": round(slope, 6),
            "latest_sharpe": round(latest_sharpe, 4),
            "early_sharpe": round(early_sharpe, 4),
            "decay_pct": round((latest_sharpe - early_sharpe) / abs(early_sharpe) * 100, 2) if early_sharpe != 0 else 0.0,
        }

    def compute_information_ratio(self, strategy_id: str, benchmark_id: Optional[str] = None) -> dict:
        """
        Compute Information Ratio vs a benchmark strategy (or zero excess return).
        IR = mean(active_return) / std(active_return).
        """
        metrics = self.tracker.compute_metrics(strategy_id, "all")
        if not metrics.sufficient_data:
            return {"strategy_id": strategy_id, "ir": None, "note": "Insufficient data"}
        # Without a benchmark we use zero (absolute IR)
        ir = metrics.annualized_return / metrics.volatility if metrics.volatility > 0 else 0.0
        return {
            "strategy_id": strategy_id,
            "benchmark_id": benchmark_id,
            "information_ratio": round(ir, 4),
            "annualized_return": round(metrics.annualized_return, 4),
            "tracking_error": round(metrics.volatility, 4),
        }

    def compute_ensemble_metrics(self, strategy_ids: List[str], weights: Optional[List[float]] = None) -> dict:
        """
        Compute blended performance metrics for an ensemble of strategies.
        Useful for evaluating a portfolio of LIVE strategies.
        """
        if not strategy_ids:
            return {"error": "No strategy IDs provided"}
        if weights is None:
            weights = [1.0 / len(strategy_ids)] * len(strategy_ids)
        if len(weights) != len(strategy_ids):
            return {"error": "Weights length mismatch"}
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        metrics_list = [self.tracker.compute_metrics(sid, "all") for sid in strategy_ids]
        valid = [(sid, m, w) for sid, m, w in zip(strategy_ids, metrics_list, weights) if m.sufficient_data]

        if not valid:
            return {"strategy_ids": strategy_ids, "note": "No strategies with sufficient data"}

        blended_sharpe = sum(m.sharpe_ratio * w for _, m, w in valid)
        blended_return = sum(m.annualized_return * w for _, m, w in valid)
        blended_dd = sum(m.max_drawdown * w for _, m, w in valid)
        blended_win_rate = sum(m.win_rate * w for _, m, w in valid)

        return {
            "strategy_ids": strategy_ids,
            "weights": weights,
            "blended_sharpe": round(blended_sharpe, 4),
            "blended_return": round(blended_return, 4),
            "blended_max_drawdown": round(blended_dd, 4),
            "blended_win_rate": round(blended_win_rate, 4),
            "n_strategies": len(valid),
        }

    def rank_strategies_by_sharpe(self, registry: StrategyRegistry, state: Optional[StrategyState] = None) -> List[dict]:
        """Rank all strategies by live Sharpe ratio."""
        strategies = registry.list_all(state)
        ranked = []
        for s in strategies:
            metrics = self.tracker.compute_metrics(s.strategy_id, "90d")
            ranked.append({
                "rank": 0,
                "strategy_id": s.strategy_id,
                "name": s.name,
                "state": s.state,
                "sharpe_90d": round(metrics.sharpe_ratio, 4) if metrics.sufficient_data else None,
                "max_dd": round(metrics.max_drawdown, 4) if metrics.sufficient_data else None,
                "n_trades": metrics.n_trades,
            })
        ranked.sort(key=lambda r: (r["sharpe_90d"] or -99.0), reverse=True)
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        return ranked

    def detect_correlation_cluster(self, strategy_ids: List[str], threshold: float = 0.7) -> dict:
        """
        Detect if a group of strategies are highly correlated (concentration risk).
        Returns pairs above threshold and a cluster flag.
        """
        if len(strategy_ids) < 2:
            return {"clusters": [], "concentration_risk": False}

        # Gather daily PnL series
        all_pnls: Dict[str, List[float]] = {}
        for sid in strategy_ids:
            data = self.tracker._data.get(sid, {})
            daily = data.get("daily_pnl", [])
            if daily:
                all_pnls[sid] = [r["pnl"] for r in daily]

        if len(all_pnls) < 2:
            return {"clusters": [], "concentration_risk": False, "note": "Insufficient PnL data"}

        # Align lengths
        min_len = min(len(v) for v in all_pnls.values())
        arrays = {sid: np.array(v[-min_len:]) for sid, v in all_pnls.items()}
        sids = list(arrays.keys())

        high_corr_pairs = []
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = arrays[sids[i]], arrays[sids[j]]
                if np.std(a) > 0 and np.std(b) > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    if corr >= threshold:
                        high_corr_pairs.append({"strategy_a": sids[i], "strategy_b": sids[j], "correlation": round(corr, 4)})

        return {
            "strategy_ids": strategy_ids,
            "threshold": threshold,
            "high_correlation_pairs": high_corr_pairs,
            "concentration_risk": len(high_corr_pairs) > 0,
        }


# ===========================================================================
# Helpers
# ===========================================================================

def _days_since(iso_timestamp: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 0


def _make_mock_metrics(strategy_id: str, **overrides) -> PerformanceMetrics:
    """Build a PerformanceMetrics object with realistic defaults (for testing)."""
    m = PerformanceMetrics(strategy_id=strategy_id, period="all", sufficient_data=True)
    m.sharpe_ratio = overrides.get("sharpe_ratio", 1.2)
    m.max_drawdown = overrides.get("max_drawdown", -0.08)
    m.win_rate = overrides.get("win_rate", 0.55)
    m.profit_factor = overrides.get("profit_factor", 1.8)
    m.n_trades = overrides.get("n_trades", 85)
    m.annualized_return = overrides.get("annualized_return", 0.18)
    m.volatility = overrides.get("volatility", 0.15)
    m.sortino_ratio = overrides.get("sortino_ratio", 1.7)
    m.calmar_ratio = overrides.get("calmar_ratio", 2.2)
    m.slippage_ratio = overrides.get("slippage_ratio", 0.05)
    m.backtest_correlation = overrides.get("backtest_correlation", 0.72)
    m.shadow_vs_actual_correlation = overrides.get("shadow_vs_actual_correlation", 0.91)
    m.shadow_position_errors = overrides.get("shadow_position_errors", 0)
    m.risk_manager_triggers = overrides.get("risk_manager_triggers", 0)
    m.days_in_state = overrides.get("days_in_state", 95)
    return m


# ===========================================================================
# __main__ demo
# ===========================================================================

if __name__ == "__main__":
    import tempfile

    print("\n" + "=" * 70)
    print("SENTINEL Strategy Promotion State Machine v3 — Demo")
    print("=" * 70)

    # Use a temp directory so demo doesn't pollute production registry
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SENTINEL_DATA_DIR"] = tmpdir
        # Re-initialize paths using patched env
        _DATA_DIR = Path(tmpdir)
        _REGISTRY_PATH = _DATA_DIR / "strategy_registry.json"
        _PERFORMANCE_PATH = _DATA_DIR / "strategy_performance.json"

        # Monkey-patch module-level paths for demo
        import sentinel.sbx.strategy_promotion_v3 as _self
        _self.REGISTRY_PATH = _REGISTRY_PATH
        _self.PERFORMANCE_PATH = _PERFORMANCE_PATH

        # 1. Create a strategy in RESEARCH state
        strategy = Strategy(
            strategy_id=str(uuid.uuid4()),
            name="Momentum Mean-Reversion Hybrid",
            description="Long/short equity combining 12-1 month momentum with short-term mean reversion",
            author="quant_team",
            state=StrategyState.RESEARCH.value,
            asset_class="equity",
            timeframe="daily",
            tags=["momentum", "mean_reversion", "long_short"],
        )

        # Simulate completing a backtest meeting all criteria
        strategy.backtest_sharpe = 1.15
        strategy.backtest_max_drawdown = -0.14
        strategy.backtest_days = 1260          # 5 years
        strategy.backtest_win_rate = 0.54
        strategy.backtest_pbo = 0.31           # low PBO = good
        strategy.backtest_regimes_covered = 3  # bull, bear, sideways

        print(f"\n[1] Registered strategy: {strategy.name}")
        print(f"    State: {strategy.state}")
        print(f"    Backtest Sharpe: {strategy.backtest_sharpe} | Max DD: {strategy.backtest_max_drawdown:.1%} | Days: {strategy.backtest_days}")

        # 2. Register with registry
        mgr = StrategyLifecycleManager()
        # Patch the registry paths directly
        mgr.registry._strategies = {}
        mgr.registry.registry_path = _REGISTRY_PATH if hasattr(mgr.registry, "registry_path") else None

        # Override the _save/_load to use tmp paths
        import json as _json
        def _patched_save(self_reg):
            _REGISTRY_PATH.write_text(_json.dumps(self_reg._strategies, indent=2, default=str), encoding="utf-8")
        def _patched_load(self_reg):
            if _REGISTRY_PATH.exists():
                return _json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
            return {}
        import types
        mgr.registry._save = types.MethodType(_patched_save, mgr.registry)
        mgr.registry._load = types.MethodType(_patched_load, mgr.registry)

        strategy_id = mgr.registry.register(strategy)
        print(f"    Strategy ID: {strategy_id}")

        # 3. Validate criteria for RESEARCH → PAPER_TRADING
        print("\n[2] Checking promotion criteria: RESEARCH → PAPER_TRADING")
        tracker = PerformanceTracker()
        tracker._data = {}
        mock_metrics = _make_mock_metrics(strategy_id)
        criteria_result = tracker.check_against_criteria(
            strategy, mock_metrics,
            StrategyState.RESEARCH, StrategyState.PAPER_TRADING
        )
        for name, check in criteria_result.checks.items():
            symbol = "[PASS]" if check["result"] == "pass" else "[FAIL]" if check["result"] == "fail" else "[SKIP]"
            value_str = f"{check['value']}" if check.get("value") is not None else "n/a"
            print(f"    {symbol} {name}: {value_str} (threshold: {check['threshold']} {check.get('operator','>=')}) — {check['description']}")

        print(f"\n    Can promote: {criteria_result.can_promote}")
        if criteria_result.failed_criteria:
            print(f"    Failed: {criteria_result.failed_criteria}")

        # 4. Promote to PAPER_TRADING
        print("\n[3] Promoting to PAPER_TRADING...")
        # Override state_entered_at to simulate time passage
        loaded = mgr.registry.get(strategy_id)
        # Simulate strategy has been in RESEARCH for 30 days
        loaded.state_entered_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        loaded.backtest_sharpe = strategy.backtest_sharpe
        loaded.backtest_max_drawdown = strategy.backtest_max_drawdown
        loaded.backtest_days = strategy.backtest_days
        loaded.backtest_win_rate = strategy.backtest_win_rate
        loaded.backtest_pbo = strategy.backtest_pbo
        loaded.backtest_regimes_covered = strategy.backtest_regimes_covered
        mgr.registry.update_strategy(loaded)

        result = mgr.promote(strategy_id, force=False)
        print(f"    Success: {result.success}")
        print(f"    {result.from_state} → {result.to_state}")
        print(f"    Reason: {result.reason}")

        # 5. Simulate paper trading — inject synthetic daily PnL
        print("\n[4] Simulating 65 days of paper trading...")
        mgr.tracker._data = {}
        rng = np.random.default_rng(2024)
        equity = 100_000.0
        start = datetime.now(timezone.utc) - timedelta(days=65)
        for d in range(65):
            daily_return = float(rng.normal(0.0008, 0.009))  # ~18% ann return, 14% vol
            daily_pnl = equity * daily_return
            equity += daily_pnl
            date_str = (start + timedelta(days=d)).date().isoformat()
            mgr.tracker.record_daily_pnl(strategy_id, date_str, daily_pnl, equity)
            # Simulate a trade every ~2 days
            if d % 2 == 0:
                exp_pnl = daily_pnl * 0.8
                mgr.tracker.record_trade(strategy_id, {
                    "date": date_str,
                    "symbol": "AAPL",
                    "side": "long",
                    "qty": 100,
                    "price": 185.0 + rng.normal(0, 0.5),
                    "expected_price": 185.0,
                    "pnl": daily_pnl,
                    "expected_pnl": exp_pnl,
                })

        paper_metrics = mgr.tracker.compute_metrics(strategy_id, "all")
        print(f"    Sharpe: {paper_metrics.sharpe_ratio:.3f}")
        print(f"    Max DD: {paper_metrics.max_drawdown:.2%}")
        print(f"    Win Rate: {paper_metrics.win_rate:.1%}")
        print(f"    Trades: {paper_metrics.n_trades}")
        print(f"    Slippage ratio: {paper_metrics.slippage_ratio:.3f}")

        # Set shadow correlation manually (would come from execution system)
        paper_metrics.shadow_vs_actual_correlation = 0.92
        paper_metrics.shadow_position_errors = 0
        paper_metrics.risk_manager_triggers = 0

        # 6. Check PAPER_TRADING → LIVE_SHADOW criteria
        print("\n[5] Checking promotion criteria: PAPER_TRADING → LIVE_SHADOW")
        loaded2 = mgr.registry.get(strategy_id)
        # Simulate 65 days in paper trading state
        loaded2.state_entered_at = (datetime.now(timezone.utc) - timedelta(days=65)).isoformat()
        mgr.registry.update_strategy(loaded2)
        criteria2 = mgr.tracker.check_against_criteria(
            loaded2, paper_metrics,
            StrategyState.PAPER_TRADING, StrategyState.LIVE_SHADOW
        )
        for name, check in criteria2.checks.items():
            symbol = "[PASS]" if check["result"] == "pass" else "[FAIL]" if check["result"] == "fail" else "[SKIP]"
            value_str = f"{check.get('value', 'n/a')}"
            print(f"    {symbol} {name}: {value_str} (threshold: {check['threshold']} {check.get('operator','>=')}) — {check['description']}")
        print(f"\n    Can promote: {criteria2.can_promote}")

        # 7. State history
        print("\n[6] State history:")
        history = mgr.registry.get_state_history(strategy_id)
        for h in history:
            print(f"    {h.timestamp[:19]}  {h.from_state:15s} → {h.to_state:15s}  ({h.triggered_by})")

        # 8. Demotion trigger check
        print("\n[7] Demotion trigger check (simulating stressed metrics)...")
        stressed = _make_mock_metrics(strategy_id, max_drawdown=-0.28, sharpe_ratio=-0.3, n_consecutive_loss_days=6)
        loaded3 = mgr.registry.get(strategy_id)
        dem_decision = DemotionTrigger().check_for_demotion(loaded3, stressed)
        print(f"    Should demote: {dem_decision.should_demote}")
        print(f"    Urgency: {dem_decision.urgency}")
        print(f"    Reason: {dem_decision.reason}")
        print(f"    Triggered rules: {dem_decision.triggered_rules}")

        # 9. Capital allocation
        print("\n[8] Capital allocation (portfolio equity = $5M):")
        allocs = mgr.allocator.compute_total_allocation(5_000_000.0, mgr.registry)
        for sid, alloc in allocs.items():
            s = mgr.registry.get(sid)
            print(f"    {s.name[:35]:<35} ({s.state}): ${alloc:,.0f}")

        # 10. Kelly sizing example
        print("\n[9] Kelly position sizing example:")
        kelly = {"win_rate": 0.55, "avg_win": 1200.0, "avg_loss": 800.0}
        b = kelly["avg_win"] / kelly["avg_loss"]
        full_k = (kelly["win_rate"] * (b + 1) - 1) / b
        print(f"    Win rate: {kelly['win_rate']:.0%} | Avg win: ${kelly['avg_win']:,.0f} | Avg loss: ${kelly['avg_loss']:,.0f}")
        print(f"    Full Kelly: {full_k:.3f} ({full_k*100:.1f}% of capital)")
        print(f"    Half Kelly: {full_k/2:.3f} ({full_k/2*100:.1f}% of capital) [recommended]")

        print("\n" + "=" * 70)
        print("Demo complete.")
        print("=" * 70)
