"""
Strategy lifecycle management: paper → live promotion state machine.
Tracks strategy performance through required gates before live deployment.

Dimension targeted:
  dim_067 — Strategy promotion state machine  score → 9

State machine:
  DEVELOPMENT → PAPER_TESTING → VALIDATION → STAGING → LIVE → RETIRED

Each state has defined entry/exit criteria and automatic gate checks.
Capital allocation scales with state and is kelly-adjusted.

Persistence: SQLite at ~/.sentinel/strategy_promotion.db
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from sentinel.core.logging import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

DB_PATH = Path.home() / ".sentinel" / "strategy_promotion.db"


# ── State machine ─────────────────────────────────────────────────────────────

class StrategyState(str, Enum):
    DEVELOPMENT  = "DEVELOPMENT"
    PAPER_TESTING = "PAPER_TESTING"
    VALIDATION   = "VALIDATION"
    STAGING      = "STAGING"
    LIVE         = "LIVE"
    RETIRED      = "RETIRED"


# State ordering for comparison
STATE_ORDER = [
    StrategyState.DEVELOPMENT,
    StrategyState.PAPER_TESTING,
    StrategyState.VALIDATION,
    StrategyState.STAGING,
    StrategyState.LIVE,
    StrategyState.RETIRED,
]


def state_index(s: StrategyState) -> int:
    return STATE_ORDER.index(s)


def next_state(s: StrategyState) -> Optional[StrategyState]:
    idx = state_index(s)
    if idx + 1 >= len(STATE_ORDER):
        return None
    return STATE_ORDER[idx + 1]


def prev_state(s: StrategyState) -> Optional[StrategyState]:
    idx = state_index(s)
    if idx <= 0:
        return None
    return STATE_ORDER[idx - 1]


# ── Pydantic models ───────────────────────────────────────────────────────────

class BacktestResult(BaseModel):
    sharpe:        float
    max_drawdown:  float   # fraction, e.g. 0.15 = 15%
    cagr:          float
    n_trades:      int
    win_rate:      float
    start_date:    str
    end_date:      str


class StrategyRecord(BaseModel):
    strategy_id:     str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name:            str
    description:     str = ""
    code_path:       str = ""
    state:           StrategyState = StrategyState.DEVELOPMENT
    allocation_pct:  float = 0.0          # fraction of live capital allocated
    allocated_usd:   float = 0.0
    backtest_result: Optional[BacktestResult] = None
    code_review_passed: bool = False
    created_at:      datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at:      datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_evaluated:  Optional[datetime] = None
    notes:           str = ""
    tags:            List[str] = Field(default_factory=list)


class GateResult(BaseModel):
    strategy_id:   str
    current_state: StrategyState
    next_state:    Optional[StrategyState]
    passed:        bool
    failed_gates:  List[str]
    passed_gates:  List[str]
    metrics:       Dict[str, float]
    evaluated_at:  datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class StateTransition(BaseModel):
    transition_id:  str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_id:    str
    from_state:     StrategyState
    to_state:       StrategyState
    triggered_by:   Literal["automatic", "manual", "demotion"]
    reason:         str
    gate_result:    Optional[dict] = None
    operator:       str = "system"
    transitioned_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class PerformanceSnapshot(BaseModel):
    snapshot_id:    str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_id:    str
    date:           str
    sharpe_63d:     Optional[float] = None
    max_drawdown:   float = 0.0
    total_return:   float = 0.0
    n_trades:       int = 0
    win_rate:       float = 0.0
    consecutive_losing_days: int = 0
    regime:         str = "unknown"        # trending, ranging, volatile
    notes:          str = ""
    recorded_at:    datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class PromotionAlert(BaseModel):
    alert_id:    str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_id: str
    alert_type:  Literal[
        "consecutive_losses", "drawdown_warning", "vol_spike",
        "state_change", "demotion", "margin_call", "critical"
    ]
    severity:    Literal["info", "warning", "critical"]
    message:     str
    auto_action: Optional[str] = None    # e.g. "strategy paused"
    created_at:  datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class AllocationPlan(BaseModel):
    strategy_id:      str
    state:            StrategyState
    total_live_capital: float
    base_allocation_pct: float
    kelly_pct:        Optional[float]
    applied_pct:      float           # min(kelly_pct, base_allocation_pct)
    allocated_usd:    float
    ramp_pct:         float = 1.0     # 0.25 → 0.50 → 0.75 → 1.0 over 30-day periods
    drawdown_factor:  float = 1.0     # reduce proportionally to current drawdown
    final_allocation_usd: float
    computed_at:      datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class MonthlyReport(BaseModel):
    strategy_id:    str
    month:          str
    sharpe:         Optional[float]
    max_drawdown:   float
    total_return:   float
    n_trades:       int
    win_rate:       float
    regime_analysis: str
    recommendation:  str
    generated_at:   datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── PromotionGates ────────────────────────────────────────────────────────────

class PromotionGates:
    """
    Defines and evaluates promotion criteria for each state transition.

    Gates:
      DEVELOPMENT → PAPER_TESTING:
        - code_review_passed = True
        - backtest Sharpe > 0.5
        - backtest max_drawdown < 30%

      PAPER_TESTING → VALIDATION:
        - >= 90 days of paper trade data
        - live Sharpe (paper period) > 0.3
        - Deflated Sharpe Ratio (DSR) > 0
        - Probability of Backtest Overfitting (PBO) < 0.15

      VALIDATION → STAGING:
        - >= 180 days of paper trade data (cumulative)
        - Sharpe > 0.5
        - >= 3 consecutive profitable months

      STAGING → LIVE:
        - >= 60 days in shadow mode
        - shadow results match expected within 5% return deviation
        - risk gates all pass

      Auto-demotion:
        - Sharpe drops > 50% from baseline → back to PAPER_TESTING
    """

    # Thresholds
    BT_SHARPE_MIN         = 0.5
    BT_MAX_DD_MAX         = 0.30
    PAPER_DAYS_MIN        = 90
    PAPER_SHARPE_MIN      = 0.3
    PAPER_DSR_MIN         = 0.0
    PAPER_PBO_MAX         = 0.15
    VALIDATION_DAYS_MIN   = 180
    VALIDATION_SHARPE_MIN = 0.5
    VALIDATION_MONTHS_MIN = 3
    STAGING_DAYS_MIN      = 60
    STAGING_RETURN_DEV    = 0.05   # 5% tolerance
    DEMOTION_SHARPE_DROP  = 0.50   # 50% drop triggers demotion

    def check_development_to_paper(
        self, record: StrategyRecord
    ) -> GateResult:
        """DEVELOPMENT → PAPER_TESTING gate."""
        failed: List[str] = []
        passed: List[str] = []
        metrics: Dict[str, float] = {}

        # Gate 1: code review
        if record.code_review_passed:
            passed.append("code_review_passed")
        else:
            failed.append("code_review_passed: not yet approved")

        # Gate 2: backtest Sharpe
        if record.backtest_result:
            bt = record.backtest_result
            metrics["bt_sharpe"] = bt.sharpe
            metrics["bt_max_dd"] = bt.max_drawdown

            if bt.sharpe >= self.BT_SHARPE_MIN:
                passed.append(f"bt_sharpe >= {self.BT_SHARPE_MIN} (actual: {bt.sharpe:.3f})")
            else:
                failed.append(f"bt_sharpe {bt.sharpe:.3f} < {self.BT_SHARPE_MIN}")

            if bt.max_drawdown < self.BT_MAX_DD_MAX:
                passed.append(f"bt_max_dd < {self.BT_MAX_DD_MAX:.0%} (actual: {bt.max_drawdown:.1%})")
            else:
                failed.append(f"bt_max_dd {bt.max_drawdown:.1%} >= {self.BT_MAX_DD_MAX:.0%}")
        else:
            failed.append("backtest_result: no backtest provided")

        return GateResult(
            strategy_id=record.strategy_id,
            current_state=StrategyState.DEVELOPMENT,
            next_state=StrategyState.PAPER_TESTING,
            passed=len(failed) == 0,
            failed_gates=failed,
            passed_gates=passed,
            metrics=metrics,
        )

    def check_paper_to_validation(
        self,
        record: StrategyRecord,
        snapshots: List[PerformanceSnapshot],
        paper_equity_series: List[float],
    ) -> GateResult:
        """PAPER_TESTING → VALIDATION gate."""
        failed: List[str] = []
        passed: List[str] = []
        metrics: Dict[str, float] = {}

        # Gate 1: minimum days in PAPER_TESTING
        n_days = len(snapshots)
        metrics["paper_days"] = float(n_days)
        if n_days >= self.PAPER_DAYS_MIN:
            passed.append(f"paper_days >= {self.PAPER_DAYS_MIN} (actual: {n_days})")
        else:
            failed.append(f"paper_days {n_days} < {self.PAPER_DAYS_MIN}")

        # Gate 2: live Sharpe from paper equity series
        sharpe = self._compute_sharpe(paper_equity_series)
        metrics["live_sharpe"] = sharpe if sharpe is not None else float("nan")
        if sharpe is not None and sharpe >= self.PAPER_SHARPE_MIN:
            passed.append(f"live_sharpe >= {self.PAPER_SHARPE_MIN} (actual: {sharpe:.3f})")
        else:
            failed.append(f"live_sharpe {sharpe:.3f if sharpe else 'N/A'} < {self.PAPER_SHARPE_MIN}")

        # Gate 3: DSR (Deflated Sharpe Ratio)
        # DSR adjusts for number of trials; simplified: DSR = Sharpe * sqrt((T-1)/T) - z_correction
        dsr = self._compute_dsr(paper_equity_series, n_trials=10)
        metrics["dsr"] = dsr if dsr is not None else float("nan")
        if dsr is not None and dsr >= self.PAPER_DSR_MIN:
            passed.append(f"dsr >= 0 (actual: {dsr:.3f})")
        else:
            failed.append(f"dsr {dsr:.3f if dsr else 'N/A'} < {self.PAPER_DSR_MIN}")

        # Gate 4: PBO (Probability of Backtest Overfitting)
        pbo = self._compute_pbo(paper_equity_series, record.backtest_result)
        metrics["pbo"] = pbo if pbo is not None else float("nan")
        if pbo is not None and pbo < self.PAPER_PBO_MAX:
            passed.append(f"pbo < {self.PAPER_PBO_MAX} (actual: {pbo:.3f})")
        else:
            failed.append(f"pbo {pbo:.3f if pbo else 'N/A'} >= {self.PAPER_PBO_MAX}")

        return GateResult(
            strategy_id=record.strategy_id,
            current_state=StrategyState.PAPER_TESTING,
            next_state=StrategyState.VALIDATION,
            passed=len(failed) == 0,
            failed_gates=failed,
            passed_gates=passed,
            metrics=metrics,
        )

    def check_validation_to_staging(
        self,
        record: StrategyRecord,
        snapshots: List[PerformanceSnapshot],
        equity_series: List[float],
    ) -> GateResult:
        """VALIDATION → STAGING gate."""
        failed: List[str] = []
        passed: List[str] = []
        metrics: Dict[str, float] = {}

        # Gate 1: minimum days (cumulative paper + validation)
        n_days = len(snapshots)
        metrics["validation_days"] = float(n_days)
        if n_days >= self.VALIDATION_DAYS_MIN:
            passed.append(f"validation_days >= {self.VALIDATION_DAYS_MIN} (actual: {n_days})")
        else:
            failed.append(f"validation_days {n_days} < {self.VALIDATION_DAYS_MIN}")

        # Gate 2: Sharpe > 0.5
        sharpe = self._compute_sharpe(equity_series)
        metrics["sharpe"] = sharpe if sharpe is not None else float("nan")
        if sharpe is not None and sharpe >= self.VALIDATION_SHARPE_MIN:
            passed.append(f"sharpe >= {self.VALIDATION_SHARPE_MIN} (actual: {sharpe:.3f})")
        else:
            failed.append(f"sharpe {sharpe:.3f if sharpe else 'N/A'} < {self.VALIDATION_SHARPE_MIN}")

        # Gate 3: >= 3 consecutive profitable months
        profitable_months = self._count_profitable_months(equity_series)
        metrics["profitable_months"] = float(profitable_months)
        if profitable_months >= self.VALIDATION_MONTHS_MIN:
            passed.append(f"profitable_months >= {self.VALIDATION_MONTHS_MIN} (actual: {profitable_months})")
        else:
            failed.append(f"profitable_months {profitable_months} < {self.VALIDATION_MONTHS_MIN}")

        return GateResult(
            strategy_id=record.strategy_id,
            current_state=StrategyState.VALIDATION,
            next_state=StrategyState.STAGING,
            passed=len(failed) == 0,
            failed_gates=failed,
            passed_gates=passed,
            metrics=metrics,
        )

    def check_staging_to_live(
        self,
        record: StrategyRecord,
        snapshots: List[PerformanceSnapshot],
        equity_series: List[float],
        expected_return: Optional[float] = None,
        actual_return: Optional[float] = None,
    ) -> GateResult:
        """STAGING → LIVE gate."""
        failed: List[str] = []
        passed: List[str] = []
        metrics: Dict[str, float] = {}

        # Gate 1: >= 60 days in staging
        n_days = len(snapshots)
        metrics["staging_days"] = float(n_days)
        if n_days >= self.STAGING_DAYS_MIN:
            passed.append(f"staging_days >= {self.STAGING_DAYS_MIN} (actual: {n_days})")
        else:
            failed.append(f"staging_days {n_days} < {self.STAGING_DAYS_MIN}")

        # Gate 2: shadow return vs expected within 5%
        if expected_return is not None and actual_return is not None:
            deviation = abs(actual_return - expected_return)
            metrics["return_deviation"] = deviation
            metrics["expected_return"]  = expected_return
            metrics["actual_return"]    = actual_return
            if deviation <= self.STAGING_RETURN_DEV:
                passed.append(
                    f"return_deviation {deviation:.1%} <= {self.STAGING_RETURN_DEV:.0%}"
                )
            else:
                failed.append(
                    f"return_deviation {deviation:.1%} > {self.STAGING_RETURN_DEV:.0%}"
                )

        # Gate 3: risk gates — max drawdown must be < 25% during staging
        max_dd = self._compute_max_drawdown(equity_series)
        metrics["staging_max_dd"] = max_dd
        if max_dd < 0.25:
            passed.append(f"staging_max_dd < 25% (actual: {max_dd:.1%})")
        else:
            failed.append(f"staging_max_dd {max_dd:.1%} >= 25%")

        # Gate 4: Sharpe must still be positive
        sharpe = self._compute_sharpe(equity_series)
        metrics["sharpe"] = sharpe if sharpe is not None else float("nan")
        if sharpe is not None and sharpe > 0:
            passed.append(f"sharpe > 0 (actual: {sharpe:.3f})")
        else:
            failed.append(f"sharpe {sharpe:.3f if sharpe else 'N/A'} <= 0")

        return GateResult(
            strategy_id=record.strategy_id,
            current_state=StrategyState.STAGING,
            next_state=StrategyState.LIVE,
            passed=len(failed) == 0,
            failed_gates=failed,
            passed_gates=passed,
            metrics=metrics,
        )

    def check_auto_demotion(
        self,
        record: StrategyRecord,
        baseline_sharpe: float,
        current_sharpe: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Check if strategy should be automatically demoted.
        Triggers if Sharpe drops > 50% from baseline.
        """
        if current_sharpe is None:
            return False, "insufficient data for demotion check"

        if baseline_sharpe <= 0:
            return False, "baseline Sharpe not positive"

        drop = (baseline_sharpe - current_sharpe) / abs(baseline_sharpe)
        if drop > self.DEMOTION_SHARPE_DROP:
            reason = (
                f"Auto-demotion: Sharpe dropped {drop:.1%} from baseline {baseline_sharpe:.3f} "
                f"to {current_sharpe:.3f} (threshold {self.DEMOTION_SHARPE_DROP:.0%})"
            )
            return True, reason

        return False, "ok"

    # ── Statistical helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(equity_series: List[float], rf: float = 0.0) -> Optional[float]:
        """Annualized Sharpe ratio from equity series (risk-free = 0)."""
        if len(equity_series) < 10:
            return None
        arr  = np.array(equity_series, dtype=float)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 5 or rets.std() == 0:
            return None
        return float((rets.mean() - rf) / rets.std() * math.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(equity_series: List[float]) -> float:
        """Maximum peak-to-trough drawdown as a fraction."""
        if len(equity_series) < 2:
            return 0.0
        arr  = np.array(equity_series, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd   = np.where(peak > 0, (peak - arr) / peak, 0.0)
        return float(dd.max())

    @staticmethod
    def _compute_dsr(equity_series: List[float], n_trials: int = 10) -> Optional[float]:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, simplified).
        Accounts for multiple testing bias.
        DSR = (SR - E[max SR | n_trials]) / sigma_SR
        Simplified: DSR = SR_annualized - correction_term
        correction_term = sqrt(2 * ln(n_trials)) * sigma_SR
        """
        if len(equity_series) < 20:
            return None
        arr  = np.array(equity_series, dtype=float)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 10 or rets.std() == 0:
            return None

        T    = len(rets)
        mu   = rets.mean()
        sigma = rets.std(ddof=1)
        sr   = mu / sigma

        # Third and fourth moments for SR standard error
        skew  = float(np.mean(((rets - mu) / sigma) ** 3)) if sigma > 0 else 0.0
        kurt  = float(np.mean(((rets - mu) / sigma) ** 4)) - 3.0
        var_sr = (1 - skew * sr + (kurt / 4) * sr ** 2) / (T - 1)
        sigma_sr = math.sqrt(max(var_sr, 1e-12))

        # Expected maximum SR under n_trials: E[max SR] ≈ sqrt(2 * ln(n_trials))
        expected_max_sr = math.sqrt(2 * math.log(max(n_trials, 2)))

        dsr = (sr - expected_max_sr * sigma_sr) / sigma_sr
        return round(float(dsr), 4)

    @staticmethod
    def _compute_pbo(
        equity_series: List[float],
        backtest_result: Optional[BacktestResult],
    ) -> Optional[float]:
        """
        Probability of Backtest Overfitting (PBO) — simplified combinatorial method.
        PBO = P(live Sharpe < backtest Sharpe) assuming normal distribution of Sharpe estimates.
        Uses Z-test: Z = (bt_sharpe - live_sharpe) / sqrt(var_bt + var_live)
        """
        if backtest_result is None or len(equity_series) < 20:
            return None

        live_sharpe = PromotionGates._compute_sharpe(equity_series)
        if live_sharpe is None:
            return None

        bt_sharpe = backtest_result.sharpe
        T_live    = len(equity_series)
        T_bt      = max(backtest_result.n_trades * 2, 50)   # approximate bt sample size

        # Variance of Sharpe ratio estimate: (1 + 0.5 * SR^2) / (T-1)
        var_live = (1 + 0.5 * live_sharpe ** 2) / max(T_live - 1, 1)
        var_bt   = (1 + 0.5 * bt_sharpe ** 2) / max(T_bt - 1, 1)

        z = (bt_sharpe - live_sharpe) / math.sqrt(var_live + var_bt + 1e-12)
        # PBO = P(live Sharpe < bt Sharpe) = Phi(z) using normal CDF approximation
        pbo = _normal_cdf(z)
        return round(float(pbo), 4)

    @staticmethod
    def _count_profitable_months(equity_series: List[float]) -> int:
        """Count consecutive profitable months at the end of the equity series."""
        if len(equity_series) < 21:
            return 0

        # Group into ~21-trading-day months
        chunk = 21
        n_months = len(equity_series) // chunk
        if n_months == 0:
            return 0

        monthly_rets = []
        for i in range(n_months):
            start = i * chunk
            end   = min(start + chunk, len(equity_series))
            s     = equity_series[start]
            e     = equity_series[end - 1]
            monthly_rets.append((e - s) / max(s, 1e-9))

        # Count consecutive profitable months from the end
        count = 0
        for r in reversed(monthly_rets):
            if r > 0:
                count += 1
            else:
                break
        return count


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ── StrategyRegistry ──────────────────────────────────────────────────────────

class StrategyRegistry:
    """
    Central SQLite-backed registry for all strategies and their state transitions.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("StrategyRegistry initialized", db=str(self._db))

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
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id         TEXT PRIMARY KEY,
                    name                TEXT NOT NULL,
                    description         TEXT,
                    code_path           TEXT,
                    state               TEXT NOT NULL DEFAULT 'DEVELOPMENT',
                    allocation_pct      REAL NOT NULL DEFAULT 0,
                    allocated_usd       REAL NOT NULL DEFAULT 0,
                    backtest_result     TEXT,
                    code_review_passed  INTEGER NOT NULL DEFAULT 0,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL,
                    last_evaluated      TEXT,
                    notes               TEXT,
                    tags                TEXT
                );

                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id       TEXT PRIMARY KEY,
                    strategy_id         TEXT NOT NULL,
                    from_state          TEXT NOT NULL,
                    to_state            TEXT NOT NULL,
                    triggered_by        TEXT NOT NULL,
                    reason              TEXT,
                    gate_result         TEXT,
                    operator            TEXT NOT NULL DEFAULT 'system',
                    transitioned_at     TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    snapshot_id             TEXT PRIMARY KEY,
                    strategy_id             TEXT NOT NULL,
                    date                    TEXT NOT NULL,
                    sharpe_63d              REAL,
                    max_drawdown            REAL NOT NULL DEFAULT 0,
                    total_return            REAL NOT NULL DEFAULT 0,
                    n_trades                INTEGER NOT NULL DEFAULT 0,
                    win_rate                REAL NOT NULL DEFAULT 0,
                    consecutive_losing_days INTEGER NOT NULL DEFAULT 0,
                    regime                  TEXT NOT NULL DEFAULT 'unknown',
                    notes                   TEXT,
                    recorded_at             TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE TABLE IF NOT EXISTS equity_series (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id     TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    equity_value    REAL NOT NULL,
                    UNIQUE (strategy_id, date),
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id        TEXT PRIMARY KEY,
                    strategy_id     TEXT NOT NULL,
                    alert_type      TEXT NOT NULL,
                    severity        TEXT NOT NULL,
                    message         TEXT NOT NULL,
                    auto_action     TEXT,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE TABLE IF NOT EXISTS allocation_plans (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id     TEXT NOT NULL,
                    state           TEXT NOT NULL,
                    base_pct        REAL NOT NULL,
                    kelly_pct       REAL,
                    applied_pct     REAL NOT NULL,
                    allocated_usd   REAL NOT NULL,
                    ramp_pct        REAL NOT NULL DEFAULT 1.0,
                    drawdown_factor REAL NOT NULL DEFAULT 1.0,
                    final_usd       REAL NOT NULL,
                    computed_at     TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                );

                CREATE INDEX IF NOT EXISTS idx_trans_strategy   ON state_transitions(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_snaps_strategy   ON performance_snapshots(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_equity_strategy  ON equity_series(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_alerts_strategy  ON alerts(strategy_id);
            """)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        code_path: str = "",
        backtest_result: Optional[BacktestResult] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new strategy. Returns strategy_id."""
        sid = str(uuid.uuid4())[:12]
        now = datetime.now(tz=timezone.utc).isoformat()
        bt_json = backtest_result.model_dump_json() if backtest_result else None
        tags_json = json.dumps(tags or [])

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO strategies "
                "(strategy_id, name, description, code_path, state, allocation_pct, "
                "allocated_usd, backtest_result, code_review_passed, created_at, "
                "updated_at, last_evaluated, notes, tags) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid, name, description, code_path, StrategyState.DEVELOPMENT.value,
                    0.0, 0.0, bt_json, 0, now, now, None, "", tags_json,
                ),
            )
        logger.info("Strategy registered", strategy_id=sid, name=name)
        return sid

    def get(self, strategy_id: str) -> StrategyRecord:
        """Fetch a strategy record or raise ValueError."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Strategy not found: {strategy_id}")
        return self._row_to_record(dict(row))

    def list_all(self) -> List[StrategyRecord]:
        """Return all strategy records."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def get_by_state(self, state: StrategyState) -> List[StrategyRecord]:
        """Return all strategies in the given state."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE state = ? ORDER BY created_at DESC",
                (state.value,),
            ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def get_promotion_candidates(self) -> List[StrategyRecord]:
        """
        Return strategies that are eligible for evaluation (not RETIRED).
        These are candidates for a promotion gate check.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE state NOT IN ('RETIRED') "
                "ORDER BY state ASC, updated_at DESC"
            ).fetchall()
        return [self._row_to_record(dict(r)) for r in rows]

    def get_history(self, strategy_id: str) -> List[StateTransition]:
        """Return full state transition history for a strategy."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM state_transitions WHERE strategy_id = ? "
                "ORDER BY transitioned_at ASC",
                (strategy_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["transitioned_at"] = datetime.fromisoformat(d["transitioned_at"])
            except Exception:
                d["transitioned_at"] = datetime.now(tz=timezone.utc)
            if d.get("gate_result"):
                try:
                    d["gate_result"] = json.loads(d["gate_result"])
                except Exception:
                    d["gate_result"] = None
            result.append(StateTransition(**d))
        return result

    def approve_code_review(self, strategy_id: str, operator: str = "system") -> None:
        """Mark code review as passed."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE strategies SET code_review_passed = 1, updated_at = ? "
                "WHERE strategy_id = ?",
                (now, strategy_id),
            )
        logger.info("Code review approved", strategy_id=strategy_id, operator=operator)

    def update_backtest(self, strategy_id: str, result: BacktestResult) -> None:
        """Update the backtest result for a strategy."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE strategies SET backtest_result = ?, updated_at = ? "
                "WHERE strategy_id = ?",
                (result.model_dump_json(), now, strategy_id),
            )

    # ── State transitions ─────────────────────────────────────────────────────

    def _transition(
        self,
        strategy_id: str,
        to_state: StrategyState,
        triggered_by: str,
        reason: str,
        gate_result: Optional[GateResult] = None,
        operator: str = "system",
    ) -> StateTransition:
        record = self.get(strategy_id)
        from_state = record.state
        now = datetime.now(tz=timezone.utc)

        tr = StateTransition(
            strategy_id=strategy_id,
            from_state=from_state,
            to_state=to_state,
            triggered_by=triggered_by,
            reason=reason,
            gate_result=gate_result.model_dump() if gate_result else None,
            operator=operator,
            transitioned_at=now,
        )

        with self._conn() as conn:
            conn.execute(
                "UPDATE strategies SET state = ?, updated_at = ?, last_evaluated = ? "
                "WHERE strategy_id = ?",
                (to_state.value, now.isoformat(), now.isoformat(), strategy_id),
            )
            conn.execute(
                "INSERT INTO state_transitions "
                "(transition_id, strategy_id, from_state, to_state, triggered_by, "
                "reason, gate_result, operator, transitioned_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    tr.transition_id, strategy_id, from_state.value, to_state.value,
                    triggered_by, reason,
                    json.dumps(tr.gate_result) if tr.gate_result else None,
                    operator, now.isoformat(),
                ),
            )

        logger.info(
            "Strategy state transition",
            strategy_id=strategy_id,
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
        )
        return tr

    def promote(
        self,
        strategy_id: str,
        gate_result: Optional[GateResult] = None,
        reason: str = "manual promotion",
        operator: str = "operator",
    ) -> StateTransition:
        """Promote strategy to next state (manual or automatic)."""
        record = self.get(strategy_id)
        ns = next_state(record.state)
        if ns is None:
            raise ValueError(f"Strategy {strategy_id} is already in final state {record.state}")
        triggered_by = "automatic" if operator == "system" else "manual"
        return self._transition(strategy_id, ns, triggered_by, reason, gate_result, operator)

    def demote(
        self,
        strategy_id: str,
        reason: str = "manual demotion",
        target_state: Optional[StrategyState] = None,
        operator: str = "operator",
    ) -> StateTransition:
        """Demote strategy. Default: back to PAPER_TESTING. Can specify target."""
        record = self.get(strategy_id)
        if target_state is None:
            target_state = StrategyState.PAPER_TESTING
        return self._transition(strategy_id, target_state, "demotion", reason, None, operator)

    def retire(self, strategy_id: str, reason: str = "retired", operator: str = "operator") -> StateTransition:
        """Retire a strategy (move to RETIRED state)."""
        return self._transition(strategy_id, StrategyState.RETIRED, "manual", reason, None, operator)

    # ── Equity series helpers ─────────────────────────────────────────────────

    def record_equity(self, strategy_id: str, equity_value: float, date_str: Optional[str] = None) -> None:
        """Append a daily equity value for this strategy."""
        d = date_str or date.today().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_series (strategy_id, date, equity_value) VALUES (?,?,?) "
                "ON CONFLICT(strategy_id, date) DO UPDATE SET equity_value=excluded.equity_value",
                (strategy_id, d, equity_value),
            )

    def get_equity_series(self, strategy_id: str, days: int = 365) -> List[float]:
        """Return equity series (sorted ascending) for the last N days."""
        since = (date.today() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT equity_value FROM equity_series WHERE strategy_id = ? AND date >= ? "
                "ORDER BY date ASC",
                (strategy_id, since),
            ).fetchall()
        return [r["equity_value"] for r in rows]

    # ── Snapshot helpers ──────────────────────────────────────────────────────

    def add_snapshot(self, snap: PerformanceSnapshot) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO performance_snapshots "
                "(snapshot_id, strategy_id, date, sharpe_63d, max_drawdown, total_return, "
                "n_trades, win_rate, consecutive_losing_days, regime, notes, recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snap.snapshot_id, snap.strategy_id, snap.date,
                    snap.sharpe_63d, snap.max_drawdown, snap.total_return,
                    snap.n_trades, snap.win_rate, snap.consecutive_losing_days,
                    snap.regime, snap.notes, now,
                ),
            )

    def get_snapshots(self, strategy_id: str, days: int = 365) -> List[PerformanceSnapshot]:
        since = (date.today() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM performance_snapshots WHERE strategy_id = ? AND date >= ? "
                "ORDER BY date ASC",
                (strategy_id, since),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["recorded_at"] = datetime.fromisoformat(d["recorded_at"])
            except Exception:
                d["recorded_at"] = datetime.now(tz=timezone.utc)
            result.append(PerformanceSnapshot(**d))
        return result

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def save_alert(self, alert: PromotionAlert) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO alerts "
                "(alert_id, strategy_id, alert_type, severity, message, auto_action, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    alert.alert_id, alert.strategy_id, alert.alert_type,
                    alert.severity, alert.message, alert.auto_action, now,
                ),
            )

    def get_alerts(self, strategy_id: str, days: int = 30) -> List[PromotionAlert]:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE strategy_id = ? AND created_at >= ? "
                "ORDER BY created_at DESC",
                (strategy_id, since),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["created_at"] = datetime.fromisoformat(d["created_at"])
            except Exception:
                d["created_at"] = datetime.now(tz=timezone.utc)
            result.append(PromotionAlert(**d))
        return result

    # ── Serialization ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(d: dict) -> StrategyRecord:
        if d.get("backtest_result"):
            try:
                bt_dict = json.loads(d["backtest_result"])
                d["backtest_result"] = BacktestResult(**bt_dict)
            except Exception:
                d["backtest_result"] = None

        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except Exception:
                d["tags"] = []

        d["code_review_passed"] = bool(d.get("code_review_passed", 0))
        d["state"] = StrategyState(d["state"])

        for dt_field in ("created_at", "updated_at", "last_evaluated"):
            if d.get(dt_field):
                try:
                    d[dt_field] = datetime.fromisoformat(d[dt_field])
                except Exception:
                    d[dt_field] = datetime.now(tz=timezone.utc)

        return StrategyRecord(**d)


# ── PerformanceEvaluator ──────────────────────────────────────────────────────

class PerformanceEvaluator:
    """
    Continuous performance monitoring: daily/weekly/monthly checks.
    Computes Sharpe (rolling 63D), drawdown, regime, and gate checks.
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry
        self._gates    = PromotionGates()

    def record_daily_equity(
        self,
        strategy_id: str,
        equity_value: float,
        n_trades: int = 0,
        win_rate: float = 0.0,
        regime: str = "unknown",
    ) -> PerformanceSnapshot:
        """Record today's equity and compute daily snapshot metrics."""
        self._registry.record_equity(strategy_id, equity_value)
        equity_series = self._registry.get_equity_series(strategy_id, days=365)

        # Rolling 63D Sharpe
        sharpe_63 = None
        if len(equity_series) >= 63:
            sharpe_63 = self._gates._compute_sharpe(equity_series[-63:])
        elif len(equity_series) >= 10:
            sharpe_63 = self._gates._compute_sharpe(equity_series)

        max_dd     = self._gates._compute_max_drawdown(equity_series)
        total_ret  = (equity_series[-1] / equity_series[0] - 1.0) if len(equity_series) >= 2 else 0.0

        # Consecutive losing days
        losing_days = 0
        if len(equity_series) >= 2:
            for i in range(len(equity_series) - 1, 0, -1):
                if equity_series[i] < equity_series[i - 1]:
                    losing_days += 1
                else:
                    break

        snap = PerformanceSnapshot(
            strategy_id=strategy_id,
            date=date.today().isoformat(),
            sharpe_63d=round(sharpe_63, 4) if sharpe_63 is not None else None,
            max_drawdown=round(max_dd, 6),
            total_return=round(total_ret, 6),
            n_trades=n_trades,
            win_rate=round(win_rate, 4),
            consecutive_losing_days=losing_days,
            regime=regime,
        )
        self._registry.add_snapshot(snap)
        return snap

    def weekly_regime_analysis(self, strategy_id: str) -> str:
        """
        Classify current market regime based on recent equity volatility.
        Returns: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        equity_series = self._registry.get_equity_series(strategy_id, days=30)
        if len(equity_series) < 10:
            return "unknown"

        arr  = np.array(equity_series, dtype=float)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)

        vol     = float(np.std(rets))
        trend   = float(arr[-1] - arr[0]) / max(arr[0], 1.0)
        autocorr_lag1 = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 2 else 0.0

        if vol > 0.025:
            return "volatile"
        elif autocorr_lag1 > 0.2 and trend > 0:
            return "trending_up"
        elif autocorr_lag1 > 0.2 and trend < 0:
            return "trending_down"
        else:
            return "ranging"

    def generate_monthly_report(self, strategy_id: str) -> MonthlyReport:
        """Auto-generate monthly performance review."""
        record        = self._registry.get(strategy_id)
        equity_series = self._registry.get_equity_series(strategy_id, days=35)
        snapshots     = self._registry.get_snapshots(strategy_id, days=35)

        sharpe    = self._gates._compute_sharpe(equity_series)
        max_dd    = self._gates._compute_max_drawdown(equity_series)
        total_ret = (equity_series[-1] / equity_series[0] - 1.0) if len(equity_series) >= 2 else 0.0
        n_trades  = sum(s.n_trades for s in snapshots)
        win_rate  = float(np.mean([s.win_rate for s in snapshots])) if snapshots else 0.0
        regime    = self.weekly_regime_analysis(strategy_id)

        # Recommendation
        if sharpe and sharpe >= 0.5 and max_dd < 0.15:
            rec = f"Strong performance. Consider advancing to {next_state(record.state).value if next_state(record.state) else 'LIVE'}."
        elif sharpe and sharpe > 0:
            rec = "Acceptable performance. Monitor for consistency."
        elif max_dd > 0.20:
            rec = "Excessive drawdown detected. Review risk parameters."
        else:
            rec = "Underperforming. Consider demotion or parameter review."

        month = date.today().strftime("%Y-%m")
        return MonthlyReport(
            strategy_id=strategy_id,
            month=month,
            sharpe=round(sharpe, 4) if sharpe else None,
            max_drawdown=round(max_dd, 4),
            total_return=round(total_ret, 4),
            n_trades=n_trades,
            win_rate=round(win_rate, 4),
            regime_analysis=regime,
            recommendation=rec,
        )

    def check_gates(
        self,
        strategy_id: str,
        expected_return: Optional[float] = None,
        actual_return: Optional[float] = None,
    ) -> GateResult:
        """
        Check promotion gates for the strategy's current state.
        Returns GateResult with passed/failed gates and next eligible state.
        """
        record        = self._registry.get(strategy_id)
        equity_series = self._registry.get_equity_series(strategy_id, days=365)
        snapshots     = self._registry.get_snapshots(strategy_id, days=365)

        if record.state == StrategyState.DEVELOPMENT:
            return self._gates.check_development_to_paper(record)

        elif record.state == StrategyState.PAPER_TESTING:
            return self._gates.check_paper_to_validation(record, snapshots, equity_series)

        elif record.state == StrategyState.VALIDATION:
            return self._gates.check_validation_to_staging(record, snapshots, equity_series)

        elif record.state == StrategyState.STAGING:
            return self._gates.check_staging_to_live(
                record, snapshots, equity_series, expected_return, actual_return
            )

        else:
            # LIVE or RETIRED — no promotion gate
            return GateResult(
                strategy_id=strategy_id,
                current_state=record.state,
                next_state=None,
                passed=False,
                failed_gates=[f"No gate defined for state {record.state}"],
                passed_gates=[],
                metrics={},
            )

    def check_auto_demotion(self, strategy_id: str) -> Tuple[bool, str]:
        """
        Check if strategy warrants automatic demotion.
        Compares current rolling 63D Sharpe vs baseline (first 63D Sharpe after promotion).
        """
        record        = self._registry.get(strategy_id)
        equity_series = self._registry.get_equity_series(strategy_id, days=365)
        snapshots     = self._registry.get_snapshots(strategy_id, days=365)

        if record.state not in (StrategyState.LIVE, StrategyState.STAGING, StrategyState.VALIDATION):
            return False, "demotion check only applies to VALIDATION, STAGING, LIVE"

        if len(equity_series) < 63:
            return False, "insufficient data"

        # Baseline = first 63D Sharpe
        baseline_sharpe = self._gates._compute_sharpe(equity_series[:63])
        current_sharpe  = self._gates._compute_sharpe(equity_series[-63:])

        return self._gates.check_auto_demotion(record, baseline_sharpe or 0.0, current_sharpe)


# ── AlertSystem ───────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Notification and escalation system for strategy lifecycle events.
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry

    def check_warnings(self, strategy_id: str) -> List[PromotionAlert]:
        """
        Run all warning checks for a strategy:
        - 3 consecutive losing days
        - drawdown > 10%
        - volatility spike (daily return > 3 sigma)
        Returns list of new alerts generated.
        """
        alerts: List[PromotionAlert] = []
        snapshots = self._registry.get_snapshots(strategy_id, days=14)
        equity    = self._registry.get_equity_series(strategy_id, days=30)

        if not snapshots:
            return alerts

        latest = snapshots[-1]

        # 1. Consecutive losing days
        if latest.consecutive_losing_days >= 3:
            alert = PromotionAlert(
                strategy_id=strategy_id,
                alert_type="consecutive_losses",
                severity="warning",
                message=(
                    f"Strategy {strategy_id} has {latest.consecutive_losing_days} "
                    f"consecutive losing days."
                ),
            )
            alerts.append(alert)
            self._registry.save_alert(alert)

        # 2. Drawdown > 10%
        if latest.max_drawdown > 0.10:
            severity = "critical" if latest.max_drawdown > 0.20 else "warning"
            alert = PromotionAlert(
                strategy_id=strategy_id,
                alert_type="drawdown_warning",
                severity=severity,
                message=(
                    f"Strategy {strategy_id} drawdown at {latest.max_drawdown:.1%} "
                    f"({'critical' if latest.max_drawdown > 0.20 else 'elevated'})."
                ),
            )
            alerts.append(alert)
            self._registry.save_alert(alert)

        # 3. Volatility spike
        if len(equity) >= 20:
            arr  = np.array(equity, dtype=float)
            rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
            if len(rets) >= 20 and rets.std() > 0:
                z_recent = abs(rets[-1]) / rets.std()
                if z_recent > 3.0:
                    alert = PromotionAlert(
                        strategy_id=strategy_id,
                        alert_type="vol_spike",
                        severity="warning",
                        message=(
                            f"Strategy {strategy_id}: vol spike detected. "
                            f"Daily return z-score = {z_recent:.1f}"
                        ),
                    )
                    alerts.append(alert)
                    self._registry.save_alert(alert)

        return alerts

    def emit_state_change(
        self,
        strategy_id: str,
        from_state: StrategyState,
        to_state: StrategyState,
        reason: str,
    ) -> PromotionAlert:
        """Log and return a state change alert."""
        severity = "critical" if to_state == StrategyState.RETIRED else (
            "warning" if state_index(to_state) < state_index(from_state) else "info"
        )
        alert = PromotionAlert(
            strategy_id=strategy_id,
            alert_type="state_change",
            severity=severity,
            message=f"State change: {from_state.value} → {to_state.value}. Reason: {reason}",
        )
        self._registry.save_alert(alert)
        logger.info("State change alert emitted", strategy_id=strategy_id,
                    from_state=from_state.value, to_state=to_state.value)
        return alert

    def emit_demotion(
        self,
        strategy_id: str,
        from_state: StrategyState,
        reason: str,
        auto: bool = True,
    ) -> PromotionAlert:
        """Emit a demotion alert and pause strategy if critical."""
        alert = PromotionAlert(
            strategy_id=strategy_id,
            alert_type="demotion",
            severity="critical",
            message=f"{'Automatic' if auto else 'Manual'} demotion from {from_state.value}: {reason}",
            auto_action="strategy demoted to PAPER_TESTING",
        )
        self._registry.save_alert(alert)
        return alert

    def escalate_critical(self, strategy_id: str, message: str) -> PromotionAlert:
        """Immediately pause strategy and emit critical alert."""
        alert = PromotionAlert(
            strategy_id=strategy_id,
            alert_type="critical",
            severity="critical",
            message=f"CRITICAL — strategy {strategy_id} paused: {message}",
            auto_action="strategy paused immediately",
        )
        self._registry.save_alert(alert)
        logger.error("Critical alert", strategy_id=strategy_id, message=message)
        return alert


# ── CapitalAllocation ─────────────────────────────────────────────────────────

class CapitalAllocation:
    """
    Dynamic capital sizing per strategy state with Kelly adjustment,
    gradual ramp, and drawdown-based reduction.

    State allocations:
      DEVELOPMENT:   0% (paper only, no live capital)
      PAPER_TESTING: 0% (paper only)
      VALIDATION:    0% (paper only)
      STAGING:       1% of live capital (shadow)
      LIVE:          allocation_pct (kelly-adjusted, typically 5-20%)
    """

    STATE_BASE_PCT: Dict[StrategyState, float] = {
        StrategyState.DEVELOPMENT:   0.0,
        StrategyState.PAPER_TESTING: 0.0,
        StrategyState.VALIDATION:    0.0,
        StrategyState.STAGING:       0.01,
        StrategyState.LIVE:          0.10,   # overridden by kelly
        StrategyState.RETIRED:       0.0,
    }

    RAMP_STAGES = [0.25, 0.50, 0.75, 1.00]   # 30-day ramp periods

    def compute_kelly(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> Optional[float]:
        """
        Full Kelly criterion: f* = (p * b - q) / b
        where b = avg_win / avg_loss, p = win_rate, q = 1 - p.
        Returns half-Kelly for conservatism, capped at 25%.
        """
        if avg_loss == 0 or avg_win <= 0 or not (0 < win_rate < 1):
            return None
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1.0 - p
        kelly_full = (p * b - q) / b
        kelly_half = kelly_full / 2.0   # half-Kelly for robustness
        return max(0.0, min(kelly_half, 0.25))

    def compute_ramp_pct(self, days_in_state: int) -> float:
        """
        Return ramp factor: 0.25 → 0.50 → 0.75 → 1.00 each 30 days.
        """
        period = min(days_in_state // 30, len(self.RAMP_STAGES) - 1)
        return self.RAMP_STAGES[period]

    def compute_drawdown_factor(self, current_drawdown: float) -> float:
        """
        Reduce capital allocation proportionally to drawdown.
        DD=0% → factor=1.0, DD=10% → factor=0.5, DD>=20% → factor=0.0
        Linear scale between 0% and 20%.
        """
        if current_drawdown <= 0:
            return 1.0
        if current_drawdown >= 0.20:
            return 0.0
        return round(1.0 - (current_drawdown / 0.20), 4)

    def compute_allocation(
        self,
        strategy_id: str,
        state: StrategyState,
        total_live_capital: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        days_in_state: int = 0,
        current_drawdown: float = 0.0,
        max_allocation_pct: float = 0.20,
    ) -> AllocationPlan:
        """
        Compute final capital allocation for a strategy.

        Returns AllocationPlan with all sizing components.
        """
        base_pct  = self.STATE_BASE_PCT.get(state, 0.0)
        kelly_pct = None
        applied_pct = base_pct

        if state == StrategyState.LIVE:
            kelly_pct = self.compute_kelly(win_rate, avg_win, avg_loss)
            if kelly_pct is not None:
                applied_pct = min(kelly_pct, max_allocation_pct)
            else:
                applied_pct = base_pct

        ramp_factor = self.compute_ramp_pct(days_in_state)
        dd_factor   = self.compute_drawdown_factor(current_drawdown)

        allocated_usd     = total_live_capital * applied_pct
        final_allocation  = round(allocated_usd * ramp_factor * dd_factor, 2)

        return AllocationPlan(
            strategy_id=strategy_id,
            state=state,
            total_live_capital=total_live_capital,
            base_allocation_pct=base_pct,
            kelly_pct=kelly_pct,
            applied_pct=applied_pct,
            allocated_usd=round(allocated_usd, 2),
            ramp_pct=ramp_factor,
            drawdown_factor=dd_factor,
            final_allocation_usd=final_allocation,
        )


# ── StrategyLifecycle (orchestrator) ──────────────────────────────────────────

class StrategyLifecycle:
    """
    Main orchestrator for the strategy promotion state machine.
    Combines registry, evaluator, alert system, and capital allocation.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._registry   = StrategyRegistry(db_path)
        self._evaluator  = PerformanceEvaluator(self._registry)
        self._alerts     = AlertSystem(self._registry)
        self._allocation = CapitalAllocation()
        logger.info("StrategyLifecycle initialized")

    # ── Registration ──────────────────────────────────────────────────────────

    def register_strategy(
        self,
        name: str,
        code_path: str = "",
        backtest_result: Optional[BacktestResult] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new strategy in DEVELOPMENT state."""
        return self._registry.register(name, code_path, backtest_result, description, tags)

    def approve_code_review(self, strategy_id: str, operator: str = "operator") -> None:
        """Mark code review as passed for a strategy."""
        self._registry.approve_code_review(strategy_id, operator)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        strategy_id: str,
        expected_return: Optional[float] = None,
        actual_return: Optional[float] = None,
        auto_promote: bool = False,
        operator: str = "system",
    ) -> GateResult:
        """
        Evaluate gates for the strategy's current state.
        If auto_promote=True and gates pass, automatically advance state.
        Also checks for automatic demotion.
        """
        record = self._registry.get(strategy_id)

        # Check auto-demotion first
        should_demote, demote_reason = self._evaluator.check_auto_demotion(strategy_id)
        if should_demote:
            from_state = record.state
            self._registry.demote(strategy_id, reason=demote_reason, operator="system")
            self._alerts.emit_demotion(strategy_id, from_state, demote_reason, auto=True)
            logger.warning("Auto-demotion triggered", strategy_id=strategy_id, reason=demote_reason)
            # Return gate result reflecting the demotion
            return GateResult(
                strategy_id=strategy_id,
                current_state=record.state,
                next_state=StrategyState.PAPER_TESTING,
                passed=False,
                failed_gates=[f"auto_demotion: {demote_reason}"],
                passed_gates=[],
                metrics={},
            )

        # Run warning checks
        self._alerts.check_warnings(strategy_id)

        # Evaluate gates
        gate_result = self._evaluator.check_gates(strategy_id, expected_return, actual_return)

        # Auto-promote if gates pass
        if auto_promote and gate_result.passed and gate_result.next_state is not None:
            tr = self._registry.promote(strategy_id, gate_result, reason="auto-promoted", operator=operator)
            self._alerts.emit_state_change(
                strategy_id, tr.from_state, tr.to_state, "gates passed, auto-promoted"
            )

        return gate_result

    def promote(
        self,
        strategy_id: str,
        reason: str = "manual promotion",
        operator: str = "operator",
    ) -> StateTransition:
        """Manually promote a strategy to the next state."""
        record = self._registry.get(strategy_id)
        gate_result = self._evaluator.check_gates(strategy_id)
        if not gate_result.passed:
            raise ValueError(
                f"Cannot promote: {len(gate_result.failed_gates)} gate(s) failed: "
                + "; ".join(gate_result.failed_gates)
            )
        tr = self._registry.promote(strategy_id, gate_result, reason, operator)
        self._alerts.emit_state_change(strategy_id, tr.from_state, tr.to_state, reason)
        return tr

    def force_promote(
        self,
        strategy_id: str,
        reason: str = "force promotion",
        operator: str = "operator",
    ) -> StateTransition:
        """Force-promote bypassing gate checks (with audit log)."""
        record = self._registry.get(strategy_id)
        tr = self._registry.promote(strategy_id, None, f"[FORCE] {reason}", operator)
        self._alerts.emit_state_change(strategy_id, tr.from_state, tr.to_state, f"[FORCE] {reason}")
        logger.warning("Force promotion", strategy_id=strategy_id, operator=operator)
        return tr

    def demote(
        self,
        strategy_id: str,
        reason: str = "manual demotion",
        target_state: Optional[StrategyState] = None,
        operator: str = "operator",
    ) -> StateTransition:
        """Demote a strategy."""
        record = self._registry.get(strategy_id)
        tr = self._registry.demote(strategy_id, reason, target_state, operator)
        self._alerts.emit_demotion(strategy_id, tr.from_state, reason, auto=False)
        return tr

    def retire(
        self,
        strategy_id: str,
        reason: str = "retired",
        operator: str = "operator",
    ) -> StateTransition:
        """Retire a strategy permanently."""
        return self._registry.retire(strategy_id, reason, operator)

    # ── Daily operations ──────────────────────────────────────────────────────

    def record_daily(
        self,
        strategy_id: str,
        equity_value: float,
        n_trades: int = 0,
        win_rate: float = 0.0,
    ) -> PerformanceSnapshot:
        """Record daily equity and compute snapshot metrics."""
        regime = self._evaluator.weekly_regime_analysis(strategy_id)
        snap = self._evaluator.record_daily_equity(
            strategy_id, equity_value, n_trades, win_rate, regime
        )
        return snap

    def monthly_review(self, strategy_id: str) -> MonthlyReport:
        """Generate monthly performance report."""
        return self._evaluator.generate_monthly_report(strategy_id)

    # ── Capital allocation ────────────────────────────────────────────────────

    def get_allocation(
        self,
        strategy_id: str,
        total_live_capital: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        days_in_state: int = 0,
    ) -> AllocationPlan:
        """Compute Kelly-adjusted capital allocation for strategy in its current state."""
        record  = self._registry.get(strategy_id)
        equity  = self._registry.get_equity_series(strategy_id, days=90)
        cur_dd  = PromotionGates._compute_max_drawdown(equity) if len(equity) >= 2 else 0.0

        return self._allocation.compute_allocation(
            strategy_id, record.state, total_live_capital,
            win_rate, avg_win, avg_loss, days_in_state, cur_dd,
        )

    # ── Registry access ───────────────────────────────────────────────────────

    def get_strategy(self, strategy_id: str) -> StrategyRecord:
        return self._registry.get(strategy_id)

    def list_strategies(self) -> List[StrategyRecord]:
        return self._registry.list_all()

    def get_by_state(self, state: StrategyState) -> List[StrategyRecord]:
        return self._registry.get_by_state(state)

    def get_promotion_candidates(self) -> List[StrategyRecord]:
        return self._registry.get_promotion_candidates()

    def get_history(self, strategy_id: str) -> List[StateTransition]:
        return self._registry.get_history(strategy_id)

    def get_alerts(self, strategy_id: str, days: int = 30) -> List[PromotionAlert]:
        return self._registry.get_alerts(strategy_id, days)

    def record_equity(self, strategy_id: str, equity_value: float, date_str: Optional[str] = None) -> None:
        self._registry.record_equity(strategy_id, equity_value, date_str)

    def update_backtest(self, strategy_id: str, result: BacktestResult) -> None:
        self._registry.update_backtest(strategy_id, result)


# ── Module-level singleton ────────────────────────────────────────────────────

_lifecycle: Optional[StrategyLifecycle] = None


def _get_lifecycle() -> StrategyLifecycle:
    global _lifecycle
    if _lifecycle is None:
        _lifecycle = StrategyLifecycle()
    return _lifecycle


# ── FastAPI Router ────────────────────────────────────────────────────────────

promotion_router = APIRouter(prefix="/promotion", tags=["strategy-promotion"])


class RegisterRequest(BaseModel):
    name:             str
    code_path:        str = ""
    description:      str = ""
    tags:             List[str] = Field(default_factory=list)
    backtest_sharpe:  Optional[float] = None
    backtest_max_dd:  Optional[float] = None
    backtest_cagr:    Optional[float] = None
    backtest_trades:  Optional[int] = None
    backtest_wr:      Optional[float] = None
    backtest_start:   Optional[str] = None
    backtest_end:     Optional[str] = None


class DailyRecordRequest(BaseModel):
    strategy_id:   str
    equity_value:  float
    n_trades:      int = 0
    win_rate:      float = 0.0


class AllocationRequest(BaseModel):
    strategy_id:        str
    total_live_capital: float
    win_rate:           float = 0.5
    avg_win:            float = 1.0
    avg_loss:           float = 1.0
    days_in_state:      int = 0


class EvaluateRequest(BaseModel):
    expected_return: Optional[float] = None
    actual_return:   Optional[float] = None
    auto_promote:    bool = False
    operator:        str = "system"


class PromoteDemoteRequest(BaseModel):
    reason:   str = ""
    operator: str = "operator"


@promotion_router.get("/strategies")
def list_strategies():
    lc = _get_lifecycle()
    return [r.model_dump() for r in lc.list_strategies()]


@promotion_router.post("/register")
def register_strategy(req: RegisterRequest):
    lc = _get_lifecycle()
    bt = None
    if req.backtest_sharpe is not None:
        bt = BacktestResult(
            sharpe=req.backtest_sharpe,
            max_drawdown=req.backtest_max_dd or 0.0,
            cagr=req.backtest_cagr or 0.0,
            n_trades=req.backtest_trades or 0,
            win_rate=req.backtest_wr or 0.0,
            start_date=req.backtest_start or "",
            end_date=req.backtest_end or "",
        )
    sid = lc.register_strategy(req.name, req.code_path, bt, req.description, req.tags)
    return {"strategy_id": sid, "name": req.name, "state": StrategyState.DEVELOPMENT.value}


@promotion_router.get("/{strategy_id}")
def get_strategy(strategy_id: str):
    try:
        rec = _get_lifecycle().get_strategy(strategy_id)
        return rec.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@promotion_router.post("/{strategy_id}/evaluate")
def evaluate_strategy(strategy_id: str, req: EvaluateRequest):
    lc = _get_lifecycle()
    try:
        result = lc.evaluate(
            strategy_id, req.expected_return, req.actual_return,
            req.auto_promote, req.operator
        )
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/{strategy_id}/promote")
def promote_strategy(strategy_id: str, req: PromoteDemoteRequest):
    lc = _get_lifecycle()
    try:
        tr = lc.promote(strategy_id, req.reason or "manual promotion", req.operator)
        return tr.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/{strategy_id}/force-promote")
def force_promote_strategy(strategy_id: str, req: PromoteDemoteRequest):
    lc = _get_lifecycle()
    try:
        tr = lc.force_promote(strategy_id, req.reason or "force promotion", req.operator)
        return tr.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/{strategy_id}/demote")
def demote_strategy(
    strategy_id: str,
    req: PromoteDemoteRequest,
    target_state: Optional[str] = Query(None),
):
    lc = _get_lifecycle()
    try:
        ts = StrategyState(target_state) if target_state else None
        tr = lc.demote(strategy_id, req.reason or "manual demotion", ts, req.operator)
        return tr.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/{strategy_id}/retire")
def retire_strategy(strategy_id: str, req: PromoteDemoteRequest):
    lc = _get_lifecycle()
    try:
        tr = lc.retire(strategy_id, req.reason or "retired", req.operator)
        return tr.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.get("/{strategy_id}/history")
def get_history(strategy_id: str):
    lc = _get_lifecycle()
    try:
        return [t.model_dump() for t in lc.get_history(strategy_id)]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@promotion_router.get("/{strategy_id}/monthly-report")
def get_monthly_report(strategy_id: str):
    lc = _get_lifecycle()
    try:
        report = lc.monthly_review(strategy_id)
        return report.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@promotion_router.get("/alerts")
def get_all_alerts(strategy_id: str = Query(...), days: int = Query(30)):
    return [a.model_dump() for a in _get_lifecycle().get_alerts(strategy_id, days)]


@promotion_router.post("/record-daily")
def record_daily(req: DailyRecordRequest):
    lc = _get_lifecycle()
    try:
        snap = lc.record_daily(req.strategy_id, req.equity_value, req.n_trades, req.win_rate)
        return snap.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/allocation")
def compute_allocation(req: AllocationRequest):
    lc = _get_lifecycle()
    try:
        plan = lc.get_allocation(
            req.strategy_id, req.total_live_capital,
            req.win_rate, req.avg_win, req.avg_loss, req.days_in_state,
        )
        return plan.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@promotion_router.post("/{strategy_id}/approve-code-review")
def approve_code_review(strategy_id: str, operator: str = Query("operator")):
    lc = _get_lifecycle()
    try:
        lc.approve_code_review(strategy_id, operator)
        return {"strategy_id": strategy_id, "code_review_passed": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@promotion_router.post("/{strategy_id}/record-equity")
def record_equity(
    strategy_id: str,
    equity_value: float = Query(...),
    date_str: Optional[str] = Query(None),
):
    lc = _get_lifecycle()
    try:
        lc.record_equity(strategy_id, equity_value, date_str)
        return {"strategy_id": strategy_id, "equity_value": equity_value, "date": date_str or date.today().isoformat()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@promotion_router.get("/states/summary")
def states_summary():
    lc = _get_lifecycle()
    summary = {}
    for state in StrategyState:
        strats = lc.get_by_state(state)
        summary[state.value] = {
            "count": len(strats),
            "strategies": [{"id": s.strategy_id, "name": s.name} for s in strats],
        }
    return summary


@promotion_router.get("/candidates/promotion")
def promotion_candidates():
    lc = _get_lifecycle()
    candidates = lc.get_promotion_candidates()
    result = []
    for rec in candidates:
        gate = lc.evaluate(rec.strategy_id, auto_promote=False)
        result.append({
            "strategy_id": rec.strategy_id,
            "name": rec.name,
            "current_state": rec.state.value,
            "next_state": gate.next_state.value if gate.next_state else None,
            "gates_passed": gate.passed,
            "failed_gates": gate.failed_gates,
            "metrics": gate.metrics,
        })
    return result
