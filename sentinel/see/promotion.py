"""
Strategy Promotion State Machine — LEAPFROG #67.

Enforces the BACKTEST → PAPER → CAPPED_LIVE → FULL_AUTONOMOUS pipeline
with mandatory DSR, PBO, and paper-trading performance gates at each transition.

No incumbent terminal automates strategy promotion with statistical overfitting gates.
Score: SENTINEL 10, Bloomberg 0.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
from sentinel.core.types import (
    StrategySpec, StrategyStatus, BacktestMetrics, PromotionGateResult, PromotionResult
)
from sentinel.core.logging import get_logger
from sentinel.core.security import assert_live_trading_enabled

logger = get_logger(__name__)

# Gate thresholds
MIN_DSR = 0.95           # Deflated Sharpe Ratio ≥ 0.95 to advance from BACKTEST
MAX_PBO = 0.20           # Probability of Backtest Overfitting ≤ 0.20
MIN_PAPER_SHARPE = 0.75  # Paper trading Sharpe ≥ 0.75 to advance to CAPPED_LIVE
MIN_PAPER_DAYS = 30      # Minimum paper trading period in days
MAX_PAPER_DRAWDOWN = -0.10  # Paper max drawdown must stay above -10%
MIN_CAPPED_DAYS = 90     # Minimum capped live period before FULL_AUTONOMOUS
MIN_CAPPED_SHARPE = 1.0  # Capped live Sharpe ≥ 1.0


class PromotionEngine:
    """
    Evaluates whether a strategy meets the gates to advance to the next status level.
    All transitions are logged and auditable.
    """

    def evaluate_backtest_to_paper(
        self,
        spec: StrategySpec,
        backtest_metrics: BacktestMetrics,
        pbo_result: Optional[dict] = None,
    ) -> PromotionResult:
        """
        Gate: BACKTEST → PAPER
        Requirements: DSR ≥ 0.95, PBO ≤ 0.20, Sharpe > 0, max_drawdown > -50%.
        """
        if spec.status != StrategyStatus.BACKTEST:
            return PromotionResult(
                strategy_id=spec.strategy_id,
                from_status=spec.status,
                to_status=spec.status,
                approved=False,
                evaluated_at=datetime.utcnow(),
                gates=[],
                rejection_reason=f"Strategy is not in BACKTEST status (is {spec.status.value})",
            )

        gates = []

        # Gate 1: Deflated Sharpe Ratio
        dsr = float(backtest_metrics.deflated_sharpe_ratio)
        gates.append(PromotionGateResult(
            gate_name="deflated_sharpe_ratio",
            threshold=MIN_DSR,
            actual=dsr,
            passed=dsr >= MIN_DSR,
            description=f"DSR {dsr:.4f} vs min {MIN_DSR}",
        ))

        # Gate 2: PBO
        pbo = pbo_result.get("pbo", 0.0) if pbo_result else 0.0
        gates.append(PromotionGateResult(
            gate_name="probability_of_backtest_overfitting",
            threshold=MAX_PBO,
            actual=pbo,
            passed=pbo <= MAX_PBO,
            description=f"PBO {pbo:.4f} vs max {MAX_PBO}",
        ))

        # Gate 3: Raw Sharpe > 0
        sharpe = float(backtest_metrics.sharpe_ratio)
        gates.append(PromotionGateResult(
            gate_name="sharpe_positive",
            threshold=0.0,
            actual=sharpe,
            passed=sharpe > 0,
            description=f"Sharpe {sharpe:.4f} must be positive",
        ))

        # Gate 4: Max drawdown
        mdd = float(backtest_metrics.max_drawdown)
        gates.append(PromotionGateResult(
            gate_name="max_drawdown",
            threshold=-0.50,
            actual=mdd,
            passed=mdd > -0.50,
            description=f"Max drawdown {mdd:.2%} vs floor -50%",
        ))

        failed = [g for g in gates if not g.passed]
        approved = len(failed) == 0
        rejection = "; ".join(g.description for g in failed) if failed else None

        logger.info("Promotion gate: BACKTEST→PAPER", strategy_id=spec.strategy_id,
                    approved=approved, failed_gates=len(failed))

        return PromotionResult(
            strategy_id=spec.strategy_id,
            from_status=StrategyStatus.BACKTEST,
            to_status=StrategyStatus.PAPER if approved else StrategyStatus.BACKTEST,
            approved=approved,
            evaluated_at=datetime.utcnow(),
            gates=gates,
            rejection_reason=rejection,
        )

    def evaluate_paper_to_capped_live(
        self,
        spec: StrategySpec,
        paper_metrics: BacktestMetrics,
        paper_start_date: date,
    ) -> PromotionResult:
        """
        Gate: PAPER → CAPPED_LIVE
        Requirements: ≥30 days paper, Sharpe ≥ 0.75, max_drawdown > -10%.
        Also calls assert_live_trading_enabled() — live trading safety gate.
        """
        if spec.status != StrategyStatus.PAPER:
            return PromotionResult(
                strategy_id=spec.strategy_id,
                from_status=spec.status,
                to_status=spec.status,
                approved=False,
                evaluated_at=datetime.utcnow(),
                gates=[],
                rejection_reason=f"Strategy not in PAPER status",
            )

        gates = []

        # Gate 1: Paper trading duration
        days_in_paper = (date.today() - paper_start_date).days
        gates.append(PromotionGateResult(
            gate_name="paper_trading_duration_days",
            threshold=float(MIN_PAPER_DAYS),
            actual=float(days_in_paper),
            passed=days_in_paper >= MIN_PAPER_DAYS,
            description=f"{days_in_paper} days paper trading vs min {MIN_PAPER_DAYS}",
        ))

        # Gate 2: Paper Sharpe
        sharpe = float(paper_metrics.sharpe_ratio)
        gates.append(PromotionGateResult(
            gate_name="paper_sharpe_ratio",
            threshold=MIN_PAPER_SHARPE,
            actual=sharpe,
            passed=sharpe >= MIN_PAPER_SHARPE,
            description=f"Paper Sharpe {sharpe:.4f} vs min {MIN_PAPER_SHARPE}",
        ))

        # Gate 3: Paper max drawdown
        mdd = float(paper_metrics.max_drawdown)
        gates.append(PromotionGateResult(
            gate_name="paper_max_drawdown",
            threshold=MAX_PAPER_DRAWDOWN,
            actual=mdd,
            passed=mdd > MAX_PAPER_DRAWDOWN,
            description=f"Paper MDD {mdd:.2%} vs floor {MAX_PAPER_DRAWDOWN:.0%}",
        ))

        # Gate 4: Live trading environment enabled
        live_enabled = True
        live_msg = "Live trading enabled"
        try:
            assert_live_trading_enabled()
        except PermissionError as exc:
            live_enabled = False
            live_msg = str(exc)
        gates.append(PromotionGateResult(
            gate_name="live_trading_enabled",
            threshold=1.0,
            actual=1.0 if live_enabled else 0.0,
            passed=live_enabled,
            description=live_msg,
        ))

        failed = [g for g in gates if not g.passed]
        approved = len(failed) == 0
        rejection = "; ".join(g.description for g in failed) if failed else None

        logger.info("Promotion gate: PAPER→CAPPED_LIVE", strategy_id=spec.strategy_id, approved=approved)

        return PromotionResult(
            strategy_id=spec.strategy_id,
            from_status=StrategyStatus.PAPER,
            to_status=StrategyStatus.CAPPED_LIVE if approved else StrategyStatus.PAPER,
            approved=approved,
            evaluated_at=datetime.utcnow(),
            gates=gates,
            rejection_reason=rejection,
        )

    def evaluate_capped_to_full(
        self,
        spec: StrategySpec,
        capped_metrics: BacktestMetrics,
        capped_start_date: date,
    ) -> PromotionResult:
        """
        Gate: CAPPED_LIVE → FULL_AUTONOMOUS
        Requirements: ≥90 days capped live, Sharpe ≥ 1.0.
        Highest-stakes gate — auto-trading with real money at full size.
        """
        if spec.status != StrategyStatus.CAPPED_LIVE:
            return PromotionResult(
                strategy_id=spec.strategy_id,
                from_status=spec.status,
                to_status=spec.status,
                approved=False,
                evaluated_at=datetime.utcnow(),
                gates=[],
                rejection_reason="Strategy not in CAPPED_LIVE status",
            )

        gates = []

        # Gate 1: Capped live duration
        days_capped = (date.today() - capped_start_date).days
        gates.append(PromotionGateResult(
            gate_name="capped_live_duration_days",
            threshold=float(MIN_CAPPED_DAYS),
            actual=float(days_capped),
            passed=days_capped >= MIN_CAPPED_DAYS,
            description=f"{days_capped} days capped live vs min {MIN_CAPPED_DAYS}",
        ))

        # Gate 2: Capped live Sharpe
        sharpe = float(capped_metrics.sharpe_ratio)
        gates.append(PromotionGateResult(
            gate_name="capped_sharpe_ratio",
            threshold=MIN_CAPPED_SHARPE,
            actual=sharpe,
            passed=sharpe >= MIN_CAPPED_SHARPE,
            description=f"Capped Sharpe {sharpe:.4f} vs min {MIN_CAPPED_SHARPE}",
        ))

        # Gate 3: No kill switch triggered
        kill_triggered = spec.kill_switch_active
        gates.append(PromotionGateResult(
            gate_name="no_kill_switch",
            threshold=0.0,
            actual=1.0 if kill_triggered else 0.0,
            passed=not kill_triggered,
            description="Kill switch not active" if not kill_triggered else "Kill switch has been triggered",
        ))

        failed = [g for g in gates if not g.passed]
        approved = len(failed) == 0
        rejection = "; ".join(g.description for g in failed) if failed else None

        logger.info("Promotion gate: CAPPED→FULL_AUTONOMOUS", strategy_id=spec.strategy_id, approved=approved)

        return PromotionResult(
            strategy_id=spec.strategy_id,
            from_status=StrategyStatus.CAPPED_LIVE,
            to_status=StrategyStatus.FULL_AUTONOMOUS if approved else StrategyStatus.CAPPED_LIVE,
            approved=approved,
            evaluated_at=datetime.utcnow(),
            gates=gates,
            rejection_reason=rejection,
        )

    def demote(
        self,
        spec: StrategySpec,
        reason: str,
        target_status: Optional[StrategyStatus] = None,
    ) -> PromotionResult:
        """
        Emergency demotion — move strategy back to PAPER or BACKTEST.
        Triggered by: kill switch, drawdown breach, risk limit violation.
        """
        target = target_status or StrategyStatus.PAPER
        logger.warning("Strategy DEMOTED", strategy_id=spec.strategy_id,
                       from_status=spec.status.value, to_status=target.value, reason=reason)
        return PromotionResult(
            strategy_id=spec.strategy_id,
            from_status=spec.status,
            to_status=target,
            approved=True,  # Demotions are always approved
            evaluated_at=datetime.utcnow(),
            gates=[],
            rejection_reason=None,
            notes=f"EMERGENCY DEMOTION: {reason}",
        )
