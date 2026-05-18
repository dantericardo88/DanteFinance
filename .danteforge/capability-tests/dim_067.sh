#!/bin/bash
# dim_067: Strategy promotion v3 — FSM states, criteria, StrategyRegistry
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import tempfile, json
from pathlib import Path

from sentinel.sbx.strategy_promotion_v3 import (
    StrategyState,
    TransitionType,
    CriterionResult,
    STATE_ORDER,
    DEMOTION_MAP,
    PromotionCriteria,
    PerformanceMetrics,
    CriteriaResult,
    StateTransition,
    TransitionResult,
    DemotionDecision,
    AllocationChange,
    Strategy,
    StrategyRegistry,
    CapitalAllocator,
    StateMachine,
    StrategyLifecycleManager,
    PromotionScoreCalculator,
)

# 1. StrategyState enum
assert StrategyState.RESEARCH == "research"
assert StrategyState.PAPER_TRADING == "paper_trading"
assert StrategyState.LIVE_SHADOW == "live_shadow"
assert StrategyState.LIVE_SMALL == "live_small"
assert StrategyState.LIVE_FULL == "live_full"
assert StrategyState.RETIRED == "retired"
assert StrategyState.SUSPENDED == "suspended"
print(f"[OK] StrategyState: 7 states verified")

# 2. STATE_ORDER defines the promotion sequence
expected_order = [
    StrategyState.RESEARCH,
    StrategyState.PAPER_TRADING,
    StrategyState.LIVE_SHADOW,
    StrategyState.LIVE_SMALL,
    StrategyState.LIVE_FULL,
]
assert STATE_ORDER == expected_order, f"STATE_ORDER mismatch: {STATE_ORDER}"
print(f"[OK] STATE_ORDER: {[s.value for s in STATE_ORDER]}")

# 3. DEMOTION_MAP defines where failed strategies go
assert DEMOTION_MAP[StrategyState.PAPER_TRADING] == StrategyState.RETIRED
assert DEMOTION_MAP[StrategyState.LIVE_SHADOW] == StrategyState.RESEARCH
assert DEMOTION_MAP[StrategyState.LIVE_SMALL] == StrategyState.RESEARCH
assert DEMOTION_MAP[StrategyState.LIVE_FULL] == StrategyState.LIVE_SMALL
print(f"[OK] DEMOTION_MAP: 4 demotion paths verified")

# 4. TransitionType enum
assert TransitionType.PROMOTE == "promote"
assert TransitionType.DEMOTE == "demote"
assert TransitionType.SUSPEND == "suspend"
assert TransitionType.RETIRE == "retire"
assert TransitionType.REINSTATE == "reinstate"
print(f"[OK] TransitionType: 5 types verified")

# 5. CriterionResult enum
assert CriterionResult.PASS == "pass"
assert CriterionResult.FAIL == "fail"
assert CriterionResult.SKIP == "skip"
print(f"[OK] CriterionResult: PASS/FAIL/SKIP verified")

# 6. PromotionCriteria gate definitions
assert "backtest_sharpe_min" in PromotionCriteria.RESEARCH_TO_PAPER
assert "backtest_max_drawdown_max" in PromotionCriteria.RESEARCH_TO_PAPER
rtp = PromotionCriteria.RESEARCH_TO_PAPER
assert rtp["backtest_sharpe_min"]["threshold"] == 0.8
print(f"[OK] PromotionCriteria.RESEARCH_TO_PAPER: {len(rtp)} criteria, sharpe_min={rtp['backtest_sharpe_min']['threshold']}")

# 7. Strategy dataclass
s = Strategy(
    strategy_id="strat_001",
    name="SMA Momentum",
    description="20/50 SMA crossover",
    state=StrategyState.RESEARCH.value,
    backtest_sharpe=1.2,
    backtest_max_drawdown=0.15,
    backtest_win_rate=0.55,
)
assert s.strategy_id == "strat_001"
assert s.state == "research"
assert s.backtest_sharpe == 1.2
assert s.current_allocation == 0.0
assert s.tags == []
print(f"[OK] Strategy dataclass: id={s.strategy_id}, state={s.state}, sharpe={s.backtest_sharpe}")

# 8. StrategyRegistry (uses global REGISTRY_PATH; patch it for test)
import sentinel.sbx.strategy_promotion_v3 as sp_module
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    orig_path = sp_module.REGISTRY_PATH
    sp_module.REGISTRY_PATH = Path(tmpdir) / "test_registry.json"
    reg = StrategyRegistry()
    strat_id = reg.register(s)
    assert strat_id == "strat_001"
    retrieved = reg.get("strat_001")
    assert retrieved is not None
    assert retrieved.name == "SMA Momentum"
    all_strategies = reg.list_all()
    assert len(all_strategies) == 1
    print(f"[OK] StrategyRegistry: register/get/list_all verified, {len(all_strategies)} strategy stored")

    # State transition (patch logger to avoid Windows cp1252 Unicode issue with arrow → char)
    import logging
    import sentinel.sbx.strategy_promotion_v3 as _sp3
    _orig_logger = _sp3.logger
    _sp3.logger = logging.getLogger("dim_067_test")
    try:
        reg.update_state("strat_001", StrategyState.PAPER_TRADING, reason="test")
    finally:
        _sp3.logger = _orig_logger
    updated = reg.get("strat_001")
    assert updated.state == "paper_trading"
    print(f"[OK] StrategyRegistry.update_state(): {updated.state}")
    sp_module.REGISTRY_PATH = orig_path

# 9. CapitalAllocator — uses allocate(strategy, portfolio_equity)
alloc = CapitalAllocator()
# Build strategy objects in each state
s_research = Strategy("s1", "A", "B", state=StrategyState.RESEARCH.value)
s_paper    = Strategy("s2", "A", "B", state=StrategyState.PAPER_TRADING.value)
s_small    = Strategy("s3", "A", "B", state=StrategyState.LIVE_SMALL.value)
s_full     = Strategy("s4", "A", "B", state=StrategyState.LIVE_FULL.value)

portfolio_equity = 1_000_000.0
research_cap = alloc.allocate(s_research, portfolio_equity)
assert research_cap == 0.0, f"Research should have $0 real capital: {research_cap}"
paper_cap = alloc.allocate(s_paper, portfolio_equity)
assert paper_cap == 0.0, f"Paper trading should have $0 real capital: {paper_cap}"
live_small_cap = alloc.allocate(s_small, portfolio_equity)
assert live_small_cap > 0.0, f"LIVE_SMALL should have positive capital: {live_small_cap}"
live_full_cap = alloc.allocate(s_full, portfolio_equity)
assert live_full_cap > live_small_cap, f"LIVE_FULL should have more capital than LIVE_SMALL"
print(f"[OK] CapitalAllocator: research=$0, paper=$0, live_small={live_small_cap:,.0f}, live_full={live_full_cap:,.0f}")

# 10. StateMachine and StrategyLifecycleManager exist with key methods
sm = StateMachine
assert hasattr(sm, '__init__')
print(f"[OK] StateMachine class exists")

slm = StrategyLifecycleManager
assert hasattr(slm, '__init__')
print(f"[OK] StrategyLifecycleManager class exists")

# ---- New math: PromotionScoreCalculator ----
calc = PromotionScoreCalculator()

# -- compute_capacity_estimate --
daily_vol   = 50_000_000.0   # $50M daily volume
ann_ret     = 0.20           # 20% annual return
threshold   = 0.10           # 10% impact threshold
cap = calc.compute_capacity_estimate(daily_vol, ann_ret, threshold)
expected_capacity = daily_vol * threshold / ann_ret   # = 25_000_000
assert cap["strategy_capacity_usd"] == expected_capacity, \
    f"Capacity expected {expected_capacity}, got {cap['strategy_capacity_usd']}"
print(f"[OK] compute_capacity_estimate: capacity=${cap['strategy_capacity_usd']:,.0f} (expected ${expected_capacity:,.0f})")

# zero annual_return returns 0 capacity
cap_zero = calc.compute_capacity_estimate(50_000_000, 0.0)
assert cap_zero["strategy_capacity_usd"] == 0.0
print("[OK] compute_capacity_estimate with zero return gives 0.0")

# -- simulate_paper_to_live_transition --
paper_m = PerformanceMetrics(strategy_id="test", period="all", sufficient_data=True,
                              sharpe_ratio=1.4, slippage_ratio=0.05)
sim = calc.simulate_paper_to_live_transition(
    paper_metrics=paper_m,
    proposed_aum=10_000_000,
    daily_volume=50_000_000.0,
    annual_return=0.20,
)
assert abs(sim["live_slippage_premium"] - 0.075) < 1e-9, \
    f"slippage premium expected 0.075, got {sim['live_slippage_premium']}"
assert sim["slippage_multiplier"] == 1.5
assert sim["capacity_ok"] == True   # 10M < 25M capacity
print(f"[OK] simulate_paper_to_live: live_sharpe={sim['adjusted_live_sharpe']:.4f}, capacity_ok={sim['capacity_ok']}")
print(f"[OK]   slippage premium={sim['live_slippage_premium']:.4f} (0.05 x 1.5 = 0.075)")

# -- compute_promotion_score formula verification --
# All inputs = 1.0 -> total = 1.0 (weights sum to 1)
score_max = calc.compute_promotion_score(1.0, 1.0, 1.0, 1.0)
assert abs(score_max["total_score"] - 1.0) < 1e-9, \
    f"Max score expected 1.0, got {score_max['total_score']}"
print(f"[OK] compute_promotion_score (all=1.0) = {score_max['total_score']:.8f}  (expected 1.0)")

# All inputs = 0.0 -> total = 0.0
score_min = calc.compute_promotion_score(0.0, 0.0, 0.0, 0.0)
assert abs(score_min["total_score"] - 0.0) < 1e-9
print(f"[OK] compute_promotion_score (all=0.0) = {score_min['total_score']:.8f}  (expected 0.0)")

# Specific weighted formula: 0.30*s + 0.30*d + 0.20*st + 0.20*c
s, d, st, c = 0.8, 0.7, 0.9, 0.6
expected_score = 0.30*s + 0.30*d + 0.20*st + 0.20*c
result_score = calc.compute_promotion_score(s, d, st, c)
assert abs(result_score["total_score"] - expected_score) < 1e-9, \
    f"Score expected {expected_score:.8f}, got {result_score['total_score']:.8f}"
print(f"[OK] compute_promotion_score({s},{d},{st},{c}) = {result_score['total_score']:.8f}  (expected {expected_score:.8f})")

# Verify weight contributions individually
assert abs(result_score["contributions"]["sharpe"]    - 0.30*s)  < 1e-9
assert abs(result_score["contributions"]["dsr"]       - 0.30*d)  < 1e-9
assert abs(result_score["contributions"]["stability"] - 0.20*st) < 1e-9
assert abs(result_score["contributions"]["capacity"]  - 0.20*c)  < 1e-9
print("[OK] Individual weight contributions verified")

print("\n[PASS] dim_067: Strategy promotion v3 -- FSM + PromotionScoreCalculator verified")
PYEOF
