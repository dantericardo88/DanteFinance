#!/usr/bin/env bash
# dim_148: Multi-agent portfolio management workflow
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from sentinel.sai.agentic_portfolio_v3 import (
    AgentMessage,
    ResearchAgent,
    RiskAgent,
    PMAgent,
    ExecutionAgent,
    MultiAgentWorkflow,
    run_portfolio_cycle,
)
print("[OK] All imports successful from sentinel.sai.agentic_portfolio_v3")

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
tickers = ['AAPL', 'MSFT', 'GOOGL']
n = len(tickers)
rng = np.random.default_rng(seed=42)
current_weights = np.array([0.4, 0.35, 0.25])
market_data = {
    'returns': rng.normal(0, 0.01, n),
    'vols': [0.20, 0.18, 0.22],
}
portfolio_value = 1_000_000.0

# ---------------------------------------------------------------------------
# 3. Individual agent tests
# ---------------------------------------------------------------------------
# ResearchAgent
research_agent = ResearchAgent()
research = research_agent.analyze('AAPL', context={'returns': [0.005, -0.002, 0.003]})
assert 'signal' in research, "FAIL: research must have 'signal'"
assert 'conviction' in research, "FAIL: research must have 'conviction'"
assert 'rationale' in research, "FAIL: research must have 'rationale'"
assert -1.0 <= research['signal'] <= 1.0, (
    f"FAIL: signal out of range: {research['signal']}"
)
assert 0.0 <= research['conviction'] <= 1.0, (
    f"FAIL: conviction out of range: {research['conviction']}"
)
print(f"[OK] ResearchAgent: signal={research['signal']:.3f}, conviction={research['conviction']:.3f}")

# RiskAgent
risk_agent = RiskAgent()
vols = np.array([0.20, 0.18, 0.22])
corr = np.eye(n) * 0.7 + np.ones((n, n)) * 0.3
np.fill_diagonal(corr, 1.0)
cov = np.outer(vols, vols) * corr
risk_check = risk_agent.check(current_weights, cov, risk_budget=0.25)
assert 'approved' in risk_check, "FAIL: risk check must have 'approved'"
assert 'portfolio_vol' in risk_check, "FAIL: risk check must have 'portfolio_vol'"
assert 'breaches' in risk_check, "FAIL: risk check must have 'breaches'"
assert isinstance(risk_check['approved'], bool), "FAIL: 'approved' must be bool"
print(f"[OK] RiskAgent: approved={risk_check['approved']}, vol={risk_check['portfolio_vol']:.4f}")

# PMAgent
pm_agent = PMAgent()
target_from_pm = pm_agent.decide(research, risk_check, current_weights)
assert len(target_from_pm) == n, f"FAIL: PM target_weights wrong length: {len(target_from_pm)}"
assert abs(target_from_pm.sum() - 1.0) < 1e-6, (
    f"FAIL: PM target weights don't sum to 1: {target_from_pm.sum():.8f}"
)
assert np.all(target_from_pm >= 0), "FAIL: PM target weights must be >= 0"
print(f"[OK] PMAgent: target_weights={target_from_pm.round(4)} (sum={target_from_pm.sum():.8f})")

# ExecutionAgent
exec_agent = ExecutionAgent()
prices = np.array([180.0, 370.0, 140.0])
orders = exec_agent.generate_orders(current_weights, target_from_pm, portfolio_value, prices)
assert isinstance(orders, list), "FAIL: generate_orders must return a list"
print(f"[OK] ExecutionAgent: generated {len(orders)} orders")
for o in orders:
    assert 'ticker' in o, f"FAIL: order missing 'ticker': {o}"
    assert 'side' in o and o['side'] in ('BUY', 'SELL'), f"FAIL: order 'side' invalid: {o}"
    assert 'notional' in o, f"FAIL: order missing 'notional': {o}"

# ---------------------------------------------------------------------------
# 4. MultiAgentWorkflow — full cycle
# ---------------------------------------------------------------------------
workflow = MultiAgentWorkflow(tickers=tickers)
result = workflow.run_cycle(
    current_weights=current_weights,
    market_data=market_data,
    portfolio_value=portfolio_value,
)

# Validate result structure
assert 'target_weights' in result, "FAIL: result missing 'target_weights'"
assert 'orders' in result, "FAIL: result missing 'orders'"
assert 'risk_report' in result, "FAIL: result missing 'risk_report'"
assert 'messages' in result, "FAIL: result missing 'messages'"
print("[OK] run_cycle returned all required keys: target_weights, orders, risk_report, messages")

# target_weights sum to 1
tw = result['target_weights']
assert abs(tw.sum() - 1.0) < 1e-6, (
    f"FAIL: target_weights don't sum to 1: {tw.sum():.8f}"
)
assert np.all(tw >= 0), "FAIL: target_weights must all be >= 0"
assert len(tw) == n, f"FAIL: target_weights length {len(tw)} != n={n}"
print(f"[OK] target_weights sum=1, all>=0, length={len(tw)}: {tw.round(4)}")

# orders is a list
assert isinstance(result['orders'], list), "FAIL: orders must be a list"
print(f"[OK] orders is a list with {len(result['orders'])} items")

# ---------------------------------------------------------------------------
# 5. message_log
# ---------------------------------------------------------------------------
messages = workflow.message_log()
assert isinstance(messages, list), "FAIL: message_log must return a list"
assert len(messages) >= 1, "FAIL: message_log should have at least 1 message"
for msg in messages:
    assert isinstance(msg, AgentMessage), (
        f"FAIL: message_log items must be AgentMessage, got {type(msg)}"
    )
    assert hasattr(msg, 'agent_id'), "FAIL: AgentMessage must have agent_id"
    assert hasattr(msg, 'role'), "FAIL: AgentMessage must have role"
    assert hasattr(msg, 'content'), "FAIL: AgentMessage must have content"
print(f"[OK] message_log: {len(messages)} AgentMessage objects")
for m in messages:
    print(f"     [{m.role}] {m.agent_id}: {m.content[:60]}...")

# ---------------------------------------------------------------------------
# 6. run_portfolio_cycle convenience function
# ---------------------------------------------------------------------------
cycle_result = run_portfolio_cycle(tickers, current_weights, market_data, portfolio_value)
assert 'target_weights' in cycle_result, "FAIL: run_portfolio_cycle missing 'target_weights'"
assert 'orders' in cycle_result, "FAIL: run_portfolio_cycle missing 'orders'"
print("[OK] run_portfolio_cycle convenience function works")

# ---------------------------------------------------------------------------
# 7. AgentMessage dataclass
# ---------------------------------------------------------------------------
msg = AgentMessage(agent_id="test-01", role="research", content="Test message", data={"x": 1})
assert msg.agent_id == "test-01"
assert msg.role == "research"
assert msg.data == {"x": 1}
print("[OK] AgentMessage dataclass construction verified")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n--- Multi-Agent Workflow Summary ---")
print(f"  Tickers          : {tickers}")
print(f"  Initial weights  : {current_weights.tolist()}")
print(f"  Target weights   : {tw.tolist()}")
print(f"  Orders generated : {len(result['orders'])}")
print(f"  Messages logged  : {len(messages)}")
print(f"  Risk approved    : {result['risk_report']['approved']}")
print(f"  Portfolio vol    : {result['risk_report']['portfolio_vol']:.4f}")

print("\n[PASS] dim_148: Multi-agent portfolio workflow")
PYEOF
