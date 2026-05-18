#!/bin/bash
# dim_054: NL Strategy Generator v3 — intent classification + template generation (no network)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

from sentinel.sai.nl_strategy_generator_v3 import (
    IntentClassifier, StrategyLanguageParserV3, StrategySpec,
    PositionSizing, Universe, QualityGateResult,
    _QUALITY_GATE_SHARPE, _QUALITY_GATE_MAX_DD, _QUALITY_GATE_MIN_TRADES,
)

# 1. IntentClassifier - verified patterns
ic = IntentClassifier()
assert ic.classify("momentum strategy 12-month lookback") == "MOMENTUM", ic.classify("momentum strategy")
assert ic.classify("mean reversion when RSI below 30 oversold") == "MEAN_REVERT"
assert ic.classify("golden cross moving average crossover") == "TREND_FOLLOW"
assert ic.classify("pairs trading long AAPL short MSFT cointegrated spread") == "PAIRS"
assert ic.classify("breakout above 52-week high") == "BREAKOUT"
assert ic.classify("earnings drift post-earnings surprise beat") == "EARNINGS_DRIFT"
assert ic.classify("carry trade dividend yield income stocks") == "CARRY"
assert ic.classify("volatility trading vix regime") == "VOLATILITY"
print("[OK] IntentClassifier: MOMENTUM, MEAN_REVERT, TREND_FOLLOW, PAIRS, BREAKOUT, EARNINGS_DRIFT, CARRY, VOLATILITY")

# 2. StrategyLanguageParserV3 - RSI conditions
parser = StrategyLanguageParserV3()
spec = parser.parse("buy when RSI below 30 and sell when RSI above 70")
assert isinstance(spec, StrategySpec)
assert len(spec.entry_conditions) >= 1, f"Expected entry conditions, got {len(spec.entry_conditions)}"
assert len(spec.exit_conditions) >= 1, f"Expected exit conditions, got {len(spec.exit_conditions)}"
entry_types = [c.condition_type for c in spec.entry_conditions]
assert "RSI_BELOW" in entry_types, f"Expected RSI_BELOW, got {entry_types}"
print(f"[OK] Parser: RSI<30 entry -> {entry_types}")

# 3. SMA crossover parsing
spec2 = parser.parse("enter when 50-day SMA crosses above 200-day SMA")
assert isinstance(spec2, StrategySpec)
all_cond_types = [c.condition_type for c in spec2.entry_conditions + spec2.exit_conditions]
print(f"[OK] Parser: 50/200 SMA cross -> conditions={all_cond_types}")

# 4. PositionSizing defaults
ps = PositionSizing()
assert ps.method == "equal_weight"
assert ps.value == 0.10
assert ps.max_positions == 10
print("[OK] PositionSizing defaults: equal_weight, 10% per position, max 10")

# 5. Universe.resolve_tickers()
u_sp500 = Universe(universe_type="SP500")
tickers = u_sp500.resolve_tickers()
assert len(tickers) >= 10
assert "AAPL" in tickers
u_custom = Universe(universe_type="CUSTOM_TICKERS", custom_tickers=["AAPL", "MSFT"])
assert u_custom.resolve_tickers() == ["AAPL", "MSFT"]
print(f"[OK] Universe.resolve_tickers(): SP500 has {len(tickers)} tickers, CUSTOM works")

# 6. QualityGateResult.summary()
qg_pass = QualityGateResult(passed=True, sharpe=1.2, max_drawdown=-0.15, num_trades=50, win_rate=0.55)
assert "PASSED" in qg_pass.summary()
qg_fail = QualityGateResult(passed=False, sharpe=0.3, max_drawdown=-0.35, num_trades=5, win_rate=0.30,
                             failure_reasons=["Sharpe below threshold", "Insufficient trades"])
assert "FAILED" in qg_fail.summary()
print("[OK] QualityGateResult.summary() formats PASSED/FAILED correctly")

# 7. Constants
assert _QUALITY_GATE_SHARPE == 0.5
assert _QUALITY_GATE_MAX_DD == -0.30
assert _QUALITY_GATE_MIN_TRADES == 20
print(f"[OK] Quality gate constants verified: Sharpe>{_QUALITY_GATE_SHARPE}, MaxDD>{_QUALITY_GATE_MAX_DD}")

print("\n[PASS] dim_054: NL strategy generator intent + parsing verified")
PYEOF
