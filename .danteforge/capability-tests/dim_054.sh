#!/bin/bash
# dim_054: NL Strategy Generator v3 -- param grid, risk layer, backtest template, rationale
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os, ast
sys.path.insert(0, os.getcwd())

from sentinel.sai.nl_strategy_generator_v3 import (
    IntentClassifier,
    StrategyLanguageParserV3,
    StrategySpec,
    PositionSizing,
    Universe,
    QualityGateResult,
    ParameterGridBuilder,
    RiskManagementLayer,
    StrategyTemplateGenerator,
    EconomicRationaleEngine,
    _QUALITY_GATE_SHARPE,
    _QUALITY_GATE_MAX_DD,
    _QUALITY_GATE_MIN_TRADES,
)

# 1. IntentClassifier
ic = IntentClassifier()
assert ic.classify("momentum strategy 12-month lookback") == "MOMENTUM"
assert ic.classify("mean reversion when RSI below 30 oversold") == "MEAN_REVERT"
assert ic.classify("golden cross moving average crossover") == "TREND_FOLLOW"
assert ic.classify("pairs trading long AAPL short MSFT cointegrated spread") == "PAIRS"
assert ic.classify("breakout above 52-week high") == "BREAKOUT"
print("[OK] IntentClassifier: MOMENTUM, MEAN_REVERT, TREND_FOLLOW, PAIRS, BREAKOUT")

# 2. ParameterGridBuilder -- strategy spec -> parameter dict with valid ranges
pgb = ParameterGridBuilder()

# RSI oversold momentum strategy
grid = pgb.build("RSI oversold momentum strategy: buy when RSI below 30, sell when RSI above 70, stop-loss 5%, target 10%")
assert "params" in grid
assert "intent" in grid
assert grid["intent"] in ("MOMENTUM", "MEAN_REVERT")
params = grid["params"]

# Must have RSI entry threshold
assert "rsi_entry" in params, f"rsi_entry missing from params: {list(params.keys())}"
assert params["rsi_entry"]["default"] == 30.0, f"RSI entry default should be 30, got {params['rsi_entry']['default']}"
assert len(params["rsi_entry"]["range"]) >= 2, "RSI entry range should have multiple values"

# Must have RSI exit threshold
assert "rsi_exit" in params, f"rsi_exit missing from params: {list(params.keys())}"
assert params["rsi_exit"]["default"] == 70.0, f"RSI exit default should be 70, got {params['rsi_exit']['default']}"

# Must have stop loss
assert "stop_loss_pct" in params, f"stop_loss_pct missing from params: {list(params.keys())}"
assert params["stop_loss_pct"]["default"] == 5.0, f"Stop loss default should be 5.0, got {params['stop_loss_pct']['default']}"

# Must have take profit
assert "take_profit_pct" in params, f"take_profit_pct missing from params: {list(params.keys())}"
assert params["take_profit_pct"]["default"] == 10.0, f"Take profit default should be 10.0, got {params['take_profit_pct']['default']}"

# Must have RSI period
assert "rsi_period" in params, f"rsi_period missing from params: {list(params.keys())}"
assert params["rsi_period"]["default"] == 14, f"RSI period default should be 14, got {params['rsi_period']['default']}"

assert grid["total_combinations"] > 1, "Should have multiple combinations for grid search"
print(f"[OK] ParameterGridBuilder: RSI oversold -> rsi_entry=30, exit=70, stop=5%, target=10%, {grid['total_combinations']} combinations")

# Trend-following grid
grid2 = pgb.build("golden cross moving average crossover strategy")
assert "fast_ma" in grid2["params"] or "stop_loss_pct" in grid2["params"]
print(f"[OK] ParameterGridBuilder: trend-follow -> {list(grid2['params'].keys())}")

# 3. RiskManagementLayer -- Kelly fraction and drawdown stop
rml = RiskManagementLayer()

# Kelly fraction: win_rate=0.55, win_loss_ratio=1.5
kelly = rml.kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, fraction=0.25)
# full_kelly = 0.55 - (0.45/1.5) = 0.55 - 0.30 = 0.25; fractional = 0.25 * 0.25 = 0.0625
full_kelly = 0.55 - (0.45 / 1.5)
expected_kelly = full_kelly * 0.25
assert abs(kelly - expected_kelly) < 0.001, f"Kelly: expected {expected_kelly:.4f}, got {kelly}"
assert 0 < kelly < 0.5, f"Kelly fraction should be positive and < 50%: {kelly}"
print(f"[OK] Kelly fraction: win_rate=0.55, W/L=1.5, 25%Kelly -> {kelly:.4f} ({kelly*100:.1f}%)")

# Apply risk layer to a parsed spec
parser = StrategyLanguageParserV3()
spec = parser.parse("buy when RSI below 30 and sell when RSI above 70, stop loss 5%")
risk = rml.apply(spec, win_rate=0.55, win_loss_ratio=1.5)

assert "kelly_fraction" in risk
assert "max_drawdown_stop" in risk
assert "correlation_filter_threshold" in risk
assert risk["max_drawdown_stop"] == 0.20, f"Max DD stop should be 0.20, got {risk['max_drawdown_stop']}"
assert risk["kelly_fraction"] > 0
assert risk["sizing_method"] == "fractional_kelly"
print(f"[OK] Risk layer: kelly={risk['kelly_fraction']:.4f}, "
      f"max_dd_stop={risk['max_drawdown_stop']:.0%}, "
      f"corr_filter={risk['correlation_filter_threshold']}")

# 4. StrategyTemplateGenerator -- backtest template compiles without errors
tmpl_gen = StrategyTemplateGenerator()

rsi_strategy_desc = "Buy when RSI below 30. Sell when RSI above 70 or stop loss 5%."
code = tmpl_gen.generate_template(rsi_strategy_desc)
assert isinstance(code, str) and len(code) > 100, "Template too short"

# Must compile without syntax errors
ok, errors = tmpl_gen.validate(code)
assert ok, f"Template has syntax errors: {errors}"

# Must define generate_signals function
try:
    tree = ast.parse(code)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "generate_signals" in func_names, f"generate_signals not found. Functions: {func_names}"
    assert "run_backtest" in func_names, f"run_backtest not found. Functions: {func_names}"
except SyntaxError as e:
    assert False, f"Template parse error: {e}"
print(f"[OK] Backtest template compiles, defines generate_signals and run_backtest")

# Momentum strategy template also compiles
momentum_code = tmpl_gen.generate_template("momentum strategy: buy top 20% 12-month return, sell after 3 months")
ok2, errs2 = tmpl_gen.validate(momentum_code)
assert ok2, f"Momentum template has errors: {errs2}"
print(f"[OK] Momentum strategy template compiles cleanly")

# 5. EconomicRationaleEngine -- momentum rationale references "trend persistence"
rationale_engine = EconomicRationaleEngine()

momentum_rationale = rationale_engine.explain("momentum strategy 12-month lookback")
assert momentum_rationale["intent"] == "MOMENTUM"
assert "trend persistence" in momentum_rationale["rationale"].lower(), (
    f"Momentum rationale should mention 'trend persistence': {momentum_rationale['rationale'][:200]}"
)
assert len(momentum_rationale["rationale"]) > 100
print(f"[OK] Momentum rationale: mentions 'trend persistence'")
print(f"     Key driver: {momentum_rationale['key_driver'][:80]}...")

# Mean-reversion rationale mentions mean-reversion concepts
mr_rationale = rationale_engine.explain("RSI oversold mean reversion strategy")
assert mr_rationale["intent"] == "MEAN_REVERT"
assert len(mr_rationale["rationale"]) > 100
print(f"[OK] Mean-revert rationale: intent=MEAN_REVERT, {len(mr_rationale['rationale'])} chars")

# Static rationale lookup
from sentinel.sai.nl_strategy_generator_v3 import EconomicRationaleEngine
momentum_text = EconomicRationaleEngine.get_rationale("MOMENTUM")
assert "trend persistence" in momentum_text.lower()
print(f"[OK] Static rationale: MOMENTUM mentions 'trend persistence'")

# 6. Quality gate constants
assert _QUALITY_GATE_SHARPE == 0.5
assert _QUALITY_GATE_MAX_DD == -0.30
assert _QUALITY_GATE_MIN_TRADES == 20
print(f"[OK] Quality gate constants: Sharpe>{_QUALITY_GATE_SHARPE}, MaxDD>{_QUALITY_GATE_MAX_DD}")

print("\n[PASS] dim_054: NL strategy generator -- param grid, risk layer, backtest template, rationale all verified")
PYEOF
