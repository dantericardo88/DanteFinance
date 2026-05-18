#!/bin/bash
# dim_061: VectorBT Backtesting v3 — NumpyPortfolio fallback (pure numpy)
set -e
cd /c/Projects/DanteFinance

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd, numpy as np
from sentinel.sbx.vectorbt_backtest_v3 import (
    NumpyPortfolio, BacktestResult, TransactionCostModel, StrategyLibrary, HAS_VBT,
)

print(f"[OK] HAS_VBT={HAS_VBT} (optional)")

# 1. Synthetic price data
idx = pd.date_range("2023-01-01", periods=252)
np.random.seed(42)
prices_arr = 100.0 * np.exp(np.cumsum(np.random.randn(252) * 0.01))
prices = pd.DataFrame({"AAPL": prices_arr}, index=idx)
entries = pd.DataFrame({"AAPL": [True] + [False]*251}, index=idx)
exits = pd.DataFrame({"AAPL": [False]*100 + [True] + [False]*151}, index=idx)

# 2. NumpyPortfolio construction
port = NumpyPortfolio(prices=prices, entries=entries, exits=exits,
                      init_cash=100_000.0, fees=0.001)
assert port.init_cash == 100_000.0
assert port.fees == 0.001
print(f"[OK] NumpyPortfolio constructed (init_cash={port.init_cash}, fees={port.fees})")

# 3. total_return() is a float
tr = port.total_return()
assert isinstance(tr, float)
print(f"[OK] NumpyPortfolio.total_return() = {tr:.4f}")

# 4. sharpe_ratio()
sr = port.sharpe_ratio()
assert isinstance(sr, float)
print(f"[OK] NumpyPortfolio.sharpe_ratio() = {sr:.4f}")

# 5. max_drawdown() <= 0
md = port.max_drawdown()
assert isinstance(md, float)
assert md <= 0, f"Max drawdown must be non-positive, got {md}"
print(f"[OK] NumpyPortfolio.max_drawdown() = {md:.4f}")

# 6. get_equity() starts at init_cash
equity = port.get_equity()
assert isinstance(equity, pd.Series)
assert len(equity) == 252
assert equity.iloc[0] == 100_000.0
print(f"[OK] NumpyPortfolio.get_equity() -> len={len(equity)}, equity[0]={equity.iloc[0]}")

# 7. get_returns() is a pd.Series
returns = port.get_returns()
assert isinstance(returns, pd.Series)
assert len(returns) == 252
print(f"[OK] NumpyPortfolio.get_returns() -> len={len(returns)}")

# 8. TransactionCostModel - zero commission
tcm = TransactionCostModel()
c0 = tcm.compute_commission(100, 150.0, model="zero")
assert c0 == 0.0
c_flat = tcm.compute_commission(100, 150.0, model="flat")
assert c_flat == 1.0
c_pct = tcm.compute_commission(100, 150.0, model="percentage")
assert c_pct > 0  # percentage commission on qty=100, price=150 is positive
print(f"[OK] TransactionCostModel: zero={c0}, flat={c_flat}, pct={c_pct:.3f}")

# 9. StrategyLibrary.sma_crossover generates boolean signals
entries2, exits2 = StrategyLibrary.sma_crossover(prices, fast=10, slow=30)
assert isinstance(entries2, pd.DataFrame)
assert isinstance(exits2, pd.DataFrame)
assert entries2.shape == prices.shape
assert entries2.dtypes.iloc[0] == bool
print(f"[OK] StrategyLibrary.sma_crossover() -> entries shape={entries2.shape}, dtype={entries2.dtypes.iloc[0]}")

# 10. StrategyLibrary.rsi_mean_reversion
e_rsi, x_rsi = StrategyLibrary.rsi_mean_reversion(prices, period=14, oversold=30, overbought=70)
assert e_rsi.shape == prices.shape
assert e_rsi.dtypes.iloc[0] == bool
print(f"[OK] StrategyLibrary.rsi_mean_reversion() -> shape={e_rsi.shape}")

print("\n[PASS] dim_061: VectorBT NumpyPortfolio fallback verified")
PYEOF
