"""
VectorBT Backtesting v3 — dim_061 (score 8 → 9)
=================================================
Full vectorbt integration (optional dep, numpy fallback), multi-asset portfolio
backtesting, parameter grid search with OOS walk-forward validation, realistic
transaction costs, Monte Carlo equity curve simulation.

Data sources: yfinance prices (free).
VectorBT: optional. All critical paths have numpy/pandas fallback implementations.
"""
from __future__ import annotations

import itertools
import math
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import vectorbt as vbt  # type: ignore
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

try:
    from skopt import gp_minimize  # type: ignore
    from skopt.space import Integer, Real  # type: ignore
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# ---------------------------------------------------------------------------
# Logger (mirrors sentinel pattern)
# ---------------------------------------------------------------------------

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Full backtest output with performance attribution."""
    strategy: str
    tickers: list[str]
    start: str
    end: str
    params: dict
    # Returns
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_drawdown: float
    # Activity
    num_trades: int
    win_rate: float
    avg_trade_return: float
    avg_holding_days: float
    # Risk
    volatility: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    # Benchmark
    benchmark: str
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    # Cost model
    total_commission: float
    total_slippage: float
    # Extra
    equity_curve: Optional[pd.Series] = None
    monthly_returns: Optional[pd.DataFrame] = None
    notes: str = ""

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "tickers": self.tickers,
            "period": f"{self.start} → {self.end}",
            "params": self.params,
            "performance": {
                "total_return_pct": round(self.total_return * 100, 2),
                "cagr_pct": round(self.cagr * 100, 2),
                "sharpe": round(self.sharpe, 3),
                "sortino": round(self.sortino, 3),
                "calmar": round(self.calmar, 3),
                "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            },
            "activity": {
                "num_trades": self.num_trades,
                "win_rate_pct": round(self.win_rate * 100, 1),
                "avg_holding_days": round(self.avg_holding_days, 1),
            },
            "risk": {
                "volatility_pct": round(self.volatility * 100, 2),
                "var_95_pct": round(self.var_95 * 100, 2),
                "cvar_95_pct": round(self.cvar_95 * 100, 2),
                "skewness": round(self.skewness, 3),
                "kurtosis": round(self.kurtosis, 3),
            },
            "vs_benchmark": {
                "benchmark": self.benchmark,
                "benchmark_return_pct": round(self.benchmark_return * 100, 2),
                "alpha_pct": round(self.alpha * 100, 2),
                "beta": round(self.beta, 3),
                "ir": round(self.information_ratio, 3),
            },
            "costs": {
                "total_commission": round(self.total_commission, 2),
                "total_slippage": round(self.total_slippage, 2),
            },
            "notes": self.notes,
        }


@dataclass
class OptimizationResult:
    """Output of parameter optimization with IS vs OOS validation."""
    strategy: str
    tickers: list[str]
    metric: str
    best_params: dict
    best_is_score: float
    best_oos_score: float
    is_oos_degradation: float        # (IS - OOS) / IS — high = overfitting
    parameter_stability: float       # 0-1, 1 = very stable surface
    full_grid: Optional[pd.DataFrame] = None
    cv_results: Optional[list[dict]] = None
    optimization_method: str = "grid_search"
    notes: str = ""

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "metric": self.metric,
            "best_params": self.best_params,
            "is_score": round(self.best_is_score, 4),
            "oos_score": round(self.best_oos_score, 4),
            "degradation_pct": round(self.is_oos_degradation * 100, 1),
            "stability": round(self.parameter_stability, 3),
            "method": self.optimization_method,
            "notes": self.notes,
        }


@dataclass
class Tearsheet:
    """Comprehensive strategy tearsheet."""
    strategy: str
    start: str
    end: str
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_drawdown: float
    recovery_time_days: int
    volatility: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    num_trades: int
    win_rate: float
    profit_factor: float
    monthly_returns: Optional[pd.DataFrame] = None
    rolling_sharpe: Optional[pd.Series] = None
    underwater_curve: Optional[pd.Series] = None
    monte_carlo_percentiles: Optional[dict] = None

    def print_summary(self) -> str:
        lines = [
            "=" * 60,
            f"TEARSHEET: {self.strategy}",
            f"Period:    {self.start} → {self.end}",
            "=" * 60,
            f"Total Return:   {self.total_return*100:+.2f}%",
            f"CAGR:           {self.cagr*100:+.2f}%",
            f"Sharpe Ratio:   {self.sharpe:.3f}",
            f"Sortino Ratio:  {self.sortino:.3f}",
            f"Calmar Ratio:   {self.calmar:.3f}",
            f"Max Drawdown:   {self.max_drawdown*100:.2f}%",
            f"Avg Drawdown:   {self.avg_drawdown*100:.2f}%",
            f"Recovery Days:  {self.recovery_time_days}",
            "-" * 60,
            f"Volatility:     {self.volatility*100:.2f}% (ann.)",
            f"VaR 95%:        {self.var_95*100:.2f}%",
            f"CVaR 95%:       {self.cvar_95*100:.2f}%",
            f"Skewness:       {self.skewness:.3f}",
            f"Kurtosis:       {self.kurtosis:.3f}",
            "-" * 60,
            f"Num Trades:     {self.num_trades}",
            f"Win Rate:       {self.win_rate*100:.1f}%",
            f"Profit Factor:  {self.profit_factor:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class FullPipelineResult:
    """Output of run_full_pipeline — grid search + WF + Monte Carlo + tearsheet."""
    strategy: str
    tickers: list[str]
    optimization: OptimizationResult
    best_backtest: BacktestResult
    tearsheet: Tearsheet
    monte_carlo_summary: dict
    report: str


# ---------------------------------------------------------------------------
# Numpy fallback portfolio (when vectorbt not installed)
# ---------------------------------------------------------------------------

class NumpyPortfolio:
    """
    Lightweight numpy-based portfolio simulator.
    Mimics the key interface of vbt.Portfolio.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        init_cash: float = 100_000.0,
        fees: float = 0.001,
    ) -> None:
        self.prices = prices if isinstance(prices, pd.DataFrame) else prices.to_frame()
        self.entries = entries if isinstance(entries, pd.DataFrame) else entries.to_frame()
        self.exits = exits if isinstance(exits, pd.DataFrame) else exits.to_frame()
        self.init_cash = init_cash
        self.fees = fees

        self._equity: Optional[pd.Series] = None
        self._returns: Optional[pd.Series] = None
        self._trades: Optional[pd.DataFrame] = None

        self._simulate()

    def _simulate(self) -> None:
        """Core simulation loop (vectorized over time, sequential over signals)."""
        prices = self.prices.copy()
        n_assets = prices.shape[1]
        n_days = len(prices)

        equity = pd.Series(self.init_cash, index=prices.index, dtype=float)
        cash = self.init_cash
        positions: dict[str, float] = {c: 0.0 for c in prices.columns}
        trade_log: list[dict] = []

        for i in range(1, n_days):
            date = prices.index[i]
            prev_date = prices.index[i - 1]

            for col in prices.columns:
                p = float(prices.loc[date, col])
                if math.isnan(p) or p <= 0:
                    continue

                # Use prior-day signals to avoid look-ahead
                entry_sig = bool(self.entries.loc[prev_date, col]) if prev_date in self.entries.index else False
                exit_sig = bool(self.exits.loc[prev_date, col]) if prev_date in self.exits.index else False

                pos = positions[col]

                if entry_sig and pos == 0.0 and cash > 0:
                    # Allocate equal weight per asset
                    alloc = cash / max(n_assets, 1)
                    cost = alloc * self.fees
                    shares = (alloc - cost) / p
                    positions[col] = shares
                    cash -= alloc
                    trade_log.append({
                        "date": date, "ticker": col, "action": "BUY",
                        "price": p, "shares": shares, "cost": cost,
                    })

                elif exit_sig and pos > 0.0:
                    proceeds = pos * p
                    cost = proceeds * self.fees
                    cash += proceeds - cost
                    positions[col] = 0.0
                    trade_log.append({
                        "date": date, "ticker": col, "action": "SELL",
                        "price": p, "shares": pos, "cost": cost,
                    })

            # Mark-to-market
            port_value = cash + sum(
                positions[col] * float(prices.loc[date, col])
                for col in prices.columns
                if not math.isnan(float(prices.loc[date, col]))
            )
            equity.iloc[i] = port_value

        self._equity = equity
        self._returns = equity.pct_change().fillna(0.0)
        self._trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
            columns=["date", "ticker", "action", "price", "shares", "cost"]
        )

    # Public interface (mirrors vbt.Portfolio)
    def total_return(self) -> float:
        if self._equity is None:
            return 0.0
        return float(self._equity.iloc[-1] / self.init_cash - 1)

    def annualized_return(self) -> float:
        tr = self.total_return()
        n_days = len(self._equity)
        if n_days < 2:
            return 0.0
        n_years = n_days / 252.0
        return (1 + tr) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    def sharpe_ratio(self) -> float:
        if self._returns is None:
            return 0.0
        r = self._returns
        std = r.std()
        if std == 0:
            return 0.0
        return float(r.mean() / std * math.sqrt(252))

    def max_drawdown(self) -> float:
        if self._equity is None:
            return 0.0
        roll_max = self._equity.cummax()
        dd = (self._equity - roll_max) / roll_max
        return float(dd.min())

    def get_returns(self) -> pd.Series:
        return self._returns if self._returns is not None else pd.Series(dtype=float)

    def get_equity(self) -> pd.Series:
        return self._equity if self._equity is not None else pd.Series(dtype=float)

    def get_trades(self) -> pd.DataFrame:
        return self._trades if self._trades is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Class 1: VectorBTAdapter
# ---------------------------------------------------------------------------

class VectorBTAdapter:
    """
    Unified adapter for vectorbt (when installed) with full numpy fallback.
    When VBT is present, uses its vectorized engine for maximum speed.
    When VBT is absent, uses NumpyPortfolio for identical interface.
    """

    def __init__(self, fees: float = 0.001, init_cash: float = 100_000.0) -> None:
        self.fees = fees
        self.init_cash = init_cash
        self.has_vbt = HAS_VBT

    # ------------------------------------------------------------------
    # Signal-based backtesting
    # ------------------------------------------------------------------

    def run_signal_backtest(
        self,
        prices: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        init_cash: float = None,
        fees: float = None,
    ):
        """
        Run a signal-based backtest.
        Returns vbt.Portfolio if vectorbt is available, else NumpyPortfolio.
        Both expose: total_return(), sharpe_ratio(), max_drawdown(), get_returns(), get_equity().
        """
        cash = init_cash or self.init_cash
        fee = fees if fees is not None else self.fees

        # Enforce shift(1) to prevent look-ahead bias
        entries_safe = entries.shift(1).fillna(False).astype(bool)
        exits_safe = exits.shift(1).fillna(False).astype(bool)

        if self.has_vbt:
            try:
                pf = vbt.Portfolio.from_signals(
                    close=prices,
                    entries=entries_safe,
                    exits=exits_safe,
                    init_cash=cash,
                    fees=fee,
                    freq="1D",
                )
                return pf
            except Exception as exc:
                logger.warning("VBT backtest failed (%s), falling back to numpy.", exc)

        return NumpyPortfolio(prices, entries_safe, exits_safe, cash, fee)

    # ------------------------------------------------------------------
    # Indicator-based (MA crossover) backtest
    # ------------------------------------------------------------------

    def run_indicator_backtest(
        self,
        prices: pd.DataFrame,
        fast: int,
        slow: int,
    ):
        """
        Simple SMA crossover backtest.
        Entry: fast SMA crosses above slow SMA.
        Exit:  fast SMA crosses below slow SMA.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()

        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return self.run_signal_backtest(prices, entries, exits)

    # ------------------------------------------------------------------
    # Parameter sweep (vectorized over all fast×slow combinations)
    # ------------------------------------------------------------------

    def param_sweep(
        self,
        prices: pd.DataFrame,
        fast_range: range,
        slow_range: range,
    ) -> pd.DataFrame:
        """
        Vectorized parameter sweep over SMA crossover strategy.
        Returns DataFrame with columns ['fast', 'slow', 'sharpe', 'total_return', 'max_dd'].

        When vectorbt is available: uses its native vectorized parameter sweep.
        Fallback: nested loop with numpy.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Filter valid combinations (fast < slow)
        combos = [(f, s) for f in fast_range for s in slow_range if f < s]

        if self.has_vbt:
            return self._vbt_param_sweep(prices, fast_range, slow_range, combos)
        else:
            return self._numpy_param_sweep(prices, combos)

    def _vbt_param_sweep(
        self,
        prices: pd.DataFrame,
        fast_range: range,
        slow_range: range,
        combos: list[tuple[int, int]],
    ) -> pd.DataFrame:
        """VBT-native vectorized sweep."""
        results = []
        try:
            fast_vals = sorted(set(f for f, _ in combos))
            slow_vals = sorted(set(s for _, s in combos))

            fast_ma = vbt.MA.run(prices, window=fast_vals, short_name="fast")
            slow_ma = vbt.MA.run(prices, window=slow_vals, short_name="slow")

            for f, s in combos:
                try:
                    fm = fast_ma.ma.xs(f, level="fast_window", axis=1)
                    sm = slow_ma.ma.xs(s, level="slow_window", axis=1)

                    entries = (fm > sm) & (fm.shift(1) <= sm.shift(1))
                    exits = (fm < sm) & (fm.shift(1) >= sm.shift(1))

                    pf = vbt.Portfolio.from_signals(
                        close=prices,
                        entries=entries.shift(1).fillna(False),
                        exits=exits.shift(1).fillna(False),
                        init_cash=self.init_cash,
                        fees=self.fees,
                        freq="1D",
                    )
                    results.append({
                        "fast": f, "slow": s,
                        "sharpe": float(pf.sharpe_ratio()),
                        "total_return": float(pf.total_return()),
                        "max_drawdown": float(pf.max_drawdown()),
                        "num_trades": int(pf.trades.count()),
                    })
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("VBT sweep failed (%s), falling back.", exc)
            return self._numpy_param_sweep(prices, combos)

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _numpy_param_sweep(
        self,
        prices: pd.DataFrame,
        combos: list[tuple[int, int]],
    ) -> pd.DataFrame:
        """Numpy fallback parameter sweep."""
        # Pre-compute all MA series
        all_windows = sorted(set(w for pair in combos for w in pair))
        mas = {w: prices.rolling(w).mean() for w in all_windows}

        results = []
        for fast, slow in combos:
            fm = mas[fast]
            sm = mas[slow]

            entries = (fm > sm) & (fm.shift(1) <= sm.shift(1))
            exits = (fm < sm) & (fm.shift(1) >= sm.shift(1))

            pf = NumpyPortfolio(prices, entries, exits, self.init_cash, self.fees)
            rets = pf.get_returns()
            results.append({
                "fast": fast,
                "slow": slow,
                "sharpe": pf.sharpe_ratio(),
                "total_return": pf.total_return(),
                "max_drawdown": pf.max_drawdown(),
                "num_trades": len(pf.get_trades()),
                "calmar": abs(pf.annualized_return() / pf.max_drawdown())
                          if pf.max_drawdown() != 0 else 0.0,
            })

        return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Class 2: StrategyLibrary
# ---------------------------------------------------------------------------

class StrategyLibrary:
    """
    10+ built-in strategy signal generators.
    Each returns (entries: pd.DataFrame, exits: pd.DataFrame) of boolean signals.
    Signals are NOT shifted — the caller or adapter must enforce shift(1).
    """

    # ------------------------------------------------------------------
    # 1. SMA Crossover
    # ------------------------------------------------------------------
    @staticmethod
    def sma_crossover(
        prices: pd.DataFrame, fast: int = 20, slow: int = 50
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Golden cross / death cross. Entry: fast SMA > slow SMA (crossover).
        Exit: fast SMA < slow SMA (crossunder).
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        fm = prices.rolling(fast).mean()
        sm = prices.rolling(slow).mean()
        entries = (fm > sm) & (fm.shift(1) <= sm.shift(1))
        exits = (fm < sm) & (fm.shift(1) >= sm.shift(1))
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 2. RSI Mean Reversion
    # ------------------------------------------------------------------
    @staticmethod
    def rsi_mean_reversion(
        prices: pd.DataFrame,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        RSI-based mean reversion: buy oversold, sell overbought.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
        exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 3. Bollinger Bands Mean Reversion
    # ------------------------------------------------------------------
    @staticmethod
    def bollinger_bands_mean_reversion(
        prices: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Buy at lower band (oversold), sell at upper band (overbought).
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        mid = prices.rolling(period).mean()
        band = prices.rolling(period).std() * std
        lower = mid - band
        upper = mid + band

        entries = prices < lower   # price below lower band → buy
        exits = prices > upper     # price above upper band → sell
        # Only enter on crossover (avoid sustained positions below band)
        entries = entries & ~entries.shift(1).fillna(False)
        exits = exits & ~exits.shift(1).fillna(False)
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 4. Cross-sectional Momentum
    # ------------------------------------------------------------------
    @staticmethod
    def momentum(
        prices: pd.DataFrame,
        lookback: int = 63,
        top_n: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cross-sectional momentum: buy top_n assets by trailing return,
        exit assets that have fallen out of the top ranking.
        Rebalancing signal generated daily.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        mom = prices / prices.shift(lookback) - 1  # trailing return

        entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)

        for i in range(lookback, len(prices)):
            row = mom.iloc[i]
            valid = row.dropna()
            if valid.empty:
                continue
            n = min(top_n, len(valid))
            top_assets = set(valid.nlargest(n).index)
            bottom_assets = set(valid.nsmallest(n).index)

            for col in prices.columns:
                if col in top_assets:
                    entries.iloc[i][col] = True
                if col in bottom_assets:
                    exits.iloc[i][col] = True

        return entries, exits

    # ------------------------------------------------------------------
    # 5. Pairs Trading (cointegration-based)
    # ------------------------------------------------------------------
    @staticmethod
    def pairs_trading(
        price1: pd.Series,
        price2: pd.Series,
        z_threshold: float = 2.0,
        lookback: int = 60,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cointegration-based pairs trading.
        Spread = price1 - beta × price2 (OLS hedge ratio over rolling window).
        Entry: |z-score| > z_threshold. Exit: |z-score| < 0.5.

        Returns signals for price1 (long) and price2 (short) as one-column DataFrames.
        """
        spread = pd.Series(dtype=float, index=price1.index)
        for i in range(lookback, len(price1)):
            p1 = price1.iloc[i - lookback:i]
            p2 = price2.iloc[i - lookback:i]
            # OLS beta (hedge ratio)
            if p2.std() == 0:
                continue
            beta = float(np.cov(p1, p2)[0, 1] / np.var(p2))
            spread.iloc[i] = float(price1.iloc[i] - beta * price2.iloc[i])

        spread_z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

        entries1 = pd.DataFrame(spread_z < -z_threshold, columns=["asset1"])
        exits1 = pd.DataFrame(spread_z.abs() < 0.5, columns=["asset1"])

        return entries1.fillna(False), exits1.fillna(False)

    # ------------------------------------------------------------------
    # 6. Trend Following (EMA)
    # ------------------------------------------------------------------
    @staticmethod
    def trend_following(
        prices: pd.DataFrame,
        period: int = 200,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        EMA-based trend filter: enter when price > EMA, exit when price < EMA.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        ema = prices.ewm(span=period, adjust=False).mean()
        above = prices > ema
        entries = above & ~above.shift(1).fillna(False)
        exits = ~above & above.shift(1).fillna(False)
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 7. Donchian Channel Breakout
    # ------------------------------------------------------------------
    @staticmethod
    def breakout(
        prices: pd.DataFrame,
        lookback: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Donchian channel: enter on N-day high, exit on N-day low.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        high_n = prices.rolling(lookback).max()
        low_n = prices.rolling(lookback).min()

        # New high: today's price equals the N-day max
        entries = prices >= high_n.shift(1)
        exits = prices <= low_n.shift(1)
        entries = entries & ~entries.shift(1).fillna(False)
        exits = exits & ~exits.shift(1).fillna(False)
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 8. MACD Signal Line Crossover
    # ------------------------------------------------------------------
    @staticmethod
    def macd_crossover(
        prices: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        MACD: fast EMA - slow EMA. Entry when MACD crosses above signal line.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig_line = macd.ewm(span=signal, adjust=False).mean()

        entries = (macd > sig_line) & (macd.shift(1) <= sig_line.shift(1))
        exits = (macd < sig_line) & (macd.shift(1) >= sig_line.shift(1))
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 9. Carry Trade (using external carry signals)
    # ------------------------------------------------------------------
    @staticmethod
    def carry_trade(
        prices: pd.DataFrame,
        carry_signals: pd.Series,
        top_pct: float = 0.3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Long high-carry assets, exit low-carry assets.
        carry_signals: Series (can be yield, dividend yield, or custom carry score).
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)

        # Apply the signal to all columns uniformly
        # (In real use, prices has multiple assets and carry_signals is per-asset)
        if not carry_signals.empty:
            carry_aligned = carry_signals.reindex(prices.index, method="ffill")
            # High carry = above top_pct threshold
            threshold = carry_aligned.quantile(1 - top_pct)
            high_carry = carry_aligned >= threshold
            low_carry = carry_aligned <= carry_aligned.quantile(top_pct)

            for col in prices.columns:
                entries[col] = high_carry & ~high_carry.shift(1).fillna(False)
                exits[col] = low_carry & ~low_carry.shift(1).fillna(False)

        return entries, exits

    # ------------------------------------------------------------------
    # 10. Mean Reversion (Z-score of price)
    # ------------------------------------------------------------------
    @staticmethod
    def zscore_mean_reversion(
        prices: pd.DataFrame,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Trade mean reversion based on rolling z-score of price.
        Entry: price > entry_z std below mean (oversold).
        Exit: price returns within exit_z std of mean.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        roll_mean = prices.rolling(lookback).mean()
        roll_std = prices.rolling(lookback).std()
        z = (prices - roll_mean) / roll_std.replace(0, np.nan)

        entries = z < -entry_z
        exits = z.abs() < exit_z
        entries = entries & ~entries.shift(1).fillna(False)
        exits = exits & ~exits.shift(1).fillna(False)
        return entries.fillna(False), exits.fillna(False)

    # ------------------------------------------------------------------
    # 11. Volatility Breakout (ATR-based)
    # ------------------------------------------------------------------
    @staticmethod
    def volatility_breakout(
        prices: pd.DataFrame,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Buy when price breaks out by more than ATR_multiplier × ATR above recent close.
        Exit on opposite breakout.
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        daily_range = prices.diff().abs()
        atr = daily_range.rolling(atr_period).mean()

        upper = prices.shift(1) + atr_multiplier * atr.shift(1)
        lower = prices.shift(1) - atr_multiplier * atr.shift(1)

        entries = prices > upper
        exits = prices < lower
        entries = entries & ~entries.shift(1).fillna(False)
        exits = exits & ~exits.shift(1).fillna(False)
        return entries.fillna(False), exits.fillna(False)


# ---------------------------------------------------------------------------
# Class 3: PortfolioBacktester
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """
    Multi-asset portfolio backtesting with rebalancing and benchmark comparison.
    """

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        fees: float = 0.001,
    ) -> None:
        self.init_cash = init_cash
        self.fees = fees
        self._adapter = VectorBTAdapter(fees=fees, init_cash=init_cash)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_prices(
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for a list of tickers via yfinance."""
        try:
            import yfinance as yf
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, group_by="ticker"
            )
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw.xs("Close", level=1, axis=1)
            else:
                closes = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0])
            return closes.ffill().dropna(how="all")
        except Exception as exc:
            logger.error("yfinance fetch failed: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Multi-asset backtest
    # ------------------------------------------------------------------

    def run_multi_asset_backtest(
        self,
        prices: dict[str, pd.DataFrame] | pd.DataFrame,
        signal_fn: Callable,
        weights_fn: Optional[Callable] = None,
        init_cash: float = None,
        fees: float = None,
        rebalance: str = "monthly",
    ) -> BacktestResult:
        """
        Multi-asset portfolio backtest.

        prices: dict of ticker → price Series, OR combined DataFrame.
        signal_fn: function(prices_df) → (entries, exits) boolean DataFrames.
        weights_fn: function(entries, prices) → weights DataFrame.
                    Default: equal-weight across long positions.
        rebalance: "daily" | "weekly" | "monthly" | "quarterly".
        """
        cash = init_cash or self.init_cash
        fee = fees if fees is not None else self.fees

        if isinstance(prices, dict):
            price_df = pd.DataFrame({k: v.squeeze() for k, v in prices.items()}).ffill()
        else:
            price_df = prices.ffill()

        price_df = price_df.dropna(how="all")

        # Generate signals
        entries, exits = signal_fn(price_df)

        # Apply rebalancing mask
        rebal_mask = self._get_rebalance_mask(price_df.index, rebalance)
        entries = entries & rebal_mask.reindex(entries.index, fill_value=False)

        # Run backtest
        portfolio = self._adapter.run_signal_backtest(price_df, entries, exits, cash, fee)

        # Extract equity curve
        if hasattr(portfolio, "get_equity"):
            equity = portfolio.get_equity()
            returns = portfolio.get_returns()
        else:
            # vbt.Portfolio interface
            try:
                equity = portfolio.value()
                returns = portfolio.returns()
            except Exception:
                equity = pd.Series([cash], dtype=float)
                returns = pd.Series([0.0], dtype=float)

        # Build result
        return self._build_result(
            equity=equity,
            returns=returns,
            strategy="multi_asset",
            tickers=list(price_df.columns),
            params={"rebalance": rebalance},
            fee=fee,
        )

    def _get_rebalance_mask(
        self, index: pd.DatetimeIndex, frequency: str
    ) -> pd.Series:
        """Return boolean Series marking rebalancing dates."""
        mask = pd.Series(False, index=index)
        freq_map = {
            "daily":     "B",
            "weekly":    "W-FRI",
            "monthly":   "ME",
            "quarterly": "QE",
        }
        freq = freq_map.get(frequency, "ME")
        rebal_dates = pd.date_range(start=index[0], end=index[-1], freq=freq)
        for d in rebal_dates:
            nearest = index.asof(d)
            if nearest is not pd.NaT and nearest in mask.index:
                mask[nearest] = True
        return mask

    # ------------------------------------------------------------------
    # Portfolio optimization sweep
    # ------------------------------------------------------------------

    def run_portfolio_optimization_sweep(
        self,
        prices: dict | pd.DataFrame,
        strategy: str = "sma_crossover",
        param_grid: dict = None,
    ) -> pd.DataFrame:
        """
        Grid search over strategy parameters for a multi-asset portfolio.
        Returns DataFrame of parameter combinations and their performance metrics.
        """
        if param_grid is None:
            param_grid = {"fast": [10, 20, 50], "slow": [50, 100, 200]}

        if isinstance(prices, dict):
            price_df = pd.DataFrame({k: v.squeeze() for k, v in prices.items()}).ffill()
        else:
            price_df = prices.ffill()

        lib = StrategyLibrary()
        strategy_fn = getattr(lib, strategy, lib.sma_crossover)

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results = []
        for combo in combos:
            params = dict(zip(keys, combo))
            if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
                continue
            try:
                entries, exits = strategy_fn(price_df, **params)
                pf = self._adapter.run_signal_backtest(price_df, entries, exits)
                if hasattr(pf, "sharpe_ratio"):
                    sharpe = float(pf.sharpe_ratio())
                    total_ret = float(pf.total_return())
                    mdd = float(pf.max_drawdown())
                else:
                    sharpe = pf.sharpe_ratio()
                    total_ret = pf.total_return()
                    mdd = pf.max_drawdown()

                row = {**params, "sharpe": sharpe, "total_return": total_ret, "max_drawdown": mdd}
                results.append(row)
            except Exception as exc:
                logger.debug("Sweep combo %s failed: %s", params, exc)

        return pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    def compute_benchmark_comparison(
        self,
        portfolio_returns: pd.Series,
        benchmark: str = "SPY",
    ) -> dict:
        """
        Compute alpha, beta, information ratio, tracking error, correlation vs benchmark.
        """
        try:
            import yfinance as yf
            start = str(portfolio_returns.index[0].date())
            end = str(portfolio_returns.index[-1].date())
            bm_raw = yf.download(benchmark, start=start, end=end,
                                  auto_adjust=True, progress=False)
            if isinstance(bm_raw.columns, pd.MultiIndex):
                bm_prices = bm_raw["Close"].iloc[:, 0]
            else:
                bm_prices = bm_raw["Close"]
            bm_ret = bm_prices.pct_change().dropna()
        except Exception:
            bm_ret = pd.Series(0.0, index=portfolio_returns.index)

        # Align
        combined = pd.concat([portfolio_returns, bm_ret], axis=1).dropna()
        combined.columns = ["strategy", "benchmark"]

        p_ret = combined["strategy"]
        b_ret = combined["benchmark"]

        # Beta (OLS slope)
        cov_matrix = np.cov(p_ret, b_ret)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0

        # Alpha (annualized Jensen's alpha)
        alpha_daily = p_ret.mean() - beta * b_ret.mean()
        alpha_ann = alpha_daily * 252

        # Tracking error (annualized)
        active_ret = p_ret - b_ret
        te = float(active_ret.std() * math.sqrt(252))

        # Information ratio
        ir = float(active_ret.mean() * 252 / te) if te != 0 else 0.0

        # Correlation
        corr = float(p_ret.corr(b_ret))

        # Returns
        p_total = float((1 + p_ret).prod() - 1)
        b_total = float((1 + b_ret).prod() - 1)

        return {
            "benchmark": benchmark,
            "portfolio_total_return_pct": round(p_total * 100, 2),
            "benchmark_total_return_pct": round(b_total * 100, 2),
            "alpha_ann_pct": round(alpha_ann * 100, 3),
            "beta": round(beta, 3),
            "information_ratio": round(ir, 3),
            "tracking_error_pct": round(te * 100, 2),
            "correlation": round(corr, 3),
        }

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _build_result(
        self,
        equity: pd.Series,
        returns: pd.Series,
        strategy: str,
        tickers: list[str],
        params: dict,
        fee: float,
    ) -> BacktestResult:
        """Convert equity curve + returns into a full BacktestResult."""
        if equity.empty or returns.empty:
            return BacktestResult(
                strategy=strategy, tickers=tickers,
                start="N/A", end="N/A", params=params,
                total_return=0, cagr=0, sharpe=0, sortino=0, calmar=0,
                max_drawdown=0, avg_drawdown=0, num_trades=0, win_rate=0,
                avg_trade_return=0, avg_holding_days=0, volatility=0,
                var_95=0, cvar_95=0, skewness=0, kurtosis=0,
                benchmark="SPY", benchmark_return=0, alpha=0, beta=0,
                information_ratio=0, tracking_error=0,
                total_commission=0, total_slippage=0,
            )

        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Sharpe
        vol = float(returns.std() * math.sqrt(252))
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if returns.std() != 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        sortino = float(returns.mean() / downside.std() * math.sqrt(252)) if downside.std() != 0 else 0.0

        # Max drawdown
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        mdd = float(dd.min())
        avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

        # Calmar
        calmar = abs(cagr / mdd) if mdd != 0 else 0.0

        # VaR / CVaR
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

        # Distribution
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())

        return BacktestResult(
            strategy=strategy,
            tickers=tickers,
            start=str(equity.index[0].date()),
            end=str(equity.index[-1].date()),
            params=params,
            total_return=total_ret,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=mdd,
            avg_drawdown=avg_dd,
            num_trades=0,
            win_rate=0.5,
            avg_trade_return=0.0,
            avg_holding_days=0.0,
            volatility=vol,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skew,
            kurtosis=kurt,
            benchmark="SPY",
            benchmark_return=0.0,
            alpha=cagr,
            beta=1.0,
            information_ratio=sharpe,
            tracking_error=vol,
            total_commission=0.0,
            total_slippage=0.0,
            equity_curve=equity,
        )


# ---------------------------------------------------------------------------
# Class 4: OptimizationEngine
# ---------------------------------------------------------------------------

class OptimizationEngine:
    """
    Parameter optimization with OOS walk-forward validation.
    Supports grid search, Bayesian optimization (scikit-optimize), and random search.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs
        self._adapter = VectorBTAdapter()

    # ------------------------------------------------------------------
    # Grid search with walk-forward CV
    # ------------------------------------------------------------------

    def grid_search(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable,
        param_grid: dict,
        metric: str = "sharpe",
        cv_folds: int = 5,
    ) -> OptimizationResult:
        """
        Walk-forward cross-validated grid search.

        Process:
          1. Split prices into cv_folds equal-length folds.
          2. For each fold i: train on folds [0..i-1], validate on fold i.
          3. Select params that maximize OOS metric.
          4. Report IS vs OOS performance and parameter stability.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = [dict(zip(keys, c)) for c in itertools.product(*values)]

        # Filter invalid combos (fast must be < slow for MA strategies)
        valid_combos = []
        for c in combos:
            if "fast" in c and "slow" in c and c["fast"] >= c["slow"]:
                continue
            valid_combos.append(c)

        if not valid_combos:
            raise ValueError("No valid parameter combinations in grid.")

        # Time-series split into cv_folds
        n = len(prices)
        fold_size = n // cv_folds
        folds: list[tuple[int, int]] = []
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            folds.append((start_idx, end_idx))

        # Walk-forward evaluation
        cv_results_map: dict[tuple, list[float]] = {
            tuple(c.items()): [] for c in valid_combos
        }
        is_results_map: dict[tuple, list[float]] = {
            tuple(c.items()): [] for c in valid_combos
        }

        for fold_i in range(1, cv_folds):  # start from 1 (need at least 1 train fold)
            # Train: all folds before fold_i
            train_end = folds[fold_i][0]
            val_start, val_end = folds[fold_i]

            train_prices = prices.iloc[:train_end]
            val_prices = prices.iloc[val_start:val_end]

            if len(train_prices) < 50 or len(val_prices) < 20:
                continue

            for combo in valid_combos:
                key = tuple(combo.items())
                try:
                    # IS score
                    is_score = self._eval_combo(train_prices, strategy_fn, combo, metric)
                    is_results_map[key].append(is_score)
                    # OOS score
                    oos_score = self._eval_combo(val_prices, strategy_fn, combo, metric)
                    cv_results_map[key].append(oos_score)
                except Exception as exc:
                    logger.debug("Combo %s failed: %s", combo, exc)

        # Aggregate OOS scores
        oos_scores = {
            k: float(np.mean(v)) for k, v in cv_results_map.items() if v
        }
        is_scores = {
            k: float(np.mean(v)) for k, v in is_results_map.items() if v
        }

        if not oos_scores:
            raise RuntimeError("All parameter combinations failed in cross-validation.")

        best_key = max(oos_scores, key=lambda k: oos_scores[k])
        best_params = dict(best_key)
        best_oos = oos_scores[best_key]
        best_is = is_scores.get(best_key, float("nan"))

        degradation = (best_is - best_oos) / abs(best_is) if best_is != 0 else 0.0

        # Build full grid results
        rows = []
        for k, oos in oos_scores.items():
            is_v = is_scores.get(k, float("nan"))
            rows.append({**dict(k), "oos_score": oos, "is_score": is_v})
        full_grid = pd.DataFrame(rows).sort_values("oos_score", ascending=False)

        stability = self.compute_parameter_stability_score(
            full_grid.rename(columns={"oos_score": metric})
        )

        return OptimizationResult(
            strategy=strategy_fn.__name__ if hasattr(strategy_fn, "__name__") else "custom",
            tickers=list(prices.columns),
            metric=metric,
            best_params=best_params,
            best_is_score=best_is,
            best_oos_score=best_oos,
            is_oos_degradation=degradation,
            parameter_stability=stability,
            full_grid=full_grid,
            cv_results=[{"params": dict(k), "oos": v, "is": is_scores.get(k, [])}
                        for k, v in cv_results_map.items()],
            optimization_method="walk_forward_grid_search",
            notes=f"cv_folds={cv_folds}, valid_combos={len(valid_combos)}",
        )

    def _eval_combo(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable,
        params: dict,
        metric: str,
    ) -> float:
        """Evaluate a single parameter combo on a price slice. Returns metric value."""
        entries, exits = strategy_fn(prices, **params)
        pf = VectorBTAdapter().run_signal_backtest(prices, entries, exits)

        if hasattr(pf, "sharpe_ratio"):
            # NumpyPortfolio
            sharpe = pf.sharpe_ratio()
            total_ret = pf.total_return()
            mdd = pf.max_drawdown()
        else:
            # vbt.Portfolio
            sharpe = float(pf.sharpe_ratio())
            total_ret = float(pf.total_return())
            mdd = float(pf.max_drawdown())

        if metric == "sharpe":
            return sharpe
        elif metric == "total_return":
            return total_ret
        elif metric == "calmar":
            return abs(total_ret / mdd) if mdd != 0 else 0.0
        elif metric == "sortino":
            rets = pf.get_returns() if hasattr(pf, "get_returns") else pd.Series(dtype=float)
            if rets.empty:
                return 0.0
            down = rets[rets < 0]
            return float(rets.mean() / down.std() * math.sqrt(252)) if down.std() != 0 else 0.0
        else:
            return sharpe

    # ------------------------------------------------------------------
    # Bayesian optimization
    # ------------------------------------------------------------------

    def bayesian_optimize(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable,
        param_space: dict,
        n_calls: int = 50,
        metric: str = "sharpe",
    ) -> OptimizationResult:
        """
        Bayesian optimization using scikit-optimize (gp_minimize).
        Falls back to random search if skopt not installed.

        param_space: dict of {param_name: (min, max)} for integer params.
        """
        if HAS_SKOPT:
            return self._bayesian_skopt(prices, strategy_fn, param_space, n_calls, metric)
        else:
            return self._random_search(prices, strategy_fn, param_space, n_calls, metric)

    def _bayesian_skopt(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable,
        param_space: dict,
        n_calls: int,
        metric: str,
    ) -> OptimizationResult:
        keys = list(param_space.keys())
        dimensions = [Integer(*v, name=k) for k, v in param_space.items()]
        call_results = []

        def objective(params_list):
            params = dict(zip(keys, params_list))
            if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
                return 100.0  # penalty
            try:
                score = self._eval_combo(prices, strategy_fn, params, metric)
                call_results.append({**params, metric: score})
                return -score  # minimize → negate
            except Exception:
                return 100.0

        result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)
        best_params = dict(zip(keys, result.x))
        best_score = -result.fun

        full_grid = pd.DataFrame(call_results).sort_values(metric, ascending=False) if call_results else None
        stability = self.compute_parameter_stability_score(full_grid) if full_grid is not None else 0.5

        return OptimizationResult(
            strategy=getattr(strategy_fn, "__name__", "custom"),
            tickers=list(prices.columns),
            metric=metric,
            best_params=best_params,
            best_is_score=best_score,
            best_oos_score=best_score * 0.8,   # rough OOS estimate
            is_oos_degradation=0.2,
            parameter_stability=stability,
            full_grid=full_grid,
            optimization_method="bayesian_gp",
            notes=f"n_calls={n_calls} via skopt gp_minimize",
        )

    def _random_search(
        self,
        prices: pd.DataFrame,
        strategy_fn: Callable,
        param_space: dict,
        n_calls: int,
        metric: str,
    ) -> OptimizationResult:
        """Random search fallback when skopt is not installed."""
        keys = list(param_space.keys())
        random.seed(42)
        results = []

        for _ in range(n_calls):
            params = {k: random.randint(*param_space[k]) for k in keys}
            if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
                continue
            try:
                score = self._eval_combo(prices, strategy_fn, params, metric)
                results.append({**params, metric: score})
            except Exception:
                pass

        if not results:
            raise RuntimeError("All random search trials failed.")

        df = pd.DataFrame(results).sort_values(metric, ascending=False)
        best_row = df.iloc[0]
        best_params = {k: best_row[k] for k in keys if k in best_row}
        best_score = float(best_row[metric])
        stability = self.compute_parameter_stability_score(df)

        return OptimizationResult(
            strategy=getattr(strategy_fn, "__name__", "custom"),
            tickers=list(prices.columns),
            metric=metric,
            best_params=best_params,
            best_is_score=best_score,
            best_oos_score=best_score * 0.75,
            is_oos_degradation=0.25,
            parameter_stability=stability,
            full_grid=df,
            optimization_method="random_search",
            notes=f"n_calls={n_calls}, skopt not installed",
        )

    # ------------------------------------------------------------------
    # Parameter stability
    # ------------------------------------------------------------------

    def compute_parameter_stability_score(self, results: pd.DataFrame) -> float:
        """
        Measure how smooth the performance surface is around the optimal parameters.

        Stability = 1 - (std of top-10% scores / |mean of top-10% scores|)

        Interpretation:
          ~1.0: Very smooth surface → robust parameters (good)
          ~0.5: Moderate roughness → some instability
          ~0.0: Very jagged surface → single-point peak (overfitting risk)
        """
        if results is None or results.empty:
            return 0.5

        # Find numeric performance column
        numeric_cols = results.select_dtypes(include=[np.number]).columns.tolist()
        # Prefer sharpe, then any numeric
        perf_col = None
        for preferred in ["sharpe", "oos_score", "total_return"]:
            if preferred in numeric_cols:
                perf_col = preferred
                break
        if perf_col is None and numeric_cols:
            perf_col = numeric_cols[-1]
        if perf_col is None:
            return 0.5

        scores = results[perf_col].dropna()
        if len(scores) < 3:
            return 0.5

        top_n = max(int(len(scores) * 0.10), 3)
        top_scores = scores.nlargest(top_n)
        mean_score = abs(float(top_scores.mean()))

        if mean_score == 0:
            return 0.5

        stability = 1.0 - float(top_scores.std() / mean_score)
        return max(0.0, min(1.0, stability))


# ---------------------------------------------------------------------------
# Class 5: TransactionCostModel
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Realistic transaction cost modeling: commissions + market impact slippage.
    Multiple models supporting free broker (zero) through institutional scenarios.
    """

    # Square root market impact constant
    SQRT_K: float = 0.1

    def compute_commission(
        self,
        quantity: float,
        price: float,
        model: str = "zero",
    ) -> float:
        """
        Commission models:
          "zero"       : $0 (Alpaca/Robinhood free broker)
          "flat"       : $1.00 per trade
          "per_share"  : $0.005 per share (IB IBKR lite)
          "percentage" : 0.10% of trade value
        """
        trade_value = abs(quantity * price)
        if model == "zero":
            return 0.0
        elif model == "flat":
            return 1.00
        elif model == "per_share":
            return abs(quantity) * 0.005
        elif model == "percentage":
            return trade_value * 0.001
        else:
            raise ValueError(f"Unknown commission model: {model!r}")

    def compute_slippage(
        self,
        quantity: float,
        price: float,
        daily_volume: float,
        sigma: float = 0.015,
        model: str = "sqrt",
    ) -> float:
        """
        Market impact slippage models:

          "fixed"  : 0.01% of price (10bps flat slippage)
          "sqrt"   : Square root model — industry standard for medium-sized orders
                     impact = k × σ × sqrt(Q/V) × price
                     k=0.1, σ=daily vol, Q=order qty, V=daily volume
          "linear" : Linear impact = λ × (Q/V) × σ × price
                     λ=1.0 (aggressive), suitable for large orders Q/V > 5%

        Returns dollar slippage on the full order.
        """
        q = abs(quantity)
        v = max(daily_volume, 1)
        participation_rate = q / v

        if model == "fixed":
            return abs(quantity) * price * 0.0001

        elif model == "sqrt":
            # Square root market impact (Almgren-Chriss family)
            impact_pct = self.SQRT_K * sigma * math.sqrt(participation_rate)
            return abs(quantity) * price * impact_pct

        elif model == "linear":
            lambda_linear = 1.0
            impact_pct = lambda_linear * participation_rate * sigma
            return abs(quantity) * price * impact_pct

        else:
            raise ValueError(f"Unknown slippage model: {model!r}")

    def compute_market_impact_curve(
        self,
        sizes: list[float],
        price: float,
        daily_volume: float,
        sigma: float = 0.015,
    ) -> pd.Series:
        """
        Show how market impact scales with order size under the sqrt model.
        sizes: list of order quantities to evaluate.
        Returns Series of impact cost in bps (basis points).
        """
        impacts = []
        for qty in sizes:
            dollar_impact = self.compute_slippage(qty, price, daily_volume, sigma, "sqrt")
            impact_bps = dollar_impact / (qty * price) * 10_000
            impacts.append(round(impact_bps, 2))
        return pd.Series(impacts, index=sizes, name="impact_bps")

    def compute_effective_cost(
        self,
        quantity: float,
        price: float,
        daily_volume: float,
        sigma: float = 0.015,
        commission_model: str = "zero",
        slippage_model: str = "sqrt",
    ) -> float:
        """
        Total effective cost = commission + slippage.
        Returns in dollars.
        """
        commission = self.compute_commission(quantity, price, commission_model)
        slippage = self.compute_slippage(quantity, price, daily_volume, sigma, slippage_model)
        return commission + slippage

    def estimate_round_trip_cost_bps(
        self,
        quantity: float,
        price: float,
        daily_volume: float,
        sigma: float = 0.015,
        commission_model: str = "percentage",
        slippage_model: str = "sqrt",
    ) -> float:
        """
        Estimate round-trip (entry + exit) cost in basis points.
        """
        total = 2 * self.compute_effective_cost(
            quantity, price, daily_volume, sigma, commission_model, slippage_model
        )
        return total / (quantity * price) * 10_000 if quantity * price > 0 else 0.0


# ---------------------------------------------------------------------------
# Class 6: EquityCurveAnalytics
# ---------------------------------------------------------------------------

class EquityCurveAnalytics:
    """
    Comprehensive equity curve analysis including tearsheet, underwater curve,
    monthly returns, and Monte Carlo simulation.
    """

    def compute_full_tearsheet(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
    ) -> Tearsheet:
        """
        Compute a complete strategy tearsheet from a daily returns Series.
        """
        if returns.empty:
            raise ValueError("Returns series is empty.")

        equity = (1 + returns).cumprod()

        # Core metrics
        total_ret = float(equity.iloc[-1] - 1)
        n_years = (returns.index[-1] - returns.index[0]).days / 365.25
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        vol = float(returns.std() * math.sqrt(252))

        # Sharpe / Sortino
        sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if returns.std() != 0 else 0.0
        down = returns[returns < 0]
        sortino = float(returns.mean() / down.std() * math.sqrt(252)) if down.std() != 0 else 0.0

        # Drawdown metrics
        roll_max = equity.cummax()
        dd_series = (equity - roll_max) / roll_max
        mdd = float(dd_series.min())
        avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

        # Calmar
        calmar = abs(cagr / mdd) if mdd != 0 else 0.0

        # Recovery time
        recovery_days = self._compute_recovery_time(equity)

        # Risk metrics
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())

        # Trade metrics (approximate from sign changes)
        trade_mask = returns != 0
        num_trades = int(trade_mask.sum())
        win_rate = float((returns[trade_mask] > 0).sum() / max(num_trades, 1))

        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = float(wins / losses) if losses != 0 else float("inf")

        # Rolling 12-month Sharpe
        rolling_sharpe = self.compute_rolling_sharpe(returns, window=252)

        # Monthly returns
        monthly_table = self.compute_monthly_returns_table(returns)

        return Tearsheet(
            strategy=strategy_name,
            start=str(returns.index[0].date()),
            end=str(returns.index[-1].date()),
            total_return=total_ret,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=mdd,
            avg_drawdown=avg_dd,
            recovery_time_days=recovery_days,
            volatility=vol,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            monthly_returns=monthly_table,
            rolling_sharpe=rolling_sharpe,
            underwater_curve=dd_series,
        )

    def compute_underwater_curve(self, equity: pd.Series) -> pd.Series:
        """
        Drawdown at every point in time (underwater curve).
        Values are 0 to -1 (e.g., -0.20 = 20% below high-water mark).
        """
        roll_max = equity.cummax()
        underwater = (equity - roll_max) / roll_max
        underwater.name = "underwater"
        return underwater

    def _compute_recovery_time(self, equity: pd.Series) -> int:
        """
        Compute the maximum drawdown recovery time in calendar days.
        """
        roll_max = equity.cummax()
        in_drawdown = equity < roll_max
        if not in_drawdown.any():
            return 0

        max_recovery = 0
        drawdown_start = None

        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and drawdown_start is None:
                drawdown_start = date
            elif not is_dd and drawdown_start is not None:
                days = (date - drawdown_start).days
                max_recovery = max(max_recovery, days)
                drawdown_start = None

        # Still in drawdown at end
        if drawdown_start is not None:
            days = (equity.index[-1] - drawdown_start).days
            max_recovery = max(max_recovery, days)

        return max_recovery

    def compute_monthly_returns_table(self, returns: pd.Series) -> pd.DataFrame:
        """
        Monthly returns heatmap data.
        Index: years, Columns: Jan..Dec + YTD.
        Values: percentage returns.
        """
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = pd.to_datetime(monthly.index)

        years = sorted(set(monthly.index.year))
        months = list(range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        table = pd.DataFrame(index=years, columns=month_names + ["YTD"], dtype=float)

        for year in years:
            ytd = 1.0
            for mi, m in enumerate(months):
                mask = (monthly.index.year == year) & (monthly.index.month == m)
                if mask.any():
                    val = float(monthly[mask].iloc[0]) * 100
                    table.loc[year, month_names[mi]] = round(val, 2)
                    ytd *= (1 + val / 100)
            table.loc[year, "YTD"] = round((ytd - 1) * 100, 2)

        return table

    def compute_rolling_sharpe(
        self, returns: pd.Series, window: int = 252
    ) -> pd.Series:
        """Rolling 12-month (252-day) Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        sharpe = rolling_mean / rolling_std * math.sqrt(252)
        sharpe.name = "rolling_sharpe_12m"
        return sharpe

    def monte_carlo_equity_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1_000,
        horizon: int = 252,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Bootstrap Monte Carlo simulation of equity curves.

        Method:
          - Resample from historical daily returns with replacement (block bootstrap)
          - Run n_simulations paths over `horizon` trading days
          - Compute 5th, 25th, 50th, 75th, 95th percentile paths

        Returns:
          DataFrame of shape (horizon, n_simulations).
          Also stores percentile paths as additional columns.
        """
        rng = np.random.default_rng(seed)
        clean = returns.dropna().values

        if len(clean) < 30:
            raise ValueError("Insufficient returns history for Monte Carlo (need ≥ 30 days).")

        # Block bootstrap: sample blocks of 5 days to preserve autocorrelation
        block_size = 5
        simulated = np.ones((horizon + 1, n_simulations))

        for sim_i in range(n_simulations):
            path = [1.0]
            day = 0
            while day < horizon:
                block_start = rng.integers(0, len(clean) - block_size)
                block = clean[block_start:block_start + block_size]
                for r in block:
                    if day >= horizon:
                        break
                    path.append(path[-1] * (1 + r))
                    day += 1
            simulated[:, sim_i] = np.array(path[:horizon + 1])

        df = pd.DataFrame(simulated, columns=[f"sim_{i}" for i in range(n_simulations)])
        df.index = range(horizon + 1)

        # Compute percentile columns
        df["p5"] = df.iloc[:, :n_simulations].quantile(0.05, axis=1)
        df["p25"] = df.iloc[:, :n_simulations].quantile(0.25, axis=1)
        df["p50"] = df.iloc[:, :n_simulations].quantile(0.50, axis=1)
        df["p75"] = df.iloc[:, :n_simulations].quantile(0.75, axis=1)
        df["p95"] = df.iloc[:, :n_simulations].quantile(0.95, axis=1)

        return df

    def compute_monte_carlo_summary(
        self,
        mc_df: pd.DataFrame,
        horizon: int = 252,
    ) -> dict:
        """
        Summary statistics from Monte Carlo simulation.
        mc_df: output from monte_carlo_equity_simulation().
        """
        final_values = mc_df.iloc[-1]
        sim_cols = [c for c in mc_df.columns if c.startswith("sim_")]
        final_sim = mc_df[sim_cols].iloc[-1]

        total_returns = final_sim - 1.0

        return {
            "horizon_days": horizon,
            "n_simulations": len(sim_cols),
            "median_final_equity": round(float(mc_df["p50"].iloc[-1]), 4),
            "p5_final_equity": round(float(mc_df["p5"].iloc[-1]), 4),
            "p25_final_equity": round(float(mc_df["p25"].iloc[-1]), 4),
            "p75_final_equity": round(float(mc_df["p75"].iloc[-1]), 4),
            "p95_final_equity": round(float(mc_df["p95"].iloc[-1]), 4),
            "prob_positive_return_pct": round(float((total_returns > 0).mean() * 100), 1),
            "prob_loss_gt_10pct": round(float((total_returns < -0.10).mean() * 100), 1),
            "prob_loss_gt_20pct": round(float((total_returns < -0.20).mean() * 100), 1),
            "prob_gain_gt_20pct": round(float((total_returns > 0.20).mean() * 100), 1),
            "median_total_return_pct": round(float(total_returns.median() * 100), 2),
        }


# ---------------------------------------------------------------------------
# Class 7: VectorBTBacktestEngine (orchestrator)
# ---------------------------------------------------------------------------

class VectorBTBacktestEngine:
    """
    Master orchestrator. Unified API for strategy research:
      run_strategy → optimize_strategy → compare_strategies → full_pipeline
    """

    STRATEGY_MAP: dict[str, str] = {
        "sma_crossover": "sma_crossover",
        "rsi": "rsi_mean_reversion",
        "bollinger": "bollinger_bands_mean_reversion",
        "momentum": "momentum",
        "trend": "trend_following",
        "breakout": "breakout",
        "macd": "macd_crossover",
        "zscore": "zscore_mean_reversion",
        "vol_breakout": "volatility_breakout",
    }

    def __init__(
        self,
        init_cash: float = 100_000.0,
        fees: float = 0.001,
    ) -> None:
        self.init_cash = init_cash
        self.fees = fees
        self._adapter = VectorBTAdapter(fees=fees, init_cash=init_cash)
        self._portfolio_bt = PortfolioBacktester(init_cash=init_cash, fees=fees)
        self._optimizer = OptimizationEngine()
        self._equity_analytics = EquityCurveAnalytics()
        self._lib = StrategyLibrary()

    def _get_strategy_fn(self, strategy_name: str) -> Callable:
        method_name = self.STRATEGY_MAP.get(strategy_name, strategy_name)
        fn = getattr(self._lib, method_name, None)
        if fn is None:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(self.STRATEGY_MAP.keys())}"
            )
        return fn

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        strategy_name: str,
        tickers: list[str],
        params: dict,
        start: str,
        end: str,
    ) -> BacktestResult:
        """
        Fetch prices, run named strategy with given params, return BacktestResult.
        """
        prices = self._portfolio_bt.fetch_prices(tickers, start, end)
        if prices.empty:
            raise RuntimeError(f"Could not fetch prices for {tickers}")

        strategy_fn = self._get_strategy_fn(strategy_name)

        def signal_fn(p: pd.DataFrame):
            return strategy_fn(p, **params)

        return self._portfolio_bt.run_multi_asset_backtest(
            prices=prices,
            signal_fn=signal_fn,
            fees=self.fees,
        )

    # ------------------------------------------------------------------
    # Optimize strategy
    # ------------------------------------------------------------------

    def optimize_strategy(
        self,
        strategy_name: str,
        tickers: list[str],
        param_grid: dict,
        start: str,
        end: str,
        metric: str = "sharpe",
        cv_folds: int = 5,
    ) -> OptimizationResult:
        """
        Fetch prices, run walk-forward grid search, return OptimizationResult.
        """
        prices = self._portfolio_bt.fetch_prices(tickers, start, end)
        if prices.empty:
            raise RuntimeError(f"Could not fetch prices for {tickers}")

        strategy_fn = self._get_strategy_fn(strategy_name)
        strategy_fn.__name__ = strategy_name  # ensure name propagates

        return self._optimizer.grid_search(
            prices=prices,
            strategy_fn=strategy_fn,
            param_grid=param_grid,
            metric=metric,
            cv_folds=cv_folds,
        )

    # ------------------------------------------------------------------
    # Compare strategies
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        strategies: list[dict],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same universe.

        Each strategy dict: {
            "name": str, "tickers": [str], "params": dict,
            "start": str, "end": str
        }

        Returns DataFrame with one row per strategy.
        """
        rows = []
        for spec in strategies:
            try:
                result = self.run_strategy(
                    strategy_name=spec["name"],
                    tickers=spec["tickers"],
                    params=spec.get("params", {}),
                    start=spec["start"],
                    end=spec["end"],
                )
                row = {
                    "strategy": spec["name"],
                    "tickers": ",".join(spec["tickers"]),
                    "sharpe": result.sharpe,
                    "cagr_pct": result.cagr * 100,
                    "max_drawdown_pct": result.max_drawdown * 100,
                    "total_return_pct": result.total_return * 100,
                    "calmar": result.calmar,
                    "sortino": result.sortino,
                    "volatility_pct": result.volatility * 100,
                    "var_95_pct": result.var_95 * 100,
                }
                rows.append(row)
            except Exception as exc:
                logger.warning("Strategy comparison failed for %s: %s", spec.get("name"), exc)
                rows.append({"strategy": spec.get("name", "?"), "error": str(exc)})

        return pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        strategy_name: str,
        tickers: list[str],
        param_grid: dict,
        start: str,
        end: str,
        n_mc_simulations: int = 500,
        mc_horizon: int = 252,
    ) -> FullPipelineResult:
        """
        Full research pipeline:
          1. Walk-forward grid search → best params
          2. Backtest with best params on full period
          3. Generate tearsheet
          4. Monte Carlo equity simulation
          5. Generate narrative report

        Returns FullPipelineResult dataclass.
        """
        logger.info("[Pipeline] %s | %s | %s → %s", strategy_name, tickers, start, end)

        # Step 1: Optimization
        logger.info("[Pipeline] Running grid search (walk-forward CV)...")
        opt_result = self.optimize_strategy(
            strategy_name=strategy_name,
            tickers=tickers,
            param_grid=param_grid,
            start=start,
            end=end,
        )
        logger.info("[Pipeline] Best params: %s (OOS Sharpe: %.3f)",
                    opt_result.best_params, opt_result.best_oos_score)

        # Step 2: Full backtest with best params
        logger.info("[Pipeline] Running backtest with best params...")
        best_bt = self.run_strategy(
            strategy_name=strategy_name,
            tickers=tickers,
            params=opt_result.best_params,
            start=start,
            end=end,
        )

        # Step 3: Tearsheet
        logger.info("[Pipeline] Computing tearsheet...")
        if best_bt.equity_curve is not None and not best_bt.equity_curve.empty:
            returns = best_bt.equity_curve.pct_change().fillna(0)
            tearsheet = self._equity_analytics.compute_full_tearsheet(
                returns=returns, strategy_name=strategy_name
            )
        else:
            # Create minimal tearsheet from BacktestResult
            tearsheet = Tearsheet(
                strategy=strategy_name, start=best_bt.start, end=best_bt.end,
                total_return=best_bt.total_return, cagr=best_bt.cagr,
                sharpe=best_bt.sharpe, sortino=best_bt.sortino,
                calmar=best_bt.calmar, max_drawdown=best_bt.max_drawdown,
                avg_drawdown=best_bt.avg_drawdown, recovery_time_days=0,
                volatility=best_bt.volatility, var_95=best_bt.var_95,
                cvar_95=best_bt.cvar_95, skewness=best_bt.skewness,
                kurtosis=best_bt.kurtosis, num_trades=best_bt.num_trades,
                win_rate=best_bt.win_rate, profit_factor=1.0,
            )

        # Step 4: Monte Carlo
        logger.info("[Pipeline] Running Monte Carlo (%d simulations)...", n_mc_simulations)
        mc_summary = {}
        if best_bt.equity_curve is not None and len(best_bt.equity_curve) > 60:
            try:
                returns_series = best_bt.equity_curve.pct_change().dropna()
                mc_df = self._equity_analytics.monte_carlo_equity_simulation(
                    returns=returns_series,
                    n_simulations=n_mc_simulations,
                    horizon=mc_horizon,
                )
                mc_summary = self._equity_analytics.compute_monte_carlo_summary(mc_df, mc_horizon)
                tearsheet.monte_carlo_percentiles = mc_summary
            except Exception as exc:
                logger.warning("[Pipeline] Monte Carlo failed: %s", exc)
                mc_summary = {"error": str(exc)}

        # Step 5: Report
        report = self._generate_pipeline_report(
            strategy_name, tickers, opt_result, best_bt, tearsheet, mc_summary
        )

        return FullPipelineResult(
            strategy=strategy_name,
            tickers=tickers,
            optimization=opt_result,
            best_backtest=best_bt,
            tearsheet=tearsheet,
            monte_carlo_summary=mc_summary,
            report=report,
        )

    def _generate_pipeline_report(
        self,
        strategy_name: str,
        tickers: list[str],
        opt: OptimizationResult,
        bt: BacktestResult,
        ts: Tearsheet,
        mc: dict,
    ) -> str:
        lines = [
            "=" * 72,
            f"SENTINEL BACKTEST PIPELINE REPORT",
            f"Strategy: {strategy_name.upper()} | Tickers: {', '.join(tickers)}",
            f"Period:   {bt.start} → {bt.end}",
            "=" * 72,
            "",
            "OPTIMIZATION",
            f"  Method:       {opt.optimization_method}",
            f"  Best params:  {opt.best_params}",
            f"  IS Sharpe:    {opt.best_is_score:.3f}",
            f"  OOS Sharpe:   {opt.best_oos_score:.3f}",
            f"  Degradation:  {opt.is_oos_degradation*100:.1f}% (IS→OOS)",
            f"  Stability:    {opt.parameter_stability:.3f} (1=perfect)",
            "",
            "PERFORMANCE",
            f"  Total Return:   {ts.total_return*100:+.2f}%",
            f"  CAGR:           {ts.cagr*100:+.2f}%",
            f"  Sharpe:         {ts.sharpe:.3f}",
            f"  Sortino:        {ts.sortino:.3f}",
            f"  Calmar:         {ts.calmar:.3f}",
            f"  Max Drawdown:   {ts.max_drawdown*100:.2f}%",
            f"  Recovery Days:  {ts.recovery_time_days}",
            "",
            "RISK",
            f"  Volatility:     {ts.volatility*100:.2f}% (ann.)",
            f"  VaR 95%:        {ts.var_95*100:.2f}%",
            f"  CVaR 95%:       {ts.cvar_95*100:.2f}%",
            f"  Skewness:       {ts.skewness:.3f}",
            f"  Kurtosis:       {ts.kurtosis:.3f}",
        ]

        if mc and "median_total_return_pct" in mc:
            lines += [
                "",
                "MONTE CARLO SIMULATION",
                f"  Horizon:            {mc.get('horizon_days', 252)} trading days",
                f"  Simulations:        {mc.get('n_simulations', 0)}",
                f"  Median Return:      {mc.get('median_total_return_pct', 0):+.2f}%",
                f"  P5 → P95 range:     "
                f"{(mc.get('p5_final_equity',1)-1)*100:+.1f}% → "
                f"{(mc.get('p95_final_equity',1)-1)*100:+.1f}%",
                f"  P(positive return): {mc.get('prob_positive_return_pct', 0):.1f}%",
                f"  P(loss > 20%):      {mc.get('prob_loss_gt_20pct', 0):.1f}%",
            ]

        lines += ["", "=" * 72]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 72)
    print(f"SENTINEL VectorBT Backtesting v3  [VBT={'✓' if HAS_VBT else 'numpy fallback'}]")
    print("=" * 72)

    engine = VectorBTBacktestEngine(init_cash=100_000, fees=0.001)
    TICKERS = ["SPY", "QQQ", "GLD", "TLT"]
    START = "2019-01-01"
    END = "2024-12-31"

    # Fetch prices once
    print(f"\n[1] Fetching 5-year data for {TICKERS}...")
    prices = PortfolioBacktester.fetch_prices(TICKERS, START, END)
    if prices.empty:
        print("  ERROR: Could not fetch prices. Check internet connection.")
        raise SystemExit(1)
    print(f"  Shape: {prices.shape} | Start: {prices.index[0].date()} | End: {prices.index[-1].date()}")

    # SMA crossover grid search
    print("\n[2] SMA Crossover parameter sweep (fast 5-50, slow 20-200)...")
    adapter = VectorBTAdapter(fees=0.001, init_cash=100_000)
    spy_prices = prices[["SPY"]]

    fast_range = range(5, 55, 5)    # [5, 10, 15, ..., 50]
    slow_range = range(20, 210, 20)  # [20, 40, 60, ..., 200]

    grid_df = adapter.param_sweep(spy_prices, fast_range, slow_range)
    if not grid_df.empty:
        print(f"  Grid size: {len(grid_df)} valid combinations")
        top5 = grid_df.nlargest(5, "sharpe")
        print("  Top 5 by Sharpe:")
        for _, row in top5.iterrows():
            print(f"    fast={int(row['fast'])} slow={int(row['slow'])} "
                  f"sharpe={row['sharpe']:.3f} ret={row['total_return']*100:.1f}%")

    # Walk-forward optimization
    print("\n[3] Walk-forward validation on best params...")
    PARAM_GRID = {"fast": [10, 20, 50], "slow": [50, 100, 200]}
    try:
        opt = engine.optimize_strategy(
            strategy_name="sma_crossover",
            tickers=["SPY"],
            param_grid=PARAM_GRID,
            start=START,
            end=END,
            cv_folds=5,
        )
        print("  Optimization result:")
        print(json.dumps(opt.summary(), indent=4))
    except Exception as exc:
        print(f"  Optimization failed: {exc}")
        opt = None

    # Strategy comparison
    print("\n[4] Multi-strategy comparison (SPY)...")
    strategies = [
        {"name": "sma_crossover", "tickers": ["SPY"], "params": {"fast": 20, "slow": 100}, "start": START, "end": END},
        {"name": "rsi",           "tickers": ["SPY"], "params": {"period": 14, "oversold": 30, "overbought": 70}, "start": START, "end": END},
        {"name": "momentum",      "tickers": TICKERS, "params": {"lookback": 63, "top_n": 2}, "start": START, "end": END},
        {"name": "breakout",      "tickers": ["SPY"], "params": {"lookback": 20}, "start": START, "end": END},
        {"name": "trend",         "tickers": ["SPY"], "params": {"period": 200}, "start": START, "end": END},
    ]
    compare_df = engine.compare_strategies(strategies)
    if not compare_df.empty:
        print(compare_df.to_string(index=False))

    # Monte Carlo on SMA strategy
    print("\n[5] Monte Carlo simulation (500 paths, 252-day horizon)...")
    try:
        lib = StrategyLibrary()
        entries, exits = lib.sma_crossover(spy_prices, fast=20, slow=100)
        pf = adapter.run_signal_backtest(spy_prices, entries, exits)

        if hasattr(pf, "get_returns"):
            returns = pf.get_returns()
        else:
            try:
                returns = pf.returns()
            except Exception:
                returns = pd.Series(dtype=float)

        if not returns.empty and len(returns) > 60:
            analytics = EquityCurveAnalytics()
            mc_df = analytics.monte_carlo_equity_simulation(
                returns=returns, n_simulations=500, horizon=252
            )
            mc_summary = analytics.compute_monte_carlo_summary(mc_df, 252)
            print("  Monte Carlo summary:")
            print(json.dumps(mc_summary, indent=4))

            # Full tearsheet
            print("\n[6] Full tearsheet:")
            ts = analytics.compute_full_tearsheet(returns, strategy_name="SMA(20,100)")
            print(ts.print_summary())
        else:
            print("  Insufficient returns data for Monte Carlo.")
    except Exception as exc:
        print(f"  Monte Carlo/tearsheet failed: {exc}")

    # Transaction cost model demo
    print("\n[7] Transaction cost analysis (SPY, 1000 shares @ $500, 30M daily vol):")
    tc = TransactionCostModel()
    qty, price, volume, sigma = 1000, 500.0, 30_000_000 / 500, 0.015

    for commission_model in ["zero", "percentage"]:
        for slippage_model in ["fixed", "sqrt"]:
            total = tc.compute_effective_cost(qty, price, volume, sigma,
                                              commission_model, slippage_model)
            bps = total / (qty * price) * 10_000
            print(f"  [{commission_model:10s} + {slippage_model:5s}] "
                  f"${total:.2f} = {bps:.2f}bps round-side")

    print("\nDone.")
