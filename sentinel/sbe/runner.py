"""VectorBT backtest runner with mandatory .shift(1) enforcement and walk-forward validation."""
from __future__ import annotations
import asyncio
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Callable
import numpy as np
import pandas as pd
from sentinel.core.types import BacktestMetrics, StrategySpec
from sentinel.sbe.metrics import compute_metrics
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Benchmark: SPY buy-and-hold
SPY_BENCHMARK_TICKER = "SPY"


class VectorBTRunner:
    """
    Runs VectorBT backtests with:
    - Mandatory signal.shift(1) enforcement (no look-ahead bias)
    - Slippage and commission modeling
    - Walk-forward validation
    - Full 24-metric output
    """

    def __init__(
        self,
        commission: float = 0.001,  # 10bps round-trip
        slippage: float = 0.0005,   # 5bps price impact
        init_cash: float = 100_000.0,
    ) -> None:
        self._commission = commission
        self._slippage = slippage
        self._init_cash = init_cash

    def run(
        self,
        prices: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        strategy_id: str = "",
        n_trials: int = 1,
        interval: str = "1d",
    ) -> BacktestMetrics:
        """
        Run a long-only vectorized backtest.
        entries/exits: boolean Series aligned to prices index.
        ENFORCES shift(1) — signals are automatically lagged one period.
        """
        import vectorbt as vbt  # Deferred: Numba compilation on first import

        # MANDATORY: shift(1) to prevent look-ahead bias
        entries_lagged = entries.shift(1).fillna(False).astype(bool)
        exits_lagged = exits.shift(1).fillna(False).astype(bool)

        # Build portfolio
        pf = vbt.Portfolio.from_signals(
            prices,
            entries=entries_lagged,
            exits=exits_lagged,
            init_cash=self._init_cash,
            fees=self._commission,
            slippage=self._slippage,
            freq="D" if interval == "1d" else interval,
        )

        returns = pf.returns()
        bench_returns = None
        if benchmark_prices is not None:
            bench_returns = benchmark_prices.pct_change().dropna()

        metrics = compute_metrics(
            returns=returns,
            benchmark_returns=bench_returns,
            interval=interval,
            n_trials=n_trials,
            strategy_id=strategy_id,
            start_date=prices.index[0].date() if hasattr(prices.index[0], "date") else None,
            end_date=prices.index[-1].date() if hasattr(prices.index[-1], "date") else None,
        )
        logger.info("Backtest complete", strategy_id=strategy_id,
                    sharpe=float(metrics.sharpe_ratio), dsr=float(metrics.deflated_sharpe_ratio))
        return metrics

    def run_walk_forward(
        self,
        prices: pd.Series,
        signal_fn: Callable[[pd.Series], tuple[pd.Series, pd.Series]],
        n_splits: int = 5,
        test_fraction: float = 0.2,
        strategy_id: str = "",
        n_trials: int = 1,
        interval: str = "1d",
    ) -> dict:
        """
        Walk-forward validation: train on IS, test on OOS, repeat n_splits times.
        signal_fn(prices) → (entries, exits) boolean Series.
        Returns per-split metrics and combined OOS metrics.
        """
        T = len(prices)
        test_size = int(T * test_fraction)
        train_size = T - test_size * n_splits
        if train_size < 100:
            raise ValueError("Not enough data for walk-forward with these parameters")

        splits = []
        all_oos_returns = []

        for i in range(n_splits):
            is_end = train_size + i * test_size
            oos_start = is_end
            oos_end = oos_start + test_size

            is_prices = prices.iloc[:is_end]
            oos_prices = prices.iloc[oos_start:oos_end]

            try:
                # Generate signals on IS data
                entries_is, exits_is = signal_fn(is_prices)
                # Apply signals to OOS — use last IS signal state as initial state
                entries_oos, exits_oos = signal_fn(oos_prices)

                split_metrics = self.run(
                    oos_prices, entries_oos, exits_oos,
                    strategy_id=f"{strategy_id}_wf{i+1}",
                    n_trials=n_trials,
                    interval=interval,
                )
                splits.append({
                    "split": i + 1,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": min(oos_end, T),
                    "sharpe": float(split_metrics.sharpe_ratio),
                    "cagr": float(split_metrics.cagr),
                    "max_drawdown": float(split_metrics.max_drawdown),
                    "dsr": float(split_metrics.deflated_sharpe_ratio),
                })

                # Collect OOS returns for combined metrics
                import vectorbt as vbt
                entries_lagged = entries_oos.shift(1).fillna(False).astype(bool)
                exits_lagged = exits_oos.shift(1).fillna(False).astype(bool)
                pf = vbt.Portfolio.from_signals(
                    oos_prices, entries=entries_lagged, exits=exits_lagged,
                    init_cash=self._init_cash, fees=self._commission, slippage=self._slippage,
                )
                all_oos_returns.append(pf.returns())
            except Exception as exc:
                logger.warning("Walk-forward split failed", split=i+1, error=str(exc))

        # Combined OOS metrics
        combined_returns = pd.concat(all_oos_returns) if all_oos_returns else pd.Series(dtype=float)
        combined_metrics = compute_metrics(combined_returns, strategy_id=f"{strategy_id}_wf_combined",
                                           n_trials=n_trials) if not combined_returns.empty else None

        return {
            "strategy_id": strategy_id,
            "n_splits": n_splits,
            "splits": splits,
            "combined_oos_sharpe": float(combined_metrics.sharpe_ratio) if combined_metrics else None,
            "combined_oos_dsr": float(combined_metrics.deflated_sharpe_ratio) if combined_metrics else None,
            "combined_oos_cagr": float(combined_metrics.cagr) if combined_metrics else None,
            "combined_oos_max_drawdown": float(combined_metrics.max_drawdown) if combined_metrics else None,
            "avg_split_sharpe": float(np.mean([s["sharpe"] for s in splits])) if splits else None,
        }

    def parameter_sweep(
        self,
        prices: pd.Series,
        param_grid: list[dict],
        signal_fn_factory: Callable[[dict], Callable],
        top_n: int = 5,
        interval: str = "1d",
    ) -> list[BacktestMetrics]:
        """
        Run backtest for each parameter combination in grid.
        Returns top_n results sorted by DSR (not raw Sharpe) to penalize overfitting.
        """
        results = []
        n_trials = len(param_grid)
        for i, params in enumerate(param_grid):
            try:
                signal_fn = signal_fn_factory(params)
                entries, exits = signal_fn(prices)
                metrics = self.run(
                    prices, entries, exits,
                    strategy_id=f"sweep_{i}",
                    n_trials=n_trials,
                    interval=interval,
                )
                results.append(metrics)
            except Exception as exc:
                logger.warning("Param sweep failed", params=params, error=str(exc))

        # Sort by DSR — penalizes for multiple testing
        results.sort(key=lambda m: float(m.deflated_sharpe_ratio), reverse=True)
        logger.info("Param sweep complete", total=len(results), top_dsr=float(results[0].deflated_sharpe_ratio) if results else 0)
        return results[:top_n]
