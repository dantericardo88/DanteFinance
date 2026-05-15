"""
Vectorized backtesting engine: NumPy/Pandas vectorized signals,
fast portfolio simulation without VectorBT dependency.
Covers: signal generation, portfolio construction, performance metrics.

dim_061 — Vectorized backtesting (VectorBT) (target: 9)
"""

from __future__ import annotations

import sqlite3
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result store (in-memory + sqlite)
# ---------------------------------------------------------------------------
_RESULT_CACHE: Dict[str, "PortfolioResult"] = {}


def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS backtest_runs (
               run_id TEXT PRIMARY KEY,
               created_at TEXT,
               strategy_name TEXT,
               metrics_json TEXT
           )"""
    )
    conn.commit()
    return conn


_DB = _get_db_conn()


# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """Vectorized signal generation — all operations on full price matrices."""

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _pct_change(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        return prices.pct_change(periods=periods)

    @staticmethod
    def _rolling_z(series_2d: pd.DataFrame, window: int) -> pd.DataFrame:
        mu = series_2d.rolling(window).mean()
        sigma = series_2d.rolling(window).std(ddof=1)
        return (series_2d - mu) / sigma.replace(0, np.nan)

    @staticmethod
    def _rank_cross_section(df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise rank from 0 to 1 (percentile)."""
        return df.rank(axis=1, pct=True, na_option="keep")

    # -----------------------------------------------------------------------
    # momentum_signals
    # -----------------------------------------------------------------------

    def momentum_signals(
        self,
        prices: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
    ) -> pd.DataFrame:
        """
        Cross-sectional momentum: rank each asset by its lookback−skip return.
        Returns percentile ranks [0,1] at each time step.
        Signal > 0.8 → long candidate; < 0.2 → short candidate.
        """
        # momentum return: from t-lookback to t-skip
        ret = prices.shift(skip) / prices.shift(lookback) - 1
        ranked = self._rank_cross_section(ret)
        return ranked.where(ranked.notna(), other=np.nan)

    # -----------------------------------------------------------------------
    # mean_reversion_signals
    # -----------------------------------------------------------------------

    def mean_reversion_signals(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Mean-reversion entry signals based on rolling z-score.
        Returns:  +1 oversold (buy), -1 overbought (sell), 0 neutral.
        """
        z = self._rolling_z(prices, lookback)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[z < -z_threshold] = 1   # oversold → buy
        signals[z > z_threshold] = -1   # overbought → sell
        signals[z.isna()] = np.nan
        return signals

    # -----------------------------------------------------------------------
    # ma_crossover_signals
    # -----------------------------------------------------------------------

    def ma_crossover_signals(
        self,
        prices: pd.DataFrame,
        fast: int = 50,
        slow: int = 200,
    ) -> pd.DataFrame:
        """
        Golden cross (+1) / death cross (-1) signals.
        Returns change in regime: +1 (just crossed up), -1 (just crossed down), 0.
        """
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()
        above = (fast_ma > slow_ma).astype(float)
        above[fast_ma.isna() | slow_ma.isna()] = np.nan
        # derivative: cross events
        cross = above.diff()
        cross[above.isna()] = np.nan
        return cross

    # -----------------------------------------------------------------------
    # rsi_signals
    # -----------------------------------------------------------------------

    def rsi_signals(
        self,
        prices: pd.DataFrame,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> pd.DataFrame:
        """
        RSI-based entry/exit signals.
        Returns: +1 (RSI crossed above oversold), -1 (crossed below overbought), 0 neutral.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
        signals[rsi.isna()] = np.nan
        return signals

    # -----------------------------------------------------------------------
    # breakout_signals
    # -----------------------------------------------------------------------

    def breakout_signals(
        self,
        prices: pd.DataFrame,
        lookback: int = 52 * 5,  # ~52 weeks in days
    ) -> pd.DataFrame:
        """
        52-week high/low breakout signals.
        +1: price breaks above 52-week high; -1: breaks below 52-week low.
        """
        roll_high = prices.shift(1).rolling(lookback).max()
        roll_low = prices.shift(1).rolling(lookback).min()

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[prices > roll_high] = 1
        signals[prices < roll_low] = -1
        signals[roll_high.isna()] = np.nan
        return signals

    # -----------------------------------------------------------------------
    # volume_signals
    # -----------------------------------------------------------------------

    def volume_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Volume spike detection.
        +1: volume > 2× rolling average AND price up; -1: vol spike AND price down.
        """
        avg_vol = volumes.rolling(lookback).mean()
        vol_ratio = volumes / avg_vol.replace(0, np.nan)
        price_ret = prices.pct_change()

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        spike = vol_ratio > 2.0
        signals[spike & (price_ret > 0)] = 1
        signals[spike & (price_ret < 0)] = -1
        signals[avg_vol.isna()] = np.nan
        return signals


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    # returns
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_vol: float = 0.0
    monthly_returns: Optional[pd.Series] = field(default=None, repr=False)

    # risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    information_ratio: float = 0.0

    # drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # trade stats
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    n_trades: int = 0

    # rolling
    rolling_sharpe: Optional[pd.Series] = field(default=None, repr=False)
    rolling_vol: Optional[pd.Series] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_vol": self.annualized_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "information_ratio": self.information_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "n_trades": self.n_trades,
        }
        return d


def _compute_performance_metrics(
    returns: pd.Series,
    trades_pnl: Optional[np.ndarray] = None,
    benchmark_returns: Optional[pd.Series] = None,
    freq: int = 252,
) -> PerformanceMetrics:
    """Compute full performance metrics via vectorized NumPy operations."""
    rets = returns.dropna().values
    n = len(rets)
    if n == 0:
        return PerformanceMetrics()

    total_return = float(np.prod(1 + rets) - 1)
    years = n / freq
    cagr = float((1 + total_return) ** (1 / max(years, 1e-9)) - 1)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(freq))

    # Sharpe (rf=0)
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(freq)) if ann_vol > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    sortino_denom = float(np.std(downside, ddof=1) * np.sqrt(freq)) if len(downside) > 1 else 1e-9
    sortino = float(np.mean(rets) * freq / sortino_denom) if sortino_denom > 0 else 0.0

    # Drawdown (vectorized)
    equity = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # Max drawdown duration
    in_dd = dd < 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    max_dd_duration = max_dur

    # Calmar
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Information ratio
    if benchmark_returns is not None:
        bret = benchmark_returns.reindex(returns.index).fillna(0).values
        active = rets - bret[: len(rets)]
        ir_vol = np.std(active, ddof=1)
        information_ratio = float(np.mean(active) / ir_vol * np.sqrt(freq)) if ir_vol > 0 else 0.0
    else:
        information_ratio = 0.0

    # Monthly returns
    ret_series = pd.Series(rets, index=returns.dropna().index)
    monthly = (1 + ret_series).resample("ME").prod() - 1

    # Trade stats
    if trades_pnl is not None and len(trades_pnl) > 0:
        wins = trades_pnl[trades_pnl > 0]
        losses = trades_pnl[trades_pnl < 0]
        n_trades = len(trades_pnl)
        win_rate = float(len(wins) / n_trades)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
        best_trade = float(trades_pnl.max())
        worst_trade = float(trades_pnl.min())
    else:
        n_trades = win_rate = avg_win = avg_loss = profit_factor = best_trade = worst_trade = 0

    # Rolling metrics
    roll_window = min(freq, n)
    roll_sr = ret_series.rolling(roll_window).apply(
        lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(freq) if np.std(x, ddof=1) > 0 else 0,
        raw=True,
    )
    roll_vol = ret_series.rolling(roll_window).std(ddof=1) * np.sqrt(freq)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_vol=ann_vol,
        monthly_returns=monthly,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        information_ratio=information_ratio,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=best_trade,
        worst_trade=worst_trade,
        n_trades=n_trades,
        rolling_sharpe=roll_sr,
        rolling_vol=roll_vol,
    )


# ---------------------------------------------------------------------------
# PortfolioResult
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    run_id: str
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: PerformanceMetrics
    strategy_name: str = "unnamed"

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "metrics": self.metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# VectorizedPortfolioSimulator
# ---------------------------------------------------------------------------

class VectorizedPortfolioSimulator:
    """
    Core vectorized backtesting engine.
    All position math done in NumPy — no Python loops over time steps.
    """

    def __init__(self):
        self._signal_engine = SignalEngine()

    def _equal_weight_positions(
        self, signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each row (time step), spread weight equally across non-zero signals.
        Long signals → positive weight, short signals → negative weight.
        """
        long_sig = (signals > 0).astype(float)
        short_sig = (signals < 0).astype(float)

        long_count = long_sig.sum(axis=1).replace(0, np.nan)
        short_count = short_sig.sum(axis=1).replace(0, np.nan)

        long_w = long_sig.div(long_count, axis=0).fillna(0)
        short_w = short_sig.div(short_count, axis=0).fillna(0)

        # scale shorts to 50% if both longs and shorts present
        has_both = (long_count.notna()) & (short_count.notna())
        short_w.loc[has_both] = short_w.loc[has_both] * 0.5
        long_w.loc[has_both] = long_w.loc[has_both] * 0.5

        return long_w - short_w

    def _compute_returns(
        self,
        positions: pd.DataFrame,
        price_returns: pd.DataFrame,
        commission: float,
        slippage: float,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Vectorized P&L computation.
        positions: N_assets × T (weights)
        price_returns: N_assets × T (period returns)
        Returns: (portfolio_returns Series, trade_log DataFrame)
        """
        # positions are decided at end of day t, executed at open t+1
        pos = positions.shift(1).fillna(0)

        # gross return
        gross = (pos * price_returns).sum(axis=1)

        # turnover: sum of absolute weight changes
        turnover = pos.diff().abs().sum(axis=1).fillna(0)

        # transaction cost
        tc = turnover * (commission + slippage)

        net_returns = gross - tc

        # trade log: detect position changes
        pos_change = pos.diff().fillna(pos)
        trades_list = []
        for col in pos_change.columns:
            col_chg = pos_change[col]
            trade_dates = col_chg[col_chg != 0].index
            for dt in trade_dates:
                trades_list.append(
                    {
                        "date": dt,
                        "asset": col,
                        "delta_weight": float(col_chg.loc[dt]),
                        "price": float(
                            price_returns.loc[dt, col]
                            if dt in price_returns.index
                            else np.nan
                        ),
                    }
                )
        trade_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
            columns=["date", "asset", "delta_weight", "price"]
        )

        return net_returns, trade_df

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        strategy_name: str = "strategy",
    ) -> PortfolioResult:
        """
        Main entry point.
        signals: DataFrame same shape as prices, values: +1, -1, 0, NaN
        prices: OHLCV-compatible close price DataFrame
        """
        # align
        prices = prices.reindex(columns=signals.columns).reindex(signals.index)
        signals = signals.reindex(prices.index)

        # compute positions
        positions = self._equal_weight_positions(signals)

        # price returns
        price_returns = prices.pct_change()

        # simulate
        net_returns, trade_df = self._compute_returns(
            positions, price_returns, commission, slippage
        )

        # equity curve
        equity = (1 + net_returns).cumprod() * initial_capital

        # trade PnL (approximate)
        trades_pnl: Optional[np.ndarray] = None
        if not trade_df.empty and "delta_weight" in trade_df.columns:
            pnl_col = trade_df["delta_weight"] * trade_df["price"].fillna(0)
            trades_pnl = pnl_col.values * initial_capital

        metrics = _compute_performance_metrics(
            net_returns, trades_pnl=trades_pnl
        )

        run_id = str(uuid.uuid4())
        result = PortfolioResult(
            run_id=run_id,
            returns=net_returns,
            equity_curve=equity,
            positions=positions,
            trades=trade_df,
            metrics=metrics,
            strategy_name=strategy_name,
        )

        _RESULT_CACHE[run_id] = result
        try:
            _DB.execute(
                "INSERT OR REPLACE INTO backtest_runs VALUES (?,?,?,?)",
                (
                    run_id,
                    datetime.utcnow().isoformat(),
                    strategy_name,
                    json.dumps(metrics.to_dict()),
                ),
            )
            _DB.commit()
        except Exception as exc:
            logger.warning("DB write failed: %s", exc)

        return result


# ---------------------------------------------------------------------------
# MultiStrategyBacktester
# ---------------------------------------------------------------------------

class MultiStrategyBacktester:
    """Run multiple strategies / parameter grids in a vectorized batch."""

    def __init__(self):
        self._sim = VectorizedPortfolioSimulator()
        self._sig = SignalEngine()

    def run_strategy_sweep(
        self,
        param_grid: Dict[str, List[Any]],
        prices: pd.DataFrame,
        signal_func: Callable[..., pd.DataFrame],
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> Dict[str, PortfolioResult]:
        """
        Test all combinations in param_grid.
        signal_func must accept (prices, **params) → signal DataFrame.
        """
        keys = list(param_grid.keys())
        combos = list(product(*[param_grid[k] for k in keys]))
        results = {}
        for combo in combos:
            params = dict(zip(keys, combo))
            label = "_".join(f"{k}={v}" for k, v in params.items())
            try:
                sigs = signal_func(prices, **params)
                res = self._sim.run(
                    sigs,
                    prices,
                    initial_capital=initial_capital,
                    commission=commission,
                    slippage=slippage,
                    strategy_name=label,
                )
                results[label] = res
            except Exception as exc:
                logger.warning("Sweep param %s failed: %s", label, exc)
        return results

    def returns_correlation(
        self, results: Dict[str, PortfolioResult]
    ) -> pd.DataFrame:
        """Correlation matrix of strategy returns."""
        df = pd.DataFrame(
            {k: v.returns for k, v in results.items()}
        )
        return df.corr()

    def ensemble_equal_weight(
        self, results: Dict[str, PortfolioResult]
    ) -> pd.Series:
        df = pd.DataFrame({k: v.returns for k, v in results.items()})
        return df.mean(axis=1)

    def ensemble_sharpe_weighted(
        self, results: Dict[str, PortfolioResult]
    ) -> pd.Series:
        sharpes = np.array([v.metrics.sharpe for v in results.values()])
        sharpes = np.clip(sharpes, 0, None)
        total = sharpes.sum()
        if total == 0:
            return self.ensemble_equal_weight(results)
        weights = sharpes / total
        df = pd.DataFrame({k: v.returns for k, v in results.items()})
        return (df * weights).sum(axis=1)

    def ensemble_min_variance(
        self, results: Dict[str, PortfolioResult]
    ) -> pd.Series:
        df = pd.DataFrame({k: v.returns for k, v in results.items()}).dropna()
        n = df.shape[1]
        cov = df.cov().values
        init_w = np.ones(n) / n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        sol = minimize(
            lambda w: w @ cov @ w,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        weights = sol.x if sol.success else init_w
        return (df * weights).sum(axis=1)


# ---------------------------------------------------------------------------
# FactorBacktester
# ---------------------------------------------------------------------------

class FactorBacktester:
    """Factor-based long/short portfolio backtesting."""

    def __init__(self):
        self._sim = VectorizedPortfolioSimulator()

    def _quintile_signals(
        self, factor: pd.DataFrame, n_quintiles: int = 5
    ) -> pd.DataFrame:
        """
        Long top quintile (+1), short bottom quintile (-1).
        """
        ranks = factor.rank(axis=1, pct=True, na_option="keep")
        threshold_top = (n_quintiles - 1) / n_quintiles
        threshold_bot = 1.0 / n_quintiles
        sigs = pd.DataFrame(0, index=factor.index, columns=factor.columns)
        sigs[ranks > threshold_top] = 1
        sigs[ranks < threshold_bot] = -1
        sigs[ranks.isna()] = np.nan
        return sigs

    def run_factor_backtest(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        n_quintiles: int = 5,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
    ) -> PortfolioResult:
        sigs = self._quintile_signals(factor, n_quintiles)
        return self._sim.run(
            sigs, prices, initial_capital=initial_capital, commission=commission
        )

    def compute_ic(
        self, factor: pd.DataFrame, forward_returns: pd.DataFrame
    ) -> pd.Series:
        """
        IC (Information Coefficient) = Spearman rank correlation
        between factor value and next-period return, computed row-by-row.
        """
        # align
        factor, forward_returns = factor.align(forward_returns)
        ics = []
        dates = []
        for dt in factor.index:
            f_row = factor.loc[dt].dropna()
            r_row = forward_returns.loc[dt].reindex(f_row.index).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 5:
                continue
            corr, _ = stats.spearmanr(f_row[common], r_row[common])
            ics.append(corr)
            dates.append(dt)
        return pd.Series(ics, index=dates, name="IC")

    def compute_icir(self, ic: pd.Series) -> float:
        """IC Information Ratio = mean(IC) / std(IC)."""
        if ic.std() == 0:
            return 0.0
        return float(ic.mean() / ic.std())

    def factor_decay(
        self, factor: pd.DataFrame, prices: pd.DataFrame, max_horizon: int = 20
    ) -> pd.Series:
        """
        Factor IC at different forward horizons (1..max_horizon days).
        Shows how quickly signal decays.
        """
        decay = {}
        for h in range(1, max_horizon + 1):
            fwd_ret = prices.pct_change(h).shift(-h)
            ic = self.compute_ic(factor, fwd_ret)
            decay[h] = ic.mean()
        return pd.Series(decay, name="IC_decay")

    def factor_turnover(self, factor: pd.DataFrame, n_quintiles: int = 5) -> float:
        """Average daily turnover of the factor portfolio."""
        sigs = self._quintile_signals(factor, n_quintiles)
        long_sig = (sigs > 0).astype(float)
        short_sig = (sigs < 0).astype(float)
        long_turn = long_sig.diff().abs().sum(axis=1).mean()
        short_turn = short_sig.diff().abs().sum(axis=1).mean()
        return float((long_turn + short_turn) / 2)


# ---------------------------------------------------------------------------
# BenchmarkComparison
# ---------------------------------------------------------------------------

class BenchmarkComparison:
    """Compare portfolio to standard benchmarks."""

    BENCHMARKS = ["SPY", "QQQ", "IWM"]

    def _fetch_benchmark(self, ticker: str, start: str, end: str) -> pd.Series:
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            closes = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            return closes.pct_change().dropna()
        except Exception as exc:
            logger.warning("Benchmark fetch failed for %s: %s", ticker, exc)
            return pd.Series(dtype=float)

    def compare(
        self,
        portfolio_returns: pd.Series,
        benchmark_ticker: str = "SPY",
    ) -> Dict[str, Any]:
        start = str(portfolio_returns.index.min().date())
        end = str(portfolio_returns.index.max().date())
        bench = self._fetch_benchmark(benchmark_ticker, start, end)

        # align
        aligned = pd.concat(
            [portfolio_returns.rename("portfolio"), bench.rename("bench")], axis=1
        ).dropna()

        if aligned.empty:
            return {"error": "Could not align returns with benchmark"}

        p = aligned["portfolio"].values
        b = aligned["bench"].values

        # active return
        active = p - b
        active_return_ann = float(np.mean(active) * 252)

        # beta via OLS
        beta_val = float(np.cov(p, b)[0, 1] / np.var(b)) if np.var(b) > 0 else 1.0

        # Jensen's alpha
        rf = 0.0  # assume rf=0
        alpha_ann = float((np.mean(p) - rf - beta_val * (np.mean(b) - rf)) * 252)

        # tracking error
        te = float(np.std(active, ddof=1) * np.sqrt(252))

        # information ratio
        ir = float(active_return_ann / te) if te > 0 else 0.0

        # upside/downside capture
        up_mask = b > 0
        down_mask = b < 0
        up_capture = (
            float(np.mean(p[up_mask]) / np.mean(b[up_mask])) if up_mask.any() else 0.0
        )
        down_capture = (
            float(np.mean(p[down_mask]) / np.mean(b[down_mask])) if down_mask.any() else 0.0
        )

        # benchmark metrics
        bench_metrics = _compute_performance_metrics(aligned["bench"])
        port_metrics = _compute_performance_metrics(aligned["portfolio"])

        return {
            "benchmark": benchmark_ticker,
            "period_start": start,
            "period_end": end,
            "portfolio_cagr": port_metrics.cagr,
            "benchmark_cagr": bench_metrics.cagr,
            "portfolio_sharpe": port_metrics.sharpe,
            "benchmark_sharpe": bench_metrics.sharpe,
            "alpha_annualized": alpha_ann,
            "beta": beta_val,
            "active_return_annualized": active_return_ann,
            "tracking_error": te,
            "information_ratio": ir,
            "upside_capture": up_capture,
            "downside_capture": down_capture,
        }

    def compare_all_benchmarks(
        self, portfolio_returns: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        return {
            ticker: self.compare(portfolio_returns, ticker)
            for ticker in self.BENCHMARKS
        }


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

backtest_router = APIRouter(prefix="/backtest", tags=["backtest"])

_simulator = VectorizedPortfolioSimulator()
_signal_engine = SignalEngine()
_multi_tester = MultiStrategyBacktester()
_factor_tester = FactorBacktester()
_bench_cmp = BenchmarkComparison()


class BacktestRunRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols")
    start: str = Field("2020-01-01", description="Start date YYYY-MM-DD")
    end: str = Field("2024-12-31", description="End date YYYY-MM-DD")
    signal_type: str = Field("momentum", description="momentum|mean_reversion|ma_crossover|rsi|breakout")
    initial_capital: float = Field(100_000.0, ge=1_000)
    commission: float = Field(0.001, ge=0, le=0.05)
    slippage: float = Field(0.0005, ge=0, le=0.01)
    lookback: int = Field(252, ge=5, le=504)
    strategy_name: str = Field("backtest")


class FactorBacktestRequest(BaseModel):
    tickers: List[str]
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    factor_type: str = Field("momentum", description="momentum|rsi|mean_reversion")
    initial_capital: float = 100_000.0


class SweepRequest(BaseModel):
    tickers: List[str]
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    signal_type: str = "momentum"
    lookback_values: List[int] = Field([63, 126, 252])
    initial_capital: float = 100_000.0


class CompareRequest(BaseModel):
    run_id: str
    benchmark: str = "SPY"


def _fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
    closes = closes.dropna(how="all")
    return closes


def _make_signals(
    prices: pd.DataFrame, signal_type: str, lookback: int
) -> pd.DataFrame:
    if signal_type == "momentum":
        return _signal_engine.momentum_signals(prices, lookback=lookback)
    elif signal_type == "mean_reversion":
        return _signal_engine.mean_reversion_signals(prices, lookback=min(lookback, 60))
    elif signal_type == "ma_crossover":
        return _signal_engine.ma_crossover_signals(prices, fast=50, slow=min(lookback, 200))
    elif signal_type == "rsi":
        return _signal_engine.rsi_signals(prices)
    elif signal_type == "breakout":
        return _signal_engine.breakout_signals(prices, lookback=lookback)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


@backtest_router.post("/run")
async def run_backtest(req: BacktestRunRequest) -> Dict[str, Any]:
    try:
        prices = _fetch_prices(req.tickers, req.start, req.end)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data returned")
        signals = _make_signals(prices, req.signal_type, req.lookback)
        result = _simulator.run(
            signals,
            prices,
            initial_capital=req.initial_capital,
            commission=req.commission,
            slippage=req.slippage,
            strategy_name=req.strategy_name,
        )
        return result.summary()
    except Exception as exc:
        logger.exception("backtest run error")
        raise HTTPException(status_code=500, detail=str(exc))


@backtest_router.post("/factor")
async def factor_backtest(req: FactorBacktestRequest) -> Dict[str, Any]:
    try:
        prices = _fetch_prices(req.tickers, req.start, req.end)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data")
        factor = _make_signals(prices, req.factor_type, 252)
        result = _factor_tester.run_factor_backtest(
            factor, prices, initial_capital=req.initial_capital
        )
        fwd_ret = prices.pct_change().shift(-1)
        ic = _factor_tester.compute_ic(factor, fwd_ret)
        icir = _factor_tester.compute_icir(ic)
        decay = _factor_tester.factor_decay(factor, prices, max_horizon=10)
        return {
            **result.summary(),
            "ic_mean": float(ic.mean()),
            "icir": icir,
            "ic_decay": decay.to_dict(),
            "factor_turnover": _factor_tester.factor_turnover(factor),
        }
    except Exception as exc:
        logger.exception("factor backtest error")
        raise HTTPException(status_code=500, detail=str(exc))


@backtest_router.post("/sweep")
async def sweep_backtest(req: SweepRequest) -> Dict[str, Any]:
    try:
        prices = _fetch_prices(req.tickers, req.start, req.end)
        if prices.empty:
            raise HTTPException(status_code=400, detail="No price data")
        param_grid = {"lookback": req.lookback_values}

        def _signal_func(p: pd.DataFrame, lookback: int) -> pd.DataFrame:
            return _make_signals(p, req.signal_type, lookback)

        results = _multi_tester.run_strategy_sweep(
            param_grid, prices, _signal_func, req.initial_capital
        )
        corr = _multi_tester.returns_correlation(results)
        ensemble_ew = _multi_tester.ensemble_equal_weight(results)
        ensemble_sw = _multi_tester.ensemble_sharpe_weighted(results)

        ew_metrics = _compute_performance_metrics(ensemble_ew)
        sw_metrics = _compute_performance_metrics(ensemble_sw)

        return {
            "strategies": {k: v.summary() for k, v in results.items()},
            "correlation_matrix": corr.to_dict(),
            "ensemble_equal_weight_sharpe": ew_metrics.sharpe,
            "ensemble_sharpe_weighted_sharpe": sw_metrics.sharpe,
        }
    except Exception as exc:
        logger.exception("sweep backtest error")
        raise HTTPException(status_code=500, detail=str(exc))


@backtest_router.get("/result/{run_id}")
async def get_result(run_id: str) -> Dict[str, Any]:
    if run_id not in _RESULT_CACHE:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return _RESULT_CACHE[run_id].summary()


@backtest_router.post("/compare")
async def compare_to_benchmark(req: CompareRequest) -> Dict[str, Any]:
    if req.run_id not in _RESULT_CACHE:
        raise HTTPException(status_code=404, detail="Run ID not found")
    result = _RESULT_CACHE[req.run_id]
    try:
        comparison = _bench_cmp.compare(result.returns, req.benchmark)
        return comparison
    except Exception as exc:
        logger.exception("benchmark compare error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Convenience: signal sweep utility (used in tests / notebooks)
# ---------------------------------------------------------------------------

def quick_backtest(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    signal_type: str = "momentum",
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> PortfolioResult:
    """Shorthand for a single quick backtest run."""
    prices = _fetch_prices(tickers, start, end)
    if prices.empty:
        raise ValueError("No price data for given tickers/dates")
    signals = _make_signals(prices, signal_type, 252)
    return _simulator.run(
        signals, prices, initial_capital=initial_capital, commission=commission
    )


# ---------------------------------------------------------------------------
# Walk-forward wrapper (uses existing walk_forward_validator patterns)
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """Expanding/rolling window walk-forward backtest."""

    def __init__(
        self,
        train_size: int = 504,
        test_size: int = 63,
        expanding: bool = True,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.expanding = expanding
        self._sim = VectorizedPortfolioSimulator()
        self._sig = SignalEngine()

    def run(
        self,
        prices: pd.DataFrame,
        signal_type: str = "momentum",
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
    ) -> Dict[str, Any]:
        n = len(prices)
        all_oos_returns: List[pd.Series] = []

        start_idx = self.train_size
        while start_idx + self.test_size <= n:
            if self.expanding:
                train_end = start_idx
                train_start = 0
            else:
                train_end = start_idx
                train_start = max(0, train_end - self.train_size)

            test_start = start_idx
            test_end = min(start_idx + self.test_size, n)

            train_prices = prices.iloc[train_start:train_end]
            test_prices = prices.iloc[test_start:test_end]

            # generate signals on test window with full-history prices context
            context_prices = prices.iloc[train_start:test_end]
            sigs = _make_signals(context_prices, signal_type, 252)
            test_sigs = sigs.iloc[train_end - train_start:]

            res = self._sim.run(
                test_sigs, test_prices, initial_capital=initial_capital, commission=commission
            )
            all_oos_returns.append(res.returns)
            start_idx += self.test_size

        if not all_oos_returns:
            return {"error": "Not enough data for walk-forward"}

        combined = pd.concat(all_oos_returns)
        metrics = _compute_performance_metrics(combined)
        return {
            "n_folds": len(all_oos_returns),
            "oos_sharpe": metrics.sharpe,
            "oos_cagr": metrics.cagr,
            "oos_max_drawdown": metrics.max_drawdown,
            "metrics": metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Turnover-constrained optimizer (for position construction)
# ---------------------------------------------------------------------------

class TurnoverConstrainedPortfolio:
    """
    Adds turnover constraint to portfolio rebalancing.
    At each step, limit how far positions can move from prior weights.
    """

    def __init__(self, max_turnover: float = 0.10):
        self.max_turnover = max_turnover

    def constrain_positions(
        self, target_positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Smooth position transitions so daily turnover <= max_turnover.
        Uses vectorized blending: actual = prior + clip(target - prior).
        """
        actual = target_positions.copy()
        for i in range(1, len(actual)):
            prior = actual.iloc[i - 1]
            delta = target_positions.iloc[i] - prior
            total_delta = delta.abs().sum()
            if total_delta > self.max_turnover:
                scale = self.max_turnover / total_delta
                delta = delta * scale
            actual.iloc[i] = prior + delta
        return actual


# ---------------------------------------------------------------------------
# Long-only constraint helper
# ---------------------------------------------------------------------------

def long_only_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Zero out any negative signals (long-only portfolio)."""
    return signals.clip(lower=0)


# ---------------------------------------------------------------------------
# Execution quality analyzer
# ---------------------------------------------------------------------------

class ExecutionQualityAnalyzer:
    """
    Analyze execution quality: implementation shortfall,
    market impact, timing luck.
    """

    def implementation_shortfall(
        self,
        decision_price: pd.Series,
        execution_price: pd.Series,
        direction: pd.Series,  # +1 buy, -1 sell
    ) -> pd.Series:
        """
        IS = (execution_price - decision_price) * direction / decision_price
        Positive IS = slippage cost.
        """
        is_bps = (execution_price - decision_price) * direction / decision_price * 10_000
        return is_bps

    def timing_luck(
        self,
        strategy_returns: pd.Series,
        n_bootstrap: int = 1000,
    ) -> Dict[str, float]:
        """
        Bootstrap timing luck: randomly shift entry by 1-5 days, measure SR distribution.
        """
        shifts = np.random.randint(1, 6, size=n_bootstrap)
        sharpes = []
        rets = strategy_returns.values
        for shift in shifts:
            shifted = np.roll(rets, shift)
            sr = np.mean(shifted) / np.std(shifted, ddof=1) * np.sqrt(252)
            sharpes.append(sr)
        sharpes_arr = np.array(sharpes)
        return {
            "mean_sharpe": float(np.mean(sharpes_arr)),
            "std_sharpe": float(np.std(sharpes_arr)),
            "p5_sharpe": float(np.percentile(sharpes_arr, 5)),
            "p95_sharpe": float(np.percentile(sharpes_arr, 95)),
        }


# ---------------------------------------------------------------------------
# Market regime filter
# ---------------------------------------------------------------------------

class RegimeFilter:
    """Apply regime filter to signals (only trade in bull market regimes)."""

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period

    def filter_signals(
        self,
        signals: pd.DataFrame,
        benchmark_prices: pd.Series,
        long_only_bull: bool = True,
    ) -> pd.DataFrame:
        """
        In bear regime (price below 200-day MA): reduce/zero positions.
        """
        ma = benchmark_prices.rolling(self.ma_period).mean()
        bull = (benchmark_prices > ma).astype(float)
        bull = bull.reindex(signals.index).fillna(0)

        filtered = signals.copy()
        if long_only_bull:
            # zero long signals in bear, keep shorts
            long_mask = filtered > 0
            bear_mask = bull == 0
            filtered[long_mask & bear_mask.values.reshape(-1, 1)] = 0
        return filtered


# ---------------------------------------------------------------------------
# Position sizing: Kelly, risk-parity
# ---------------------------------------------------------------------------

class PositionSizer:
    """Alternative position sizing methods."""

    @staticmethod
    def kelly_fraction(
        win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25
    ) -> float:
        """
        Kelly criterion: f* = (bp - q) / b
        b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        fraction: use fractional Kelly (default 25%) for safety.
        """
        if avg_loss == 0:
            return 0.0
        b = abs(avg_win / avg_loss)
        q = 1 - win_rate
        kelly = (b * win_rate - q) / b
        return max(0.0, min(kelly * fraction, 1.0))

    @staticmethod
    def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk parity: each asset contributes equally to portfolio variance.
        Solved via iterative procedure.
        """
        n = cov_matrix.shape[0]
        w = np.ones(n) / n

        for _ in range(500):
            sigma = np.sqrt(w @ cov_matrix @ w)
            mrc = cov_matrix @ w / sigma  # marginal risk contribution
            rc = w * mrc  # risk contribution
            target_rc = sigma / n
            w = w * target_rc / rc
            w = w / w.sum()

        return w

    @staticmethod
    def vol_target_weights(
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        target_vol: float = 0.15,
        lookback: int = 63,
    ) -> pd.DataFrame:
        """
        Scale each position so it contributes target_vol/N annualized vol.
        """
        rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
        per_asset_target = target_vol / np.sqrt((signals != 0).sum(axis=1).replace(0, np.nan))
        weights = signals * per_asset_target.values.reshape(-1, 1) / rolling_vol.replace(0, np.nan)
        return weights.fillna(0).clip(-2, 2)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SignalEngine",
    "VectorizedPortfolioSimulator",
    "PortfolioResult",
    "PerformanceMetrics",
    "MultiStrategyBacktester",
    "FactorBacktester",
    "BenchmarkComparison",
    "WalkForwardBacktester",
    "TurnoverConstrainedPortfolio",
    "ExecutionQualityAnalyzer",
    "RegimeFilter",
    "PositionSizer",
    "backtest_router",
    "quick_backtest",
    "_compute_performance_metrics",
]
