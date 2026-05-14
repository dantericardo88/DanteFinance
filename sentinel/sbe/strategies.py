"""Built-in strategy library for SBE — returns signal functions for named strategies."""
from __future__ import annotations
from typing import Callable
import pandas as pd
import numpy as np


def get_strategy_fn(name: str, **params) -> Callable[[pd.Series], tuple[pd.Series, pd.Series]]:
    """Return a (prices → entries, exits) function for the named strategy."""
    strategies = {
        "momentum": momentum_strategy,
        "mean_reversion": mean_reversion_strategy,
        "ma_crossover": ma_crossover_strategy,
        "rsi": rsi_strategy,
        "breakout": breakout_strategy,
        "dual_momentum": dual_momentum_strategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies)}")
    fn = strategies[name]
    # Bind params
    def bound_fn(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        return fn(prices, **params)
    return bound_fn


def momentum_strategy(
    prices: pd.Series,
    lookback: int = 252,
    holding: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """12-month momentum (Jegadeesh-Titman style). Enter on positive momentum, exit on negative."""
    import pandas_ta as ta
    mom = prices.pct_change(lookback)
    entries = (mom > 0) & (mom.shift(1) <= 0)
    exits = (mom < 0) & (mom.shift(1) >= 0)
    return entries, exits


def mean_reversion_strategy(
    prices: pd.Series,
    bb_length: int = 20,
    bb_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger Band mean reversion: buy lower band touch, sell upper band touch."""
    rolling_mean = prices.rolling(bb_length).mean()
    rolling_std = prices.rolling(bb_length).std()
    upper = rolling_mean + bb_std * rolling_std
    lower = rolling_mean - bb_std * rolling_std

    entries = prices < lower
    exits = prices > upper
    return entries, exits


def ma_crossover_strategy(
    prices: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> tuple[pd.Series, pd.Series]:
    """Golden/death cross strategy on simple moving averages."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return entries, exits


def rsi_strategy(
    prices: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[pd.Series, pd.Series]:
    """RSI mean reversion: enter on oversold, exit on overbought."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)

    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
    return entries, exits


def breakout_strategy(
    prices: pd.Series,
    channel_length: int = 55,
    exit_length: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel breakout (Turtle Trading). Enter on N-day high, exit on M-day low."""
    high_channel = prices.rolling(channel_length).max()
    exit_channel = prices.rolling(exit_length).min()

    entries = prices >= high_channel.shift(1)
    exits = prices <= exit_channel.shift(1)
    return entries, exits


def dual_momentum_strategy(
    prices: pd.Series,
    lookback: int = 126,
    risk_free_threshold: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Gary Antonacci's Dual Momentum: absolute momentum filter.
    Enter when price > price[lookback] AND return > risk_free_threshold.
    """
    ret = prices.pct_change(lookback)
    in_market = ret > risk_free_threshold

    entries = in_market & ~in_market.shift(1).fillna(False)
    exits = ~in_market & in_market.shift(1).fillna(False)
    return entries, exits
