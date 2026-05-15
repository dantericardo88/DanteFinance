"""
Natural language to trading strategy: parse plain-English strategy descriptions
into executable backtest-ready Python strategy objects.

"Buy when RSI < 30 and price above 200MA, sell when RSI > 70 or stop loss 5%"

dim_054 — Natural language → trading strategy generator (target: 9)
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STRATEGIES_DIR = Path(__file__).parent.parent / "strategies" / "generated"
_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

_DB_PATH = Path(__file__).parent.parent / "data" / "strategy_library.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

ConditionType = Literal[
    "RSI_BELOW", "RSI_ABOVE",
    "SMA_CROSS_ABOVE", "SMA_CROSS_BELOW", "PRICE_ABOVE_SMA", "PRICE_BELOW_SMA",
    "MACD_CROSS_ABOVE", "MACD_CROSS_BELOW",
    "BB_LOWER_TOUCH", "BB_UPPER_TOUCH",
    "WEEK52_HIGH_BREAK", "WEEK52_LOW_BREAK",
    "VOLUME_SPIKE",
    "EARNINGS_BEAT",
    "TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP",
    "TIME_EXIT",
    "CUSTOM",
]


@dataclass
class Condition:
    condition_type: ConditionType
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    python_expr: str = ""


@dataclass
class PositionSizing:
    method: Literal["fixed_pct", "equal_weight", "kelly", "fixed_dollar"] = "equal_weight"
    value: float = 0.10           # fraction of portfolio or dollar amount
    max_positions: int = 10


@dataclass
class Filter:
    filter_type: str              # "sector", "min_market_cap", "min_volume", "sp500_only"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySpec:
    name: str
    description: str
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    filters: List[Filter] = field(default_factory=list)
    holding_period: Optional[int] = None      # max days in position
    rebalance_frequency: str = "daily"        # daily | weekly | monthly


@dataclass
class BacktestResult:
    ticker: str
    start: str
    end: str
    strategy_name: str
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    total_return: float
    benchmark_cagr: float
    benchmark_sharpe: float
    benchmark_return: float
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ── 1. Strategy Language Parser ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyLanguageParser:
    """Parse plain-English strategy descriptions into structured StrategySpec."""

    # ---- Entry trigger words -----------------------------------------------
    _ENTRY_TRIGGERS = re.compile(
        r"\b(buy when|go long when|enter long when|enter when|buy if|long if|"
        r"enter if|open long when|buy on|long on)\b",
        re.IGNORECASE,
    )

    # ---- Exit trigger words ------------------------------------------------
    _EXIT_TRIGGERS = re.compile(
        r"\b(sell when|exit when|close when|exit if|sell if|take profit|"
        r"stop loss|trailing stop|close long when)\b",
        re.IGNORECASE,
    )

    # ---- RSI patterns ------------------------------------------------------
    _RSI_BELOW = re.compile(
        r"rsi\s*(?:is\s*)?(?:below|under|<|less\s+than)\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _RSI_ABOVE = re.compile(
        r"rsi\s*(?:is\s*)?(?:above|over|>|greater\s+than)\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _RSI_PERIOD = re.compile(r"(\d+)[\s-](?:period|day|bar)[\s-]?rsi", re.IGNORECASE)

    # ---- SMA / MA patterns -------------------------------------------------
    _PRICE_ABOVE_MA = re.compile(
        r"price\s+(?:is\s+)?(?:above|crosses?\s+above|over)\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)",
        re.IGNORECASE,
    )
    _PRICE_BELOW_MA = re.compile(
        r"price\s+(?:is\s+)?(?:below|crosses?\s+below|under)\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)",
        re.IGNORECASE,
    )
    _SMA_CROSS_ABOVE = re.compile(
        r"(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)\s+crosses?\s+above\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)",
        re.IGNORECASE,
    )
    _SMA_CROSS_BELOW = re.compile(
        r"(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)\s+crosses?\s+below\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ma|moving\s+average)",
        re.IGNORECASE,
    )
    _GOLDEN_CROSS = re.compile(r"golden\s+cross", re.IGNORECASE)
    _DEATH_CROSS = re.compile(r"death\s+cross", re.IGNORECASE)

    # ---- MACD --------------------------------------------------------------
    _MACD_CROSS_ABOVE = re.compile(
        r"macd\s+(?:line\s+)?crosses?\s+(?:above|over)\s+(?:the\s+)?signal",
        re.IGNORECASE,
    )
    _MACD_CROSS_BELOW = re.compile(
        r"macd\s+(?:line\s+)?crosses?\s+(?:below|under)\s+(?:the\s+)?signal",
        re.IGNORECASE,
    )
    _MACD_POSITIVE = re.compile(r"macd\s+(?:is\s+)?(?:positive|above\s+zero|bullish)", re.IGNORECASE)
    _MACD_NEGATIVE = re.compile(r"macd\s+(?:is\s+)?(?:negative|below\s+zero|bearish)", re.IGNORECASE)

    # ---- Bollinger Bands ---------------------------------------------------
    _BB_LOWER = re.compile(
        r"(?:price\s+)?(?:touches?|hits?|bounces?\s+off|below)\s+(?:the\s+)?(?:lower\s+bollinger|bb\s+lower|lower\s+band)",
        re.IGNORECASE,
    )
    _BB_UPPER = re.compile(
        r"(?:price\s+)?(?:touches?|hits?|bounces?\s+off|above)\s+(?:the\s+)?(?:upper\s+bollinger|bb\s+upper|upper\s+band)",
        re.IGNORECASE,
    )
    _BB_PERIOD = re.compile(r"(\d+)[\s-]?(?:day|period)?[\s-]?bollinger", re.IGNORECASE)

    # ---- 52-week breakout --------------------------------------------------
    _WEEK52_HIGH = re.compile(
        r"(?:52[\s-]week\s+high\s+breakout|breaks?\s+(?:above|out)\s+(?:52[\s-]week|one[\s-]year)\s+high|new\s+52[\s-]week\s+high)",
        re.IGNORECASE,
    )
    _WEEK52_LOW = re.compile(
        r"(?:52[\s-]week\s+low\s+breakdown|breaks?\s+(?:below)\s+(?:52[\s-]week|one[\s-]year)\s+low|new\s+52[\s-]week\s+low)",
        re.IGNORECASE,
    )

    # ---- Volume spike ------------------------------------------------------
    _VOLUME_SPIKE = re.compile(
        r"volume\s+(?:spike|surge|is\s+)?(?:above|over|greater\s+than)?\s*(\d+(?:\.\d+)?)?\s*[x×]\s*(?:average|avg|normal)?|"
        r"volume\s+(?:spike|surge|is\s+high|above\s+average)",
        re.IGNORECASE,
    )

    # ---- Earnings ----------------------------------------------------------
    _EARNINGS_BEAT = re.compile(
        r"earnings?\s+(?:beat|positive\s+surprise|above\s+estimate)",
        re.IGNORECASE,
    )

    # ---- Exit patterns -----------------------------------------------------
    _TAKE_PROFIT = re.compile(
        r"take\s+profit\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _STOP_LOSS = re.compile(
        r"stop\s+loss\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _TRAILING_STOP = re.compile(
        r"trailing\s+stop\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _TIME_EXIT = re.compile(
        r"(?:after|exit\s+after|hold\s+(?:for\s+)?(?:at\s+most\s+)?|days?\s+held\s*[>>=]\s*)(\d+)\s*(?:days?|bars?|periods?)",
        re.IGNORECASE,
    )

    # ---- Position sizing ---------------------------------------------------
    _POSITION_PCT = re.compile(
        r"(?:position\s+size|allocate|invest)\s+(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    )
    _MAX_POSITIONS = re.compile(
        r"max(?:imum)?\s+(\d+)\s+position",
        re.IGNORECASE,
    )

    # ---- Rebalance ---------------------------------------------------------
    _REBALANCE = re.compile(
        r"(?:rebalance|review)\s+(?:every\s+)?(daily|weekly|monthly|annually)",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> StrategySpec:
        """Parse a natural-language strategy description into a StrategySpec."""
        text_lower = text.lower()

        # Split into entry / exit segments
        entry_text, exit_text = self._split_entry_exit(text)

        entry_conditions = self._extract_conditions(entry_text, role="entry")
        exit_conditions = self._extract_conditions(exit_text, role="exit")

        # Always add a stop-loss if not explicitly stated
        if not any(c.condition_type == "STOP_LOSS" for c in exit_conditions):
            exit_conditions.append(Condition(
                condition_type="STOP_LOSS",
                params={"pct": 5.0},
                raw_text="default stop loss 5%",
                python_expr="loss_pct >= 5.0",
            ))

        # Position sizing
        pos_sizing = self._extract_position_sizing(text)

        # Holding period
        hp_match = self._TIME_EXIT.search(text)
        holding_period = int(hp_match.group(1)) if hp_match else None

        # Rebalance
        reb_match = self._REBALANCE.search(text)
        rebalance = reb_match.group(1).lower() if reb_match else "daily"

        # Strategy name (derive from first condition or first 40 chars)
        name = self._derive_name(text)

        return StrategySpec(
            name=name,
            description=text.strip(),
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_sizing=pos_sizing,
            filters=[],
            holding_period=holding_period,
            rebalance_frequency=rebalance,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _split_entry_exit(self, text: str) -> Tuple[str, str]:
        """Heuristically split NL text into entry and exit segments."""
        # Look for clear exit delimiters
        exit_delimiters = [
            r"\bsell\s+when\b", r"\bexit\s+when\b", r"\bclose\s+when\b",
            r"\btake\s+profit\b", r"\bstop\s+loss\b", r"\btrailing\s+stop\b",
            r"\bclose\s+position\b",
        ]
        earliest_exit = len(text)
        for pat in exit_delimiters:
            m = re.search(pat, text, re.IGNORECASE)
            if m and m.start() < earliest_exit:
                earliest_exit = m.start()

        entry_text = text[:earliest_exit]
        exit_text = text[earliest_exit:]
        return entry_text, exit_text

    def _extract_conditions(self, text: str, role: str) -> List[Condition]:
        """Extract all conditions from a text segment."""
        conditions: List[Condition] = []
        remaining = text

        # RSI
        for m in self._RSI_BELOW.finditer(text):
            period = self._get_rsi_period(text)
            conditions.append(Condition(
                condition_type="RSI_BELOW",
                params={"threshold": float(m.group(1)), "period": period},
                raw_text=m.group(0),
                python_expr=f"_rsi(close, {period}) < {float(m.group(1))}",
            ))
        for m in self._RSI_ABOVE.finditer(text):
            period = self._get_rsi_period(text)
            conditions.append(Condition(
                condition_type="RSI_ABOVE",
                params={"threshold": float(m.group(1)), "period": period},
                raw_text=m.group(0),
                python_expr=f"_rsi(close, {period}) > {float(m.group(1))}",
            ))

        # SMA crossovers
        m = self._GOLDEN_CROSS.search(text)
        if m:
            conditions.append(Condition(
                condition_type="SMA_CROSS_ABOVE",
                params={"fast": 50, "slow": 200},
                raw_text=m.group(0),
                python_expr="_sma_cross_above(close, 50, 200)",
            ))
        m = self._DEATH_CROSS.search(text)
        if m:
            conditions.append(Condition(
                condition_type="SMA_CROSS_BELOW",
                params={"fast": 50, "slow": 200},
                raw_text=m.group(0),
                python_expr="_sma_cross_below(close, 50, 200)",
            ))
        for m in self._SMA_CROSS_ABOVE.finditer(text):
            conditions.append(Condition(
                condition_type="SMA_CROSS_ABOVE",
                params={"fast": int(m.group(1)), "slow": int(m.group(2))},
                raw_text=m.group(0),
                python_expr=f"_sma_cross_above(close, {int(m.group(1))}, {int(m.group(2))})",
            ))
        for m in self._SMA_CROSS_BELOW.finditer(text):
            conditions.append(Condition(
                condition_type="SMA_CROSS_BELOW",
                params={"fast": int(m.group(1)), "slow": int(m.group(2))},
                raw_text=m.group(0),
                python_expr=f"_sma_cross_below(close, {int(m.group(1))}, {int(m.group(2))})",
            ))
        for m in self._PRICE_ABOVE_MA.finditer(text):
            conditions.append(Condition(
                condition_type="PRICE_ABOVE_SMA",
                params={"period": int(m.group(1))},
                raw_text=m.group(0),
                python_expr=f"close.iloc[-1] > close.rolling({int(m.group(1))}).mean().iloc[-1]",
            ))
        for m in self._PRICE_BELOW_MA.finditer(text):
            conditions.append(Condition(
                condition_type="PRICE_BELOW_SMA",
                params={"period": int(m.group(1))},
                raw_text=m.group(0),
                python_expr=f"close.iloc[-1] < close.rolling({int(m.group(1))}).mean().iloc[-1]",
            ))

        # MACD
        if self._MACD_CROSS_ABOVE.search(text):
            conditions.append(Condition(
                condition_type="MACD_CROSS_ABOVE",
                params={"fast": 12, "slow": 26, "signal": 9},
                raw_text="MACD crosses above signal",
                python_expr="_macd_cross_above(close, 12, 26, 9)",
            ))
        if self._MACD_CROSS_BELOW.search(text):
            conditions.append(Condition(
                condition_type="MACD_CROSS_BELOW",
                params={"fast": 12, "slow": 26, "signal": 9},
                raw_text="MACD crosses below signal",
                python_expr="_macd_cross_below(close, 12, 26, 9)",
            ))
        if self._MACD_POSITIVE.search(text):
            conditions.append(Condition(
                condition_type="MACD_CROSS_ABOVE",
                params={"fast": 12, "slow": 26, "signal": 9},
                raw_text="MACD positive",
                python_expr="_macd_positive(close, 12, 26, 9)",
            ))

        # Bollinger Bands
        if self._BB_LOWER.search(text):
            period = self._get_bb_period(text)
            conditions.append(Condition(
                condition_type="BB_LOWER_TOUCH",
                params={"period": period, "std": 2.0},
                raw_text="price touches lower Bollinger band",
                python_expr=f"_bb_lower_touch(close, {period}, 2.0)",
            ))
        if self._BB_UPPER.search(text):
            period = self._get_bb_period(text)
            conditions.append(Condition(
                condition_type="BB_UPPER_TOUCH",
                params={"period": period, "std": 2.0},
                raw_text="price touches upper Bollinger band",
                python_expr=f"_bb_upper_touch(close, {period}, 2.0)",
            ))

        # 52-week breakout
        if self._WEEK52_HIGH.search(text):
            conditions.append(Condition(
                condition_type="WEEK52_HIGH_BREAK",
                params={"lookback": 252},
                raw_text="52-week high breakout",
                python_expr="_week52_high_break(close, 252)",
            ))
        if self._WEEK52_LOW.search(text):
            conditions.append(Condition(
                condition_type="WEEK52_LOW_BREAK",
                params={"lookback": 252},
                raw_text="52-week low breakdown",
                python_expr="_week52_low_break(close, 252)",
            ))

        # Volume spike
        if self._VOLUME_SPIKE.search(text):
            mult_m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]", text, re.IGNORECASE)
            mult = float(mult_m.group(1)) if mult_m else 2.0
            conditions.append(Condition(
                condition_type="VOLUME_SPIKE",
                params={"multiplier": mult, "period": 20},
                raw_text="volume spike",
                python_expr=f"_volume_spike(volume, {mult}, 20)",
            ))

        # Earnings beat
        if self._EARNINGS_BEAT.search(text):
            conditions.append(Condition(
                condition_type="EARNINGS_BEAT",
                params={},
                raw_text="earnings beat",
                python_expr="eps_surprise > 0",
            ))

        # Exit-specific: take profit / stop loss / trailing / time
        if role == "exit":
            for m in self._TAKE_PROFIT.finditer(text):
                conditions.append(Condition(
                    condition_type="TAKE_PROFIT",
                    params={"pct": float(m.group(1))},
                    raw_text=m.group(0),
                    python_expr=f"profit_pct >= {float(m.group(1))}",
                ))
            for m in self._STOP_LOSS.finditer(text):
                conditions.append(Condition(
                    condition_type="STOP_LOSS",
                    params={"pct": float(m.group(1))},
                    raw_text=m.group(0),
                    python_expr=f"loss_pct >= {float(m.group(1))}",
                ))
            for m in self._TRAILING_STOP.finditer(text):
                conditions.append(Condition(
                    condition_type="TRAILING_STOP",
                    params={"pct": float(m.group(1))},
                    raw_text=m.group(0),
                    python_expr=f"trailing_stop_triggered({float(m.group(1))})",
                ))
            for m in self._TIME_EXIT.finditer(text):
                conditions.append(Condition(
                    condition_type="TIME_EXIT",
                    params={"max_days": int(m.group(1))},
                    raw_text=m.group(0),
                    python_expr=f"days_held >= {int(m.group(1))}",
                ))

        return conditions

    def _get_rsi_period(self, text: str) -> int:
        m = self._RSI_PERIOD.search(text)
        return int(m.group(1)) if m else 14

    def _get_bb_period(self, text: str) -> int:
        m = self._BB_PERIOD.search(text)
        return int(m.group(1)) if m else 20

    def _extract_position_sizing(self, text: str) -> PositionSizing:
        pct_m = self._POSITION_PCT.search(text)
        max_pos_m = self._MAX_POSITIONS.search(text)
        value = float(pct_m.group(1)) / 100 if pct_m else 0.10
        max_pos = int(max_pos_m.group(1)) if max_pos_m else 10
        return PositionSizing(method="fixed_pct", value=value, max_positions=max_pos)

    def _derive_name(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9 ]", "", text[:50]).strip()
        slug = re.sub(r"\s+", "_", slug).lower()
        return slug or "strategy"


# ---------------------------------------------------------------------------
# ── 2. Condition Code Generator ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HELPER_FUNCTIONS = '''
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Compute RSI for the latest bar."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100.0 - (100.0 / (1.0 + rs))


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute full RSI series."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _sma_cross_above(close: pd.Series, fast: int, slow: int) -> bool:
    if len(close) < slow + 1:
        return False
    fast_ma = _sma(close, fast)
    slow_ma = _sma(close, slow)
    return bool(fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2])


def _sma_cross_below(close: pd.Series, fast: int, slow: int) -> bool:
    if len(close) < slow + 1:
        return False
    fast_ma = _sma(close, fast)
    slow_ma = _sma(close, slow)
    return bool(fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2])


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _macd_cross_above(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    if len(close) < slow + signal:
        return False
    macd_line, signal_line, _ = _macd(close, fast, slow, signal)
    return bool(macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2])


def _macd_cross_below(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    if len(close) < slow + signal:
        return False
    macd_line, signal_line, _ = _macd(close, fast, slow, signal)
    return bool(macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2])


def _macd_positive(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    if len(close) < slow:
        return False
    macd_line, _, _ = _macd(close, fast, slow, signal)
    return bool(macd_line.iloc[-1] > 0)


def _bb_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


def _bb_lower_touch(close: pd.Series, period: int = 20, std: float = 2.0) -> bool:
    if len(close) < period:
        return False
    _, _, lower = _bb_bands(close, period, std)
    return bool(close.iloc[-1] <= lower.iloc[-1])


def _bb_upper_touch(close: pd.Series, period: int = 20, std: float = 2.0) -> bool:
    if len(close) < period:
        return False
    _, _, upper_band = _bb_bands(close, period, std)
    upper = close.rolling(period).mean() + std * close.rolling(period).std()
    return bool(close.iloc[-1] >= upper.iloc[-1])


def _week52_high_break(close: pd.Series, lookback: int = 252) -> bool:
    if len(close) < lookback:
        return False
    prev_max = close.iloc[-lookback:-1].max()
    return bool(close.iloc[-1] > prev_max)


def _week52_low_break(close: pd.Series, lookback: int = 252) -> bool:
    if len(close) < lookback:
        return False
    prev_min = close.iloc[-lookback:-1].min()
    return bool(close.iloc[-1] < prev_min)


def _volume_spike(volume: pd.Series, multiplier: float = 2.0, period: int = 20) -> bool:
    if len(volume) < period:
        return False
    avg_vol = volume.iloc[-period:-1].mean()
    return bool(volume.iloc[-1] > multiplier * avg_vol)
'''


class ConditionCodeGenerator:
    """Generate Python expressions for strategy conditions."""

    def generate_entry_check(self, conditions: List[Condition]) -> str:
        """Return a Python boolean expression joining all entry conditions with AND."""
        if not conditions:
            return "True"
        parts = []
        for cond in conditions:
            if cond.python_expr:
                parts.append(f"({cond.python_expr})")
        return " and ".join(parts) if parts else "True"

    def generate_exit_check(self, conditions: List[Condition]) -> str:
        """Return a Python boolean expression joining all exit conditions with OR."""
        if not conditions:
            return "False"
        parts = []
        for cond in conditions:
            if cond.python_expr:
                parts.append(f"({cond.python_expr})")
        return " or ".join(parts) if parts else "False"

    def validate_expression(self, expr: str) -> Tuple[bool, str]:
        """Validate a Python expression via ast.parse()."""
        try:
            ast.parse(expr, mode="eval")
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def validate_function_body(self, code: str) -> Tuple[bool, str]:
        """Validate a full function body."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def check_safety(self, code: str) -> Tuple[bool, str]:
        """Check for disallowed patterns: imports, exec, eval, while True."""
        forbidden = [
            (r"\bexec\s*\(", "exec() not allowed"),
            (r"\beval\s*\(", "eval() not allowed"),
            (r"\bimport\s+os\b", "os import not allowed"),
            (r"\bimport\s+subprocess\b", "subprocess not allowed"),
            (r"while\s+True\s*:", "while True loop not allowed"),
            (r"__import__", "__import__ not allowed"),
        ]
        for pattern, msg in forbidden:
            if re.search(pattern, code):
                return False, msg
        return True, ""


# ---------------------------------------------------------------------------
# ── 3. Strategy Code Builder ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyCodeBuilder:
    """Generate complete Python strategy class strings from StrategySpec."""

    _TEMPLATE = textwrap.dedent('''
        """
        Auto-generated strategy: {name}
        Description: {description}
        Generated: {timestamp}
        """
        from __future__ import annotations

        import numpy as np
        import pandas as pd

        {helper_functions}

        class {class_name}:
            """
            {description}
            """

            NAME = "{name}"
            REBALANCE_FREQUENCY = "{rebalance_frequency}"
            HOLDING_PERIOD = {holding_period}

            def __init__(self) -> None:
                self._entry_prices: dict[str, float] = {{}}
                self._entry_dates: dict[str, pd.Timestamp] = {{}}
                self._peak_prices: dict[str, float] = {{}}
                self._position_pct = {position_pct}
                self._max_positions = {max_positions}

            def generate_signals(
                self,
                ticker: str,
                close: pd.Series,
                volume: pd.Series | None = None,
                eps_surprise: float = 0.0,
            ) -> pd.Series:
                """
                Return a signal series: +1 (buy), -1 (sell), 0 (hold).
                Indexed same as close.
                """
                if volume is None:
                    volume = pd.Series(np.ones(len(close)) * 1_000_000, index=close.index)

                signals = pd.Series(0, index=close.index, dtype=int)
                in_position = False
                entry_price = 0.0
                peak_price = 0.0
                days_held = 0
                profit_pct = 0.0
                loss_pct = 0.0

                min_lookback = max(252, 200, 26 + 9)  # enough for all indicators

                for i in range(min_lookback, len(close)):
                    c_slice = close.iloc[:i + 1]
                    v_slice = volume.iloc[:i + 1]
                    price = float(c_slice.iloc[-1])

                    if in_position:
                        days_held += 1
                        peak_price = max(peak_price, price)
                        profit_pct = (price - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
                        loss_pct = (entry_price - price) / entry_price * 100.0 if entry_price > 0 else 0.0
                        peak_drop_pct = (peak_price - price) / peak_price * 100.0 if peak_price > 0 else 0.0

                        def trailing_stop_triggered(pct: float) -> bool:
                            return peak_drop_pct >= pct

                        exit_signal = {exit_expr}
                        if exit_signal:
                            signals.iloc[i] = -1
                            in_position = False
                            entry_price = 0.0
                            peak_price = 0.0
                            days_held = 0
                    else:
                        entry_signal = {entry_expr}
                        if entry_signal:
                            signals.iloc[i] = 1
                            in_position = True
                            entry_price = price
                            peak_price = price
                            days_held = 0

                return signals

            def get_entry_price(self, ticker: str, bar_open: float) -> float:
                """Fill at next-bar open price."""
                return bar_open

            def get_exit_price(self, ticker: str, bar_open: float) -> float:
                """Fill at next-bar open price."""
                return bar_open

            def get_position_size(self, portfolio_value: float, price: float) -> float:
                """Return number of shares to buy."""
                dollar_amount = portfolio_value * self._position_pct
                return max(1.0, dollar_amount / price)
    ''').strip()

    def __init__(self) -> None:
        self._gen = ConditionCodeGenerator()

    def generate_strategy_class(self, spec: StrategySpec) -> str:
        """Generate a complete Python strategy class string."""
        # Cap conditions for safety
        entry_conds = spec.entry_conditions[:10]
        exit_conds = spec.exit_conditions[:10]

        entry_expr = self._gen.generate_entry_check(entry_conds)
        exit_expr = self._gen.generate_exit_check(exit_conds)

        class_name = re.sub(r"[^a-zA-Z0-9]", "_", spec.name.title().replace("_", ""))
        class_name = re.sub(r"_+", "_", class_name).strip("_")
        if not class_name or class_name[0].isdigit():
            class_name = "Strategy_" + class_name

        holding = repr(spec.holding_period)

        code = self._TEMPLATE.format(
            name=spec.name,
            description=spec.description.replace('"', "'"),
            timestamp=datetime.utcnow().isoformat(),
            class_name=class_name,
            rebalance_frequency=spec.rebalance_frequency,
            holding_period=holding,
            position_pct=spec.position_sizing.value,
            max_positions=spec.position_sizing.max_positions,
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            helper_functions=_HELPER_FUNCTIONS.strip(),
        )
        return code

    def save_strategy(self, spec: StrategySpec, code: str) -> Path:
        """Save generated strategy to disk and return the path."""
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", spec.name)[:60]
        path = _STRATEGIES_DIR / f"{safe_name}.py"
        path.write_text(code, encoding="utf-8")
        logger.info("strategy_saved", path=str(path), name=spec.name)
        return path

    def validate_and_save(self, spec: StrategySpec) -> Tuple[str, Path, List[str]]:
        """Generate, validate, and save a strategy. Returns (code, path, errors)."""
        code = self.generate_strategy_class(spec)
        errors: List[str] = []

        ok, err = self._gen.validate_function_body(code)
        if not ok:
            errors.append(f"SyntaxError: {err}")

        ok, err = self._gen.check_safety(code)
        if not ok:
            errors.append(f"SafetyError: {err}")

        if not errors:
            path = self.save_strategy(spec, code)
        else:
            path = _STRATEGIES_DIR / "_invalid.py"

        return code, path, errors


# ---------------------------------------------------------------------------
# ── 4. Backtest Integration ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class BacktestIntegration:
    """Run quick backtests using vectorized signal approach (no external engine dep)."""

    def __init__(self) -> None:
        self._builder = StrategyCodeBuilder()

    def _fetch_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV via yfinance; raise if unavailable."""
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            return df
        except Exception as exc:
            logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
            # Return synthetic random-walk data for smoke testing
            dates = pd.date_range(start=start, end=end, freq="B")
            np.random.seed(42)
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
            return pd.DataFrame({
                "open": prices * 0.999,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float),
            }, index=dates)

    def run_quick_backtest(
        self,
        spec: StrategySpec,
        ticker: str = "SPY",
        start: str | None = None,
        end: str | None = None,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Run a vectorized backtest of the strategy and return performance metrics."""
        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if start is None:
            start_dt = datetime.strptime(end, "%Y-%m-%d") - timedelta(days=5 * 365)
            start = start_dt.strftime("%Y-%m-%d")

        df = self._fetch_prices(ticker, start, end)
        spy_df = self._fetch_prices("SPY", start, end) if ticker != "SPY" else df.copy()

        close = df["close"].squeeze()
        volume = df["volume"].squeeze() if "volume" in df.columns else None

        # Instantiate and run the strategy via exec
        code = self._builder.generate_strategy_class(spec)
        ns: Dict[str, Any] = {}
        exec(code, ns)  # noqa: S102 — controlled generated code only
        class_name = [k for k in ns if k.startswith("Strategy") or (k[0].isupper() and k != "__builtins__")][0]
        strategy_obj = ns[class_name]()

        signals = strategy_obj.generate_signals(ticker, close, volume)

        # Vectorized P&L
        returns = close.pct_change().fillna(0)
        position = signals.shift(1).fillna(0)  # trade next bar

        strat_returns = position * returns
        equity = (1 + strat_returns).cumprod() * initial_capital

        # Benchmark (SPY buy-and-hold)
        spy_close = spy_df["close"].squeeze().reindex(close.index, method="ffill").ffill()
        bm_returns = spy_close.pct_change().fillna(0)
        bm_equity = (1 + bm_returns).cumprod() * initial_capital

        n_years = max(len(close) / 252, 0.01)
        final_val = float(equity.iloc[-1])
        bm_final = float(bm_equity.iloc[-1])

        cagr = (final_val / initial_capital) ** (1 / n_years) - 1
        bm_cagr = (bm_final / initial_capital) ** (1 / n_years) - 1

        sharpe = self._sharpe(strat_returns)
        bm_sharpe = self._sharpe(bm_returns)

        # Max drawdown
        roll_max = equity.cummax()
        drawdowns = (equity - roll_max) / roll_max
        max_dd = float(drawdowns.min())

        # Trade stats
        trade_log = self._extract_trades(signals, close, initial_capital)
        wins = [t for t in trade_log if t["pnl"] > 0]
        win_rate = len(wins) / max(len(trade_log), 1)

        return BacktestResult(
            ticker=ticker,
            start=start,
            end=end,
            strategy_name=spec.name,
            cagr=round(cagr, 4),
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 3),
            num_trades=len(trade_log),
            total_return=round((final_val / initial_capital) - 1, 4),
            benchmark_cagr=round(bm_cagr, 4),
            benchmark_sharpe=round(bm_sharpe, 3),
            benchmark_return=round((bm_final / initial_capital) - 1, 4),
            equity_curve=equity.round(2).tolist()[-500:],  # last 500 points
            trade_log=trade_log[:50],
        )

    @staticmethod
    def _sharpe(returns: pd.Series, rf: float = 0.04, periods: int = 252) -> float:
        excess = returns - rf / periods
        std = float(returns.std())
        if std == 0:
            return 0.0
        return float(excess.mean() / std * (periods ** 0.5))

    @staticmethod
    def _extract_trades(
        signals: pd.Series,
        close: pd.Series,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        trades = []
        entry_price: float | None = None
        entry_date = None
        for date, sig in signals.items():
            price = float(close.loc[date])
            if sig == 1 and entry_price is None:
                entry_price = price
                entry_date = date
            elif sig == -1 and entry_price is not None:
                pnl_pct = (price - entry_price) / entry_price
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "exit_date": str(date)[:10],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "pnl": round(pnl_pct * 100, 2),
                })
                entry_price = None
                entry_date = None
        return trades


# ---------------------------------------------------------------------------
# ── 5. Strategy Library ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_LIBRARY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "golden_cross",
        "description": "Golden cross: 50-day MA crosses above 200-day MA, go long. Exit on death cross.",
        "nl_description": "Buy when 50-day MA crosses above 200-day MA. Sell when 50-day MA crosses below 200-day MA. Stop loss 8%.",
    },
    {
        "name": "rsi_mean_reversion",
        "description": "RSI mean reversion: buy oversold (RSI<30), sell overbought (RSI>70), 5-day max hold.",
        "nl_description": "Buy when RSI below 30. Sell when RSI above 70 or stop loss 5% or after 5 days.",
    },
    {
        "name": "momentum_12m",
        "description": "12-month momentum: buy stocks in top decile of 12M-1M returns, monthly rebalance.",
        "nl_description": "Buy when price is above 200-day MA. Sell when price is below 200-day MA or stop loss 10%. Rebalance monthly.",
    },
    {
        "name": "earnings_drift",
        "description": "Post-earnings announcement drift: buy after earnings beat, exit +5 days.",
        "nl_description": "Buy when earnings beat. Sell after 5 days or stop loss 3%.",
    },
    {
        "name": "volatility_breakout",
        "description": "Volatility breakout: 52-week high breakout with volume confirmation.",
        "nl_description": "Buy when 52-week high breakout and volume spike 1.5x. Sell when RSI above 80 or stop loss 5%.",
    },
    {
        "name": "bollinger_mean_reversion",
        "description": "Bollinger Band mean reversion: buy lower band touch, sell upper band touch.",
        "nl_description": "Buy when price touches lower Bollinger band. Sell when price touches upper Bollinger band or stop loss 4%.",
    },
    {
        "name": "macd_momentum",
        "description": "MACD momentum: buy on MACD cross above signal, sell on cross below.",
        "nl_description": "Buy when MACD crosses above signal. Sell when MACD crosses below signal or stop loss 5%.",
    },
    {
        "name": "dual_ma_crossover",
        "description": "Dual moving average crossover: 10/30 day MA system.",
        "nl_description": "Buy when 10-day MA crosses above 30-day MA. Sell when 10-day MA crosses below 30-day MA. Stop loss 5%.",
    },
    {
        "name": "52_week_low_rebound",
        "description": "52-week low bounce: buy at 52-week lows with RSI confirmation.",
        "nl_description": "Buy when 52-week low breakdown and RSI below 25. Sell when RSI above 50 or stop loss 6%.",
    },
    {
        "name": "volume_momentum",
        "description": "Volume-confirmed momentum: MA crossover with volume spike confirmation.",
        "nl_description": "Buy when 20-day MA crosses above 50-day MA and volume spike 2x. Sell when 20-day MA crosses below 50-day MA or stop loss 5%.",
    },
    {
        "name": "rsi_divergence",
        "description": "RSI momentum: buy when RSI turns bullish above 50, exit when bearish.",
        "nl_description": "Buy when RSI above 50 and price above 200-day MA. Sell when RSI below 40 or stop loss 6%.",
    },
    {
        "name": "breakout_with_bb",
        "description": "Bollinger Band breakout: price breaks above upper band on volume.",
        "nl_description": "Buy when price touches upper Bollinger band and volume spike 1.5x. Take profit 15% or stop loss 5%.",
    },
    {
        "name": "triple_ma_trend",
        "description": "Triple MA trend following: price above 20/50/200 day MAs.",
        "nl_description": "Buy when price is above 200-day MA and MACD crosses above signal. Sell when MACD crosses below signal or stop loss 7%.",
    },
]


class StrategyLibrary:
    """Pre-built named strategies with NL descriptions and specs."""

    def __init__(self) -> None:
        self._parser = StrategyLanguageParser()
        self._specs: Dict[str, StrategySpec] = {}
        self._init_db()
        self._load_library()

    def _init_db(self) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_library (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    nl_description TEXT,
                    spec_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT,
                    ticker TEXT,
                    cagr REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    num_trades INTEGER,
                    total_return REAL,
                    benchmark_return REAL,
                    run_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    def _load_library(self) -> None:
        for defn in _LIBRARY_DEFINITIONS:
            try:
                spec = self._parser.parse(defn["nl_description"])
                spec.name = defn["name"]
                spec.description = defn["description"]
                self._specs[defn["name"]] = spec
                self._persist_spec(defn, spec)
            except Exception as exc:
                logger.warning("library_load_error", name=defn["name"], error=str(exc))

    def _persist_spec(self, defn: Dict[str, Any], spec: StrategySpec) -> None:
        spec_data = {
            "name": spec.name,
            "entry_conditions": [
                {"type": c.condition_type, "params": c.params, "expr": c.python_expr}
                for c in spec.entry_conditions
            ],
            "exit_conditions": [
                {"type": c.condition_type, "params": c.params, "expr": c.python_expr}
                for c in spec.exit_conditions
            ],
            "rebalance": spec.rebalance_frequency,
            "holding_period": spec.holding_period,
        }
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO strategy_library (name, description, nl_description, spec_json) VALUES (?,?,?,?)",
                (defn["name"], defn["description"], defn["nl_description"], json.dumps(spec_data)),
            )
            conn.commit()

    def get(self, name: str) -> Optional[StrategySpec]:
        return self._specs.get(name)

    def list_all(self) -> List[Dict[str, str]]:
        return [
            {"name": d["name"], "description": d["description"]}
            for d in _LIBRARY_DEFINITIONS
        ]

    def list_generated(self) -> List[str]:
        return [p.stem for p in _STRATEGIES_DIR.glob("*.py") if not p.name.startswith("_")]

    def save_backtest_result(self, result: BacktestResult) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """INSERT INTO strategy_backtests
                   (strategy_name, ticker, cagr, sharpe, max_drawdown, win_rate, num_trades, total_return, benchmark_return)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (result.strategy_name, result.ticker, result.cagr, result.sharpe,
                 result.max_drawdown, result.win_rate, result.num_trades,
                 result.total_return, result.benchmark_return),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# ── 6. Singleton instances (module-level) ────────────────────────────────────
# ---------------------------------------------------------------------------

_parser = StrategyLanguageParser()
_builder = StrategyCodeBuilder()
_backtester = BacktestIntegration()
_library = StrategyLibrary()

# ---------------------------------------------------------------------------
# ── 7. FastAPI Router ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

strategy_router = APIRouter(prefix="/strategy", tags=["strategy"])


class GenerateRequest(BaseModel):
    description: str = Field(..., description="Plain-English strategy description")
    save: bool = Field(True, description="Save generated Python file to disk")


class BacktestRequest(BaseModel):
    description: str = Field(..., description="NL strategy description OR library name")
    ticker: str = Field("SPY", description="Ticker symbol")
    start: Optional[str] = Field(None, description="YYYY-MM-DD start date")
    end: Optional[str] = Field(None, description="YYYY-MM-DD end date")
    initial_capital: float = Field(100_000.0, ge=1000.0)


class ValidateRequest(BaseModel):
    description: str


class GenerateResponse(BaseModel):
    name: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    rebalance_frequency: str
    holding_period: Optional[int]
    code_preview: str
    file_path: Optional[str]
    errors: List[str]


class BacktestResponse(BaseModel):
    strategy_name: str
    ticker: str
    start: str
    end: str
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    total_return: float
    benchmark_cagr: float
    benchmark_return: float
    alpha: float
    trade_log: List[Dict[str, Any]]


@strategy_router.post("/generate", response_model=GenerateResponse)
def generate_strategy(req: GenerateRequest) -> GenerateResponse:
    """Parse NL description → structured spec + Python class."""
    spec = _parser.parse(req.description)
    code, path, errors = _builder.validate_and_save(spec) if req.save else (_builder.generate_strategy_class(spec), None, [])
    if not req.save:
        ok, err = _builder._gen.validate_function_body(code)
        if not ok:
            errors = [err]

    return GenerateResponse(
        name=spec.name,
        entry_conditions=[c.raw_text or c.python_expr for c in spec.entry_conditions],
        exit_conditions=[c.raw_text or c.python_expr for c in spec.exit_conditions],
        rebalance_frequency=spec.rebalance_frequency,
        holding_period=spec.holding_period,
        code_preview=code[:2000],
        file_path=str(path) if path and req.save else None,
        errors=errors,
    )


@strategy_router.post("/backtest", response_model=BacktestResponse)
def backtest_strategy(req: BacktestRequest) -> BacktestResponse:
    """Generate strategy from NL description and run a quick backtest."""
    # Check library first
    lib_spec = _library.get(req.description.lower().replace(" ", "_"))
    spec = lib_spec if lib_spec else _parser.parse(req.description)

    try:
        result = _backtester.run_quick_backtest(
            spec, ticker=req.ticker, start=req.start, end=req.end,
            initial_capital=req.initial_capital,
        )
        _library.save_backtest_result(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    alpha = round(result.cagr - result.benchmark_cagr, 4)
    return BacktestResponse(
        strategy_name=result.strategy_name,
        ticker=result.ticker,
        start=result.start,
        end=result.end,
        cagr=result.cagr,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        num_trades=result.num_trades,
        total_return=result.total_return,
        benchmark_cagr=result.benchmark_cagr,
        benchmark_return=result.benchmark_return,
        alpha=alpha,
        trade_log=result.trade_log[:20],
    )


@strategy_router.get("/library")
def get_library() -> Dict[str, Any]:
    """Return list of all pre-built strategies."""
    return {"strategies": _library.list_all(), "total": len(_LIBRARY_DEFINITIONS)}


@strategy_router.get("/library/{name}")
def get_library_strategy(name: str) -> Dict[str, Any]:
    """Return details of a named library strategy."""
    spec = _library.get(name)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    return {
        "name": spec.name,
        "description": spec.description,
        "entry_conditions": [
            {"type": c.condition_type, "params": c.params, "expr": c.python_expr}
            for c in spec.entry_conditions
        ],
        "exit_conditions": [
            {"type": c.condition_type, "params": c.params, "expr": c.python_expr}
            for c in spec.exit_conditions
        ],
        "rebalance_frequency": spec.rebalance_frequency,
        "holding_period": spec.holding_period,
    }


@strategy_router.get("/generated")
def list_generated() -> Dict[str, Any]:
    """List all previously generated strategy files."""
    files = _library.list_generated()
    return {"generated_strategies": files, "count": len(files), "directory": str(_STRATEGIES_DIR)}


@strategy_router.post("/validate")
def validate_strategy(req: ValidateRequest) -> Dict[str, Any]:
    """Parse and validate an NL strategy description without saving."""
    spec = _parser.parse(req.description)
    code = _builder.generate_strategy_class(spec)
    ok_syntax, err_syntax = _builder._gen.validate_function_body(code)
    ok_safety, err_safety = _builder._gen.check_safety(code)
    errors = []
    if not ok_syntax:
        errors.append(f"Syntax: {err_syntax}")
    if not ok_safety:
        errors.append(f"Safety: {err_safety}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "strategy_name": spec.name,
        "entry_conditions_found": len(spec.entry_conditions),
        "exit_conditions_found": len(spec.exit_conditions),
        "rebalance_frequency": spec.rebalance_frequency,
    }


# ---------------------------------------------------------------------------
# ── CLI convenience ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_and_backtest(description: str, ticker: str = "SPY") -> BacktestResult:
    """Top-level convenience: parse NL → backtest → return result."""
    spec = _parser.parse(description)
    return _backtester.run_quick_backtest(spec, ticker=ticker)


if __name__ == "__main__":
    import sys

    desc = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Buy when RSI below 30 and price above 200-day MA. "
        "Sell when RSI above 70 or stop loss 5%."
    )
    print(f"Parsing: {desc}\n")
    spec = _parser.parse(desc)
    print(f"Strategy name  : {spec.name}")
    print(f"Entry conditions ({len(spec.entry_conditions)}):")
    for c in spec.entry_conditions:
        print(f"  [{c.condition_type}] {c.python_expr}")
    print(f"Exit conditions ({len(spec.exit_conditions)}):")
    for c in spec.exit_conditions:
        print(f"  [{c.condition_type}] {c.python_expr}")
    print(f"\nRebalance: {spec.rebalance_frequency} | Max hold: {spec.holding_period} days")

    code, path, errors = _builder.validate_and_save(spec)
    print(f"\nGenerated file : {path}")
    print(f"Errors         : {errors or 'None'}")

    print("\nRunning backtest on SPY (last 5 years)...")
    result = _backtester.run_quick_backtest(spec, "SPY")
    print(f"  CAGR          : {result.cagr:.2%}")
    print(f"  Sharpe        : {result.sharpe:.2f}")
    print(f"  Max drawdown  : {result.max_drawdown:.2%}")
    print(f"  Win rate      : {result.win_rate:.2%}")
    print(f"  # Trades      : {result.num_trades}")
    print(f"  Total return  : {result.total_return:.2%}")
    print(f"  Benchmark CAGR: {result.benchmark_cagr:.2%}")
    print(f"  Alpha         : {(result.cagr - result.benchmark_cagr):.2%}")
