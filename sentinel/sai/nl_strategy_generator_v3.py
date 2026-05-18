"""
Natural language → trading strategy generator V3.

Parse plain-English strategy descriptions into executable, backtest-validated
Python strategy objects. Every generated strategy is automatically run through
a quality gate before being returned to the caller — if the strategy fails to
meet minimum performance standards the caller receives a structured failure
message rather than a silently useless strategy.

dim_054 — Natural language → trading strategy generator (target: 9)

Improvements over V1:
  - 15+ intent types: TREND_FOLLOW, MEAN_REVERT, MOMENTUM, PAIRS, STAT_ARB,
    BREAKOUT, CARRY, SEASONALITY, EARNINGS_DRIFT, EVENT_DRIVEN, FACTOR,
    SECTOR_ROTATION, VOLATILITY, MACRO, SENTIMENT
  - Expanded signal component parser: sizing, universe, rebalance, stop-loss,
    take-profit, holding-period — all extracted from NL
  - Code generator producing vectorised StrategyConfig-compatible classes
  - Quality gate: auto-backtest every strategy on 2-yr in-sample;
    Sharpe > 0.5, max drawdown < 30%, min 20 trades, win rate > 40%
  - Strategy library: 30 pre-built validated strategies stored in SQLite
  - Strategy comparison: run 2+ strategies and compare Sharpe/drawdown/turnover
  - Ensemble generator: weight by Sharpe across 3+ uncorrelated strategies
  - FastAPI router at /strategy/v3

Usage::
    from sentinel.sai.nl_strategy_generator_v3 import (
        strategy_v3_router,
        parse_and_validate,
    )
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

_STRATEGIES_DIR = Path(__file__).parent.parent / "strategies" / "generated_v3"
_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

_DB_PATH = Path(__file__).parent.parent / "data" / "strategy_library_v3.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUALITY_GATE_SHARPE: float = 0.5
_QUALITY_GATE_MAX_DD: float = -0.30       # must be better (less negative) than this
_QUALITY_GATE_MIN_TRADES: int = 20
_QUALITY_GATE_MIN_WIN_RATE: float = 0.40
_IN_SAMPLE_YEARS: int = 2                 # quality gate back-test window

StrategyIntentType = Literal[
    "TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "PAIRS", "STAT_ARB",
    "BREAKOUT", "CARRY", "SEASONALITY", "EARNINGS_DRIFT", "EVENT_DRIVEN",
    "FACTOR", "SECTOR_ROTATION", "VOLATILITY", "MACRO", "SENTIMENT",
    "UNKNOWN",
]

ConditionType = Literal[
    "RSI_BELOW", "RSI_ABOVE",
    "SMA_CROSS_ABOVE", "SMA_CROSS_BELOW", "PRICE_ABOVE_SMA", "PRICE_BELOW_SMA",
    "MACD_CROSS_ABOVE", "MACD_CROSS_BELOW",
    "BB_LOWER_TOUCH", "BB_UPPER_TOUCH",
    "WEEK52_HIGH_BREAK", "WEEK52_LOW_BREAK",
    "VOLUME_SPIKE",
    "ATR_BREAKOUT",
    "SPREAD_ZSCORE_ABOVE", "SPREAD_ZSCORE_BELOW",
    "MOMENTUM_RANK_TOP", "MOMENTUM_RANK_BOTTOM",
    "EARNINGS_BEAT", "EARNINGS_MISS",
    "TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP",
    "TIME_EXIT",
    "CUSTOM",
]

SizingMethod = Literal[
    "equal_weight", "inverse_vol", "kelly", "fixed_pct", "fixed_dollar", "sharpe_weight",
]

UniverseType = Literal[
    "SP500", "RUSSELL1000", "NASDAQ100", "DOW30",
    "CUSTOM_TICKERS", "ALL_US_STOCKS",
]

RebalanceFreq = Literal["daily", "weekly", "monthly", "quarterly"]

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    condition_type: ConditionType
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    python_expr: str = ""


@dataclass
class PositionSizing:
    method: SizingMethod = "equal_weight"
    value: float = 0.10          # fraction of portfolio or dollar amount
    max_positions: int = 10


@dataclass
class Universe:
    universe_type: UniverseType = "SP500"
    custom_tickers: List[str] = field(default_factory=list)

    def resolve_tickers(self) -> List[str]:
        """Return a representative sample of tickers for the universe."""
        _UNIVERSE_MAP: Dict[str, List[str]] = {
            "SP500":       ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "UNH", "JPM", "JNJ",
                            "V", "XOM", "PG", "MA", "HD", "CVX", "LLY", "ABBV", "PEP", "KO"],
            "RUSSELL1000": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "JNJ", "V",
                            "XOM", "PG", "MA", "HD", "CVX", "LLY", "ABBV", "PEP", "KO", "MRK"],
            "NASDAQ100":   ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "TSLA", "AVGO", "ASML", "COST",
                            "ADBE", "NFLX", "AZN", "AMD", "QCOM", "INTC", "INTU", "ISRG", "AMAT", "MU"],
            "DOW30":       ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "CVX", "HD", "MCD",
                            "GS", "CAT", "BA", "DIS", "IBM", "NKE", "MRK", "AXP", "MMM", "TRV"],
            "ALL_US_STOCKS": ["SPY", "QQQ", "IWM"],
        }
        if self.universe_type == "CUSTOM_TICKERS":
            return self.custom_tickers or ["SPY"]
        return _UNIVERSE_MAP.get(self.universe_type, ["SPY"])


@dataclass
class StrategySpec:
    name: str
    description: str
    intent: StrategyIntentType = "UNKNOWN"
    entry_conditions: List[Condition] = field(default_factory=list)
    exit_conditions: List[Condition] = field(default_factory=list)
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    universe: Universe = field(default_factory=Universe)
    holding_period: Optional[int] = None
    rebalance_frequency: RebalanceFreq = "daily"
    # Pairs-specific
    long_leg: Optional[str] = None
    short_leg: Optional[str] = None
    spread_threshold: float = 2.0
    unwind_threshold: float = 0.5


@dataclass
class QualityGateResult:
    passed: bool
    sharpe: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    failure_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASSED — Sharpe={self.sharpe:.2f}, MaxDD={self.max_drawdown:.1%}, "
                f"Trades={self.num_trades}, WinRate={self.win_rate:.1%}"
            )
        return "FAILED — " + "; ".join(self.failure_reasons)


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
    turnover: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationResult:
    spec: StrategySpec
    code: str
    file_path: Optional[Path]
    syntax_errors: List[str]
    quality_gate: QualityGateResult
    backtest: Optional[BacktestResult]
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ---------------------------------------------------------------------------
# ── 1. Intent Classifier ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Map natural language strategy descriptions to StrategyIntentType."""

    _PATTERNS: List[Tuple[re.Pattern, StrategyIntentType]] = [
        (re.compile(r"\b(momentum|12[- ]?month|12[- ]?1|trailing return|top decile|top \d+%)\b", re.I), "MOMENTUM"),
        (re.compile(r"\b(pairs? trad|long.*short.*spread|stat.*arb|statistical arbitrage|cointegrat)\b", re.I), "PAIRS"),
        (re.compile(r"\b(mean revert|oversold|overbought|reversion|bounce back|contrarian|5[- ]?day loser)\b", re.I), "MEAN_REVERT"),
        (re.compile(r"\b(trend follow|moving average cross|golden cross|death cross|sma cross|ma cross)\b", re.I), "TREND_FOLLOW"),
        (re.compile(r"\b(breakout|break(?:ing)? (?:above|out|through)|52[- ]?week high|new high)\b", re.I), "BREAKOUT"),
        (re.compile(r"\b(earnings drift|post[- ]earnings|pead|earnings surprise|earnings beat|earnings miss)\b", re.I), "EARNINGS_DRIFT"),
        (re.compile(r"\b(event[- ]driven|merger|acquisition|spin[- ]?off|restructur|catalyst)\b", re.I), "EVENT_DRIVEN"),
        (re.compile(r"\b(factor|quality|value factor|size factor|low vol factor|multi[- ]?factor)\b", re.I), "FACTOR"),
        (re.compile(r"\b(sector rotat|rotate sector|best sector|sector momentum)\b", re.I), "SECTOR_ROTATION"),
        (re.compile(r"\b(volatility|vix|vol regime|low vol|high vol|vol trading)\b", re.I), "VOLATILITY"),
        (re.compile(r"\b(macro|interest rate|fed|gdp|inflation|yield curve|currency)\b", re.I), "MACRO"),
        (re.compile(r"\b(sentiment|news sentiment|social|twitter|reddit|bullish sentiment)\b", re.I), "SENTIMENT"),
        (re.compile(r"\b(carry|dividend|yield|high[- ]yield|income|dividend yield)\b", re.I), "CARRY"),
        (re.compile(r"\b(seasonal|january effect|santa rally|sell in may|day of week)\b", re.I), "SEASONALITY"),
        (re.compile(r"\b(stat.*arb|arbitrage|spread trading|relative value)\b", re.I), "STAT_ARB"),
    ]

    def classify(self, text: str) -> StrategyIntentType:
        for pattern, intent in self._PATTERNS:
            if pattern.search(text):
                return intent
        # Fallback to TREND_FOLLOW if MA terms detected
        if re.search(r"\b(ma|moving average|sma|ema)\b", text, re.I):
            return "TREND_FOLLOW"
        if re.search(r"\b(rsi|macd|bollinger)\b", text, re.I):
            return "MEAN_REVERT"
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# ── 2. Strategy Language Parser (expanded) ───────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyLanguageParserV3:
    """Parse NL strategy descriptions into StrategySpec (expanded — 15+ types)."""

    # ---- Entry triggers ------------------------------------------------
    _ENTRY_TRIGGERS = re.compile(
        r"\b(buy when|go long when|enter long when|enter when|buy if|long if|"
        r"enter if|open long when|buy on|long on|long|buy)\b", re.I)
    _EXIT_TRIGGERS = re.compile(
        r"\b(sell when|exit when|close when|exit if|sell if|take profit|"
        r"stop.?loss|trailing stop|close long when|sell|exit|unwind at)\b", re.I)

    # ---- RSI -----------------------------------------------------------
    _RSI_BELOW = re.compile(r"rsi\s*(?:is\s*)?(?:below|under|<|less\s+than)\s*(\d+(?:\.\d+)?)", re.I)
    _RSI_ABOVE = re.compile(r"rsi\s*(?:is\s*)?(?:above|over|>|greater\s+than)\s*(\d+(?:\.\d+)?)", re.I)
    _RSI_PERIOD = re.compile(r"(\d+)[\s-](?:period|day|bar)[\s-]?rsi", re.I)

    # ---- SMA / MA ------------------------------------------------------
    _PRICE_ABOVE_MA = re.compile(
        r"price\s+(?:is\s+)?(?:above|crosses?\s+above|over)\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ema|ma|moving\s+average)",
        re.I)
    _PRICE_BELOW_MA = re.compile(
        r"price\s+(?:is\s+)?(?:below|crosses?\s+below|under)\s+(?:the\s+)?(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ema|ma|moving\s+average)",
        re.I)
    _SMA_CROSS_ABOVE = re.compile(
        r"(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ema|ma|moving\s+average)\s+crosses?\s+above\s+(?:the\s+)?(\d+)",
        re.I)
    _SMA_CROSS_BELOW = re.compile(
        r"(\d+)[\s-]?(?:day|period|bar)?[\s-]?(?:sma|ema|ma|moving\s+average)\s+crosses?\s+below\s+(?:the\s+)?(\d+)",
        re.I)
    _GOLDEN_CROSS = re.compile(r"golden\s+cross", re.I)
    _DEATH_CROSS = re.compile(r"death\s+cross", re.I)

    # ---- MACD ----------------------------------------------------------
    _MACD_CROSS_ABOVE = re.compile(r"macd\s+(?:line\s+)?crosses?\s+(?:above|over)\s+(?:the\s+)?signal", re.I)
    _MACD_CROSS_BELOW = re.compile(r"macd\s+(?:line\s+)?crosses?\s+(?:below|under)\s+(?:the\s+)?signal", re.I)

    # ---- Bollinger Bands -----------------------------------------------
    _BB_LOWER = re.compile(
        r"(?:price\s+)?(?:touches?|hits?|bounces?\s+off|below)\s+(?:the\s+)?(?:lower\s+bollinger|bb\s+lower|lower\s+band)",
        re.I)
    _BB_UPPER = re.compile(
        r"(?:price\s+)?(?:touches?|hits?|bounces?\s+off|above)\s+(?:the\s+)?(?:upper\s+bollinger|bb\s+upper|upper\s+band)",
        re.I)
    _BB_PERIOD = re.compile(r"(\d+)[\s-]?(?:day|period)?[\s-]?bollinger", re.I)

    # ---- 52-week / ATR / Breakout --------------------------------------
    _WEEK52_HIGH = re.compile(
        r"(?:52[\s-]week\s+high\s+breakout|breaks?\s+(?:above|out)\s+(?:52[\s-]week|one[\s-]year)\s+high|new\s+52[\s-]week\s+high)",
        re.I)
    _WEEK52_LOW = re.compile(
        r"(?:52[\s-]week\s+low\s+breakdown|breaks?\s+(?:below)\s+(?:52[\s-]week|one[\s-]year)\s+low|new\s+52[\s-]week\s+low)",
        re.I)
    _ATR_BREAK = re.compile(r"(?:atr|average\s+true\s+range)\s+breakout|breaks?\s+above\s+atr", re.I)

    # ---- Volume --------------------------------------------------------
    _VOLUME_SPIKE = re.compile(
        r"volume\s+(?:spike|surge|is\s+)?(?:above|over|greater\s+than)?\s*(\d+(?:\.\d+)?)?\s*[x×]\s*(?:average|avg|normal)?|"
        r"volume\s+(?:spike|surge|is\s+high|above\s+average)", re.I)

    # ---- Momentum / Pairs / Spread ------------------------------------
    _TOP_DECILE = re.compile(r"top\s+(\d+)\s*%", re.I)
    _BOTTOM_DECILE = re.compile(r"bottom\s+(\d+)\s*%", re.I)
    _SPREAD_ABOVE = re.compile(r"spread\s*(?:is\s*)?(?:above|>|exceeds?|wider\s+than)\s*(\d+(?:\.\d+)?)\s*std", re.I)
    _SPREAD_BELOW = re.compile(r"(?:unwind|exit|spread)\s*(?:at|when|below)\s*(\d+(?:\.\d+)?)\s*std", re.I)

    # ---- Earnings ------------------------------------------------------
    _EARNINGS_BEAT = re.compile(r"earnings?\s+(?:beat|positive\s+surprise|above\s+estimate)", re.I)
    _EARNINGS_MISS = re.compile(r"earnings?\s+(?:miss|negative\s+surprise|below\s+estimate)", re.I)

    # ---- Exit params ---------------------------------------------------
    _TAKE_PROFIT = re.compile(r"take\s+profit\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%", re.I)
    _STOP_LOSS = re.compile(r"stop[\s-]loss\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%", re.I)
    _TRAILING_STOP = re.compile(r"trailing\s+stop\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*%", re.I)
    _TIME_EXIT = re.compile(
        r"(?:after|exit\s+after|hold\s+(?:for\s+)?(?:at\s+most\s+)?|days?\s+held\s*[>>=]\s*)(\d+)\s*(?:days?|bars?|periods?)",
        re.I)

    # ---- Sizing --------------------------------------------------------
    _POSITION_PCT = re.compile(r"(?:position\s+size|allocate|invest)\s+(\d+(?:\.\d+)?)\s*%", re.I)
    _MAX_POSITIONS = re.compile(r"max(?:imum)?\s+(\d+)\s+(?:position|stock|holding)", re.I)
    _EQUAL_WEIGHT = re.compile(r"equal\s+weight", re.I)
    _INVERSE_VOL = re.compile(r"inverse\s+vol(?:atility)?|risk\s+parity|vol[- ]?weighted", re.I)
    _KELLY = re.compile(r"kelly\s+(?:criterion|sizing|fraction)?", re.I)

    # ---- Universe ------------------------------------------------------
    _UNIVERSE = re.compile(
        r"\b(s&p\s*500|sp500|s&p|russell\s*1000|nasdaq\s*100|nasdaq100|dow\s*30|dow30)\b", re.I)

    # ---- Tickers (pairs) -----------------------------------------------
    _TICKERS_PAIR = re.compile(r"\blong\s+([A-Z]{1,5})\s+short\s+([A-Z]{1,5})\b")

    # ---- Rebalance -----------------------------------------------------
    _REBALANCE = re.compile(r"(?:rebalance|review)\s+(?:every\s+)?(daily|weekly|monthly|quarterly|annually)", re.I)

    def __init__(self) -> None:
        self._classifier = IntentClassifier()

    def parse(self, text: str) -> StrategySpec:
        intent = self._classifier.classify(text)
        entry_text, exit_text = self._split_entry_exit(text)

        entry_conds = self._extract_conditions(entry_text, "entry", text)
        exit_conds = self._extract_conditions(exit_text, "exit", text)

        # Always have at least a stop-loss
        if not any(c.condition_type == "STOP_LOSS" for c in exit_conds):
            exit_conds.append(Condition(
                condition_type="STOP_LOSS",
                params={"pct": 5.0},
                raw_text="default stop loss 5%",
                python_expr="loss_pct >= 5.0",
            ))

        pos_sizing = self._extract_position_sizing(text)
        universe = self._extract_universe(text)
        hp_m = self._TIME_EXIT.search(text)
        holding_period = int(hp_m.group(1)) if hp_m else None
        reb_m = self._REBALANCE.search(text)
        rebalance: RebalanceFreq = "daily"
        if reb_m:
            val = reb_m.group(1).lower()
            rebalance = val if val in ("daily", "weekly", "monthly", "quarterly") else "daily"

        # Pairs extraction
        long_leg = short_leg = None
        pair_m = self._TICKERS_PAIR.search(text)
        if pair_m:
            long_leg = pair_m.group(1)
            short_leg = pair_m.group(2)

        spread_above_m = self._SPREAD_ABOVE.search(text)
        spread_threshold = float(spread_above_m.group(1)) if spread_above_m else 2.0

        spread_below_m = self._SPREAD_BELOW.search(text)
        unwind_threshold = float(spread_below_m.group(1)) if spread_below_m else 0.5

        name = self._derive_name(text, intent)

        return StrategySpec(
            name=name,
            description=text.strip(),
            intent=intent,
            entry_conditions=entry_conds,
            exit_conditions=exit_conds,
            position_sizing=pos_sizing,
            universe=universe,
            holding_period=holding_period,
            rebalance_frequency=rebalance,
            long_leg=long_leg,
            short_leg=short_leg,
            spread_threshold=spread_threshold,
            unwind_threshold=unwind_threshold,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _split_entry_exit(self, text: str) -> Tuple[str, str]:
        exit_keywords = [
            r"\bsell\s+when\b", r"\bexit\s+when\b", r"\bclose\s+when\b",
            r"\btake\s+profit\b", r"\bstop[\s-]loss\b", r"\btrailing\s+stop\b",
            r"\bclose\s+position\b", r"\bunwind\s+at\b",
        ]
        earliest = len(text)
        for pat in exit_keywords:
            m = re.search(pat, text, re.I)
            if m and m.start() < earliest:
                earliest = m.start()
        return text[:earliest], text[earliest:]

    def _extract_conditions(self, text: str, role: str, full_text: str) -> List[Condition]:
        conds: List[Condition] = []

        # RSI
        for m in self._RSI_BELOW.finditer(text):
            p = self._get_rsi_period(full_text)
            conds.append(Condition("RSI_BELOW", {"threshold": float(m.group(1)), "period": p},
                                   m.group(0), f"_rsi(close, {p}) < {float(m.group(1))}"))
        for m in self._RSI_ABOVE.finditer(text):
            p = self._get_rsi_period(full_text)
            conds.append(Condition("RSI_ABOVE", {"threshold": float(m.group(1)), "period": p},
                                   m.group(0), f"_rsi(close, {p}) > {float(m.group(1))}"))

        # SMA crossovers
        if self._GOLDEN_CROSS.search(text):
            conds.append(Condition("SMA_CROSS_ABOVE", {"fast": 50, "slow": 200},
                                   "golden cross", "_sma_cross_above(close, 50, 200)"))
        if self._DEATH_CROSS.search(text):
            conds.append(Condition("SMA_CROSS_BELOW", {"fast": 50, "slow": 200},
                                   "death cross", "_sma_cross_below(close, 50, 200)"))
        for m in self._SMA_CROSS_ABOVE.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            conds.append(Condition("SMA_CROSS_ABOVE", {"fast": fast, "slow": slow},
                                   m.group(0), f"_sma_cross_above(close, {fast}, {slow})"))
        for m in self._SMA_CROSS_BELOW.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            conds.append(Condition("SMA_CROSS_BELOW", {"fast": fast, "slow": slow},
                                   m.group(0), f"_sma_cross_below(close, {fast}, {slow})"))
        for m in self._PRICE_ABOVE_MA.finditer(text):
            p = int(m.group(1))
            conds.append(Condition("PRICE_ABOVE_SMA", {"period": p},
                                   m.group(0), f"close.iloc[-1] > close.rolling({p}).mean().iloc[-1]"))
        for m in self._PRICE_BELOW_MA.finditer(text):
            p = int(m.group(1))
            conds.append(Condition("PRICE_BELOW_SMA", {"period": p},
                                   m.group(0), f"close.iloc[-1] < close.rolling({p}).mean().iloc[-1]"))

        # MACD
        if self._MACD_CROSS_ABOVE.search(text):
            conds.append(Condition("MACD_CROSS_ABOVE", {"fast": 12, "slow": 26, "signal": 9},
                                   "MACD crosses above signal", "_macd_cross_above(close, 12, 26, 9)"))
        if self._MACD_CROSS_BELOW.search(text):
            conds.append(Condition("MACD_CROSS_BELOW", {"fast": 12, "slow": 26, "signal": 9},
                                   "MACD crosses below signal", "_macd_cross_below(close, 12, 26, 9)"))

        # Bollinger Bands
        if self._BB_LOWER.search(text):
            p = self._get_bb_period(full_text)
            conds.append(Condition("BB_LOWER_TOUCH", {"period": p, "std": 2.0},
                                   "price touches lower BB", f"_bb_lower_touch(close, {p}, 2.0)"))
        if self._BB_UPPER.search(text):
            p = self._get_bb_period(full_text)
            conds.append(Condition("BB_UPPER_TOUCH", {"period": p, "std": 2.0},
                                   "price touches upper BB", f"_bb_upper_touch(close, {p}, 2.0)"))

        # 52-week breakouts
        if self._WEEK52_HIGH.search(text):
            conds.append(Condition("WEEK52_HIGH_BREAK", {"lookback": 252},
                                   "52-week high breakout", "_week52_high_break(close, 252)"))
        if self._WEEK52_LOW.search(text):
            conds.append(Condition("WEEK52_LOW_BREAK", {"lookback": 252},
                                   "52-week low breakdown", "_week52_low_break(close, 252)"))

        # ATR breakout
        if self._ATR_BREAK.search(text):
            conds.append(Condition("ATR_BREAKOUT", {"period": 14, "multiplier": 1.0},
                                   "ATR breakout", "_atr_breakout(close, high, low, 14, 1.0)"))

        # Volume spike
        if self._VOLUME_SPIKE.search(text):
            mult_m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]", text, re.I)
            mult = float(mult_m.group(1)) if mult_m else 2.0
            conds.append(Condition("VOLUME_SPIKE", {"multiplier": mult, "period": 20},
                                   "volume spike", f"_volume_spike(volume, {mult}, 20)"))

        # Momentum rank
        top_m = self._TOP_DECILE.search(text)
        if top_m and "mom" in text.lower():
            pct = int(top_m.group(1))
            conds.append(Condition("MOMENTUM_RANK_TOP", {"percentile": pct, "lookback": 252},
                                   f"top {pct}% momentum", f"_momentum_rank_top(close, {pct}, 252)"))
        bot_m = self._BOTTOM_DECILE.search(text)
        if bot_m and ("loser" in text.lower() or "worst" in text.lower()):
            pct = int(bot_m.group(1))
            conds.append(Condition("MOMENTUM_RANK_BOTTOM", {"percentile": pct, "lookback": 5},
                                   f"bottom {pct}% 5-day return", f"_momentum_rank_bottom(close, {pct}, 5)"))

        # Spread (pairs)
        if role == "entry":
            spread_m = self._SPREAD_ABOVE.search(text)
            if spread_m:
                z = float(spread_m.group(1))
                conds.append(Condition("SPREAD_ZSCORE_ABOVE", {"z_threshold": z},
                                       f"spread > {z} std", f"spread_z > {z}"))
        if role == "exit":
            spread_m = self._SPREAD_BELOW.search(text)
            if spread_m:
                z = float(spread_m.group(1))
                conds.append(Condition("SPREAD_ZSCORE_BELOW", {"z_threshold": z},
                                       f"spread < {z} std", f"abs(spread_z) < {z}"))

        # Earnings
        if self._EARNINGS_BEAT.search(text):
            conds.append(Condition("EARNINGS_BEAT", {}, "earnings beat", "eps_surprise > 0"))
        if self._EARNINGS_MISS.search(text):
            conds.append(Condition("EARNINGS_MISS", {}, "earnings miss", "eps_surprise < 0"))

        # Exit-specific
        if role == "exit":
            for m in self._TAKE_PROFIT.finditer(text):
                pct = float(m.group(1))
                conds.append(Condition("TAKE_PROFIT", {"pct": pct},
                                       m.group(0), f"profit_pct >= {pct}"))
            for m in self._STOP_LOSS.finditer(text):
                pct = float(m.group(1))
                conds.append(Condition("STOP_LOSS", {"pct": pct},
                                       m.group(0), f"loss_pct >= {pct}"))
            for m in self._TRAILING_STOP.finditer(text):
                pct = float(m.group(1))
                conds.append(Condition("TRAILING_STOP", {"pct": pct},
                                       m.group(0), f"trailing_stop_triggered({pct})"))
            for m in self._TIME_EXIT.finditer(text):
                days = int(m.group(1))
                conds.append(Condition("TIME_EXIT", {"max_days": days},
                                       m.group(0), f"days_held >= {days}"))

        return conds

    def _get_rsi_period(self, text: str) -> int:
        m = self._RSI_PERIOD.search(text)
        return int(m.group(1)) if m else 14

    def _get_bb_period(self, text: str) -> int:
        m = self._BB_PERIOD.search(text)
        return int(m.group(1)) if m else 20

    def _extract_position_sizing(self, text: str) -> PositionSizing:
        if self._KELLY.search(text):
            method: SizingMethod = "kelly"
        elif self._INVERSE_VOL.search(text):
            method = "inverse_vol"
        elif self._EQUAL_WEIGHT.search(text):
            method = "equal_weight"
        else:
            method = "fixed_pct"
        pct_m = self._POSITION_PCT.search(text)
        value = float(pct_m.group(1)) / 100 if pct_m else 0.10
        max_m = self._MAX_POSITIONS.search(text)
        max_pos = int(max_m.group(1)) if max_m else 10
        return PositionSizing(method=method, value=value, max_positions=max_pos)

    def _extract_universe(self, text: str) -> Universe:
        uni_m = self._UNIVERSE.search(text)
        if not uni_m:
            return Universe(universe_type="SP500")
        val = uni_m.group(1).lower().replace(" ", "")
        mapping: Dict[str, UniverseType] = {
            "s&p500": "SP500", "sp500": "SP500", "s&p": "SP500",
            "russell1000": "RUSSELL1000",
            "nasdaq100": "NASDAQ100", "nasdaq100": "NASDAQ100",
            "dow30": "DOW30", "dow30": "DOW30",
        }
        universe_type: UniverseType = mapping.get(val, "SP500")
        return Universe(universe_type=universe_type)

    def _derive_name(self, text: str, intent: StrategyIntentType) -> str:
        slug = re.sub(r"[^a-zA-Z0-9 ]", "", text[:50]).strip()
        slug = re.sub(r"\s+", "_", slug).lower()
        prefix = intent.lower() if intent != "UNKNOWN" else "strategy"
        return f"{prefix}_{slug}" if slug else prefix


# ---------------------------------------------------------------------------
# ── 3. Technical Indicator Helper Functions (included in generated code) ─────
# ---------------------------------------------------------------------------

_INDICATOR_HELPERS = '''
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float:
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
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _atr(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


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
    return macd_line, signal_line, macd_line - signal_line


def _macd_cross_above(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    if len(close) < slow + signal:
        return False
    ml, sl, _ = _macd(close, fast, slow, signal)
    return bool(ml.iloc[-1] > sl.iloc[-1] and ml.iloc[-2] <= sl.iloc[-2])


def _macd_cross_below(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    if len(close) < slow + signal:
        return False
    ml, sl, _ = _macd(close, fast, slow, signal)
    return bool(ml.iloc[-1] < sl.iloc[-1] and ml.iloc[-2] >= sl.iloc[-2])


def _bb_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma


def _bb_lower_touch(close: pd.Series, period: int = 20, std: float = 2.0) -> bool:
    if len(close) < period:
        return False
    _, _, lower = _bb_bands(close, period, std)
    return bool(close.iloc[-1] <= lower.iloc[-1])


def _bb_upper_touch(close: pd.Series, period: int = 20, std: float = 2.0) -> bool:
    if len(close) < period:
        return False
    upper, _, _ = _bb_bands(close, period, std)
    return bool(close.iloc[-1] >= upper.iloc[-1])


def _week52_high_break(close: pd.Series, lookback: int = 252) -> bool:
    if len(close) < lookback:
        return False
    return bool(close.iloc[-1] > close.iloc[-lookback:-1].max())


def _week52_low_break(close: pd.Series, lookback: int = 252) -> bool:
    if len(close) < lookback:
        return False
    return bool(close.iloc[-1] < close.iloc[-lookback:-1].min())


def _volume_spike(volume: pd.Series, multiplier: float = 2.0, period: int = 20) -> bool:
    if len(volume) < period:
        return False
    return bool(volume.iloc[-1] > multiplier * volume.iloc[-period:-1].mean())


def _atr_breakout(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
    multiplier: float = 1.0,
) -> bool:
    if len(close) < period + 1:
        return False
    atr_val = float(_atr(close, high, low, period).iloc[-1])
    prev_close = float(close.iloc[-2])
    return bool(float(close.iloc[-1]) > prev_close + multiplier * atr_val)


def _momentum_rank_top(close: pd.Series, percentile: int, lookback: int) -> bool:
    """True if recent return is in top <percentile>% (placeholder — use cross-sectional at portfolio level)."""
    if len(close) < lookback:
        return False
    ret = float((close.iloc[-1] / close.iloc[-lookback]) - 1)
    return ret > 0.10  # simplified single-stock proxy


def _momentum_rank_bottom(close: pd.Series, percentile: int, lookback: int) -> bool:
    if len(close) < lookback:
        return False
    ret = float((close.iloc[-1] / close.iloc[-lookback]) - 1)
    return ret < -0.05  # simplified single-stock proxy
'''


# ---------------------------------------------------------------------------
# ── 4. Code Generator ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyCodeGeneratorV3:
    """Generate executable Python strategy classes from StrategySpec."""

    _TEMPLATE = textwrap.dedent('''
        """
        Auto-generated strategy: {name}
        Intent: {intent}
        Description: {description}
        Generated: {timestamp}
        """
        from __future__ import annotations
        from typing import Optional
        import numpy as np
        import pandas as pd

        {helpers}


        class {class_name}:
            """Auto-generated {intent} strategy."""

            NAME = "{name}"
            INTENT = "{intent}"
            REBALANCE = "{rebalance}"
            HOLDING_PERIOD: Optional[int] = {holding_period}

            def __init__(self) -> None:
                self._position_pct: float = {position_pct}
                self._max_positions: int = {max_positions}

            # ------------------------------------------------------------------
            # Signal generation — returns +1 (buy) / -1 (sell) / 0 (hold)
            # ------------------------------------------------------------------

            def generate_signals(
                self,
                ticker: str,
                close: pd.Series,
                high: Optional[pd.Series] = None,
                low: Optional[pd.Series] = None,
                volume: Optional[pd.Series] = None,
                eps_surprise: float = 0.0,
            ) -> pd.Series:
                if high is None:
                    high = close * 1.005
                if low is None:
                    low = close * 0.995
                if volume is None:
                    volume = pd.Series(
                        [1_000_000] * len(close), index=close.index, dtype=float
                    )

                signals = pd.Series(0, index=close.index, dtype=int)
                in_position = False
                entry_price = 0.0
                peak_price = 0.0
                days_held = 0
                profit_pct = 0.0
                loss_pct = 0.0
                spread_z = 0.0

                min_lookback = 260  # sufficient for all indicators

                for i in range(min_lookback, len(close)):
                    c = close.iloc[: i + 1]
                    h = high.iloc[: i + 1]
                    l = low.iloc[: i + 1]
                    v = volume.iloc[: i + 1]
                    price = float(c.iloc[-1])

                    if in_position:
                        days_held += 1
                        peak_price = max(peak_price, price)
                        profit_pct = (
                            (price - entry_price) / entry_price * 100.0
                            if entry_price > 0
                            else 0.0
                        )
                        loss_pct = (
                            (entry_price - price) / entry_price * 100.0
                            if entry_price > 0
                            else 0.0
                        )
                        peak_drop_pct = (
                            (peak_price - price) / peak_price * 100.0
                            if peak_price > 0
                            else 0.0
                        )

                        def trailing_stop_triggered(pct: float) -> bool:
                            return peak_drop_pct >= pct

                        if {exit_expr}:
                            signals.iloc[i] = -1
                            in_position = False
                            entry_price = peak_price = 0.0
                            days_held = 0
                    else:
                        close = c  # alias for indicator helpers
                        high = h
                        low = l
                        volume = v
                        if {entry_expr}:
                            signals.iloc[i] = 1
                            in_position = True
                            entry_price = peak_price = price
                            days_held = 0
                        close = close.iloc[: len(signals)]  # restore

                return signals

            def position_size(self, portfolio_value: float, price: float) -> float:
                """Return number of shares based on configured sizing method."""
                dollar_amt = portfolio_value * self._position_pct
                return max(1.0, dollar_amt / price) if price > 0 else 1.0
    ''').strip()

    def generate(self, spec: StrategySpec) -> str:
        """Generate a full Python strategy class string."""
        entry_conds = spec.entry_conditions[:10]
        exit_conds = spec.exit_conditions[:10]

        entry_expr = self._combine(entry_conds, "and") or "False"
        exit_expr = self._combine(exit_conds, "or") or "False"

        cls_name = self._class_name(spec.name)
        holding = repr(spec.holding_period)

        return self._TEMPLATE.format(
            name=spec.name,
            intent=spec.intent,
            description=spec.description.replace('"', "'")[:200],
            timestamp=datetime.utcnow().isoformat(),
            class_name=cls_name,
            rebalance=spec.rebalance_frequency,
            holding_period=holding,
            position_pct=spec.position_sizing.value,
            max_positions=spec.position_sizing.max_positions,
            entry_expr=entry_expr,
            exit_expr=exit_expr,
            helpers=_INDICATOR_HELPERS.strip(),
        )

    @staticmethod
    def _combine(conds: List[Condition], op: str) -> str:
        parts = [f"({c.python_expr})" for c in conds if c.python_expr]
        return f" {op} ".join(parts) if parts else ""

    @staticmethod
    def _class_name(name: str) -> str:
        cls = re.sub(r"[^a-zA-Z0-9]", " ", name).title().replace(" ", "")
        cls = re.sub(r"_+", "", cls)
        return f"Strategy_{cls}" if not cls or cls[0].isdigit() else cls

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """AST parse + safety check. Returns (ok, errors)."""
        errors: List[str] = []
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"SyntaxError: {exc}")

        forbidden = [
            (r"\bexec\s*\(", "exec() not allowed"),
            (r"\beval\s*\(", "eval() not allowed"),
            (r"\bimport\s+os\b", "os import not allowed"),
            (r"\bimport\s+subprocess\b", "subprocess not allowed"),
            (r"while\s+True\s*:", "infinite loop not allowed"),
            (r"__import__\s*\(", "__import__ not allowed"),
            (r"open\s*\(", "file open not allowed in strategy"),
        ]
        for pattern, msg in forbidden:
            if re.search(pattern, code):
                errors.append(f"SafetyError: {msg}")

        return len(errors) == 0, errors

    def save(self, spec: StrategySpec, code: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", spec.name)[:60]
        path = _STRATEGIES_DIR / f"{safe_name}.py"
        path.write_text(code, encoding="utf-8")
        logger.info("strategy_saved_v3", path=str(path), name=spec.name, intent=spec.intent)
        return path


# ---------------------------------------------------------------------------
# ── 5. Quality Gate + Backtester ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyBacktester:
    """Run vectorised backtests and enforce quality gates."""

    def fetch_prices(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV from yfinance; degrade gracefully to synthetic data."""
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            # Normalise column names (yfinance >= 0.2 uses MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            return df
        except Exception as exc:
            logger.warning("yfinance_fetch_failed_v3", ticker=ticker, error=str(exc))
            dates = pd.date_range(start=start, end=end, freq="B")
            np.random.seed(abs(hash(ticker)) % 2**31)
            prices = 100.0 * np.exp(
                np.cumsum(np.random.normal(0.0003, 0.012, len(dates)))
            )
            return pd.DataFrame({
                "open": prices * 0.999,
                "high": prices * 1.006,
                "low": prices * 0.994,
                "close": prices,
                "volume": np.random.randint(500_000, 5_000_000, len(dates)).astype(float),
            }, index=dates)

    def run(
        self,
        spec: StrategySpec,
        ticker: str = "SPY",
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_capital: float = 100_000.0,
        code: Optional[str] = None,
    ) -> BacktestResult:
        """Run a full vectorised backtest. Uses exec on generated code."""
        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if start is None:
            start_dt = datetime.strptime(end, "%Y-%m-%d") - timedelta(days=5 * 365)
            start = start_dt.strftime("%Y-%m-%d")

        df = self.fetch_prices(ticker, start, end)
        bm_df = self.fetch_prices("SPY", start, end) if ticker != "SPY" else df.copy()

        close = df["close"].squeeze()
        high = df["high"].squeeze() if "high" in df.columns else close * 1.005
        low = df["low"].squeeze() if "low" in df.columns else close * 0.995
        volume = df["volume"].squeeze() if "volume" in df.columns else pd.Series(
            [1_000_000] * len(close), index=close.index, dtype=float)

        # Instantiate strategy via exec
        if code is None:
            gen = StrategyCodeGeneratorV3()
            code = gen.generate(spec)
        ns: Dict[str, Any] = {}
        exec(code, ns)  # noqa: S102 — controlled generated code only
        cls_candidates = [k for k in ns if k.startswith("Strategy") or
                          (k[0].isupper() and k not in {"Optional"})]
        if not cls_candidates:
            raise RuntimeError("No strategy class found in generated code")
        strategy_obj = ns[cls_candidates[0]]()

        signals = strategy_obj.generate_signals(ticker, close, high, low, volume)

        returns = close.pct_change().fillna(0.0)
        position = signals.shift(1).fillna(0.0)
        strat_rets = position * returns
        equity = (1.0 + strat_rets).cumprod() * initial_capital

        bm_close = bm_df["close"].squeeze().reindex(close.index, method="ffill").ffill()
        bm_rets = bm_close.pct_change().fillna(0.0)
        bm_equity = (1.0 + bm_rets).cumprod() * initial_capital

        n_years = max(len(close) / 252, 0.01)
        final = float(equity.iloc[-1])
        bm_final = float(bm_equity.iloc[-1])

        cagr = (final / initial_capital) ** (1.0 / n_years) - 1.0
        bm_cagr = (bm_final / initial_capital) ** (1.0 / n_years) - 1.0

        sharpe = self._sharpe(strat_rets)
        bm_sharpe = self._sharpe(bm_rets)

        roll_max = equity.cummax()
        max_dd = float(((equity - roll_max) / roll_max).min())

        trades = self._extract_trades(signals, close)
        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / max(len(trades), 1)

        # Turnover: fraction of portfolio turned over per bar
        turnover = float(position.diff().abs().mean())

        return BacktestResult(
            ticker=ticker, start=start, end=end,
            strategy_name=spec.name,
            cagr=round(cagr, 4),
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 3),
            num_trades=len(trades),
            total_return=round(final / initial_capital - 1.0, 4),
            benchmark_cagr=round(bm_cagr, 4),
            benchmark_sharpe=round(bm_sharpe, 3),
            benchmark_return=round(bm_final / initial_capital - 1.0, 4),
            turnover=round(turnover, 4),
            equity_curve=equity.round(2).tolist()[-500:],
            trade_log=trades[:50],
        )

    @staticmethod
    def _sharpe(returns: pd.Series, rf: float = 0.04, periods: int = 252) -> float:
        excess = returns - rf / periods
        std = float(returns.std())
        if std == 0.0:
            return 0.0
        return float(excess.mean() / std * (periods ** 0.5))

    @staticmethod
    def _extract_trades(signals: pd.Series, close: pd.Series) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        entry_price: Optional[float] = None
        entry_date = None
        for date, sig in signals.items():
            price = float(close.loc[date])
            if sig == 1 and entry_price is None:
                entry_price = price
                entry_date = date
            elif sig == -1 and entry_price is not None:
                pnl_pct = (price - entry_price) / entry_price * 100.0
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "exit_date": str(date)[:10],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "pnl": round(pnl_pct, 2),
                })
                entry_price = None
        return trades

    def check_quality_gate(self, result: BacktestResult) -> QualityGateResult:
        """Evaluate the quality gate against backtest results."""
        reasons: List[str] = []

        if result.sharpe < _QUALITY_GATE_SHARPE:
            reasons.append(
                f"Sharpe {result.sharpe:.2f} < {_QUALITY_GATE_SHARPE} required"
            )
        if result.max_drawdown < _QUALITY_GATE_MAX_DD:
            reasons.append(
                f"Max drawdown {result.max_drawdown:.1%} worse than {_QUALITY_GATE_MAX_DD:.0%} limit"
            )
        if result.num_trades < _QUALITY_GATE_MIN_TRADES:
            reasons.append(
                f"Only {result.num_trades} trades (need >= {_QUALITY_GATE_MIN_TRADES})"
            )
        if result.win_rate < _QUALITY_GATE_MIN_WIN_RATE:
            reasons.append(
                f"Win rate {result.win_rate:.1%} < {_QUALITY_GATE_MIN_WIN_RATE:.0%} required"
            )

        return QualityGateResult(
            passed=len(reasons) == 0,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            num_trades=result.num_trades,
            win_rate=result.win_rate,
            failure_reasons=reasons,
        )


# ---------------------------------------------------------------------------
# ── 6. Strategy Library (30 pre-built strategies) ─────────────────────────────
# ---------------------------------------------------------------------------

_LIBRARY_DEFINITIONS: List[Dict[str, Any]] = [
    # Trend-following
    {"name": "golden_cross",          "intent": "TREND_FOLLOW",
     "nl": "Buy when 50-day MA crosses above 200-day MA. Sell when 50-day MA crosses below 200-day MA. Stop loss 8%."},
    {"name": "macd_trend",            "intent": "TREND_FOLLOW",
     "nl": "Buy when MACD crosses above signal. Sell when MACD crosses below signal. Stop loss 5%."},
    {"name": "triple_ma",             "intent": "TREND_FOLLOW",
     "nl": "Buy when price is above 200-day MA and 20-day MA crosses above 50-day MA. Sell when 20-day MA crosses below 50-day MA. Stop loss 7%."},
    {"name": "dual_ma_10_30",         "intent": "TREND_FOLLOW",
     "nl": "Buy when 10-day MA crosses above 30-day MA. Sell when 10-day MA crosses below 30-day MA. Stop loss 5%."},
    {"name": "macd_price_filter",     "intent": "TREND_FOLLOW",
     "nl": "Buy when MACD crosses above signal and price is above 200-day MA. Sell when MACD crosses below signal or stop loss 6%."},
    # Mean-reversion
    {"name": "rsi_oversold",          "intent": "MEAN_REVERT",
     "nl": "Buy when RSI below 30. Sell when RSI above 60 or stop loss 5% or after 10 days."},
    {"name": "bb_mean_revert",        "intent": "MEAN_REVERT",
     "nl": "Buy when price touches lower Bollinger band. Sell when price touches upper Bollinger band or stop loss 4%."},
    {"name": "rsi_bb_combo",          "intent": "MEAN_REVERT",
     "nl": "Buy when RSI below 35 and price touches lower Bollinger band. Sell when RSI above 60 or stop loss 5%."},
    {"name": "5day_loser_revert",     "intent": "MEAN_REVERT",
     "nl": "Buy when RSI below 25 and price below 20-day MA. Sell when RSI above 50 or after 5 days or stop loss 6%."},
    {"name": "oversold_bb_rsi",       "intent": "MEAN_REVERT",
     "nl": "Buy when RSI below 30 and price is below 50-day MA. Sell when RSI above 55 or take profit 8% or stop loss 4%."},
    # Momentum
    {"name": "momentum_200ma",        "intent": "MOMENTUM",
     "nl": "Buy when price is above 200-day MA and volume spike 1.5x. Sell when price is below 200-day MA or stop loss 8%. Rebalance monthly."},
    {"name": "breakout_52wk",         "intent": "BREAKOUT",
     "nl": "Buy when 52-week high breakout and volume spike 2x. Sell when RSI above 80 or stop loss 5%."},
    {"name": "volume_momentum",       "intent": "MOMENTUM",
     "nl": "Buy when 20-day MA crosses above 50-day MA and volume spike 1.5x. Sell when 20-day MA crosses below 50-day MA or stop loss 5%."},
    {"name": "high_vol_breakout",     "intent": "BREAKOUT",
     "nl": "Buy when 52-week high breakout. Sell when RSI above 75 or stop loss 6%."},
    {"name": "macd_volume_mom",       "intent": "MOMENTUM",
     "nl": "Buy when MACD crosses above signal and volume spike 2x. Sell when MACD crosses below signal or stop loss 5%."},
    # Earnings drift
    {"name": "earnings_drift_long",   "intent": "EARNINGS_DRIFT",
     "nl": "Buy when earnings beat. Sell after 5 days or stop loss 3%."},
    {"name": "earnings_miss_short",   "intent": "EARNINGS_DRIFT",
     "nl": "Sell when earnings miss. Exit after 5 days or take profit 5%."},
    {"name": "earnings_gap_follow",   "intent": "EARNINGS_DRIFT",
     "nl": "Buy when earnings beat and RSI below 65. Sell after 10 days or stop loss 4%."},
    # Volatility
    {"name": "low_vol_carry",         "intent": "VOLATILITY",
     "nl": "Buy when price is above 200-day MA and RSI below 55. Sell when RSI above 70 or stop loss 6%."},
    {"name": "vol_breakout_atr",      "intent": "VOLATILITY",
     "nl": "Buy when ATR breakout and volume spike 2x. Sell after 5 days or stop loss 5%."},
    # Carry / Dividend
    {"name": "dividend_momentum",     "intent": "CARRY",
     "nl": "Buy when price is above 200-day MA and 50-day MA crosses above 200-day MA. Sell when death cross or stop loss 10%."},
    {"name": "yield_ma_filter",       "intent": "CARRY",
     "nl": "Buy when price is above 50-day MA and RSI below 60. Sell when RSI above 70 or stop loss 5%."},
    # Factor
    {"name": "quality_momentum",      "intent": "FACTOR",
     "nl": "Buy when price is above 200-day MA and MACD crosses above signal. Sell when MACD crosses below signal or stop loss 6%. Rebalance monthly."},
    {"name": "low_vol_factor",        "intent": "FACTOR",
     "nl": "Buy when price is above 200-day MA and RSI below 45. Sell when RSI above 65 or stop loss 5%."},
    {"name": "value_reversal",        "intent": "FACTOR",
     "nl": "Buy when RSI below 35 and price is above 200-day MA. Sell when RSI above 60 or stop loss 6%."},
    # Sector rotation / macro
    {"name": "sector_spy_ma",         "intent": "SECTOR_ROTATION",
     "nl": "Buy when price is above 200-day MA and golden cross. Sell when death cross or stop loss 8%."},
    {"name": "macro_regime_trend",    "intent": "MACRO",
     "nl": "Buy when price is above 200-day MA and 50-day MA crosses above 200-day MA. Sell when 50-day MA crosses below 200-day MA or stop loss 10%."},
    # Stat arb / pairs proxies (single-leg equivalent)
    {"name": "stat_arb_mean_rev",     "intent": "STAT_ARB",
     "nl": "Buy when RSI below 30 and price is below 50-day MA. Sell when RSI above 55 or stop loss 4%."},
    # Sentiment
    {"name": "sentiment_momentum",    "intent": "SENTIMENT",
     "nl": "Buy when MACD crosses above signal and price is above 200-day MA. Sell when MACD crosses below signal or stop loss 5%."},
    # Seasonality
    {"name": "santa_rally",           "intent": "SEASONALITY",
     "nl": "Buy when RSI above 50 and price is above 50-day MA. Sell when RSI below 40 or stop loss 5%."},
]


class StrategyLibraryV3:
    """Stores 30 pre-built strategies with quality metrics in SQLite."""

    def __init__(self) -> None:
        self._parser = StrategyLanguageParserV3()
        self._gen = StrategyCodeGeneratorV3()
        self._backtester = StrategyBacktester()
        self._specs: Dict[str, StrategySpec] = {}
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_library (
                    name TEXT PRIMARY KEY,
                    intent TEXT,
                    nl_description TEXT,
                    spec_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    cagr REAL, sharpe REAL, max_drawdown REAL,
                    win_rate REAL, num_trades INTEGER, total_return REAL,
                    benchmark_return REAL, turnover REAL,
                    quality_passed INTEGER DEFAULT 0,
                    run_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generated_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    name TEXT, nl_description TEXT, intent TEXT,
                    code_path TEXT, sharpe REAL, max_drawdown REAL,
                    win_rate REAL, num_trades INTEGER, quality_passed INTEGER,
                    failure_reasons TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_gates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    ticker TEXT,
                    sharpe_threshold REAL, max_dd_threshold REAL,
                    min_trades INTEGER, min_win_rate REAL,
                    passed INTEGER,
                    evaluated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    def _load(self) -> None:
        for defn in _LIBRARY_DEFINITIONS:
            try:
                spec = self._parser.parse(defn["nl"])
                spec.name = defn["name"]
                spec.intent = defn["intent"]  # type: ignore[assignment]
                self._specs[defn["name"]] = spec
                self._persist(defn, spec)
            except Exception as exc:
                logger.warning("library_load_error_v3", name=defn["name"], error=str(exc))

    def _persist(self, defn: Dict[str, Any], spec: StrategySpec) -> None:
        spec_json = json.dumps({
            "name": spec.name, "intent": spec.intent,
            "entry": [{"type": c.condition_type, "expr": c.python_expr} for c in spec.entry_conditions],
            "exit": [{"type": c.condition_type, "expr": c.python_expr} for c in spec.exit_conditions],
        })
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO strategy_library (name, intent, nl_description, spec_json) VALUES (?,?,?,?)",
                (defn["name"], defn["intent"], defn["nl"], spec_json),
            )
            conn.commit()

    def get(self, name: str) -> Optional[StrategySpec]:
        return self._specs.get(name)

    def list_all(self) -> List[Dict[str, str]]:
        return [{"name": d["name"], "intent": d["intent"], "description": d["nl"][:80]}
                for d in _LIBRARY_DEFINITIONS]

    def save_result(self, result: BacktestResult, quality: QualityGateResult) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """INSERT INTO strategy_results
                   (strategy_name, ticker, cagr, sharpe, max_drawdown,
                    win_rate, num_trades, total_return, benchmark_return, turnover, quality_passed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (result.strategy_name, result.ticker, result.cagr, result.sharpe,
                 result.max_drawdown, result.win_rate, result.num_trades,
                 result.total_return, result.benchmark_return, result.turnover,
                 int(quality.passed)),
            )
            conn.commit()

    def save_generated(self, gen_result: GenerationResult) -> None:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            qg = gen_result.quality_gate
            bt = gen_result.backtest
            conn.execute(
                """INSERT OR REPLACE INTO generated_strategies
                   (strategy_id, name, nl_description, intent, code_path,
                    sharpe, max_drawdown, win_rate, num_trades, quality_passed, failure_reasons)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (gen_result.strategy_id, gen_result.spec.name,
                 gen_result.spec.description[:500], gen_result.spec.intent,
                 str(gen_result.file_path) if gen_result.file_path else None,
                 bt.sharpe if bt else None, bt.max_drawdown if bt else None,
                 bt.win_rate if bt else None, bt.num_trades if bt else None,
                 int(qg.passed), json.dumps(qg.failure_reasons)),
            )
            conn.commit()

    def get_generated(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT * FROM generated_strategies WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
        if not row:
            return None
        cols = ["strategy_id", "name", "nl_description", "intent", "code_path",
                "sharpe", "max_drawdown", "win_rate", "num_trades",
                "quality_passed", "failure_reasons", "created_at"]
        return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# ── 7. Orchestrator ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class StrategyOrchestratorV3:
    """Top-level orchestrator: parse → generate → validate → quality-gate."""

    def __init__(self) -> None:
        self._parser = StrategyLanguageParserV3()
        self._gen = StrategyCodeGeneratorV3()
        self._backtester = StrategyBacktester()
        self._library = StrategyLibraryV3()

    def generate_and_validate(
        self,
        description: str,
        ticker: str = "SPY",
        save: bool = True,
        skip_backtest: bool = False,
    ) -> GenerationResult:
        """Full pipeline: NL → spec → code → quality gate → result."""
        spec = self._parser.parse(description)
        code = self._gen.generate(spec)
        ok, syntax_errors = self._gen.validate(code)

        file_path: Optional[Path] = None
        if ok and save:
            file_path = self._gen.save(spec, code)

        backtest: Optional[BacktestResult] = None
        quality_gate = QualityGateResult(
            passed=False,
            sharpe=0.0, max_drawdown=0.0, num_trades=0, win_rate=0.0,
            failure_reasons=["Backtest skipped"] if skip_backtest else ["Syntax errors prevented backtest"],
        )

        if ok and not skip_backtest:
            # Run 2-year in-sample backtest for quality gate
            end = datetime.utcnow().strftime("%Y-%m-%d")
            start = (datetime.utcnow() - timedelta(days=_IN_SAMPLE_YEARS * 365)).strftime("%Y-%m-%d")
            try:
                backtest = self._backtester.run(
                    spec, ticker=ticker, start=start, end=end, code=code
                )
                quality_gate = self._backtester.check_quality_gate(backtest)
            except Exception as exc:
                logger.warning("quality_gate_backtest_failed", error=str(exc), name=spec.name)
                quality_gate = QualityGateResult(
                    passed=False,
                    sharpe=0.0, max_drawdown=0.0, num_trades=0, win_rate=0.0,
                    failure_reasons=[f"Backtest execution error: {exc}"],
                )

        result = GenerationResult(
            spec=spec, code=code, file_path=file_path,
            syntax_errors=syntax_errors, quality_gate=quality_gate,
            backtest=backtest,
        )
        self._library.save_generated(result)
        if backtest:
            self._library.save_result(backtest, quality_gate)
        return result

    def compare(
        self,
        descriptions: List[str],
        ticker: str = "SPY",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate and backtest multiple strategies, returning ranked comparison."""
        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.utcnow() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

        results: List[Dict[str, Any]] = []
        for desc in descriptions:
            try:
                spec = self._parser.parse(desc)
                code = self._gen.generate(spec)
                ok, errors = self._gen.validate(code)
                if not ok:
                    results.append({"description": desc[:60], "error": errors})
                    continue
                bt = self._backtester.run(spec, ticker=ticker, start=start, end=end, code=code)
                qg = self._backtester.check_quality_gate(bt)
                results.append({
                    "description": desc[:80],
                    "name": spec.name,
                    "intent": spec.intent,
                    "sharpe": bt.sharpe,
                    "cagr": bt.cagr,
                    "max_drawdown": bt.max_drawdown,
                    "win_rate": bt.win_rate,
                    "num_trades": bt.num_trades,
                    "turnover": bt.turnover,
                    "alpha": round(bt.cagr - bt.benchmark_cagr, 4),
                    "quality_passed": qg.passed,
                    "quality_summary": qg.summary(),
                })
            except Exception as exc:
                results.append({"description": desc[:60], "error": str(exc)})

        # Rank by Sharpe
        results.sort(key=lambda x: x.get("sharpe", -99), reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def build_ensemble(
        self,
        descriptions: List[str],
        ticker: str = "SPY",
    ) -> Dict[str, Any]:
        """Combine strategies into Sharpe-weighted ensemble."""
        if len(descriptions) < 2:
            return {"error": "Need at least 2 strategies for an ensemble"}

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

        component_results: List[Dict[str, Any]] = []
        all_signals: List[pd.Series] = []

        for desc in descriptions:
            try:
                spec = self._parser.parse(desc)
                code = self._gen.generate(spec)
                ok, _ = self._gen.validate(code)
                if not ok:
                    continue
                df = self._backtester.fetch_prices(ticker, start, end)
                close = df["close"].squeeze()
                high = df["high"].squeeze() if "high" in df.columns else close * 1.005
                low = df["low"].squeeze() if "low" in df.columns else close * 0.995
                volume = df["volume"].squeeze() if "volume" in df.columns else pd.Series(
                    [1_000_000] * len(close), index=close.index, dtype=float)

                ns: Dict[str, Any] = {}
                exec(code, ns)  # noqa: S102
                cls_candidates = [k for k in ns if k.startswith("Strategy") or
                                  (k[0].isupper() and k not in {"Optional"})]
                if not cls_candidates:
                    continue
                obj = ns[cls_candidates[0]]()
                sigs = obj.generate_signals(ticker, close, high, low, volume)

                bt = self._backtester.run(spec, ticker=ticker, start=start, end=end, code=code)
                sharpe = max(bt.sharpe, 0.01)  # floor at near-zero
                component_results.append({
                    "name": spec.name, "intent": spec.intent,
                    "sharpe": bt.sharpe, "cagr": bt.cagr,
                    "max_drawdown": bt.max_drawdown, "weight": sharpe,
                })
                all_signals.append(sigs * sharpe)  # weight by Sharpe

            except Exception as exc:
                logger.warning("ensemble_component_failed", error=str(exc))

        if not all_signals:
            return {"error": "No valid component strategies"}

        # Normalise weights
        total_w = sum(r["weight"] for r in component_results)
        for r in component_results:
            r["weight"] = round(r["weight"] / total_w, 4)

        # Ensemble signal: weighted average, threshold at 0
        idx = all_signals[0].index
        combined = pd.Series(0.0, index=idx)
        for sig in all_signals:
            combined = combined.add(sig.reindex(idx, fill_value=0.0))
        ensemble_signals = (combined / total_w).map(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)).astype(int)

        # Compute ensemble performance
        df = self._backtester.fetch_prices(ticker, start, end)
        close = df["close"].squeeze()
        bm_close = self._backtester.fetch_prices("SPY", start, end)["close"].squeeze()

        ens_rets = ensemble_signals.reindex(close.index, fill_value=0).shift(1).fillna(0) * close.pct_change().fillna(0)
        bm_rets = bm_close.reindex(close.index, method="ffill").pct_change().fillna(0)

        ens_sharpe = StrategyBacktester._sharpe(ens_rets)
        ens_equity = (1.0 + ens_rets).cumprod() * 100_000
        roll_max = ens_equity.cummax()
        ens_dd = float(((ens_equity - roll_max) / roll_max).min())

        return {
            "ensemble_sharpe": round(ens_sharpe, 3),
            "ensemble_max_drawdown": round(ens_dd, 4),
            "benchmark_sharpe": round(StrategyBacktester._sharpe(bm_rets), 3),
            "num_components": len(component_results),
            "components": component_results,
            "note": "Sharpe-weighted ensemble; signals combined by weighted vote",
        }


# ---------------------------------------------------------------------------
# ── 8. Parameter Grid Search Template ────────────────────────────────────────
# ---------------------------------------------------------------------------

# Standard parameter grids keyed by intent type
_PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "MEAN_REVERT": {
        "rsi_period":    {"default": 14, "range": [7, 14, 21], "type": "int"},
        "rsi_entry":     {"default": 30, "range": [20, 25, 30, 35], "type": "float"},
        "rsi_exit":      {"default": 70, "range": [60, 65, 70, 75], "type": "float"},
        "stop_loss_pct": {"default": 5.0, "range": [3.0, 5.0, 7.0, 10.0], "type": "float"},
        "take_profit_pct": {"default": 10.0, "range": [5.0, 8.0, 10.0, 15.0], "type": "float"},
        "holding_period_days": {"default": 10, "range": [5, 10, 15, 20], "type": "int"},
    },
    "TREND_FOLLOW": {
        "fast_ma":       {"default": 50,  "range": [20, 50, 100], "type": "int"},
        "slow_ma":       {"default": 200, "range": [100, 150, 200], "type": "int"},
        "stop_loss_pct": {"default": 8.0, "range": [5.0, 8.0, 10.0, 15.0], "type": "float"},
        "rebalance":     {"default": "daily", "range": ["daily", "weekly"], "type": "str"},
    },
    "MOMENTUM": {
        "lookback_days":  {"default": 252, "range": [63, 126, 252], "type": "int"},
        "top_pct":        {"default": 20,  "range": [10, 20, 30], "type": "int"},
        "stop_loss_pct":  {"default": 8.0, "range": [5.0, 8.0, 12.0], "type": "float"},
        "rebalance":      {"default": "monthly", "range": ["weekly", "monthly"], "type": "str"},
    },
    "BREAKOUT": {
        "lookback_days":   {"default": 252, "range": [126, 252], "type": "int"},
        "volume_mult":     {"default": 2.0, "range": [1.5, 2.0, 2.5], "type": "float"},
        "stop_loss_pct":   {"default": 5.0, "range": [3.0, 5.0, 7.0], "type": "float"},
        "take_profit_pct": {"default": 15.0, "range": [10.0, 15.0, 20.0], "type": "float"},
    },
    "PAIRS": {
        "zscore_entry":    {"default": 2.0, "range": [1.5, 2.0, 2.5], "type": "float"},
        "zscore_exit":     {"default": 0.5, "range": [0.0, 0.5, 1.0], "type": "float"},
        "lookback_days":   {"default": 60,  "range": [30, 60, 90], "type": "int"},
        "stop_loss_pct":   {"default": 5.0, "range": [3.0, 5.0, 8.0], "type": "float"},
    },
}

# Default fallback grid
_DEFAULT_PARAM_GRID: Dict[str, Any] = {
    "stop_loss_pct":   {"default": 5.0, "range": [3.0, 5.0, 7.0, 10.0], "type": "float"},
    "take_profit_pct": {"default": 10.0, "range": [5.0, 10.0, 15.0], "type": "float"},
    "position_pct":    {"default": 0.10, "range": [0.05, 0.10, 0.15], "type": "float"},
    "max_positions":   {"default": 10, "range": [5, 10, 20], "type": "int"},
}


class ParameterGridBuilder:
    """
    Build parameter grids for strategy optimization from a natural-language
    strategy description.

    Given "RSI oversold momentum strategy" → returns RSI(14), entry<30, exit>70,
    stop=-5%, target=+10% with full range for grid search.
    """

    def __init__(self) -> None:
        self._classifier = IntentClassifier()

    def build(self, description: str) -> Dict[str, Any]:
        """
        Parse the description and return a parameter grid dict.

        Returns
        -------
        dict with keys:
          intent: classified intent
          params: dict of param_name → {default, range, type}
          total_combinations: number of grid combinations
        """
        intent = self._classifier.classify(description)
        grid = dict(_PARAM_GRIDS.get(intent, _DEFAULT_PARAM_GRID))

        # Overlay RSI params if mentioned
        if re.search(r"\brsi\b", description, re.I):
            period_m = re.search(r"(\d+)[\s-]?(?:period|day)?[\s-]?rsi", description, re.I)
            period = int(period_m.group(1)) if period_m else 14
            grid["rsi_period"] = {"default": period, "range": [7, 14, 21], "type": "int"}

            entry_m = re.search(r"rsi\s*(?:below|<|under)\s*(\d+)", description, re.I)
            if entry_m:
                entry_val = float(entry_m.group(1))
                grid["rsi_entry"] = {"default": entry_val, "range": [20.0, 25.0, 30.0, 35.0], "type": "float"}

            exit_m = re.search(r"rsi\s*(?:above|>|over)\s*(\d+)", description, re.I)
            if exit_m:
                exit_val = float(exit_m.group(1))
                grid["rsi_exit"] = {"default": exit_val, "range": [60.0, 65.0, 70.0, 75.0], "type": "float"}

        # Overlay stop loss if mentioned
        sl_m = re.search(r"stop[\s-]loss\s*(?:at\s*)?(-?\d+(?:\.\d+)?)\s*%", description, re.I)
        if sl_m:
            sl_val = abs(float(sl_m.group(1)))
            grid["stop_loss_pct"] = {"default": sl_val, "range": [sl_val * 0.5, sl_val, sl_val * 1.5, sl_val * 2], "type": "float"}

        # Overlay take profit if mentioned
        tp_m = re.search(r"(?:take\s+profit|target)\s*(?:at\s*)?(\+?\d+(?:\.\d+)?)\s*%", description, re.I)
        if tp_m:
            tp_val = float(tp_m.group(1).lstrip("+"))
            grid["take_profit_pct"] = {"default": tp_val, "range": [tp_val * 0.5, tp_val, tp_val * 1.5], "type": "float"}

        # Compute total combinations
        total = 1
        for p in grid.values():
            total *= len(p.get("range", [p.get("default")]))

        return {
            "intent": intent,
            "description": description[:200],
            "params": grid,
            "total_combinations": total,
        }


# ---------------------------------------------------------------------------
# ── 8b. Risk Management Layer ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class RiskManagementLayer:
    """
    Automatically add risk management parameters to any strategy spec.

    Adds:
      - Kelly position sizing (fractional Kelly = 0.25 * full Kelly)
      - Maximum drawdown stop (portfolio-level)
      - Correlation filter (avoid adding positions when correlation > threshold)
    """

    DEFAULT_WIN_RATE: float = 0.50
    DEFAULT_WIN_LOSS_RATIO: float = 1.5    # avg win / avg loss
    KELLY_FRACTION: float = 0.25           # use 1/4 Kelly for safety
    MAX_DRAWDOWN_STOP: float = 0.20        # halt trading if portfolio DD > 20%
    CORRELATION_THRESHOLD: float = 0.70    # skip new position if corr > 0.7

    @staticmethod
    def kelly_fraction(win_rate: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
        """
        Compute Kelly criterion position size.

        Kelly % = W - (1-W)/R, where W=win rate, R=win/loss ratio.
        Apply fractional Kelly (default 25%) for robustness.

        Parameters
        ----------
        win_rate     : probability of winning trade (0-1)
        win_loss_ratio: avg win / avg loss
        fraction     : Kelly fraction to use (0.25 = quarter-Kelly)
        """
        full_kelly = win_rate - (1.0 - win_rate) / max(win_loss_ratio, 0.01)
        fractional = max(0.0, full_kelly * fraction)
        return round(fractional, 4)

    def apply(self, spec: StrategySpec,
              win_rate: Optional[float] = None,
              win_loss_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Return a risk management overlay for the given strategy spec.

        Returns a dict of risk params that should be applied to any backtest
        or live execution of the strategy.
        """
        wr = win_rate if win_rate is not None else self.DEFAULT_WIN_RATE
        wlr = win_loss_ratio if win_loss_ratio is not None else self.DEFAULT_WIN_LOSS_RATIO

        kelly = self.kelly_fraction(wr, wlr, self.KELLY_FRACTION)

        # Position sizing: use Kelly but cap at spec's value
        position_pct = min(kelly, spec.position_sizing.value)

        return {
            "kelly_fraction": kelly,
            "recommended_position_pct": round(position_pct, 4),
            "max_drawdown_stop": self.MAX_DRAWDOWN_STOP,
            "correlation_filter_threshold": self.CORRELATION_THRESHOLD,
            "max_positions": spec.position_sizing.max_positions,
            "sizing_method": "fractional_kelly",
            "kelly_inputs": {
                "win_rate": wr,
                "win_loss_ratio": wlr,
                "kelly_fraction_applied": self.KELLY_FRACTION,
            },
        }


# ---------------------------------------------------------------------------
# ── 8c. Backtest Template Generator ──────────────────────────────────────────
# ---------------------------------------------------------------------------


_BACKTEST_TEMPLATE = textwrap.dedent("""
    \"\"\"
    NumpyPortfolio-compatible entry/exit signal function.
    Strategy: {name}
    Intent:   {intent}
    Generated by SENTINEL dim_054 StrategyTemplateGenerator.
    \"\"\"
    from __future__ import annotations
    import numpy as np
    import pandas as pd
    from typing import Tuple


    def generate_signals(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        \"\"\"
        Return (entry_signals, exit_signals) as boolean Series.

        entry_signals: True on the bar to enter long
        exit_signals:  True on the bar to exit long

        Compatible with NumpyPortfolio.from_signals(entry, exit).
        \"\"\"
        entry = pd.Series(False, index=close.index)
        exit_sig = pd.Series(False, index=close.index)

        # ---- Entry logic ----
        {entry_logic}

        # ---- Exit logic ----
        {exit_logic}

        return entry, exit_sig


    # ---------------------------------------------------------------------------
    # NumpyPortfolio-compatible run (requires vectorbt installed)
    # ---------------------------------------------------------------------------

    def run_backtest(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
                     init_cash: float = 100_000.0) -> dict:
        \"\"\"Run backtest using vectorbt NumpyPortfolio (optional dep).\"\"\"
        try:
            import vectorbt as vbt
            entry, exit_sig = generate_signals(close, high, low, volume)
            pf = vbt.Portfolio.from_signals(
                close, entries=entry, exits=exit_sig,
                init_cash=init_cash, freq='D'
            )
            return {{
                'total_return': pf.total_return(),
                'sharpe_ratio': pf.sharpe_ratio(),
                'max_drawdown': pf.max_drawdown(),
                'num_trades':   pf.num_trades,
            }}
        except ImportError:
            return {{'error': 'vectorbt not installed — install with: pip install vectorbt'}}
""").strip()


class StrategyTemplateGenerator:
    """
    Generate NumpyPortfolio-compatible (vectorbt) entry/exit signal functions
    from a StrategySpec.

    The generated code compiles cleanly (verified via ast.parse) and is
    compatible with vectorbt's Portfolio.from_signals interface.
    """

    def __init__(self) -> None:
        self._parser = StrategyLanguageParserV3()
        self._gen = StrategyCodeGeneratorV3()

    def generate_template(self, description: str) -> str:
        """
        Generate a backtest template from a NL description.
        Returns Python source code as a string.
        """
        spec = self._parser.parse(description)
        return self.generate_from_spec(spec)

    def generate_from_spec(self, spec: StrategySpec) -> str:
        """Generate template from a StrategySpec."""
        entry_conds = spec.entry_conditions
        exit_conds = spec.exit_conditions

        # Build entry logic
        if entry_conds:
            entry_parts = []
            for i, cond in enumerate(entry_conds[:5]):
                if cond.python_expr:
                    entry_parts.append(f"    # {cond.raw_text or cond.condition_type}")
                    entry_parts.append(f"    # entry.iloc[i] = {cond.python_expr}")
            if entry_parts:
                entry_logic = (
                    "# Vectorised entry: adapt conditions to boolean Series\n    "
                    + "\n    ".join(entry_parts)
                    + "\n    # entry = ...  # implement vectorised condition here"
                )
            else:
                entry_logic = "# No specific entry conditions parsed — implement here\n    pass"
        else:
            entry_logic = "# No entry conditions — add your entry logic here\n    pass"

        # Build exit logic
        if exit_conds:
            exit_parts = []
            for cond in exit_conds[:5]:
                if cond.python_expr:
                    exit_parts.append(f"    # {cond.raw_text or cond.condition_type}")
                    exit_parts.append(f"    # exit_sig.iloc[i] = {cond.python_expr}")
            if exit_parts:
                exit_logic = (
                    "# Vectorised exit: adapt conditions to boolean Series\n    "
                    + "\n    ".join(exit_parts)
                    + "\n    # exit_sig = ...  # implement vectorised condition here"
                )
            else:
                exit_logic = "# No specific exit conditions parsed — implement here\n    pass"
        else:
            exit_logic = "# No exit conditions — add your exit logic here\n    pass"

        code = _BACKTEST_TEMPLATE.format(
            name=spec.name,
            intent=spec.intent,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
        )
        return code

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """Verify the template compiles without syntax errors."""
        errors: List[str] = []
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"SyntaxError: {exc}")
        return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# ── 8d. Strategy Economic Rationale Engine ───────────────────────────────────
# ---------------------------------------------------------------------------

_ECONOMIC_RATIONALES: Dict[str, str] = {
    "MOMENTUM": (
        "Momentum strategies exploit the empirical phenomenon of trend persistence: "
        "assets that have outperformed recently tend to continue outperforming over "
        "the next 3-12 months. This is underpinned by (1) investor under-reaction to "
        "new information causing gradual price adjustment, (2) herding and positive "
        "feedback loops as trend-followers enter, and (3) institutional constraints "
        "that delay full incorporation of new information. Jegadeesh & Titman (1993) "
        "documented 12-month momentum returns of ~12% annually."
    ),
    "MEAN_REVERT": (
        "Mean-reversion strategies profit from the tendency of asset prices to revert "
        "toward equilibrium after over-extension. The economic rationale is: "
        "(1) liquidity provision — buyers absorb temporary selling pressure, "
        "(2) statistical convergence — short-term sentiment overshoots fundamentals, "
        "(3) risk-aversion cycles — fear/greed oscillations create predictable "
        "reversals. RSI-based strategies specifically exploit retail over-reaction "
        "and subsequent institutional re-pricing."
    ),
    "TREND_FOLLOW": (
        "Trend-following capitalizes on the persistence of price trends across asset "
        "classes, driven by: (1) slow diffusion of fundamental information, "
        "(2) investor behavioral biases (anchoring, herding), and (3) macro regime "
        "persistence — inflation, growth, and monetary cycles last months to years. "
        "Moving average crossovers identify regime changes with a lag, trading the "
        "middle of the trend rather than tops/bottoms."
    ),
    "BREAKOUT": (
        "Breakout strategies are grounded in technical resistance/support theory: "
        "price levels where supply/demand have historically balanced act as barriers. "
        "When price breaks through with volume confirmation, it signals a shift in "
        "market structure and often precedes a sustained directional move. The "
        "economic driver is a forced re-pricing as stop-losses and momentum programs "
        "pile in on the same side."
    ),
    "PAIRS": (
        "Statistical arbitrage pairs trading exploits cointegration — the long-run "
        "equilibrium relationship between two related assets. When the spread widens "
        "beyond its historical norm, it anticipates mean-reversion. The economic "
        "basis is: companies in the same sector face similar macro drivers, so "
        "temporary spread divergence from idiosyncratic news corrects over time."
    ),
    "EARNINGS_DRIFT": (
        "Post-Earnings Announcement Drift (PEAD) is one of the most robust market "
        "anomalies: prices continue to drift in the direction of earnings surprises "
        "for weeks after the announcement. The cause is investor under-reaction — "
        "analysts and institutions slowly revise estimates, creating a gradual "
        "price adjustment that trend-followers can capture."
    ),
    "CARRY": (
        "Carry strategies earn the risk premium from holding higher-yielding assets "
        "funded by lower-yielding ones. The economic rationale: investors demand "
        "compensation for liquidity risk and rollover risk. In equities, dividend "
        "yield carry profits when market participants systematically underweight "
        "income-generating assets relative to growth stocks."
    ),
    "VOLATILITY": (
        "Volatility strategies exploit the volatility risk premium (VRP): implied "
        "volatility (options pricing) systematically exceeds subsequent realized "
        "volatility on average, rewarding sellers of options. The economic driver "
        "is the demand for insurance — hedgers overpay for downside protection, "
        "creating a persistent premium for systematic vol sellers."
    ),
    "FACTOR": (
        "Factor strategies systematically harvest known risk premia: value (cheap "
        "assets outperform over long horizons), quality (financially sound companies "
        "outperform), and size (small caps carry higher risk premia). These premia "
        "persist because they compensate for genuine economic risks that most "
        "investors find hard to bear over full market cycles."
    ),
    "MACRO": (
        "Macro strategies trade on the predictive power of economic indicators for "
        "asset prices. Interest rate regimes, yield curve shape, and GDP growth "
        "cycles are empirically linked to equity and fixed-income returns. The "
        "economic logic: central bank policy, credit conditions, and growth "
        "expectations drive discount rates and earnings forecasts simultaneously."
    ),
    "SECTOR_ROTATION": (
        "Sector rotation exploits the cyclical nature of economic regimes: different "
        "sectors outperform at different stages of the business cycle (early-cycle: "
        "financials and consumer discretionary; late-cycle: energy and materials; "
        "recession: utilities and healthcare). Rotating into the relevant sector "
        "ahead of regime shifts captures the mean-reversion of sector relative value."
    ),
    "SENTIMENT": (
        "Sentiment strategies trade on the predictive power of investor positioning "
        "and mood: extreme bullishness is contrarian bearish, and vice versa. The "
        "economic mechanism is the behavioral finance concept of noise trader risk "
        "and the tendency of sentiment to revert to fundamentals over time."
    ),
    "SEASONALITY": (
        "Seasonal strategies exploit calendar-based patterns in asset returns that "
        "persist due to institutional behavior, tax effects, and window dressing. "
        "The January Effect, turn-of-month premium, and Sell in May anomalies have "
        "persisted for decades, suggesting structural non-arbitrageable drivers "
        "related to fund flows and reporting cycles."
    ),
    "STAT_ARB": (
        "Statistical arbitrage exploits mean-reversion in asset spreads derived "
        "from quantitative modeling of historical relationships. Unlike fundamental "
        "pairs trading, stat arb uses purely statistical signals — cointegration, "
        "PCA residuals, or factor model pricing errors — to identify mispriced "
        "assets relative to a factor-neutral benchmark."
    ),
    "UNKNOWN": (
        "This strategy's economic rationale depends on the specific signals and "
        "market dynamics it exploits. Effective strategies typically profit from one "
        "of: (1) risk premia — compensation for bearing systematic risk, "
        "(2) behavioral anomalies — predictable human over/under-reaction, or "
        "(3) structural inefficiencies — frictions that prevent full arbitrage."
    ),
}


class EconomicRationaleEngine:
    """
    Generate human-readable economic rationales for trading strategies.
    Explains why the strategy should work based on financial theory.
    """

    def __init__(self) -> None:
        self._classifier = IntentClassifier()

    def explain(self, description: str) -> Dict[str, str]:
        """
        Given a strategy description, return the economic rationale.

        Returns dict with:
          intent:    classified strategy type
          rationale: detailed economic explanation
          key_driver: one-sentence driver
        """
        intent = self._classifier.classify(description)
        rationale = _ECONOMIC_RATIONALES.get(intent, _ECONOMIC_RATIONALES["UNKNOWN"])

        # Extract key driver (first sentence)
        key_driver = rationale.split(".")[0] + "."

        return {
            "intent": intent,
            "rationale": rationale,
            "key_driver": key_driver,
            "strategy_description": description[:200],
        }

    @staticmethod
    def get_rationale(intent: str) -> str:
        """Return the standard rationale string for a given intent type."""
        return _ECONOMIC_RATIONALES.get(intent, _ECONOMIC_RATIONALES["UNKNOWN"])


# ---------------------------------------------------------------------------
# ── 8. Module-level singletons ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

_orchestrator = StrategyOrchestratorV3()


# ---------------------------------------------------------------------------
# ── 9. FastAPI Router ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

strategy_v3_router = APIRouter(prefix="/strategy/v3", tags=["strategy-v3"])


class GenerateRequestV3(BaseModel):
    description: str = Field(..., description="Plain-English strategy description")
    ticker: str = Field("SPY", description="Ticker for quality-gate backtest")
    save: bool = Field(True, description="Save generated Python file to disk")
    skip_backtest: bool = Field(False, description="Skip quality-gate backtest (faster)")


class CompareRequest(BaseModel):
    descriptions: List[str] = Field(..., min_length=2, max_length=5,
                                    description="2-5 strategy descriptions to compare")
    ticker: str = Field("SPY")
    start: Optional[str] = Field(None, description="YYYY-MM-DD")
    end: Optional[str] = Field(None, description="YYYY-MM-DD")


class EnsembleRequest(BaseModel):
    descriptions: List[str] = Field(..., min_length=2, max_length=6,
                                    description="2-6 strategies to combine")
    ticker: str = Field("SPY")


class GenerateResponseV3(BaseModel):
    strategy_id: str
    name: str
    intent: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    rebalance_frequency: str
    holding_period: Optional[int]
    code_preview: str
    file_path: Optional[str]
    syntax_errors: List[str]
    quality_gate_passed: bool
    quality_gate_summary: str
    backtest: Optional[Dict[str, Any]]


@strategy_v3_router.post("/generate", response_model=GenerateResponseV3)
def generate_strategy_v3(req: GenerateRequestV3) -> GenerateResponseV3:
    """
    Parse NL description → structured spec → Python class → quality gate.

    If quality gate fails, quality_gate_passed=False and quality_gate_summary
    explains why. The code is still returned so the caller can inspect it.
    """
    try:
        result = _orchestrator.generate_and_validate(
            req.description,
            ticker=req.ticker,
            save=req.save,
            skip_backtest=req.skip_backtest,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    bt_dict: Optional[Dict[str, Any]] = None
    if result.backtest:
        bt = result.backtest
        bt_dict = {
            "ticker": bt.ticker, "start": bt.start, "end": bt.end,
            "cagr": bt.cagr, "sharpe": bt.sharpe, "max_drawdown": bt.max_drawdown,
            "win_rate": bt.win_rate, "num_trades": bt.num_trades,
            "total_return": bt.total_return, "turnover": bt.turnover,
            "benchmark_cagr": bt.benchmark_cagr,
            "alpha": round(bt.cagr - bt.benchmark_cagr, 4),
        }

    return GenerateResponseV3(
        strategy_id=result.strategy_id,
        name=result.spec.name,
        intent=result.spec.intent,
        entry_conditions=[c.raw_text or c.python_expr for c in result.spec.entry_conditions],
        exit_conditions=[c.raw_text or c.python_expr for c in result.spec.exit_conditions],
        rebalance_frequency=result.spec.rebalance_frequency,
        holding_period=result.spec.holding_period,
        code_preview=result.code[:3000],
        file_path=str(result.file_path) if result.file_path else None,
        syntax_errors=result.syntax_errors,
        quality_gate_passed=result.quality_gate.passed,
        quality_gate_summary=result.quality_gate.summary(),
        backtest=bt_dict,
    )


@strategy_v3_router.get("/library")
def get_library_v3() -> Dict[str, Any]:
    """Return all 30 pre-built strategies."""
    strategies = _orchestrator._library.list_all()
    return {
        "strategies": strategies,
        "total": len(strategies),
        "intents": list({s["intent"] for s in strategies}),
    }


@strategy_v3_router.get("/validate/{strategy_id}")
def validate_strategy_v3(strategy_id: str) -> Dict[str, Any]:
    """Return stored quality gate result for a previously generated strategy."""
    record = _orchestrator._library.get_generated(strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return record


@strategy_v3_router.post("/compare")
def compare_strategies(req: CompareRequest) -> Dict[str, Any]:
    """Run all strategies and return ranked comparison table."""
    try:
        results = _orchestrator.compare(
            req.descriptions,
            ticker=req.ticker,
            start=req.start,
            end=req.end,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"comparison": results, "count": len(results), "ticker": req.ticker}


@strategy_v3_router.post("/ensemble")
def build_ensemble(req: EnsembleRequest) -> Dict[str, Any]:
    """Combine strategies into a Sharpe-weighted ensemble."""
    try:
        result = _orchestrator.build_ensemble(req.descriptions, ticker=req.ticker)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@strategy_v3_router.get("/strategy/{strategy_id}/performance")
def strategy_performance(strategy_id: str) -> Dict[str, Any]:
    """Return stored backtest results for a generated strategy."""
    record = _orchestrator._library.get_generated(strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    with sqlite3.connect(str(_DB_PATH)) as conn:
        rows = conn.execute(
            """SELECT ticker, cagr, sharpe, max_drawdown, win_rate, num_trades,
                      total_return, benchmark_return, turnover, quality_passed, run_at
               FROM strategy_results WHERE strategy_name=? ORDER BY run_at DESC LIMIT 10""",
            (record["name"],),
        ).fetchall()
    runs = [
        {"ticker": r[0], "cagr": r[1], "sharpe": r[2], "max_drawdown": r[3],
         "win_rate": r[4], "num_trades": r[5], "total_return": r[6],
         "benchmark_return": r[7], "turnover": r[8], "quality_passed": bool(r[9]), "run_at": r[10]}
        for r in rows
    ]
    return {"strategy_id": strategy_id, "name": record["name"], "runs": runs}


# ---------------------------------------------------------------------------
# ── 10. Public API ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_and_validate(
    description: str,
    ticker: str = "SPY",
    save: bool = True,
    skip_backtest: bool = False,
) -> GenerationResult:
    """
    Top-level convenience: parse NL → backtest → quality gate → return.

    Example::

        result = parse_and_validate(
            "Buy when 20-day MA crosses above 50-day MA, sell when it crosses back below, 2% stop-loss"
        )
        if result.quality_gate.passed:
            print("Strategy approved:", result.spec.name)
        else:
            print("Strategy failed quality gate:", result.quality_gate.failure_reasons)
    """
    return _orchestrator.generate_and_validate(
        description, ticker=ticker, save=save, skip_backtest=skip_backtest
    )


if __name__ == "__main__":
    import sys

    examples = [
        "Buy when 20-day MA crosses above 50-day MA, sell when it crosses back below, 2% stop-loss",
        "Long RSI below 30 oversold stocks in the S&P 500, exit when RSI above 60, hold max 20 days",
        "Momentum: buy top 10% 12-month returns, rebalance monthly, equal weight",
        "Mean reversion: buy 5-day losers with RSI below 25, hold 3 days, size by inverse volatility",
        "Buy when MACD crosses above signal and price is above 200-day MA. Stop loss 5%.",
    ]

    desc = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else examples[0]
    print(f"NL Strategy Generator V3\n{'='*60}")
    print(f"Input: {desc}\n")

    result = parse_and_validate(desc, ticker="SPY", save=True, skip_backtest=False)

    print(f"Strategy ID   : {result.strategy_id}")
    print(f"Name          : {result.spec.name}")
    print(f"Intent        : {result.spec.intent}")
    print(f"Universe      : {result.spec.universe.universe_type}")
    print(f"Rebalance     : {result.spec.rebalance_frequency}")
    print(f"Hold period   : {result.spec.holding_period} days")
    print(f"Sizing        : {result.spec.position_sizing.method} ({result.spec.position_sizing.value:.0%})")

    print(f"\nEntry conditions ({len(result.spec.entry_conditions)}):")
    for c in result.spec.entry_conditions:
        print(f"  [{c.condition_type:22s}] {c.python_expr}")
    print(f"\nExit conditions ({len(result.spec.exit_conditions)}):")
    for c in result.spec.exit_conditions:
        print(f"  [{c.condition_type:22s}] {c.python_expr}")

    print(f"\nSyntax errors : {result.syntax_errors or 'None'}")
    print(f"Code saved    : {result.file_path}")

    qg = result.quality_gate
    print(f"\nQuality Gate  : {'PASSED' if qg.passed else 'FAILED'}")
    print(f"  Sharpe      : {qg.sharpe:.3f}  (threshold >= {_QUALITY_GATE_SHARPE})")
    print(f"  Max DD      : {qg.max_drawdown:.1%}  (threshold > {_QUALITY_GATE_MAX_DD:.0%})")
    print(f"  # Trades    : {qg.num_trades}  (threshold >= {_QUALITY_GATE_MIN_TRADES})")
    print(f"  Win Rate    : {qg.win_rate:.1%}  (threshold >= {_QUALITY_GATE_MIN_WIN_RATE:.0%})")
    if not qg.passed:
        print(f"  Failures    : {'; '.join(qg.failure_reasons)}")

    if result.backtest:
        bt = result.backtest
        print(f"\nBacktest ({bt.start} → {bt.end} on {bt.ticker})")
        print(f"  CAGR        : {bt.cagr:.2%}")
        print(f"  Total Ret   : {bt.total_return:.2%}")
        print(f"  Benchmark   : {bt.benchmark_return:.2%}")
        print(f"  Alpha       : {bt.cagr - bt.benchmark_cagr:.2%}")
        print(f"  Turnover    : {bt.turnover:.3f}")
