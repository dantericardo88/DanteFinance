"""
SIL Strategy Generator — SENTINEL.

Translates natural-language strategy descriptions into structured GeneratedStrategy
objects using Claude tool-use. Dimension 54 of the competitive matrix.
"""
from __future__ import annotations

import asyncio
from typing import Any

import anthropic
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class SignalDefinition(BaseModel):
    indicator: str           # "RSI", "SMA", "MACD", "Price", "Volume", etc.
    operator: str            # ">", "<", "crosses_above", "crosses_below", "between"
    threshold: float | None = None
    threshold_high: float | None = None  # for "between"
    lookback: int = 14
    description: str = ""


class PositionSizingSpec(BaseModel):
    method: str = "equal"    # "equal", "kelly", "vol_target", "risk_parity"
    target_vol: float = 0.15
    max_position_pct: float = 0.25
    kelly_fraction: float = 0.5


class UniverseSpec(BaseModel):
    asset_class: str = "equity"
    tickers: list[str] = []
    market_cap_min: float | None = None
    market_cap_max: float | None = None
    sectors: list[str] = []
    index: str | None = None  # "SPY", "QQQ", "IWM"


class GeneratedStrategy(BaseModel):
    name: str
    description: str
    universe: UniverseSpec
    entry_signals: list[SignalDefinition]
    exit_signals: list[SignalDefinition]
    stop_loss_pct: float = 0.08
    take_profit_pct: float | None = None
    holding_period_bars: int | None = None
    position_sizing: PositionSizingSpec
    rebalance_frequency: str = "daily"
    confidence: float
    warnings: list[str] = []


# ── Claude Tool Definitions ───────────────────────────────────────────────────

STRATEGY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "define_universe",
        "description": "Set the trading universe. Call once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_class": {"type": "string", "enum": ["equity", "etf", "crypto", "fx", "bond", "commodity"]},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "market_cap_min": {"type": "number", "description": "Min market cap in billions USD."},
                "market_cap_max": {"type": "number", "description": "Max market cap in billions USD."},
                "sectors": {"type": "array", "items": {"type": "string"}},
                "index": {"type": "string", "description": "Index ETF, e.g. 'SPY', 'QQQ', 'IWM'."},
            },
            "required": ["asset_class"],
        },
    },
    {
        "name": "add_entry_signal",
        "description": "Add one entry condition (AND-logic with other entry signals). Call multiple times.",
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string", "description": "e.g. RSI, SMA, EMA, MACD, Z_score, Volume, Price, ATR"},
                "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "crosses_above", "crosses_below", "between"]},
                "threshold": {"type": "number", "description": "Primary threshold; lower bound for 'between'."},
                "threshold_high": {"type": "number", "description": "Upper bound for 'between'. Null otherwise."},
                "lookback": {"type": "integer", "default": 14, "description": "Lookback period in bars."},
                "description": {"type": "string", "description": "What this signal captures."},
            },
            "required": ["indicator", "operator", "description"],
        },
    },
    {
        "name": "add_exit_signal",
        "description": "Add one exit condition (OR-logic). Call multiple times.",
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string"},
                "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "crosses_above", "crosses_below", "between"]},
                "threshold": {"type": "number"},
                "threshold_high": {"type": "number"},
                "lookback": {"type": "integer", "default": 14},
                "description": {"type": "string"},
            },
            "required": ["indicator", "operator", "description"],
        },
    },
    {
        "name": "set_position_sizing",
        "description": "Set position sizing method and parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["equal", "kelly", "vol_target", "risk_parity"]},
                "target_vol": {"type": "number", "default": 0.15, "description": "Annualised target vol (0-1)."},
                "max_position_pct": {"type": "number", "default": 0.25},
                "kelly_fraction": {"type": "number", "default": 0.5},
            },
            "required": ["method"],
        },
    },
    {
        "name": "set_risk_parameters",
        "description": "Set stop-loss, take-profit, holding period, rebalance frequency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "stop_loss_pct": {"type": "number", "default": 0.08},
                "take_profit_pct": {"type": "number", "description": "Null means no target."},
                "holding_period_bars": {"type": "integer", "description": "Max holding bars; null = signal-driven."},
                "rebalance_frequency": {"type": "string", "enum": ["intraday", "daily", "weekly", "monthly"], "default": "daily"},
            },
            "required": [],
        },
    },
    {
        "name": "finalize_strategy",
        "description": "Name and finalize the strategy. Call this last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short strategy name."},
                "description": {"type": "string", "description": "One-paragraph description."},
                "confidence": {"type": "number", "description": "0-1 confidence you understood the description."},
                "warnings": {"type": "array", "items": {"type": "string"}, "description": "Ambiguities or caveats."},
            },
            "required": ["name", "description", "confidence"],
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a quantitative strategy architect for SENTINEL. Translate the user's natural language "
    "strategy description into precise quantitative signals using the provided tools. Specify lookback "
    "periods explicitly (RSI-14, SMA-50, etc.). Add warnings for ambiguities. "
    "Order: define_universe → add_entry_signal (one+) → add_exit_signal (one+) → "
    "set_position_sizing → set_risk_parameters → finalize_strategy. "
    "MUST call finalize_strategy last. Use tools only — no prose."
)

# ── Rule-based Fallback ───────────────────────────────────────────────────────

_RULE_TEMPLATES: dict[str, dict[str, Any]] = {
    "mean_reversion": {
        "keywords": {"mean reversion", "mean-reversion", "z-score", "zscore"},
        "name": "Mean Reversion Z-Score",
        "entry": [SignalDefinition(indicator="Z_score", operator="<", threshold=-2.0, lookback=20, description="2+ stdev below mean — oversold.")],
        "exit":  [SignalDefinition(indicator="Z_score", operator=">", threshold=0.0,  lookback=20, description="Price reverts to mean.")],
        "stop": 0.06, "sizing": PositionSizingSpec(method="equal"),
        "desc": "Buys when Z-score < -2 (20-day window), exits at mean reversion (Z > 0).",
    },
    "rsi": {
        "keywords": {"rsi", "oversold", "overbought"},
        "name": "RSI Oscillator",
        "entry": [SignalDefinition(indicator="RSI", operator="<", threshold=30.0, lookback=14, description="RSI-14 < 30 — oversold.")],
        "exit":  [SignalDefinition(indicator="RSI", operator=">", threshold=70.0, lookback=14, description="RSI-14 > 70 — overbought.")],
        "stop": 0.07, "sizing": PositionSizingSpec(method="equal"),
        "desc": "Classic RSI mean-reversion: enter below 30, exit above 70.",
    },
    "macd": {
        "keywords": {"macd"},
        "name": "MACD Signal Cross",
        "entry": [SignalDefinition(indicator="MACD", operator="crosses_above", threshold=0.0, lookback=26, description="MACD crosses above signal — bullish.")],
        "exit":  [SignalDefinition(indicator="MACD", operator="crosses_below", threshold=0.0, lookback=26, description="MACD crosses below signal — exit.")],
        "stop": 0.08, "sizing": PositionSizingSpec(method="equal"),
        "desc": "Enters on MACD/signal bullish cross (12/26/9), exits on reverse cross.",
    },
    "momentum": {
        "keywords": {"momentum", "trend", "breakout", "golden cross"},
        "name": "Golden Cross Momentum + RSI Filter",
        "entry": [
            SignalDefinition(indicator="SMA", operator="crosses_above", threshold=200.0, lookback=50, description="SMA-50 > SMA-200 — golden cross."),
            SignalDefinition(indicator="RSI", operator=">", threshold=50.0, lookback=14, description="RSI > 50 confirms momentum."),
        ],
        "exit": [SignalDefinition(indicator="SMA", operator="crosses_below", threshold=200.0, lookback=50, description="Death cross — exit.")],
        "stop": 0.10, "sizing": PositionSizingSpec(method="vol_target", target_vol=0.12, max_position_pct=0.15),
        "desc": "Golden cross (SMA-50/200) filtered by RSI-14 > 50. Exit on death cross.",
    },
    "value": {
        "keywords": {"value", "cheap", "undervalued", "pe ratio"},
        "name": "Fundamental Value Screen",
        "entry": [SignalDefinition(indicator="PE_ratio", operator="<", threshold=15.0, lookback=1, description="P/E < 15 — classic value.")],
        "exit":  [SignalDefinition(indicator="PE_ratio", operator=">", threshold=25.0, lookback=1, description="P/E > 25 — exit at fair value.")],
        "stop": 0.12, "sizing": PositionSizingSpec(method="equal"),
        "desc": "Buys stocks with P/E < 15, exits at P/E > 25. Requires fundamental screener integration.",
        "warnings": ["PE_ratio requires SSE screener adapter wired to SBE backtest engine."],
    },
}

_DEFAULT_TEMPLATE = {
    "name": "SMA 50/200 Crossover",
    "entry": [SignalDefinition(indicator="SMA", operator="crosses_above", threshold=200.0, lookback=50, description="Golden cross.")],
    "exit":  [SignalDefinition(indicator="SMA", operator="crosses_below", threshold=200.0, lookback=50, description="Death cross.")],
    "stop": 0.10, "sizing": PositionSizingSpec(method="equal"),
    "desc": "Default dual-SMA crossover strategy. Buys on golden cross, exits on death cross.",
    "warnings": ["No strategy type matched — defaulted to SMA 50/200 crossover."],
}


def generate_strategy_rule_based(description: str) -> GeneratedStrategy:
    """Parse description keywords → concrete strategy. Never returns stubs."""
    dl = description.lower()
    template = _DEFAULT_TEMPLATE
    for key, tmpl in _RULE_TEMPLATES.items():
        if any(kw in dl for kw in tmpl["keywords"]):
            template = tmpl
            break

    warnings = ["Generated by rule-based fallback — no ANTHROPIC_API_KEY provided."]
    warnings.extend(template.get("warnings", []))

    return GeneratedStrategy(
        name=template["name"],
        description=template["desc"],
        universe=UniverseSpec(asset_class="equity", index="SPY"),
        entry_signals=template["entry"],
        exit_signals=template["exit"],
        stop_loss_pct=template["stop"],
        position_sizing=template["sizing"],
        rebalance_frequency="daily",
        confidence=0.4,
        warnings=warnings,
    )


# ── Tool-call State Accumulator ───────────────────────────────────────────────

class _StrategyState:
    """Mutable state updated as Claude calls tools."""

    def __init__(self) -> None:
        self.universe = UniverseSpec()
        self.entry_signals: list[SignalDefinition] = []
        self.exit_signals: list[SignalDefinition] = []
        self.position_sizing = PositionSizingSpec()
        self.stop_loss_pct = 0.08
        self.take_profit_pct: float | None = None
        self.holding_period_bars: int | None = None
        self.rebalance_frequency = "daily"
        self.name = "Unnamed Strategy"
        self.description = ""
        self.confidence = 0.5
        self.warnings: list[str] = []
        self.finalized = False

    def apply_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "define_universe":
            self.universe = UniverseSpec(
                asset_class=args.get("asset_class", "equity"),
                tickers=args.get("tickers") or [],
                market_cap_min=args.get("market_cap_min"),
                market_cap_max=args.get("market_cap_max"),
                sectors=args.get("sectors") or [],
                index=args.get("index"),
            )
            return f"Universe: {self.universe.asset_class}, index={self.universe.index}"

        elif name in ("add_entry_signal", "add_exit_signal"):
            sig = SignalDefinition(
                indicator=args["indicator"],
                operator=args["operator"],
                threshold=args.get("threshold"),
                threshold_high=args.get("threshold_high"),
                lookback=args.get("lookback", 14),
                description=args.get("description", ""),
            )
            if name == "add_entry_signal":
                self.entry_signals.append(sig)
            else:
                self.exit_signals.append(sig)
            return f"{name}: {sig.indicator} {sig.operator} {sig.threshold}"

        elif name == "set_position_sizing":
            self.position_sizing = PositionSizingSpec(
                method=args.get("method", "equal"),
                target_vol=args.get("target_vol", 0.15),
                max_position_pct=args.get("max_position_pct", 0.25),
                kelly_fraction=args.get("kelly_fraction", 0.5),
            )
            return f"Sizing: {self.position_sizing.method}"

        elif name == "set_risk_parameters":
            self.stop_loss_pct = args.get("stop_loss_pct", 0.08)
            self.take_profit_pct = args.get("take_profit_pct")
            self.holding_period_bars = args.get("holding_period_bars")
            self.rebalance_frequency = args.get("rebalance_frequency", "daily")
            return f"Risk: stop={self.stop_loss_pct:.1%}, rebalance={self.rebalance_frequency}"

        elif name == "finalize_strategy":
            self.name = args.get("name", "Unnamed Strategy")
            self.description = args.get("description", "")
            self.confidence = float(args.get("confidence", 0.5))
            self.warnings = args.get("warnings") or []
            self.finalized = True
            return f"Finalized: {self.name}"

        return f"Unknown tool: {name}"

    def to_strategy(self) -> GeneratedStrategy:
        return GeneratedStrategy(
            name=self.name, description=self.description,
            universe=self.universe,
            entry_signals=self.entry_signals, exit_signals=self.exit_signals,
            stop_loss_pct=self.stop_loss_pct, take_profit_pct=self.take_profit_pct,
            holding_period_bars=self.holding_period_bars,
            position_sizing=self.position_sizing,
            rebalance_frequency=self.rebalance_frequency,
            confidence=self.confidence, warnings=self.warnings,
        )


# ── Main async function ───────────────────────────────────────────────────────

async def generate_strategy(
    description: str,
    anthropic_api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> GeneratedStrategy:
    """
    Translate natural-language strategy description → GeneratedStrategy via Claude tool-use.
    Falls back to rule-based parser if no API key. Never raises.
    """
    if not anthropic_api_key:
        logger.info("strategy_generator: no API key — rule-based fallback")
        return generate_strategy_rule_based(description)

    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    state = _StrategyState()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Translate this trading strategy:\n\n{description}"}
    ]

    for iteration in range(1, 11):  # max 10 iterations
        try:
            response = await client.messages.create(
                model=model, max_tokens=2048, system=_SYSTEM_PROMPT,
                tools=STRATEGY_TOOLS,  # type: ignore[arg-type]
                tool_choice={"type": "auto"}, messages=messages,
            )
        except anthropic.APIError as exc:
            logger.error("strategy_generator: API error", error=str(exc))
            state.warnings.append(f"API error iteration {iteration}: {exc}")
            break

        assistant_content: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        has_tools = False

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                has_tools = True
                result_text = state.apply_tool(block.name, block.input)  # type: ignore[arg-type]
                logger.info("strategy_generator: tool", tool=block.name, iter=iteration)
                assistant_content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})

        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if has_tools and tool_results:
            messages.append({"role": "user", "content": tool_results})

        if state.finalized:
            logger.info("strategy_generator: finalized", name=state.name)
            break
        if not has_tools:
            state.warnings.append("Claude stopped without finalizing — strategy may be incomplete.")
            break

    # Guard: inject rule-based signals if Claude produced none
    if not state.entry_signals:
        fb = generate_strategy_rule_based(description)
        state.entry_signals = fb.entry_signals
        state.exit_signals = fb.exit_signals
        state.warnings.append("No signals from Claude — injected rule-based fallback.")
        if not state.finalized:
            state.name, state.description, state.confidence = fb.name, fb.description, 0.3

    return state.to_strategy()


def generate_strategy_sync(
    description: str,
    anthropic_api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> GeneratedStrategy:
    """Synchronous wrapper for non-async callers."""
    return asyncio.run(generate_strategy(description, anthropic_api_key, model))
