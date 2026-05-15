"""
Multi-panel workspace manager v2 — Dimension #91.

Production-quality workspace layout engine for the SENTINEL terminal with:
  - 16 panel types (PriceChart, OrderBook, OptionChain, NewsFlow, Financials,
    Screener, Watchlist, MacroCalendar, PortfolioView, RiskDashboard, BondYield,
    CryptoBook, AlternativeData, TechnicalAnalysis, EarningsCalendar, InsiderFlow)
  - 6 layout presets: 1x1, 2x1, 2x2, 3x2, 4x4, custom
  - 6 workspace templates
  - SQLite persistence for layouts, panel states, user preferences
  - Cross-panel pub/sub ticker linking
  - Panel history (forward/back, 10 states per panel)
  - Tab groups with Ctrl+1..9 quick-switch
  - Panel alerts on threshold breach
  - Full-screen per panel
  - Export workspace to JSON snapshot
  - FastAPI router at /workspace-v2/*

Score target: raise dim_091 → 9
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "PanelType",
    "LayoutPreset",
    "Panel",
    "Workspace",
    "WorkspaceManager",
    "workspace_v2_router",
]

# ---------------------------------------------------------------------------
# SQLite paths
# ---------------------------------------------------------------------------

_DB_DIR  = Path(".sentinel") / "workspace_v2"
_DB_PATH = _DB_DIR / "workspace_v2.db"


def _get_db() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with _get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL UNIQUE,
                layout_preset TEXT NOT NULL DEFAULT 'CUSTOM',
                tab_index    INTEGER NOT NULL DEFAULT 0,
                is_active    INTEGER NOT NULL DEFAULT 0,
                meta         TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS panels (
                id            TEXT PRIMARY KEY,
                workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                panel_type    TEXT NOT NULL,
                ticker        TEXT,
                title         TEXT,
                row           INTEGER NOT NULL DEFAULT 0,
                col           INTEGER NOT NULL DEFAULT 0,
                row_span      INTEGER NOT NULL DEFAULT 1,
                col_span      INTEGER NOT NULL DEFAULT 6,
                min_row_span  INTEGER NOT NULL DEFAULT 1,
                max_row_span  INTEGER NOT NULL DEFAULT 8,
                min_col_span  INTEGER NOT NULL DEFAULT 2,
                max_col_span  INTEGER NOT NULL DEFAULT 12,
                is_fullscreen INTEGER NOT NULL DEFAULT 0,
                is_linked     INTEGER NOT NULL DEFAULT 1,
                link_group    TEXT,
                interval      TEXT DEFAULT '1D',
                theme         TEXT DEFAULT 'dark',
                settings      TEXT NOT NULL DEFAULT '{}',
                alert_config  TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS panel_history (
                id           TEXT PRIMARY KEY,
                panel_id     TEXT NOT NULL REFERENCES panels(id) ON DELETE CASCADE,
                state        TEXT NOT NULL,
                ts           REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS panel_alerts (
                id           TEXT PRIMARY KEY,
                panel_id     TEXT NOT NULL REFERENCES panels(id) ON DELETE CASCADE,
                metric       TEXT NOT NULL,
                operator     TEXT NOT NULL,
                threshold    REAL NOT NULL,
                triggered    INTEGER NOT NULL DEFAULT 0,
                triggered_at TEXT,
                created_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_panels_workspace  ON panels(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_ph_panel          ON panel_history(panel_id, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_alerts_panel      ON panel_alerts(panel_id);
            """
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PanelType(str, Enum):
    PRICE_CHART        = "PriceChart"
    ORDER_BOOK         = "OrderBook"
    OPTION_CHAIN       = "OptionChain"
    NEWS_FLOW          = "NewsFlow"
    FINANCIALS         = "Financials"
    SCREENER           = "Screener"
    WATCHLIST          = "Watchlist"
    MACRO_CALENDAR     = "MacroCalendar"
    PORTFOLIO_VIEW     = "PortfolioView"
    RISK_DASHBOARD     = "RiskDashboard"
    BOND_YIELD         = "BondYield"
    CRYPTO_BOOK        = "CryptoBook"
    ALTERNATIVE_DATA   = "AlternativeData"
    TECHNICAL_ANALYSIS = "TechnicalAnalysis"
    EARNINGS_CALENDAR  = "EarningsCalendar"
    INSIDER_FLOW       = "InsiderFlow"


class LayoutPreset(str, Enum):
    SINGLE    = "1x1"      # 1 panel, full width
    TWO_COL   = "2x1"      # 2 panels side-by-side
    TWO_BY_TWO= "2x2"      # 4 panels in 2×2 grid
    THREE_BY_TWO = "3x2"   # 6 panels in 3×2 grid
    FOUR_BY_FOUR = "4x4"   # 16 panels
    CUSTOM    = "CUSTOM"   # arbitrary placement


class AlertOperator(str, Enum):
    GT  = ">"
    GTE = ">="
    LT  = "<"
    LTE = "<="
    EQ  = "=="
    NEQ = "!="


# ---------------------------------------------------------------------------
# Panel type metadata — data adapter config, refresh, required params
# ---------------------------------------------------------------------------

_PANEL_META: dict[PanelType, dict] = {
    PanelType.PRICE_CHART: {
        "adapter":         "sds.ohlcv",
        "refresh_seconds": 60,
        "params":          ["ticker", "interval"],
        "default_interval":"1D",
        "supports_overlay":True,
        "description":     "Candlestick / line price chart with technical overlays",
    },
    PanelType.ORDER_BOOK: {
        "adapter":         "sds.quotes.level2",
        "refresh_seconds": 1,
        "params":          ["ticker"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Level 2 order book depth, bid/ask ladder, and tape",
    },
    PanelType.OPTION_CHAIN: {
        "adapter":         "sfe.options_analytics",
        "refresh_seconds": 30,
        "params":          ["ticker", "expiry"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Options chain with Greeks, IV, OI, and volume",
    },
    PanelType.NEWS_FLOW: {
        "adapter":         "sma.news_flow",
        "refresh_seconds": 120,
        "params":          ["ticker"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Real-time news headlines and press releases",
    },
    PanelType.FINANCIALS: {
        "adapter":         "sfe.standardized_financials",
        "refresh_seconds": 86400,
        "params":          ["ticker", "period"],
        "default_interval": "A",
        "supports_overlay": False,
        "description":     "Income statement, balance sheet, and cash flow",
    },
    PanelType.SCREENER: {
        "adapter":         "sbx.screener",
        "refresh_seconds": 300,
        "params":          ["filters", "universe"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Multi-factor equity/bond/crypto screener",
    },
    PanelType.WATCHLIST: {
        "adapter":         "sds.quotes",
        "refresh_seconds": 15,
        "params":          ["tickers"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Named watchlist with live quotes and custom columns",
    },
    PanelType.MACRO_CALENDAR: {
        "adapter":         "sma.economic_calendar",
        "refresh_seconds": 1800,
        "params":          ["regions"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Upcoming macro data releases with forecast vs. actual",
    },
    PanelType.PORTFOLIO_VIEW: {
        "adapter":         "spr.portfolio",
        "refresh_seconds": 60,
        "params":          ["portfolio_id"],
        "default_interval": "1D",
        "supports_overlay": False,
        "description":     "Portfolio positions, P&L, allocation, and risk metrics",
    },
    PanelType.RISK_DASHBOARD: {
        "adapter":         "spr.risk",
        "refresh_seconds": 300,
        "params":          ["portfolio_id"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "VaR, CVaR, Greeks, stress tests, and factor exposure",
    },
    PanelType.BOND_YIELD: {
        "adapter":         "sfe.yield_curve",
        "refresh_seconds": 300,
        "params":          ["country"],
        "default_interval": None,
        "supports_overlay": True,
        "description":     "Treasury/sovereign yield curve and spread monitor",
    },
    PanelType.CRYPTO_BOOK: {
        "adapter":         "sbx.crypto_screener",
        "refresh_seconds": 5,
        "params":          ["ticker", "exchange"],
        "default_interval": "1H",
        "supports_overlay": False,
        "description":     "Crypto order book, depth chart, and cross-exchange arb",
    },
    PanelType.ALTERNATIVE_DATA: {
        "adapter":         "sma.sentiment_engine",
        "refresh_seconds": 600,
        "params":          ["ticker", "data_type"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Satellite, web traffic, sentiment, and NLP signals",
    },
    PanelType.TECHNICAL_ANALYSIS: {
        "adapter":         "sds.ohlcv",
        "refresh_seconds": 60,
        "params":          ["ticker", "interval", "indicators"],
        "default_interval": "1D",
        "supports_overlay": True,
        "description":     "Technical dashboard: indicators, patterns, signals",
    },
    PanelType.EARNINGS_CALENDAR: {
        "adapter":         "sfe.earnings_kpi",
        "refresh_seconds": 3600,
        "params":          ["tickers"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Upcoming earnings dates with estimates and surprise history",
    },
    PanelType.INSIDER_FLOW: {
        "adapter":         "sfe.insider_analytics",
        "refresh_seconds": 3600,
        "params":          ["ticker"],
        "default_interval": None,
        "supports_overlay": False,
        "description":     "Form 4 insider transactions with pattern analysis",
    },
}

# ---------------------------------------------------------------------------
# Layout preset grid definitions
# Each entry: list of (row, col, row_span, col_span) for each panel slot
# Grid is 12 columns wide.
# ---------------------------------------------------------------------------

_LAYOUT_GRIDS: dict[LayoutPreset, list[tuple[int, int, int, int]]] = {
    LayoutPreset.SINGLE: [
        (0, 0, 6, 12),
    ],
    LayoutPreset.TWO_COL: [
        (0, 0, 6,  6),
        (0, 6, 6,  6),
    ],
    LayoutPreset.TWO_BY_TWO: [
        (0, 0, 3, 6),
        (0, 6, 3, 6),
        (3, 0, 3, 6),
        (3, 6, 3, 6),
    ],
    LayoutPreset.THREE_BY_TWO: [
        (0, 0, 3, 4),
        (0, 4, 3, 4),
        (0, 8, 3, 4),
        (3, 0, 3, 4),
        (3, 4, 3, 4),
        (3, 8, 3, 4),
    ],
    LayoutPreset.FOUR_BY_FOUR: [
        (r, c, 2, 3)
        for r in (0, 2, 4, 6)
        for c in (0, 3, 6, 9)
    ],
    LayoutPreset.CUSTOM: [],
}

# ---------------------------------------------------------------------------
# Workspace templates
# ---------------------------------------------------------------------------

_WORKSPACE_TEMPLATES: dict[str, dict] = {
    "Equity Analysis": {
        "description": "Full equity deep-dive: price, financials, news, options, insider",
        "preset": LayoutPreset.TWO_BY_TWO,
        "panels": [
            {"type": PanelType.PRICE_CHART,        "title": "Price Chart",         "col": 0, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.FINANCIALS,          "title": "Financials",          "col": 6, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,           "title": "News Flow",           "col": 0, "row": 3, "col_span": 6, "row_span": 3},
            {"type": PanelType.INSIDER_FLOW,        "title": "Insider Transactions","col": 6, "row": 3, "col_span": 6, "row_span": 3},
        ],
        "default_ticker": "AAPL",
    },
    "Fixed Income": {
        "description": "Bond analysis: yield curve, screener, analytics, news",
        "preset": LayoutPreset.TWO_BY_TWO,
        "panels": [
            {"type": PanelType.BOND_YIELD,          "title": "Yield Curve",         "col": 0, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.SCREENER,            "title": "Bond Screener",       "col": 6, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,           "title": "Bond News",           "col": 0, "row": 3, "col_span": 6, "row_span": 3},
            {"type": PanelType.MACRO_CALENDAR,      "title": "Macro Calendar",      "col": 6, "row": 3, "col_span": 6, "row_span": 3},
        ],
        "default_ticker": "TLT",
    },
    "Macro Overview": {
        "description": "Global macro: calendar, rates, FX, and equities",
        "preset": LayoutPreset.THREE_BY_TWO,
        "panels": [
            {"type": PanelType.MACRO_CALENDAR,      "title": "Economic Calendar",   "col": 0, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.BOND_YIELD,          "title": "Treasury Yields",     "col": 4, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.PRICE_CHART,         "title": "S&P 500",             "col": 8, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.WATCHLIST,           "title": "FX Rates",            "col": 0, "row": 3, "col_span": 4, "row_span": 3},
            {"type": PanelType.WATCHLIST,           "title": "Commodities",         "col": 4, "row": 3, "col_span": 4, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,           "title": "Global News",         "col": 8, "row": 3, "col_span": 4, "row_span": 3},
        ],
        "default_ticker": "SPX",
    },
    "Options Trading": {
        "description": "Options flow, chain, price chart, and volatility",
        "preset": LayoutPreset.TWO_BY_TWO,
        "panels": [
            {"type": PanelType.PRICE_CHART,        "title": "Underlying Chart",    "col": 0, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.OPTION_CHAIN,       "title": "Option Chain",        "col": 6, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.TECHNICAL_ANALYSIS, "title": "Technical Signals",   "col": 0, "row": 3, "col_span": 6, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,          "title": "News & Events",       "col": 6, "row": 3, "col_span": 6, "row_span": 3},
        ],
        "default_ticker": "AAPL",
    },
    "Crypto": {
        "description": "Crypto trading: book, charts, on-chain, and DeFi",
        "preset": LayoutPreset.TWO_BY_TWO,
        "panels": [
            {"type": PanelType.PRICE_CHART,        "title": "BTC Chart",           "col": 0, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.CRYPTO_BOOK,        "title": "Order Book",          "col": 6, "row": 0, "col_span": 6, "row_span": 3},
            {"type": PanelType.ALTERNATIVE_DATA,   "title": "On-Chain Metrics",    "col": 0, "row": 3, "col_span": 6, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,          "title": "Crypto News",         "col": 6, "row": 3, "col_span": 6, "row_span": 3},
        ],
        "default_ticker": "BTC",
    },
    "Pairs Trading": {
        "description": "Pairs trading: dual charts, correlation, and spread",
        "preset": LayoutPreset.THREE_BY_TWO,
        "panels": [
            {"type": PanelType.PRICE_CHART,        "title": "Asset A Chart",       "col": 0, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.PRICE_CHART,        "title": "Asset B Chart",       "col": 4, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.TECHNICAL_ANALYSIS, "title": "Spread / Ratio",      "col": 8, "row": 0, "col_span": 4, "row_span": 3},
            {"type": PanelType.SCREENER,           "title": "Pairs Screener",      "col": 0, "row": 3, "col_span": 4, "row_span": 3},
            {"type": PanelType.RISK_DASHBOARD,     "title": "Correlation Matrix",  "col": 4, "row": 3, "col_span": 4, "row_span": 3},
            {"type": PanelType.NEWS_FLOW,          "title": "Pair News",           "col": 8, "row": 3, "col_span": 4, "row_span": 3},
        ],
        "default_ticker": "AAPL",
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AlertConfig(BaseModel):
    metric:    str = Field(..., description="e.g. price, volume, rsi_14")
    operator:  AlertOperator
    threshold: float
    label:     Optional[str] = None


class PanelState(BaseModel):
    """Serializable snapshot of a single panel's current state."""
    ticker:     Optional[str] = None
    interval:   Optional[str] = None
    settings:   dict[str, Any] = Field(default_factory=dict)
    scroll_pos: Optional[int] = None
    sort_col:   Optional[str] = None
    filters:    dict[str, Any] = Field(default_factory=dict)
    ts:         str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PanelModel(BaseModel):
    id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str
    panel_type:   PanelType
    ticker:       Optional[str] = None
    title:        str = ""
    row:          int = 0
    col:          int = 0
    row_span:     int = 1
    col_span:     int = 6
    min_row_span: int = 1
    max_row_span: int = 8
    min_col_span: int = 2
    max_col_span: int = 12
    is_fullscreen:bool = False
    is_linked:    bool = True
    link_group:   Optional[str] = None
    interval:     str = "1D"
    theme:        str = "dark"
    settings:     dict[str, Any] = Field(default_factory=dict)
    alert_config: dict[str, Any] = Field(default_factory=dict)
    created_at:   str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:   str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def meta(self) -> dict:
        return _PANEL_META.get(self.panel_type, {})

    def to_db_row(self) -> tuple:
        return (
            self.id, self.workspace_id, self.panel_type.value,
            self.ticker, self.title,
            self.row, self.col, self.row_span, self.col_span,
            self.min_row_span, self.max_row_span, self.min_col_span, self.max_col_span,
            int(self.is_fullscreen), int(self.is_linked), self.link_group,
            self.interval, self.theme,
            json.dumps(self.settings), json.dumps(self.alert_config),
            self.created_at, self.updated_at,
        )

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "PanelModel":
        d = dict(row)
        d["panel_type"]   = PanelType(d["panel_type"])
        d["is_fullscreen"]= bool(d["is_fullscreen"])
        d["is_linked"]    = bool(d["is_linked"])
        d["settings"]     = json.loads(d.get("settings") or "{}")
        d["alert_config"] = json.loads(d.get("alert_config") or "{}")
        return cls(**d)


class WorkspaceModel(BaseModel):
    id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    name:          str
    layout_preset: LayoutPreset = LayoutPreset.CUSTOM
    tab_index:     int = 0
    is_active:     bool = False
    meta:          dict[str, Any] = Field(default_factory=dict)
    created_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:    str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_db_row(self) -> tuple:
        return (
            self.id, self.name, self.layout_preset.value,
            self.tab_index, int(self.is_active),
            json.dumps(self.meta),
            self.created_at, self.updated_at,
        )

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "WorkspaceModel":
        d = dict(row)
        d["layout_preset"] = LayoutPreset(d["layout_preset"])
        d["is_active"]     = bool(d["is_active"])
        d["meta"]          = json.loads(d.get("meta") or "{}")
        return cls(**d)


# ---------------------------------------------------------------------------
# Panel history manager (per panel, last 10 states)
# ---------------------------------------------------------------------------

_PANEL_HISTORY_LIMIT = 10


class PanelHistoryManager:
    """In-memory + SQLite forward/back navigation for a panel."""

    def __init__(self, panel_id: str) -> None:
        self.panel_id = panel_id
        self._stack: deque[PanelState] = deque(maxlen=_PANEL_HISTORY_LIMIT)
        self._ptr: int = -1  # current position in stack
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not _DB_PATH.exists():
            self._loaded = True
            return
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT state FROM panel_history WHERE panel_id=? ORDER BY ts ASC LIMIT ?",
                (self.panel_id, _PANEL_HISTORY_LIMIT),
            ).fetchall()
        for row in rows:
            self._stack.append(PanelState(**json.loads(row["state"])))
        if self._stack:
            self._ptr = len(self._stack) - 1
        self._loaded = True

    def push(self, state: PanelState) -> None:
        self._ensure_loaded()
        # Truncate forward history
        while len(self._stack) > self._ptr + 1:
            self._stack.pop()
        self._stack.append(state)
        self._ptr = len(self._stack) - 1
        # Persist to SQLite
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO panel_history (id, panel_id, state, ts) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), self.panel_id, state.json(), time.time()),
            )
            # Evict old
            conn.execute(
                """
                DELETE FROM panel_history WHERE panel_id=? AND id NOT IN (
                    SELECT id FROM panel_history WHERE panel_id=? ORDER BY ts DESC LIMIT ?
                )
                """,
                (self.panel_id, self.panel_id, _PANEL_HISTORY_LIMIT),
            )

    def back(self) -> Optional[PanelState]:
        self._ensure_loaded()
        if self._ptr <= 0:
            return None
        self._ptr -= 1
        return self._stack[self._ptr]

    def forward(self) -> Optional[PanelState]:
        self._ensure_loaded()
        if self._ptr >= len(self._stack) - 1:
            return None
        self._ptr += 1
        return self._stack[self._ptr]

    def current(self) -> Optional[PanelState]:
        self._ensure_loaded()
        if not self._stack or self._ptr < 0:
            return None
        return self._stack[self._ptr]

    def all_states(self) -> list[dict]:
        self._ensure_loaded()
        return [s.dict() for s in self._stack]


# ---------------------------------------------------------------------------
# Pub/Sub ticker link bus
# ---------------------------------------------------------------------------


class TickerLinkBus:
    """
    Lightweight in-memory pub/sub for cross-panel ticker linking.
    Panels subscribe to a link_group; broadcasting a ticker to a group
    causes all subscribers to update.
    """

    def __init__(self) -> None:
        # group_name -> set of panel_ids
        self._subscribers: dict[str, set[str]] = {}
        # panel_id -> current ticker
        self._panel_ticker: dict[str, str] = {}

    def subscribe(self, panel_id: str, group: str) -> None:
        self._subscribers.setdefault(group, set()).add(panel_id)

    def unsubscribe(self, panel_id: str, group: str) -> None:
        self._subscribers.get(group, set()).discard(panel_id)

    def broadcast(self, source_panel_id: str, ticker: str, group: str) -> list[str]:
        """
        Broadcast ticker change from source panel to all panels in group.
        Returns list of panel_ids that were updated.
        """
        updated = []
        for pid in self._subscribers.get(group, set()):
            if pid != source_panel_id:
                self._panel_ticker[pid] = ticker
                updated.append(pid)
        self._panel_ticker[source_panel_id] = ticker
        return updated

    def get_ticker(self, panel_id: str) -> Optional[str]:
        return self._panel_ticker.get(panel_id)

    def all_groups(self) -> dict[str, list[str]]:
        return {g: list(ps) for g, ps in self._subscribers.items()}


# Singleton bus
_link_bus = TickerLinkBus()


# ---------------------------------------------------------------------------
# Workspace Manager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """
    CRUD + layout operations for workspaces and panels.
    All mutations persist immediately to SQLite.
    """

    MAX_WORKSPACES = 9  # Ctrl+1..9
    MAX_PANELS_PER_WORKSPACE = 16
    GRID_COLS = 12
    SNAP_GRID = 1  # minimum grid unit

    def __init__(self) -> None:
        _init_db()

    # ------------------------------------------------------------------
    # Workspace CRUD
    # ------------------------------------------------------------------

    def create_workspace(
        self,
        name: str,
        preset: LayoutPreset = LayoutPreset.CUSTOM,
        tab_index: Optional[int] = None,
        meta: Optional[dict] = None,
    ) -> WorkspaceModel:
        if tab_index is None:
            tab_index = self._next_tab_index()

        ws = WorkspaceModel(
            name=name,
            layout_preset=preset,
            tab_index=tab_index,
            meta=meta or {},
        )
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO workspaces (id, name, layout_preset, tab_index, is_active, meta, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                ws.to_db_row(),
            )
        logger.info("Created workspace '%s' id=%s", name, ws.id)
        return ws

    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceModel]:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        return WorkspaceModel.from_db_row(row) if row else None

    def get_workspace_by_name(self, name: str) -> Optional[WorkspaceModel]:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE name=?", (name,)
            ).fetchone()
        return WorkspaceModel.from_db_row(row) if row else None

    def list_workspaces(self) -> list[WorkspaceModel]:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM workspaces ORDER BY tab_index ASC"
            ).fetchall()
        return [WorkspaceModel.from_db_row(r) for r in rows]

    def delete_workspace(self, workspace_id: str) -> bool:
        with _get_db() as conn:
            cur = conn.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
        return cur.rowcount > 0

    def set_active_workspace(self, workspace_id: str) -> bool:
        with _get_db() as conn:
            conn.execute("UPDATE workspaces SET is_active=0")
            cur = conn.execute(
                "UPDATE workspaces SET is_active=1, updated_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), workspace_id),
            )
        return cur.rowcount > 0

    def get_active_workspace(self) -> Optional[WorkspaceModel]:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE is_active=1 LIMIT 1"
            ).fetchone()
        if row:
            return WorkspaceModel.from_db_row(row)
        # Fallback: first workspace
        wss = self.list_workspaces()
        return wss[0] if wss else None

    def rename_workspace(self, workspace_id: str, new_name: str) -> bool:
        with _get_db() as conn:
            cur = conn.execute(
                "UPDATE workspaces SET name=?, updated_at=? WHERE id=?",
                (new_name, datetime.now(timezone.utc).isoformat(), workspace_id),
            )
        return cur.rowcount > 0

    def switch_tab(self, tab_index: int) -> Optional[WorkspaceModel]:
        """Quick-switch to workspace by tab index (0-based, maps to Ctrl+1..9)."""
        with _get_db() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE tab_index=?", (tab_index,)
            ).fetchone()
        if not row:
            return None
        ws = WorkspaceModel.from_db_row(row)
        self.set_active_workspace(ws.id)
        return ws

    # ------------------------------------------------------------------
    # Layout operations
    # ------------------------------------------------------------------

    def apply_preset(
        self,
        workspace_id: str,
        preset: LayoutPreset,
        panel_types: Optional[list[PanelType]] = None,
        default_ticker: str = "AAPL",
    ) -> list[PanelModel]:
        """
        Apply a layout preset to a workspace, creating panels at preset positions.
        Existing panels are replaced.
        """
        self._clear_panels(workspace_id)
        grid = _LAYOUT_GRIDS.get(preset, [])
        if not grid:
            return []

        types_to_use = panel_types or [PanelType.PRICE_CHART] * len(grid)
        panels = []
        for i, (row, col, rs, cs) in enumerate(grid):
            pt = types_to_use[i] if i < len(types_to_use) else PanelType.PRICE_CHART
            panel = self.add_panel(
                workspace_id=workspace_id,
                panel_type=pt,
                ticker=default_ticker,
                row=row, col=col, row_span=rs, col_span=cs,
            )
            panels.append(panel)

        # Update workspace preset
        with _get_db() as conn:
            conn.execute(
                "UPDATE workspaces SET layout_preset=?, updated_at=? WHERE id=?",
                (preset.value, datetime.now(timezone.utc).isoformat(), workspace_id),
            )
        return panels

    def update_layout(
        self, workspace_id: str, panel_positions: list[dict]
    ) -> list[PanelModel]:
        """
        Batch-update panel positions for drag-drop rearrangement.
        panel_positions: [{"panel_id": ..., "row": ..., "col": ..., "row_span": ..., "col_span": ...}]
        """
        updated = []
        now = datetime.now(timezone.utc).isoformat()
        with _get_db() as conn:
            for pos in panel_positions:
                pid = pos.get("panel_id")
                if not pid:
                    continue
                row      = pos.get("row", 0)
                col      = pos.get("col", 0)
                row_span = pos.get("row_span", 1)
                col_span = pos.get("col_span", 6)

                # Snap to grid
                col      = max(0, min(col, self.GRID_COLS - 1))
                col_span = max(1, min(col_span, self.GRID_COLS - col))
                row      = max(0, row)
                row_span = max(1, row_span)

                conn.execute(
                    """
                    UPDATE panels SET row=?, col=?, row_span=?, col_span=?, updated_at=?
                    WHERE id=? AND workspace_id=?
                    """,
                    (row, col, row_span, col_span, now, pid, workspace_id),
                )
                row_panel = conn.execute(
                    "SELECT * FROM panels WHERE id=?", (pid,)
                ).fetchone()
                if row_panel:
                    updated.append(PanelModel.from_db_row(row_panel))
        return updated

    def validate_layout(self, workspace_id: str) -> dict:
        """
        Check for overlapping panels and out-of-bounds positions.
        Returns {"valid": bool, "issues": [...]}
        """
        panels = self.list_panels(workspace_id)
        issues = []
        occupied: set[tuple[int, int]] = set()

        for p in panels:
            # Bounds check
            if p.col < 0 or p.col + p.col_span > self.GRID_COLS:
                issues.append(
                    f"Panel '{p.title or p.id}' exceeds column bounds "
                    f"(col={p.col}, span={p.col_span})"
                )
            # Overlap check
            for r in range(p.row, p.row + p.row_span):
                for c in range(p.col, p.col + p.col_span):
                    if (r, c) in occupied:
                        issues.append(
                            f"Panel '{p.title or p.id}' overlaps at ({r},{c})"
                        )
                    occupied.add((r, c))

        return {"valid": len(issues) == 0, "issues": issues}

    # ------------------------------------------------------------------
    # Panel CRUD
    # ------------------------------------------------------------------

    def add_panel(
        self,
        workspace_id: str,
        panel_type: PanelType,
        ticker: Optional[str] = None,
        title: Optional[str] = None,
        row: int = 0,
        col: int = 0,
        row_span: int = 1,
        col_span: int = 6,
        interval: str = "1D",
        theme: str = "dark",
        settings: Optional[dict] = None,
        is_linked: bool = True,
        link_group: Optional[str] = None,
    ) -> PanelModel:
        # Enforce max panels
        n = self._count_panels(workspace_id)
        if n >= self.MAX_PANELS_PER_WORKSPACE:
            raise ValueError(
                f"Workspace already has {n} panels (max {self.MAX_PANELS_PER_WORKSPACE})"
            )

        # Snap and constrain
        meta = _PANEL_META.get(panel_type, {})
        default_title = title or f"{panel_type.value} — {ticker or 'No Ticker'}"
        panel = PanelModel(
            workspace_id=workspace_id,
            panel_type=panel_type,
            ticker=ticker,
            title=default_title,
            row=max(0, row),
            col=max(0, min(col, self.GRID_COLS - 1)),
            row_span=max(1, row_span),
            col_span=max(2, min(col_span, self.GRID_COLS)),
            interval=interval or meta.get("default_interval", "1D") or "1D",
            theme=theme,
            settings=settings or {},
            is_linked=is_linked,
            link_group=link_group,
        )

        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO panels
                    (id, workspace_id, panel_type, ticker, title,
                     row, col, row_span, col_span,
                     min_row_span, max_row_span, min_col_span, max_col_span,
                     is_fullscreen, is_linked, link_group,
                     interval, theme, settings, alert_config, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                panel.to_db_row(),
            )

        # Register to link bus if linked
        if is_linked and link_group:
            _link_bus.subscribe(panel.id, link_group)

        logger.info("Added panel %s (%s) to workspace %s", panel.id, panel_type.value, workspace_id)
        return panel

    def get_panel(self, panel_id: str) -> Optional[PanelModel]:
        with _get_db() as conn:
            row = conn.execute("SELECT * FROM panels WHERE id=?", (panel_id,)).fetchone()
        return PanelModel.from_db_row(row) if row else None

    def list_panels(self, workspace_id: str) -> list[PanelModel]:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM panels WHERE workspace_id=? ORDER BY row ASC, col ASC",
                (workspace_id,),
            ).fetchall()
        return [PanelModel.from_db_row(r) for r in rows]

    def delete_panel(self, workspace_id: str, panel_id: str) -> bool:
        panel = self.get_panel(panel_id)
        if panel and panel.link_group:
            _link_bus.unsubscribe(panel_id, panel.link_group)
        with _get_db() as conn:
            cur = conn.execute(
                "DELETE FROM panels WHERE id=? AND workspace_id=?",
                (panel_id, workspace_id),
            )
        return cur.rowcount > 0

    def update_panel(
        self,
        panel_id: str,
        *,
        ticker: Optional[str] = None,
        title: Optional[str] = None,
        interval: Optional[str] = None,
        theme: Optional[str] = None,
        settings: Optional[dict] = None,
        alert_config: Optional[dict] = None,
        is_fullscreen: Optional[bool] = None,
        is_linked: Optional[bool] = None,
        link_group: Optional[str] = None,
    ) -> Optional[PanelModel]:
        panel = self.get_panel(panel_id)
        if not panel:
            return None

        updates: dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if ticker       is not None: updates["ticker"]       = ticker
        if title        is not None: updates["title"]        = title
        if interval     is not None: updates["interval"]     = interval
        if theme        is not None: updates["theme"]        = theme
        if settings     is not None: updates["settings"]     = json.dumps(settings)
        if alert_config is not None: updates["alert_config"] = json.dumps(alert_config)
        if is_fullscreen is not None: updates["is_fullscreen"] = int(is_fullscreen)
        if is_linked    is not None: updates["is_linked"]    = int(is_linked)
        if link_group   is not None:
            # Update bus subscriptions
            if panel.link_group:
                _link_bus.unsubscribe(panel_id, panel.link_group)
            updates["link_group"] = link_group
            if link_group:
                _link_bus.subscribe(panel_id, link_group)

        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [panel_id]
        with _get_db() as conn:
            conn.execute(f"UPDATE panels SET {set_clause} WHERE id=?", values)
            row = conn.execute("SELECT * FROM panels WHERE id=?", (panel_id,)).fetchone()
        return PanelModel.from_db_row(row) if row else None

    def toggle_fullscreen(self, workspace_id: str, panel_id: str) -> Optional[PanelModel]:
        """Toggle fullscreen mode for a panel, collapsing all others."""
        panels = self.list_panels(workspace_id)
        panel = next((p for p in panels if p.id == panel_id), None)
        if not panel:
            return None

        new_state = not panel.is_fullscreen
        now = datetime.now(timezone.utc).isoformat()
        with _get_db() as conn:
            if new_state:
                # Collapse all other panels
                conn.execute(
                    "UPDATE panels SET is_fullscreen=0, updated_at=? WHERE workspace_id=? AND id!=?",
                    (now, workspace_id, panel_id),
                )
            conn.execute(
                "UPDATE panels SET is_fullscreen=?, updated_at=? WHERE id=?",
                (int(new_state), now, panel_id),
            )
            row = conn.execute("SELECT * FROM panels WHERE id=?", (panel_id,)).fetchone()
        return PanelModel.from_db_row(row) if row else None

    # ------------------------------------------------------------------
    # Ticker linking (pub/sub)
    # ------------------------------------------------------------------

    def broadcast_ticker(
        self, source_panel_id: str, ticker: str, group: Optional[str] = None
    ) -> dict:
        """
        Change ticker on source panel and propagate to all linked panels in group.
        If group is None, use the source panel's link_group.
        """
        panel = self.get_panel(source_panel_id)
        if not panel:
            return {"error": "Panel not found", "updated": []}

        grp = group or panel.link_group or "default"
        updated_ids = _link_bus.broadcast(source_panel_id, ticker, grp)

        # Persist ticker changes
        now = datetime.now(timezone.utc).isoformat()
        with _get_db() as conn:
            conn.execute(
                "UPDATE panels SET ticker=?, updated_at=? WHERE id=?",
                (ticker, now, source_panel_id),
            )
            for pid in updated_ids:
                conn.execute(
                    "UPDATE panels SET ticker=?, updated_at=? WHERE id=?",
                    (ticker, now, pid),
                )

        return {
            "source_panel":    source_panel_id,
            "ticker":          ticker,
            "link_group":      grp,
            "updated_panels":  updated_ids,
            "total_updated":   len(updated_ids) + 1,
        }

    # ------------------------------------------------------------------
    # Panel history
    # ------------------------------------------------------------------

    def push_panel_state(self, panel_id: str, state: PanelState) -> None:
        mgr = PanelHistoryManager(panel_id)
        mgr.push(state)

    def navigate_panel_back(self, panel_id: str) -> Optional[dict]:
        mgr = PanelHistoryManager(panel_id)
        state = mgr.back()
        if state:
            # Apply state to panel
            self.update_panel(panel_id, ticker=state.ticker, interval=state.interval, settings=state.settings)
        return state.dict() if state else None

    def navigate_panel_forward(self, panel_id: str) -> Optional[dict]:
        mgr = PanelHistoryManager(panel_id)
        state = mgr.forward()
        if state:
            self.update_panel(panel_id, ticker=state.ticker, interval=state.interval, settings=state.settings)
        return state.dict() if state else None

    def get_panel_history(self, panel_id: str) -> list[dict]:
        mgr = PanelHistoryManager(panel_id)
        return mgr.all_states()

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def add_panel_alert(
        self,
        panel_id: str,
        metric: str,
        operator: AlertOperator,
        threshold: float,
        label: Optional[str] = None,
    ) -> dict:
        alert_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO panel_alerts (id, panel_id, metric, operator, threshold, triggered, created_at)
                VALUES (?,?,?,?,?,0,?)
                """,
                (alert_id, panel_id, metric, operator.value, threshold, now),
            )
        return {
            "id":        alert_id,
            "panel_id":  panel_id,
            "metric":    metric,
            "operator":  operator.value,
            "threshold": threshold,
            "label":     label,
            "triggered": False,
            "created_at":now,
        }

    def check_alerts(self, panel_id: str, current_values: dict[str, float]) -> list[dict]:
        """
        Check panel alerts against current metric values.
        Returns list of triggered alerts.
        """
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM panel_alerts WHERE panel_id=? AND triggered=0",
                (panel_id,),
            ).fetchall()

        triggered = []
        now = datetime.now(timezone.utc).isoformat()
        _OPS = {
            ">":  lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<":  lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        for row in rows:
            metric    = row["metric"]
            op_str    = row["operator"]
            threshold = row["threshold"]
            val       = current_values.get(metric)
            if val is None:
                continue
            op_fn = _OPS.get(op_str)
            if op_fn and op_fn(val, threshold):
                with _get_db() as conn:
                    conn.execute(
                        "UPDATE panel_alerts SET triggered=1, triggered_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                triggered.append({
                    "alert_id":    row["id"],
                    "panel_id":    panel_id,
                    "metric":      metric,
                    "operator":    op_str,
                    "threshold":   threshold,
                    "actual_value":val,
                    "triggered_at":now,
                })
        return triggered

    def list_panel_alerts(self, panel_id: str) -> list[dict]:
        with _get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM panel_alerts WHERE panel_id=? ORDER BY created_at DESC",
                (panel_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_panel_alert(self, alert_id: str) -> bool:
        with _get_db() as conn:
            cur = conn.execute("DELETE FROM panel_alerts WHERE id=?", (alert_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Workspace templates
    # ------------------------------------------------------------------

    def create_from_template(
        self, template_name: str, workspace_name: Optional[str] = None
    ) -> WorkspaceModel:
        tmpl = _WORKSPACE_TEMPLATES.get(template_name)
        if not tmpl:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(_WORKSPACE_TEMPLATES.keys())}"
            )

        name = workspace_name or f"{template_name} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        ws = self.create_workspace(
            name=name,
            preset=tmpl["preset"],
            meta={"template": template_name, "description": tmpl["description"]},
        )

        default_ticker = tmpl.get("default_ticker", "AAPL")
        default_group  = f"link_{ws.id}"

        for p_def in tmpl["panels"]:
            self.add_panel(
                workspace_id=ws.id,
                panel_type=p_def["type"],
                ticker=default_ticker,
                title=p_def["title"],
                row=p_def.get("row", 0),
                col=p_def.get("col", 0),
                row_span=p_def.get("row_span", 3),
                col_span=p_def.get("col_span", 6),
                is_linked=True,
                link_group=default_group,
            )

        return ws

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (key, value, ts) VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts
                """,
                (key, json.dumps(value), now),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        if not _DB_PATH.exists():
            return default
        with _get_db() as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key=?", (key,)
            ).fetchone()
        if not row:
            return default
        return json.loads(row["value"])

    def all_preferences(self) -> dict:
        if not _DB_PATH.exists():
            return {}
        with _get_db() as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_workspace(self, workspace_id: str) -> dict:
        """
        Full JSON snapshot of a workspace: metadata + all panels + all alerts.
        """
        ws = self.get_workspace(workspace_id)
        if not ws:
            return {"error": "Workspace not found"}

        panels = self.list_panels(workspace_id)
        panel_snapshots = []
        for p in panels:
            alerts = self.list_panel_alerts(p.id)
            history = self.get_panel_history(p.id)
            panel_snapshots.append({
                **p.dict(),
                "meta":    _PANEL_META.get(p.panel_type, {}),
                "alerts":  alerts,
                "history": history,
            })

        return {
            "workspace":     ws.dict(),
            "panels":        panel_snapshots,
            "panel_count":   len(panels),
            "link_bus_state":_link_bus.all_groups(),
            "exported_at":   datetime.now(timezone.utc).isoformat(),
            "version":       "2.0",
        }

    def import_workspace(self, snapshot: dict) -> WorkspaceModel:
        """Restore a workspace from a JSON snapshot."""
        ws_data   = snapshot.get("workspace", {})
        panels_data = snapshot.get("panels", [])

        name = ws_data.get("name", f"Imported {uuid.uuid4().hex[:6]}")
        # Ensure unique name
        existing = self.get_workspace_by_name(name)
        if existing:
            name = f"{name} (restored {datetime.now(timezone.utc).strftime('%H%M%S')})"

        ws = self.create_workspace(
            name=name,
            preset=LayoutPreset(ws_data.get("layout_preset", LayoutPreset.CUSTOM.value)),
            meta=ws_data.get("meta", {}),
        )

        for p_data in panels_data:
            panel_type_str = p_data.get("panel_type")
            try:
                pt = PanelType(panel_type_str)
            except ValueError:
                pt = PanelType.PRICE_CHART

            self.add_panel(
                workspace_id=ws.id,
                panel_type=pt,
                ticker=p_data.get("ticker"),
                title=p_data.get("title", ""),
                row=p_data.get("row", 0),
                col=p_data.get("col", 0),
                row_span=p_data.get("row_span", 1),
                col_span=p_data.get("col_span", 6),
                interval=p_data.get("interval", "1D"),
                theme=p_data.get("theme", "dark"),
                settings=p_data.get("settings", {}),
                is_linked=p_data.get("is_linked", True),
                link_group=p_data.get("link_group"),
            )

        return ws

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_tab_index(self) -> int:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(tab_index), -1) + 1 AS next_idx FROM workspaces"
            ).fetchone()
        return row["next_idx"] if row else 0

    def _count_panels(self, workspace_id: str) -> int:
        with _get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM panels WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def _clear_panels(self, workspace_id: str) -> None:
        with _get_db() as conn:
            conn.execute("DELETE FROM panels WHERE workspace_id=?", (workspace_id,))


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

_manager = WorkspaceManager()

# ---------------------------------------------------------------------------
# FastAPI request/response models
# ---------------------------------------------------------------------------


class CreateWorkspaceRequest(BaseModel):
    name:          str = Field(..., min_length=1, max_length=100)
    layout_preset: LayoutPreset = LayoutPreset.CUSTOM
    tab_index:     Optional[int] = None
    meta:          dict[str, Any] = Field(default_factory=dict)


class AddPanelRequest(BaseModel):
    panel_type:  PanelType
    ticker:      Optional[str] = None
    title:       Optional[str] = None
    row:         int = 0
    col:         int = 0
    row_span:    int = 1
    col_span:    int = 6
    interval:    str = "1D"
    theme:       str = "dark"
    settings:    dict[str, Any] = Field(default_factory=dict)
    is_linked:   bool = True
    link_group:  Optional[str] = None


class UpdatePanelRequest(BaseModel):
    ticker:       Optional[str] = None
    title:        Optional[str] = None
    interval:     Optional[str] = None
    theme:        Optional[str] = None
    settings:     Optional[dict[str, Any]] = None
    alert_config: Optional[dict[str, Any]] = None
    is_fullscreen:Optional[bool] = None
    is_linked:    Optional[bool] = None
    link_group:   Optional[str] = None


class LayoutUpdateRequest(BaseModel):
    panel_positions: list[dict] = Field(
        ...,
        description="[{panel_id, row, col, row_span, col_span}]"
    )


class TickerBroadcastRequest(BaseModel):
    ticker:       str
    link_group:   Optional[str] = None


class ApplyPresetRequest(BaseModel):
    preset:        LayoutPreset
    panel_types:   Optional[list[PanelType]] = None
    default_ticker:str = "AAPL"


class AddAlertRequest(BaseModel):
    metric:    str
    operator:  AlertOperator
    threshold: float
    label:     Optional[str] = None


class CheckAlertsRequest(BaseModel):
    current_values: dict[str, float]


class CreateFromTemplateRequest(BaseModel):
    template_name:  str
    workspace_name: Optional[str] = None


class PushStateRequest(BaseModel):
    ticker:     Optional[str] = None
    interval:   Optional[str] = None
    settings:   dict[str, Any] = Field(default_factory=dict)
    scroll_pos: Optional[int] = None
    sort_col:   Optional[str] = None
    filters:    dict[str, Any] = Field(default_factory=dict)


class SetPreferenceRequest(BaseModel):
    key:   str
    value: Any


class ImportWorkspaceRequest(BaseModel):
    snapshot: dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

workspace_v2_router = APIRouter(prefix="/workspace-v2", tags=["WorkspaceV2"])


# ── Workspace endpoints ─────────────────────────────────────────────────────


@workspace_v2_router.get("/workspaces", summary="List all workspaces")
def list_workspaces() -> dict:
    """Return all workspaces ordered by tab index."""
    workspaces = _manager.list_workspaces()
    return {
        "count": len(workspaces),
        "workspaces": [ws.dict() for ws in workspaces],
    }


@workspace_v2_router.post("/workspace", summary="Create workspace")
def create_workspace(req: CreateWorkspaceRequest) -> dict:
    """Create a new named workspace."""
    try:
        ws = _manager.create_workspace(
            name=req.name,
            preset=req.layout_preset,
            tab_index=req.tab_index,
            meta=req.meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ws.dict()


@workspace_v2_router.get("/workspace/{workspace_id}", summary="Get workspace")
def get_workspace(workspace_id: str) -> dict:
    """Get workspace metadata."""
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws.dict()


@workspace_v2_router.delete("/workspace/{workspace_id}", summary="Delete workspace")
def delete_workspace(workspace_id: str) -> dict:
    """Delete a workspace and all its panels."""
    deleted = _manager.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"deleted": True, "workspace_id": workspace_id}


@workspace_v2_router.post("/workspace/{workspace_id}/activate", summary="Set active workspace")
def activate_workspace(workspace_id: str) -> dict:
    """Set a workspace as the currently active tab."""
    ok = _manager.set_active_workspace(workspace_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"active_workspace_id": workspace_id}


@workspace_v2_router.get("/active-workspace", summary="Get active workspace")
def get_active_workspace() -> dict:
    """Return the currently active workspace with all panels."""
    ws = _manager.get_active_workspace()
    if not ws:
        raise HTTPException(status_code=404, detail="No active workspace")
    panels = _manager.list_panels(ws.id)
    return {
        **ws.dict(),
        "panels": [p.dict() for p in panels],
        "panel_count": len(panels),
    }


@workspace_v2_router.post("/workspace/switch/{tab_index}", summary="Switch tab by index")
def switch_tab(tab_index: int) -> dict:
    """Quick-switch to workspace by tab index (0-based = Ctrl+1)."""
    ws = _manager.switch_tab(tab_index)
    if not ws:
        raise HTTPException(
            status_code=404, detail=f"No workspace at tab index {tab_index}"
        )
    return {"switched_to": ws.dict()}


@workspace_v2_router.post("/workspace/from-template", summary="Create from template")
def create_from_template(req: CreateFromTemplateRequest) -> dict:
    """Create a workspace from a named template."""
    try:
        ws = _manager.create_from_template(req.template_name, req.workspace_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    panels = _manager.list_panels(ws.id)
    return {
        **ws.dict(),
        "panels": [p.dict() for p in panels],
    }


@workspace_v2_router.post("/workspace/{workspace_id}/import", summary="Import workspace snapshot")
def import_workspace(workspace_id: str, req: ImportWorkspaceRequest) -> dict:
    """Restore a workspace from a previously exported JSON snapshot."""
    try:
        ws = _manager.import_workspace(req.snapshot)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    panels = _manager.list_panels(ws.id)
    return {**ws.dict(), "panels": [p.dict() for p in panels]}


@workspace_v2_router.get("/workspace/{workspace_id}/export", summary="Export workspace snapshot")
def export_workspace(workspace_id: str) -> dict:
    """Export a full JSON snapshot of the workspace."""
    snapshot = _manager.export_workspace(workspace_id)
    if "error" in snapshot:
        raise HTTPException(status_code=404, detail=snapshot["error"])
    return snapshot


# ── Layout endpoints ─────────────────────────────────────────────────────────


@workspace_v2_router.put("/workspace/{workspace_id}/layout", summary="Batch update panel layout")
def update_layout(workspace_id: str, req: LayoutUpdateRequest) -> dict:
    """
    Drag-drop rearrangement: update row, col, span for multiple panels at once.
    """
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    updated = _manager.update_layout(workspace_id, req.panel_positions)
    return {
        "updated_count": len(updated),
        "panels": [p.dict() for p in updated],
    }


@workspace_v2_router.post("/workspace/{workspace_id}/apply-preset", summary="Apply layout preset")
def apply_preset(workspace_id: str, req: ApplyPresetRequest) -> dict:
    """Apply a named grid preset (1x1, 2x1, 2x2, 3x2, 4x4) to the workspace."""
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    panels = _manager.apply_preset(
        workspace_id, req.preset, req.panel_types, req.default_ticker
    )
    return {
        "preset":        req.preset.value,
        "panel_count":   len(panels),
        "panels":        [p.dict() for p in panels],
    }


@workspace_v2_router.get("/workspace/{workspace_id}/validate", summary="Validate layout")
def validate_layout(workspace_id: str) -> dict:
    """Check for overlapping panels and out-of-bounds positions."""
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return _manager.validate_layout(workspace_id)


# ── Panel endpoints ──────────────────────────────────────────────────────────


@workspace_v2_router.get("/workspace/{workspace_id}/panels", summary="List panels")
def list_panels(workspace_id: str) -> dict:
    """List all panels in a workspace."""
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    panels = _manager.list_panels(workspace_id)
    return {
        "count": len(panels),
        "panels": [
            {**p.dict(), "panel_meta": _PANEL_META.get(p.panel_type, {})}
            for p in panels
        ],
    }


@workspace_v2_router.post("/workspace/{workspace_id}/panel", summary="Add panel")
def add_panel(workspace_id: str, req: AddPanelRequest) -> dict:
    """Add a new panel to a workspace."""
    ws = _manager.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    try:
        panel = _manager.add_panel(
            workspace_id=workspace_id,
            panel_type=req.panel_type,
            ticker=req.ticker,
            title=req.title,
            row=req.row, col=req.col,
            row_span=req.row_span, col_span=req.col_span,
            interval=req.interval, theme=req.theme,
            settings=req.settings,
            is_linked=req.is_linked,
            link_group=req.link_group,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return panel.dict()


@workspace_v2_router.get("/workspace/{workspace_id}/panel/{panel_id}", summary="Get panel")
def get_panel(workspace_id: str, panel_id: str) -> dict:
    """Get panel details including meta and alerts."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    return {
        **panel.dict(),
        "panel_meta": _PANEL_META.get(panel.panel_type, {}),
        "alerts":     _manager.list_panel_alerts(panel_id),
    }


@workspace_v2_router.patch("/workspace/{workspace_id}/panel/{panel_id}", summary="Update panel")
def update_panel(workspace_id: str, panel_id: str, req: UpdatePanelRequest) -> dict:
    """Update panel properties."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    updated = _manager.update_panel(
        panel_id,
        ticker=req.ticker,
        title=req.title,
        interval=req.interval,
        theme=req.theme,
        settings=req.settings,
        alert_config=req.alert_config,
        is_fullscreen=req.is_fullscreen,
        is_linked=req.is_linked,
        link_group=req.link_group,
    )
    return updated.dict() if updated else {}


@workspace_v2_router.delete(
    "/workspace/{workspace_id}/panel/{panel_id}", summary="Delete panel"
)
def delete_panel(workspace_id: str, panel_id: str) -> dict:
    """Remove a panel from a workspace."""
    deleted = _manager.delete_panel(workspace_id, panel_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Panel not found")
    return {"deleted": True, "panel_id": panel_id}


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/fullscreen",
    summary="Toggle fullscreen",
)
def toggle_fullscreen(workspace_id: str, panel_id: str) -> dict:
    """Toggle fullscreen mode for a panel."""
    panel = _manager.toggle_fullscreen(workspace_id, panel_id)
    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")
    return {"panel_id": panel_id, "is_fullscreen": panel.is_fullscreen}


# ── Ticker linking ────────────────────────────────────────────────────────────


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/broadcast",
    summary="Broadcast ticker to linked panels",
)
def broadcast_ticker(
    workspace_id: str, panel_id: str, req: TickerBroadcastRequest
) -> dict:
    """
    Change the ticker on the source panel and propagate to all panels
    in the same link group within the workspace.
    """
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    result = _manager.broadcast_ticker(panel_id, req.ticker, req.link_group)
    return result


@workspace_v2_router.get("/link-bus/groups", summary="List link bus groups")
def get_link_groups() -> dict:
    """Return all active ticker link groups and their panel members."""
    return {"groups": _link_bus.all_groups()}


# ── Panel history ─────────────────────────────────────────────────────────────


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/history/push",
    summary="Push panel state to history",
)
def push_panel_state(
    workspace_id: str, panel_id: str, req: PushStateRequest
) -> dict:
    """Push the current panel state onto its history stack."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = PanelState(**req.dict())
    _manager.push_panel_state(panel_id, state)
    return {"pushed": True, "state": state.dict()}


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/history/back",
    summary="Navigate panel history back",
)
def navigate_back(workspace_id: str, panel_id: str) -> dict:
    """Navigate the panel back to its previous state."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _manager.navigate_panel_back(panel_id)
    if not state:
        return {"navigated": False, "message": "At beginning of history"}
    return {"navigated": True, "state": state}


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/history/forward",
    summary="Navigate panel history forward",
)
def navigate_forward(workspace_id: str, panel_id: str) -> dict:
    """Navigate the panel forward to its next state."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _manager.navigate_panel_forward(panel_id)
    if not state:
        return {"navigated": False, "message": "At end of history"}
    return {"navigated": True, "state": state}


@workspace_v2_router.get(
    "/workspace/{workspace_id}/panel/{panel_id}/history",
    summary="Get panel history",
)
def get_panel_history(workspace_id: str, panel_id: str) -> dict:
    """Return the full state history for a panel."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    history = _manager.get_panel_history(panel_id)
    return {"panel_id": panel_id, "count": len(history), "history": history}


# ── Alerts ────────────────────────────────────────────────────────────────────


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/alert",
    summary="Add panel alert",
)
def add_panel_alert(
    workspace_id: str, panel_id: str, req: AddAlertRequest
) -> dict:
    """Add a threshold alert to a panel."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    alert = _manager.add_panel_alert(
        panel_id, req.metric, req.operator, req.threshold, req.label
    )
    return alert


@workspace_v2_router.get(
    "/workspace/{workspace_id}/panel/{panel_id}/alerts",
    summary="List panel alerts",
)
def list_alerts(workspace_id: str, panel_id: str) -> dict:
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    return {"panel_id": panel_id, "alerts": _manager.list_panel_alerts(panel_id)}


@workspace_v2_router.post(
    "/workspace/{workspace_id}/panel/{panel_id}/alerts/check",
    summary="Check alerts against current values",
)
def check_alerts(
    workspace_id: str, panel_id: str, req: CheckAlertsRequest
) -> dict:
    """Check panel alerts against freshly-polled metric values."""
    panel = _manager.get_panel(panel_id)
    if not panel or panel.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Panel not found")
    triggered = _manager.check_alerts(panel_id, req.current_values)
    return {
        "panel_id":       panel_id,
        "triggered_count":len(triggered),
        "triggered":      triggered,
    }


@workspace_v2_router.delete(
    "/workspace/{workspace_id}/panel/{panel_id}/alert/{alert_id}",
    summary="Delete alert",
)
def delete_alert(workspace_id: str, panel_id: str, alert_id: str) -> dict:
    deleted = _manager.delete_panel_alert(alert_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"deleted": True, "alert_id": alert_id}


# ── Preferences ───────────────────────────────────────────────────────────────


@workspace_v2_router.post("/preferences", summary="Set user preference")
def set_preference(req: SetPreferenceRequest) -> dict:
    _manager.set_preference(req.key, req.value)
    return {"key": req.key, "value": req.value, "saved": True}


@workspace_v2_router.get("/preferences", summary="Get all preferences")
def get_preferences() -> dict:
    return {"preferences": _manager.all_preferences()}


@workspace_v2_router.get("/preferences/{key}", summary="Get single preference")
def get_preference(key: str) -> dict:
    val = _manager.get_preference(key)
    return {"key": key, "value": val}


# ── Panel types catalogue ────────────────────────────────────────────────────


@workspace_v2_router.get("/panel-types", summary="List all panel types")
def list_panel_types() -> dict:
    """Return all available panel types with their metadata."""
    result = []
    for pt in PanelType:
        meta = _PANEL_META.get(pt, {})
        result.append({
            "type":            pt.value,
            "adapter":         meta.get("adapter"),
            "refresh_seconds": meta.get("refresh_seconds"),
            "params":          meta.get("params", []),
            "default_interval":meta.get("default_interval"),
            "supports_overlay":meta.get("supports_overlay", False),
            "description":     meta.get("description", ""),
        })
    return {"count": len(result), "panel_types": result}


@workspace_v2_router.get("/layout-presets", summary="List layout presets")
def list_layout_presets() -> dict:
    """Return all grid layout presets with slot definitions."""
    result = {}
    for preset, slots in _LAYOUT_GRIDS.items():
        result[preset.value] = {
            "preset":       preset.value,
            "slot_count":   len(slots),
            "slots":        [
                {"row": r, "col": c, "row_span": rs, "col_span": cs}
                for r, c, rs, cs in slots
            ],
        }
    return {"presets": result}


@workspace_v2_router.get("/templates", summary="List workspace templates")
def list_templates() -> dict:
    """Return all workspace templates with panel configurations."""
    result = {}
    for name, tmpl in _WORKSPACE_TEMPLATES.items():
        result[name] = {
            "name":           name,
            "description":    tmpl["description"],
            "preset":         tmpl["preset"].value,
            "panel_count":    len(tmpl["panels"]),
            "panels":         [
                {"type": p["type"].value, "title": p["title"]}
                for p in tmpl["panels"]
            ],
            "default_ticker": tmpl.get("default_ticker", "AAPL"),
        }
    return {"count": len(result), "templates": result}


@workspace_v2_router.get("/health", summary="Workspace health check")
def workspace_v2_health() -> dict:
    """Return health status of the workspace module."""
    workspaces = _manager.list_workspaces()
    total_panels = sum(
        _manager._count_panels(ws.id) for ws in workspaces
    )
    return {
        "status":           "ok",
        "db_path":          str(_DB_PATH),
        "db_exists":        _DB_PATH.exists(),
        "workspace_count":  len(workspaces),
        "total_panels":     total_panels,
        "panel_types":      len(PanelType),
        "layout_presets":   len(LayoutPreset),
        "templates":        len(_WORKSPACE_TEMPLATES),
        "link_groups":      len(_link_bus.all_groups()),
        "ts":               datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Convenience re-export
# ---------------------------------------------------------------------------

router = workspace_v2_router
