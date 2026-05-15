"""
Multi-panel workspace layout manager — Dimension #91.

Provides a data model + layout engine for the SENTINEL terminal UI.
Rendering is delegated to TradingView / the frontend; this module owns:

  - Panel / Workspace dataclasses with full JSON serialisation
  - WorkspaceManager: layout presets, CRUD, symbol-broadcast, TV export
  - WorkspaceStateStore: persists the active workspace across sessions
  - FastAPI ``workspace_router``: REST API for all workspace operations

Score target: raise dim_091 from 5 → 9+
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "PanelType",
    "LayoutType",
    "Panel",
    "Workspace",
    "WorkspaceManager",
    "WorkspaceStateStore",
    "workspace_router",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKSPACES_DIR = Path(".sentinel") / "workspaces"
_STATE_DIR = Path(".sentinel") / "state"
_ACTIVE_WORKSPACE_FILE = _STATE_DIR / "active_workspace.json"

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PanelType(str, Enum):
    """Supported panel content types."""

    CHART = "chart"
    DATA = "data"
    NEWS = "news"
    WATCHLIST = "watchlist"
    SCREENER = "screener"
    MACRO = "macro"
    FINANCIALS = "financials"
    INSIDER = "insider"
    YIELD_CURVE = "yield_curve"
    FX = "fx"
    FUTURES = "futures"
    CALENDAR = "calendar"
    DEFI = "defi"
    ONCHAIN = "onchain"
    PORTFOLIO = "portfolio"
    PNL = "pnl"
    RISK = "risk"
    CORRELATION = "correlation"


class LayoutType(str, Enum):
    """Named layout grid configurations."""

    SINGLE = "SINGLE"
    TWO_COL = "TWO_COL"
    THREE_COL = "THREE_COL"
    QUAD = "QUAD"
    SIX_PANEL = "SIX_PANEL"
    CUSTOM = "CUSTOM"


# ---------------------------------------------------------------------------
# Panel dataclass
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """
    Represents a single panel within a workspace grid.

    Grid coordinates use a 12-column, N-row layout convention (CSS Grid /
    TradingView-style). ``rowspan`` and ``colspan`` are expressed in grid units.

    Attributes
    ----------
    id:
        Unique panel identifier (UUID4 string).
    type:
        Content type — controls which data adapter powers this panel.
    symbol:
        Primary instrument symbol (e.g. ``"AAPL"``, ``"BTCUSDT"``).
    interval:
        Chart interval string (e.g. ``"1D"``, ``"4H"``, ``"1W"``).
    indicators:
        List of indicator keys to display (e.g. ``["EMA_20", "RSI_14"]``).
    row:
        Zero-based grid row of the panel's top-left corner.
    col:
        Zero-based grid column of the panel's top-left corner.
    rowspan:
        How many rows this panel spans (default 1).
    colspan:
        How many columns this panel spans in a 12-col grid (default 6).
    title:
        Display title shown in the panel header.
    settings:
        Arbitrary panel-specific configuration (theme, locale, etc.).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: PanelType = PanelType.CHART
    symbol: str = ""
    interval: str = "1D"
    indicators: list[str] = field(default_factory=list)
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 6
    title: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Panel":
        data = dict(data)
        data["type"] = PanelType(data.get("type", PanelType.CHART.value))
        return cls(**data)


# ---------------------------------------------------------------------------
# Workspace dataclass
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """
    A named collection of panels arranged in a grid layout.

    Attributes
    ----------
    name:
        Human-readable workspace name (used as the primary key for storage).
    layout_type:
        Describes the high-level grid arrangement.
    panels:
        Ordered list of panels in this workspace.
    created_at:
        ISO-8601 UTC creation timestamp.
    updated_at:
        ISO-8601 UTC last-modification timestamp.
    meta:
        Arbitrary metadata (author, tags, description, etc.).
    """

    name: str
    layout_type: LayoutType = LayoutType.CUSTOM
    panels: list[Panel] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layout_type": self.layout_type.value,
            "panels": [p.to_dict() for p in self.panels],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Workspace":
        panels = [Panel.from_dict(p) for p in data.get("panels", [])]
        return cls(
            name=data["name"],
            layout_type=LayoutType(data.get("layout_type", LayoutType.CUSTOM.value)),
            panels=panels,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            meta=data.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------

# Panel-data-adapter config: maps PanelType → fetch config skeleton
_PANEL_ADAPTER_MAP: dict[PanelType, dict] = {
    PanelType.CHART: {
        "adapter": "sds.ohlcv",
        "refresh_seconds": 60,
        "params": ["symbol", "interval"],
    },
    PanelType.DATA: {
        "adapter": "sfe.fundamentals",
        "refresh_seconds": 3600,
        "params": ["symbol"],
    },
    PanelType.NEWS: {
        "adapter": "sma.news_flow",
        "refresh_seconds": 300,
        "params": ["symbol"],
    },
    PanelType.WATCHLIST: {
        "adapter": "sds.quotes",
        "refresh_seconds": 15,
        "params": ["symbols"],
    },
    PanelType.SCREENER: {
        "adapter": "sbx.screener",
        "refresh_seconds": 300,
        "params": ["filters"],
    },
    PanelType.MACRO: {
        "adapter": "sma.global_macro",
        "refresh_seconds": 900,
        "params": [],
    },
    PanelType.FINANCIALS: {
        "adapter": "sfe.fundamentals",
        "refresh_seconds": 86400,
        "params": ["symbol"],
    },
    PanelType.INSIDER: {
        "adapter": "sfe.form4_parser",
        "refresh_seconds": 3600,
        "params": ["symbol"],
    },
    PanelType.YIELD_CURVE: {
        "adapter": "sfe.yield_curve",
        "refresh_seconds": 900,
        "params": [],
    },
    PanelType.FX: {
        "adapter": "sfe.fx_analytics",
        "refresh_seconds": 60,
        "params": ["pairs"],
    },
    PanelType.FUTURES: {
        "adapter": "sds.futures",
        "refresh_seconds": 60,
        "params": ["symbols"],
    },
    PanelType.CALENDAR: {
        "adapter": "sma.economic_calendar",
        "refresh_seconds": 1800,
        "params": [],
    },
    PanelType.DEFI: {
        "adapter": "sfe.onchain_metrics",
        "refresh_seconds": 300,
        "params": ["protocol"],
    },
    PanelType.ONCHAIN: {
        "adapter": "sfe.onchain_metrics",
        "refresh_seconds": 300,
        "params": ["coin"],
    },
    PanelType.PORTFOLIO: {
        "adapter": "spr.portfolio",
        "refresh_seconds": 30,
        "params": [],
    },
    PanelType.PNL: {
        "adapter": "spr.pnl",
        "refresh_seconds": 30,
        "params": [],
    },
    PanelType.RISK: {
        "adapter": "spr.risk",
        "refresh_seconds": 300,
        "params": [],
    },
    PanelType.CORRELATION: {
        "adapter": "spr.correlation",
        "refresh_seconds": 900,
        "params": ["symbols"],
    },
}


class WorkspaceManager:
    """
    Central manager for SENTINEL workspace layouts.

    Handles creation from presets, CRUD operations, JSON persistence, and
    TradingView-compatible layout export.

    Parameters
    ----------
    workspaces_dir:
        Directory where workspace JSON files are stored.
        Defaults to ``.sentinel/workspaces/``.
    """

    # ── Layout Presets ────────────────────────────────────────────────────────

    LAYOUT_PRESETS: dict[str, list[dict]] = {
        # Bloomberg-style: main chart (2×2 grid units), data table (top-right),
        # news feed (right), watchlist (bottom-left), macro overview (bottom-mid)
        "bloomberg_style": [
            {
                "type": PanelType.CHART,
                "row": 0, "col": 0, "rowspan": 2, "colspan": 8,
                "title": "Main Chart",
                "interval": "1D",
            },
            {
                "type": PanelType.DATA,
                "row": 0, "col": 8, "rowspan": 1, "colspan": 4,
                "title": "Key Statistics",
            },
            {
                "type": PanelType.NEWS,
                "row": 1, "col": 8, "rowspan": 1, "colspan": 4,
                "title": "News Flow",
            },
            {
                "type": PanelType.WATCHLIST,
                "row": 2, "col": 0, "rowspan": 1, "colspan": 4,
                "title": "Watchlist",
            },
            {
                "type": PanelType.MACRO,
                "row": 2, "col": 4, "rowspan": 1, "colspan": 4,
                "title": "Macro Overview",
            },
            {
                "type": PanelType.SCREENER,
                "row": 2, "col": 8, "rowspan": 1, "colspan": 4,
                "title": "Screener",
            },
        ],
        # Equity research: chart + financials + news + insider activity
        "equity_research": [
            {
                "type": PanelType.CHART,
                "row": 0, "col": 0, "rowspan": 2, "colspan": 7,
                "title": "Price Chart",
                "interval": "1W",
                "indicators": ["EMA_50", "EMA_200", "Volume"],
            },
            {
                "type": PanelType.FINANCIALS,
                "row": 0, "col": 7, "rowspan": 1, "colspan": 5,
                "title": "Financials",
            },
            {
                "type": PanelType.NEWS,
                "row": 1, "col": 7, "rowspan": 1, "colspan": 5,
                "title": "News & Sentiment",
            },
            {
                "type": PanelType.INSIDER,
                "row": 2, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "Insider Activity",
            },
            {
                "type": PanelType.DATA,
                "row": 2, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "Analyst Estimates",
            },
        ],
        # Macro trader: yield curve + FX + equity futures + economic calendar
        "macro_trader": [
            {
                "type": PanelType.YIELD_CURVE,
                "row": 0, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "Yield Curve",
            },
            {
                "type": PanelType.FX,
                "row": 0, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "FX Rates",
                "settings": {"pairs": ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CNH"]},
            },
            {
                "type": PanelType.FUTURES,
                "row": 1, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "Equity Futures",
                "settings": {"symbols": ["ES", "NQ", "RTY", "YM"]},
            },
            {
                "type": PanelType.CALENDAR,
                "row": 1, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "Economic Calendar",
            },
        ],
        # Crypto desk: BTC chart + ETH chart + DeFi TVL + on-chain metrics
        "crypto_desk": [
            {
                "type": PanelType.CHART,
                "row": 0, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "BTC/USD",
                "symbol": "BTCUSDT",
                "interval": "4H",
                "indicators": ["EMA_20", "RSI_14", "MACD"],
            },
            {
                "type": PanelType.CHART,
                "row": 0, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "ETH/USD",
                "symbol": "ETHUSDT",
                "interval": "4H",
                "indicators": ["EMA_20", "RSI_14"],
            },
            {
                "type": PanelType.DEFI,
                "row": 1, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "DeFi TVL",
            },
            {
                "type": PanelType.ONCHAIN,
                "row": 1, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "On-Chain Metrics",
                "settings": {"coin": "bitcoin"},
            },
        ],
        # Portfolio monitor: positions + P&L + risk decomposition + correlation
        "portfolio_monitor": [
            {
                "type": PanelType.PORTFOLIO,
                "row": 0, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "Positions",
            },
            {
                "type": PanelType.PNL,
                "row": 0, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "P&L Attribution",
            },
            {
                "type": PanelType.RISK,
                "row": 1, "col": 0, "rowspan": 1, "colspan": 6,
                "title": "Risk Decomposition",
            },
            {
                "type": PanelType.CORRELATION,
                "row": 1, "col": 6, "rowspan": 1, "colspan": 6,
                "title": "Correlation Heatmap",
            },
        ],
    }

    # ── Layout type → grid dimensions ─────────────────────────────────────────

    _LAYOUT_DIMS: dict[LayoutType, dict] = {
        LayoutType.SINGLE: {"rows": 1, "cols": 12},
        LayoutType.TWO_COL: {"rows": 1, "cols": 12},
        LayoutType.THREE_COL: {"rows": 1, "cols": 12},
        LayoutType.QUAD: {"rows": 2, "cols": 12},
        LayoutType.SIX_PANEL: {"rows": 3, "cols": 12},
        LayoutType.CUSTOM: {"rows": 4, "cols": 12},
    }

    def __init__(self, workspaces_dir: Path = _WORKSPACES_DIR) -> None:
        self._dir = workspaces_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        # In-memory registry: name → Workspace
        self._registry: dict[str, Workspace] = {}
        self._load_all_from_disk()

    # ── internal ──────────────────────────────────────────────────────────────

    def _path_for(self, name: str) -> Path:
        safe_name = name.replace(" ", "_").lower()
        return self._dir / f"{safe_name}.json"

    def _load_all_from_disk(self) -> None:
        """Pre-populate the in-memory registry from persisted JSON files."""
        for fp in self._dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                ws = Workspace.from_dict(data)
                self._registry[ws.name] = ws
            except Exception as exc:
                logger.warning("workspace load failed for %s: %s", fp.name, exc)

    def _panels_from_preset(
        self,
        preset_name: str,
        symbols: Optional[list[str]] = None,
    ) -> list[Panel]:
        """Instantiate Panel objects from a preset spec, optionally injecting symbols."""
        specs = self.LAYOUT_PRESETS.get(preset_name, [])
        panels: list[Panel] = []
        sym_iter = iter(symbols or [])

        for spec in specs:
            spec = dict(spec)
            panel_type: PanelType = spec.pop("type")

            # Inject symbol into chart panels if provided
            if panel_type == PanelType.CHART:
                sym = next(sym_iter, spec.get("symbol", ""))
            else:
                sym = spec.get("symbol", "")

            panel = Panel(
                type=panel_type,
                symbol=sym,
                interval=spec.pop("interval", "1D"),
                indicators=spec.pop("indicators", []),
                row=spec.pop("row", 0),
                col=spec.pop("col", 0),
                rowspan=spec.pop("rowspan", 1),
                colspan=spec.pop("colspan", 6),
                title=spec.pop("title", ""),
                settings=spec.pop("settings", {}),
            )
            panels.append(panel)
        return panels

    # ── public ────────────────────────────────────────────────────────────────

    def create_workspace(
        self,
        name: str,
        layout_type: str = "CUSTOM",
        symbols: Optional[list[str]] = None,
        preset: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Workspace:
        """
        Create a new workspace, optionally from a named preset.

        Parameters
        ----------
        name:
            Unique workspace name.
        layout_type:
            One of the :class:`LayoutType` enum values.
        symbols:
            Symbols to inject into chart panels (in order).
        preset:
            Name of a layout preset from :attr:`LAYOUT_PRESETS`.
        meta:
            Optional metadata dict (author, description, tags, …).

        Returns
        -------
        The newly created :class:`Workspace`.

        Raises
        ------
        ValueError:
            If a workspace with the same name already exists.
        """
        if name in self._registry:
            raise ValueError(f"Workspace {name!r} already exists")

        lt = LayoutType(layout_type)
        panels: list[Panel] = []

        if preset and preset in self.LAYOUT_PRESETS:
            panels = self._panels_from_preset(preset, symbols)
        elif lt == LayoutType.SINGLE:
            sym = (symbols or [""])[0]
            panels = [Panel(type=PanelType.CHART, symbol=sym, colspan=12, title="Chart")]
        elif lt == LayoutType.TWO_COL:
            syms = (symbols or ["", ""])
            panels = [
                Panel(type=PanelType.CHART, symbol=syms[0], col=0, colspan=6, title="Left"),
                Panel(type=PanelType.CHART, symbol=syms[1], col=6, colspan=6, title="Right"),
            ]
        elif lt == LayoutType.THREE_COL:
            syms = list((symbols or []) + [""] * 3)
            panels = [
                Panel(type=PanelType.CHART, symbol=syms[0], col=0, colspan=4, title="Left"),
                Panel(type=PanelType.CHART, symbol=syms[1], col=4, colspan=4, title="Center"),
                Panel(type=PanelType.CHART, symbol=syms[2], col=8, colspan=4, title="Right"),
            ]
        elif lt == LayoutType.QUAD:
            syms = list((symbols or []) + [""] * 4)
            panels = [
                Panel(type=PanelType.CHART, symbol=syms[0], row=0, col=0, colspan=6, title="TL"),
                Panel(type=PanelType.CHART, symbol=syms[1], row=0, col=6, colspan=6, title="TR"),
                Panel(type=PanelType.CHART, symbol=syms[2], row=1, col=0, colspan=6, title="BL"),
                Panel(type=PanelType.CHART, symbol=syms[3], row=1, col=6, colspan=6, title="BR"),
            ]
        elif lt == LayoutType.SIX_PANEL:
            syms = list((symbols or []) + [""] * 6)
            for idx in range(6):
                r, c = divmod(idx, 3)
                panels.append(Panel(
                    type=PanelType.CHART,
                    symbol=syms[idx],
                    row=r, col=c * 4,
                    colspan=4,
                    title=f"Panel {idx + 1}",
                ))

        ws = Workspace(
            name=name,
            layout_type=lt,
            panels=panels,
            meta=meta or {},
        )
        self._registry[name] = ws
        self.save_workspace(ws)
        logger.info("workspace created: %s (layout=%s, panels=%d)", name, lt.value, len(panels))
        return ws

    def add_panel(self, workspace_id: str, panel: Panel) -> Workspace:
        """
        Add a panel to an existing workspace.

        Parameters
        ----------
        workspace_id:
            Workspace ``name``.
        panel:
            :class:`Panel` instance to add.
        """
        ws = self._get_or_raise(workspace_id)
        ws.panels.append(panel)
        ws._touch()
        self.save_workspace(ws)
        return ws

    def remove_panel(self, workspace_id: str, panel_id: str) -> Workspace:
        """
        Remove a panel by its ``id``.

        Parameters
        ----------
        workspace_id:
            Workspace ``name``.
        panel_id:
            UUID string of the panel to remove.

        Raises
        ------
        KeyError:
            If the panel is not found.
        """
        ws = self._get_or_raise(workspace_id)
        before = len(ws.panels)
        ws.panels = [p for p in ws.panels if p.id != panel_id]
        if len(ws.panels) == before:
            raise KeyError(f"Panel {panel_id!r} not found in workspace {workspace_id!r}")
        ws._touch()
        self.save_workspace(ws)
        return ws

    def save_workspace(self, workspace: Workspace, path: Optional[str] = None) -> None:
        """
        Persist a workspace as JSON.

        Parameters
        ----------
        workspace:
            The workspace to serialise.
        path:
            Override the default storage path.
        """
        fp = Path(path) if path else self._path_for(workspace.name)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(
            json.dumps(workspace.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._registry[workspace.name] = workspace
        logger.debug("workspace saved: %s → %s", workspace.name, fp)

    def load_workspace(self, name: str) -> Workspace:
        """
        Load a workspace by name (from cache or disk).

        Raises
        ------
        FileNotFoundError:
            If the workspace JSON file does not exist.
        """
        if name in self._registry:
            return self._registry[name]

        fp = self._path_for(name)
        if not fp.exists():
            raise FileNotFoundError(f"Workspace {name!r} not found at {fp}")

        data = json.loads(fp.read_text(encoding="utf-8"))
        ws = Workspace.from_dict(data)
        self._registry[name] = ws
        return ws

    def list_workspaces(self) -> list[dict]:
        """
        Return a summary list of all persisted workspaces.

        Each entry contains ``name``, ``layout_type``, ``panel_count``,
        ``created_at``, and ``updated_at``.
        """
        summaries = []
        for fp in self._dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "name": data.get("name", fp.stem),
                        "layout_type": data.get("layout_type", "CUSTOM"),
                        "panel_count": len(data.get("panels", [])),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "meta": data.get("meta", {}),
                    }
                )
            except Exception as exc:
                logger.warning("list_workspaces: could not read %s: %s", fp.name, exc)
        return sorted(summaries, key=lambda x: x.get("updated_at", ""), reverse=True)

    def apply_symbol(self, workspace_id: str, symbol: str) -> Workspace:
        """
        Broadcast a symbol change to all chart panels in the workspace.

        Parameters
        ----------
        workspace_id:
            Workspace ``name``.
        symbol:
            New symbol string (e.g. ``"TSLA"``).
        """
        ws = self._get_or_raise(workspace_id)
        for panel in ws.panels:
            if panel.type == PanelType.CHART:
                panel.symbol = symbol
        ws._touch()
        self.save_workspace(ws)
        logger.info("apply_symbol: %s → all chart panels in %s", symbol, workspace_id)
        return ws

    def export_layout_json(self, workspace: Workspace) -> dict:
        """
        Export a TradingView-compatible layout JSON representation.

        The structure mirrors TradingView's ``multiChartLayout`` format so
        that the frontend can feed it directly to the TV widget API.
        """
        charts = []
        for panel in workspace.panels:
            chart_entry: dict[str, Any] = {
                "id": panel.id,
                "type": panel.type.value,
                "layout": {
                    "row": panel.row,
                    "col": panel.col,
                    "rowspan": panel.rowspan,
                    "colspan": panel.colspan,
                },
                "title": panel.title or panel.type.value.title(),
                "content": {},
            }
            if panel.type == PanelType.CHART:
                chart_entry["content"] = {
                    "symbol": panel.symbol,
                    "interval": panel.interval,
                    "indicators": panel.indicators,
                    "theme": panel.settings.get("theme", "dark"),
                    "locale": panel.settings.get("locale", "en"),
                    "toolbar_bg": panel.settings.get("toolbar_bg", "#1a1a1a"),
                }
            else:
                chart_entry["content"] = panel.settings

            charts.append(chart_entry)

        dims = self._LAYOUT_DIMS.get(workspace.layout_type, {"rows": 4, "cols": 12})

        return {
            "version": 1,
            "workspace": {
                "name": workspace.name,
                "layout_type": workspace.layout_type.value,
                "grid": dims,
                "created_at": workspace.created_at,
                "updated_at": workspace.updated_at,
            },
            "panels": charts,
        }

    def get_panel_data_config(self, panel: Panel) -> dict:
        """
        Return the data-fetch configuration for a given panel.

        The returned dict drives the SDS adapter layer:
          - ``adapter``: dotted module path
          - ``refresh_seconds``: how often to refetch
          - ``params``: list of required parameter names
          - ``resolved``: parameter values resolved from the panel

        Parameters
        ----------
        panel:
            The panel whose data config to resolve.
        """
        base = dict(_PANEL_ADAPTER_MAP.get(panel.type, {
            "adapter": "sds.generic",
            "refresh_seconds": 300,
            "params": [],
        }))

        resolved: dict[str, Any] = {}
        for param in base.get("params", []):
            if param == "symbol":
                resolved["symbol"] = panel.symbol
            elif param == "symbols":
                resolved["symbols"] = (
                    panel.settings.get("symbols", [panel.symbol] if panel.symbol else [])
                )
            elif param == "interval":
                resolved["interval"] = panel.interval
            elif param == "filters":
                resolved["filters"] = panel.settings.get("filters", {})
            elif param == "pairs":
                resolved["pairs"] = panel.settings.get("pairs", [])
            elif param == "protocol":
                resolved["protocol"] = panel.settings.get("protocol", "")
            elif param == "coin":
                resolved["coin"] = panel.settings.get("coin", "bitcoin")

        base["resolved"] = resolved
        base["panel_id"] = panel.id
        base["panel_type"] = panel.type.value
        return base

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_or_raise(self, workspace_id: str) -> Workspace:
        """Retrieve from registry or raise HTTPException(404)."""
        if workspace_id in self._registry:
            return self._registry[workspace_id]
        try:
            return self.load_workspace(workspace_id)
        except FileNotFoundError:
            raise KeyError(f"Workspace {workspace_id!r} not found")


# ---------------------------------------------------------------------------
# WorkspaceStateStore
# ---------------------------------------------------------------------------


class WorkspaceStateStore:
    """
    Persists the currently-active workspace name across sessions.

    Writes to ``.sentinel/state/active_workspace.json``.
    """

    def __init__(self, state_file: Path = _ACTIVE_WORKSPACE_FILE) -> None:
        self._file = state_file
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def set_active(self, workspace_name: str) -> None:
        """Persist the active workspace name."""
        payload = {
            "active_workspace": workspace_name,
            "set_at": datetime.now(timezone.utc).isoformat(),
        }
        self._file.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("active workspace set: %s", workspace_name)

    def get_active(self) -> Optional[str]:
        """Return the active workspace name, or None if not set."""
        if not self._file.exists():
            return None
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            return data.get("active_workspace")
        except Exception:
            return None

    def clear(self) -> None:
        """Remove the active workspace record."""
        if self._file.exists():
            self._file.unlink()


# ---------------------------------------------------------------------------
# Pydantic request/response models (FastAPI)
# ---------------------------------------------------------------------------


class PanelIn(BaseModel):
    """Request body for adding/updating a panel."""

    type: PanelType = PanelType.CHART
    symbol: str = ""
    interval: str = "1D"
    indicators: list[str] = Field(default_factory=list)
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 6
    title: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class WorkspaceCreateIn(BaseModel):
    """Request body for workspace creation."""

    name: str
    layout_type: str = LayoutType.CUSTOM.value
    symbols: list[str] = Field(default_factory=list)
    preset: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ApplySymbolIn(BaseModel):
    """Request body for symbol broadcast."""

    symbol: str


class WorkspaceSummary(BaseModel):
    """Lightweight workspace summary for listing."""

    name: str
    layout_type: str
    panel_count: int
    created_at: str
    updated_at: str
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

workspace_router = APIRouter(prefix="/api/workspaces", tags=["Workspaces"])

# Singleton manager shared across requests
_manager = WorkspaceManager()
_state_store = WorkspaceStateStore()


@workspace_router.get("/", response_model=list[WorkspaceSummary])
def list_workspaces() -> list[dict]:
    """List all saved workspaces (name, layout_type, panel_count, timestamps)."""
    return _manager.list_workspaces()


@workspace_router.post("/", response_model=dict)
def create_workspace(body: WorkspaceCreateIn) -> dict:
    """
    Create a new workspace.

    Optionally pass a ``preset`` name from the built-in presets:
    ``bloomberg_style``, ``equity_research``, ``macro_trader``,
    ``crypto_desk``, ``portfolio_monitor``.
    """
    try:
        ws = _manager.create_workspace(
            name=body.name,
            layout_type=body.layout_type,
            symbols=body.symbols or None,
            preset=body.preset,
            meta=body.meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    _state_store.set_active(ws.name)
    return ws.to_dict()


@workspace_router.get("/{name}", response_model=dict)
def get_workspace(name: str) -> dict:
    """Load a workspace by name."""
    try:
        ws = _manager.load_workspace(name)
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Workspace {name!r} not found")
    return ws.to_dict()


@workspace_router.put("/{name}/panels", response_model=dict)
def add_panel(name: str, body: PanelIn) -> dict:
    """Add a new panel to the workspace."""
    try:
        ws = _manager.load_workspace(name)
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Workspace {name!r} not found")

    panel = Panel(
        type=body.type,
        symbol=body.symbol,
        interval=body.interval,
        indicators=body.indicators,
        row=body.row,
        col=body.col,
        rowspan=body.rowspan,
        colspan=body.colspan,
        title=body.title,
        settings=body.settings,
    )
    ws = _manager.add_panel(name, panel)
    return ws.to_dict()


@workspace_router.delete("/{name}/panels/{panel_id}", response_model=dict)
def remove_panel(name: str, panel_id: str) -> dict:
    """Remove a panel from the workspace by panel ID."""
    try:
        ws = _manager.remove_panel(name, panel_id)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ws.to_dict()


@workspace_router.post("/{name}/apply-symbol", response_model=dict)
def apply_symbol(name: str, body: ApplySymbolIn) -> dict:
    """Broadcast a symbol change to all chart panels in the workspace."""
    try:
        ws = _manager.apply_symbol(name, body.symbol)
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Workspace {name!r} not found")
    return ws.to_dict()


@workspace_router.get("/{name}/export", response_model=dict)
def export_layout(name: str) -> dict:
    """Export a TradingView-compatible layout JSON for the workspace."""
    try:
        ws = _manager.load_workspace(name)
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Workspace {name!r} not found")
    return _manager.export_layout_json(ws)


@workspace_router.get("/{name}/panels/{panel_id}/data-config", response_model=dict)
def panel_data_config(name: str, panel_id: str) -> dict:
    """Return the data-fetch configuration for a specific panel."""
    try:
        ws = _manager.load_workspace(name)
    except (FileNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail=f"Workspace {name!r} not found")

    panel = next((p for p in ws.panels if p.id == panel_id), None)
    if panel is None:
        raise HTTPException(
            status_code=404, detail=f"Panel {panel_id!r} not found in {name!r}"
        )
    return _manager.get_panel_data_config(panel)


@workspace_router.get("/presets/list", response_model=list[str])
def list_presets() -> list[str]:
    """List available layout preset names."""
    return list(WorkspaceManager.LAYOUT_PRESETS.keys())


@workspace_router.get("/state/active", response_model=dict)
def get_active_workspace() -> dict:
    """Return the currently active workspace name."""
    name = _state_store.get_active()
    return {"active_workspace": name}


@workspace_router.post("/state/active", response_model=dict)
def set_active_workspace(body: dict) -> dict:
    """Set the active workspace by name."""
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")
    _state_store.set_active(name)
    return {"active_workspace": name, "status": "ok"}
