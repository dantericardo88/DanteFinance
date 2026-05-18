"""Multi-panel Bloomberg-style workspace — Dash (primary), Rich (fallback), plain text (final fallback).

Dimension 091: Multi-panel workspace (score 7 → 9).

Usage:
    python -m sentinel.ui.workspace_v3                    # auto-selects best renderer
    python -m sentinel.ui.workspace_v3 --ticker MSFT      # deep-dive on MSFT
    python -m sentinel.ui.workspace_v3 --layout TRADING_DESK --port 8051
"""
from __future__ import annotations

import base64
import dataclasses
import datetime
import http.server
import json
import logging
import os
import pathlib
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)

# ── Optional dependency guards ─────────────────────────────────────────────────

try:
    import dash
    from dash import Dash, Input, Output, dcc, html
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel as RichPanel
    from rich.layout import Layout
    from rich.console import Console
    from rich.columns import Columns
    from rich.text import Text
    from rich.style import Style
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from fastapi import FastAPI
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Enums ──────────────────────────────────────────────────────────────────────


class PanelType(Enum):
    PRICE_CHART = "price_chart"
    FUNDAMENTALS = "fundamentals"
    NEWS_FEED = "news_feed"
    OPTIONS_CHAIN = "options_chain"
    PORTFOLIO = "portfolio"
    RISK_DASHBOARD = "risk_dashboard"
    ORDER_BOOK = "order_book"
    MACRO_CALENDAR = "macro_calendar"
    SENTIMENT = "sentiment"
    SCREENER = "screener"
    FACTOR_EXPOSURE = "factor_exposure"
    EARNINGS = "earnings"
    WATCHLIST = "watchlist"
    CUSTOM = "custom"


class RendererMode(Enum):
    DASH = "dash"
    RICH = "rich"
    PLAIN = "plain"


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class Panel:
    id: str
    panel_type: PanelType
    title: str
    ticker: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    width_pct: float = 50.0          # percentage of workspace width
    height_pct: float = 50.0         # percentage of workspace height
    refresh_interval_seconds: int = 60
    last_updated: Optional[datetime.datetime] = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["panel_type"] = self.panel_type.value
        if self.last_updated:
            d["last_updated"] = self.last_updated.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        d = dict(d)
        d["panel_type"] = PanelType(d["panel_type"])
        if d.get("last_updated"):
            d["last_updated"] = datetime.datetime.fromisoformat(d["last_updated"])
        return cls(**d)


@dataclass
class WorkspaceLayout:
    name: str
    description: str
    panels: List[Panel] = field(default_factory=list)


@dataclass
class Workspace:
    id: str
    layout: WorkspaceLayout
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "layout": {
                "name": self.layout.name,
                "description": self.layout.description,
                "panels": [p.to_dict() for p in self.layout.panels],
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        panels = [Panel.from_dict(p) for p in d["layout"]["panels"]]
        layout = WorkspaceLayout(
            name=d["layout"]["name"],
            description=d["layout"]["description"],
            panels=panels,
        )
        return cls(
            id=d["id"],
            layout=layout,
            created_at=datetime.datetime.fromisoformat(d["created_at"]),
            metadata=d.get("metadata", {}),
        )


# ── Layout Presets ─────────────────────────────────────────────────────────────


class LayoutPresets:
    """Factory for predefined workspace layouts."""

    @staticmethod
    def equity_deep_dive(ticker: str = "AAPL") -> WorkspaceLayout:
        return WorkspaceLayout(
            name="EQUITY_DEEP_DIVE",
            description="4-panel equity deep-dive: price chart, fundamentals, news, options chain",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.PRICE_CHART,
                    title=f"Price Chart — {ticker}",
                    ticker=ticker,
                    width_pct=50.0,
                    height_pct=50.0,
                    refresh_interval_seconds=30,
                    config={"timeframe": "1d", "n_bars": 252},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.FUNDAMENTALS,
                    title=f"Fundamentals — {ticker}",
                    ticker=ticker,
                    width_pct=25.0,
                    height_pct=50.0,
                    refresh_interval_seconds=300,
                    config={},
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.NEWS_FEED,
                    title=f"News — {ticker}",
                    ticker=ticker,
                    width_pct=25.0,
                    height_pct=50.0,
                    refresh_interval_seconds=120,
                    config={"n": 10},
                ),
                Panel(
                    id="p4",
                    panel_type=PanelType.OPTIONS_CHAIN,
                    title=f"Options Chain — {ticker}",
                    ticker=ticker,
                    width_pct=100.0,
                    height_pct=50.0,
                    refresh_interval_seconds=60,
                    config={},
                ),
            ],
        )

    @staticmethod
    def portfolio_monitor() -> WorkspaceLayout:
        return WorkspaceLayout(
            name="PORTFOLIO_MONITOR",
            description="Portfolio positions, risk dashboard, factor exposure, macro calendar",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.PORTFOLIO,
                    title="Portfolio Positions",
                    width_pct=100.0,
                    height_pct=50.0,
                    refresh_interval_seconds=60,
                    config={},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.RISK_DASHBOARD,
                    title="Risk Dashboard",
                    width_pct=50.0,
                    height_pct=25.0,
                    refresh_interval_seconds=120,
                    config={},
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.FACTOR_EXPOSURE,
                    title="Factor Exposure",
                    width_pct=50.0,
                    height_pct=25.0,
                    refresh_interval_seconds=300,
                    config={},
                ),
                Panel(
                    id="p4",
                    panel_type=PanelType.MACRO_CALENDAR,
                    title="Macro Calendar",
                    width_pct=100.0,
                    height_pct=25.0,
                    refresh_interval_seconds=600,
                    config={"days": 7},
                ),
            ],
        )

    @staticmethod
    def trading_desk(ticker: str = "SPY") -> WorkspaceLayout:
        return WorkspaceLayout(
            name="TRADING_DESK",
            description="Order book, price chart, options flow, news feed",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.ORDER_BOOK,
                    title=f"Order Book — {ticker}",
                    ticker=ticker,
                    width_pct=30.0,
                    height_pct=50.0,
                    refresh_interval_seconds=5,
                    config={},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.PRICE_CHART,
                    title=f"Price Chart — {ticker}",
                    ticker=ticker,
                    width_pct=40.0,
                    height_pct=50.0,
                    refresh_interval_seconds=15,
                    config={"timeframe": "5m", "n_bars": 78},
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.OPTIONS_CHAIN,
                    title=f"Options Flow — {ticker}",
                    ticker=ticker,
                    width_pct=30.0,
                    height_pct=50.0,
                    refresh_interval_seconds=30,
                    config={},
                ),
                Panel(
                    id="p4",
                    panel_type=PanelType.NEWS_FEED,
                    title="News Feed",
                    ticker=ticker,
                    width_pct=100.0,
                    height_pct=50.0,
                    refresh_interval_seconds=60,
                    config={"n": 15},
                ),
            ],
        )

    @staticmethod
    def macro_watch() -> WorkspaceLayout:
        return WorkspaceLayout(
            name="MACRO_WATCH",
            description="Yield curves, global indices, FX rates, economic calendar",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.PRICE_CHART,
                    title="Yield Curves",
                    ticker="^TNX",
                    width_pct=33.0,
                    height_pct=50.0,
                    refresh_interval_seconds=120,
                    config={"chart_type": "yield_curve"},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.WATCHLIST,
                    title="Global Indices",
                    width_pct=34.0,
                    height_pct=50.0,
                    refresh_interval_seconds=60,
                    config={
                        "tickers": ["^GSPC", "^FTSE", "^N225", "^HSI", "^GDAXI", "^FCHI"]
                    },
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.WATCHLIST,
                    title="FX Rates",
                    width_pct=33.0,
                    height_pct=50.0,
                    refresh_interval_seconds=30,
                    config={
                        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCNH=X", "AUDUSD=X"]
                    },
                ),
                Panel(
                    id="p4",
                    panel_type=PanelType.MACRO_CALENDAR,
                    title="Economic Calendar",
                    width_pct=100.0,
                    height_pct=50.0,
                    refresh_interval_seconds=900,
                    config={"days": 7},
                ),
            ],
        )

    @staticmethod
    def crypto_dashboard() -> WorkspaceLayout:
        return WorkspaceLayout(
            name="CRYPTO_DASHBOARD",
            description="BTC/ETH prices, on-chain metrics, DeFi TVL, Fear & Greed",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.PRICE_CHART,
                    title="BTC / ETH Prices",
                    ticker="BTC-USD",
                    width_pct=100.0,
                    height_pct=50.0,
                    refresh_interval_seconds=30,
                    config={"tickers": ["BTC-USD", "ETH-USD"], "n_bars": 90},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.CUSTOM,
                    title="On-Chain Metrics",
                    width_pct=100.0,
                    height_pct=20.0,
                    refresh_interval_seconds=300,
                    config={"source": "glassnode_free"},
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.CUSTOM,
                    title="DeFi TVL",
                    width_pct=50.0,
                    height_pct=30.0,
                    refresh_interval_seconds=300,
                    config={"source": "defillama"},
                ),
                Panel(
                    id="p4",
                    panel_type=PanelType.SENTIMENT,
                    title="Fear & Greed Index",
                    width_pct=50.0,
                    height_pct=30.0,
                    refresh_interval_seconds=600,
                    config={"source": "alternative_me"},
                ),
            ],
        )

    @staticmethod
    def screener_workspace() -> WorkspaceLayout:
        return WorkspaceLayout(
            name="SCREENER_WORKSPACE",
            description="Fundamental screener, technical screener, NL screener",
            panels=[
                Panel(
                    id="p1",
                    panel_type=PanelType.SCREENER,
                    title="Fundamental Screener",
                    width_pct=40.0,
                    height_pct=100.0,
                    refresh_interval_seconds=300,
                    config={"mode": "fundamental"},
                ),
                Panel(
                    id="p2",
                    panel_type=PanelType.SCREENER,
                    title="Technical Screener",
                    width_pct=40.0,
                    height_pct=100.0,
                    refresh_interval_seconds=60,
                    config={"mode": "technical"},
                ),
                Panel(
                    id="p3",
                    panel_type=PanelType.SCREENER,
                    title="Natural Language Screener",
                    width_pct=20.0,
                    height_pct=100.0,
                    refresh_interval_seconds=120,
                    config={"mode": "nl"},
                ),
            ],
        )

    PRESET_MAP: Dict[str, str] = {
        "1": "EQUITY_DEEP_DIVE",
        "2": "PORTFOLIO_MONITOR",
        "3": "TRADING_DESK",
        "4": "MACRO_WATCH",
        "5": "CRYPTO_DASHBOARD",
        "6": "SCREENER_WORKSPACE",
    }

    @classmethod
    def get(cls, name: str, ticker: str = "AAPL") -> WorkspaceLayout:
        name_upper = name.upper()
        if name_upper == "EQUITY_DEEP_DIVE":
            return cls.equity_deep_dive(ticker)
        elif name_upper == "PORTFOLIO_MONITOR":
            return cls.portfolio_monitor()
        elif name_upper == "TRADING_DESK":
            return cls.trading_desk(ticker)
        elif name_upper == "MACRO_WATCH":
            return cls.macro_watch()
        elif name_upper == "CRYPTO_DASHBOARD":
            return cls.crypto_dashboard()
        elif name_upper == "SCREENER_WORKSPACE":
            return cls.screener_workspace()
        else:
            raise ValueError(f"Unknown layout preset: {name!r}")


# ── Workspace Manager ──────────────────────────────────────────────────────────


class WorkspaceManager:
    """Create, mutate, persist, and load workspaces."""

    def __init__(self, storage_dir: str = "sentinel/data/workspaces"):
        self.storage_dir = storage_dir
        self._workspaces: Dict[str, Workspace] = {}
        os.makedirs(storage_dir, exist_ok=True)

    def create_workspace(self, layout_name: str, ticker: str = "AAPL") -> Workspace:
        layout = LayoutPresets.get(layout_name, ticker)
        ws = Workspace(id=str(uuid.uuid4()), layout=layout)
        self._workspaces[ws.id] = ws
        logger.info("Created workspace %s with layout %s", ws.id, layout_name)
        return ws

    def add_panel(self, workspace_id: str, panel: Panel) -> None:
        ws = self._get(workspace_id)
        ws.layout.panels.append(panel)
        logger.debug("Added panel %s to workspace %s", panel.id, workspace_id)

    def remove_panel(self, workspace_id: str, panel_id: str) -> None:
        ws = self._get(workspace_id)
        ws.layout.panels = [p for p in ws.layout.panels if p.id != panel_id]

    def resize_panel(
        self,
        workspace_id: str,
        panel_id: str,
        width_pct: float,
        height_pct: float,
    ) -> None:
        ws = self._get(workspace_id)
        for p in ws.layout.panels:
            if p.id == panel_id:
                p.width_pct = max(5.0, min(100.0, width_pct))
                p.height_pct = max(5.0, min(100.0, height_pct))
                return
        raise KeyError(f"Panel {panel_id} not found in workspace {workspace_id}")

    def save_workspace(self, workspace_id: str, path: Optional[str] = None) -> str:
        ws = self._get(workspace_id)
        if path is None:
            path = os.path.join(self.storage_dir, f"{workspace_id}.json")
        with open(path, "w") as fh:
            json.dump(ws.to_dict(), fh, indent=2)
        logger.info("Saved workspace %s to %s", workspace_id, path)
        return path

    def load_workspace(self, path: str) -> Workspace:
        with open(path) as fh:
            data = json.load(fh)
        ws = Workspace.from_dict(data)
        self._workspaces[ws.id] = ws
        return ws

    def list_saved_workspaces(self) -> List[str]:
        try:
            return [
                f for f in os.listdir(self.storage_dir) if f.endswith(".json")
            ]
        except FileNotFoundError:
            return []

    def _get(self, workspace_id: str) -> Workspace:
        if workspace_id not in self._workspaces:
            raise KeyError(f"Workspace {workspace_id} not found")
        return self._workspaces[workspace_id]


# ── Panel State Persistence ────────────────────────────────────────────────────

_SENTINEL_STATE_DIR = pathlib.Path.home() / ".sentinel"
_WORKSPACE_STATE_FILE = _SENTINEL_STATE_DIR / "workspace_state.json"


def save_workspace_state(workspace: Workspace, path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Persist workspace layout to ~/.sentinel/workspace_state.json (or custom path)."""
    target = path or _WORKSPACE_STATE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(workspace.to_dict(), fh, indent=2, default=str)
    logger.debug("Workspace state saved to %s", target)
    return target


def load_workspace_state(path: Optional[pathlib.Path] = None) -> Optional[Workspace]:
    """Load workspace layout from ~/.sentinel/workspace_state.json (or custom path)."""
    target = path or _WORKSPACE_STATE_FILE
    if not target.exists():
        return None
    try:
        with open(target, encoding="utf-8") as fh:
            data = json.load(fh)
        ws = Workspace.from_dict(data)
        logger.debug("Workspace state loaded from %s", target)
        return ws
    except Exception as exc:
        logger.warning("Failed to load workspace state from %s: %s", target, exc)
        return None


# ── Workspace Exporter ─────────────────────────────────────────────────────────


class WorkspaceExporter:
    """
    Export workspace configs and data snapshots.

    Supports:
      - JSON export: panel configs + data snapshots
      - HTML export: self-contained report with embedded base64 PNG charts
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._renderer = PanelRenderer(mode=RendererMode.PLAIN)

    def export_json(self, path: str) -> str:
        """
        Save all panel configs + plain-text data snapshots to a JSON file.
        Returns the path written.
        """
        ws_dict = self.workspace.to_dict()
        snapshots: Dict[str, str] = {}
        for panel in self.workspace.layout.panels:
            try:
                content = self._renderer.render_panel(panel)
                snapshots[panel.id] = str(content)
            except Exception as exc:
                snapshots[panel.id] = f"ERROR: {exc}"

        export_data = {
            "workspace": ws_dict,
            "snapshots": snapshots,
            "exported_at": datetime.datetime.utcnow().isoformat(),
        }
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(export_data, fh, indent=2, default=str)
        logger.info("Workspace exported to JSON: %s", out_path)
        return str(out_path)

    def export_html(self, path: str) -> str:
        """
        Generate a self-contained HTML report.
        Tries to embed matplotlib PNG charts as base64; falls back to ASCII
        sparklines when matplotlib is unavailable.
        Returns the path written.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            HAS_MPL = True
        except ImportError:
            HAS_MPL = False

        panels_html = []
        for panel in self.workspace.layout.panels:
            chart_html = ""
            # Attempt a simple matplotlib chart for price-chart panels
            if HAS_MPL and panel.panel_type == PanelType.PRICE_CHART and panel.ticker:
                try:
                    df = _fetch_ohlcv(panel.ticker, period="3mo", interval="1d")
                    if df is not None and len(df) > 5:
                        closes = df["Close"].dropna().tolist()
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.plot(closes, color="#00b894", linewidth=1.2)
                        ax.set_facecolor("#1e272e")
                        fig.patch.set_facecolor("#1e272e")
                        ax.tick_params(colors="#636e72", labelsize=7)
                        ax.set_title(f"{panel.ticker} Close", color="#dfe6e9", fontsize=9)
                        buf = base64.b64encode(_fig_to_png(fig)).decode("utf-8")
                        plt.close(fig)
                        chart_html = f'<img src="data:image/png;base64,{buf}" style="width:100%;max-width:560px;">'
                except Exception:
                    pass

            content = self._renderer.render_panel(panel)
            content_text = str(content).replace("<", "&lt;").replace(">", "&gt;")
            panels_html.append(
                f"""<div class="panel">
  <div class="panel-header">{panel.title}</div>
  <div class="panel-body">
    {chart_html}
    <pre>{content_text}</pre>
  </div>
</div>"""
            )

        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        body = "\n".join(panels_html)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENTINEL — {self.workspace.layout.name}</title>
<style>
  body {{ background:#1a1a2e; color:#dfe6e9; font-family:'Roboto Mono',monospace; margin:0; padding:16px; }}
  h1 {{ color:#74b9ff; font-size:18px; margin-bottom:4px; }}
  .meta {{ color:#636e72; font-size:11px; margin-bottom:16px; }}
  .panel {{ background:#1e272e; border:1px solid #2d3436; border-radius:4px;
            margin-bottom:12px; overflow:hidden; }}
  .panel-header {{ background:#2d3436; padding:6px 12px; font-size:12px;
                   font-weight:600; color:#74b9ff; }}
  .panel-body {{ padding:8px 12px; }}
  pre {{ font-size:11px; white-space:pre-wrap; word-break:break-word; margin:0; color:#b2bec3; }}
</style>
</head>
<body>
<h1>SENTINEL — {self.workspace.layout.name}</h1>
<div class="meta">Generated: {now_str} | Workspace ID: {self.workspace.id}</div>
{body}
</body>
</html>"""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Workspace exported to HTML: %s", out_path)
        return str(out_path)


def _fig_to_png(fig: Any) -> bytes:
    """Render a matplotlib figure to PNG bytes."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return buf.read()


# ── Panel Refresh Scheduler ────────────────────────────────────────────────────


class PanelScheduler:
    """
    Tracks panel TTLs and determines which panels are stale.

    Usage (pure computation — no threads required for tests):
        scheduler = PanelScheduler(workspace)
        scheduler.tick()          # increments refresh counter for stale panels
        scheduler.refresh_counts  # Dict[panel_id, int]
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.refresh_counts: Dict[str, int] = {
            p.id: 0 for p in workspace.layout.panels
        }
        self._last_check: datetime.datetime = datetime.datetime.utcnow()

    def stale_panels(self, now: Optional[datetime.datetime] = None) -> List[Panel]:
        """Return panels whose last_updated is older than their TTL (or never updated)."""
        now = now or datetime.datetime.utcnow()
        stale = []
        for panel in self.workspace.layout.panels:
            if panel.last_updated is None:
                stale.append(panel)
            else:
                age = (now - panel.last_updated).total_seconds()
                if age >= panel.refresh_interval_seconds:
                    stale.append(panel)
        return stale

    def tick(self, now: Optional[datetime.datetime] = None) -> List[str]:
        """
        Advance the scheduler clock. Increments refresh_counts for panels
        that are past their TTL.  Returns the list of panel IDs that fired.
        Does NOT actually fetch data (pure state machine).
        """
        now = now or datetime.datetime.utcnow()
        fired: List[str] = []
        for panel in self.workspace.layout.panels:
            if panel.last_updated is None:
                self.refresh_counts[panel.id] = self.refresh_counts.get(panel.id, 0) + 1
                fired.append(panel.id)
            else:
                age = (now - panel.last_updated).total_seconds()
                if age >= panel.refresh_interval_seconds:
                    self.refresh_counts[panel.id] = self.refresh_counts.get(panel.id, 0) + 1
                    fired.append(panel.id)
        self._last_check = now
        return fired

    def mark_refreshed(self, panel_id: str, when: Optional[datetime.datetime] = None) -> None:
        """Mark a panel as just-refreshed so it won't fire again until next TTL."""
        now = when or datetime.datetime.utcnow()
        for panel in self.workspace.layout.panels:
            if panel.id == panel_id:
                panel.last_updated = now
                return


# ── Data Fetchers (internal helpers) ──────────────────────────────────────────


def _fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[Any]:
    """Fetch OHLCV via yfinance; return DataFrame or None."""
    if not HAS_YF or not HAS_PANDAS:
        return None
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            return None
        return data
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def _fetch_info(ticker: str) -> dict:
    """Fetch ticker info dict via yfinance."""
    if not HAS_YF:
        return {}
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception as exc:
        logger.warning("yfinance info failed for %s: %s", ticker, exc)
        return {}


def _fetch_news_rss(ticker: str, n: int = 10) -> List[Dict[str, str]]:
    """Fetch news headlines from Yahoo Finance RSS."""
    if not HAS_REQUESTS:
        return []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")
            title = title_el.text if title_el is not None else "No title"
            link = link_el.text if link_el is not None else ""
            pub = pubdate_el.text if pubdate_el is not None else ""
            items.append({"title": title, "link": link, "pubDate": pub})
            if len(items) >= n:
                break
        return items
    except Exception as exc:
        logger.warning("RSS fetch failed for %s: %s", ticker, exc)
        return []


def _simple_sentiment(text: str) -> float:
    """Very simple lexicon sentiment: returns -1..1."""
    positive = {"surge", "gain", "rise", "up", "beat", "record", "growth",
                 "profit", "strong", "buy", "upgrade", "outperform", "rally",
                 "soar", "jump", "increase"}
    negative = {"fall", "drop", "decline", "down", "miss", "loss", "weak",
                 "sell", "downgrade", "underperform", "crash", "plunge",
                 "slump", "cut", "fear", "risk", "warn"}
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def _compute_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Compute RSI from a list of close prices."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    if not gains:
        return 0.0
    if not losses:
        return 100.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def _sparkline(values: List[float], width: int = 40) -> str:
    """Generate ASCII sparkline from a list of values."""
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng == 0:
        return blocks[4] * min(len(values), width)
    step = len(values) / width
    result = []
    for i in range(min(len(values), width)):
        idx = int(i * step)
        v = values[min(idx, len(values) - 1)]
        level = int((v - mn) / rng * (len(blocks) - 1))
        result.append(blocks[level])
    return "".join(result)


# ── Panel Renderer ─────────────────────────────────────────────────────────────


class PanelRenderer:
    """Render each panel type to Dash component, Rich widget, or plain text."""

    def __init__(self, mode: RendererMode = RendererMode.PLAIN):
        self.mode = mode

    # ── Price Chart ──────────────────────────────────────────────────────────

    def render_price_chart(
        self,
        ticker: str,
        timeframe: str = "1d",
        n_bars: int = 252,
        chart_type: str = "candlestick",
        extra_tickers: Optional[List[str]] = None,
    ) -> Any:
        period_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "3mo",
                      "1d": "1y", "1wk": "5y", "1mo": "10y"}
        period = period_map.get(timeframe, "1y")
        df = _fetch_ohlcv(ticker, period=period, interval=timeframe)

        if self.mode == RendererMode.DASH and HAS_PLOTLY and HAS_DASH:
            if df is not None and len(df) > 0:
                df_tail = df.tail(n_bars)
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.03,
                )
                fig.add_trace(
                    go.Candlestick(
                        x=df_tail.index,
                        open=df_tail["Open"],
                        high=df_tail["High"],
                        low=df_tail["Low"],
                        close=df_tail["Close"],
                        name=ticker,
                        increasing_line_color="#00b894",
                        decreasing_line_color="#d63031",
                    ),
                    row=1, col=1,
                )
                if "Volume" in df_tail.columns:
                    colors = [
                        "#00b894" if df_tail["Close"].iloc[i] >= df_tail["Open"].iloc[i]
                        else "#d63031"
                        for i in range(len(df_tail))
                    ]
                    fig.add_trace(
                        go.Bar(
                            x=df_tail.index,
                            y=df_tail["Volume"],
                            marker_color=colors,
                            name="Volume",
                            showlegend=False,
                        ),
                        row=2, col=1,
                    )
                # 50-day and 200-day MA
                for w, color in [(50, "#fdcb6e"), (200, "#74b9ff")]:
                    if len(df_tail) >= w:
                        fig.add_trace(
                            go.Scatter(
                                x=df_tail.index,
                                y=df_tail["Close"].rolling(w).mean(),
                                mode="lines",
                                name=f"MA{w}",
                                line=dict(color=color, width=1),
                            ),
                            row=1, col=1,
                        )
                fig.update_layout(
                    template="plotly_dark",
                    title=f"{ticker} — {timeframe}",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=40, r=20, t=40, b=20),
                    height=None,
                    legend=dict(orientation="h", y=1.02),
                )
                return dcc.Graph(figure=fig, style={"height": "100%"}, config={"displayModeBar": False})
            else:
                return html.Div(f"No data for {ticker}", style={"color": "#636e72", "padding": "20px"})

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title=f"{ticker} Price Chart", show_header=True, header_style="bold cyan")
            t.add_column("Metric", style="cyan")
            t.add_column("Value", justify="right")
            if df is not None and len(df) > 0:
                close_vals = df["Close"].dropna().tolist()[-n_bars:]
                last = close_vals[-1] if close_vals else float("nan")
                chg = (close_vals[-1] - close_vals[-2]) / close_vals[-2] * 100 if len(close_vals) >= 2 else 0
                spark = _sparkline(close_vals, width=30)
                t.add_row("Ticker", ticker)
                t.add_row("Last Price", f"${last:,.2f}")
                chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
                t.add_row("1d Change", Text(chg_str, style="green" if chg >= 0 else "red"))
                t.add_row("Sparkline", spark)
                if "Volume" in df.columns:
                    vol = df["Volume"].iloc[-1]
                    t.add_row("Volume", f"{vol:,.0f}")
            else:
                t.add_row("Status", "No data available")
            return t

        else:
            lines = [f"=== PRICE CHART: {ticker} ({timeframe}) ==="]
            if df is not None and len(df) > 0:
                close_vals = df["Close"].dropna().tolist()[-n_bars:]
                last = close_vals[-1] if close_vals else float("nan")
                chg = (close_vals[-1] - close_vals[-2]) / close_vals[-2] * 100 if len(close_vals) >= 2 else 0
                spark = _sparkline(close_vals, width=40)
                lines.append(f"  Last:   ${last:,.2f}")
                sign = "+" if chg >= 0 else ""
                lines.append(f"  Change: {sign}{chg:.2f}%")
                lines.append(f"  {spark}")
            else:
                lines.append("  [No price data available]")
            return "\n".join(lines)

    # ── Fundamentals ─────────────────────────────────────────────────────────

    def render_fundamentals(self, ticker: str) -> Any:
        info = _fetch_info(ticker)
        keys = {
            "trailingPE": "P/E (TTM)",
            "forwardPE": "P/E (Fwd)",
            "enterpriseToEbitda": "EV/EBITDA",
            "totalRevenue": "Revenue",
            "trailingEps": "EPS (TTM)",
            "grossMargins": "Gross Margin",
            "operatingMargins": "Op. Margin",
            "returnOnEquity": "ROE",
            "marketCap": "Market Cap",
            "dividendYield": "Div. Yield",
            "beta": "Beta",
            "fiftyTwoWeekHigh": "52W High",
            "fiftyTwoWeekLow": "52W Low",
        }

        def _fmt(k: str, v: Any) -> str:
            if v is None:
                return "N/A"
            if k in ("totalRevenue", "marketCap"):
                if v >= 1e12:
                    return f"${v/1e12:.2f}T"
                elif v >= 1e9:
                    return f"${v/1e9:.2f}B"
                elif v >= 1e6:
                    return f"${v/1e6:.2f}M"
                return f"${v:,.0f}"
            elif k in ("grossMargins", "operatingMargins", "returnOnEquity", "dividendYield"):
                return f"{v * 100:.1f}%"
            elif isinstance(v, float):
                return f"{v:.2f}x" if k.endswith("PE") or k == "enterpriseToEbitda" else f"{v:.2f}"
            return str(v)

        rows = [(label, _fmt(k, info.get(k))) for k, label in keys.items()]

        if self.mode == RendererMode.DASH and HAS_DASH:
            cells = []
            for label, val in rows:
                cells.append(
                    html.Div([
                        html.Span(label, style={"color": "#636e72", "fontSize": "11px"}),
                        html.Br(),
                        html.Span(val, style={"color": "#dfe6e9", "fontWeight": "600", "fontSize": "13px"}),
                    ], style={"padding": "6px 10px", "borderBottom": "1px solid #2d3436"})
                )
            return html.Div(
                cells,
                style={"overflowY": "auto", "height": "100%", "fontFamily": "monospace"},
            )

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title=f"Fundamentals — {ticker}", show_header=True, header_style="bold magenta")
            t.add_column("Metric", style="cyan")
            t.add_column("Value", justify="right")
            for label, val in rows:
                t.add_row(label, val)
            return t

        else:
            lines = [f"=== FUNDAMENTALS: {ticker} ==="]
            for label, val in rows:
                lines.append(f"  {label:<20} {val:>12}")
            return "\n".join(lines)

    # ── News Feed ─────────────────────────────────────────────────────────────

    def render_news_feed(self, ticker: str, n: int = 10) -> Any:
        articles = _fetch_news_rss(ticker, n=n)
        if not articles:
            articles = [{"title": f"No live news available for {ticker}. Check network.", "link": "", "pubDate": ""}]

        scored = [(a["title"], a.get("pubDate", "")[:16], _simple_sentiment(a["title"])) for a in articles]

        if self.mode == RendererMode.DASH and HAS_DASH:
            items = []
            for title, pub, score in scored:
                color = "#00b894" if score > 0.05 else "#d63031" if score < -0.05 else "#636e72"
                items.append(
                    html.Div([
                        html.Span(f"{'▲' if score > 0.05 else '▼' if score < -0.05 else '◆'} ",
                                  style={"color": color}),
                        html.Span(title[:90], style={"fontSize": "12px", "color": "#dfe6e9"}),
                        html.Br(),
                        html.Span(pub, style={"color": "#636e72", "fontSize": "10px"}),
                        html.Span(f" | Sentiment: {score:+.2f}",
                                  style={"color": color, "fontSize": "10px"}),
                    ], style={"padding": "6px 8px", "borderBottom": "1px solid #2d3436"})
                )
            return html.Div(items, style={"overflowY": "auto", "height": "100%", "fontFamily": "monospace"})

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title=f"News — {ticker}", show_header=True, header_style="bold yellow")
            t.add_column("Headline", max_width=55, no_wrap=False)
            t.add_column("Date", width=12)
            t.add_column("Sent", width=6, justify="right")
            for title, pub, score in scored:
                color = "green" if score > 0.05 else "red" if score < -0.05 else "white"
                t.add_row(title[:55], pub, Text(f"{score:+.2f}", style=color))
            return t

        else:
            lines = [f"=== NEWS FEED: {ticker} ==="]
            for title, pub, score in scored:
                sign = "+" if score > 0.05 else "-" if score < -0.05 else "~"
                lines.append(f"  [{sign}] {title[:70]}")
                lines.append(f"       {pub}  sentiment={score:+.2f}")
            return "\n".join(lines)

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def render_portfolio(self, holdings: Optional[Dict[str, dict]] = None) -> Any:
        if holdings is None:
            holdings = {
                "AAPL":  {"qty": 100, "avg_cost": 145.20},
                "MSFT":  {"qty":  50, "avg_cost": 310.50},
                "GOOGL": {"qty":  20, "avg_cost": 2750.00},
                "NVDA":  {"qty":  30, "avg_cost": 420.00},
                "BRK-B": {"qty":  40, "avg_cost": 285.00},
            }

        rows = []
        total_value = 0.0
        total_cost = 0.0
        for ticker, pos in holdings.items():
            qty = pos.get("qty", 0)
            cost = pos.get("avg_cost", 0.0)
            info = _fetch_info(ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice") or cost
            value = qty * price
            cost_basis = qty * cost
            pnl = value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0.0
            total_value += value
            total_cost += cost_basis
            rows.append((ticker, qty, cost, price, value, pnl, pnl_pct))

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0.0

        if self.mode == RendererMode.DASH and HAS_DASH:
            header = html.Tr([
                html.Th(c, style={"padding": "6px", "color": "#74b9ff", "textAlign": "right" if i > 0 else "left"})
                for i, c in enumerate(["Ticker", "Qty", "Avg Cost", "Price", "Value", "P&L", "P&L %"])
            ])
            body_rows = []
            for ticker, qty, cost, price, value, pnl, pnl_pct in rows:
                color = "#00b894" if pnl >= 0 else "#d63031"
                body_rows.append(html.Tr([
                    html.Td(ticker, style={"padding": "4px 6px", "color": "#dfe6e9"}),
                    html.Td(f"{qty:,}", style={"padding": "4px 6px", "textAlign": "right", "color": "#dfe6e9"}),
                    html.Td(f"${cost:,.2f}", style={"padding": "4px 6px", "textAlign": "right", "color": "#b2bec3"}),
                    html.Td(f"${price:,.2f}", style={"padding": "4px 6px", "textAlign": "right", "color": "#dfe6e9"}),
                    html.Td(f"${value:,.0f}", style={"padding": "4px 6px", "textAlign": "right", "color": "#dfe6e9"}),
                    html.Td(f"${pnl:+,.0f}", style={"padding": "4px 6px", "textAlign": "right", "color": color}),
                    html.Td(f"{pnl_pct:+.1f}%", style={"padding": "4px 6px", "textAlign": "right", "color": color}),
                ]))
            summary_color = "#00b894" if total_pnl >= 0 else "#d63031"
            body_rows.append(html.Tr([
                html.Td("TOTAL", style={"padding": "6px", "fontWeight": "bold", "color": "#74b9ff"}),
                html.Td("", style={"padding": "6px"}),
                html.Td("", style={"padding": "6px"}),
                html.Td("", style={"padding": "6px"}),
                html.Td(f"${total_value:,.0f}", style={"padding": "6px", "textAlign": "right", "fontWeight": "bold", "color": "#dfe6e9"}),
                html.Td(f"${total_pnl:+,.0f}", style={"padding": "6px", "textAlign": "right", "fontWeight": "bold", "color": summary_color}),
                html.Td(f"{total_pnl_pct:+.1f}%", style={"padding": "6px", "textAlign": "right", "fontWeight": "bold", "color": summary_color}),
            ]))
            return html.Table(
                [html.Thead(header), html.Tbody(body_rows)],
                style={"width": "100%", "fontFamily": "monospace", "fontSize": "13px",
                       "borderCollapse": "collapse"},
            )

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title="Portfolio", show_header=True, header_style="bold blue")
            for col in ["Ticker", "Qty", "Avg Cost", "Price", "Value", "P&L", "P&L %"]:
                t.add_column(col, justify="right" if col != "Ticker" else "left")
            for ticker, qty, cost, price, value, pnl, pnl_pct in rows:
                color = "green" if pnl >= 0 else "red"
                t.add_row(
                    ticker, str(qty), f"${cost:,.2f}", f"${price:,.2f}",
                    f"${value:,.0f}",
                    Text(f"${pnl:+,.0f}", style=color),
                    Text(f"{pnl_pct:+.1f}%", style=color),
                )
            t.add_row("TOTAL", "", "", "",
                      f"${total_value:,.0f}",
                      Text(f"${total_pnl:+,.0f}", style="green" if total_pnl >= 0 else "red"),
                      Text(f"{total_pnl_pct:+.1f}%", style="green" if total_pnl_pct >= 0 else "red"))
            return t

        else:
            lines = ["=== PORTFOLIO ==="]
            header = f"  {'Ticker':<8} {'Qty':>6} {'AvgCost':>10} {'Price':>10} {'Value':>12} {'P&L':>10} {'P&L%':>7}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for ticker, qty, cost, price, value, pnl, pnl_pct in rows:
                lines.append(
                    f"  {ticker:<8} {qty:>6} ${cost:>9,.2f} ${price:>9,.2f}"
                    f" ${value:>11,.0f} ${pnl:>+9,.0f} {pnl_pct:>+6.1f}%"
                )
            lines.append("  " + "-" * (len(header) - 2))
            lines.append(
                f"  {'TOTAL':<8} {'':>6} {'':>10} {'':>10}"
                f" ${total_value:>11,.0f} ${total_pnl:>+9,.0f} {total_pnl_pct:>+6.1f}%"
            )
            return "\n".join(lines)

    # ── Risk Dashboard ────────────────────────────────────────────────────────

    def render_risk_dashboard(self, holdings: Optional[Dict[str, dict]] = None) -> Any:
        import math

        if holdings is None:
            holdings = {"SPY": {"qty": 100, "avg_cost": 450.0}}

        # Compute simple portfolio metrics from price series
        tickers = list(holdings.keys())
        returns_matrix: Dict[str, List[float]] = {}
        for t in tickers:
            df = _fetch_ohlcv(t, period="1y", interval="1d")
            if df is not None and len(df) > 5 and HAS_PANDAS:
                import pandas as pd
                closes = df["Close"].dropna()
                rets = closes.pct_change().dropna().tolist()
                returns_matrix[t] = rets
            else:
                returns_matrix[t] = []

        # Compute portfolio daily returns (equal-weighted fallback)
        min_len = min((len(v) for v in returns_matrix.values() if v), default=0)
        port_rets: List[float] = []
        if min_len > 10:
            n = len(tickers)
            for i in range(min_len):
                pr = sum(returns_matrix[t][-(min_len - i)] for t in tickers if returns_matrix[t]) / n
                port_rets.append(pr)

        # VaR / CVaR (95%)
        if port_rets:
            sorted_r = sorted(port_rets)
            cutoff = int(len(sorted_r) * 0.05)
            var_95 = abs(sorted_r[cutoff]) * 100
            cvar_95 = abs(sum(sorted_r[:cutoff]) / max(cutoff, 1)) * 100
        else:
            var_95 = cvar_95 = float("nan")

        # Max drawdown
        if port_rets:
            cumulative = [1.0]
            for r in port_rets:
                cumulative.append(cumulative[-1] * (1 + r))
            peak = cumulative[0]
            max_dd = 0.0
            for v in cumulative:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            max_dd *= 100
        else:
            max_dd = float("nan")

        # Sharpe ratio (annualised, rf=4%)
        if port_rets and len(port_rets) > 1:
            mean_r = sum(port_rets) / len(port_rets)
            variance = sum((r - mean_r) ** 2 for r in port_rets) / (len(port_rets) - 1)
            std_r = math.sqrt(variance)
            rf_daily = 0.04 / 252
            sharpe = ((mean_r - rf_daily) / std_r * math.sqrt(252)) if std_r > 0 else float("nan")
        else:
            sharpe = float("nan")

        metrics = [
            ("VaR (95%, 1-day)", f"{var_95:.2f}%"),
            ("CVaR (95%, 1-day)", f"{cvar_95:.2f}%"),
            ("Max Drawdown (1Y)", f"{max_dd:.2f}%"),
            ("Sharpe Ratio (1Y)", f"{sharpe:.2f}" if not math.isnan(sharpe) else "N/A"),
        ]

        if self.mode == RendererMode.DASH and HAS_DASH:
            cards = []
            colors = ["#d63031", "#e17055", "#e67e22", "#00b894"]
            for (label, val), color in zip(metrics, colors):
                cards.append(
                    html.Div([
                        html.Div(label, style={"color": "#b2bec3", "fontSize": "11px"}),
                        html.Div(val, style={"color": color, "fontSize": "22px", "fontWeight": "bold"}),
                    ], style={"padding": "12px", "border": f"1px solid {color}",
                              "borderRadius": "4px", "margin": "4px",
                              "backgroundColor": "#1e272e", "flex": "1"})
                )
            return html.Div(cards, style={"display": "flex", "flexWrap": "wrap", "height": "100%"})

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title="Risk Dashboard", show_header=True, header_style="bold red")
            t.add_column("Metric", style="cyan")
            t.add_column("Value", justify="right", style="bold")
            for label, val in metrics:
                t.add_row(label, val)
            return t

        else:
            lines = ["=== RISK DASHBOARD ==="]
            for label, val in metrics:
                lines.append(f"  {label:<25} {val:>10}")
            return "\n".join(lines)

    # ── Macro Calendar ────────────────────────────────────────────────────────

    def render_macro_calendar(self, days: int = 7) -> Any:
        # Attempt to import local module; graceful fallback
        events: List[dict] = []
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from sentinel.sma.economic_calendar_v3 import EconomicCalendarFetcher  # type: ignore
            cal = EconomicCalendarFetcher()
            raw = cal.fetch_upcoming(days=days)
            if raw:
                for e in raw[:20]:
                    events.append({
                        "date": str(e.get("date", "")),
                        "event": str(e.get("event", e.get("title", ""))),
                        "country": str(e.get("country", "US")),
                        "impact": str(e.get("impact", "M")),
                    })
        except Exception:
            pass

        if not events:
            # Synthetic upcoming events
            base = datetime.date.today()
            events = [
                {"date": str(base + datetime.timedelta(days=1)), "event": "Fed Minutes", "country": "US", "impact": "H"},
                {"date": str(base + datetime.timedelta(days=2)), "event": "CPI (MoM)", "country": "US", "impact": "H"},
                {"date": str(base + datetime.timedelta(days=3)), "event": "Initial Jobless Claims", "country": "US", "impact": "M"},
                {"date": str(base + datetime.timedelta(days=4)), "event": "PPI (MoM)", "country": "US", "impact": "M"},
                {"date": str(base + datetime.timedelta(days=5)), "event": "Retail Sales (MoM)", "country": "US", "impact": "H"},
                {"date": str(base + datetime.timedelta(days=6)), "event": "U. of Mich. Consumer Sentiment", "country": "US", "impact": "M"},
            ]

        impact_colors = {"H": "#d63031", "M": "#e17055", "L": "#00b894"}

        if self.mode == RendererMode.DASH and HAS_DASH:
            rows = []
            for ev in events:
                color = impact_colors.get(ev["impact"], "#636e72")
                rows.append(html.Tr([
                    html.Td(ev["date"], style={"padding": "4px 8px", "color": "#b2bec3", "fontSize": "12px"}),
                    html.Td(ev["country"], style={"padding": "4px 8px", "color": "#74b9ff", "fontSize": "12px"}),
                    html.Td(ev["event"], style={"padding": "4px 8px", "color": "#dfe6e9", "fontSize": "12px"}),
                    html.Td(ev["impact"], style={"padding": "4px 8px", "color": color, "fontWeight": "bold", "fontSize": "12px"}),
                ]))
            return html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th(c, style={"padding": "6px 8px", "color": "#74b9ff"})
                        for c in ["Date", "Country", "Event", "Impact"]
                    ])),
                    html.Tbody(rows),
                ],
                style={"width": "100%", "fontFamily": "monospace", "borderCollapse": "collapse"},
            )

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title=f"Macro Calendar ({days}d)", show_header=True, header_style="bold green")
            t.add_column("Date", width=12)
            t.add_column("Country", width=8)
            t.add_column("Event", max_width=40)
            t.add_column("Impact", width=6)
            for ev in events:
                color = {"H": "red", "M": "yellow", "L": "green"}.get(ev["impact"], "white")
                t.add_row(ev["date"], ev["country"], ev["event"],
                          Text(ev["impact"], style=color))
            return t

        else:
            lines = [f"=== MACRO CALENDAR (next {days} days) ==="]
            for ev in events:
                flag = {"H": "!!!", "M": " ! ", "L": "   "}.get(ev["impact"], "   ")
                lines.append(f"  {flag} {ev['date']}  [{ev['country']}]  {ev['event']}")
            return "\n".join(lines)

    # ── Watchlist ─────────────────────────────────────────────────────────────

    def render_watchlist(self, tickers: List[str]) -> Any:
        rows = []
        for ticker in tickers:
            info = _fetch_info(ticker)
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            chg1d = info.get("regularMarketChangePercent") or 0.0
            hi52 = info.get("fiftyTwoWeekHigh") or price
            lo52 = info.get("fiftyTwoWeekLow") or price

            # RSI from price history
            df = _fetch_ohlcv(ticker, period="3mo", interval="1d")
            rsi = None
            chg1w = chg1m = 0.0
            if df is not None and len(df) > 20:
                closes = df["Close"].dropna().tolist()
                rsi = _compute_rsi(closes)
                if len(closes) >= 5:
                    chg1w = (closes[-1] - closes[-5]) / closes[-5] * 100
                if len(closes) >= 21:
                    chg1m = (closes[-1] - closes[-21]) / closes[-21] * 100
            rows.append((ticker, price, chg1d, chg1w, chg1m, rsi, lo52, hi52))

        def _chg_str(v: float) -> str:
            return f"{v:+.2f}%"

        if self.mode == RendererMode.DASH and HAS_DASH:
            header_row = html.Tr([
                html.Th(c, style={"padding": "5px 8px", "color": "#74b9ff", "textAlign": "right" if i > 0 else "left"})
                for i, c in enumerate(["Ticker", "Price", "1D", "1W", "1M", "RSI", "52W L", "52W H"])
            ])
            body_rows = []
            for ticker, price, c1d, c1w, c1m, rsi, lo, hi in rows:
                def color_chg(v: float) -> str:
                    return "#00b894" if v >= 0 else "#d63031"
                rsi_color = "#d63031" if (rsi or 50) > 70 else "#00b894" if (rsi or 50) < 30 else "#dfe6e9"
                body_rows.append(html.Tr([
                    html.Td(ticker, style={"padding": "4px 8px", "color": "#dfe6e9"}),
                    html.Td(f"${price:,.2f}", style={"padding": "4px 8px", "textAlign": "right", "color": "#dfe6e9"}),
                    html.Td(_chg_str(c1d), style={"padding": "4px 8px", "textAlign": "right", "color": color_chg(c1d)}),
                    html.Td(_chg_str(c1w), style={"padding": "4px 8px", "textAlign": "right", "color": color_chg(c1w)}),
                    html.Td(_chg_str(c1m), style={"padding": "4px 8px", "textAlign": "right", "color": color_chg(c1m)}),
                    html.Td(f"{rsi:.0f}" if rsi else "—", style={"padding": "4px 8px", "textAlign": "right", "color": rsi_color}),
                    html.Td(f"${lo:,.2f}", style={"padding": "4px 8px", "textAlign": "right", "color": "#636e72"}),
                    html.Td(f"${hi:,.2f}", style={"padding": "4px 8px", "textAlign": "right", "color": "#636e72"}),
                ]))
            return html.Table(
                [html.Thead(header_row), html.Tbody(body_rows)],
                style={"width": "100%", "fontFamily": "monospace", "fontSize": "12px",
                       "borderCollapse": "collapse"},
            )

        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title="Watchlist", show_header=True, header_style="bold cyan")
            for col in ["Ticker", "Price", "1D %", "1W %", "1M %", "RSI", "52W L", "52W H"]:
                t.add_column(col, justify="right" if col != "Ticker" else "left")
            for ticker, price, c1d, c1w, c1m, rsi, lo, hi in rows:
                rsi_col = "red" if (rsi or 50) > 70 else "green" if (rsi or 50) < 30 else "white"
                t.add_row(
                    ticker, f"${price:,.2f}",
                    Text(_chg_str(c1d), style="green" if c1d >= 0 else "red"),
                    Text(_chg_str(c1w), style="green" if c1w >= 0 else "red"),
                    Text(_chg_str(c1m), style="green" if c1m >= 0 else "red"),
                    Text(f"{rsi:.0f}" if rsi else "—", style=rsi_col),
                    f"${lo:,.2f}", f"${hi:,.2f}",
                )
            return t

        else:
            lines = ["=== WATCHLIST ==="]
            header = f"  {'Ticker':<8} {'Price':>10} {'1D':>8} {'1W':>8} {'1M':>8} {'RSI':>6}"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for ticker, price, c1d, c1w, c1m, rsi, lo, hi in rows:
                lines.append(
                    f"  {ticker:<8} ${price:>9,.2f} {c1d:>+7.2f}% {c1w:>+7.2f}% {c1m:>+7.2f}%"
                    f" {rsi:>5.0f}" if rsi else f"  {ticker:<8} ${price:>9,.2f} {c1d:>+7.2f}% {c1w:>+7.2f}% {c1m:>+7.2f}%     —"
                )
            return "\n".join(lines)

    # ── Generic panel dispatch ────────────────────────────────────────────────

    def render_panel(self, panel: Panel) -> Any:
        panel.last_updated = datetime.datetime.utcnow()
        cfg = panel.config or {}
        try:
            if panel.panel_type == PanelType.PRICE_CHART:
                return self.render_price_chart(
                    panel.ticker or "SPY",
                    timeframe=cfg.get("timeframe", "1d"),
                    n_bars=cfg.get("n_bars", 252),
                )
            elif panel.panel_type == PanelType.FUNDAMENTALS:
                return self.render_fundamentals(panel.ticker or "AAPL")
            elif panel.panel_type == PanelType.NEWS_FEED:
                return self.render_news_feed(panel.ticker or "AAPL", n=cfg.get("n", 10))
            elif panel.panel_type == PanelType.PORTFOLIO:
                return self.render_portfolio(cfg.get("holdings"))
            elif panel.panel_type == PanelType.RISK_DASHBOARD:
                return self.render_risk_dashboard(cfg.get("holdings"))
            elif panel.panel_type == PanelType.MACRO_CALENDAR:
                return self.render_macro_calendar(days=cfg.get("days", 7))
            elif panel.panel_type == PanelType.WATCHLIST:
                tickers = cfg.get("tickers", [panel.ticker] if panel.ticker else ["SPY", "QQQ", "IWM"])
                return self.render_watchlist(tickers)
            elif panel.panel_type == PanelType.EARNINGS:
                return self.render_fundamentals(panel.ticker or "AAPL")
            elif panel.panel_type == PanelType.OPTIONS_CHAIN:
                return self._render_options_stub(panel.ticker or "AAPL")
            elif panel.panel_type == PanelType.FACTOR_EXPOSURE:
                return self._render_factor_stub()
            elif panel.panel_type == PanelType.SENTIMENT:
                return self._render_sentiment_stub(cfg.get("source", "alternative_me"))
            else:
                return self._render_placeholder(panel.title)
        except Exception as exc:
            logger.error("Panel render error (%s): %s", panel.panel_type, exc, exc_info=True)
            return self._render_error(panel.title, str(exc))

    def _render_options_stub(self, ticker: str) -> Any:
        """Options chain placeholder — full impl in options_chain_v3.py."""
        try:
            if not HAS_YF:
                raise ImportError("yfinance not available")
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                raise ValueError("No options data")
            exp = exps[0]
            chain = t.option_chain(exp)
            calls = chain.calls.head(8) if HAS_PANDAS else None
            if self.mode == RendererMode.DASH and HAS_DASH and calls is not None:
                rows = []
                for _, row in calls.iterrows():
                    rows.append(html.Tr([
                        html.Td(f"{row.get('strike', '—'):.2f}", style={"padding": "3px 8px", "color": "#74b9ff"}),
                        html.Td(f"{row.get('bid', 0):.2f}", style={"padding": "3px 8px", "color": "#dfe6e9"}),
                        html.Td(f"{row.get('ask', 0):.2f}", style={"padding": "3px 8px", "color": "#dfe6e9"}),
                        html.Td(f"{row.get('impliedVolatility', 0):.1%}", style={"padding": "3px 8px", "color": "#fdcb6e"}),
                        html.Td(f"{row.get('openInterest', 0):,.0f}", style={"padding": "3px 8px", "color": "#b2bec3"}),
                    ]))
                header = html.Tr([
                    html.Th(c, style={"padding": "5px 8px", "color": "#74b9ff"})
                    for c in ["Strike", "Bid", "Ask", "IV", "OI"]
                ])
                return html.Div([
                    html.H4(f"Options Chain — {ticker} (exp: {exp})",
                            style={"color": "#74b9ff", "margin": "8px", "fontSize": "13px"}),
                    html.Table([html.Thead(header), html.Tbody(rows)],
                               style={"fontFamily": "monospace", "fontSize": "12px",
                                      "borderCollapse": "collapse", "width": "100%"}),
                ])
            elif self.mode == RendererMode.RICH and HAS_RICH and calls is not None:
                tb = Table(title=f"Options Chain — {ticker} ({exp})", show_header=True)
                for col in ["Strike", "Bid", "Ask", "IV", "OI"]:
                    tb.add_column(col, justify="right")
                for _, row in calls.iterrows():
                    tb.add_row(
                        f"{row.get('strike', 0):.2f}",
                        f"{row.get('bid', 0):.2f}",
                        f"{row.get('ask', 0):.2f}",
                        f"{row.get('impliedVolatility', 0):.1%}",
                        f"{row.get('openInterest', 0):,.0f}",
                    )
                return tb
            else:
                return f"=== OPTIONS CHAIN: {ticker} exp={exp} ===\n  [Use --mode dash for full options view]"
        except Exception as exc:
            return self._render_placeholder(f"Options Chain — {ticker}\n  ({exc})")

    def _render_factor_stub(self) -> Any:
        """Factor exposure placeholder."""
        factors = [
            ("Market Beta", 1.12, "long equity mkt"),
            ("Size (SMB)", -0.23, "tilt to large-cap"),
            ("Value (HML)", 0.45, "value tilt"),
            ("Momentum", 0.31, "recent winner"),
            ("Quality (RMW)", 0.68, "profitable firms"),
            ("Investment (CMA)", -0.12, "aggressive inv."),
        ]
        if self.mode == RendererMode.DASH and HAS_DASH:
            rows = []
            for name, exp, desc in factors:
                color = "#00b894" if exp > 0 else "#d63031"
                bar_w = min(abs(exp) * 60, 80)
                rows.append(html.Div([
                    html.Span(name, style={"width": "120px", "display": "inline-block", "color": "#b2bec3", "fontSize": "12px"}),
                    html.Span("", style={"display": "inline-block", "width": f"{bar_w}px",
                                         "height": "14px", "backgroundColor": color, "verticalAlign": "middle"}),
                    html.Span(f" {exp:+.2f}", style={"color": color, "fontSize": "12px", "marginLeft": "6px"}),
                    html.Span(f" {desc}", style={"color": "#636e72", "fontSize": "11px", "marginLeft": "4px"}),
                ], style={"padding": "5px 10px"}))
            return html.Div(rows)
        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title="Factor Exposure", show_header=True, header_style="bold purple")
            t.add_column("Factor")
            t.add_column("Loading", justify="right")
            t.add_column("Description")
            for name, exp, desc in factors:
                t.add_row(name, Text(f"{exp:+.2f}", style="green" if exp > 0 else "red"), desc)
            return t
        else:
            lines = ["=== FACTOR EXPOSURE ==="]
            for name, exp, desc in factors:
                bar = "█" * int(abs(exp) * 10)
                lines.append(f"  {name:<20} {exp:+.2f}  {bar}  {desc}")
            return "\n".join(lines)

    def _render_sentiment_stub(self, source: str = "alternative_me") -> Any:
        score, label = 45, "Fear"
        if HAS_REQUESTS:
            try:
                resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
                data = resp.json()
                score = int(data["data"][0]["value"])
                label = data["data"][0]["value_classification"]
            except Exception:
                pass

        color = "#d63031" if score < 25 else "#e17055" if score < 50 else "#fdcb6e" if score < 75 else "#00b894"

        if self.mode == RendererMode.DASH and HAS_DASH:
            return html.Div([
                html.H3("Fear & Greed Index", style={"color": "#74b9ff", "textAlign": "center", "margin": "10px 0 5px"}),
                html.Div(str(score), style={"fontSize": "64px", "fontWeight": "bold", "color": color,
                                             "textAlign": "center", "fontFamily": "monospace"}),
                html.Div(label, style={"fontSize": "20px", "color": color, "textAlign": "center"}),
                html.Div("Source: alternative.me", style={"color": "#636e72", "fontSize": "10px",
                                                            "textAlign": "center", "marginTop": "8px"}),
            ], style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "height": "100%"})
        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(title="Fear & Greed Index", show_header=False)
            t.add_column("Metric")
            t.add_column("Value", justify="right")
            col = "red" if score < 25 else "yellow" if score < 50 else "green"
            t.add_row("Score", Text(str(score), style=f"bold {col}"))
            t.add_row("Label", Text(label, style=col))
            return t
        else:
            return f"=== SENTIMENT ===\n  Fear & Greed Index: {score} ({label})"

    def _render_placeholder(self, title: str) -> Any:
        if self.mode == RendererMode.DASH and HAS_DASH:
            return html.Div([
                html.Div("⬡", style={"fontSize": "32px", "color": "#2d3436", "textAlign": "center"}),
                html.Div(title, style={"color": "#636e72", "textAlign": "center", "fontSize": "13px"}),
            ], style={"display": "flex", "flexDirection": "column", "justifyContent": "center",
                      "height": "100%", "alignItems": "center"})
        elif self.mode == RendererMode.RICH and HAS_RICH:
            return RichPanel(f"[dim]{title}[/dim]", style="dim")
        else:
            return f"=== {title.upper()} ===\n  [No data available]"

    def _render_error(self, title: str, error: str) -> Any:
        if self.mode == RendererMode.DASH and HAS_DASH:
            return html.Div([
                html.Div("Error", style={"color": "#d63031", "fontWeight": "bold"}),
                html.Div(error[:120], style={"color": "#b2bec3", "fontSize": "11px"}),
            ], style={"padding": "12px"})
        elif self.mode == RendererMode.RICH and HAS_RICH:
            t = Table(show_header=False)
            t.add_column("info")
            t.add_row(Text(f"Error: {error[:80]}", style="red"))
            return t
        else:
            return f"=== {title} ===\n  ERROR: {error[:100]}"


# ── Dash Workspace App ─────────────────────────────────────────────────────────


class DashWorkspaceApp:
    """Build and run the full Dash multi-panel workspace."""

    DARK_CSS = """
    body { background-color: #1a1a2e; color: #dfe6e9; font-family: 'Roboto Mono', monospace; margin: 0; }
    .workspace-grid { display: grid; gap: 4px; padding: 4px; height: 100vh; box-sizing: border-box; }
    .panel { background: #1e272e; border: 1px solid #2d3436; border-radius: 4px; overflow: hidden;
              display: flex; flex-direction: column; }
    .panel-header { background: #2d3436; padding: 6px 12px; font-size: 12px; font-weight: 600;
                    color: #74b9ff; border-bottom: 1px solid #3d4852; display: flex;
                    justify-content: space-between; align-items: center; }
    .panel-body { flex: 1; overflow: auto; padding: 0; }
    .timestamp { color: #636e72; font-size: 10px; }
    """

    def __init__(self):
        if not HAS_DASH:
            raise ImportError("Dash not installed. pip install dash")
        self.renderer = PanelRenderer(mode=RendererMode.DASH)

    def _panel_layout_style(self, panels: List[Panel]) -> Tuple[str, str]:
        """Derive CSS grid template from panel widths."""
        # group panels by row heuristic: sum widths to 100
        row_templates = []
        current_row: List[float] = []
        current_sum = 0.0
        for panel in panels:
            current_row.append(panel.width_pct)
            current_sum += panel.width_pct
            if current_sum >= 99.0:
                row_templates.append(current_row)
                current_row = []
                current_sum = 0.0
        if current_row:
            row_templates.append(current_row)

        col_fractions = " ".join(f"{w}fr" for row in row_templates for w in row)
        row_heights = " ".join(f"{p.height_pct}fr" for p in panels[:len(row_templates)])
        return col_fractions, row_heights

    def build_app(self, workspace: Workspace) -> "dash.Dash":
        app = Dash(
            __name__,
            title=f"SENTINEL — {workspace.layout.name}",
            update_title=None,
        )

        panels = workspace.layout.panels
        panel_divs = []

        for panel in panels:
            panel_content_id = f"panel-content-{panel.id}"
            interval_id = f"interval-{panel.id}"
            panel_divs.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(panel.title, className="panel-title"),
                                html.Span(id=f"ts-{panel.id}", className="timestamp"),
                            ],
                            className="panel-header",
                        ),
                        html.Div(
                            id=panel_content_id,
                            className="panel-body",
                            style={"padding": "4px"},
                        ),
                        dcc.Interval(
                            id=interval_id,
                            interval=panel.refresh_interval_seconds * 1000,
                            n_intervals=0,
                        ),
                    ],
                    className="panel",
                    style={
                        "gridColumn": f"span {max(1, int(panel.width_pct / 25))}",
                        "minHeight": "200px",
                    },
                )
            )

        # Grid template: up to 4 columns
        cols = 4
        app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Span("SENTINEL", style={"color": "#74b9ff", "fontWeight": "bold", "fontSize": "16px"}),
                        html.Span(f" | {workspace.layout.name}", style={"color": "#636e72", "fontSize": "13px"}),
                        html.Span(" | Bloomberg-grade Terminal", style={"color": "#2d3436", "fontSize": "11px"}),
                    ],
                    style={"background": "#0a0a1a", "padding": "8px 16px",
                           "borderBottom": "1px solid #2d3436", "display": "flex",
                           "alignItems": "center", "gap": "0px"},
                ),
                html.Div(
                    panel_divs,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": f"repeat({cols}, 1fr)",
                        "gap": "4px",
                        "padding": "4px",
                        "height": "calc(100vh - 42px)",
                        "boxSizing": "border-box",
                    },
                ),
            ],
            style={"height": "100vh", "background": "#1a1a2e"},
        )

        # Inject CSS
        app.index_string = app.index_string.replace(
            "</head>",
            f"<style>{self.DARK_CSS}</style></head>",
        )

        # Register callbacks for each panel
        for panel in panels:
            panel_ref = panel  # capture

            @app.callback(
                [
                    Output(f"panel-content-{panel_ref.id}", "children"),
                    Output(f"ts-{panel_ref.id}", "children"),
                ],
                Input(f"interval-{panel_ref.id}", "n_intervals"),
            )
            def _update(n_intervals, _panel=panel_ref):
                content = self.renderer.render_panel(_panel)
                ts = f"Updated {datetime.datetime.now().strftime('%H:%M:%S')}"
                return content, ts

        return app

    def run(self, workspace: Workspace, port: int = 8050, debug: bool = False):
        app = self.build_app(workspace)
        url = f"http://localhost:{port}"
        logger.info("Starting Dash workspace at %s", url)
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        app.run(debug=debug, port=port, host="0.0.0.0", use_reloader=False)


# ── Rich Workspace App ─────────────────────────────────────────────────────────


class RichWorkspaceApp:
    """Terminal-based multi-panel workspace using Rich Live."""

    def __init__(self):
        if not HAS_RICH:
            raise ImportError("Rich not installed. pip install rich")
        self.renderer = PanelRenderer(mode=RendererMode.RICH)
        self.console = Console()
        self._stop = threading.Event()

    def _build_layout(self, workspace: Workspace) -> Layout:
        layout = Layout(name="root")
        panels = workspace.layout.panels

        # Header
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["header"].update(
            RichPanel(
                f"[bold cyan]SENTINEL[/] — [dim]{workspace.layout.name}[/]   "
                f"[dim]Press Ctrl+C to exit[/]",
                style="on #1a1a2e",
            )
        )

        # Body: divide panels by rows
        if len(panels) <= 1:
            layout["body"].update(
                self._make_panel_widget(panels[0]) if panels else RichPanel("[dim]No panels[/]")
            )
        elif len(panels) == 2:
            layout["body"].split_row(
                Layout(name="p0"),
                Layout(name="p1"),
            )
            for i, p in enumerate(panels):
                layout["body"][f"p{i}"].update(self._make_panel_widget(p))
        elif len(panels) == 3:
            layout["body"].split_column(
                Layout(name="top"),
                Layout(name="bottom"),
            )
            layout["body"]["top"].split_row(
                Layout(name="p0"),
                Layout(name="p1"),
            )
            layout["body"]["top"]["p0"].update(self._make_panel_widget(panels[0]))
            layout["body"]["top"]["p1"].update(self._make_panel_widget(panels[1]))
            layout["body"]["bottom"].update(self._make_panel_widget(panels[2]))
        else:
            layout["body"].split_column(
                Layout(name="top"),
                Layout(name="bottom"),
            )
            layout["body"]["top"].split_row(
                Layout(name="p0"),
                Layout(name="p1"),
            )
            layout["body"]["bottom"].split_row(
                Layout(name="p2"),
                Layout(name="p3"),
            )
            for i, p in enumerate(panels[:4]):
                row = "top" if i < 2 else "bottom"
                col = f"p{i}"
                layout["body"][row][col].update(self._make_panel_widget(p))
        return layout

    def _make_panel_widget(self, panel: Panel) -> Any:
        content = self.renderer.render_panel(panel)
        if isinstance(content, Table):
            return RichPanel(content, title=panel.title, border_style="dim blue")
        else:
            return RichPanel(str(content), title=panel.title, border_style="dim blue")

    def run(self, workspace: Workspace, refresh_seconds: int = 30):
        self.console.print(f"[bold cyan]SENTINEL — {workspace.layout.name}[/]")
        self.console.print("[dim]Building layout...[/]")

        layout = self._build_layout(workspace)

        def _refresh_loop(live: "Live"):
            while not self._stop.is_set():
                for _ in range(refresh_seconds * 10):
                    if self._stop.is_set():
                        break
                    time.sleep(0.1)
                if not self._stop.is_set():
                    new_layout = self._build_layout(workspace)
                    live.update(new_layout)

        try:
            with Live(layout, console=self.console, screen=True, refresh_per_second=1) as live:
                t = threading.Thread(target=_refresh_loop, args=(live,), daemon=True)
                t.start()
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            self._stop.set()
            self.console.print("\n[bold yellow]Exiting SENTINEL workspace.[/]")


# ── Plain Text Workspace ───────────────────────────────────────────────────────


class PlainWorkspaceApp:
    """Fallback plain-text workspace — no optional dependencies required."""

    def __init__(self):
        self.renderer = PanelRenderer(mode=RendererMode.PLAIN)

    def run(self, workspace: Workspace, refresh_seconds: int = 60):
        print("=" * 70)
        print(f"  SENTINEL — {workspace.layout.name}")
        print(f"  {workspace.layout.description}")
        print("=" * 70)
        print(f"  Panels: {len(workspace.layout.panels)}")
        print(f"  Auto-refresh: {refresh_seconds}s  |  Ctrl+C to exit")
        print()

        try:
            while True:
                print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print("-" * 70)
                for panel in workspace.layout.panels:
                    content = self.renderer.render_panel(panel)
                    print(content)
                    print()
                print(f"\nNext refresh in {refresh_seconds}s...")
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            print("\nExiting SENTINEL workspace.")


# ── Workspace HTTP Server ──────────────────────────────────────────────────────


class WorkspaceServer:
    """Serve workspace state as JSON over HTTP."""

    def __init__(self, workspace: Workspace, port: int = 9090):
        self.workspace = workspace
        self.port = port
        self._renderer = PanelRenderer(mode=RendererMode.PLAIN)

    def _get_state(self) -> dict:
        state = self.workspace.to_dict()
        state["panels_data"] = {}
        for panel in self.workspace.layout.panels:
            try:
                content = self._renderer.render_panel(panel)
                state["panels_data"][panel.id] = {"content": str(content), "type": panel.panel_type.value}
            except Exception as exc:
                state["panels_data"][panel.id] = {"error": str(exc)}
        return state

    def _build_fastapi_app(self) -> "FastAPI":
        app = FastAPI(title="SENTINEL Workspace API")

        @app.get("/health")
        def health():
            return {"status": "ok", "workspace": self.workspace.layout.name}

        @app.get("/workspace/{workspace_id}/state")
        def get_state(workspace_id: str):
            if workspace_id != self.workspace.id and workspace_id != "current":
                from fastapi import HTTPException
                raise HTTPException(404, "Workspace not found")
            return self._get_state()

        @app.get("/workspace/{workspace_id}/panels/{panel_id}")
        def get_panel(workspace_id: str, panel_id: str):
            for p in self.workspace.layout.panels:
                if p.id == panel_id:
                    content = self._renderer.render_panel(p)
                    return {"panel_id": panel_id, "content": str(content), "updated": p.last_updated.isoformat() if p.last_updated else None}
            from fastapi import HTTPException
            raise HTTPException(404, f"Panel {panel_id} not found")

        return app

    def run_fastapi(self, host: str = "0.0.0.0"):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI not available")
        import uvicorn
        app = self._build_fastapi_app()
        uvicorn.run(app, host=host, port=self.port, log_level="warning")

    def run_simple(self, host: str = "0.0.0.0"):
        """Fallback: stdlib http.server."""
        workspace_ref = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # suppress

            def do_GET(self):
                if self.path.startswith("/workspace"):
                    body = json.dumps(workspace_ref._get_state(), default=str).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(404)
                    self.end_headers()

        server = http.server.HTTPServer((host, workspace_ref.port), Handler)
        logger.info("Workspace JSON server on http://%s:%d", host, workspace_ref.port)
        server.serve_forever()

    def start_background(self):
        """Start server in a background daemon thread."""
        if HAS_FASTAPI:
            t = threading.Thread(target=self.run_fastapi, daemon=True)
        else:
            t = threading.Thread(target=self.run_simple, daemon=True)
        t.start()
        return t


# ── Keyboard Handler (terminal mode) ──────────────────────────────────────────


def _keyboard_shortcuts_info() -> str:
    return (
        "Keyboard shortcuts (terminal mode):\n"
        "  q     — quit\n"
        "  r     — force refresh all panels\n"
        "  1-6   — switch to preset layout (1=EQUITY_DEEP_DIVE ... 6=SCREENER_WORKSPACE)\n"
        "  +     — add custom panel\n"
        "  -     — remove last panel\n"
    )


# ── Main Entrypoint ────────────────────────────────────────────────────────────


def _select_renderer() -> RendererMode:
    if HAS_DASH:
        return RendererMode.DASH
    if HAS_RICH:
        return RendererMode.RICH
    return RendererMode.PLAIN


def launch_workspace(
    layout: str = "EQUITY_DEEP_DIVE",
    ticker: str = "AAPL",
    port: int = 8050,
    debug: bool = False,
    json_server_port: int = 9090,
    refresh_seconds: int = 60,
    serve_json: bool = True,
):
    """High-level entry point — selects best available renderer automatically."""
    mgr = WorkspaceManager()
    ws = mgr.create_workspace(layout, ticker=ticker)
    mode = _select_renderer()

    logger.info("SENTINEL workspace: layout=%s ticker=%s mode=%s", layout, ticker, mode.value)
    print(_keyboard_shortcuts_info())

    # Start JSON API server in background
    if serve_json:
        srv = WorkspaceServer(ws, port=json_server_port)
        srv.start_background()
        print(f"  Workspace state API: http://localhost:{json_server_port}/workspace/current/state")

    if mode == RendererMode.DASH:
        print(f"  Dash UI: http://localhost:{port}")
        app_obj = DashWorkspaceApp()
        app_obj.run(ws, port=port, debug=debug)
    elif mode == RendererMode.RICH:
        app_obj = RichWorkspaceApp()
        app_obj.run(ws, refresh_seconds=refresh_seconds)
    else:
        app_obj = PlainWorkspaceApp()
        app_obj.run(ws, refresh_seconds=refresh_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SENTINEL Multi-Panel Workspace")
    parser.add_argument("--ticker", default="AAPL", help="Primary ticker")
    parser.add_argument(
        "--layout",
        default="EQUITY_DEEP_DIVE",
        choices=["EQUITY_DEEP_DIVE", "PORTFOLIO_MONITOR", "TRADING_DESK",
                 "MACRO_WATCH", "CRYPTO_DASHBOARD", "SCREENER_WORKSPACE"],
        help="Workspace layout preset",
    )
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    parser.add_argument("--json-port", type=int, default=9090, help="JSON API server port")
    parser.add_argument("--debug", action="store_true", help="Dash debug mode")
    parser.add_argument("--refresh", type=int, default=60, help="Refresh interval (seconds, Rich/plain modes)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\nSENTINEL — Bloomberg-grade Financial Terminal")
    print(f"  Layout:  {args.layout}")
    print(f"  Ticker:  {args.ticker}")
    print(f"  Dash:    {'available' if HAS_DASH else 'NOT installed (pip install dash)'}")
    print(f"  Rich:    {'available' if HAS_RICH else 'NOT installed (pip install rich)'}")
    print(f"  yfinance:{'available' if HAS_YF else 'NOT installed (pip install yfinance)'}")
    print()

    launch_workspace(
        layout=args.layout,
        ticker=args.ticker,
        port=args.port,
        debug=args.debug,
        json_server_port=args.json_port,
        refresh_seconds=args.refresh,
    )
