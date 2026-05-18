#!/usr/bin/env bash
# dim_091: Workspace — panel layout config logic and serialization
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import datetime

from sentinel.ui.workspace_v3 import (
    Panel,
    PanelType,
    WorkspaceLayout,
    Workspace,
    LayoutPresets,
    RendererMode,
)

# Test PanelType enum
assert PanelType.PRICE_CHART.value == "price_chart"
assert PanelType.FUNDAMENTALS.value == "fundamentals"
assert PanelType.OPTIONS_CHAIN.value == "options_chain"
assert PanelType.PORTFOLIO.value == "portfolio"
assert PanelType.RISK_DASHBOARD.value == "risk_dashboard"
print(f"[OK] PanelType enum: {len(PanelType)} panel types defined")

# Test RendererMode enum
assert RendererMode.DASH.value == "dash"
assert RendererMode.RICH.value == "rich"
assert RendererMode.PLAIN.value == "plain"
print(f"[OK] RendererMode enum: DASH/RICH/PLAIN")

# Test Panel construction and to_dict/from_dict round-trip
panel = Panel(
    id="test_p1",
    panel_type=PanelType.PRICE_CHART,
    title="Price Chart — AAPL",
    ticker="AAPL",
    width_pct=50.0,
    height_pct=50.0,
    refresh_interval_seconds=30,
    config={"timeframe": "1d", "n_bars": 252},
)
assert panel.id == "test_p1"
assert panel.panel_type == PanelType.PRICE_CHART
assert panel.ticker == "AAPL"
assert panel.width_pct == 50.0
print(f"[OK] Panel: id={panel.id} type={panel.panel_type.value} ticker={panel.ticker}")

# Test Panel.to_dict()
d = panel.to_dict()
assert d["id"] == "test_p1"
assert d["panel_type"] == "price_chart"
assert d["ticker"] == "AAPL"
assert d["config"]["timeframe"] == "1d"
print(f"[OK] Panel.to_dict(): panel_type serialized as string '{d['panel_type']}'")

# Test Panel.from_dict() round-trip
panel2 = Panel.from_dict(d)
assert panel2.id == panel.id
assert panel2.panel_type == PanelType.PRICE_CHART
assert panel2.ticker == panel.ticker
assert panel2.config == panel.config
print(f"[OK] Panel.from_dict(): round-trip preserves type={panel2.panel_type.value}")

# Test LayoutPresets.equity_deep_dive()
layout_eq = LayoutPresets.equity_deep_dive("MSFT")
assert layout_eq.name == "EQUITY_DEEP_DIVE", f"Expected EQUITY_DEEP_DIVE: {layout_eq.name}"
assert len(layout_eq.panels) == 4, f"Expected 4 panels: {len(layout_eq.panels)}"
# Verify first panel is price chart for MSFT
assert layout_eq.panels[0].panel_type == PanelType.PRICE_CHART
assert layout_eq.panels[0].ticker == "MSFT"
# Verify second panel is fundamentals
assert layout_eq.panels[1].panel_type == PanelType.FUNDAMENTALS
# Check width percentages sum properly (each row ~ 100%)
total_top_width = layout_eq.panels[0].width_pct + layout_eq.panels[1].width_pct + layout_eq.panels[2].width_pct
assert abs(total_top_width - 100.0) < 1.0, f"Top panels width should sum to ~100: {total_top_width}"
print(f"[OK] LayoutPresets.equity_deep_dive: 4 panels, top row width={total_top_width:.1f}%")

# Test LayoutPresets.portfolio_monitor()
layout_port = LayoutPresets.portfolio_monitor()
assert layout_port.name == "PORTFOLIO_MONITOR", f"Expected PORTFOLIO_MONITOR: {layout_port.name}"
assert len(layout_port.panels) >= 3, f"Expected >= 3 panels: {len(layout_port.panels)}"
panel_types = [p.panel_type for p in layout_port.panels]
assert PanelType.PORTFOLIO in panel_types, "Should have PORTFOLIO panel"
assert PanelType.RISK_DASHBOARD in panel_types, "Should have RISK_DASHBOARD panel"
print(f"[OK] LayoutPresets.portfolio_monitor: {len(layout_port.panels)} panels")

# Test LayoutPresets.trading_desk()
layout_td = LayoutPresets.trading_desk("SPY")
assert layout_td.name == "TRADING_DESK"
assert len(layout_td.panels) >= 3
# Trading desk should have very short refresh (order book ~5s)
order_book_panels = [p for p in layout_td.panels if p.panel_type == PanelType.ORDER_BOOK]
assert len(order_book_panels) >= 1, "Trading desk should have an order book panel"
assert order_book_panels[0].refresh_interval_seconds <= 10, \
    f"Order book should refresh frequently: {order_book_panels[0].refresh_interval_seconds}s"
print(f"[OK] LayoutPresets.trading_desk: {len(layout_td.panels)} panels, order_book refresh={order_book_panels[0].refresh_interval_seconds}s")

# Test LayoutPresets.macro_watch()
layout_macro = LayoutPresets.macro_watch()
assert layout_macro.name == "MACRO_WATCH"
assert len(layout_macro.panels) >= 3
print(f"[OK] LayoutPresets.macro_watch: {len(layout_macro.panels)} panels")

# Test Workspace serialization round-trip
import uuid
workspace = Workspace(
    id=str(uuid.uuid4()),
    layout=layout_eq,
    metadata={"user": "test", "version": "3"},
)
ws_dict = workspace.to_dict()
assert "id" in ws_dict
assert "layout" in ws_dict
assert "panels" in ws_dict["layout"]
assert len(ws_dict["layout"]["panels"]) == 4
print(f"[OK] Workspace.to_dict(): {len(ws_dict['layout']['panels'])} panels serialized")

# Restore from dict
workspace2 = Workspace.from_dict(ws_dict)
assert workspace2.id == workspace.id
assert workspace2.layout.name == "EQUITY_DEEP_DIVE"
assert len(workspace2.layout.panels) == 4
assert workspace2.layout.panels[0].panel_type == PanelType.PRICE_CHART
assert workspace2.metadata["user"] == "test"
print(f"[OK] Workspace.from_dict(): round-trip complete, panels restored correctly")

# Verify panel refresh intervals are reasonable
for layout in [layout_eq, layout_port, layout_td, layout_macro]:
    for p in layout.panels:
        assert p.refresh_interval_seconds > 0, \
            f"Refresh interval must be positive: {p.id}={p.refresh_interval_seconds}"
        assert p.refresh_interval_seconds <= 3600, \
            f"Refresh interval should not exceed 1h: {p.id}={p.refresh_interval_seconds}"
        assert 0 < p.width_pct <= 100, f"Width pct must be in (0, 100]: {p.width_pct}"
        assert 0 < p.height_pct <= 100, f"Height pct must be in (0, 100]: {p.height_pct}"
print("[OK] All panels have valid refresh intervals and dimensions")

print("\n[PASS] dim_091: Workspace")
PYEOF
