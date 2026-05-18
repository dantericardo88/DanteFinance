#!/usr/bin/env bash
# dim_091: Workspace — panel layout config, serialization, export, scheduler
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import datetime, json, tempfile, pathlib

from sentinel.ui.workspace_v3 import (
    Panel,
    PanelType,
    WorkspaceLayout,
    Workspace,
    LayoutPresets,
    RendererMode,
    WorkspaceExporter,
    PanelScheduler,
    save_workspace_state,
    load_workspace_state,
)

# ── PanelType / RendererMode enums ──────────────────────────────────────────
assert PanelType.PRICE_CHART.value == "price_chart"
assert PanelType.FUNDAMENTALS.value == "fundamentals"
assert PanelType.OPTIONS_CHAIN.value == "options_chain"
assert PanelType.PORTFOLIO.value == "portfolio"
assert PanelType.RISK_DASHBOARD.value == "risk_dashboard"
print(f"[OK] PanelType enum: {len(PanelType)} panel types defined")

assert RendererMode.DASH.value == "dash"
assert RendererMode.RICH.value == "rich"
assert RendererMode.PLAIN.value == "plain"
print("[OK] RendererMode enum: DASH/RICH/PLAIN")

# ── Panel round-trip serialization ──────────────────────────────────────────
import uuid
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
d = panel.to_dict()
assert d["panel_type"] == "price_chart"
panel2 = Panel.from_dict(d)
assert panel2.panel_type == PanelType.PRICE_CHART
assert panel2.config == panel.config
print(f"[OK] Panel.to_dict()/from_dict() round-trip preserved type={panel2.panel_type.value}")

# ── Workspace round-trip (WorkspaceState serialization) ─────────────────────
layout_eq = LayoutPresets.equity_deep_dive("MSFT")
workspace = Workspace(
    id=str(uuid.uuid4()),
    layout=layout_eq,
    metadata={"user": "test", "version": "3"},
)
ws_dict = workspace.to_dict()
assert len(ws_dict["layout"]["panels"]) == 4
workspace2 = Workspace.from_dict(ws_dict)
assert workspace2.id == workspace.id
assert workspace2.layout.name == "EQUITY_DEEP_DIVE"
assert workspace2.layout.panels[0].panel_type == PanelType.PRICE_CHART
assert workspace2.metadata["user"] == "test"
print("[OK] Workspace.to_dict()/from_dict() round-trip: 4 panels restored")

# ── Panel state persistence (save/load) ─────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    state_path = pathlib.Path(tmpdir) / "workspace_state.json"
    saved_path = save_workspace_state(workspace, path=state_path)
    assert state_path.exists(), "State file should exist after save"
    loaded = load_workspace_state(path=state_path)
    assert loaded is not None, "load_workspace_state should return a Workspace"
    assert loaded.id == workspace.id
    assert loaded.layout.name == workspace.layout.name
    assert len(loaded.layout.panels) == len(workspace.layout.panels)
    print(f"[OK] save/load workspace state: id={loaded.id} panels={len(loaded.layout.panels)}")

# ── Export to JSON ───────────────────────────────────────────────────────────
exporter = WorkspaceExporter(workspace)
with tempfile.TemporaryDirectory() as tmpdir:
    json_path = os.path.join(tmpdir, "export.json")
    result_path = exporter.export_json(json_path)
    assert os.path.exists(result_path), "JSON export file should exist"
    with open(result_path) as fh:
        data = json.load(fh)
    assert "workspace" in data, "JSON export must have 'workspace' key"
    assert "snapshots" in data, "JSON export must have 'snapshots' key"
    assert "exported_at" in data, "JSON export must have 'exported_at' key"
    assert "layout" in data["workspace"], "workspace should have layout"
    panels_in_export = data["workspace"]["layout"]["panels"]
    assert len(panels_in_export) == 4, f"Export should have 4 panels: {len(panels_in_export)}"
    # Verify snapshots has an entry per panel
    for p in workspace.layout.panels:
        assert p.id in data["snapshots"], f"Snapshot missing for panel {p.id}"
    print(f"[OK] export_json: valid JSON with {len(panels_in_export)} panel configs")

# ── Export to HTML ───────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    html_path = os.path.join(tmpdir, "export.html")
    result_path = exporter.export_html(html_path)
    assert os.path.exists(result_path), "HTML export file should exist"
    with open(result_path, encoding="utf-8") as fh:
        html_content = fh.read()
    assert "<!DOCTYPE html>" in html_content, "Should be a valid HTML document"
    assert "SENTINEL" in html_content, "Should mention SENTINEL"
    assert workspace.layout.name in html_content, "Should include layout name"
    # At least one panel header should appear
    assert any(p.title in html_content for p in workspace.layout.panels), \
        "At least one panel title should be in HTML"
    print("[OK] export_html: valid self-contained HTML report")

# ── Scheduler tick (stale panel detection) ──────────────────────────────────
# Build a workspace with panels that have last_updated = None (never refreshed)
layout_td = LayoutPresets.trading_desk("SPY")
ws_sched = Workspace(id=str(uuid.uuid4()), layout=layout_td)
scheduler = PanelScheduler(ws_sched)

# All panels start with last_updated=None → all are stale → all fire on first tick
fired = scheduler.tick()
assert len(fired) == len(ws_sched.layout.panels), \
    f"All {len(ws_sched.layout.panels)} panels should fire on first tick: got {len(fired)}"
for pid in fired:
    assert scheduler.refresh_counts[pid] == 1, f"Panel {pid} should have count=1"
print(f"[OK] Scheduler first tick: {len(fired)} panels fired (all stale on init)")

# Mark one panel as just-refreshed; a second tick should NOT re-fire it
first_panel = ws_sched.layout.panels[0]
scheduler.mark_refreshed(first_panel.id)
fired2 = scheduler.tick()
# The marked panel should NOT be in fired2 (just refreshed, TTL not yet elapsed)
assert first_panel.id not in fired2, \
    f"Just-refreshed panel should not re-fire: {first_panel.id} in {fired2}"
# Other panels (still last_updated=None) should still fire
remaining = [p for p in ws_sched.layout.panels if p.id != first_panel.id]
for p in remaining:
    assert p.id in fired2, f"Stale panel {p.id} should fire on second tick"
print(f"[OK] Scheduler second tick: {len(fired2)} stale panels fired, refreshed panel skipped")

# Force-expire one panel by backdating its last_updated beyond TTL
first_panel.last_updated = datetime.datetime.utcnow() - datetime.timedelta(
    seconds=first_panel.refresh_interval_seconds + 5
)
fired3 = scheduler.tick()
assert first_panel.id in fired3, "Expired panel should fire after TTL elapsed"
assert scheduler.refresh_counts[first_panel.id] == 2, \
    f"Panel refresh count should be 2 after 2 fires: {scheduler.refresh_counts[first_panel.id]}"
print(f"[OK] Scheduler TTL expiry: panel re-fired after TTL elapsed (count={scheduler.refresh_counts[first_panel.id]})")

# ── Layout presets validation ────────────────────────────────────────────────
for layout in [layout_eq, LayoutPresets.portfolio_monitor(), layout_td, LayoutPresets.macro_watch()]:
    for p in layout.panels:
        assert p.refresh_interval_seconds > 0
        assert p.refresh_interval_seconds <= 3600
        assert 0 < p.width_pct <= 100
        assert 0 < p.height_pct <= 100
print("[OK] All panels have valid refresh intervals and dimensions")

print("\n[PASS] dim_091: Workspace")
PYEOF
