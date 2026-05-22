#!/bin/bash
# dim_018: Analyst consensus estimates aggregation
# Exit 0 = dimension verified  |  Exit 1 = not verified
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

import numpy as np

from sentinel.sai.analyst_consensus_v3 import (
    AnalystEstimate, ConsensusSnapshot, ConsensusAnalytics,
    EPSRevisionTracker, AnalystTracker,
    consensus_target, buy_sell_ratio, eps_revision_signal, upside_to_target,
)

# ---- Build 6 analyst estimates: 3 Buy, 2 Hold, 1 Sell --------------------
estimates = [
    AnalystEstimate("a1", "GS",  "Buy",          120.0, 4.50, 5.10, "2025-01-01"),
    AnalystEstimate("a2", "MS",  "Buy",          125.0, 4.60, 5.20, "2025-01-01"),
    AnalystEstimate("a3", "JPM", "Buy",          130.0, 4.55, 5.15, "2025-01-01"),
    AnalystEstimate("a4", "WF",  "Hold",         100.0, 4.20, 4.80, "2025-01-01"),
    AnalystEstimate("a5", "DB",  "Hold",         105.0, 4.30, 4.90, "2025-01-01"),
    AnalystEstimate("a6", "UBS", "Sell",          85.0, 3.80, 4.20, "2025-01-01"),
]

snap = ConsensusSnapshot(ticker="XYZ", current_price=110.0, estimates=estimates)

# ---- Test 1: n_analysts = 6 -----------------------------------------------
assert snap.n_analysts == 6, f"Expected 6 analysts, got {snap.n_analysts}"
print(f"[OK] n_analysts={snap.n_analysts}")

# ---- Test 2: mean_target ≈ 110.83 ----------------------------------------
mean_t = snap.mean_target
targets = [120, 125, 130, 100, 105, 85]
expected_mean = sum(targets) / len(targets)
assert abs(mean_t - expected_mean) < 0.5, f"mean_target {mean_t:.2f} != {expected_mean:.2f}"
print(f"[OK] mean_target={mean_t:.2f}")

# ---- Test 3: upside % is finite -------------------------------------------
upside = snap.upside_pct
assert math.isfinite(upside), f"upside_pct should be finite, got {upside}"
print(f"[OK] upside_pct={upside:.2f}%")

# ---- Test 4: buy_pct = 50% (3 buys out of 6) ----------------------------
bp = snap.buy_pct
assert abs(bp - 50.0) < 0.1, f"buy_pct {bp:.1f} should be 50.0%"
print(f"[OK] buy_pct={bp:.1f}%")

# ---- Test 5: consensus_score in (2.5, 4.0) --------------------------------
# Scores: Buy=4, Buy=4, Buy=4, Hold=3, Hold=3, Sell=1 => mean=(4+4+4+3+3+1)/6=19/6≈3.17
cs = snap.consensus_score
assert 2.5 < cs < 4.0, f"consensus_score {cs:.2f} not in (2.5, 4.0)"
print(f"[OK] consensus_score={cs:.3f} in (2.5, 4.0)")

# ---- Test 6: recommendation distribution contains Buy/Hold/Sell ----------
ca = ConsensusAnalytics(snap)
rec_dist = ca.recommendation_distribution()
assert rec_dist.get("Buy", 0) == 3, f"Expected 3 Buy, got {rec_dist.get('Buy',0)}"
assert rec_dist.get("Hold", 0) == 2, f"Expected 2 Hold, got {rec_dist.get('Hold',0)}"
assert rec_dist.get("Sell", 0) == 1, f"Expected 1 Sell, got {rec_dist.get('Sell',0)}"
print(f"[OK] recommendation_distribution: {rec_dist}")

# ---- Test 7: eps_consensus_cy finite -------------------------------------
eps_cy = snap.eps_consensus_cy
assert math.isfinite(eps_cy), f"eps_consensus_cy should be finite"
print(f"[OK] eps_consensus_cy={eps_cy:.3f}")

# ---- Test 8: target dispersion_cv > 0 ------------------------------------
td = ca.target_distribution()
cv = td["dispersion_cv"]
assert cv > 0, f"dispersion_cv {cv:.4f} should be > 0"
print(f"[OK] dispersion_cv={cv:.4f}")

# ---- Test 9: revision_trend same snapshot → all zeros -------------------
rev = ca.revision_trend(snap)
assert rev["n_upgrades"] == 0
assert rev["n_downgrades"] == 0
assert abs(rev["eps_revision_pct"]) < 1e-9
assert abs(rev["target_revision_pct"]) < 1e-9
print(f"[OK] revision_trend same snapshot: n_upgrades={rev['n_upgrades']}, n_downgrades={rev['n_downgrades']}")

# ---- Test 10: EPSRevisionTracker.revision_ratio = 0.667 -----------------
tracker = EPSRevisionTracker()
old_eps = [4.0, 4.5, 4.2, 4.8, 5.0, 4.3]
new_eps = [4.2, 4.6, 4.1, 4.9, 4.8, 4.5]  # 4 up, 2 down
ratio = tracker.revision_ratio(old_eps, new_eps)
assert abs(ratio - 4/6) < 1e-6, f"revision_ratio {ratio:.4f} != {4/6:.4f}"
print(f"[OK] revision_ratio={ratio:.4f} (4 up / 6 total = {4/6:.4f})")

# ---- Test 11: AnalystTracker accuracy in [0, 1] --------------------------
at = AnalystTracker()
forecasts = np.array([4.0, 4.5, 5.0, 4.2])
actuals   = np.array([4.1, 4.3, 4.9, 4.4])
acc = at.accuracy(forecasts, actuals)
assert 0.0 <= acc <= 1.0, f"accuracy {acc:.4f} not in [0, 1]"
print(f"[OK] analyst accuracy={acc:.4f}")

# ---- Test 12: upside_to_target convenience function ----------------------
u = upside_to_target(current_price=110.0, mean_target=mean_t)
assert math.isfinite(u), f"upside_to_target should be finite"
print(f"[OK] upside_to_target={u:.2f}%")

print("[PASS] dim_018: Analyst consensus aggregation")
PYEOF
