#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import numpy as np
import math
from datetime import date

from sentinel.sma.regime_detector_v3 import (
    ViterbiHMM,
    RegimeState,
    MacroSnapshot,
    RegimeDetectionResult,
    RegimeAlert,
    RiskDuration,
)

# Test RegimeState dataclass
rs = RegimeState(
    name="EXPANSION",
    sub_regime="mid",
    confidence=0.85,
    state_idx=0,
)
assert rs.name == "EXPANSION"
assert rs.confidence == 0.85
s = str(rs)
assert "EXPANSION" in s
assert "mid" in s
print(f"[OK] RegimeState: {s}")

# Test MacroSnapshot dataclass
snap = MacroSnapshot(
    date=date(2024, 3, 15),
    yield_curve_2y10y=-0.5,
    yield_curve_3m10y=-1.2,
    hy_spread=3.8,
    unemployment=3.7,
    initial_claims=210000.0,
    payrolls_mom=275000.0,
    indpro_yoy=1.2,
    core_cpi_yoy=3.9,
    breakeven_10y=2.4,
    vix=15.0,
    vix_3m_avg=14.5,
    consumer_sentiment=79.0,
    gdpnow=2.5,
    composite_leading=0.3,
)
assert snap.yield_curve_2y10y == -0.5
assert snap.vix == 15.0
print("[OK] MacroSnapshot dataclass created successfully")

# Test ViterbiHMM initialization
hmm = ViterbiHMM(n_states=4, max_iter=50, tol=1e-4, random_state=42)
assert hmm.n_states == 4
assert hmm.is_fitted is False
assert len(hmm.state_labels) == 4
assert "EXPANSION" in hmm.state_labels
assert "RECESSION" in hmm.state_labels
print(f"[OK] ViterbiHMM initialized with state labels: {hmm.state_labels}")

# Test ViterbiHMM.fit on synthetic macro data (small, fast dataset)
np.random.seed(42)
# 3 features, 40 time steps — enough to test fit, fast to converge
n_samples = 40
n_features = 3
X = np.random.randn(n_samples, n_features)
# Make data slightly regime-like by adding structure
X[:10] += [1.0, -0.5, 0.5]    # "expansion" regime
X[10:20] += [0.0, 0.0, 0.0]   # "transition" regime
X[20:30] += [-1.0, 0.5, -0.5] # "recession" regime
X[30:] += [0.5, -0.3, 0.3]    # "recovery" regime

hmm_fast = ViterbiHMM(n_states=4, max_iter=15, tol=1e-3, random_state=42)
hmm_fast.fit(X)
assert hmm_fast.is_fitted is True
assert hmm_fast.means is not None
assert hmm_fast.covars is not None
assert hmm_fast.means.shape == (4, 3)
hmm = hmm_fast  # reuse for subsequent tests
print(f"[OK] ViterbiHMM.fit succeeded on {n_samples}x{n_features} data")

# Test ViterbiHMM.predict
states = hmm.predict(X)
assert len(states) == n_samples
assert all(0 <= s < 4 for s in states)
print(f"[OK] ViterbiHMM.predict returned {len(states)} states, unique={set(states)}")

# Test current state detection
current_state, conf = hmm.get_current_state(X)
assert isinstance(current_state, str)
assert 0.0 <= conf <= 1.0
print(f"[OK] ViterbiHMM.get_current_state: {current_state} (confidence={conf:.2%})")

# Test RegimeAlert dataclass
alert = RegimeAlert(
    from_regime="EXPANSION",
    to_regime="SLOWDOWN",
    alert_date=date(2024, 3, 15),
    confidence=0.75,
    features_at_change={"yield_curve_2y10y": -0.5, "vix": 18.0},
    message="Regime transition detected: EXPANSION -> SLOWDOWN",
)
assert alert.from_regime == "EXPANSION"
assert "SLOWDOWN" in alert.message
print("[OK] RegimeAlert dataclass created successfully")

# Test RiskDuration dataclass
rd = RiskDuration(
    regime="EXPANSION",
    avg_duration_months=38.5,
    min_duration=12,
    max_duration=92,
    n_episodes=10,
)
assert rd.avg_duration_months == 38.5
assert rd.n_episodes == 10
print("[OK] RiskDuration dataclass created successfully")

print("[PASS]")
PYEOF
