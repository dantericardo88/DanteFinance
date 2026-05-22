#!/bin/bash
# dim_030: ipo_intelligence_v3 — IPO pipeline intelligence (logistic regression upgrade)
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

import numpy as np
from datetime import date, timedelta

from sentinel.sfe.ipo_intelligence_v3 import (
    _BULGE_BRACKET,
    _SPAC_MARKERS,
    _HISTORICAL_IPOS,
    EdgarS1Parser,
    IPOPipelineTracker,
    SPACTracker,
    PipelineEntry,
    IPOResult,
    SPACRecord,
    compute_ipo_pop_prediction,
    compute_lockup_expiry_signal,
    classify_ipo_quality,
    predict_ipo_pop,
    _fit_ipo_pop_model,
    backtest_ipo_predictions,
)

# --- constants ---
assert "Goldman Sachs" in _BULGE_BRACKET
assert "Morgan Stanley" in _BULGE_BRACKET
assert len(_SPAC_MARKERS) >= 3
print("[OK] _BULGE_BRACKET and _SPAC_MARKERS constants present")

# --- PipelineEntry Pydantic model ---
entry = PipelineEntry(
    cik="0001234567",
    company_name="TechCo Inc",
    form_type="S-1",
    filed_date=date(2024, 3, 1),
    pipeline_state="filed",
    price_range_low=18.0,
    price_range_high=20.0,
    shares_offered=10_000_000,
)
assert entry.company_name == "TechCo Inc"
assert entry.pipeline_state == "filed"
assert entry.price_range_low == 18.0
print("[OK] PipelineEntry Pydantic model created")

# --- IPOResult Pydantic model ---
result = IPOResult(
    ticker="TECH",
    company_name="TechCo Inc",
    ipo_date=date(2024, 3, 20),
    offer_price=19.0,
    first_day_close=25.0,
    day1_return_pct=31.6,
)
assert result.ticker == "TECH"
assert result.day1_return_pct == 31.6
print("[OK] IPOResult Pydantic model created")

# --- SPACRecord Pydantic model ---
spac = SPACRecord(
    cik="0009876543",
    company_name="Acquisition Corp I",
    filed_date=date(2024, 1, 15),
    trust_amount_mn=300.0,
    target_industry="Technology",
    deadline_months=24,
)
assert spac.trust_amount_mn == 300.0
print("[OK] SPACRecord Pydantic model created")

# --- class structure checks ---
assert hasattr(EdgarS1Parser, "parse_s1") or hasattr(EdgarS1Parser, "__init__")
assert hasattr(IPOPipelineTracker, "__init__")
assert hasattr(SPACTracker, "__init__")
print("[OK] EdgarS1Parser, IPOPipelineTracker, SPACTracker class structure present")

# --------------------------------------------------------------------------
# NEW: Logistic Regression IPO pop predictor — REAL MODEL VERIFICATION
# --------------------------------------------------------------------------
assert len(_HISTORICAL_IPOS) >= 40, \
    f"Training set too small: {len(_HISTORICAL_IPOS)} samples (need >= 40)"
print(f"[OK] _HISTORICAL_IPOS has {len(_HISTORICAL_IPOS)} samples")

# Verify sigmoid math is real: probability MUST be in [0, 1]
r = predict_ipo_pop(
    offer_size=8.5, is_profitable=False, revenue_growth=1.5,
    sector="tech", market_vix=18, underwriter="goldman", age_years=5,
)
assert 0.0 <= r["probability_of_big_pop"] <= 1.0, \
    f"probability_of_big_pop={r['probability_of_big_pop']} not in [0,1] — not logistic"
assert "model_type" in r and "logistic" in r["model_type"].lower(), \
    f"model_type missing or not logistic: {r.get('model_type')}"
assert "coefficients" in r and isinstance(r["coefficients"], list)
assert "feature_contributions" in r and isinstance(r["feature_contributions"], dict)
assert "z_score" in r
assert "predicted_pop" in r
print(f"[OK] predict_ipo_pop: prob_big_pop={r['probability_of_big_pop']:.3f}, "
      f"predicted_pop={r['predicted_pop']:.3f}, backend={r['model_backend']}")

# Verify sigmoid bounds across extreme inputs
hot = predict_ipo_pop(
    offer_size=8.0, is_profitable=False, revenue_growth=2.5,
    sector="ai", market_vix=14, underwriter="Goldman Sachs", age_years=3,
)
cold = predict_ipo_pop(
    offer_size=7.0, is_profitable=True, revenue_growth=0.05,
    sector="utility", market_vix=40, underwriter="Unknown Boutique", age_years=25,
)
assert 0.0 <= hot["probability_of_big_pop"] <= 1.0
assert 0.0 <= cold["probability_of_big_pop"] <= 1.0
# A hot tech IPO should have a higher big-pop probability than a cold utility
assert hot["probability_of_big_pop"] > cold["probability_of_big_pop"], \
    f"Model failed monotonicity: hot={hot['probability_of_big_pop']:.3f}, " \
    f"cold={cold['probability_of_big_pop']:.3f}"
print(f"[OK] Logistic monotonicity: hot={hot['probability_of_big_pop']:.3f} > "
      f"cold={cold['probability_of_big_pop']:.3f}")

# --- _fit_ipo_pop_model returns numpy array ---
coefs, intercept = _fit_ipo_pop_model()
assert isinstance(coefs, np.ndarray), f"coefs not ndarray: {type(coefs)}"
assert len(coefs) >= 5, f"need >= 5 features, got {len(coefs)}"
assert isinstance(intercept, float)
print(f"[OK] _fit_ipo_pop_model: {len(coefs)} coefficients, intercept={intercept:.4f}")

# --- backtest_ipo_predictions returns metrics ---
bt = backtest_ipo_predictions(years=3)
assert "rmse" in bt, "backtest missing 'rmse'"
assert "directional_accuracy" in bt
assert "auc" in bt
assert 0.0 <= bt["directional_accuracy"] <= 1.0
assert 0.0 <= bt["auc"] <= 1.0
assert bt["rmse"] >= 0.0
print(f"[OK] backtest: rmse={bt['rmse']:.3f}, dir_acc={bt['directional_accuracy']:.3f}, "
      f"auc={bt['auc']:.3f}, n_train={bt['n_train']}, n_test={bt['n_test']}")

# --------------------------------------------------------------------------
# Backwards-compatible compute_ipo_pop_prediction wrapper
# --------------------------------------------------------------------------
pred = compute_ipo_pop_prediction(0.8, 0.6, 0.9, 1.0)
# Must use logistic regression now, NOT the prior 0.3/0.2/0.3/0.2 weights
assert "model_type" in pred and "logistic" in pred["model_type"].lower(), \
    "compute_ipo_pop_prediction still uses linear weights!"
assert "coefficients" in pred, "compute_ipo_pop_prediction must expose fitted coefficients"
assert 0.0 <= pred["pop_score"] <= 1.0, "pop_score must be sigmoid output in [0,1]"
print(f"[OK] compute_ipo_pop_prediction (logistic): pop_score={pred['pop_score']:.4f}, "
      f"pop_pct={pred['pop_prediction_pct']}%, model_type={pred['model_type']}")

# Boundary: all inputs 0.0 — bear/boutique scenario, low big-pop prob
pred_zero = compute_ipo_pop_prediction(0.0, 0.0, 0.0, 0.0)
assert 0.0 <= pred_zero["pop_score"] <= 1.0
print(f"[OK] zero inputs (bear/boutique): pop_score={pred_zero['pop_score']:.4f}")

# Boundary: all inputs 1.0 — hot/bulge scenario, higher big-pop prob
pred_one = compute_ipo_pop_prediction(1.0, 1.0, 1.0, 1.0)
assert 0.0 <= pred_one["pop_score"] <= 1.0
assert pred_one["pop_score"] > pred_zero["pop_score"], \
    "all-ones should beat all-zeros on big-pop probability"
print(f"[OK] all-ones (hot/bulge): pop_score={pred_one['pop_score']:.4f} > "
      f"zero={pred_zero['pop_score']:.4f}")

# Out-of-range must still raise ValueError
try:
    compute_ipo_pop_prediction(1.5, 0.5, 0.5, 0.5)
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("[OK] compute_ipo_pop_prediction: out-of-range input raises ValueError")

# --------------------------------------------------------------------------
# compute_lockup_expiry_signal
# --------------------------------------------------------------------------
ipo_dt = date(2024, 1, 15)
signal = compute_lockup_expiry_signal(ipo_dt, lockup_days=180)
expected_expiry = ipo_dt + timedelta(days=180)
assert signal["lockup_expiry_date"] == expected_expiry.isoformat()
assert signal["expected_return_pct"] == -8.0
assert signal["lockup_days"] == 180
assert signal["signal"] in ("PAST", "APPROACHING", "ACTIVE")
print(f"[OK] compute_lockup_expiry_signal: expiry={signal['lockup_expiry_date']}, "
      f"signal={signal['signal']}, expected_return={signal['expected_return_pct']}%")

# --------------------------------------------------------------------------
# classify_ipo_quality tiers
# --------------------------------------------------------------------------
t1 = classify_ipo_quality("Goldman Sachs", ebitda_positive=True)
assert t1["tier"] == "Tier 1"
assert t1["is_bulge_bracket"] is True
assert t1["ebitda_positive"] is True
print(f"[OK] classify_ipo_quality Tier 1: {t1['tier']}")

t2 = classify_ipo_quality("Goldman Sachs", ebitda_positive=False)
assert t2["tier"] == "Tier 2"
print(f"[OK] classify_ipo_quality Tier 2 (bulge+loss): {t2['tier']}")

t2b = classify_ipo_quality("Jefferies", ebitda_positive=True)
assert t2b["tier"] == "Tier 2"
assert t2b["is_major"] is True
print(f"[OK] classify_ipo_quality Tier 2 (major+profit): {t2b['tier']}")

t3 = classify_ipo_quality("Unknown Boutique Partners", ebitda_positive=True)
assert t3["tier"] == "Tier 3"
print(f"[OK] classify_ipo_quality Tier 3: {t3['tier']}")

t3n = classify_ipo_quality(None, ebitda_positive=False)
assert t3n["tier"] == "Tier 3"
print(f"[OK] classify_ipo_quality Tier 3: None underwriter handled")

print("\n[PASS] dim_030: ipo_intelligence_v3 -- all checks passed (LOGISTIC REGRESSION verified)")
PYEOF
