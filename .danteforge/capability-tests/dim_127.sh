#!/usr/bin/env bash
# dim_127: Customer concentration & contract-win intelligence analytics
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

try:
    from sentinel.spm.customer_concentration_v3 import (
        Customer,
        ContractOpportunity,
        ConcentrationRisk,
        CustomerConcentrationAnalyzer,
        PipelineAnalytics,
        hhi,
        customer_lifetime_value,
        revenue_at_risk,
        pipeline_expected_value,
    )
except ImportError as e:
    print(f"[NOT-BUILT] dim_127: import failed -- {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# Setup: 5 customers totalling $100M revenue
# ACME=40%, BigCorp=25%, MidCo=20%, SmallA=10%, SmallB=5%
# ------------------------------------------------------------------
TOTAL_REV = 100_000_000.0

customers = [
    Customer("ACME",    40_000_000.0, years_as_customer=5.0, retention_probability=0.90, gross_margin_pct=0.30, contract_end_year=2026, industry="tech"),
    Customer("BigCorp", 25_000_000.0, years_as_customer=3.0, retention_probability=0.85, gross_margin_pct=0.25, contract_end_year=2027, industry="finance"),
    Customer("MidCo",   20_000_000.0, years_as_customer=4.0, retention_probability=0.80, gross_margin_pct=0.28, contract_end_year=2026, industry="retail"),
    Customer("SmallA",  10_000_000.0, years_as_customer=2.0, retention_probability=0.75, gross_margin_pct=0.22, contract_end_year=2025, industry="tech"),
    Customer("SmallB",   5_000_000.0, years_as_customer=1.0, retention_probability=0.70, gross_margin_pct=0.20, contract_end_year=2028, industry="healthcare"),
]

analyzer = CustomerConcentrationAnalyzer(customers, TOTAL_REV)

# ------------------------------------------------------------------
# Test 1: HHI should be in (0.25, 0.50)
# Hand calculation: 0.40^2+0.25^2+0.20^2+0.10^2+0.05^2
#                 = 0.16+0.0625+0.04+0.01+0.0025 = 0.275
# ------------------------------------------------------------------
hhi_val = analyzer.hhi()
assert 0.25 < hhi_val < 0.50, f"HHI {hhi_val} not in (0.25, 0.50)"
print(f"[OK] HHI = {hhi_val:.4f} (expected ~0.275, range 0.25-0.50)")

# ------------------------------------------------------------------
# Test 2: Standalone hhi() function
# ------------------------------------------------------------------
hhi_standalone = hhi([40, 25, 20, 10, 5])
assert abs(hhi_standalone - hhi_val) < 1e-6, \
    f"Standalone hhi() {hhi_standalone} != analyzer.hhi() {hhi_val}"
print(f"[OK] Standalone hhi() consistent: {hhi_standalone:.4f}")

# ------------------------------------------------------------------
# Test 3: top_customer_pct == 40.0
# ------------------------------------------------------------------
risk = analyzer.assess()
assert abs(risk.top_customer_pct - 40.0) < 1e-6, \
    f"top_customer_pct {risk.top_customer_pct} != 40.0"
print(f"[OK] top_customer_pct = {risk.top_customer_pct:.1f}%")

# ------------------------------------------------------------------
# Test 4: CR5 == 100.0 (all 5 customers)
# ------------------------------------------------------------------
cr5 = analyzer.top_n_concentration(n=5)
assert abs(cr5 - 100.0) < 1e-6, f"CR5 {cr5} != 100.0"
print(f"[OK] CR5 = {cr5:.1f}%")

# ------------------------------------------------------------------
# Test 5: CLV for ACME (margin=30%, retention=0.90, discount=0.10)
# CLV = 12M * (0.90 / (1 + 0.10 - 0.90)) = 12M * (0.90 / 0.20) = 54M
# ------------------------------------------------------------------
acme = customers[0]
clv_acme = analyzer.clv(acme, discount_rate=0.10)
assert clv_acme > 0, f"CLV for ACME should be positive, got {clv_acme}"
expected_clv = 12_000_000.0 * (0.90 / 0.20)  # = 54M
assert abs(clv_acme - expected_clv) < 1.0, \
    f"CLV ACME {clv_acme:.0f} != expected {expected_clv:.0f}"
print(f"[OK] CLV (ACME, retention=0.90, margin=30%) = ${clv_acme:,.0f}")

# ------------------------------------------------------------------
# Test 6: Standalone customer_lifetime_value
# ------------------------------------------------------------------
clv_fn = customer_lifetime_value(12_000_000.0, 0.85, 0.10)
assert clv_fn > 0, "Standalone CLV function returned non-positive value"
print(f"[OK] customer_lifetime_value() = ${clv_fn:,.0f}")

# ------------------------------------------------------------------
# Test 7: revenue_at_risk for top customer (ACME, churn_prob = 0.10)
# ACME retention=0.90 → churn=0.10 → at_risk = 40M * 0.10 = 4M
# ------------------------------------------------------------------
rar_analyzer = analyzer.revenue_at_risk()
expected_rar = 40_000_000.0 * 0.10  # = 4_000_000
assert abs(rar_analyzer - expected_rar) < 1.0, \
    f"revenue_at_risk {rar_analyzer:.0f} != expected {expected_rar:.0f}"
print(f"[OK] revenue_at_risk (ACME, churn 10%) = ${rar_analyzer:,.0f}")

# Standalone function
rar_fn = revenue_at_risk(40_000_000.0, 0.10)
assert abs(rar_fn - 4_000_000.0) < 1.0, f"Standalone revenue_at_risk {rar_fn}"
print(f"[OK] Standalone revenue_at_risk(40M, 0.10) = ${rar_fn:,.0f}")

# ------------------------------------------------------------------
# Test 8: assess() returns ConcentrationRisk with correct category
# HHI ~0.275 → 'high' (0.25 ≤ HHI < 0.40)
# ------------------------------------------------------------------
assert isinstance(risk, ConcentrationRisk), "assess() must return ConcentrationRisk"
assert risk.concentration_category in ("high", "critical"), \
    f"Expected 'high' or 'critical' for HHI {risk.hhi:.3f}, got '{risk.concentration_category}'"
assert risk.clv_weighted_avg > 0, "CLV weighted avg must be positive"
print(f"[OK] ConcentrationRisk category = '{risk.concentration_category}' "
      f"(HHI={risk.hhi:.4f}, CLV_wavg=${risk.clv_weighted_avg:,.0f})")

# ------------------------------------------------------------------
# Test 9: Pipeline analytics — 3 deals
# values: 1M/2M/3M, probs: 0.8/0.5/0.3
# pipeline_value = 1M*0.8 + 2M*0.5 + 3M*0.3 = 0.8+1.0+0.9 = 2.7M
# ------------------------------------------------------------------
opps = [
    ContractOpportunity("Deal A", 1_000_000.0, 0.8, "negotiation", 30),
    ContractOpportunity("Deal B", 2_000_000.0, 0.5, "proposal",    90),
    ContractOpportunity("Deal C", 3_000_000.0, 0.3, "qualified",  120),
]
pipeline = PipelineAnalytics(opps)
pv = pipeline.pipeline_value()
expected_pv = 2_700_000.0
assert abs(pv - expected_pv) < 1.0, f"pipeline_value {pv:.0f} != expected {expected_pv:.0f}"
print(f"[OK] pipeline_value = ${pv:,.0f} (expected $2,700,000)")

# Standalone
pv_fn = pipeline_expected_value(opps)
assert abs(pv_fn - expected_pv) < 1.0, f"pipeline_expected_value {pv_fn}"
print(f"[OK] pipeline_expected_value() = ${pv_fn:,.0f}")

# Stage breakdown
sb = pipeline.stage_breakdown()
assert isinstance(sb, dict), "stage_breakdown must return dict"
print(f"[OK] stage_breakdown keys: {list(sb.keys())}")

# Win rate (revenue-weighted)
wr = pipeline.win_rate()
assert 0 < wr <= 1, f"win_rate {wr} not in (0,1]"
print(f"[OK] win_rate = {wr:.3f}")

# Industry breakdown
ib = analyzer.industry_breakdown()
assert abs(sum(ib.values()) - 1.0) < 1e-6, "industry_breakdown fractions must sum to 1"
print(f"[OK] industry_breakdown: {ib}")

# Renewal risk calendar
calendar = analyzer.renewal_risk_calendar(current_year=2025)
assert isinstance(calendar, list) and len(calendar) == 5
print(f"[OK] renewal_risk_calendar: {len(calendar)} entries, first={calendar[0]['customer']}")

print("\n[PASS] dim_127: Customer concentration analytics")
PYEOF
