#!/bin/bash
# dim_038: Bond analytics engine (QuantLib: DV01, OAS, z-spread, duration, convexity)
# Tests module structure, pure FRED->tenor mapping, and graceful QuantLib fallback
set -e

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# 1. Module imports cleanly even without QuantLib
import sentinel.sbx.bond_analytics as ba
print(f"[OK] bond_analytics imported (QL available: {ba._QL_AVAILABLE})")

# 2. All four public functions exist
assert hasattr(ba, 'build_yield_curve'), "Missing build_yield_curve"
assert hasattr(ba, 'price_fixed_rate_bond'), "Missing price_fixed_rate_bond"
assert hasattr(ba, 'compute_z_spread'), "Missing compute_z_spread"
assert hasattr(ba, 'build_treasury_rates_from_fred'), "Missing build_treasury_rates_from_fred"
print("[OK] All 4 bond analytics functions present")

# 3. _require_ql raises descriptive ImportError when QL missing (not a bare crash)
if not ba._QL_AVAILABLE:
    try:
        from datetime import date
        ba.build_yield_curve({'10Y': 4.5}, date.today())
        assert False, "Should have raised ImportError"
    except ImportError as e:
        assert "QuantLib" in str(e), f"Error message should mention QuantLib: {e}"
        print("[OK] Missing QuantLib raises descriptive ImportError (not a crash)")

# 4. Pure helper: build_treasury_rates_from_fred (no QuantLib required)
class _FakePoint:
    def __init__(self, t, v):
        self.time = t
        self.value = v

fred_data = {
    'DGS1MO': [_FakePoint('2024-01-01', 5.30)],
    'DGS3MO': [_FakePoint('2024-01-01', 5.35)],
    'DGS6MO': [_FakePoint('2024-01-01', 5.25)],
    'DGS1':   [_FakePoint('2024-01-01', 5.00)],
    'DGS2':   [_FakePoint('2024-01-01', 4.70)],
    'DGS5':   [_FakePoint('2024-01-01', 4.40)],
    'DGS10':  [_FakePoint('2024-01-01', 4.20)],
    'DGS20':  [_FakePoint('2024-01-01', 4.50)],
    'DGS30':  [_FakePoint('2024-01-01', 4.35)],
}
rates = ba.build_treasury_rates_from_fred(fred_data)
assert isinstance(rates, dict)
assert rates.get('10Y') == 4.20, f"10Y rate mismatch: {rates.get('10Y')}"
assert rates.get('1M') == 5.30, f"1M rate mismatch: {rates.get('1M')}"
assert rates.get('30Y') == 4.35, f"30Y rate mismatch: {rates.get('30Y')}"
assert '2Y' in rates and '5Y' in rates
print(f"[OK] FRED->tenor mapping: {len(rates)} tenors, 10Y={rates['10Y']}%, 30Y={rates['30Y']}%")

# 5. Empty FRED data returns empty dict
empty_rates = ba.build_treasury_rates_from_fred({})
assert empty_rates == {}
print("[OK] Empty FRED input handled gracefully")

# 6. Unknown series IDs skipped silently
partial = ba.build_treasury_rates_from_fred({'UNKNOWN_SERIES': [_FakePoint('2024-01-01', 3.0)]})
assert partial == {}
print("[OK] Unknown FRED series IDs skipped gracefully")

# 7. Module-level availability flag
assert isinstance(ba._QL_AVAILABLE, bool)
print(f"[OK] _QL_AVAILABLE={ba._QL_AVAILABLE} (module-level flag present)")

print("\n[PASS] dim_038: Bond analytics engine -- all checks passed")
PYEOF
