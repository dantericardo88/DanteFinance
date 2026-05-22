#!/usr/bin/env bash
# dim_113: Exotic options pricing — barrier, Asian, lookback, digital, chooser
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os, math
sys.path.insert(0, os.getcwd())

from sentinel.sbx.exotic_options_v3 import (
    BarrierOption, AsianOption, LookbackOption,
    ExoticPricer, price_barrier, price_asian_geometric,
    price_asian_mc, price_lookback_mc, price_digital,
    price_exotic, bs_call, bs_put,
)

pricer = ExoticPricer()

# ── Parameters ────────────────────────────────────────────────────────────────
S, K, H, T, r, sigma = 100.0, 100.0, 90.0, 1.0, 0.05, 0.2
discount = math.exp(-r * T)  # ≈ 0.9512

# ── 1. Barrier down-and-out call: 0 < price < vanilla call ───────────────────
vanilla_call = bs_call(S, K, T, r, sigma)
barrier_out = price_barrier(S, K, H, T, r, sigma, barrier_type='down-and-out', option_type='call')
assert barrier_out > 0, f"Down-and-out call must be > 0, got {barrier_out:.4f}"
assert barrier_out < vanilla_call, (
    f"Barrier reduces value: barrier_out={barrier_out:.4f} must be < vanilla={vanilla_call:.4f}"
)
print(f"[OK] Barrier down-and-out call: {barrier_out:.4f}  (vanilla={vanilla_call:.4f})")

# ── 2. In-Out parity: down-and-in + down-and-out = vanilla call ───────────────
barrier_in = price_barrier(S, K, H, T, r, sigma, barrier_type='down-and-in', option_type='call')
parity_err = abs(barrier_in + barrier_out - vanilla_call)
assert parity_err < 0.01, (
    f"In-out parity failed: in={barrier_in:.4f} + out={barrier_out:.4f} = "
    f"{barrier_in + barrier_out:.4f}, vanilla={vanilla_call:.4f}, err={parity_err:.4f}"
)
print(f"[OK] In-out parity: in={barrier_in:.4f} + out={barrier_out:.4f} ~= vanilla={vanilla_call:.4f}  "
      f"(err={parity_err:.5f})")

# ── 3. Asian geometric < vanilla call ─────────────────────────────────────────
asian_geo = price_asian_geometric(S, K, T, r, sigma)
assert asian_geo > 0, f"Geometric Asian call must be > 0, got {asian_geo:.4f}"
assert asian_geo < vanilla_call, (
    f"Asian geometric must be cheaper than vanilla: {asian_geo:.4f} < {vanilla_call:.4f}"
)
print(f"[OK] Asian geometric call: {asian_geo:.4f}  (vanilla={vanilla_call:.4f})")

# ── 4. Asian arithmetic MC close to geometric ─────────────────────────────────
asian_arith = price_asian_mc(S, K, T, r, sigma, n_sims=10000, n_steps=252, seed=42)
assert asian_arith > 0, f"Arithmetic Asian call must be > 0, got {asian_arith:.4f}"
diff = abs(asian_arith - asian_geo)
assert diff < 2.0, (
    f"Arithmetic and geometric Asian should be close: arith={asian_arith:.4f}, "
    f"geo={asian_geo:.4f}, diff={diff:.4f}"
)
print(f"[OK] Asian arithmetic MC: {asian_arith:.4f}  |arith - geo| = {diff:.4f} < 2.0")

# ── 5. Lookback floating call > vanilla call ───────────────────────────────────
lookback = price_lookback_mc(S, T, r, sigma)
assert lookback > vanilla_call, (
    f"Lookback should be more expensive than vanilla: "
    f"lookback={lookback:.4f}, vanilla={vanilla_call:.4f}"
)
print(f"[OK] Lookback floating call: {lookback:.4f}  > vanilla={vanilla_call:.4f}")

# ── 6. Digital cash-or-nothing: 0 < price < e^(-rT) ─────────────────────────
cash_call = price_digital(S, K, T, r, sigma, digital_type='cash', option_type='call')
assert cash_call > 0, f"Cash-or-nothing call must be > 0, got {cash_call:.4f}"
assert cash_call < discount, (
    f"Cash-or-nothing call must be < discount={discount:.4f}, got {cash_call:.4f}"
)
print(f"[OK] Digital cash-or-nothing call: {cash_call:.4f}  (e^(-rT)={discount:.4f})")

# ── 7. Digital asset-or-nothing > cash-or-nothing ────────────────────────────
asset_call = price_digital(S, K, T, r, sigma, digital_type='asset', option_type='call')
assert asset_call > cash_call, (
    f"Asset-or-nothing > cash-or-nothing: asset={asset_call:.4f}, cash={cash_call:.4f}"
)
print(f"[OK] Digital asset-or-nothing call: {asset_call:.4f}  > cash={cash_call:.4f}")

# ── 8. Put-call parity for digital: cash_put + cash_call = e^(-rT) ───────────
cash_put = price_digital(S, K, T, r, sigma, digital_type='cash', option_type='put')
parity_digital = abs(cash_call + cash_put - discount)
assert parity_digital < 0.001, (
    f"Digital put-call parity failed: call={cash_call:.4f} + put={cash_put:.4f} = "
    f"{cash_call + cash_put:.4f}, e^(-rT)={discount:.4f}, err={parity_digital:.5f}"
)
print(f"[OK] Digital put-call parity: call={cash_call:.4f} + put={cash_put:.4f} = "
      f"{cash_call + cash_put:.4f} ~= e^(-rT)={discount:.4f}  (err={parity_digital:.5f})")

# ── 9. Chooser option > both call and put alone ────────────────────────────────
t_c = 0.5
chooser = pricer.chooser(S, K, T, t_c, r, sigma)
call_alone = bs_call(S, K, T, r, sigma)
put_alone  = bs_put(S, K, T, r, sigma)
assert chooser >= max(call_alone, put_alone), (
    f"Chooser must be >= max(call, put): chooser={chooser:.4f}, "
    f"call={call_alone:.4f}, put={put_alone:.4f}"
)
print(f"[OK] Chooser option: {chooser:.4f}  (call={call_alone:.4f}, put={put_alone:.4f})")

# ── 10. BarrierOption dataclass API ──────────────────────────────────────────
bo = BarrierOption(S=S, K=K, H=H, T=T, r=r, sigma=sigma, barrier_type='down-and-out', option_type='call')
bo_price = bo.price()
assert abs(bo_price - barrier_out) < 1e-8, f"BarrierOption.price() mismatch: {bo_price} vs {barrier_out}"
print(f"[OK] BarrierOption dataclass: price={bo_price:.4f}")

# ── 11. AsianOption dataclass API ─────────────────────────────────────────────
ao_geo = AsianOption(S=S, K=K, T=T, r=r, sigma=sigma, averaging='geometric').price()
assert abs(ao_geo - asian_geo) < 1e-8, f"AsianOption.price() geometric mismatch"
ao_arith = AsianOption(S=S, K=K, T=T, r=r, sigma=sigma, averaging='arithmetic', seed=42).price()
assert abs(ao_arith - asian_arith) < 1e-8, f"AsianOption.price() arithmetic mismatch"
delta = AsianOption(S=S, K=K, T=T, r=r, sigma=sigma, averaging='geometric').delta()
assert 0.0 < delta < 1.0, f"Asian geometric call delta must be in (0,1): {delta:.4f}"
print(f"[OK] AsianOption dataclass: geo={ao_geo:.4f}, arith={ao_arith:.4f}, delta={delta:.4f}")

# ── 12. LookbackOption dataclass API ─────────────────────────────────────────
lo = LookbackOption(S=S, K=0.0, T=T, r=r, sigma=sigma, strike_type='floating', option_type='call')
lo_price = lo.price()
assert abs(lo_price - lookback) < 1e-8, f"LookbackOption.price() mismatch"
print(f"[OK] LookbackOption dataclass: price={lo_price:.4f}")

# ── 13. price_exotic generic entry point ──────────────────────────────────────
exotic_barrier = price_exotic('barrier', S=S, K=K, H=H, T=T, r=r, sigma=sigma,
                               barrier_type='down-and-out', option_type='call')
assert abs(exotic_barrier - barrier_out) < 1e-8, "price_exotic barrier mismatch"
exotic_geo = price_exotic('asian_geometric', S=S, K=K, T=T, r=r, sigma=sigma)
assert abs(exotic_geo - asian_geo) < 1e-8, "price_exotic asian_geometric mismatch"
print(f"[OK] price_exotic (exotic_type=): barrier={exotic_barrier:.4f}, asian_geo={exotic_geo:.4f}")

print("\n[PASS] dim_113: Exotic options pricing")
PYEOF
