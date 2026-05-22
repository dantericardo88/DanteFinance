#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
python - <<'PYEOF'
import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
np.random.seed(42)

from sentinel.sai.satellite_signals_v3 import (
    SatelliteObservation, NDVIAnalytics, ChangeDetector,
    EconomicActivityProxy, SatelliteSignalGenerator,
    ndvi, change_detection, economic_activity
)

H, W = 64, 64
red = np.random.randint(50, 150, (H, W)).astype(float)
nir = np.random.randint(100, 200, (H, W)).astype(float)
green = np.random.randint(50, 120, (H, W)).astype(float)
obs = SatelliteObservation(timestamp=0.0, red_band=red, nir_band=nir, green_band=green)

# NDVI
ndvi_map = ndvi(red, nir)
assert ndvi_map.shape == (H, W), f"Shape {ndvi_map.shape}"
assert np.all(ndvi_map >= -1) and np.all(ndvi_map <= 1), "NDVI out of range"
# Where NIR > Red -> positive NDVI
veg_mask = nir > red
assert np.all(ndvi_map[veg_mask] > 0), "NDVI > 0 where NIR > Red"

# NDVIAnalytics
analytics = NDVIAnalytics()
ndvi_img = analytics.compute(obs)
assert ndvi_img.shape == (H, W)
obs_list = [SatelliteObservation(float(t), red*(1+t*0.01), nir*(1-t*0.01), green, "test")
            for t in range(8)]
ts = analytics.time_series(obs_list)
assert len(ts) == 8
slope = analytics.trend(ts)
assert np.isfinite(slope)

# ChangeDetector
cd = ChangeDetector()
before = np.zeros((H, W))
after = np.zeros((H, W))
after[:16, :16] = 200.0  # change top-left quadrant
mask = cd.detect(before, after, threshold_sigma=2.0)
assert mask.shape == (H, W), f"Mask shape {mask.shape}"
# Top-left region should be detected as changed
changed_in_region = mask[:16, :16].mean()
changed_outside = mask[16:, 16:].mean()
assert changed_in_region > changed_outside, "Changed region must have more detections"
mag = cd.change_magnitude(before, after)
assert mag > 0, f"Change magnitude should be > 0, got {mag}"
frac = cd.change_fraction(before, after)
assert 0 < frac <= 1, f"Change fraction {frac} out of range"

# EconomicActivityProxy
eap = EconomicActivityProxy()
bright = np.full((H, W), 200.0)
dark = np.full((H, W), 50.0)
ai_bright = eap.activity_index(bright)
ai_dark = eap.activity_index(dark)
assert ai_bright > ai_dark, "Bright image should have higher activity index"
occ_dark = eap.parking_occupancy(dark, dark_threshold=80)
occ_bright = eap.parking_occupancy(bright, dark_threshold=80)
assert occ_dark > occ_bright, "Mostly-dark image -> higher occupancy"

# SatelliteSignalGenerator
sg = SatelliteSignalGenerator()
ndvi_ts = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
sig = sg.ndvi_signal(ndvi_ts, window=4)
assert np.isfinite(sig), f"NDVI signal not finite: {sig}"
act_ts = np.array([0.3, 0.32, 0.28, 0.35, 0.70])  # spike at end
act_sig = sg.activity_signal(act_ts)
assert np.isfinite(act_sig)
imgs = [np.random.rand(H, W) * i for i in range(1, 6)]
chg = sg.change_signal(imgs)
assert len(chg) == 4  # n-1 frame differences
composite = sg.composite_signal(obs_list)
assert np.isfinite(composite)

# Module-level functions
ea = economic_activity(bright)
assert 0 <= ea <= 1
cd_mask = change_detection(before, after)
assert cd_mask.dtype == bool or cd_mask.dtype == np.bool_

print(f"NDVI range: [{ndvi_map.min():.3f}, {ndvi_map.max():.3f}]")
print(f"Change fraction: {frac:.3f}, Activity bright={ai_bright:.3f} > dark={ai_dark:.3f}")
print(f"NDVI signal: {sig:.4f}, Activity signal: {act_sig:.4f}")
print("[PASS] dim_130: Satellite imagery alt signals framework")
PYEOF
