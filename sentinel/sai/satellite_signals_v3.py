"""
sentinel/sai/satellite_signals_v3.py
Satellite imagery + computer vision alternative signals framework.
Pure math — processes numpy arrays representing image data, no actual downloads.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SatelliteObservation:
    timestamp: float
    red_band: np.ndarray      # shape (H, W), values 0-255
    nir_band: np.ndarray      # near-infrared
    green_band: np.ndarray
    location: str = 'unknown'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red + 1e-8), range [-1, 1]"""
    red_f = red.astype(float)
    nir_f = nir.astype(float)
    result = (nir_f - red_f) / (nir_f + red_f + 1e-8)
    return np.clip(result, -1.0, 1.0)


def change_detection(before: np.ndarray, after: np.ndarray,
                     sigma: float = 2.0) -> np.ndarray:
    """Returns boolean mask of changed pixels (diff > sigma * std of diff)."""
    diff = after.astype(float) - before.astype(float)
    std = np.std(diff)
    if std < 1e-12:
        return np.zeros(before.shape, dtype=bool)
    threshold = sigma * std
    return np.abs(diff) > threshold


def economic_activity(pixel_intensities: np.ndarray) -> float:
    """Normalized mean intensity: mean(pixels) / 255, clamped to [0, 1]."""
    val = float(np.mean(pixel_intensities)) / 255.0
    return float(np.clip(val, 0.0, 1.0))


# ---------------------------------------------------------------------------
# NDVIAnalytics
# ---------------------------------------------------------------------------

class NDVIAnalytics:
    def compute(self, obs: SatelliteObservation) -> np.ndarray:
        """NDVI = (NIR - Red) / (NIR + Red), range [-1, 1]"""
        return ndvi(obs.red_band, obs.nir_band)

    def time_series(self, observations: List[SatelliteObservation]) -> np.ndarray:
        """Mean NDVI per observation."""
        return np.array([float(np.mean(self.compute(obs))) for obs in observations])

    def trend(self, ndvi_series: np.ndarray) -> float:
        """OLS slope of NDVI over time (index as time axis)."""
        n = len(ndvi_series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = ndvi_series.mean()
        numerator = float(np.sum((x - x_mean) * (ndvi_series - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))
        if abs(denominator) < 1e-12:
            return 0.0
        return numerator / denominator


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    def detect(self, before: np.ndarray, after: np.ndarray,
               threshold_sigma: float = 2.0) -> np.ndarray:
        """Boolean mask of pixels that changed beyond threshold_sigma std devs."""
        return change_detection(before, after, sigma=threshold_sigma)

    def change_magnitude(self, before: np.ndarray, after: np.ndarray) -> float:
        """Mean absolute pixel difference."""
        return float(np.mean(np.abs(after.astype(float) - before.astype(float))))

    def change_fraction(self, before: np.ndarray, after: np.ndarray,
                        threshold_sigma: float = 2.0) -> float:
        """Fraction of pixels that changed."""
        mask = self.detect(before, after, threshold_sigma=threshold_sigma)
        return float(mask.mean())


# ---------------------------------------------------------------------------
# EconomicActivityProxy
# ---------------------------------------------------------------------------

class EconomicActivityProxy:
    def activity_index(self, intensity_map: np.ndarray) -> float:
        """Mean pixel intensity normalized to [0, 1]."""
        return economic_activity(intensity_map)

    def parking_occupancy(self, image: np.ndarray,
                          dark_threshold: int = 80) -> float:
        """Fraction of pixels below dark_threshold (proxies cars vs empty spaces)."""
        return float(np.mean(image < dark_threshold))

    def port_congestion(self, vessel_pixels: np.ndarray) -> float:
        """Fraction of bright pixels (vessels) in port region (above mid-range 128)."""
        return float(np.mean(vessel_pixels > 128))


# ---------------------------------------------------------------------------
# SatelliteSignalGenerator
# ---------------------------------------------------------------------------

class SatelliteSignalGenerator:
    def __init__(self):
        self._ndvi_analytics = NDVIAnalytics()
        self._change_detector = ChangeDetector()
        self._eap = EconomicActivityProxy()

    def ndvi_signal(self, ndvi_series: np.ndarray, window: int = 4) -> float:
        """Momentum: mean(last window) - mean(prior window) of NDVI."""
        n = len(ndvi_series)
        if n < 2 * window:
            # fall back: latest half vs first half
            half = max(1, n // 2)
            return float(np.mean(ndvi_series[half:]) - np.mean(ndvi_series[:half]))
        later = ndvi_series[-window:]
        prior = ndvi_series[-2 * window:-window]
        return float(np.mean(later) - np.mean(prior))

    def activity_signal(self, activity_series: np.ndarray) -> float:
        """Z-score of latest activity vs rolling mean."""
        if len(activity_series) < 2:
            return 0.0
        hist = activity_series[:-1]
        latest = float(activity_series[-1])
        mean = float(np.mean(hist))
        std = float(np.std(hist))
        if std < 1e-12:
            return 0.0
        return (latest - mean) / std

    def change_signal(self, images: List[np.ndarray]) -> np.ndarray:
        """Frame-to-frame change magnitudes (n-1 values)."""
        if len(images) < 2:
            return np.array([])
        mags = []
        for i in range(len(images) - 1):
            mags.append(self._change_detector.change_magnitude(images[i], images[i + 1]))
        return np.array(mags)

    def composite_signal(self, obs: List[SatelliteObservation]) -> float:
        """Weighted NDVI trend + activity z-score."""
        if not obs:
            return 0.0
        ndvi_ts = self._ndvi_analytics.time_series(obs)
        ndvi_trend = self._ndvi_analytics.trend(ndvi_ts)

        activity_ts = np.array([
            self._eap.activity_index(o.nir_band) for o in obs
        ])
        act_z = self.activity_signal(activity_ts)

        # Weighted combination: 0.6 trend (normalised) + 0.4 activity z-score
        # Normalise trend to roughly the same scale as z-score
        norm_trend = float(np.clip(ndvi_trend * 100.0, -5.0, 5.0))
        return 0.6 * norm_trend + 0.4 * act_z


# ---------------------------------------------------------------------------
# Backward-compatible aliases expected by the old dim_130.sh stub
# ---------------------------------------------------------------------------

class SatelliteSignal:
    """Thin compatibility shim for legacy tests."""
    pass


class ParkingLotCounter:
    """Thin compatibility shim for legacy tests."""
    pass


def compute_satellite_alpha(obs: Optional[SatelliteObservation] = None) -> float:
    """Stub for legacy compatibility."""
    return 0.0
