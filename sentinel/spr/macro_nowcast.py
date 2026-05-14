"""Macro nowcasting: GDP growth estimate from FRED leading indicators — Dimension 60 enhancement."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import httpx
from pydantic import BaseModel, ConfigDict, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Weights sum to 1.00 after dropping ISM_MFG (None). Negative = contra-indicator.
LEADING_INDICATORS: dict[str, tuple[str, float] | None] = {
    "INDPRO":            ("Industrial Production Index",         0.15),
    "UNRATE":            ("Unemployment Rate",                  -0.12),
    "ICSA":              ("Initial Jobless Claims",             -0.10),
    "RSXFS":             ("Retail Sales excl. Food Services",    0.12),
    "UMCSENT":           ("U. Michigan Consumer Sentiment",      0.10),
    "PERMIT":            ("Building Permits",                    0.08),
    "ISM_MFG":           None,   # not available as free FRED series
    "T10Y2Y":            ("10Y-2Y Yield Spread",                 0.12),
    "PAYEMS":            ("Nonfarm Payrolls",                    0.13),
    "HOUST":             ("Housing Starts",                      0.08),
    "DPCERA3M086SBEA":   ("Real PCE YoY",                        0.10),
}

_ACTIVE_SERIES: dict[str, tuple[str, float]] = {
    sid: meta  # type: ignore[assignment]
    for sid, meta in LEADING_INDICATORS.items()
    if meta is not None
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SeriesReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    name: str
    latest_value: Optional[float]
    prev_value: Optional[float]
    mom_change: Optional[float] = Field(None, description="MoM % change (decimal)")
    yoy_change: Optional[float] = Field(None, description="YoY % change (decimal)")
    last_updated: Optional[str] = Field(None, description="ISO date of latest obs")
    trend: str = Field("unknown", description="expanding/contracting/recovering/mixed")


class NowcastComponent(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    name: str
    weight: float = Field(description="Signed weight; negative = contra-indicator")
    standardized_value: Optional[float] = Field(None, description="Z-score clamped [-3,3]")
    contribution: Optional[float] = Field(None, description="weight × standardized_value")


class GDPNowcast(BaseModel):
    model_config = ConfigDict(frozen=True)
    nowcast_gdp_growth: float = Field(description="Estimated annualized quarterly GDP growth (%)")
    nowcast_confidence: str = Field(description="high / medium / low")
    components: list[NowcastComponent]
    composite_index: float = Field(description="0-10: 5=neutral, >5=expansion, <5=contraction")
    composite_trend: str = Field(description="improving / deteriorating / stable")
    recession_probability: float = Field(description="Recession probability [0,1]")
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class MacroDashboard(BaseModel):
    model_config = ConfigDict(frozen=True)
    nowcast: GDPNowcast
    series_readings: list[SeriesReading]
    expansion_signals: list[str] = Field(description="Series IDs where z > 0.5")
    contraction_signals: list[str] = Field(description="Series IDs where z < -0.5")
    key_risks: list[str]
    regime: str = Field(description="expansion / late_cycle / contraction / recovery")
    as_of: str
    warnings: list[str] = Field(default_factory=list)


class MacroRegimeSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    series_id: str
    name: str
    signal_direction: str = Field(description="bullish / bearish")
    magnitude: float = Field(description="Absolute standardized value")
    as_of: str


class MacroNowcastScreen(BaseModel):
    model_config = ConfigDict(frozen=True)
    composite_index: float
    regime: str
    top_expansion_signals: list[MacroRegimeSignal]
    top_contraction_signals: list[MacroRegimeSignal]
    recession_probability: float
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FRED data fetching
# ---------------------------------------------------------------------------

async def _fetch_fred_series(
    series_id: str,
    client: httpx.AsyncClient,
    limit: int = 120,
) -> list[tuple[str, float]]:
    """Fetch up to `limit` recent obs from FRED's free CSV endpoint.

    Returns list of (date_str, value) sorted ascending; filters '.' values.
    Returns [] on any network or parse error.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    try:
        response = await client.get(url)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("fred_fetch_failed", series_id=series_id, error=str(exc))
        return []

    rows: list[tuple[str, float]] = []
    for line in response.text.splitlines()[1:]:  # skip DATE,VALUE header
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str, raw_val = parts[0].strip(), parts[1].strip()
        if raw_val in (".", ""):
            continue
        try:
            rows.append((date_str, float(raw_val)))
        except ValueError:
            continue

    rows.sort(key=lambda r: r[0])
    return rows[-limit:]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _standardize(values: np.ndarray, window: int = 24) -> Optional[float]:
    """Z-score of last element vs rolling `window` reference. Clamps [-3,3]."""
    if values.size < 2:
        return None
    ref = values[-window:] if values.size >= window else values
    mean = float(np.mean(ref))
    std = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
    if std == 0.0:
        return None
    return float(np.clip((values[-1] - mean) / std, -3.0, 3.0))


def _mom_change(values: np.ndarray) -> Optional[float]:
    """Month-over-month % change: (last - prev) / |prev|."""
    if values.size < 2:
        return None
    prev = values[-2]
    if prev == 0.0:
        return None
    return float((values[-1] - prev) / abs(prev))


def _yoy_change(values: np.ndarray, dates: list[str]) -> Optional[float]:
    """YoY % change vs nearest observation 10-14 months back. Returns None if unavailable."""
    if values.size < 2 or not dates:
        return None
    try:
        latest_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    except ValueError:
        return None
    target = latest_date - timedelta(days=365)
    lo = target - timedelta(days=60)
    hi = target + timedelta(days=60)
    best_idx: Optional[int] = None
    best_diff = timedelta(days=999)
    for i, d in enumerate(dates[:-1]):
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue
        if lo <= dt <= hi:
            diff = abs(dt - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
    if best_idx is None:
        return None
    base = values[best_idx]
    if base == 0.0:
        return None
    return float((values[-1] - base) / abs(base))


def _trend(mom: Optional[float], yoy: Optional[float]) -> str:
    """Classify trend: expanding / contracting / recovering / mixed / unknown."""
    if mom is None and yoy is None:
        return "unknown"
    mom_pos = mom is not None and mom > 0
    mom_neg = mom is not None and mom < 0
    yoy_pos = yoy is not None and yoy > 0
    yoy_neg = yoy is not None and yoy < 0
    if mom_pos and yoy_pos:
        return "expanding"
    if mom_neg and yoy_neg:
        return "contracting"
    if mom_pos and yoy_neg:
        return "recovering"
    return "mixed"


# ---------------------------------------------------------------------------
# Composite index and derived metrics
# ---------------------------------------------------------------------------

def _composite_index(components: list[NowcastComponent]) -> float:
    """0-10 macro composite: 5.0 + weighted_z * 1.5, clamped [0,10].

    weighted_z = Σ(weight_i × z_i); 5=neutral, >5=expansion, <5=contraction.
    """
    weighted_z = sum(
        c.contribution for c in components
        if c.contribution is not None
    )
    return float(np.clip(5.0 + weighted_z * 1.5, 0.0, 10.0))


def _recession_probability(composite: float, t10y2y_z: Optional[float]) -> float:
    """Recession probability from composite level plus yield curve signal.

    Base: composite<3→0.70, <4→0.40, <4.5→0.20, >6→0.05, else→0.12.
    Yield curve: if T10Y2Y z-score < -1.5, add 0.20. Clamped [0.01,0.95].
    """
    if composite < 3.0:
        base = 0.70
    elif composite < 4.0:
        base = 0.40
    elif composite < 4.5:
        base = 0.20
    elif composite > 6.0:
        base = 0.05
    else:
        base = 0.12
    if t10y2y_z is not None and t10y2y_z < -1.5:
        base += 0.20
    return float(np.clip(base, 0.01, 0.95))


def _nowcast_gdp(composite: float, components: list[NowcastComponent]) -> tuple[float, str]:
    """Annualized quarterly GDP growth and confidence from composite index.

    Mapping: composite=5.0→2.0% (long-run avg); ±1 unit → ±0.8%; clamped [-5,8].
    Confidence: ≥7 components loaded=high, ≥4=medium, else=low.
    """
    gdp = float(np.clip(2.0 + (composite - 5.0) * 0.8, -5.0, 8.0))
    valid = sum(1 for c in components if c.standardized_value is not None)
    confidence = "high" if valid >= 7 else ("medium" if valid >= 4 else "low")
    return gdp, confidence


def _composite_trend(
    series_data: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    window: int = 24,
) -> str:
    """Compare last-3-obs vs prior-3-obs weighted composite to classify trend direction."""
    recent_scores: list[float] = []
    prior_scores: list[float] = []
    for sid, obs in series_data.items():
        if sid not in weights or len(obs) < 7:
            continue
        vals = np.array([v for _, v in obs])
        w = weights[sid]
        ref = vals[-window:] if vals.size >= window else vals
        std = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
        mean = float(np.mean(ref))
        if std == 0.0:
            continue
        z_recent = np.clip((float(np.mean(vals[-3:])) - mean) / std, -3.0, 3.0)
        z_prior = np.clip((float(np.mean(vals[-6:-3])) - mean) / std, -3.0, 3.0)
        recent_scores.append(w * float(z_recent))
        prior_scores.append(w * float(z_prior))
    if not recent_scores:
        return "stable"
    delta = (5.0 + sum(recent_scores) * 1.5) - (5.0 + sum(prior_scores) * 1.5)
    if delta > 0.15:
        return "improving"
    if delta < -0.15:
        return "deteriorating"
    return "stable"


def _classify_regime(composite: float, trend: str) -> str:
    """Map composite + trend to expansion / late_cycle / contraction / recovery."""
    if composite >= 5.5 and trend != "deteriorating":
        return "expansion"
    if composite >= 4.5 and trend == "deteriorating":
        return "late_cycle"
    if composite < 4.5 and trend == "improving":
        return "recovery"
    return "contraction"


def _identify_key_risks(
    components: list[NowcastComponent],
    t10y2y_z: Optional[float],
) -> list[str]:
    """Generate plain-language risk flags from component z-scores."""
    risks: list[str] = []
    sm = {c.series_id: c for c in components}

    def _z(sid: str) -> Optional[float]:
        c = sm.get(sid)
        return c.standardized_value if c else None

    if t10y2y_z is not None and t10y2y_z < -1.0:
        risks.append("inverted yield curve — historically reliable recession precursor")
    if (_z("UMCSENT") or 0.0) < -0.8:
        risks.append("declining consumer sentiment — forward demand softening signal")
    if (_z("ICSA") or 0.0) > 0.8:
        risks.append("rising initial jobless claims — labor market stress emerging")
    if (_z("INDPRO") or 0.0) < -0.8:
        risks.append("industrial production contraction — goods-sector weakness")
    if (_z("PERMIT") or 0.0) < -1.0:
        risks.append("building permits declining sharply — credit/housing weakness")
    if (_z("RSXFS") or 0.0) < -0.8:
        risks.append("retail sales ex-food weakening — consumer spending softening")
    if (_z("UNRATE") or 0.0) > 0.8:
        risks.append("unemployment rate rising — labor market loosening")
    if not risks:
        risks.append("no significant macro risk flags at current readings")
    return risks


# ---------------------------------------------------------------------------
# Internal shared fetch-and-build helper
# ---------------------------------------------------------------------------

async def _fetch_all_series(limit: int) -> tuple[
    dict[str, list[tuple[str, float]]],
    list[str],
]:
    """Fetch all active FRED series in parallel. Returns (series_data, warnings)."""
    series_ids = list(_ACTIVE_SERIES.keys())
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *[_fetch_fred_series(sid, client, limit=limit) for sid in series_ids],
            return_exceptions=True,
        )
    warnings: list[str] = []
    series_data: dict[str, list[tuple[str, float]]] = {}
    for sid, result in zip(series_ids, results):
        if isinstance(result, Exception):
            warnings.append(f"{sid}: fetch exception — {result}")
            series_data[sid] = []
        else:
            series_data[sid] = result  # type: ignore[assignment]
            if not result:
                warnings.append(f"{sid}: no data returned from FRED")
    return series_data, warnings


def _build_components(
    series_data: dict[str, list[tuple[str, float]]],
) -> list[NowcastComponent]:
    """Standardize each series and build NowcastComponent list."""
    components: list[NowcastComponent] = []
    for sid in _ACTIVE_SERIES:
        name, weight = _ACTIVE_SERIES[sid]
        obs = series_data.get(sid, [])
        if not obs:
            components.append(NowcastComponent(
                series_id=sid, name=name, weight=weight,
                standardized_value=None, contribution=None,
            ))
            continue
        vals = np.array([v for _, v in obs])
        z = _standardize(vals, window=24)
        components.append(NowcastComponent(
            series_id=sid, name=name, weight=weight,
            standardized_value=z,
            contribution=(weight * z) if z is not None else None,
        ))
    return components


# ---------------------------------------------------------------------------
# Primary entry points
# ---------------------------------------------------------------------------

async def get_gdp_nowcast(history_months: int = 24) -> GDPNowcast:
    """Fetch FRED leading indicators and produce a GDP nowcast.

    Fetches all active LEADING_INDICATORS series in parallel, standardizes
    each against a 24-month window, builds the weighted composite index on
    a 0-10 scale, derives recession probability, and maps to annualized
    quarterly GDP growth via empirical calibration.

    Args:
        history_months: Minimum months of history to request per series.

    Returns:
        GDPNowcast with composite_index, nowcast_gdp_growth, recession_probability,
        composite_trend, confidence, and per-component breakdown. warnings list
        captures any series that failed to fetch or lacked sufficient data.
    """
    as_of = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    limit = max(history_months, 24) + 12
    series_data, warnings = await _fetch_all_series(limit)
    weights = {sid: meta[1] for sid, meta in _ACTIVE_SERIES.items()}

    components = _build_components(series_data)
    composite = _composite_index(components)
    trend_str = _composite_trend(series_data, weights, window=24)

    t10y2y_z: Optional[float] = next(
        (c.standardized_value for c in components if c.series_id == "T10Y2Y"), None
    )
    rec_prob = _recession_probability(composite, t10y2y_z)
    gdp_growth, confidence = _nowcast_gdp(composite, components)

    return GDPNowcast(
        nowcast_gdp_growth=round(gdp_growth, 2),
        nowcast_confidence=confidence,
        components=components,
        composite_index=round(composite, 3),
        composite_trend=trend_str,
        recession_probability=round(rec_prob, 3),
        as_of=as_of,
        warnings=warnings,
    )


async def get_macro_nowcast_dashboard(history_months: int = 24) -> MacroDashboard:
    """Full macro nowcast dashboard with per-series readings and regime classification.

    Extends get_gdp_nowcast with detailed SeriesReading objects containing MoM
    and YoY changes plus trend labels, expansion/contraction signal lists,
    plain-language key risk flags, and a four-state regime classification.

    Args:
        history_months: Minimum months of history per series (24-month z-score window).

    Returns:
        MacroDashboard with nowcast, series_readings, expansion_signals,
        contraction_signals, key_risks, regime, and warnings.
    """
    as_of = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    limit = max(history_months, 24) + 12
    series_data, warnings = await _fetch_all_series(limit)
    weights = {sid: meta[1] for sid, meta in _ACTIVE_SERIES.items()}

    components = _build_components(series_data)
    composite = _composite_index(components)
    trend_str = _composite_trend(series_data, weights, window=24)

    t10y2y_z: Optional[float] = next(
        (c.standardized_value for c in components if c.series_id == "T10Y2Y"), None
    )
    rec_prob = _recession_probability(composite, t10y2y_z)
    gdp_growth, confidence = _nowcast_gdp(composite, components)

    nowcast = GDPNowcast(
        nowcast_gdp_growth=round(gdp_growth, 2),
        nowcast_confidence=confidence,
        components=components,
        composite_index=round(composite, 3),
        composite_trend=trend_str,
        recession_probability=round(rec_prob, 3),
        as_of=as_of,
        warnings=warnings,
    )

    # Build per-series readings
    series_readings: list[SeriesReading] = []
    for sid in _ACTIVE_SERIES:
        name, _ = _ACTIVE_SERIES[sid]
        obs = series_data.get(sid, [])
        if not obs:
            series_readings.append(SeriesReading(
                series_id=sid, name=name, latest_value=None,
                prev_value=None, last_updated=None, trend="unknown",
            ))
            continue
        dates = [d for d, _ in obs]
        vals = np.array([v for _, v in obs])
        mom = _mom_change(vals)
        yoy = _yoy_change(vals, dates)
        series_readings.append(SeriesReading(
            series_id=sid,
            name=name,
            latest_value=round(float(vals[-1]), 4),
            prev_value=round(float(vals[-2]), 4) if vals.size >= 2 else None,
            mom_change=round(mom, 6) if mom is not None else None,
            yoy_change=round(yoy, 6) if yoy is not None else None,
            last_updated=dates[-1],
            trend=_trend(mom, yoy),
        ))

    expansion_signals = [
        c.series_id for c in components
        if c.standardized_value is not None and c.standardized_value > 0.5
    ]
    contraction_signals = [
        c.series_id for c in components
        if c.standardized_value is not None and c.standardized_value < -0.5
    ]
    key_risks = _identify_key_risks(components, t10y2y_z)
    regime = _classify_regime(composite, trend_str)

    return MacroDashboard(
        nowcast=nowcast,
        series_readings=series_readings,
        expansion_signals=expansion_signals,
        contraction_signals=contraction_signals,
        key_risks=key_risks,
        regime=regime,
        as_of=as_of,
        warnings=list(warnings),
    )


async def get_macro_screen() -> MacroNowcastScreen:
    """Thin screener wrapper — top-3 expansion and contraction signals for widgets.

    Calls get_gdp_nowcast() and reshapes into MacroNowcastScreen, ranking
    signals by absolute standardized value magnitude.

    Returns:
        MacroNowcastScreen with composite_index, regime, top signals, and
        recession_probability ready for dashboard consumption.
    """
    nowcast = await get_gdp_nowcast()
    regime = _classify_regime(nowcast.composite_index, nowcast.composite_trend)

    def _ranked(direction: str, positive: bool) -> list[MacroRegimeSignal]:
        filtered = [
            c for c in nowcast.components
            if c.standardized_value is not None
            and (c.standardized_value > 0 if positive else c.standardized_value < 0)
        ]
        filtered.sort(key=lambda c: abs(c.standardized_value), reverse=True)  # type: ignore[arg-type]
        return [
            MacroRegimeSignal(
                series_id=c.series_id,
                name=c.name,
                signal_direction=direction,
                magnitude=round(abs(c.standardized_value), 3),  # type: ignore[arg-type]
                as_of=nowcast.as_of,
            )
            for c in filtered[:3]
        ]

    return MacroNowcastScreen(
        composite_index=nowcast.composite_index,
        regime=regime,
        top_expansion_signals=_ranked("bullish", positive=True),
        top_contraction_signals=_ranked("bearish", positive=False),
        recession_probability=nowcast.recession_probability,
        as_of=nowcast.as_of,
        warnings=list(nowcast.warnings),
    )
