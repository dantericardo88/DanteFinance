"""Economic time series forecasting — AR(p), VAR(p), ARIMA(p,d,q), nowcast index.

FRED CSV endpoint (no API key). Pure numpy/scipy — no statsmodels. Dim 60, target 5/10.
"""
from __future__ import annotations

import asyncio
import csv
import io
from datetime import date
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

MACRO_SERIES: dict[str, str] = {
    "GDPC1": "Real GDP (quarterly, SAAR)",
    "UNRATE": "Unemployment Rate (%)",
    "CPIAUCSL": "CPI All Urban",
    "FEDFUNDS": "Federal Funds Rate (%)",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "INDPRO": "Industrial Production Index",
    "HOUST": "Housing Starts (thousands)",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "PAYEMS": "Total Nonfarm Payrolls (thousands)",
    "M2SL": "M2 Money Supply (billions)",
}

NOWCAST_WEIGHTS: dict[str, float] = {
    "UNRATE": 0.25,   # inverted
    "INDPRO": 0.25,
    "UMCSENT": 0.20,
    "PAYEMS": 0.20,   # month-over-month change
    "HOUST": 0.10,
}

DEFAULT_SERIES = ["UNRATE", "CPIAUCSL", "FEDFUNDS", "T10Y2Y", "INDPRO"]
MIN_OBS_AR = 20
MIN_OBS_VAR = 40


class ForecastPoint(BaseModel):
    period: str
    value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class SeriesForecast(BaseModel):
    series_id: str
    series_name: str
    frequency: str
    last_actual: float
    last_actual_date: str
    forecast_horizon: int
    model_used: str
    model_order: str
    in_sample_rmse: Optional[float] = None
    forecasts: list[ForecastPoint]


class NowcastIndex(BaseModel):
    score: float
    signal: str
    components: dict[str, float]
    as_of: str


class EconForecastResult(BaseModel):
    forecasts: list[SeriesForecast]
    nowcast: NowcastIndex
    var_forecasts: Optional[list[SeriesForecast]] = None
    warnings: list[str] = Field(default_factory=list)
    as_of: str


async def _fetch_fred_series(
    client: httpx.AsyncClient,
    series_id: str,
    limit_obs: int = 120,
) -> list[tuple[str, float]]:
    """Return last limit_obs (date, value) pairs, skipping missing dots."""
    url = f"{FRED_CSV_BASE}{series_id}"
    try:
        resp = await client.get(url, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("FRED CSV fetch failed", series_id=series_id, error=str(exc))
        raise

    reader = csv.reader(io.StringIO(resp.text))
    next(reader, None)  # skip header
    rows: list[tuple[str, float]] = []
    for row in reader:
        if len(row) < 2 or row[1].strip() == ".":
            continue
        try:
            rows.append((row[0].strip(), float(row[1].strip())))
        except ValueError:
            continue

    return rows[-limit_obs:] if len(rows) > limit_obs else rows


def _next_period_label(last_date: str, step: int, frequency: str) -> str:
    """Generate a period label step periods after last_date."""
    try:
        year, month, *_ = last_date.split("-")
        year_i, month_i = int(year), int(month)
    except (ValueError, IndexError):
        return f"T+{step}"

    if frequency == "quarterly":
        total_q = (year_i - 1) * 4 + (month_i - 1) // 3 + step
        q_year = (total_q - 1) // 4 + 1
        q_num = (total_q - 1) % 4 + 1
        return f"{q_year}-Q{q_num}"

    total_m = (year_i - 1) * 12 + month_i + step
    m_year = (total_m - 1) // 12 + 1
    m_month = (total_m - 1) % 12 + 1
    return f"{m_year}-{m_month:02d}"


def _detect_frequency(dates: list[str]) -> str:
    if len(dates) < 2:
        return "monthly"
    try:
        d0 = date.fromisoformat(dates[0])
        d1 = date.fromisoformat(dates[1])
        gap = abs((d1 - d0).days)
        return "quarterly" if gap > 60 else "monthly"
    except ValueError:
        return "monthly"


def _ar_fit(y: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """OLS AR(p)+constant. Coefficients: [phi_1, ..., phi_p, const]."""
    n = len(y)
    if n - p < p + 1:
        raise ValueError(f"Too few observations ({n}) for AR({p})")

    X = np.column_stack([y[p - i - 1: n - i - 1] for i in range(p)] + [np.ones(n - p)])
    y_target = y[p:]
    coeffs, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)

    residuals = y_target - X @ coeffs
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return coeffs, rmse


def _ar_forecast(y: np.ndarray, coeffs: np.ndarray, p: int, h: int) -> np.ndarray:
    """Recursive h-step AR forecast."""
    history = list(y[-p:])
    phi = coeffs[:p]
    const = coeffs[p]
    preds = []
    for _ in range(h):
        val = const + float(np.dot(phi, history[-p:][::-1]))
        preds.append(val)
        history.append(val)
    return np.array(preds)


def _arima_fit(y: np.ndarray, p: int, d: int, q: int) -> tuple[np.ndarray, float, str]:
    """Fit ARIMA(p,d,q). Returns (differenced_series, rmse, model_label)."""
    from scipy.optimize import minimize  # lazy import

    yd = y.copy()
    for _ in range(d):
        yd = np.diff(yd)

    n = len(yd)
    if n < max(p, q) + 5:
        raise ValueError("Insufficient obs after differencing")

    def neg_loglik(params: np.ndarray) -> float:
        ar = params[:p]
        ma = params[p: p + q]
        sigma2 = max(params[p + q] ** 2, 1e-8)
        residuals = np.zeros(n)
        for t in range(n):
            ar_part = sum(ar[i] * yd[t - i - 1] if t - i - 1 >= 0 else 0.0 for i in range(p))
            ma_part = sum(ma[j] * residuals[t - j - 1] if t - j - 1 >= 0 else 0.0 for j in range(q))
            residuals[t] = yd[t] - ar_part - ma_part
        return 0.5 * (n * np.log(2 * np.pi * sigma2) + np.sum(residuals ** 2) / sigma2)

    x0 = np.zeros(p + q + 1)
    x0[-1] = np.std(yd) if np.std(yd) > 0 else 1.0

    result = minimize(neg_loglik, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5})

    if not result.success:
        # Fall back: pure AR(p) on differenced series
        try:
            coeffs, rmse = _ar_fit(yd, max(p, 1))
            return yd, rmse, f"AR({max(p,1)}) [fallback]"
        except Exception:
            raise ValueError("ARIMA fit and AR fallback both failed")

    params = result.x
    ar = params[:p]
    ma = params[p: p + q]
    residuals = np.zeros(n)
    for t in range(n):
        ar_part = sum(ar[i] * yd[t - i - 1] if t - i - 1 >= 0 else 0.0 for i in range(p))
        ma_part = sum(ma[j] * residuals[t - j - 1] if t - j - 1 >= 0 else 0.0 for j in range(q))
        residuals[t] = yd[t] - ar_part - ma_part

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return yd, rmse, f"ARIMA({p},{d},{q})"


def _arima_forecast(y_orig: np.ndarray, p: int, d: int, h: int) -> np.ndarray:
    """Forecast h steps from ARIMA fit. Integrates back d times from last actual."""
    yd = y_orig.copy()
    diffs: list[np.ndarray] = []
    for _ in range(d):
        diffs.append(yd.copy())
        yd = np.diff(yd)

    try:
        coeffs, _ = _ar_fit(yd, max(p, 1))
        preds_d = _ar_forecast(yd, coeffs, max(p, 1), h)
    except Exception:
        preds_d = np.full(h, yd[-1] if len(yd) > 0 else 0.0)

    # Integrate predictions back d times
    result = preds_d.copy()
    for diff_arr in reversed(diffs):
        last_val = diff_arr[-1]
        result = np.cumsum(np.insert(result, 0, last_val))[1:]

    return result


def _var_fit(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """VAR(p) OLS. Y=(T,k). B shape=(k*p+1, k). Regressor: [y_{t-1},..,y_{t-p}, 1]."""
    T, k = Y.shape
    rows = T - p

    X = np.ones((rows, k * p + 1))
    for lag in range(1, p + 1):
        col_start = (lag - 1) * k
        X[:, col_start: col_start + k] = Y[p - lag: T - lag]

    y_target = Y[p:]  # (rows, k)
    B, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    residuals = y_target - X @ B
    return B, residuals


def _var_forecast(Y: np.ndarray, B: np.ndarray, p: int, h: int) -> np.ndarray:
    """Recursive VAR forecast. Returns (h, k) array."""
    T, k = Y.shape
    history = list(Y[-p:])

    forecasts = []
    for _ in range(h):
        x = np.ones(k * p + 1)
        for lag in range(1, p + 1):
            col_start = (lag - 1) * k
            x[col_start: col_start + k] = history[-lag]
        y_hat = x @ B
        forecasts.append(y_hat)
        history.append(y_hat)

    return np.array(forecasts)


def _compute_nowcast(series_data: dict[str, list[tuple[str, float]]]) -> NowcastIndex:
    """Weighted Z-score composite. UNRATE inverted. PAYEMS uses MoM change."""
    components: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for sid, weight in NOWCAST_WEIGHTS.items():
        data = series_data.get(sid)
        if not data or len(data) < 12:
            logger.warning("Insufficient data for nowcast component", series_id=sid)
            continue

        vals = np.array([v for _, v in data])

        if sid == "PAYEMS":
            vals = np.diff(vals)
            if len(vals) == 0:
                continue

        mean = float(np.mean(vals[:-1]))
        std = float(np.std(vals[:-1]))
        if std < 1e-10:
            continue

        latest = float(vals[-1])
        z = (latest - mean) / std

        if sid == "UNRATE":
            z = -z

        components[sid] = round(z, 4)
        weighted_sum += z * weight
        weight_total += weight

    raw_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    scaled = float(np.clip(raw_score / 3.0, -1.0, 1.0) * 10.0)
    signal = "expansion" if scaled > 1.5 else "contraction" if scaled < -1.5 else "neutral"
    return NowcastIndex(score=round(scaled, 2), signal=signal, components=components, as_of=date.today().isoformat())


def _build_series_forecast(
    series_id: str, data: list[tuple[str, float]], horizon: int, ar_order: int, warnings: list[str]
) -> Optional[SeriesForecast]:
    if len(data) < MIN_OBS_AR:
        warnings.append(f"{series_id}: only {len(data)} obs, need {MIN_OBS_AR} — skipped")
        return None

    dates = [d for d, _ in data]
    vals = np.array([v for _, v in data])
    frequency = _detect_frequency(dates)
    last_date, last_val = dates[-1], float(vals[-1])

    try:
        _, rmse, model_label = _arima_fit(vals, ar_order, 1, 1)
        preds = _arima_forecast(vals, ar_order, 1, horizon)
        model_used, model_order = "ARIMA(p,d,q)", model_label
    except Exception as exc:
        logger.debug("ARIMA failed, using AR", series_id=series_id, error=str(exc))
        try:
            p = max(min(ar_order, len(vals) // 4), 1)
            coeffs, rmse = _ar_fit(vals, p)
            preds = _ar_forecast(vals, coeffs, p, horizon)
            model_used, model_order = "AR(p)", f"AR({p})"
        except Exception as exc2:
            warnings.append(f"{series_id}: AR fit failed — {exc2}")
            return None

    try:
        p_check = max(min(ar_order, len(vals) // 4, len(vals) - 1), 1)
        _, rmse_ar = _ar_fit(vals, p_check)
    except Exception:
        rmse_ar = float(np.std(np.diff(vals))) if len(vals) > 1 else 1.0

    forecast_points = [
        ForecastPoint(
            period=_next_period_label(last_date, i, frequency),
            value=round(float(preds[i - 1]), 4),
            confidence_lower=round(float(preds[i - 1]) - 1.96 * rmse_ar * float(np.sqrt(i)), 4),
            confidence_upper=round(float(preds[i - 1]) + 1.96 * rmse_ar * float(np.sqrt(i)), 4),
        )
        for i in range(1, horizon + 1)
    ]
    return SeriesForecast(
        series_id=series_id,
        series_name=MACRO_SERIES.get(series_id, series_id),
        frequency=frequency,
        last_actual=round(last_val, 4),
        last_actual_date=last_date,
        forecast_horizon=horizon,
        model_used=model_used,
        model_order=model_order,
        in_sample_rmse=round(rmse, 6),
        forecasts=forecast_points,
    )


def _build_var_forecasts(
    series_data: dict[str, list[tuple[str, float]]],
    series_ids: list[str],
    horizon: int,
    p: int,
    warnings: list[str],
) -> Optional[list[SeriesForecast]]:
    aligned: dict[str, np.ndarray] = {}
    for sid in series_ids:
        data = series_data.get(sid, [])
        if len(data) < MIN_OBS_VAR:
            warnings.append(f"VAR skipped {sid}: only {len(data)} obs (need {MIN_OBS_VAR})")
            return None
        aligned[sid] = np.array([v for _, v in data[-MIN_OBS_VAR:]])

    if len(aligned) < 2:
        warnings.append("VAR requires at least 2 series with sufficient data")
        return None

    active_ids = list(aligned.keys())
    min_len = min(len(v) for v in aligned.values())
    Y = np.column_stack([aligned[sid][-min_len:] for sid in active_ids])

    try:
        B, residuals = _var_fit(Y, p)
        var_preds = _var_forecast(Y, B, p, horizon)
    except Exception as exc:
        warnings.append(f"VAR fit failed: {exc}")
        return None

    rmse_per_series = [float(np.sqrt(np.mean(residuals[:, i] ** 2))) for i in range(len(active_ids))]
    results: list[SeriesForecast] = []
    for col_i, sid in enumerate(active_ids):
        data = series_data[sid]
        dates = [d for d, _ in data]
        last_date, frequency = dates[-1], _detect_frequency(dates)
        rmse = rmse_per_series[col_i]
        forecast_points = [
            ForecastPoint(
                period=_next_period_label(last_date, s, frequency),
                value=round(float(var_preds[s - 1, col_i]), 4),
                confidence_lower=round(float(var_preds[s - 1, col_i]) - 1.96 * rmse * float(np.sqrt(s)), 4),
                confidence_upper=round(float(var_preds[s - 1, col_i]) + 1.96 * rmse * float(np.sqrt(s)), 4),
            )
            for s in range(1, horizon + 1)
        ]
        results.append(SeriesForecast(
            series_id=sid,
            series_name=MACRO_SERIES.get(sid, sid),
            frequency=frequency,
            last_actual=round(float(Y[-1, col_i]), 4),
            last_actual_date=last_date,
            forecast_horizon=horizon,
            model_used="VAR(p)",
            model_order=f"VAR({p})",
            in_sample_rmse=round(rmse, 6),
            forecasts=forecast_points,
        ))
    return results


async def get_econ_forecast(
    series_ids: list[str] | None = None,
    horizon: int = 6,
    ar_order: int = 3,
    include_var: bool = True,
) -> EconForecastResult:
    """Fetch FRED macro series and forecast horizon periods ahead.

    Uses AR(p)/ARIMA(p,d,q) per series, optionally a joint VAR(p), and a
    nowcast composite index from current-period indicators.
    """
    if series_ids is None:
        series_ids = DEFAULT_SERIES

    nowcast_ids = list(NOWCAST_WEIGHTS.keys())
    all_ids = list(dict.fromkeys(series_ids + nowcast_ids))

    warnings: list[str] = []
    series_data: dict[str, list[tuple[str, float]]] = {}

    async with httpx.AsyncClient() as client:
        fetch_results = await asyncio.gather(
            *[_fetch_fred_series(client, sid) for sid in all_ids], return_exceptions=True
        )

    for sid, result in zip(all_ids, fetch_results):
        if isinstance(result, Exception):
            warnings.append(f"{sid}: fetch failed — {result}")
            logger.warning("Series fetch failed", series_id=sid)
        else:
            series_data[sid] = result
            logger.info("Fetched series", series_id=sid, obs=len(result))

    forecasts: list[SeriesForecast] = []
    for sid in series_ids:
        data = series_data.get(sid)
        if not data:
            warnings.append(f"{sid}: no data fetched")
            continue
        sf = _build_series_forecast(sid, data, horizon, ar_order, warnings)
        if sf is not None:
            forecasts.append(sf)

    var_forecasts: Optional[list[SeriesForecast]] = None
    if include_var and len(series_ids) >= 2:
        var_forecasts = _build_var_forecasts(series_data, series_ids, horizon, ar_order, warnings)

    nowcast = _compute_nowcast(series_data)
    logger.info("Econ forecast complete", n_series=len(forecasts), nowcast_score=nowcast.score, warnings=len(warnings))
    return EconForecastResult(
        forecasts=forecasts, nowcast=nowcast, var_forecasts=var_forecasts,
        warnings=warnings, as_of=date.today().isoformat(),
    )
