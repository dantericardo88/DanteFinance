"""Persistent price alerting system — stores alerts in ~/.sentinel/alerts.json, polls prices via yfinance."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

_ALERTS_PATH = Path.home() / ".sentinel" / "alerts.json"


class AlertType(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PCT_CHANGE_UP = "pct_change_up"
    PCT_CHANGE_DOWN = "pct_change_down"
    VOLUME_SPIKE = "volume_spike"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    MA_CROSSOVER = "ma_crossover"
    MA_CROSSUNDER = "ma_crossunder"


class PriceAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    alert_type: AlertType
    threshold: Optional[float] = None
    baseline_price: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    triggered_at: Optional[str] = None
    triggered_price: Optional[float] = None
    active: bool = True
    note: str = ""
    params: dict = Field(default_factory=dict)


class AlertCheckResult(BaseModel):
    ticker: str
    current_price: float
    triggered_alerts: list[PriceAlert]
    active_alerts: list[PriceAlert]
    warnings: list[str] = Field(default_factory=list)


class AlertSummary(BaseModel):
    total_active: int
    total_triggered_today: int
    by_ticker: dict[str, int]
    triggered: list[AlertCheckResult]
    as_of: str


# ── Sync persistence helpers (called via asyncio.to_thread) ──────────────────

def _load_alerts() -> list[PriceAlert]:
    if not _ALERTS_PATH.exists():
        return []
    try:
        raw = json.loads(_ALERTS_PATH.read_text(encoding="utf-8"))
        return [PriceAlert.model_validate(a) for a in raw]
    except Exception as exc:
        logger.warning("alerts_load_failed", path=str(_ALERTS_PATH), error=str(exc))
        return []


def _save_alerts(alerts: list[PriceAlert]) -> None:
    _ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [a.model_dump(mode="json") for a in alerts]
    _ALERTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Indicator helpers ────────────────────────────────────────────────────────

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _compute_ma(closes: np.ndarray, window: int) -> float:
    if len(closes) < window:
        return float(np.mean(closes))
    return float(np.mean(closes[-window:]))


# ── Condition evaluation ─────────────────────────────────────────────────────

def _evaluate_alert(
    alert: PriceAlert,
    closes: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    warnings: list[str],
) -> bool:
    t = alert.alert_type

    if t == AlertType.PRICE_ABOVE:
        return alert.threshold is not None and current_price > alert.threshold

    if t == AlertType.PRICE_BELOW:
        return alert.threshold is not None and current_price < alert.threshold

    if t == AlertType.PCT_CHANGE_UP:
        if alert.baseline_price is None or alert.threshold is None:
            return False
        pct = (current_price - alert.baseline_price) / alert.baseline_price * 100
        return pct >= alert.threshold

    if t == AlertType.PCT_CHANGE_DOWN:
        if alert.baseline_price is None or alert.threshold is None:
            return False
        pct = (alert.baseline_price - current_price) / alert.baseline_price * 100
        return pct >= alert.threshold

    if t == AlertType.VOLUME_SPIKE:
        if len(volumes) < 21:
            warnings.append(f"{alert.ticker}: insufficient volume history for spike check")
            return False
        multiplier = float(alert.params.get("multiplier", 2.5))
        today_vol = float(volumes[-1])
        avg_vol = float(np.mean(volumes[-21:-1]))
        return avg_vol > 0 and today_vol > multiplier * avg_vol

    if t == AlertType.RSI_OVERBOUGHT:
        if len(closes) < 16:
            warnings.append(f"{alert.ticker}: insufficient history for RSI")
            return False
        threshold = float(alert.params.get("rsi_threshold", 70))
        rsi = _compute_rsi(closes)
        return rsi > threshold

    if t == AlertType.RSI_OVERSOLD:
        if len(closes) < 16:
            warnings.append(f"{alert.ticker}: insufficient history for RSI")
            return False
        threshold = float(alert.params.get("rsi_threshold", 30))
        rsi = _compute_rsi(closes)
        return rsi < threshold

    if t in (AlertType.MA_CROSSOVER, AlertType.MA_CROSSUNDER):
        fast_w = int(alert.params.get("fast", 10))
        slow_w = int(alert.params.get("slow", 30))
        if len(closes) < slow_w + 1:
            warnings.append(f"{alert.ticker}: insufficient history for MA({slow_w})")
            return False
        # Compare yesterday's relationship vs today's to detect a cross
        fast_now = _compute_ma(closes, fast_w)
        slow_now = _compute_ma(closes, slow_w)
        fast_prev = _compute_ma(closes[:-1], fast_w)
        slow_prev = _compute_ma(closes[:-1], slow_w)
        if t == AlertType.MA_CROSSOVER:
            return fast_prev <= slow_prev and fast_now > slow_now
        return fast_prev >= slow_prev and fast_now < slow_now

    return False


# ── Sync yfinance fetch (called via asyncio.to_thread) ──────────────────────

def _fetch_ticker_data(ticker: str) -> dict | None:
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="60d", interval="1d")
        if hist.empty:
            return None
        closes = hist["Close"].to_numpy(dtype=float)
        volumes = hist["Volume"].to_numpy(dtype=float)
        return {"closes": closes, "volumes": volumes}
    except Exception as exc:
        logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
        return None


# ── Sync baseline price fetch ────────────────────────────────────────────────

def _fetch_current_price(ticker: str) -> float | None:
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="2d", interval="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


# ── Public async API ─────────────────────────────────────────────────────────

async def create_alert(
    ticker: str,
    alert_type: AlertType | str,
    threshold: float | None = None,
    note: str = "",
    params: dict | None = None,
) -> PriceAlert:
    """Create and persist a new alert. Returns the created alert."""
    if isinstance(alert_type, str):
        alert_type = AlertType(alert_type)

    baseline_price = await asyncio.to_thread(_fetch_current_price, ticker.upper())

    alert = PriceAlert(
        ticker=ticker.upper(),
        alert_type=alert_type,
        threshold=threshold,
        baseline_price=baseline_price,
        note=note,
        params=params or {},
    )

    alerts = await asyncio.to_thread(_load_alerts)
    alerts.append(alert)
    await asyncio.to_thread(_save_alerts, alerts)

    logger.info(
        "alert_created",
        id=alert.id,
        ticker=alert.ticker,
        type=alert.alert_type,
        threshold=threshold,
        baseline=baseline_price,
    )
    return alert


async def check_alerts(tickers: list[str] | None = None) -> AlertSummary:
    """Check current prices against all active alerts. Returns triggered alerts."""
    all_alerts = await asyncio.to_thread(_load_alerts)
    active = [a for a in all_alerts if a.active]

    if tickers:
        upper = {t.upper() for t in tickers}
        active = [a for a in active if a.ticker in upper]

    if not active:
        return AlertSummary(
            total_active=0,
            total_triggered_today=0,
            by_ticker={},
            triggered=[],
            as_of=datetime.utcnow().isoformat(),
        )

    grouped: dict[str, list[PriceAlert]] = {}
    for a in active:
        grouped.setdefault(a.ticker, []).append(a)

    # Fetch all tickers concurrently
    fetch_tasks = {
        ticker: asyncio.to_thread(_fetch_ticker_data, ticker)
        for ticker in grouped
    }
    fetch_results: dict[str, dict | None] = {}
    for ticker, coro in fetch_tasks.items():
        fetch_results[ticker] = await coro

    triggered_results: list[AlertCheckResult] = []
    today_prefix = datetime.utcnow().date().isoformat()
    newly_triggered: list[PriceAlert] = []

    by_ticker: dict[str, int] = {}

    for ticker, ticker_alerts in grouped.items():
        data = fetch_results.get(ticker)
        warnings: list[str] = []

        if data is None:
            warnings.append(f"Could not fetch data for {ticker}")
            by_ticker[ticker] = len(ticker_alerts)
            continue

        closes = data["closes"]
        volumes = data["volumes"]
        current_price = float(closes[-1])

        triggered: list[PriceAlert] = []
        still_active: list[PriceAlert] = []

        for alert in ticker_alerts:
            fired = _evaluate_alert(alert, closes, volumes, current_price, warnings)
            if fired:
                alert.triggered_at = datetime.utcnow().isoformat()
                alert.triggered_price = current_price
                alert.active = False
                triggered.append(alert)
                newly_triggered.append(alert)
                logger.info(
                    "alert_triggered",
                    id=alert.id,
                    ticker=ticker,
                    type=alert.alert_type,
                    price=current_price,
                )
            else:
                still_active.append(alert)

        by_ticker[ticker] = len(still_active)

        if triggered:
            triggered_results.append(
                AlertCheckResult(
                    ticker=ticker,
                    current_price=current_price,
                    triggered_alerts=triggered,
                    active_alerts=still_active,
                    warnings=warnings,
                )
            )

    if newly_triggered:
        # Merge triggered state back into the full alert list
        triggered_ids = {a.id for a in newly_triggered}
        updated_map = {a.id: a for a in newly_triggered}
        merged = [updated_map.get(a.id, a) for a in all_alerts]
        await asyncio.to_thread(_save_alerts, merged)

    total_triggered_today = sum(
        1 for r in triggered_results
        for a in r.triggered_alerts
        if a.triggered_at and a.triggered_at.startswith(today_prefix)
    )

    return AlertSummary(
        total_active=sum(by_ticker.values()),
        total_triggered_today=total_triggered_today,
        by_ticker=by_ticker,
        triggered=triggered_results,
        as_of=datetime.utcnow().isoformat(),
    )


async def delete_alert(alert_id: str) -> bool:
    """Delete alert by ID. Returns True if found and deleted."""
    alerts = await asyncio.to_thread(_load_alerts)
    original_len = len(alerts)
    alerts = [a for a in alerts if a.id != alert_id]
    if len(alerts) == original_len:
        return False
    await asyncio.to_thread(_save_alerts, alerts)
    logger.info("alert_deleted", id=alert_id)
    return True


async def list_alerts(ticker: str | None = None, active_only: bool = True) -> list[PriceAlert]:
    """List all alerts, optionally filtered by ticker."""
    alerts = await asyncio.to_thread(_load_alerts)
    if active_only:
        alerts = [a for a in alerts if a.active]
    if ticker:
        alerts = [a for a in alerts if a.ticker == ticker.upper()]
    return alerts
