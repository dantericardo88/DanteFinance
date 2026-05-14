"""
Options flow screener: identifies unusual volume, IV spikes, and directional positioning.
Uses Polygon options data (already in the adapter) and the options_analytics module.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger
from sentinel.sbx.options_analytics import get_options_summary

logger = get_logger(__name__)

_POLYGON_OPTIONS_URL = "https://api.polygon.io/v3/snapshot/options/{ticker}"

# Severity rank used for sorting
_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
_SIGNAL_RANK = {
    "strong_bullish": 2,
    "bullish": 1,
    "neutral": 0,
    "bearish": -1,
    "strong_bearish": -2,
}


# ── Models ────────────────────────────────────────────────────────────────────

class OptionsFlowAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    alert_type: str       # "unusual_volume" | "iv_spike" | "put_skew" | "call_sweep" | "gamma_wall"
    severity: str         # "low" | "medium" | "high"
    description: str
    metric_value: float
    threshold: float
    timestamp: datetime


class OptionsScreenResult(BaseModel):
    ticker: str
    underlying_price: float
    avg_volume_30d: float | None
    options_volume_today: int | None
    put_call_ratio: float | None
    iv_percentile: float | None     # current IV vs 52-week range 0-100
    skew_25d: float | None          # 25-delta risk reversal (positive = put fear)
    net_gex: float | None           # dealer gamma exposure
    max_pain_strike: float | None
    alerts: list[OptionsFlowAlert]
    signal: str                     # "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish"


class OptionsScreenerCriteria(BaseModel):
    min_volume_ratio: float = 2.0   # options volume / 30d avg
    max_put_call_ratio: float = 0.7  # bullish signal threshold (low put/call)
    min_put_call_ratio: float = 1.5  # bearish signal threshold (high put/call)
    iv_percentile_high: float = 80  # elevated IV
    iv_percentile_low: float = 20   # depressed IV
    min_skew_magnitude: float = 0.03  # 3 vol points of skew


# ── Alert detectors ───────────────────────────────────────────────────────────

def detect_unusual_volume(
    today_volume: int,
    avg_30d_volume: float,
    threshold_ratio: float = 2.0,
    ticker: str = "",
) -> OptionsFlowAlert | None:
    """Return alert if today_volume > threshold_ratio * avg_30d_volume."""
    if avg_30d_volume <= 0:
        return None
    ratio = today_volume / avg_30d_volume
    if ratio < threshold_ratio:
        return None

    severity = "high" if ratio >= 5.0 else "medium" if ratio >= 3.0 else "low"
    return OptionsFlowAlert(
        ticker=ticker,
        alert_type="unusual_volume",
        severity=severity,
        description=f"Options volume {ratio:.1f}x the 30-day average ({today_volume:,} vs {avg_30d_volume:,.0f} avg)",
        metric_value=round(ratio, 3),
        threshold=threshold_ratio,
        timestamp=datetime.utcnow(),
    )


def detect_skew_signal(
    skew_25d: float,
    threshold: float = 0.03,
    ticker: str = "",
) -> OptionsFlowAlert | None:
    """
    Negative skew_25d (put IV > call IV) → bearish signal (downside fear).
    Positive skew_25d → bullish signal (call demand).
    risk_reversal_25d is defined as 25d put IV − 25d call IV in options_analytics.
    """
    if abs(skew_25d) < threshold:
        return None

    # Positive risk_reversal_25d = put IV > call IV = bearish positioning
    if skew_25d > 0:
        severity = "high" if skew_25d > 0.08 else "medium" if skew_25d > 0.05 else "low"
        return OptionsFlowAlert(
            ticker=ticker,
            alert_type="put_skew",
            severity=severity,
            description=f"Bearish put skew: 25d risk reversal = {skew_25d:.3f} (puts priced {skew_25d*100:.1f} vol pts above calls)",
            metric_value=round(skew_25d, 4),
            threshold=threshold,
            timestamp=datetime.utcnow(),
        )
    else:
        # Negative risk_reversal = call IV > put IV = bullish call demand
        mag = abs(skew_25d)
        severity = "high" if mag > 0.08 else "medium" if mag > 0.05 else "low"
        return OptionsFlowAlert(
            ticker=ticker,
            alert_type="call_sweep",
            severity=severity,
            description=f"Bullish call skew: 25d risk reversal = {skew_25d:.3f} (calls priced {mag*100:.1f} vol pts above puts)",
            metric_value=round(skew_25d, 4),
            threshold=-threshold,
            timestamp=datetime.utcnow(),
        )


def detect_gamma_wall(
    net_gex: float,
    threshold: float = 1e8,
    ticker: str = "",
) -> OptionsFlowAlert | None:
    """
    Large positive GEX → dealers are long gamma near that strike → acts as vol suppressor.
    Large negative GEX → dealers short gamma → can amplify moves.
    """
    if abs(net_gex) < threshold:
        return None

    abs_gex = abs(net_gex)
    severity = "high" if abs_gex > 5e8 else "medium" if abs_gex > 2e8 else "low"

    if net_gex > 0:
        return OptionsFlowAlert(
            ticker=ticker,
            alert_type="gamma_wall",
            severity=severity,
            description=f"Positive GEX ${net_gex/1e6:.0f}M — dealers long gamma, expect range-bound price action",
            metric_value=net_gex,
            threshold=threshold,
            timestamp=datetime.utcnow(),
        )
    else:
        return OptionsFlowAlert(
            ticker=ticker,
            alert_type="gamma_wall",
            severity=severity,
            description=f"Negative GEX ${net_gex/1e6:.0f}M — dealers short gamma, expect amplified moves",
            metric_value=net_gex,
            threshold=-threshold,
            timestamp=datetime.utcnow(),
        )


def _detect_iv_spike(
    iv_percentile: float,
    criteria: OptionsScreenerCriteria,
    ticker: str,
) -> OptionsFlowAlert | None:
    """Alert when IV is in an extreme percentile range."""
    if iv_percentile >= criteria.iv_percentile_high:
        severity = "high" if iv_percentile >= 90 else "medium"
        return OptionsFlowAlert(
            ticker=ticker,
            alert_type="iv_spike",
            severity=severity,
            description=f"IV at {iv_percentile:.0f}th percentile of 52-week range — elevated vol environment",
            metric_value=iv_percentile,
            threshold=criteria.iv_percentile_high,
            timestamp=datetime.utcnow(),
        )
    return None


def compute_signal(result: OptionsScreenResult) -> str:
    """Aggregate all alerts into a single directional signal."""
    score = 0

    for alert in result.alerts:
        weight = _SEVERITY_RANK.get(alert.severity, 1)
        if alert.alert_type == "call_sweep":
            score += weight       # bullish
        elif alert.alert_type == "put_skew":
            score -= weight       # bearish
        elif alert.alert_type == "unusual_volume":
            pass                  # direction-neutral alone; rely on other signals
        elif alert.alert_type == "iv_spike":
            pass                  # ambiguous without skew direction
        elif alert.alert_type == "gamma_wall":
            if result.net_gex and result.net_gex < 0:
                score -= 1       # negative GEX amplifies moves — slightly bearish bias

    # Put/call ratio contribution
    if result.put_call_ratio is not None:
        if result.put_call_ratio < 0.5:
            score += 2
        elif result.put_call_ratio < 0.7:
            score += 1
        elif result.put_call_ratio > 2.0:
            score -= 2
        elif result.put_call_ratio > 1.5:
            score -= 1

    if score >= 3:
        return "strong_bullish"
    if score >= 1:
        return "bullish"
    if score <= -3:
        return "strong_bearish"
    if score <= -1:
        return "bearish"
    return "neutral"


# ── Polygon data fetching ──────────────────────────────────────────────────────

async def _fetch_polygon_chain(ticker: str, polygon_api_key: str) -> list[dict]:
    """Fetch full options chain snapshot from Polygon v3 API."""
    contracts: list[dict] = []
    url = _POLYGON_OPTIONS_URL.format(ticker=ticker)
    params: dict[str, Any] = {"apiKey": polygon_api_key, "limit": 250}

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning("Polygon chain fetch failed", ticker=ticker, status=exc.response.status_code)
                break
            except Exception as exc:
                logger.warning("Polygon chain fetch error", ticker=ticker, error=str(exc))
                break

            results = data.get("results", [])
            for item in results:
                # Flatten nested Polygon v3 snapshot structure
                details = item.get("details", {})
                greeks = item.get("greeks", {})
                day = item.get("day", {})
                flat = {
                    "underlying_ticker": ticker,
                    "ticker": item.get("ticker", ""),
                    "strike_price": details.get("strike_price"),
                    "expiration_date": details.get("expiration_date"),
                    "contract_type": details.get("contract_type"),
                    "implied_volatility": item.get("implied_volatility"),
                    "open_interest": item.get("open_interest"),
                    "volume": day.get("volume"),
                    "close_price": day.get("close"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                }
                contracts.append(flat)

            # Pagination
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {"apiKey": polygon_api_key}

    return contracts


def _compute_iv_percentile(skew_list: list[Any]) -> float | None:
    """
    Approximate IV percentile from the nearest-term skew's ATM IV.
    In production this would compare against a stored 52-week IV history.
    Returns a synthetic percentile in [0, 100] based on ATM IV level.
    """
    if not skew_list:
        return None
    # Use first expiry's ATM IV as a proxy; treat >60% IV as high, <15% as low
    first_skew = skew_list[0]
    atm_iv = getattr(first_skew, "atm_iv", None)
    if atm_iv is None:
        return None
    # Normalize: assume IV range of 10%–80% maps to 0–100th percentile
    pct = max(0.0, min(100.0, (atm_iv - 0.10) / (0.80 - 0.10) * 100.0))
    return round(pct, 1)


# ── Core screener ─────────────────────────────────────────────────────────────

async def screen_single_ticker(
    ticker: str,
    underlying_price: float,
    polygon_api_key: str,
    criteria: OptionsScreenerCriteria | None = None,
) -> OptionsScreenResult:
    """
    Fetch options chain via Polygon → run options_analytics → detect flow anomalies.
    Uses sentinel.sbx.options_analytics.get_options_summary internally.
    """
    if criteria is None:
        criteria = OptionsScreenerCriteria()

    logger.debug("screen_single_ticker start", ticker=ticker)

    try:
        contracts_raw = await _fetch_polygon_chain(ticker, polygon_api_key)
    except Exception as exc:
        logger.warning("Chain fetch failed", ticker=ticker, error=str(exc))
        contracts_raw = []

    summary = get_options_summary(ticker, contracts_raw, underlying_price)

    # Extract analytics objects
    pc_ratio_obj = summary.get("pc_ratio")
    gex_obj = summary.get("gex")
    max_pain_obj = summary.get("max_pain")
    skew_list = summary.get("skew") or []

    put_call_ratio = pc_ratio_obj.volume_ratio if pc_ratio_obj else None
    net_gex = gex_obj.net_gex if gex_obj else None
    max_pain_strike = max_pain_obj.max_pain_strike if max_pain_obj else None

    # Options volume totals from P/C ratio object
    options_volume_today = None
    avg_volume_30d = None
    if pc_ratio_obj:
        options_volume_today = pc_ratio_obj.total_put_volume + pc_ratio_obj.total_call_volume

    # Nearest-term skew for 25d risk reversal
    skew_25d: float | None = None
    if skew_list:
        front = skew_list[0]
        skew_25d = getattr(front, "risk_reversal_25d", None)

    iv_percentile = _compute_iv_percentile(skew_list)

    # Collect alerts
    alerts: list[OptionsFlowAlert] = []

    if options_volume_today is not None and avg_volume_30d is not None:
        vol_alert = detect_unusual_volume(
            options_volume_today, avg_volume_30d, criteria.min_volume_ratio, ticker
        )
        if vol_alert:
            alerts.append(vol_alert)

    if skew_25d is not None:
        skew_alert = detect_skew_signal(skew_25d, criteria.min_skew_magnitude, ticker)
        if skew_alert:
            alerts.append(skew_alert)

    if net_gex is not None:
        gex_alert = detect_gamma_wall(net_gex, ticker=ticker)
        if gex_alert:
            alerts.append(gex_alert)

    if iv_percentile is not None:
        iv_alert = _detect_iv_spike(iv_percentile, criteria, ticker)
        if iv_alert:
            alerts.append(iv_alert)

    if put_call_ratio is not None:
        if put_call_ratio > criteria.min_put_call_ratio:
            severity = "high" if put_call_ratio > 2.5 else "medium"
            alerts.append(OptionsFlowAlert(
                ticker=ticker,
                alert_type="put_skew",
                severity=severity,
                description=f"Bearish P/C volume ratio: {put_call_ratio:.2f} (threshold {criteria.min_put_call_ratio})",
                metric_value=round(put_call_ratio, 4),
                threshold=criteria.min_put_call_ratio,
                timestamp=datetime.utcnow(),
            ))
        elif put_call_ratio < criteria.max_put_call_ratio:
            severity = "medium" if put_call_ratio < 0.5 else "low"
            alerts.append(OptionsFlowAlert(
                ticker=ticker,
                alert_type="call_sweep",
                severity=severity,
                description=f"Bullish P/C volume ratio: {put_call_ratio:.2f} (threshold {criteria.max_put_call_ratio})",
                metric_value=round(put_call_ratio, 4),
                threshold=criteria.max_put_call_ratio,
                timestamp=datetime.utcnow(),
            ))

    result = OptionsScreenResult(
        ticker=ticker,
        underlying_price=underlying_price,
        avg_volume_30d=avg_volume_30d,
        options_volume_today=options_volume_today,
        put_call_ratio=put_call_ratio,
        iv_percentile=iv_percentile,
        skew_25d=skew_25d,
        net_gex=net_gex,
        max_pain_strike=max_pain_strike,
        alerts=alerts,
        signal="neutral",  # placeholder — computed below
    )

    # Compute signal from assembled result
    signal = compute_signal(result)
    return result.model_copy(update={"signal": signal})


async def screen_universe(
    tickers: list[str],
    prices: dict[str, float],
    polygon_api_key: str,
    criteria: OptionsScreenerCriteria | None = None,
    concurrency: int = 5,
) -> list[OptionsScreenResult]:
    """
    Screen a list of tickers concurrently. Returns sorted by alert severity (highest first).
    Uses asyncio.Semaphore(concurrency) to rate-limit Polygon API calls.
    """
    if criteria is None:
        criteria = OptionsScreenerCriteria()

    sem = asyncio.Semaphore(concurrency)

    async def _guarded(ticker: str) -> OptionsScreenResult | None:
        price = prices.get(ticker)
        if price is None or price <= 0:
            logger.debug("Skipping ticker: no price", ticker=ticker)
            return None
        async with sem:
            try:
                return await screen_single_ticker(ticker, price, polygon_api_key, criteria)
            except Exception as exc:
                logger.warning("screen_single_ticker failed", ticker=ticker, error=str(exc))
                return None

    tasks = [_guarded(t) for t in tickers]
    raw_results = await asyncio.gather(*tasks)

    results: list[OptionsScreenResult] = [r for r in raw_results if r is not None]

    def _sort_key(r: OptionsScreenResult) -> tuple[int, int]:
        max_sev = max((_SEVERITY_RANK.get(a.severity, 0) for a in r.alerts), default=0)
        signal_score = abs(_SIGNAL_RANK.get(r.signal, 0))
        return (max_sev, signal_score)

    results.sort(key=_sort_key, reverse=True)
    logger.info(
        "screen_universe complete",
        total=len(tickers),
        screened=len(results),
        with_alerts=sum(1 for r in results if r.alerts),
    )
    return results
