"""Alternative data proxy signals from free public sources: FRED PCE, GDELT, shipping, trends — Dimension 94 enhancement."""
from __future__ import annotations
import asyncio
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import httpx
from pydantic import BaseModel, ConfigDict, Field
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ── FRED series IDs ───────────────────────────────────────────────────────────
PCE_SERIES: dict[str, str] = {
    "retail_goods": "PCND",
    "durable_goods": "PCDG",
    "services": "PCESV",
    "food_bev": "DFXARC1M027SBEA",
    "energy": "DSERC1M027SBEA",
    "recreation": "DNDGRC1M027SBEA",
}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SHIPPING_SERIES = "DCOILWTICO"      # WTI as shipping-cost proxy
MANUFACTURING_SERIES = "MANEMP"    # Manufacturing employment — activity proxy
PORT_ACTIVITY_SERIES = "IPB50001N" # Industrial production: Manufacturing

# ── GDELT 2.0 API ─────────────────────────────────────────────────────────────
GDELT_API = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=timelinevolnorm&format=json"
    "&startdatetime={start}&enddatetime={end}&smoothing=7"
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class PCESignal(BaseModel):
    """Signal for a single PCE spending category."""
    model_config = ConfigDict(frozen=True)
    category: str
    series_id: str
    latest_value: Optional[float] = None
    mom_change: Optional[float] = None   # percent MoM
    yoy_change: Optional[float] = None   # percent YoY
    trend: str = "stable"                # "expanding" | "contracting" | "stable"
    z_score: Optional[float] = None      # vs 24-month window, clamped [-3, 3]

class PCEDashboard(BaseModel):
    model_config = ConfigDict(frozen=True)
    signals: list[PCESignal] = Field(default_factory=list)
    consumer_strength_score: float = 5.0   # 0-10 composite
    spending_rotation: str = "balanced"    # "goods → services" | "goods → services (lag)" | "balanced"
    as_of: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)

class GDELTSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: str
    volume_7d_avg: Optional[float] = None
    volume_30d_avg: Optional[float] = None
    volume_trend: str = "stable"           # "surging" | "declining" | "stable"
    sentiment_tone: Optional[float] = None # GDELT AvgTone [-10, +10]
    peak_date: Optional[str] = None
    as_of: datetime = Field(default_factory=datetime.utcnow)

class TrendsSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    ticker: str
    interest_7d: Optional[float] = None   # 0-100 Google normalized
    interest_30d: Optional[float] = None
    interest_52w_high: Optional[float] = None
    interest_52w_low: Optional[float] = None
    trend_direction: str = "stable"        # "rising" | "falling" | "stable"
    breakout: bool = False                 # True when 7d > 1.5× 30d avg

class ShippingProxySignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    manufacturing_pmi_proxy: Optional[float] = None  # MANEMP MoM % change
    industrial_production: Optional[float] = None    # IPB50001N latest
    wti_price: Optional[float] = None                # DCOILWTICO latest
    shipping_regime: str = "neutral"                 # "expansion" | "contraction" | "neutral"
    as_of: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)

class AltDataDashboard(BaseModel):
    model_config = ConfigDict(frozen=True)
    pce_dashboard: Optional[PCEDashboard] = None
    gdelt_signals: list[GDELTSignal] = Field(default_factory=list)
    trends_signals: list[TrendsSignal] = Field(default_factory=list)
    shipping_signal: Optional[ShippingProxySignal] = None
    composite_alt_score: float = 5.0      # 0-10
    alt_regime: str = "neutral"           # "bullish" | "bearish" | "neutral"
    key_insights: list[str] = Field(default_factory=list)  # 3-5 observations
    as_of: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)

class AltDataSignalRow(BaseModel):
    """Flat row for tabular display."""
    model_config = ConfigDict(frozen=True)
    category: str
    signal: str
    value_str: str
    trend: str
    as_of: datetime = Field(default_factory=datetime.utcnow)

# ── FRED helpers ──────────────────────────────────────────────────────────────
async def _fetch_fred_pce_series(sid: str, client: httpx.AsyncClient) -> list[tuple[str, float]]:
    """GET FRED CSV for sid; return last 36 months of (date, value) pairs, missing-value filtered."""
    url = FRED_CSV.format(sid=sid)
    try:
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("fred_fetch_failed", series=sid, error=str(exc))
        return []
    rows: list[tuple[str, float]] = []
    for line in resp.text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2 or parts[1].strip() == ".":
            continue
        try:
            rows.append((parts[0].strip(), float(parts[1].strip())))
        except ValueError:
            continue
    return rows[-36:]

def _pce_z_score(values: list[float], window: int = 24) -> Optional[float]:
    """Z-score of values[-1] vs values[-window:]; clamped to [-3, 3]."""
    if len(values) < window + 1:
        return None
    w = values[-window:]
    std = float(np.std(w, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.clip((values[-1] - float(np.mean(w))) / std, -3.0, 3.0))

def _pce_trend(mom: Optional[float], yoy: Optional[float]) -> str:
    """Both positive → expanding; both negative → contracting; else → stable."""
    if mom is None or yoy is None:
        return "stable"
    if mom > 0 and yoy > 0:
        return "expanding"
    if mom < 0 and yoy < 0:
        return "contracting"
    return "stable"

def _build_pce_signal(category: str, sid: str, rows: list[tuple[str, float]]) -> PCESignal:
    if not rows:
        return PCESignal(category=category, series_id=sid)
    values = [v for _, v in rows]
    latest = values[-1]
    mom = (values[-1] - values[-2]) / abs(values[-2]) * 100.0 if len(values) >= 2 and values[-2] != 0 else None
    yoy = (values[-1] - values[-13]) / abs(values[-13]) * 100.0 if len(values) >= 13 and values[-13] != 0 else None
    return PCESignal(
        category=category, series_id=sid,
        latest_value=round(latest, 2),
        mom_change=round(mom, 4) if mom is not None else None,
        yoy_change=round(yoy, 4) if yoy is not None else None,
        trend=_pce_trend(mom, yoy),
        z_score=round(z, 4) if (z := _pce_z_score(values)) is not None else None,
    )

async def get_pce_dashboard(client: httpx.AsyncClient) -> PCEDashboard:
    """Fetch all PCE series in parallel; compute consumer_strength_score and spending_rotation."""
    warnings: list[str] = []
    results = await asyncio.gather(
        *[_fetch_fred_pce_series(sid, client) for sid in PCE_SERIES.values()],
        return_exceptions=True,
    )
    category_rows: dict[str, list[tuple[str, float]]] = {}
    for category, result in zip(PCE_SERIES.keys(), results):
        if isinstance(result, Exception):
            warnings.append(f"PCE fetch failed for {category}: {result}")
            category_rows[category] = []
        else:
            category_rows[category] = result  # type: ignore[assignment]

    signals = [_build_pce_signal(cat, sid, category_rows[cat]) for cat, sid in PCE_SERIES.items()]

    # Weighted z-score → 0-10 consumer strength; weights sum to 1.0
    weights = {"retail_goods": 0.25, "durable_goods": 0.20, "services": 0.30,
               "food_bev": 0.10, "energy": 0.08, "recreation": 0.07}
    sig_by_cat = {s.category: s for s in signals}
    weighted_z = total_w = 0.0
    for cat, w in weights.items():
        s = sig_by_cat.get(cat)
        if s and s.z_score is not None:
            weighted_z += w * s.z_score
            total_w += w
    if total_w > 0:
        score = round(float(np.clip((weighted_z / total_w + 3.0) / 6.0 * 10.0, 0.0, 10.0)), 2)
    else:
        score = 5.0
        warnings.append("Insufficient PCE data for consumer strength score")

    # Spending rotation: durable goods YoY vs services YoY
    g = sig_by_cat.get("durable_goods")
    sv = sig_by_cat.get("services")
    rotation = "balanced"
    if g and sv and g.yoy_change is not None and sv.yoy_change is not None:
        if g.yoy_change > sv.yoy_change + 2.0:
            rotation = "goods → services (lag)"
        elif sv.yoy_change > g.yoy_change + 2.0:
            rotation = "goods → services"

    return PCEDashboard(signals=signals, consumer_strength_score=score,
                        spending_rotation=rotation, as_of=datetime.utcnow(), warnings=warnings)

# ── GDELT helpers ─────────────────────────────────────────────────────────────
async def _fetch_gdelt_signal(query: str, client: httpx.AsyncClient) -> Optional[GDELTSignal]:
    """Fetch GDELT 2.0 normalized-volume timeline for query; 15 s timeout."""
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    url = GDELT_API.format(
        query=urllib.parse.quote(query),
        start=start.strftime("%Y%m%d%H%M%S"),
        end=end.strftime("%Y%m%d%H%M%S"),
    )
    try:
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("gdelt_fetch_failed", query=query, error=str(exc))
        return None

    timeline = data.get("timeline")
    if not timeline or not isinstance(timeline, list):
        return None
    vol_pts = timeline[0].get("data", [])
    vol_vals = [pt["value"] for pt in vol_pts if "value" in pt]
    if not vol_vals:
        return None

    v7 = round(float(np.mean(vol_vals[-7:])), 4) if len(vol_vals) >= 7 else None
    v30 = round(float(np.mean(vol_vals[-30:])), 4) if len(vol_vals) >= 30 else None
    if v7 is not None and v30 is not None and v30 > 0:
        ratio = v7 / v30
        vol_trend = "surging" if ratio >= 1.4 else "declining" if ratio <= 0.7 else "stable"
    else:
        vol_trend = "stable"

    peak_date: Optional[str] = max(vol_pts, key=lambda p: p.get("value", 0.0)).get("date") if vol_pts else None

    tone: Optional[float] = None
    if len(timeline) >= 2:
        tone_vals = [pt["value"] for pt in timeline[1].get("data", []) if "value" in pt]
        if tone_vals:
            tone = round(float(np.mean(tone_vals[-7:])), 4)

    return GDELTSignal(query=query, volume_7d_avg=v7, volume_30d_avg=v30,
                       volume_trend=vol_trend, sentiment_tone=tone,
                       peak_date=peak_date, as_of=datetime.utcnow())

# ── Google Trends (pytrends lazy import) ──────────────────────────────────────
def _build_trends_signals_sync(tickers: list[str]) -> list[TrendsSignal]:
    """Blocking pytrends call — runs inside asyncio.to_thread."""
    try:
        from pytrends.request import TrendReq  # type: ignore[import]
    except ImportError:
        logger.warning("pytrends_not_installed")
        return []
    try:
        pt = TrendReq(hl="en-US", tz=360)
        pt.build_payload(tickers[:5], timeframe="today 12-m")
        df = pt.interest_over_time()
    except Exception as exc:
        logger.warning("pytrends_fetch_failed", error=str(exc))
        return []
    if df is None or df.empty:
        return []

    signals: list[TrendsSignal] = []
    for ticker in tickers[:5]:
        if ticker not in df.columns:
            signals.append(TrendsSignal(ticker=ticker))
            continue
        vals = df[ticker].dropna().values.astype(float)
        if len(vals) == 0:
            signals.append(TrendsSignal(ticker=ticker))
            continue
        i7 = round(float(np.mean(vals[-1:])), 2)
        i30 = round(float(np.mean(vals[-4:])), 2) if len(vals) >= 4 else None
        hi = round(float(np.max(vals)), 2)
        lo = round(float(np.min(vals)), 2)
        breakout = bool(i30 and i30 > 0 and i7 > 1.5 * i30)
        slope = float(np.polyfit(np.arange(len(vals[-8:]), dtype=float), vals[-8:], 1)[0]) if len(vals) >= 8 else 0.0
        direction = "rising" if slope > 1.0 else "falling" if slope < -1.0 else "stable"
        signals.append(TrendsSignal(ticker=ticker, interest_7d=i7, interest_30d=i30,
                                    interest_52w_high=hi, interest_52w_low=lo,
                                    trend_direction=direction, breakout=breakout))
    return signals

async def _fetch_google_trends(tickers: list[str]) -> list[TrendsSignal]:
    try:
        return await asyncio.to_thread(_build_trends_signals_sync, tickers)
    except Exception as exc:
        logger.warning("google_trends_thread_failed", error=str(exc))
        return []

# ── Shipping / manufacturing proxy ────────────────────────────────────────────
async def _fetch_shipping_proxy(client: httpx.AsyncClient) -> ShippingProxySignal:
    """Fetch MANEMP, IPB50001N, DCOILWTICO from FRED in parallel; classify shipping regime."""
    warnings: list[str] = []
    mfg_rows, ip_rows, wti_rows = await asyncio.gather(
        _fetch_fred_pce_series(MANUFACTURING_SERIES, client),
        _fetch_fred_pce_series(PORT_ACTIVITY_SERIES, client),
        _fetch_fred_pce_series(SHIPPING_SERIES, client),
    )
    mfg_proxy: Optional[float] = None
    if len(mfg_rows) >= 2 and mfg_rows[-2][1] != 0:
        mfg_proxy = round((mfg_rows[-1][1] - mfg_rows[-2][1]) / abs(mfg_rows[-2][1]) * 100.0, 4)
    else:
        warnings.append("Insufficient manufacturing employment data")

    ip_latest: Optional[float] = None
    ip_mom: Optional[float] = None
    if ip_rows:
        ip_latest = round(ip_rows[-1][1], 2)
        if len(ip_rows) >= 2 and ip_rows[-2][1] != 0:
            ip_mom = round((ip_rows[-1][1] - ip_rows[-2][1]) / abs(ip_rows[-2][1]) * 100.0, 4)
    else:
        warnings.append("Insufficient industrial production data")

    wti = round(wti_rows[-1][1], 2) if wti_rows else None
    if not wti_rows:
        warnings.append("Insufficient WTI data")

    expanding = (ip_mom is not None and ip_mom > 0) and (mfg_proxy is not None and mfg_proxy > 0)
    contracting = (ip_mom is not None and ip_mom < 0) and (mfg_proxy is not None and mfg_proxy < 0)
    regime = "expansion" if expanding else "contraction" if contracting else "neutral"

    return ShippingProxySignal(manufacturing_pmi_proxy=mfg_proxy, industrial_production=ip_latest,
                               wti_price=wti, shipping_regime=regime,
                               as_of=datetime.utcnow(), warnings=warnings)

# ── Composite score ───────────────────────────────────────────────────────────
def _composite_alt_score(pce: PCEDashboard, gdelt_signals: list[GDELTSignal],
                          trends: list[TrendsSignal], shipping: ShippingProxySignal) -> float:
    """0-10 composite: PCE 40% + GDELT 20% + Trends 20% + Shipping 20%."""
    _vol_map = {"surging": 8.0, "stable": 5.0, "declining": 2.0}
    _reg_map = {"expansion": 8.0, "neutral": 5.0, "contraction": 2.0}
    pce_c = pce.consumer_strength_score * 0.40
    gdelt_c = float(np.mean([_vol_map.get(g.volume_trend, 5.0) for g in gdelt_signals])) * 0.20 if gdelt_signals else 5.0 * 0.20
    trends_c = (sum(1 for t in trends if t.breakout) / len(trends) * 10.0 * 0.20) if trends else 0.0
    ship_c = _reg_map.get(shipping.shipping_regime, 5.0) * 0.20
    return round(float(np.clip(pce_c + gdelt_c + trends_c + ship_c, 0.0, 10.0)), 2)

# ── Key insights ──────────────────────────────────────────────────────────────
def _generate_key_insights(pce: PCEDashboard, gdelt_signals: list[GDELTSignal],
                            trends: list[TrendsSignal], shipping: ShippingProxySignal) -> list[str]:
    """Generate 3-5 human-readable observations from the strongest alt-data signals."""
    insights: list[str] = []
    pce_label = "strong" if pce.consumer_strength_score >= 7.0 else "weak" if pce.consumer_strength_score <= 3.0 else "moderate"
    insights.append(f"Consumer spending is {pce_label} (PCE score {pce.consumer_strength_score:.1f}/10); "
                    f"rotation: {pce.spending_rotation}.")
    expanding = [s for s in pce.signals if s.trend == "expanding" and s.yoy_change is not None]
    if expanding:
        top = max(expanding, key=lambda s: s.yoy_change or 0.0)
        insights.append(f"Fastest-expanding PCE category: {top.category} (YoY {top.yoy_change:+.1f}%).")
    surging = [g for g in gdelt_signals if g.volume_trend == "surging"]
    if surging:
        g0 = surging[0]
        insights.append(f"News volume for '{g0.query}' surging: "
                        f"7d avg {g0.volume_7d_avg:.2f} vs 30d avg {g0.volume_30d_avg:.2f}.")
    elif gdelt_signals:
        avg_vol = float(np.mean([g.volume_7d_avg or 0.0 for g in gdelt_signals]))
        insights.append(f"Aggregate news volume for tracked tickers stable at {avg_vol:.2f} normalized units.")
    breakouts = [t.ticker for t in trends if t.breakout]
    if breakouts:
        insights.append(f"Google Trends breakout detected: {', '.join(breakouts)} (7d > 1.5× 30d avg).")
    mfg_str = f"MfgMoM={shipping.manufacturing_pmi_proxy:+.2f}%" if shipping.manufacturing_pmi_proxy is not None else "MfgMoM=n/a"
    insights.append(f"Shipping proxy regime: {shipping.shipping_regime} "
                    f"(IP={shipping.industrial_production}, {mfg_str}, WTI={shipping.wti_price}).")
    return insights[:5]

# ── Public entry points ───────────────────────────────────────────────────────
async def get_alt_data_dashboard(tickers: Optional[list[str]] = None) -> AltDataDashboard:
    """
    Assemble a full alternative-data dashboard from FRED PCE, GDELT news volume,
    Google Trends, and FRED-based shipping/manufacturing proxies.

    Parameters
    ----------
    tickers:
        Consumer-proxy ticker symbols for GDELT and Trends queries.
        Defaults to ["AMZN", "WMT", "TSLA"].

    Returns
    -------
    AltDataDashboard
        Composite view with 0-10 score and bullish/bearish/neutral regime label.
    """
    if tickers is None:
        tickers = ["AMZN", "WMT", "TSLA"]
    warnings: list[str] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "SENTINEL/1.0 research@sentinel.ai"},
        follow_redirects=True,
    ) as client:
        pce_task = asyncio.create_task(get_pce_dashboard(client))
        ship_task = asyncio.create_task(_fetch_shipping_proxy(client))
        gdelt_tasks = [asyncio.create_task(_fetch_gdelt_signal(t, client)) for t in tickers]
        trends_task = asyncio.create_task(_fetch_google_trends(tickers))

        pce_result, ship_result = await asyncio.gather(pce_task, ship_task, return_exceptions=True)
        gdelt_raw = await asyncio.gather(*gdelt_tasks, return_exceptions=True)
        trends_result = await trends_task

    if isinstance(pce_result, Exception):
        warnings.append(f"PCE dashboard failed: {pce_result}")
        pce_dashboard: PCEDashboard = PCEDashboard(warnings=[str(pce_result)])
    else:
        pce_dashboard = pce_result  # type: ignore[assignment]

    if isinstance(ship_result, Exception):
        warnings.append(f"Shipping proxy failed: {ship_result}")
        shipping_signal: ShippingProxySignal = ShippingProxySignal(warnings=[str(ship_result)])
    else:
        shipping_signal = ship_result  # type: ignore[assignment]

    gdelt_signals: list[GDELTSignal] = []
    for ticker, res in zip(tickers, gdelt_raw):
        if isinstance(res, Exception):
            warnings.append(f"GDELT failed for {ticker}: {res}")
        elif res is not None:
            gdelt_signals.append(res)

    trends_signals: list[TrendsSignal] = trends_result if isinstance(trends_result, list) else []

    composite = _composite_alt_score(pce_dashboard, gdelt_signals, trends_signals, shipping_signal)
    alt_regime = "bullish" if composite > 6.0 else "bearish" if composite < 4.0 else "neutral"
    insights = _generate_key_insights(pce_dashboard, gdelt_signals, trends_signals, shipping_signal, composite)

    return AltDataDashboard(
        pce_dashboard=pce_dashboard, gdelt_signals=gdelt_signals,
        trends_signals=trends_signals, shipping_signal=shipping_signal,
        composite_alt_score=composite, alt_regime=alt_regime,
        key_insights=insights, as_of=datetime.utcnow(), warnings=warnings,
    )

async def get_alt_signal_summary(tickers: Optional[list[str]] = None) -> list[AltDataSignalRow]:
    """
    Thin wrapper: calls get_alt_data_dashboard and flattens all signals into
    a list of AltDataSignalRow objects for tabular display.

    Parameters
    ----------
    tickers:
        Consumer-proxy tickers. Defaults to ["AMZN", "WMT", "TSLA"].

    Returns
    -------
    list[AltDataSignalRow]
        Flat rows ordered: PCE categories → GDELT → Trends → Shipping → Composite.
    """
    db = await get_alt_data_dashboard(tickers=tickers)
    rows: list[AltDataSignalRow] = []
    now = db.as_of

    if db.pce_dashboard:
        for s in db.pce_dashboard.signals:
            yoy_str = f"{s.yoy_change:+.2f}% YoY" if s.yoy_change is not None else "n/a"
            z_str = f"z={s.z_score:.2f}" if s.z_score is not None else "z=n/a"
            rows.append(AltDataSignalRow(category="PCE", signal=s.category,
                                         value_str=f"{s.latest_value} ({yoy_str}, {z_str})",
                                         trend=s.trend, as_of=now))
        rows.append(AltDataSignalRow(
            category="PCE", signal="consumer_strength",
            value_str=f"{db.pce_dashboard.consumer_strength_score:.1f}/10 | {db.pce_dashboard.spending_rotation}",
            trend="composite", as_of=now,
        ))

    for g in db.gdelt_signals:
        vol_str = (f"7d={g.volume_7d_avg:.2f} 30d={g.volume_30d_avg:.2f}"
                   if g.volume_7d_avg is not None and g.volume_30d_avg is not None else "n/a")
        tone_str = f" tone={g.sentiment_tone:.1f}" if g.sentiment_tone is not None else ""
        rows.append(AltDataSignalRow(category="GDELT", signal=g.query,
                                     value_str=f"{vol_str}{tone_str}",
                                     trend=g.volume_trend, as_of=now))

    for t in db.trends_signals:
        val_str = (f"7d={t.interest_7d} 30d={t.interest_30d} "
                   f"52w[{t.interest_52w_low}-{t.interest_52w_high}]"
                   f"{' BREAKOUT' if t.breakout else ''}")
        rows.append(AltDataSignalRow(category="Trends", signal=t.ticker,
                                     value_str=val_str, trend=t.trend_direction, as_of=now))

    if db.shipping_signal:
        sh = db.shipping_signal
        mfg_str = f"MfgMoM={sh.manufacturing_pmi_proxy:+.2f}%" if sh.manufacturing_pmi_proxy is not None else "MfgMoM=n/a"
        rows.append(AltDataSignalRow(
            category="Shipping", signal="manufacturing_proxy",
            value_str=f"IP={sh.industrial_production} {mfg_str} WTI={sh.wti_price}",
            trend=sh.shipping_regime, as_of=now,
        ))

    rows.append(AltDataSignalRow(
        category="ALT_COMPOSITE", signal="alt_data_score",
        value_str=f"{db.composite_alt_score:.1f}/10",
        trend=db.alt_regime, as_of=now,
    ))
    return rows
