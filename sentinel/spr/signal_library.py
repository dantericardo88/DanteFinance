"""
Composite alpha signal library — Dimension 68 (AI-driven factor research).

Synthesises congressional STOCK Act trades, CFTC COT data, and SEC Form 4
insider transactions into z-scored alpha signals with cluster detection.

Sources (primary → fallback):
  Congress : sod.congressional.CongressionalTradeTracker → sds.adapters.congress_adapter
  Insider  : sds.adapters.insider_adapter → sfe.form4_parser
  COT      : sma.cot_report.COTClient
"""
from __future__ import annotations
import asyncio
import statistics
from datetime import date, timedelta
from typing import Optional
from pydantic import BaseModel
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_AMOUNT_MIDPOINTS: dict[str, float] = {
    "$1,001 - $15,000": 8_000, "$15,001 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000, "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000, "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000, "Over $5,000,000": 7_500_000,
}
_CHAMBER_W = {"senate": 1.2, "house": 1.0}
_C_SUITE = {"ceo", "chief executive", "cfo", "chief financial", "coo", "chief operating", "president"}


# ── Models ─────────────────────────────────────────────────────────────────────

class AlphaSignal(BaseModel):
    source: str
    ticker: Optional[str] = None
    commodity: Optional[str] = None
    signal: str                    # strong_buy | buy | neutral | sell | strong_sell
    strength: float                # 0-1
    direction: float               # -1 to +1
    z_score: Optional[float] = None
    lookback_days: int
    data_points: int
    last_event_date: Optional[date] = None
    explanation: str
    confidence: str                # high | medium | low

class CongressAlphaResult(BaseModel):
    ticker: str; signal: AlphaSignal; net_buy_amount: float
    unique_members_buying: int; unique_members_selling: int
    cluster_detected: bool; most_notable_member: Optional[str] = None

class COTAlphaResult(BaseModel):
    market: str; signal: AlphaSignal
    cot_index: Optional[float] = None
    nc_net: Optional[int] = None; nc_net_change: Optional[int] = None
    extreme_positioning: bool

class InsiderAlphaResult(BaseModel):
    ticker: str; signal: AlphaSignal
    net_shares_purchased: float; cluster_score: float
    c_suite_buying: bool; days_since_last_purchase: Optional[int] = None

class CompositeSignal(BaseModel):
    ticker: str
    congress: Optional[CongressAlphaResult] = None
    insider: Optional[InsiderAlphaResult] = None
    composite_signal: AlphaSignal
    signal_count: int
    alignment: str   # all_bullish | all_bearish | mixed | insufficient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _neutral(source: str, ticker: Optional[str] = None, commodity: Optional[str] = None,
             lookback_days: int = 180, reason: str = "No data") -> AlphaSignal:
    return AlphaSignal(source=source, ticker=ticker, commodity=commodity,
                       signal="neutral", strength=0.0, direction=0.0,
                       lookback_days=lookback_days, data_points=0,
                       explanation=reason, confidence="low")

def _label(direction: float, z: Optional[float]) -> str:
    """Map direction + z-score to discrete signal label."""
    az = abs(z) if z is not None else abs(direction) * 3
    pos = direction > 0
    if az >= 2.0: return "strong_buy" if pos else "strong_sell"
    if az >= 1.0: return "buy" if pos else "sell"
    return "neutral"

def _zscore(val: float, history: list[float]) -> Optional[float]:
    if len(history) < 4: return None
    try:
        std = statistics.stdev(history)
        return round((val - statistics.mean(history)) / std, 2) if std else 0.0
    except Exception: return None

def _confidence(n: int, z: Optional[float]) -> str:
    if n >= 10 and z is not None and abs(z) >= 1.5: return "high"
    if n >= 4: return "medium"
    return "low"

def _norm_tx(raw: str) -> str:
    r = raw.lower()
    if r in ("p", "purchase", "buy"): return "buy"
    if r in ("s", "sale", "sell"): return "sell"
    return "other"


# ── Congressional alpha ─────────────────────────────────────────────────────────

async def congress_alpha_signal(ticker: str, lookback_days: int = 180) -> CongressAlphaResult:
    """
    Congressional STOCK Act alpha signal for a ticker.

    Fetches disclosures via CongressionalTradeTracker (primary) or CongressAdapter
    (fallback). Net buy = Σ purchase midpoints − Σ sale midpoints, weighted by
    chamber (Senate 1.2×, House 1.0×). Z-score vs rolling 12-month buckets.
    Cluster = 3+ unique members trading within any 30-day window.
    """
    null = CongressAlphaResult(
        ticker=ticker,
        signal=_neutral("congress", ticker=ticker, lookback_days=lookback_days),
        net_buy_amount=0.0, unique_members_buying=0, unique_members_selling=0,
        cluster_detected=False,
    )
    cutoff = date.today() - timedelta(days=lookback_days)
    trades: list[dict] = []

    try:
        from sentinel.sod.congressional import CongressionalTradeTracker  # type: ignore
        tracker = CongressionalTradeTracker()
        raw = await tracker.fetch_all_recent(lookback_days=max(lookback_days, 365))
        for t in raw:
            if not (t.ticker and t.ticker.upper() == ticker.upper()): continue
            trades.append({"politician": t.politician_name, "chamber": t.chamber.lower(),
                           "tx_type": (t.tx_type or "").lower(), "tx_date": t.tx_date or date.min,
                           "amount_low": float(t.amount_low or 0), "amount_high": float(t.amount_high or 0)})
    except Exception as exc:
        logger.warning("CongressionalTradeTracker unavailable", error=str(exc))

    if not trades:
        try:
            from sentinel.sds.adapters.congress_adapter import CongressAdapter  # type: ignore
            raw_d = await CongressAdapter().fetch_trades_by_ticker(ticker, since=cutoff)
            for t in raw_d:
                lo, hi = float(t.get("amount_low") or 0), float(t.get("amount_high") or 0)
                trades.append({"politician": t.get("politician_name", "?"), "chamber": (t.get("chamber") or "house").lower(),
                                "tx_type": _norm_tx(t.get("tx_type") or t.get("tx_code") or ""),
                                "tx_date": t.get("tx_date") or date.min, "amount_low": lo, "amount_high": hi})
        except Exception as exc:
            logger.warning("CongressAdapter unavailable", error=str(exc))

    window = [t for t in trades if t["tx_date"] >= cutoff]
    if not window: return null

    buyers: set[str] = set(); sellers: set[str] = set()
    net_buy = 0.0; last_date: Optional[date] = None; notable = None; notable_amt = 0.0

    for t in window:
        mid = (t["amount_low"] + t["amount_high"]) / 2 * _CHAMBER_W.get(t["chamber"], 1.0)
        if t["tx_type"] == "buy":
            net_buy += mid; buyers.add(t["politician"])
            if mid > notable_amt: notable_amt, notable = mid, t["politician"]
        elif t["tx_type"] == "sell":
            net_buy -= mid; sellers.add(t["politician"])
        d = t["tx_date"]
        if last_date is None or d > last_date: last_date = d

    # Cluster: 3+ unique buyers, or 3 trades within any 30-day window
    cluster = len(buyers) >= 3
    if not cluster:
        sd = sorted(t["tx_date"] for t in window)
        cluster = any(sum(1 for dd in sd[i:] if dd <= d + timedelta(30)) >= 3 for i, d in enumerate(sd))

    # Z-score: monthly net-buy buckets from full trade history
    monthly = [sum((t["amount_low"] + t["amount_high"]) / 2 * (1 if t["tx_type"] == "buy" else -1 if t["tx_type"] == "sell" else 0)
                   for t in trades if (date.today() - timedelta((m) * 30)) <= t["tx_date"] < (date.today() - timedelta((m - 1) * 30)))
               for m in range(1, 13)]
    z = _zscore(net_buy, monthly)
    strength = min(abs(z) / 3.0, 1.0) if z else min(abs(net_buy) / 5_000_000, 1.0)
    direction = 1.0 if net_buy > 0 else -1.0 if net_buy < 0 else 0.0
    expl = (f"{len(window)} trades by {len(buyers | sellers)} members in {lookback_days}d. "
            f"Net buy: ${net_buy:,.0f}. {'Cluster detected. ' if cluster else ''}"
            f"Z={z:.2f}." if z else f"{len(window)} trades, net ${net_buy:,.0f}.")

    return CongressAlphaResult(
        ticker=ticker,
        signal=AlphaSignal(source="congress", ticker=ticker, signal=_label(direction, z),
                           strength=round(strength, 4), direction=round(direction * strength, 4),
                           z_score=z, lookback_days=lookback_days, data_points=len(window),
                           last_event_date=last_date, explanation=expl,
                           confidence=_confidence(len(window), z)),
        net_buy_amount=round(net_buy, 2), unique_members_buying=len(buyers),
        unique_members_selling=len(sellers), cluster_detected=cluster, most_notable_member=notable,
    )


# ── COT alpha ──────────────────────────────────────────────────────────────────

async def cot_alpha_signal(market: str, lookback_weeks: int = 52) -> COTAlphaResult:
    """
    CFTC COT contrarian signal. COT index > 80 = crowded long = bearish.
    COT index < 20 = crowded short = bullish. Strength = |index − 50| / 50.
    """
    null = COTAlphaResult(market=market, signal=_neutral("cot", commodity=market,
                          lookback_days=lookback_weeks * 7, reason="COT data unavailable"),
                          extreme_positioning=False)
    try:
        from sentinel.sma.cot_report import COTClient  # type: ignore
        client = COTClient()
        yr = date.today().year
        await client.load_range(yr - (2 if lookback_weeks > 52 else 1), yr)
        df = client.compute_cot_index(market, lookback_weeks=lookback_weeks)
        if df is None or df.empty: return null

        latest, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else df.iloc[-1])
        idx = float(latest.get("cot_index", 50) or 50)
        nc = int(latest.get("net_position", 0) or 0)
        chg = nc - int(prev.get("net_position", 0) or 0)
        extreme = idx >= 80 or idx <= 20
        strength = round(abs(idx - 50) / 50, 4)
        direction = round(-((idx - 50) / 50), 4)  # contrarian: >50 = bearish = negative dir

        if idx >= 90: lbl = "strong_sell"
        elif idx >= 80: lbl = "sell"
        elif idx <= 10: lbl = "strong_buy"
        elif idx <= 20: lbl = "buy"
        elif 40 <= idx <= 60: lbl = "neutral"
        else: lbl = "buy" if idx < 50 else "sell"

        z_hist = df["net_position"].dropna().tolist()
        z = _zscore(float(nc), z_hist[:-1])
        if z: z = round(-z, 2)  # contrarian: flip z sign

        expl = (f"COT index={idx:.1f}. Speculators {'extremely long (bearish signal)' if idx >= 80 else 'extremely short (bullish signal)' if idx <= 20 else 'neutral-ish'}. "
                f"Net pos: {nc:+,} (Δ{chg:+,}).")
        conf = "high" if len(df) >= 40 and extreme else ("medium" if len(df) >= 20 else "low")
        rpt_date = latest["date"].date() if hasattr(latest["date"], "date") else date.today()

        return COTAlphaResult(
            market=market,
            signal=AlphaSignal(source="cot", commodity=market, signal=lbl, strength=strength,
                               direction=direction, z_score=z, lookback_days=lookback_weeks * 7,
                               data_points=len(df), last_event_date=rpt_date, explanation=expl,
                               confidence=conf),
            cot_index=round(idx, 1), nc_net=nc, nc_net_change=chg, extreme_positioning=extreme,
        )
    except Exception as exc:
        logger.error("COT alpha error", market=market, error=str(exc))
        return null


# ── Insider alpha ───────────────────────────────────────────────────────────────

async def insider_alpha_signal(ticker: str, lookback_days: int = 90) -> InsiderAlphaResult:
    """
    Insider alpha signal from Form 4 open-market purchases (code = "P" only).
    Sales excluded — often diversification, not information-driven.

    Cluster score (0-10): buyers*1.5 (max 6) + c-suite bonus (3) + size bonus (1).
    Signal: score >= 7 = strong_buy, 5-7 = buy, 3-5 = neutral, <3 = weak/neutral.
    """
    null = InsiderAlphaResult(ticker=ticker,
                               signal=_neutral("insider", ticker=ticker, lookback_days=lookback_days),
                               net_shares_purchased=0.0, cluster_score=0.0,
                               c_suite_buying=False)
    cutoff = date.today() - timedelta(days=lookback_days)
    purchases: list[dict] = []

    try:
        from sentinel.sds.adapters.insider_adapter import fetch_insider_transactions_for_ticker  # type: ignore
        raw = await fetch_insider_transactions_for_ticker(ticker, since=cutoff)
        for t in raw:
            code = (t.get("transaction_code") or t.get("tx_code") or "").upper()
            if code != "P": continue
            tx_date = t.get("transaction_date") or t.get("tx_date")
            if isinstance(tx_date, str):
                from datetime import datetime as _dt
                try: tx_date = _dt.fromisoformat(tx_date).date()
                except Exception: tx_date = None
            if tx_date and tx_date < cutoff: continue
            purchases.append({"name": t.get("owner_name") or "?",
                               "role": (t.get("owner_role") or "").lower(),
                               "shares": float(t.get("shares") or 0),
                               "price": float(t.get("price_per_share") or 0),
                               "date": tx_date})
    except Exception as exc:
        logger.warning("insider_adapter unavailable", error=str(exc))

    if not purchases: return null

    total_shares = sum(p["shares"] for p in purchases)
    total_value = sum(p["shares"] * p["price"] for p in purchases)
    unique = {p["name"] for p in purchases}
    c_suite = any(any(t in p["role"] for t in _C_SUITE) for p in purchases)

    cluster_score = min(len(unique) * 1.5, 6.0) + (3.0 if c_suite else 0.0) + (1.0 if total_value >= 1_000_000 else 0.0)
    cluster_score = min(cluster_score, 10.0)

    purchase_dates = sorted(p["date"] for p in purchases if p.get("date"))
    days_since = (date.today() - max(purchase_dates)).days if purchase_dates else None
    last_date = max(purchase_dates) if purchase_dates else None

    if cluster_score >= 7.0: lbl, direction, strength, conf = "strong_buy", 0.9, 0.9, "high"
    elif cluster_score >= 5.0: lbl, direction, strength, conf = "buy", 0.6, 0.6, "medium"
    elif cluster_score >= 3.0: lbl, direction, strength, conf = "neutral", 0.2, 0.2, "low"
    else: lbl, direction, strength, conf = "neutral", 0.0, 0.0, "low"

    parts = [f"{len(unique)} insider(s) purchased {total_shares:,.0f} sh (~${total_value:,.0f}) in {lookback_days}d.",
             f"Cluster score: {cluster_score:.1f}/10."]
    if c_suite: parts.append("C-suite buying.")
    if days_since is not None: parts.append(f"Last purchase: {days_since}d ago.")

    return InsiderAlphaResult(
        ticker=ticker,
        signal=AlphaSignal(source="insider", ticker=ticker, signal=lbl, strength=round(strength, 4),
                           direction=round(direction, 4), z_score=None, lookback_days=lookback_days,
                           data_points=len(purchases), last_event_date=last_date,
                           explanation=" ".join(parts), confidence=conf),
        net_shares_purchased=round(total_shares, 2), cluster_score=round(cluster_score, 2),
        c_suite_buying=c_suite, days_since_last_purchase=days_since,
    )


# ── Composite ──────────────────────────────────────────────────────────────────

def composite_alpha_signal(congress: Optional[CongressAlphaResult],
                            insider: Optional[InsiderAlphaResult]) -> CompositeSignal:
    """
    Combine available signals. Weights: congress=0.4, insider=0.6.
    Alignment: all_bullish / all_bearish / mixed / insufficient.
    Mixed signals downgrade confidence one tier.
    """
    ticker = (congress.ticker if congress else None) or (insider.ticker if insider else "")
    avail: list[tuple[float, float, AlphaSignal]] = []  # (weight, direction, signal)
    if congress and congress.signal.data_points > 0: avail.append((0.4, congress.signal.direction, congress.signal))
    if insider and insider.signal.data_points > 0:   avail.append((0.6, insider.signal.direction, insider.signal))

    if not avail:
        return CompositeSignal(ticker=ticker, congress=congress, insider=insider,
                               composite_signal=_neutral("composite", ticker=ticker, reason="No signals"),
                               signal_count=0, alignment="insufficient")

    tw = sum(w for w, _, _ in avail)
    comp_dir = sum(w * d for w, d, _ in avail) / tw
    comp_str = min(abs(comp_dir), 1.0)
    dirs = [d for _, d, _ in avail]
    alignment = ("insufficient" if len(avail) < 2 else
                 "all_bullish" if all(d > 0 for d in dirs) else
                 "all_bearish" if all(d < 0 for d in dirs) else "mixed")

    zv = [(w, s.z_score) for w, _, s in avail if s.z_score is not None]
    comp_z: Optional[float] = round(sum(w * z for w, z in zv) / sum(w for w, _ in zv), 2) if zv else None

    conf_rank = {"high": 3, "medium": 2, "low": 1}
    best = max(conf_rank.get(s.confidence, 1) for _, _, s in avail)
    conf_map = {3: "high", 2: "medium", 1: "low"}
    downgrade = {3: "medium", 2: "low", 1: "low"}
    conf = downgrade[best] if alignment == "mixed" else conf_map[best]

    src_labels = [f"{s.source}={s.signal}({s.direction:+.2f})" for _, _, s in avail]
    expl = f"Composite [{', '.join(src_labels)}] dir={comp_dir:+.3f} alignment={alignment}."
    lookback = max(s.lookback_days for _, _, s in avail)
    last_dates = [s.last_event_date for _, _, s in avail if s.last_event_date]

    return CompositeSignal(
        ticker=ticker, congress=congress, insider=insider,
        composite_signal=AlphaSignal(
            source="composite", ticker=ticker, signal=_label(comp_dir, comp_z),
            strength=round(comp_str, 4), direction=round(comp_dir, 4), z_score=comp_z,
            lookback_days=lookback, data_points=sum(s.data_points for _, _, s in avail),
            last_event_date=max(last_dates) if last_dates else None,
            explanation=expl, confidence=conf),
        signal_count=len(avail), alignment=alignment,
    )


async def get_full_signal(ticker: str) -> CompositeSignal:
    """Convenience: fetch congress + insider signals concurrently, return composite."""
    c_res, i_res = await asyncio.gather(
        congress_alpha_signal(ticker), insider_alpha_signal(ticker), return_exceptions=True
    )
    return composite_alpha_signal(
        c_res if isinstance(c_res, CongressAlphaResult) else None,
        i_res if isinstance(i_res, InsiderAlphaResult) else None,
    )
