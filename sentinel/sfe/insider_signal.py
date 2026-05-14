"""insider_signal.py — Aggregate insider trading signal analytics (Dim 26, Wave-15b).

Builds composite insider conviction signals from yfinance Form 4 data:
net purchase ratio, cluster buy detection, officer sentiment, unusual size,
recent momentum, and a 0-10 composite score with verdict.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_OFFICER_KEYWORDS = frozenset(["ceo", "cfo", "cto", "coo", "president", "chairman", "chief"])
_PURCHASE_KW = frozenset(["purchase", "buy", "bought"])
_SALE_KW = frozenset(["sale", "sell", "sold"])
_CLUSTER_DAYS = 30
_UNUSUAL_MULT = 5.0
_RECENT_DAYS = 30
_PRIOR_DAYS = 150


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class InsiderTransaction(BaseModel):
    model_config = ConfigDict(frozen=True)

    insider_name: str
    position: str
    transaction_type: str
    shares: Optional[int] = None
    value_usd: Optional[float] = None
    date: str


class InsiderSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    period_days: int
    total_transactions: int
    buy_transactions: int
    sell_transactions: int
    total_buy_value_usd: float
    total_sell_value_usd: float
    net_purchase_ratio: Optional[float]
    cluster_buy: bool
    officer_sentiment: str
    unusual_size_buy: bool
    recent_momentum_positive: Optional[bool]
    signal_score: float
    signal_verdict: str
    score_components: dict[str, float]
    recent_transactions: list[InsiderTransaction]
    as_of: str
    warnings: list[str]


class InsiderSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    signal_score: float
    signal_verdict: str
    net_purchase_ratio: Optional[float]
    cluster_buy: bool
    officer_sentiment: str


class InsiderBuyingScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_score_filter: float
    results: list[InsiderSummary]
    cluster_buys: list[str]
    officer_buyers: list[str]
    top_signals: list[str]
    avg_signal_score: Optional[float]
    as_of: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Sync yfinance fetch
# ---------------------------------------------------------------------------


def _fetch_insider_sync(ticker: str) -> dict:
    """Fetch insider_transactions via yfinance. Lazy-imports yfinance and pandas."""
    import yfinance as yf  # noqa: PLC0415
    import pandas as pd    # noqa: PLC0415

    result: dict = {"df": None, "errors": []}
    try:
        df = yf.Ticker(ticker).insider_transactions
        if isinstance(df, pd.DataFrame) and not df.empty:
            result["df"] = df.copy()
    except Exception as exc:
        result["errors"].append(f"yfinance insider_transactions failed: {exc}")
    return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _classify_tx(tx_str: str) -> str:
    t = str(tx_str).lower()
    if any(k in t for k in _PURCHASE_KW):
        return "Purchase"
    if any(k in t for k in _SALE_KW):
        return "Sale"
    if "option" in t or "exercise" in t:
        return "Option Exercise"
    return tx_str


def _is_officer(position: str) -> bool:
    return any(kw in str(position).lower() for kw in _OFFICER_KEYWORDS)


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


def _safe_int(val) -> Optional[int]:
    try:
        f = float(val)
        return None if np.isnan(f) else int(f)
    except Exception:
        return None


def _parse_date_str(val) -> Optional[str]:
    if val is None:
        return None
    try:
        import pandas as pd  # noqa: PLC0415
        ts = pd.Timestamp(val)
        return None if ts is pd.NaT else ts.strftime("%Y-%m-%d")
    except Exception:
        return str(val)


def _parse_rows(df, cutoff_date: datetime) -> list[dict]:
    """Normalise yfinance insider_transactions DataFrame rows within date window."""
    import pandas as pd  # noqa: PLC0415

    col = {c.lower().replace(" ", "_"): c for c in df.columns}
    date_col = col.get("start_date") or col.get("date") or col.get("startdate")
    insider_col = col.get("insider") or col.get("name")
    position_col = col.get("position") or col.get("title")
    tx_col = col.get("transaction") or col.get("transaction_type")
    shares_col = col.get("shares")
    value_col = col.get("value")

    rows: list[dict] = []
    for idx, row in df.iterrows():
        raw_date = row.get(date_col) if date_col else None
        if raw_date is None or (isinstance(raw_date, float) and np.isnan(raw_date)):
            raw_date = idx  # fallback to DataFrame index

        date_str = _parse_date_str(raw_date)
        tx_dt: Optional[datetime] = None
        if date_str:
            try:
                tx_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                pass

        if tx_dt is None or tx_dt < cutoff_date:
            continue

        rows.append({
            "date": date_str,
            "tx_datetime": tx_dt,
            "insider_name": str(row.get(insider_col, "") if insider_col else "").strip() or "Unknown",
            "position": str(row.get(position_col, "") if position_col else "").strip(),
            "tx_type": _classify_tx(str(row.get(tx_col, "") if tx_col else "")),
            "shares": _safe_int(row.get(shares_col)) if shares_col else None,
            "value_usd": _safe_float(row.get(value_col)) if value_col else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def _net_ratio(buy_val: float, sell_val: float) -> Optional[float]:
    total = buy_val + sell_val
    return None if total == 0 else round((buy_val - sell_val) / total, 6)


def _officer_sentiment(rows: list[dict]) -> str:
    bv = sum(r["value_usd"] or 0.0 for r in rows if _is_officer(r["position"]) and r["tx_type"] == "Purchase")
    sv = sum(r["value_usd"] or 0.0 for r in rows if _is_officer(r["position"]) and r["tx_type"] == "Sale")
    if bv == 0 and sv == 0:
        return "neutral"
    return "buying" if bv > sv else "selling"


def _cluster_buy(rows: list[dict], now: datetime) -> bool:
    cutoff = now - timedelta(days=_CLUSTER_DAYS)
    buyers = {r["insider_name"] for r in rows if r["tx_type"] == "Purchase" and r["tx_datetime"] >= cutoff}
    return len(buyers) >= 2


def _unusual_size_buy(rows: list[dict]) -> bool:
    vals = [r["value_usd"] for r in rows if r["value_usd"] and r["value_usd"] > 0]
    if not vals:
        return False
    avg = float(np.mean(vals))
    return avg > 0 and any(
        r["value_usd"] and r["value_usd"] > _UNUSUAL_MULT * avg
        for r in rows if r["tx_type"] == "Purchase"
    )


def _recent_momentum(rows: list[dict], now: datetime) -> Optional[bool]:
    recent_cut = now - timedelta(days=_RECENT_DAYS)
    prior_cut = now - timedelta(days=_RECENT_DAYS + _PRIOR_DAYS)

    def _ratio(subset: list[dict]) -> Optional[float]:
        bv = sum(r["value_usd"] or 0.0 for r in subset if r["tx_type"] == "Purchase")
        sv = sum(r["value_usd"] or 0.0 for r in subset if r["tx_type"] == "Sale")
        return _net_ratio(bv, sv)

    r_ratio = _ratio([r for r in rows if r["tx_datetime"] >= recent_cut])
    p_ratio = _ratio([r for r in rows if prior_cut <= r["tx_datetime"] < recent_cut])
    return None if r_ratio is None or p_ratio is None else r_ratio > p_ratio


def _score_and_verdict(
    npr: Optional[float],
    cluster: bool,
    officer: str,
    unusual: bool,
    momentum: Optional[bool],
) -> tuple[float, dict[str, float], str]:
    c: dict[str, float] = {}
    if npr is not None:
        if npr > 0.5:
            c["net_purchase_ratio_strong"] = 3.0
        elif npr > 0.0:
            c["net_purchase_ratio_positive"] = 1.5
        elif npr < -0.5:
            c["net_purchase_ratio_strong_sell"] = -3.0
    c["cluster_buy"] = 2.0 if cluster else 0.0
    c["officer_sentiment"] = 2.0 if officer == "buying" else 0.0
    c["unusual_size_buy"] = 1.0 if unusual else 0.0
    c["recent_momentum"] = 1.5 if momentum else 0.0
    score = float(np.clip(sum(c.values()), 0.0, 10.0))
    if score >= 7.0:
        verdict = "strong buy signal"
    elif score >= 5.0:
        verdict = "moderate buy"
    elif score >= 3.0:
        verdict = "neutral"
    else:
        verdict = "sell signal"
    return score, c, verdict


def _null_signal(ticker: str, days_back: int, as_of: str, warnings: list[str]) -> InsiderSignal:
    _, comps, verdict = _score_and_verdict(None, False, "neutral", False, None)
    return InsiderSignal(
        ticker=ticker, period_days=days_back, total_transactions=0,
        buy_transactions=0, sell_transactions=0,
        total_buy_value_usd=0.0, total_sell_value_usd=0.0,
        net_purchase_ratio=None, cluster_buy=False, officer_sentiment="neutral",
        unusual_size_buy=False, recent_momentum_positive=None,
        signal_score=0.0, signal_verdict=verdict, score_components=comps,
        recent_transactions=[], as_of=as_of, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Entry point: get_insider_signal
# ---------------------------------------------------------------------------


async def get_insider_signal(ticker: str, days_back: int = 180) -> InsiderSignal:
    """Fetch and score insider trading signals for a single ticker."""
    ticker_upper = ticker.upper().strip()
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.utcnow()
    warnings: list[str] = []

    logger.info("Fetching insider signal", ticker=ticker_upper, days_back=days_back)

    try:
        raw = await asyncio.to_thread(_fetch_insider_sync, ticker_upper)
    except Exception as exc:
        warnings.append(f"Fetch failed: {exc}")
        return _null_signal(ticker_upper, days_back, as_of, warnings)

    warnings.extend(raw.get("errors", []))
    df = raw.get("df")
    if df is None:
        warnings.append("No insider transaction data available")
        return _null_signal(ticker_upper, days_back, as_of, warnings)

    try:
        rows = _parse_rows(df, now - timedelta(days=days_back))
    except Exception as exc:
        warnings.append(f"Row parsing failed: {exc}")
        return _null_signal(ticker_upper, days_back, as_of, warnings)

    if not rows:
        warnings.append(f"No insider transactions found in last {days_back} days")
        return _null_signal(ticker_upper, days_back, as_of, warnings)

    buy_rows = [r for r in rows if r["tx_type"] == "Purchase"]
    sell_rows = [r for r in rows if r["tx_type"] == "Sale"]
    total_buy_value = sum(r["value_usd"] or 0.0 for r in buy_rows)
    total_sell_value = sum(r["value_usd"] or 0.0 for r in sell_rows)

    npr = _net_ratio(total_buy_value, total_sell_value)
    cluster = _cluster_buy(rows, now)
    officer = _officer_sentiment(rows)
    unusual = _unusual_size_buy(rows)
    momentum = _recent_momentum(rows, now)

    score, comps, verdict = _score_and_verdict(npr, cluster, officer, unusual, momentum)

    sorted_rows = sorted(rows, key=lambda r: r["tx_datetime"], reverse=True)
    recent_txns = [
        InsiderTransaction(
            insider_name=r["insider_name"], position=r["position"],
            transaction_type=r["tx_type"], shares=r["shares"],
            value_usd=r["value_usd"], date=r["date"] or "",
        )
        for r in sorted_rows[:10]
    ]

    signal = InsiderSignal(
        ticker=ticker_upper, period_days=days_back,
        total_transactions=len(rows), buy_transactions=len(buy_rows),
        sell_transactions=len(sell_rows),
        total_buy_value_usd=round(total_buy_value, 2),
        total_sell_value_usd=round(total_sell_value, 2),
        net_purchase_ratio=npr, cluster_buy=cluster, officer_sentiment=officer,
        unusual_size_buy=unusual, recent_momentum_positive=momentum,
        signal_score=round(score, 4), signal_verdict=verdict,
        score_components=comps, recent_transactions=recent_txns,
        as_of=as_of, warnings=warnings,
    )
    logger.info(
        "Insider signal complete", ticker=ticker_upper, score=score,
        verdict=verdict, cluster_buy=cluster, officer_sentiment=officer,
        transactions=len(rows),
    )
    return signal


# ---------------------------------------------------------------------------
# Entry point: screen_insider_buying
# ---------------------------------------------------------------------------


async def screen_insider_buying(
    tickers: list[str],
    min_signal_score: float = 5.0,
) -> InsiderBuyingScreen:
    """Parallel-screen tickers for insider buying conviction signals."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings: list[str] = []

    logger.info("Screening insider buying", tickers=len(tickers), min_signal_score=min_signal_score)

    raw_results: list[InsiderSignal | BaseException] = await asyncio.gather(
        *[get_insider_signal(t) for t in tickers], return_exceptions=True,
    )

    summaries: list[InsiderSummary] = []
    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            warnings.append(f"Error fetching {ticker}: {res}")
            continue
        warnings.extend(res.warnings)
        summaries.append(InsiderSummary(
            ticker=res.ticker, signal_score=res.signal_score,
            signal_verdict=res.signal_verdict, net_purchase_ratio=res.net_purchase_ratio,
            cluster_buy=res.cluster_buy, officer_sentiment=res.officer_sentiment,
        ))

    filtered = sorted(
        [s for s in summaries if s.signal_score >= min_signal_score],
        key=lambda s: s.signal_score, reverse=True,
    )
    cluster_buys = [s.ticker for s in summaries if s.cluster_buy]
    officer_buyers = [s.ticker for s in summaries if s.officer_sentiment == "buying"]
    top_signals = [s.ticker for s in summaries if s.signal_score >= 7.0]
    all_scores = [s.signal_score for s in summaries]
    avg_signal_score: Optional[float] = (
        round(float(np.mean(all_scores)), 4) if all_scores else None
    )

    screen = InsiderBuyingScreen(
        tickers_screened=len(tickers), min_score_filter=min_signal_score,
        results=filtered, cluster_buys=cluster_buys, officer_buyers=officer_buyers,
        top_signals=top_signals, avg_signal_score=avg_signal_score,
        as_of=as_of, warnings=warnings,
    )
    logger.info(
        "Insider buying screen complete", tickers_screened=len(tickers),
        results=len(filtered), cluster_buys=len(cluster_buys),
        officer_buyers=len(officer_buyers), top_signals=len(top_signals),
        avg_score=avg_signal_score,
    )
    return screen
