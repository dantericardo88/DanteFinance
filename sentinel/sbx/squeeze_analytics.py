"""Advanced short-squeeze signal analytics — Wave-14d.

Signals: Days-to-Cover, SI % float, SI MoM change, borrow cost proxy,
price momentum, options gamma exposure, composite squeeze score (0-10).
"""
from __future__ import annotations

import asyncio
import csv
import io
from datetime import datetime, timedelta, date
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_FINRA_CSV_BASE = "https://cdn.finra.org/equity/regsho/daily"
_UA = "SENTINEL/1.0 research@sentinel.ai"
_TIMEOUT = 15


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ShortSignals(BaseModel):
    model_config = ConfigDict(frozen=True)

    short_interest_shares: int | None
    si_pct_float: float | None
    days_to_cover: float | None
    si_change_mom_pct: float | None
    borrow_cost_proxy_pct: float | None
    borrow_difficulty: str
    float_shares: int | None
    avg_daily_volume_30d: float | None


class PriceMomentum(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_price: float | None
    return_5d_pct: float | None
    return_20d_pct: float | None
    return_60d_pct: float | None
    rsi_14: float | None


class SqueezeAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    short_signals: ShortSignals
    price_momentum: PriceMomentum
    call_put_oi_ratio: float | None
    gamma_squeeze_risk: bool
    squeeze_score: float
    squeeze_verdict: str
    score_components: dict[str, float]
    finra_date: str | None
    as_of: str
    warnings: list[str]


class SqueezeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    squeeze_score: float
    verdict: str
    si_pct_float: float | None
    days_to_cover: float | None
    borrow_difficulty: str


class SqueezeScreen(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers_screened: int
    min_score_filter: float
    results: list[SqueezeSummary]
    highest_squeeze_risk: list[str]
    gamma_squeeze_candidates: list[str]
    avg_si_pct_float: float | None
    as_of: str
    warnings: list[str]


# ── RSI ───────────────────────────────────────────────────────────────────────

def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    return 100.0 - 100.0 / (1.0 + rs)


# ── FINRA CSV fetch ───────────────────────────────────────────────────────────

async def _fetch_finra_short_volume(ticker: str) -> tuple[int | None, str | None]:
    """Walk back up to 5 days to find latest FINRA CNMS CSV; return (short_volume, date_str)."""
    ticker_upper = ticker.upper()
    today = date.today()
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for delta in range(6):
            candidate = today - timedelta(days=delta)
            date_str = candidate.strftime("%Y%m%d")
            url = f"{_FINRA_CSV_BASE}/CNMSshvol{date_str}.txt"
            try:
                resp = await client.get(url, headers={"User-Agent": _UA})
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                text = resp.content.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(text), delimiter="|")
                for row in reader:
                    norm = {k.strip().upper(): v for k, v in row.items()}
                    sym = norm.get("SYMBOL", "").strip().upper()
                    if sym == ticker_upper:
                        short_vol_raw = norm.get("SHORTVOLUME", "")
                        try:
                            return int(short_vol_raw.replace(",", "").strip()), date_str
                        except (ValueError, AttributeError):
                            return None, date_str
            except Exception as exc:
                logger.warning("FINRA CSV fetch error", date=date_str, error=str(exc))
                continue
    return None, None


# ── yfinance sync fetch (runs in thread) ──────────────────────────────────────

def _fetch_equity_data_sync(ticker: str) -> dict[str, Any]:
    """Fetch all equity data needed via yfinance. Run inside asyncio.to_thread."""
    import yfinance as yf  # noqa: PLC0415 — lazy import required

    result: dict[str, Any] = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result["float_shares"] = info.get("floatShares")
        result["shares_short"] = info.get("sharesShort")
        result["shares_short_prior"] = info.get("sharesShortPriorMonth")
        result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")

        # 30-day volume history
        hist30 = t.history(period="30d")
        if hist30 is not None and not hist30.empty and "Volume" in hist30.columns:
            result["avg_vol_30d"] = float(hist30["Volume"].mean())
        else:
            result["avg_vol_30d"] = None

        # 60-day price history for momentum
        hist60 = t.history(period="65d")
        if hist60 is not None and not hist60.empty and "Close" in hist60.columns:
            closes = hist60["Close"].dropna().values
            result["closes"] = closes
        else:
            result["closes"] = None

        # Options: nearest expiry call/put OI
        try:
            exps = t.options
            if exps:
                chain = t.option_chain(exps[0])
                call_oi = int(chain.calls["openInterest"].sum()) if not chain.calls.empty else 0
                put_oi = int(chain.puts["openInterest"].sum()) if not chain.puts.empty else 0
                result["call_oi"] = call_oi
                result["put_oi"] = put_oi
            else:
                result["call_oi"] = None
                result["put_oi"] = None
        except Exception:
            result["call_oi"] = None
            result["put_oi"] = None

    except Exception as exc:
        logger.error("yfinance fetch error", ticker=ticker, error=str(exc))

    return result


# ── Signal helpers ────────────────────────────────────────────────────────────

def _borrow_difficulty(borrow_pct: float) -> str:
    if borrow_pct < 1.0:
        return "easy borrow"
    if borrow_pct < 5.0:
        return "moderate"
    if borrow_pct < 15.0:
        return "hard"
    return "special"


def _pct_return(closes: np.ndarray, lookback: int) -> float | None:
    if closes is None or len(closes) < lookback + 1:
        return None
    base = closes[-(lookback + 1)]
    end = closes[-1]
    if base == 0:
        return None
    return round((end - base) / base * 100, 4)


def _compute_score(
    dtc: float | None,
    si_pct: float | None,
    si_mom: float | None,
    ret_20d: float | None,
    borrow_diff: str,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}

    # DTC score (0-8)
    if dtc is None:
        dtc_score = 0.0
    elif dtc >= 20:
        dtc_score = 8.0
    elif dtc >= 10:
        dtc_score = 6.0
    elif dtc >= 5:
        dtc_score = 4.0
    elif dtc >= 2:
        dtc_score = 2.0
    else:
        dtc_score = 0.0
    components["dtc_score"] = dtc_score

    # SI float score (0-4)
    if si_pct is None:
        si_score = 0.0
    elif si_pct >= 20:
        si_score = 4.0
    elif si_pct >= 15:
        si_score = 3.0
    elif si_pct >= 10:
        si_score = 2.0
    elif si_pct >= 5:
        si_score = 1.0
    else:
        si_score = 0.0
    components["si_float_score"] = si_score

    # Short momentum: +1 increasing, -1 decreasing
    mom_score = 0.0
    if si_mom is not None:
        mom_score = 1.0 if si_mom > 0 else (-1.0 if si_mom < 0 else 0.0)
    components["short_momentum_score"] = mom_score

    # Price momentum: +1 if up 20d (catalyst), -1 if down
    price_score = 0.0
    if ret_20d is not None:
        price_score = 1.0 if ret_20d > 0 else -1.0
    components["price_momentum_score"] = price_score

    # Borrow difficulty: +1 if hard or special
    borrow_score = 1.0 if borrow_diff in ("hard", "special") else 0.0
    components["borrow_difficulty_score"] = borrow_score

    total = dtc_score + si_score + mom_score + price_score + borrow_score
    clamped = round(max(0.0, min(10.0, total)), 2)
    return clamped, components


def _verdict(score: float) -> str:
    if score >= 8:
        return "high squeeze risk"
    if score >= 6:
        return "elevated squeeze risk"
    if score >= 4:
        return "moderate squeeze risk"
    return "low squeeze risk"


def _gamma_squeeze_risk(
    ret_20d: float | None,
    si_pct: float | None,
    call_put_ratio: float | None,
) -> bool:
    # Elevated gamma risk: price up + high SI float, or heavily call-skewed options
    if ret_20d is not None and si_pct is not None:
        if ret_20d > 0 and si_pct > 15:
            return True
    if call_put_ratio is not None and call_put_ratio > 1.5:
        return True
    return False


# ── Core analytics builder ────────────────────────────────────────────────────

async def get_squeeze_analytics(ticker: str) -> SqueezeAnalytics:
    """Compute full short-squeeze analytics for a single ticker."""
    ticker = ticker.upper().strip()
    warnings: list[str] = []
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Parallel: FINRA CSV + yfinance
    finra_task = asyncio.create_task(_fetch_finra_short_volume(ticker))
    eq_task = asyncio.create_task(asyncio.to_thread(_fetch_equity_data_sync, ticker))
    (finra_short_vol, finra_date), eq = await asyncio.gather(finra_task, eq_task)

    float_shares: int | None = eq.get("float_shares")
    shares_short: int | None = eq.get("shares_short")
    shares_short_prior: int | None = eq.get("shares_short_prior")
    avg_vol_30d: float | None = eq.get("avg_vol_30d")
    closes: np.ndarray | None = eq.get("closes")
    current_price: float | None = eq.get("current_price")
    call_oi: int | None = eq.get("call_oi")
    put_oi: int | None = eq.get("put_oi")

    # Short interest: prefer yfinance sharesShort; FINRA CSV gives short volume (proxy)
    short_interest_shares: int | None = shares_short
    if short_interest_shares is None and finra_short_vol is not None:
        short_interest_shares = finra_short_vol
        warnings.append("short_interest_shares sourced from FINRA daily short volume (not total SI)")

    # SI % float
    si_pct_float: float | None = None
    if short_interest_shares and float_shares and float_shares > 0:
        si_pct_float = round(short_interest_shares / float_shares * 100, 4)

    # Days-to-cover
    dtc: float | None = None
    if short_interest_shares and avg_vol_30d and avg_vol_30d > 0:
        dtc = round(short_interest_shares / avg_vol_30d, 4)
    elif dtc is None:
        warnings.append("days_to_cover unavailable: missing short_interest or volume")

    # SI change MoM
    si_change_mom: float | None = None
    if shares_short and shares_short_prior and shares_short_prior > 0:
        si_change_mom = round((shares_short - shares_short_prior) / shares_short_prior * 100, 4)
    elif shares_short is None:
        warnings.append("si_change_mom unavailable: sharesShort not returned by yfinance")

    # Borrow cost proxy
    borrow_proxy: float | None = None
    borrow_diff = "easy borrow"
    if si_pct_float is not None:
        borrow_proxy = round(max(0.3, si_pct_float * 0.15), 4)
        borrow_diff = _borrow_difficulty(borrow_proxy)
    else:
        borrow_proxy = 0.3
        warnings.append("borrow_cost_proxy defaulted to 0.3%: si_pct_float unavailable")

    # Price momentum
    ret_5d = _pct_return(closes, 5) if closes is not None else None
    ret_20d = _pct_return(closes, 20) if closes is not None else None
    ret_60d = _pct_return(closes, 60) if closes is not None else None

    if closes is None:
        warnings.append("price history unavailable: momentum signals degraded")
        current_price = current_price  # keep as-is

    rsi_14: float | None = None
    if closes is not None and len(closes) >= 16:
        try:
            rsi_14 = round(_compute_rsi(closes), 4)
        except Exception:
            warnings.append("RSI computation failed")

    # Options call/put OI ratio
    call_put_ratio: float | None = None
    if call_oi is not None and put_oi is not None and put_oi > 0:
        call_put_ratio = round(call_oi / put_oi, 4)
    elif call_oi is None:
        warnings.append("options OI unavailable: no listed options or fetch failed")

    # Gamma squeeze risk
    gamma_risk = _gamma_squeeze_risk(ret_20d, si_pct_float, call_put_ratio)

    # Also flag "bears trapped" signal in warnings
    if ret_20d is not None and dtc is not None:
        if ret_20d < -5 and dtc > 5:
            warnings.append("squeeze_potential: bears trapped — price down >5% with DTC>5")

    # Composite score
    score, components = _compute_score(dtc, si_pct_float, si_change_mom, ret_20d, borrow_diff)
    verdict = _verdict(score)

    logger.info(
        "squeeze_analytics computed",
        ticker=ticker,
        score=score,
        verdict=verdict,
        dtc=dtc,
        si_pct_float=si_pct_float,
        gamma_risk=gamma_risk,
    )

    return SqueezeAnalytics(
        ticker=ticker,
        short_signals=ShortSignals(
            short_interest_shares=short_interest_shares,
            si_pct_float=si_pct_float,
            days_to_cover=dtc,
            si_change_mom_pct=si_change_mom,
            borrow_cost_proxy_pct=borrow_proxy,
            borrow_difficulty=borrow_diff,
            float_shares=float_shares,
            avg_daily_volume_30d=avg_vol_30d,
        ),
        price_momentum=PriceMomentum(
            current_price=current_price,
            return_5d_pct=ret_5d,
            return_20d_pct=ret_20d,
            return_60d_pct=ret_60d,
            rsi_14=rsi_14,
        ),
        call_put_oi_ratio=call_put_ratio,
        gamma_squeeze_risk=gamma_risk,
        squeeze_score=score,
        squeeze_verdict=verdict,
        score_components=components,
        finra_date=finra_date,
        as_of=as_of,
        warnings=warnings,
    )


# ── Screener ──────────────────────────────────────────────────────────────────

async def screen_squeeze_candidates(
    tickers: list[str],
    min_squeeze_score: float = 5.0,
) -> SqueezeScreen:
    """Screen a list of tickers for squeeze candidates; filter by min_squeeze_score."""
    as_of = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    all_warnings: list[str] = []

    tasks = [get_squeeze_analytics(t) for t in tickers]
    raw_results: list[SqueezeAnalytics | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    summaries: list[SqueezeSummary] = []
    gamma_candidates: list[str] = []
    si_pct_values: list[float] = []

    for ticker, res in zip(tickers, raw_results):
        if isinstance(res, BaseException):
            all_warnings.append(f"{ticker}: fetch failed — {res}")
            logger.error("screen ticker error", ticker=ticker, error=str(res))
            continue

        if res.short_signals.si_pct_float is not None:
            si_pct_values.append(res.short_signals.si_pct_float)

        if res.gamma_squeeze_risk:
            gamma_candidates.append(res.ticker)

        if res.squeeze_score >= min_squeeze_score:
            summaries.append(SqueezeSummary(
                ticker=res.ticker,
                squeeze_score=res.squeeze_score,
                verdict=res.squeeze_verdict,
                si_pct_float=res.short_signals.si_pct_float,
                days_to_cover=res.short_signals.days_to_cover,
                borrow_difficulty=res.short_signals.borrow_difficulty,
            ))

        all_warnings.extend(res.warnings)

    summaries.sort(key=lambda s: s.squeeze_score, reverse=True)
    top5 = [s.ticker for s in summaries[:5]]

    avg_si = round(sum(si_pct_values) / len(si_pct_values), 4) if si_pct_values else None

    logger.info(
        "screen_squeeze_candidates complete",
        screened=len(tickers),
        passed_filter=len(summaries),
        min_score=min_squeeze_score,
        gamma_candidates=len(gamma_candidates),
    )

    return SqueezeScreen(
        tickers_screened=len(tickers),
        min_score_filter=min_squeeze_score,
        results=summaries,
        highest_squeeze_risk=top5,
        gamma_squeeze_candidates=gamma_candidates,
        avg_si_pct_float=avg_si,
        as_of=as_of,
        warnings=list(dict.fromkeys(all_warnings)),  # deduplicate, preserve order
    )
