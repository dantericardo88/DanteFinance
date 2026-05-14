"""
ETF analytics — Dimension #56.

Holdings, factor exposure, estimated flows, and peer comparison via yfinance.
Score target: SENTINEL 5, Bloomberg 9. Dim 56: 0 → 5.
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone

import yfinance as yf
from pydantic import BaseModel, Field

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_RF           = 0.05   # risk-free rate proxy
_TD           = 252    # trading days per year
_SPY          = "SPY"
_TOP_N        = 10


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ETFHolding(BaseModel):
    symbol: str
    name: str | None = None
    weight: float
    sector: str | None = None
    market_value: float | None = None


class ETFFactorExposure(BaseModel):
    market_beta: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    quality_factor: float
    volatility_factor: float


class ETFProfile(BaseModel):
    ticker: str
    name: str | None = None
    aum_billions: float | None = None
    expense_ratio: float | None = None
    nav: float | None = None
    premium_discount_pct: float | None = None
    holdings_count: int = 0
    top_holdings: list[ETFHolding] = Field(default_factory=list)
    sector_weights: dict[str, float] = Field(default_factory=dict)
    factor_exposure: ETFFactorExposure
    ytd_return: float | None = None
    one_year_return: float | None = None
    three_year_return: float | None = None
    tracking_error: float | None = None
    sharpe_ratio: float | None = None
    estimated_flow_30d: float | None = None
    as_of: str
    data_source: str
    warnings: list[str] = Field(default_factory=list)


class ETFComparisonRow(BaseModel):
    ticker: str
    name: str | None = None
    aum_billions: float | None = None
    expense_ratio: float | None = None
    ytd_return: float | None = None
    one_year_return: float | None = None
    sharpe_ratio: float | None = None
    holdings_count: int = 0
    top_sector: str | None = None


class ETFComparison(BaseModel):
    tickers: list[str]
    rows: list[ETFComparisonRow]
    best_ytd: str | None = None
    best_sharpe: str | None = None
    lowest_cost: str | None = None
    as_of: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val: object) -> float | None:
    """Safe float conversion; returns None for NaN/inf/non-numeric."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _fetch_raw(ticker: str) -> dict:
    tk, spy_tk = yf.Ticker(ticker), yf.Ticker(_SPY)
    info: dict = {}
    try:
        info = tk.info or {}
    except Exception as e:
        logger.warning("info failed %s: %s", ticker, e)

    hist_3y = hist_1mo = fund_data = spy_hist = None
    try:
        hist_3y = tk.history(period="3y", auto_adjust=True)
    except Exception as e:
        logger.warning("3y history failed %s: %s", ticker, e)
    try:
        hist_1mo = tk.history(period="1mo", auto_adjust=True)
    except Exception as e:
        logger.warning("1mo history failed %s: %s", ticker, e)
    try:
        fund_data = tk.funds_data
    except Exception as e:
        logger.debug("funds_data unavailable %s: %s", ticker, e)
    try:
        spy_hist = spy_tk.history(period="3y", auto_adjust=True)
    except Exception as e:
        logger.warning("SPY history failed: %s", e)

    return {"info": info, "hist_3y": hist_3y, "hist_1mo": hist_1mo,
            "fund_data": fund_data, "spy_hist": spy_hist}


def _parse_holdings(raw: dict, w: list[str]) -> tuple[list[ETFHolding], int]:
    fd, info, holdings = raw["fund_data"], raw["info"], []
    if fd is not None:
        try:
            th = fd.top_holdings
            if th is not None and not th.empty:
                for sym, row in th.iterrows():
                    wt = _sf(row.get("holdingPercent", row.get("Holding Percent")))
                    if wt is None:
                        try:
                            wt = _sf(list(row)[0])
                        except Exception:
                            wt = None
                    holdings.append(ETFHolding(
                        symbol=str(sym),
                        name=str(row.get("holdingName", row.get("Holding Name", ""))) or None,
                        weight=wt or 0.0,
                    ))
        except Exception as e:
            w.append(f"fund_data.top_holdings parse failed: {e}")

    if not holdings:
        try:
            for h in (info.get("holdings") or []):
                holdings.append(ETFHolding(
                    symbol=h.get("symbol", ""),
                    name=h.get("holdingName") or None,
                    weight=_sf(h.get("holdingPercent", 0)) or 0.0,
                ))
        except Exception as e:
            w.append(f"info holdings parse failed: {e}")

    if not holdings:
        w.append("No holdings data available")

    count = len(holdings)
    try:
        count = int(info.get("totalHoldings") or info.get("holdingsCount") or count)
    except (TypeError, ValueError):
        pass

    return sorted(holdings, key=lambda h: h.weight, reverse=True)[:_TOP_N], count


def _parse_sectors(raw: dict, w: list[str]) -> dict[str, float]:
    fd, info, sw = raw["fund_data"], raw["info"], {}
    if fd is not None:
        try:
            for sector, weight in (fd.sector_weightings or {}).items():
                v = _sf(weight)
                if v is not None:
                    sw[str(sector)] = v
        except Exception as e:
            w.append(f"fund_data.sector_weightings parse failed: {e}")
    if not sw:
        try:
            for d in (info.get("sectorWeightings") or []):
                if isinstance(d, dict):
                    for sector, weight in d.items():
                        v = _sf(weight)
                        if v is not None:
                            sw[sector] = v
        except Exception as e:
            w.append(f"info sectorWeightings parse failed: {e}")
    return sw


def _calc_returns(hist_3y, w: list[str]) -> tuple[float | None, float | None, float | None]:
    if hist_3y is None or hist_3y.empty:
        w.append("No price history — returns unavailable")
        return None, None, None
    close = hist_3y["Close"].dropna()
    if close.empty:
        return None, None, None
    now = _sf(close.iloc[-1])
    if now is None:
        return None, None, None

    ytd = one_yr = three_yr = None
    try:
        today = datetime.now(tz=timezone.utc)
        jan1  = datetime(today.year, 1, 1, tzinfo=timezone.utc)
        sl = close[close.index >= jan1]
        if sl.empty:
            sl = close[close.index.year == today.year]
        if not sl.empty:
            p = _sf(sl.iloc[0])
            if p:
                ytd = (now / p) - 1
    except Exception as e:
        w.append(f"YTD calc failed: {e}")
    try:
        if len(close) >= _TD:
            p = _sf(close.iloc[-_TD])
            if p:
                one_yr = (now / p) - 1
    except Exception as e:
        w.append(f"1Y return calc failed: {e}")
    try:
        if len(close) >= _TD * 3:
            p = _sf(close.iloc[0])
            if p:
                three_yr = (now / p) ** (1 / 3) - 1
    except Exception as e:
        w.append(f"3Y return calc failed: {e}")

    return ytd, one_yr, three_yr


def _calc_sharpe(hist_3y, w: list[str]) -> float | None:
    if hist_3y is None or hist_3y.empty:
        return None
    try:
        daily_r = hist_3y["Close"].dropna().pct_change().dropna()
        if len(daily_r) < 30:
            return None
        ann_ret = float(daily_r.mean()) * _TD
        ann_std = float(daily_r.std()) * math.sqrt(_TD)
        return None if ann_std == 0 else (ann_ret - _RF) / ann_std
    except Exception as e:
        w.append(f"Sharpe calc failed: {e}")
        return None


def _estimate_flow(raw: dict, w: list[str]) -> float | None:
    info, hist_1mo = raw["info"], raw["hist_1mo"]
    nav_now = _sf(info.get("navPrice") or info.get("regularMarketPrice"))
    shares  = _sf(info.get("sharesOutstanding"))
    if nav_now is None or shares is None:
        w.append("Cannot estimate 30d flow — missing nav or sharesOutstanding")
        return None
    if hist_1mo is None or hist_1mo.empty:
        return None
    try:
        cl = hist_1mo["Close"].dropna()
        if len(cl) >= 2:
            p0 = _sf(cl.iloc[0])
            if p0:
                return (nav_now * shares - p0 * shares) / 1e6
    except Exception as e:
        w.append(f"30d flow estimate failed: {e}")
    return None


def _calc_factors(raw: dict, w: list[str]) -> ETFFactorExposure:
    info, hist_3y, spy_hist = raw["info"], raw["hist_3y"], raw["spy_hist"]

    market_beta = 1.0
    try:
        if hist_3y is not None and spy_hist is not None:
            er = hist_3y["Close"].dropna().pct_change().dropna()
            sr = spy_hist["Close"].dropna().pct_change().dropna()
            common = er.index.intersection(sr.index)
            er2, sr2 = er.loc[common].iloc[-60:], sr.loc[common].iloc[-60:]
            if len(er2) >= 20:
                var = float(sr2.var())
                if var:
                    market_beta = float(er2.cov(sr2)) / var
    except Exception as e:
        w.append(f"Beta calc failed: {e}")

    size_factor = 0.0
    try:
        avg_mcap = _sf(info.get("averageMarketCap") or info.get("medianMarketCap"))
        if avg_mcap is None:
            aum = _sf(info.get("totalAssets"))
            if aum is not None:
                avg_mcap = aum / max(1, int(info.get("totalHoldings", 100)))
        if avg_mcap is not None:
            size_factor = max(-1.0, min(1.0, -math.log10(max(avg_mcap, 1e6) / 1e10) / 2))
    except Exception as e:
        w.append(f"Size factor failed: {e}")

    value_factor = 0.0
    try:
        pb = _sf(info.get("priceToBook"))
        if pb is not None:
            value_factor = max(-1.0, min(1.0, -(pb - 3.0) / 3.0))
    except Exception as e:
        w.append(f"Value factor failed: {e}")

    momentum_factor = 0.0
    try:
        if hist_3y is not None and not hist_3y.empty:
            cl = hist_3y["Close"].dropna()
            if len(cl) >= _TD:
                p1m, p12m = float(cl.iloc[-21]), float(cl.iloc[-_TD])
                if p12m:
                    momentum_factor = max(-1.0, min(1.0, (p1m / p12m - 1) / 0.30))
    except Exception as e:
        w.append(f"Momentum factor failed: {e}")

    quality_factor = 0.0
    try:
        te = _sf(info.get("trailingEps"))
        fe = _sf(info.get("forwardEps"))
        if te is not None and fe is not None and fe != 0:
            quality_factor = max(-1.0, min(1.0, (fe - te) / abs(fe)))
        elif te is not None:
            quality_factor = 0.3 if te > 0 else -0.3
    except Exception as e:
        w.append(f"Quality factor failed: {e}")

    try:
        volatility_factor = max(-1.0, min(1.0, (1.0 - market_beta) / 0.7))
    except Exception:
        volatility_factor = 0.0

    return ETFFactorExposure(
        market_beta=round(market_beta, 4),
        size_factor=round(size_factor, 4),
        value_factor=round(value_factor, 4),
        momentum_factor=round(momentum_factor, 4),
        quality_factor=round(quality_factor, 4),
        volatility_factor=round(volatility_factor, 4),
    )


def _build_profile(ticker: str, raw: dict) -> ETFProfile:
    ticker = ticker.upper()
    info   = raw["info"]
    warns: list[str] = []

    name = info.get("longName") or info.get("shortName") or None

    aum_billions: float | None = None
    try:
        ta = _sf(info.get("totalAssets"))
        if ta is not None:
            aum_billions = ta / 1e9
    except Exception as e:
        warns.append(f"AUM parse failed: {e}")

    expense_ratio = _sf(info.get("annualReportExpenseRatio")) or _sf(info.get("expenseRatio"))

    nav   = _sf(info.get("navPrice"))
    price = _sf(info.get("regularMarketPrice"))
    prem  = ((price - nav) / nav * 100) if (nav and price) else None

    top_holdings, holdings_count = _parse_holdings(raw, warns)
    sector_weights               = _parse_sectors(raw, warns)
    ytd, one_yr, three_yr        = _calc_returns(raw["hist_3y"], warns)
    sharpe                       = _calc_sharpe(raw["hist_3y"], warns)
    flow_30d                     = _estimate_flow(raw, warns)
    factors                      = _calc_factors(raw, warns)

    return ETFProfile(
        ticker=ticker,
        name=name,
        aum_billions=round(aum_billions, 3) if aum_billions is not None else None,
        expense_ratio=expense_ratio,
        nav=nav,
        premium_discount_pct=round(prem, 4) if prem is not None else None,
        holdings_count=holdings_count,
        top_holdings=top_holdings,
        sector_weights=sector_weights,
        factor_exposure=factors,
        ytd_return=round(ytd, 6) if ytd is not None else None,
        one_year_return=round(one_yr, 6) if one_yr is not None else None,
        three_year_return=round(three_yr, 6) if three_yr is not None else None,
        tracking_error=None,
        sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
        estimated_flow_30d=round(flow_30d, 2) if flow_30d is not None else None,
        as_of=date.today().isoformat(),
        data_source="yfinance",
        warnings=warns,
    )


def _to_row(p: ETFProfile) -> ETFComparisonRow:
    top_sector = max(p.sector_weights, key=p.sector_weights.__getitem__) if p.sector_weights else None
    return ETFComparisonRow(
        ticker=p.ticker, name=p.name, aum_billions=p.aum_billions,
        expense_ratio=p.expense_ratio, ytd_return=p.ytd_return,
        one_year_return=p.one_year_return, sharpe_ratio=p.sharpe_ratio,
        holdings_count=p.holdings_count, top_sector=top_sector,
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def get_etf_profile(ticker: str) -> ETFProfile:
    """Return a full ETFProfile for *ticker*. I/O is off-thread."""
    logger.info("Fetching ETF profile for %s", ticker.upper())
    raw = await asyncio.to_thread(_fetch_raw, ticker)
    return _build_profile(ticker, raw)


async def compare_etfs(tickers: list[str]) -> ETFComparison:
    """Fetch profiles for all *tickers* in parallel and return ETFComparison."""
    if not tickers:
        return ETFComparison(tickers=[], rows=[], as_of=date.today().isoformat(),
                             warnings=["No tickers provided"])

    logger.info("Comparing ETFs: %s", tickers)
    results = await asyncio.gather(*[get_etf_profile(t) for t in tickers],
                                   return_exceptions=True)

    rows: list[ETFComparisonRow] = []
    warns: list[str] = []
    for ticker, res in zip(tickers, results):
        if isinstance(res, Exception):
            warns.append(f"{ticker}: fetch failed — {res}")
        else:
            rows.append(_to_row(res))
            warns.extend(res.warnings)

    def _best(lst: list[ETFComparisonRow], key, fn) -> str | None:
        valid = [r for r in lst if getattr(r, key) is not None]
        return fn(valid, key=lambda r: getattr(r, key)).ticker if valid else None

    return ETFComparison(
        tickers=[t.upper() for t in tickers],
        rows=rows,
        best_ytd=_best(rows, "ytd_return", max),
        best_sharpe=_best(rows, "sharpe_ratio", max),
        lowest_cost=_best(rows, "expense_ratio", min),
        as_of=date.today().isoformat(),
        warnings=warns,
    )
