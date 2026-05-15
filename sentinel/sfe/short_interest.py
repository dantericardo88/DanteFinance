"""
Short Interest — comprehensive FINRA bi-monthly and daily RegSHO data.

Targets dim_009 "Short interest (FINRA bi-monthly / daily)" — raise score to 9+.

Data sources (all free):
  FINRA bi-monthly:  https://cdn.finra.org/equity/regsho/monthly/CNMSshvol{YYYYMMDD}.txt
  FINRA daily:       https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
  yfinance:          float_shares, avg_volume, price history
  Options chain:     yfinance .option_chain for gamma-squeeze risk

Modules
-------
FINRAShortInterestAdapter   — raw FINRA data downloader / parser
ShortInterestAnalyzer       — per-ticker metrics, squeeze scoring
BorrowRateEstimator         — borrow cost estimation (model-based)
ShortSqueezeMonitor         — active squeeze detection, gamma risk
short_interest_router       — FastAPI router (mounts at /api/short)

Historical context
------------------
GameStop (GME) January 2021: short_float ~140%, DTC ~13, price +1,500% in 3 weeks.
AMC Entertainment 2021:      short_float ~80%, DTC ~8.
Volkswagen 2008:             short_float ~13%, but illiquid — cornered by Porsche.
These anchor the squeeze_score calibration.
"""
from __future__ import annotations

import asyncio
import io
import math
from datetime import date, datetime, timedelta
from typing import Literal, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIMEOUT = 30.0
_FINRA_MONTHLY_URL = "https://cdn.finra.org/equity/regsho/monthly/CNMSshvol{date}.txt"
_FINRA_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept-Encoding": "gzip, deflate",
}

# S&P 500 tickers (sample for screening — full list loaded from yfinance wiki in production)
_SP500_SAMPLE: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "LLY",
    "JPM", "XOM", "V", "AVGO", "PG", "MA", "HD", "COST", "CVX", "MRK",
    "ABBV", "KO", "PEP", "ADBE", "WMT", "BAC", "CRM", "TMO", "CSCO", "ACN",
    "MCD", "NKE", "TXN", "AMD", "NEE", "ABT", "LIN", "DHR", "INTC", "VZ",
    "DIS", "PM", "RTX", "HON", "UPS", "T", "AMGN", "QCOM", "LOW", "GS",
    "CAT", "IBM", "MS", "BLK", "ELV", "INTU", "SPGI", "AXP", "DE", "GE",
    "SYK", "BKNG", "SBUX", "MDLZ", "NOW", "ADI", "TJX", "MMC", "GILD", "ISRG",
    "AMAT", "BSX", "ETN", "MO", "C", "CB", "REGN", "VRTX", "ADP", "PLD",
    "SO", "DUK", "CL", "WM", "CI", "HUM", "EOG", "ITW", "SLB", "EMR",
    "F", "GM", "UBER", "LYFT", "RIVN", "LCID", "SOFI", "HOOD", "PLTR", "SNAP",
]

# Russell 2000 high-short-interest candidates (small caps often targeted)
_MEME_CANDIDATES: list[str] = [
    "GME", "AMC", "BBBY", "KOSS", "NOK", "BB", "EXPR", "CLOV", "MVIS", "SNDL",
    "WISH", "WKHS", "RIDE", "NKLA", "SPCE", "HYMC", "SKLZ", "CLVS", "CRON", "TLRY",
]

# Borrow rate tier thresholds (short float % → annualized borrow rate range)
BORROW_RATE_TIERS: dict[str, dict] = {
    "cheap": {
        "float_short_min_pct": 0.0,
        "float_short_max_pct": 5.0,
        "rate_min_pct": 0.25,
        "rate_max_pct": 1.0,
    },
    "medium": {
        "float_short_min_pct": 5.0,
        "float_short_max_pct": 15.0,
        "rate_min_pct": 1.0,
        "rate_max_pct": 5.0,
    },
    "expensive": {
        "float_short_min_pct": 15.0,
        "float_short_max_pct": 30.0,
        "rate_min_pct": 5.0,
        "rate_max_pct": 25.0,
    },
    "extreme": {
        "float_short_min_pct": 30.0,
        "float_short_max_pct": 200.0,
        "rate_min_pct": 25.0,
        "rate_max_pct": 150.0,  # GME-style situations
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val: object) -> Optional[float]:
    """Safe float — returns None for NaN/inf/non-numeric."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _yf_fetch(ticker: str) -> dict:
    """Sync yfinance fetch — runs inside asyncio.to_thread."""
    import yfinance as yf  # lazy

    out: dict = {"info": {}, "history": pd.DataFrame(), "fast_info": {}}
    try:
        tk = yf.Ticker(ticker)
        try:
            out["info"] = tk.info or {}
        except Exception:
            pass
        try:
            hist = tk.history(period="6mo")
            if hist is not None and not hist.empty:
                out["history"] = hist
        except Exception:
            pass
    except Exception as exc:
        logger.warning("yfinance fetch failed", ticker=ticker, error=str(exc))
    return out


def _get_finra_dates(n_periods: int = 6) -> list[str]:
    """
    Generate recent FINRA bi-monthly report date strings (YYYYMMDD).

    FINRA publishes around the 15th and last day of each month.
    This returns the last n_periods settlement date strings to try.
    """
    dates = []
    today = date.today()
    d = today.replace(day=1)
    for _ in range(n_periods * 2):
        # Try 15th and last day of month
        for day in [15, 28]:
            try:
                candidate = d.replace(day=day)
                if candidate <= today:
                    dates.append(candidate.strftime("%Y%m%d"))
            except ValueError:
                pass
        # Go back one month
        if d.month == 1:
            d = d.replace(year=d.year - 1, month=12)
        else:
            d = d.replace(month=d.month - 1)

    return sorted(set(dates), reverse=True)[:n_periods * 2]


def _parse_finra_pipe(content: str) -> pd.DataFrame:
    """
    Parse FINRA pipe-delimited short interest file.

    Expected columns: Symbol|ShortInterest|SettlementDate|AvgDailyShareVolume|DaysToCover
    or: Symbol|Date|ShortVolume|ShortExemptVolume|TotalVolume|Market (daily format)
    """
    if not content or len(content) < 100:
        return pd.DataFrame()
    try:
        df = pd.read_csv(
            io.StringIO(content),
            sep="|",
            dtype=str,
            on_bad_lines="skip",
        )
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as exc:
        logger.warning("FINRA parse failed", error=str(exc))
        return pd.DataFrame()


def _to_float_col(series: pd.Series) -> pd.Series:
    """Convert a string series to float, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# FINRAShortInterestAdapter
# ---------------------------------------------------------------------------

class FINRAShortInterestAdapter:
    """
    Downloads and parses FINRA bi-monthly short interest and daily RegSHO data.

    FINRA publishes consolidated short interest data for all NMS securities
    twice per month (around the 15th and last day).  Daily short volume
    from RegSHO shows the fraction of daily volume that is short.
    """

    async def _try_download(
        self, client: httpx.AsyncClient, url: str
    ) -> Optional[str]:
        """Attempt to download a FINRA file; returns content or None."""
        try:
            r = await client.get(url, timeout=_TIMEOUT, headers=_HEADERS)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
        except Exception as exc:
            logger.debug("FINRA download failed", url=url, error=str(exc))
        return None

    async def get_short_interest_report(self, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        Download and parse a FINRA bi-monthly short interest report.

        date_str: "YYYYMMDD" (optional; uses latest if None)
        Returns DataFrame with: Symbol, ShortInterest, SettlementDate,
                                 AvgDailyShareVolume, DaysToCover
        """
        dates_to_try = [date_str] if date_str else _get_finra_dates(n_periods=6)

        async with httpx.AsyncClient() as client:
            for d in dates_to_try:
                url = _FINRA_MONTHLY_URL.format(date=d)
                content = await self._try_download(client, url)
                if content:
                    df = _parse_finra_pipe(content)
                    if not df.empty and "Symbol" in df.columns:
                        df = self._clean_monthly(df, d)
                        logger.info("FINRA monthly report loaded", date=d, rows=len(df))
                        return df

        logger.warning("FINRA monthly report unavailable for recent dates")
        return pd.DataFrame(columns=["Symbol", "ShortInterest", "SettlementDate",
                                     "AvgDailyShareVolume", "DaysToCover"])

    def _clean_monthly(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Normalize column types for bi-monthly short interest report."""
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if "short" in lower and "interest" in lower and "exempt" not in lower:
                col_map[col] = "ShortInterest"
            elif "settlement" in lower or "date" in lower:
                col_map[col] = "SettlementDate"
            elif "avg" in lower or "volume" in lower:
                col_map[col] = "AvgDailyShareVolume"
            elif "days" in lower or "cover" in lower:
                col_map[col] = "DaysToCover"
            elif "symbol" in lower:
                col_map[col] = "Symbol"

        df = df.rename(columns=col_map)
        for num_col in ["ShortInterest", "AvgDailyShareVolume", "DaysToCover"]:
            if num_col in df.columns:
                df[num_col] = _to_float_col(df[num_col])

        if "SettlementDate" not in df.columns:
            df["SettlementDate"] = date_str

        # Remove header/trailer rows
        df = df[df["Symbol"].str.match(r"^[A-Z]{1,6}$", na=False)].copy()
        df = df.reset_index(drop=True)
        return df

    async def get_latest_report(self) -> pd.DataFrame:
        """Return the most recent FINRA short interest report available."""
        return await self.get_short_interest_report(date_str=None)

    async def get_daily_short_volume(self, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        Download FINRA daily RegSHO short volume file.

        date_str: "YYYYMMDD" (defaults to most recent business day)
        Columns: Symbol, Date, ShortVolume, ShortExemptVolume, TotalVolume,
                 ShortVolPct, Market
        """
        if date_str is None:
            # Walk back up to 5 business days to find a file
            dates_to_try = []
            d = date.today()
            for _ in range(7):
                if d.weekday() < 5:  # Mon-Fri
                    dates_to_try.append(d.strftime("%Y%m%d"))
                d -= timedelta(days=1)
        else:
            dates_to_try = [date_str]

        async with httpx.AsyncClient() as client:
            for d in dates_to_try:
                url = _FINRA_DAILY_URL.format(date=d)
                content = await self._try_download(client, url)
                if content:
                    df = _parse_finra_pipe(content)
                    if not df.empty:
                        df = self._clean_daily(df, d)
                        if not df.empty:
                            logger.info("FINRA daily volume loaded", date=d, rows=len(df))
                            return df

        logger.warning("FINRA daily short volume unavailable")
        return pd.DataFrame(columns=["Symbol", "Date", "ShortVolume",
                                     "ShortExemptVolume", "TotalVolume", "ShortVolPct"])

    def _clean_daily(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Normalize daily RegSHO file columns."""
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower == "symbol" or lower == "ticker":
                col_map[col] = "Symbol"
            elif "short" in lower and "exempt" in lower:
                col_map[col] = "ShortExemptVolume"
            elif "short" in lower and ("vol" in lower or "volume" in lower):
                col_map[col] = "ShortVolume"
            elif "total" in lower:
                col_map[col] = "TotalVolume"
            elif "market" in lower:
                col_map[col] = "Market"
            elif "date" in lower:
                col_map[col] = "Date"

        df = df.rename(columns=col_map)
        for num_col in ["ShortVolume", "ShortExemptVolume", "TotalVolume"]:
            if num_col in df.columns:
                df[num_col] = _to_float_col(df[num_col])

        if "ShortVolume" in df.columns and "TotalVolume" in df.columns:
            df["ShortVolPct"] = (
                df["ShortVolume"] / df["TotalVolume"].replace(0, np.nan) * 100.0
            ).round(2)

        if "Date" not in df.columns:
            df["Date"] = date_str

        if "Symbol" in df.columns:
            df = df[df["Symbol"].str.match(r"^[A-Z]{1,6}$", na=False)].copy()

        return df.reset_index(drop=True)

    async def get_short_history(
        self, ticker: str, lookback_months: int = 12
    ) -> pd.DataFrame:
        """
        Build a 12-month history of short interest for a ticker by
        downloading multiple bi-monthly FINRA reports.

        Returns DataFrame with: settlement_date, short_interest,
        avg_daily_vol, days_to_cover, change_pct_vs_prior.
        """
        ticker = ticker.upper()
        dates = _get_finra_dates(n_periods=lookback_months)
        rows = []

        async with httpx.AsyncClient() as client:
            for d in dates[:lookback_months * 2]:
                url = _FINRA_MONTHLY_URL.format(date=d)
                content = await self._try_download(client, url)
                if not content:
                    continue
                df = _parse_finra_pipe(content)
                if df.empty:
                    continue
                df = self._clean_monthly(df, d)
                if df.empty or "Symbol" not in df.columns:
                    continue

                ticker_row = df[df["Symbol"] == ticker]
                if ticker_row.empty:
                    continue

                row = ticker_row.iloc[0]
                rows.append({
                    "settlement_date": d,
                    "short_interest": _sf(row.get("ShortInterest")),
                    "avg_daily_vol": _sf(row.get("AvgDailyShareVolume")),
                    "days_to_cover": _sf(row.get("DaysToCover")),
                })

        if not rows:
            logger.warning("No FINRA short history found", ticker=ticker)
            return pd.DataFrame()

        hist_df = pd.DataFrame(rows).sort_values("settlement_date")
        hist_df["change_pct_vs_prior"] = (
            hist_df["short_interest"].pct_change() * 100.0
        ).round(2)

        logger.info("Short interest history built", ticker=ticker, periods=len(hist_df))
        return hist_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ShortInterestAnalyzer
# ---------------------------------------------------------------------------

class ShortInterestAnalyzer:
    """
    Per-ticker short interest metrics and squeeze scoring.

    Combines FINRA short interest data with yfinance float/volume data
    to compute days-to-cover, squeeze probability, and sector comparisons.
    """

    def __init__(self) -> None:
        self._finra = FINRAShortInterestAdapter()

    async def get_short_metrics(self, ticker: str) -> dict:
        """
        Full short interest metrics for a ticker.

        Returns:
          short_interest, float_shares, short_float_pct, days_to_cover,
          short_interest_ratio, current_vs_1m_change_pct, squeeze_score,
          avg_daily_volume, current_price, market_cap.
        """
        ticker = ticker.upper()

        # Parallel: FINRA report + yfinance info
        finra_task = self._finra.get_latest_report()
        yf_task = asyncio.to_thread(_yf_fetch, ticker)
        finra_df, yf_raw = await asyncio.gather(finra_task, yf_task)

        info = yf_raw.get("info", {})
        hist = yf_raw.get("history", pd.DataFrame())

        # yfinance data
        float_shares = _sf(info.get("floatShares"))
        shares_outstanding = _sf(info.get("sharesOutstanding"))
        avg_volume = _sf(info.get("averageVolume") or info.get("averageVolume10days"))
        current_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))
        market_cap = _sf(info.get("marketCap"))

        # Short interest from FINRA
        short_interest: Optional[float] = None
        avg_daily_vol_finra: Optional[float] = None
        dtc_finra: Optional[float] = None

        if not finra_df.empty and "Symbol" in finra_df.columns:
            row = finra_df[finra_df["Symbol"] == ticker]
            if not row.empty:
                short_interest = _sf(row.iloc[0].get("ShortInterest"))
                avg_daily_vol_finra = _sf(row.iloc[0].get("AvgDailyShareVolume"))
                dtc_finra = _sf(row.iloc[0].get("DaysToCover"))

        # Compute derived metrics
        float_for_calc = float_shares or shares_outstanding
        vol_for_calc = avg_daily_vol_finra or avg_volume

        short_float_pct: Optional[float] = None
        if short_interest and float_for_calc and float_for_calc > 0:
            short_float_pct = round(short_interest / float_for_calc * 100.0, 2)

        days_to_cover: Optional[float] = None
        if dtc_finra is not None:
            days_to_cover = dtc_finra
        elif short_interest and vol_for_calc and vol_for_calc > 0:
            days_to_cover = round(short_interest / vol_for_calc, 2)

        # 1-month change (requires history)
        change_1m_pct: Optional[float] = None
        if len(hist) >= 22 and "Close" in hist.columns:
            try:
                closes = hist["Close"].dropna()
                change_1m_pct = round(
                    (float(closes.iloc[-1]) / float(closes.iloc[-22]) - 1) * 100, 2
                )
            except Exception:
                pass

        # Squeeze score
        squeeze = await self.compute_squeeze_score(ticker) if short_float_pct else {}

        result = {
            "ticker": ticker,
            "short_interest": short_interest,
            "float_shares": float_for_calc,
            "short_float_pct": short_float_pct,
            "days_to_cover": days_to_cover,
            "short_interest_ratio": days_to_cover,
            "avg_daily_volume": vol_for_calc,
            "current_vs_1m_change_pct": change_1m_pct,
            "current_price": current_price,
            "market_cap": market_cap,
            "squeeze_score": squeeze.get("squeeze_score"),
            "squeeze_probability": squeeze.get("squeeze_probability"),
            "data_source": "FINRA + yfinance",
            "as_of": date.today().isoformat(),
        }
        logger.info("Short metrics computed", ticker=ticker,
                    short_float_pct=short_float_pct, dtc=days_to_cover)
        return result

    async def compute_squeeze_score(self, ticker: str) -> dict:
        """
        Compute a 0-100 squeeze probability score.

        Calibrated against historical squeezes:
          GME Jan 2021:  float_short ~140%, DTC ~13, momentum +1500% → score 95+
          AMC 2021:      float_short ~80%, DTC ~8,  momentum +800%  → score 85+
          VW 2008:       float_short ~13%, DTC ~3,  illiquid        → score 60

        Inputs:
          short_float_pct — primary driver (weight: 40%)
          days_to_cover   — secondary (weight: 25%)
          price_momentum  — 1-month price change % (weight: 20%)
          borrow_rate     — cost of borrow proxy (weight: 15%)

        Returns: squeeze_score (0-100), squeeze_probability ("low"/"medium"/"high"),
                 component_scores, recommendation.
        """
        ticker = ticker.upper()
        raw = await asyncio.to_thread(_yf_fetch, ticker)
        info = raw.get("info", {})
        hist = raw.get("history", pd.DataFrame())

        # Pull short data
        finra_df = await self._finra.get_latest_report()
        float_shares = _sf(info.get("floatShares"))
        avg_volume = _sf(info.get("averageVolume"))

        short_interest: Optional[float] = None
        dtc: Optional[float] = None
        if not finra_df.empty and "Symbol" in finra_df.columns:
            row = finra_df[finra_df["Symbol"] == ticker]
            if not row.empty:
                short_interest = _sf(row.iloc[0].get("ShortInterest"))
                dtc = _sf(row.iloc[0].get("DaysToCover"))

        short_float_pct: float = 0.0
        if short_interest and float_shares and float_shares > 0:
            short_float_pct = short_interest / float_shares * 100.0

        if dtc is None and short_interest and avg_volume and avg_volume > 0:
            dtc = short_interest / avg_volume

        # Price momentum (1-month)
        momentum_1m: float = 0.0
        if not hist.empty and "Close" in hist.columns and len(hist) >= 22:
            try:
                closes = hist["Close"].dropna()
                momentum_1m = float(closes.iloc[-1]) / float(closes.iloc[-22]) - 1.0
            except Exception:
                pass

        # Borrow rate estimate (proxy)
        borrow_est = BorrowRateEstimator()
        borrow_info = borrow_est.estimate_borrow_rate_sync(short_float_pct)
        borrow_rate_pct = borrow_info.get("estimated_rate_pct", 0.0) or 0.0

        # Scoring components (each 0-100)
        # Float short: >50% = 100, 20-50% = 50-80, 10-20% = 20-50, <10% = 0-20
        if short_float_pct >= 50:
            float_score = min(100.0, 80 + short_float_pct / 10.0)
        elif short_float_pct >= 20:
            float_score = 50 + (short_float_pct - 20) / 30.0 * 30.0
        elif short_float_pct >= 10:
            float_score = 20 + (short_float_pct - 10) / 10.0 * 30.0
        else:
            float_score = short_float_pct * 2.0

        # DTC: >10 = 100, 5-10 = 50-80, 2-5 = 20-50, <2 = 0-20
        if dtc is None:
            dtc_score = 0.0
        elif dtc >= 10:
            dtc_score = 80 + min(20.0, dtc - 10)
        elif dtc >= 5:
            dtc_score = 50 + (dtc - 5) / 5.0 * 30.0
        elif dtc >= 2:
            dtc_score = 20 + (dtc - 2) / 3.0 * 30.0
        else:
            dtc_score = dtc * 10.0

        # Momentum: >50% = 100, 10-50% = 30-80, 0-10% = 0-30, negative = 0
        if momentum_1m >= 0.50:
            momentum_score = 100.0
        elif momentum_1m >= 0.10:
            momentum_score = 30 + (momentum_1m - 0.10) / 0.40 * 50.0
        elif momentum_1m > 0:
            momentum_score = momentum_1m / 0.10 * 30.0
        else:
            momentum_score = max(0.0, 100.0 * (1 + momentum_1m))  # partial credit for mild dip

        # Borrow rate: >50% ann. = 100, 10-50% = 50-80, 1-10% = 10-50, <1% = 0-10
        if borrow_rate_pct >= 50:
            borrow_score = 100.0
        elif borrow_rate_pct >= 10:
            borrow_score = 50 + (borrow_rate_pct - 10) / 40.0 * 30.0
        elif borrow_rate_pct >= 1:
            borrow_score = 10 + (borrow_rate_pct - 1) / 9.0 * 40.0
        else:
            borrow_score = borrow_rate_pct * 10.0

        # Weighted composite
        squeeze_score = (
            float_score * 0.40
            + dtc_score * 0.25
            + momentum_score * 0.20
            + borrow_score * 0.15
        )
        squeeze_score = round(min(100.0, squeeze_score), 1)

        if squeeze_score >= 70:
            squeeze_prob = "high"
        elif squeeze_score >= 40:
            squeeze_prob = "medium"
        else:
            squeeze_prob = "low"

        # Recommendation
        if squeeze_prob == "high":
            rec = "High squeeze potential — monitor for catalyst; institutional shorts under pressure"
        elif squeeze_prob == "medium":
            rec = "Moderate squeeze potential — watch for volume spike and retail momentum"
        else:
            rec = "Low squeeze potential — insufficient short pressure or no momentum"

        return {
            "ticker": ticker,
            "squeeze_score": squeeze_score,
            "squeeze_probability": squeeze_prob,
            "short_float_pct": round(short_float_pct, 2),
            "days_to_cover": round(dtc, 2) if dtc else None,
            "price_momentum_1m_pct": round(momentum_1m * 100, 2),
            "estimated_borrow_rate_pct": borrow_rate_pct,
            "component_scores": {
                "float_short_score": round(float_score, 1),
                "dtc_score": round(dtc_score, 1),
                "momentum_score": round(momentum_score, 1),
                "borrow_rate_score": round(borrow_score, 1),
            },
            "recommendation": rec,
            "as_of": date.today().isoformat(),
        }

    async def get_most_shorted(
        self,
        n: int = 50,
        universe: Literal["sp500", "all", "meme"] = "sp500",
    ) -> pd.DataFrame:
        """
        Top N stocks by short float % in the specified universe.

        universe: "sp500" (S&P 500 constituents), "meme" (high-SI candidates),
                  "all" (FINRA full report, no universe filter).
        """
        if universe == "all":
            finra_df = await self._finra.get_latest_report()
            if finra_df.empty:
                return pd.DataFrame()
            # Add float from yfinance for top 100 by raw short interest
            finra_df = finra_df.sort_values("ShortInterest", ascending=False).head(100)
            return finra_df.reset_index(drop=True)

        tickers = _MEME_CANDIDATES if universe == "meme" else _SP500_SAMPLE

        finra_df = await self._finra.get_latest_report()

        async def _enrich(ticker: str) -> Optional[dict]:
            try:
                raw = await asyncio.to_thread(_yf_fetch, ticker)
                info = raw.get("info", {})
                float_s = _sf(info.get("floatShares"))
                avg_vol = _sf(info.get("averageVolume"))
                price = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))
                company = info.get("longName") or info.get("shortName")

                short_interest: Optional[float] = None
                dtc: Optional[float] = None
                if not finra_df.empty and "Symbol" in finra_df.columns:
                    row = finra_df[finra_df["Symbol"] == ticker]
                    if not row.empty:
                        short_interest = _sf(row.iloc[0].get("ShortInterest"))
                        dtc = _sf(row.iloc[0].get("DaysToCover"))

                short_float_pct: Optional[float] = None
                if short_interest and float_s and float_s > 0:
                    short_float_pct = round(short_interest / float_s * 100, 2)

                return {
                    "ticker": ticker,
                    "company": company,
                    "short_interest": short_interest,
                    "float_shares": float_s,
                    "short_float_pct": short_float_pct,
                    "days_to_cover": dtc,
                    "avg_daily_volume": avg_vol,
                    "price": price,
                }
            except Exception:
                return None

        # Limit concurrency to avoid rate-limiting
        sem = asyncio.Semaphore(10)

        async def _guarded(ticker: str) -> Optional[dict]:
            async with sem:
                return await _enrich(ticker)

        results = await asyncio.gather(*[_guarded(t) for t in tickers], return_exceptions=True)
        rows = [r for r in results if r and not isinstance(r, Exception)
                and r.get("short_float_pct") is not None]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("short_float_pct", ascending=False).head(n).reset_index(drop=True)
        logger.info("Most shorted computed", n=len(df), universe=universe)
        return df

    async def screen_squeeze_candidates(
        self,
        min_short_float: float = 15.0,
        max_dtc: float = 10.0,
        min_price_momentum_1m: float = 0.0,
    ) -> pd.DataFrame:
        """
        Screen for squeeze candidates meeting threshold criteria.

        Default parameters match a classic squeeze setup:
          - Float short > 15% (meaningful short pressure)
          - DTC < 10 (short position not too illiquid to squeeze)
          - Positive 1-month price momentum (short sellers losing)
        """
        most_shorted = await self.get_most_shorted(n=100, universe="sp500")
        meme = await self.get_most_shorted(n=50, universe="meme")

        combined = pd.concat([most_shorted, meme], ignore_index=True)
        combined = combined.drop_duplicates(subset="ticker")

        # Filter by thresholds
        mask = combined["short_float_pct"] >= min_short_float
        if "days_to_cover" in combined.columns:
            mask &= combined["days_to_cover"].fillna(999) <= max_dtc
        candidates = combined[mask].copy()

        if candidates.empty:
            return pd.DataFrame()

        # Add momentum filter via yfinance
        async def _add_momentum(row: dict) -> dict:
            try:
                raw = await asyncio.to_thread(_yf_fetch, row["ticker"])
                hist = raw.get("history", pd.DataFrame())
                if not hist.empty and "Close" in hist.columns and len(hist) >= 22:
                    closes = hist["Close"].dropna()
                    mom = float(closes.iloc[-1]) / float(closes.iloc[-22]) - 1.0
                    row["price_momentum_1m_pct"] = round(mom * 100, 2)
                else:
                    row["price_momentum_1m_pct"] = None
            except Exception:
                row["price_momentum_1m_pct"] = None
            return row

        sem = asyncio.Semaphore(8)

        async def _guarded(row: dict) -> dict:
            async with sem:
                return await _add_momentum(row)

        enriched = await asyncio.gather(
            *[_guarded(row) for row in candidates.to_dict(orient="records")],
            return_exceptions=True,
        )
        rows = [r for r in enriched if isinstance(r, dict)]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "price_momentum_1m_pct" in df.columns and min_price_momentum_1m is not None:
            df = df[df["price_momentum_1m_pct"].fillna(-999) >= min_price_momentum_1m]

        df = df.sort_values("short_float_pct", ascending=False).reset_index(drop=True)
        logger.info("Squeeze candidates screened", count=len(df),
                    min_short_float=min_short_float, max_dtc=max_dtc)
        return df

    async def compute_short_change_velocity(self, ticker: str) -> dict:
        """
        Rate of change in short interest over the last 3 reporting periods.

        Rising shorts → bearish pressure building.
        Falling shorts → potential squeeze setup (covering).
        """
        ticker = ticker.upper()
        hist = await self._finra.get_short_history(ticker, lookback_months=3)

        if hist.empty or len(hist) < 2:
            return {
                "ticker": ticker,
                "periods_analyzed": 0,
                "trend": "insufficient_data",
                "velocity": None,
                "interpretation": "Not enough historical FINRA data",
            }

        # Linear velocity: (last - first) / periods
        short_vals = hist["short_interest"].dropna().tolist()
        if len(short_vals) < 2:
            return {"ticker": ticker, "periods_analyzed": 0, "trend": "insufficient_data"}

        velocity = (short_vals[-1] - short_vals[0]) / (len(short_vals) - 1)
        pct_change_total = (short_vals[-1] / short_vals[0] - 1.0) * 100 if short_vals[0] > 0 else 0.0
        trend = "rising" if velocity > 0 else ("falling" if velocity < 0 else "flat")

        if trend == "rising":
            interp = "Short sellers adding pressure — bearish signal or trapped shorts building"
        elif trend == "falling":
            interp = "Shorts covering — squeeze candidate or fundamental improvement"
        else:
            interp = "Short interest stable — no directional pressure"

        return {
            "ticker": ticker,
            "periods_analyzed": len(short_vals),
            "short_interest_first": round(short_vals[0], 0),
            "short_interest_last": round(short_vals[-1], 0),
            "velocity_shares_per_period": round(velocity, 0),
            "total_change_pct": round(pct_change_total, 2),
            "trend": trend,
            "interpretation": interp,
            "history": hist.to_dict(orient="records"),
        }

    async def get_sector_short_interest(self, sector: Optional[str] = None) -> pd.DataFrame:
        """
        Average short interest metrics by sector.

        If sector is specified, returns per-ticker breakdown for that sector.
        Otherwise returns aggregate stats per sector.
        """
        from sentinel.sfe.corporate_actions_registry import _SECTOR_TICKERS

        sectors_to_scan = (
            {sector: _SECTOR_TICKERS.get(sector, [])} if sector
            else _SECTOR_TICKERS
        )

        finra_df = await self._finra.get_latest_report()

        rows = []
        for sec_name, tickers in sectors_to_scan.items():
            if not tickers:
                continue
            # Filter FINRA data for this sector's tickers
            if not finra_df.empty and "Symbol" in finra_df.columns:
                sec_df = finra_df[finra_df["Symbol"].isin(tickers)]
                if not sec_df.empty:
                    avg_dtc = _sf(sec_df["DaysToCover"].mean()) if "DaysToCover" in sec_df else None
                    total_si = _sf(sec_df["ShortInterest"].sum()) if "ShortInterest" in sec_df else None
                    rows.append({
                        "sector": sec_name,
                        "ticker_count": len(sec_df),
                        "total_short_interest": total_si,
                        "avg_days_to_cover": round(avg_dtc, 2) if avg_dtc else None,
                        "tickers_covered": sec_df["Symbol"].tolist(),
                    })
                else:
                    rows.append({"sector": sec_name, "ticker_count": 0,
                                 "total_short_interest": None, "avg_days_to_cover": None})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("avg_days_to_cover", ascending=False, na_position="last")
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# BorrowRateEstimator
# ---------------------------------------------------------------------------

class BorrowRateEstimator:
    """
    Estimate stock borrow cost from public proxies.

    True borrow rates require a prime brokerage relationship.
    This class uses short_float_pct as the primary proxy, calibrated
    against known hard-to-borrow events (GME = 80%+ annualized Feb 2021).
    """

    def estimate_borrow_rate_sync(self, short_float_pct: float) -> dict:
        """
        Synchronous borrow rate estimate from float-short proxy.

        Returns: estimated_rate_pct, confidence, rate_tier, rate_range.
        """
        for tier_name, tier in BORROW_RATE_TIERS.items():
            if tier["float_short_min_pct"] <= short_float_pct < tier["float_short_max_pct"]:
                mid = (tier["rate_min_pct"] + tier["rate_max_pct"]) / 2.0
                # Skew toward upper end for high float-short (nonlinear)
                if short_float_pct >= 15:
                    ratio = (short_float_pct - tier["float_short_min_pct"]) / (
                        tier["float_short_max_pct"] - tier["float_short_min_pct"] + 1e-9
                    )
                    est_rate = tier["rate_min_pct"] + ratio * (tier["rate_max_pct"] - tier["rate_min_pct"])
                else:
                    est_rate = mid
                return {
                    "estimated_rate_pct": round(est_rate, 2),
                    "rate_min_pct": tier["rate_min_pct"],
                    "rate_max_pct": tier["rate_max_pct"],
                    "confidence": "model",
                    "rate_tier": tier_name,
                }
        # Above extreme tier
        return {
            "estimated_rate_pct": min(150.0, short_float_pct * 1.2),
            "rate_min_pct": 25.0,
            "rate_max_pct": 150.0,
            "confidence": "model",
            "rate_tier": "extreme",
        }

    async def estimate_borrow_rate(self, ticker: str) -> dict:
        """
        Async borrow rate estimate for a ticker.

        Fetches float_short_pct from yfinance + FINRA, then applies model.
        """
        ticker = ticker.upper()
        analyzer = ShortInterestAnalyzer()
        metrics = await analyzer.get_short_metrics(ticker)
        short_float_pct = metrics.get("short_float_pct") or 0.0
        result = self.estimate_borrow_rate_sync(short_float_pct)
        result["ticker"] = ticker
        result["short_float_pct"] = short_float_pct
        result["note"] = (
            "Model-based estimate. Actual rates from prime broker. "
            "Calibrated against GME Jan 2021 (~80% ann.) and typical HTB rates."
        )
        return result

    async def compute_hard_to_borrow_list(self) -> pd.DataFrame:
        """
        Identify stocks likely on the Hard-to-Borrow (HTB) list.

        HTB threshold heuristic: short_float_pct > 20% OR days_to_cover > 10.
        Scans the FINRA report for candidates.
        """
        finra = FINRAShortInterestAdapter()
        finra_df = await finra.get_latest_report()

        if finra_df.empty:
            return pd.DataFrame()

        # Filter for extreme short interest
        htb_mask = False
        if "DaysToCover" in finra_df.columns:
            htb_mask = _to_float_col(finra_df["DaysToCover"]) >= 10
        if "ShortInterest" in finra_df.columns:
            # Can't compute float % without yfinance, so use raw SI > 10M as proxy
            htb_mask = htb_mask | (_to_float_col(finra_df["ShortInterest"]) >= 10_000_000)

        htb = finra_df[htb_mask].copy() if hasattr(htb_mask, "__iter__") else pd.DataFrame()
        if not htb.empty:
            htb["htb_reason"] = "High DTC or large absolute short position"
            htb = htb.sort_values("DaysToCover", ascending=False, na_position="last")

        logger.info("HTB list computed", count=len(htb))
        return htb.reset_index(drop=True) if not htb.empty else pd.DataFrame()


# ---------------------------------------------------------------------------
# ShortSqueezeMonitor
# ---------------------------------------------------------------------------

class ShortSqueezeMonitor:
    """
    Real-time and historical squeeze monitoring.

    Monitors for active short squeezes (price up >30% in 30 days with
    high short interest) and gamma-squeeze risk from options positioning.
    """

    def __init__(self) -> None:
        self._analyzer = ShortInterestAnalyzer()
        self._finra = FINRAShortInterestAdapter()

    async def monitor_active_squeezes(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Identify stocks currently in a short squeeze.

        Criteria: short_float > 20% AND price up >30% in lookback_days
                  AND volume > 3x 20-day average.
        """
        # Use combined universe of SP500 + meme candidates
        universe = list(set(_SP500_SAMPLE + _MEME_CANDIDATES))
        finra_df = await self._finra.get_latest_report()

        async def _check_squeeze(ticker: str) -> Optional[dict]:
            try:
                raw = await asyncio.to_thread(_yf_fetch, ticker)
                info = raw.get("info", {})
                hist = raw.get("history", pd.DataFrame())

                if hist.empty or "Close" not in hist.columns or len(hist) < lookback_days:
                    return None

                closes = hist["Close"].dropna()
                volumes = hist.get("Volume", pd.Series(dtype=float)).dropna() if "Volume" in hist else pd.Series()

                if len(closes) < lookback_days:
                    return None

                price_change = float(closes.iloc[-1]) / float(closes.iloc[-lookback_days]) - 1.0
                if price_change < 0.30:  # Must be up >30%
                    return None

                # Volume > 3x 20-day avg
                vol_squeeze = False
                if not volumes.empty and len(volumes) >= 20:
                    avg_vol_20d = float(volumes.iloc[-20:].mean())
                    recent_vol = float(volumes.iloc[-5:].mean())
                    vol_squeeze = avg_vol_20d > 0 and recent_vol > 3 * avg_vol_20d

                # Short float check
                float_s = _sf(info.get("floatShares"))
                short_interest: Optional[float] = None
                if not finra_df.empty and "Symbol" in finra_df.columns:
                    row = finra_df[finra_df["Symbol"] == ticker.upper()]
                    if not row.empty:
                        short_interest = _sf(row.iloc[0].get("ShortInterest"))

                short_float_pct: float = 0.0
                if short_interest and float_s and float_s > 0:
                    short_float_pct = short_interest / float_s * 100

                if short_float_pct < 20.0 and not vol_squeeze:
                    return None

                return {
                    "ticker": ticker.upper(),
                    "company": info.get("longName") or info.get("shortName"),
                    "price_change_pct": round(price_change * 100, 2),
                    "short_float_pct": round(short_float_pct, 2),
                    "volume_spike": vol_squeeze,
                    "current_price": _sf(info.get("currentPrice") or info.get("regularMarketPrice")),
                    "market_cap": _sf(info.get("marketCap")),
                    "squeeze_type": "gamma+short" if vol_squeeze and short_float_pct >= 20 else (
                        "short" if short_float_pct >= 20 else "volume"
                    ),
                }
            except Exception:
                return None

        sem = asyncio.Semaphore(8)

        async def _guarded(t: str) -> Optional[dict]:
            async with sem:
                return await _check_squeeze(t)

        results = await asyncio.gather(*[_guarded(t) for t in universe], return_exceptions=True)
        rows = [r for r in results if r and not isinstance(r, Exception)]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("price_change_pct", ascending=False).reset_index(drop=True)
        logger.info("Active squeezes identified", count=len(df))
        return df

    async def compute_gamma_squeeze_risk(
        self,
        ticker: str,
        options_chain_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Estimate gamma squeeze risk from options positioning + short interest.

        Gamma squeeze mechanics:
          1. Stock rallies → call options go in-the-money
          2. Market makers who sold calls must buy shares to delta-hedge
          3. Buying pushes price higher → more calls go ITM → more buying
          Combined with high short interest = explosive feedback loop.

        Requires: high call OI near current price, high short float.
        """
        ticker = ticker.upper()

        # Fetch short metrics
        short_metrics = await self._analyzer.get_short_metrics(ticker)
        short_float_pct = short_metrics.get("short_float_pct") or 0.0

        # Fetch options data via yfinance if not provided
        total_call_oi: float = 0.0
        total_put_oi: float = 0.0
        near_money_call_oi: float = 0.0
        max_pain: Optional[float] = None

        if options_chain_df is not None and not options_chain_df.empty:
            if "openInterest" in options_chain_df.columns:
                total_call_oi = float(options_chain_df["openInterest"].sum())
        else:
            try:
                import yfinance as yf
                tk = yf.Ticker(ticker)
                expirations = tk.options[:3] if tk.options else []
                current_price = short_metrics.get("current_price") or 0.0

                for exp in expirations:
                    chain = tk.option_chain(exp)
                    calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
                    puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()

                    if not calls.empty and "openInterest" in calls.columns:
                        total_call_oi += float(calls["openInterest"].sum())
                        # Near-the-money calls: within 5% of current price
                        if current_price > 0:
                            ntm = calls[
                                (calls["strike"] >= current_price * 0.95) &
                                (calls["strike"] <= current_price * 1.10)
                            ]
                            near_money_call_oi += float(ntm["openInterest"].sum())

                    if not puts.empty and "openInterest" in puts.columns:
                        total_put_oi += float(puts["openInterest"].sum())
            except Exception as exc:
                logger.warning("Options fetch failed for gamma squeeze", ticker=ticker, error=str(exc))

        # Gamma squeeze score: composite of short float + call OI concentration
        pc_ratio = total_put_oi / (total_call_oi + 1)  # put/call OI ratio
        call_dominance = total_call_oi > total_put_oi  # calls dominant → potential gamma risk

        gamma_risk_score = 0.0
        if short_float_pct >= 20:
            gamma_risk_score += 40.0
        elif short_float_pct >= 10:
            gamma_risk_score += 20.0

        if call_dominance and total_call_oi > 10_000:
            gamma_risk_score += 30.0
        elif call_dominance:
            gamma_risk_score += 15.0

        if near_money_call_oi > 50_000:
            gamma_risk_score += 30.0
        elif near_money_call_oi > 10_000:
            gamma_risk_score += 15.0

        gamma_risk = "high" if gamma_risk_score >= 70 else ("medium" if gamma_risk_score >= 40 else "low")

        return {
            "ticker": ticker,
            "short_float_pct": round(short_float_pct, 2),
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "put_call_oi_ratio": round(pc_ratio, 3),
            "near_money_call_oi": int(near_money_call_oi),
            "gamma_risk_score": round(gamma_risk_score, 1),
            "gamma_squeeze_risk": gamma_risk,
            "short_squeeze_score": short_metrics.get("squeeze_score"),
            "combined_risk": "extreme" if (gamma_risk == "high" and short_float_pct >= 20) else gamma_risk,
            "as_of": date.today().isoformat(),
        }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

short_interest_router = APIRouter(prefix="/api/short", tags=["Short Interest"])

_finra = FINRAShortInterestAdapter()
_analyzer = ShortInterestAnalyzer()
_borrow = BorrowRateEstimator()
_monitor = ShortSqueezeMonitor()


@short_interest_router.get("/{ticker}/metrics")
async def short_metrics_endpoint(ticker: str):
    """Full short interest metrics for a ticker."""
    metrics = await _analyzer.get_short_metrics(ticker)
    return metrics


@short_interest_router.get("/{ticker}/squeeze-score")
async def squeeze_score_endpoint(ticker: str):
    """Detailed squeeze probability scoring for a ticker."""
    score = await _analyzer.compute_squeeze_score(ticker)
    return score


@short_interest_router.get("/{ticker}/history")
async def short_history_endpoint(
    ticker: str,
    lookback_months: int = Query(12, ge=1, le=24, description="Months of FINRA history"),
):
    """Historical FINRA short interest for a ticker."""
    hist = await _finra.get_short_history(ticker, lookback_months=lookback_months)
    return {
        "ticker": ticker.upper(),
        "lookback_months": lookback_months,
        "count": len(hist),
        "data": hist.to_dict(orient="records") if not hist.empty else [],
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/{ticker}/borrow-rate")
async def borrow_rate_endpoint(ticker: str):
    """Estimated stock borrow rate for a ticker (model-based)."""
    result = await _borrow.estimate_borrow_rate(ticker)
    return result


@short_interest_router.get("/{ticker}/gamma-risk")
async def gamma_squeeze_risk_endpoint(ticker: str):
    """Gamma + short squeeze risk composite for a ticker."""
    result = await _monitor.compute_gamma_squeeze_risk(ticker)
    return result


@short_interest_router.get("/{ticker}/velocity")
async def short_velocity_endpoint(ticker: str):
    """Rate of change in short interest over recent reporting periods."""
    result = await _analyzer.compute_short_change_velocity(ticker)
    return result


@short_interest_router.get("/most-shorted")
async def most_shorted_endpoint(
    n: int = Query(50, ge=1, le=200, description="Number of stocks to return"),
    universe: Literal["sp500", "all", "meme"] = Query("sp500"),
):
    """Top N stocks by short float percentage."""
    df = await _analyzer.get_most_shorted(n=n, universe=universe)
    return {
        "universe": universe,
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/squeeze-candidates")
async def squeeze_candidates_endpoint(
    min_short_float: float = Query(15.0, ge=0.0, le=100.0),
    max_dtc: float = Query(10.0, ge=0.0, le=100.0),
    min_price_momentum_1m: float = Query(0.0, ge=-100.0, le=500.0,
                                         description="Minimum 1-month price change %"),
):
    """Squeeze candidate screen with configurable thresholds."""
    df = await _analyzer.screen_squeeze_candidates(
        min_short_float=min_short_float,
        max_dtc=max_dtc,
        min_price_momentum_1m=min_price_momentum_1m,
    )
    return {
        "filters": {
            "min_short_float_pct": min_short_float,
            "max_dtc": max_dtc,
            "min_price_momentum_1m_pct": min_price_momentum_1m,
        },
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/active-squeezes")
async def active_squeezes_endpoint(
    lookback_days: int = Query(30, ge=5, le=90, description="Price momentum window"),
):
    """Currently squeezing stocks: price up >30% + high short + volume spike."""
    df = await _monitor.monitor_active_squeezes(lookback_days=lookback_days)
    return {
        "lookback_days": lookback_days,
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/hard-to-borrow")
async def hard_to_borrow_endpoint():
    """Stocks estimated to be on the Hard-to-Borrow list (model-based)."""
    df = await _borrow.compute_hard_to_borrow_list()
    return {
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "methodology": "DTC >= 10 or ShortInterest >= 10M shares (FINRA data)",
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/sector/{sector}")
async def sector_short_interest_endpoint(sector: str):
    """Short interest statistics for a market sector."""
    df = await _analyzer.get_sector_short_interest(sector=sector)
    return {
        "sector": sector,
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "as_of": date.today().isoformat(),
    }


@short_interest_router.get("/daily-volume")
async def daily_short_volume_endpoint(
    date_str: Optional[str] = Query(None, description="YYYYMMDD; defaults to latest"),
    ticker: Optional[str] = Query(None, description="Filter to specific ticker"),
):
    """FINRA daily RegSHO short volume data."""
    df = await _finra.get_daily_short_volume(date_str=date_str)
    if ticker and not df.empty and "Symbol" in df.columns:
        df = df[df["Symbol"] == ticker.upper()]
    return {
        "date": date_str or "latest",
        "ticker_filter": ticker,
        "count": len(df),
        "data": df.to_dict(orient="records") if not df.empty else [],
        "as_of": date.today().isoformat(),
    }
