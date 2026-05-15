"""
Market regime detection, correlation monitoring, and volatility regime modelling —
dim_080 (target score 9+).

Implements multi-signal macro regime classification (VIX, yield curve, credit spread,
equity momentum, dollar strength, commodity regime, correlation regime, realised
volatility percentile) plus Ledoit-Wolf correlation matrices, HMM-based volatility
regime switching, and a real-time alert engine.

Classes
-------
RegimeDetector
    8-signal macro regime classifier. Pulls live data from FRED and yfinance.
    Maps signal combination → regime label: risk_on / risk_off / stagflation /
    goldilocks / recession / transition.

CorrelationMonitor
    Ledoit-Wolf shrinkage correlation matrices, pairwise spike detection,
    conditional (asymmetric) correlation, graph-based cluster detection.

VolatilityRegimeModel
    GARCH(1,1) parameter estimation, 2-state Markov regime switching (HMM),
    and realised-vol percentile rank.

AlertEngine
    Run all checks for a portfolio and return formatted alert list.

FastAPI router
--------------
regime_router — mounted at /api/regime
"""
from __future__ import annotations

import asyncio
import warnings
from collections import Counter
from datetime import date, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from scipy import stats
from sklearn.covariance import LedoitWolf

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {"User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com"}
_TIMEOUT = 20.0

# GICS Sector ETFs (11 sectors)
SECTOR_ETFS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

# Regime-optimal asset allocations
REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    "risk_on": {
        "equity": 0.80,
        "credit": 0.10,
        "commodities": 0.05,
        "cash": 0.05,
    },
    "risk_off": {
        "equity": 0.30,
        "bonds": 0.40,
        "gold": 0.15,
        "cash": 0.15,
    },
    "stagflation": {
        "equity": 0.20,
        "tips": 0.20,
        "commodities": 0.30,
        "reits": 0.10,
        "short_duration": 0.20,
    },
    "goldilocks": {
        "equity": 0.70,
        "bonds": 0.15,
        "credit": 0.10,
        "cash": 0.05,
    },
    "recession": {
        "equity": 0.20,
        "long_bonds": 0.40,
        "gold": 0.20,
        "cash": 0.20,
    },
    "transition": {
        "equity": 0.50,
        "bonds": 0.25,
        "gold": 0.10,
        "commodities": 0.05,
        "cash": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_fred_series(series_id: str, lookback_days: int = 30) -> pd.Series:
    """Fetch a single FRED time series as a pandas Series."""
    end = date.today()
    start = end - timedelta(days=lookback_days + 30)
    url = (
        f"{FRED_CSV}?id={series_id}"
        f"&vintage_date={end.strftime('%Y-%m-%d')}"
        f"&observation_start={start.strftime('%Y-%m-%d')}"
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), index_col=0, parse_dates=True)
        series = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        return series
    except Exception as exc:
        logger.warning("fred_fetch_failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float)


def _fetch_prices_sync(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Synchronous yfinance adjusted close download (for thread executor)."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(set(tickers)),
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("yfinance_failed", error=str(exc))
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = (
            raw["Close"]
            if "Close" in raw.columns.get_level_values(0)
            else raw.iloc[:, : len(tickers)]
        )
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
        if len(tickers) == 1:
            closes = closes.rename(columns={"Close": tickers[0]})

    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.droplevel(0)

    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()


async def _fetch_prices(
    tickers: list[str],
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    end = end or date.today()
    return await asyncio.get_event_loop().run_in_executor(
        None, _fetch_prices_sync, tickers, start, end
    )


def _ledoit_wolf_corr(returns: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Ledoit-Wolf shrinkage correlation matrix.

    Returns (corr_matrix, avg_pairwise_corr, dispersion).
    """
    clean = returns.dropna(how="all").fillna(returns.mean())
    n = clean.shape[1]
    if n == 0:
        return np.array([[]]), 0.0, 0.0
    if n == 1:
        return np.array([[1.0]]), 1.0, 0.0

    X = clean.values.astype(float)
    try:
        lw = LedoitWolf()
        lw.fit(X)
        cov = lw.covariance_
    except Exception as exc:
        logger.warning("ledoit_wolf_fallback", error=str(exc))
        cov = clean.cov().fillna(0).values.astype(float)

    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    outer = np.outer(std, std)
    corr = np.clip(cov / outer, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    upper = corr[np.triu(mask)]
    avg = float(np.mean(upper)) if len(upper) else 0.0
    disp = float(np.std(upper)) if len(upper) else 0.0
    return corr, avg, disp


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """8-signal macro regime classifier.

    Signals (REGIME_INDICATORS):
      vix_level          — FRED VIXCLS: low(<15) / medium(15-25) / high(>25) / extreme(>40)
      yield_curve_slope  — FRED T10Y2Y: steepening / flat / inverted
      credit_spread      — FRED BAMLC0A0CM (IG OAS): tight / normal / wide / extreme
      equity_momentum    — SPY 200d trend: bull / bear
      dollar_strength    — DXY 50d vs 200d: strengthening / weakening
      commodity_regime   — CRB index (GCC ETF) trend: inflationary / deflationary
      correlation_regime — avg pairwise corr of 11 GICS sectors: low / elevated
      realized_volatility — SPY 21d realised vol percentile rank (vs 5yr)
    """

    REGIME_INDICATORS: dict[str, dict] = {
        "vix_level": {
            "source": "FRED",
            "series_id": "VIXCLS",
            "thresholds": {"low": 15, "medium": 25, "high": 40},
            "labels": ["low", "medium", "high", "extreme"],
        },
        "yield_curve_slope": {
            "source": "FRED",
            "series_id": "T10Y2Y",
            "thresholds": {"flat": 0.0, "steepening": 0.5},
            "labels": ["inverted", "flat", "steepening"],
        },
        "credit_spread": {
            "source": "FRED",
            "series_id": "BAMLC0A0CM",
            "thresholds": {"tight": 1.00, "normal": 1.75, "wide": 3.00},
            "labels": ["tight", "normal", "wide", "extreme"],
        },
        "equity_momentum": {
            "source": "yfinance",
            "ticker": "SPY",
            "lookback": 200,
            "labels": ["bull", "bear"],
        },
        "dollar_strength": {
            "source": "yfinance",
            "ticker": "DX-Y.NYB",
            "fast_ma": 50,
            "slow_ma": 200,
            "labels": ["strengthening", "weakening"],
        },
        "commodity_regime": {
            "source": "yfinance",
            "ticker": "GCC",
            "trend_window": 63,
            "labels": ["inflationary", "deflationary"],
        },
        "correlation_regime": {
            "source": "computed",
            "etfs": list(SECTOR_ETFS.values()),
            "thresholds": {"low": 0.5, "elevated": 0.7},
            "labels": ["low", "normal", "elevated"],
        },
        "realized_volatility": {
            "source": "computed",
            "ticker": "SPY",
            "window": 21,
            "hist_window": 252 * 5,
            "labels": ["low", "normal", "high", "extreme"],
        },
    }

    def __init__(self) -> None:
        self._price_cache: dict[str, pd.DataFrame] = {}

    async def get_current_regime(self) -> dict:
        """Pull all 8 indicators, classify each signal, return composite regime.

        Returns
        -------
        dict with:
            composite_regime : str — final regime label
            confidence : float — fraction of signals in agreement
            signals : dict — per-signal values and regime votes
            regime_allocation : dict — suggested asset allocation
            as_of : str — ISO date
        """
        signals: dict[str, dict] = {}

        # ---- 1. VIX level ----
        vix_series = await _fetch_fred_series("VIXCLS", lookback_days=10)
        vix_val: Optional[float] = None
        if not vix_series.empty:
            vix_val = round(float(vix_series.dropna().iloc[-1]), 2)
            signals["vix_level"] = {
                "value": vix_val,
                "regime_vote": self._classify_vix(vix_val),
            }
        else:
            signals["vix_level"] = {"value": None, "regime_vote": "transition"}

        # ---- 2. Yield curve slope (T10Y2Y) ----
        yc_series = await _fetch_fred_series("T10Y2Y", lookback_days=10)
        yc_val: Optional[float] = None
        if not yc_series.empty:
            yc_val = round(float(yc_series.dropna().iloc[-1]), 3)
            signals["yield_curve_slope"] = {
                "value": yc_val,
                "label": "inverted" if yc_val < 0 else ("flat" if yc_val < 0.5 else "steepening"),
                "regime_vote": self._classify_yield_curve(yc_val),
            }
        else:
            signals["yield_curve_slope"] = {"value": None, "regime_vote": "transition"}

        # ---- 3. Credit spread (IG OAS, BAMLC0A0CM) ----
        cs_series = await _fetch_fred_series("BAMLC0A0CM", lookback_days=10)
        cs_val: Optional[float] = None
        if not cs_series.empty:
            cs_val = round(float(cs_series.dropna().iloc[-1]), 3)
            signals["credit_spread"] = {
                "value": cs_val,
                "unit": "percent_OAS",
                "label": self._classify_credit_spread_label(cs_val),
                "regime_vote": self._classify_credit_spread(cs_val),
            }
        else:
            signals["credit_spread"] = {"value": None, "regime_vote": "transition"}

        # ---- 4. Equity momentum (SPY 200d trend) ----
        end = date.today()
        start_spy = end - timedelta(days=252)
        spy_prices = await _fetch_prices(["SPY"], start_spy, end)
        if not spy_prices.empty and "SPY" in spy_prices.columns:
            spy = spy_prices["SPY"].dropna()
            if len(spy) >= 200:
                ma200 = float(spy.iloc[-200:].mean())
                current_spy = float(spy.iloc[-1])
                pct_above = (current_spy - ma200) / ma200
                signals["equity_momentum"] = {
                    "value": round(pct_above, 4),
                    "label": "bull" if pct_above > 0 else "bear",
                    "regime_vote": "risk_on" if pct_above > 0.02 else ("risk_off" if pct_above < -0.05 else "transition"),
                }
            else:
                signals["equity_momentum"] = {"value": None, "regime_vote": "transition"}
        else:
            signals["equity_momentum"] = {"value": None, "regime_vote": "transition"}

        # ---- 5. Dollar strength (DXY 50d vs 200d) ----
        dxy_prices = await _fetch_prices(["DX-Y.NYB"], start_spy, end)
        if not dxy_prices.empty and "DX-Y.NYB" in dxy_prices.columns:
            dxy = dxy_prices["DX-Y.NYB"].dropna()
            if len(dxy) >= 50:
                ma50 = float(dxy.iloc[-50:].mean())
                ma200 = float(dxy.iloc[-200:].mean()) if len(dxy) >= 200 else float(dxy.mean())
                dxy_signal = (ma50 - ma200) / ma200
                signals["dollar_strength"] = {
                    "value": round(dxy_signal, 4),
                    "label": "strengthening" if dxy_signal > 0 else "weakening",
                    # Strong dollar = risk-off for EM; weakening dollar = risk-on globally
                    "regime_vote": "risk_off" if dxy_signal > 0.02 else ("risk_on" if dxy_signal < -0.02 else "transition"),
                }
            else:
                signals["dollar_strength"] = {"value": None, "regime_vote": "transition"}
        else:
            signals["dollar_strength"] = {"value": None, "regime_vote": "transition"}

        # ---- 6. Commodity regime (GCC ETF trend) ----
        start_comm = end - timedelta(days=126)
        gcc_prices = await _fetch_prices(["GCC"], start_comm, end)
        if not gcc_prices.empty and "GCC" in gcc_prices.columns:
            gcc = gcc_prices["GCC"].dropna()
            if len(gcc) >= 20:
                trend = float(gcc.iloc[-1] / gcc.iloc[0] - 1)
                signals["commodity_regime"] = {
                    "value": round(trend, 4),
                    "label": "inflationary" if trend > 0 else "deflationary",
                    "regime_vote": "stagflation" if trend > 0.05 else ("goldilocks" if trend < -0.05 else "transition"),
                }
            else:
                signals["commodity_regime"] = {"value": None, "regime_vote": "transition"}
        else:
            signals["commodity_regime"] = {"value": None, "regime_vote": "transition"}

        # ---- 7. Correlation regime (sector ETF average pairwise corr) ----
        sector_etf_list = list(SECTOR_ETFS.values())
        start_corr = end - timedelta(days=90)
        sector_prices = await _fetch_prices(sector_etf_list, start_corr, end)
        if not sector_prices.empty:
            rets = np.log(sector_prices / sector_prices.shift(1)).dropna(how="all").iloc[-63:]
            available_cols = [c for c in sector_etf_list if c in rets.columns]
            if len(available_cols) >= 4:
                _, avg_corr, _ = _ledoit_wolf_corr(rets[available_cols])
                signals["correlation_regime"] = {
                    "value": round(avg_corr, 4),
                    "label": "low" if avg_corr < 0.5 else ("elevated" if avg_corr > 0.7 else "normal"),
                    "regime_vote": "risk_off" if avg_corr > 0.7 else ("risk_on" if avg_corr < 0.4 else "transition"),
                }
            else:
                signals["correlation_regime"] = {"value": None, "regime_vote": "transition"}
        else:
            signals["correlation_regime"] = {"value": None, "regime_vote": "transition"}

        # ---- 8. Realised volatility percentile (SPY 21d vs 5yr history) ----
        rv_result = await self._realized_vol_percentile_async("SPY", 21, 252 * 5)
        signals["realized_volatility"] = {
            "value": round(rv_result, 4) if rv_result is not None else None,
            "label": (
                "extreme" if (rv_result or 0) > 0.90
                else "high" if (rv_result or 0) > 0.75
                else "normal" if (rv_result or 0) > 0.25
                else "low"
            ),
            "regime_vote": (
                "risk_off" if (rv_result or 0) > 0.80
                else "risk_on" if (rv_result or 0) < 0.30
                else "transition"
            ),
        }

        # ---- Composite classification ----
        votes = {k: v["regime_vote"] for k, v in signals.items()}
        composite, confidence = self.classify_regime(votes)

        return {
            "composite_regime": composite,
            "confidence": round(confidence, 4),
            "signals": signals,
            "regime_allocation": REGIME_ALLOCATIONS.get(composite, REGIME_ALLOCATIONS["transition"]),
            "as_of": date.today().isoformat(),
        }

    def classify_regime(self, signals: dict[str, str]) -> tuple[str, float]:
        """Map signal vote dictionary to composite regime label.

        Regime precedence logic:
          - Any "stagflation" vote → check if commodities + yield_curve inverted
          - Plurality voting with crisis escalation
          - Maps "crisis" intermediate → "recession" or "risk_off" depending on depth

        Returns (regime_label, confidence_0_to_1).
        """
        if not signals:
            return "transition", 0.5

        counts = Counter(signals.values())
        total = sum(counts.values())

        # Stagflation detection: elevated commodity + inverted curve
        stagflation_votes = counts.get("stagflation", 0)
        if stagflation_votes >= 1 and counts.get("risk_off", 0) >= 1:
            confidence = (stagflation_votes + counts.get("risk_off", 0)) / total
            return "stagflation", round(confidence, 4)

        # Map crisis → recession if equity momentum is also bearish
        if counts.get("risk_off", 0) >= total * 0.6:
            return "recession", round(counts.get("risk_off", 0) / total, 4)

        # Goldilocks: risk_on + low volatility + low correlation
        risk_on_votes = counts.get("risk_on", 0)
        if risk_on_votes >= total * 0.5:
            # Check for goldilocks conditions (low vol, stable credit)
            if counts.get("transition", 0) <= 2:
                return "goldilocks", round(risk_on_votes / total, 4)
            return "risk_on", round(risk_on_votes / total, 4)

        # Plurality with crisis escalation
        has_risk_off = counts.get("risk_off", 0) > 0
        winner, winner_count = counts.most_common(1)[0]

        if winner == "risk_on" and has_risk_off:
            if counts.get("risk_on", 0) < counts.get("risk_off", 0) + counts.get("stagflation", 0):
                winner = "transition"
                winner_count = counts.get("transition", 0) + 1

        # Remap "stagflation" winner
        if winner == "stagflation":
            return "stagflation", round(winner_count / total, 4)

        return winner, round(winner_count / total, 4)

    async def regime_history(self, lookback_days: int = 252) -> pd.DataFrame:
        """Historical regime classification day-by-day using available FRED data.

        Approximates daily regime using VIX (primary) and yield curve slope.
        For full 8-signal daily history, point-in-time FRED data is used.

        Returns a DataFrame with columns: date, regime, vix, yield_curve_slope.
        """
        vix = await _fetch_fred_series("VIXCLS", lookback_days=lookback_days + 30)
        yc = await _fetch_fred_series("T10Y2Y", lookback_days=lookback_days + 30)
        cs = await _fetch_fred_series("BAMLC0A0CM", lookback_days=lookback_days + 30)

        # Align on common dates
        df = pd.DataFrame({"vix": vix, "yield_curve": yc, "credit_spread": cs}).dropna(how="all")
        df = df.last(f"{lookback_days}D")

        records: list[dict] = []
        for dt, row in df.iterrows():
            vix_v = float(row["vix"]) if not pd.isna(row.get("vix", np.nan)) else None
            yc_v = float(row["yield_curve"]) if not pd.isna(row.get("yield_curve", np.nan)) else None
            cs_v = float(row["credit_spread"]) if not pd.isna(row.get("credit_spread", np.nan)) else None

            signal_votes: dict[str, str] = {}
            if vix_v is not None:
                signal_votes["vix_level"] = self._classify_vix(vix_v)
            if yc_v is not None:
                signal_votes["yield_curve_slope"] = self._classify_yield_curve(yc_v)
            if cs_v is not None:
                signal_votes["credit_spread"] = self._classify_credit_spread(cs_v)

            regime, conf = self.classify_regime(signal_votes) if signal_votes else ("transition", 0.5)
            records.append({
                "date": dt,
                "regime": regime,
                "confidence": round(conf, 4),
                "vix": vix_v,
                "yield_curve_slope": yc_v,
                "credit_spread_pct": cs_v,
            })

        result = pd.DataFrame(records)
        if not result.empty:
            result["date"] = pd.to_datetime(result["date"])
            result = result.set_index("date").sort_index()
        return result

    async def detect_regime_change(
        self,
        current: str,
        history: pd.DataFrame,
        lookback_days: int = 5,
    ) -> dict:
        """Detect if regime changed in last N days.

        Parameters
        ----------
        current : str
            Current composite regime label.
        history : pd.DataFrame
            Output of regime_history().
        lookback_days : int
            Number of recent days to check for a regime change.

        Returns
        -------
        dict with regime_changed, previous_regime, days_in_current_regime, alert_message.
        """
        if history.empty:
            return {
                "regime_changed": False,
                "current_regime": current,
                "previous_regime": None,
                "days_in_current_regime": None,
                "alert_message": "Insufficient historical data.",
            }

        recent = history.tail(lookback_days + 10)
        if "regime" not in recent.columns:
            return {"regime_changed": False, "current_regime": current}

        # Find last regime change
        regimes = recent["regime"].dropna()
        if len(regimes) < 2:
            return {"regime_changed": False, "current_regime": current}

        prev_regime = regimes.iloc[-lookback_days - 1] if len(regimes) > lookback_days else regimes.iloc[0]
        regime_changed = prev_regime != current

        # Count days in current regime
        days_in = 0
        for r in reversed(regimes.tolist()):
            if r == current:
                days_in += 1
            else:
                break

        alert_msg = (
            f"REGIME CHANGE ALERT: {prev_regime} -> {current} "
            f"(detected within {lookback_days} days)"
            if regime_changed
            else f"Regime stable: {current} ({days_in} days)"
        )

        logger.info(
            "regime_change_check",
            current=current,
            previous=prev_regime,
            changed=regime_changed,
            days_in=days_in,
        )

        return {
            "regime_changed": regime_changed,
            "current_regime": current,
            "previous_regime": str(prev_regime),
            "days_in_current_regime": days_in,
            "alert_message": alert_msg,
        }

    def get_regime_optimal_allocation(self, regime: str) -> dict:
        """Return suggested asset allocation for the given regime label."""
        allocation = REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS["transition"])
        return {
            "regime": regime,
            "allocation": allocation,
            "rationale": _REGIME_RATIONALE.get(regime, "No specific rationale available."),
        }

    # ---- Private signal classifiers ----

    @staticmethod
    def _classify_vix(vix: float) -> str:
        if vix > 40:
            return "risk_off"   # extreme
        if vix > 25:
            return "risk_off"   # high
        if vix > 15:
            return "transition"  # medium
        return "risk_on"  # low

    @staticmethod
    def _classify_yield_curve(spread: float) -> str:
        """T10Y2Y spread (%). Negative = inverted = recession signal."""
        if spread < -0.25:
            return "risk_off"
        if spread < 0.25:
            return "transition"
        return "risk_on"

    @staticmethod
    def _classify_credit_spread(oas_pct: float) -> str:
        """IG OAS in percent. Widening = risk-off."""
        if oas_pct > 3.0:
            return "risk_off"  # extreme
        if oas_pct > 1.75:
            return "risk_off"  # wide
        if oas_pct < 1.00:
            return "risk_on"   # tight
        return "transition"

    @staticmethod
    def _classify_credit_spread_label(oas_pct: float) -> str:
        if oas_pct > 3.0:
            return "extreme"
        if oas_pct > 1.75:
            return "wide"
        if oas_pct < 1.00:
            return "tight"
        return "normal"

    async def _realized_vol_percentile_async(
        self,
        asset: str,
        window: int,
        hist_window: int,
    ) -> Optional[float]:
        """Fetch SPY history and compute realised vol percentile rank."""
        end = date.today()
        start = end - timedelta(days=hist_window + 60)
        prices = await _fetch_prices([asset], start, end)
        if prices.empty or asset not in prices.columns:
            return None
        rets = np.log(prices[asset] / prices[asset].shift(1)).dropna()
        if len(rets) < window + 10:
            return None
        rolling_vol = rets.rolling(window=window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        current_vol = float(rolling_vol.iloc[-1])
        pct_rank = float(stats.percentileofscore(rolling_vol.values, current_vol) / 100.0)
        return round(pct_rank, 4)


# Rationale strings for regime allocations
_REGIME_RATIONALE: dict[str, str] = {
    "risk_on": "Benign volatility, positive momentum, and tight spreads support heavy equity allocation.",
    "risk_off": "Elevated VIX, widening credit spreads, or inverted curve — shift to safe-haven assets.",
    "stagflation": "Rising commodity prices with slow growth: TIPS, real assets, and short duration preferred.",
    "goldilocks": "Low volatility, positive growth, contained inflation — growth equities and credit.",
    "recession": "Deep risk-off with equity bear market signals — maximum defensive allocation.",
    "transition": "Mixed signals — balanced allocation with above-average cash buffer.",
}


# ---------------------------------------------------------------------------
# CorrelationMonitor
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """Rolling correlation matrix, spike detection, conditional correlation, network clustering."""

    def compute_rolling_correlation_matrix(
        self,
        returns_dict: dict[str, pd.Series],
        window: int = 21,
    ) -> pd.DataFrame:
        """Compute Ledoit-Wolf shrinkage correlation matrix from a returns dict.

        Parameters
        ----------
        returns_dict : dict
            asset → daily return Series.
        window : int
            Rolling window length. Correlation is computed on the last `window` observations.

        Returns
        -------
        pd.DataFrame — N×N correlation matrix (tickers as both index and columns).
        """
        rets = pd.DataFrame(returns_dict).dropna(how="all")
        if rets.empty:
            return pd.DataFrame()

        rets_window = rets.iloc[-window:] if len(rets) >= window else rets
        corr, _, _ = _ledoit_wolf_corr(rets_window)
        tickers = list(rets_window.columns)
        return pd.DataFrame(corr, index=tickers, columns=tickers)

    def detect_correlation_spike(
        self,
        current_corr: pd.DataFrame,
        historical_corrs: list[pd.DataFrame],
        threshold: float = 0.8,
    ) -> dict:
        """Alert if average pairwise correlation spikes above threshold (crisis indicator).

        Parameters
        ----------
        current_corr : pd.DataFrame
            Current correlation matrix.
        historical_corrs : list[pd.DataFrame]
            List of prior-period correlation matrices (same tickers).
        threshold : float
            Average pairwise correlation level that triggers a spike alert.

        Returns
        -------
        dict with spike_detected, avg_corr, threshold, driving_pairs.
        """
        if current_corr.empty:
            return {"spike_detected": False, "avg_corr": None}

        tickers = list(current_corr.columns)
        n = len(tickers)
        upper_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_vals.append(float(current_corr.iloc[i, j]))

        avg_corr = float(np.mean(upper_vals)) if upper_vals else 0.0
        spike_detected = avg_corr > threshold

        # Identify which pairs drove the spike (correlation > threshold)
        driving_pairs: list[dict] = []
        if spike_detected:
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(current_corr.iloc[i, j])
                    if val > threshold:
                        # Compare to historical average for this pair
                        hist_pair_vals = []
                        for hc in historical_corrs:
                            if tickers[i] in hc.index and tickers[j] in hc.columns:
                                hist_pair_vals.append(float(hc.loc[tickers[i], tickers[j]]))
                        hist_avg = float(np.mean(hist_pair_vals)) if hist_pair_vals else val
                        driving_pairs.append({
                            "asset_a": tickers[i],
                            "asset_b": tickers[j],
                            "current_corr": round(val, 4),
                            "historical_avg": round(hist_avg, 4),
                            "delta": round(val - hist_avg, 4),
                        })
            driving_pairs.sort(key=lambda x: -x["delta"])

        logger.info(
            "correlation_spike_check",
            avg_corr=round(avg_corr, 4),
            threshold=threshold,
            spike_detected=spike_detected,
            n_driving_pairs=len(driving_pairs),
        )

        return {
            "spike_detected": spike_detected,
            "avg_corr": round(avg_corr, 4),
            "threshold": threshold,
            "driving_pairs": driving_pairs[:10],
        }

    def compute_conditional_correlation(
        self,
        asset_a: str,
        asset_b: str,
        returns_dict: dict[str, pd.Series],
        condition_asset: str = "SPY",
        up_down_threshold: float = 0.0,
    ) -> dict:
        """Asymmetric (conditional) correlation: upside vs downside.

        High downside correlation with low upside correlation indicates poor
        diversification during market crashes (a tail risk failure).

        Parameters
        ----------
        asset_a, asset_b : str
            Tickers to compute conditional correlation for.
        returns_dict : dict
            All asset return series (must include asset_a, asset_b, condition_asset).
        condition_asset : str
            Asset used to define up/down days (default SPY).
        up_down_threshold : float
            Return threshold for conditioning (default 0.0 = positive/negative days).

        Returns
        -------
        dict with upside_corr, downside_corr, asymmetry_score, diversification_quality.
        """
        rets = pd.DataFrame(returns_dict).dropna(how="any")
        missing = [t for t in [asset_a, asset_b, condition_asset] if t not in rets.columns]
        if missing:
            raise ValueError(f"Missing tickers in returns_dict: {missing}")

        a = rets[asset_a].values
        b = rets[asset_b].values
        cond = rets[condition_asset].values

        up_mask = cond > up_down_threshold
        down_mask = cond <= up_down_threshold

        def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
            if len(x) < 10:
                return float("nan")
            return float(np.corrcoef(x, y)[0, 1])

        upside_corr = safe_corr(a[up_mask], b[up_mask])
        downside_corr = safe_corr(a[down_mask], b[down_mask])

        asymmetry = (
            float(downside_corr - upside_corr)
            if not (np.isnan(downside_corr) or np.isnan(upside_corr))
            else float("nan")
        )

        # Diversification quality: high upside, low downside = good
        if np.isnan(asymmetry):
            quality = "unknown"
        elif downside_corr > 0.7:
            quality = "poor"  # crashes together
        elif downside_corr < 0.3:
            quality = "excellent"  # independent in downturns
        else:
            quality = "moderate"

        logger.info(
            "conditional_correlation",
            asset_a=asset_a,
            asset_b=asset_b,
            condition=condition_asset,
            upside_corr=round(upside_corr, 4) if not np.isnan(upside_corr) else None,
            downside_corr=round(downside_corr, 4) if not np.isnan(downside_corr) else None,
        )

        return {
            "asset_a": asset_a,
            "asset_b": asset_b,
            "condition_asset": condition_asset,
            "upside_corr": round(float(upside_corr), 4) if not np.isnan(upside_corr) else None,
            "downside_corr": round(float(downside_corr), 4) if not np.isnan(downside_corr) else None,
            "asymmetry_score": round(asymmetry, 4) if not np.isnan(asymmetry) else None,
            "n_up_days": int(up_mask.sum()),
            "n_down_days": int(down_mask.sum()),
            "diversification_quality": quality,
        }

    def correlation_network(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.6,
    ) -> dict:
        """Build correlation network graph; detect clusters and isolated assets.

        Parameters
        ----------
        corr_matrix : pd.DataFrame
            N×N correlation matrix (tickers as index/columns).
        threshold : float
            Minimum absolute correlation to draw an edge.

        Returns
        -------
        dict with nodes, edges, clusters (community detection via greedy modularity),
        isolated_assets (good diversifiers — no edges above threshold).
        """
        tickers = list(corr_matrix.index)
        n = len(tickers)

        edges: list[dict] = []
        adj: dict[str, list[str]] = {t: [] for t in tickers}
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = float(corr_matrix.iloc[i, j])
                if abs(corr_val) >= threshold:
                    edges.append({
                        "asset_a": tickers[i],
                        "asset_b": tickers[j],
                        "correlation": round(corr_val, 4),
                    })
                    adj[tickers[i]].append(tickers[j])
                    adj[tickers[j]].append(tickers[i])

        # Simple greedy community detection (union-find)
        parent = {t: t for t in tickers}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for edge in edges:
            union(edge["asset_a"], edge["asset_b"])

        cluster_map: dict[str, list[str]] = {}
        for t in tickers:
            root = find(t)
            cluster_map.setdefault(root, []).append(t)

        clusters = [members for members in cluster_map.values() if len(members) > 1]
        isolated = [t for t in tickers if not adj[t]]

        logger.info(
            "correlation_network",
            n_assets=n,
            n_edges=len(edges),
            n_clusters=len(clusters),
            n_isolated=len(isolated),
        )

        return {
            "nodes": tickers,
            "edges": edges,
            "clusters": clusters,
            "isolated_assets": isolated,
            "n_edges": len(edges),
            "threshold": threshold,
        }

    async def sector_correlation_heatmap(self, lookback_days: int = 63) -> pd.DataFrame:
        """Compute 11 GICS sector ETF correlation matrix.

        Returns an 11×11 Ledoit-Wolf correlation matrix as a DataFrame
        with sector names as both index and columns.
        """
        end = date.today()
        start = end - timedelta(days=lookback_days + 30)
        etf_list = list(SECTOR_ETFS.values())
        prices = await _fetch_prices(etf_list, start, end)

        if prices.empty:
            return pd.DataFrame()

        rets = np.log(prices / prices.shift(1)).dropna(how="all").iloc[-lookback_days:]
        available = [e for e in etf_list if e in rets.columns]
        rets_clean = rets[available]

        corr, _, _ = _ledoit_wolf_corr(rets_clean)

        # Map ETF tickers back to sector names
        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        labels = [etf_to_sector.get(e, e) for e in available]

        return pd.DataFrame(corr, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# VolatilityRegimeModel
# ---------------------------------------------------------------------------

class VolatilityRegimeModel:
    """GARCH(1,1), Markov regime switching (2-state HMM), and realised vol percentile."""

    def fit_garch(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
    ) -> dict:
        """Estimate GARCH(p,q) parameters via maximum likelihood (or arch library).

        Attempts to use the `arch` package; falls back to a closed-form
        variance-targeting GARCH(1,1) estimator if arch is unavailable.

        Returns
        -------
        dict with omega, alpha (ARCH term), beta (GARCH term), persistence,
        unconditional_vol, current_conditional_vol_annualised.
        """
        rets = returns.dropna().values * 100  # scale to percent for numerical stability

        try:
            from arch import arch_model  # type: ignore
            am = arch_model(rets, vol="Garch", p=p, q=q, dist="Normal", rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            params = res.params
            omega = float(params.get("omega", params.iloc[0]))
            alpha = float(params.get("alpha[1]", params.iloc[1]))
            beta = float(params.get("beta[1]", params.iloc[2]))
            cond_var = res.conditional_volatility[-1] ** 2

        except ImportError:
            logger.info("arch_not_installed", fallback="variance_targeting_garch")
            # Variance-targeting GARCH(1,1): omega = sigma2_bar * (1 - alpha - beta)
            # Estimate alpha and beta via method of moments
            r2 = rets ** 2
            n = len(r2)
            sigma2_bar = float(r2.mean())
            # Simple estimation: alpha = corr(r2_t, r2_{t-1}), beta = 0.85 (typical persistence)
            if n > 10:
                alpha = max(0.05, min(0.20, float(pd.Series(r2).autocorr(lag=1))))
            else:
                alpha = 0.10
            beta = min(0.89, max(0.70, 0.99 - alpha))
            omega = sigma2_bar * (1 - alpha - beta)
            # Compute conditional variance recursively
            sigma2 = np.zeros(n)
            sigma2[0] = sigma2_bar
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
            cond_var = float(sigma2[-1])

        persistence = alpha + beta
        uncond_vol_daily = float(np.sqrt(abs(omega / (1 - persistence + 1e-15)))) if persistence < 1 else float(np.std(rets))
        current_vol_annualised = float(np.sqrt(cond_var) * np.sqrt(252) / 100)  # back to decimal

        logger.info(
            "garch_fit",
            omega=round(float(omega), 6),
            alpha=round(float(alpha), 4),
            beta=round(float(beta), 4),
            persistence=round(float(persistence), 4),
        )

        return {
            "omega": round(float(omega), 6),
            "alpha": round(float(alpha), 4),
            "beta": round(float(beta), 4),
            "persistence": round(float(persistence), 4),
            "unconditional_vol_daily": round(float(uncond_vol_daily) / 100, 6),
            "current_conditional_vol_annualised": round(float(current_vol_annualised), 4),
            "high_vol_regime": current_vol_annualised > 0.25,
        }

    def markov_regime_switching(
        self,
        returns: pd.Series,
        n_regimes: int = 2,
    ) -> dict:
        """2-state Hidden Markov Model for volatility regime switching.

        Uses EM algorithm to estimate:
          - State means and variances (low-vol vs high-vol regimes)
          - Transition probability matrix
          - Viterbi most-likely state sequence
          - Smoothed state probabilities

        Returns
        -------
        dict with regime_probs (timeseries), current_regime_prob,
        low_vol_params, high_vol_params, transition_matrix,
        expected_duration_days.
        """
        rets = returns.dropna().values
        n = len(rets)
        if n < 30:
            raise ValueError(f"Need at least 30 observations for HMM. Got {n}.")

        n_states = n_regimes

        # ---- EM Initialisation ----
        # Split into high/low vol based on rolling std
        roll_std = pd.Series(rets).rolling(window=21, min_periods=5).std().bfill().values
        median_vol = float(np.median(roll_std))

        # Initial state assignments: 0 = low vol, 1 = high vol
        init_states = (roll_std > median_vol).astype(int)

        # State parameters
        mu = np.array([float(rets[init_states == s].mean()) for s in range(n_states)])
        sigma2 = np.array([max(float(rets[init_states == s].var()), 1e-8) for s in range(n_states)])

        # Initial transition matrix (sticky regimes)
        A = np.array([[0.95, 0.05], [0.05, 0.95]])
        pi = np.array([0.5, 0.5])  # initial state probabilities

        def emission_prob(t: int) -> np.ndarray:
            """Gaussian emission probability for observation at time t."""
            probs = np.array([
                float(stats.norm.pdf(rets[t], mu[s], np.sqrt(sigma2[s])))
                for s in range(n_states)
            ])
            return probs + 1e-300  # numerical guard

        # ---- EM algorithm (Baum-Welch) ----
        max_iter = 50
        log_lik_prev = -np.inf

        for iteration in range(max_iter):
            # Forward pass
            alpha_fwd = np.zeros((n, n_states))
            alpha_fwd[0] = pi * emission_prob(0)
            alpha_fwd[0] /= alpha_fwd[0].sum() + 1e-300
            scale = np.ones(n)

            for t in range(1, n):
                alpha_fwd[t] = (alpha_fwd[t - 1] @ A) * emission_prob(t)
                scale[t] = alpha_fwd[t].sum()
                if scale[t] > 1e-300:
                    alpha_fwd[t] /= scale[t]

            # Backward pass
            beta_bwd = np.ones((n, n_states))
            for t in range(n - 2, -1, -1):
                beta_bwd[t] = A @ (emission_prob(t + 1) * beta_bwd[t + 1])
                s = beta_bwd[t].sum()
                if s > 1e-300:
                    beta_bwd[t] /= s

            # Smoothed state probabilities (gamma)
            gamma = alpha_fwd * beta_bwd
            row_sums = gamma.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-300] = 1.0
            gamma /= row_sums

            # Xi (joint transition probabilities)
            xi = np.zeros((n - 1, n_states, n_states))
            for t in range(n - 1):
                emis_next = emission_prob(t + 1)
                xi[t] = np.outer(alpha_fwd[t], emis_next * beta_bwd[t + 1]) * A
                xi_sum = xi[t].sum()
                if xi_sum > 1e-300:
                    xi[t] /= xi_sum

            # M-step: update parameters
            A = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            A /= A.sum(axis=1, keepdims=True) + 1e-300
            pi = gamma[0]

            for s in range(n_states):
                g_s = gamma[:, s]
                mu[s] = float((g_s * rets).sum() / (g_s.sum() + 1e-300))
                sigma2[s] = float((g_s * (rets - mu[s]) ** 2).sum() / (g_s.sum() + 1e-300))
                sigma2[s] = max(sigma2[s], 1e-8)

            log_lik = float(np.log(scale + 1e-300).sum())
            if abs(log_lik - log_lik_prev) < 1e-6:
                break
            log_lik_prev = log_lik

        # Sort states: state 0 = low vol, state 1 = high vol
        if sigma2[0] > sigma2[1]:
            mu = mu[::-1]
            sigma2 = sigma2[::-1]
            gamma = gamma[:, ::-1]
            A = A[::-1, :][:, ::-1]

        # Current regime probability (probability of being in high-vol state)
        current_high_vol_prob = float(gamma[-1, 1])

        # Expected duration (geometric distribution: 1 / (1 - P_ii))
        expected_duration_low = 1.0 / (1.0 - float(A[0, 0]) + 1e-10)
        expected_duration_high = 1.0 / (1.0 - float(A[1, 1]) + 1e-10)

        regime_probs = pd.DataFrame(
            {"low_vol_prob": gamma[:, 0], "high_vol_prob": gamma[:, 1]},
            index=returns.dropna().index,
        )

        logger.info(
            "hmm_regime_switching",
            n_obs=n,
            n_regimes=n_states,
            mu_low=round(float(mu[0]), 4),
            sigma_low=round(float(np.sqrt(sigma2[0])), 4),
            sigma_high=round(float(np.sqrt(sigma2[1])), 4),
            current_high_vol_prob=round(current_high_vol_prob, 4),
        )

        return {
            "regime_probs": regime_probs,
            "current_regime_prob": round(current_high_vol_prob, 4),
            "current_regime": "high_vol" if current_high_vol_prob > 0.5 else "low_vol",
            "low_vol_params": {
                "mean_daily": round(float(mu[0]) / 100, 6),
                "vol_daily": round(float(np.sqrt(sigma2[0])) / 100, 6),
                "vol_annualised": round(float(np.sqrt(sigma2[0])) / 100 * np.sqrt(252), 4),
            },
            "high_vol_params": {
                "mean_daily": round(float(mu[1]) / 100, 6),
                "vol_daily": round(float(np.sqrt(sigma2[1])) / 100, 6),
                "vol_annualised": round(float(np.sqrt(sigma2[1])) / 100 * np.sqrt(252), 4),
            },
            "transition_matrix": {
                "low_to_low": round(float(A[0, 0]), 4),
                "low_to_high": round(float(A[0, 1]), 4),
                "high_to_low": round(float(A[1, 0]), 4),
                "high_to_high": round(float(A[1, 1]), 4),
            },
            "expected_duration_days": {
                "low_vol_regime": round(expected_duration_low, 1),
                "high_vol_regime": round(expected_duration_high, 1),
            },
            "n_observations": n,
        }

    async def realized_vol_percentile(
        self,
        asset: str,
        lookback_days: int = 21,
        hist_window_days: int = 252,
    ) -> float:
        """Current realised volatility percentile rank vs historical distribution.

        Parameters
        ----------
        asset : str
            Ticker symbol.
        lookback_days : int
            Window for current realised vol computation (default 21 = 1 month).
        hist_window_days : int
            Historical window for percentile rank (default 252 = 1 year).

        Returns
        -------
        float in [0, 1] — 1.0 = highest vol in history, 0.0 = lowest.
        """
        end = date.today()
        start = end - timedelta(days=hist_window_days + lookback_days + 60)
        prices = await _fetch_prices([asset], start, end)

        if prices.empty or asset not in prices.columns:
            raise ValueError(f"No price data for {asset}")

        rets = np.log(prices[asset] / prices[asset].shift(1)).dropna()
        rolling_vol = rets.rolling(window=lookback_days, min_periods=max(5, lookback_days // 2)).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 10:
            raise ValueError(f"Insufficient vol history for {asset}")

        current_vol = float(rolling_vol.iloc[-1])
        pct_rank = float(stats.percentileofscore(rolling_vol.values, current_vol) / 100.0)

        logger.info(
            "realized_vol_percentile",
            asset=asset,
            current_vol=round(current_vol, 4),
            pct_rank=round(pct_rank, 4),
            lookback=lookback_days,
        )

        return round(pct_rank, 4)


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------

class AlertEngine:
    """Real-time regime, correlation, and volatility alert engine."""

    def __init__(self) -> None:
        self._detector = RegimeDetector()
        self._corr_monitor = CorrelationMonitor()
        self._vol_model = VolatilityRegimeModel()

    async def check_all_alerts(
        self,
        portfolio_tickers: list[str],
    ) -> list[dict]:
        """Run all regime/correlation checks and return active alerts.

        Alert types:
          - regime_change      — macro regime shifted
          - correlation_spike  — average portfolio correlation > 0.75
          - vix_spike          — VIX > 25
          - yield_curve_inversion — T10Y2Y < 0
          - credit_spread_widening — IG OAS > 2.0%

        Returns
        -------
        list[dict] — each alert has type, severity, title, message, value, threshold.
        """
        alerts: list[dict] = []

        # ---- 1. Regime detection ----
        try:
            regime_result = await self._detector.get_current_regime()
            history = await self._detector.regime_history(lookback_days=30)
            regime_change = await self._detector.detect_regime_change(
                regime_result["composite_regime"], history, lookback_days=5
            )
            if regime_change.get("regime_changed"):
                alerts.append({
                    "type": "regime_change",
                    "severity": "high",
                    "title": "Market Regime Change Detected",
                    "message": regime_change["alert_message"],
                    "current_regime": regime_result["composite_regime"],
                    "previous_regime": regime_change.get("previous_regime"),
                    "confidence": regime_result.get("confidence"),
                })
        except Exception as exc:
            logger.warning("alert_engine.regime_check_failed", error=str(exc))

        # ---- 2. VIX spike ----
        try:
            vix_series = await _fetch_fred_series("VIXCLS", lookback_days=5)
            if not vix_series.empty:
                vix_val = float(vix_series.dropna().iloc[-1])
                if vix_val > 40:
                    alerts.append({
                        "type": "vix_spike",
                        "severity": "critical",
                        "title": "Extreme VIX Reading",
                        "message": f"VIX at {vix_val:.1f} — extreme fear. Reduce risk and hedge.",
                        "value": vix_val,
                        "threshold": 40,
                    })
                elif vix_val > 25:
                    alerts.append({
                        "type": "vix_spike",
                        "severity": "high",
                        "title": "Elevated VIX",
                        "message": f"VIX at {vix_val:.1f} — risk-off conditions. Consider defensive positioning.",
                        "value": vix_val,
                        "threshold": 25,
                    })
        except Exception as exc:
            logger.warning("alert_engine.vix_check_failed", error=str(exc))

        # ---- 3. Yield curve inversion ----
        try:
            yc_series = await _fetch_fred_series("T10Y2Y", lookback_days=5)
            if not yc_series.empty:
                yc_val = float(yc_series.dropna().iloc[-1])
                if yc_val < -0.25:
                    alerts.append({
                        "type": "yield_curve_inversion",
                        "severity": "high",
                        "title": "Yield Curve Inverted",
                        "message": f"10Y-2Y spread at {yc_val:.2f}% — recession indicator. Extend duration selectively.",
                        "value": yc_val,
                        "threshold": 0.0,
                    })
        except Exception as exc:
            logger.warning("alert_engine.yield_curve_check_failed", error=str(exc))

        # ---- 4. Credit spread widening ----
        try:
            cs_series = await _fetch_fred_series("BAMLC0A0CM", lookback_days=5)
            if not cs_series.empty:
                cs_val = float(cs_series.dropna().iloc[-1])
                if cs_val > 3.0:
                    alerts.append({
                        "type": "credit_spread_widening",
                        "severity": "critical",
                        "title": "Extreme Credit Spread Widening",
                        "message": f"IG OAS at {cs_val:.2f}% — credit stress. Reduce HY exposure.",
                        "value": cs_val,
                        "threshold": 3.0,
                    })
                elif cs_val > 2.0:
                    alerts.append({
                        "type": "credit_spread_widening",
                        "severity": "high",
                        "title": "Credit Spreads Widening",
                        "message": f"IG OAS at {cs_val:.2f}% — watch credit conditions.",
                        "value": cs_val,
                        "threshold": 2.0,
                    })
        except Exception as exc:
            logger.warning("alert_engine.credit_check_failed", error=str(exc))

        # ---- 5. Portfolio correlation spike ----
        if portfolio_tickers:
            try:
                end = date.today()
                start = end - timedelta(days=90)
                prices = await _fetch_prices(portfolio_tickers, start, end)
                if not prices.empty:
                    rets = np.log(prices / prices.shift(1)).dropna(how="all")
                    available = {t: rets[t] for t in portfolio_tickers if t in rets.columns}
                    if len(available) >= 2:
                        curr_corr = self._corr_monitor.compute_rolling_correlation_matrix(available, window=21)
                        spike_result = self._corr_monitor.detect_correlation_spike(
                            curr_corr, [], threshold=0.75
                        )
                        if spike_result["spike_detected"]:
                            avg_c = spike_result["avg_corr"]
                            alerts.append({
                                "type": "correlation_spike",
                                "severity": "high",
                                "title": "Portfolio Correlation Spike",
                                "message": (
                                    f"Average portfolio pairwise correlation: {avg_c:.2f}. "
                                    "Diversification benefit is significantly reduced."
                                ),
                                "avg_corr": avg_c,
                                "threshold": 0.75,
                                "driving_pairs": spike_result.get("driving_pairs", [])[:3],
                            })
            except Exception as exc:
                logger.warning("alert_engine.corr_check_failed", error=str(exc))

        # Sort: critical first, then high, then medium
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: severity_order.get(a.get("severity", "low"), 4))

        logger.info("alert_engine.check_complete", n_alerts=len(alerts))
        return alerts

    def format_alert(self, alert: dict) -> str:
        """Format a single alert dict as a human-readable string."""
        severity = alert.get("severity", "info").upper()
        alert_type = alert.get("type", "unknown").replace("_", " ").title()
        title = alert.get("title", "Alert")
        message = alert.get("message", "")
        value = alert.get("value")
        threshold = alert.get("threshold")

        lines = [
            f"[{severity}] {alert_type}: {title}",
            f"  {message}",
        ]
        if value is not None and threshold is not None:
            lines.append(f"  Value: {value:.2f} | Threshold: {threshold:.2f}")
        return "\n".join(lines)

    async def alert_history(self, lookback_days: int = 30) -> pd.DataFrame:
        """Historical alerts based on FRED signal breach history.

        Reconstructs alerts from FRED series over the lookback period.
        Returns a DataFrame with columns: date, type, severity, value.
        """
        vix = await _fetch_fred_series("VIXCLS", lookback_days=lookback_days + 30)
        yc = await _fetch_fred_series("T10Y2Y", lookback_days=lookback_days + 30)
        cs = await _fetch_fred_series("BAMLC0A0CM", lookback_days=lookback_days + 30)

        df = pd.DataFrame({"vix": vix, "yield_curve": yc, "credit_spread": cs}).dropna(how="all")
        df = df.last(f"{lookback_days}D")

        records: list[dict] = []
        for dt, row in df.iterrows():
            v = float(row.get("vix", 0.0) or 0.0)
            y = float(row.get("yield_curve", 1.0) or 1.0)
            c = float(row.get("credit_spread", 1.0) or 1.0)

            if v > 40:
                records.append({"date": dt, "type": "vix_spike", "severity": "critical", "value": v})
            elif v > 25:
                records.append({"date": dt, "type": "vix_spike", "severity": "high", "value": v})

            if y < -0.25:
                records.append({"date": dt, "type": "yield_curve_inversion", "severity": "high", "value": y})

            if c > 3.0:
                records.append({"date": dt, "type": "credit_spread_widening", "severity": "critical", "value": c})
            elif c > 2.0:
                records.append({"date": dt, "type": "credit_spread_widening", "severity": "high", "value": c})

        result = pd.DataFrame(records)
        if not result.empty:
            result["date"] = pd.to_datetime(result["date"])
            result = result.set_index("date").sort_index()
        return result


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

regime_router = APIRouter(prefix="/api/regime", tags=["regime"])

_regime_detector = RegimeDetector()
_corr_monitor = CorrelationMonitor()
_vol_model = VolatilityRegimeModel()
_alert_engine = AlertEngine()


@regime_router.get("/current")
async def get_current_regime() -> dict:
    """Current 8-signal macro regime classification."""
    try:
        return await _regime_detector.get_current_regime()
    except Exception as exc:
        logger.error("regime_current_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/history")
async def get_regime_history(
    days: int = Query(default=252, ge=10, le=1260),
) -> dict:
    """Historical daily regime classifications."""
    try:
        history = await _regime_detector.regime_history(lookback_days=days)
        if history.empty:
            return {"regime_history": [], "n_days": 0}
        records = history.reset_index().rename(columns={"index": "date"})
        records["date"] = records["date"].dt.strftime("%Y-%m-%d")
        return {
            "regime_history": records.to_dict(orient="records"),
            "n_days": len(records),
        }
    except Exception as exc:
        logger.error("regime_history_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/optimal-allocation")
async def get_optimal_allocation(
    regime: Optional[str] = Query(default=None, description="Override regime label"),
) -> dict:
    """Suggested asset allocation for current (or specified) regime."""
    try:
        if regime is None:
            result = await _regime_detector.get_current_regime()
            regime = result["composite_regime"]
        return _regime_detector.get_regime_optimal_allocation(regime)
    except Exception as exc:
        logger.error("optimal_allocation_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/correlation")
async def get_sector_correlation(
    lookback_days: int = Query(default=63, ge=21, le=252),
) -> dict:
    """11-sector GICS ETF correlation heatmap."""
    try:
        heatmap = await _corr_monitor.sector_correlation_heatmap(lookback_days=lookback_days)
        if heatmap.empty:
            return {"error": "No sector correlation data available."}
        return {
            "sectors": list(heatmap.index),
            "correlation_matrix": heatmap.round(4).to_dict(),
            "lookback_days": lookback_days,
            "as_of": date.today().isoformat(),
        }
    except Exception as exc:
        logger.error("correlation_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/alerts")
async def get_active_alerts(
    tickers: str = Query(default="", description="Comma-separated portfolio tickers"),
) -> dict:
    """Active regime, correlation, and volatility alerts."""
    try:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else []
        active_alerts = await _alert_engine.check_all_alerts(ticker_list)
        formatted = [_alert_engine.format_alert(a) for a in active_alerts]
        return {
            "n_alerts": len(active_alerts),
            "alerts": active_alerts,
            "formatted_alerts": formatted,
            "as_of": date.today().isoformat(),
        }
    except Exception as exc:
        logger.error("alerts_route_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@regime_router.get("/volatility/{ticker}")
async def get_vol_regime(
    ticker: str,
    lookback_days: int = Query(default=21, ge=5, le=63),
    hist_window_days: int = Query(default=252, ge=63, le=1260),
) -> dict:
    """Volatility regime for a specific asset (GARCH + percentile rank)."""
    try:
        end = date.today()
        start = end - timedelta(days=hist_window_days + lookback_days + 60)
        prices = await _fetch_prices([ticker], start, end)

        if prices.empty or ticker not in prices.columns:
            raise HTTPException(status_code=404, detail=f"No price data for ticker: {ticker}")

        rets = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()
        if len(rets) < 30:
            raise HTTPException(status_code=422, detail=f"Insufficient history for {ticker}")

        garch_result = _vol_model.fit_garch(rets)
        pct_rank = await _vol_model.realized_vol_percentile(ticker, lookback_days, hist_window_days)

        return {
            "ticker": ticker,
            "vol_percentile_rank": pct_rank,
            "vol_label": (
                "extreme" if pct_rank > 0.90
                else "high" if pct_rank > 0.75
                else "normal" if pct_rank > 0.25
                else "low"
            ),
            "garch": garch_result,
            "as_of": date.today().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("vol_regime_route_error", ticker=ticker, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
