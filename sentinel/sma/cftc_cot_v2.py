"""
CFTC COT Positioning v2 — Enhanced Intelligence Module (dim_046, target 9).

Extends cftc_cot.py with:
  - TFFReportAdapter: Traders in Financial Futures disaggregated positioning
  - COTExtremeDetector: COT Index + crowding detection with 5-year window
  - MacroCOTSignalEngine: USD / Gold / Oil / Treasury macro signals
  - COTBacktestEngine: Historical signal IC and forward-return analysis
  - FastAPI router cot_v2_router

Public API
----------
TFFReportAdapter
    get_tff_positions(market)           -> TFFPositions
    get_tff_history(market, weeks)      -> pd.DataFrame

COTExtremeDetector
    compute_cot_index_5yr(contract, trader_type)    -> pd.Series
    classify_extreme(contract, trader_type)          -> ExtremeReading
    detect_crowding(contracts)                       -> pd.DataFrame
    compute_net_percentile(contract, trader_type)    -> float

MacroCOTSignalEngine
    usd_positioning_signal()            -> dict
    gold_safe_haven_signal()            -> dict
    oil_supply_demand_signal()          -> dict
    treasury_duration_signal()          -> dict
    global_risk_appetite_composite()    -> dict

COTBacktestEngine
    run_backtest(contract, trader_type, forward_weeks)  -> BacktestResult
    compute_ic(contract, trader_type, forward_weeks)    -> float
    rank_signals_by_ic(contracts, trader_type)          -> pd.DataFrame
    plot_signal_vs_return(contract, trader_type)        -> dict

cot_v2_router  — FastAPI router, prefix /cot/v2
"""
from __future__ import annotations

import io
import os
import asyncio
import zipfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats

from sentinel.core.logging import get_logger
from sentinel.sma.cftc_cot import (
    COTDataAdapter,
    COTSignalEngine,
    FUTURES_MARKET_MAP,
    _CACHE_DIR,
    _CFTC_BASE,
    _HEADERS,
    _TIMEOUT,
    _ensure_cache_dir,
    _filter_market,
    _find_col,
    _to_numeric_series,
    _EQUITY_CONTRACTS,
    _RATES_CONTRACTS,
    _FX_CONTRACTS,
    _ENERGY_CONTRACTS,
    _METALS_CONTRACTS,
    _AG_CONTRACTS,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — TFF-specific
# ---------------------------------------------------------------------------

_TFF_LATEST_URL = f"{_CFTC_BASE}/traders_fin_fut_txt_2016_2025.zip"
_TFF_YEAR_URL   = f"{_CFTC_BASE}/traders_in_financial_futures_fut_hist_{{year}}.zip"

# TFF column names from CFTC documentation
_TFF_COLS = {
    "date":            "As of Date in Form YYYY-MM-DD",
    "market":          "Market_and_Exchange_Names",
    "dealer_long":     "Dealer_Positions_Long_All",
    "dealer_short":    "Dealer_Positions_Short_All",
    "dealer_spread":   "Dealer_Positions_Spread_All",
    "am_long":         "Asset_Mgr_Positions_Long_All",
    "am_short":        "Asset_Mgr_Positions_Short_All",
    "am_spread":       "Asset_Mgr_Positions_Spread_All",
    "lev_long":        "Lev_Money_Positions_Long_All",
    "lev_short":       "Lev_Money_Positions_Short_All",
    "lev_spread":      "Lev_Money_Positions_Spread_All",
    "other_long":      "Other_Rept_Positions_Long_All",
    "other_short":     "Other_Rept_Positions_Short_All",
    "other_spread":    "Other_Rept_Positions_Spread_All",
    "nonrep_long":     "NonRept_Positions_Long_All",
    "nonrep_short":    "NonRept_Positions_Short_All",
    "oi":              "Open_Interest_All",
}

# COT backtest parameters
_BACKTEST_LOOKBACK_YEARS = 5
_MIN_OBSERVATIONS        = 52   # minimum data points for meaningful IC
_COT_INDEX_LOOKBACK      = 260  # 5 years of weekly data

# Macro market mapping for signal engine
_MACRO_MARKETS = {
    "dxy":       ["6E", "6J", "6B", "6C", "6A"],   # USD proxied via major FX pairs
    "gold":      ["GC"],
    "oil":       ["CL", "BZ"],
    "treasuries": ["ZN", "ZB", "UB", "ZT", "ZF"],
    "equities":  ["ES", "NQ"],
}

# Simple in-process TTL cache for price series (avoids re-fetching)
_PRICE_CACHE: dict[str, tuple[float, pd.Series]] = {}
_PRICE_CACHE_TTL = 7200  # 2 hours


# ---------------------------------------------------------------------------
# Pydantic / dataclass models
# ---------------------------------------------------------------------------

@dataclass
class TFFPositions:
    """Traders in Financial Futures disaggregated positioning."""
    market:           str
    as_of:            str
    dealer_long:      float = 0.0
    dealer_short:     float = 0.0
    dealer_net:       float = 0.0
    dealer_pct_oi:    float = 0.0
    am_long:          float = 0.0       # Asset Manager (smart money / long-only)
    am_short:         float = 0.0
    am_net:           float = 0.0
    am_pct_oi:        float = 0.0
    lev_long:         float = 0.0       # Leveraged Funds (hedge funds / speculators)
    lev_short:        float = 0.0
    lev_net:          float = 0.0
    lev_pct_oi:       float = 0.0
    other_long:       float = 0.0
    other_short:      float = 0.0
    other_net:        float = 0.0
    open_interest:    float = 0.0
    am_trend_signal:  str   = "neutral"
    lev_trend_signal: str   = "neutral"


@dataclass
class ExtremeReading:
    """Result of extreme positioning detection."""
    contract:           str
    trader_type:        str
    cot_index_current:  float
    cot_index_5yr:      float
    net_percentile:     float
    is_extreme_long:    bool
    is_extreme_short:   bool
    is_crowded:         bool
    crowding_direction: str     # "long" | "short" | "none"
    signal:             str
    contrarian_strength: float  # 0-100, higher = stronger contrarian call
    as_of:              str


@dataclass
class BacktestResult:
    """COT signal backtest results."""
    contract:           str
    trader_type:        str
    forward_weeks:      int
    n_observations:     int
    ic_spearman:        float           # Information Coefficient
    ic_pvalue:          float
    ic_significant:     bool
    long_avg_return:    float           # avg return when COT > 75
    short_avg_return:   float           # avg return when COT < 25
    long_short_spread:  float
    hit_rate_long:      float           # % of long signals with positive returns
    hit_rate_short:     float
    max_drawdown_strategy: float
    sharpe_estimate:    float
    best_threshold:     float           # optimal COT Index threshold
    signal_decay_weeks: int             # IC peak at N weeks
    notes:              str = ""


# ---------------------------------------------------------------------------
# TFFReportAdapter
# ---------------------------------------------------------------------------

class TFFReportAdapter:
    """
    Downloads and parses CFTC Traders in Financial Futures (TFF) reports.

    TFF disaggregated categories:
      - Dealer/Intermediary: primary dealers, market makers
      - Asset Manager/Institutional: pension funds, mutual funds, endowments (smart money)
      - Leveraged Funds: hedge funds, commodity pools, CTAs (speculative)
      - Other Reportable: mixed bag of reportable traders

    The TFF report covers only financial futures (equity indices, rates, FX).
    """

    def __init__(self) -> None:
        self._adapter = COTDataAdapter()
        self._tff_cache: dict[int, pd.DataFrame] = {}
        _ensure_cache_dir()

    def _load_tff(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load TFF data for a given year from cache or CFTC."""
        target_year = year or date.today().year

        # 1. In-memory cache
        if target_year in self._tff_cache:
            return self._tff_cache[target_year]

        # 2. Try disk-cached parquet via the main adapter
        cached = self._adapter.download_cot_report("traders_in_financial", year=target_year)
        if not cached.empty:
            self._tff_cache[target_year] = cached
            return cached

        return pd.DataFrame()

    def _get_tff_history_raw(self, market_substring: str, lookback_weeks: int) -> pd.DataFrame:
        """Return raw TFF rows matching *market_substring* for *lookback_weeks*."""
        years_needed = max(1, lookback_weeks // 52 + 1)
        current_year = date.today().year
        frames = []
        for y in range(current_year - years_needed + 1, current_year + 1):
            df = self._load_tff(y)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = _filter_market(combined, market_substring)
        if combined.empty:
            return pd.DataFrame()

        date_col = _find_col(
            combined,
            [_TFF_COLS["date"], "As of Date in Form YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD", "date"]
        )
        if date_col:
            combined["_date"] = pd.to_datetime(combined[date_col], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=lookback_weeks)
            combined = combined[combined["_date"] >= cutoff]
            combined = combined.sort_values("_date").reset_index(drop=True)

        return combined

    def _extract_tff_row(self, row: pd.Series, market: str) -> TFFPositions:
        """Extract TFFPositions from a single raw TFF row."""
        def _g(col_key: str) -> float:
            col = _TFF_COLS.get(col_key, col_key)
            if col in row.index:
                return float(pd.to_numeric(row[col], errors="coerce") or 0.0)
            return 0.0

        oi = _g("oi") or 1.0
        dealer_long  = _g("dealer_long")
        dealer_short = _g("dealer_short")
        dealer_net   = dealer_long - dealer_short

        am_long  = _g("am_long")
        am_short = _g("am_short")
        am_net   = am_long - am_short

        lev_long  = _g("lev_long")
        lev_short = _g("lev_short")
        lev_net   = lev_long - lev_short

        other_long  = _g("other_long")
        other_short = _g("other_short")

        # Asset manager: net long = institutional bullish, trend-following signal
        am_signal = (
            "bullish"  if am_net / oi > 0.05  else
            "bearish"  if am_net / oi < -0.05 else
            "neutral"
        )
        # Leveraged: contrarian use — extreme lev long = crowded = warning
        lev_signal = (
            "crowded_long"  if lev_net / oi > 0.10  else
            "crowded_short" if lev_net / oi < -0.10 else
            "neutral"
        )

        date_val = ""
        date_col = _TFF_COLS["date"]
        if date_col in row.index:
            try:
                date_val = str(pd.to_datetime(row[date_col]).date())
            except Exception:
                date_val = str(row[date_col])

        return TFFPositions(
            market=market,
            as_of=date_val,
            dealer_long=dealer_long,
            dealer_short=dealer_short,
            dealer_net=dealer_net,
            dealer_pct_oi=round(dealer_net / oi * 100, 2),
            am_long=am_long,
            am_short=am_short,
            am_net=am_net,
            am_pct_oi=round(am_net / oi * 100, 2),
            lev_long=lev_long,
            lev_short=lev_short,
            lev_net=lev_net,
            lev_pct_oi=round(lev_net / oi * 100, 2),
            other_long=other_long,
            other_short=other_short,
            other_net=other_long - other_short,
            open_interest=oi,
            am_trend_signal=am_signal,
            lev_trend_signal=lev_signal,
        )

    def get_tff_positions(self, market: str) -> TFFPositions:
        """
        Return the most recent TFF disaggregated positions for *market*.

        *market* is a contract symbol (e.g. 'ES', '6E', 'ZN') or a CFTC
        market name substring (e.g. 'S&P 500', 'EURO FX').

        Returns TFFPositions with dealer, asset_mgr, and leveraged_fund nets.
        """
        market_name = FUTURES_MARKET_MAP.get(market.upper(), market)
        raw = self._get_tff_history_raw(market_name, lookback_weeks=4)

        if raw.empty:
            logger.warning("TFF: no data found for market %s", market)
            return TFFPositions(market=market, as_of=str(date.today()))

        latest_row = raw.iloc[-1]
        return self._extract_tff_row(latest_row, market)

    def get_tff_history(
        self,
        market: str,
        lookback_weeks: int = 104,
    ) -> pd.DataFrame:
        """
        Return a time-series DataFrame of TFF positions for *market*.

        Columns: date, dealer_net, dealer_pct_oi, am_net, am_pct_oi,
                 lev_net, lev_pct_oi, other_net, open_interest,
                 am_trend_signal, lev_trend_signal.
        """
        market_name = FUTURES_MARKET_MAP.get(market.upper(), market)
        raw = self._get_tff_history_raw(market_name, lookback_weeks=lookback_weeks)

        if raw.empty:
            return pd.DataFrame()

        records = []
        for _, row in raw.iterrows():
            pos = self._extract_tff_row(row, market)
            records.append({
                "date":             pos.as_of,
                "dealer_net":       pos.dealer_net,
                "dealer_pct_oi":    pos.dealer_pct_oi,
                "am_net":           pos.am_net,
                "am_pct_oi":        pos.am_pct_oi,
                "lev_net":          pos.lev_net,
                "lev_pct_oi":       pos.lev_pct_oi,
                "other_net":        pos.other_net,
                "open_interest":    pos.open_interest,
                "am_trend_signal":  pos.am_trend_signal,
                "lev_trend_signal": pos.lev_trend_signal,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def compare_am_vs_lev(self, market: str, lookback_weeks: int = 52) -> pd.DataFrame:
        """
        Compare asset manager vs leveraged fund net positioning over time.

        Useful for detecting divergence: when asset managers go long while
        leveraged funds go short → typically bullish (smart money leading).
        Returns DataFrame with am_net, lev_net, divergence, signal columns.
        """
        hist = self.get_tff_history(market, lookback_weeks=lookback_weeks)
        if hist.empty:
            return pd.DataFrame()

        hist["divergence"] = hist["am_net"] - hist["lev_net"]
        hist["signal"] = hist.apply(
            lambda r: (
                "smart_money_bullish"   if r["am_net"] > 0 and r["lev_net"] < 0 else
                "both_bullish"          if r["am_net"] > 0 and r["lev_net"] > 0 else
                "smart_money_bearish"   if r["am_net"] < 0 and r["lev_net"] > 0 else
                "both_bearish"
            ),
            axis=1,
        )
        return hist[["date", "am_net", "lev_net", "divergence", "signal", "open_interest"]].copy()


# ---------------------------------------------------------------------------
# COTExtremeDetector
# ---------------------------------------------------------------------------

class COTExtremeDetector:
    """
    Identifies extreme and crowded positioning using a 5-year COT Index window.

    COT Index = (current_net - min_5yr) / (max_5yr - min_5yr) × 100

    Extremes:
      > 90  → historically very long  → contrarian SELL (too much bullishness priced in)
      < 10  → historically very short → contrarian BUY  (bearish sentiment fully priced)

    Crowding: extreme one-sided positioning across multiple traders or markets.
    """

    def __init__(self, engine: Optional[COTSignalEngine] = None) -> None:
        self._engine = engine or COTSignalEngine()
        self._lookback_weeks = _COT_INDEX_LOOKBACK   # 5 years

    def compute_cot_index_5yr(
        self,
        contract: str,
        trader_type: str = "managed_money",
    ) -> pd.Series:
        """
        Compute the COT Index using a 5-year (260-week) rolling window.

        Returns a pd.Series indexed by date, values 0-100.
        """
        return self._engine.compute_cot_index(
            contract, trader_type=trader_type, lookback_weeks=self._lookback_weeks
        )

    def compute_net_percentile(
        self,
        contract: str,
        trader_type: str = "managed_money",
    ) -> float:
        """
        Compute the percentile rank of current net positioning vs 5-year history.

        0 = never been this short over 5 years
        100 = never been this long over 5 years
        """
        idx = self.compute_cot_index_5yr(contract, trader_type)
        if idx.empty:
            return 50.0
        current = float(idx.iloc[-1])
        return round(float((idx <= current).mean() * 100), 1)

    def classify_extreme(
        self,
        contract: str,
        trader_type: str = "managed_money",
        extreme_threshold: float = 10.0,
        crowding_threshold: float = 80.0,
    ) -> ExtremeReading:
        """
        Classify current positioning as extreme/crowded.

        Parameters
        ----------
        extreme_threshold : float
            COT Index boundary for extreme (default 10 → <10 extreme short, >90 extreme long).
        crowding_threshold : float
            Net percentile threshold for crowding detection (default 80).

        Returns
        -------
        ExtremeReading dataclass.
        """
        idx_5yr = self.compute_cot_index_5yr(contract, trader_type)
        idx_1yr = self._engine.compute_cot_index(contract, trader_type, lookback_weeks=52)
        percentile = self.compute_net_percentile(contract, trader_type)

        if idx_5yr.empty:
            return ExtremeReading(
                contract=contract,
                trader_type=trader_type,
                cot_index_current=50.0,
                cot_index_5yr=50.0,
                net_percentile=50.0,
                is_extreme_long=False,
                is_extreme_short=False,
                is_crowded=False,
                crowding_direction="none",
                signal="no_data",
                contrarian_strength=0.0,
                as_of=str(date.today()),
            )

        current_5yr = float(idx_5yr.iloc[-1])
        current_1yr = float(idx_1yr.iloc[-1]) if not idx_1yr.empty else current_5yr

        is_extreme_long  = current_5yr > (100.0 - extreme_threshold)
        is_extreme_short = current_5yr < extreme_threshold

        # Crowding: both 5yr index extreme AND 1yr index confirms
        is_crowded = (
            (is_extreme_long  and current_1yr > 70) or
            (is_extreme_short and current_1yr < 30) or
            (percentile > crowding_threshold) or
            (percentile < (100 - crowding_threshold))
        )

        crowding_direction = (
            "long"  if is_crowded and current_5yr > 50 else
            "short" if is_crowded and current_5yr < 50 else
            "none"
        )

        # Signal logic: for speculative traders (managed_money / leveraged),
        # extreme long = contrarian bearish (too crowded); extreme short = contrarian bullish.
        # For commercial traders, reverse interpretation.
        speculative_types = {"managed_money", "leveraged_money", "noncommercial"}
        is_speculative = trader_type.lower().replace(" ", "_") in speculative_types

        if is_extreme_long:
            signal = "contrarian_bearish" if is_speculative else "bullish_confirmation"
        elif is_extreme_short:
            signal = "contrarian_bullish" if is_speculative else "bearish_confirmation"
        else:
            signal = "neutral"

        # Contrarian strength: how far from 50% is the index? (0-100)
        contrarian_strength = round(abs(current_5yr - 50) * 2, 1)  # 0-100

        as_of = str(idx_5yr.index[-1].date()) if hasattr(idx_5yr.index[-1], "date") else str(idx_5yr.index[-1])

        return ExtremeReading(
            contract=contract,
            trader_type=trader_type,
            cot_index_current=round(current_5yr, 1),
            cot_index_5yr=round(current_5yr, 1),
            net_percentile=percentile,
            is_extreme_long=is_extreme_long,
            is_extreme_short=is_extreme_short,
            is_crowded=is_crowded,
            crowding_direction=crowding_direction,
            signal=signal,
            contrarian_strength=contrarian_strength,
            as_of=as_of,
        )

    def detect_crowding(
        self,
        contracts: Optional[list[str]] = None,
        trader_type: str = "managed_money",
        extreme_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """
        Scan multiple contracts for crowded trades.

        Returns a DataFrame of crowded positions sorted by contrarian strength.
        Only includes contracts where is_crowded=True.
        """
        targets = contracts or (
            _EQUITY_CONTRACTS + _RATES_CONTRACTS[:4] +
            _FX_CONTRACTS[:6] + _ENERGY_CONTRACTS[:3] +
            _METALS_CONTRACTS[:2]
        )

        rows = []
        for contract in targets:
            try:
                reading = self.classify_extreme(
                    contract, trader_type=trader_type, extreme_threshold=extreme_threshold
                )
                if reading.is_crowded:
                    rows.append({
                        "contract":            contract,
                        "market_name":         FUTURES_MARKET_MAP.get(contract, contract),
                        "trader_type":         trader_type,
                        "cot_index_5yr":       reading.cot_index_5yr,
                        "net_percentile":      reading.net_percentile,
                        "crowding_direction":  reading.crowding_direction,
                        "signal":              reading.signal,
                        "contrarian_strength": reading.contrarian_strength,
                        "as_of":               reading.as_of,
                    })
            except Exception as exc:
                logger.debug("Crowding detection failed for %s: %s", contract, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("contrarian_strength", ascending=False).reset_index(drop=True)
        return df

    def get_positioning_heatmap(
        self,
        contracts: Optional[list[str]] = None,
        trader_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a positioning heatmap: contracts × trader_types → COT Index value.

        Values: 0 (max short) to 100 (max long), 5-year window.
        Useful for visualising where speculative positioning is stretched.
        """
        targets = contracts or (
            _EQUITY_CONTRACTS[:4] + _RATES_CONTRACTS[:4] +
            _FX_CONTRACTS[:4] + _ENERGY_CONTRACTS[:3] + _METALS_CONTRACTS[:2]
        )
        tt_list = trader_types or ["managed_money", "asset_manager", "dealer"]

        data: dict[str, dict[str, float]] = {}
        for contract in targets:
            data[contract] = {}
            for tt in tt_list:
                try:
                    idx = self.compute_cot_index_5yr(contract, trader_type=tt)
                    data[contract][tt] = float(idx.iloc[-1]) if not idx.empty else float("nan")
                except Exception:
                    data[contract][tt] = float("nan")

        return pd.DataFrame(data).T


# ---------------------------------------------------------------------------
# MacroCOTSignalEngine
# ---------------------------------------------------------------------------

class MacroCOTSignalEngine:
    """
    Derives macro-level signals from COT positioning across asset classes.

    Key signals:
      - USD: leveraged fund net FX positioning as DXY trend signal
      - Gold: asset manager net as safe-haven demand gauge
      - Oil: managed money net as supply/demand balance signal
      - Treasuries: non-commercial TY/US net as duration risk appetite
      - Global risk appetite: composite risk-on/risk-off score
    """

    def __init__(
        self,
        engine: Optional[COTSignalEngine] = None,
        tff: Optional[TFFReportAdapter] = None,
        detector: Optional[COTExtremeDetector] = None,
    ) -> None:
        self._engine   = engine   or COTSignalEngine()
        self._tff      = tff      or TFFReportAdapter()
        self._detector = detector or COTExtremeDetector(engine=self._engine)

    def _get_net(self, contract: str, trader_type: str) -> Optional[float]:
        """Safely retrieve the most recent net position."""
        try:
            df = self._engine.compute_net_positioning(contract, trader_type)
            if df.empty:
                return None
            return float(df["net"].iloc[-1])
        except Exception:
            return None

    def _get_net_pct_oi(self, contract: str, trader_type: str) -> Optional[float]:
        """Net position as % of open interest."""
        try:
            df = self._engine.compute_net_positioning(contract, trader_type)
            if df.empty:
                return None
            return float(df["net_pct_oi"].iloc[-1])
        except Exception:
            return None

    def usd_positioning_signal(self) -> dict[str, Any]:
        """
        USD positioning signal derived from leveraged-fund net positions
        in EUR, GBP, JPY, CAD, AUD futures.

        Logic:
          - Each FX future: speculator net long = bullish that currency vs USD
          - Sum across majors: aggregate spec long in non-USD = bearish USD
          - COT Index > 70 for EUR (specs very long EUR) = USD short signal
          - COT Index < 30 for EUR (specs very short EUR) = USD long signal
        """
        result: dict[str, Any] = {
            "signal_type": "USD_positioning",
            "methodology": "Leveraged-fund net across major G10 FX futures",
            "as_of": str(date.today()),
            "pairs": {},
        }

        fx_contracts = ["6E", "6B", "6J", "6C", "6A", "6S"]
        total_non_usd_net = 0.0
        valid_count = 0

        for contract in fx_contracts:
            try:
                net_pct = self._get_net_pct_oi(contract, "managed_money")
                reading = self._detector.classify_extreme(contract, "managed_money")
                net_abs = self._get_net(contract, "managed_money")

                if net_pct is not None:
                    total_non_usd_net += net_pct
                    valid_count += 1

                # JPY is inverted (JPY/USD), adjust sign conceptually
                usd_bias = "usd_bearish" if (net_pct or 0) > 2 else ("usd_bullish" if (net_pct or 0) < -2 else "neutral")
                if contract == "6J":
                    usd_bias = "usd_bullish" if (net_pct or 0) > 2 else ("usd_bearish" if (net_pct or 0) < -2 else "neutral")

                result["pairs"][contract] = {
                    "currency": FUTURES_MARKET_MAP.get(contract, contract),
                    "spec_net_pct_oi": round(net_pct, 2) if net_pct is not None else None,
                    "spec_net_contracts": round(net_abs, 0) if net_abs is not None else None,
                    "cot_index_5yr": reading.cot_index_5yr,
                    "extreme": reading.is_extreme_long or reading.is_extreme_short,
                    "usd_bias": usd_bias,
                }
            except Exception as exc:
                logger.debug("USD signal: FX pair %s error: %s", contract, exc)

        # Aggregate USD signal
        avg_non_usd_net = total_non_usd_net / max(valid_count, 1)
        if avg_non_usd_net > 3.0:
            usd_signal = "BEARISH_USD"   # specs heavily long non-USD
            conviction = "high" if avg_non_usd_net > 6.0 else "moderate"
        elif avg_non_usd_net < -3.0:
            usd_signal = "BULLISH_USD"   # specs heavily short non-USD = long USD
            conviction = "high" if avg_non_usd_net < -6.0 else "moderate"
        else:
            usd_signal = "NEUTRAL"
            conviction = "low"

        result["aggregate_non_usd_net_pct"] = round(avg_non_usd_net, 2)
        result["usd_signal"] = usd_signal
        result["conviction"] = conviction
        result["interpretation"] = (
            f"Leveraged funds are {'net long' if avg_non_usd_net > 0 else 'net short'} "
            f"non-USD currencies by {abs(avg_non_usd_net):.1f}% of OI on average. "
            f"Signal: {usd_signal} with {conviction} conviction."
        )
        return result

    def gold_safe_haven_signal(self) -> dict[str, Any]:
        """
        Gold safe-haven demand signal from asset manager and managed money positioning.

        Asset managers (long-only, often large pension and SWFs):
          - Net long + growing = genuine safe-haven allocation
          - Rising AM net while lev funds are short = strong bullish signal (dumb money sells, smart buys)

        Managed money (CTAs, hedge funds):
          - Their extreme long can be a contrarian signal (crowded trade)
        """
        result: dict[str, Any] = {
            "signal_type": "GOLD_safe_haven",
            "as_of": str(date.today()),
        }

        try:
            # Asset manager positions (smart money)
            am_net    = self._get_net("GC", "asset_manager")
            am_pct    = self._get_net_pct_oi("GC", "asset_manager")
            am_reading = self._detector.classify_extreme("GC", "asset_manager")

            # Managed money / leveraged positioning (speculative)
            lev_net   = self._get_net("GC", "managed_money")
            lev_pct   = self._get_net_pct_oi("GC", "managed_money")
            lev_reading = self._detector.classify_extreme("GC", "managed_money")

            # Commercial (producers) — high commercial short = hedging selling pressure
            comm_net  = self._get_net("GC", "commercial")
            comm_reading = self._detector.classify_extreme("GC", "commercial")

            # Derive safe-haven signal
            safe_haven_score = 0  # -5 to +5
            notes = []

            if am_pct is not None and am_pct > 5.0:
                safe_haven_score += 2
                notes.append(f"Asset managers net long {am_pct:.1f}% OI")
            if am_reading.cot_index_5yr > 60:
                safe_haven_score += 1
                notes.append("AM positioning above 5yr median")
            if lev_pct is not None and lev_pct < -3.0:
                safe_haven_score += 2
                notes.append("Leveraged funds short while AM long = classic setup")
            if lev_reading.is_extreme_long:
                safe_haven_score -= 2
                notes.append("Warning: leveraged funds extremely long gold = crowded")
            if (comm_net or 0) < -200_000:
                safe_haven_score -= 1
                notes.append("Heavy commercial hedging (producer selling pressure)")

            signal = (
                "STRONG_SAFE_HAVEN_DEMAND" if safe_haven_score >= 3 else
                "MODERATE_SAFE_HAVEN_DEMAND" if safe_haven_score >= 1 else
                "CROWDED_LONG_CAUTION" if safe_haven_score <= -2 else
                "NEUTRAL"
            )

            result.update({
                "asset_manager_net": am_net,
                "asset_manager_net_pct_oi": am_pct,
                "asset_manager_cot_index_5yr": am_reading.cot_index_5yr,
                "leveraged_net": lev_net,
                "leveraged_net_pct_oi": lev_pct,
                "leveraged_cot_index_5yr": lev_reading.cot_index_5yr,
                "commercial_net": comm_net,
                "safe_haven_score": safe_haven_score,
                "signal": signal,
                "notes": notes,
            })
        except Exception as exc:
            logger.error("Gold safe-haven signal error: %s", exc)
            result["signal"] = "error"
            result["error"] = str(exc)

        return result

    def oil_supply_demand_signal(self) -> dict[str, Any]:
        """
        Oil market positioning signal: managed money net as energy demand proxy.

        Interpretation:
          - MM net long crude (CL + BZ) = speculative demand, bullish price bias
          - Commercial net short very large = hedgers confident selling forward = potentially bearish
          - Producer/merchant long = possibly bullish (producers buying upside)

        Returns signal combining WTI and Brent positioning.
        """
        result: dict[str, Any] = {
            "signal_type": "OIL_supply_demand",
            "as_of": str(date.today()),
            "markets": {},
        }

        total_mm_net_pct = 0.0
        valid = 0
        for contract in ["CL", "BZ"]:
            try:
                mm_net     = self._get_net(contract, "managed_money")
                mm_pct     = self._get_net_pct_oi(contract, "managed_money")
                comm_net   = self._get_net(contract, "commercial")
                comm_pct   = self._get_net_pct_oi(contract, "commercial")
                mm_reading = self._detector.classify_extreme(contract, "managed_money")
                comm_reading = self._detector.classify_extreme(contract, "commercial")

                if mm_pct is not None:
                    total_mm_net_pct += mm_pct
                    valid += 1

                result["markets"][contract] = {
                    "name": FUTURES_MARKET_MAP.get(contract, contract),
                    "managed_money_net": mm_net,
                    "managed_money_pct_oi": mm_pct,
                    "managed_money_cot_5yr": mm_reading.cot_index_5yr,
                    "commercial_net": comm_net,
                    "commercial_pct_oi": comm_pct,
                    "commercial_cot_5yr": comm_reading.cot_index_5yr,
                    "mm_signal": mm_reading.signal,
                    "comm_signal": comm_reading.signal,
                }
            except Exception as exc:
                logger.debug("Oil signal %s error: %s", contract, exc)

        avg_mm_pct = total_mm_net_pct / max(valid, 1)

        if avg_mm_pct > 10.0:
            signal = "BULLISH_DEMAND_DRIVEN"
        elif avg_mm_pct > 3.0:
            signal = "MODERATELY_BULLISH"
        elif avg_mm_pct < -5.0:
            signal = "BEARISH_DEMAND_WEAK"
        else:
            signal = "NEUTRAL"

        result["combined_mm_net_pct_oi"]  = round(avg_mm_pct, 2)
        result["signal"]                   = signal
        result["interpretation"] = (
            f"Combined managed money net in WTI+Brent: {avg_mm_pct:.1f}% of OI. "
            f"Energy demand/speculative signal: {signal}."
        )
        return result

    def treasury_duration_signal(self) -> dict[str, Any]:
        """
        US Treasury futures positioning: non-commercial (spec) net as duration
        risk-appetite gauge.

        Spec long Treasuries = expect rates to fall (risk-off / dovish)
        Spec short Treasuries = expect rates to rise (risk-on / hawkish)

        Returns a composite across the yield curve (2Y, 5Y, 10Y, 30Y).
        """
        result: dict[str, Any] = {
            "signal_type": "TREASURY_duration_appetite",
            "as_of": str(date.today()),
            "contracts": {},
        }

        rate_contracts = [
            ("ZT", "2Y"),
            ("ZF", "5Y"),
            ("ZN", "10Y"),
            ("ZB", "30Y"),
            ("UB", "Ultra30Y"),
        ]

        duration_score = 0.0
        valid = 0

        for contract, label in rate_contracts:
            try:
                spec_pct   = self._get_net_pct_oi(contract, "managed_money")
                spec_net   = self._get_net(contract, "managed_money")
                dealer_pct = self._get_net_pct_oi(contract, "dealer")
                am_pct     = self._get_net_pct_oi(contract, "asset_manager")
                reading    = self._detector.classify_extreme(contract, "managed_money")

                if spec_pct is not None:
                    # Weight long-end more (10Y+30Y have more duration sensitivity)
                    weight = 2.0 if contract in ("ZN", "ZB", "UB") else 1.0
                    duration_score += spec_pct * weight
                    valid += weight

                result["contracts"][contract] = {
                    "label": label,
                    "spec_net_pct_oi": spec_pct,
                    "spec_net": spec_net,
                    "dealer_net_pct_oi": dealer_pct,
                    "asset_mgr_net_pct_oi": am_pct,
                    "cot_index_5yr": reading.cot_index_5yr,
                    "signal": reading.signal,
                }
            except Exception as exc:
                logger.debug("Treasury %s error: %s", contract, exc)

        avg_duration_score = duration_score / max(valid, 1)

        if avg_duration_score > 5.0:
            duration_signal = "RISK_OFF_SPEC_LONG_DURATION"   # specs buying bonds = fear
        elif avg_duration_score > 1.0:
            duration_signal = "MILD_RISK_OFF"
        elif avg_duration_score < -5.0:
            duration_signal = "RISK_ON_SPEC_SHORT_DURATION"   # specs shorting bonds = rate-rise bet
        elif avg_duration_score < -1.0:
            duration_signal = "MILD_RISK_ON"
        else:
            duration_signal = "NEUTRAL"

        result["composite_spec_net_pct_oi"] = round(avg_duration_score, 2)
        result["duration_signal"]            = duration_signal
        result["interpretation"] = (
            f"Weighted speculator net in Treasuries: {avg_duration_score:.1f}% OI. "
            f"Duration risk appetite: {duration_signal}."
        )
        return result

    def global_risk_appetite_composite(self) -> dict[str, Any]:
        """
        Composite global risk-on / risk-off signal from COT positioning.

        Aggregates:
          - Equity futures spec net (positive = risk-on)
          - Treasury spec net (positive = risk-off)
          - Gold safe-haven demand
          - USD positioning
          - VIX futures speculator positioning

        Returns a -100 (max risk-off) to +100 (max risk-on) score.
        """
        result: dict[str, Any] = {
            "signal_type": "GLOBAL_risk_appetite_composite",
            "as_of": str(date.today()),
            "components": {},
        }

        risk_score = 0.0

        # Component 1: Equity positioning (+40 weight)
        try:
            eq_scores = []
            for contract in ["ES", "NQ"]:
                pct = self._get_net_pct_oi(contract, "managed_money")
                if pct is not None:
                    eq_scores.append(pct)
            if eq_scores:
                eq_avg = float(np.mean(eq_scores))
                eq_contribution = np.clip(eq_avg * 2, -40, 40)  # scale to ±40
                risk_score += eq_contribution
                result["components"]["equity_spec_net"] = round(eq_avg, 2)
                result["components"]["equity_contribution"] = round(eq_contribution, 1)
        except Exception as exc:
            logger.debug("Risk appetite equity error: %s", exc)

        # Component 2: Treasury positioning (-30 weight, inverse)
        try:
            tn_pct = self._get_net_pct_oi("ZN", "managed_money")
            if tn_pct is not None:
                tn_contribution = np.clip(-tn_pct * 1.5, -30, 30)  # spec long bonds = risk-off
                risk_score += tn_contribution
                result["components"]["treasury_spec_net"] = round(tn_pct, 2)
                result["components"]["treasury_contribution"] = round(tn_contribution, 1)
        except Exception as exc:
            logger.debug("Risk appetite treasury error: %s", exc)

        # Component 3: Gold safe-haven score (-15 weight)
        try:
            gold_signal = self.gold_safe_haven_signal()
            gold_score  = gold_signal.get("safe_haven_score", 0)
            gold_contribution = np.clip(-gold_score * 5, -15, 15)  # demand = risk-off
            risk_score += gold_contribution
            result["components"]["gold_safe_haven_score"] = gold_score
            result["components"]["gold_contribution"] = round(gold_contribution, 1)
        except Exception as exc:
            logger.debug("Risk appetite gold error: %s", exc)

        # Component 4: VIX futures — managed money net (spec long VIX = fear = risk-off)
        try:
            vix_pct = self._get_net_pct_oi("VX", "managed_money")
            if vix_pct is not None:
                vix_contribution = np.clip(-vix_pct * 2, -15, 15)
                risk_score += vix_contribution
                result["components"]["vix_spec_net"] = round(vix_pct, 2)
                result["components"]["vix_contribution"] = round(vix_contribution, 1)
        except Exception as exc:
            logger.debug("Risk appetite VIX error: %s", exc)

        risk_score = float(np.clip(risk_score, -100, 100))

        if risk_score > 40:
            regime = "RISK_ON"
            description = "Speculators positioned for growth; equity longs, treasury shorts"
        elif risk_score > 15:
            regime = "MILD_RISK_ON"
            description = "Moderate bullish positioning across risk assets"
        elif risk_score < -40:
            regime = "RISK_OFF"
            description = "Defensive positioning: bond longs, equity shorts, gold demand"
        elif risk_score < -15:
            regime = "MILD_RISK_OFF"
            description = "Some defensive positioning building"
        else:
            regime = "NEUTRAL"
            description = "No strong directional bias from COT positioning"

        result["composite_score"] = round(risk_score, 1)
        result["regime"]          = regime
        result["description"]     = description

        return result


# ---------------------------------------------------------------------------
# COTBacktestEngine
# ---------------------------------------------------------------------------

class COTBacktestEngine:
    """
    Backtests COT signals against historical price returns.

    Methodology:
      - COT data released every Friday; trade execution assumed Monday open
      - Signal: COT Index > threshold → long signal; < (100-threshold) → short
      - Forward returns computed at N-week horizon
      - IC = Spearman rank correlation (COT Index, forward_return)

    Note: Price data must be supplied via price_series or fetched externally.
    This engine works with whatever price series is provided; internal synthetic
    prices are used when none are supplied (for CI/testing purposes only).
    """

    def __init__(
        self,
        engine: Optional[COTSignalEngine] = None,
        detector: Optional[COTExtremeDetector] = None,
    ) -> None:
        self._engine   = engine   or COTSignalEngine()
        self._detector = detector or COTExtremeDetector(engine=self._engine)

    def _get_cot_series(self, contract: str, trader_type: str) -> pd.Series:
        """Return 5-year COT Index series indexed by date."""
        return self._detector.compute_cot_index_5yr(contract, trader_type)

    def _align_returns(
        self,
        cot_series: pd.Series,
        price_series: pd.Series,
        forward_weeks: int,
    ) -> pd.DataFrame:
        """
        Align COT Index with forward price returns.

        COT is released Friday; we assume Monday entry (1 business day lag).
        Returns DataFrame with columns: date, cot_index, forward_return.
        """
        if cot_series.empty or price_series.empty:
            return pd.DataFrame()

        # Resample price to weekly (Friday close)
        price_weekly = price_series.resample("W-FRI").last().dropna()
        price_fwd = price_weekly.shift(-forward_weeks)
        fwd_return = (price_fwd / price_weekly - 1.0).rename("forward_return")

        # Align COT with price (both on weekly Friday schedule)
        cot_df = cot_series.rename("cot_index").to_frame()
        combined = cot_df.join(fwd_return, how="inner").dropna()
        combined.index.name = "date"
        return combined.reset_index()

    def _synthetic_price(self, contract: str, cot_series: pd.Series) -> pd.Series:
        """
        Generate a weakly correlated synthetic price series for testing.
        In production, wire this to the SDS price database.
        """
        np.random.seed(abs(hash(contract)) % 2**31)
        n = len(cot_series)
        if n == 0:
            return pd.Series(dtype=float)

        # Slight negative correlation for commodity commercials (commercial long → price fell)
        noise  = np.random.normal(0, 0.02, n)
        signal = (cot_series.values - 50) / 50 * -0.01  # very weak signal
        log_returns = signal + noise
        prices = 100 * np.exp(np.cumsum(log_returns))
        return pd.Series(prices, index=cot_series.index)

    def run_backtest(
        self,
        contract: str,
        trader_type: str = "managed_money",
        forward_weeks: int = 4,
        price_series: Optional[pd.Series] = None,
        long_threshold:  float = 75.0,
        short_threshold: float = 25.0,
    ) -> BacktestResult:
        """
        Run a full COT signal backtest.

        Parameters
        ----------
        contract : str
            Futures contract symbol.
        trader_type : str
            COT trader category.
        forward_weeks : int
            Forecast horizon in weeks.
        price_series : pd.Series, optional
            Weekly price series indexed by date. If None, synthetic prices are used.
        long_threshold : float
            COT Index level above which a long signal is generated.
        short_threshold : float
            COT Index level below which a short signal is generated.

        Returns
        -------
        BacktestResult dataclass with IC, hit rate, and Sharpe estimate.
        """
        cot_series = self._get_cot_series(contract, trader_type)
        if cot_series.empty or len(cot_series) < _MIN_OBSERVATIONS:
            return BacktestResult(
                contract=contract, trader_type=trader_type,
                forward_weeks=forward_weeks, n_observations=len(cot_series),
                ic_spearman=0.0, ic_pvalue=1.0, ic_significant=False,
                long_avg_return=0.0, short_avg_return=0.0, long_short_spread=0.0,
                hit_rate_long=0.0, hit_rate_short=0.0,
                max_drawdown_strategy=0.0, sharpe_estimate=0.0, best_threshold=50.0,
                signal_decay_weeks=forward_weeks, notes="Insufficient COT history",
            )

        prices = price_series if price_series is not None else self._synthetic_price(contract, cot_series)
        aligned = self._align_returns(cot_series, prices, forward_weeks)

        if aligned.empty or len(aligned) < _MIN_OBSERVATIONS:
            return BacktestResult(
                contract=contract, trader_type=trader_type,
                forward_weeks=forward_weeks, n_observations=len(aligned),
                ic_spearman=0.0, ic_pvalue=1.0, ic_significant=False,
                long_avg_return=0.0, short_avg_return=0.0, long_short_spread=0.0,
                hit_rate_long=0.0, hit_rate_short=0.0,
                max_drawdown_strategy=0.0, sharpe_estimate=0.0, best_threshold=50.0,
                signal_decay_weeks=forward_weeks, notes="Alignment failed",
            )

        cot_arr = aligned["cot_index"].values
        ret_arr = aligned["forward_return"].values

        # IC: Spearman rank correlation
        ic_val, ic_pval = stats.spearmanr(cot_arr, ret_arr, nan_policy="omit")
        ic_significant = bool(ic_pval < 0.10 and abs(ic_val) > 0.10)

        # Long / short signal buckets
        long_mask  = cot_arr >= long_threshold
        short_mask = cot_arr <= short_threshold

        long_returns  = ret_arr[long_mask]
        short_returns = ret_arr[short_mask]

        long_avg  = float(np.mean(long_returns))  if len(long_returns)  > 0 else 0.0
        short_avg = float(np.mean(short_returns)) if len(short_returns) > 0 else 0.0
        spread    = long_avg - short_avg  # if COT index is predictive, this should be positive

        hit_rate_long  = float(np.mean(long_returns  > 0)) if len(long_returns)  > 0 else 0.5
        hit_rate_short = float(np.mean(short_returns < 0)) if len(short_returns) > 0 else 0.5

        # Simulate long-short strategy: long when COT > threshold, short otherwise
        strategy_positions = np.where(cot_arr >= long_threshold, 1.0,
                             np.where(cot_arr <= short_threshold, -1.0, 0.0))
        strategy_returns = strategy_positions * ret_arr
        strategy_returns = strategy_returns[strategy_returns != 0]

        # Max drawdown (simple cumulative)
        if len(strategy_returns) > 1:
            cumulative = np.cumprod(1 + strategy_returns)
            rolling_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_dd = float(np.min(drawdowns))
        else:
            max_dd = 0.0

        sharpe = (
            float(np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(52))
            if len(strategy_returns) > 2 and np.std(strategy_returns) > 0 else 0.0
        )

        # Signal decay: find peak IC at different horizons (1-8 weeks) — simplified here
        best_threshold = _compute_best_threshold(cot_arr, ret_arr)

        notes = (
            f"IC={ic_val:.3f} p={ic_pval:.3f}; "
            f"N={len(aligned)} obs; "
            f"long avg={long_avg:.3%} short avg={short_avg:.3%}; "
            f"{'SIGNIFICANT' if ic_significant else 'not significant'}"
        )

        return BacktestResult(
            contract=contract,
            trader_type=trader_type,
            forward_weeks=forward_weeks,
            n_observations=len(aligned),
            ic_spearman=round(float(ic_val), 4) if not np.isnan(ic_val) else 0.0,
            ic_pvalue=round(float(ic_pval), 4) if not np.isnan(ic_pval) else 1.0,
            ic_significant=ic_significant,
            long_avg_return=round(long_avg, 5),
            short_avg_return=round(short_avg, 5),
            long_short_spread=round(spread, 5),
            hit_rate_long=round(hit_rate_long, 3),
            hit_rate_short=round(hit_rate_short, 3),
            max_drawdown_strategy=round(max_dd, 4),
            sharpe_estimate=round(sharpe, 3),
            best_threshold=best_threshold,
            signal_decay_weeks=forward_weeks,
            notes=notes,
        )

    def compute_ic(
        self,
        contract: str,
        trader_type: str = "managed_money",
        forward_weeks: int = 4,
        price_series: Optional[pd.Series] = None,
    ) -> float:
        """
        Compute the Spearman IC for a single contract/trader_type at a given horizon.

        Returns the IC value (float), or 0.0 if insufficient data.
        """
        result = self.run_backtest(
            contract, trader_type, forward_weeks, price_series=price_series
        )
        return result.ic_spearman

    def rank_signals_by_ic(
        self,
        contracts: Optional[list[str]] = None,
        trader_type: str = "managed_money",
        forward_weeks: int = 4,
    ) -> pd.DataFrame:
        """
        Rank multiple markets by their COT signal IC (predictive power).

        Returns a DataFrame sorted by |IC| descending, with statistical significance flags.
        """
        targets = contracts or (
            _EQUITY_CONTRACTS[:4] + _RATES_CONTRACTS[:4] +
            _FX_CONTRACTS[:4] + _ENERGY_CONTRACTS[:3] +
            _METALS_CONTRACTS[:3] + _AG_CONTRACTS[:4]
        )

        rows = []
        for contract in targets:
            try:
                result = self.run_backtest(contract, trader_type, forward_weeks)
                rows.append({
                    "contract":           contract,
                    "market_name":        FUTURES_MARKET_MAP.get(contract, contract),
                    "trader_type":        trader_type,
                    "forward_weeks":      forward_weeks,
                    "ic_spearman":        result.ic_spearman,
                    "ic_pvalue":          result.ic_pvalue,
                    "ic_significant":     result.ic_significant,
                    "n_observations":     result.n_observations,
                    "long_short_spread":  result.long_short_spread,
                    "hit_rate_long":      result.hit_rate_long,
                    "sharpe_estimate":    result.sharpe_estimate,
                    "long_avg_return":    result.long_avg_return,
                    "short_avg_return":   result.short_avg_return,
                })
            except Exception as exc:
                logger.warning("IC ranking failed for %s: %s", contract, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["abs_ic"] = df["ic_spearman"].abs()
            df = df.sort_values("abs_ic", ascending=False).reset_index(drop=True)
            df = df.drop(columns=["abs_ic"])
        return df

    def plot_signal_vs_return(
        self,
        contract: str,
        trader_type: str = "managed_money",
        forward_weeks: int = 4,
        price_series: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Return data suitable for plotting COT Index vs forward return scatter.

        Returns a dict with x (COT Index) and y (forward return) lists,
        plus regression line parameters and IC annotation.
        """
        cot_series = self._get_cot_series(contract, trader_type)
        prices = price_series if price_series is not None else self._synthetic_price(contract, cot_series)
        aligned = self._align_returns(cot_series, prices, forward_weeks)

        if aligned.empty:
            return {"contract": contract, "status": "no_data"}

        x = aligned["cot_index"].tolist()
        y = (aligned["forward_return"] * 100).tolist()  # convert to %

        # Linear regression for trendline
        slope, intercept, r_val, p_val, _ = stats.linregress(
            aligned["cot_index"].values,
            aligned["forward_return"].values,
        )

        ic_val, ic_pval = stats.spearmanr(x, y, nan_policy="omit")

        return {
            "contract":        contract,
            "market_name":     FUTURES_MARKET_MAP.get(contract, contract),
            "trader_type":     trader_type,
            "forward_weeks":   forward_weeks,
            "x_cot_index":     x,
            "y_fwd_return_pct": y,
            "dates":           aligned["date"].astype(str).tolist(),
            "regression": {
                "slope":      round(float(slope), 6),
                "intercept":  round(float(intercept), 6),
                "r_squared":  round(float(r_val**2), 4),
            },
            "ic_spearman":   round(float(ic_val) if not np.isnan(ic_val) else 0.0, 4),
            "ic_pvalue":     round(float(ic_pval) if not np.isnan(ic_pval) else 1.0, 4),
            "n_observations": len(x),
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_best_threshold(cot_arr: np.ndarray, ret_arr: np.ndarray) -> float:
    """
    Find the COT Index threshold that maximises the long-short return spread.
    Searches thresholds from 30 to 70 in steps of 5.
    """
    best_spread = -np.inf
    best_threshold = 50.0
    for thresh in range(30, 75, 5):
        long_ret  = ret_arr[cot_arr >= thresh]
        short_ret = ret_arr[cot_arr <= (100 - thresh)]
        if len(long_ret) < 5 or len(short_ret) < 5:
            continue
        spread = float(np.mean(long_ret)) - float(np.mean(short_ret))
        if spread > best_spread:
            best_spread = spread
            best_threshold = float(thresh)
    return best_threshold


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Query, HTTPException
    from pydantic import BaseModel

    cot_v2_router = APIRouter(prefix="/cot/v2", tags=["CFTC COT v2"])

    # Singleton instances
    _adapter_v2  = COTDataAdapter()
    _engine_v2   = COTSignalEngine(adapter=_adapter_v2)
    _tff_v2      = TFFReportAdapter()
    _detector_v2 = COTExtremeDetector(engine=_engine_v2)
    _macro_v2    = MacroCOTSignalEngine(engine=_engine_v2, tff=_tff_v2, detector=_detector_v2)
    _backtest_v2 = COTBacktestEngine(engine=_engine_v2, detector=_detector_v2)

    # ── Response models ──────────────────────────────────────────────────────

    class TFFPositionResponse(BaseModel):
        market: str
        as_of: str
        dealer_net: float
        dealer_pct_oi: float
        am_net: float
        am_pct_oi: float
        lev_net: float
        lev_pct_oi: float
        other_net: float
        open_interest: float
        am_trend_signal: str
        lev_trend_signal: str

    class ExtremeReadingResponse(BaseModel):
        contract: str
        trader_type: str
        cot_index_5yr: float
        net_percentile: float
        is_extreme_long: bool
        is_extreme_short: bool
        is_crowded: bool
        crowding_direction: str
        signal: str
        contrarian_strength: float
        as_of: str

    class BacktestResponse(BaseModel):
        contract: str
        trader_type: str
        forward_weeks: int
        n_observations: int
        ic_spearman: float
        ic_pvalue: float
        ic_significant: bool
        long_avg_return: float
        short_avg_return: float
        long_short_spread: float
        hit_rate_long: float
        hit_rate_short: float
        sharpe_estimate: float
        best_threshold: float
        notes: str

    # ── Endpoints ────────────────────────────────────────────────────────────

    @cot_v2_router.get("/positions/{market}")
    def api_v2_positions(
        market: str,
        trader_type: str = Query("managed_money", description="Trader type for net positioning"),
        weeks: int = Query(52, ge=4, le=520, description="History lookback in weeks"),
    ):
        """
        Full COT net positioning for a futures market.

        Returns latest snapshot plus recent history.
        """
        contract = market.upper()
        df = _engine_v2.compute_net_positioning(contract, trader_type=trader_type)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No COT data for {contract}")

        cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=weeks)
        df = df[df["date"] >= cutoff]
        latest = df.iloc[-1].to_dict()
        latest["date"] = str(latest.get("date", ""))

        reading = _detector_v2.classify_extreme(contract, trader_type)

        return {
            "latest": latest,
            "extreme_reading": {
                "cot_index_5yr":       reading.cot_index_5yr,
                "net_percentile":      reading.net_percentile,
                "is_extreme_long":     reading.is_extreme_long,
                "is_extreme_short":    reading.is_extreme_short,
                "is_crowded":          reading.is_crowded,
                "crowding_direction":  reading.crowding_direction,
                "signal":              reading.signal,
                "contrarian_strength": reading.contrarian_strength,
            },
            "history": [
                {**r, "date": str(r.get("date", ""))}
                for r in df.tail(52).to_dict(orient="records")
            ],
        }

    @cot_v2_router.get("/extremes")
    def api_v2_extremes(
        trader_type: str = Query("managed_money"),
        threshold: float = Query(10.0, ge=1.0, le=30.0),
        contracts: str = Query("", description="Comma-separated list; empty = all major"),
    ):
        """
        Scan all (or specified) markets for extreme COT positioning and crowding.

        Returns only contracts with is_crowded=True, sorted by contrarian strength.
        """
        contract_list = [c.strip().upper() for c in contracts.split(",") if c.strip()] or None
        df = _detector_v2.detect_crowding(
            contracts=contract_list, trader_type=trader_type, extreme_threshold=threshold
        )
        return {
            "count": len(df),
            "trader_type": trader_type,
            "threshold": threshold,
            "crowded_trades": df.to_dict(orient="records"),
        }

    @cot_v2_router.get("/signals")
    def api_v2_signals():
        """
        Global macro COT signals: USD, gold, oil, treasury duration, and composite.
        """
        return {
            "usd_positioning":      _macro_v2.usd_positioning_signal(),
            "gold_safe_haven":      _macro_v2.gold_safe_haven_signal(),
            "oil_supply_demand":    _macro_v2.oil_supply_demand_signal(),
            "treasury_duration":    _macro_v2.treasury_duration_signal(),
            "risk_appetite":        _macro_v2.global_risk_appetite_composite(),
        }

    @cot_v2_router.get("/tff/{market}")
    def api_v2_tff(
        market: str,
        lookback_weeks: int = Query(52, ge=4, le=260),
    ):
        """
        Traders in Financial Futures (TFF) disaggregated positioning.

        Returns latest snapshot and history of dealer / asset manager /
        leveraged fund net positions.
        """
        contract = market.upper()
        latest = _tff_v2.get_tff_positions(contract)
        history = _tff_v2.get_tff_history(contract, lookback_weeks=lookback_weeks)
        comparison = _tff_v2.compare_am_vs_lev(contract, lookback_weeks=lookback_weeks)

        hist_records = []
        if not history.empty:
            history["date"] = history["date"].astype(str)
            hist_records = history.tail(52).to_dict(orient="records")

        comp_records = []
        if not comparison.empty:
            comparison["date"] = comparison["date"].astype(str)
            comp_records = comparison.tail(52).to_dict(orient="records")

        return {
            "latest": {
                "market":           latest.market,
                "as_of":            latest.as_of,
                "dealer_net":       latest.dealer_net,
                "dealer_pct_oi":    latest.dealer_pct_oi,
                "am_net":           latest.am_net,
                "am_pct_oi":        latest.am_pct_oi,
                "lev_net":          latest.lev_net,
                "lev_pct_oi":       latest.lev_pct_oi,
                "other_net":        latest.other_net,
                "open_interest":    latest.open_interest,
                "am_trend_signal":  latest.am_trend_signal,
                "lev_trend_signal": latest.lev_trend_signal,
            },
            "history":    hist_records,
            "am_vs_lev":  comp_records,
        }

    @cot_v2_router.get("/backtest")
    def api_v2_backtest(
        contracts: str = Query("", description="Comma-separated; empty = major markets"),
        trader_type: str = Query("managed_money"),
        forward_weeks: int = Query(4, ge=1, le=26),
    ):
        """
        Rank COT signals by historical predictive power (Information Coefficient).

        Returns all markets sorted by |IC| with significance flags.
        Note: Uses synthetic prices for demonstration when live price data
        is not wired up; connect price_series via the BacktestEngine directly
        for production accuracy.
        """
        contract_list = [c.strip().upper() for c in contracts.split(",") if c.strip()] or None
        df = _backtest_v2.rank_signals_by_ic(
            contracts=contract_list,
            trader_type=trader_type,
            forward_weeks=forward_weeks,
        )
        return {
            "trader_type":    trader_type,
            "forward_weeks":  forward_weeks,
            "n_markets_tested": len(df),
            "significant_count": int(df["ic_significant"].sum()) if not df.empty else 0,
            "rankings": df.to_dict(orient="records"),
        }

    @cot_v2_router.get("/heatmap")
    def api_v2_heatmap(
        contracts: str = Query("", description="Comma-separated contract codes"),
    ):
        """
        COT Index heatmap: contracts × trader_types matrix.

        Values 0-100: 0 = max short (5yr), 100 = max long (5yr).
        """
        contract_list = [c.strip().upper() for c in contracts.split(",") if c.strip()] or None
        df = _detector_v2.get_positioning_heatmap(contracts=contract_list)
        if df.empty:
            return {"status": "no_data"}
        return {
            "contracts":   df.index.tolist(),
            "trader_types": df.columns.tolist(),
            "data": df.round(1).to_dict(orient="index"),
        }

    @cot_v2_router.get("/backtest/{contract}")
    def api_v2_backtest_single(
        contract: str,
        trader_type: str = Query("managed_money"),
        forward_weeks: int = Query(4, ge=1, le=26),
        long_threshold: float = Query(75.0, ge=50.0, le=95.0),
        short_threshold: float = Query(25.0, ge=5.0, le=50.0),
    ):
        """
        Detailed backtest for a single contract including scatter data for charting.
        """
        c = contract.upper()
        result = _backtest_v2.run_backtest(
            c, trader_type, forward_weeks,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )
        scatter = _backtest_v2.plot_signal_vs_return(c, trader_type, forward_weeks)
        return {
            "backtest": {
                "contract":           result.contract,
                "trader_type":        result.trader_type,
                "forward_weeks":      result.forward_weeks,
                "n_observations":     result.n_observations,
                "ic_spearman":        result.ic_spearman,
                "ic_pvalue":          result.ic_pvalue,
                "ic_significant":     result.ic_significant,
                "long_avg_return":    result.long_avg_return,
                "short_avg_return":   result.short_avg_return,
                "long_short_spread":  result.long_short_spread,
                "hit_rate_long":      result.hit_rate_long,
                "hit_rate_short":     result.hit_rate_short,
                "max_drawdown_strategy": result.max_drawdown_strategy,
                "sharpe_estimate":    result.sharpe_estimate,
                "best_threshold":     result.best_threshold,
                "notes":              result.notes,
            },
            "scatter_data": scatter,
        }

except ImportError:
    cot_v2_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — cot_v2_router not registered")
