"""
CFTC COT Positioning — Comprehensive Intelligence Module (dim_046, target 9+).

Downloads and analyses all four CFTC Commitments of Traders report types
(Legacy, Disaggregated, Financial, Traders-in-Financial-Futures).
Provides COT Index, extreme-positioning detection, macro sentiment signals,
and a full positioning dashboard across 60+ futures markets.

Public API
----------
COTDataAdapter
    download_cot_report(report_type, year)   -> pd.DataFrame
    get_latest_cot()                          -> pd.DataFrame
    get_cot_history(contract, lookback_weeks) -> pd.DataFrame

COTSignalEngine
    compute_net_positioning(contract, trader_type)          -> pd.DataFrame
    compute_cot_index(contract, trader_type, lookback_weeks)-> pd.Series
    detect_extremes(contract, trader_type, threshold)       -> dict
    compute_large_trader_ratio(contract)                    -> pd.DataFrame
    get_positioning_dashboard(contracts)                    -> pd.DataFrame
    compute_trend_following_signal(contract, price_series)  -> dict

COTMacroSignals
    equity_futures_sentiment()            -> dict
    rates_futures_positioning()           -> dict
    fx_positioning_summary()              -> pd.DataFrame
    commodity_positioning_summary(cat)   -> pd.DataFrame

cot_router  — FastAPI router, prefix /api/cot
"""
from __future__ import annotations

import io
import os
import asyncio
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CFTC_BASE = "https://www.cftc.gov/files/dea/history"
_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
}
_TIMEOUT = 120.0

# Local cache directory
_CACHE_DIR = Path(os.environ.get("SENTINEL_CACHE_DIR", ".sentinel/cache")) / "cot"

# CFTC filename patterns per report type
_COT_URL_PATTERNS: dict[str, str] = {
    "legacy":                f"{_CFTC_BASE}/com_disagg_txt_hist_{{year}}.zip",
    "disaggregated":         f"{_CFTC_BASE}/fut_disagg_txt_hist_2006_{{year}}.zip",
    "financial":             f"{_CFTC_BASE}/fin_fut_txt_hist_{{year}}.zip",
    "traders_in_financial":  f"{_CFTC_BASE}/traders_in_financial_futures_fut_hist_{{year}}.zip",
}

# Latest-year URLs (covers the current and recent years in one file)
_COT_LATEST_PATTERNS: dict[str, str] = {
    "legacy":                f"{_CFTC_BASE}/com_disagg_txt_2016_2025.zip",
    "disaggregated":         f"{_CFTC_BASE}/fut_disagg_txt_2016_2025.zip",
    "financial":             f"{_CFTC_BASE}/fin_fut_txt_2016_2025.zip",
    "traders_in_financial":  f"{_CFTC_BASE}/traders_fin_fut_txt_2016_2025.zip",
}

_VALID_REPORT_TYPES = frozenset(_COT_URL_PATTERNS.keys())

# ---------------------------------------------------------------------------
# FUTURES_MARKET_MAP — 60+ contracts: symbol → CFTC market name substring
# ---------------------------------------------------------------------------

FUTURES_MARKET_MAP: dict[str, str] = {
    # ── Equity ──────────────────────────────────────────────────────────────
    "ES":  "E-MINI S&P 500",
    "NQ":  "E-MINI NASDAQ-100",
    "RTY": "E-MINI RUSSELL 2000",
    "YM":  "MINI DOW JONES",
    "VX":  "CBOE VOLATILITY INDEX",
    "NKD": "NIKKEI STOCK AVERAGE",
    # ── Rates ────────────────────────────────────────────────────────────────
    "ZT":  "2-YEAR U.S. TREASURY",
    "ZF":  "5-YEAR U.S. TREASURY",
    "ZN":  "10-YEAR U.S. TREASURY",
    "ZB":  "U.S. TREASURY BONDS",
    "UB":  "ULTRA U.S. TREASURY BONDS",
    "FF":  "30-DAY FEDERAL FUNDS",
    "GE":  "EURODOLLAR",
    "SR3": "3-MONTH SOFR",
    # ── FX ───────────────────────────────────────────────────────────────────
    "6E":  "EURO FX",
    "6B":  "BRITISH POUND STERLING",
    "6J":  "JAPANESE YEN",
    "6C":  "CANADIAN DOLLAR",
    "6A":  "AUSTRALIAN DOLLAR",
    "6S":  "SWISS FRANC",
    "6M":  "MEXICAN PESO",
    "6L":  "BRAZILIAN REAL",
    "6N":  "NEW ZEALAND DOLLAR",
    "6Z":  "SOUTH AFRICAN RAND",
    "6R":  "RUSSIAN RUBLE",
    "CNH": "CHINESE RENMINBI",
    # ── Energy ───────────────────────────────────────────────────────────────
    "CL":  "CRUDE OIL, LIGHT SWEET",
    "BZ":  "BRENT CRUDE OIL",
    "NG":  "NATURAL GAS",
    "HO":  "NO. 2 HEATING OIL",
    "RB":  "RBOB GASOLINE",
    "CL3": "WTI-BRENT SPREAD",
    "PA":  "PROPANE",
    # ── Metals ───────────────────────────────────────────────────────────────
    "GC":  "GOLD",
    "SI":  "SILVER",
    "HG":  "COPPER",
    "PL":  "PLATINUM",
    "PA":  "PALLADIUM",
    "ALI": "ALUMINUM",
    # ── Agriculture ──────────────────────────────────────────────────────────
    "ZC":  "CORN",
    "ZW":  "WHEAT-SRW",
    "KE":  "WHEAT-HRW",
    "ZS":  "SOYBEANS",
    "ZL":  "SOYBEAN OIL",
    "ZM":  "SOYBEAN MEAL",
    "SB":  "SUGAR NO. 11",
    "KC":  "COFFEE C",
    "CT":  "COTTON NO. 2",
    "CC":  "COCOA",
    "OJ":  "FROZEN CONCENTRATED ORANGE JUICE",
    "LH":  "LEAN HOGS",
    "LE":  "LIVE CATTLE",
    "FC":  "FEEDER CATTLE",
    # ── Crypto ───────────────────────────────────────────────────────────────
    "BTC": "BITCOIN",
    "ETH": "ETHER",
    "MBT": "MICRO BITCOIN",
}

# Sector category groupings for macro signals
_EQUITY_CONTRACTS  = ["ES", "NQ", "RTY", "YM"]
_RATES_CONTRACTS   = ["ZT", "ZF", "ZN", "ZB", "UB", "FF", "GE"]
_FX_CONTRACTS      = ["6E", "6B", "6J", "6C", "6A", "6S", "6M", "6L"]
_ENERGY_CONTRACTS  = ["CL", "BZ", "NG", "HO", "RB"]
_METALS_CONTRACTS  = ["GC", "SI", "HG", "PL"]
_AG_CONTRACTS      = ["ZC", "ZW", "ZS", "ZL", "SB", "KC", "CT", "CC"]

# ---------------------------------------------------------------------------
# Column name mappings per report type (CFTC uses different column names)
# ---------------------------------------------------------------------------

_LEGACY_COLS = {
    "date":          "Report_Date_as_YYYY-MM-DD",
    "market":        "Market_and_Exchange_Names",
    "comm_long":     "Comm_Positions_Long_All",
    "comm_short":    "Comm_Positions_Short_All",
    "noncomm_long":  "NonComm_Positions_Long_All",
    "noncomm_short": "NonComm_Positions_Short_All",
    "nonrep_long":   "NonRept_Positions_Long_All",
    "nonrep_short":  "NonRept_Positions_Short_All",
    "oi":            "Open_Interest_All",
}

_DISAGG_COLS = {
    "date":         "As of Date in Form YYYY-MM-DD",
    "market":       "Market_and_Exchange_Names",
    "prod_long":    "Prod_Merc_Positions_Long_All",
    "prod_short":   "Prod_Merc_Positions_Short_All",
    "swap_long":    "Swap_Positions_Long_All",
    "swap_short":   "Swap_Positions_Short_All",
    "mm_long":      "M_Money_Positions_Long_All",
    "mm_short":     "M_Money_Positions_Short_All",
    "other_long":   "Other_Rept_Positions_Long_All",
    "other_short":  "Other_Rept_Positions_Short_All",
    "nonrep_long":  "NonRept_Positions_Long_All",
    "nonrep_short": "NonRept_Positions_Short_All",
    "oi":           "Open_Interest_All",
}

_FINANCIAL_COLS = {
    "date":           "As of Date in Form YYYY-MM-DD",
    "market":         "Market_and_Exchange_Names",
    "dealer_long":    "Dealer_Positions_Long_All",
    "dealer_short":   "Dealer_Positions_Short_All",
    "am_long":        "Asset_Mgr_Positions_Long_All",
    "am_short":       "Asset_Mgr_Positions_Short_All",
    "lev_long":       "Lev_Money_Positions_Long_All",
    "lev_short":      "Lev_Money_Positions_Short_All",
    "other_long":     "Other_Rept_Positions_Long_All",
    "other_short":    "Other_Rept_Positions_Short_All",
    "nonrep_long":    "NonRept_Positions_Long_All",
    "nonrep_short":   "NonRept_Positions_Short_All",
    "oi":             "Open_Interest_All",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(report_type: str, year: int) -> Path:
    return _CACHE_DIR / f"{report_type}_{year}.parquet"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name from *candidates*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _parse_date_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce")
    return pd.Series([pd.NaT] * len(df))


def _filter_market(df: pd.DataFrame, market_substring: str) -> pd.DataFrame:
    """Case-insensitive substring filter on the CFTC market name column."""
    market_col = _find_col(df, ["Market_and_Exchange_Names", "market"])
    if market_col is None:
        return pd.DataFrame()
    mask = df[market_col].str.upper().str.contains(
        market_substring.upper(), na=False, regex=False
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# COTDataAdapter
# ---------------------------------------------------------------------------


class COTDataAdapter:
    """
    Downloads, caches, and parses CFTC Commitments of Traders reports.

    report_type options
    -------------------
    "legacy"               — Commercial / NonCommercial / NonReportable
    "disaggregated"        — Producer / SwapDealer / ManagedMoney / Other
    "financial"            — Dealer / AssetManager / LeveragedMoney / Other
    "traders_in_financial" — Concentration ratios in financial futures
    """

    def __init__(self) -> None:
        self._frames: dict[str, dict[int, pd.DataFrame]] = {}
        _ensure_cache_dir()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, report_type: str, year: int) -> str:
        pattern = _COT_URL_PATTERNS.get(report_type, _COT_URL_PATTERNS["disaggregated"])
        return pattern.format(year=year)

    def _latest_url(self, report_type: str) -> str:
        return _COT_LATEST_PATTERNS.get(report_type, _COT_LATEST_PATTERNS["disaggregated"])

    async def _download_zip(self, url: str) -> pd.DataFrame:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        # CFTC ZIPs contain a single CSV/TXT file
        txt_name = next(
            (n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))),
            zf.namelist()[0],
        )
        with zf.open(txt_name) as f:
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")

        df.columns = [c.strip() for c in df.columns]
        return df

    def _load_from_cache(self, report_type: str, year: int) -> Optional[pd.DataFrame]:
        p = _cache_path(report_type, year)
        if p.exists():
            age_hours = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
            # Keep historical years cached indefinitely; refresh current year every 12h
            if year < date.today().year or age_hours < 12:
                try:
                    return pd.read_parquet(p)
                except Exception as exc:
                    logger.warning("Cache read failed", path=str(p), error=str(exc))
        return None

    def _save_to_cache(self, df: pd.DataFrame, report_type: str, year: int) -> None:
        p = _cache_path(report_type, year)
        try:
            df.to_parquet(p, index=False)
        except Exception as exc:
            logger.warning("Cache write failed", path=str(p), error=str(exc))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def download_cot_report(
        self, report_type: str = "legacy", year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download and return a CFTC COT report.

        Parameters
        ----------
        report_type : str
            One of "legacy", "disaggregated", "financial", "traders_in_financial".
        year : int, optional
            Calendar year. Defaults to current year (uses the multi-year bundle).
        """
        if report_type not in _VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid report_type '{report_type}'. Choose from: {sorted(_VALID_REPORT_TYPES)}"
            )

        target_year = year or date.today().year

        # 1. Try in-memory cache
        if report_type in self._frames and target_year in self._frames[report_type]:
            return self._frames[report_type][target_year]

        # 2. Try disk cache
        cached = self._load_from_cache(report_type, target_year)
        if cached is not None:
            self._frames.setdefault(report_type, {})[target_year] = cached
            return cached

        # 3. Download
        url = (
            self._latest_url(report_type)
            if target_year == date.today().year
            else self._url(report_type, target_year)
        )
        logger.info("Downloading COT report", report_type=report_type, year=target_year, url=url)

        try:
            df = asyncio.run(self._download_zip(url))
            self._save_to_cache(df, report_type, target_year)
            self._frames.setdefault(report_type, {})[target_year] = df
            logger.info("COT report downloaded", report_type=report_type, rows=len(df))
            return df
        except Exception as exc:
            logger.error(
                "COT download failed", report_type=report_type, year=target_year, error=str(exc)
            )
            return pd.DataFrame()

    def get_latest_cot(self, report_type: str = "disaggregated") -> pd.DataFrame:
        """Download the most recent weekly COT report (current-year bundle)."""
        return self.download_cot_report(report_type=report_type, year=None)

    def get_cot_history(
        self,
        contract: str,
        lookback_weeks: int = 104,
        report_type: str = "disaggregated",
    ) -> pd.DataFrame:
        """
        Return a time-series DataFrame for *contract* spanning *lookback_weeks*.

        The *contract* string is matched as a case-insensitive substring against
        CFTC market names (e.g. "S&P 500", "GOLD", "EURO FX").
        """
        market_name = FUTURES_MARKET_MAP.get(contract.upper(), contract)
        years_needed = max(1, lookback_weeks // 52 + 1)
        current_year = date.today().year

        frames = []
        for y in range(current_year - years_needed + 1, current_year + 1):
            df = self.download_cot_report(report_type=report_type, year=y)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = _filter_market(combined, market_name)

        date_col = _find_col(
            combined,
            ["As of Date in Form YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD", "date"],
        )
        if date_col:
            combined["date"] = pd.to_datetime(combined[date_col], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=lookback_weeks)
            combined = combined[combined["date"] >= cutoff]
            combined = combined.sort_values("date").reset_index(drop=True)

        return combined


# ---------------------------------------------------------------------------
# COTSignalEngine
# ---------------------------------------------------------------------------


class COTSignalEngine:
    """
    Computes positioning signals, COT Index, and extremes from raw CFTC data.
    """

    def __init__(self, adapter: Optional[COTDataAdapter] = None) -> None:
        self._adapter = adapter or COTDataAdapter()

    # ------------------------------------------------------------------
    # Net positioning
    # ------------------------------------------------------------------

    def compute_net_positioning(
        self, contract: str, trader_type: str = "commercial"
    ) -> pd.DataFrame:
        """
        Compute net (long - short) positioning for *trader_type* as both
        an absolute figure and as a percentage of open interest.

        trader_type options
        -------------------
        commercial / noncommercial (legacy)
        managed_money / producer / swap_dealer / other_reportable (disaggregated)
        dealer / asset_manager / leveraged_money (financial)
        """
        df = self._adapter.get_cot_history(contract, lookback_weeks=260)
        if df.empty:
            return pd.DataFrame()

        date_col = _find_col(
            df, ["date", "As of Date in Form YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD"]
        )

        # Resolve column names by trader type
        tt = trader_type.lower().replace(" ", "_")
        long_candidates: list[str] = []
        short_candidates: list[str] = []

        if tt in ("commercial", "comm"):
            long_candidates  = ["Comm_Positions_Long_All", "Prod_Merc_Positions_Long_All"]
            short_candidates = ["Comm_Positions_Short_All", "Prod_Merc_Positions_Short_All"]
        elif tt in ("noncommercial", "non_commercial", "speculator"):
            long_candidates  = ["NonComm_Positions_Long_All", "M_Money_Positions_Long_All"]
            short_candidates = ["NonComm_Positions_Short_All", "M_Money_Positions_Short_All"]
        elif tt == "managed_money":
            long_candidates  = ["M_Money_Positions_Long_All"]
            short_candidates = ["M_Money_Positions_Short_All"]
        elif tt == "producer":
            long_candidates  = ["Prod_Merc_Positions_Long_All"]
            short_candidates = ["Prod_Merc_Positions_Short_All"]
        elif tt == "swap_dealer":
            long_candidates  = ["Swap_Positions_Long_All"]
            short_candidates = ["Swap_Positions_Short_All"]
        elif tt == "dealer":
            long_candidates  = ["Dealer_Positions_Long_All"]
            short_candidates = ["Dealer_Positions_Short_All"]
        elif tt == "asset_manager":
            long_candidates  = ["Asset_Mgr_Positions_Long_All"]
            short_candidates = ["Asset_Mgr_Positions_Short_All"]
        elif tt == "leveraged_money":
            long_candidates  = ["Lev_Money_Positions_Long_All"]
            short_candidates = ["Lev_Money_Positions_Short_All"]
        else:
            long_candidates  = ["NonComm_Positions_Long_All", "M_Money_Positions_Long_All"]
            short_candidates = ["NonComm_Positions_Short_All", "M_Money_Positions_Short_All"]

        long_col  = _find_col(df, long_candidates)
        short_col = _find_col(df, short_candidates)
        oi_col    = _find_col(df, ["Open_Interest_All", "oi"])

        if not long_col or not short_col:
            logger.error(
                "Could not resolve COT columns",
                trader_type=trader_type,
                available=list(df.columns[:15]),
            )
            return pd.DataFrame()

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
        result["long"]  = _to_numeric_series(df[long_col])
        result["short"] = _to_numeric_series(df[short_col])
        result["net"]   = result["long"] - result["short"]

        if oi_col:
            oi = _to_numeric_series(df[oi_col])
            result["net_pct_oi"] = (result["net"] / oi.replace(0, float("nan")) * 100).round(2)
            result["open_interest"] = oi
        else:
            result["net_pct_oi"] = float("nan")
            result["open_interest"] = float("nan")

        result["contract"]    = contract.upper()
        result["trader_type"] = trader_type
        return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # COT Index
    # ------------------------------------------------------------------

    def compute_cot_index(
        self,
        contract: str,
        trader_type: str = "commercial",
        lookback_weeks: int = 52,
    ) -> pd.Series:
        """
        COT Index = (current_net - min_net_N_wk) / (max_net_N_wk - min_net_N_wk) × 100.

        0 = maximum short over the lookback (historically bullish contrarian signal
        for commercials in commodity futures).
        100 = maximum long.

        Returns a pd.Series indexed by date.
        """
        df = self.compute_net_positioning(contract, trader_type)
        if df.empty:
            return pd.Series(dtype=float)

        net = df.set_index("date")["net"].sort_index()
        roll_max = net.rolling(window=lookback_weeks, min_periods=max(1, lookback_weeks // 4)).max()
        roll_min = net.rolling(window=lookback_weeks, min_periods=max(1, lookback_weeks // 4)).min()
        denom = (roll_max - roll_min).replace(0, float("nan"))
        cot_index = ((net - roll_min) / denom * 100).round(2)
        cot_index.name = f"cot_index_{trader_type}"
        return cot_index.dropna()

    # ------------------------------------------------------------------
    # Extreme detection
    # ------------------------------------------------------------------

    def detect_extremes(
        self,
        contract: str,
        trader_type: str = "commercial",
        threshold: float = 10.0,
    ) -> dict[str, Any]:
        """
        Flag if the current COT Index is in extreme territory.

        For commercial traders in commodity futures:
          - COT Index < threshold  → extreme short → contrarian BULLISH signal
          - COT Index > (100 - threshold) → extreme long → contrarian BEARISH signal

        Returns a dict with: is_extreme_short, is_extreme_long,
        cot_index_current, percentile_rank, signal.
        """
        idx = self.compute_cot_index(contract, trader_type=trader_type)
        if idx.empty:
            return {"contract": contract, "status": "no_data"}

        current = float(idx.iloc[-1])
        percentile_rank = float((idx <= current).mean() * 100)

        is_extreme_short = current < threshold
        is_extreme_long  = current > (100.0 - threshold)

        if is_extreme_short:
            signal = "bullish_contrarian"   # commercials maximally short → price reversal up
        elif is_extreme_long:
            signal = "bearish_contrarian"
        else:
            signal = "neutral"

        return {
            "contract": contract,
            "trader_type": trader_type,
            "cot_index_current": round(current, 2),
            "percentile_rank": round(percentile_rank, 2),
            "is_extreme_short": is_extreme_short,
            "is_extreme_long": is_extreme_long,
            "signal": signal,
            "threshold": threshold,
            "lookback_points": len(idx),
            "as_of": str(idx.index[-1].date()) if hasattr(idx.index[-1], "date") else str(idx.index[-1]),
        }

    # ------------------------------------------------------------------
    # Large trader ratio
    # ------------------------------------------------------------------

    def compute_large_trader_ratio(self, contract: str) -> pd.DataFrame:
        """
        Reportable large traders vs non-reportable small traders positioning ratio.
        Uses the disaggregated report: large = sum of all reportable categories.
        """
        df = self._adapter.get_cot_history(contract, lookback_weeks=104, report_type="disaggregated")
        if df.empty:
            return pd.DataFrame()

        date_col = _find_col(df, ["date", "As of Date in Form YYYY-MM-DD"])

        reportable_long_cols = [
            c for c in df.columns
            if "Long_All" in c and "NonRept" not in c and "Spreading" not in c
        ]
        reportable_short_cols = [
            c for c in df.columns
            if "Short_All" in c and "NonRept" not in c and "Spreading" not in c
        ]
        nonrep_long_col  = _find_col(df, ["NonRept_Positions_Long_All"])
        nonrep_short_col = _find_col(df, ["NonRept_Positions_Short_All"])

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT

        if reportable_long_cols:
            result["large_net"] = sum(
                _to_numeric_series(df[c]) for c in reportable_long_cols
            ) - sum(
                _to_numeric_series(df[c]) for c in reportable_short_cols
            )
        else:
            result["large_net"] = float("nan")

        if nonrep_long_col and nonrep_short_col:
            result["small_net"] = (
                _to_numeric_series(df[nonrep_long_col])
                - _to_numeric_series(df[nonrep_short_col])
            )
        else:
            result["small_net"] = float("nan")

        result["ratio"] = (
            result["large_net"] / result["small_net"].replace(0, float("nan"))
        ).round(3)
        result["contract"] = contract.upper()

        return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Positioning dashboard
    # ------------------------------------------------------------------

    def get_positioning_dashboard(
        self, contracts: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Snapshot table: for each contract — net commercial, net speculator,
        COT index (commercial), COT index (speculator), signal.
        """
        targets = contracts or (
            _EQUITY_CONTRACTS[:4]
            + _RATES_CONTRACTS[:4]
            + _FX_CONTRACTS[:4]
            + _ENERGY_CONTRACTS[:3]
            + _METALS_CONTRACTS[:2]
            + _AG_CONTRACTS[:3]
        )

        rows = []
        for contract in targets:
            try:
                extremes_comm = self.detect_extremes(contract, "commercial")
                extremes_spec = self.detect_extremes(contract, "managed_money")

                net_comm_df = self.compute_net_positioning(contract, "commercial")
                net_spec_df = self.compute_net_positioning(contract, "managed_money")

                net_comm = float(net_comm_df["net"].iloc[-1]) if not net_comm_df.empty else float("nan")
                net_spec = float(net_spec_df["net"].iloc[-1]) if not net_spec_df.empty else float("nan")
                net_pct  = float(net_comm_df["net_pct_oi"].iloc[-1]) if not net_comm_df.empty else float("nan")

                cot_idx_comm = extremes_comm.get("cot_index_current", float("nan"))
                cot_idx_spec = extremes_spec.get("cot_index_current", float("nan"))
                signal       = extremes_comm.get("signal", "neutral")

                rows.append(
                    {
                        "contract": contract,
                        "market_name": FUTURES_MARKET_MAP.get(contract.upper(), contract),
                        "net_commercial": round(net_comm, 0) if net_comm == net_comm else None,
                        "net_speculator": round(net_spec, 0) if net_spec == net_spec else None,
                        "net_comm_pct_oi": round(net_pct, 2) if net_pct == net_pct else None,
                        "cot_index_commercial": cot_idx_comm,
                        "cot_index_speculator": cot_idx_spec,
                        "signal": signal,
                    }
                )
            except Exception as exc:
                logger.warning("Dashboard row failed", contract=contract, error=str(exc))
                rows.append({"contract": contract, "signal": "error"})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Trend-following signal
    # ------------------------------------------------------------------

    def compute_trend_following_signal(
        self,
        contract: str,
        price_series: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Combine managed-money COT positioning with price trend (if provided).

        Rules
        -----
        - Managed money extremely long (COT Index > 90) + price uptrend
          → momentum confirmation (strong bull)
        - Managed money extremely long + price downtrend / reversal
          → crowded trade caution (bear divergence)
        - Managed money extremely short (COT Index < 10) + price downtrend
          → momentum confirmation (strong bear) — contrarian bullish setup
        """
        extremes = self.detect_extremes(contract, trader_type="managed_money")
        cot_idx  = extremes.get("cot_index_current", 50.0)

        result: dict[str, Any] = {
            "contract": contract,
            "cot_index_managed_money": cot_idx,
            "is_extreme_long": extremes.get("is_extreme_long", False),
            "is_extreme_short": extremes.get("is_extreme_short", False),
        }

        price_trend: Optional[str] = None
        if price_series is not None and len(price_series) >= 20:
            sma20 = price_series.rolling(20).mean()
            last_price = float(price_series.iloc[-1])
            last_sma   = float(sma20.iloc[-1])
            price_trend = "uptrend" if last_price > last_sma else "downtrend"
            result["price_vs_sma20"] = round((last_price / last_sma - 1) * 100, 2)

        result["price_trend"] = price_trend

        if extremes.get("is_extreme_long"):
            if price_trend == "uptrend":
                result["signal"] = "momentum_confirmation_bullish"
            elif price_trend == "downtrend":
                result["signal"] = "crowded_trade_caution_bearish_divergence"
            else:
                result["signal"] = "extreme_long_watch"
        elif extremes.get("is_extreme_short"):
            if price_trend == "downtrend":
                result["signal"] = "contrarian_bullish_setup"
            elif price_trend == "uptrend":
                result["signal"] = "short_squeeze_risk"
            else:
                result["signal"] = "extreme_short_watch"
        else:
            result["signal"] = "neutral"

        result["description"] = {
            "momentum_confirmation_bullish": "Speculators maximally long + uptrend intact; momentum trade valid.",
            "crowded_trade_caution_bearish_divergence": "Speculators maximally long but price reversing; crowded trade risk.",
            "contrarian_bullish_setup": "Speculators maximally short; contrarian long setup with price downtrend.",
            "short_squeeze_risk": "Speculators maximally short but price rising; short-squeeze potential.",
            "extreme_long_watch": "Speculator positioning extreme — monitor for reversal.",
            "extreme_short_watch": "Speculator positioning at historical lows — monitor for reversal.",
            "neutral": "COT positioning within normal range; no contrarian signal.",
        }.get(str(result.get("signal")), "")

        return result


# ---------------------------------------------------------------------------
# COTMacroSignals
# ---------------------------------------------------------------------------


class COTMacroSignals:
    """
    Synthesises COT data across asset classes into macro-level sentiment gauges.
    """

    def __init__(self, engine: Optional[COTSignalEngine] = None) -> None:
        self._engine = engine or COTSignalEngine()

    # ------------------------------------------------------------------
    # Equity sentiment
    # ------------------------------------------------------------------

    def equity_futures_sentiment(self) -> dict[str, Any]:
        """
        S&P 500 and NASDAQ-100 COT positioning as a market sentiment gauge.
        Dealer short (asset manager long) = institutional bullish bias.
        """
        result: dict[str, Any] = {"as_of": str(date.today()), "markets": {}}

        for contract in ["ES", "NQ"]:
            try:
                extremes_mm  = self._engine.detect_extremes(contract, "managed_money")
                extremes_am  = self._engine.detect_extremes(contract, "asset_manager")
                net_mm_df    = self._engine.compute_net_positioning(contract, "managed_money")
                net_am_df    = self._engine.compute_net_positioning(contract, "asset_manager")

                result["markets"][contract] = {
                    "market_name": FUTURES_MARKET_MAP.get(contract, contract),
                    "managed_money_cot_index": extremes_mm.get("cot_index_current"),
                    "managed_money_signal": extremes_mm.get("signal"),
                    "asset_manager_cot_index": extremes_am.get("cot_index_current"),
                    "asset_manager_net": float(net_am_df["net"].iloc[-1]) if not net_am_df.empty else None,
                    "managed_money_net": float(net_mm_df["net"].iloc[-1]) if not net_mm_df.empty else None,
                }
            except Exception as exc:
                logger.warning("Equity sentiment failed", contract=contract, error=str(exc))
                result["markets"][contract] = {"status": "error", "error": str(exc)}

        # Derive aggregate sentiment
        signals = [
            v.get("managed_money_signal", "neutral")
            for v in result["markets"].values()
            if isinstance(v, dict)
        ]
        bullish  = sum(1 for s in signals if "bullish" in str(s))
        bearish  = sum(1 for s in signals if "bearish" in str(s))
        result["aggregate_sentiment"] = (
            "bullish" if bullish > bearish else ("bearish" if bearish > bullish else "neutral")
        )
        return result

    # ------------------------------------------------------------------
    # Rates positioning
    # ------------------------------------------------------------------

    def rates_futures_positioning(self) -> dict[str, Any]:
        """
        Treasury futures positioning (dealer vs asset manager) as a
        rate direction signal. Dealer short → expecting rates to rise.
        """
        result: dict[str, Any] = {"as_of": str(date.today()), "contracts": {}}

        for contract in _RATES_CONTRACTS[:6]:
            try:
                extremes_dealer = self._engine.detect_extremes(contract, "dealer")
                extremes_am     = self._engine.detect_extremes(contract, "asset_manager")
                net_dealer_df   = self._engine.compute_net_positioning(contract, "dealer")
                net_am_df       = self._engine.compute_net_positioning(contract, "asset_manager")

                result["contracts"][contract] = {
                    "market_name": FUTURES_MARKET_MAP.get(contract, contract),
                    "dealer_cot_index": extremes_dealer.get("cot_index_current"),
                    "dealer_signal": extremes_dealer.get("signal"),
                    "dealer_net": float(net_dealer_df["net"].iloc[-1]) if not net_dealer_df.empty else None,
                    "asset_manager_cot_index": extremes_am.get("cot_index_current"),
                    "asset_manager_net": float(net_am_df["net"].iloc[-1]) if not net_am_df.empty else None,
                }
            except Exception as exc:
                logger.warning("Rates positioning failed", contract=contract, error=str(exc))

        return result

    # ------------------------------------------------------------------
    # FX positioning
    # ------------------------------------------------------------------

    def fx_positioning_summary(self) -> pd.DataFrame:
        """
        USD vs major currencies: net speculator (large trader) positioning
        as a USD strength/weakness gauge.
        """
        rows = []
        for contract in _FX_CONTRACTS:
            try:
                net_df   = self._engine.compute_net_positioning(contract, "managed_money")
                extremes = self._engine.detect_extremes(contract, "managed_money")

                if net_df.empty:
                    continue
                latest = net_df.iloc[-1]
                rows.append(
                    {
                        "contract": contract,
                        "currency": FUTURES_MARKET_MAP.get(contract, contract),
                        "net_speculator": float(latest.get("net", 0)),
                        "net_pct_oi": float(latest.get("net_pct_oi", float("nan"))),
                        "cot_index": extremes.get("cot_index_current"),
                        "signal": extremes.get("signal"),
                        "is_extreme": extremes.get("is_extreme_long") or extremes.get("is_extreme_short"),
                    }
                )
            except Exception as exc:
                logger.warning("FX positioning failed", contract=contract, error=str(exc))

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("net_speculator", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Commodity positioning
    # ------------------------------------------------------------------

    def commodity_positioning_summary(self, category: str = "all") -> pd.DataFrame:
        """
        Commercial and speculator net positioning for commodity futures.

        category : "all" | "energy" | "metals" | "agriculture"
        """
        cat = category.lower()
        if cat == "energy":
            contracts = _ENERGY_CONTRACTS
        elif cat in ("metals", "metal"):
            contracts = _METALS_CONTRACTS
        elif cat in ("agriculture", "ag", "agri"):
            contracts = _AG_CONTRACTS
        else:
            contracts = _ENERGY_CONTRACTS + _METALS_CONTRACTS + _AG_CONTRACTS

        rows = []
        for contract in contracts:
            try:
                net_comm = self._engine.compute_net_positioning(contract, "commercial")
                net_spec = self._engine.compute_net_positioning(contract, "managed_money")
                ext_comm = self._engine.detect_extremes(contract, "commercial")

                comm_net = float(net_comm["net"].iloc[-1]) if not net_comm.empty else float("nan")
                spec_net = float(net_spec["net"].iloc[-1]) if not net_spec.empty else float("nan")

                rows.append(
                    {
                        "contract": contract,
                        "commodity": FUTURES_MARKET_MAP.get(contract, contract),
                        "commercial_net": comm_net,
                        "speculator_net": spec_net,
                        "cot_index_commercial": ext_comm.get("cot_index_current"),
                        "signal": ext_comm.get("signal", "neutral"),
                        "category": (
                            "energy" if contract in _ENERGY_CONTRACTS
                            else "metals" if contract in _METALS_CONTRACTS
                            else "agriculture"
                        ),
                    }
                )
            except Exception as exc:
                logger.warning("Commodity summary failed", contract=contract, error=str(exc))

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("cot_index_commercial").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Query

    cot_router = APIRouter(prefix="/api/cot", tags=["CFTC COT Positioning"])

    _adapter = COTDataAdapter()
    _engine  = COTSignalEngine(adapter=_adapter)
    _macro   = COTMacroSignals(engine=_engine)

    @cot_router.get("/{contract}/positioning")
    def api_cot_positioning(
        contract: str,
        trader_type: str = Query("commercial", description="commercial | managed_money | dealer | asset_manager"),
    ):
        """Latest COT positioning breakdown for a futures contract."""
        df = _engine.compute_net_positioning(contract.upper(), trader_type=trader_type)
        if df.empty:
            return {"contract": contract, "status": "no_data"}
        latest = df.iloc[-1].to_dict()
        latest["date"] = str(latest.get("date", ""))
        return latest

    @cot_router.get("/{contract}/history")
    def api_cot_history(
        contract: str,
        weeks: int = Query(52, ge=4, le=520),
        trader_type: str = Query("commercial"),
    ):
        """Historical COT net positioning time series."""
        df = _engine.compute_net_positioning(contract.upper(), trader_type=trader_type)
        if df.empty:
            return []
        cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=weeks)
        df = df[df["date"] >= cutoff]
        df["date"] = df["date"].astype(str)
        return df.to_dict(orient="records")

    @cot_router.get("/{contract}/signals")
    def api_cot_signals(
        contract: str,
        trader_type: str = Query("commercial"),
        threshold: float = Query(10.0, ge=1.0, le=30.0),
    ):
        """COT Index extremes and trading signals for a contract."""
        extremes = _engine.detect_extremes(
            contract.upper(), trader_type=trader_type, threshold=threshold
        )
        trend_signal = _engine.compute_trend_following_signal(contract.upper())
        return {**extremes, "trend_following": trend_signal}

    @cot_router.get("/dashboard")
    def api_cot_dashboard(
        contracts: str = Query(
            default="",
            description="Comma-separated list of contract codes (e.g. ES,GC,CL). Leave empty for default.",
        )
    ):
        """All major contracts positioning summary table."""
        contract_list = [c.strip().upper() for c in contracts.split(",") if c.strip()] or None
        df = _engine.get_positioning_dashboard(contracts=contract_list)
        return df.to_dict(orient="records")

    @cot_router.get("/macro")
    def api_cot_macro():
        """Macro-level sentiment signals across equity, rates, FX, and commodities."""
        equity  = _macro.equity_futures_sentiment()
        rates   = _macro.rates_futures_positioning()
        fx_df   = _macro.fx_positioning_summary()
        comm_df = _macro.commodity_positioning_summary()
        return {
            "equity_sentiment": equity,
            "rates_positioning": rates,
            "fx_summary": fx_df.to_dict(orient="records"),
            "commodity_summary": comm_df.to_dict(orient="records"),
        }

except ImportError:
    cot_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — cot_router not registered")
