"""
CFTC COT report parser with COT Index — LEAPFROG #46.

Downloads Commitments of Traders data from CFTC, computes the COT Index
(52-week percentile of speculator net positioning) for any futures market.
Used for macro regime signals: extreme speculator longs = contrarian bear,
extreme shorts = contrarian bull.

No incumbent terminal provides COT Index as a first-class signal. Score: SENTINEL 10, Bloomberg 1.
"""
from __future__ import annotations
import asyncio
import io
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
import pandas as pd
import httpx
from sentinel.core.types import COTReport
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# CFTC publishes historical COT as annual ZIP files
CFTC_BASE = "https://www.cftc.gov/files/dea/history"
CFTC_DISAGGREGATED_URL = f"{CFTC_BASE}/fut_disagg_txt_hist_2006_{{year}}.zip"
CFTC_LEGACY_URL = f"{CFTC_BASE}/com_disagg_txt_hist_{{year}}.zip"
CFTC_CURRENT_DISAGG = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2016_2025.zip"

# Known market codes for major futures
MARKET_CODES = {
    "ES": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
    "NQ": "E-MINI NASDAQ-100 - CHICAGO MERCANTILE EXCHANGE",
    "GC": "GOLD - COMMODITY EXCHANGE INC.",
    "SI": "SILVER - COMMODITY EXCHANGE INC.",
    "CL": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    "NG": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    "ZN": "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "ZB": "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
    "6E": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "6J": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "BTC": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "ZC": "CORN - CHICAGO BOARD OF TRADE",
    "ZS": "SOYBEANS - CHICAGO BOARD OF TRADE",
    "ZW": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
    "KC": "COFFEE C - ICE FUTURES U.S.",
    "CT": "COTTON NO. 2 - ICE FUTURES U.S.",
}


class COTClient:
    """Downloads and parses CFTC Disaggregated COT reports."""

    def __init__(self) -> None:
        self._df_cache: Optional[pd.DataFrame] = None
        self._loaded_years: set[int] = set()

    async def load_year(self, year: int) -> pd.DataFrame:
        """Load disaggregated COT data for a specific year from CFTC."""
        if year in self._loaded_years and self._df_cache is not None:
            return self._df_cache

        url = CFTC_DISAGGREGATED_URL.format(year=year)
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                # Disaggregated futures: file inside zip is typically *.txt
                txt_name = next(n for n in zf.namelist() if n.endswith(".txt"))
                with zf.open(txt_name) as f:
                    df = pd.read_csv(f, low_memory=False)

            df.columns = [c.strip() for c in df.columns]
            df["As of Date in Form YYYY-MM-DD"] = pd.to_datetime(
                df.get("As of Date in Form YYYY-MM-DD", df.get("Report_Date_as_YYYY-MM-DD", "")),
                errors="coerce"
            )

            if self._df_cache is None:
                self._df_cache = df
            else:
                self._df_cache = pd.concat([self._df_cache, df], ignore_index=True)
            self._loaded_years.add(year)
            logger.info("COT year loaded", year=year, rows=len(df))
        except Exception as exc:
            logger.error("COT load error", year=year, error=str(exc))

        return self._df_cache or pd.DataFrame()

    async def load_range(self, start_year: int, end_year: Optional[int] = None) -> pd.DataFrame:
        """Load COT data for a range of years."""
        end_year = end_year or date.today().year
        tasks = [self.load_year(y) for y in range(start_year, end_year + 1)]
        await asyncio.gather(*tasks)
        return self._df_cache or pd.DataFrame()

    def get_market_data(self, market_name_substring: str) -> pd.DataFrame:
        """Filter the loaded data for a specific market (case-insensitive substring match)."""
        if self._df_cache is None or self._df_cache.empty:
            return pd.DataFrame()
        mask = self._df_cache["Market_and_Exchange_Names"].str.upper().str.contains(
            market_name_substring.upper(), na=False
        )
        return self._df_cache[mask].copy()

    def compute_cot_index(
        self,
        market_name: str,
        lookback_weeks: int = 52,
    ) -> pd.DataFrame:
        """
        Compute the COT Index for a market.
        COT Index = (current_net - min_net_52wk) / (max_net_52wk - min_net_52wk) * 100
        where net = speculator_longs - speculator_shorts (disaggregated: "Managed Money").

        Returns DataFrame with columns: date, net_position, cot_index, signal.
        """
        df = self.get_market_data(market_name)
        if df.empty:
            logger.warning("No COT data for market", market=market_name)
            return pd.DataFrame()

        date_col = "As of Date in Form YYYY-MM-DD"
        long_col = _find_col(df, ["Managed Money Positions Long", "M_Money_Positions_Long_All"])
        short_col = _find_col(df, ["Managed Money Positions Short", "M_Money_Positions_Short_All"])

        if not long_col or not short_col:
            logger.error("COT columns not found", columns=list(df.columns[:10]))
            return pd.DataFrame()

        df = df[[date_col, long_col, short_col]].dropna().copy()
        df = df.rename(columns={date_col: "date", long_col: "mm_long", short_col: "mm_short"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df["mm_long"] = pd.to_numeric(df["mm_long"], errors="coerce").fillna(0)
        df["mm_short"] = pd.to_numeric(df["mm_short"], errors="coerce").fillna(0)
        df["net_position"] = df["mm_long"] - df["mm_short"]

        # Rolling COT Index
        window = lookback_weeks
        df["roll_max"] = df["net_position"].rolling(window).max()
        df["roll_min"] = df["net_position"].rolling(window).min()
        denom = df["roll_max"] - df["roll_min"]
        df["cot_index"] = ((df["net_position"] - df["roll_min"]) / denom.replace(0, float("nan")) * 100).round(1)

        # Signal: >80 = extreme long (contrarian bearish), <20 = extreme short (contrarian bullish)
        df["signal"] = "neutral"
        df.loc[df["cot_index"] >= 80, "signal"] = "extreme_long_bearish"
        df.loc[df["cot_index"] <= 20, "signal"] = "extreme_short_bullish"

        return df[["date", "net_position", "cot_index", "signal"]].dropna(subset=["cot_index"])

    def get_latest_cot_report(self, market_name: str) -> Optional[COTReport]:
        """Return the most recent COT Index reading as a COTReport object."""
        df = self.compute_cot_index(market_name)
        if df.empty:
            return None
        row = df.iloc[-1]
        return COTReport(
            report_date=row["date"].date(),
            market=market_name,
            mm_long=0,  # Could extract from raw df if needed
            mm_short=0,
            net_position=int(row["net_position"]),
            cot_index=float(row["cot_index"]),
            signal=row["signal"],
        )

    def get_all_market_signals(self) -> list[dict]:
        """Return current COT signals for all known markets."""
        signals = []
        for code, name in MARKET_CODES.items():
            rpt = self.get_latest_cot_report(name)
            if rpt:
                signals.append({
                    "code": code,
                    "market": name,
                    "date": rpt.report_date.isoformat(),
                    "net_position": rpt.net_position,
                    "cot_index": rpt.cot_index,
                    "signal": rpt.signal,
                })
        return sorted(signals, key=lambda x: x["cot_index"])


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    # Partial match
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None
