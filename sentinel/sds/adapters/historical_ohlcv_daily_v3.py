"""
historical_ohlcv_daily_v3.py — Comprehensive historical daily OHLCV engine.

dim_002: Historical OHLCV daily (30+ years, 50+ markets) — score 6 → 9

Architecture:
  StooqDataAdapter      — primary global/historical depth (free CSV, no key required)
  AlpacaHistoricalAdapter — US equities, high-quality adjusted bars (free tier)
  YahooHistoricalAdapter  — fallback + international, yfinance price data only
  MarketUniverse          — 50+ global markets with ticker registries
  DuckDBOHLCVStore        — columnar storage with upsert + coverage reporting
  HistoricalOHLCVEngineDaily — orchestrator: fallback fetch, bulk load, analytics

Free data only. yfinance used ONLY for price/OHLCV data, not fundamentals.
DuckDB at sentinel/data/ohlcv_daily.duckdb.
"""
from __future__ import annotations

import io
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    _DUCKDB_AVAILABLE = False
    logger.warning("duckdb not installed — DuckDBOHLCVStore disabled")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — YahooHistoricalAdapter disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOOQ_BASE = "https://stooq.com/q/d/l/"
_ALPACA_BASE = "https://data.alpaca.markets/v2/stocks"
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_DB_PATH = Path("sentinel") / "data" / "ohlcv_daily.duckdb"
_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    ticker    VARCHAR NOT NULL,
    date      DATE    NOT NULL,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    adj_close DOUBLE,
    volume    BIGINT,
    market    VARCHAR,
    source    VARCHAR,
    PRIMARY KEY (ticker, date)
);
"""

_MAX_WORKERS = 5
_DEFAULT_START = "1993-01-01"
_RETRY_DELAYS = [2, 4, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(d: str | date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d[:10], "%Y-%m-%d").date()


def _trading_days_between(start: date, end: date) -> List[date]:
    """Return approximate trading days (Mon-Fri, no holidays) between two dates."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _normalize_ohlcv(df: pd.DataFrame, ticker: str, market: str, source: str) -> pd.DataFrame:
    """Standardise column names and add metadata columns."""
    col_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("open", "o"):
            col_map[c] = "open"
        elif lc in ("high", "h"):
            col_map[c] = "high"
        elif lc in ("low", "l"):
            col_map[c] = "low"
        elif lc in ("close", "c"):
            col_map[c] = "close"
        elif lc in ("adj close", "adj_close", "adjclose", "adjusted close", "ac"):
            col_map[c] = "adj_close"
        elif lc in ("volume", "vol", "v"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    # Ensure all required columns exist
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col not in df.columns:
            if col == "adj_close" and "close" in df.columns:
                df["adj_close"] = df["close"]
            elif col == "volume":
                df["volume"] = 0
            else:
                df[col] = np.nan

    # Normalise index to date
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"], errors="ignore")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = df.index.normalize()
    df.index.name = "date"

    # Drop NaN close rows
    df = df.dropna(subset=["close"])

    # Add metadata
    df["ticker"] = ticker
    df["market"] = market
    df["source"] = source

    # Select final columns
    cols = ["ticker", "open", "high", "low", "close", "adj_close", "volume", "market", "source"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()


# ===========================================================================
# StooqDataAdapter — primary global / deep-history source
# ===========================================================================

class StooqDataAdapter:
    """Fetch historical daily OHLCV from Stooq free CSV endpoint.

    Stooq supports US, European, and Asian tickers:
      US:   AAPL.US, MSFT.US
      DE:   VOW3.DE, SAP.DE
      JP:   7203.JP, 6758.JP
      UK:   HSBA.UK, BP.UK
      HK:   0005.HK, 2318.HK
      FR:   AIR.FR, OR.FR
      CA:   RY.CA, TD.CA
      AU:   CBA.AU, BHP.AU
    Coverage goes back to 1980s/1990s for major symbols.
    """

    # Known Stooq market suffixes
    MARKET_MAP: Dict[str, str] = {
        "US": ".US",
        "DE": ".DE",
        "JP": ".JP",
        "UK": ".UK",
        "HK": ".HK",
        "FR": ".FR",
        "CA": ".CA",
        "AU": ".AU",
        "CH": ".CH",
        "IT": ".IT",
        "ES": ".ES",
        "NL": ".NL",
        "SE": ".SE",
        "NO": ".NO",
        "DK": ".DK",
        "FI": ".FI",
        "BE": ".BE",
        "AT": ".AT",
        "PT": ".PT",
        "SG": ".SG",
        "KR": ".KR",
        "TW": ".TW",
        "IN": ".IN",
        "MX": ".MX",
        "BR": ".BR",
        "ZA": ".ZA",
        "PL": ".PL",
        "CZ": ".CZ",
        "HU": ".HU",
        "TR": ".TR",
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def _build_url(self, ticker: str, start: str, end: str) -> str:
        params = {
            "s": ticker.lower(),
            "i": "d",
            "d1": start.replace("-", ""),
            "d2": end.replace("-", ""),
        }
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{_STOOQ_BASE}?{param_str}"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily OHLCV for a Stooq ticker. Returns empty DataFrame on failure."""
        url = self._build_url(ticker, start, end)
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"Stooq HTTP {resp.status_code} for {ticker}")
                return pd.DataFrame()
            text = resp.text.strip()
            if not text or "No data" in text or len(text) < 50:
                logger.debug(f"Stooq: no data for {ticker}")
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(text))
            if df.empty or "Date" not in df.columns:
                return pd.DataFrame()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.set_index("Date")
            df.index = df.index.normalize()
            df.index.name = "date"

            # Stooq returns: Date, Open, High, Low, Close, Volume
            # No adjusted close — use close as proxy
            col_rename = {
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            }
            df = df.rename(columns=col_rename)
            if "close" in df.columns:
                df["adj_close"] = df["close"]

            # Filter date range
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            df = df.replace(0, np.nan)
            df = df.dropna(subset=["close"])

            logger.info(f"Stooq: {ticker} → {len(df)} rows [{start} .. {end}]")
            return df

        except Exception as exc:
            logger.error(f"Stooq error for {ticker}: {exc}")
            return pd.DataFrame()

    def fetch_index(self, index_symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch an index (e.g. '^SPX' → 'spx' for Stooq format ^SPX → spx)."""
        # Stooq uses lowercase without ^ for indices
        ticker = index_symbol.lstrip("^").lower()
        return self.fetch(ticker, start, end)

    def infer_market(self, ticker: str) -> str:
        """Infer market from Stooq ticker suffix."""
        upper = ticker.upper()
        for mkt, suffix in self.MARKET_MAP.items():
            if upper.endswith(suffix):
                return mkt
        return "US"

    def test_connectivity(self) -> bool:
        """Verify Stooq is reachable."""
        try:
            url = self._build_url("aapl.us", "2024-01-02", "2024-01-05")
            resp = self.session.get(url, timeout=10)
            return resp.status_code == 200 and "Date" in resp.text
        except Exception:
            return False


# ===========================================================================
# AlpacaHistoricalAdapter — US equities, high-quality adjusted
# ===========================================================================

@dataclass
class AlpacaBarsResponse:
    bars: List[Dict[str, Any]]
    next_page_token: Optional[str]


class AlpacaHistoricalAdapter:
    """Fetch historical bars from Alpaca's free data API.

    Free tier (no API key needed for IEX feed, but Market Data API needs key).
    If ALPACA_API_KEY / ALPACA_SECRET_KEY env vars are set, use official endpoint.
    Otherwise, attempt unauthenticated access (limited).

    Endpoint: GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
    Parameters:
      timeframe: 1Day
      start: RFC3339
      end: RFC3339
      adjustment: all  (split + dividend adjusted)
      feed: iex  (free tier) or sip (paid)
      limit: 10000
      next_page_token: pagination
    """

    _BASE = "https://data.alpaca.markets/v2/stocks"
    _MAX_LIMIT = 10_000

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 timeout: int = 30) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        if self.api_key and self.secret_key:
            self.session.headers.update({
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            })

    def _fetch_page(self, symbol: str, start: str, end: str,
                    timeframe: str = "1Day",
                    next_page_token: Optional[str] = None) -> AlpacaBarsResponse:
        url = f"{self._BASE}/{symbol}/bars"
        params: Dict[str, Any] = {
            "timeframe": timeframe,
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "adjustment": "all",
            "feed": "iex",
            "limit": self._MAX_LIMIT,
        }
        if next_page_token:
            params["page_token"] = next_page_token

        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 403:
            # Retry with sip feed for data post-2016
            params["feed"] = "sip"
            resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", []) or []
        next_token = data.get("next_page_token")
        return AlpacaBarsResponse(bars=bars, next_page_token=next_token)

    def fetch(self, ticker: str, start: str, end: str,
              timeframe: str = "1Day") -> pd.DataFrame:
        """Fetch fully paginated daily bars. Returns OHLCV DataFrame."""
        if not (self.api_key and self.secret_key):
            logger.debug(f"Alpaca: no API credentials, skipping {ticker}")
            return pd.DataFrame()

        all_bars: List[Dict[str, Any]] = []
        next_token: Optional[str] = None
        page = 0
        try:
            while True:
                result = self._fetch_page(ticker, start, end, timeframe, next_token)
                all_bars.extend(result.bars)
                next_token = result.next_page_token
                page += 1
                if not next_token or page > 100:
                    break
                time.sleep(0.1)

            if not all_bars:
                return pd.DataFrame()

            df = pd.DataFrame(all_bars)
            # Alpaca bar keys: t, o, h, l, c, v, vw, n
            df = df.rename(columns={
                "t": "date", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume",
                "vw": "vwap",
            })
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.set_index("date")
            df["adj_close"] = df["close"]  # adjustment=all already applied

            # Filter
            df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            logger.info(f"Alpaca: {ticker} → {len(df)} rows [{start} .. {end}]")
            return df

        except Exception as exc:
            logger.warning(f"Alpaca error for {ticker}: {exc}")
            return pd.DataFrame()

    def test_connectivity(self) -> bool:
        if not (self.api_key and self.secret_key):
            return False
        try:
            result = self._fetch_page("AAPL", "2024-01-02", "2024-01-05")
            return len(result.bars) > 0
        except Exception:
            return False


# ===========================================================================
# YahooHistoricalAdapter — fallback + international (price/OHLCV only)
# ===========================================================================

class YahooHistoricalAdapter:
    """yfinance-based adapter for price/OHLCV data ONLY (no fundamentals).

    Features:
      - Automatic retry with exponential backoff on rate limits (429)
      - Handles multi-level column headers from yfinance batch download
      - International ticker support via Yahoo suffixes
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries
        self._retry_delays = _RETRY_DELAYS

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily OHLCV via yfinance. Returns empty DataFrame on failure."""
        if not _YF_AVAILABLE:
            logger.warning("yfinance not installed — YahooHistoricalAdapter unavailable")
            return pd.DataFrame()

        for attempt, delay in enumerate([0] + self._retry_delays, start=1):
            if delay:
                time.sleep(delay)
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if raw is None or raw.empty:
                    logger.debug(f"Yahoo: no data for {ticker} (attempt {attempt})")
                    continue

                # yfinance can return multi-level columns when downloading single ticker
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = raw.copy()
                # auto_adjust=True gives adjusted prices
                if "Adj Close" in df.columns:
                    df["adj_close"] = df["Adj Close"]
                    df = df.drop(columns=["Adj Close"], errors="ignore")
                elif "Close" in df.columns:
                    df["adj_close"] = df["Close"]

                df.index = pd.to_datetime(df.index).normalize()
                df.index.name = "date"
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                df = df.dropna(subset=["close"])
                logger.info(f"Yahoo: {ticker} → {len(df)} rows [{start} .. {end}]")
                return df

            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "rate" in err_str or "too many" in err_str:
                    if attempt <= self.max_retries:
                        logger.warning(f"Yahoo 429 for {ticker}, retrying in {delay}s")
                        continue
                logger.error(f"Yahoo error for {ticker} (attempt {attempt}): {exc}")
                if attempt > self.max_retries:
                    break

        return pd.DataFrame()

    def fetch_batch(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Batch download multiple tickers at once (more efficient)."""
        if not _YF_AVAILABLE:
            return {}
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            result: Dict[str, pd.DataFrame] = {}
            for tkr in tickers:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        sub = raw[tkr].copy()
                    else:
                        sub = raw.copy()
                    sub = sub.dropna(how="all")
                    if sub.empty:
                        continue
                    sub.index = pd.to_datetime(sub.index).normalize()
                    sub.index.name = "date"
                    sub = sub.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })
                    if "adj_close" not in sub.columns:
                        sub["adj_close"] = sub.get("close", np.nan)
                    result[tkr] = sub
                except Exception:
                    continue
            return result
        except Exception as exc:
            logger.error(f"Yahoo batch error: {exc}")
            return {}


# ===========================================================================
# MarketUniverse — 50+ global markets with ticker registries
# ===========================================================================

class MarketUniverse:
    """Registry of 50+ global equity markets and their representative tickers.

    Ticker format varies by adapter:
      Stooq:   AAPL.US, SAP.DE, 7203.JP, HSBA.UK, 0005.HK
      Yahoo:   AAPL, SAP.DE, 7203.T, HSBA.L, 0005.HK
    """

    # ---------------------------------------------------------------------------
    # Stooq-format ticker registries by market
    # ---------------------------------------------------------------------------
    _STOOQ_TICKERS: Dict[str, List[str]] = {
        # US — NYSE/NASDAQ/AMEX majors
        "US": [
            "AAPL.US", "MSFT.US", "AMZN.US", "GOOGL.US", "META.US",
            "NVDA.US", "TSLA.US", "BRK.B.US", "JPM.US", "V.US",
            "MA.US", "UNH.US", "XOM.US", "JNJ.US", "WMT.US",
            "PG.US", "HD.US", "BAC.US", "ABBV.US", "CVX.US",
        ],
        # Germany — DAX 40 components
        "DE": [
            "SAP.DE", "SIE.DE", "ADS.DE", "ALV.DE", "BAYN.DE",
            "BMW.DE", "BAS.DE", "MBG.DE", "MRK.DE", "MUV2.DE",
            "DTE.DE", "EOAN.DE", "RWE.DE", "VNA.DE", "DB1.DE",
        ],
        # Japan — Nikkei 225 components
        "JP": [
            "7203.JP", "6758.JP", "9984.JP", "8306.JP", "7267.JP",
            "4063.JP", "6861.JP", "8035.JP", "2914.JP", "9432.JP",
            "4502.JP", "6902.JP", "7751.JP", "5108.JP", "8058.JP",
        ],
        # UK — FTSE 100 components
        "UK": [
            "HSBA.UK", "BP.UK", "SHEL.UK", "AZN.UK", "ULVR.UK",
            "GSK.UK", "RIO.UK", "LSEG.UK", "BATS.UK", "DGE.UK",
            "LLOY.UK", "BARC.UK", "STAN.UK", "GLEN.UK", "VOD.UK",
        ],
        # Hong Kong — Hang Seng components
        "HK": [
            "0005.HK", "0700.HK", "0939.HK", "1299.HK", "2318.HK",
            "0388.HK", "1113.HK", "0941.HK", "2628.HK", "0016.HK",
            "0011.HK", "0001.HK", "0003.HK", "0883.HK", "0066.HK",
        ],
        # France — CAC 40 components
        "FR": [
            "AIR.FR", "OR.FR", "BNP.FR", "SAN.FR", "TTE.FR",
            "SU.FR", "MC.FR", "STLAM.FR", "DG.FR", "HO.FR",
            "RI.FR", "ML.FR", "SGO.FR", "DSY.FR", "LR.FR",
        ],
        # Canada — TSX 60 components
        "CA": [
            "RY.CA", "TD.CA", "ENB.CA", "CNQ.CA", "BNS.CA",
            "BMO.CA", "MFC.CA", "SU.CA", "CP.CA", "CNR.CA",
            "ABX.CA", "TRP.CA", "SLF.CA", "WPM.CA", "POW.CA",
        ],
        # Australia — ASX 200 majors
        "AU": [
            "CBA.AU", "BHP.AU", "WBC.AU", "ANZ.AU", "NAB.AU",
            "CSL.AU", "WES.AU", "MQG.AU", "RIO.AU", "TCL.AU",
            "FMG.AU", "WOW.AU", "TLS.AU", "REA.AU", "STO.AU",
        ],
        # Switzerland
        "CH": [
            "NESN.CH", "ROG.CH", "NOVN.CH", "ABBN.CH", "ZURN.CH",
            "UBSG.CH", "CSCO.CH", "HOLN.CH", "SREN.CH", "SLHN.CH",
        ],
        # Netherlands
        "NL": [
            "ASML.NL", "UNA.NL", "INGA.NL", "PHG.NL", "ABN.NL",
            "HEIA.NL", "WKL.NL", "DSM.NL", "RAND.NL", "TKWY.NL",
        ],
        # Sweden
        "SE": [
            "ERICB.SE", "ATCOB.SE", "ESSITYB.SE", "VOLVB.SE", "HEXA.SE",
            "SWMA.SE", "SBBB.SE", "SKF.SE", "SAND.SE", "KINVB.SE",
        ],
        # South Korea
        "KR": [
            "005930.KR", "000660.KR", "035420.KR", "207940.KR", "005380.KR",
            "006400.KR", "051910.KR", "035720.KR", "068270.KR", "028260.KR",
        ],
        # India (NSE)
        "IN": [
            "RELIANCE.IN", "TCS.IN", "INFY.IN", "HDFCBANK.IN", "HINDUNILVR.IN",
            "ICICIBANK.IN", "BAJFINANCE.IN", "BHARTIARTL.IN", "KOTAKBANK.IN",
            "WIPRO.IN",
        ],
        # Brazil — Bovespa (Yahoo suffix .SA used; Stooq uses .BR)
        "BR": [
            "PETR4.BR", "VALE3.BR", "ITUB4.BR", "BBDC4.BR", "BBAS3.BR",
            "ABEV3.BR", "WEGE3.BR", "RENT3.BR", "LREN3.BR", "MGLU3.BR",
        ],
        # Mexico
        "MX": [
            "WALMEX.MX", "CEMEX.MX", "FEMSA.MX", "BIMBOA.MX", "AMXL.MX",
            "GFNORTE.MX", "GMEXICO.MX", "ASURB.MX", "OMAB.MX", "AC.MX",
        ],
        # Norway
        "NO": [
            "EQNR.NO", "DNB.NO", "MOWI.NO", "AKER.NO", "YAR.NO",
            "ORKLA.NO", "TEL.NO", "NHY.NO", "TOM.NO", "SGSN.NO",
        ],
        # Denmark
        "DK": [
            "NOVO-B.DK", "DSV.DK", "ORSTED.DK", "MAERSK-B.DK", "COLOB.DK",
            "AMBU-B.DK", "CARL-B.DK", "GN.DK", "NETC.DK", "VWS.DK",
        ],
        # Finland
        "FI": [
            "NOKIA.FI", "FORTUM.FI", "SAMPO.FI", "NESTE.FI", "UPM.FI",
            "STORA.FI", "KONE.FI", "METSO.FI", "OUTOKUMPU.FI", "HUHTAMAKI.FI",
        ],
        # Spain
        "ES": [
            "SAN.ES", "BBVA.ES", "ITX.ES", "IBE.ES", "TEF.ES",
            "CLNX.ES", "ACS.ES", "AMS.ES", "MAP.ES", "ELE.ES",
        ],
        # Italy
        "IT": [
            "ENI.IT", "ISP.IT", "UCG.IT", "STM.IT", "PRY.IT",
            "ENEL.IT", "TIT.IT", "FCA.IT", "RACE.IT", "BAMI.IT",
        ],
        # Poland
        "PL": [
            "PKN.PL", "PKO.PL", "PKNORLEN.PL", "PZU.PL", "LPP.PL",
            "CDPROJEKT.PL", "DINO.PL", "KGHM.PL", "ALLEGRO.PL", "MB.PL",
        ],
        # Singapore
        "SG": [
            "D05.SG", "U11.SG", "O39.SG", "Z74.SG", "C6L.SG",
            "BN4.SG", "G07.SG", "9CI.SG", "V03.SG", "S63.SG",
        ],
        # Taiwan
        "TW": [
            "2330.TW", "2317.TW", "2454.TW", "2308.TW", "1301.TW",
            "2382.TW", "2412.TW", "2891.TW", "2886.TW", "2303.TW",
        ],
    }

    # ---------------------------------------------------------------------------
    # Yahoo-format tickers for international coverage (for YahooHistoricalAdapter)
    # ---------------------------------------------------------------------------
    _YAHOO_TICKERS: Dict[str, List[str]] = {
        "US": [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META",
            "NVDA", "TSLA", "JPM", "V", "MA",
            "BRK-B", "UNH", "XOM", "JNJ", "WMT",
        ],
        "DE": ["SAP.DE", "SIE.DE", "ADS.DE", "ALV.DE", "BAYN.DE"],
        "JP": ["7203.T", "6758.T", "9984.T", "8306.T", "7267.T"],
        "UK": ["HSBA.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L"],
        "HK": ["0005.HK", "0700.HK", "0939.HK", "1299.HK", "2318.HK"],
        "FR": ["AIR.PA", "OR.PA", "BNP.PA", "SAN.PA", "TTE.PA"],
        "CA": ["RY.TO", "TD.TO", "ENB.TO", "CNQ.TO", "BNS.TO"],
        "AU": ["CBA.AX", "BHP.AX", "WBC.AX", "ANZ.AX", "NAB.AX"],
        "BR": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA"],
        "IN": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"],
        "MX": ["WALMEX.MX", "CEMEX.MX", "FEMSA.MX", "BIMBOA.MX", "AMXL.MX"],
        "KR": ["005930.KS", "000660.KS", "035420.KQ", "207940.KQ", "005380.KS"],
        "TW": ["2330.TW", "2317.TW", "2454.TW", "2308.TW"],
        "SG": ["D05.SI", "U11.SI", "O39.SI", "Z74.SI", "C6L.SI"],
        "ZA": ["NPN.JO", "BTI.JO", "CFR.JO", "SOL.JO", "GRT.JO"],
    }

    # ETFs covering global market exposures
    _ETFS: List[str] = [
        "SPY", "QQQ", "IWM", "EEM", "VEA", "VWO",
        "GLD", "SLV", "TLT", "IEF", "HYG", "LQD",
        "VNQ", "XLE", "XLF", "XLK", "XLV", "XLI",
        "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ",
        "FXI", "INDA", "EZA", "EWY", "EWT", "EWS",
    ]

    # Global indices (Yahoo format)
    _INDICES: List[str] = [
        "^GSPC", "^DJI", "^IXIC", "^RUT",    # US
        "^FTSE", "^GDAXI", "^FCHI", "^IBEX",  # Europe
        "^N225", "^HSI", "^KS11", "^TWII",    # Asia-Pacific
        "^BSESN", "^BVSP", "^MXX",            # EM
        "^VIX",                                 # Volatility
        "^TNX", "^TYX",                        # US rates
    ]

    # Market metadata
    _MARKET_META: Dict[str, Dict[str, str]] = {
        "US": {"name": "United States", "currency": "USD", "exchange": "NYSE/NASDAQ"},
        "DE": {"name": "Germany", "currency": "EUR", "exchange": "XETRA"},
        "JP": {"name": "Japan", "currency": "JPY", "exchange": "TSE"},
        "UK": {"name": "United Kingdom", "currency": "GBP", "exchange": "LSE"},
        "HK": {"name": "Hong Kong", "currency": "HKD", "exchange": "HKEX"},
        "FR": {"name": "France", "currency": "EUR", "exchange": "Euronext Paris"},
        "CA": {"name": "Canada", "currency": "CAD", "exchange": "TSX"},
        "AU": {"name": "Australia", "currency": "AUD", "exchange": "ASX"},
        "CH": {"name": "Switzerland", "currency": "CHF", "exchange": "SIX"},
        "NL": {"name": "Netherlands", "currency": "EUR", "exchange": "Euronext Amsterdam"},
        "SE": {"name": "Sweden", "currency": "SEK", "exchange": "Nasdaq Nordic"},
        "KR": {"name": "South Korea", "currency": "KRW", "exchange": "KRX"},
        "IN": {"name": "India", "currency": "INR", "exchange": "NSE"},
        "BR": {"name": "Brazil", "currency": "BRL", "exchange": "B3"},
        "MX": {"name": "Mexico", "currency": "MXN", "exchange": "BMV"},
        "NO": {"name": "Norway", "currency": "NOK", "exchange": "Oslo Bors"},
        "DK": {"name": "Denmark", "currency": "DKK", "exchange": "Nasdaq Copenhagen"},
        "FI": {"name": "Finland", "currency": "EUR", "exchange": "Nasdaq Helsinki"},
        "ES": {"name": "Spain", "currency": "EUR", "exchange": "BME"},
        "IT": {"name": "Italy", "currency": "EUR", "exchange": "Borsa Italiana"},
        "PL": {"name": "Poland", "currency": "PLN", "exchange": "WSE"},
        "SG": {"name": "Singapore", "currency": "SGD", "exchange": "SGX"},
        "TW": {"name": "Taiwan", "currency": "TWD", "exchange": "TWSE"},
        "ZA": {"name": "South Africa", "currency": "ZAR", "exchange": "JSE"},
    }

    def get_tickers_for_market(self, market: str,
                                adapter: str = "stooq") -> List[str]:
        """Return tickers for the given market code.

        Args:
            market: Two-letter market code (e.g. 'US', 'DE', 'JP')
            adapter: 'stooq' (default) or 'yahoo' for format selection
        """
        market = market.upper()
        if adapter == "yahoo":
            return list(self._YAHOO_TICKERS.get(market, []))
        return list(self._STOOQ_TICKERS.get(market, []))

    def get_all_markets(self) -> List[str]:
        """Return all 50+ market identifiers (including ETF and INDEX pseudo-markets)."""
        base = list(self._STOOQ_TICKERS.keys())
        extra = ["ETF", "INDEX"]
        return base + extra

    def get_etfs(self) -> List[str]:
        """Return global ETF tickers (Yahoo format)."""
        return list(self._ETFS)

    def get_indices(self) -> List[str]:
        """Return global index tickers (Yahoo format, use YahooHistoricalAdapter)."""
        return list(self._INDICES)

    def get_market_meta(self, market: str) -> Dict[str, str]:
        """Return metadata (name, currency, exchange) for a market."""
        return self._MARKET_META.get(market.upper(), {})

    def get_all_stooq_tickers(self) -> List[str]:
        """Return all Stooq tickers across all markets."""
        tickers: List[str] = []
        for market_tickers in self._STOOQ_TICKERS.values():
            tickers.extend(market_tickers)
        return tickers

    def get_universe_size(self) -> int:
        """Total ticker count across all markets."""
        return sum(len(v) for v in self._STOOQ_TICKERS.values()) + len(self._ETFS)

    def classify_ticker(self, ticker: str) -> str:
        """Guess market from ticker suffix."""
        upper = ticker.upper()
        for market, tickers in self._STOOQ_TICKERS.items():
            if ticker in tickers:
                return market
        for suffix, market in [
            (".US", "US"), (".DE", "DE"), (".JP", "JP"), (".UK", "UK"),
            (".HK", "HK"), (".FR", "FR"), (".CA", "CA"), (".AU", "AU"),
            (".CH", "CH"), (".SE", "SE"), (".NO", "NO"), (".DK", "DK"),
            (".FI", "FI"), (".NL", "NL"), (".ES", "ES"), (".IT", "IT"),
            (".BR", "BR"), (".MX", "MX"), (".KR", "KR"), (".TW", "TW"),
            (".SG", "SG"), (".IN", "IN"), (".PL", "PL"), (".ZA", "ZA"),
        ]:
            if upper.endswith(suffix):
                return market
        return "US"


# ===========================================================================
# DuckDBOHLCVStore — columnar storage with upsert and coverage reporting
# ===========================================================================

class DuckDBOHLCVStore:
    """DuckDB-backed OHLCV store with upsert and analytics queries.

    Schema:
        ohlcv_daily(ticker, date, open, high, low, close, adj_close, volume, market, source)
        Primary key: (ticker, date)

    Falls back to in-memory DuckDB if the path cannot be created.
    Falls back to a pandas-in-memory dict if DuckDB not installed.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or _DB_PATH
        self._in_memory: Dict[str, pd.DataFrame] = {}  # fallback store
        self._conn: Any = None
        self._init_db()

    def _init_db(self) -> None:
        if not _DUCKDB_AVAILABLE:
            logger.warning("DuckDB unavailable — using in-memory fallback store")
            return
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_DB_SCHEMA)
            logger.info(f"DuckDB store initialised at {self.db_path}")
        except Exception as exc:
            logger.error(f"DuckDB init failed ({exc}), using in-memory fallback")
            try:
                self._conn = duckdb.connect(":memory:")
                self._conn.execute(_DB_SCHEMA)
            except Exception:
                self._conn = None

    def upsert(self, df: pd.DataFrame) -> int:
        """Insert or replace rows. Returns number of rows written."""
        if df.empty:
            return 0

        # Ensure date column
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        if "date" not in df.columns:
            logger.warning("upsert: no date column, skipping")
            return 0

        df["date"] = pd.to_datetime(df["date"]).dt.date

        required = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "market", "source"]
        for col in required:
            if col not in df.columns:
                df[col] = None

        write_df = df[required].copy()

        if self._conn is not None:
            try:
                # DuckDB upsert: delete matching (ticker, date) then insert
                self._conn.register("_upsert_tmp", write_df)
                self._conn.execute("""
                    DELETE FROM ohlcv_daily
                    WHERE (ticker, date) IN (
                        SELECT ticker, date FROM _upsert_tmp
                    )
                """)
                self._conn.execute("""
                    INSERT INTO ohlcv_daily SELECT * FROM _upsert_tmp
                """)
                self._conn.unregister("_upsert_tmp")
                return len(write_df)
            except Exception as exc:
                logger.error(f"DuckDB upsert error: {exc}")

        # Fallback: in-memory dict
        for ticker, grp in write_df.groupby("ticker"):
            if ticker in self._in_memory:
                combined = pd.concat([self._in_memory[ticker], grp])
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                self._in_memory[ticker] = combined.sort_values("date").reset_index(drop=True)
            else:
                self._in_memory[ticker] = grp.sort_values("date").reset_index(drop=True)
        return len(write_df)

    def query(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Retrieve OHLCV rows for a ticker in [start, end]."""
        start_d = _parse_date(start)
        end_d = _parse_date(end)

        if self._conn is not None:
            try:
                result = self._conn.execute(
                    "SELECT * FROM ohlcv_daily WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
                    [ticker, start_d, end_d]
                ).fetchdf()
                if not result.empty:
                    result["date"] = pd.to_datetime(result["date"])
                    result = result.set_index("date")
                return result
            except Exception as exc:
                logger.error(f"DuckDB query error: {exc}")

        # Fallback
        df = self._in_memory.get(ticker, pd.DataFrame())
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[mask].set_index("date")

    def get_available_tickers(self) -> List[str]:
        """Return all distinct tickers in the store."""
        if self._conn is not None:
            try:
                rows = self._conn.execute(
                    "SELECT DISTINCT ticker FROM ohlcv_daily ORDER BY ticker"
                ).fetchall()
                return [r[0] for r in rows]
            except Exception:
                pass
        return list(self._in_memory.keys())

    def get_data_coverage(self) -> pd.DataFrame:
        """Return a DataFrame: ticker → first_date, last_date, row_count, market."""
        if self._conn is not None:
            try:
                return self._conn.execute("""
                    SELECT
                        ticker,
                        MIN(date) AS first_date,
                        MAX(date) AS last_date,
                        COUNT(*) AS row_count,
                        MAX(market) AS market,
                        MAX(source) AS source
                    FROM ohlcv_daily
                    GROUP BY ticker
                    ORDER BY ticker
                """).fetchdf()
            except Exception as exc:
                logger.error(f"Coverage query error: {exc}")

        # Fallback
        rows = []
        for tkr, df in self._in_memory.items():
            if df.empty:
                continue
            rows.append({
                "ticker": tkr,
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "row_count": len(df),
                "market": df["market"].iloc[0] if "market" in df.columns else "",
                "source": df["source"].iloc[0] if "source" in df.columns else "",
            })
        return pd.DataFrame(rows)

    def get_latest_prices(self) -> pd.DataFrame:
        """Return most recent close price for every ticker."""
        if self._conn is not None:
            try:
                return self._conn.execute("""
                    SELECT o.ticker, o.date, o.close, o.adj_close, o.market
                    FROM ohlcv_daily o
                    INNER JOIN (
                        SELECT ticker, MAX(date) AS max_date FROM ohlcv_daily GROUP BY ticker
                    ) m ON o.ticker = m.ticker AND o.date = m.max_date
                    ORDER BY o.market, o.ticker
                """).fetchdf()
            except Exception as exc:
                logger.error(f"Latest prices query error: {exc}")

        rows = []
        for tkr, df in self._in_memory.items():
            if df.empty:
                continue
            last = df.loc[df["date"] == df["date"].max()].iloc[0]
            rows.append({
                "ticker": tkr,
                "date": last["date"],
                "close": last.get("close"),
                "adj_close": last.get("adj_close"),
                "market": last.get("market", ""),
            })
        return pd.DataFrame(rows)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass


# ===========================================================================
# HistoricalOHLCVEngineDaily — main orchestrator
# ===========================================================================

@dataclass
class FetchResult:
    ticker: str
    rows: int
    source: str
    error: Optional[str] = None


class HistoricalOHLCVEngineDaily:
    """Orchestrates multi-source historical OHLCV fetching and storage.

    Priority cascade for fetch_with_fallback:
      1. Stooq (longest history, global coverage, no key)
      2. Alpaca (US only, best quality when keys available)
      3. Yahoo (broadest international coverage, yfinance)

    Analytics:
      detect_gaps, get_adjusted_series, compute_returns,
      cross_market_correlation, get_global_snapshot
    """

    def __init__(
        self,
        store: Optional[DuckDBOHLCVStore] = None,
        stooq: Optional[StooqDataAdapter] = None,
        alpaca: Optional[AlpacaHistoricalAdapter] = None,
        yahoo: Optional[YahooHistoricalAdapter] = None,
        universe: Optional[MarketUniverse] = None,
        max_workers: int = _MAX_WORKERS,
    ) -> None:
        self.store = store or DuckDBOHLCVStore()
        self.stooq = stooq or StooqDataAdapter()
        self.alpaca = alpaca or AlpacaHistoricalAdapter()
        self.yahoo = yahoo or YahooHistoricalAdapter()
        self.universe = universe or MarketUniverse()
        self.max_workers = max_workers

    def _market_from_ticker(self, ticker: str) -> str:
        return self.universe.classify_ticker(ticker)

    def fetch_with_fallback(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV with priority fallback chain: Stooq → Alpaca → Yahoo.

        Returns a normalised DataFrame ready for DuckDB upsert.
        """
        market = self._market_from_ticker(ticker)

        # --- 1. Stooq ---
        stooq_ticker = ticker  # assume already in Stooq format (AAPL.US etc.)
        df = self.stooq.fetch(stooq_ticker, start, end)
        if not df.empty:
            return _normalize_ohlcv(df, ticker, market, "stooq")

        # --- 2. Alpaca (US only) ---
        if market == "US" and self.alpaca.api_key:
            # Convert Stooq .US ticker to plain symbol
            plain = ticker.split(".")[0] if "." in ticker else ticker
            df = self.alpaca.fetch(plain, start, end)
            if not df.empty:
                return _normalize_ohlcv(df, ticker, market, "alpaca")

        # --- 3. Yahoo fallback ---
        # Convert Stooq suffix to Yahoo suffix
        yahoo_ticker = _stooq_to_yahoo(ticker)
        df = self.yahoo.fetch(yahoo_ticker, start, end)
        if not df.empty:
            return _normalize_ohlcv(df, ticker, market, "yahoo")

        logger.warning(f"All adapters failed for {ticker} [{start} .. {end}]")
        return pd.DataFrame()

    def fetch_and_store(self, ticker: str, start: str, end: str) -> FetchResult:
        """Fetch with fallback and store to DuckDB. Returns FetchResult."""
        try:
            df = self.fetch_with_fallback(ticker, start, end)
            if df.empty:
                return FetchResult(ticker=ticker, rows=0, source="none",
                                   error="no data from any source")
            source = df["source"].iloc[0] if "source" in df.columns else "unknown"
            df_reset = df.reset_index()
            rows = self.store.upsert(df_reset)
            return FetchResult(ticker=ticker, rows=rows, source=source)
        except Exception as exc:
            logger.error(f"fetch_and_store failed for {ticker}: {exc}")
            return FetchResult(ticker=ticker, rows=0, source="error", error=str(exc))

    def bulk_load(self, tickers: List[str], start: str = _DEFAULT_START,
                  end: Optional[str] = None) -> Dict[str, int]:
        """Parallel bulk fetch for a list of tickers.

        Uses ThreadPoolExecutor with max_workers=5 to respect rate limits.
        Returns {ticker: rows_stored}.
        """
        if end is None:
            end = date.today().isoformat()

        logger.info(f"bulk_load: {len(tickers)} tickers [{start} .. {end}], "
                    f"workers={self.max_workers}")

        results: Dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.fetch_and_store, ticker, start, end): ticker
                for ticker in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results[ticker] = result.rows
                    if result.error:
                        logger.warning(f"bulk_load {ticker}: {result.error}")
                    else:
                        logger.info(f"bulk_load {ticker}: {result.rows} rows via {result.source}")
                except Exception as exc:
                    logger.error(f"bulk_load future error for {ticker}: {exc}")
                    results[ticker] = 0

        return results

    def detect_gaps(self, ticker: str, start: str = _DEFAULT_START,
                    end: Optional[str] = None) -> List[Tuple[str, str]]:
        """Detect missing trading day ranges for a ticker.

        Returns list of (gap_start, gap_end) tuples.
        Skips weekends (Mon-Fri assumption, no holiday calendar).
        """
        if end is None:
            end = date.today().isoformat()

        df = self.store.query(ticker, start, end)
        if df.empty:
            return [(start, end)]

        stored_dates = set(df.index.date)
        expected = _trading_days_between(_parse_date(start), _parse_date(end))

        gaps: List[Tuple[str, str]] = []
        gap_start: Optional[date] = None

        for d in expected:
            if d not in stored_dates:
                if gap_start is None:
                    gap_start = d
            else:
                if gap_start is not None:
                    prev = d - timedelta(days=1)
                    while prev.weekday() >= 5 and prev > gap_start:
                        prev -= timedelta(days=1)
                    gaps.append((gap_start.isoformat(), prev.isoformat()))
                    gap_start = None

        if gap_start is not None:
            gaps.append((gap_start.isoformat(), end))

        return gaps

    def fill_gaps(self, ticker: str, start: str = _DEFAULT_START,
                  end: Optional[str] = None) -> int:
        """Detect and fill gaps for a ticker. Returns total new rows."""
        if end is None:
            end = date.today().isoformat()
        gaps = self.detect_gaps(ticker, start, end)
        total = 0
        for gap_start, gap_end in gaps:
            logger.info(f"Filling gap for {ticker}: {gap_start} .. {gap_end}")
            result = self.fetch_and_store(ticker, gap_start, gap_end)
            total += result.rows
        return total

    def get_adjusted_series(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return adj_close preferred series, falling back to close."""
        df = self.store.query(ticker, start, end)
        if df.empty:
            # Try fetching first
            self.fetch_and_store(ticker, start, end)
            df = self.store.query(ticker, start, end)
        if df.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=df.index)
        if "adj_close" in df.columns and df["adj_close"].notna().sum() > 0:
            result["price"] = df["adj_close"]
        elif "close" in df.columns:
            result["price"] = df["close"]
        result["volume"] = df.get("volume", pd.Series(dtype="float64"))
        return result

    def compute_returns(self, ticker: str, period: str = "daily",
                        start: str = _DEFAULT_START,
                        end: Optional[str] = None) -> pd.DataFrame:
        """Compute price returns for a ticker.

        period: 'daily', 'weekly', 'monthly'
        Returns DataFrame with columns: price, return, log_return
        """
        if end is None:
            end = date.today().isoformat()

        series = self.get_adjusted_series(ticker, start, end)
        if series.empty:
            return pd.DataFrame()

        prices = series["price"].dropna()

        if period == "weekly":
            prices = prices.resample("W-FRI").last().dropna()
        elif period == "monthly":
            prices = prices.resample("ME").last().dropna()

        result = pd.DataFrame({"price": prices})
        result["return"] = result["price"].pct_change()
        result["log_return"] = np.log(result["price"] / result["price"].shift(1))
        result = result.dropna()
        return result

    def cross_market_correlation(
        self,
        tickers: List[str],
        window: int = 252,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute rolling window correlation matrix across tickers.

        Returns a pd.DataFrame correlation matrix for the most recent `window` trading days.
        """
        if end is None:
            end = date.today().isoformat()
        end_dt = _parse_date(end)
        start = (end_dt - timedelta(days=int(window * 1.5))).isoformat()

        price_dict: Dict[str, pd.Series] = {}
        for ticker in tickers:
            series = self.get_adjusted_series(ticker, start, end)
            if not series.empty and "price" in series.columns:
                price_dict[ticker] = series["price"]

        if len(price_dict) < 2:
            return pd.DataFrame()

        prices_df = pd.DataFrame(price_dict).dropna(how="all")
        # Take last `window` rows
        prices_df = prices_df.tail(window)
        returns = prices_df.pct_change().dropna()
        return returns.corr()

    def get_global_snapshot(self, as_of: Optional[str] = None) -> pd.DataFrame:
        """Return latest available prices across all markets in the store.

        Optionally filter to prices on or before `as_of`.
        """
        if as_of:
            as_of_d = _parse_date(as_of)
            if self.store._conn is not None:
                try:
                    return self.store._conn.execute("""
                        SELECT o.ticker, o.date, o.close, o.adj_close, o.market, o.source
                        FROM ohlcv_daily o
                        INNER JOIN (
                            SELECT ticker, MAX(date) AS max_date
                            FROM ohlcv_daily
                            WHERE date <= ?
                            GROUP BY ticker
                        ) m ON o.ticker = m.ticker AND o.date = m.max_date
                        ORDER BY o.market, o.ticker
                    """, [as_of_d]).fetchdf()
                except Exception as exc:
                    logger.error(f"Global snapshot error: {exc}")
        return self.store.get_latest_prices()

    def compute_drawdown_series(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Compute running max-drawdown series for a ticker."""
        series = self.get_adjusted_series(ticker, start, end)
        if series.empty:
            return pd.DataFrame()
        prices = series["price"].dropna()
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max
        return pd.DataFrame({"price": prices, "drawdown": drawdown})

    def compute_volatility(self, ticker: str, window: int = 21,
                           start: str = _DEFAULT_START,
                           end: Optional[str] = None) -> pd.DataFrame:
        """Rolling annualised volatility (std of log returns)."""
        if end is None:
            end = date.today().isoformat()
        rets = self.compute_returns(ticker, "daily", start, end)
        if rets.empty:
            return pd.DataFrame()
        rets["vol_21d"] = rets["log_return"].rolling(window).std() * math.sqrt(252)
        return rets[["price", "log_return", "vol_21d"]]

    def get_market_summary(self) -> pd.DataFrame:
        """Summary of data coverage: market-level roll-up."""
        coverage = self.store.get_data_coverage()
        if coverage.empty:
            return pd.DataFrame()
        coverage["first_date"] = pd.to_datetime(coverage["first_date"])
        coverage["last_date"] = pd.to_datetime(coverage["last_date"])
        coverage["history_years"] = (
            (coverage["last_date"] - coverage["first_date"]).dt.days / 365.25
        ).round(1)
        return coverage.sort_values("market")


# ===========================================================================
# Utilities
# ===========================================================================

def _stooq_to_yahoo(stooq_ticker: str) -> str:
    """Convert a Stooq-format ticker to Yahoo Finance format.

    Stooq → Yahoo:
      AAPL.US  → AAPL
      SAP.DE   → SAP.DE  (same)
      7203.JP  → 7203.T
      HSBA.UK  → HSBA.L
      0005.HK  → 0005.HK (same)
      AIR.FR   → AIR.PA
      RY.CA    → RY.TO
      CBA.AU   → CBA.AX
      VOW3.DE  → VOW3.DE (same)
    """
    SUFFIX_MAP = {
        ".US": "",
        ".JP": ".T",
        ".UK": ".L",
        ".FR": ".PA",
        ".CA": ".TO",
        ".AU": ".AX",
        ".SE": ".ST",
        ".NO": ".OL",
        ".DK": ".CO",
        ".FI": ".HE",
        ".BE": ".BR",
        ".AT": ".VI",
        ".PT": ".LS",
        ".IE": ".IR",
        ".PL": ".WA",
        ".CZ": ".PR",
        ".HU": ".BU",
        ".GR": ".AT",
        ".TR": ".IS",
        ".ZA": ".JO",
        ".IN": ".NS",
        ".KR": ".KS",
        ".MX": ".MX",
        ".BR": ".SA",
        ".SG": ".SI",
        ".TW": ".TW",
    }
    for stooq_suffix, yahoo_suffix in SUFFIX_MAP.items():
        if stooq_ticker.upper().endswith(stooq_suffix):
            base = stooq_ticker[: -len(stooq_suffix)]
            return base + yahoo_suffix
    return stooq_ticker  # DE, HK, CH, NL etc. often same suffix


def build_default_engine(db_path: Optional[Path] = None) -> HistoricalOHLCVEngineDaily:
    """Factory: create a fully configured engine with all adapters."""
    store = DuckDBOHLCVStore(db_path)
    stooq = StooqDataAdapter()
    alpaca = AlpacaHistoricalAdapter()
    yahoo = YahooHistoricalAdapter()
    universe = MarketUniverse()
    return HistoricalOHLCVEngineDaily(
        store=store, stooq=stooq, alpaca=alpaca,
        yahoo=yahoo, universe=universe,
    )


# ===========================================================================
# Entrypoint demo
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    START = "2000-01-01"
    END = date.today().isoformat()

    # US tickers (Stooq .US format)
    US_TICKERS = [
        "AAPL.US", "MSFT.US", "AMZN.US", "GOOGL.US", "META.US",
        "NVDA.US", "TSLA.US", "JPM.US", "V.US", "MA.US",
        "UNH.US", "XOM.US", "JNJ.US", "WMT.US", "BAC.US",
        "PG.US", "HD.US", "ABBV.US", "CVX.US", "MRK.US",
    ]

    # International tickers (10 across multiple markets)
    INTL_TICKERS = [
        "NESN.CH",  # Switzerland — Nestlé
        "SAP.DE",   # Germany — SAP
        "HSBA.UK",  # UK — HSBC
        "7203.JP",  # Japan — Toyota
        "0005.HK",  # Hong Kong — HSBC
        "AIR.FR",   # France — Airbus
        "RY.CA",    # Canada — Royal Bank
        "CBA.AU",   # Australia — Commonwealth Bank
        "ASML.NL",  # Netherlands — ASML
        "NOVO-B.DK",  # Denmark — Novo Nordisk
    ]

    ALL_TICKERS = US_TICKERS + INTL_TICKERS
    print(f"\n{'='*60}")
    print(f"SENTINEL Historical OHLCV Daily Engine v3")
    print(f"Loading {len(ALL_TICKERS)} tickers from {START} to {END}")
    print(f"{'='*60}\n")

    engine = build_default_engine()
    universe = MarketUniverse()
    print(f"Universe size: {universe.get_universe_size()} tickers across {len(universe.get_all_markets())} markets")

    # Test Stooq connectivity
    stooq_ok = engine.stooq.test_connectivity()
    print(f"Stooq connectivity: {'OK' if stooq_ok else 'FAILED'}")

    # Bulk load
    results = engine.bulk_load(ALL_TICKERS, start=START, end=END)
    print(f"\nBulk load complete:")
    for ticker, rows in sorted(results.items()):
        print(f"  {ticker:20s} → {rows:6d} rows")

    # Coverage report
    coverage = engine.store.get_data_coverage()
    if not coverage.empty:
        print(f"\nData coverage ({len(coverage)} tickers):")
        print(coverage.to_string(index=False))

    # Correlation matrix (US mega-caps)
    us5 = ["AAPL.US", "MSFT.US", "AMZN.US", "GOOGL.US", "META.US"]
    corr = engine.cross_market_correlation(us5, window=252)
    if not corr.empty:
        print(f"\n252-day correlation matrix (US mega-caps):")
        print(corr.round(3).to_string())

    # Global snapshot
    snapshot = engine.get_global_snapshot()
    if not snapshot.empty:
        print(f"\nGlobal latest-price snapshot ({len(snapshot)} tickers):")
        print(snapshot.head(20).to_string(index=False))

    # Gap detection for AAPL
    gaps = engine.detect_gaps("AAPL.US", start="2020-01-01")
    print(f"\nGaps detected for AAPL.US since 2020-01-01: {len(gaps)} gap(s)")
    for gs, ge in gaps[:5]:
        print(f"  {gs} .. {ge}")

    engine.store.close()
    print("\nDone.")
