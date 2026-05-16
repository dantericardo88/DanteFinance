"""
futures_v3.py — Comprehensive futures term structure & continuous contracts (dim_005).

Futures universe: 40+ contracts across equity index, energy, metals, agriculture,
fixed income, FX proxies, and crypto futures.

Data sources (all free):
  1. yfinance — primary front-month OHLCV, multi-expiry suffix symbols
  2. CME Group free delayed API — multiple contract months per product
  3. FRED — treasury futures proxies, CPI/PPI for convenience yield estimates
  4. CFTC COT v2 — positioning context via existing sentinel module

Analytics:
  - Term structure: M1/M2/M3 prices, contango/backwardation classification
  - Roll yield annualized, calendar spreads (abs + pct)
  - Panama backward price adjustment for continuous contracts
  - Roll-date detection via volume crossover
  - Seasonal return patterns (monthly, weekly)
  - Historical volatility of continuous contracts (21d, 63d)
  - Term structure of volatility (HV M1 vs M2)
  - Convenience yield estimation (cost-of-carry model)
  - Basis (futures price - spot price)

Storage (SQLite):
  - futures_universe        — contract metadata
  - contract_prices         — per-expiry OHLCV
  - continuous_series       — adjusted + unadjusted continuous prices
  - term_structure_snapshots— daily snapshot of M1/M2/M3 + analytics
  - roll_history            — detected roll events with adjustment factors

FastAPI router: /futures/v3
  GET /universe
  GET /term-structure/{symbol}
  GET /continuous/{symbol}?adjusted=true
  GET /roll-yield/{symbol}
  GET /contango-backwardation
  GET /basis/{symbol}
  GET /calendar-spread/{symbol}
  GET /seasonal/{symbol}

Dependencies: requests, sqlite3, pandas, numpy, fastapi, yfinance
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    yf = None  # type: ignore[assignment]
    _YF_OK = False

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "futures_v3.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept": "application/json,text/plain,*/*",
}
_TIMEOUT = 20.0
_RATE_DELAY = 0.25  # seconds between external requests

# ---------------------------------------------------------------------------
# Futures Universe Definition
# ---------------------------------------------------------------------------

@dataclass
class FuturesContract:
    """Metadata for a futures product."""
    symbol: str          # canonical root (e.g. "CL")
    yf_symbol: str       # yfinance front-month ticker (e.g. "CL=F")
    name: str
    exchange: str
    asset_class: str     # "equity_index","energy","metals","agriculture","fixed_income","fx","crypto"
    cme_product_id: Optional[str] = None   # CME product ID for term structure API
    spot_symbol: Optional[str] = None      # yfinance spot ticker for basis calc
    expiry_convention: str = "monthly"     # "monthly" | "quarterly"
    tick_size: float = 0.01
    contract_size: float = 1.0
    unit: str = "USD"
    # Optional: list of known expiry-suffix yfinance symbols for M1/M2/M3
    expiry_tickers: List[str] = field(default_factory=list)
    is_continuous: bool = True


# Full universe — 44 contracts
FUTURES_UNIVERSE: List[FuturesContract] = [
    # ── Equity Index ──────────────────────────────────────────────────────────
    FuturesContract("ES", "ES=F", "E-mini S&P 500", "CME", "equity_index",
                    cme_product_id="13601", spot_symbol="^GSPC",
                    tick_size=0.25, contract_size=50.0),
    FuturesContract("NQ", "NQ=F", "E-mini NASDAQ-100", "CME", "equity_index",
                    cme_product_id="13603", spot_symbol="^NDX",
                    tick_size=0.25, contract_size=20.0),
    FuturesContract("YM", "YM=F", "E-mini Dow Jones", "CBOT", "equity_index",
                    cme_product_id="12460", spot_symbol="^DJI",
                    tick_size=1.0, contract_size=5.0),
    FuturesContract("RTY", "RTY=F", "E-mini Russell 2000", "CME", "equity_index",
                    cme_product_id="239085", spot_symbol="^RUT",
                    tick_size=0.1, contract_size=50.0),
    FuturesContract("VX", "^VIX", "VIX Futures (spot proxy)", "CBOE", "equity_index",
                    spot_symbol="^VIX", tick_size=0.05, contract_size=1000.0),
    # ── Energy ────────────────────────────────────────────────────────────────
    FuturesContract("CL", "CL=F", "WTI Crude Oil", "NYMEX", "energy",
                    cme_product_id="13563", spot_symbol="CL=F",
                    tick_size=0.01, contract_size=1000.0, unit="BBL"),
    FuturesContract("NG", "NG=F", "Natural Gas", "NYMEX", "energy",
                    cme_product_id="13455", spot_symbol="NG=F",
                    tick_size=0.001, contract_size=10000.0, unit="MMBTU"),
    FuturesContract("RB", "RB=F", "RBOB Gasoline", "NYMEX", "energy",
                    cme_product_id="13522", tick_size=0.0001, contract_size=42000.0),
    FuturesContract("HO", "HO=F", "Heating Oil", "NYMEX", "energy",
                    cme_product_id="13691", tick_size=0.0001, contract_size=42000.0),
    # ── Metals ────────────────────────────────────────────────────────────────
    FuturesContract("GC", "GC=F", "Gold", "COMEX", "metals",
                    cme_product_id="13836", spot_symbol="GC=F",
                    tick_size=0.1, contract_size=100.0, unit="OZ"),
    FuturesContract("SI", "SI=F", "Silver", "COMEX", "metals",
                    cme_product_id="13838", spot_symbol="SI=F",
                    tick_size=0.005, contract_size=5000.0, unit="OZ"),
    FuturesContract("HG", "HG=F", "Copper", "COMEX", "metals",
                    cme_product_id="13440", spot_symbol="HG=F",
                    tick_size=0.0005, contract_size=25000.0, unit="LB"),
    FuturesContract("PL", "PL=F", "Platinum", "NYMEX", "metals",
                    cme_product_id="13936", tick_size=0.1, contract_size=50.0),
    FuturesContract("PA", "PA=F", "Palladium", "NYMEX", "metals",
                    cme_product_id="13935", tick_size=0.05, contract_size=100.0),
    # ── Agriculture ───────────────────────────────────────────────────────────
    FuturesContract("ZC", "ZC=F", "Corn", "CBOT", "agriculture",
                    cme_product_id="13010", tick_size=0.25, contract_size=5000.0, unit="BU"),
    FuturesContract("ZS", "ZS=F", "Soybeans", "CBOT", "agriculture",
                    cme_product_id="13061", tick_size=0.25, contract_size=5000.0, unit="BU"),
    FuturesContract("ZW", "ZW=F", "Wheat (CBOT)", "CBOT", "agriculture",
                    cme_product_id="13028", tick_size=0.25, contract_size=5000.0, unit="BU"),
    FuturesContract("KC", "KC=F", "Coffee C", "ICE", "agriculture",
                    tick_size=0.05, contract_size=37500.0, unit="LB"),
    FuturesContract("CT", "CT=F", "Cotton #2", "ICE", "agriculture",
                    tick_size=0.01, contract_size=50000.0, unit="LB"),
    FuturesContract("SB", "SB=F", "Sugar #11", "ICE", "agriculture",
                    tick_size=0.01, contract_size=112000.0, unit="LB"),
    FuturesContract("CC", "CC=F", "Cocoa", "ICE", "agriculture",
                    tick_size=1.0, contract_size=10.0, unit="MT"),
    # ── Fixed Income ──────────────────────────────────────────────────────────
    FuturesContract("ZB", "ZB=F", "30-Year T-Bond", "CBOT", "fixed_income",
                    cme_product_id="13064", tick_size=0.03125, contract_size=100000.0),
    FuturesContract("ZN", "ZN=F", "10-Year T-Note", "CBOT", "fixed_income",
                    cme_product_id="13874", tick_size=0.015625, contract_size=100000.0),
    FuturesContract("ZF", "ZF=F", "5-Year T-Note", "CBOT", "fixed_income",
                    cme_product_id="13876", tick_size=0.0078125, contract_size=100000.0),
    FuturesContract("ZT", "ZT=F", "2-Year T-Note", "CBOT", "fixed_income",
                    cme_product_id="13879", tick_size=0.00390625, contract_size=200000.0),
    # ── FX (spot proxies — no listed futures on yfinance free) ───────────────
    FuturesContract("EURUSD", "EURUSD=X", "EUR/USD FX Futures Proxy", "CME", "fx",
                    spot_symbol="EURUSD=X", tick_size=0.00005, contract_size=125000.0,
                    is_continuous=False),
    FuturesContract("JPYUSD", "JPYUSD=X", "JPY/USD FX Futures Proxy", "CME", "fx",
                    spot_symbol="JPYUSD=X", tick_size=0.0000005, contract_size=12500000.0,
                    is_continuous=False),
    FuturesContract("GBPUSD", "GBPUSD=X", "GBP/USD FX Futures Proxy", "CME", "fx",
                    spot_symbol="GBPUSD=X", tick_size=0.0001, contract_size=62500.0,
                    is_continuous=False),
    FuturesContract("AUDUSD", "AUDUSD=X", "AUD/USD FX Futures Proxy", "CME", "fx",
                    spot_symbol="AUDUSD=X", tick_size=0.0001, contract_size=100000.0,
                    is_continuous=False),
    FuturesContract("CADUSD", "CADUSD=X", "CAD/USD FX Futures Proxy", "CME", "fx",
                    spot_symbol="CADUSD=X", tick_size=0.00005, contract_size=100000.0,
                    is_continuous=False),
    # ── Crypto (CME Bitcoin/Ether Futures) ────────────────────────────────────
    FuturesContract("BTC", "BTC=F", "CME Bitcoin Futures", "CME", "crypto",
                    cme_product_id="10679", spot_symbol="BTC-USD",
                    tick_size=5.0, contract_size=5.0, unit="BTC"),
    FuturesContract("ETH", "ETH=F", "CME Ether Futures", "CME", "crypto",
                    cme_product_id="10680", spot_symbol="ETH-USD",
                    tick_size=0.25, contract_size=50.0, unit="ETH"),
    # ── Additional equity index & vol ─────────────────────────────────────────
    FuturesContract("MES", "MES=F", "Micro E-mini S&P 500", "CME", "equity_index",
                    spot_symbol="^GSPC", tick_size=0.25, contract_size=5.0),
    FuturesContract("MNQ", "MNQ=F", "Micro E-mini NASDAQ-100", "CME", "equity_index",
                    spot_symbol="^NDX", tick_size=0.25, contract_size=2.0),
    # ── Additional grains ─────────────────────────────────────────────────────
    FuturesContract("ZL", "ZL=F", "Soybean Oil", "CBOT", "agriculture",
                    tick_size=0.01, contract_size=60000.0, unit="LB"),
    FuturesContract("ZM", "ZM=F", "Soybean Meal", "CBOT", "agriculture",
                    tick_size=0.1, contract_size=100.0, unit="TON"),
    FuturesContract("ZO", "ZO=F", "Oats", "CBOT", "agriculture",
                    tick_size=0.25, contract_size=5000.0, unit="BU"),
    FuturesContract("ZR", "ZR=F", "Rough Rice", "CBOT", "agriculture",
                    tick_size=0.005, contract_size=2000.0, unit="CWT"),
    # ── Livestock ─────────────────────────────────────────────────────────────
    FuturesContract("LE", "LE=F", "Live Cattle", "CME", "agriculture",
                    tick_size=0.025, contract_size=40000.0, unit="LB"),
    FuturesContract("HE", "HE=F", "Lean Hogs", "CME", "agriculture",
                    tick_size=0.025, contract_size=40000.0, unit="LB"),
    FuturesContract("GF", "GF=F", "Feeder Cattle", "CME", "agriculture",
                    tick_size=0.025, contract_size=50000.0, unit="LB"),
    # ── Additional metals ─────────────────────────────────────────────────────
    FuturesContract("ALI", "ALI=F", "Aluminum", "NYMEX", "metals",
                    tick_size=0.0001, contract_size=44000.0, unit="LB"),
    # ── Equity vol ────────────────────────────────────────────────────────────
    FuturesContract("VXM", "^VIX", "Mini-VIX Futures Proxy", "CBOE", "equity_index",
                    spot_symbol="^VIX", tick_size=0.05, contract_size=100.0),
]

# Build lookup by symbol
_UNIVERSE_MAP: Dict[str, FuturesContract] = {c.symbol: c for c in FUTURES_UNIVERSE}

# CME term structure API (free delayed data)
_CME_QUOTES_URL = (
    "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/{product_id}/G"
)

# Month code → ordinal (for expiry parsing)
_MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────────────────────

_DB_LOCK = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with _DB_LOCK, _get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS futures_universe (
            symbol          TEXT PRIMARY KEY,
            yf_symbol       TEXT NOT NULL,
            name            TEXT NOT NULL,
            exchange        TEXT NOT NULL,
            asset_class     TEXT NOT NULL,
            cme_product_id  TEXT,
            spot_symbol     TEXT,
            tick_size       REAL,
            contract_size   REAL,
            unit            TEXT,
            updated_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS contract_prices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            expiry_label    TEXT NOT NULL,   -- "M1","M2","M3" or "CLZ24"
            trade_date      TEXT NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          REAL,
            open_interest   REAL,
            source          TEXT DEFAULT 'yfinance',
            fetched_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(symbol, expiry_label, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_cp_sym_date
            ON contract_prices(symbol, trade_date);

        CREATE TABLE IF NOT EXISTS continuous_series (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            trade_date      TEXT NOT NULL,
            close_adj       REAL,            -- Panama backward-adjusted
            close_raw       REAL,            -- unadjusted front-month close
            volume          REAL,
            hv_21d          REAL,
            hv_63d          REAL,
            roll_date_flag  INTEGER DEFAULT 0,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(symbol, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_cs_sym_date
            ON continuous_series(symbol, trade_date);

        CREATE TABLE IF NOT EXISTS term_structure_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            snap_date       TEXT NOT NULL,
            m1_price        REAL,
            m2_price        REAL,
            m3_price        REAL,
            m1_expiry       TEXT,
            m2_expiry       TEXT,
            m3_expiry       TEXT,
            contango_flag   INTEGER,         -- 1=contango, 0=backwardation
            roll_yield_ann  REAL,            -- annualised M1-M2 roll yield
            calendar_m1m2   REAL,            -- M1 - M2 absolute spread
            calendar_m2m3   REAL,
            calendar_m1m3   REAL,
            basis           REAL,            -- futures - spot
            convenience_yield REAL,
            hv_m1_21d       REAL,
            hv_m2_21d       REAL,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(symbol, snap_date)
        );

        CREATE TABLE IF NOT EXISTS roll_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            roll_date       TEXT NOT NULL,
            near_expiry     TEXT,
            far_expiry      TEXT,
            near_price      REAL,
            far_price       REAL,
            price_adj_factor REAL,           -- additive offset applied (Panama)
            detection_method TEXT DEFAULT 'volume_crossover',
            UNIQUE(symbol, roll_date)
        );
        """)
        # Populate universe metadata
        _upsert_universe(conn)
        conn.commit()
    logger.info("futures_v3 DB initialised at %s", _DB_PATH)


def _upsert_universe(conn: sqlite3.Connection) -> None:
    for c in FUTURES_UNIVERSE:
        conn.execute("""
            INSERT OR REPLACE INTO futures_universe
            (symbol, yf_symbol, name, exchange, asset_class, cme_product_id,
             spot_symbol, tick_size, contract_size, unit, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))
        """, (c.symbol, c.yf_symbol, c.name, c.exchange, c.asset_class,
              c.cme_product_id, c.spot_symbol, c.tick_size, c.contract_size, c.unit))


# Run once at import time
_init_db()

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _trading_days_between(d1: date, d2: date) -> int:
    """Approximate trading days between two dates (Mon–Fri only)."""
    total = 0
    cur = min(d1, d2)
    end = max(d1, d2)
    while cur < end:
        if cur.weekday() < 5:
            total += 1
        cur += timedelta(days=1)
    return max(total, 1)


def _annualise_roll_yield(near: float, far: float, days_between: int) -> float:
    """Roll yield = (near - far) / near × (365 / days_between)."""
    if near <= 0 or days_between <= 0:
        return 0.0
    return (near - far) / near * (365.0 / days_between)


def _hv(series: pd.Series, window: int) -> pd.Series:
    """Historical volatility (annualised) from log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ──────────────────────────────────────────────────────────────────────────────
# YFINANCE FETCHER
# ──────────────────────────────────────────────────────────────────────────────

class YFinanceFetcher:
    """Thin wrapper around yfinance for front-month futures OHLCV."""

    def fetch(
        self,
        yf_ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        if not _YF_OK:
            logger.warning("yfinance not installed — skipping %s", yf_ticker)
            return None
        try:
            raw = yf.download(
                yf_ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                return None
            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index).normalize()
            return raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        except Exception as exc:
            logger.warning("yfinance error for %s: %s", yf_ticker, exc)
            return None

    def fetch_multi(
        self,
        tickers: List[str],
        period: str = "2y",
    ) -> Dict[str, Optional[pd.DataFrame]]:
        result: Dict[str, Optional[pd.DataFrame]] = {}
        for ticker in tickers:
            result[ticker] = self.fetch(ticker, period=period)
            time.sleep(_RATE_DELAY)
        return result


_yf_fetcher = YFinanceFetcher()

# ──────────────────────────────────────────────────────────────────────────────
# CME GROUP TERM STRUCTURE ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CMEContractQuote:
    expiry_label: str   # e.g. "CLZ24"
    month_code: str
    year: int
    last: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    change: Optional[float]


class CMETermStructureAdapter:
    """
    Fetches multiple contract months from CME Group's free delayed quote API.
    URL: https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/{productId}/G
    Returns JSON with 'quotes' list, each having priorSettle, volume, openInterest, etc.
    """

    _session: requests.Session

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def fetch_term_structure(
        self, product_id: str, max_contracts: int = 6
    ) -> List[CMEContractQuote]:
        url = _CME_QUOTES_URL.format(product_id=product_id)
        try:
            resp = self._session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("CME API error for product %s: %s", product_id, exc)
            return []

        quotes_raw = data.get("quotes", [])
        results: List[CMEContractQuote] = []
        for q in quotes_raw[:max_contracts]:
            expiry = q.get("expirationCode", "")
            last = _safe_float(q.get("last") or q.get("priorSettle"))
            vol = _safe_float(q.get("volume"))
            oi = _safe_float(q.get("openInterest"))
            chg = _safe_float(q.get("change"))
            # Parse month/year from expiry code (e.g. "CLZ24", "Z24")
            month_code, year = _parse_expiry_code(expiry)
            results.append(CMEContractQuote(
                expiry_label=expiry,
                month_code=month_code,
                year=year,
                last=last,
                volume=vol,
                open_interest=oi,
                change=chg,
            ))
        return results


def _parse_expiry_code(code: str) -> Tuple[str, int]:
    """Extract month code and 2-digit year from codes like 'CLZ24' or 'Z24' or 'Z2024'."""
    if not code:
        return "", 0
    # Strip leading alpha (product) chars, find the last letter (month) + digits (year)
    import re
    m = re.search(r"([FGHJKMNQUVXZ])(\d{2,4})$", code.upper())
    if m:
        month_code = m.group(1)
        yr_str = m.group(2)
        year = int(yr_str) if len(yr_str) == 4 else 2000 + int(yr_str)
        return month_code, year
    return "", 0


_cme_adapter = CMETermStructureAdapter()

# ──────────────────────────────────────────────────────────────────────────────
# TERM STRUCTURE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TermStructurePoint:
    symbol: str
    snap_date: date
    m1_price: Optional[float]
    m2_price: Optional[float]
    m3_price: Optional[float]
    m1_expiry: str
    m2_expiry: str
    m3_expiry: str
    contango_flag: Optional[bool]       # True=contango, False=backwardation
    roll_yield_ann: Optional[float]     # M1→M2 annualised
    roll_yield_m2m3: Optional[float]    # M2→M3 annualised
    calendar_m1m2: Optional[float]
    calendar_m1m3: Optional[float]
    calendar_m2m3: Optional[float]
    basis: Optional[float]              # futures (M1) - spot
    convenience_yield: Optional[float]
    hv_m1_21d: Optional[float]
    hv_m2_21d: Optional[float]


class TermStructureEngine:
    """
    Builds term structure snapshots for a given symbol using:
      1. CME Group API (if cme_product_id is set)
      2. yfinance expiry-suffix tickers (where available)
      3. Fall back: use front-month as M1, shift series by ~30/60 days for M2/M3
    """

    def get_term_structure(self, symbol: str) -> Optional[TermStructurePoint]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            logger.warning("Unknown symbol: %s", symbol)
            return None

        snap_date = date.today()
        m1 = m2 = m3 = None
        m1_label = m2_label = m3_label = ""
        days_m1m2 = 30
        days_m2m3 = 30

        # ── Try CME API first ─────────────────────────────────────────────
        if contract.cme_product_id:
            quotes = _cme_adapter.fetch_term_structure(contract.cme_product_id, max_contracts=6)
            time.sleep(_RATE_DELAY)
            valid = [q for q in quotes if q.last is not None and q.last > 0]
            if len(valid) >= 2:
                m1 = valid[0].last
                m1_label = valid[0].expiry_label
                m2 = valid[1].last
                m2_label = valid[1].expiry_label
                if len(valid) >= 3:
                    m3 = valid[2].last
                    m3_label = valid[2].expiry_label

        # ── Fall back: yfinance front-month ───────────────────────────────
        if m1 is None:
            df = _yf_fetcher.fetch(contract.yf_symbol, period="5d")
            if df is not None and not df.empty:
                m1 = _safe_float(df["Close"].iloc[-1])
                m1_label = "M1"

        # ── Spot price for basis ──────────────────────────────────────────
        spot = None
        if contract.spot_symbol and contract.spot_symbol != contract.yf_symbol:
            spot_df = _yf_fetcher.fetch(contract.spot_symbol, period="5d")
            if spot_df is not None and not spot_df.empty:
                spot = _safe_float(spot_df["Close"].iloc[-1])
            time.sleep(_RATE_DELAY)

        # ── Historical volatility from continuous series ───────────────────
        hv_m1 = hv_m2 = None
        hist_df = self._load_continuous_prices(symbol, days=90)
        if hist_df is not None and len(hist_df) >= 22:
            hv_series = _hv(hist_df["close_raw"].dropna(), 21)
            hv_m1 = _safe_float(hv_series.iloc[-1]) if not hv_series.empty else None

        # ── Analytics ─────────────────────────────────────────────────────
        contango = None
        roll_yield = None
        roll_yield_m2m3 = None
        cal_m1m2 = cal_m1m3 = cal_m2m3 = None
        basis = None
        convenience = None

        if m1 and m2:
            contango = m2 > m1
            cal_m1m2 = m1 - m2
            roll_yield = _annualise_roll_yield(m1, m2, days_m1m2)
            if m3:
                cal_m1m3 = m1 - m3
                cal_m2m3 = m2 - m3
                roll_yield_m2m3 = _annualise_roll_yield(m2, m3, days_m2m3)

        if m1 and spot:
            basis = m1 - spot
            # Convenience yield from cost-of-carry: cy ≈ r - (F/S - 1) × (365/T)
            # Use 5% as generic risk-free rate proxy; T=30 days to front expiry
            risk_free = 0.05
            T = 30 / 365.0
            if spot > 0 and T > 0:
                carry = (m1 / spot - 1) / T
                convenience = risk_free - carry

        ts = TermStructurePoint(
            symbol=symbol,
            snap_date=snap_date,
            m1_price=m1,
            m2_price=m2,
            m3_price=m3,
            m1_expiry=m1_label,
            m2_expiry=m2_label,
            m3_expiry=m3_label,
            contango_flag=contango,
            roll_yield_ann=roll_yield,
            roll_yield_m2m3=roll_yield_m2m3,
            calendar_m1m2=cal_m1m2,
            calendar_m1m3=cal_m1m3,
            calendar_m2m3=cal_m2m3,
            basis=basis,
            convenience_yield=convenience,
            hv_m1_21d=hv_m1,
            hv_m2_21d=hv_m2,
        )
        self._persist_snapshot(ts)
        return ts

    def _load_continuous_prices(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        with _get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT trade_date, close_raw FROM continuous_series "
                "WHERE symbol=? AND trade_date>=? ORDER BY trade_date",
                conn, params=(symbol, cutoff)
            )
        if df.empty:
            return None
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df

    def _persist_snapshot(self, ts: TermStructurePoint) -> None:
        with _DB_LOCK, _get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO term_structure_snapshots
                (symbol, snap_date, m1_price, m2_price, m3_price,
                 m1_expiry, m2_expiry, m3_expiry,
                 contango_flag, roll_yield_ann, calendar_m1m2,
                 calendar_m2m3, calendar_m1m3, basis, convenience_yield,
                 hv_m1_21d, hv_m2_21d, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            """, (
                ts.symbol, ts.snap_date.isoformat(),
                ts.m1_price, ts.m2_price, ts.m3_price,
                ts.m1_expiry, ts.m2_expiry, ts.m3_expiry,
                1 if ts.contango_flag else (0 if ts.contango_flag is not None else None),
                ts.roll_yield_ann,
                ts.calendar_m1m2, ts.calendar_m2m3, ts.calendar_m1m3,
                ts.basis, ts.convenience_yield,
                ts.hv_m1_21d, ts.hv_m2_21d,
            ))
            conn.commit()


_ts_engine = TermStructureEngine()

# ──────────────────────────────────────────────────────────────────────────────
# CONTINUOUS CONTRACT BUILDER (Panama Backward Adjustment)
# ──────────────────────────────────────────────────────────────────────────────

class ContinuousContractBuilder:
    """
    Builds Panama backward-adjusted continuous price series.

    Algorithm:
      1. Fetch front-month OHLCV history (2 years) via yfinance
      2. Detect roll dates via volume crossover heuristic:
         - Since yfinance only gives the front-month symbol, we simulate roll
           detection using sharp open-interest-like drops in volume.
         - We look for calendar-month transitions (third Friday → next contract).
      3. On each detected roll date, compute additive adjustment:
         adj_factor = price_near (day before roll) - price_far (day of roll)
         Apply cumulatively backwards to keep the series continuous.
      4. Store both adjusted and unadjusted in continuous_series table.
    """

    def build_continuous(
        self, symbol: str, lookback_years: int = 2, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            return None

        if not contract.is_continuous:
            logger.info("Skipping continuous build for FX proxy %s", symbol)
            return self._build_fx_proxy(contract)

        # Fetch front-month data
        period = f"{lookback_years}y"
        df = _yf_fetcher.fetch(contract.yf_symbol, period=period)
        if df is None or df.empty:
            logger.warning("No yfinance data for %s", symbol)
            return None

        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        # Roll date detection: quarterly roll windows (March, June, Sep, Dec for equity futures)
        # For other assets: monthly rolls.
        roll_dates = self._detect_roll_dates(df, contract)
        roll_records = []

        # Build Panama series: start with raw close, apply adjustments backwards
        close_raw = df["Close"].copy()
        close_adj = close_raw.copy()
        cumulative_adj = 0.0

        for roll_dt in sorted(roll_dates, reverse=True):
            # Find the business day before the roll
            idx_pos = df.index.searchsorted(roll_dt)
            if idx_pos == 0 or idx_pos >= len(df):
                continue
            near_price = _safe_float(df["Close"].iloc[idx_pos - 1])
            far_price = _safe_float(df["Close"].iloc[idx_pos])
            if near_price is None or far_price is None:
                continue
            # Additive adjustment: shift all earlier prices up by the gap
            adj_factor = near_price - far_price
            cumulative_adj += adj_factor
            # Apply to all rows before the roll date
            mask = df.index < roll_dt
            close_adj[mask] += adj_factor
            roll_records.append({
                "symbol": symbol,
                "roll_date": roll_dt.date().isoformat(),
                "near_price": near_price,
                "far_price": far_price,
                "price_adj_factor": adj_factor,
            })

        # Compute HV
        hv21 = _hv(close_adj, 21)
        hv63 = _hv(close_adj, 63)

        # Build output DataFrame
        result = pd.DataFrame({
            "trade_date": df.index.date,
            "close_adj": close_adj.values,
            "close_raw": close_raw.values,
            "volume": df["Volume"].values,
            "hv_21d": hv21.values,
            "hv_63d": hv63.values,
            "roll_date_flag": [
                1 if d in {r["roll_date"] for r in roll_records} else 0
                for d in [str(dt.date()) for dt in df.index]
            ],
        })

        self._persist_continuous(symbol, result)
        self._persist_rolls(roll_records)
        return result

    def _detect_roll_dates(
        self, df: pd.DataFrame, contract: FuturesContract
    ) -> List[pd.Timestamp]:
        """
        Heuristic roll date detection.

        For equity index futures (quarterly): third Friday of March/Jun/Sep/Dec.
        For commodities: volume-based detection — large drops in volume signal roll.
        For fixed income: quarterly (same as equity).
        """
        roll_dates: List[pd.Timestamp] = []

        if contract.asset_class in ("equity_index", "fixed_income"):
            # Quarterly expiry: third Friday of March, June, September, December
            roll_months = {3, 6, 9, 12}
            for ts in df.index:
                if ts.month in roll_months:
                    third_fri = self._third_friday(ts.year, ts.month)
                    if pd.Timestamp(third_fri) not in roll_dates:
                        roll_dates.append(pd.Timestamp(third_fri))
        elif contract.asset_class == "crypto":
            # CME crypto: last Friday of every month
            for ts in df.index:
                last_fri = self._last_friday(ts.year, ts.month)
                tslf = pd.Timestamp(last_fri)
                if tslf not in roll_dates:
                    roll_dates.append(tslf)
        else:
            # Monthly expiry: detect by volume-drop heuristic
            vol = df["Volume"].copy().fillna(0)
            # Normalise: rolling 5-day avg
            vol_5 = vol.rolling(5).mean()
            # A roll occurs when volume drops > 40% relative to 5d avg, near month-end
            for i in range(5, len(df)):
                ts = df.index[i]
                day_of_month = ts.day
                if not (22 <= day_of_month <= 31):
                    continue
                if vol_5.iloc[i] > 0:
                    vol_ratio = vol.iloc[i] / vol_5.iloc[i]
                    if vol_ratio < 0.6:
                        # Next trading day is the roll
                        if i + 1 < len(df):
                            roll_dates.append(df.index[i + 1])

        # Deduplicate and filter to dates within the data range
        unique_rolls = sorted(set(roll_dates))
        return [r for r in unique_rolls if r in df.index]

    def _third_friday(self, year: int, month: int) -> date:
        """Return the third Friday of the given month."""
        first = date(year, month, 1)
        first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
        return first_friday + timedelta(weeks=2)

    def _last_friday(self, year: int, month: int) -> date:
        """Return the last Friday of the given month."""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        days_since_fri = (last_day.weekday() - 4) % 7
        return last_day - timedelta(days=days_since_fri)

    def _build_fx_proxy(self, contract: FuturesContract) -> Optional[pd.DataFrame]:
        """For FX spot-proxy contracts, just return raw spot series."""
        df = _yf_fetcher.fetch(contract.yf_symbol, period="2y")
        if df is None or df.empty:
            return None
        result = pd.DataFrame({
            "trade_date": df.index.date,
            "close_adj": df["Close"].values,
            "close_raw": df["Close"].values,
            "volume": df["Volume"].values,
            "hv_21d": _hv(df["Close"], 21).values,
            "hv_63d": _hv(df["Close"], 63).values,
            "roll_date_flag": [0] * len(df),
        })
        self._persist_continuous(contract.symbol, result)
        return result

    def _persist_continuous(self, symbol: str, df: pd.DataFrame) -> None:
        rows = []
        for _, row in df.iterrows():
            rows.append((
                symbol,
                str(row["trade_date"]),
                _safe_float(row.get("close_adj")),
                _safe_float(row.get("close_raw")),
                _safe_float(row.get("volume")),
                _safe_float(row.get("hv_21d")),
                _safe_float(row.get("hv_63d")),
                int(row.get("roll_date_flag", 0)),
            ))
        with _DB_LOCK, _get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO continuous_series
                (symbol, trade_date, close_adj, close_raw, volume,
                 hv_21d, hv_63d, roll_date_flag, updated_at)
                VALUES (?,?,?,?,?,?,?,?,datetime('now'))
            """, rows)
            conn.commit()
        logger.info("Persisted %d continuous rows for %s", len(rows), symbol)

    def _persist_rolls(self, records: List[Dict[str, Any]]) -> None:
        with _DB_LOCK, _get_connection() as conn:
            for r in records:
                conn.execute("""
                    INSERT OR IGNORE INTO roll_history
                    (symbol, roll_date, near_price, far_price, price_adj_factor,
                     detection_method)
                    VALUES (?,?,?,?,?,'volume_crossover')
                """, (r["symbol"], r["roll_date"], r["near_price"],
                      r["far_price"], r["price_adj_factor"]))
            conn.commit()


_continuous_builder = ContinuousContractBuilder()

# ──────────────────────────────────────────────────────────────────────────────
# SEASONAL PATTERN ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SeasonalPattern:
    symbol: str
    monthly_returns: Dict[str, float]    # month name → avg monthly return
    best_month: str
    worst_month: str
    seasonality_score: float             # R² of monthly pattern vs random
    seasonal_notes: str


class SeasonalPatternEngine:
    """
    Compute monthly return seasonality from the continuous (adjusted) series.
    Uses up to 5 years of history.
    """

    _MONTH_NAMES = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    _SEASONAL_NOTES: Dict[str, str] = {
        "CL": "Energy demand peaks winter (Nov–Jan) and summer (Jun–Jul drive season).",
        "NG": "Strong winter seasonality (Nov–Feb); shoulder months (Mar, Oct) weak.",
        "GC": "Gold often strengthens Jan–Feb (Asian New Year) and Sep–Oct.",
        "ZC": "Corn planting season (Apr–May) and harvest (Sep–Oct) drive volatility.",
        "ZS": "Soybeans: South American harvest (Feb–Mar) and US planting (Apr–May).",
        "ZW": "Wheat seasonally weak post-harvest (Jul–Aug); winter wheat rally Dec–Feb.",
        "KC": "Coffee: Brazil harvest (Apr–Sep) depresses prices; frost risk May–Jul.",
        "CT": "Cotton: planting (Apr) and harvest (Sep–Nov) drive price cycles.",
        "SB": "Sugar: Brazil harvest (Apr–Oct) and Indian crop (Oct–Mar) key drivers.",
        "ES": "Equity futures: Jan Effect, summer doldrums (Jul–Aug), Oct volatility.",
        "VX": "VIX: seasonally higher Sep–Oct; lower Jun–Aug.",
    }

    def compute_seasonal(self, symbol: str) -> Optional[SeasonalPattern]:
        # Load up to 5 years of continuous data
        cutoff = (date.today() - timedelta(days=365 * 5)).isoformat()
        with _get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT trade_date, close_adj FROM continuous_series "
                "WHERE symbol=? AND trade_date>=? AND close_adj IS NOT NULL "
                "ORDER BY trade_date",
                conn, params=(symbol, cutoff)
            )

        if df.empty or len(df) < 60:
            # No stored data — fetch and build first
            cont = _continuous_builder.build_continuous(symbol, lookback_years=2)
            if cont is None:
                return None
            df = cont[["trade_date", "close_adj"]].dropna()
            df.columns = ["trade_date", "close_adj"]

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        # Monthly resampled returns
        monthly = df["close_adj"].resample("ME").last().pct_change().dropna()
        monthly_df = pd.DataFrame({
            "month": monthly.index.month,
            "return": monthly.values,
        })
        by_month = monthly_df.groupby("month")["return"].mean()

        monthly_returns = {
            self._MONTH_NAMES[m - 1]: round(float(v) * 100, 3)
            for m, v in by_month.items()
        }
        best_month_num = int(by_month.idxmax()) if not by_month.empty else 1
        worst_month_num = int(by_month.idxmin()) if not by_month.empty else 12
        best_month = self._MONTH_NAMES[best_month_num - 1]
        worst_month = self._MONTH_NAMES[worst_month_num - 1]

        # Seasonality score: std of monthly averages / overall std
        overall_std = float(monthly_df["return"].std()) if len(monthly_df) > 1 else 1.0
        monthly_std = float(by_month.std()) if len(by_month) > 1 else 0.0
        score = min(1.0, monthly_std / overall_std) if overall_std > 0 else 0.0

        notes = self._SEASONAL_NOTES.get(symbol, "No specific seasonal notes available.")
        return SeasonalPattern(
            symbol=symbol,
            monthly_returns=monthly_returns,
            best_month=best_month,
            worst_month=worst_month,
            seasonality_score=round(score, 4),
            seasonal_notes=notes,
        )


_seasonal_engine = SeasonalPatternEngine()

# ──────────────────────────────────────────────────────────────────────────────
# CONTANGO / BACKWARDATION SCREENER
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContangoBackwardationReading:
    symbol: str
    name: str
    asset_class: str
    status: str         # "contango" | "backwardation" | "flat" | "unknown"
    m1_price: Optional[float]
    m2_price: Optional[float]
    roll_yield_ann: Optional[float]
    snap_date: str


class ContangoBackwardationScreener:
    """Screen all universe contracts for current contango/backwardation status."""

    def screen_all(self, max_symbols: int = 20) -> List[ContangoBackwardationReading]:
        results: List[ContangoBackwardationReading] = []
        # First try DB cache (today's snapshots)
        today = date.today().isoformat()
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT symbol, m1_price, m2_price, contango_flag, roll_yield_ann, snap_date "
                "FROM term_structure_snapshots WHERE snap_date=?",
                (today,)
            ).fetchall()

        cached = {row["symbol"]: row for row in rows}
        # Check universe
        target_symbols = [
            c.symbol for c in FUTURES_UNIVERSE
            if c.asset_class not in ("fx",) and c.cme_product_id is not None
        ][:max_symbols]

        for symbol in target_symbols:
            contract = _UNIVERSE_MAP[symbol]
            if symbol in cached:
                row = cached[symbol]
                m1 = row["m1_price"]
                m2 = row["m2_price"]
                flag = row["contango_flag"]
                roll_yld = row["roll_yield_ann"]
                snap = row["snap_date"]
            else:
                # Live fetch
                ts = _ts_engine.get_term_structure(symbol)
                time.sleep(_RATE_DELAY)
                if ts is None:
                    continue
                m1 = ts.m1_price
                m2 = ts.m2_price
                flag = ts.contango_flag
                roll_yld = ts.roll_yield_ann
                snap = ts.snap_date.isoformat()

            if flag is None:
                status = "unknown"
            elif flag:
                status = "contango"
            else:
                m1v = m1 or 0
                m2v = m2 or 0
                if abs(m1v - m2v) < 0.001 * max(m1v, m2v, 1):
                    status = "flat"
                else:
                    status = "backwardation"

            results.append(ContangoBackwardationReading(
                symbol=symbol,
                name=contract.name,
                asset_class=contract.asset_class,
                status=status,
                m1_price=m1,
                m2_price=m2,
                roll_yield_ann=roll_yld,
                snap_date=snap,
            ))
        return results


_cb_screener = ContangoBackwardationScreener()

# ──────────────────────────────────────────────────────────────────────────────
# ROLL YIELD SERVICE
# ──────────────────────────────────────────────────────────────────────────────

class RollYieldService:
    """Retrieve or compute roll yield for a symbol."""

    def get_roll_yield(self, symbol: str) -> Dict[str, Any]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            return {"error": f"Unknown symbol: {symbol}"}

        # Try cached snapshot
        today = date.today().isoformat()
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT roll_yield_ann, m1_price, m2_price, m1_expiry, m2_expiry, snap_date "
                "FROM term_structure_snapshots WHERE symbol=? AND snap_date=?",
                (symbol, today)
            ).fetchone()

        if row and row["roll_yield_ann"] is not None:
            return {
                "symbol": symbol,
                "name": contract.name,
                "roll_yield_ann_pct": round(row["roll_yield_ann"] * 100, 4),
                "roll_yield_interpretation": (
                    "Positive (backwardation — long roll yield)" if row["roll_yield_ann"] > 0
                    else "Negative (contango — roll cost)"
                ),
                "m1_price": row["m1_price"],
                "m2_price": row["m2_price"],
                "m1_expiry": row["m1_expiry"],
                "m2_expiry": row["m2_expiry"],
                "snap_date": row["snap_date"],
                "source": "cache",
            }

        # Live compute
        ts = _ts_engine.get_term_structure(symbol)
        if ts is None or ts.roll_yield_ann is None:
            return {"symbol": symbol, "error": "Insufficient term structure data"}

        return {
            "symbol": symbol,
            "name": contract.name,
            "roll_yield_ann_pct": round(ts.roll_yield_ann * 100, 4),
            "roll_yield_interpretation": (
                "Positive (backwardation — long roll yield)" if ts.roll_yield_ann > 0
                else "Negative (contango — roll cost)"
            ),
            "m1_price": ts.m1_price,
            "m2_price": ts.m2_price,
            "m1_expiry": ts.m1_expiry,
            "m2_expiry": ts.m2_expiry,
            "snap_date": ts.snap_date.isoformat(),
            "source": "live",
        }


_roll_yield_svc = RollYieldService()

# ──────────────────────────────────────────────────────────────────────────────
# BASIS SERVICE
# ──────────────────────────────────────────────────────────────────────────────

class BasisService:
    """Compute basis (futures M1 - spot) for symbols with a spot_symbol."""

    def get_basis(self, symbol: str) -> Dict[str, Any]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            return {"error": f"Unknown symbol: {symbol}"}
        if not contract.spot_symbol:
            return {"symbol": symbol, "error": "No spot symbol configured for basis calculation."}

        # Try cached
        today = date.today().isoformat()
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT basis, convenience_yield, m1_price, snap_date "
                "FROM term_structure_snapshots WHERE symbol=? AND snap_date=?",
                (symbol, today)
            ).fetchone()

        if row and row["basis"] is not None:
            return {
                "symbol": symbol,
                "name": contract.name,
                "futures_m1": row["m1_price"],
                "basis": row["basis"],
                "convenience_yield": row["convenience_yield"],
                "snap_date": row["snap_date"],
                "interpretation": self._interpret_basis(symbol, row["basis"]),
                "source": "cache",
            }

        # Live fetch both
        fut_df = _yf_fetcher.fetch(contract.yf_symbol, period="5d")
        spot_df = _yf_fetcher.fetch(contract.spot_symbol, period="5d")
        fut_price = spot_price = None
        if fut_df is not None and not fut_df.empty:
            fut_price = _safe_float(fut_df["Close"].iloc[-1])
        if spot_df is not None and not spot_df.empty:
            spot_price = _safe_float(spot_df["Close"].iloc[-1])

        if fut_price is None or spot_price is None:
            return {"symbol": symbol, "error": "Could not fetch futures or spot price."}

        basis = fut_price - spot_price
        return {
            "symbol": symbol,
            "name": contract.name,
            "futures_m1": fut_price,
            "spot": spot_price,
            "basis": round(basis, 6),
            "basis_pct": round(basis / spot_price * 100, 4) if spot_price else None,
            "snap_date": today,
            "interpretation": self._interpret_basis(symbol, basis),
            "source": "live",
        }

    def _interpret_basis(self, symbol: str, basis: Optional[float]) -> str:
        if basis is None:
            return "N/A"
        contract = _UNIVERSE_MAP.get(symbol)
        asset_class = contract.asset_class if contract else "unknown"
        if basis > 0:
            if asset_class in ("metals", "energy", "agriculture"):
                return (
                    "Positive basis: futures > spot. Normal for storable commodities "
                    "(reflects storage + financing costs). May indicate contango."
                )
            return "Positive basis: futures trading at premium to spot (normal carry)."
        elif basis < 0:
            if asset_class in ("metals", "energy", "agriculture"):
                return (
                    "Negative basis: futures < spot. Backwardation signal — market "
                    "pricing near-term scarcity or high convenience yield."
                )
            return "Negative basis: futures at discount to spot."
        return "Zero basis: futures at par with spot."


_basis_svc = BasisService()

# ──────────────────────────────────────────────────────────────────────────────
# CALENDAR SPREAD SERVICE
# ──────────────────────────────────────────────────────────────────────────────

class CalendarSpreadService:
    """Compute M1-M2, M1-M3, M2-M3 calendar spreads."""

    def get_calendar_spread(self, symbol: str) -> Dict[str, Any]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            return {"error": f"Unknown symbol: {symbol}"}

        today = date.today().isoformat()
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT m1_price, m2_price, m3_price, m1_expiry, m2_expiry, m3_expiry, "
                "calendar_m1m2, calendar_m2m3, calendar_m1m3, snap_date "
                "FROM term_structure_snapshots WHERE symbol=? AND snap_date=?",
                (symbol, today)
            ).fetchone()

        if row and row["m1_price"] is not None:
            return self._format_spread(contract, dict(row))

        # Live
        ts = _ts_engine.get_term_structure(symbol)
        if ts is None:
            return {"symbol": symbol, "error": "Could not compute term structure."}

        return {
            "symbol": symbol,
            "name": contract.name,
            "asset_class": contract.asset_class,
            "m1": {"expiry": ts.m1_expiry, "price": ts.m1_price},
            "m2": {"expiry": ts.m2_expiry, "price": ts.m2_price},
            "m3": {"expiry": ts.m3_expiry, "price": ts.m3_price},
            "spreads": {
                "m1_m2_abs": round(ts.calendar_m1m2, 6) if ts.calendar_m1m2 is not None else None,
                "m1_m2_pct": (
                    round(ts.calendar_m1m2 / ts.m1_price * 100, 4)
                    if ts.calendar_m1m2 is not None and ts.m1_price
                    else None
                ),
                "m1_m3_abs": round(ts.calendar_m1m3, 6) if ts.calendar_m1m3 is not None else None,
                "m2_m3_abs": round(ts.calendar_m2m3, 6) if ts.calendar_m2m3 is not None else None,
            },
            "snap_date": ts.snap_date.isoformat(),
            "source": "live",
        }

    def _format_spread(
        self, contract: FuturesContract, row: Dict[str, Any]
    ) -> Dict[str, Any]:
        m1 = row.get("m1_price")
        m2 = row.get("m2_price")
        m3 = row.get("m3_price")
        cal_m1m2 = row.get("calendar_m1m2")
        cal_m1m3 = row.get("calendar_m1m3")
        cal_m2m3 = row.get("calendar_m2m3")
        return {
            "symbol": contract.symbol,
            "name": contract.name,
            "asset_class": contract.asset_class,
            "m1": {"expiry": row.get("m1_expiry"), "price": m1},
            "m2": {"expiry": row.get("m2_expiry"), "price": m2},
            "m3": {"expiry": row.get("m3_expiry"), "price": m3},
            "spreads": {
                "m1_m2_abs": round(cal_m1m2, 6) if cal_m1m2 is not None else None,
                "m1_m2_pct": (
                    round(cal_m1m2 / m1 * 100, 4) if cal_m1m2 is not None and m1 else None
                ),
                "m1_m3_abs": round(cal_m1m3, 6) if cal_m1m3 is not None else None,
                "m2_m3_abs": round(cal_m2m3, 6) if cal_m2m3 is not None else None,
            },
            "snap_date": row.get("snap_date"),
            "source": "cache",
        }


_cal_spread_svc = CalendarSpreadService()

# ──────────────────────────────────────────────────────────────────────────────
# CONTINUOUS SERIES RETRIEVAL SERVICE
# ──────────────────────────────────────────────────────────────────────────────

class ContinuousSeriesService:
    """Retrieve stored continuous series; build if not found."""

    def get_series(
        self,
        symbol: str,
        adjusted: bool = True,
        lookback_days: int = 504,
    ) -> Dict[str, Any]:
        contract = _UNIVERSE_MAP.get(symbol)
        if not contract:
            return {"error": f"Unknown symbol: {symbol}"}

        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        price_col = "close_adj" if adjusted else "close_raw"

        with _get_connection() as conn:
            df = pd.read_sql_query(
                f"SELECT trade_date, {price_col} as price, volume, hv_21d, hv_63d, "
                "roll_date_flag FROM continuous_series "
                "WHERE symbol=? AND trade_date>=? ORDER BY trade_date",
                conn, params=(symbol, cutoff)
            )

        if df.empty:
            # Build first
            built = _continuous_builder.build_continuous(symbol)
            if built is None:
                return {"symbol": symbol, "error": "No data available."}
            return self.get_series(symbol, adjusted, lookback_days)

        records = df.to_dict(orient="records")
        return {
            "symbol": symbol,
            "name": contract.name,
            "asset_class": contract.asset_class,
            "adjusted": adjusted,
            "method": "Panama backward adjustment" if adjusted else "Raw front-month",
            "count": len(records),
            "start_date": df["trade_date"].iloc[0] if not df.empty else None,
            "end_date": df["trade_date"].iloc[-1] if not df.empty else None,
            "data": records,
        }


_continuous_svc = ContinuousSeriesService()

# ──────────────────────────────────────────────────────────────────────────────
# BULK REFRESH SERVICE
# ──────────────────────────────────────────────────────────────────────────────

class BulkRefreshService:
    """Refresh continuous series and term structures for the full universe."""

    def refresh_continuous_all(
        self, asset_classes: Optional[List[str]] = None, max_symbols: int = 44
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        targets = [
            c for c in FUTURES_UNIVERSE
            if (asset_classes is None or c.asset_class in asset_classes)
        ][:max_symbols]

        for contract in targets:
            try:
                built = _continuous_builder.build_continuous(contract.symbol, lookback_years=2)
                results[contract.symbol] = "ok" if built is not None else "no_data"
            except Exception as exc:
                logger.warning("Continuous build failed for %s: %s", contract.symbol, exc)
                results[contract.symbol] = f"error: {exc}"
            time.sleep(_RATE_DELAY * 2)

        return results

    def refresh_term_structures(self, max_symbols: int = 20) -> Dict[str, str]:
        results: Dict[str, str] = {}
        targets = [
            c for c in FUTURES_UNIVERSE
            if c.cme_product_id is not None
        ][:max_symbols]

        for contract in targets:
            try:
                ts = _ts_engine.get_term_structure(contract.symbol)
                results[contract.symbol] = "ok" if ts else "no_data"
            except Exception as exc:
                logger.warning("TS fetch failed for %s: %s", contract.symbol, exc)
                results[contract.symbol] = f"error: {exc}"
            time.sleep(_RATE_DELAY * 2)

        return results


_bulk_refresh = BulkRefreshService()

# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────

class TermStructureResponse(BaseModel):
    symbol: str
    name: str
    asset_class: str
    snap_date: str
    m1_price: Optional[float] = None
    m2_price: Optional[float] = None
    m3_price: Optional[float] = None
    m1_expiry: str = ""
    m2_expiry: str = ""
    m3_expiry: str = ""
    contango_flag: Optional[bool] = None
    status: str = "unknown"
    roll_yield_ann_pct: Optional[float] = None
    roll_yield_m2m3_ann_pct: Optional[float] = None
    calendar_m1m2: Optional[float] = None
    calendar_m1m3: Optional[float] = None
    calendar_m2m3: Optional[float] = None
    basis: Optional[float] = None
    convenience_yield: Optional[float] = None
    hv_m1_21d_pct: Optional[float] = None
    hv_m2_21d_pct: Optional[float] = None


class UniverseContractResponse(BaseModel):
    symbol: str
    yf_symbol: str
    name: str
    exchange: str
    asset_class: str
    cme_product_id: Optional[str] = None
    spot_symbol: Optional[str] = None
    tick_size: float
    contract_size: float
    unit: str


class ContangoBackwardationResponse(BaseModel):
    symbol: str
    name: str
    asset_class: str
    status: str
    m1_price: Optional[float] = None
    m2_price: Optional[float] = None
    roll_yield_ann_pct: Optional[float] = None
    snap_date: str


class SeasonalResponse(BaseModel):
    symbol: str
    monthly_returns_pct: Dict[str, float]
    best_month: str
    worst_month: str
    seasonality_score: float
    seasonal_notes: str


class RollHistoryItem(BaseModel):
    roll_date: str
    near_price: Optional[float]
    far_price: Optional[float]
    price_adj_factor: Optional[float]
    detection_method: str


# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTER
# ──────────────────────────────────────────────────────────────────────────────

futures_v3_router = APIRouter(prefix="/futures/v3", tags=["futures-v3"])


@futures_v3_router.get("/universe", response_model=List[UniverseContractResponse])
def get_universe(
    asset_class: Optional[str] = Query(None, description="Filter by asset class"),
) -> List[UniverseContractResponse]:
    """Return the full futures universe with metadata."""
    contracts = FUTURES_UNIVERSE
    if asset_class:
        contracts = [c for c in contracts if c.asset_class == asset_class]
    return [
        UniverseContractResponse(
            symbol=c.symbol,
            yf_symbol=c.yf_symbol,
            name=c.name,
            exchange=c.exchange,
            asset_class=c.asset_class,
            cme_product_id=c.cme_product_id,
            spot_symbol=c.spot_symbol,
            tick_size=c.tick_size,
            contract_size=c.contract_size,
            unit=c.unit,
        )
        for c in contracts
    ]


@futures_v3_router.get("/term-structure/{symbol}", response_model=TermStructureResponse)
def get_term_structure(symbol: str) -> TermStructureResponse:
    """
    Return M1/M2/M3 prices, contango/backwardation status, roll yield,
    calendar spreads, basis, and volatility for a symbol.
    """
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not in futures universe.")

    contract = _UNIVERSE_MAP[sym]

    # Try cached today
    today = date.today().isoformat()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM term_structure_snapshots WHERE symbol=? AND snap_date=?",
            (sym, today)
        ).fetchone()

    if row:
        flag = row["contango_flag"]
        status = "contango" if flag == 1 else ("backwardation" if flag == 0 else "unknown")
        return TermStructureResponse(
            symbol=sym,
            name=contract.name,
            asset_class=contract.asset_class,
            snap_date=row["snap_date"],
            m1_price=row["m1_price"],
            m2_price=row["m2_price"],
            m3_price=row["m3_price"],
            m1_expiry=row["m1_expiry"] or "",
            m2_expiry=row["m2_expiry"] or "",
            m3_expiry=row["m3_expiry"] or "",
            contango_flag=bool(flag) if flag is not None else None,
            status=status,
            roll_yield_ann_pct=(
                round(row["roll_yield_ann"] * 100, 4) if row["roll_yield_ann"] is not None else None
            ),
            calendar_m1m2=row["calendar_m1m2"],
            calendar_m1m3=row["calendar_m1m3"],
            calendar_m2m3=row["calendar_m2m3"],
            basis=row["basis"],
            convenience_yield=row["convenience_yield"],
            hv_m1_21d_pct=(
                round(row["hv_m1_21d"] * 100, 2) if row["hv_m1_21d"] is not None else None
            ),
            hv_m2_21d_pct=(
                round(row["hv_m2_21d"] * 100, 2) if row["hv_m2_21d"] is not None else None
            ),
        )

    # Live compute
    ts = _ts_engine.get_term_structure(sym)
    if ts is None:
        raise HTTPException(status_code=502, detail="Could not fetch term structure data.")

    flag = ts.contango_flag
    status = "contango" if flag is True else ("backwardation" if flag is False else "unknown")
    return TermStructureResponse(
        symbol=sym,
        name=contract.name,
        asset_class=contract.asset_class,
        snap_date=ts.snap_date.isoformat(),
        m1_price=ts.m1_price,
        m2_price=ts.m2_price,
        m3_price=ts.m3_price,
        m1_expiry=ts.m1_expiry,
        m2_expiry=ts.m2_expiry,
        m3_expiry=ts.m3_expiry,
        contango_flag=ts.contango_flag,
        status=status,
        roll_yield_ann_pct=(
            round(ts.roll_yield_ann * 100, 4) if ts.roll_yield_ann is not None else None
        ),
        roll_yield_m2m3_ann_pct=(
            round(ts.roll_yield_m2m3 * 100, 4) if ts.roll_yield_m2m3 is not None else None
        ),
        calendar_m1m2=ts.calendar_m1m2,
        calendar_m1m3=ts.calendar_m1m3,
        calendar_m2m3=ts.calendar_m2m3,
        basis=ts.basis,
        convenience_yield=ts.convenience_yield,
        hv_m1_21d_pct=(
            round(ts.hv_m1_21d * 100, 2) if ts.hv_m1_21d is not None else None
        ),
        hv_m2_21d_pct=(
            round(ts.hv_m2_21d * 100, 2) if ts.hv_m2_21d is not None else None
        ),
    )


@futures_v3_router.get("/continuous/{symbol}")
def get_continuous(
    symbol: str,
    adjusted: bool = Query(True, description="True=Panama adjusted, False=raw front-month"),
    lookback_days: int = Query(504, ge=30, le=3650, description="Days of history to return"),
) -> Dict[str, Any]:
    """
    Return the continuous contract series (Panama backward-adjusted or raw).
    Builds and persists the series on first call.
    """
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    return _continuous_svc.get_series(sym, adjusted=adjusted, lookback_days=lookback_days)


@futures_v3_router.get("/roll-yield/{symbol}")
def get_roll_yield(symbol: str) -> Dict[str, Any]:
    """Return annualised roll yield for M1→M2 and interpretation."""
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    result = _roll_yield_svc.get_roll_yield(sym)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@futures_v3_router.get("/contango-backwardation", response_model=List[ContangoBackwardationResponse])
def get_contango_backwardation(
    asset_class: Optional[str] = Query(None, description="Filter by asset class"),
    status_filter: Optional[str] = Query(None, description="'contango','backwardation','flat'"),
    max_symbols: int = Query(20, ge=1, le=44),
) -> List[ContangoBackwardationResponse]:
    """
    Screen all futures contracts for contango/backwardation status.
    Returns today's cached snapshots where available; fetches live otherwise.
    """
    readings = _cb_screener.screen_all(max_symbols=max_symbols)
    if asset_class:
        readings = [r for r in readings if r.asset_class == asset_class]
    if status_filter:
        readings = [r for r in readings if r.status == status_filter]
    return [
        ContangoBackwardationResponse(
            symbol=r.symbol,
            name=r.name,
            asset_class=r.asset_class,
            status=r.status,
            m1_price=r.m1_price,
            m2_price=r.m2_price,
            roll_yield_ann_pct=(
                round(r.roll_yield_ann * 100, 4) if r.roll_yield_ann is not None else None
            ),
            snap_date=r.snap_date,
        )
        for r in readings
    ]


@futures_v3_router.get("/basis/{symbol}")
def get_basis(symbol: str) -> Dict[str, Any]:
    """Return basis (futures M1 - spot) and convenience yield estimate."""
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    result = _basis_svc.get_basis(sym)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@futures_v3_router.get("/calendar-spread/{symbol}")
def get_calendar_spread(symbol: str) -> Dict[str, Any]:
    """Return M1-M2, M1-M3, M2-M3 calendar spreads (absolute and percentage)."""
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    result = _cal_spread_svc.get_calendar_spread(sym)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@futures_v3_router.get("/seasonal/{symbol}", response_model=SeasonalResponse)
def get_seasonal(symbol: str) -> SeasonalResponse:
    """
    Return monthly return seasonality pattern for the symbol.
    Uses up to 5 years of continuous (adjusted) history.
    """
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    pattern = _seasonal_engine.compute_seasonal(sym)
    if pattern is None:
        raise HTTPException(status_code=502, detail="Insufficient history for seasonality.")
    return SeasonalResponse(
        symbol=sym,
        monthly_returns_pct=pattern.monthly_returns,
        best_month=pattern.best_month,
        worst_month=pattern.worst_month,
        seasonality_score=pattern.seasonality_score,
        seasonal_notes=pattern.seasonal_notes,
    )


@futures_v3_router.get("/roll-history/{symbol}", response_model=List[RollHistoryItem])
def get_roll_history(
    symbol: str,
    limit: int = Query(50, ge=1, le=500),
) -> List[RollHistoryItem]:
    """Return historical roll events (detected roll dates, prices, adjustment factors)."""
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT roll_date, near_price, far_price, price_adj_factor, detection_method "
            "FROM roll_history WHERE symbol=? ORDER BY roll_date DESC LIMIT ?",
            (sym, limit)
        ).fetchall()
    return [
        RollHistoryItem(
            roll_date=row["roll_date"],
            near_price=row["near_price"],
            far_price=row["far_price"],
            price_adj_factor=row["price_adj_factor"],
            detection_method=row["detection_method"] or "volume_crossover",
        )
        for row in rows
    ]


@futures_v3_router.post("/refresh/continuous")
def refresh_continuous(
    symbols: Optional[List[str]] = None,
    asset_class: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Trigger a rebuild of continuous series for specified symbols or asset class.
    If neither is specified, refreshes all continuous-eligible contracts.
    """
    if symbols:
        results = {}
        for sym in symbols:
            sym = sym.upper()
            if sym not in _UNIVERSE_MAP:
                results[sym] = "not_found"
                continue
            try:
                built = _continuous_builder.build_continuous(sym, force_refresh=True)
                results[sym] = "ok" if built is not None else "no_data"
            except Exception as exc:
                results[sym] = f"error: {exc}"
            time.sleep(_RATE_DELAY)
    else:
        ac_list = [asset_class] if asset_class else None
        results = _bulk_refresh.refresh_continuous_all(asset_classes=ac_list)
    return {"status": "done", "results": results}


@futures_v3_router.post("/refresh/term-structures")
def refresh_term_structures(
    max_symbols: int = Query(20, ge=1, le=44),
) -> Dict[str, Any]:
    """Refresh term structure snapshots for all CME-mapped contracts."""
    results = _bulk_refresh.refresh_term_structures(max_symbols=max_symbols)
    return {"status": "done", "results": results}


@futures_v3_router.get("/volatility/{symbol}")
def get_volatility(symbol: str) -> Dict[str, Any]:
    """
    Return historical volatility profile for the continuous contract:
    21d HV, 63d HV, and the latest term structure of vol (M1 vs M2).
    """
    sym = symbol.upper()
    if sym not in _UNIVERSE_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found.")

    contract = _UNIVERSE_MAP[sym]
    cutoff = (date.today() - timedelta(days=126)).isoformat()
    with _get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT trade_date, close_adj, hv_21d, hv_63d FROM continuous_series "
            "WHERE symbol=? AND trade_date>=? AND close_adj IS NOT NULL ORDER BY trade_date",
            conn, params=(sym, cutoff)
        )

    if df.empty:
        built = _continuous_builder.build_continuous(sym)
        if built is None:
            raise HTTPException(status_code=502, detail="No data available.")
        return get_volatility(symbol)

    latest = df.iloc[-1]
    hv21 = _safe_float(latest["hv_21d"])
    hv63 = _safe_float(latest["hv_63d"])

    # Term structure of vol from DB snapshot
    today = date.today().isoformat()
    with _get_connection() as conn:
        ts_row = conn.execute(
            "SELECT hv_m1_21d, hv_m2_21d FROM term_structure_snapshots "
            "WHERE symbol=? ORDER BY snap_date DESC LIMIT 1",
            (sym,)
        ).fetchone()

    hv_m1 = hv_m2 = None
    if ts_row:
        hv_m1 = ts_row["hv_m1_21d"]
        hv_m2 = ts_row["hv_m2_21d"]

    return {
        "symbol": sym,
        "name": contract.name,
        "continuous_hv": {
            "hv_21d_pct": round(hv21 * 100, 2) if hv21 else None,
            "hv_63d_pct": round(hv63 * 100, 2) if hv63 else None,
            "as_of": str(latest["trade_date"]),
        },
        "term_structure_of_vol": {
            "m1_hv_21d_pct": round(hv_m1 * 100, 2) if hv_m1 else None,
            "m2_hv_21d_pct": round(hv_m2 * 100, 2) if hv_m2 else None,
            "vol_term_premium": (
                round((hv_m2 - hv_m1) * 100, 2)
                if hv_m1 and hv_m2 else None
            ),
        },
        "recent_data_points": len(df),
    }


@futures_v3_router.get("/health")
def health() -> Dict[str, Any]:
    """Return health status of the futures_v3 module."""
    with _get_connection() as conn:
        uni_count = conn.execute("SELECT COUNT(*) FROM futures_universe").fetchone()[0]
        cont_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM continuous_series").fetchone()[0]
        ts_count = conn.execute(
            "SELECT COUNT(*) FROM term_structure_snapshots WHERE snap_date=?",
            (date.today().isoformat(),)
        ).fetchone()[0]
        roll_count = conn.execute("SELECT COUNT(*) FROM roll_history").fetchone()[0]
    return {
        "status": "ok",
        "db_path": str(_DB_PATH),
        "universe_contracts": uni_count,
        "symbols_with_continuous": cont_count,
        "todays_ts_snapshots": ts_count,
        "roll_events_stored": roll_count,
        "yfinance_available": _YF_OK,
    }
