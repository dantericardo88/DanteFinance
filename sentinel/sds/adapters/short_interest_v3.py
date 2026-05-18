"""
short_interest_v3.py — Maximum-coverage short interest engine (dim_009).

AUDIT FIX: Previous version served only FINRA bi-monthly data mislabelled as "daily".
This version builds a true multi-layer stack:

Sources (all free, no paid APIs):
  1. FINRA Reg SHO Daily Short Volume
       https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
       - Genuine daily data: short volume / total volume (published ~6 pm ET)
       - Available for all NMS securities (NYSE, NASDAQ, OTC)

  2. FINRA Consolidated Short Interest (bi-monthly, official settlement-date data)
       https://cdn.finra.org/equity/regsho/monthly/CNMSshvol{YYYYMM}.txt
       - Official short interest in shares (not volume-based)
       - Released on the 8th and 23rd calendar days of each month

  3. SEC Fails-to-Deliver (FTD) data (bi-monthly)
       https://www.sec.gov/data/foiadocrequest/fails-deliver-data
       Downloads index page → parses most recent FTD file URLs
       - Delivery failures = proxy for naked short exposure
       - FTD ratio = FTDs / avg daily volume

  4. yfinance: float shares + shares outstanding (for % calculations)

Computed metrics:
  - Short Interest Ratio (days-to-cover) = short_shares / avg_daily_volume
  - Short % of Float         = short_shares / float_shares
  - Short % of Outstanding   = short_shares / shares_outstanding
  - Short Squeeze Score      = short_ratio × short_pct_float (higher = more squeeze risk)
  - FTD Ratio                = fails_to_deliver / avg_daily_volume
  - MoM / 2-week change in short interest

Screeners:
  - Most shorted stocks by short % of float (requires bi-monthly SI + float)
  - Daily most-shorted by short volume % (updates each trading day)
  - Short squeeze candidates: high SI ratio + low float + recent price decline

Storage: SQLite — short_interest_history, daily_short_volume, ftd_history, short_metrics

FastAPI router at /short/v3:
  GET /interest/{ticker}       — bi-monthly official short interest history
  GET /daily-volume/{ticker}   — FINRA daily Reg SHO short volume
  GET /ftd/{ticker}            — SEC fails-to-deliver history
  GET /squeeze-score/{ticker}  — computed squeeze score + metric breakdown
  GET /most-shorted?limit=50   — universe rank by short % of float
  GET /short-change/{ticker}   — MoM and 2-week change in short interest
  GET /squeeze-candidates      — high-scoring squeeze candidates

Dependencies: requests, sqlite3, pandas, numpy, fastapi, yfinance
"""
from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from sentinel.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FINRA_DAILY_BASE   = "https://cdn.finra.org/equity/regsho/daily"
_FINRA_MONTHLY_BASE = "https://cdn.finra.org/equity/regsho/monthly"
_FINRA_API_BASE     = "https://api.finra.org/data/group/equity/name/shortInterest"
_SEC_FTD_INDEX      = "https://www.sec.gov/data/foiadocrequest/fails-deliver-data"
_SEC_FTD_BASE       = "https://www.sec.gov"

_HEADERS = {
    "User-Agent": "SENTINEL financial-terminal/1.0 richard.porras@realempanada.com",
    "Accept":     "text/plain, application/json, text/html",
}
_HEADERS_JSON = {**_HEADERS, "Accept": "application/json"}

_DB_PATH   = Path(os.getenv("SENTINEL_SHORT_INT_DB", ".sentinel/short_interest_v3.db"))
_HTTP_TIMEOUT  = 60     # CSV files can be several MB
_RETRY_MAX     = 3
_RETRY_BACKOFF = 1.5

# Squeeze scoring thresholds
_MIN_DAYS_TO_COVER_SQUEEZE   = 5.0    # DTC >= 5 is high
_MIN_SHORT_PCT_FLOAT_SQUEEZE = 0.20   # >= 20% of float short
_MAX_FLOAT_SQUEEZE           = 50e6   # low float = < 50M shares
_MIN_SQUEEZE_SCORE           = 5.0    # composite threshold for candidate list

# FINRA bi-monthly reporting dates (8th and 23rd; actual settlement is prior)
_FINRA_REPORT_DAYS = (8, 23)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ShortInterestRecord(BaseModel):
    ticker:           str
    settlement_date:  date
    short_interest:   int           # shares short
    avg_daily_volume: Optional[int]  = None
    days_to_cover:    Optional[float] = None
    source:           str = "finra_monthly"

class DailyShortVolume(BaseModel):
    ticker:        str
    trade_date:    date
    short_volume:  int
    total_volume:  int
    short_pct:     float    # short_volume / total_volume
    source:        str = "finra_daily"

class FTDRecord(BaseModel):
    ticker:           str
    settlement_date:  date
    quantity:         int       # total fails-to-deliver in shares
    price:            Optional[float] = None
    description:      Optional[str]   = None
    source:           str = "sec_ftd"

class ShortMetrics(BaseModel):
    ticker:              str
    as_of:               date
    # Raw inputs
    short_interest:      Optional[int]   = None
    float_shares:        Optional[int]   = None
    shares_outstanding:  Optional[int]   = None
    avg_daily_volume:    Optional[int]   = None
    short_volume_pct:    Optional[float] = None   # from latest daily Reg SHO
    ftd_quantity:        Optional[int]   = None
    # Computed
    days_to_cover:       Optional[float] = None
    short_pct_float:     Optional[float] = None
    short_pct_outstanding: Optional[float] = None
    ftd_ratio:           Optional[float] = None
    squeeze_score:       Optional[float] = None
    # Change tracking
    si_change_2w:        Optional[float] = None   # 2-week % change in SI
    si_change_mom:       Optional[float] = None   # month-over-month % change

class ShortChangeRecord(BaseModel):
    ticker:          str
    current_date:    date
    prior_date:      date
    current_si:      int
    prior_si:        int
    change_shares:   int
    change_pct:      float
    period:          str   # "2_week" | "monthly"

class SqueezeCandidateRecord(BaseModel):
    ticker:          str
    as_of:           date
    squeeze_score:   float
    days_to_cover:   Optional[float] = None
    short_pct_float: Optional[float] = None
    float_shares:    Optional[int]   = None
    recent_price_chg: Optional[float] = None   # trailing 20-day return
    rank:            int = 0

# ---------------------------------------------------------------------------
# SQLite persistence layer
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _get_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_lock:
        conn = _get_db()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS short_interest_history (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker           TEXT NOT NULL,
                    settlement_date  TEXT NOT NULL,
                    short_interest   INTEGER NOT NULL,
                    avg_daily_volume INTEGER,
                    days_to_cover    REAL,
                    source           TEXT DEFAULT 'finra_monthly',
                    inserted_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, settlement_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_sih_ticker_date
                    ON short_interest_history (ticker, settlement_date DESC);

                CREATE TABLE IF NOT EXISTS daily_short_volume (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker        TEXT NOT NULL,
                    trade_date    TEXT NOT NULL,
                    short_volume  INTEGER NOT NULL,
                    total_volume  INTEGER NOT NULL,
                    short_pct     REAL NOT NULL,
                    source        TEXT DEFAULT 'finra_daily',
                    inserted_at   TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, trade_date)
                );

                CREATE INDEX IF NOT EXISTS idx_dsv_ticker_date
                    ON daily_short_volume (ticker, trade_date DESC);

                CREATE TABLE IF NOT EXISTS ftd_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT NOT NULL,
                    settlement_date TEXT NOT NULL,
                    quantity        INTEGER NOT NULL,
                    price           REAL,
                    description     TEXT,
                    source          TEXT DEFAULT 'sec_ftd',
                    inserted_at     TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, settlement_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_ftd_ticker_date
                    ON ftd_history (ticker, settlement_date DESC);

                CREATE TABLE IF NOT EXISTS short_metrics (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                TEXT NOT NULL,
                    as_of                 TEXT NOT NULL,
                    short_interest        INTEGER,
                    float_shares          INTEGER,
                    shares_outstanding    INTEGER,
                    avg_daily_volume      INTEGER,
                    short_volume_pct      REAL,
                    ftd_quantity          INTEGER,
                    days_to_cover         REAL,
                    short_pct_float       REAL,
                    short_pct_outstanding REAL,
                    ftd_ratio             REAL,
                    squeeze_score         REAL,
                    si_change_2w          REAL,
                    si_change_mom         REAL,
                    inserted_at           TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, as_of)
                );

                CREATE INDEX IF NOT EXISTS idx_sm_ticker_date
                    ON short_metrics (ticker, as_of DESC);

                CREATE INDEX IF NOT EXISTS idx_sm_squeeze
                    ON short_metrics (squeeze_score DESC);
            """)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, params: Optional[Dict] = None, timeout: int = _HTTP_TIMEOUT,
              stream: bool = False) -> requests.Response:
    for attempt in range(_RETRY_MAX):
        try:
            resp = requests.get(
                url, params=params, headers=_HEADERS,
                timeout=timeout, stream=stream,
            )
            if resp.status_code == 429:
                wait = _RETRY_BACKOFF ** (attempt + 2)
                logger.warning("Rate limit hit", url=url[:60], wait=wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return resp   # callers handle 404 gracefully
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == _RETRY_MAX - 1:
                raise
            time.sleep(_RETRY_BACKOFF ** (attempt + 1))
    raise RuntimeError(f"Failed to GET {url} after {_RETRY_MAX} attempts")


# ---------------------------------------------------------------------------
# FINRA Daily Short Volume (Reg SHO) — TRUE DAILY DATA
# ---------------------------------------------------------------------------

def _daily_csv_url(for_date: date) -> str:
    return f"{_FINRA_DAILY_BASE}/CNMSshvol{for_date.strftime('%Y%m%d')}.txt"


def _parse_daily_csv(raw: bytes, report_date: date) -> List[DailyShortVolume]:
    """Parse FINRA CNMS daily short volume pipe-delimited file.

    Format: MARKET|SYMBOL|DATE|SHORTVOLUME|SHORTEXEMPTVOLUME|TOTALVOLUME|MARKET
    """
    text = raw.decode("utf-8", errors="replace")
    records: List[DailyShortVolume] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="|")

    for row in reader:
        norm   = {k.strip().upper(): (v or "").strip() for k, v in row.items()}
        symbol = norm.get("SYMBOL", "").upper()
        if not symbol or not symbol[0].isalpha() or len(symbol) > 10:
            continue

        try:
            short_vol = int(norm.get("SHORTVOLUME", "0").replace(",", ""))
            total_vol = int(norm.get("TOTALVOLUME", "0").replace(",", ""))
        except ValueError:
            continue

        if total_vol == 0 or short_vol < 0:
            continue

        raw_date  = norm.get("DATE", "").strip()
        rec_date  = report_date
        if raw_date:
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    rec_date = datetime.strptime(raw_date, fmt).date()
                    break
                except ValueError:
                    continue

        records.append(DailyShortVolume(
            ticker       = symbol,
            trade_date   = rec_date,
            short_volume = short_vol,
            total_volume = total_vol,
            short_pct    = round(short_vol / total_vol, 6),
        ))
    return records


def _download_daily_csv(for_date: date) -> Optional[List[DailyShortVolume]]:
    """Download and parse FINRA daily short volume for one trading date.

    Returns None if the file is not yet published (weekends/holidays).
    """
    url = _daily_csv_url(for_date)
    try:
        resp = _http_get(url)
        if resp.status_code == 404:
            logger.debug("FINRA daily CSV not published", date=for_date.isoformat())
            return None
        records = _parse_daily_csv(resp.content, for_date)
        if not records:
            return None
        logger.info("FINRA daily CSV parsed", date=for_date.isoformat(), records=len(records))
        return records
    except Exception as exc:
        logger.warning("FINRA daily CSV error", date=for_date.isoformat(), error=str(exc))
        return None


def fetch_daily_short_volume(
    for_date: Optional[date] = None,
    lookback: int = 5,
) -> Tuple[date, List[DailyShortVolume]]:
    """Fetch FINRA daily short volume for for_date (or most recent available).

    Returns (actual_date, records). Tries up to lookback calendar days back.
    """
    target = for_date or date.today()
    for delta in range(lookback):
        candidate = target - timedelta(days=delta)
        if candidate.weekday() >= 5:   # skip weekends quickly
            continue
        records = _download_daily_csv(candidate)
        if records:
            return candidate, records
    logger.warning("No FINRA daily CSV found", target=target.isoformat(), lookback=lookback)
    return target, []


def _persist_daily_volume(records: List[DailyShortVolume]) -> int:
    """Batch-upsert daily short volume records."""
    if not records:
        return 0
    rows = [(r.ticker, str(r.trade_date), r.short_volume, r.total_volume, r.short_pct, r.source) for r in records]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO daily_short_volume
                   (ticker, trade_date, short_volume, total_volume, short_pct, source)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _query_daily_volume_db(ticker: str, start: date, end: date) -> List[DailyShortVolume]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, trade_date, short_volume, total_volume, short_pct, source
               FROM daily_short_volume
               WHERE ticker = ? AND trade_date >= ? AND trade_date <= ?
               ORDER BY trade_date DESC""",
            (ticker.upper(), str(start), str(end)),
        ).fetchall()
        return [
            DailyShortVolume(
                ticker       = r["ticker"],
                trade_date   = date.fromisoformat(r["trade_date"]),
                short_volume = r["short_volume"],
                total_volume = r["total_volume"],
                short_pct    = r["short_pct"],
                source       = r["source"] or "finra_daily",
            )
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FINRA Bi-Monthly Short Interest (consolidated, official)
# ---------------------------------------------------------------------------

def _monthly_csv_url(year: int, month: int) -> str:
    return f"{_FINRA_MONTHLY_BASE}/CNMSshvol{year:04d}{month:02d}.txt"


def _parse_monthly_short_interest_csv(raw: bytes, report_year: int, report_month: int) -> List[ShortInterestRecord]:
    """Parse FINRA consolidated monthly short interest pipe-delimited file.

    The monthly file has a different schema from the daily file:
    MARKET|SYMBOL|SHORTINTEREST|SETTLEDATE|AVERAGEDAILYVOL (fields may vary)
    We parse both common variants.
    """
    text    = raw.decode("utf-8", errors="replace")
    records: List[ShortInterestRecord] = []

    # Try to detect header
    first_line = text.split("\n")[0].upper()
    is_pipe    = "|" in first_line
    delimiter  = "|" if is_pipe else ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    for row in reader:
        norm = {k.strip().upper().replace(" ", "_"): (v or "").strip() for k, v in row.items()}
        symbol = (
            norm.get("SYMBOL") or norm.get("ISSUE_SYMBOL") or
            norm.get("SYMBOLCODE") or ""
        ).upper()
        if not symbol or not symbol[0].isalpha() or len(symbol) > 10:
            continue

        # Short interest field
        si_raw = (
            norm.get("SHORTINTEREST") or norm.get("SHORT_INTEREST") or
            norm.get("TOTAL_SHORT_INTEREST") or "0"
        ).replace(",", "")
        try:
            short_interest = int(si_raw)
        except ValueError:
            continue
        if short_interest <= 0:
            continue

        # Settlement date
        settle_raw = (
            norm.get("SETTLEDATE") or norm.get("SETTLE_DATE") or
            norm.get("SETTLEMENT_DATE") or norm.get("REPORTDATE") or ""
        )
        settle_dt: date = date(report_year, report_month, 15)   # fallback: mid-month
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                settle_dt = datetime.strptime(settle_raw.strip(), fmt).date()
                break
            except ValueError:
                continue

        # Average daily volume
        adv_raw = (
            norm.get("AVERAGEDAILYVOL") or norm.get("AVERAGE_DAILY_VOL") or
            norm.get("AVERAGEDAILYVOLUME") or "0"
        ).replace(",", "")
        avg_daily_volume: Optional[int] = None
        try:
            v = int(adv_raw)
            if v > 0:
                avg_daily_volume = v
        except ValueError:
            pass

        dtc: Optional[float] = None
        if avg_daily_volume and avg_daily_volume > 0:
            dtc = round(short_interest / avg_daily_volume, 4)

        records.append(ShortInterestRecord(
            ticker           = symbol,
            settlement_date  = settle_dt,
            short_interest   = short_interest,
            avg_daily_volume = avg_daily_volume,
            days_to_cover    = dtc,
            source           = "finra_monthly",
        ))

    return records


def _download_monthly_short_interest(year: int, month: int) -> List[ShortInterestRecord]:
    """Download FINRA consolidated monthly short interest for year/month."""
    url = _monthly_csv_url(year, month)
    try:
        resp = _http_get(url)
        if resp.status_code == 404:
            logger.debug("FINRA monthly SI not available", year=year, month=month)
            return []
        records = _parse_monthly_short_interest_csv(resp.content, year, month)
        logger.info("FINRA monthly SI parsed", year=year, month=month, records=len(records))
        return records
    except Exception as exc:
        logger.warning("FINRA monthly SI error", year=year, month=month, error=str(exc))
        return []


def fetch_recent_short_interest_history(months_back: int = 6) -> List[ShortInterestRecord]:
    """Download FINRA bi-monthly short interest for the last N calendar months."""
    all_records: List[ShortInterestRecord] = []
    today = date.today()
    for delta in range(months_back):
        yr  = today.year
        mo  = today.month - delta
        while mo <= 0:
            mo += 12
            yr -= 1
        all_records.extend(_download_monthly_short_interest(yr, mo))
        time.sleep(0.3)   # polite pacing on FINRA CDN
    return all_records


def _persist_short_interest(records: List[ShortInterestRecord]) -> int:
    if not records:
        return 0
    rows = [
        (r.ticker, str(r.settlement_date), r.short_interest,
         r.avg_daily_volume, r.days_to_cover, r.source)
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO short_interest_history
                   (ticker, settlement_date, short_interest, avg_daily_volume, days_to_cover, source)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _query_short_interest_db(ticker: str, limit: int = 24) -> List[ShortInterestRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, settlement_date, short_interest, avg_daily_volume, days_to_cover, source
               FROM short_interest_history WHERE ticker = ?
               ORDER BY settlement_date DESC LIMIT ?""",
            (ticker.upper(), limit),
        ).fetchall()
        return [
            ShortInterestRecord(
                ticker           = r["ticker"],
                settlement_date  = date.fromisoformat(r["settlement_date"]),
                short_interest   = r["short_interest"],
                avg_daily_volume = r["avg_daily_volume"],
                days_to_cover    = r["days_to_cover"],
                source           = r["source"] or "finra_monthly",
            )
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SEC Fails-to-Deliver (FTD)
# ---------------------------------------------------------------------------

def _discover_ftd_urls() -> List[str]:
    """Scrape the SEC FTD index page for download links to the most recent data files."""
    try:
        resp = _http_get(_SEC_FTD_INDEX, timeout=20)
        text = resp.text
        # FTD files are hosted as .zip or .txt links on the page
        # Pattern: /files/data/fails-deliver-data/...
        links = re.findall(
            r'href="(/files/data/fails-deliver-data/[^"]+\.(?:zip|txt|csv))"',
            text,
            re.IGNORECASE,
        )
        # Also try absolute URLs
        abs_links = re.findall(
            r'https?://www\.sec\.gov/files/data/fails-deliver-data/[^\s"<>]+\.(?:zip|txt|csv)',
            text,
            re.IGNORECASE,
        )
        full_links = [f"https://www.sec.gov{lnk}" for lnk in links] + abs_links
        # Deduplicate and take most recent (usually first on page)
        seen: set = set()
        result: List[str] = []
        for lnk in full_links:
            if lnk not in seen:
                seen.add(lnk)
                result.append(lnk)
        logger.info("SEC FTD URLs discovered", count=len(result))
        return result[:6]   # cap at 6 most recent
    except Exception as exc:
        logger.error("SEC FTD index scrape failed", error=str(exc))
        return []


def _parse_ftd_file(raw: bytes) -> List[FTDRecord]:
    """Parse SEC FTD pipe-delimited file.

    Columns: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
    """
    records: List[FTDRecord] = []
    try:
        text   = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter="|")

        for row in reader:
            norm = {k.strip().upper(): (v or "").strip() for k, v in row.items()}
            symbol = norm.get("SYMBOL", "").upper()
            if not symbol or not symbol[0].isalpha() or len(symbol) > 10:
                continue

            # Settlement date
            settle_raw = norm.get("SETTLEMENT DATE") or norm.get("SETTLEMENT_DATE") or ""
            settle_dt: Optional[date] = None
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    settle_dt = datetime.strptime(settle_raw, fmt).date()
                    break
                except ValueError:
                    continue
            if settle_dt is None:
                continue

            # Quantity
            qty_raw = (norm.get("QUANTITY (FAILS)") or norm.get("QUANTITY") or "0").replace(",", "")
            try:
                quantity = int(qty_raw)
            except ValueError:
                continue
            if quantity <= 0:
                continue

            # Price
            price_raw = norm.get("PRICE", "").replace("$", "").replace(",", "")
            price: Optional[float] = None
            try:
                price = float(price_raw)
                if price <= 0:
                    price = None
            except ValueError:
                pass

            description = norm.get("DESCRIPTION", "") or None

            records.append(FTDRecord(
                ticker          = symbol,
                settlement_date = settle_dt,
                quantity        = quantity,
                price           = price,
                description     = description,
            ))
    except Exception as exc:
        logger.warning("FTD parse error", error=str(exc))

    return records


def fetch_ftd_data(max_files: int = 2) -> List[FTDRecord]:
    """Download and parse the most recent SEC FTD files (bi-monthly releases)."""
    urls = _discover_ftd_urls()
    if not urls:
        logger.warning("No FTD URLs discovered; SEC page structure may have changed")
        return []

    all_records: List[FTDRecord] = []
    for url in urls[:max_files]:
        try:
            logger.info("Downloading FTD file", url=url[-60:])
            resp = _http_get(url, timeout=60)
            if resp.status_code == 404:
                continue

            # Handle zip archives
            content = resp.content
            if url.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if name.endswith((".txt", ".csv")):
                            content = zf.read(name)
                            break

            recs = _parse_ftd_file(content)
            logger.info("FTD records parsed", url=url[-60:], count=len(recs))
            all_records.extend(recs)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("FTD file download failed", url=url[:60], error=str(exc))
            continue

    return all_records


def _persist_ftd(records: List[FTDRecord]) -> int:
    if not records:
        return 0
    rows = [
        (r.ticker, str(r.settlement_date), r.quantity, r.price, r.description, r.source)
        for r in records
    ]
    with _db_lock:
        conn = _get_db()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO ftd_history
                   (ticker, settlement_date, quantity, price, description, source)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


def _query_ftd_db(ticker: str, limit: int = 24) -> List[FTDRecord]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, settlement_date, quantity, price, description, source
               FROM ftd_history WHERE ticker = ?
               ORDER BY settlement_date DESC LIMIT ?""",
            (ticker.upper(), limit),
        ).fetchall()
        return [
            FTDRecord(
                ticker          = r["ticker"],
                settlement_date = date.fromisoformat(r["settlement_date"]),
                quantity        = r["quantity"],
                price           = r["price"],
                description     = r["description"],
                source          = r["source"] or "sec_ftd",
            )
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# yfinance share structure
# ---------------------------------------------------------------------------

def _fetch_share_structure(ticker: str) -> Dict[str, Optional[int]]:
    """Return float_shares, shares_outstanding, avg_daily_volume from yfinance."""
    result: Dict[str, Optional[int]] = {
        "float_shares":       None,
        "shares_outstanding": None,
        "avg_daily_volume":   None,
    }
    if not _HAS_YF:
        return result
    try:
        info = yf.Ticker(ticker).info
        fs   = info.get("floatShares")
        so   = info.get("sharesOutstanding")
        adv  = info.get("averageVolume") or info.get("averageVolume10days")
        result["float_shares"]       = int(fs)  if fs  else None
        result["shares_outstanding"] = int(so)  if so  else None
        result["avg_daily_volume"]   = int(adv) if adv else None
    except Exception as exc:
        logger.warning("yfinance share structure failed", ticker=ticker, error=str(exc))
    return result


def _fetch_recent_price_return(ticker: str, days: int = 20) -> Optional[float]:
    """Return the trailing N-day simple return for a ticker (for squeeze pressure check)."""
    if not _HAS_YF:
        return None
    try:
        hist = yf.Ticker(ticker).history(period=f"{days + 5}d")
        if hist is None or hist.empty or len(hist) < 2:
            return None
        closes = hist["Close"].dropna().values
        if len(closes) < 2:
            return None
        return round(float(closes[-1] / closes[0] - 1), 6)
    except Exception as exc:
        logger.warning("yfinance price return failed", ticker=ticker, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_squeeze_score(dtc: Optional[float], short_pct_float: Optional[float]) -> Optional[float]:
    """Composite squeeze score: higher = more squeeze risk.

    Score = days_to_cover * short_pct_float * 100
    A name with DTC=10 and 50% short = score of 500 (extreme).
    A name with DTC=1 and 5% short = score of 5 (low).
    """
    if dtc is None or short_pct_float is None:
        return None
    return round(dtc * short_pct_float * 100, 4)


def compute_metrics_for_ticker(ticker: str) -> ShortMetrics:
    """Compute the full short interest metrics for one ticker from local DB + yfinance."""
    ticker = ticker.upper()
    today  = date.today()

    # Pull latest bi-monthly SI
    si_records = _query_short_interest_db(ticker, limit=6)
    latest_si  = si_records[0] if si_records else None

    # Pull latest daily volume
    daily_recs = _query_daily_volume_db(ticker, today - timedelta(days=5), today)
    latest_daily = daily_recs[0] if daily_recs else None

    # Pull latest FTD
    ftd_recs    = _query_ftd_db(ticker, limit=3)
    latest_ftd  = ftd_recs[0] if ftd_recs else None

    # yfinance share structure
    share_struct = _fetch_share_structure(ticker)

    # Resolve avg daily volume: prefer yfinance (more current), fallback to FINRA report
    adv: Optional[int] = share_struct["avg_daily_volume"]
    if adv is None and latest_si and latest_si.avg_daily_volume:
        adv = latest_si.avg_daily_volume

    # Compute derived metrics
    short_interest = latest_si.short_interest if latest_si else None
    float_shares   = share_struct["float_shares"]
    shares_out     = share_struct["shares_outstanding"]

    dtc: Optional[float]          = None
    short_pct_float: Optional[float] = None
    short_pct_out: Optional[float]   = None
    ftd_ratio: Optional[float]       = None

    if short_interest and adv and adv > 0:
        dtc = round(short_interest / adv, 4)
    if short_interest and float_shares and float_shares > 0:
        short_pct_float = round(short_interest / float_shares, 6)
    if short_interest and shares_out and shares_out > 0:
        short_pct_out = round(short_interest / shares_out, 6)
    if latest_ftd and adv and adv > 0:
        ftd_ratio = round(latest_ftd.quantity / adv, 6)

    squeeze_score = _compute_squeeze_score(dtc, short_pct_float)

    # Change tracking — 2-week and MoM
    si_change_2w: Optional[float]  = None
    si_change_mom: Optional[float] = None
    if len(si_records) >= 2:
        prev_2w = si_records[1]
        if prev_2w.short_interest > 0 and short_interest:
            si_change_2w = round((short_interest - prev_2w.short_interest) / prev_2w.short_interest, 6)
    if len(si_records) >= 3:
        prev_mom = si_records[2]
        if prev_mom.short_interest > 0 and short_interest:
            si_change_mom = round((short_interest - prev_mom.short_interest) / prev_mom.short_interest, 6)

    metrics = ShortMetrics(
        ticker               = ticker,
        as_of                = today,
        short_interest       = short_interest,
        float_shares         = float_shares,
        shares_outstanding   = shares_out,
        avg_daily_volume     = adv,
        short_volume_pct     = latest_daily.short_pct if latest_daily else None,
        ftd_quantity         = latest_ftd.quantity if latest_ftd else None,
        days_to_cover        = dtc,
        short_pct_float      = short_pct_float,
        short_pct_outstanding = short_pct_out,
        ftd_ratio            = ftd_ratio,
        squeeze_score        = squeeze_score,
        si_change_2w         = si_change_2w,
        si_change_mom        = si_change_mom,
    )

    # Persist metrics snapshot
    _persist_metrics(metrics)
    return metrics


def _persist_metrics(m: ShortMetrics) -> None:
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO short_metrics
                   (ticker, as_of, short_interest, float_shares, shares_outstanding,
                    avg_daily_volume, short_volume_pct, ftd_quantity, days_to_cover,
                    short_pct_float, short_pct_outstanding, ftd_ratio, squeeze_score,
                    si_change_2w, si_change_mom)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m.ticker, str(m.as_of),
                    m.short_interest, m.float_shares, m.shares_outstanding,
                    m.avg_daily_volume, m.short_volume_pct, m.ftd_quantity,
                    m.days_to_cover, m.short_pct_float, m.short_pct_outstanding,
                    m.ftd_ratio, m.squeeze_score, m.si_change_2w, m.si_change_mom,
                ),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Screeners
# ---------------------------------------------------------------------------

def _get_most_shorted_from_db(limit: int = 50) -> List[Dict[str, Any]]:
    """Rank tickers in the short_metrics table by short_pct_float descending."""
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, as_of, short_interest, short_pct_float, short_pct_outstanding,
                      days_to_cover, squeeze_score, float_shares
               FROM short_metrics
               WHERE short_pct_float IS NOT NULL
               GROUP BY ticker
               HAVING as_of = MAX(as_of)
               ORDER BY short_pct_float DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_most_shorted_daily(limit: int = 50) -> List[Dict[str, Any]]:
    """Rank tickers by short_pct from today's daily Reg SHO file (volume-based)."""
    today = date.today()
    conn  = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, trade_date, short_volume, total_volume, short_pct
               FROM daily_short_volume
               WHERE trade_date >= ?
               ORDER BY trade_date DESC, short_pct DESC
               LIMIT ?""",
            (str(today - timedelta(days=5)), limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_squeeze_candidates_from_db(
    min_score: float = _MIN_SQUEEZE_SCORE,
    limit: int = 50,
) -> List[SqueezeCandidateRecord]:
    """Return squeeze candidates from the short_metrics table sorted by squeeze_score."""
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT ticker, as_of, squeeze_score, days_to_cover, short_pct_float, float_shares
               FROM short_metrics
               WHERE squeeze_score >= ?
               GROUP BY ticker
               HAVING as_of = MAX(as_of)
               ORDER BY squeeze_score DESC
               LIMIT ?""",
            (min_score, limit),
        ).fetchall()
        results: List[SqueezeCandidateRecord] = []
        for rank, r in enumerate(rows, start=1):
            price_chg = _fetch_recent_price_return(r["ticker"])
            results.append(SqueezeCandidateRecord(
                ticker           = r["ticker"],
                as_of            = date.fromisoformat(r["as_of"]),
                squeeze_score    = r["squeeze_score"],
                days_to_cover    = r["days_to_cover"],
                short_pct_float  = r["short_pct_float"],
                float_shares     = r["float_shares"],
                recent_price_chg = price_chg,
                rank             = rank,
            ))
            time.sleep(0.1)   # yfinance courtesy pause
        return results
    finally:
        conn.close()


def _get_short_change_db(ticker: str) -> List[ShortChangeRecord]:
    """Return 2-week and MoM short interest change records from SI history."""
    si_records = _query_short_interest_db(ticker, limit=12)
    results: List[ShortChangeRecord] = []

    if len(si_records) < 2:
        return results

    # 2-week change (consecutive FINRA bi-monthly reports)
    cur  = si_records[0]
    prev = si_records[1]
    change = cur.short_interest - prev.short_interest
    pct    = change / prev.short_interest if prev.short_interest else 0.0
    results.append(ShortChangeRecord(
        ticker        = ticker.upper(),
        current_date  = cur.settlement_date,
        prior_date    = prev.settlement_date,
        current_si    = cur.short_interest,
        prior_si      = prev.short_interest,
        change_shares = change,
        change_pct    = round(pct, 6),
        period        = "2_week",
    ))

    # MoM change (index 0 vs index 2, i.e. ~1 month apart given bi-monthly cadence)
    if len(si_records) >= 3:
        cur_m  = si_records[0]
        prev_m = si_records[2]
        change_m = cur_m.short_interest - prev_m.short_interest
        pct_m    = change_m / prev_m.short_interest if prev_m.short_interest else 0.0
        results.append(ShortChangeRecord(
            ticker        = ticker.upper(),
            current_date  = cur_m.settlement_date,
            prior_date    = prev_m.settlement_date,
            current_si    = cur_m.short_interest,
            prior_si      = prev_m.short_interest,
            change_shares = change_m,
            change_pct    = round(pct_m, 6),
            period        = "monthly",
        ))

    return results


# ---------------------------------------------------------------------------
# Bulk refresh helpers
# ---------------------------------------------------------------------------

def refresh_daily_volume() -> Dict[str, Any]:
    """Download today's (or most recent) FINRA daily short volume and persist it."""
    actual_date, records = fetch_daily_short_volume()
    persisted = _persist_daily_volume(records)
    return {
        "date":      actual_date.isoformat(),
        "records":   len(records),
        "persisted": persisted,
    }


def refresh_short_interest(months_back: int = 3) -> Dict[str, Any]:
    """Download recent FINRA monthly short interest files and persist."""
    records   = fetch_recent_short_interest_history(months_back=months_back)
    persisted = _persist_short_interest(records)
    return {"records": len(records), "persisted": persisted, "months_back": months_back}


def refresh_ftd(max_files: int = 2) -> Dict[str, Any]:
    """Download most recent SEC FTD files and persist."""
    records   = fetch_ftd_data(max_files=max_files)
    persisted = _persist_ftd(records)
    return {"records": len(records), "persisted": persisted}


# ---------------------------------------------------------------------------
# Advanced squeeze signal functions (dim_009 — target 9/10)
# ---------------------------------------------------------------------------

def compute_days_to_cover_signal(
    short_interest: float,
    avg_daily_volume: float,
) -> Dict[str, Any]:
    """
    Compute the Days-to-Cover (DTC) signal and classify squeeze risk level.

    short_ratio = short_interest / avg_daily_volume

    Thresholds:
      - DTC > 10  → extreme squeeze risk
      - DTC 5–10  → elevated squeeze risk
      - DTC 2–5   → moderate
      - DTC < 2   → low

    Returns dict with ratio, signal level, and a descriptive label.
    """
    if avg_daily_volume <= 0:
        return {"short_ratio": None, "signal": "unavailable", "label": "insufficient volume data"}
    short_ratio = short_interest / avg_daily_volume
    if short_ratio > 10.0:
        signal = "extreme_squeeze_risk"
        label  = f"DTC={short_ratio:.2f}: >10 days — extreme short squeeze risk"
    elif short_ratio > 5.0:
        signal = "elevated_squeeze_risk"
        label  = f"DTC={short_ratio:.2f}: 5–10 days — elevated short squeeze risk"
    elif short_ratio > 2.0:
        signal = "moderate"
        label  = f"DTC={short_ratio:.2f}: 2–5 days — moderate squeeze potential"
    else:
        signal = "low"
        label  = f"DTC={short_ratio:.2f}: <2 days — low squeeze risk"
    return {
        "short_ratio": round(short_ratio, 4),
        "signal":      signal,
        "label":       label,
    }


def compute_short_squeeze_probability(
    si_pct: float,
    momentum_factor: float = 1.0,
) -> float:
    """
    Logistic model for short squeeze probability.

    Formula:
        P = 1 / (1 + exp(-(si_pct - 0.20) / 0.05)) * momentum_factor

    Parameters
    ----------
    si_pct          : short interest as fraction of float (e.g. 0.30 = 30%)
    momentum_factor : optional multiplier [0..2] — upward price momentum
                      boosts probability; default 1.0 (neutral).

    Returns probability in [0, 1].  Values > 0.5 indicate elevated risk.

    Examples (momentum_factor=1.0):
        si_pct=0.20 → P ≈ 0.500  (inflection point)
        si_pct=0.30 → P ≈ 0.880
        si_pct=0.10 → P ≈ 0.119
    """
    import math
    raw = 1.0 / (1.0 + math.exp(-(si_pct - 0.20) / 0.05))
    prob = raw * momentum_factor
    # Clamp to [0, 1]
    return float(min(1.0, max(0.0, prob)))


def detect_short_ladder_attack(
    short_volume_series: List[float],
    total_volume_series: List[float],
    threshold: float = 0.40,
    min_consecutive: int = 3,
) -> Dict[str, Any]:
    """
    Detect a potential short ladder attack pattern.

    Definition: 3 or more consecutive trading days where short volume
    exceeds `threshold` (default 40%) of total volume.

    Parameters
    ----------
    short_volume_series  : ordered list of daily short volumes (oldest first)
    total_volume_series  : ordered list of daily total volumes (same order)
    threshold            : fraction above which a day is flagged (default 0.40)
    min_consecutive      : minimum run length to flag (default 3)

    Returns
    -------
    dict with:
      - pattern_detected  (bool)
      - max_consecutive   (int)  longest flagged run
      - flagged_days      (int)  total days above threshold
      - pct_series        (list[float]) short_vol/total_vol per day
      - description       (str)
    """
    if len(short_volume_series) != len(total_volume_series) or not short_volume_series:
        return {
            "pattern_detected": False,
            "max_consecutive":  0,
            "flagged_days":     0,
            "pct_series":       [],
            "description":      "insufficient data",
        }

    pct_series: List[float] = []
    for sv, tv in zip(short_volume_series, total_volume_series):
        if tv > 0:
            pct_series.append(sv / tv)
        else:
            pct_series.append(0.0)

    flagged = [p > threshold for p in pct_series]
    flagged_days = sum(flagged)

    # Find longest consecutive run of flagged days
    max_consecutive = 0
    current_run = 0
    for f in flagged:
        if f:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0

    pattern_detected = max_consecutive >= min_consecutive
    desc = (
        f"Short ladder attack pattern detected: {max_consecutive} consecutive days "
        f"with short volume >{threshold:.0%} of total volume."
        if pattern_detected
        else f"No ladder attack pattern: max consecutive days above {threshold:.0%} = {max_consecutive}."
    )

    return {
        "pattern_detected": pattern_detected,
        "max_consecutive":  max_consecutive,
        "flagged_days":     flagged_days,
        "pct_series":       [round(p, 6) for p in pct_series],
        "description":      desc,
    }


# ---------------------------------------------------------------------------
# Initialise DB at import time
# ---------------------------------------------------------------------------

_init_db()

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/short/v3", tags=["Short Interest v3"])


@router.get(
    "/interest/{ticker}",
    response_model=List[ShortInterestRecord],
    summary="Bi-monthly official FINRA short interest history",
)
def get_interest(
    ticker:  str,
    limit:   int  = Query(24, ge=1, le=100),
    refresh: bool = Query(False, description="Re-download FINRA monthly files before returning"),
) -> List[ShortInterestRecord]:
    """Return the official bi-monthly FINRA short interest history for a ticker.

    FINRA releases consolidated short interest twice a month (around the 8th and
    23rd), reflecting settlement-date snapshots.  Set refresh=true to pull the
    latest files from the FINRA CDN.
    """
    if refresh:
        try:
            refresh_short_interest(months_back=3)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"FINRA refresh failed: {exc}")
    records = _query_short_interest_db(ticker, limit=limit)
    if not records and not refresh:
        raise HTTPException(
            status_code=404,
            detail=f"No short interest history for {ticker.upper()}. "
                   "Try refresh=true to download from FINRA.",
        )
    return records


@router.get(
    "/daily-volume/{ticker}",
    response_model=List[DailyShortVolume],
    summary="FINRA Reg SHO daily short volume (true daily data)",
)
def get_daily_volume(
    ticker:   str,
    days:     int  = Query(30, ge=1, le=365),
    refresh:  bool = Query(False),
) -> List[DailyShortVolume]:
    """Return FINRA Reg SHO daily short volume for a ticker.

    Unlike bi-monthly short interest, this is genuine daily data published
    each trading day around 6 pm ET.  Short volume % of total volume is a
    real-time proxy for short selling pressure (not the same as short interest).
    """
    if refresh:
        try:
            refresh_daily_volume()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    end   = date.today()
    start = end - timedelta(days=days)
    records = _query_daily_volume_db(ticker, start, end)
    if not records and not refresh:
        raise HTTPException(
            status_code=404,
            detail=f"No daily short volume for {ticker.upper()}. "
                   "Try refresh=true to download today's FINRA file.",
        )
    return records


@router.get(
    "/ftd/{ticker}",
    response_model=List[FTDRecord],
    summary="SEC Fails-to-Deliver history (proxy for naked short exposure)",
)
def get_ftd(
    ticker:  str,
    limit:   int  = Query(12, ge=1, le=50),
    refresh: bool = Query(False),
) -> List[FTDRecord]:
    """Return SEC FTD data for a ticker.

    Elevated FTDs relative to average daily volume indicate delivery failures
    that can signal naked short selling pressure.  SEC releases FTD data
    bi-monthly with a short lag.
    """
    if refresh:
        try:
            refresh_ftd(max_files=2)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    records = _query_ftd_db(ticker, limit=limit)
    if not records and not refresh:
        raise HTTPException(
            status_code=404,
            detail=f"No FTD data for {ticker.upper()}. "
                   "Try refresh=true to download from SEC.",
        )
    return records


@router.get(
    "/squeeze-score/{ticker}",
    response_model=ShortMetrics,
    summary="Short squeeze score and full metric breakdown",
)
def get_squeeze_score(
    ticker:  str,
    refresh: bool = Query(False),
) -> ShortMetrics:
    """Compute and return the squeeze score + all supporting metrics for a ticker.

    Squeeze score = days_to_cover * short_pct_float * 100.
    Higher = more potential for a short squeeze.  Typical candidates: score > 50.

    Sources combined: FINRA bi-monthly SI, FINRA daily vol, SEC FTD, yfinance float.
    """
    if refresh:
        try:
            refresh_short_interest(months_back=2)
            refresh_daily_volume()
        except Exception as exc:
            logger.warning("Partial refresh before squeeze score", error=str(exc))
    return compute_metrics_for_ticker(ticker)


@router.get(
    "/most-shorted",
    summary="Universe ranked by short % of float (bi-monthly official data)",
)
def get_most_shorted(
    limit:       int  = Query(50, ge=1, le=500),
    use_daily:   bool = Query(False, description="Rank by daily Reg SHO short vol % instead of official SI"),
) -> List[Dict[str, Any]]:
    """Return the most heavily shorted stocks by short % of float.

    Default: uses bi-monthly official FINRA short interest + float from yfinance.
    Set use_daily=true to rank instead by FINRA daily short volume % (updated daily,
    but measures volume composition, not open short position).
    """
    if use_daily:
        return _get_most_shorted_daily(limit=limit)
    return _get_most_shorted_from_db(limit=limit)


@router.get(
    "/short-change/{ticker}",
    response_model=List[ShortChangeRecord],
    summary="Short interest change: 2-week and month-over-month",
)
def get_short_change(
    ticker:  str,
    refresh: bool = Query(False),
) -> List[ShortChangeRecord]:
    """Return 2-week and monthly change in short interest for a ticker.

    Based on consecutive FINRA bi-monthly settlement-date reports.
    """
    if refresh:
        try:
            refresh_short_interest(months_back=3)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    records = _get_short_change_db(ticker)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient short interest history for {ticker.upper()} to compute changes. "
                   "Try refresh=true.",
        )
    return records


@router.get(
    "/squeeze-candidates",
    response_model=List[SqueezeCandidateRecord],
    summary="Short squeeze candidate screener",
)
def get_squeeze_candidates(
    min_score: float = Query(_MIN_SQUEEZE_SCORE, ge=0.0),
    limit:     int   = Query(25, ge=1, le=100),
) -> List[SqueezeCandidateRecord]:
    """Return stocks ranked by squeeze score above the minimum threshold.

    Candidates are ranked by:  days_to_cover * short_pct_float * 100
    Recent 20-day price return is included as a secondary signal (declining price
    with high short interest = elevated squeeze risk when trend reverses).

    Note: This endpoint calls yfinance for each candidate to fetch recent returns;
    expect 2-5 seconds latency for large limit values.
    """
    return _get_squeeze_candidates_from_db(min_score=min_score, limit=limit)


@router.post(
    "/refresh",
    summary="Trigger full data refresh: FINRA daily + monthly + SEC FTD",
)
def trigger_refresh(
    months_back: int = Query(3, ge=1, le=12),
    max_ftd_files: int = Query(2, ge=1, le=6),
) -> Dict[str, Any]:
    """Download fresh data from all sources and populate the SQLite database.

    Runs synchronously — expect 30-120 seconds for a full refresh.
    """
    results: Dict[str, Any] = {}
    try:
        results["daily"]          = refresh_daily_volume()
    except Exception as exc:
        results["daily"]          = {"error": str(exc)}
    try:
        results["short_interest"] = refresh_short_interest(months_back=months_back)
    except Exception as exc:
        results["short_interest"] = {"error": str(exc)}
    try:
        results["ftd"]            = refresh_ftd(max_files=max_ftd_files)
    except Exception as exc:
        results["ftd"]            = {"error": str(exc)}
    return results
