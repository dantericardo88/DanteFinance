"""DuckDB-backed OHLCV cache — cache-first with yfinance fallback.

Stores fetched bars locally in ~/.sentinel/cache/ohlcv.duckdb so that:
  - Re-runs are instant (no network).
  - The system works fully offline after first warm.
  - Large bulk downloads are chunked to avoid yfinance API limits.

Public API
----------
get_ohlcv(ticker, start, end, interval='1d') -> pd.DataFrame
    Cache-first read; falls back to yfinance and writes through.

warm_cache(tickers, years=10) -> dict
    Bulk warm for a list of tickers with progress tracking.

cache_stats() -> dict
    Summary: rows, tickers, date range, staleness.

get_cache() -> OHLCVCache
    Module-level singleton accessor.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports — never crash at import time if optional deps missing
try:
    import duckdb
    _DUCKDB_OK = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _DUCKDB_OK = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]
    _YF_OK = False

# Default cache location — override via SENTINEL_CACHE_PATH env var
_DEFAULT_CACHE_DIR = Path.home() / ".sentinel" / "cache"
_DEFAULT_DB_NAME = "ohlcv.duckdb"

# yfinance is reliable up to ~2-year windows; chunk larger requests
_CHUNK_YEARS = 2

_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_cache (
    ticker   VARCHAR NOT NULL,
    interval VARCHAR NOT NULL DEFAULT '1d',
    ts       DATE    NOT NULL,
    open     DOUBLE,
    high     DOUBLE,
    low      DOUBLE,
    close    DOUBLE,
    volume   BIGINT,
    fetched_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_ts
    ON ohlcv_cache (ticker, interval, ts);
"""


class OHLCVCache:
    """DuckDB-backed OHLCV cache with yfinance fallback.

    Thread-safety: DuckDB connections are not thread-safe. Each method opens
    its own short-lived connection and closes it immediately. For async
    workloads, run in an executor or use the module-level ``get_cache()``
    singleton which is safe for a single event loop.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            cache_dir = Path(
                os.environ.get("SENTINEL_CACHE_PATH", str(_DEFAULT_CACHE_DIR))
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / _DEFAULT_DB_NAME)
        self.db_path = db_path
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        if not _DUCKDB_OK:
            logger.warning("duckdb not installed — cache disabled")
            return
        with duckdb.connect(self.db_path) as con:
            con.executescript(_DDL)

    # ── Core read/write ───────────────────────────────────────────────────────

    def _read_cached(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str,
    ) -> "pd.DataFrame | None":
        """Return cached bars or None if cache is empty / unavailable."""
        if not (_DUCKDB_OK and _PANDAS_OK):
            return None
        with duckdb.connect(self.db_path) as con:
            df = con.execute(
                """
                SELECT ts AS Date, open AS Open, high AS High,
                       low AS Low, close AS Close, volume AS Volume
                FROM ohlcv_cache
                WHERE ticker = ?
                  AND interval = ?
                  AND ts >= ?
                  AND ts <= ?
                ORDER BY ts
                """,
                [ticker, interval, start.isoformat(), end.isoformat()],
            ).df()
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df

    def _write_cache(
        self,
        ticker: str,
        interval: str,
        df: "pd.DataFrame",
    ) -> int:
        """Upsert DataFrame rows into cache. Returns number of rows written."""
        if not (_DUCKDB_OK and _PANDAS_OK) or df is None or df.empty:
            return 0

        rows = []
        for ts, row in df.iterrows():
            def _safe(col: str) -> float | None:
                v = row.get(col)
                return float(v) if v is not None and pd.notna(v) else None

            rows.append((
                ticker,
                interval,
                ts.date() if hasattr(ts, "date") else ts,
                _safe("Open"),
                _safe("High"),
                _safe("Low"),
                _safe("Close"),
                int(row.get("Volume", 0) or 0),
            ))

        if not rows:
            return 0

        with duckdb.connect(self.db_path) as con:
            con.executemany(
                """
                INSERT INTO ohlcv_cache
                    (ticker, interval, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, interval, ts)
                DO UPDATE SET
                    open       = excluded.open,
                    high       = excluded.high,
                    low        = excluded.low,
                    close      = excluded.close,
                    volume     = excluded.volume,
                    fetched_at = current_timestamp
                """,
                rows,
            )
        return len(rows)

    # ── Chunk helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _date_chunks(
        start: date,
        end: date,
        chunk_years: int = _CHUNK_YEARS,
    ) -> list[tuple[date, date]]:
        """Split [start, end] into ≤chunk_years segments (avoids yfinance API limits)."""
        chunks: list[tuple[date, date]] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(
                date(
                    chunk_start.year + chunk_years,
                    chunk_start.month,
                    chunk_start.day,
                ),
                end,
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        return chunks

    # ── yfinance fallback ─────────────────────────────────────────────────────

    def _fetch_yfinance(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> "pd.DataFrame":
        """Download from yfinance for a single chunk. Returns raw DataFrame."""
        if not (_YF_OK and _PANDAS_OK):
            return pd.DataFrame() if _PANDAS_OK else None  # type: ignore[return-value]

        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return df

    # ── Public API ────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> "pd.DataFrame":
        """Cache-first OHLCV fetch. Falls back to yfinance on cache miss.

        Parameters
        ----------
        ticker:   e.g. 'AAPL', 'SPY', 'BTC-USD'
        start:    inclusive start date
        end:      inclusive end date
        interval: yfinance interval string ('1d', '1wk', '1mo', etc.)

        Returns
        -------
        pd.DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        Empty DataFrame if data unavailable.
        """
        ticker = ticker.upper()

        # 1. Cache hit?
        cached = self._read_cached(ticker, start, end, interval)
        if cached is not None and not cached.empty:
            logger.debug("Cache hit", ticker=ticker, start=start, end=end)
            return cached

        # 2. Fetch from yfinance in 2-year chunks
        if not (_YF_OK and _PANDAS_OK):
            logger.warning("yfinance/pandas not available", ticker=ticker)
            return pd.DataFrame() if _PANDAS_OK else None  # type: ignore[return-value]

        chunks = self._date_chunks(start, end)
        frames: list["pd.DataFrame"] = []

        for chunk_start, chunk_end in chunks:
            try:
                df_chunk = self._fetch_yfinance(ticker, chunk_start, chunk_end, interval)
                if df_chunk is not None and not df_chunk.empty:
                    # Flatten MultiIndex columns (yfinance >= 0.2 with single ticker)
                    if isinstance(df_chunk.columns, pd.MultiIndex):
                        df_chunk.columns = df_chunk.columns.get_level_values(0)
                    rows_written = self._write_cache(ticker, interval, df_chunk)
                    logger.debug(
                        "Chunk fetched+cached",
                        ticker=ticker,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        rows=rows_written,
                    )
                    frames.append(df_chunk)
            except Exception as exc:
                logger.warning(
                    "yfinance chunk failed",
                    ticker=ticker,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    error=str(exc),
                )

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames)
        # Deduplicate (overlapping chunk edges)
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        # Filter to requested range
        result = result.loc[
            (result.index >= pd.Timestamp(start))
            & (result.index <= pd.Timestamp(end))
        ]
        return result

    def warm_cache(
        self,
        tickers: list[str],
        years: int = 10,
        interval: str = "1d",
        delay_s: float = 0.3,
    ) -> dict:
        """Bulk warm cache for a list of tickers.

        Parameters
        ----------
        tickers:  list of ticker symbols
        years:    how many years of history to fetch
        interval: OHLCV interval
        delay_s:  courtesy delay between tickers (seconds)

        Returns
        -------
        dict with keys: ok, failed, skipped, total_rows, errors
        """
        end = date.today()
        start = date(end.year - years, end.month, end.day)

        ok: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        total_rows = 0

        logger.info(
            "warm_cache starting",
            tickers=len(tickers),
            years=years,
            start=start,
            end=end,
        )

        for i, ticker in enumerate(tickers, 1):
            t0 = time.monotonic()
            try:
                # Skip if already fully cached
                cached = self._read_cached(ticker, start, end, interval)
                if cached is not None and not cached.empty:
                    logger.debug("Already cached — skip", ticker=ticker)
                    ok.append(ticker)
                    total_rows += len(cached)
                    continue

                df = self.get_ohlcv(ticker, start, end, interval)
                if df is not None and not df.empty:
                    ok.append(ticker)
                    total_rows += len(df)
                    elapsed = round(time.monotonic() - t0, 2)
                    logger.info(
                        "Warmed",
                        ticker=ticker,
                        rows=len(df),
                        elapsed=elapsed,
                        progress=f"{i}/{len(tickers)}",
                    )
                else:
                    failed.append(ticker)
                    errors.append(f"{ticker}: empty response")
                    logger.warning("No data", ticker=ticker, progress=f"{i}/{len(tickers)}")

            except Exception as exc:
                failed.append(ticker)
                errors.append(f"{ticker}: {exc}")
                logger.error(
                    "warm_cache error",
                    ticker=ticker,
                    error=str(exc),
                    progress=f"{i}/{len(tickers)}",
                )

            if delay_s > 0:
                time.sleep(delay_s)

        result = {
            "ok": len(ok),
            "failed": len(failed),
            "skipped": 0,
            "total_rows": total_rows,
            "errors": errors,
        }
        logger.info("warm_cache complete", **{k: v for k, v in result.items() if k != "errors"})
        return result

    def cache_stats(self) -> dict:
        """Return summary statistics about the local cache.

        Returns
        -------
        dict with keys:
          total_rows     — total OHLCV bars cached
          tickers        — number of unique tickers
          ticker_list    — sorted list of cached tickers
          earliest_date  — oldest bar date
          latest_date    — newest bar date
          stale_tickers  — tickers whose latest bar is > 2 trading days old
          db_size_mb     — DuckDB file size in MB
        """
        if not _DUCKDB_OK:
            return {"error": "duckdb not installed"}

        today = date.today()
        staleness_threshold = today - timedelta(days=3)

        try:
            with duckdb.connect(self.db_path) as con:
                summary = con.execute(
                    """
                    SELECT
                        COUNT(*)                                    AS total_rows,
                        COUNT(DISTINCT ticker)                      AS tickers,
                        MIN(ts)                                     AS earliest_date,
                        MAX(ts)                                     AS latest_date
                    FROM ohlcv_cache
                    """
                ).fetchone()

                ticker_list = [
                    r[0]
                    for r in con.execute(
                        "SELECT DISTINCT ticker FROM ohlcv_cache ORDER BY ticker"
                    ).fetchall()
                ]

                stale = [
                    r[0]
                    for r in con.execute(
                        """
                        SELECT ticker
                        FROM (
                            SELECT ticker, MAX(ts) AS latest
                            FROM ohlcv_cache
                            WHERE interval = '1d'
                            GROUP BY ticker
                        )
                        WHERE latest < ?
                        ORDER BY ticker
                        """,
                        [staleness_threshold.isoformat()],
                    ).fetchall()
                ]

            db_size_mb = round(
                os.path.getsize(self.db_path) / (1024 * 1024), 2
            ) if os.path.exists(self.db_path) else 0.0

            return {
                "total_rows": summary[0] or 0,
                "tickers": summary[1] or 0,
                "ticker_list": ticker_list,
                "earliest_date": str(summary[2]) if summary[2] else None,
                "latest_date": str(summary[3]) if summary[3] else None,
                "stale_tickers": stale,
                "db_size_mb": db_size_mb,
            }
        except Exception as exc:
            logger.error("cache_stats failed", error=str(exc))
            return {"error": str(exc)}


# ── Module-level singleton ────────────────────────────────────────────────────

_cache_instance: Optional[OHLCVCache] = None


def get_cache(db_path: Optional[str] = None) -> OHLCVCache:
    """Return (or create) the module-level OHLCVCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = OHLCVCache(db_path=db_path)
    return _cache_instance
