"""FINRA short interest adapter — daily short volume CSV and bi-monthly Reg SHO data."""
from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel.core.logging import get_logger
from sentinel.core.types import DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_FINRA_CSV_BASE = "https://cdn.finra.org/equity/regsho/daily"
_FINRA_SHORT_INT_URL = "https://api.finra.org/data/group/equity/name/shortInterest"

_TIMEOUT = httpx.Timeout(60.0)   # CSV files can be several MB
_HEADERS = {
    "User-Agent": "SENTINEL/1.0 (DanteFinance; contact@danteai.com)",
    "Accept": "application/json",
}
_PAGE_LIMIT = 500   # FINRA API max rows per request


# ── Models ───────────────────────────────────────────────────────────────────

class ShortVolumeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    date: date
    short_volume: int
    total_volume: int
    short_pct: float    # short_volume / total_volume


class ShortInterestRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    settlement_date: date
    short_interest: int
    days_to_cover: float | None = None
    source: str = "finra"


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _safe_int(v: str) -> int | None:
    try:
        return int(v.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_finra_date(s: str) -> date | None:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_short_volume_csv(raw: bytes, report_date: date) -> list[ShortVolumeRecord]:
    """Parse FINRA CNMS short volume pipe-delimited file.

    Format: MARKET|SYMBOL|DATE|SHORTVOLUME|SHORTEXEMPTVOLUME|TOTALVOLUME|MARKET
    Footer / summary lines are skipped via symbol sanity checks.
    """
    text = raw.decode("utf-8", errors="replace")
    records: list[ShortVolumeRecord] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="|")

    for row in reader:
        normalised = {k.strip().upper(): v for k, v in row.items()}
        symbol = normalised.get("SYMBOL", "").strip().upper()
        # Skip summary rows, blank lines, and implausibly long "symbols"
        if not symbol or not symbol[0].isalpha() or len(symbol) > 10:
            continue

        short_vol = _safe_int(normalised.get("SHORTVOLUME", ""))
        total_vol = _safe_int(normalised.get("TOTALVOLUME", ""))
        if short_vol is None or total_vol is None or total_vol == 0:
            continue

        raw_date = normalised.get("DATE", "").strip()
        rec_date = _parse_finra_date(raw_date) if raw_date else report_date

        records.append(ShortVolumeRecord(
            ticker=symbol,
            date=rec_date or report_date,
            short_volume=short_vol,
            total_volume=total_vol,
            short_pct=round(short_vol / total_vol, 6),
        ))

    return records


# ── ShortInterestAdapter ──────────────────────────────────────────────────────

class ShortInterestAdapter(BaseAdapter):
    """FINRA short interest adapter.

    Two data sources:
    - Daily short volume CSV  (cdn.finra.org) — published each trading day ~4 pm ET
    - FINRA shortInterest API (api.finra.org) — bi-monthly settlement data
    """

    name = "short_interest"
    rate_limit_per_min = 30     # FINRA CDN has no published limit; be polite
    FINRA_CSV_BASE = _FINRA_CSV_BASE

    def __init__(self) -> None:
        super().__init__()
        # In-process cache keyed by YYYYMMDD — avoids re-downloading within a session
        self._csv_cache: dict[str, list[ShortVolumeRecord]] = {}

    # ── Internal HTTP ─────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _get_bytes(self, url: str) -> bytes:
        """Download raw bytes with tenacity retry (for CSV files)."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers={"User-Agent": _HEADERS["User-Agent"]})
            if resp.status_code == 429:
                raise httpx.HTTPError("429 Too Many Requests")
            resp.raise_for_status()
            return resp.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _get_json_finra(self, url: str, params: dict | None = None) -> Any:
        """GET FINRA JSON API with retry."""
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                raise httpx.HTTPError("429 Too Many Requests")
            resp.raise_for_status()
            return resp.json()

    # ── CSV download ──────────────────────────────────────────────────────────

    async def _download_csv_for_date(self, candidate: date) -> list[ShortVolumeRecord] | None:
        """Try to download and parse the FINRA daily CSV for *candidate*.

        Returns None if the file is not yet published (404) or parses empty.
        """
        cache_key = candidate.strftime("%Y%m%d")
        if cache_key in self._csv_cache:
            return self._csv_cache[cache_key]

        url = f"{self.FINRA_CSV_BASE}/CNMSshvol{cache_key}.txt"
        try:
            logger.info("Fetching FINRA short volume CSV", date=cache_key, url=url)
            raw = await self._get_bytes(url)
            records = _parse_short_volume_csv(raw, candidate)
            if records:
                self._csv_cache[cache_key] = records
                logger.info("FINRA short volume parsed", date=cache_key, records=len(records))
                return records
            logger.warning("FINRA CSV parsed but empty", date=cache_key)
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("FINRA CSV not yet published", date=cache_key)
                return None
            logger.error("FINRA CSV HTTP error", date=cache_key, status=exc.response.status_code)
            return None
        except Exception as exc:
            logger.error("FINRA CSV fetch error", date=cache_key, error=str(exc))
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_short_volume(self, for_date: date | None = None) -> list[ShortVolumeRecord]:
        """Download FINRA daily short volume CSV.

        If *for_date* is None, tries today then steps back up to 4 calendar
        days to account for the ~4 pm ET publication lag and weekends/holidays.
        Returns the first successfully parsed file found, or [] if none available.
        """
        await self._throttle()

        if for_date is not None:
            result = await self._download_csv_for_date(for_date)
            return result or []

        today = date.today()
        for delta in range(5):
            candidate = today - timedelta(days=delta)
            result = await self._download_csv_for_date(candidate)
            if result:
                return result

        logger.warning("No FINRA short volume CSV found for any candidate date")
        return []

    async def fetch_for_ticker(self, ticker: str, for_date: date | None = None) -> ShortVolumeRecord | None:
        """Get short volume for a specific ticker from the latest daily file.

        Returns None if the ticker is absent from the file.
        """
        ticker = ticker.upper().strip()
        records = await self.fetch_short_volume(for_date)
        for rec in records:
            if rec.ticker == ticker:
                return rec
        logger.info("Ticker not found in FINRA short volume file", ticker=ticker)
        return None

    async def get_most_shorted(self, limit: int = 50) -> list[ShortVolumeRecord]:
        """Return top *limit* names by short_pct descending from the latest daily file."""
        records = await self.fetch_short_volume()
        ranked = sorted(records, key=lambda r: r.short_pct, reverse=True)
        result = ranked[:limit]
        logger.info("Most shorted by volume pct", returned=len(result))
        return result

    async def get_squeeze_candidates(self, min_short_pct: float = 0.40) -> list[ShortVolumeRecord]:
        """Filter for heavily shorted names — potential squeeze setups.

        A short_pct >= 0.40 means at least 40% of reported volume on that day
        was short-side, flagging crowded shorts that can be forced to cover quickly.
        Results are sorted by short_pct descending.
        """
        records = await self.fetch_short_volume()
        candidates = [r for r in records if r.short_pct >= min_short_pct]
        candidates.sort(key=lambda r: r.short_pct, reverse=True)
        logger.info(
            "Short squeeze candidates",
            screened=len(records),
            candidates=len(candidates),
            min_short_pct=min_short_pct,
        )
        return candidates

    # ── FINRA JSON API (bi-monthly short interest) ───────────────────────────

    async def _fetch_short_interest_page(
        self,
        offset: int,
        tickers: list[str] | None,
    ) -> list[dict]:
        """Fetch one page of FINRA shortInterest API data."""
        params: dict[str, Any] = {
            "limit": _PAGE_LIMIT,
            "offset": offset,
            "fields": "symbolCode,shortInterest,settlementDate,averageDailyShareVolume",
            "compareFilters": "gt:shortInterest:0",
            "sortFields": "-shortInterest",
        }
        if tickers:
            params["domainFilters"] = f"symbolCode:{'|'.join(tickers)}"

        try:
            data = await self._get_json_finra(_FINRA_SHORT_INT_URL, params=params)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data") or data.get("results") or []
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "FINRA shortInterest page failed",
                offset=offset,
                status=exc.response.status_code,
            )
            return []
        except Exception as exc:
            logger.error("FINRA shortInterest page error", offset=offset, error=str(exc))
            return []

    def _build_short_interest_record(self, row: dict) -> ShortInterestRecord | None:
        """Map one FINRA API row to ShortInterestRecord."""
        ticker = (
            row.get("symbolCode")
            or row.get("issueSymbolIdentifier")
            or row.get("symbol")
            or ""
        ).strip().upper()
        if not ticker:
            return None

        raw_date = row.get("settlementDate") or row.get("reportDate") or ""
        settlement_date = _parse_finra_date(str(raw_date)) if raw_date else None
        if settlement_date is None:
            return None

        short_interest = _safe_int(str(row.get("shortInterest", "")))
        if short_interest is None:
            return None

        avg_vol_raw = row.get("averageDailyShareVolume") or row.get("avgDailyVolume")
        avg_daily_volume = _safe_int(str(avg_vol_raw)) if avg_vol_raw else None

        days_to_cover: float | None = None
        if avg_daily_volume and avg_daily_volume > 0:
            days_to_cover = round(short_interest / avg_daily_volume, 4)

        return ShortInterestRecord(
            ticker=ticker,
            settlement_date=settlement_date,
            short_interest=short_interest,
            days_to_cover=days_to_cover,
        )

    async def fetch_short_interest(
        self,
        tickers: list[str] | None = None,
    ) -> list[ShortInterestRecord]:
        """Fetch bi-monthly FINRA short interest data via the JSON API.

        If *tickers* is None, pages through the full universe (safety cap: 20 000 rows).
        """
        await self._throttle()
        records: list[ShortInterestRecord] = []

        if tickers:
            normalized = [t.strip().upper() for t in tickers]
            rows = await self._fetch_short_interest_page(0, normalized)
            for row in rows:
                rec = self._build_short_interest_record(row)
                if rec:
                    records.append(rec)
            logger.info("FINRA short interest (filtered)", tickers=len(normalized), records=len(records))
            return records

        offset = 0
        while True:
            rows = await self._fetch_short_interest_page(offset, None)
            if not rows:
                break
            for row in rows:
                rec = self._build_short_interest_record(row)
                if rec:
                    records.append(rec)
            if len(rows) < _PAGE_LIMIT:
                break   # last page
            offset += _PAGE_LIMIT
            if offset >= 20_000:
                logger.warning("FINRA short interest pagination cap reached", offset=offset)
                break

        logger.info("FINRA short interest (full)", records=len(records))
        return records

    # ── BaseAdapter contract ──────────────────────────────────────────────────

    async def fetch_ohlcv(self, ticker, start, end, interval="1d", figi=None):
        """Not applicable for short interest data."""
        return []

    async def health_check(self) -> DataHealthEvent:
        """Verify the FINRA CSV endpoint is accessible by probing the most recent file."""
        try:
            records = await self.fetch_short_volume()
            if records:
                return self._ok_event()
            return self._error_event("FINRA CSV returned 0 records")
        except Exception as exc:
            logger.error("ShortInterest health check failed", error=str(exc))
            return self._error_event(str(exc))
