"""Congressional trades adapter — STOCK Act disclosures for House and Senate.

Data sources (both free, no API key required):
  House: House Stock Watcher community project
         https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json
  Senate: Senate Stock Watcher community project
          https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json

Both sources parse official STOCK Act filings (PFD reports) and expose them as
structured JSON. Data includes ~48h lag from official disclosure.

Why this matters:
  - Congresspeople outperform SPY by ~7% annualized (Ziobrowski 2004, 2011)
  - Average disclosure lag is 28 days (legal max 45 days)
  - Late filers are correlated with larger outperformance
"""
from __future__ import annotations
import asyncio
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import httpx

from sentinel.core.types import DataHealthEvent
from sentinel.sds.base_adapter import BaseAdapter
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

HOUSE_URL = "https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json"
SENATE_URL = "https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json"

# Amount range midpoints for normalizing the disclosure buckets
AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8000,
    "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000,
    "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000,
    "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000,
    "$5,000,001 - $25,000,000": 15000000,
    "$25,000,001 - $50,000,000": 37500000,
    "Over $50,000,000": 75000000,
}


def _parse_amount(raw: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
    """Return (low, high) as Decimal from a STOCK Act amount range string."""
    if not raw:
        return None, None
    for label, mid in AMOUNT_MIDPOINTS.items():
        if raw.strip() in (label, label.replace(",", "")):
            # Reconstruct low/high from label
            parts = label.replace("$", "").replace(",", "").split(" - ")
            if len(parts) == 2:
                try:
                    return Decimal(parts[0].strip()), Decimal(parts[1].strip())
                except Exception:
                    pass
            return Decimal(str(mid)), Decimal(str(mid))
    return None, None


def _parse_date(raw: str) -> Optional[date]:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _tx_code(tx_type: str) -> str:
    """Normalise to SEC transaction codes: P = purchase, S = sale."""
    t = (tx_type or "").lower()
    if "purchase" in t or "buy" in t:
        return "P"
    if "sale" in t or "sell" in t or "sold" in t:
        return "S"
    if "exchange" in t:
        return "X"
    return "O"


class CongressAdapter(BaseAdapter):
    """Fetches and normalises STOCK Act congressional trade disclosures."""

    name = "congress"
    rate_limit_per_min = 10  # S3 buckets — be polite

    async def fetch_ohlcv(self, ticker, start, end, interval="1d", figi=None):
        return []  # Congress adapter doesn't provide price data

    async def fetch_house_trades(
        self,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Fetch all House member trades. Returns list of normalised dicts."""
        await self._throttle()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(HOUSE_URL)
                resp.raise_for_status()
                raw: list[dict] = resp.json()
        except Exception as exc:
            logger.error("House trades fetch failed", error=str(exc))
            return []

        trades = []
        for item in raw:
            tx_date = _parse_date(item.get("transaction_date") or item.get("disclosure_date"))
            if since and tx_date and tx_date < since:
                continue
            filed_date = _parse_date(item.get("disclosure_date"))
            low, high = _parse_amount(item.get("amount", ""))
            ticker = (item.get("ticker") or "").strip().upper() or None
            trades.append({
                "politician_name": item.get("representative", ""),
                "chamber": "house",
                "party": (item.get("party") or "")[:5].upper(),
                "state": (item.get("state") or "")[:5].upper(),
                "ticker": ticker,
                "figi": None,
                "asset_name": item.get("asset_description", ""),
                "tx_date": tx_date,
                "filed_date": filed_date,
                "tx_type": item.get("type", ""),
                "tx_code": _tx_code(item.get("type", "")),
                "amount_low": low,
                "amount_high": high,
                "filing_lag_days": (
                    (filed_date - tx_date).days
                    if tx_date and filed_date else None
                ),
                "late_filing": (
                    (filed_date - tx_date).days > 45
                    if tx_date and filed_date else False
                ),
                "source": "house_stock_watcher",
                "disclosure_url": item.get("ptr_link"),
            })

        logger.info("House trades fetched", count=len(trades))
        return trades

    async def fetch_senate_trades(
        self,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Fetch all Senate member trades. Returns list of normalised dicts."""
        await self._throttle()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(SENATE_URL)
                resp.raise_for_status()
                raw: list[dict] = resp.json()
        except Exception as exc:
            logger.error("Senate trades fetch failed", error=str(exc))
            return []

        trades = []
        for item in raw:
            tx_date = _parse_date(item.get("transaction_date"))
            if since and tx_date and tx_date < since:
                continue
            filed_date = _parse_date(item.get("disclosure_date"))
            low, high = _parse_amount(item.get("amount", ""))
            ticker = (item.get("ticker") or "").strip().upper() or None
            trades.append({
                "politician_name": item.get("first_name", "") + " " + item.get("last_name", ""),
                "chamber": "senate",
                "party": (item.get("party") or "")[:5].upper(),
                "state": (item.get("state") or "")[:5].upper(),
                "ticker": ticker,
                "figi": None,
                "asset_name": item.get("asset_description", ""),
                "tx_date": tx_date,
                "filed_date": filed_date,
                "tx_type": item.get("type", ""),
                "tx_code": _tx_code(item.get("type", "")),
                "amount_low": low,
                "amount_high": high,
                "filing_lag_days": (
                    (filed_date - tx_date).days
                    if tx_date and filed_date else None
                ),
                "late_filing": (
                    (filed_date - tx_date).days > 45
                    if tx_date and filed_date else False
                ),
                "source": "senate_stock_watcher",
                "disclosure_url": item.get("link"),
            })

        logger.info("Senate trades fetched", count=len(trades))
        return trades

    async def fetch_all_trades(
        self,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Fetch both chambers concurrently."""
        house, senate = await asyncio.gather(
            self.fetch_house_trades(since=since),
            self.fetch_senate_trades(since=since),
        )
        return house + senate

    async def fetch_trades_by_ticker(
        self,
        ticker: str,
        since: Optional[date] = None,
    ) -> list[dict]:
        """Return all trades for a specific ticker across both chambers."""
        all_trades = await self.fetch_all_trades(since=since)
        return [t for t in all_trades if t.get("ticker") == ticker.upper()]

    async def health_check(self) -> DataHealthEvent:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(HOUSE_URL, headers={"Range": "bytes=0-1000"})
                if resp.status_code in (200, 206):
                    return self._ok_event()
            return self._error_event("House stock watcher unreachable")
        except Exception as exc:
            return self._error_event(str(exc))
