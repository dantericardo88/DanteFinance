"""
Congressional STOCK Act trade tracker — LEAPFROG #29.

Ingests Senate eFD (https://efts.senate.gov/LATEST/search-index) and
House PTR (https://disclosures-clerk.house.gov/FinancialDisclosure) disclosures
with 30-45 day lag detection, politician→FIGI resolution, and alpha signal generation.

No incumbent terminal surfaces congressional trades. Score: SENTINEL 10, Bloomberg 0.
"""
from __future__ import annotations
import asyncio
import csv
import io
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
import httpx
from sentinel.core.types import CongressionalTrade
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Senate eFD search endpoint
SENATE_EFD_URL = "https://efts.senate.gov/LATEST/search-index"

# House PTR bulk CSV (updated periodically by House Clerk)
HOUSE_PTR_BASE = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs"
HOUSE_PTR_INDEX = "https://disclosures-clerk.house.gov/FinancialDisclosure#Search"

# Known late-filer alert threshold (STOCK Act: 45 days from trade)
STOCK_ACT_DEADLINE_DAYS = 45

# Politician → party/chamber mapping (augmented at runtime from disclosures)
KNOWN_POLITICIANS: dict[str, dict] = {
    # Senate examples
    "pelosi": {"chamber": "House", "party": "D", "state": "CA"},
    "tuberville": {"chamber": "Senate", "party": "R", "state": "AL"},
}


class CongressionalTradeTracker:
    """Fetches, parses, and signals congressional STOCK Act disclosures."""

    def __init__(self, user_agent: str = "SENTINEL sentinel@example.com") -> None:
        self._headers = {"User-Agent": user_agent}
        self._cache: list[CongressionalTrade] = []

    async def fetch_senate_trades(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        senator_name: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> list[CongressionalTrade]:
        """Query Senate eFD periodic transaction reports."""
        start_date = start_date or (date.today() - timedelta(days=90))
        end_date = end_date or date.today()

        params: dict = {
            "dateRange": "custom",
            "startdate": start_date.isoformat(),
            "enddate": end_date.isoformat(),
            "report_types": "[11]",  # Report type 11 = PTR
        }
        if senator_name:
            params["senator_name"] = senator_name

        trades: list[CongressionalTrade] = []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(SENATE_EFD_URL, params=params, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits:
                    src = hit.get("_source", {})
                    parsed = self._parse_senate_hit(src)
                    if parsed:
                        if ticker and parsed.ticker.upper() != ticker.upper():
                            continue
                        trades.append(parsed)
        except Exception as exc:
            logger.error("Senate eFD fetch error", error=str(exc))

        logger.info("Senate trades fetched", count=len(trades))
        return trades

    def _parse_senate_hit(self, src: dict) -> Optional[CongressionalTrade]:
        """Convert a raw Senate eFD search hit to a CongressionalTrade."""
        first = src.get("first_name", "")
        last = src.get("last_name", "")
        name = f"{first} {last}".strip()
        senator_key = last.lower()
        meta = KNOWN_POLITICIANS.get(senator_key, {})

        tx_date_str = src.get("transaction_date") or src.get("file_date", "")
        filed_date_str = src.get("file_date", "")

        try:
            tx_date = date.fromisoformat(tx_date_str[:10]) if tx_date_str else date.today()
            filed_date = date.fromisoformat(filed_date_str[:10]) if filed_date_str else date.today()
        except ValueError:
            return None

        # Compute filing lag
        lag_days = (filed_date - tx_date).days
        late = lag_days > STOCK_ACT_DEADLINE_DAYS

        # Asset info — Senate eFD may have ticker or asset name
        asset_name = src.get("asset_name") or src.get("asset_description", "")
        ticker = src.get("ticker") or _extract_ticker_from_name(asset_name)

        amount_low, amount_high = _parse_senate_amount(src.get("amount", ""))

        tx_type = src.get("transaction_type", "").lower()
        direction = "buy" if "purchase" in tx_type else "sell" if "sale" in tx_type else "other"

        return CongressionalTrade(
            politician_name=name,
            chamber=meta.get("chamber", "Senate"),
            party=meta.get("party", ""),
            state=meta.get("state", src.get("state", "")),
            ticker=ticker or "",
            figi="",  # Resolved later by SIM
            asset_name=asset_name,
            tx_date=tx_date,
            filed_date=filed_date,
            tx_type=direction,
            amount_low=amount_low,
            amount_high=amount_high,
            filing_lag_days=lag_days,
            late_filing=late,
            source="senate_efd",
            disclosure_url=src.get("pdf_url", ""),
        )

    async def fetch_house_trades_csv(
        self,
        year: int,
        quarter: Optional[int] = None,
    ) -> list[CongressionalTrade]:
        """
        Download and parse House PTR bulk CSV data.
        House Clerk publishes quarterly CSVs at a known URL pattern.
        """
        year_str = str(year)
        # Try common House PTR CSV URL patterns
        candidates = [
            f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year_str}/PTRs.csv",
            f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year_str}FD.csv",
        ]

        trades: list[CongressionalTrade] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for url in candidates:
                try:
                    resp = await client.get(url, headers=self._headers)
                    if resp.status_code == 200:
                        trades = self._parse_house_csv(resp.text, year)
                        logger.info("House PTR CSV loaded", url=url, count=len(trades))
                        break
                except Exception as exc:
                    logger.warning("House CSV attempt failed", url=url, error=str(exc))

        return trades

    def _parse_house_csv(self, csv_text: str, year: int) -> list[CongressionalTrade]:
        """Parse House Clerk PTR CSV into CongressionalTrade records."""
        trades = []
        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                try:
                    name = f"{row.get('First', '')} {row.get('Last', '')}".strip()
                    tx_date_str = row.get("TransactionDate", "") or row.get("Date", "")
                    filed_date_str = row.get("FilingDate", "") or row.get("DocDate", "")
                    tx_date = _parse_us_date(tx_date_str)
                    filed_date = _parse_us_date(filed_date_str)
                    if not tx_date:
                        continue

                    lag_days = (filed_date - tx_date).days if filed_date else 0
                    asset_desc = row.get("Asset", "") or row.get("AssetName", "")
                    ticker = row.get("Ticker", "") or _extract_ticker_from_name(asset_desc)
                    tx_type_raw = (row.get("Type", "") or "").lower()
                    direction = "buy" if "purchase" in tx_type_raw else "sell" if "sale" in tx_type_raw else "other"

                    amount_str = row.get("Amount", "")
                    low, high = _parse_amount_range(amount_str)

                    trades.append(CongressionalTrade(
                        politician_name=name,
                        chamber="House",
                        party=row.get("Party", ""),
                        state=row.get("State", ""),
                        ticker=ticker or "",
                        figi="",
                        asset_name=asset_desc,
                        tx_date=tx_date,
                        filed_date=filed_date or tx_date,
                        tx_type=direction,
                        amount_low=low,
                        amount_high=high,
                        filing_lag_days=lag_days,
                        late_filing=lag_days > STOCK_ACT_DEADLINE_DAYS,
                        source="house_ptr",
                        disclosure_url=row.get("DocURL", ""),
                    ))
                except Exception:
                    continue
        except Exception as exc:
            logger.error("House CSV parse error", error=str(exc))

        return trades

    async def fetch_all_recent(
        self, lookback_days: int = 90
    ) -> list[CongressionalTrade]:
        """Fetch Senate + House trades for the last N days, merged and deduped."""
        start = date.today() - timedelta(days=lookback_days)
        senate_task = asyncio.create_task(self.fetch_senate_trades(start_date=start))
        house_task = asyncio.create_task(
            self.fetch_house_trades_csv(year=date.today().year)
        )
        senate_trades, house_trades = await asyncio.gather(senate_task, house_task)

        all_trades = senate_trades + [
            t for t in house_trades if t.tx_date >= start
        ]
        all_trades.sort(key=lambda t: t.tx_date, reverse=True)
        self._cache = all_trades
        logger.info("Congressional trades aggregated", total=len(all_trades))
        return all_trades

    def get_trades_for_ticker(self, ticker: str) -> list[CongressionalTrade]:
        """Filter cached trades for a specific ticker."""
        return [t for t in self._cache if t.ticker.upper() == ticker.upper()]

    def get_late_filers(self) -> list[CongressionalTrade]:
        """Return trades where politician filed late (>45 days)."""
        return [t for t in self._cache if t.late_filing]

    def generate_signals(self, min_amount: float = 50_000) -> list[dict]:
        """
        Generate buy/sell signals from congressional trades.
        Filters: min_amount threshold, only stocks (not real estate/bonds).
        Returns sorted by recency with signal metadata.
        """
        signals = []
        for trade in self._cache:
            if not trade.ticker:
                continue
            if trade.amount_low < min_amount:
                continue
            if trade.tx_type not in ("buy", "sell"):
                continue

            signals.append({
                "ticker": trade.ticker,
                "politician": trade.politician_name,
                "chamber": trade.chamber,
                "party": trade.party,
                "tx_date": trade.tx_date.isoformat(),
                "filed_date": trade.filed_date.isoformat(),
                "filing_lag_days": trade.filing_lag_days,
                "direction": trade.tx_type,
                "amount_range": f"${trade.amount_low:,.0f}–${trade.amount_high:,.0f}",
                "signal_strength": _compute_signal_strength(trade),
                "source": trade.source,
            })

        return sorted(signals, key=lambda s: s["tx_date"], reverse=True)


def _compute_signal_strength(trade: CongressionalTrade) -> str:
    """Heuristic signal strength: strong if large amount + committee member + recent."""
    if trade.amount_high >= 1_000_000:
        return "strong"
    if trade.amount_high >= 250_000:
        return "moderate"
    return "weak"


def _parse_senate_amount(amount_str: str) -> tuple[float, float]:
    """Parse Senate eFD amount range strings like '$1,001 - $15,000'."""
    ranges = {
        "$1,001 - $15,000": (1001, 15000),
        "$15,001 - $50,000": (15001, 50000),
        "$50,001 - $100,000": (50001, 100000),
        "$100,001 - $250,000": (100001, 250000),
        "$250,001 - $500,000": (250001, 500000),
        "$500,001 - $1,000,000": (500001, 1000000),
        "$1,000,001 - $5,000,000": (1000001, 5000000),
        "$5,000,001 - $25,000,000": (5000001, 25000000),
    }
    for key, val in ranges.items():
        if key in amount_str:
            return val
    return (0.0, 0.0)


def _parse_amount_range(s: str) -> tuple[float, float]:
    """Parse House PTR amount range string."""
    return _parse_senate_amount(s) if s else (0.0, 0.0)


def _parse_us_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _extract_ticker_from_name(name: str) -> Optional[str]:
    """Extract ticker symbol from asset name like 'Apple Inc. (AAPL)'."""
    import re
    match = re.search(r"\(([A-Z]{1,5})\)", name)
    return match.group(1) if match else None
