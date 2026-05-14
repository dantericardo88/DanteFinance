"""Survivorship bias registry.

Tracks delisted, bankrupt, and acquired securities so backtests include them.
A universe built only from currently-listed stocks inflates returns by 1–4%/yr
because it systematically excludes the stocks that went to zero.

Usage:
    universe = build_point_in_time_universe(active_tickers, as_of_date=date(2007, 1, 1))
    # universe now includes Lehman, Bear Stearns, Enron etc. that were alive in 2007

Seeds: 8 famous failures with verified CIKs. Extend via register() or EDGAR discovery.
"""
from __future__ import annotations
from datetime import date
from typing import Optional

from sentinel.core.types import DelistReason, SurvivorshipRecord
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_SEEDS: list[SurvivorshipRecord] = [
    SurvivorshipRecord(
        cik="0000806157", ticker="LEHMQ", company_name="Lehman Brothers Holdings Inc",
        delist_date=date(2008, 9, 15), delist_reason=DelistReason.BANKRUPTCY, exchange="NYSE",
        notes="Filed Chapter 11 2008-09-15. Largest US bankruptcy at the time ($639B assets).",
    ),
    SurvivorshipRecord(
        cik="0000101830", ticker="ENRNQ", company_name="Enron Corporation",
        delist_date=date(2001, 12, 2), delist_reason=DelistReason.BANKRUPTCY, exchange="NYSE",
        notes="Filed Chapter 11 2001-12-02. Accounting fraud. $63B in assets.",
    ),
    SurvivorshipRecord(
        cik="0000777819", ticker="BSC", company_name="Bear Stearns Companies Inc",
        delist_date=date(2008, 5, 30), delist_reason=DelistReason.ACQUISITION, exchange="NYSE",
        notes="Emergency acquisition by JP Morgan at $10/share (down from $172 peak).",
    ),
    SurvivorshipRecord(
        cik="0000200406", ticker="CC", company_name="Circuit City Stores Inc",
        delist_date=date(2009, 3, 8), delist_reason=DelistReason.BANKRUPTCY, exchange="NYSE",
        notes="Filed Chapter 11 2008-11-10. Full liquidation completed March 2009.",
    ),
    SurvivorshipRecord(
        cik="0000933136", ticker="WAMUQ", company_name="Washington Mutual Inc",
        delist_date=date(2008, 9, 26), delist_reason=DelistReason.BANKRUPTCY, exchange="NASDAQ",
        notes="Largest US bank failure. FDIC seized WaMu Bank 2008-09-25. $307B in assets.",
    ),
    SurvivorshipRecord(
        cik="0000040987", ticker="GMGMQ", company_name="General Motors Corporation",
        delist_date=date(2009, 6, 1), delist_reason=DelistReason.BANKRUPTCY, exchange="NYSE",
        notes="Filed Chapter 11 2009-06-01. Old GM liquidated; New GM emerged via sale.",
    ),
    SurvivorshipRecord(
        cik="0001005210", ticker="WCOME", company_name="WorldCom Inc",
        delist_date=date(2002, 7, 21), delist_reason=DelistReason.BANKRUPTCY, exchange="NASDAQ",
        notes="Filed Chapter 11 2002-07-21. $107B in assets — largest US bankruptcy at time.",
    ),
    SurvivorshipRecord(
        cik="0000023666", ticker="DELPQ", company_name="Delphi Corporation",
        delist_date=date(2005, 10, 8), delist_reason=DelistReason.BANKRUPTCY, exchange="NYSE",
        notes="Filed Chapter 11 2005-10-08. Largest US auto-parts maker bankruptcy.",
    ),
]

_registry: dict[str, SurvivorshipRecord] = {r.cik: r for r in _SEEDS}


def register(record: SurvivorshipRecord) -> None:
    _registry[record.cik] = record
    logger.info("Survivorship record registered", cik=record.cik, ticker=record.ticker)


def get_all_delisted() -> list[SurvivorshipRecord]:
    return list(_registry.values())


def is_delisted(cik: str) -> bool:
    return cik in _registry


def lookup_by_ticker(ticker: str) -> list[SurvivorshipRecord]:
    return [r for r in _registry.values() if r.ticker.upper() == ticker.upper()]


def get_delisted_in_range(start: date, end: date) -> list[SurvivorshipRecord]:
    return [r for r in _registry.values() if start <= r.delist_date <= end]


def build_point_in_time_universe(
    active_tickers: list[str],
    as_of_date: date,
    include_delisted: bool = True,
) -> list[str]:
    """Build a survivorship-bias-free universe for a historical backtest date.

    Adds delisted securities that were still alive (delist_date > as_of_date)
    to the active_tickers list. Without this, backtests systematically exclude
    companies that subsequently failed, inflating simulated returns.
    """
    universe = list(active_tickers)
    if not include_delisted:
        return universe

    added = 0
    for record in _registry.values():
        if record.delist_date > as_of_date and record.ticker not in universe:
            universe.append(record.ticker)
            added += 1

    if added:
        logger.info(
            "Survivorship-bias correction applied",
            as_of_date=as_of_date,
            added_delisted=added,
            total_universe=len(universe),
        )

    return universe
