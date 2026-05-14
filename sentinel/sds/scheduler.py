"""SENTINEL ingestion scheduler — APScheduler async jobs.

Jobs:
  - eod_ohlcv:         Daily EOD equity ingest, Mon-Fri 4:30 PM ET
  - fred_macro:        Daily FRED macro update, 6:00 PM ET
  - corporate_actions: Weekly split/dividend sync, Sunday midnight ET

All jobs use misfire_grace_time so a missed fire (e.g. system sleep) runs
once on wake rather than being silently skipped.
"""
from __future__ import annotations
import asyncio
from datetime import date, datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None

# Representative S&P 500 sample — replace with a DB-driven ticker list
_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH",
    "LLY", "JPM", "V", "XOM", "MA", "AVGO", "HD", "PG", "CVX", "COST", "ABBV",
    "MRK", "KO", "PEP", "WMT", "BAC", "DIS", "CSCO", "ACN", "TMO", "MCD",
    "SPY", "QQQ", "IWM", "GLD", "TLT",   # key ETFs
]

_FRED_SERIES = [
    "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "T10Y2Y", "VIXCLS",
    "DGS10", "DGS2", "BAMLH0A0HYM2", "DEXUSEU", "DEXJPUS",
    "M2SL", "SOFR", "T10YIE", "PAYEMS",
]


async def _job_eod_ohlcv() -> None:
    """Fetch EOD bars, apply CA adjustments, validate, persist to TimescaleDB."""
    from sentinel.sds.normalizer import fetch_ohlcv_with_fallback
    from sentinel.sds.corporate_actions import fetch_and_apply, fetch_splits_yfinance
    from sentinel.sds.validator import validate_single_source
    from sentinel.sds.provenance import create_receipt
    from sentinel.sds.gap_detector import detect_gaps
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository

    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    end = datetime.combine(today, datetime.max.time())

    logger.info("EOD OHLCV ingest started", tickers=len(_EQUITY_UNIVERSE), date=today.isoformat())
    ingested = errors = 0

    session_factory = get_session_factory()

    for ticker in _EQUITY_UNIVERSE:
        try:
            bars = await fetch_ohlcv_with_fallback(ticker, start, end, "1d")
            if not bars:
                logger.warning("EOD: no bars returned", ticker=ticker)
                errors += 1
                continue

            bars = await fetch_and_apply(ticker, bars)
            bars, report = validate_single_source(bars, ticker)

            if not bars:
                logger.warning("EOD: all bars rejected by validator", ticker=ticker)
                errors += 1
                continue

            gap = detect_gaps(bars, ticker, today, today)
            if gap.is_silent_throttle:
                errors += 1
                continue

            async with session_factory() as session:
                await repository.write_ohlcv_bars(bars, ticker, "1d", session)
                receipt = create_receipt(
                    bars, source=bars[0].source, ticker=ticker, interval="1d"
                )
                await repository.write_provenance_receipt(receipt, session)

            ingested += 1

        except Exception as exc:
            logger.error("EOD ingest error", ticker=ticker, error=str(exc))
            errors += 1

    logger.info("EOD OHLCV ingest complete", ingested=ingested, errors=errors)


async def _job_fred_macro() -> None:
    """Fetch latest FRED macro series and persist to macro_data table."""
    from sentinel.sds import get_adapter
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository

    fred = get_adapter("fred")
    if fred is None:
        logger.warning("FRED adapter not registered — skipping macro update")
        return

    logger.info("FRED macro update started", series=len(_FRED_SERIES))
    updated = errors = 0
    session_factory = get_session_factory()

    for series_id in _FRED_SERIES:
        try:
            points = await fred.fetch_series(series_id)
            if points:
                async with session_factory() as session:
                    await repository.write_macro_points(points, session)
                updated += 1
        except Exception as exc:
            logger.error("FRED series error", series=series_id, error=str(exc))
            errors += 1

    logger.info("FRED macro update complete", updated=updated, errors=errors)


async def _job_congressional_trades() -> None:
    """Weekly sync of STOCK Act disclosures from House and Senate."""
    from sentinel.sds.ingest import ingest_congressional_trades
    from sentinel.sds.db import get_session_factory
    from datetime import date, timedelta

    since = date.today() - timedelta(days=14)  # two-week window catches any backlog
    session_factory = get_session_factory()
    logger.info("Congressional trades sync started")

    async with session_factory() as session:
        result = await ingest_congressional_trades(session=session, since=since)

    if result.ok:
        logger.info("Congressional trades sync complete",
                    house=result.house, senate=result.senate)
    else:
        logger.error("Congressional trades sync failed",
                     status=result.status, error=result.error)


async def _job_cot_data() -> None:
    """Weekly COT update — fetch latest CFTC release and update cot_index signals."""
    from sentinel.sds.ingest import ingest_cot_data
    from sentinel.sds.db import get_session_factory

    session_factory = get_session_factory()
    logger.info("COT data sync started")

    # Only load current year + last year for weekly refresh
    from datetime import date
    async with session_factory() as session:
        result = await ingest_cot_data(
            session=session,
            start_year=date.today().year - 1,
        )

    if result.ok:
        logger.info("COT sync complete", markets=result.markets, records=result.records)
    else:
        logger.error("COT sync failed", status=result.status, error=result.error)


async def _job_news_sentiment() -> None:
    """Daily news fetch + FinBERT sentiment for the equity universe."""
    from sentinel.sds.ingest import ingest_news
    from sentinel.sds.db import get_session_factory
    from datetime import date, timedelta

    today = date.today()
    from_date = today - timedelta(days=2)   # 2-day window catches weekend gaps
    session_factory = get_session_factory()
    logger.info("News sentiment sync started", tickers=len(_EQUITY_UNIVERSE))

    ok = errors = no_key = 0
    for ticker in _EQUITY_UNIVERSE:
        try:
            async with session_factory() as session:
                result = await ingest_news(
                    ticker=ticker, session=session,
                    from_date=from_date, to_date=today,
                )
            if result.ok:
                ok += 1
            elif result.status == "no_key":
                no_key += 1
                break   # No key = no point continuing
            else:
                errors += 1
        except Exception as exc:
            logger.error("News sync error", ticker=ticker, error=str(exc))
            errors += 1

    logger.info("News sentiment sync complete", ok=ok, errors=errors, no_key=no_key)

    # After news ingest, populate any missing embeddings
    if ok > 0:
        try:
            from sentinel.sil.news_embeddings import embed_pending_articles
            async with session_factory() as _session:
                n = await embed_pending_articles(_session, batch_size=128)
            if n > 0:
                logger.info("News embeddings populated", articles=n)
        except Exception as exc:
            logger.warning("News embedding step failed", error=str(exc))


async def _job_corporate_actions() -> None:
    """Weekly sync of split/dividend history and persist to corporate_actions table."""
    from sentinel.sds.corporate_actions import fetch_splits_yfinance, fetch_dividends_yfinance
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository

    logger.info("Corporate actions sync started", tickers=len(_EQUITY_UNIVERSE))
    synced = errors = 0
    loop = asyncio.get_event_loop()
    session_factory = get_session_factory()

    for ticker in _EQUITY_UNIVERSE:
        try:
            splits = await loop.run_in_executor(None, lambda t=ticker: fetch_splits_yfinance(t))
            raw_divs = await loop.run_in_executor(
                None, lambda t=ticker: fetch_dividends_yfinance(t)
            )
            # Resolve dividend factors using prior-close prices (replaces 1.0 placeholder)
            from sentinel.sds.corporate_actions import resolve_dividend_factors
            divs = await resolve_dividend_factors(raw_divs, ticker)

            all_actions = splits + divs
            if all_actions:
                async with session_factory() as session:
                    n = await repository.upsert_corporate_actions(all_actions, session)
                logger.info("CA persisted", ticker=ticker, splits=len(splits),
                            dividends=len(divs), rows=n)
            synced += 1
        except Exception as exc:
            logger.error("CA sync error", ticker=ticker, error=str(exc))
            errors += 1

    logger.info("Corporate actions sync complete", synced=synced, errors=errors)


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="America/New_York")

        _scheduler.add_job(
            _job_eod_ohlcv,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York"),
            id="eod_ohlcv",
            replace_existing=True,
            misfire_grace_time=300,   # 5 min — tolerate brief system sleep
        )
        _scheduler.add_job(
            _job_fred_macro,
            CronTrigger(hour=18, minute=0, timezone="America/New_York"),
            id="fred_macro",
            replace_existing=True,
            misfire_grace_time=600,
        )
        _scheduler.add_job(
            _job_corporate_actions,
            CronTrigger(day_of_week="sun", hour=0, minute=0, timezone="America/New_York"),
            id="corporate_actions",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        _scheduler.add_job(
            _job_congressional_trades,
            CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="America/New_York"),
            id="congressional_trades",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        _scheduler.add_job(
            _job_cot_data,
            # CFTC releases COT every Friday after 3:30 PM ET; process Saturday morning
            CronTrigger(day_of_week="sat", hour=6, minute=0, timezone="America/New_York"),
            id="cot_data",
            replace_existing=True,
            misfire_grace_time=7200,
        )
        _scheduler.add_job(
            _job_news_sentiment,
            # After EOD ingest; includes any after-hours press releases
            CronTrigger(hour=19, minute=0, timezone="America/New_York"),
            id="news_sentiment",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        logger.info("Scheduler configured", jobs=6)

    return _scheduler


async def start_scheduler() -> None:
    sched = get_scheduler()
    sched.start()
    logger.info("Scheduler started", jobs=[j.id for j in sched.get_jobs()])


async def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
