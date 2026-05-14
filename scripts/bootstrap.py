"""
SENTINEL bootstrap — run once after `make docker-up` to initialize the system.
Creates DB tables, loads instrument master from OpenFIGI, loads EDGAR ticker→CIK map.
"""
from __future__ import annotations
import asyncio
from datetime import date
from sentinel.core.config import get_settings
from sentinel.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("bootstrap")


async def main() -> None:
    s = get_settings()
    logger.info("Starting SENTINEL bootstrap", db=s.database_url[:40])

    # 1. Warm up EDGAR ticker→CIK mapping
    logger.info("Step 1/4: Loading EDGAR company tickers...")
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
    tickers = await edgar.load_company_tickers()
    logger.info("EDGAR tickers loaded", count=len(tickers))

    # 2. Pre-populate instrument master for S&P 500 core tickers
    logger.info("Step 2/4: Resolving S&P 500 tickers via OpenFIGI → writing to instruments table...")
    SP500_SAMPLE = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM",
        "JNJ", "V", "PG", "UNH", "HD", "MA", "MRK", "ABBV", "CVX", "KO", "PEP",
        "SPY", "QQQ", "IWM", "GLD", "TLT", "HYG", "LQD", "EEM", "VNQ", "XLE",
    ]
    from sentinel.sim.openfigi_client import OpenFIGIClient
    from sentinel.sds.db import get_session_factory as _gsf
    figi_client = OpenFIGIClient(api_key=s.openfigi_api_key)
    _sf = _gsf()
    async with _sf() as _session:
        figi_map = await figi_client.bulk_resolve_tickers(SP500_SAMPLE, session=_session)
    resolved = sum(1 for v in figi_map.values() if v)
    logger.info("FIGIs resolved and persisted to instruments", resolved=resolved, total=len(SP500_SAMPLE))

    # 3. Run adapter health checks
    logger.info("Step 3/4: Running adapter health checks...")
    from sentinel.sds import build_default_adapters
    from sentinel.sds.normalizer import run_all_health_checks
    build_default_adapters()
    events = await run_all_health_checks()
    for ev in events:
        if ev.severity.value in ("error", "critical"):
            logger.warning("Adapter unhealthy", adapter=ev.adapter, event_type=ev.event_type)
        else:
            logger.info("Adapter healthy", adapter=ev.adapter)

    # 4. Fetch and persist initial FRED core series
    logger.info("Step 4/4: Pre-fetching FRED core 33 series and writing to macro_data...")
    from sentinel.sds.ingest import ingest_macro_series
    FRED_CORE = [
        "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "T10Y2Y", "VIXCLS",
        "DGS10", "DGS2", "DGS1", "DGS3MO", "BAMLH0A0HYM2", "M2SL",
        "SOFR", "T10YIE", "PAYEMS", "DEXUSEU", "DEXJPUS",
    ]
    _sf2 = _gsf()
    macro_ok = 0
    for series_id in FRED_CORE:
        try:
            async with _sf2() as _session:
                result = await ingest_macro_series(series_id=series_id, session=_session)
            if result.ok:
                macro_ok += 1
        except Exception as exc:
            logger.warning("FRED series failed", series=series_id, error=str(exc))
    logger.info("FRED core series persisted to macro_data", ok=macro_ok, total=len(FRED_CORE))

    # 5. Seed survivorship registry to PostgreSQL
    logger.info("Step 5/5: Seeding survivorship registry to DB...")
    from sentinel.sds.survivorship import get_all_delisted
    from sentinel.sds.db import get_session_factory
    from sentinel.sds import repository

    session_factory = get_session_factory()
    delisted = get_all_delisted()
    async with session_factory() as session:
        n = await repository.upsert_survivorship_records(delisted, session)
    logger.info("Survivorship registry seeded", records=len(delisted), inserted=n)

    # 6. Seed congressional trades (3-year history)
    logger.info("Step 6/7: Seeding congressional trades (STOCK Act disclosures, 3 years)...")
    from sentinel.sds.ingest import ingest_congressional_trades
    from datetime import timedelta
    since_congress = date.today().replace(year=date.today().year - 3)
    _sf3 = get_session_factory()
    try:
        async with _sf3() as _session:
            cong_result = await ingest_congressional_trades(session=_session, since=since_congress)
        if cong_result.ok:
            logger.info("Congressional trades seeded",
                        house=cong_result.house, senate=cong_result.senate)
        else:
            logger.warning("Congressional trades seed failed",
                           status=cong_result.status, error=cong_result.error)
    except Exception as exc:
        logger.warning("Congressional trades seed exception", error=str(exc))

    # 7. Seed COT data (2-year history)
    logger.info("Step 7/7: Seeding CFTC COT data (2 years)...")
    from sentinel.sds.ingest import ingest_cot_data
    _sf4 = get_session_factory()
    try:
        async with _sf4() as _session:
            cot_result = await ingest_cot_data(session=_session, start_year=date.today().year - 2)
        if cot_result.ok:
            logger.info("COT data seeded", markets=cot_result.markets, records=cot_result.records)
        else:
            logger.warning("COT seed failed", status=cot_result.status, error=cot_result.error)
    except Exception as exc:
        logger.warning("COT seed exception", error=str(exc))

    logger.info("Bootstrap complete! SENTINEL is ready.")
    logger.info("Start services: make terminal (UI) | make api (REST) | make mcp (AI)")


if __name__ == "__main__":
    asyncio.run(main())
