"""SENTINEL backfill — populate the data lake from scratch or resume incrementally.

Every call goes through sentinel.sds.ingest which runs the full pipeline:
    fetch → validate → CA adjust → gap detect → persist → provenance receipt

Nothing is discarded. Every bar that passes validation ends up in TimescaleDB.

Usage:
    python scripts/backfill.py --mode all --years 10          # full historical load
    python scripts/backfill.py --mode ohlcv --years 5         # price history only
    python scripts/backfill.py --mode fred                    # macro series
    python scripts/backfill.py --mode edgar                   # XBRL fundamentals (full SP500)
    python scripts/backfill.py --mode edgar --ticker AAPL     # single company
    python scripts/backfill.py --mode cot --years 5           # CFTC COT reports
    python scripts/backfill.py --mode ca                      # corporate actions
    python scripts/backfill.py --status                       # show what's in the DB
    python scripts/backfill.py --mode ohlcv --resume          # skip tickers already ingested
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import date, datetime, timedelta

from sentinel.core.config import get_settings
from sentinel.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("backfill")

# ── Universe definitions ───────────────────────────────────────────────────────

SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "MRK", "ABBV", "CVX", "KO", "PEP", "AVGO", "LLY",
    "WMT", "BAC", "CSCO", "ACN", "MCD", "ORCL", "TMO", "COST", "NEE", "DHR", "TXN",
    "WFC", "PM", "CRM", "NFLX", "ADBE", "AMGN", "CAT", "HON", "QCOM", "IBM", "VZ",
    "INTC", "GS", "MS", "RTX", "AXP", "SBUX",
    # Key ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT", "HYG", "LQD", "EEM", "VNQ", "XLE",
]

FRED_SERIES = [
    "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "T10Y2Y", "VIXCLS",
    "DGS10", "DGS2", "DGS1", "DGS3MO", "DGS6MO", "DGS5", "DGS20", "DGS30",
    "BAMLH0A0HYM2", "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS",
    "M2SL", "SOFR", "T10YIE", "PAYEMS", "UMCSENT", "HOUST", "INDPRO",
    "RETAILSMNSA", "PCE", "PCEPI", "DCOILWTICO", "BAMLC0A0CM",
]


# ── Status ────────────────────────────────────────────────────────────────────

async def show_status() -> None:
    """Print a data lake catalog report showing exactly what's been ingested."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.catalog import get_coverage, format_catalog_report

    session_factory = get_session_factory()
    async with session_factory() as session:
        summary = await get_coverage(session)
    print(format_catalog_report(summary))


# ── OHLCV backfill ────────────────────────────────────────────────────────────

async def backfill_ohlcv(
    years: int = 10,
    resume: bool = False,
    tickers: list[str] | None = None,
) -> None:
    """Ingest daily OHLCV bars for the full universe into TimescaleDB."""
    from sentinel.sds import build_default_adapters
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_ticker_ohlcv

    build_default_adapters()
    session_factory = get_session_factory()

    end = datetime.combine(date.today(), datetime.max.time())
    start = datetime.combine(date.today() - timedelta(days=365 * years), datetime.min.time())

    universe = tickers or SP500_TICKERS
    ok = errors = skipped = 0

    logger.info("OHLCV backfill starting",
                tickers=len(universe), years=years, resume=resume,
                start=start.date().isoformat(), end=end.date().isoformat())

    for i, ticker in enumerate(universe, 1):
        try:
            async with session_factory() as session:
                result = await ingest_ticker_ohlcv(
                    ticker=ticker,
                    interval="1d",
                    start=start,
                    end=end,
                    session=session,
                    skip_if_exists=resume,
                )

            if result.status == "ok":
                logger.info("OHLCV ingested",
                            ticker=ticker, bars=result.bars,
                            gaps=result.gaps, source=result.source,
                            progress=f"{i}/{len(universe)}")
                ok += 1
            elif result.status == "skipped_existing":
                skipped += 1
            elif result.status == "no_data":
                logger.warning("No data", ticker=ticker, progress=f"{i}/{len(universe)}")
                errors += 1
            elif result.status == "silent_throttle":
                logger.error("Silent throttle detected — rate limited",
                             ticker=ticker, progress=f"{i}/{len(universe)}")
                errors += 1
                await asyncio.sleep(5)
            elif result.status == "all_rejected":
                logger.warning("All bars rejected by validator",
                               ticker=ticker, rejected=result.rejected,
                               progress=f"{i}/{len(universe)}")
                errors += 1
            else:
                logger.error("Ingest error", ticker=ticker, error=result.error,
                             progress=f"{i}/{len(universe)}")
                errors += 1

        except Exception as exc:
            logger.error("OHLCV backfill exception", ticker=ticker, error=str(exc))
            errors += 1

        # Courtesy delay — yfinance is generous but not infinite
        await asyncio.sleep(0.3)

    logger.info("OHLCV backfill complete",
                ok=ok, errors=errors, skipped=skipped, total=len(universe))


# ── Macro backfill ─────────────────────────────────────────────────────────────

async def backfill_fred(years: int = 30) -> None:
    """Fetch full FRED history and persist to macro_data table."""
    from sentinel.sds import build_default_adapters
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_macro_series

    build_default_adapters()
    session_factory = get_session_factory()

    start = date.today() - timedelta(days=365 * years)
    ok = errors = 0

    logger.info("FRED backfill starting", series=len(FRED_SERIES), years=years)

    for series_id in FRED_SERIES:
        try:
            async with session_factory() as session:
                result = await ingest_macro_series(
                    series_id=series_id,
                    session=session,
                    start=start,
                )
            if result.ok:
                logger.info("FRED ingested", series=series_id, points=result.points)
                ok += 1
            elif result.status == "no_data":
                logger.warning("No data", series=series_id)
                errors += 1
            else:
                logger.error("FRED error", series=series_id, error=result.error)
                errors += 1
        except Exception as exc:
            logger.error("FRED exception", series=series_id, error=str(exc))
            errors += 1

        await asyncio.sleep(0.1)

    logger.info("FRED backfill complete", ok=ok, errors=errors, total=len(FRED_SERIES))


# ── EDGAR fundamentals backfill ───────────────────────────────────────────────

async def backfill_edgar(
    cik: str | None = None,
    ticker: str | None = None,
) -> None:
    """Fetch EDGAR XBRL companyfacts and persist to financial_facts table."""
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_edgar_facts

    s = get_settings()
    edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
    await edgar.load_company_tickers()
    session_factory = get_session_factory()

    # Build CIK list
    cik_ticker_pairs: list[tuple[str, str]] = []
    if cik:
        cik_ticker_pairs = [(cik.zfill(10), ticker or cik)]
    elif ticker:
        c = edgar.ticker_to_cik(ticker)
        if c:
            cik_ticker_pairs = [(c, ticker)]
        else:
            logger.error("CIK not found for ticker", ticker=ticker)
            return
    else:
        for t in SP500_TICKERS:
            c = edgar.ticker_to_cik(t)
            if c:
                cik_ticker_pairs.append((c, t))

    ok = errors = no_data = 0
    logger.info("EDGAR backfill starting", companies=len(cik_ticker_pairs))

    for c, t in cik_ticker_pairs:
        try:
            async with session_factory() as session:
                result = await ingest_edgar_facts(cik=c, ticker=t, session=session)

            if result.ok:
                logger.info("EDGAR ingested", cik=c, ticker=t, facts=result.facts)
                ok += 1
            elif result.status == "no_data":
                no_data += 1
            elif result.status == "no_facts":
                logger.warning("No extractable facts", cik=c, ticker=t)
                no_data += 1
            else:
                logger.error("EDGAR error", cik=c, ticker=t, error=result.error)
                errors += 1
        except Exception as exc:
            logger.error("EDGAR exception", cik=c, ticker=t, error=str(exc))
            errors += 1

        await asyncio.sleep(0.15)  # SEC rate limit: 10 req/sec

    logger.info("EDGAR backfill complete",
                ok=ok, no_data=no_data, errors=errors, total=len(cik_ticker_pairs))


# ── Corporate actions backfill ────────────────────────────────────────────────

async def backfill_ca(tickers: list[str] | None = None) -> None:
    """Fetch splits + dividends (with resolved factors) and persist to corporate_actions."""
    from sentinel.sds import build_default_adapters
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_corporate_actions

    build_default_adapters()
    session_factory = get_session_factory()

    universe = tickers or SP500_TICKERS
    ok = errors = 0

    logger.info("Corporate actions backfill starting", tickers=len(universe))

    for ticker in universe:
        try:
            async with session_factory() as session:
                n = await ingest_corporate_actions(ticker=ticker, session=session)
            if n > 0:
                logger.info("CA ingested", ticker=ticker, rows=n)
            ok += 1
        except Exception as exc:
            logger.error("CA exception", ticker=ticker, error=str(exc))
            errors += 1

        await asyncio.sleep(0.3)

    logger.info("CA backfill complete", ok=ok, errors=errors, total=len(universe))


# ── COT backfill ──────────────────────────────────────────────────────────────

async def backfill_congress(since_years: int = 3) -> None:
    """Fetch STOCK Act disclosures from House and Senate, persist to congressional_trades."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_congressional_trades

    since = date.today().replace(year=date.today().year - since_years)
    session_factory = get_session_factory()

    logger.info("Congressional trades backfill starting", since=since.isoformat())
    async with session_factory() as session:
        result = await ingest_congressional_trades(session=session, since=since)

    if result.ok:
        logger.info("Congressional trades ingested",
                    house=result.house, senate=result.senate,
                    total=result.house + result.senate)
    else:
        logger.error("Congressional trades failed", status=result.status, error=result.error)


async def backfill_cot(years: int = 5) -> None:
    """Download CFTC COT data, compute COT Index, and persist to cot_data table."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_cot_data

    start_year = date.today().year - years
    session_factory = get_session_factory()

    logger.info("COT backfill starting", start_year=start_year, years=years)
    async with session_factory() as session:
        result = await ingest_cot_data(session=session, start_year=start_year)

    if result.ok:
        logger.info("COT data persisted",
                    markets=result.markets, records=result.records)
    else:
        logger.error("COT backfill failed", status=result.status, error=result.error)


# ── Insider transactions backfill ─────────────────────────────────────────────

# High-profile insiders: Berkshire Hathaway (CIK 0001067983),
# Apple (CIK 0000320193 — insiders file under company CIK),
# and a selection of large-cap executives.
# In practice, populate from your own watchlist.
INSIDER_CIK_TICKERS = [
    ("0000320193", "AAPL"),
    ("0000789019", "MSFT"),
    ("0001018724", "AMZN"),
    ("0001045810", "NVDA"),
    ("0001652044", "GOOGL"),
    ("0001326801", "META"),
    ("0001318605", "TSLA"),
    ("0000051143", "IBM"),
    ("0000019617", "JPM"),
    ("0000070858", "BA"),
]

# Top institutional managers by AUM (Berkshire, Vanguard, BlackRock, etc.)
INSTITUTIONAL_MANAGER_CIKS = [
    "0001067983",  # Berkshire Hathaway
    "0000102909",  # Vanguard
    "0001364742",  # BlackRock
    "0000831641",  # State Street
    "0001045810",  # NVIDIA (tracks own buybacks via 13F)
    "0001166559",  # Bridgewater Associates
    "0001336528",  # Renaissance Technologies
]


async def backfill_insider(
    since_years: int = 2,
    tickers: list[str] | None = None,
) -> None:
    """Fetch Form 4 insider transactions for key CIKs and persist to insider_transactions."""
    from sentinel.sds.adapters.edgar_adapter import EDGARAdapter
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_insider_transactions

    s = get_settings()
    edgar = EDGARAdapter(user_agent=s.edgar_user_agent)
    await edgar.load_company_tickers()
    session_factory = get_session_factory()

    since = date.today().replace(year=date.today().year - since_years)
    pairs = INSIDER_CIK_TICKERS

    # If specific tickers requested, look up their CIKs
    if tickers:
        pairs = []
        for t in tickers:
            c = edgar.ticker_to_cik(t)
            if c:
                pairs.append((c, t))
            else:
                logger.warning("CIK not found for insider ticker", ticker=t)

    ok = errors = no_data = 0
    logger.info("Insider transactions backfill starting",
                pairs=len(pairs), since=since.isoformat())

    for cik, ticker in pairs:
        try:
            async with session_factory() as session:
                result = await ingest_insider_transactions(
                    cik=cik, ticker=ticker, session=session, since=since, limit=100
                )
            if result.ok:
                logger.info("Insider transactions ingested",
                            cik=cik, ticker=ticker, count=result.transactions)
                ok += 1
            elif result.status == "no_data":
                logger.warning("No Form 4 data", cik=cik, ticker=ticker)
                no_data += 1
            else:
                logger.error("Insider ingest error", cik=cik, ticker=ticker, error=result.error)
                errors += 1
        except Exception as exc:
            logger.error("Insider backfill exception", cik=cik, ticker=ticker, error=str(exc))
            errors += 1

        await asyncio.sleep(0.5)  # EDGAR rate limit

    logger.info("Insider backfill complete", ok=ok, no_data=no_data, errors=errors)


async def backfill_institutional(since_years: int = 2) -> None:
    """Fetch 13F-HR institutional holdings for major managers, persist to institutional_holdings."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_institutional_holdings

    session_factory = get_session_factory()
    since = date.today().replace(year=date.today().year - since_years)
    ok = errors = no_data = 0

    logger.info("Institutional holdings backfill starting",
                managers=len(INSTITUTIONAL_MANAGER_CIKS), since=since.isoformat())

    for manager_cik in INSTITUTIONAL_MANAGER_CIKS:
        try:
            async with session_factory() as session:
                result = await ingest_institutional_holdings(
                    manager_cik=manager_cik, session=session,
                    since=since, limit=8,
                )
            if result.ok:
                logger.info("13F ingested", cik=manager_cik, holdings=result.holdings)
                ok += 1
            elif result.status == "no_data":
                logger.warning("No 13F filings", cik=manager_cik)
                no_data += 1
            else:
                logger.error("13F error", cik=manager_cik, error=result.error)
                errors += 1
        except Exception as exc:
            logger.error("13F exception", cik=manager_cik, error=str(exc))
            errors += 1

        await asyncio.sleep(1.0)  # EDGAR rate limit — 13F XMLs are large

    logger.info("Institutional holdings backfill complete",
                ok=ok, no_data=no_data, errors=errors)


async def backfill_rag_documents(max_filings: int = 500) -> None:
    """Ingest EDGAR 10-K/10-Q XBRL facts into the RAG document_chunks table.

    Groups financial_facts rows by (ticker, accession, form, filed) and renders
    each filing as structured text before chunking + embedding.
    """
    from sentinel.sds.db import get_session_factory
    from sentinel.sil.rag import ingest_document
    from sentinel.core.config import get_settings
    from sqlalchemy import text

    settings = get_settings()
    db_url = settings.database_url
    session_factory = get_session_factory()

    logger.info("RAG EDGAR ingest starting", max_filings=max_filings)

    # Pull distinct filings with at least 5 facts each
    async with session_factory() as session:
        result = await session.execute(text("""
            SELECT
                COALESCE(i.ticker, ff.figi) AS ticker,
                ff.accession,
                ff.form,
                ff.filed,
                COUNT(*) AS fact_count
            FROM financial_facts ff
            LEFT JOIN instruments i ON ff.figi = i.figi
            WHERE ff.form IN ('10-K', '10-Q')
              AND ff.accession IS NOT NULL
            GROUP BY COALESCE(i.ticker, ff.figi), ff.accession, ff.form, ff.filed
            HAVING COUNT(*) >= 5
            ORDER BY ff.filed DESC NULLS LAST
            LIMIT :limit
        """), {"limit": max_filings})
        filings = [dict(r) for r in result.mappings()]

    if not filings:
        logger.warning("No EDGAR filings in financial_facts — run edgar backfill first")
        return

    logger.info("Filings to ingest into RAG", count=len(filings))
    ingested = skipped = errors = 0

    for filing in filings:
        ticker = filing["ticker"]
        accession = filing["accession"]
        form = filing["form"]
        filed = filing["filed"]
        doc_id = f"edgar:{accession}"

        try:
            # Pull all facts for this accession
            async with session_factory() as session:
                result = await session.execute(text("""
                    SELECT concept, label, value, unit, period_start, period_end
                    FROM financial_facts
                    WHERE accession = :accession
                    ORDER BY period_end DESC, concept
                """), {"accession": accession})
                facts = [dict(r) for r in result.mappings()]

            if not facts:
                skipped += 1
                continue

            # Render facts as structured text
            lines: list[str] = [
                f"EDGAR Filing: {form} | Ticker: {ticker} | Filed: {filed} | Accession: {accession}",
                "",
            ]
            for fact in facts:
                label = fact.get("label") or fact.get("concept", "")
                value = fact.get("value")
                unit = fact.get("unit", "")
                period_end = fact.get("period_end", "")
                period_start = fact.get("period_start", "")
                if value is not None:
                    val_str = f"{float(value):,.2f}" if unit in ("USD",) else str(value)
                    period_str = f"{period_start} to {period_end}" if period_start else str(period_end)
                    lines.append(f"{label}: {val_str} {unit} (period: {period_str})")

            doc_text = "\n".join(lines)
            chunks_written = await ingest_document(
                db_url=db_url,
                text=doc_text,
                doc_id=doc_id,
                ticker=ticker if ticker and not ticker.startswith("BBG") else None,
                doc_type=form,
                filed_date=filed,
                metadata={"accession": accession, "form": form},
            )
            ingested += 1
            logger.info("RAG ingest ok", ticker=ticker, form=form, chunks=chunks_written)

        except Exception as exc:
            logger.error("RAG ingest error", accession=accession, error=str(exc))
            errors += 1

        await asyncio.sleep(0.05)   # yield event loop between filings

    logger.info("RAG EDGAR ingest complete",
                ingested=ingested, skipped=skipped, errors=errors)


async def backfill_embeddings() -> None:
    """Populate sentence-transformer embeddings for all news_articles rows missing them."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sil.news_embeddings import embed_pending_articles, build_vector_index

    session_factory = get_session_factory()
    logger.info("Embeddings backfill starting")

    async with session_factory() as session:
        n = await embed_pending_articles(session, batch_size=128)
    logger.info("Embeddings written", articles=n)

    if n > 0:
        async with session_factory() as session:
            await build_vector_index(session)
        logger.info("IVFFlat vector index built/verified")


DEFAULT_CRYPTO_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "DOGE/USDT",
]


async def backfill_crypto(
    pairs: list[str] | None = None,
    interval: str = "1d",
    years: int = 3,
    exchange: str = "binance",
) -> None:
    """Backfill crypto OHLCV for major pairs via CCXT."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_crypto_ohlcv

    session_factory = get_session_factory()
    pairs = pairs or DEFAULT_CRYPTO_PAIRS
    start = datetime.combine(date.today().replace(year=date.today().year - years), datetime.min.time())
    end_dt = datetime.utcnow()

    ok = errors = no_adapter = 0
    logger.info("Crypto backfill starting", pairs=len(pairs), years=years)

    for pair in pairs:
        try:
            async with session_factory() as session:
                result = await ingest_crypto_ohlcv(
                    pair=pair, interval=interval,
                    start=start, end=end_dt,
                    exchange=exchange, session=session,
                )
            if result.ok:
                logger.info("Crypto ingested", pair=pair, bars=result.bars)
                ok += 1
            elif result.status == "no_adapter":
                logger.warning("CCXT adapter not configured — set exchange credentials in .env")
                no_adapter += 1
                break
            else:
                logger.warning("Crypto no data", pair=pair, status=result.status)
                errors += 1
        except Exception as exc:
            logger.error("Crypto backfill error", pair=pair, error=str(exc))
            errors += 1
        await asyncio.sleep(1.0)

    logger.info("Crypto backfill complete", ok=ok, errors=errors, no_adapter=no_adapter)


DEFAULT_CONTINUOUS_FUTURES = {
    # yfinance "=F" tickers → CONT: labelling in our DB
    "ES=F": "CONT:ES1",   # E-mini S&P 500 (CME)
    "NQ=F": "CONT:NQ1",   # E-mini Nasdaq-100 (CME)
    "YM=F": "CONT:YM1",   # E-mini Dow Jones (CBOT)
    "RTY=F": "CONT:RTY1", # E-mini Russell 2000 (CME)
    "CL=F": "CONT:CL1",   # Crude Oil WTI (NYMEX)
    "GC=F": "CONT:GC1",   # Gold (COMEX)
    "SI=F": "CONT:SI1",   # Silver (COMEX)
    "NG=F": "CONT:NG1",   # Natural Gas (NYMEX)
    "HG=F": "CONT:HG1",   # Copper (COMEX)
    "ZB=F": "CONT:ZB1",   # 30-Year US Treasury Bond (CBOT)
    "ZN=F": "CONT:ZN1",   # 10-Year US Treasury Note (CBOT)
    "ZF=F": "CONT:ZF1",   # 5-Year US Treasury Note (CBOT)
}


async def backfill_continuous_futures(
    years: int = 10,
    futures: dict[str, str] | None = None,
) -> None:
    """Ingest back-adjusted continuous futures series from yfinance.

    yfinance's '=F' tickers (e.g. 'ES=F') expose CME-style back-adjusted
    continuous series.  We re-label them with 'CONT:' FIGIs for downstream
    strategy use so they can be joined with other OHLCV data.
    """
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_ticker_ohlcv
    from sentinel.sds.repository import upsert_instrument

    session_factory = get_session_factory()
    universe = futures or DEFAULT_CONTINUOUS_FUTURES
    start_dt = datetime.combine(
        date.today().replace(year=date.today().year - years),
        datetime.min.time(),
    )
    end_dt = datetime.utcnow()

    ok = errors = 0
    logger.info("Continuous futures backfill starting",
                contracts=len(universe), years=years)

    for yf_ticker, cont_label in universe.items():
        try:
            # Register synthetic instrument with CONT: FIGI
            async with session_factory() as session:
                await upsert_instrument(
                    figi=cont_label,
                    ticker=yf_ticker,
                    session=session,
                    name=f"Continuous {cont_label}",
                    exchange="CME",
                    asset_class="FUTURE",
                    currency="USD",
                )

            async with session_factory() as session:
                result = await ingest_ticker_ohlcv(
                    ticker=yf_ticker,
                    interval="1d",
                    start=start_dt,
                    end=end_dt,
                    session=session,
                )

            if result.ok:
                logger.info("Continuous futures ingested",
                            ticker=yf_ticker, label=cont_label, bars=result.bars)
                ok += 1
            else:
                logger.warning("Continuous futures no data",
                               ticker=yf_ticker, status=result.status)
                errors += 1
        except Exception as exc:
            logger.error("Continuous futures error", ticker=yf_ticker, error=str(exc))
            errors += 1
        await asyncio.sleep(0.5)

    logger.info("Continuous futures backfill complete", ok=ok, errors=errors)


async def backfill_news(
    tickers: list[str] | None = None,
    days: int = 30,
) -> None:
    """Fetch news + sentiment for the equity universe and persist to news_articles."""
    from sentinel.sds.db import get_session_factory
    from sentinel.sds.ingest import ingest_news

    session_factory = get_session_factory()
    universe = tickers or SP500_TICKERS
    from_date = date.today() - timedelta(days=days)

    ok = errors = no_key = no_data = 0
    logger.info("News backfill starting", tickers=len(universe), days=days)

    for ticker in universe:
        try:
            async with session_factory() as session:
                result = await ingest_news(
                    ticker=ticker, session=session,
                    from_date=from_date, to_date=date.today(),
                )
            if result.ok:
                logger.info("News ingested", ticker=ticker, articles=result.articles)
                ok += 1
            elif result.status == "no_key":
                logger.warning("No Finnhub API key — set FINNHUB_API_KEY in .env")
                no_key += 1
                break  # No key = no point continuing
            elif result.status == "no_data":
                no_data += 1
            else:
                logger.error("News error", ticker=ticker, error=result.error)
                errors += 1
        except Exception as exc:
            logger.error("News exception", ticker=ticker, error=str(exc))
            errors += 1

        await asyncio.sleep(0.5)

    logger.info("News backfill complete",
                ok=ok, no_data=no_data, errors=errors, no_key=no_key)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL data lake backfill")
    parser.add_argument(
        "--mode",
        choices=["ohlcv", "fred", "edgar", "ca", "cot", "congress",
                 "insider", "institutional", "news", "embeddings", "rag",
                 "crypto", "continuous", "all"],
        default="all",
    )
    parser.add_argument("--years", type=int, default=10,
                        help="Years of history to backfill (default: 10)")
    parser.add_argument("--cik", type=str, default=None,
                        help="Single CIK for EDGAR mode")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker (EDGAR or OHLCV mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tickers that already have provenance records")
    parser.add_argument("--status", action="store_true",
                        help="Show data lake catalog report and exit")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history for news backfill (default: 30)")
    parser.add_argument("--manager-cik", type=str, default=None,
                        help="Single manager CIK for institutional mode")
    args = parser.parse_args()

    if args.status:
        await show_status()
        return

    logger.info("Backfill started", mode=args.mode, years=args.years, resume=args.resume)

    tickers = [args.ticker] if args.ticker and args.mode == "ohlcv" else None
    news_tickers = [args.ticker] if args.ticker and args.mode == "news" else None

    if args.mode in ("ohlcv", "all"):
        await backfill_ohlcv(years=args.years, resume=args.resume, tickers=tickers)

    if args.mode in ("fred", "all"):
        await backfill_fred(years=max(args.years, 30))

    if args.mode in ("ca", "all"):
        await backfill_ca()

    if args.mode in ("cot", "all"):
        await backfill_cot(years=min(args.years, 5))

    if args.mode in ("congress", "all"):
        await backfill_congress(since_years=min(args.years, 5))

    if args.mode in ("edgar", "all"):
        await backfill_edgar(cik=args.cik, ticker=args.ticker)

    if args.mode in ("insider",):
        insider_tickers = [args.ticker] if args.ticker else None
        await backfill_insider(since_years=min(args.years, 3), tickers=insider_tickers)

    if args.mode in ("institutional",):
        if args.manager_cik:
            from sentinel.sds.db import get_session_factory
            from sentinel.sds.ingest import ingest_institutional_holdings
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await ingest_institutional_holdings(
                    manager_cik=args.manager_cik, session=session
                )
            logger.info("Single manager 13F done", status=result.status, holdings=result.holdings)
        else:
            await backfill_institutional(since_years=min(args.years, 2))

    if args.mode in ("news",):
        await backfill_news(tickers=news_tickers, days=args.days)

    if args.mode in ("embeddings",):
        await backfill_embeddings()

    if args.mode in ("rag",):
        await backfill_rag_documents()

    if args.mode in ("crypto",):
        await backfill_crypto(years=min(args.years, 3))

    if args.mode in ("continuous",):
        await backfill_continuous_futures(years=min(args.years, 20))

    # Final status after any run
    logger.info("Backfill complete — running catalog report")
    await show_status()


if __name__ == "__main__":
    asyncio.run(main())
